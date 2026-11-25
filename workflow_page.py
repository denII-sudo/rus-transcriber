"""Локальный экран П5: запуск, состояние и результат нового workflow без терминала.

Отдаёт workflow.html на секретном адресе (только 127.0.0.1) и принимает запросы к core.py:
запуск обработки, состояние и готовность, чтение стенограммы и конспектов, добровольные
уточнения, история ревизий с просмотром старой версии и возвратом как текущей, общий список
конфликтов файлов, изменённых вне программы (скачивание копии человека, сравнение и явная
замена файла версией программы после показа; правка в стенограмму не переносится), резерв
состояния БД. Сам выключается после --hours часов без запросов.

    python workflow_page.py [--port 8766] [--hours 6]

Реальные и старые данные не подключаются до П5 и отдельно согласованной миграции.
"""
import argparse
import json
import mimetypes
import secrets
import socket
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402,F401
import core  # noqa: E402
import llm_provider  # noqa: E402
import storage  # noqa: E402

SKELETON = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px)}'
            'body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}'
            '[hidden]{display:none!important}</style></head><body>{BODY}</body></html>')
mimetypes.add_type("audio/mpeg", ".mp3")
MAX_BODY_BYTES = 64 * 1024
MAX_DRAIN_BYTES = 1024 * 1024   # отказанное тело до этого размера дочитывается и отбрасывается перед ответом
DRAIN_TIMEOUT = 2               # секунд на всё такое дочитывание вместе (общий срок, а не на одно чтение)
# Тот же приём — в review_page.py (там свои копии констант и _refuse): правя один, смотри и другой.


class ExclusiveHTTPServer(ThreadingHTTPServer):
    """Сервер, который не делит порт с другим процессом.

    HTTPServer включает SO_REUSEADDR, а на Windows с ним второй экран молча занимает порт,
    который уже слушает прежний: новый адрес выдаётся, но запросы уходят старому.
    Без него и с SO_EXCLUSIVEADDRUSE занятый порт даёт OSError, и берётся следующий.
    """
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def pick_audio_path():
    """Нативное окно выбора файла Windows: диалог в отдельном процессе, экран продолжает отвечать.

    Веб-страница не может получить полный путь из <input type="file">, поэтому диалог
    открывает сам локальный сервер. Путь выбирает человек; сервер путь не угадывает.
    """
    import subprocess
    code = ("import tkinter as tk; from tkinter import filedialog; r = tk.Tk(); r.withdraw(); "
            "r.attributes('-topmost', True); "
            "print(filedialog.askopenfilename(title='Выберите запись', "
            "filetypes=[('Аудиозаписи', '*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma'), ('Все файлы', '*.*')])); "
            "r.destroy()")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, encoding="utf-8", timeout=300, creationflags=0x08000000).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def segment_text(segment):
    words = segment.get("words") or []
    if isinstance(segment.get("text"), str) and segment["text"]:
        return segment["text"]
    return " ".join(str(word.get("w", "")) for word in words if isinstance(word, dict))


def result_view(data):
    """Сокращённый вид SQLite-записи для экрана; полные данные остаются в БД и файлах."""
    segments = []
    for segment in data.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        segments.append({"start": segment.get("start"), "end": segment.get("end"),
                         "speaker": segment.get("speaker") or segment.get("label"),
                         "text": segment_text(segment)})
    doubts = [{"id": item.get("id"), "text": item.get("text"), "t1": item.get("t1"),
               "t2": item.get("t2"), "status": item.get("status", "open")}
              for item in data.get("review") or [] if isinstance(item, dict)]
    return {"recording_id": data.get("recording_id"), "title": data.get("title"),
            "duration": data.get("duration"), "speakers": data.get("speakers"),
            "speaker_names": data.get("speaker_names"),
            "names_confirmed": data.get("names_confirmed"),
            "source_revision": data.get("source_revision"),
            "review_revision": data.get("review_revision"),
            "document_sync": data.get("document_sync"),
            "files": {"json": data.get("json"), "markdown": data.get("markdown"),
                      "manifest": data.get("manifest")},
            "segments": segments, "doubts": doubts}


def state_ru(state):
    return {"queued": "в очереди", "starting": "запуск", "running": "выполняется",
            "processed": "обработано", "waiting_model": "ожидает конспект от модели",
            "completed": "завершено", "failed": "ошибка модельного шага",
            "interrupted": "прервано"}.get(state, state)


def make_server(port):
    """Собрать локальный сервер экрана; возвращает (server, url, last)."""
    last = [time.time()]
    token = secrets.token_urlsafe(32)
    prefix = f"/{token}/"

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # URL содержит секрет сессии: не записывать запрос целиком в журнал.
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8", headers=(), close=False):
            b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(b)

        def _length(self):
            """Объявленная длина тела POST или None, если заголовок неверен или тело идёт частями."""
            length = self.headers.get("Content-Length", "")
            if len(length) > 12 or not length.isascii() or not length.isdecimal() or self.headers.get("Transfer-Encoding"):
                return None
            return int(length)

        def _refuse(self, code, body):
            """Ранний отказ на POST, пока тело ещё не прочитано. Закрытие соединения с непрочитанным телом в Windows
            даёт клиенту сброс (WinError 10053) вместо ответа: браузер видит обрыв, а не отказ. Поэтому объявленное
            тело до MAX_DRAIN_BYTES дочитывается и отбрасывается, всё вместе не дольше DRAIN_TIMEOUT от начала (общий
            срок: перед каждым куском сокету ставится остаток), потом идёт ответ; больше, без длины или не пришедшее
            в срок — отвечаем сразу, соединение закрывается. Тот же приём — в review_page.py."""
            n = self._length()
            if n and n <= MAX_DRAIN_BYTES:
                deadline = time.monotonic() + DRAIN_TIMEOUT
                left = n
                try:
                    while left > 0:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self.connection.settimeout(remaining)
                        chunk = self.rfile.read1(min(65536, left))
                        if not chunk:
                            break
                        left -= len(chunk)
                except OSError:
                    pass
                self.connection.settimeout(DRAIN_TIMEOUT)     # запись ответа — со своим запасом, не с остатком срока
            self._send(code, body, close=True)

        def _path(self, post=False):
            refuse = self._refuse if post else self._send
            host = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != host:
                refuse(403, {"error": "недопустимый адрес"})
                return None
            if post and self.headers.get("Origin") != f"http://{host}":
                refuse(403, {"error": "недопустимый источник"})
                return None
            u = urlparse(self.path)
            if not u.path.startswith(prefix):
                refuse(403, {"error": "недействительная сессия"})
                return None
            return u.path[len(prefix):]

        def do_GET(self):
            u = self._path()
            if u is None:
                return
            last[0] = time.time()
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                if u in ("", "index.html"):
                    html = (HERE / "workflow.html").read_text(encoding="utf-8")
                    return self._send(200, SKELETON.replace("{BODY}", html).encode("utf-8"),
                                      "text/html; charset=utf-8")
                if u == "api/pick":
                    return self._send(200, {"path": pick_audio_path()})
                if u == "api/requests":
                    return self._send(200, {"requests": core.workflow_list(30)})
                if u == "api/status":
                    status = core.workflow_status(q.get("request_id", ""))
                    request = status["request"]
                    return self._send(200, {
                        "request_id": q.get("request_id"), "state": request["state"],
                        "state_ru": state_ru(request["state"]),
                        "source": request["source"], "recording_id": request["recording_id"],
                        "policy_revision": request["policy_revision"],
                        "policy": request["policy"], "model_task": request["model_task"],
                        "model_failure": request["model_failure"],
                        "readiness": status["readiness"]})
                if u == "api/questions":
                    return self._send(200, core.workflow_questions(q.get("request_id", "")))
                if u == "api/result":
                    status = core.workflow_status(q.get("request_id", ""))
                    recording_id = status["request"]["recording_id"]
                    if not recording_id:
                        return self._send(200, {"recording_id": None})
                    return self._send(200, result_view(core.read_recording_result(recording_id)))
                if u == "api/conspects":
                    recording_id = q.get("recording_id", "")
                    return self._send(200, {"conspects": core.recording_conspects(recording_id)})
                if u == "api/history":
                    kind = q.get("kind", "transcript")
                    return self._send(200, {"revisions": core.recording_revisions(
                        q.get("recording_id", ""), kind)})
                if u == "api/revision":
                    return self._send(200, core.read_recording_revision(
                        q.get("recording_id", ""), q.get("kind", "transcript"),
                        int(q.get("revision", "0"))))
                if u == "api/conflicts/all":
                    return self._send(200, core.list_all_conflicts())
                if u == "api/conflicts":
                    recording_id = q.get("recording_id", "")
                    return self._send(200, {
                        "documents": core.list_document_conflicts(recording_id),
                        "conspects": core.list_conspect_conflicts(recording_id)})
                return self._send(404, {"error": "не найдено"})
            except core.CoreError as ex:
                return self._send(400, {"error": str(ex)})
            except (storage.StorageError, sqlite3.Error, OSError) as ex:
                return self._send(400, {"error": core.explain_error(ex)})
            except (ValueError, KeyError) as ex:
                return self._send(400, {"error": f"не удалось выполнить запрос: {ex}"})
            except Exception as ex:    # неизвестный сбой: человек видит русское пояснение и причину, а не оборванную связь
                return self._send(500, {"error": core.explain_error(ex)})

        def do_POST(self):
            u = self._path(post=True)
            if u is None:
                return
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return self._refuse(415, {"error": "недопустимый тип данных"})
            n = self._length()
            if n is None:
                return self._refuse(400, {"error": "неверная длина запроса"})
            if n <= 0 or n > MAX_BODY_BYTES:
                return self._refuse(413, {"error": "недопустимый размер запроса"})
            self.connection.settimeout(30)
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                if not isinstance(body, dict):
                    raise ValueError("ожидается объект запроса")
                last[0] = time.time()
                if u == "api/start":
                    audio = str(body.get("audio", "")).strip()
                    if not audio or len(audio) > 300:
                        return self._send(400, {"error": "укажите путь к записи"})
                    for key in ("require_conspect", "allow_cloud_model"):
                        if type(body.get(key)) is not bool:
                            raise ValueError(f"{key}: требуется логическое значение")
                    import product_policy
                    policy = product_policy.resolve({
                        key: body[key] for key in (
                            "require_conspect", "allow_cloud_model", "participation",
                            "goal", "question_budget") if key in body
                    })
                    return self._send(200, core.start_workflow(audio, policy))
                if u == "api/backup":
                    destination = str(body.get("destination", "")).strip()
                    if not destination or len(destination) > 300:
                        return self._send(400, {"error": "укажите путь к новому файлу резерва"})
                    return self._send(200, core.backup_store(destination))
                if u == "api/verify/prepare":
                    return self._send(200, core.workflow_verification_snapshot(
                        str(body.get("request_id", ""))))
                if u == "api/verify/confirm":
                    return self._send(200, core.workflow_confirm_verification(body.get("snapshot")))
                if u == "api/quotes/repair/prepare":
                    return self._send(200, core.workflow_repair_quotes_prepare(str(body.get("request_id", ""))))
                if u == "api/quotes/repair/confirm":
                    # Облако вызывается только по явному согласию из окна и по снимку того, что человек видел.
                    if body.get("allow_cloud_model") is not True:
                        return self._send(400, {"error": "Нужно явное согласие на отправку: ничего не отправлено."})
                    snapshot = body.get("snapshot")
                    if not isinstance(snapshot, dict):
                        return self._send(400, {"error": "Нет снимка того, что уйдёт в облако: откройте окно заново."})
                    return self._send(200, core.workflow_repair_quotes(
                        str(snapshot.get("request_id", "")), via="screen", snapshot=snapshot, allow_cloud_model=True))
                if u == "api/restore/prepare":
                    return self._send(200, core.prepare_restore_snapshot(
                        str(body.get("recording_id", "")), int(body.get("source_revision", 0)),
                        int(body.get("review_revision", 0))))
                if u == "api/restore/confirm":
                    return self._send(200, core.restore_recording_revision(body.get("snapshot")))
                conflict = (str(body.get("recording_id", "")), str(body.get("conflict_id", "")),
                            str(body.get("kind") or "document"))    # kind — область: document или conspect
                if u == "api/conflict/preview":
                    return self._send(200, core.preview_conflict(*conflict))
                if u == "api/conflict/copy":
                    copy = core.conflict_copy(*conflict)
                    return self._send(200, copy["data"], "application/octet-stream", [
                        ("Content-Disposition", "attachment; filename*=UTF-8''" + quote(copy["name"], safe=""))])
                if u == "api/conflict/resolve":
                    if body.get("confirm") is not True:
                        return self._send(400, {"error": "Нужно явное подтверждение замены файла."})
                    return self._send(200, core.replace_conflict_file(body.get("snapshot")))
                if u == "api/conflict/retry":
                    return self._send(200, core.retry_conflict_file(*conflict))
                return self._send(404, {"error": "не найдено"})
            except core.CoreError as ex:
                return self._send(400, {"error": str(ex)})
            except (storage.StorageError, sqlite3.Error, OSError) as ex:
                return self._send(400, {"error": core.explain_error(ex)})
            except (ValueError, KeyError) as ex:
                return self._send(400, {"error": f"не удалось обработать запрос: {ex}"})
            except llm_provider.ProviderError as ex:     # отказ облачного исполнителя: человек видит его причину
                return self._send(400, {"error": str(ex), "error_code": ex.code})
            except Exception as ex:    # неизвестный сбой: человек видит русское пояснение и причину, а не оборванную связь
                return self._send(500, {"error": core.explain_error(ex)})

    srv, port = None, port
    for port in range(port, port + 20):
        try:
            srv = ExclusiveHTTPServer(("127.0.0.1", port), H)
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit("нет свободного порта")
    return srv, f"http://127.0.0.1:{port}{prefix}", last


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--hours", type=float, default=6.0)
    a = ap.parse_args(argv)
    srv, url, last = make_server(a.port)
    print(f"[экран] {url}", file=sys.stderr, flush=True)

    def idle():
        while time.time() - last[0] < a.hours * 3600:
            time.sleep(30)
        srv.shutdown()

    threading.Thread(target=idle, daemon=True).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
