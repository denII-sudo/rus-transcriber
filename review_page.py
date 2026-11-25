"""Локальная страница карточек разбора с секретным адресом для каждого запуска.

Отдаёт cards.html (та же страница, что публикуется в облако), manifest.json и клипы пакета,
принимает ответы касанием (POST /api/answer) и голосом (POST /api/voice — запись из браузера,
распознаётся здесь же GigaAM). Ответы пишутся в answers.json пакета с source=local.
Сама выключается после --hours часов без запросов.

    python review_page.py <...transcript.json> --status <файл статуса> [--port 8765] [--hours 3]
"""
import argparse
import json
import mimetypes
import secrets
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402,F401
import review_pack as P  # noqa: E402

SKELETON = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);'
            'padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif}'
            'img{max-width:100%}[hidden]{display:none!important}</style></head><body>{BODY}</body></html>')
mimetypes.add_type("audio/mpeg", ".mp3")
MAX_ANSWER_BYTES = 64 * 1024
MAX_VOICE_BYTES = 16 * 1024 * 1024

# Причины, которые страница может показать человеку дословно: в них нет путей, текста ответов и
# чужих данных. Сообщение с таким началом и подробностями после «:» показывается без подробностей.
SAFE_ERRORS = (
    "версия пакета изменилась",
    P.REV_STALE,
    P.REV_UNKNOWN,
    "карточка не найдена",
    "неизвестное событие карточки",
    "неверный идентификатор сессии страницы",
    "пакет разбора не собран: сначала build_review_pack",
    "Ответ неизвестного происхождения нельзя применять как человеческий; необходимо явное подтверждение",
    "Ответы сохранены в БД; обновление документа не завершено",
    "Ответы сохранены, но конспекты не обновлены",
)


def safe_error(ex):
    """Причина ошибки запроса по-русски для страницы: известная причина или общая фраза с кодом.
    Текст исключения, пути и ответы на страницу не попадают."""
    code = type(ex).__name__
    import core
    import documents
    if isinstance(ex, (documents.NotWritten, documents.NotSynced)):
        return str(ex)       # исход записи по-русски: имя файла и следующий шаг, пути и текст ответов в нём нет
    if code == "Conflict" and type(ex).__module__ == "storage" and str(ex) in (
            "unresolved document conflict", "unresolved conspect conflict"):
        return core.say_error(ex)
    if code == "CoreError" and ex.args and isinstance(ex.args[0], str) and ex.args[0].startswith("Записано в базу, но"):
        return ex.args[0]
    if code == "Conflict" and type(ex).__module__ == "storage":
        return ("запись изменилась, пока применялись ответы; ничего не применено — обновите страницу "
                f"и повторите (код: {code})")
    if isinstance(ex, json.JSONDecodeError):
        return f"запрос повреждён: неверный JSON (код: {code})"
    if (isinstance(ex, (ValueError, RuntimeError)) or code == "CoreError") and ex.args and isinstance(ex.args[0], str):
        message = ex.args[0]
        for known in SAFE_ERRORS:
            if message == known or message.startswith(known + ":"):
                return known
    return f"не удалось обработать запрос (код: {code})"


class ExclusiveHTTPServer(ThreadingHTTPServer):
    """Сервер, который не делит порт с другим процессом.

    HTTPServer включает SO_REUSEADDR, а на Windows с ним второй сервер молча занимает порт,
    который уже слушает прежняя страница: новый адрес «открыт», но запросы уходят старой.
    Без него и с SO_EXCLUSIVEADDRUSE занятый порт даёт OSError, и main() берёт следующий.
    """
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def answers_payload(json_path):
    """Ответ api/answers: состояние пакета и то, что по нему считает Python для страницы."""
    store = P.answers_state(json_path, {"answers": {}})
    store["applied_answers"] = store.get("applied_answers") or {}
    man = P.manifest(json_path) or {}
    # Состояние каждого ответа считает Python; страница его только показывает.
    store["answer_states"] = P.answer_states_of(store, man)
    # места, уже прослушанные целиком (раздел 10): их кнопки прослушивания доступны сразу
    store["heard_ids"] = P.heard_ids(store, man)
    # По ним страница узнаёт, что карточки заменены или дописаны, и перечитывает manifest.json.
    store["pack_version"], store["pack_rev"] = man.get("version"), man.get("pack_rev", 0)
    if "_expected_versions" in store:
        # SQLite (Х1): события карточек странице не нужны, а у таблицы событий нет предела 2000
        store.pop("interaction_events", None)
        store.pop("interaction_events_dropped", None)
    return store


def bind_free_port(start, handler, attempts=20):
    """Первый свободный порт начиная с start -> (сервер, порт) или (None, None)."""
    for port in range(start, start + attempts):
        try:
            return ExclusiveHTTPServer(("127.0.0.1", port), handler), port
        except OSError:
            continue
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--status")
    ap.add_argument("--markdown")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--hours", type=float, default=3.0)
    a = ap.parse_args()
    pd = P.pack_dir(a.json_path)
    last = [time.time()]
    lock = threading.Lock()
    token = secrets.token_urlsafe(32)
    prefix = f"/{token}/"

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # URL содержит секрет сессии: не записывать запрос целиком в журнал.
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(b)

        def _path(self, post=False):
            host = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != host:
                self._send(403, {"error": "недопустимый адрес"})
                return None
            if post and self.headers.get("Origin") != f"http://{host}":
                self._send(403, {"error": "недопустимый источник"})
                return None
            u = urlparse(self.path)
            if not u.path.startswith(prefix):
                self._send(403, {"error": "недействительная сессия"})
                return None
            return u.path[len(prefix):]

        def do_GET(self):
            u = self._path()
            if u is None:
                return
            last[0] = time.time()
            if u in ("", "index.html"):
                html = (HERE / "cards.html").read_text(encoding="utf-8")
                return self._send(200, SKELETON.replace("{BODY}", html).encode("utf-8"), "text/html; charset=utf-8")
            if u == "api/answers":
                return self._send(200, answers_payload(a.json_path))
            if u == "manifest.json":
                manifest = P.manifest(a.json_path)
                if manifest is None:
                    return self._send(404, {"error": "пакет разбора не найден"})
                return self._send(200, manifest)
            f = (pd / u).resolve()
            if pd.resolve() in f.parents and f.is_file() and (f.parent == pd.resolve() or
                    (pd / "clips").resolve() in f.parents) \
                    and f.name != "answers.json":
                return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            self._send(404, {"error": "не найдено"})

        def do_POST(self):
            u = self._path(post=True)
            if u is None:
                return
            if u not in ("api/answer", "api/voice", "api/apply", "api/preview", "api/event"):
                return self._send(404, {"error": "не найдено"})
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            types = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/wav": "wav"}
            if (u != "api/voice" and ctype != "application/json") or (u == "api/voice" and ctype not in types):
                return self._send(415, {"error": "недопустимый тип данных"})
            length = self.headers.get("Content-Length", "")
            if len(length) > 12 or not length.isascii() or not length.isdecimal() or self.headers.get("Transfer-Encoding"):
                return self._send(400, {"error": "неверная длина запроса"})
            n = int(length)
            if n <= 0 or n > (MAX_VOICE_BYTES if u == "api/voice" else MAX_ANSWER_BYTES):
                return self._send(413, {"error": "недопустимый размер запроса"})
            self.connection.settimeout(30)
            try:
                body = self.rfile.read(n)
                if len(body) != n:
                    return self._send(400, {"error": "неполный запрос"})
                last[0] = time.time()
                if u == "api/preview":
                    obj = json.loads(body.decode("utf-8"))
                    man = P.manifest(a.json_path) or {}
                    item = next((it for it in man.get("items", []) if isinstance(obj, dict)
                                 and it.get("id") == obj.get("id")), None)
                    if not isinstance(obj, dict) or obj.get("version") != man.get("version") or item is None \
                            or not isinstance(obj.get("text"), str) or not obj["text"].strip():
                        return self._send(400, {"error": "неверная карточка, версия или текст"})
                    stale = P.rev_refusal(obj.get("rev"), item)
                    if stale:
                        return self._send(400, {"error": stale})
                    with lock:
                        aid = obj["id"]
                        free = P.free_answers(a.json_path)
                        free[aid] = {"text": obj["text"]}
                        decisions, _ = P.decisions(a.json_path)
                        decisions = [d for d in decisions if d.get("id") != aid]
                        conflicts = [c for c in P.preflight_answers(a.json_path, a.markdown, decisions, free)
                                     if aid in c["cards"]]
                        if conflicts:
                            result = {"cards": [], "preflight_conflicts": conflicts}
                        else:
                            result = P.apply_free(a.json_path, a.markdown, {aid: free[aid]}, dry_run=True)
                    return self._send(200, result)
                if u == "api/apply":
                    obj = json.loads(body.decode("utf-8"))
                    if not isinstance(obj, dict) or obj.get("version") != (P.manifest(a.json_path) or {}).get("version"):
                        return self._send(400, {"error": "версия пакета изменилась"})
                    if not a.markdown:
                        return self._send(400, {"error": "путь к Markdown не указан; запустите страницу через review-page"})
                    import core
                    with lock:
                        rep = core.apply_answers_files(a.json_path, a.markdown)
                    return self._send(200, rep)
                if u == "api/answer":
                    ans = json.loads(body.decode("utf-8"))
                    if not isinstance(ans, dict):
                        return self._send(400, {"error": "ожидается объект ответа"})
                    # Браузер не должен указывать произвольный путь к аудиофайлу.
                    ans.pop("audio", None)
                    with lock:
                        rep = P.add_answers(a.json_path, [ans], "local", trusted_actor="human")
                    return self._send(200, rep)
                if u == "api/event":
                    event = json.loads(body.decode("utf-8"))
                    if not isinstance(event, dict):
                        return self._send(400, {"error": "ожидается объект события"})
                    with lock:
                        rep = P.add_interaction_event(
                            a.json_path, event.get("id"), event.get("version"),
                            event.get("type"), event.get("session"), event.get("rev"))
                    return self._send(200, rep)
                if u == "api/voice":
                    q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                    man = P.manifest(a.json_path) or {}
                    item = next((it for it in man.get("items", []) if it.get("id") == q.get("id")), None)
                    if q.get("version") != man.get("version") or item is None:
                        return self._send(400, {"error": "неизвестная карточка или версия"})
                    rev = q.get("rev")
                    stale = P.rev_refusal(int(rev) if rev is not None and rev.isdecimal() and rev.isascii()
                                          else rev, item)
                    if stale:
                        return self._send(400, {"error": stale})
                    f = pd / "voice" / f"page-{uuid.uuid4().hex}.{types[ctype]}"
                    f.parent.mkdir(exist_ok=True)
                    f.write_bytes(body)
                    text = P.recognize_here([f]).get(str(f), "")
                    return self._send(200, {"heard": text, "match": P.match(text, item)})
            except Exception as ex:
                print(f"[page] ошибка запроса: {type(ex).__name__}", file=sys.stderr, flush=True)
                return self._send(400, {"error": safe_error(ex)})
            self._send(404, {"error": "не найдено"})

    srv, port = bind_free_port(a.port, H)
    if srv is None:
        raise SystemExit("нет свободного порта")
    url = f"http://127.0.0.1:{port}{prefix}"
    st = {"state": "running", "kind": "review_page", "stage": "страница открыта", "url": url, "progress": 0,
          "started_ts": round(time.time(), 1), "stage_started_ts": round(time.time(), 1),
          "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if a.status:
        P._write(a.status, st)

    def idle():
        while time.time() - last[0] < a.hours * 3600:
            time.sleep(30)
        srv.shutdown()

    threading.Thread(target=idle, daemon=True).start()
    # При прямом запуске без файла статуса адрес больше негде получить.
    print(f"[page] сервер запущен на порту {port}" if a.status else f"[page] {url}",
          file=sys.stderr, flush=True)
    srv.serve_forever()
    if a.status:
        P._write(a.status, {**st, "state": "done", "stage": "страница закрыта (простой)",
                            "updated": time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    main()
