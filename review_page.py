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
import re
import secrets
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlparse

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
MAX_DRAIN_BYTES = 1024 * 1024   # слишком большое, но не больше этого тело дочитывается и отбрасывается перед ответом 413
DRAIN_TIMEOUT = 2               # секунд на всё такое дочитывание вместе (общий срок, а не на одно чтение)
# Тот же приём — в workflow_page.py (там свои копии констант и _refuse): правя один, смотри и другой.

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
    # 9.3: место открыто; ответ на карточке не внесётся (версии стенограммы карточки нет в базе) — только чтение
    store["card_conditions"] = P.card_conditions(json_path, man)
    # По ним страница узнаёт, что карточки заменены или дописаны, и перечитывает manifest.json.
    store["pack_version"], store["pack_rev"] = man.get("version"), man.get("pack_rev", 0)
    if "_expected_versions" in store:
        # SQLite (Х1): события карточек странице не нужны, а у таблицы событий нет предела 2000
        store.pop("interaction_events", None)
        store.pop("interaction_events_dropped", None)
    return store


def preview_answer(json_path, markdown, aid, text):
    """Ответ api/preview на свой текст text карточки aid, без записи (вызывать под замком страницы).
    КН-14: сначала те же проверки, что у «Применить» (answer_guard: версия стенограммы, слова вне карточки) —
    {"cards": [], "guard": причина}; затем пересечение с другими сохранёнными ответами — preflight_conflicts;
    иначе — изменение apply_free без записи."""
    guard = P.answer_guard(json_path, markdown, aid, {"kind": "free", "text": text})
    if guard:
        return {"cards": [], "guard": guard}
    free = P.free_answers(json_path)
    free[aid] = {"text": text}
    decisions, _ = P.decisions(json_path)
    decisions = [d for d in decisions if d.get("id") != aid]
    conflicts = [c for c in P.preflight_answers(json_path, markdown, decisions, free) if aid in c["cards"]]
    if conflicts:
        return {"cards": [], "preflight_conflicts": conflicts}
    return P.apply_free(json_path, markdown, {aid: free[aid]}, dry_run=True)


def legacy_clip(json_path, rel):
    """Клип карточки пакета, перенесённого П7, из прежней папки пакета (рядом со старой стенограммой) или None.

    Перенос клипы не копирует (Р-3): у перенесённых карточек ссылка clips/<id>.mp3 указывает в новую папку пакета,
    а файл остался в прежней. Только чтение: путь должен быть клипом карточки или её истории в текущем manifest,
    .mp3 без «..», и после resolve() — внутри <прежняя папка пакета>/clips. Прежняя папка — только по
    legacy_job_json из manifest."""
    rel = unquote(rel)
    p = PurePosixPath(rel)
    if "\\" in rel or p.is_absolute() or ".." in p.parts or p.suffix.lower() != ".mp3" or p.parts[:1] != ("clips",):
        return None
    man = P.manifest(json_path) or {}
    legacy = man.get("legacy_job_json")
    if not isinstance(legacy, str) or not legacy:
        return None
    listed = set()
    for it in man.get("items") or []:
        if isinstance(it, dict):
            listed.update(ref.get("clip") for ref in [it] + [h for h in it.get("history") or [] if isinstance(h, dict)])
    if rel not in listed:
        return None
    old = P.pack_dir(legacy)
    f = (old / rel).resolve()
    return f if (old / "clips").resolve() in f.parents and f.is_file() else None


def byte_range(header, size):
    """Один диапазон заголовка Range (bytes=a-b, a-, -n) -> (начало, конец включительно); None — отдать файл
    целиком (заголовка нет, он непонятен, несколько диапазонов или конец раньше начала); False — диапазон
    за концом файла (ответ 416)."""
    m = re.fullmatch(r"bytes=(\d{0,15})-(\d{0,15})", (header or "").strip())
    if not m or not (m[1] or m[2]):
        return None
    if not m[1]:
        n = int(m[2])
        return (max(0, size - n), size - 1) if n and size else False
    start = int(m[1])
    if start >= size:
        return False
    end = min(int(m[2]), size - 1) if m[2] else size - 1
    return (start, end) if end >= start else None


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

        def _send(self, code, body, ctype="application/json; charset=utf-8", close=False, headers=None):
            b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
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
            в срок — отвечаем сразу, соединение закрывается. Тот же приём — в workflow_page.py."""
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
                return self._send_file(f)
            # клип старой карточки перенесённого пакета — из прежней папки пакета, только чтение (Р-3)
            f = legacy_clip(a.json_path, u)
            if f is not None:
                return self._send_file(f)
            self._send(404, {"error": "не найдено"})

        def _send_file(self, f):
            """Файл пакета; с Range — его часть (206): без этого Chromium не перематывает клип."""
            data = f.read_bytes()
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            rng = byte_range(self.headers.get("Range"), len(data))
            if rng is False:
                return self._send(416, b"", ctype, headers={"Content-Range": f"bytes */{len(data)}",
                                                            "Accept-Ranges": "bytes"})
            if rng is None:
                return self._send(200, data, ctype, headers={"Accept-Ranges": "bytes"})
            start, end = rng
            return self._send(206, data[start:end + 1], ctype, headers={
                "Content-Range": f"bytes {start}-{end}/{len(data)}", "Accept-Ranges": "bytes"})

        def do_POST(self):
            u = self._path(post=True)
            if u is None:
                return
            if u not in ("api/answer", "api/voice", "api/apply", "api/preview", "api/event"):
                return self._refuse(404, {"error": "не найдено"})
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            types = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/wav": "wav"}
            if (u != "api/voice" and ctype != "application/json") or (u == "api/voice" and ctype not in types):
                return self._refuse(415, {"error": "недопустимый тип данных"})
            n = self._length()
            if n is None:
                return self._refuse(400, {"error": "неверная длина запроса"})
            if n <= 0 or n > (MAX_VOICE_BYTES if u == "api/voice" else MAX_ANSWER_BYTES):
                return self._refuse(413, {"error": "недопустимый размер запроса"})
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
                        result = preview_answer(a.json_path, a.markdown, obj["id"], obj["text"])
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
