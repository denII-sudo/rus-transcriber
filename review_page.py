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
                store = P._read(pd / "answers.json", {"answers": {}})
                store["applied_answers"] = store.get("applied_answers") or {}
                return self._send(200, store)
            f = (pd / u).resolve()
            if pd.resolve() in f.parents and f.is_file() and f.parent.name in ("clips", pd.name) \
                    and f.name != "answers.json":
                return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            self._send(404, {"error": "не найдено"})

        def do_POST(self):
            u = self._path(post=True)
            if u is None:
                return
            if u not in ("api/answer", "api/voice", "api/apply", "api/preview"):
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
                    if not isinstance(obj, dict) or obj.get("version") != man.get("version") or not any(
                            it.get("id") == obj.get("id") for it in man.get("items", [])) or not isinstance(obj.get("text"), str) or not obj["text"].strip():
                        return self._send(400, {"error": "неверная карточка, версия или текст"})
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
                        rep = P.add_answers(a.json_path, [ans], "local")
                    return self._send(200, rep)
                if u == "api/voice":
                    q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                    man = P.manifest(a.json_path) or {}
                    item = next((it for it in man.get("items", []) if it.get("id") == q.get("id")), None)
                    if q.get("version") != man.get("version") or item is None:
                        return self._send(400, {"error": "неизвестная карточка или версия"})
                    f = pd / "voice" / f"page-{uuid.uuid4().hex}.{types[ctype]}"
                    f.parent.mkdir(exist_ok=True)
                    f.write_bytes(body)
                    text = P.recognize_here([f]).get(str(f), "")
                    return self._send(200, {"heard": text, "match": P.match(text, item)})
            except Exception as ex:
                print(f"[page] ошибка запроса: {type(ex).__name__}", file=sys.stderr, flush=True)
                return self._send(400, {"error": "не удалось обработать запрос"})
            self._send(404, {"error": "не найдено"})

    srv, port = None, a.port
    for port in range(a.port, a.port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), H)
            break
        except OSError:
            continue
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
