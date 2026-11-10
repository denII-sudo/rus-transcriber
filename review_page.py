"""Локальная страница карточек разбора: http://127.0.0.1:<порт>/ (только с этого компьютера).

Отдаёт cards.html (та же страница, что публикуется в облако), manifest.json и клипы пакета,
принимает ответы касанием (POST /api/answer) и голосом (POST /api/voice — запись из браузера,
распознаётся здесь же GigaAM). Ответы пишутся в answers.json пакета с source=local.
Сама выключается после --hours часов без запросов.

    python review_page.py <...transcript.json> --status <файл статуса> [--port 8765] [--hours 3]
"""
import argparse
import json
import mimetypes
import sys
import threading
import time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--status")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--hours", type=float, default=3.0)
    a = ap.parse_args()
    pd = P.pack_dir(a.json_path)
    last = [time.time()]
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print("[page] " + fmt % args, file=sys.stderr, flush=True)

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            last[0] = time.time()
            u = urlparse(self.path).path
            if u in ("/", "/index.html"):
                html = (HERE / "cards.html").read_text(encoding="utf-8")
                return self._send(200, SKELETON.replace("{BODY}", html).encode("utf-8"), "text/html; charset=utf-8")
            if u == "/api/answers":
                return self._send(200, P._read(pd / "answers.json", {"answers": {}}))
            f = (pd / u.lstrip("/")).resolve()
            if pd.resolve() in f.parents and f.is_file() and f.parent.name in ("clips", pd.name) \
                    and f.name != "answers.json":
                return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            self._send(404, {"error": "не найдено"})

        def do_POST(self):
            last[0] = time.time()
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            try:
                if u.path == "/api/answer":
                    ans = json.loads(body.decode("utf-8"))
                    with lock:
                        rep = P.add_answers(a.json_path, [ans], "local")
                    return self._send(200, rep)
                if u.path == "/api/voice":
                    q = {k: v[0] for k, v in parse_qs(u.query).items()}
                    ext = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/wav": "wav"} \
                        .get((self.headers.get("Content-Type") or "").split(";")[0], "webm")
                    f = pd / "voice" / f"{q['id']}-page-{int(time.time())}.{ext}"
                    f.parent.mkdir(exist_ok=True)
                    f.write_bytes(body)
                    text = P.recognize_here([f]).get(str(f), "")
                    with lock:
                        rep = P.add_answers(a.json_path, [{"id": q["id"], "version": q.get("version"),
                                                           "text": text, "audio": str(f), "heard": True}], "local")
                    rep["heard"] = text
                    return self._send(200, rep)
            except Exception as ex:
                return self._send(400, {"error": f"{type(ex).__name__}: {ex}"})
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
    url = f"http://127.0.0.1:{port}/"
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
    print(f"[page] {url}", file=sys.stderr, flush=True)
    srv.serve_forever()
    if a.status:
        P._write(a.status, {**st, "state": "done", "stage": "страница закрыта (простой)",
                            "updated": time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    main()
