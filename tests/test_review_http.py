"""HTTP-защита локальных карточек на синтетическом пакете, без моделей и реальных данных."""
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.parse import quote, urlsplit


PROJECT = Path(__file__).resolve().parent.parent


class ReviewHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="review_http_")
        root = Path(cls.tmp.name)
        cls.pack = root / "demo.review"
        (cls.pack / "clips").mkdir(parents=True)
        (cls.pack / "clips" / "r1.mp3").write_bytes(b"demo audio")
        (cls.pack / "manifest.json").write_text(json.dumps({
            "version": "v1", "audio": "demo.wav", "items": [{"id": "r1", "t": "00:00:01",
                "clip": "clips/r1.mp3", "text": "да", "speaker": "Участник",
                "question": "Как правильно?", "options": [{"label": "да", "decision": {}}]}]
        }), encoding="utf-8")
        (cls.pack / "answers.old.json").write_text('{"archive": true}', encoding="utf-8")
        (cls.pack / "cloud.json").write_text('{"debug": true}', encoding="utf-8")
        cls.status = root / "status.json"
        data, temp = root / "data", root / "temp"
        data.mkdir()
        temp.mkdir()
        env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=str(temp), TMP=str(temp),
                   PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        # Голосовой маршрут проверяем без загрузки модели.
        code = ("import sys, review_page; "
                "review_page.P.recognize_here=lambda files: {str(p): 'первый' for p in files}; "
                "sys.argv=['review_page.py', sys.argv[1], '--status', sys.argv[2], '--port', '19350']; "
                "review_page.main()")
        cls.proc = subprocess.Popen([sys.executable, "-B", "-c", code, str(root / "demo.transcript.json"),
                                     str(cls.status)], cwd=str(PROJECT), env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if cls.status.exists():
                try:
                    url = json.loads(cls.status.read_text(encoding="utf-8")).get("url")
                    if url:
                        cls.url = urlsplit(url)
                        return
                except (OSError, ValueError):
                    pass
            if cls.proc.poll() is not None:
                break
            time.sleep(0.1)
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.tmp.cleanup()
        raise RuntimeError("тестовая страница не запустилась")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.tmp.cleanup()

    def request(self, path="", method="GET", body=None, headers=None, absolute=False):
        conn = http.client.HTTPConnection("127.0.0.1", self.url.port, timeout=10)
        try:
            conn.request(method, path if absolute else self.url.path + path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            conn.close()

    def post(self, path, body, **headers):
        return self.request(path, "POST", body, {"Origin": f"http://127.0.0.1:{self.url.port}", **headers})

    def test_page_and_debug_files_require_session(self):
        status, page, headers = self.request()
        self.assertEqual(status, 200)
        self.assertIn(b"api/answers", page)
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        for name in ("manifest.json", "clips/r1.mp3", "answers.old.json", "cloud.json", "api/answers"):
            with self.subTest(name=name):
                self.assertEqual(self.request(name)[0], 200)
                self.assertEqual(self.request("/" + name, absolute=True)[0], 403)

    def test_host_origin_token_and_content_type(self):
        payload = json.dumps({"id": "r1", "version": "v1", "option": 0}).encode()
        origin = f"http://127.0.0.1:{self.url.port}"
        self.assertEqual(self.request("api/answers", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request("api/answer", "POST", payload,
                         {"Origin": "https://evil.example", "Content-Type": "text/plain"})[0], 403)
        self.assertEqual(self.request("api/answer", "POST", payload,
                         {"Content-Type": "application/json"})[0], 403)
        self.assertEqual(self.post("api/answer", payload, **{"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("/api/answer", "POST", payload,
                         {"Origin": origin, "Content-Type": "application/json"}, absolute=True)[0], 403)
        status, data, _ = self.post("api/answer", payload, **{"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["accepted"][0]["id"], "r1")

    def test_voice_rejects_traversal_before_write_and_accepts_valid_card(self):
        suffix = "?id=" + quote(r"..\..\escape", safe="") + "&version=v1"
        self.assertEqual(self.post("api/voice" + suffix, b"demo", **{"Content-Type": "audio/wav"})[0], 400)
        self.assertEqual(list(Path(self.tmp.name).glob("escape*")), [])
        self.assertEqual(self.post("api/voice?id=r1&version=old", b"demo",
                                  **{"Content-Type": "audio/wav"})[0], 400)
        status, data, _ = self.post("api/voice?id=r1&version=v1", b"demo",
                                    **{"Content-Type": "audio/wav"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["heard"], "первый")
        self.assertEqual(len(list((self.pack / "voice").glob("page-*.wav"))), 1)

    def test_limits(self):
        from review_page import MAX_ANSWER_BYTES, MAX_VOICE_BYTES
        self.assertEqual(self.post("api/answer", b"x" * (MAX_ANSWER_BYTES + 1),
                                   **{"Content-Type": "application/json"})[0], 413)
        # Не отправляем большой массив: сервер обязан отклонить по заголовку до чтения тела.
        conn = http.client.HTTPConnection("127.0.0.1", self.url.port, timeout=10)
        try:
            conn.putrequest("POST", self.url.path + "api/voice?id=r1&version=v1")
            conn.putheader("Origin", f"http://127.0.0.1:{self.url.port}")
            conn.putheader("Content-Type", "audio/wav")
            conn.putheader("Content-Length", str(MAX_VOICE_BYTES + 1))
            conn.endheaders()
            self.assertEqual(conn.getresponse().status, 413)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
