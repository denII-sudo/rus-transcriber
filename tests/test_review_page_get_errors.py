"""Ошибки чтения страницы возвращаются как безопасный JSON, а не обрыв соединения."""
import http.client
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.parse import urlsplit


PROJECT = Path(__file__).resolve().parent.parent


class GetErrorsHTTP(unittest.TestCase):
    def test_answers_success_and_read_error(self):
        with tempfile.TemporaryDirectory(prefix="review_page_get_") as tmp:
            root = Path(tmp)
            status, flag = root / "status.json", root / "fail"
            stderr = root / "stderr.txt"
            script = """
import sys
import os
import threading
from pathlib import Path
import review_page
threading.Thread(target=lambda: (sys.stdin.buffer.read(), os._exit(0)), daemon=True).start()
flag = Path(sys.argv[1])
def answers(_):
    if flag.exists():
        raise RuntimeError("private detail")
    return {"answers": {}}
review_page.answers_payload = answers
sys.argv = ["review_page.py", sys.argv[2], "--status", sys.argv[3], "--port", "19380"]
review_page.main()
"""
            with stderr.open("w+", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    [sys.executable, "-B", "-c", script, str(flag), str(root / "recording.json"), str(status)],
                    cwd=str(PROJECT), env=os.environ.copy(), stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=log)
                try:
                    for _ in range(100):
                        if status.exists():
                            try:
                                url = urlsplit(json.loads(status.read_text(encoding="utf-8"))["url"])
                                break
                            except (OSError, ValueError, KeyError):
                                pass
                        if proc.poll() is not None:
                            self.fail("сервер завершился до запуска")
                        time.sleep(0.1)
                    else:
                        self.fail("сервер не запустился")

                    def get():
                        conn = http.client.HTTPConnection("127.0.0.1", url.port, timeout=10)
                        try:
                            conn.request("GET", url.path + "api/answers")
                            response = conn.getresponse()
                            return response.status, json.loads(response.read())
                        finally:
                            conn.close()

                    self.assertEqual(get(), (200, {"answers": {}}))
                    flag.touch()
                    self.assertEqual(get(), (500, {"error": "не удалось обработать запрос (код: RuntimeError)"}))
                    log.flush()
                    log.seek(0)
                    output = log.read()
                    self.assertIn("[page] ошибка запроса: RuntimeError", output)
                    self.assertNotIn("private detail", output)
                    self.assertNotIn("Traceback", output)
                finally:
                    proc.stdin.close()
                    proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
