"""Аудит страницы карточек 28.11 (И2): ошибка при GET — ответ 500 с понятным текстом, а не обрыв соединения.

Настоящий `review_page.py` в подпроцессе; подменено только чтение ответов (`answers_payload` падает). Данные
синтетические, всё во временной папке.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.parse import urlsplit

import test_review_page_clips as clips

PROJECT = clips.PROJECT
RUN = ("import sys; sys.path.insert(0, sys.argv[1]); import review_page as r\n"
       "def boom(_):\n    raise RuntimeError('синтетический сбой чтения')\n"
       "r.answers_payload = boom; sys.argv = ['review_page.py'] + sys.argv[2:]; r.main()\n")


class GetErrorHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="review_page_get_")
        root = Path(cls.tmp.name)
        jp, _ = clips.migrated(root)
        cls.status = root / "status.json"
        data, temp = root / "data", root / "temp"
        data.mkdir()
        temp.mkdir()
        env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=str(temp), TMP=str(temp),
                   PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        cls.proc = subprocess.Popen([sys.executable, "-B", "-c", RUN, str(PROJECT), str(jp), "--status",
                                     str(cls.status), "--port", "19381"], cwd=str(PROJECT), env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
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
        cls.tearDownClass()
        raise RuntimeError("тестовая страница не запустилась")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.stderr = cls.proc.communicate(timeout=10)[1].decode("utf-8", "replace")
        cls.tmp.cleanup()

    get = clips.LegacyClipsHTTP.get

    def test_failed_read_answers_500_with_text(self):
        code, body, headers = self.get("api/answers")
        self.assertEqual(code, 500)
        error = json.loads(body)["error"]
        self.assertTrue(error)
        self.assertNotIn("синтетический сбой чтения", error, "текст исключения на страницу не попадает")
        self.assertEqual(self.get("clips/r1.mp3")[0], 200, "сервер жив после ошибки")


if __name__ == "__main__":
    unittest.main()
