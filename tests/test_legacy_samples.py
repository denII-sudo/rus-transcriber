"""Образцы старого формата (tests/legacy_samples/): хэши записаны, личных путей и данных нет.

Образцы собрала программа один раз (Л-3.1) — случайная правка руками должна ронять тест, а не менять образец
молча. Менять образец можно только пересборкой скриптом из описания PR на `main` до Л-5, с новым хэшем.
"""
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

import legacy_sample_loader as samples


class SampleFiles(unittest.TestCase):
    def test_set_of_samples_and_hashes_are_recorded(self):
        found = sorted(p.name for p in samples.SAMPLES.iterdir())
        self.assertEqual(found, sorted(samples.HASHES), "образец добавлен или убран без записи хэша")
        for name, digest in samples.HASHES.items():
            blob = (samples.SAMPLES / name).read_bytes()
            self.assertEqual(hashlib.sha256(blob).hexdigest(), digest, f"образец {name} изменён")

    def test_encoding_line_ends_and_no_personal_paths(self):
        for p in samples.SAMPLES.iterdir():
            blob = p.read_bytes()
            self.assertFalse(blob.startswith(b"\xef\xbb\xbf"), f"{p.name}: BOM")
            self.assertNotIn(b"\r", blob, f"{p.name}: CRLF")
            text = blob.decode("utf-8")
            self.assertIsNone(re.search(r"[A-Za-z]:\\\\|[A-Za-z]:/|/home/|\\\\Users\\\\|AppData", text),
                              f"{p.name}: в образце настоящий путь")
            data = json.loads(text)
            self.assertIn(data["transcript"], data["files"], p.name)
            self.assertIn("{AUDIO}", data["files"][data["transcript"]], f"{p.name}: путь к аудио не заменён меткой")

    def test_load_puts_paths_of_this_record(self):
        with tempfile.TemporaryDirectory(prefix="legacy-sample-") as tmp:
            root = Path(tmp)
            side = root / "встреча.rus-transcriber"
            audio = root / "встреча.wav"
            jp = samples.load("built", side, audio)
            self.assertTrue(jp.is_file())
            self.assertEqual(json.loads(jp.read_text(encoding="utf-8"))["file"], str(audio))
            pack = jp.with_name("встреча.review")
            self.assertEqual(json.loads((pack / "manifest.json").read_text(encoding="utf-8"))["job_json"], str(jp))
            self.assertEqual(json.loads((pack / "answers.json").read_text(encoding="utf-8"))["manifest"]["job_json"],
                             str(jp))


if __name__ == "__main__":
    unittest.main()
