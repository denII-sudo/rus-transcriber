"""Мелкие правки локального экрана: регистр путей, местное время, тёмная тема."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core_screen  # noqa: E402


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class _Db:
    def __init__(self, root, relative):
        self.root, self.relative = root, relative

    def execute(self, sql, args=()):
        if "allowed_roots" in sql:
            return _Rows([{"path": self.root}])
        return _Rows([{"kind": "md", "relative_path": self.relative}])


class _Store:
    def __init__(self, root, relative):
        self.db = _Db(root, relative)
        self.relative = relative

    def current_identity(self):
        return {}

    def list_conspects(self, recording_id):
        return [{"root_id": "r", "conspect_id": "c", "name": "n", "current_revision": 1,
                 "source_revision": 1, "payload": {}}]

    def close(self):
        pass


class ScreenSmallFixes(unittest.TestCase):
    def test_conspect_file_path_keeps_case_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "МоиКонспекты"
            real.mkdir()
            (real / "Заметка.md").write_text("x", encoding="utf-8")
            lowered = os.path.normcase(str(real))      # так хранит корень база
            store = _Store(lowered, "Заметка.md")
            with mock.patch("recordings.open_store", return_value=store), \
                    mock.patch.object(core_screen, "_conspect_screen_text", return_value=("", [])), \
                    mock.patch.object(core_screen, "quote_repair_view", return_value=None):
                result = core_screen.recording_conspects("rec")
            self.assertEqual(result[0]["files"]["md"], str(real / "Заметка.md"))

    def test_transcript_file_paths_keep_case_from_disk(self):
        import workflow_page
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "МоиЗаписи"
            real.mkdir()
            (real / "Встреча.transcript.md").write_text("x", encoding="utf-8")
            lowered = os.path.normcase(str(real / "Встреча.transcript.md"))
            shown = workflow_page._shown_files({"markdown": lowered, "json": None})
            self.assertEqual(shown["markdown"], str(real / "Встреча.transcript.md"))
            self.assertIsNone(shown["json"])

    def test_page_has_local_time_helper_and_dark_theme(self):
        html = (ROOT / "workflow.html").read_text(encoding="utf-8")
        self.assertIn("getTimezoneOffset", html)
        self.assertIn("@media (prefers-color-scheme:dark)", html)
        style = html.split("</style>")[0]
        self.assertNotIn("#ddd;", style.split("@media")[1])   # цвета правил — через переменные
        self.assertIn("var(--bg)", style)


if __name__ == "__main__":
    unittest.main()
