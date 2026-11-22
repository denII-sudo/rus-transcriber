"""Неполнота ASR сохраняется в пользовательских статусах CLI и MCP."""
import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core
import migrate
import rt


class StatusTests(unittest.TestCase):
    def test_compact_transcript_keeps_warning_without_diagnostic_details(self):
        data = {"file": "synthetic.wav", "duration": "00:00:02", "speakers": [], "segments": [],
                "quality": {"asr_incomplete_chunks": 2, "vote3_skipped": "asr_incomplete",
                            "asr_diagnostics": {"large": "hidden"}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            job = {"json": str(path)}
            compact = json.loads(core._text_only(job, "json"))
            self.assertEqual(compact["asr_summary"]["asr_incomplete_chunks"], 2)
            self.assertNotIn("quality", compact)
            md = core._text_only(job, "markdown")
            self.assertIn("не распознано фрагментов: 2", md)
            self.assertIn("Голосование трёх моделей отключено", md)

    def test_compact_transcript_shows_partial_vote_skip(self):
        partial = {"reason": "asr_incomplete", "spans": 1, "seconds": 0.7}
        data = {"file": "synthetic.wav", "duration": "00:00:02", "speakers": [], "segments": [],
                "quality": {"asr_incomplete_chunks": 1, "vote3_partial": partial}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            job = {"json": str(path)}
            compact = json.loads(core._text_only(job, "json"))
            self.assertEqual(compact["asr_summary"]["vote3_partial"], partial)
            md = core._text_only(job, "markdown")
            self.assertIn("пропущено на отрезках с неполным результатом ASR: 1 (около 0,7 с)", md)
            self.assertNotIn("Голосование трёх моделей отключено", md)

    def test_cli_and_mcp_show_incomplete_result(self):
        status = {"state": "done", "quality": {"asr_incomplete_chunks": 2,
                                                 "vote3_skipped": "whisper_incomplete"}}
        with patch.object(core, "job_status", return_value=status):
            self.assertEqual(core.status("synthetic")["quality"], status["quality"])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rt.run(rt.build().parse_args(["status", "synthetic"]))
            self.assertEqual(json.loads(out.getvalue())["quality"], status["quality"])
            with patch.object(migrate, "run", return_value={"status": "test"}):
                import server
            result = asyncio.run(server.transcription_status("synthetic", 0))
            self.assertEqual(json.loads(result)["quality"], status["quality"])


if __name__ == "__main__":
    unittest.main()
