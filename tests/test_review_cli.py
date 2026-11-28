"""Независимый от клиента путь карточек; только синтетические данные."""
import atexit
import os
import shutil
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch

if "RUS_TRANSCRIBER_DATA" not in os.environ:
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="review_cli_")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
import core
import rt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db_record as DB  # noqa: E402  (У-2а: запись в базе вместо несуществующего demo.json)


def demo_record(test):
    """Пути синтетической записи в базе (У-2а): режим хранения выбирает сама запись, не умолчание."""
    folder = Path(tempfile.mkdtemp(prefix="review_cli_rec_"))
    test.addCleanup(shutil.rmtree, folder, ignore_errors=True)
    jp, md, _ = DB.new_record(test, {"file": "x", "duration": "00:00:02", "speakers": [], "segments": []}, folder, "demo")
    return str(jp), str(md)


class ReviewCLI(unittest.TestCase):
    # У-2а-1: test_only_successful_answers_are_marked_applied удалён — файловый путь внесения на подменах;
    # в базе исход каждой карточки проверяет test_answer_outcomes.EveryAnswerHasOutcome.


    def test_neutral_questions(self):
        data = {"review": [{"id": "r1", "status": "open"},
                           {"id": "r2", "status": "resolved"},
                           {"id": "r3", "status": "human"}]}
        jp, _ = demo_record(self)
        with patch.object(core, "_done_json", return_value={"json": jp}), \
                patch("transcript.load", return_value=data), patch("review_pack.build", return_value={}) as build:
            core.build_review_pack("demo")
        items = build.call_args.args[1]
        self.assertEqual([x["id"] for x in items], ["r1", "r3"])
        self.assertTrue(all(x["options"][0]["decision"] == {} for x in items))

    def test_cli_dispatch(self):
        for cmd, name in [("review-build", "build_review_pack"), ("review-page", "start_review_page"),
                          ("review-apply", "apply_answers")]:
            with self.subTest(cmd=cmd), patch.object(core, name, return_value={}) as call, patch.object(rt, "out"):
                self.assertEqual(rt.main([cmd, "demo"]), 0)
                self.assertEqual(call.call_args.args[0], "demo")

    def test_missing_pack_does_not_launch(self):
        with patch.object(core, "_done_json", return_value={"json": "demo.json"}), \
                patch("review_pack.manifest", return_value=None), patch.object(core, "launch") as launch:
            with self.assertRaises(core.CoreError):
                core.start_review_page("demo")
            launch.assert_not_called()

    def test_page_returns_url(self):
        with patch.object(core, "_done_json", return_value={"json": "demo.json"}), \
                patch("review_pack.manifest", return_value={"version": "test"}), \
                patch.object(core, "launch", return_value="page"), \
                patch.object(core, "job_status", return_value={"url": "http://127.0.0.1:8765/"}):
            self.assertEqual(core.start_review_page("demo")["job_id"], "page")
