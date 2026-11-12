"""Независимый от клиента путь карточек; только синтетические данные."""
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("RUS_TRANSCRIBER_DATA", tempfile.mkdtemp(prefix="review_cli_"))
import core
import rt


class ReviewCLI(unittest.TestCase):
    def test_only_successful_answers_are_marked_applied(self):
        with patch.object(core, "job_json", return_value=("demo.json", "demo.md")), \
                patch("review_pack.decisions", return_value=([{"id": "r1"}, {"id": "r2"}], {"human": []})), \
                patch("transcript.apply_review", return_value=(None, {"successful_ids": ["r1"],
                                                                      "not_found": ["r2"]})), \
                patch("review_pack.free_answers", return_value={}), \
                patch("review_pack.mark_applied") as mark:
            core.apply_answers("demo")
        mark.assert_called_once_with("demo.json", {"r1"})

    def test_neutral_questions(self):
        data = {"review": [{"id": "r1", "status": "open"},
                           {"id": "r2", "status": "resolved"},
                           {"id": "r3", "status": "human"}]}
        with patch.object(core, "_done_json", return_value={"json": "demo.json"}), \
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
