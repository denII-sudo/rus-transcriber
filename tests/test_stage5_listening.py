"""Явное прослушивание проблемного интервала не является автоматическим решением."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import review_pack
import transcript
import core


class ListeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.jp, self.md, audio = root / "test.transcript.json", root / "test.transcript.md", root / "test.wav"
        audio.write_bytes(b"synthetic")
        self.data = {"file": str(audio), "duration": "00:00:02", "speakers": [],
                     "segments": [], "quality": {"missed_speech_diagnostics": {
                         "schema_version": 1, "algorithm": "diarization-turn-uncovered-gaps",
                         "threshold_seconds": 1.0,
                         "speech_intervals": [{"id": "turn-0001", "start": 0.0, "end": 1.2}],
                         "candidates": [{"id": "missed-0001", "turn_id": "turn-0001",
                                         "start": 0.0, "end": 1.0, "duration_seconds": 1.0,
                                         "threshold_seconds": 1.0, "meets_threshold": True,
                                         "left_boundary": "turn_start", "right_boundary": "word_start",
                                         "decision": "warning"}]}},
                     "review": [{"id": "r1", "t": 0.0, "t1": 0.0, "t2": 1.0,
                                  "gap": [0.0, 1.0], "speaker": None, "text": "", "reasons": ["asr"],
                                  "diagnostic_id": "missed-0001",
                                  "asr_problem": {"model": "whisper", "state": "empty_unknown", "words_from": []}}]}
        transcript.save_new(self.data, self.jp, self.md)

    def test_only_explicit_human_silence_confirmation(self):
        with patch.object(transcript, "_refresh_conspects", return_value=None):
            data, rep = transcript.apply_review(self.jp, self.md, [{"id": "r1", "listened": "silence"}], by="claude")
            self.assertNotIn("listening", data["review"][0])
            self.assertTrue(rep["conflicts"])
            data, rep = transcript.apply_review(self.jp, self.md, [{"id": "r1", "listened": "silence"}], by="human")
        self.assertEqual(data["review"][0]["listening"]["result"], "silence")
        self.assertEqual(data["review"][0]["listening"]["by"], "human")

    def test_changed_words_invalidate_silence(self):
        with patch.object(transcript, "_refresh_conspects", return_value=None):
            data, _ = transcript.apply_review(self.jp, self.md, [{"id": "r1", "listened": "silence"}], by="human")
        data["segments"] = [{"speaker": "S", "start": 0.1, "end": 0.2, "text": "да",
                             "words": [{"w": "да", "s": 0.1, "e": 0.2}]}]
        transcript.save(data, self.jp, self.md)
        item = transcript.load(self.jp)["review"][0]
        self.assertEqual(item["status"], "open")
        self.assertNotIn("listening", item)

    def test_silence_cannot_override_existing_words(self):
        self.data["segments"] = [{"speaker": "S", "start": 0.1, "end": 0.2, "text": "да",
                                  "words": [{"w": "да", "s": 0.1, "e": 0.2}]}]
        transcript.save(self.data, self.jp, self.md)
        with patch.object(transcript, "_refresh_conspects", return_value=None):
            data, report = transcript.apply_review(self.jp, self.md, [{"id": "r1", "listened": "silence"}], by="human")
        self.assertTrue(report["conflicts"])
        self.assertNotIn("listening", data["review"][0])

    def test_silence_cannot_override_word_crossing_boundary(self):
        self.data["review"][0].update(t=0.5, t1=0.5, gap=[0.5, 1.0])
        self.data["segments"] = [{"speaker": "S", "start": 0.4, "end": 0.6, "text": "да",
                                  "words": [{"w": "да", "s": 0.4, "e": 0.6}]}]
        transcript.save(self.data, self.jp, self.md)
        with patch.object(transcript, "_refresh_conspects", return_value=None):
            data, report = transcript.apply_review(self.jp, self.md, [{"id": "r1", "listened": "silence"}], by="human")
        self.assertTrue(report["conflicts"])
        self.assertNotIn("listening", data["review"][0])

    def test_card_options_are_explicit_and_stale_pack_is_rejected(self):
        with patch.object(review_pack, "cut_clip", return_value=0):
            review_pack.build(self.jp, [{"id": "r1", "options": [{"label": "OK", "decision": {}}]}])
        man = review_pack.manifest(self.jp)
        self.assertEqual(man["items"][0]["options"][1]["decision"], {"listened": "silence"})
        self.assertEqual(man["items"][0]["diagnostic_id"], "missed-0001")
        self.assertEqual(man["items"][0]["automation_diagnostic"]["candidate"]["decision"], "warning")
        # played — место прослушано на странице (раздел 10)
        review_pack.add_answers(self.jp, [{"id": "r1", "version": man["version"], "option": 1, "played": True}],
                                "local", trusted_actor="human")
        stored = review_pack._read(review_pack.pack_dir(self.jp) / "answers.json", {})
        self.assertEqual(stored["answers"]["r1"]["diagnostic_id"], "missed-0001")
        # К-5: сохранение без смены слов интервала кнопку не отклоняет, смена слов интервала — отклоняет
        transcript.save(transcript.load(self.jp), self.jp, self.md)
        decisions, report = review_pack.decisions(self.jp)
        self.assertEqual([d["id"] for d in decisions], ["r1"])
        self.assertNotIn("source_conflict", report)
        data = transcript.load(self.jp)
        data["segments"] = [{"speaker": "S", "start": 0.1, "end": 0.2, "text": "да",
                             "words": [{"w": "да", "s": 0.1, "e": 0.2}]}]
        transcript.save(data, self.jp, self.md)
        decisions, report = review_pack.decisions(self.jp)
        self.assertEqual(decisions, [])
        self.assertIn("source_conflict", report)

    def test_card_answer_applies_and_can_be_reverted(self):
        with patch.object(review_pack, "cut_clip", return_value=0):
            review_pack.build(self.jp, [{"id": "r1", "options": [{"label": "OK", "decision": {}}]}])
        man = review_pack.manifest(self.jp)
        review_pack.add_answers(self.jp, [{"id": "r1", "version": man["version"], "option": 1, "played": True}],
                                "local", trusted_actor="human")
        with patch.object(transcript, "_refresh_conspects", return_value=None):
            core.apply_answers_files(self.jp, self.md)
            self.assertEqual(transcript.load(self.jp)["review"][0]["listening"]["result"], "silence")
            stored = review_pack._read(review_pack.pack_dir(self.jp) / "answers.json", {})
            self.assertEqual(stored["application_events"][-1]["diagnostic_id"], "missed-0001")
            self.assertEqual(stored["application_events"][-1]["action"], "applied")
            review_pack.add_answers(self.jp, [{"id": "r1", "version": man["version"], "kind": "revert"}],
                                    "local", trusted_actor="human")
            core.apply_answers_files(self.jp, self.md)
        item = transcript.load(self.jp)["review"][0]
        self.assertEqual(item["status"], "open")
        self.assertNotIn("listening", item)
        stored = review_pack._read(review_pack.pack_dir(self.jp) / "answers.json", {})
        self.assertEqual(stored["application_events"][-1]["action"], "reverted")
        self.assertEqual(stored["application_events"][-1]["diagnostic_id"], "missed-0001")


if __name__ == "__main__":
    unittest.main()
