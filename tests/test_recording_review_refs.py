import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import recordings
import review_pack
import settings
import storage


class RecordingReviewReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "state.sqlite3"
        self.audio = self.root / "synthetic.wav"
        self.audio.write_bytes(b"synthetic audio")
        self.layout = recordings.document_layout(self.audio)
        self.layout["root"].mkdir()
        self.store = recordings.initialize_store(self.db, display_name="Тест")
        self.addCleanup(self.store.close)
        self.root_id, self.targets = recordings.prepare_new_layout(
            self.store, self.layout["root"], self.layout["paths"])
        self.recording_id = str(uuid.uuid4())
        self.data = {
            "file": str(self.audio),
            "duration": "00:00:02",
            "speakers": ["Анна"],
            "segments": [{
                "start": 0.0, "end": 1.0, "speaker": "Анна", "text": "раз два",
                "words": [
                    {"word_id": "w1", "w": "раз", "s": 0.0, "e": 0.4},
                    {"word_id": "w2", "w": "два", "s": 0.5, "e": 1.0},
                ],
            }],
            "review": [{"id": "r1", "status": "open", "t1": 0.0, "t2": 1.0,
                        "word_ids": ["w1", "w2"]}],
            "questions": [{"id": "r1", "question": "Проверьте",
                           "word_ids": ["w1", "w2"]}],
            "answers": {},
        }
        recordings.commit_worker_result(
            self.store, self.data, self.audio, self.root_id,
            recordings.operation_context(self.store, "model"),
            recording_id=self.recording_id, operation_id=str(uuid.uuid4()),
            generated_at="2026-11-23T00:00:00Z", document_paths=self.targets)
        recordings.materialize_recording(self.store, self.recording_id)
        self.json_path = self.layout["root"] / self.targets["json"]
        self.markdown_path = self.layout["root"] / self.targets["md"]

    def test_commit_recording_data_removes_only_disappeared_references(self):
        current = recordings.load_recording(self.store, self.recording_id)
        current["segments"][0]["words"].pop()
        current["segments"][0]["text"] = "раз"
        current["segments"][0]["end"] = 0.4
        current["review"][0]["word_ids"] = ["w1", "w2"]
        current["questions"][0]["word_ids"] = ["w1", "w2"]

        with mock.patch.object(settings.paths, "STATE_DB", self.db):
            result = recordings.commit_recording_data(
                self.store, self.recording_id, current,
                {"source_revision": 1, "review_revision": 1},
                str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
                self.root_id, "2026-11-23T00:01:00Z")

        saved = self.store.get_current(self.recording_id)["review"]["payload"]
        self.assertEqual(result["source_revision"], 2)
        self.assertEqual(saved["review"][0]["word_ids"], ["w1"])
        self.assertEqual(saved["questions"][0]["word_ids"], ["w1"])

    def test_write_review_pack_state_removes_disappeared_question_references(self):
        current = recordings.load_recording(self.store, self.recording_id)
        current["segments"][0]["words"].pop()
        current["segments"][0]["text"] = "раз"
        current["segments"][0]["end"] = 0.4
        current["review"][0]["word_ids"] = ["w1"]
        current["questions"][0]["word_ids"] = ["w1"]
        with mock.patch.object(settings.paths, "STATE_DB", self.db):
            recordings.commit_recording_data(
                self.store, self.recording_id, current,
                {"source_revision": 1, "review_revision": 1},
                str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
                self.root_id, "2026-11-23T00:01:00Z")

            state = recordings.read_review_pack_state(self.json_path)
            state["manifest"] = {"items": [{
                "id": "r1", "question": "Проверьте", "word_ids": ["w1", "w2"],
            }]}
            recordings.write_review_pack_state(self.json_path, state)

        saved = self.store.get_current(self.recording_id)["review"]["payload"]
        self.assertEqual(saved["questions"][0]["word_ids"], ["w1"])
        self.assertEqual(saved["undo"]["review_pack"]["manifest"]["items"][0]["word_ids"], ["w1"])


if __name__ == "__main__":
    unittest.main()
