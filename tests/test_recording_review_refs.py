import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import core
import recordings
import review_pack
import settings
import storage
import transcript


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
            "review": [{"id": "r1", "status": "open", "t": 0.0, "t1": 0.0, "t2": 1.0,
                        "speaker": "Анна", "text": "раз два",
                        "word_ids": ["w1", "w2"]},
                       {"id": "r2", "status": "open", "t": 0.0, "t1": 0.0, "t2": 1.0,
                        "speaker": "Анна", "text": "раз два",
                        "word_ids": ["w2"]}],
            "questions": [{"id": "r1", "question": "Проверьте",
                           "word_ids": ["w1", "w2"]},
                          {"id": "r2", "question": "И это",
                           "word_ids": ["w2"]}],
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
        with mock.patch.object(settings.paths, "STATE_DB", self.db):
            state = recordings.read_review_pack_state(self.json_path)
            state["version"] = "synthetic-v1"
            state["manifest"] = {"items": [
                {"id": "r1", "question": "Проверьте", "word_ids": ["w1", "w2"]},
                {"id": "r2", "question": "И это", "word_ids": ["w2"]},
            ]}
            recordings.write_review_pack_state(self.json_path, state)
        before_manifest = self.store.get_current(self.recording_id)["review"]["payload"][
            "undo"]["review_pack"]["manifest"]
        before_manifest_bytes = storage._json(before_manifest).encode("utf-8")

        current = recordings.load_recording(self.store, self.recording_id)
        current["segments"][0]["words"].pop()
        current["segments"][0]["text"] = "раз"
        current["segments"][0]["end"] = 0.4
        current["review"][0]["word_ids"] = ["w1", "w2"]
        current["review"][1]["word_ids"] = ["w2"]
        current["questions"][0]["word_ids"] = ["w1", "w2"]
        current["questions"][1]["word_ids"] = ["w2"]

        with mock.patch.object(settings.paths, "STATE_DB", self.db):
            result = recordings.commit_recording_data(
                self.store, self.recording_id, current,
                {"source_revision": current["source_revision"],
                 "review_revision": current["review_revision"]},
                str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
                self.root_id, "2026-11-23T00:01:00Z")

        saved = self.store.get_current(self.recording_id)["review"]["payload"]
        self.assertEqual(result["source_revision"], 2)
        self.assertEqual(saved["review"][0]["word_ids"], ["w1"])
        self.assertNotIn("word_ids", saved["review"][1])
        self.assertEqual(saved["questions"][0]["word_ids"], ["w1"])
        self.assertNotIn("word_ids", saved["questions"][1])
        self.assertEqual(
            storage._json(saved["undo"]["review_pack"]["manifest"]).encode("utf-8"),
            before_manifest_bytes)

        no_change = recordings.load_recording(self.store, self.recording_id)
        unchanged = recordings.commit_recording_data(
            self.store, self.recording_id, no_change,
            {"source_revision": 2, "review_revision": 3},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            self.root_id, "2026-11-23T00:02:00Z")
        self.assertTrue(unchanged["unchanged"])
        self.assertEqual(self.store.get_current(self.recording_id)["transcript"]["revision"], 2)
        self.assertEqual(self.store.get_current(self.recording_id)["review"]["revision"], 3)

        replacement = recordings.load_recording(self.store, self.recording_id)
        replacement["segments"][0]["words"][0]["w"] = "иной"
        replacement["segments"][0]["text"] = "иной"
        recordings.commit_recording_data(
            self.store, self.recording_id, replacement,
            {"source_revision": 2, "review_revision": 3},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            self.root_id, "2026-11-23T00:03:00Z")
        saved = self.store.get_current(self.recording_id)["review"]["payload"]
        self.assertEqual(saved["review"][0]["word_ids"], ["w1"])
        self.assertEqual(saved["questions"][0]["word_ids"], ["w1"])

    def test_legacy_files_keeps_existing_reference_behavior(self):
        legacy_audio = self.root / "legacy.wav"
        legacy_audio.write_bytes(b"synthetic audio")
        legacy_layout = recordings.document_layout(legacy_audio)
        legacy_layout["root"].mkdir()
        json_path = legacy_layout["root"] / legacy_layout["paths"]["json"]
        markdown_path = legacy_layout["root"] / legacy_layout["paths"]["md"]
        legacy = dict(self.data)
        legacy["file"] = str(legacy_audio)
        legacy["segments"] = [{
            **self.data["segments"][0],
            "words": [self.data["segments"][0]["words"][0]],
            "text": "раз",
            "end": 0.4,
        }]
        # legacy_files принимает прежние ссылки на исчезнувшие слова; PR 2а меняет только sqlite.
        with mock.patch.object(settings, "recording_backend", return_value="legacy_files"), \
                mock.patch.object(settings.paths, "STATE_DB", self.root / "legacy-state.sqlite3"):
            transcript.save(legacy, json_path, markdown_path)
        self.assertEqual(transcript.load(json_path)["questions"][0]["word_ids"], ["w1", "w2"])

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
            state["version"] = "synthetic-v1"
            state["manifest"] = {"items": [{
                "id": "r1", "question": "Проверьте", "word_ids": ["w1", "w2"],
            }]}
            recordings.write_review_pack_state(self.json_path, state)

        saved = self.store.get_current(self.recording_id)["review"]["payload"]
        self.assertEqual(saved["questions"][0]["word_ids"], ["w1"])
        self.assertEqual(saved["undo"]["review_pack"]["manifest"]["items"][0]["word_ids"],
                         ["w1", "w2"])

    def test_edit_span_merge_keeps_pack_manifest_and_allows_next_answer(self):
        question = {"id": "r1", "question": "Проверьте", "recommended": 0, "options": [
            {"label": "Оставить", "decision": {}},
            {"label": "Другое слово", "decision": {"text": "иное"}},
        ]}
        with mock.patch.object(settings.paths, "STATE_DB", self.db), \
                mock.patch.object(review_pack, "cut_clip",
                                  side_effect=lambda _source, t1, _t2, _out: t1):
            pack = review_pack.build(self.json_path, [question])
            before_manifest = recordings.read_review_pack_state(self.json_path)["manifest"]
            before_manifest_bytes = storage._json(before_manifest).encode("utf-8")

            transcript.edit_span(self.json_path, self.markdown_path, 0.0, 1.0,
                                 "одно", by="human", resolve=False)

            review = self.store.get_current(self.recording_id)["review"]["payload"]
            self.assertNotIn("word_ids", review["questions"][0])
            self.assertEqual(storage._json(
                review["undo"]["review_pack"]["manifest"]).encode("utf-8"),
                before_manifest_bytes)

            accepted = review_pack.add_answers(
                self.json_path, [{"id": "r1", "version": pack["version"], "option": 1}],
                "local", trusted_actor="human")
            self.assertEqual(accepted["answered"], 1)
            self.assertEqual(review_pack.answers_state(self.json_path)["answers"]["r1"]["option"], 1)
            applied = core.apply_answers_files(self.json_path, self.markdown_path)
            self.assertEqual(applied["applied_from_answers"], 1, applied)
            self.assertEqual(transcript.load(self.json_path)["segments"][0]["words"][0]["w"], "иное")
            repeated, repeated_report = review_pack.decisions(self.json_path)
            self.assertEqual(repeated, [])
            self.assertEqual(repeated_report["unchanged"], ["r1"])
            review_pack.add_answers(
                self.json_path, [{"id": "r1", "version": pack["version"], "kind": "revert"}],
                "local", trusted_actor="human")
            undone = core.apply_answers_files(self.json_path, self.markdown_path)
            self.assertEqual(undone["reverted"], ["r1"])
            self.assertEqual(transcript.load(self.json_path)["segments"][0]["words"][0]["w"], "одно")


if __name__ == "__main__":
    unittest.main()
