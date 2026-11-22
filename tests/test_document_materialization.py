"""Проверки подготовки, авторства и архива документов П1."""
import hashlib
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

import documents
import storage


def uid():
    return str(uuid.uuid4())


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p1b-", dir=os.environ["TEMP"])
        self.store = storage.initialize(Path(self.temp.name) / "state.sqlite3")
        self.user = self.store.create_user("<ник>")
        self.installation = uid()
        self.store.bind_installation(self.installation, self.user)
        self.ctx = self.store.capture_context(self.installation, "human")
        self.recording = uid()
        self.source = {"segments": [{"segment_id": "s", "speaker": "<голос>",
                                     "words": [{"word_id": "w", "w": "привет"}]}]}
        self.review = {"review": [{"id": "r", "status": "open", "word_ids": ["w"]}],
                       "answers": {}}
        self.store.create_recording(self.recording, "tag1", self.source, self.review, uid(), self.ctx)
        self.versions = {"source_revision": 1, "review_revision": 1}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_bytes_manifest_and_historical_name(self):
        op = uid()
        res = self.store.save_document(self.recording, self.versions, "2026-01-01Z", op, self.ctx)
        old = self.store.get_document(self.recording, res["document_revision"])
        manifest = json.loads(old["manifest"])
        self.assertEqual(len(manifest["files"]), 2)
        self.assertNotIn("manifest", manifest["files"])
        for field, suffix in (("md", ".md"), ("json", ".json")):
            entry = next(v for k, v in manifest["files"].items() if k.endswith(suffix))
            self.assertEqual(entry["sha256"], hashlib.sha256(old[field]).hexdigest())
            self.assertEqual(entry["size"], len(old[field]))
        self.assertIn("&lt;ник&gt;", old["md"].decode())
        self.assertIn("&lt;голос&gt;", old["md"].decode())
        self.assertIn("не обязательно", old["md"].decode())
        self.store.rename_user(self.user, "новое")
        self.assertEqual(res, self.store.save_document(self.recording, self.versions, "2026-01-01Z",
                                                        op, self.ctx))
        self.assertEqual(old["md"], self.store.get_document(self.recording, 1)["md"])
        new = self.store.save_document(self.recording, self.versions, "2026-01-02Z", uid(),
                                       self.store.capture_context(self.installation, "human"))
        self.assertIn("&lt;ник&gt;", self.store.get_document(self.recording, new["document_revision"])
                      ["md"].decode())

    def test_restore_old_tuple_preserves_payload_and_new_action(self):
        self.store.save_document(self.recording, self.versions, "old", uid(), self.ctx)
        changed = {"segments": [{"segment_id": "s", "speaker": None,
                                  "words": [{"word_id": "w", "w": "исправлено"}]}]}
        self.store.commit_change(self.recording, self.versions, uid(), self.ctx,
                                 transcript=changed, review=self.review)
        new_user = self.store.create_user("редактор")
        self.store.bind_installation(self.installation, new_user)
        ctx = self.store.capture_context(self.installation, "human")
        op = uid()
        result = self.store.restore_as_new_revision(self.recording, 1, 1,
                                                    {"source_revision": 2, "review_revision": 2},
                                                    op, ctx)
        self.assertEqual(result, self.store.restore_as_new_revision(
            self.recording, 1, 1, {"source_revision": 2, "review_revision": 2}, op, ctx))
        self.assertEqual(result["source_revision"], 3)
        source = self.store.get_revision(self.recording, "transcript", 3)
        self.assertEqual(source["payload"], self.source)
        self.assertEqual(source["restored_from"], 1)
        self.assertEqual(source["context"]["author_ref"]["user_id"], new_user)
        self.assertEqual(self.store.get_document(self.recording, 1)["source_revision"], 1)
        self.store.save_document(self.recording, {"source_revision": 3, "review_revision": 3},
                                 "new", uid(), ctx)
        current = json.loads(self.store.get_document(self.recording, 2)["json"])
        self.assertEqual(current["passport"]["source_revision"], 3)
        self.assertEqual(current["source_action"]["author_ref"]["user_id"], new_user)
        self.assertEqual(current["source_content_action"]["author_ref"]["user_id"], self.user)

    def test_invalid_reference_and_stale_document(self):
        bad = {"review": [{"id": "r", "word_ids": ["missing"]}], "answers": {}}
        with self.assertRaises(ValueError):
            self.store.commit_change(self.recording, self.versions, uid(), self.ctx, review=bad)
        self.store.commit_change(self.recording, self.versions, uid(), self.ctx, review=self.review)
        with self.assertRaises(storage.Conflict):
            self.store.save_document(self.recording, self.versions, "now", uid(), self.ctx)

    def test_paths_are_per_recording_and_not_derived_from_name(self):
        a = documents.prepare(self.recording, 1, 1, 1, self.source, self.review,
                              self.ctx, self.ctx, "now")
        b = documents.prepare(uid(), 1, 1, 1, self.source, self.review,
                              self.ctx, self.ctx, "now")
        self.assertFalse(set(a["paths"].values()) & set(b["paths"].values()))
        self.assertNotIn("ник", str(a["paths"]))

    def test_archived_byte_corruption_is_detected(self):
        self.store.save_document(self.recording, self.versions, "now", uid(), self.ctx)
        self.store.db.execute("UPDATE document_versions SET md=? WHERE recording_id=?", (b"bad", self.recording))
        with self.assertRaises(storage.StorageError):
            self.store.get_document(self.recording, 1)

    def test_annotation_origin_staleness_and_immutable_bytes(self):
        model = self.store.capture_context(self.installation, "model")
        annotation = {"text": "<заметка>", "source_revision": 1, "context": model}
        first_op = uid()
        self.store.save_document(self.recording, self.versions, "now", first_op, self.ctx,
                                 annotation=annotation)
        first = self.store.get_document(self.recording, 1)
        payload = json.loads(first["json"])
        self.assertIsNone(payload["annotation"]["context"]["author_ref"]["user_id"])
        self.assertEqual(payload["annotation"]["context"]["initiated_by"]["user_id"], self.user)
        self.assertIn("&lt;заметка&gt;", first["md"].decode())
        self.store.commit_change(self.recording, self.versions, uid(), self.ctx,
                                 transcript=self.source, review=self.review)
        self.store.save_document(self.recording, {"source_revision": 2, "review_revision": 2},
                                 "later", uid(), self.ctx, annotation=annotation)
        second = self.store.get_document(self.recording, 2)
        self.assertIn("опорная ревизия отличается", second["md"].decode())
        self.assertEqual(first["md"], self.store.get_document(self.recording, 1)["md"])
        different = dict(annotation, text="другая")
        with self.assertRaises(storage.Conflict):
            self.store.save_document(self.recording, self.versions, "now", first_op, self.ctx,
                                     annotation=different)

    def test_annotation_must_reference_existing_revision(self):
        model = self.store.capture_context(self.installation, "model")
        annotation = {"text": "текст", "source_revision": 99, "context": model}
        with self.assertRaises(KeyError):
            self.store.save_document(self.recording, self.versions, "now", uid(), self.ctx,
                                     annotation=annotation)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM document_versions").fetchone()[0], 0)

    def test_open_json_does_not_dump_internal_diagnostics_or_answers(self):
        source = dict(self.source, quality={"internal_marker": "not-for-export"})
        review = {"review": [{"id": "r", "word_ids": ["w"],
                               "internal_marker": "not-for-export"}],
                  "answers": {"r": {"audio": "not-for-export"}}}
        prepared = documents.prepare(self.recording, 1, 1, 1, source, review,
                                     self.ctx, self.ctx, "now")
        self.assertNotIn(b"not-for-export", prepared["json"])
        self.assertIn("полная история здесь не представлены", prepared["md"].decode())
        source["segments"] = [{"segment_id": "s", "speaker": {"secret": "not-for-export"}, "words": []}]
        with self.assertRaises(ValueError):
            documents.prepare(self.recording, 1, 1, 1, source, {"review": [], "answers": {}},
                              self.ctx, self.ctx, "now")

    def test_word_answer_and_action_attribution_remain_separate(self):
        responder = self.ctx["author_ref"]
        source = {"segments": [
            {"segment_id": "s", "speaker": "А", "words": [
                {"word_id": "w", "w": "слово", "provenance": {
                    "text": {"by": "human", "author_ref": responder}}}]},
            {"segment_id": "s2", "speaker": None, "words": [
                {"word_id": "w2", "w": "ещё"}]}]}
        review = {"review": [{"id": "r", "status": "open", "word_ids": ["w"]}],
                  "answers": {"r": {"text": "да", "actor_kind": "human", "author_ref": responder}}}
        second = self.store.create_user("Применивший")
        self.store.bind_installation(self.installation, second)
        applier = self.store.capture_context(self.installation, "human")
        review["applications"] = [{"id": "apply-1", "kind": "apply", "answer_id": "r",
                                   "context": applier}]
        self.store.commit_change(self.recording, self.versions, uid(), applier,
                                 transcript=source, review=review)
        self.store.save_document(self.recording, {"source_revision": 2, "review_revision": 2},
                                 "now", uid(), applier)
        document = self.store.get_document(self.recording, 1)
        public = json.loads(document["json"])
        self.assertEqual(public["transcript"]["segments"][0]["words"][0]["provenance"]
                         ["text"]["author_ref"]["user_id"], self.user)
        self.assertEqual(public["review"]["answers"]["r"]["author_ref"]["user_id"], self.user)
        self.assertEqual(public["review"]["applications"][0]["context"]["author_ref"]["user_id"], second)
        self.assertEqual(public["source_action"]["author_ref"]["user_id"], second)
        self.assertIn("&lt;ник&gt;", document["md"].decode())
        self.assertIn("Применивший", document["md"].decode())
        self.assertEqual(len(public["statistics"]["speakers"]), 2)
        names = json.loads(document["manifest"])["mentioned_authors"]
        self.assertEqual({row["user_id"] for row in names}, {self.user, second})

    def test_archive_manifest_and_conspect_old_link(self):
        self.store.save_document(self.recording, self.versions, "first", uid(), self.ctx)
        old = self.store.get_document(self.recording, 1)
        self.store.commit_change(self.recording, self.versions, uid(), self.ctx,
                                 transcript=self.source, review=self.review)
        self.store.save_document(self.recording, {"source_revision": 2, "review_revision": 2},
                                 "second", uid(), self.ctx)
        paths = documents.archive_paths(self.recording, 1)
        root = Path(self.temp.name) / "result"
        for kind in ("md", "json", "manifest"):
            target = root / paths[kind]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(old["archive_manifest"] if kind == "manifest" else old[kind])
        conspect = documents.prepare_conspect("Конспект", "Содержание", self.recording, 1)
        self.assertIn(paths["md"].encode(), conspect)
        self.assertEqual((root / paths["md"]).read_bytes(), old["md"])
        manifest = json.loads((root / paths["manifest"]).read_bytes())
        for filename, entry in manifest["files"].items():
            content = ((root / paths["manifest"]).parent / filename).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])

    def test_legacy_human_does_not_acquire_current_user(self):
        source = {"segments": [{"segment_id": "s", "speaker": None, "words": [
            {"word_id": "w", "w": "слово", "provenance": {"text": {"by": "human",
                                                                      "legacy_author_label": "ник"}}}]}]}
        result = documents.prepare(self.recording, 1, 1, 1, source, self.review,
                                   self.ctx, self.ctx, "now")
        entry = json.loads(result["json"])["transcript"]["segments"][0]["words"][0]["provenance"]["text"]
        self.assertEqual(entry["by"], "human")
        self.assertEqual(entry["legacy_author_label"], "ник")
        self.assertNotIn("author_ref", entry)

    def test_unknown_author_is_not_replaced_by_current_user(self):
        unknown = {"user_id": None, "user_name_snapshot": None}
        source = {"segments": [{"segment_id": "s", "speaker": None, "words": [
            {"word_id": "w", "w": "слово", "provenance": {"text": {"author_ref": unknown}}}]}]}
        doc = documents.prepare(self.recording, 1, 1, 1, source, self.review, self.ctx, self.ctx, "now")
        author = json.loads(doc["json"])["transcript"]["segments"][0]["words"][0]["provenance"]["text"]["author_ref"]
        self.assertEqual(author, unknown)

    def test_model_answer_cannot_be_given_human_author(self):
        review = {"review": [{"id": "r", "word_ids": ["w"]}],
                  "answers": {"r": {"actor_kind": "model", "author_ref": self.ctx["author_ref"]}}}
        with self.assertRaises(ValueError):
            documents.prepare(self.recording, 1, 1, 1, self.source, review,
                              self.ctx, self.ctx, "now")

    def test_incomplete_asr_and_missed_speech_remain_visible(self):
        source = dict(self.source, quality={"asr_incomplete_chunks": 2, "vote3_skipped": True})
        review = {"review": [{"id": "r", "status": "open", "word_ids": ["w"],
                               "gap": [1.0, 3.0]}], "answers": {}}
        prepared = documents.prepare(self.recording, 1, 1, 1, source, review,
                                     self.ctx, self.ctx, "now")
        md = prepared["md"].decode()
        self.assertIn("не распознано фрагментов: 2", md)
        self.assertIn("голосование трёх моделей было пропущено", md)
        self.assertIn("Возможный пропуск речи", md)
        self.assertEqual(json.loads(prepared["json"])["review"]["review"][0]["gap"], [1.0, 3.0])

    def test_partial_vote_skip_remains_visible(self):
        partial = {"reason": "whisper_incomplete", "spans": 2, "seconds": 3.4}
        source = dict(self.source, quality={"vote3_partial": partial})
        review = {"review": [], "answers": {}}
        prepared = documents.prepare(self.recording, 1, 1, 1, source, review, self.ctx, self.ctx, "now")
        self.assertIn("пропущено на отрезках с неполным результатом распознавания: 2 (около 3,4 с)",
                      prepared["md"].decode())
        self.assertEqual(json.loads(prepared["json"])["transcript"]["quality"], {"vote3_partial": partial})
        for bad in ("yes", {"reason": "other", "spans": 1, "seconds": 1.0}, {"reason": "asr_incomplete"},
                    {"reason": "asr_incomplete", "spans": "1", "seconds": 1.0},
                    {"reason": "asr_incomplete", "spans": 1, "seconds": -1.0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                documents.prepare(self.recording, 1, 1, 1, dict(self.source, quality={"vote3_partial": bad}),
                                  review, self.ctx, self.ctx, "now")

    def test_order_speaker_names_and_undo_projection(self):
        source = {"speakers": ["s1", "s2"], "speaker_names": {"s1": "Анна", "s2": "Борис"},
                  "names_source": "voices", "names_confirmed": False, "title": "Тест",
                  "duration": "00:03:20", "recorded_at": "2026-01-01", "codec": "synthetic",
                  "segments": [{"segment_id": "s", "speaker": "s1", "words": [
                      {"word_id": "w", "w": "да", "provenance": {"order": {"by": "human", "edit": "e"}}}]}]}
        review = {"review": [{"id": "r", "word_ids": ["w"]}], "answers": {},
                  "undo": {"r": {"source_revision": 1, "context": self.ctx}},
                  "questions": [{"id": "r", "question": "Как?", "word_ids": ["w"]}]}
        doc = documents.prepare(self.recording, 1, 1, 1, source, review, self.ctx, self.ctx, "now")
        public = json.loads(doc["json"])
        self.assertEqual(public["transcript"]["speaker_names"]["s1"], "Анна")
        self.assertEqual(public["passport"]["source"]["duration"], "00:03:20")
        self.assertEqual(public["transcript"]["segments"][0]["words"][0]["provenance"]["order"]["edit"], "e")
        self.assertEqual(public["review"]["undo"]["r"]["context"], self.ctx)
        self.assertEqual(public["review"]["questions"][0]["question"], "Как?")
        self.assertIn("Анна", doc["md"].decode())
        self.assertIn("не подтверждены", doc["md"].decode())

    def test_undo_export_is_explicitly_bounded(self):
        review = {"review": [{"id": "r", "word_ids": ["w"]}], "answers": {},
                  "undo": {"r": {"source_revision": 1, "review_revision": 1, "answer_id": "r",
                                 "kind": "apply", "context": self.ctx,
                                 "secret": "not-for-export",
                                 "snapshot": {"secret": "not-for-export"}}}}
        doc = documents.prepare(self.recording, 1, 1, 1, self.source, review,
                                self.ctx, self.ctx, "now")
        public = json.loads(doc["json"])["review"]["undo"]["r"]
        self.assertNotIn("secret", public)
        self.assertNotIn("snapshot", public)
        self.assertEqual(set(public), {"source_revision", "review_revision", "answer_id",
                                       "kind", "context"})
        self.assertNotIn(b"not-for-export", doc["json"])

    def test_annotation_inherited_or_explicitly_removed(self):
        annotation = {"text": "текст", "source_revision": 1, "context": self.ctx}
        self.store.save_document(self.recording, self.versions, "one", uid(), self.ctx, annotation=annotation)
        self.store.save_document(self.recording, self.versions, "two", uid(), self.ctx)
        self.assertEqual(json.loads(self.store.get_document(self.recording, 2)["json"])["annotation"], annotation)
        self.store.save_document(self.recording, self.versions, "three", uid(), self.ctx, annotation=None)
        self.assertIsNone(json.loads(self.store.get_document(self.recording, 3)["json"])["annotation"])


if __name__ == "__main__":
    unittest.main()
