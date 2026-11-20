import hashlib
import json
import contextlib
import io
import sqlite3
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import documents
import storage
from tools import verify_document_files


def uid():
    return str(uuid.uuid4())


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p1c-", dir=os.environ["TEMP"])
        self.base = Path(self.temp.name)
        self.root = self.base / "documents"
        self.root.mkdir()
        self.path = self.base / "state.sqlite3"
        self.store = storage.initialize(self.path)
        user = self.store.create_user("ник")
        installation = uid()
        self.store.bind_installation(installation, user)
        self.ctx = self.store.capture_context(installation, "human")
        self.root_id = self.store.register_root(self.root)
        self.rid = uid()
        self.transcript = {"segments": [{"segment_id": "s", "speaker": "А",
                                         "words": [{"word_id": "w", "w": "один"}]}]}
        self.review = {"review": [{"id": "r", "word_ids": ["w"]}], "answers": {}}
        self.op = uid()

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def create(self):
        return self.store.create_recording_with_document(
            self.rid, "tag", self.transcript, self.review, self.op, self.ctx,
            root_id=self.root_id, generated_at="one")

    def change(self, expected, value, stamp):
        transcript = {"segments": [{"segment_id": "s", "speaker": "А",
                                     "words": [{"word_id": "w", "w": value}]}]}
        return self.store.commit_change_with_document(
            self.rid, expected, uid(), self.ctx, transcript=transcript, review=self.review,
            root_id=self.root_id, generated_at=stamp)

    def test_commit_enqueue_and_retry(self):
        created = self.create()
        self.assertEqual(created["document_sync"], "pending")
        self.assertEqual(documents.materialize_pending(self.store, self.root_id, self.root), [(self.rid, 1)])
        self.assertEqual(self.create()["document_sync"], "synced")
        doc = self.store.get_document(self.rid, 1)
        self.assertEqual((self.root / (self.rid + ".transcript.md")).read_bytes(), doc["md"])
        self.assertEqual(json.loads((self.root / (self.rid + ".manifest.json")).read_bytes())
                         ["passport"]["document_revision"], 1)

    def test_materialized_documents_are_readable_without_database(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        document = self.store.get_document(self.rid, 1)
        current = documents.current_paths(self.rid)
        archived = documents.archive_paths(self.rid, 1)
        expected = {"md": document["md"], "json": document["json"]}

        self.store.close()
        unavailable = self.path.with_suffix(".unavailable")
        self.path.rename(unavailable)
        try:
            with self.assertRaises(storage.StorageError):
                storage.open_store(self.path)
            for kind in ("md", "json"):
                self.assertEqual((self.root / current[kind]).read_bytes(), expected[kind])
            for kind, relative in archived.items():
                expected_archive = document["archive_manifest"] if kind == "manifest" else document[kind]
                self.assertEqual((self.root / relative).read_bytes(), expected_archive)
            manifest = json.loads((self.root / current["manifest"]).read_bytes())
            for filename, entry in manifest["files"].items():
                content = (self.root / filename).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
            self.assertEqual(manifest["passport"]["recording_id"], self.rid)
        finally:
            unavailable.rename(self.path)
            self.store = storage.open_store(self.path)

    def test_two_recordings_have_independent_files_and_manifests(self):
        self.create()
        second = uid()
        self.store.create_recording_with_document(
            second, "tag", self.transcript, self.review, uid(), self.ctx,
            root_id=self.root_id, generated_at="one")
        documents.materialize_pending(self.store, self.root_id, self.root)
        first_manifest = json.loads((self.root / (self.rid + ".manifest.json")).read_bytes())
        second_manifest = json.loads((self.root / (second + ".manifest.json")).read_bytes())
        self.assertEqual(first_manifest["passport"]["recording_id"], self.rid)
        self.assertEqual(second_manifest["passport"]["recording_id"], second)
        self.assertFalse(set(first_manifest["files"]) & set(second_manifest["files"]))
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM file_locations").fetchone()[0], 12)
        locations = self.store.db.execute("SELECT confirmed_revision,confirmed_hash,generation "
                                          "FROM file_locations").fetchall()
        self.assertTrue(all(r["confirmed_revision"] == 1 and r["confirmed_hash"] and r["generation"] == 1
                            for r in locations))

    def test_create_with_order_and_names_is_materialized(self):
        self.transcript["speaker_names"] = {"А": "Имя"}
        self.transcript["segments"][0]["words"][0]["provenance"] = {"order": {"by": "human", "edit": "x"}}
        self.transcript["segments"][0]["words"].append(
            {"word_id": "inserted", "w": "вставлено", "provenance": {"order": {"by": "human", "edit": "y"}}})
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        doc = json.loads((self.root / (self.rid + ".transcript.json")).read_bytes())
        self.assertEqual(doc["transcript"]["segments"][0]["words"][0]["provenance"]["order"]["edit"], "x")
        self.assertEqual(doc["transcript"]["segments"][0]["words"][1]["provenance"]["order"]["edit"], "y")
        self.assertIn("Имя", (self.root / (self.rid + ".transcript.md")).read_text(encoding="utf-8"))

    def test_synced_missing_file_and_external_edit_are_reconciled(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        target = self.root / (self.rid + ".transcript.md")
        target.unlink()
        documents.materialize_pending(self.store, self.root_id, self.root)
        self.assertEqual(target.read_bytes(), self.store.get_document(self.rid, 1)["md"])
        event = self.store.db.execute("SELECT event_type,kind FROM materialization_events").fetchone()
        self.assertEqual((event["event_type"], event["kind"]), ("missing_restored", "md"))
        target.write_bytes(b"edited")
        documents.materialize_pending(self.store, self.root_id, self.root)
        self.assertEqual(target.read_bytes(), b"edited")
        self.assertEqual(self.store.db.execute("SELECT state FROM document_outbox").fetchone()[0], "conflict")
        self.assertEqual(self.store.db.execute("SELECT observed_bytes FROM document_conflicts").fetchone()[0], b"edited")

    def test_large_conflict_stays_external_and_is_not_resolvable_automatically(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        target = self.root / (self.rid + ".transcript.md")
        target.write_bytes(b"x" * (documents.MAX_CONFLICT_BYTES + 1))
        documents.materialize_pending(self.store, self.root_id, self.root)
        conflict = self.store.db.execute("SELECT * FROM document_conflicts").fetchone()
        self.assertIsNone(conflict["observed_bytes"])
        self.assertEqual(conflict["observed_size"], documents.MAX_CONFLICT_BYTES + 1)
        with self.assertRaises(storage.Conflict):
            self.store.resolve_document_conflict(conflict["conflict_id"], uid(), self.ctx)
        self.assertEqual(target.stat().st_size, documents.MAX_CONFLICT_BYTES + 1)

    def test_player_event_does_not_create_document_or_source_revision(self):
        self.create()
        expected = {"source_revision": 1, "review_revision": 1}
        op = uid()
        event = self.store.commit_player_event(self.rid, expected, op, self.ctx, "audio_ended")
        self.assertEqual(event, self.store.commit_player_event(self.rid, expected, op, self.ctx, "audio_ended"))
        self.assertEqual((event["source_revision"], event["review_revision"]), (1, 2))
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM document_versions").fetchone()[0], 1)
        saved = self.store.get_revision(self.rid, "review", 2)["payload"]["interaction_events"][0]
        self.assertEqual(saved["event_type"], "audio_ended")
        self.assertNotIn("human_verified", saved)

    def test_heartbeat_does_not_grow_review_and_events_are_bounded(self):
        review = dict(self.review, interaction_events=[{"event_type": "audio_play"}] * 2000)
        self.store.create_recording_with_document(self.rid, "tag", self.transcript, review, self.op,
                                                   self.ctx, root_id=self.root_id, generated_at="one")
        versions = {"source_revision": 1, "review_revision": 1}
        result = self.store.commit_player_event(self.rid, versions, uid(), self.ctx, "heartbeat")
        self.assertEqual(result["review_revision"], 1)
        result = self.store.commit_player_event(self.rid, versions, uid(), self.ctx, "audio_play")
        saved = self.store.get_revision(self.rid, "review", result["review_revision"])["payload"]
        self.assertEqual(len(saved["interaction_events"]), 2000)
        self.assertEqual(saved["interaction_events_evicted"], 1)

    def test_edit_keeps_annotation_and_marks_stale(self):
        self.create()
        model = self.store.capture_context(self.ctx["origin_installation_id"], "model")
        annotation = {"text": "Аннотация", "source_revision": 1, "context": model}
        self.store.save_document(self.rid, {"source_revision": 1, "review_revision": 1},
                                 "annotation", uid(), self.ctx, root_id=self.root_id,
                                 annotation=annotation)
        result = self.change({"source_revision": 1, "review_revision": 1}, "два", "two")
        self.assertEqual(result["document_revision"], 3)
        doc = self.store.get_document(self.rid, 3)
        self.assertEqual(json.loads(doc["json"])["annotation"], annotation)
        self.assertIn("опорная ревизия отличается", doc["md"].decode())

    def test_atomic_rollback_if_outbox_insert_fails(self):
        self.store.db.execute("""CREATE TRIGGER inject_failure BEFORE INSERT ON document_outbox
                                 BEGIN SELECT RAISE(ABORT, 'injected'); END""")
        with self.assertRaises(Exception):
            self.create()
        for table in ("recordings", "operations", "document_versions", "document_outbox"):
            self.assertEqual(self.store.db.execute("SELECT count(*) FROM " + table).fetchone()[0], 0)

    def test_partial_v2_then_v3_finishes_in_order(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        self.change({"source_revision": 1, "review_revision": 1}, "два", "two")
        def fail(boundary, rid, revision):
            if boundary == "replace_md" and revision == 2:
                raise RuntimeError("simulated interruption")
        with self.assertRaises(RuntimeError):
            documents.materialize_pending(self.store, self.root_id, self.root, fault=fail)
        self.change({"source_revision": 2, "review_revision": 2}, "три", "three")
        self.assertEqual(documents.materialize_pending(self.store, self.root_id, self.root),
                         [(self.rid, 2), (self.rid, 3)])
        self.assertEqual((self.root / (self.rid + ".transcript.md")).read_bytes(),
                         self.store.get_document(self.rid, 3)["md"])

    def test_external_edit_conflict_blocks_changes(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        target = self.root / (self.rid + ".transcript.md")
        target.write_bytes(b"external edit")
        self.change({"source_revision": 1, "review_revision": 1}, "два", "two")
        self.assertEqual(documents.materialize_pending(self.store, self.root_id, self.root), [])
        self.assertEqual(target.read_bytes(), b"external edit")
        with self.assertRaises(storage.Conflict):
            self.change({"source_revision": 2, "review_revision": 2}, "три", "three")
        conflict = self.store.db.execute("SELECT * FROM document_conflicts").fetchone()
        self.assertEqual(conflict["observed_bytes"], b"external edit")
        self.store.resolve_document_conflict(conflict["conflict_id"], uid(), self.ctx)
        self.assertEqual(documents.materialize_pending(self.store, self.root_id, self.root), [(self.rid, 2)])
        self.assertEqual(target.read_bytes(), self.store.get_document(self.rid, 2)["md"])
        self.assertEqual(self.store.db.execute("SELECT observed_bytes FROM document_conflicts").fetchone()[0],
                         b"external edit")

    def test_change_after_conflict_resolution_is_not_overwritten(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        target = self.root / (self.rid + ".transcript.md")
        target.write_bytes(b"external one")
        self.change({"source_revision": 1, "review_revision": 1}, "два", "two")
        documents.materialize_pending(self.store, self.root_id, self.root)
        first = self.store.db.execute("SELECT conflict_id FROM document_conflicts").fetchone()[0]
        self.store.resolve_document_conflict(first, uid(), self.ctx)
        target.write_bytes(b"external two")
        documents.materialize_pending(self.store, self.root_id, self.root)
        self.assertEqual(target.read_bytes(), b"external two")
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM document_conflicts").fetchone()[0], 2)

    def test_root_must_be_reauthorized(self):
        self.create()
        other = self.base / "other"
        other.mkdir()
        with self.assertRaises(ValueError):
            documents.materialize_pending(self.store, self.root_id, other)
        self.assertEqual(list(other.iterdir()), [])
        with self.assertRaises(ValueError):
            documents._target(self.root, "../outside")
        with self.assertRaises(ValueError):
            documents.checked_root(self.root / ".." / "other")

    def test_duplicate_root_and_unknown_user_are_domain_errors(self):
        with self.assertRaises(storage.Conflict):
            self.store.register_root(self.root)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM allowed_roots").fetchone()[0], 1)
        installations = self.store.db.execute("SELECT count(*) FROM installations").fetchone()[0]
        with self.assertRaises(ValueError):
            self.store.bind_installation(uid(), uid())
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM installations").fetchone()[0],
                         installations)

    def test_registered_roots_cannot_overlap_in_either_registration_order(self):
        parent = self.base / "root-parent"
        child = parent / "nested"
        sibling = self.base / "root-sibling"
        child.mkdir(parents=True)
        sibling.mkdir()

        self.store.register_root(parent)
        with self.assertRaises(storage.Conflict):
            self.store.register_root(child)
        self.store.register_root(sibling)

        other_path = self.base / "other-store.sqlite3"
        other = storage.initialize(other_path)
        try:
            other.register_root(child)
            with self.assertRaises(storage.Conflict):
                other.register_root(parent)
        finally:
            other.close()

    def test_local_path_checks_the_resolved_target(self):
        if os.name != "nt":
            self.skipTest("Windows drive type check only")
        with mock.patch("pathlib.Path.resolve",
                        return_value=Path(r"\\server\share\state.sqlite3")):
            with self.assertRaises(ValueError):
                storage.check_local_path(self.path)

    def test_junction_root_rejected_when_creation_permitted(self):
        if os.name != "nt":
            self.skipTest("Windows junction only")
        link = self.base / "root-junction"
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(self.root)],
                                capture_output=True)
        if result.returncode != 0:
            self.skipTest("creating junction unavailable")
        try:
            with self.assertRaises(ValueError):
                documents.checked_root(link)
        finally:
            os.rmdir(link)

    def test_invalid_current_target_after_synced_marks_error(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        target = self.root / (self.rid + ".transcript.md")
        target.unlink()
        target.mkdir()
        with self.assertRaises(ValueError):
            documents.materialize_pending(self.store, self.root_id, self.root)
        self.assertEqual(self.store.db.execute("SELECT state FROM document_outbox").fetchone()[0], "error")
        target.rmdir()
        self.assertEqual(documents.materialize_pending(self.store, self.root_id, self.root), [(self.rid, 1)])
        self.assertEqual(target.read_bytes(), self.store.get_document(self.rid, 1)["md"])

    def test_symlink_root_rejected_when_creation_permitted(self):
        link = self.base / "root-link"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except OSError as exc:
            self.skipTest("creating directory symlink unavailable: " + type(exc).__name__)
        with self.assertRaises(ValueError):
            documents.checked_root(link)

    def test_two_processes_cannot_commit_same_head(self):
        self.create()
        script = r'''
import json, sys, uuid
import storage
s=storage.open_store(sys.argv[1])
rid=sys.argv[2]
ctx=json.loads(sys.argv[3])
try:
 s.commit_change(rid, {"source_revision":1,"review_revision":1}, str(uuid.uuid4()), ctx,
                 review={"review":[{"id":"r","word_ids":["w"]}],"answers":{}})
 print("ok")
except storage.Conflict:
 print("conflict")
finally:
 s.close()
'''
        args = [sys.executable, "-B", "-c", script, str(self.path), self.rid, json.dumps(self.ctx)]
        a = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        b = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out_a, err_a = a.communicate(timeout=30)
        out_b, err_b = b.communicate(timeout=30)
        self.assertEqual((a.returncode, b.returncode), (0, 0), (err_a, err_b))
        self.assertEqual(sorted([out_a.strip(), out_b.strip()]), ["conflict", "ok"])

    def test_two_materializers_serialize_same_root(self):
        self.create()
        script = r'''
import sys
import documents, storage
s=storage.open_store(sys.argv[1])
try:
 documents.materialize_pending(s,sys.argv[2],sys.argv[3])
finally:
 s.close()
'''
        args = [sys.executable, "-B", "-c", script, str(self.path), self.root_id, str(self.root)]
        a = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        b = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _, err_a = a.communicate(timeout=30)
        _, err_b = b.communicate(timeout=30)
        self.assertEqual((a.returncode, b.returncode), (0, 0), (err_a, err_b))
        self.assertEqual(self.store.db.execute("SELECT state FROM document_outbox").fetchone()[0], "synced")
        self.assertEqual((self.root / (self.rid + ".transcript.md")).read_bytes(),
                         self.store.get_document(self.rid, 1)["md"])

    def test_restore_with_document_keeps_old_archive_and_queues_new_revision(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        old_md = self.store.get_document(self.rid, 1)["md"]
        self.change({"source_revision": 1, "review_revision": 1}, "два", "two")
        documents.materialize_pending(self.store, self.root_id, self.root)
        other = self.store.create_user("редактор")
        installation = uid()
        self.store.bind_installation(installation, other)
        ctx = self.store.capture_context(installation, "human")
        result = self.store.restore_with_document(
            self.rid, 1, 1, {"source_revision": 2, "review_revision": 2},
            uid(), ctx, root_id=self.root_id, generated_at="restore")
        self.assertEqual((result["source_revision"], result["document_revision"]), (3, 3))
        documents.materialize_pending(self.store, self.root_id, self.root)
        current = json.loads(self.store.get_document(self.rid, 3)["json"])
        self.assertEqual(current["source_action"]["author_ref"]["user_id"], other)
        self.assertEqual(current["source_content_action"]["author_ref"], self.ctx["author_ref"])
        self.assertEqual((self.root / documents.archive_paths(self.rid, 1)["md"]).read_bytes(), old_md)

    def test_answer_apply_undo_keep_separate_authors(self):
        self.create()
        responder = self.ctx
        answered = {"review": [{"id": "r", "word_ids": ["w"]}],
                    "answers": {"r": {"text": "новое", "actor_kind": "human",
                                      "author_ref": responder["author_ref"]}}}
        self.store.commit_change_with_document(
            self.rid, {"source_revision": 1, "review_revision": 1}, uid(), responder,
            review=answered, root_id=self.root_id, generated_at="answer")
        user2 = self.store.create_user("применил")
        install2 = uid()
        self.store.bind_installation(install2, user2)
        applier = self.store.capture_context(install2, "human")
        applied = dict(answered, applications=[{"id": "a", "kind": "apply", "answer_id": "r",
                                                "context": applier}],
                       undo={"r": {"source_revision": 1}})
        changed = {"segments": [{"segment_id": "s", "speaker": "А",
                                 "words": [{"word_id": "w", "w": "новое"}]}]}
        self.store.commit_change_with_document(
            self.rid, {"source_revision": 1, "review_revision": 2}, uid(), applier,
            transcript=changed, review=applied, root_id=self.root_id, generated_at="apply")
        user3 = self.store.create_user("отменил")
        install3 = uid()
        self.store.bind_installation(install3, user3)
        undoer = self.store.capture_context(install3, "human")
        undone = dict(applied, applications=applied["applications"] + [
            {"id": "u", "kind": "undo", "answer_id": "r", "context": undoer}])
        self.store.commit_change_with_document(
            self.rid, {"source_revision": 2, "review_revision": 3}, uid(), undoer,
            transcript=self.transcript, review=undone, root_id=self.root_id, generated_at="undo")
        current = self.store.get_current(self.rid)
        self.assertEqual(current["transcript"]["payload"], self.transcript)
        self.assertEqual(current["review"]["payload"]["answers"]["r"]["author_ref"], responder["author_ref"])
        public = json.loads(self.store.get_document(self.rid, 4)["json"])["review"]
        self.assertEqual(public["applications"][0]["context"]["author_ref"]["user_id"], user2)
        self.assertEqual(public["applications"][1]["context"]["author_ref"]["user_id"], user3)
        self.assertEqual(public["answers"]["r"]["author_ref"], responder["author_ref"])

    def test_backup_restore_requires_new_roots_and_installation(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        self.change({"source_revision": 1, "review_revision": 1}, "два", "two")
        backup = self.base / "backup.sqlite3"
        self.assertTrue(self.store.backup(backup)["inactive"])
        before = (self.root / (self.rid + ".transcript.md")).read_bytes()
        restored = storage.open_store(backup)
        try:
            with self.assertRaises(ValueError):
                documents.materialize_pending(restored, self.root_id, self.root)
            with self.assertRaises(storage.Conflict):
                restored.bind_installation(uid(), self.ctx["author_ref"]["user_id"])
            with self.assertRaises(storage.Conflict):
                restored.create_user("new")
            with self.assertRaises(storage.Conflict):
                restored.rename_user(self.ctx["author_ref"]["user_id"], "new")
            new_root = self.base / "new-root"
            new_root.mkdir()
            with self.assertRaises(ValueError):
                restored.rebind_restored_copy({self.root_id: self.root}, uid(),
                                              self.ctx["author_ref"]["user_id"])
            with self.assertRaises(ValueError):
                restored.rebind_restored_copy({self.root_id: self.base}, uid(),
                                              self.ctx["author_ref"]["user_id"])
            child = self.root / "child"
            child.mkdir()
            with self.assertRaises(ValueError):
                restored.rebind_restored_copy({self.root_id: child}, uid(),
                                              self.ctx["author_ref"]["user_id"])
            restored.rebind_restored_copy({self.root_id: new_root}, uid(),
                                          self.ctx["author_ref"]["user_id"])
            self.assertEqual(documents.materialize_pending(restored, self.root_id, new_root),
                             [(self.rid, 1), (self.rid, 2)])
            self.assertEqual((new_root / (self.rid + ".transcript.md")).read_bytes(),
                             restored.get_document(self.rid, 2)["md"])
            self.assertEqual((self.root / (self.rid + ".transcript.md")).read_bytes(), before)
            self.assertEqual(restored.get_document(self.rid, 1)["md"], self.store.get_document(self.rid, 1)["md"])
            self.assertEqual(storage.validate_copy(backup)["integrity"], "ok")
        finally:
            restored.close()

    def test_inherited_annotation_survives_restore(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        annotation = {"text": "историческая заметка", "source_revision": 1, "context": self.ctx}
        self.store.save_document(self.rid, {"source_revision": 1, "review_revision": 1},
                                 "annotation", uid(), self.ctx, root_id=self.root_id,
                                 annotation=annotation)
        backup = self.base / "annotation-backup.sqlite3"
        self.store.backup(backup)
        restored = storage.open_store(backup)
        new_root = self.base / "annotation-root"
        new_root.mkdir()
        new_install = uid()
        try:
            restored.rebind_restored_copy({self.root_id: new_root}, new_install,
                                          self.ctx["author_ref"]["user_id"])
            new_ctx = restored.capture_context(new_install, "human")
            saved = restored.save_document(
                self.rid, {"source_revision": 1, "review_revision": 1}, "after-restore",
                uid(), new_ctx, root_id=self.root_id)
            payload = json.loads(restored.get_document(self.rid, saved["document_revision"])["json"])
            self.assertEqual(payload["annotation"], annotation)
            changed = {"segments": [{"segment_id": "s", "speaker": "А",
                                      "words": [{"word_id": "w", "w": "два"}]}]}
            result = restored.commit_change_with_document(
                self.rid, {"source_revision": 1, "review_revision": 1}, uid(), new_ctx,
                transcript=changed, review=self.review, root_id=self.root_id,
                generated_at="after-change")
            payload = json.loads(restored.get_document(self.rid, result["document_revision"])["json"])
            self.assertEqual(payload["annotation"], annotation)
        finally:
            restored.close()

    def test_initialization_and_backup_leave_no_staged_service_files(self):
        # После проверки копии только для чтения не должно остаться -wal/-shm временного имени.
        self.assertEqual(sorted(p.name for p in self.base.glob("state.sqlite3.incomplete-*")), [])
        self.create()
        target = self.base / "copy.sqlite3"
        self.store.backup(target)
        self.assertTrue(target.is_file())
        self.assertEqual(sorted(p.name for p in self.base.glob("*.incomplete-*")), [])
        other = self.base / "other.sqlite3"
        storage.initialize(other).close()
        self.assertEqual(sorted(p.name for p in self.base.glob("*.incomplete-*")), [])

    def test_failed_initialization_and_backup_do_not_publish_partial_destination(self):
        self.create()
        target = self.base / "bad-backup.sqlite3"
        with mock.patch("storage.validate_copy", side_effect=storage.StorageError("injected")):
            with self.assertRaises(storage.StorageError):
                self.store.backup(target)
        self.assertFalse(target.exists())
        self.assertEqual(len(list(self.base.glob("bad-backup.sqlite3.incomplete-*"))), 1)
        target2 = self.base / "bad-init.sqlite3"
        with mock.patch("storage.validate_copy", side_effect=storage.StorageError("injected")):
            with self.assertRaises(storage.StorageError):
                storage.initialize(target2)
        self.assertFalse(target2.exists())
        self.assertEqual(len(list(self.base.glob("bad-init.sqlite3.incomplete-*"))), 1)

    def test_validate_copy_rejects_document_with_incompatible_pair(self):
        self.create()
        self.change({"source_revision": 1, "review_revision": 1}, "два", "two")
        prepared = documents.prepare(self.rid, 1, 2, 2, self.transcript, self.review,
                                     self.ctx, self.ctx, "synthetic")
        self.store.db.execute("UPDATE document_versions SET source_revision=1, md=?, json=?, "
                              "manifest=?, archive_manifest=?, hashes=? WHERE recording_id=? AND document_revision=2",
                              (prepared["md"], prepared["json"], prepared["manifest"],
                               prepared["archive_manifest"], json.dumps(prepared["hashes"]), self.rid))
        with self.assertRaises(storage.StorageError):
            storage.validate_copy(self.path)

    def test_busy_and_invalid_schema_are_reported(self):
        invalid = self.base / "empty.sqlite3"
        sqlite3.connect(invalid).close()
        with self.assertRaises(storage.UnsupportedSchema):
            storage.open_store(invalid)
        invalid.write_bytes(b"not sqlite")
        with self.assertRaises(storage.UnsupportedSchema):
            storage.open_store(invalid)
        if os.name == "nt":
            with self.assertRaises(ValueError):
                storage.check_local_path(r"\\server\share\state.sqlite3")
        other = storage.open_store(self.path)
        try:
            self.store.db.execute("BEGIN IMMEDIATE")
            other.db.execute("PRAGMA busy_timeout=1")
            with self.assertRaises(storage.Busy):
                other.create_user("заблокировано")
        finally:
            self.store.db.execute("ROLLBACK")
            other.close()

    def test_restored_roots_may_not_overlap_each_other(self):
        root2 = self.base / "other-old"
        root2.mkdir()
        second = self.store.register_root(root2)
        backup = self.base / "copy.sqlite3"
        self.store.backup(backup)
        restored = storage.open_store(backup)
        parent = self.base / "new-parent"
        child = parent / "child"
        child.mkdir(parents=True)
        try:
            with self.assertRaises(ValueError):
                restored.rebind_restored_copy({self.root_id: parent, second: child}, uid(),
                                              self.ctx["author_ref"]["user_id"])
        finally:
            restored.close()

    def test_archive_link_race_records_conflict(self):
        self.create()
        def race(source, dest):
            Path(dest).write_bytes(b"external archive")
            raise FileExistsError("injected")
        with mock.patch("documents.os.link", side_effect=race):
            documents.materialize_pending(self.store, self.root_id, self.root)
        row = self.store.db.execute("SELECT * FROM document_conflicts").fetchone()
        self.assertEqual(row["kind"], "archive_md")
        self.assertEqual(row["observed_bytes"], b"external archive")
        self.assertEqual(self.store.db.execute("SELECT state FROM document_outbox").fetchone()[0], "conflict")

    def test_fault_at_every_write_boundary_recovers(self):
        self.create()
        boundaries = ("writing", "archive_md", "archive_json", "archive_manifest",
                      "replace_md", "replace_json", "replace_manifest", "before_ack", "after_ack")
        for boundary in boundaries:
            with self.subTest(boundary=boundary):
                target = self.base / ("backup-" + boundary + ".sqlite3")
                self.store.backup(target)
                restored = storage.open_store(target)
                root = self.base / ("root-" + boundary)
                root.mkdir()
                try:
                    restored.rebind_restored_copy({self.root_id: root}, uid(), self.ctx["author_ref"]["user_id"])
                    def fail(where, rid, revision):
                        if where == boundary:
                            raise RuntimeError("injected")
                    with self.assertRaises(RuntimeError):
                        documents.materialize_pending(restored, self.root_id, root, fault=fail)
                    documents.materialize_pending(restored, self.root_id, root)
                    self.assertEqual((root / (self.rid + ".transcript.md")).read_bytes(),
                                     restored.get_document(self.rid, 1)["md"])
                    self.assertEqual(restored.db.execute(
                        "SELECT state FROM document_outbox").fetchone()[0], "synced")
                finally:
                    restored.close()


    def test_verify_reports_consistent_files_without_writes(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        before = self.files_snapshot()
        report = documents.verify_files(self.store, self.root_id, self.root)
        self.assertTrue(report["consistent"])
        self.assertEqual(report["checked"], 6)
        self.assertEqual(report["counts"]["ok"], 6)
        self.assertEqual({entry["outbox_state"] for entry in report["entries"]}, {"synced"})
        self.assertEqual(self.files_snapshot(), before)

    def test_verify_reports_missing_and_modified_files_and_does_not_repair(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        current = self.root / (self.rid + ".transcript.md")
        archive = self.root / documents.archive_paths(self.rid, 1)["md"]
        current.write_bytes(b"external edit")
        archive.unlink()
        report = documents.verify_files(self.store, self.root_id, self.root)
        self.assertFalse(report["consistent"])
        statuses = {(entry["kind"], entry["status"]) for entry in report["entries"]}
        self.assertIn(("md", "modified"), statuses)
        self.assertIn(("archive_md", "missing"), statuses)
        self.assertEqual(current.read_bytes(), b"external edit")
        self.assertFalse(archive.exists())
        self.assertEqual(self.store.db.execute(
            "SELECT count(*) FROM materialization_events").fetchone()[0], 0)
        self.assertEqual(self.store.db.execute("SELECT state FROM document_outbox").fetchone()[0], "synced")
        documents.materialize_pending(self.store, self.root_id, self.root)
        self.assertEqual(current.read_bytes(), b"external edit")
        self.assertFalse(archive.exists())

    def test_verify_reports_unconfirmed_and_registry_gaps(self):
        self.create()
        report = documents.verify_files(self.store, self.root_id, self.root)
        self.assertEqual(report["counts"]["unconfirmed"], 6)
        self.assertEqual({entry["outbox_state"] for entry in report["entries"]}, {"pending"})
        archive_md = documents.archive_paths(self.rid, 1)["md"]
        self.store.db.execute("DELETE FROM file_locations WHERE relative_path=?", (archive_md,))
        report = documents.verify_files(self.store, self.root_id, self.root)
        self.assertEqual(report["counts"]["registry_incomplete"], 1)
        gap = next(entry for entry in report["entries"] if entry["status"] == "registry_incomplete")
        self.assertEqual((gap["kind"], gap["relative_path"]), ("archive_md", archive_md))
        self.store.db.execute("UPDATE file_locations SET confirmed_revision=1, confirmed_hash=?, "
                              "generation=1 WHERE kind='md'", ("0" * 64,))
        report = documents.verify_files(self.store, self.root_id, self.root)
        self.assertEqual(report["counts"]["registry_mismatch"], 1)

    def test_verify_reports_unreadable_stored_hashes(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        self.store.db.execute("UPDATE document_versions SET hashes='{bad json' WHERE recording_id=?",
                              (self.rid,))
        report = documents.verify_files(self.store, self.root_id, self.root)
        self.assertEqual(report["counts"]["registry_mismatch"], 6)
        self.assertFalse(report["consistent"])

    def test_verify_reports_invalid_target_and_conflict_without_raising(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        archive = self.root / documents.archive_paths(self.rid, 1)["md"]
        archive.unlink()
        archive.mkdir()
        report = documents.verify_files(self.store, self.root_id, self.root)
        entry = next(item for item in report["entries"] if item["kind"] == "archive_md")
        self.assertEqual(entry["status"], "invalid_target")
        self.assertFalse(report["consistent"])
        archive.rmdir()
        target = self.root / (self.rid + ".transcript.md")
        target.write_bytes(b"edited")
        documents.materialize_pending(self.store, self.root_id, self.root)
        report = documents.verify_files(self.store, self.root_id, self.root)
        entry = next(item for item in report["entries"]
                     if item["relative_path"] == self.rid + ".transcript.md")
        self.assertEqual(entry["status"], "unresolved_conflict")
        self.assertEqual(entry["observed_sha256"], hashlib.sha256(b"edited").hexdigest())

    def test_verify_after_rebind_reports_unconfirmed_then_ok(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        backup = self.base / "verify-backup.sqlite3"
        self.store.backup(backup)
        restored = storage.open_store(backup)
        new_root = self.base / "verify-restored"
        new_root.mkdir()
        try:
            restored.rebind_restored_copy({self.root_id: new_root},
                                          uid(), self.ctx["author_ref"]["user_id"])
            report = documents.verify_files(restored, self.root_id, new_root)
            self.assertEqual(report["counts"]["unconfirmed"], 6)
            documents.materialize_pending(restored, self.root_id, new_root)
            report = documents.verify_files(restored, self.root_id, new_root)
            self.assertTrue(report["consistent"])
            self.assertEqual(report["counts"]["ok"], 6)
        finally:
            restored.close()

    def test_verify_requires_the_registered_root(self):
        other = self.base / "not-a-root"
        other.mkdir()
        with self.assertRaises(ValueError):
            documents.verify_files(self.store, self.root_id, other)

    def test_readonly_store_rejects_writes(self):
        readonly = storage.open_store_readonly(self.path)
        try:
            self.assertEqual(readonly.db.execute("PRAGMA query_only").fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                readonly.db.execute("UPDATE restore_guard SET locked=1 WHERE id=1")
        finally:
            readonly.close()

    def test_verify_cli_emits_json_and_status_code(self):
        self.create()
        documents.materialize_pending(self.store, self.root_id, self.root)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = verify_document_files.main([
                "--db", str(self.path), "--root-id", self.root_id, "--root", str(self.root)])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(report["consistent"])

    def files_snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}


if __name__ == "__main__":
    unittest.main()
