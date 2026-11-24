"""РАР-6.1, PR 5а: три текущих файла записи сверяются до первой замены, замена идёт md → json → manifest.

Раньше сверка и замена чередовались, а порядок брался из ключей паспорта (в сохранённом JSON они отсортированы):
json → manifest → md. Конфликт на втором или третьем файле оставлял первые уже замененными (смешанный набор),
а обрыв процесса — набор из файлов разных версий в порядке, который не совпадал с «manifest последним».
Режим — `sqlite` (материализатор есть только в нём). Данные синтетические, всё во временной папке (TEMP песочницы).
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import uuid
from pathlib import Path
from unittest import mock

import documents
import storage

REPO = Path(__file__).resolve().parent.parent
KINDS = ("md", "json", "manifest")
BOUNDARIES = ("writing", "archive_md", "archive_json", "archive_manifest",
              "replace_md", "replace_json", "replace_manifest", "before_ack", "after_ack")


def uid():
    return str(uuid.uuid4())


def transcript(value):
    return {"segments": [{"segment_id": "s", "speaker": "А", "words": [{"word_id": "w", "w": value}]}]}


REVIEW = {"review": [{"id": "r", "word_ids": ["w"]}], "answers": {}}


class Env:
    """Одна база и один корень документов; записей может быть несколько."""

    def __init__(self, base, label="env"):
        self.base = Path(base) / label
        self.root = self.base / "documents"
        self.root.mkdir(parents=True)
        self.path = self.base / "state.sqlite3"
        self.store = storage.initialize(self.path)
        user = self.store.create_user("ник")
        installation = uid()
        self.store.bind_installation(installation, user)
        self.ctx = self.store.capture_context(installation, "human")
        self.root_id = self.store.register_root(self.root)
        self.revision = {}

    def create(self, tag="tag", value="один"):
        rid = uid()
        self.store.create_recording_with_document(
            rid, tag, transcript(value), REVIEW, uid(), self.ctx, root_id=self.root_id, generated_at="one")
        self.revision[rid] = 1
        return rid

    def change(self, rid, value, stamp="two"):
        n = self.revision[rid]
        self.store.commit_change_with_document(
            rid, {"source_revision": n, "review_revision": n}, uid(), self.ctx, transcript=transcript(value),
            review=REVIEW, root_id=self.root_id, generated_at=stamp)
        self.revision[rid] = n + 1

    def materialize(self, **kwargs):
        return documents.materialize_pending(self.store, self.root_id, self.root, **kwargs)

    def file(self, rid, kind):
        return self.root / documents.current_paths(rid)[kind]

    def version(self, rid, kind):
        """Какой версии программы равен файл: 1, 2…; None — файла нет; 0 — чужие байты."""
        digest = documents._file_hash(self.file(rid, kind))
        if digest is None:
            return None
        for revision in range(1, self.revision[rid] + 1):
            if self.store.get_document(rid, revision)["hashes"][kind] == digest:
                return revision
        return 0

    def versions(self, rid):
        return {kind: self.version(rid, kind) for kind in KINDS}

    def outbox(self, rid, revision):
        row = self.store.db.execute("SELECT state, error_code FROM document_outbox WHERE recording_id=? "
                                    "AND document_revision=?", (rid, revision)).fetchone()
        return row["state"], row["error_code"]

    def conflicts(self):
        return [tuple(row) for row in self.store.db.execute(
            "SELECT recording_id, kind FROM document_conflicts ORDER BY rowid")]

    def resolve(self, rid):
        for row in self.store.db.execute("SELECT conflict_id FROM document_conflicts WHERE recording_id=? AND "
                                         "resolution_operation_id IS NULL", (rid,)).fetchall():
            self.store.resolve_document_conflict(row[0], uid(), self.ctx)

    def close(self):
        self.store.close()


def current_writes(call):
    """Порядок замен текущих файлов (архив не считается) при вызове call()."""
    seen = []
    real = documents._write_atomic

    def spy(target, data, **kwargs):
        if "history" not in Path(target).parts:
            seen.append(Path(target).name.split(".", 1)[1])
        return real(target, data, **kwargs)

    with mock.patch.object(documents, "_write_atomic", spy):
        call()
    return seen


class WriteOrderCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p5a-", dir=os.environ["TEMP"])
        self.addCleanup(self.temp.cleanup)

    def env(self, label="env"):
        env = Env(self.temp.name, label)
        self.addCleanup(env.close)
        return env

    def synced(self, label="env", revised=True):
        """Запись с записанной версией 1; revised — и версия 2 стоит в очереди."""
        env = self.env(label)
        rid = env.create()
        env.materialize()
        if revised:
            env.change(rid, "два")
        return env, rid


class AllThreeFilesAreCheckedBeforeTheFirstReplacement(WriteOrderCase):
    def test_external_edit_of_any_file_replaces_none_of_the_others(self):
        for kind in KINDS:
            with self.subTest(edited=kind):
                env, rid = self.synced(kind)
                env.file(rid, kind).write_bytes(b"external " + kind.encode())
                self.assertEqual(env.materialize(), [])
                others = {k: v for k, v in env.versions(rid).items() if k != kind}
                self.assertEqual(others, {k: 1 for k in KINDS if k != kind})
                self.assertEqual(env.version(rid, kind), 0)
                self.assertEqual(env.conflicts(), [(rid, kind)])
                self.assertEqual(env.outbox(rid, 2)[0], "conflict")

    def test_order_of_replacements_is_md_json_manifest_for_a_stored_document(self):
        env, rid = self.synced()
        written = current_writes(env.materialize)
        self.assertEqual(written, ["transcript.md", "transcript.json", "manifest.json"])
        self.assertEqual(env.versions(rid), {"md": 2, "json": 2, "manifest": 2})

    def test_damaged_copy_of_the_last_file_replaces_none(self):
        env, rid = self.synced()
        env.file(rid, "manifest").write_bytes(b"my manifest")
        env.materialize()
        env.resolve(rid)
        for row in env.store.db.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND "
                                        "tbl_name='document_conflicts'").fetchall():
            env.store.db.execute(f'DROP TRIGGER "{row[0]}"')
        env.store.db.execute("UPDATE document_conflicts SET observed_bytes=?", (b"damaged",))
        with self.assertRaises(storage.StorageError):
            env.materialize()
        self.assertEqual(env.versions(rid), {"md": 1, "json": 1, "manifest": 0})
        self.assertEqual(env.outbox(rid, 2), ("error", "damaged_copy"))

    def test_file_that_disappeared_after_confirmation_replaces_none(self):
        env, rid = self.synced()
        env.file(rid, "md").write_bytes(b"my text")
        env.materialize()
        env.resolve(rid)
        env.file(rid, "md").unlink()
        self.assertEqual(env.materialize(), [])
        self.assertEqual(env.versions(rid), {"md": None, "json": 1, "manifest": 1})
        self.assertEqual(env.outbox(rid, 2), ("error", "unclassified_conflict"))

    def test_directory_in_place_of_the_last_file_replaces_none(self):
        env, rid = self.synced()
        manifest = env.file(rid, "manifest")
        manifest.unlink()
        manifest.mkdir()
        with self.assertRaises(ValueError):
            env.materialize()
        self.assertEqual({k: env.version(rid, k) for k in ("md", "json")}, {"md": 1, "json": 1})
        self.assertEqual(env.outbox(rid, 2), ("error", "io_or_interrupted"))

    def test_other_record_of_the_root_is_written_in_full(self):
        env, first = self.synced("two-records", revised=False)
        second = env.create("tag2")
        env.materialize()
        env.change(first, "два")
        env.change(second, "два")
        env.file(first, "md").write_bytes(b"external")
        env.materialize()
        self.assertEqual(env.versions(first), {"md": 0, "json": 1, "manifest": 1})
        self.assertEqual(env.versions(second), {"md": 2, "json": 2, "manifest": 2})

    def test_repeat_after_refusal_registers_no_second_conflict_and_writes_nothing(self):
        env, rid = self.synced()
        env.file(rid, "json").write_bytes(b"external")
        env.materialize()
        before = env.versions(rid)
        for _ in range(2):
            env.materialize()
        self.assertEqual(env.conflicts(), [(rid, "json")])
        self.assertEqual(env.versions(rid), before)


class RaceBetweenTheCheckAndTheReplacement(WriteOrderCase):
    def test_edit_made_after_the_check_is_not_overwritten(self):
        env, rid = self.synced()

        def fault(boundary, record, revision):
            if boundary == "replace_md" and revision == 2:
                env.file(rid, "json").write_bytes(b"late edit")

        self.assertEqual(env.materialize(fault=fault), [])
        self.assertEqual(env.file(rid, "json").read_bytes(), b"late edit")
        self.assertEqual(env.versions(rid), {"md": 2, "json": 0, "manifest": 1})
        self.assertEqual(env.conflicts(), [(rid, "json")])
        env.resolve(rid)
        self.assertEqual(env.materialize(), [(rid, 2)])
        self.assertEqual(env.versions(rid), {"md": 2, "json": 2, "manifest": 2})


class MissingFilesAreRestoredAfterTheCheck(WriteOrderCase):
    def test_other_file_edited_outside_blocks_restoring(self):
        env, rid = self.synced(revised=False)
        env.file(rid, "json").unlink()
        env.file(rid, "md").write_bytes(b"external")
        env.materialize()
        self.assertIsNone(env.version(rid, "json"))
        self.assertEqual(env.conflicts(), [(rid, "md")])
        self.assertEqual(env.store.db.execute("SELECT count(*) FROM materialization_events").fetchone()[0], 0)
        env.resolve(rid)
        self.assertEqual(env.materialize(), [(rid, 1)])
        self.assertEqual(env.versions(rid), {"md": 1, "json": 1, "manifest": 1})   # «заменить» вернуло версию программы

    def test_edited_file_later_in_the_order_blocks_restoring_of_earlier_ones(self):
        env, rid = self.synced(revised=False)
        env.file(rid, "md").unlink()
        env.file(rid, "manifest").write_bytes(b"external")
        env.materialize()
        self.assertIsNone(env.version(rid, "md"))            # пропавший md не восстановлен: manifest изменён снаружи
        self.assertEqual(env.conflicts(), [(rid, "manifest")])
        self.assertEqual(env.store.db.execute("SELECT count(*) FROM materialization_events").fetchone()[0], 0)

    def test_all_missing_are_restored_in_order(self):
        env, rid = self.synced(revised=False)
        for kind in KINDS:
            env.file(rid, kind).unlink()
        written = current_writes(env.materialize)
        self.assertEqual(written, ["transcript.md", "transcript.json", "manifest.json"])
        events = [row[0] for row in env.store.db.execute(
            "SELECT kind FROM materialization_events ORDER BY rowid")]
        self.assertEqual(events, ["md", "json", "manifest"])
        self.assertEqual(env.versions(rid), {"md": 1, "json": 1, "manifest": 1})


CHILD = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, sys.argv[1])
    import documents, storage
    from pathlib import Path
    store = storage.open_store(Path(sys.argv[2]))
    boundary, root_id, root = sys.argv[3], sys.argv[4], Path(sys.argv[5])

    def fault(where, rid, revision):
        if where == boundary and revision == 2:
            os._exit(9)

    documents.materialize_pending(store, root_id, root, fault=fault)
    sys.exit(0)
""")


class CrashOnEveryBoundary(WriteOrderCase):
    def crash(self, label, boundary):
        env, rid = self.synced(label)
        env.close()
        done = subprocess.run([sys.executable, "-B", "-c", CHILD, str(REPO), str(env.path), boundary,
                               env.root_id, str(env.root)], capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 9, done.stderr)
        env.store = storage.open_store(env.path)
        self.addCleanup(env.close)
        return env, rid

    def test_files_after_the_crash_are_a_prefix_of_the_order_and_the_next_call_finishes(self):
        for boundary in BOUNDARIES:
            with self.subTest(boundary=boundary):
                env, rid = self.crash("crash-" + boundary, boundary)
                found = env.versions(rid)
                self.assertTrue(set(found.values()) <= {1, 2}, found)         # только версии, которые принимает прежний код
                order = [found[kind] for kind in KINDS]
                self.assertEqual(order, sorted(order, reverse=True), f"manifest не может быть новее md и json: {found}")
                self.assertEqual(env.outbox(rid, 2)[0], "synced" if boundary == "after_ack" else "writing")
                expected = env.store.db.execute("SELECT expected_previous_hashes FROM document_outbox WHERE "
                                                "recording_id=? AND document_revision=2", (rid,)).fetchone()[0]
                previous = env.store.get_document(rid, 1)["hashes"]
                self.assertEqual(storage._loads(expected), {kind: previous[kind] for kind in KINDS})
                self.assertEqual(env.materialize(), [] if boundary == "after_ack" else [(rid, 2)])
                self.assertEqual(env.versions(rid), {"md": 2, "json": 2, "manifest": 2})
                self.assertEqual(env.outbox(rid, 2), ("synced", None))
                self.assertTrue(documents.verify_files(env.store, env.root_id, env.root)["consistent"])


class StateLeftByThePreviousCode(WriteOrderCase):
    def test_previous_crash_states_are_finished(self):
        # прежний порядок json → manifest → md: после замены json и после замены manifest
        for label, newer in (("json-only", ("json",)), ("json-manifest", ("json", "manifest"))):
            with self.subTest(left=label):
                env, rid = self.synced(label)
                document = env.store.get_document(rid, 2)
                for kind in newer:
                    env.file(rid, kind).write_bytes(document[kind])
                env.store.db.execute("UPDATE document_outbox SET state='writing' WHERE recording_id=? AND "
                                     "document_revision=2", (rid,))
                self.assertEqual(env.materialize(), [(rid, 2)])
                self.assertEqual(env.versions(rid), {"md": 2, "json": 2, "manifest": 2})


if __name__ == "__main__":
    unittest.main()
