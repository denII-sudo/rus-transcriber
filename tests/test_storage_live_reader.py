"""БД-2 (S-1, О-3): читатель живой базы видит записанное другим соединением после своего открытия.

Прежде open_store_readonly на закрытой базе без -wal/-shm открывал её с immutable=1: SQLite считает такой файл
неизменным, и читатель после чужой записи видел прежнее состояние (опыт А — писатель держит соединение, данные в
-wal; опыт Б — писатель записал и закрылся, журнал перенесён в основной файл). Здесь оба опыта детерминированно в
одном процессе: второе соединение пишет между открытием читателя и его запросом. На прежнем коде тесты опытов
падают (читатель видит 1 пользователя вместо 4). Тесты не опираются на отказ Windows переименовать открытый файл.

Снимок (open_snapshot_readonly, validate_copy по умолчанию) — для файлов, которые никто не пишет, — по-прежнему
следов не оставляет. Все базы синтетические, во временной папке.
"""
import os
import shutil
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import core
import core_screen
import paths
import recordings
import review_pack
import storage


def names(directory):
    return sorted(p.name for p in Path(directory).iterdir())


def add_users(path, count, *, keep_open=False):
    """Другое соединение записывает count пользователей (каждый — отдельная транзакция, поле 64 КБ, чтобы
    записи занимали много страниц, как в опытах 28.11.2026)."""
    writer = storage.open_store(path)
    for _ in range(count):
        writer.db.execute("BEGIN IMMEDIATE")
        writer.db.execute("INSERT INTO local_users VALUES (?, ?, 't', 't')", (str(uuid.uuid4()), "x" * 65536))
        writer.db.execute("COMMIT")
    if keep_open:
        return writer
    writer.close()
    return None


def users(store):
    return store.db.execute("SELECT count(*) FROM local_users").fetchone()[0]


class LiveReaderExperiments(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="live-reader-", dir=os.environ.get("TEMP"))
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.path = self.base / "state.sqlite3"
        storage.initialize(self.path, initial_display_name="Тест").close()
        self.assertEqual(names(self.base), ["state.sqlite3"])      # закрытая база без служебных файлов

    def test_a_writer_keeps_connection_reader_sees_wal(self):
        reader = storage.open_store_readonly(self.path)
        try:
            self.assertEqual(users(reader), 1)
            writer = add_users(self.path, 3, keep_open=True)
            try:
                self.assertEqual(users(reader), 4, "читатель не видит подтверждённое в -wal (immutable)")
            finally:
                writer.close()
            self.assertEqual(users(reader), 4)
        finally:
            reader.close()

    def test_b_writer_writes_and_closes_reader_sees_checkpointed_data(self):
        reader = storage.open_store_readonly(self.path)
        try:
            self.assertEqual(users(reader), 1)
            add_users(self.path, 3)
            self.assertEqual(users(reader), 4, "читатель видит прежнее состояние после чужой записи")
            self.assertEqual(users(reader), 4)                 # соседний запрос — без ошибки и то же
            self.assertEqual(reader.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        finally:
            reader.close()

    def test_live_reader_is_plain_readonly_and_rejects_writes(self):
        reader = storage.open_store_readonly(self.path)
        try:
            self.assertEqual(reader.db.execute("PRAGMA query_only").fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                reader.db.execute("UPDATE restore_guard SET locked=1 WHERE id=1")
        finally:
            reader.close()

    def test_live_reader_traces_are_removed_by_next_writer(self):
        # цена mode=ro (опыт Г): на закрытой базе остаются -wal (0 байт) и -shm; основной файл не меняется
        before = self.path.read_bytes()
        storage.open_store_readonly(self.path).close()
        self.assertEqual(self.path.read_bytes(), before)
        left = names(self.base)
        self.assertTrue(set(left) <= {"state.sqlite3", "state.sqlite3-wal", "state.sqlite3-shm"}, left)
        wal = self.base / "state.sqlite3-wal"
        if wal.exists():
            self.assertEqual(wal.stat().st_size, 0)
        storage.open_store(self.path).close()                  # следующее закрытие соединения на запись
        self.assertEqual(names(self.base), ["state.sqlite3"])

    def test_snapshot_reader_leaves_no_traces(self):
        store = storage.open_snapshot_readonly(self.path)
        try:
            self.assertEqual(users(store), 1)
            self.assertEqual(store.db.execute("PRAGMA query_only").fetchone()[0], 1)
        finally:
            store.close()
        storage.validate_copy(self.path)                     # по умолчанию — снимок
        storage.validate_copy(self.path, live=False)
        self.assertEqual(names(self.base), ["state.sqlite3"])

    def test_snapshot_reader_with_journal_still_reads_it(self):
        # снимок с -wal рядом (не наш случай, но страховка): immutable=1 не годится, читается обычным mode=ro
        writer = add_users(self.path, 3, keep_open=True)
        try:
            store = storage.open_snapshot_readonly(self.path)
            try:
                self.assertEqual(users(store), 4)
            finally:
                store.close()
        finally:
            writer.close()

    def test_validate_copy_live_sees_data_of_open_writer(self):
        writer = storage.open_store(self.path)
        try:
            writer.db.execute("UPDATE local_users SET display_name='ИЗМЕНЕНО'")
            reader_check = storage.validate_copy(self.path, live=True)
            self.assertEqual(reader_check["integrity"], "ok")
            store = storage.open_store_readonly(self.path)
            try:
                self.assertEqual(store.db.execute("SELECT display_name FROM local_users").fetchone()[0], "ИЗМЕНЕНО")
            finally:
                store.close()
        finally:
            writer.close()


class LiveReaderUpperLevel(unittest.TestCase):
    """Экран и проверка версий карточек: чужая запись между открытием читателя и его запросами видна."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="live-screen-", dir=os.environ.get("TEMP")))
        self.addCleanup(shutil.rmtree, self.base, True)
        self.db = self.base / "state.sqlite3"
        patcher = mock.patch.object(paths, "STATE_DB", self.db)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = storage.initialize(self.db, initial_display_name="Тест")
        self.first = self.make_recording(self.store, "Первая.wav")
        self.store.close()
        self.assertEqual(names(self.base), sorted(["state.sqlite3", "Записи"]))     # база закрыта, следов нет

    def make_recording(self, store, name):
        source = self.base / "Записи" / name
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(b"synthetic")
        layout = recordings.document_layout(source, "")
        layout["root"].mkdir()
        root_id, doc_paths = recordings.prepare_new_layout(store, layout["root"], layout["paths"])
        ident = str(uuid.uuid4())
        data = {"file": str(source), "duration": "00:00:02", "speakers": ["Спикер 1"], "quality": {},
                "review": [], "review_done": False,
                "segments": [{"start": 0, "end": 2, "speaker": "Спикер 1",
                              "words": [{"w": "слово", "s": 0, "e": 1}, {"w": "второе", "s": 1, "e": 2}]}]}
        recordings.commit_worker_result(
            store, data, source, root_id, recordings.operation_context(store, "model"), recording_id=ident,
            operation_id=str(uuid.uuid4()), generated_at="2026-01-01T00:00:00Z", document_paths=doc_paths)
        recordings.materialize_recording(store, ident)
        return ident

    def change_text(self, store, ident):
        current = store.get_current(ident)
        transcript = current["transcript"]["payload"]
        transcript["segments"][0]["words"][1]["w"] = "исправлено"
        binding = recordings.document_binding(store, ident)
        store.commit_change_with_document(
            ident, {"source_revision": current["transcript"]["revision"],
                    "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(store, "human"),
            transcript=transcript, review=current["review"]["payload"], root_id=binding["root_id"],
            generated_at="2026-01-01T00:02:00Z")

    def write_after_reader_opens(self, action):
        """Подменить recordings.open_store: открыть первого читателя как обычно, затем другое соединение пишет и
        держит соединение до конца теста (опыт А: подтверждённое лежит только в -wal), и только потом вызывающий
        выполняет запросы. Прежний читатель immutable=1 -wal не читал вовсе, поэтому видел прежнее состояние
        независимо от того, какие страницы успел прочитать до записи."""
        real = recordings.open_store
        writers = []

        def opened(path=None, readonly=False):
            store = real(path, readonly=readonly)
            if readonly and not writers:
                writer = real(path)
                writers.append(writer)
                self.addCleanup(writer.close)
                action(writer)
            return store
        return mock.patch.object(recordings, "open_store", side_effect=opened)

    def test_recording_list_sees_recording_added_by_another_connection(self):
        with self.write_after_reader_opens(lambda writer: self.make_recording(writer, "Вторая.wav")):
            listed = core_screen.recording_list()
        self.assertEqual(listed["total"], 2)
        self.assertEqual(sorted(item["name"] for item in listed["items"]), ["Вторая.wav", "Первая.wav"])

    def test_recording_view_sees_new_text_revision(self):
        with self.write_after_reader_opens(lambda writer: self.change_text(writer, self.first)):
            view = core_screen.recording_view(self.first)
        self.assertEqual(view["recording"]["source_revision"], 2)

    def test_transcript_revisions_see_new_revision(self):
        store = storage.open_store(self.db)
        try:
            binding = recordings.document_binding(store, self.first)
        finally:
            store.close()
        json_path = Path(binding["root"]) / binding["paths"]["json"]
        with self.write_after_reader_opens(lambda writer: self.change_text(writer, self.first)):
            revisions = review_pack.transcript_revisions(json_path)
        self.assertEqual(revisions, {1, 2})

    def test_backup_inspection_stays_a_snapshot_without_traces(self):
        store = storage.open_store(self.db)
        try:
            store.backup(self.base / "backup.sqlite3")
        finally:
            store.close()
        before = names(self.base)
        info = core.inspect_backup(self.base / "backup.sqlite3")
        self.assertEqual(info["recordings"], 1)
        core.inspect_backup(self.base / "backup.sqlite3")      # повторное открытие резерва — тоже без следов
        self.assertEqual(names(self.base), before)
        self.assertNotIn("backup.sqlite3-shm", before)

    def test_working_database_given_as_backup_is_read_live(self):
        with mock.patch.object(storage, "open_snapshot_readonly",
                               side_effect=AssertionError("рабочая база прочитана как снимок")):
            info = core.inspect_backup(self.db)
        self.assertEqual(info["recordings"], 1)
        self.assertFalse(info["restore_guard_locked"])


@unittest.skipUnless(os.name == "nt", "замена существующей базы при восстановлении пока только в Windows")
class RestoreAfterLiveReader(unittest.TestCase):
    """После БД-2 пустой -wal и -shm у закрытой базы — обычное состояние (их оставляет чтение). Восстановление
    поверх такой базы проходит, а поздний отказ называет их точно, не как «внесённый журнал»."""

    def setUp(self):
        import core_conflicts
        self.core_conflicts = core_conflicts
        self.base = Path(tempfile.mkdtemp(prefix="live-restore-", dir=os.environ.get("TEMP")))
        self.addCleanup(shutil.rmtree, self.base, True)
        for name in ("r1", "r2"):
            (self.base / name).mkdir()
        self.db = self.base / "state.sqlite3"
        store = storage.initialize(self.db, initial_display_name="Тест")
        try:
            store.register_root(self.base / "r1")
            self.user_id = store.current_identity()["user_id"]
            self.backup = self.base / "b1.sqlite3"
            store.backup(self.backup, allow_damaged_copies=True)
            self.root_id = store.db.execute("SELECT root_id FROM allowed_roots").fetchone()[0]
        finally:
            store.close()
        add_users(self.db, 2)
        storage.open_store_readonly(self.db).close()      # чтение оставило служебные файлы
        self.assertIn("state.sqlite3-shm", names(self.base))

    def restore(self):
        return core.restore_from_backup(self.backup, self.db, {self.root_id: str(self.base / "r2")}, self.user_id,
                                        replace=True)

    def test_restore_over_database_left_by_reader(self):
        result = self.restore()
        saved = Path(result["previous_saved_as"])
        store = storage.open_snapshot_readonly(saved)
        try:
            self.assertEqual(users(store), 3)                 # прежняя база целиком в before-restore
        finally:
            store.close()
        restored = storage.open_store_readonly(self.db)
        try:
            self.assertEqual(users(restored), 1)              # восстановлен ровно резерв
        finally:
            restored.close()

    def test_late_refusal_names_reader_traces_not_journal(self):
        with mock.patch.object(self.core_conflicts, "_publish", side_effect=OSError("synthetic failure")):
            with self.assertRaises(core.CoreError) as caught:
                self.restore()
        message = str(caught.exception)
        self.assertIn("убраны служебные файлы, которые оставило чтение", message)
        self.assertNotIn("журнал прежней базы (-wal) внесён", message)
        store = storage.open_store_readonly(self.db)
        try:
            self.assertEqual(users(store), 3)                 # прежняя база на месте со всеми данными
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
