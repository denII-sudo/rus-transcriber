"""Чтение снимка (резерв, своя копия) «только на чтение» не должно оставлять рядом служебные файлы -wal/-shm.

База всегда в режиме WAL. Соединение mode=ro на закрытой базе создаёт -wal/-shm и при
закрытии не может их убрать; рядом с резервной копией остались бы лишние файлы (а на каталоге
только для чтения открытие вообще может не удаться). Поэтому снимок (open_snapshot_readonly,
validate_copy по умолчанию) открывается неизменяемым. Это верно только для закрытой базы без
служебных файлов; если -wal/-shm уже есть, меняется или создаётся -shm (второй класс ниже).
Живая база (open_store_readonly) с БД-2 читается обычным mode=ro и может оставить -wal/-shm:
это проверяет tests/test_storage_live_reader.py. Все базы здесь синтетические и лежат во временной папке.
"""
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import core
import storage


def side_files(directory):
    return sorted(p.name for p in Path(directory).iterdir())


class ReadonlyLeavesNoTraces(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ro-side-", dir=os.environ.get("TEMP"))
        self.base = Path(self.temp.name)
        self.path = self.base / "state.sqlite3"
        storage.initialize(self.path, initial_display_name="Тест").close()
        # Исходное состояние: закрытая база без служебных файлов.
        self.assertEqual(side_files(self.base), ["state.sqlite3"])

    def tearDown(self):
        self.temp.cleanup()

    def test_open_snapshot_readonly_leaves_no_side_files(self):
        store = storage.open_snapshot_readonly(self.path)
        try:
            store.db.execute("SELECT count(*) FROM recordings").fetchone()
        finally:
            store.close()
        self.assertEqual(side_files(self.base), ["state.sqlite3"])

    def test_validate_copy_leaves_no_side_files(self):
        storage.validate_copy(self.path)
        self.assertEqual(side_files(self.base), ["state.sqlite3"])

    def test_inspect_backup_leaves_no_side_files(self):
        backup = self.base / "backup.sqlite3"
        store = storage.open_store(self.path)
        try:
            store.current_identity()
            store.backup(backup)
        finally:
            store.close()
        before = side_files(self.base)
        self.assertEqual(before, ["backup.sqlite3", "state.sqlite3"])
        core.inspect_backup(backup)
        self.assertEqual(side_files(self.base), before)

    def test_readonly_does_not_change_database_bytes(self):
        before = self.path.read_bytes()
        storage.open_store_readonly(self.path).close()
        storage.validate_copy(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_readonly_still_rejects_writes(self):
        store = storage.open_store_readonly(self.path)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                store.db.execute("UPDATE restore_guard SET locked=1 WHERE id=1")
        finally:
            store.close()

    def test_readonly_sees_uncheckpointed_data_of_live_writer(self):
        # Пока писатель держит базу открытой, свежие данные лежат в -wal. Читатель обязан их
        # видеть: режим immutable=1 их игнорирует, поэтому его нельзя применять при живом -wal.
        writer = storage.open_store(self.path)
        try:
            writer.db.execute("UPDATE local_users SET display_name='ИЗМЕНЕНО'")
            self.assertIn("state.sqlite3-wal", side_files(self.base))
            reader = storage.open_store_readonly(self.path)
            try:
                name = reader.db.execute("SELECT display_name FROM local_users").fetchone()[0]
            finally:
                reader.close()
            self.assertEqual(name, "ИЗМЕНЕНО")
        finally:
            writer.close()
        # Служебные файлы писателя читатель не должен ни оставить, ни удалить раньше времени.
        self.assertEqual(side_files(self.base), ["state.sqlite3"])


def file_bytes(directory):
    return {p.name: p.read_bytes() for p in Path(directory).iterdir()}


class ReadonlyWithServiceFiles(unittest.TestCase):
    """Что именно меняется рядом с базой, если -wal/-shm уже есть (см. _connect_snapshot_readonly).

    Читатель WAL всегда отмечается в -shm, поэтому «без следов» верно только для закрытой
    базы. Здесь зафиксировано реальное поведение: основной файл и -wal не меняются, данные
    записей не затрагиваются; меняется (или создаётся) только -shm.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ro-wal-", dir=os.environ.get("TEMP"))
        self.base = Path(self.temp.name)
        self.live = self.base / "live"
        self.live.mkdir()
        self.path = self.live / "state.sqlite3"
        storage.initialize(self.path, initial_display_name="Тест").close()

    def tearDown(self):
        self.temp.cleanup()

    def read_name(self, path):
        store = storage.open_store_readonly(path)
        try:
            return store.db.execute("SELECT display_name FROM local_users").fetchone()[0]
        finally:
            store.close()

    def copy_of_live_files(self, writer, names):
        """Копия файлов базы при открытом писателе: имитация состояния после сбоя."""
        crash = self.base / ("crash-" + "-".join(n.rsplit("-", 1)[-1] for n in names))
        crash.mkdir()
        for name in names:
            shutil.copy2(self.live / name, crash / name)
        return crash / "state.sqlite3"

    def test_live_writer_changes_only_shm(self):
        writer = storage.open_store(self.path)
        try:
            writer.db.execute("UPDATE local_users SET display_name='ЖИВОЙ'")
            before = file_bytes(self.live)
            self.assertEqual(sorted(before),
                             ["state.sqlite3", "state.sqlite3-shm", "state.sqlite3-wal"])
            self.assertEqual(self.read_name(self.path), "ЖИВОЙ")
            after = file_bytes(self.live)
        finally:
            writer.close()
        self.assertEqual(sorted(after), sorted(before))
        self.assertEqual(after["state.sqlite3"], before["state.sqlite3"])
        self.assertEqual(after["state.sqlite3-wal"], before["state.sqlite3-wal"])

    def test_after_crash_wal_and_shm_change_only_shm(self):
        writer = storage.open_store(self.path)
        try:
            writer.db.execute("UPDATE local_users SET display_name='ПОСЛЕ СБОЯ'")
            copy = self.copy_of_live_files(
                writer, ["state.sqlite3", "state.sqlite3-wal", "state.sqlite3-shm"])
        finally:
            writer.close()
        before = file_bytes(copy.parent)
        self.assertEqual(self.read_name(copy), "ПОСЛЕ СБОЯ")
        after = file_bytes(copy.parent)
        self.assertEqual(sorted(after), sorted(before))
        self.assertEqual(after["state.sqlite3"], before["state.sqlite3"])
        self.assertEqual(after["state.sqlite3-wal"], before["state.sqlite3-wal"])

    def test_only_wal_creates_shm_and_data_is_visible(self):
        writer = storage.open_store(self.path)
        try:
            writer.db.execute("UPDATE local_users SET display_name='ТОЛЬКО WAL'")
            copy = self.copy_of_live_files(writer, ["state.sqlite3", "state.sqlite3-wal"])
        finally:
            writer.close()
        before = file_bytes(copy.parent)
        self.assertEqual(sorted(before), ["state.sqlite3", "state.sqlite3-wal"])
        # Данные из -wal видны (immutable=1 их бы не увидел), но создаётся новый -shm:
        # это известное ограничение, а не обещание «без следов».
        self.assertEqual(self.read_name(copy), "ТОЛЬКО WAL")
        after = file_bytes(copy.parent)
        self.assertEqual(sorted(after), ["state.sqlite3", "state.sqlite3-shm", "state.sqlite3-wal"])
        self.assertEqual(after["state.sqlite3"], before["state.sqlite3"])
        self.assertEqual(after["state.sqlite3-wal"], before["state.sqlite3-wal"])


if __name__ == "__main__":
    unittest.main()
