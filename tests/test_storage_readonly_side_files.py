"""Чтение базы «только на чтение» не должно оставлять рядом служебные файлы -wal/-shm.

База всегда в режиме WAL. Соединение mode=ro на закрытой базе создаёт -wal/-shm и при
закрытии не может их убрать; в папке данных или рядом с резервной копией остаются лишние
файлы (а на каталоге только для чтения открытие вообще может не удаться).
Все базы здесь синтетические и лежат во временной папке.
"""
import os
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

    def test_open_store_readonly_leaves_no_side_files(self):
        store = storage.open_store_readonly(self.path)
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


if __name__ == "__main__":
    unittest.main()
