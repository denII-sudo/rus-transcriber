"""storage._check_schema на испорченных базах: что принимается и что отказывается (и открытие/осмотр/проверка копии с ним)."""
import atexit
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="rt-schema-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:                       # данные программы — во временную папку до любого импорта
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import storage  # noqa: E402
from storage import SCHEMA_VERSION, UnsupportedSchema  # noqa: E402


def change(path, *statements):
    db = sqlite3.connect(path, isolation_level=None)
    try:
        db.execute("PRAGMA foreign_keys=OFF")
        for sql in statements:
            db.execute(sql)
    finally:
        db.close()


def verdict(path):
    """None — база подходит; иначе класс отказа."""
    try:
        db = storage._connect_readonly(path)
    except Exception as exc:                                    # не база: отказ уже при открытии
        return type(exc).__name__
    try:
        storage._check_schema(db)
        return None
    except Exception as exc:
        return type(exc).__name__
    finally:
        db.close()


class CheckSchemaCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rt-schema-case-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def make(self, name, *statements):
        path = self.base / (name + ".sqlite3")
        storage.initialize(path, initial_display_name="Тест").close()
        for side in self.base.glob(name + ".sqlite3-*"):
            side.unlink()
        if statements:
            change(path, *statements)
        return path

    def test_every_case_is_accepted_or_refused_as_expected(self):
        no_result = ("DROP TABLE workflow_steps",
                     "CREATE TABLE workflow_steps (step_id TEXT PRIMARY KEY, run_id TEXT, kind TEXT, "
                     "state TEXT, created_at TEXT, updated_at TEXT)")
        view = ("DROP TABLE workflow_steps",
                "CREATE VIEW workflow_steps AS SELECT '' AS step_id, '' AS run_id, '' AS kind, '' AS state, "
                "'' AS result_json, '' AS created_at, '' AS updated_at")
        text = self.base / "not_a_db.sqlite3"
        text.write_text("это не база данных" * 50, encoding="utf-8")
        empty = self.base / "empty.sqlite3"
        empty.write_bytes(b"")
        cases = [
            ("нормальная", self.make("ok"), None),
            ("нет таблицы", self.make("no_table", "DROP TABLE workflow_steps"), "UnsupportedSchema"),
            ("нет столбца", self.make("no_column", *no_result), "UnsupportedSchema"),
            ("view вместо таблицы", self.make("view", *view), "UnsupportedSchema"),
            ("чужая версия схемы", self.make("version", f"UPDATE store_meta SET schema_version={SCHEMA_VERSION + 1}"),
             "UnsupportedSchema"),
            ("две строки версии", self.make("two_rows", f"INSERT INTO store_meta VALUES ('other-store', {SCHEMA_VERSION})"),
             "UnsupportedSchema"),
            ("не база", text, "UnsupportedSchema"),
            ("пустой файл", empty, "UnsupportedSchema"),
            # допускаются сегодня; тест фиксирует это, чтобы изменение было заметным, а не случайным
            ("лишняя таблица", self.make("extra_table", "CREATE TABLE extra_stuff (x)"), None),
            ("лишний столбец", self.make("extra_column", "ALTER TABLE workflow_steps ADD COLUMN extra TEXT"), None),
            ("чужой user_version (не проверяется)", self.make("user_version", "PRAGMA user_version=99"), None),
        ]
        for label, path, expected in cases:
            with self.subTest(label):
                self.assertEqual(verdict(path), expected)

    def test_open_store_readonly_and_validate_copy_refuse_the_same_database(self):
        path = self.make("refuse", "DROP TABLE workflow_steps")
        for opener in (storage.open_store, storage.open_store_readonly, storage.validate_copy):
            with self.subTest(opener.__name__):
                with self.assertRaises(UnsupportedSchema):
                    opener(path)


if __name__ == "__main__":
    unittest.main()
