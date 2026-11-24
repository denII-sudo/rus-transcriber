"""Шаблон пустой базы для тестов (tests/dbtemplate.py): копия равна свежей базе, копии независимы."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import dbtemplate
import recordings
import storage


def dump(store):
    """Схема и содержимое: строки таблиц без идентификаторов и времени (они у каждой свежей базы свои)."""
    schema = [tuple(r) for r in store.db.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY name")]
    counts, rows = {}, {}
    for table in [r[1] for r in schema if r[0] == "table" and not r[1].startswith("sqlite_")]:
        counts[table] = store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        names = [c[1] for c in store.db.execute(f"PRAGMA table_info({table})")]
        keep = [n for n in names if not (n.endswith("_id") or n.endswith("_at") or n in ("created", "updated", "ts"))]
        if keep and counts[table]:
            rows[table] = sorted(tuple(r) for r in store.db.execute(f"SELECT {', '.join(keep)} FROM {table}"))
    return schema, counts, rows


class DbTemplateTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="rt-dbtpl-test-"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)

    def open_pair(self, name="Тест"):
        copy = dbtemplate.initialize(self.base / "copy.sqlite3", name)
        self.addCleanup(copy.close)
        fresh = recordings.initialize_store(self.base / "fresh.sqlite3", display_name=name)
        self.addCleanup(fresh.close)
        return copy, fresh

    def test_copy_equals_a_freshly_created_database(self):
        copy, fresh = self.open_pair()
        self.assertEqual(dump(copy), dump(fresh))
        self.assertEqual(copy.current_identity()["display_name"], "Тест")
        self.assertEqual(copy.db.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        self.assertEqual(copy.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        storage.validate_copy(self.base / "copy.sqlite3")                       # та же полная проверка, что у резерва

    def test_copy_leaves_no_side_files_and_template_is_not_the_copy(self):
        copy, _ = self.open_pair()
        copy.close()
        self.assertEqual(sorted(p.name for p in self.base.glob("copy.sqlite3*")), ["copy.sqlite3"])
        self.assertNotEqual(Path(dbtemplate._template("Тест")), self.base / "copy.sqlite3")

    def test_copies_are_independent_of_each_other_and_of_the_template(self):
        first = dbtemplate.initialize(self.base / "a.sqlite3")
        second = dbtemplate.initialize(self.base / "b.sqlite3")
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        first.db.execute("INSERT INTO local_users (user_id, display_name, created_at, updated_at) "
                         "VALUES ('extra-user', 'Другой', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')")
        counts = lambda s: s.db.execute("SELECT count(*) FROM local_users").fetchone()[0]
        self.assertEqual(counts(first), counts(second) + 1)
        third = dbtemplate.initialize(self.base / "c.sqlite3")
        self.addCleanup(third.close)
        self.assertEqual(counts(third), counts(second))                          # шаблон не тронут записью в копию

    def test_the_documented_difference_copies_share_identifiers_fresh_databases_do_not(self):
        a = dbtemplate.initialize(self.base / "a.sqlite3")
        b = dbtemplate.initialize(self.base / "b.sqlite3")
        fresh = recordings.initialize_store(self.base / "fresh.sqlite3", display_name="Тест")
        for s in (a, b, fresh):
            self.addCleanup(s.close)
        self.assertEqual(a.current_identity(), b.current_identity())
        self.assertNotEqual(a.current_identity()["user_id"], fresh.current_identity()["user_id"])

    def test_existing_file_is_not_replaced_and_each_name_has_its_own_template(self):
        (self.base / "x.sqlite3").write_bytes(b"not a database")
        with self.assertRaises(FileExistsError):
            dbtemplate.initialize(self.base / "x.sqlite3")
        self.assertEqual((self.base / "x.sqlite3").read_bytes(), b"not a database")
        named = dbtemplate.initialize(self.base / "n.sqlite3", "Инициатор")
        self.addCleanup(named.close)
        self.assertEqual(named.current_identity()["display_name"], "Инициатор")

    def test_template_lives_in_this_process_temp_dir_and_the_schema_tests_still_build_from_scratch(self):
        self.assertTrue(Path(dbtemplate._template("Тест")).is_file())
        self.assertIn(os.path.normcase(tempfile.gettempdir()), os.path.normcase(str(dbtemplate._state["dir"])))
        # проверки схемы, initialize и резерва создают базу сами, не через шаблон
        for name in ("test_storage_recovery.py", "test_storage_revisions.py", "test_p6_acceptance.py"):
            text = (Path(__file__).resolve().parent / name).read_text(encoding="utf-8")
            self.assertNotIn("dbtemplate", text, name)


if __name__ == "__main__":
    unittest.main()
