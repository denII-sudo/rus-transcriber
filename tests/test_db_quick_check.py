"""БД-4(а): quick_check главной базы один раз на процесс, запрет записи при провале, часть самопроверки.

Только синтетика во временной папке: база создаётся заново, страница портится записью байтов в закрытый файл.
"""
import atexit
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-quick-check-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:                       # данные программы — во временную папку до любого импорта
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import dbguard  # noqa: E402
import hypstore  # noqa: E402
import paths  # noqa: E402
import recordings  # noqa: E402
import selfcheck  # noqa: E402
import storage  # noqa: E402


def make_store(path, users=400):
    """Синтетическая база: строки пользователей занимают десятки страниц одного дерева."""
    store = storage.initialize(path, initial_display_name="Тест")
    store.db.execute("BEGIN")
    for i in range(users):
        store.db.execute("INSERT INTO local_users VALUES (?, ?, ?, ?)",
                         (f"00000000-0000-4000-8000-{i:012d}", "имя " * 60,
                          "2026-11-28T00:00:00+00:00", "2026-11-28T00:00:00+00:00"))
    store.db.execute("COMMIT")
    store.close()                                    # последнее соединение: журнал внесён, -wal/-shm убраны


def damage_page(path):
    """Испортить заголовок страницы дерева local_users (не первую страницу схемы): тип страницы становится
    недопустимым. Файл закрыт, служебных файлов рядом нет."""
    db = sqlite3.connect(path)
    page_size = db.execute("PRAGMA page_size").fetchone()[0]
    root = db.execute("SELECT rootpage FROM sqlite_master WHERE name='local_users'").fetchone()[0]
    pages = [row[0] for row in db.execute("SELECT pageno FROM dbstat WHERE name='local_users' AND pagetype='leaf'")] \
        if _has_dbstat(db) else []
    db.close()
    target = max(pages) if pages else root
    with open(path, "r+b") as f:
        f.seek((target - 1) * page_size)
        f.write(b"\x7f" * 16)


def _has_dbstat(db):
    try:
        db.execute("SELECT 1 FROM dbstat LIMIT 1").fetchall()
        return True
    except sqlite3.OperationalError:
        return False


class QuickCheckTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="qc-", dir=_TMP))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.main = self.base / "state.sqlite3"
        patcher = mock.patch.object(paths, "STATE_DB", self.main)
        patcher.start()
        self.addCleanup(patcher.stop)
        dbguard._forget_all()
        self.addCleanup(dbguard._forget_all)

    def test_intact_and_damaged_page(self):
        make_store(self.main)
        self.assertIs(dbguard.quick_check(self.main)["ok"], True)
        damage_page(self.main)
        result = dbguard.quick_check(self.main)
        self.assertIs(result["ok"], False)
        self.assertTrue(result["messages"])
        self.assertFalse(any(Path(str(self.main) + s).exists() for s in ("-wal", "-shm")), "проверка без следов")

    def test_not_a_database(self):
        self.main.write_bytes(b"not a database at all" * 200)
        self.assertIs(dbguard.quick_check(self.main)["ok"], False)

    def test_damaged_main_refuses_writes_but_reads(self):
        make_store(self.main)
        damage_page(self.main)
        before = self.main.read_bytes()
        with self.assertRaises(dbguard.Damaged) as caught:
            recordings.open_store()
        self.assertIn("не прошла проверку целостности", str(caught.exception))
        with self.assertRaises(dbguard.Damaged):
            recordings.open_store(self.main)                     # тот же файл явным путём
        with self.assertRaises(dbguard.Damaged):
            hypstore._database_part(None, None, None, True, self.main)
        readonly = recordings.open_store(readonly=True)          # чтение работает: схема на первой странице цела
        readonly.close()
        self.assertEqual(self.main.read_bytes(), before, "отказ ничего не пишет в базу")
        self.assertFalse(dbguard.view()["ok"])
        self.assertIn("quick_check", dbguard.view()["message"])

    def test_checked_once_per_file(self):
        make_store(self.main)
        with mock.patch.object(dbguard, "quick_check", wraps=dbguard.quick_check) as checked:
            recordings.open_store().close()
            recordings.open_store().close()
        self.assertEqual(checked.call_count, 1)

    def test_other_database_file_is_not_checked(self):
        other = self.base / "other.sqlite3"
        make_store(other)
        with mock.patch.object(dbguard, "quick_check") as checked:
            recordings.open_store(other).close()
        checked.assert_not_called()
        self.assertIsNone(dbguard.ensure_checked(other))

    def test_new_file_at_the_same_path_is_checked_again(self):
        """Восстановление публикует другой файл под тем же именем: прежний провал к нему не относится."""
        make_store(self.main)
        damage_page(self.main)
        with self.assertRaises(dbguard.Damaged):
            recordings.open_store()
        good = self.base / "good.sqlite3"
        make_store(good)
        self.main.unlink()
        os.rename(good, self.main)
        recordings.open_store().close()

    def test_busy_is_not_remembered(self):
        make_store(self.main)
        with mock.patch.object(dbguard, "quick_check", return_value={"ok": None, "reason": "busy", "messages": [],
                                                                       "seconds": 0}) as checked:
            recordings.open_store().close()
            recordings.open_store().close()
        self.assertEqual(checked.call_count, 2)
        self.assertIsNone(dbguard.state())

    def test_self_check_part(self):
        self.assertEqual(selfcheck.run_database()["state"], "absent")
        make_store(self.main)
        report = selfcheck.run(tests=False, repo=False, morph=False)
        self.assertTrue(report["database"]["ok"])
        self.assertTrue(report["ok"])
        damage_page(self.main)
        report = selfcheck.run(tests=False, repo=False, morph=False)
        self.assertEqual(report["database"]["state"], "damaged")
        self.assertFalse(report["ok"])
        self.assertIn("database", report["summary"])
        self.assertNotIn(str(self.base), str(report["database"]), "без путей")
        self.assertNotIn("database", selfcheck.run(tests=False, repo=False, morph=False, database=False))


if __name__ == "__main__":
    unittest.main()
