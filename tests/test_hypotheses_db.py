"""Гипотезы моделей в базе записей (Г-1): таблица hypotheses, очистка по записи, сторож удаления записи,
база до Г-1, резерв и проверка копии.

Только синтетика во временной папке: база записей — storage.initialize, записи гипотез — малые словари (их
форму проверяет tests/test_hypotheses_save.py). Пути базы и папки гипотез передаются явно; настоящая папка
данных не читается.
"""
import atexit
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-hyp-db-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP

import hypstore  # noqa: E402
import recordings  # noqa: E402
import storage  # noqa: E402
from tests.test_hypotheses_save import make_store, stored  # noqa: E402

SOUND = "abcdef012345"


def run_id(n):
    return f"20261126-1000{n:02d}-{n:06x}"


class HypothesesDbTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="hyp-db-", dir=_TMP))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.db = self.root / "state.sqlite3"
        self.hyp = self.root / "hypotheses"
        self.rec_a, self.rec_b = make_store(self.db)

    def put(self, n, recording_id, sound=SOUND, record=None):
        return hypstore.write_db(record if record is not None else {"schema": "hyp-2", "n": n}, sound, run_id(n),
                                 recording_id, store_path=self.db)

    def cleanup(self, **kw):
        return hypstore.cleanup(data_dir=self.hyp, store_path=self.db, **kw)

    def test_two_recordings_of_one_sound_lose_only_their_own(self):
        """Тот же звук у двух записей: очистка по записи удаляет только её гипотезы."""
        self.put(1, self.rec_a)
        self.put(2, self.rec_b)
        self.put(3, self.rec_a)
        self.assertEqual(self.cleanup(unit_id=SOUND)["database"]["runs"], 3)
        report = self.cleanup(recording_id=self.rec_a)
        self.assertEqual((report["database"]["runs"], report["database"]["recordings"], report["applied"]),
                         (2, 1, False))
        self.assertEqual(len(stored(self.db)), 3)                    # отчёт ничего не удалил
        applied = self.cleanup(recording_id=self.rec_a, apply=True)
        self.assertEqual(applied["database"], report["database"])   # удалено ровно показанное
        self.assertEqual([row["recording_id"] for row, _ in stored(self.db)], [self.rec_b])

    def test_report_equals_what_is_deleted_by_age(self):
        self.put(1, self.rec_a)
        self.assertEqual(self.cleanup(older_than_days=1)["database"]["runs"], 0)          # только что записано
        later = time.time() + 2 * 86400
        report = self.cleanup(older_than_days=1, now=later)
        self.assertEqual(report["database"]["runs"], 1)
        self.assertEqual(self.cleanup(older_than_days=1, now=later, apply=True)["database"], report["database"])
        self.assertEqual(stored(self.db), [])

    def test_summary_numbers_are_the_cleanup_numbers(self):
        """Объём в get_settings и self_check — те же числа, что у очистки без отбора (одна выборка)."""
        self.put(1, self.rec_a, record={"schema": "hyp-2", "text": "пример " * 50})
        self.put(2, self.rec_b)
        summary = hypstore.summary({"save_hypotheses": True}, data_dir=self.hyp, store_path=self.db)
        report = self.cleanup()
        self.assertTrue(summary["enabled"])
        share = summary["database"].pop("share_of_database")
        self.assertEqual(summary["database"], report["database"])
        self.assertEqual(summary["database"]["recordings"], 2)
        self.assertAlmostEqual(share, report["database"]["bytes"] / self.db.stat().st_size, places=4)
        self.assertEqual(summary["old_files"], {"files": 0, "bytes": 0, "units": 0})

    def test_summary_never_raises(self):
        broken = self.root / "broken.sqlite3"
        broken.write_bytes(b"not a database")
        summary = hypstore.summary({}, data_dir=self.hyp, store_path=broken)
        self.assertFalse(summary["enabled"])
        self.assertIn("error", summary["database"])
        self.assertEqual(hypstore.summary({}, data_dir=self.hyp, store_path=self.root / "нет.sqlite3")["database"],
                         {"runs": 0, "recordings": 0, "bytes": 0, "share_of_database": None})

    def test_deleting_a_recording_with_hypotheses_is_refused(self):
        """Сторож будущего удаления записи: внешний ключ таблицы гипотез не даёт удалить запись, пока у неё
        есть гипотезы; без гипотез таблица гипотез удалению не мешает."""
        self.put(1, self.rec_a)
        store = storage.open_store(self.db)
        try:
            fks = [tuple(r)[2:4] for r in store.db.execute("PRAGMA foreign_key_list(hypotheses)")]
            self.assertEqual(fks, [("recordings", "recording_id")])
            self.assertEqual(store.db.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            with self.assertRaises(sqlite3.IntegrityError):
                store.db.execute("DELETE FROM recordings WHERE recording_id=?", (self.rec_a,))
        finally:
            store.close()
        # кто именно мешает: на копии без проверки ключей запись удаляется, проверка таблицы гипотез находит сироту
        copy = self.root / "copy.sqlite3"
        shutil.copyfile(self.db, copy)
        db = sqlite3.connect(copy)
        try:
            db.execute("PRAGMA foreign_keys=OFF")
            for rid in (self.rec_a, self.rec_b):
                db.execute("DELETE FROM recordings WHERE recording_id=?", (rid,))
            orphans = db.execute("PRAGMA foreign_key_check(hypotheses)").fetchall()
        finally:
            db.close()
        self.assertEqual(len(orphans), 1)                    # гипотезы есть только у первой записи

    def test_run_is_written_once_and_ids_are_checked(self):
        self.put(1, self.rec_a)
        with self.assertRaises(storage.Conflict):
            self.put(1, self.rec_b)
        with self.assertRaises(KeyError):
            self.put(2, str(uuid.uuid4()))
        for args in ((SOUND, "x", self.rec_a), ("ABCDEF012345", run_id(3), self.rec_a), (SOUND, run_id(3), ""),
                     (SOUND, run_id(3), None)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                hypstore.write_db({}, *args, store_path=self.db)
        with self.assertRaises(ValueError):
            self.cleanup(recording_id="не-номер")
        self.assertEqual(len(stored(self.db)), 1)

    def test_store_before_g1_gets_backup_then_empty_table(self):
        with sqlite3.connect(self.db) as db:
            db.execute("DROP TABLE hypotheses")
        db.close()
        store = storage.open_store(self.db)                  # база без таблицы — допустимая схема 4
        try:
            self.assertFalse(store.has_hypotheses())
            self.assertEqual(store.hypotheses_summary(), {"runs": 0, "recordings": 0, "bytes": 0})
            self.assertEqual(store.hypotheses_rows(), [])
        finally:
            store.close()
        self.assertEqual(self.cleanup()["database"]["runs"], 0)
        with mock.patch.object(storage.Store, "backup", side_effect=OSError("диск полон")):
            with self.assertRaises(OSError):
                self.put(1, self.rec_a)
        store = storage.open_store(self.db)
        try:
            self.assertFalse(store.has_hypotheses())         # резерв не удался — таблицы нет
        finally:
            store.close()
        self.put(1, self.rec_a)
        backups = sorted((self.root / "db-backups").glob("state-before-hypotheses-*.sqlite3"))
        self.assertEqual(len(backups), 1)
        old = storage.open_store_readonly(backups[0])
        try:
            self.assertFalse(old.has_hypotheses())           # резерв — база как была до таблицы
        finally:
            old.close()
        self.put(2, self.rec_a)                              # таблица есть — второго резерва нет
        self.assertEqual(len(list((self.root / "db-backups").glob("state-before-hypotheses-*"))), 1)
        self.assertEqual(len(stored(self.db)), 2)

    def test_backup_keeps_hypotheses_and_copy_check_sees_damage(self):
        """Свой резерв базы (backup_state, rt.py db-backup) содержит гипотезы вместе с записью."""
        self.put(1, self.rec_a, record={"schema": "hyp-2", "text": "пример"})
        store = recordings.open_store(self.db)
        try:
            target = self.root / "backup.sqlite3"
            store.backup(target)
        finally:
            store.close()
        self.assertEqual([rec for _, rec in stored(target)], [{"schema": "hyp-2", "text": "пример"}])
        storage.validate_copy(target)
        for sql in ("UPDATE hypotheses SET size_bytes=size_bytes+1", "UPDATE hypotheses SET record_json='[1]', "
                    "size_bytes=3"):
            with self.subTest(sql=sql):
                bad = self.root / f"bad-{uuid.uuid4().hex[:6]}.sqlite3"
                shutil.copyfile(target, bad)
                with sqlite3.connect(bad) as db:
                    db.execute(sql)
                db.close()
                with self.assertRaises(storage.StorageError):
                    storage.validate_copy(bad)

    def test_old_files_are_cleaned_by_recording_inside(self):
        """Прежние файлы (до Г-1): отбор по записи — по ids.recording_id внутри файла."""
        a = hypstore.write({"schema": "hyp-2", "ids": {"recording_id": self.rec_a}}, SOUND, run_id(1), self.hyp)
        b = hypstore.write({"schema": "hyp-2", "ids": {"recording_id": {"status": "not_available"}}}, SOUND,
                           run_id(2), self.hyp)
        report = self.cleanup(recording_id=self.rec_a, apply=True)
        self.assertEqual((report["files"], report["database"]["runs"]), (1, 0))
        self.assertFalse(a.exists())
        self.assertTrue(b.exists())


if __name__ == "__main__":
    unittest.main()
