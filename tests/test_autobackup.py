"""БД-4(б): автоматический резерв раз в рабочий день, прореживание И-1а, замена копии дня, защита от испорченных.

Эталон прореживания — таблицы 2.1 (N = 9) и 2.2 (N = 8) из ops `research/2026-11-28-sqlite-audit/opus-plan-2-answers.md`
(вывод `thinning_sim_3.py`). Только синтетика во временной папке; часы и пояс подменяются, время записей
задаётся строками базы.
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-autobackup-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:                       # данные программы — во временную папку до любого импорта
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import autobackup  # noqa: E402
import dbguard  # noqa: E402
import paths  # noqa: E402
import selfcheck  # noqa: E402
import storage  # noqa: E402

TZ = timezone(timedelta(hours=3))                    # пояс компьютера в тестах: Москва
TABLE_N9 = {9: "1 2 3 4 5 6 7 8 9", 10: "1 3 4 5 6 7 8 9 10", 11: "1 3 5 6 7 8 9 10 11",
            12: "1 3 5 7 8 9 10 11 12", 13: "1 3 5 7 9 10 11 12 13", 14: "1 3 5 7 9 11 12 13 14",
            15: "1 3 5 7 9 11 13 14 15", 16: "1 3 5 7 9 11 14 15 16", 17: "1 3 5 7 9 11 15 16 17",
            18: "1 3 5 7 9 11 16 17 18", 19: "1 3 5 7 9 11 17 18 19"}
TABLE_N8 = {8: "1 2 3 4 5 6 7 8", 9: "1 3 4 5 6 7 8 9", 10: "1 3 5 6 7 8 9 10", 11: "1 3 5 7 8 9 10 11",
            12: "1 3 5 7 9 10 11 12", 13: "1 3 5 7 9 11 12 13", 14: "1 3 5 7 9 12 13 14", 15: "1 3 5 7 9 13 14 15"}


def local(day, hour, minute=0, second=0):
    """Местное время TZ -> datetime с поясом."""
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=TZ)


class WorkdayTests(unittest.TestCase):
    def test_boundary_at_four(self):
        day = date(2030, 3, 5)
        self.assertEqual(autobackup.workday(local(day, 3, 59, 59), TZ), "2030-03-04")
        self.assertEqual(autobackup.workday(local(day, 4, 0, 0), TZ), "2030-03-05")
        self.assertEqual(autobackup.workday(local(day, 0, 40), TZ), "2030-03-04")
        self.assertEqual(autobackup.workday(local(day, 23, 59), TZ), "2030-03-05")

    def test_time_zone_of_the_computer_counts(self):
        moment = datetime(2030, 3, 5, 1, 30, tzinfo=timezone.utc)          # 04:30 в Москве, 20:30 накануне в Нью-Йорке
        self.assertEqual(autobackup.workday(moment, TZ), "2030-03-05")
        self.assertEqual(autobackup.workday(moment, timezone(timedelta(hours=-5))), "2030-03-04")
        self.assertEqual(autobackup.workday("2030-03-05T01:30:00+00:00", TZ), "2030-03-05")
        self.assertEqual(autobackup.workday("2030-03-05T01:30:00", TZ), "2030-03-05", "время без пояса — UTC")


class FakeCopies:
    """Опись и файлы копий одинакового размера: прореживание без базы."""

    def __init__(self, case, size=1000):
        self.base = Path(tempfile.mkdtemp(prefix="rot-", dir=_TMP))
        case.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.folder = autobackup.Folder(self.base / "state.sqlite3")
        self.folder.prepare()
        self.manifest = self.folder.load_manifest()
        self.size = size
        self.questions = []

    def step(self, number, limit):
        day = (date(2030, 1, 1) + timedelta(days=number)).isoformat()
        name = f"state-{day}.sqlite3"
        body = (f"копия {number} ".encode() * self.size)[:self.size]
        (self.folder.path / name).write_bytes(body)
        self.manifest["copies"].append({"name": name, "workday": day, "size": self.size, "step": number,
                                        "last_write": f"{day}T12:00:00+00:00",
                                        "sha256": autobackup._sha256(self.folder.path / name)})
        self.folder.save_manifest(self.manifest)
        self.questions.append(autobackup.rotate(self.folder, self.manifest, limit))
        return [entry["step"] for entry in self.manifest["copies"]]


class RotationTests(unittest.TestCase):
    def run_table(self, n, table, steps):
        copies = FakeCopies(self)
        for number in range(1, steps + 1):
            kept = copies.step(number, n * copies.size)
            if number in table:
                self.assertEqual(" ".join(map(str, kept)), table[number], f"N={n}, шаг {number}")
        files = sorted(p.name for p in copies.folder.path.glob("state-*.sqlite3"))
        self.assertEqual(files, sorted(entry["name"] for entry in copies.manifest["copies"]), "файлы = опись")
        return copies

    def test_reference_table_n9(self):
        copies = self.run_table(9, TABLE_N9, 19)
        self.assertEqual([q for q in copies.questions[:15] if q], [], "до шага 2N − 2 вопроса нет")
        self.assertEqual(copies.questions[15], {"kind": "limit"}, "вопрос — с шага 16")

    def test_reference_table_n8(self):
        copies = self.run_table(8, TABLE_N8, 15)
        self.assertIsNone(copies.questions[12])
        self.assertEqual(copies.questions[13], {"kind": "limit"}, "вопрос — с шага 14")

    def test_oldest_and_three_newest_always_kept(self):
        for n in range(4, 11):
            copies = FakeCopies(self)
            for number in range(1, 6 * n + 1):
                kept = copies.step(number, n * copies.size)
                self.assertEqual(kept[0], 1, (n, number))
                self.assertLessEqual(len(kept), n, (n, number))
                if number >= 3:
                    self.assertEqual(kept[-3:], list(range(number - 2, number + 1)), (n, number))

    def test_count_limit_ten(self):
        copies = FakeCopies(self)
        for number in range(1, 30):
            kept = copies.step(number, 10 ** 12)
        self.assertEqual(len(kept), 10)
        self.assertEqual(kept[0], 1)

    def test_small_limit_keeps_four_protected_and_asks(self):
        """Ответ владельца на В-1: самая старая и три свежие хранятся сверх предела, человеку — сразу вопрос."""
        copies = FakeCopies(self)
        for number in range(1, 8):
            kept = copies.step(number, 3 * copies.size)
        self.assertEqual(kept, [1, 5, 6, 7])
        self.assertEqual(copies.questions[-1], {"kind": "small_limit"})

    def test_limit_lowered_mid_pass(self):
        copies = FakeCopies(self)
        for number in range(1, 12):
            kept = copies.step(number, 9 * copies.size)
        kept = copies.step(12, 6 * copies.size)                       # база выросла: помещается меньше копий
        self.assertLessEqual(len(kept), 6)
        self.assertEqual((kept[0], kept[-3:]), (1, [10, 11, 12]))

    def test_delete_refuses_foreign_names_links_and_changed_files(self):
        copies = FakeCopies(self)
        copies.step(1, 10 ** 9)
        entry = copies.manifest["copies"][0]
        with self.assertRaises(autobackup.ManifestError):
            copies.folder.delete_copy("notes.txt", 1, "0" * 64)
        with self.assertRaises(autobackup.ManifestError):
            copies.folder.delete_copy(entry["name"], entry["size"], "0" * 64)
        self.assertTrue((copies.folder.path / entry["name"]).exists())


class DailyTests(unittest.TestCase):
    """Шаг при запуске на настоящей синтетической базе (backup API, validate_copy)."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="daily-", dir=_TMP))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.main = self.base / "state.sqlite3"
        patcher = mock.patch.object(paths, "STATE_DB", self.main)
        patcher.start()
        self.addCleanup(patcher.stop)
        dbguard._forget_all()
        self.addCleanup(dbguard._forget_all)
        storage.initialize(self.main, initial_display_name="Тест").close()
        self.folder = autobackup.Folder(self.main)
        self.config = {"db_backup_auto": True, "db_backup_max_gb": 1.0}
        self.users = 0

    def write(self, moment):
        """Запись в базу с заданным временем (строка пользователя: свой столбец времени)."""
        self.users += 1
        store = storage.open_store(self.main)
        try:
            stamp = moment.astimezone(timezone.utc).isoformat()
            store.db.execute("INSERT INTO local_users VALUES (?, ?, ?, ?)",
                             (f"00000000-0000-4000-8000-{self.users:012d}", f"u{self.users}", stamp, stamp))
        finally:
            store.close()

    def launch(self, moment, **extra):
        return autobackup.start_daily(now=moment, tz=TZ, background=False, config={**self.config, **extra})

    def manifest(self):
        return json.loads(self.folder.manifest_path.read_text(encoding="utf-8"))

    def names(self):
        return [entry["name"] for entry in self.manifest()["copies"]]

    def files(self):
        return sorted(p.name for p in self.folder.path.iterdir() if p.is_file())

    def test_first_launch_of_workday_dates_copy_by_last_write(self):
        monday = date(2030, 3, 4)
        self.write(local(monday, 20, 0))
        self.write(local(monday + timedelta(days=1), 0, 40))          # ночь вторника — ещё понедельник
        result = self.launch(local(monday + timedelta(days=1), 9, 0))
        self.assertEqual(result["state"], "added")
        self.assertEqual(self.names(), ["state-2030-03-04.sqlite3"])
        copy = self.folder.path / "state-2030-03-04.sqlite3"
        opened = storage.open_store_readonly(copy)
        try:
            self.assertEqual(opened.db.execute("SELECT locked FROM restore_guard").fetchone()[0], 1)
        finally:
            opened.close()
        self.assertEqual(self.launch(local(monday + timedelta(days=1), 15, 0))["state"], "done_today")
        self.assertFalse([p for p in self.folder.path.iterdir() if ".incomplete-" in p.name])

    def test_launch_at_two_after_copy_of_the_same_workday(self):
        monday = date(2030, 3, 4)
        self.write(local(monday, 10, 0))
        self.assertEqual(self.launch(local(monday, 12, 0))["state"], "added")
        self.write(local(monday, 23, 0))
        self.assertEqual(self.launch(local(monday + timedelta(days=1), 2, 0))["state"], "done_today",
                         "02:00 вторника — ещё рабочий день понедельника: копия посреди ночного сеанса не снимается")

    def test_unchanged_base_is_skipped(self):
        monday = date(2030, 3, 4)
        self.write(local(monday, 10, 0))
        self.launch(local(monday, 12, 0))
        self.assertEqual(self.launch(local(monday + timedelta(days=1), 9, 0))["state"], "unchanged")
        self.assertEqual(len(self.names()), 1)

    def test_same_workday_later_write_replaces_copy(self):
        monday = date(2030, 3, 4)
        self.write(local(monday, 10, 0))
        self.launch(local(monday, 12, 0))
        first = self.manifest()["copies"][0]["sha256"]
        self.write(local(monday + timedelta(days=1), 3, 59))         # 03:59 — ещё понедельник
        result = self.launch(local(monday + timedelta(days=1), 9, 0))
        self.assertEqual(result["state"], "replaced")
        copies = self.manifest()["copies"]
        self.assertEqual([c["name"] for c in copies], ["state-2030-03-04.sqlite3"])
        self.assertNotEqual(copies[0]["sha256"], first)
        self.assertEqual(self.files(), [".lock", "manifest.json", "state-2030-03-04.sqlite3", "status.json"])
        self.write(local(monday + timedelta(days=1), 4, 0))          # 04:00 — уже вторник
        self.assertEqual(self.launch(local(monday + timedelta(days=2), 9, 0))["state"], "added")
        self.assertEqual(self.names(), ["state-2030-03-04.sqlite3", "state-2030-03-05.sqlite3"])

    def test_crash_between_rename_and_publish_keeps_previous_copy(self):
        monday = date(2030, 3, 4)
        self.write(local(monday, 10, 0))
        self.launch(local(monday, 12, 0))
        previous = self.manifest()["copies"][0]
        self.write(local(monday, 18, 0))

        class Crash(BaseException):
            pass
        with mock.patch.object(storage, "_publish_new", side_effect=Crash), self.assertRaises(Crash):
            self.launch(local(monday + timedelta(days=1), 9, 0))
        asides = [p for p in self.folder.path.iterdir() if ".replaced-" in p.name]
        self.assertEqual(len(asides), 1, "прежняя копия цела под вторым именем")
        self.assertEqual(autobackup._sha256(asides[0]), previous["sha256"])
        self.assertIn("replacing", self.manifest()["copies"][0], "в описи она действующая")
        result = self.launch(local(monday + timedelta(days=1), 10, 0))  # следующий запуск: возврат на место и шаг
        self.assertEqual(result["state"], "replaced")
        self.assertFalse([p for p in self.folder.path.iterdir() if ".replaced-" in p.name])
        self.assertEqual(self.names(), ["state-2030-03-04.sqlite3"])
        self.assertNotIn("replacing", self.manifest()["copies"][0])

    def test_crash_after_publish_is_completed_by_next_launch(self):
        monday = date(2030, 3, 4)
        self.write(local(monday, 10, 0))
        self.launch(local(monday, 12, 0))
        self.write(local(monday, 18, 0))
        real = self.folder.save_manifest
        calls = []

        class Crash(BaseException):
            pass

        def save(manifest):
            calls.append(1)
            if len(calls) == 2:                                       # 1 — «заменяется», 2 — после публикации
                raise Crash
            real(manifest)
        with mock.patch.object(autobackup.Folder, "save_manifest", side_effect=lambda m: save(m)), \
                self.assertRaises(Crash):
            self.launch(local(monday + timedelta(days=1), 9, 0))
        self.assertEqual(self.launch(local(monday + timedelta(days=1), 10, 0))["state"], "unchanged",
                         "опубликованная копия доведена по описи; новых записей нет")
        copies = self.manifest()["copies"]
        self.assertEqual(len(copies), 1)
        self.assertEqual(autobackup._sha256(self.folder.path / copies[0]["name"]), copies[0]["sha256"])
        self.assertFalse([p for p in self.folder.path.iterdir() if ".replaced-" in p.name])

    def test_rollback_of_time_makes_copy_suspect_and_deletes_nothing(self):
        monday = date(2030, 3, 4)
        self.write(local(monday, 10, 0))
        self.write(local(monday, 11, 0))
        self.launch(local(monday, 12, 0))
        store = storage.open_store(self.main)
        try:                                     # «тихая порча»: последняя запись исчезла, строение файла цело
            store.db.execute("UPDATE local_users SET created_at=?, updated_at=? WHERE display_name='u2'",
                             ("2030-03-03T08:00:00+00:00",) * 2)
            store.db.execute("UPDATE local_users SET display_name='u2b' WHERE display_name='u1'")
        finally:
            store.close()
        self.write(local(monday, 9, 0))           # новая запись раньше прежней последней
        before = self.names()
        result = self.launch(local(monday + timedelta(days=1), 9, 0))
        self.assertEqual(result["state"], "suspect")
        self.assertEqual(self.names(), before)
        self.assertEqual(len(list(self.folder.suspect_path.iterdir())), 1)
        view = autobackup.status_view()
        self.assertTrue(any("подозрительная" in line for line in view["lines"]))

    def test_damaged_base_takes_no_copy(self):
        monday = date(2030, 3, 4)
        self.write(local(monday, 10, 0))
        self.launch(local(monday, 12, 0))
        self.write(local(monday + timedelta(days=1), 10, 0))
        with mock.patch.object(dbguard, "quick_check", return_value={"ok": False, "messages": ["x"], "seconds": 0}):
            dbguard._forget_all()
            self.assertEqual(self.launch(local(monday + timedelta(days=2), 9, 0))["state"], "damaged")
        self.assertEqual(len(self.names()), 1)
        self.assertTrue(any("целостности" in line for line in autobackup.status_view()["lines"]))

    def test_broken_manifest_stops_without_deleting(self):
        monday = date(2030, 3, 4)
        self.write(local(monday, 10, 0))
        self.launch(local(monday, 12, 0))
        self.folder.manifest_path.write_text("{не json", encoding="utf-8")
        self.write(local(monday + timedelta(days=1), 10, 0))
        self.assertEqual(self.launch(local(monday + timedelta(days=2), 9, 0))["state"], "failed")
        self.assertTrue((self.folder.path / "state-2030-03-04.sqlite3").exists())
        self.assertTrue(any("опись" in line for line in autobackup.status_view()["lines"]))

    def test_rotation_on_real_copies_n_plus_two_and_foreign_files(self):
        start = date(2030, 3, 4)
        self.write(local(start, 10, 0))
        self.launch(local(start, 12, 0))
        size = self.manifest()["copies"][0]["size"]
        limit = (size * 5 + size // 2) / autobackup.GIB                 # помещается 5 копий
        foreign = {"notes.txt": b"note", "state-2030-01-01.sqlite3.bak": b"bak"}
        for name, body in foreign.items():
            (self.folder.path / name).write_bytes(body)
        other = self.main.parent / "db-backups" / "state-before-hypotheses-20300101T000000Z.sqlite3"
        other.write_bytes(b"other kind of backup")
        for number in range(1, 7):                                    # всего 7 копий = N + 2
            day = start + timedelta(days=number)
            self.write(local(day, 10, 0))
            self.launch(local(day, 12, 0), db_backup_max_gb=limit)
        days = [entry["workday"][-2:] for entry in self.manifest()["copies"]]
        self.assertEqual(days, ["04", "06", "08", "09", "10"], "И-1а при N = 5: шаги 1 3 5 6 7")
        for name, body in foreign.items():
            self.assertEqual((self.folder.path / name).read_bytes(), body)
        self.assertEqual(other.read_bytes(), b"other kind of backup")

    def test_second_process_skips_while_locked(self):
        self.write(local(date(2030, 3, 4), 10, 0))
        holder = autobackup.Folder(self.main)
        holder.prepare()
        self.assertTrue(holder.try_lock())
        try:
            self.assertEqual(self.launch(local(date(2030, 3, 4), 12, 0))["state"], "locked")
        finally:
            holder.unlock()
        self.assertEqual(self.launch(local(date(2030, 3, 4), 12, 5))["state"], "added")

    def test_snapshot_is_taken_before_start_returns(self):
        """Снимок — синхронно при запуске (до первой записи процесса и переходов схемы), доводка — в фоне."""
        self.write(local(date(2030, 3, 4), 10, 0))
        import threading
        gate, seen = threading.Event(), {}
        real = storage.finish_snapshot

        def slow(staged, **kwargs):
            seen["size"] = Path(staged).stat().st_size
            gate.wait(10)
            return real(staged, **kwargs)
        with mock.patch.object(storage, "finish_snapshot", side_effect=slow):
            result = autobackup.start_daily(now=local(date(2030, 3, 4), 12, 0), tz=TZ, config=self.config)
            self.assertEqual(result["state"], "started")
            staged = [p for p in self.folder.path.iterdir() if ".incomplete-" in p.name]
            self.assertEqual(len(staged), 1)
            self.assertGreater(staged[0].stat().st_size, 0)
            self.write(local(date(2030, 3, 4), 13, 0))               # запись после старта в копию не попадает
            gate.set()
            for thread in threading.enumerate():
                if thread.name == "autobackup":
                    thread.join(30)
        self.assertEqual(self.manifest()["copies"][0]["last_write"], local(date(2030, 3, 4), 10, 0)
                         .astimezone(timezone.utc).isoformat())

    def test_question_and_answers(self):
        copies_dir = self.folder
        copies_dir.prepare()
        status = {"question": {"kind": "limit", "count": 9, "oldest_workday": "2030-03-01", "limit_gb": 1.0,
                               "needed_gb": 1.2}}
        copies_dir.save_status(status)
        copies_dir.save_manifest({"version": 1, "copies": [], "thin_pointer": 7, "thin_done": True, "suspect": []})
        view = autobackup.status_view()
        self.assertTrue(any("достигли предела 1 ГБ" in line for line in view["lines"]))
        self.assertEqual(autobackup.answer("keep")["answered"], "keep")
        self.assertIsNotNone(autobackup.status_view()["question"], "«оставить» — спросить при следующем запуске")
        autobackup.answer("thin")
        manifest = self.manifest()
        self.assertEqual((manifest["thin_done"], manifest["thin_pointer"]), (False, 1))
        self.assertIsNone(autobackup.status_view()["question"])
        with mock.patch("settings.update") as update, \
                mock.patch("settings.load_config", return_value={"db_backup_max_gb": 1.0}):
            with self.assertRaises(ValueError):
                autobackup.answer("raise", 0.5)
            autobackup.answer("raise", 2)
        update.assert_called_once_with(db_backup_max_gb=2.0)
        with self.assertRaises(ValueError):
            autobackup.answer("delete-everything")

    def test_self_check_lines(self):
        self.write(local(date(2030, 3, 4), 10, 0))
        self.launch(local(date(2030, 3, 4), 12, 0))
        report = selfcheck.run(tests=False, repo=False, morph=False)
        self.assertEqual(report["database"]["autobackup"]["copies"], 1)
        self.assertTrue(report["ok"])
        self.write(local(date(2030, 3, 12), 10, 0))
        lines = autobackup.self_check_report(tz=TZ)["lines"]
        self.assertTrue(any("нет уже 8 дн." in line for line in lines), lines)

    def test_switched_off(self):
        self.write(local(date(2030, 3, 4), 10, 0))
        self.assertEqual(self.launch(local(date(2030, 3, 4), 12, 0), db_backup_auto=False)["state"], "off")
        self.assertFalse(self.folder.path.exists())


if __name__ == "__main__":
    unittest.main()
