"""В-2: копия самой свежей проверенной автокопии на другом диске (automirror), решение владельца 28.11.2026.

Только синтетика во временной папке. «Другой диск» здесь — другая временная папка того же тома: проверяется и
предупреждение «копия не независима». Часы и пояс подменяются, как в test_autobackup.
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-automirror-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:                       # данные программы — во временную папку до любого импорта
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import autobackup  # noqa: E402
import automirror  # noqa: E402
import dbguard  # noqa: E402
import paths  # noqa: E402
import settings  # noqa: E402
import storage  # noqa: E402

TZ = timezone(timedelta(hours=3))


def local(day, hour, minute=0):
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=TZ)


def join_threads(name):
    for thread in threading.enumerate():
        if thread.name == name:
            thread.join(30)


class MirrorTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="data-", dir=_TMP))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.other = Path(tempfile.mkdtemp(prefix="other-disk-", dir=_TMP))
        self.addCleanup(shutil.rmtree, self.other, ignore_errors=True)
        self.main = self.base / "state.sqlite3"
        patcher = mock.patch.object(paths, "STATE_DB", self.main)
        patcher.start()
        self.addCleanup(patcher.stop)
        dbguard._forget_all()
        self.addCleanup(dbguard._forget_all)
        storage.initialize(self.main, initial_display_name="Тест").close()
        self.folder = autobackup.Folder(self.main)
        self.mirror = self.other / automirror.SUBFOLDER
        self.config = {"db_backup_auto": True, "db_backup_max_gb": 1.0, "db_backup_mirror_dir": str(self.other)}
        self.users = 0

    def write(self, moment):
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

    def auto_manifest(self):
        return json.loads(self.folder.manifest_path.read_text(encoding="utf-8"))

    def mirror_manifest(self):
        return json.loads((self.mirror / "manifest.json").read_text(encoding="utf-8"))

    def mirror_status(self):
        return self.folder.load_status().get("mirror", {})

    def lines(self, moment=None):
        return autobackup.status_view(now=moment, tz=TZ)["lines"]

    def test_off_by_default_and_without_setting(self):
        self.assertEqual(settings.DEFAULTS["db_backup_mirror_dir"], "")
        self.write(local(date(2030, 3, 4), 10, 0))
        result = self.launch(local(date(2030, 3, 4), 12, 0), db_backup_mirror_dir="")
        self.assertEqual(result["state"], "added")
        self.assertNotIn("mirror", result)
        self.assertEqual(list(self.other.iterdir()), [])
        self.assertNotIn("mirror", self.folder.load_status())

    def test_newest_verified_copy_goes_to_the_other_disk(self):
        self.write(local(date(2030, 3, 4), 10, 0))
        result = self.launch(local(date(2030, 3, 4), 12, 0))
        self.assertEqual((result["state"], result["mirror"]), ("added", "added"))
        source = self.auto_manifest()["copies"][0]
        copy = self.mirror / source["name"]
        self.assertEqual(storage._file_sha256(copy), source["sha256"])
        manifest = self.mirror_manifest()
        self.assertEqual(manifest["owner"], self.auto_manifest()["mirror_owner"])
        self.assertEqual([(c["name"], c["sha256"]) for c in manifest["copies"]], [(source["name"], source["sha256"])])
        self.assertFalse([p for p in self.mirror.iterdir() if ".incomplete-" in p.name])
        self.assertIn("last_success", self.mirror_status())
        self.assertEqual(self.launch(local(date(2030, 3, 4), 15, 0))["mirror"], "current")

    def test_same_volume_is_warned(self):
        self.write(local(date(2030, 3, 4), 10, 0))
        self.launch(local(date(2030, 3, 4), 12, 0))
        self.assertTrue(any("на том же диске" in line for line in self.lines()), self.lines())

    def test_unplugged_disk_is_deferred_then_caught_up(self):
        unplugged = self.other / "внешний"
        self.config["db_backup_mirror_dir"] = str(unplugged)
        self.mirror = unplugged / automirror.SUBFOLDER
        day = date(2030, 3, 4)
        self.write(local(day, 10, 0))
        self.assertEqual(self.launch(local(day, 12, 0))["mirror"], "unavailable")
        for later in (2, 3):
            self.write(local(day + timedelta(days=later - 1), 10, 0))
            self.launch(local(day + timedelta(days=later - 1), 12, 0))
            self.assertFalse([line for line in self.lines(local(day + timedelta(days=later - 1), 12, 0))
                              if "другом диске" in line], "до трёх рабочих дней — без неудачи")
        fourth = local(day + timedelta(days=3), 12, 0)
        self.assertTrue(any("не делается 3 дн." in line for line in self.lines(fourth)), self.lines(fourth))
        self.write(local(day + timedelta(days=3), 10, 0))
        self.launch(fourth)                                            # всё ещё отключён: копия дня в auto есть
        unplugged.mkdir()                                              # подключили позже в тот же день
        result = self.launch(local(day + timedelta(days=3), 15, 0))
        self.assertEqual((result["state"], result["mirror"]), ("done_today", "added"))
        newest = autobackup._newest(self.auto_manifest())
        self.assertEqual([c["sha256"] for c in self.mirror_manifest()["copies"]], [newest["sha256"]],
                         "на другой диск идёт только самая свежая копия")
        self.assertFalse([line for line in self.lines(fourth) if "не делается" in line])

    def test_copy_of_the_same_workday_is_replaced_there_too(self):
        monday = date(2030, 3, 4)
        self.write(local(monday, 10, 0))
        self.launch(local(monday, 12, 0))
        self.write(local(monday + timedelta(days=1), 3, 0))           # ещё понедельник
        result = self.launch(local(monday + timedelta(days=1), 9, 0))
        self.assertEqual((result["state"], result["mirror"]), ("replaced", "replaced"))
        source = self.auto_manifest()["copies"][0]
        self.assertEqual([(c["name"], c["sha256"]) for c in self.mirror_manifest()["copies"]],
                         [(source["name"], source["sha256"])])
        self.assertEqual(sorted(p.name for p in self.mirror.iterdir()), ["manifest.json", source["name"]])

    def test_same_rule_thins_the_other_disk_and_keeps_foreign_files(self):
        start = date(2030, 3, 4)
        self.write(local(start, 10, 0))
        self.launch(local(start, 12, 0))
        size = self.auto_manifest()["copies"][0]["size"]
        limit = (size * 5 + size // 2) / autobackup.GIB              # помещается 5 копий
        foreign = {self.other / "мои заметки.txt": b"note", self.other / "state-2030-01-01.sqlite3": b"manual",
                   self.mirror / "notes.txt": b"inside"}
        for path, body in foreign.items():
            path.write_bytes(body)
        for number in range(1, 7):
            day = start + timedelta(days=number)
            self.write(local(day, 10, 0))
            self.launch(local(day, 12, 0), db_backup_max_gb=limit)
        days = [entry["workday"][-2:] for entry in self.mirror_manifest()["copies"]]
        self.assertEqual(days, ["04", "06", "08", "09", "10"], "то же правило И-1а, что в auto")
        self.assertEqual(days, [entry["workday"][-2:] for entry in self.auto_manifest()["copies"]])
        for path, body in foreign.items():
            self.assertEqual(path.read_bytes(), body)

    def test_foreign_subfolder_is_not_touched(self):
        self.mirror.mkdir()
        theirs = {"version": 1, "owner": "f" * 32, "copies": [], "thin_pointer": 1, "thin_done": False,
                  "suspect": []}
        (self.mirror / "manifest.json").write_text(json.dumps(theirs), encoding="utf-8")
        (self.mirror / "state-2030-01-01.sqlite3").write_bytes(b"their copy")
        self.write(local(date(2030, 3, 4), 10, 0))
        result = self.launch(local(date(2030, 3, 4), 12, 0))
        self.assertEqual((result["state"], result["mirror"], result["kind"]), ("added", "failed", "foreign"))
        self.assertEqual(json.loads((self.mirror / "manifest.json").read_text(encoding="utf-8")), theirs)
        self.assertEqual((self.mirror / "state-2030-01-01.sqlite3").read_bytes(), b"their copy")
        self.assertTrue(any("другой папки данных" in line for line in self.lines()))

    def test_changed_source_is_not_copied(self):
        self.write(local(date(2030, 3, 4), 10, 0))
        self.launch(local(date(2030, 3, 4), 12, 0), db_backup_mirror_dir="")
        source = self.folder.path / self.auto_manifest()["copies"][0]["name"]
        body = bytearray(source.read_bytes())
        body[-1] ^= 0xFF                                               # тот же размер, другие байты
        source.write_bytes(bytes(body))
        result = self.launch(local(date(2030, 3, 4), 15, 0))
        self.assertEqual((result["mirror"], result["kind"]), ("failed", "copy"))
        self.assertEqual(sorted(p.name for p in self.mirror.iterdir()), [], "ни копии, ни временного файла")
        self.assertTrue(any("не удалось скопировать" in line for line in self.lines()))

    def test_place_rules(self):
        data = self.base
        with self.assertRaises(automirror.Failure):
            automirror.check_place(str(data / "db-backups"), data)    # внутри папки данных
        (data / "inner").mkdir()
        with self.assertRaises(automirror.Failure):
            automirror.check_place(str(data / "inner"), data)
        with self.assertRaises(automirror.Failure):
            automirror.check_place(str(data.parent), data)            # содержит папку данных
        with self.assertRaises(automirror.Failure):
            automirror.check_place("relative\\folder", data)
        (self.other / "file.txt").write_bytes(b"x")
        with self.assertRaises(automirror.Failure):
            automirror.check_place(str(self.other / "file.txt"), data)
        with self.assertRaises(automirror.Unavailable):
            automirror.check_place(str(self.other / "missing"), data)
        root, warnings = automirror.check_place(str(self.other), data)
        self.assertEqual((root, warnings), (self.other, ["same_volume"]))
        with mock.patch.dict(os.environ, {"OneDrive": str(self.other.parent)}):
            self.assertIn("cloud", automirror.check_place(str(self.other), data)[1])

    def test_screen_sets_checks_and_switches_off(self):
        self.addCleanup(settings.update, db_backup_mirror_dir="")
        self.write(local(date(2030, 3, 4), 10, 0))
        autobackup.start_daily(tz=TZ, background=False, config={**self.config, "db_backup_mirror_dir": ""})
        with self.assertRaises(ValueError):
            automirror.set_dir(str(self.base / "db-backups"))
        with self.assertRaises(ValueError):
            automirror.set_dir(str(self.other / "missing"))
        self.assertEqual(settings.load_config()["db_backup_mirror_dir"], "", "отказ — настройка не записана")
        answer = automirror.set_dir(str(self.other))
        join_threads("automirror")
        self.assertEqual(answer["step"], "started")
        self.assertEqual(settings.load_config()["db_backup_mirror_dir"], str(self.other))
        self.assertTrue(any("том же диске" in line for line in answer["warnings"]))
        self.assertEqual(len(self.mirror_manifest()["copies"]), 1, "копия сразу после выбора папки")
        self.assertEqual(automirror.screen_view()["dir"], str(self.other))
        self.assertEqual(automirror.set_dir("")["step"], "off")
        self.assertEqual(settings.load_config()["db_backup_mirror_dir"], "")
        self.assertNotIn("mirror", self.folder.load_status())
        self.assertTrue((self.mirror / "manifest.json").is_file(), "выключение копии там ничего не удаляет")

    def test_question_of_the_other_disk_and_thin_answer(self):
        self.write(local(date(2030, 3, 4), 10, 0))
        self.launch(local(date(2030, 3, 4), 12, 0))
        status = self.folder.load_status()
        status["mirror"]["question"] = {"kind": "limit", "count": 9, "oldest_workday": "2030-03-01",
                                        "limit_gb": 1.0, "needed_gb": 1.2}
        self.folder.save_status(status)
        manifest = self.mirror_manifest()
        manifest["thin_pointer"], manifest["thin_done"] = 5, True
        (self.mirror / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        view = autobackup.status_view()
        self.assertEqual(view["question"]["kind"], "limit")
        self.assertTrue(any("на другом диске достигли предела" in line for line in view["lines"]))
        with mock.patch("settings.load_config", return_value=self.config):
            autobackup.answer("thin")
        self.assertEqual((self.mirror_manifest()["thin_pointer"], self.mirror_manifest()["thin_done"]), (1, False))
        self.assertIsNone(autobackup.status_view()["question"])

    def test_self_check_part(self):
        self.write(local(date(2030, 3, 4), 10, 0))
        self.launch(local(date(2030, 3, 4), 12, 0))
        part = autobackup.self_check_report(tz=TZ)["mirror"]
        self.assertTrue(part["enabled"])
        self.assertIsNotNone(part["last_success"])
        self.assertNotIn(str(self.other), json.dumps(autobackup.self_check_report(tz=TZ), ensure_ascii=False),
                         "в самопроверке нет путей")


if __name__ == "__main__":
    unittest.main()
