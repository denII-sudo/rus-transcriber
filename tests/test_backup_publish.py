"""БД-7: публикация резерва без жёстких ссылок, съёмный диск за настройкой человека, фоновый резерв экрана.

Только синтетика во временной папке. Том без жёстких ссылок и съёмный диск подменяются (отказ os.link с кодом
Windows, тип диска): настоящего FAT/exFAT и флешки в CI нет.
"""
import atexit
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-backup-publish-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:                       # данные программы — во временную папку до любого импорта
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import core  # noqa: E402
import core_conflicts  # noqa: E402
import dbguard  # noqa: E402
import paths  # noqa: E402
import settings  # noqa: E402
import storage  # noqa: E402
import workflow_page  # noqa: E402

WINDOWS = os.name == "nt"


def no_links_error():
    """Отказ CreateHardLinkW на томе FAT/exFAT: ERROR_INVALID_FUNCTION."""
    if WINDOWS:
        return OSError(0, "Incorrect function", None, 1)
    error = OSError(1, "Operation not permitted")
    return error


class Base(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="pub-", dir=_TMP))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.main = self.base / "state.sqlite3"
        patcher = mock.patch.object(paths, "STATE_DB", self.main)
        patcher.start()
        self.addCleanup(patcher.stop)
        dbguard._forget_all()
        self.addCleanup(dbguard._forget_all)
        self.store = storage.initialize(self.main, initial_display_name="Тест")
        self.addCleanup(self.store.close)

    def leftovers(self, folder):
        return sorted(p.name for p in Path(folder).iterdir() if ".incomplete-" in p.name)

    def assert_backup(self, path):
        copy = storage.open_store_readonly(path)
        try:
            self.assertEqual(copy.db.execute("SELECT locked FROM restore_guard").fetchone()[0], 1)
        finally:
            copy.close()


class PublishTests(Base):
    @unittest.skipUnless(WINDOWS, "переименование без замены — свойство Windows")
    def test_volume_without_hard_links_publishes_by_rename(self):
        target = self.base / "fat" / "copy.sqlite3"
        target.parent.mkdir()
        with mock.patch("storage.os.link", side_effect=no_links_error()) as link:
            report = self.store.backup(target)
        link.assert_called_once()
        self.assertEqual(report["integrity"], "ok")
        self.assert_backup(target)
        self.assertEqual(self.leftovers(target.parent), [])

    @unittest.skipUnless(WINDOWS, "переименование без замены — свойство Windows")
    def test_rename_never_replaces_existing_file(self):
        staged = self.base / "staged.bin"
        staged.write_bytes(b"new")
        destination = self.base / "existing.bin"
        destination.write_bytes(b"old")
        with mock.patch("storage.os.link", side_effect=no_links_error()), self.assertRaises(FileExistsError):
            storage._publish_new(staged, destination)
        self.assertEqual(destination.read_bytes(), b"old")
        self.assertEqual(staged.read_bytes(), b"new")

    @unittest.skipIf(WINDOWS, "вне Windows rename заменяет молча — там отказ")
    def test_posix_volume_without_links_is_refused(self):
        staged = self.base / "staged.bin"
        staged.write_bytes(b"new")
        with mock.patch("storage.os.link", side_effect=no_links_error()), self.assertRaises(OSError):
            storage._publish_new(staged, self.base / "copy.bin")
        self.assertFalse((self.base / "copy.bin").exists())

    def test_other_link_errors_are_not_hidden(self):
        target = self.base / "copy.sqlite3"
        with mock.patch("storage.os.link", side_effect=OSError("injected")), self.assertRaises(OSError):
            self.store.backup(target)
        self.assertFalse(target.exists())
        self.assertEqual(self.leftovers(self.base), [])


class RemovableTests(Base):
    def setUp(self):
        super().setUp()
        self.stick = self.base / "stick"
        self.stick.mkdir()
        stick = os.path.normcase(str(self.stick.resolve()))
        self.drive = mock.patch.object(
            storage, "_drive_type",
            side_effect=lambda p: 2 if os.path.normcase(str(Path(p).resolve())).startswith(stick) else 3)
        self.drive.start()
        self.addCleanup(self.drive.stop)

    @unittest.skipUnless(WINDOWS, "тип диска проверяется только в Windows")
    def test_removable_refused_by_default_with_reason(self):
        with self.assertRaises(ValueError) as caught:
            core_conflicts.backup_store(self.stick / "copy.sqlite3")
        self.assertIn("db_backup_allow_removable", str(caught.exception))
        self.assertEqual(list(self.stick.iterdir()), [])
        with self.assertRaises(ValueError) as other:          # папка данных на съёмном диске — как прежде
            storage.check_local_path(self.stick / "state.sqlite3")
        self.assertNotIn("Резерв", str(other.exception))

    def test_removable_allowed_by_setting_copies_through_local_folder(self):
        stages = []
        with mock.patch.object(settings, "load_config", return_value={"db_backup_allow_removable": True}):
            result = core_conflicts.backup_store(self.stick / "copy.sqlite3", on_stage=stages.append)
        self.assertEqual(result["report"]["integrity"], "ok")
        local = self.base / "check.sqlite3"                    # программа не открывает базы на съёмном носителе
        shutil.copyfile(self.stick / "copy.sqlite3", local)
        self.assert_backup(local)
        self.assertEqual(stages, ["copy", "verify", "transfer", "publish"])
        self.assertEqual(self.leftovers(self.stick), [])
        self.assertEqual(self.leftovers(self.base), [])

    def test_changed_bytes_on_the_drive_leave_nothing(self):
        with mock.patch.object(storage, "_file_sha256", side_effect=["a" * 64, "b" * 64]), \
                self.assertRaises(storage.StorageError):
            self.store.backup(self.stick / "copy.sqlite3", allow_removable=True)
        self.assertEqual(list(self.stick.iterdir()), [])
        self.assertEqual(self.leftovers(self.base), [])


class ScreenTaskTests(unittest.TestCase):
    def test_background_backup_reports_stages_and_result(self):
        release = threading.Event()
        seen = []

        def fake(destination, on_stage=None):
            on_stage("copy")
            seen.append(destination)
            release.wait(10)
            on_stage("verify")
            return {"destination": destination, "report": {"integrity": "ok", "schema_version": 4}}

        tasks = workflow_page.BackupTasks()
        with mock.patch.object(core, "backup_store", side_effect=fake):
            started = tasks.start("C:/b.sqlite3")
            self.assertEqual(started["state"], "running")
            with self.assertRaises(core.CoreError):
                tasks.start("C:/c.sqlite3")                      # один резерв за раз
            release.set()
            done = tasks.wait(started["task_id"])
        self.assertEqual(done["state"], "done")
        self.assertEqual(done["result"]["report"]["integrity"], "ok")
        self.assertEqual(seen, ["C:/b.sqlite3"])

    def test_background_failure_is_explained(self):
        tasks = workflow_page.BackupTasks()
        with mock.patch.object(core, "backup_store", side_effect=ValueError(storage.REMOVABLE_REFUSED)):
            failed = tasks.wait(tasks.start("E:/b.sqlite3")["task_id"])
        self.assertEqual(failed["state"], "failed")
        self.assertIn("db_backup_allow_removable", failed["error"])
        with self.assertRaises(core.CoreError):
            tasks.status("нет-такого")
        with mock.patch.object(core, "backup_store", side_effect=storage.Busy("database is busy")):
            busy = tasks.wait(tasks.start("C:/b.sqlite3")["task_id"])
        self.assertIn("занята", busy["error"])


if __name__ == "__main__":
    unittest.main()
