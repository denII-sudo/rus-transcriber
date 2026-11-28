"""БД-8а и БД-3а: детектор облачных и сетевых папок (cloudpaths.py), предупреждения и версия SQLite в self_check.

Системные вызовы (FindFirstFileW, GetDriveTypeW, CfGetSyncRootInfoByPath, реестр, info.json Dropbox) подменяются;
настоящие облачные папки не проверяются. Положительный контроль на Windows — настоящая временная папка: «локальная»,
CfGetSyncRootInfoByPath отвечает «не корень синхронизации»."""
import atexit
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
# как во всех модулях тестов: временная папка данных задаётся ДО первого импорта программы
_DATA_TMP = tempfile.mkdtemp(prefix="rt-data-")
os.environ["RUS_TRANSCRIBER_DATA"] = _DATA_TMP
atexit.register(shutil.rmtree, _DATA_TMP, ignore_errors=True)
sys.path.insert(0, str(ROOT))

import cloudpaths  # noqa: E402
import selfcheck  # noqa: E402

DIR = 0x10
CLOUD_TAG = 0x9000701A          # OneDrive
YANDEX_TAG = 0x9000301A
PINNED = 0x80000


class FakeWindows:
    """Подмена признаков: записи каталогов, тип диска, корни синхронизации, папки клиентов, Cloud Files API."""

    def __init__(self, entries=None, drive=3, sync_roots=(), client_folders=(), cf=False):
        self.entries = {str(PureWindowsPath(k)).casefold(): v for k, v in (entries or {}).items()}
        self.drive, self.roots, self.folders, self.cf = drive, list(sync_roots), list(client_folders), cf
        self.cf_calls, self.find_calls = [], []

    def find(self, path):
        self.find_calls.append(str(path))
        return self.entries.get(str(path).casefold())

    def cf_call(self, directory):
        self.cf_calls.append(str(directory))
        if isinstance(self.cf, Exception):
            raise self.cf
        return self.cf

    def patches(self):
        return [mock.patch.object(cloudpaths, "_find_entry", self.find),
                mock.patch.object(cloudpaths, "_drive_type", lambda anchor: self.drive),
                mock.patch.object(cloudpaths, "_sync_roots", lambda: self.roots),
                mock.patch.object(cloudpaths, "_client_folders", lambda: self.folders),
                mock.patch.object(cloudpaths, "_cf_sync_root", self.cf_call)]

    def classify(self, path):
        for p in self.patches():
            p.start()
        try:
            return cloudpaths._classify_windows(PureWindowsPath(path))
        finally:
            mock.patch.stopall()


def dirs(*paths, **special):
    out = {p: (DIR, 0) for p in paths}
    out.update({k.replace("__", "\\"): v for k, v in special.items()})
    return out


class TestSigns(unittest.TestCase):
    def test_plain_local_folder_is_local_and_cf_asked_for_nearest_directory(self):
        fake = FakeWindows(dirs(r"C:\Home", r"C:\Home\u", r"C:\Home\u\data"))
        r = fake.classify(r"C:\Home\u\data\backup.sqlite3")       # файла ещё нет
        self.assertEqual(r, {"kind": "local", "client": None, "basis": None})
        self.assertEqual(fake.cf_calls, [r"C:\Home\u\data"])

    def test_cloud_files_tag_on_a_parent(self):
        entries = dirs(r"C:\Home", r"C:\Home\u", r"C:\Home\u\Sync\data")
        entries[r"C:\Home\u\Sync"] = (DIR | 0x400, CLOUD_TAG)
        fake = FakeWindows(entries)
        r = fake.classify(r"C:\Home\u\Sync\data")
        self.assertEqual((r["kind"], r["basis"], r["client"]), ("cloud", "tag", None))
        self.assertEqual(fake.cf_calls, [], "Cloud Files API не нужен, если тег уже найден")

    def test_tag_with_client_name_from_registry(self):
        entries = dirs(r"C:\Home", r"C:\Home\u")
        entries[r"C:\Home\u\OneDrive"] = (DIR | 0x400, CLOUD_TAG)
        fake = FakeWindows(entries, client_folders=[("OneDrive", r"C:\Home\u\OneDrive")])
        r = fake.classify(r"C:\Home\u\OneDrive\rus-transcriber-data")
        self.assertEqual((r["kind"], r["basis"], r["client"]), ("cloud", "tag", "OneDrive"))

    def test_recall_and_pin_flags_without_tag(self):
        entries = dirs(r"D:\x")
        entries[r"D:\x\y"] = (DIR | PINNED, 0)
        self.assertEqual(FakeWindows(entries).classify(r"D:\x\y\z")["basis"], "tag")

    def test_other_reparse_tags_are_not_cloud(self):
        entries = dirs(r"D:\x")
        entries[r"D:\x\link"] = (DIR | 0x400, 0xA0000003)        # junction
        self.assertEqual(FakeWindows(entries).classify(r"D:\x\link")["kind"], "local")

    def test_sync_root_from_registry(self):
        fake = FakeWindows(dirs(r"E:\Cloud", r"E:\Cloud\data"), sync_roots=[("Яндекс Диск", r"E:\Cloud")])
        r = fake.classify(r"E:\cloud\DATA")                       # регистр букв не важен
        self.assertEqual((r["kind"], r["basis"], r["client"]), ("cloud", "sync_root", "Яндекс Диск"))

    def test_registry_path_with_environment_variable(self):
        fake = FakeWindows(sync_roots=[("OneDrive", r"%RT_TEST_HOME%\OneDrive")])
        with mock.patch.dict(os.environ, {"RT_TEST_HOME": r"C:\Home\u"}):
            r = fake.classify(r"C:\Home\u\OneDrive\data")
        self.assertEqual((r["kind"], r["basis"], r["client"]), ("cloud", "sync_root", "OneDrive"))

    def test_old_yandex_client_folder_without_tag_o8(self):
        fake = FakeWindows(dirs(r"D:\YD", r"D:\YD\data"), client_folders=[("Яндекс Диск", r"D:\YD")])
        r = fake.classify(r"D:\YD\data")
        self.assertEqual((r["kind"], r["basis"], r["client"]), ("cloud", "client_folder", "Яндекс Диск"))

    def test_sibling_with_common_prefix_is_not_under_root(self):
        fake = FakeWindows(dirs(r"D:\Docs2", r"D:\Docs2\data"), client_folders=[("Dropbox", r"D:\Docs")])
        self.assertEqual(fake.classify(r"D:\Docs2\data")["kind"], "local")

    def test_cloud_files_api(self):
        fake = FakeWindows(dirs(r"F:\a", r"F:\a\b"), cf=True)
        r = fake.classify(r"F:\a\b")
        self.assertEqual((r["kind"], r["basis"]), ("cloud", "cf_api"))
        self.assertEqual(fake.cf_calls, [r"F:\a\b"])

    def test_cf_not_called_when_nothing_exists(self):
        fake = FakeWindows({}, cf=True)
        self.assertEqual(fake.classify(r"F:\missing\b")["kind"], "local")
        self.assertEqual(fake.cf_calls, [])

    def test_folder_name_is_a_weak_sign(self):
        for path, client in ((r"C:\Home\u\Dropbox\data", "Dropbox"), (r"C:\Home\u\Яндекс.Диск\x", "Яндекс Диск"),
                             (r"C:\Home\u\OneDrive - Фирма\x", "OneDrive"), (r"G:\Мой диск\x", "Google Drive")):
            r = FakeWindows().classify(path)
            self.assertEqual((r["kind"], r["basis"], r["client"]), ("cloud", "name", client), path)

    def test_unc_and_network_drive(self):
        fake = FakeWindows()
        self.assertEqual(fake.classify(r"\\server\share\data"), {"kind": "network", "client": None, "basis": "unc"})
        self.assertEqual(fake.find_calls, [])
        self.assertEqual(FakeWindows(drive=4).classify(r"Z:\data")["basis"], "drive")

    def test_failing_signs_do_not_stop_the_others(self):
        def boom(*_a):
            raise OSError("access denied")
        fake = FakeWindows(dirs(r"D:\YD"), client_folders=[("Яндекс Диск", r"D:\YD")], cf=OSError("x"))
        with mock.patch.object(cloudpaths, "_sync_roots", boom):
            for p in fake.patches()[:2] + fake.patches()[3:]:
                p.start()
            try:
                r = cloudpaths._classify_windows(PureWindowsPath(r"D:\YD\data"))
                local = cloudpaths._classify_windows(PureWindowsPath(r"D:\plain"))
            finally:
                mock.patch.stopall()
        self.assertEqual(r["basis"], "client_folder")
        self.assertEqual(local["kind"], "local")


class TestClassify(unittest.TestCase):
    def setUp(self):
        cloudpaths.clear_cache()
        self.addCleanup(cloudpaths.clear_cache)

    @unittest.skipUnless(os.name == "nt", "Windows")
    def test_result_is_cached_per_path(self):
        with mock.patch.object(cloudpaths, "_classify_windows",
                               return_value={"kind": "local", "client": None, "basis": None}) as inner:
            cloudpaths.classify(_DATA_TMP)
            cloudpaths.classify(_DATA_TMP)
            self.assertEqual(inner.call_count, 1)
            cloudpaths.clear_cache()
            cloudpaths.classify(_DATA_TMP)
            self.assertEqual(inner.call_count, 2)

    @unittest.skipUnless(os.name == "nt", "Windows")
    def test_detector_failure_is_unknown_not_an_error(self):
        with mock.patch.object(cloudpaths, "_classify_windows", side_effect=RuntimeError("сбой")):
            self.assertEqual(cloudpaths.classify(_DATA_TMP)["kind"], "unknown")
            self.assertIsNone(cloudpaths.warning(_DATA_TMP, "data"))

    @unittest.skipIf(os.name == "nt", "вне Windows")
    def test_other_systems_are_unknown_for_now(self):
        self.assertEqual(cloudpaths.classify(_DATA_TMP)["kind"], "unknown")

    @unittest.skipUnless(os.name == "nt", "положительный контроль — настоящие вызовы Windows")
    def test_real_windows_calls_on_temp_folder(self):
        tmp = Path(tempfile.mkdtemp(prefix="rt-cloud-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        entry = cloudpaths._find_entry(tmp)
        self.assertIsNotNone(entry)
        self.assertTrue(entry[0] & DIR)
        self.assertIsNone(cloudpaths._find_entry(tmp / "нет-такого"))
        self.assertEqual(cloudpaths._drive_type(tmp.anchor), 3)
        self.assertIs(cloudpaths._cf_sync_root(tmp), False, "временная папка — не корень синхронизации")
        self.assertIsInstance(cloudpaths._sync_roots(), list)
        self.assertIsInstance(cloudpaths._client_folders(), list)
        self.assertEqual(cloudpaths.classify(tmp / "резерв.sqlite3")["kind"], "local")
        self.assertEqual(list(tmp.iterdir()), [], "детектор ничего не создаёт")


class TestWarnings(unittest.TestCase):
    def _with(self, mapping):
        return mock.patch.object(cloudpaths, "classify", lambda p: dict(mapping.get(str(p), mapping["*"])))

    def test_texts_have_no_path_and_name_the_client(self):
        secret = os.path.join(_DATA_TMP, "Секретная-папка")
        cases = {"cloud": {"kind": "cloud", "client": "OneDrive", "basis": "tag"},
                 "name": {"kind": "cloud", "client": "Dropbox", "basis": "name"},
                 "network": {"kind": "network", "client": None, "basis": "unc"}}
        for key, info in cases.items():
            with self._with({"*": info}):
                for what in ("data", "backups", "backup"):
                    text = cloudpaths.warning(secret, what)
                    self.assertTrue(text.startswith("Внимание:"), (key, what))
                    self.assertNotIn("Секретная", text)
                    self.assertNotIn(_DATA_TMP, text)
        with self._with({"*": cases["cloud"]}):
            self.assertIn("облачной папке OneDrive", cloudpaths.warning(secret, "data"))
            self.assertIn("RUS_TRANSCRIBER_DATA", cloudpaths.warning(secret, "data"))
        with self._with({"*": cases["name"]}):
            self.assertIn("по имени похожа", cloudpaths.warning(secret, "backup"))
        with self._with({"*": {"kind": "local", "client": None, "basis": None}}):
            self.assertIsNone(cloudpaths.warning(secret, "data"))

    def test_backups_folder_reported_only_when_it_differs(self):
        data = Path(tempfile.mkdtemp(prefix="rt-dd-"))
        self.addCleanup(shutil.rmtree, data, ignore_errors=True)
        cloud = {"kind": "cloud", "client": "OneDrive", "basis": "tag"}
        local = {"kind": "local", "client": None, "basis": None}
        with self._with({"*": local}):
            self.assertEqual(cloudpaths.data_dir_warnings(data), [])
        with self._with({"*": cloud}):
            self.assertEqual(len(cloudpaths.data_dir_warnings(data)), 1, "без db-backups — только папка данных")
            (data / "db-backups").mkdir()
            self.assertEqual(len(cloudpaths.data_dir_warnings(data)), 1, "резервы там же — одна строка")
        with self._with({str(data / "db-backups"): cloud, "*": local}):
            found = cloudpaths.data_dir_warnings(data)
            self.assertEqual(len(found), 1)
            self.assertIn("db-backups", found[0])


class TestSqliteVersion(unittest.TestCase):
    def test_wal_reset_fix_by_version(self):
        table = {"3.49.1": False, "3.50.4": False, "3.50.6": False, "3.50.7": True, "3.50.8": True,
                 "3.51.2": False, "3.51.3": True, "3.52.0": True, "3.54.1": True, "3.44.5": False, "3.44.6": True,
                 "3.44.7": True, "3.45.0": False, "3.7.0": False, "4.0": True, "x.y": None, "3": None}
        for version, expected in table.items():
            self.assertIs(selfcheck.wal_reset_fixed(version), expected, version)

    def _run(self, version, paths_module=None, warnings=()):
        with mock.patch.dict(sys.modules):
            sys.modules.pop("paths", None)
            if paths_module is not None:
                sys.modules["paths"] = paths_module
            with mock.patch("sqlite3.sqlite_version", version), \
                    mock.patch.object(cloudpaths, "data_dir_warnings", return_value=list(warnings)):
                return selfcheck.run(tests=False, repo=False, morph=False)

    def test_old_sqlite_warns_without_failing(self):
        r = self._run("3.49.1")
        self.assertTrue(r["ok"])
        self.assertEqual(r["storage"]["sqlite"], "3.49.1")
        self.assertIs(r["storage"]["wal_reset_fixed"], False)
        self.assertIn("WAL-reset", r["storage"]["warnings"][0])
        self.assertIn("предупреждения: SQLite без исправления WAL-reset", r["summary"])
        self.assertTrue(r["summary"].startswith("всё в порядке"))
        self.assertNotIn("labels", r["storage"])

    def test_fixed_sqlite_and_no_paths_module(self):
        r = self._run("3.51.3")
        self.assertEqual(r["summary"], "всё в порядке")
        self.assertEqual(r["storage"]["warnings"], [])
        self.assertEqual(r["storage"]["data_dir"]["kind"], "not_checked")

    def test_data_dir_checked_when_paths_is_loaded(self):
        fake = types.SimpleNamespace(DATA_DIR=Path(_DATA_TMP))
        r = self._run("3.51.3", fake, ["Внимание: папка данных в облаке"])
        self.assertEqual(r["storage"]["warnings"], ["Внимание: папка данных в облаке"])
        self.assertIn("папка данных или резервов", r["summary"])
        self.assertTrue(r["ok"])

    def test_storage_failure_does_not_break_selfcheck(self):
        with mock.patch.object(selfcheck, "run_storage", side_effect=RuntimeError("сбой")):
            r = selfcheck.run(tests=False, repo=False, morph=False)
        self.assertTrue(r["ok"])
        self.assertIn("RuntimeError", r["storage"]["error"])
        self.assertFalse(selfcheck.run(tests=False, repo=False, morph=False, storage=False).get("storage"))


WARN = "Внимание: Папка данных программы лежит в облачной папке OneDrive: <проверка>"


class TestShownToHumanAndAgent(unittest.TestCase):
    def test_settings_screen_and_backup(self):
        import core
        import core_conflicts
        import workflow_page
        with mock.patch.object(cloudpaths, "data_dir_warnings", return_value=[WARN]):
            self.assertEqual(core.settings_info()["data_dir_warnings"], [WARN])
            banner = workflow_page.data_dir_banner()
        self.assertIn('class="warn"', banner)
        # полоса вставляется после первого «</style>» страницы: метка должна остаться в workflow.html
        self.assertIn("</style>\n", (ROOT / "workflow.html").read_text(encoding="utf-8"))
        self.assertIn("&lt;проверка&gt;", banner)
        self.assertNotIn("<проверка>", banner)
        with mock.patch.object(cloudpaths, "data_dir_warnings", return_value=[]):
            self.assertNotIn("data_dir_warnings", core.settings_info())
            self.assertEqual(workflow_page.data_dir_banner(), "")
        with mock.patch.object(cloudpaths, "data_dir_warnings", side_effect=RuntimeError("сбой")):
            self.assertEqual(workflow_page.data_dir_banner(), "")
            self.assertNotIn("data_dir_warnings", core.settings_info())
        with mock.patch.object(cloudpaths, "warning", return_value=WARN):
            self.assertEqual(core_conflicts.backup_destination_warning("x.sqlite3"), [WARN])
        with mock.patch.object(cloudpaths, "warning", side_effect=RuntimeError("сбой")):
            self.assertEqual(core_conflicts.backup_destination_warning("x.sqlite3"), [])

    def test_server_notice_and_agent_backup_warning(self):
        import server
        err = io.StringIO()
        with mock.patch.object(cloudpaths, "data_dir_warnings", return_value=[WARN]), mock.patch("sys.stderr", err):
            notice = server._data_dir_notice()
        self.assertIn(WARN, notice)
        self.assertIn("Сообщите это человеку", notice)
        self.assertIn(WARN, err.getvalue())
        self.assertNotIn(_DATA_TMP, err.getvalue())
        with mock.patch.object(cloudpaths, "data_dir_warnings", side_effect=RuntimeError("сбой")):
            self.assertEqual(server._data_dir_notice(), "")
        report = {"destination": "d", "report": {"damaged_copies": [{"conflict_id": "c1", "relative_path": "a"}]},
                  "warning": [WARN, "Повреждённых сохранённых копий правок: 1.", "- «a» — конфликт c1"]}
        with mock.patch.object(server.core, "backup_store", return_value=report), \
                mock.patch.object(server.core, "backup_destination_warning", return_value=[WARN]):
            result = server._backup_for_agent("d")
        self.assertEqual(result["warning"], [WARN, "Повреждённых сохранённых копий правок: 1."])


if __name__ == "__main__":
    unittest.main()
