"""База записей установки: нет — создаётся штатно и выбирается; есть — проверяется чтением и выбирается.

`recordings.prepare_installation_store` (шаг «База записей» install.ps1, `rt.py db-init`). Всё на синтетике во
временной папке; настоящая папка данных не задевается.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import paths
import recordings
import settings
import storage

ROOT = Path(__file__).resolve().parent.parent


def _digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


class InstallationStoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="install-store-")
        self.addCleanup(temp.cleanup)
        self.data = Path(temp.name) / "data"
        for sub in ("groups", "jobs", "bench"):
            (self.data / sub).mkdir(parents=True)
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in (("DATA_DIR", self.data), ("GROUPS", self.data / "groups"),
                            ("JOBS", self.data / "jobs"), ("BENCH", self.data / "bench"),
                            ("CONFIG_FILE", self.data / "config.json"), ("STATE_DB", self.data / "state.sqlite3")):
            stack.enter_context(mock.patch.object(paths, name, value))
        self.db = self.data / "state.sqlite3"
        self.config = self.data / "config.json"

    def write_config(self, payload, bom=False):
        self.config.write_bytes((b"\xef\xbb\xbf" if bom else b"") + json.dumps(payload).encode("utf-8"))

    def make_store(self, name="Синтетический"):
        recordings.initialize_store(self.db, display_name=name).close()

    def assert_untouched(self, before, report):
        self.assertEqual((_digest(self.db), _digest(self.config)), before)
        self.assertEqual(report["changed"], [])

    def test_no_store_creates_and_selects_it_keeping_other_settings(self):
        self.write_config({"schema_version": 1, "asr_device": "GPU", "save_hypotheses": True})
        report = recordings.prepare_installation_store()
        self.assertEqual((report["case"], report["ok"], report["changed"]),
                         ("created", True, ["state.sqlite3", "config.json"]))
        storage.validate_copy(self.db)
        store = recordings.open_store(self.db, readonly=True)
        try:
            self.assertEqual(store.current_identity()["display_name"], "")      # имя не спрашивается
        finally:
            store.close()
        saved = json.loads(self.config.read_text(encoding="utf-8"))
        self.assertEqual((saved["recording_backend"], saved["asr_device"], saved["save_hypotheses"]),
                         ("sqlite", "GPU", True))
        self.assertEqual(list(self.data.glob("state.sqlite3.incomplete-*")), [])

    def test_existing_selected_store_is_only_read(self):
        self.make_store()
        self.write_config({"recording_backend": "sqlite"}, bom=True)      # BOM читается (utf-8-sig)
        before = (_digest(self.db), _digest(self.config))
        report = recordings.prepare_installation_store()
        self.assertEqual((report["case"], report["ok"]), ("ready", True))
        self.assert_untouched(before, report)
        self.assertEqual(sorted(p.name for p in self.data.iterdir() if p.name.startswith("state")), ["state.sqlite3"])

    def test_existing_store_not_selected_gets_selected_without_changing_it(self):
        for config in (None, {"recording_backend": "legacy_files"}, {"recording_backend": "other"},
                       {"recording_backend": 5}, {"asr_device": "CPU"}):
            with self.subTest(config=config):
                self.setUp()
                self.make_store()
                if config is not None:
                    self.write_config(config)
                before_db = _digest(self.db)
                report = recordings.prepare_installation_store()
                self.assertEqual((report["case"], report["ok"], report["changed"]), ("selected", True, ["config.json"]))
                self.assertEqual(_digest(self.db), before_db)
                self.assertEqual(settings.recording_backend(), "sqlite")

    def test_broken_store_is_reported_and_not_repaired(self):
        for content in (b"", b"not a database" * 100):
            with self.subTest(size=len(content)):
                self.setUp()
                self.db.write_bytes(content)
                self.write_config({"recording_backend": "sqlite"})
                before = (_digest(self.db), _digest(self.config))
                report = recordings.prepare_installation_store()
                self.assertEqual((report["case"], report["ok"]), ("error", False))
                self.assertIn("не открывается", report["message"])
                self.assert_untouched(before, report)

    def test_unreadable_config_is_not_rewritten(self):
        for raw in (b"{ broken", b"[1, 2]"):
            with self.subTest(raw=raw):
                self.setUp()
                self.config.write_bytes(raw)
                report = recordings.prepare_installation_store()
                self.assertEqual((report["case"], report["ok"]), ("error", False))
                self.assertEqual(self.config.read_bytes(), raw)
                self.assertFalse(self.db.exists())

    def test_creation_failure_reports_reason_and_keeps_config(self):
        self.write_config({"asr_device": "GPU"})
        before = _digest(self.config)
        for ex in (ValueError("only local fixed drives are supported"), OSError("no hard links"),
                   storage.sqlite3.OperationalError("disk I/O error")):
            with self.subTest(ex=type(ex).__name__), mock.patch.object(recordings, "initialize_store", side_effect=ex):
                report = recordings.prepare_installation_store()
                self.assertEqual((report["case"], report["ok"]), ("error", False))
                self.assertIn(type(ex).__name__, report["message"])
                self.assertEqual(_digest(self.config), before)

    def test_failed_selection_is_finished_by_next_run(self):
        with mock.patch.object(settings, "set_recording_backend", side_effect=OSError("занят")):
            report = recordings.prepare_installation_store()
        self.assertEqual((report["ok"], report["changed"]), (False, ["state.sqlite3"]))
        before_db = _digest(self.db)
        report = recordings.prepare_installation_store()
        self.assertEqual((report["case"], report["ok"]), ("selected", True))
        self.assertEqual(_digest(self.db), before_db)

    def test_second_run_changes_nothing(self):
        self.assertTrue(recordings.prepare_installation_store()["ok"])
        before = (_digest(self.db), _digest(self.config))
        report = recordings.prepare_installation_store()
        self.assertEqual(report["case"], "ready")
        self.assert_untouched(before, report)

    def test_check_only_changes_nothing(self):
        report = recordings.prepare_installation_store(check_only=True)
        self.assertEqual((report["case"], report["ok"], report["changed"]), ("created", True, []))
        self.assertFalse(self.db.exists())
        self.assertFalse(self.config.exists())
        self.make_store()
        report = recordings.prepare_installation_store(check_only=True)
        self.assertEqual((report["case"], report["changed"]), ("selected", []))
        self.assertFalse(self.config.exists())


class InstallationStoreEntryPointsTests(unittest.TestCase):
    def test_rt_db_init_reports_json_and_exit_code(self):
        import rt
        for ok, code in ((True, 0), (False, 1)):
            with self.subTest(ok=ok), \
                    mock.patch.object(recordings, "prepare_installation_store",
                                      return_value={"ok": ok, "case": "created", "message": "m"}) as prepare, \
                    mock.patch.object(rt, "out") as out:
                self.assertEqual(rt.main(["db-init", "--check"]), code)
                prepare.assert_called_once_with(check_only=True)
                out.assert_called_once()

    def test_rt_db_init_turns_errors_into_json(self):
        import rt
        with mock.patch.object(recordings, "prepare_installation_store", side_effect=ValueError("network paths")), \
                mock.patch.object(rt, "out") as out:
            self.assertEqual(rt.main(["db-init"]), 1)
        report = out.call_args.args[0]
        self.assertEqual((report["ok"], report["case"]), (False, "error"))
        self.assertIn("network paths", report["message"])

    def test_real_db_init_with_environment_creates_store(self):
        with tempfile.TemporaryDirectory(prefix="install-store-rt-") as temp:
            data = Path(temp) / "data"
            env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=temp, TMP=temp,
                       PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            run = subprocess.run([sys.executable, "-B", str(ROOT / "rt.py"), "db-init"], cwd=ROOT, env=env,
                                 capture_output=True, text=True, encoding="utf-8", timeout=120)
            self.assertEqual(run.returncode, 0, run.stderr[-2000:])
            self.assertEqual(json.loads(run.stdout)["case"], "created")
            self.assertTrue((data / "state.sqlite3").is_file())

    def test_server_and_core_import_do_not_create_store(self):
        with tempfile.TemporaryDirectory(prefix="install-store-import-") as temp:
            data = Path(temp) / "data"
            env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=temp, TMP=temp,
                       PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            run = subprocess.run([sys.executable, "-B", "-c", "import core, server"], cwd=ROOT, env=env,
                                 capture_output=True, text=True, encoding="utf-8", timeout=120)
            self.assertEqual(run.returncode, 0, run.stderr[-2000:])
            self.assertFalse((data / "state.sqlite3").exists())
            self.assertFalse((data / "config.json").exists())

    def test_installer_runs_store_step_after_packages_and_before_token(self):
        text = (ROOT / "install.ps1").read_text(encoding="utf-8-sig")
        packages, step, token = (text.index('Step "7/11"'), text.index('Step "8/11" "База записей"'),
                                 text.index('Step "9/11" "Токен Hugging Face"'))
        self.assertLess(packages, step)
        self.assertLess(step, token)
        body = text[step:token]
        self.assertIn('$env:RUS_TRANSCRIBER_DATA = $DataDir', body)
        self.assertIn('"rt.py") db-init', body)
        self.assertIn("$env:RUS_TRANSCRIBER_DATA = $prevData", body)      # переменная возвращается
        # шаг внутри ветки «не режим приёмки»: там нет окружения программы
        acceptance = text.rindex("if ($AcceptanceCopyOnly) {", 0, step)
        self.assertIn("venv, зависимости и HF-проверки пропущены", text[acceptance:step])
        devices = text[packages:step]                 # шаг 7 импортирует paths: папка данных та же, что у установки
        self.assertLess(devices.index("$env:RUS_TRANSCRIBER_DATA = $DataDir"), devices.index("import paths, backends"))
        self.assertIn("finally { $env:RUS_TRANSCRIBER_DATA = $prevData }", devices)
        ending = text[text.index('Step "11/11"'):]
        self.assertIn("if (($storeOk -eq $false) -or -not $selfCheckOk) { exit 2 }", ending)


if __name__ == "__main__":
    unittest.main()
