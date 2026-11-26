"""Новая установка сама создаёт базу записей; установку со следами работы в файлах не меняет.

`recordings.prepare_new_installation` (шаг «База записей» install.ps1, `rt.py db-init`) — случаи А–З плана
шага 1 отключения файлового режима. Всё на синтетике во временной папке; настоящая папка данных не задевается.
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

import migrate
import paths
import recordings
import settings
import storage

ROOT = Path(__file__).resolve().parent.parent


def _digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


class NewInstallStoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="new-install-store-")
        self.addCleanup(temp.cleanup)
        self.data = Path(temp.name) / "data"
        for sub in ("groups", "jobs", "bench"):           # их создаёт импорт paths.py при любом запуске
            (self.data / sub).mkdir(parents=True)
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in (("DATA_DIR", self.data), ("GROUPS", self.data / "groups"),
                            ("JOBS", self.data / "jobs"), ("BENCH", self.data / "bench"),
                            ("CONFIG_FILE", self.data / "config.json"), ("STATE_DB", self.data / "state.sqlite3"),
                            ("HYPOTHESES", self.data / "hypotheses")):
            stack.enter_context(mock.patch.object(paths, name, value))
        self.pending = stack.enter_context(mock.patch.object(migrate, "pending", return_value=False))
        self.db = self.data / "state.sqlite3"
        self.config = self.data / "config.json"

    def write_config(self, payload):
        self.config.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def job(self, name, kind):
        (self.data / "jobs" / f"{name}.json").write_text(json.dumps({"state": "done", "kind": kind}),
                                                          encoding="utf-8")

    def assert_untouched(self, before_db, before_config, report):
        self.assertEqual(_digest(self.db), before_db)
        self.assertEqual(_digest(self.config), before_config)
        self.assertEqual(report["changed"], [])

    def assert_created(self, report):
        self.assertEqual((report["case"], report["ok"]), ("new", True))
        self.assertEqual(report["changed"], ["state.sqlite3", "config.json"])
        storage.validate_copy(self.db)
        store = recordings.open_store(self.db, readonly=True)
        try:
            self.assertEqual(store.current_identity()["display_name"], "")      # имя не спрашивается
        finally:
            store.close()
        self.assertEqual(json.loads(self.config.read_text(encoding="utf-8"))["recording_backend"], "sqlite")
        self.assertEqual(settings.recording_backend(), "sqlite")
        self.assertEqual(list(self.data.glob("state.sqlite3.incomplete-*")), [])

    # --- А, Б: следов нет ---
    def test_a_empty_data_dir_creates_store_and_selects_it(self):
        self.assert_created(recordings.prepare_new_installation())

    def test_b_bench_only_data_dir_creates_store_and_keeps_other_settings(self):
        self.write_config({"schema_version": 1, "asr_device": "GPU", "save_hypotheses": True})
        self.job("20261126-100000-b00001", "benchmark")
        (self.data / "bench" / "report.md").write_text("синтетический отчёт", encoding="utf-8")
        (self.data / "jobs" / "config.lock").write_bytes(b"")
        (self.data / "jobs" / "voice-locks").mkdir()
        (self.data / "jobs" / "slots").mkdir()
        (self.data / "jobs" / "slots" / "slot0.lock").write_bytes(b"")
        self.assert_created(recordings.prepare_new_installation())
        saved = json.loads(self.config.read_text(encoding="utf-8"))
        self.assertEqual((saved["asr_device"], saved["save_hypotheses"]), ("GPU", True))

    # --- В: следы работы в файлах — ничего не меняется ---
    def test_c_any_trace_of_file_work_changes_nothing(self):
        traces = {
            "jobs": lambda: self.job("20261126-100000-t00001", "transcribe"),
            "jobs-unreadable": lambda: (self.data / "jobs" / "broken.json").write_text("{", encoding="utf-8"),
            "groups": lambda: ((self.data / "groups" / "g").mkdir(),
                               (self.data / "groups" / "g" / "vocab.json").write_text("{}", encoding="utf-8")),
            "hypotheses": lambda: ((self.data / "hypotheses").mkdir(),
                                   (self.data / "hypotheses" / "x.json").write_text("{}", encoding="utf-8")),
            "exports": lambda: ((self.data / "exports").mkdir(), (self.data / "exports" / "a.zip").write_bytes(b"z")),
            "_deleted": lambda: ((self.data / "_deleted" / "g").mkdir(parents=True),
                                 (self.data / "_deleted" / "g" / "voices.json").write_text("{}", encoding="utf-8")),
            "_backup": lambda: ((self.data / "_backup-import-1").mkdir(),
                                (self.data / "_backup-import-1" / "config.json").write_text("{}", encoding="utf-8")),
            "legacy_layout": lambda: self.pending.configure_mock(return_value=True),
            "explicit-legacy": lambda: self.write_config({"recording_backend": "legacy_files"}),
        }
        for name, make in traces.items():
            with self.subTest(trace=name):
                self.setUp()
                make()
                before = (_digest(self.db), _digest(self.config))
                report = recordings.prepare_new_installation()
                self.assertEqual((report["case"], report["ok"]), ("legacy", True))
                self.assertIn("docs/legacy-migration.md", report["message"])
                self.assert_untouched(*before, report)
                self.assertFalse(self.db.exists())
                self.assertEqual(settings.recording_backend(), "legacy_files")

    def test_c_unreadable_config_is_not_rewritten(self):
        self.config.write_text("{ повреждён", encoding="utf-8")
        before = (_digest(self.db), _digest(self.config))
        report = recordings.prepare_new_installation()
        self.assertEqual((report["case"], report["ok"]), ("legacy", True))
        self.assert_untouched(*before, report)

    # --- Г, Д: база уже есть ---
    def test_d_existing_selected_store_is_not_touched(self):
        recordings.initialize_store(self.db, display_name="Синтетический").close()
        self.write_config({"recording_backend": "sqlite"})
        before = (_digest(self.db), _digest(self.config))
        report = recordings.prepare_new_installation()
        self.assertEqual((report["case"], report["ok"]), ("sqlite", True))
        self.assert_untouched(*before, report)

    def test_e_store_with_file_mode_is_left_as_is(self):
        recordings.initialize_store(self.db, display_name="Синтетический").close()
        self.write_config({"recording_backend": "legacy_files"})
        before = (_digest(self.db), _digest(self.config))
        report = recordings.prepare_new_installation()
        self.assertEqual((report["case"], report["ok"]), ("legacy_with_db", True))
        self.assert_untouched(*before, report)

    def test_e_store_without_choice_but_with_file_traces_is_left_as_is(self):
        recordings.initialize_store(self.db, display_name="").close()
        self.job("20261126-100000-t00001", "transcribe")
        before = (_digest(self.db), _digest(self.config))
        report = recordings.prepare_new_installation()
        self.assertEqual(report["case"], "legacy_with_db")
        self.assert_untouched(*before, report)

    # --- Е: выбрана база, а её нет ---
    def test_f_selected_but_missing_store_is_not_recreated(self):
        self.write_config({"recording_backend": "sqlite"})
        for sidecar in (False, True):
            with self.subTest(sidecar=sidecar):
                if sidecar:
                    self.write_config({})
                    self.db.with_name("state.sqlite3-wal").write_bytes(b"")
                before = (_digest(self.db), _digest(self.config))
                report = recordings.prepare_new_installation()
                self.assertEqual((report["case"], report["ok"]), ("missing_db", False))
                self.assert_untouched(*before, report)
                self.assertFalse(self.db.exists())

    # --- Ж: повтор ---
    def test_g_second_run_changes_nothing(self):
        self.assert_created(recordings.prepare_new_installation())
        before = (_digest(self.db), _digest(self.config))
        report = recordings.prepare_new_installation()
        self.assertEqual((report["case"], report["ok"]), ("sqlite", True))
        self.assert_untouched(*before, report)

    # --- З: прошлый запуск создал базу, но не успел её выбрать ---
    def test_h_empty_store_without_choice_finishes_switch(self):
        recordings.initialize_store(self.db, display_name="").close()
        self.write_config({"asr_device": "CPU"})
        before_db = _digest(self.db)
        report = recordings.prepare_new_installation()
        self.assertEqual((report["case"], report["ok"], report["changed"]), ("finished_switch", True, ["config.json"]))
        self.assertEqual(_digest(self.db), before_db)
        self.assertEqual(settings.recording_backend(), "sqlite")

    def test_h_store_with_operations_is_not_switched(self):
        store = recordings.initialize_store(self.db, display_name="")
        try:
            store.db.execute("INSERT INTO operations(operation_id, kind, parameters_hash, context_json, result_json, "
                             "created_at) VALUES ('synthetic', 'synthetic', 'x', '{}', '{}', '2026-11-26T00:00:00Z')")
            store.db.commit()
        finally:
            store.close()
        before = (_digest(self.db), _digest(self.config))
        report = recordings.prepare_new_installation()
        self.assertEqual(report["case"], "legacy_with_db")
        self.assert_untouched(*before, report)
        self.assertEqual(settings.recording_backend(), "legacy_files")

    # --- отказы ---
    def test_failed_creation_leaves_config_untouched(self):
        self.write_config({"asr_device": "GPU"})
        before = _digest(self.config)
        with mock.patch.object(recordings, "initialize_store", side_effect=ValueError("only local fixed drives")):
            report = recordings.prepare_new_installation()
        self.assertEqual((report["case"], report["ok"]), ("new", False))
        self.assertIn("не готова", report["message"])
        self.assertEqual(_digest(self.config), before)
        self.assertEqual(settings.recording_backend(), "legacy_files")

    def test_failed_selection_is_finished_by_next_run(self):
        with mock.patch.object(settings, "set_recording_backend", side_effect=OSError("занят")):
            report = recordings.prepare_new_installation()
        self.assertEqual((report["ok"], report["changed"]), (False, ["state.sqlite3"]))
        self.assertTrue(self.db.exists())
        report = recordings.prepare_new_installation()
        self.assertEqual((report["case"], report["ok"]), ("finished_switch", True))

    def test_check_only_changes_nothing(self):
        report = recordings.prepare_new_installation(check_only=True)
        self.assertEqual((report["case"], report["ok"], report["changed"]), ("new", True, []))
        self.assertFalse(self.db.exists())
        self.assertFalse(self.config.exists())


class NewInstallStoreEntryPointsTests(unittest.TestCase):
    def test_rt_db_init_reports_json_and_exit_code(self):
        import rt
        for ok, code in ((True, 0), (False, 1)):
            with self.subTest(ok=ok), \
                    mock.patch.object(recordings, "prepare_new_installation",
                                      return_value={"ok": ok, "case": "new", "message": "m"}) as prepare, \
                    mock.patch.object(rt, "out") as out:
                self.assertEqual(rt.main(["db-init", "--check"]), code)
                prepare.assert_called_once_with(check_only=True)
                out.assert_called_once()

    def test_server_and_core_import_do_not_create_store(self):
        with tempfile.TemporaryDirectory(prefix="new-install-import-") as temp:
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
