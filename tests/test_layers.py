"""Ядро (core.py) без MCP, командная строка (rt.py) и подтверждение в инструментах MCP. Запуск: python -m unittest discover -s tests"""
import atexit
import shutil
import asyncio
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-data-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_OLD_ENV = os.environ.get("RUS_TRANSCRIBER_DATA")
os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import confirm  # noqa: E402
import core  # noqa: E402
import paths  # noqa: E402
import rt  # noqa: E402
if _OLD_ENV is None:
    os.environ.pop("RUS_TRANSCRIBER_DATA", None)
else:
    os.environ["RUS_TRANSCRIBER_DATA"] = _OLD_ENV


def put_group(name):
    gd = paths.group_dir(name)
    gd.mkdir(parents=True, exist_ok=True)
    (gd / "vocab.json").write_text(json.dumps({"group": name}), encoding="utf-8")
    (gd / "voices.json").write_text(json.dumps({"speakers": {}, "version": 3}), encoding="utf-8")
    return gd


def run_rt(argv, tty=False):
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err), \
            mock.patch.object(sys.stdin, "isatty", return_value=tty), mock.patch.object(sys.stdout, "isatty", return_value=tty):
        rc = rt.main(argv)
    return rc, buf.getvalue(), err.getvalue()


class CoreJobsTests(unittest.TestCase):
    def setUp(self):
        self.ids = []

    def tearDown(self):
        for i in self.ids:
            for ext in (".json", ".log", ".pid"):
                (core.JOBS / f"{i}{ext}").unlink(missing_ok=True)

    def job(self, name, state, **kw):
        self.ids.append(name)
        (core.JOBS / f"{name}.json").write_text(json.dumps({"state": state, **kw}), encoding="utf-8")
        return name

    def test_unknown_job(self):
        self.assertIsNone(core.job_status("нет-такой"))
        with self.assertRaises(core.CoreError):
            core.status("нет-такой")

    def test_lost_process_becomes_error(self):
        j = self.job("t-lost", "running", stage="расшифровка")
        (core.JOBS / f"{j}.pid").write_text("999999", encoding="utf-8")
        with mock.patch.object(core, "alive", return_value=False):
            s = core.job_status(j)
        self.assertEqual(s["state"], "error")

    def test_status_live_fields_and_stage_key(self):
        j = self.job("t-run", "running", stage="поиск речи 3/10", started_ts=1.0, stage_started_ts=2.0)
        s = core.status(j)
        self.assertIn("elapsed_sec", s)
        self.assertNotIn("started_ts", s)
        self.assertEqual(core.stage_key({"state": "running", "stage": "поиск речи 3/10"}),
                         core.stage_key({"state": "running", "stage": "поиск речи 9/10"}))
        self.assertNotEqual(core.stage_key({"state": "running", "stage": "а 1/2"}),
                            core.stage_key({"state": "running", "stage": "б 1/2"}))

    def test_read_not_ready_and_start_errors(self):
        j = self.job("t-wait", "running", stage="этап", progress=5)
        with self.assertRaises(core.CoreError):
            core.read_transcript(j)
        with self.assertRaises(core.CoreError):
            core.start_transcription("/нет/такого/файла.mp3")
        f = Path(_TMP) / "x.wav"
        f.write_bytes(b"0")
        with self.assertRaises(core.CoreError):
            core.start_transcription(str(f), whisper="лишнее")

    def test_cancel_finished_job_is_noop(self):
        j = self.job("t-done", "done", stage="готово")
        self.assertIn("note", core.cancel_job(j))

    def test_list_jobs(self):
        j = self.job("t-list", "done", stage="готово", progress=100, kind="transcribe", file="a.mp3")
        row = next(r for r in core.list_jobs(50) if r["job_id"] == j)
        self.assertEqual(row["state"], "done")

    def test_describe_does_not_change_data(self):
        gd = put_group("lay-desc")
        try:
            before = sorted(p.name for p in gd.iterdir())
            self.assertTrue(core.describe_delete("lay-desc", True)[1].startswith("НАВСЕГДА"))
            self.assertTrue(core.describe_export("lay-desc", "", False))
            self.assertEqual(sorted(p.name for p in gd.iterdir()), before)
            with self.assertRaises(core.CoreError):
                core.describe_delete("нет-такой-группы")
        finally:
            shutil.rmtree(gd, ignore_errors=True)


class CliTests(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree(paths.group_dir("lay-cli"), ignore_errors=True)
        shutil.rmtree(paths.DATA_DIR / "_deleted", ignore_errors=True)

    def test_readonly_commands_work_without_terminal(self):
        rc, out, _ = run_rt(["settings"])
        self.assertEqual(rc, 0)
        self.assertIn("data_dir", json.loads(out))
        rc, out, _ = run_rt(["jobs", "--limit", "1"])
        self.assertEqual(rc, 0)
        json.loads(out)

    def test_unknown_job_is_error_code_1(self):
        rc, _, err = run_rt(["status", "нет-такой"])
        self.assertEqual(rc, 1)
        self.assertIn("не найдена", err)

    def test_dangerous_commands_refuse_without_terminal(self):
        gd = put_group("lay-cli")
        rc, _, err = run_rt(["delete-group", "lay-cli"])
        self.assertEqual(rc, 2)
        self.assertTrue(gd.is_dir())                              # группа на месте
        self.assertIn("терминал", err)
        rc, _, _ = run_rt(["export", "--group", "lay-cli"])
        self.assertEqual(rc, 2)
        self.assertEqual(run_rt(["cleanup", "--days", "0", "--apply"])[0], 2)

    def test_delete_requires_typed_name(self):
        gd = put_group("lay-cli")
        with mock.patch("builtins.input", return_value="не то"):
            rc, _, _ = run_rt(["delete-group", "lay-cli"], tty=True)
        self.assertEqual(rc, 2)
        self.assertTrue(gd.is_dir())
        with mock.patch("builtins.input", return_value="lay-cli"):
            rc, out, _ = run_rt(["delete-group", "lay-cli"], tty=True)
        self.assertEqual(rc, 0)
        self.assertFalse(gd.is_dir())
        self.assertIn("перенесена", out)

    def test_dry_runs_need_no_confirmation(self):
        self.assertEqual(run_rt(["cleanup", "--days", "36500"])[0], 0)
        self.assertEqual(run_rt(["migrate"])[0], 0)


class ServerGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import migrate
            with mock.patch.object(migrate, "run", return_value={"status": "пропущено в тесте"}):
                import server     # при импорте сервер запускает перенос данных — в тесте он не нужен
        except ImportError:
            raise unittest.SkipTest("нет пакета mcp")
        cls.server = server

    def tearDown(self):
        shutil.rmtree(paths.group_dir("lay-gate"), ignore_errors=True)
        shutil.rmtree(paths.DATA_DIR / "_deleted", ignore_errors=True)

    def call(self, fn, *a, **kw):
        return asyncio.run(fn(*a, **kw))

    def test_delete_without_human_does_nothing(self):
        gd = put_group("lay-gate")
        with mock.patch.object(confirm, "require", side_effect=confirm.NotConfirmed("Человек не подтвердил")):
            r = self.call(self.server.delete_group, "lay-gate", "lay-gate", True)
        self.assertIn("не подтвердил", r)
        self.assertTrue(gd.is_dir())

    def test_delete_with_human_yes_works_and_dialog_text_comes_from_program(self):
        gd = put_group("lay-gate")
        seen = {}

        def fake(action, args, lines):
            seen.update(action=action, args=args, lines=lines)
            return "dialog"
        with mock.patch.object(confirm, "require", side_effect=fake):
            r = self.call(self.server.delete_group, "lay-gate", "lay-gate", False)
        self.assertFalse(gd.is_dir())
        self.assertIn("перенесена", r)
        self.assertIn("lay-gate", "\n".join(seen["lines"]))
        self.assertEqual(seen["args"], {"group": "lay-gate", "permanent": False})

    def test_wrong_confirm_text_stops_before_dialog(self):
        gd = put_group("lay-gate")
        with mock.patch.object(confirm, "require") as req:
            r = self.call(self.server.delete_group, "lay-gate", "другое", False)
        req.assert_not_called()
        self.assertIn("Ошибка", r)
        self.assertTrue(gd.is_dir())

    def test_read_only_calls_need_no_confirmation(self):
        with mock.patch.object(confirm, "require", side_effect=AssertionError("не должно спрашивать")):
            self.call(self.server.cleanup_jobs, 36500, False)
            self.call(self.server.migrate_data, True)

    def test_import_apply_is_gated_but_report_is_not(self):
        put_group("lay-gate")
        with mock.patch.object(confirm, "require", return_value="dialog"):
            r = json.loads(self.call(self.server.export_group, "lay-gate"))
        zp = r["archive"] if "archive" in r else next(v for v in r.values() if isinstance(v, str) and v.endswith(".zip"))
        shutil.rmtree(paths.group_dir("lay-gate"))
        with mock.patch.object(confirm, "require", side_effect=AssertionError("отчёт не требует подтверждения")):
            rep = json.loads(self.call(self.server.import_group, zp, "merge", False))
        self.assertFalse(rep["applied"])
        self.assertFalse(paths.group_dir("lay-gate").is_dir())
        with mock.patch.object(confirm, "require", side_effect=confirm.NotConfirmed("нет")):
            self.assertEqual(self.call(self.server.import_group, zp, "merge", True), "нет")
        self.assertFalse(paths.group_dir("lay-gate").is_dir())          # без «да» ничего не загружено
        with mock.patch.object(confirm, "require", return_value="dialog"):
            self.call(self.server.import_group, zp, "merge", True)
        self.assertTrue(paths.group_dir("lay-gate").is_dir())

    def test_export_and_cleanup_apply_are_gated(self):
        put_group("lay-gate")
        with mock.patch.object(confirm, "require", side_effect=confirm.NotConfirmed("нет")) as req:
            self.assertEqual(self.call(self.server.export_group, "lay-gate"), "нет")
            self.assertEqual(self.call(self.server.cleanup_jobs, 0, True), "нет")
            self.assertEqual(self.call(self.server.migrate_data, False), "нет")
        self.assertEqual(req.call_count, 3)
        self.assertFalse(list((paths.DATA_DIR / "exports").glob("*lay-gate*")) if (paths.DATA_DIR / "exports").exists() else [])


if __name__ == "__main__":
    unittest.main()
