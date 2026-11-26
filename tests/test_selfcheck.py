import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import selfcheck  # noqa: E402

OUT_FAIL = """..F.E
======================================================================
FAIL: test_a (tests.test_x.Case.test_a)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "x.py", line 3, in test_a
    self.assertEqual(1, 2)
AssertionError: 1 != 2

======================================================================
ERROR: test_b (tests.test_x.Case.test_b)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "x.py", line 5, in test_b
    import nope
ModuleNotFoundError: No module named 'nope'

----------------------------------------------------------------------
Ran 5 tests in 0.012s

FAILED (failures=1, errors=1, skipped=2)
"""


class SelfCheck(unittest.TestCase):
    def test_parse_ok(self):
        r = selfcheck.parse_unittest("....\n----------------------------------------------------------------------\n"
                                     "Ran 115 tests in 14.020s\n\nOK (skipped=1)\n")
        self.assertEqual((r["ran"], r["ok"], r["skipped"], r["failed"]), (115, True, 1, []))

    def test_parse_slowest(self):
        r = selfcheck.parse_unittest("..\nSlowest test durations\n" + "-" * 70 + "\n"
                                     "41.120s    test_x (tests.test_a.C.test_x)\n0.500s     test_y (tests.test_b.D.test_y)\n\n"
                                     + "-" * 70 + "\nRan 2 tests in 41.7s\n\nOK\n")
        self.assertEqual(r["slowest"][0], {"sec": 41.12, "test": "tests.test_a.C.test_x"})
        self.assertTrue(r["ok"])

    def test_parse_failures(self):
        r = selfcheck.parse_unittest(OUT_FAIL)
        self.assertEqual((r["ran"], r["ok"], r["failures"], r["errors"], r["skipped"]), (5, False, 1, 1, 2))
        self.assertEqual([f["kind"] for f in r["failed"]], ["FAIL", "ERROR"])
        self.assertEqual(r["failed"][0]["test"], "tests.test_x.Case.test_a")
        self.assertEqual(r["failed"][0]["why"], "AssertionError: 1 != 2")
        self.assertIn("No module named", r["failed"][1]["why"])

    @staticmethod
    def _counts(summary):
        r = selfcheck.parse_unittest("Ran 5 tests in 1.000s\n\n" + summary + "\n")
        return (r["ok"], r["failures"], r["errors"], r["skipped"], r["expected_failures"], r["unexpected_successes"])

    def test_expected_failures_are_not_failures(self):
        # «expected failures=2» — помеченные @unittest.expectedFailure известные ограничения, не падения
        self.assertEqual(self._counts("OK (skipped=1, expected failures=2)"), (True, 0, 0, 1, 2, 0))

    def test_real_failures_keep_their_count_next_to_expected_ones(self):
        self.assertEqual(self._counts("FAILED (failures=1, skipped=1, expected failures=2)"), (False, 1, 0, 1, 2, 0))

    def test_unexpected_success_is_counted_separately(self):
        self.assertEqual(self._counts("FAILED (expected failures=1, unexpected successes=1)"), (False, 0, 0, 0, 1, 1))

    def test_plain_ok_has_zero_expected_and_unexpected(self):
        self.assertEqual(self._counts("OK"), (True, 0, 0, 0, 0, 0))

    def test_mask_hides_words_and_user_paths(self):
        m = selfcheck.mask([("a.py:3", "запрещённое слово «Секрет»"), ("b.md:1", r"путь с именем пользователя: C:\Users\user\x"),
                            ("groups/x.json", "файл личных данных или результатов не должен быть в репозитории")])
        self.assertNotIn("Секрет", str(m))
        self.assertEqual(m[1]["why"], "путь с именем пользователя")
        self.assertEqual(m[2]["why"], "файл личных данных или результатов не должен быть в репозитории")

    def _with_check_repo(self, body):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tools").mkdir()
            (root / "tools" / "check_repo.py").write_text(
                "def check(staged=False, require_words=True):\n" + body, encoding="utf-8")
            old = selfcheck.HERE
            selfcheck.HERE = root
            try:
                return selfcheck.run_check_repo()
            finally:
                selfcheck.HERE = old

    def test_empty_word_list_is_partial_not_failure(self):
        # на домашней установке приватного списка слов нет: проверка путей выполнена, это не поломка
        r = self._with_check_repo("    return [], set(), False\n")
        self.assertTrue(r["ok"])
        self.assertTrue(r["partial"])
        self.assertEqual((r["words"], r["violations"]), (0, 0))
        self.assertIn("проверка слов не выполнена", r["note"])

    def test_violations_fail_even_without_word_list(self):
        r = self._with_check_repo("    return [('a.py:1', 'нарушение')], set(), False\n")
        self.assertFalse(r["ok"])
        self.assertEqual(r["violations"], 1)

    def test_tests_run_with_isolated_temp_data_and_no_bytecode(self):
        seen = {}

        def run(cmd, *, cwd, env, **kwargs):
            seen.update(cmd=cmd, cwd=cwd, env=env, stdin=kwargs.get("stdin"))
            return mock.Mock(returncode=0, stdout="", stderr="Ran 1 test in 0.01s\n\nOK\n")

        with mock.patch.dict(selfcheck.os.environ, {selfcheck.JOBS_ENV: "1"}), \
                mock.patch.object(selfcheck.subprocess, "run", side_effect=run):
            result = selfcheck.run_tests()
        env = seen["env"]
        self.assertTrue(result["ok"])
        self.assertIn("-B", seen["cmd"])
        # из терминала тест, ждущий ответа человека, зависал бы: ввод тестам закрыт
        self.assertIs(seen["stdin"], selfcheck.subprocess.DEVNULL)
        self.assertEqual(env["TEMP"], env["TMP"])
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertTrue(Path(env["RUS_TRANSCRIBER_DATA"]).is_relative_to(Path(env["TEMP"]).parent))
        self.assertFalse(Path(env["TEMP"]).exists())


    def test_timeout_names_the_limit_in_minutes(self):
        def run(cmd, **kwargs):
            raise selfcheck.subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        with mock.patch.dict(selfcheck.os.environ, {selfcheck.JOBS_ENV: "1"}), \
                mock.patch.object(selfcheck.subprocess, "run", side_effect=run):
            result = selfcheck.run_tests(timeout=720)
        self.assertFalse(result["ok"])
        self.assertEqual(result["ran"], 0)
        self.assertEqual(result["error"], "тесты не уложились в 12 мин (720 с)")


PARALLEL_OK = """запуск: 4 групп(ы) параллельно, модулей 200
g1: Ran 50 за 80.0 с, OK (по часам 80.1 с)
----------------------------------------------------------------------
Ran 200 tests in 85.123s

OK (skipped=3)
групп 4; суммарное время групп 310 с
"""
PARALLEL_FAILED = """g2: Ran 50 за 80.0 с, FAILED (по часам 80.1 с)
===== g2: полный вывод упавшей группы =====
.......F
======================================================================
FAIL: test_a (tests.test_x.Case.test_a)
----------------------------------------------------------------------
Traceback (most recent call last):
AssertionError: 1 != 2

----------------------------------------------------------------------
Ran 50 tests in 80.000s

OK
===== конец вывода g2 =====
НАРУШЕНИЕ: g2: итог unittest OK, но процесс вышел с кодом 3
----------------------------------------------------------------------
Ran 200 tests in 90.000s

FAILED (failures=1, run problems=1)
групп 4; суммарное время групп 330 с
"""


class ParallelSelfCheckTests(unittest.TestCase):
    """Параллельный запуск тестов в самопроверке (tests/run_parallel.py) и откат к одному процессу."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rt-selfcheck-par-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "tests").mkdir()
        (self.root / "tests" / "run_parallel.py").write_text("# заглушка\n", encoding="utf-8")
        patcher = mock.patch.object(selfcheck, "HERE", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def env(self, value):
        env = {k: v for k, v in selfcheck.os.environ.items() if k != selfcheck.JOBS_ENV}
        if value is not None:
            env[selfcheck.JOBS_ENV] = value
        patcher = mock.patch.dict(selfcheck.os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def sequential_ok(self):
        return mock.patch.object(selfcheck.subprocess, "run", return_value=mock.Mock(
            returncode=0, stdout="", stderr="Ran 7 tests in 1.0s\n\nOK\n"))

    def test_parse_parallel_takes_the_total_not_the_first_group(self):
        r = selfcheck.parse_parallel(PARALLEL_OK)
        self.assertEqual((r["ran"], r["ok"], r["skipped"], r["groups"]), (200, True, 3, 4))

    def test_parse_parallel_reports_failed_tests_and_run_problems_and_ignores_ok_inside_a_group(self):
        r = selfcheck.parse_parallel(PARALLEL_FAILED)
        self.assertEqual((r["ran"], r["ok"], r["failures"]), (200, False, 1))
        self.assertEqual([(f["kind"], f["test"]) for f in r["failed"]], [("FAIL", "tests.test_x.Case.test_a")])
        self.assertIn("вышел с кодом 3", r["run_problems"][0])

    def test_by_default_tests_run_in_groups_with_a_group_timeout_inside_the_limit(self):
        self.env(None)
        with mock.patch.object(selfcheck.os, "cpu_count", return_value=8), \
                mock.patch.object(selfcheck, "_run_tree", return_value=(0, PARALLEL_OK)) as tree:
            result = selfcheck.run_tests(timeout=720)
        cmd, cwd, env, timeout = tree.call_args[0]
        self.assertEqual((result["ok"], result["ran"], result["mode"], result["groups"]), (True, 200, "parallel", 4))
        self.assertIn("-B", cmd)
        self.assertTrue(cmd[2].endswith("run_parallel.py"))
        self.assertEqual(cmd[cmd.index("--timeout") + 1], str(720 - selfcheck.GROUP_MARGIN))
        self.assertNotIn("--jobs", cmd)
        self.assertEqual((timeout, Path(cwd), env["PYTHONDONTWRITEBYTECODE"]), (720, self.root, "1"))

    def test_setting_selects_the_number_of_groups_or_the_old_single_process(self):
        self.env("3")
        with mock.patch.object(selfcheck, "_run_tree", return_value=(0, PARALLEL_OK)) as tree:
            selfcheck.run_tests()
        cmd = tree.call_args[0][0]
        self.assertEqual(cmd[cmd.index("--jobs") + 1], "3")
        self.env("1")
        with mock.patch.object(selfcheck, "_run_tree") as tree, self.sequential_ok():
            result = selfcheck.run_tests()
        tree.assert_not_called()
        self.assertEqual((result["ok"], result["mode"]), (True, "sequential"))

    def test_one_processor_runs_sequentially(self):
        self.env(None)
        with mock.patch.object(selfcheck.os, "cpu_count", return_value=1), \
                mock.patch.object(selfcheck, "_run_tree") as tree, self.sequential_ok():
            result = selfcheck.run_tests()
        tree.assert_not_called()
        self.assertEqual(result["mode"], "sequential")

    def test_failed_tests_are_reported_and_do_not_trigger_the_fallback(self):
        self.env("4")
        with mock.patch.object(selfcheck, "_run_tree", return_value=(1, PARALLEL_FAILED)), \
                mock.patch.object(selfcheck.subprocess, "run") as seq:
            result = selfcheck.run_tests()
        seq.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertEqual((result["failures"], result["exit_code"], result["mode"]), (1, 1, "parallel"))
        self.assertNotIn("fallback", result)

    def test_zero_exit_code_is_required_even_if_the_summary_says_ok(self):
        self.env("4")
        with mock.patch.object(selfcheck, "_run_tree", return_value=(5, PARALLEL_OK)):
            result = selfcheck.run_tests()
        self.assertFalse(result["ok"])

    def test_launcher_that_gave_no_total_falls_back_to_one_process_and_says_why(self):
        self.env("4")
        broken = "РАСКЛАДКА: модуль test_x не попал ни в одну группу\nRan 0 tests in 0.000s\n\nFAILED (run problems=1)\n"
        with mock.patch.object(selfcheck, "_run_tree", return_value=(2, "traceback без итога")), self.sequential_ok():
            result = selfcheck.run_tests()
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "sequential")
        self.assertIn("не дал итога (код 2)", result["fallback"])
        with mock.patch.object(selfcheck, "_run_tree", return_value=(2, broken)), self.sequential_ok():
            self.assertIn("не дал итога", selfcheck.run_tests()["fallback"])

    def test_missing_launcher_or_failed_start_falls_back(self):
        self.env("4")
        (self.root / "tests" / "run_parallel.py").unlink()
        with mock.patch.object(selfcheck, "_run_tree") as tree, self.sequential_ok():
            result = selfcheck.run_tests()
        tree.assert_not_called()
        self.assertEqual(result["fallback"], "нет tests/run_parallel.py")
        (self.root / "tests" / "run_parallel.py").write_text("#\n", encoding="utf-8")
        with mock.patch.object(selfcheck, "_run_tree", side_effect=OSError("нет доступа")), self.sequential_ok():
            self.assertIn("нет доступа", selfcheck.run_tests()["fallback"])

    def test_fallback_gets_only_the_time_that_is_left_and_is_refused_when_there_is_none(self):
        self.env("4")
        with mock.patch.object(selfcheck, "_run_tree", return_value=(2, "без итога")), self.sequential_ok() as seq:
            selfcheck.run_tests(timeout=720)
        self.assertLessEqual(seq.call_args.kwargs["timeout"], 720)
        with mock.patch.object(selfcheck, "_run_tree", return_value=(2, "без итога")), \
                mock.patch.object(selfcheck.subprocess, "run") as seq:
            result = selfcheck.run_tests(timeout=100)
        seq.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertIn("не осталось времени", result["error"])

    def test_timeout_names_the_limit_and_does_not_fall_back(self):
        self.env("4")
        with mock.patch.object(selfcheck, "_run_tree", side_effect=selfcheck.subprocess.TimeoutExpired("x", 720)), \
                mock.patch.object(selfcheck.subprocess, "run") as seq:
            result = selfcheck.run_tests(timeout=720)
        seq.assert_not_called()
        self.assertEqual((result["ok"], result["ran"], result["error"]), (False, 0, "тесты не уложились в 12 мин (720 с)"))

    def test_installer_shows_launch_problems_and_the_fallback_note(self):
        text = (Path(__file__).resolve().parent.parent / "install.ps1").read_text(encoding="utf-8-sig")
        step = text[text.index('Step "11/11"'):]
        self.assertIn("$t.run_problems", step)
        self.assertIn("$t.fallback", step)


class ParallelSelfCheckEndToEnd(unittest.TestCase):
    """Настоящий запуск: игрушечный набор из трёх модулей через run_tests и tests/run_parallel.py, без нашего набора."""

    def build(self, failing=False):
        tmp = tempfile.TemporaryDirectory(prefix="rt-selfcheck-e2e-")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "tests").mkdir()
        shutil.copy(Path(__file__).resolve().parent / "run_parallel.py", root / "tests" / "run_parallel.py")
        for name, body in (("test_a", "self.assertTrue(True)"), ("test_b", "self.assertEqual(1, 1)"),
                           ("test_c", "self.assertEqual(1, 2)" if failing else "self.assertTrue(True)")):
            (root / "tests" / f"{name}.py").write_text(
                "import unittest\n\nclass T(unittest.TestCase):\n    def test_one(self):\n        " + body + "\n"
                "    def test_two(self):\n        pass\n", encoding="utf-8")
        patcher = mock.patch.object(selfcheck, "HERE", root)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(selfcheck.os.environ, {selfcheck.JOBS_ENV: "2"})
        env.start()
        self.addCleanup(env.stop)

    def test_real_parallel_run_of_a_small_suite_is_counted_and_ok(self):
        self.build()
        result = selfcheck.run_tests(timeout=300)
        self.assertEqual((result["ok"], result["ran"], result["mode"], result["groups"], result["exit_code"]),
                         (True, 6, "parallel", 2, 0), result)
        self.assertNotIn("fallback", result)

    def test_real_failure_in_one_group_is_reported_with_the_test_name(self):
        self.build(failing=True)
        result = selfcheck.run_tests(timeout=300)
        self.assertFalse(result["ok"], result)
        self.assertEqual((result["ran"], result["failures"], result["mode"]), (6, 1, "parallel"))
        self.assertEqual([f["test"] for f in result["failed"]], ["test_c.T.test_one"])
        self.assertNotIn("fallback", result)



class InstallerSelfCheckStepTests(unittest.TestCase):
    """Шаг 10 install.ps1: обещанный предел совпадает с selfcheck, причина без чисел показывается текстом."""

    def setUp(self):
        self.text = (Path(__file__).resolve().parent.parent / "install.ps1").read_text(encoding="utf-8-sig")

    def test_installer_announces_the_same_limit_as_selfcheck(self):
        self.assertEqual(selfcheck.TESTS_TIMEOUT % 60, 0)
        minutes = selfcheck.TESTS_TIMEOUT // 60
        self.assertIn(f"предел — {minutes} мин. Окно не закрывайте", self.text)
        # объявление стоит до запуска selfcheck.py, иначе человек видит молчание
        self.assertLess(self.text.index("Окно не закрывайте"), self.text.index('(Join-Path $Target "selfcheck.py")'))

    def test_installer_shows_the_error_instead_of_empty_counts(self):
        step = self.text[self.text.index('Step "11/11"'):]
        error_at = step.index("if ($t -and $t.error) {")
        counts_at = step.index("тестов {1}, падений {2}")
        self.assertLess(error_at, step.index("} else {", error_at))
        self.assertLess(step.index("} else {", error_at), counts_at)
        self.assertIn("самопроверка: $($t.error)", step)

if __name__ == "__main__":
    unittest.main()
