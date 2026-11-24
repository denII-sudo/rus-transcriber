"""Параллельный запуск тестов (tests/run_parallel.py): раскладка, сверки, итог, нарочные поломки.

Все проверки — на синтетических модулях во временном «репозитории»; настоящий набор не запускается.
"""
import fnmatch
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import run_parallel as rp

REAL_TESTS = Path(__file__).resolve().parent

OK_MODULE = "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n\n    def test_b(self):\n        pass\n"
ENV_MODULE = ("import os\nimport unittest\n\n\nclass T(unittest.TestCase):\n    def test_env(self):\n"
              "        print('ENV', os.environ['RUS_TRANSCRIBER_DATA'], os.environ['TEMP'], os.environ['TMP'])\n")


def fail_module():
    return ("import unittest\n\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        pass\n\n"
            "    def test_bad(self):\n        self.assertEqual(1, 2, 'нарочное падение')\n")


class Synthetic(unittest.TestCase):
    """Временный репозиторий с папкой tests/ из синтетических модулей."""

    def setUp(self):
        self.repo = Path(tempfile.mkdtemp(prefix="rt-runpar-"))
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        (self.repo / "tests").mkdir()

    def module(self, name, source=OK_MODULE):
        (self.repo / "tests" / f"{name}.py").write_text(source, encoding="utf-8", newline="\n")

    def table(self, **seconds):
        (self.repo / "tests" / "parallel_durations.json").write_text(
            json.dumps({"modules": seconds, "together": []}), encoding="utf-8")

    def run_launcher(self, *args, mutate=None):
        out = io.StringIO()
        code = rp.run(list(args), repo=self.repo, out=out, mutate=mutate)
        return code, out.getvalue()


class DiscoveryAndPlanTests(unittest.TestCase):
    def test_launcher_and_table_are_not_picked_up_by_discover(self):
        for name in ("run_parallel.py", "parallel_durations.json"):
            self.assertFalse(fnmatch.fnmatch(name, rp.PATTERN), name)
        modules = rp.list_modules(REAL_TESTS)
        self.assertNotIn("run_parallel", modules)
        self.assertIn("test_run_parallel", modules)
        self.assertEqual(modules, sorted(p.stem for p in REAL_TESTS.glob("test*.py")))

    def test_shipped_table_is_readable_and_positive(self):
        seconds, together = rp.load_table(REAL_TESTS / "parallel_durations.json")
        self.assertGreater(len(seconds), 50)
        self.assertTrue(all(v >= 0 for v in seconds.values()))
        self.assertEqual(together, [])
        self.assertEqual(rp.verify_plan(rp.list_modules(REAL_TESTS),
                                        rp.plan_groups(rp.list_modules(REAL_TESTS), seconds, 4)), [])

    def test_every_module_goes_to_exactly_one_group_for_any_seed_and_jobs(self):
        modules = [f"test_m{i}" for i in range(23)]
        seconds = {m: float(i % 7) for i, m in enumerate(modules)}
        for jobs in (1, 2, 4, 30):
            for seed in (None, 0, 1, 12345):
                groups = rp.plan_groups(modules, seconds, jobs, seed=seed)
                with self.subTest(jobs=jobs, seed=seed):
                    self.assertEqual(rp.verify_plan(modules, groups), [])
                    self.assertLessEqual(len(groups), min(jobs, len(modules)))
                    self.assertTrue(all(groups))

    def test_groups_are_balanced_and_stable_without_seed(self):
        modules = ["test_a", "test_b", "test_c", "test_d", "test_e"]
        seconds = {"test_a": 10.0, "test_b": 6.0, "test_c": 5.0, "test_d": 4.0, "test_e": 1.0}
        first = rp.plan_groups(modules, seconds, 2)
        self.assertEqual(first, rp.plan_groups(modules, seconds, 2))
        loads = sorted(sum(seconds[m] for m in g) for g in first)
        self.assertEqual(sum(loads), 26.0)
        self.assertLessEqual(loads[1] - loads[0], 2.0)                      # «самый тяжёлый — в самую лёгкую»: 14 и 12

    def test_seed_replays_the_same_layout_and_other_seeds_differ(self):
        modules = [f"test_m{i}" for i in range(20)]
        seconds = {m: 1.0 + i for i, m in enumerate(modules)}
        a = rp.plan_groups(modules, seconds, 4, seed=5)
        self.assertEqual(a, rp.plan_groups(modules, seconds, 4, seed=5))
        layouts = {json.dumps(rp.plan_groups(modules, seconds, 4, seed=s)) for s in range(8)}
        self.assertGreater(len(layouts), 4)

    def test_modules_marked_together_stay_in_one_group(self):
        modules = [f"test_m{i}" for i in range(10)]
        seconds = {m: 1.0 for m in modules}
        for seed in (None, 1, 2, 3):
            groups = rp.plan_groups(modules, seconds, 4, together=[["test_m1", "test_m8"]], seed=seed)
            holder = [g for g in groups if "test_m1" in g]
            self.assertIn("test_m8", holder[0])

    def test_missing_duplicate_and_unknown_modules_are_caught(self):
        modules = ["test_a", "test_b", "test_c"]
        self.assertEqual(rp.verify_plan(modules, [["test_a", "test_b"], ["test_c"]]), [])
        self.assertIn("модуль test_c не попал ни в одну группу", rp.verify_plan(modules, [["test_a"], ["test_b"]]))
        twice = rp.verify_plan(modules, [["test_a", "test_b"], ["test_b", "test_c"]])
        self.assertEqual(twice, ["модуль test_b в нескольких группах: [1, 2]"])
        self.assertIn("в группах есть модуль test_z, которого нет в наборе",
                      rp.verify_plan(modules, [["test_a", "test_b", "test_c"], ["test_z"]]))

    def test_unittest_summary_is_parsed(self):
        ok = rp.parse_summary("....\n" + "-" * 70 + "\nRan 12 tests in 3.456s\n\nOK (skipped=1, expected failures=2)\n")
        self.assertEqual((ok["ran"], ok["ok"], ok["skipped"], ok["expected"]), (12, True, 1, 2))
        bad = rp.parse_summary("Ran 5 test in 0.1s\n\nFAILED (failures=1, errors=2, unexpected successes=1)\n")
        self.assertEqual((bad["ran"], bad["ok"], bad["failures"], bad["errors"], bad["unexpected"]),
                         (5, False, 1, 2, 1))
        self.assertIsNone(rp.parse_summary("Ran 5 tests in 0.1s\n"))
        self.assertIsNone(rp.parse_summary(""))

    def test_durations_block_is_summed_by_module(self):
        text = "Slowest test durations\n1.500s     test_x (test_one.A.test_x)\n0.500s     test_y (test_one.B.test_y)\n" \
               "2.000s     test_z (test_two.A.test_z)\n"
        self.assertEqual(rp.module_durations(text), {"test_one": 2.0, "test_two": 2.0})


class RunTests(Synthetic):
    def test_all_modules_run_once_and_numbers_equal_discover(self):
        for name in ("test_a", "test_b", "test_c"):
            self.module(name)
        code, text = self.run_launcher("--jobs", "2")
        self.assertEqual(code, 0, text)
        self.assertRegex(text, r"Ran 6 tests in [\d.]+s")
        self.assertIn("\nOK\n", text)
        # то же число даёт один процесс discover
        code, seq = self.run_launcher("--sequential")
        self.assertEqual(code, 0, seq)
        self.assertRegex(seq, r"Ran 6 tests in [\d.]+s")

    def test_failing_test_is_seen_in_the_total_with_full_output_of_its_group(self):
        self.module("test_good")
        self.module("test_broken", fail_module())
        self.module("test_other")
        code, text = self.run_launcher("--jobs", "3")
        self.assertEqual(code, 1, text)
        self.assertIn("FAILED (failures=1", text)
        self.assertRegex(text, r"Ran 6 tests in")
        self.assertIn("нарочное падение", text)
        self.assertIn("AssertionError", text)
        self.assertEqual(text.count("полный вывод упавшей группы"), 1)      # только у упавшей группы
        code, seq = self.run_launcher("--sequential")
        self.assertEqual(code, 1, seq)
        self.assertIn("FAILED (failures=1", seq)

    def test_module_that_cannot_be_imported_is_an_error_not_a_loss(self):
        self.module("test_good")
        self.module("test_noimport", "import module_that_does_not_exist\n")
        code, text = self.run_launcher("--jobs", "2")
        self.assertEqual(code, 1, text)
        self.assertIn("errors=1", text)

    def test_module_dropped_from_the_layout_is_caught_before_anything_runs(self):
        for name in ("test_a", "test_b", "test_c"):
            self.module(name)
        code, text = self.run_launcher("--jobs", "2", mutate=lambda groups: [[m for m in g if m != "test_b"] for g in groups])
        self.assertEqual(code, 2, text)
        self.assertIn("РАСКЛАДКА: модуль test_b не попал ни в одну группу", text)
        self.assertIn("Ran 0 tests", text)
        self.assertIn("FAILED", text)

    def test_module_run_twice_is_caught(self):
        for name in ("test_a", "test_b"):
            self.module(name)
        code, text = self.run_launcher("--jobs", "2", mutate=lambda groups: groups + [["test_a"]])
        self.assertEqual(code, 2, text)
        self.assertIn("РАСКЛАДКА: модуль test_a в нескольких группах", text)

    def test_count_that_differs_from_discover_is_caught(self):
        # discover передаёт load_tests шаблон поиска, запуск по имени — нет: второй тест виден только discover
        self.module("test_split", OK_MODULE + (
            "\n\ndef load_tests(loader, tests, pattern):\n"
            "    if pattern is not None:\n"
            "        class Extra(unittest.TestCase):\n"
            "            def test_only_in_discover(self):\n"
            "                pass\n"
            "        tests.addTests(loader.loadTestsFromTestCase(Extra))\n"
            "    return tests\n"))
        self.module("test_plain")
        code, text = self.run_launcher("--jobs", "2")
        self.assertEqual(code, 1, text)
        self.assertRegex(text, r"НАРУШЕНИЕ: g\d: запущено 2 тестов, по discover у её модулей 3")

    def test_group_without_summary_line_is_a_failure(self):
        self.module("test_good")
        self.module("test_dies", "import os\nimport unittest\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                                 "        os._exit(3)\n")
        code, text = self.run_launcher("--jobs", "2")
        self.assertEqual(code, 1, text)
        self.assertIn("нет итоговой строки unittest", text)
        self.assertIn("run problems=", text)

    def test_hanging_group_is_killed_and_reported(self):
        self.module("test_good")
        self.module("test_hangs", "import time\nimport unittest\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                                  "        time.sleep(120)\n")
        started = time.time()
        code, text = self.run_launcher("--jobs", "2", "--timeout", "4")
        self.assertEqual(code, 1, text)
        self.assertLess(time.time() - started, 60)
        self.assertIn("превышен срок 4 с, группа убита", text)

    def test_every_group_gets_its_own_sandbox_which_is_removed(self):
        for name in ("test_a", "test_b", "test_c"):
            self.module(name, ENV_MODULE)
        code, text = self.run_launcher("--jobs", "3", "-v")
        self.assertEqual(code, 0, text)
        seen = re.findall(r"ENV (\S+) (\S+) (\S+)", text)
        self.assertEqual(len(seen), 3, text)
        self.assertEqual(len({s[0] for s in seen}), 3)                      # разные каталоги данных
        self.assertEqual(len({s[1] for s in seen}), 3)
        for data, temp, tmp in seen:
            self.assertEqual(temp, tmp)
            self.assertNotEqual(data, os.environ.get("RUS_TRANSCRIBER_DATA"))
            self.assertFalse(Path(data).parent.parent.exists(), "песочница не убрана")
        self.assertRegex(text, r"\[g\d\] ")

    def test_leftover_in_the_group_temp_is_a_warning_and_strict_makes_it_a_failure(self):
        self.module("test_leaks", "import os\nimport tempfile\nimport unittest\n\n\nclass T(unittest.TestCase):\n"
                                  "    def test_x(self):\n        fd, _ = tempfile.mkstemp()\n        os.close(fd)\n")
        self.module("test_clean")
        code, text = self.run_launcher("--jobs", "2")
        self.assertEqual(code, 0, text)
        self.assertIn("предупреждение:", text)
        code, text = self.run_launcher("--jobs", "2", "--strict-tmp")
        self.assertEqual(code, 1, text)
        self.assertIn("НАРУШЕНИЕ:", text)

    def test_changed_repository_tree_is_caught(self):
        if shutil.which("git") is None:
            self.skipTest("нет git")
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True, capture_output=True)
        self.module("test_clean")
        self.module("test_writes", "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                                   "        open('stray-file.txt', 'w').write('x')\n")
        code, text = self.run_launcher("--jobs", "2")
        self.assertEqual(code, 1, text)
        self.assertIn("дерево репозитория изменилось", text)
        self.assertIn("stray-file.txt", text)
        (self.repo / "stray-file.txt").unlink()
        self.module("test_writes")                                          # прежний безобидный модуль
        code, text = self.run_launcher("--jobs", "2")
        self.assertEqual(code, 0, text)

    def test_shuffle_prints_seed_and_a_given_seed_is_repeatable(self):
        for name in ("test_a", "test_b", "test_c", "test_d"):
            self.module(name)
        self.table(test_a=4.0, test_b=3.0, test_c=2.0, test_d=1.0)
        code, text = self.run_launcher("--jobs", "2", "--shuffle", "77")
        self.assertEqual(code, 0, text)
        self.assertIn("seed=77", text)
        self.assertIn("seed 77", text)
        code, text = self.run_launcher("--jobs", "2", "--shuffle")
        self.assertEqual(code, 0, text)
        self.assertRegex(text, r"seed=\d+")

    @unittest.skipIf(sys.version_info < (3, 12), "--durations есть в Python 3.12")
    def test_update_table_writes_module_seconds(self):
        slow = ("import time\nimport unittest\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                "        time.sleep(0.05)\n")                               # короче 0,001 с unittest в список не берёт
        self.module("test_a", slow)
        self.module("test_b", slow)
        code, text = self.run_launcher("--jobs", "2", "--update-table")
        self.assertEqual(code, 0, text)
        data = json.loads((self.repo / "tests" / "parallel_durations.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(data["modules"]), ["test_a", "test_b"])


if __name__ == "__main__":
    unittest.main()
