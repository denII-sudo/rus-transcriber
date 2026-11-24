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
from unittest import mock

import run_parallel as rp

REAL_TESTS = Path(__file__).resolve().parent

OK_MODULE = "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n\n    def test_b(self):\n        pass\n"
ENV_MODULE = ("import os\nimport unittest\n\n\nclass T(unittest.TestCase):\n    def test_env(self):\n"
              "        keys = ('RUS_TRANSCRIBER_DATA', 'TEMP', 'TMP', 'HF_HOME', 'TORCH_HOME', 'HF_HUB_CACHE')\n"
              "        print('ENV|' + '|'.join(os.environ.get(k, '-') for k in keys))\n")   # «|», чтобы пробел в пути не мешал
ENV_FIELD = r"([^|\r\n]*)"
ENV_LINE = "ENV\\|" + "\\|".join([ENV_FIELD] * 6)
EXIT3_MODULE = ("import atexit\nimport os\nimport unittest\n\natexit.register(lambda: os._exit(3))   # код выхода 3 после «OK»\n\n\n"
                "class T(unittest.TestCase):\n    def test_a(self):\n        pass\n")


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
        try:
            code = rp.run(list(args), repo=self.repo, out=out, mutate=mutate)
        finally:
            self.clean_left_sandboxes(out.getvalue())
        return code, out.getvalue()

    def left_sandboxes(self, text):
        return [Path(p) for p in re.findall(r"песочница оставлена для разбора: (.+)", text)]

    def clean_left_sandboxes(self, text):
        # запускатель после неудачи оставляет песочницу (AGENTS.md); тест её убирает в конце, а не оставляет мусор в TEMP
        for path in self.left_sandboxes(text):
            self.addCleanup(shutil.rmtree, path, ignore_errors=True)


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

    def test_seed_shuffles_only_the_order_inside_groups_not_their_content(self):
        modules = [f"test_m{i}" for i in range(20)]
        seconds = {m: 1.0 + i for i, m in enumerate(modules)}
        base = rp.plan_groups(modules, seconds, 4)
        a = rp.plan_groups(modules, seconds, 4, seed=5)
        self.assertEqual(a, rp.plan_groups(modules, seconds, 4, seed=5))          # seed повторяет порядок
        orders = set()
        for s in range(8):
            groups = rp.plan_groups(modules, seconds, 4, seed=s)
            self.assertEqual([sorted(g) for g in groups], base, "seed изменил состав групп")
            orders.add(json.dumps(groups))
        self.assertGreater(len(orders), 4)                                        # а порядок внутри групп меняется

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


CLASSIFIED = '''import unittest

BACKENDS = ("legacy_files", "sqlite")


def helper_both():
    for backend in BACKENDS:
        pass


class A(unittest.TestCase):
    def loop_both(self):
        for backend in ("legacy_files", "sqlite"):
            pass

    def test_loop(self):
        for backend in BACKENDS:
            pass

    def test_through_class_helper(self):
        self.loop_both()

    def test_through_module_helper(self):
        helper_both()

    def test_only_files(self):
        mode = "legacy_files"

    def test_plain(self):
        pass


class Pinned(unittest.TestCase):
    backend = "legacy_files"

    def test_pinned(self):
        pass
'''


class LegacyEstimateTests(Synthetic):
    """Время режима legacy_files отделено в таблице, раскладка ровна и до, и после его ухода."""

    def test_tests_are_classified_by_their_text(self):
        self.module("test_m", CLASSIFIED)
        self.assertEqual(rp.classify_tests(self.repo / "tests"), {
            "test_m.A.test_loop": "both", "test_m.A.test_through_class_helper": "both",
            "test_m.A.test_through_module_helper": "both", "test_m.A.test_only_files": "legacy",
            "test_m.Pinned.test_pinned": "legacy"})

    def test_estimate_after_removal_keeps_sqlite_share_of_both_mode_tests_only(self):
        self.module("test_m", CLASSIFIED)
        times = {"test_m.A.test_loop": 11.0, "test_m.A.test_only_files": 2.0, "test_m.A.test_plain": 3.0,
                 "test_other.B.test_x": 5.0}
        table = rp.build_table(times, self.repo / "tests")
        self.assertEqual(table["modules"], {"test_m": 16.0, "test_other": 5.0})
        # 11 с гоняют оба режима: после ухода остаётся доля sqlite 4,5/5,5 = 9 с; «только файлы» исчезает; прочее как было
        self.assertEqual(table["after_legacy_removal"], {"test_m": 12.0, "test_other": 5.0})

    def test_without_legacy_files_in_tests_the_estimate_equals_the_current_time(self):
        self.module("test_m", OK_MODULE)
        table = rp.build_table({"test_m.T.test_a": 3.0}, self.repo / "tests")
        self.assertEqual(table["modules"], table["after_legacy_removal"])

    def test_table_is_rebuilt_from_a_saved_log_with_prefixes_and_timestamps(self):
        self.module("test_m", CLASSIFIED)
        self.module("test_keep")
        self.table(test_keep=7.0, test_gone=9.0)                                     # test_gone в наборе уже нет
        log = self.repo / "run.log"
        log.write_text("tests\tRun\t2026-11-25T01:00:00.1Z .....\n"
                       "tests\tRun\t2026-11-25T01:00:01.0Z [g1] Slowest test durations\n"
                       "tests\tRun\t2026-11-25T01:00:01.1Z [g1] 11.000s     test_loop (test_m.A.test_loop)\n"
                       "tests\tRun\t2026-11-25T01:00:01.2Z [g2] 1.500s     test_plain (test_m.A.test_plain)\n"
                       "tests\tRun\t2026-11-25T01:00:01.3Z [g1] test_plain (test_m.A.test_plain) ... ok\n",
                       encoding="utf-8")
        code, text = self.run_launcher("--table-from", str(log))
        self.assertEqual(code, 0, text)
        data = json.loads((self.repo / "tests" / "parallel_durations.json").read_text(encoding="utf-8"))
        self.assertEqual(data["modules"], {"test_keep": 7.0, "test_m": 12.5})     # неупомянутый модуль сохранён, ушедший убран
        self.assertEqual(data["after_legacy_removal"]["test_m"], 10.5)
        self.assertNotIn("test_gone", data["after_legacy_removal"])
        code, text = self.run_launcher("--table-from", str(self.repo / "tests" / "test_m.py"))
        self.assertEqual(code, 2, text)                                              # в файле нет блока длительностей

    def test_plan_prints_loads_in_both_states_without_running_anything(self):
        for name in ("test_a", "test_b", "test_c", "test_d"):
            self.module(name, "raise SystemExit('не должен запускаться')\n")
        (self.repo / "tests" / "parallel_durations.json").write_text(json.dumps({
            "modules": {"test_a": 10, "test_b": 0, "test_c": 5, "test_d": 5},
            "after_legacy_removal": {"test_a": 0, "test_b": 10, "test_c": 5, "test_d": 5}}), encoding="utf-8")
        code, text = self.run_launcher("--plan", "--jobs", "2")
        self.assertEqual(code, 0, text)
        self.assertRegex(text, r"g1: .*сейчас 10 с, после ухода legacy_files 10 с")
        self.assertRegex(text, r"g2: .*сейчас 10 с, после ухода legacy_files 10 с")
        self.assertIn("самая длинная группа: сейчас 10 с, после 10 с", text)

    def test_layout_is_even_in_both_states_where_averaging_would_not_be(self):
        modules = ["a", "b", "c", "d"]
        now = {"a": 10, "b": 0, "c": 5, "d": 5}
        after = {"a": 0, "b": 10, "c": 5, "d": 5}
        groups = rp.plan_groups(modules, now, 2, after=after)
        self.assertEqual(rp.verify_plan(modules, groups), [])
        self.assertEqual([sum(now[m] for m in g) for g in groups], [10, 10])
        self.assertEqual([sum(after[m] for m in g) for g in groups], [10, 10])

    def test_shipped_table_has_the_estimate_for_every_module_and_layout_is_even(self):
        data = json.loads((REAL_TESTS / "parallel_durations.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(data["modules"]), sorted(data["after_legacy_removal"]))
        modules = rp.list_modules(REAL_TESTS)
        seconds, _ = rp.load_table(REAL_TESTS / "parallel_durations.json")
        after = rp.load_after(REAL_TESTS / "parallel_durations.json")
        groups = rp.plan_groups(modules, seconds, 4, after=after)
        self.assertEqual(rp.verify_plan(modules, groups), [])
        for table in (seconds, after):
            loads = [sum(table.get(m, 0.0) for m in g) for g in groups]
            self.assertLess(max(loads) - min(loads), 0.1 * max(loads) + 15, loads)


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
        with mock.patch.dict(os.environ, {"HF_HOME": "outer-hf-home", "HF_HUB_CACHE": "outer-hub-cache"}):
            code, text = self.run_launcher("--jobs", "3", "-v")
        self.assertEqual(code, 0, text)
        seen = re.findall(ENV_LINE, text)
        self.assertEqual(len(seen), 3, text)
        self.assertEqual(len({s[0] for s in seen}), 3)                      # разные каталоги данных
        self.assertEqual(len({s[1] for s in seen}), 3)
        for data, temp, tmp, hf_home, torch_home, hub_cache in seen:
            sandbox = Path(data).parent.parent
            self.assertEqual(temp, tmp)
            self.assertIn(Path(temp), Path(data).parents)               # как требует test_data_safety: данные под TEMP
            self.assertNotEqual(data, os.environ.get("RUS_TRANSCRIBER_DATA"))
            self.assertIn(sandbox, Path(hf_home).parents)                # кэши моделей — в песочнице группы, не внешние
            self.assertIn(sandbox, Path(torch_home).parents)
            self.assertEqual(hub_cache, "-", "HF_HUB_CACHE снаружи не должен доходить до группы")
            self.assertFalse(sandbox.exists(), "песочница не убрана")
        self.assertEqual(len({s[3] for s in seen}), 3)
        self.assertRegex(text, r"\[g\d\] ")
        self.assertNotIn("песочница оставлена", text)

    def test_sandbox_stays_after_a_failure_and_its_path_is_printed(self):
        self.module("test_good")
        self.module("test_broken", fail_module())
        code, text = self.run_launcher("--jobs", "2")
        self.assertEqual(code, 1, text)
        left = self.left_sandboxes(text)
        self.assertEqual(len(left), 1, text)
        self.assertTrue(left[0].is_dir(), "песочница после неудачного прогона должна остаться")
        self.assertTrue((left[0] / "g1").exists() or (left[0] / "g2").exists())
        self.assertTrue(text.rstrip().splitlines()[-1].startswith("песочница оставлена"), text)   # путь — в итоге

    def test_group_that_prints_ok_but_exits_with_a_nonzero_code_is_a_failure(self):
        self.module("test_good")
        self.module("test_exit3", EXIT3_MODULE)
        code, text = self.run_launcher("--jobs", "2")
        self.assertEqual(code, 1, text)
        self.assertRegex(text, r"НАРУШЕНИЕ: g\d: итог unittest OK, но процесс вышел с кодом 3")
        self.assertIn("FAILED", text)
        self.assertEqual(text.count("полный вывод упавшей группы"), 1)

    def test_interrupt_kills_the_groups_and_leaves_the_sandbox(self):
        self.module("test_good")
        self.module("test_hangs", "import time\nimport unittest\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                                  "        time.sleep(120)\n")
        spawned = []
        real_spawn = rp.spawn

        def spy(argv, cwd, env):
            proc = real_spawn(argv, cwd, env)
            spawned.append(proc)
            return proc

        real_wait = rp.Run.wait

        def interrupted(run, deadline):
            if run.label == "g1":
                time.sleep(1.5)                         # группы уже начаты
                raise KeyboardInterrupt
            return real_wait(run, deadline)

        out = io.StringIO()
        with mock.patch.object(rp, "spawn", spy), mock.patch.object(rp.Run, "wait", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                rp.run(["--jobs", "2", "--no-count"], repo=self.repo, out=out)
        self.clean_left_sandboxes(out.getvalue())
        self.assertGreaterEqual(len(spawned), 2)
        deadline = time.time() + 30
        while time.time() < deadline and any(p.poll() is None for p in spawned):
            time.sleep(0.2)
        self.assertTrue(all(p.poll() is not None for p in spawned), "группы остались работать после прерывания")
        left = self.left_sandboxes(out.getvalue())
        self.assertEqual(len(left), 1, out.getvalue())
        self.assertTrue(left[0].is_dir())

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

    @unittest.skipIf(sys.version_info < (3, 12), "--durations есть в Python 3.12")
    def test_update_table_with_a_module_in_the_table_that_is_not_in_the_suite(self):
        # раньше имя `after` в запускателе было занято и таблицей, и снимком дерева: падение при модуле «из таблицы, не из набора»
        slow = ("import time\nimport unittest\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                "        time.sleep(0.05)\n")
        self.module("test_a", slow)
        self.module("test_b", slow)
        (self.repo / "tests" / "parallel_durations.json").write_text(json.dumps({
            "modules": {"test_a": 1.0, "test_gone": 9.0}, "after_legacy_removal": {"test_a": 1.0, "test_gone": 9.0},
            "together": []}), encoding="utf-8")
        code, text = self.run_launcher("--jobs", "2", "--update-table")
        self.assertEqual(code, 0, text)
        self.assertNotIn("Traceback", text)
        data = json.loads((self.repo / "tests" / "parallel_durations.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(data["modules"]), ["test_a", "test_b"])
        self.assertEqual(sorted(data["after_legacy_removal"]), ["test_a", "test_b"])


if __name__ == "__main__":
    unittest.main()
