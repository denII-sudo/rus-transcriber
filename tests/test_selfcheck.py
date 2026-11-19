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

        with mock.patch.object(selfcheck.subprocess, "run", side_effect=run):
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


if __name__ == "__main__":
    unittest.main()
