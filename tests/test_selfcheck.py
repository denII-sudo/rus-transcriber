import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
