"""The answer phrase table preserves the exact closed lists used by freeform."""
import unittest

import freeform


EXPECTED_AGREE = {
    "да", "верно", "так", "правильно", "все верно", "всё верно", "да верно",
    "ага", "угу", "так и есть", "как есть", "да так",
}
EXPECTED_NEGATIVE_PREFIXES = ("нет", "это", "я", "нет-нет")
EXPECTED_NEGATIVE_INTENSIFIERS = ("совершенно", "совсем", "абсолютно")
EXPECTED_NEGATIVE_CORES = (
    "неверно", "не верно", "неправильно", "не правильно", "не согласен",
    "не согласна", "не согласны", "не согласен с этим", "не согласна с этим",
    "не согласны с этим",
)
EXPECTED_NEGATIVES = frozenset(
    " ".join(part for part in (prefix, intensifier, core) if part)
    for prefix in ("",) + EXPECTED_NEGATIVE_PREFIXES
    for intensifier in ("",) + EXPECTED_NEGATIVE_INTENSIFIERS
    for core in EXPECTED_NEGATIVE_CORES
)


class AnswerPhraseSnapshotTests(unittest.TestCase):
    def test_current_agree_and_negative_sets_are_preserved(self):
        self.assertEqual(freeform.AGREE, EXPECTED_AGREE)
        self.assertEqual(freeform.NEGATIVES, EXPECTED_NEGATIVES)


if __name__ == "__main__":
    unittest.main()
