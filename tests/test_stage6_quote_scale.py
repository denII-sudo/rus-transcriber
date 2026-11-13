"""Этап 6: синтетический масштабный контроль цитат без моделей и пользовательских данных."""
import unittest

import quotes


class QuoteScale(unittest.TestCase):
    def test_many_identical_occurrences_remain_ambiguous(self):
        tokens = [("слово", "Анна", i * 0.2, None) for i in range(10000)]
        matches = quotes.candidates(["слово"] * 4, tokens, min_score=1.0)
        self.assertEqual(len(matches), 9997)


if __name__ == "__main__":
    unittest.main()
