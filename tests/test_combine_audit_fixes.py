"""Исправления по независимому аудиту сведения нескольких моделей (words.combine_hypotheses, vote_n).

1. Модель без гипотезы (None) голосование не ограничивает: сбойные куски третьей модели не исключают
   отрезки, если у неё нет гипотезы (это видно только при дополнительных моделях).
2. Имя дополнительной роли не должно давать поле слова, которое уже занято: alt_<роль> для роли span
   совпало бы с alt_span (длиной спорного отрезка). Такие имена отклоняются (ValueError).

Только синтетика. При трёх моделях результат прежний — это закреплено отпечатками в
tests/test_combine_n_models.py и tests/test_run_words_golden.py.
"""
import unittest

import words
from tests.test_combine_n_models import ten, text, ok_diag, TEN

RULE = {"keep": 2, "replace": 2}


def diag_with_failed_whisper(*roles):
    diag = ok_diag(*roles)
    diag["whisper"] = {"chunks": [{"state": "error", "own_span": [3.0, 6.0]},
                                  {"state": "ok", "own_span": [0.0, 3.0]},
                                  {"state": "ok", "own_span": [6.0, 10.0]}]}
    return diag


class AbsentModelDoesNotRestrictVoteTests(unittest.TestCase):

    def run_case(self, whisper):
        main = ten()
        got = words.combine_hypotheses(
            {"primary": main, "second": ten({4: "пятак"}), "whisper": whisper, "fourth": ten({4: "пятак"})},
            diag_with_failed_whisper("primary", "second", "fourth"), rule=RULE)
        return main, got

    def test_failed_chunks_of_absent_third_model_do_not_exclude_spans(self):
        main, got = self.run_case(None)
        self.assertNotIn("vote3_partial", got["quality"])       # голосование по всей записи
        self.assertEqual(main[4]["w"], "пятак")                   # вторая и четвёртая модели заменили слово
        self.assertEqual(main[4]["voted"], "ctc+fourth")
        self.assertEqual(got["quality"]["vote"]["replaced"], 1)

    def test_failed_chunks_of_present_third_model_still_exclude_their_span(self):
        main, got = self.run_case(text(ten()))
        self.assertEqual(got["quality"]["vote3_partial"], {"reason": "whisper_incomplete", "spans": 1,
                                                          "seconds": 3.0})
        self.assertEqual(got["evidence"]["whisper"], [(3.0, 6.0, "error", ["primary", "second"])])
        self.assertEqual(main[4]["w"], "пять")                    # отрезок 3–6 с исключён: только сверка

    def test_evidence_of_absent_model_is_still_reported(self):
        _, got = self.run_case(None)
        self.assertEqual(got["evidence"]["whisper"], [(3.0, 6.0, "error", ["primary", "second"])])


class ExtraRoleNamesTests(unittest.TestCase):

    def combine(self, role):
        return words.combine_hypotheses(
            {"primary": ten(), "second": ten({1: "дважды"}), "whisper": text(ten({1: "двое"})),
             role: ten({1: "двойка"})}, ok_diag("primary", "second", "whisper", role), rule=RULE)

    def test_names_that_collide_with_word_fields_are_refused(self):
        for role in ("span", "ins", "place", "", None, 5):
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.combine(role)

    def test_vote_n_refuses_the_same_names(self):
        for role in ("span", "ins", "place", ""):
            with self.subTest(role=role), self.assertRaises(ValueError):
                words.vote_n(ten(), [("second", ten()), (role, ten())], {"keep": 2, "replace": 1})

    def test_refused_even_when_the_role_has_no_hypothesis(self):
        with self.assertRaises(ValueError):
            words.combine_hypotheses({"primary": ten(), "second": ten(), "whisper": text(ten()), "span": None},
                                     ok_diag("primary", "second", "whisper"))

    def test_ordinary_role_gives_its_own_field_and_span_length_stays(self):
        main = ten()
        words.combine_hypotheses(
            {"primary": main, "second": ten({1: "дважды"}), "whisper": text(ten({1: "двое"})),
             "fourth": ten({1: "двойка"})}, ok_diag("primary", "second", "whisper", "fourth"), rule=RULE)
        self.assertEqual((main[1]["alt_fourth"], main[1]["alt_span"]), ("двойка", 1))
        self.assertEqual(TEN[1], main[1]["w"])


if __name__ == "__main__":
    unittest.main()
