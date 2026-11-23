"""Протокол голосования (words.vote_n / combine_hypotheses с trace) и проверки записи гипотез по аудиту
PR №129/№131: время итогового слова у вставки, варианты у оставленных слов, причина отключения голосования,
имена ролей, проверка путей перед записью.

Только синтетика. Результат голосования при трёх моделях прежний — отпечатки в tests/test_combine_n_models.py
и tests/test_run_words_golden.py.
"""
import copy
import json
import unittest

import words
from tests.test_combine_n_models import TEN, ok_diag, ten, text, wd


def without(i, ws=None):
    ws = ten() if ws is None else ws
    return [w for k, w in enumerate(ws) if k != i]


class InsertedTimeTests(unittest.TestCase):
    """Вставленное слово: t — время итогового слова после _fit_times, t_model — время у модели."""

    def test_inserted_place_has_time_of_final_word(self):
        main = without(5)                                     # основная не слышит «шесть»
        second = [dict(w) for w in ten()]
        second[5]["s"], second[5]["e"] = 1.0, 1.3              # время у второй модели смещено
        trace = {}
        words.vote_n(main, [("second", second), ("whisper", text(ten()))], words.TWO_OF_THREE, trace)
        final = next(w for w in main if w["w"] == "шесть")
        self.assertEqual(final["s"], 4.0)                     # _fit_times: не раньше предыдущего слова
        (place,) = [p for p in trace["places"] if p["decision"] == "inserted"]
        self.assertEqual(place["t"], final["s"])
        self.assertEqual(place["t_model"], 1.0)

    def test_replaced_place_has_time_of_main_and_of_result(self):
        main = ten({4: "пятак"})
        second = ten()
        second[4]["s"] = 3.5
        trace = {}
        words.vote_n(main, [("second", second), ("whisper", text(ten()))], words.TWO_OF_THREE, trace)
        (place,) = [p for p in trace["places"] if p["decision"] == "replaced"]
        self.assertEqual(place["t"], 4.0)                     # начало слов основной
        self.assertEqual(place["t_result"], next(w["s"] for w in main if w["w"] == "пять"))

    def test_trace_does_not_change_result(self):
        for build in (lambda: without(5), lambda: ten({4: "пятак"})):
            plain, traced = build(), build()
            others = [("second", ten()), ("whisper", text(ten()))]
            s1 = words.vote_n(plain, copy.deepcopy(others), words.TWO_OF_THREE)
            s2 = words.vote_n(traced, copy.deepcopy(others), words.TWO_OF_THREE, {})
            self.assertEqual(json.dumps([plain, s1], ensure_ascii=False), json.dumps([traced, s2], ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
