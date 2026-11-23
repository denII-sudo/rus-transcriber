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


class KeptVariantsTests(unittest.TestCase):
    """У оставленного не всеми слова — варианты несогласных моделей."""

    def test_kept_place_has_variants_of_disagreeing_roles(self):
        main = ten()
        trace = {}
        words.vote_n(main, [("second", ten({2: "тройка"})), ("whisper", text(without(6)))], words.TWO_OF_THREE,
                     trace)
        kept = {p["main"]: p for p in trace["places"] if p["decision"] == "kept"}
        self.assertEqual(kept["три"]["support"], ["primary", "whisper"])
        self.assertEqual(kept["три"]["variants"], {"second": "тройка"})
        self.assertEqual(kept["семь"]["variants"], {"whisper": ""})       # третья модель слова не слышит


class DisabledVoteTraceTests(unittest.TestCase):
    """Голосование отключено: в протоколе причина, роли и исключённые отрезки, мест нет."""

    def combine(self, diag, whisper=True):
        trace = {}
        words.combine_hypotheses({"primary": ten(), "second": ten({4: "пятак"}),
                                  "whisper": text(ten()) if whisper else None}, diag, trace=trace)
        return trace

    def check(self, trace, reason, roles):
        self.assertEqual(trace["schema"], "vote-trace-1")
        self.assertEqual(trace["disabled"], {"reason": reason})
        self.assertEqual(trace["roles"], roles)
        self.assertEqual((trace["places"], trace["unanimous"]), ([], 0))
        self.assertIn("skip_spans", trace)
        self.assertIn("merged_skip_spans", trace)
        json.dumps(trace)

    def test_too_few_hypotheses(self):
        self.check(self.combine(ok_diag("primary", "second"), whisper=False), "too_few_hypotheses",
                   ["primary", "second"])

    def test_primary_failed(self):
        diag = ok_diag("primary", "second", "whisper")
        diag["primary"] = {"chunks": [{"state": "error", "own_span": [3.0, 6.0]},
                                      {"state": "ok", "own_span": [0.0, 3.0]}]}
        self.check(self.combine(diag), "primary_failed", ["primary", "second", "whisper"])

    def test_all_words_in_skip_spans(self):
        diag = ok_diag("primary", "second", "whisper")
        diag["whisper"] = {"chunks": [{"state": "error", "own_span": [0.0, 10.0]}]}
        trace = self.combine(diag)
        self.check(trace, "all_in_skip_spans", ["primary", "second", "whisper"])
        self.assertEqual(trace["merged_skip_spans"], [[0.0, 10.0]])
        self.assertEqual([s["role"] for s in trace["skip_spans"]], ["whisper"])

    def test_disabled_trace_does_not_change_result(self):
        diag = ok_diag("primary", "second", "whisper")
        diag["whisper"] = {"chunks": [{"state": "error", "own_span": [0.0, 10.0]}]}
        outs = []
        for trace in (None, {}):
            main = ten()
            got = words.combine_hypotheses({"primary": main, "second": ten({4: "пятак"}), "whisper": text(ten())},
                                           copy.deepcopy(diag), **({} if trace is None else {"trace": trace}))
            outs.append(json.dumps([main, got], ensure_ascii=False, default=str))
        self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main()
