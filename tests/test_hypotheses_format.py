"""Формат гипотез hyp-2 (РАР-4.2, шаг 3, PR Б1): протокол голосования и сборка записи.

Только синтетика. Протокол голосования (trace) ничего не меняет в результате: с ним и без него слова и
статистика те же байт в байт — проверяется отпечатками из tests/test_combine_n_models.py.
"""
import copy
import json
import unittest

import hypotheses
import words
from tests.test_combine_n_models import (COMBINE3_FINGERPRINT, VOTE3_FINGERPRINT, combine_outputs, fingerprint,
                                         ok_diag, ten, text, vote3_outputs)

RULE = {"keep": 2, "replace": 2}


class TraceDoesNotChangeResultTests(unittest.TestCase):

    def test_vote_n_with_trace_equals_vote3_fingerprint(self):
        def vote(main, second, third):
            return words.vote_n(main, [("second", second), ("whisper", third)], words.TWO_OF_THREE, trace={})
        self.assertEqual(fingerprint(vote3_outputs(vote)), VOTE3_FINGERPRINT)

    def test_combine_with_trace_equals_combine3_fingerprint(self):
        def combine(hyps, diag):
            return words.combine_hypotheses(hyps, diag, trace={})
        self.assertEqual(fingerprint(combine_outputs(combine)), COMBINE3_FINGERPRINT)

    def test_combine_without_trace_unchanged(self):
        self.assertEqual(fingerprint(combine_outputs(words.combine_hypotheses)), COMBINE3_FINGERPRINT)

    def test_trace_with_absent_third_and_absent_second(self):
        for hyps in ({"primary": ten(), "second": ten({1: "дважды"}), "whisper": None},
                     {"primary": ten(), "second": None, "whisper": text(ten({1: "двое"}))}):
            with self.subTest(absent=[k for k, v in hyps.items() if v is None]):
                plain, traced = copy.deepcopy(hyps), copy.deepcopy(hyps)
                tr = {}
                a = words.combine_hypotheses(plain, ok_diag("primary", "second", "whisper"))
                b = words.combine_hypotheses(traced, ok_diag("primary", "second", "whisper"), trace=tr)
                self.assertEqual(json.dumps([plain, a]), json.dumps([traced, b]))
                # меньше трёх гипотез — голосования нет, в протоколе причина и пустой список мест
                self.assertEqual((tr["disabled"], tr["places"]), ({"reason": "too_few_hypotheses"}, []))


class TraceContentTests(unittest.TestCase):

    def run_trace(self, second, third, fourth=None, rule=RULE, diag=None):
        main = ten()
        hyps = {"primary": main, "second": second, "whisper": third}
        roles = ["primary", "second", "whisper"]
        if fourth is not None:
            hyps["fourth"] = fourth
            roles.append("fourth")
        tr = {}
        words.combine_hypotheses(hyps, diag or ok_diag(*roles), rule=rule, trace=tr)
        return main, tr

    def test_three_models_replaced_place(self):
        main, tr = self.run_trace(ten({1: "дважды"}), text(ten({1: "дважды"})))
        self.assertEqual((tr["schema"], tr["rule"], tr["roles"]),
                         ("vote-trace-1", RULE, ["primary", "second", "whisper"]))
        self.assertEqual(tr["unanimous"], 9)
        self.assertEqual(tr["places"], [{"decision": "replaced", "t": 1.0, "main": "два",
                                         "variants": {"second": "дважды", "whisper": "дважды"},
                                         "winner_roles": ["second", "whisper"], "t_result": 1.0}])
        self.assertEqual(main[1]["voted"], "ctc+whisper")

    def test_kept_with_partial_support_and_disputed_place(self):
        main, tr = self.run_trace(ten({1: "дважды", 3: "четвёрка"}), text(ten({3: "четверо"})),
                                  ten({3: "четвёрка"}), rule={"keep": 2, "replace": 3})
        decisions = [(p["decision"], p["t"]) for p in tr["places"]]
        self.assertEqual(decisions, [("kept", 1.0), ("disputed", 3.0)])
        self.assertEqual(tr["places"][0]["support"], ["primary", "whisper", "fourth"])
        self.assertEqual(tr["places"][1]["variants"],
                         {"second": "четвёрка", "whisper": "четверо", "fourth": "четвёрка"})
        self.assertEqual(tr["places"][1]["winner_roles"], [])

    def test_inserted_place(self):
        second, fourth = ten(), ten()
        second[5:5] = [{"w": "кошка", "s": 4.5, "e": 4.8, "c": 1.0}]
        fourth[5:5] = [{"w": "кошка", "s": 4.6, "e": 4.9, "c": 1.0}]
        _, tr = self.run_trace(second, text(ten()), fourth)
        ins = [p for p in tr["places"] if p["decision"] == "inserted"]
        self.assertEqual(len(ins), 1)
        self.assertEqual((ins[0]["t"], ins[0]["winner_roles"]), (4.5, ["second", "fourth"]))
        self.assertEqual(ins[0]["variants"], {"second": "кошка", "fourth": "кошка"})

    def test_two_models_have_no_vote_and_trace_says_why(self):
        main = ten()
        tr = {}
        words.combine_hypotheses({"primary": main, "second": ten(), "whisper": None},
                                 ok_diag("primary", "second"), trace=tr)
        self.assertEqual(tr["disabled"], {"reason": "too_few_hypotheses"})
        self.assertEqual(tr["places"], [])

    def test_skip_spans_are_recorded_with_role_and_reason(self):
        diag = ok_diag("primary", "second", "whisper", "fourth")
        diag["fourth"] = {"chunks": [{"state": "error", "own_span": [3.0, 6.0]},
                                     {"state": "ok", "own_span": [0.0, 3.0]},
                                     {"state": "ok", "own_span": [6.0, 10.0]}]}
        _, tr = self.run_trace(ten(), text(ten()), ten(), diag=diag)
        self.assertEqual(tr["skip_spans"], [{"start": 3.0, "end": 6.0, "role": "fourth", "reason": "error"}])
        self.assertEqual(tr["merged_skip_spans"], [[3.0, 6.0]])

    def test_vote_n_trace_is_json_serializable(self):
        _, tr = self.run_trace(ten({1: "дважды"}), text(ten({1: "двое"})), ten({1: "двойка"}))
        json.dumps(tr, ensure_ascii=False)
        self.assertEqual(tr["places"][0]["decision"], "disputed")


class RecordTests(unittest.TestCase):

    def roles(self):
        return [{"role": "primary", "model": {"name": "v3_e2e_rnnt"}, "words": ten(),
                 "word_times": hypotheses.available()},
                {"role": "third", "model": {"name": "whisper"}, "words": hypotheses.not_available("только текст"),
                 "word_times": hypotheses.not_available("Whisper — только текст")}]

    def test_record_has_uniform_blocks_and_explicit_gaps(self):
        rec = hypotheses.build_record(self.roles(), legacy={"combine_rule": "vote3_partial", "primary": {"x": 1}})
        self.assertEqual(rec["schema"], "hyp-2")
        self.assertEqual(hypotheses.check(rec), [])
        for block in rec["roles"]:
            self.assertEqual(tuple(block), hypotheses.ROLE_FIELDS)
        third = rec["roles"][1]
        self.assertEqual(third["engine"], {"status": "not_measured", "reason": "движок не передан"})
        self.assertTrue(hypotheses.is_missing(third["word_times"]))
        for name in hypotheses.META_FIELDS:
            self.assertTrue(hypotheses.is_missing(rec[name]), name)
        self.assertEqual((rec["combine_rule"], rec["primary"]), ("vote3_partial", {"x": 1}))   # ключи lab-hyp-1
        json.dumps(rec, ensure_ascii=False)

    def test_record_refuses_repeated_roles_and_unknown_meta(self):
        r = self.roles()
        with self.assertRaises(ValueError):
            hypotheses.build_record([r[0], r[0]])
        with self.assertRaises(ValueError):
            hypotheses.build_record(r, meta={"disk_path": "C:\\x"})

    def test_meta_blocks_are_stored_as_given(self):
        rec = hypotheses.build_record(self.roles(), meta={"program": {"version": "1"}}, trace={"schema": "t"})
        self.assertEqual(rec["program"], {"version": "1"})
        self.assertEqual(rec["vote_trace"], {"schema": "t"})

    def test_read_accepts_both_versions_and_refuses_foreign(self):
        old = {"schema": "lab-hyp-1", "primary": {"words": []}}
        new = hypotheses.build_record(self.roles())
        self.assertIs(hypotheses.read(old), old)
        self.assertIs(hypotheses.read(new), new)
        self.assertEqual(hypotheses.read({"primary": 1}), {"primary": 1})      # базы до отметки формата
        for bad in ({"schema": "other-9"}, [], None, "x"):
            self.assertEqual(hypotheses.read(bad), {})

    def test_check_finds_broken_records(self):
        rec = hypotheses.build_record(self.roles())
        rec["roles"][0].pop("words")
        del rec["hardware"]
        self.assertEqual(len(hypotheses.check(rec)), 2)
        self.assertEqual(hypotheses.check({"schema": "lab-hyp-1"}), ["schema не hyp-2"])


if __name__ == "__main__":
    unittest.main()
