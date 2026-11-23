"""Сведение гипотез любого числа моделей (РАР-4, этап 4.2, шаг 2): words.vote_n и
words.combine_hypotheses с дополнительными ролями и явным правилом rule.

Только синтетические гипотезы: модели не загружаются. При трёх моделях результат обязан быть
прежним байт в байт: отпечатки SHA-256 ниже сняты на коде до шага 2 (vote3 и combine_hypotheses
на сотнях случайных синтетических гипотез с фиксированным зерном, вместе с порядком ключей).
Правило для четырёх и более моделей в программу не встроено: без явного rule — отказ.

Отпечатки печатает
    COMBINE_N_PRINT=1 python -B -m unittest tests.test_combine_n_models
"""
import copy
import hashlib
import json
import os
import random
import unittest

import words

# Отпечатки поведения при трёх моделях, сняты на коде main 6f97124 (до шага 2).
VOTE3_FINGERPRINT = "6db2521f728e2580997d3055279c0f0c29e22905073789de5090d307885e23e3"
COMBINE3_FINGERPRINT = "e12c61ea517c31e3145268e10b5e71125373c34e8ce7a63d3e0045588cf7a463"

VOCAB = ["да", "нет", "дом", "кот", "мир", "ну", "вот", "сейчас", "щас", "прям", "прямо", "что", "чё",
         "какое-то", "какойто", "ещё", "еще", "—", "два", "дважды", "Привет,", "дела?"]


def gen_words(rng, n):
    out, t = [], 0.0
    for _ in range(n):
        t += rng.choice([0.2, 0.3, 0.5, 0.8])
        out.append({"w": rng.choice(VOCAB), "s": round(t, 2), "e": round(t + 0.25, 2),
                    "c": rng.choice([0.4, 0.9, 1.0])})
    return out


def mutate(rng, ws, shift=0.0):
    """Копия гипотезы с заменами, пропусками и вставками слов; времена сдвинуты на shift."""
    out = []
    for w in ws:
        r = rng.random()
        if r < 0.12:
            continue
        x = dict(w)
        if r < 0.3:
            x["w"] = rng.choice(VOCAB)
        x["s"], x["e"] = round(x["s"] + shift, 2), round(x["e"] + shift, 2)
        out.append(x)
        if rng.random() < 0.1:
            out.append({"w": rng.choice(VOCAB), "s": round(x["s"] + 0.1, 2), "e": round(x["s"] + 0.2, 2),
                        "c": 0.8})
    return out


def gen_case(rng):
    main = gen_words(rng, rng.randint(0, 25))
    second = mutate(rng, main, rng.choice([0.0, 0.0, 0.05, -0.3, 0.6]))
    base = rng.choice([main, second, second])
    third = [w["w"] for w in mutate(rng, base)]
    return main, second, third


def dump(obj):
    return json.dumps(obj, ensure_ascii=False)


def fingerprint(items):
    h = hashlib.sha256()
    for item in items:
        h.update(dump(item).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def vote3_outputs(vote):
    rng = random.Random(20261123)
    out = []
    for _ in range(400):
        main, second, third = gen_case(rng)
        stats = vote(main, second, third)
        out.append([main, stats])
    return out


def gen_diagnostics(rng, end):
    """Куски трёх проходов: от одного до трёх отрезков own_span, иногда сбойные."""
    diag = {}
    for stage in ("primary", "second", "whisper"):
        cuts = sorted(round(rng.uniform(0.0, end), 2) for _ in range(rng.randint(0, 2)))
        bounds = [0.0] + cuts + [round(end + 1.0, 2)]
        chunks = []
        for a, b in zip(bounds, bounds[1:]):
            r = rng.random()
            state = "ok" if r < 0.8 else ("error" if r < 0.9 else "empty_unknown")
            rec = {"state": state, "own_span": [a, b]}
            if stage == "whisper" and state == "ok" and rng.random() < 0.05:
                rec["filtered_all"] = True
            chunks.append(rec)
        diag[stage] = {"chunks": chunks}
    return diag


def combine_outputs(combine):
    rng = random.Random(20261124)
    out = []
    for _ in range(400):
        main, second, third = gen_case(rng)
        end = main[-1]["s"] if main else 1.0
        diag = gen_diagnostics(rng, end)
        if rng.random() < 0.1:
            second = None
        if rng.random() < 0.15:
            third = None
        fused = combine({"primary": main, "second": second, "whisper": third}, diag)
        out.append([main, fused])
    return out


class ThreeModelsUnchangedTests(unittest.TestCase):
    """При трёх моделях результат прежний: отпечатки сняты на коде до шага 2."""

    def check(self, items, expected, name):
        got = fingerprint(items)
        if os.environ.get("COMBINE_N_PRINT"):
            print(f"{name} = {got!r}")
        self.assertEqual(got, expected)

    def test_vote3_unchanged(self):
        self.check(vote3_outputs(words.vote3), VOTE3_FINGERPRINT, "VOTE3_FINGERPRINT")

    def test_vote_n_with_two_of_three_equals_vote3(self):
        def vote(main, second, third):
            return words.vote_n(main, [("second", second), ("whisper", third)], words.TWO_OF_THREE)
        self.check(vote3_outputs(vote), VOTE3_FINGERPRINT, "VOTE3_FINGERPRINT")

    def test_combine_hypotheses_unchanged(self):
        self.check(combine_outputs(words.combine_hypotheses), COMBINE3_FINGERPRINT, "COMBINE3_FINGERPRINT")

    def test_explicit_two_of_three_rule_equals_default(self):
        def combine(hyps, diag):
            return words.combine_hypotheses(hyps, diag, rule={"keep": 2, "replace": 2})
        self.check(combine_outputs(combine), COMBINE3_FINGERPRINT, "COMBINE3_FINGERPRINT")


# --- четыре модели: синтетическая запись 0–10 с, слово в секунду ---
TEN = ["один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять"]


def wd(text, s):
    return {"w": text, "s": float(s), "e": float(s) + 0.3, "c": 0.9}


def ten(replace=None):
    """Десять слов; replace — {номер: другое слово}."""
    return [wd((replace or {}).get(i, t), i) for i, t in enumerate(TEN)]


def text(ws):
    return [w["w"] for w in ws]


def ok_diag(*roles):
    return {r: {"chunks": [{"state": "ok", "own_span": [0.0, 10.0]}]} for r in roles}


ROLES4 = ("primary", "second", "whisper", "fourth")


class FourModelsTests(unittest.TestCase):

    def combine(self, second, whisper, fourth, rule, diag=None):
        self.main = ten()
        return words.combine_hypotheses({"primary": self.main, "second": second, "whisper": whisper,
                                         "fourth": fourth}, diag or ok_diag(*ROLES4), rule=rule)

    def test_fourth_model_keeps_primary_word_with_keep_two(self):
        got = self.combine(ten({1: "дважды"}), text(ten({1: "дважды"})), ten(), {"keep": 2, "replace": 2})
        self.assertEqual(text(self.main), TEN)   # «два» поддержала четвёртая модель — слово осталось
        self.assertFalse(any(w.get("voted") for w in self.main))
        self.assertEqual(list(got["quality"]["vote"].items()),
                         [("agree4", 9), ("agree3", 0), ("agree2", 1), ("replaced", 0), ("inserted", 0),
                          ("disputed", 0)])
        self.assertNotIn("asr_votes", got["quality"])

    def test_keep_three_replaces_word_supported_by_two_others(self):
        got = self.combine(ten({1: "дважды"}), text(ten({1: "дважды"})), ten(), {"keep": 3, "replace": 2})
        self.assertEqual(self.main[1], {"w": "дважды", "s": 1.0, "e": 1.3, "c": 1.0, "voted": "ctc+whisper",
                                        "model": "два"})
        self.assertEqual(list(got["quality"]["vote"].items()),
                         [("agree4", 9), ("agree3", 0), ("replaced", 1), ("inserted", 0), ("disputed", 0)])
        self.assertEqual(got["quality"]["asr_votes"], [{"t": 1.0, "t2": 1.0, "model": "два", "to": "дважды"}])

    def test_insertion_heard_by_second_and_fourth(self):
        second = ten()
        second[5:5] = [wd("кошка", 4.5)]
        fourth = ten()
        fourth[5:5] = [wd("кошка", 4.6)]
        got = self.combine(second, text(ten()), fourth, {"keep": 2, "replace": 2})
        self.assertEqual(text(self.main), TEN[:5] + ["кошка"] + TEN[5:])
        self.assertEqual(self.main[5], {"w": "кошка", "s": 4.5, "e": 4.8, "c": 1.0, "voted": "ctc+fourth"})
        self.assertEqual(got["quality"]["vote"]["inserted"], 1)

    def test_all_disagree_gives_variants_of_every_model(self):
        got = self.combine(ten({1: "дважды"}), text(ten({1: "двое"})), ten({1: "двойка"}),
                           {"keep": 2, "replace": 2})
        self.assertEqual(list(self.main[1].items()),
                         [("w", "два"), ("s", 1.0), ("e", 1.3), ("c", 0.9), ("agree", False), ("alt", "дважды"),
                          ("alt_whisper", "двое"), ("alt_fourth", "двойка"), ("alt_span", 1)])
        self.assertEqual(got["quality"]["vote"]["disputed"], 1)

    def test_text_only_models_cannot_replace_without_times(self):
        self.combine(ten({1: "двойка"}), text(ten({1: "дважды"})), text(ten({1: "дважды"})),
                     {"keep": 2, "replace": 2})
        self.assertEqual(self.main[1]["w"], "два")
        self.assertEqual((self.main[1]["alt"], self.main[1]["alt_whisper"], self.main[1]["alt_fourth"]),
                         ("двойка", "дважды", "дважды"))

    def test_four_models_without_rule_are_refused(self):
        with self.assertRaises(ValueError):
            self.combine(ten(), text(ten()), ten(), None)

    def test_bad_rules_are_refused(self):
        for rule in ({"keep": 5, "replace": 2}, {"keep": 2, "replace": 0}, {"keep": 2, "replace": 4},
                     {"keep": "2", "replace": 2}, {"keep": True, "replace": 2}, {"keep": 2},
                     {"keep": 2, "replace": 2, "extra": 1}, [2, 2]):
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                self.combine(ten(), text(ten()), ten(), rule)


class TwoAndThreeModelsTests(unittest.TestCase):

    def test_two_models_compare_with_mark_alt(self):
        second = ten({1: "дважды"})
        expected = ten()
        words.mark_alt(expected, second)
        for rule in (None, {"keep": 2, "replace": 1}):
            with self.subTest(rule=rule):
                main = ten()
                got = words.combine_hypotheses({"primary": main, "second": second}, ok_diag("primary", "second"),
                                               rule=rule)
                self.assertEqual(main, expected)
                self.assertEqual(got["quality"], {})

    def test_three_models_with_extra_role_use_two_of_three(self):
        second, third = ten({1: "дважды"}), text(ten({1: "дважды", 4: "пятак"}))
        expected = ten()
        stats = words.vote3(expected, second, third)
        main = ten()
        got = words.combine_hypotheses({"primary": main, "second": second, "whisper": None, "fourth": third},
                                       ok_diag("primary", "second", "fourth"))
        self.assertEqual(got["quality"]["vote"], stats)
        self.assertEqual(text(main), text(expected))

    def test_fourth_pass_missing_does_not_change_three_model_vote(self):
        second, third = ten({1: "дважды"}), text(ten({1: "дважды"}))
        expected = ten()
        stats = words.vote3(expected, second, third)
        main = ten()
        diag = {**ok_diag("primary", "second", "whisper"),
                "fourth": {"chunks": [{"state": "error", "own_span": [0.0, 10.0]}]}}
        got = words.combine_hypotheses({"primary": main, "second": second, "whisper": third, "fourth": None}, diag)
        self.assertEqual(main, expected)
        self.assertEqual(got["quality"]["vote"], stats)
        self.assertEqual(got["quality"]["asr_incomplete_chunks"], 1)
        self.assertEqual(got["evidence"]["other"], [(0.0, 10.0, "fourth", "error", ["primary", "second"])])


class FailedChunksWithFourModelsTests(unittest.TestCase):
    """Сбойный кусок любой модели исключает из голосования только свой отрезок own_span."""

    RULE = {"keep": 2, "replace": 2}

    def setUp(self):
        self.main = ten()
        self.second = ten({1: "дважды", 7: "восемью"})
        self.third = text(ten({1: "дважды", 7: "восемью"}))
        self.fourth = ten({4: "пятак"})

    def diag(self, role, span, state="error"):
        a, b = span
        recs = [{"state": state, "own_span": [a, b]}]
        recs += [{"state": "ok", "own_span": list(x)} for x in ((0.0, a), (b, 10.0)) if x[1] > x[0]]
        return {**ok_diag(*ROLES4), role: {"chunks": sorted(recs, key=lambda r: r["own_span"])}}

    def expected(self, spans):
        """Эталон: vote_n по словам вне отрезков, сверка со второй моделью внутри, сборка по времени."""
        def inside(w):
            return any(a <= w["s"] < b for a, b in spans)
        main = copy.deepcopy(self.main)
        keep = [w for w in main if not inside(w)]
        others = [("second", [w for w in self.second if not inside(w)]), ("whisper", list(self.third)),
                  ("fourth", [w for w in self.fourth if not inside(w)])]
        stats = words.vote_n(keep, copy.deepcopy(others), self.RULE)
        out = list(keep)
        for a, b in reversed(spans):
            mp = [w for w in main if a <= w["s"] < b]
            words.mark_alt(mp, [w for w in self.second if a <= w["s"] < b])
            pos = next((i for i, w in enumerate(out) if w["s"] >= a), len(out))
            out[pos:pos] = mp
        return out, stats

    def combine(self, diag):
        return words.combine_hypotheses({"primary": self.main, "second": self.second, "whisper": self.third,
                                         "fourth": self.fourth}, diag, rule=self.RULE)

    def test_fourth_model_failure_excludes_only_its_span(self):
        want, stats = self.expected([(3.0, 6.0)])
        got = self.combine(self.diag("fourth", (3.0, 6.0)))
        self.assertEqual(self.main, want)
        self.assertEqual(got["quality"]["vote"], stats)
        # вне отрезка голосуют четыре модели: «два» и «восемь» поддержала только четвёртая
        self.assertEqual(stats, {"agree4": 5, "agree3": 0, "agree2": 2, "replaced": 0, "inserted": 0,
                                 "disputed": 0})
        self.assertEqual(got["quality"]["vote3_partial"], {"reason": "asr_incomplete", "spans": 1, "seconds": 3.0})
        self.assertEqual(got["quality"]["asr_incomplete_chunks"], 1)
        self.assertEqual(got["evidence"]["other"], [(3.0, 6.0, "fourth", "error", ["primary", "second"])])

    def test_second_model_failure_with_four_models(self):
        want, stats = self.expected([(0.5, 2.0)])
        got = self.combine(self.diag("second", (0.5, 2.0), state="empty_unknown"))
        self.assertEqual(self.main, want)
        self.assertEqual(self.main[1]["w"], "два")                   # внутри отрезка — только сверка
        self.assertEqual(got["quality"]["vote3_partial"], {"reason": "asr_incomplete", "spans": 1, "seconds": 1.5})

    def test_third_model_failure_with_four_models(self):
        want, stats = self.expected([(7.0, 8.0)])
        got = self.combine(self.diag("whisper", (7.0, 8.0)))
        self.assertEqual(self.main, want)
        self.assertEqual(got["quality"]["vote3_partial"], {"reason": "whisper_incomplete", "spans": 1, "seconds": 1.0})
        self.assertEqual(got["evidence"]["whisper"], [(7.0, 8.0, "error", ["primary", "second"])])

    def test_primary_failure_disables_vote(self):
        expected = copy.deepcopy(self.main)
        words.mark_alt(expected, self.second)
        got = self.combine(self.diag("primary", (3.0, 6.0)))
        self.assertEqual(self.main, expected)
        self.assertNotIn("vote", got["quality"])
        self.assertEqual(got["quality"]["vote3_skipped"], "asr_incomplete")


if __name__ == "__main__":
    unittest.main()
