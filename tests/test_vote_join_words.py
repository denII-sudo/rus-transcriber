"""Голосование vote_n: варианты разных моделей согласны, только если совпадают слова, а не буквы
после склейки без пробелов («пя ть» и «п ять» — разные варианты).

Только синтетика. Сверка со старым сравнением на случайных входах — в SweepTests.
"""
import copy
import json
import random
import unittest

import words
from tests.test_combine_n_models import TEN, text, ten, wd

TWO = words.TWO_OF_THREE


def with_split(parts, at=4):
    """ten() с заменой слова номер at на несколько слов parts (времена в пределах секунды)."""
    ws = ten()
    new = [wd(p, at + k * 0.3) for k, p in enumerate(parts)]
    return ws[:at] + new + ws[at + 1:]


def vote(main, second, third, trace=None):
    return words.vote_n(main, [("second", second), ("whisper", third)], TWO, trace)


class JoinedVariantsTests(unittest.TestCase):

    def test_same_letters_other_split_is_disputed(self):
        main = ten({4: "пятак"})
        trace = {}
        stats = vote(main, with_split(["пя", "ть"]), ["один", "два", "три", "четыре", "п", "ять", "шесть",
                                                       "семь", "восемь", "девять", "десять"], trace)
        self.assertEqual((stats["replaced"], stats["disputed"]), (0, 1))
        (place,) = [p for p in trace["places"] if p["decision"] == "disputed"]
        self.assertEqual(place["variants"], {"second": "пя ть", "whisper": "п ять"})
        self.assertEqual(place["winner_roles"], [])
        self.assertEqual(next(w for w in main if w.get("alt_span"))["alt"], "пя ть")

    def test_same_split_is_replaced_as_before(self):
        main = ten({4: "пятак"})
        third = ["один", "два", "три", "четыре", "пя", "ть", "шесть", "семь", "восемь", "девять", "десять"]
        stats = vote(main, with_split(["пя", "ть"]), third)
        self.assertEqual((stats["replaced"], stats["disputed"]), (1, 0))
        self.assertEqual([w["w"] for w in main][4:6], ["пя", "ть"])

    def test_inserted_variants_with_other_split_do_not_win(self):
        main = [w for k, w in enumerate(ten()) if k != 5]
        second = with_split(["шес", "ть"], at=5)
        third = text(ten())
        third[5:6] = ["ше", "сть"]
        stats = vote(main, second, third)
        self.assertEqual(stats["inserted"], 0)
        self.assertNotIn("шес", [w["w"] for w in main])

    def test_inserted_same_split_wins(self):
        main = [w for k, w in enumerate(ten()) if k != 5]
        second = with_split(["шес", "ть"], at=5)
        third = text(ten())
        third[5:6] = ["шес", "ть"]
        stats = vote(main, second, third)
        self.assertEqual(stats["inserted"], 2)


def _old_group_vote(main, others):
    """Прежнее правило группировки (склейка без пробелов) — только для сверки."""
    orig = words._winner

    def old(cands, need, need_text):
        return orig([(h, js, txt.replace(" ", "")) for h, js, txt in cands], need, need_text)
    words._winner = old
    try:
        return words.vote_n(main, others, TWO)
    finally:
        words._winner = orig


def random_models(rnd, n_models):
    """Основная и ещё n_models-1 моделей: каждая — общий текст с случайными заменами, склейками и разрезами слов
    (одни и те же буквы по-разному разбиты на слова), у части моделей кусок пустой (сбой)."""
    base = [rnd.choice(["ab", "c", "a", "bc", "d", "e", "f", "abc"]) for _ in range(rnd.randint(3, 8))]

    def variant():
        out = []
        for t in base:
            r = rnd.random()
            if r < 0.15:
                out.append(rnd.choice(["x", "y", "d"]))
            elif r < 0.30 and len(t) > 1:
                k = rnd.randint(1, len(t) - 1)
                out += [t[:k], t[k:]]
            elif r < 0.40 and out:
                out[-1] += t
            elif r < 0.45:
                continue
            else:
                out.append(t)
        return out or ["x"]
    main = [wd(t, k) for k, t in enumerate(variant())]
    others = []
    for i in range(n_models - 1):
        toks = [] if rnd.random() < 0.05 else variant()
        role = ("second", "whisper", "fourth")[i]
        others.append((role, [wd(t, k) for k, t in enumerate(toks)] if role != "whisper" else toks))
    return main, others


class SweepTests(unittest.TestCase):
    """Старое и новое сравнение на 3000 случайных входах для 3 и 4 моделей: исход различается только там,
    где варианты разных моделей совпали бы после склейки без пробелов, но различаются по словам."""

    def run_sweep(self, n_models, rule, n=3000):
        rnd = random.Random(20261124 + n_models)
        seen = []
        orig = words._winner

        def spy(cands, need, need_text):
            by_join = {}
            for h, js, txt in cands:
                by_join.setdefault(txt.replace(" ", ""), set()).add(txt)
            seen.append(any(len(v) > 1 for v in by_join.values()))
            return orig(cands, need, need_text)

        def old(cands, need, need_text):
            return orig([(h, js, txt.replace(" ", "")) for h, js, txt in cands], need, need_text)
        changed = explained = 0
        for _ in range(n):
            main, others = random_models(rnd, n_models)
            words._winner, seen[:] = spy, []
            m_new = copy.deepcopy(main)
            s_new = words.vote_n(m_new, copy.deepcopy(others), rule)
            hit = any(seen)
            words._winner = old
            m_old = copy.deepcopy(main)
            try:
                s_old = words.vote_n(m_old, copy.deepcopy(others), rule)
            finally:
                words._winner = orig
            if json.dumps([m_old, s_old], ensure_ascii=False) != json.dumps([m_new, s_new], ensure_ascii=False):
                changed += 1
                explained += hit
        return changed, explained

    def check(self, n_models, rule):
        changed, explained = self.run_sweep(n_models, rule)
        self.assertEqual(changed, explained)          # каждое изменение — от расстановки пробелов
        return changed

    def test_three_models(self):
        self.assertGreater(self.check(3, words.TWO_OF_THREE), 0)

    def test_four_models(self):
        for rule in ({"keep": 2, "replace": 2}, {"keep": 3, "replace": 3}, {"keep": 2, "replace": 3}):
            self.check(4, rule)


if __name__ == "__main__":
    unittest.main()
