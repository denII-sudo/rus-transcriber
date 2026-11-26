"""РАР-5.1, П51-1: порядок мест на разборе — `priority.assess` и `priority.plan`.

Чистые функции: модуль не импортирует приложение (paths, core, server), поэтому папка данных не нужна.
Все места синтетические.
"""
import ast
import copy
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import buildinfo  # noqa: E402  (только стандартная библиотека)
import priority as P  # noqa: E402


def root_place(n, t, important=False, **extra):
    """Спор моделей с другим корнем; important — вариант с числом."""
    return {"id": f"r{n}", "t": t, "t1": t, "t2": t, "w1": 1, "reasons": ["asr"],
            "text": "сказал", "alt_ctc": "пять" if important else "показал", **extra}


def voice_place(n, t, text="я думаю что это так"):
    return {"id": f"r{n}", "t": t, "t1": t, "t2": t + 1, "w1": 0, "reasons": ["voice"], "text": text}


def conf_place(n, t, text="слово"):
    return {"id": f"r{n}", "t": t, "t1": t, "t2": t, "w1": 1, "reasons": ["conf"], "text": text}


def gap_place(n, t, seconds):
    return {"id": f"r{n}", "t": t, "t1": t, "t2": t + seconds, "gap": [t, t + seconds], "reasons": ["missed"],
            "text": "", "note": "речь без распознанных слов"}


def overlap_place(n, t, text="ну ладно"):
    return {"id": f"r{n}", "t": t, "t1": t, "t2": t, "w1": 1, "reasons": ["overlap", "voice"], "text": text}


def mixed_places():
    items = [gap_place(1, 10.0, 3.0), gap_place(2, 20.0, 1.2), root_place(3, 30.0, important=True),
             root_place(4, 40.0), voice_place(5, 50.0), conf_place(6, 60.0), overlap_place(7, 70.0),
             overlap_place(8, 80.0, text="было 15"), conf_place(9, 90.0, text="угу"),
             {"id": "r10", "t": 100.0, "t1": 100.0, "t2": 103.0, "gap": [100.0, 103.0], "reasons": ["asr", "missed"],
              "text": "", "asr_problem": {"model": "whisper", "state": "error", "words_from": []}},
             {"id": "r11", "t": 110.0, "seg": 3, "w1": 0, "w2": 4, "reasons": ["gender"], "text": "я пришла"}]
    return items


def total_count(result):
    return result["shown"] + sum(result["remaining_by_level"].values()) + result["overlap_queue"]


class ModuleShape(unittest.TestCase):
    def test_pure_module_imports_only_standard_library(self):
        tree = ast.parse((PROJECT / "priority.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertLessEqual(names, {"hashlib", "math", "re"})

    def test_installer_ships_new_root_module(self):
        # install.ps1 копирует файлы корня по списку buildinfo.py --list-root (маска *.py): строки не нужно
        self.assertIn("priority.py", buildinfo.root_files(PROJECT))

    def test_file_is_utf8_without_bom_and_lf(self):
        raw = (PROJECT / "priority.py").read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r\n", raw)
        raw.decode("utf-8")


class Assess(unittest.TestCase):
    def test_repeat_gives_same_answer_and_input_unchanged(self):
        items = mixed_places()
        before = copy.deepcopy(items)
        first = [P.assess(it, {"terms": ["Петров"]}) for it in items]
        second = [P.assess(it, {"terms": ["Петров"]}) for it in items]
        self.assertEqual(first, second)
        self.assertEqual(items, before)

    def test_place_without_new_fields(self):
        a = P.assess({"id": "r1"})
        self.assertEqual((a["level"], a["queue"], a["kind"]), ("optional", "main", "other"))
        a = P.assess({"id": "r1", "reasons": None, "text": None, "alt_ctc": 5, "gap": "x", "w1": None})
        self.assertEqual(a["level"], "optional")
        with self.assertRaises(TypeError):
            P.assess(["r1"])

    def test_model_disagreement_is_not_technical_failure(self):
        base = {"id": "r1", "t": 5.0, "t1": 5.0, "t2": 5.0, "gap": [5.0, 5.04], "w1": 2, "reasons": ["asr"],
                "text": "сказал", "asr_problem": {"model": "whisper", "state": "disagreement",
                                                   "words_from": ["primary", "second"]}}
        plain = P.assess({**base, "alt_whisper": "показал"})
        self.assertFalse(plain["signs"]["tech"])
        self.assertEqual((plain["level"], plain["kind"]), ("wanted", "root"))
        same = P.assess({**base, "alt_whisper": "сказала"})
        self.assertEqual((same["level"], same["kind"]), ("wanted", "dispute"))
        number = P.assess({**base, "alt_whisper": "семь"})
        self.assertEqual(number["level"], "required")
        for state in ("error", "empty_unknown", "filtered", "whisper_incomplete"):
            failure = P.assess({**base, "reasons": ["asr", "missed"], "text": "",
                                "asr_problem": {"model": "whisper", "state": state}})
            self.assertTrue(failure["signs"]["tech"], state)
            self.assertEqual(failure["level"], "required", state)

    def test_gap_from_two_seconds_required_shorter_optional(self):
        self.assertEqual(P.assess(gap_place(1, 0.0, 2.0))["level"], "required")
        self.assertEqual(P.assess(gap_place(1, 0.0, 7.5))["level"], "required")
        short = P.assess(gap_place(1, 0.0, 1.99))
        self.assertEqual((short["level"], short["kind"]), ("optional", "short_gap"))
        no_times = P.assess({"id": "r1", "reasons": ["missed"], "text": ""})
        self.assertEqual(no_times["level"], "optional")
        self.assertIsNone(no_times["signs"]["gap_seconds"])

    def test_root_rule_sixty_percent(self):
        self.assertTrue(P.same_root("сказал", "сказала"))
        self.assertTrue(P.same_root("Ёлка", "елки"))
        self.assertFalse(P.same_root("пять", "шесть"))
        self.assertFalse(P.same_root("работа", "робот"))
        # общее начало 3 из 5 = 60 %: корень один; 2 из 5 — другой
        self.assertTrue(P.same_root("домик", "домой"))
        self.assertFalse(P.same_root("дорог", "донос"))

    def test_text_word_without_variant_is_not_root_difference(self):
        # вариант CTC содержит только слова со своим вариантом: «было» без пары не считается расхождением
        a = P.assess({"id": "r1", "w1": 0, "reasons": ["asr"], "text": "было сказано", "alt_ctc": "сказана"})
        self.assertFalse(a["signs"]["root"])

    def test_important_word_kinds(self):
        def kinds(text, alt, terms=(), w1=3):
            a = P.assess({"id": "r1", "w1": w1, "reasons": ["asr"], "text": text, "alt_ctc": alt},
                         {"terms": list(terms)})
            return a["level"], a["signs"]["important_root"]

        self.assertEqual(kinds("в среду", "в 15"), ("required", ["number"]))
        self.assertEqual(kinds("в среду", "в тридцать"), ("required", ["number"]))
        self.assertEqual(kinds("он знал", "он не"), ("required", ["negation"]))
        self.assertEqual(kinds("отдал Петрову", "отдал петухову", terms=["Петров"]), ("required", ["term"]))
        self.assertEqual(kinds("приехал Олег", "приехал облик"), ("required", ["capital"]))
        # заглавная в начале реплики — не признак
        self.assertEqual(kinds("Олег", "облик", w1=0), ("wanted", []))
        # слова, похожие на числительные, числом не считаются
        self.assertEqual(kinds("на стол", "на стул"), ("wanted", []))

    def test_filler_and_conf_only_are_optional(self):
        self.assertEqual(P.assess(conf_place(1, 0.0, text="угу"))["importance"], 0)
        self.assertEqual(P.assess(conf_place(1, 0.0))["level"], "optional")
        filler_voice = P.assess(voice_place(1, 0.0, text="ага"))
        self.assertEqual(filler_voice["level"], "optional")
        self.assertEqual(P.assess(voice_place(1, 0.0))["pool"], "speaker")
        # «нет» — не заполнитель
        self.assertEqual(P.assess(voice_place(1, 0.0, text="нет"))["level"], "wanted")

    def test_overlap_with_important_word_required_else_separate_queue(self):
        plain = P.assess(overlap_place(1, 0.0))
        self.assertEqual((plain["queue"], plain["level"]), ("overlap", "optional"))
        important = P.assess(overlap_place(1, 0.0, text="было 15"))
        self.assertEqual((important["queue"], important["level"]), ("main", "required"))


class Plan(unittest.TestCase):
    def test_shares_and_free_time_passes_down(self):
        items = ([root_place(n, float(n)) for n in range(1, 101)]
                 + [voice_place(n, float(n)) for n in range(101, 201)]
                 + [conf_place(n, float(n)) for n in range(201, 301)])
        result = P.plan(items, 20, seed="rec-1")
        by_block = {}
        for row in result["selected"]:
            by_block[row["block"]] = by_block.get(row["block"], 0) + row["seconds"]
        self.assertEqual(by_block["wanted"], 780)          # 13 мин: 39 мест по 20 с
        self.assertEqual(by_block["speaker"], 289)         # 5 мин: 17 мест по 17 с, 11 с переходит дальше
        self.assertLessEqual(by_block["random"], 120 + 11)
        self.assertLessEqual(result["estimate_seconds"], 1200)
        self.assertGreater(result["estimate_seconds"], 1200 - 20)
        self.assertEqual(total_count(result), len(items))
        # без мест «кто говорит» их 5 минут достаются следующим
        only_words = P.plan([root_place(n, float(n)) for n in range(1, 101)], 20, seed="rec-1")
        self.assertEqual(only_words["shown"], 60)
        self.assertEqual(sum(r["block"] == "random" for r in only_words["selected"]), 21)

    def test_required_first_in_words_share(self):
        items = [root_place(n, float(n)) for n in range(1, 41)] + [root_place(n, float(n), important=True)
                                                                    for n in range(41, 46)]
        result = P.plan(items, 5, seed="s")
        required = [r["id"] for r in result["selected"] if r["block"] == "required"]
        self.assertEqual(required, [f"r{n}" for n in range(41, 46)])
        # 2 минуты: доля слов 78 с вмещает 3 места «обязательно»; остальные два — в остатке, видны числом
        small = P.plan(items, 2, seed="s")
        self.assertEqual([r["id"] for r in small["selected"] if r["block"] == "required"], ["r41", "r42", "r43"])
        self.assertEqual(small["remaining_by_level"]["required"], 2)

    def test_all_long_gaps_shown_with_overflow(self):
        items = [gap_place(n, float(n * 10), 3.0) for n in range(1, 11)] + [root_place(11, 500.0, important=True)]
        result = P.plan(items, 1, seed="s")
        gaps = [r for r in result["selected"] if r["id"] != "r11"]
        self.assertEqual(len(gaps), 10)
        self.assertNotIn("r11", [r["id"] for r in result["selected"]])
        self.assertEqual(result["overflow_seconds"], 10 * 23 - 60)
        self.assertTrue(result["gaps_over_budget"])
        self.assertEqual(result["remaining_by_level"]["required"], 1)
        fits = P.plan(items[:2], 20, seed="s")
        self.assertEqual((fits["overflow_seconds"], fits["gaps_over_budget"]), (0.0, False))

    def test_overlap_queue_outside_minutes(self):
        items = [overlap_place(n, float(n)) for n in range(1, 6)] + [overlap_place(6, 6.0, text="было 15")]
        result = P.plan(items, 60, seed="s")
        self.assertEqual([r["id"] for r in result["selected"]], ["r6"])
        self.assertEqual(result["overlap_queue"], 5)
        self.assertEqual(total_count(result), 6)

    def test_random_share_constant_for_same_recording(self):
        items = ([root_place(n, float(n)) for n in range(1, 81)]
                 + [conf_place(n, float(n)) for n in range(81, 181)])
        first = P.plan(items, 20, seed="rec-7")
        self.assertEqual(first, P.plan(list(reversed(items)), 20, seed="rec-7"))
        picks = {s: [r["id"] for r in P.plan(items, 20, seed=s)["selected"] if r["pick"] == "random"]
                 for s in ("rec-1", "rec-2", "rec-3")}
        self.assertTrue(all(picks.values()))
        self.assertGreater(len({tuple(v) for v in picks.values()}), 1)

    def test_short_recording_floor_and_unknown_duration(self):
        self.assertEqual(P.budget_minutes(180), P.MIN_MINUTES)
        self.assertEqual(P.MIN_MINUTES, 5)
        self.assertEqual(P.budget_minutes(3600), 20)
        self.assertEqual(P.budget_minutes(5400), 30)
        for unknown in (None, 0, -5, float("nan"), float("inf"), "600", True):
            self.assertIsNone(P.budget_minutes(unknown), unknown)
        for bad in (None, 0, -1, float("nan"), "20"):
            with self.assertRaises(ValueError):
                P.plan(mixed_places(), bad)

    def test_shown_plus_remaining_equals_places(self):
        items = mixed_places()
        for minutes in (0.1, 1, 3, 20, 300):
            for first in (True, False):
                result = P.plan(items, minutes, seed="s", first_portion=first)
                self.assertEqual(total_count(result), len(items), (minutes, first))
                self.assertEqual(len({r["id"] for r in result["selected"]}), result["shown"])
        everything = P.plan(items, 300, seed="s")
        self.assertEqual(everything["remaining_by_level"], {"required": 0, "wanted": 0, "optional": 0})
        self.assertEqual(everything["overlap_queue"], 1)

    def test_next_portion_same_order_without_shares(self):
        items = ([voice_place(n, float(n)) for n in range(1, 21)]
                 + [root_place(n, float(n)) for n in range(21, 41)]
                 + [root_place(n, float(n), important=True) for n in range(41, 44)]
                 + [conf_place(n, float(n)) for n in range(44, 64)])
        result = P.plan(items, 3, seed="s", first_portion=False)
        self.assertFalse(any(r["pick"] == "random" for r in result["selected"]))
        levels = [r["level"] for r in result["selected"]]
        self.assertEqual(levels[:3], ["required"] * 3)
        self.assertNotIn("optional", levels)
        self.assertLessEqual(result["estimate_seconds"], 180)

    def test_page_order_blocks_then_time(self):
        result = P.plan(mixed_places(), 300, seed="s")
        blocks = [r["block"] for r in result["selected"]]
        self.assertEqual(blocks, sorted(blocks, key=P.BLOCKS.index))
        for block in set(blocks):
            ids = [r["id"] for r in result["selected"] if r["block"] == block]
            times = [next(it["t"] for it in mixed_places() if it["id"] == i) for i in ids]
            self.assertEqual(times, sorted(times), block)

    def test_bad_ids_rejected_and_input_unchanged(self):
        items = mixed_places()
        before = copy.deepcopy(items)
        P.plan(items, 5, seed="s")
        self.assertEqual(items, before)
        with self.assertRaises(ValueError):
            P.plan([root_place(1, 1.0), root_place(1, 2.0)], 5)
        with self.assertRaises(ValueError):
            P.plan([{"t": 1.0}], 5)
        self.assertEqual(P.plan([], 5)["shown"], 0)

    def test_places_without_times_go_last_in_block(self):
        items = [root_place(1, 5.0), {**root_place(2, 0.0), "t": None, "t1": None}]
        result = P.plan(items, 20, seed="s")
        self.assertEqual([r["id"] for r in result["selected"]], ["r1", "r2"])


if __name__ == "__main__":
    unittest.main()
