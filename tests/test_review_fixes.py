"""Правки разбора по сессии разбора 09.11.2026, на выдуманных словах и именах (Анна, Борис):
говорящий вставки без «Имя:», правка человека целиком (наложение, автоудаление, повтор), опечатки только
для обычных слов, написание чисел и версий, подсказки словаря, правка отрезка, повторное открытие места,
фильтры списка мест, общий словарь без group."""
import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="rt-data-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_OLD_ENV = os.environ.get("RUS_TRANSCRIBER_DATA")
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import freeform as F  # noqa: E402
import transcript as T  # noqa: E402

NAMES = ["Анна", "Борис"]


def W(text, t0, step=0.3, dur=0.25):
    return [{"w": w, "s": round(t0 + i * step, 2), "e": round(t0 + i * step + dur, 2), "c": 0.9}
            for i, w in enumerate(text.split())]


def seg(spk, words):
    return {"speaker": spk, "label": spk, "start": words[0]["s"], "end": words[-1]["e"],
            "text": " ".join(w["w"] for w in words), "words": words}


def item(rid, t1, t2, spk, text="", reasons=("asr",)):
    return {"id": rid, "t": t1, "t1": t1, "t2": t2, "speaker": spk, "text": text, "reasons": list(reasons),
            "status": "open"}


class Case:
    """Стенограмма во временной папке + применение свободных ответов как в apply_free."""

    def __init__(self, segments, review):
        self.d = Path(tempfile.mkdtemp(dir=_TMP))
        self.json = self.d / "x.transcript.json"
        self.md = self.d / "x.transcript.md"
        data = {"file": "C:/rec/x.mp3", "duration": "00:01:00", "speakers": NAMES, "segments": segments,
                "review": review, "edits": []}
        T.save(data, self.json, self.md)

    def answer(self, answers, vocab_words=()):
        data = T.load(self.json)
        man = {"items": [{k: v for k, v in r.items()} for r in data["review"] if r["id"] in answers]}
        free = {k: {"text": v, "ts": "2026-11-09T10:00:00"} for k, v in answers.items()}
        edits, reports, closed, _ = F.plan(man, free, data, vocab_words)
        out, rep = T.apply_word_edits(self.json, self.md, edits, by="human", closed=closed,
                                      answers={k: v for k, v in answers.items()})
        human = [r for r in reports if r.get("human")]
        if human:
            T.apply_review(self.json, self.md, [{"id": r["id"], "status": "human", "note": r["human"]}
                                                for r in human], by="human")
            out = T.load(self.json)
        return out, rep, {r["id"]: r for r in reports}

    def lines(self, data=None):
        data = data or T.load(self.json)
        return [f"{s['speaker']}: {s['text']}" for s in data["segments"]]


class InsertSpeaker(unittest.TestCase):
    def test_unlabelled_insert_continues_reply_of_preceding_words(self):
        # «Просто оно того не стоит, это время» — «это время» продолжает реплику Бориса
        c = Case([seg("Борис", W("Просто оно того не стоит.", 10.0)), seg("Анна", W("Да.", 13.0))],
                 [item("r100", 11.2, 11.2, "Борис", "стоит.")])
        data, rep, reps = c.answer({"r100": "Просто оно того не стоит, это время."})
        self.assertEqual(c.lines(data)[0], "Борис: Просто оно того не стоит, это время.")
        self.assertEqual(c.lines(data)[1], "Анна: Да.")
        log = data["edits"][-1]
        self.assertEqual(log["after"]["speaker"], "Борис")
        self.assertTrue(all(not x.startswith("?") for x in reps["r100"]["inserted"]))

    def test_insert_at_reply_boundary_goes_to_speaker_before_it(self):
        c = Case([seg("Анна", W("я пришла", 0.0)), seg("Борис", W("ну и что", 1.0))],
                 [item("r1", 0.3, 1.0, "Анна", "пришла ну")])
        data, _, _ = c.answer({"r1": "я пришла вчера ну и что"})
        self.assertEqual(c.lines(data), ["Анна: я пришла вчера", "Борис: ну и что"])

    def test_missed_speech_without_speaker_goes_to_human(self):
        c = Case([seg("Анна", W("раз два", 0.0)), seg("Борис", W("три", 5.0))],
                 [item("r2", 2.0, 3.0, None, "", reasons=("missed",))])
        data, rep, reps = c.answer({"r2": "что-то тихое"})
        self.assertIn("говорящего", reps["r2"]["human"])
        r2 = next(i for i in data["review"] if i["id"] == "r2")
        self.assertEqual(r2["status"], "human")
        self.assertEqual(c.lines(data), ["Анна: раз два", "Борис: три"])     # ничего не вставлено


class HumanEditWhole(unittest.TestCase):
    def test_insert_after_its_anchor_not_by_estimated_time(self):
        # «Анна: и он у нас»: «у нас» — внутри реплики Анны, а не после «Борис: Ну,»
        c = Case([seg("Анна", W("и он", 262.0)), seg("Борис", [{"w": "Ну,", "s": 262.5, "e": 262.7, "c": 0.8}])],
                 [item("r57", 262.0, 262.3, "Анна", "и он")])
        data, _, _ = c.answer({"r57": "Анна: и он у нас"})
        self.assertEqual(c.lines(data), ["Анна: и он у нас", "Борис: Ну,"])

    def test_foreign_word_inside_reply_moves_after_it_as_overlap(self):
        c = Case([seg("Анна", W("и он", 0.0)), seg("Борис", W("Ну,", 0.55)), seg("Анна", W("у нас", 0.9))],
                 [item("r57", 0.0, 1.2, "Анна", "и он Ну, у нас")])
        data, rep, reps = c.answer({"r57": "Анна: и он у нас"})
        self.assertEqual(c.lines(data), ["Анна: и он у нас", "Борис: Ну,"])
        self.assertTrue(data["segments"][1].get("overlap"))
        self.assertIn("одновременно", T.render_md(data, text_only=True))
        self.assertEqual(reps["r57"]["overlap"], ["Борис: Ну,"])

    def test_same_speaker_word_not_named_inside_reply_is_deleted(self):
        c = Case([seg("Анна", W("и он он у нас", 0.0))], [item("r3", 0.0, 1.2, "Анна", "и он он у нас")])
        data, rep, reps = c.answer({"r3": "Анна: и он у нас"})
        self.assertEqual(c.lines(data), ["Анна: и он у нас"])
        self.assertEqual(rep["delete"], 1)
        self.assertFalse(rep["proposed_delete"])

    def test_duplicate_of_inserted_word_is_removed(self):
        # модель услышала «нас» позже, после Бориса; человек: «и он у нас» — «нас» в аудио одно
        c = Case([seg("Анна", W("и он", 0.0)), seg("Борис", W("Ну, слушай, давай", 0.6)),
                  seg("Анна", [{"w": "нас", "s": 1.6, "e": 1.8, "c": 0.5}])],
                 [item("r4", 0.0, 0.3, "Анна", "и он")])
        data, rep, reps = c.answer({"r4": "Анна: и он у нас"})
        self.assertEqual(c.lines(data), ["Анна: и он у нас", "Борис: Ну, слушай, давай"])
        self.assertEqual(reps["r4"]["deleted"], ["нас"])

    def test_word_at_boundary_of_two_replies_is_only_proposed(self):
        c = Case([seg("Анна", W("я тут", 0.0)), seg("Борис", W("эээ а я там", 0.7))],
                 [item("r5", 0.0, 1.6, "Анна", "я тут эээ а я там")])
        data, rep, _ = c.answer({"r5": "Анна: я тут Борис: а я там"})
        self.assertIn("эээ", " ".join(c.lines(data)))
        self.assertEqual([d["word"] for d in rep["proposed_delete"]], ["эээ"])

    def test_card_is_not_applied_partially(self):
        c = Case([seg("Анна", W("раз два три", 0.0))], [item("r6", 0.0, 0.6, "Анна", "раз два три")])
        edits = [{"op": "replace", "t": 0.0, "w": "раз", "text": "один", "card": "r6"},
                 {"op": "replace", "t": 9.99, "w": "нет", "text": "ноль", "card": "r6"}]
        data, rep = T.apply_word_edits(c.json, c.md, edits, by="human", answers={"r6": "один два ноль"})
        self.assertEqual(c.lines(data), ["Анна: раз два три"])
        self.assertIn("r6", rep["failed"])
        r6 = next(i for i in data["review"] if i["id"] == "r6")
        self.assertEqual(r6["status"], "human")
        self.assertIn("не применён", r6["note"])


class Typos(unittest.TestCase):
    def test_typo_fixed_only_for_ordinary_lowercase_words(self):
        w = {"w": "прикольно", "alt": "", "alt_whisper": ""}
        self.assertEqual(F.fix_typo("приколько", w, frozenset(), frozenset())[0], "прикольно")
        for raw, model in (("Quartzline", "Kwartzline"), ("ПРИКОЛЬКО", "ПРИКОЛЬНО"), ("Приколько", "Прикольно"),
                           ("v2.19", "v2.18"), ("1-ая", "1-ой"), ("Вереск", "Верес")):
            self.assertEqual(F.fix_typo(raw, {"w": model}, frozenset(), frozenset())[0], raw, raw)
        self.assertEqual(F.fix_typo("берёзк", {"w": "берёзка"}, frozenset(), frozenset({"берёзк"}))[0], "берёзк")

    def test_human_spelling_of_name_and_version_wins_across_whole_phrase(self):
        # место разбора — только «Kwartzline», а человек написал всю фразу: «2.19.3» тоже применяется
        c = Case([seg("Борис", W("Это Kwartzline Pro 219.3 и всё", 0.0))], [item("r88", 0.3, 0.3, "Борис", "Kwartzline")])
        data, rep, reps = c.answer({"r88": "Это Quartzline Pro 2.19.3"})
        self.assertEqual(c.lines(data), ["Борис: Это Quartzline Pro 2.19.3 и всё"])
        self.assertFalse(reps["r88"]["typos"])

    def test_answer_wider_than_window_extends_it(self):
        words = W("один два три четыре пять шесть семь восемь девять десять", 0.0, step=1.0)
        c = Case([seg("Борис", words)], [item("r9", 8.0, 8.0, "Борис", "девять")])
        data, _, _ = c.answer({"r9": "два три четыре пять шесть семь восемь девять десять"})
        self.assertEqual(c.lines(data), ["Борис: один два три четыре пять шесть семь восемь девять десять"])
        c2 = Case([seg("Борис", W("один Два три четыре пять шесть семь восемь девять", 0.0, step=1.0))],
                  [item("r9", 8.0, 8.0, "Борис", "девять")])
        data, _, _ = c2.answer({"r9": "один два три четыре пять шесть семь восемь девятый"})
        self.assertEqual(c2.lines(data), ["Борис: один два три четыре пять шесть семь восемь девятый"])


class VocabSuggestions(unittest.TestCase):
    def test_function_and_ordinary_words_are_not_suggested(self):
        data = {"speakers": NAMES, "segments": [seg("Борис", W("по тетрадями и тетрадями", 0.0))]}
        for wd in ("Если", "Есть", "Это", "Анна", "Тетрадями", "на"):
            self.assertFalse(T.vocab_candidate(wd, data), wd)
        for wd in ("Quartzline", "Лисотека"):
            self.assertTrue(T.vocab_candidate(wd, data), wd)

    def test_with_morphology_common_nouns_are_not_names(self):
        import freeform
        if not freeform._morph():
            self.skipTest("нет pymorphy3")
        # «Вереск» — обычное слово (растение): совпадает с названием — только через replace_if, не подсказкой
        self.assertFalse(T.vocab_candidate("Вереск"))
        self.assertFalse(T.vocab_candidate("Тетрадями"))
        self.assertTrue(T.vocab_candidate("Москва"))
        self.assertTrue(T.vocab_candidate("Лисотека"))

    def test_suggestions_from_answers_have_ready_entry(self):
        c = Case([seg("Борис", W("ну если Kwartzline тут", 0.0))], [item("r1", 0.6, 0.6, "Борис", "Kwartzline")])
        data, _, _ = c.answer({"r1": "ну Если Quartzline тут"})
        sug = [v["to"] for v in data.get("vocab_suggestions") or []]
        self.assertIn("Quartzline", sug)
        self.assertNotIn("Если", sug)
        self.assertEqual(next(v for v in data["vocab_suggestions"] if v["to"] == "Quartzline")["add"],
                         {"terms": ["Quartzline"]})

    def test_apply_review_pair_gives_replace_entry(self):
        c = Case([seg("Борис", W("это Kwartzline", 0.0))], [item("r1", 0.3, 0.3, "Борис", "Kwartzline")])
        _, rep = T.apply_review(c.json, c.md, [{"id": "r1", "text": "Quartzline"}])
        s = next(v for v in rep["vocab_suggestions"] if v["to"] == "Quartzline")
        self.assertEqual(s["add"], {"replace": {"Kwartzline": "Quartzline"}})
        # обычное слово (без pymorphy3 — встречается в стенограмме со строчной) в словарь не предлагается
        c2 = Case([seg("Борис", W("тетрадями это тетрадь", 0.0))], [item("r1", 0.6, 0.6, "Борис", "тетрадь")])
        _, rep2 = T.apply_review(c2.json, c2.md, [{"id": "r1", "text": "Тетрадями"}])
        self.assertFalse([v for v in rep2["vocab_suggestions"] if v["to"] == "Тетрадями"])


class SpanAndReopen(unittest.TestCase):
    def _case(self):
        return Case([seg("Борис", W("версия 219.3 готова", 0.0)), seg("Анна", W("ага", 2.0))],
                    [item("r1", 0.3, 0.3, "Борис", "219.3"), item("r2", 2.0, 2.0, "Анна", "ага")])

    def test_edit_span_text_and_speaker(self):
        c = self._case()
        data, rep = T.edit_span(c.json, c.md, 0.3, 0.3, "2.19.3", note="версия по экрану", by="human")
        self.assertEqual(c.lines(data)[0], "Борис: версия 2.19.3 готова")
        self.assertEqual(rep["resolved"], ["r1"])
        self.assertEqual(data["edits"][-1]["note"], "правка отрезка: версия по экрану")
        data, _ = T.edit_span(c.json, c.md, 2.0, 2.0, "ага", speaker="Борис")
        self.assertEqual(c.lines(data), ["Борис: версия 2.19.3 готова ага"])

    def test_edit_span_insert_delete_and_errors(self):
        c = self._case()
        with self.assertRaises(ValueError):
            T.edit_span(c.json, c.md, 5.0, 6.0, "тихо")             # нет слов и нет говорящего
        with self.assertRaises(ValueError):
            T.edit_span(c.json, c.md, 0.0, 0.3, "x", speaker="Глеб")   # нет такого участника
        data, _ = T.edit_span(c.json, c.md, 5.0, 6.0, "тихо сказала", speaker="Анна")
        self.assertEqual(c.lines(data)[-1], "Анна: ага тихо сказала")
        data, _ = T.edit_span(c.json, c.md, 0.6, 0.6, "")
        self.assertEqual(c.lines(data)[0], "Борис: версия 219.3")

    def test_reopen(self):
        c = self._case()
        T.apply_review(c.json, c.md, [{"id": "r1"}, {"id": "r2", "status": "human", "note": "неразборчиво"}])
        data, rep = T.reopen(c.json, c.md, ["r1", "r2", "r99"])
        self.assertEqual(rep["reopened"], ["r1", "r2"])
        self.assertEqual(rep["unknown"], ["r99"])
        self.assertTrue(all(i["status"] == "open" and "note" not in i for i in data["review"]))
        self.assertFalse(data["review_done"])


class ReviewList(unittest.TestCase):
    def test_filters_and_brief(self):
        rv = [item("r1", 0.0, 0.2, "Борис", "а" * 300, ("voice",)), item("r2", 1.0, 3.0, "Анна", "б", ("asr", "conf")),
              {**item("r3", 4.0, 4.5, "Анна", "в", ("asr",)), "status": "resolved"}]
        c = Case([seg("Борис", W("а", 0.0)), seg("Анна", W("б в г д е ж", 1.0, step=0.6))], rv)
        r = T.review_list(c.json, brief=True)
        self.assertEqual(r["total"], 2)
        self.assertEqual(r["by_reason"], {"voice": 1, "asr": 1, "conf": 1})
        self.assertNotIn("reply", r["items"][0])
        self.assertLessEqual(len(r["items"][0]["text"]), 120)
        self.assertEqual([i["id"] for i in T.review_list(c.json, reasons=["asr"])["items"]], ["r2"])
        self.assertEqual([i["id"] for i in T.review_list(c.json, min_sec=1.0)["items"]], ["r2"])
        self.assertEqual([i["id"] for i in T.review_list(c.json, status="all", brief=True)["items"]], ["r1", "r2", "r3"])
        self.assertEqual(T.review_list(c.json, limit=1)["next_offset"], 1)


class SharedVocab(unittest.TestCase):
    def test_transcript_with_shared_vocab_never_feeds_group_profile(self):
        try:
            import numpy  # noqa: F401
            import voices
        except Exception:
            self.skipTest("нет numpy")
        d = Path(tempfile.mkdtemp(dir=_TMP))
        (d / "a.transcript.json").write_text(json.dumps(
            {"file": str(d / "a.mp3"), "no_profile": True, "names_confirmed": True, "group": None,
             "segments": [seg("Анна", W("раз два", 0.0))]}, ensure_ascii=False), encoding="utf-8")

        class Boom:
            def spans(self, *a, **k):
                raise AssertionError("не должно считать отпечатки")
        prof = voices.load_profile(d / "нет.json")
        self.assertFalse(voices.learn(prof, d, Boom(), lambda p: None, group="demo-group"))


if __name__ == "__main__":
    unittest.main()
