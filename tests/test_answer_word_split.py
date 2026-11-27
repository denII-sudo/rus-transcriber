"""Ф234-11б: ответ «n слов → m слов» вносится m отдельными словами, место сохраняет ссылки на свои слова.

Синтетическая стенограмма «тест слово ещё раз» у выдуманного Глеба: одна реплика, два места — r1 «тест слово»
(два слова), r2 «ещё» (одно). Прежде ответ, менявший число слов, вносился одним словом с пробелами и временем всего
отрезка, а строка места продолжала ссылаться на заменённые слова (при записи в базу ссылки пропадали). Теперь новые
слова — отдельные, время делится поровну внутри заменённого отрезка, при замене одного слова первое новое наследует
его ID, у остальных ID новые, и строки мест ссылаются на новые слова. То же — у правки отрезка (`edit_span`) и у
своего ответа человека (`apply_word_edits`): одно правило на одно действие.
"""
import unittest
from unittest import mock

import test_answer_outcomes as base  # noqa: E402  (тот же sandbox и те же правила исходов)

import conspect  # noqa: E402
import core  # noqa: E402
import review_pack  # noqa: E402
import transcript  # noqa: E402
import voices  # noqa: E402

BACKENDS = base.BACKENDS
_result_data = base.result_data            # прежние данные: в тестах base.result_data подменяется
START = {"w1": (0.2, 0.6), "w2": (0.8, 1.4), "w3": (1.6, 2.0), "w4": (2.2, 2.6)}


def split_data():
    words = [{"w": text, "s": START[wid][0], "e": START[wid][1], "word_id": wid}
             for text, wid in (("тест", "w1"), ("слово", "w2"), ("ещё", "w3"), ("раз", "w4"))]
    data = _result_data()
    data["segments"] = [{"start": 0.2, "end": 2.6, "label": "speaker-1", "speaker": "Глеб",
                         "text": "тест слово ещё раз", "words": words}]
    data["review"] = [{"id": "r1", "status": "open", "text": "тест слово", "t": 0.2, "t1": 0.2, "t2": 1.4,
                       "speaker": "Глеб", "word_ids": ["w1", "w2"]},
                      {"id": "r2", "status": "open", "text": "ещё", "t": 1.6, "t1": 1.6, "t2": 2.0,
                       "speaker": "Глеб", "word_ids": ["w3"]}]
    return data


SPLIT_QUESTIONS = [
    {"id": "r1", "question": "Проверьте", "recommended": 0, "options": [
        {"label": "Три слова", "decision": {"text": "один два три"}},
        {"label": "Одно слово", "decision": {"text": "одно"}},
        {"label": "Вставка внутри", "decision": {"text": "тест новое лишнее слово"}},
        {"label": "Три слова, сказала Вера", "decision": {"text": "один два три", "speaker": "Вера"}}]},
    {"id": "r2", "question": "Проверьте", "recommended": 0, "options": [
        {"label": "Три слова", "decision": {"text": "первый второй третий"}},
        {"label": "Другое слово", "decision": {"text": "иное"}}]},
]


class Case(base.AnswerOutcomeCase):
    def setUp(self):
        for name, value in (("result_data", split_data), ("QUESTIONS", SPLIT_QUESTIONS)):
            patcher = mock.patch.object(base, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def applied(self, jp, md, version, aid, option):
        self.save(jp, version, aid=aid, option=option)
        rep = self.apply(jp, md)
        self.assertEqual(self.outcomes(rep)[aid], "applied", rep.get("outcomes"))
        return transcript.load(jp)

    @staticmethod
    def flat(data):
        return [w for s in data["segments"] for w in s["words"]]

    @staticmethod
    def place(data, rid):
        return next(i for i in data["review"] if i["id"] == rid)

    def ids_of(self, data, texts):
        by_text = {w["w"]: w for w in self.flat(data)}
        self.assertTrue(set(texts) <= set(by_text), [w["w"] for w in self.flat(data)])
        return [by_text[x].get("word_id") for x in texts]

    def check_split(self, data, texts, t1, t2):
        """Слова texts — отдельные, с ID, время возрастает внутри [t1; t2], у каждого s < e."""
        words = [w for w in self.flat(data) if w["w"] in texts]
        self.assertEqual([w["w"] for w in words], texts)
        self.assertTrue(all(w.get("word_id") for w in words), words)
        self.assertEqual(len({w["word_id"] for w in words}), len(words))
        self.assertAlmostEqual(words[0]["s"], t1)
        self.assertAlmostEqual(words[-1]["e"], t2)
        for a, b in zip(words, words[1:]):
            self.assertLessEqual(a["e"], b["s"] + 1e-9)
        self.assertTrue(all(t1 - 1e-9 <= w["s"] < w["e"] <= t2 + 1e-9 for w in words), words)
        return words


class CardAnswers(Case):
    def test_two_words_become_three_words(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                data = self.applied(jp, md, version, "r1", 0)
                self.assertEqual(self.words(jp), ["один", "два", "три", "ещё", "раз"])
                words = self.check_split(data, ["один", "два", "три"], 0.2, 1.4)
                self.assertFalse({"w1", "w2"} & {w["word_id"] for w in words}, "у двух заменённых нет наследника")
                self.assertEqual(self.place(data, "r1")["word_ids"], [w["word_id"] for w in words])
                self.assertEqual(self.place(data, "r2")["word_ids"], ["w3"])

    def test_two_words_become_one_word(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                data = self.applied(jp, md, version, "r1", 1)
                self.assertEqual(self.words(jp), ["одно", "ещё", "раз"])
                (word,) = self.check_split(data, ["одно"], 0.2, 1.4)
                self.assertEqual(self.place(data, "r1")["word_ids"], [word["word_id"]])

    def test_one_word_becomes_three_and_first_keeps_its_id(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                data = self.applied(jp, md, version, "r2", 0)
                words = self.check_split(data, ["первый", "второй", "третий"], 1.6, 2.0)
                self.assertEqual(words[0]["word_id"], "w3")
                self.assertNotIn("w3", [w["word_id"] for w in words[1:]])
                self.assertEqual(self.place(data, "r2")["word_ids"], [w["word_id"] for w in words])
                # ссылка конспекта на прежний ID покрывает все слова, что из него получились (part_of)
                self.assertEqual(conspect._id_text(data, ["w3"]), ("первый второй третий", "Глеб"))
                self.assertEqual([w.get("part_of") for w in words], [None, "w3", "w3"])

    def test_words_inserted_inside_place(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                data = self.applied(jp, md, version, "r1", 2)
                self.assertEqual(self.words(jp), ["тест", "новое", "лишнее", "слово", "ещё", "раз"])
                ids = self.ids_of(data, ["тест", "новое", "лишнее", "слово"])
                self.assertEqual(ids[0], "w1")
                self.assertEqual(ids[3], "w2")
                self.assertTrue(all(ids), ids)
                self.assertEqual(self.place(data, "r1")["word_ids"], ids)
                inserted = [w for w in self.flat(data) if w["w"] in ("новое", "лишнее")]
                self.assertTrue(0.6 - 1e-9 <= inserted[0]["s"] <= inserted[1]["s"] <= 0.8 + 1e-9, inserted)

    def test_answer_is_undone_and_replaced(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                start = transcript.load(jp)
                self.applied(jp, md, version, "r1", 0)
                self.save(jp, version, aid="r1", kind="revert")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "reverted", rep.get("undo_conflicts"))
                now = transcript.load(jp)
                self.assertEqual(now["segments"], start["segments"])
                self.assertEqual(now["review"], start["review"])
        with base.answer_files("sqlite") as (jp, md, version):
            start = transcript.load(jp)
            self.applied(jp, md, version, "r1", 1)
            self.save(jp, version, aid="r1", option=0)
            rep = self.apply(jp, md)
            self.assertEqual(rep.get("replaced"), ["r1"], rep.get("undo_conflicts"))
            data = transcript.load(jp)
            self.assertEqual(self.words(jp), ["один", "два", "три", "ещё", "раз"])
            self.assertEqual(self.place(data, "r1")["word_ids"], self.ids_of(data, ["один", "два", "три"]))
            self.save(jp, version, aid="r1", kind="revert")
            rep = self.apply(jp, md)
            self.assertEqual(self.outcomes(rep)["r1"], "reverted", rep.get("undo_conflicts"))
            self.assertEqual(transcript.load(jp)["segments"], start["segments"])

    def test_new_place_on_one_of_new_words(self):
        """Карточка на одно из трёх новых слов: ссылается на одно слово, ответ человека меняет только его.
        Первую правку вносит агент: поверх ответа человека на другую карточку второй ответ — конфликт (так задумано)."""
        with base.answer_files("sqlite") as (jp, md, version):
            data, rep = transcript.apply_review(jp, md, [{"id": "r1", "text": "один два три"}], by="claude")
            self.assertEqual(rep["applied"], 1, rep)
            (two,) = [w for w in self.flat(data) if w["w"] == "два"]
            data["review"].append({"id": "r3", "status": "open", "text": "два", "t": two["s"], "t1": two["s"],
                                   "t2": two["e"], "speaker": "Глеб", "word_ids": [two["word_id"]]})
            transcript.save(data, jp, md)
            data = transcript.load(jp)
            self.assertEqual([w["w"] for w in review_pack._place_words(data, self.place(data, "r3"))], ["два"])
            data, rep = transcript.apply_review(jp, md, [{"id": "r3", "text": "двое"}], by="human")
            self.assertEqual(rep["applied"], 1, rep)
            self.assertEqual(self.words(jp), ["один", "двое", "три", "ещё", "раз"])
            self.assertEqual(self.place(data, "r3")["word_ids"], [two["word_id"]])

    def test_voice_span_of_answer_keeps_its_length(self):
        """Голоса: слова человека после ответа 2 → 3 дают тот же отрезок, что заменённые слова."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                data = self.applied(jp, md, version, "r1", 3)
                spans = [(name, round(s, 3), round(e, 3)) for name, s, e, human in voices._spans(data, False)
                         if human]
                self.assertEqual(spans, [("Вера", 0.2, 1.4)])

    def test_confirmed_speaker_of_one_word_stays_on_all_new_words(self):
        """Голоса: у слова подтверждён говорящий; ответ 1 → 2 — отметку получают оба слова, отрезок тот же."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                data = transcript.load(jp)
                speaker = {"by": "human", "edit": "x1"}
                self.flat(data)[2]["provenance"] = {"speaker": dict(speaker)}
                transcript.save(data, jp, md)
                before = [s for s in voices._spans(transcript.load(jp), False) if s[3]]
                data, rep = transcript.apply_review(jp, md, [{"id": "r2", "text": "снова опять"}], by="human")
                self.assertEqual(rep["applied"], 1, rep)
                words = self.check_split(data, ["снова", "опять"], 1.6, 2.0)
                self.assertEqual([w["provenance"]["speaker"] for w in words], [speaker] * 2)
                self.assertEqual([s for s in voices._spans(transcript.load(jp), False) if s[3]], before)

    def test_quote_from_part_of_answer_has_time_of_its_word(self):
        with base.answer_files("sqlite") as (jp, md, version):
            data = self.applied(jp, md, version, "r1", 0)
            (two,) = [w for w in self.flat(data) if w["w"] == "два"]
            times = {tok[0]: tok[2] for tok in conspect._tokens(data)}
            self.assertAlmostEqual(times["два"], two["s"])
            self.assertGreater(times["два"], 0.2, "время цитаты — время всего отрезка, а не слова")
            self.assertEqual(conspect._id_text(data, [two["word_id"]]), ("два", "Глеб"))

    def test_answers_keeping_word_count_as_before(self):
        """Контроль: одно слово → одно и два → два — ID и время прежние."""
        for backend in BACKENDS:
            with self.subTest(backend=backend, case="одно → одно"), base.answer_files(backend) as (jp, md, v):
                data = self.applied(jp, md, v, "r2", 1)
                (word,) = [w for w in self.flat(data) if w["w"] == "иное"]
                self.assertEqual((word["word_id"], word["s"], word["e"]), ("w3", 1.6, 2.0))
                self.assertEqual(self.place(data, "r2")["word_ids"], ["w3"])
            with self.subTest(backend=backend, case="два → два"), base.answer_files(backend) as (jp, md, v):
                data, rep = transcript.apply_review(jp, md, [{"id": "r1", "text": "два слова"}], by="human")
                self.assertEqual(rep["applied"], 1, rep)
                got = [(w["w"], w["word_id"], w["s"], w["e"]) for w in self.flat(data)[:2]]
                self.assertEqual(got, [("два", "w1", 0.2, 0.6), ("слова", "w2", 0.8, 1.4)])
                self.assertEqual(self.place(data, "r1")["word_ids"], ["w1", "w2"])


def with_sub_place():
    """Те же данные и место r3 — только на слове «слово» (w2), внутри r1."""
    data = split_data()
    data["review"].append({"id": "r3", "status": "open", "text": "слово", "t": 0.8, "t1": 0.8, "t2": 1.4,
                           "speaker": "Глеб", "word_ids": ["w2"]})
    return data


class PlaceRows(Case):
    """Строки мест после перелинковки не пересекаются заново и отменяются вместе с ответом (#242)."""

    def test_answer_over_place_with_sub_place(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), mock.patch.object(base, "result_data", with_sub_place), \
                    base.answer_files(backend) as (jp, md, version):
                start = transcript.load(jp)
                data = self.applied(jp, md, version, "r1", 0)
                ids = self.ids_of(data, ["один", "два", "три"])
                self.assertEqual(self.place(data, "r1")["word_ids"], ids)
                # новые слова получены из слов двух мест — частично покрытому r3 не достаются
                self.assertNotIn("word_ids", self.place(data, "r3"))
                self.save(jp, version, aid="r1", kind="revert")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "reverted", rep.get("undo_conflicts"))
                now = transcript.load(jp)
                self.assertEqual(now["segments"], start["segments"])
                self.assertEqual(now["review"], start["review"])

    def test_one_word_of_sub_place_split_stays_in_both(self):
        """1 → 2 внутри r3: слова получены целиком из его слова — их получают и r3, и охватывающее r1."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), mock.patch.object(base, "result_data", with_sub_place), \
                    base.answer_files(backend) as (jp, md, version):
                data, rep = transcript.apply_review(jp, md, [{"id": "r1", "text": "тест сло во"}], by="human")
                self.assertEqual(rep["applied"], 1, rep)
                ids = self.ids_of(data, ["тест", "сло", "во"])
                self.assertEqual(self.place(data, "r1")["word_ids"], ids)
                self.assertEqual(self.place(data, "r3")["word_ids"], ids[1:])

    def test_span_over_two_places_mixing_their_words(self):
        """Правка отрезка заменяет слово r1 и слово r2 тремя словами: ни одно не достаётся ни r1, ни r2."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                transcript.edit_span(jp, md, 0.8, 2.0, "икс игрек зет", by="claude")
                data = transcript.load(jp)
                self.assertEqual(self.words(jp), ["тест", "икс", "игрек", "зет", "раз"])
                self.assertEqual(self.place(data, "r1")["word_ids"], ["w1"])
                self.assertNotIn("word_ids", self.place(data, "r2"))

    def test_gap_place_rows_are_not_relinked(self):
        data = split_data()
        data["review"].append({"id": "g1", "reasons": ["missed"], "status": "open", "t": 0.6, "t1": 0.6, "t2": 0.8,
                               "word_ids": ["w2"]})
        old = [w for w in self.flat(data) if w["word_id"] in ("w1", "w2")]
        origin = []
        transcript._diff_words(old, "один два три", "r1", origin=origin)
        transcript._relink_places(data, old, origin)
        self.assertEqual(self.place(data, "g1")["word_ids"], ["w2"], "строку пропуска трогать нельзя")
        self.assertEqual(len(self.place(data, "r1")["word_ids"]), 3)


class ConspectLinks(Case):
    """Ссылки конспекта по ID на слово, которое ответ заменил несколькими словами (1 → m), не усекаются."""

    def test_links_over_split_word(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                before = transcript.load(jp)
                self.assertEqual(conspect._id_text(before, ["w1", "w2"]), ("тест слово", "Глеб"))
                transcript.edit_span(jp, md, 0.8, 1.4, "сло во", by="claude")
                data = transcript.load(jp)
                for ids, text in ((["w2"], "сло во"), (["w1", "w2"], "тест сло во"), (["w2", "w3"], "сло во ещё"),
                                  (["w1", "w2", "w3"], "тест сло во ещё")):
                    self.assertEqual(conspect._id_text(data, ids), (text, "Глеб"), ids)

    def test_link_over_two_replaced_words_as_before(self):
        """2 → 3: у двух заменённых слов нет наследника ID — ссылка не находится, как и на main (склеенное слово
        получало новый ID)."""
        with base.answer_files("sqlite") as (jp, md, version):
            data = self.applied(jp, md, version, "r1", 0)
            self.assertEqual(conspect._id_text(data, ["w1", "w2"]), (None, None))


class Recognized(Case):
    """«Как распознано» на карточке противоречия — из заменённых слов, не из новых."""

    def test_recognized_text_after_split_and_insert(self):
        for option in (0, 2):
            with self.subTest(option=option), base.answer_files("sqlite") as (jp, md, version):
                data = self.applied(jp, md, version, "r1", option)
                words = review_pack._place_words(data, self.place(data, "r1"))
                self.assertEqual(review_pack.recognized_text(words), "тест слово")


class Inserts(Case):
    def test_inserted_words_are_marked_and_not_voice_material(self):
        """Вставка внутри места — `inserted`: в подтверждённой расшифровке её оценённое время в «авто» не идёт."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                data = self.applied(jp, md, version, "r1", 2)
                inserted = [w for w in self.flat(data) if w["w"] in ("новое", "лишнее")]
                self.assertTrue(all(w.get("inserted") for w in inserted), inserted)
                self.assertFalse(any(w.get("inserted") for w in self.flat(data) if w["w"] not in ("новое", "лишнее")))
                for s, e in [(s, e) for _n, s, e, _h in voices._spans(data, True)]:
                    self.assertFalse([w for w in inserted if s <= w["s"] < e], (s, e))

    def test_inserts_at_edges_and_between_adjacent_words_are_zero_length(self):
        old = [{"w": "тест", "s": 0.2, "e": 0.6, "word_id": "w1"}, {"w": "слово", "s": 0.6, "e": 1.4, "word_id": "w2"}]
        for text, inserted in (("в начале тест слово", ["в", "начале"]), ("тест слово в конце", ["в", "конце"]),
                               ("тест между вплотную слово", ["между", "вплотную"])):
            with self.subTest(text=text):
                out, _, _ = transcript._diff_words([dict(w) for w in old], text, "r1")
                self.assertEqual([w["w"] for w in out], text.split())
                new = [w for w in out if w["w"] in inserted]
                self.assertTrue(all(w["inserted"] and w["s"] == w["e"] for w in new), new)


class WordEdits(Case):
    """Правка отрезка и свой ответ человека идут через apply_word_edits — то же правило."""

    def test_span_edit_one_word_to_two(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                transcript.edit_span(jp, md, 1.6, 2.0, "снова опять", by="claude")
                data = transcript.load(jp)
                self.assertEqual(self.words(jp), ["тест", "слово", "снова", "опять", "раз"])
                words = self.check_split(data, ["снова", "опять"], 1.6, 2.0)
                self.assertEqual(words[0]["word_id"], "w3")
                self.assertEqual(self.place(data, "r2")["word_ids"], [w["word_id"] for w in words])

    def test_own_answer_two_words_to_three(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                edit = {"op": "place_text", "t1": 0.2, "t2": 1.4, "text": "один два три", "card": "r1",
                        "word_ids": ["w1", "w2"]}
                _, rep = transcript.apply_word_edits(jp, md, [edit], by="human", listened_by="human")
                self.assertFalse(rep["failed"], rep)
                data = transcript.load(jp)
                words = self.check_split(data, ["один", "два", "три"], 0.2, 1.4)
                self.assertEqual(self.place(data, "r1")["word_ids"], [w["word_id"] for w in words])

    def test_span_over_two_places_relinks_each(self):
        """Правка отрезка задевает часть слов места r1 и всё место r2: каждое место — на свои новые слова."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                transcript.edit_span(jp, md, 0.2, 2.0, "тест сло во ещё", by="claude")
                data = transcript.load(jp)
                self.assertEqual(self.words(jp), ["тест", "сло", "во", "ещё", "раз"])
                ids = self.ids_of(data, ["тест", "сло", "во", "ещё"])
                self.assertEqual(ids[:2], ["w1", "w2"])
                self.assertEqual(self.place(data, "r1")["word_ids"], ids[:3])
                self.assertEqual(self.place(data, "r2")["word_ids"], ["w3"])

    def test_span_deleting_all_words_of_place_drops_references(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                transcript.edit_span(jp, md, 0.2, 1.4, "", by="claude")
                data = transcript.load(jp)
                self.assertEqual(self.words(jp), ["ещё", "раз"])
                self.assertNotIn("word_ids", self.place(data, "r1"))
                self.assertEqual(self.place(data, "r2")["word_ids"], ["w3"])


class DiffWords(unittest.TestCase):
    """Чистая функция: разбиение, время, ID, источник каждого нового слова."""

    OLD = [{"w": "тест", "s": 0.2, "e": 0.6, "word_id": "w1"}, {"w": "слово", "s": 0.8, "e": 1.4, "word_id": "w2"}]

    def test_split_and_origin(self):
        origin = []
        out, conflicts, why = transcript._diff_words([dict(w) for w in self.OLD], "один два три", "r1",
                                                     origin=origin)
        self.assertEqual(([w["w"] for w in out], conflicts, why), (["один", "два", "три"], [], []))
        self.assertEqual([(round(w["s"], 3), round(w["e"], 3)) for w in out], [(0.2, 0.6), (0.6, 1.0), (1.0, 1.4)])
        self.assertEqual([w["model"] for w in out], ["тест слово", "", ""])
        self.assertEqual([src for _, src in origin], [["w1", "w2"]] * 3)
        self.assertTrue(all(a is b for a, (b, _) in zip(out, origin)))

    def test_words_without_time_keep_old_times(self):
        """Слова без времени (перенесённые старые данные): новые слова получают прежние s и e, без падения."""
        old = [{"w": "тест", "s": None, "e": None, "word_id": "w1"}, {"w": "слово", "s": None, "e": None}]
        out, _, _ = transcript._diff_words(old, "один два три", "r1")
        self.assertEqual([(w["w"], w["s"], w["e"]) for w in out], [(x, None, None) for x in ("один", "два", "три")])
        out, _, _ = transcript._diff_words(old, "тест новое лишнее слово", "r1")
        self.assertEqual([w["w"] for w in out], ["тест", "новое", "лишнее", "слово"])
        old = [{"w": "тест", "s": 0.12345, "e": 0.9}]
        out, _, _ = transcript._diff_words(old, "раз два", "r1")
        self.assertEqual([(w["s"], w["e"]) for w in out], [(0.12345, 0.512), (0.512, 0.9)])

    def test_very_short_word_split_keeps_order(self):
        old = [{"w": "а", "s": 1.0006, "e": 1.0009, "word_id": "w1"}]
        for text in ("б в", "б в г д"):
            out, _, _ = transcript._diff_words([dict(w) for w in old], text, "r1")
            times = [t for w in out for t in (w["s"], w["e"])]
            self.assertEqual(times, sorted(times), out)
            self.assertEqual((out[0]["s"], out[-1]["e"]), (1.0006, 1.0009))

    def test_one_word_to_two_inherits_id_and_provenance(self):
        old = [{"w": "ещё", "s": 1.6, "e": 2.0, "word_id": "w3", "provenance": {"text": {"by": "model"}}}]
        out, _, _ = transcript._diff_words(old, "снова опять", "r2")
        self.assertEqual([w.get("word_id") for w in out], ["w3", None])
        self.assertEqual([w.get("provenance") for w in out], [{"text": {"by": "model"}}] * 2)
        self.assertIsNot(out[0]["provenance"], out[1]["provenance"])
        self.assertEqual([(w["s"], w["e"]) for w in out], [(1.6, 1.8), (1.8, 2.0)])


if __name__ == "__main__":
    unittest.main()
