"""РАР-6.1, PR 7: свой ответ человека на карточке закрывает место «модели расходятся».

Место «модели расходятся» (asr_problem.state == "disagreement") закрывается ответом человека на свою карточку,
если ответ задел слово расхождения: заменил, удалил, слил, вставил внутри места или вписал то же слово. Отметка
прослушивания (listening.via == "card") ставится после выдачи номеров слов. Ответ, после которого место осталось
открытым, получает исход applied_open, а не applied. Все данные синтетические.
"""
import ast
import copy
import re
import unittest

import test_pause_buttons_short_answers as six
from test_pause_buttons_short_answers import BACKENDS, files

core = six.core
documents = six.documents
review_pack = six.review_pack
transcript = six.transcript
PROJECT = six.PROJECT
PROBLEM = {"model": "whisper", "state": "disagreement", "words_from": ["primary", "second"]}
REMINDER = "Слова интервала изменились после прослушивания; нужна новая проверка."


def result(two_words=True, alt=True, state="disagreement", word3="дальше"):
    """Как у 6б, но r3 — «модели расходятся»: слово расхождения w3 «дальше» (2,5), рядом согласное w6 «идём»."""
    data = six.result()
    seg = data["segments"][0]
    w3 = seg["words"][2]
    w3["w"] = word3
    data["review"][2]["text"] = word3
    if alt:
        w3.update(alt_whisper="дальнейшем", alt_span=1, agree=False)
    r3 = data["review"][2]
    if two_words:
        seg["words"].insert(3, six.word("w6", "идём", 2.75, 2.9))
        r3.update(t2=2.75, text="дальше идём", word_ids=["w3", "w6"])
    seg["text"] = " ".join(w["w"] for w in seg["words"])
    r3.update(alt_whisper="дальнейшем", gap=[r3["t1"], round(r3["t2"] + 0.04, 2)],
              asr_problem=dict(PROBLEM, state=state))
    return data


class Base(six.Case):
    def check_outcome_matches_place(self, jp, rep):
        """Т6: исход «применён» — только у ответа, чьё место закрыто; отчёт равен сохранённому состоянию."""
        places = {i["id"]: i for i in transcript.load(jp)["review"]}
        for aid, row in self.outcomes(rep).items():
            if row["outcome"] == "applied" and aid in places:
                self.assertEqual(places[aid]["status"], "resolved", (aid, row))

    def closed_by_card(self, jp, rid="r3"):
        place = self.place_of(jp, rid)
        self.assertEqual(place["status"], "resolved", place)
        listening = place.get("listening") or {}
        self.assertEqual((listening.get("by"), listening.get("result"), listening.get("via")),
                         ("human", "words", "card"), place)
        self.assertTrue(listening.get("fingerprint"))
        return place

    def still_open(self, jp, rid="r3"):
        place = self.place_of(jp, rid)
        self.assertNotEqual(place["status"], "resolved", place)
        self.assertNotIn("listening", place)
        return place


class CardAnswerClosesTests(Base):
    """Т1, Т4: свой ответ задел слово расхождения — место закрыто, исход applied."""

    CASES = (("замена", {}, "дальнейшем идём", ["начало", "вот", "дальнейшем", "идём", "та"]),
             ("вставка внутри места", {}, "дальше же идём", ["начало", "вот", "дальше", "же", "идём", "та"]),
             ("то же слово", {"two_words": False}, "дальше", ["начало", "вот", "дальше", "та"]),
             ("то же короткое слово (путь «как есть»)", {"two_words": False, "word3": "так"}, "так",
              ["начало", "вот", "так", "та"]))

    def test_own_answer_closes_the_place(self):
        for name, kw, text, words in self.CASES:
            for backend in BACKENDS:
                with self.subTest(case=name, backend=backend), files(backend, result(**kw)) as (jp, md, version):
                    self.save(jp, version, "r3", text=text)
                    rep = self.apply(jp, md)
                    self.assertEqual(self.outcomes(rep)["r3"]["outcome"], "applied", rep.get("outcomes"))
                    self.assertEqual(self.texts(jp), words)
                    self.closed_by_card(jp)
                    self.check_outcome_matches_place(jp, rep)

    def test_mark_is_taken_after_word_ids_and_survives_the_next_save(self):
        """Т4 («дыра 1»): вставленное слово получило word_id до отпечатка; отметка переживает сохранение."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, result()) as (jp, md, version):
                self.save(jp, version, "r3", text="дальше же идём")
                self.apply(jp, md)
                inserted = next(w for w in self.flat(jp) if w["w"] == "же")
                self.assertTrue(inserted.get("word_id"))
                before = self.closed_by_card(jp)
                transcript.save(transcript.load(jp), jp, md)
                self.assertEqual(self.closed_by_card(jp)["listening"], before["listening"])

    def test_deleting_words_of_the_place_closes_it(self):
        """Т1: удаление слова расхождения и всех слов места (слова расхождения взяты до правки), слияние."""
        def delete(*words):
            return [{"op": "delete", "t": t, "w": w, "word_id": wid, "card": "r3"} for wid, w, t in words]
        cases = (("удаление", delete(("w3", "дальше", 2.5)), ["начало", "вот", "идём", "та"]),
                 ("удаление всех слов места", delete(("w3", "дальше", 2.5), ("w6", "идём", 2.75)),
                  ["начало", "вот", "та"]),
                 ("слияние двух слов", [{"op": "place_text", "t1": 2.5, "t2": 2.75, "word_ids": ["w3", "w6"],
                                         "text": "дальшеидём", "card": "r3"}], ["начало", "вот", "дальшеидём", "та"]),
                 ("вставка внутри места", [{"op": "insert", "s": 2.71, "e": 2.74, "text": "же", "speaker": "Анна",
                                            "after_id": "w3", "card": "r3"}],
                  ["начало", "вот", "дальше", "же", "идём", "та"]))
        for name, edits, words in cases:
            for backend in BACKENDS:
                with self.subTest(case=name, backend=backend), files(backend, result()) as (jp, md, _v):
                    transcript.apply_word_edits(jp, md, copy.deepcopy(edits), by="human", listened_by="human")
                    self.assertEqual(self.texts(jp), words)
                    self.closed_by_card(jp)


class CardAnswerDoesNotCloseTests(Base):
    """Т2, Т3: прежние исключения — место открыто, отметки нет, исход не applied."""

    def test_answer_beside_the_disagreement_word(self):
        """Правка карточки задела только согласное слово места: место открыто, отметки нет."""
        edit = {"op": "replace", "t": 2.75, "w": "идём", "word_id": "w6", "text": "пойдём", "card": "r3"}
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, result()) as (jp, md, _v):
                transcript.apply_word_edits(jp, md, [edit], by="human", listened_by="human")
                self.assertEqual(self.texts(jp), ["начало", "вот", "дальше", "пойдём", "та"])
                self.still_open(jp)

    def test_repeated_apply_does_not_run_the_answer_again(self):
        """applied_open — применённый ответ: повтор «Применить» его второй раз не прогоняет, карточка не падает."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, result(state="empty_unknown")) as (jp, md, version):
                self.save(jp, version, "r3", text="дальше пойдём")
                core.apply_answers_files(jp, md)
                before = transcript.load(jp)
                again = core.apply_answers_files(jp, md)
                self.assertEqual(transcript.load(jp), before)
                self.assertEqual(self.outcomes(again)["r3"]["outcome"], "applied_open")
                place = self.still_open(jp)
                self.assertEqual(place["status"], "open")
                self.assertNotIn("ответ не применён", place.get("note") or "")
                self.assertIn("r3", review_pack._applied_map(review_pack.answers_state(jp)))

    def test_applied_open_place_stays_among_open_places(self):
        """Место с applied_open не прячется: оно среди нерешённых и помечено в тексте; причина — прослушать."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, result(state="empty_unknown")) as (jp, md, version):
                self.save(jp, version, "r3", text="дальше пойдём")
                row = self.outcomes(self.apply(jp, md))["r3"]
                data = transcript.load(jp)
                self.assertIn("r3", [i["id"] for i in transcript.unresolved(data)])
                self.assertFalse(data["review_done"])
                self.assertIn("r3", {rid for marks in transcript.doubt_marks(data).values() for rid in marks.values()})
                self.assertIn("прослушайте интервал", row["why"])
                self.assertEqual(review_pack.answer_states(jp)["r3"]["state"], "applied_open")

    def test_failed_pass_is_not_closed_by_an_answer(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, result(state="empty_unknown")) as (jp, md, version):
                self.save(jp, version, "r3", text="дальнейшем идём")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r3"]["outcome"], "applied_open", rep.get("outcomes"))
                self.still_open(jp)

    def test_name_only_agent_and_other_card_do_not_mark(self):
        replace = {"op": "replace", "t": 2.5, "w": "дальше", "word_id": "w3", "text": "дальнейшем"}
        cases = (("только имя", [{"op": "relabel", "t": 2.5, "word_id": "w3", "speaker": "Борис", "card": "r3"}],
                  {"by": "human"}),
                 ("агент", [dict(replace, card="r3")], {"by": "claude"}),
                 ("через другую карточку", [dict(replace, card="r4")], {"by": "human", "closed": {"r3": "r4"}}))
        for name, edits, kw in cases:
            for backend in BACKENDS:
                with self.subTest(case=name, backend=backend), files(backend, result()) as (jp, md, _v):
                    transcript.apply_word_edits(jp, md, copy.deepcopy(edits), listened_by="human", **kw)
                    self.still_open(jp)

    def test_edit_span_by_human_does_not_mark(self):
        """Т3: правка отрезка человеком записана, место «модели расходятся» открыто."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, result()) as (jp, md, _v):
                transcript.edit_span(jp, md, 2.5, 2.7, "дальнейшем", by="human")
                self.assertIn("дальнейшем", self.texts(jp))
                self.still_open(jp)
            # правка с via (правка отрезка) отметку не ставит, даже если вызывающий передал listened_by
            with self.subTest(backend=backend, how="via"), files(backend, result()) as (jp, md, _v):
                edit = {"op": "replace", "t": 2.5, "w": "дальше", "word_id": "w3", "text": "дальнейшем", "card": "r3"}
                transcript.apply_word_edits(jp, md, [edit], by="human", via="edit_span", listened_by="human")
                self.assertIn("дальнейшем", self.texts(jp))
                self.still_open(jp)

    def test_short_agreement_is_not_listening(self):
        """«верно» не говорит, что человек прослушал: как в 6б, место открыто."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, result(two_words=False)) as (jp, md, version):
                self.save(jp, version, "r3", text="верно")
                rep = self.apply(jp, md)
                self.assertNotIn(self.outcomes(rep)["r3"]["outcome"], ("applied", "applied_open"))
                self.still_open(jp)


class LaterChangeTests(Base):
    """Т5, Т7, Т11: позднее изменение слов открывает место; исход — applied_open; откат снимает отметку."""

    def test_later_edit_reopens_and_outcome_follows(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, result()) as (jp, md, version):
                self.save(jp, version, "r3", text="дальнейшем идём")
                self.apply(jp, md)
                self.closed_by_card(jp)
                # отдельная правка слов интервала (правка отрезка по месту, решённому человеком, отказывает)
                data = transcript.load(jp)
                next(w for s in data["segments"] for w in s["words"] if w["word_id"] == "w6")["w"] = "пойдём"
                transcript.save(data, jp, md)
                self.assertIn(REMINDER, self.still_open(jp).get("note", ""))
                # Т7: простое повторное применение место не закрывает, исход честный
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r3"]["outcome"], "applied_open", rep.get("outcomes"))
                self.still_open(jp)
                self.check_outcome_matches_place(jp, rep)

    def test_applied_open_has_a_reason(self):
        self.assertIn("applied_open", review_pack.OUTCOME_REASONS)
        self.assertIn("открыт", review_pack.OUTCOME_REASONS["applied_open"])
        page = (PROJECT / "cards.html").read_text(encoding="utf-8")
        status = re.search(r"const STATUS = \{(.*?)\};", page, re.S).group(1)
        self.assertIn("applied_open:", status)

    def test_reopen_and_undo_remove_the_mark(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend, how="reopen"), files(backend, result()) as (jp, md, version):
                self.save(jp, version, "r3", text="дальнейшем идём")
                self.apply(jp, md)
                self.closed_by_card(jp)
                transcript.reopen(jp, md, ["r3"], by="human")
                self.still_open(jp)
            with self.subTest(backend=backend, how="undo"), files(backend, result()) as (jp, md, version):
                self.save(jp, version, "r3", text="дальнейшем идём")
                self.apply(jp, md)
                self.closed_by_card(jp)
                self.save(jp, version, "r3", kind="revert")
                rep = core.apply_answers_files(jp, md)
                self.assertEqual(self.outcomes(rep)["r3"]["outcome"], "reverted", rep.get("outcomes"))
                self.assertEqual(self.texts(jp), ["начало", "вот", "дальше", "идём", "та"])
                self.still_open(jp)


class DoubtsSectionTests(Base):
    """Т8: «Сомнения» — текущее слово, русский статус, округлённое время; пропуск — только у пропуска."""

    def test_renderer_version(self):
        self.assertEqual(documents.RENDERER_VERSION, 3)

    def test_sqlite_doubts(self):
        with files("sqlite", result(state="empty_unknown")) as (jp, md, version):
            self.save(jp, version, "r3", text="дальше пойдём")
            self.apply(jp, md)
            text = md.read_text(encoding="utf-8")
            doubts = text.split("## Сомнения", 1)[1]
            line = next(x for x in doubts.splitlines() if "r3" in x)
            self.assertIn("пойдём", line)
            self.assertIn("открыто", line)
            self.assertIn("00:00:02", line)
            self.assertNotRegex(doubts, r"\b(open|resolved|human)\b")
            gap_lines = [x for x in text.splitlines() if "Возможный пропуск речи" in x]
            self.assertTrue(gap_lines)
            self.assertTrue(all("r3" not in x for x in gap_lines), gap_lines)
            self.assertTrue(any("r1" in x and "00:00:01" in x for x in gap_lines), gap_lines)

    def test_legacy_doubts(self):
        with files("legacy_files", result(state="empty_unknown")) as (jp, md, version):
            self.save(jp, version, "r3", text="дальше пойдём")
            self.apply(jp, md)
            section = md.read_text(encoding="utf-8").split("## Сомнительные места", 1)[1]
            line = next(x for x in section.splitlines() if "00:00:02" in x and "пойдём" in x)
            self.assertIn("«дальше пойдём»", line)
            self.assertIn("открыто", line)


class OnePlaceRuleTests(unittest.TestCase):
    """Т10: правило «место открыто» записано один раз — transcript.place_open / place_unresolved."""

    ALLOWED = {("transcript.py", "place_open"), ("transcript.py", "place_unresolved"),
               ("workflow.py", "_place_unresolved")}     # workflow.py — работа линии 2, названное исключение

    def test_no_copies_of_the_open_rule(self):
        found = set()
        for path in sorted(PROJECT.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                for node in ast.walk(func):
                    if isinstance(node, ast.Compare) and self.is_open_rule(node):
                        found.add((path.name, func.name))
        # вложенная функция попадает и под внешнюю: сверяем только внешние имена вне разрешённых
        self.assertEqual(sorted(found - self.ALLOWED), [])

    @staticmethod
    def is_open_rule(node):
        left = node.left
        if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute) and left.func.attr == "get"
                and [getattr(a, "value", None) for a in left.args] == ["status", "open"]):
            return False
        values = []
        for comp in node.comparators:
            items = comp.elts if isinstance(comp, (ast.Tuple, ast.List, ast.Set)) else [comp]
            values += [getattr(x, "value", None) for x in items]
        return "open" in values


if __name__ == "__main__":
    unittest.main()
