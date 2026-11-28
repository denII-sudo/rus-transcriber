"""Н-9, PR А: тексты на карточках (слова владельца 28.11: «вписать с предупреждением», «да, писать причину»).

П3 — слово ответа на пропуск совпало с соседним словом другого говорящего: вписывается, но в предпросмотре и
отчёте «Применить» есть предупреждение. П5 — на карточке пропуска пример «Имя: слова» и запрет слов соседних
карточек. П8 — причина правки агента (note) на карточке противоречия и объяснение агенту, что делать.
Все данные синтетические.
"""
import pathlib
import unittest
from unittest import mock

import test_contradiction_card as cc
import test_pause_buttons_short_answers as six
from test_pause_buttons_short_answers import BACKENDS, files

core, review_pack, transcript = six.core, six.review_pack, six.transcript
CARDS = (pathlib.Path(__file__).resolve().parent.parent / "cards.html").read_text(encoding="utf-8")


def two_speakers():
    data = six.result()
    data["speakers"] = ["Анна", "Борис"]
    return data


class NeighborDupWarningTests(six.Case):
    def answer(self, backend, text):
        with files(backend, two_speakers()) as (jp, md, version):
            self.save(jp, version, text=text)
            rep = core.apply_answers_files(jp, md)
            return rep, None, self.texts(jp)

    def test_other_speaker_neighbor_word_is_written_with_warning(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                rep, _, texts = self.answer(backend, "Борис: вот")
                self.assertEqual(texts.count("вот"), 2, texts)          # вписано, как прежде
                (card,) = [c for c in rep["free_text"]["cards"] if c["id"] == "r1"]
                (warn,) = card["warnings"]
                self.assertIn("последнее слово ответа «вот»", warn)
                self.assertIn("Анна, 2,00 с", warn)
                self.assertIn("будет вписано второй раз", warn)
                (done,) = card["warnings_done"]                          # отчёт «Применить» — итог, не прогноз
                self.assertIn("вписано второй раз", done)
                self.assertNotIn("будет", done)

    def test_same_speaker_neighbor_is_not_inserted_and_not_warned(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                rep, _, texts = self.answer(backend, "Анна: новое вот")
                self.assertEqual(texts.count("вот"), 1, texts)
                (card,) = [c for c in rep["free_text"]["cards"] if c["id"] == "r1"]
                self.assertNotIn("warnings", card)

    def test_different_word_gives_no_warning(self):
        rep, _, texts = self.answer("sqlite", "Борис: ну")
        self.assertIn("ну", texts)
        (card,) = [c for c in rep["free_text"]["cards"] if c["id"] == "r1"]
        self.assertNotIn("warnings", card)

    def test_page_shows_warnings_in_preview_and_report(self):
        self.assertIn('one.warnings.join("; ")', CARDS)
        self.assertIn("(c.warnings_done || c.warnings || []).forEach(w => action.push(", CARDS)


class GluedWordsTests(six.Case):
    """PR Б, слово владельца «Да, делить»: знак препинания без пробела между словами — граница слов."""

    def test_split(self):
        F = six.freeform if hasattr(six, "freeform") else __import__("freeform")
        cases = {"Вот.Ну": ["Вот.", "Ну"], "Вот.ну": ["Вот.", "ну"], "Ну,да": ["Ну,", "да"],
                 "Да!Нет?Ну": ["Да!", "Нет?", "Ну"], "3.5": ["3.5"], "т.е. так": ["т.е.", "так"],
                 "и т.д.": ["и", "т.д."], "ну-ка": ["ну-ка"], "Вот... Ну": ["Вот", "Ну"]}
        for text, words in cases.items():
            with self.subTest(text=text):
                self.assertEqual([w for _, ws in F.split_replies(text, ["Анна"]) for w in ws], words)

    def test_glued_neighbor_word_is_not_doubled(self):
        """Н-9: «Анна: новое.вот» на пропуск перед «вот» того же говорящего — «вот» второй раз не вписан."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.save(jp, version, text="Анна: новое.вот")
                core.apply_answers_files(jp, md)
                texts = self.texts(jp)
                self.assertEqual(texts.count("вот"), 1, texts)
                self.assertIn("новое.", texts)


class GapCardTextTests(unittest.TestCase):
    def test_gap_card_has_example_and_neighbor_rule(self):
        self.assertIn("«Анна: Вот. Ну»", CARDS)
        self.assertIn("слова соседних карточек сюда не пишите", CARDS)
        self.assertIn('(it.reasons || []).includes("missed")', CARDS)


class ContradictionReasonTests(cc.Base):
    def test_agent_note_reaches_card_and_message_explains_next_step(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.answer_r4(jp, md, version)
                rep = core.edit_span_files(jp, md, 3.0, 3.0, "тут", note="по смыслу фразы", by="claude")
                self.not_written(rep, "r4")
                self.assertIn("эти слова решил человек (карточка r4)", rep["text"])
                self.assertIn("не повторяйте правку", rep["text"])
                self.assertEqual(self.contradiction(jp, "r4")["proposals"][-1]["note"], "по смыслу фразы")
                accept = [o for o in self.labels(jp, "r4") if o.startswith("Принять новое")]
                self.assertEqual(len(accept), 1)
                self.assertIn("причина: по смыслу фразы", accept[0])

    def test_each_proposal_option_has_its_own_reason(self):
        """Находка (1) проверки №257: причина у каждого варианта, а не только у последнего предложения."""
        with files("sqlite") as (jp, md, version):
            self.answer_r4(jp, md, version)
            core.edit_span_files(jp, md, 3.0, 3.0, "тут", note="первая причина", by="claude")
            core.edit_span_files(jp, md, 3.0, 3.0, "там", note="вторая причина", by="claude")
            accept = [o for o in self.labels(jp, "r4") if o.startswith("Принять новое")]
            self.assertEqual(len(accept), 2, accept)
            self.assertIn("«тут»", accept[0])
            self.assertIn("первая причина", accept[0])
            self.assertIn("вторая причина", accept[1])

    def test_same_proposal_with_new_reason_is_kept(self):
        """Находка (2): повтор того же текста с новой причиной сохраняет и её."""
        with files("sqlite") as (jp, md, version):
            self.answer_r4(jp, md, version)
            core.edit_span_files(jp, md, 3.0, 3.0, "тут", note="первая причина", by="claude")
            again = core.edit_span_files(jp, md, 3.0, 3.0, "тут", note="уточнённая причина", by="claude")
            self.assertTrue(again["new"], again)
            notes = [p.get("note") for p in self.contradiction(jp, "r4")["proposals"]]
            self.assertEqual(notes, ["первая причина", "уточнённая причина"])
            same = core.edit_span_files(jp, md, 3.0, 3.0, "тут", note="уточнённая причина", by="claude")
            self.assertFalse(same["new"], same)

    def test_message_follows_card_state(self):
        """Находка (3): без открытого пакета не обещать, что предложение отправлено карточкой."""
        with files("sqlite") as (jp, md, version):
            self.answer_r4(jp, md, version)
            with mock.patch.object(review_pack, "contradiction_cards", return_value={"r4": "pending"}):
                rep = core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="claude")
            self.assertEqual(rep["card"], "pending")
            self.assertNotIn("отправлено ему карточкой", rep["text"])
            self.assertIn("открытого пакета карточек нет", rep["text"])
            with mock.patch.object(review_pack, "contradiction_cards", side_effect=OSError("диск")):
                rep = core.edit_span_files(jp, md, 3.0, 3.0, "там", by="claude")
            self.assertEqual(rep["card"], "not_replaced")
            self.assertIn("повторите ту же правку", rep["text"])

    def test_without_note_question_has_no_reason(self):
        with files("sqlite") as (jp, md, version):
            self.answer_r4(jp, md, version)
            core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="claude")
            self.assertNotIn("note", self.contradiction(jp, "r4")["proposals"][-1])
            self.assertFalse(any("причина:" in o for o in self.labels(jp, "r4")))
            self.assertNotIn("причина:", self.card(jp, "r4")["question"])

    def test_human_own_span_is_named_in_message(self):
        self.assertEqual(transcript._decided_by({"card": "span@12.30"}), " (своей правкой отрезка с 12,30 с)")
        self.assertEqual(transcript._decided_by({}), "")


if __name__ == "__main__":
    unittest.main()
