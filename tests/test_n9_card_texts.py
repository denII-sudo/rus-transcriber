"""Н-9, PR А: тексты на карточках (слова владельца 28.11: «вписать с предупреждением», «да, писать причину»).

П3 — слово ответа на пропуск совпало с соседним словом другого говорящего: вписывается, но в предпросмотре и
отчёте «Применить» есть предупреждение. П5 — на карточке пропуска пример «Имя: слова» и запрет слов соседних
карточек. П8 — причина правки агента (note) на карточке противоречия и объяснение агенту, что делать.
Все данные синтетические.
"""
import pathlib
import unittest

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
                self.assertIn("второй раз", warn)

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
        self.assertIn("(c.warnings || []).forEach(w => action.push(", CARDS)


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
                self.assertIn("причина: по смыслу фразы", self.card(jp, "r4")["question"])

    def test_without_note_question_has_no_reason(self):
        with files("sqlite") as (jp, md, version):
            self.answer_r4(jp, md, version)
            core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="claude")
            self.assertNotIn("note", self.contradiction(jp, "r4")["proposals"][-1])
            self.assertNotIn("причина:", self.card(jp, "r4")["question"])

    def test_human_own_span_is_named_in_message(self):
        self.assertEqual(transcript._decided_by({"card": "span@12.30"}), " (своей правкой отрезка с 12,30 с)")
        self.assertEqual(transcript._decided_by({}), "")


if __name__ == "__main__":
    unittest.main()
