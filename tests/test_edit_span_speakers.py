"""Правка отрезка с несколькими говорящими и отчёт `resolved` — в обоих режимах хранения (РАР-6.1, PR 3б).

Раньше новый текст отрезка целиком отдавался первому говорящему, даже если текст не менялся: слово другого
говорящего молча меняло автора речи. Теперь отрезок, где слова двух и более говорящих, а `speaker` не указан,
не принимается при любом непустом тексте: отказ без записи. Удаление слов (пустой текст) никому слов не отдаёт
и разрешено. В отчёте `resolved` — только места, которые действительно закрыты в сохранённой стенограмме
(место со сбоем распознавания без прослушивания при сохранении возвращается в `open`).
Синтетика: слова Глеба w1–w3 (0.2–1.5) и слово Веры w4 (1.7–2.0), места r1–r4.
"""
import unittest

import test_edit_span_seen as seen  # noqa: E402  (те же слова и места)
import test_human_confirmation as hc  # noqa: E402

import transcript  # noqa: E402

BACKENDS = hc.BACKENDS
ALL = "тест слово дальше тоже"


def state(jp):
    return transcript.load(jp), jp.read_bytes()


def speakers_of(jp):
    return {w["word_id"]: s["speaker"] for s in transcript.load(jp)["segments"] for w in s["words"]}


class SeveralSpeakersNeedSpeaker(unittest.TestCase):
    def test_changed_text_over_two_speakers_is_refused_without_writing(self):
        for backend in BACKENDS:
            for by in ("claude", "human"):
                with self.subTest(backend=backend, by=by), seen.files(backend) as (jp, md):
                    before = state(jp)
                    with self.assertRaises(ValueError) as ctx:
                        transcript.edit_span(jp, md, 0.2, 2.0, "тест слово дальше тоже вот", by=by)
                    text = str(ctx.exception)
                    self.assertIn("несколько говорящих", text)
                    self.assertIn("Вера", text)
                    self.assertIn("Глеб", text)
                    self.assertIn("укажите speaker", text)
                    self.assertEqual(state(jp), before)          # ревизия, файлы и журнал правок не тронуты

    def test_text_that_moves_the_other_speakers_word_is_refused_too(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), seen.files(backend) as (jp, md):
                before = state(jp)
                with self.assertRaises(ValueError):
                    transcript.edit_span(jp, md, 0.2, 2.0, "тоже", by="claude")      # w4 перешло бы к Глебу
                self.assertEqual(state(jp), before)

    def test_with_speaker_the_edit_goes_through(self):
        for backend in BACKENDS:
            for by in ("claude", "human"):
                with self.subTest(backend=backend, by=by), seen.files(backend) as (jp, md):
                    transcript.edit_span(jp, md, 0.2, 2.0, "тест слово дальше тоже вот", speaker="Глеб", by=by)
                    self.assertEqual(set(speakers_of(jp).values()), {"Глеб"})

    def test_unchanged_text_is_refused_too_because_it_used_to_move_the_other_speakers_word(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), seen.files(backend) as (jp, md):
                before = state(jp)
                with self.assertRaises(ValueError) as ctx:
                    transcript.edit_span(jp, md, 0.2, 2.0, ALL, by="claude")
                self.assertIn("несколько говорящих", str(ctx.exception))
                self.assertEqual(state(jp), before)

    def test_deleting_words_needs_no_speaker(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), seen.files(backend) as (jp, md):
                transcript.edit_span(jp, md, 0.2, 2.0, "", by="claude")
                self.assertEqual(speakers_of(jp), {})

    def test_one_speaker_works_as_before(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), seen.files(backend) as (jp, md):
                transcript.edit_span(jp, md, 0.2, 1.5, "тесто слово дальше", by="claude")
                self.assertEqual(hc.word(jp, "w1")["w"], "тесто")

    def test_free_text_answer_over_two_speakers_goes_to_human_with_reason(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), seen.files(backend) as (jp, md):
                edit = {"op": "place_text", "t1": 0.2, "t2": 2.0, "text": "тест слово дальше тоже вот",
                        "card": "r1"}
                before = state(jp)
                _, rep = transcript.apply_word_edits(jp, md, [edit], by="human")
                self.assertIn("несколько говорящих", rep["failed"]["r1"])
                self.assertEqual(speakers_of(jp), {"w1": "Глеб", "w2": "Глеб", "w3": "Глеб", "w4": "Вера"})
                self.assertEqual(hc.item(jp, "r1")["status"], "human")          # место уходит человеку с причиной
                self.assertNotEqual(state(jp), before)                          # статус места записан, слова — нет


class ReportNamesOnlyPlacesReallyClosed(unittest.TestCase):
    def assert_honest(self, jp, resolved_before, rep):
        saved = {i["id"] for i in transcript.load(jp)["review"] if i["status"] == "resolved"}
        self.assertEqual(set(rep["resolved"]), saved - resolved_before)
        self.assertEqual(rep["resolved"], sorted(rep["resolved"]))

    def resolved_ids(self, jp):
        return {i["id"] for i in transcript.load(jp)["review"] if i["status"] == "resolved"}

    def test_all_places_inside_the_span(self):
        for backend in BACKENDS:
            for by in ("claude", "human"):
                with self.subTest(backend=backend, by=by), seen.files(backend) as (jp, md):
                    _, rep = transcript.edit_span(jp, md, 0.2, 1.5, "тесто слово дальше", by=by)
                    self.assertEqual(rep["resolved"], ["r1", "r2", "r3"])
                    self.assertEqual(hc.item(jp, "r4")["status"], "open")        # сбой распознавания — только прослушиванием
                    self.assert_honest(jp, set(), rep)

    def test_place_partly_inside_stays_open_and_is_not_named(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), seen.files(backend) as (jp, md):
                _, rep = transcript.edit_span(jp, md, 0.2, 0.7, "тесто слово", by="claude")
                self.assertEqual(rep["resolved"], ["r1", "r3"])
                self.assertEqual(hc.item(jp, "r2")["status"], "open")
                self.assert_honest(jp, set(), rep)

    def test_place_decided_before_is_not_named_again(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), seen.files(backend) as (jp, md):
                transcript.apply_review(jp, md, [{"id": "r1"}], by="human")
                before = self.resolved_ids(jp)
                _, rep = transcript.edit_span(jp, md, 0.2, 1.5, "тесто слово дальше", by="human")
                self.assertNotIn("r1", rep["resolved"])
                self.assert_honest(jp, before, rep)

    def test_place_left_to_human_is_closed_only_by_human(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), seen.files(backend) as (jp, md):
                transcript.apply_review(jp, md, [{"id": "r2", "status": "human"}], by="human")
                before_all = state(jp)
                with self.assertRaises(ValueError):
                    transcript.edit_span(jp, md, 0.2, 1.5, "тесто слово дальше", by="claude")
                self.assertEqual(state(jp), before_all)
                before = self.resolved_ids(jp)
                _, rep = transcript.edit_span(jp, md, 0.2, 1.5, "тесто слово дальше", by="human")
                self.assertIn("r2", rep["resolved"])
                self.assert_honest(jp, before, rep)

    def test_no_resolve_names_nothing(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), seen.files(backend) as (jp, md):
                _, rep = transcript.edit_span(jp, md, 0.2, 1.5, "тесто слово дальше", by="claude", resolve=False)
                self.assertEqual(rep["resolved"], [])
                self.assertEqual(self.resolved_ids(jp), set())


if __name__ == "__main__":
    unittest.main()
