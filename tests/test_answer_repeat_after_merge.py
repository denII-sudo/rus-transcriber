"""Повторное применение ответа после того, как он слил или удалил слова карточки — в обоих режимах.

Синтетическая стенограмма «тест слово», одна карточка на обоих словах. Ответ оставляет одно слово
или сливает два в одно; повтор применения ничего не меняет и не падает. В sqlite исчезнувшие ссылки
на слова убраны из карточек документа, а в пакете ответов (manifest) остаются.
"""
import copy
import unittest
from unittest import mock

import test_answer_outcomes as base  # noqa: E402  (тот же sandbox и те же правила исходов)

import core  # noqa: E402
import recordings  # noqa: E402
import review_pack  # noqa: E402

BACKENDS = base.BACKENDS
ORIGINAL_DATA = base.result_data


def two_word_card():
    data = ORIGINAL_DATA()
    data["review"] = [{"id": "r1", "status": "open", "text": "тест слово", "t": 0.2, "t1": 0.2, "t2": 1.0,
                       "speaker": "Глеб", "word_ids": ["w1", "w2"]}]
    return data


QUESTIONS = [{"id": "r1", "question": "Проверьте", "recommended": 0, "options": [
    {"label": "Слить", "decision": {"text": "другое"}},
    {"label": "Оставить первое", "decision": {"text": "тест"}},
    {"label": "Без изменений", "decision": {"text": "тест слово"}}]}]


class RepeatAfterMerge(base.AnswerOutcomeCase):
    def run_case(self, backend, option, words):
        with mock.patch.object(base, "result_data", two_word_card), \
                mock.patch.object(base, "QUESTIONS", copy.deepcopy(QUESTIONS)), \
                base.answer_files(backend) as (jp, md, version):
            self.save(jp, version, option=option)
            rep = self.apply(jp, md)            # второй запуск внутри: без ошибки, ничего не меняется
            self.assertEqual(self.outcomes(rep)["r1"], "applied")
            self.assertEqual(self.words(jp), words)
            return jp, md

    def test_repeat_after_merge_into_one_word(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self.run_case(backend, 0, ["другое"])

    def test_repeat_after_removing_a_word(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self.run_case(backend, 1, ["тест"])

    def test_repeat_without_changes(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self.run_case(backend, 2, ["тест", "слово"])

    def test_document_drops_vanished_refs_but_pack_keeps_them(self):
        with mock.patch.object(base, "result_data", two_word_card), \
                mock.patch.object(base, "QUESTIONS", copy.deepcopy(QUESTIONS)), \
                base.answer_files("sqlite") as (jp, md, version):
            self.save(jp, version, option=0)
            core.apply_answers_files(jp, md)
            core.apply_answers_files(jp, md)
            store, binding = recordings.open_for_document(jp)
            try:
                data = recordings.load_recording(store, binding["recording_id"])
            finally:
                store.close()
            self.assertNotIn("word_ids", data["questions"][0])
            self.assertEqual(review_pack.manifest(jp)["items"][0]["word_ids"], ["w1", "w2"])

    def test_unchanged_commit_is_not_an_error(self):
        # Даже если черновик снова принёс исчезнувшие ссылки, commit ничего не пишет — это не сбой.
        real = recordings.merge_answer_draft

        def old_merge(data, state):
            result = real(data, state)
            items = {i["id"]: i for i in (state.get("manifest") or {}).get("items") or []}
            for q in result["questions"]:
                if "word_ids" in items.get(q["id"], {}):
                    q["word_ids"] = list(items[q["id"]]["word_ids"])
            return result

        with mock.patch.object(base, "result_data", two_word_card), \
                mock.patch.object(base, "QUESTIONS", copy.deepcopy(QUESTIONS)), \
                base.answer_files("sqlite") as (jp, md, version):
            self.save(jp, version, option=0)
            core.apply_answers_files(jp, md)
            before = self.pack_state(jp)
            with mock.patch.object(recordings, "merge_answer_draft", old_merge):
                rep = core.apply_answers_files(jp, md)
            self.check_outcomes(jp, rep)
            self.assertEqual(self.pack_state(jp), before)
            self.assertEqual(rep["document_sync"], "synced")


if __name__ == "__main__":
    unittest.main()
