"""Повторное применение ответа после того, как он слил или удалил слова карточки — в обоих режимах.

Синтетическая стенограмма «тест слово», одна карточка на обоих словах. Ответ оставляет одно слово
или сливает два в одно; повтор применения ничего не меняет и не падает. В sqlite исчезнувшие ссылки
на слова убраны из карточек документа, а в пакете ответов (manifest) остаются.
"""
import copy
import json
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

    def test_repeat_takes_the_unchanged_branch_without_commit(self):
        # Правка А: повтор после слияния слов идёт веткой «ничего не изменилось», commit не вызывается.
        with mock.patch.object(base, "result_data", two_word_card),                 mock.patch.object(base, "QUESTIONS", copy.deepcopy(QUESTIONS)),                 base.answer_files("sqlite") as (jp, md, version):
            self.save(jp, version, option=0)
            core.apply_answers_files(jp, md)
            with mock.patch.object(recordings, "commit_recording_data",
                                   wraps=recordings.commit_recording_data) as commit:
                core.apply_answers_files(jp, md)
            commit.assert_not_called()

    @staticmethod
    def disk_words(jp):
        data = json.loads(jp.read_text(encoding="utf-8"))["transcript"]      # документ на диске
        return [w["w"] for s in data["segments"] for w in s["words"]]

    def failed_first_then_repeat(self, merge=None):
        """Первое применение: документ не записан на диск (ошибка). Повтор должен дописать его."""
        with mock.patch.object(base, "result_data", two_word_card),                 mock.patch.object(base, "QUESTIONS", copy.deepcopy(QUESTIONS)),                 base.answer_files("sqlite") as (jp, md, version):
            self.save(jp, version, option=0)
            with mock.patch.object(recordings, "materialize_recording", return_value=[]):
                with self.assertRaises(core.CoreError) as ctx:
                    core.apply_answers_files(jp, md)
            self.assertIn("обновление документа не завершено", str(ctx.exception))
            self.assertEqual(self.disk_words(jp), ["тест", "слово"])     # на диске ещё старый документ
            if merge is None:
                rep = core.apply_answers_files(jp, md)
            else:
                with mock.patch.object(recordings, "merge_answer_draft", merge):
                    rep = core.apply_answers_files(jp, md)
            self.assertEqual(rep["document_sync"], "synced")
            self.assertEqual(self.disk_words(jp), ["другое"])            # документ дописан
            store, binding = recordings.open_for_document(jp)
            try:
                latest = recordings.document_binding(store, binding["recording_id"])
                self.assertEqual(recordings.document_sync_state(
                    store, binding["recording_id"], latest["document_revision"]), "synced")
            finally:
                store.close()

    def test_repeat_writes_unsynced_document_on_unchanged_candidate(self):
        self.failed_first_then_repeat()

    def test_repeat_writes_unsynced_document_on_unchanged_commit(self):
        real = recordings.merge_answer_draft

        def old_merge(data, state):
            result = real(data, state)
            items = {i["id"]: i for i in (state.get("manifest") or {}).get("items") or []}
            for q in result["questions"]:
                if "word_ids" in items.get(q["id"], {}):
                    q["word_ids"] = list(items[q["id"]]["word_ids"])
            return result

        self.failed_first_then_repeat(old_merge)


if __name__ == "__main__":
    unittest.main()
