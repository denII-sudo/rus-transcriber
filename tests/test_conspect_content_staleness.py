"""Свёртка устаревает по содержанию, а не по номеру ревизии (РАР-7, этап 7.2). Синтетика, sqlite, без облака.

Показ карточки, сборка пакета карточек и сохранённый, но не применённый ответ меняют номер
ревизии разбора, но не то, что видит модель. Такие события не должны ронять идущий модельный
шаг, делать готовую свёртку устаревшей и снимать отметку проверки человеком. Изменение
содержания (слово, говорящий, признак сомнения, цель) по-прежнему делает свёртку устаревшей.
"""
import copy
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import conspect
import core
import llm_provider
import model_tasks
import recordings
import storage
import workflow


class ConspectContentStalenessTests(unittest.TestCase):
    def setUp(self):
        # Договор версий 1–3 (ответ — перечень слов, блоки по 500 слов): так доделываются задания,
        # сохранённые до этапа 7.4. Договор версии 4 проверяет tests/test_sections_contract.py.
        patcher = mock.patch.object(model_tasks, "CURRENT_VERSION", 3)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.root = Path(tempfile.mkdtemp(prefix="content-staleness-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.dbpath = self.root / "state.sqlite3"
        self.store = storage.initialize(self.dbpath, initial_display_name="Тест")
        self.addCleanup(self.store.close)
        self.calls = 0

    # ---------- синтетическая запись и запрос ----------

    def prepare_request(self, policy=None, doubt=False):
        """Запись из двух реплик разных говорящих; doubt — одно открытое сомнение на слове w0."""
        policy = {"require_conspect": True, "allow_cloud_model": True, **(policy or {})}
        request = workflow.create(self.store, self.source, policy)
        workflow.claim_start(self.store, request)
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        recording_id = str(uuid.uuid4())
        data = {"file": str(self.source), "duration": "00:00:04", "speakers": ["Спикер 1", "Спикер 2"],
                "quality": {}, "review": [], "review_done": False,
                "segments": [{"start": 0, "end": 2, "speaker": "Спикер 1",
                              "words": [{"w": f"w{i}", "s": i, "e": i + 1} for i in range(2)]},
                             {"start": 2, "end": 4, "speaker": "Спикер 2",
                              "words": [{"w": f"w{i}", "s": i, "e": i + 1} for i in range(2, 4)]}]}
        recordings.commit_worker_result(
            self.store, data, self.source, root_id, recordings.operation_context(self.store, "model"),
            recording_id=recording_id, operation_id=str(uuid.uuid4()),
            generated_at="2026-01-01T00:00:00Z", document_paths=paths)
        recordings.materialize_recording(self.store, recording_id)
        with self.store._transaction():
            self.store.db.execute("UPDATE workflow_requests SET recording_id=?, state='processed' WHERE request_id=?",
                                  (recording_id, request))
        if doubt:
            def add(transcript, review):
                word = transcript["segments"][0]["words"][0]
                review["review"] = [{"id": "doubt-1", "status": "open", "word_ids": [word["word_id"]],
                                     "t1": 0, "t2": 1}]
            self.commit(request, add)
        return request

    def recording(self, request):
        return workflow.get(self.store, request)["recording_id"]

    def commit(self, request, change):
        """Новая ревизия через обычную запись снимков; change(transcript, review) правит копии."""
        recording = self.recording(request)
        current = self.store.get_current(recording)
        transcript = copy.deepcopy(current["transcript"]["payload"])
        review = copy.deepcopy(current["review"]["payload"])
        change(transcript, review)
        binding = recordings.document_binding(self.store, recording)
        self.store.commit_change_with_document(
            recording, {"source_revision": current["transcript"]["revision"],
                        "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            transcript=None if transcript == current["transcript"]["payload"] else transcript,
            review=review, root_id=binding["root_id"], generated_at="2026-01-01T00:00:01Z")
        recordings.materialize_recording(self.store, recording)

    # События, которые меняют номер ревизии разбора, но не содержание задания.
    def show_card(self, request):
        recording = self.recording(request)
        head = recordings.recording_head(self.store, recording)
        self.store.commit_player_event(
            recording, {"source_revision": head["source_revision"], "review_revision": head["review_revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"), "card_visible")

    def build_pack(self, request):
        def change(_transcript, review):
            review["question_version"] = "pack-1"
            review["questions"] = [{"id": item["id"], "word_ids": item.get("word_ids", [])}
                                   for item in review.get("review", [])]
        self.commit(request, change)

    def save_answer(self, request):
        def change(_transcript, review):
            review["question_version"] = "pack-1"
            review["questions"] = [{"id": item["id"], "word_ids": item.get("word_ids", [])}
                                   for item in review.get("review", [])]
            review["answers"] = {item["id"]: {"id": item["id"], "version": "pack-1", "choice": "a"}
                                 for item in review.get("review", [])}
        self.commit(request, change)

    # События, которые меняют содержание задания.
    def apply_answer(self, request):
        def change(_transcript, review):
            for item in review["review"]:
                item["status"] = "resolved"
        self.commit(request, change)

    def change_word(self, request):
        def change(transcript, _review):
            transcript["segments"][1]["words"][0]["w"] = "другое"
        self.commit(request, change)

    def change_speaker(self, request):
        def change(transcript, _review):
            transcript["segments"][1]["speaker"] = None       # говорящий второй реплики не определён
        self.commit(request, change)

    def merge_turns(self, request):
        def change(transcript, _review):
            transcript["segments"][1]["speaker"] = "Спикер 1"
        self.commit(request, change)

    # ---------- подмена облака ----------

    def client(self, before_answer=None):
        test = self

        class Client:
            def __enter__(self): return self
            def __exit__(self, *args): return False

            def generate(self, task, **_kwargs):
                test.calls += 1
                if before_answer:
                    before_answer()
                return {"source_revision": task["source_revision"], "sections": [{
                    "text": "Итог блока.", "evidence": [word["word_id"] for word in task["words"]]}]}, "{}"
        return Client

    def execute(self, request, before_answer=None, **kwargs):
        with mock.patch("llm_provider.OpenCodeTextClient", self.client(before_answer)):
            return workflow.execute_model(self.store, request, **kwargs)

    def status(self, request):
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(self.dbpath)):
            return core.workflow_status(request)

    def assert_conspect_current(self, request):
        readiness = self.status(request)["readiness"]
        self.assertEqual(readiness["execution"], "completed")
        self.assertNotIn("заказанный конспект отсутствует", readiness["blockers"])

    def assert_conspect_stale(self, request):
        readiness = self.status(request)["readiness"]
        self.assertEqual(readiness["execution"], "waiting_model")
        self.assertIn("заказанный конспект отсутствует", readiness["blockers"])

    def agent_answer(self, task):
        return {"source_revision": task["source_revision"], "sections": [{
            "text": "Итог.", "evidence": [word["word_id"] for word in task["words"]]}]}

    # ---------- 1. Ревизия изменилась, содержание то же: идущий шаг принимает ответ ----------

    def test_card_shown_during_cloud_call_does_not_fail_the_step(self):
        request = self.prepare_request(doubt=True)
        result = self.execute(request, before_answer=lambda: self.show_card(request))
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.calls, 1)
        self.assertEqual(len(self.store.list_conspects(self.recording(request))), 1)

    def test_pack_built_during_cloud_call_does_not_fail_the_step(self):
        request = self.prepare_request(doubt=True)
        result = self.execute(request, before_answer=lambda: self.build_pack(request))
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.calls, 1)

    def test_saved_unapplied_answer_during_cloud_call_does_not_fail_the_step(self):
        request = self.prepare_request(doubt=True)
        result = self.execute(request, before_answer=lambda: self.save_answer(request))
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.calls, 1)

    def test_card_shown_between_blocks_keeps_saved_blocks(self):
        request = self.prepare_request()
        with mock.patch("workflow.MODEL_PART_WORDS", 2):
            calls = []

            def once():
                calls.append(1)
                if len(calls) == 1:
                    self.show_card(request)
            result = self.execute(request, before_answer=once)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.calls, 2)
        self.assertEqual(result["completed_parts"], 2)

    def test_agent_answer_after_card_shown_is_accepted_and_bound_to_current_revisions(self):
        request = self.prepare_request(doubt=True)
        task = workflow.prepare_model_task(self.store, request)
        self.show_card(request)
        self.save_answer(request)
        again = workflow.prepare_model_task(self.store, request)
        self.assertEqual(again, task)                        # задание не переписано: блоки не теряют связь
        workflow.accept_model_result(self.store, request, self.agent_answer(task))
        after = workflow.get(self.store, request)
        self.assertEqual(after["state"], "completed")
        head = recordings.recording_head(self.store, after["recording_id"])
        saved = self.store.list_conspects(after["recording_id"])[0]
        self.assertEqual(saved["source_revision"], head["source_revision"])
        self.assertEqual(after["model_task"]["review_revision"], task["review_revision"])
        self.assertNotEqual(head["review_revision"], task["review_revision"])

    def test_failed_step_can_be_continued_after_card_shown(self):
        request = self.prepare_request()
        with mock.patch("workflow.MODEL_PART_WORDS", 2):
            def fail_second():
                if self.calls == 2:
                    raise llm_provider.ProviderError("offline", "синтетический обрыв")
            with self.assertRaises(llm_provider.ProviderError):
                self.execute(request, before_answer=fail_second)
            self.show_card(request)
            calls = self.calls
            result = self.execute(request, retry_failed=True)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.calls, calls + 1)              # повторён только недостающий блок

    # ---------- 2. Готовая свёртка остаётся актуальной ----------

    def test_completed_conspect_stays_current_after_card_pack_and_saved_answer(self):
        request = self.prepare_request(doubt=True)
        self.execute(request)
        self.assert_conspect_current(request)
        for event in (self.show_card, self.build_pack, self.save_answer):
            event(request)
            self.assert_conspect_current(request)
        calls = self.calls
        self.assertEqual(self.execute(request)["state"], "completed")
        self.assertEqual(self.calls, calls)                  # повторного вызова облака нет

    # ---------- 3. Отметка проверки человеком не снимается ----------

    def confirm_verification(self, request):
        snapshot = workflow.verification_snapshot(self.store, request)
        workflow.confirm_verification(self.store, snapshot)
        self.assertEqual(self.status(request)["readiness"]["human_verification"], "confirmed")

    def test_human_verification_survives_card_pack_and_saved_answer(self):
        request = self.prepare_request({"require_human_verification": True}, doubt=True)
        self.confirm_verification(request)
        for event in (self.show_card, self.build_pack, self.save_answer):
            event(request)
            self.assertEqual(self.status(request)["readiness"]["human_verification"], "confirmed")

    # ---------- 4. Содержание изменилось: как раньше ----------

    def test_content_change_during_call_fails_honestly_with_restart_exit(self):
        for change in (self.apply_answer, self.change_word, self.change_speaker, self.merge_turns):
            with self.subTest(change=change.__name__):
                self.setUp()
                request = self.prepare_request(doubt=True)
                with self.assertRaises(llm_provider.ProviderError) as error:
                    self.execute(request, before_answer=lambda: change(request))
                self.assertEqual(error.exception.code, "revision_changed")
                self.assertIn("restart", str(error.exception))
                with self.assertRaisesRegex(storage.Conflict, "restart"):
                    self.execute(request, retry_failed=True)
                self.assertEqual(self.execute(request, restart=True)["state"], "completed")

    def test_agent_answer_after_content_change_is_refused(self):
        request = self.prepare_request(doubt=True)
        task = workflow.prepare_model_task(self.store, request)
        self.change_word(request)
        with self.assertRaises(storage.Conflict):
            workflow.accept_model_result(self.store, request, self.agent_answer(task))

    def test_completed_conspect_goes_stale_on_content_change(self):
        for change in (self.apply_answer, self.change_word, self.change_speaker):
            with self.subTest(change=change.__name__):
                self.setUp()
                request = self.prepare_request(doubt=True)
                self.execute(request)
                change(request)
                self.assert_conspect_stale(request)

    def test_completed_conspect_goes_stale_on_goal_change(self):
        request = self.prepare_request()
        self.execute(request)
        workflow.change_policy(self.store, request, {"require_conspect": True, "allow_cloud_model": True,
                                                     "goal": "Только решения"},
                               workflow.get(self.store, request)["policy_revision"])
        self.assert_conspect_stale(request)

    def test_human_verification_is_dropped_on_content_change(self):
        for change in (self.apply_answer, self.change_word):
            with self.subTest(change=change.__name__):
                self.setUp()
                request = self.prepare_request({"require_human_verification": True}, doubt=True)
                self.confirm_verification(request)
                change(request)
                self.assertEqual(self.status(request)["readiness"]["human_verification"], "none")

    # ---------- 5. Отпечаток содержания ----------

    # ---------- Задание, собранное до осторожного правила для слов без времени ----------

    def untimed_word_with_place(self, request):
        """Слово w1 без времени и открытое место без ссылок рядом с ним (перенесённые старые данные)."""
        def change(transcript, review):
            word = transcript["segments"][0]["words"][1]
            word.pop("s"), word.pop("e")
            review["review"] = [{"id": "doubt-1", "status": "open", "t1": 1.2, "t2": 1.8}]
        self.commit(request, change)

    def old_rule(self, request):
        """Пометки по прежнему правилу: место привязано к соседним словам со временем, слово без времени не задето."""
        current = self.store.get_current(self.recording(request))["transcript"]["payload"]
        words = [word["word_id"] for segment in current["segments"] for word in segment["words"]]
        return mock.patch("workflow._precise_doubts", return_value=({words[0], words[2]}, 0))

    def test_completed_conspect_built_by_old_rule_goes_stale(self):
        request = self.prepare_request()
        self.untimed_word_with_place(request)
        tokens = conspect._tokens

        def timed_tokens(data):
            # Связывание цитат само требует время у каждого слова; для проверки устаревания
            # слову без времени на время сборки свёртки подставляется время соседа.
            data = copy.deepcopy(data)
            for segment in data["segments"]:
                for word in segment["words"]:
                    word.setdefault("s", 1.0)
            return tokens(data)
        with self.old_rule(request), mock.patch("conspect._tokens", timed_tokens):
            self.assertEqual(self.execute(request)["state"], "completed")
            self.assert_conspect_current(request)
        self.assert_conspect_stale(request)

    def test_waiting_task_built_by_old_rule_needs_restart_without_cloud(self):
        request = self.prepare_request()
        self.untimed_word_with_place(request)
        with self.old_rule(request):
            workflow.prepare_model_task(self.store, request)
        with self.assertRaises(llm_provider.ProviderError) as error:
            self.execute(request)
        self.assertEqual(error.exception.code, "revision_changed")
        self.assertIn("restart", str(error.exception))
        self.assertEqual(self.calls, 0)

    def test_fingerprint_ignores_revisions_only(self):
        request = self.prepare_request(doubt=True)
        task = workflow.prepare_model_task(self.store, request)
        moved = {**task, "source_revision": task["source_revision"] + 5, "review_revision": task["review_revision"] + 7}
        self.assertEqual(workflow.content_fingerprint(moved), workflow.content_fingerprint(task))
        for key, value in (("goal", "другая цель"),
                           ("words", [{**task["words"][0], "uncertain": not task["words"][0]["uncertain"]}]
                            + task["words"][1:]),
                           ("words", [{**task["words"][0], "text": "иное"}] + task["words"][1:]),
                           ("turns", [{**task["turns"][0], "speaker": "Участник 3"}] + task["turns"][1:])):
            with self.subTest(key=key):
                self.assertNotEqual(workflow.content_fingerprint({**task, key: value}),
                                    workflow.content_fingerprint(task))


if __name__ == "__main__":
    unittest.main()
