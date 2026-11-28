import asyncio
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import core
import model_tasks
import recordings
import storage
import workflow
import worker


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        # Договор версий 1–3 (ответ — перечень слов, блоки по 500 слов): так доделываются задания,
        # сохранённые до этапа 7.4. Договор версии 4 проверяет tests/test_sections_contract.py.
        patcher = mock.patch.object(model_tasks, "CURRENT_VERSION", 3)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.root = Path(tempfile.mkdtemp(prefix="p3-workflow-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.store = storage.initialize(self.root / "state.sqlite3", initial_display_name="Тест")

    def tearDown(self):
        self.store.close()

    def make_recording(self, count=1, review=None, segments=None, speaker_names=None):
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        ident = str(uuid.uuid4())
        data = {"file": str(self.source), "duration": "00:00:01", "speakers": ["Спикер 1"],
                "quality": {}, "review": review or [], "review_done": False,
                "segments": segments or [{"start": 0, "end": count, "speaker": "Спикер 1",
                                          "words": [{"w": "слово", "s": i, "e": i + 1} for i in range(count)]}]}
        if speaker_names:
            data["speaker_names"] = speaker_names
        recordings.commit_worker_result(
            self.store, data, self.source, root_id, recordings.operation_context(self.store, "model"),
            recording_id=ident, operation_id=str(uuid.uuid4()),
            generated_at="2026-01-01T00:00:00Z", document_paths=paths)
        recordings.materialize_recording(self.store, ident)
        return ident

    def attach_recording(self, request, recording):
        with self.store._transaction():
            self.store.db.execute("UPDATE workflow_requests SET recording_id=? WHERE request_id=?",
                                  (recording, request))

    def test_request_survives_reopen_and_start_is_only_claimed_once(self):
        ident = workflow.create(self.store, self.source)
        job = workflow.claim_start(self.store, ident)
        with self.assertRaises(storage.Conflict):
            workflow.claim_start(self.store, ident)
        self.store.close()
        self.store = storage.open_store(self.root / "state.sqlite3")
        self.assertEqual(workflow.get(self.store, ident)["job_id"], job)
        self.assertEqual(workflow.observe(self.store, ident, None)["state"], "starting")

    def test_model_result_materializes_and_replay_does_not_duplicate(self):
        request = workflow.create(self.store, self.source, {"require_conspect": True})
        workflow.claim_start(self.store, request)
        recording = self.make_recording()
        self.attach_recording(request, recording)
        workflow.observe(self.store, request, {"state": "done", "recording_id": recording})
        task = workflow.prepare_model_task(self.store, request)
        self.assertEqual(task["words"][0]["text"], "слово")
        answer = {"source_revision": task["source_revision"],
                  "sections": [{"text": "Обсуждалась тема.", "evidence": [task["words"][0]["word_id"]]}]}
        first = workflow.accept_model_result(self.store, request, answer)
        again = workflow.accept_model_result(self.store, request, answer)
        self.assertEqual(first, again)
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")
        self.assertEqual(len(self.store.list_conspects(recording)), 1)
        self.assertEqual(workflow.prepare_model_task(self.store, request), task)   # чтение ничего не меняет
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")
        workflow.prepare_model_task(self.store, request, restart=True)             # новое поколение — явно
        workflow.accept_model_result(self.store, request, answer)
        self.assertEqual(self.store.list_conspects(recording)[0]["revision"], 2)
        runs = workflow.get(self.store, request)["runs"]
        self.assertEqual(len([run for run in runs if run["kind"] == "model"]), 2)

    def test_unlinked_quote_is_accepted_with_a_mark(self):
        # Цитаты по одной (РАР-7, 7.5): не найденная цитата не отклоняет ответ — она печатается без кавычек
        # с пометкой. До 7.5 такой ответ отклонялся (для заданий версии 1 это осталось: см.
        # tests/test_quotes_one_by_one.py).
        request = workflow.create(self.store, self.source, {"require_conspect": True})
        workflow.claim_start(self.store, request)
        recording = self.make_recording()
        self.attach_recording(request, recording)
        workflow.observe(self.store, request, {"state": "done", "recording_id": recording})
        task = workflow.prepare_model_task(self.store, request)
        answer = {"source_revision": task["source_revision"],
                  "sections": [{"text": "«совершенно другое»", "evidence": [task["words"][0]["word_id"]]}]}
        workflow.accept_model_result(self.store, request, answer)
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")
        payload, md = self.conspect_files(recording)
        self.assertEqual(payload["quotes"], {})
        self.assertIn("совершенно другое (не найдено в стенограмме)", md)
        self.assertNotIn("«", md)

    def test_committed_worker_output_recovers_without_rerunning_worker(self):
        request = workflow.create(self.store, self.source)
        workflow.claim_start(self.store, request)
        recording = self.make_recording()
        self.attach_recording(request, recording)
        self.assertEqual(workflow.observe(
            self.store, request, {"state": "error", "recording_id": recording})["state"], "processed")
        with self.assertRaises(storage.Conflict):
            workflow.claim_start(self.store, request)

    def test_captured_name_survives_rename(self):
        request = workflow.create(self.store, self.source)
        identity = self.store.current_identity()
        self.store.rename_user(identity["user_id"], "Новое имя")
        context = workflow.get(self.store, request)["context"]
        self.assertEqual(context["initiated_by"]["user_name_snapshot"], "Тест")
        self.assertEqual(context["actor_kind"], "model")

    def test_policy_revision_preserves_history_and_rejects_stale_change(self):
        request = workflow.create(self.store, self.source, {"goal": "Первая цель"})
        self.assertEqual(workflow.change_policy(self.store, request, {"goal": "Другая цель"}, 1), 2)
        with self.assertRaises(storage.Conflict):
            workflow.change_policy(self.store, request, {}, 1)
        saved = workflow.get(self.store, request)
        self.assertEqual([p["policy"]["goal"] for p in saved["policy_history"]],
                         ["Первая цель", "Другая цель"])

    def test_cloud_denial_cannot_be_reversed_on_existing_request(self):
        request = workflow.create(self.store, self.source, {"allow_cloud_model": False})
        with self.assertRaisesRegex(storage.Conflict, "создайте новый запрос"):
            workflow.change_policy(self.store, request, {"allow_cloud_model": True}, 1)
        saved = workflow.get(self.store, request)
        self.assertFalse(saved["policy"]["allow_cloud_model"])
        self.assertEqual(saved["policy_revision"], 1)

    def test_model_commit_survives_materialization_failure(self):
        request = workflow.create(self.store, self.source, {"require_conspect": True})
        workflow.claim_start(self.store, request)
        recording = self.make_recording()
        self.attach_recording(request, recording)
        workflow.observe(self.store, request, {"state": "done", "recording_id": recording})
        task = workflow.prepare_model_task(self.store, request)
        answer = {"source_revision": task["source_revision"],
                  "sections": [{"text": "Краткий итог.", "evidence": [task["words"][0]["word_id"]]}]}
        with mock.patch("recordings.materialize_conspects", side_effect=OSError("synthetic failure")):
            with self.assertRaises(OSError):
                workflow.accept_model_result(self.store, request, answer)
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")
        workflow.accept_model_result(self.store, request, answer)
        self.assertEqual(len(self.store.list_conspects(recording)), 1)

    def test_stale_review_rejects_model_part_without_using_repair_budget(self):
        request = workflow.create(self.store, self.source, {"require_conspect": True})
        workflow.claim_start(self.store, request)
        recording = self.make_recording()
        self.attach_recording(request, recording)
        workflow.observe(self.store, request, {"state": "done", "recording_id": recording})
        part = workflow.model_part(self.store, request, 0)
        current = self.store.get_current(recording)
        binding = recordings.document_binding(self.store, recording)
        # Устаревает содержание (новое открытое сомнение), а не только номер ревизии разбора.
        review = json.loads(json.dumps(current["review"]["payload"]))
        review["review"] = review.get("review", []) + [
            {"id": str(uuid.uuid4()), "status": "open", "word_ids": [part["task"]["words"][0]["word_id"]],
             "t1": 0, "t2": 1}]
        self.store.commit_change_with_document(
            recording, {"source_revision": current["transcript"]["revision"],
                        "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            review=review, root_id=binding["root_id"],
            generated_at="2026-01-01T00:00:01Z")
        answer = {"source_revision": part["task"]["source_revision"], "sections": [{
            "text": "Итог.", "evidence": [part["task"]["words"][0]["word_id"]]}]}
        with self.assertRaises(storage.Conflict):
            workflow.accept_model_part(self.store, request, 0, answer)
        self.assertEqual(workflow.get(self.store, request)["model_attempts"], 0)

    def test_core_exposes_readiness_and_preserves_context_at_launch(self):
        self.store.close()
        dbpath = self.root / "state.sqlite3"
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)), \
             mock.patch("settings.recording_backend", return_value="sqlite"), \
             mock.patch("core.start_transcription", return_value={"job_id": "stub"}) as launch:
            started = core.start_workflow(self.source, {"require_conspect": True})
            self.assertFalse(launch.call_args.kwargs["voices"])
            self.assertEqual(launch.call_args.kwargs["workflow_context"]["initiated_by"]["user_name_snapshot"], "Тест")
            with mock.patch("core.job_status", return_value={"state": "running"}):
                result = core.workflow_status(started["request_id"])
            self.assertEqual(result["readiness"]["deliverable"], "blocked")
            self.assertEqual(result["readiness"]["execution"], "running")
        self.store = storage.open_store(dbpath)
        saved = workflow.get(self.store, started["request_id"])
        self.assertEqual(saved["assessment_history"][0]["assessment"]["policy_revision"], 1)

    def test_spawn_failure_keeps_request_runs_and_steps_interrupted(self):
        self.store.close()
        dbpath = self.root / "state.sqlite3"
        self.store = storage.open_store(dbpath)
        jobs = self.root / "jobs"
        jobs.mkdir()

        def failed_launch(script, args, src, kind, *, initial_state=None, job_id=None):
            (jobs / f"{job_id}.json").write_text(json.dumps(
                {**(initial_state or {}), "state": "queued", "kind": kind, "file": str(src)}),
                encoding="utf-8")
            raise OSError("synthetic process creation failure")

        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)), \
             mock.patch("settings.recording_backend", return_value="sqlite"), \
             mock.patch.object(core, "JOBS", jobs), \
             mock.patch.object(core, "launch", side_effect=failed_launch):
            started = core.start_workflow(self.source)
            saved = workflow.get(self.store, started["request_id"])
            self.assertEqual(started["state"], "interrupted")
            self.assertEqual(saved["state"], "interrupted")
            self.assertEqual([(run["kind"], run["state"]) for run in saved["runs"]],
                             [("asr", "interrupted")])
            self.assertEqual([step["state"] for step in saved["steps"]], ["interrupted"])
            status = core.workflow_status(started["request_id"])
            self.assertEqual(status["request"]["state"], "interrupted")

    def test_worker_commit_is_linked_atomically_and_recoverable_without_job(self):
        request_id = workflow.create(self.store, self.source)
        workflow.claim_start(self.store, request_id)
        request = workflow.get(self.store, request_id)
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        data = {"file": str(self.source), "duration": "00:00:01", "speakers": ["Спикер 1"],
                "quality": {}, "review": [], "review_done": False,
                "segments": [{"start": 0, "end": 1, "speaker": "Спикер 1",
                              "words": [{"w": "слово", "s": 0, "e": 1}]}]}
        status = mock.Mock()
        status.data = {"workflow_request_id": request_id}
        target = {"root_id": root_id, "paths": paths, "context": request["context"]}
        dbpath = self.root / "state.sqlite3"
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)):
            result = worker._save_worker_result(data, self.source, status, target, None, None)
        self.assertEqual(workflow.get(self.store, request_id)["recording_id"], result["recording_id"])
        observed = workflow.observe(self.store, request_id, None)
        self.assertEqual(observed["state"], "processed")
        self.assertEqual(observed["recording_id"], result["recording_id"])

    def test_missing_document_blocks_readiness(self):
        request = workflow.create(self.store, self.source)
        workflow.claim_start(self.store, request)
        recording = self.make_recording()
        self.attach_recording(request, recording)
        workflow.observe(self.store, request, {"state": "done", "recording_id": recording})
        dbpath = self.root / "state.sqlite3"
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)), \
             mock.patch("documents._file_digest", side_effect=OSError("synthetic read failure")):
            result = core.workflow_status(request)
        self.assertEqual(result["readiness"]["document_sync"], "error")
        self.assertEqual(result["readiness"]["deliverable"], "blocked")

    def test_backup_disables_inflight_work(self):
        request = workflow.create(self.store, self.source)
        workflow.claim_start(self.store, request)
        backup = self.root / "backup.sqlite3"
        self.store.backup(backup)
        copy = storage.open_store(backup)
        try:
            row = copy.db.execute("SELECT state, job_id FROM workflow_requests WHERE request_id=?",
                                  (request,)).fetchone()
            self.assertEqual(row["state"], "interrupted")
            self.assertIsNone(row["job_id"])
        finally:
            copy.close()

    def test_explicit_verification_is_revision_bound_and_preserves_author(self):
        request = workflow.create(self.store, self.source, {"require_human_verification": True})
        workflow.claim_start(self.store, request)
        recording = self.make_recording()
        self.attach_recording(request, recording)
        workflow.observe(self.store, request, {"state": "done", "recording_id": recording})
        snapshot = workflow.verification_snapshot(self.store, request)
        verification = workflow.confirm_verification(self.store, snapshot)
        self.assertEqual(verification["human_verification"], "confirmed")
        self.assertEqual(workflow.confirm_verification(self.store, snapshot), verification)
        row = self.store.db.execute("SELECT context_json FROM workflow_verifications").fetchone()
        self.assertEqual(storage._loads(row["context_json"])["actor_kind"], "human")
        identity = self.store.current_identity()
        self.store.rename_user(identity["user_id"], "Другой снимок имени")
        with self.assertRaises(storage.Conflict):
            workflow.confirm_verification(self.store, snapshot)
        dbpath = self.root / "state.sqlite3"
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)):
            result = core.workflow_status(request)
        self.assertEqual(result["readiness"]["human_verification"], "confirmed")
        current = self.store.get_current(recording)
        binding = recordings.document_binding(self.store, recording)
        # Новая ревизия без изменения содержания (показ карточки) отметку не снимает (РАР-7, этап 7.2).
        head = recordings.recording_head(self.store, recording)
        self.store.commit_player_event(
            recording, {"source_revision": head["source_revision"], "review_revision": head["review_revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"), "card_visible")
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)):
            shown = core.workflow_status(request)
        self.assertEqual(shown["readiness"]["human_verification"], "confirmed")
        current = self.store.get_current(recording)
        transcript = json.loads(json.dumps(current["transcript"]["payload"]))
        transcript["segments"][0]["words"][0]["w"] = "другое"
        self.store.commit_change_with_document(
            recording, {"source_revision": current["transcript"]["revision"],
                        "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            transcript=transcript, review=current["review"]["payload"], root_id=binding["root_id"],
            generated_at="2026-01-01T00:00:01Z")
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)):
            changed = core.workflow_status(request)
        self.assertEqual(changed["readiness"]["human_verification"], "none")

    def test_abandon_blocks_late_worker_commit(self):
        request_id = workflow.create(self.store, self.source)
        workflow.claim_start(self.store, request_id)
        request = workflow.get(self.store, request_id)
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        snapshot = workflow.abandon_snapshot(self.store, request_id)
        workflow.abandon(self.store, snapshot)
        status = mock.Mock()
        status.data = {"workflow_request_id": request_id}
        target = {"root_id": root_id, "paths": paths, "context": request["context"]}
        dbpath = self.root / "state.sqlite3"
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)):
            with self.assertRaises(RuntimeError):
                worker._save_worker_result({}, self.source, status, target, None, None)
        self.assertIsNone(workflow.get(self.store, request_id)["recording_id"])

    def test_abandon_cannot_discard_already_committed_recording(self):
        request = workflow.create(self.store, self.source)
        workflow.claim_start(self.store, request)
        snapshot = workflow.abandon_snapshot(self.store, request)
        recording = self.make_recording()
        self.attach_recording(request, recording)
        with self.assertRaises(storage.Conflict):
            workflow.abandon(self.store, snapshot)

    def test_long_recording_parts_survive_reopen_and_assemble_once(self):
        request = workflow.create(self.store, self.source, {"require_conspect": True})
        workflow.claim_start(self.store, request)
        recording = self.make_recording(count=501)
        self.attach_recording(request, recording)
        workflow.observe(self.store, request, {"state": "done", "recording_id": recording})
        first = workflow.model_part(self.store, request, 0)
        second = workflow.model_part(self.store, request, 1)
        self.assertEqual((len(first["task"]["words"]), len(second["task"]["words"])), (500, 1))
        def response(part):
            return {"source_revision": part["task"]["source_revision"], "sections": [{
                "text": "Итог блока.", "evidence": [word["word_id"] for word in part["task"]["words"]]}]}
        self.assertEqual(workflow.accept_model_part(self.store, request, 0, response(first))["state"], "waiting_model")
        different = response(first)
        different["sections"][0]["text"] = "Другой итог."
        with self.assertRaises(storage.Conflict):
            workflow.accept_model_part(self.store, request, 0, different)
        self.store.close()
        self.store = storage.open_store(self.root / "state.sqlite3")
        self.assertEqual(workflow.accept_model_part(self.store, request, 1, response(second))["state"], "completed")
        self.assertEqual(workflow.accept_model_part(self.store, request, 1, response(second))["state"], "completed")
        self.assertEqual(len(self.store.list_conspects(recording)), 1)

    # ---------- РАР-1: реплики, условные метки говорящих, границы блоков ----------

    @staticmethod
    def segment(speaker, texts, start=0.0):
        """Синтетическая реплика: слова по секунде, начиная с start."""
        return {"start": start, "end": start + len(texts), "speaker": speaker,
                "words": [{"w": text, "s": start + i, "e": start + i + 0.5} for i, text in enumerate(texts)]}

    def ready_request(self, **recording):
        request = workflow.create(self.store, self.source, {"require_conspect": True})
        workflow.claim_start(self.store, request)
        recording_id = self.make_recording(**recording)
        self.attach_recording(request, recording_id)
        workflow.observe(self.store, request, {"state": "done", "recording_id": recording_id})
        return request, recording_id

    def all_parts(self, request):
        first = workflow.model_part(self.store, request, 0)
        return [first] + [workflow.model_part(self.store, request, index)
                          for index in range(1, first["part_count"])]

    @staticmethod
    def part_answer(part, text="Итог блока."):
        return {"source_revision": part["task"]["source_revision"], "sections": [{
            "text": text, "evidence": [word["word_id"] for word in part["task"]["words"]]}]}

    def conspect_files(self, recording_id):
        saved = self.store.list_conspects(recording_id)[0]
        root = recordings.document_binding(self.store, recording_id)["root"]
        return saved["payload"], (root / saved["md_relative"]).read_text(encoding="utf-8")

    def test_new_task_is_version_3_with_labels_instead_of_names(self):
        request, _ = self.ready_request(segments=[
            self.segment("Анна Синтетическая", ["добрый", "день."]),
            self.segment("Спикер 2", ["да"], start=2),
            self.segment(None, ["шум"], start=3),
            self.segment("Анна Синтетическая", ["начнём"], start=4)])
        task = workflow.prepare_model_task(self.store, request)
        self.assertEqual((task["version"], task["unlocated_doubts"]), (3, 0))
        self.assertEqual(task["turns"], [
            {"turn": 1, "speaker": "Участник 1"}, {"turn": 2, "speaker": "Участник 2"},
            {"turn": 3, "speaker": "Говорящий не определён"}, {"turn": 4, "speaker": "Участник 1"}])
        self.assertEqual([(word["text"], word["turn"]) for word in task["words"]],
                         [("добрый", 1), ("день.", 1), ("да", 2), ("шум", 3), ("начнём", 4)])
        self.assertEqual(set(task["words"][0]), {"word_id", "text", "uncertain", "turn"})
        dumped = json.dumps(task, ensure_ascii=False)
        for private in ("Анна", "Синтетическая", "Спикер", str(self.source), "synthetic.wav"):
            self.assertNotIn(private, dumped)
        self.assertEqual(workflow.get(self.store, request)["model_task"], task)

    def test_block_ends_on_turn_boundary(self):
        request, _ = self.ready_request(segments=[
            self.segment("Спикер 1", ["слово"] * 300),
            self.segment("Спикер 2", ["слово"] * 300, start=300),
            self.segment("Спикер 1", ["слово"] * 100, start=600)])
        task = workflow.prepare_model_task(self.store, request)
        parts = self.all_parts(request)
        self.assertEqual([len(part["task"]["words"]) for part in parts], [300, 400])
        self.assertEqual([[turn["turn"] for turn in part["task"]["turns"]] for part in parts], [[1], [2, 3]])
        self.assertEqual([word for part in parts for word in part["task"]["words"]], task["words"])
        self.assertTrue(all(len(part["task"]["words"]) <= workflow.MODEL_PART_WORDS for part in parts))
        self.assertEqual(workflow.accept_model_part(self.store, request, 0, self.part_answer(parts[0]))["state"],
                         "waiting_model")
        self.assertEqual(workflow.accept_model_part(self.store, request, 1, self.part_answer(parts[1]))["state"],
                         "completed")

    def test_long_turn_is_split_at_sentence_end_without_losing_or_repeating_words(self):
        texts = [f"с{i}" for i in range(1200)]
        texts[449] = "конец."          # 450-е слово кончает предложение
        texts[899] = "да?»"            # 900-е — тоже, с закрывающей кавычкой
        request, _ = self.ready_request(segments=[
            self.segment("Спикер 1", ["привет"] * 10),
            self.segment("Спикер 2", texts, start=10),
            self.segment("Спикер 1", ["пока"] * 5, start=1210)])
        task = workflow.prepare_model_task(self.store, request)
        parts = self.all_parts(request)
        # короткая реплика; длинная реплика тремя кусками; её хвост вместе со следующей репликой
        self.assertEqual([len(part["task"]["words"]) for part in parts], [10, 450, 450, 305])
        self.assertEqual(parts[1]["task"]["words"][-1]["text"], "конец.")
        self.assertEqual(parts[2]["task"]["words"][-1]["text"], "да?»")
        self.assertEqual([[turn["turn"] for turn in part["task"]["turns"]] for part in parts],
                         [[1], [2], [2], [2, 3]])
        ids = [word["word_id"] for part in parts for word in part["task"]["words"]]
        self.assertEqual(ids, [word["word_id"] for word in task["words"]])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 1215)

    def test_long_turn_without_sentence_end_is_split_at_the_limit(self):
        request, _ = self.ready_request(segments=[self.segment("Спикер 1", ["слово"] * 1200)])
        task = workflow.prepare_model_task(self.store, request)
        parts = self.all_parts(request)
        self.assertEqual([len(part["task"]["words"]) for part in parts], [500, 500, 200])
        self.assertEqual([word for part in parts for word in part["task"]["words"]], task["words"])

    def test_part_bounds_cover_every_word_exactly_once(self):
        cases = {
            "пусто": [],
            "ровно предел": [500],
            "мелкие реплики": [7] * 200,
            "длинная среди коротких": [3, 1700, 499, 2, 501],
        }
        for name, sizes in cases.items():
            with self.subTest(name=name):
                words, turns = [], []
                for number, size in enumerate(sizes, 1):
                    turns.append({"turn": number, "speaker": "Участник 1"})
                    words += [{"word_id": f"{number}-{i}", "text": "конец." if i % 97 == 96 else "слово",
                               "turn": number} for i in range(size)]
                task = workflow.model_tasks.build("recording", 1, words, turns=turns)
                bounds = workflow._part_bounds(task)
                self.assertEqual([start for start, _ in bounds], [0] + [end for _, end in bounds][:-1] if bounds else [])
                self.assertEqual(bounds[-1][1] if bounds else 0, len(words))
                self.assertTrue(all(0 < end - start <= workflow.MODEL_PART_WORDS for start, end in bounds))
                for start, end in bounds:      # блок кончается на границе реплики или посреди длинной
                    if end < len(words) and words[end]["turn"] == words[end - 1]["turn"]:
                        self.assertGreater(sizes[words[end]["turn"] - 1], workflow.MODEL_PART_WORDS)
                    workflow._part_task(task, (start, end))

    def test_saved_version_1_task_is_finished_and_replayed_unchanged(self):
        # 300 + 201 слово: версия 2 разрезала бы по границе реплики (300 и 201), версия 1 — 500 и 1.
        request, recording = self.ready_request(segments=[
            self.segment("Анна", ["слово"] * 300), self.segment("Спикер 2", ["слово"] * 201, start=300)])
        fresh = workflow.prepare_model_task(self.store, request)
        legacy = workflow.model_tasks.build(
            fresh["recording_id"], fresh["source_revision"],
            [{key: value for key, value in word.items() if key != "turn"} for word in fresh["words"]],
            fresh["goal"], fresh["review_revision"])
        self.assertNotIn("version", legacy)
        with self.store._transaction():        # так задание лежало в базе до версии 2
            self.store.db.execute("UPDATE workflow_requests SET model_task_json=? WHERE request_id=?",
                                  (storage._json(legacy), request))
        self.assertEqual(workflow.prepare_model_task(self.store, request), legacy)
        parts = self.all_parts(request)
        self.assertEqual([len(part["task"]["words"]) for part in parts], [500, 1])
        self.assertTrue(all("version" not in part["task"] and "turns" not in part["task"]
                            and "turn" not in part["task"]["words"][0] for part in parts))
        self.assertEqual(workflow.get(self.store, request)["model_task"], legacy)
        self.assertEqual(workflow.accept_model_part(self.store, request, 0, self.part_answer(parts[0]))["state"],
                         "waiting_model")
        self.store.close()
        self.store = storage.open_store(self.root / "state.sqlite3")
        done = workflow.accept_model_part(self.store, request, 1, self.part_answer(parts[1]))
        self.assertEqual(done["state"], "completed")
        # повтор блока и целого ответа после завершения
        self.assertEqual(workflow.accept_model_part(self.store, request, 1, self.part_answer(parts[1]))["state"],
                         "completed")
        whole = {"source_revision": legacy["source_revision"],
                 "sections": [self.part_answer(part)["sections"][0] for part in parts]}
        self.assertEqual(workflow.accept_model_result(self.store, request, whole), done["result"])
        self.assertEqual(workflow.get(self.store, request)["model_task"], legacy)
        payload, md = self.conspect_files(recording)
        # файл версии 1 — прежнего вида: без расшифровки меток и времени
        self.assertTrue(md.startswith("# Конспект\n\nСоздан моделью; проверка человеком не подтверждена.\n\n"
                                      + conspect.INFO_MARK + "\nСобытие: "))
        self.assertIn("Составлен: моделью OpenAI через OpenCode\n", md)
        self.assertTrue(md.endswith("\n\nИтог блока.\n\nИтог блока.\n"))
        self.assertNotIn("Участники", md)
        self.assertNotIn("speaker_labels", payload)
        self.assertEqual(len(self.store.list_conspects(recording)), 1)
        # новое поколение задания для того же запроса (явное «начать заново») — уже текущей версии 3
        self.assertEqual(workflow.prepare_model_task(self.store, request), legacy)
        self.assertEqual(workflow.prepare_model_task(self.store, request, restart=True)["version"], 3)

    def test_conspect_file_explains_labels_and_gives_section_time(self):
        request, recording = self.ready_request(
            speaker_names={"Спикер 3": "Борис Синтетический"},
            segments=[self.segment("Анна Синтетическая", ["обсудим", "план."], start=75),
                      self.segment("Спикер 2", ["согласен"], start=3725),
                      self.segment("Спикер 3", ["хорошо"], start=3730),
                      self.segment(None, ["шум"], start=3735)])
        task = workflow.prepare_model_task(self.store, request)
        ids = [word["word_id"] for word in task["words"]]
        answer = {"source_revision": task["source_revision"], "sections": [
            {"text": "Участник 1 предложил обсудить план.", "evidence": ids[:2]},
            {"text": "Участник 2 и Участник 3 согласились.", "evidence": ids[2:]}]}
        workflow.accept_model_result(self.store, request, answer)
        payload, md = self.conspect_files(recording)
        self.assertTrue(md.startswith("# Конспект\n\nСоздан моделью; проверка человеком не подтверждена.\n\n"))
        self.assertIn("- Участник 1 — Анна Синтетическая\n", md)
        self.assertIn("- Участник 2 — имя не указано (в стенограмме: Спикер 2)\n", md)
        self.assertIn("- Участник 3 — Борис Синтетический\n", md)
        self.assertIn("- Говорящий не определён — ", md)
        self.assertIn("[00:01:15]\nУчастник 1 предложил обсудить план.", md)
        self.assertIn("[01:02:05]\nУчастник 2 и Участник 3 согласились.", md)
        self.assertLess(md.index("Участник 1 — Анна"), md.index("[00:01:15]"))
        # метка привязана к словам (первое слово каждой её реплики), имён в привязке нет
        self.assertEqual(payload["speaker_labels"], [
            {"label": "Участник 1", "word_ids": [ids[0]]}, {"label": "Участник 2", "word_ids": [ids[2]]},
            {"label": "Участник 3", "word_ids": [ids[3]]},
            {"label": "Говорящий не определён", "word_ids": [ids[4]]}])
        self.assertEqual(payload["evidence"], answer["sections"])       # ответ модели сохранён как есть
        self.assertIn("[00:01:15]", payload["template"])                # время переживает пересборку
        self.assertNotIn("Анна", payload["template"])
        # имена и время остаются на этом компьютере: в сохранённом задании их нет
        stored_task = workflow.get(self.store, request)["model_task"]
        stored = json.dumps(stored_task, ensure_ascii=False)
        for private in ("Анна", "Борис", "Синтетическ", "Спикер"):
            self.assertNotIn(private, stored)
        self.assertTrue(all(set(word) == {"word_id", "text", "uncertain", "turn"} for word in stored_task["words"]))
        self.assertTrue(all(set(turn) == {"turn", "speaker"} for turn in stored_task["turns"]))
        self.assertEqual(workflow.accept_model_result(self.store, request, answer)["sections"], answer["sections"])
        self.assertEqual(self.conspect_files(recording)[1], md)

    def test_label_legend_survives_rebuild_and_shows_current_names_after_rename(self):
        import paths
        import transcript
        # основной сценарий: свёртка создана сразу после распознавания, говорящие ещё безымянные
        request, recording = self.ready_request(segments=[
            self.segment("Спикер 1", ["обсудим", "план."], start=75),
            self.segment("Спикер 2", ["согласен"], start=80),
            self.segment("Спикер 1", ["тогда", "начнём"], start=85)])
        task = workflow.prepare_model_task(self.store, request)
        ids = [word["word_id"] for word in task["words"]]
        workflow.accept_model_result(self.store, request, {"source_revision": task["source_revision"], "sections": [
            {"text": "Участник 1 предложил план, Участник 2 согласился.", "evidence": ids}]})
        payload, md = self.conspect_files(recording)
        self.assertIn("- Участник 1 — имя не указано (в стенограмме: Спикер 1)\n", md)
        self.assertEqual(payload["speaker_labels"][0], {"label": "Участник 1", "word_ids": [ids[0], ids[3]]})
        binding = recordings.document_binding(self.store, recording)
        json_path = binding["root"] / binding["paths"]["json"]
        md_path = binding["root"] / binding["paths"]["md"]
        with mock.patch.object(paths, "STATE_DB", self.root / "state.sqlite3"):
            renamed, unknown = transcript.rename(json_path, md_path, {"Спикер 1": "Вера Синтетическая"})
        self.assertEqual(unknown, [])
        self.assertNotIn("error", renamed["_conspects"])
        saved = self.store.list_conspects(recording)[0]
        self.assertEqual(saved["current_revision"], 2)          # файл пересобран после переименования
        payload, md = self.conspect_files(recording)
        self.assertTrue(md.startswith("# Конспект\n\nСоздан моделью; проверка человеком не подтверждена.\n\n"))
        self.assertIn("- Участник 1 — Вера Синтетическая\n", md)
        self.assertIn("- Участник 2 — имя не указано (в стенограмме: Спикер 2)\n", md)
        self.assertNotIn("Спикер 1", md)
        self.assertIn("[00:01:15]\nУчастник 1 предложил план, Участник 2 согласился.", md)
        self.assertEqual(payload["speaker_labels"][0], {"label": "Участник 1", "word_ids": [ids[0], ids[3]]})
        self.assertNotIn("Вера", json.dumps(payload["speaker_labels"], ensure_ascii=False))

    def test_label_legend_follows_words_not_names(self):
        import conspect
        words = lambda *idents: [{"word_id": ident, "w": "слово", "s": 1.0} for ident in idents]
        data = {"recording_id": "r", "source_revision": 2, "speaker_names": {"Спикер 2": "Борис Синтетический"},
                "segments": [{"speaker": "Анна Синтетическая", "words": words("a1", "a2")},
                             {"speaker": "Спикер 2", "words": words("b1")},
                             {"speaker": "Анна Синтетическая", "words": words("a3")},
                             {"speaker": "Спикер 7", "words": words("c1")},
                             {"speaker": None, "words": words("d1")}]}
        obj = {"template": "[00:00:01]\nТекст раздела.", "quotes": {}, "recording_id": "r", "analysis_revision": None,
               "provenance": "model", "speaker_labels": [
                   {"label": "Участник 1", "word_ids": ["a1", "b1", "a3"]},     # часть реплик переотнесена
                   {"label": "Участник 2", "word_ids": ["c1"]},
                   {"label": "Участник 3", "word_ids": ["удалено"]},
                   {"label": "Говорящий не определён", "word_ids": ["d1"]}]}
        md, _ = conspect.render(obj, data, update_last=False)
        self.assertEqual(md, (
            "# Конспект\n\nСоздан моделью; проверка человеком не подтверждена.\n\n"
            "<!-- rus-transcriber: сведения шапки, ставит программа -->\n"
            "Событие: не указано\nДата записи: не установлена\nСоставлен: моделью OpenAI через OpenCode\n"
            "Стенограмма: версия 2\nМеста разбора: неизвестно (данные разбора не переданы)\n\n"
            "Участники (условные метки; имена — по текущей стенограмме):\n\n"
            "- Участник 1 — Анна Синтетическая; часть реплик в текущей стенограмме: Борис Синтетический\n"
            "- Участник 2 — имя не указано (в стенограмме: Спикер 7)\n"
            "- Участник 3 — реплики не найдены в текущей стенограмме\n"
            "- Говорящий не определён — говорящий в стенограмме не указан\n\n"
            "Время в квадратных скобках — начало раздела в записи (часы:минуты:секунды).\n\n"
            "[00:00:01]\nТекст раздела."))
        # после подтверждения человеком на текущей ревизии строка об отсутствии проверки не пишется
        confirmed, _ = conspect.render({**obj, "analysis_revision": 2}, data, update_last=False)
        self.assertTrue(confirmed.startswith("# Конспект\n\nСоздан моделью.\n\n"))
        # конспект без привязки меток: без списка участников и времени, но со сведениями шапки (7.6а)
        plain = {key: value for key, value in obj.items() if key != "speaker_labels"}
        plain_md = conspect.render(plain, data, update_last=False)[0]
        self.assertTrue(plain_md.endswith("\n\n[00:00:01]\nТекст раздела."))
        self.assertNotIn("Участники", plain_md)
        self.assertIn("Составлен: моделью OpenAI через OpenCode\n", plain_md)

    def test_section_time_spans_is_pure_and_tolerates_missing_words(self):
        transcript = {"segments": [
            {"segment_id": "s1", "speaker": "Спикер 1", "words": [
                {"word_id": "a", "w": "раз", "s": 10.0, "e": 10.4}, {"word_id": "b", "w": "два", "s": 11.0, "e": 11.5}]},
            {"segment_id": "s2", "speaker": "Спикер 2", "words": [
                {"word_id": "c", "w": "три", "s": 20.0, "e": 20.2}, {"word_id": "d", "w": "без", "s": 30.0},
                {"word_id": "e", "w": "времени"}]}]}
        sections = [{"text": "Первый.", "evidence": ["b", "a"]}, {"text": "Второй.", "evidence": ["c", "d", "нет"]},
                    {"text": "Третий.", "evidence": ["e", "нет"]}]
        before = json.dumps([transcript, sections], ensure_ascii=False, sort_keys=True)
        spans = workflow.section_time_spans(sections, transcript)
        self.assertEqual(spans, [{"t1": 10.0, "t2": 11.5}, {"t1": 20.0, "t2": 30.0}, {"t1": None, "t2": None}])
        self.assertEqual(json.dumps([transcript, sections], ensure_ascii=False, sort_keys=True), before)
        self.assertEqual(workflow.section_time_spans([], transcript), [])

    def test_section_time_spans_matches_saved_conspect(self):
        request, recording = self.ready_request(segments=[
            self.segment("Спикер 1", ["раз", "два"], start=5), self.segment("Спикер 2", ["три"], start=40)])
        task = workflow.prepare_model_task(self.store, request)
        ids = [word["word_id"] for word in task["words"]]
        workflow.accept_model_result(self.store, request, {"source_revision": task["source_revision"], "sections": [
            {"text": "Начало.", "evidence": ids[:2]}, {"text": "Конец.", "evidence": ids[2:]}]})
        saved = self.store.list_conspects(recording)[0]["payload"]
        snapshot = self.store.get_current(recording)["transcript"]["payload"]
        self.assertEqual(workflow.section_time_spans(saved["evidence"], snapshot),
                         [{"t1": 5, "t2": 6.5}, {"t1": 40, "t2": 40.5}])

    def test_model_task_pages_carry_only_their_turns(self):
        request, _ = self.ready_request(segments=[
            self.segment("Спикер 1", ["раз", "два"]), self.segment("Спикер 2", ["три"], start=2)])
        dbpath = self.root / "state.sqlite3"
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)):
            first = core.workflow_model_task(request, 0, 2)
            second = core.workflow_model_task(request, 2, 2)
        self.assertEqual((first["version"], first["total_words"], first["next_offset"]), (3, 3, 2))
        self.assertEqual(first["turns"], [{"turn": 1, "speaker": "Участник 1"}])
        self.assertEqual(second["turns"], [{"turn": 2, "speaker": "Участник 2"}])
        self.assertIsNone(second["next_offset"])

    def test_cli_result_file_is_read(self):
        import rt
        result_file = self.root / "response.json"
        result_file.write_text(json.dumps({"example": True}), encoding="utf-8")
        arguments = rt.build().parse_args(["workflow-result", "request", str(result_file)])
        with mock.patch("core.workflow_model_result", return_value={"ok": True}) as action, \
             mock.patch("rt.out"):
            rt.run(arguments)
        action.assert_called_once_with("request", {"example": True})

    def test_confirmation_off_cannot_mark_human_verified(self):
        import server
        action = mock.Mock(return_value={"ok": True})
        with mock.patch.object(server.confirm, "require", return_value="off"):
            result = asyncio.run(server._gated("test", {}, lambda: ["test"], action, require_human=True))
        action.assert_not_called()
        self.assertIn("не выполнено", result)
        with mock.patch.object(server.confirm, "require", return_value="dialog"):
            asyncio.run(server._gated("test", {}, lambda: ["test"], action, require_human=True))
        action.assert_called_once()

    def test_questions_are_bounded_and_do_not_decide_for_human(self):
        request = workflow.create(self.store, self.source, {"question_budget": 1})
        workflow.claim_start(self.store, request)
        recording = self.make_recording(review=[{"id": "r1", "status": "open", "text": "первое"},
                                                 {"id": "r2", "status": "open", "text": "второе"}])
        self.attach_recording(request, recording)
        workflow.observe(self.store, request, {"state": "done", "recording_id": recording})
        dbpath = self.root / "state.sqlite3"
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)):
            one = core.workflow_questions(request)
            again = core.workflow_questions(request)
        self.assertEqual([q["id"] for q in one["questions"]], ["r1"])
        self.assertEqual(one, again)
        self.assertEqual(one["remaining"], 1)
        self.assertNotIn("answer", one["questions"][0])
        current = self.store.get_current(recording)
        review = current["review"]["payload"]
        review["review"][0]["status"] = "resolved"
        binding = recordings.document_binding(self.store, recording)
        self.store.commit_change_with_document(
            recording, {"source_revision": current["transcript"]["revision"],
                        "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            review=review, root_id=binding["root_id"], generated_at="2026-01-01T00:00:01Z")
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)):
            after = core.workflow_questions(request)
        self.assertEqual(after["questions"], [])
        self.assertEqual(after["remaining"], 1)

    def test_one_start_through_worker_client_and_status(self):
        dbpath = self.root / "state.sqlite3"
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(dbpath)), \
             mock.patch("settings.recording_backend", return_value="sqlite"), \
             mock.patch("core.launch", side_effect=lambda *args, **kwargs: kwargs["job_id"]) as launch:
            started = core.start_workflow(self.source, {"require_conspect": True})
            initial = launch.call_args.kwargs["initial_state"]
            self.assertEqual(initial["workflow_request_id"], started["request_id"])
            layout = recordings.document_layout(self.source, "")
            layout["root"].mkdir()
            root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
            status = mock.Mock()
            status.data = initial
            target = {"root_id": root_id, "paths": paths, "context": initial["recording_context"]}
            data = {"file": str(self.source), "duration": "00:00:01", "speakers": ["Спикер 1"],
                    "quality": {}, "review": [], "review_done": False,
                    "segments": [{"start": 0, "end": 1, "speaker": "Спикер 1",
                                  "words": [{"w": "слово", "s": 0, "e": 1}]}]}
            saved = worker._save_worker_result(data, self.source, status, target, None, None)
            with mock.patch("core.job_status", return_value={"state": "done", **saved}):
                before = core.workflow_status(started["request_id"])
            self.assertEqual(before["readiness"]["execution"], "waiting_model")
            part = core.workflow_model_part(started["request_id"], 0)
            answer = {"source_revision": part["task"]["source_revision"], "sections": [{
                "text": "Итог.", "evidence": [part["task"]["words"][0]["word_id"]]}]}
            self.assertEqual(core.workflow_model_part_result(started["request_id"], 0, answer)["state"], "completed")
            after = core.workflow_status(started["request_id"])
            self.assertEqual(after["readiness"]["execution"], "completed")
            self.assertEqual(after["readiness"]["human_verification"], "none")

    def test_worker_commit_moves_asr_run_to_processed_atomically(self):
        request = workflow.create(self.store, self.source, {"require_conspect": True})
        workflow.claim_start(self.store, request)
        before = workflow.get(self.store, request)
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        status = mock.Mock()
        status.data = {"workflow_request_id": request}
        target = {"root_id": root_id, "paths": paths, "context": before["context"]}
        data = {"file": str(self.source), "duration": "00:00:01", "speakers": ["Спикер 1"],
                "quality": {}, "review": [], "review_done": False,
                "segments": [{"start": 0, "end": 1, "speaker": "Спикер 1",
                              "words": [{"w": "слово", "s": 0, "e": 1}]}]}
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(self.root / "state.sqlite3")):
            worker._save_worker_result(data, self.source, status, target, None, None)
        saved = workflow.get(self.store, request)
        self.assertEqual(saved["state"], "processed")
        self.assertEqual(saved["runs"][0]["state"], "completed")


if __name__ == "__main__":
    unittest.main()
