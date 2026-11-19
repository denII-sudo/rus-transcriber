import asyncio
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import core
import recordings
import storage
import workflow
import worker


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="p3-workflow-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.store = storage.initialize(self.root / "state.sqlite3", initial_display_name="Тест")

    def tearDown(self):
        self.store.close()

    def make_recording(self, count=1, review=None):
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        ident = str(uuid.uuid4())
        data = {"file": str(self.source), "duration": "00:00:01", "speakers": ["Спикер 1"],
                "quality": {}, "review": review or [], "review_done": False,
                "segments": [{"start": 0, "end": count, "speaker": "Спикер 1",
                              "words": [{"w": "слово", "s": i, "e": i + 1} for i in range(count)]}]}
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
        workflow.prepare_model_task(self.store, request)
        workflow.accept_model_result(self.store, request, answer)
        self.assertEqual(self.store.list_conspects(recording)[0]["revision"], 2)
        runs = workflow.get(self.store, request)["runs"]
        self.assertEqual(len([run for run in runs if run["kind"] == "model"]), 2)

    def test_unlinked_quote_does_not_commit(self):
        request = workflow.create(self.store, self.source, {"require_conspect": True})
        workflow.claim_start(self.store, request)
        recording = self.make_recording()
        self.attach_recording(request, recording)
        workflow.observe(self.store, request, {"state": "done", "recording_id": recording})
        task = workflow.prepare_model_task(self.store, request)
        answer = {"source_revision": task["source_revision"],
                  "sections": [{"text": "«совершенно другое»", "evidence": [task["words"][0]["word_id"]]}]}
        with self.assertRaises(ValueError):
            workflow.accept_model_result(self.store, request, answer)
        self.assertEqual(workflow.get(self.store, request)["state"], "waiting_model")
        self.assertEqual(self.store.list_conspects(recording), [])
        with self.assertRaises(ValueError):
            workflow.accept_model_result(self.store, request, answer)
        self.assertEqual(workflow.get(self.store, request)["state"], "failed")
        self.assertEqual(workflow.get(self.store, request)["model_attempts"], 2)

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
        self.store.commit_change_with_document(
            recording, {"source_revision": current["transcript"]["revision"],
                        "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            review=current["review"]["payload"], root_id=binding["root_id"],
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
        self.store.commit_change_with_document(
            recording, {"source_revision": current["transcript"]["revision"],
                        "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            review=current["review"]["payload"], root_id=binding["root_id"],
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
        import migrate
        with mock.patch.object(migrate, "run"):
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
