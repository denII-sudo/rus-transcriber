"""Фиксатор переходов состояния сохранённого процесса П3, которые делают core.py и worker.py.

Проверяется точное состояние трёх таблиц (запрос, прогоны, шаги) после запуска, сбоя
запуска и сохранения результата распознавания, а также отказы сохранить результат.
Данные синтетические, база — во временной папке.
"""
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


def table_states(store, request_id):
    request = store.db.execute("SELECT state, job_id IS NOT NULL AS has_job, recording_id "
                               "FROM workflow_requests WHERE request_id=?", (request_id,)).fetchone()
    runs = [(row["kind"], row["state"]) for row in store.db.execute(
        "SELECT kind, state FROM workflow_runs WHERE request_id=? ORDER BY created_at, kind", (request_id,))]
    steps = [(row["kind"], row["state"], storage._loads(row["result_json"]) if row["result_json"] else None)
             for row in store.db.execute(
                 "SELECT s.kind, s.state, s.result_json FROM workflow_steps s JOIN workflow_runs r USING(run_id) "
                 "WHERE r.request_id=? ORDER BY s.created_at, s.kind", (request_id,))]
    return {"request": (request["state"], bool(request["has_job"]), request["recording_id"]),
            "runs": runs, "steps": steps}


class WorkflowTransitionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="workflow-transitions-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.dbpath = self.root / "state.sqlite3"
        self.store = storage.initialize(self.dbpath, initial_display_name="Тест")
        self.addCleanup(lambda: self.store.close())

    def open_store(self):
        return mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(self.dbpath))

    def worker_target(self, request_id):
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir(exist_ok=True)
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        return {"root_id": root_id, "paths": paths, "context": workflow.get(self.store, request_id)["context"]}

    def save_result(self, request_id, target, source=None):
        status = mock.Mock()
        status.data = {"workflow_request_id": request_id}
        data = {"file": str(self.source), "duration": "00:00:01", "speakers": ["Спикер 1"],
                "quality": {}, "review": [], "review_done": False,
                "segments": [{"start": 0, "end": 1, "speaker": "Спикер 1",
                              "words": [{"w": "слово", "s": 0, "e": 1}]}]}
        with self.open_store():
            return worker._save_worker_result(data, source or self.source, status, target, None, None)

    def test_claim_start_moves_everything_to_running(self):
        request = workflow.create(self.store, self.source)
        self.assertEqual(table_states(self.store, request),
                         {"request": ("queued", False, None), "runs": [("asr", "queued")],
                          "steps": [("asr", "queued", None)]})
        workflow.claim_start(self.store, request)
        self.assertEqual(table_states(self.store, request),
                         {"request": ("starting", True, None), "runs": [("asr", "running")],
                          "steps": [("asr", "running", None)]})

    def test_launch_failure_marks_running_asr_interrupted(self):
        with self.open_store(), mock.patch("settings.recording_backend", return_value="sqlite"), \
                mock.patch.object(core, "start_transcription", side_effect=OSError("synthetic spawn failure")):
            started = core.start_workflow(self.source)
        self.assertEqual(started["state"], "interrupted")
        self.assertEqual(started["note"], "Исход запуска неизвестен; автоматический повтор отключён.")
        self.assertEqual(table_states(self.store, started["request_id"]),
                         {"request": ("interrupted", True, None), "runs": [("asr", "interrupted")],
                          "steps": [("asr", "interrupted", None)]})

    def test_worker_commit_links_recording_and_completes_asr(self):
        request = workflow.create(self.store, self.source)
        workflow.claim_start(self.store, request)
        before = workflow.get(self.store, request)["updated_at"]
        result = self.save_result(request, self.worker_target(request))
        recording = result["recording_id"]
        self.assertEqual(recording, str(uuid.uuid5(uuid.UUID(request), "worker-recording")))
        self.assertEqual(table_states(self.store, request),
                         {"request": ("processed", True, recording), "runs": [("asr", "completed")],
                          "steps": [("asr", "completed", {"recording_id": recording})]})
        self.assertGreaterEqual(workflow.get(self.store, request)["updated_at"], before)

    def test_worker_commit_from_interrupted_request_is_accepted(self):
        request = workflow.create(self.store, self.source)
        workflow.claim_start(self.store, request)
        with self.store._transaction():
            self.store.db.execute("UPDATE workflow_requests SET state='interrupted' WHERE request_id=?", (request,))
        result = self.save_result(request, self.worker_target(request))
        self.assertEqual(table_states(self.store, request)["request"], ("processed", True, result["recording_id"]))

    def test_worker_commit_is_refused_and_nothing_changes(self):
        cases = {
            "queued": lambda request, target: (None, target),
            "context": lambda request, target: (claim(request), {**target, "context": {**target["context"],
                                                                                     "actor_kind": "human"}}),
            "source": lambda request, target: (claim(request), target),
            "linked": lambda request, target: (claim(request) or link(request), target),
        }

        def claim(request):
            workflow.claim_start(self.store, request)

        def link(request):
            other = self.root / "linked-other.wav"
            other.write_bytes(b"synthetic")
            layout = recordings.document_layout(other, "")
            layout["root"].mkdir()
            root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
            ident = str(uuid.uuid4())
            recordings.commit_worker_result(
                self.store, {"file": str(other), "duration": "00:00:01", "speakers": [], "quality": {},
                             "review": [], "review_done": False, "segments": []},
                other, root_id, recordings.operation_context(self.store, "model"), recording_id=ident,
                operation_id=str(uuid.uuid4()), generated_at="2026-01-01T00:00:00Z", document_paths=paths)
            with self.store._transaction():
                self.store.db.execute("UPDATE workflow_requests SET recording_id=? WHERE request_id=?",
                                      (ident, request))

        for name, prepare in cases.items():
            with self.subTest(case=name):
                self.source = self.root / f"{name}.wav"
                self.source.write_bytes(b"synthetic")
                request = workflow.create(self.store, self.source)
                target = self.worker_target(request)
                _, target = prepare(request, target)
                before = table_states(self.store, request)
                count = self.store.db.execute("SELECT count(*) FROM recordings").fetchone()[0]
                other = self.root / "other.wav" if name == "source" else None
                if other:
                    other.write_bytes(b"synthetic")
                with self.assertRaisesRegex(RuntimeError, "workflow request no longer permits this worker result"):
                    self.save_result(request, target, source=other)
                self.assertEqual(table_states(self.store, request), before)
                self.assertEqual(self.store.db.execute("SELECT count(*) FROM recordings").fetchone()[0], count)

    def test_shared_reads_of_heads_and_document_state(self):
        """Общие чтения, которыми core.py и workflow.py заменили одинаковые запросы."""
        request = workflow.create(self.store, self.source)
        workflow.claim_start(self.store, request)
        target = self.worker_target(request)
        recording = self.save_result(request, target)["recording_id"]
        head = recordings.recording_head(self.store, recording)
        current = self.store.get_current(recording)
        self.assertEqual((head["source_revision"], head["review_revision"]),
                         (current["transcript"]["revision"], current["review"]["revision"]))
        self.assertIsNone(recordings.recording_head(self.store, str(uuid.uuid4())))
        binding = recordings.document_binding(self.store, recording)
        self.assertEqual(recordings.document_sync_state(self.store, recording, binding["document_revision"]),
                         "synced")
        with self.assertRaises(TypeError):          # как прежний row["state"] у отсутствующей строки
            recordings.document_sync_state(self.store, recording, binding["document_revision"] + 1)
        self.assertEqual(workflow.policy_head(self.store, request)["policy_revision"], 1)
        workflow.change_policy(self.store, request, {"goal": "Другая цель"}, 1)
        self.assertEqual(workflow.policy_head(self.store, request)["policy_revision"], 2)
        self.assertIsNone(workflow.policy_head(self.store, str(uuid.uuid4())))


if __name__ == "__main__":
    unittest.main()
