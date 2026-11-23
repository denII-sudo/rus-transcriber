"""Модельный шаг свёртки: нет тупиков и ложного «готово» (РАР-7, этап 7.1). Синтетика, sqlite."""
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import llm_provider
import recordings
import storage
import workflow


class ModelStepNoDeadEndsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="model-step-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.store = storage.initialize(self.root / "state.sqlite3", initial_display_name="Тест")
        self.addCleanup(self.store.close)
        self.calls = 0

    def prepare_request(self, count=3):
        request = workflow.create(self.store, self.source, {"require_conspect": True, "allow_cloud_model": True})
        workflow.claim_start(self.store, request)
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        recording_id = str(uuid.uuid4())
        data = {"file": str(self.source), "duration": "00:00:01", "speakers": ["Спикер 1"],
                "quality": {}, "review": [], "review_done": False,
                "segments": [{"start": 0, "end": count, "speaker": "Спикер 1",
                              "words": [{"w": f"w{i}", "s": i, "e": i + 1} for i in range(count)]}]}
        recordings.commit_worker_result(
            self.store, data, self.source, root_id, recordings.operation_context(self.store, "model"),
            recording_id=recording_id, operation_id=str(uuid.uuid4()),
            generated_at="2026-01-01T00:00:00Z", document_paths=paths)
        recordings.materialize_recording(self.store, recording_id)
        with self.store._transaction():
            self.store.db.execute("UPDATE workflow_requests SET recording_id=?, state='processed' WHERE request_id=?",
                                  (recording_id, request))
        return request

    def client(self, before_answer=None):
        """Подмена сети: отвечает одним разделом на весь блок; before_answer — что случается во время вызова."""
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

    def change_review(self, request):
        recording = workflow.get(self.store, request)["recording_id"]
        current = self.store.get_current(recording)
        binding = recordings.document_binding(self.store, recording)
        self.store.commit_change_with_document(
            recording, {"source_revision": current["transcript"]["revision"],
                        "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            review=current["review"]["payload"], root_id=binding["root_id"],
            generated_at="2026-01-01T00:00:01Z")

    def execute(self, request, before_answer=None, **kwargs):
        with mock.patch("llm_provider.OpenCodeTextClient", self.client(before_answer)):
            return workflow.execute_model(self.store, request, **kwargs)

    def conspect_md(self, request):
        recording = workflow.get(self.store, request)["recording_id"]
        row = self.store.db.execute(
            "SELECT r.path, f.relative_path FROM conspect_file_locations f JOIN allowed_roots r USING(root_id) "
            "JOIN conspects c USING(conspect_id) WHERE c.recording_id=? AND f.kind='md'", (recording,)).fetchone()
        return Path(row["path"]) / row["relative_path"]

    # 1. Ревизия изменилась во время вызова: продолжить нельзя, но можно начать заново.
    def test_review_changed_during_call_can_be_restarted(self):
        request = self.prepare_request()
        with self.assertRaises(llm_provider.ProviderError) as error:
            self.execute(request, before_answer=lambda: self.change_review(request))
        self.assertEqual(error.exception.code, "revision_changed")
        self.assertIn("restart", str(error.exception))
        self.assertEqual(workflow.get(self.store, request)["state"], "failed")
        with self.assertRaisesRegex(storage.Conflict, "restart"):
            self.execute(request, retry_failed=True)
        calls = self.calls
        result = self.execute(request, restart=True)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.calls, calls + 1)
        self.assertEqual(len(self.store.list_conspects(workflow.get(self.store, request)["recording_id"])), 1)

    def test_failed_step_can_be_restarted_through_the_agent_task(self):
        request = self.prepare_request()
        with self.assertRaises(llm_provider.ProviderError):
            self.execute(request, before_answer=lambda: self.change_review(request))
        task = workflow.prepare_model_task(self.store, request, restart=True)
        self.assertEqual(workflow.get(self.store, request)["state"], "waiting_model")
        answer = {"source_revision": task["source_revision"], "sections": [{
            "text": "Итог.", "evidence": [word["word_id"] for word in task["words"]]}]}
        workflow.accept_model_result(self.store, request, answer)
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")

    # 2. Предел размера: отказ до вызова, состояние не меняется, выходы названы.
    def test_limit_refuses_before_any_change_and_names_exits(self):
        request = self.prepare_request(count=6)
        with mock.patch("llm_provider.MAX_WORDS", 5):
            for _ in range(2):
                with self.assertRaises(llm_provider.ProviderError) as error:
                    self.execute(request)
                self.assertEqual(error.exception.code, "limit")
                self.assertEqual(workflow.get(self.store, request)["state"], "processed")
        self.assertEqual(self.calls, 0)
        self.assertIn("workflow_model_part", str(error.exception))
        self.assertIn("workflow_change_policy", str(error.exception))
        part = workflow.model_part(self.store, request, 0)          # выход через ИИ-агента открыт
        self.assertEqual(len(part["task"]["words"]), 6)

    # 3. Все блоки сохранены, итога нет: повтор собирает итог без вызова облака.
    def test_saved_blocks_without_result_are_assembled_without_cloud(self):
        request = self.prepare_request(count=501)
        task = workflow.prepare_model_task(self.store, request)
        for index in range(2):
            part = workflow.model_part(self.store, request, index)
            answer = {"source_revision": task["source_revision"], "sections": [{
                "text": "Блок.", "evidence": [word["word_id"] for word in part["task"]["words"]]}]}
            if index == 1:
                with mock.patch("workflow.accept_model_result", side_effect=OSError("synthetic")):
                    with self.assertRaises(OSError):
                        workflow.accept_model_part(self.store, request, index, answer)
            else:
                workflow.accept_model_part(self.store, request, index, answer)
        self.assertEqual(workflow.get(self.store, request)["state"], "waiting_model")
        result = self.execute(request)
        self.assertEqual((result["state"], result["completed_parts"], result["part_count"]), ("completed", 2, 2))
        self.assertEqual(self.calls, 0)
        self.assertEqual(len(self.store.list_conspects(workflow.get(self.store, request)["recording_id"])), 1)

    def test_completed_parts_counts_saved_blocks_on_failure(self):
        request = self.prepare_request(count=501)
        task = workflow.prepare_model_task(self.store, request)

        def fail_second():
            if self.calls == 2:
                raise llm_provider.ProviderError("offline", "синтетический обрыв")
        with self.assertRaises(llm_provider.ProviderError):
            self.execute(request, before_answer=fail_second)
        status = workflow.get(self.store, request)
        self.assertEqual(status["state"], "failed")
        task_hash = storage._hash({"task": task, "run_id": status["model_run_id"]})
        self.assertEqual(workflow.saved_part_count(self.store, request_id=request, task_hash=task_hash), 1)

    # 4. Чтение задания у готовой свёртки ничего не меняет.
    def test_reading_task_of_completed_conspect_changes_nothing(self):
        request = self.prepare_request()
        self.execute(request)
        before = workflow.get(self.store, request)
        task = workflow.prepare_model_task(self.store, request)
        after = workflow.get(self.store, request)
        self.assertEqual(task, before["model_task"])
        self.assertEqual((after["state"], after["model_result"], after["model_run_id"]),
                         ("completed", before["model_result"], before["model_run_id"]))

    # 5. Файл свёртки изменён вне программы — отказ до вызова облака.
    def test_changed_conspect_file_refuses_before_cloud(self):
        request = self.prepare_request()
        self.execute(request)
        path = self.conspect_md(request)
        path.write_text("моя правка\n", encoding="utf-8")
        calls = self.calls
        with self.assertRaises(llm_provider.ProviderError) as error:
            self.execute(request, restart=True)
        self.assertEqual(error.exception.code, "conspect_file_changed")
        self.assertIn(path.name, str(error.exception))
        self.assertEqual(self.calls, calls)
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")
        self.assertEqual(path.read_text(encoding="utf-8"), "моя правка\n")

    def test_files_not_written_is_not_reported_as_done(self):
        request = self.prepare_request()
        with mock.patch("recordings.materialize_conspects", return_value=[]):
            with self.assertRaises(llm_provider.ProviderError) as error:
                self.execute(request)
        self.assertEqual(error.exception.code, "files_not_written")
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")
        calls = self.calls
        with mock.patch("recordings.materialize_conspects", return_value=[]):
            with self.assertRaises(llm_provider.ProviderError):
                self.execute(request)
        result = self.execute(request)           # повтор доводит до итога: файлы записаны, облака нет
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.calls, calls)
        self.assertTrue(self.conspect_md(request).is_file())

    def test_restart_and_retry_together_are_refused(self):
        request = self.prepare_request()
        with self.assertRaises(ValueError):
            self.execute(request, retry_failed=True, restart=True)
        self.assertEqual(self.calls, 0)


if __name__ == "__main__":
    unittest.main()
