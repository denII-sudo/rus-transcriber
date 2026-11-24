"""П5: локальный экран и эксплуатация — история, возврат ревизии, конфликты, резерв, HTTP-экран."""
import gc
import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import dbtemplate
import documents
import model_tasks
import paths
import recordings
import storage
import workflow

import core
import rt
import workflow_page


def sandbox(case):
    """Временная папка теста; удаляется после него."""
    root = Path(tempfile.mkdtemp(prefix="p5-"))

    def remove():
        # соединения SQLite, оставшиеся у обработчиков страницы, могут держать файл базы в Windows:
        # сборщик мусора нужен только если папка не удалилась сразу
        shutil.rmtree(root, ignore_errors=True)
        if root.exists():
            gc.collect()
            shutil.rmtree(root, ignore_errors=True)
    case.addCleanup(remove)
    return root


class P5ServiceTests(unittest.TestCase):
    def setUp(self):
        # Договор версий 1–3 (ответ — перечень слов, блоки по 500 слов): так доделываются задания,
        # сохранённые до этапа 7.4. Договор версии 4 проверяет tests/test_sections_contract.py.
        patcher = mock.patch.object(model_tasks, "CURRENT_VERSION", 3)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.base = sandbox(self)
        self.store = dbtemplate.initialize(self.base / "state.sqlite3", "Тест")
        self.source = self.base / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.patcher = mock.patch.object(paths, "STATE_DB", self.base / "state.sqlite3")
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def tearDown(self):
        self.store.close()

    def make_recording(self, count=2):
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, doc_paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        ident = str(uuid.uuid4())
        words = [{"w": "слово", "s": i, "e": i + 1} for i in range(count)]
        data = {"file": str(self.source), "duration": "00:00:02", "speakers": ["Спикер 1"],
                "quality": {}, "review": [], "review_done": False,
                "segments": [{"start": 0, "end": count, "speaker": "Спикер 1", "words": words}]}
        recordings.commit_worker_result(
            self.store, data, self.source, root_id, recordings.operation_context(self.store, "model"),
            recording_id=ident, operation_id=str(uuid.uuid4()),
            generated_at="2026-01-01T00:00:00Z", document_paths=doc_paths)
        recordings.materialize_recording(self.store, ident)
        return ident, root_id, doc_paths

    def change_text(self, ident, root_id, text):
        current = self.store.get_current(ident)
        transcript = current["transcript"]["payload"]
        review = current["review"]["payload"]
        transcript["segments"][0]["words"][0]["w"] = text
        transcript["segments"][0]["text"] = text
        return self.store.commit_change_with_document(
            ident, {"source_revision": current["transcript"]["revision"],
                    "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            transcript=transcript, review=review, root_id=root_id,
            generated_at="2026-01-01T00:01:00Z")

    def test_history_and_restore_round_trip(self):
        ident, root_id, doc_paths = self.make_recording()
        self.change_text(ident, root_id, "исправлено")
        history = core.recording_revisions(ident, "transcript")
        self.assertEqual([item["revision"] for item in history], [1, 2])
        self.assertEqual(history[0]["actor_kind"], "model")
        self.assertEqual(history[1]["actor_kind"], "human")
        old = core.read_recording_revision(ident, "transcript", 1)
        self.assertEqual(old["payload"]["segments"][0]["words"][0]["w"], "слово")
        with self.assertRaises(core.CoreError):
            core.recording_revisions(ident, "непонятно")

        snapshot = core.prepare_restore_snapshot(ident, 1, 1)
        self.assertEqual(snapshot["expected_versions"], {"source_revision": 2, "review_revision": 2})
        result = core.restore_recording_revision(snapshot)
        self.assertTrue(result["materialized"])
        current = self.store.get_current(ident)
        self.assertEqual(current["transcript"]["revision"], 3)
        self.assertEqual(current["transcript"]["payload"]["segments"][0]["words"][0]["w"], "слово")
        self.assertEqual(current["review"]["revision"], 3)
        restored = self.store.get_revision(ident, "transcript", 3)
        self.assertEqual(restored["restored_from"], 1)
        binding = recordings.document_binding(self.store, ident)
        self.assertTrue(documents.verify_files(self.store, binding["root_id"], binding["root"])["consistent"])

    def test_history_exposes_compatible_restore_pairs(self):
        ident, root_id, _ = self.make_recording()
        current = self.store.get_current(ident)
        self.store.commit_change_with_document(
            ident, {"source_revision": 1, "review_revision": 1},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            review=current["review"]["payload"], root_id=root_id,
            generated_at="2026-01-01T00:01:00Z")
        self.change_text(ident, root_id, "исправлено")

        transcript = core.recording_revisions(ident, "transcript")
        review = core.recording_revisions(ident, "review")
        self.assertEqual([item["restore_review_revision"] for item in transcript], [2, 3])
        self.assertEqual([(item["source_revision"], item["revision"]) for item in review],
                         [(1, 1), (1, 2), (2, 3)])
        self.assertEqual(core.prepare_restore_snapshot(ident, 1, transcript[0]["restore_review_revision"])
                         ["source_revision"], 1)
        self.assertEqual(core.prepare_restore_snapshot(ident, 2, review[2]["revision"])
                         ["review_revision"], 3)

    def test_restore_rejects_stale_snapshot(self):
        ident, root_id, _ = self.make_recording()
        snapshot = core.prepare_restore_snapshot(ident, 1, 1)
        self.change_text(ident, root_id, "ещё правка")
        with self.assertRaises(core.CoreError):
            core.restore_recording_revision(snapshot)

    def save_plain_conspect(self, ident, name, payload):
        """Сохранить конспект с заданным payload напрямую (как старые данные или чужие инструменты)."""
        binding = recordings.document_binding(self.store, ident)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.store.save_conspect(
            ident, str(uuid.uuid4()), name, 1, None, payload, data, b"# synthetic\n",
            f"{name}.conspect.json", f"{name}.conspect.md", binding["root_id"], str(uuid.uuid4()),
            recordings.operation_context(self.store, "human"))

    def test_screen_conspects_show_section_texts(self):
        ident, _, _ = self.make_recording()
        # свёртка, сделанная исполнителем: разделы лежат в payload["evidence"]
        request = workflow.create(self.store, self.source, {"require_conspect": True})
        workflow.claim_start(self.store, request)
        with self.store._transaction():
            self.store.db.execute("UPDATE workflow_requests SET recording_id=? WHERE request_id=?",
                                  (ident, request))
        workflow.observe(self.store, request, {"state": "done", "recording_id": ident})
        task = workflow.prepare_model_task(self.store, request)
        ids = [word["word_id"] for word in task["words"]]
        workflow.accept_model_result(self.store, request, {
            "source_revision": task["source_revision"],
            "sections": [{"text": "Первый раздел.", "evidence": ids[:1]},
                         {"text": "Второй раздел.", "evidence": ids[1:]}]})
        # прежнее поле `sections` и конспект без разделов
        self.save_plain_conspect(ident, "old-sections", {"sections": [{"text": "Старый раздел."}]})
        self.save_plain_conspect(ident, "no-sections", {"name": "no-sections"})

        by_name = {item["name"]: item for item in core.recording_conspects(ident)}
        self.assertEqual(by_name["workflow-" + request]["sections"],
                         [{"text": "Первый раздел."}, {"text": "Второй раздел."}])
        self.assertEqual(by_name["old-sections"]["sections"], [{"text": "Старый раздел."}])
        self.assertEqual(by_name["no-sections"]["sections"], [])

    def test_conflict_preview_shows_preserved_bytes(self):
        ident, root_id, doc_paths = self.make_recording()
        binding = recordings.document_binding(self.store, ident)
        md_target = Path(binding["root"]) / binding["paths"]["md"]
        md_target.write_text("внешняя правка", encoding="utf-8")
        recordings.materialize_recording(self.store, ident)
        conflicts = core.list_document_conflicts(ident)
        self.assertEqual(len(conflicts), 1)
        conflict_id = conflicts[0]["conflict_id"]
        preview = core.preview_conflict(ident, conflict_id, "document")
        self.assertTrue(preview["preserved"])
        self.assertEqual(preview["preview"], "внешняя правка")
        self.assertFalse(preview["truncated"])
        self.assertEqual(core.list_conspect_conflicts(ident), [])
        with self.assertRaises(core.CoreError):
            core.preview_conflict(str(uuid.uuid4()), conflict_id, "document")
        with self.assertRaises(core.CoreError):
            core.preview_conflict(ident, conflict_id, "conspect")

    def test_backup_creates_validated_inactive_copy(self):
        ident, _, _ = self.make_recording()
        destination = self.base / "backup.sqlite3"
        report = core.backup_store(destination)
        self.assertEqual(report["report"]["integrity"], "ok")
        copy = storage.open_store_readonly(destination)
        try:
            rows = copy.db.execute("SELECT active FROM installations").fetchall()
            self.assertEqual([row[0] for row in rows], [0])
        finally:
            copy.close()

    def test_restore_wizard_round_trip(self):
        ident, root_id, _ = self.make_recording()
        binding = recordings.document_binding(self.store, ident)
        old_root = Path(binding["root"])
        self.assertTrue((old_root / binding["paths"]["md"]).is_file())
        backup = self.base / "backup.sqlite3"
        core.backup_store(backup)
        info = core.inspect_backup(backup)
        self.assertTrue(info["restore_guard_locked"])
        self.assertEqual(info["recordings"], 1)
        self.assertEqual([root["root_id"] for root in info["roots"]], [root_id])
        user_id = self.store.current_identity()["user_id"]

        new_root = self.base / "restored-docs"
        new_root.mkdir()
        destination = self.base / "restored.sqlite3"
        report = core.restore_from_backup(backup, destination, {root_id: new_root}, user_id)
        self.assertEqual(report["destination"], str(destination))
        self.assertIsNone(report["previous_saved_as"])
        self.assertTrue((new_root / binding["paths"]["md"]).is_file())
        self.assertFalse(list(self.base.glob("restored.sqlite3.incomplete-*")))

        restored = storage.open_store(destination)
        try:
            identity = restored.current_identity()
            self.assertNotEqual(identity["installation_id"], self.store.current_identity()["installation_id"])
            self.assertEqual(identity["user_id"], user_id)
            self.assertEqual(restored.get_current(ident)["transcript"]["revision"], 1)
        finally:
            restored.close()
        rebind = storage.open_store(destination)
        try:
            self.assertTrue(documents.verify_files(rebind, root_id, new_root)["consistent"])
        finally:
            rebind.close()

    def test_restore_keeps_previous_database_and_rejects_bad_targets(self):
        ident, root_id, _ = self.make_recording()
        backup = self.base / "backup.sqlite3"
        core.backup_store(backup)
        user_id = self.store.current_identity()["user_id"]
        destination = self.base / "restored.sqlite3"
        docs_a = self.base / "docs-a"
        docs_a.mkdir()
        core.restore_from_backup(backup, destination, {root_id: docs_a}, user_id)
        # повторная замена сохраняет прежнюю базу рядом
        second = self.base / "docs-b"
        second.mkdir()
        report = core.restore_from_backup(backup, destination, {root_id: second}, user_id)
        self.assertTrue(report["previous_saved_as"])
        self.assertTrue(Path(report["previous_saved_as"]).is_file())
        # восстановление поверх самого резерва запрещено
        with self.assertRaises(core.CoreError):
            core.restore_from_backup(backup, backup, {root_id: second}, user_id)
        # без mapping мастер не выполняется
        with self.assertRaises(core.CoreError):
            core.restore_from_backup(backup, self.base / "other.sqlite3", {}, user_id)


class P5ScreenHTTPTests(unittest.TestCase):
    def setUp(self):
        self.base = sandbox(self)
        self.patcher = mock.patch.object(paths, "STATE_DB", self.base / "state.sqlite3")
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        dbtemplate.initialize(self.base / "state.sqlite3", "Тест")
        self.srv, self.url, _ = workflow_page.make_server(8790)
        self.prefix = urlsplit(self.url).path
        self.host = f"127.0.0.1:{self.srv.server_port}"
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)

    def get(self, path):
        conn = http.client.HTTPConnection(self.host, timeout=10)
        try:
            conn.request("GET", self.prefix + path, headers={"Host": self.host})
            response = conn.getresponse()
            raw = response.read().decode("utf-8")
            try:
                return response.status, json.loads(raw)
            except ValueError:
                return response.status, raw
        finally:
            conn.close()

    def post(self, path, body):
        conn = http.client.HTTPConnection(self.host, timeout=10)
        try:
            payload = json.dumps(body, ensure_ascii=False)
            conn.request("POST", self.prefix + path, body=payload.encode("utf-8"),
                         headers={"Host": self.host, "Content-Type": "application/json",
                                  "Origin": f"http://{self.host}"})
            response = conn.getresponse()
            return response.status, json.loads(response.read().decode("utf-8"))
        finally:
            conn.close()

    def test_page_and_api_flows(self):
        status, html = self.get("")
        self.assertEqual(status, 200)
        self.assertIn("обработка записи", html)
        status, data = self.get("api/requests")
        self.assertEqual((status, data["requests"]), (200, []))
        with mock.patch.object(core, "start_workflow",
                               return_value={"request_id": str(uuid.uuid4()), "job_id": "job"}):
            status, started = self.post("api/start", {"audio": str(self.base / "a.wav"),
                                                      "require_conspect": True,
                                                      "allow_cloud_model": False})
        self.assertEqual(status, 200)
        status, backed = self.post("api/backup", {"destination": str(self.base / "copy.sqlite3")})
        self.assertEqual(status, 200)
        self.assertEqual(backed["report"]["integrity"], "ok")
        status, fail = self.post("api/start", {"audio": ""})
        self.assertEqual(status, 400)

    def test_unknown_route_and_bad_body(self):
        self.assertEqual(self.get("api/none")[0], 404)
        status, error = self.post("api/backup", {})
        self.assertEqual(status, 400)
        # неизвестный запрос и отсутствие записи — понятная ошибка, не обрыв соединения
        unknown = "00000000-0000-4000-8000-000000000000"
        self.assertEqual(self.get(f"api/status?request_id={unknown}")[0], 400)
        self.assertEqual(self.get(f"api/result?request_id={unknown}")[0], 400)
        status, history = self.get(f"api/history?recording_id={unknown}&kind=transcript")
        self.assertEqual((status, history["revisions"]), (200, []))

    def test_start_rejects_string_booleans_and_non_integer_budget(self):
        with mock.patch.object(core, "start_workflow") as start:
            status, error = self.post("api/start", {
                "audio": str(self.base / "synthetic.wav"),
                "require_conspect": "false", "allow_cloud_model": False})
            self.assertEqual(status, 400)
            self.assertIn("логическое", error["error"])
            status, error = self.post("api/start", {
                "audio": str(self.base / "synthetic.wav"),
                "require_conspect": False, "allow_cloud_model": False,
                "question_budget": 1.5})
            self.assertEqual(status, 400)
            self.assertIn("целое", error["error"])
        start.assert_not_called()

    def test_rt_screen_passes_cli_arguments_to_page_server(self):
        parsed = rt.build().parse_args(["screen", "--port", "8799", "--hours", "1.5"])
        with mock.patch("workflow_page.main") as page_main:
            rt.run(parsed)
        page_main.assert_called_once_with(["--port", "8799", "--hours", "1.5"])


if __name__ == "__main__":
    unittest.main()
