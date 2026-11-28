"""П5: локальный экран и эксплуатация — история, возврат ревизии, конфликты, резерв, HTTP-экран."""
import gc
import http.client
import json
import datetime
import os
import select
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock
from urllib.parse import quote, urlsplit

import documents
import model_tasks
import paths
import recordings
import settings
import storage
import workflow

import core
import core_screen
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
        self.store = storage.initialize(self.base / "state.sqlite3", initial_display_name="Тест")
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
        # разделы собраны так же, как в файле свёртки (О255-М1): со временем начала раздела; без цитат текст тот же
        self.assertEqual(by_name["workflow-" + request]["sections"],
                         [{"text": "[00:00:00]\nПервый раздел."}, {"text": "[00:00:01]\nВторой раздел."}])
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


class ScreenRecordingListTests(unittest.TestCase):
    """М-5: список всех записей базы на экране и открытие записи без запроса — только чтение."""

    def setUp(self):
        self.base = sandbox(self)
        self.patcher = mock.patch.object(paths, "STATE_DB", self.base / "state.sqlite3")
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.store = storage.initialize(self.base / "state.sqlite3", initial_display_name="Тест")
        self.addCleanup(self.store.close)

    def make_recording(self, name, *, quality=None, review=(), title=None, recorded_at=None, materialize=True):
        source = self.base / "Записи" / name
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(b"synthetic")
        layout = recordings.document_layout(source, "")
        layout["root"].mkdir()
        root_id, doc_paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        ident = str(uuid.uuid4())
        words = [{"w": "секретное", "s": 0, "e": 1}, {"w": "слово", "s": 1, "e": 2}]
        data = {"file": str(source), "duration": "00:00:02", "speakers": ["Анна Тестова"],
                "names_confirmed": True, "quality": dict(quality or {}), "review": [], "review_done": False,
                "segments": [{"start": 0, "end": 2, "speaker": "Анна Тестова", "words": words}]}
        if title is not None:
            data["title"] = title
        if recorded_at is not None:
            data["recorded_at"] = recorded_at
        recordings.commit_worker_result(
            self.store, data, source, root_id, recordings.operation_context(self.store, "model"),
            recording_id=ident, operation_id=str(uuid.uuid4()),
            generated_at="2026-01-01T00:00:00Z", document_paths=doc_paths)
        if review:
            current = self.store.get_current(ident)
            payload = current["review"]["payload"]
            payload["review"] = [dict(item, id=f"r{index}", t=0, t1=0, t2=1, text="секретное")
                                 for index, item in enumerate(review, 1)]
            self.store.commit_change_with_document(
                ident, {"source_revision": 1, "review_revision": 1}, str(uuid.uuid4()),
                recordings.operation_context(self.store, "human"), review=payload, root_id=root_id,
                generated_at="2026-01-01T00:01:00Z")
        if materialize:
            recordings.materialize_recording(self.store, ident)
        return ident

    def by_name(self):
        return {item["name"]: item for item in core_screen.recording_list()["items"]}

    def test_without_store_the_list_explains_why_it_is_empty(self):
        self.store.close()
        (self.base / "state.sqlite3").unlink()
        with mock.patch.object(settings, "recording_backend", return_value="legacy_files"):
            listed = core_screen.recording_list()
        self.assertEqual((listed["available"], listed["items"], listed["total"]), (False, [], 0))
        self.assertIn("Базы записей ещё нет", listed["reason"])
        self.assertIn("rt.py legacy-read", listed["reason"])          # переноса нет (У-1), только чтение
        self.assertNotIn("legacy-migration", listed["reason"])
        with mock.patch.object(settings, "recording_backend", return_value="sqlite"):
            self.assertNotIn("в файлах", core_screen.recording_list()["reason"])

    def test_empty_store_lists_nothing(self):
        listed = core_screen.recording_list()
        self.assertEqual((listed["available"], listed["items"], listed["total"], listed["truncated"]),
                         (True, [], 0, False))

    def test_rows_show_origin_variant_request_and_names(self):
        program = self.make_recording("Встреча.wav", title="Планёрка", recorded_at="2026-11-01T10:00:00")
        self.make_recording("Старая.wav", quality={"migration": {"tool": "t", "variant": {"tag": "a", "main": True}}})
        self.make_recording("Старая-2.wav", quality={"migration": {"tool": "t", "variant": {"tag": "b", "main": False}}})
        self.make_recording("Старая-3.wav", quality={"migration": {"tool": "t"}})
        request = workflow.create(self.store, self.base / "Записи" / "Встреча.wav", {"require_conspect": False})
        with self.store._transaction():
            self.store.db.execute("UPDATE workflow_requests SET recording_id=? WHERE request_id=?", (program, request))
        rows = self.by_name()
        self.assertEqual(set(rows), {"Встреча.wav", "Старая.wav", "Старая-2.wav", "Старая-3.wav"})
        first = rows["Встреча.wav"]
        self.assertEqual((first["origin"], first["variant"], first["title"], first["recorded_at"], first["duration"]),
                         ("program", None, "Планёрка", "2026-11-01T10:00:00", "00:00:02"))
        self.assertEqual(first["request"], {"request_id": request, "state": workflow.get(self.store, request)["state"]})
        self.assertIs(first["names_confirmed"], True)
        self.assertIsNone(first["error"])
        self.assertEqual((rows["Старая.wav"]["origin"], rows["Старая.wav"]["variant"]),
                         ("migrated", {"tag": "a", "main": True}))
        self.assertEqual(rows["Старая-2.wav"]["variant"], {"tag": "b", "main": False})
        self.assertEqual(rows["Старая-3.wav"]["variant"], {"tag": None, "main": None})
        self.assertIsNone(rows["Старая.wav"]["request"])

    def test_places_are_counted_by_the_program_rule(self):
        self.make_recording("Разбор.wav", review=[{"status": "open"}, {"status": "human"},
                                                  {"status": "resolved"}, {}])
        self.make_recording("Чисто.wav", review=[{"status": "resolved"}])
        self.make_recording("Пусто.wav")
        rows = self.by_name()
        self.assertEqual(rows["Разбор.wav"]["places"], {"unresolved": 3, "open": 2, "total": 4})
        self.assertEqual(rows["Чисто.wav"]["places"], {"unresolved": 0, "open": 0, "total": 1})
        self.assertEqual(rows["Пусто.wav"]["places"], {"unresolved": 0, "open": 0, "total": 0})

    def test_document_state_conflicts_and_conspects(self):
        synced = self.make_recording("Записана.wav")
        self.make_recording("Ждёт.wav", materialize=False)
        conflicted = self.make_recording("Конфликт.wav")
        failed = self.make_recording("Сбой.wav")
        binding = recordings.document_binding(self.store, conflicted)
        (Path(binding["root"]) / binding["paths"]["md"]).write_text("внешняя правка", encoding="utf-8")
        self.change_text(conflicted)
        with self.store._transaction():
            self.store.db.execute("UPDATE document_outbox SET state='error' WHERE recording_id=?", (failed,))
        binding = recordings.document_binding(self.store, synced)
        self.store.save_conspect(
            synced, str(uuid.uuid4()), "заметки", 1, None, {"sections": []}, b'{"sections": []}', b"# synthetic\n",
            "заметки.conspect.json", "заметки.conspect.md", binding["root_id"], str(uuid.uuid4()),
            recordings.operation_context(self.store, "human"))
        rows = self.by_name()
        self.assertEqual({name: row["document_sync"] for name, row in rows.items()},
                         {"Записана.wav": "synced", "Ждёт.wav": "pending", "Конфликт.wav": "conflict",
                          "Сбой.wav": "error"})
        self.assertEqual(rows["Конфликт.wav"]["open_conflicts"], 1)
        self.assertEqual(rows["Записана.wav"]["open_conflicts"], 0)
        self.assertEqual((rows["Записана.wav"]["conspects"], rows["Ждёт.wav"]["conspects"]), (1, 0))
        # решённый конфликт не считается открытым
        with self.store._transaction():
            self.store.db.execute(
                "UPDATE document_conflicts SET resolution_operation_id=(SELECT operation_id FROM operations LIMIT 1) "
                "WHERE recording_id=?", (conflicted,))
        self.assertEqual(self.by_name()["Конфликт.wav"]["open_conflicts"], 0)

    def change_text(self, ident):
        current = self.store.get_current(ident)
        transcript = current["transcript"]["payload"]
        transcript["segments"][0]["words"][1]["w"] = "исправлено"
        binding = recordings.document_binding(self.store, ident)
        self.store.commit_change_with_document(
            ident, {"source_revision": current["transcript"]["revision"],
                    "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            transcript=transcript, review=current["review"]["payload"], root_id=binding["root_id"],
            generated_at="2026-01-01T00:02:00Z")
        recordings.materialize_recording(self.store, ident)

    def test_list_shows_no_paths_text_or_speaker_names(self):
        self.make_recording("Встреча.wav", review=[{"status": "open"}])
        dumped = json.dumps(core_screen.recording_list(), ensure_ascii=False)
        for secret in (str(self.base), "Записи", "секретное", "Анна", "segments", "words", "location", "speakers"):
            self.assertNotIn(secret, dumped)
        self.assertIn("Встреча.wav", dumped)

    def test_latest_change_comes_first(self):
        first = self.make_recording("Первая.wav")
        self.make_recording("Вторая.wav")
        operations = [row[0] for row in self.store.db.execute("SELECT created_at FROM operations ORDER BY created_at")]
        self.assertEqual(len(set(operations)), len(operations))
        self.assertEqual([item["name"] for item in core_screen.recording_list()["items"]], ["Вторая.wav", "Первая.wav"])
        self.change_text(first)
        self.assertEqual([item["name"] for item in core_screen.recording_list()["items"]], ["Первая.wav", "Вторая.wav"])

    def test_unreadable_row_is_marked_and_the_rest_still_listed(self):
        broken = self.make_recording("Сломана.wav")
        self.make_recording("Целая.wav")
        with self.store._transaction():
            self.store.db.execute("UPDATE transcript_revisions SET payload='{' WHERE recording_id=?", (broken,))
        rows = self.by_name()
        self.assertIn("сломана.wav", rows)    # имя — из пути в базе, он хранится в нижнем регистре
        self.assertTrue(rows["сломана.wav"]["error"].startswith("Запись не читается"))
        self.assertNotIn("places", rows["сломана.wav"])
        self.assertIsNone(rows["Целая.wav"]["error"])
        with self.assertRaises((core.CoreError, storage.StorageError)):
            core_screen.recording_view(broken)

    def test_list_is_capped(self):
        for name in ("А.wav", "Б.wav", "В.wav"):
            self.make_recording(name)
        with mock.patch.object(core_screen, "RECORDING_LIST_LIMIT", 2):
            listed = core_screen.recording_list()
        self.assertEqual((len(listed["items"]), listed["total"], listed["truncated"]), (2, 3, True))

    def test_view_matches_request_result_and_rejects_bad_ids(self):
        ident = self.make_recording("Встреча.wav")
        request = workflow.create(self.store, self.base / "Записи" / "Встреча.wav", {"require_conspect": False})
        with self.store._transaction():
            self.store.db.execute("UPDATE workflow_requests SET recording_id=? WHERE request_id=?", (ident, request))
        view = core_screen.recording_view(ident)
        self.assertEqual(view["recording"]["name"], "Встреча.wav")
        self.assertEqual([item["request_id"] for item in view["requests"]], [request])
        self.assertEqual(workflow_page.result_view(view["data"]),
                         workflow_page.result_view(core.read_recording_result(ident)))
        for bad in ("", "не-номер", str(uuid.uuid4()).upper(), str(uuid.uuid4())):
            with self.subTest(bad=bad), self.assertRaises(core.CoreError):
                core_screen.recording_view(bad)

    def test_http_routes_are_read_only_and_guarded(self):
        ident = self.make_recording("Встреча.wav")
        srv, url, _ = workflow_page.make_server(8790)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        prefix, host = urlsplit(url).path, f"127.0.0.1:{srv.server_port}"

        def request(method, target, headers=None, body=None):
            conn = http.client.HTTPConnection(host, timeout=10)
            try:
                fields = {"Host": host}
                fields.update(headers or {})
                conn.request(method, target, body=body, headers=fields)
                response = conn.getresponse()
                return response.status, json.loads(response.read().decode("utf-8"))
            finally:
                conn.close()

        status, listed = request("GET", prefix + "api/recordings")
        self.assertEqual((status, [item["recording_id"] for item in listed["items"]]), (200, [ident]))
        status, opened = request("GET", prefix + f"api/recording?recording_id={ident}")
        self.assertEqual(status, 200)
        self.assertEqual(opened["result"], workflow_page.result_view(core.read_recording_result(ident)))
        self.assertEqual((opened["recording"]["name"], opened["requests"]), ("Встреча.wav", []))
        self.assertEqual(request("GET", prefix + "api/recordings", {"Host": "evil.example"})[0], 403)
        self.assertEqual(request("GET", prefix.rstrip("/") + "x/api/recordings")[0], 403)
        for bad in ("не-номер", str(uuid.uuid4())):
            with self.subTest(bad=bad):
                status, error = request("GET", prefix + f"api/recording?recording_id={quote(bad)}")
                self.assertEqual(status, 400)
                self.assertRegex(error["error"], "[А-Яа-яЁё]")
                self.assertNotIn("Traceback", error["error"])
        # записи через эти адреса нет: POST получает отказ, а не сброс соединения
        for route in ("api/recordings", "api/recording"):
            with self.subTest(route=route):
                status, error = request("POST", prefix + route, {
                    "Origin": f"http://{host}", "Content-Type": "application/json"}, b'{"recording_id": "x"}')
                self.assertEqual(status, 404)
                self.assertIn("error", error)


class P5ScreenHTTPTests(unittest.TestCase):
    def setUp(self):
        self.base = sandbox(self)
        self.patcher = mock.patch.object(paths, "STATE_DB", self.base / "state.sqlite3")
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        recordings.initialize_store(self.base / "state.sqlite3", display_name="Тест")
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

    def raw_head(self, path, length, headers=None, target=None):
        # копия помощника tests/test_review_http.py (тот же приём в review_page.py): правя один, смотри и другой
        fields = {"Host": self.host, "Origin": f"http://{self.host}", "Content-Type": "application/json",
                  "Content-Length": str(length), "Connection": "close"}
        fields.update(headers or {})
        return (f"POST {target or self.prefix + path} HTTP/1.1\r\n"
                + "".join(f"{k}: {v}\r\n" for k, v in fields.items()) + "\r\n").encode()

    def raw_post(self, path, length, parts, pause=0.1, headers=None, target=None):
        """Запрос на сыром сокете: после первой части тела — пауза перед каждой следующей (тело приходит уже после отказа)."""
        with socket.create_connection(("127.0.0.1", self.srv.server_port), timeout=10) as sock:
            sock.sendall(self.raw_head(path, length, headers, target) + parts[0])
            for part in parts[1:]:
                time.sleep(pause)
                sock.sendall(part)
            data = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
        top, _, body = data.partition(b"\r\n\r\n")
        lines = top.decode("iso-8859-1").split("\r\n")
        return int(lines[0].split()[1]), dict(line.lower().split(": ", 1) for line in lines[1:]), body

    def test_every_early_refusal_reads_the_body_first_so_the_client_gets_the_answer(self):
        # Без дочитывания тела в Windows клиент получал сброс соединения (WinError 10053) вместо ответа
        body = b'{"audio": "x"}'
        limit = workflow_page.MAX_BODY_BYTES + 1
        cases = (
            ("403 адрес", 403, {"path": "api/start", "headers": {"Host": "evil.example"}}),
            ("403 источник", 403, {"path": "api/start", "headers": {"Origin": "https://evil.example"}}),
            ("403 сессия", 403, {"path": "", "target": self.prefix.rstrip("/") + "x/api/start"}),
            ("415 тип", 415, {"path": "api/start", "headers": {"Content-Type": "text/plain"}}),
            ("413 размер", 413, {"path": "api/start", "length": limit}),
        )
        for name, code, case in cases:
            with self.subTest(name):
                length = case.get("length", len(body))
                parts = [body[:5], body[5:]] if length == len(body) else [b"x" * 100, b"x" * (length - 100)]
                status, headers, answer = self.raw_post(case["path"], length, parts, headers=case.get("headers"),
                                                        target=case.get("target"))
                self.assertEqual(status, code)
                self.assertEqual(headers["connection"], "close")
                self.assertIn("error", json.loads(answer))

    def test_a_body_that_trickles_in_does_not_hold_the_refusal_longer_than_the_drain_timeout(self):
        started = time.time()
        with socket.create_connection(("127.0.0.1", self.srv.server_port), timeout=10) as sock:
            sock.sendall(self.raw_head("api/start", workflow_page.MAX_BODY_BYTES + 1))
            while time.time() - started < workflow_page.DRAIN_TIMEOUT + 5:
                if select.select([sock], [], [], 0.5)[0]:
                    break
                try:
                    sock.sendall(b"x")
                except OSError:
                    break
            seconds = time.time() - started
        self.assertGreaterEqual(seconds, workflow_page.DRAIN_TIMEOUT - 0.5)
        self.assertLess(seconds, workflow_page.DRAIN_TIMEOUT + 1.5)

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

    def test_start_passes_speaker_hints_and_form_is_optional(self):
        status, html = self.get("")
        for needle in ('id="who-mode"', 'id="who-add"', 'id="who-enroll" checked', "как могут обращаться"):
            self.assertIn(needle, html)
        audio = str(self.base / "a.wav")
        base = {"audio": audio, "require_conspect": False, "allow_cloud_model": False}
        with mock.patch.object(core, "start_workflow", return_value={"request_id": "r", "job_id": "j"}) as start:
            self.assertEqual(self.post("api/start", base)[0], 200)
            self.assertNotIn("speaker_hints", start.call_args.args[1])      # старый вызов: как раньше
            hints = {"count_mode": "approx", "count": 4, "stray_voices": True, "group": "demo",
                     "participants": [{"name": "Анна", "forms": []}, {"name": "Борис", "forms": ["Боря"]}],
                     "enroll_profile": False}
            self.assertEqual(self.post("api/start", {**base, "speaker_hints": hints})[0], 200)
            got = start.call_args.args[1]["speaker_hints"]
            self.assertEqual((got["count_mode"], got["count"], got["enroll_profile"]), ("approx", 4, False))
            self.assertEqual([p["name"] for p in got["participants"]], ["Анна", "Борис"])
            start.reset_mock()
            status, error = self.post("api/start", {**base, "speaker_hints": {"count_mode": "exact", "count": 99}})
            self.assertEqual(status, 400)
            start.assert_not_called()

    def interrupted_request(self):
        """Запрос, запущенный давно и без файла статуса задачи: программа сама считает запуск не подтвердившимся."""
        source = self.base / "synthetic.wav"
        source.write_bytes(b"synthetic")
        store = recordings.open_store()
        try:
            request = workflow.create(store, source, {"require_conspect": False})
            workflow.claim_start(store, request)
            old = (datetime.datetime.now(datetime.timezone.utc)
                   - datetime.timedelta(seconds=workflow.LAUNCH_CONFIRM_SEC + 60)).isoformat()
            with store._transaction():
                store.db.execute("UPDATE workflow_requests SET updated_at=? WHERE request_id=?", (old, request))
        finally:
            store.close()
        return request, source

    def test_unconfirmed_launch_shows_note_and_cancel_goes_through_confirmation(self):
        request, source = self.interrupted_request()
        status, shown = self.get(f"api/status?request_id={request}")
        self.assertEqual(status, 200)
        self.assertEqual((shown["state"], shown["state_ru"]), ("interrupted", "прервано"))
        self.assertIn("не подтвердился", shown["launch_note"])
        self.assertEqual(shown["source"], str(source.resolve()))
        # «Отменить»: сначала снимок, отказ — только по подтверждению; без снимка ничего не меняется
        status, error = self.post("api/abandon/confirm", {"snapshot": None})
        self.assertEqual(status, 400)
        self.assertEqual(self.get(f"api/status?request_id={request}")[1]["state"], "interrupted")
        status, snapshot = self.post("api/abandon/prepare", {"request_id": request})
        self.assertEqual((status, snapshot["request_id"]), (200, request))
        self.assertEqual(self.get(f"api/status?request_id={request}")[1]["state"], "interrupted")
        status, done = self.post("api/abandon/confirm", {"snapshot": snapshot})
        self.assertEqual((status, done["state"]), (200, "failed"))
        status, after = self.get(f"api/status?request_id={request}")
        self.assertEqual((after["state"], after["state_ru"]), ("failed", "запуск отменён"))
        listed = self.get("api/requests")[1]["requests"]
        self.assertEqual([(row["request_id"], row["abandoned"]) for row in listed], [(request, True)])
        self.assertIsNone(after["launch_note"])
        # повторно отменить уже отменённое нельзя
        self.assertEqual(self.post("api/abandon/prepare", {"request_id": request})[0], 400)

    def test_young_launch_has_no_note(self):
        source = self.base / "synthetic.wav"
        source.write_bytes(b"synthetic")
        store = recordings.open_store()
        try:
            request = workflow.create(store, source)
            workflow.claim_start(store, request)
        finally:
            store.close()
        status, shown = self.get(f"api/status?request_id={request}")
        self.assertEqual((shown["state"], shown["launch_note"]), ("starting", None))

    @unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
    def test_page_shows_note_and_two_buttons_only_for_unconfirmed_launch(self):
        """Вид страницы: что видит человек на прерванном запросе (JavaScript экрана выполняется в Node)."""
        html = (Path(workflow_page.__file__).parent / "workflow.html").read_text(encoding="utf-8")
        esc = [line for line in html.splitlines() if line.startswith("const esc = ")][0]
        code = html[html.index("function renderStatus(s)"):html.index("async function showVerify()")]
        program = esc + chr(10) + code + """
        const note = "Запуск не подтвердился или был прерван: можно запустить снова или отменить.";
        const base = {state: "interrupted", state_ru: "прервано", source: "C:/a.wav", recording_id: null,
                      launch_note: note, readiness: {}, policy: {require_conspect: true, allow_cloud_model: false,
                      goal: "цель", question_budget: 3, speaker_hints: null, participation: undefined}};
        const out = {
          shown: renderStatus(base),
          running: renderStatus({...base, state: "running", launch_note: null}),
          saved: renderStatus({...base, recording_id: "r1"}),
          body: launchAgainBody(base)};
        console.log(JSON.stringify(out));
        """
        result = subprocess.run([shutil.which("node")], input=program, capture_output=True, text=True,
                                encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        out = json.loads(result.stdout)
        for needle in ("Запуск не подтвердился", 'id="launch-again">Запустить снова<',
                       'id="launch-cancel">Отменить<'):
            self.assertIn(needle, out["shown"])
        for other in ("running", "saved"):                      # не прерван или результат уже есть — ни текста, ни кнопок
            self.assertNotIn("launch-again", out[other])
            self.assertNotIn("Запуск не подтвердился", out[other])
        # «Запустить снова» вызывает обычный запуск с тем же файлом и настройками, без пустых полей
        self.assertEqual(out["body"], {"audio": "C:/a.wav", "require_conspect": True,
                                       "allow_cloud_model": False, "goal": "цель", "question_budget": 3})

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
