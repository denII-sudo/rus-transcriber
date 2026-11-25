"""Х1 линии хранения: события карточек — в таблице review_events, без ревизии разбора на каждое событие.

План — приватный репозиторий: research/2026-11-25-storage-line/plan.md, раздел 8.1. Ревизия разбора пишется,
только когда меняются прослушанные отрезки (heard). Разбор прежнего формата (список interaction_events) читается.
База без таблицы (до обновления): при первом событии резерв, потом пустая таблица; без переноса, перезаписи и
удаления. Резерв не удался — показы не пишутся, предупреждение есть, heard пишется.

Только хранение записей в базе (sqlite). Данные синтетические.
"""
import atexit
import copy
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-review-events-data-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_pause_buttons_short_answers as six  # noqa: E402
from test_place_heard import old_format_events  # noqa: E402

core = six.core
recordings = six.recordings
review_pack = six.review_pack
files = six.files
import storage  # noqa: E402

SESSION = "page-session-1"


def head(jp):
    store, binding = recordings.open_for_document(jp, readonly=True)
    try:
        current = store.get_current(binding["recording_id"])
        rows = store.review_events(binding["recording_id"]) if store.has_review_events() else None
        return {"review_revision": current["review"]["revision"], "payload": current["review"]["payload"],
                "table": rows, "db": Path(store.path), "recording_id": binding["recording_id"]}
    finally:
        store.close()


def drop_events_table(jp):
    """База до Х1: та же схема 4, но без таблицы событий."""
    db = head(jp)["db"]
    conn = sqlite3.connect(db)
    try:
        conn.execute("DROP TABLE review_events")
        conn.commit()
    finally:
        conn.close()


class ReviewEvents(unittest.TestCase):
    def setUp(self):
        recordings._REVIEW_EVENTS_BACKUP_FAILED.clear()
        self.addCleanup(recordings._REVIEW_EVENTS_BACKUP_FAILED.clear)

    def test_event_is_a_row_not_a_revision(self):
        """Показ и проигрывание — строки таблицы; номер ревизии разбора не растёт; повтор показа в сессии — нет."""
        with files("sqlite") as (jp, _md, version):
            before = head(jp)
            self.assertEqual(before["table"], [])
            self.assertNotIn("interaction_events", before["payload"])
            self.assertTrue(review_pack.add_interaction_event(jp, "r1", version, "card_visible", SESSION)["stored"])
            self.assertTrue(review_pack.add_interaction_event(jp, "r1", version, "audio_play", SESSION)["stored"])
            again = review_pack.add_interaction_event(jp, "r1", version, "card_visible", SESSION)
            self.assertEqual(again, {"stored": False, "duplicate": True})
            after = head(jp)
            self.assertEqual(after["review_revision"], before["review_revision"])
            self.assertEqual([(e["id"], e["type"], e["session"]) for e in after["table"]],
                             [("r1", "card_visible", SESSION), ("r1", "audio_play", SESSION)])
            self.assertNotIn("interaction_events", after["payload"])
            state = recordings.read_review_pack_state(jp)
            self.assertEqual(state["interaction_events"], after["table"])
            self.assertEqual(storage.validate_copy(after["db"])["integrity"], "ok")

    def test_heard_change_writes_one_review_revision(self):
        """Прослушивание места: ревизия разбора с heard и строка события; повторное прослушивание — только строка."""
        with files("sqlite") as (jp, _md, version):
            before = head(jp)
            review_pack.add_interaction_event(jp, "r1", version, "place_heard", SESSION)
            first = head(jp)
            self.assertEqual(first["review_revision"], before["review_revision"] + 1)
            self.assertEqual(first["payload"]["undo"]["review_pack"]["heard"], {"": [[1.0, 2.0]]})
            self.assertNotIn("interaction_events", first["payload"])
            review_pack.add_interaction_event(jp, "r1", version, "place_heard", SESSION)
            second = head(jp)
            self.assertEqual(second["review_revision"], first["review_revision"], "heard не изменился")
            self.assertEqual([e["type"] for e in second["table"]], ["place_heard", "place_heard"])

    def test_operation_replay_does_not_duplicate(self):
        """Повтор с тем же ключом операции: ни второй строки, ни второй ревизии разбора."""
        with files("sqlite") as (jp, _md, version):
            store, binding = recordings.open_for_document(jp)
            try:
                rid = binding["recording_id"]
                current = store.get_current(rid)
                review = copy.deepcopy(current["review"]["payload"])
                review.setdefault("undo", {}).setdefault("review_pack", {})["heard"] = {"": [[0.0, 1.0]]}
                expected = {"source_revision": current["transcript"]["revision"],
                            "review_revision": current["review"]["revision"]}
                op, ctx = str(uuid.uuid4()), recordings.operation_context(store, "human")
                event = {"id": "r1", "type": "place_heard", "version": version}
                first = store.append_review_event(rid, event, op, ctx, expected_versions=expected, review=review)
                self.assertEqual(first, store.append_review_event(rid, event, op, ctx, expected_versions=expected,
                                                                  review=review))
                self.assertEqual(len(store.review_events(rid)), 1)
                self.assertEqual(store.get_current(rid)["review"]["revision"], expected["review_revision"] + 1)
                with self.assertRaises(storage.Conflict):
                    store.append_review_event(rid, {"other": 1}, op, ctx)
                with self.assertRaises(ValueError):
                    store.append_review_event(rid, ["not", "object"], str(uuid.uuid4()), ctx)
                with self.assertRaises(ValueError):
                    store.append_review_event(rid, {"x": "я" * storage.MAX_REVIEW_EVENT_BYTES}, str(uuid.uuid4()), ctx)
            finally:
                store.close()

    def test_old_format_list_is_read_and_kept(self):
        """Прежний разбор со списком событий: читается вместе с таблицей; список не дописывается и не теряется."""
        with files("sqlite") as (jp, md, version):
            old = [{"id": "r2", "version": version, "type": "card_visible", "session": "old-session-1",
                    "source": "local", "ts": "2026-11-20T10:00:00"}]
            old_format_events(jp, old)
            review_pack.add_interaction_event(jp, "r1", version, "card_visible", SESSION)
            # повтор показа, записанного прежним форматом, тоже узнаётся
            dup = review_pack.add_interaction_event(jp, "r2", version, "card_visible", "old-session-1")
            self.assertEqual(dup, {"stored": False, "duplicate": True})
            state = recordings.read_review_pack_state(jp)
            self.assertEqual([e["id"] for e in state["interaction_events"]], ["r2", "r1"])
            # ответ и «Применить» пишут новые ревизии: прежний список переходит как есть, строки таблицы в него
            # не копируются
            review_pack.add_answers(jp, [{"id": "r2", "version": version, "option": 0,
                                          "ts": "2026-11-25T10:00:00"}], "local", trusted_actor="human")
            core.apply_answers_files(jp, md)
            now = head(jp)
            self.assertEqual(now["payload"]["interaction_events"], old)
            self.assertEqual([e["id"] for e in now["table"]], ["r1"])
            self.assertEqual(storage.validate_copy(now["db"])["integrity"], "ok")

    def test_new_format_apply_does_not_copy_table_into_review(self):
        """Новый формат: после событий и «Применить» в разборе нет списка событий, события — только в таблице."""
        with files("sqlite") as (jp, md, version):
            review_pack.add_interaction_event(jp, "r1", version, "card_visible", SESSION)
            review_pack.add_answers(jp, [{"id": "r2", "version": version, "option": 0,
                                          "ts": "2026-11-25T10:00:00"}], "local", trusted_actor="human")
            core.apply_answers_files(jp, md)
            now = head(jp)
            self.assertNotIn("interaction_events", now["payload"])
            self.assertEqual([e["type"] for e in now["table"]], ["card_visible"])

    def test_missing_table_gets_backup_then_empty_table(self):
        """База без таблицы: первое событие — проверенный неактивный резерв в db-backups, потом пустая таблица;
        прежний список не переносится, ревизии разбора не удаляются и не переписываются."""
        with files("sqlite") as (jp, _md, version):
            old = [{"id": "r2", "version": version, "type": "card_visible", "session": "old-session-1"}]
            old_format_events(jp, old)
            drop_events_table(jp)
            before = head(jp)
            self.assertIsNone(before["table"])
            self.assertEqual(storage.validate_copy(before["db"])["integrity"], "ok")
            store = storage.open_store(before["db"])
            try:
                revisions = store.db.execute("SELECT revision, payload_hash FROM review_revisions "
                                             "ORDER BY revision").fetchall()
                revisions = [tuple(row) for row in revisions]
            finally:
                store.close()
            self.assertTrue(review_pack.add_interaction_event(jp, "r1", version, "card_visible", SESSION)["stored"])
            backups = sorted((before["db"].parent / "db-backups").glob("state-before-review-events-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            report = storage.validate_copy(backups[0])
            self.assertTrue(report["inactive"])
            copy_store = storage.open_store_readonly(backups[0])
            try:
                self.assertFalse(copy_store.has_review_events())
            finally:
                copy_store.close()
            after = head(jp)
            self.assertEqual(after["review_revision"], before["review_revision"])
            self.assertEqual([e["id"] for e in after["table"]], ["r1"])
            self.assertEqual(after["payload"]["interaction_events"], old)
            store = storage.open_store(after["db"])
            try:
                self.assertEqual([tuple(row) for row in store.db.execute(
                    "SELECT revision, payload_hash FROM review_revisions ORDER BY revision")], revisions)
            finally:
                store.close()
            review_pack.add_interaction_event(jp, "r1", version, "audio_play", SESSION)
            self.assertEqual(len(list((before["db"].parent / "db-backups").glob("*.sqlite3"))), 1,
                             "второй резерв при уже созданной таблице")

    def test_backup_failure_skips_events_and_keeps_heard(self):
        """Резерв не удался: показ не записан, предупреждение есть; прослушивание места записано ревизией разбора;
        резерв в этом процессе больше не пробуется; ревизия на показ не создаётся."""
        with files("sqlite") as (jp, _md, version):
            drop_events_table(jp)
            before = head(jp)
            with mock.patch.object(storage.Store, "backup", side_effect=OSError("no space")) as backup:
                shown = review_pack.add_interaction_event(jp, "r1", version, "card_visible", SESSION)
                self.assertFalse(shown["stored"])
                self.assertEqual(shown["warning"], recordings.REVIEW_EVENTS_NOT_UPGRADED)
                self.assertIn("база не обновлена", shown["warning"])
                self.assertEqual(head(jp)["review_revision"], before["review_revision"])
                heard = review_pack.add_interaction_event(jp, "r1", version, "place_heard", SESSION)
                self.assertFalse(heard["stored"])
                self.assertIn("warning", heard)
                self.assertEqual(backup.call_count, 1, "резерв пробуется на каждое событие")
            after = head(jp)
            self.assertIsNone(after["table"], "таблица создана без резерва")
            self.assertEqual(after["review_revision"], before["review_revision"] + 1)
            self.assertEqual(after["payload"]["undo"]["review_pack"]["heard"], {"": [[1.0, 2.0]]})
            self.assertNotIn("interaction_events", after["payload"])
            self.assertEqual(review_pack.heard_ids(review_pack.answers_state(jp), review_pack.manifest(jp)),
                             ["r1", "r2"])
            self.assertFalse((before["db"].parent / "db-backups").exists() and
                             list((before["db"].parent / "db-backups").glob("*.sqlite3")))

    def test_busy_backup_is_retried(self):
        """Занятая база при резерве — не отказ навсегда: следующее событие пробует снова."""
        with files("sqlite") as (jp, _md, version):
            drop_events_table(jp)
            with mock.patch.object(storage.Store, "backup", side_effect=storage.Busy("database is busy")):
                with self.assertRaises(storage.Busy):
                    review_pack.add_interaction_event(jp, "r1", version, "card_visible", SESSION)
            self.assertTrue(review_pack.add_interaction_event(jp, "r1", version, "card_visible", SESSION)["stored"])
            self.assertEqual(len(head(jp)["table"]), 1)

    def test_storage_refuses_event_without_table(self):
        with files("sqlite") as (jp, _md, _version):
            drop_events_table(jp)
            store, binding = recordings.open_for_document(jp)
            try:
                self.assertEqual(store.review_events(binding["recording_id"]), [])
                with self.assertRaises(storage.ReviewEventsMissing):
                    store.append_review_event(binding["recording_id"], {"type": "card_visible"}, str(uuid.uuid4()),
                                              recordings.operation_context(store, "human"))
                store.create_review_events_table()
                store.create_review_events_table()
                self.assertTrue(store.has_review_events())
            finally:
                store.close()


    def test_page_answers_do_not_carry_events(self):
        """api/answers в SQLite: события карточек странице не отдаются (таблица без предела 2000)."""
        import review_page
        with files("sqlite") as (jp, _md, version):
            review_pack.add_interaction_event(jp, "r1", version, "card_visible", SESSION)
            payload = review_page.answers_payload(jp)
            self.assertNotIn("interaction_events", payload)
            self.assertNotIn("interaction_events_dropped", payload)
            self.assertEqual(payload["pack_version"], version)
            self.assertIn("answer_states", payload)


class ValidateCopy(unittest.TestCase):
    def make(self, root, table=True):
        db = Path(root) / "state.sqlite3"
        storage.initialize(db, initial_display_name="Тест").close()
        if not table:
            conn = sqlite3.connect(db)
            conn.execute("DROP TABLE review_events")
            conn.commit()
            conn.close()
        return db

    def test_with_and_without_table(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(storage.validate_copy(self.make(root))["integrity"], "ok")
        with tempfile.TemporaryDirectory() as root:
            db = self.make(root, table=False)
            self.assertEqual(storage.validate_copy(db)["integrity"], "ok")
            store = storage.open_store(db)
            try:
                self.assertFalse(store.has_review_events())
            finally:
                store.close()

    def test_bad_rows_and_columns_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            db = self.make(root)
            store = storage.open_store(db)
            try:
                identity = store.current_identity()
                ctx = store.capture_context(identity["installation_id"], "human")
                rid = str(uuid.uuid4())
                store.create_recording(rid, "x.wav", {"segments": []}, {"review": []}, str(uuid.uuid4()), ctx)
                store.append_review_event(rid, {"type": "card_visible"}, str(uuid.uuid4()), ctx)
            finally:
                store.close()
            self.assertEqual(storage.validate_copy(db)["integrity"], "ok")
            conn = sqlite3.connect(db)
            conn.execute("UPDATE review_events SET event_json='[1]'")
            conn.commit()
            conn.close()
            with self.assertRaisesRegex(storage.StorageError, "review event must be an object"):
                storage.validate_copy(db)
            conn = sqlite3.connect(db)
            conn.execute("UPDATE review_events SET event_json='{}', origin='page', operation_id=("
                         "SELECT operation_id FROM operations WHERE kind='create')")
            conn.commit()
            conn.close()
            with self.assertRaisesRegex(storage.StorageError, "operation mismatch"):
                storage.validate_copy(db)
        with tempfile.TemporaryDirectory() as root:
            db = self.make(root, table=False)
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE review_events (recording_id TEXT)")
            conn.commit()
            conn.close()
            with self.assertRaises(storage.UnsupportedSchema):
                storage.open_store(db)
            with self.assertRaises(storage.UnsupportedSchema):
                storage.validate_copy(db)


if __name__ == "__main__":
    unittest.main()
