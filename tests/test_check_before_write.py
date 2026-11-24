"""РАР-6.1, PR 5б-1: файлы записи сверяются до записи в базу, а не после.

Раньше новая ревизия сначала попадала в базу, а ручная правка файла находилась уже при записи файла: база новая,
файл ручной. Теперь `documents.guard_before_write` сверяет три текущих файла до записи; при ручной правке ничего
не записывается (ревизии и очередь не меняются), конфликт регистрируется один раз, вызывающий получает отказ
`documents.NotWritten` с именем файла. Правило допустимости файла — одна функция `documents.file_verdict`.
Режим — `sqlite`. Данные синтетические, всё во временной папке (TEMP песочницы).
"""
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import core
import documents
import recordings
import review_pack
import test_recordings
import transcript
from test_document_write_order import Env, KINDS, REVIEW, uid
from test_document_write_order import transcript as doc_transcript

TABLES = ("transcript_revisions", "review_revisions", "document_versions", "document_outbox")


def tree(base):
    """Все файлы и папки под base: путь -> байты (у папки None)."""
    found = {}
    for path in sorted(Path(base).rglob("*")):
        found[path.relative_to(base).as_posix()] = path.read_bytes() if path.is_file() else None
    return found


def counts(store):
    return {name: store.db.execute(f"SELECT count(*) FROM {name}").fetchone()[0] for name in TABLES}


def head_state(store, rid):
    return store.db.execute("SELECT state FROM document_outbox WHERE recording_id=? "
                            "ORDER BY document_revision DESC LIMIT 1", (rid,)).fetchone()[0]


class FileVerdictTests(unittest.TestCase):
    """Единственное правило допустимости текущего файла: чистая функция без чтения диска и базы."""

    @staticmethod
    def override(content=b"human", *, damaged=False):
        return {"observed_hash": documents._digest(content), "observed_size": len(content),
                "observed_bytes": b"damaged" if damaged else content}

    def test_table(self):
        new, old, other = documents._digest(b"new"), documents._digest(b"old"), documents._digest(b"other")
        human = self.override()
        cases = [
            ((new, new, old, None), "ok"),
            ((old, new, old, None), "stale"),
            ((None, new, old, None), "missing"),
            ((None, new, None, None), "missing"),
            ((other, new, old, None), "external"),
            ((other, new, None, None), "external"),
            ((human["observed_hash"], new, old, human), "accepted"),
            ((new, new, old, human), "ok"),
            ((other, new, old, human), "external"),
            ((old, new, old, human), "external"),      # предыдущая версия допустима только без принятого решения
            ((None, new, old, human), "gone_after_confirmation"),
            ((human["observed_hash"], new, old, self.override(damaged=True)), "damaged_copy"),
            ((new, new, old, self.override(damaged=True)), "ok"),   # файл уже равен версии программы: копия не нужна
        ]
        for number, (args, expected) in enumerate(cases):
            with self.subTest(case=number, expected=expected):
                self.assertEqual(documents.file_verdict(*args), expected)


class SyncedRecording(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p5b-", dir=os.environ["TEMP"])
        self.addCleanup(self.temp.cleanup)
        self.env = self.new_env("env")
        self.rid = self.env.create()
        self.env.materialize()

    def new_env(self, label):
        env = Env(self.temp.name, label)
        self.addCleanup(env.close)
        return env

    @staticmethod
    def refusing(suffix):
        """Подмена чтения хэша: файл с таким окончанием «не читается»."""
        real = documents._file_hash

        def refuse(path):
            if Path(path).name.endswith(suffix):
                raise PermissionError("busy")
            return real(path)
        return refuse


class CheckRecordingFilesTests(SyncedRecording):
    def test_reports_every_file_and_writes_nothing(self):
        env, rid = self.env, self.rid
        env.file(rid, "md").write_bytes(b"external")
        env.file(rid, "json").unlink()
        before, db = tree(env.base), counts(env.store)
        self.assertEqual(documents.check_recording_files(env.store, rid),
                         {"md": "external", "json": "missing", "manifest": "ok"})
        self.assertEqual(tree(env.base), before)
        self.assertEqual(counts(env.store), db)
        self.assertEqual(env.conflicts(), [])

    def test_unwritten_next_revision_is_not_an_external_edit(self):
        env, rid = self.env, self.rid
        env.change(rid, "два")        # версия 2 стоит в очереди, файлы ещё версии 1
        self.assertEqual(documents.check_recording_files(env.store, rid), {kind: "ok" for kind in KINDS})
        env.file(rid, "md").write_bytes(b"external")
        self.assertEqual(documents.check_recording_files(env.store, rid)["md"], "external")

    def test_directory_and_unreadable_file_are_named(self):
        env, rid = self.env, self.rid
        md = env.file(rid, "md")
        md.unlink()
        md.mkdir()
        with mock.patch.object(documents, "_file_hash", self.refusing(".transcript.json")):
            self.assertEqual(documents.check_recording_files(env.store, rid),
                             {"md": "invalid_target", "json": "unreadable", "manifest": "ok"})


class GuardBeforeWriteTests(SyncedRecording):
    def test_external_edit_of_any_file_registers_one_conflict_and_refuses(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                env = self.new_env("env-" + kind)
                rid = env.create()
                env.materialize()
                env.file(rid, kind).write_bytes(b"external")
                db = counts(env.store)
                with self.assertRaises(documents.NotWritten) as caught:
                    documents.guard_before_write(env.store, rid)
                self.assertIn(env.file(rid, kind).name, str(caught.exception))
                self.assertEqual(counts(env.store), db)
                self.assertEqual(env.conflicts(), [(rid, kind)])
                self.assertEqual(head_state(env.store, rid), "conflict")
                self.assertEqual(env.file(rid, kind).read_bytes(), b"external")
                documents.guard_before_write(env.store, rid)      # повтор: конфликт один, запись закроет хранилище
                self.assertEqual(env.conflicts(), [(rid, kind)])

    def test_unreadable_first_file_refuses_without_conflict_for_it_or_for_the_second(self):
        env, rid = self.env, self.rid
        env.file(rid, "json").write_bytes(b"external")
        db = counts(env.store)
        with mock.patch.object(documents, "_file_hash", self.refusing(".transcript.md")):
            with self.assertRaises(documents.NotWritten) as caught:
                documents.guard_before_write(env.store, rid)
        self.assertIn(env.file(rid, "md").name, str(caught.exception))
        self.assertEqual((env.conflicts(), counts(env.store), head_state(env.store, rid)), ([], db, "synced"))

    def test_directory_instead_of_file_refuses_without_conflict(self):
        env, rid = self.env, self.rid
        md = env.file(rid, "md")
        md.unlink()
        md.mkdir()
        db = counts(env.store)
        with self.assertRaises(documents.NotWritten):
            documents.guard_before_write(env.store, rid)
        self.assertEqual((env.conflicts(), counts(env.store)), ([], db))

    def test_link_instead_of_file_is_refused_before_any_read(self):
        env, rid = self.env, self.rid
        reads = []
        real = documents._file_digest

        def spy(path):
            reads.append(Path(path).name)
            return real(path)
        with mock.patch.object(documents, "_not_reparse", side_effect=ValueError("reparse")), \
                mock.patch.object(documents, "_file_digest", spy):
            with self.assertRaises(documents.NotWritten):
                documents.guard_before_write(env.store, rid)
        self.assertEqual((reads, env.conflicts()), ([], []))

    def test_edit_seen_only_before_taking_the_lock_is_not_a_conflict(self):
        # Сосед-процесс дописывал версию 2, пока шла первая сверка; под блокировкой файлы уже согласованы.
        env, rid = self.env, self.rid
        real = documents.check_recording_files
        calls = []

        def stale_first(store, recording_id):
            calls.append(1)
            if len(calls) == 1:
                env.change(rid, "два")
                env.materialize()
                return {"md": "external", "json": "ok", "manifest": "ok"}
            return real(store, recording_id)
        with mock.patch.object(documents, "check_recording_files", stale_first):
            documents.guard_before_write(env.store, rid)
        self.assertEqual(len(calls), 2)
        self.assertEqual((env.conflicts(), head_state(env.store, rid)), ([], "synced"))

    def test_missing_file_is_not_a_reason_to_refuse(self):
        env, rid = self.env, self.rid
        env.file(rid, "json").unlink()
        documents.guard_before_write(env.store, rid)
        self.assertEqual((env.conflicts(), head_state(env.store, rid)), ([], "synced"))


class ReturnedFileAndResolvedConflict(SyncedRecording):
    """Два обычных жизненных пути, которые различает режим проверки: «синхронизированная ревизия» и «решённый конфликт»."""

    def test_older_version_bytes_in_a_synced_revision_are_an_external_edit(self):
        env, rid = self.env, self.rid
        env.change(rid, "два")
        env.materialize()
        env.file(rid, "md").write_bytes(env.store.get_document(rid, 1)["md"])     # человек вернул файл версии 1
        self.assertEqual(documents.check_recording_files(env.store, rid)["md"], "external")
        with self.assertRaises(documents.NotWritten):
            documents.guard_before_write(env.store, rid)
        self.assertEqual(env.conflicts(), [(rid, "md")])

    def test_resolved_conflict_lets_the_write_pass_while_the_file_is_still_the_human_one(self):
        env, rid = self.env, self.rid
        env.file(rid, "md").write_bytes(b"external")
        with self.assertRaises(documents.NotWritten):
            documents.guard_before_write(env.store, rid)
        env.resolve(rid)                   # человек решил: заменить версией программы; файл пока ещё его
        self.assertEqual(documents.check_recording_files(env.store, rid), {kind: "ok" for kind in KINDS})
        documents.guard_before_write(env.store, rid)
        self.assertEqual(env.conflicts(), [(rid, "md")])      # нового конфликта нет

    def test_file_gone_after_confirmation_is_refused_without_a_new_conflict(self):
        env, rid = self.env, self.rid
        env.file(rid, "md").write_bytes(b"external")
        with self.assertRaises(documents.NotWritten):
            documents.guard_before_write(env.store, rid)
        env.resolve(rid)
        env.file(rid, "md").unlink()
        self.assertEqual(documents.check_recording_files(env.store, rid)["md"], "gone")
        with self.assertRaises(documents.NotWritten) as caught:
            documents.guard_before_write(env.store, rid)
        self.assertEqual(caught.exception.status, "gone")
        self.assertEqual(env.conflicts(), [(rid, "md")])

    def test_damaged_copy_of_the_decision_is_refused_in_words_without_a_new_conflict(self):
        env, rid = self.env, self.rid
        env.file(rid, "md").write_bytes(b"external")
        with self.assertRaises(documents.NotWritten):
            documents.guard_before_write(env.store, rid)
        env.resolve(rid)
        for found in env.store.db.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='document_conflicts'").fetchall():
            env.store.db.execute(f'DROP TRIGGER "{found[0]}"')
        env.store.db.execute("UPDATE document_conflicts SET observed_bytes=?", (b"damaged",))
        with self.assertRaises(documents.NotWritten) as caught:
            documents.guard_before_write(env.store, rid)
        self.assertEqual(caught.exception.status, "damaged_copy")
        self.assertTrue(str(caught.exception).startswith("Не записано"))
        self.assertEqual(env.conflicts(), [(rid, "md")])


class RegistrationRunsUnderTheRootLock(SyncedRecording):
    def hold_lock(self, seconds=None):
        """Держать блокировку корня в другом потоке; -> событие «отпустить» (автоотпуск через seconds)."""
        held, release = threading.Event(), threading.Event()

        def holder():
            with documents._file_lock(documents.checked_root(self.env.root)):
                held.set()
                release.wait(30)
        thread = threading.Thread(target=holder)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(release.set)
        self.assertTrue(held.wait(10))
        if seconds is not None:
            timer = threading.Timer(seconds, release.set)
            timer.start()
            self.addCleanup(timer.cancel)
        return release

    def test_registration_waits_for_the_lock(self):
        env, rid = self.env, self.rid
        env.file(rid, "md").write_bytes(b"external")
        self.hold_lock(seconds=1.0)
        started = time.monotonic()
        with self.assertRaises(documents.NotWritten) as caught:
            documents.guard_before_write(env.store, rid)
        self.assertGreaterEqual(time.monotonic() - started, 0.8)
        self.assertTrue(str(caught.exception).startswith("Не записано"))
        self.assertEqual(env.conflicts(), [(rid, "md")])

    def test_clean_files_and_the_pure_check_do_not_wait_for_the_lock(self):
        env, rid = self.env, self.rid
        self.hold_lock()
        started = time.monotonic()
        documents.guard_before_write(env.store, rid)
        env.file(rid, "md").write_bytes(b"external")
        self.assertEqual(documents.check_recording_files(env.store, rid)["md"], "external")
        self.assertLess(time.monotonic() - started, 0.8)
        self.assertEqual(env.conflicts(), [])

    def test_failure_to_wait_for_the_lock_is_a_refusal_not_an_oserror(self):
        env, rid = self.env, self.rid
        env.file(rid, "md").write_bytes(b"external")
        db = counts(env.store)
        with mock.patch.object(documents, "_file_lock", side_effect=OSError("timeout")):
            with self.assertRaises(documents.NotWritten) as caught:
                documents.guard_before_write(env.store, rid)
        self.assertEqual(caught.exception.status, "locked")
        self.assertTrue(str(caught.exception).startswith("Не записано"))
        self.assertEqual((env.conflicts(), counts(env.store), head_state(env.store, rid)), ([], db, "synced"))

    def test_conflict_registered_by_a_neighbour_meanwhile_is_not_registered_twice(self):
        env, rid = self.env, self.rid
        real = documents.check_recording_files
        calls = []

        def neighbour_registers_first(store, recording_id):
            calls.append(1)
            if len(calls) == 1:
                target = env.file(rid, "md")
                target.write_bytes(b"external")
                documents._record_conflict(env.store, env.root_id, rid, 1, documents._FileConflict("md", target))
                return {"md": "external", "json": "ok", "manifest": "ok"}
            return real(store, recording_id)
        with mock.patch.object(documents, "check_recording_files", neighbour_registers_first):
            documents.guard_before_write(env.store, rid)
        self.assertEqual(env.conflicts(), [(rid, "md")])

    def test_unreadable_file_when_reading_it_for_the_conflict_is_a_refusal_without_conflict(self):
        env, rid = self.env, self.rid
        env.file(rid, "md").write_bytes(b"external")
        with mock.patch.object(documents, "_FileConflict", side_effect=PermissionError("busy")):
            with self.assertRaises(documents.NotWritten) as caught:
                documents.guard_before_write(env.store, rid)
        self.assertEqual(caught.exception.status, "unreadable")
        self.assertEqual((env.conflicts(), head_state(env.store, rid)), ([], "synced"))


class CheckCreatesNothing(SyncedRecording):
    def test_missing_folder_of_the_files_is_not_created(self):
        env = self.new_env("nested")
        rid = uid()
        nested = {kind: "docs/" + name for kind, name in documents.current_paths(rid).items()}
        env.store.create_recording_with_document(
            rid, "tag", doc_transcript("один"), REVIEW, uid(), env.ctx, root_id=env.root_id, generated_at="one",
            document_paths=nested)
        env.revision[rid] = 1
        env.materialize()
        shutil.rmtree(env.root / "docs")
        before = tree(env.base)
        self.assertEqual(documents.check_recording_files(env.store, rid), {kind: "missing" for kind in KINDS})
        documents.guard_before_write(env.store, rid)
        self.assertFalse((env.root / "docs").exists())
        self.assertEqual(tree(env.base), before)


class RootNotConfirmed(SyncedRecording):
    def test_unconfirmed_root_is_refused_with_the_next_step_for_the_human(self):
        env, rid = self.env, self.rid
        env.store.db.execute("UPDATE allowed_roots SET requires_remap=1")
        db = counts(env.store)
        with self.assertRaises(documents.NotWritten) as caught:
            documents.guard_before_write(env.store, rid)
        text = str(caught.exception)
        self.assertTrue(text.startswith("Не записано"))
        self.assertIn("верните её на прежнее место", text)
        self.assertIn("Если база восстановлена из резерва — завершите восстановление: rt.py db-restore.", text)
        self.assertEqual((env.conflicts(), counts(env.store)), ([], db))


class EveryWritePathChecksFirst(unittest.TestCase):
    """Четыре пути изменения: сохранение стенограммы, применение ответов, страница карточек, возврат ревизии."""

    helper = test_recordings.WorkerSnapshotTests()

    def prepared(self, name, jp, md):
        """Всё, что идёт до ручной правки файла; результат — вызов самого пути изменения."""
        question = {"id": "r1", "question": "Проверьте", "options": [
            {"label": "Другое слово", "decision": {"text": "другое"}}]}
        if name == "save":
            return lambda: self.edit_word(jp, md)
        if name == "restore":
            self.edit_word(jp, md)
            store, binding = recordings.open_for_document(jp)
            try:
                snapshot = core.prepare_restore_snapshot(binding["recording_id"], 1, 1)
            finally:
                store.close()
            return lambda: core.restore_recording_revision(snapshot)
        pack = review_pack.build(jp, [question])
        row = {"id": "r1", "version": pack["version"], "text": "другое", "heard": False}
        if name == "card":
            return lambda: review_pack.add_answers(jp, [row], "local", trusted_actor="human")
        review_pack.add_answers(jp, [row], "local", trusted_actor="human")
        return lambda: core.apply_answers_files(jp, md)

    @staticmethod
    def edit_word(jp, md):
        data = transcript.load(jp)
        data["segments"][0]["words"][0]["w"] = "правка"
        data["segments"][0]["text"] = "правка"
        transcript.save(data, jp, md)

    def test_manual_edit_of_any_file_stops_every_path_before_the_database(self):
        for name in ("save", "card", "answers", "restore"):
            for kind in KINDS:
                with self.subTest(path=name, kind=kind):
                    with self.helper.answer_files("sqlite", self.helper.answer_result(word_ids=True)) as (jp, md):
                        run = self.prepared(name, jp, md)
                        store, binding = recordings.open_for_document(jp)
                        try:
                            target = binding["root"] / binding["paths"][kind]
                            target.write_bytes(b"external")

                            def others():
                                return {k: (binding["root"] / p).read_bytes()
                                        for k, p in binding["paths"].items() if k != kind}
                            before, db = others(), counts(store)
                            with self.assertRaises(documents.NotWritten) as caught:
                                run()
                            self.assertIn(target.name, str(caught.exception))
                            self.assertEqual(counts(store), db)
                            self.assertEqual([row[0] for row in store.db.execute(
                                "SELECT kind FROM document_conflicts")], [kind])
                            self.assertEqual(target.read_bytes(), b"external")
                            self.assertEqual(others(), before)
                        finally:
                            store.close()

    def test_missing_file_is_restored_by_the_write(self):
        with self.helper.answer_files("sqlite", self.helper.answer_result(word_ids=True)) as (jp, md):
            md.unlink()
            self.edit_word(jp, md)
            self.assertIn("правка", md.read_text(encoding="utf-8"))
            store, binding = recordings.open_for_document(jp)
            try:
                self.assertEqual(store.db.execute("SELECT count(*) FROM document_conflicts").fetchone()[0], 0)
                self.assertTrue(documents.verify_files(store, binding["root_id"], binding["root"])["consistent"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
