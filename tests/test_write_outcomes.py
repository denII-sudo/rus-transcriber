"""РАР-6.1, PR 5б-2: исходы записи и соседи.

Что проверяется (режим `sqlite`, данные синтетические, всё во временной папке TEMP песочницы):
- ошибка записи чужой записи того же корня не ломает свою; своя ошибка не теряется (изоляция соседей);
- файл, занятый другой программой, не обрывает запись: исход «записано в базу, файл обновится позже»;
- правка файла между сверкой и записью: исход «записано, но файл не обновлён: его изменили во время записи»;
- один русский текст исхода во всех местах показа (`rt.py`, MCP, обе страницы), английских нет;
- файл больше 8 МиБ: когда он снова равен версии программы, конфликт закрывается без копии.
"""
import asyncio
import contextlib
import io
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import core
import documents
import recordings
import review_page
import rt
import storage
import test_conflict_replace
import test_recordings
import transcript
from test_document_write_order import Env, KINDS

REPO = Path(__file__).resolve().parent.parent
CYRILLIC = re.compile("[А-Яа-яЁё]")


def failing(names, error):
    """Подмена `documents._write_atomic`: запись файлов с этими именами падает, остальные идут как есть."""
    real = documents._write_atomic

    def write(target, data, **kwargs):
        if Path(target).name in names:
            raise error
        return real(target, data, **kwargs)
    return mock.patch.object(documents, "_write_atomic", write)


def busy(names):
    """Подмена замены файла: файл с таким именем занят другой программой (в Windows — PermissionError)."""
    real = os.replace

    def replace(source, target, *args, **kwargs):
        if Path(target).name in names:
            raise PermissionError(13, "файл занят другой программой")
        return real(source, target, *args, **kwargs)
    return mock.patch.object(documents.os, "replace", replace)


class OutcomeCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p5b2-", dir=os.environ["TEMP"])
        self.addCleanup(self.temp.cleanup)
        self.env = Env(self.temp.name)
        self.addCleanup(self.env.close)

    def two(self):
        """Две записи одного корня, у обеих версия 2 стоит в очереди. -> (первая по порядку очереди, вторая)."""
        env = self.env
        first, second = sorted((env.create(), env.create()))
        env.materialize()
        env.change(first, "два")
        env.change(second, "два")
        return first, second

    def names(self, rid):
        return {self.env.file(rid, kind).name for kind in KINDS}


class NeighboursAreIsolated(OutcomeCase):
    def test_neighbour_write_error_does_not_break_my_write_in_either_order(self):
        for mine_first in (True, False):
            with self.subTest(mine_first=mine_first):
                if not mine_first:
                    self.setUp()
                first, second = self.two()
                mine, other = (first, second) if mine_first else (second, first)
                with failing(self.names(other), OSError("диск недоступен")):
                    completed = self.env.materialize(recording_id=mine)
                self.assertIn((mine, 2), completed)
                self.assertEqual(self.env.versions(mine), {kind: 2 for kind in KINDS})
                self.assertEqual(self.env.outbox(other, 2), ("error", "io_or_interrupted"))

    def test_my_own_write_error_is_not_lost_and_the_neighbour_is_still_written(self):
        for mine_first in (True, False):
            with self.subTest(mine_first=mine_first):
                if not mine_first:
                    self.setUp()
                first, second = self.two()
                mine, other = (first, second) if mine_first else (second, first)
                with failing(self.names(mine), OSError("диск недоступен")):
                    with self.assertRaises(OSError):
                        self.env.materialize(recording_id=mine)
                self.assertEqual(self.env.outbox(mine, 2), ("error", "io_or_interrupted"))
                self.assertEqual(self.env.versions(other), {kind: 2 for kind in KINDS})

    def test_neighbour_with_a_missing_file_that_cannot_be_restored_does_not_break_my_write(self):
        env = self.env
        first, second = sorted((env.create(), env.create()))
        env.materialize()
        env.file(first, "md").unlink()          # у соседа нет файла, дописать его не удаётся
        env.change(second, "два")
        with failing(self.names(first), OSError("диск недоступен")):
            completed = env.materialize(recording_id=second)
        self.assertIn((second, 2), completed)
        self.assertEqual(env.versions(second), {kind: 2 for kind in KINDS})

    def test_neighbour_in_conflict_does_not_break_my_write(self):
        first, second = self.two()
        self.env.file(first, "md").write_bytes(b"external")
        completed = self.env.materialize(recording_id=second)
        self.assertIn((second, 2), completed)
        self.assertEqual(self.env.outbox(first, 2)[0], "conflict")

    def test_without_recording_id_the_first_error_still_stops_the_whole_queue(self):
        first, second = self.two()
        with failing(self.names(first), OSError("диск недоступен")):
            with self.assertRaises(OSError):
                self.env.materialize()


class BusyFile(OutcomeCase):
    def setUp(self):
        super().setUp()
        self.rid = self.env.create()
        self.env.materialize()
        self.env.change(self.rid, "два")

    def test_busy_file_is_not_an_exception_and_the_queue_keeps_the_revision(self):
        env, rid = self.env, self.rid
        with busy({env.file(rid, "json").name}):
            completed = env.materialize(recording_id=rid)
        self.assertEqual(completed, [])
        self.assertEqual(env.outbox(rid, 2), ("pending", "busy"))
        self.assertEqual(env.versions(rid)["md"], 2, "то, что можно записать, записано: набор допишут при повторе")
        self.assertEqual(env.versions(rid)["json"], 1)
        self.assertEqual(env.conflicts(), [])

    def test_next_call_finishes_the_revision(self):
        env, rid = self.env, self.rid
        with busy({env.file(rid, "json").name}):
            env.materialize(recording_id=rid)
        self.assertEqual(env.materialize(recording_id=rid), [(rid, 2)])
        self.assertEqual(env.outbox(rid, 2), ("synced", None))
        self.assertEqual(env.versions(rid), {kind: 2 for kind in KINDS})

    def test_busy_neighbour_does_not_break_my_write(self):
        env = self.env
        other = env.create()        # чужая запись: её первая ревизия ещё не записана, а файл занят
        with busy(self.names(other)):
            completed = env.materialize(recording_id=self.rid)
        self.assertIn((self.rid, 2), completed)
        self.assertEqual(env.outbox(other, 1), ("pending", "busy"))

    def test_the_outcome_says_which_file_will_be_updated_later(self):
        env, rid = self.env, self.rid
        with busy({env.file(rid, "json").name}):
            completed = env.materialize(recording_id=rid)
        with self.assertRaises(documents.NotSynced) as caught:
            documents.require_written(env.store, rid, 2, completed)
        self.assertEqual(caught.exception.status, "later")
        self.assertTrue(str(caught.exception).startswith("Записано в базу"))
        self.assertIn(env.file(rid, "json").name, str(caught.exception))
        self.assertNotIn(env.file(rid, "md").name, str(caught.exception))
        self.assertRegex(str(caught.exception), "позже|повтор")

    def test_a_synced_revision_is_not_an_outcome_even_if_a_neighbour_process_wrote_it(self):
        env, rid = self.env, self.rid
        env.materialize()
        documents.require_written(env.store, rid, 2, [])      # уже записано другим вызовом: исхода нет


class WrittenButFileChangedMeanwhile(unittest.TestCase):
    helper = test_recordings.WorkerSnapshotTests()

    def test_edit_between_the_check_and_the_write_is_reported_in_words(self):
        real = documents.guard_before_write

        def guard_then_edit(store, recording_id):
            real(store, recording_id)
            binding = recordings.document_binding(store, recording_id)
            (binding["root"] / binding["paths"]["md"]).write_bytes(b"external")     # правка после сверки
        with self.helper.answer_files("sqlite", self.helper.answer_result(word_ids=True)) as (jp, md):
            data = transcript.load(jp)
            data["segments"][0]["words"][0]["w"] = "правка"
            data["segments"][0]["text"] = "правка"
            with mock.patch.object(documents, "guard_before_write", guard_then_edit):
                with self.assertRaises(documents.NotSynced) as caught:
                    transcript.save(data, jp, md)
            self.assertEqual(caught.exception.status, "changed")
            self.assertTrue(str(caught.exception).startswith("Записано в базу"))
            self.assertIn(md.name, str(caught.exception))
            self.assertIn("изменили", str(caught.exception))
            self.assertEqual(md.read_bytes(), b"external", "файл человека не тронут")
            store, binding = recordings.open_for_document(jp)
            try:
                self.assertEqual(store.db.execute("SELECT count(*) FROM document_conflicts").fetchone()[0], 1)
                self.assertEqual(binding["document_revision"], 2, "ревизия в базе записана")
            finally:
                store.close()


class RestoreAnswersHonestly(unittest.TestCase):
    helper = test_recordings.WorkerSnapshotTests()

    def snapshot(self, jp, md):
        data = transcript.load(jp)
        data["segments"][0]["words"][0]["w"] = "правка"
        data["segments"][0]["text"] = "правка"
        transcript.save(data, jp, md)
        store, binding = recordings.open_for_document(jp)
        try:
            return core.prepare_restore_snapshot(binding["recording_id"], 1, 1), binding
        finally:
            store.close()

    def test_restore_with_a_busy_file_is_not_a_plain_success(self):
        with self.helper.answer_files("sqlite", self.helper.answer_result(word_ids=True)) as (jp, md):
            snapshot, binding = self.snapshot(jp, md)
            with busy({md.name}):
                result = core.restore_recording_revision(snapshot)
            self.assertIs(result["materialized"], False)
            self.assertEqual(result["outcome"], "later")
            self.assertIn(md.name, result["reason"])
            self.assertTrue(result["reason"].startswith("Записано в базу"))

    def test_restore_that_wrote_everything_says_so(self):
        with self.helper.answer_files("sqlite", self.helper.answer_result(word_ids=True)) as (jp, md):
            snapshot, binding = self.snapshot(jp, md)
            result = core.restore_recording_revision(snapshot)
            self.assertIs(result["materialized"], True)
            self.assertEqual(result["outcome"], "written")


class OneRussianTextEverywhere(unittest.TestCase):
    NOT_WRITTEN = documents.NotWritten("Не записано: файл «а.transcript.md» изменён вне программы. Ничего не сохранено.",
                                       "md", "external")
    NOT_SYNCED = documents.NotSynced("Записано в базу, но файл «а.transcript.json» обновится позже.", "later")
    OPEN_CONFLICT = storage.Conflict("unresolved document conflict")

    def test_core_translation_keeps_russian_and_translates_storage_codes(self):
        for error in (self.NOT_WRITTEN, self.NOT_SYNCED):
            self.assertEqual(core.say_error(error), str(error))
            self.assertEqual(core.explain_error(error), str(error))
        self.assertEqual(core.say_error(self.OPEN_CONFLICT), "Сначала решите конфликт внешней правки документа.")
        self.assertEqual(core.say_error(storage.Conflict("unresolved conspect conflict")),
                         "Сначала решите конфликт внешней правки свёртки.")
        self.assertEqual(core.say_error(ValueError("прочее")), "прочее", "остальные отказы как были")

    def test_rt_prints_the_text_and_not_a_traceback(self):
        for error in (self.NOT_WRITTEN, self.NOT_SYNCED, self.OPEN_CONFLICT):
            with self.subTest(error=str(error)):
                stderr = io.StringIO()
                with mock.patch.object(rt, "run", side_effect=error), contextlib.redirect_stderr(stderr):
                    code = rt.main(["settings"])
                self.assertEqual(code, 1)
                self.assertIn("Ошибка: " + core.say_error(error), stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_mcp_says_the_same_text(self):
        import server

        def raises(error):
            def run():
                raise error
            return run
        for error in (self.NOT_WRITTEN, self.NOT_SYNCED, self.OPEN_CONFLICT):
            with self.subTest(error=str(error)):
                answer = asyncio.run(server._gated(None, {}, lambda: [], raises(error)))
                self.assertEqual(answer, "Ошибка: " + core.say_error(error))
                self.assertRegex(answer, CYRILLIC)

    def test_review_page_shows_the_reason_instead_of_a_code(self):
        self.assertEqual(review_page.safe_error(self.NOT_WRITTEN), str(self.NOT_WRITTEN))
        self.assertEqual(review_page.safe_error(self.NOT_SYNCED), str(self.NOT_SYNCED))
        self.assertEqual(review_page.safe_error(self.OPEN_CONFLICT), core.say_error(self.OPEN_CONFLICT))
        self.assertIn("изменилась", review_page.safe_error(storage.Conflict("stale revisions")))

    def test_no_english_outcome_texts_are_left_in_the_code(self):
        for path in sorted(REPO.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(file=path.name):
                self.assertNotIn("materialization is pending", text)
                self.assertNotIn("SQLite commit succeeded", text)

    def test_cards_page_keeps_the_answer_in_the_field_when_the_write_is_refused(self):
        html = (REPO / "cards.html").read_text(encoding="utf-8")
        send = html[html.index("async function send("):html.index("function flash(")]
        before_request = send[:send.index("await api.answer")]
        self.assertNotIn(".value", before_request, "поле очищается только после успеха")
        self.assertNotIn("reset()", before_request)
        self.assertIn("Не сохранилось: ", send)


class OversizeConflictCloses(test_conflict_replace.ScreenCase):
    BIG = b"x" * (documents.MAX_CONFLICT_BYTES + 1)

    def over_limit_conflict(self, scope):
        case = self.conflict(scope, self.BIG)
        self.assertFalse(self.item(case)["preserved"])
        return case

    def test_file_equal_to_the_program_version_can_be_closed_without_a_copy(self):
        for scope in test_conflict_replace.SCOPES:
            with self.subTest(scope=scope):
                case = self.over_limit_conflict(scope)
                program = self.program(case.rec, scope, "md")
                case.path.write_bytes(program)
                row = self.item(case)
                self.assertEqual((row["preserved"], row["can_replace"]), (False, True))
                shown = self.preview(case)
                self.assertEqual((shown["effect"], shown["can_replace"]), ("close", True))
                result = self.replace(shown["snapshot"])
                self.assertEqual(result["outcome"], "replaced")
                self.assertEqual(case.path.read_bytes(), program, "файл не менялся")
                self.assertEqual(self.item(case)["status"], "resolved")
                self.assertEqual(self.open_rows(case.rec, scope), [])

    def test_a_closed_document_conflict_lets_the_record_be_changed_again(self):
        case = self.over_limit_conflict("document")
        case.path.write_bytes(self.program(case.rec, "document", "md"))
        self.replace(self.snapshot(case))
        self.change_text(case.rec, "новое слово")                      # открытого конфликта больше нет
        self.materialize(case.rec, "document")
        self.assertIn("новое слово".encode("utf-8"), case.path.read_bytes())

    def test_other_content_stays_refused_and_the_reason_names_the_way_out(self):
        for scope in test_conflict_replace.SCOPES:
            with self.subTest(scope=scope):
                case = self.over_limit_conflict(scope)
                shown = self.preview(case)
                self.assertEqual((shown["can_replace"], shown["snapshot"]), (False, None))
                self.assertIn("8 МиБ", shown["reason"])
                self.assertIn("закрыть", shown["reason"])
                self.refused("api/conflict/resolve", {"snapshot": self.body(case), "confirm": True})
                self.assertEqual(case.path.read_bytes(), self.BIG)

    def test_agent_route_closes_only_when_the_file_equals_the_program_version(self):
        case = self.over_limit_conflict("document")
        with self.assertRaises(core.CoreError):
            core.resolve_document_conflict(case.rid, case.id)
        self.assertEqual(case.path.read_bytes(), self.BIG)
        case.path.write_bytes(self.program(case.rec, "document", "md"))
        self.assertTrue(core.resolve_document_conflict(case.rid, case.id)["materialized"])
        self.assertEqual(self.open_rows(case.rec, "document"), [])

    def test_a_closed_conflict_never_lets_the_writer_replace_the_humans_file_later(self):
        # решение «закрыть» принято, а файл потом снова стал большим чужим: копии нет, писатель не затирает его
        case = self.over_limit_conflict("document")
        case.path.write_bytes(self.program(case.rec, "document", "md"))
        self.replace(self.snapshot(case))
        case.path.write_bytes(self.BIG)
        with self.assertRaises(documents.NotWritten):
            documents.guard_before_write(self.store, case.rid)
        self.assertEqual(case.path.read_bytes(), self.BIG)


if __name__ == "__main__":
    unittest.main()
