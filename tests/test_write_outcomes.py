"""РАР-6.1, PR 5б-2: исходы записи и соседи.

Что проверяется (режим `sqlite`, данные синтетические, всё во временной папке TEMP песочницы):
- ошибка записи чужой записи того же корня не ломает свою; своя ошибка не теряется (изоляция соседей);
- файл, который не удалось заменить (занят другой программой или защищён от записи), не обрывает запись:
  исход «записано в базу, файл обновится позже», занятая ревизия держит следующие ревизии той же записи;
- восстановление из резерва называет ревизии, файлы которых ещё не записаны;
- правка файла между сверкой и записью: исход «записано, но файл не обновлён: его изменили во время записи»;
- один русский текст исхода во всех местах показа (`rt.py`, MCP, обе страницы), английских нет;
- файл больше 8 МиБ: когда он снова равен версии программы, конфликт закрывается без копии.
"""
import asyncio
import contextlib
import http.client
import io
import json
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

import core
import documents
import recordings
import review_page
import rt
import storage
import test_conflict_replace
import test_p5_screen
import test_recordings
import transcript
from test_document_write_order import Env, KINDS
from urllib.parse import urlsplit

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

    def test_my_own_missing_file_that_cannot_be_restored_is_not_lost(self):
        env = self.env
        mine, other = env.create(), env.create()
        env.materialize()
        env.file(mine, "md").unlink()
        with failing(self.names(mine), OSError("диск недоступен")):
            with self.assertRaises(OSError):
                env.materialize(recording_id=mine)

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

    def test_a_busy_revision_holds_back_the_next_revision_of_the_same_recording(self):
        env, rid = self.env, self.rid
        env.change(rid, "три")                                 # в очереди ревизии 2 и 3
        json_name = env.file(rid, "json").name
        real, refused = os.replace, []

        def replace(source, target, *args, **kwargs):
            if Path(target).name == json_name and not refused:   # занят один раз, на ревизии 2
                refused.append(target)
                raise PermissionError(13, "файл занят другой программой")
            return real(source, target, *args, **kwargs)
        with mock.patch.object(documents.os, "replace", replace):
            self.assertEqual(env.materialize(recording_id=rid), [])
        self.assertEqual(env.outbox(rid, 2), ("pending", "busy"))
        self.assertEqual(env.outbox(rid, 3), ("pending", None), "ревизия 3 не обгоняет занятую ревизию 2")
        self.assertEqual(env.versions(rid), {"md": 2, "json": 1, "manifest": 1})
        self.assertEqual(env.materialize(recording_id=rid), [(rid, 2), (rid, 3)])
        self.assertEqual(env.versions(rid), {kind: 3 for kind in KINDS})

    def test_the_text_names_both_causes_and_both_ways_out(self):
        # PermissionError при замене — файл занят другой программой или защищён от записи («только чтение», права)
        env, rid = self.env, self.rid
        with busy({env.file(rid, "json").name}):
            completed = env.materialize(recording_id=rid)
        text = str(documents.unsynced(env.store, rid, 2, completed))
        for words in ("занят другой программой", "защищён от записи", "Закройте файл", "снимите защиту от записи"):
            self.assertIn(words, text)
        self.assertNotIn("не дошла", text, "общая фраза — для очереди без кода busy")

    @unittest.skipUnless(os.name == "nt", "атрибут «только чтение» мешает замене файла только в Windows")
    def test_a_read_only_file_is_not_called_only_busy(self):
        env, rid = self.env, self.rid
        target = env.file(rid, "json")
        os.chmod(target, stat.S_IREAD)                          # настоящий атрибут, без подмены os.replace
        self.addCleanup(os.chmod, target, stat.S_IREAD | stat.S_IWRITE)
        completed = env.materialize(recording_id=rid)
        self.assertEqual(env.outbox(rid, 2), ("pending", "busy"))
        self.assertIn("защищён от записи", str(documents.unsynced(env.store, rid, 2, completed)))
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
        self.assertEqual(env.materialize(recording_id=rid), [(rid, 2)])
        self.assertEqual(env.versions(rid), {kind: 2 for kind in KINDS})


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


class CardsPageWriteOutcome(unittest.TestCase):
    helper = test_recordings.WorkerSnapshotTests()

    def test_card_answer_with_a_busy_file_is_recorded_and_says_the_file_follows_later(self):
        import review_pack
        question = {"id": "r1", "question": "Проверьте", "options": [
            {"label": "Другое слово", "decision": {"text": "другое"}}]}
        with self.helper.answer_files("sqlite", self.helper.answer_result(word_ids=True)) as (jp, md):
            pack = review_pack.build(jp, [question])
            row = {"id": "r1", "version": pack["version"], "text": "другое", "heard": False}
            with busy({md.name}):
                with self.assertRaises(documents.NotSynced) as caught:
                    review_pack.add_answers(jp, [row], "local", trusted_actor="human")
            self.assertEqual(caught.exception.status, "later")
            self.assertEqual(review_page.safe_error(caught.exception), str(caught.exception))


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


class RestoreFromBackupNamesWhatIsNotWritten(unittest.TestCase):
    """Без recording_id занятый файл больше не обрывает восстановление: человек должен узнать, что не записано."""
    setUp = test_p5_screen.P5ServiceTests.setUp
    tearDown = test_p5_screen.P5ServiceTests.tearDown
    make_recording = test_p5_screen.P5ServiceTests.make_recording
    change_text = test_p5_screen.P5ServiceTests.change_text

    def restore(self, names=(), later_revisions=0):
        ident, root_id, doc_paths = self.make_recording()
        for number in range(later_revisions):      # ревизии за первой: они ждут её, пока файл занят
            self.change_text(ident, root_id, f"правка{number}")
        backup = self.base / "backup.sqlite3"
        core.backup_store(backup)
        user_id = self.store.current_identity()["user_id"]
        new_root = self.base / "restored-docs"
        new_root.mkdir()
        with busy({Path(doc_paths[kind]).name for kind in names}):
            report = core.restore_from_backup(backup, self.base / "restored.sqlite3", {root_id: new_root}, user_id)
        return ident, report

    def test_restore_with_a_busy_file_names_the_revision_left_unwritten(self):
        ident, report = self.restore(["json"])
        self.assertEqual(report["materialized"], [])
        self.assertEqual([(row["recording_id"], row["error_code"]) for row in report["not_written"]],
                         [(ident, "busy")])
        first = report["warning"][0]
        self.assertIn("не все файлы", first)
        self.assertIn("защищён от записи", first)
        self.assertIn("sync_recording_documents", first)
        self.assertTrue(any(ident in line for line in report["warning"][1:]))
        self.assertTrue(report["warning"][1].endswith("файл занят другой программой или защищён от записи"),
                        report["warning"][1])

    def test_warning_lines_tell_the_busy_revision_from_the_one_waiting_behind_it(self):
        ident, report = self.restore(["json"], later_revisions=1)
        rows = sorted(report["not_written"], key=lambda row: row["document_revision"])
        self.assertEqual([row["recording_id"] for row in rows], [ident, ident])
        busy_row, waiting_row = rows
        self.assertEqual(busy_row["error_code"], "busy")
        self.assertNotEqual(waiting_row["error_code"], "busy")
        lines = report["warning"][1:3]
        self.assertEqual(len(lines), 2, report["warning"])
        busy_line, waiting_line = (next(x for x in lines if f"ревизия {row['document_revision']}:" in x)
                                   for row in (busy_row, waiting_row))
        self.assertTrue(busy_line.endswith("файл занят другой программой или защищён от записи"), busy_line)
        self.assertTrue(waiting_line.endswith("ждёт предыдущую ревизию этой записи"), waiting_line)
        self.assertNotIn("занят", waiting_line)
        self.assertNotIn("ждёт предыдущую", busy_line)

    def test_restore_that_wrote_everything_has_no_such_warning(self):
        ident, report = self.restore()
        self.assertEqual(report["not_written"], [])
        self.assertNotIn("warning", report)


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

    def test_cards_page_apply_shows_the_outcome_and_not_the_code(self):
        # путь «Применить ответы»: core.apply_answers_files отдаёт исход как CoreError (тип прежнего отказа)
        message = str(self.NOT_SYNCED)
        with tempfile.TemporaryDirectory(prefix="p5b2-page-", dir=os.environ["TEMP"]) as temp:
            root = Path(temp)
            pack = root / "demo.review"
            (pack / "clips").mkdir(parents=True)
            (pack / "manifest.json").write_text(json.dumps({"version": "v1", "audio": "demo.wav", "items": [
                {"id": "r1", "t": "00:00:01", "clip": "clips/r1.mp3", "text": "да", "speaker": "Участник",
                 "t1": 0.0, "t2": 0.0, "question": "Как?", "options": [{"label": "да", "decision": {}}]}]}),
                encoding="utf-8")
            status, data, tmp = root / "status.json", root / "data", root / "temp"
            data.mkdir()
            tmp.mkdir()
            env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=str(tmp), TMP=str(tmp),
                       PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
            code = ("import sys, core, review_page\n"
                    "message = sys.argv[4]\n"
                    "def refused(*a, **k):\n"
                    "    raise core.CoreError(message)\n"
                    "core.apply_answers_files = refused\n"
                    "sys.argv = ['review_page.py', sys.argv[1], '--status', sys.argv[2], '--markdown', sys.argv[3],"
                    " '--port', sys.argv[5]]\n"
                    "review_page.main()\n")
            with socket.socket() as probe:      # свободный порт: постоянный мог быть занят другой программой
                probe.bind(("127.0.0.1", 0))
                port = min(probe.getsockname()[1], 65535 - 20)      # bind_free_port перебирает 20 портов вверх
            proc = subprocess.Popen([sys.executable, "-B", "-c", code, str(root / "demo.transcript.json"),
                                     str(status), str(root / "demo.transcript.md"), message, str(port)],
                                    cwd=str(REPO), env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                url = None
                for _ in range(100):
                    try:
                        url = json.loads(status.read_text(encoding="utf-8")).get("url")
                    except (OSError, ValueError):
                        url = None
                    if url or proc.poll() is not None:
                        break
                    time.sleep(0.1)
                self.assertTrue(url, "тестовая страница не запустилась")
                address = urlsplit(url)
                conn = http.client.HTTPConnection("127.0.0.1", address.port, timeout=10)
                try:
                    conn.request("POST", address.path + "api/apply", body=json.dumps({"version": "v1"}).encode(),
                                 headers={"Origin": f"http://127.0.0.1:{address.port}",
                                          "Content-Type": "application/json"})
                    response = conn.getresponse()
                    error = json.loads(response.read().decode("utf-8"))["error"]
                finally:
                    conn.close()
            finally:
                proc.terminate()
                proc.wait(timeout=10)
        self.assertEqual(response.status, 400)
        self.assertEqual(error, message)
        self.assertNotIn("код:", error)

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


class McpWritingToolsSpeakRussian(unittest.TestCase):
    """Инструменты MCP, которые меняют запись: отказ хранилища — строкой ответа по-русски, не исключением."""
    helper = test_recordings.WorkerSnapshotTests()

    @contextlib.contextmanager
    def changed_file(self):
        with self.helper.answer_files("sqlite", self.helper.answer_result(word_ids=True)) as (jp, md):
            word = transcript.load(jp)["segments"][0]["words"][0]
            md.write_bytes(b"human edit")                          # правка человека рядом с аудио
            with mock.patch.object(core, "_done_json", return_value={"json": str(jp), "markdown": str(md)}):
                yield word, md

    def test_edit_span_with_an_open_conflict_answers_in_russian(self):
        import server
        with self.changed_file() as (word, md):
            first = server.edit_span("job", word["s"], word["e"], "правка", "", "", "claude")
            second = server.edit_span("job", word["s"], word["e"], "правка два", "", "", "claude")
        self.assertTrue(first.startswith("Не записано"), first)
        self.assertEqual(second, "Сначала решите конфликт внешней правки документа.")

    def test_reopen_review_returns_the_outcome_instead_of_raising(self):
        import server
        with self.changed_file() as (word, md):
            server.edit_span("job", word["s"], word["e"], "правка", "", "", "claude")    # конфликт открыт
            answer = server.reopen_review("job", "r1")
        self.assertTrue(answer.startswith("Записано в базу, но"), answer)
        self.assertIn(md.name, answer)

    def test_other_writing_tools_answer_storage_refusals_in_words(self):
        import server
        calls = {"legacy_speaker_review": lambda: server.legacy_speaker_review("job", True),
                 "build_review_pack": lambda: server.build_review_pack("job"),
                 "reopen_review": lambda: server.reopen_review("job", "r1"),
                 "edit_span": lambda: server.edit_span("job", 0.0, 1.0, "слово", "", "", "claude")}
        for name, call in calls.items():
            for error in (OneRussianTextEverywhere.NOT_SYNCED, OneRussianTextEverywhere.OPEN_CONFLICT):
                with self.subTest(tool=name, error=str(error)):
                    with mock.patch.object(core, name, side_effect=error):
                        answer = call()
                    self.assertEqual(answer, core.say_error(error))
                    self.assertRegex(answer, CYRILLIC)


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
                self.assertEqual((self.item(case)["can_replace"], self.item(case)["status"]), (False, "open"))
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

    def test_the_store_refuses_a_conflict_without_a_copy_unless_asked(self):
        # охрана не только в core: само хранилище без without_copy не закрывает конфликт без копии
        for scope in test_conflict_replace.SCOPES:
            with self.subTest(scope=scope):
                case = self.over_limit_conflict(scope)
                case.path.write_bytes(self.program(case.rec, scope, "md"))     # даже когда файл уже равен версии
                resolve = getattr(self.store, f"resolve_{scope}_conflict")
                with self.assertRaises(storage.Conflict) as caught:
                    resolve(case.id, str(uuid.uuid4()), recordings.operation_context(self.store, "human"))
                self.assertEqual(str(caught.exception), "external file too large to preserve; manual resolution required")
                self.assertEqual(self.item(case)["status"], "open")

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
