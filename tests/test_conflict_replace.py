"""Конфликт внешней правки: копия человека и явная замена файла версией программы.

Проверяется только наблюдаемое: байты файлов, строки базы, ответы локального экрана и MCP.

Договор HTTP (адреса после секретного префикса; все POST — JSON с Origin экрана):

GET  api/conflicts/all -> {available, reason, items}
     Нет файла базы: 200, available=false, reason по-русски, items=[]; ничего не создаётся.
     items — открытые и решённые конфликты всей базы:
       conflict_id, scope ("document"|"conspect"), recording_id, title (по имени записи),
       name (у свёртки), kind (вид файла), relative_path, file (полный путь), observed_size,
       preserved (bool), status ("open"|"not_written"|"resolved"), can_replace (bool), can_retry (bool).
     not_written — решён, это последний конфликт этого файла, очередь записи объекта не доведена,
       а в файле не последняя версия программы. can_retry — not_written и очередь объекта не ждёт
       решения другого конфликта.

POST api/conflict/preview {recording_id, conflict_id, kind: scope}; ничего не меняет
     -> прежние поля (conflict_id, kind, relative_path, observed_hash, observed_size, preserved,
        preview, truncated) и
        file;
        file_state: "program" — равен последней версии программы | "same" — равен копии | "changed" |
                    "missing" — файла нет | "unavailable" — не читается, не файл, ссылка;
        removed, added — строки копии, которые пропадут, и строки последней версии программы, которые
                    появятся (по порядку, без переводов строк). Сравнение учитывает место строки:
                    перестановка и перенос — тоже потеря. Длинную строку сокращать можно лишь так,
                    чтобы изменённое место осталось видно;
        removed_total, added_total, shown_all — строки и списки можно сокращать, тогда shown_all=false;
                    слишком долгое сравнение можно огрубить, но не занижая число пропадающих строк;
        newlines_only — отличаются только CRLF/CR/LF и перевод строки в конце;
        also_replaces — relative_path других файлов того же объекта, которые заменятся этим же
                    подтверждением по прежнему решению;
        can_replace, reason, snapshot (null, если заменять нельзя).

POST api/conflict/copy {recording_id, conflict_id, kind: scope}; ничего не меняет
     -> сохранённые байты целиком: application/octet-stream, Content-Disposition attachment
        с filename*=UTF-8''<имя с прежним расширением>. Байты не сохранены — 400.

POST api/conflict/resolve {snapshot, confirm: true}
     -> 400, конфликт остаётся открытым: нет подтверждения или снимка; снимок изменён или устарел
        (сам конфликт, последняя версия программы); файла нет, он не читается, не обычный файл или
        путь стал ссылкой; архив; копия не сохранена.
        Изменение файла после показа отказом не является: решение записывается, файл остаётся цел.
     -> 200 {outcome, new_conflict_id, reason}:
        replaced — в файле последняя версия программы (даже если не записался другой файл);
        changed — появился открытый конфликт этого же файла (new_conflict_id);
        not_written — всё остальное после записи решения (reason по-русски).

POST api/conflict/retry {recording_id, conflict_id, kind: scope}
     -> открытый или чужой конфликт — 400; иначе только прежняя запись файлов, без новых ревизий
        и решений; ответ как у resolve.

Причина not_written называет настоящую причину; при журнале свёртки — выход «вернуть копию на
место и повторить». Отказ и причина «файла нет» называют имя, под которым нужно положить копию.

MCP: окна двух resolve_*_conflict говорят «заменить файл»; нет файла — отказ до записи решения.
Остальное у MCP прежнее: состав ответа {recording_id, document_revision | conspect_id, revision;
materialized}, materialized — ревизия дописана целиком, сбой записи — «Ошибка: …» с причиной.
"""
import asyncio
import gc
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import unquote, urlsplit

import confirm
import core
import documents
import migrate
import paths
import recordings
import settings
import storage
import workflow_page

with mock.patch.object(migrate, "run"):
    import server

PROJECT = Path(__file__).resolve().parent.parent
SCOPES = ("document", "conspect")
REVISION_TABLES = ("transcript_revisions", "review_revisions", "document_versions", "conspect_revisions")
CYRILLIC = re.compile("[А-Яа-яЁё]")
LATE_EDIT = "правка, сохранённая после показа\n".encode("utf-8")


def same_path(left, right):
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def only_in(left, right):
    """Строки left, которых нет в right; в подготовленных здесь текстах это и есть различие."""
    def lines(data):
        found = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        return found[:-1] if found[-1] == "" else found
    other = set(lines(right))
    return [line for line in lines(left) if line not in other]


def edited(data):
    return data.replace("альфа".encode("utf-8"), "альфа, моя правка".encode("utf-8"))


def failing_replace(target):
    """Файл занят: не удаётся замена именно этого файла, остальные пишутся как обычно."""
    original = os.replace

    def replace(src, dst, *args, **kwargs):
        if same_path(dst, target):
            raise PermissionError(13, "synthetic busy file", str(dst))
        return original(src, dst, *args, **kwargs)
    return mock.patch("os.replace", replace)


class ScreenCase(unittest.TestCase):
    backend = "sqlite"
    with_database = True

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="conflict-replace-")).resolve()

        def remove():
            # соединения SQLite обработчиков экрана держат файл базы в Windows
            gc.collect()
            shutil.rmtree(self.base, ignore_errors=True)
        self.addCleanup(remove)
        self.database = self.base / "state.sqlite3"
        for patcher in (mock.patch.object(paths, "STATE_DB", self.database),
                        mock.patch.object(settings, "recording_backend", return_value=self.backend)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.count = 0
        if self.with_database:
            self.store = storage.initialize(self.database, initial_display_name="Тест")
            self.addCleanup(self.store.close)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = min(probe.getsockname()[1], 65535 - 20)  # make_server перебирает 20 портов вверх
        self.srv, url, _ = workflow_page.make_server(port)
        self.prefix = urlsplit(url).path
        self.host = f"127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)

    # ---------- HTTP ----------
    def request(self, method, path, body=None, *, origin="screen", prefix=None, host=None,
                ctype="application/json"):
        payload = b"" if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        lines = [f"{method} {self.prefix if prefix is None else prefix}{path} HTTP/1.1",
                 f"Host: {host or self.host}", f"Content-Length: {len(payload)}", "Connection: close"]
        if method == "POST":
            lines.append(f"Content-Type: {ctype}")
            if origin is not None:
                lines.append("Origin: " + (f"http://{self.host}" if origin == "screen" else origin))
        # заголовки и тело одной отправкой: отказ до чтения тела в Windows иначе рвёт соединение
        with socket.create_connection(("127.0.0.1", self.srv.server_port), timeout=90) as sock:
            sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("utf-8") + payload)
            response = http.client.HTTPResponse(sock, method=method)
            try:
                response.begin()
                return response.status, dict(response.getheaders()), response.read()
            finally:
                response.close()

    def ok(self, method, path, body=None):
        status, _, raw = self.request(method, path, body)
        self.assertEqual(status, 200, f"{method} {path}: {raw[:300].decode('utf-8', 'replace')}")
        data = json.loads(raw)
        self.assertFalse(data.get("error"), data)
        return data

    def refused(self, path, body):
        status, _, raw = self.request("POST", path, body)
        self.assertEqual(status, 400, f"{path}: {raw[:300].decode('utf-8', 'replace')}")
        self.assertRegex(json.loads(raw)["error"], CYRILLIC)

    def body(self, case, **changes):
        return {"recording_id": case.rid, "conflict_id": case.id, "kind": case.scope, **changes}

    def listing(self):
        return self.ok("GET", "api/conflicts/all")

    def item(self, case):
        found = [row for row in self.listing()["items"] if row["conflict_id"] == case.id]
        self.assertEqual(len(found), 1, "конфликт должен быть в общем списке один раз")
        return found[0]

    def preview(self, case):
        shown = self.ok("POST", "api/conflict/preview", self.body(case))
        self.assertIn("snapshot", shown, "предпросмотр должен готовить сравнение и снимок")
        return shown

    def snapshot(self, case):
        found = self.preview(case)["snapshot"]
        self.assertTrue(found, "заменяемый конфликт должен получить снимок")
        return found

    def replace(self, snapshot):
        return self.ok("POST", "api/conflict/resolve", {"snapshot": snapshot, "confirm": True})

    def retry(self, case):
        return self.ok("POST", "api/conflict/retry", self.body(case))

    def copy(self, case):
        status, _, raw = self.request("POST", "api/conflict/copy", self.body(case))
        self.assertEqual(status, 200, f"копия: {raw[:300].decode('utf-8', 'replace')}")
        return raw

    # ---------- выдуманные записи ----------
    def recording(self, *, root=None, ident=None):
        self.count += 1
        source = self.base / f"Встреча пример {self.count}.wav"
        source.write_bytes(b"synthetic")
        layout = recordings.document_layout(source)
        root = root or layout["root"]
        root.mkdir(exist_ok=True)
        root_id, relatives = recordings.prepare_new_layout(self.store, root, layout["paths"])
        ident = ident or str(uuid.uuid4())
        segments = [{"start": i * 2, "end": i * 2 + 2, "speaker": "Спикер 1",
                     "words": [{"w": word, "s": i * 2, "e": i * 2 + 1}]}
                    for i, word in enumerate(("альфа", "бета", "гамма"))]
        data = {"file": str(source), "duration": "00:00:06", "speakers": ["Спикер 1"], "quality": {},
                "review": [], "review_done": False, "segments": segments}
        recordings.commit_worker_result(
            self.store, data, source, root_id, recordings.operation_context(self.store, "model"),
            recording_id=ident, operation_id=str(uuid.uuid4()),
            generated_at="2026-01-01T00:00:00Z", document_paths=relatives)
        recordings.materialize_recording(self.store, ident)
        return SimpleNamespace(rid=ident, root=root, root_id=root_id, stem=source.stem,
                               files={kind: root / relative for kind, relative in relatives.items()},
                               conspect_id=str(uuid.uuid4()), conspect_revision=0,
                               conspect_files={kind: root / f"{source.stem}.Краткий.conspect.{kind}"
                                               for kind in ("json", "md")})

    def change_text(self, rec, word):
        current = self.store.get_current(rec.rid)
        transcript = current["transcript"]["payload"]
        transcript["segments"][1]["words"][0]["w"] = word
        transcript["segments"][1]["text"] = word
        return self.store.commit_change_with_document(
            rec.rid, {"source_revision": current["transcript"]["revision"],
                      "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"), transcript=transcript,
            review=current["review"]["payload"], root_id=rec.root_id,
            generated_at="2026-01-01T00:01:00Z")

    def save_conspect(self, rec, md=None):
        revision = rec.conspect_revision + 1
        payload = {"name": "Краткий", "recording_id": rec.rid, "synthetic_revision": revision}
        self.store.save_conspect(
            rec.rid, rec.conspect_id, "Краткий", self.store.get_current(rec.rid)["transcript"]["revision"],
            rec.conspect_revision or None, payload, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            md or f"# Краткий\n\nальфа бета\n\nревизия {revision}\n".encode("utf-8"),
            rec.conspect_files["json"].name, rec.conspect_files["md"].name, rec.root_id,
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"))
        rec.conspect_revision = revision

    def program(self, rec, scope, kind):
        if scope == "document":
            revision = self.store.db.execute(
                "SELECT max(document_revision) FROM document_versions WHERE recording_id=?",
                (rec.rid,)).fetchone()[0]
            return bytes(self.store.get_document(rec.rid, revision)[kind])
        return bytes(self.store.get_conspect(rec.conspect_id, rec.conspect_revision)[kind])

    def materialize(self, rec, scope):
        if scope == "document":
            return recordings.materialize_recording(self.store, rec.rid)
        return recordings.materialize_conspects(self.store, rec.rid)

    def conflict(self, scope, body=None, *, kind="md", rec=None):
        """Открытый конфликт; body — байты, сохранённые человеком, или функция от прежних."""
        rec = rec or self.recording()
        if scope == "conspect":
            self.save_conspect(rec)
            self.materialize(rec, scope)
        path = (rec.files if scope == "document" else rec.conspect_files)[kind]
        original = path.read_bytes()
        if body is None:
            saved = edited(original) if kind == "md" else original + b"\n"
        else:
            saved = body(original) if callable(body) else body
        self.assertNotEqual(saved, original)
        path.write_bytes(saved)
        if scope == "conspect":
            self.save_conspect(rec)
        self.materialize(rec, scope)
        return self.open_case(rec, scope, kind)

    def conspect_texts(self, program, mine):
        """Конфликт свёртки с заданным текстом программы и заданной правкой человека."""
        rec = self.recording()
        self.save_conspect(rec, md=program.encode("utf-8"))
        rec.conspect_files["md"].write_bytes(mine.encode("utf-8"))
        self.materialize(rec, "conspect")
        return self.open_case(rec, "conspect", "md")

    def both_conspect_files(self):
        rec = self.recording()
        self.save_conspect(rec)
        self.materialize(rec, "conspect")
        for path in rec.conspect_files.values():
            path.write_bytes(b"my edit\n")
        self.save_conspect(rec)
        self.materialize(rec, "conspect")
        return self.open_case(rec, "conspect", "md"), self.open_case(rec, "conspect", "json")

    def open_rows(self, rec, scope):
        if scope == "document":
            return core.list_document_conflicts(rec.rid)
        return core.list_conspect_conflicts(rec.rid)

    def open_case(self, rec, scope, kind):
        rows = [row for row in self.open_rows(rec, scope) if row["kind"] == kind]
        self.assertEqual(len(rows), 1, "подготовка теста: нужен один открытый конфликт файла")
        found = self.store.db.execute(f"SELECT observed_bytes FROM {scope}_conflicts WHERE conflict_id=?",
                                      (rows[0]["conflict_id"],)).fetchone()[0]
        return SimpleNamespace(rec=rec, rid=rec.rid, id=rows[0]["conflict_id"], scope=scope, kind=kind,
                               relative=rows[0]["relative_path"], path=rec.root / rows[0]["relative_path"],
                               saved=None if found is None else bytes(found))

    def is_open(self, case):
        return self.store.db.execute(
            f"SELECT resolution_operation_id FROM {case.scope}_conflicts WHERE conflict_id=?",
            (case.id,)).fetchone()[0] is None

    def resolve_directly(self, case):
        """Решение записано, а файлы не формировались: так остаётся после прежней версии или сбоя."""
        getattr(self.store, f"resolve_{case.scope}_conflict")(
            case.id, str(uuid.uuid4()), recordings.operation_context(self.store, "human"))

    def after_decision(self, scope, action):
        """Выполнить action сразу после записи решения, до формирования файлов."""
        name = f"resolve_{scope}_conflict"
        original = getattr(storage.Store, name)

        def resolve(store, *args, **kwargs):
            result = original(store, *args, **kwargs)
            action()
            return result
        return mock.patch.object(storage.Store, name, resolve)

    # ---------- состояние ----------
    def counts(self, *tables):
        return {table: self.store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in tables}

    def state(self):
        """Вся база построчно и все файлы песочницы побайтно."""
        db = sqlite3.connect(self.database)
        try:
            rows = hashlib.sha256()
            for line in db.iterdump():
                rows.update(line.encode("utf-8", "surrogatepass"))
        finally:
            db.close()
        files = {}
        for path in self.base.rglob("*"):
            if not path.name.startswith(self.database.name):
                files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"
        return rows.hexdigest(), files

    def assert_untouched(self, case, content):
        self.assertTrue(case.path.read_bytes() == content, "файл человека изменён")
        self.assertTrue(self.is_open(case), "конфликт помечен решённым")


class ConflictListTests(ScreenCase):
    def test_list_shows_conflicts_of_recordings_without_request(self):
        cases = [self.conflict(scope) for scope in SCOPES]
        self.assertEqual(core.workflow_list(), [])
        data = self.listing()
        self.assertIs(data["available"], True)
        self.assertEqual({row["conflict_id"] for row in data["items"]}, {case.id for case in cases})
        for case in cases:
            row = self.item(case)
            self.assertEqual((row["scope"], row["recording_id"], row["kind"], row["relative_path"]),
                             (case.scope, case.rid, "md", case.relative))
            self.assertIn(case.rec.stem.casefold(), row["title"].casefold())
            self.assertTrue(same_path(row["file"], case.path), row["file"])
            self.assertEqual(row["observed_size"], len(case.saved))
            self.assertIs(row["preserved"], True)
            self.assertEqual((row["status"], row["can_replace"], row["can_retry"]), ("open", True, False))
            if case.scope == "conspect":
                self.assertEqual(row["name"], "Краткий")

    def test_database_without_conflicts_has_empty_list(self):
        self.recording()
        data = self.listing()
        self.assertEqual((data["available"], data["items"]), (True, []))

    def test_resolved_conflicts_stay_with_downloadable_copy(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.replace(self.snapshot(case))
                row = self.item(case)
                self.assertEqual((row["status"], row["can_replace"], row["can_retry"]),
                                 ("resolved", False, False))
                self.assertEqual(self.copy(case), case.saved)
                self.assertEqual(self.open_rows(case.rec, scope), [])

    def test_copy_left_by_previous_decision_is_listed(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.resolve_directly(case)
                self.materialize(case.rec, scope)
                self.assertEqual(self.item(case)["status"], "resolved")
                self.assertEqual(self.copy(case), case.saved)

    def test_decided_but_unwritten_file_is_not_shown_as_done(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.resolve_directly(case)
                row = self.item(case)
                self.assertEqual((row["status"], row["can_replace"], row["can_retry"]),
                                 ("not_written", False, True))

    def test_list_and_copy_work_when_folder_is_gone(self):
        opened, decided = self.conflict("document"), self.conflict("conspect")
        self.resolve_directly(decided)
        for case in (opened, decided):
            os.rename(case.rec.root, case.rec.root.with_name(case.rec.root.name + "-away"))
        self.assertEqual(self.item(opened)["status"], "open")
        self.assertEqual(self.item(decided)["status"], "not_written")
        self.assertEqual(self.copy(opened), opened.saved)
        self.assertEqual(self.copy(decided), decided.saved)
        shown = self.preview(opened)
        self.assertIn(shown["file_state"], ("missing", "unavailable"))
        self.assertEqual((shown["can_replace"], shown["snapshot"]), (False, None))

    def test_list_changes_nothing(self):
        self.conflict("document")
        self.resolve_directly(self.conflict("conspect"))
        before = self.state()
        self.listing()
        self.assertEqual(self.state(), before)


class ConflictCopyTests(ScreenCase):
    def test_copy_is_byte_exact(self):
        bodies = {"long": "\n".join(f"строка правки {i}" for i in range(3000)).encode("utf-8"),
                  "crlf-bom": "﻿моя правка\r\nвторая\r\n".encode("utf-8"),
                  "not-utf8": "моя правка".encode("cp1251") + b"\x00\xff",
                  "empty": b""}
        for scope in SCOPES:
            for label, body in bodies.items():
                with self.subTest(scope=scope, body=label):
                    self.assertTrue(self.copy(self.conflict(scope, body)) == body, "копия не побайтная")

    def test_copy_at_size_limit_is_whole(self):
        body = b"x" * documents.MAX_CONFLICT_BYTES
        self.assertTrue(self.copy(self.conflict("document", body)) == body, "копия не целая")

    def test_copy_is_attachment_with_safe_name(self):
        for scope in SCOPES:
            for kind in ("md", "json"):
                with self.subTest(scope=scope, kind=kind):
                    case = self.conflict(scope, kind=kind)
                    status, headers, raw = self.request("POST", "api/conflict/copy", self.body(case))
                    self.assertEqual((status, raw), (200, case.saved))
                    self.assertEqual(headers["Content-Type"].split(";")[0], "application/octet-stream")
                    self.assertEqual(headers["Cache-Control"], "no-store")
                    self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                    disposition = headers["Content-Disposition"]
                    self.assertTrue(disposition.startswith("attachment"), disposition)
                    self.assertTrue(disposition.isascii(), disposition)
                    found = re.search(r"filename\*=UTF-8''([^;]+)", disposition)
                    self.assertIsNotNone(found, disposition)
                    name = unquote(found.group(1))
                    self.assertTrue(name.endswith("." + kind), name)
                    self.assertFalse(set(name) & set('/\\:*?"<>|\r\n'), name)
                    self.assertNotIn(self.prefix.strip("/"), str(headers))

    def test_copy_changes_nothing(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                before = self.state()
                self.copy(case)
                self.assertEqual(self.state(), before)

    def test_copy_does_not_need_the_file(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                case.path.unlink()
                self.assertEqual(self.copy(case), case.saved)
                self.assertFalse(case.path.exists())

    def test_copy_not_saved_over_limit_is_refused(self):
        case = self.conflict("document", b"x" * (documents.MAX_CONFLICT_BYTES + 1))
        self.assertIsNone(case.saved)
        self.refused("api/conflict/copy", self.body(case))

    def test_copy_of_another_recording_or_kind_is_refused(self):
        case = self.conflict("document")
        self.copy(case)
        for changes in ({"recording_id": str(uuid.uuid4())}, {"kind": "conspect"}, {"kind": "непонятно"},
                        {"conflict_id": str(uuid.uuid4())}, {"conflict_id": "не идентификатор"}):
            with self.subTest(changes=changes):
                self.refused("api/conflict/copy", self.body(case, **changes))


class ConflictPreviewTests(ScreenCase):
    def test_truncation_counts_characters_not_bytes(self):
        for size, cut in ((12000, False), (20000, False), (20001, True)):
            with self.subTest(size=size):
                case = self.conflict("document", ("я" * size).encode("utf-8"))
                shown = core.preview_conflict(case.rid, case.id, "document")
                self.assertEqual((len(shown["preview"]), shown["truncated"]), (min(size, 20000), cut))

    def test_preview_shows_lines_that_go_and_come(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                program = self.program(case.rec, scope, "md")
                shown = self.preview(case)
                self.assertEqual(shown["removed"], only_in(case.saved, program))
                self.assertEqual(shown["added"], only_in(program, case.saved))
                self.assertIn("альфа, моя правка", "\n".join(shown["removed"]))
                self.assertEqual((shown["removed_total"], shown["added_total"]),
                                 (len(shown["removed"]), len(shown["added"])))
                self.assertEqual((shown["shown_all"], shown["newlines_only"], shown["also_replaces"]),
                                 (True, False, []))
                self.assertEqual((shown["kind"], shown["file_state"], shown["can_replace"]), ("md", "same", True))
                self.assertTrue(same_path(shown["file"], case.path))
                self.assertTrue(shown["snapshot"])

    def test_preview_changes_nothing(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                before = self.state()
                self.snapshot(case)
                self.assertEqual(self.state(), before)

    def test_preview_compares_with_latest_program_version(self):
        rec = self.recording()
        self.change_text(rec, "вторая версия")
        self.change_text(rec, "третья версия")
        rec.files["md"].write_bytes(edited(rec.files["md"].read_bytes()))
        self.materialize(rec, "document")
        document = self.open_case(rec, "document", "md")

        other = self.recording()
        self.save_conspect(other)
        self.save_conspect(other)
        other.conspect_files["md"].write_bytes("моя свёртка\n".encode("utf-8"))
        self.materialize(other, "conspect")
        conspect = self.open_case(other, "conspect", "md")

        for case in (document, conspect):
            with self.subTest(scope=case.scope):
                latest = self.program(case.rec, case.scope, "md")
                shown = self.preview(case)
                self.assertEqual(shown["added"], only_in(latest, case.saved))
                self.assertEqual(self.replace(shown["snapshot"])["outcome"], "replaced")
                self.assertEqual(case.path.read_bytes(), latest)

    def test_only_line_endings_are_named_separately(self):
        variants = {"crlf": lambda data: data.replace(b"\n", b"\r\n"),
                    "cr": lambda data: data.replace(b"\n", b"\r"),
                    "final": lambda data: data[:-1] if data.endswith(b"\n") else data + b"\n"}
        for label, change in variants.items():
            with self.subTest(variant=label):
                shown = self.preview(self.conflict("document", change))
                self.assertIs(shown["newlines_only"], True)
                self.assertEqual((shown["removed"], shown["added"], shown["removed_total"], shown["added_total"]),
                                 ([], [], 0, 0))
                self.assertIs(shown["can_replace"], True)

    def test_line_endings_do_not_hide_real_edit(self):
        case = self.conflict("document", lambda data: edited(data).replace(b"\n", b"\r\n"))
        program = self.program(case.rec, "document", "md")
        shown = self.preview(case)
        self.assertIs(shown["newlines_only"], False)
        self.assertEqual(shown["removed"], only_in(case.saved, program))
        self.assertEqual(shown["added"], only_in(program, case.saved))

    def test_invisible_edits_are_not_called_line_endings(self):
        variants = {"bom": lambda data: b"\xef\xbb\xbf" + data,
                    "form-feed": lambda data: data.replace(b"\n", b"\x0c", 1),
                    "line-separator": lambda data: data.replace(b"\n", " ".encode("utf-8"), 1),
                    "space": lambda data: data.replace(b"\n", b" \n", 1)}
        for label, change in variants.items():
            with self.subTest(variant=label):
                shown = self.preview(self.conflict("document", change))
                self.assertIs(shown["newlines_only"], False)
                self.assertGreater(shown["removed_total"], 0)
                self.assertTrue(shown["removed"])

    def test_all_removed_lines_are_counted(self):
        body = "".join(f"моя строка {i}\n" for i in range(300)).encode("utf-8")
        case = self.conflict("document", body)
        expected = only_in(body, self.program(case.rec, "document", "md"))
        shown = self.preview(case)
        self.assertEqual(shown["removed_total"], len(expected))
        self.assertGreaterEqual(len(shown["removed"]), 100)
        self.assertEqual(shown["removed"], expected[:len(shown["removed"])])
        self.assertEqual(shown["shown_all"], len(shown["removed"]) == len(expected)
                         and len(shown["added"]) == shown["added_total"])

    def test_large_text_is_compared_in_reasonable_time(self):
        variants = {"repeated": "да\nнет\nугу\n\n" * 1000,
                    "long": "".join(f"реплика {i}\n\n" for i in range(10000))}
        for label, text in variants.items():
            with self.subTest(variant=label):
                case = self.conflict("document", text.encode("utf-8"))
                started = time.monotonic()
                shown = self.preview(case)
                self.assertLess(time.monotonic() - started, 30)
                self.assertGreater(shown["removed_total"], 0)

    def test_many_scattered_edits_are_compared_in_reasonable_time(self):
        # «заменить все» в Блокноте: изменена каждая вторая строка длинного файла
        lines = [f"реплика {i}" for i in range(60000)]
        rec = self.recording()
        self.save_conspect(rec, md=("\n".join(lines) + "\n").encode("utf-8"))
        changed = [line + "!" if i % 2 else line for i, line in enumerate(lines)]
        rec.conspect_files["md"].write_bytes(("\n".join(changed) + "\n").encode("utf-8"))
        self.materialize(rec, "conspect")
        case = self.open_case(rec, "conspect", "md")
        started = time.monotonic()
        shown = self.preview(case)
        self.assertLess(time.monotonic() - started, 30)
        self.assertGreaterEqual(shown["removed_total"], 30000)
        self.assertIs(shown["shown_all"], len(shown["removed"]) == shown["removed_total"]
                      and len(shown["added"]) == shown["added_total"])

    def test_limit_sized_copy_does_not_flood_preview(self):
        case = self.conflict("document", b"x" * documents.MAX_CONFLICT_BYTES)
        status, _, raw = self.request("POST", "api/conflict/preview", self.body(case))
        self.assertEqual(status, 200)
        self.assertLess(len(raw), 4 * 1024 * 1024)
        shown = json.loads(raw)
        self.assertIn("snapshot", shown, "предпросмотр должен готовить сравнение и снимок")
        self.assertIs(shown["shown_all"], False)
        self.assertTrue(shown["snapshot"])
        self.assertLess(len(json.dumps(shown["snapshot"])), 16 * 1024)

    def test_preview_tells_that_file_differs_from_shown_copy(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                program = self.program(case.rec, scope, "md")
                case.path.write_bytes(LATE_EDIT)
                shown = self.preview(case)
                self.assertEqual((shown["file_state"], shown["can_replace"]), ("changed", True))
                self.assertEqual(shown["removed"], only_in(case.saved, program))
                case.path.write_bytes(program)
                self.assertEqual(self.preview(case)["file_state"], "program")

    def test_preview_without_file_explains_and_gives_no_snapshot(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                case.path.unlink()
                shown = self.preview(case)
                self.assertEqual((shown["file_state"], shown["can_replace"], shown["snapshot"]),
                                 ("missing", False, None))
                self.assertRegex(shown["reason"], CYRILLIC)
                self.assertTrue(shown["removed"])

    def test_resolved_copy_is_view_only(self):
        case = self.conflict("document")
        self.replace(self.snapshot(case))
        shown = self.preview(case)
        self.assertEqual((shown["can_replace"], shown["snapshot"]), (False, None))

    def test_preview_names_file_waiting_for_this_confirmation(self):
        first, second = self.both_conspect_files()
        self.assertEqual(self.preview(second)["also_replaces"], [])
        self.replace(self.snapshot(first))
        self.assertEqual(self.preview(second)["also_replaces"], [first.relative])

    def test_preview_names_returned_copy_that_goes_with_other_file(self):
        first = self.conflict("document")
        self.resolve_directly(first)
        self.materialize(first.rec, "document")
        # человек вернул скачанную копию на место и поправил второй файл той же записи
        first.path.write_bytes(first.saved)
        first.rec.files["json"].write_bytes(b"{}\n")
        self.materialize(first.rec, "document")
        second = self.open_case(first.rec, "document", "json")
        self.assertEqual(self.preview(second)["also_replaces"], [first.relative])


class ConflictReplaceTests(ScreenCase):
    def test_replace_writes_program_version_and_keeps_copy(self):
        for scope in SCOPES:
            for kind in ("md", "json"):
                with self.subTest(scope=scope, kind=kind):
                    case = self.conflict(scope, kind=kind)
                    snapshot = self.snapshot(case)
                    before = self.counts("operations", *REVISION_TABLES)
                    result = self.replace(snapshot)
                    self.assertEqual((result["outcome"], result["new_conflict_id"]), ("replaced", None))
                    self.assertEqual(case.path.read_bytes(), self.program(case.rec, scope, kind))
                    self.assertFalse(self.is_open(case))
                    self.assertEqual(self.copy(case), case.saved)
                    self.assertEqual(self.counts("operations", *REVISION_TABLES),
                                     {**before, "operations": before["operations"] + 1},
                                     "правка не переносится в стенограмму, новых ревизий нет")

    def test_empty_file_can_be_replaced(self):
        case = self.conflict("document", b"")
        self.assertEqual(self.replace(self.snapshot(case))["outcome"], "replaced")
        self.assertEqual(case.path.read_bytes(), self.program(case.rec, "document", "md"))

    def test_edit_works_after_replace(self):
        case = self.conflict("document")
        with self.assertRaises(storage.Conflict):
            self.change_text(case.rec, "омега")
        self.replace(self.snapshot(case))
        self.change_text(case.rec, "омега")
        self.materialize(case.rec, "document")
        self.assertIn("омега".encode("utf-8"), case.path.read_bytes())

    def test_replace_without_snapshot_is_refused(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                bodies = [{**self.body(case), "confirm": True}]
                for snapshot in (None, {}, [], "снимок", 1, self.body(case)):
                    bodies.append({**self.body(case), "confirm": True, "snapshot": snapshot})
                for body in bodies:
                    self.refused("api/conflict/resolve", body)
                    self.assert_untouched(case, case.saved)

    def test_replace_needs_literal_confirmation(self):
        case = self.conflict("document")
        snapshot = self.snapshot(case)
        for value in (None, False, 0, 1, "true", "false", [True]):
            with self.subTest(confirm=value):
                body = {"snapshot": snapshot}
                if value is not None:
                    body["confirm"] = value
                self.refused("api/conflict/resolve", body)
                self.assert_untouched(case, case.saved)

    def test_altered_snapshot_is_refused(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                self.assertIsInstance(snapshot, dict)
                for key, value in snapshot.items():
                    for wrong in ({other: item for other, item in snapshot.items() if other != key},
                                  {**snapshot, key: "x" if value != "x" else "y"}):
                        self.refused("api/conflict/resolve", {"snapshot": wrong, "confirm": True})
                        self.assert_untouched(case, case.saved)

    def test_snapshot_cannot_be_moved_to_another_conflict(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                shown, hidden = self.conflict(scope), self.conflict(scope, b"another edit\n")
                text = json.dumps(self.snapshot(shown))
                text = text.replace(shown.id, hidden.id).replace(shown.rid, hidden.rid)
                self.refused("api/conflict/resolve", {"snapshot": json.loads(text), "confirm": True})
                self.assert_untouched(hidden, hidden.saved)

    def test_snapshot_is_stale_after_new_program_version(self):
        case = self.conflict("document")
        snapshot = self.snapshot(case)
        # новая версия программы успела записаться в базу, пока человек смотрел на сравнение
        with mock.patch.object(storage.Store, "_check_document_conflict", return_value=None):
            self.change_text(case.rec, "новее показанного")
        self.refused("api/conflict/resolve", {"snapshot": snapshot, "confirm": True})
        self.assert_untouched(case, case.saved)
        self.assertEqual(self.replace(self.snapshot(case))["outcome"], "replaced")
        self.assertEqual(case.path.read_bytes(), self.program(case.rec, "document", "md"))

    def test_repeated_confirmation_cannot_replace_returned_copy(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                self.assertEqual(self.replace(snapshot)["outcome"], "replaced")
                case.path.write_bytes(case.saved)
                operations = self.counts("operations")
                self.refused("api/conflict/resolve", {"snapshot": snapshot, "confirm": True})
                self.assertEqual(case.path.read_bytes(), case.saved)
                self.assertEqual(self.counts("operations"), operations)

    def test_snapshot_is_stale_after_decision_elsewhere(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                self.resolve_directly(case)
                self.refused("api/conflict/resolve", {"snapshot": snapshot, "confirm": True})
                self.assertEqual(case.path.read_bytes(), case.saved)

    def test_parallel_confirmations_make_one_decision(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                body = {"snapshot": self.snapshot(case), "confirm": True}
                operations = self.counts("operations")["operations"]
                barrier, answers = threading.Barrier(2), []

                def send():
                    barrier.wait(10)
                    answers.append(self.request("POST", "api/conflict/resolve", body))
                threads = [threading.Thread(target=send) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(100)
                self.assertEqual(sorted(answer[0] for answer in answers), [200, 400])
                self.assertEqual(self.counts("operations")["operations"], operations + 1)
                won = json.loads(next(raw for status, _, raw in answers if status == 200))
                # второй запрос мог на миг держать файл открытым для сверки; тогда нужен повтор
                self.assertIn(won["outcome"], ("replaced", "not_written"))
                if won["outcome"] == "not_written":
                    self.assertEqual(self.retry(case)["outcome"], "replaced")
                self.assertEqual(case.path.read_bytes(), self.program(case.rec, scope, "md"))

    def test_missing_file_is_refused_before_decision(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                case.path.unlink()
                before = self.state()
                self.refused("api/conflict/resolve", {"snapshot": snapshot, "confirm": True})
                self.assertEqual(self.state(), before)
                # выход для человека: положить скачанную копию на место и заменить
                case.path.write_bytes(self.copy(case))
                self.assertEqual(self.replace(self.snapshot(case))["outcome"], "replaced")

    def test_folder_instead_of_file_is_refused_before_decision(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                case.path.unlink()
                case.path.mkdir()
                shown = self.preview(case)
                self.assertEqual((shown["can_replace"], shown["snapshot"]), (False, None))
                self.refused("api/conflict/resolve", {"snapshot": snapshot, "confirm": True})
                self.assertTrue(self.is_open(case))
                self.assertTrue(case.path.is_dir())

    @unittest.skipUnless(os.name == "nt", "соединения каталогов есть только в Windows")
    def test_junction_is_refused_before_decision(self):
        import _winapi
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                actual = case.rec.root.with_name(case.rec.root.name + "-actual")
                os.rename(case.rec.root, actual)
                _winapi.CreateJunction(str(actual), str(case.rec.root))
                try:
                    shown = self.preview(case)
                    self.assertEqual((shown["file_state"], shown["can_replace"]), ("unavailable", False))
                    self.refused("api/conflict/resolve", {"snapshot": snapshot, "confirm": True})
                    self.assert_untouched(case, case.saved)
                finally:
                    os.rmdir(case.rec.root)
                    os.rename(actual, case.rec.root)

    @unittest.skipUnless(os.name == "nt", "запрет чтения блокировкой есть только в Windows")
    def test_unreadable_file_is_refused_before_decision(self):
        import msvcrt
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                with open(case.path, "r+b") as held:
                    msvcrt.locking(held.fileno(), msvcrt.LK_NBLCK, len(case.saved))
                    try:
                        shown = self.preview(case)
                        self.assertEqual((shown["file_state"], shown["can_replace"]), ("unavailable", False))
                        self.refused("api/conflict/resolve", {"snapshot": snapshot, "confirm": True})
                        self.assertTrue(self.is_open(case))
                    finally:
                        held.seek(0)
                        msvcrt.locking(held.fileno(), msvcrt.LK_UNLCK, len(case.saved))
                self.assert_untouched(case, case.saved)

    def test_over_limit_file_is_never_replaced(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                body = b"x" * (documents.MAX_CONFLICT_BYTES + 1)
                case = self.conflict(scope, body)
                row = self.item(case)
                self.assertEqual((row["preserved"], row["can_replace"]), (False, False))
                shown = self.preview(case)
                self.assertEqual((shown["can_replace"], shown["snapshot"]), (False, None))
                self.assertRegex(shown["reason"], CYRILLIC)
                self.refused("api/conflict/resolve", {"snapshot": self.body(case), "confirm": True})
                self.assert_untouched(case, body)

    def test_archive_is_never_replaced_but_can_be_copied(self):
        rec = self.recording()
        self.change_text(rec, "вторая версия")
        path = rec.root / "history" / rec.rid / "document-2" / "transcript.md"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"foreign archive\n")
        self.materialize(rec, "document")
        case = self.open_case(rec, "document", "archive_md")
        self.assertIs(self.item(case)["can_replace"], False)
        shown = self.preview(case)
        self.assertEqual((shown["can_replace"], shown["snapshot"]), (False, None))
        self.assertRegex(shown["reason"], CYRILLIC)
        self.refused("api/conflict/resolve", {"snapshot": self.body(case), "confirm": True})
        self.assert_untouched(case, b"foreign archive\n")
        self.assertEqual(self.copy(case), b"foreign archive\n")

    def test_file_changed_after_preview_gets_new_copy(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                case.path.write_bytes(LATE_EDIT)
                result = self.replace(snapshot)
                self.assertEqual(result["outcome"], "changed")
                self.assertEqual(case.path.read_bytes(), LATE_EDIT)
                fresh = self.open_case(case.rec, scope, "md")
                self.assertEqual(result["new_conflict_id"], fresh.id)
                self.assertNotEqual(fresh.id, case.id)
                self.assertEqual(self.copy(fresh), LATE_EDIT)
                self.assertEqual(self.copy(case), case.saved)
                old, new = self.item(case), self.item(fresh)
                self.assertEqual((old["status"], old["can_retry"]), ("resolved", False))
                self.assertEqual((new["status"], new["can_replace"]), ("open", True))
                self.refused("api/conflict/resolve", {"snapshot": snapshot, "confirm": True})
                self.assert_untouched(fresh, LATE_EDIT)
                self.assertEqual(self.replace(self.snapshot(fresh))["outcome"], "replaced")

    def test_file_changed_between_decision_and_write_is_kept(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                with self.after_decision(scope, lambda: case.path.write_bytes(LATE_EDIT)):
                    result = self.replace(snapshot)
                self.assertEqual(result["outcome"], "changed")
                self.assertEqual(case.path.read_bytes(), LATE_EDIT)
                fresh = self.open_case(case.rec, scope, "md")
                self.assertEqual((result["new_conflict_id"], fresh.saved), (fresh.id, LATE_EDIT))

    def test_file_changed_to_over_limit_is_kept_without_claiming_copy(self):
        case = self.conflict("document")
        snapshot = self.snapshot(case)
        big = b"x" * (documents.MAX_CONFLICT_BYTES + 1)
        case.path.write_bytes(big)
        result = self.replace(snapshot)
        self.assertEqual(result["outcome"], "changed")
        self.assertTrue(case.path.read_bytes() == big, "файл человека изменён")
        fresh = self.open_case(case.rec, "document", "md")
        row = self.item(fresh)
        self.assertEqual((result["new_conflict_id"], row["preserved"], row["can_replace"]),
                         (fresh.id, False, False))

    def test_file_deleted_between_decision_and_write_is_reported(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                with self.after_decision(scope, case.path.unlink):
                    result = self.replace(snapshot)
                self.assertEqual((result["outcome"], result["new_conflict_id"]), ("not_written", None))
                self.assertRegex(result["reason"], CYRILLIC)
                self.assertFalse(case.path.exists())
                self.assertEqual(self.item(case)["status"], "not_written")

    def test_failed_write_is_not_reported_as_replaced(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                with failing_replace(case.path):
                    result = self.replace(snapshot)
                self.assertEqual((result["outcome"], result["new_conflict_id"]), ("not_written", None))
                self.assertRegex(result["reason"], CYRILLIC)
                self.assertEqual(case.path.read_bytes(), case.saved)
                row = self.item(case)
                self.assertEqual((row["status"], row["can_retry"], row["can_replace"]),
                                 ("not_written", True, False))
                self.assertEqual(self.copy(case), case.saved)

    @unittest.skipUnless(os.name == "nt", "открытый файл мешает замене только в Windows")
    def test_file_held_by_another_program_waits_for_retry(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                with open(case.path, "rb"):
                    self.assertEqual(self.replace(snapshot)["outcome"], "not_written")
                    self.assertEqual(self.item(case)["status"], "not_written")
                self.assertEqual(self.retry(case)["outcome"], "replaced")
                self.assertEqual(case.path.read_bytes(), self.program(case.rec, scope, "md"))

    def test_outcome_is_about_this_file_when_other_recording_fails(self):
        root = self.base / "common"
        mine = self.recording(root=root, ident="00000000-0000-4000-8000-000000000001")
        other = self.recording(root=root, ident="ffffffff-0000-4000-8000-000000000001")
        case = self.conflict("document", rec=mine)
        snapshot = self.snapshot(case)
        other.files["md"].unlink()
        with failing_replace(other.files["md"]):
            result = self.replace(snapshot)
        self.assertEqual(result["outcome"], "replaced")
        self.assertEqual(case.path.read_bytes(), self.program(mine, "document", "md"))
        self.assertEqual(self.item(case)["status"], "resolved")

    def test_write_stopped_by_other_recording_can_be_retried(self):
        root = self.base / "common"
        other = self.recording(root=root, ident="00000000-0000-4000-8000-000000000001")
        mine = self.recording(root=root, ident="ffffffff-0000-4000-8000-000000000001")
        case = self.conflict("document", rec=mine)
        snapshot = self.snapshot(case)
        self.change_text(other, "другая запись")
        with failing_replace(other.files["md"]):
            self.assertEqual(self.replace(snapshot)["outcome"], "not_written")
        self.assertEqual(case.path.read_bytes(), case.saved)
        self.assertIs(self.item(case)["can_retry"], True)
        self.assertEqual(self.retry(case)["outcome"], "replaced")

    def test_document_replaced_while_second_file_waits(self):
        case = self.conflict("document", kind="json")
        snapshot = self.snapshot(case)
        md = case.rec.files["md"]
        md.write_bytes(LATE_EDIT)
        self.assertEqual(self.replace(snapshot)["outcome"], "replaced")
        self.assertEqual(case.path.read_bytes(), self.program(case.rec, "document", "json"))
        self.assertEqual(md.read_bytes(), LATE_EDIT)
        self.assertEqual(self.item(case)["status"], "resolved")
        self.assertEqual(self.item(self.open_case(case.rec, "document", "md"))["status"], "open")

    def test_document_waiting_for_second_file_is_not_called_replaced(self):
        case = self.conflict("document")
        snapshot = self.snapshot(case)
        case.rec.files["json"].write_bytes(b"{}\n")
        result = self.replace(snapshot)
        self.assertEqual((result["outcome"], result["new_conflict_id"]), ("not_written", None))
        self.assertEqual(case.path.read_bytes(), case.saved)
        row = self.item(case)
        self.assertEqual((row["status"], row["can_retry"]), ("not_written", False))
        second = self.open_case(case.rec, "document", "json")
        self.assertEqual(self.replace(self.snapshot(second))["outcome"], "replaced")
        self.assertEqual(case.path.read_bytes(), self.program(case.rec, "document", "md"))
        self.assertEqual(self.item(case)["status"], "resolved")

    def test_conspect_waiting_for_second_file_is_not_called_replaced(self):
        first, second = self.both_conspect_files()
        result = self.replace(self.snapshot(first))
        self.assertEqual((result["outcome"], result["new_conflict_id"]), ("not_written", None))
        self.assertEqual(first.path.read_bytes(), first.saved)
        row = self.item(first)
        self.assertEqual((row["status"], row["can_retry"]), ("not_written", False))
        self.assertEqual(self.replace(self.snapshot(second))["outcome"], "replaced")
        for case in (first, second):
            self.assertEqual(case.path.read_bytes(), self.program(case.rec, "conspect", case.kind))
            self.assertEqual(self.item(case)["status"], "resolved")


class ConflictRetryTests(ScreenCase):
    def test_retry_finishes_failed_replace_without_new_revisions(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                with failing_replace(case.path):
                    self.assertEqual(self.replace(snapshot)["outcome"], "not_written")
                    self.assertEqual(self.retry(case)["outcome"], "not_written")
                before = self.counts("operations", *REVISION_TABLES)
                self.assertEqual(self.retry(case)["outcome"], "replaced")
                self.assertEqual(case.path.read_bytes(), self.program(case.rec, scope, "md"))
                self.assertEqual(self.counts("operations", *REVISION_TABLES), before)
                row = self.item(case)
                self.assertEqual((row["status"], row["can_retry"]), ("resolved", False))

    def test_retry_never_decides_open_conflict(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                before = self.state()
                self.refused("api/conflict/retry", self.body(case))
                self.assertEqual(self.state(), before)

    def test_retry_of_another_recording_is_refused(self):
        case = self.conflict("document")
        self.resolve_directly(case)
        self.refused("api/conflict/retry", self.body(case, recording_id=str(uuid.uuid4())))
        self.refused("api/conflict/retry", self.body(case, kind="conspect"))
        self.assertEqual(case.path.read_bytes(), case.saved)
        self.assertEqual(self.retry(case)["outcome"], "replaced")

    def test_retry_waits_for_returned_copy(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.resolve_directly(case)
                case.path.unlink()
                result = self.retry(case)
                self.assertEqual(result["outcome"], "not_written")
                self.assertRegex(result["reason"], CYRILLIC)
                self.assertFalse(case.path.exists())
                case.path.write_bytes(self.copy(case))
                self.assertEqual(self.retry(case)["outcome"], "replaced")

    def test_retry_keeps_document_edited_after_failure(self):
        case = self.conflict("document")
        snapshot = self.snapshot(case)
        with failing_replace(case.path):
            self.replace(snapshot)
        case.path.write_bytes(LATE_EDIT)
        result = self.retry(case)
        self.assertEqual(result["outcome"], "changed")
        self.assertEqual(case.path.read_bytes(), LATE_EDIT)
        fresh = self.open_case(case.rec, "document", "md")
        self.assertEqual((result["new_conflict_id"], fresh.saved), (fresh.id, LATE_EDIT))

    def test_retry_keeps_conspect_edited_after_failure(self):
        case = self.conflict("conspect")
        snapshot = self.snapshot(case)
        with failing_replace(case.path):
            self.replace(snapshot)
        case.path.write_bytes(LATE_EDIT)
        result = self.retry(case)
        self.assertEqual(result["outcome"], "not_written")
        self.assertRegex(result["reason"], CYRILLIC)
        self.assertEqual(case.path.read_bytes(), LATE_EDIT)
        case.path.write_bytes(self.copy(case))
        self.assertEqual(self.retry(case)["outcome"], "replaced")

    def test_repeated_retry_changes_nothing(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.resolve_directly(case)
                self.assertEqual(self.retry(case)["outcome"], "replaced")
                before = self.state()
                status, _, raw = self.request("POST", "api/conflict/retry", self.body(case))
                self.assertIn(status, (200, 400), raw[:300])
                self.assertEqual(self.state(), before)

    def test_retry_cannot_replace_returned_copy(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.resolve_directly(case)
                self.assertEqual(self.retry(case)["outcome"], "replaced")
                case.path.write_bytes(case.saved)
                status, _, raw = self.request("POST", "api/conflict/retry", self.body(case))
                self.assertIn(status, (200, 400), raw[:300])
                if status == 200:
                    self.assertNotEqual(json.loads(raw)["outcome"], "replaced")
                self.assertEqual(case.path.read_bytes(), case.saved)


class ConflictRequestProtectionTests(ScreenCase):
    def test_posts_require_screen_origin_host_token_and_json(self):
        case = self.conflict("document")
        snapshot = self.snapshot(case)
        bodies = {"preview": self.body(case), "copy": self.body(case), "retry": self.body(case),
                  "resolve": {"snapshot": snapshot, "confirm": True}}
        before = self.state()
        for route, body in bodies.items():
            for expected, options in ((403, {"origin": None}), (403, {"origin": "null"}),
                                      (403, {"origin": "http://example.test"}),
                                      (403, {"origin": f"http://localhost:{self.srv.server_port}"}),
                                      (403, {"host": f"localhost:{self.srv.server_port}"}),
                                      (403, {"prefix": "/not-the-session/"}),
                                      (415, {"ctype": "text/plain"})):
                with self.subTest(route=route, options=options):
                    status, _, raw = self.request("POST", "api/conflict/" + route, body, **options)
                    self.assertEqual(status, expected)
                    self.assertNotIn("моя правка".encode("utf-8"), raw)
        self.assertEqual(self.state(), before)

    def test_copy_and_changes_are_not_available_by_get(self):
        case = self.conflict("document")
        self.copy(case)
        before = self.state()
        query = f"?recording_id={case.rid}&conflict_id={case.id}&kind=document"
        for route in ("preview", "copy", "retry", "resolve"):
            with self.subTest(route=route):
                self.assertEqual(self.request("GET", f"api/conflict/{route}{query}")[0], 404)
        self.assertEqual(self.state(), before)


class ConflictMCPTests(ScreenCase):
    def mcp_resolve(self, case, **answer):
        with mock.patch.object(confirm, "require", **(answer or {"return_value": "dialog"})) as asked:
            response = asyncio.run(getattr(server, f"resolve_{case.scope}_conflict")(case.rid, case.id))
        return response, asked

    def test_window_says_file_will_be_replaced(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                response, asked = self.mcp_resolve(case)
                asked.assert_called_once()
                action, args, lines = asked.call_args.args
                text = "\n".join(lines)
                self.assertIn("заменить файл", action.casefold())
                self.assertIn("заменить файл", text.casefold())
                self.assertIn(case.relative, text)
                self.assertEqual((args["recording_id"], args["conflict_id"], args["hash"]),
                                 (case.rid, case.id, hashlib.sha256(case.saved).hexdigest()))
                self.assertIs(json.loads(response)["materialized"], True)

    def test_missing_file_is_refused_before_decision(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                case.path.unlink()
                response, _ = self.mcp_resolve(case)
                self.assertTrue(self.is_open(case), "конфликт помечен решённым, хотя файла нет")
                self.assertIn("файл", response.casefold())
                self.assertFalse(case.path.exists())

    def test_refused_window_changes_nothing(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                before = self.state()
                response, _ = self.mcp_resolve(case, side_effect=confirm.NotConfirmed("отказ"))
                self.assertIn("отказ", response)
                self.assertEqual(self.state(), before)

    def test_failed_write_stays_visible_on_screen(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                with failing_replace(case.path):
                    self.mcp_resolve(case)
                row = self.item(case)
                self.assertEqual((row["status"], row["can_retry"]), ("not_written", True))
                self.assertEqual(self.retry(case)["outcome"], "replaced")

    def test_sync_does_not_replace_open_conflict(self):
        case = self.conflict("document")
        json.loads(server.sync_recording_documents(case.rid))
        self.assert_untouched(case, case.saved)

    def test_existing_lists_keep_shape_and_only_open_rows(self):
        common = {"conflict_id", "kind", "relative_path", "observed_hash", "observed_size", "preserved"}
        fields = {"document": common | {"document_revision"},
                  "conspect": common | {"conspect_id", "name", "revision"}}
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                listed = getattr(server, f"list_{scope}_conflicts")
                rows = json.loads(listed(case.rid))
                self.assertEqual([row["conflict_id"] for row in rows], [case.id])
                self.assertEqual(set(rows[0]), fields[scope])
                self.assertIs(type(rows[0]["preserved"]), int)
                self.resolve_directly(case)
                self.assertEqual(json.loads(listed(case.rid)), [])


class NoDatabaseTests(ScreenCase):
    backend = "legacy_files"
    with_database = False

    def test_screen_list_explains_and_creates_nothing(self):
        for backend in ("legacy_files", "sqlite"):
            with self.subTest(backend=backend), \
                    mock.patch.object(settings, "recording_backend", return_value=backend):
                data = self.listing()
                self.assertEqual((data["available"], data["items"]), (False, []))
                self.assertRegex(data["reason"], CYRILLIC)
                self.assertEqual(list(self.base.iterdir()), [])

    def test_page_lists_and_actions_create_nothing(self):
        self.assertEqual(self.request("GET", "")[0], 200)
        ident = str(uuid.uuid4())
        for scope in SCOPES:
            self.assertIn("error", json.loads(getattr(server, f"list_{scope}_conflicts")(ident)))
            body = {"recording_id": ident, "conflict_id": ident, "kind": scope}
            for route in ("preview", "resolve", "copy", "retry"):
                # прежде copy и retry не было вовсе (404); успехом ответ быть не может
                self.assertIn(self.request("POST", "api/conflict/" + route, body)[0], (400, 404))
        self.assertEqual(list(self.base.iterdir()), [])


class LegacyWithDatabaseTests(ScreenCase):
    backend = "legacy_files"

    def test_existing_database_is_still_listed(self):
        case = self.conflict("document")
        self.assertEqual(self.item(case)["status"], "open")


class ConflictTruthTests(ScreenCase):
    """Сравнение не прячет потерь, а причина и отказ не уводят человека по ложному пути."""

    def test_moved_lines_are_losses(self):
        program = "Анна: да\nБорис: нет\nАнна: беру\nБорис: жду\n"
        variants = {
            "swap": ("Борис: нет\nАнна: да\nАнна: беру\nБорис: жду\n", 1),
            "move": ("Борис: нет\nАнна: беру\nБорис: жду\nАнна: да\n", 1),
            "move and edit": ("Борис: нет\nАнна: беру два\nБорис: жду\nАнна: да\n", 2)}
        for label, (mine, at_least) in variants.items():
            with self.subTest(variant=label):
                shown = self.preview(self.conspect_texts(program, mine))
                self.assertGreaterEqual(shown["removed_total"], at_least)
                self.assertTrue(set(shown["removed"]) <= set(mine.split("\n")), shown["removed"])
                self.assertTrue(shown["added"])
                self.assertIs(shown["newlines_only"], False)

    def test_replica_moved_to_another_speaker_is_loss(self):
        shown = self.preview(self.conspect_texts("Анна:\nкогда\nБорис:\nзавтра\n",
                                                 "Анна:\nзавтра\nБорис:\nкогда\n"))
        self.assertGreater(shown["removed_total"], 0)
        self.assertTrue(shown["removed"])

    def test_edit_inside_long_line_is_visible(self):
        line = "реплика монолога " * 500
        variants = {"end": line + "ВАЖНАЯ ПРАВКА", "middle": line[:4000] + "ВАЖНАЯ ПРАВКА" + line[4000:]}
        for label, mine in variants.items():
            with self.subTest(place=label):
                shown = self.preview(self.conspect_texts(line + "\n", mine + "\n"))
                self.assertTrue("ВАЖНАЯ ПРАВКА" in "\n".join(shown["removed"]),
                                "правка в длинной строке не видна перед заменой")

    def test_missing_file_names_where_to_put_the_copy(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                snapshot = self.snapshot(case)
                case.path.unlink()
                self.assertIn(case.path.name, self.preview(case)["reason"])
                status, _, raw = self.request("POST", "api/conflict/resolve",
                                              {"snapshot": snapshot, "confirm": True})
                self.assertEqual(status, 400)
                self.assertIn(case.path.name, json.loads(raw)["error"])
                self.resolve_directly(case)
                self.assertIn(case.path.name, self.retry(case)["reason"])

    def test_journal_reason_gives_the_way_out(self):
        case = self.conflict("conspect")
        snapshot = self.snapshot(case)
        with failing_replace(case.path):
            self.replace(snapshot)
        case.path.write_bytes(LATE_EDIT)
        reason = self.retry(case)["reason"].casefold()
        self.assertNotIn("открыт в другой программе", reason)
        self.assertRegex(reason, "верн|полож")
        self.assertIn("копи", reason)

    def test_busy_database_is_named(self):
        case = self.conflict("document")
        snapshot = self.snapshot(case)
        with mock.patch.object(recordings, "materialize_recording",
                               side_effect=storage.Busy("database is busy")):
            result = self.replace(snapshot)
        self.assertEqual(result["outcome"], "not_written")
        self.assertRegex(result["reason"].casefold(), r"\bбаз")
        self.assertNotIn("открыт в другой программе", result["reason"])

    def test_program_error_is_not_blamed_on_editor(self):
        case = self.conflict("document")
        snapshot = self.snapshot(case)
        with mock.patch.object(recordings, "materialize_recording", side_effect=RuntimeError("synthetic bug")):
            result = self.replace(snapshot)
        self.assertEqual(result["outcome"], "not_written")
        self.assertIn("ошибк", result["reason"].casefold())
        self.assertNotIn("открыт в другой программе", result["reason"])

    def test_outcome_is_returned_if_database_fails_after_decision(self):
        for label, error in (("busy", storage.Busy("database is busy")),
                             ("sqlite", sqlite3.OperationalError("disk I/O error"))):
            with self.subTest(error=label):
                case = self.conflict("document")
                snapshot = self.snapshot(case)
                original = storage.Store.get_document
                decided = []

                def get_document(store, *args, **kwargs):
                    if decided:
                        raise error
                    return original(store, *args, **kwargs)
                with self.after_decision("document", lambda: decided.append(True)), \
                        mock.patch.object(storage.Store, "get_document", get_document):
                    try:
                        status, _, raw = self.request("POST", "api/conflict/resolve",
                                                      {"snapshot": snapshot, "confirm": True})
                    except (http.client.HTTPException, ConnectionError) as ex:
                        self.fail(f"решение записано, а ответа нет: {type(ex).__name__}")
                self.assertFalse(self.is_open(case))
                self.assertEqual(status, 200, f"решение записано, а ответ — отказ: {raw[:200]!r}")
                result = json.loads(raw)
                self.assertEqual(result["outcome"], "not_written")
                self.assertRegex(result["reason"], CYRILLIC)
                self.assertEqual(case.path.read_bytes(), case.saved)


class ConflictMCPCompatibilityTests(ScreenCase):
    """У MCP по плану меняются только текст окна и отказ при пропавшем файле."""

    def resolve(self, case):
        with mock.patch.object(confirm, "require", return_value="dialog"):
            return asyncio.run(getattr(server, f"resolve_{case.scope}_conflict")(case.rid, case.id))

    def test_response_keeps_shape(self):
        fields = {"document": {"recording_id", "document_revision", "materialized"},
                  "conspect": {"conspect_id", "revision", "materialized"}}
        for scope in SCOPES:
            with self.subTest(scope=scope):
                answer = json.loads(self.resolve(self.conflict(scope)))
                self.assertEqual(set(answer), fields[scope])
                self.assertIs(answer["materialized"], True)

    def test_materialized_still_means_whole_revision(self):
        case = self.conflict("document", kind="json")
        case.rec.files["md"].write_bytes(LATE_EDIT)
        self.assertIs(json.loads(self.resolve(case))["materialized"], False)
        self.assertEqual(case.path.read_bytes(), self.program(case.rec, "document", "json"))
        self.assertEqual([row["kind"] for row in core.list_document_conflicts(case.rid)], ["md"])

    def test_failed_write_is_still_error_with_cause(self):
        for scope in SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                with failing_replace(case.path):
                    answer = self.resolve(case)
                self.assertTrue(answer.startswith("Ошибка:"), answer)
                self.assertIn("synthetic busy file", answer)


class LegacyPromiseTests(ScreenCase):
    backend = "legacy_files"
    with_database = False

    def test_legacy_mode_does_not_promise_protection(self):
        self.assertRegex(self.listing()["reason"].casefold(), "перезапи|затира")


class ConflictReadableTests(ScreenCase):
    """Невидимое, кодировка, повреждённая строка и сбои названы словами."""

    def damage(self, table, column, key, value):
        """Испортить сохранённое в базе песочницы в обход её защиты от изменений."""
        for found in self.store.db.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (table,)).fetchall():
            self.store.db.execute(f'DROP TRIGGER "{found[0]}"')
        self.store.db.execute(f"UPDATE {table} SET {column}=? WHERE {key}=?", (b"damaged", value))

    def words(self, shown):
        return " ".join(shown["removed_notes"] + shown["added_notes"] + shown["file_notes"]).casefold()

    def test_invisible_difference_is_named(self):
        line = "Анна: беру два"
        variants = {
            "bom": ("﻿" + line + "\n", "bom"),
            "space": (line + "  \n", "пробел"),
            "tab": (line + "\t\n", "табуляц"),
            "nbsp": (line.replace(" ", " ") + "\n", "неразрывн"),
            "zero width": (line.replace("беру", "бе​ру") + "\n", "нулев"),
            "line endings": (line + "\r\n", "концы строк")}
        for label, (mine, word) in variants.items():
            with self.subTest(variant=label):
                case = self.conspect_texts(line + "\n", mine)
                self.assertIn(word, self.words(self.preview(case)))
                self.assertEqual(self.copy(case), mine.encode("utf-8"))

    def test_invisible_difference_is_named_in_ordinary_long_document(self):
        lines = [f"**[{i:05d}]** реплика" for i in range(200)]
        mine = "\n\n".join(line + " " for line in lines) + "\n"
        shown = self.preview(self.conspect_texts("\n\n".join(lines) + "\n", mine))
        self.assertIn("пробел", self.words(shown), "различие не видно и не названо")

    def test_repeated_lines_do_not_make_comparison_unbounded(self):
        mine, theirs = [], []
        for part in range(200):     # около 0,27 МиБ; на пределе копии таких мест в тридцать раз больше
            mine += [f"## {part}"] + ["Да.", "п"] * 149
            theirs += [f"## {part}"] + ["Да.", "р"] * 149
        case = self.conspect_texts("\n".join(theirs) + "\n", "\n".join(mine) + "\n")
        started = time.monotonic()
        shown = self.preview(case)
        self.assertGreaterEqual(shown["removed_total"], 200 * 149)
        self.assertLess(time.monotonic() - started, 10, "работа сравнения не ограничена")

    def test_edit_in_long_line_next_to_another_edit_is_visible(self):
        line = "реплика монолога " * 200
        variants = {"inserted line before": "заметка\n" + line + "ВАЖНАЯ ПРАВКА\n",
                    "invisible sign before": "﻿" + line + "ВАЖНАЯ ПРАВКА\n"}
        for label, mine in variants.items():
            with self.subTest(variant=label):
                shown = self.preview(self.conspect_texts(line + "\n", mine))
                self.assertTrue("ВАЖНАЯ ПРАВКА" in "\n".join(shown["removed"]),
                                "правка в длинной строке не видна перед заменой")

    def test_copy_not_in_utf8_is_named(self):
        text = "Анна: моя правка\n"
        variants = {"ansi": text.encode("cp1251"), "utf-16 le": b"\xff\xfe" + text.encode("utf-16-le"),
                    "utf-16 be": b"\xfe\xff" + text.encode("utf-16-be"),
                    "utf-16 without mark": text.encode("utf-16-le")}
        for label, saved in variants.items():
            with self.subTest(encoding=label):
                case = self.conflict("conspect", saved)
                shown = self.preview(case)
                notes = " ".join(shown["file_notes"]).casefold()
                self.assertIn("кодировк", notes)
                if label in ("utf-16 le", "utf-16 be"):
                    self.assertIn(text.rstrip(), shown["removed"])
                else:
                    self.assertIn("неточн", notes)
                self.assertEqual(self.copy(case), saved)

    def test_damaged_version_does_not_break_list_and_cannot_be_written(self):
        for scope, table, key in (("document", "document_versions", "recording_id"),
                                  ("conspect", "conspect_revisions", "conspect_id")):
            with self.subTest(scope=scope):
                bad, waiting, good = self.conflict(scope), self.conflict(scope), self.conflict(scope)
                snapshot = self.snapshot(bad)
                self.resolve_directly(waiting)
                for case in (bad, waiting):
                    self.damage(table, "md", key, case.rid if scope == "document" else case.rec.conspect_id)
                for case in (bad, waiting):
                    row = self.item(case)
                    self.assertEqual(row["status"], "damaged")
                    self.assertRegex(row["problem"], CYRILLIC)
                    self.assertFalse(row["can_replace"] or row["can_retry"])
                    self.assertIsNone(self.preview(case)["snapshot"])
                    self.refused("api/conflict/retry", self.body(case))
                    self.assertEqual(case.path.read_bytes(), case.saved)
                    self.assertEqual(self.copy(case), case.saved)
                self.refused("api/conflict/resolve", {"snapshot": snapshot, "confirm": True})
                self.assert_untouched(bad, bad.saved)
                self.assertIs(self.item(good)["can_replace"], True)
                self.assertEqual(self.replace(self.snapshot(good))["outcome"], "replaced")

    def test_damaged_copy_cannot_be_replaced(self):
        case = self.conflict("document")
        snapshot = self.snapshot(case)
        self.damage("document_conflicts", "observed_bytes", "conflict_id", case.id)
        self.assertIsNone(self.preview(case)["snapshot"])
        self.refused("api/conflict/resolve", {"snapshot": snapshot, "confirm": True})
        self.assert_untouched(case, case.saved)

    def test_damaged_copy_is_not_lost_by_retry(self):
        case = self.conflict("document")
        self.resolve_directly(case)
        self.damage("document_conflicts", "observed_bytes", "conflict_id", case.id)
        status, _, _ = self.request("POST", "api/conflict/retry", self.body(case))
        self.assertTrue(case.path.read_bytes() == case.saved,
                        f"код {status}: файл заменён, а единственная его копия в базе повреждена")

    def test_failures_are_explained_in_russian(self):
        for error in (storage.StorageError("cannot open database"), storage.Busy("database is busy"),
                      sqlite3.OperationalError("database is locked"), RuntimeError("synthetic bug")):
            with self.subTest(error=str(error)), mock.patch.object(recordings, "open_store", side_effect=error):
                status, _, raw = self.request("GET", "api/conflicts/all")
                self.assertIn(status, (400, 500))
                message = json.loads(raw)["error"]
                self.assertRegex(message, CYRILLIC)
                if isinstance(error, RuntimeError):
                    self.assertIn(str(error), message)

    def test_own_refusal_is_not_called_database_error(self):
        with mock.patch.object(core, "workflow_confirm_verification",
                               side_effect=storage.Conflict("снимок подтверждения устарел")):
            status, _, raw = self.request("POST", "api/verify/confirm", {"snapshot": {}})
        self.assertEqual(status, 400)
        message = json.loads(raw)["error"]
        self.assertIn("снимок подтверждения устарел", message)
        self.assertNotIn("ошибка базы", message.casefold())

    def test_names_keep_case_from_disk(self):
        case = self.conflict("document")
        before, row, shown = self.state(), self.item(case), self.preview(case)
        self.assertEqual(row["title"], f"{case.rec.stem}.wav")
        name = next(n for n in os.listdir(case.path.parent) if n.casefold() == case.path.name.casefold())
        for found in (row["file"], shown["file"]):
            self.assertEqual(Path(found).name, name)
            self.assertIn(self.base.name, found)
        self.assertEqual(self.preview(case)["snapshot"], shown["snapshot"])
        self.assertEqual(self.state(), before)

    def test_reason_does_not_call_overwritten_file_untouched(self):
        for error in (RuntimeError("synthetic bug"), PermissionError(13, "synthetic busy file")):
            with self.subTest(error=type(error).__name__):
                rec = self.recording()
                self.save_conspect(rec)
                self.materialize(rec, "conspect")
                rec.conspect_files["md"].write_bytes(LATE_EDIT)
                for _ in range(2):
                    self.save_conspect(rec)      # в очереди две версии программы
                self.materialize(rec, "conspect")
                case = self.open_case(rec, "conspect", "md")
                snapshot = self.snapshot(case)
                original, writes = os.replace, []

                def replace(src, dst, *args, **kwargs):
                    if same_path(dst, case.path):
                        writes.append(dst)
                        if len(writes) == 2:     # первая версия записана, вторая не записалась
                            raise error
                    return original(src, dst, *args, **kwargs)
                with mock.patch("os.replace", replace):
                    result = self.replace(snapshot)
                self.assertNotEqual(case.path.read_bytes(), case.saved, "подготовка: правка уже заменена")
                self.assertEqual(result["outcome"], "not_written")
                self.assertNotIn("не тронут", result["reason"].casefold())

    def test_journal_reason_names_file_that_stops_it(self):
        case = self.conflict("conspect")
        snapshot = self.snapshot(case)
        with failing_replace(case.path):
            self.replace(snapshot)
        other = case.rec.conspect_files["json"]
        other.write_bytes(LATE_EDIT)
        reason = self.retry(case)["reason"].casefold()
        self.assertEqual(case.path.read_bytes(), case.saved)
        self.assertTrue(other.name.casefold() in reason, f"мешает другой файл свёртки, а совет о другом: {reason}")


# Страница в Node.js с маленькой подменой DOM: только то, что видит и нажимает человек.
PAGE_RUNNER = r"""
const vm = require("vm");
const {html, scenarios} = JSON.parse(require("fs").readFileSync(0, "utf8"));
const script = /<script>([\s\S]*)<\/script>/.exec(html)[1];

async function run(sc) {
  const shown = [], requests = [];
  class El {
    constructor(tag) {
      Object.assign(this, {tagName: tag.toUpperCase(), children: [], parentNode: null, own: "", html: "",
        className: "", dataset: {}, style: {}, hidden: false, open: false, value: "", checked: false, root: false});
      this.classList = {add: () => {}, remove: () => {}, toggle: () => {}, contains: () => false};
    }
    get childNodes() { return this.children; }
    get firstChild() { return this.children[0] || null; }
    get parentElement() { return this.parentNode; }
    get isConnected() { return this.root || !!(this.parentNode && this.parentNode.isConnected); }
    set textContent(t) { this.children = []; this.html = ""; this.own = String(t ?? ""); }
    get textContent() { return this.own + this.html + this.children.map(c => c.textContent).join("\n"); }
    get innerText() { return this.textContent; }
    set innerHTML(t) { this.children = []; this.own = ""; this.html = String(t); }
    get innerHTML() { return this.html; }
    append(...xs) {
      for (let x of xs) {
        if (!(x instanceof El)) { const t = new El("text"); t.own = String(x); x = t; }
        if (x.parentNode) x.remove();
        x.parentNode = this;
        this.children.push(x);
      }
    }
    appendChild(x) { this.append(x); return x; }
    prepend(...xs) { this.append(...xs); }
    insertBefore(x) { this.append(x); return x; }
    replaceChildren(...xs) {
      this.children.forEach(c => { c.parentNode = null; });
      this.children = []; this.own = ""; this.html = "";
      this.append(...xs);
    }
    remove() {
      if (!this.parentNode) return;
      this.parentNode.children = this.parentNode.children.filter(c => c !== this);
      this.parentNode = null;
    }
    all() { return this.children.flatMap(c => [c, ...c.all()]); }
    querySelectorAll(q) {
      return this.all().filter(e => q[0] === "." ? String(e.className).split(" ").includes(q.slice(1))
        : e.tagName === q.toUpperCase());
    }
    querySelector(q) { return this.querySelectorAll(q)[0] || null; }
    closest() { return null; }
    setAttribute(k, v) { this[k === "class" ? "className" : k] = v === "" ? true : v; }
    removeAttribute(k) { this[k] = false; }
    addEventListener(k, f) { this["on" + k] = f; }
    scrollIntoView() { shown.push(this.textContent); }
    focus() { shown.push(this.textContent); }
    click() { if (!this.disabled && this.onclick) return this.onclick({target: this, preventDefault() {}}); }
  }
  const ids = {};
  for (const m of html.matchAll(/<(\w+)[^>]*?\bid="([^"]+)"/g)) {
    ids[m[2]] = new El(m[1]);
    ids[m[2]].root = true;
  }
  const body = new El("body");
  body.root = true;
  const every = () => [...Object.values(ids), body].flatMap(e => [e, ...e.all()]);
  const document = {body, createElement: t => new El(t), getElementById: id => ids[id] || null,
    createTextNode: t => { const e = new El("text"); e.own = t; return e; },
    querySelector: q => q[0] === "#" ? ids[q.slice(1)] || null : null,
    querySelectorAll: () => [], addEventListener: () => {}};
  const replies = {
    "api/requests": {requests: []}, "api/conflicts/all": {available: true, reason: null, items: sc.items || []},
    "api/conflict/preview": sc.preview, "api/conflict/resolve": sc.result, "api/conflict/retry": sc.result,
    "api/status": {request_id: "q", state: "completed", readiness: {}}, "api/questions": {questions: []},
    "api/result": {recording_id: "r", segments: [], doubts: [], speakers: [], files: {}},
    "api/conspects": {conspects: []}, "api/verify/prepare": {document: "d"},
    "api/restore/prepare": {document: "d", source_revision: 1, review_revision: 1},
    "api/restore/confirm": {materialized: false}, "api/history": {revisions: []}};
  let offline = false;
  async function fetch(path) {
    const key = String(path).split("?")[0];
    requests.push(key);
    if (key === sc.fail) {
      if (sc.after) replies["api/conflicts/all"] = {available: true, reason: null, items: sc.after};
      if (sc.offline) offline = true;
      else return {ok: false, status: 400, json: async () => ({error: "отказ для проверки"})};
    }
    if (offline) throw new TypeError("Failed to fetch");
    const data = replies[key] || {};
    return {ok: true, status: 200, json: async () => JSON.parse(JSON.stringify(data)), headers: {get: () => ""}};
  }
  const context = {document, fetch, setTimeout, clearTimeout, console, encodeURIComponent, decodeURIComponent,
    requestAnimationFrame: f => setTimeout(f, 0), location: {pathname: "/t/"}, URL, confirm: () => false};
  context.window = context;
  vm.createContext(context);
  vm.runInContext(script, context);
  const settle = async () => { for (let i = 0; i < 12; i++) await new Promise(r => setTimeout(r, 0)); };
  await settle();
  const out = {};
  const card = () => every().find(e => String(e.className).split(" ").includes("conflict"));
  if (sc.request) {
    requests.length = 0;
    context.openRequest("q");
    await settle();
    out.afterOpen = [...requests];
  }
  if (sc.restore) {
    await context.prepareRestore("transcript", 1, 1, 1);
    requests.length = 0;
    await every().find(e => e.tagName === "BUTTON" && e.textContent === "Подтвердить возврат").click();
    await settle();
    out.afterRestore = [...requests];
  }
  if (card()) {
    out.card = card().textContent;
    out.buttons = card().querySelectorAll("button").map(b => b.textContent);
    out.folded = false;
    for (let e = card(); e; e = e.parentNode) {
      if ((e.tagName === "DETAILS" && !e.open) || e.hidden) out.folded = true;
    }
  }
  if (sc.compare || sc.confirm) {
    const look = card().querySelectorAll("button").find(b => /^(Заменить файл|Показать сравнение)/.test(b.textContent));
    await look.click();
    await settle();
    if (sc.fail === "api/conflict/preview") {
      out.left = every().filter(e => e.tagName === "BUTTON" && e.isConnected).map(b => b.textContent);
      out.note = ids["conflicts-note"].textContent + ids["conflicts-list"].textContent;
      out.shown = [...shown];
      return out;
    }
    const panel = card().querySelector(".panel");
    out.panel = panel.textContent;
    const ask = panel.querySelector(".ask");
    out.question = ask ? ask.textContent : "";
    if (sc.confirm) {
      requests.length = 0;
      shown.length = 0;
      await ask.querySelectorAll("button").find(b => b.textContent !== "Отмена").click();
      await settle();
      out.afterConfirm = [...requests];
      out.shown = [...shown];
      out.left = every().filter(e => e.tagName === "BUTTON" && e.isConnected).map(b => b.textContent);
      out.note = ids["conflicts-note"].textContent + ids["conflicts-list"].textContent;
    }
  }
  return out;
}

(async () => {
  const out = {};
  for (const [name, sc] of Object.entries(scenarios)) {
    try { out[name] = await run(sc); } catch (e) { out[name] = {error: String(e.stack || e)}; }
  }
  process.stdout.write(JSON.stringify(out), () => process.exit(0));
})();
"""


def page_scenarios():
    item = {"conflict_id": "c", "scope": "document", "recording_id": "r", "title": "встреча.wav", "name": None,
            "kind": "md", "relative_path": "Встреча.transcript.md", "file": "c:/x/Встреча.transcript.md",
            "observed_size": 10, "preserved": True, "status": "open", "can_replace": True, "can_retry": False}
    shown = {"conflict_id": "c", "kind": "md", "relative_path": item["relative_path"], "file": item["file"],
             "preserved": True, "preview": "моя правка", "truncated": False, "file_state": "same",
             "removed": ["моя правка"], "added": ["версия"], "removed_total": 1, "added_total": 1,
             "shown_all": True, "newlines_only": False, "also_replaces": [], "can_replace": True,
             "reason": None, "snapshot": {"any": 1}}
    closed = {"can_replace": False, "snapshot": None}
    return {
        "open": {"items": [item]},
        "changed": {"items": [item], "compare": True, "preview": {**shown, "file_state": "changed"}},
        "big": {"items": [{**item, "preserved": False, "can_replace": False, "observed_size": 9000000}],
                "compare": True,
                "preview": {**shown, **closed, "preserved": False, "preview": None, "removed": [], "added": [],
                            "removed_total": 0, "added_total": 0, "reason": "Файл больше 8 МиБ."}},
        "archive": {"items": [{**item, "kind": "archive_md", "can_replace": False}]},
        "resolved": {"items": [{**item, "status": "resolved", "can_replace": False}], "compare": True,
                     "preview": {**shown, **closed, "file_state": "program", "reason": "Уже решён."}},
        "waiting": {"items": [{**item, "status": "not_written", "can_replace": False}]},
        "request": {"request": True, "items": [item]},
        "restore": {"request": True, "restore": True, "items": []},
        "invisible": {"items": [item], "compare": True,
                      "preview": {**shown, "file_notes": ["пометка о кодировке"],
                                  "removed_notes": ["пометка о пробеле"], "added_notes": [""]}},
        "damaged": {"items": [{**item, "status": "damaged", "can_replace": False,
                               "problem": "версия повреждена"}]},
        "refused preview": {"items": [item], "compare": True, "preview": shown, "fail": "api/conflict/preview",
                            "after": [{**item, "status": "resolved", "can_replace": False}]},
        "refused replace": {"items": [item], "confirm": True, "preview": shown, "fail": "api/conflict/resolve",
                            "after": [{**item, "status": "resolved", "can_replace": False}]},
        "offline": {"items": [item], "confirm": True, "preview": shown, "fail": "api/conflict/resolve",
                    "offline": True},
        "unchecked": {"items": [item], "confirm": True, "preview": shown,
                      "result": {"outcome": "not_written", "new_conflict_id": None,
                                 "reason": "Решение принято, но проверить, что лежит в файле, не удалось."}},
        "confirm": {"request": True, "confirm": True, "items": [item], "preview": shown,
                    "result": {"outcome": "replaced", "new_conflict_id": None, "reason": None}}}


@unittest.skipUnless(shutil.which("node"), "Node.js нужен для проверки JavaScript")
class ConflictPageBehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        given = json.dumps({"html": (PROJECT / "workflow.html").read_text(encoding="utf-8"),
                            "scenarios": page_scenarios()}, ensure_ascii=False)
        done = subprocess.run([shutil.which("node"), "-e", PAGE_RUNNER], input=given, capture_output=True,
                              text=True, encoding="utf-8", timeout=60)
        cls.seen = json.loads(done.stdout) if done.returncode == 0 and done.stdout else {}
        cls.failure = done.stderr

    def seen_in(self, name):
        self.assertIn(name, self.seen, self.failure)
        self.assertNotIn("error", self.seen[name], self.seen[name].get("error"))
        return self.seen[name]

    def test_question_does_not_promise_replace_of_changed_file(self):
        question = self.seen_in("changed")["question"]
        self.assertTrue(question, "вопрос перед действием остаётся")
        self.assertNotIn("будет заменён версией программы", question)

    def test_unsaved_big_file_is_not_called_equal_to_copy(self):
        self.assertNotIn("равен вашей сохранённой копии", self.seen_in("big")["panel"])

    def test_dead_end_does_not_advise_to_repeat_edit(self):
        for name in ("big", "archive"):
            with self.subTest(conflict=name):
                self.assertNotIn("повторите", self.seen_in(name)["card"].casefold())

    def test_open_conflict_tells_how_edit_reaches_transcript(self):
        self.assertIn("карточ", self.seen_in("open")["card"].casefold())

    def test_resolved_copy_is_not_described_as_future_replace(self):
        self.assertNotIn("при замене", self.seen_in("resolved")["panel"].casefold())

    def test_waiting_for_second_file_is_not_called_failed_write(self):
        self.assertNotIn("не удалось", self.seen_in("waiting")["card"].casefold())

    def test_saved_copies_are_folded(self):
        self.assertTrue(self.seen_in("resolved")["folded"], "решённые копии отодвигают форму запуска")

    def test_opening_request_refreshes_conflicts(self):
        self.assertIn("api/conflicts/all", self.seen_in("request")["afterOpen"])

    def test_restore_refreshes_conflicts(self):
        self.assertIn("api/conflicts/all", self.seen_in("restore")["afterRestore"])

    def test_replace_refreshes_open_request(self):
        self.assertIn("api/status", self.seen_in("confirm")["afterConfirm"])

    def test_outcome_is_brought_into_view(self):
        self.assertTrue(any("заменён" in text for text in self.seen_in("confirm")["shown"]),
                        "итог остаётся за краем экрана")

    def test_notes_are_shown(self):
        panel = self.seen_in("invisible")["panel"]
        self.assertIn("пометка о кодировке", panel)
        self.assertIn("пометка о пробеле", panel)

    def test_damaged_row_has_reason_and_no_replace(self):
        seen = self.seen_in("damaged")
        self.assertIn("версия повреждена", seen["card"])
        self.assertFalse([b for b in seen["buttons"] if b.startswith(("Заменить", "Повторить"))])

    def test_refusal_refreshes_list_and_stays_visible(self):
        for name in ("refused preview", "refused replace"):
            with self.subTest(action=name):
                seen = self.seen_in(name)
                self.assertFalse([b for b in seen["left"] if b.startswith(("Заменить файл", "Да,"))])
                self.assertIn("отказ для проверки", seen["note"])
                self.assertTrue(any("отказ для проверки" in text for text in seen["shown"]))

    def test_lost_connection_is_russian_and_leaves_no_replace(self):
        seen = self.seen_in("offline")
        self.assertFalse([b for b in seen["left"] if b.startswith(("Заменить файл", "Да,"))])
        self.assertIn("нет связи", seen["note"].casefold())
        self.assertNotIn("failed to fetch", seen["note"].casefold())

    def test_unchecked_outcome_is_not_called_not_replaced(self):
        self.assertNotIn("НЕ заменён", self.seen_in("unchecked")["note"])


class ConflictPageTests(unittest.TestCase):
    """Только видимое без браузера; поведение страницы проверяется в браузере."""

    def setUp(self):
        self.html = (PROJECT / "workflow.html").read_text(encoding="utf-8")

    def test_page_offers_honest_actions(self):
        for text in ("Скачать мою правку", "Заменить файл версией программы", "Повторить", "Сохранённые копии"):
            with self.subTest(text=text):
                self.assertTrue(text in self.html, f"на странице нет: {text}")
        self.assertFalse("Сохранить внешнюю правку" in self.html, "осталась кнопка мнимого сохранения")

    def test_page_asks_by_its_own_element(self):
        dialogs = re.findall(r"(?<![\w.$])(?:(?:window|self|globalThis)\.)?(?:confirm|alert|prompt)\s*\(",
                             self.html)
        self.assertEqual(dialogs, [])

    def test_conflicts_are_at_top_outside_request_detail(self):
        conflicts = self.html.index('id="conflicts"')
        self.assertLess(conflicts, self.html.index('id="audio"'))
        self.assertLess(conflicts, self.html.index('id="detail"'))

    def test_section_does_not_promise_protection_outside_database(self):
        section = self.html[self.html.index('id="conflicts"'):]
        self.assertIn("вне базы", section[:section.index("</section>")].casefold())

    def test_page_uses_common_list(self):
        self.assertTrue("api/conflicts/all" in self.html, "страница не читает общий список конфликтов")


if __name__ == "__main__":
    unittest.main()
