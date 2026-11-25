"""О1, О2, О4: копия проверяется там, где действительно заменяется файл."""
import ast
import hashlib
import json
import uuid
from types import SimpleNamespace
from unittest import mock

import confirm
import conspect
import core
import documents
import recordings
import storage
import test_conflict_replace as fixtures
import workflow


class WriteGuardTests(fixtures.ScreenCase):
    def damage(self, case, field="observed_bytes"):
        table = case.scope + "_conflicts"
        # Только синтетическая база теста: имитируем повреждение после принятого решения.
        for row in self.store.db.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (table,)).fetchall():
            self.store.db.execute(f'DROP TRIGGER "{row[0]}"')
        value = len(case.saved) + 1 if field == "observed_size" else b"damaged"
        self.store.db.execute(f"UPDATE {table} SET {field}=? WHERE conflict_id=?", (value, case.id))

    def duplicate(self, case, **changes):
        """Новая открытая копия того же файла; прежняя история остаётся нетронутой."""
        row = dict(self.store.db.execute(
            f"SELECT * FROM {case.scope}_conflicts WHERE conflict_id=?", (case.id,)).fetchone())
        row.update(conflict_id=str(uuid.uuid4()), observed_bytes=case.saved, resolution_operation_id=None)
        row.update(changes)
        self.store.db.execute(
            f"INSERT INTO {case.scope}_conflicts ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
            tuple(row.values()))
        table, key, ident, revision = (
            ("document_outbox", "recording_id", case.rid, row["document_revision"])
            if case.scope == "document" else
            ("conspect_outbox", "conspect_id", row["conspect_id"], row["revision"]))
        rev_column = "document_revision" if case.scope == "document" else "revision"
        self.store.db.execute(
            f"UPDATE {table} SET state='conflict' WHERE {key}=? AND {rev_column}=?", (ident, revision))
        return SimpleNamespace(**{**vars(case), "id": row["conflict_id"]})

    def conspect_pair(self):
        rec = self.recording()
        objects = []
        for name in ("А-первая", "Я-вторая"):
            obj = SimpleNamespace(**vars(rec))
            obj.conspect_id = str(uuid.uuid4())
            obj.conspect_files = {kind: rec.root / f"{rec.stem}.{name}.conspect.{kind}"
                                 for kind in ("json", "md")}
            objects.append(obj)
            for revision in (1, 2):
                payload = {"name": name, "revision": revision}
                self.store.save_conspect(
                    rec.rid, obj.conspect_id, name, 1, revision - 1 or None, payload,
                    json.dumps(payload).encode("utf-8"), f"Программа {revision}\n".encode("utf-8"),
                    obj.conspect_files["json"].name, obj.conspect_files["md"].name, rec.root_id,
                    str(uuid.uuid4()), recordings.operation_context(self.store, "human"))
                if revision == 1:
                    recordings.materialize_conspects(self.store, rec.rid)
                    obj.conspect_files["md"].write_bytes("Правка человека\n".encode("utf-8"))
            recordings.materialize_conspects(self.store, rec.rid)
        cases = []
        for obj in objects:
            row = self.store.db.execute(
                "SELECT * FROM conspect_conflicts WHERE conspect_id=? AND kind='md' "
                "AND resolution_operation_id IS NULL", (obj.conspect_id,)).fetchone()
            cases.append(SimpleNamespace(
                rec=obj, rid=rec.rid, id=row["conflict_id"], scope="conspect", kind="md",
                relative=row["relative_path"], path=obj.conspect_files["md"], saved=bytes(row["observed_bytes"])))
        return cases

    def pair(self, scope):
        if scope == "conspect":
            return self.conspect_pair()
        ident = uuid.uuid4().int & ~1
        bad_rec = self.recording(ident=str(uuid.UUID(int=ident)))
        bad = self.conflict("document", rec=bad_rec)
        other = self.recording(root=bad_rec.root, ident=str(uuid.UUID(int=ident + 1)))
        return bad, self.conflict("document", rec=other)

    def test_neighbor_copy_is_protected_for_all_entry_points(self):
        for scope in fixtures.SCOPES:
            for entry in ("sync", "retry", "replace", "agent"):
                with self.subTest(scope=scope, entry=entry):
                    # Отдельный root для каждого варианта; повреждённая запись идёт первой.
                    bad, target = self.pair(scope)
                    self.resolve_directly(bad)
                    self.damage(bad)
                    if entry in ("sync", "retry"):
                        self.resolve_directly(target)
                    try:
                        if entry == "sync":
                            self.materialize(target.rec, scope)
                        elif entry == "retry":
                            self.retry(target)
                        elif entry == "replace":
                            self.replace(self.snapshot(target))
                        else:
                            getattr(core, f"resolve_{scope}_conflict")(target.rid, target.id)
                    except (core.CoreError, storage.StorageError):
                        pass  # До записи — отказ; после решения экран также может дать not_written.
                    self.assertEqual(bad.path.read_bytes(), bad.saved,
                                     "Соседний файл заменён при повреждённой копии")
                    self.assertNotEqual(target.path.read_bytes(), target.saved,
                                        "Исправный файл остановлен чужой повреждённой копией")

    def test_current_override_checks_size_too(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.resolve_directly(case)
                self.damage(case, "observed_size")
                with self.assertRaises(storage.StorageError):
                    self.materialize(case.rec, scope)
                self.assertEqual(case.path.read_bytes(), case.saved)

    def test_new_intact_resolution_supersedes_damaged_history(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.resolve_directly(case)
                self.damage(case)
                fresh = self.duplicate(case)
                result = self.replace(self.snapshot(fresh))
                self.assertEqual(result["outcome"], "replaced")
                self.assertEqual(self.copy(fresh), case.saved)

    def test_already_written_program_does_not_need_historical_copy(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.resolve_directly(case)
                self.damage(case)
                desired = self.program(case.rec, scope, "md")
                case.path.write_bytes(desired)
                self.materialize(case.rec, scope)
                self.assertEqual(case.path.read_bytes(), desired)

    def test_new_external_edit_is_kept_even_when_old_copy_is_damaged(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.resolve_directly(case)
                self.damage(case)
                case.path.write_bytes(fixtures.LATE_EDIT)
                self.materialize(case.rec, scope)
                fresh = self.open_case(case.rec, scope, "md")
                self.assertEqual(fresh.saved, fixtures.LATE_EDIT)
                self.assertEqual(case.path.read_bytes(), fixtures.LATE_EDIT)

    def test_open_large_neighbor_does_not_refuse_a_valid_decision(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                big = b"x" * (8 * 1024 * 1024 + 1)
                paths = case.rec.files if scope == "document" else case.rec.conspect_files
                relative = paths["json"].relative_to(case.rec.root).as_posix()
                paths["json"].write_bytes(big)
                self.duplicate(case, kind="json", relative_path=relative, observed_bytes=None,
                               observed_size=len(big), observed_hash=hashlib.sha256(big).hexdigest())
                result = self.replace(self.snapshot(case))
                self.assertEqual(result["outcome"], "not_written")
                self.assertIn("ждёт решения", result["reason"])
                self.assertFalse(self.is_open(case))
                self.assertEqual(case.path.read_bytes(), case.saved)
                self.assertTrue(paths["json"].read_bytes() == big)

    def test_journal_recovery_does_not_lose_the_copy(self):
        case = self.conflict("conspect")
        with fixtures.failing_replace(case.path):
            self.replace(self.snapshot(case))
        self.damage(case)
        result = self.retry(case)
        self.assertEqual(result["outcome"], "not_written")
        self.assertIn("копия", result["reason"].casefold())
        self.assertEqual(case.path.read_bytes(), case.saved)

    def test_recovery_after_replacement_restores_edit_before_guard(self):
        for kind in ("json", "md"):
            with self.subTest(kind=kind):
                case = self.conflict("conspect")
                target = case.rec.conspect_files[kind]
                original = conspect._replace_bytes

                def fail_after_write(path, value):
                    original(path, value)
                    if fixtures.same_path(path, target):
                        raise OSError("synthetic interruption after replacement")

                with mock.patch.object(conspect, "_replace_bytes", side_effect=fail_after_write):
                    self.replace(self.snapshot(case))
                self.damage(case)
                with self.assertRaises(storage.StorageError):
                    self.materialize(case.rec, "conspect")
                self.assertEqual(case.path.read_bytes(), case.saved)

    def test_damaged_copy_does_not_stop_a_neighbour_recording(self):
        for bad_first in (True, False):
            with self.subTest(bad_first=bad_first):
                low = uuid.uuid4().int & ~1
                ids = [str(uuid.UUID(int=low + i)) for i in (0, 1)]
                bad_id, good_id = ids if bad_first else ids[::-1]
                bad = self.conflict("document", rec=self.recording(ident=bad_id))
                self.resolve_directly(bad)
                self.damage(bad)
                good = self.recording(root=bad.rec.root, ident=good_id)    # создание и есть запись файлов соседа
                changed = self.change_text(good, "правка")
                self.assertIn((good.rid, changed["document_revision"]), self.materialize(good, "document"))
                self.assertEqual(good.files["md"].read_bytes(), self.program(good, "document", "md"))
                self.assertEqual(core.sync_recording_documents(good.rid)["document_sync"], "synced")
                self.assertEqual(bad.path.read_bytes(), bad.saved)
                with self.assertRaises(storage.StorageError):
                    self.materialize(bad.rec, "document")

    def test_damaged_copy_does_not_stop_a_neighbour_conspect(self):
        for name in ("А-полная", "Я-полная"):    # до и после повреждённой «Краткий»
            with self.subTest(name=name):
                bad = self.conflict("conspect")
                self.resolve_directly(bad)
                self.damage(bad)
                rec = bad.rec
                paths = {kind: rec.root / f"{rec.stem}.{name}.conspect.{kind}" for kind in ("json", "md")}
                value = json.dumps({"recording_id": rec.rid, "name": name}, ensure_ascii=False).encode("utf-8")
                recordings.save_conspect_files(rec.files["json"], [(paths["json"], value), (paths["md"], b"other\n")])
                self.assertEqual(paths["md"].read_bytes(), b"other\n")
                self.assertEqual(bad.path.read_bytes(), bad.saved)
                with self.assertRaises(storage.StorageError):
                    self.materialize(rec, "conspect")

    def test_neighbour_decision_through_mcp_succeeds(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                bad, target = self.pair(scope)
                self.resolve_directly(bad)
                self.damage(bad)
                result = getattr(core, f"resolve_{scope}_conflict")(target.rid, target.id)
                self.assertTrue(result["materialized"])
                self.assertEqual(bad.path.read_bytes(), bad.saved)

    def test_later_revision_waits_for_the_damaged_conspect(self):
        case = self.conflict("conspect")
        self.resolve_directly(case)
        self.damage(case)
        self.save_conspect(case.rec)
        with self.assertRaises(storage.StorageError):
            self.materialize(case.rec, "conspect")
        self.assertEqual(case.path.read_bytes(), case.saved)
        self.assertEqual(self.store.db.execute(
            "SELECT count(*) FROM conspect_conflicts WHERE resolution_operation_id IS NULL AND conspect_id=?",
            (case.rec.conspect_id,)).fetchone()[0], 0)

    def test_damaged_copy_advice_really_leads_out(self):
        for scope in fixtures.SCOPES:
            for kind in (("md", "json", "manifest") if scope == "document" else ("md", "json")):
                with self.subTest(scope=scope, kind=kind):
                    case = self.conflict(scope, body=lambda original: original + b"xy", kind=kind)
                    self.resolve_directly(case)
                    self.damage(case)
                    reason = self.retry(case)["reason"]
                    self.assertIn("измените", reason.casefold())
                    self.assertNotIn("восстановите копию", reason.casefold())
                    self.assertNotIn("если ошибка повторится", reason.casefold())
                    case.path.write_bytes(case.saved[:-1])    # ровно то, что советует причина
                    self.assertEqual(self.retry(case)["outcome"], "changed")
                    fresh = self.open_case(case.rec, scope, kind)
                    self.assertEqual(self.replace(self.snapshot(fresh))["outcome"], "replaced")
                    self.assertEqual(self.copy(fresh), case.saved[:-1])

    def test_advice_works_at_the_copy_size_limit_and_for_an_empty_file(self):
        limit = documents.MAX_CONFLICT_BYTES
        for scope in fixtures.SCOPES:
            for size in (limit, limit - 1, 1, 0):
                with self.subTest(scope=scope, size=size):
                    case = self.conflict(scope, body=b"x" * size)
                    self.resolve_directly(case)
                    self.damage(case)
                    reason = self.retry(case)["reason"].casefold()
                    self.assertIn("удалите один", reason)
                    self.assertIn("пустой файл", reason)
                    self.assertIn("не больше 8 миб", reason)
                    self.assertNotIn("добавьте пустую строку", reason)
                    changed = case.saved[:-1] if size else b"x"
                    case.path.write_bytes(changed)
                    self.assertEqual(self.retry(case)["outcome"], "changed")
                    fresh = self.open_case(case.rec, scope, "md")
                    self.assertTrue(fresh.saved == changed, "новая копия не сохранена")
                    self.assertEqual(self.replace(self.snapshot(fresh))["outcome"], "replaced")

    def test_advice_names_the_case_when_the_file_becomes_the_program_version(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                rec = self.recording()
                if scope == "conspect":
                    self.save_conspect(rec)
                    self.materialize(rec, scope)
                    path = rec.conspect_files["md"]
                    path.write_bytes(b"temporary edit")
                    self.save_conspect(rec)
                else:
                    path = rec.files["md"]
                path.write_bytes(self.program(rec, scope, "md") + b"x")
                self.materialize(rec, scope)
                case = self.open_case(rec, scope, "md")
                self.resolve_directly(case)
                self.damage(case)
                self.assertIn("совпадёт с версией программы", self.retry(case)["reason"])
                path.write_bytes(case.saved[:-1])    # совет выполнен, и правки в файле больше нет
                self.assertEqual(self.item(case)["status"], "resolved")
                self.assertTrue(self.materialize(rec, scope))

    def test_unresolvable_neighbour_is_named_before_and_after_the_decision(self):
        rec = self.recording()
        self.save_conspect(rec)
        self.materialize(rec, "conspect")
        rec.conspect_files["md"].write_bytes(b"my edit\n")
        rec.conspect_files["json"].write_bytes(b"x" * (8 * 1024 * 1024 + 1))
        self.save_conspect(rec)
        self.materialize(rec, "conspect")
        case = self.open_case(rec, "conspect", "md")
        shown = self.preview(case)
        name = rec.conspect_files["json"].name
        self.assertTrue(any(name in note and "нельзя" in note for note in shown["file_notes"]), shown["file_notes"])
        reason = self.replace(shown["snapshot"])["reason"]
        self.assertIn(name, reason)
        self.assertIn("нельзя", reason)
        self.assertNotIn("когда вы решите", reason.casefold())

    def test_own_damaged_file_is_not_reported_as_success(self):
        md, other = self.both_conspect_files()
        self.resolve_directly(other)
        self.damage(other)
        with self.assertRaises(core.CoreError):
            core.resolve_conspect_conflict(md.rid, md.id)
        self.assertEqual(md.path.read_bytes(), md.saved)
        self.assertEqual(other.path.read_bytes(), other.saved)

    def test_mcp_window_warns_not_to_edit_during_replacement(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                _, asked = fixtures.ConflictMCPTests.mcp_resolve(self, case)
                self.assertIn("пока идёт замена", "\n".join(asked.call_args.args[2]).casefold())

    def test_workflow_conspect_is_not_stopped_by_a_neighbour(self):
        bad = self.conflict("conspect")
        self.resolve_directly(bad)
        self.damage(bad)
        rec, request_id, name = bad.rec, str(uuid.uuid4()), "Заказанный"
        ident = str(uuid.uuid5(uuid.UUID(rec.rid), "conspect:workflow-" + request_id))
        paths = {kind: rec.root / f"{rec.stem}.{name}.conspect.{kind}" for kind in ("json", "md")}
        payload = {"name": name}
        self.store.save_conspect(
            rec.rid, ident, name, 1, None, payload, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            b"ordered\n", paths["json"].name, paths["md"].name, rec.root_id, str(uuid.uuid4()),
            recordings.operation_context(self.store, "human"))
        workflow._ensure_conspect_materialized(self.store, {"recording_id": rec.rid, "request_id": request_id})
        self.assertEqual(paths["md"].read_bytes(), b"ordered\n")
        self.assertEqual(bad.path.read_bytes(), bad.saved)

    def test_waiting_replacement_is_named_in_preview_and_snapshot(self):
        md, other = self.both_conspect_files()
        shown = self.preview(md)
        self.assertEqual(shown.get("waits_for"), [other.path.name])
        self.resolve_directly(other)    # ожидание кончилось: прежнее «сейчас не изменится» уже неверно
        self.refused("api/conflict/resolve", {"snapshot": shown["snapshot"], "confirm": True})
        self.assertEqual(md.path.read_bytes(), md.saved)
        self.assertTrue(self.is_open(md))
        self.assertEqual(self.preview(md).get("waits_for"), [])

    def test_waiting_document_window_does_not_promise_replacement_now(self):
        case = self.conflict("document")
        archive = self.duplicate(case, kind="archive_md", relative_path="history/прежняя.md")
        _, asked = fixtures.ConflictMCPTests.mcp_resolve(self, case)
        action, _, lines = asked.call_args.args
        text = "\n".join([action, *lines]).casefold()
        self.assertIn("сейчас файл не изменится", text)
        self.assertIn("прежняя.md", text)
        self.assertNotIn("заменить файл на диске", text)
        self.assertEqual(case.path.read_bytes(), case.saved)
        self.assertTrue(self.is_open(archive))

    def test_waiting_conspect_window_does_not_promise_replacement_now(self):
        md, other = self.both_conspect_files()
        _, asked = fixtures.ConflictMCPTests.mcp_resolve(self, md)
        action, _, lines = asked.call_args.args
        text = "\n".join([action, *lines]).casefold()
        self.assertIn("сейчас файл не изменится", text)
        self.assertIn(other.path.name.casefold(), text)
        self.assertNotIn("заменить файл на диске", text)
        self.assertEqual(md.path.read_bytes(), md.saved)

    def test_copy_with_a_wrong_size_is_not_downloaded(self):
        case = self.conflict("document")
        self.damage(case, "observed_size")
        status, _, _ = self.request("POST", "api/conflict/copy", self.body(case))
        self.assertEqual(status, 400)


class PromiseTests(fixtures.ScreenCase):
    """Обещание равно делу: что сказано перед «Да» и в карточке, то программа потом и делает с файлом."""

    damage = WriteGuardTests.damage
    duplicate = WriteGuardTests.duplicate
    ACTION = {"replace": "заменить файл", "close": "закрыть конфликт", "recopy": "не будет заменён",
              "wait": "сейчас файл не изменится", "stopped": "сейчас файл не изменится"}

    def prepare(self, scope, kind, state, neighbour):
        """Открытый конфликт файла kind; neighbour — другой файл того же объекта."""
        target = self.conflict(scope, body=lambda original: original + b"xy", kind=kind)
        if neighbour:
            paths = target.rec.files if scope == "document" else target.rec.conspect_files
            other_kind = "json" if kind == "md" else "md"
            saved = paths[other_kind].read_bytes() + b"other"
            paths[other_kind].write_bytes(saved)
            other = self.duplicate(
                target, kind=other_kind, relative_path=paths[other_kind].relative_to(target.rec.root).as_posix(),
                observed_bytes=saved, observed_size=len(saved), observed_hash=hashlib.sha256(saved).hexdigest())
            other.kind = other_kind
            if neighbour == "open damaged":
                self.damage(other)
            elif neighbour in ("decided", "decided damaged", "decided changed"):
                self.resolve_directly(other)
                if neighbour == "decided damaged":
                    self.damage(other)
                if neighbour == "decided changed":    # решение принято, а файл потом снова изменили
                    paths[other_kind].write_bytes(saved + b"later")
        if state == "changed":
            target.path.write_bytes(fixtures.LATE_EDIT)
        elif state == "program":
            target.path.write_bytes(self.program(target.rec, scope, kind))
        return target

    def happened(self, case, before, outcome):
        now = case.path.read_bytes()
        if outcome == "changed":
            self.assertEqual(now, before)
            return "recopy"
        if outcome == "replaced":
            self.assertEqual(now, self.program(case.rec, case.scope, case.kind))
            return "close" if now == before else "replace"
        self.assertEqual(now, before)
        return "unchanged"

    def test_promise_before_yes_is_what_happens(self):
        for scope in fixtures.SCOPES:
            for kind in ("md", "json"):
                for state in ("same", "changed", "program"):
                    for neighbour in (None, "open", "open damaged", "decided", "decided damaged"):
                        with self.subTest(scope=scope, kind=kind, state=state, neighbour=neighbour):
                            case = self.prepare(scope, kind, state, neighbour)
                            _, asked = fixtures.ConflictMCPTests.mcp_resolve(
                                self, case, side_effect=confirm.NotConfirmed("проба текста"))
                            action, _, lines = asked.call_args.args
                            shown = self.preview(case)
                            promised = shown.get("effect")
                            before = case.path.read_bytes()
                            done = self.happened(case, before, self.replace(shown["snapshot"])["outcome"])
                            self.assertEqual("unchanged" if promised in ("wait", "stopped") else promised, done)
                            self.assertIn(self.ACTION[promised], action.casefold())
                            if promised != "replace":
                                self.assertNotIn("заменить файл на диске", "\n".join(lines).casefold())

    def test_card_promise_is_what_retry_does(self):
        for scope in fixtures.SCOPES:
            for kind in ("md", "json"):
                for state in ("same", "changed"):
                    for damaged in (False, True):
                        for neighbour in (None, "decided", "decided damaged"):
                            with self.subTest(scope=scope, kind=kind, state=state, damaged=damaged,
                                              neighbour=neighbour):
                                # своя база на случай: общий список конфликтов иначе растёт с каждым случаем (n²)
                                self.new_database()
                                case = self.prepare(scope, kind, state, neighbour)
                                self.resolve_directly(case)
                                if damaged:
                                    self.damage(case)
                                card = self.item(case)
                                self.assertEqual(bool(card.get("copy_problem")), damaged)
                                self.assertTrue(card["can_retry"])
                                promised = card.get("next_effect")
                                before = case.path.read_bytes()
                                done = self.happened(case, before, self.retry(case)["outcome"])
                                self.assertEqual("unchanged" if promised == "stopped" else promised, done)
                                self.assertEqual("может довести" in card["next_step"], done == "replace")

    def test_waiting_damaged_copy_never_promises_a_later_replacement(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.prepare(scope, "md", "same", "open")
                reason = self.replace(self.snapshot(case))["reason"]
                self.assertIn("допишет", reason)
                self.damage(case)
                card = self.item(case)
                self.assertTrue(card["copy_problem"])
                self.assertNotIn("допишет", card["next_step"])
                self.assertNotIn("может довести", card["next_step"])

    def test_open_damaged_copy_is_refused_before_the_window(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.damage(case)
                card = self.item(case)
                self.assertTrue(card.get("copy_problem"))
                self.assertFalse(card["can_replace"])
                response, asked = fixtures.ConflictMCPTests.mcp_resolve(self, case)
                asked.assert_not_called()
                self.assertIn("повреждена", response)
                self.assertTrue(self.is_open(case))

    def test_missing_file_with_damaged_copy_does_not_advise_downloading_it(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                self.resolve_directly(case)
                self.damage(case)
                case.path.unlink()
                for text in (self.item(case).get("next_step") or "нет пояснения", self.retry(case)["reason"]):
                    self.assertNotIn("Скачайте копию", text)
                    self.assertIn("повреждена", text)

    # Т13: сосед «решение принято, файл изменён после решения». Новое состояние соседа — только там, где путь кода
    # разный: у документа важен порядок файлов (md раньше json) и состояние самого файла, у свёртки порядка нет
    # (внешние правки ищутся по всем файлам сразу), поэтому у неё один вид файла. Состояние «уже равен версии
    # программы» идёт общим путём (закрыть конфликт, сосед не важен) и покрыто матрицей выше.
    # (область, файл, его состояние) -> обещание. Порядок записи документа md → json → manifest: изменённый md
    # даёт новый конфликт по себе раньше соседа json, а сосед md останавливает json при любом его состоянии.
    NEIGHBOUR_CHANGED = {("document", "md", "same"): "stopped", ("document", "md", "changed"): "recopy",
                         ("document", "json", "same"): "stopped", ("document", "json", "changed"): "stopped",
                         ("conspect", "md", "same"): "stopped", ("conspect", "md", "changed"): "recopy"}

    def new_conflicts_on(self, case):
        """Виды файлов, по которым запись открыла новые конфликты."""
        return sorted(row["kind"] for row in self.open_rows(case.rec, case.scope) if row["conflict_id"] != case.id)

    def test_neighbour_changed_after_decision_promise_is_what_happens(self):
        for (scope, kind, state), expected in self.NEIGHBOUR_CHANGED.items():
            with self.subTest(scope=scope, kind=kind, state=state):
                case = self.prepare(scope, kind, state, "decided changed")
                other = "json" if kind == "md" else "md"
                other_name = (case.rec.files if scope == "document" else case.rec.conspect_files)[other].name
                _, asked = fixtures.ConflictMCPTests.mcp_resolve(
                    self, case, side_effect=confirm.NotConfirmed("проба текста"))
                action, _, lines = asked.call_args.args
                shown = self.preview(case)
                promised = shown["effect"]
                self.assertEqual(promised, expected)
                # экран и писатель решают одинаково: где экран называет остановку соседом, там писатель встаёт
                # на соседе; где обещает новый конфликт по этому файлу — открывает его по этому файлу
                if promised == "stopped":
                    self.assertEqual(shown["stops_for"], [other_name])
                    self.assertEqual(shown["stops_changed"], [other_name])
                    self.assertIn("изменили после", "\n".join(lines))
                    self.assertIn(other_name, "\n".join(lines))
                before = case.path.read_bytes()
                done = self.happened(case, before, self.replace(shown["snapshot"])["outcome"])
                self.assertEqual("unchanged" if promised in ("wait", "stopped") else promised, done)
                self.assertIn(self.ACTION[promised], action.casefold())
                self.assertNotEqual(promised, "replace")
                self.assertNotIn("заменить файл на диске", "\n".join(lines).casefold())
                fresh = self.new_conflicts_on(case)
                if promised == "recopy":    # у свёртки новый конфликт и по соседу: правки ищутся по всем файлам
                    self.assertIn(kind, fresh)
                else:
                    self.assertEqual(fresh, [other])

    def test_card_promise_with_neighbour_changed_after_decision_is_what_retry_does(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.prepare(scope, "md", "same", "decided changed")
                self.resolve_directly(case)
                card = self.item(case)
                self.assertEqual(card["next_effect"], "stopped")
                self.assertIn("изменили после", card["next_step"])
                self.assertNotIn("повреждённая копия", card["next_step"])
                before = case.path.read_bytes()
                self.assertEqual(self.happened(case, before, self.retry(case)["outcome"]), "unchanged")
                self.assertEqual(self.new_conflicts_on(case), ["json"])

    def test_snapshot_without_the_new_stop_reason_is_refused(self):
        """Страница, открытая до обновления, обещала замену, не зная соседа, изменённого после решения: снимок без
        этой остановки устарел, заменять по нему нельзя. Состав снимка прежний — причина входит в stops_for."""
        case = self.prepare("document", "md", "same", "decided changed")
        snapshot = dict(self.snapshot(case))
        self.assertNotIn("stops_changed", snapshot)
        snapshot["stops_for"] = []    # прежний экран соседа не видел и обещал замену
        before = case.path.read_bytes()
        self.refused("api/conflict/resolve", {"snapshot": snapshot, "confirm": True})
        self.assertEqual(case.path.read_bytes(), before)
        self.assertTrue(self.is_open(case))


class OrderRuleTests(fixtures.unittest.TestCase):
    """Порядок отказов набора документа — одна функция; запись встаёт на первом файле в порядке md → json → manifest."""

    def test_refusals_follow_the_write_order(self):
        verdicts = {"manifest": "external", "json": "damaged_copy", "md": "external"}
        self.assertEqual(documents.refusals(verdicts), ["md", "json", "manifest"])
        self.assertEqual(documents.refusals({"md": "ok", "json": "stale", "manifest": "missing"}), [])
        self.assertEqual(documents.refusals({"md": "accepted", "json": "error"}), ["json"])

    def test_what_stops_a_file_depends_on_its_place_and_its_own_state(self):
        self.assertEqual(documents.stopped_by({"md": "accepted", "json": "external"}, "md"), ["json"])
        self.assertEqual(documents.stopped_by({"md": "external", "json": "external"}, "md"), [])
        self.assertEqual(documents.stopped_by({"md": "external", "json": "external"}, "json"), ["md"])
        self.assertEqual(documents.stopped_by({"md": "external", "json": "accepted"}, "json"), ["md"])


class OneFileRuleTests(fixtures.unittest.TestCase):
    """Т14: правило допустимости текущего файла записано один раз — в documents.file_verdict."""

    def test_only_file_verdict_compares_the_current_file_with_allowed_versions(self):
        found = []
        for source in sorted(fixtures.PROJECT.glob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for function in ast.walk(tree):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(function):
                    if (isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.NotIn)
                            and isinstance(node.left, ast.Name) and "current" in node.left.id
                            and isinstance(node.comparators[0], ast.Tuple)):
                        found.append(f"{source.name}:{function.name}")
        # conspect.recover_pending — другое правило: откат прерванной записи по журналу (файл — старый или новый)
        self.assertEqual(sorted(set(found)), ["conspect.py:recover_pending", "documents.py:file_verdict"])

