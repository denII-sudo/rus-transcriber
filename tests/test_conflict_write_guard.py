"""О1, О2, О4: копия проверяется там, где действительно заменяется файл."""
import hashlib
import json
import uuid
from types import SimpleNamespace
from unittest import mock

import conspect
import core
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
                    case = self.conflict(scope, kind=kind)
                    self.resolve_directly(case)
                    self.damage(case)
                    reason = self.retry(case)["reason"]
                    self.assertIn("измените", reason.casefold())
                    self.assertNotIn("восстановите копию", reason.casefold())
                    self.assertNotIn("если ошибка повторится", reason.casefold())
                    case.path.write_bytes(case.saved + b"\n")    # ровно то, что советует причина
                    self.assertEqual(self.retry(case)["outcome"], "changed")
                    fresh = self.open_case(case.rec, scope, kind)
                    self.assertEqual(self.replace(self.snapshot(fresh))["outcome"], "replaced")
                    self.assertEqual(self.copy(fresh), case.saved + b"\n")

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

    def test_copy_with_a_wrong_size_is_not_downloaded(self):
        case = self.conflict("document")
        self.damage(case, "observed_size")
        status, _, _ = self.request("POST", "api/conflict/copy", self.body(case))
        self.assertEqual(status, 400)
