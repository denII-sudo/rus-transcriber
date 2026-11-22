"""О1, О2, О4: копия проверяется там, где действительно заменяется файл."""
import hashlib
import json
import uuid
from types import SimpleNamespace

import core
import recordings
import storage
import test_conflict_replace as fixtures


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
