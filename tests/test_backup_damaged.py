"""Резерв базы при повреждённой копии конфликта: создаётся с предупреждением; другое повреждение — отказ."""
import asyncio
import contextlib
import hashlib
import io
import json
import shutil
import sqlite3
import subprocess
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import confirm
import core
import documents
import rt
import server
import storage
import test_conflict_replace as fixtures

STAGES = ("open", "decided", "history")


class BackupCase(fixtures.ScreenCase):
    def case(self, scope, stage="open"):
        case = self.conflict(scope)
        if stage != "open":
            self.resolve_directly(case)
        if stage == "history":
            self.materialize(case.rec, scope)
        return case

    def damage(self, case, field="observed_bytes"):
        value = {"observed_bytes": b"damaged", "observed_size": len(case.saved) + 1,
                 "observed_hash": "0" * 64}[field]
        self.break_row(case.scope + "_conflicts", field, value, "conflict_id=?", (case.id,))

    def break_row(self, table, field, value, where="1", args=()):
        """Только синтетическая база теста: повреждение в обход защит."""
        triggers = [tuple(row) for row in self.store.db.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (table,))]
        for name, _ in triggers:
            self.store.db.execute(f'DROP TRIGGER "{name}"')
        self.store.db.execute(f"UPDATE {table} SET {field}=? WHERE {where}", (value, *args))
        for _, sql in triggers:
            self.store.db.execute(sql)

    def destination(self):
        self.count += 1
        return self.base / f"backup-{self.count}.sqlite3"

    def backup(self):
        destination = self.destination()
        return destination, core.backup_store(destination)

    def named(self, report):
        return {(row["scope"], row["conflict_id"]) for row in report.get("damaged_copies", [])}

    def copies(self, database):
        db = sqlite3.connect(database)
        try:
            return {scope: sorted(db.execute(
                f"SELECT conflict_id, observed_bytes, observed_hash, observed_size FROM {scope}_conflicts").fetchall())
                for scope in fixtures.SCOPES}
        finally:
            db.close()

    def mapping(self):
        result = {}
        for (root_id,) in self.store.db.execute("SELECT root_id FROM allowed_roots").fetchall():
            self.count += 1
            result[root_id] = self.base / f"restored-docs-{self.count}"
            result[root_id].mkdir()
        return result


class BackupWarningTests(BackupCase):
    def test_every_kind_of_damaged_copy_is_named_and_source_is_untouched(self):
        damaged = set()
        for scope in fixtures.SCOPES:
            for stage in STAGES:
                for field in ("observed_bytes", "observed_size", "observed_hash"):
                    with self.subTest(scope=scope, stage=stage, field=field):
                        case = self.case(scope, stage)
                        self.damage(case, field)
                        damaged.add((scope, case.id))
                        before = self.copies(self.database)
                        destination, result = self.backup()
                        self.assertTrue(destination.is_file())
                        self.assertEqual(self.named(result["report"]), damaged)
                        self.assertIn(str(len(damaged)), result["warning"][0])
                        self.assertNotIn(case.relative, result["warning"][0], "первая строка идёт агенту: без путей")
                        listed = "\n".join(result["warning"][1:])
                        self.assertIn(case.id, listed)
                        self.assertIn(case.relative, listed)
                        self.assertEqual(self.copies(self.database), before, "рабочая база не меняется")
                        self.assertEqual(self.copies(destination), before, "копии попадают в резерв как есть")
                        self.assertFalse(list(self.base.glob(destination.name + ".incomplete-*")))

    def test_healthy_and_too_large_copies_give_the_previous_answer(self):
        for scope in fixtures.SCOPES:
            self.case(scope, "history")
        big = self.conflict("document", body=b"x" * (documents.MAX_CONFLICT_BYTES + 1))
        self.assertIsNone(big.saved)
        _, result = self.backup()
        self.assertEqual(set(result), {"destination", "report"})
        self.assertEqual(set(result["report"]), {"integrity", "foreign_keys", "schema_version", "inactive"})

    def test_strict_checks_stay_strict_by_default(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.case(scope, "history")
                self.damage(case)
                with self.assertRaisesRegex(storage.StorageError, "copy hash mismatch"):
                    storage.validate_copy(self.database)
                destination = self.destination()
                with self.assertRaisesRegex(storage.StorageError, "copy hash mismatch"):
                    self.store.backup(destination)
                self.assertFalse(destination.exists())
                self.break_row(scope + "_conflicts", "observed_bytes", case.saved, "conflict_id=?", (case.id,))


class OtherDamageTests(BackupCase):
    def other_damage(self, kind):
        case = self.case("document")
        self.case("conspect")
        if kind == "transcript":
            self.break_row("transcript_revisions", "payload_hash", "0" * 64)
        elif kind == "conspect":
            self.break_row("conspect_revisions", "md", b"damaged")
        elif kind == "queue":    # проверяется уже после копий конфликтов
            self.break_row("conspect_outbox", "expected_previous_hashes", '{"md":"x"}')
        elif kind == "document registry":
            self.break_row("file_locations", "canonical_path", str(self.base / "другое"),
                           "relative_path=?", (case.relative,))
        else:
            self.store.db.execute("PRAGMA foreign_keys=OFF")
            self.break_row("conspect_conflicts", "revision", 99)
            self.store.db.execute("PRAGMA foreign_keys=ON")
        return case

    def check(self, kind, with_copy):
        case = self.other_damage(kind)
        if with_copy:
            self.damage(case)
        destination = self.destination()
        with self.assertRaises(storage.StorageError):
            core.backup_store(destination)
        self.assertFalse(destination.exists())

    def test_transcript(self):
        self.check("transcript", False)

    def test_transcript_with_damaged_copy(self):
        self.check("transcript", True)

    def test_conspect(self):
        self.check("conspect", False)

    def test_conspect_with_damaged_copy(self):
        self.check("conspect", True)

    def test_queue(self):
        self.check("queue", False)

    def test_queue_with_damaged_copy(self):
        self.check("queue", True)

    def test_document_registry(self):
        self.check("document registry", False)

    def test_document_registry_with_damaged_copy(self):
        self.check("document registry", True)

    def test_foreign_key(self):
        self.check("foreign key", False)

    def test_foreign_key_with_damaged_copy(self):
        self.check("foreign key", True)


class RestoreWarningTests(BackupCase):
    def test_restore_and_next_backup_name_the_same_copies(self):
        damaged = set()
        for scope in fixtures.SCOPES:
            for stage in STAGES:
                case = self.case(scope, stage)
                self.damage(case)
                damaged.add((scope, case.id))
        backup, _ = self.backup()
        info = core.inspect_backup(backup)
        self.assertEqual(self.named(info), damaged)
        self.assertIn(str(len(damaged)), info["warning"][0])
        destination = self.base / "restored.sqlite3"
        result = core.restore_from_backup(backup, destination, self.mapping(), info["users"][0]["user_id"])
        self.assertEqual(self.named(result["validation"]), damaged)
        self.assertEqual(result["warning"], info["warning"])
        self.assertEqual(self.copies(destination), self.copies(backup))
        restored = storage.open_store(destination)
        try:
            again = restored.backup(self.destination(), allow_damaged_copies=True)
        finally:
            restored.close()
        self.assertEqual(self.named(again), damaged)

    def test_healthy_backup_has_nothing_to_warn_about(self):
        self.case("document", "history")
        backup, _ = self.backup()
        info = core.inspect_backup(backup)
        self.assertEqual((info["damaged_copies"], info["warning"]), ([], []))
        result = core.restore_from_backup(
            backup, self.base / "restored.sqlite3", self.mapping(), info["users"][0]["user_id"])
        self.assertNotIn("warning", result)
        self.assertNotIn("damaged_copies", result["validation"])

    def test_restore_never_replaces_the_only_intact_edit(self):
        """В новую папку заранее скопировали правку, а её копия в базе повреждена."""
        case = self.case("document", "history")
        self.damage(case)
        backup, _ = self.backup()
        mapping = self.mapping()
        edit = mapping[case.rec.root_id] / case.relative
        edit.parent.mkdir(parents=True, exist_ok=True)
        edit.write_bytes(case.saved)
        destination = self.base / "restored.sqlite3"
        user = core.inspect_backup(backup)["users"][0]["user_id"]
        with self.assertRaises(core.CoreError):
            core.restore_from_backup(backup, destination, mapping, user)
        self.assertEqual(hashlib.sha256(edit.read_bytes()).digest(), hashlib.sha256(case.saved).digest())
        self.assertFalse(destination.exists())


class WarningIsShownTests(BackupCase):
    def test_agent_gets_count_and_numbers_without_paths_or_content(self):
        case = self.case("document", "history")
        self.damage(case)
        destination = self.destination()
        with mock.patch.object(confirm, "require", return_value="dialog") as asked:
            raw = asyncio.run(server.backup_state(str(destination)))
        self.assertIn("повреждённые копии правок", "\n".join(asked.call_args.args[2]).casefold())
        answer = json.loads(raw)
        self.assertEqual(answer["report"]["damaged_copies"], {"count": 1, "conflict_ids": [case.id]})
        self.assertEqual(len(answer["warning"]), 1)
        self.assertTrue(destination.is_file())
        for private in (case.relative, case.rec.stem, case.rid):
            self.assertNotIn(private, raw)

    def test_agent_answer_without_damage_is_unchanged(self):
        self.case("document")
        with mock.patch.object(confirm, "require", return_value="dialog"):
            answer = json.loads(asyncio.run(server.backup_state(str(self.destination()))))
        self.assertEqual(set(answer), {"destination", "report"})
        self.assertNotIn("damaged_copies", answer["report"])

    def test_refused_window_creates_nothing(self):
        self.damage(self.case("document"))
        destination = self.destination()
        with mock.patch.object(confirm, "require", side_effect=confirm.NotConfirmed("нет")):
            asyncio.run(server.backup_state(str(destination)))
        self.assertFalse(destination.exists())

    def test_command_warns_after_creation(self):
        case = self.case("conspect", "history")
        self.damage(case)
        shown, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(shown), contextlib.redirect_stderr(errors):
            rt.main(["db-backup", str(self.destination())])
        self.assertIn("с предупреждением", shown.getvalue())
        self.assertIn(case.id, errors.getvalue())
        self.assertIn(case.relative, errors.getvalue())

    def test_wizard_warns_before_question_and_after_restore(self):
        case = self.case("document", "history")
        self.damage(case)
        backup, _ = self.backup()
        args = SimpleNamespace(backup=str(backup), dest=str(self.base / "restored.sqlite3"),
                               root=[f"{key}={path}" for key, path in self.mapping().items()])
        errors = io.StringIO()
        with mock.patch.object(rt, "human_ok", return_value=True) as asked, mock.patch.object(rt, "log"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            self.assertEqual(rt.cmd_db_restore(args), 0)
        self.assertIn(case.id, "\n".join(asked.call_args.args[0]))
        self.assertIn(case.id, errors.getvalue())

    def test_screen_answer_has_the_warning(self):
        case = self.case("document")
        self.damage(case)
        answer = self.ok("POST", "api/backup", {"destination": str(self.destination())})
        self.assertIn(case.id, "\n".join(answer["warning"]))

    @unittest.skipUnless(shutil.which("node"), "Node.js нужен для проверки JavaScript")
    def test_page_shows_warning_as_text(self):
        runner = fixtures.PAGE_RUNNER.replace(
            '"api/history": {revisions: []}};', '"api/history": {revisions: []}, "api/backup": sc.backup};'
        ).replace('  if (sc.compare || sc.confirm) {', '''  if (sc.backup) {
    await ids["backup"].click();
    await settle();
    const report = ids["backup-report"];
    out.html = report.innerHTML;
    out.text = report.textContent;
  }
  if (sc.compare || sc.confirm) {''')
        report = {"integrity": "ok", "schema_version": 4}
        name = f"- «<img src=x>.md» — конфликт {uuid.uuid4()}"
        scenarios = {"clean": {"backup": {"destination": "c:/b", "report": report}},
                     "damaged": {"backup": {"destination": "c:/b", "report": report,
                                            "warning": ["Повреждено копий: 1.", name]}}}
        result = subprocess.run(
            [shutil.which("node"), "-e", runner],
            input=json.dumps({"html": (fixtures.PROJECT / "workflow.html").read_text(encoding="utf-8"),
                              "scenarios": scenarios}, ensure_ascii=False),
            capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = json.loads(result.stdout)
        self.assertNotIn("с предупреждением", seen["clean"]["text"])
        self.assertIn("с предупреждением", seen["damaged"]["text"])
        self.assertIn(name, seen["damaged"]["text"])
        self.assertNotIn("<img", seen["damaged"]["html"], "имя файла не исполняется как разметка")


if __name__ == "__main__":
    unittest.main()
