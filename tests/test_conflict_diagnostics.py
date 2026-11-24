"""Регрессии О3, О5–О7 PR 91: сравнение и проверяемые объяснения."""
import json
import shutil
import sqlite3
import subprocess
import unittest
import uuid
from unittest import mock

import confirm
import core
import documents
import storage
import test_conflict_replace as fixtures


class ComparisonTests(unittest.TestCase):
    def test_two_edits_in_large_document_are_not_the_whole_document(self):
        original = [f"Реплика {index}" for index in range(250000)]
        changed = original.copy()
        changed[4], changed[-5] = "Первая правка", "Последняя правка"
        gone, came = core._diff_lines(changed, original)
        expected = {4, len(original) - 5}
        self.assertTrue(set(gone) == expected, f"Вместо двух изменений отмечено {len(gone)} строк")
        self.assertTrue(set(came) == expected, f"Вместо двух изменений отмечено {len(came)} строк")

    def test_refinement_work_is_bounded_even_with_nested_anchors(self):
        # Каждая вложенная опора отделяет лишь одну строку: без общего бюджета — квадрат.
        original = [str(index // 2) if index % 2 else "повтор" for index in range(8000)]
        changed = ["другой повтор" if index % 2 == 0 else line
                   for index, line in enumerate(original)]
        original_anchors = core._anchors
        searched = []

        def anchors(a, b):
            searched.append(len(a) + len(b))
            return original_anchors(a, b)

        with mock.patch.object(core, "_anchors", side_effect=anchors):
            core._diff_lines(changed, original)
        limit = core.CONFLICT_DIFF_WORK + 5 * (len(original) + len(changed))
        self.assertLessEqual(5 * sum(searched), limit)

    def test_coarse_comparison_is_explicit(self):
        details = {}
        with mock.patch.object(core, "CONFLICT_DIFF_WORK", 0):
            gone, came = core._diff_lines(["а"] * 400, ["б"] * 400, details=details)
        self.assertTrue(details["approximate"])
        self.assertEqual((len(gone), len(came)), (400, 400))

    def test_exact_comparison_is_not_called_coarse(self):
        details = {}
        core._diff_lines(["а", "правка", "б"], ["а", "б"], details=details)
        self.assertFalse(details["approximate"])

    @staticmethod
    def wide_pair():
        # 400 × 400 строк больше CONFLICT_DIFF_CELLS: точное сравнение идёт только через опоры и бюджет уточнения
        original = [f"строка {index}" for index in range(400)]
        changed = original.copy()
        changed[0], changed[-1] = "первая правка", "последняя правка"
        return changed, original

    def test_patched_anchors_reach_the_comparison(self):
        # подмена через core должна действовать и на вызов внутри core_conflicts._diff_lines
        changed, original = self.wide_pair()
        details = {}
        gone, came = core._diff_lines(changed, original, details=details)
        self.assertEqual((set(gone), set(came), details["approximate"]), ({0, 399}, {0, 399}, False))
        with mock.patch.object(core, "_anchors", return_value=[]) as anchors:
            core._diff_lines(changed, original, details=details)
        self.assertTrue(anchors.called, "подмена core._anchors не дошла до _diff_lines")
        self.assertTrue(details["approximate"])

    def test_patched_refinement_budget_reaches_the_comparison(self):
        # без бюджета уточнения та же пара сравнивается грубо; без подмены — точно (предыдущий тест)
        changed, original = self.wide_pair()
        details = {}
        with mock.patch.object(core, "CONFLICT_DIFF_WORK", 0):
            core._diff_lines(changed, original, details=details)
        self.assertTrue(details["approximate"], "подмена core.CONFLICT_DIFF_WORK не дошла до _diff_lines")


class DiagnosticTests(fixtures.ScreenCase):
    def test_unrelated_journal_is_not_named_as_a_blocker(self):
        case = self.conflict("conspect")
        with fixtures.failing_replace(case.path):
            self.replace(self.snapshot(case))
        case.rec.conspect_files["json"].write_bytes(fixtures.LATE_EDIT)
        unrelated = case.path.parent / "000-посторонний.conspect.json"
        unrelated.write_bytes(b"changed")
        journal = unrelated.with_name(unrelated.name + ".conspect.pending")
        journal.write_text(json.dumps({"version": 1, "files": [{
            "name": unrelated.name, "old_hash": "old", "new_hash": "new", "old": None
        }]}), encoding="utf-8")
        reason = self.retry(case)["reason"]
        self.assertIn(case.rec.conspect_files["json"].name, reason)
        self.assertNotIn(unrelated.name, reason)
        self.assertNotIn(journal.name, reason)

    def test_changed_file_does_not_imply_partial_write(self):
        text = core._file_words({"observed_hash": "old"}, "new")
        self.assertNotIn("часть записи", text)
        self.assertIn("не совпадает", text)

    def test_english_refusals_are_not_database_failures_on_old_routes(self):
        for reason in ("stale revisions", "unresolved document conflict", "synthetic refusal"):
            with self.subTest(reason=reason), mock.patch.object(
                    core, "workflow_confirm_verification", side_effect=storage.Conflict(reason)):
                status, _, raw = self.request("POST", "api/verify/confirm", {"snapshot": {}})
            self.assertEqual(status, 400)
            text = json.loads(raw)["error"]
            self.assertNotIn("ошибка базы", text.casefold())
            self.assertRegex(text, fixtures.CYRILLIC)

    def test_busy_is_still_busy_and_database_failure_is_not_a_refusal(self):
        self.assertIn("занята", core.explain_error(storage.Busy("database is busy")))
        self.assertIn("Ошибка базы", core.explain_error(sqlite3.DatabaseError("synthetic failure")))

    def test_preview_names_coarse_counts(self):
        with mock.patch.object(core, "CONFLICT_DIFF_WORK", 0):
            shown = self.preview(self.conspect_texts("б\n" * 400, "а\n" * 400))
        self.assertTrue(shown["comparison_approximate"])
        self.assertTrue(any("оценк" in note for note in shown["file_notes"]))


@unittest.skipUnless(shutil.which("node"), "Node.js нужен для проверки JavaScript")
class ApproximatePageTests(unittest.TestCase):
    def test_estimated_counts_are_visible_and_resolved_copy_stays_neutral(self):
        scenarios = {}
        for name in ("changed", "resolved"):
            scenario = fixtures.page_scenarios()[name]
            scenario["preview"].update(comparison_approximate=True, removed_total=999, added_total=1000)
            scenarios[name] = scenario
        result = subprocess.run(
            [shutil.which("node"), "-e", fixtures.PAGE_RUNNER],
            input=json.dumps({"html": (fixtures.PROJECT / "workflow.html").read_text(encoding="utf-8"),
                              "scenarios": scenarios}, ensure_ascii=False),
            capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = json.loads(result.stdout)
        for name in scenarios:
            self.assertNotIn("error", seen[name])
            self.assertIn("не более 999", seen[name]["panel"])
            self.assertIn("оценка сверху", seen[name]["panel"])
        self.assertNotIn("при замене", seen["resolved"]["panel"].casefold())


@unittest.skipUnless(shutil.which("node"), "Node.js нужен для проверки JavaScript")
class WaitingPageTests(unittest.TestCase):
    def test_confirmation_does_not_promise_what_will_not_happen_now(self):
        scenarios = {}
        for state in ("same", "changed"):
            scenario = fixtures.page_scenarios()["changed"]
            scenario["preview"] = {**scenario["preview"], "file_state": state, "waits_for": ["Другой.json"],
                                   "also_replaces": ["Третий.md"]}
            scenarios[state] = scenario
        result = subprocess.run(
            [shutil.which("node"), "-e", fixtures.PAGE_RUNNER],
            input=json.dumps({"html": (fixtures.PROJECT / "workflow.html").read_text(encoding="utf-8"),
                              "scenarios": scenarios}, ensure_ascii=False),
            capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = json.loads(result.stdout)
        for state in scenarios:
            with self.subTest(state=state):
                question = seen[state]["question"]
                self.assertIn("Сейчас файл", question)
                self.assertIn("не изменится", question)
                self.assertIn("Другой.json", question)
                self.assertIn("Да, записать решение", question)
                for false in ("будет заменён версией программы", "Да, заменить файл", "Да, сохранить новую копию",
                              "сохранит новую копию нынешнего"):
                    self.assertNotIn(false, question)
                self.assertNotIn("подтверждение запишет заново", seen[state]["panel"])


@unittest.skipUnless(shutil.which("node"), "Node.js нужен для проверки JavaScript")
class WaitingJourneyTests(fixtures.ScreenCase):
    """Весь путь ожидания на настоящих ответах программы: ни один текст не обещает невозможного."""

    FALSE = ("когда вы решите", "при любой следующей записи", "когда ожидание кончится", "позже программа заменит",
             "будет заменён версией программы", "программа сохранит новую копию его нынешнего",
             "подтверждение запишет заново")

    def page(self, **scenario):
        result = subprocess.run(
            [shutil.which("node"), "-e", fixtures.PAGE_RUNNER],
            input=json.dumps({"html": (fixtures.PROJECT / "workflow.html").read_text(encoding="utf-8"),
                              "scenarios": {"page": scenario}}, ensure_ascii=False),
            capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = json.loads(result.stdout)["page"]
        self.assertNotIn("error", seen)
        return seen

    def waiting(self, kind):
        if kind == "resolvable":
            return self.both_conspect_files()[0]
        if kind == "big":
            rec = self.recording()
            self.save_conspect(rec)
            self.materialize(rec, "conspect")
            rec.conspect_files["md"].write_bytes(b"my edit\n")
            rec.conspect_files["json"].write_bytes(b"x" * (documents.MAX_CONFLICT_BYTES + 1))
            self.save_conspect(rec)
            self.materialize(rec, "conspect")
            return self.open_case(rec, "conspect", "md")
        case = self.conflict("document")
        row = dict(self.store.db.execute(
            "SELECT * FROM document_conflicts WHERE conflict_id=?", (case.id,)).fetchone())
        row.update(conflict_id=str(uuid.uuid4()), kind="archive_md", relative_path="history/прежняя.md")
        self.store.db.execute(
            f"INSERT INTO document_conflicts ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
            tuple(row.values()))
        return case

    def test_no_stage_promises_what_cannot_happen(self):
        for kind in ("big", "archive", "resolvable"):
            for changed in (False, True):
                with self.subTest(kind=kind, changed=changed):
                    case = self.waiting(kind)
                    if changed:
                        case.path.write_bytes(fixtures.LATE_EDIT)
                    _, asked = fixtures.ConflictMCPTests.mcp_resolve(
                        self, case, side_effect=confirm.NotConfirmed("проба текста"))
                    action, _, lines = asked.call_args.args
                    before, shown = self.item(case), self.preview(case)
                    asks = self.page(items=[before], compare=True, preview=shown)
                    result = self.replace(shown["snapshot"])
                    after = self.item(case)
                    self.assertEqual((after["status"], after["can_retry"]), ("not_written", False))
                    told = self.page(items=[before], confirm=True, preview=shown, result=result)
                    refreshed = self.page(items=[after], compare=True, preview=self.preview(case))
                    stages = {"окно MCP": "\n".join([action, *lines]),
                              "до решения": asks["card"] + asks["panel"],
                              "итог": "\n".join(told["shown"]) + told["note"],
                              "после обновления": refreshed["card"] + refreshed["panel"]}
                    for stage, text in stages.items():
                        for false in self.FALSE:
                            self.assertNotIn(false, text.casefold(), stage)
                    if kind != "resolvable":
                        for stage, text in (("окно MCP", stages["окно MCP"]), ("вопрос", asks["question"]),
                                            ("итог", stages["итог"]), ("карточка", refreshed["card"])):
                            self.assertIn("нельзя", text.replace("отменить его нельзя", "")
                                          .replace("Решение нельзя отменить", "")
                                          .replace("Перенести правку файла в стенограмму здесь нельзя", ""), stage)

    def test_changed_file_over_limit_confirmation_does_not_promise_a_copy(self):
        for scope in fixtures.SCOPES:
            with self.subTest(scope=scope):
                case = self.conflict(scope)
                changed = b"x" * (documents.MAX_CONFLICT_BYTES + 1)
                case.path.write_bytes(changed)
                shown = self.preview(case)
                self.assertEqual(shown["effect"], "recopy")
                page = self.page(items=[self.item(case)], compare=True, preview=shown)
                self.assertIn("Да, записать решение", page["question"])
                self.assertNotIn("Да, сохранить новую копию", page["question"])
                result = self.replace(shown["snapshot"])
                self.assertEqual(result["outcome"], "changed")
                self.assertIsNone(self.open_case(case.rec, scope, "md").saved)
                self.assertEqual(case.path.read_bytes(), changed)


@unittest.skipUnless(shutil.which("node"), "Node.js нужен для проверки JavaScript")
class ChangedPageTests(unittest.TestCase):
    def test_new_copy_is_promised_only_within_the_size_limit(self):
        result = subprocess.run(
            [shutil.which("node"), "-e", fixtures.PAGE_RUNNER],
            input=json.dumps({"html": (fixtures.PROJECT / "workflow.html").read_text(encoding="utf-8"),
                              "scenarios": {"changed": fixtures.page_scenarios()["changed"]}}, ensure_ascii=False),
            capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = json.loads(result.stdout)["changed"]
        self.assertIn("не больше 8 МиБ", seen["question"])
        self.assertEqual(seen["panel"].count("не больше 8 МиБ"), 2, "и строка состояния файла, и вопрос")

    def test_missing_new_conflict_does_not_claim_copy_is_saved(self):
        scenario = fixtures.page_scenarios()["changed"]
        scenario.update(confirm=True, result={"outcome": "changed", "new_conflict_id": "new", "reason": None})
        result = subprocess.run(
            [shutil.which("node"), "-e", fixtures.PAGE_RUNNER],
            input=json.dumps({"html": (fixtures.PROJECT / "workflow.html").read_text(encoding="utf-8"),
                              "scenarios": {"changed": scenario}}, ensure_ascii=False),
            capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = json.loads(result.stdout)["changed"]
        self.assertNotIn("error", seen)
        self.assertIn("не удалось проверить", seen["note"].casefold())
        self.assertNotIn("Программа сохранила его новую копию", seen["note"])


if __name__ == "__main__":
    unittest.main()
