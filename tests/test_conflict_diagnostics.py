"""Регрессии О3, О5–О7 PR 91: сравнение и проверяемые объяснения."""
import json
import shutil
import sqlite3
import subprocess
import unittest
from unittest import mock

import core
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


if __name__ == "__main__":
    unittest.main()
