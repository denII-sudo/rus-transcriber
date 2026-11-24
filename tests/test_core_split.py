"""Разрезка `core.py` на модули: имена, порядок загрузки, свободные имена (переносы: «голоса», «конфликты»).

`core.py` остаётся прослойкой: все прежние имена `core.X` работают, но каждое определено в одном месте — в новом
модуле (`core_voices.X.__module__ == "core_voices"`, `core.X is core_voices.X`). Всё, что новый модуль берёт из
`core.py` вне своего раздела, он берёт как `core.X` в момент вызова: так подмены `patch.object(core, "job_json")`
в тестах продолжают действовать. Каждый следующий перенос добавляет строку в SPLIT.
"""
import ast
import builtins
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import core

REPO = Path(__file__).resolve().parent.parent

# модуль -> имена, которые `core.py` переотдаёт под прежними названиями
SPLIT = {
    "core_voices": (
        "REFRESH_MEANINGS", "_speaker_voice_summary", "_voice_hash", "_voice_job_map", "_voice_vocab_context",
        "backup_voice_profile", "backup_voice_profiles", "preview_voice_learning", "prepare_voice_action",
        "revoke_voice_source", "set_voice_auto_learn", "set_voice_quality_policy", "start_voice_update",
        "voice_action_lines", "voice_profiles", "voice_quality", "voice_refresh_plan",
    ),
    "core_conflicts": (
        "CONFLICT_DIFF_CELLS", "CONFLICT_DIFF_WORK", "CONFLICT_PREVIEW_CHARS", "CONFLICT_SCOPES",
        "CONFLICT_SHOWN_CHARS", "CONFLICT_SHOWN_LINES", "_CONFLICT_COLUMNS", "_CONFLICT_SQL", "_KNOWN_CONFLICTS",
        "_WIDE_SPACES", "_ZERO_WIDTH", "_also_replaced", "_anchors", "_answer_confirmation_hash", "_conflict_facts",
        "_conflict_item", "_conflict_row", "_conflict_rows", "_copy_name", "_copy_problem",
        "_damaged_copies_warning", "_decode_copy", "_diff_lines", "_effect", "_eol_names", "_failure_reason",
        "_file_at", "_file_notes", "_file_outcome", "_file_words", "_invisible_note", "_is_journal_stop",
        "_journal_blockers", "_journal_reason", "_lines", "_listed_program_version", "_missing_text", "_next_step",
        "_not_written_warning", "_observed_bytes", "_present", "_program_version", "_queue_unfinished", "_real_case",
        "_recording_reference", "_refusal", "_replace_file", "_resolve_for_agent", "_same_object", "_set_stops",
        "_show_text", "_shown_file", "_stops_words", "_unresolvable_wait", "_view", "_waits_for",
        "_write_and_report", "backup_store", "confirm_review_answers", "conflict_copy", "explain_error",
        "find_recording_results", "inspect_backup", "list_all_conflicts", "list_conspect_conflicts",
        "list_document_conflicts", "prepare_answer_confirmation", "preview_conflict", "read_recording_result",
        "replace_conflict_file", "replacement_effect", "resolve_conspect_conflict", "resolve_document_conflict",
        "restore_from_backup", "retry_conflict_file", "say_error", "sync_recording_documents",
    ),
}


def bound_names(tree):
    """Все имена, которые модуль где-либо связывает: импорты, def/class, присваивания, параметры, циклы и т. п."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


class CoreSplitTests(unittest.TestCase):
    def isolated_env(self, temp):
        return dict(os.environ, RUS_TRANSCRIBER_DATA=str(Path(temp) / "data"), TEMP=temp, TMP=temp,
                    PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")

    def run_python(self, code):
        with tempfile.TemporaryDirectory(prefix="core-split-") as temp:
            return subprocess.run([sys.executable, "-B", "-c", code], cwd=str(REPO), env=self.isolated_env(temp),
                                  capture_output=True, text=True, encoding="utf-8", timeout=120)

    def test_every_name_is_reachable_from_core_and_defined_once(self):
        for module_name, names in SPLIT.items():
            module = __import__(module_name)
            for name in names:
                with self.subTest(module=module_name, name=name):
                    self.assertTrue(hasattr(module, name), "нет в новом модуле")
                    self.assertIs(getattr(core, name), getattr(module, name), "core.X не тот же объект")
                    owner = getattr(getattr(module, name), "__module__", module_name)
                    self.assertEqual(owner, module_name, "остаточная копия или чужое определение")

    def test_core_source_has_no_definition_of_a_moved_name(self):
        tree = ast.parse((REPO / "core.py").read_text(encoding="utf-8"))
        defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        defined |= {target.id for node in tree.body if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}
        for module_name, names in SPLIT.items():
            for name in names:
                with self.subTest(module=module_name, name=name):
                    self.assertNotIn(name, defined)

    def test_either_module_can_be_loaded_first_in_a_clean_process(self):
        for module_name, names in SPLIT.items():
            check = f"assert all(getattr(core, n) is getattr(m, n) for n in {list(names)!r}); print('ok')"
            for label, code in (
                    ("первым core", f"import core; import {module_name} as m; {check}"),
                    ("первым " + module_name, f"import {module_name} as m; import core; {check}"),
                    ("только core", "import core; print('ok')"),
                    ("только " + module_name, f"import {module_name}; print('ok')")):
                with self.subTest(module=module_name, order=label):
                    result = self.run_python(code)
                    self.assertEqual(result.returncode, 0, result.stderr[-800:])
                    self.assertIn("ok", result.stdout)

    def test_new_module_takes_nothing_from_core_by_copy_and_has_no_free_names(self):
        for module_name in SPLIT:
            path = REPO / f"{module_name}.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            bound = bound_names(tree) | set(dir(builtins)) | {"__file__", "__name__"}
            with self.subTest(module=module_name):
                # имя, определённое в core.py вне своего раздела, берётся как core.X, а не копируется
                copied = [node.lineno for node in ast.walk(tree)
                          if isinstance(node, ast.ImportFrom) and node.module == "core"]
                self.assertEqual(copied, [], f"from core import … (строки {copied}): подмена в тестах не подействует")
                free = sorted({node.id for node in ast.walk(tree)
                               if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                               and node.id not in bound})
                self.assertEqual(free, [], "имена не определены в модуле: берите их как core.X")
                # core.X должен существовать в core
                missing = sorted({node.attr for node in ast.walk(tree)
                                  if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                                  and node.value.id == "core" and isinstance(node.ctx, ast.Load)
                                  and not hasattr(core, node.attr)})
                self.assertEqual(missing, [], "у core нет таких имён")

    def test_new_module_defines_no_error_class_of_its_own(self):
        # ошибки раздела — это core.CoreError: свой класс не ловился бы assertRaises(core.CoreError)
        for module_name in SPLIT:
            tree = ast.parse((REPO / f"{module_name}.py").read_text(encoding="utf-8"))
            with self.subTest(module=module_name):
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                self.assertNotIn("CoreError", classes)

    def test_import_of_core_stays_last_in_the_new_module(self):
        # `import core` наверху файла ломает загрузку, когда первым загружен новый модуль
        for module_name in SPLIT:
            tree = ast.parse((REPO / f"{module_name}.py").read_text(encoding="utf-8"))
            with self.subTest(module=module_name):
                top = [i for i, node in enumerate(tree.body)
                       if isinstance(node, ast.Import) and any(a.name == "core" for a in node.names)]
                self.assertEqual(top, [len(tree.body) - 1], "import core — последний оператор модуля")


if __name__ == "__main__":
    unittest.main()
