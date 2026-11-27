"""Страж переносимости (М-1, решение А-9): новый код не должен тихо завязываться на Windows.

Ищет в .py файлах репозитория приёмы, привязанные к Windows (windll, msvcrt, winreg, creationflags,
taskkill, powershell, путь с папкой Scripts), и сверяет файлы с явным списком ALLOWED. Новый файл
с таким приёмом роняет тест: вынесите зависимое от системы в общее место или осознанно добавьте файл
в список с причиной в PR (исследование переноса, раздел 4).

Список ALLOWED — снимок на сегодня. Он должен только сокращаться: когда линия Linux соберёт всё
зависящее от Windows в один модуль (osdeps.py), в списке останется он да служебные файлы.
Файл из списка, где приёмов уже нет, тоже роняет тест — чтобы список не оставался лишним.

Только чтение исходников (ast, без импорта paths, core, server). Комментарии и строки документации
не считаются; считаются имена, атрибуты, импорты, аргументы вызовов и обычные строки.
"""
import ast
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()

# Приёмы, привязанные к Windows. Scripts — только как часть пути (.venv\Scripts, ".venv", "Scripts").
PATTERN = re.compile(r"windll|msvcrt|winreg|creationflags|taskkill|powershell|(?:^|[\\/])Scripts(?:$|[\\/])",
                     re.IGNORECASE)
WORDS = re.compile(r"windll|msvcrt|winreg|creationflags|taskkill|powershell|Scripts", re.IGNORECASE)

# Файлы, где такое есть сегодня (26.11.2026), и почему. Сокращает линия Linux/macOS.
ALLOWED = {
    # рабочий код: окно без консоли, процессы, блокировки — кандидаты на osdeps.py
    "buildinfo.py": "CREATE_NO_WINDOW у запуска git",
    "confirm.py": "окно подтверждения MessageBoxW, creationflags",
    "core.py": "kernel32, taskkill, creationflags при запуске задач",
    "documents.py": "msvcrt.locking",
    "jobslots.py": "msvcrt.locking",
    "llm_provider.py": "iphlpapi/kernel32, taskkill, creationflags",
    "review_pack.py": "creationflags (с условием os.name)",
    "runinfo.py": "winreg, kernel32, PowerShell для сведений о железе",
    "selfcheck.py": "taskkill, creationflags (с условием os.name)",
    "storage.py": "GetDriveTypeW: проверка локального диска",
    "voice_io.py": "winmm, creationflags (только Windows)",
    "workflow_page.py": "creationflags без условия (на POSIX ошибка)",
    # тексты с путём .venv\Scripts
    "bench.py": "подсказка с путём .venv\\Scripts",
    "download_hint.py": "команды PowerShell и путь .venv\\Scripts",
    # служебное и лаборатория
    "lab/lab_sandbox.py": "CREATE_NO_WINDOW",
    "tools/agent_workspaces.py": "msvcrt, kernel32, PowerShell",
    # тесты Windows-части
    "tests/run_parallel.py": "taskkill, creationflags (с условием os.name)",
    "tests/test_conflict_replace.py": "msvcrt.locking в проверке блокировки",
    "tests/test_download_hint.py": "путь .venv\\Scripts в ожидаемом тексте",
    "tests/test_install_uninstall.py": "поиск PowerShell для install.ps1",
    "tests/test_install_update.py": "поиск PowerShell для install.ps1",
    "tests/test_text_hygiene.py": "сообщение о файлах PowerShell без BOM",
    "tests/test_p6_acceptance.py": "поиск powershell.exe",
}

SKIP_DIRS = {"__pycache__", "node_modules", "build", "dist"}


def _docstrings(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                out.add(id(body[0].value))
    return out


def hits(source):
    """Приёмы в исходнике без комментариев и строк документации: список (строка, текст)."""
    if not WORDS.search(source):        # быстрый отсев: разбирать дерево нужно не везде
        return []
    tree = ast.parse(source)
    docs = _docstrings(tree)
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.keyword) and node.arg:
            names = [node.arg]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [a.name for a in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            names = [node.value]
        for name in names:
            if PATTERN.search(name):
                found.append((getattr(node, "lineno", 0), name.strip()[:60]))
    return found


def python_files():
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]
        for name in files:
            path = Path(base) / name
            if name.endswith(".py") and path.resolve() != SELF:
                yield path


class PortabilityGuard(unittest.TestCase):
    def test_detector_sees_code_but_not_comments(self):
        sep = chr(92)
        src = ('"""doc: powershell windll"""\n# taskkill в комментарии\n'
               "import msvcrt\nx = ctypes.windll.kernel32\nsubprocess.run(a, creationflags=1)\n"
               f's = ".venv{sep}Scripts{sep}python.exe"\n')
        self.assertEqual(sorted(n for _line, n in hits(src)),
                         sorted(["msvcrt", "windll", "creationflags", f".venv{sep}Scripts{sep}python.exe"]))
        self.assertEqual(hits('"""powershell"""\n# windll\nx = 1\ny = "Scripts of the day"\n'), [])

    def test_windows_specific_code_only_in_allowed_files(self):
        unexpected, stale = [], []
        for path in python_files():
            rel = path.relative_to(ROOT).as_posix()
            try:
                found = hits(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            if found and rel not in ALLOWED:
                unexpected.append(f"{rel}:{found[0][0]}: {found[0][1]}")
            elif not found and rel in ALLOWED:
                stale.append(rel)
        self.assertEqual(unexpected, [], "новый код зависит от Windows вне списка ALLOWED: вынесите зависимое "
                                         "от системы в общее место или добавьте файл в список с причиной")
        self.assertEqual(stale, [], "в этих файлах приёмов, зависящих от Windows, больше нет: уберите их из ALLOWED")


if __name__ == "__main__":
    unittest.main()
