"""tools/check_repo.py --root: скрипт из одной папки проверяет файлы другой и не исполняет код из проверяемой папки.

Зачем: в CI на PR проверка запрещённых слов должна идти скриптом из main поверх файлов PR (см. docs/ci-automerge.md).
Данные синтетические: слово и имя пользователя придуманы, список слов и папка данных — во временной папке.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import check_repo  # noqa: E402

WORD = "Тестослов"            # придуманное запрещённое слово
USER = "Zebraname"            # придуманное имя пользователя в пути


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data = self.tmp / "data"
        self.data.mkdir()
        self.words = self.tmp / "words.txt"
        self.words.write_text(WORD + "\n", encoding="utf-8")
        self.env = {"RUS_TRANSCRIBER_DATA": str(self.data), "RUS_FORBIDDEN_WORDS": str(self.words)}
        patcher = patch.dict(os.environ, self.env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tree(self, name, files):
        root = self.tmp / name
        for rel, text in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
        return root

    def run_script(self, *args):
        env = dict(os.environ, **self.env, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        return subprocess.run([sys.executable, "-B", str(TOOLS / "check_repo.py"), *args], env=env,
                              capture_output=True, text=True, encoding="utf-8", timeout=120)


class RootArgument(Base):
    def test_check_reads_the_given_root_not_the_script_folder(self):
        dirty = self.tree("dirty", {"a.py": f"# здесь {WORD} упомянут\n"})
        clean = self.tree("clean", {"a.py": "x = 1\n"})
        problems, words, _ = check_repo.check(root=dirty)
        self.assertEqual([p[0] for p in problems], ["a.py:1"])
        self.assertIn(WORD, words)
        problems, _, _ = check_repo.check(root=clean)
        self.assertEqual(problems, [])

    def test_without_root_the_module_root_is_used_at_call_time(self):
        """Обратная совместимость: тесты и selfcheck подменяют check_repo.ROOT и зовут check() без root."""
        dirty = self.tree("dirty", {"a.py": f"# {WORD}\n"})
        with patch.object(check_repo, "ROOT", dirty):
            problems, _, _ = check_repo.check()
            files, _ = check_repo.tracked(False)
        self.assertEqual([p[0] for p in problems], ["a.py:1"])
        self.assertEqual(files, ["a.py"])

    def test_git_checkout_is_listed_by_git_in_the_given_root(self):
        if not shutil.which("git"):
            self.skipTest("нужен git")
        # путь с именем пользователя собирается из частей: целиком в исходнике его нельзя писать, иначе
        # test_repo_clean.test_no_personal_data_tracked примет эту строку за настоящий личный путь
        user_path = "\\".join(["C:", "Users", USER, "x"])
        root = self.tree("repo", {"a.py": f"p = {user_path!r}\n", "ignored.txt": "x\n"})
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=env)
        subprocess.run(["git", "add", "a.py"], cwd=root, check=True, env=env)   # ignored.txt не добавлен
        problems, _, is_git = check_repo.check(root=root, hide_words=True)
        self.assertTrue(is_git)
        self.assertEqual([p[0] for p in problems], ["a.py:1"])
        self.assertNotIn(USER, problems[0][1])


class CommandLine(Base):
    def test_exit_codes_and_hidden_words(self):
        dirty = self.tree("dirty", {"a.py": f"# {WORD}\n"})
        clean = self.tree("clean", {"a.py": "x = 1\n"})
        r = self.run_script("--root", str(dirty), "--hide-words")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("a.py:1", r.stdout)
        self.assertNotIn(WORD, r.stdout + r.stderr)
        r = self.run_script(f"--root={clean}")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_bad_root_is_not_a_success(self):
        for args in (["--root", str(self.tmp / "no-such-folder")], ["--root"], ["--root="], ["--root", "--hide-words"]):
            with self.subTest(args=args):
                r = self.run_script(*args)
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIn("Проверка не выполнена", r.stderr)

    def test_code_of_the_checked_folder_is_not_executed(self):
        """Скрипт «из main» читает файлы PR как данные: tools/check_repo.py внутри проверяемой папки не запускается."""
        marker = self.tmp / "executed.marker"
        planted = f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
        root = self.tree("pr", {"tools/check_repo.py": planted, "conftest.py": planted, "a.py": "x = 1\n"})
        r = self.run_script("--root", str(root))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
