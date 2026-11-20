"""Репозиторий чист от личных данных (имена из локальных данных, пути пользователя, профили голосов).
Запуск: python -m unittest discover -s tests. Слова берутся из папки данных; нет данных — проверка слов пропускается."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import check_repo  # noqa: E402


class T(unittest.TestCase):
    def test_no_personal_data_tracked(self):
        problems, words, _ = check_repo.check()
        self.assertEqual(problems, [], "\n".join(f"{w}: {why}" for w, why in problems))

    def test_fallback_without_git_skips_environment_and_data_dirs(self):
        """Установленная копия — папка без git и без .gitignore: .venv и cache не файлы репозитория."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel in (".venv/pyvenv.cfg", "cache/pip/blob", "groups/demo/vocab.json", "jobs/j.json",
                        "__pycache__/x.pyc", "server.py"):
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x", encoding="utf-8")
            old = check_repo.ROOT
            check_repo.ROOT = root
            try:
                files, is_git = check_repo.tracked(False)
            finally:
                check_repo.ROOT = old
        self.assertFalse(is_git)
        self.assertEqual(files, ["server.py"])

    def test_fallback_without_git_skips_model_cache(self):
        """Кэш моделей в установленной копии лежит в hf-cache: это не файлы репозитория."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model = root / "hf-cache" / "hub" / "models--org--model" / "config.json"
            model.parent.mkdir(parents=True)
            model.write_text("x", encoding="utf-8")
            (root / "server.py").write_text("x", encoding="utf-8")
            old = check_repo.ROOT
            check_repo.ROOT = root
            try:
                files, is_git = check_repo.tracked(False)
            finally:
                check_repo.ROOT = old
        self.assertFalse(is_git)
        self.assertEqual(files, ["server.py"])

    def test_default_group_name_is_not_a_forbidden_word(self):
        """Общая группа «default» есть в любой установке, личных данных в этом имени нет."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "groups" / "default").mkdir(parents=True)
            (root / "groups" / "Сосны").mkdir(parents=True)
            (root / "groups" / "default" / "voices.json").write_text(
                '{"participants": [{"name": "Иван Примеров"}]}', encoding="utf-8")
            old_env = os.environ.get("RUS_TRANSCRIBER_DATA")
            os.environ["RUS_TRANSCRIBER_DATA"] = str(root)
            try:
                words = check_repo.forbidden_words()
            finally:
                if old_env is None:
                    os.environ.pop("RUS_TRANSCRIBER_DATA", None)
                else:
                    os.environ["RUS_TRANSCRIBER_DATA"] = old_env
        self.assertNotIn("default", words)
        self.assertIn("Сосны", words)
        self.assertIn("Иван", words)

    def test_user_path_regex_allows_placeholders(self):
        rx = check_repo.USER_PATH
        self.assertEqual(rx.search(r"C:\Users\Иван\x").group(1), "Иван")
        self.assertIsNone(rx.search(r"C:\Users\<вы>\x"))
        self.assertIsNone(rx.search(r"C:\Users\%USERNAME%\x"))


if __name__ == "__main__":
    unittest.main()
