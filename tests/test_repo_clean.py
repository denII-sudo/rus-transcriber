"""Репозиторий чист от личных данных (имена из локальных данных, пути пользователя, профили голосов).
Запуск: python -m unittest discover -s tests. Слова берутся из папки данных; нет данных — проверка слов пропускается."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import check_repo  # noqa: E402


class T(unittest.TestCase):
    def test_no_personal_data_tracked(self):
        problems, words, _ = check_repo.check()
        self.assertEqual(problems, [], "\n".join(f"{w}: {why}" for w, why in problems))

    def test_user_path_regex_allows_placeholders(self):
        rx = check_repo.USER_PATH
        self.assertEqual(rx.search(r"C:\Users\Иван\x").group(1), "Иван")
        self.assertIsNone(rx.search(r"C:\Users\<вы>\x"))
        self.assertIsNone(rx.search(r"C:\Users\%USERNAME%\x"))


if __name__ == "__main__":
    unittest.main()
