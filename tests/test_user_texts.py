"""Тексты для пользователя и его ИИ-агента без внутренних кодов пакетов разработки (П1–П7, P1–P7).

Решение владельца 25.11.2026: коды пакетов остаются в документах разработки, но из интерфейса и описаний
функций для пользователя убраны — пользователи не программисты. Проверяются инструкции и описания
инструментов MCP (server.py), справка rt.py, страницы и пользовательский раздел README «Как пользоваться».
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = re.compile(r"(?<![\w])[ПP][1-7](?![\w])")
USER_FILES = ("server.py", "rt.py", "workflow.html", "cards.html")


def user_part_of_readme(text):
    start = text.index("\n## Как пользоваться")
    end = text.index("\n## ", start + 1)
    return text[start:end]


class UserTextsTests(unittest.TestCase):
    def test_no_package_codes_in_user_files(self):
        for name in USER_FILES:
            path = ROOT / name
            if not path.exists():
                continue
            with self.subTest(file=name):
                found = [m.group(0) for m in CODE.finditer(path.read_text(encoding="utf-8"))]
                self.assertEqual(found, [], f"{name}: коды пакетов в тексте для пользователя")

    def test_no_package_codes_in_readme_usage(self):
        path = ROOT / "README.md"
        if not path.exists():
            self.skipTest("README.md не поставлен")
        found = CODE.findall(user_part_of_readme(path.read_text(encoding="utf-8")))
        self.assertEqual(found, [], "README «Как пользоваться»: коды пакетов")

    def test_pattern_catches_codes_and_spares_other_words(self):
        self.assertEqual(CODE.findall("экран П5; П3/P4; (П7)"), ["П5", "П3", "P4", "П7"])
        self.assertEqual(CODE.findall("MP4, П10, ПП5, P4x, mp3"), [])


if __name__ == "__main__":
    unittest.main()
