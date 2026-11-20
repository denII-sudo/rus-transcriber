"""Оформление текстовых файлов репозитория по правилам AGENTS.md: UTF-8, BOM только у файлов PowerShell,
переносы строк LF, относительные ссылки в .md ведут на файлы репозитория.
Запуск: python -m unittest discover -s tests. В копии без git (установленная программа) проверка репозитория
пропускается: там нет полного набора документов, а файлы разработки прежних версий могут остаться."""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import check_repo  # noqa: E402

BOM = b"\xef\xbb\xbf"
# [текст](цель) и ![картинка](цель); цель без пробелов, можно в <угловых скобках> и с "заголовком"
LINK = re.compile(r"\[[^\]]*\]\(<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^(```|~~~).*?^\1", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def encoding_problems(root, files):
    """Нарушения кодировки и переносов строк в перечисленных файлах (двоичные пропускаются)."""
    problems = []
    for rel in files:
        p = Path(root) / rel
        if not p.is_file():
            continue
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{rel}: текст не в UTF-8")
            continue
        ps1 = rel.lower().endswith(".ps1")
        if ps1 and not data.startswith(BOM):
            problems.append(f"{rel}: у файла PowerShell нет BOM (Windows PowerShell 5.1 прочтёт кириллицу как ANSI)")
        if not ps1 and data.startswith(BOM):
            problems.append(f"{rel}: лишний BOM (BOM нужен только файлам .ps1)")
        if b"\r" in data:
            problems.append(f"{rel}: есть символ CR — нужны переносы строк LF")
    return problems


def link_problems(root, files):
    """Относительные ссылки в .md, цель которых не входит в перечисленные файлы или их папки."""
    known = set(files)
    for rel in files:
        parts = rel.split("/")
        known.update("/".join(parts[:i]) for i in range(1, len(parts)))
    problems = []
    for rel in files:
        p = Path(root) / rel
        if not rel.lower().endswith(".md") or not p.is_file():
            continue
        text = p.read_text(encoding="utf-8-sig")
        # блоки кода заменяем пустыми строками, чтобы номера строк остались верными
        text = FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
        for ln, line in enumerate(text.splitlines(), 1):
            for m in LINK.finditer(INLINE_CODE.sub("", line)):
                target = m.group(1)
                if target.startswith("#") or SCHEME.match(target):
                    continue
                path = unquote(target.split("#", 1)[0].split("?", 1)[0])
                if not path:
                    continue
                base = "" if path.startswith("/") else os.path.dirname(rel)
                norm = os.path.normpath(os.path.join(base, path.lstrip("/"))).replace("\\", "/")
                if norm not in known:
                    problems.append(f"{rel}:{ln}: ссылка «{target}» ведёт не на файл репозитория")
    return problems


class RepositoryText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, is_git = check_repo.tracked(False)
        if not is_git:
            raise unittest.SkipTest("копия без git: оформление файлов проверяется в рабочей копии и в CI")

    def test_utf8_bom_and_line_endings(self):
        problems = encoding_problems(check_repo.ROOT, self.files)
        self.assertEqual(problems, [], "\n".join(problems))

    def test_relative_links_in_markdown(self):
        problems = link_problems(check_repo.ROOT, self.files)
        self.assertEqual(problems, [], "\n".join(problems))


class CheckerFindsProblems(unittest.TestCase):
    """Сами проверки замечают нарушения (синтетические файлы во временной папке)."""

    def test_encoding_problems(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "ok.md").write_bytes("текст\n".encode("utf-8"))
            (root / "ok.ps1").write_bytes(BOM + "Write-Host 'привет'\n".encode("utf-8"))
            (root / "crlf.md").write_bytes(b"line\r\n")
            (root / "bom.py").write_bytes(BOM + b"x = 1\n")
            (root / "nobom.ps1").write_bytes(b"Write-Host 1\n")
            (root / "cp1251.txt").write_bytes("текст".encode("cp1251"))
            (root / "image.bin").write_bytes(b"\x89PNG\0\r\n")
            files = ["ok.md", "ok.ps1", "crlf.md", "bom.py", "nobom.ps1", "cp1251.txt", "image.bin"]
            problems = encoding_problems(root, files)
        found = sorted(p.split(":", 1)[0] for p in problems)
        self.assertEqual(found, ["bom.py", "cp1251.txt", "crlf.md", "nobom.ps1"])

    def test_link_problems(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "docs" / "sub").mkdir(parents=True)
            (root / "README.md").write_text(
                "[a](docs/guide.md) [b](docs/guide.md#раздел) [c](https://example.com/x.md)\n"
                "[d](#якорь) [e](docs/missing.md) `[f](docs/in-code.md)` [g](docs/sub)\n"
                "```\n[h](docs/in-fence.md)\n```\n"
                "[i](<docs/guide.md> \"заголовок\") [j](/docs/guide.md) [k](mailto:x@example.com)\n",
                encoding="utf-8")
            (root / "docs" / "guide.md").write_text("[back](../README.md) [up](../nope.md)\n", encoding="utf-8")
            (root / "docs" / "sub" / "x.md").write_text("x\n", encoding="utf-8")
            problems = link_problems(root, ["README.md", "docs/guide.md", "docs/sub/x.md"])
        self.assertEqual(len(problems), 2, "\n".join(problems))
        self.assertTrue(problems[0].startswith("README.md:2:") and "docs/missing.md" in problems[0])
        self.assertTrue(problems[1].startswith("docs/guide.md:1:") and "../nope.md" in problems[1])


if __name__ == "__main__":
    unittest.main()
