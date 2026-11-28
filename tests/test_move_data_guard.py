"""tools/move_data.py (БД-1б): отказ до любых изменений, если в источнике или назначении есть база SQLite.

Данные синтетические, всё во временной папке; настоящая папка данных и окружение пользователя не используются:
APPDATA, LOCALAPPDATA и USERPROFILE подменены на пустую временную папку, чтобы стандартные источники не нашлись.
"""
import contextlib
import io
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import move_data  # noqa: E402


class MoveDataGuardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.home = self.tmp / "home"
        self.home.mkdir()
        env = {"APPDATA": str(self.home / "r"), "LOCALAPPDATA": str(self.home / "l"),
               "USERPROFILE": str(self.home)}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.src = self.tmp / "src"
        self.dest = self.tmp / "dest"
        self.src.mkdir()

    def run_tool(self, *extra):
        argv = ["move_data.py", "--src", str(self.src), "--dest", str(self.dest), *extra]
        out = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            try:
                move_data.main()
                return None, out.getvalue()
            except SystemExit as ex:
                return ex.code, out.getvalue()

    def snapshot(self):
        return sorted((str(p.relative_to(self.tmp)), p.stat().st_size if p.is_file() else -1)
                      for p in self.tmp.rglob("*") if "home" not in p.relative_to(self.tmp).parts)

    def test_plain_folder_without_database_is_still_moved(self):
        (self.src / "note.txt").write_text("синтетика", encoding="utf-8")
        code, _out = self.run_tool("--keep-source")
        self.assertIsNone(code)
        self.assertEqual((self.dest / "note.txt").read_text(encoding="utf-8"), "синтетика")

    def test_source_with_state_database_set_is_refused_before_any_change(self):
        for number, name in enumerate(("state.sqlite3", "state.sqlite3-wal", "state.sqlite3-shm", "state.sqlite3-journal",
                     "state.sqlite3.incomplete-" + "a" * 32, "state.sqlite3.before-restore-20260101-000000",
                     "STATE.SQLITE3-WAL")):
            with self.subTest(name=name):
                folder = self.tmp / f"src-{number}"
                folder.mkdir()
                (folder / "note.txt").write_text("x", encoding="utf-8")
                (folder / name).write_bytes(b"not really a database")
                self.src = folder
                before = self.snapshot()
                for extra in ((), ("--dry-run",)):
                    code, _out = self.run_tool(*extra)
                    self.assertIsInstance(code, str, extra)
                    self.assertIn("Отказ", code)
                    self.assertIn(name, code)
                self.assertEqual(self.snapshot(), before)           # ничего не создано и не переименовано
                self.assertFalse(self.dest.exists())

    def test_database_in_a_subfolder_and_unusual_name_is_found_by_header(self):
        sub = self.src / "db-backups"
        sub.mkdir()
        con = sqlite3.connect(sub / "copy.bin")                 # имя обычное: признак — заголовок файла
        con.execute("CREATE TABLE t(x)")
        con.commit()
        con.close()
        code, _out = self.run_tool()
        self.assertIsInstance(code, str)
        self.assertIn("заголовок SQLite", code)
        self.assertFalse(self.dest.exists())

    def test_backup_names_without_state_prefix_are_refused(self):
        for name in ("state-before-hypotheses-20260101.sqlite3", "bench_runs.sqlite3", "x.sqlite3-wal"):
            with self.subTest(name=name):
                folder = self.tmp / ("b-" + name)
                folder.mkdir()
                (folder / name).write_bytes(b"")
                self.src = folder
                code, _out = self.run_tool()
                self.assertIsInstance(code, str)
                self.assertIn(name, code)

    def test_destination_with_database_is_refused(self):
        (self.src / "note.txt").write_text("x", encoding="utf-8")
        self.dest.mkdir()
        (self.dest / "state.sqlite3").write_bytes(b"live")
        before = self.snapshot()
        code, _out = self.run_tool()
        self.assertIsInstance(code, str)
        self.assertIn("назначение", code)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual((self.dest / "state.sqlite3").read_bytes(), b"live")
        self.assertFalse((self.dest / "migration.log").exists())

    def test_refusal_does_not_delete_or_rename_the_database(self):
        db = self.src / "state.sqlite3"
        db.write_bytes(b"live")
        self.run_tool()
        self.assertEqual(db.read_bytes(), b"live")
        self.assertEqual([p.name for p in self.tmp.glob("src*")], ["src"])         # не переименована в .moved-*


POWERSHELL = shutil.which("powershell")        # Windows PowerShell 5.1 (powershell.exe); pwsh 7 — другой файл


@unittest.skipUnless(POWERSHELL and os.name == "nt", "для проверки move_data.ps1 нужен Windows PowerShell")
class MoveDataPowerShellGuardTests(unittest.TestCase):
    """tools/move_data.ps1 — тот же отказ, что у move_data.py: до любых изменений, в том числе с -DryRun."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        home = self.tmp / "home"
        (home / "l").mkdir(parents=True)
        (home / "r").mkdir()
        self.env = dict(os.environ, APPDATA=str(home / "r"), LOCALAPPDATA=str(home / "l"), USERPROFILE=str(home))
        # Окружение CI запускает тесты из PowerShell 7: его PSModulePath ломает автозагрузку модулей у Windows
        # PowerShell 5.1 (Get-FileHash «не найден»). Своё окружение 5.1 строит сам.
        for key in [key for key in self.env if key.lower() == "psmodulepath"]:      # на Windows ключи — ЗАГЛАВНЫМИ
            del self.env[key]
        self.src = self.tmp / "src"
        self.dest = self.tmp / "dest"
        self.src.mkdir()
        self.script = TOOLS / "move_data.ps1"

    def run_script(self, *extra):
        command = [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                   str(self.script), "-Src", str(self.src), "-Dest", str(self.dest), *extra]
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env=self.env, cwd=str(self.tmp), timeout=120)

    def tree(self):
        return sorted(str(p.relative_to(self.tmp)) for p in self.tmp.rglob("*")
                      if "home" not in p.relative_to(self.tmp).parts)

    def test_script_keeps_bom_and_lf(self):
        raw = self.script.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r", raw)

    def test_plain_folder_is_still_moved(self):
        (self.src / "note.txt").write_text("синтетика", encoding="utf-8")
        result = self.run_script("-KeepSource")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.dest / "note.txt").read_text(encoding="utf-8"), "синтетика")

    def test_dry_run_on_plain_folder_creates_nothing(self):
        (self.src / "note.txt").write_text("x", encoding="utf-8")
        before = self.tree()
        result = self.run_script("-DryRun")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.tree(), before)

    def test_source_with_database_set_is_refused_before_any_change(self):
        for number, name in enumerate(("state.sqlite3", "state.sqlite3-wal", "STATE.SQLITE3-SHM",
                                       "state.sqlite3.before-restore-20260101-000000",
                                       "state-before-hypotheses-20260101.sqlite3")):
            with self.subTest(name=name):
                folder = self.tmp / f"src-{number}"
                folder.mkdir()
                (folder / name).write_bytes(b"not really a database")
                self.src = folder
                before = self.tree()
                for extra in ((), ("-DryRun",)):
                    result = self.run_script(*extra)
                    self.assertNotEqual(result.returncode, 0, extra)
                    # консоль PowerShell 5.1 отдаёт кириллицу в OEM-кодировке: по тексту проверяем только «SQLite»
                    self.assertIn("SQLite", result.stdout + result.stderr)
                    self.assertEqual(self.tree(), before)
                self.assertFalse(self.dest.exists())

    def test_database_header_and_destination_are_checked(self):
        con = sqlite3.connect(self.src / "copy.bin")
        con.execute("CREATE TABLE t(x)")
        con.commit()
        con.close()
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SQLite", result.stdout + result.stderr)
        self.assertFalse(self.dest.exists())
        (self.src / "copy.bin").unlink()
        (self.src / "note.txt").write_text("x", encoding="utf-8")
        self.dest.mkdir()
        (self.dest / "state.sqlite3").write_bytes(b"live")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.dest / "state.sqlite3").read_bytes(), b"live")
        self.assertFalse((self.dest / "note.txt").exists())


if __name__ == "__main__":
    unittest.main()
