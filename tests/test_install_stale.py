"""Обновление через install.ps1: устаревшие файлы прежней сборки и режимы -ExtraFiles (нужен Windows PowerShell)."""
import os
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _shell():
    for name in ("powershell", "pwsh"):
        found = shutil.which(name)
        if found:
            return found
    return None


@unittest.skipUnless(os.name == "nt" and _shell() and shutil.which("python"), "нужны Windows, PowerShell и python")
class InstallStaleFilesTests(unittest.TestCase):
    def test_stale_files_modes_and_links_in_a_temporary_sandbox(self):
        # Обновление копирует файлы только в режиме приёмки внутри временной папки гарнитуры;
        # установка и данные владельца не затрагиваются. Ввод закрыт: установщик не должен ничего спрашивать.
        result = subprocess.run(
            [_shell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             str(ROOT / "tests" / "install_stale_harness.ps1"), "-Script", str(ROOT / "install.ps1")],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=600)
        tail = (result.stdout + result.stderr)[-4000:]
        self.assertEqual(result.returncode, 0, tail)
        self.assertIn("FAILED: 0", result.stdout, tail)

    def test_same_scenarios_in_a_console_with_code_page_65001(self):
        # Windows PowerShell 5.1 в консоли 65001 дописывает BOM в начало ввода внешней программы: прежняя
        # передача списка через конвейер ломала удаление. Те же сценарии под chcp 65001 (сам сценарий L
        # гарнитуры кроме того задаёт $OutputEncoding с BOM, поэтому не зависит от консоли тестового запуска).
        command = ('chcp 65001 >nul & "%s" -NoProfile -ExecutionPolicy Bypass -File "%s" -Script "%s"'
                   % (_shell(), ROOT / "tests" / "install_stale_harness.ps1", ROOT / "install.ps1"))
        result = subprocess.run('cmd /d /s /c "' + command + '"',  # одна строка: свои кавычки cmd
                                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=600)
        tail = (result.stdout + result.stderr)[-4000:]
        self.assertEqual(result.returncode, 0, tail)
        self.assertIn("FAILED: 0", result.stdout, tail)


if __name__ == "__main__":
    unittest.main()
