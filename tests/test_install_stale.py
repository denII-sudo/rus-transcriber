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


if __name__ == "__main__":
    unittest.main()
