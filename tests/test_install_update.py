"""Обновление через install.ps1: помощники и проверки перед изменениями (нужен PowerShell)."""
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent


def _shell():
    for name in (("powershell",) if os.name == "nt" else ()) + ("pwsh",):
        found = shutil.which(name)
        if found:
            return found
    return None


def setUpModule():
    """An inherited GIT_DIR (hooks, rebase --exec) overrides git -C: never write to that repository."""
    environment = patch.dict(os.environ)
    environment.start()
    unittest.addModuleCleanup(environment.stop)
    for name in [n for n in os.environ if n.upper().startswith("GIT_")]:
        del os.environ[name]


@unittest.skipUnless(_shell(), "PowerShell не найден")
class InstallUpdateTests(unittest.TestCase):
    def test_update_helpers_and_preflight_guards(self):
        result = subprocess.run(
            [_shell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             str(ROOT / "tests" / "install_update_harness.ps1"), "-Script", str(ROOT / "install.ps1")],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=600)
        tail = (result.stdout + result.stderr)[-4000:]
        self.assertEqual(result.returncode, 0, tail)
        self.assertIn("FAILED: 0", result.stdout, tail)


if __name__ == "__main__":
    unittest.main()
