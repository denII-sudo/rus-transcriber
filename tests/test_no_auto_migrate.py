"""Перенос старой раскладки данных — только явной командой, не при старте сервера."""
import ast
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


class NoAutoMigrateTests(unittest.TestCase):
    def test_server_does_not_run_migration_on_import(self):
        tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
        imports = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "migrate"]
        self.assertNotIn("migrate", imports)
        self.assertEqual(calls, [])

    def test_settings_show_pending_legacy_layout(self):
        import core
        import migrate
        for pending in (True, False):
            with self.subTest(pending=pending), mock.patch.object(migrate, "pending", return_value=pending):
                self.assertIs(core.settings_info()["legacy_migration_pending"], pending)


if __name__ == "__main__":
    unittest.main()
