"""Подтверждение опасных действий человеком: окно, терминал, одноразовость, журнал. Запуск: python -m unittest discover -s tests"""
import atexit
import shutil
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-data-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_OLD_ENV = os.environ.get("RUS_TRANSCRIBER_DATA")
os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import confirm  # noqa: E402
import paths  # noqa: E402
import settings  # noqa: E402
if _OLD_ENV is None:
    os.environ.pop("RUS_TRANSCRIBER_DATA", None)
else:
    os.environ["RUS_TRANSCRIBER_DATA"] = _OLD_ENV


def cfg(**kv):
    return mock.patch.object(settings, "load_config", return_value={**settings.DEFAULTS, **kv})


class ConfirmTests(unittest.TestCase):
    def setUp(self):
        for p in (confirm.PENDING, confirm.APPROVED, confirm.LOG):
            if p.exists():
                p.unlink()

    def log_rows(self):
        return [json.loads(l) for l in confirm.LOG.read_text(encoding="utf-8").splitlines()] if confirm.LOG.exists() else []

    def test_key_depends_on_action_and_args(self):
        a = confirm.key_of("Удаление группы", {"group": "g", "permanent": False})
        self.assertEqual(a, confirm.key_of("Удаление группы", {"permanent": False, "group": "g"}))
        self.assertNotEqual(a, confirm.key_of("Удаление группы", {"group": "g", "permanent": True}))
        self.assertNotEqual(a, confirm.key_of("Выгрузка группы", {"group": "g", "permanent": False}))

    def test_mode_default_and_unknown_is_strict(self):
        with cfg():
            self.assertEqual(confirm.mode(), "auto")
        with cfg(confirm_mode="что-то"):
            self.assertEqual(confirm.mode(), "auto")           # опечатка не отключает подтверждение
        with cfg(confirm_mode="off"):
            self.assertEqual(confirm.mode(), "off")

    def test_off_allows_and_logs(self):
        with cfg(confirm_mode="off"):
            self.assertEqual(confirm.require("Очистка", {"d": 1}, ["x"]), "off")
        self.assertIn("off", self.log_rows()[-1]["how"])

    def test_dialog_no_blocks_and_logs(self):
        with cfg(), mock.patch.object(confirm, "dialog_available", return_value=True), \
                mock.patch.object(confirm, "ask_dialog", return_value=False):
            with self.assertRaises(confirm.NotConfirmed) as cm:
                confirm.require("Удаление группы", {"g": "x"}, ["Удаление группы «x»"])
        self.assertIn("не подтвердил", str(cm.exception))
        self.assertEqual(self.log_rows()[-1]["decision"], "отказ или нет ответа")

    def test_dialog_yes_allows_and_shows_program_text(self):
        seen = {}

        def fake(text, timeout):
            seen["text"] = text
            return True
        with cfg(), mock.patch.object(confirm, "dialog_available", return_value=True), \
                mock.patch.object(confirm, "ask_dialog", side_effect=fake):
            self.assertEqual(confirm.require("Удаление группы", {"g": "x"}, ["Удаление группы «x» (файлов: 2)."]), "dialog")
        self.assertIn("файлов: 2", seen["text"])                 # текст окна — описание программы
        self.assertEqual(self.log_rows()[-1]["decision"], "разрешено")

    def test_terminal_flow_is_one_time_and_bound_to_args(self):
        with cfg(), mock.patch.object(confirm, "dialog_available", return_value=False):
            with self.assertRaises(confirm.NotConfirmed) as cm:
                confirm.require("Выгрузка группы", {"g": "a"}, ["Выгрузка"])
            self.assertIn("rt.py approve", str(cm.exception))
            key = confirm.key_of("Выгрузка группы", {"g": "a"})
            self.assertIn(key, confirm.pending_requests())
            # до approve разрешения нет
            with self.assertRaises(confirm.NotConfirmed):
                confirm.require("Выгрузка группы", {"g": "a"}, ["Выгрузка"])
            self.assertTrue(confirm.approve(key))
            # другие параметры — не то, что человек разрешил
            with self.assertRaises(confirm.NotConfirmed):
                confirm.require("Выгрузка группы", {"g": "b"}, ["Выгрузка"])
            self.assertEqual(confirm.require("Выгрузка группы", {"g": "a"}, ["Выгрузка"]), "terminal")
            # разрешение одноразовое
            with self.assertRaises(confirm.NotConfirmed):
                confirm.require("Выгрузка группы", {"g": "a"}, ["Выгрузка"])

    def test_approve_unknown_key_fails(self):
        self.assertFalse(confirm.approve("нет-такого"))

    def test_expired_approval_and_request(self):
        with cfg(), mock.patch.object(confirm, "dialog_available", return_value=False):
            with self.assertRaises(confirm.NotConfirmed):
                confirm.require("Очистка", {"d": 1}, ["x"])
            key = confirm.key_of("Очистка", {"d": 1})
            confirm.approve(key)
            real = time.time()
            with mock.patch("confirm.time.time", return_value=real + confirm.TTL + 5):
                self.assertFalse(confirm.consume(key))           # срок вышел
                self.assertEqual(confirm.pending_requests(), {})

    def test_dialog_only_mode_without_screen_refuses(self):
        with cfg(confirm_mode="dialog"), mock.patch.object(confirm, "dialog_available", return_value=False):
            with self.assertRaises(confirm.NotConfirmed):
                confirm.require("Очистка", {"d": 1}, ["x"])
        self.assertEqual(confirm.pending_requests(), {})

    def test_ask_dialog_results(self):
        ok = subprocess.CompletedProcess([], 0)
        no = subprocess.CompletedProcess([], 1)
        with mock.patch.object(confirm.subprocess, "run", return_value=ok):
            self.assertTrue(confirm.ask_dialog("x", 5))
        with mock.patch.object(confirm.subprocess, "run", return_value=no):
            self.assertFalse(confirm.ask_dialog("x", 5))
        with mock.patch.object(confirm.subprocess, "run", side_effect=subprocess.TimeoutExpired("c", 5)):
            self.assertFalse(confirm.ask_dialog("x", 5))          # нет ответа — отказ
        with mock.patch.object(confirm.subprocess, "run", side_effect=OSError("нет")):
            self.assertFalse(confirm.ask_dialog("x", 5))

    def test_clean_strips_controls_and_limits(self):
        t = confirm.clean("а\x00б\x1b[31mв\n" + "я" * 500)
        self.assertNotIn("\x00", t)
        self.assertNotIn("\x1b", t)
        self.assertTrue(all(len(l) <= 220 for l in t.split("\n")))
        self.assertLessEqual(len(confirm.clean("я\n" * 2000)), 1410)


class NoSelfApproveTests(unittest.TestCase):
    def test_no_mcp_tool_can_approve(self):
        """Ни один инструмент MCP не выдаёт разрешение: модель не может подтвердить за человека."""
        try:
            import migrate
            with mock.patch.object(migrate, "run", return_value={"status": "пропущено в тесте"}):
                import server
        except ImportError:
            self.skipTest("нет пакета mcp")
        names = [t.name for t in server.mcp._tool_manager.list_tools()] if hasattr(server.mcp, "_tool_manager") else None
        if names is None:
            self.skipTest("не удалось получить список инструментов этой версии mcp")
        self.assertFalse([n for n in names if "approve" in n or "confirm" in n or "allow" in n], names)

    def test_core_and_confirm_do_not_import_mcp(self):
        for fn in ("core.py", "confirm.py", "rt.py"):
            tree = ast.parse((ROOT / fn).read_text(encoding="utf-8"))
            mods = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module] + \
                   [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            self.assertFalse([m for m in mods if m.split(".")[0] == "mcp"], f"{fn} не должен знать про MCP")


if __name__ == "__main__":
    unittest.main()
