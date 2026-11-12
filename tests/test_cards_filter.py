"""Проверка фильтра страницы карточек без зависимости от MCP-клиента."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest


class CardsFilterTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js нужен для проверки JavaScript")
    def test_answered_cards_hidden_even_after_application(self):
        html = (Path(__file__).resolve().parent.parent / "cards.html").read_text(encoding="utf-8")
        match = re.search(r"const show = man\.items\.filter\((.*?)\);", html)
        self.assertIsNotNone(match)
        script = """
const man = {items: [{id: 'r1'}, {id: 'r2'}], applied: true};
const answers = {r1: {kind: 'option'}};
let showAll = false;
const filter = %s;
const open = man.items.filter(filter).map(it => it.id);
showAll = true;
const all = man.items.filter(filter).map(it => it.id);
console.log(JSON.stringify({open, all}));
""" % match.group(1)
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), {"open": ["r2"], "all": ["r1", "r2"]})
