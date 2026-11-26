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
        # К-13: «без ответа» считает по состоянию ответа (needsAnswer); применённый ответ — отвечено. Берётся
        # настоящее начало сценария страницы (needsAnswer, stateOf) до того же признака, что у других тестов
        # страницы, — без привязки к форме записи функций.
        head = html.split("<script>", 1)[1].split("  // ответы этой версии вопросов")[0]
        self.assertIn("needsAnswer", head)
        script = 'global.location={hostname:"127.0.0.1"};\n' + head + """
man = {items: [{id: 'r1'}, {id: 'r2'}], applied: true,
       answer_states: {r1: {state: 'applied', ts: 't1', source: 'local'}}};
answers = {r1: {kind: 'option', option: 0, ts: 't1', source: 'local'}};
if (stateOf(man.items[0], answers.r1) !== 'applied') throw Error('состояние ответа');
showAll = false;
const filter = %s;
const open = man.items.filter(filter).map(it => it.id);
showAll = true;
const all = man.items.filter(filter).map(it => it.id);
console.log(JSON.stringify({open, all}));
})();
""" % match.group(1)
        result = subprocess.run([shutil.which("node")], input=script, capture_output=True, text=True,
                                encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"open": ["r2"], "all": ["r1", "r2"]})


if __name__ == "__main__":
    unittest.main()
