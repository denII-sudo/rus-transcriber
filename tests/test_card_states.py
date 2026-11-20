"""Проверка состояний карточек без браузера и настоящих данных."""
import shutil
import subprocess
import unittest
from pathlib import Path


class CardStates(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
    def test_apply_displays_voice_warning(self):
        html = (Path(__file__).resolve().parent.parent / "cards.html").read_text(encoding="utf-8")
        handler = html.split('  $("apply").onclick = ', 1)[1].split("\n\n  async function start()", 1)[0]
        script = r'''
          const box = {hidden: true, textContent: ""};
          const button = {disabled: false};
          const $ = id => id === "apply-report" ? box : button;
          const confirm = () => true;
          const api = {apply: async () => ({voice_profile_warning: {message: "Вклад мог остаться"}})};
          let showAll = false;
          const render = () => {};
          const handler = ''' + handler + r'''
          handler().then(() => {
            if (!box.textContent.includes("Вклад мог остаться")) throw Error("warning missing");
            if (button.disabled || box.hidden) throw Error("report state");
          }).catch(e => { console.error(e); process.exitCode = 1; });
        '''
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
    def test_applied_is_per_card_and_signature(self):
        html = (Path(__file__).resolve().parent.parent / "cards.html").read_text(encoding="utf-8")
        prefix = html.split('  // ответы этой версии вопросов')[0].split('<script>', 1)[1]
        script = 'global.location={hostname:"127.0.0.1"};\n' + prefix + r'''
          man = {applied_answers:{r1:["option",0,null]}};
          const one={id:"r1"}, two={id:"r2"};
          const a={kind:"option",option:0}, changed={kind:"option",option:1};
          if (!isApplied(one,a) || isApplied(two,a) || isApplied(one,changed) || !pending(one,changed)) throw Error("card state");
          if (cardStatus(one,a) !== "применён ответ" || cardStatus(two,a) !== "ответ сохранён") throw Error("status");
          if (cardStatus(two,{kind:"human"}) !== "требует проверки") throw Error("human");
          man.conflicts=[{id:"r1"}]; if (!cardStatus(one,a).startsWith("конфликт")) throw Error("conflict");
          if (!appliedCard(one,a)) throw Error("applied card");
          editing.add("r1"); if (appliedCard(one,a)) throw Error("editing card");
        })();
        '''
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
    def test_option_preview_hides_word_ids(self):
        # длинный список ID слов без пробелов растягивал кнопку варианта, ответ сжимался в столбик
        html = (Path(__file__).resolve().parent.parent / "cards.html").read_text(encoding="utf-8")
        fn = "function optionPreview" + html.split("  function optionPreview", 1)[1].split("\n  }\n", 1)[0] + "\n  }"
        script = fn + r'''
          const ids = ["06bce98f82c742068df0513adb17d321", "e3bd632f58744bbbb055f3efb2618c50"];
          const yes = optionPreview({word_ids: ids}), no = optionPreview({speaker: "Б", word_ids: ids});
          if (yes !== "подтвердить говорящего у слов места (2)") throw Error(yes);
          if (no !== "говорящий → Б; подтвердить говорящего у слов места (2)") throw Error(no);
          if (yes.includes(ids[0]) || no.includes("word_ids")) throw Error("ids shown");
        '''
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(".opt{display:flex;flex-wrap:wrap;", html)
        self.assertRegex(html, r"\.opt small\{[^}]*overflow-wrap:anywhere")
