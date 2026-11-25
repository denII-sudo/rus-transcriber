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
        # строки отчёта собирает отдельная функция страницы
        lines = "  function reportLines" + html.split("  function reportLines", 1)[1].split("\n  }\n", 1)[0] + "\n  }\n"
        script = lines + r'''
          const man = {items: []};
          const box = {hidden: true, textContent: "", style: {}};
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
    def test_apply_renders_with_fresh_states(self):
        # После «Применить» страница сначала забирает ответы и состояния у программы, потом рисует карточки.
        html = (Path(__file__).resolve().parent.parent / "cards.html").read_text(encoding="utf-8")
        script = html.split("<script>", 1)[1]
        prefix = script.split("  // ответы этой версии вопросов")[0]
        layer = "  // ответы этой версии вопросов" + script.split("  // ответы этой версии вопросов", 1)[1].split(
            "  async function cloudApi()", 1)[0]
        handler = html.split('  $("apply").onclick = ', 1)[1].split("\n\n  async function start()", 1)[0]
        lines = "  function reportLines" + html.split("  function reportLines", 1)[1].split("\n  }\n", 1)[0] + "\n  }\n"
        stubs = r'''
          global.location = {hostname: "127.0.0.1"};
          const box = {hidden: true, textContent: "", style: {}}, button = {disabled: false};
          global.document = {getElementById: id => id === "apply-report" ? box : button};
          global.confirm = () => true;
          const fresh = {answers: {r1: {id: "r1", version: "v", kind: "option", option: 0, ts: "t1", source: "local"}},
                         answer_states: {r1: {state: "applied", ts: "t1", source: "local"}}};
          global.fetch = async url => ({ok: true, json: async () => url === "api/answers" ? fresh : {outcomes: []}});
        '''
        body = r'''
          man = {version: "v", items: [{id: "r1", options: []}], answers: {}};
          answers = {r1: {id: "r1", version: "v", kind: "revert", ts: "t0", source: "local"}};
          man.answer_states = {r1: {state: "revert_pending", ts: "t0", source: "local"}};
          api = localApi();
          const seen = [];
          render = () => seen.push(stateOf({id: "r1"}, answers.r1));
          const handler = ''' + handler + r'''
          handler().then(() => {
            if (seen.join() !== "applied") throw Error("карточка нарисована со старым состоянием: " + seen.join());
          }).catch(e => { console.error(e); process.exitCode = 1; });
        })();
        '''
        source = stubs + prefix + lines + layer + "  let render;\n" + body
        result = subprocess.run([shutil.which("node"), "-e", source], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
    def test_appended_cards_appear_and_old_answers_stay(self):
        """PR 17а: после дописывания порции (pack_rev + 1, version та же) страница перечитывает manifest —
        новые карточки видны, ответ и состояние прежней карточки и номера карточек прежние."""
        html = (Path(__file__).resolve().parent.parent / "cards.html").read_text(encoding="utf-8")
        script = html.split("<script>", 1)[1]
        prefix = script.split("  // ответы этой версии вопросов")[0]
        layer = "  // ответы этой версии вопросов" + script.split("  // ответы этой версии вопросов", 1)[1].split(
            "  async function cloudApi()", 1)[0]
        reload = "  async function reloadManifest()" + html.split("  async function reloadManifest()", 1)[1].split(
            "\n  }\n", 1)[0] + "\n  }\n"
        stubs = r"""
          global.location = {hostname: "127.0.0.1"};
          global.document = {getElementById: () => ({})};
          const r1 = {id: "r1", version: "v", kind: "option", option: 0, ts: "t1", source: "local"};
          const state = {pack_version: "v", pack_rev: 1, answers: {r1}, applied_answers: {r1: "sig"},
                         answer_states: {r1: {state: "applied", ts: "t1", source: "local"}}};
          const fresh = {version: "v", pack_rev: 1, items: [{id: "r1", options: []}, {id: "r2", options: []}]};
          const seen = [];
          global.fetch = async url => { seen.push(url);
            return {ok: true, json: async () => url === "api/answers" ? state : fresh}; };
        """
        body = r"""
          man = {version: "v", pack_rev: 0, items: [{id: "r1", options: []}], answers: {}};
          const header = () => {};
          api = localApi();
          api.load().then(got => {
            if (!seen.includes("manifest.json")) throw Error("manifest не перечитан");
            if (man.items.map(i => i.id).join() !== "r1,r2") throw Error("новые карточки не видны: " + man.items.map(i => i.id));
            if (!got.r1 || got.r1.option !== 0) throw Error("ответ прежней карточки потерян");
            if (stateOf({id: "r1"}, got.r1) !== "applied") throw Error("состояние прежней карточки: " + stateOf({id: "r1"}, got.r1));
            if (got.r2) throw Error("у новой карточки ответ");
            if (man.items.findIndex(i => i.id === "r1") !== 0) throw Error("номер прежней карточки изменился");
          }).catch(e => { console.error(e); process.exitCode = 1; });
        })();
        """
        source = stubs + prefix + reload + layer + "  let render;\n" + body
        result = subprocess.run([shutil.which("node"), "-e", source], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
    def test_card_state_comes_from_program(self):
        # Состояние ответа приходит из программы (answer_states) и относится только к тому же ответу.
        html = (Path(__file__).resolve().parent.parent / "cards.html").read_text(encoding="utf-8")
        prefix = html.split('  // ответы этой версии вопросов')[0].split('<script>', 1)[1]
        script = 'global.location={hostname:"127.0.0.1"};\n' + prefix + r'''
          const a={kind:"option",option:0,ts:"t1",source:"local"}, changed={kind:"option",option:1,ts:"t2",source:"local"};
          man = {answer_states:{r1:{state:"applied",ts:"t1",source:"local",why:"применён"}}};
          const one={id:"r1"}, two={id:"r2"};
          if (stateOf(one,a) !== "applied" || stateOf(two,a) !== "saved" || stateOf(one,changed) !== "saved") throw Error("card state");
          if (cardStatus(one,a) !== "применён ответ" || cardStatus(two,a) !== "ответ сохранён") throw Error("status");
          if (cardStatus(two,{kind:"human"}) !== "оставлено человеку — ещё не применено") throw Error("human");   // К-9
          man.answer_states.r1 = {state:"pending_new",ts:"t2",source:"local",why:"ждёт"};
          if (!pending(one,changed) || cardStatus(one,changed) !== "новый ответ — не применён" || !appliedCard(one,changed)) throw Error("pending");
          man.answer_states.r1 = {state:"kept_previous",ts:"t2",source:"local",why:"оставлен"};
          if (cardStatus(one,changed) !== "применён прежний ответ" || !appliedCard(one,changed)) throw Error("kept");
          man.answer_states.r1 = {state:"applied",ts:"t1",source:"local"};
          man.conflicts=[{id:"r1"}]; if (cardStatus(one,a) !== "применён ответ" || !hasConflict(one)) throw Error("conflict");  // К-23: конфликт — строкой под ответом
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
