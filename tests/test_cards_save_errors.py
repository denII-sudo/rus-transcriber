"""Аудит страницы карточек 28.11 (Н-А1, Н-А2): исходы сохранения различаются.

Н-А1: запись прошла, а перечитать список не удалось — страница пишет «Сохранено, но список ответов не обновился»,
а не «Не сохранилось»; отказ сервера — по-прежнему «Не сохранилось»; то же для «Применить».
Н-А2: облачная запись — карточка меняется только после успешного `set`; при отказе сообщение видно на карточке,
прежний ответ остаётся; ошибка подписки — предупреждение. Сценарии — по воспроизведению (ops
`research/2026-11-28-audit-na-repro/`). Node-заглушки, данные синтетические.
"""
import shutil
import unittest

from test_cards_ui_buttons import page

PRE = r'''
  man = {version: "v1", items: [{id: "c1", options: [{label: "да"}, {label: "нет"}]}], answer_states: {}};
  answers = {}; live = {};
  const shown = {};
  flash = (id, text) => { shown[id] = text; };
  globalThis.render = () => {};
  const stateEl = {textContent: "", isConnected: true};
  const resp = (status, body) => ({ok: status < 300, status, json: async () => body});
'''


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class LocalSave(unittest.TestCase):
    def run_page(self, body):
        result = page(PRE + body)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_saved_but_reload_failed_is_not_a_refusal(self):
        self.run_page(r'''
          global.fetch = async (url) => {
            if (url === "api/answer") return resp(200, {accepted: [{id: "c1", answer: "нет"}]});
            throw new Error("Failed to fetch");                    // перечитывание оборвалось
          };
          api = localApi();
          await send(man.items[0], {option: 1}, stateEl);
          if (shown.c1 !== RELOAD_FAILED) throw Error("показано: " + shown.c1 + " / " + stateEl.textContent);
          if (stateEl.textContent.startsWith("Не сохранилось")) throw Error(stateEl.textContent);
        ''')

    def test_server_refusal_is_still_a_refusal(self):
        self.run_page(r'''
          global.fetch = async () => resp(400, {error: "плохой ответ"});
          api = localApi();
          await send(man.items[0], {option: 1}, stateEl);
          if (stateEl.textContent !== "Не сохранилось: плохой ответ") throw Error(stateEl.textContent);
        ''')

    def test_apply_saved_but_reload_failed_is_marked(self):
        self.run_page(r'''
          global.fetch = async (url) => {
            if (url === "api/apply") return resp(200, {outcomes: []});
            throw new Error("Failed to fetch");
          };
          const rep = await localApi().apply();
          if (!rep.reloadFailed) throw Error(JSON.stringify(rep));
          global.fetch = async () => resp(200, {answers: {}});
          const ok = await localApi().apply();
          if (ok.reloadFailed) throw Error("успех помечен как сбой");
        ''')


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class CloudSave(unittest.TestCase):
    CLOUD = r'''
      const notes = [];
      note = (text, kind) => notes.push(text);
      let snapErr = null, setFails = false;
      global.claude = {use: async (what) => what === "db" ? {collection: () => ({
        doc: () => ({set: async () => { if (setFails) throw new Error("синтетический отказ записи"); }}),
        get: async () => ({docs: []}),
        onSnapshot: (ok, err) => { snapErr = err; }})} : null};
    '''

    def run_page(self, body):
        result = page(PRE + self.CLOUD + body)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_write_keeps_previous_answer_and_is_visible(self):
        self.run_page(r'''
          api = await cloudApi();
          live = {c1: {id: "c1", option: 0, ts: "t0", source: "cloud"}}; answers = merge();
          setFails = true;
          stateEl.isConnected = false;                    // карточку перерисовали — прежний элемент снят
          await send(man.items[0], {option: 1}, stateEl);
          if (shown.c1 !== "Не сохранилось: синтетический отказ записи") throw Error("видно: " + shown.c1);
          if (live.c1.option !== 0) throw Error("незаписанный ответ показан как сохранённый: " + JSON.stringify(live.c1));
          setFails = false;
          await send(man.items[0], {option: 1}, stateEl);
          if (live.c1.option !== 1) throw Error("записанный ответ не показан");
        ''')

    def test_subscription_error_warns(self):
        self.run_page(r'''
          api = await cloudApi();
          api.subscribe();
          snapErr(new Error("обрыв"));
          if (!notes.includes(SUBSCRIBE_FAILED)) throw Error(JSON.stringify(notes));
        ''')


if __name__ == "__main__":
    unittest.main()
