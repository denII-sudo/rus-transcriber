"""РАР-6.1, К-б: надписи страницы карточек — находки проверки карточек К-1, К-7, К-8, К-9, К-14, К-17, К-19,
К-21, К-24 и находки проверки аудита И1, И4 (план исправлений v2.3, приватный репозиторий:
research/2026-11-25-cards-scenarios).

Только хранение записей в базе (sqlite): решение владельца «в доработках новых не учитывай файловую запись».
Страница проверяется функциями cards.html в Node (без браузера). Данные синтетические.
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-card-labels-data-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_pause_buttons_short_answers as six  # noqa: E402
from test_pause_buttons_short_answers import files  # noqa: E402

core = six.core
review_pack = six.review_pack
PROJECT = six.PROJECT
REPLACE = "Заменить на «тот»"


def page(body):
    """Начало сценария cards.html (состояния, тексты, отчёт) и функции от разбора ответов до cardFor — в Node.
    Документ — заглушка: строка предупреждения вверху (mode-note) и строки состояния карточек."""
    html = (PROJECT / "cards.html").read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1]
    head, rest = script.split("  // ответы этой версии вопросов", 1)
    middle = "  // ответы этой версии вопросов" + rest.split("  function card(it)", 1)[0]
    stubs = ('global.location={hostname:"127.0.0.1"};\n'
             'const modeNote={textContent:"", dataset:{}, hidden:true}, states={};\n'
             'global.document={getElementById:id => id === "mode-note" ? modeNote : null,\n'
             '  querySelector:sel => { const m = /data-id="([^"]+)"/.exec(sel); return m ? (states[m[1]] = states[m[1]] || '
             '{textContent:""}) : null; },\n'
             '  createElement:() => ({style:{}, textContent:"", dataset:{}})};\n')
    wrapped = "(async () => {\n" + body + "\n})().catch(e => { console.error(e && e.message || e); process.exit(1); });\n"
    # сценарий длиннее предела командной строки Windows — через stdin
    return subprocess.run([shutil.which("node")], input=stubs + head + middle + wrapped + "})();\n",
                          capture_output=True, text=True, encoding="utf-8")


class Texts(unittest.TestCase):
    def test_listen_texts_name_no_button_and_no_pause(self):
        """К-7: причина applied_open и LISTEN_STALE не называют «Я прослушал» и «паузу»."""
        why = review_pack.OUTCOME_REASONS["applied_open"]
        self.assertNotIn("Я прослушал", why)
        self.assertIn("кнопкой прослушивания на карточке", why)
        self.assertNotIn("пауз", review_pack.LISTEN_STALE)
        self.assertIn("интервале прослушивания", review_pack.LISTEN_STALE)


class RepeatApply(six.Case):
    def test_second_apply_reports_unchanged(self):
        """К-14: повторное «Применить» отдаёт неменявшийся ответ в unchanged; страница — «Новых изменений нет»."""
        with files("sqlite") as (jp, md, version):
            self.press(jp, version, REPLACE, "r4")
            first = core.apply_answers_files(jp, md)
            self.assertNotIn("r4", first.get("unchanged") or [])
            rep = core.apply_answers_files(jp, md)
            self.assertIn("r4", rep.get("unchanged") or [])
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
            if not shutil.which("node"):
                return
            items = review_pack.manifest(jp)["items"]
            out = page("console.log(JSON.stringify(reportLines(%s, %s)));" % (
                json.dumps({k: rep.get(k) for k in ("outcomes", "unchanged")}, ensure_ascii=False),
                json.dumps(items, ensure_ascii=False)))
            self.assertEqual(out.returncode, 0, out.stderr)
            lines = json.loads(out.stdout)
            self.assertEqual(lines[0], "Новых изменений нет")
            self.assertTrue(any(line.endswith("ответ применён раньше — без изменений") for line in lines), lines)


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class PageLabels(unittest.TestCase):
    def run_page(self, body):
        result = page(body)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_contradiction_preview(self):
        """К-1: превью вариантов противоречия — словами, без хэша предложения и служебных имён."""
        self.run_page(r'''
          const keep = optionPreview({contradiction: "keep"});
          const accept = optionPreview({text: "тут", contradiction: "accept", proposal: "f00dbabe"});
          const raw = optionPreview({text: "та", contradiction: "raw"});
          if (keep !== "слова не меняются, противоречие закрывается") throw Error(keep);
          if (accept !== "текст → «тут»; противоречие закрывается") throw Error(accept);
          if (raw !== "текст → «та»; противоречие закрывается") throw Error(raw);
        ''')

    def test_history_kinds_and_applied(self):
        """К-21: сначала вид ответа; К-8: применённая правка осталась, ждущий ответ — ответить на новой редакции."""
        self.run_page(r'''
          const h = (answer, applied) => ({rev: 1, question: "q", options: [{label: "x"}], answer, applied});
          const it = {history: [h({kind: "skip", text: "пропущено"}, false), h({kind: "revert"}, false),
                                h({kind: "human", text: "не слышно"}, false), h({kind: "option", option: 0}, true),
                                h({kind: "free", text: "те"}, false), {rev: 2, question: "q2"}]};
          const lines = historyNote(it).textContent.split("\n");
          const want = ["ответ: пропущено (не применён", "ответ: отмена ответа (", "ответ: оставлено человеку («не слышно») (",
                        "ответ: 1 — x (был применён к стенограмме; правка осталась — отменить её с карточки нельзя, " +
                        "исправьте ответом на эту карточку или правкой отрезка)", "ответ: «те» (не применён: ответьте на новой редакции)",
                        "ответа не было."];
          want.forEach((w, n) => { if (!lines[n].includes("— " + w)) throw Error(n + ": " + lines[n]); });
        ''')

    def test_saved_left_to_human(self):
        """К-9: сохранённое «Оставить человеку» до «Применить»."""
        self.run_page(r'''
          man = {answer_states: {}};
          const s = cardStatus({id: "r1"}, {kind: "human", ts: "t", source: "local"});
          if (s !== "оставлено человеку — ещё не применено") throw Error(s);
        ''')

    def test_report_unchanged_lines(self):
        """К-14: строки неменявшихся ответов; «Новых изменений нет» — только если таких все."""
        self.run_page(r'''
          const items = [{id: "r1", t: "00:01"}, {id: "r2", t: "00:02"}, {id: "r3", t: "00:03"}];
          const o = (id, card, outcome) => ({id, card, outcome, why: "исход " + outcome});
          let lines = reportLines({outcomes: [o("r1", 1, "applied"), o("r2", 2, "applied_check")], unchanged: ["r1", "r2"]}, items);
          const want = ["Новых изменений нет", "Карточка 1 (00:01): ответ применён раньше — без изменений",
                        "Карточка 2 (00:02): ответ применён раньше — ждёт вашей проверки"];
          if (JSON.stringify(lines) !== JSON.stringify(want)) throw Error(JSON.stringify(lines));
          lines = reportLines({outcomes: [o("r1", 1, "applied"), o("r3", 3, "applied")], unchanged: ["r1"]}, items);
          if (lines[0] === "Новых изменений нет" || lines[1] !== "Карточка 3 (00:03): исход applied") throw Error(JSON.stringify(lines));
          // место открыто после прежнего ответа: исход не прячется за «без изменений»
          lines = reportLines({outcomes: [o("r1", 1, "applied_open")], unchanged: ["r1"]}, items);
          if (JSON.stringify(lines) !== JSON.stringify(["Карточка 1 (00:01): исход applied_open"])) throw Error(JSON.stringify(lines));
        ''')

    def test_negative_over_applied_names_the_applied_answer(self):
        """К-19: «неверно» поверх применённого — строка называет применённый ответ."""
        self.run_page(r'''
          const it = {id: "r1", options: [{label: "Анна"}]}, a = {kind: "free", text: "неверно", ts: "t2", source: "local"};
          man = {answer_states: {r1: {state: "kept_previous", ts: "t2", source: "local"}}, applied_answers: {r1: ["option", 0, null]}};
          const text = answerText(a, it);
          if (text !== "применён: 1 — Анна; ваше «неверно» его не меняет") throw Error(text);
          man.answer_states.r1.state = "negative";
          if (answerText(a, it) !== "свой ответ: «неверно»") throw Error(answerText(a, it));
        ''')

    def test_asked_text_caption(self):
        """К-24: подпись «текст на момент вопроса» — у применённого и отменённого ответа, не у сохранённого."""
        self.run_page(r'''
          const it = {id: "r1"}, a = {kind: "option", option: 0, ts: "t", source: "local"};
          const at = (state) => { man = {answer_states: {r1: {state, ts: "t", source: "local"}}}; return showsAskedText(it, a); };
          if (!at("applied") || !at("reverted") || !at("applied_open")) throw Error("применён или отменён");
          if (at("saved") || showsAskedText(it, undefined)) throw Error("сохранён или без ответа");
        ''')

    def test_stale_refusal_text_and_note_carried(self):
        """К-17: отказ «вопрос обновлён» — для варианта «выберите ответ ещё раз», для своего текста — прежний;
        набранное примечание переходит в новую редакцию карточки."""
        self.run_page(r'''
          const it = {id: "r1", rev: 2, options: [{label: "x"}]};
          man = {version: "v1", answer_states: {}};
          api = {async answer() { return {rejected: [{why: STALE}]}; }};
          global.humanNote = () => ""; global.render = () => {};
          await send(it, {option: 0}, {textContent: ""});
          if (states.r1.textContent !== STALE_CHOICE_TEXT) throw Error(states.r1.textContent);
          await send(it, {text: "те"}, {textContent: ""});
          if (states.r1.textContent !== STALE_TEXT) throw Error(states.r1.textContent);
          // cardFor: прежняя редакция с набранным текстом и примечанием → новая редакция
          const el = () => { const f = {".free input": {value: ""}, ".human-note": {value: ""}}; return {querySelector: s => f[s]}; };
          global.card = el;
          const one = cardFor(it);
          one.querySelector(".free input").value = "свой"; one.querySelector(".human-note").value = "сверил";
          it.rev = 3;
          const two = cardFor(it);
          if (two === one || two.querySelector(".human-note").value !== "сверил" || two.querySelector(".free input").value !== "свой")
            throw Error("примечание не перенесено");
        ''')

    def test_stale_when_new_revision_does_not_load(self):
        """И1: новая редакция не загрузилась — карточка не перерисовывается и не названа «показанной заново»."""
        self.run_page(r'''
          const it = {id: "r1"}; let drawn = 0;
          global.render = () => { drawn++; };
          api = {async load() { throw new Error("503"); }};
          if (!(await staleShown(new Error(STALE), it))) throw Error("отказ не узнан");
          if (states.r1.textContent !== STALE_LOAD_FAILED || drawn) throw Error(states.r1.textContent + " " + drawn);
          api = {async load() { return {}; }};
          await staleShown(new Error(STALE), it, STALE_CHOICE_TEXT);
          if (states.r1.textContent !== STALE_CHOICE_TEXT || drawn !== 1) throw Error(states.r1.textContent + " " + drawn);
          if (await staleShown(new Error("сеть"), it)) throw Error("чужой отказ");
        ''')

    def test_return_to_tab_failure_is_shown(self):
        """И1: возврат на вкладку без обновления — предупреждение; удачная загрузка снимает только его."""
        self.run_page(r'''
          global.render = () => {};
          api = {async load() { throw new Error("503"); }};
          await refreshOnReturn();
          if (modeNote.textContent !== RETURN_FAILED || modeNote.dataset.kind !== "warn") throw Error(modeNote.textContent);
          api = {async load() { return {}; }};
          await refreshOnReturn();
          if (modeNote.textContent !== "") throw Error("предупреждение не снято");
          note("другое предупреждение", "warn");
          await refreshOnReturn();
          if (modeNote.textContent !== "другое предупреждение") throw Error("снято чужое предупреждение");
        ''')

    def test_event_failure_warns_once(self):
        """И4: отказ записи отметки — одно предупреждение на сессию страницы; «вопрос обновлён» — без него."""
        self.run_page(r'''
          man = {version: "v1"}; let shown = 0;
          const it = {id: "r1"}, tick = () => new Promise(r => setTimeout(r, 0));
          api = {event: async () => { throw new Error(STALE); }};
          sendInteraction(it, "audio_play"); await tick();
          if (modeNote.textContent) throw Error("предупреждение при замене карточки");
          api = {event: async () => { shown++; throw new Error("500"); }};
          sendInteraction(it, "audio_play"); sendInteraction(it, "audio_ended"); await tick();
          if (shown !== 2 || modeNote.textContent !== EVENTS_FAILED) throw Error(modeNote.textContent);
          note(""); sendInteraction(it, "card_visible"); await tick();
          if (modeNote.textContent) throw Error("предупреждение повторилось");
        ''')


if __name__ == "__main__":
    unittest.main()
