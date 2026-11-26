"""РАР-6.1, К-а2: страница карточек и её малые серверные части — находки проверки карточек К-13, К-18, К-12,
К-16, К-2 (план исправлений v2.1, приватный репозиторий: research/2026-11-25-cards-scenarios).

Только хранение записей в базе (sqlite): решение владельца «в доработках новых не учитывай файловую запись».
Страница проверяется функциями cards.html в Node (без браузера, микрофона и облачного сервиса). Данные синтетические.
"""
import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-card-page-data-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_pause_buttons_short_answers as six  # noqa: E402
from test_pause_buttons_short_answers import files  # noqa: E402
import test_contradiction_card as cc  # noqa: E402

core = six.core
review_pack = six.review_pack
transcript = six.transcript
PROJECT = six.PROJECT
REPLACE = "Заменить на «тот»"
NOTE = "проверил по записи"


class ContradictionNote(cc.Base):
    """К-2: примечание на карточке противоречия — в истории места, когда ответ применён; иначе — «не записано»."""

    def contradicted(self, jp, md, version):
        self.answer_r4(jp, md, version)
        core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="claude")
        self.assertTrue(self.card(jp, "r4").get("seen_fp"))

    def rev(self, jp):
        return self.card(jp, "r4")["rev"]

    def test_option_note_in_history(self):
        with files("sqlite") as (jp, md, version):
            self.contradicted(jp, md, version)
            item = self.card(jp, "r4")
            option = next(n for n, o in enumerate(item["options"]) if o["label"].startswith(cc.ACCEPT))
            self.save(jp, version, "r4", rev=self.rev(jp), option=option, human_note=NOTE)
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
            self.assertEqual(self.place_of(jp, "r4")["contradiction"]["history"][-1].get("human_note"), NOTE)
            self.assertEqual(rep["notes_not_written"], [])

    def test_own_text_note_in_history_and_edits_log(self):
        with files("sqlite") as (jp, md, version):
            self.contradicted(jp, md, version)
            self.save(jp, version, "r4", rev=self.rev(jp), text="те", human_note=NOTE)
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
            self.assertEqual(self.place_of(jp, "r4")["contradiction"]["history"][-1].get("human_note"), NOTE)
            log = [e for e in transcript.load(jp).get("edits") or [] if e.get("id") == "r4"]
            self.assertEqual(log[-1].get("human_note"), NOTE)
            self.assertEqual(rep["notes_not_written"], [])

    def test_left_to_human_note_keeps_the_choice_open(self):
        with files("sqlite") as (jp, md, version):
            self.contradicted(jp, md, version)
            self.save(jp, version, "r4", rev=self.rev(jp), kind="human", human_note=NOTE)
            rep = core.apply_answers_files(jp, md)
            place = self.place_of(jp, "r4")
            self.assertEqual(place["status"], "human")
            self.assertEqual(place["contradiction"]["notes"][-1]["human_note"], NOTE)
            self.assertEqual(place["contradiction"]["notes"][-1]["choice"], "human")
            self.assertTrue(place["contradiction"]["proposals"], "противоречие ждёт выбора")
            self.assertEqual(rep["notes_not_written"], [])

    def test_own_text_left_to_human_note_is_not_written(self):
        """Свой текст, который программа не поняла как решение и оставила человеку: примечание не записано."""
        with files("sqlite") as (jp, md, version):
            self.contradicted(jp, md, version)
            self.save(jp, version, "r4", rev=self.rev(jp), text="не Анна: тут", human_note=NOTE)
            rep = core.apply_answers_files(jp, md)
            row = self.outcomes(rep)["r4"]
            self.assertEqual(row["outcome"], "not_applied", rep.get("outcomes"))
            self.assertEqual(self.place_of(jp, "r4")["status"], "human")
            self.assertIn(review_pack.NOTE_NOT_WRITTEN, row.get("detail", ""))
            self.assertEqual(rep["notes_not_written"], ["r4"])
            self.assertNotIn("notes", self.place_of(jp, "r4")["contradiction"])
            self.assertNotIn(NOTE, str(self.place_of(jp, "r4")["contradiction"].get("history")))

    def test_note_on_decided_place_is_not_written(self):
        with files("sqlite") as (jp, md, version):
            self.contradicted(jp, md, version)
            # место решено другим путём, пока карточка была открыта: выбор записан, предложений больше нет
            # (с ожидающими предложениями сохранение снова открывает место)
            data = transcript.load(jp)
            place = next(i for i in data["review"] if i["id"] == "r4")
            transcript.settle_contradiction(place, "keep", "human")
            place["status"] = "resolved"
            transcript.save(data, jp, md)
            self.assertEqual(self.place_of(jp, "r4")["status"], "resolved")
            self.save(jp, version, "r4", rev=self.rev(jp), kind="human", human_note=NOTE)
            rep = core.apply_answers_files(jp, md)
            row = self.outcomes(rep)["r4"]
            self.assertEqual(row["outcome"], "already_decided", rep.get("outcomes"))
            self.assertIn(review_pack.NOTE_NOT_WRITTEN, row.get("detail", ""))
            self.assertEqual(rep["notes_not_written"], ["r4"])
            self.assertNotIn("notes", self.place_of(jp, "r4")["contradiction"])
            self.assertIn(review_pack.NOTE_NOT_WRITTEN, review_pack.answer_states(jp)["r4"].get("detail", ""))


class RepeatIsNotAnAnswer(six.Case):
    """К-12: «повтори» — не ответ: не сохраняется и с текстом; применённый ответ остаётся."""

    def test_heard_repeat_is_not_stored(self):
        with files("sqlite") as (jp, _md, version):
            rep = self.save(jp, version, "r4", text="повтори", heard=True)
            self.assertEqual(rep["accepted"][0]["stored"], False)
            self.assertEqual(rep["accepted"][0]["note"], review_pack.REPEAT_NOT_STORED)
            self.assertNotIn("r4", review_pack.answers_state(jp)["answers"])

    def test_cloud_repeat_with_text_over_applied_answer(self):
        with files("sqlite") as (jp, md, version):
            self.press(jp, version, REPLACE, "r4")
            core.apply_answers_files(jp, md)
            before = review_pack.answers_state(jp)
            rep = review_pack.add_answers(jp, [{"id": "r4", "version": version, "kind": "repeat", "text": "ещё раз",
                                                "ts": "2026-11-24T23:00:00"}], "cloud")
            self.assertEqual(rep["accepted"][0]["note"], review_pack.REPEAT_NOT_STORED)
            after = review_pack.answers_state(jp)
            self.assertEqual(after["answers"]["r4"], before["answers"]["r4"])
            self.assertEqual(after["applied_answers"]["r4"], before["applied_answers"]["r4"])
            self.assertEqual(after["undo_snapshots"]["r4"], before["undo_snapshots"]["r4"])


class UndoWithoutSnapshot(six.Case):
    """К-18: без снимка — признак undo: false (неактивная «Отменить нельзя»); со снимком и поздней правкой — уточнение."""

    def applied(self, jp, md, version):
        self.press(jp, version, REPLACE, "r4")
        core.apply_answers_files(jp, md)

    def test_flag_only_without_snapshot(self):
        with files("sqlite") as (jp, md, version):
            self.applied(jp, md, version)
            self.assertNotIn("undo", review_pack.answer_states(jp)["r4"])
            state = review_pack.answers_state(jp)
            state["undo_snapshots"].pop("r4")
            review_pack._write_answers(jp, state)
            self.assertIs(review_pack.answer_states(jp)["r4"]["undo"], False)

    def test_later_edit_detail_keeps_the_button(self):
        with files("sqlite") as (jp, md, version):
            self.applied(jp, md, version)
            core.edit_span_files(jp, md, 2.0, 2.0, "вон", by="human")
            self.save(jp, version, "r4", kind="revert")
            rep = core.apply_answers_files(jp, md)
            row = self.outcomes(rep)["r4"]
            self.assertEqual(row["outcome"], "undo_conflict", rep.get("outcomes"))
            self.assertIn(review_pack.UNDO_LATER_EDIT, row.get("detail", ""))
            view = review_pack.answer_states(jp)["r4"]
            self.assertNotIn("undo", view)
            self.assertIn(review_pack.UNDO_LATER_EDIT, view.get("detail", ""))


def page(body, host="127.0.0.1"):
    """Функции cards.html в Node: начало сценария (состояния, тексты), отправка, озвучивание и разбор голоса."""
    html = (PROJECT / "cards.html").read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1]
    head = script.split("  // ответы этой версии вопросов")[0]
    send = "  async function send(" + script.split("  async function send(", 1)[1].split("  // Отказ «вопрос обновлён»")[0]
    voice = "  function speak(it)" + script.split("  function speak(it)", 1)[1].split("  // примечание человека")[0]
    stubs = (f'global.location={{hostname:"{host}"}};\n'
             'const spoken=[]; global.speechSynthesis={cancel(){}, getVoices(){return []}, speak(u){spoken.push(u.text)}};\n'
             'global.window={speechSynthesis:global.speechSynthesis}; global.SpeechSynthesisUtterance=function(t){this.text=t};\n')
    wrapped = ("(async () => {\n" + body + "\n})().catch(e => { console.error(e && e.message || e); process.exit(1); });\n")
    return subprocess.run([shutil.which("node"), "-e", stubs + head + send + voice + wrapped + "})();\n"],
                          capture_output=True, text=True, encoding="utf-8")


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class PageParts(unittest.TestCase):
    def run_page(self, body, host="127.0.0.1"):
        result = page(body, host)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_open_places_are_not_counted_as_answered(self):
        """К-13: исходы, требующие действия, — в «без ответа»; пустой список — по тому, есть ли что применять."""
        self.run_page(r'''
          const st = (s) => ({state: s, ts: "t", source: "local"});
          man = {items: ["r1","r2","r3","r4","r5","r6"].map(id => ({id})),
                 answer_states: {r1: st("negative"), r2: st("applied"), r3: st("saved"), r4: st("reverted"),
                                 r5: st("applied_check"), r6: st("applied_open")}};
          for (const it of man.items) answers[it.id] = {id: it.id, ts: "t", source: "local", option: 0};
          const open = man.items.filter(needsAnswer).map(it => it.id).join();
          if (open !== "r1,r4,r5,r6") throw Error(open);
          if (!emptyText().startsWith("Ответы сохранены")) throw Error(emptyText());
          man.answer_states.r3 = st("applied");
          if (emptyText() !== "Все места разобраны, применять нечего. Нажмите «все», чтобы увидеть карточки.") throw Error(emptyText());
          delete answers.r2;
          if (!needsAnswer({id: "r2"})) throw Error("место без ответа");
        ''')

    def test_cloud_counts_every_answer(self):
        self.run_page(r'''
          man = {items: [{id: "r1"}], answer_states: {r1: {state: "negative", ts: "t", source: "local"}}};
          answers.r1 = {id: "r1", ts: "t", source: "local", option: 0};
          if (needsAnswer({id: "r1"})) throw Error("в облаке состояния до переноса неизвестны");
        ''', host="example.org")

    def test_undo_button_without_snapshot(self):
        """В-1 (было К-18): у ответа без снимка неактивной «Отменить нельзя» больше нет — кнопок нет совсем (card),
        признак undo === false; у ждущего нового ответа поверх него — строка «Вернуть как было» вместо рецепта
        с отменой. Строка undo_conflict без повтора заголовка."""
        self.run_page(r'''
          const a = {id: "r1", option: 1, ts: "t", source: "local"}, it = {id: "r1", options: [{label: "x"}, {label: "y"}]};
          man = {answer_states: {r1: {state: "applied", ts: "t", source: "local", undo: false, detail: "строка В-1"}}};
          if (undoOf(it) !== false) throw Error("признак");
          if (programNote(it, a) !== "строка В-1") throw Error(programNote(it, a));
          man.answer_states.r1.state = "pending_new"; man.answer_states.r1.why = "рецепт с «Подготовить отмену»";
          if (!programNote(it, a).startsWith(PENDING_NO_UNDO) || programNote(it, a).includes("Подготовить отмену"))
            throw Error(programNote(it, a));
          man.answer_states.r1.state = "applied";
          delete man.answer_states.r1.undo;
          const b = revertButton(it, a);      // К-В3, было «Подготовить отмену»: отмена — в режиме изменения
          if (b.disabled || b.label !== "Отменить ответ") throw Error(JSON.stringify(b));
          const r = {id: "r1", kind: "revert", ts: "t", source: "local"};
          man.answer_states.r1.state = "undo_conflict";
          if (answerText(r, it) !== "отмена ответа не выполнена") throw Error(answerText(r, it));
        ''')

    def test_cloud_revert(self):
        """К-16: отмена из облака у применённого ответа — «отмена сохранена в облаке», кнопка «Не отменять»;
        устаревший manifest (ноутбук уже отменил ответ) — прежнее «сохранён в облаке»."""
        self.run_page(r'''
          const it = {id: "r1", options: [{label: "x"}]}, r = {id: "r1", kind: "revert", ts: "t2", source: "cloud"};
          man = {answer_states: {r1: {state: "applied", ts: "t1", source: "local"}}, applied_answers: {r1: ["option", 0, null]}};
          if (stateOf(it, r) !== "cloud_revert") throw Error(stateOf(it, r));
          if (cardStatus(it, r) !== "отмена сохранена в облаке — применённый ответ пока на месте") throw Error(cardStatus(it, r));
          if (!holdsApplied(it, r) || revertButton(it, r).label !== "Не отменять") throw Error("кнопки");
          if (programNote(it, r)) throw Error("уточнение прежнего ответа");
          man.applied_answers = {};
          if (stateOf(it, r) !== "cloud_saved") throw Error(stateOf(it, r));
        ''', host="example.org")

    def test_voice_proposal_and_repeat(self):
        """К-12: предполагаемый ответ словами человека; «повтори» озвучивает вопрос заново, кнопки «Подтвердить» нет."""
        self.run_page(r'''
          const it = {id: "r1", question: "Кто говорит?", options: [{label: "Анна"}, {label: "Борис"}], recommended: null};
          const texts = ["free", "human", "skip", "repeat"].map(kind => voiceProposal({match: {kind}}, it).text).join("|");
          if (texts !== "свой ответ|оставить человеку|пропустить|повторить вопрос") throw Error(texts);
          if (voiceProposal({match: {kind: "option", option: 1}}, it).text !== "Борис") throw Error("вариант");
          if (voiceProposal({}, it).text !== "не распознан") throw Error("пусто");
          const state = {textContent: ""};
          const p = voiceResult(it, {heard: "повтори", match: {kind: "repeat"}}, state);
          if (!p.repeat || state.textContent !== REPEAT_TEXT) throw Error(state.textContent);
          if (spoken.length !== 1 || !spoken[0].startsWith("Кто говорит?")) throw Error(JSON.stringify(spoken));
        ''')

    def test_note_on_laptop_and_skip_without_note(self):
        """К-2: примечание видно в строке ответа на ноутбуке; «Пропустить» его не отправляет и говорит об этом."""
        self.run_page(r'''
          const it = {id: "r1", options: [{label: "x"}]};
          if (answerText({kind: "human", human_note: "сверил"}, it) !== "оставлено человеку · примечание: «сверил»")
            throw Error(answerText({kind: "human", human_note: "сверил"}, it));
          man = {version: "v1", answer_states: {}};
          const sent = []; let flashed = "";
          api = {async answer(a) { sent.push(a); return {accepted: [{id: "r1", kind: a.kind || "option"}]}; }};
          // send берёт их из внешней области: в Node — глобальные заглушки
          global.humanNote = () => "сверил"; global.render = () => {}; global.flash = (_id, text) => { flashed = text; };
          const stateEl = {textContent: ""};
          await send(it, {kind: "skip", text: "пропущено"}, stateEl);
          if ("human_note" in sent[0]) throw Error("примечание ушло с пропуском");
          if (flashed !== SKIP_NOTE_TEXT) throw Error(flashed);
          await send(it, {kind: "human"}, stateEl);
          if (sent[1].human_note !== "сверил") throw Error(JSON.stringify(sent[1]));
        ''')

    def test_note_is_not_shown_in_cloud(self):
        self.run_page(r'''
          if (answerText({kind: "human", human_note: "сверил"}, {id: "r1"}) !== "оставлено человеку") throw Error("облако");
        ''', host="example.org")


if __name__ == "__main__":
    unittest.main()
