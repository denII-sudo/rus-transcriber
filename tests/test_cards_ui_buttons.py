"""Кнопки страницы карточек, шаги А+Б (план research/2026-11-26-cards-ui-buttons/plan.md приватного репозитория).

Карточка, собранная по версии стенограммы, которой нет в базе (у перенесённых из старых данных — версия 0):
страница узнаёт об этом признаком `card_conditions`, «Предпросмотр» говорит то же, что «Применить»
(`answer_guard`), дособирание обновляет её на открытом месте и после «Пропустить» (В-2). Внесённый ответ без
снимка — строка с шагом вместо кнопок (В-1). Уточнение о версии не ставится под пропуском (КН-5). Данные
синтетические, во временной папке.
"""
import atexit
import os
import shutil
import subprocess
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="rt-cards-ui-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:          # временная папка данных — до импорта программы
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP

import unittest  # noqa: E402
from unittest import mock  # noqa: E402

import test_pause_buttons_short_answers as six  # noqa: E402
from test_pause_buttons_short_answers import files  # noqa: E402

core = six.core
review_pack = six.review_pack
transcript = six.transcript
REPLACE = "Заменить на «тот»"


def only(*ids):
    return [q for q in six.questions() if q["id"] in ids]


def stale(jp):
    """Пакет как после переноса: версия стенограммы карточек — 0, такой в базе нет."""
    store = review_pack.answers_state(jp)
    store["manifest"]["source_revision"] = 0
    for item in store["manifest"]["items"]:
        item.pop("source_revision", None)
    review_pack._write_answers(jp, store)


def item_of(jp, aid):
    return next(i for i in review_pack.manifest(jp)["items"] if i["id"] == aid)


class Base(six.Case):
    def append(self, jp, qs):
        with mock.patch.object(core, "_done_json", return_value={"json": str(jp)}):
            return core.build_review_pack("demo", qs)


class Conditions(Base):
    def test_card_without_source_revision_cannot_apply(self):
        with files("sqlite") as (jp, _md, _version):
            before = review_pack.card_conditions(jp)
            self.assertEqual({k: v["cannot_apply"] for k, v in before.items()},
                             {"r1": False, "r2": False, "r3": False, "r4": False})
            self.assertTrue(all(v["place_open"] for v in before.values()))
            stale(jp)
            self.assertTrue(all(v["cannot_apply"] for v in review_pack.card_conditions(jp).values()))

    def test_file_storage_has_no_revision_check(self):
        with files("legacy_files") as (jp, _md, _version):
            stale(jp)
            self.assertFalse(any(v["cannot_apply"] for v in review_pack.card_conditions(jp).values()))


class SourceRefresh(Base):
    """В-2: дособирание обновляет карточку без версии стенограммы на открытом месте."""

    def test_skipped_card_gets_a_new_revision_once(self):
        with files("sqlite", qs=only("r4")) as (jp, _md, version):
            stale(jp)
            self.save(jp, version, "r4", kind="skip", text="пропущено")
            self.assertEqual(review_pack.answer_states(jp)["r4"]["state"], "skipped")
            rep = self.append(jp, only("r4"))          # тот же вопрос: без В-2 было бы «без изменений»
            self.assertEqual([r["id"] for r in rep["replaced"]], ["r4"])
            self.assertEqual(item_of(jp, "r4")["rev"], 2)
            self.assertFalse(review_pack.card_conditions(jp)["r4"]["cannot_apply"])
            again = self.append(jp, only("r4"))
            self.assertEqual({r["id"]: r["why"] for r in again["skipped"]}, {"r4": "уже в пакете"})

    def test_card_without_answer_and_with_saved_answer(self):
        with files("sqlite", qs=only("r2", "r4")) as (jp, _md, version):
            stale(jp)
            self.press(jp, version, REPLACE, "r4")
            rep = self.append(jp, only("r2", "r4"))
            self.assertEqual(sorted(r["id"] for r in rep["replaced"]), ["r2", "r4"])
            # ответ, который не внёсся бы никогда, остаётся в истории карточки — его видно на странице
            self.assertEqual(item_of(jp, "r4")["history"][-1]["answer"]["option"], 1)
            self.assertNotIn("r4", review_pack.answer_states(jp))

    def test_same_question_without_source_is_not_unchanged(self):
        """Тот же вопрос по версии, которой нет в базе, — новая редакция с нынешней версией, а не «без изменений»
        (_same_question): иначе карточка осталась бы неприменимой навсегда."""
        with files("sqlite", qs=only("r2")) as (jp, _md, _version):
            stale(jp)
            rep = self.append(jp, only("r2"))
            self.assertEqual((rep["unchanged"], [r["id"] for r in rep["replaced"]]), ([], ["r2"]))
            item = item_of(jp, "r2")
            self.assertGreater(review_pack.card_source_revision(item, review_pack.manifest(jp)), 0)
            self.assertEqual(self.append(jp, only("r2"))["skipped"], [{"id": "r2", "why": "уже в пакете"}])

    def test_closed_place_is_not_refreshed(self):
        with files("sqlite", qs=only("r4")) as (jp, _md, version):
            stale(jp)
            self.save(jp, version, "r4", kind="skip", text="пропущено")
            real = transcript.place_unresolved
            with mock.patch.object(transcript, "place_unresolved",
                                   side_effect=lambda it: it.get("id") != "r4" and real(it)):
                rep = self.append(jp, only("r4"))
            self.assertEqual({r["id"]: r["why"] for r in rep["skipped"]}, {"r4": "уже в пакете"})

    def test_applied_answer_keeps_its_card(self):
        with files("sqlite", qs=only("r4")) as (jp, md, version):
            self.press(jp, version, REPLACE, "r4")
            self.assertEqual(self.outcomes(self.apply(jp, md))["r4"]["outcome"], "applied")
            stale(jp)
            real = transcript.place_unresolved
            with mock.patch.object(transcript, "place_unresolved",
                                   side_effect=lambda it: it.get("id") == "r4" or real(it)):
                rep = self.append(jp, only("r4"))
            self.assertEqual({r["id"]: r["why"] for r in rep["skipped"]}, {"r4": "уже в пакете"})
            self.assertEqual(review_pack.answer_states(jp)["r4"]["state"], "applied")


class ApplyAndPreview(Base):
    def test_skip_has_no_source_detail_and_option_has_next_step(self):
        with files("sqlite", qs=only("r2", "r4")) as (jp, md, version):
            stale(jp)
            self.save(jp, version, "r2", kind="skip", text="пропущено")
            self.press(jp, version, REPLACE, "r4")
            rows = self.outcomes(self.apply(jp, md))
            self.assertEqual(rows["r2"]["outcome"], "skipped")
            self.assertNotIn("detail", rows["r2"])                              # КН-5
            self.assertEqual(rows["r4"]["outcome"], "apply_conflict")
            self.assertIn(review_pack.SOURCE_UNAVAILABLE, rows["r4"]["detail"])
            self.assertIn(review_pack.NEXT_STEP_OPEN, rows["r4"]["detail"])      # место открыто
            self.assertNotIn("detail", review_pack.answer_states(jp)["r2"])

    def test_preview_guard_says_what_apply_says(self):
        with files("sqlite", qs=only("r2")) as (jp, md, version):
            self.assertIsNone(review_pack.answer_guard(jp, md, "r2", {"kind": "free", "text": "вот"}))
            outside = {"kind": "free", "text": "вот другое та"}              # меняет соседнее «дальше»
            why = review_pack.answer_guard(jp, md, "r2", outside)
            self.assertEqual(why, review_pack.FREE_OUTSIDE)
            self.save(jp, version, "r2", **outside)
            row = self.outcomes(core.apply_answers_files(jp, md))["r2"]
            self.assertEqual(row["outcome"], "apply_conflict")
            self.assertIn(why, row["detail"])
        with files("sqlite", qs=only("r4")) as (jp, md, version):
            stale(jp)
            why = review_pack.answer_guard(jp, md, "r4", {"kind": "free", "text": "тот"})
            self.assertTrue(why.startswith(review_pack.SOURCE_UNAVAILABLE), why)
            self.press(jp, version, REPLACE, "r4")
            self.assertEqual(self.outcomes(self.apply(jp, md))["r4"]["detail"], why)


    def test_preview_endpoint_runs_guard_first(self):
        """КН-14: api/preview (review_page.preview_answer) сначала говорит то, что скажет «Применить»."""
        import review_page
        with files("sqlite", qs=only("r2")) as (jp, md, _version):
            self.assertNotIn("guard", review_page.preview_answer(jp, md, "r2", "вот"))
            self.assertEqual(review_page.preview_answer(jp, md, "r2", "вот другое та"),
                             {"cards": [], "guard": review_pack.FREE_OUTSIDE})
        with files("sqlite", qs=only("r4")) as (jp, md, _version):
            stale(jp)
            got = review_page.preview_answer(jp, md, "r4", "тот")
            self.assertEqual(got["cards"], [])
            self.assertTrue(got["guard"].startswith(review_pack.SOURCE_UNAVAILABLE), got)


class Texts(unittest.TestCase):
    ANSWER = {"id": "r1", "option": 0, "ts": "2026-11-26T10:00:00", "source": "local"}

    def states(self, answer, **store):
        return review_pack.answer_states_of(
            {"answers": {"r1": answer}, "applied_answers": {"r1": review_pack._sig(answer)}, **store},
            {"items": [{"id": "r1"}]})["r1"]

    def test_legacy_answer_without_snapshot(self):
        row = self.states(dict(self.ANSWER, origin="legacy_review_pack:demo"))
        self.assertIs(row["undo"], False)
        self.assertEqual(row["detail"], review_pack.UNDO_LEGACY.format(n=1))
        self.assertIn("«исправь место карточки 1»", row["detail"])

    def test_other_answer_without_snapshot(self):
        self.assertEqual(self.states(dict(self.ANSWER))["detail"], review_pack.UNDO_UNAVAILABLE.format(n=1))

    def test_old_source_detail_hidden_under_skip(self):
        answer = {"id": "r1", "kind": "skip", "ts": "2026-11-26T10:00:00", "source": "local"}
        row = review_pack.answer_states_of(
            {"answers": {"r1": answer}, "outcome_details": {"r1": {
                "ts": answer["ts"], "source": "local", "detail": ["недоступна исходная ревизия карточки"]}}},
            {"items": [{"id": "r1"}]})["r1"]
        self.assertEqual(row["state"], "skipped")
        self.assertNotIn("detail", row)

    def test_rebuild_refusal_without_snapshot_names_the_way_back(self):
        new = {"id": "r1", "option": 1, "ts": "2026-11-26T10:01:00", "source": "local"}
        store = {"answers": {"r1": new}, "applied_answers": {"r1": review_pack._sig(self.ANSWER)}}
        waiting = review_pack.rebuild_check(store, {"items": [{"id": "r1"}]})["waiting"]
        self.assertEqual(waiting, [{"card": 1, "id": "r1", "state": "pending_new", "undo": False}])
        self.assertIn(review_pack.REBUILD_EXIT_NO_UNDO, str(review_pack.RebuildRefused(waiting)))
        self.assertNotIn("Подготовить отмену", str(review_pack.RebuildRefused(waiting)))

    def test_texts_have_no_technical_words(self):
        """КН-9: тексты для человека без «ревизии», «снимка», «интервала», «пакета»."""
        texts = [review_pack.OUTCOME_REASONS[k] for k in ("applied_open", "already_decided", "apply_conflict",
                                                          "undo_conflict", "source_conflict", "listen_mismatch",
                                                          "left_for_human")]
        texts += [review_pack.UNDO_UNAVAILABLE, review_pack.UNDO_LEGACY, review_pack.SOURCE_UNAVAILABLE,
                  review_pack.WORDS_CHANGED, review_pack.FREE_OUTSIDE, review_pack.LISTEN_STALE,
                  review_pack.NEXT_STEP_OPEN, review_pack.NEXT_STEP_CLOSED, review_pack.REBUILD_EXIT_NO_UNDO]
        for text in texts:
            for word in ("ревизи", "снимк", "интервал", "пакет", "редакци"):
                self.assertNotIn(word, text.lower(), text)


def page(body):
    """Начало сценария cards.html и функции до card() — в Node. Документ — заглушка, элементы — простые объекты."""
    html = (six.PROJECT / "cards.html").read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1]
    head, rest = script.split("  // ответы этой версии вопросов", 1)
    middle = "  // ответы этой версии вопросов" + rest.split("  function card(it)", 1)[0]
    stubs = ('global.location={hostname:"127.0.0.1"};\n'
             'const el = (tag) => ({tag, children:[], dataset:{}, style:{}, textContent:"", attrs:{},\n'
             '  append(...xs){ this.children.push(...xs); }, setAttribute(k, v){ this.attrs[k] = v; }});\n'
             'global.document={getElementById:() => null, querySelector:() => null, createElement:el};\n'
             'const textOf = (n) => typeof n === "string" ? n : [n.textContent, ...(n.children || []).map(textOf)].join(" ");\n'
             'const find = (n, f) => f(n) ? n : (n.children || []).map(x => find(x, f)).find(Boolean);\n')
    wrapped = "(async () => {\n" + body + "\n})().catch(e => { console.error(e && e.message || e); process.exit(1); });\n"
    return subprocess.run([shutil.which("node")], input=stubs + head + middle + wrapped + "})();\n",
                          capture_output=True, text=True, encoding="utf-8")


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class Page(unittest.TestCase):
    """Страница (раздел 3.3 плана): перечни состояний, вид 9.3, предпросмотр, число на «Применить»."""

    def run_page(self, body):
        result = page(body)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_needs_answer_and_status(self):
        """В-4: «оставлено вам» — без ответа; карточка 9.3 — без ответа, пока её не оставили как есть."""
        self.run_page(r'''
          const st = (s) => ({state: s, ts: "t", source: "local"});
          man = {items: ["r1", "r2", "r3", "r4"].map(id => ({id, options: []})),
                 answer_states: {r1: st("left_for_human"), r2: st("saved"), r3: st("skipped"), r4: st("applied")},
                 card_conditions: {r2: {cannot_apply: true}, r3: {cannot_apply: true}, r4: {cannot_apply: true}}};
          answers = {r1: {id: "r1", ts: "t", source: "local", kind: "human"},
                     r2: {id: "r2", ts: "t", source: "local", option: 0},
                     r3: {id: "r3", ts: "t", source: "local", kind: "skip", text: "пропущено"},
                     r4: {id: "r4", ts: "t", source: "local", option: 0}};
          const open = man.items.filter(needsAnswer).map(it => it.id).join();
          if (open !== "r1,r2") throw Error(open);
          if (cardStatus(man.items[0], answers.r1) !== "оставлено вам — выберите ответ") throw Error(cardStatus(man.items[0], answers.r1));
          if (cardStatus(man.items[1], answers.r2) !== STALE_SAVED) throw Error(cardStatus(man.items[1], answers.r2));
          if (cannotApply(man.items[3], answers.r4)) throw Error("у внесённого ответа — обычный вид");
          if (STATUS.already_decided !== "место решено раньше — этот ответ не вносился") throw Error(STATUS.already_decided);
          if (waitingCount() !== 1) throw Error("ждут «Применить»: " + waitingCount());
        ''')

    def test_stale_view(self):
        """9.3: главная кнопка «Оставить как есть» = «Пропустить»; шаг для Claude — по состоянию места, с номером."""
        self.run_page(r'''
          const it = {id: "r2", question: "Что сказано?", options: [{label: "тот"}, {label: "этот"}]};
          man = {items: [{id: "r1"}, it], answer_states: {}, card_conditions: {r2: {cannot_apply: true, place_open: false}}};
          const sent = []; send = (card, a) => sent.push(a);
          let box = staleView(it, null, {});
          const text = textOf(box);
          if (!text.includes(STALE_TITLE) || !text.includes(STALE_PLACE.closed)) throw Error(text);
          if (!text.includes("«переоткрой место карточки 2 и дособери карточки»")) throw Error(text);
          if (!text.includes("Что сказано?") || !text.includes("этот")) throw Error("прежний вопрос не свёрнут в текст");
          const keep = find(box, n => n.tag === "button");
          if (keep.textContent !== "Оставить как есть" || keep.className !== "btn primary") throw Error(keep.textContent);
          keep.onclick();
          if (JSON.stringify(sent) !== '[{"kind":"skip","text":"пропущено"}]') throw Error(JSON.stringify(sent));
          man.card_conditions.r2.place_open = true;
          box = staleView(it, {kind: "skip", ts: "t", source: "local"}, {});
          if (!textOf(box).includes(STALE_STEP.open)) throw Error(textOf(box));
          if (find(box, n => n.tag === "button").attrs["aria-pressed"] !== "true") throw Error("пропуск не отмечен");
        ''')

    def test_preview_text(self):
        """КН-14: сначала проверка программы; хвост «может измениться» — только при других сохранённых ответах."""
        self.run_page(r'''
          man = {items: [{id: "r1"}, {id: "r2"}], answer_states: {}};
          answers = {};
          const it = man.items[0];
          if (previewText(it, {guard: "причина"}) !== "Этот ответ не внесётся: причина") throw Error("guard");
          if (!previewText(it, {preflight_conflicts: [{}]}).startsWith("Этот ответ не внесётся: он затрагивает")) throw Error("preflight");
          let t = previewText(it, {cards: [{was_now: "было → стало"}], not_found: ["r1"]});
          if (t !== "Предпросмотр (без записи): было → стало; программа не нашла в нынешнем тексте слова этой карточки.") throw Error(t);
          t = previewText(it, {cards: [{was_now: "Анна: восьмое → Анна: восьмое", applied: ["«восьмое» → «восьмое слово»"]}]});
          if (t !== "Предпросмотр (без записи): «восьмое» → «восьмое слово».") throw Error(t);
          // то же в итоге «Применить»
          const lines = reportLines({free_text: {cards: [{id: "r1", was_now: "Анна: восьмое → Анна: восьмое",
                                                          applied: ["«восьмое» → «восьмое слово»"]}]}}, man.items);
          if (lines.join() !== "Карточка 1: «восьмое» → «восьмое слово»") throw Error(lines.join());
          answers.r2 = {id: "r2", ts: "t", source: "local", option: 0};
          t = previewText(it, {cards: [], left_to_human: ["r1"]});
          if (!t.includes("программа не внесёт это сама: место останется вам") || !t.endsWith("другие сохранённые ответы.")) throw Error(t);
        ''')

    def test_apply_label_counts_waiting_answers(self):
        """КН-10: на «Применить» — число ответов, ждущих применения; нет таких — без числа."""
        self.run_page(r'''
          const st = (s) => ({state: s, ts: "t", source: "local"});
          man = {items: ["r1", "r2", "r3"].map(id => ({id, options: []})),
                 answer_states: {r1: st("saved"), r2: st("pending_new"), r3: st("applied")}};
          answers = {r1: {id: "r1", ts: "t", source: "local", option: 0}, r2: {id: "r2", ts: "t", source: "local", option: 1},
                     r3: {id: "r3", ts: "t", source: "local", option: 0}};
          if (applyLabel() !== "Применить сохранённые ответы (2)") throw Error(applyLabel());
          man.answer_states.r1 = st("applied"); man.answer_states.r2 = st("applied");
          if (applyLabel() !== "Применить сохранённые ответы") throw Error(applyLabel());
        ''')

    def test_applied_answer_buttons(self):
        """В-1: у внесённого ответа без снимка нет «Изменить ответ» и отмены; у нового поверх него — «Вернуть как было»."""
        self.run_page(r'''
          const it = {id: "r1", options: [{label: "x"}, {label: "y"}]}, a = {id: "r1", option: 1, ts: "t", source: "local"};
          const st = (s, extra) => ({r1: {state: s, ts: "t", source: "local", ...extra}});
          const acts = () => appliedActions(it, a).join();
          man = {items: [it], answer_states: st("applied"), applied_answers: {r1: ["option", 0, null]}};
          if (acts() !== "edit,revert") throw Error(acts());
          man.answer_states = st("applied", {undo: false});
          if (acts() !== "") throw Error(acts());
          man.answer_states = st("pending_new", {undo: false});
          if (acts() !== "back") throw Error(acts());
          man.answer_states = st("applied_check");
          if (acts() !== "confirm,edit,revert") throw Error(acts());
        ''')

    def test_disabled_button_style(self):
        """КН-12: неактивная кнопка бледнее и без подсветки при наведении."""
        html = (six.PROJECT / "cards.html").read_text(encoding="utf-8")
        self.assertIn(".btn:disabled{opacity:.5;cursor:not-allowed}", html)
        self.assertIn(".btn:hover:not(:disabled){", html)
        self.assertNotIn(".btn:hover{", html)
        self.assertNotIn("Отменить нельзя", html)


if __name__ == "__main__":
    unittest.main()
