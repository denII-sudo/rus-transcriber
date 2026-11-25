"""РАР-6.1, раздел 10 плана исправлений v2.3 (приватный репозиторий: research/2026-11-25-cards-scenarios):
кнопка «Я прослушал…» принимается, только если весь отрезок места хотя бы раз проигран — частями можно,
перемотка не в счёт. Прослушанное хранится отрезками записи (heard) и переживает перезагрузку, пересборку пакета
и замену карточек; облачной отметке played доверяем, старый облачный ответ без неё — not_heard.

Только хранение записей в базе (sqlite): решение владельца «в доработках новых не учитывай файловую запись».
Страница проверяется функциями cards.html в Node с заглушкой документа (без браузера). Данные синтетические.
"""
import atexit
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-place-heard-data-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_pause_buttons_short_answers as six  # noqa: E402
from test_pause_buttons_short_answers import SILENCE, files, questions, result  # noqa: E402

core = six.core
recordings = six.recordings
review_pack = six.review_pack
transcript = six.transcript
PROJECT = six.PROJECT
SHA = "a" * 64
SESSION = "page-session-1"


def old_format_events(jp, events):
    """Разбор прежнего формата (до Х1): события карточек списком interaction_events в ревизии разбора."""
    store, binding = recordings.open_for_document(jp)
    try:
        current = store.get_current(binding["recording_id"])
        review = copy.deepcopy(current["review"]["payload"])
        review["interaction_events"] = copy.deepcopy(events)
        store.commit_change(binding["recording_id"],
                            {"source_revision": current["transcript"]["revision"],
                             "review_revision": current["review"]["revision"]},
                            str(uuid.uuid4()), recordings.operation_context(store, "system"), review=review)
    finally:
        store.close()


def place(t1=1.0, t2=2.0, aid="r1"):
    return {"id": aid, "t1": t1, "t2": t2}


class Spans(unittest.TestCase):
    def test_pieces_count_and_skipped_part_does_not(self):
        """Частями — прослушано; пропуск больше допуска 0,2 с — нет; отрезки сливаются; другой звук не в счёт."""
        man = {"version": "v", "audio_sha256": "A", "items": [place()]}
        self.assertTrue(review_pack.place_heard({"heard": {"A": [[0.9, 1.4], [1.55, 2.1]]}}, man, place()))
        self.assertFalse(review_pack.place_heard({"heard": {"A": [[0.9, 1.3], [1.6, 2.1]]}}, man, place()))
        self.assertTrue(review_pack.place_heard({"heard": {"A": [[1.15, 1.9]]}}, man, place()))
        self.assertFalse(review_pack.place_heard({"heard": {"A": [[1.3, 2.0]]}}, man, place()))
        self.assertFalse(review_pack.place_heard({"heard": {"B": [[0.0, 5.0]]}}, man, place()))
        store = {"heard": {"A": []}}
        review_pack._note_heard(store, man, place(1.0, 2.0))
        review_pack._note_heard(store, man, place(1.5, 3.0))
        review_pack._note_heard(store, man, place(5.0, 6.0))
        self.assertEqual(store["heard"], {"A": [[1.0, 3.0], [5.0, 6.0]]})

    def test_old_audio_ended_counts_once(self):
        """10.6: без heard место засчитывает прежнее audio_ended этой редакции; после переноса — только heard."""
        man = {"version": "v", "audio_sha256": "A", "items": [place(), place(3.0, 4.0, "r2")]}
        store = {"interaction_events": [{"id": "r1", "type": "audio_ended", "version": "v"},
                                        {"id": "r2", "type": "audio_ended", "version": "old"}]}
        self.assertTrue(review_pack.place_heard(store, man, place()))
        self.assertFalse(review_pack.place_heard(store, man, place(3.0, 4.0, "r2")), "событие прежней версии")
        review_pack._heard_store(store, man)
        self.assertEqual(store["heard"], {"A": [[1.0, 2.0]]})
        store["interaction_events"].append({"id": "r2", "type": "audio_ended", "version": "v"})
        self.assertFalse(review_pack.place_heard(store, man, place(3.0, 4.0, "r2")),
                         "доигрывание до конца после переноса засчиталось")

    def test_rebuild_carries_heard_and_keeps_other_audio_apart(self):
        """Пересборка: прежнее heard переходит; пакет до раздела 10 (без отпечатка) — та же запись; другой звук — нет."""
        old = {"manifest": {"version": "v", "items": []}, "heard": {"": [[1.0, 2.0]]}}
        self.assertEqual(review_pack._heard_carried(old, {"audio_sha256": "A"}), {"A": [[1.0, 2.0]]})
        other = {"manifest": {"version": "v", "audio_sha256": "B", "items": []}, "heard": {"B": [[1.0, 2.0]]}}
        carried = review_pack._heard_carried(other, {"audio_sha256": "A"})
        self.assertEqual(carried, {"B": [[1.0, 2.0]], "A": []})
        self.assertFalse(review_pack.place_heard({"heard": carried}, {"audio_sha256": "A"}, place()))

    def test_not_heard_texts(self):
        why = review_pack.OUTCOME_REASONS["not_heard"]
        self.assertIn("не прослушано целиком", why)
        self.assertIn("перемотка не в счёт", why)
        self.assertEqual(review_pack.rebuild_group("not_heard"), "unapplicable")


class Apply(six.Case):
    def listen(self, jp, version, aid="r1", source="local", **row):
        item = next(i for i in review_pack.manifest(jp)["items"] if i["id"] == aid)
        n = [o["label"] for o in item["options"]].index(SILENCE)
        if source == "local":
            return self.save(jp, version, aid, option=n, **row)
        six.Case._ts += 1
        rep = review_pack.add_answers(jp, [{"id": aid, "version": version, "option": n,
                                            "ts": f"2026-11-25T12:00:{six.Case._ts % 60:02d}", **row}], "cloud")
        # облачный ответ применяется после подтверждения человека — как confirm_review_answers
        with recordings.operation_for_document(jp, "human"):
            state = recordings.read_review_pack_state(jp)
            state["answers"][aid]["confirmation_context"] = recordings.context_for_document(jp)
            review_pack._write_answers(jp, state)
        return rep

    def outcome(self, jp, md):
        rep = core.apply_answers_files(jp, md)
        return self.outcomes(rep)["r1"]["outcome"], rep

    @staticmethod
    def place_open(jp):
        return transcript.place_unresolved(next(r for r in transcript.load(jp)["review"] if r["id"] == "r1"))

    def test_listen_button_needs_the_place_played(self):
        """Кнопка без прослушивания — not_heard, место открыто; после place_heard — применена."""
        with files("sqlite") as (jp, md, version):
            self.listen(jp, version, played=False)
            outcome, rep = self.outcome(jp, md)
            self.assertEqual(outcome, "not_heard", rep.get("outcomes"))
            self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "not_heard")
            self.assertTrue(self.place_open(jp))
            self.assertEqual(review_pack.heard_ids(review_pack.answers_state(jp), review_pack.manifest(jp)), [])
            stored = review_pack.add_interaction_event(jp, "r1", version, "place_heard", SESSION)
            self.assertTrue(stored["stored"])
            outcome, rep = self.outcome(jp, md)
            self.assertEqual(outcome, "applied", rep.get("outcomes"))
            self.assertFalse(self.place_open(jp))
            # «Применить» (запись стенограммы вместе с пакетом) прослушанное не снимает
            self.assertEqual(recordings.read_review_pack_state(jp)["heard"], {"": [[1.0, 2.0]]})

    def test_heard_survives_reload_upsert_and_rebuild(self):
        """heard в базе: перечитывается, остаётся после замены карточки и пересборки пакета; кнопка — сразу."""
        data = result()
        data["audio_sha256"] = SHA
        with files("sqlite", data) as (jp, md, version):
            self.assertEqual(review_pack.manifest(jp)["audio_sha256"], SHA)
            review_pack.add_interaction_event(jp, "r1", version, "place_heard", SESSION)
            self.assertEqual(recordings.read_review_pack_state(jp)["heard"], {SHA: [[1.0, 2.0]]})
            qs = questions()
            qs[0]["question"] = "Что в паузе? (уточнено)"
            rep = review_pack.upsert_cards(jp, [qs[0]], why="тест")
            self.assertEqual([r["id"] for r in rep["replaced"]], ["r1"])
            # r2 (2,0 с) лежит внутри прослушанного отрезка паузы r1 — тот же звук засчитывается соседней карточке
            self.assertEqual(review_pack.heard_ids(review_pack.answers_state(jp), review_pack.manifest(jp)),
                             ["r1", "r2"])
            review_pack.build(jp, copy.deepcopy(questions()))
            new = review_pack.manifest(jp)["version"]
            self.assertNotEqual(new, version)
            self.assertEqual(recordings.read_review_pack_state(jp)["heard"], {SHA: [[1.0, 2.0]]})
            self.listen(jp, new, rev=review_pack.card_rev(review_pack.manifest(jp)["items"][0]), played=False)
            outcome, rep = self.outcome(jp, md)
            self.assertEqual(outcome, "applied", rep.get("outcomes"))

    def test_cloud_mark_is_trusted_and_old_cloud_answer_is_not(self):
        """Облако: played — место прослушано (доверяем); ответ без отметки (страница до обновления) — not_heard."""
        with files("sqlite") as (jp, md, version):
            self.listen(jp, version, source="cloud")
            outcome, rep = self.outcome(jp, md)
            self.assertEqual(outcome, "not_heard", rep.get("outcomes"))
        with files("sqlite") as (jp, md, version):
            self.listen(jp, version, source="cloud", played=True)
            self.assertEqual(recordings.read_review_pack_state(jp)["heard"], {"": [[1.0, 2.0]]})
            outcome, rep = self.outcome(jp, md)
            self.assertEqual(outcome, "applied", rep.get("outcomes"))

    def test_cloud_bundle_has_heard_ids_and_no_audio_fingerprint(self):
        data = result()
        data["audio_sha256"] = SHA
        with files("sqlite", data) as (jp, _md, version):
            review_pack.add_interaction_event(jp, "r1", version, "place_heard", SESSION)
            bundle = review_pack.cloud_bundle(jp)
            cloud = json.loads((Path(bundle["base"]) / bundle["files"]["manifest.json"]).read_text(encoding="utf-8"))
            self.assertEqual(cloud["heard_ids"], ["r1", "r2"])   # r2 (2,0 с) — внутри отрезка r1
            self.assertNotIn("audio_sha256", cloud)

    def test_old_pack_audio_ended_still_counts(self):
        """10.6: пакет до обновления, место доиграно до конца раньше — кнопка применяется без нового прослушивания."""
        # пакет собран до раздела 10: heard в состоянии нет
        with mock.patch.object(review_pack, "_heard_carried", return_value=None),                 files("sqlite") as (jp, md, version):
            state = recordings.read_review_pack_state(jp)
            self.assertNotIn("heard", state)
            state["manifest"].pop("audio_sha256", None)
            review_pack._write_answers(jp, state)
            # база до Х1: события лежат списком в разборе; состояние пакета их больше не пишет — прямо в снимок
            old_format_events(jp, [{"id": "r1", "version": version, "type": "audio_ended",
                                    "session": SESSION, "source": "local", "ts": "2026-11-20T10:00:00"}])
            raw = recordings.read_review_pack_state(jp)
            self.assertNotIn("heard", raw)
            self.listen(jp, version, played=False)
            self.assertEqual(recordings.read_review_pack_state(jp).get("heard"), None)
            outcome, rep = self.outcome(jp, md)
            self.assertEqual(outcome, "applied", rep.get("outcomes"))
            # первое событие страницы после обновления переносит прежнее audio_ended в heard (10.6)
            review_pack.add_interaction_event(jp, "r2", version, "card_visible", SESSION)
            self.assertEqual(recordings.read_review_pack_state(jp)["heard"], {"": [[1.0, 2.0]]})


def card_page(body):
    """Начало сценария cards.html и функции до render (с card) — в Node, с маленькой заглушкой документа."""
    html = (PROJECT / "cards.html").read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1]
    head = script.split("  function render()", 1)[0]
    dom = r'''
global.location = {hostname: "127.0.0.1"}; global.window = {};
class El {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.style = {};
    this.attrs = {}; this.on = {}; this.textContent = ""; this.className = ""; this.disabled = false; this.title = ""; }
  append(...xs) { xs.forEach(x => { if (typeof x === "string") x = Object.assign(new El("#text"), {textContent: x});
    if (x.tagName === "#FRAGMENT") { x.children.forEach(y => this.append(y)); return; }
    x.parentElement = this; this.children.push(x); }); }
  appendChild(x) { this.append(x); return x; }
  remove() { const p = this.parentElement; if (p) p.children = p.children.filter(c => c !== this); }
  replaceChildren(...xs) { this.children = []; this.append(...xs); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  addEventListener(t, f) { (this.on[t] = this.on[t] || []).push(f); }
  fire(t) { (this.on[t] || []).forEach(f => f()); }
  set innerHTML(h) { this.children = []; for (const m of h.matchAll(/<span class="([^"]+)"><\/span>/g))
    this.append(Object.assign(new El("span"), {className: m[1]})); }
  get classList() { const el = this; return {add(c) { el.className = (el.className + " " + c).trim(); },
    remove(c) { el.className = el.className.split(" ").filter(x => x !== c).join(" "); }}; }
  all() { return this.children.flatMap(c => [c, ...c.all()]); }
  matches(sel) { const m = /^([a-z]+)?((?:\.[\w-]+)*)(?:\[data-([\w-]+)\])?$/.exec(sel); if (!m) return false;
    const cls = this.className.split(" ");
    return (!m[1] || this.tagName === m[1].toUpperCase()) && m[2].split(".").filter(Boolean).every(c => cls.includes(c))
      && (!m[3] || m[3] in this.dataset); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  querySelectorAll(sel) { const parts = sel.split(" "); const last = parts.pop();
    return this.all().filter(e => e.matches(last)); }
}
global.document = {createElement: t => new El(t), createDocumentFragment: () => new El("#fragment"),
  createTextNode: t => Object.assign(new El("#text"), {textContent: t}), getElementById: () => null,
  querySelector: () => null};
'''
    wrapped = "(async () => {\n" + body + "\n})().catch(e => { console.error(e && e.stack || e); process.exit(1); });\n"
    return subprocess.run([shutil.which("node")], input=dom + head + wrapped + "})();\n",
                          capture_output=True, text=True, encoding="utf-8")


ITEM = {"id": "r1", "rev": 1, "t": "00:00:01", "t1": 1.0, "t2": 2.0, "clip": "clips/r1.r1.mp3",
        "clip_t1": 3.5, "clip_t2": 4.5, "speaker": "Анна", "text": "", "reply": "Анна: начало",
        "question": "Что в паузе?", "reasons": ["missed"], "recommended": None,
        "options": [{"label": "Прослушал: слов нет", "decision": {"listened": "silence"}},
                    {"label": "Вписать", "decision": {"text": "ну"}}]}


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class Page(unittest.TestCase):
    def run_page(self, body, item=ITEM, heard_ids=()):
        setup = ("const events = [], sent = [];\n"
                 "api = {voiceMode: null, async event(e) { events.push(e); return {}; },"
                 " async answer(a) { sent.push(a); return {}; }};\n"
                 "man = {version: 'v1', items: [%s], heard_ids: %s};\n"
                 "const it = man.items[0]; let c = card(it);\n"
                 "const listen = () => c.querySelectorAll('button.opt').filter(b => 'listen' in b.dataset);\n"
                 "const hint = () => (c.querySelector('.listen-hint') || {}).textContent || '';\n"
                 "const play = (...r) => { const au = c.querySelector('audio');"
                 " au.played = {length: r.length, start: i => r[i][0], end: i => r[i][1]}; au.fire('timeupdate'); };\n"
                 % (json.dumps(item, ensure_ascii=False), json.dumps(list(heard_ids))))
        out = card_page(setup + body)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout

    def test_buttons_open_after_the_whole_place_is_played(self):
        """Кнопка неактивна до прослушивания; частями — да, с пропуском — нет; событие одно; ответ с played."""
        self.run_page(r'''
if (listen().length !== 1 || !listen()[0].disabled) throw Error("кнопка прослушивания активна до прослушивания");
if (!hint().startsWith("Сначала прослушайте место целиком")) throw Error("нет подсказки: " + hint());
if (c.querySelectorAll("button.opt").filter(b => !b.disabled).length !== 1) throw Error("обычный вариант закрыт");
play([0, 2.0], [3.0, 3.9]);       // до места и начало места
play([4.2, 5.0]);                 // перемотка через 3,9–4,2: больше допуска
if (!listen()[0].disabled || events.some(e => e.type === "place_heard")) throw Error("пропуск засчитан");
play([3.9, 4.15]);                // дослушан кусок, который пропускали
if (listen()[0].disabled) throw Error("кнопка не открылась после прослушивания");
if (hint()) throw Error("подсказка осталась");
play([0, 6]);
if (events.filter(e => e.type === "place_heard").length !== 1) throw Error("событий place_heard: " + JSON.stringify(events));
listen()[0].onclick(); await new Promise(r => setTimeout(r, 0));
if (!sent.length || sent[0].played !== true || sent[0].option !== 0) throw Error("ответ без played: " + JSON.stringify(sent));
c = card(it);                     // новая отрисовка: прослушанное в этой сессии помнится
if (listen()[0].disabled) throw Error("после перерисовки кнопка снова закрыта");
''')

    def test_heard_before_opens_buttons_and_no_clip_hides_them(self):
        """heard_ids от программы — кнопка сразу; без клипа и без прослушанного — кнопок нет, подсказка про звук."""
        self.run_page(r'''
if (listen().length !== 1 || listen()[0].disabled || hint()) throw Error("прослушанное раньше не открыло кнопку");
listen()[0].onclick(); await new Promise(r => setTimeout(r, 0));
if ("played" in sent[0]) throw Error("played без прослушивания на этой странице");
''', heard_ids=["r1"])
        no_clip = dict(ITEM, clip=None, clip_t1=None, clip_t2=None)
        self.run_page(r'''
if (listen().length) throw Error("кнопка прослушивания без звука");
if (hint() !== NO_CLIP_LISTEN) throw Error("подсказка: " + hint());
if (c.querySelectorAll("button.opt").length !== 1) throw Error("обычный вариант пропал");
''', item=no_clip)

    def test_not_heard_needs_an_answer(self):
        self.run_page(r'''
if (!NEEDS_ANSWER.includes("not_heard") || !STATUS.not_heard) throw Error("not_heard не ждёт ответа");
''')


if __name__ == "__main__":
    unittest.main()
