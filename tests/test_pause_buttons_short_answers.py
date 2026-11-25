"""РАР-6.1, PR 6б: кнопки паузы (В7) и короткие ответы по таблице слов (В8).

Пропуск («missed» в reasons) закрывается кнопками «Прослушал: слов нет» и «Прослушал: только междометия,
не записывать»; «слов нет» проверяется по словам, начатым в паузе. Короткие ответы разбираются одной
таблицей answer_phrases.tsv и для текста, и для голоса. Все данные синтетические.
"""
import atexit
import contextlib
import copy
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-pause6b-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import core  # noqa: E402
import documents  # noqa: E402
import freeform  # noqa: E402
import recordings  # noqa: E402
import review_pack  # noqa: E402
import settings  # noqa: E402
import transcript  # noqa: E402

BACKENDS = ("legacy_files", "sqlite")
SILENCE = "Прослушал: слов нет"
INTERJECTIONS = "Прослушал: только междометия, не записывать"
WORDS = "Прослушал: слова в стенограмме верны"
MARK = "[междометия, не расшифровано]"


def word(wid, text, s, e=None):
    return {"word_id": wid, "w": text, "s": s, "e": s + 0.2 if e is None else e}


def result(inside=False, r2_word="вот", gap_key=True, extra_review=()):
    """Пауза r1 [1,0; 2,0) (хвост «начало» заходит в неё), слово r2 (2,0), «модели расходятся» r3 (2,5),
    слово r4 «та» (3,0) с рекомендацией."""
    words = [word("w1", "начало", 0.5, 1.05), word("w2", r2_word, 2.0, 2.2), word("w3", "дальше", 2.5, 2.7),
             word("w5", "та", 3.0, 3.2)]
    if inside:
        words.insert(1, word("w4", "ага", 1.4, 1.6))
    r1 = {"id": "r1", "status": "open", "t": 1.0, "t1": 1.0, "t2": 2.0, "speaker": "Анна",
          "reasons": ["missed"], "text": ""}
    if gap_key:
        r1["gap"] = [1.0, 2.0]
    review = [r1,
              {"id": "r2", "status": "open", "t": 2.0, "t1": 2.0, "t2": 2.0, "speaker": "Анна",
               "reasons": ["conf"], "text": r2_word, "word_ids": ["w2"]},
              {"id": "r3", "status": "open", "t": 2.5, "t1": 2.5, "t2": 2.5, "speaker": "Анна",
               "reasons": ["asr"], "text": "дальше", "word_ids": ["w3"]},
              {"id": "r4", "status": "open", "t": 3.0, "t1": 3.0, "t2": 3.0, "speaker": "Анна",
               "reasons": ["conf"], "text": "та", "word_ids": ["w5"]}] + list(extra_review)
    return {
        "file": "synthetic.wav", "duration": "00:00:04", "speakers": ["Анна"], "speaker_labels": ["speaker-1"],
        "timing": {"total_sec": 4}, "group": "demo", "quality": {"asr_incomplete_chunks": 0},
        "vocab_suggestions": [], "review_done": False, "edits": [],
        "segments": [{"start": 0.5, "end": 3.2, "label": "speaker-1", "speaker": "Анна",
                      "text": " ".join(w["w"] for w in words), "words": words}],
        "review": review}


def played_for_listen(jp, aid, row):
    """Раздел 10: кнопка «Я прослушал…» принимается после прослушивания места. Тесты прежних правил отвечают, как
    страница после проигрывания всего отрезка места (played); явное played в ответе не трогается."""
    if row.get("option") is None or "played" in row:
        return row
    item = next((i for i in (review_pack.manifest(jp) or {}).get("items") or [] if i["id"] == aid), None)
    opts = (item or {}).get("options") or []
    listen = 0 <= row["option"] < len(opts) and (opts[row["option"]].get("decision") or {}).get("listened")
    return dict(row, played=True) if listen else row


def two_replies():
    """Те же слова двумя репликами: «начало» до паузы, остальное — после неё."""
    data = result()
    seg = data["segments"][0]
    first, second = dict(seg, words=seg["words"][:1], start=0.5, end=1.05), \
        dict(seg, words=seg["words"][1:], start=2.0, end=3.2)
    first["text"], second["text"] = "начало", "вот дальше та"
    # другой говорящий: соседние реплики одного говорящего сохранение сливает в одну
    second.update(speaker="Борис", label="speaker-2")
    data["segments"] = [first, second]
    data["speakers"], data["speaker_labels"] = ["Анна", "Борис"], ["speaker-1", "speaker-2"]
    return data


AS_IS = {"label": "Оставить как есть", "decision": {}}


def questions():
    return [{"id": "r1", "question": "Что в паузе?", "recommended": 0,
             "options": [AS_IS, {"label": "Вписать «ну»", "decision": {"text": "ну"}}]},
            {"id": "r2", "question": "Проверьте", "recommended": 0, "options": [AS_IS]},
            {"id": "r3", "question": "Модели расходятся", "recommended": 1,
             "options": [AS_IS, {"label": "«дальнейшем»", "decision": {"text": "дальнейшем"}}]},
            {"id": "r4", "question": "Проверьте", "recommended": 1,
             "options": [AS_IS, {"label": "Заменить на «тот»", "decision": {"text": "тот"}}]}]


@contextlib.contextmanager
def files(backend, data=None, qs=None):
    """Пара файлов стенограммы в выбранном режиме хранения и собранный пакет: (jp, md, версия)."""
    with tempfile.TemporaryDirectory(dir=_TMP) as temp:
        root = Path(temp)
        db, audio = root / "state.sqlite3", root / "sample.wav"
        audio.write_bytes(b"synthetic")
        data = copy.deepcopy(data or result())
        data["file"] = str(audio)
        layout = recordings.document_layout(audio)
        layout["root"].mkdir()
        store = None
        with mock.patch.object(settings.paths, "STATE_DB", db), \
                mock.patch.object(review_pack, "cut_clip", side_effect=lambda _s, t1, _t2, _out: t1):
            try:
                if backend == "sqlite":
                    store = recordings.initialize_store(db, display_name="Тест")
                    root_id, targets = recordings.prepare_new_layout(store, layout["root"], layout["paths"])
                    rid = str(uuid.uuid4())
                    recordings.commit_worker_result(
                        store, data, audio, root_id, recordings.operation_context(store, "model"),
                        recording_id=rid, operation_id=str(uuid.uuid4()),
                        generated_at="2026-11-16T00:00:00Z", document_paths=targets)
                    recordings.materialize_recording(store, rid)
                    jp, md = layout["root"] / targets["json"], layout["root"] / targets["md"]
                else:
                    jp, md = layout["root"] / layout["paths"]["json"], layout["root"] / layout["paths"]["md"]
                    transcript.save(data, jp, md)
                pack = review_pack.build(jp, copy.deepcopy(questions() if qs is None else qs))
                yield jp, md, pack.get("version")
            finally:
                if store:
                    store.close()


def old_pack(jp, change):
    """Пакет, собранный до 6б: change(элемент r1) правит его и в состоянии ответов, и в manifest.json файлов."""
    state = review_pack.answers_state(jp)
    change(next(i for i in state["manifest"]["items"] if i["id"] == "r1"))
    review_pack._write_answers(jp, state)
    path = review_pack.pack_dir(jp) / "manifest.json"
    if recordings.read_review_pack_state(jp) is None and path.exists():
        man = review_pack._read(path, {})
        change(next(i for i in man["items"] if i["id"] == "r1"))
        review_pack._write(path, man)


class Case(unittest.TestCase):
    _ts = 0

    def save(self, jp, version, aid="r1", **row):
        Case._ts += 1
        answer = {"id": aid, "version": version, "ts": f"2026-11-24T10:{Case._ts // 60:02d}:"
                  f"{Case._ts % 60:02d}", **played_for_listen(jp, aid, row)}
        return review_pack.add_answers(jp, [answer], "local", trusted_actor="human")

    def press(self, jp, version, label, aid="r1"):
        item = next(i for i in review_pack.manifest(jp)["items"] if i["id"] == aid)
        n = [o["label"] for o in item["options"]].index(label)
        return self.save(jp, version, aid, option=n)

    @staticmethod
    def flat(jp):
        return [w for s in transcript.load(jp)["segments"] for w in s["words"]]

    def texts(self, jp):
        return [w["w"] for w in self.flat(jp)]

    def word(self, jp, wid):
        return next(w for w in self.flat(jp) if w.get("word_id") == wid)

    @staticmethod
    def place_of(jp, rid):
        return next(i for i in transcript.load(jp)["review"] if i["id"] == rid)

    @staticmethod
    def outcomes(rep):
        return {row["id"]: row for row in rep.get("outcomes") or []}

    def apply(self, jp, md):
        """Применить ответы как человек; повторный запуск ничего не меняет (тест 16)."""
        rep = core.apply_answers_files(jp, md)
        before = transcript.load(jp)
        again = core.apply_answers_files(jp, md)
        self.assertEqual(transcript.load(jp), before, "повторный запуск изменил стенограмму")
        self.assertEqual({k: v["outcome"] for k, v in self.outcomes(again).items()},
                         {k: v["outcome"] for k, v in self.outcomes(rep).items()})
        return rep


# ---------- 1.1. кнопки паузы ----------
class BuildButtonsTests(Case):
    def test_gap_places_get_the_two_buttons_and_agent_options_are_dropped(self):
        """Тест 1: пропуск ["missed"] и ["asr", "missed"] — кнопки В7; со словами внутри — ещё «слова верны»."""
        asr_missed = {"id": "r5", "status": "open", "t": 3.5, "t1": 3.5, "t2": 3.9, "gap": [3.5, 3.9],
                      "speaker": None, "text": "", "reasons": ["asr", "missed"],
                      "asr_problem": {"model": "whisper", "state": "empty_unknown", "words_from": []}}
        qs = questions() + [{"id": "r5", "options": [AS_IS]}]
        for inside in (False, True):
            with self.subTest(inside=inside), \
                    files("legacy_files", result(inside=inside, extra_review=[asr_missed]), qs) as (jp, _md, _v):
                items = {i["id"]: i for i in review_pack.manifest(jp)["items"]}
                want = ([WORDS] if inside else []) + [SILENCE, INTERJECTIONS]
                self.assertEqual([o["label"] for o in items["r1"]["options"]], want)
                self.assertEqual([o["label"] for o in items["r5"]["options"]], [SILENCE, INTERJECTIONS])
                self.assertEqual(items["r1"]["options"][-1]["decision"], {"listened": "interjections"})
                self.assertIsNone(items["r1"]["recommended"])
                self.assertIsNone(items["r5"]["recommended"])
                self.assertEqual(items["r4"]["recommended"], 1)
                self.assertTrue(all("listened" not in o["decision"] for o in items["r2"]["options"]))

    def test_asr_problem_without_missed_keeps_its_buttons(self):
        asr = {"id": "r5", "status": "open", "t": 3.0, "t1": 3.0, "t2": 3.2, "gap": [3.0, 3.2],
               "speaker": None, "text": "", "reasons": ["asr"],
               "asr_problem": {"model": "whisper", "state": "empty_unknown", "words_from": []}}
        # К-5 (К-а1): кнопки сбоя прохода сверяются по отпечатку слов своего интервала, как у паузы
        with files("sqlite", result(extra_review=[asr]), questions() + [{"id": "r5", "options": [AS_IS]}]) \
                as (jp, _md, _v):
            item = next(i for i in review_pack.manifest(jp)["items"] if i["id"] == "r5")
            self.assertEqual([o["decision"] for o in item["options"]],
                             [{"listened": "words"}, {"listened": "silence"}])
            self.assertEqual(item["listen_fp"], transcript._listening_fingerprint(
                transcript._interval_words(transcript.load(jp), 3.0, 3.2)))

    def test_listen_fp_is_the_fingerprint_of_overlapping_words(self):
        """Тест 2: отпечаток — по словам, перекрывающим интервал (их пересверяет сохранение)."""
        with files("legacy_files") as (jp, _md, _v):
            item = next(i for i in review_pack.manifest(jp)["items"] if i["id"] == "r1")
            data = transcript.load(jp)
            self.assertEqual(item["listen_fp"],
                             transcript._listening_fingerprint(transcript._interval_words(data, 1.0, 2.0)))
            self.assertNotEqual(item["listen_fp"], transcript._listening_fingerprint([]))

    def test_gap_place_without_a_good_gap_gets_no_buttons(self):
        for bad in (None, [2.0, 1.0], [1.0, float("nan")]):
            data = result(gap_key=False)
            if bad is not None:
                data["review"][0]["gap"] = bad
            with self.subTest(gap=bad), files("legacy_files", data) as (jp, _md, _v):
                item = next(i for i in review_pack.manifest(jp)["items"] if i["id"] == "r1")
                self.assertEqual([o["label"] for o in item["options"]], ["Оставить как есть", "Вписать «ну»"])
                self.assertNotIn("listen_fp", item)


class PauseButtonsApplyTests(Case):
    def test_silence_with_previous_word_tail_closes_and_survives_a_save(self):
        """Тест 3 (оба): хвост «начало» заходит в паузу — «слов нет» всё равно принимается."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.press(jp, version, SILENCE)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "applied", rep.get("outcomes"))
                item = self.place_of(jp, "r1")
                self.assertEqual((item["status"], item["listening"]["result"]), ("resolved", "silence"))
                transcript.edit_span(jp, md, 3.0, 3.2, "тот", by="human")
                item = self.place_of(jp, "r1")
                self.assertEqual(item["status"], "resolved")
                self.assertEqual(item["listening"]["result"], "silence")

    def test_buttons_contradicting_words_are_refused_with_pause_outcome(self):
        """Тест 4: «слов нет» и «только междометия» при слове в паузе; «слова верны» на пустой паузе."""
        cases = ((True, "silence"), (True, "interjections"), (False, "words"))
        for inside, listened in cases:
            with self.subTest(inside=inside, listened=listened), \
                    files("legacy_files", result(inside=inside)) as (jp, md, _v):
                before = transcript.load(jp)
                _, rep = transcript.apply_review(jp, md, [{"id": "r1", "listened": listened}], by="human")
                self.assertEqual(rep["successful_ids"], [])
                self.assertEqual(rep.get("pause_refused"), ["r1"])
                why = rep["conflicts"][0]["why"]
                self.assertIn("в паузе есть слова («ага»)" if inside else "противоречит словам", why)
                self.assertEqual(transcript.load(jp), before)
        # К-3 (К-а1): кнопка против слов паузы — свой исход listen_mismatch (новое — только в режиме базы)
        with files("sqlite", result(inside=True)) as (jp, md, version):
            self.press(jp, version, SILENCE)
            rep = core.apply_answers_files(jp, md)
            row = self.outcomes(rep)["r1"]
            self.assertEqual(row["outcome"], "listen_mismatch", rep.get("outcomes"))
            self.assertIn("в паузе есть слова", row.get("detail", ""))

    def test_words_button_on_pause_with_words_confirms_them(self):
        with files("legacy_files", result(inside=True)) as (jp, md, version):
            self.press(jp, version, WORDS)
            rep = self.apply(jp, md)
            self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "applied", rep.get("outcomes"))
            self.assertEqual(self.place_of(jp, "r1")["listening"]["result"], "words")

    def test_button_validity_follows_the_interval_not_the_revision(self):
        """Тест 5 (оба): правка в стороне — кнопка годна; правка слова в интервале — нет; старый пакет — строго."""
        cases = (("elsewhere", True, "applied"), ("interval", True, "source_conflict"),
                 ("elsewhere", False, "source_conflict"))
        for backend in BACKENDS:
            for where, new_pack, want in cases:
                with self.subTest(backend=backend, where=where, new_pack=new_pack), \
                        files(backend) as (jp, md, version):
                    if not new_pack:            # пакет до 6б: отпечатка интервала нет
                        old_pack(jp, lambda item: item.pop("listen_fp"))
                    self.press(jp, version, SILENCE)
                    if where == "elsewhere":
                        transcript.edit_span(jp, md, 3.0, 3.2, "тот", by="human")
                    else:
                        transcript.edit_span(jp, md, 0.5, 1.05, "начала", by="human")
                    rep = core.apply_answers_files(jp, md)
                    row = self.outcomes(rep)["r1"]
                    self.assertEqual(row["outcome"], want, rep.get("outcomes"))
                    if where == "interval":
                        self.assertIn(review_pack.LISTEN_STALE, row.get("detail", ""))   # К-7: без «паузы»
                        self.assertNotIn("listening", self.place_of(jp, "r1"))

    def test_agent_cannot_press_a_pause_button(self):
        """Тест 6."""
        with files("legacy_files") as (jp, md, _v):
            before = transcript.load(jp)
            for listened in ("silence", "interjections"):
                _, rep = transcript.apply_review(jp, md, [{"id": "r1", "listened": listened}], by="claude")
                self.assertEqual(rep["conflicts"][0]["why"], "нет явного подтверждения человека")
            self.assertEqual(transcript.load(jp), before)

    def test_interjections_mark_in_both_renderers(self):
        """Тест 7 (оба): пометка на месте паузы, слов не добавилось."""
        self.assertEqual(documents.RENDERER_VERSION, 3)
        data = two_replies()
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, data) as (jp, md, version):
                self.press(jp, version, INTERJECTIONS)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.texts(jp), ["начало", "вот", "дальше", "та"])
                text = md.read_text(encoding="utf-8")
                # sqlite: файл — документ documents.prepare (раздел «Текст»), а не render_md
                self.assertEqual("## Текст" in text, backend == "sqlite")
                self.assertEqual(text.count(MARK), 1, text)
                self.assertIn(f"_{MARK} 00:00:01–00:00:02_", text)
                # пометка — между репликой до паузы и репликой после неё
                self.assertLess(text.index("начало"), text.index(MARK))
                self.assertLess(text.index(MARK), text.index("вот дальше та"))
                short = transcript.render_md(transcript.load(jp), text_only=True)
                self.assertIn(MARK, short)

    def test_mark_goes_between_replies_by_time(self):
        data = two_replies()
        data["review"][0]["listening"] = {"by": "human", "result": "interjections"}
        lines = transcript.render_md(data, text_only=True).splitlines()
        at = [n for n, line in enumerate(lines) if MARK in line]
        self.assertEqual(len(at), 1)
        self.assertIn("начало", lines[at[0] - 2])
        self.assertIn("вот дальше та", lines[at[0] + 2])
        data["review"][0]["listening"]["result"] = "silence"
        self.assertNotIn(MARK, transcript.render_md(data))

    def test_layout_boundary_and_tail(self):
        """Правило места пометки: реплика с началом ровно в начале паузы — пометка перед ней (>=);
        позже паузы ни одна реплика не начинается — пометка в конце."""
        item = {"id": "r1", "gap": [1.0, 2.0], "reasons": ["missed"], "t1": 1.0, "t2": 2.0,
                "listening": {"by": "human", "result": "interjections"}}
        line = f"_{MARK} 00:00:01–00:00:02_"
        self.assertEqual(transcript.interjection_layout([{"start": 0.5}, {"start": 1.0}], [item]), ({1: [line]}, []))
        self.assertEqual(transcript.interjection_layout([{"start": 0.5}, {"start": 1.01}], [item]),
                         ({1: [line]}, []))
        self.assertEqual(transcript.interjection_layout([{"start": 0.5}], [item]), ({}, [line]))
        for result_ in ("words", "silence"):
            with self.subTest(result=result_):
                other = dict(item, listening={"by": "human", "result": result_})
                self.assertEqual(transcript.interjection_layout([{"start": 0.5}], [other]), ({}, []))

    def test_one_speaker_around_the_pause_gets_the_mark_after_the_reply(self):
        """Один говорящий до и после паузы: сохранение сливает реплики в одну, начатую до паузы, —
        пометка встаёт после этой реплики (хвостовое правило), в обоих отрисовщиках."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.press(jp, version, INTERJECTIONS)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(len(transcript.load(jp)["segments"]), 1)
                text = md.read_text(encoding="utf-8")
                self.assertEqual("## Текст" in text, backend == "sqlite")
                self.assertEqual(text.count(MARK), 1, text)
                short = transcript.render_md(transcript.load(jp), text_only=True)
                for shown in (text, short):
                    lines = shown.splitlines()
                    reply = [n for n, line in enumerate(lines) if "начало" in line and "дальше" in line]
                    mark = [n for n, line in enumerate(lines) if MARK in line]
                    self.assertEqual((len(reply), len(mark)), (1, 1), shown)
                    self.assertLess(reply[0], mark[0], shown)

    def test_words_and_silence_buttons_give_no_mark(self):
        """Пометку даёт только «только междометия»: «слова верны» и «слов нет» — без пометки."""
        for backend in BACKENDS:
            for label, data in ((WORDS, result(inside=True)), (SILENCE, two_replies())):
                with self.subTest(backend=backend, label=label), files(backend, data) as (jp, md, version):
                    self.press(jp, version, label)
                    rep = self.apply(jp, md)
                    self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "applied", rep.get("outcomes"))
                    self.assertEqual(self.place_of(jp, "r1")["status"], "resolved")
                    self.assertNotIn(MARK, md.read_text(encoding="utf-8"))
                    self.assertNotIn(MARK, transcript.render_md(transcript.load(jp), text_only=True))

    def test_listening_record_stays_on_this_computer(self):
        """В публичный разбор документа — только итог прослушивания; отпечаток интервала — не в облачный пакет."""
        review = {"review": [{"id": "r1", "status": "resolved", "t": 1.0, "t1": 1.0, "t2": 2.0, "gap": [1.0, 2.0],
                              "reasons": ["missed"], "text": "",
                              "listening": {"by": "human", "result": "interjections",
                                            "at": "2026-11-24T10:00:00Z", "fingerprint": "f" * 64}}]}
        public = documents._public_review(review)
        self.assertEqual(public["review"][0]["listening"], {"result": "interjections"})
        with files("legacy_files") as (jp, _md, _version):
            self.assertIn("listen_fp", next(i for i in review_pack.manifest(jp)["items"] if i["id"] == "r1"))
            bundle = review_pack.cloud_bundle(jp)
            cloud = review_pack._read(Path(bundle["base"]) / bundle["files"]["manifest.json"], {})
            self.assertTrue(cloud["items"])
            for item in cloud["items"]:
                self.assertFalse({"listen_fp", "listening"} & set(item), item)
            # локальный пакет не тронут: сверка кнопок по-прежнему по отпечатку
            self.assertIn("listen_fp", next(i for i in review_pack.manifest(jp)["items"] if i["id"] == "r1"))

    def test_refusal_on_a_place_without_gap_is_not_a_pause_outcome(self):
        """Отказ прослушивания на месте с проблемой распознавания без пропуска — не «нужна кнопка паузы»."""
        asr = {"id": "r5", "status": "open", "t": 3.0, "t1": 3.0, "t2": 3.2, "gap": [3.0, 3.2],
               "speaker": "Анна", "text": "та", "reasons": ["asr"],
               "asr_problem": {"model": "whisper", "state": "empty_unknown", "words_from": []}}
        with files("legacy_files", result(extra_review=[asr]),
                   questions() + [{"id": "r5", "options": [AS_IS]}]) as (jp, md, _v):
            _, rep = transcript.apply_review(jp, md, [{"id": "r5", "listened": "silence"}], by="human")
            self.assertEqual([c["id"] for c in rep.get("conflicts") or []], ["r5"])
            self.assertNotIn("r5", rep.get("pause_refused") or [])
            _, rep = transcript.apply_review(jp, md, [{"id": "r5", "listened": "interjections"}], by="human")
            self.assertEqual([c["id"] for c in rep.get("conflicts") or []], ["r5"])
            self.assertNotIn("r5", rep.get("pause_refused") or [])
            # для сравнения: тот же отказ на пропуске — исход паузы
            _, rep = transcript.apply_review(jp, md, [{"id": "r1", "listened": "words"}], by="human")
            self.assertIn("r1", rep.get("pause_refused") or [])

    def test_mark_is_dropped_when_words_appear_in_the_pause(self):
        """Тест 8: слово в паузе после «только междометия» — место открыто, пометки нет."""
        with files("legacy_files") as (jp, md, version):
            self.press(jp, version, INTERJECTIONS)
            core.apply_answers_files(jp, md)
            transcript.edit_span(jp, md, 1.4, 1.6, "ой", speaker="Анна", by="human")
            item = self.place_of(jp, "r1")
            self.assertEqual(item["status"], "open")
            self.assertNotIn("listening", item)
            self.assertIn("нужна новая проверка", item.get("note", ""))
            self.assertNotIn(MARK, md.read_text(encoding="utf-8"))

    def test_old_pack_as_is_on_gap_pause_names_the_buttons(self):
        """Тест 17 (оба): вариант агента старого пакета на пустой паузе — исход «нужна кнопка паузы»."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                old_pack(jp, lambda item: (item.update(options=[AS_IS]), item.pop("listen_fp")))
                self.save(jp, version, option=0)
                rep = self.apply(jp, md)
                row = self.outcomes(rep)["r1"]
                self.assertEqual(row["outcome"], "pause_refused", rep.get("outcomes"))
                self.assertEqual(row["why"], review_pack.OUTCOME_REASONS["pause_refused"])
                self.assertIn(SILENCE, row.get("detail", ""))
                self.assertEqual(self.place_of(jp, "r1")["status"], "open")


# ---------- 1.2. таблица и короткие ответы ----------
def write_table(rows):
    path = Path(tempfile.mkdtemp(dir=_TMP)) / "answer_phrases.tsv"
    body = "id\tkind\ttext\tnote\n" + "".join("\t".join(r) + "\n" for r in rows)
    path.write_bytes(body.encode("utf-8"))
    return path


BASE_ROWS = [("A001", "short_word", "да", ""), ("A002", "confirm", "верно", ""),
             ("E001", "evaluation_stem", "согла", ""), ("N001", "negative_prefix", "нет", ""),
             ("N005", "negative_intensifier", "совсем", ""), ("N008", "negative_core", "неверно", "")]


class TableLoaderTests(unittest.TestCase):
    """Тест 9."""

    def test_good_table_and_no_as_kind_agree(self):
        phrases = freeform._read_answer_phrases(write_table(BASE_ROWS))
        self.assertEqual(phrases["short_word"], ["да"])
        with self.assertRaisesRegex(ValueError, "неизвестный kind agree"):
            freeform._read_answer_phrases(write_table(BASE_ROWS + [("A099", "agree", "ну да", "")]))

    def test_each_new_kind_is_required(self):
        for kind in ("short_word", "confirm", "evaluation_stem"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "обязательные категории: " + kind):
                freeform._read_answer_phrases(write_table([r for r in BASE_ROWS if r[1] != kind]))

    def test_phrase_in_short_word_and_confirm_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "и в short_word, и в confirm"):
            freeform._read_answer_phrases(write_table(BASE_ROWS + [("A099", "confirm", "да", "")]))

    def test_repeat_is_checked_inside_a_kind(self):
        phrases = freeform._read_answer_phrases(write_table(BASE_ROWS + [("A013", "short_word", "нет", "")]))
        self.assertIn("нет", phrases["short_word"])
        self.assertIn("нет", phrases["negative_prefix"])
        with self.assertRaisesRegex(ValueError, "повторная фраза"):
            freeform._read_answer_phrases(write_table(BASE_ROWS + [("A099", "short_word", "да", "")]))

    def test_shipped_table_ids_are_kept(self):
        rows = [line.split("\t") for line in
                freeform.ANSWER_PHRASES_PATH.read_text(encoding="utf-8").splitlines()[1:]]
        kinds = {r[0]: r[1] for r in rows}
        self.assertEqual({k for k, v in kinds.items() if v == "short_word"}, {"A001", "A003", "A008", "A009", "A013"})
        self.assertEqual({k for k, v in kinds.items() if v == "confirm"},
                         {"A002", "A004", "A005", "A006", "A007", "A010", "A011", "A012"})
        self.assertEqual({k for k, v in kinds.items() if v == "evaluation_stem"}, {f"E00{n}" for n in range(1, 6)})


def table():
    rows = [line.split("\t") for line in
            freeform.ANSWER_PHRASES_PATH.read_text(encoding="utf-8").splitlines()[1:]]
    return {kind: [r[2] for r in rows if r[1] == kind] for kind in ("short_word", "confirm", "evaluation_stem")}


NAMES = ["Анна", "Борис"]


class ClassifyTests(unittest.TestCase):
    """Тест 10: из таблицы, а не из списка в тесте."""

    def test_every_table_phrase_gets_its_kind(self):
        t = table()
        for kind, phrases in (("short_word", t["short_word"]), ("confirm", t["confirm"])):
            for phrase in phrases:
                for shown in (phrase, phrase.capitalize(), phrase.upper() + "!", phrase + "…",
                              phrase.replace("е", "ё")):
                    with self.subTest(phrase=shown):
                        self.assertEqual(freeform.classify(shown, NAMES), {"kind": kind})

    def test_evaluation_stems_and_the_name_label(self):
        for stem in table()["evaluation_stem"]:
            for shown in (stem + "ен", "я " + stem + "на", stem + "ую вариант"):
                with self.subTest(answer=shown):
                    self.assertEqual(freeform.classify(shown, NAMES), {"kind": "evaluation"})
        self.assertEqual(freeform.classify("согласен с первым вариантом тут", NAMES), {"kind": "text"})
        self.assertEqual(freeform.classify("Анна: верно", NAMES), {"kind": "text"})
        self.assertEqual(freeform.classify("Анна: согласна", NAMES), {"kind": "text"})
        self.assertEqual(freeform.classify("не согласен", NAMES), {"kind": "negative"})
        self.assertEqual(freeform.classify("нет", NAMES), {"kind": "short_word"})

    def test_evaluation_limit_is_exactly_four_words(self):
        self.assertEqual(freeform.EVALUATION_MAX_WORDS, 4)
        for answer, kind in (("мы с ним согласны", "evaluation"), ("я с этим совершенно согласен", "text"),
                             ("ну это было верно", "evaluation"), ("ну это было вполне верно", "text")):
            with self.subTest(answer=answer):
                self.assertEqual(len(answer.split()), 4 if kind == "evaluation" else 5)
                self.assertEqual(freeform.classify(answer, NAMES), {"kind": kind})

    def test_hint_quotes_the_answer_as_written(self):
        """Подсказка повторяет фразу человека без нормализации: ё и регистр сохраняются."""
        pause = man_items()["r1"]
        hint = freeform._short_answer(pause, "confirm", "Всё  верно", [], True)
        self.assertIn("«Имя: Всё верно»", hint)
        place = [{"w": "та", "s": 3.0}]
        hint = freeform._short_answer(man_items()["r4"], "short_word", "Ага", place, False)
        self.assertIn("«Ага»", hint)
        self.assertIn("«Имя: Ага»", hint)
        self.assertNotIn("Имя: ага", hint)


def flat_words(data):
    return sorted(({**w, "_spk": s["speaker"]} for s in data["segments"] for w in s["words"]),
                  key=lambda w: w["s"])


def man_items(data=None, labels_r3=("Оставить как есть", "«дальнейшем»")):
    data = data or result()
    items = {}
    for it in data["review"]:
        items[it["id"]] = {k: it[k] for k in ("id", "t1", "t2", "speaker", "reasons", "text") if k in it}
        if it.get("word_ids") and not transcript.is_gap_place(it):
            items[it["id"]]["word_ids"] = it["word_ids"]
    items["r3"]["options"] = [{"label": labels_r3[0], "decision": {}},
                              {"label": labels_r3[1], "decision": {"text": "дальнейшем"}}]
    return items


class CardEditsTableTests(unittest.TestCase):
    """Тест 11: каждая фраза таблицы × вид места → действие по таблице В8."""

    def run_card(self, item, text, data=None):
        data = data or result()
        out = freeform.card_edits(copy.deepcopy(item), {"text": text}, flat_words(data), NAMES)
        return out, out["report"]

    def test_matrix(self):
        t = table()
        items = man_items()
        for phrase in t["short_word"]:
            with self.subTest(place="пауза", phrase=phrase):
                out, rep = self.run_card(items["r1"], phrase)
                self.assertEqual(out["kind"], "missed")
                self.assertEqual([e["text"] for e in out["edits"]], [phrase])
            for place, wid in (("слово", "r4"), ("МР", "r3")):
                with self.subTest(place=place, phrase=phrase):
                    data = result()
                    target = next(w for w in data["segments"][0]["words"] if w["word_id"] == items[wid]["word_ids"][0])
                    target["w"] = phrase
                    out, rep = self.run_card(items[wid], phrase, data)
                    self.assertTrue(rep.get("confirm_as_is"), rep)
                    self.assertEqual(out["edits"], [])
                    out, rep = self.run_card(items[wid], phrase)          # слово места другое
                    self.assertEqual(rep["kind"], "hint")
                    self.assertEqual(out["edits"], [])
                    self.assertIn(f"напишите «Имя: {phrase}»", rep["human"])
        for phrase in t["confirm"]:
            with self.subTest(place="пауза", phrase=phrase):
                out, rep = self.run_card(items["r1"], phrase)
                self.assertEqual((rep["kind"], out["edits"]), ("hint", []))
                self.assertIn(SILENCE, rep["human"])
            with self.subTest(place="слово", phrase=phrase):
                out, rep = self.run_card(items["r4"], phrase)
                self.assertTrue(rep.get("confirm_as_is"))
                self.assertEqual(out["edits"], [])
            with self.subTest(place="МР", phrase=phrase):
                out, rep = self.run_card(items["r3"], phrase)
                self.assertEqual(rep["kind"], "hint")
                self.assertIn("выберите вариант «Оставить как есть»", rep["human"])
        for stem in t["evaluation_stem"]:
            for wid in ("r1", "r4", "r3"):
                with self.subTest(place=wid, stem=stem):
                    out, rep = self.run_card(items[wid], stem + "ен")
                    self.assertEqual((rep["kind"], out["edits"]), ("hint", []))
                    self.assertEqual(rep["human"], freeform.HINT_EVALUATION)

    def test_disagreement_without_an_as_is_option(self):
        items = man_items(labels_r3=("«дальнейшем»", "«дальше»"))
        items["r3"]["options"][0]["decision"] = {"text": "дальнейшем"}
        items["r3"]["options"][1]["decision"] = {"text": "дальше"}
        _, rep = self.run_card(items["r3"], "верно")
        self.assertEqual(rep["kind"], "hint")
        self.assertNotIn("выберите вариант", rep["human"])
        self.assertIn("впишите услышанные слова", rep["human"])

    def test_pause_short_word_without_speaker_is_refused_as_before(self):
        item = man_items()["r1"]
        item["speaker"] = None
        out, rep = self.run_card(item, "да")
        self.assertEqual(out["kind"], "unmatched")
        self.assertIn("не удалось определить говорящего", rep["human"])

    def test_speaker_card_is_the_named_exception(self):
        item = dict(man_items()["r4"], answer_mode="speaker")
        for phrase in ("да", "верно"):
            out, rep = self.run_card(item, phrase)
            self.assertEqual((out["kind"], out["edits"]), ("agree", []))
            self.assertIsNone(rep["human"])
            self.assertFalse(rep.get("confirm_as_is"))
        out, rep = self.run_card(item, "согласен")
        self.assertEqual(rep["kind"], "speaker")


class ShortAnswersApplyTests(Case):
    def test_yes_on_empty_pause_is_written_as_a_word(self):
        """Тест 12 (оба): «да» на пустой паузе с говорящим вписывается словом."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.save(jp, version, text="да")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.texts(jp), ["начало", "да", "вот", "дальше", "та"])
                self.assertEqual(self.place_of(jp, "r1")["status"], "resolved")

    def test_short_words_and_confirms_on_word_places(self):
        """Тесты 13, 14, 16 (оба): «да» на «да» — как есть; «да» на «та» — подсказка; «верно» на МР — подсказка."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, result(r2_word="да")) as (jp, md, version):
                self.save(jp, version, "r2", text="да")
                self.save(jp, version, "r4", text="да")
                self.save(jp, version, "r3", text="верно")
                rep = self.apply(jp, md)            # повтор с подсказками ничего не пишет (тест 16)
                rows = self.outcomes(rep)
                self.assertEqual(rows["r2"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.place_of(jp, "r2")["status"], "resolved")
                self.assertEqual(self.word(jp, "w2").get("by"), "human")
                for aid in ("r4", "r3"):
                    self.assertEqual(rows[aid]["outcome"], "hint", rep.get("outcomes"))
                    self.assertEqual(self.place_of(jp, aid)["status"], "open")
                self.assertIn("напишите «Имя: да»", rows["r4"].get("detail", ""))
                self.assertIn("«Оставить как есть»", rows["r3"].get("detail", ""))
                self.assertEqual(self.texts(jp), ["начало", "да", "дальше", "та"])
                self.assertNotIn("by", self.word(jp, "w5"))

    def test_confirm_keeps_the_words_and_ignores_the_recommendation(self):
        """Тест 14 (оба): «верно» на месте-слове с рекомендацией — как есть, рекомендация не применена."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.save(jp, version, "r4", text="Верно!")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.texts(jp), ["начало", "вот", "дальше", "та"])
                self.assertEqual(self.word(jp, "w5").get("by"), "human")
                self.assertEqual(self.place_of(jp, "r4")["status"], "resolved")

    def test_confirm_on_asr_problem_needs_listening(self):
        """Тест 18: «верно» на месте с проблемой распознавания не обходит прослушивание."""
        data = result()
        data["review"][3]["asr_problem"] = {"model": "whisper", "state": "disagreement",
                                            "words_from": ["primary", "second"]}
        data["review"][3]["gap"] = [3.0, 3.04]
        with files("legacy_files", data) as (jp, md, version):
            self.save(jp, version, "r4", text="верно")
            rep = self.apply(jp, md)
            row = self.outcomes(rep)["r4"]
            self.assertNotEqual(row["outcome"], "applied")
            self.assertIn("нужно отдельное подтверждение прослушивания", row.get("detail", ""))
            self.assertNotEqual(self.place_of(jp, "r4")["status"], "resolved")
            self.assertNotIn("by", self.word(jp, "w5"))


class VoiceMatchTests(unittest.TestCase):
    """Тест 15: голос = текст."""

    def test_short_answers_by_voice_are_free(self):
        item = {"options": [{"label": "Оставить как есть"}, {"label": "Заменить на «тот»"}], "recommended": 1}
        for said in ("да", "верно", "всё верно", "согласен", "как предложено", "рекомендую", "правильно",
                     "ага", "угу", "нет", "неверно", "совсем не согласен"):
            with self.subTest(said=said):
                self.assertEqual(review_pack.match(said, item), {"kind": "free"})
        self.assertEqual(review_pack.match("второй", item), {"kind": "option", "option": 1})
        self.assertEqual(review_pack.match("2", item), {"kind": "option", "option": 1})
        self.assertEqual(review_pack.match("оставь человеку", item)["kind"], "human")
        self.assertEqual(review_pack.match("повтори", item)["kind"], "repeat")

    def test_option_number_next_to_confirm_selects_option(self):
        """Номер варианта рядом со словом подтверждения или оценки выбирает вариант, как до 6б."""
        item = {"options": [{"label": "Оставить как есть"}, {"label": "Заменить на «тот»"},
                            {"label": "Заменить на «этот»"}], "recommended": 1}
        for said, n in (("верно, второй", 1), ("согласен, первый", 0), ("правильно, второй вариант", 1),
                        ("второй, верно", 1), ("верно 2", 1), ("да, второй", 1), ("Да, третий.", 2)):
            with self.subTest(said=said):
                self.assertEqual(review_pack.match(said, item), {"kind": "option", "option": n})
        # номер вне списка вариантов — не номер: короткий ответ разбирается таблицей
        self.assertEqual(review_pack.match("верно, пятый", item), {"kind": "free"})


if __name__ == "__main__":
    unittest.main()
