"""Анализ обычной работы лаборатории (шаг 3а, PR-Б): гипотезы моделей и ответы человека по копии prepare --work.

Только синтетика во временной папке: запись в базе и карточки — штатными recordings, review_pack и
core.apply_answers_files; правка вне карточки — core.edit_span_files от человека; гипотезы — файл hyp-2 со
словами трёх ролей и протоколом голосования words.combine_hypotheses. Затем prepare --work и analyze.
Установщик папку lab/ не ставит: в установленной копии (selfcheck.py) набор пропускается.
"""
import atexit
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-lab-work-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "lab"
HAVE_LAB = (LAB / "lab_work.py").is_file()
sys.path.insert(0, str(ROOT))
if HAVE_LAB:
    sys.path.insert(0, str(LAB))
    import lab_analyze  # noqa: E402
    import lab_sandbox  # noqa: E402
    import lab_work  # noqa: E402

import core  # noqa: E402
import hypotheses  # noqa: E402
import recordings  # noqa: E402
import review_pack  # noqa: E402
import settings  # noqa: E402
import words as W  # noqa: E402

SECRET = "Тайнословие"
PCM = "c0ffee" * 10 + "abcd"            # хэш звука (64 знака): общий у двух записей одного звука

PRIMARY = [{"w": "тест", "s": 0.2, "e": 0.5, "c": 0.9}, {"w": "слово", "s": 0.7, "e": 1.0, "c": 0.4},
           {"w": SECRET, "s": 1.2, "e": 1.5, "c": 0.8}]
THIRD = ["тест", "слива", SECRET]       # третья спорит только на «слово»
QUESTIONS = [
    {"id": "r1", "question": "Как правильно?", "recommended": 1, "options": [
        {"label": "другое", "decision": {"text": "другое"}}, {"label": "тест", "decision": {"text": "тест"}}]},
    {"id": "r2", "question": "Как правильно?", "recommended": 0, "options": [
        {"label": "слово", "decision": {"text": "слово"}}, {"label": "слива", "decision": {"text": "слива"}}]},
]


def fused():
    """Сведение гипотез тем же правилом, что расшифровка: -> (слова основной, quality, протокол)."""
    main = copy.deepcopy(PRIMARY)
    trace = {}
    out = W.combine_hypotheses({"primary": main, "second": copy.deepcopy(PRIMARY), "whisper": list(THIRD)}, {},
                               trace=trace)
    return main, out["quality"], trace


def result_data(audio, sha):
    main, quality, _ = fused()
    ws = [{"w": w["w"], "s": w["s"], "e": w["e"], "c": w["c"], "word_id": f"w{i}",
           **({"flags": ["asr"]} if not w.get("agree", True) else {})} for i, w in enumerate(main, 1)]
    return {"file": str(audio), "duration": "00:00:02", "speakers": ["Глеб"], "speaker_labels": ["speaker-1"],
            "timing": {"total_sec": 4}, "group": "demo", "audio_sha256": sha,
            "quality": {"asr_diagnostics": {}, **quality}, "vocab_suggestions": [],
            "segments": [{"start": 0.2, "end": 1.5, "label": "speaker-1", "speaker": "Глеб",
                          "text": " ".join(w["w"] for w in ws), "words": ws}],
            "review": [{"id": "r1", "status": "open", "text": "тест", "t": 0.2, "t1": 0.2, "t2": 0.5,
                        "speaker": "Глеб", "word_ids": ["w1"], "reasons": ["conf"]},
                       {"id": "r2", "status": "open", "text": "слово", "t": 0.7, "t1": 0.7, "t2": 1.0,
                        "speaker": "Глеб", "word_ids": ["w2"], "reasons": ["asr"]}],
            "review_done": False, "edits": []}


def hyp_record(recording_id, sha, heard=True, primary=None):
    _main, _q, trace = fused()
    chunk = {"index": 0, "audio_span": [0.0, 2.0], "own_span": [0.0, 2.0], "state": "ok"}
    roles = [{"role": "primary", "chunks": [chunk], "words": copy.deepcopy(primary or PRIMARY)},
             {"role": "second", "chunks": [chunk], "words": copy.deepcopy(PRIMARY)},
             {"role": "third", "chunks": [{**chunk, "heard": heard, "text": " ".join(THIRD)}], "words": None}]
    return hypotheses.build_record(
        roles, legacy={"combine_rule": "vote3_partial"},
        combine={"function": "vote_n", "third_words": list(THIRD), "partial": None, "skipped": None},
        trace=trace, meta={"audio": {"audio_sha256": sha, "pcm_sha256": PCM, "duration_sec": 2.0},
                           "ids": {"recording_id": recording_id, "run_id": "x"}})


class Source:
    """Синтетическая настоящая папка данных: база записей, документы рядом с аудио, папка гипотез."""

    def __init__(self, base):
        self.base = base
        self.data = base / "data"
        self.audio_dir = base / "audio"
        self.data.mkdir()
        self.audio_dir.mkdir()
        self.db = self.data / "state.sqlite3"
        self.store = recordings.initialize_store(self.db, display_name="Тест")
        self.n = 0

    def close(self):
        try:
            self.store.close()
        except Exception:
            pass

    def recording(self, answers, hyp=True, sha=None, heard=True, hyp_primary=None, edit=None):
        """Запись с двумя карточками; answers — {id: поля ответа}. -> (recording_id, путь стенограммы)."""
        self.n += 1
        audio = self.audio_dir / f"rec{self.n}.wav"
        audio.write_bytes(b"synthetic" + bytes([self.n]))
        sha = sha if sha is not None else hashlib.sha256(audio.read_bytes()).hexdigest()
        layout = recordings.document_layout(audio)
        layout["root"].mkdir()
        with mock.patch.object(settings.paths, "STATE_DB", self.db), \
                mock.patch.object(review_pack, "cut_clip", side_effect=lambda _s, t1, _t2, _out: t1):
            root_id, targets = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
            rid = str(uuid.uuid4())
            recordings.commit_worker_result(
                self.store, result_data(audio, sha), audio, root_id, recordings.operation_context(self.store, "model"),
                recording_id=rid, operation_id=str(uuid.uuid4()), generated_at="2026-11-25T00:00:00Z",
                document_paths=targets)
            recordings.materialize_recording(self.store, rid)
            jp, md = layout["root"] / targets["json"], layout["root"] / targets["md"]
            pack = review_pack.build(jp, copy.deepcopy(QUESTIONS))
            for k, (aid, row) in enumerate(answers.items()):
                actor = row.pop("_actor", "human")
                review_pack.add_answers(jp, [{"id": aid, "version": pack["version"],
                                              "ts": f"2026-11-25T10:00:{k:02d}", **row}], "local",
                                        trusted_actor=actor)
            try:
                core.apply_answers_files(jp, md)
            except core.CoreError:
                pass       # ответ неизвестного автора программа как человеческий не применяет — он сохранён
            if edit:
                core.edit_span_files(jp, md, *edit, by="human")
        if hyp:
            rec = hyp_record(rid, sha, heard=heard, primary=hyp_primary)
            folder = self.data / "hypotheses" / PCM[:12]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"20261125-1000{self.n:02d}-aaaaaa.json").write_text(json.dumps(rec, ensure_ascii=False),
                                                                           encoding="utf-8")
        return rid, jp


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class WorkAnalysis(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="work-", dir=_TMP))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.src = Source(self.base)
        self.addCleanup(self.src.close)
        self.program = self.base / "program"
        self.program.mkdir()

    def analyze(self):
        self.src.store.close()
        sandbox = self.base / f"sandbox{len(list(self.base.glob('sandbox*')))}"
        lab_sandbox.prepare(sandbox, self.src.data, self.program, "skip", env={}, log=lambda *_: None, work=True)
        s, _marker, units = lab_sandbox.load_sandbox(sandbox, env={})
        analysis = lab_analyze.analyze(s, units, log=lambda *_: None)
        return analysis, s

    def test_agree_but_corrected_and_disagree_confirmed(self):
        self.src.recording({"r1": {"option": 0}, "r2": {"option": 0}})
        a, s = self.analyze()
        work = a["work"]
        self.assertEqual(a["version"], 3)
        self.assertEqual(work["recordings"]["states"]["ok"], 1)
        self.assertEqual(work["places"]["rows"]["eligible"], 2)
        self.assertEqual(work["places"]["window_by"], {"words": 2, "time": 0, "pause": 0})
        cards = work["m1"]["cards"]
        self.assertEqual(cards["agree_corrected"], 1, work)          # «тест» -> «другое»: модели согласны
        self.assertEqual(cards["disagree_confirmed"], 1, work)       # «слово»: третья спорила, человек оставил
        self.assertEqual(work["m1"]["by_reason"]["conf"]["agree_corrected"], 1)
        self.assertEqual(work["m2"]["card"]["agree_corrected"], {"n": 1, "median": 0.9, "min": 0.9, "max": 0.9})
        self.assertEqual(work["m1"]["shares"]["card"],
                         {"corrected_when_agree": 1.0, "corrected_when_disagree": 0.0, "places": 2,
                          "recordings": 1, "few_data": lab_work.FEW})
        self.assertEqual(work["recordings_with_agree_corrected"], 1)
        self.assertEqual(work["roles_matched_human"]["none"], 1)     # «другое» не дала ни одна модель
        self.assertEqual(work["roles_matched_human"]["primary"], 1)
        unit = work["recordings"]["units"][0]
        self.assertEqual(unit["sound_id"], PCM[:12])
        self.assertEqual(len(unit["file_unit"]), 16)
        examples = (s / "results" / "examples.md").read_text(encoding="utf-8")
        self.assertIn("другое", examples)
        report = (s / "results" / "report.md").read_text(encoding="utf-8")
        self.assertIn("## 6. Обычная работа", report)
        self.assertNotIn("другое", report)

    def test_window_is_taken_from_first_revision_when_times_moved(self):
        """Правка сдвинула время слов в текущей ревизии: окно места — по ревизии 1 (звук не менялся)."""
        self.src.recording({"r1": {"option": 0}, "r2": {"option": 0}})
        _a, s = self.analyze()
        rec_row = lab_work.read_recordings(s / "source" / "state.sqlite3")[0]
        for seg in rec_row["current"]["segments"]:
            for w in seg["words"]:
                w["s"], w["e"] = float(w["s"]) + 5, float(w["e"]) + 5
        manifest = json.loads((s / "source" / lab_sandbox.HYP_MANIFEST).read_text(encoding="utf-8"))
        hyps, _orphan = lab_work.read_hypotheses(s / "source" / "hypotheses", manifest)
        rows = lab_work.recording_places(rec_row, hyps[rec_row["recording_id"]][0], W)
        cards = [r for r in rows if r["source"] == "card"]
        self.assertEqual([r["window_by"] for r in cards], ["words", "words"])
        self.assertEqual([r.get("window") for r in cards], [[0.2, 0.5], [0.7, 1.0]])

    def test_analysis_json_has_no_texts_paths_or_ids(self):
        rid, _jp = self.src.recording({"r1": {"option": 0}, "r2": {"option": 0}})
        _a, s = self.analyze()
        text = (s / "export" / "analysis.json").read_text(encoding="utf-8")
        for bad in (SECRET, "другое", "слива", rid, str(self.base), '"r1"', '"r2"', "w1"):
            self.assertNotIn(bad, text)
        report = (s / "results" / "report.md").read_text(encoding="utf-8")
        line = next(x for x in report.splitlines() if x.startswith("Доли (card):"))
        self.assertIn("мест 2, записей 1", line)          # процент показан и при малых данных — с пометкой
        self.assertIn(lab_work.FEW, line)
        self.assertRegex(line, r"corrected_when_agree = \d+(\.\d)?%")

    def test_answers_not_of_human_and_unanswered_are_separate_rows(self):
        """Ответ неизвестного автора программа не применяет: он «сохранён, не применён». Применённый ответ
        не человека (данные до проверки автора) — строка not_human: Rules.test_classification_of_answers."""
        self.src.recording({"r1": {"option": 0, "_actor": "unknown"}})
        work = self.analyze()[0]["work"]
        rows = work["places"]["rows"]
        self.assertEqual((rows["saved_not_applied"], rows["unanswered"], rows["eligible"]), (1, 1, 0), rows)
        self.assertEqual(sum(work["m1"]["cards"].values()), 0)

    def test_skip_and_left_for_human(self):
        self.src.recording({"r1": {"kind": "human"}, "r2": {"kind": "skip", "text": "пропуск"}})
        rows = self.analyze()[0]["work"]["places"]["rows"]
        self.assertEqual((rows["left_for_human"], rows["skipped"], rows["eligible"]), (1, 1, 0), rows)

    def test_edit_outside_cards_is_a_separate_row(self):
        self.src.recording({}, edit=(1.2, 1.5, "иное"))
        work = self.analyze()[0]["work"]
        self.assertEqual(work["places"]["edits_without_card"], 1, work)
        self.assertEqual(work["m1"]["edits"]["agree_corrected"], 1)
        self.assertEqual(sum(work["m1"]["cards"].values()), 0)

    def test_linking_states(self):
        self.src.recording({"r1": {"option": 0}}, hyp=False)                        # гипотез нет
        self.src.recording({"r1": {"option": 0}}, sha="")                            # хэша файла нет
        rid, _ = self.src.recording({"r1": {"option": 0}})
        folder = self.src.data / "hypotheses" / PCM[:12]
        f = sorted(folder.iterdir())[-1]
        rec = json.loads(f.read_text(encoding="utf-8"))
        rec["audio"]["audio_sha256"] = "0" * 64                                     # гипотезы от другого файла
        f.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        other = [dict(w) for w in PRIMARY]
        other[0]["w"] = "чужое"                                                     # гипотезы чужой записи
        self.src.recording({"r1": {"option": 0}}, hyp_primary=other + [{"w": "лишнее", "s": 1.7, "e": 1.9, "c": 0.5}])
        states = self.analyze()[0]["work"]["recordings"]["states"]
        self.assertEqual({k: v for k, v in states.items() if v},
                         {"no_hypotheses": 1, "no_file_hash": 1, "other_file": 1, "mismatch": 1})

    def test_two_recordings_of_one_sound_with_different_decisions(self):
        self.src.recording({"r1": {"option": 0}})        # исправил
        self.src.recording({"r1": {"option": 1}})        # подтвердил «тест»
        work = self.analyze()[0]["work"]
        self.assertEqual(work["places"]["rows"]["conflicting_between_variants"], 2, work["places"])
        self.assertEqual(sum(work["m1"]["cards"].values()), 0)

    def test_third_that_did_not_hear_is_not_counted(self):
        self.src.recording({"r1": {"option": 0}, "r2": {"option": 0}}, heard=False)
        work = self.analyze()[0]["work"]
        self.assertEqual(work["m1"]["cards"]["agreement_unknown"], 2, work["m1"])
        self.assertEqual(work["roles_matched_human"]["third"], 0)

    def test_old_sandbox_without_work_is_not_measured(self):
        self.src.store.close()
        sandbox = self.base / "plain"
        lab_sandbox.prepare(sandbox, self.src.data, self.program, "skip", env={}, log=lambda *_: None)
        s, _m, units = lab_sandbox.load_sandbox(sandbox, env={})
        a = lab_analyze.analyze(s, units, log=lambda *_: None)
        self.assertIn("not_measured", a["work"])


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class Rules(unittest.TestCase):
    def test_every_outcome_is_classified(self):
        self.assertEqual(set(lab_work.OUTCOME_ROW), set(review_pack.OUTCOME_REASONS))
        self.assertTrue(set(lab_work.OUTCOME_ROW.values()) <= set(lab_work.ROWS))

    def classify(self, answer, state="applied", item=None):
        item = item or {"id": "r1", "options": [{"label": "x", "decision": {"text": "x"}}]}
        return lab_work.classify_answer("r1", item, answer, {"r1": {"state": state}}, review_pack)[0]

    def test_classification_of_answers(self):
        human = {"actor_kind": "human"}
        self.assertEqual(self.classify(None), "unanswered")
        self.assertEqual(self.classify({**human, "kind": "option", "option": 0}), "eligible")
        self.assertEqual(self.classify({**human, "kind": "free", "text": "x"}), "eligible")
        self.assertEqual(self.classify({**human, "kind": "free"}, "applied_open"), "eligible")
        self.assertEqual(self.classify({**human, "kind": "option", "option": 0}, "saved"), "saved_not_applied")
        self.assertEqual(self.classify({**human, "kind": "option", "option": 0}, "reverted"), "reverted")
        self.assertEqual(self.classify({**human, "kind": "revert"}, "skipped"), "reverted")
        self.assertEqual(self.classify({**human, "kind": "option", "option": 0}, "apply_conflict"), "not_applied")
        self.assertEqual(self.classify({**human, "kind": "option", "option": 0}, "listen_mismatch"), "not_applied")
        self.assertEqual(self.classify({**human, "kind": "option", "option": 0}, "applied_check"), "applied_unchecked")
        self.assertEqual(self.classify({"kind": "option", "option": 0}), "not_human")
        self.assertEqual(self.classify({"actor_kind": "model", "kind": "option", "option": 0}), "not_human")
        speaker = {"id": "r1", "options": [{"label": "Вера", "decision": {"speaker": "Вера"}}]}
        self.assertEqual(self.classify({**human, "kind": "option", "option": 0}, item=speaker), "speaker_only")
        mode = {"id": "r1", "answer_mode": "speaker", "options": []}
        self.assertEqual(self.classify({**human, "kind": "free", "text": "Вера"}, item=mode), "speaker_only")
        # ответ прежней редакции карточки — даже применённый — не в счёт
        new_rev = {"id": "r1", "rev": 2, "options": [{"label": "x", "decision": {"text": "x"}}],
                   "history": [{"rev": 1, "answer": {"kind": "option", "option": 0}, "applied": True}]}
        self.assertEqual(self.classify({**human, "kind": "option", "option": 0}, item=new_rev), "previous_revision")
        self.assertEqual(self.classify(None, item=new_rev), "previous_revision")

    def test_window_from_first_revision(self):
        first = [{"word_id": "a", "s": 1.0, "e": 1.4}, {"word_id": "b", "s": 1.5, "e": 2.0}]
        self.assertEqual(lab_work.place_window({"word_ids": ["a", "b"], "t1": 9, "t2": 9}, first), (1.0, 2.0, "words"))
        self.assertEqual(lab_work.place_window({"word_ids": ["a", "new"], "t1": 3, "t2": 4}, first), (3.0, 4.0, "time"))
        self.assertEqual(lab_work.place_window({"gap": [5, 6], "t1": 5, "t2": 6}, first), (5.0, 6.0, "pause"))

    def test_shares_always_shown_few_data_below_30_places_or_3_recordings(self):
        table = {c: 0 for c in lab_work.CELLS}
        table.update(agree_corrected=3, agree_confirmed=27)
        for places, recs in ((29, 3), (30, 2)):
            sh = lab_work._shares(table, places, recs)
            self.assertEqual((sh["corrected_when_agree"], sh["places"], sh["recordings"], sh["few_data"]),
                             (0.1, places, recs, lab_work.FEW))
            self.assertIsNone(sh["corrected_when_disagree"])
        self.assertNotIn("few_data", lab_work._shares(table, 30, 3))
        self.assertEqual(lab_work._shares(table, 30, 3)["corrected_when_agree"], 0.1)

    def test_models_agree_needs_all_three_roles(self):
        rec = hyp_record("r", "s")
        self.assertTrue(lab_work.models_agree(rec, 0.2, 0.5))          # «тест» — единогласно
        self.assertFalse(lab_work.models_agree(rec, 0.7, 1.0))         # «слово» — третья против
        rec["vote_trace"]["roles"] = ["primary", "second"]
        self.assertIsNone(lab_work.models_agree(rec, 0.2, 0.5))
        self.assertEqual(lab_work.role_texts(hyp_record("r", "s"), 0.7, 1.0)["third"], ["слива"])
        self.assertIsNone(lab_work.role_texts(hyp_record("r", "s", heard=False), 0.7, 1.0)["third"])


# Замороженные образцы форматов гипотез (раздел 0.3 плана): изменение формата, ломающее чтение, роняет тест.
FROZEN_LAB_HYP_1 = {"schema": "lab-hyp-1", "combine_rule": "vote3_partial",
                    "primary": {"words": [{"w": "раз", "s": 0.0, "e": 0.3, "c": 0.9}]},
                    "second": {"words": [{"w": "раз", "s": 0.0, "e": 0.3, "c": 0.8}]},
                    "third": {"chunks": [{"chunk_index": 0, "text": "раз"}], "pieces": 1, "mode": "all"},
                    "vote": {"function": "vote3", "third_words": ["раз"]}}
FROZEN_HYP_2 = {"schema": "hyp-2", "combine_rule": "vote3_partial",
                "audio": {"audio_sha256": "a" * 64, "pcm_sha256": "b" * 64, "duration_sec": 1.0},
                "program": {"version": "x"}, "environment": {}, "hardware": {}, "conditions": {}, "config": {},
                "ids": {"recording_id": "rec", "run_id": "20261125-100000-aaaaaa"},
                "roles": [{"role": "primary", "model": {}, "engine": {}, "timing": {}, "chunks": [],
                           "words": [{"w": "раз", "s": 0.0, "e": 0.3, "c": 0.9}], "word_times": {"status": "available"}},
                          {"role": "second", "model": {}, "engine": {}, "timing": {}, "chunks": [],
                           "words": [{"w": "раз", "s": 0.0, "e": 0.3, "c": 0.8}], "word_times": {"status": "available"}},
                          {"role": "third", "model": {}, "engine": {}, "timing": {}, "chunks": [],
                           "words": {"status": "not_available"}, "word_times": {"status": "not_available"}}],
                "combine": {"function": "vote_n", "third_words": ["раз"]}, "vote_trace": {"places": []}}


class FrozenFormats(unittest.TestCase):
    def test_every_known_version_reads_with_same_words_and_roles(self):
        old = hypotheses.read(json.loads(json.dumps(FROZEN_LAB_HYP_1)))
        self.assertEqual(old["primary"]["words"][0]["w"], "раз")
        self.assertEqual(old["vote"]["third_words"], ["раз"])
        new = hypotheses.read(json.loads(json.dumps(FROZEN_HYP_2)))
        self.assertEqual([r["role"] for r in new["roles"]], ["primary", "second", "third"])
        self.assertEqual(new["roles"][0]["words"][0]["w"], "раз")
        self.assertEqual(hypotheses.check(new), [])
        self.assertEqual(hypotheses.read({"schema": "hyp-9"}), {})
        if HAVE_LAB:
            self.assertEqual(lab_work._role_words(new, "second")[0]["c"], 0.8)


if __name__ == "__main__":
    unittest.main()
