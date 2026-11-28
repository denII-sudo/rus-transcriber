"""Наложение при неизмеренном отрыве голосов (Т-1-Н4): «не измерено» — не «далеко».

Только синтетика: звук — тишина, отпечаток голоса подменён по времени (голос A до границы, B после),
модели не загружаются. Работают настоящие words.assign_voices, words._voice_by_diarization,
words.doubt_flags; RunWordsTests — настоящий worker.run_words и worker._review_places (подменены только
проходы моделей, слой разговора и эмоции), неисключающая разметка pyannote подменена списком отрезков.

Было: слово в наложении получало пометку `overlap`, только если отрыв голосов мал; отрыв 1,0 при одной
опоре и у слов, которым голос дан по разметке, не измерялся, но считался «голоса далеко» — место, где
говорят двое, в разбор не попадало (копия МУРКО: наложение 1,6 с, мест 0). Стало: такое слово в наложении
получает `overlap`; голос слов не меняется; служебный признак `margin_unmeasured` в стенограмму не выходит.
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

_TMP = tempfile.mkdtemp(prefix="rt-overlap-unmeasured-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP

import conspect  # noqa: E402
import core  # noqa: E402
import quotes  # noqa: E402
import transcript  # noqa: E402
import words as W  # noqa: E402
import worker  # noqa: E402
import workflow  # noqa: E402

VA = np.array([1.0, 0.0, 0.0])
VB = np.array([0.0, 1.0, 0.0])
S1, S2 = "Спикер 1", "Спикер 2"


class FakeEmb:
    """Отпечаток отрезка — доля голоса B по времени (граница border), нормированный."""

    def __init__(self, border):
        self.border = border

    def spans(self, audio, spans, min_len=0.0, progress=None, **_):
        out = []
        for s, e in spans:
            if e - s < min_len:
                out.append(None)
                continue
            b = max(0.0, e - max(s, self.border)) / max(e - s, 1e-6)
            v = (1 - b) * VA + b * VB
            out.append(v / np.linalg.norm(v))
        return out


def words_between(t1, t2, step=0.4):
    out, x = [], t1
    while x + 0.3 <= t2:
        out.append({"w": "слово", "s": round(x, 2), "e": round(x + 0.3, 2), "c": 1.0})
        x += step
    return out


def audio_for(sec):
    return np.zeros(int(sec * W.SR), dtype=np.float32)


def flagged(ws, flag):
    return [w["s"] for w in ws if flag in (w.get("flags") or [])]


def inside(ws, t1, t2):
    """Начала слов, которые doubt_flags относит к наложению [t1, t2) (по s + 0,1)."""
    return [w["s"] for w in ws if t1 <= w["s"] + 0.1 < t2]


class DoubtFlagsTests(unittest.TestCase):
    def test_single_reference_word_in_overlap_is_flagged(self):
        """Одна опора: отрыв сравнить не с чем. Слова в наложении 12,0–13,5 — с `overlap`, вне — без."""
        ws = words_between(0.0, 20.0)
        W.assign_voices(audio_for(21.0), ws, {S1: VA}, FakeEmb(99.0), prior=W.LabelAt([(0.0, 20.0, S1)]))
        self.assertTrue(all(w["spk"] == S1 and w["margin"] == 1.0 for w in ws))
        W.doubt_flags(ws, [(12.0, 13.5)])
        self.assertEqual(flagged(ws, "overlap"), inside(ws, 12.0, 13.5))
        self.assertEqual(len(flagged(ws, "overlap")), 4)
        self.assertEqual(flagged(ws, "voice"), [])

    def test_diarization_voice_word_in_overlap_is_flagged(self):
        """Голос B без опоры, слова подписаны по разметке (Т-1-Н1): в наложении — с `overlap`."""
        measured = [dict(w, spk="A", margin=0.5) for w in words_between(0.0, 20.0)]
        extra = words_between(21.0, 24.0)
        W._voice_by_diarization(extra, W.LabelAt([(21.0, 24.0, S2)]), {S2})
        W.doubt_flags(measured + extra, [(21.5, 22.7)])
        self.assertEqual(flagged(extra, "overlap"), inside(extra, 21.5, 22.7))
        self.assertEqual(len(flagged(extra, "overlap")), 3)
        self.assertEqual(flagged(measured, "overlap"), [])
        self.assertTrue(all(w["spk"] == S2 for w in extra))

    def test_distant_measured_voices_in_overlap_not_flagged(self):
        """Контроль: две опоры, отрыв измерен и велик (1,0) — наложение без `overlap`, как прежде."""
        ws = words_between(0.0, 10.0) + words_between(10.5, 20.5)
        prior = W.LabelAt([(0.0, 10.0, S1), (10.5, 20.5, S2)])
        W.assign_voices(audio_for(21.5), ws, {S1: VA, S2: VB}, FakeEmb(10.25), prior=prior)
        W.doubt_flags(ws, [(3.0, 4.5)])
        self.assertTrue(inside(ws, 3.0, 4.5))
        self.assertEqual(flagged(ws, "overlap"), [])
        self.assertFalse(any(w.get("margin_unmeasured") for w in ws))


class _Status:
    def set(self, **_):
        pass


class FakeAnnotation:
    """Неисключающая разметка pyannote: itertracks(yield_label=True) по списку (начало, конец, метка)."""

    def __init__(self, tracks):
        self.tracks = tracks

    def itertracks(self, yield_label=False):
        for s, e, lbl in self.tracks:
            yield SimpleNamespace(start=s, end=e), None, lbl


def run_words(raw, full_tracks, border, label_name=None, pvecs=None, cents=None, allowed=()):
    """Настоящий worker.run_words (по умолчанию без профилей); места для разбора — настоящие.
    -> данные расшифровки."""
    label_name = dict(label_name or {})
    order = {}
    for _, _, spk in sorted(raw):
        order.setdefault(spk, f"Спикер {len(order) + 1}")
    turns = [{"s": s, "e": e, "label": order[spk]} for s, e, spk in worker.build_turns(raw)]
    ws = []
    for t in turns:
        ws += words_between(t["s"], t["e"])
    dur = max(e for _, e, _ in raw) + 1.0
    voice_ctx = {"pvecs": dict(pvecs or {}), "cents": dict(cents or {}),
                 "mapping": {lbl: (nm, 0.9) for lbl, nm in label_name.items()}, "confusions": {}}
    quality = {"vocab": None, "speaker_fixes": [], "vocab_fixes": [], "review": [], "asr_diagnostics": {}}
    fused = {"quality": {}, "evidence": {"whisper": [], "other": []}}
    with mock.patch.object(worker, "primary_pass", return_value=ws), \
            mock.patch.object(worker, "second_pass", return_value=None), \
            mock.patch.object(worker, "whisper_plan", return_value={"model": None}), \
            mock.patch.object(worker.W, "combine_hypotheses", return_value=fused), \
            mock.patch.object(worker, "_check_gender"), \
            mock.patch.object(worker, "_interaction_layer"), mock.patch.object(worker, "_emotions"):
        return worker.run_words(audio_for(dur), "synthetic.wav", {}, _Status(), list(raw), order, turns,
                                FakeAnnotation(full_tracks), FakeEmb(border), dict(label_name),
                                {nm: lbl for lbl, nm in label_name.items()}, voice_ctx, list(allowed), {}, None,
                                quality, None, SimpleNamespace(status=""))


class RunWordsTests(unittest.TestCase):
    def test_overlap_with_single_reference_becomes_review_place(self):
        """Один кластер (опора по нему, Т-1-Н5), pyannote нашёл второй голос 1,5 с поверх него: место для
        разбора с причиной `overlap` появляется; голос слов прежний; служебного признака в стенограмме нет."""
        raw = [(0.0, 10.0, "SPEAKER_00"), (10.5, 20.5, "SPEAKER_00")]
        full = raw + [(12.0, 13.5, "SPEAKER_01")]
        data = run_words(raw, full, 99.0)
        places = [it for it in data["review"] if "overlap" in it.get("reasons", [])]
        self.assertEqual(len(places), 1)
        self.assertTrue(12.0 <= places[0]["t1"] and places[0]["t2"] < 13.5)
        all_words = [w for seg in data["segments"] for w in seg["words"]]
        self.assertEqual({seg["label"] for seg in data["segments"]}, {S1})
        self.assertFalse(any("margin_unmeasured" in w for w in all_words))
        self.assertEqual(len([w for w in all_words if "overlap" in (w.get("flags") or [])]), 4)
        self.assertEqual(data["quality"]["overlaps_sec"], 1.5)
        self.assertNotIn("overlap_unmeasured_skipped", data["quality"])      # 1,5 с из 20 с речи — 8 %


RAW_ONE = [(0.0, 10.0, "SPEAKER_00"), (10.5, 20.5, "SPEAKER_00")]          # 20 с речи без пауз
SHADOW = [(1.0, 8.0, "SPEAKER_01"), (11.0, 19.0, "SPEAKER_01")]            # «тень» эха: 15 с, 75 % речи
VA_NEAR = np.array([0.995, 0.0998, 0.0]) / np.linalg.norm([0.995, 0.0998, 0.0])   # голос, близкий к A


class EchoShareTests(unittest.TestCase):
    """Т-1, Н-3 перепроверки №246: наложения pyannote больше 30 % речи записи (сумма отрезков разметки без
    пауз) — скорее эхо или гул: правило «отрыв не измерен — overlap» не действует, вместо мест — предупреждение;
    пометка при измеренных близких голосах — прежняя."""

    def test_echo_like_overlaps_give_warning_not_places(self):
        data = run_words(RAW_ONE, RAW_ONE + SHADOW, 99.0)
        self.assertEqual([it for it in data["review"] if "overlap" in it.get("reasons", [])], [])
        self.assertEqual(data["quality"]["overlap_unmeasured_skipped"],
                         {"share": 0.75, "sec": 15.0, "spans": [[1.0, 8.0], [11.0, 19.0]]})
        words_ = [w for seg in data["segments"] for w in seg["words"]]
        self.assertFalse(any("overlap" in (w.get("flags") or []) for w in words_))
        self.assertEqual({seg["label"] for seg in data["segments"]}, {S1})       # голос слов прежний
        md = transcript.render_md(dict(data, file="synthetic.wav", duration="00:00:21"))
        self.assertIn("Внимание: на 75 % речи звучит больше одного голоса сразу (эхо, гул или разговор "
                      "вперебивку). Там не сверено, чей голос у слова, поэтому цитаты оттуда не считаются "
                      "проверенными. Прослушайте: 00:00:01–00:00:08, 00:00:11–00:00:19.", md)

    def test_share_around_threshold(self):
        """29 % и ровно 30 % речи — ещё не эхо: места наложений ставятся, предупреждения нет; 31 % — мест нет,
        есть предупреждение."""
        for half, places in ((2.9, 2), (3.0, 2), (3.1, 0)):           # два наложения по half с из 20 с речи
            with self.subTest(share=2 * half / 20):
                shadow = [(2.0, 2.0 + half, "SPEAKER_01"), (12.0, 12.0 + half, "SPEAKER_01")]
                data = run_words(RAW_ONE, RAW_ONE + shadow, 99.0)
                self.assertEqual(len([it for it in data["review"] if "overlap" in it.get("reasons", [])]), places)
                if places:
                    self.assertNotIn("overlap_unmeasured_skipped", data["quality"])
                else:
                    self.assertEqual(data["quality"]["overlap_unmeasured_skipped"],
                                     {"share": 0.31, "sec": 6.2, "spans": [[2.0, 5.1], [12.0, 15.1]]})

    def test_measured_close_voices_still_flagged_with_profiles(self):
        """Профили двух участников с близкими голосами (две опоры, отрыв измерен и мал): при той же доле
        наложений 75 % слова в наложениях помечаются, как до №246; предупреждения нет — пропусков нет."""
        data = run_words(RAW_ONE, RAW_ONE + SHADOW, 99.0, label_name={S1: "Анна"},
                         pvecs={"Анна": VA, "Борис": VA_NEAR}, cents={S1: VA}, allowed=["Анна", "Борис"])
        words_ = [w for seg in data["segments"] for w in seg["words"]]
        self.assertFalse(any("margin_unmeasured" in w for w in words_))
        in_shadow = [w for w in words_ if any(a <= w["s"] + 0.1 < b for a, b, _ in SHADOW)]
        self.assertTrue(in_shadow)
        self.assertTrue(all("overlap" in (w.get("flags") or []) for w in in_shadow))
        self.assertTrue([it for it in data["review"] if "overlap" in it.get("reasons", [])])
        self.assertNotIn("overlap_unmeasured_skipped", data["quality"])

    def test_warning_share_rounds_up_and_bad_field_is_ignored(self):
        """Доля в предупреждении округляется вверх до десятых: 30,04 % — «30,1 %»; испорченное поле
        не роняет render_md и предупреждения не даёт."""
        self.assertIn("на 30,1 % речи", transcript.overlap_skipped_note({"share": 0.3004, "sec": 6.0}))
        self.assertIn("на 75 % речи", transcript.overlap_skipped_note({"share": 0.75, "sec": 15.0}))
        base = {"file": "synthetic.wav", "duration": "00:00:02", "speakers": [], "segments": []}
        for bad in ("yes", {"share": float("nan"), "sec": 1.0}, {"share": float("inf"), "sec": 1.0},
                    {"share": 1.5, "sec": 1.0}, {"share": "0.5", "sec": 1.0}, {"share": True, "sec": 1.0}):
            with self.subTest(bad=bad):
                self.assertIsNone(transcript.overlap_skipped_note(bad))
                md = transcript.render_md(dict(base, quality={"overlap_unmeasured_skipped": bad}))
                self.assertNotIn("больше одного голоса", md)

    def test_share_just_above_threshold_is_not_written_as_threshold(self):
        """Н28-6: доля 0,30004 (места сняты по правилу «больше 30 %») хранится вверх до 4 знаков и пишется
        «30,1 %», а не «30 %»."""
        shadow = [(2.0, 5.0004, "SPEAKER_01"), (12.0, 15.0004, "SPEAKER_01")]      # 6,0008 с из 20 с
        data = run_words(RAW_ONE, RAW_ONE + shadow, 99.0)
        self.assertEqual([it for it in data["review"] if "overlap" in it.get("reasons", [])], [])
        info = data["quality"]["overlap_unmeasured_skipped"]
        self.assertGreater(info["share"], W.OVERLAP_UNMEASURED_MAX_SHARE)
        self.assertIn("на 30,1 % речи", transcript.overlap_skipped_note(info))

    def test_note_lists_first_spans_and_rest(self):
        """Н28-2: в предупреждении — первые пять отрезков с таймкодами и «ещё K»; без отрезков (запись до
        Н28-1) — прежняя концовка; испорченные отрезки отбрасываются, предупреждение остаётся."""
        spans = [[float(10 * k), float(10 * k + 2)] for k in range(7)]
        note = transcript.overlap_skipped_note({"share": 0.5, "sec": 14.0, "spans": spans})
        self.assertIn("Прослушайте: 00:00:00–00:00:02, 00:00:10–00:00:12, 00:00:20–00:00:22, "
                      "00:00:30–00:00:32, 00:00:40–00:00:42 и ещё 2.", note)
        self.assertTrue(transcript.overlap_skipped_note({"share": 0.5, "sec": 1.0}).endswith(
            "цитаты оттуда не считаются проверенными. Прослушайте."))
        for bad in ([[2.0, 1.0]], [[1.0]], "x", [[1.0, float("nan")]], [[True, 2.0]]):
            with self.subTest(bad=bad):
                info = transcript.overlap_skipped_info({"share": 0.5, "sec": 1.0, "spans": bad})
                self.assertEqual(info, {"share": 0.5, "sec": 1.0})
        # Р255-2: испорченный отрезок отбрасывается по одному, годные остаются
        info = transcript.overlap_skipped_info({"share": 0.5, "sec": 1.0,
                                                "spans": [[2.0, 4.0], [6.0, "x"], [12.0, 12.0], (14.0, 15)]})
        self.assertEqual(info["spans"], [[2.0, 4.0], [14.0, 15]])

    def test_status_summary_caps_spans(self):
        """Н28-2: сводка quality статуса задачи несёт поле; отрезков — не больше десяти и число остальных."""
        spans = [[float(k), k + 0.5] for k in range(12)]
        out = worker._overlap_status({"overlap_unmeasured_skipped": {"share": 0.5, "sec": 6.0, "spans": spans}})
        self.assertEqual(out, {"share": 0.5, "sec": 6.0, "spans": spans[:10], "spans_more": 2})
        self.assertIsNone(worker._overlap_status({}))

    def test_spans_only_where_unmeasured_words_are(self):
        """Р255-4: в spans — только наложения со словами без измеренного отрыва; наложение двух посторонних
        меток вне реплик основного голоса (слов нет) в перечень не попадает."""
        extra = [(22.0, 26.0, "SPEAKER_01"), (22.0, 26.0, "SPEAKER_02")]
        data = run_words(RAW_ONE, RAW_ONE + [(1.0, 8.0, "SPEAKER_01")] + extra, 99.0)
        info = data["quality"]["overlap_unmeasured_skipped"]
        self.assertGreater(info["share"], W.OVERLAP_UNMEASURED_MAX_SHARE)
        self.assertEqual(info["spans"], [[1.0, 8.0]])

    def test_doubt_flags_switch_touches_only_unmeasured(self):
        """Прямо doubt_flags: unmeasured_overlap=False снимает только пометку неизмеренных слов."""
        unmeasured = [dict(w, spk=S1, margin=1.0, margin_unmeasured=True) for w in words_between(0.0, 5.0)]
        close = [dict(w, spk=S2, margin=0.01) for w in words_between(10.0, 15.0)]
        W.doubt_flags(unmeasured + close, [(1.0, 4.0), (11.0, 14.0)], unmeasured_overlap=False)
        self.assertEqual(flagged(unmeasured, "overlap"), [])
        self.assertEqual(flagged(close, "overlap"), inside(close, 11.0, 14.0))


def quote_data(spans):
    """Сценарий s249q перепроверки 28.11: одна опора, 20 с речи, «тень» второй метки на отрезках spans."""
    data = run_words(RAW_ONE, RAW_ONE + [(s, e, "SPEAKER_01") for s, e in spans], 99.0)
    words_ = [w for seg in data["segments"] for w in seg["words"]]
    for k, w in enumerate(words_):
        w.setdefault("word_id", f"x{k}")
    data["review"] = data.get("review") or []
    return data, words_


class QuoteBlockersTests(unittest.TestCase):
    """Н28-1: цитата свёртки из слов пропущенного наложения не печатается в кавычках как проверенная."""

    def quote(self, words_):
        return [w["word_id"] for w in words_ if 6.0 <= w["s"] + 0.1 < 9.0 or 12.0 <= w["s"] + 0.1 < 14.0]

    def test_share_023_keeps_places_and_open_blockers(self):
        data, words_ = quote_data([(2.0, 4.0), (12.0, 14.6)])
        self.assertNotIn("overlap_unmeasured_skipped", data["quality"])
        reasons = {b["reason"] for b in workflow.quote_blockers(data, self.quote(words_))}
        self.assertEqual(reasons, {"open"})

    def test_share_035_blocks_quote_without_places(self):
        data, words_ = quote_data([(2.0, 4.0), (6.0, 9.0), (12.0, 14.0)])
        self.assertEqual([it for it in data["review"] if "overlap" in it.get("reasons", [])], [])
        self.assertEqual(data["quality"]["overlap_unmeasured_skipped"]["spans"],
                         [[2.0, 4.0], [6.0, 9.0], [12.0, 14.0]])
        blockers = workflow.quote_blockers(data, self.quote(words_))
        self.assertEqual(blockers, [{"id": "overlap_skipped:1", "reason": "overlap_skipped"},
                                    {"id": "overlap_skipped:2", "reason": "overlap_skipped"}])
        outside = [w["word_id"] for w in words_ if 15.0 <= w["s"] < 19.0]       # вне отрезков — чисто
        self.assertEqual(workflow.quote_blockers(data, outside), [])

    def test_human_touched_word_and_old_record(self):
        data, words_ = quote_data([(2.0, 4.0), (6.0, 9.0), (12.0, 14.0)])
        ids = [w["word_id"] for w in words_ if 6.0 <= w["s"] + 0.1 < 9.0]
        for w in words_:
            if w["word_id"] in ids:
                w["by"] = "human"                       # человек прослушал и поправил эти слова
        self.assertEqual(workflow.quote_blockers(data, ids), [])
        for w in words_:
            w.pop("by", None)
        old = dict(data, quality={"overlap_unmeasured_skipped": {"share": 0.35, "sec": 7.0}})
        self.assertEqual(workflow.quote_blockers(old, ids), [])      # запись до Н28-1: как раньше

    def test_untimed_quote_word_is_unlocated(self):
        data, words_ = quote_data([(2.0, 4.0), (6.0, 9.0), (12.0, 14.0)])
        words_[0].pop("s")
        self.assertEqual(workflow.quote_blockers(data, [words_[0]["word_id"]]),
                         [{"id": "overlap_skipped", "reason": "unlocated"}])


TOK = ["альфа", "бета", "гамма", "дельта", "эпсилон", "дзета", "эта", "тета", "йота", "каппа", "лямбда",
       "мю", "ню", "кси", "омикрон", "пи", "ро", "сигма", "тау", "ипсилон", "йота", "каппа", "лямбда", "мю"]
SPANS = [[2.0, 4.0], [6.0, 9.0], [12.0, 14.0]]
IN_SPAN = ["w8", "w9", "w10", "w11"]          # «йота каппа лямбда мю» с 6,4 с — в отрезке [6, 9)
OUTSIDE = ["w20", "w21", "w22", "w23"]        # та же фраза с 16 с — вне отрезков


def agent_data(spans=SPANS):
    """Одна реплика «Спикер 1», 24 слова, слово k — с k·0,8 с; фраза «йота каппа лямбда мю» дважды."""
    field = {"share": 0.35, "sec": 7.0, **({"spans": spans} if spans is not None else {})}
    ws = [{"word_id": f"w{k}", "w": t, "s": round(k * 0.8, 2), "e": round(k * 0.8 + 0.3, 2), "c": 1.0}
          for k, t in enumerate(TOK)]
    return {"recording_id": "rec", "source_revision": 1, "speaker_names": {}, "review": [], "file": "s.wav",
            "duration": "00:00:20", "speakers": [S1], "quality": {"overlap_unmeasured_skipped": field},
            "segments": [{"segment_id": "g1", "speaker": S1, "start": 0.0, "end": 20.0, "text": " ".join(TOK),
                          "words": ws}]}


class AgentConspectTests(unittest.TestCase):
    """Р255-1: конспект агента (check_quotes, conspect_link/conspect_render, conspect_link_selected) — цитата
    из слов отрезка пропущенного наложения даёт сомнение и в «» как сверенная не выходит."""

    TEXT = "Итог: Спикер 1: «йота каппа лямбда мю»."

    def test_check_quotes_reports_overlap_doubt(self):
        check = quotes.check("Спикер 1: «альфа бета гамма» и «эта тета йота каппа»", agent_data())
        self.assertEqual(check["summary"]["in_doubt"], 1)
        row = next(r for r in check["problems"] if r.get("doubt"))
        self.assertEqual(row["doubt"], ["overlap_skipped:1"])
        old = quotes.check("Спикер 1: «эта тета йота каппа»", agent_data(spans=None))     # запись до Н28-1
        self.assertEqual(old["summary"]["in_doubt"], 0)

    def test_link_and_render_drop_quotes_for_overlap_words(self):
        for spans, word_ids, quoted in ((SPANS, IN_SPAN, False), (SPANS, OUTSIDE, True), (None, IN_SPAN, True)):
            with self.subTest(spans=spans, word_ids=word_ids):
                data = agent_data(spans)
                selection = {"recording_id": "rec", "source_revision": 1, "word_ids": word_ids}
                obj, rep = conspect.link(self.TEXT, data, selections={"1": selection})      # путь link_selected
                self.assertEqual(rep["linked"], 1)
                md, rrep = conspect.render(obj, data, update_last=False)
                if quoted:
                    self.assertIn("Спикер 1: «йота каппа лямбда мю»", md)
                    self.assertNotIn("overlap_unchecked", rrep)
                else:
                    self.assertIn("Спикер 1: йота каппа лямбда мю (по распознаванию, не проверено, 00:00:06)", md)
                    self.assertNotIn("«йота", md)
                    self.assertEqual(len(rrep["overlap_unchecked"]), 1)
                    self.assertEqual(quotes.check(md, data)["summary"]["verbatim"], 0)

    def test_plain_link_of_unique_quote_in_span(self):
        """conspect_link без выбора: цитата однозначна — связана и всё равно печатается без кавычек."""
        data = agent_data()
        obj, rep = conspect.link("Итог: Спикер 1: «тета йота каппа».", data)
        self.assertEqual(rep["linked"], 1)
        md, _ = conspect.render(obj, data, update_last=False)
        self.assertIn("Спикер 1: тета йота каппа (по распознаванию, не проверено, 00:00:05)", md)

    def test_read_transcript_json_summary_is_checked_and_capped(self):
        """Р255-5: JSON read_transcript — поле проверено (испорченные отрезки отброшены) и сокращено."""
        spans = [[float(k), k + 0.5] for k in range(12)] + [[3.0, "x"]]
        data = dict(agent_data(spans), segments=[])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "t.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            summary = json.loads(core._text_only({"json": str(path)}, "json"))["asr_summary"]
        self.assertEqual(summary["overlap_unmeasured_skipped"],
                         {"share": 0.35, "sec": 7.0, "spans": spans[:10], "spans_more": 2})


def split_data(spans=((5.4, 7.0),), review=()):
    """Реплика «Спикер 1»: «начало» 1,0 с, «оригинал» 4,8–6,0 с (w1), «конец» 8,0 с. Отрезки spans — пропущенные
    наложения записи (доля выше порога). Правило попадания: начало слова + 0,1 с внутри [начало, конец)."""
    field = {"share": 0.35, "sec": 7.0, **({"spans": [list(x) for x in spans]} if spans is not None else {})}
    ws = [{"word_id": "w0", "w": "начало", "s": 1.0, "e": 1.5, "c": 1.0},
          {"word_id": "w1", "w": "оригинал", "s": 4.8, "e": 6.0, "c": 1.0},
          {"word_id": "w3", "w": "конец", "s": 8.0, "e": 8.5, "c": 1.0}]
    return {"recording_id": "rec", "source_revision": 1, "speaker_names": {}, "review": list(review),
            "file": "s.wav", "duration": "00:00:10", "speakers": [S1],
            "quality": {"overlap_unmeasured_skipped": field},
            "segments": [{"segment_id": "g1", "speaker": S1, "start": 0.0, "end": 9.0,
                          "text": "начало оригинал конец", "words": ws}]}


def split_word(data, wid, text, item="span@1"):
    """Штатное разбиение слова wid на слова text (как ответ или правка отрезка: _diff_words + _relink_places)."""
    seg = data["segments"][0]
    k = next(i for i, w in enumerate(seg["words"]) if w["word_id"] == wid)
    old = [seg["words"][k]]
    origin = []
    new, conflicts, _ = transcript._diff_words(old, text, item, origin=origin)
    assert not conflicts
    transcript._relink_places(data, old, origin)
    seg["words"][k:k + 1] = new
    seg["text"] = " ".join(w["w"] for w in seg["words"])
    return [w["word_id"] for w in new]


def policy_copy(obj):
    return dict(obj, quote_policy=1)


class SplitWordQuoteTests(unittest.TestCase):
    """А255-1 (аудит OpenCode №255): сохранённая ссылка на слово, разбитое на части, печатает все части —
    блокировки (непроверенные отрезки, открытые места, правки человека) проверяются по всем напечатанным словам,
    в обеих разновидностях конспекта; строка «в кавычках / без проверки» совпадает с напечатанным."""

    TEXT = "Итог: Спикер 1: «оригинал»."

    def render_both(self, obj, data):
        md, rep = conspect.render(obj, data, update_last=False)
        pmd, prep = conspect.render(policy_copy(obj), data, update_last=False)
        return md, rep, pmd

    def assert_header_matches(self, pmd):
        """Строка «Цитат: N в кавычках, M без проверки» — по тому, что напечатано в теле."""
        head, body = pmd.split("Итог:", 1)
        quoted = body.count("«")
        marked = body.count("(по распознаванию") + body.count("(место в стенограмме не найдено)")
        self.assertIn(f"Цитат: {quoted} в кавычках, {marked} без проверки", head)

    def assert_blocked(self, md, rep, pmd, text):
        self.assertNotIn(f"«{text}»", md)
        self.assertIn(f"Спикер 1: {text} (по распознаванию, не проверено, 00:00:04)", md)
        self.assertEqual(len(rep["overlap_unchecked"]), 1)
        self.assertNotIn(f"«{text}»", pmd)
        self.assertIn(f"{text} (по распознаванию, не проверено, 00:00:04)", pmd)
        self.assert_header_matches(pmd)
        self.assertIn("Цитат: 0 в кавычках, 1 без проверки", pmd)

    def assert_quoted(self, md, rep, pmd, text):
        self.assertIn(f"Спикер 1: «{text}»", md)
        self.assertNotIn("overlap_unchecked", rep)
        self.assertIn(f"«{text}»", pmd)
        self.assert_header_matches(pmd)
        self.assertIn("Цитат: 1 в кавычках, 0 без проверки", pmd)

    def test_split_after_link_checks_every_printed_part(self):
        """Ссылка на w1 сохранена до разбиения; «альфа» 4,8 с — вне отрезка 5,4–7,0, «бета» 5,4 с — внутри."""
        data = split_data()
        obj, rep = conspect.link(self.TEXT, data)
        self.assertEqual(obj["quotes"]["1"]["parts"][0]["word_ids"], ["w1"])
        self.assert_quoted(*self.render_both(obj, data), "оригинал")       # до разбиения слово вне отрезка
        split_word(data, "w1", "альфа бета")
        self.assertEqual(conspect._id_text(data, ["w1"])[0], "альфа бета")
        self.assertEqual(workflow.overlap_blockers(data, ["w1"]), [])      # первая часть одна — чиста
        self.assert_blocked(*self.render_both(obj, data), "альфа бета")

    def test_parts_on_both_sides_of_span_boundary(self):
        """Граница отрезка между частями: задета любая часть — без кавычек; ни одна — в кавычках."""
        for spans, blocked in (([5.4, 7.0], True), ([4.5, 5.3], True), ([7.0, 7.5], False), ([5.6, 7.0], False)):
            with self.subTest(spans=spans):
                data = split_data(spans=(spans,))
                obj, _ = conspect.link(self.TEXT, data)
                split_word(data, "w1", "альфа бета")
                result = self.render_both(obj, data)
                (self.assert_blocked if blocked else self.assert_quoted)(*result, "альфа бета")

    def test_split_before_link(self):
        """Слово разбито до связывания: ссылка сразу на обе части; check_quotes даёт сомнение по второй."""
        data = split_data()
        split_word(data, "w1", "альфа бета")
        obj, rep = conspect.link("Итог: Спикер 1: «альфа бета».", data)
        self.assertEqual(rep["linked"], 1)
        self.assertEqual(len(obj["quotes"]["1"]["parts"][0]["word_ids"]), 2)
        self.assert_blocked(*self.render_both(obj, data), "альфа бета")
        check = quotes.check("Спикер 1: «альфа бета»", data)
        self.assertEqual(check["summary"]["in_doubt"], 1)
        self.assertEqual(next(r for r in check["problems"] if r.get("doubt"))["doubt"], ["overlap_skipped:0"])

    def test_repeated_edit_of_a_part(self):
        """Повторная правка: «бета» разбита ещё раз — части помнят корень w1; в отрезке только «гамма»."""
        data = split_data(spans=((5.75, 7.0),))
        obj, _ = conspect.link(self.TEXT, data)
        _, beta = split_word(data, "w1", "альфа бета")
        self.assert_quoted(*self.render_both(obj, data), "альфа бета")
        split_word(data, beta, "бета гамма", item="span@2")
        self.assertEqual(conspect._id_text(data, ["w1"])[0], "альфа бета гамма")
        self.assert_blocked(*self.render_both(obj, data), "альфа бета гамма")

    def test_human_touch_counts_per_printed_part(self):
        """Правка человека снимает отрезок только со своего слова: тронута первая часть — цитата всё равно
        без кавычек; тронута часть в отрезке — в кавычках."""
        for touched, blocked in ((0, True), (1, False)):
            with self.subTest(touched=touched):
                data = split_data()
                obj, _ = conspect.link(self.TEXT, data)
                ids = split_word(data, "w1", "альфа бета")
                word = next(w for w in data["segments"][0]["words"] if w["word_id"] == ids[touched])
                word["by"] = "human"
                result = self.render_both(obj, data)
                (self.assert_blocked if blocked else self.assert_quoted)(*result, "альфа бета")

    def test_open_place_on_a_later_part_blocks_policy_quote(self):
        """Открытое место на второй части (не на сохранённом ID) задевает цитату свёртки «цитаты по одной»."""
        data = split_data(spans=None)
        obj, _ = conspect.link(self.TEXT, data)
        ids = split_word(data, "w1", "альфа бета")
        data["review"] = [{"id": "r9", "status": "open", "reasons": ["low_conf"], "t": 5.4, "t1": 5.4, "t2": 6.0,
                           "speaker": S1, "word_ids": [ids[1]]}]
        self.assertEqual(workflow.quote_blockers(data, ["w1"]), [])
        pmd, _ = conspect.render(policy_copy(obj), data, update_last=False)
        self.assertIn("альфа бета (по распознаванию, не проверено, 00:00:04)", pmd)
        self.assert_header_matches(pmd)
        self.assertIn("Цитат: 0 в кавычках, 1 без проверки", pmd)

    def test_old_links(self):
        """Старые ссылки: запись до Н28-1 (без отрезков) — как раньше, в кавычках; ссылка, которую раскрыть
        однозначно нельзя (часть слова ушла в слово другого корня), в «цитатах по одной» — без кавычек."""
        data = split_data(spans=None)
        obj, _ = conspect.link(self.TEXT, data)
        split_word(data, "w1", "альфа бета")
        md, rep, pmd = self.render_both(obj, data)
        self.assertIn("Спикер 1: «альфа бета»", md)
        self.assertIn("Цитат: 1 в кавычках, 0 без проверки", pmd)
        data = split_data()
        obj, _ = conspect.link(self.TEXT, data)
        split_word(data, "w1", "альфа бета")
        data["segments"][0]["words"][2]["merged_from"] = ["w1"]
        self.assertEqual(conspect._id_text(data, ["w1"]), (None, None))
        pmd, prep = conspect.render(policy_copy(obj), data, update_last=False)
        self.assertNotIn("«", pmd.split("Итог:", 1)[1])
        self.assertEqual(len(prep["missing"]), 1)
        self.assert_header_matches(pmd)

    # --- О255-1, О255-2 (первый приём OpenCode №255): ссылка на часть разбитого слова ---------------------------

    def link_after_split(self, quote, spans=((5.65, 7.0),)):
        """Слово w1 разбито на «альфа» (w1, 4,8 с) и «бета» (новый ID, part_of=w1, 5,4 с); ссылка — на quote."""
        data = split_data(spans=spans)
        ids = split_word(data, "w1", "альфа бета")
        obj, rep = conspect.link(f"Итог: Спикер 1: «{quote}».", data)
        self.assertEqual(rep["linked"], 1)
        return data, obj, ids

    def assert_not_found(self, md, rep, pmd, text):
        """Слова ссылки нельзя раскрыть однозначно: кавычек и «проверенной» цитаты нет, место названо в отчёте."""
        self.assertEqual(len(rep["missing"]), 1)
        self.assertNotIn(f"«{text}»", md)
        self.assertIn(f"«{text} [место в стенограмме не найдено]»", md)
        self.assertNotIn("«", pmd.split("Итог:", 1)[1])
        self.assertIn(f"{text} (место в стенограмме не найдено)", pmd)
        self.assert_header_matches(pmd)
        self.assertIn("Цитат: 0 в кавычках, 1 без проверки", pmd)

    def test_o255_1_link_to_second_part_then_part_split_again(self):
        """О255-1, воспроизведение на штатной _diff_words: ссылка только на «бета» (part_of=w1); «бета» разбита ещё
        раз («бета гамма»), у «гаммы» часть того же корня w1, не ID «беты», и «гамма» лежит в отрезке наложения.
        Прежде ссылка раскрывалась только в «бета», «гамма» не проверялась. Теперь — неоднозначно, без кавычек."""
        data, obj, ids = self.link_after_split("бета")
        beta = ids[1]
        self.assertEqual(obj["quotes"]["1"]["parts"][0]["word_ids"], [beta])
        self.assertEqual(obj["quotes"]["1"]["parts"][0]["family"], ids)            # снимок семьи при связывании
        self.assertEqual(conspect._id_words(data, [beta], ids), ("бета", S1, [beta]))
        self.assert_quoted_at(self.render_both(obj, data), "бета")                   # пока не тронута — чисто
        split_word(data, beta, "бета гамма", item="span@2")
        words = {w["w"]: w for w in data["segments"][0]["words"]}
        self.assertEqual(words["гамма"]["part_of"], "w1")                           # корень, а не ID «беты»
        self.assertEqual(conspect._id_words(data, [beta], ids), (None, None, None))
        self.assert_not_found(*self.render_both(obj, data), "бета")

    def test_o255_1_link_to_second_part_then_part_replaced(self):
        """О255-1 с заменой: «бета» заменена двумя словами «дельта эпсилон» (первое наследует ID «беты»); «эпсилон»
        в отрезке. Прежде в кавычках выходило «дельта» — урезанная цитата; теперь место не найдено."""
        data, obj, ids = self.link_after_split("бета")
        split_word(data, ids[1], "дельта эпсилон", item="span@1")
        self.assertEqual([w["w"] for w in data["segments"][0]["words"]], ["начало", "альфа", "дельта", "эпсилон", "конец"])
        self.assert_not_found(*self.render_both(obj, data), "бета")

    def test_o255_1_other_part_split_is_not_ours_but_unclear(self):
        """Ссылка на «бета», а «альфу» (корень w1) заменяют двумя словами: новое слово — тоже член семьи w1. По
        записи не определить, чьё оно, — неоднозначно, кавычек нет (осторожно: цитата «бета» от этого не менялась)."""
        data, obj, ids = self.link_after_split("бета")
        split_word(data, "w1", "эта дельта", item="span@1")
        self.assert_not_found(*self.render_both(obj, data), "бета")

    def test_second_part_link_without_later_edits(self):
        """Контроль: ссылка на вторую часть, пока части не трогали, — как всегда: проверяются слова ссылки, и
        только они. Отрезок на первой части цитату «бета» не задевает, на самой «бета» — задевает."""
        data, obj, ids = self.link_after_split("бета", spans=((4.5, 5.3),))      # отрезок только на «альфа»
        self.assert_quoted_at(self.render_both(obj, data), "бета")
        data, obj, ids = self.link_after_split("бета", spans=((5.35, 5.6),))     # «бета»: 5,4 + 0,1 = 5,5 внутри
        md, rep, pmd = self.render_both(obj, data)
        self.assertIn("Спикер 1: бета (по распознаванию, не проверено, 00:00:05)", md)
        self.assertIn("бета (по распознаванию, не проверено, 00:00:05)", pmd)
        self.assert_header_matches(pmd)

    def test_second_part_link_that_covers_the_root_follows_the_root(self):
        """Ссылка на обе части («альфа бета», корень w1 выбран): новые части корня входят, как и раньше; гамма в
        отрезке — цитата без кавычек, напечатано «альфа бета гамма»."""
        data, obj, ids = self.link_after_split("альфа бета")
        self.assertEqual(obj["quotes"]["1"]["parts"][0]["word_ids"], ids)
        split_word(data, ids[1], "бета гамма", item="span@2")
        self.assert_blocked(*self.render_both(obj, data), "альфа бета гамма")

    def test_o255_2_link_to_first_part_does_not_pull_in_the_neighbour(self):
        """О255-2, воспроизведение: ссылка только на «альфа» после разбиения (ID корня w1); отрезок — только на
        «бета». Прежде w1 раскрывался и на «бета» — чистая цитата «альфа» теряла кавычки. Теперь печатается и
        проверяется ровно «альфа»."""
        data, obj, ids = self.link_after_split("альфа", spans=((5.5, 7.0),))
        self.assertEqual(ids[0], "w1")
        self.assertEqual(obj["quotes"]["1"]["base"], "альфа")                        # не «альфа бета»
        self.assertEqual(conspect._id_words(data, ["w1"], obj["quotes"]["1"]["parts"][0]["family"]),
                         ("альфа", S1, ["w1"]))
        self.assert_quoted_at(self.render_both(obj, data), "альфа")
        # отрезок на самой «альфа» — цитата без кавычек, время — слова «альфа»
        data, obj, ids = self.link_after_split("альфа", spans=((4.5, 5.0),))
        md, rep, pmd = self.render_both(obj, data)
        self.assertIn("Спикер 1: альфа (по распознаванию, не проверено, 00:00:04)", md)
        self.assertIn("альфа (по распознаванию, не проверено, 00:00:04)", pmd)
        self.assert_header_matches(pmd)

    def test_o255_2_open_place_on_neighbour_does_not_block_policy_quote(self):
        """О255-2 для «цитат по одной»: открытое место на соседней части не снимает кавычки; на самой цитируемой —
        снимает."""
        for on_word, blocked in ((1, False), (0, True)):
            with self.subTest(place_on_part=on_word):
                data, obj, ids = self.link_after_split("альфа", spans=None)
                data["review"] = [{"id": "r9", "status": "open", "reasons": ["low_conf"], "t": 5.0, "t1": 5.0, "t2": 6.0,
                                   "speaker": S1, "word_ids": [ids[on_word]]}]
                pmd, _ = conspect.render(policy_copy(obj), data, update_last=False)
                body = pmd.split("Итог:", 1)[1]
                self.assertEqual("«альфа»" in body, not blocked)
                self.assertEqual("(по распознаванию, не проверено" in body, blocked)
                self.assert_header_matches(pmd)

    def test_o255_2_link_to_first_part_follows_its_own_later_split(self):
        """Ссылка на «альфа» (после разбиения), потом саму «альфу» заменяют двумя словами «эта дельта»: новая часть
        «дельта» — часть «альфы», в цитате и в проверке; соседняя «бета» по-прежнему вне цитаты."""
        data, obj, ids = self.link_after_split("альфа", spans=None)
        split_word(data, "w1", "эта дельта", item="span@1")
        delta = next(w["word_id"] for w in data["segments"][0]["words"] if w["w"] == "дельта")
        self.assertEqual(conspect._id_words(data, ["w1"], obj["quotes"]["1"]["parts"][0]["family"])[2], ["w1", delta])
        for place, blocked in (({"word_ids": [ids[1]]}, False), ({"word_ids": [delta]}, True)):
            with self.subTest(place=place):
                data["review"] = [{"id": "r9", "status": "open", "reasons": ["low_conf"], "t": 5.0, "t1": 5.0, "t2": 6.0,
                                   "speaker": S1, **place}]
                pmd, rep = conspect.render(policy_copy(obj), data, update_last=False)
                body = pmd.split("Итог:", 1)[1]
                self.assertEqual("(по распознаванию, не проверено" in body, blocked)
                self.assertIn("эта дельта", body)                                    # текст изменился — выводится новый
                self.assertNotIn("бета", body)
                self.assert_header_matches(pmd)

    def test_s255_2_place_on_a_root_id_missing_from_the_transcript(self):
        """С255-2 (проверка Sonnet, S9): место разбора ссылается на корневой ID w1, слова с этим ID в стенограмме
        уже нет (остались части), поэтому раскрытый набор его не содержит. Блокировка мест идёт по объединению
        сохранённых и напечатанных ID — цитата свёртки без кавычек; отрезки наложений — по-прежнему только по
        напечатанным словам (О255-2)."""
        data = split_data(spans=None)
        obj, _ = conspect.link(self.TEXT, data)
        data["segments"][0]["words"][1] = {"word_id": "p1", "w": "оригинал", "part_of": "w1", "s": 4.8, "e": 6.0, "c": 1.0}
        data["review"] = [{"id": "r9", "status": "open", "reasons": ["low_conf"], "t": 5.4, "t1": 5.4, "t2": 6.0,
                           "speaker": S1, "word_ids": ["w1"]}]
        self.assertEqual(conspect._id_words(data, ["w1"]), ("оригинал", S1, ["p1"]))
        self.assertEqual(workflow.quote_blockers(data, ["p1"]), [])                  # по раскрытому — пусто (как в S9)
        pmd, _ = conspect.render(policy_copy(obj), data, update_last=False)
        self.assertIn("оригинал (по распознаванию, не проверено, 00:00:04)", pmd)
        self.assertNotIn("«", pmd.split("Итог:", 1)[1])
        self.assert_header_matches(pmd)
        self.assertIn("Цитат: 0 в кавычках, 1 без проверки", pmd)

    def test_quote_blockers_overlap_ids_limit_only_the_spans(self):
        """overlap_ids сужает проверку отрезков, но не мест: слово w0 лежит в отрезке, w1 — нет."""
        data = split_data(spans=((0.5, 2.0),))
        data["review"] = [{"id": "r9", "status": "open", "reasons": ["low_conf"], "t": 5.4, "t1": 5.4, "t2": 6.0,
                           "speaker": S1, "word_ids": ["w3"]}]
        self.assertEqual([b["id"] for b in workflow.quote_blockers(data, ["w1", "w0"])], ["overlap_skipped:0"])
        self.assertEqual(workflow.quote_blockers(data, ["w1", "w0"], overlap_ids=["w1"]), [])
        self.assertEqual([b["id"] for b in workflow.quote_blockers(data, ["w1", "w3"], overlap_ids=["w1"])], ["r9"])

    def test_old_link_to_a_part_without_family_snapshot(self):
        """Ссылка без снимка семьи (связана до этой правки): часть без новых слов рядом — как раньше, в кавычках;
        сразу за частью идёт слово её семьи (новая часть или сосед) — неоднозначно, без кавычек."""
        data, obj, ids = self.link_after_split("бета", spans=None)
        del obj["quotes"]["1"]["parts"][0]["family"]
        self.assert_quoted_at(self.render_both(obj, data), "бета")
        split_word(data, ids[1], "бета гамма", item="span@2")
        self.assert_not_found(*self.render_both(obj, data), "бета")

    def test_family_snapshot_only_when_words_were_split(self):
        """Снимок семьи пишется, только если слова уже разбиты: обычная ссылка на неразбитое слово — как раньше."""
        data = split_data()
        obj, _ = conspect.link(self.TEXT, data)
        self.assertEqual(set(obj["quotes"]["1"]["parts"][0]), {"t1", "t2", "word_ids"})

    def assert_quoted_at(self, result, text):
        md, rep, pmd = result
        self.assertIn(f"Спикер 1: «{text}»", md)
        self.assertNotIn("overlap_unchecked", rep)
        self.assertEqual(rep["missing"], [])
        self.assertIn(f"«{text}»", pmd)
        self.assert_header_matches(pmd)
        self.assertIn("Цитат: 1 в кавычках, 0 без проверки", pmd)

    def test_end_to_end_words_link_edit_render(self):
        """Сквозной путь на файлах: стенограмма → conspect.save_linked (ссылка) → transcript.edit_span
        (агент разбивает слово) → conspect.rerender (сборка) — обе разновидности конспекта и check_quotes."""
        with tempfile.TemporaryDirectory(dir=_TMP) as folder:
            jp = Path(folder) / "s.transcript.json"
            md_path = Path(folder) / "s.md"
            data = split_data()
            transcript.save(data, jp, md_path)
            saved = conspect.save_linked(jp, "Агент", self.TEXT)
            self.assertEqual(saved["linked"], 1)
            self.assertIn("Спикер 1: «оригинал»", Path(saved["conspect_md"]).read_text(encoding="utf-8"))
            policy_obj = json.loads(Path(saved["conspect_json"]).read_text(encoding="utf-8"))
            policy_obj.update(name="Свёртка", quote_policy=1)
            pj, pm = conspect.path_for(jp, "Свёртка")
            pj.write_text(json.dumps(policy_obj, ensure_ascii=False), encoding="utf-8")
            transcript.edit_span(jp, md_path, 4.7, 5.0, "альфа бета", by="claude")
            now = transcript.load(jp)
            self.assertEqual([w["w"] for w in now["segments"][0]["words"]], ["начало", "альфа", "бета", "конец"])
            out = conspect.rerender(jp)
            agent_md = Path(out["Агент"]["conspect_md"]).read_text(encoding="utf-8")
            policy_md = Path(out["Свёртка"]["conspect_md"]).read_text(encoding="utf-8")
            self.assertNotIn("«альфа бета»", agent_md)
            self.assertIn("Спикер 1: альфа бета (по распознаванию, не проверено, 00:00:04)", agent_md)
            self.assertNotIn("«альфа бета»", policy_md)
            self.assert_header_matches(policy_md)
            self.assertIn("Цитат: 0 в кавычках, 1 без проверки", policy_md)
            self.assertEqual(out["Агент"]["check"]["verbatim"], 0)
            self.assertEqual(quotes.check("Спикер 1: «альфа бета»", now)["summary"]["in_doubt"], 1)


if __name__ == "__main__":
    unittest.main()
