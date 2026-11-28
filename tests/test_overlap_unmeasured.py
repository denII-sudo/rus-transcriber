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
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

_TMP = tempfile.mkdtemp(prefix="rt-overlap-unmeasured-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP

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
        self.assertIn("одновременную речь на 75 % речи — похоже на эхо, гул или долгую одновременную речь", md)
        self.assertIn("Если говорили одновременно — прослушайте: 00:00:01–00:00:08, 00:00:11–00:00:19.", md)

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
                self.assertNotIn("похоже на эхо", md)

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
        self.assertIn("прослушайте: 00:00:00–00:00:02, 00:00:10–00:00:12, 00:00:20–00:00:22, "
                      "00:00:30–00:00:32, 00:00:40–00:00:42 и ещё 2.", note)
        self.assertTrue(transcript.overlap_skipped_note({"share": 0.5, "sec": 1.0}).endswith("прослушайте."))
        for bad in ([[2.0, 1.0]], [[1.0]], "x", [[1.0, float("nan")]], [[True, 2.0]]):
            with self.subTest(bad=bad):
                info = transcript.overlap_skipped_info({"share": 0.5, "sec": 1.0, "spans": bad})
                self.assertEqual(info, {"share": 0.5, "sec": 1.0})

    def test_status_summary_caps_spans(self):
        """Н28-2: сводка quality статуса задачи несёт поле; отрезков — не больше десяти и число остальных."""
        spans = [[float(k), k + 0.5] for k in range(12)]
        out = worker._overlap_status({"overlap_unmeasured_skipped": {"share": 0.5, "sec": 6.0, "spans": spans}})
        self.assertEqual(out, {"share": 0.5, "sec": 6.0, "spans": spans[:10], "spans_more": 2})
        self.assertIsNone(worker._overlap_status({}))

    def test_doubt_flags_switch_touches_only_unmeasured(self):
        """Прямо doubt_flags: unmeasured_overlap=False снимает только пометку неизмеренных слов."""
        unmeasured = [dict(w, spk=S1, margin=1.0, margin_unmeasured=True) for w in words_between(0.0, 5.0)]
        close = [dict(w, spk=S2, margin=0.01) for w in words_between(10.0, 15.0)]
        W.doubt_flags(unmeasured + close, [(1.0, 4.0), (11.0, 14.0)], unmeasured_overlap=False)
        self.assertEqual(flagged(unmeasured, "overlap"), [])
        self.assertEqual(flagged(close, "overlap"), inside(close, 11.0, 14.0))


if __name__ == "__main__":
    unittest.main()


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
