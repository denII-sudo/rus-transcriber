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
        self.assertEqual(data["quality"]["overlap_unmeasured_skipped"], {"share": 0.75, "sec": 15.0})
        words_ = [w for seg in data["segments"] for w in seg["words"]]
        self.assertFalse(any("overlap" in (w.get("flags") or []) for w in words_))
        self.assertEqual({seg["label"] for seg in data["segments"]}, {S1})       # голос слов прежний
        md = transcript.render_md(dict(data, file="synthetic.wav", duration="00:00:21"))
        self.assertIn("одновременную речь на 75 % речи — похоже на эхо или гул", md)
        self.assertIn("Если говорили одновременно — прослушайте.", md)

    def test_share_at_threshold_keeps_places(self):
        """Ровно 30 % (6 с из 20 с) — ещё не эхо: места наложений ставятся, предупреждения нет."""
        data = run_words(RAW_ONE, RAW_ONE + [(2.0, 5.0, "SPEAKER_01"), (12.0, 15.0, "SPEAKER_01")], 99.0)
        self.assertEqual(len([it for it in data["review"] if "overlap" in it.get("reasons", [])]), 2)
        self.assertNotIn("overlap_unmeasured_skipped", data["quality"])

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

    def test_doubt_flags_switch_touches_only_unmeasured(self):
        """Прямо doubt_flags: unmeasured_overlap=False снимает только пометку неизмеренных слов."""
        unmeasured = [dict(w, spk=S1, margin=1.0, margin_unmeasured=True) for w in words_between(0.0, 5.0)]
        close = [dict(w, spk=S2, margin=0.01) for w in words_between(10.0, 15.0)]
        W.doubt_flags(unmeasured + close, [(1.0, 4.0), (11.0, 14.0)], unmeasured_overlap=False)
        self.assertEqual(flagged(unmeasured, "overlap"), [])
        self.assertEqual(flagged(close, "overlap"), inside(close, 11.0, 14.0))


if __name__ == "__main__":
    unittest.main()
