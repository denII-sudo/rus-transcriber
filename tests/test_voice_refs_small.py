"""Опорные голоса без профиля (Т-1-Н5, Т-1-Н1): короткая запись без опор и короткий голос без опоры.

Только синтетика: звук — тишина, отпечаток голоса подменён по времени (голос A до границы, B после),
модели не загружаются. Работают настоящие worker._voice_references, worker._without_reference,
words.assign_voices, words.build_replies, words.doubt_flags и voices._spans.

Было: опор нет — np.stack падает (ValueError, стенограммы нет); голос короче 10 с без профиля молча
отдавался ближайшей опоре. Стало: опора — неопознанный кластер от 3 с с центром; слова голосов без опоры
остаются под меткой разметки pyannote (голос назначает программа, как диаризация), новых пометок
сомнения это не добавляет; в профиль такие слова не идут, пока человек не подтвердит имя.
"""
import atexit
import inspect
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

_TMP = tempfile.mkdtemp(prefix="rt-voice-refs-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP

import voices  # noqa: E402
import words as W  # noqa: E402
import worker  # noqa: E402

VA = np.array([1.0, 0.0, 0.0])
VB = np.array([0.0, 1.0, 0.0])
S1, S2 = "Спикер 1", "Спикер 2"


class FakeEmb:
    """Отпечаток отрезка — доля голоса B по времени (граница border), нормированный."""

    def __init__(self, border):
        self.border = border
        self.seconds = 0.0

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


def turn(s, e, label):
    return {"s": s, "e": e, "label": label}


def words_in(turns, step=0.4):
    out = []
    for t in turns:
        x = t["s"]
        while x + 0.3 <= t["e"]:
            out.append({"w": "слово", "s": round(x, 2), "e": round(x + 0.3, 2), "c": 1.0, "truth": t["label"]})
            x += step
    return out


def run(turns, border):
    """Как run_words: опоры, метки без опоры, голос каждого слова. -> (слова, опоры, метки без опоры)."""
    dur = max(t["e"] for t in turns) + 1.0
    audio = np.zeros(int(dur * W.SR), dtype=np.float32)
    emb = FakeEmb(border)
    refs, found = worker._voice_references(audio, turns, emb, {}, {}, [])
    # на коде до исправления нет _without_reference и ключа unreferenced: тот же тест там падает на сути
    # (np.stack, слова чужой опоре), а не на импорте
    unref = worker._without_reference(turns, {}, refs) if hasattr(worker, "_without_reference") else set()
    extra = {"unreferenced": unref} if "unreferenced" in inspect.signature(W.assign_voices).parameters else {}
    ws = words_in(turns)
    prior = W.LabelAt([(t["s"], t["e"], t["label"]) for t in turns])
    W.assign_voices(audio, ws, refs, emb, prior=prior, info={"found": found}, **extra)
    return ws, refs, unref


def by_truth(ws):
    out = {}
    for w in ws:
        out.setdefault(w["truth"], set()).add(w["spk"])
    return out


A20 = [turn(0.0, 10.0, S1), turn(10.5, 20.5, S1)]


class NoReferenceTests(unittest.TestCase):
    """Т-1-Н5: ни у одного голоса нет опоры."""

    def test_short_single_voice_gets_reference(self):
        """Короткое голосовое: один голос, 4,2 с (сцена Лаборатории s11). Было — опор нет, ValueError в np.stack;
        стало — кластер от 3 с с центром и есть опора."""
        ws, refs, unref = run([turn(0.5, 2.7, S1), turn(3.5, 5.5, S1)], 99.0)
        self.assertEqual((sorted(refs), unref), ([S1], set()))
        self.assertEqual(by_truth(ws), {S1: {S1}})

    def test_very_short_single_voice_is_transcribed(self):
        """Один голос, 2,4 с — ниже порога, опор нет. Было — ValueError в np.stack."""
        ws, refs, unref = run([turn(0.5, 2.9, S1)], 99.0)
        self.assertEqual((refs, unref), ({}, {S1}))
        self.assertEqual(by_truth(ws), {S1: {S1}})
        self.assertTrue(all(w["margin"] == 1.0 and "sims" not in w for w in ws))
        W.doubt_flags(ws, [])
        self.assertEqual([w for w in ws if w.get("flags")], [])             # новых пометок и карточек нет
        self.assertEqual([r["spk"] for r in W.build_replies(ws)], [S1])

    def test_two_short_voices_keep_diarization_labels(self):
        """Реплики двух голосов короче 1,5 с (центров нет): голоса — по разметке, не слиты в один."""
        ws, refs, _ = run([turn(0.0, 1.4, S1), turn(1.6, 3.0, S2), turn(3.2, 4.6, S1)], 1.5)
        self.assertEqual(refs, {})
        self.assertEqual(by_truth(ws), {S1: {S1}, S2: {S2}})
        W.doubt_flags(ws, [])
        self.assertEqual([w for w in ws if w.get("flags")], [])

    def test_word_outside_diarization_takes_previous_voice(self):
        audio = np.zeros(8 * W.SR, dtype=np.float32)
        ws = [{"w": "а", "s": 0.2, "e": 0.5}, {"w": "б", "s": 5.0, "e": 5.3}, {"w": "в", "s": 0.6, "e": 0.9}]
        W.assign_voices(audio, ws, {}, FakeEmb(99.0), prior=W.LabelAt([(0.0, 1.0, S2)]))
        self.assertEqual([w["spk"] for w in ws], [S2, S2, S2])
        ws = [{"w": "а", "s": 0.2, "e": 0.5}]
        W.assign_voices(audio, ws, {}, FakeEmb(99.0), prior=None)
        self.assertEqual(ws[0]["spk"], S1)


class ShortVoiceTests(unittest.TestCase):
    """Т-1-Н1: голос без профиля, у которого меньше 10 с речи."""

    def test_six_seconds_get_their_own_reference(self):
        """A 20 с, B 6 с одним кластером. Было — все слова B у Спикера 1."""
        ws, refs, unref = run(A20 + [turn(21.0, 24.0, S2), turn(24.5, 27.5, S2)], 21.0)
        self.assertEqual((sorted(refs), unref), ([S1, S2], set()))
        self.assertEqual(by_truth(ws), {S1: {S1}, S2: {S2}})

    def test_voice_without_centre_keeps_its_label(self):
        """B — реплики короче 1,5 с (центра нет, опоры нет): слова B не отдаются Спикеру 1, новых пометок нет."""
        ws, refs, unref = run(A20 + [turn(21.0, 22.2, S2), turn(22.6, 23.8, S2)], 21.0)
        self.assertEqual((sorted(refs), unref), ([S1], {S2}))
        self.assertEqual(by_truth(ws), {S1: {S1}, S2: {S2}})
        W.doubt_flags(ws, [])
        self.assertEqual([w for w in ws if w.get("flags")], [])

    def test_short_cluster_below_threshold_keeps_its_label(self):
        """B — одна реплика 2,4 с (центр есть, но меньше порога 3 с): опоры нет, метка остаётся."""
        ws, refs, unref = run(A20 + [turn(21.0, 23.4, S2)], 21.0)
        self.assertEqual((sorted(refs), unref), ([S1], {S2}))
        self.assertEqual(by_truth(ws), {S1: {S1}, S2: {S2}})

    def test_long_voices_unchanged(self):
        """Контроль: A 20 с, B 10,5 с — как до исправления: обе опоры, слова своим."""
        ws, refs, unref = run(A20 + [turn(21.0, 26.0, S2), turn(26.5, 32.0, S2)], 21.0)
        self.assertEqual((sorted(refs), unref), ([S1, S2], set()))
        self.assertEqual(by_truth(ws), {S1: {S1}, S2: {S2}})

    @unittest.skipUnless(hasattr(worker, "_without_reference"), "до исправления функции нет")
    def test_identified_cluster_is_not_without_reference(self):
        self.assertEqual(worker._without_reference([turn(0, 5, S1), turn(5, 6, S2)], {S2: "Дима"}, {"Дима": VB}),
                         {S1})


class ProfileTests(unittest.TestCase):
    """Слова голоса без опоры не попадают в профиль сами: только после имени, подтверждённого человеком."""

    def test_label_without_name_is_not_learned(self):
        ws, _, _ = run(A20 + [turn(21.0, 22.2, S2), turn(22.6, 23.8, S2)], 21.0)
        segments = [{"speaker": r["spk"], "start": r["words"][0]["s"], "end": r["words"][-1]["e"],
                     "words": [{k: w[k] for k in ("w", "s", "e")} for w in r["words"]]}
                    for r in W.build_replies(ws)]
        self.assertEqual([s["speaker"] for s in segments], [S1, S2])
        report = {}
        self.assertEqual(voices._spans({"segments": segments, "names_confirmed": True}, True, report), [])
        self.assertEqual(report.get("unknown_speaker"), 2)


if __name__ == "__main__":
    unittest.main()
