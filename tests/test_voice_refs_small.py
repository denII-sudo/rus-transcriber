"""Опорные голоса без профиля (Т-1-Н5, Т-1-Н1): короткая запись без опор и короткий голос без опоры.

Только синтетика: звук — тишина, отпечаток голоса подменён по времени (голос A до границы, B после),
модели не загружаются. Работают настоящие worker._voice_references, worker._without_reference,
words.assign_voices, words.build_replies, words.doubt_flags и voices._spans; RunWordsTests — настоящий
worker.run_words, у которого подменены только проходы моделей, места для разбора, слой разговора и эмоции.

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
from types import SimpleNamespace
from unittest import mock

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
        self.assertEqual([w for w in ws if w["truth"] == S2 and "sims" in w], [])    # чужих сходств нет
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


class _Status:
    def set(self, **_):
        pass


def run_words(raw, label_name, pvecs, cents, allowed, border, extra_words=()):
    """Настоящий worker.run_words: проходы моделей подменены словами по репликам raw. -> данные расшифровки."""
    order = {}
    for _, _, spk in sorted(raw):
        order.setdefault(spk, f"Спикер {len(order) + 1}")
    turns = [turn(s, e, order[spk]) for s, e, spk in worker.build_turns(raw)]      # как в worker.main
    ws = words_in(turns) + [dict(w) for w in extra_words]
    ws.sort(key=lambda w: w["s"])
    dur = max(e for _, e, _ in raw) + 1.0
    audio = np.zeros(int(dur * W.SR), dtype=np.float32)
    mapping = {lbl: (nm, 0.9) for lbl, nm in label_name.items()}
    voice_ctx = {"pvecs": pvecs, "cents": cents, "mapping": mapping, "confusions": {}}
    quality = {"vocab": None, "speaker_fixes": [], "vocab_fixes": [], "review": [], "asr_diagnostics": {}}
    fused = {"quality": {}, "evidence": {"whisper": [], "other": []}}
    with mock.patch.object(worker, "primary_pass", return_value=ws),             mock.patch.object(worker, "second_pass", return_value=None),             mock.patch.object(worker, "whisper_plan", return_value={"model": None}),             mock.patch.object(worker.W, "combine_hypotheses", return_value=fused),             mock.patch.object(worker, "_check_gender"),             mock.patch.object(worker, "_review_places", return_value=([], 0.5, [])),             mock.patch.object(worker, "_interaction_layer"), mock.patch.object(worker, "_emotions"):
        return worker.run_words(audio, "synthetic.wav", {}, _Status(), list(raw), order, turns, None,
                                FakeEmb(border), dict(label_name), {nm: lbl for lbl, nm in label_name.items()},
                                voice_ctx, allowed, {}, None, quality, None, SimpleNamespace(status=""))


def spoken_by(data, t1, t2):
    """{(метка, подпись)} слов в отрезке [t1, t2)."""
    return {(seg["label"], seg["speaker"]) for seg in data["segments"] for w in seg["words"] if t1 <= w["s"] < t2}


VC = np.array([0.0, 0.0, 1.0])
RAW_A = [(0.0, 10.0, "SPEAKER_00"), (10.5, 20.5, "SPEAKER_00")]


class RunWordsTests(unittest.TestCase):
    """Через настоящий run_words, профили есть: участник из профиля без кластера; кластер без опоры — как до
    Т-1-Н1, по отпечатку к ближайшей опоре (метки «Спикер N» по разметке — только без профилей)."""

    def setUp(self):
        self.raw = RAW_A + [(21.0, 22.2, "SPEAKER_01"), (22.6, 23.8, "SPEAKER_01")]   # B: опоры нет
        self.label_name = {S1: "Анна"}
        self.cents = {S1: VA}

    def test_profile_name_gets_free_label(self):
        """Голос B — Борис (в профиле, своего опознанного кластера нет): его слова уходят Борису по отпечатку,
        и имя встаёт на свободную метку «Спикер 3», а не на метку «Спикер 2» кластера B. Речи у B 4 с: меньше
        3 с слова отдал бы другому правило слабого голоса (MIN_VOICE_SEC), как и до Т-1-Н1."""
        raw = RAW_A + [(21.0, 23.0, "SPEAKER_01"), (23.4, 25.4, "SPEAKER_01")]
        data = run_words(raw, self.label_name, {"Анна": VA, "Борис": VB}, self.cents, ["Анна", "Борис"], 21.0)
        self.assertEqual(spoken_by(data, 0.0, 20.5), {(S1, "Анна")})
        self.assertEqual(spoken_by(data, 21.0, 25.4), {("Спикер 3", "Борис")})
        self.assertEqual(data.get("speaker_names"), {S1: "Анна", "Спикер 3": "Борис"})
        self.assertNotIn("voice_refs", data["quality"])

    def test_tiny_cluster_word_goes_by_fingerprint(self):
        """Без профилей. Метка, у которой все отрезки короче MIN_SEG (0,3 с; часто ложный крошечный кластер
        pyannote): в turns её нет, и её слово, как до Т-1-Н1, получает голос по отпечатку — здесь голос A, —
        а не метку «Спикер 3» с отрывом 1,0, которая рвала бы реплику."""
        raw = self.raw + [(24.5, 24.75, "SPEAKER_02")]
        tiny = [{"w": "ага", "s": 24.5, "e": 24.7, "c": 1.0, "truth": S1}]
        data = run_words(raw, {}, {}, {}, [], 99.0, tiny)
        self.assertEqual(spoken_by(data, 24.4, 24.8), {(S1, S1)})
        self.assertEqual(data["quality"]["voice_refs"]["without_reference"], [S2])
        self.assertNotIn("Спикер 3", data["speaker_labels"])

    def test_profile_only_participant_gets_free_label(self):
        """Борис из профиля говорит, но своего кластера у него нет (pyannote слил его с Анной): его слова
        подписаны Борисом под свободной меткой, не под меткой голоса B."""
        raw = [(0.0, 10.0, "SPEAKER_00"), (10.5, 20.5, "SPEAKER_00"), (21.0, 22.2, "SPEAKER_01"),
               (22.6, 23.8, "SPEAKER_01"), (25.0, 31.0, "SPEAKER_00")]
        # отпечаток: до 25 с — голос Анны (и у кластера B: при профилях его слова идут по отпечатку), после — Борис
        data = run_words(raw, self.label_name, {"Анна": VA, "Борис": VB}, self.cents, ["Анна", "Борис"], 25.0)
        self.assertEqual(spoken_by(data, 21.0, 24.0), {(S1, "Анна")})
        self.assertEqual(spoken_by(data, 25.5, 31.0), {("Спикер 3", "Борис")})
        self.assertEqual(data.get("speaker_names"), {S1: "Анна", "Спикер 3": "Борис"})



VA_PROFILE = (VA + 0.6 * VC) / np.linalg.norm(VA + 0.6 * VC)     # профиль Анны снят с другой записи: канал иной


class ProfileShortClusterTests(unittest.TestCase):
    """Перепроверка Opus PR №243, находка 1: при профилях короткий неопознанный кусок речи участника не
    становится опорой (порог 10 с) и не уводит слова основного кластера участника под «Спикер N»; его
    собственные слова, как до Т-1-Н1, идут по отпечатку к ближайшей опоре (решение Архитектора разработки)."""

    def test_short_unmatched_part_does_not_take_participant_words(self):
        raw = RAW_A + [(21.0, 24.0, "SPEAKER_01"), (24.5, 27.5, "SPEAKER_01")]      # 6 с той же Анны
        data = run_words(raw, {S1: "Анна"}, {"Анна": VA_PROFILE}, {S1: VA, S2: VA}, ["Анна"], 99.0)
        self.assertEqual(spoken_by(data, 0.0, 20.5), {(S1, "Анна")})
        self.assertNotIn("voice_refs", data["quality"])

    def test_short_cluster_words_go_to_nearest_reference(self):
        """С профилем: слова короткого неопознанного кластера остаются у ближайшей опоры (как до Т-1-Н1)."""
        raw = RAW_A + [(21.0, 24.0, "SPEAKER_01"), (24.5, 27.5, "SPEAKER_01")]
        data = run_words(raw, {S1: "Анна"}, {"Анна": VA_PROFILE}, {S1: VA, S2: VA}, ["Анна"], 99.0)
        self.assertEqual(spoken_by(data, 21.0, 27.5), {(S1, "Анна")})
        self.assertEqual(data["speaker_labels"], [S1])

    def test_without_profiles_short_voice_still_gets_reference(self):
        """Без профилей порог прежний для Т-1-Н1 — 3 с (сцены Лаборатории без профилей)."""
        raw = RAW_A + [(21.0, 24.0, "SPEAKER_01"), (24.5, 27.5, "SPEAKER_01")]
        data = run_words(raw, {}, {}, {}, [], 21.0)
        self.assertEqual(spoken_by(data, 0.0, 20.5), {(S1, S1)})
        self.assertEqual(spoken_by(data, 21.0, 27.5), {(S2, S2)})
        self.assertNotIn("voice_refs", data["quality"])

    def test_threshold_counts_speech_not_merged_turns(self):
        """3 с — речь по отрезкам разметки, без пауз: 1,0 + 0,6 + 0,7 = 2,3 с, склеенная реплика 3,3 с."""
        raw = RAW_A + [(21.0, 22.0, "SPEAKER_01"), (22.5, 23.1, "SPEAKER_01"), (23.6, 24.3, "SPEAKER_01")]
        data = run_words(raw, {}, {}, {}, [], 21.0)
        self.assertEqual(data["quality"].get("voice_refs"),
                         {"refs": 1, "without_reference": [S2], "assigned_by": "diarization"})
        self.assertEqual(spoken_by(data, 21.0, 24.3), {(S2, S2)})


class DoubtThresholdTests(unittest.TestCase):
    """Перепроверка Opus PR №243, находка 3: неизмеренный отрыв не поднимает порог сомнения измеренных слов."""

    def test_unmeasured_margins_do_not_raise_threshold(self):
        def measured():
            return [{"w": "а", "s": i * 0.1, "e": i * 0.1 + 0.08, "spk": "A", "margin": i / 2000}
                    for i in range(200)]
        alone = measured()
        W.doubt_flags(alone, [])
        mixed = measured()
        extra = [{"w": "б", "s": 30 + i * 0.1, "e": 30 + i * 0.1 + 0.08} for i in range(100)]
        W._voice_by_diarization(extra, W.LabelAt([(29.0, 45.0, "B")]), {"B"})
        thr = W.doubt_flags(mixed + extra, [])
        self.assertEqual(sum("voice" in (w.get("flags") or []) for w in mixed),
                         sum("voice" in (w.get("flags") or []) for w in alone))
        self.assertLess(thr, 0.05)
        self.assertEqual([w for w in extra if w.get("flags")], [])


if __name__ == "__main__":
    unittest.main()
