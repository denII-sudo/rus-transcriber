"""Любая модель на втором и третьем месте голосования (основная не меняется).

Модели — синтетические заглушки: настоящие модели не загружаются и не скачиваются. Логика
спрашивает возможности модели (transcript.asr_model_caps), а не имя «CTC» или «Whisper»:
порядок слов по времени после голосования, нарезка кусков третьей модели, чистка текста только
у Whisper, подписи для человека, вторая модель без времён слов, одинаковые модели, выбор движка.
"""
import atexit
import contextlib
import copy
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

if "paths" not in sys.modules:          # папка данных — временная, до первого импорта программы
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-models-data-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backends  # noqa: E402
import transcript  # noqa: E402
import words  # noqa: E402
import worker  # noqa: E402
import test_run_words_golden as G  # noqa: E402

SR = worker.SR
WHISPER = "onnx-community/whisper-large-v3-turbo"
ML = "multilingual_large_ctc"


def wd(text, s, e=None):
    return {"w": text, "s": float(s), "e": float(s) + 0.3 if e is None else float(e), "c": 0.9}


def starts(ws):
    return [w["s"] for w in ws]


class Caps(unittest.TestCase):
    """Таблица возможностей: одно место для всего конвейера."""

    def test_known_models(self):
        c = transcript.asr_model_caps
        self.assertEqual((c("v3_e2e_ctc")["label"], c("v3_e2e_ctc")["word_times"]), ("CTC", True))
        self.assertEqual((c(ML)["label"], c(ML)["word_times"], c(ML)["punctuation"]),
                         ("Multilingual large", True, False))
        self.assertEqual((c(WHISPER)["label"], c(WHISPER)["word_times"], c(WHISPER)["whisper"],
                          c(WHISPER)["ct2"]), ("Whisper", False, True, True))
        self.assertFalse(c("Org/Whisper-Small")["ct2"])
        self.assertFalse(c(WHISPER)["torch"])
        unknown = c("some_model")
        self.assertEqual((unknown["label"], unknown["word_times"], unknown["whisper"]),
                         ("some_model", None, False))

    def test_label_rule_is_shared(self):
        self.assertIs(transcript.third_model_label, transcript.model_label)
        self.assertEqual(worker._third_label(ML, "whisper-ct2"), "Whisper")    # старый результат подмены
        self.assertEqual(transcript.model_label("v3_ctc"), "CTC")


# ---------- 4. слова итога идут по времени ----------
TEN = ["Один,", "два,", "три,", "четыре,", "пять,", "шесть,", "семь,", "восемь,", "девять,", "десять."]


def shifted_second(shift):
    """Вторая модель как GigaAM Multilingual: строчные без пунктуации, времена смещены на shift;
    «дважды» вместо «два» и лишнее «кошка» после «пять» — как у третьей модели."""
    ws = [wd(words.norm_word(t), i + shift) for i, t in enumerate(TEN)]
    ws[1]["w"] = "дважды"
    ws[5:5] = [wd("кошка", 4.5 + shift)]
    return ws


THIRD = ["Один,", "дважды,", "три,", "четыре,", "пять,", "кошка,", "шесть,", "семь,", "восемь,", "девять,",
         "десять."]


class TimeOrder(unittest.TestCase):
    def main(self):
        return [wd(t, i) for i, t in enumerate(TEN)]

    def test_shifted_second_times_kept_in_order(self):
        for shift in (1.6, -0.8, 0.45):
            with self.subTest(shift=shift):
                main = self.main()
                stats = words.vote3(main, shifted_second(shift), list(THIRD))
                self.assertEqual((stats["replaced"], stats["inserted"]), (1, 1))
                self.assertEqual([w["w"] for w in main][:7],
                                 ["Один,", "дважды", "три,", "четыре,", "пять,", "кошка", "шесть,"])
                self.assertEqual(starts(main), sorted(starts(main)))
                self.assertTrue(all(w["e"] >= w["s"] for w in main))

    def test_reproduced_without_fit(self):
        """Причина: голосование ставило слова второй модели по тексту, а время брало её."""
        main = self.main()
        with mock.patch.object(words, "_fit_times", lambda out: None):
            words.vote3(main, shifted_second(1.6), list(THIRD))
        self.assertNotEqual(starts(main), sorted(starts(main)))

    def test_consistent_times_unchanged(self):
        """У согласованных времён (GigaAM CTC той же семьи) итог прежний."""
        ctc = shifted_second(0.0)
        fit, raw = self.main(), self.main()
        words.vote3(fit, copy.deepcopy(ctc), list(THIRD))
        with mock.patch.object(words, "_fit_times", lambda out: None):
            words.vote3(raw, copy.deepcopy(ctc), list(THIRD))
        self.assertEqual(fit, raw)

    def test_partial_exclusion_with_shifted_second(self):
        main = self.main()
        diag = {"primary": G.chunks((0.0, 10.0)), "second": G.chunks((0.0, 10.0)),
                "whisper": {"chunks": [{"state": "ok", "own_span": [0.0, 6.0]},
                                       {"state": "empty_unknown", "own_span": [6.0, 8.0]},
                                       {"state": "ok", "own_span": [8.0, 10.0]}]}}
        third = [t for t in THIRD if t not in ("семь,", "восемь,")]
        q = words.combine_hypotheses({"primary": main, "second": shifted_second(1.6), "whisper": third},
                                     diag)["quality"]
        self.assertIn("vote3_partial", q)
        self.assertEqual(starts(main), sorted(starts(main)))

    def test_run_words_with_shifted_second_model(self):
        shifted = [[(t, s + 0.9) for t, s in chunk] for chunk in G.SECOND]
        data, _ = G.run_scenario({"asr_second_model": "second"}, second_script=shifted)
        flat = [w["s"] for seg in data["segments"] for w in seg["words"]]
        self.assertEqual(flat, sorted(flat))
        self.assertIn("там", [w["w"] for seg in data["segments"] for w in seg["words"]])


# ---------- 1. нарезка по возможностям модели ----------
class ThirdChunks(unittest.TestCase):
    def plan(self, model, long_chunks=True):
        audio = np.zeros(int(70 * SR), dtype=np.float32)
        regions = [[0.0, 70.0]]
        chunks = words.make_chunks(audio, regions)
        cfg = {"asr_model": "v3_e2e_rnnt", "asr_second_model": "v3_e2e_ctc", "asr_third_model": model,
               "asr_third_mode": "all", "asr_third_long_chunks": long_chunks}
        a = SimpleNamespace(whisper="", whisper_max_repeat=-1)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            plan = worker.whisper_plan(audio, regions, chunks, cfg, a, [], [], {})
        return plan, chunks, err.getvalue()

    def test_whisper_keeps_long_chunks(self):
        plan, chunks, err = self.plan(WHISPER)
        self.assertNotEqual(plan["chunks"], chunks)
        self.assertTrue(max(c["own"][1] - c["own"][0] for c in plan["chunks"]) > words.CHUNK_MAX)
        self.assertEqual(err, "")

    def test_other_model_gets_primary_chunks(self):
        plan, chunks, err = self.plan(ML)
        self.assertEqual(plan["chunks"], chunks)
        self.assertEqual(len([x for x in err.splitlines() if "длинные куски" in x]), 1)

    def test_owner_setting_unchanged(self):
        """Рабочая настройка владельца: Multilingual large, asr_third_long_chunks=false."""
        plan, chunks, err = self.plan(ML, long_chunks=False)
        self.assertEqual(plan["chunks"], chunks)
        self.assertEqual(err, "")
        plan, chunks, _ = self.plan(WHISPER, long_chunks=False)
        self.assertEqual(plan["chunks"], chunks)


# ---------- 2. чистка «субтитров» и повторов — только для Whisper ----------
class ThirdTextCleaning(unittest.TestCase):
    TEXTS = ["корректор сказал что всё хорошо да да да да да", "продолжение следует"]

    def run_words(self, model):
        plan = {"model": model, "max_repeat": 3, "selected": [0, 1],
                "chunks": [{"a": 0, "b": 5, "own": (0, 5)}, {"a": 5, "b": 9, "own": (5, 9)}]}
        diag = [{"index": 0, "state": "ok"}, {"index": 1, "state": "ok"}]
        tokens, removed = worker._whisper_words(plan, list(self.TEXTS), diag, [])
        return tokens, removed, diag

    def test_other_model_text_is_kept(self):
        tokens, removed, diag = self.run_words(ML)
        self.assertEqual(tokens, " ".join(self.TEXTS).split())
        self.assertEqual(removed, 0)
        self.assertFalse(any(d.get("filtered_all") for d in diag))

    def test_whisper_text_is_cleaned_as_before(self):
        tokens, removed, diag = self.run_words(WHISPER)
        self.assertEqual(tokens, [])                      # «корректор» — до конца текста без точки
        self.assertEqual(removed, 12)
        self.assertEqual([d.get("filtered_all") for d in diag], [True, True])
        self.assertEqual(words.third_text("Да да да да да.", 3, True), words.clean_whisper("Да да да да да.", 3))

    def test_clean_whisper_harms_text_without_punctuation(self):
        """Почему не для всех: на тексте без точек правило «субтитров» стирает весь остаток куска."""
        self.assertEqual(words.clean_whisper(self.TEXTS[0], 3)[0], "")
        self.assertEqual(words.clean_whisper(self.TEXTS[0], 0, junk=False),
                         (self.TEXTS[0], 0))


# ---------- 3. тексты для человека ----------
class HumanTexts(unittest.TestCase):
    ITEM = {"id": "r1", "t": 5.0, "speaker": "Анна", "text": "дело", "reasons": ["asr"],
            "alt_ctc": "тело", "alt_whisper": "дали"}

    def md(self, timing=None, quality=None):
        d = {"file": "x.wav", "duration": "00:00:10", "speakers": ["Анна"], "review": [dict(self.ITEM)],
             "segments": [{"speaker": "Анна", "start": 0, "end": 10, "text": "дело", "words": []}],
             "quality": quality or {}}
        if timing is not None:
            d["timing"] = timing
        return transcript.render_md(d)

    def line(self, timing):
        return [x for x in self.md(timing).splitlines() if "«тело»" in x][0]

    def test_default_lines_byte_for_byte(self):
        expected = "- 00:00:05 Анна: «дело» — модели распознали по-разному; CTC: «тело»; Whisper: «дали»"
        for timing in (None, {}, {"asr2_model": "v3_e2e_ctc", "asr3_model": WHISPER,
                                  "asr3_device": "whisper-ct2:CPU:int8"},
                       {"asr2_model": "v3_e2e_ctc", "asr3_model": WHISPER, "asr3_device": "onnx-cpu:CPU:int8"}):
            with self.subTest(timing=timing):
                self.assertEqual(self.line(timing), expected)

    def test_real_models_are_named(self):
        line = self.line({"asr2_model": ML, "asr3_model": WHISPER, "asr3_device": "onnx-cpu:CPU:int8"})
        self.assertIn("; Multilingual large: «тело»; Whisper: «дали»", line)
        self.assertNotIn("CTC", line)
        line = self.line({"asr2_model": "v3_e2e_ctc", "asr3_model": ML, "asr3_device": "onnx-cpu:CPU:"})
        self.assertIn("; CTC: «тело»; Multilingual large: «дали»", line)

    def test_review_note_names_third_model(self):
        data, _ = G.run_scenario(whisper={"onnx-cpu": {"fail_load": True}})
        notes = {i["note"] for i in data["review"] if i.get("asr_problem", {}).get("state") == "error"}
        self.assertEqual(notes, {"Результат Whisper неполон или неизвестен; прослушайте фрагмент."})
        data, _ = G.run_scenario({"asr_third_model": ML}, whisper={"onnx-cpu": {"fail_load": True}})
        problems = [i for i in data["review"] if i.get("asr_problem", {}).get("state") == "error"]
        self.assertTrue(problems)
        self.assertEqual({i["note"] for i in problems},
                         {"Результат Multilingual large неполон или неизвестен; прослушайте фрагмент."})
        self.assertEqual({i["asr_problem"]["model"] for i in problems}, {"whisper"})   # ключ данных прежний

    def stage_labels(self, second):
        seen = []

        class St:
            def set(self, **kw):
                pass

        def prog(lo, hi, name):
            return lambda done, total: seen.append(name)

        class Fake:
            name, model_name = "fake", second

            def transcribe_timestamped(self, pieces, progress=None, diagnostics=None):
                progress(1, 1)
                diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
                return [None for _ in pieces]

        quality = {"asr_diagnostics": {}, "asr_run": {"models": {}}}
        chunks = [{"a": 0.0, "b": 1.0, "own": (0.0, 1.0)}]
        worker.second_pass(np.zeros(SR, dtype=np.float32), chunks,
                           {"asr_model": "v3_e2e_rnnt", "asr_second_model": second},
                           quality, lambda cfg: (Fake(), None), prog, {})
        return seen

    def test_second_stage_label(self):
        self.assertEqual(self.stage_labels("v3_e2e_ctc"), ["второй проход распознавания (CTC)"])
        self.assertEqual(self.stage_labels(ML), ["второй проход распознавания (Multilingual large)"])


# ---------- 5. вторая модель без времён слов ----------
class SecondWithoutWordTimes(unittest.TestCase):
    def test_refused_with_one_clear_line(self):
        made, err = [], io.StringIO()
        quality = {"asr_diagnostics": {}, "asr_run": {"models": {}}}
        with contextlib.redirect_stderr(err):
            alt = worker.second_pass(np.zeros(SR, dtype=np.float32), [{"a": 0, "b": 1, "own": (0, 1)}],
                                     {"asr_model": "v3_e2e_rnnt", "asr_second_model": WHISPER}, quality,
                                     lambda cfg: made.append(cfg), None, {})
        self.assertIsNone(alt)
        self.assertEqual(made, [])                           # модель не загружается и не подменяется
        self.assertEqual(quality["asr2_skipped"], {"reason": "no_word_times", "model": WHISPER})
        self.assertNotIn("second", quality["asr_diagnostics"])
        self.assertEqual(len([x for x in err.getvalue().splitlines() if "не даёт времён слов" in x]), 1)

    def test_pipeline_is_safe(self):
        with contextlib.redirect_stderr(io.StringIO()):
            data, calls = G.run_scenario({"asr_second_model": "org/whisper-small"})
        q = data["quality"]
        self.assertEqual(q["asr2_skipped"]["reason"], "no_word_times")
        for key in ("vote", "vote3_skipped", "vote3_partial", "asr_incomplete_chunks"):
            self.assertNotIn(key, q)
        self.assertEqual([m for m in calls.models if m[0] == "make_asr"
                          and m[1].get("asr_model") != "primary"], [])     # ни второй, ни третьей
        self.assertFalse(any(w.get("alt") for seg in data["segments"] for w in seg["words"]))
        md = transcript.render_md({**data, "file": "x.wav", "duration": "00:00:30"})
        self.assertIn("_Вторая модель (Whisper) не даёт времён слов и вторым номером не используется: "
                      "сверки и голосования моделей нет._", md)


# ---------- 6. одинаковые модели ----------
class SameModels(unittest.TestCase):
    def words_of(self, data):
        return [(w["w"], w["s"], w.get("alt"), w.get("flags")) for seg in data["segments"] for w in seg["words"]]

    def test_third_same_as_second_is_skipped(self):
        off, _ = G.run_scenario({"asr_third_mode": "off"})
        with contextlib.redirect_stderr(io.StringIO()):
            same, calls = G.run_scenario({"asr_third_model": "second"})
        self.assertEqual(same["quality"]["asr3_skipped"], {"reason": "same_as_second", "model": "second"})
        self.assertNotIn("whisper", same["quality"]["asr_diagnostics"])
        self.assertNotIn("vote", same["quality"])
        self.assertEqual(self.words_of(same), self.words_of(off))      # как без третьей модели
        self.assertEqual(len([m for m in calls.models if m[0] == "texts"]), 0)

    def test_third_same_as_primary_is_skipped(self):
        with contextlib.redirect_stderr(io.StringIO()):
            same, _ = G.run_scenario({"asr_third_model": "primary"})
        self.assertEqual(same["quality"]["asr3_skipped"]["reason"], "same_as_primary")
        md = transcript.render_md({**same, "file": "x.wav", "duration": "00:00:30"})
        self.assertIn("_Третья модель (primary) совпадает с основной и не считалась", md)

    def test_second_same_as_primary_is_explicit(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            data, _ = G.run_scenario({"asr_second_model": "primary"})
        self.assertEqual(data["quality"]["asr2_skipped"], {"reason": "same_as_primary", "model": "primary"})
        self.assertIn("совпадает с основной", err.getvalue())

    def test_defaults_have_no_skip_marks(self):
        data, _ = G.run_scenario()
        self.assertNotIn("asr2_skipped", data["quality"])
        self.assertNotIn("asr3_skipped", data["quality"])


# ---------- 7. выбор движка ----------
class EngineChoice(unittest.TestCase):
    def test_second_model_never_on_whisper_only_engine(self):
        made = []

        class Fake:
            name, model_name = "fake", ML

            def transcribe_timestamped(self, pieces, progress=None, diagnostics=None):
                diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
                return [None for _ in pieces]

        quality = {"asr_diagnostics": {}, "asr_run": {"models": {}}}
        with contextlib.redirect_stderr(io.StringIO()):
            worker.second_pass(np.zeros(SR, dtype=np.float32), [{"a": 0, "b": 1, "own": (0, 1)}],
                               {"asr_model": "v3_e2e_rnnt", "asr_backend": "onnx-cpu", "asr_second_model": ML,
                                "asr_second_backend": "whisper-ct2"}, quality,
                               lambda cfg: (made.append(cfg), (Fake(), None))[1],
                               lambda lo, hi, name: (lambda d, t: None), {})
        self.assertEqual([(c["asr_model"], c["asr_backend"]) for c in made], [(ML, "onnx-cpu")])

    def tried(self, model):
        tried = []

        class Onnx:
            def __init__(self, *a, provider=None, **kw):
                tried.append(provider)
                raise RuntimeError("synthetic")

        class Torch:
            def __init__(self, *a, **kw):
                tried.append("torch")
                raise RuntimeError("synthetic")

        with mock.patch.object(backends, "OnnxASR", Onnx), mock.patch.object(backends, "TorchASR", Torch), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError):
                backends.make_asr({"asr_model": model, "asr_backend": "onnx-cpu", "asr_device": "CPU"})
        return tried

    def test_torch_fallback_follows_caps(self):
        self.assertNotIn("torch", self.tried(WHISPER))
        self.assertNotIn("torch", self.tried("Org/Whisper-Small"))
        self.assertIn("torch", self.tried("v3_e2e_ctc"))
        self.assertIn("torch", self.tried(ML))           # у пакета gigaam Multilingual есть

    def test_other_whisper_on_ct2_is_not_silent(self):
        err, events = io.StringIO(), []

        class Fake:
            name = "fake"

            def __init__(self, *a, **kw):
                events.append("whisper-ct2")

            def transcribe_texts(self, pieces, progress=None, diagnostics=None):
                diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
                return ["ok" for _ in pieces]

        plan = {"model": "org/whisper-small", "mode": "all", "max_repeat": 3, "selected": [0],
                "chunks": [{"a": 0, "b": 1, "own": (0, 1)}], "pieces": [np.zeros(SR, dtype=np.float32)]}
        quality = {"asr_diagnostics": {}, "asr_run": {"models": {}}}
        with mock.patch.object(backends, "WhisperCT2", Fake), contextlib.redirect_stderr(err):
            worker.whisper_pass(plan, [], {"asr_third_backend": "whisper-ct2"}, SimpleNamespace(set=lambda **k: None),
                                quality, None, lambda lo, hi, name: (lambda d, t: None), {})
        self.assertEqual(events, ["whisper-ct2"])
        self.assertIn("считает Whisper large-v3-turbo, а не эту модель", err.getvalue())
        err2 = io.StringIO()
        plan["model"] = WHISPER
        with mock.patch.object(backends, "WhisperCT2", Fake), contextlib.redirect_stderr(err2):
            worker.whisper_pass(plan, [], {"asr_third_backend": "whisper-ct2"}, SimpleNamespace(set=lambda **k: None),
                                quality, None, lambda lo, hi, name: (lambda d, t: None), {})
        self.assertNotIn("а не эту модель", err2.getvalue())


if __name__ == "__main__":
    unittest.main()
