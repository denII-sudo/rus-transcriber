"""Фиксатор поведения пословной обработки (worker.run_words) при трёх моделях.

Модели, эмбеддер голоса и разметка pyannote подменены синтетическими: настоящие модели не
загружаются и не скачиваются, звук — тишина. Настоящие функции words.py работают: нарезка,
сверка проходов, голосование «два из трёх», голос по словам, реплики, метки сомнения.

Каждый сценарий сводит результат в канонический JSON (порядок ключей сохраняется, замеры
времени заменены именами ключей) и сравнивает его отпечаток SHA-256 с записанным. Отпечатки
сняты с кода до разбиения run_words на этапы (РАР-4, этап 4.2, шаг 1): совпадение значит, что
результат, сведения о сбоях, вызовы моделей и ход статуса прежние.

Если поведение меняется намеренно, новые отпечатки печатает
    RUN_WORDS_GOLDEN_PRINT=1 python -B -m unittest tests.test_run_words_golden
"""
import copy
import hashlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import backends
import words
import worker

SR = worker.SR

# --- синтетическая запись: Анна 0–8 с, Борис 8,5–15 с, Анна 16–27 с, Борис 27,5–29,5 с без слов ---
RAW = [(0.0, 8.0, "SPEAKER_00"), (8.5, 15.0, "SPEAKER_01"), (16.0, 27.0, "SPEAKER_00"),
       (27.5, 29.5, "SPEAKER_01")]
ORDER = {"SPEAKER_00": "Спикер 1", "SPEAKER_01": "Спикер 2"}
TURNS = [{"s": s, "e": e, "label": ORDER[spk]} for s, e, spk in RAW]
NAMES = {"Спикер 1": "Анна", "Спикер 2": "Борис"}

# Куски GigaAM по этой разметке: own (0; 15,15) и (15,85; 27,15) — начало второго куска 15,85.
PRIMARY = [
    [(" привет", 0.5), (",", 0.6), (" борька", 1.0), (" как", 1.5), (" дела", 2.0), ("?", 2.1),
     (" я", 3.0), (" сделала", 3.3), (" это", 4.0), (" очень", 4.5), (" хорошо", 5.0), (".", 5.1),
     (" да", 9.0), (" мы", 9.5), (" пошли", 10.0), (" домой", 10.5), (" вчера", 11.0),
     (" вечером", 11.5), (" я", 12.5), (" устала", 12.8)],
    [(" ну", 0.5), (" ладно", 1.0), (" тогда", 1.5), (" увидимся", 2.0), (" завтра", 3.0), (".", 3.1)],
]
SECOND = [
    [(" привет", 0.5), (" борька", 1.0), (" как", 1.5), (" дело", 2.0), (" я", 3.0), (" сделала", 3.3),
     (" это", 4.0), (" очень", 4.5), (" хорошо", 5.0), (" да", 9.0), (" мы", 9.5), (" пошли", 10.0),
     (" там", 10.3), (" домой", 10.5), (" вчера", 11.0), (" вечерком", 11.5), (" я", 12.5),
     (" устала", 12.8)],
    [(" ну", 0.5), (" ладна", 1.0), (" тогда", 1.5), (" увидимся", 2.0), (" завтра", 3.0)],
]
# Текст Whisper по кускам: первый кусок записи длиннее 14,5 с, второй короче.
WHISPER_TEXTS = ("Привет, Борька, как дело? Я сделала, это очень хорошо. Да, мы пошли там домой вчера "
                 "вечерами, я устала.",
                 "Ну ладно, тогда увидимся завтра завтра завтра завтра завтра. Продолжение следует...")
LOW_CONFIDENCE = {"очень"}       # у этого слова основной проход неуверен


def result(tokens, low=()):
    return SimpleNamespace(tokens=[t for t, _ in tokens], timestamps=[s for _, s in tokens],
                           logprobs=[-1.0 if t.strip() in low else -0.05 for t, _ in tokens],
                           text="".join(t for t, _ in tokens))


class Calls:
    """Журнал обращений: какие модели и с какими настройками создавались, ход статуса."""

    def __init__(self):
        self.models, self.status = [], []


class FakeTimestamped:
    def __init__(self, name, script, calls, errors=(), low=()):
        self.name, self.model_name = name, "model-" + name
        self.script, self.calls, self.errors, self.low = script, calls, set(errors), set(low)

    def transcribe_timestamped(self, pieces, progress=None, diagnostics=None):
        self.calls.models.append(("timestamped", self.name, [len(p) for p in pieces]))
        out = []
        for i, piece in enumerate(pieces):
            failed = i in self.errors or i >= len(self.script)
            diagnostics.append({"index": i, "samples": len(piece), "state": "error" if failed else "ok",
                                "attempts": [{"result": "error" if failed else "ok", "batch": [i]}]})
            out.append(None if failed else result(self.script[i], self.low))
        if progress:
            progress(len(pieces), len(pieces))
        return out


class FakeTexts:
    """Текстовая модель (Whisper): ошибки по номерам кусков в вызове, текст — по длине куска."""

    def __init__(self, name, calls, errors=(), empty=(), fail_load=False):
        if fail_load:
            raise RuntimeError("synthetic load failure " + name)
        self.name, self.calls = name, calls
        self.errors, self.empty = set(errors), set(empty)
        self.compile_sec = 1.5

    def transcribe_texts(self, pieces, progress=None, diagnostics=None):
        self.calls.models.append(("texts", self.name, [len(p) for p in pieces]))
        out = []
        for i, piece in enumerate(pieces):
            state = "error" if i in self.errors else ("empty_unknown" if i in self.empty else "ok")
            diagnostics.append({"index": i, "samples": len(piece), "state": state,
                                "attempts": [{"result": state, "batch": [i]}]})
            out.append(WHISPER_TEXTS[0 if len(piece) > 14.5 * SR else 1] if state == "ok" else "")
        if progress:
            progress(len(pieces), len(pieces))
        return out


class FakeEmbedder:
    """Голос по середине окна: Анна [1, 0], Борис [0, 1], у смены говорящего — смесь."""
    seconds = 0.0

    def spans(self, audio, spans, min_len=None, progress=None):
        out = []
        for s, e in spans:
            c = (s + e) / 2
            if 8.25 <= c < 8.75 or 15.25 <= c < 15.75:
                out.append(np.array([0.5, 0.5]))
            elif 8.75 <= c < 15.25 or c >= 27.25:
                out.append(np.array([0.125, 0.875]))
            else:
                out.append(np.array([0.875, 0.125]))
        return out


class Segment:
    def __init__(self, start, end):
        self.start, self.end = start, end


class Annotation:
    """Разметка с наложением голосов 9,0–9,75 с."""

    def itertracks(self, yield_label=True):
        for s, e, spk in RAW + [(9.0, 9.75, "SPEAKER_00")]:
            yield Segment(s, e), None, spk


class Status:
    def __init__(self, calls):
        self.calls = calls

    def set(self, **kwargs):
        self.calls.status.append(kwargs)


def run_scenario(cfg_extra=None, *, primary_errors=(), second_fails=False, whisper=None, args=None,
                 second_script=SECOND):
    """-> (данные run_words, журнал вызовов). whisper — {"backend": {...параметры FakeTexts}}."""
    calls = Calls()
    whisper = whisper or {}
    cfg = {"asr_model": "primary", "asr_backend": "onnx-cpu", "asr_device": "CPU", "asr_batch": 8,
           "asr_second_model": "second", "asr_second_batch": 1,
           "asr_third_model": "whisper", "asr_third_mode": "all", "asr_third_device": "CPU",
           "asr_third_backend": "", "asr_third_batch": 4, "asr_third_quant_cpu": "int8",
           "asr_third_max_repeat": 3, "emotions": False, "interaction": False, "review_missed": True,
           "word_win_near_change": 0, **(cfg_extra or {})}

    def make_asr(conf):
        calls.models.append(("make_asr", {k: conf.get(k) for k in sorted(conf) if k.startswith("asr_")}))
        model, backend = conf.get("asr_model"), conf.get("asr_backend")
        if model == "primary":
            return FakeTimestamped("primary", PRIMARY, calls, errors=primary_errors, low=LOW_CONFIDENCE), None
        if model == "second":
            if second_fails:
                raise RuntimeError("synthetic second load failure")
            return FakeTimestamped("second", second_script, calls), None
        params = whisper.get(backend, {})
        return FakeTexts(backend, calls, **params), (["synthetic fallback"] if backend == "onnx-cpu" else None)

    def ov(model, device=None, quant=None, language=None, batch=None):
        return FakeTexts("whisper-ov", calls, **whisper.get("whisper-ov", {}))

    def ct2(cpu_threads=0):
        return FakeTexts("whisper-ct2", calls, **whisper.get("whisper-ct2", {}))

    label_name = dict(NAMES)
    label_of = {v: k for k, v in NAMES.items()}
    voice_ctx = {"pvecs": {"Анна": np.array([1.0, 0.0]), "Борис": np.array([0.0, 1.0])},
                 "cents": {"Спикер 1": np.array([1.0, 0.0]), "Спикер 2": np.array([0.0, 1.0])},
                 "mapping": {"Спикер 1": ("Анна", 0.9), "Спикер 2": ("Борис", 0.9)},
                 "confusions": {("Анна", "Борис"): 1}}
    quality = {"vocab": None, "speaker_fixes": [], "vocab_fixes": [], "review": []}
    a = args or SimpleNamespace(whisper="", whisper_max_repeat=-1, no_emotions=True)
    audio = np.zeros(int(30 * SR), dtype=np.float32)
    meta = {"date_utc": "synthetic", "program": "synthetic", "packages": {}, "models": {}}
    with mock.patch.object(backends, "WhisperOV", ov), mock.patch.object(backends, "WhisperCT2", ct2), \
            mock.patch.object(worker, "asr_run_metadata", return_value=meta):
        data = worker.run_words(audio, "synthetic.wav", cfg, Status(calls), list(RAW), dict(ORDER),
                                copy.deepcopy(TURNS), Annotation(), FakeEmbedder(), label_name, label_of,
                                voice_ctx, ["Анна", "Борис"], {"Анна": "f", "Борис": "m"},
                                {"replace": {"борька": "Борис"}}, quality, make_asr, a)
    return data, calls


TIME_KEYS = {"asr_sec", "asr2_sec", "asr3_sec", "asr3_load_sec", "word_voice_sec", "interaction_sec",
             "emotions_sec"}


def canonical(data, calls):
    """Канонический вид: значения замеров времени скрыты, порядок ключей сохранён."""
    value = copy.deepcopy(data)
    value["_timing"] = {k: ("<time>" if k in TIME_KEYS else v) for k, v in value["_timing"].items()}
    return json.dumps({"data": value, "models": calls.models, "status": calls.status},
                      ensure_ascii=False, separators=(",", ":"), default=str)


def digest(data, calls):
    return hashlib.sha256(canonical(data, calls).encode("utf-8")).hexdigest()


SCENARIOS = {
    # все три модели отработали: голосование «два из трёх» (замена, вставка, спор, чистка Whisper)
    "three_models": dict(),
    # второй проход не загрузился: сведения о сбое по каждому куску, без голосования
    "second_fails": dict(second_fails=True),
    # Whisper не загрузился ни на одном движке: голосование пропущено, места для прослушивания
    "whisper_fails": dict(whisper={"onnx-cpu": {"fail_load": True}}),
    # Whisper только по кускам расхождения RNN-T и CTC, короткие куски
    # (второй кусок оба прохода слышат одинаково — Whisper его не считает)
    "whisper_doubt": dict(cfg_extra={"asr_third_mode": "doubt", "asr_third_long_chunks": False},
                          second_script=[SECOND[0], PRIMARY[1]]),
    # без Whisper: сверка основного прохода со вторым
    "whisper_off": dict(cfg_extra={"asr_third_mode": "off"}),
    # ошибка основного прохода на втором куске
    "primary_chunk_error": dict(primary_errors=(1,)),
    # faster-whisper с ошибкой на куске, повтор на onnx-cpu удался; второй проход на своём устройстве
    "ct2_retry": dict(cfg_extra={"asr_third_backend": "whisper-ct2", "asr_third_long_chunks": False,
                                 "asr_second_device": "GPU"},
                      whisper={"whisper-ct2": {"errors": (1,)}}),
    # Whisper на видеокарте: ошибка загрузки GPU, затем onnx-cpu с пустым куском; голосование
    # пропущено только на отрезке пустого куска (отпечаток снят заново 22.11.2026, vote3_partial)
    "gpu_then_cpu": dict(cfg_extra={"asr_third_device": "GPU", "asr_third_long_chunks": False},
                         whisper={"whisper-ov": {"fail_load": True}, "onnx-cpu": {"empty": (0,)}}),
    # ограничения из аргументов запуска: без ограничения повторов, Whisper только в местах расхождения
    "args_override": dict(args=SimpleNamespace(whisper="doubt", whisper_max_repeat=0, no_emotions=True)),
}

# 22.11.2026 (любая модель вторым и третьим номером): подпись этапа второго прохода называет
# модель — «второй проход распознавания (second)» у синтетической модели «second» вместо прежнего
# «(CTC)»; у GigaAM CTC подпись прежняя. Другого отличия нет: обратная замена подписи в каноническом
# виде даёт прежние отпечатки всех сценариев (second_fails подписи не содержит и не изменился).
EXPECTED = {
    "three_models": "64cfb7584947c48cf95bdad005071cf5afa5a5cae23a718e93b0a9d18738f91a",
    "second_fails": "e2638ce9942cac3ec1c3a64bee7520d84b18ee61d8c6b8f7af9ad2e4650ad259",
    "whisper_fails": "7c1470d356a39efd2d89954a300e8c400e11d5048d7d43650b07407017a57eb0",
    "whisper_doubt": "92038eaee5f3008d319cc89afe0b0ed17b07f83700181aeb176e9a384d03f459",
    "whisper_off": "764331baf52406f6b793edab08e5671c12914de2157b279fa485f9880c45fb2c",
    "primary_chunk_error": "e4641671fac09839fb4fde37e7400145be00506a318323ba52c29c02eae2483d",
    "ct2_retry": "bc9e3ec6b809d7c18d926aad764eba4a9c0a34f3db619ada84f0e3d085e945f8",
    "gpu_then_cpu": "3f2c02aec99d7491e74b98cf2ac707ae8ef73541c4d3d9eb63d8c0665f58027d",
    "args_override": "52289a92987d8ebde448dc33484744dd2b5fe58170c73b51897c3462f81f3adf",
}


class RunWordsGoldenTests(unittest.TestCase):
    def test_scenarios_match_recorded_behaviour(self):
        printing = os.environ.get("RUN_WORDS_GOLDEN_PRINT")
        for name, params in SCENARIOS.items():
            data, calls = run_scenario(**params)
            got = digest(data, calls)
            if printing:
                print(f'    "{name}": "{got}",')
                if printing == "full":
                    print(canonical(data, calls))
                continue
            with self.subTest(scenario=name):
                self.assertEqual(got, EXPECTED[name], canonical(data, calls)[:4000])

    def test_three_models_vote_is_readable(self):
        """То же, что зашито в отпечаток, но словами: что именно решило голосование."""
        data, calls = run_scenario()
        quality = data["quality"]
        self.assertEqual(quality["vote"], {"agree3": 19, "agree2": 1, "replaced": 1, "inserted": 1,
                                           "disputed": 1})
        self.assertEqual(quality["asr_votes"], [{"t": 2.0, "t2": 2.0, "model": "дела?", "to": "дело"},
                                                {"t": 10.3, "t2": 10.3, "model": "", "to": "там"}])
        words = [w["w"] for seg in data["segments"] for w in seg["words"]]
        self.assertIn("дело", words)             # CTC и Whisper согласны против RNN-T
        self.assertIn("там", words)              # обе другие модели слышат слово, RNN-T — нет
        self.assertIn("Борис", words)            # словарь группы
        self.assertEqual(quality["vocab_fixes"], [{"t": 1.0, "fixes": [("борька", "Борис")]}])
        disputed = [w for seg in data["segments"] for w in seg["words"] if w.get("alt_whisper")]
        self.assertEqual([(w["w"], w["alt"], w["alt_whisper"]) for w in disputed],
                         [("вечером", "вечерком", "вечерами,")])
        # повторы «завтра» сверх трёх и «субтитры» убраны из второго куска Whisper
        self.assertEqual([c["removed_words"] for c in quality["asr_diagnostics"]["whisper"]["chunks"]], [0, 4])
        self.assertNotIn("vote3_skipped", quality)
        self.assertNotIn("asr_incomplete_chunks", quality)
        self.assertEqual([m for m in calls.models if m[0] == "make_asr"][1][1]["asr_batch"], 1)


def hypothesis(tokens, offset=0.0):
    return words.result_words(result(tokens), offset)


class CombineHypothesesTests(unittest.TestCase):
    """Стык «гипотезы -> слова и пометки» (words.combine_hypotheses) на синтетических гипотезах."""

    def setUp(self):
        self.main = hypothesis(PRIMARY[0])
        self.second = hypothesis(SECOND[0])
        self.whisper = [t for t in words.clean_whisper(WHISPER_TEXTS[0])[0].split() if words.norm_word(t)]
        self.ok = {"primary": {"chunks": [{"state": "ok", "own_span": [0.0, 15.15]}]},
                   "second": {"chunks": [{"state": "ok", "own_span": [0.0, 15.15]}]},
                   "whisper": {"chunks": [{"state": "ok", "own_span": [0.0, 15.15]}]}}

    def test_three_models_vote_like_vote3(self):
        expected_words = copy.deepcopy(self.main)
        stats = words.vote3(expected_words, copy.deepcopy(self.second), list(self.whisper))
        got = words.combine_hypotheses({"primary": self.main, "second": self.second, "whisper": self.whisper},
                                       self.ok)
        self.assertEqual(self.main, expected_words)
        self.assertEqual(got, {"quality": {"vote": stats, "asr_votes": [
            {"t": 2.0, "t2": 2.0, "model": "дела?", "to": "дело"},
            {"t": 10.3, "t2": 10.3, "model": "", "to": "там"}]},
            "evidence": {"whisper": [], "other": []}})

    def test_incomplete_whisper_skips_vote_and_compares_two_passes(self):
        expected_words = copy.deepcopy(self.main)
        words.mark_alt(expected_words, self.second)
        diagnostics = {**self.ok, "whisper": {"chunks": [{"state": "error", "own_span": [0.0, 15.15]}]}}
        got = words.combine_hypotheses({"primary": self.main, "second": self.second, "whisper": None},
                                       diagnostics)
        self.assertEqual(self.main, expected_words)
        self.assertEqual(got, {"quality": {"vote3_skipped": "whisper_incomplete", "asr_incomplete_chunks": 1},
                               "evidence": {"whisper": [(0.0, 15.15, "error", ["primary", "second"])],
                                            "other": []}})

    def test_failed_second_pass_leaves_primary_words_untouched(self):
        before = copy.deepcopy(self.main)
        diagnostics = {**self.ok, "second": {"chunks": [{"state": "error", "own_span": [0.0, 15.15]}]}}
        got = words.combine_hypotheses({"primary": self.main, "second": None, "whisper": None}, diagnostics)
        self.assertEqual(self.main, before)
        self.assertEqual(got, {"quality": {"asr_incomplete_chunks": 1},
                               "evidence": {"whisper": [], "other": [(0.0, 15.15, "second", "error", ["primary"])]}})


# --- частичное исключение отрезков из голосования: синтетическая запись 0–10 с, слово в секунду ---
TEN = ["один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять"]


def wd(text, s):
    return {"w": text, "s": float(s), "e": float(s) + 0.3, "c": 0.9}


def chunks(*spans, state="ok", **extra):
    return {"chunks": [{"state": state, "own_span": list(span), **extra} for span in spans]}


class PartialVoteTests(unittest.TestCase):
    """Сбойный кусок третьей или второй модели исключает из vote3 только свой отрезок own_span."""

    def setUp(self):
        self.main = [wd(t, i) for i, t in enumerate(TEN)]
        # CTC: «дважды» вместо «два» (с Whisper против RNN-T), «четверо» вместо «четыре»,
        # «кошка» после «пять» и «ещё» после «восемь» — слова, которых нет у RNN-T
        self.ctc = [wd(t, i) for i, t in enumerate(TEN)]
        self.ctc[1]["w"], self.ctc[3]["w"] = "дважды", "четверо"
        self.ctc[5:5] = [wd("кошка", 4.5)]
        self.ctc[9:9] = [wd("ещё", 7.5)]
        self.ok = {"primary": chunks((0.0, 10.0)), "second": chunks((0.0, 10.0))}

    def whisper_without(self, *spans):
        """Текст Whisper: «дважды» и «ещё» как у CTC, «четыре» как у RNN-T, «кошка» не слышит;
        без слов на сбойных отрезках (времена нужны только тесту, Whisper их не даёт)."""
        heard = [("четыре" if w["w"] == "четверо" else w["w"], w["s"]) for w in self.ctc if w["w"] != "кошка"]
        return [t for t, s in heard if not any(a <= s < b for a, b in spans)]

    def whisper_diag(self, bad, state="empty_unknown", **extra):
        good = [(a, b) for a, b in ((0.0, bad[0]), (bad[1], 10.0)) if b > a]
        rec = chunks(bad, state=state, **extra)["chunks"] + chunks(*good)["chunks"]
        return {"chunks": sorted(rec, key=lambda r: r["own_span"])}

    def expected(self, spans, whisper):
        """Эталон: vote3 по словам вне отрезков, mark_alt внутри, сборка по времени."""
        def inside(w):
            return any(a <= w["s"] < b for a, b in spans)
        main, ctc = copy.deepcopy(self.main), copy.deepcopy(self.ctc)
        keep = [w for w in main if not inside(w)]
        stats = words.vote3(keep, [w for w in ctc if not inside(w)], list(whisper))
        parts = []
        for a, b in spans:
            mp = [w for w in main if a <= w["s"] < b]
            words.mark_alt(mp, [w for w in ctc if a <= w["s"] < b])
            parts.append((a, mp))
        out = list(keep)
        for a, mp in reversed(parts):
            pos = next((i for i, w in enumerate(out) if w["s"] >= a), len(out))
            out[pos:pos] = mp
        return out, stats

    def combine(self, whisper, diagnostics):
        return words.combine_hypotheses({"primary": self.main, "second": self.ctc, "whisper": whisper},
                                        diagnostics)

    def assert_time_order(self):
        self.assertEqual([w["s"] for w in self.main], sorted(w["s"] for w in self.main))

    def test_whisper_failure_in_the_middle(self):
        whisper = self.whisper_without((3.0, 6.0))
        diag = {**self.ok, "whisper": self.whisper_diag((3.0, 6.0))}
        want, stats = self.expected([(3.0, 6.0)], whisper)
        whole = copy.deepcopy(self.main)
        words.mark_alt(whole, self.ctc)
        got = self.combine(whisper, diag)
        self.assertEqual(self.main, want)
        self.assert_time_order()
        quality = got["quality"]
        self.assertEqual(quality["vote"], stats)
        self.assertEqual(quality["vote3_partial"], {"reason": "whisper_incomplete", "spans": 1, "seconds": 3.0})
        self.assertNotIn("vote3_skipped", quality)
        self.assertNotIn("asr_incomplete_chunks", quality)
        # вне отрезка голосование заменяет и вставляет слова
        self.assertEqual(quality["asr_votes"], [{"t": 1.0, "t2": 1.0, "model": "два", "to": "дважды"},
                                                {"t": 7.5, "t2": 7.5, "model": "", "to": "ещё"}])
        self.assertEqual([w["w"] for w in self.main],
                         ["один", "дважды", "три", "четыре", "пять", "шесть", "семь", "восемь", "ещё",
                          "девять", "десять"])
        # на отрезке — то же, что дала бы сверка двух моделей по всей записи (включая alt_ins)
        span = [w for w in self.main if 3.0 <= w["s"] < 6.0]
        self.assertEqual(span, [w for w in whole if 3.0 <= w["s"] < 6.0])
        self.assertEqual((span[0]["alt"], span[0]["agree"]), ("четверо", False))
        self.assertEqual(span[2]["alt_ins"], "кошка")
        self.assertEqual(got["evidence"], {"whisper": [(3.0, 6.0, "empty_unknown", ["primary", "second"])],
                                           "other": []})

    def test_whisper_failure_at_the_start(self):
        whisper = self.whisper_without((0.0, 3.0))
        want, stats = self.expected([(0.0, 3.0)], whisper)
        quality = self.combine(whisper, {**self.ok, "whisper": self.whisper_diag((0.0, 3.0))})["quality"]
        self.assertEqual(self.main, want)
        self.assert_time_order()
        self.assertEqual(quality["vote"], stats)
        self.assertEqual(quality["vote3_partial"]["spans"], 1)
        self.assertEqual([w["w"] for w in self.main][:3], ["один", "два", "три"])   # «два» не заменено
        self.assertEqual((self.main[1]["alt"], self.main[1]["agree"]), ("дважды", False))
        self.assertNotIn("voted", self.main[1])
        self.assertIn("ещё", [w["w"] for w in self.main])

    def test_whisper_failure_at_the_end(self):
        whisper = self.whisper_without((7.0, 10.0))
        want, stats = self.expected([(7.0, 10.0)], whisper)
        quality = self.combine(whisper, {**self.ok, "whisper": self.whisper_diag((7.0, 10.0))})["quality"]
        self.assertEqual(self.main, want)
        self.assert_time_order()
        self.assertEqual(quality["vote"], stats)
        self.assertEqual(self.main[1]["w"], "дважды")
        self.assertNotIn("ещё", [w["w"] for w in self.main])      # на отрезке без Whisper не вставляем
        self.assertEqual(self.main[-3]["w"], "восемь")
        self.assertEqual((self.main[-2]["w"], self.main[-2]["alt_ins"]), ("девять", "ещё"))

    def test_two_separate_failed_spans(self):
        spans = [(1.0, 3.0), (6.0, 8.0)]
        whisper = self.whisper_without(*spans)
        diag = {**self.ok, "whisper": {"chunks": [
            {"state": "ok", "own_span": [0.0, 1.0]}, {"state": "error", "own_span": [1.0, 3.0]},
            {"state": "ok", "own_span": [3.0, 6.0]}, {"state": "empty_unknown", "own_span": [6.0, 8.0]},
            {"state": "ok", "own_span": [8.0, 10.0]}]}}
        want, stats = self.expected(spans, whisper)
        quality = self.combine(whisper, diag)["quality"]
        self.assertEqual(self.main, want)
        self.assert_time_order()
        self.assertEqual(quality["vote3_partial"], {"reason": "whisper_incomplete", "spans": 2, "seconds": 4.0})
        self.assertEqual(quality["asr_incomplete_chunks"], 1)
        self.assertNotIn("vote3_skipped", quality)
        self.assertEqual(self.main[1]["w"], "два")               # на первом отрезке — только сверка
        self.assertEqual(self.main[1]["alt"], "дважды")
        # вне отрезков — «два из трёх»: Whisper поддержал RNN-T против CTC
        self.assertEqual((self.main[3]["w"], self.main[3]["agree"]), ("четыре", True))
        self.assertNotIn("alt", self.main[3])
        self.assertEqual(quality["vote"]["agree2"], 1)

    def test_adjacent_failed_chunks_form_one_span(self):
        whisper = self.whisper_without((3.0, 6.0))
        diag = {**self.ok, "whisper": {"chunks": [
            {"state": "ok", "own_span": [0.0, 3.0]}, {"state": "error", "own_span": [3.0, 4.5]},
            {"state": "error", "own_span": [4.5, 6.0]}, {"state": "ok", "own_span": [6.0, 10.0]}]}}
        quality = self.combine(whisper, diag)["quality"]
        self.assertEqual(quality["vote3_partial"], {"reason": "whisper_incomplete", "spans": 1, "seconds": 3.0})

    def test_filtered_whisper_chunk_is_excluded_like_a_failure(self):
        whisper = self.whisper_without((3.0, 6.0))
        want, stats = self.expected([(3.0, 6.0)], whisper)
        diag = {**self.ok, "whisper": self.whisper_diag((3.0, 6.0), state="ok", filtered_all=True)}
        got = self.combine(whisper, diag)
        self.assertEqual(self.main, want)
        self.assertEqual(got["quality"]["vote3_partial"]["reason"], "whisper_incomplete")
        self.assertEqual(got["evidence"]["whisper"], [(3.0, 6.0, "filtered", ["primary", "second"])])

    def test_second_model_failure_excludes_only_its_span(self):
        self.ctc = [w for w in self.ctc if not 3.0 <= w["s"] < 6.0]      # там у CTC слов нет
        whisper = self.whisper_without()                                  # Whisper слышит всё
        diag = {"primary": chunks((0.0, 10.0)),
                "second": {"chunks": [{"state": "ok", "own_span": [0.0, 3.0]},
                                      {"state": "error", "own_span": [3.0, 6.0]},
                                      {"state": "ok", "own_span": [6.0, 10.0]}]},
                "whisper": chunks((0.0, 10.0))}
        want, stats = self.expected([(3.0, 6.0)], whisper)
        whole = copy.deepcopy(self.main)
        words.mark_alt(whole, self.ctc)
        got = self.combine(whisper, diag)
        self.assertEqual(self.main, want)
        self.assert_time_order()
        quality = got["quality"]
        self.assertEqual(quality["vote"], stats)
        self.assertEqual(quality["vote3_partial"], {"reason": "asr_incomplete", "spans": 1, "seconds": 3.0})
        self.assertEqual(quality["asr_incomplete_chunks"], 1)
        self.assertNotIn("vote3_skipped", quality)
        self.assertEqual(self.main[1]["w"], "дважды")
        span = [w for w in self.main if 3.0 <= w["s"] < 6.0]
        self.assertEqual(span, [w for w in whole if 3.0 <= w["s"] < 6.0])
        self.assertTrue(all(w["agree"] is False for w in span))
        self.assertEqual(got["evidence"]["other"], [(3.0, 6.0, "second", "error", ["primary"])])

    def test_primary_failure_still_disables_vote_for_whole_record(self):
        self.main = [w for w in self.main if not 3.0 <= w["s"] < 6.0]
        whole = copy.deepcopy(self.main)
        words.mark_alt(whole, self.ctc)
        diag = {"primary": {"chunks": [{"state": "ok", "own_span": [0.0, 3.0]},
                                       {"state": "error", "own_span": [3.0, 6.0]},
                                       {"state": "ok", "own_span": [6.0, 10.0]}]},
                "second": chunks((0.0, 10.0)), "whisper": chunks((0.0, 10.0))}
        quality = self.combine(self.whisper_without(), diag)["quality"]
        self.assertEqual(self.main, whole)
        self.assertEqual(quality, {"vote3_skipped": "asr_incomplete", "asr_incomplete_chunks": 1})

    def test_nothing_left_outside_spans_disables_vote(self):
        whole = copy.deepcopy(self.main)
        words.mark_alt(whole, self.ctc)
        diag = {**self.ok, "whisper": chunks((0.0, 10.0), state="empty_unknown")}
        quality = self.combine([], diag)["quality"]
        self.assertEqual(self.main, whole)
        self.assertEqual(quality, {"vote3_skipped": "whisper_incomplete"})

    def test_word_on_span_boundary(self):
        # слово ровно на начале отрезка исключено, ровно на конце — голосует
        self.ctc[3]["w"] = "четыре"
        self.ctc[7]["w"] = "семью"                    # индекс 7 — «семь» (6,0 с) после вставки «кошка»
        whisper = [t if t != "семь" else "семью" for t in self.whisper_without((3.0, 6.0))]
        self.main[3]["w"] = "черыре"                  # 3,0 с — на отрезке: только сверка с CTC
        quality = self.combine(whisper, {**self.ok, "whisper": self.whisper_diag((3.0, 6.0))})["quality"]
        self.assert_time_order()
        at = {w["s"]: w for w in self.main}
        self.assertEqual((at[3.0]["w"], at[3.0]["alt"], at[3.0]["agree"]), ("черыре", "четыре", False))
        self.assertNotIn("voted", at[3.0])
        self.assertEqual((at[6.0]["w"], at[6.0]["voted"], at[6.0]["model"]), ("семью", "ctc+whisper", "семь"))
        self.assertIn({"t": 6.0, "t2": 6.0, "model": "семь", "to": "семью"}, quality["asr_votes"])

    def test_words_heard_only_by_second_on_empty_primary_span(self):
        # на сбойном отрезке Whisper основная модель слов не дала, вторая — дала: как в mark_alt,
        # они становятся alt_ins следующего слова основной модели
        self.main = [w for w in self.main if not 3.0 <= w["s"] < 6.0]
        self.ctc = [w for w in self.ctc if not 3.0 <= w["s"] < 6.0] + [wd("кошка", 4.0)]
        self.ctc.sort(key=lambda w: w["s"])
        whisper = self.whisper_without((3.0, 6.0))
        quality = self.combine(whisper, {**self.ok, "whisper": self.whisper_diag((3.0, 6.0))})["quality"]
        self.assert_time_order()
        nxt = next(w for w in self.main if w["s"] >= 6.0)
        self.assertEqual((nxt["w"], nxt["alt_ins"], nxt["agree"]), ("семь", "кошка", False))
        self.assertIn("vote3_partial", quality)


if __name__ == "__main__":
    unittest.main()
