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
    # Whisper на видеокарте: ошибка загрузки GPU, затем onnx-cpu с пустым куском
    "gpu_then_cpu": dict(cfg_extra={"asr_third_device": "GPU", "asr_third_long_chunks": False},
                         whisper={"whisper-ov": {"fail_load": True}, "onnx-cpu": {"empty": (0,)}}),
    # ограничения из аргументов запуска: без ограничения повторов, Whisper только в местах расхождения
    "args_override": dict(args=SimpleNamespace(whisper="doubt", whisper_max_repeat=0, no_emotions=True)),
}

EXPECTED = {
    "three_models": "15c4ca2808faa51ab159784b7a9b7e29dcf3e513c97ef78f4069d071692e1aec",
    "second_fails": "e2638ce9942cac3ec1c3a64bee7520d84b18ee61d8c6b8f7af9ad2e4650ad259",
    "whisper_fails": "49b8845fcbd2f899a6b5b5b362c75691ec535f1655fbc99aad31bc4cd4042f39",
    "whisper_doubt": "fa19746260accfa914ac45de0833c31ca3975e6841c51dbea6c62e2dd2527717",
    "whisper_off": "6da72866aa7e965d4e27d478d1d7209d525aff76f16d9443940df227d1e0b6d6",
    "primary_chunk_error": "68a6adf67d3f089127f29d5401f4daf7121be1a4e662f78571fa402867ef3315",
    "ct2_retry": "7cda80af0b5a47d051729fa76b6492a36b3233989d09acd2f2730c7b1d1c2de1",
    "gpu_then_cpu": "26d25ba2b34eab9eb670b31db0277255120cad80ca5e29eb6692082ad0a86e80",
    "args_override": "ed4dd3dafd09eeab4b1e6e29775549f036e79036fb4bd23b0eb2a2b2f7bf5347",
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


if __name__ == "__main__":
    unittest.main()
