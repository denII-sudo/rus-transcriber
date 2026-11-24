"""Тест скорости (bench.py) на общих функциях расшифровки (РАР-4.2, шаг 4).

Вариант считается так же, как в расшифровке: куски — words.make_chunks (у третьей модели —
worker.whisper_plan), основная и вторая — worker.timestamped_pass (слова со временами и сведения
о кусках), третья — transcribe_texts со сведениями и worker._whisper_words; голосование в отчёте —
words.combine_hypotheses. Только синтетика и подменные модели.
"""
import atexit
import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-bench-common-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

import backends  # noqa: E402
import bench  # noqa: E402
import runinfo  # noqa: E402
import words as W  # noqa: E402

SR = 16000


def noise(sec, seed=1):
    """Сплошной шум без пауз: и расшифровка, и тест режут его в тихом месте внутри «речи»."""
    rnd = np.random.default_rng(seed)
    return (rnd.standard_normal(int(sec * SR)) * 0.1).astype(np.float32)


def result(*pairs):
    """TimestampedResult подменной модели: (слово, время в куске)."""
    return types.SimpleNamespace(tokens=[" " + w for w, _ in pairs], timestamps=[t for _, t in pairs],
                                 logprobs=None, text=" ".join(w for w, _ in pairs))


class Recorder:
    """Подменная модель: записывает длины кусков и способ вызова."""
    calls = []
    name = "cpu"
    model_name = "fake"

    def __init__(self, *a, **kw):
        pass

    def transcribe(self, piece):
        raise AssertionError("тест скорости должен вызывать модель как расшифровка")

    def transcribe_many(self, pieces, progress=None):
        raise AssertionError("тест скорости должен вызывать модель как расшифровка (со временами)")

    def transcribe_timestamped(self, pieces, progress=None, diagnostics=None):
        Recorder.calls.append(("timestamped", [len(p) / SR for p in pieces], diagnostics is not None))
        if diagnostics is not None:
            diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
        return [result(("раз", 0.2), ("два", 1.0)) for _ in pieces]

    def transcribe_texts(self, pieces, progress=None, diagnostics=None):
        Recorder.calls.append(("texts", [len(p) / SR for p in pieces], diagnostics is not None))
        if diagnostics is not None:
            diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
        return ["Продолжение следует..."] + ["раз два"] * (len(pieces) - 1)


def run(variant, audio, chunk_max=20.0, max_repeat=3):
    tmp = Path(tempfile.mkdtemp(prefix="rt-bench-common-one-", dir=_TMP))
    npy = tmp / "audio.npy"
    np.save(npy, audio)
    Recorder.calls = []
    out = io.StringIO()
    with mock.patch.object(backends, "OnnxASR", Recorder), mock.patch.object(backends, "WhisperCT2", Recorder), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        bench.run_one(variant, str(npy), "v3_e2e_rnnt", chunk_max, max_repeat)
    (line,) = [x for x in out.getvalue().splitlines() if x.startswith("@@RESULT@@")]
    return json.loads(line[len("@@RESULT@@"):])


RNNT = {"id": "onnx-cpu", "family": "rnnt", "asr_backend": "onnx-cpu"}
CTC = {"id": "ctc-cpu", "family": "ctc", "asr_backend": "onnx-cpu", "asr_model": "v3_e2e_ctc"}
WHISPER = {"id": "whisper-ct2", "family": "whisper", "asr_backend": "whisper-ct2"}
ML = {"id": "ml-cpu", "family": "multilingual", "asr_backend": "onnx-cpu", "asr_model": "multilingual_ctc"}


def spans(chunks):
    return [round(c["b"] - c["a"], 3) for c in chunks]


class ChunksLikeTranscriptionTests(unittest.TestCase):

    def test_primary_and_second_cut_by_make_chunks(self):
        audio = noise(45)
        expected = spans(W.make_chunks(audio, [(0.0, 45.0)]))
        self.assertGreater(sum(expected), 45.0)               # нахлёст внутри речи, как в расшифровке
        for variant in (RNNT, CTC):
            run(variant, audio)
            (_, warm, _), (kind, lens, with_diag) = Recorder.calls
            self.assertEqual(kind, "timestamped")
            self.assertTrue(with_diag)
            self.assertEqual([round(x, 3) for x in lens], expected)
            self.assertEqual(len(warm), 1)                    # прогрев — тем же вызовом, на первом куске

    def test_whisper_cut_by_whisper_plan(self):
        audio = noise(70)
        expected = spans(W.make_chunks(audio, [(0.0, 70.0)], max_len=W.WHISPER_MAX, min_len=W.WHISPER_MIN))
        run(WHISPER, audio)
        kind, lens, with_diag = Recorder.calls[-1]
        self.assertEqual((kind, with_diag), ("texts", True))
        self.assertEqual([round(x, 3) for x in lens], expected)

    def test_other_third_model_uses_chunks_of_primary(self):
        audio = noise(45)
        run(ML, audio)
        self.assertEqual([round(x, 3) for x in Recorder.calls[-1][1]],
                         spans(W.make_chunks(audio, [(0.0, 45.0)])))


class ModelCallLikeTranscriptionTests(unittest.TestCase):

    def test_primary_gives_timed_words_and_chunk_diagnostics(self):
        r = run(RNNT, noise(45))
        self.assertTrue(r["hyp_words"])
        self.assertTrue(all(set(w) >= {"w", "s", "e"} for w in r["hyp_words"]))
        self.assertEqual(len(r["diagnostics"]), r["chunks"])
        self.assertTrue(all("own_span" in d for d in r["diagnostics"]))
        self.assertEqual(r["text"], " ".join(w["w"] for w in r["hyp_words"]))

    def test_third_words_are_cleaned_like_transcription(self):
        r = run(WHISPER, noise(70))
        self.assertTrue(all(isinstance(w, str) for w in r["hyp_words"]))
        self.assertNotIn("Продолжение", r["text"])            # «субтитры» убраны, как в _whisper_words
        self.assertTrue(r["diagnostics"][0].get("filtered_all"))


def wd(w, s):
    return {"w": w, "s": float(s), "e": float(s) + 0.3, "c": 0.9}


class VoteLikeTranscriptionTests(unittest.TestCase):

    def hyps(self):
        main = [wd("раз", 1), wd("два", 2), wd("три", 7), wd("четыре", 8)]
        ctc = [wd("пять", 7), wd("четыре", 8)]                    # кусок 0–5 с у второй — ошибка
        third = ["раз", "два", "пять", "четыре"]
        diag = {"primary": {"chunks": [{"state": "ok", "own_span": [0.0, 5.0]},
                                       {"state": "ok", "own_span": [5.0, 10.0]}]},
                "second": {"chunks": [{"state": "error", "own_span": [0.0, 5.0]},
                                      {"state": "ok", "own_span": [5.0, 10.0]}]},
                "whisper": {"chunks": [{"state": "ok", "own_span": [0.0, 10.0]}]}}
        return main, ctc, third, diag

    def test_failed_chunk_excludes_its_span_as_in_transcription(self):
        main, ctc, third, diag = self.hyps()
        got = bench.tiebreak_stats(copy.deepcopy(main), copy.deepcopy(ctc), list(third), diag)
        fused = W.combine_hypotheses({"primary": copy.deepcopy(main), "second": copy.deepcopy(ctc),
                                      "whisper": list(third)}, diag)
        vote = fused["quality"]["vote"]
        self.assertEqual((got["for_ctc"], got["unresolved"], got["inserted"]),
                         (vote["replaced"], vote["disputed"], vote["inserted"]))
        self.assertEqual(got["skipped_spans"], 1)
        # прежний подсчёт по текстам сбойного куска не видел и считал голосование по всей записи
        old = bench.tiebreak_stats(" ".join(w["w"] for w in main), " ".join(w["w"] for w in ctc),
                                   " ".join(third))
        self.assertNotEqual((old["for_ctc"], old["unresolved"]), (got["for_ctc"], got["unresolved"]))

    def test_text_arguments_keep_old_meaning(self):
        self.assertEqual(bench.tiebreak_stats("раз два три", "раз пять три", "раз шесть три"),
                         {"contested": 1, "for_rnnt": 0, "for_ctc": 0, "unresolved": 1, "inserted": 0})


class MethodVersionTests(unittest.TestCase):

    def test_method_version_is_bench_2(self):
        self.assertEqual(runinfo.METHOD_VERSION, "bench-2")


if __name__ == "__main__":
    unittest.main()
