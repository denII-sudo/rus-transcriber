"""Покусковая диагностика и ограниченный повтор без реальных моделей и записей."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-asr-data-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backends as B
import words as W


def result(text="слово"):
    return SimpleNamespace(tokens=[" " + text] if text else [], timestamps=[0.1] if text else [],
                           logprobs=[-0.1] if text else [], text=text)


class Fake:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def with_timestamps(self):
        return self

    def recognize(self, pieces, **kw):
        self.calls.append([len(p) for p in pieces])
        return self.fn(pieces, len(self.calls))


def backend(model, batch=2):
    a = B.OnnxASR.__new__(B.OnnxASR)
    a.model, a.batch, a.pad_to, a.language = model, batch, 0, "ru"
    return a


class DiagnosticTests(unittest.TestCase):
    def test_success_has_identical_words_and_no_retry(self):
        audio = np.ones(32000, np.float32)
        chunks = [{"a": 0, "b": 1, "own": (0, 1)}, {"a": 1, "b": 2, "own": (1, 2)}]
        f = Fake(lambda xs, n: [result() for _ in xs])
        plain = W.transcribe(backend(f), audio, chunks)
        d = []
        observed = W.transcribe(backend(f), audio, chunks, diagnostics=d)
        self.assertEqual(plain, observed)
        self.assertEqual([x["state"] for x in d], ["ok", "ok"])
        self.assertEqual(d[1]["audio_span"], [1, 2])
        self.assertEqual(d[0]["owned_words"], 1)
        self.assertNotIn("слово", str(d))

    def test_batch_error_retry_only_affected_and_preserves_indices(self):
        def call(xs, n):
            if n == 1:
                raise RuntimeError("private message not recorded")
            return [result("да") for _ in xs]
        f = Fake(call)
        d = []
        r = backend(f, 2).transcribe_timestamped([np.zeros(30), np.zeros(10), np.zeros(20)], diagnostics=d)
        self.assertEqual([x.text for x in r], ["да"] * 3)
        self.assertEqual([len(x["attempts"]) for x in d], [1, 2, 2])
        self.assertTrue(d[1]["attempts"][1]["retry"])
        self.assertNotIn("private", str(d))

    def test_single_error_retries_once_and_progress_finishes(self):
        def call(xs, n):
            raise RuntimeError("failure")
        f, d, progress = Fake(call), [], []
        r = backend(f, 1).transcribe_timestamped([np.zeros(10)], diagnostics=d,
                                                  progress=lambda a, b: progress.append((a, b)))
        self.assertEqual(r, [None])
        self.assertEqual(len(f.calls), 2)
        self.assertEqual(d[0]["state"], "error")
        self.assertEqual(progress, [(1, 1)])

    def test_empty_is_unknown_and_never_retried(self):
        f = Fake(lambda xs, n: [result("") for _ in xs])
        d = []
        backend(f).transcribe_timestamped([np.zeros(10)], diagnostics=d)
        self.assertEqual(d[0]["state"], "empty_unknown")
        self.assertEqual(len(f.calls), 1)

    def test_malformed_response_retries_instead_of_splitting_string(self):
        f = Fake(lambda xs, n: "да" if n == 1 else ["восстановлено"])
        a = B.WhisperOV.__new__(B.WhisperOV)
        a.adapter, a.batch, a.language = f, 2, "ru"
        d = []
        r = a.transcribe_texts([np.zeros(10), np.zeros(20)], diagnostics=d)
        self.assertEqual(r, ["восстановлено", "восстановлено"])
        self.assertEqual(d[0]["attempts"][0]["error_type"], "TypeError")
        self.assertEqual(len(f.calls), 3)

    def test_only_failed_retry_stays_failed(self):
        def call(xs, n):
            if n == 1 or len(xs[0]) == 10:
                raise RuntimeError("error")
            return ["успех"]
        f = Fake(call)
        d = []
        r = backend(f).transcribe_texts([np.zeros(10), np.zeros(20)], diagnostics=d)
        self.assertEqual(r, ["", "успех"])
        self.assertEqual([x["state"] for x in d], ["error", "ok"])

    def test_original_and_padded_lengths(self):
        f = Fake(lambda xs, n: [result()])
        a = backend(f, 1)
        a.pad_to = 2
        d = []
        a.transcribe_timestamped([np.ones(16000, np.float32)], diagnostics=d)
        self.assertEqual(d[0]["samples"], 16000)
        self.assertEqual(d[0]["attempts"][0]["input_samples"], 32000)


if __name__ == "__main__":
    unittest.main()
