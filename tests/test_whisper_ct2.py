"""Whisper через faster-whisper (WhisperCT2): откат на прежний движок без моделей
и реальных записей. Проверяются требования ревью PR #7: ошибочный фрагмент
faster-whisper повторяется на onnx-cpu, отсутствие пакета ведёт к прежней цепочке,
пустой текст ошибкой не считается."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import backends
import words
import worker


class Status:
    def set(self, **kwargs):
        pass


class Main:
    name = "fake-primary"
    model_name = "fake"
    tokens = ([" test"], [0.1])          # токены и времена каждого куска

    def transcribe_timestamped(self, pieces, progress=None, diagnostics=None):
        diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
        toks, ts = self.tokens
        return [SimpleNamespace(tokens=list(toks), timestamps=list(ts), logprobs=[-0.1] * len(toks),
                                text="".join(toks)) for _ in pieces]


class WhisperCT2WorkerTests(unittest.TestCase):
    """Сценарии worker.run_words с подменёнными бэкендами: ни одна модель не грузится."""

    def run_worker(self, fail_indices=(0,), empty_indices=(), ct2_load_fails=False, cpu_fails=False,
                   ct2_text="от faster-whisper", main_tokens=None):
        cpu_calls = []

        class CT2:
            name = "fake-ct2"
            compile_sec = None

            def __init__(self, *args, **kwargs):
                if ct2_load_fails:      # как отсутствие пакета faster-whisper
                    raise ImportError("No module named 'faster_whisper'")

            def transcribe_texts(self, pieces, progress=None, diagnostics=None):
                records, out = [], []
                for i, piece in enumerate(pieces):
                    if i in fail_indices:
                        state, result, text = "error", "error", ""
                    elif i in empty_indices:
                        state, result, text = "empty_unknown", "empty", ""
                    else:
                        state, result, text = "ok", "ok", ct2_text
                    records.append({"index": i, "samples": len(piece), "state": state,
                                    "attempts": [{"result": result, "error_type": "RuntimeError",
                                                  "retry": False, "batch": [i], "input_samples": len(piece),
                                                  "elapsed_sec": 0.01}]})
                    out.append(text)
                if diagnostics is not None:
                    diagnostics.extend(records)
                if progress:
                    progress(len(pieces), len(pieces))
                return out

        class CPU:
            name = "fake-onnx-cpu"

            def transcribe_texts(self, pieces, diagnostics=None, progress=None):
                cpu_calls.extend(len(p) for p in pieces)
                diagnostics.extend({"index": i, "state": "error" if cpu_fails else "ok",
                                    "attempts": [{"result": "error" if cpu_fails else "ok", "batch": [i]}]}
                                   for i in range(len(pieces)))
                return ["" if cpu_fails else "от onnx-cpu" for _ in pieces]

        def make_asr(cfg):
            return (CPU() if cfg.get("asr_backend") == "onnx-cpu" else Main()), None

        cfg = {"asr_model": "primary", "asr_second_model": "second", "asr_third_model": "onnx-community/whisper-large-v3-turbo",
               "asr_third_device": "CPU", "asr_third_backend": "whisper-ct2", "asr_third_mode": "all",
               "asr_third_long_chunks": False, "asr_third_quant_cpu": "int8", "asr_third_batch": 4,
               "emotions": False, "review_missed": False}
        a = SimpleNamespace(whisper="", whisper_max_repeat=-1, no_emotions=True)
        quality = {"speaker_fixes": [], "vocab_fixes": []}
        chunks = [{"a": 0, "b": 1, "own": (0, 1)}, {"a": 1, "b": 3, "own": (1, 3)}]
        with patch.object(backends, "WhisperCT2", CT2), \
             patch.object(words, "make_chunks", return_value=chunks), \
             patch.object(words, "assign_voices", return_value=0), \
             patch.object(words, "build_replies", return_value=[]), \
             patch.object(words, "overlap_regions", return_value=[]), \
             patch.object(words, "doubt_flags", return_value=0), \
             patch.object(words, "review_items", return_value=[]), \
             patch.object(words, "vote3", return_value={}) as vote3, \
             patch.object(Main, "tokens", main_tokens or Main.tokens), \
             patch.object(words, "disagreement_times", return_value=[]):
            data = worker.run_words(np.zeros(48000, dtype=np.float32), "synthetic", cfg, Status(),
                                    [(0., 3., "s")], {"s": "L"}, [], None, None, {}, {},
                                    {"cents": {"L": [1]}}, None, {}, None, quality, make_asr, a)
        self.vote3_args = vote3.call_args
        return cpu_calls, quality, data

    def test_failed_chunk_is_retried_on_onnx_cpu(self):
        cpu_calls, quality, _ = self.run_worker()
        self.assertEqual(cpu_calls, [16000], "повтор должен касаться только ошибочного фрагмента")
        records = quality["asr_diagnostics"]["whisper"]["chunks"]
        self.assertEqual([r["state"] for r in records], ["ok", "ok"])
        self.assertEqual([a["backend"] for a in records[0]["attempts"]], ["whisper-ct2:CPU", "onnx-cpu:CPU"])
        self.assertEqual([a["backend"] for a in records[1]["attempts"]], ["whisper-ct2:CPU"])
        self.assertNotIn("asr_incomplete_chunks", quality)

    def test_unrecovered_chunk_stays_incomplete(self):
        cpu_calls, quality, _ = self.run_worker(cpu_fails=True)
        self.assertEqual(cpu_calls, [16000])
        self.assertEqual(quality["asr_incomplete_chunks"], 1)
        records = quality["asr_diagnostics"]["whisper"]["chunks"]
        self.assertEqual(records[0]["state"], "error")
        self.assertEqual([a["backend"] for a in records[0]["attempts"]], ["whisper-ct2:CPU", "onnx-cpu:CPU"])

    def test_missing_package_uses_previous_chain(self):
        cpu_calls, quality, _ = self.run_worker(fail_indices=(), ct2_load_fails=True)
        self.assertEqual(cpu_calls, [16000, 32000], "при ImportError расчёт идёт прежней цепочкой")
        records = quality["asr_diagnostics"]["whisper"]["chunks"]
        self.assertEqual([r["state"] for r in records], ["ok", "ok"])
        self.assertEqual([a["backend"] for a in records[0]["attempts"]], ["whisper-ct2:CPU", "onnx-cpu:CPU"])
        self.assertEqual(records[0]["attempts"][0]["error_type"], "ImportError")
        self.assertTrue(any("ImportError" in e for e in quality.get("asr3_errors", [])))
        self.assertEqual(quality["asr_run"]["models"]["whisper"]["backend"], "fake-onnx-cpu")

    def test_empty_text_is_not_retried(self):
        cpu_calls, quality, _ = self.run_worker(fail_indices=(), empty_indices=(0,))
        self.assertEqual(cpu_calls, [], "пустой текст — не ошибка, повтор не нужен")
        records = quality["asr_diagnostics"]["whisper"]["chunks"]
        self.assertEqual([r["state"] for r in records], ["empty_unknown", "ok"])

    def test_punctuation_only_tokens_do_not_reach_the_vote(self):
        # «—» отдельным токеном в начале куска GigaAM и отдельным «словом» в тексте Whisper:
        # в голосование «два из трёх» не попадает слово, у которого norm_word пуст
        self.run_worker(fail_indices=(), ct2_text="— от faster-whisper —",
                        main_tokens=([" —", " test"], [0.05, 0.1]))
        main, ctc, third = self.vote3_args.args
        self.assertEqual([w["w"] for w in main], ["— test", "— test"])
        self.assertEqual([w["s"] for w in main], [0.1, 1.1])        # время — у слова, не у знака
        self.assertEqual([w["w"] for w in ctc], ["— test", "— test"])
        self.assertEqual(third, ["от", "faster-whisper"] * 2)
        self.assertTrue(all(words.norm_word(w["w"]) for w in main + ctc))


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeModel:
    """Минимальная замена WhisperModel: без весов, без загрузки, без сети."""

    def __init__(self, fail_on_samples=None, text="привет"):
        self.fail_on_samples = fail_on_samples
        self.text = text

    def transcribe(self, audio, beam_size=1, vad_filter=False, language=None):
        if self.fail_on_samples is not None and len(audio) == self.fail_on_samples:
            raise RuntimeError("synthetic model failure")
        return [FakeSegment(self.text)], None


def make_backend(model):
    """WhisperCT2 с подменённой моделью: минуем __init__ (он грузит веса)."""
    obj = object.__new__(backends.WhisperCT2)
    obj.model = model
    obj.language = "ru"
    obj.name = "whisper-ct2:test"
    obj.model_name = "test"
    obj.compile_sec = None
    return obj


class WhisperCT2BackendTests(unittest.TestCase):
    def test_failed_piece_does_not_break_run(self):
        backend = make_backend(FakeModel(fail_on_samples=2))
        diagnostics, progress = [], []
        pieces = [np.zeros(1, dtype=np.float32), np.zeros(2, dtype=np.float32), np.zeros(0, dtype=np.float32)]
        texts = backend.transcribe_texts(pieces, progress=lambda done, total: progress.append((done, total)),
                                         diagnostics=diagnostics)
        self.assertEqual(texts, ["привет", "", ""])
        self.assertEqual([r["state"] for r in diagnostics], ["ok", "error", "error"])
        self.assertEqual(diagnostics[1]["attempts"][0]["error_type"], "RuntimeError")
        self.assertEqual(diagnostics[2]["attempts"][0]["error_type"], "ValueError")
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])

    def test_missing_package_raises_on_creation(self):
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with self.assertRaises(ImportError):
                backends.WhisperCT2()


if __name__ == "__main__":
    unittest.main()
