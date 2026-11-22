"""Проверка перехода Whisper между устройствами без моделей и реальных записей."""
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

    def transcribe_timestamped(self, pieces, progress=None, diagnostics=None):
        diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
        return [SimpleNamespace(tokens=[" test"], timestamps=[0.1], logprobs=[-0.1], text="test")
                for _ in pieces]


class WhisperFallbackTests(unittest.TestCase):
    def test_missed_warning_links_to_its_diagnostic_candidate(self):
        items = [{"id": "r1", "t": 0.2, "t1": 0.2, "t2": 1.2, "reasons": ["missed"]},
                 {"id": "r2", "t": 1.2, "t1": 1.2, "t2": 2.0, "reasons": ["asr"]}]
        trace = {"candidates": [
            {"id": "missed-0001", "start": 0.2, "end": 1.2, "decision": "warning"}]}
        self.assertIs(worker.link_missed_diagnostics(items, trace), items)
        self.assertEqual(items[0]["diagnostic_id"], "missed-0001")
        self.assertNotIn("diagnostic_id", items[1])

    def run_worker(self, cpu_fails=False, load_fails=False, mode="all", disagreement=None,
                   primary_fails=False, second_load_fails=False):
        calls = []

        class GPU:
            name = "fake-gpu"

            def __init__(self, *args, **kwargs):
                if load_fails:
                    raise RuntimeError("synthetic GPU load failure")

            def transcribe_texts(self, pieces, progress=None, diagnostics=None):
                class Adapter:
                    def recognize(self, batch, **kwargs):
                        if len(batch) > 1 or len(batch[0]) == 16000:
                            raise RuntimeError("synthetic")
                        return ["ok"]
                batches = [list(range(len(pieces)))] if pieces else []
                result, records = backends._observed_recognize(Adapter(), pieces, batches,
                                                                progress=progress)
                diagnostics.extend(records)
                return result

        class CPU:
            name = "fake-cpu"

            def transcribe_texts(self, pieces, diagnostics=None, progress=None):
                calls.extend(len(p) for p in pieces)
                diagnostics.extend({"index": i, "state": "error" if cpu_fails else "ok",
                                    "attempts": [{"result": "error" if cpu_fails else "ok", "batch": [i]}]}
                                   for i in range(len(pieces)))
                return ["" if cpu_fails else "recovered" for _ in pieces]

        class FailedMain(Main):
            def transcribe_timestamped(self, pieces, progress=None, diagnostics=None):
                diagnostics.extend({"index": i, "state": "error", "attempts": []} for i in range(len(pieces)))
                return [None for _ in pieces]

        def make_asr(cfg):
            if load_fails and cfg.get("asr_backend") == "onnx-cpu":
                raise RuntimeError("synthetic CPU load failure")
            if second_load_fails and cfg.get("asr_model") == "second":
                raise RuntimeError("synthetic second load failure")
            if primary_fails and cfg.get("asr_model") == "primary":
                return FailedMain(), None
            return (CPU() if cfg.get("asr_backend") == "onnx-cpu" else Main()), None

        cfg = {"asr_model": "primary", "asr_second_model": "second", "asr_third_model": "whisper",
               "asr_third_device": "GPU", "asr_third_mode": mode, "asr_third_long_chunks": False,
               "emotions": False, "review_missed": False}
        a = SimpleNamespace(whisper="", whisper_max_repeat=-1, no_emotions=True)
        quality = {"speaker_fixes": [], "vocab_fixes": []}
        chunks = [{"a": 0, "b": 1, "own": (0, 1)}, {"a": 1, "b": 3, "own": (1, 3)}]
        with patch.object(backends, "WhisperOV", GPU), patch.object(words, "make_chunks", return_value=chunks), \
             patch.object(words, "assign_voices", return_value=0), patch.object(words, "build_replies", return_value=[]), \
             patch.object(words, "overlap_regions", return_value=[]), patch.object(words, "doubt_flags", return_value=0), \
             patch.object(words, "review_items", return_value=[]), patch.object(words, "vote3", return_value={}), \
             patch.object(words, "disagreement_times", return_value=disagreement or []):
            data = worker.run_words(np.zeros(48000, dtype=np.float32), "synthetic", cfg, Status(),
                                    [(0., 3., "s")], {"s": "L"}, [], None, None, {}, {},
                                    {"cents": {"L": [1]}}, None, {}, None, quality, make_asr, a)
        return calls, quality, data

    def test_only_failed_chunk_is_retried_on_cpu(self):
        calls, quality, _ = self.run_worker()
        self.assertEqual(calls, [16000])
        records = quality["asr_diagnostics"]["whisper"]["chunks"]
        self.assertEqual([r["state"] for r in records], ["ok", "ok"])
        self.assertEqual([a["backend"] for a in records[0]["attempts"]],
                         ["whisper-ov:GPU", "whisper-ov:GPU", "onnx-cpu:CPU"])
        self.assertNotIn("asr_incomplete_chunks", quality)
        self.assertEqual(quality["asr_run"]["models"]["whisper"]["id"], "whisper")

    def test_unrecovered_chunk_remains_incomplete(self):
        calls, quality, data = self.run_worker(cpu_fails=True)
        self.assertEqual(calls, [16000])
        self.assertEqual(quality["asr_incomplete_chunks"], 1)
        self.assertEqual(quality["asr_diagnostics"]["whisper"]["chunks"][0]["state"], "error")
        self.assertEqual(data["review"][0]["asr_problem"]["words_from"], ["primary", "second"])
        # голосование пропущено только на отрезке сбойного куска (0–1 с), на втором куске идёт
        self.assertEqual(quality["vote3_partial"], {"reason": "whisper_incomplete", "spans": 1, "seconds": 1})
        self.assertNotIn("vote3_skipped", quality)
        self.assertEqual(quality["vote"], {})

    def test_all_whisper_chunks_failed_disable_vote(self):
        _, quality, _ = self.run_worker(load_fails=True)
        self.assertEqual(quality["vote3_skipped"], "whisper_incomplete")
        self.assertNotIn("vote3_partial", quality)
        self.assertNotIn("vote", quality)

    def test_both_model_load_failures_report_each_chunk(self):
        _, quality, data = self.run_worker(load_fails=True)
        records = quality["asr_diagnostics"]["whisper"]["chunks"]
        self.assertEqual(quality["asr_incomplete_chunks"], 2)
        self.assertEqual([r["chunk_index"] for r in records], [0, 1])
        self.assertEqual([a["backend"] for a in records[0]["attempts"]],
                         ["whisper-ov:GPU", "onnx-cpu:CPU"])
        self.assertEqual(len(data["review"]), 2)

    def test_off_does_not_load_whisper(self):
        _, quality, data = self.run_worker(load_fails=True, mode="off")
        self.assertNotIn("whisper", quality["asr_diagnostics"])
        self.assertNotIn("asr_incomplete_chunks", quality)
        self.assertFalse(quality["missed_speech_diagnostics"]["enabled"])
        self.assertEqual(quality["missed_speech_diagnostics"]["candidates"], [])
        self.assertEqual(data["quality"]["missed_speech"], [])

    def test_doubt_keeps_original_chunk_index(self):
        calls, quality, _ = self.run_worker(mode="doubt", disagreement=[1.5])
        self.assertEqual(calls, [])
        records = quality["asr_diagnostics"]["whisper"]["chunks"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["index"], 0)
        self.assertEqual(records[0]["chunk_index"], 1)

    def test_doubt_without_disagreement_has_no_whisper_chunks(self):
        calls, quality, _ = self.run_worker(mode="doubt")
        self.assertEqual(calls, [])
        self.assertEqual(quality["asr_diagnostics"]["whisper"]["chunks"], [])

    def test_primary_failure_has_review_with_second_pass_evidence(self):
        _, quality, data = self.run_worker(mode="off", primary_fails=True)
        self.assertEqual(quality["asr_incomplete_chunks"], 2)
        problems = [item["asr_problem"] for item in data["review"]]
        self.assertEqual([p["model"] for p in problems], ["primary", "primary"])
        self.assertEqual(problems[0]["words_from"], ["second"])

    def test_primary_failure_disables_three_model_vote(self):
        _, quality, _ = self.run_worker(primary_fails=True)
        self.assertEqual(quality["vote3_skipped"], "asr_incomplete")
        self.assertNotIn("vote", quality)

    def test_second_pass_load_failure_is_incomplete(self):
        _, quality, data = self.run_worker(mode="off", second_load_fails=True)
        self.assertEqual(quality["asr_incomplete_chunks"], 2)
        self.assertEqual([x["asr_problem"]["model"] for x in data["review"]], ["second", "second"])
        self.assertEqual(data["review"][0]["asr_problem"]["words_from"], ["primary"])


class ThirdModelStageLabelTests(unittest.TestCase):
    """Подпись этапа третьей модели в статусе называет настоящую модель."""

    def stages(self, model, cfg_extra=None):
        seen = []

        class Recorder:
            def set(self, **kwargs):
                if "stage" in kwargs:
                    seen.append(kwargs["stage"])

        class Fake:
            name = "fake-third"

            def transcribe_texts(self, pieces, progress=None, diagnostics=None):
                progress(1, 1)
                diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
                return ["ok" for _ in pieces]

        def prog(lo, hi, name):
            return lambda done, total: seen.append(name)

        st = Recorder()
        plan = {"model": model, "mode": "all", "max_repeat": 3, "chunks": [{"a": 0, "b": 1, "own": (0, 1)}],
                "selected": [0], "pieces": [np.zeros(16000, dtype=np.float32)]}
        cfg = {"asr_third_device": "CPU", **(cfg_extra or {})}
        quality = {"asr_diagnostics": {}, "asr_run": {"models": {}}}
        result = worker.whisper_pass(plan, [], cfg, st, quality, lambda c: (Fake(), None), prog, {})
        self.assertEqual(result, ["ok"])
        self.assertNotIn("asr3_errors", quality)
        return seen

    def test_label_names(self):
        self.assertEqual(worker._third_label("onnx-community/whisper-large-v3-turbo"), "Whisper")
        self.assertEqual(worker._third_label("multilingual_large_ctc"), "Multilingual large")
        self.assertEqual(worker._third_label("some_other_model"), "some_other_model")
        self.assertEqual(worker._third_label("multilingual_large_ctc", "whisper-ct2"), "Whisper")

    def test_whisper_stage_text_is_unchanged(self):
        seen = self.stages("onnx-community/whisper-large-v3-turbo")
        self.assertIn("третья модель: загрузка Whisper (CPU)", seen)
        self.assertIn("третья модель (Whisper, CPU)", seen)

    def test_multilingual_stage_text_names_real_model(self):
        seen = self.stages("multilingual_large_ctc")
        self.assertIn("третья модель: загрузка Multilingual large (CPU)", seen)
        self.assertIn("третья модель (Multilingual large, CPU)", seen)
        self.assertFalse([s for s in seen if "Whisper" in s])


if __name__ == "__main__":
    unittest.main()
