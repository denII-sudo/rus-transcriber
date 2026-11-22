"""Проверка перехода Whisper между устройствами без моделей и реальных записей."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import backends
import transcript
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


class ThirdModelEngineChoiceTests(unittest.TestCase):
    """Движки, умеющие только Whisper, не подменяют выбранную другую третью модель."""

    WHISPER = "onnx-community/whisper-large-v3-turbo"
    MULTI = "multilingual_large_ctc"

    def run_pass(self, model, cfg_extra, ct2_fails=True, ov_fails=True):
        events, made, errors = [], [], io.StringIO()

        class Fake:
            def __init__(self, name):
                self.name = name

            def transcribe_texts(self, pieces, progress=None, diagnostics=None):
                diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
                return ["ok" for _ in pieces]

        def engine(name, fails):
            def build(*args, **kwargs):
                events.append(name)
                if fails:
                    raise RuntimeError(f"synthetic {name} load failure")
                return Fake(name)
            return build

        def make_asr(cfg):
            events.append("onnx-cpu")
            made.append(dict(cfg))
            return Fake("onnx-cpu"), None

        class St:
            def set(self, **kwargs):
                pass

        plan = {"model": model, "mode": "all", "max_repeat": 3, "chunks": [{"a": 0, "b": 1, "own": (0, 1)}],
                "selected": [0], "pieces": [np.zeros(16000, dtype=np.float32)]}
        quality, timing = {"asr_diagnostics": {}, "asr_run": {"models": {}}}, {}
        with patch.object(backends, "WhisperCT2", engine("whisper-ct2", ct2_fails)), \
             patch.object(backends, "WhisperOV", engine("whisper-ov", ov_fails)), \
             contextlib.redirect_stderr(errors):
            result = worker.whisper_pass(plan, [], {"asr_third_device": "CPU", **cfg_extra}, St(), quality,
                                         make_asr, lambda lo, hi, name: (lambda done, total: None), timing)
        return result, events, made, timing, quality, errors.getvalue()

    def test_other_model_ignores_whisper_ct2_backend(self):
        result, events, made, timing, quality, err = self.run_pass(
            self.MULTI, {"asr_third_backend": "whisper-ct2"})
        self.assertEqual(result, ["ok"])
        self.assertEqual(events, ["onnx-cpu"])                       # Whisper-движок не вызывался
        self.assertEqual(made[0]["asr_model"], self.MULTI)
        self.assertEqual(made[0]["asr_backend"], "onnx-cpu")
        self.assertEqual(timing["asr3_model"], self.MULTI)
        self.assertEqual(timing["asr3_device"], "onnx-cpu:CPU:int8")
        self.assertNotIn("asr3_errors", quality)
        lines = [x for x in err.splitlines() if "пропущен" in x]
        self.assertEqual(len(lines), 1)
        self.assertIn("whisper-ct2", lines[0])
        self.assertIn("умеет только Whisper", lines[0])

    def test_other_model_ignores_gpu_device(self):
        result, events, made, timing, _, err = self.run_pass(self.MULTI, {"asr_third_device": "GPU"})
        self.assertEqual(result, ["ok"])
        self.assertEqual(events, ["onnx-cpu"])
        self.assertEqual(made[0]["asr_model"], self.MULTI)
        self.assertEqual(timing["asr3_device"], "onnx-cpu:CPU:int8")
        lines = [x for x in err.splitlines() if "пропущен" in x]
        self.assertEqual(len(lines), 1)
        self.assertIn("whisper-ov", lines[0])

    def test_other_model_with_both_whisper_engines_gives_one_line(self):
        _, events, _, _, _, err = self.run_pass(
            self.MULTI, {"asr_third_backend": "whisper-ct2", "asr_third_device": "GPU"})
        self.assertEqual(events, ["onnx-cpu"])
        lines = [x for x in err.splitlines() if "пропущен" in x]
        self.assertEqual(len(lines), 1)
        self.assertIn("whisper-ct2", lines[0])
        self.assertIn("whisper-ov", lines[0])

    def test_other_model_on_cpu_without_whisper_backend_is_silent(self):
        _, events, _, _, _, err = self.run_pass(self.MULTI, {})
        self.assertEqual(events, ["onnx-cpu"])
        self.assertNotIn("пропущен", err)

    def test_whisper_model_keeps_attempt_order(self):
        result, events, made, timing, quality, err = self.run_pass(
            self.WHISPER, {"asr_third_backend": "whisper-ct2", "asr_third_device": "GPU"})
        self.assertEqual(result, ["ok"])
        self.assertEqual(events, ["whisper-ct2", "whisper-ov", "onnx-cpu"])
        self.assertEqual(made[0]["asr_model"], self.WHISPER)
        self.assertEqual(timing["asr3_device"], "onnx-cpu:CPU:int8")
        self.assertEqual(len(quality["asr3_errors"]), 2)
        self.assertNotIn("пропущен", err)

    def test_whisper_model_uses_ct2_first_when_it_loads(self):
        _, events, _, timing, _, _ = self.run_pass(
            self.WHISPER, {"asr_third_backend": "whisper-ct2", "asr_third_device": "GPU"}, ct2_fails=False)
        self.assertEqual(events, ["whisper-ct2"])
        self.assertEqual(timing["asr3_device"], "whisper-ct2:CPU:int8")
        self.assertEqual(timing["asr3_model"], self.WHISPER)

    def test_whisper_name_check_ignores_case(self):
        # faster-whisper умеет только large-v3-turbo (с 22.11.2026 другой Whisper он пропускает),
        # поэтому регистр проверяется на имени этой модели
        _, events, _, _, _, _ = self.run_pass("Org/Whisper-Large-V3-Turbo", {"asr_third_backend": "whisper-ct2"})
        self.assertEqual(events[0], "whisper-ct2")

    def test_other_whisper_skips_ct2_but_keeps_openvino(self):
        result, events, made, timing, _, err = self.run_pass(
            "Org/Whisper-Small", {"asr_third_backend": "whisper-ct2", "asr_third_device": "GPU"})
        self.assertEqual(result, ["ok"])
        self.assertEqual(events, ["whisper-ov", "onnx-cpu"])     # OpenVINO Whisper грузит модель по имени
        self.assertEqual(made[0]["asr_model"], "Org/Whisper-Small")
        lines = [x for x in err.splitlines() if "пропущен" in x]
        self.assertEqual(len(lines), 1)
        self.assertIn("whisper-ct2", lines[0])


class ThirdModelTranscriptLabelTests(unittest.TestCase):
    """Вариант третьей модели в стенограмме подписан настоящей моделью; без сведений — «Whisper»."""

    ITEM = {"id": "r1", "t": 5.0, "speaker": "Анна", "text": "дело", "reasons": ["asr"],
            "alt_ctc": "тело", "alt_whisper": "дали"}

    def data(self, timing):
        d = {"file": "x.wav", "duration": "00:00:10", "speakers": ["Анна"], "review": [dict(self.ITEM)],
             "segments": [{"speaker": "Анна", "start": 0, "end": 10, "text": "дело", "words": []}]}
        if timing is not None:
            d["timing"] = timing
        return d

    def line(self, timing):
        md = transcript.render_md(self.data(timing))
        return [x for x in md.splitlines() if "CTC:" in x][0]

    def test_whisper_run_is_unchanged(self):
        expected = "- 00:00:05 Анна: «дело» — модели распознали по-разному; CTC: «тело»; Whisper: «дали»"
        self.assertEqual(self.line({"asr3_model": "onnx-community/whisper-large-v3-turbo",
                                    "asr3_device": "whisper-ov:GPU:fp16"}), expected)
        self.assertEqual(self.line(None), expected)
        self.assertEqual(self.line({}), expected)
        self.assertEqual(self.line({"asr3_model": None}), expected)
        self.assertEqual(self.line("не словарь"), expected)

    def test_other_model_is_named(self):
        line = self.line({"asr3_model": "multilingual_large_ctc", "asr3_device": "onnx-cpu:CPU:int8"})
        self.assertTrue(line.endswith("; Multilingual large: «дали»"))
        self.assertNotIn("Whisper", line)

    def test_unknown_model_is_named_by_id(self):
        line = self.line({"asr3_model": "some_other_model", "asr3_device": "onnx-cpu:CPU:int8"})
        self.assertTrue(line.endswith("; some_other_model: «дали»"))

    def test_old_result_where_whisper_engine_replaced_the_model_says_whisper(self):
        line = self.line({"asr3_model": "multilingual_large_ctc", "asr3_device": "whisper-ct2:CPU:int8"})
        self.assertTrue(line.endswith("; Whisper: «дали»"))

    def test_item_line_default_label(self):
        self.assertTrue(transcript._item_line(self.ITEM).endswith("; Whisper: «дали»"))

    def test_one_rule_for_worker_and_transcript(self):
        for model, backend in [(None, ""), ("", ""), ("whisper-x", ""), ("multilingual_large_ctc", ""),
                               ("multilingual_large_ctc", "whisper-ov"), ("some_other_model", "onnx-cpu")]:
            self.assertEqual(worker._third_label(model, backend), transcript.third_model_label(model, backend))


if __name__ == "__main__":
    unittest.main()
