"""Тест скорости: устройства выбираются отдельно для RNN-T, CTC и Whisper.

Регрессии:
  11.11.2026 — тест скорости сохранял только настройки GigaAM, а `asr_third_device`
  оставался умолчанием «GPU». Теперь каждая группа измеряется и записывается сама.
  Чистая установка — процессор; NPU Meteor Lake (без FP32) не тестируется и не предлагается.

Запуск: python -m unittest discover -s tests
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# как во всех модулях тестов: временная папка данных задаётся ДО первого импорта программы
_TMP = tempfile.mkdtemp(prefix="rt-bench-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench  # noqa: E402
import paths  # noqa: E402
import settings  # noqa: E402

NPU_METEOR = {"available": True, "fp32": False, "name": "Intel(R) AI Boost",
              "capabilities": ["FP16", "INT8"]}
NPU_NEW = {"available": True, "fp32": True, "name": "Intel(R) AI Boost",
           "capabilities": ["FP32", "FP16", "INT8"]}


class DeviceOfTests(unittest.TestCase):
    def test_openvino_gpu(self):
        self.assertEqual(bench.device_of({"asr_backend": "openvino", "asr_device": "GPU"}), "GPU")
        self.assertEqual(bench.device_of({"asr_backend": "openvino", "asr_device": "gpu"}), "GPU")

    def test_openvino_cpu(self):
        self.assertEqual(bench.device_of({"asr_backend": "openvino", "asr_device": "CPU"}), "CPU")

    def test_openvino_without_device_is_gpu(self):
        self.assertEqual(bench.device_of({"asr_backend": "openvino"}), "GPU")

    def test_cpu_backends(self):
        self.assertEqual(bench.device_of({"asr_backend": "onnx-cpu"}), "CPU")
        self.assertEqual(bench.device_of({"asr_backend": "torch"}), "CPU")

    def test_directml_is_gpu(self):
        self.assertEqual(bench.device_of({"asr_backend": "dml"}), "GPU")

    def test_explicit_npu(self):
        self.assertEqual(bench.device_of({"device": "npu", "asr_backend": "openvino"}), "NPU")


class VariantPolicyTests(unittest.TestCase):
    def test_meteor_lake_npu_is_not_offered(self):
        ids = [v["id"] for v in bench.select_variants("", npu=NPU_METEOR)]
        self.assertNotIn("ov-npu-pad22", ids)
        self.assertNotIn("ctc-npu-pad22", ids)
        self.assertIn("onnx-cpu", ids)
        self.assertIn("ov-gpu", ids)

    def test_order_is_cpu_then_gpu_then_npu(self):
        chosen = bench.select_variants("", npu=NPU_NEW)
        devs = [v["device"] for v in chosen]
        last_cpu = max(i for i, d in enumerate(devs) if d == "CPU")
        first_gpu = min(i for i, d in enumerate(devs) if d == "GPU")
        last_gpu = max(i for i, d in enumerate(devs) if d == "GPU")
        first_npu = min(i for i, d in enumerate(devs) if d == "NPU")
        self.assertLess(last_cpu, first_gpu)
        self.assertLess(last_gpu, first_npu)

    def test_new_npu_is_testable(self):
        ids = [v["id"] for v in bench.select_variants("", npu=NPU_NEW)]
        self.assertIn("ov-npu-pad22", ids)
        self.assertIn("ctc-npu-pad22", ids)

    def test_whisper_is_measured_separately(self):
        ids = [v["id"] for v in bench.select_variants("", npu=NPU_METEOR)]
        self.assertIn("whisper-gpu", ids)
        self.assertIn("whisper-cpu", ids)
        self.assertLess(ids.index("whisper-cpu"), ids.index("whisper-gpu"))

    def test_no_whisper(self):
        ids = [v["id"] for v in bench.select_variants("", npu=NPU_METEOR, with_whisper=False)]
        self.assertNotIn("whisper-gpu", ids)

    def test_only_npu_on_meteor_lake_is_refused(self):
        self.assertEqual(bench.select_variants("ov-npu-pad22", npu=NPU_METEOR), [])


class SavedSettingsTests(unittest.TestCase):
    def test_cpu_main_gpu_ctc_cpu_whisper(self):
        main = {"family": "rnnt", "device": "CPU", "asr_backend": "onnx-cpu", "asr_batch": 16}
        second = {"family": "ctc", "device": "GPU", "asr_backend": "openvino",
                  "asr_device": "GPU", "asr_model": "v3_e2e_ctc", "asr_batch": 8}
        whisper = {"family": "whisper", "device": "CPU", "asr_third_device": "CPU"}
        out = bench.saved_settings(main, second, whisper)
        self.assertEqual(out["asr_backend"], "onnx-cpu")
        self.assertEqual(out["asr_device"], "CPU")
        self.assertEqual(out["asr_batch"], 16)
        self.assertEqual(out["asr_second_backend"], "openvino")
        self.assertEqual(out["asr_second_device"], "GPU")
        self.assertEqual(out["asr_second_batch"], 8)
        self.assertEqual(out["asr_third_device"], "CPU")
        self.assertFalse(out["asr_allow_npu"])

    def test_gpu_main_cpu_ctc(self):
        main = {"family": "rnnt", "device": "GPU", "asr_backend": "openvino",
                "asr_device": "GPU", "pad_to_seconds": 22}
        second = {"family": "ctc", "device": "CPU", "asr_backend": "onnx-cpu",
                  "asr_model": "v3_e2e_ctc"}
        out = bench.saved_settings(main, second, None)
        self.assertEqual(out["asr_device"], "GPU")
        self.assertEqual(out["pad_to_seconds"], 22)
        self.assertEqual(out["asr_second_device"], "CPU")
        self.assertNotIn("asr_third_device", out)

    def test_npu_choice_enables_npu(self):
        main = {"family": "rnnt", "device": "NPU", "asr_backend": "openvino", "asr_device": "NPU"}
        out = bench.saved_settings(main)
        self.assertEqual(out["asr_device"], "NPU")
        self.assertTrue(out["asr_allow_npu"])

    def test_npu_second_pass_enables_npu(self):
        main = {"family": "rnnt", "device": "CPU", "asr_backend": "onnx-cpu"}
        second = {"family": "ctc", "device": "NPU", "asr_backend": "openvino",
                  "asr_device": "NPU", "asr_model": "v3_e2e_ctc"}
        out = bench.saved_settings(main, second)
        self.assertTrue(out["asr_allow_npu"])

    def test_cpu_choice_is_written_explicitly(self):
        # после теста скорости выбор этой машины должен быть виден в config.json,
        # даже если он совпал с общим умолчанием «процессор»
        self.addCleanup(paths.CONFIG_FILE.unlink, missing_ok=True)
        settings.update(explicit=["asr_device"], asr_device="CPU")
        saved = json.loads(paths.CONFIG_FILE.read_text(encoding="utf-8"))
        self.assertEqual(saved["asr_device"], "CPU")


class BestOfTests(unittest.TestCase):
    def test_npu_with_wrong_text_is_not_chosen(self):
        results = [
            {"family": "rnnt", "id": "onnx-cpu", "rtfx": 28.0, "similarity": 1.0, "text": "верный текст"},
            {"family": "rnnt", "id": "ov-npu-pad22", "rtfx": 96.8, "similarity": 0.56, "text": "неверный"},
        ]
        best = bench.best_of(results, "rnnt")
        self.assertEqual(best["id"], "onnx-cpu")

    def test_ctc_is_chosen_separately(self):
        results = [
            {"family": "rnnt", "id": "onnx-cpu", "rtfx": 28.0, "similarity": 1.0, "text": "текст"},
            {"family": "ctc", "id": "ctc-gpu", "rtfx": 50.0, "similarity": 0.99, "text": "текст"},
        ]
        self.assertEqual(bench.best_of(results, "rnnt")["id"], "onnx-cpu")
        self.assertEqual(bench.best_of(results, "ctc")["id"], "ctc-gpu")

    def test_empty_text_is_not_chosen(self):
        results = [{"family": "rnnt", "id": "onnx-cpu", "rtfx": 28.0, "similarity": 1.0, "text": ""}]
        self.assertIsNone(bench.best_of(results, "rnnt"))


class MergeVariantTests(unittest.TestCase):
    def test_batch_and_pad_come_from_variant(self):
        # регрессия: в config.json попадал batch=1, хотя лучшим выбран вариант с пакетом 8
        variants = [{"id": "onnx-cpu-b8", "family": "rnnt", "device": "CPU",
                     "asr_backend": "onnx-cpu", "asr_batch": 8}]
        result = {"id": "onnx-cpu-b8", "family": "rnnt", "device": "CPU",
                  "rtfx": 26.0, "text": "текст"}
        merged = bench.merge_variant(result, variants)
        self.assertEqual(merged["asr_batch"], 8)
        self.assertEqual(merged["rtfx"], 26.0)
        self.assertEqual(merged["text"], "текст")

    def test_no_result(self):
        self.assertIsNone(bench.merge_variant(None, []))


class FamilyRefTests(unittest.TestCase):
    def test_ctc_is_compared_to_ctc_cpu(self):
        results = [
            {"id": "onnx-cpu", "family": "rnnt", "device": "CPU", "text": "раз два три"},
            {"id": "ctc-cpu", "family": "ctc", "device": "CPU", "text": "раз два"},
            {"id": "ctc-gpu", "family": "ctc", "device": "GPU", "text": "раз два"},
        ]
        refs = bench.family_refs(results)
        self.assertEqual(refs["ctc"]["id"], "ctc-cpu")
        self.assertEqual(refs["rnnt"]["id"], "onnx-cpu")

    def test_torch_is_the_rnnt_reference_when_available(self):
        results = [
            {"id": "torch-cpu", "family": "torch", "device": "CPU", "text": "эталон"},
            {"id": "onnx-cpu", "family": "rnnt", "device": "CPU", "text": "почти эталон"},
        ]
        self.assertEqual(bench.family_refs(results)["rnnt"]["id"], "torch-cpu")


class HumanChoiceTests(unittest.TestCase):
    def test_word_stats(self):
        s = bench.word_stats("раз два три четыре", "раз три четыре")
        self.assertEqual(s["missing"], 1)
        self.assertEqual(s["words"], 3)

    def test_word_stats_counts_extra(self):
        s = bench.word_stats("раз два", "раз два три")
        self.assertEqual(s["extra"], 1)
        self.assertEqual(s["missing"], 0)

    def test_threshold_from_config_and_cli(self):
        self.assertEqual(bench.threshold_from({}, None), 0.97)
        self.assertEqual(bench.threshold_from({"bench_min_similarity": 0.99}, None), 0.99)
        self.assertEqual(bench.threshold_from({"bench_min_similarity": 0.99}, 0.95), 0.95)
        self.assertEqual(bench.threshold_from({}, 1.5), 1.0)

    def test_choice_summary_priorities(self):
        results = [
            {"id": "cpu", "family": "rnnt", "device": "CPU", "rtfx": 10, "similarity": 1.0, "text": "a"},
            {"id": "gpu", "family": "rnnt", "device": "GPU", "rtfx": 50, "similarity": 0.90, "text": "a"},
            {"id": "gpu-ok", "family": "rnnt", "device": "GPU", "rtfx": 40, "similarity": 0.98, "text": "a"},
        ]
        ch = bench.choice_summary(results, "rnnt", 0.97)
        self.assertEqual(ch["accurate"]["id"], "cpu")
        self.assertEqual(ch["fast_ok"]["id"], "gpu-ok")
        self.assertEqual(ch["fastest"]["id"], "gpu")

    def test_threshold_changes_best_of(self):
        results = [
            {"id": "cpu", "family": "rnnt", "rtfx": 10, "similarity": 0.995, "text": "a"},
            {"id": "gpu", "family": "rnnt", "rtfx": 50, "similarity": 0.98, "text": "a"},
        ]
        self.assertEqual(bench.best_of(results, "rnnt", 0.97)["id"], "gpu")
        self.assertEqual(bench.best_of(results, "rnnt", 0.99)["id"], "cpu")

    def test_cli_threshold_is_saved(self):
        main = {"family": "rnnt", "device": "CPU", "asr_backend": "onnx-cpu", "asr_batch": 1}
        out = bench.saved_settings(main, min_similarity=0.99)
        self.assertEqual(out["bench_min_similarity"], 0.99)


class DefaultsTests(unittest.TestCase):
    def test_quality_threshold_has_default(self):
        value = settings.DEFAULTS["bench_min_similarity"]
        self.assertTrue(0.9 <= value <= 1.0, value)

    def test_clean_install_is_cpu_everywhere(self):
        self.assertEqual(settings.DEFAULTS["asr_device"], "CPU")
        self.assertEqual(settings.DEFAULTS["asr_second_device"], "")   # как у основного, то есть CPU
        self.assertEqual(settings.DEFAULTS["asr_second_batch"], 0)     # как у основного
        self.assertEqual(settings.DEFAULTS["asr_third_device"], "CPU")
        self.assertEqual(settings.DEFAULTS["diar_ov_device"], "CPU")
        self.assertFalse(settings.DEFAULTS["asr_allow_npu"])

    def test_second_pass_has_own_keys(self):
        for key in ("asr_second_model", "asr_second_backend", "asr_second_device", "asr_second_batch"):
            self.assertIn(key, settings.DEFAULTS, key)

    def test_bench_saves_separate_groups(self):
        for key in ("asr_device", "asr_second_device", "asr_third_device", "asr_allow_npu"):
            self.assertIn(key, bench.CONFIG_KEYS, key)


if __name__ == "__main__":
    unittest.main()
