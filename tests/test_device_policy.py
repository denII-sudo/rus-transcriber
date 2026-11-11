"""Правила выбора устройств и защита от NPU без FP32.

Регрессия 11.11.2026: NPU Meteor Lake не поддерживает FP32, а его компилятор считает
GigaAM неверно. Такой NPU нельзя ни предлагать, ни молча подменять процессором.

Запуск: python -m unittest discover -s tests
"""
import atexit
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-dev-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backends  # noqa: E402


class NormalizeTests(unittest.TestCase):
    def test_names(self):
        self.assertEqual(backends.normalize_device("cpu"), "CPU")
        self.assertEqual(backends.normalize_device(" gpu "), "GPU")
        self.assertEqual(backends.normalize_device("hetero:npu,cpu"), "HETERO:NPU,CPU")
        self.assertEqual(backends.normalize_device(None), "CPU")

    def test_backend_by_device(self):
        self.assertEqual(backends.backend_for_device("CPU"), "onnx-cpu")
        self.assertEqual(backends.backend_for_device("gpu"), "openvino")
        self.assertEqual(backends.backend_for_device("NPU"), "openvino")


class NpuCapabilityTests(unittest.TestCase):
    def test_meteor_lake_has_no_fp32(self):
        self.assertFalse(backends._npu_fp32(["FP16", "INT8", "EXPORT_IMPORT"]))

    def test_newer_npu_has_fp32(self):
        self.assertTrue(backends._npu_fp32(["FP32", "FP16", "INT8"]))
        self.assertTrue(backends._npu_fp32(["fp32"]))

    def test_npu_info_shape(self):
        info = backends.npu_info()
        for key in ("available", "fp32", "name", "capabilities", "reason"):
            self.assertIn(key, info)
        self.assertIsInstance(info["fp32"], bool)
        # на этой машине NPU есть, но FP32 нет — проверяем, что причина названа
        if info["available"] and not info["fp32"]:
            self.assertIn("FP32", info["reason"])


class MakeAsrOrderTests(unittest.TestCase):
    def test_cpu_default_never_tries_directml(self):
        calls = []

        class FakeASR:
            def __init__(self, model, device="CPU", provider="openvino", **kw):
                calls.append((provider, device))
                if provider != "cpu":
                    raise RuntimeError("нет такого устройства")

        with mock.patch.object(backends, "OnnxASR", FakeASR):
            asr, errs = backends.make_asr({"asr_backend": "auto", "asr_device": "CPU"})
        self.assertIsInstance(asr, FakeASR)
        self.assertEqual(calls, [("openvino", "CPU"), ("cpu", "CPU")])
        self.assertEqual(len(errs), 1)


class NpuGuardTests(unittest.TestCase):
    def test_npu_requires_explicit_permission(self):
        # до загрузки модели: запрет должен сработать сразу и без скачивания
        with mock.patch.object(backends, "available_providers",
                               return_value=["OpenVINOExecutionProvider"]):
            with self.assertRaises(RuntimeError) as ctx:
                backends.OnnxASR("v3_e2e_rnnt", device="NPU", provider="openvino")
        self.assertIn("NPU", str(ctx.exception))

    def test_actual_providers_from_encoder(self):
        class Sess:
            def get_providers(self):
                return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

        class Adapter:
            _encoder = Sess()

        class Model:
            asr = Adapter()

        self.assertEqual(backends.actual_providers(Model()),
                         ["OpenVINOExecutionProvider", "CPUExecutionProvider"])

    def test_actual_providers_from_ctc_model(self):
        class Sess:
            def get_providers(self):
                return ["CPUExecutionProvider"]

        class Adapter:
            _model = Sess()

        class Model:
            asr = Adapter()

        self.assertEqual(backends.actual_providers(Model()), ["CPUExecutionProvider"])
        self.assertEqual(backends.actual_providers(object()), [])


if __name__ == "__main__":
    unittest.main()
