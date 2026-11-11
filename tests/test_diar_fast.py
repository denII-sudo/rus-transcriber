"""Имя устройства OpenVINO для разделения спикеров (diar_fast).

Регрессия 11.11.2026: в config.json стояло `diar_ov_device: "cpu"` строчными буквами,
OpenVINO такое имя не знает («Device with "cpu" name is not registered»), и ускорение
молча отключалось — отпечатки голосов считал обычный путь PyTorch.

Запуск: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import diar_fast  # noqa: E402


class DeviceNameTests(unittest.TestCase):
    def test_lowercase_is_uppercased(self):
        self.assertEqual(diar_fast.device_name("cpu"), "CPU")
        self.assertEqual(diar_fast.device_name("gpu"), "GPU")
        self.assertEqual(diar_fast.device_name("npu"), "NPU")

    def test_default_and_spaces(self):
        self.assertEqual(diar_fast.device_name(None), "GPU")
        self.assertEqual(diar_fast.device_name(""), "GPU")
        self.assertEqual(diar_fast.device_name(" gpu "), "GPU")

    def test_compound_names(self):
        self.assertEqual(diar_fast.device_name("hetero:npu,cpu"), "HETERO:NPU,CPU")
        self.assertEqual(diar_fast.device_name("GPU.0"), "GPU.0")


if __name__ == "__main__":
    unittest.main()
