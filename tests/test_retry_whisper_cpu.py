"""Повтор ошибочных фрагментов третьей модели на CPU (worker._retry_whisper_on_cpu).

Исполнитель синтетический, модели не загружаются. Проверяется, что при сбое посреди повтора
ошибка дописывается только ещё не заменённым фрагментам, а уже успешно повторённые сохраняют
свой текст, состояние и сведения о попытках.
"""
import atexit
import copy
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

if "paths" not in sys.modules:   # папка данных — временная ДО первого импорта программы
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-retry-cpu-data-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import worker  # noqa: E402


class FakeStatus:
    def set(self, **kwargs):
        pass


class FakeCpu:
    """Возвращает по тексту на фрагмент; у фрагмента `bad_pos` диагностика без «attempts»."""

    def __init__(self, bad_pos=None, count_mismatch=False):
        self.bad_pos = bad_pos
        self.count_mismatch = count_mismatch

    def transcribe_texts(self, pieces, diagnostics=None, progress=None):
        texts = []
        for k, _ in enumerate(pieces):
            if k == self.bad_pos:
                diagnostics.append({"state": "ok"})
            else:
                diagnostics.append({"state": "ok", "attempts": [
                    {"result": "ok", "backend": "x", "batch": [k]}]})
            texts.append(f"повтор-{k}")
        if self.count_mismatch:
            texts.pop()
        return texts


def failed_diag():
    return {"state": "error", "attempts": [
        {"result": "error", "backend": "whisper-ov:GPU", "error_type": "RuntimeError"}]}


class RetryWhisperOnCpuTests(unittest.TestCase):
    def run_retry(self, cpu, failed_indices, size=4):
        texts = [f"исх-{i}" for i in range(size)]
        diag = [failed_diag() for _ in range(size)]
        before = copy.deepcopy(diag)
        pieces = [bytes([i]) for i in range(size)]
        worker._retry_whisper_on_cpu(failed_indices, texts, diag, pieces, "m", {}, FakeStatus(),
                                     lambda cfg: (cpu, None), lambda *a: None)
        return texts, diag, before

    def assert_untouched(self, texts, diag, before, i):
        self.assertEqual(texts[i], f"исх-{i}")
        self.assertEqual(diag[i], before[i])

    def assert_replaced(self, texts, diag, before, i, pos):
        self.assertEqual(texts[i], f"повтор-{pos}")
        self.assertEqual(diag[i]["state"], "ok")
        self.assertEqual(diag[i]["attempts"], before[i]["attempts"] + [
            {"result": "ok", "backend": "onnx-cpu:CPU", "batch": [[1, 2, 3][pos]]}])

    def assert_failed_once(self, diag, before, i, error_type):
        self.assertEqual(diag[i]["state"], "error")
        self.assertEqual(diag[i]["attempts"], before[i]["attempts"] + [
            {"result": "error", "backend": "onnx-cpu:CPU",
             "error_type": error_type, "fallback": True}])

    def test_all_replaced(self):
        texts, diag, before = self.run_retry(FakeCpu(), [1, 2, 3])
        self.assert_untouched(texts, diag, before, 0)
        for pos, i in enumerate([1, 2, 3]):
            self.assert_replaced(texts, diag, before, i, pos)

    def test_failure_on_second_position_keeps_first_replacement(self):
        texts, diag, before = self.run_retry(FakeCpu(bad_pos=1), [1, 2, 3])
        self.assert_untouched(texts, diag, before, 0)
        self.assert_replaced(texts, diag, before, 1, 0)
        for i in (2, 3):
            self.assertEqual(texts[i], f"исх-{i}")
            self.assert_failed_once(diag, before, i, "KeyError")

    def test_failure_on_third_position_keeps_two_replacements(self):
        texts, diag, before = self.run_retry(FakeCpu(bad_pos=2), [1, 2, 3])
        self.assert_untouched(texts, diag, before, 0)
        self.assert_replaced(texts, diag, before, 1, 0)
        self.assert_replaced(texts, diag, before, 2, 1)
        self.assertEqual(texts[3], "исх-3")
        self.assert_failed_once(diag, before, 3, "KeyError")

    def test_failure_before_loop_marks_every_fragment(self):
        texts, diag, before = self.run_retry(FakeCpu(count_mismatch=True), [1, 2, 3])
        self.assert_untouched(texts, diag, before, 0)
        for i in (1, 2, 3):
            self.assertEqual(texts[i], f"исх-{i}")
            self.assert_failed_once(diag, before, i, "ValueError")


if __name__ == "__main__":
    unittest.main()
