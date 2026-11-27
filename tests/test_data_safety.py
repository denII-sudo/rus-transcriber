"""Защита настоящих данных: тесты не должны выходить за временную папку. Регрессия случая 09.11.2026 (тест унёс
настоящую папку данных; прежний перенос старой раскладки, который это сделал, удалён в У-1).
Запуск: python -m unittest discover -s tests"""
import atexit
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# как во всех модулях тестов: временная папка данных задаётся ДО первого импорта программы
_DATA_TMP = tempfile.mkdtemp(prefix="rt-data-")
os.environ["RUS_TRANSCRIBER_DATA"] = _DATA_TMP
atexit.register(shutil.rmtree, _DATA_TMP, ignore_errors=True)


class TestEnvIsolationTests(unittest.TestCase):
    def test_tests_run_in_temp_data_dir(self):
        """Если этот тест падает, какой-то тестовый модуль загрузил paths до задания временной папки данных:
        запуск тестов задел бы настоящие данные. Каждый модуль тестов обязан задавать RUS_TRANSCRIBER_DATA на
        временную папку ДО первого импорта программы."""
        sys.path.insert(0, str(ROOT))
        import paths
        tmp = Path(tempfile.gettempdir()).resolve()
        self.assertTrue(paths.EXPLICIT, "папка данных в тестах должна быть задана явно")
        self.assertTrue(tmp in paths.DATA_DIR.parents, f"{paths.DATA_DIR} не во временной папке {tmp}")


if __name__ == "__main__":
    unittest.main()
