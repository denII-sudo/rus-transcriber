"""Защита настоящих данных: перенос из «прежних мест» не должен срабатывать, если папка данных задана явно,
и тесты не должны выходить за временную папку. Регрессия случая 09.11.2026 (тест унёс настоящую папку данных).
Запуск: python -m unittest discover -s tests"""
import atexit
import shutil
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# как во всех модулях тестов: временная папка данных задаётся ДО первого импорта программы
_DATA_TMP = tempfile.mkdtemp(prefix="rt-data-")
os.environ["RUS_TRANSCRIBER_DATA"] = _DATA_TMP
atexit.register(shutil.rmtree, _DATA_TMP, ignore_errors=True)

SCRIPT = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
import paths, migrate
if sys.argv[2] == "pop-env":
    os.environ.pop("RUS_TRANSCRIBER_DATA", None)     # как делали тесты: переменную снимают после импорта
old = [str(p) for p in paths.old_default_dirs()]
res = migrate.run()
print(json.dumps({"data_dir": str(paths.DATA_DIR), "explicit": paths.EXPLICIT, "old_dirs": old,
                  "status": (res or {}).get("status")}, ensure_ascii=False))
"""


def run_in_fake_home(home, data_env=None, pop_env=False):
    env = {k: v for k, v in os.environ.items() if k not in ("RUS_TRANSCRIBER_DATA",)}
    env.update(HOME=str(home), USERPROFILE=str(home), LOCALAPPDATA=str(home / "AppData" / "Local"),
               APPDATA=str(home / "AppData" / "Roaming"), PYTHONDONTWRITEBYTECODE="1")
    if data_env:
        env["RUS_TRANSCRIBER_DATA"] = str(data_env)
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(ROOT), "pop-env" if pop_env else "keep"], env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    if r.returncode != 0:
        raise AssertionError(r.stderr)
    return json.loads(r.stdout.strip().splitlines()[-1])


def put_old(home):
    """Файл «настоящих данных» в месте, которое программа считает прежним по умолчанию."""
    f = home / "rus-transcriber-data" / "groups" / "g" / "voices.json"
    f.parent.mkdir(parents=True)
    f.write_text('{"speakers": {}}', encoding="utf-8")
    return f


class DataSafetyTests(unittest.TestCase):
    def test_explicit_data_dir_never_relocates_even_if_env_is_removed_later(self):
        with tempfile.TemporaryDirectory() as t:
            home = Path(t) / "home"
            home.mkdir()
            real = put_old(home)
            out = run_in_fake_home(home, data_env=Path(t) / "data", pop_env=True)
            self.assertTrue(out["explicit"])
            self.assertEqual(out["old_dirs"], [])                       # «прежних мест» нет
            self.assertTrue(real.is_file())                             # данные на месте
            self.assertFalse(list(home.glob("rus-transcriber-data.moved-*")))   # ничего не переименовано

    def test_default_data_dir_still_relocates_old_place(self):
        """Контроль: перенос из прежнего места при обычной установке по-прежнему работает."""
        with tempfile.TemporaryDirectory() as t:
            home = Path(t) / "home"
            home.mkdir()
            real = put_old(home)
            out = run_in_fake_home(home)
            self.assertFalse(out["explicit"])
            self.assertTrue(out["old_dirs"], out)
            moved = Path(out["data_dir"]) / "groups" / "g" / "voices.json"
            self.assertTrue(moved.is_file(), out)                       # скопировано в папку данных
            self.assertFalse(real.exists())                             # прежняя папка переименована
            self.assertTrue(list(home.glob("rus-transcriber-data.moved-*")))


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
