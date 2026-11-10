"""Проверки переноса данных на выдуманной группе (Анна, Борис, Вера, Глеб). Запуск: python -m unittest discover -s tests"""
import atexit
import shutil
import json
import os
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="rt-data-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_OLD_ENV = os.environ.get("RUS_TRANSCRIBER_DATA")
os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_io  # noqa: E402
import paths  # noqa: E402
import voices  # noqa: E402
if _OLD_ENV is None:                  # paths уже прочитал переменную; другие тесты видят прежнюю
    os.environ.pop("RUS_TRANSCRIBER_DATA", None)
else:
    os.environ["RUS_TRANSCRIBER_DATA"] = _OLD_ENV


def make_voices(speakers):
    return {"version": voices.PROFILE_VERSION, "model": "test-model", "revision": 1,
            "speakers": {n: {"parts": {r"C:\Users\someone\Audio\запись1.mp3": {"sec": s, "vec": [0.1, 0.2]}}}
                         for n, s in speakers.items()},
            "sources": {r"C:\Users\someone\Audio\запись1.transcript.json": 123.0},
            "history": [], "warnings": [], "confusions": {}, "identity": {},
            "quarantine": {r"C:\Users\someone\Audio\запись1.mp3": {"Вера": {"json": r"C:\x\y.json"}}}}


def put_group(name, spk, vocab=None):
    gd = paths.group_dir(name)
    gd.mkdir(parents=True, exist_ok=True)
    (gd / "voices.json").write_text(json.dumps(make_voices(spk), ensure_ascii=False), encoding="utf-8")
    (gd / "vocab.json").write_text(json.dumps(vocab or {"group": name, "participants": [{"name": "Анна"}]},
                                              ensure_ascii=False), encoding="utf-8")


class T(unittest.TestCase):
    def setUp(self):
        import shutil
        for d in (paths.GROUPS, paths.DATA_DIR / "_deleted", paths.DATA_DIR / "exports"):
            shutil.rmtree(d, ignore_errors=True)
        paths.GROUPS.mkdir(parents=True, exist_ok=True)

    def test_export_has_no_absolute_paths_and_no_quarantine(self):
        put_group("demo", {"Анна": 100, "Борис": 50})
        r = data_io.export_data()
        with zipfile.ZipFile(r["zip"]) as z:
            text = b"".join(z.read(n) for n in z.namelist()).decode("utf-8")
            man = json.loads(z.read("manifest.json"))
        self.assertNotIn("C:\\\\Users", text)
        self.assertNotIn("someone", text)
        self.assertEqual(man["schema_version"], paths.DATA_SCHEMA)
        self.assertTrue(r["warnings"])                      # карантин не перенесён — сказано явно

    def test_roundtrip_replace_and_merge(self):
        put_group("demo", {"Анна": 100, "Борис": 50})
        z = data_io.export_data()["zip"]
        data_io.delete_group("demo", "demo")
        rep = data_io.import_data(z, "merge", apply=True)
        self.assertEqual(rep["groups"]["demo"]["action"], "новая группа")
        d = json.loads((paths.group_dir("demo") / "voices.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(d["speakers"]), ["Анна", "Борис"])
        # слияние: у Анны в данных меньше речи, чем в архиве; Глеб новый только здесь
        put_group("demo", {"Анна": 10, "Глеб": 40})
        rep = data_io.import_data(z, "merge", apply=True)["groups"]["demo"]
        self.assertEqual(rep["added_speakers"], ["Борис"])
        self.assertEqual(rep["conflicts"][0]["kept"], "imported")
        d = json.loads((paths.group_dir("demo") / "voices.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(d["speakers"]), ["Анна", "Борис", "Глеб"])
        self.assertEqual(d["revision"], 2)

    def test_dry_run_changes_nothing(self):
        put_group("demo", {"Анна": 100})
        z = data_io.export_data()["zip"]
        put_group("demo", {"Глеб": 5})
        before = (paths.group_dir("demo") / "voices.json").read_bytes()
        data_io.import_data(z, "replace", apply=False)
        self.assertEqual(before, (paths.group_dir("demo") / "voices.json").read_bytes())

    def test_replace_makes_backup(self):
        put_group("demo", {"Анна": 100})
        z = data_io.export_data()["zip"]
        put_group("demo", {"Глеб": 5})
        rep = data_io.import_data(z, "replace", apply=True)
        self.assertTrue(Path(rep["backup"], "groups", "demo", "voices.json").exists())

    def test_bad_archives_rejected(self):
        put_group("demo", {"Анна": 100})
        z = data_io.export_data()["zip"]
        evil = Path(_TMP) / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zz:
            zz.writestr("../evil.txt", "x")
        with self.assertRaises(RuntimeError):
            data_io.import_data(str(evil))
        bad = Path(_TMP) / "tampered.zip"
        with zipfile.ZipFile(z) as src, zipfile.ZipFile(bad, "w") as dst:
            for n in src.namelist():
                b = src.read(n)
                dst.writestr(n, b + b" " if n.endswith("voices.json") else b)
        with self.assertRaises(RuntimeError):
            data_io.import_data(str(bad))
        future = Path(_TMP) / "future.zip"
        with zipfile.ZipFile(future, "w") as zz:
            zz.writestr("manifest.json", json.dumps({"app": "rus-transcriber", "schema_version": 99, "files": {}}))
        with self.assertRaises(RuntimeError):
            data_io.import_data(str(future))

    def test_delete_group_needs_confirmation(self):
        put_group("demo", {"Анна": 100})
        with self.assertRaises(RuntimeError):
            data_io.delete_group("demo", "нет")
        self.assertTrue(paths.group_dir("demo").exists())
        r = data_io.delete_group("demo", "demo")
        self.assertFalse(paths.group_dir("demo").exists())
        self.assertTrue(Path(r["folder"], "voices.json").exists())

    def test_cleanup_and_audit(self):
        old = paths.JOBS / "old.json"
        old.write_text('{"state":"done"}', encoding="utf-8")
        t = time.time() - 40 * 86400
        os.utime(old, (t, t))
        new = paths.JOBS / "new.json"
        new.write_text('{"state":"done"}', encoding="utf-8")
        self.assertEqual(data_io.cleanup_jobs(30, apply=False)["files"], 1)
        self.assertTrue(old.exists())
        data_io.cleanup_jobs(30, apply=True)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        put_group("demo", {"Анна": 1})
        self.assertTrue(data_io.audit()["files_with_absolute_paths"])


if __name__ == "__main__":
    unittest.main()
