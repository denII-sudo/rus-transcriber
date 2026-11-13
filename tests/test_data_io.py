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
from unittest.mock import patch
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="rt-data-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_OLD_ENV = os.environ.get("RUS_TRANSCRIBER_DATA")
os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_io  # noqa: E402
import paths  # noqa: E402
import voices  # noqa: E402
import settings  # noqa: E402
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

    def test_voice_settings_roundtrip_merge_replace_and_preview(self):
        put_group("demo", {"Анна": 100})
        settings.save_group_voice_settings("demo", {"auto_learn": False, "reason": "sufficient"})
        archive = data_io.export_data()["zip"]
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read("manifest.json"))
            member = "groups/demo/voice-settings.json"
            self.assertEqual(json.loads(z.read(member)), {"auto_learn": False, "reason": "sufficient"})
            self.assertIn(member, manifest["files"])
        settings.save_group_voice_settings("demo", {"auto_learn": True, "reason": "manual"})
        data_io.import_data(archive, "replace", apply=False)
        self.assertTrue(settings.group_voice_settings("demo")["auto_learn"])
        data_io.import_data(archive, "merge", apply=True)
        self.assertTrue(settings.group_voice_settings("demo")["auto_learn"])
        with patch.object(data_io.time, "strftime", return_value="voice-settings-test"):
            result = data_io.import_data(archive, "replace", apply=True)
        self.assertFalse(settings.group_voice_settings("demo")["auto_learn"])
        backup = Path(result["backup"], "groups", "demo", "voice-settings.json")
        self.assertTrue(json.loads(backup.read_text(encoding="utf-8"))["auto_learn"])
        data_io.delete_group("demo", "demo")
        data_io.import_data(archive, apply=True)
        self.assertFalse(settings.group_voice_settings("demo")["auto_learn"])

    def test_old_archive_new_group_uses_global_default(self):
        put_group("demo", {"Анна": 100})
        source = data_io.export_data()["zip"]
        old = Path(_TMP) / "old-archive.zip"
        member = "groups/demo/voice-settings.json"
        with zipfile.ZipFile(source) as src, zipfile.ZipFile(old, "w") as dst:
            manifest = json.loads(src.read("manifest.json"))
            del manifest["files"][member]
            manifest["groups"][0]["files"].remove(member)
            dst.writestr("manifest.json", json.dumps(manifest))
            for name in src.namelist():
                if name not in ("manifest.json", member):
                    dst.writestr(name, src.read(name))
        data_io.delete_group("demo", "demo")
        with patch.object(settings, "load_config", return_value={"voice_auto_learn_default": False}):
            data_io.import_data(old, apply=True)
        self.assertFalse(settings.group_voice_settings("demo")["auto_learn"])
        settings.save_group_voice_settings("demo", {"auto_learn": True, "reason": "manual"})
        with patch.object(data_io.time, "strftime", return_value="old-archive-test"):
            data_io.import_data(old, mode="replace", apply=True)
        self.assertTrue(settings.group_voice_settings("demo")["auto_learn"])

    def test_modified_voice_settings_fails_checksum(self):
        put_group("demo", {"Анна": 100})
        source = data_io.export_data()["zip"]
        target = Path(_TMP) / "bad-settings.zip"
        with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as dst:
            for name in src.namelist():
                blob = src.read(name)
                dst.writestr(name, blob + b" " if name.endswith("voice-settings.json") else blob)
        with self.assertRaises(RuntimeError):
            data_io.import_data(target, apply=True)

    def test_invalid_voice_settings_rejected_even_with_valid_checksum(self):
        put_group("demo", {"Анна": 100})
        source = data_io.export_data()["zip"]
        target = Path(_TMP) / "invalid-settings.zip"
        member = "groups/demo/voice-settings.json"
        blob = b'{"auto_learn":"false"}'
        with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as dst:
            manifest = json.loads(src.read("manifest.json"))
            manifest["files"][member] = {"sha256": data_io._sha(blob), "size": len(blob)}
            dst.writestr("manifest.json", json.dumps(manifest))
            for name in src.namelist():
                if name != "manifest.json":
                    dst.writestr(name, blob if name == member else src.read(name))
        before = (paths.group_dir("demo") / "voices.json").read_bytes()
        with self.assertRaises(RuntimeError):
            data_io.import_data(target, mode="replace", apply=True)
        self.assertEqual(before, (paths.group_dir("demo") / "voices.json").read_bytes())

    def test_invalid_archived_voice_profiles_fail_before_any_changes(self):
        put_group("demo", {"Анна": 100})
        put_group("other", {"Борис": 20})
        source = data_io.export_data()["zip"]
        gd = paths.group_dir("demo")
        before = {p.relative_to(paths.GROUPS): p.read_bytes() for p in paths.GROUPS.rglob("*") if p.is_file()}
        valid = make_voices({"Вера": 20})
        old = {**valid, "version": voices.PROFILE_VERSION - 1}
        missing_vector = make_voices({"Вера": 20})
        next(iter(missing_vector["speakers"]["Вера"]["parts"].values())).pop("vec")
        for index, blob in enumerate((json.dumps(old).encode(), json.dumps(missing_vector).encode(), b"{invalid")):
            target = Path(_TMP) / f"invalid-profile-{index}.zip"
            member = "groups/other/voices.json"
            with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as dst:
                manifest = json.loads(src.read("manifest.json"))
                manifest["files"][member] = {"sha256": data_io._sha(blob), "size": len(blob)}
                dst.writestr("manifest.json", json.dumps(manifest))
                for name in src.namelist():
                    if name != "manifest.json":
                        dst.writestr(name, blob if name == member else src.read(name))
            for mode in ("merge", "replace"):
                for apply in (False, True):
                    with self.subTest(index=index, mode=mode, apply=apply):
                        with self.assertRaisesRegex(RuntimeError, "голосовой профиль|Голосовой профиль"):
                            data_io.import_data(target, mode=mode, apply=apply)
                        after = {p.relative_to(paths.GROUPS): p.read_bytes() for p in paths.GROUPS.rglob("*") if p.is_file()}
                        self.assertEqual(before, after)
        self.assertTrue(gd.exists())

    def test_import_rejects_invalid_profile_for_new_group(self):
        put_group("demo", {"Анна": 100})
        source = data_io.export_data()["zip"]
        target = Path(_TMP) / "invalid-new.zip"
        member = "groups/demo/voices.json"
        blob = json.dumps({"version": voices.PROFILE_VERSION - 1, "speakers": {}, "sources": {}}).encode()
        with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as dst:
            manifest = json.loads(src.read("manifest.json"))
            manifest["files"][member] = {"sha256": data_io._sha(blob), "size": len(blob)}
            dst.writestr("manifest.json", json.dumps(manifest))
            for name in src.namelist():
                if name != "manifest.json":
                    dst.writestr(name, blob if name == member else src.read(name))
        data_io.delete_group("demo", "demo")
        for mode in ("merge", "replace"):
            with self.assertRaises(RuntimeError):
                data_io.import_data(target, mode=mode, apply=True)
            self.assertFalse(paths.group_dir("demo").exists())

    def test_merge_rejects_incompatible_combined_vector_dimensions(self):
        put_group("demo", {"Анна": 100})
        source = data_io.export_data()["zip"]
        local = make_voices({"Борис": 20})
        next(iter(local["speakers"]["Борис"]["parts"].values()))["vec"] = [1.0]
        path = paths.group_dir("demo") / "voices.json"
        path.write_text(json.dumps(local), encoding="utf-8")
        before = path.read_bytes()
        result = data_io.import_data(source, mode="merge", apply=True)
        self.assertIn("пропущено", result["groups"]["demo"]["voices"])
        self.assertEqual(before, path.read_bytes())

    def test_unlisted_voice_settings_rejected(self):
        put_group("demo", {"Анна": 100})
        source = data_io.export_data()["zip"]
        target = Path(_TMP) / "unlisted-settings.zip"
        member = "groups/demo/voice-settings.json"
        with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as dst:
            manifest = json.loads(src.read("manifest.json"))
            del manifest["files"][member]
            dst.writestr("manifest.json", json.dumps(manifest))
            for name in src.namelist():
                if name != "manifest.json":
                    dst.writestr(name, src.read(name))
        with self.assertRaises(RuntimeError):
            data_io.import_data(target, apply=True)

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
