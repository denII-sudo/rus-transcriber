"""Профили неподдерживаемой версии и повреждённые не меняются ручными операциями и импортом."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core
import data_io
import paths
import voices


def recording(root, group="demo-group"):
    audio = root / "demo.wav"
    audio.touch()
    source = root / "demo.transcript.json"
    source.write_text(json.dumps({"file": str(audio), "group": group, "names_confirmed": True,
                                  "segments": [{"speaker": "Анна", "words": [{"s": 0, "e": 3}]}]}),
                      encoding="utf-8")
    return audio, source


def profile(version=None, sources=None, speakers=None):
    return {"version": voices.PROFILE_VERSION if version is None else version,
            "model": "wespeaker-resnet34 (pyannote community-1)",
            "speakers": speakers if speakers is not None else {},
            "sources": {} if sources is None else sources}


class VoiceProfileCompat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.groups, self.jobs = self.root / "groups", self.root / "jobs"
        self.patches = [patch.object(paths, "GROUPS", self.groups), patch.object(paths, "JOBS", self.jobs),
                        patch.object(paths, "DATA_DIR", self.root),
                        patch.object(paths, "CONFIG_FILE", self.root / "config.json")]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def group_dir(self):
        folder = self.groups / "demo-group"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def write_profile(self, content):
        target = self.group_dir() / "voices.json"
        target.write_text(content, encoding="utf-8")
        return target

    def legacy_content(self, audio):
        return json.dumps(profile(version=voices.PROFILE_VERSION - 1, speakers={
            "Анна": {"parts": {str(audio.resolve()): {"vec": [1], "sec": 30}}}}), ensure_ascii=False)

    def test_preview_refuses_incompatible_and_corrupt_profile_without_writing(self):
        audio, source = recording(self.root)
        corrupt = [self.legacy_content(audio), '{bad json', json.dumps(profile(sources=[]))]
        for content in corrupt:
            with self.subTest(content=content):
                target = self.write_profile(content)
                before = target.read_bytes()
                with patch.object(core, "job_json", return_value=(str(source), None)):
                    with self.assertRaises(core.CoreError) as ctx:
                        core.preview_voice_learning("demo")
                self.assertIn("несовместим или повреждён", str(ctx.exception))
                self.assertEqual(target.read_bytes(), before)
                self.assertFalse((self.group_dir() / "voice-backups").exists())

    def test_revoke_refuses_incompatible_profile_without_backup(self):
        audio, source = recording(self.root)
        target = self.write_profile(self.legacy_content(audio))
        before = target.read_bytes()
        snapshot = {"group": "demo-group", "json": str(source), "source_hash": core._voice_hash(source),
                    "profile_hash": core._voice_hash(target), "audio_hash": "absent",
                    "report": {"existing_contributions": ["Анна"]}}
        with self.assertRaises(core.CoreError) as ctx:
            core.revoke_voice_source(snapshot)
        self.assertIn("несовместим или повреждён", str(ctx.exception))
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse((self.group_dir() / "voice-backups").exists())

    def test_voice_profiles_reports_error_instead_of_sufficiency(self):
        audio, _ = recording(self.root)
        target = self.write_profile(self.legacy_content(audio))
        before = target.read_bytes()
        report = core.voice_profiles()["profiles"][0]
        self.assertEqual(report["group"], "demo-group")
        self.assertIn("неподдерживаемая версия", report["profile_error"])
        self.assertNotIn("sufficiency", report)
        self.assertNotIn("auto_learn", report)
        self.assertEqual(target.read_bytes(), before)

    def test_import_merge_leaves_incompatible_profile_untouched(self):
        audio, _ = recording(self.root)
        valid = profile(speakers={"Анна": {"parts": {"demo.wav": {"vec": [1, 0], "sec": 40}}}})
        self.write_profile(json.dumps(valid, ensure_ascii=False))
        archive = data_io.export_data(dest_dir=self.root / "out", group="demo-group")["zip"]
        legacy = self.legacy_content(audio)
        target = self.write_profile(legacy)
        before = target.read_bytes()
        report = data_io.import_data(archive, mode="merge", apply=True)
        self.assertEqual(target.read_bytes(), before)
        self.assertIn("пропущено", report["groups"]["demo-group"]["voices"])
        self.assertIsNone(report["backup"])
        self.assertTrue(any("не изменён" in warning for warning in report["warnings"]))

    def test_import_replace_still_replaces_with_backup(self):
        audio, _ = recording(self.root)
        valid = profile(speakers={"Анна": {"parts": {"demo.wav": {"vec": [1, 0], "sec": 40}}}})
        self.write_profile(json.dumps(valid, ensure_ascii=False))
        archive = data_io.export_data(dest_dir=self.root / "out", group="demo-group")["zip"]
        target = self.write_profile(self.legacy_content(audio))
        legacy_bytes = target.read_bytes()
        report = data_io.import_data(archive, mode="replace", apply=True)
        self.assertEqual(report["groups"]["demo-group"]["voices"], "заменены")
        self.assertIsNotNone(report["backup"])
        self.assertEqual((Path(report["backup"]) / "groups" / "demo-group" / "voices.json").read_bytes(),
                         legacy_bytes)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["version"], voices.PROFILE_VERSION)

    def test_missing_profile_is_still_previewed_and_revoke_reports_absent(self):
        audio, source = recording(self.root)
        with patch.object(core, "job_json", return_value=(str(source), None)):
            preview = core.preview_voice_learning("demo")
        self.assertEqual(preview["existing_contributions"], [])
        self.assertFalse((self.group_dir() / "voices.json").exists())
        snapshot = {"group": "demo-group", "json": str(source), "source_hash": core._voice_hash(source),
                    "profile_hash": "absent", "report": {"existing_contributions": []}}
        with self.assertRaises(core.CoreError) as ctx:
            core.revoke_voice_source(snapshot)
        self.assertIn("профиль отсутствует", str(ctx.exception))
        self.assertFalse((self.group_dir() / "voice-backups").exists())


if __name__ == "__main__":
    unittest.main()
