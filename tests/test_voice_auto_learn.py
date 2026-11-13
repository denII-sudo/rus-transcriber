"""Изолированные проверки настроек и остановки пополнения голосов."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import paths
import settings
import voices


def profile(seconds=(30, 30, 30)):
    return {"speakers": {"Анна": {"parts": {str(i): {"auto_sec": n, "human_sec": 0, "selection_version": voices.SELECTION_VERSION}
                                                for i, n in enumerate(seconds)}}}}


class VoiceAutoLearn(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [patch.object(paths, "GROUPS", root / "groups"),
                        patch.object(paths, "CONFIG_FILE", root / "config.json"),
                        patch.object(paths, "DATA_DIR", root), patch.object(paths, "JOBS", root / "jobs")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_new_group_copies_global_value_only_once(self):
        settings.update(voice_auto_learn_default=False)
        self.assertFalse(settings.group_voice_settings("demo-group", initialize=True)["auto_learn"])
        settings.update(voice_auto_learn_default=True)
        self.assertFalse(settings.group_voice_settings("demo-group")["auto_learn"])
        self.assertTrue(settings.group_voice_settings("other", initialize=True)["auto_learn"])

    def test_changing_default_preserves_existing_group_value(self):
        import core
        group = paths.group_dir("demo-group")
        group.mkdir(parents=True)
        (group / "vocab.json").write_text("{}", encoding="utf-8")
        core.set_voice_auto_learn(False)
        self.assertTrue(settings.group_voice_settings("demo-group")["auto_learn"])
        self.assertFalse(settings.group_voice_settings("new-group", initialize=True)["auto_learn"])

    def test_explicit_group_setting_overrides_default(self):
        import core
        group = paths.group_dir("demo-group")
        group.mkdir(parents=True)
        (group / "vocab.json").write_text("{}", encoding="utf-8")
        settings.update(voice_auto_learn_default=False)
        core.set_voice_auto_learn(True, "demo-group")
        self.assertTrue(settings.group_voice_settings("demo-group")["auto_learn"])

    def test_requires_both_duration_and_three_sources_for_each_person(self):
        self.assertTrue(voices.profile_sufficiency(profile(), ["Анна"])["sufficient"])
        self.assertFalse(voices.profile_sufficiency(profile((45, 45)), ["Анна"])["sufficient"])
        self.assertFalse(voices.profile_sufficiency(profile((30, 30, 29)), ["Анна"])["sufficient"])
        self.assertFalse(voices.profile_sufficiency(profile(), ["Анна", "Борис"])["sufficient"])
        self.assertFalse(voices.profile_sufficiency({"speakers": {}}, [])["sufficient"])
        self.assertFalse(voices.profile_sufficiency(profile((86, 2, 2)), ["Анна"])["sufficient"])
        self.assertTrue(voices.profile_sufficiency(profile((50, 20, 20)), ["Анна"])["sufficient"])
        self.assertEqual(voices.profile_sufficiency(profile((100, 20, 20)), ["Анна"])["speakers"]["Анна"]["seconds"], 100)

    def test_sufficiency_stops_without_learning_and_stays_stopped(self):
        calls = []
        data = profile()
        result, status = voices.auto_learn_group(data, "demo-group", ["Анна"],
                                                  lambda: calls.append("learn"), lambda: calls.append("save"))
        self.assertEqual(result, [])
        self.assertEqual(status["reason"], "sufficient")
        self.assertFalse(settings.group_voice_settings("demo-group")["auto_learn"])
        voices.auto_learn_group(profile((1, 1, 1)), "demo-group", ["Анна"],
                                lambda: calls.append("learn"), lambda: calls.append("save"))
        self.assertEqual(calls, [])

    def test_learning_can_reach_threshold_then_freezes(self):
        data = profile((30, 30, 29))
        calls = []
        def learn():
            data["speakers"]["Анна"]["parts"]["2"]["auto_sec"] = 30
            return ["demo"]
        result, status = voices.auto_learn_group(data, "demo-group", ["Анна"], learn, lambda: calls.append("save"))
        self.assertEqual(result, ["demo"])
        self.assertEqual(calls, ["save"])
        self.assertFalse(status["enabled"])

    def test_corrupt_settings_fails_closed_without_learning(self):
        p = paths.group_dir("demo-group") / "voice-settings.json"
        p.parent.mkdir(parents=True)
        p.write_text("{", encoding="utf-8")
        _, status = voices.auto_learn_group(profile(), "demo-group", [],
                                            lambda: self.fail("learn"), lambda: self.fail("save"))
        self.assertEqual(status["reason"], "settings_error")

    def test_old_weighted_seconds_not_counted(self):
        data = {"speakers": {"Анна": {"parts": {str(i): {"sec": 100} for i in range(3)}}}}
        self.assertFalse(voices.profile_sufficiency(data, ["Анна"])["sufficient"])

    def test_legacy_and_verified_parts_are_counted_separately(self):
        data = profile()
        parts = data["speakers"]["Анна"]["parts"]
        for part in parts.values():
            part.pop("selection_version")
        old = voices.profile_sufficiency(data, ["Анна"])["speakers"]["Анна"]
        self.assertFalse(old["sufficient"])
        self.assertEqual((old["verified_seconds"], old["verified_recordings"]), (0, 0))
        self.assertEqual((old["legacy_seconds"], old["unverified_records"]), (90, 3))
        parts["0"]["selection_version"] = voices.SELECTION_VERSION
        mixed = voices.profile_sufficiency(data, ["Анна"])["speakers"]["Анна"]
        self.assertFalse(mixed["sufficient"])
        self.assertEqual((mixed["verified_seconds"], mixed["legacy_seconds"]), (30, 60))
        for part in parts.values():
            part["selection_version"] = voices.SELECTION_VERSION
        self.assertTrue(voices.profile_sufficiency(data, ["Анна"])["sufficient"])

    def test_unmeasured_legacy_part_is_visible_without_guessing_seconds(self):
        data = {"speakers": {"Анна": {"parts": {"old": {"sec": 100}}}}}
        report = voices.profile_sufficiency(data, ["Анна"])["speakers"]["Анна"]
        self.assertEqual(report["legacy_unmeasured_records"], 1)
        self.assertEqual(report["legacy_seconds"], 0)
        self.assertFalse(report["sufficient"])

    def test_legacy_alone_does_not_freeze_or_modify_profile(self):
        import copy
        data = profile()
        for part in data["speakers"]["Анна"]["parts"].values():
            part.pop("selection_version")
        before = copy.deepcopy(data)
        _, status = voices.auto_learn_group(data, "demo-group", ["Анна"], lambda: [], lambda: self.fail("save"))
        self.assertTrue(status["enabled"])
        self.assertEqual(data, before)

    def test_voice_profiles_explains_legacy_and_verified_amounts(self):
        import core
        group = paths.group_dir("demo-group")
        group.mkdir(parents=True)
        data = profile()
        parts = data["speakers"]["Анна"]["parts"]
        for item in parts.values():
            item["sec"] = item["auto_sec"]
        parts["0"].pop("selection_version")
        (group / "voices.json").write_text(json.dumps(data), encoding="utf-8")
        report = core.voice_profiles()["profiles"][0]["sufficiency"]["speakers"]["Анна"]
        self.assertEqual((report["verified_seconds"], report["verified_recordings"]), (60, 2))
        self.assertEqual((report["legacy_seconds"], report["legacy_recordings"]), (30, 1))
        self.assertFalse(report["sufficient"])

    def test_weight_diagnostics_shows_weights_without_vectors_or_absolute_paths(self):
        import core
        group = paths.group_dir("demo-group")
        group.mkdir(parents=True)
        data = {"speakers": {"Анна": {"parts": {
            "folder-a/demo-a.wav": {"sec": 60, "auto_sec": 60, "human_sec": 0, "vec": [0.1, 0.2]},
            "folder-b/demo-b.wav": {"sec": 30, "auto_sec": 10, "human_sec": 7, "vec": [0.3, 0.4]}}}}}
        path = group / "voices.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        before = path.read_bytes()
        report = core.voice_profiles()["profiles"][0]["weight_diagnostics"]["Анна"]
        self.assertTrue(report["available"])
        self.assertEqual(report["total_weight"], 90)
        self.assertEqual([p["weight_share_percent"] for p in report["parts"]], [66.67, 33.33])
        self.assertEqual([p["recording"] for p in report["parts"]], ["demo-a.wav", "demo-b.wav"])
        self.assertNotIn("vec", json.dumps(report))
        self.assertNotIn("folder-a", json.dumps(report))
        self.assertEqual(path.read_bytes(), before)

    def test_weight_diagnostics_fails_closed_for_invalid_or_zero_weights(self):
        for weight in (None, -1, float("nan"), float("inf"), True, 0):
            with self.subTest(weight=weight):
                data = {"speakers": {"Анна": {"parts": {"demo.wav": {"sec": weight}}}}}
                report = voices.voice_weight_diagnostics(data)["Анна"]
                self.assertFalse(report["available"])
                self.assertNotIn("parts", report)

    def test_weight_diagnostics_disambiguates_duplicate_basenames(self):
        data = {"speakers": {"Анна": {"parts": {"a/demo.wav": {"sec": 1}, "b/demo.wav": {"sec": 1}}}}}
        parts = voices.voice_weight_diagnostics(data)["Анна"]["parts"]
        self.assertEqual([p["recording_index"] for p in parts], [1, 2])
        self.assertEqual([p["weight_share_percent"] for p in parts], [50, 50])
