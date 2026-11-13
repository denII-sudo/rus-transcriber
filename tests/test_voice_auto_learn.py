"""Изолированные проверки настроек и остановки пополнения голосов."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paths
import settings
import voices


def profile(seconds=(30, 30, 30)):
    return {"speakers": {"Анна": {"parts": {str(i): {"auto_sec": n, "human_sec": 0}
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
