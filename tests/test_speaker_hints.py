"""Подсказки «кто на записи» (PR 1): политика, запуск, границы диаризации, галочка пополнения профиля."""
import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import core
import paths
import product_policy
import voices
import worker

ROOT = Path(__file__).resolve().parent.parent


def hints(**kw):
    base = {"count_mode": "approx", "count": 4,
            "participants": [{"name": "Анна", "forms": ["Аня"]}, {"name": "Борис"}],
            "stray_voices": False, "group": "demo", "enroll_profile": True}
    return {**base, **kw}


class PolicyHintsTests(unittest.TestCase):
    def test_default_has_no_hints_and_old_policies_work(self):
        # без подсказок ключа нет: запросы совместимы с кодом до PR в обе стороны
        self.assertNotIn("speaker_hints", product_policy.resolve())
        self.assertNotIn("speaker_hints", product_policy.resolve({"goal": "x"}))
        self.assertNotIn("speaker_hints", product_policy.resolve({"speaker_hints": None}))
        self.assertEqual(set(product_policy.resolve()), set(product_policy.DEFAULTS))

    def test_valid_hints_are_normalized(self):
        got = product_policy.resolve({"speaker_hints": hints()})["speaker_hints"]
        self.assertEqual(got["participants"], [{"name": "Анна", "forms": ["Аня"]}, {"name": "Борис", "forms": []}])
        self.assertTrue(got["enroll_profile"])
        self.assertEqual(product_policy.resolve({"speaker_hints": {}})["speaker_hints"]["count_mode"], "unknown")

    def test_bad_hints_are_rejected(self):
        bad = [hints(count_mode="many"), hints(count=0), hints(count=31), hints(count=True),
               hints(stray_voices="y"), hints(enroll_profile=1), hints(participants="Анна"),
               hints(participants=[{"name": ""}]), hints(participants=[{"name": "А, Б"}]),
               hints(participants=[{"name": "Анна"}, {"name": "анна"}]),
               hints(participants=[{"name": "А", "forms": "x"}]), {"extra": 1}, "x"]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                product_policy.resolve({"speaker_hints": value})

    def test_bounds(self):
        def b(**kw):
            return product_policy.diarization_bounds(product_policy.normalize_speaker_hints(hints(**kw)))
        self.assertEqual(b(count_mode="exact", count=4), (4, 0, 0))
        self.assertEqual(b(count_mode="approx", count=4), (0, 3, 5))
        self.assertEqual(b(count_mode="approx", count=1), (0, 1, 2))
        self.assertEqual(b(count_mode="unknown"), (0, 0, 0))
        self.assertEqual(product_policy.diarization_bounds(None), (0, 0, 0))

    def test_hints_never_reach_model_task(self):
        # задание модели строится только из слов и цели (policy["goal"]); подсказок в этих модулях нет
        for name in ("model_tasks.py", "workflow.py"):
            self.assertNotIn("speaker_hints", (ROOT / name).read_text(encoding="utf-8"), name)


class TextHintsTests(unittest.TestCase):
    def test_nothing_given_means_old_behaviour(self):
        self.assertIsNone(product_policy.hints_from_text())
        # «не пополнять» без прочих подсказок не игнорируется: голоса только сопоставляются
        got = product_policy.hints_from_text(enroll_profile=False)
        self.assertEqual((got["count_mode"], got["enroll_profile"]), ("unknown", False))

    def test_text_forms(self):
        got = product_policy.hints_from_text(4, True, "Анна=Аня/Анне, Борис; Вера", "demo", True, False)
        self.assertEqual((got["count_mode"], got["count"], got["stray_voices"], got["enroll_profile"]),
                         ("approx", 4, True, False))
        self.assertEqual(got["participants"], [{"name": "Анна", "forms": ["Аня", "Анне"]},
                                               {"name": "Борис", "forms": []}, {"name": "Вера", "forms": []}])
        self.assertEqual(product_policy.hints_from_text(2)["count_mode"], "exact")
        self.assertEqual(product_policy.hints_from_text(participants="Анна")["count_mode"], "unknown")


class StartWorkflowTests(unittest.TestCase):
    def start(self, hints, groups=()):
        if hints is not None:
            hints = product_policy.normalize_speaker_hints(hints)
        policy = {"speaker_hints": hints} if hints is not None else {}
        with mock.patch.object(core.settings, "recording_backend", return_value="sqlite"),              mock.patch("recordings.open_store"), mock.patch("workflow.create", return_value="r"),              mock.patch("workflow.claim_start", return_value="j"),              mock.patch("workflow.get", return_value={"source": "a.wav", "context": {}, "policy": policy}),              mock.patch.object(paths, "list_groups", return_value=list(groups)),              mock.patch.object(core, "start_transcription") as start:
            result = core.start_workflow("a.wav", policy)
        return result, start

    def test_empty_form_sends_hints_with_voices_and_enroll(self):
        empty = {"count_mode": "unknown", "count": 0, "participants": [], "stray_voices": False,
                 "group": "", "enroll_profile": True}
        result, start = self.start(empty)
        self.assertTrue(start.call_args.kwargs["voices"])
        self.assertTrue(start.call_args.kwargs["enroll"])
        self.assertNotIn("warnings", result)

    def test_no_hints_means_no_voices(self):
        _, start = self.start(None)
        self.assertFalse(start.call_args.kwargs["voices"])

    def test_group_with_space_is_refused_before_anything_starts(self):
        with self.assertRaises(core.CoreError):
            self.start({"group": "моя группа"})

    def test_unknown_group_gives_warning_known_does_not(self):
        result, _ = self.start({"group": "demo"})
        self.assertIn("demo", result["warnings"][0])
        result, _ = self.start({"group": "demo"}, groups=["demo"])
        self.assertNotIn("warnings", result)


class LaunchArgsTests(unittest.TestCase):
    def test_no_hints_is_old_launch(self):
        self.assertEqual(core._hint_launch_args(None), {})

    def test_hints_to_arguments(self):
        h = product_policy.normalize_speaker_hints(hints(enroll_profile=False, stray_voices=True))
        self.assertEqual(core._hint_launch_args(h), {
            "num_speakers": 0, "min_speakers": 3, "max_speakers": 5, "names": "Анна, Борис", "group": "demo",
            "stray_voices": True, "enroll": False, "screen_hints": True})

    def launch_args(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "a.wav"
            audio.touch()
            with mock.patch.object(core.settings, "recording_backend", return_value="sqlite"), \
                 mock.patch("recordings.open_store"), \
                 mock.patch.object(core, "launch", return_value="j") as launch:
                core.start_transcription(audio, workflow_context={"x": 1}, **kw)
                return launch.call_args.args[1]

    def test_start_transcription_builds_worker_args(self):
        args = self.launch_args(voices=True, min_speakers=3, max_speakers=5, stray_voices=True,
                                enroll=False, screen_hints=True)
        self.assertEqual(args[args.index("--min-speakers") + 1], "3")
        self.assertEqual(args[args.index("--max-speakers") + 1], "5")
        for flag in ("--stray-voices", "--no-enroll", "--screen-hints"):
            self.assertIn(flag, args)
        self.assertNotIn("--no-voices", args)

    def test_old_call_has_no_new_flags(self):
        args = self.launch_args(voices=False)
        for flag in ("--min-speakers", "--max-speakers", "--stray-voices", "--no-enroll", "--screen-hints"):
            self.assertNotIn(flag, args)


class WorkerBoundsTests(unittest.TestCase):
    def ns(self, **kw):
        base = dict(speakers=0, min_speakers=0, max_speakers=0, stray_voices=False)
        return argparse.Namespace(**{**base, **kw})

    def test_old_bounds_unchanged(self):
        self.assertEqual(worker.diarization_bounds(self.ns(), ["А", "Б"], ["А", "Б"]), (3, 2))
        self.assertEqual(worker.diarization_bounds(self.ns(), [], []), (0, 0))
        self.assertEqual(worker.diarization_bounds(self.ns(speakers=2), ["А"], ["А"]), (0, 0))

    def test_stray_and_explicit_bounds(self):
        self.assertEqual(worker.diarization_bounds(self.ns(stray_voices=True), ["А", "Б"], ["А", "Б"]), (5, 2))
        self.assertEqual(worker.diarization_bounds(self.ns(min_speakers=3, max_speakers=5), ["А"], ["А"]), (5, 3))
        self.assertEqual(worker.diarization_bounds(self.ns(min_speakers=3, max_speakers=5, speakers=4), [], []), (0, 0))

    def test_diarize_passes_bounds_to_pipeline(self):
        calls = {}

        class Pipe:
            segmentation_batch_size = 1

            def to(self, device):
                pass

            def __call__(self, data, hook=None, **kwargs):
                calls.update(kwargs)
                raise StopIteration

        fake_pyannote = types.ModuleType("pyannote.audio")
        fake_pyannote.Pipeline = mock.MagicMock()
        fake_pyannote.Pipeline.from_pretrained.return_value = Pipe()
        modules = {"torch": mock.MagicMock(), "pyannote": types.ModuleType("pyannote"),
                   "pyannote.audio": fake_pyannote, "diar_fast": types.ModuleType("diar_fast")}
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaises(StopIteration):
                worker.diarize(mock.MagicMock(), 0, "t", {}, mock.MagicMock(), max_speakers=5, min_speakers=3)
        self.assertEqual(calls, {"max_speakers": 5, "min_speakers": 3})


class EnrollCheckboxTests(unittest.TestCase):
    def run_prepare(self, enroll):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = root / "groups"
            (groups / "g").mkdir(parents=True)
            quality = {}
            with mock.patch.object(paths, "GROUPS", groups), mock.patch.object(paths, "JOBS", root / "jobs"), \
                 mock.patch.object(voices, "auto_learn_group",
                                   return_value=(["x.wav"], {"enabled": True})) as auto, \
                 mock.patch.object(voices, "save_profile") as save:
                worker.prepare_group_voice_profile("g", root / "a.wav", object(), [], quality, enroll=enroll)
            return auto, save, quality

    def test_enroll_on_calls_auto_learn(self):
        auto, _, quality = self.run_prepare(True)
        auto.assert_called_once()
        self.assertEqual(quality["voice_auto_learn"], {"enabled": True})

    def test_enroll_off_never_learns_or_saves(self):
        auto, save, quality = self.run_prepare(False)
        auto.assert_not_called()
        save.assert_not_called()
        self.assertEqual(quality["voice_auto_learn"], {"enabled": False, "reason": "enroll_off"})


if __name__ == "__main__":
    unittest.main()
