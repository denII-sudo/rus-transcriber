"""Проверка worker без моделей и настоящих пользовательских данных."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import core
import paths
import voices
import worker


class Status:
    def __init__(self):
        self.data = {}

    def set(self, **values):
        self.data.update(values)


class VoiceWorkerUpdate(unittest.TestCase):
    def test_confirmed_update_and_source_conflict(self):
        for change_source in (False, True):
            with self.subTest(change_source=change_source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                groups, jobs = root / "groups", root / "jobs"
                group = groups / "demo-group"
                group.mkdir(parents=True)
                audio = root / "demo.wav"
                audio.write_bytes(b"synthetic audio")
                source = root / "demo.transcript.json"
                data = {"file": str(audio), "group": "demo-group", "names_confirmed": True,
                        "segments": [{"speaker": "Анна", "words": [{"s": 0, "e": 3}]}]}
                source.write_text(json.dumps(data), encoding="utf-8")
                key = str(audio.resolve())
                profile = {"version": voices.PROFILE_VERSION, "model": "synthetic", "sources": {},
                           "speakers": {"Анна": {"parts": {key: {"vec": [0, 1], "sec": 5}}}}}
                target = group / "voices.json"
                target.write_text(json.dumps(profile), encoding="utf-8")
                before = target.read_bytes()
                args = types.SimpleNamespace(learn_only=str(source), voice_source_hash=core._voice_hash(source),
                    voice_profile_hash=core._voice_hash(target), voice_audio_hash=core._voice_hash(audio),
                    voice_group="demo-group")

                class FakeEmbedder:
                    def __init__(self, pipe):
                        pass

                    def spans(self, audio_data, spans, cap):
                        if change_source:
                            source.write_text(json.dumps({**data, "source_revision": 2}), encoding="utf-8")
                        return [np.array([1.0, 0.0])] * len(spans)

                fake_audio = types.ModuleType("pyannote.audio")
                fake_audio.Pipeline = types.SimpleNamespace(from_pretrained=lambda *a, **k: object())
                status = Status()
                with patch.object(paths, "GROUPS", groups), patch.object(paths, "JOBS", jobs), \
                     patch.dict("sys.modules", {"pyannote.audio": fake_audio}), \
                     patch.object(worker, "resolve_vocab", return_value=(None, {}, "demo-group")), \
                     patch.object(worker, "load_audio", return_value=np.zeros(48000)), \
                     patch.object(voices, "Embedder", FakeEmbedder):
                    if change_source:
                        with self.assertRaises(RuntimeError):
                            worker.learn_only(args, status, {}, audio)
                    else:
                        worker.learn_only(args, status, {}, audio)
                if change_source:
                    self.assertEqual(target.read_bytes(), before)
                    self.assertFalse((group / "voice-backups").exists())
                    self.assertEqual(status.data["state"], "error")
                else:
                    self.assertEqual(status.data["state"], "done")
                    self.assertEqual(Path(status.data["backup"]).read_bytes(), before)
                    after = json.loads(target.read_text(encoding="utf-8"))
                    self.assertEqual(after["speakers"]["Анна"]["parts"][key]["selection_version"],
                                     voices.SELECTION_VERSION)
