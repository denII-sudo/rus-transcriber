"""Л-4а: новые записи — только в базу (на синтетике, во временных папках).

Запуск расшифровки при хранении в файлах или без базы отказывает до тяжёлой задачи; worker всегда пишет в
базу, старого флага --storage-backend нет; ручной запуск worker без --status берёт автора из базы.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import core
import recordings
import settings
import storage
import tmpfiles
import voices
import worker

ROOT = Path(__file__).resolve().parent.parent


def worker_result():
    return {
        "file": "synthetic.wav", "duration": "00:00:02", "speakers": ["Спикер 1"],
        "speaker_labels": ["speaker-1"], "_timing": {}, "quality": {"asr_incomplete_chunks": 0},
        "segments": [{"start": 0.2, "end": 1.2, "label": "speaker-1", "speaker": "Спикер 1", "text": "тест",
                      "words": [{"w": "тест", "s": 0.2, "e": 0.5}]}],
        "review": [], "review_done": False, "edits": [],
    }


class FakeEmbedder:
    seconds = 0

    def __init__(self, _pipe):
        pass


def recording_count(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute("SELECT count(*) FROM recordings").fetchone()[0]
    finally:
        con.close()


class StartTranscriptionRefusesWithoutStore(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / "встреча.wav"
        self.src.write_bytes(b"synthetic")
        launch = mock.patch.object(core, "launch", return_value="job-synthetic")
        self.launch = launch.start()
        self.addCleanup(launch.stop)

    def assert_refused_with_hint(self, ctx):
        self.assertIn("rt.py db-init", str(ctx.exception))
        self.assertIn("rt.py legacy-read", str(ctx.exception))       # переноса нет (У-1), только чтение
        self.assertNotIn("legacy-migration", str(ctx.exception))
        self.launch.assert_not_called()

    def test_legacy_files_choice_is_refused_before_launch(self):
        with mock.patch.object(settings, "recording_backend", return_value="legacy_files"), \
                self.assertRaises(core.CoreError) as ctx:
            core.start_transcription(str(self.src))
        self.assertIn("только в базе", str(ctx.exception))
        self.assert_refused_with_hint(ctx)

    def test_invalid_choice_is_refused_before_launch(self):
        with mock.patch.object(settings, "recording_backend", side_effect=ValueError("bad")), \
                self.assertRaises(core.CoreError) as ctx:
            core.start_transcription(str(self.src))
        self.assert_refused_with_hint(ctx)

    def test_missing_store_is_refused_before_launch(self):
        with mock.patch.object(settings, "recording_backend", return_value="sqlite"), \
                mock.patch.object(recordings, "open_store", side_effect=storage.StorageError("нет базы")), \
                self.assertRaises(core.CoreError) as ctx:
            core.start_transcription(str(self.src))
        self.assertIn("База записей", str(ctx.exception))
        self.assert_refused_with_hint(ctx)

    def test_argument_errors_come_before_the_storage_check(self):
        with mock.patch.object(settings, "recording_backend", return_value="legacy_files"), \
                self.assertRaises(core.CoreError) as ctx:
            core.start_transcription(str(self.src), whisper="лишнее")
        self.assertIn("whisper", str(ctx.exception))

    def test_store_launches_worker_with_context_and_without_backend_flag(self):
        db = self.root / "state.sqlite3"
        recordings.initialize_store(db, display_name="Инициатор").close()
        with mock.patch.object(settings, "recording_backend", return_value="sqlite"), \
                mock.patch.object(recordings, "open_store", side_effect=lambda: storage.open_store(db)):
            started = core.start_transcription(str(self.src), tag="t1")
        self.assertEqual(started["job_id"], "job-synthetic")
        script, args, _src, kind = self.launch.call_args.args
        self.assertEqual((script, kind), ("worker.py", "transcribe"))
        self.assertNotIn("--storage-backend", args)
        context = self.launch.call_args.kwargs["initial_state"]["recording_context"]
        self.assertEqual(context["actor_kind"], "model")


class WorkerWritesOnlyToStore(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tmp").mkdir()
        self.db = self.root / "state.sqlite3"
        self.src = self.root / "audio.wav"
        self.src.write_bytes(b"synthetic placeholder")
        recordings.initialize_store(self.db, display_name="Инициатор").close()

    def run_worker(self, argv, load_audio=None):
        load_audio = load_audio or mock.Mock(return_value=[0] * worker.SR)
        with mock.patch.object(settings.paths, "STATE_DB", self.db), \
                mock.patch.object(sys, "argv", ["worker.py", str(self.src), *argv, "--no-voices"]), \
                mock.patch.dict(os.environ, {"HF_TOKEN": "synthetic-token", "TEMP": str(self.root / "tmp"),
                                             "TMP": str(self.root / "tmp")}), \
                mock.patch.object(worker, "wait_slot"), \
                mock.patch.object(tmpfiles, "sweep_stale"), \
                mock.patch.object(worker, "load_config", return_value={"pipeline": "words"}), \
                mock.patch.object(worker, "resolve_vocab_details",
                                  return_value={"path": None, "vocab": {}, "group": None, "report": {}}), \
                mock.patch.object(worker, "load_audio", load_audio), \
                mock.patch.object(worker, "diarize", return_value=([(0.0, 1.0, "s1")], {}, None, None)), \
                mock.patch.object(voices, "Embedder", FakeEmbedder), \
                mock.patch.object(worker, "run_words", return_value=worker_result()):
            worker.main()
        return load_audio

    def test_old_backend_flag_is_rejected(self):
        with mock.patch.object(sys, "argv", ["worker.py", str(self.src), "--storage-backend", "legacy_files"]), \
                mock.patch.object(sys, "stderr"), self.assertRaises(SystemExit) as ctx:
            worker.main()
        self.assertEqual(ctx.exception.code, 2)

    def test_manual_run_without_status_takes_author_from_store(self):
        self.run_worker([])
        self.assertEqual(recording_count(self.db), 1)
        exports = list((self.root / "audio.rus-transcriber").glob("audio.transcript.json"))
        self.assertEqual(len(exports), 1)
        passport = json.loads(exports[0].read_text(encoding="utf-8"))["passport"]
        self.assertEqual(passport["storage_backend"], "sqlite")
        store = storage.open_store(self.db)
        try:
            recording_id = passport["recording_id"]
            self.assertEqual(store.get_current(recording_id)["review"]["context"]["actor_kind"], "model")
        finally:
            store.close()

    def test_program_run_without_captured_author_is_refused_before_heavy_work(self):
        status = self.root / "status.json"
        status.write_text("{}", encoding="utf-8")
        load_audio = mock.Mock(return_value=[0] * worker.SR)
        with self.assertRaises(SystemExit) as ctx:
            self.run_worker(["--status", str(status)], load_audio)
        self.assertEqual(ctx.exception.code, 1)
        load_audio.assert_not_called()
        result = json.loads(status.read_text(encoding="utf-8"))
        self.assertEqual(result["state"], "error")
        self.assertIn("initiator context", result["error"])
        self.assertEqual(recording_count(self.db), 0)

    def test_save_without_store_target_is_an_error_and_writes_nothing(self):
        json_path, md_path = self.root / "a.transcript.json", self.root / "a.transcript.md"
        with self.assertRaises(RuntimeError):
            worker._save_worker_result(worker_result(), self.src, mock.Mock(data={}), None, json_path, md_path)
        self.assertFalse(json_path.exists())
        self.assertFalse(md_path.exists())

    def test_worker_has_no_file_writer_for_new_records(self):
        source = (ROOT / "worker.py").read_text(encoding="utf-8")
        for name in ("save_new(", "ensure_new_result(", "storage-backend", "storage_backend"):
            self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
