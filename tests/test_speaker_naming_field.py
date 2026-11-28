"""Поле стенограммы `speaker_naming` (PR 1 плана «кто на записи»): приём и хранение, без поведения."""
import copy
import unittest
import uuid

import documents
import recordings
import storage

NAMING = {"Спикер 1": {"name": "Анна", "source": "form", "confirmed": False, "revision": 3},
          "Спикер 2": {"name": "Неизвестный 1", "source": "human", "confirmed": True,
                       "revision": 4, "unknown": True}}


class SpeakerNamingFieldTests(unittest.TestCase):
    def test_field_is_in_storage_and_worker_lists(self):
        self.assertIn("speaker_naming", storage.TRANSCRIPT_FIELDS)
        self.assertIn("speaker_naming", recordings._TRANSCRIPT_FIELDS)
        self.assertIn("speaker_naming", recordings._WORKER_FIELDS)

    def test_snapshot_accepts_valid_and_old_transcripts(self):
        storage._snapshot({"speakers": ["Спикер 1"], "speaker_naming": copy.deepcopy(NAMING)},
                          storage.TRANSCRIPT_FIELDS)
        storage._snapshot({"speakers": ["Спикер 1"]}, storage.TRANSCRIPT_FIELDS)

    def test_snapshot_rejects_bad_values(self):
        bad = [
            "x", {"Спикер 1": "Анна"},
            {"Спикер 1": {"name": "", "source": "form", "confirmed": False}},
            {"Спикер 1": {"name": "А", "source": "magic", "confirmed": False}},
            {"Спикер 1": {"name": "А", "source": "form", "confirmed": "no"}},
            {"Спикер 1": {"name": "А", "source": "form", "confirmed": False, "extra": 1}},
            {"Спикер 1": {"name": "А", "source": "form", "confirmed": False, "revision": -1}},
            {"Спикер 1": {"name": "А", "source": "form", "confirmed": False, "revision": True}},
        ]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                storage._snapshot({"speaker_naming": value}, storage.TRANSCRIPT_FIELDS)

    def test_worker_result_passes_field_to_transcript(self):
        data = {"file": "s.wav", "duration": "00:00:01", "speakers": ["Спикер 1"],
                "speaker_naming": copy.deepcopy(NAMING),
                "segments": [{"start": 0.0, "end": 1.0, "speaker": "Спикер 1", "text": "тест",
                              "words": [{"w": "тест", "s": 0.0, "e": 0.5}]}]}
        _, transcript, _ = recordings.snapshots_from_worker(data, str(uuid.uuid4()))
        self.assertEqual(transcript["speaker_naming"], NAMING)
        data.pop("speaker_naming")
        _, transcript, _ = recordings.snapshots_from_worker(data, str(uuid.uuid4()))
        self.assertNotIn("speaker_naming", transcript)

    def test_public_transcript_has_field_only_when_present(self):
        base = {"speakers": ["Спикер 1"], "speaker_names": {}}
        self.assertNotIn("speaker_naming", documents._public_transcript(base))
        public = documents._public_transcript({**base, "speaker_naming": copy.deepcopy(NAMING)})
        self.assertEqual(public["speaker_naming"], NAMING)
        with self.assertRaises(ValueError):
            documents._public_transcript({**base, "speaker_naming": {"a": {"name": "x"}}})


if __name__ == "__main__":
    unittest.main()
