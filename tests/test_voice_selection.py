"""Синтетические проверки отбора фрагментов голоса; модели и профили не используются."""
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

import voices


def recording(word=None, **segment_fields):
    segment = {"speaker": "Анна", "start": 0.0, "end": 3.0,
               "words": [word] if word else []}
    segment.update(segment_fields)
    return {"segments": [segment]}


class VoiceOverlapSelection(unittest.TestCase):
    def test_overlap_formats_are_excluded_for_human_and_automatic_words(self):
        for human in (False, True):
            for marker in ({"overlap": True}, {"flags": ["overlap"]}):
                with self.subTest(human=human, marker=marker):
                    word = {"s": 0.0, "e": 3.0, **marker}
                    if human:
                        word["by"] = "human"
                    self.assertEqual(voices._spans(recording(word), True), [])

    def test_segment_overlap_excludes_words_and_legacy_segments(self):
        word = {"s": 0.0, "e": 3.0, "by": "human"}
        self.assertEqual(voices._spans(recording(word, overlap=True), True), [])
        self.assertEqual(voices._spans(recording(overlap=True), True), [])

    def test_clean_word_is_still_selected(self):
        word = {"s": 0.0, "e": 3.0, "provenance": {"speaker": {"by": "human"}}}
        self.assertEqual(voices._spans(recording(word), True), [("Анна", 0.0, 3.0, True)])

    def test_text_and_legacy_by_do_not_confirm_speaker(self):
        for extra in ({"by": "human"}, {"by": "human", "provenance": {"text": {"by": "human"}}}):
            word = {"s": 0.0, "e": 3.0, **extra}
            self.assertEqual(voices._spans(recording(word), False), [])
            self.assertEqual(voices._spans(recording(word), True), [("Анна", 0.0, 3.0, False)])

    def test_long_pause_and_reverse_order_are_not_joined(self):
        words = [{"s": 0.0, "e": 1.0}, {"s": 9.0, "e": 10.0}, {"s": 4.0, "e": 5.0}]
        report = {}
        self.assertEqual(voices._spans(recording(words=words), True, report), [])
        self.assertEqual(report["gap_or_order"], 2)
        self.assertEqual(report["too_short"], 3)

    def test_half_second_pause_is_allowed_but_larger_pause_splits(self):
        words = [{"s": 0.0, "e": 1.0}, {"s": 1.5, "e": 2.5}, {"s": 3.01, "e": 5.1}]
        self.assertEqual(voices._spans(recording(words=words), True),
                         [("Анна", 0.0, 2.5, False), ("Анна", 3.01, 5.1, False)])

    def test_report_contains_reasons_not_voice_vectors(self):
        report = {}
        voices._spans(recording({"s": 0.0, "e": 3.0, "overlap": True}), True, report)
        self.assertEqual(report, {"overlap": 1})

    def test_legacy_without_word_times_is_skipped(self):
        report = {}
        self.assertEqual(voices._spans(recording(), True, report), [])
        self.assertEqual(report, {"legacy_without_word_times": 1})

    def test_windows_end_on_word_boundaries(self):
        words = [{"s": float(i), "e": i + 0.9} for i in range(12)]
        spans = voices._spans(recording(words=words), True)
        self.assertEqual(spans, [("Анна", 0.0, 7.9, False), ("Анна", 8.0, 11.9, False)])

    def test_learning_receives_no_window_covering_long_silence(self):
        class FakeEmbedder:
            def spans(self, audio, spans, cap):
                self.received = spans
                return [np.array([1.0, 0.0])] * len(spans)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_file = root / "demo.wav"
            audio_file.touch()
            words = [{"s": 0.0, "e": 1.0}, {"s": 9.0, "e": 10.0}]
            for word in words:
                word["provenance"] = {"speaker": {"by": "human"}}
            data = recording(words=words)
            data.update({"file": str(audio_file), "names_confirmed": True})
            (root / "demo.transcript.json").write_text(json.dumps(data), encoding="utf-8")
            profile = {"speakers": {}, "sources": {}}
            embedder = FakeEmbedder()
            report = {}
            voices.learn(profile, root, embedder, lambda _: np.zeros(160000), selection_report=report)
            self.assertEqual(embedder.received, [(0.0, 1.0), (9.0, 10.0)])
            self.assertEqual(report["gap_or_order"], 1)
            self.assertFalse((root / "voices.json").exists())
