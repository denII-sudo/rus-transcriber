"""Синтетические проверки отбора фрагментов голоса; модели и профили не используются."""
import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np

import voices

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import db_record as DB  # noqa: E402  (У-2а: источник голоса — запись в базе, не файл без базы)


def recording(word=None, **segment_fields):
    segment = {"speaker": "Анна", "start": 0.0, "end": 3.0,
               "words": [word] if word else []}
    segment.update(segment_fields)
    return {"segments": [segment]}


class VoiceOverlapSelection(unittest.TestCase):
    def test_unknown_group_manual_preview_uses_vocab_read_only(self):
        import core
        import paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups, jobs = root / "groups", root / "jobs"
            group = groups / "demo-group"
            group.mkdir(parents=True)
            audio = root / "demo.wav"
            audio.write_bytes(b"synthetic audio")
            stale_vocab = root / "removed-vocab.json"
            data = {"file": str(audio), "names_confirmed": True,
                    "quality": {"vocab": str(stale_vocab)},
                    "segments": [
                        {"speaker": "Анна", "words": [{"s": 0.0, "e": 3.0, "w": "привет"}]},
                        {"speaker": "Чужое имя", "words": [{"s": 4.0, "e": 7.0, "w": "слово"}]}]}
            source, _, _ = DB.new_record(self, data, root, "demo")
            source_before = source.read_bytes()
            group_vocab = group / "vocab.json"
            vocab_data = {"group": "demo-group", "participants": [{"name": "Анна", "gender": "ж"}]}
            group_vocab.write_text(json.dumps(vocab_data), encoding="utf-8")
            vocab_before = group_vocab.read_bytes()
            with patch.object(core, "job_json", return_value=(str(source), None)), \
                 patch.object(paths, "GROUPS", groups), patch.object(paths, "JOBS", jobs):
                result = core.preview_voice_learning("demo", group_override="demo-group")
            self.assertEqual(result["group"], "demo-group")
            self.assertEqual(result["group_selection"]["method"], "manual_override")
            self.assertEqual(result["candidates"], {"Анна": {"windows": 1, "seconds": 3.0, "human_windows": 0}})
            self.assertEqual(result["names_outside_vocab"], ["Чужое имя"])
            self.assertEqual(result["vocab"]["path"], str(group_vocab.resolve()))
            stale = next(item for item in result["vocab_resolution"]["candidates"]
                         if item["source"] == "quality.vocab")
            self.assertEqual(stale["status"], "missing")
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(group_vocab.read_bytes(), vocab_before)
            self.assertFalse((group / "voices.json").exists())

    def test_manual_group_preview_requires_matching_vocab_and_no_conflicting_binding(self):
        import core
        import paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = root / "groups"
            (groups / "alpha").mkdir(parents=True)
            (groups / "beta").mkdir()
            audio = root / "demo.wav"
            audio.touch()
            source = root / "demo.transcript.json"
            data = {"file": str(audio), "segments": []}
            source.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(core, "job_json", return_value=(str(source), None)), \
                 patch.object(paths, "GROUPS", groups):
                with self.assertRaises(core.CoreError):
                    core.preview_voice_learning("demo", group_override="alpha")
                (groups / "alpha" / "vocab.json").write_text(
                    json.dumps({"group": "alpha", "participants": []}), encoding="utf-8")
                source.write_text(json.dumps({**data, "group": "beta"}), encoding="utf-8")
                with self.assertRaises(core.CoreError):
                    core.preview_voice_learning("demo", group_override="alpha")
                source.write_text(json.dumps(data), encoding="utf-8")
                foreign = root / "foreign.json"
                foreign.write_text(json.dumps({"group": "beta"}), encoding="utf-8")
                source.write_text(json.dumps({**data, "quality": {"vocab": str(foreign)}}), encoding="utf-8")
                with self.assertRaises(core.CoreError):
                    core.preview_voice_learning("demo", group_override="alpha")
                source.write_text(json.dumps(data), encoding="utf-8")
                sidecar = root / "rus-transcriber.vocab.json"
                sidecar.write_text(json.dumps({"group": "beta"}), encoding="utf-8")
                with self.assertRaises(core.CoreError):
                    core.preview_voice_learning("demo", group_override="alpha")

    def test_manual_group_snapshot_binds_vocab_hash_and_changed_vocab_blocks_launch(self):
        import core
        import paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups, jobs = root / "groups", root / "jobs"
            group = groups / "demo-group"
            group.mkdir(parents=True)
            audio = root / "demo.wav"
            audio.write_bytes(b"synthetic audio")
            data = {"file": str(audio), "segments": []}
            source, _, _ = DB.new_record(self, data, root, "demo")
            group_vocab = group / "vocab.json"
            group_vocab.write_text(json.dumps({"group": "demo-group", "participants": []}), encoding="utf-8")
            with patch.object(core, "job_json", return_value=(str(source), None)), \
                 patch.object(paths, "GROUPS", groups), patch.object(paths, "JOBS", jobs), \
                 patch.object(core, "launch") as launch:
                snapshot = core.prepare_voice_action("demo", group_override="demo-group")
                self.assertTrue(snapshot["manual_group_override"])
                self.assertEqual(snapshot["vocab_hash"], core._voice_hash(group_vocab))
                self.assertIn("выбрана вручную", "\n".join(core.voice_action_lines(snapshot)))
                core.start_voice_update(snapshot)
                args = launch.call_args.args[1]
                self.assertIn("--voice-group-override", args)
                self.assertIn("demo-group", args)
                self.assertIn("--voice-vocab-hash", args)
                self.assertIn(snapshot["vocab_hash"], args)
                launch.reset_mock()
                group_vocab.write_text(json.dumps({"group": "demo-group", "participants": [{"name": "changed"}]}),
                                       encoding="utf-8")
                with self.assertRaises(core.CoreError):
                    core.start_voice_update(snapshot)
            launch.assert_not_called()

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

    def test_unmarked_foreign_overlap_rejected_but_touching_allowed(self):
        first = {"speaker": "Анна", "words": [{"s": 0.0, "e": 3.0}]}
        second = {"speaker": "Борис", "words": [{"s": 2.9, "e": 5.0}]}
        report = {}
        self.assertEqual(voices._spans({"segments": [first, second]}, True, report), [])
        self.assertEqual(report["foreign_overlap"], 2)
        second["words"][0]["s"] = 3.0
        self.assertEqual(len(voices._spans({"segments": [first, second]}, True)), 2)

    def test_overlap_inside_pause_between_words_rejects_window(self):
        first = {"speaker": "Анна", "words": [{"s": 0, "e": 1}, {"s": 1.4, "e": 3}]}
        second = {"speaker": "Борис", "words": [{"s": 1.1, "e": 1.3}]}
        self.assertEqual(voices._spans({"segments": [first, second]}, True), [])

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
            source_key = str((root / "demo.transcript.json").resolve())
            self.assertEqual(profile["source_selection_versions"][source_key], voices.SELECTION_VERSION)
            self.assertFalse((root / "voices.json").exists())

    def test_existing_contribution_is_preserved_until_preview_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "demo.wav"
            audio.touch()
            data = recording({"s": 0.0, "e": 3.0})
            data.update({"file": str(audio), "names_confirmed": True})
            transcript = root / "demo.transcript.json"
            transcript.write_text(json.dumps(data), encoding="utf-8")
            key = str(audio.resolve())
            part = {"vec": np.array([1.0, 0.0]), "sec": 5.0}
            profile = {"speakers": {"Анна": {"parts": {key: part}}}, "sources": {}}
            report = {}
            result = voices.learn(profile, root, None, lambda _: self.fail("audio loaded"),
                                  only=[transcript], selection_report=report)
            self.assertEqual(result, [])
            self.assertIs(profile["speakers"]["Анна"]["parts"][key], part)
            self.assertEqual(report["existing_requires_preview"], 1)

    def test_no_candidates_is_not_reported_as_learned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "demo.wav"
            audio.touch()
            data = recording(overlap=True)
            data.update({"file": str(audio), "names_confirmed": True})
            source = root / "demo.transcript.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            profile = {"speakers": {}, "sources": {}}
            report = {}
            self.assertEqual(voices.learn(profile, root, None, lambda _: self.fail("audio loaded"),
                                          selection_report=report), [])
            self.assertNotIn("revision", profile)
            self.assertEqual(report["no_usable_vectors"], 1)

    def test_unchanged_legacy_source_requires_review_not_automatic_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "demo.wav"
            audio.touch()
            data = recording({"s": 0.0, "e": 3.0})
            data.update({"file": str(audio), "names_confirmed": True})
            source = root / "demo.transcript.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            key, audio_key = str(source.resolve()), str(audio.resolve())
            part = {"vec": np.array([1.0, 0.0]), "sec": 30.0}
            profile = {"speakers": {"Анна": {"parts": {audio_key: part}}}, "sources": {key: source.stat().st_mtime}}
            report = {}
            self.assertEqual(voices.learn(profile, root, None, lambda _: self.fail("audio loaded"), selection_report=report), [])
            self.assertEqual(report, {"source_needs_review": 1})
            self.assertIs(profile["speakers"]["Анна"]["parts"][audio_key], part)
            profile["source_selection_versions"] = {key: voices.SELECTION_VERSION}
            report = {}
            self.assertEqual(voices.learn(profile, root, None, lambda _: self.fail("audio loaded"), selection_report=report), [])
            self.assertEqual(report, {"source_needs_review": 1})
            part["selection_version"] = voices.SELECTION_VERSION
            report = {}
            self.assertEqual(voices.learn(profile, root, None, lambda _: self.fail("audio loaded"), selection_report=report), [])
            self.assertEqual(report, {"already_processed": 1})
            profile["quarantine"] = {audio_key: {"Анна": {"note": "synthetic"}}}
            report = {}
            voices.learn(profile, root, None, lambda _: self.fail("audio loaded"), selection_report=report)
            self.assertEqual(report, {"source_needs_review": 1})

    def test_confirmed_replacement_keeps_names_without_candidates(self):
        class FakeEmbedder:
            def spans(self, audio, spans, cap):
                return [np.array([1.0, 0.0])] * len(spans)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "demo.wav"
            audio.touch()
            data = recording({"s": 0.0, "e": 3.0})
            data.update({"file": str(audio), "names_confirmed": True})
            source = root / "demo.transcript.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            key = str(audio.resolve())
            old = {"vec": np.array([0.0, 1.0]), "sec": 5.0}
            profile = {"speakers": {"Анна": {"parts": {key: old}}, "Борис": {"parts": {key: old}}}, "sources": {}}
            learned = voices.learn(profile, root, FakeEmbedder(), lambda _: np.zeros(48000), only=[source], allow_replace=True)
            self.assertEqual(learned, [source.name])
            self.assertEqual(profile["speakers"]["Анна"]["parts"][key]["selection_version"], voices.SELECTION_VERSION)
            self.assertIs(profile["speakers"]["Борис"]["parts"][key], old)

    def test_replacement_preserves_old_part_if_identity_is_different(self):
        class FakeEmbedder:
            def spans(self, audio, spans, cap):
                return [np.array([1.0, 0.0])] * len(spans)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "demo.wav"
            audio.touch()
            data = recording({"s": 0.0, "e": 3.0})
            data.update({"file": str(audio), "names_confirmed": True})
            source = root / "demo.transcript.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            key = str(audio.resolve())
            old = {"vec": np.array([0.0, 1.0]), "sec": 5.0}
            profile = {"speakers": {"Анна": {"parts": {key: old}}}, "sources": {},
                       "identity": {key: {"Анна": "different"}}, "quarantine": {key: {"Борис": {"note": "keep"}}}}
            report = {}
            voices.learn(profile, root, FakeEmbedder(), lambda _: np.zeros(48000), only=[source],
                         allow_replace=True, selection_report=report)
            self.assertIs(profile["speakers"]["Анна"]["parts"][key], old)
            self.assertIn("Борис", profile["quarantine"][key])
            self.assertEqual(report["identity_different"], 1)

    def test_failure_after_update_rolls_back_profile_and_success_counters(self):
        class FakeEmbedder:
            def spans(self, audio, spans, cap):
                return [np.array([1.0, 0.0])] * len(spans)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "demo.wav"
            audio.touch()
            data = recording({"s": 0.0, "e": 3.0})
            data.update({"file": str(audio), "names_confirmed": True})
            source = root / "demo.transcript.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            profile = {"speakers": {}, "sources": {}}
            report = {}
            with patch.object(voices, "voice_vectors", side_effect=[{}, {}, RuntimeError("synthetic failure")]):
                result = voices.learn(profile, root, FakeEmbedder(), lambda _: np.zeros(48000), selection_report=report)
            self.assertEqual(result, [])
            self.assertEqual(profile["speakers"], {})
            self.assertNotIn("revision", profile)
            self.assertEqual(report, {"processing_error": 1})
            self.assertEqual(voices.explain_selection_report(report)[0]["description"],
                             "Источник пропущен из-за ошибки; изменения этого источника отменены")

    def test_preview_is_read_only_and_does_not_return_vectors(self):
        import core
        import paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = root / "groups"
            group = groups / "demo-group"
            group.mkdir(parents=True)
            audio = root / "demo.wav"
            audio.touch()
            data = recording({"s": 0.0, "e": 3.0})
            data.update({"file": str(audio), "group": "demo-group", "names_confirmed": True})
            source, _, _ = DB.new_record(self, data, root, "demo")
            profile = {"version": voices.PROFILE_VERSION,
                       "speakers": {"Анна": {"parts": {str(audio.resolve()): {"vec": [0.1, 0.2], "sec": 1}}}},
                       "sources": {}}
            target = group / "voices.json"
            target.write_text(json.dumps(profile), encoding="utf-8")
            before = target.read_bytes()
            with patch.object(core, "job_json", return_value=(str(source), None)), patch.object(paths, "GROUPS", groups):
                result = core.preview_voice_learning("demo")
            self.assertEqual(result["existing_contributions"], ["Анна"])
            self.assertEqual(result["candidates"]["Анна"]["windows"], 1)
            self.assertNotIn("vec", json.dumps(result))
            self.assertEqual(target.read_bytes(), before)

    def test_revoke_after_cancel_preserves_other_sources_and_makes_backup(self):
        import core
        import paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups, jobs = root / "groups", root / "jobs"
            group = groups / "demo-group"
            group.mkdir(parents=True)
            audio = root / "demo.wav"
            audio.touch()
            data = recording({"s": 0.0, "e": 1.0})
            data.update({"file": str(audio), "group": "demo-group", "names_confirmed": False,
                         "edits": [{"by": "human", "reverted": True}]})
            source, _, _ = DB.new_record(self, data, root, "demo")
            key = str(audio.resolve())
            profile = {"version": voices.PROFILE_VERSION, "speakers": {"Анна": {"parts": {key: {"vec": [1], "sec": 1},
                                                        "other": {"vec": [2], "sec": 2}}}},
                       "sources": {str(source.resolve()): 1}, "confusions": {key: {"a|b": 1}}, "revision": 1}
            target = group / "voices.json"
            target.write_text(json.dumps(profile), encoding="utf-8")
            before = target.read_bytes()
            with patch.object(core, "job_json", return_value=(str(source), None)), patch.object(paths, "GROUPS", groups), patch.object(paths, "JOBS", jobs):
                snapshot = core.prepare_voice_action("demo")
                result = core.revoke_voice_source(snapshot)
            self.assertEqual(Path(result["backup"]).read_bytes(), before)
            after = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(set(after["speakers"]["Анна"]["parts"]), {"other"})
            self.assertNotIn(key, after["confusions"])
            self.assertEqual(after["revision"], 2)

    def test_changed_transcript_blocks_revoke_without_backup(self):
        import core
        import paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups, jobs = root / "groups", root / "jobs"
            group = groups / "demo-group"
            group.mkdir(parents=True)
            audio = root / "demo.wav"
            audio.touch()
            data = recording({"s": 0.0, "e": 3.0})
            data.update({"file": str(audio), "group": "demo-group", "names_confirmed": True})
            source, _, _ = DB.new_record(self, data, root, "demo")
            target = group / "voices.json"
            target.write_text(json.dumps({"version": voices.PROFILE_VERSION,
                "speakers": {"Анна": {"parts": {str(audio.resolve()): {"vec": [1], "sec": 1}}}}, "sources": {}}), encoding="utf-8")
            before = target.read_bytes()
            with patch.object(core, "job_json", return_value=(str(source), None)), patch.object(paths, "GROUPS", groups), patch.object(paths, "JOBS", jobs):
                snapshot = core.prepare_voice_action("demo")
                DB.put(source, {**data, "duration": "00:00:09"})   # стенограмма изменилась (поля «revision» база не берёт)
                with self.assertRaises(core.CoreError):
                    core.revoke_voice_source(snapshot)
            self.assertEqual(target.read_bytes(), before)
            self.assertFalse((group / "voice-backups").exists())

    def test_changed_audio_blocks_confirmed_update(self):
        import core
        import paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups, jobs = root / "groups", root / "jobs"
            (groups / "demo-group").mkdir(parents=True)
            audio = root / "demo.wav"
            audio.write_bytes(b"first")
            data = recording({"s": 0.0, "e": 3.0})
            data.update({"file": str(audio), "group": "demo-group", "names_confirmed": True})
            source, _, _ = DB.new_record(self, data, root, "demo")
            with patch.object(core, "job_json", return_value=(str(source), None)), patch.object(paths, "GROUPS", groups), patch.object(paths, "JOBS", jobs), patch.object(core, "launch") as launch:
                snapshot = core.prepare_voice_action("demo")
                audio.write_bytes(b"other")
                with self.assertRaises(core.CoreError):
                    core.start_voice_update(snapshot)
                launch.assert_not_called()

    def test_changed_profile_blocks_confirmed_update(self):
        import core
        import paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups, jobs = root / "groups", root / "jobs"
            group = groups / "demo-group"
            group.mkdir(parents=True)
            audio = root / "demo.wav"
            audio.write_bytes(b"synthetic audio")
            source, _, _ = DB.new_record(self, {"file": str(audio), "group": "demo-group",
                                          "names_confirmed": True, "segments": []}, root, "demo")
            target = group / "voices.json"
            profile = {"version": voices.PROFILE_VERSION, "speakers": {}, "sources": {}}
            target.write_text(json.dumps(profile), encoding="utf-8")
            with patch.object(core, "job_json", return_value=(str(source), None)), \
                 patch.object(paths, "GROUPS", groups), patch.object(paths, "JOBS", jobs), \
                 patch.object(core, "launch") as launch:
                snapshot = core.prepare_voice_action("demo")
                target.write_text(json.dumps({**profile, "revision": 2}), encoding="utf-8")
                with self.assertRaises(core.CoreError):
                    core.start_voice_update(snapshot)
            launch.assert_not_called()

    def test_changed_profile_blocks_revocation_and_preserves_newer_profile(self):
        import core
        import paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups, jobs = root / "groups", root / "jobs"
            group = groups / "demo-group"
            group.mkdir(parents=True)
            audio = root / "demo.wav"
            audio.touch()
            data = recording({"s": 0.0, "e": 3.0})
            data.update({"file": str(audio), "group": "demo-group", "names_confirmed": True})
            source, _, _ = DB.new_record(self, data, root, "demo")
            profile = {"version": voices.PROFILE_VERSION,
                       "speakers": {"Анна": {"parts": {str(audio.resolve()): {"vec": [1], "sec": 1}}}}, "sources": {}}
            target = group / "voices.json"
            target.write_text(json.dumps(profile), encoding="utf-8")
            with patch.object(core, "job_json", return_value=(str(source), None)), patch.object(paths, "GROUPS", groups), patch.object(paths, "JOBS", jobs):
                snapshot = core.prepare_voice_action("demo")
                target.write_text(json.dumps({**profile, "revision": 2}), encoding="utf-8")
                newer = target.read_bytes()
                with self.assertRaises(core.CoreError):
                    core.revoke_voice_source(snapshot)
            self.assertEqual(target.read_bytes(), newer)
            self.assertFalse((group / "voice-backups").exists())
