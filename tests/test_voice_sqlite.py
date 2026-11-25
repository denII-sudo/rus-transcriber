"""Голоса и записи в базе: страховочная сетка до обучения профилей по SQLite (Г1).

Только синтетические данные во временной папке; модели, настоящие профили и база не используются.
Закрепляется: (1) запись, перенесённая в базу так же, как `tools/import_legacy.py`, даёт через
`recordings.load_recording` тот же отбор отрезков, что и старый файл; (2) правка агента и отменённая
отметка человека не считаются подтверждением человека ни в файле, ни в базе; (3) нынешний отказ
обучаться по SQLite-записи во всех трёх местах; (4) прежнее поведение `voices.learn` на файлах.
"""
import copy
import json
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import import_legacy as IL  # noqa: E402
import core  # noqa: E402
import documents  # noqa: E402
import paths  # noqa: E402
import recordings  # noqa: E402
import storage  # noqa: E402
import voices  # noqa: E402
import worker  # noqa: E402


def word(s, e, **fields):
    return {"w": "слово", "s": s, "e": e, **fields}


def by(author, reverted=False):
    speaker = {"by": author, **({"reverted": True} if reverted else {})}
    return {"provenance": {"speaker": speaker}}


def legacy_data(audio, confirmed=True):
    """Старая стенограмма: человек, модель, сомнение, правка агента, отменённая отметка, наложение."""
    return {
        "file": str(audio), "group": "demo-group", "names_confirmed": confirmed,
        "speakers": ["Анна", "Борис", "Вера", "Спикер 1"],
        "segments": [
            {"speaker": "Анна", "start": 0.0, "end": 4.6, "text": "a", "words": [
                word(0.0, 0.9, **by("human")), word(0.9, 1.8, **by("human")),
                word(1.8, 2.6), word(2.6, 3.4), word(3.4, 4.2), word(4.2, 4.6, flags=["doubt"])]},
            {"speaker": "Борис", "start": 5.0, "end": 8.0, "text": "b", "words": [
                word(5.0, 6.0, **by("claude")), word(6.0, 7.0, **by("human", reverted=True)),
                word(7.0, 8.0, inserted=True)]},
            {"speaker": "Вера", "start": 7.5, "end": 9.5, "text": "c", "words": [word(7.5, 9.5)]},
            {"speaker": "Вера", "start": 20.0, "end": 23.0, "text": "e", "words": [word(20.0, 23.0)]},
            {"speaker": "Спикер 1", "start": 10.0, "end": 13.0, "text": "d", "words": [word(10.0, 13.0)]},
        ],
        "edits": [
            {"by": "human", "before": {"speaker": "Борис"}, "after": {"speaker": "Анна"}},
            {"by": "claude", "before": {"speaker": "Анна"}, "after": {"speaker": "Вера"}},
            {"by": "human", "reverted": True, "before": {"speaker": "Вера"}, "after": {"speaker": "Борис"}},
        ],
        "review": [], "review_done": True,
    }


def agent_only_data(audio):
    """Имена не подтверждены; подписи говорящих — только правка агента и отменённая отметка человека."""
    return {
        "file": str(audio), "group": "demo-group",
        "segments": [{"speaker": "Анна", "start": 0.0, "end": 6.0, "text": "a", "words": [
            word(0.0, 3.0, **by("claude")), word(3.0, 6.0, **by("human", reverted=True))]}],
        "edits": [{"by": "claude", "before": {"speaker": "Борис"}, "after": {"speaker": "Анна"}},
                  {"by": "human", "reverted": True, "before": {"speaker": "Вера"}, "after": {"speaker": "Анна"}}],
        "review": [],
    }


def selection(data):
    report = {}
    spans = voices._spans(data, voices._confirmed(data), report)
    return spans, report


class FakeEmbedder:
    def spans(self, audio, spans, cap):
        return [np.array([1.0, 0.0])] * len(spans)


class Status:
    def __init__(self):
        self.data = {}

    def set(self, **values):
        self.data.update(values)


class SyntheticStore(unittest.TestCase):
    """Временная база и папка записи; перенос — тем же путём, что у `import_legacy`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="voice-sqlite-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "state.sqlite3"
        self.store = storage.initialize(self.db)
        self.addCleanup(self.store.close)
        user = self.store.create_user("Тест")
        self.store.bind_installation(str(uuid.uuid4()), user)
        self.audio = self.root / "meeting.wav"
        self.audio.write_bytes(b"synthetic audio placeholder")

    def migrate(self, legacy, legacy_path, tag=""):
        """Записать старую стенограмму в базу как перенос П7; -> (recording_id, путь выгрузки, source)."""
        legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        source = {"path": str(legacy_path), "sha256": IL.sha256_file(legacy_path),
                  "size": legacy_path.stat().st_size}
        rid, transcript, review, _ = IL.build_snapshots(
            json.loads(legacy_path.read_text(encoding="utf-8")),
            recording_id=IL.recording_id_for(legacy_path, source["sha256"]), metadata={}, field_sources={},
            variant={"tag": tag, "main": True}, source=source)
        layout = recordings.document_layout(self.audio, tag)
        layout["root"].mkdir(exist_ok=True)
        root_id = self.store.get_or_register_root(layout["root"])
        self.store.create_recording_with_document(
            rid, recordings.normalize_location(self.audio), transcript, review, str(uuid.uuid4()),
            recordings.operation_context(self.store, "system"), root_id=root_id,
            generated_at="2026-11-25T00:00:00+00:00", document_paths=layout["paths"])
        documents.materialize_pending(self.store, root_id, layout["root"])
        return rid, layout["root"] / layout["paths"]["json"], source


class MigratedRecordingParity(SyntheticStore):
    def test_load_recording_gives_same_selection_as_legacy_file(self):
        for confirmed in (True, False):
            with self.subTest(confirmed=confirmed):
                legacy = legacy_data(self.audio, confirmed)
                rid, _, source = self.migrate(legacy, self.root / f"meeting.v{int(confirmed)}.transcript.json",
                                              tag=f"v{int(confirmed)}")
                loaded = recordings.load_recording(self.store, rid)
                self.assertEqual(selection(loaded), selection(legacy))
                _, report = selection(legacy)
                self.assertEqual(report.get("accepted_human"), 1)
                self.assertEqual(report.get("accepted_auto", 0), 3 if confirmed else 0)
                self.assertEqual("foreign_overlap" in report, confirmed)   # без подтверждения слов модели нет
                self.assertIn("unknown_speaker", report)
                for check in (voices._confirmed, voices.has_human_edits, voices._human_pairs, voices.group_of):
                    self.assertEqual(check(loaded), check(legacy), check.__name__)
                self.assertEqual(voices._human_pairs(loaded), {("Борис", "Анна"): 1})
                self.assertEqual((loaded["file"], loaded.get("no_profile")), (legacy["file"], None))
                # связь со старым ключом профиля, на которую опирается защита от двойного учёта
                self.assertEqual(loaded["quality"]["migration"]["source"], source)
                self.assertEqual(loaded["source_revision"], 1)


class AgentEditIsNotHuman(SyntheticStore):
    def test_agent_edit_and_reverted_mark_give_no_human_seconds_in_file_and_store(self):
        legacy = agent_only_data(self.audio)
        rid, _, _ = self.migrate(legacy, self.root / "meeting.transcript.json")
        loaded = recordings.load_recording(self.store, rid)
        for name, data in (("file", legacy), ("store", loaded)):
            with self.subTest(source=name):
                self.assertFalse(voices._confirmed(data))
                self.assertFalse(voices.has_human_edits(data))
                self.assertEqual(voices._human_pairs(data), {})
                self.assertFalse(any(voices._word_human(w) for s in data["segments"] for w in s["words"]))
                spans, report = selection(data)
                self.assertEqual(spans, [])
                self.assertNotIn("accepted_human", report)

    def test_learn_skips_source_confirmed_only_by_agent(self):
        folder = self.root / "only-agent"
        folder.mkdir()
        (folder / "meeting.transcript.json").write_text(json.dumps(agent_only_data(self.audio)), encoding="utf-8")
        profile, report = voices.empty_profile(), {}
        learned = voices.learn(profile, folder, FakeEmbedder(), lambda _: self.fail("audio loaded"),
                               selection_report=report)
        self.assertEqual((learned, report, profile["speakers"], profile["sources"]),
                         ([], {"unconfirmed_source": 1}, {}, {}))


class CurrentSqliteRefusal(SyntheticStore):
    """Нынешний отказ: Г2 заменит его рабочим путём и поменяет эти проверки вместе с кодом."""

    def test_learn_preview_and_worker_refuse_sqlite_record_without_touching_files(self):
        legacy_path = self.root / "meeting.transcript.json"
        rid, export, _ = self.migrate(legacy_data(self.audio), legacy_path)
        legacy_path.unlink()                    # в папке аудио остаётся только выгрузка базы
        export_before = export.read_bytes()
        groups = self.root / "groups"
        (groups / "demo-group").mkdir(parents=True)
        profile, report = voices.empty_profile(), {}
        self.assertEqual(voices.learn(profile, self.root, FakeEmbedder(), lambda _: self.fail("audio loaded"),
                                      selection_report=report), [])
        self.assertEqual(report, {"sqlite_source_not_supported": 1})
        self.assertEqual((profile["speakers"], profile["sources"]), ({}, {}))
        status = Status()
        with patch.object(paths, "STATE_DB", self.db), patch.object(paths, "GROUPS", groups), \
                patch.object(core, "job_json", return_value=(str(export), None)):
            for action in (core.preview_voice_learning, core.prepare_voice_action):
                with self.subTest(action=action.__name__), \
                        self.assertRaisesRegex(core.CoreError, "SQLite-записи пока не поддерживается"):
                    action("synthetic-job")
            args = types.SimpleNamespace(learn_only=str(export), voice_group_override="")
            with self.assertRaisesRegex(RuntimeError, "SQLite-записи пока не поддерживается"):
                worker.learn_only(args, status, {}, self.audio)
        self.assertEqual(status.data["state"], "error")
        self.assertEqual(export.read_bytes(), export_before)
        self.assertEqual(list((groups / "demo-group").iterdir()), [])
        self.assertEqual(recordings.load_recording(self.store, rid)["source_revision"], 1)


class LegacyFilesBaseline(unittest.TestCase):
    """Прежнее поведение на файлах, которое перестройка на слой источников не должна менять."""

    def test_learn_keys_parts_by_audio_and_sources_by_transcript_then_skips_repeat(self):
        with tempfile.TemporaryDirectory(prefix="voice-files-") as directory:
            root = Path(directory)
            audio = root / "meeting.wav"
            audio.write_bytes(b"synthetic audio placeholder")
            source = root / "meeting.transcript.json"
            source.write_text(json.dumps(legacy_data(audio), ensure_ascii=False), encoding="utf-8")
            profile, report = voices.empty_profile(), {}
            learned = voices.learn(profile, root, FakeEmbedder(), lambda _: np.zeros(16000),
                                   group="demo-group", selection_report=report)
            self.assertEqual(learned, [source.name])
            audio_key, source_key = str(audio.resolve()), str(source.resolve())
            self.assertEqual(sorted(profile["speakers"]), ["Анна", "Борис", "Вера"])
            for name in ("Анна", "Борис", "Вера"):
                self.assertEqual(list(profile["speakers"][name]["parts"]), [audio_key])
            anna = profile["speakers"]["Анна"]["parts"][audio_key]
            self.assertEqual({k: anna[k] for k in ("from", "confirmed", "selection_version")},
                             {"from": source.name, "confirmed": True,
                              "selection_version": voices.SELECTION_VERSION})
            # человек главнее: ×3 только у отрезка с отметкой человека; слова агента и отменённая
            # отметка у Бориса идут как обычные слова модели подтверждённой расшифровки
            seconds = {name: (part["human_sec"], part["auto_sec"], part["sec"])
                       for name, sp in profile["speakers"].items() for part in sp["parts"].values()}
            self.assertEqual(seconds, {"Анна": (1.8, 2.4, 7.8), "Борис": (0.0, 2.0, 2.0),
                                       "Вера": (0.0, 3.0, 3.0)})
            self.assertEqual(list(profile["sources"]), [source_key])
            self.assertEqual(profile["sources"][source_key], voices.source_fingerprint(source, audio))
            self.assertEqual(profile["source_selection_versions"], {source_key: voices.SELECTION_VERSION})
            self.assertEqual(profile["confusions"], {audio_key: {"Борис|Анна": 1}})
            self.assertEqual(profile["revision"], 1)
            self.assertEqual(report.get("added"), 3)

            snapshot = copy.deepcopy(profile)
            again = {}
            self.assertEqual(voices.learn(profile, root, FakeEmbedder(), lambda _: self.fail("audio loaded"),
                                          group="demo-group", selection_report=again), [])
            self.assertEqual(again, {"already_processed": 1})
            self.assertEqual(json.dumps(profile, default=list, sort_keys=True),
                             json.dumps(snapshot, default=list, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
