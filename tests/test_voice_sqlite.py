"""Голоса и записи в базе: страховочная сетка до обучения профилей по SQLite (Г1).

Только синтетические данные во временной папке; модели, настоящие профили и база не используются.
Закрепляется: (1) запись, перенесённая в базу так же, как `tools/import_legacy.py`, даёт через
`recordings.load_recording` тот же отбор отрезков, что и старый файл; (2) правка агента и отменённая
отметка человека не считаются подтверждением человека ни в файле, ни в базе; (3) ручной путь по записи в
базе (Г2): предпросмотр без записи в базу, подтверждённое обновление и отзыв под резервом, перевязка старого
ключа, отказ при изменении записи, повтор без изменений; (4) прежнее поведение `voices.learn` на файлах.
"""
import copy
import hashlib
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
import core_voices  # noqa: E402
import paths  # noqa: E402
import recordings  # noqa: E402
import settings  # noqa: E402
import storage  # noqa: E402
import transcript  # noqa: E402
import voice_sources  # noqa: E402
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

    def migrate(self, legacy, legacy_path, tag="", recording_id=None):
        """Записать старую стенограмму в базу как перенос П7; -> (recording_id, путь выгрузки, source)."""
        legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        source = {"path": str(legacy_path), "sha256": IL.sha256_file(legacy_path),
                  "size": legacy_path.stat().st_size}
        rid, snapshot, review, _ = IL.build_snapshots(
            json.loads(legacy_path.read_text(encoding="utf-8")),
            recording_id=recording_id or IL.recording_id_for(legacy_path, source["sha256"]), metadata={},
            field_sources={}, variant={"tag": tag, "main": True}, source=source)
        layout = recordings.document_layout(self.audio, tag)
        layout["root"].mkdir(exist_ok=True)
        root_id = self.store.get_or_register_root(layout["root"])
        self.store.create_recording_with_document(
            rid, recordings.normalize_location(self.audio), snapshot, review, str(uuid.uuid4()),
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


CONFIRMATION = {"by": "human", "note": "синтетическая пометка с переноса", "decided": "2026-11-20"}


class ManualSqlitePath(SyntheticStore):
    """Ручной путь по записи в базе: предпросмотр, подтверждённое обновление воркером и отзыв."""

    def setUp(self):
        super().setUp()
        self.groups = self.root / "groups"
        self.group = self.groups / "demo-group"
        self.group.mkdir(parents=True)
        (self.group / "vocab.json").write_text(json.dumps(
            {"group": "demo-group", "participants": [{"name": n} for n in ("Анна", "Борис", "Вера")]},
            ensure_ascii=False), encoding="utf-8")
        self.profile_path = self.group / "voices.json"
        self.legacy_path = self.root / "meeting.transcript.json"
        self.rid, self.export, _ = self.migrate(legacy_data(self.audio), self.legacy_path)
        self.ref = voice_sources.RECORDING_PREFIX + self.rid
        self.audio_key = str(self.audio.resolve())
        self.legacy_key = str(self.legacy_path.resolve())
        # профиль, обученный до переноса по старому файлу, с пометкой «кем подтверждены голоса»
        profile = voices.empty_profile()
        profile["speakers"]["Анна"] = {"parts": {self.audio_key: {
            "vec": np.array([1.0, 0.0]), "sec": 9.0, "human_sec": 0.0, "auto_sec": 9.0,
            "from": self.legacy_path.name, "confirmed": True, "selection_version": voices.SELECTION_VERSION,
            "confirmation": CONFIRMATION}}}
        profile["sources"][self.legacy_key] = voices.source_fingerprint(self.legacy_path, self.audio)
        profile["source_selection_versions"] = {self.legacy_key: voices.SELECTION_VERSION}
        voices.save_profile(self.profile_path, profile)
        for item in (patch.object(paths, "STATE_DB", self.db), patch.object(paths, "GROUPS", self.groups),
                     patch.object(paths, "JOBS", self.root / "jobs")):
            item.start()
            self.addCleanup(item.stop)

    def backups(self):
        folder = self.group / "voice-backups"
        return sorted(folder.iterdir()) if folder.is_dir() else []

    def start(self, snapshot):
        """start_voice_update без запуска процесса; -> аргументы воркера."""
        launched = []
        with patch.object(core, "launch", side_effect=lambda script, args, src, kind, **_: launched.append(args)
                          or "job-learn"):
            core.start_voice_update(snapshot)
        self.assertEqual(len(launched), 1)
        return launched[0]

    def run_worker(self, args):
        values = {key[2:].replace("-", "_"): value for key, value in zip(args[::2], args[1::2])}
        namespace = types.SimpleNamespace(vocab="", group="", **values)
        fake_audio = types.ModuleType("pyannote.audio")
        fake_audio.Pipeline = types.SimpleNamespace(from_pretrained=lambda *a, **k: object())
        status = Status()
        with patch.dict("sys.modules", {"pyannote.audio": fake_audio}), \
                patch.object(worker, "load_audio", return_value=np.zeros(16000)), \
                patch.object(voices, "Embedder", lambda _pipe: FakeEmbedder()):
            worker.learn_only(namespace, status, {}, self.audio)
        return status

    def commit_new_revision(self):
        data = recordings.load_recording(self.store, self.rid)
        data["segments"][0]["text"] = "исправлено человеком после просмотра"
        root_id = self.store.get_or_register_root(recordings.document_layout(self.audio)["root"])
        recordings.commit_recording_data(
            self.store, self.rid, data, {"source_revision": 1, "review_revision": 1}, str(uuid.uuid4()),
            recordings.operation_context(self.store, "human"), root_id, "2026-11-25T01:00:00+00:00")
        self.assertEqual(self.store.get_current(self.rid)["transcript"]["revision"], 2)

    def store_state(self):
        tables = [row[0] for row in self.store.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        counts = {name: self.store.db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] for name in tables}
        heads = [tuple(row) for row in self.store.db.execute(
            "SELECT recording_id, source_revision, review_revision FROM recordings ORDER BY recording_id")]
        # data_version этого соединения меняется при любой фиксации из другого соединения, даже без новых строк
        version = self.store.db.execute("PRAGMA data_version").fetchone()[0]
        content = hashlib.sha256("\n".join(self.store.db.iterdump()).encode("utf-8")).hexdigest()
        return counts, heads, version, content

    def test_preview_and_prepare_read_revision_without_writing_store_or_files(self):
        before = (self.store_state(), self.export.read_bytes(), self.profile_path.read_bytes())
        # черновик ответов и файл выгрузки в обучение не идут: transcript.load не вызывается;
        # база открывается только на чтение: соединение на запись — отказ теста
        with patch.object(transcript, "load", side_effect=AssertionError("transcript.load в обучении")), \
                patch.object(storage, "open_store", side_effect=AssertionError("база открыта на запись")):
            report = core.preview_voice_learning(self.ref)
            snapshot = core.prepare_voice_action(self.ref)
        self.assertEqual(before, (self.store_state(), self.export.read_bytes(), self.profile_path.read_bytes()))
        self.assertEqual(report["source"], {"kind": "sqlite", "name": self.export.name, "recording": self.ref,
                                            "revision": 1, "migrated_from": self.legacy_path.name})
        self.assertEqual(report["existing_contributions"], ["Анна"])
        self.assertEqual(report["confirmation_replaced"], ["Анна"])
        self.assertEqual(report["other_variant_contributions"], [])
        self.assertEqual(sorted(report["candidates"]), ["Анна", "Борис", "Вера"])
        store_id = self.store.db.execute("SELECT store_id FROM store_meta").fetchone()[0]
        payload_hash = self.store.get_current(self.rid)["transcript"]["payload_hash"]
        self.assertEqual(snapshot["source_hash"], f"sqlite:{store_id}:{self.rid}:1:{payload_hash}")
        self.assertEqual((snapshot["source_ref"], snapshot["source_kind"]), (self.ref, "sqlite"))
        lines = core.voice_action_lines(snapshot)
        self.assertIn(f"Источник: запись в базе, ревизия 1, перенесена из {self.legacy_path.name}; "
                      "её старый ключ в профиле снимается.", lines)
        self.assertIn("Пометка «кем подтверждены голоса» заменится подсчётом секунд (прежняя — в истории): Анна.",
                      lines)
        for revoke in (False, True):
            lines = core.voice_action_lines(snapshot, revoke=revoke)
            self.assertTrue(all(len(line) <= 200 for line in lines) and len("\n".join(lines)) <= 1100)
        # длинный отчёт отбора сворачивается в одну строку, обязательные строки остаются
        crowded = copy.deepcopy(snapshot)
        crowded["report"]["selection_explanation"] = [
            {"description": f"Синтетическая причина отбора номер {i} с длинным пояснением", "count": i}
            for i in range(30)]
        lines = core.voice_action_lines(crowded)
        self.assertIn("Предварительный отбор: причин 30, подробно — в предпросмотре "
                      "(результат модели может отличаться).", lines)
        self.assertTrue(any(line.startswith("Пометка «кем подтверждены голоса»") for line in lines))
        self.assertLessEqual(len("\n".join(lines)), 1100)

    def test_confirmed_update_replaces_part_rekeys_old_source_and_replay_changes_nothing(self):
        before = self.profile_path.read_bytes()
        snapshot = core.prepare_voice_action(self.ref)
        args = self.start(snapshot)
        self.assertEqual(args[:2], ["--learn-only", self.ref])
        status = self.run_worker(args)
        self.assertEqual(status.data["state"], "done")
        self.assertEqual(status.data["learned"], [self.export.name])
        profile = voices.load_profile(self.profile_path)
        self.assertEqual(sorted(profile["speakers"]), ["Анна", "Борис", "Вера"])
        for speaker in profile["speakers"].values():          # один вклад на аудио, не второй рядом
            self.assertEqual(list(speaker["parts"]), [self.audio_key])
        anna = profile["speakers"]["Анна"]["parts"][self.audio_key]
        self.assertNotIn("confirmation", anna)
        self.assertEqual((anna["human_sec"], anna["auto_sec"], anna["from"]), (1.8, 2.4, self.export.name))
        self.assertEqual(list(profile["sources"]), [self.ref])
        fingerprint = profile["sources"][self.ref]
        self.assertEqual((fingerprint["kind"], fingerprint["recording_id"], fingerprint["transcript_revision"]),
                         ("sqlite", self.rid, 1))
        self.assertEqual(profile["source_selection_versions"], {self.ref: voices.SELECTION_VERSION})
        rekeyed, learned = profile["history"][-2:]
        self.assertEqual((learned["source"], learned["previous_confirmation"]), (self.ref, {"Анна": CONFIRMATION}))
        self.assertEqual((rekeyed["kind"], rekeyed["from"], rekeyed["to"]),
                         ("source_rekeyed", [self.legacy_path.name], self.ref))
        self.assertEqual([path.read_bytes() for path in self.backups()], [before])

        after = self.profile_path.read_bytes()
        with patch.object(core, "launch", side_effect=AssertionError("повтор запущен")), \
                self.assertRaisesRegex(core.CoreError, "повторите"):
            core.start_voice_update(snapshot)
        with self.assertRaisesRegex(RuntimeError, "нужен новый просмотр"):
            self.run_worker(args)
        self.assertEqual(self.profile_path.read_bytes(), after)
        self.assertEqual(len(self.backups()), 1)

    def test_revoke_removes_part_and_linked_old_key_after_backup(self):
        before = self.profile_path.read_bytes()
        snapshot = core.prepare_voice_action(self.ref)
        revoke_lines = core.voice_action_lines(snapshot, revoke=True)
        self.assertTrue(any("её старый ключ в профиле снимается" in line for line in revoke_lines))
        self.assertFalse(any("Пометка" in line for line in revoke_lines))
        result = core.revoke_voice_source(snapshot)
        self.assertEqual(result["revoked"], ["Анна"])
        self.assertEqual(Path(result["backup"]).read_bytes(), before)
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        self.assertEqual((profile["speakers"]["Анна"]["parts"], profile["sources"],
                          profile["source_selection_versions"]), ({}, {}, {}))
        self.assertEqual((profile["history"][-1]["kind"], profile["history"][-1]["source"]),
                         ("source_revoked", self.ref))

    def test_failure_between_backup_and_write_keeps_profile_bytes_and_backup(self):
        before = self.profile_path.read_bytes()
        args = self.start(core.prepare_voice_action(self.ref))
        real_replace = Path.replace

        def failing_replace(path, target):
            if Path(path).suffix == ".tmp":
                raise OSError("сбой записи профиля (проверка)")
            return real_replace(path, target)

        with patch.object(Path, "replace", failing_replace), self.assertRaisesRegex(OSError, "сбой записи"):
            self.run_worker(args)
        self.assertEqual(self.profile_path.read_bytes(), before)
        self.assertEqual([path.read_bytes() for path in self.backups()], [before])
        with patch.object(core_voices.os, "replace", side_effect=OSError("сбой замены (проверка)")), \
                self.assertRaisesRegex(OSError, "сбой замены"):
            core.revoke_voice_source(core.prepare_voice_action(self.ref))
        self.assertEqual(self.profile_path.read_bytes(), before)
        self.assertEqual([path.read_bytes() for path in self.backups()], [before, before])
        self.assertEqual(sorted(path.name for path in self.group.iterdir()),
                         ["vocab.json", "voice-backups", "voices.json"])

    def test_record_changed_after_preview_is_refused_without_writing(self):
        before = self.profile_path.read_bytes()
        snapshot = core.prepare_voice_action(self.ref)
        args = self.start(snapshot)
        self.commit_new_revision()
        with patch.object(core, "launch", side_effect=AssertionError("запуск по старым данным")), \
                self.assertRaisesRegex(core.CoreError, "повторите"):
            core.start_voice_update(snapshot)
        with self.assertRaisesRegex(RuntimeError, "нужен новый просмотр"):
            self.run_worker(args)
        with self.assertRaisesRegex(core.CoreError, "повторите"):
            core.revoke_voice_source(snapshot)
        self.assertEqual(self.profile_path.read_bytes(), before)
        self.assertEqual(self.backups(), [])

    def test_other_store_identity_changes_token_and_fingerprint(self):
        snapshot = core.prepare_voice_action(self.ref)
        with patch.object(voice_sources, "_store_id", return_value="восстановленная-база"), \
                patch.object(core, "launch", side_effect=AssertionError("запуск по чужой базе")), \
                self.assertRaisesRegex(core.CoreError, "повторите"):
            core.start_voice_update(snapshot)
        source = voice_sources.open_ref(self.ref)
        fingerprint = source.fingerprint(self.audio)
        for field, value in (("store_id", "другая"), ("transcript_revision", 2), ("transcript_payload_hash", "x")):
            self.assertEqual(voices.source_fingerprint_status(fingerprint, {**fingerprint, field: value}, 0.0),
                             "changed")
        self.assertEqual(voices.source_fingerprint_status(fingerprint, dict(fingerprint), 0.0), "unchanged")

    def test_audio_key_in_other_letter_case_does_not_add_second_part(self):
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        variant = self.audio_key.swapcase()
        profile["speakers"]["Анна"]["parts"] = {variant: profile["speakers"]["Анна"]["parts"][self.audio_key]}
        self.profile_path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
        snapshot = core.prepare_voice_action(self.ref)
        self.assertEqual(snapshot["report"]["existing_contributions"], ["Анна"])
        self.run_worker(self.start(snapshot))
        profile = voices.load_profile(self.profile_path)
        self.assertEqual({name: list(s["parts"]) for name, s in profile["speakers"].items()},
                         {name: [variant] for name in ("Анна", "Борис", "Вера")})

    def test_second_variant_of_same_audio_replaces_with_warning(self):
        second = legacy_data(self.audio)
        second["segments"][0]["text"] = "второй вариант"
        rid, export, _ = self.migrate(second, self.root / "meeting.v2.transcript.json", tag="v2")
        snapshot = core.prepare_voice_action(voice_sources.RECORDING_PREFIX + rid)
        self.assertEqual(snapshot["report"]["other_variant_contributions"], [self.legacy_path.name])
        lines = core.voice_action_lines(snapshot)
        self.assertIn(f"Заменит вклад другого варианта стенограммы: {self.legacy_path.name}.", lines)
        self.assertLessEqual(len("\n".join(lines)), core_voices.VOICE_ACTION_TEXT_LIMIT)
        self.run_worker(self.start(snapshot))
        profile = voices.load_profile(self.profile_path)
        self.assertEqual(list(profile["speakers"]["Анна"]["parts"]), [self.audio_key])
        self.assertEqual(profile["speakers"]["Анна"]["parts"][self.audio_key]["from"], export.name)
        # ключ старого файла первого варианта не снимается: вторая копия перенесена из другого файла
        self.assertEqual(sorted(profile["sources"]), sorted([self.legacy_key, "recording:" + rid]))

    def test_revoke_by_second_variant_warns_that_other_variant_contribution_goes_too(self):
        second = legacy_data(self.audio)
        second["segments"][0]["text"] = "второй вариант"
        rid, _, _ = self.migrate(second, self.root / "meeting.v2.transcript.json", tag="v2")
        snapshot = core.prepare_voice_action(voice_sources.RECORDING_PREFIX + rid)
        lines = core.voice_action_lines(snapshot, revoke=True)
        # вклад хранится по ключу аудио: отзыв второго варианта снимает и вклад, обученный по первому
        self.assertIn(f"Снимет и вклад другого варианта стенограммы: {self.legacy_path.name}.", lines)
        self.assertIn("Удалить перечисленные вклады этого аудио; другие записи не меняются.", lines)
        self.assertFalse(any("другие источники не меняются" in line for line in lines))
        self.assertLessEqual(len("\n".join(lines)), core_voices.VOICE_ACTION_TEXT_LIMIT)
        result = core.revoke_voice_source(snapshot)
        self.assertEqual(result["revoked"], ["Анна"])
        self.assertEqual(voices.load_profile(self.profile_path)["speakers"]["Анна"]["parts"], {})

    def test_old_job_id_redirects_only_to_single_unchanged_copy(self):
        with patch.object(settings, "recording_backend", return_value="sqlite"), \
                patch.object(core, "job_json", return_value=(str(self.legacy_path), None)):
            snapshot = core.prepare_voice_action("old-job")
            self.assertEqual(snapshot["report"]["source"]["redirected_from"], self.legacy_path.name)
            self.assertEqual(snapshot["source_ref"], self.ref)
            self.assertTrue(any("найдена по старому номеру задачи" in line
                                for line in core.voice_action_lines(snapshot)))
            self.assertEqual(self.start(snapshot)[:2], ["--learn-only", self.ref])
            self.migrate(legacy_data(self.audio), self.legacy_path, tag="copy", recording_id=str(uuid.uuid4()))
            with self.assertRaisesRegex(core.CoreError, r"нет единственной копии в базе \(найдено 2\)"):
                core.preview_voice_learning("old-job")
        lonely = self.root / "other.transcript.json"
        lonely.write_text(json.dumps(legacy_data(self.audio)), encoding="utf-8")
        with patch.object(settings, "recording_backend", return_value="sqlite"), \
                patch.object(core, "job_json", return_value=(str(lonely), None)), \
                self.assertRaisesRegex(core.CoreError, r"нет единственной копии в базе \(найдено 0\)"):
            core.preview_voice_learning("old-job")

    def test_worker_refuses_record_without_confirmed_snapshot(self):
        before = self.profile_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "только после просмотра и подтверждения"):
            self.run_worker(["--learn-only", self.ref])
        self.assertEqual(self.profile_path.read_bytes(), before)
        report = {}
        self.assertEqual(voices.learn(voices.empty_profile(), self.root, FakeEmbedder(), lambda _: np.zeros(1),
                                      only=[voice_sources.open_ref(self.ref)], selection_report=report), [])
        self.assertEqual(report, {"sqlite_source_not_supported": 1})

    def test_server_tools_change_nothing_without_human_confirmation(self):
        import asyncio
        import confirm
        import migrate
        with patch.object(migrate, "run"):
            import server
        before = self.profile_path.read_bytes()
        with patch.object(confirm, "require", side_effect=confirm.NotConfirmed("отказ человека")), \
                patch.object(core, "launch", side_effect=AssertionError("запуск без подтверждения")):
            self.assertIn("отказ человека", asyncio.run(server.update_voice_profiles(self.ref)))
            self.assertIn("отказ человека", asyncio.run(server.revoke_voice_profiles(self.ref)))
        self.assertEqual(self.profile_path.read_bytes(), before)
        self.assertEqual(self.backups(), [])


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
