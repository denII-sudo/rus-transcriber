"""М-2: старый номер задачи после переключения на базу ведёт к перенесённой записи (В-2, «Переадресация»).

Только синтетические данные во временной папке. Перенос — тем же путём, что `tools/import_legacy.py`
(`build_snapshots`, `recording_id_for`, `sha256_file`), как в `tests/test_voice_sqlite.py`. Закрепляется:
переадресация по единственной копии с тем же содержимым и по тому же пути (в любом написании); отказы с
подсказкой (не перенесена, изменена после переноса, файла нет, файла нет и по его пути ничего не переносили,
несколько копий, копия файла по другому пути); не основной вариант; без изменений — задачи, сделанные уже в
базе, и неактивная база; переадресация только читает.
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-legacy-redirect-data-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import import_legacy as IL  # noqa: E402
import core  # noqa: E402
import documents  # noqa: E402
import paths  # noqa: E402
import recordings  # noqa: E402
import settings  # noqa: E402
import storage  # noqa: E402
import transcript  # noqa: E402

JOB = "20261126-101500-tabcde"


def legacy_data(audio, text="старый текст", recording_id=None):
    """Стенограмма старого формата: две реплики, слова с временем."""
    data = {
        "file": str(audio), "duration": 13.0, "speakers": ["Анна", "Борис"], "review": [], "edits": [],
        "segments": [
            {"speaker": "Анна", "start": 0.0, "end": 2.0, "text": text,
             "words": [{"w": w, "s": float(i), "e": float(i) + 1.0} for i, w in enumerate(text.split())]},
            {"speaker": "Борис", "start": 10.0, "end": 13.0, "text": "ответ",
             "words": [{"w": "ответ", "s": 10.0, "e": 13.0}]},
        ],
    }
    if recording_id:
        data["recording_id"] = recording_id
    return data


def facts(path):
    p = Path(path)
    return (p.read_bytes(), p.stat().st_mtime_ns) if p.exists() else None


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="legacy-redirect-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "state.sqlite3"
        self.store = storage.initialize(self.db)
        self.addCleanup(self.store.close)
        user = self.store.create_user("Тест")
        self.store.bind_installation(str(uuid.uuid4()), user)
        self.audio = self.root / "meeting.wav"
        self.audio.write_bytes(b"synthetic audio placeholder")
        self.jobs = self.root / "jobs"
        self.jobs.mkdir()
        self.legacy = self.root / "meeting.transcript.json"
        self.backend = "sqlite"
        for item in (patch.object(paths, "STATE_DB", self.db), patch.object(core, "JOBS", self.jobs),
                     patch.object(settings, "recording_backend", side_effect=lambda: self.backend)):
            item.start()
            self.addCleanup(item.stop)

    def write_legacy(self, data, path=None):
        path = path or self.legacy
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        path.with_suffix("").with_suffix(".md").write_text("# старый md\n", encoding="utf-8")
        return path

    def migrate(self, path=None, tag="", main=True, recording_id=None):
        """Перенести старый файл в базу как П7; -> recording_id."""
        path = path or self.legacy
        source = {"path": str(path), "sha256": IL.sha256_file(path), "size": path.stat().st_size}
        legacy = json.loads(path.read_text(encoding="utf-8"))
        rid = recording_id or IL.recording_id_from(legacy, path, source["sha256"])[0]
        rid, snapshot, review, _ = IL.build_snapshots(
            legacy, recording_id=rid, metadata={}, field_sources={},
            variant={"tag": tag, "main": main}, source=source)
        layout = recordings.document_layout(self.audio, tag)
        layout["root"].mkdir(exist_ok=True)
        root_id = self.store.get_or_register_root(layout["root"])
        self.store.create_recording_with_document(
            rid, recordings.normalize_location(self.audio), snapshot, review, str(uuid.uuid4()),
            recordings.operation_context(self.store, "system"), root_id=root_id,
            generated_at="2026-11-26T00:00:00+00:00", document_paths=layout["paths"])
        documents.materialize_pending(self.store, root_id, layout["root"])
        return rid

    def job(self, path=None, job_id=JOB):
        """Файл готовой задачи; path — Path или строка, записанная в задачу как есть."""
        path = path or self.legacy
        (self.jobs / f"{job_id}.json").write_text(json.dumps(
            {"state": "done", "kind": "transcribe", "file": str(self.audio), "json": str(path),
             "markdown": str(Path(path).with_suffix("").with_suffix(".md"))}, ensure_ascii=False), encoding="utf-8")
        return job_id

    def operations(self):
        return self.store.db.execute("SELECT COUNT(*) FROM operations").fetchone()[0]

    def doc_of(self, rid):
        binding = recordings.document_binding(self.store, rid)
        return str(binding["root"] / binding["paths"]["json"])


class Redirect(Fixture):
    def test_old_job_reads_migrated_record_without_writing(self):
        """Чтение по старому номеру — запись в базе; файл задачи, старые файлы и база не меняются."""
        self.write_legacy(legacy_data(self.audio, recording_id=uuid.uuid4().hex))
        rid = self.migrate()
        self.job()
        watched = [self.jobs / f"{JOB}.json", self.legacy, self.legacy.with_suffix("").with_suffix(".md")]
        before, ops = [facts(p) for p in watched], self.operations()
        s = core.job_status(JOB)
        self.assertEqual(s["recording_id"], rid)
        self.assertEqual(s["recording_ref"], "recording:" + rid)
        self.assertEqual(s["json"], self.doc_of(rid))
        self.assertEqual(s["legacy_json"], str(self.legacy))
        self.assertEqual(s["redirected_from"], JOB)
        self.assertNotIn("variant_main", s)
        self.assertEqual(core.job_json(JOB), (s["json"], s["markdown"]))
        self.assertIn("старый текст", core.read_transcript(JOB))
        self.assertEqual(core.review_items(JOB)["total"], 0)
        self.assertEqual([facts(p) for p in watched], before)
        self.assertEqual(self.operations(), ops, "переадресация записала в базу")

    def test_write_through_old_job_goes_to_record(self):
        """Правка по старому номеру — в записи базы, как через recording:<id>; старый файл не меняется."""
        self.write_legacy(legacy_data(self.audio))
        rid = self.migrate()
        self.job()
        before = facts(self.legacy)
        core.edit_span(JOB, 10.0, 13.0, "новый ответ", by="claude")
        self.assertEqual(facts(self.legacy), before)
        data = recordings.load_recording(self.store, rid)
        self.assertEqual(data["segments"][1]["text"], "новый ответ")
        self.assertIn("новый ответ", core.read_transcript("recording:" + rid))

    def test_old_id_fast_path_and_scan(self):
        """Прежний recording_id в файле — без перебора записей; без него (uuid5) — перебором."""
        self.write_legacy(legacy_data(self.audio, recording_id=uuid.uuid4().hex))
        rid = self.migrate()
        with patch.object(recordings, "migrated_copies", side_effect=AssertionError("перебор")):
            self.assertEqual(recordings.find_migrated(self.store, self.legacy)["recording_id"], rid)
        other = self.write_legacy(legacy_data(self.audio, "другой вариант"), self.root / "meeting.b.transcript.json")
        rid_b = self.migrate(other, tag="b", main=False)
        self.assertEqual(rid_b, IL.recording_id_for(other, IL.sha256_file(other)))
        with patch.object(recordings, "migrated_copies", wraps=recordings.migrated_copies) as scan:
            self.assertEqual(recordings.find_migrated(self.store, other)["recording_id"], rid_b)
        self.assertEqual(scan.call_count, 1)

    def test_not_main_variant_redirects_to_itself(self):
        """Задача делала не основной вариант: переадресация на него же, в статусе — основной."""
        self.write_legacy(legacy_data(self.audio))
        main = self.migrate()
        variant = self.write_legacy(legacy_data(self.audio, "вариант"), self.root / "meeting.v2.transcript.json")
        rid = self.migrate(variant, tag="v2", main=False)
        self.job(variant)
        s = core.job_status(JOB)
        self.assertEqual(s["recording_id"], rid)
        self.assertIs(s["variant_main"], False)
        self.assertEqual(s["main_recording_id"], main)
        self.assertIn("вариант", core.read_transcript(JOB))

    def test_two_copies_picks_the_one_with_same_content(self):
        """Файл перенесён, изменён и перенесён снова: ведёт к копии с тем же содержимым, другая названа."""
        self.write_legacy(legacy_data(self.audio, "первая версия"))
        first = self.migrate()
        self.write_legacy(legacy_data(self.audio, "вторая версия"))
        second = self.migrate(tag="again")
        self.assertNotEqual(first, second)
        self.job()
        s = core.job_status(JOB)
        self.assertEqual(s["recording_id"], second)
        self.assertIn("recording:" + first, s["migration_hint"])

    def test_path_spelling_variants(self):
        """Путь в задаче записан иначе (косые черты, `..`, на Windows — регистр): та же запись."""
        self.write_legacy(legacy_data(self.audio, recording_id=uuid.uuid4().hex))
        rid = self.migrate()
        (self.root / "sub").mkdir()
        variants = [str(self.legacy).replace("\\", "/"), str(self.root / "sub" / ".." / self.legacy.name)]
        if os.name == "nt":
            variants.append(str(self.legacy).swapcase())
        for variant in variants:
            with self.subTest(variant=variant):
                self.job(variant)
                s = core.job_status(JOB)
                self.assertEqual(s.get("recording_id"), rid)
                self.assertEqual(s["legacy_json"], variant)


class Refusals(Fixture):
    def refused(self, pattern):
        with self.assertRaisesRegex(storage.Conflict, pattern):
            core.read_transcript(JOB)

    def test_not_migrated(self):
        self.write_legacy(legacy_data(self.audio))
        self.job()
        ops = self.operations()
        s = core.job_status(JOB)
        self.assertEqual(s["json"], str(self.legacy))
        self.assertIn("не перенесена в базу", s["migration_hint"])
        self.refused("not registered in the active store; эта стенограмма не перенесена в базу")
        self.assertEqual(self.operations(), ops)

    def test_changed_after_migration(self):
        self.write_legacy(legacy_data(self.audio))
        rid = self.migrate()
        self.write_legacy(legacy_data(self.audio, "правка после переноса"))
        self.job()
        self.assertNotIn("recording_id", core.job_status(JOB))
        self.refused(f"изменился после переноса в базу; запись в базе — recording:{rid}")

    def test_file_missing(self):
        self.write_legacy(legacy_data(self.audio))
        rid = self.migrate()
        self.job()
        self.legacy.unlink()
        self.assertIn(f"старого файла нет на диске; записи в базе по этому пути — recording:{rid}",
                      core.job_status(JOB)["migration_hint"])

    def test_file_missing_and_nothing_at_that_path(self):
        """Ф209-1: старого файла нет, по его пути ничего не переносили (стенограмму перенесли из другой папки).

        Записи не угадываются; отказ не советует переносить файл, которого нет, и не зовёт к legacy_files.
        """
        moved = self.root / "moved"
        moved.mkdir()
        other = self.write_legacy(legacy_data(self.audio, recording_id=uuid.uuid4().hex), moved / self.legacy.name)
        self.migrate(other)
        self.job()
        self.assertFalse(self.legacy.exists())
        ops = self.operations()
        s = core.job_status(JOB)
        self.assertEqual(s["json"], str(self.legacy))
        self.assertNotIn("recording_id", s)
        hint = s["migration_hint"]
        self.assertIn("старого файла нет на диске и записей по этому пути в базе нет; если стенограмму перенесли "
                      "из другой папки, найдите запись в списке записей и укажите recording:<id>", hint)
        for advice in ("import_legacy", "legacy_files", "перенесите"):
            self.assertNotIn(advice, hint)
        self.assertEqual(recordings.find_migrated(self.store, self.legacy)["state"], "not_found")
        self.refused("not registered in the active store; старого файла нет на диске и записей по этому пути")
        self.assertEqual(self.operations(), ops)

    def test_same_recording_id_other_path(self):
        """Копия файла с тем же recording_id и тем же содержимым, но по другому пути: не та запись."""
        self.write_legacy(legacy_data(self.audio, recording_id=uuid.uuid4().hex))
        self.migrate()
        copy = self.root / "copy.transcript.json"
        copy.write_bytes(self.legacy.read_bytes())
        self.assertEqual(IL.sha256_file(copy), IL.sha256_file(self.legacy))
        self.job(copy)
        s = core.job_status(JOB)
        self.assertEqual(s["json"], str(copy))
        self.assertNotIn("recording_id", s)
        self.assertEqual(recordings.find_migrated(self.store, copy)["state"], "not_migrated")

    def test_two_copies_with_same_content(self):
        self.write_legacy(legacy_data(self.audio))
        first = self.migrate()
        second = self.migrate(tag="copy", recording_id=str(uuid.uuid4()))
        self.job()
        hint = core.job_status(JOB)["migration_hint"]
        self.assertIn("несколько записей", hint)
        self.assertIn("recording:" + first, hint)
        self.assertIn("recording:" + second, hint)

    def test_direct_path_gets_hint_not_substitution(self):
        """Прямой путь к старой стенограмме: прежний класс отказа, в тексте — перенесённая запись."""
        self.write_legacy(legacy_data(self.audio))
        rid = self.migrate()
        with self.assertRaisesRegex(storage.Conflict, f"перенесена в базу: recording:{rid} — используйте его"):
            transcript.load(self.legacy)


class Unchanged(Fixture):
    def test_job_made_in_store_is_not_searched(self):
        self.write_legacy(legacy_data(self.audio))
        rid = self.migrate()
        doc = Path(self.doc_of(rid))
        self.job(doc)
        with patch.object(recordings, "find_migrated", side_effect=AssertionError("поиск переноса")):
            s = core.job_status(JOB)
        self.assertEqual(s["json"], str(doc))
        self.assertNotIn("redirected_from", s)

    def test_inactive_store_keeps_refusal(self):
        self.write_legacy(legacy_data(self.audio))
        self.migrate()
        self.job()
        self.store.db.execute("UPDATE restore_guard SET locked=1 WHERE id=1")
        self.store.db.commit()
        s = core.job_status(JOB)
        self.assertEqual(s["json"], str(self.legacy))
        self.assertNotIn("recording_id", s)
        with self.assertRaises(storage.Conflict):
            core.read_transcript(JOB)

    def test_list_jobs_reads_raw_status(self):
        self.write_legacy(legacy_data(self.audio))
        self.migrate()
        self.job()
        with patch.object(recordings, "open_store", side_effect=AssertionError("база открыта")):
            rows = core.list_jobs()
        self.assertEqual([r["job_id"] for r in rows], [JOB])


if __name__ == "__main__":
    unittest.main()
