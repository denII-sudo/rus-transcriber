"""Перенос старого файлового формата в SQLite: только синтетические записи во временной папке."""
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import import_legacy as IL  # noqa: E402
import legacy_sample_loader as samples  # noqa: E402
import paths  # noqa: E402
import recordings  # noqa: E402
import storage  # noqa: E402
import voices  # noqa: E402


def word(w, s, e):
    return {"w": w, "s": s, "e": e, "c": 0.9}


def legacy_transcript(audio):
    """Старый JSON как у прежнего worker: без ID слов, сомнения ссылаются на seg/w1/w2."""
    return {
        "file": str(audio), "duration": "00:00:20", "speakers": ["Анна", "Борис"],
        "speaker_names": {"Спикер 1": "Анна", "Спикер 2": "Борис"}, "names_source": "rename", "names_confirmed": True,
        "timing": {"total_sec": 1.0}, "speaker_labels": {"0": "Спикер 1"}, "group": "demo",
        "quality": {}, "edits": [{"id": "e1", "t": 1.0, "by": "claude", "before": "а", "after": "б", "note": ""},
                                {"id": "e2", "t": 2.0, "by": "human", "before": "в", "after": "г", "note": ""}],
        "segments": [
            {"start": 0.0, "end": 4.0, "speaker": "Анна", "text": "договор аренды подписан сегодня", "label": "0",
             "words": [word("договор", 0.0, 0.5), word("аренды", 0.6, 1.0), word("подписан", 1.1, 1.6),
                       word("сегодня", 1.7, 2.2)]},
            {"start": 5.0, "end": 9.0, "speaker": "Борис", "text": "срок три года", "label": "1",
             "words": [word("срок", 5.0, 5.4), word("три", 5.5, 5.8), word("года", 5.9, 6.3)]},
            {"start": 10.0, "end": 12.0, "speaker": "Борис", "text": "текст без слов", "label": "1"},
            {"start": 13.0, "end": 16.0, "speaker": "Анна", "text": "да да", "label": "0",
             "words": [word("да", 13.0, 13.2), word("да", 13.3, 13.5)]},
        ],
        "review": [
            {"id": "r-ok", "seg": 0, "w1": 1, "w2": 3, "t": 0.6, "t1": 0.6, "t2": 1.6, "speaker": "Анна",
             "text": "аренды подписан", "reasons": ["низкая уверенность"], "status": "resolved"},
            {"id": "r-time", "seg": 7, "w1": 0, "w2": 1, "t": 5.5, "t1": 5.4, "t2": 5.9, "speaker": "Борис",
             "text": "три", "reasons": ["расхождение"]},
            {"id": "r-ambig", "seg": 3, "w1": 5, "w2": 6, "t": 13.0, "t1": 12.9, "t2": 13.6, "speaker": "Анна",
             "text": "да", "reasons": ["расхождение"], "status": "resolved"},
            {"id": "r-edited", "seg": 9, "w1": 0, "w2": 1, "t": 6.0, "t1": 5.85, "t2": 6.35, "speaker": "Борис",
             "text": "годы", "reasons": ["расхождение"], "status": "resolved"},
            {"id": "r-gap", "t": 10.0, "t1": 10.0, "t2": 12.0, "speaker": "Борис", "text": "пауза", "reasons": ["пропуск"]},
        ],
        "review_done": False,
    }


def legacy_conspect(json_path):
    return {"version": 1, "name": "конспект", "job_json": str(json_path),
            "template": "Итоги: {{q:1}} и {{q:2}}.\n",
            "quotes": {"1": {"author": "договор аренды подписан", "speaker": "Анна", "parts": [], "base": "",
                             "base_speaker": "", "last": "", "last_speaker": ""},
                       "2": {"author": "срок три года", "speaker": None, "parts": [], "base": "",
                             "base_speaker": "", "last": "", "last_speaker": ""}}}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="import-legacy-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = self.root / "встреча.mp3"
        self.audio.write_bytes(b"synthetic audio")
        self.side = self.root / "встреча.rus-transcriber"
        self.side.mkdir()
        self.tj = self.side / "встреча.v2.transcript.json"
        self.tj.write_text(json.dumps(legacy_transcript(self.audio), ensure_ascii=False), encoding="utf-8")
        (self.side / "встреча.v2.transcript.md").write_text("старый md\n", encoding="utf-8")
        self.cj = self.side / "встреча.v2.конспект.conspect.json"
        self.cj.write_text(json.dumps(legacy_conspect(self.tj), ensure_ascii=False), encoding="utf-8")
        self.pack = self.side / "встреча.v2.review"
        self.pack.mkdir()
        # согласованный пакет старого формата: answers.json и manifest.json одной версии (так пишет review_pack.build)
        (self.pack / "answers.json").write_text(json.dumps({
            "version": "q7", "answers": {"r-time": {"id": "r-time", "version": "q7", "source": "local",
                                                    "ts": 1794129404.7, "kind": "option", "option": 1},
                                         "missing": {"id": "missing", "kind": "text", "text": "x"}},
            "conflicts": [], "rejected": [], "applied_answers": {"r-time": True}, "applied_at": "2026-11-08T16:41:40"},
            ensure_ascii=False), encoding="utf-8")
        (self.pack / "manifest.json").write_text(json.dumps({
            "version": "q7", "items": [{"id": "r-time", "question": "Кто говорит?", "text": "три", "t1": 5.4, "t2": 5.9,
                                        "options": [{"label": "Анна"}, {"label": "Борис"}]}]},
            ensure_ascii=False), encoding="utf-8")

    def plan(self):
        return {"version": 1, "recordings": [{
            "audio": str(self.audio),
            "metadata": {"title": "встреча", "recorded_at": "2026-11-06T20:32:00+07:00", "recorded_timezone": "+07:00"},
            "field_sources": {"title": "имя файла (предположение)", "recorded_at": "имя файла, указание владельца"},
            "variants": [{"transcript": str(self.tj), "tag": "v2", "order": 1, "main": True,
                          "main_note": "основная версия вычислена по истории обработки",
                          "conspects": [str(self.cj)], "review_packs": [str(self.pack)]}]}]}

    def hashes(self):
        return {p.relative_to(self.side).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.side.rglob("*") if p.is_file()}


class ConversionTests(Fixture):
    def snapshots(self):
        legacy = json.loads(self.tj.read_text(encoding="utf-8"))
        rid = IL.recording_id_for(self.tj, "x")
        return IL.build_snapshots(legacy, recording_id=rid, metadata={"title": "встреча", "audio_available": True},
                                  field_sources={"title": "имя файла (предположение)"},
                                  variant={"tag": "v2", "main": True, "main_note": "по истории"},
                                  source={"path": str(self.tj), "sha256": "x", "size": 1},
                                  answers_packs=[{"name": "p", "answers": {"r-time": {"version": "q7", "kind": "option"}}}])

    def test_review_references_become_word_ids(self):
        _, transcript, review, stats = self.snapshots()
        items = {r["id"]: r for r in review["review"]}
        words = [w for s in transcript["segments"] for w in s.get("words", [])]
        by_id = {w["word_id"]: w["w"] for w in words}
        self.assertEqual([by_id[i] for i in items["r-ok"]["word_ids"]], ["аренды", "подписан"])
        self.assertNotIn("migration", items["r-ok"])
        # устаревшая ссылка: единственное совпадение по времени и тексту
        self.assertEqual([by_id[i] for i in items["r-time"]["word_ids"]], ["три"])
        self.assertEqual((items["r-time"]["seg"], items["r-time"]["w1"], items["r-time"]["w2"]), (1, 1, 2))
        self.assertEqual(items["r-time"]["migration"]["anchor"], "time")
        self.assertEqual(items["r-time"]["migration"]["legacy_ref"], {"seg": 7, "w1": 0, "w2": 1})
        # два одинаковых «да» — неоднозначно: ссылка снята, решение остаётся за человеком
        self.assertNotIn("word_ids", items["r-ambig"])
        self.assertNotIn("seg", items["r-ambig"])
        self.assertEqual(items["r-ambig"]["migration"]["anchor"], "not_found")
        self.assertEqual(items["r-ambig"]["status"], "resolved")
        self.assertNotIn("migration", items["r-gap"])
        # текст правили после разбора: место однозначно по времени, привязано с пометкой
        self.assertEqual([by_id[i] for i in items["r-edited"]["word_ids"]], ["года"])
        self.assertFalse(items["r-edited"]["migration"]["text_same"])
        self.assertIn("текст места менялся", items["r-edited"]["migration"]["note"])
        self.assertTrue(items["r-time"]["migration"]["text_same"])
        self.assertEqual(stats["review"], {"по номерам": 1, "по времени": 1, "по времени, текст менялся": 1,
                                           "место не найдено": 1, "только по времени": 1})

    def test_ids_metadata_provenance_and_validation(self):
        rid, transcript, review, _ = self.snapshots()
        self.assertTrue(all(s.get("segment_id") for s in transcript["segments"]))
        self.assertNotIn("words", transcript["segments"][2])
        self.assertEqual(transcript["title"], "встреча")
        self.assertTrue(transcript["audio_available"])
        mig = transcript["quality"]["migration"]
        self.assertEqual(mig["field_sources"]["title"], "имя файла (предположение)")
        self.assertEqual(mig["variant"]["main_note"], "по истории")
        self.assertEqual(transcript["quality"]["runtime"]["group"], "demo")
        self.assertEqual([e["by"] for e in transcript["edits"]], ["claude", "human"])
        storage.validate_pair(transcript, review)

    def test_existing_metadata_is_not_overwritten(self):
        legacy = json.loads(self.tj.read_text(encoding="utf-8"))
        legacy["title"] = "из стенограммы"
        _, transcript, _, _ = IL.build_snapshots(legacy, recording_id=IL.recording_id_for(self.tj, "x"),
                                                 metadata={"title": "из имени"}, field_sources={}, variant={},
                                                 source={})
        self.assertEqual(transcript["title"], "из стенограммы")
        self.assertNotIn("title", transcript["quality"]["migration"]["field_sources"])

    def test_answers_without_current_pack_are_archived_not_pending(self):
        """В-1 «Перенос целиком»: ответы без текущего пакета (нет answers.json и manifest.json одной версии) не
        становятся ответами разбора — иначе пересборка ждала бы их «Применить» (М-1); они хранятся как есть."""
        _, _, review, stats = self.snapshots()
        self.assertEqual(review["answers"], {})
        self.assertIsNone(review.get("question_version"))
        files = review["undo"]["legacy_review_packs"]["files"]
        self.assertEqual(files[0]["file"], "p")
        self.assertEqual(files[0]["answers"]["r-time"], {"version": "q7", "kind": "option"})
        self.assertEqual(stats["pack_state"]["state"], "archived_only")
        review2 = {"review": [{"id": "a"}], "answers": {}}
        self.assertEqual(IL.import_answers(review2, {"a": {}, "b": {}}, "p"), (1, 1, 0))
        review3 = {"review": [{"id": "a"}], "answers": {}}
        IL.import_answers(review3, {"a": {"version": "q7"}}, "p", keep_version=True)
        self.assertEqual(review3["answers"]["a"]["version"], "q7")
        self.assertNotIn("legacy_version", review3["answers"]["a"])

    def test_conspect_markdown_restored_from_placeholders(self):
        md = IL.legacy_conspect_markdown(legacy_conspect(self.tj))
        self.assertEqual(md, "Итоги: Анна: «договор аренды подписан» и «срок три года».\n")
        broken = legacy_conspect(self.tj)
        broken["template"] += "{{q:9}}"
        with self.assertRaises(ValueError):
            IL.legacy_conspect_markdown(broken)

    def test_file_saved_by_current_program_keeps_ids_and_runtime_fields(self):
        """После переноса подтверждений голосов старый файл сохранён штатно: есть recording_id и ID слов.
        Файл — образец (Л-3.1): его один раз сохранила программа поверх `legacy_transcript`, образец заморожен."""
        samples.load("saved", self.side, self.audio)
        saved = json.loads(self.tj.read_text(encoding="utf-8"))
        self.assertIn("recording_id", saved)
        ids = [w["word_id"] for s in saved["segments"] for w in s.get("words", [])]
        _, transcript_snap, _, _ = IL.build_snapshots(saved, recording_id=IL.recording_id_for(self.tj, "y"),
                                                      metadata={}, field_sources={}, variant={}, source={"path": "p"})
        self.assertEqual([w["word_id"] for s in transcript_snap["segments"] for w in s.get("words", [])], ids)
        runtime = transcript_snap["quality"]["migration"]["source"]["legacy_runtime"]
        self.assertEqual(runtime["recording_id"], saved["recording_id"])
        self.assertEqual(runtime["source_revision"], saved["source_revision"])

    def test_archived_answers_stay_out_of_current_answers(self):
        (self.pack / "answers.20261108101950-aaaaaa.json").write_text(json.dumps({
            "version": "old", "answers": {"r-time": {"id": "r-time", "version": "q1", "kind": "option", "option": 0},
                                           "r-ok": {"id": "r-ok", "version": "q1", "kind": "option", "option": 0}}},
            ensure_ascii=False), encoding="utf-8")
        legacy, source, packs, _, manifests = IL._read_variant(self.plan()["recordings"][0],
                                                                self.plan()["recordings"][0]["variants"][0])
        self.assertEqual([p["name"] for p in packs],
                         ["встреча.v2.review", "встреча.v2.review/answers.20261108101950-aaaaaa.json"])
        _, _, review, stats = IL.build_snapshots(legacy, recording_id=IL.recording_id_for(self.tj, "z"), metadata={},
                                                 field_sources={}, variant={}, source=source, answers_packs=packs,
                                                 manifests=manifests)
        # ответ текущего пакета — с версией пакета; ответ архива — не в ответах разбора, а в архиве как есть
        self.assertEqual(review["answers"]["r-time"]["version"], "q7")
        self.assertEqual(review["question_version"], "q7")
        self.assertNotIn("r-ok", review["answers"])
        files = review["undo"]["legacy_review_packs"]["files"]
        self.assertEqual(files[0]["file"], "встреча.v2.review/answers.20261108101950-aaaaaa.json")
        self.assertEqual(files[0]["answers"]["r-ok"]["version"], "q1")

    def test_human_page_answer_keeps_actor_without_legacy_label(self):
        review = {"review": [{"id": "a"}], "answers": {}}
        IL.import_answers(review, {"a": {"id": "a", "source": "local", "actor_kind": "human", "ts": "2026-11-21T10:00:00"}}, "p")
        self.assertEqual(review["answers"]["a"]["actor_kind"], "human")
        self.assertNotIn("legacy_author_label", review["answers"]["a"])

    def test_recording_id_is_stable(self):
        self.assertEqual(IL.recording_id_for(self.tj, "a"), IL.recording_id_for(self.tj, "a"))
        self.assertNotEqual(IL.recording_id_for(self.tj, "a"), IL.recording_id_for(self.tj, "b"))


class PlanTests(Fixture):
    def write(self, plan):
        p = self.root / "plan.json"
        p.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        return p

    def test_valid_plan_loads(self):
        self.assertEqual(len(IL.load_plan(self.write(self.plan()))["recordings"]), 1)

    def test_plan_errors(self):
        cases = []
        p = self.plan(); p["recordings"][0]["variants"][0]["main"] = False; cases.append(p)
        p = self.plan(); p["recordings"][0]["variants"].append(dict(p["recordings"][0]["variants"][0])); cases.append(p)
        p = self.plan(); p["recordings"][0]["metadata"]["speakers"] = []; cases.append(p)
        p = self.plan(); p["version"] = 2; cases.append(p)
        for plan in cases:
            with self.subTest(plan=plan["version"]), self.assertRaises(IL.PlanError):
                IL.load_plan(self.write(plan))


class CheckAndApplyTests(Fixture):
    def setUp(self):
        super().setUp()
        self.db = self.root / "data" / "state.sqlite3"
        self.db.parent.mkdir()
        patcher = mock.patch.object(paths, "STATE_DB", self.db)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_check_writes_nothing(self):
        before = self.hashes()
        report = IL.run(self.plan(), "check", ffprobe="нет-ffprobe")
        self.assertEqual(report[0]["status"], "ok", report)
        self.assertEqual(report[0]["conspects"][0]["linked"], 2)
        self.assertEqual(self.hashes(), before)
        self.assertFalse(self.db.exists())
        self.assertFalse((self.side / IL.DOC_SUBDIR).exists())

    def test_apply_creates_records_and_documents_without_touching_legacy_files(self):
        recordings.initialize_store(self.db, display_name="Владелец").close()
        before = self.hashes()
        report = IL.run(self.plan(), "apply", ffprobe="нет-ffprobe")
        entry = report[0]
        self.assertEqual(entry["status"], "imported", entry)
        for name, digest in before.items():
            self.assertEqual(self.hashes()[name], digest, name)
        for target in entry["documents"].values():
            self.assertTrue(Path(target).is_file(), target)
            self.assertEqual(Path(target).parent.name, IL.DOC_SUBDIR)
        self.assertEqual(entry["conspects_saved"][0]["unlinked"], 0)
        store = recordings.open_store(self.db)
        try:
            current = store.get_current(entry["recording_id"])
            self.assertEqual(current["transcript"]["payload"]["recorded_at"], "2026-11-06T20:32:00+07:00")
            self.assertEqual(current["transcript"]["payload"]["audio_sha256"],
                             hashlib.sha256(b"synthetic audio").hexdigest())
            answer = current["review"]["payload"]["answers"]["r-time"]
            self.assertEqual(answer["origin"], "legacy_review_pack:встреча.v2.review")
            # формат программы (review_pack._now): UTC «ГГГГ-ММ-ДДTЧЧ:ММ:СС», без дубля исходного числа
            self.assertRegex(answer["ts"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")
            self.assertNotIn("legacy_ts", answer)
            self.assertLessEqual(answer["ts"], "2026-11-08T16:41:40")   # ответ раньше применения пакета
            packs = current["transcript"]["payload"]["quality"]["migration"]["review_packs"]
            self.assertEqual(packs[0]["ts_converted"], 1)
            self.assertNotIn("actor_kind", answer)
            self.assertEqual(len(store.list_conspects(entry["recording_id"])), 1)
            op = IL.uuid.uuid5(IL.NAMESPACE, "legacy-import:" + entry["recording_id"])
            ctx = store.db.execute("SELECT context_json, author_user_id, initiator_user_id FROM operations "
                                   "WHERE operation_id=?", (str(op),)).fetchone()
            self.assertEqual(json.loads(ctx[0])["actor_kind"], "system")
            self.assertIsNone(ctx[1])          # автор не человек: перенос сделала программа
            self.assertIsNotNone(ctx[2])       # инициатор — владелец установки
        finally:
            store.close()
        again = IL.run(self.plan(), "apply", ffprobe="нет-ffprobe")
        self.assertEqual(again[0]["status"], "skipped")

    def test_missing_audio_is_reported_not_invented(self):
        self.audio.unlink()
        report = IL.run(self.plan(), "check", ffprobe="нет-ffprobe")
        self.assertEqual(report[0]["status"], "ok", report)
        facts = IL.audio_facts(self.audio)
        self.assertEqual(facts, {"audio_available": False})


class A1Tests(Fixture):
    """recording_id, вопросы карточек, сверка MD, конфликт и комплект отчётов."""

    def setUp(self):
        super().setUp()
        self.db = self.root / "data" / "state.sqlite3"
        self.db.parent.mkdir()
        p = mock.patch.object(paths, "STATE_DB", self.db)
        p.start()
        self.addCleanup(p.stop)

    def test_recording_id_rules(self):
        old = "0123456789abcdef0123456789abcdef"
        self.assertEqual(IL.recording_id_from({"recording_id": old}, self.tj, "x"),
                         ("01234567-89ab-cdef-0123-456789abcdef", "сохранён старый recording_id"))
        rid, rule = IL.recording_id_from({}, self.tj, "x")
        self.assertEqual(rid, IL.recording_id_for(self.tj, "x"))
        self.assertIn("uuid5", rule)
        self.assertEqual(IL.recording_id_from({"recording_id": "не-uuid"}, self.tj, "x")[0], rid)

    def test_questions_from_current_manifest_archive_kept_aside(self):
        legacy = json.loads(self.tj.read_text(encoding="utf-8"))
        manifests = [{"name": "p", "version": "v2", "current": True,
                      "items": [{"id": "r-time", "question": "Кто говорит?", "text": "три", "word_ids": ["нет-такого"]},
                                {"id": "лишний", "question": "?"}]},
                     {"name": "p/manifest.v1.json", "version": "v1", "items": [
                         {"id": "r-time", "question": "старый вопрос"}, {"id": "r-ok", "question": "Так?", "text": "аренды"},
                         {"id": "лишний", "question": "?"}]}]
        packs = [{"name": "p", "answers": {}, "raw": {"version": "v2", "answers": {}}, "current": True}]
        _, _, review, stats = IL.build_snapshots(legacy, recording_id=IL.recording_id_for(self.tj, "q"), metadata={},
                                                 field_sources={}, variant={}, source={}, answers_packs=packs,
                                                 manifests=manifests)
        # вопросы — карточки текущего пакета (как review_pack.build пишет в базу), только к существующим местам
        qs = {q["id"]: q for q in review["questions"]}
        self.assertEqual(set(qs), {"r-time"})
        self.assertEqual(qs["r-time"]["question"], "Кто говорит?")
        self.assertNotIn("word_ids", qs["r-time"])            # ссылка на несуществующее слово отброшена
        self.assertEqual(stats["questions"], 1)
        files = review["undo"]["legacy_review_packs"]["files"]
        self.assertEqual([(f["file"], f["kind"]) for f in files], [("p/manifest.v1.json", "manifest")])
        self.assertEqual([q["id"] for q in files[0]["questions"]], ["r-time", "r-ok", "лишний"])

    def test_md_check(self):
        import transcript
        legacy = json.loads(self.tj.read_text(encoding="utf-8"))
        self.assertFalse(IL.md_check(self.tj, legacy)["same"])
        (self.side / "встреча.v2.transcript.md").write_text(transcript.render_md(legacy), encoding="utf-8")
        self.assertTrue(IL.md_check(self.tj, legacy)["same"])

    def test_same_id_from_changed_source_is_conflict(self):
        legacy = json.loads(self.tj.read_text(encoding="utf-8"))
        legacy["recording_id"] = "0123456789abcdef0123456789abcdef"
        self.tj.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        recordings.initialize_store(self.db, display_name="Владелец").close()
        self.assertEqual(IL.run(self.plan(), "apply", ffprobe="нет")[0]["status"], "imported")
        legacy["segments"][0]["text"] = "изменено после переноса"
        self.tj.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        entry = IL.run(self.plan(), "apply", ffprobe="нет")[0]
        self.assertEqual(entry["status"], "conflict")

    def test_main_guards_data_dir_and_writes_verified_set(self):
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(self.plan(), ensure_ascii=False), encoding="utf-8")
        out = self.root / "reports"
        self.assertEqual(IL.main(["apply", "--plan", str(plan_path), "--out", str(out)]), 2)
        self.assertEqual(IL.main(["apply", "--plan", str(plan_path), "--out", str(out),
                                  "--data-dir", str(self.root / "чужая")]), 2)
        self.assertFalse(self.db.exists())
        recordings.initialize_store(self.db, display_name="Владелец").close()
        with mock.patch.object(IL, "audio_facts", return_value={"audio_available": True}):
            code = IL.main(["apply", "--plan", str(plan_path), "--out", str(out), "--data-dir", str(paths.DATA_DIR)])
        verification = json.loads((out / "verification.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, verification)
        self.assertEqual(verification["status"], "verified_copy")
        self.assertEqual(verification["activation"], "not_performed")
        self.assertTrue(all(verification["checks"].values()), verification["checks"])
        self.assertTrue(all(ok is not False for r in verification["records"] for ok in r["checks"].values()))
        mapping = json.loads((out / "mapping.json").read_text(encoding="utf-8"))
        self.assertEqual(mapping[0]["database_revision"], 1)
        self.assertTrue((out / "rehearsal" / "state.sqlite3").is_file())
        for name in ("inventory.json", "plan.json", "report.json", "report.md"):
            self.assertTrue((out / name).is_file(), name)


class MarkVoicesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="import-legacy-voices-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for name, value in (("GROUPS", root / "groups"), ("JOBS", root / "jobs"), ("DATA_DIR", root)):
            p = mock.patch.object(paths, name, value)
            p.start()
            self.addCleanup(p.stop)
        prof = voices.empty_profile()
        prof["speakers"] = {"Анна": {"parts": {"a.mp3": {"sec": 10.0, "vec": [0.1] * 4, "confirmed": False}}}}
        self.path = paths.voices_path("demo")
        self.path.parent.mkdir(parents=True)
        voices.save_profile(self.path, prof)
        self.plan = {"voice_confirmation": {"demo": {"by": "program", "note": "автоматика программы",
                                                     "decided": "владелец, 20.11.2026"}}}

    def test_preview_changes_nothing_apply_marks_and_backs_up(self):
        before = self.path.read_bytes()
        preview = IL.mark_voices(self.plan)
        self.assertEqual(preview[0]["parts_changed"], 1)
        self.assertEqual(self.path.read_bytes(), before)
        applied = IL.mark_voices(self.plan, apply=True)
        self.assertTrue(applied[0]["applied"])
        self.assertEqual(Path(applied[0]["backup"]).read_bytes(), before)
        part = json.loads(self.path.read_text(encoding="utf-8"))["speakers"]["Анна"]["parts"]["a.mp3"]
        self.assertEqual(part["confirmation"]["by"], "program")
        self.assertEqual(part["vec"], [0.1] * 4)
        self.assertFalse(part["confirmed"])
        self.assertEqual(IL.mark_voices(self.plan, apply=True)[0]["parts_changed"], 0)

    def test_unknown_decision_rejected(self):
        with self.assertRaises(IL.PlanError):
            IL.mark_voices({"voice_confirmation": {"demo": {"by": "кто-то"}}})


if __name__ == "__main__":
    unittest.main()
