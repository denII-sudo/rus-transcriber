"""Перенос П7: состояние пакета карточек целиком (В-1, М-1) и ремонт repair-packs. Только синтетика во временной папке.

Решение владельца по В-1 (26.11.2026): «Перенос целиком». Решения Архитектора (26.11.2026): снимки отмены не
переносятся (Р-1), ответы архивных пакетов — в undo.legacy_review_packs (Р-2), клипы не копируются (Р-3); ответы
закрытых мест без отметки «применён» остаются ждущими и только считаются (Р-4 решит владелец).
"""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import import_legacy as IL  # noqa: E402
import paths  # noqa: E402
import recordings  # noqa: E402
import review_pack  # noqa: E402
import settings  # noqa: E402
import storage  # noqa: E402
from test_import_legacy import legacy_transcript  # noqa: E402

APPLIED_AT = 1794129500              # секунды с 1970 года, как в старом answers.json
BEFORE, AFTER = 1794129400.5, 1794129600.0
VERSION = "20261108101950-abcdef"


def old_pack_import(review, transcript, packs, manifests, recording_id, legacy_runtime=None):
    """Прежний перенос (до В-1): только ответы с legacy_version и упрощённые вопросы — так перенесены записи с М-1."""
    converted = 0
    for pack in packs:
        converted += IL.import_answers(review, pack.get("answers"), pack["name"])[2]
    IL.import_questions(review, transcript, manifests)
    return {"state": "none", "answers_current": 0, "answers_archived": 0, "ts_converted": converted}


def item(aid, t1, t2, text):
    return {"id": aid, "t1": t1, "t2": t2, "text": text, "question": "Проверьте место", "speaker": "Борис",
            "options": [{"label": "Оставить как есть", "decision": {}}, {"label": "Исправить", "decision": {}}]}


class PackFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="import-legacy-packs-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = self.root / "встреча.mp3"
        self.audio.write_bytes(b"synthetic audio")
        self.side = self.root / "встреча.rus-transcriber"
        self.side.mkdir()
        self.tj = self.side / "встреча.v2.transcript.json"
        self.tj.write_text(json.dumps(legacy_transcript(self.audio), ensure_ascii=False), encoding="utf-8")
        self.pack = self.side / "встреча.v2.review"
        (self.pack / "clips").mkdir(parents=True)
        (self.pack / "clips" / "r-time.mp3").write_bytes(b"clip")
        self.write_pack()
        self.db = self.root / "data" / "state.sqlite3"
        self.db.parent.mkdir()
        for target, value in ((paths, ("STATE_DB", self.db)), (paths, ("DATA_DIR", self.db.parent))):
            p = mock.patch.object(target, *value)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(settings, "recording_backend", return_value="sqlite")
        p.start()
        self.addCleanup(p.stop)

    def answers(self):
        """r-ok — применён (старая отметка true), r-ambig — применён (подпись), r-time — сохранён после применения:
        ждёт «Применить» на открытом месте; r-edited — ответ без отметки на закрытом месте (Р-4)."""
        return {
            "r-ok": {"id": "r-ok", "version": VERSION, "source": "local", "ts": BEFORE, "kind": "option", "option": 0},
            "r-ambig": {"id": "r-ambig", "version": VERSION, "source": "cloud", "ts": BEFORE, "kind": "free",
                        "text": "да"},
            "r-time": {"id": "r-time", "version": VERSION, "source": "local", "ts": AFTER, "kind": "option",
                       "option": 1},
            "r-edited": {"id": "r-edited", "version": VERSION, "source": "local", "ts": AFTER, "kind": "option",
                         "option": 0},
            "нет-места": {"id": "нет-места", "version": VERSION, "kind": "option", "option": 0},
        }

    def write_pack(self, **override):
        state = {"version": VERSION, "answers": self.answers(), "conflicts": [{"id": "r-time", "why": "расхождение"}],
                 "rejected": [{"id": "r-time", "why": "версия", "source": "cloud", "ts": "2026-11-08T10:00:00"}],
                 "applied_answers": {"r-ok": True, "r-ambig": ["free", None, "да"]}, "applied_at": APPLIED_AT,
                 "undo_snapshots": {"r-ok": {"format": 2, "segment_patches": [], "review_patches": [],
                                             "edits_before": 0, "after": "x"}},
                 "application_events": [{"id": "r-ok", "version": VERSION, "answer_source": "local",
                                         "answer_ts": BEFORE, "action": "applied", "outcome": "applied",
                                         "at": APPLIED_AT}],
                 "interaction_events": [{"id": "r-time", "version": VERSION, "type": "audio_ended",
                                         "session": "s" * 10, "source": "local", "ts": "2026-11-08T09:00:00"}]}
        state.update(override)
        manifest = {"version": VERSION, "pack_rev": 0, "job_json": str(self.tj), "audio": self.audio.name,
                    "recording_id": "0123456789abcdef0123456789abcdef", "source_revision": None,
                    "clips_dir": "clips", "items": [item("r-ok", 0.6, 1.6, "аренды подписан"),
                                                    item("r-time", 5.4, 5.9, "три"), item("r-ambig", 12.9, 13.6, "да"),
                                                    item("r-edited", 5.85, 6.35, "годы")]}
        (self.pack / "answers.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        (self.pack / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        (self.pack / "answers.old-1.json").write_text(json.dumps({
            "version": "old-1", "answers": {"r-gap": {"id": "r-gap", "version": "old-1", "kind": "skip"}}},
            ensure_ascii=False), encoding="utf-8")
        (self.pack / "manifest.old-1.json").write_text(json.dumps({
            "version": "old-1", "items": [item("r-gap", 10.0, 12.0, "пауза")]}, ensure_ascii=False), encoding="utf-8")

    def plan(self):
        return {"version": 1, "recordings": [{
            "audio": str(self.audio), "metadata": {"title": "встреча"},
            "variants": [{"transcript": str(self.tj), "tag": "v2", "order": 1, "main": True,
                          "review_packs": [str(self.pack)]}]}]}

    def write_plan(self):
        p = self.root / "plan.json"
        p.write_text(json.dumps(self.plan(), ensure_ascii=False), encoding="utf-8")
        return p

    def legacy_hashes(self):
        """Старые файлы рядом с аудио: аудио, стенограмма, папка разбора с клипами и архивами (документы v4, их
        история и замок штатно создаёт запись документов — это не старые файлы)."""
        files = [self.audio, self.tj] + sorted(p for p in self.pack.rglob("*") if p.is_file())
        return {p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

    def db_hashes(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.db.parent.glob("state.sqlite3*") if p.is_file()}

    def init_db(self):
        recordings.initialize_store(self.db, display_name="Владелец").close()

    def apply(self, old=False):
        with mock.patch.object(IL, "audio_facts", return_value={"audio_available": True}):
            if old:
                with mock.patch.object(IL, "pack_state_from_legacy", old_pack_import):
                    return IL.run(self.plan(), "apply")[0]
            return IL.run(self.plan(), "apply")[0]

    def current(self, rid):
        store = recordings.open_store(self.db)
        try:
            return store.get_current(rid)
        finally:
            store.close()


class TransferTests(PackFixture):
    def test_apply_transfers_pack_state_whole(self):
        self.init_db()
        before = self.legacy_hashes()
        entry = self.apply()
        self.assertEqual(entry["status"], "imported", entry)
        self.assertEqual(self.legacy_hashes(), before)          # старые файлы, клипы и архивы не меняются
        rid = entry["recording_id"]
        state = recordings.read_review_pack_state(entry["documents"]["json"])
        man = state["manifest"]
        self.assertEqual(state["version"], VERSION)
        self.assertEqual(man["recording_id"], rid)
        self.assertEqual(man["legacy_recording_id"], "0123456789abcdef0123456789abcdef")
        self.assertEqual(man["source_revision"], 0)               # ревизия пакета неизвестна — не выдаётся за 1
        self.assertEqual(man["job_json"], entry["documents"]["json"])
        self.assertEqual(man["legacy_job_json"], str(self.tj))
        self.assertEqual(man["clips_dir"], "clips")               # Р-3: клипы остаются в старой папке
        self.assertFalse((Path(entry["documents"]["json"]).parent / "встреча.v2.review" / "clips").exists())
        self.assertEqual(set(state["answers"]), {"r-ok", "r-ambig", "r-time", "r-edited"})
        self.assertEqual(state["answers"]["r-ok"]["version"], VERSION)
        self.assertRegex(state["answers"]["r-ok"]["ts"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")
        self.assertEqual(state["applied_at"], IL._utc_text(APPLIED_AT))
        self.assertEqual(state["applied_answers"], {"r-ok": ["option", 0, None], "r-ambig": ["free", None, "да"]})
        self.assertEqual(state["undo_snapshots"], {})              # Р-1
        self.assertEqual(state["rejected"][0]["why"], "версия")
        self.assertEqual(state["conflicts"][0]["why"], "расхождение")
        self.assertEqual(state["application_events"][0]["answer_ts"], state["answers"]["r-ok"]["ts"])
        self.assertEqual([e["type"] for e in state["interaction_events"]], ["audio_ended"])
        states = review_pack.answer_states_of(state, man)
        self.assertEqual({k: v["state"] for k, v in states.items()},
                         {"r-ok": "applied", "r-ambig": "applied", "r-time": "saved", "r-edited": "already_decided"})
        # Р-4 (решение владельца): ответ закрытого места без отметки — событие переноса «место уже решено»
        decided = [e for e in state["application_events"] if e.get("action") == review_pack.LEGACY_DECIDED_ACTION]
        self.assertEqual([(e["id"], e["outcome"]) for e in decided], [("r-edited", "already_decided")])
        self.assertNotIn("r-edited", state["applied_answers"])
        self.assertIs(states["r-ok"]["undo"], False)              # снимка нет — «Отменить нельзя», честно
        self.assertEqual(review_pack.heard_ids(state, man), ["r-time"])   # переход 10.6 по audio_ended
        review = self.current(rid)["review"]["payload"]
        self.assertEqual([q["id"] for q in review["questions"]], ["r-ok", "r-time", "r-ambig", "r-edited"])
        files = review["undo"]["legacy_review_packs"]["files"]
        self.assertEqual({(f["file"], f["kind"]) for f in files},
                         {("встреча.v2.review/answers.old-1.json", "answers"),
                          ("встреча.v2.review/manifest.old-1.json", "manifest")})
        ps = self.current(rid)["transcript"]["payload"]["quality"]["migration"]["pack_state"]
        self.assertEqual((ps["state"], ps["answers_current"], ps["answers_archived"], ps["applied"]), ("full", 4, 1, 2))
        self.assertEqual(ps["applied_marks"], {"kept": 1, "from_true": 1, "dropped": 0, "from_applied_at": 0})
        self.assertEqual((ps["waiting"], ps["waiting_closed_place"], ps["undo_snapshots_dropped"]), (1, 0, 1))
        self.assertEqual(ps["already_decided_marked"], 1)

    def test_rebuild_guard_counts_only_current_answers(self):
        """М-1: архивный ответ не ждёт «Применить»; ждут только ответы текущего пакета — с номером карточки."""
        self.init_db()
        entry = self.apply()
        with self.assertRaises(review_pack.RebuildRefused) as caught:
            review_pack.check_rebuild(entry["documents"]["json"])
        # ответ закрытого места помечен «место уже решено» (Р-4) и пересборку не блокирует
        self.assertEqual([(w["card"], w["id"]) for w in caught.exception.waiting], [(2, "r-time")])
        self.assertNotIn("None", str(caught.exception))

    def test_without_waiting_answers_rebuild_passes_and_apply_runs(self):
        answers = self.answers()
        for aid in ("r-time", "r-edited"):
            answers.pop(aid)
        self.write_pack(answers=answers)
        self.init_db()
        entry = self.apply()
        self.assertEqual(review_pack.check_rebuild(entry["documents"]["json"])["waiting"], [])
        import core
        report = core.apply_answers("recording:" + entry["recording_id"])     # было TypeError (М-1)
        self.assertIsInstance(report, dict)

    def test_closed_place_answer_marked_already_decided_does_not_block_apply(self):
        """Р-4: без ответа открытого места «Применить» проходит; ответ закрытого места не применяется (исход
        «место уже решено» из пометки переноса), стенограмма не меняется."""
        answers = self.answers()
        answers.pop("r-time")
        self.write_pack(answers=answers)
        self.init_db()
        entry = self.apply()
        import core
        before = self.current(entry["recording_id"])["transcript"]["payload"]
        report = core.apply_answers("recording:" + entry["recording_id"])
        self.assertEqual({o["id"]: o["outcome"] for o in report["outcomes"]}["r-edited"], "already_decided")
        self.assertNotIn("r-edited", recordings.read_review_pack_state(entry["documents"]["json"])["applied_answers"])
        self.assertEqual(self.current(entry["recording_id"])["transcript"]["payload"], before)
        states = review_pack.answer_states(entry["documents"]["json"])
        self.assertEqual(states["r-edited"]["state"], "already_decided")
        self.assertEqual(review_pack.check_rebuild(entry["documents"]["json"])["waiting"], [])

    def test_waiting_answer_without_author_is_refused(self):
        """Ждущий старый ответ без автора не применяется вслепую: штатная защита «Применить» требует
        подтверждения (Ф207-1). Сверку ревизий проверяют тесты карточек настоящего вида (Ф207-5,
        test_import_legacy_behavior.RealCardsApplyTests). Стенограмма не меняется."""
        self.init_db()
        entry = self.apply()
        import core
        before = self.current(entry["recording_id"])["transcript"]["payload"]["segments"]
        with self.assertRaisesRegex(core.CoreError, "неизвестного происхождения"):
            core.apply_answers("recording:" + entry["recording_id"])
        self.assertEqual(self.current(entry["recording_id"])["transcript"]["payload"]["segments"], before)
        state = recordings.read_review_pack_state(entry["documents"]["json"])
        self.assertNotIn("r-time", state["applied_answers"])

    def test_source_revision_mapping(self):
        self.assertEqual(IL._db_revision(5, 5), 1)
        self.assertEqual(IL._db_revision(4, 5), 0)
        self.assertEqual(IL._db_revision(None, None), 0)
        legacy = json.loads(self.tj.read_text(encoding="utf-8"))
        legacy["source_revision"] = 5
        packs, manifests = IL._read_pack_files([self.pack])
        manifests[0]["source_revision"] = 5
        manifests[0]["items"][1]["source_revision"] = 4
        _, _, review, _ = IL.build_snapshots(legacy, recording_id=IL.recording_id_for(self.tj, "s"), metadata={},
                                             field_sources={}, variant={}, source={}, answers_packs=packs,
                                             manifests=manifests)
        man = review["undo"]["review_pack"]["manifest"]
        self.assertEqual((man["source_revision"], man["legacy_source_revision"]), (1, 5))
        self.assertEqual((man["items"][1]["source_revision"], man["items"][1]["legacy_source_revision"]), (0, 4))

    def test_old_formats_applied_at_rule_heard_and_event_cap(self):
        """Пакет до отметок applied_answers: применённые — ответы не позже applied_at; heard и события — как есть."""
        heard = {"": [[5.0, 6.0]]}
        events = [{"id": "r-time", "version": VERSION, "type": "card_visible", "session": "s" * 10, "ts": "t"}] * 2005
        packs, manifests = IL._read_pack_files([self.pack])
        raw = packs[0]["raw"]
        raw.pop("applied_answers")
        raw.update(heard=heard, interaction_events=events, interaction_events_dropped=3)
        legacy = json.loads(self.tj.read_text(encoding="utf-8"))
        _, _, review, stats = IL.build_snapshots(legacy, recording_id=IL.recording_id_for(self.tj, "f"), metadata={},
                                                 field_sources={}, variant={}, source={}, answers_packs=packs,
                                                 manifests=manifests)
        pack = review["undo"]["review_pack"]
        self.assertEqual(set(pack["applied_answers"]), {"r-ok", "r-ambig"})      # ts не позже applied_at
        self.assertEqual(stats["pack_state"]["applied_marks"]["from_applied_at"], 2)
        self.assertEqual(pack["heard"], heard)
        self.assertEqual(len(review["interaction_events"]), IL.MAX_INTERACTION_EVENTS)
        self.assertEqual(review["interaction_events_evicted"], 3 + 5)

    def test_true_mark_after_applied_at_is_not_invented(self):
        answers = self.answers()
        self.write_pack(answers=answers, applied_answers={"r-time": True})     # ответ позже применения пакета
        packs, manifests = IL._read_pack_files([self.pack])
        legacy = json.loads(self.tj.read_text(encoding="utf-8"))
        _, _, review, stats = IL.build_snapshots(legacy, recording_id=IL.recording_id_for(self.tj, "t"), metadata={},
                                                 field_sources={}, variant={}, source={}, answers_packs=packs,
                                                 manifests=manifests)
        self.assertEqual(review["undo"]["review_pack"]["applied_answers"], {})
        self.assertEqual(stats["pack_state"]["applied_marks"]["dropped"], 1)

    def test_applied_at_rule_compares_fractions_of_second(self):
        """Ф207-4: правило «не позже applied_at» — по исходным секундам с долями, как review_pack._applied_map:
        ответ в ту же секунду, но позже применения, ждёт; ответ раньше в ту же секунду — применён."""
        answers = self.answers()
        answers["r-ok"]["ts"], answers["r-time"]["ts"] = APPLIED_AT + 0.1, APPLIED_AT + 0.7
        for aid in ("r-ambig", "r-edited"):
            answers.pop(aid)
        self.write_pack(answers=answers, applied_at=APPLIED_AT + 0.2)
        packs, manifests = IL._read_pack_files([self.pack])
        packs[0]["raw"].pop("applied_answers")
        self.assertEqual(IL._utc_text(APPLIED_AT + 0.7), IL._utc_text(APPLIED_AT + 0.2))   # строки равны
        legacy = json.loads(self.tj.read_text(encoding="utf-8"))
        _, _, review, stats = IL.build_snapshots(legacy, recording_id=IL.recording_id_for(self.tj, "s"), metadata={},
                                                 field_sources={}, variant={}, source={}, answers_packs=packs,
                                                 manifests=manifests)
        self.assertEqual(set(review["undo"]["review_pack"]["applied_answers"]), {"r-ok"})
        self.assertEqual((stats["pack_state"]["applied_marks"]["from_applied_at"], stats["pack_state"]["waiting"]),
                         (1, 1))

    def test_new_import_of_archived_pack_is_not_a_repair_candidate(self):
        """Ф207-6: запись, которую новый перенос сразу положил в архив, ремонт считает исправленной."""
        self.write_pack(version="другая")
        self.init_db()
        entry = self.apply()
        self.assertEqual(entry["status"], "imported")
        self.assertEqual(IL.repair_packs(db_only=True)["records"][0]["status"], "already_repaired")
        rec = IL.repair_packs(IL.load_plan(self.write_plan()))["records"][0]
        self.assertEqual(rec["status"], "already_repaired")
        self.assertIn("новым переносом", rec["why"])

    def test_old_pack_formats_are_named_in_inventory(self):
        """Опись (а1) не отказывает старым видам пакета: рукописная фикстура — поля, которых нынешний код уже не
        пишет (секунды, отметки true), и прежний формат answers.json с копией manifest и _expected_versions."""
        self.write_pack(manifest={"version": VERSION, "items": []}, _expected_versions={"source_revision": 1})
        packs, manifests = IL._read_pack_files([self.pack])
        self.assertEqual(IL.unknown_pack_keys(packs, manifests), [])

    def test_unknown_pack_key_refuses_record_before_writing(self):
        """а1: ключ без назначения в описи — отказ записи unknown_pack_keys (код 1), ключ назван, ничего не
        пишется; проверка без записи показывает это до apply."""
        self.write_pack(new_state={"x": 1})
        plan = self.write_plan()
        self.init_db()
        before = self.db_hashes()
        entry = IL.run(self.plan(), "check")[0]
        self.assertEqual(entry["status"], "unknown_pack_keys")
        self.assertEqual(entry["unknown_pack_keys"], ["встреча.v2.review: new_state"])
        with mock.patch.object(IL, "audio_facts", return_value={"audio_available": True}):
            self.assertEqual(IL.run(self.plan(), "apply")[0]["status"], "unknown_pack_keys")
            self.assertEqual(IL.main(["check", "--plan", str(plan)]), 1)
        self.assertEqual(self.db_hashes(), before)

    def test_mismatched_versions_archive_answers(self):
        self.write_pack(version="другая")
        packs, manifests = IL._read_pack_files([self.pack])
        legacy = json.loads(self.tj.read_text(encoding="utf-8"))
        _, _, review, stats = IL.build_snapshots(legacy, recording_id=IL.recording_id_for(self.tj, "m"), metadata={},
                                                 field_sources={}, variant={}, source={}, answers_packs=packs,
                                                 manifests=manifests)
        self.assertEqual(review["answers"], {})
        self.assertNotIn("review_pack", review["undo"])
        self.assertEqual(stats["pack_state"]["state"], "archived_only")
        self.assertIn("не совпадают", stats["pack_state"]["reason"])
        self.assertEqual(stats["pack_state"]["answers_archived"], 6)

    def test_repeat_apply_of_old_record_hints_repair(self):
        self.init_db()
        self.assertEqual(self.apply(old=True)["status"], "imported")
        again = self.apply()
        self.assertEqual((again["status"], again.get("pack_state")), ("skipped", "missing"))
        self.assertIn("repair-packs", again["hint"])


class RepairTests(PackFixture):
    def setUp(self):
        super().setUp()
        self.init_db()
        self.entry = self.apply(old=True)             # перенос прежней версией: признак М-1
        self.rid = self.entry["recording_id"]
        review = self.current(self.rid)["review"]["payload"]
        self.assertTrue(IL.pack_state_missing(review))
        self.assertTrue(review["answers"])

    def run_main(self, *args):
        return IL.main(["repair-packs", *args])

    def test_dry_run_writes_nothing(self):
        before_db, before_files = self.db_hashes(), self.legacy_hashes()
        docs = {p: Path(p).read_bytes() for p in self.entry["documents"].values()}
        report = IL.repair_packs(IL.load_plan(self.write_plan()))
        self.assertEqual([e["status"] for e in report["records"]], ["would_repair"])
        ps = report["records"][0]["pack_state"]
        self.assertEqual((ps["state"], ps["answers_current"], ps["waiting"], ps["waiting_closed_place"]), ("full", 4, 1, 0))
        self.assertEqual(ps["already_decided_marked"], 1)           # Р-4: сухой прогон показывает число пометок
        self.assertEqual(self.db_hashes(), before_db)
        self.assertEqual(self.legacy_hashes(), before_files)
        self.assertEqual({p: Path(p).read_bytes() for p in docs}, docs)
        self.assertFalse((self.db.parent / "db-backups").exists())

    def test_db_only_needs_no_plan_and_reads_no_pack_files(self):
        before_db = self.db_hashes()
        with mock.patch.object(IL, "_read_pack_files", side_effect=AssertionError("файлы пакета не читаются")):
            report = IL.repair_packs(db_only=True)
        rec = report["records"][0]
        self.assertEqual(rec["status"], "candidate")
        # прежний перенос положил в ответы и архивный ответ на открытое место (r-gap): 4 + 1
        self.assertEqual((rec["answers"], rec["answers_on_closed_places"], rec["pack_files"]), (5, 3, 2))
        self.assertEqual(self.db_hashes(), before_db)
        self.assertEqual(self.run_main("--db-only"), 0)
        self.assertEqual(self.run_main("--db-only", "--apply"), 2)

    def test_apply_repairs_with_backup_once_and_repeat_changes_nothing(self):
        before_files = self.legacy_hashes()
        plan = self.write_plan()
        self.assertEqual(self.run_main("--plan", str(plan), "--apply"), 2)            # без --data-dir
        self.assertEqual(self.run_main("--plan", str(plan), "--apply", "--data-dir", str(self.db.parent)), 0)
        backups = list((self.db.parent / "db-backups").glob("state-before-pack-repair-*.sqlite3"))
        self.assertEqual(len(backups), 1)
        storage.validate_copy(backups[0])
        current = self.current(self.rid)
        self.assertEqual((current["transcript"]["revision"], current["review"]["revision"]), (1, 2))
        review = current["review"]["payload"]
        self.assertFalse(IL.pack_state_missing(review))
        self.assertEqual(review["question_version"], VERSION)
        self.assertEqual(review["undo"]["legacy_pack_repair"]["tool"], IL.REPAIR_TOOL)
        self.assertEqual(set(review["answers"]), {"r-ok", "r-ambig", "r-time", "r-edited"})
        state = recordings.read_review_pack_state(self.entry["documents"]["json"])
        self.assertEqual(set(state["applied_answers"]), {"r-ok", "r-ambig"})
        self.assertEqual(IL.os.path.normcase(state["manifest"]["job_json"]),
                         IL.os.path.normcase(self.entry["documents"]["json"]))
        self.assertIn(VERSION, Path(self.entry["documents"]["json"]).read_text(encoding="utf-8"))   # документ v4 обновлён
        self.assertEqual(self.legacy_hashes(), before_files)
        old_items = {i["id"]: i.get("status") for i in self.current(self.rid)["review"]["payload"]["review"]}
        self.assertEqual(old_items, {i["id"]: i.get("status") for i in review["review"]})
        db_after = self.db_hashes()
        report = IL.repair_packs(IL.load_plan(plan), apply=True)
        self.assertEqual(report["records"][0]["status"], "already_repaired")
        self.assertIsNone(report["backup"])
        self.assertEqual(self.db_hashes(), db_after)

    def test_record_changed_after_import_is_refused(self):
        store = recordings.open_store(self.db)
        try:
            current = store.get_current(self.rid)
            review = copy.deepcopy(current["review"]["payload"])
            review["review_done"] = True
            store.commit_change(self.rid, {"source_revision": 1, "review_revision": 1}, str(IL.uuid.uuid4()),
                                recordings.operation_context(store, "human"), review=review)
        finally:
            store.close()
        report = IL.repair_packs(IL.load_plan(self.write_plan()), apply=True)
        self.assertEqual(report["records"][0]["status"], "changed_after_import")
        self.assertIsNone(report["backup"])
        self.assertEqual(self.current(self.rid)["review"]["revision"], 2)

    def test_changed_pack_files_are_refused(self):
        answers = self.answers()
        answers["r-ok"]["option"] = 1
        self.write_pack(answers=answers)
        report = IL.repair_packs(IL.load_plan(self.write_plan()), apply=True)
        self.assertEqual(report["records"][0]["status"], "pack_files_changed")
        self.assertIsNone(report["backup"])
        self.assertEqual(self.current(self.rid)["review"]["revision"], 1)
        self.assertEqual(self.run_main("--plan", str(self.write_plan())), 1)

    def test_failed_backup_writes_nothing(self):
        before_db = self.db_hashes()
        with mock.patch.object(storage.Store, "backup", side_effect=OSError("нет места")):
            code = self.run_main("--plan", str(self.write_plan()), "--apply", "--data-dir", str(self.db.parent))
        self.assertEqual(code, 2)
        self.assertEqual(self.current(self.rid)["review"]["revision"], 1)
        self.assertEqual(self.db_hashes(), before_db)

    def test_damaged_edit_copy_does_not_block_backup(self):
        """Ф207-3: повреждённая копия правки — строка отчёта, не отказ резерва (как у других штатных резервов);
        копия проверяется один раз, незаконченного файла в db-backups не остаётся."""
        damaged = [{"scope": "document", "conflict_id": "c-1", "relative_path": "v4/x.transcript.md"}]
        out = self.root / "repair-out"
        with mock.patch.object(storage, "damaged_conflict_copies", return_value=damaged), \
                mock.patch.object(storage, "validate_copy", wraps=storage.validate_copy) as checks:
            code = self.run_main("--plan", str(self.write_plan()), "--apply", "--data-dir", str(self.db.parent),
                                 "--out", str(out))
        self.assertEqual(code, 0)
        self.assertEqual(checks.call_count, 1)
        self.assertEqual(self.current(self.rid)["review"]["revision"], 2)
        folder = self.db.parent / "db-backups"
        self.assertEqual([p.suffix for p in folder.iterdir()], [".sqlite3"])
        report = json.loads((out / "repair-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["backup_damaged_copies"], damaged)
        self.assertEqual(report["records"][0]["status"], "repaired")

    def test_repeat_repair_of_archived_pack_is_already_repaired(self):
        """Ф207-6: ремонт пакета, ушедшего в архив целиком (версии не совпали), повтором — «уже исправлено», код 0."""
        self.write_pack(version="другая")
        plan = self.write_plan()
        report = IL.repair_packs(IL.load_plan(plan), apply=True)
        self.assertEqual(report["records"][0]["status"], "repaired")
        self.assertEqual(report["records"][0]["pack_state"]["state"], "archived_only")
        db_after = self.db_hashes()
        again = IL.repair_packs(IL.load_plan(plan), apply=True)
        self.assertEqual(again["records"][0]["status"], "already_repaired")
        self.assertEqual(self.db_hashes(), db_after)
        self.assertEqual(self.run_main("--plan", str(plan), "--apply", "--data-dir", str(self.db.parent)), 0)

    def test_documents_failure_after_db_write_is_reported_and_repeat_writes_them(self):
        """Ф207-2: сбой записи файлов после записи в базу — не «ошибка», а «в базе исправлено, документы не
        записаны» (код 1); повтор --apply дописывает документы без новой ревизии."""
        plan = self.write_plan()
        with mock.patch.object(recordings, "materialize_recording", side_effect=OSError("диск занят")):
            report = IL.repair_packs(IL.load_plan(plan), apply=True)
        rec = report["records"][0]
        self.assertEqual(rec["status"], "repaired_documents_pending", rec)
        self.assertIn("диск занят", rec["error"])
        self.assertEqual(self.current(self.rid)["review"]["revision"], 2)
        self.assertNotIn(VERSION, Path(self.entry["documents"]["json"]).read_text(encoding="utf-8"))
        self.assertEqual(IL.repair_packs(db_only=True)["records"][0]["status"], "repaired_documents_pending")
        self.assertEqual(self.run_main("--plan", str(plan)), 1)                     # сухой прогон: нужен повтор
        again = IL.repair_packs(IL.load_plan(plan), apply=True)
        self.assertEqual(again["records"][0]["status"], "documents_written", again)
        self.assertIsNone(again["backup"])
        self.assertEqual(self.current(self.rid)["review"]["revision"], 2)
        self.assertIn(VERSION, Path(self.entry["documents"]["json"]).read_text(encoding="utf-8"))
        store = recordings.open_store(self.db)
        try:
            self.assertEqual(IL._document_state(store, self.rid), "synced")
        finally:
            store.close()
        self.assertEqual(IL.repair_packs(IL.load_plan(plan), apply=True)["records"][0]["status"], "already_repaired")

    def test_repair_marks_closed_place_answers_already_decided(self):
        """Р-4 (решение владельца 26.11.2026): ремонт помечает старый ответ закрытого места «место уже решено»;
        «Применить» после ремонта отказывает только из-за ответа открытого места без автора (r-time); повтор
        ремонта ничего не меняет."""
        plan = self.write_plan()
        self.assertEqual(IL.repair_packs(IL.load_plan(plan))["records"][0]["pack_state"]["already_decided_marked"], 1)
        rec = IL.repair_packs(IL.load_plan(plan), apply=True)["records"][0]
        self.assertEqual((rec["status"], rec["pack_state"]["already_decided_marked"]), ("repaired", 1))
        jp = self.entry["documents"]["json"]
        self.assertEqual({k: v["state"] for k, v in review_pack.answer_states(jp).items()}["r-edited"],
                         "already_decided")
        with self.assertRaises(review_pack.RebuildRefused) as caught:
            review_pack.check_rebuild(jp)
        self.assertEqual([w["id"] for w in caught.exception.waiting], ["r-time"])
        import core
        before = self.current(self.rid)["transcript"]["payload"]
        state = recordings.read_review_pack_state(jp)
        self.assertTrue(review_pack.legacy_decided(state, "r-edited", state["answers"]["r-edited"]))
        self.assertFalse(review_pack.legacy_decided(state, "r-time", state["answers"]["r-time"]))
        with self.assertRaisesRegex(core.CoreError, "неизвестного происхождения"):
            core.apply_answers("recording:" + self.rid)
        self.assertEqual(self.current(self.rid)["transcript"]["payload"], before)
        db_after = self.db_hashes()
        self.assertEqual(IL.repair_packs(IL.load_plan(plan), apply=True)["records"][0]["status"], "already_repaired")
        self.assertEqual(self.db_hashes(), db_after)

    def test_unknown_pack_key_refuses_repair(self):
        """а1 в ремонте: сухой прогон и --apply называют ключ, база не меняется, код 1."""
        self.write_pack(new_state={"x": 1})
        plan = self.write_plan()
        before = self.db_hashes()
        rec = IL.repair_packs(IL.load_plan(plan))["records"][0]
        self.assertEqual((rec["status"], rec["unknown_pack_keys"]), ("unknown_pack_keys", ["встреча.v2.review: new_state"]))
        self.assertEqual(self.run_main("--plan", str(plan), "--apply", "--data-dir", str(self.db.parent)), 1)
        self.assertEqual(self.db_hashes(), before)
        self.assertFalse((self.db.parent / "db-backups").exists())

    def test_not_migrated_and_nothing_to_repair(self):
        plan = self.plan()
        other = self.side / "другая.transcript.json"
        other.write_text(json.dumps(legacy_transcript(self.audio), ensure_ascii=False) + " ", encoding="utf-8")
        plan["recordings"][0]["variants"].append({"transcript": str(other), "tag": "x", "order": 2})
        report = IL.repair_packs(plan)
        self.assertEqual([e["status"] for e in report["records"]], ["would_repair", "not_migrated"])


if __name__ == "__main__":
    unittest.main()
