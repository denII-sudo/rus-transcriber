"""Перенос П7 и ремонт repair-packs: пакет карточек, собранный самой программой, и те же действия после переноса.

Механизмы А и Б разбора причин М-1 (ops research/2026-11-26-migration-root-cause, раздел 4) и находка Ф207-5:
- фикстура — запись в файловом режиме, пакет которой собран штатными вызовами (review_pack.build, add_answers,
  core.apply_answers_files, add_interaction_event), а не написан руками: программа собрала её один раз, и она
  заморожена в образцах tests/legacy_samples/ (Л-3.1; загрузчик — tests/legacy_sample_loader.py), чтобы тесты
  не зависели от файловых веток, которые уйдут в Л-5;
- опись ключей пакета (import_legacy.LEGACY_PACK_INVENTORY): ключ без назначения роняет тест;
- после переноса (и после ремонта записи, перенесённой прежней версией) работают те же действия, что до него:
  проверка пересборки, «Применить», дописывание порции карточек.
Звук не режется (cut_clip подменён), аудио — синтетический файл. Только синтетика во временной папке.
"""
import atexit
import fnmatch
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-import-behavior-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:          # временная папка данных — до импорта программы
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "tools"))
sys.path.insert(0, str(PROJECT))

import core  # noqa: E402
import import_legacy as IL  # noqa: E402
import legacy_sample_loader as samples  # noqa: E402
import paths  # noqa: E402
import recordings  # noqa: E402
import review_pack  # noqa: E402
import settings  # noqa: E402
import transcript  # noqa: E402

AS_IS = {"label": "Оставить как есть", "decision": {}}


def questions(ids=("r2", "r3", "r4")):
    rows = {"r2": [AS_IS], "r3": [AS_IS, {"label": "«дальнейшем»", "decision": {"text": "дальнейшем"}}],
            "r4": [AS_IS, {"label": "Заменить на «тот»", "decision": {"text": "тот"}}],
            "r5": [AS_IS]}
    return [{"id": i, "question": "Проверьте", "recommended": 0, "options": rows[i]} for i in ids]


class Built(unittest.TestCase):
    """Запись файлового режима рядом с аудио и её пакет, собранный программой (образец «built»): r3 — ответ
    человека применён («Применить» файлового режима), r4 — ответ человека ждёт «Применить», r2 — без ответа,
    r5 — без карточки."""
    _ts = 0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=_TMP)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = self.root / "встреча.wav"
        self.audio.write_bytes(b"synthetic audio")
        side = self.root / "встреча.rus-transcriber"
        side.mkdir()
        self.jp, self.md = side / "встреча.transcript.json", side / "встреча.transcript.md"
        self.db = self.root / "data" / "state.sqlite3"
        self.db.parent.mkdir()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(paths, "STATE_DB", self.db))
        stack.enter_context(mock.patch.object(paths, "DATA_DIR", self.db.parent))
        stack.enter_context(mock.patch.object(review_pack, "cut_clip", side_effect=lambda _s, t1, _t2, _o: t1))
        self.backend = "legacy_files"
        stack.enter_context(mock.patch.object(settings, "recording_backend", side_effect=lambda: self.backend))
        samples.load("built", side, self.audio)
        self.version = review_pack.manifest(self.jp)["version"]
        self.pack = review_pack.pack_dir(self.jp)

    def answer(self, aid, option, actor="human", jp=None):
        Built._ts += 1
        row = {"id": aid, "version": self.version, "ts": f"2026-11-20T10:{Built._ts // 60:02d}:{Built._ts % 60:02d}",
               "option": option}
        return review_pack.add_answers(jp or self.jp, [row], "local", trusted_actor=actor)

    def plan(self):
        return {"version": 1, "recordings": [{
            "audio": str(self.audio), "metadata": {"title": "встреча"},
            "variants": [{"transcript": str(self.jp), "tag": "v1", "order": 1, "main": True,
                          "review_packs": [str(self.pack)]}]}]}

    def migrate(self, old=False):
        self.backend = "sqlite"
        recordings.initialize_store(self.db, display_name="Владелец").close()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(IL, "audio_facts", return_value={"audio_available": True}))
            if old:
                stack.enter_context(mock.patch.object(IL, "pack_state_from_legacy", old_pack_import))
            entry = IL.run(self.plan(), "apply")[0]
        self.assertEqual(entry["status"], "imported", entry)
        return entry


def old_pack_import(review, transcript_, packs, manifests, recording_id, legacy_runtime=None):
    """Прежний перенос (до В-1): только ответы и упрощённые вопросы — так перенесены записи с М-1."""
    for pack in packs:
        IL.import_answers(review, pack.get("answers"), pack["name"])
    IL.import_questions(review, transcript_, manifests)
    return {"state": "none", "answers_current": 0, "answers_archived": 0, "ts_converted": 0}


def waiting(jp):
    """Ответы, из-за которых пересборка пакета отказывает (review_pack.check_rebuild): [id, …]."""
    try:
        review_pack.check_rebuild(jp)
    except review_pack.RebuildRefused as ex:
        return [w["id"] for w in ex.waiting]
    return []


def states(jp):
    return {k: v["state"] for k, v in review_pack.answer_states(jp).items()}


class InventoryTests(Built):
    def test_every_key_and_file_of_a_built_pack_has_a_destination(self):
        """А: пакет, собранный программой (с архивом прежней редакции), — каждый ключ answers*.json и
        manifest*.json и каждый файл папки разбора названы в описи; ключ, которого нет в описи, роняет этот тест.
        Пакет — образец «built_archive» (Л-3.1): программа применила r4 и пересобрала пакет с карточками r2…r5,
        прежняя редакция ушла в архив; ключи, которые она запишет в будущем, образец уже не покажет — файловый
        режим новых пакетов не создаёт (Л-4а)."""
        samples.load("built_archive", self.jp.parent, self.audio)
        packs, manifests = IL._read_pack_files([self.pack])
        self.assertGreater(len(packs), 1, "нет архива прежней редакции — фикстура не покрывает архивы")
        self.assertEqual(IL.unknown_pack_keys(packs, manifests), [])
        patterns = IL.LEGACY_PACK_INVENTORY["files"]
        unnamed = [p.name for p in self.pack.iterdir() if not any(fnmatch.fnmatch(p.name, pat) for pat in patterns)]
        self.assertEqual(unnamed, [])
        # опись не пустая сама по себе: ключи нынешнего формата в ней есть
        keys = {k for p in packs for k in p["raw"]}
        self.assertTrue({"applied_answers", "application_events", "interaction_events", "heard"} <= keys)


class RealCardsApplyTests(Built):
    """Ф207-5: «Применить» после переноса на карточках настоящего вида (с t и rev, conflicts с was)."""

    def test_waiting_answer_on_other_revision_gets_honest_refusal(self):
        # r3 применён в файловом режиме — ревизия стенограммы выросла после сборки пакета
        entry = self.migrate()
        dj = entry["documents"]["json"]
        self.assertEqual(recordings.read_review_pack_state(dj)["manifest"]["source_revision"], 0)
        before = transcript.load(dj)["segments"]
        rep = core.apply_answers("recording:" + entry["recording_id"])
        out = {o["id"]: o for o in rep["outcomes"]}
        self.assertEqual(out["r4"]["outcome"], "apply_conflict", rep["outcomes"])
        self.assertEqual(transcript.load(dj)["segments"], before)

    def test_waiting_answer_on_imported_revision_is_applied(self):
        tj = samples.load("waiting", self.jp.parent, self.audio)      # r3 ждёт «Применить», ревизия та же, что у пакета
        md = tj.with_suffix(".md")
        self.backend = "legacy_files"
        self.version = review_pack.manifest(tj)["version"]
        self.jp, self.md, self.pack = tj, md, review_pack.pack_dir(tj)
        entry = self.migrate()
        dj = entry["documents"]["json"]
        self.assertEqual(recordings.read_review_pack_state(dj)["manifest"]["source_revision"], 1)
        rep = core.apply_answers("recording:" + entry["recording_id"])
        self.assertEqual({o["id"]: o["outcome"] for o in rep["outcomes"]}.get("r3"), "applied", rep["outcomes"])
        data = transcript.load(dj)
        self.assertIn("дальнейшем", [w["w"] for s in data["segments"] for w in s["words"]])
        self.assertIn(next(i for i in data["review"] if i["id"] == "r3")["status"], IL.CLOSED_STATUSES)
        store = recordings.open_store(self.db)
        try:
            self.assertEqual(store.get_current(entry["recording_id"])["transcript"]["revision"], 2)
        finally:
            store.close()


class GuardOnOrdinaryRecordTests(Built):
    def test_answer_without_author_on_closed_place_still_blocks_apply(self):
        """Р-4: защита «Применить» пропускает только ответ с событием переноса legacy_import/already_decided.
        На обычной записи (без переноса) ответ без автора на закрытом месте по-прежнему требует подтверждения."""
        data = transcript.load(self.jp)
        next(i for i in data["review"] if i["id"] == "r2")["status"] = "resolved"
        transcript.save(data, self.jp, self.md)
        self.answer("r2", 0, actor="unknown")
        before = transcript.load(self.jp)["segments"]
        with self.assertRaisesRegex(core.CoreError, "неизвестного происхождения"):
            core.apply_answers_files(self.jp, self.md)
        self.assertEqual(transcript.load(self.jp)["segments"], before)


class SameActionsAfterImportTests(Built):
    """Б: после переноса работают те же действия над записью, что до него: состояния ответов, проверка
    пересборки, «Применить», дописывание порции (пункт 2а плана М-1)."""

    def same_actions(self, entry, before_states):
        dj = entry["documents"]["json"]
        self.assertEqual(states(dj), before_states)
        # в режиме базы пересборка не теряет ждущий ответ (PR 17-Б): отказ называет его; в файлах проверки нет
        self.assertEqual(waiting(dj), ["r4"])
        rep = core.apply_answers("recording:" + entry["recording_id"])
        self.assertIsInstance(rep, dict)
        self.assertEqual(waiting(dj), [])                 # ответ отработан — пересборка и дописывание открыты
        pack_rev = review_pack.manifest(dj)["pack_rev"]
        with mock.patch.object(core, "_done_json", return_value={"json": dj}):
            core.build_review_pack("recording:" + entry["recording_id"], questions(("r5",)))
        man = review_pack.manifest(dj)
        self.assertEqual([i["id"] for i in man["items"]], ["r2", "r3", "r4", "r5"])
        self.assertEqual(man["pack_rev"], pack_rev + 1)
        state = recordings.read_review_pack_state(dj)
        self.assertEqual(state["version"], self.version)  # дописано в тот же пакет, не пересобрано
        self.assertIn("r3", state["applied_answers"])
        self.assertEqual(states(dj)["r3"], "applied")

    def test_new_import(self):
        before_states = states(self.jp)
        self.assertEqual(before_states, {"r3": "applied", "r4": "saved"})
        self.same_actions(self.migrate(), before_states)

    def test_old_import_repaired(self):
        before_states = states(self.jp)
        entry = self.migrate(old=True)
        store = recordings.open_store(self.db)
        try:
            self.assertTrue(IL.pack_state_missing(store.get_current(entry["recording_id"])["review"]["payload"]))
        finally:
            store.close()
        report = IL.repair_packs(self.plan(), apply=True)
        self.assertEqual(report["records"][0]["status"], "repaired", report)
        self.same_actions(entry, before_states)


if __name__ == "__main__":
    unittest.main()
