import copy
import asyncio
import contextlib
import json
import os
import http.client
import socket
import subprocess
import sys
import tempfile
import time
import conspect
import core
import unittest
import uuid
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import documents
import freeform
import recordings
import review_pack
import settings
import storage
import transcript
import workflow


class AnswerRevisionGuardTests(unittest.TestCase):
    def fixture(self):
        helper = WorkerSnapshotTests()
        data = helper.answer_result(word_ids=True)
        data["segments"][0]["words"] = [
            {"w": "один", "s": 0.2, "e": 0.4, "word_id": "w1"},
            {"w": "два", "s": 0.5, "e": 0.7, "word_id": "w2"},
            {"w": "три", "s": 0.8, "e": 1.0, "word_id": "w3"},
        ]
        data["segments"][0]["text"] = "один два три"
        data["review"] = [
            {"id": "r1", "status": "open", "text": "один", "t": 0.2, "t1": 0.2,
             "t2": 0.4, "speaker": "Глеб", "word_ids": ["w1"]},
            {"id": "r2", "status": "open", "text": "три", "t": 0.8, "t1": 0.8,
             "t2": 1.0, "speaker": "Глеб", "word_ids": ["w3"]},
        ]
        return helper, data

    def build(self, jp):
        return review_pack.build(jp, [
            {"id": "r1", "question": "Проверьте", "options": [
                {"label": "Заменить", "decision": {"text": "первый"}}]},
            {"id": "r2", "question": "Проверьте", "options": [
                {"label": "Оставить", "decision": {}}]},
        ])

    def answer(self, jp, pack, free=False):
        row = {"id": "r1", "version": pack["version"]}
        row.update({"kind": "free", "text": "первый"} if free else {"option": 0})
        review_pack.add_answers(jp, [row], "local", trusted_actor="human")

    @staticmethod
    def strip_word_ids(jp):
        """Старый вид карточки: ответ ищет место по времени, номеров слов в manifest нет."""
        state = review_pack.answers_state(jp)
        for item in state["manifest"]["items"]:
            item.pop("word_ids", None)
        review_pack._write_answers(jp, state)

    def test_unrelated_edit_then_answer_and_undo(self):
        helper, data = self.fixture()
        with helper.answer_files("sqlite", data) as (jp, md):
            pack = self.build(jp)
            manifest = copy.deepcopy(review_pack.manifest(jp))
            transcript.edit_span(jp, md, 0.8, 1.0, "последний", by="human")
            self.answer(jp, pack)
            result = core.apply_answers_files(jp, md)
            self.assertEqual(result["applied_from_answers"], 1)
            self.assertEqual(helper.card_state(jp)[0], ["первый", "два", "последний"])
            self.assertEqual(review_pack.manifest(jp), manifest)
            review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"], "kind": "revert"}],
                                    "local", trusted_actor="human")
            self.assertEqual(core.apply_answers_files(jp, md)["reverted"], ["r1"])
            self.assertEqual(helper.card_state(jp)[0], ["один", "два", "последний"])

    def test_late_edit_blocks_option_and_free(self):
        for free in (False, True):
            helper, data = self.fixture()
            with self.subTest(free=free), helper.answer_files("sqlite", data) as (jp, md):
                pack = self.build(jp)
                transcript.edit_span(jp, md, 0.2, 0.4, "ручной", by="human")
                before = copy.deepcopy(transcript.load(jp)["segments"])
                self.answer(jp, pack, free=free)
                result = core.apply_answers_files(jp, md)
                self.assertEqual(transcript.load(jp)["segments"], before)
                self.assertNotIn("r1", review_pack.answers_state(jp)["applied_answers"])
                self.assertEqual(result["applied_from_answers"], 0)
                self.assertTrue(result.get("revision_conflicts"))

    def test_missing_id_blocks_without_retargeting(self):
        for backend in ("sqlite", "legacy_files"):
            helper, data = self.fixture()
            with self.subTest(backend=backend), helper.answer_files(backend, data) as (jp, md):
                pack = self.build(jp)
                transcript.edit_span(jp, md, 0.2, 0.7, "вместе", by="human")
                before = copy.deepcopy(transcript.load(jp)["segments"])
                self.answer(jp, pack)
                result = core.apply_answers_files(jp, md)
                self.assertEqual(transcript.load(jp)["segments"], before)
                self.assertNotIn("r1", review_pack.answers_state(jp)["applied_answers"])
                self.assertTrue(result.get("revision_conflicts"))

    def test_pruned_refs_target_and_model_doubts(self):
        for partial in (True, False):
            helper, data = self.fixture()
            data["review"] = [data["review"][0]]
            data["review"][0].update(t2=1.0, word_ids=["w1", "w2"] if partial else ["w1"])
            with self.subTest(partial=partial), helper.answer_files("sqlite", data) as (jp, md):
                review_pack.build(jp, [{"id": "r1", "question": "Проверьте", "options": [
                    {"label": "Оставить", "decision": {}}]}])
                manifest = copy.deepcopy(review_pack.manifest(jp))
                transcript.edit_span(jp, md, 0.2, 0.4, "", by="human")
                current = transcript.load(jp)
                place = current["review"][0]
                question = current["questions"][0]
                if partial:
                    self.assertEqual(place["word_ids"], ["w2"])
                    self.assertEqual(question["word_ids"], ["w2"])
                else:
                    self.assertNotIn("word_ids", place)
                    self.assertNotIn("word_ids", question)
                loc = transcript._find_span(current, place)
                self.assertIsNotNone(loc)
                si, start, end = loc
                expected = ["w2"] if partial else ["w2", "w3"]
                self.assertEqual([w["word_id"] for w in current["segments"][si]["words"][start:end]], expected)
                self.assertEqual(review_pack.manifest(jp), manifest)
                task = workflow._build_task(
                    {"recording_id": "synthetic-recording", "policy": {"goal": ""}},
                    {"transcript": {"revision": 1, "payload": current},
                     "review": {"revision": 1, "payload": {"review": [place]}}})
                self.assertEqual([w["word_id"] for w in task["words"] if w["uncertain"]], expected)
                self.assertEqual(task["unlocated_doubts"], 0)

    def test_missing_source_revision_is_a_conflict(self):
        helper, data = self.fixture()
        with helper.answer_files("sqlite", data) as (jp, md):
            pack = self.build(jp)
            self.answer(jp, pack)
            before = copy.deepcopy(transcript.load(jp)["segments"])
            with mock.patch.object(core, "_answer_source_snapshot", return_value=None):
                result = core.apply_answers_files(jp, md)
            self.assertTrue(result["revision_conflicts"])
            self.assertEqual(transcript.load(jp)["segments"], before)
            self.assertNotIn("r1", review_pack.answers_state(jp)["applied_answers"])

    def test_free_answer_cannot_change_neighbor(self):
        helper, data = self.fixture()
        with helper.answer_files("sqlite", data) as (jp, md):
            pack = self.build(jp)
            review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"],
                                          "kind": "free", "text": "один другой три"}],
                                    "local", trusted_actor="human")
            before = copy.deepcopy(transcript.load(jp)["segments"])
            result = core.apply_answers_files(jp, md)
            self.assertTrue(result.get("revision_conflicts"))
            self.assertEqual(transcript.load(jp)["segments"], before)
            self.assertNotIn("r1", review_pack.answers_state(jp)["applied_answers"])

    def test_comparison_checks_order_and_edit_metadata(self):
        helper, data = self.fixture()
        item = {"id": "r1", "word_ids": ["w1", "w2"]}
        manifest = {"items": [item]}
        state = {"answers": {"r1": {"kind": "option", "option": 0}}}
        source = copy.deepcopy(data)
        cases = []
        changed = copy.deepcopy(data)
        changed["segments"][0]["words"][0]["edited"] = "span@0.20"
        cases.append(changed)
        changed = copy.deepcopy(data)
        changed["segments"][0]["words"][0]["provenance"] = {"text": {"by": "human"}}
        cases.append(changed)
        changed = copy.deepcopy(data)
        changed["segments"][0]["words"][:2] = reversed(changed["segments"][0]["words"][:2])
        cases.append(changed)
        for changed in cases:
            self.assertIn("r1", core._answer_revision_conflicts(changed, manifest, state, source, sqlite=True))

    def test_legacy_card_without_ids_blocks_late_edits(self):
        cases = [
            ((0.2, 0.4, "один", "Борис"), {"speaker": "Глеб"}),
            ((0.2, 0.4, "Один", None), {"text": "один"}),
        ]
        for edit, decision in cases:
            helper, data = self.fixture()
            with self.subTest(edit=edit), helper.answer_files("sqlite", data) as (jp, md):
                pack = review_pack.build(jp, [{
                    "id": "r1", "question": "Проверьте", "options": [
                        {"label": "Вариант", "decision": decision}]}])
                self.strip_word_ids(jp)
                transcript.edit_span(jp, md, *edit, by="human")
                before = copy.deepcopy(transcript.load(jp)["segments"])
                review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"], "option": 0}],
                                        "local", trusted_actor="human")
                result = core.apply_answers_files(jp, md)
                self.assertEqual(transcript.load(jp)["segments"], before)
                self.assertEqual(result["applied_from_answers"], 0)
                self.assertTrue(result.get("revision_conflicts"))
                self.assertNotIn("r1", review_pack.answers_state(jp)["applied_answers"])

    def test_legacy_card_without_ids_blocks_insertion_in_window(self):
        helper, data = self.fixture()
        with helper.answer_files("sqlite", data) as (jp, md):
            pack = self.build(jp)
            self.strip_word_ids(jp)
            transcript.apply_word_edits(jp, md, [{
                "op": "insert", "s": 0.35, "e": 0.39, "text": "лишний",
                "speaker": "Глеб", "after_id": "w1", "card": "вставка"}], by="human")
            before = copy.deepcopy(transcript.load(jp)["segments"])
            self.answer(jp, pack)
            result = core.apply_answers_files(jp, md)
            self.assertEqual(transcript.load(jp)["segments"], before)
            self.assertEqual(result["applied_from_answers"], 0)
            self.assertTrue(result.get("revision_conflicts"))
            self.assertNotIn("r1", review_pack.answers_state(jp)["applied_answers"])

    def test_legacy_card_without_ids_allows_unrelated_edit(self):
        helper, data = self.fixture()
        with helper.answer_files("sqlite", data) as (jp, md):
            pack = self.build(jp)
            self.strip_word_ids(jp)
            transcript.edit_span(jp, md, 0.8, 1.0, "последний", by="human")
            self.answer(jp, pack)
            result = core.apply_answers_files(jp, md)
            self.assertEqual(result["applied_from_answers"], 1)
            self.assertEqual(helper.card_state(jp)[0], ["первый", "два", "последний"])


class WorkerSnapshotTests(unittest.TestCase):
    def worker_result(self):
        return {
            "file": "synthetic.wav",
            "duration": "00:00:02",
            "speakers": ["Спикер 1"],
            "speaker_labels": ["speaker-1"],
            "timing": {"total_sec": 4},
            "group": "demo",
            "quality": {"asr_incomplete_chunks": 0},
            "vocab_suggestions": [{"to": "Кварцлайн", "from": "Кварцлайнъ", "card": "r1"}],
            "segments": [{
                "start": 0.2, "end": 1.2, "label": "speaker-1", "speaker": "Спикер 1",
                "text": "тест", "words": [{"w": "тест", "s": 0.2, "e": 0.5}],
            }],
            "review": [{
                "id": "r1", "status": "open", "text": "тест",
                "t": 0.2, "t1": 0.2, "t2": 0.5, "speaker": "Спикер 1", "word_ids": [],
            }],
            "review_done": False,
            "edits": [],
        }

    NAME_SPEAKERS = ["Спикер 1", "Спикер 10", "Анна", "Анна Петровна", "Борис", "Вера", "Глеб"]

    def answer_result(self, word_ids=False):
        """Результат с одним словом «тест» у Глеба: любой названный в ответе говорящий — это правка."""
        data = self.worker_result()
        data["speakers"] = list(self.NAME_SPEAKERS)
        data["segments"][0]["speaker"] = "Глеб"
        data["review"][0]["speaker"] = "Глеб"
        if word_ids:
            data["segments"][0]["words"][0]["word_id"] = "w1"
            data["review"][0]["word_ids"] = ["w1"]
        else:                     # старый вид карточки: без номеров слов (пустой список — не старый вид)
            data["review"][0].pop("word_ids", None)
        return data

    @contextlib.contextmanager
    def answer_files(self, backend, data=None):
        """Пара файлов стенограммы в выбранном режиме хранения: (jp, md). Папка и база — временные."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db, audio = root / "state.sqlite3", root / "sample.wav"
            audio.write_bytes(b"synthetic")
            data = copy.deepcopy(data or self.answer_result())
            data["file"] = str(audio)
            layout = recordings.document_layout(audio)
            layout["root"].mkdir()
            store = None
            with mock.patch.object(settings.paths, "STATE_DB", db), \
                    mock.patch.object(review_pack, "cut_clip", side_effect=lambda _s, t1, _t2, _out: t1):
                try:
                    if backend == "sqlite":
                        store = recordings.initialize_store(db, display_name="Тест")
                        root_id, targets = recordings.prepare_new_layout(store, layout["root"], layout["paths"])
                        rid = str(uuid.uuid4())
                        recordings.commit_worker_result(
                            store, data, audio, root_id, recordings.operation_context(store, "model"),
                            recording_id=rid, operation_id=str(uuid.uuid4()),
                            generated_at="2026-11-16T00:00:00Z", document_paths=targets)
                        recordings.materialize_recording(store, rid)
                        jp, md = layout["root"] / targets["json"], layout["root"] / targets["md"]
                    else:
                        jp, md = layout["root"] / layout["paths"]["json"], layout["root"] / layout["paths"]["md"]
                        transcript.save(data, jp, md)
                    yield jp, md
                finally:
                    if store:
                        store.close()

    @staticmethod
    def answer_card(jp, md, answers, recommended=0):
        """Собрать пакет, по очереди сохранить и применить ответы человека: [(текст, голос?)] | {"option": n}.
        -> отчёт последнего применения."""
        question = {"id": "r1", "question": "Проверьте", "recommended": recommended, "options": [
            {"label": "Другое слово", "decision": {"text": "другое"}},
            {"label": "Третье слово", "decision": {"text": "третье"}}]}
        pack = review_pack.build(jp, [question])
        result = None
        for answer in answers:
            row = {"id": "r1", "version": pack["version"]}
            if isinstance(answer, dict):
                row.update(answer)
            else:
                row.update(text=answer[0], heard=answer[1])
            review_pack.add_answers(jp, [row], "local", trusted_actor="human")
            result = core.apply_answers_files(jp, md)
        return result

    @staticmethod
    def card_state(jp):
        d = transcript.load(jp)
        return ([w["w"] for s in d["segments"] for w in s["words"]],
                [s["speaker"] for s in d["segments"]], d["review"][0].get("status"))

    def test_names_and_labels_in_both_storage_modes(self):
        # Различий между режимами хранения в разборе ответа нет: один и тот же ожидаемый итог для обоих.
        cases = [
            ("Спикер 1:", "name", "Спикер 1"),
            ("Спикер 1", "name", "Спикер 1"),
            ("Спикер 10", "name", "Спикер 10"),
            ("Анна", "name", "Анна"),
            ("Анна Петровна", "name", "Анна Петровна"),
            ("Борис, не Анна", "name", "Борис"),
            ("не Анна, а Вера", "name", "Вера"),
            ("это не Анна", "unmatched", None),
            ("нет, не Анна", "unmatched", None),
            ("не Анна и не Борис", "unmatched", None),
            ("ни Анна, ни Борис", "unmatched", None),
            ("не Анна: тест", "unmatched", None),
            ("ни Анна: тест", "unmatched", None),
            ("нет, Анна: тест", "unmatched", None),
            ("не Спикер 1: тест", "unmatched", None),
            ("Спикер 3: тест", "unmatched", None),
        ]
        for backend in ("legacy_files", "sqlite"):
            with self.subTest(backend=backend), self.answer_files(backend) as (jp, _md):
                loaded = transcript.load(jp)
                man = {"items": [dict(loaded["review"][0])]}
                for text, kind, speaker in cases:
                    with self.subTest(backend=backend, answer=text):
                        edits, reports, _, _ = freeform.plan(man, {"r1": {"text": text}}, loaded)
                        self.assertEqual(reports[0]["kind"], kind)
                        if speaker:
                            self.assertEqual([e["speaker"] for e in edits], [speaker])
                        if kind == "unmatched":
                            self.assertFalse(edits)
                            self.assertTrue(reports[0]["human"])

    def test_name_answers_applied_in_both_storage_modes(self):
        # Сквозной путь: ответ сохранён и применён. Итог одинаков для sqlite и legacy_files.
        named = [("Спикер 1:", "Спикер 1"), ("Спикер 1", "Спикер 1"), ("Спикер 10", "Спикер 10"),
                 ("Анна", "Анна"), ("Анна Петровна", "Анна Петровна"), ("Борис, не Анна", "Борис"),
                 ("не Анна, а Вера", "Вера"), ("нет, не Анна, а Вера", "Вера")]
        to_human = ["это не Анна", "нет, не Анна", "не Анна и не Борис", "ни Анна, ни Борис",
                    "не Анна: тест", "нет, Анна: тест", "Спикер 3: тест"]
        for backend in ("legacy_files", "sqlite"):
            for ids in (False, True):
                for text, speaker in named:
                    with self.subTest(backend=backend, word_ids=ids, answer=text), \
                            self.answer_files(backend, self.answer_result(ids)) as (jp, md):
                        self.answer_card(jp, md, [(text, False)])
                        self.assertEqual(self.card_state(jp)[:2], (["тест"], [speaker]))
                        if "не Анна" in text:      # отрицаемое имя не назначено ни одной правкой
                            edits = transcript.load(jp).get("edits") or []
                            self.assertNotIn("Анна", json.dumps(edits, ensure_ascii=False).replace(text, ""))
                for text in to_human:
                    with self.subTest(backend=backend, word_ids=ids, answer=text), \
                            self.answer_files(backend, self.answer_result(ids)) as (jp, md):
                        result = self.answer_card(jp, md, [(text, False)])
                        self.assertEqual(self.card_state(jp), (["тест"], ["Глеб"], "human"))
                        self.assertTrue(result["free_text"]["left_to_human"])

    def test_negative_answers_remain_open_in_both_storage_modes(self):
        negatives = ("неверно", "не верно", "неправильно", "не согласен",
                     "это неверно", "нет, неверно", "не согласен с этим")
        item = {"id": "r1", "options": [{"label": "верный вариант", "decision": {"text": "другое"}}],
                "recommended": 0}
        for backend in ("legacy_files", "sqlite"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                db = root / "state.sqlite3"
                audio = root / "sample.wav"
                audio.write_bytes(b"synthetic")
                data = self.worker_result()
                data["file"] = str(audio)
                layout = recordings.document_layout(audio)
                layout["root"].mkdir()
                store = None
                with mock.patch.object(settings.paths, "STATE_DB", db):
                    try:
                        if backend == "sqlite":
                            store = recordings.initialize_store(db, display_name="Тест")
                            root_id, targets = recordings.prepare_new_layout(store, layout["root"], layout["paths"])
                            rid = str(uuid.uuid4())
                            recordings.commit_worker_result(
                                store, data, audio, root_id, recordings.operation_context(store, "model"),
                                recording_id=rid, operation_id=str(uuid.uuid4()),
                                generated_at="2026-11-16T00:00:00Z", document_paths=targets)
                            recordings.materialize_recording(store, rid)
                            jp, md = layout["root"] / targets["json"], layout["root"] / targets["md"]
                        else:
                            jp, md = layout["root"] / layout["paths"]["json"], layout["root"] / layout["paths"]["md"]
                            transcript.save(data, jp, md)
                        before = transcript.load(jp)
                        card = dict(before["review"][0])
                        card.pop("word_ids", None)
                        man = {"items": [card], "recording_id": before.get("recording_id")}
                        with mock.patch.object(review_pack, "manifest", return_value=man):
                            for text in negatives:
                                for mode in ("typed", "voice"):
                                    with self.subTest(backend=backend, mode=mode, answer=text):
                                        if mode == "voice":
                                            self.assertEqual(review_pack.match(text, item)["kind"], "free")
                                        result = review_pack.apply_free(jp, md, {"r1": {"text": text}})
                                        self.assertTrue(result["left_to_human"])
                                        self.assertIn("слово", result["left_to_human"][0]["why"])
                                        self.assertEqual(transcript.load(jp), before)
                        # PR 6б (В8): голос = текст — «всё верно» разбирает таблица слов, рекомендацию не выбирает
                        self.assertEqual(review_pack.match("всё верно", item)["kind"], "free")
                    finally:
                        if store:
                            store.close()

    def test_negative_does_not_undo_applied_answer_in_both_modes(self):
        for backend in ("legacy_files", "sqlite"):
            for text, mode in ((text, mode) for text in ("неверно", "не верно", "неправильно", "не согласен",
                                                         "это неверно", "нет, неверно", "не согласен с этим",
                                                         "неверно: слово")
                               for mode in ("typed", "voice")):
                with self.subTest(backend=backend, answer=text, mode=mode), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    db, audio = root / "state.sqlite3", root / "sample.wav"
                    audio.write_bytes(b"synthetic")
                    data = self.worker_result()
                    data["file"] = str(audio)
                    data["review"][0].pop("word_ids", None)
                    layout = recordings.document_layout(audio)
                    layout["root"].mkdir()
                    store = None
                    with mock.patch.object(settings.paths, "STATE_DB", db):
                        try:
                            if backend == "sqlite":
                                store = recordings.initialize_store(db, display_name="Тест")
                                root_id, targets = recordings.prepare_new_layout(store, layout["root"], layout["paths"])
                                rid = str(uuid.uuid4())
                                recordings.commit_worker_result(
                                    store, data, audio, root_id, recordings.operation_context(store, "model"),
                                    recording_id=rid, operation_id=str(uuid.uuid4()),
                                    generated_at="2026-11-16T00:00:00Z", document_paths=targets)
                                recordings.materialize_recording(store, rid)
                                jp, md = layout["root"] / targets["json"], layout["root"] / targets["md"]
                            else:
                                jp = layout["root"] / layout["paths"]["json"]
                                md = layout["root"] / layout["paths"]["md"]
                                transcript.save(data, jp, md)
                            question = {"id": "r1", "question": "Проверьте", "options": [
                                {"label": "Другое", "decision": {"text": "другое"}}]}
                            with mock.patch.object(review_pack, "cut_clip",
                                                   side_effect=lambda _s, t1, _t2, _out: t1):
                                pack = review_pack.build(jp, [question])
                            review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"], "option": 0}],
                                                    "local", trusted_actor="human")
                            core.apply_answers_files(jp, md)
                            before = transcript.load(jp)
                            self.assertEqual(before["segments"][0]["words"][0]["w"], "другое")
                            review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"],
                                                          "text": text, "heard": mode == "voice"}],
                                                    "local", trusted_actor="human")
                            result = core.apply_answers_files(jp, md)
                            after = transcript.load(jp)
                            self.assertEqual(after["segments"], before["segments"])
                            self.assertEqual(after["review"], before["review"])
                            self.assertEqual(after["edits"], before["edits"])
                            self.assertFalse(result["reverted"])
                            self.assertIn("слово", result["free_text"]["left_to_human"][0]["why"])
                        finally:
                            if store:
                                store.close()

    # Ответы на карточку «тест» у Глеба с рекомендацией «другое» (вариант 1) и вариантом 2 «третье».
    # Закрытый список отрицаний: печатью и голосом — стенограмма не меняется, место открыто.
    NEGATIVES = ("неверно", "не верно", "неправильно", "не правильно", "не согласен", "не согласна", "не согласны",
                 "это неверно", "нет, неверно", "не согласен с этим", "я не согласен", "совершенно неверно",
                 "совсем не верно", "нет-нет, неверно", "абсолютно неправильно", "неверно: слово",
                 "не согласен: слово", "НЕВЕРНО", "Неверно!")

    def test_negative_answers_keep_place_open_in_both_modes(self):
        for backend in ("legacy_files", "sqlite"):
            for text in self.NEGATIVES:
                for heard in (False, True):
                    with self.subTest(backend=backend, answer=text, voice=heard), \
                            self.answer_files(backend) as (jp, md):
                        result = self.answer_card(jp, md, [(text, heard)])
                        self.assertEqual(self.card_state(jp), (["тест"], ["Глеб"], "open"))
                        self.assertIn("слово", result["free_text"]["left_to_human"][0]["why"])

    def test_negative_then_correct_answer_in_both_modes(self):
        for backend in ("legacy_files", "sqlite"):
            for negative in ("неверно", "совсем не верно", "неверно: слово"):
                for heard in (False, True):
                    with self.subTest(backend=backend, answer=negative, voice=heard, then="text"), \
                            self.answer_files(backend) as (jp, md):
                        self.answer_card(jp, md, [(negative, heard), ("слово", False)])
                        self.assertEqual(self.card_state(jp)[0], ["слово"])
                    with self.subTest(backend=backend, answer=negative, voice=heard, then="option"), \
                            self.answer_files(backend) as (jp, md):
                        self.answer_card(jp, md, [(negative, heard), {"option": 1}])
                        self.assertEqual(self.card_state(jp)[0], ["третье"])

    # Остальные ответы — как до закрытого списка отрицаний: (ответ, итог печатью, итог голосом),
    # итог — (слова карточки, говорящий, статус места). «наверно» голосом больше не подтверждает рекомендацию.
    # PR 6б (В8): короткие ответы — по таблице answer_phrases.tsv, голос = текст: «верно», «как есть» — как есть;
    # «да», «нет» на другом слове и похожее на оценку («согласен») — подсказка, место открыто.
    AS_BEFORE = (
        ("согласен", ("тест", "open"), ("тест", "open")),
        ("согласна", ("тест", "open"), ("тест", "open")),
        ("принимаю", ("принимаю", "resolved"), ("принимаю", "resolved")),
        ("рекомендую", ("тест", "open"), ("тест", "open")),
        ("ну да", ("ну да", "resolved"), ("ну да", "resolved")),
        ("да, это я", ("да, это я", "resolved"), ("да, это я", "resolved")),
        ("верно", ("тест", "resolved"), ("тест", "resolved")),
        ("Верно.", ("тест", "resolved"), ("тест", "resolved")),
        ("да, верно", ("тест", "resolved"), ("тест", "resolved")),
        ("да", ("тест", "open"), ("тест", "open")),
        ("ага", ("тест", "open"), ("тест", "open")),
        ("угу", ("тест", "open"), ("тест", "open")),
        ("всё правильно", ("тест", "open"), ("тест", "open")),
        ("да, согласен", ("тест", "open"), ("тест", "open")),
        ("я согласен", ("тест", "open"), ("тест", "open")),
        ("согласен с этим", ("тест", "open"), ("тест", "open")),
        ("совершенно верно", ("тест", "open"), ("тест", "open")),
        ("абсолютно правильно", ("тест", "open"), ("тест", "open")),
        ("да, всё верно", ("тест", "open"), ("тест", "open")),
        ("принимаю рекомендацию", ("тест", "open"), ("тест", "open")),
        ("да, как рекомендовано", ("тест", "open"), ("тест", "open")),
        ("как предложено", ("тест", "open"), ("тест", "open")),
        ("так", ("тест", "open"), ("тест", "open")),
        ("как есть", ("тест", "resolved"), ("тест", "resolved")),
        ("да да", ("да да", "resolved"), ("да да", "resolved")),
        ("мы согласны", ("тест", "open"), ("тест", "open")),
        ("Борис Иванович", ("тест", "human"), ("тест", "human")),
        ("Анна Иванова", ("тест", "human"), ("тест", "human")),
        ("Вера Николаевна", ("тест", "human"), ("тест", "human")),
        ("нет", ("тест", "open"), ("тест", "open")),
        ("не совсем верно", ("тест", "open"), ("тест", "open")),
        ("вряд ли верно", ("тест", "open"), ("тест", "open")),
        ("нет, это не так", ("нет, это не так", "resolved"), ("нет, это не так", "resolved")),
        ("да, но неверно", ("да, но неверно", "resolved"), ("да, но неверно", "resolved")),
        ("неверное слово", ("неверное слово", "resolved"), ("неверное слово", "resolved")),
        ("он верно сказал", ("тест", "open"), ("тест", "open")),
        ("вариант 2: да", ("вариант 2: да", "resolved"), ("третье", "resolved")),
        ("проверено", ("проверено", "resolved"), ("проверено", "resolved")),
        ("Анна: слово", ("тест", "human"), ("тест", "human")),
        ("наверно", ("наверно", "resolved"), ("наверно", "resolved")),
    )

    def test_other_answers_behave_as_before_in_both_modes(self):
        for backend in ("legacy_files", "sqlite"):
            for text, typed, voice in self.AS_BEFORE:
                for heard, (word, status) in ((False, typed), (True, voice)):
                    with self.subTest(backend=backend, answer=text, voice=heard), \
                            self.answer_files(backend) as (jp, md):
                        self.answer_card(jp, md, [(text, heard)])
                        words, speakers, got = self.card_state(jp)
                        self.assertEqual((" ".join(words), speakers, got), (word, ["Глеб"], status))

    def test_maps_worker_output_and_preserves_runtime_metadata(self):
        source = self.worker_result()
        before = copy.deepcopy(source)
        recording_id = str(uuid.uuid4())

        mapped_id, transcript, review = recordings.snapshots_from_worker(source, recording_id)
        _, replay_transcript, replay_review = recordings.snapshots_from_worker(source, recording_id)

        self.assertEqual(source, before)
        self.assertEqual(mapped_id, recording_id)
        self.assertEqual(transcript["quality"]["runtime"], {
            "speaker_labels": ["speaker-1"],
            "timing": {"total_sec": 4},
            "group": "demo",
        })
        self.assertNotIn("review", transcript)
        self.assertEqual(review["review"][0]["id"], "r1")
        self.assertEqual(len(transcript["segments"][0]["segment_id"]), 36)
        self.assertEqual(len(transcript["segments"][0]["words"][0]["word_id"]), 36)
        self.assertEqual(transcript, replay_transcript)
        self.assertEqual(review, replay_review)

    def test_existing_unique_ids_are_preserved(self):
        data = self.worker_result()
        data["segments"][0]["segment_id"] = "segment-a"
        data["segments"][0]["words"][0]["word_id"] = "word-a"

        _, transcript, _ = recordings.snapshots_from_worker(data)

        self.assertEqual(transcript["segments"][0]["segment_id"], "segment-a")
        self.assertEqual(transcript["segments"][0]["words"][0]["word_id"], "word-a")

    def test_unknown_worker_field_is_rejected_not_dropped(self):
        data = self.worker_result()
        data["unmapped_feature"] = {"important": True}

        with self.assertRaisesRegex(ValueError, "unsupported worker fields"):
            recordings.snapshots_from_worker(data)

    def test_invalid_worker_segments_are_rejected(self):
        data = self.worker_result()
        data["segments"] = [{"start": 2, "end": 1, "words": []}]

        with self.assertRaises(ValueError):
            recordings.snapshots_from_worker(data)

    def test_document_layout_uses_legacy_stem_and_tag_names(self):
        layout = recordings.document_layout(Path("C:/synthetic/meeting.wav"), "draft-x")
        self.assertEqual(layout["root"], Path("C:/synthetic/meeting.rus-transcriber"))
        self.assertEqual(layout["paths"], {
            "md": "meeting.draft-x.transcript.md",
            "json": "meeting.draft-x.transcript.json",
            "manifest": "meeting.draft-x.manifest.json",
        })

    def test_default_state_database_path_is_declared_not_created(self):
        self.assertEqual(settings.paths.STATE_DB, settings.paths.DATA_DIR / "state.sqlite3")
        self.assertFalse(settings.paths.STATE_DB.exists())

    def test_explicit_store_initializes_and_keeps_identity_in_sqlite(self):
        with tempfile.TemporaryDirectory(prefix="p2-identity-", dir=os.environ["TEMP"]) as temp:
            path = Path(temp) / "state.sqlite3"
            with mock.patch.object(settings.paths, "STATE_DB", path):
                store = recordings.initialize_store(display_name="")
            try:
                original = store.current_identity()
                self.assertEqual(original["display_name"], "")
                context = store.capture_context(original["installation_id"], "human")
                self.assertEqual(context["author_ref"]["user_name_snapshot"], "")
                self.assertEqual(settings.set_user_display_name(store, "Имя / ник"), "Имя / ник")
                self.assertEqual(settings.user_display_name(store), "Имя / ник")
                self.assertFalse(settings.paths.CONFIG_FILE.exists())
            finally:
                store.close()

            reopened = recordings.open_store(path)
            try:
                current = reopened.current_identity()
                self.assertEqual(current["installation_id"], original["installation_id"])
                self.assertEqual(current["user_id"], original["user_id"])
                self.assertEqual(current["display_name"], "Имя / ник")
                self.assertEqual(context["author_ref"]["user_name_snapshot"], "")
                with self.assertRaises(FileExistsError):
                    recordings.initialize_store(path, display_name="не заменять")
            finally:
                reopened.close()

    def test_invalid_initial_identity_fails_before_creating_database(self):
        with tempfile.TemporaryDirectory(prefix="p2-invalid-identity-", dir=os.environ["TEMP"]) as temp:
            path = Path(temp) / "state.sqlite3"
            with self.assertRaises(ValueError):
                recordings.initialize_store(path, display_name=None)
            self.assertFalse(path.exists())

    def test_ambiguous_active_installations_are_not_guessed(self):
        with tempfile.TemporaryDirectory(prefix="p2-ambiguous-identity-", dir=os.environ["TEMP"]) as temp:
            store = recordings.initialize_store(Path(temp) / "state.sqlite3")
            try:
                other_user = store.create_user("Другой")
                store.bind_installation(str(uuid.uuid4()), other_user)
                with self.assertRaises(storage.StorageError):
                    store.current_identity()
            finally:
                store.close()

    def test_synthetic_tags_share_root_and_keep_independent_targets(self):
        with tempfile.TemporaryDirectory(prefix="p2-recording-", dir=os.environ["TEMP"]) as temp:
            base = Path(temp)
            store = storage.initialize(base / "state.sqlite3")
            try:
                user_id = store.create_user("Тест")
                installation_id = str(uuid.uuid4())
                store.bind_installation(installation_id, user_id)
                context = store.capture_context(installation_id, "model")
                source = base / "meeting.wav"
                source.write_bytes(b"synthetic audio placeholder")
                draft_layout = recordings.document_layout(source, "draft")
                final_layout = recordings.document_layout(source, "final")
                root = draft_layout["root"]
                root.mkdir()
                root_id = store.get_or_register_root(root)
                self.assertEqual(store.get_or_register_root(final_layout["root"]), root_id)
                nested_root = root / "nested"
                nested_root.mkdir()
                with self.assertRaises(storage.Conflict):
                    store.get_or_register_root(nested_root)
                draft_id, draft_operation = str(uuid.uuid4()), str(uuid.uuid4())
                final_id, final_operation = str(uuid.uuid4()), str(uuid.uuid4())

                def commit(layout, recording_id, operation_id):
                    data = self.worker_result()
                    data["file"] = str(source)
                    return recordings.commit_worker_result(
                        store, data, str(source), root_id, context,
                        recording_id=recording_id, operation_id=operation_id,
                        generated_at="2026-11-16T00:00:00+00:00",
                        document_paths=layout["paths"])

                draft = commit(draft_layout, draft_id, draft_operation)
                replay = commit(draft_layout, draft_id, draft_operation)
                final = commit(final_layout, final_id, final_operation)
                self.assertEqual(replay, draft)
                self.assertNotEqual(draft["recording_id"], final["recording_id"])
                self.assertEqual(draft["document_sync"], "pending")
                self.assertEqual(final["document_sync"], "pending")
                with self.assertRaises(storage.Conflict):
                    commit(draft_layout, str(uuid.uuid4()), str(uuid.uuid4()))
                self.assertEqual(
                    {item["recording_id"] for item in recordings.find_recordings(store, source)},
                    {draft_id, final_id})
                self.assertEqual(
                    {row["relative_path"] for row in store.db.execute(
                        "SELECT relative_path FROM file_locations WHERE recording_id=?", (draft_id,))},
                    set(draft_layout["paths"].values()) |
                    {f"history/{draft_id}/document-1/transcript.md",
                     f"history/{draft_id}/document-1/transcript.json",
                     f"history/{draft_id}/document-1/manifest.json"})
                self.assertEqual(
                    {row["relative_path"] for row in store.db.execute(
                        "SELECT relative_path FROM file_locations WHERE recording_id=?", (final_id,))},
                    set(final_layout["paths"].values()) |
                    {f"history/{final_id}/document-1/transcript.md",
                     f"history/{final_id}/document-1/transcript.json",
                     f"history/{final_id}/document-1/manifest.json"})

                completed = documents.materialize_pending(store, root_id, root)
                self.assertEqual(set(completed), {(draft_id, 1), (final_id, 1)})
                for relative in (*draft_layout["paths"].values(), *final_layout["paths"].values()):
                    self.assertTrue((root / relative).is_file())
                self.assertTrue(documents.verify_files(store, root_id, root)["consistent"])
                loaded = recordings.load_recording(store, draft_id)
                self.assertEqual(loaded["file"], str(source))
                self.assertEqual(loaded["source_revision"], 1)
                self.assertEqual(loaded["document_sync"], "synced")
                binding = recordings.document_binding(store, draft_id)
                self.assertEqual(binding["paths"], draft_layout["paths"])
                self.assertEqual(recordings.resolve_document(store, loaded["json"])["recording_id"], draft_id)
                self.assertIsNone(recordings.resolve_document(store, root / "history" / draft_id /
                                                               "document-1" / "transcript.json"))
                self.assertEqual(loaded["timing"], {"total_sec": 4})
                self.assertEqual(loaded["vocab_suggestions"],
                                 [{"to": "Кварцлайн", "from": "Кварцлайнъ", "card": "r1"}])
                self.assertEqual(store.get_document(draft_id, 1)["json"][:1], b"{")

                human_context = store.capture_context(installation_id, "human")
                loaded["segments"][0]["words"][0]["w"] = "исправлено"
                loaded["segments"][0]["text"] = "исправлено"
                changed = recordings.commit_recording_data(
                    store, draft_id, loaded, {"source_revision": 1, "review_revision": 1},
                    str(uuid.uuid4()), human_context, root_id, "2026-11-16T00:01:00+00:00")
                self.assertEqual(changed["document_revision"], 2)
                self.assertEqual(changed["source_revision"], 2)
                self.assertEqual(changed["review_revision"], 2)
                self.assertEqual(
                    documents.materialize_pending(store, root_id, root),
                    [(draft_id, 2)])
                self.assertTrue((root / draft_layout["paths"]["md"]).is_file())
                self.assertTrue((root / f"history/{draft_id}/document-2/transcript.md").is_file())
                self.assertTrue(documents.verify_files(store, root_id, root)["consistent"])
                self.assertEqual(recordings.load_recording(store, draft_id)["segments"][0]["text"],
                                 "исправлено")
                loaded["segments"][0]["words"][0]["w"] = "устаревшая правка"
                loaded["segments"][0]["text"] = "устаревшая правка"
                with self.assertRaises(storage.Conflict):
                    recordings.commit_recording_data(
                        store, draft_id, loaded, {"source_revision": 1, "review_revision": 1},
                        str(uuid.uuid4()), human_context, root_id, "2026-11-16T00:02:00+00:00")
                self.assertEqual(store.get_current(draft_id)["transcript"]["revision"], 2)

                import migrate
                with mock.patch.object(migrate, "run"):
                    import server
                import confirm
                md_target = root / draft_layout["paths"]["md"]
                md_target.write_text("external document edit", encoding="utf-8")
                recordings.materialize_recording(store, draft_id)
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"):
                    document_conflicts = core.list_document_conflicts(draft_id)
                    self.assertEqual(len(document_conflicts), 1)
                    conflict_id = document_conflicts[0]["conflict_id"]
                    with mock.patch.object(confirm, "require", side_effect=confirm.NotConfirmed("отказ")):
                        self.assertIn("отказ", asyncio.run(server.resolve_document_conflict(draft_id, conflict_id)))
                    self.assertEqual(md_target.read_text(encoding="utf-8"), "external document edit")
                    with mock.patch.object(confirm, "require", return_value="dialog") as approved:
                        response = asyncio.run(server.resolve_document_conflict(draft_id, conflict_id))
                    self.assertTrue(json.loads(response)["materialized"])
                    approved.assert_called_once()
                self.assertTrue(documents.verify_files(store, root_id, root)["consistent"])

                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"):
                    renamed, unknown = transcript.rename(
                        recordings.load_recording(store, draft_id)["json"],
                        recordings.load_recording(store, draft_id)["markdown"],
                        {"speaker-1": "Тестовый"})
                self.assertEqual(unknown, [])
                self.assertEqual(renamed["speakers"], ["Тестовый"])
                current = store.get_current(draft_id)
                self.assertEqual(current["transcript"]["revision"], 3)
                self.assertEqual(current["review"]["revision"], 3)
                operation = store.db.execute(
                    "SELECT context_json FROM operations WHERE operation_id=("
                    "SELECT operation_id FROM transcript_revisions WHERE recording_id=? AND revision=3)",
                    (draft_id,)).fetchone()
                self.assertEqual(storage._loads(operation["context_json"])["actor_kind"], "human")
                self.assertTrue(documents.verify_files(store, root_id, root)["consistent"])

                json_path = recordings.load_recording(store, draft_id)["json"]
                import voices
                selection = {}
                profile = {"speakers": {}, "sources": {}}
                self.assertEqual(voices.learn(profile, root, None, None, only=[json_path],
                                              selection_report=selection), [])
                self.assertEqual(selection["sqlite_source_not_supported"], 1)
                self.assertEqual(profile["speakers"], {})
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"), \
                        mock.patch.object(core, "job_json", return_value=(json_path, None)):
                    with self.assertRaisesRegex(core.CoreError, "пока не поддерживается"):
                        core.preview_voice_learning("synthetic-job")
                item = {"id": "r1", "question": "Проверьте", "options": [
                    {"label": "Оставить", "decision": {}},
                    {"label": "Другое", "decision": {"text": "другое"}}]}
                def cut_with_concurrent_change(_source, t1, _t2, _out):
                    Path(_out).write_bytes(b"new clip")
                    state = recordings.read_review_pack_state(json_path)
                    state["interaction_events"].append({"type": "synthetic"})
                    recordings.write_review_pack_state(json_path, state)
                    return t1

                def cut_clip(_source, t1, _t2, output):
                    Path(output).write_bytes(b"original clip")
                    return t1

                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"), \
                        mock.patch.object(review_pack, "cut_clip", side_effect=cut_with_concurrent_change):
                    with self.assertRaisesRegex(storage.Conflict, "changed while building"):
                        review_pack.build(json_path, [item])
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"), \
                        mock.patch.object(review_pack, "cut_clip", side_effect=cut_clip):
                    pack = review_pack.build(json_path, [item])
                    self.assertEqual(review_pack.manifest(json_path)["version"], pack["version"])
                    self.assertFalse((review_pack.pack_dir(json_path) / "manifest.json").exists())
                    import migrate
                    with mock.patch.object(migrate, "run"):
                        import server
                    with mock.patch.object(server, "_job_json", return_value=(json_path, None)):
                        info = json.loads(server.review_pack_info("synthetic-job"))
                    self.assertEqual(info["manifest_storage"], "sqlite")
                    self.assertNotIn("manifest.json", info["files"])
                    matches = json.loads(server.find_recording_results(str(source)))
                    self.assertIn(draft_id, [row["recording_id"] for row in matches])
                    independent = json.loads(server.read_recording_result(draft_id))
                    self.assertEqual(independent["recording_id"], draft_id)
                    self.assertEqual(independent["segments"][0]["text"], "исправлено")
                    clip = review_pack.manifest(json_path)["items"][0]["clip"]
                    with mock.patch.object(review_pack, "cut_clip", side_effect=cut_with_concurrent_change):
                        with self.assertRaises(storage.Conflict):
                            review_pack.build(json_path, [item])
                    self.assertEqual(review_pack.manifest(json_path)["version"], pack["version"])
                    self.assertEqual((review_pack.pack_dir(json_path) / clip).read_bytes(), b"original clip")
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", 0))
                        port = sock.getsockname()[1]
                    status_path = base / "page-status.json"
                    env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(base), TEMP=str(base), TMP=str(base))
                    process = subprocess.Popen(
                        [sys.executable, "-B", "review_page.py", json_path, "--status", str(status_path),
                         "--port", str(port)],
                        cwd=str(Path(__file__).resolve().parent.parent), env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    try:
                        url = None
                        for _ in range(100):
                            if status_path.exists():
                                url = json.loads(status_path.read_text(encoding="utf-8")).get("url")
                                if url:
                                    break
                            if process.poll() is not None:
                                break
                            time.sleep(0.05)
                        self.assertIsNotNone(url)
                        address = urlsplit(url)
                        conn = http.client.HTTPConnection("127.0.0.1", address.port, timeout=10)
                        try:
                            conn.request("GET", address.path + "manifest.json")
                            response = conn.getresponse()
                            self.assertEqual(response.status, 200)
                            self.assertEqual(json.loads(response.read())["version"], pack["version"])
                            conn.request("GET", address.path + clip)
                            response = conn.getresponse()
                            self.assertEqual(response.status, 200)
                            self.assertEqual(response.read(), b"original clip")
                        finally:
                            conn.close()
                    finally:
                        process.terminate()
                        process.wait(timeout=10)
                    accepted = review_pack.add_answers(
                        json_path, [{"id": "r1", "version": pack["version"], "option": 0}], "local")
                    self.assertEqual(accepted["answered"], 1)
                    self.assertFalse((review_pack.pack_dir(json_path) / "answers.json").exists())
                    self.assertEqual(review_pack.answers_state(json_path)["answers"]["r1"]["option"], 0)
                    stale = review_pack.answers_state(json_path)
                    fresh = review_pack.answers_state(json_path)
                    fresh["interaction_events"].append({"type": "test"})
                    recordings.write_review_pack_state(json_path, fresh)
                    stale["answers"] = {}
                    with self.assertRaises(storage.Conflict):
                        recordings.write_review_pack_state(json_path, stale)
                    with mock.patch.object(recordings, "materialize_recording", return_value=[]):
                        with self.assertRaisesRegex(RuntimeError, "materialization is pending"):
                            transcript.rename(json_path, "unused", {"Тестовый": "Другое"})
                    self.assertEqual(store.get_current(draft_id)["transcript"]["revision"], 4)
                    recordings.materialize_recording(store, draft_id)
                self.assertTrue(documents.verify_files(store, root_id, root)["consistent"])

                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"):
                    saved = conspect.save_linked(json_path, "Краткий", "Обсуждается «тест».")
                expected_cj, expected_cm = conspect.path_for(json_path, "Краткий")
                self.assertEqual(saved["conspect_json"], str(expected_cj))
                self.assertEqual(saved["conspect_md"], str(expected_cm))
                stored_conspects = store.list_conspects(draft_id)
                self.assertEqual([x["name"] for x in stored_conspects], ["Краткий"])
                self.assertEqual(stored_conspects[0]["payload"]["recording_id"], draft_id)
                self.assertEqual(stored_conspects[0]["context"]["actor_kind"], "unknown")
                saved_row = stored_conspects[0]
                with self.assertRaisesRegex(ValueError, "does not match"):
                    store.save_conspect(
                        draft_id, saved_row["conspect_id"], saved_row["name"],
                        saved_row["source_revision"], saved_row["current_revision"],
                        saved_row["payload"], b"{}", saved_row["md"],
                        saved_row["json_relative"], saved_row["md_relative"], root_id,
                        str(uuid.uuid4()), human_context)
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"):
                    with self.assertRaisesRegex(storage.Conflict, "already exists"):
                        conspect.write_files(json_path, [
                            (expected_cj, stored_conspects[0]["json"]),
                            (expected_cm, stored_conspects[0]["md"]),
                        ], create=True)
                    with self.assertRaisesRegex(storage.Conflict, "already exists"):
                        conspect.save_linked(json_path, "Краткий", "Новый неподтверждённый текст.")
                self.assertEqual(store.list_conspects(draft_id)[0]["current_revision"], 1)
                collision_paths = documents.paths_for_basename("collision")
                collision_paths["md"] = stored_conspects[0]["md_relative"]
                with self.assertRaises(storage.Conflict):
                    recordings.commit_worker_result(
                        store, self.worker_result(), str(source), root_id, context,
                        recording_id=str(uuid.uuid4()), operation_id=str(uuid.uuid4()),
                        generated_at="2026-11-16T00:00:00+00:00",
                        document_paths=collision_paths)
                self.assertEqual(store.db.execute(
                    "SELECT state FROM conspect_outbox WHERE conspect_id=? AND revision=1",
                    (stored_conspects[0]["conspect_id"],)).fetchone()["state"], "synced")
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"):
                    rerendered = conspect.rerender(json_path, "Краткий")
                self.assertIn("Краткий", rerendered)
                self.assertEqual(store.list_conspects(draft_id)[0]["current_revision"], 2)
                self.assertTrue((Path(saved["conspect_json"])).is_file())
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"), \
                        mock.patch.object(core, "_done_json", return_value={"json": json_path}):
                    confirmation = core.describe_conspect_confirmation("synthetic-job", "Краткий")
                    confirmed = core.confirm_conspect_analysis(confirmation)
                self.assertEqual(confirmed["analysis_revision"], 4)
                self.assertEqual(store.list_conspects(draft_id)[0]["current_revision"], 3)
                self.assertEqual(store.list_conspects(draft_id)[0]["context"]["actor_kind"], "human")
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"), \
                        mock.patch.object(conspect, "_write_files_raw", side_effect=OSError("synthetic disk error")):
                    with self.assertRaises(OSError):
                        conspect.rerender(json_path, "Краткий")
                pending = store.list_conspects(draft_id)[0]
                self.assertEqual(pending["current_revision"], 4)
                self.assertEqual(store.db.execute(
                    "SELECT state FROM conspect_outbox WHERE conspect_id=? AND revision=4",
                    (pending["conspect_id"],)).fetchone()["state"], "pending")
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"):
                    self.assertEqual(
                        recordings.materialize_conspects(store, draft_id),
                        [(pending["conspect_id"], 4)])
                replace = conspect._replace_bytes
                calls = []

                def interrupt_replace(path, value):
                    calls.append(path)
                    if len(calls) == 3:
                        raise OSError("synthetic interruption")
                    return replace(path, value)

                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"), \
                        mock.patch.object(conspect, "_replace_bytes", side_effect=interrupt_replace):
                    with self.assertRaisesRegex(OSError, "synthetic interruption"):
                        conspect.rerender(json_path, "Краткий")
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"):
                    self.assertEqual(recordings.materialize_conspects(store, draft_id),
                                     [(pending["conspect_id"], 5)])
                Path(saved["conspect_md"]).write_text("external edit", encoding="utf-8")
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"):
                    with self.assertRaises(storage.StorageError):
                        conspect.rerender(json_path, "Краткий")
                self.assertEqual(store.db.execute(
                    "SELECT state FROM conspect_outbox WHERE conspect_id=? AND revision=6",
                    (pending["conspect_id"],)).fetchone()["state"], "conflict")
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"):
                    with self.assertRaisesRegex(storage.Conflict, "unresolved conspect conflict"):
                        conspect.rerender(json_path, "Краткий")
                self.assertEqual(store.list_conspects(draft_id)[0]["current_revision"], 6)
                backup_path = base / "backup.sqlite3"
                store.backup(backup_path)
                restored_root = base / "restored"
                restored_root.mkdir()
                backup = storage.open_store(backup_path)
                try:
                    location = backup.db.execute(
                        "SELECT conspect_id, kind, canonical_path FROM conspect_file_locations LIMIT 1").fetchone()
                    backup.db.execute(
                        "UPDATE conspect_file_locations SET canonical_path=? WHERE conspect_id=? AND kind=?",
                        (location["canonical_path"] + ".invalid", location["conspect_id"], location["kind"]))
                    with self.assertRaisesRegex(storage.StorageError, "conspect file target mismatch"):
                        storage.validate_copy(backup_path)
                    backup.db.execute(
                        "UPDATE conspect_file_locations SET canonical_path=? WHERE conspect_id=? AND kind=?",
                        (location["canonical_path"], location["conspect_id"], location["kind"]))
                    backup.rebind_restored_copy({root_id: restored_root}, str(uuid.uuid4()), user_id)
                    self.assertEqual(backup.db.execute(
                        "SELECT locked FROM restore_guard WHERE id=1").fetchone()[0], 0)
                    locations = backup.db.execute(
                        "SELECT canonical_path, confirmed_revision FROM conspect_file_locations").fetchall()
                    self.assertTrue(all(Path(row["canonical_path"]).is_relative_to(restored_root)
                                        and row["confirmed_revision"] is None for row in locations))
                    documents.materialize_pending(backup, root_id, restored_root)
                    completed = recordings.materialize_conspects(backup, draft_id)
                    self.assertEqual(completed, [(pending["conspect_id"], i) for i in range(1, 6)])
                    self.assertEqual(backup.db.execute(
                        "SELECT state FROM conspect_outbox WHERE conspect_id=? AND revision=6",
                        (pending["conspect_id"],)).fetchone()["state"], "conflict")
                    self.assertTrue((restored_root / stored_conspects[0]["md_relative"]).is_file())
                finally:
                    backup.close()
                conflict = store.db.execute(
                    "SELECT * FROM conspect_conflicts WHERE conspect_id=? AND revision=6",
                    (pending["conspect_id"],)).fetchone()
                self.assertEqual(conflict["observed_bytes"], b"external edit")
                import confirm
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"), \
                        mock.patch.object(confirm, "require", side_effect=confirm.NotConfirmed("отказ")):
                    response = asyncio.run(server.resolve_conspect_conflict(draft_id, conflict["conflict_id"]))
                self.assertIn("отказ", response)
                self.assertIsNone(store.db.execute(
                    "SELECT resolution_operation_id FROM conspect_conflicts WHERE conflict_id=?",
                    (conflict["conflict_id"],)).fetchone()[0])
                store.resolve_conspect_conflict(conflict["conflict_id"], str(uuid.uuid4()), human_context)
                Path(saved["conspect_md"]).write_text("new external edit", encoding="utf-8")
                self.assertEqual(recordings.materialize_conspects(store, draft_id), [])
                new_conflict = store.db.execute(
                    "SELECT * FROM conspect_conflicts WHERE conspect_id=? AND revision=6 "
                    "AND resolution_operation_id IS NULL", (pending["conspect_id"],)).fetchone()
                self.assertEqual(new_conflict["observed_bytes"], b"new external edit")
                with mock.patch.object(settings.paths, "STATE_DB", base / "state.sqlite3"), \
                        mock.patch.object(confirm, "require", return_value="dialog") as approved:
                    response = asyncio.run(server.resolve_conspect_conflict(draft_id, new_conflict["conflict_id"]))
                self.assertTrue(json.loads(response)["materialized"])
                approved.assert_called_once()
                self.assertEqual(Path(saved["conspect_md"]).read_bytes(),
                                 store.get_conspect(pending["conspect_id"], 6)["md"])
            finally:
                store.close()


class BackendSelectionTests(unittest.TestCase):
    def test_sqlite_selection_never_falls_back_for_unregistered_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "state.sqlite3"
            document = root / "unregistered.transcript.json"
            document.write_text("{}", encoding="utf-8")
            with mock.patch.object(settings, "recording_backend", return_value="sqlite"), \
                    mock.patch.object(settings.paths, "STATE_DB", db):
                with self.assertRaises(storage.StorageError):
                    recordings.open_for_document(document)
                store = recordings.initialize_store(db)
                store.close()
                with self.assertRaises(storage.Conflict):
                    recordings.open_for_document(document)

    def test_worker_main_creates_sqlite_result_with_synthetic_pipeline(self):
        import worker
        import voices
        import tmpfiles

        class FakeEmbedder:
            seconds = 0

            def __init__(self, _pipe):
                pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "state.sqlite3"
            src = root / "audio.wav"
            src.write_bytes(b"synthetic placeholder")
            store = recordings.initialize_store(db, display_name="Инициатор")
            try:
                context = recordings.operation_context(store, "model")
            finally:
                store.close()
            status = root / "status.json"
            status.write_text(json.dumps({"recording_context": context}), encoding="utf-8")
            data = WorkerSnapshotTests().worker_result()
            data["review"][0].pop("word_ids")
            data["_timing"] = {}
            args = ["worker.py", str(src), "--status", str(status), "--storage-backend", "sqlite", "--no-voices"]
            with mock.patch.object(settings.paths, "STATE_DB", db), \
                    mock.patch.object(sys, "argv", args), \
                    mock.patch.dict(os.environ, {"HF_TOKEN": "synthetic-token"}), \
                    mock.patch.object(worker, "wait_slot"), \
                    mock.patch.object(tmpfiles, "sweep_stale"), \
                    mock.patch.object(worker, "load_config", return_value={"pipeline": "words"}), \
                    mock.patch.object(worker, "resolve_vocab_details",
                                      return_value={"path": None, "vocab": {}, "group": None, "report": {}}), \
                    mock.patch.object(worker, "load_audio", return_value=[0] * worker.SR), \
                    mock.patch.object(worker, "diarize",
                                      return_value=([(0.0, 1.0, "s1")], {}, None, None)), \
                    mock.patch.object(voices, "Embedder", FakeEmbedder), \
                    mock.patch.object(worker, "run_words", return_value=data):
                worker.main()
            result = json.loads(status.read_text(encoding="utf-8"))
            self.assertEqual(result["state"], "done")
            self.assertEqual(result["document_sync"], "synced")
            with mock.patch.object(settings.paths, "STATE_DB", db), \
                    mock.patch.object(review_pack, "cut_clip", side_effect=lambda _s, t1, _t2, _out: t1):
                jp, md = result["json"], result["markdown"]
                transcript.rename(jp, md, {"speaker-1": "Тестовый"})
                transcript.apply_review(jp, md, [{"id": "r1", "status": "human"}])
                checked = recordings.open_store(db)
                try:
                    self.assertEqual(checked.get_current(result["recording_id"])["review"]["context"]
                                     ["actor_kind"], "model")
                finally:
                    checked.close()
                pack = review_pack.build(jp, [{"id": "r1", "question": "Проверьте",
                                               "options": [{"label": "Оставить", "decision": {}}]}])
                review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"], "option": 0}],
                                        "local", trusted_actor="human")
                self.assertEqual(core.apply_answers_files(jp, md)["applied_from_answers"], 1)
                review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"], "kind": "revert"}],
                                        "local", trusted_actor="human")
                self.assertEqual(core.apply_answers_files(jp, md)["reverted"], ["r1"])
                conspect.save_linked(jp, "Краткий", "Обсуждается «тест».")
                with mock.patch.object(core, "job_status", side_effect=AssertionError("job is unavailable")):
                    independent = core.read_recording_result(result["recording_id"])
                    reference = independent["recording_ref"]
                    self.assertIn("тест", core.read_transcript(reference))
                    self.assertEqual(tuple(map(os.path.normcase, core.job_json(reference))),
                                     tuple(map(os.path.normcase, (jp, md))))
                    self.assertIsInstance(core.review_items(reference), dict)
                    core.reopen_review(reference, ["r1"])
                    import migrate
                    with mock.patch.object(migrate, "run"):
                        import server
                    import confirm
                    self.assertIn("локальную карточку",
                                  server.edit_span(reference, 0.2, 0.5, "другое", by="human"))
                    with mock.patch.object(confirm, "require", side_effect=confirm.NotConfirmed("отказ")):
                        denied = asyncio.run(server.rename_speakers(reference, {"speaker-1": "Другое имя"}))
                    self.assertIn("отказ", denied)
                    self.assertEqual(transcript.load(jp)["speaker_names"]["speaker-1"], "Тестовый")
                    def change_during_confirmation(*_args):
                        concurrent = recordings.open_store(db)
                        try:
                            current = concurrent.get_current(result["recording_id"])
                            review = copy.deepcopy(current["review"]["payload"])
                            review.setdefault("interaction_events", []).append({"type": "concurrent"})
                            concurrent.commit_change(
                                result["recording_id"],
                                {"source_revision": current["transcript"]["revision"],
                                 "review_revision": current["review"]["revision"]},
                                str(uuid.uuid4()), recordings.operation_context(concurrent, "human"), review=review)
                        finally:
                            concurrent.close()
                        return "dialog"

                    with mock.patch.object(confirm, "require", side_effect=change_during_confirmation,
                                           return_value="dialog"):
                        stale = asyncio.run(server.rename_speakers(reference, {"speaker-1": "Другое имя"}))
                    self.assertIn("изменилась", stale)
                    self.assertEqual(transcript.load(jp)["speaker_names"]["speaker-1"], "Тестовый")
                    with mock.patch.object(confirm, "require", return_value="dialog") as approved:
                        renamed = asyncio.run(server.rename_speakers(reference, {"speaker-1": "Другое имя"}))
                    self.assertEqual(json.loads(renamed)["mapping"]["speaker-1"], "Другое имя")
                    approved.assert_called_once()
                self.assertEqual(independent["recording_id"], result["recording_id"])
            store = recordings.open_store(db)
            try:
                saved = recordings.load_recording(store, result["recording_id"])
                self.assertEqual(saved["file"], str(src))
                self.assertEqual(store.get_revision(result["recording_id"], "transcript", 1)["context"]
                                 ["initiated_by"]["user_name_snapshot"], "Инициатор")
            finally:
                store.close()

    def test_answer_application_and_undo_commit_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "state.sqlite3"
            audio = root / "sample.wav"
            audio.write_bytes(b"synthetic")
            layout = recordings.document_layout(audio)
            layout["root"].mkdir()
            store = recordings.initialize_store(db, display_name="Первый")
            try:
                root_id, targets = recordings.prepare_new_layout(store, layout["root"], layout["paths"])
                data = WorkerSnapshotTests().worker_result()
                data["file"] = str(audio)
                data["review"][0].pop("word_ids")
                rid = str(uuid.uuid4())
                recordings.commit_worker_result(
                    store, data, audio, root_id, recordings.operation_context(store, "model"),
                    recording_id=rid, operation_id=str(uuid.uuid4()), generated_at="2026-11-16T00:00:00Z",
                    document_paths=targets)
                recordings.materialize_recording(store, rid)
                jp = layout["root"] / targets["json"]
                md = layout["root"] / targets["md"]
                item = {"id": "r1", "question": "Проверьте", "options": [
                    {"label": "Оставить", "decision": {}},
                    {"label": "Другое", "decision": {"text": "другое"}}]}
                with mock.patch.object(settings.paths, "STATE_DB", db), \
                        mock.patch.object(review_pack, "cut_clip",
                                          side_effect=lambda _s, t1, _t2, _out: t1):
                    pack = review_pack.build(jp, [item])
                    review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"], "option": 0}], "local")
                    answer = review_pack.answers_state(jp)["answers"]["r1"]
                    self.assertEqual(answer["actor_kind"], "unknown")
                    self.assertIsNone(answer["author_ref"]["user_id"])
                    self.assertEqual(answer["initiated_by"]["user_name_snapshot"], "Первый")
                    document = store.get_document(rid, recordings.document_binding(store, rid)["document_revision"])
                    public_answer = json.loads(document["json"])["review"]["answers"]["r1"]
                    self.assertEqual(public_answer["option"], 0)
                    self.assertEqual(public_answer["initiated_by"]["user_name_snapshot"], "Первый")
                    with self.assertRaisesRegex(core.CoreError, "неизвестного происхождения"):
                        core.apply_answers_files(jp, md)
                    import migrate
                    with mock.patch.object(migrate, "run"):
                        import server
                    import confirm
                    with mock.patch.object(confirm, "require", side_effect=confirm.NotConfirmed("отказ")):
                        self.assertIn("отказ", asyncio.run(server.request_review_answer_verification(rid, ["r1"])))
                    self.assertNotIn("confirmation_context", review_pack.answers_state(jp)["answers"]["r1"])
                    with mock.patch.object(confirm, "require", return_value="dialog") as approved:
                        response = asyncio.run(server.request_review_answer_verification(rid, ["r1"]))
                    self.assertEqual(json.loads(response)["confirmed"], ["r1"])
                    approved.assert_called_once()
                    settings.set_user_display_name(store, "Второй")
                    before = store.get_current(rid)
                    with mock.patch.object(recordings, "commit_recording_data", side_effect=OSError("failure")):
                        with self.assertRaises(OSError):
                            core.apply_answers_files(jp, md)
                    self.assertEqual(store.get_current(rid), before)
                    original_apply = core._apply_answers_files_locked

                    def concurrent_change(path, markdown):
                        report = original_apply(path, markdown)
                        current = store.get_current(rid)
                        review = copy.deepcopy(current["review"]["payload"])
                        review.setdefault("interaction_events", []).append({"type": "concurrent"})
                        store.commit_change(
                            rid, {"source_revision": current["transcript"]["revision"],
                                  "review_revision": current["review"]["revision"]},
                            str(uuid.uuid4()), recordings.operation_context(store, "human"), review=review)
                        return report

                    with mock.patch.object(core, "_apply_answers_files_locked", side_effect=concurrent_change):
                        with self.assertRaises(storage.Conflict):
                            core.apply_answers_files(jp, md)
                    concurrent = store.get_current(rid)
                    self.assertEqual(concurrent["transcript"]["payload"], before["transcript"]["payload"])
                    self.assertEqual(concurrent["review"]["payload"]["interaction_events"][-1]["type"], "concurrent")
                    self.assertNotIn("r1", review_pack.answers_state(jp)["applied_answers"])
                    result = core.apply_answers_files(jp, md)
                    self.assertEqual(result["applied_from_answers"], 1)
                    current = store.get_current(rid)
                    self.assertEqual(current["transcript"]["operation_id"], current["review"]["operation_id"])
                    document = store.get_document(rid, recordings.document_binding(store, rid)["document_revision"])
                    self.assertEqual(document["operation_id"], current["review"]["operation_id"])
                    self.assertIn("r1", review_pack.answers_state(jp)["applied_answers"])
                    applied_state = review_pack.answers_state(jp)
                    self.assertEqual(applied_state["answers"]["r1"]["actor_kind"], "unknown")
                    self.assertEqual(applied_state["answers"]["r1"]["confirmation_context"]["author_ref"]
                                     ["user_name_snapshot"], "Первый")
                    self.assertEqual(applied_state["application_events"][-1]["context"]["author_ref"]
                                     ["user_name_snapshot"], "Второй")
                    self.assertEqual(transcript.load(jp)["review"][0]["status"], "resolved")
                    review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"], "option": 1}],
                                            "local", trusted_actor="human")
                    human_answer = review_pack.answers_state(jp)["answers"]["r1"]
                    self.assertEqual(human_answer["actor_kind"], "human")
                    self.assertEqual(human_answer["author_ref"]["user_name_snapshot"], "Второй")
                    changed = core.apply_answers_files(jp, md)
                    # Применённый ответ другим не заменяется: новый ждёт отмены прежнего.
                    self.assertEqual(changed["pending_new"], ["r1"])
                    self.assertEqual(transcript.load(jp)["segments"][0]["words"][0]["w"], "тест")
                    review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"],
                                                  "kind": "revert"}], "local", trusted_actor="human")
                    undone = core.apply_answers_files(jp, md)
                    self.assertEqual(undone["reverted"], ["r1"])
                    self.assertEqual(transcript.load(jp)["review"][0]["status"], "open")
                    self.assertEqual(transcript.load(jp)["segments"][0]["words"][0]["w"], "тест")
                    self.assertNotIn("r1", review_pack.answers_state(jp)["applied_answers"])
                    review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"],
                                                  "kind": "free", "text": "новое"}], "local", trusted_actor="human")
                    free = core.apply_answers_files(jp, md)
                    self.assertIn("r1", free["free_text"]["successful_ids"])
                    self.assertEqual(transcript.load(jp)["segments"][0]["words"][0]["w"], "новое")
                    review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"],
                                                  "kind": "revert"}], "local", trusted_actor="human")
                    self.assertEqual(core.apply_answers_files(jp, md)["reverted"], ["r1"])
                    self.assertEqual(transcript.load(jp)["segments"][0]["words"][0]["w"], "тест")
                    review_pack.add_answers(jp, [{"id": "r1", "version": pack["version"],
                                                  "option": 0}], "local", trusted_actor="human")
                    with mock.patch.object(recordings, "materialize_recording", return_value=[]):
                        with self.assertRaisesRegex(core.CoreError, "документа не завершено"):
                            core.apply_answers_files(jp, md)
                    committed = store.get_current(rid)
                    self.assertIn("r1", review_pack.answers_state(jp)["applied_answers"])
                    retried = core.apply_answers_files(jp, md)
                    self.assertEqual(retried["document_sync"], "synced")
                    self.assertEqual(store.get_current(rid), committed)
                    with mock.patch.object(recordings, "refresh_conspects",
                                           side_effect=OSError("synthetic failure")):
                        with self.assertRaisesRegex(core.CoreError, "конспекты не обновлены"):
                            core.apply_answers_files(jp, md)
                    self.assertEqual(core.apply_answers_files(jp, md)["document_sync"], "synced")
                    self.assertEqual(store.get_current(rid), committed)
            finally:
                store.close()

    def test_installer_includes_recordings_dependencies(self):
        installer = (Path(__file__).resolve().parent.parent / "install.ps1").read_text(encoding="utf-8-sig")
        for filename in ("recordings.py", "storage.py", "documents.py"):
            self.assertIn(f'"{filename}"', installer)

    def test_sqlite_backend_requires_explicit_initialized_store(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "state.sqlite3"
            config = root / "config.json"
            with mock.patch.object(settings.paths, "STATE_DB", db_path), \
                    mock.patch.object(settings.paths, "CONFIG_FILE", config), \
                    mock.patch.object(settings.paths, "DATA_DIR", root):
                with self.assertRaises(storage.StorageError):
                    settings.set_recording_backend("sqlite")
                self.assertEqual(settings.recording_backend(), "legacy_files")
                store = recordings.initialize_store(db_path, display_name="Синтетический")
                store.close()
                settings.set_recording_backend("sqlite")
                self.assertEqual(settings.recording_backend(), "sqlite")

    def test_worker_adapter_commits_before_materializing(self):
        import worker

        class Status:
            def __init__(self):
                self.values = {}

            def set(self, **values):
                self.values.update(values)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "state.sqlite3"
            audio = root / "sample.wav"
            audio.write_bytes(b"synthetic audio")
            store = recordings.initialize_store(db_path, display_name="Тест")
            store.close()
            layout = recordings.document_layout(audio)
            layout["root"].mkdir()
            store = recordings.open_store(db_path)
            try:
                root_id, paths = recordings.prepare_new_layout(store, layout["root"], layout["paths"])
                context = recordings.operation_context(store, "model")
                settings.set_user_display_name(store, "Новое имя")
            finally:
                store.close()
            status = Status()
            data = WorkerSnapshotTests().worker_result()
            data["file"] = str(audio)
            with mock.patch.object(settings.paths, "STATE_DB", db_path):
                with mock.patch.object(recordings, "materialize_recording", return_value=[]):
                    with self.assertRaisesRegex(RuntimeError, "materialization is pending"):
                        worker._save_worker_result(
                            data, audio, status, {"root_id": root_id, "paths": paths, "context": context},
                            layout["root"] / paths["json"], layout["root"] / paths["md"])
                recording_id = status.values["recording_id"]
                saved = core.sync_recording_documents(recording_id)
                store = recordings.open_store(db_path)
                try:
                    current = store.get_current(recording_id)
                    self.assertEqual(current["transcript"]["revision"], 1)
                    self.assertEqual(current["transcript"]["context"]["initiated_by"]["user_name_snapshot"], "Тест")
                    self.assertEqual(saved["document_sync"], "synced")
                    self.assertEqual(status.values["recording_id"], saved["recording_id"])
                    self.assertTrue(documents.verify_files(store, root_id, layout["root"])["consistent"])
                finally:
                    store.close()


if __name__ == "__main__":
    unittest.main()
