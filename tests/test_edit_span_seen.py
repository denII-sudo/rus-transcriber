"""Отметка «видел» у неизменённых слов правки отрезка человека (РАР-6.1, вариант В) — в обоих режимах хранения.

Человек правит отрезок: изменённые слова получают его подтверждение (`by`, `edited`, `provenance`), а неизменённые —
только отметку `word["seen"]` вне `provenance`. Отметка не делает слово «тронутым человеком» (`human_touched`),
не закрывает место, не считается прослушиванием и не идёт в голосовые профили. Агент слово с отметкой править
может: слово попадает в `seen_changed` отчёта и журнала, у изменённого слова отметка снимается.

Различие режимов при показе: в `legacy_files` отметка и числа видны в `*.transcript.json` и в `.md`; в `sqlite`
отметка лежит в снимке стенограммы в БД, а публичный документ её не показывает (`documents.py` не расширяется).
Синтетика: три слова Глеба (w1 «тест», w2 «слово», w3 «дальше») и слово Веры (w4).
"""
import contextlib
import json
import unittest
from unittest import mock

import test_human_confirmation as hc  # noqa: E402

import transcript  # noqa: E402
import voices  # noqa: E402

BACKENDS = hc.BACKENDS
SEEN_1 = {"by": "human", "edit": "span@0.20", "via": "edit_span"}


ORIGINAL = hc.result_data


def only_place_changed(test, before, after, rid):
    """PR 13б: противоречие ничего лишнего не меняет. Реплики и журнал правок прежние; в review изменилось
    только место rid — предложение (`contradiction`), причина «contradiction» и статус."""
    test.assertEqual(after["segments"], before["segments"])
    test.assertEqual(after.get("edits"), before.get("edits"))

    def strip(review):
        out = []
        for item in review or []:
            item = dict(item)
            if item.get("id") == rid:
                item.pop("contradiction", None)
                item.pop("status", None)
                item["reasons"] = [r for r in item.get("reasons") or [] if r != "contradiction"]
            out.append(item)
        return out
    test.assertEqual(strip(after.get("review")), strip(before.get("review")))
    test.assertTrue(next(i for i in after["review"] if i["id"] == rid)["contradiction"]["proposals"])


def many_words():
    data = ORIGINAL()
    data["segments"][0]["words"].append({"w": "дальше", "s": 1.2, "e": 1.5, "word_id": "w3"})
    data["segments"][0]["end"] = 1.5
    data["segments"][0]["text"] = "тест слово дальше"
    data["segments"].append({"start": 1.7, "end": 2.0, "label": "speaker-2", "speaker": "Вера", "text": "тоже",
                             "words": [{"w": "тоже", "s": 1.7, "e": 2.0, "word_id": "w4"}]})
    return data


@contextlib.contextmanager
def files(backend):
    with mock.patch.object(hc, "result_data", many_words), hc.files(backend) as (jp, md, version):
        yield jp, md


def seen_of(jp, wid):
    return hc.word(jp, wid).get("seen")


def human_sees_all(jp, md):
    """Человек просмотрел отрезок и ничего не поменял; места разбора остаются открытыми."""
    transcript.edit_span(jp, md, 0.2, 1.5, "тест слово дальше", by="human", resolve=False)


class HumanMarksUnchangedWords(unittest.TestCase):
    def test_changed_word_is_confirmed_unchanged_are_only_seen(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                transcript.edit_span(jp, md, 0.2, 1.5, "тест словно дальше", by="human")
                w1, w2, w3 = (hc.word(jp, x) for x in ("w1", "w2", "w3"))
                self.assertEqual(w1["seen"], SEEN_1)
                self.assertEqual(w3["seen"], SEEN_1)
                for w in (w1, w3):                    # прежние отметки не тронуты, подтверждения нет
                    self.assertNotIn("by", w)
                    self.assertNotIn("edited", w)
                    self.assertNotIn("provenance", w)
                    self.assertFalse(transcript.human_touched(w))
                self.assertNotIn("seen", w2)
                self.assertEqual((w2["by"], w2["edited"]), ("human", "span@0.20"))
                self.assertTrue(transcript.human_touched(w2))

    def test_agent_edit_marks_only_changed_words_and_no_seen(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                transcript.edit_span(jp, md, 0.2, 1.5, "тест словно дальше", by="claude")
                for wid in ("w1", "w2", "w3"):
                    self.assertIsNone(seen_of(jp, wid))
                self.assertEqual(hc.word(jp, "w2")["by"], "claude")

    def test_seen_does_not_close_places_confirm_listening_or_feed_voices(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                transcript.edit_span(jp, md, 0.2, 0.7, "тест слово", by="human")    # w2 начинается на краю отрезка
                self.assertEqual(seen_of(jp, "w2"), SEEN_1)
                self.assertEqual(hc.item(jp, "r1")["status"], "resolved")            # целиком в отрезке
                r2, r4 = hc.item(jp, "r2"), hc.item(jp, "r4")
                self.assertEqual((r2["status"], r4["status"]), ("open", "open"))      # сама отметка места не закрывает
                self.assertNotIn("listening", r4)
                self.assertFalse(voices._word_human(hc.word(jp, "w2")))

    def test_repeated_call_keeps_one_mark(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                human_sees_all(jp, md)
                human_sees_all(jp, md)
                self.assertEqual(seen_of(jp, "w1"), SEEN_1)
                self.assertEqual(sorted(hc.word(jp, "w1")), ["e", "s", "seen", "w", "word_id"])

    def test_second_human_edit_replaces_mark_and_changing_removes_it(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                human_sees_all(jp, md)
                transcript.edit_span(jp, md, 0.7, 1.5, "слово дальше", by="human", resolve=False)
                self.assertEqual(seen_of(jp, "w1")["edit"], "span@0.20")
                self.assertEqual(seen_of(jp, "w2")["edit"], "span@0.70")
                self.assertEqual(seen_of(jp, "w3")["edit"], "span@0.70")
                transcript.edit_span(jp, md, 0.7, 1.0, "иное", by="human", resolve=False)
                w2 = hc.word(jp, "w2")
                self.assertNotIn("seen", w2)
                self.assertEqual((w2["w"], w2["by"], w2["edited"]), ("иное", "human", "span@0.70"))
                self.assertEqual(w2["provenance"]["text"]["edit"], "span@0.70")
                self.assertEqual(seen_of(jp, "w3")["edit"], "span@0.70")

    def test_human_card_answer_over_seen_word_drops_mark_without_list(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                human_sees_all(jp, md)
                _, report = transcript.apply_review(jp, md, [{"id": "r1", "text": "другое"}], by="human")
                self.assertNotIn("seen_changed", report)
                w1 = hc.word(jp, "w1")
                self.assertNotIn("seen", w1)
                self.assertEqual(w1["by"], "human")
                self.assertEqual(seen_of(jp, "w2"), SEEN_1)       # соседние слова другого места не затронуты


class AgentChangesSeenWords(unittest.TestCase):
    def listed(self, w, wid="w1", t=0.2, edit="span@0.20"):
        return {"word_id": wid, "t": t, "w": w, "seen_edit": edit}

    def test_edit_span_agent_lists_seen_words_it_changed(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                human_sees_all(jp, md)
                self.assertFalse(transcript.human_touched(hc.word(jp, "w1")))
                _, rep = transcript.edit_span(jp, md, 0.2, 1.5, "тесто слово дальше", by="claude")
                self.assertEqual(rep["seen_changed"], [self.listed("тест")])
                self.assertEqual(transcript.load(jp)["edits"][-1]["seen_changed"], [self.listed("тест")])
                self.assertNotIn("seen", hc.word(jp, "w1"))               # увиденного текста больше нет
                self.assertEqual(seen_of(jp, "w2"), SEEN_1)                # неизменённые слова отметку сохранили
                self.assertEqual(seen_of(jp, "w3"), SEEN_1)
                self.assertEqual(hc.word(jp, "w1")["by"], "claude")

    def test_agent_edit_that_leaves_seen_words_alone_lists_nothing(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                human_sees_all(jp, md)
                _, rep = transcript.edit_span(jp, md, 1.7, 2.0, "тоже", speaker="Глеб", by="claude")
                self.assertEqual(rep["seen_changed"], [])
                self.assertNotIn("seen_changed", transcript.load(jp)["edits"][-1])
                self.assertEqual(seen_of(jp, "w1"), SEEN_1)

    def test_speaker_change_of_seen_word_counts_as_change(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                human_sees_all(jp, md)
                _, rep = transcript.edit_span(jp, md, 1.2, 1.5, "дальше", speaker="Вера", by="claude")
                self.assertEqual(rep["seen_changed"], [self.listed("дальше", "w3", 1.2)])
                self.assertNotIn("seen", hc.word(jp, "w3"))

    def test_deleted_seen_word_is_listed(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                human_sees_all(jp, md)
                _, rep = transcript.edit_span(jp, md, 1.2, 1.5, "", by="claude")
                self.assertEqual(rep["seen_changed"], [self.listed("дальше", "w3", 1.2)])

    def test_apply_review_agent_changes_seen_word(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                human_sees_all(jp, md)
                _, report = transcript.apply_review(jp, md, [{"id": "r1", "text": "другое"}], by="claude")
                self.assertEqual(report["applied"], 1)
                self.assertEqual(report["seen_changed"], [self.listed("тест")])
                self.assertEqual(transcript.load(jp)["edits"][-1]["seen_changed"], [self.listed("тест")])
                self.assertNotIn("seen", hc.word(jp, "w1"))
                self.assertEqual(hc.word(jp, "w1")["w"], "другое")

    def test_apply_review_agent_speaker_change_and_one_bad_item_does_not_stop_others(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                human_sees_all(jp, md)
                _, report = transcript.apply_review(
                    jp, md, [{"id": "нет такого"}, {"id": "r1", "speaker": "Вера"}], by="claude")
                self.assertEqual(report["unknown"], ["нет такого"])
                self.assertEqual(report["seen_changed"], [self.listed("тест")])
                self.assertEqual(seen_of(jp, "w2"), SEEN_1)

    def test_apply_word_edits_agent_replace_lists_seen_word(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                human_sees_all(jp, md)
                _, rep = transcript.apply_word_edits(
                    jp, md, [{"op": "replace", "t": 0.7, "w": "слово", "text": "слова", "card": "x1"}], by="claude")
                self.assertEqual(rep["seen_changed"], [self.listed("слово", "w2", 0.7)])

    def test_refused_agent_edit_writes_nothing_and_keeps_marks(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                human_sees_all(jp, md)
                transcript.apply_review(jp, md, [{"id": "r2", "text": "новое"}], by="human")   # w2 — слово человека
                before = transcript.load(jp)
                # PR 13б: не ошибка, а противоречие — слова и отметки «видел» не меняются, у места — предложение
                _, rep = transcript.edit_span(jp, md, 0.2, 1.5, "тесто новое дальше", by="claude")
                self.assertIs(rep["written"], False)
                only_place_changed(self, before, transcript.load(jp), rep["contradiction"])
                self.assertEqual(seen_of(jp, "w1"), SEEN_1)

    def test_failed_save_leaves_no_mark_and_retry_works(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md):
                before = (transcript.load(jp), jp.read_bytes())
                with mock.patch.object(transcript, "save", side_effect=RuntimeError("сбой записи")):
                    with self.assertRaises(RuntimeError):
                        human_sees_all(jp, md)
                self.assertEqual((transcript.load(jp), jp.read_bytes()), before)
                human_sees_all(jp, md)
                self.assertEqual(seen_of(jp, "w1"), SEEN_1)


class WhatTheFilesShow(unittest.TestCase):
    def test_legacy_files_show_mark_and_counts(self):
        with files("legacy_files") as (jp, md):
            transcript.edit_span(jp, md, 0.2, 1.5, "тест словно дальше", by="human")
            on_disk = json.loads(jp.read_text(encoding="utf-8"))
            words = [w for s in on_disk["segments"] for w in s["words"]]
            self.assertEqual([w.get("seen") for w in words], [SEEN_1, None, SEEN_1, None])
            data = transcript.load(jp)
            self.assertIn("(человек, правка отрезка; изменено 1, видел 2)",
                          transcript.render_md({**data, "review_done": True}))
            self.assertEqual(hc.item(jp, "r1")["note"],
                             "решено правкой отрезка span@0.20 (человек; изменено 1, видел 2)")
            self.assertEqual(data["edits"][-1]["changed"], 1)
            self.assertEqual(data["edits"][-1]["seen"], 2)

    def test_agent_line_names_how_many_changed_words_were_seen(self):
        with files("legacy_files") as (jp, md):
            human_sees_all(jp, md)
            transcript.edit_span(jp, md, 0.2, 1.5, "тесто слово дальше", by="claude")
            line = transcript.render_md({**transcript.load(jp), "review_done": True})
            self.assertIn("(агент, правка отрезка; изменено 1, из них видел человек 1)", line)

    def test_no_counts_when_nothing_was_only_seen(self):
        with files("legacy_files") as (jp, md):
            transcript.edit_span(jp, md, 0.2, 0.5, "тесть", by="human")
            self.assertEqual(hc.item(jp, "r1")["note"], "решено правкой отрезка span@0.20 (человек)")
            self.assertIn("(человек, правка отрезка)",
                          transcript.render_md({**transcript.load(jp), "review_done": True}))

    def test_sqlite_keeps_mark_in_snapshot_but_not_in_public_document(self):
        with files("sqlite") as (jp, md):
            transcript.edit_span(jp, md, 0.2, 1.5, "тест словно дальше", by="human")
            self.assertEqual(seen_of(jp, "w1"), SEEN_1)                          # снимок в БД хранит
            self.assertEqual(transcript.load(jp)["edits"][-1]["seen"], 2)
            text = jp.read_text(encoding="utf-8")
            self.assertNotIn('"seen"', text)                                      # публичный документ не расширяется
            self.assertNotIn("edit_span", text)
            self.assertEqual(json.loads(text)["transcript"]["segments"][0]["words"][0]["word_id"], "w1")


if __name__ == "__main__":
    unittest.main()
