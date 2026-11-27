"""К-В4: отмена и замена внесённого ответа по участку места (снимок формата 3) — в обоих режимах хранения.

Синтетическая стенограмма «тест слово» у выдуманного Глеба: одна реплика, две карточки в ней (r1 — «тест»,
r2 — «слово»). Прежде снимок отмены хранил отпечаток всей стенограммы: после любого другого изменения ответ
нельзя было ни отменить, ни заменить, а из пакета ответов одного «Применить» — все, кроме последнего внесённого
(КВ4-1). Теперь снимок находит свой участок по содержимому: правки других мест (в том числе в той же реплике)
отмене не мешают, правка того же места — отказ, как прежде. Отмена помечает отменёнными только свои правки журнала
(КВ4-2). Замена, где новый ответ не внёсся, при хранении в базе возвращает прежний ответ (Ф223-2, решение владельца
1А), строка об этом хранится под карточкой до нового ответа на ней (Ф223-4, решение 2А).
"""
import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import test_answer_outcomes as base  # noqa: E402  (тот же sandbox и те же правила исходов)
import test_cards_ui_buttons as ui  # noqa: E402  (страница в Node)
import test_card_findings_behavior as fcb  # noqa: E402  (К-23: ноутбук и облако расходились)

import core  # noqa: E402
import review_pack  # noqa: E402
import transcript  # noqa: E402

BACKENDS = base.BACKENDS


class BatchThenUndoEarlier(base.AnswerOutcomeCase):
    """КВ4-1: два ответа одним «Применить» — заменить и отменить можно оба, не только последний внесённый."""

    def batch(self, jp, md, version):
        self.save(jp, version, aid="r1", option=0)
        self.save(jp, version, aid="r2", option=0)
        rep = self.apply(jp, md)
        self.assertEqual(self.outcomes(rep), {"r1": "applied", "r2": "applied"})
        self.assertEqual(self.words(jp), ["другое", "новое"])

    def test_replace_first_applied_answer_of_batch(self):
        # «Применить» вносит ответы с конца записи: r2 внесён первым, r1 — после него в той же реплике
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                self.batch(jp, md, version)
                self.save(jp, version, aid="r2", option=1)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep), {"r1": "applied", "r2": "applied"})
                self.assertEqual(rep["replaced"], ["r2"])
                self.assertEqual(self.words(jp), ["другое", "иное"])

    def test_revert_first_applied_answer_keeps_the_other(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                self.batch(jp, md, version)
                self.save(jp, version, aid="r2", kind="revert")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r2"], "reverted")
                self.assertEqual(self.words(jp), ["другое", "слово"])
                data = transcript.load(jp)
                self.assertEqual({r["id"]: r["status"] for r in data["review"]}, {"r1": "resolved", "r2": "open"})
                # КВ4-2: отменена только своя правка журнала, правка r1 — нет
                self.assertEqual({e["id"]: bool(e.get("reverted")) for e in data["edits"]},
                                 {"r1": False, "r2": True})
                self.assertEqual(next(e for e in data["edits"] if e["id"] == "r1")["by"], "human")
                # и r1 после этого тоже отменяется
                self.save(jp, version, aid="r1", kind="revert")
                self.assertEqual(self.outcomes(self.apply(jp, md))["r1"], "reverted")
                self.assertEqual(self.words(jp), ["тест", "слово"])


class LaterChanges(base.AnswerOutcomeCase):
    """Правка после ответа: другого места — отмена проходит, чужая правка цела; того же места — отказ."""

    def test_later_edit_of_other_place_in_same_reply(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                self.save(jp, version, option=0)
                self.apply(jp, md)
                transcript.edit_span(jp, md, 0.7, 1.0, "ручное", by="human")      # место r2, та же реплика
                self.save(jp, version, kind="revert")
                rep = core.apply_answers_files(jp, md)
                self.check_outcomes(jp, rep)
                self.assertEqual(self.outcomes(rep)["r1"], "reverted")
                self.assertEqual(self.words(jp), ["тест", "ручное"])
                edits = transcript.load(jp)["edits"]
                manual = [e for e in edits if e.get("via") == "edit_span" or e.get("id") != "r1"]
                self.assertTrue(manual, edits)
                self.assertTrue(all(not e.get("reverted") and e.get("by") == "human" for e in manual), edits)
                self.assertTrue(all(e.get("reverted") for e in edits if e.get("id") == "r1"), edits)

    def test_later_edit_of_same_place_refuses(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), base.answer_files(backend) as (jp, md, version):
                self.save(jp, version, option=0)
                self.apply(jp, md)
                transcript.edit_span(jp, md, 0.2, 0.5, "ручное", by="human")
                later = transcript.load(jp)
                self.save(jp, version, kind="revert")
                rep = core.apply_answers_files(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "undo_conflict")
                self.assertEqual(transcript.load(jp)["segments"], later["segments"])
                self.assertEqual(transcript.load(jp)["edits"], later["edits"])


def word(w, s, wid):
    return {"w": w, "s": s, "e": round(s + 0.3, 2), "word_id": wid}


def reply(speaker, *words):
    return {"start": words[0]["s"], "end": words[-1]["e"], "label": speaker, "speaker": speaker,
            "text": " ".join(w["w"] for w in words), "words": list(words)}


def place(rid, t, text, speaker, wid):
    return {"id": rid, "status": "open", "text": text, "t": t, "t1": t, "t2": t, "speaker": speaker,
            "word_ids": [wid]}


def format_two_patches(old, new):
    """Участки снимка формата 2 (до К-В4): номера в new (start/remove) и строки old (restore)."""
    from difflib import SequenceMatcher
    codes = [transcript._row_code(x) for x in old], [transcript._row_code(x) for x in new]
    return [{"start": j1, "remove": j2 - j1, "restore": copy.deepcopy(old[i1:i2])}
            for tag, i1, i2, j1, j2 in SequenceMatcher(None, *codes, autojunk=False).get_opcodes() if tag != "equal"]


class Snapshot(unittest.TestCase):
    """Снимок формата 3 на уровне стенограммы (файловый режим, без пакета карточек)."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=base._TMP))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.jp, self.md = self.dir / "t.json", self.dir / "t.md"

    def save(self, segments, review, edits=()):
        data = base.result_data()
        data.update(segments=copy.deepcopy(segments), review=copy.deepcopy(review), edits=list(edits))
        transcript.save(data, self.jp, self.md)
        return transcript.load(self.jp)

    def answer(self, decision):
        before = transcript.load(self.jp)
        after, rep = transcript.apply_review(self.jp, self.md, [decision], by="human")
        self.assertIn(decision["id"], rep["successful_ids"], rep)
        return transcript.answer_snapshot(before, after)

    def two_replies(self):
        return self.save([reply("Глеб", word("тест", 0.2, "w1"), word("слово", 0.7, "w2")),
                          reply("Вера", word("ещё", 5.0, "w3"), word("раз", 5.5, "w4"))],
                         [place("r1", 0.2, "тест", "Глеб", "w1"), place("r2", 5.0, "ещё", "Вера", "w3")])

    def test_rows_shifted_by_later_answer_are_found_by_content(self):
        """Более поздний ответ выше по тексту разделил реплику — номера строк сдвинулись, отмена находит участок."""
        start = self.two_replies()
        snap = self.answer({"id": "r2", "text": "снова"})
        self.answer({"id": "r1", "speaker": "Вера"})          # «тест» — Вере: реплика Глеба делится на две
        self.assertEqual(len(transcript.load(self.jp)["segments"]), 3)
        split = transcript.load(self.jp)["segments"][:2]
        result = transcript.undo_answer(self.jp, self.md, "r2", snap)
        self.assertTrue(result.get("reverted"), result)
        now = transcript.load(self.jp)
        self.assertEqual(now["segments"][:2], split, "чужое изменение тронуто")
        self.assertEqual(now["segments"][2], start["segments"][1])

    def test_same_reply_edited_later_restores_only_own_words(self):
        start = self.two_replies()
        snap = self.answer({"id": "r1", "text": "другое"})
        transcript.edit_span(self.jp, self.md, 0.7, 0.7, "ручное", by="human")
        result = transcript.undo_answer(self.jp, self.md, "r1", snap)
        self.assertTrue(result.get("reverted"), result)
        seg = transcript.load(self.jp)["segments"][0]
        self.assertEqual([w["w"] for w in seg["words"]], ["тест", "ручное"])
        self.assertEqual(seg["text"], "тест ручное")
        self.assertEqual(seg["words"][0], start["segments"][0]["words"][0])
        self.assertEqual((seg["start"], seg["end"]), (start["segments"][0]["start"], start["segments"][0]["end"]))

    def test_reply_edge_is_restored_only_if_nobody_else_moved_it(self):
        """Ответ сдвинул начало реплики (крайнее слово), позже в той же реплике правили другое слово. Начало
        возвращается, только если его с тех пор никто не менял; иначе — отказ, как прежде."""
        before = self.save([reply("Глеб", word("тест", 0.2, "w1"), word("слово", 0.7, "w2"))],
                           [place("r1", 0.2, "тест", "Глеб", "w1")])
        after = copy.deepcopy(before)
        seg = after["segments"][0]
        seg["words"][0].update(w="другое", s=0.3)
        seg["start"], seg["text"] = 0.3, "другое слово"
        after["review"][0]["status"] = "resolved"
        snap = transcript.answer_snapshot(before, after)
        for moved, expect in ((None, (0.2, ["тест", "ручное"])), (0.25, None)):
            with self.subTest(moved=moved):
                now = copy.deepcopy(after)
                seg = now["segments"][0]
                seg["words"][1]["w"], seg["text"] = "ручное", "другое ручное"
                if moved is not None:
                    seg["start"] = moved
                transcript.save(now, self.jp, self.md)
                now = transcript.load(self.jp)
                result = transcript.undo_answer(self.jp, self.md, "r1", snap)
                if expect is None:
                    self.assertIn("conflict", result)
                    self.assertEqual(transcript.load(self.jp), now)
                else:
                    self.assertTrue(result.get("reverted"), result)
                    seg = transcript.load(self.jp)["segments"][0]
                    self.assertEqual((seg["start"], [w["w"] for w in seg["words"]]), expect)

    def test_ambiguous_after_region_refuses(self):
        """Ф234-1: участок «после» встречается в стенограмме дважды — где вернуть, неясно: отказ, без изменений
        (а не первое вхождение). На настоящих данных почти недостижимо: строки несут время и ID слов."""
        self.two_replies()
        snap = self.answer({"id": "r1", "text": "другое"})
        now = transcript.load(self.jp)["segments"]
        self.assertIsNotNone(transcript._restore_regions(now, snap["segment_patches"], True))
        # Двойник строки «после» (сохранение такую стенограмму не допустит: соседние реплики одного говорящего
        # сливаются) — проверка на уровне возврата участков, до и после своей строки.
        for where in (0, len(now)):
            with self.subTest(where=where):
                rows = copy.deepcopy(now)
                rows.insert(where, copy.deepcopy(now[0]))
                self.assertIsNone(transcript._restore_regions(rows, snap["segment_patches"], True))

    def test_deleted_row_returns_between_its_neighbours(self):
        a, x, b = (reply("Глеб", word("тест", 0.2, "w1")), reply("Вера", word("эээ", 2.0, "w2")),
                   reply("Глеб", word("слово", 4.0, "w3")))
        review = [place("r1", 2.0, "эээ", "Вера", "w2")]
        before = self.save([a, x, b], review)
        after = copy.deepcopy(before)
        after["segments"] = [after["segments"][0], after["segments"][2]]
        after["review"][0]["status"] = "resolved"
        snap = transcript.answer_snapshot(before, after)
        self.assertEqual([(p["after"], len(p["restore"])) for p in snap["segment_patches"]], [([], 1)])
        self.assertNotIn("start", snap["segment_patches"][0])      # Ф234-9: номеров строк в формате 3 нет
        # позже — другое место: соседи участка те же
        after["segments"].append(reply("Вера", word("потом", 9.0, "w9")))
        transcript.save(after, self.jp, self.md)
        after = transcript.load(self.jp)
        self.assertTrue(transcript.undo_answer(self.jp, self.md, "r1", snap).get("reverted"))
        self.assertEqual([s["text"] for s in transcript.load(self.jp)["segments"]], ["тест", "эээ", "слово", "потом"])
        # соседа изменили — где вернуть строку, неясно: отказ
        changed = copy.deepcopy(after)
        changed["segments"][1]["words"][0]["w"] = "слова"
        changed["segments"][1]["text"] = "слова"
        transcript.save(changed, self.jp, self.md)
        self.assertIn("conflict", transcript.undo_answer(self.jp, self.md, "r1", snap))
        self.assertEqual(transcript.load(self.jp)["segments"], changed["segments"])

    def test_old_format_two_snapshot_keeps_old_rule(self):
        """Снимок формата 2 (внесён до К-В4): отменяется, только если стенограмма не менялась вовсе."""
        self.two_replies()
        before = transcript.load(self.jp)
        after, _ = transcript.apply_review(self.jp, self.md, [{"id": "r2", "text": "снова"}], by="human")
        old = {"format": 2, "segment_patches": format_two_patches(before["segments"], after["segments"]),
               "review_patches": format_two_patches(before["review"], after["review"]),
               "edits_before": len(before["edits"]), "after": transcript._undo_fingerprint(after)}
        transcript.edit_span(self.jp, self.md, 0.7, 0.7, "ручное", by="human")    # другое место
        self.assertIn("conflict", transcript.undo_answer(self.jp, self.md, "r2", old))
        transcript.save(after, self.jp, self.md)
        self.assertTrue(transcript.undo_answer(self.jp, self.md, "r2", old).get("reverted"))
        self.assertEqual(transcript.load(self.jp)["segments"], before["segments"])

    def test_format_three_written_with_row_numbers_still_undoes(self):
        """Ф234-9: снимки формата 3, внесённые до снятия номеров строк (#234), несут лишние start/remove —
        отмена их не читает и проходит, в том числе после изменения другого места (номера тогда неверны)."""
        start = self.two_replies()
        snap = self.answer({"id": "r2", "text": "снова"})
        with_numbers = copy.deepcopy(snap)
        for key in ("segment_patches", "review_patches"):
            for patch in with_numbers[key]:
                patch.update(start=999, remove=7)
        transcript.edit_span(self.jp, self.md, 0.7, 0.7, "ручное", by="human")    # другое место
        self.assertTrue(transcript.undo_answer(self.jp, self.md, "r2", with_numbers).get("reverted"))
        self.assertEqual(transcript.load(self.jp)["segments"][1], start["segments"][1])

    def test_confirm_snapshot_of_format_three(self):
        """К-23: подтверждение меняет только строку места — снимок прежний, отмена вернёт место к состоянию до ответа."""
        start = self.two_replies()
        snap = self.answer({"id": "r1", "text": "другое"})
        self.answer({"id": "r2", "text": "снова"})              # позже — другое место
        before = transcript.load(self.jp)
        after = copy.deepcopy(before)
        after["review"][0]["note"] = "подтверждено человеком"
        kept = transcript.confirm_snapshot(snap, before, after)
        self.assertIsNotNone(kept)
        self.assertEqual(kept["segment_patches"], snap["segment_patches"])
        moved = copy.deepcopy(after)
        moved["segments"][0]["words"][1]["w"] = "иначе"
        self.assertIsNone(transcript.confirm_snapshot(snap, before, moved), "слова изменились — снимок не годится")
        transcript.save(after, self.jp, self.md)
        self.assertIn("conflict", transcript.undo_answer(self.jp, self.md, "r1", snap), "старый снимок после подтверждения")
        self.assertTrue(transcript.undo_answer(self.jp, self.md, "r1", kept).get("reverted"))
        now = transcript.load(self.jp)
        self.assertEqual(now["segments"][0], start["segments"][0])
        self.assertEqual([w["w"] for w in now["segments"][1]["words"]], ["снова", "раз"])

    def test_damaged_format_three_is_not_applied(self):
        self.two_replies()
        snap = self.answer({"id": "r1", "text": "другое"})
        now = transcript.load(self.jp)
        for damage in (lambda s: s.update(segment_patches="x"), lambda s: s["segment_patches"][0].update(after=None),
                       lambda s: s.update(edits="x"), lambda s: s.update(edits_before=99)):
            broken = copy.deepcopy(snap)
            damage(broken)
            self.assertIn("conflict", transcript.undo_answer(self.jp, self.md, "r1", broken))
            self.assertEqual(transcript.load(self.jp), now)

    # Ф234-11: запись в базу убирает из строк мест ссылки на исчезнувшие слова (recordings._prune_missing_word_ids),
    # снимок снят до записи. Здесь запись в базу изображается той же функцией над файлом стенограммы.
    def store_as_database(self):
        data = transcript.load(self.jp)
        base.recordings._prune_missing_word_ids(data, data)
        transcript.save(data, self.jp, self.md)
        return transcript.load(self.jp)

    def two_word_place(self):
        return self.save([reply("Глеб", word("тест", 0.2, "w1"), word("слово", 0.7, "w2"), word("ещё", 1.2, "w3"))],
                         [dict(place("r1", 0.2, "тест слово", "Глеб", "w1"), word_ids=["w1", "w2"]),
                          place("r2", 1.2, "ещё", "Глеб", "w3")])

    def test_word_count_answer_undoes_after_database_dropped_its_references(self):
        """Ф234-11: два слова → одно; строка места в снимке ссылается на исчезнувшие w1, w2, в базе ссылок нет."""
        for numbers in (False, True):
            with self.subTest(snapshot_with_row_numbers=numbers):
                start = self.two_word_place()
                snap = self.answer({"id": "r1", "text": "одно"})
                if numbers:                                   # снимок, внесённый #234 (Ф234-9)
                    for key in ("segment_patches", "review_patches"):
                        for patch in snap[key]:
                            patch.update(start=0, remove=1)
                stored = self.store_as_database()
                self.assertNotIn("word_ids", stored["review"][0], "запись в базу убрала ссылки места")
                self.assertTrue(transcript.undo_answer(self.jp, self.md, "r1", snap).get("reverted"))
                now = transcript.load(self.jp)
                self.assertEqual(now["segments"], start["segments"])
                self.assertEqual(now["review"][0]["word_ids"], ["w1", "w2"])
                self.assertEqual(now["review"][0]["status"], "open")
                self.assertEqual(self.store_as_database()["review"][0]["word_ids"], ["w1", "w2"])

    def test_confirm_of_word_count_answer_after_database_dropped_its_references(self):
        """Ф234-11, К-23: подтверждение в следующем «Применить» видит стенограмму из базы — снимок годен."""
        start = self.two_word_place()
        snap = self.answer({"id": "r1", "text": "одно"})
        before = self.store_as_database()
        after = copy.deepcopy(before)
        after["review"][0]["note"] = "подтверждено человеком"
        kept = transcript.confirm_snapshot(snap, before, after)
        self.assertIsNotNone(kept, "подтверждение потеряло снимок")
        transcript.save(after, self.jp, self.md)
        self.assertTrue(transcript.undo_answer(self.jp, self.md, "r1", kept).get("reverted"))
        self.assertEqual(transcript.load(self.jp)["segments"], start["segments"])

    def test_place_rows_equal_without_missing_references_refuse(self):
        """Ф234-11, защита: строки мест без id, одинаковые без ссылок на исчезнувшие слова, — отказ, а не первая."""
        rows = [{"text": "x", "word_ids": ["w1"]}, {"text": "x"}, {"text": "y"}]
        patch = {"restore": [{"text": "x", "word_ids": ["w1", "w2"]}],
                 "after": [{"text": "x", "word_ids": ["w9"]}]}
        segments = [reply("Глеб", word("тест", 0.2, "w5"))]
        self.assertIsNone(transcript._restore_regions(rows, [patch], False, transcript._live_word_ids(segments)))
        rows[1]["text"] = "z"                                 # теперь строка «x» одна — участок найден
        self.assertEqual(transcript._restore_regions(rows, [patch], False, transcript._live_word_ids(segments)),
                         [{"text": "x", "word_ids": ["w1", "w2"]}, {"text": "z"}, {"text": "y"}])
        # ссылка на слово, которое есть в репликах, по-прежнему различает строки
        live_ref = {"restore": [{"text": "y"}], "after": [{"text": "y", "word_ids": ["w5"]}]}
        self.assertIsNone(transcript._restore_regions(rows, [live_ref], False, transcript._live_word_ids(segments)))


class RollbackInDatabase(base.AnswerOutcomeCase):
    """Ф223-2 (решение владельца 1А): замена, где новый ответ не внёсся, возвращает прежний — при хранении в базе.
    Ф223-4 (решение 2А): строка об этом — под карточкой, до нового ответа на ней."""

    def failing(self, jp, md, version, **answer):
        self.save(jp, version, option=0)
        self.apply(jp, md)
        before = transcript.load(jp)
        self.save(jp, version, **answer)

        def fails(json_path, md_path, decisions, **kw):
            return transcript.load(json_path), {"successful_ids": [], "not_found": [d["id"] for d in decisions]}

        def free_fails(json_path, md_path, answers, **kw):
            return {"successful_ids": [], "not_found": list(answers)}

        with mock.patch.object(transcript, "apply_review", side_effect=fails), \
                mock.patch.object(review_pack, "apply_free", side_effect=free_fails):
            rep = core.apply_answers_files(jp, md)
        self.check_outcomes(jp, rep)
        return before, rep

    def test_previous_answer_returns(self):
        for name, answer in (("кнопка", {"option": 1}), ("свой текст", {"text": "пятое"})):
            with self.subTest(case=name), base.answer_files("sqlite") as (jp, md, version):
                before, rep = self.failing(jp, md, version, **answer)
                row = next(r for r in rep["outcomes"] if r["id"] == "r1")
                self.assertEqual(row["outcome"], "not_found")
                self.assertIn(review_pack.REPLACE_ROLLED_BACK, row["detail"])
                self.assertNotIn(review_pack.REPLACED_NOT_APPLIED, row["detail"])
                self.assertEqual(rep["replaced"], [])
                self.assertEqual(rep["replace_rolled_back"], ["r1"])
                self.assertNotIn("voice_profile_warning", rep)
                now = transcript.load(jp)
                for key in ("segments", "review", "edits"):
                    self.assertEqual(now[key], before[key], f"{key}: не вернулось к прежнему ответу")
                state = self.pack_state(jp)
                self.assertEqual(state["applied_answers"]["r1"], ["option", 0, None])
                self.assertIn("r1", state["undo_snapshots"])
                self.assertNotIn("replaced", [e["outcome"] for e in state["application_events"] if e["id"] == "r1"])
                # страница после перезагрузки: исход, признак «прежний на месте» и строка под карточкой
                view = review_pack.answer_states(jp)["r1"]
                self.assertEqual(view["state"], "not_found")
                self.assertTrue(view.get("previous_kept"))
                self.assertIn(review_pack.REPLACE_ROLLED_BACK, view["detail"])
                # следующее «Применить» без помехи заменяет
                rep = self.apply(jp, md)
                self.assertEqual(rep["replaced"], ["r1"])
                self.assertEqual(self.outcomes(rep)["r1"], "applied")
                self.assertNotEqual(self.words(jp)[0], "другое")
                self.assertNotIn("previous_kept", review_pack.answer_states(jp)["r1"])

    def test_evaluation_answer_does_not_say_place_is_open(self):
        """Приёмка: свой ответ «согласен» (оценка, не слова) поверх внесённого — подсказка; прежний возвращён,
        и ни причина, ни подсказка не говорят «место осталось открытым»: оно закрыто прежним ответом."""
        with base.answer_files("sqlite") as (jp, md, version):
            self.save(jp, version, option=0)
            self.apply(jp, md)
            before = self.words(jp)
            self.save(jp, version, text="согласен")
            rep = core.apply_answers_files(jp, md)
            self.check_outcomes(jp, rep)
            row = next(r for r in rep["outcomes"] if r["id"] == "r1")
            self.assertEqual(row["outcome"], "hint")
            self.assertEqual(rep["replace_rolled_back"], ["r1"])
            self.assertEqual(self.words(jp), before)
            view = review_pack.answer_states(jp)["r1"]
            for text in (row["why"], row["detail"], view["why"], view["detail"]):
                self.assertNotIn("открыт", text)
            self.assertIn(review_pack.REPLACE_ROLLED_BACK, view["detail"])
            # Ф234-2: без повторов — о прежнем ответе один раз (строка возврата), в причине и подсказке — ни разу
            for why, detail in ((row["why"], row["detail"]), (view["why"], view["detail"])):
                self.assertEqual((why + " — " + detail).count("прежний ответ"), 1, (why, detail))
            self.assertTrue(row["why"].startswith("не применён: короткий ответ не понят как слова"), row["why"])

    def test_previous_kept_text_drops_open_place_clause(self):
        """Ф234-2: каждая оговорка «место осталось открытым» из причин и подсказок убирается без следа."""
        import freeform
        texts = [review_pack.OUTCOME_REASONS["hint"], review_pack.OUTCOME_REASONS["negative"], freeform.HINT_EVALUATION]
        self.assertTrue(all("место осталось открытым" in t for t in texts))
        got = [review_pack.previous_kept_text(t) for t in texts]
        self.assertEqual(got, ["не применён: короткий ответ не понят как слова — подсказка ниже",
                               "отрицание: правка не применена",
                               freeform.HINT_EVALUATION.replace("место осталось открытым; ", "")])
        self.assertNotIn("  ", "".join(got))
        self.assertEqual(review_pack.previous_kept_text("место осталось открытым"), "прежний ответ на месте")
        self.assertIsNone(review_pack.previous_kept_text(None))

    def test_back_restores_and_new_answer_drops_the_line(self):
        with base.answer_files("sqlite") as (jp, md, version):
            before, _ = self.failing(jp, md, version, option=1)
            self.save(jp, version, text="шестое")                     # новый ответ — строка уходит (решение 2А)
            view = review_pack.answer_states(jp)["r1"]
            self.assertEqual(view["state"], "pending_new")
            self.assertNotIn(review_pack.REPLACE_ROLLED_BACK, view.get("detail", ""))
            self.assertNotIn("previous_kept", view)
            rep = self.save(jp, version, kind="keep")                 # «Вернуть как было»
            self.assertTrue(rep["accepted"][0].get("restored"), rep)
            self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "applied")
            self.assertEqual(transcript.load(jp)["segments"], before["segments"])

    def test_leave_to_human_still_replaces(self):
        """«Оставить человеку» поверх внесённого: место отдано человеку — замена состоялась, возврата нет."""
        with base.answer_files("sqlite") as (jp, md, version):
            self.save(jp, version, option=0)
            self.apply(jp, md)
            self.save(jp, version, kind="human")
            rep = self.apply(jp, md)
            self.assertEqual(rep["replaced"], ["r1"])
            self.assertNotIn("replace_rolled_back", rep)
            self.assertEqual(self.words(jp), ["тест", "слово"])

    def test_file_mode_has_no_rollback(self):
        with base.answer_files("legacy_files") as (jp, md, version):
            _, rep = self.failing(jp, md, version, option=1)
            self.assertNotIn("replace_rolled_back", rep)
            self.assertEqual(rep["replaced"], ["r1"])
            self.assertEqual(self.words(jp), ["тест", "слово"])


class ConfirmedThenUndoKeepsNeighbour(fcb.Base):
    """Ф234-10 (вторая приёмка #234): ответ внесён с пометкой «проверить» (К-23: ноутбук и облако расходились),
    затем внесён ответ на другое место той же реплики, затем человек подтвердил первый и отменил его. До К-В4 отмена
    возвращала реплику целиком и молча снимала второй внесённый ответ; теперь — только свои слова."""

    def test_undo_of_confirmed_answer_keeps_other_place_of_same_reply(self):
        with fcb.files("sqlite") as (jp, md, version):
            rep = fcb.ConflictOfLaptopAndCloud.conflicted(self, jp, md, version)        # r4: «та» → «тот»
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied_check", rep.get("outcomes"))
            self.press(jp, version, "«дальнейшем»", "r3")                               # та же реплика, r3
            rep = self.apply(jp, md)
            self.assertEqual(self.outcomes(rep)["r3"]["outcome"], "applied", rep.get("outcomes"))
            self.press(jp, version, fcb.REPLACE, "r4")                                  # подтверждение r4
            rep = self.apply(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
            self.assertEqual(len(transcript.load(jp)["segments"]), 1, "r3 и r4 — в одной реплике")
            self.save(jp, version, "r4", kind="revert")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "reverted", rep.get("outcomes"))
            self.assertEqual(self.word(jp, "w5")["w"], "та")
            self.assertEqual(self.word(jp, "w3")["w"], "дальнейшем", "отмена r4 сняла ответ r3")
            self.assertEqual(self.place_of(jp, "r3")["status"], "resolved")
            self.assertEqual(self.states(jp)["r3"]["state"], "applied")
            self.assertIn("r3", self.state(jp)["undo_snapshots"])


_result_data = base.result_data            # прежние данные: в WordCountAnswer base.result_data подменяется


def word_count_data():
    """Одна реплика «тест слово ещё раз»; r1 — двухсловное место «тест слово», r2 — «ещё»."""
    words = [word(t, s, f"w{n}") for n, (t, s) in enumerate((("тест", 0.2), ("слово", 0.7), ("ещё", 1.2),
                                                              ("раз", 1.7)), 1)]
    data = _result_data()
    data["segments"] = [dict(reply("Глеб", *words), label="speaker-1")]
    data["review"] = [dict(place("r1", 0.2, "тест слово", "Глеб", "w1"), t2=1.0, word_ids=["w1", "w2"]),
                      dict(place("r2", 1.2, "ещё", "Глеб", "w3"), t2=1.5)]
    return data


WORD_COUNT_QUESTIONS = [
    {"id": "r1", "question": "Проверьте", "recommended": 0, "options": [
        {"label": "Одно слово", "decision": {"text": "одно"}},
        {"label": "Три слова", "decision": {"text": "один два три"}},
        {"label": "Первое слово", "decision": {"text": "тест"}}]},
    {"id": "r2", "question": "Проверьте", "recommended": 0, "options": [
        {"label": "Другое слово", "decision": {"text": "иное"}},
        {"label": "Убрать слово", "decision": {"text": ""}},
        {"label": "Сказала Вера", "decision": {"speaker": "Вера"}}]},
]


class WordCountAnswer(base.AnswerOutcomeCase):
    """Ф234-11 (вторая приёмка #234): ответ, менявший число слов места, при хранении в базе нельзя было ни отменить,
    ни заменить — запись в базу убирает из строки места ссылки на исчезнувшие слова, а участок «после» снимка их
    хранит. Теперь строки мест сравниваются по тому же правилу, что запись."""

    def setUp(self):
        for name, value in (("result_data", word_count_data), ("QUESTIONS", WORD_COUNT_QUESTIONS)):
            patcher = mock.patch.object(base, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def applied(self, jp, md, version, aid, option):
        self.save(jp, version, aid=aid, option=option)
        rep = self.apply(jp, md)
        self.assertEqual(self.outcomes(rep)[aid], "applied", rep.get("outcomes"))

    def reverted(self, jp, md, version, aid, start):
        self.save(jp, version, aid=aid, kind="revert")
        rep = self.apply(jp, md)
        self.assertEqual(self.outcomes(rep)[aid], "reverted", rep.get("undo_conflicts"))
        now = transcript.load(jp)
        self.assertEqual(now["segments"], start["segments"])
        self.assertEqual(now["review"], start["review"])

    def test_answer_changing_word_count_is_undone(self):
        for backend in BACKENDS:
            for option, case in enumerate(("два → одно", "два → три", "два → первое из них")):
                with self.subTest(backend=backend, case=case), base.answer_files(backend) as (jp, md, version):
                    start = transcript.load(jp)
                    self.applied(jp, md, version, "r1", option)
                    self.reverted(jp, md, version, "r1", start)

    def test_answer_changing_word_count_is_replaced(self):
        with base.answer_files("sqlite") as (jp, md, version):
            start = transcript.load(jp)
            self.applied(jp, md, version, "r1", 0)
            self.save(jp, version, aid="r1", option=1)
            rep = self.apply(jp, md)
            self.assertEqual(rep.get("replaced"), ["r1"], rep.get("undo_conflicts"))
            self.assertEqual(self.outcomes(rep)["r1"], "applied")
            self.assertEqual(" ".join(self.words(jp)), "один два три ещё раз")
            self.reverted(jp, md, version, "r1", start)

    def test_word_of_answer_edited_later_refuses(self):
        """Защита прежняя: слово ответа после него правили — отмена отказывает, правка цела."""
        with base.answer_files("sqlite") as (jp, md, version):
            self.applied(jp, md, version, "r1", 0)
            # слово ответа изменено позже (edit_span над словом человека не пишет, а открывает противоречие)
            later = transcript.load(jp)
            later["segments"][0]["words"][0]["w"] = "ручное"
            transcript.save(later, jp, md)
            later = transcript.load(jp)
            self.assertEqual(self.words(jp)[0], "ручное")
            self.save(jp, version, aid="r1", kind="revert")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r1"], "undo_conflict")
            self.assertEqual(transcript.load(jp)["segments"], later["segments"])

    def test_answers_keeping_word_references_undo_as_before(self):
        """Контроль: одно слово → одно, удаление единственного слова места, смена говорящего — как прежде."""
        for backend in BACKENDS:
            for option, case in enumerate(("одно → одно", "удаление слова", "смена говорящего")):
                with self.subTest(backend=backend, case=case), base.answer_files(backend) as (jp, md, version):
                    start = transcript.load(jp)
                    self.applied(jp, md, version, "r2", option)
                    self.reverted(jp, md, version, "r2", start)


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class Page(unittest.TestCase):
    def test_previous_kept_card(self):
        """Новый ответ не внесён, прежний возвращён: заголовок «прежний ответ на месте», причина и строка программы,
        кнопки «Вернуть как было» и «Изменить ответ»; карточка не «без ответа»."""
        result = ui.page(r'''
          const it = {id: "r1", options: [{label: "x"}, {label: "y"}]}, a = {id: "r1", option: 1, ts: "t", source: "local"};
          man = {items: [it], applied_answers: {r1: ["option", 0, null]},
                 answer_states: {r1: {state: "not_found", ts: "t", source: "local", previous_kept: true,
                                      why: "не применён: место карточки не найдено", detail: "прежний ответ возвращён на место: …"}}};
          answers = {r1: a};
          if (cardStatus(it, a) !== "новый ответ не внесён — прежний ответ на месте") throw Error(cardStatus(it, a));
          if (appliedActions(it, a).join() !== "back,edit") throw Error(appliedActions(it, a).join());
          if (!holdsApplied(it, a)) throw Error("прежний ответ не на месте");
          if (needsAnswer(it)) throw Error("карточка с внесённым ответом — «без ответа»");
          const note = programNote(it, a);
          if (!note.startsWith("не применён: место карточки не найдено") || !note.includes("прежний ответ возвращён")) throw Error(note);
          // без признака (старый ответ, другая отметка времени) — прежний вид
          man.answer_states.r1.ts = "другое";
          if (holdsApplied(it, a)) throw Error("признак чужого ответа");
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
