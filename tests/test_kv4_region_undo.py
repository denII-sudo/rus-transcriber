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

    def test_deleted_row_returns_between_its_neighbours(self):
        a, x, b = (reply("Глеб", word("тест", 0.2, "w1")), reply("Вера", word("эээ", 2.0, "w2")),
                   reply("Глеб", word("слово", 4.0, "w3")))
        review = [place("r1", 2.0, "эээ", "Вера", "w2")]
        before = self.save([a, x, b], review)
        after = copy.deepcopy(before)
        after["segments"] = [after["segments"][0], after["segments"][2]]
        after["review"][0]["status"] = "resolved"
        snap = transcript.answer_snapshot(before, after)
        self.assertEqual([(p["remove"], len(p["restore"])) for p in snap["segment_patches"]], [(0, 1)])
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
        old = {"format": 2, "segment_patches": transcript._patches(before["segments"], after["segments"], False),
               "review_patches": transcript._patches(before["review"], after["review"], False),
               "edits_before": len(before["edits"]), "after": transcript._undo_fingerprint(after)}
        transcript.edit_span(self.jp, self.md, 0.7, 0.7, "ручное", by="human")    # другое место
        self.assertIn("conflict", transcript.undo_answer(self.jp, self.md, "r2", old))
        transcript.save(after, self.jp, self.md)
        self.assertTrue(transcript.undo_answer(self.jp, self.md, "r2", old).get("reverted"))
        self.assertEqual(transcript.load(self.jp)["segments"], before["segments"])

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
            self.assertIn("прежний ответ на месте", row["why"])
            self.assertIn(review_pack.REPLACE_ROLLED_BACK, view["detail"])

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
