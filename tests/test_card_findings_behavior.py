"""РАР-6.1, К-а1: учёт и исходы ответов карточек — находки проверки карточек К-5, К-23, К-11, К-10, К-3/К-4,
К-22, К-15, К-20, К-6 (план исправлений v2.1, приватный репозиторий: research/2026-11-25-cards-scenarios).

Решение владельца: новое делается и проверяется только при хранении записей в базе (sqlite). Файловый режим
закреплён одним тестом прежнего поведения (FileModeKeepsPreviousBehaviour). Все данные синтетические.
"""
import json
import shutil
import subprocess
import unittest
from unittest import mock

import test_pause_buttons_short_answers as six     # первым: задаёт папку данных и путь к программе
from test_pause_buttons_short_answers import AS_IS, SILENCE, files
import test_own_answer_closes_disagreement as own  # noqa: E402
import test_contradiction_card as cc  # noqa: E402

core = six.core
review_pack = six.review_pack
transcript = six.transcript
recordings = six.recordings
PROJECT = six.PROJECT

REPLACE = "Заменить на «тот»"
WORDS = "Я прослушал: слова в стенограмме верны"
QUIET = "Я прослушал: здесь тишина"


class Base(six.Case):
    def cloud(self, jp, version, aid, **row):
        """Ответ со страницы в облаке: перенесён import_answers (источник cloud, автор не подтверждён)."""
        six.Case._ts += 1
        answer = {"id": aid, "version": version, "ts": f"2026-11-24T10:{six.Case._ts // 60:02d}:"
                  f"{six.Case._ts % 60:02d}", **six.played_for_listen(jp, aid, row)}
        rep = review_pack.add_answers(jp, [answer], "cloud")
        # ответ облака применяется после отдельного подтверждения человека — как confirm_review_answers
        with recordings.operation_for_document(jp, "human"):
            state = recordings.read_review_pack_state(jp)
            if aid in state["answers"] and state["answers"][aid].get("source") == "cloud":
                state["answers"][aid]["confirmation_context"] = recordings.context_for_document(jp)
                review_pack._write_answers(jp, state)
        return rep

    def option(self, jp, aid, label):
        item = self.card(jp, aid)
        return [o["label"] for o in item["options"]].index(label)

    @staticmethod
    def card(jp, aid):
        return next(i for i in review_pack.manifest(jp)["items"] if i["id"] == aid)

    @staticmethod
    def state(jp):
        return review_pack.answers_state(jp)

    def states(self, jp):
        return review_pack.answer_states(jp)

    def conflicts(self, jp, aid):
        return [c for c in self.state(jp).get("conflicts") or [] if c.get("id") == aid]


class ConflictOfLaptopAndCloud(Base):
    """К-23, К-11: более поздний из расходившихся ответов применён и учтён как применённый; человек его проверяет."""

    def conflicted(self, jp, md, version):
        """Ноутбук «как есть» на r4, позже облако «Заменить на «тот»» → «Применить»."""
        self.press(jp, version, AS_IS["label"], "r4")
        self.cloud(jp, version, "r4", option=self.option(jp, "r4", REPLACE))
        self.assertTrue(self.conflicts(jp, "r4"))
        return core.apply_answers_files(jp, md)

    def test_later_answer_is_applied_and_counted(self):
        with files("sqlite") as (jp, md, version):
            rep = self.conflicted(jp, md, version)
            row = self.outcomes(rep)["r4"]
            self.assertEqual(row["outcome"], "applied_check", rep.get("outcomes"))
            self.assertIn("раньше на это место был другой ответ (1: Оставить как есть)", row.get("detail", ""))
            self.assertEqual(self.word(jp, "w5")["w"], "тот")
            state = self.state(jp)
            self.assertIn("r4", state["applied_answers"])
            self.assertIn("r4", state["undo_snapshots"])
            self.assertEqual(self.place_of(jp, "r4")["status"], "human")
            view = self.states(jp)["r4"]
            self.assertEqual(view["state"], "applied_check")
            self.assertIn("раньше на это место был другой ответ", view.get("detail", ""))

    def test_repeated_apply_keeps_the_check_card_and_undo(self):
        """Ф-К-1: повтор «Применить» — applied_check, не applied_open: редакция и снимок на месте, событий нет."""
        with files("sqlite") as (jp, md, version):
            self.conflicted(jp, md, version)
            before = self.state(jp)
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied_check", rep.get("outcomes"))
            self.assertEqual(self.card(jp, "r4")["rev"], 1)
            after = self.state(jp)
            self.assertEqual(after["undo_snapshots"]["r4"], before["undo_snapshots"]["r4"])
            self.assertEqual(after["application_events"], before["application_events"])
            self.assertFalse(rep.get("replaced_cards"))

    def test_undo_returns_words_and_open_place(self):
        with files("sqlite") as (jp, md, version):
            self.conflicted(jp, md, version)
            self.save(jp, version, "r4", kind="revert")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "reverted", rep.get("outcomes"))
            self.assertEqual(self.word(jp, "w5")["w"], "та")
            self.assertEqual(self.place_of(jp, "r4")["status"], "open")

    def test_confirmation_closes_the_place_and_keeps_the_snapshot(self):
        """Ф-К-2: «Ответ верен — оставить» (тот же ответ с ноутбука) — место закрыто человеком, отмена прежняя."""
        with files("sqlite") as (jp, md, version):
            self.conflicted(jp, md, version)
            snapshot = self.state(jp)["undo_snapshots"]["r4"]
            self.press(jp, version, REPLACE, "r4")
            self.assertEqual(self.states(jp)["r4"]["state"], "saved")
            self.assertFalse(self.conflicts(jp, "r4"))
            rep = self.apply(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
            self.assertEqual(self.place_of(jp, "r4")["status"], "resolved")
            kept = self.state(jp)["undo_snapshots"]["r4"]
            # участки «до» прежние; «после» у строки места — уже с подтверждением (иначе отмена отказала бы).
            # К-В4, было: формат 2, менялся только отпечаток «after»; стало: формат 3, строки «после» участков места
            self.assertEqual(kept["format"], 3)
            self.assertEqual({k: v for k, v in kept.items() if k != "review_patches"},
                             {k: v for k, v in snapshot.items() if k != "review_patches"})
            self.assertEqual([p["restore"] for p in kept["review_patches"]],
                             [p["restore"] for p in snapshot["review_patches"]])
            self.assertNotEqual(kept["review_patches"], snapshot["review_patches"])
            self.assertEqual(self.states(jp)["r4"]["state"], "applied")
            self.save(jp, version, "r4", kind="revert")
            core.apply_answers_files(jp, md)
            self.assertEqual(self.word(jp, "w5")["w"], "та")

    def test_agent_or_cloud_repeat_is_not_a_confirmation(self):
        with files("sqlite") as (jp, md, version):
            self.conflicted(jp, md, version)
            self.cloud(jp, version, "r4", option=self.option(jp, "r4", REPLACE))
            self.assertEqual(self.states(jp)["r4"]["state"], "applied_check")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied_check", rep.get("outcomes"))
            self.assertEqual(self.place_of(jp, "r4")["status"], "human")

    def test_new_laptop_answer_clears_the_conflict(self):
        with files("sqlite") as (jp, md, version):
            self.conflicted(jp, md, version)
            self.press(jp, version, AS_IS["label"], "r4")
            self.assertFalse(self.conflicts(jp, "r4"))
            self.assertEqual(self.states(jp)["r4"]["state"], "pending_new")

    def test_answers_that_do_not_clear_the_conflict(self):
        """Ф-К-3: ответ старше сохранённого и ответ без подтверждённого автора конфликт не снимают."""
        with files("sqlite") as (jp, md, version):
            self.conflicted(jp, md, version)
            review_pack.add_answers(jp, [{"id": "r4", "version": version, "ts": "2026-11-01T00:00:00",
                                          "option": 0}], "local", trusted_actor="human")
            self.assertTrue(self.conflicts(jp, "r4"), "ответ, проигравший по времени, снял конфликт")
            review_pack.add_answers(jp, [{"id": "r4", "version": version, "option": 0}], "local")
            self.assertTrue(self.conflicts(jp, "r4"), "ответ без подтверждённого автора снял конфликт")

    def test_keep_after_revert_returns_the_check(self):
        """«Не отменять» после «Подготовить отмену» возвращает состояние «проверьте»."""
        with files("sqlite") as (jp, md, version):
            self.conflicted(jp, md, version)
            self.save(jp, version, "r4", kind="revert")
            self.save(jp, version, "r4", kind="keep")
            self.assertEqual(self.states(jp)["r4"]["state"], "applied_check")

    def test_laptop_over_cloud_not_yet_applied_keeps_the_conflict(self):
        """Ф-К-4, узкий вариант: страница могла не показать ответ облака — расхождение записывается."""
        with files("sqlite") as (jp, md, version):
            self.cloud(jp, version, "r4", option=self.option(jp, "r4", REPLACE))
            self.press(jp, version, AS_IS["label"], "r4")
            self.assertEqual(len(self.conflicts(jp, "r4")), 1)

    def test_listen_button_as_later_answer_is_left_for_human(self):
        """Ф-К-8: кнопка прослушивания при расхождении не применяется — left_for_human, не чужой отказ."""
        with files("sqlite", own.result()) as (jp, md, version):
            self.press(jp, version, QUIET, "r3")
            self.cloud(jp, version, "r3", option=self.option(jp, "r3", WORDS))
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r3"]["outcome"], "left_for_human", rep.get("outcomes"))

    def test_own_text_as_later_answer_is_applied_check(self):
        with files("sqlite") as (jp, md, version):
            self.press(jp, version, AS_IS["label"], "r4")
            self.cloud(jp, version, "r4", text="тот")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied_check", rep.get("outcomes"))
            self.assertEqual(self.word(jp, "w5")["w"], "тот")


class ListenFingerprintOfAsrCards(Base):
    """К-5: кнопка прослушивания «модели расходятся» сверяется со словами своего интервала, а не с ревизией."""

    def test_other_place_edit_does_not_refuse_the_button(self):
        with files("sqlite", own.result()) as (jp, md, version):
            self.assertTrue(self.card(jp, "r3").get("listen_fp"))
            self.press(jp, version, REPLACE, "r4")
            core.apply_answers_files(jp, md)
            self.press(jp, version, WORDS, "r3")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r3"]["outcome"], "applied", rep.get("outcomes"))
            self.assertEqual(self.place_of(jp, "r3")["status"], "resolved")

    def test_changed_interval_words_refuse_the_button(self):
        with files("sqlite", own.result()) as (jp, md, version):
            self.press(jp, version, WORDS, "r3")
            data = transcript.load(jp)
            next(w for s in data["segments"] for w in s["words"] if w["word_id"] == "w6")["w"] = "пойдём"
            transcript.save(data, jp, md)
            rep = core.apply_answers_files(jp, md)
            row = self.outcomes(rep)["r3"]
            self.assertNotEqual(row["outcome"], "applied", rep.get("outcomes"))
            self.assertNotEqual(self.place_of(jp, "r3")["status"], "resolved")

    def test_old_pack_without_fingerprint_keeps_the_revision_check(self):
        with files("sqlite", own.result()) as (jp, md, version):
            state = self.state(jp)
            next(i for i in state["manifest"]["items"] if i["id"] == "r3").pop("listen_fp")
            review_pack._write_answers(jp, state)
            self.press(jp, version, REPLACE, "r4")
            core.apply_answers_files(jp, md)
            self.press(jp, version, WORDS, "r3")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r3"]["outcome"], "source_conflict", rep.get("outcomes"))

    def test_card_fingerprint_ignores_only_a_missing_old_key(self):
        """Ф-К-9: listen_fp не учитывается, только если у прежней карточки ключа нет, а у новой есть."""
        base = {"question": "q", "options": [], "t1": 1, "t2": 2}
        same = review_pack._same_question
        self.assertTrue(same(dict(base), dict(base, listen_fp="a")))
        self.assertFalse(same(dict(base, listen_fp="b"), dict(base, listen_fp="a")))
        self.assertFalse(same(dict(base, listen_fp="a"), dict(base)))
        self.assertFalse(same(dict(base), dict(base, listen_fp="a", question="другой")))

    def test_repeating_the_question_over_an_old_card_writes_nothing(self):
        """Со стороны 17а: старая карточка без listen_fp и тот же вопрос — не новая редакция."""
        with files("sqlite", own.result()) as (jp, md, _version):
            state = self.state(jp)
            next(i for i in state["manifest"]["items"] if i["id"] == "r3").pop("listen_fp")
            review_pack._write_answers(jp, state)
            q = six.questions()[2]
            rep = review_pack.upsert_cards(jp, [q], why="тест")
            self.assertIn("r3", rep["unchanged"], rep)
            self.assertEqual(self.card(jp, "r3")["rev"], 1)

    def test_failed_pass_card_has_its_state(self):
        """К-6: состояние прохода на карточке; в отпечаток вопроса не входит."""
        with files("sqlite", own.result(state="error")) as (jp, _md, _version):
            item = self.card(jp, "r3")
            self.assertEqual(item.get("asr_state"), "error")
            self.assertEqual(review_pack._card_fp(item), review_pack._card_fp(dict(item, asr_state="x")))


class AgentLeavesToHuman(Base):
    """К-10: вариант агента «оставить человеку» — как кнопка «Оставить человеку»."""
    LABEL = "Не знаю — оставить человеку"

    def questions(self, mode=None):
        qs = six.questions()
        qs[1] = dict(qs[1], options=[AS_IS, {"label": self.LABEL, "decision": {"status": "human"}}],
                     **({"answer_mode": mode} if mode else {}))
        return qs

    def test_open_place(self):
        with files("sqlite", qs=self.questions()) as (jp, md, version):
            self.press(jp, version, self.LABEL, "r2")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r2"]["outcome"], "left_for_human", rep.get("outcomes"))
            self.assertEqual(self.place_of(jp, "r2")["status"], "human")

    @staticmethod
    def decide(jp, md):
        """Место r2 решено без правки слов (слова карточки те же)."""
        data = transcript.load(jp)
        next(i for i in data["review"] if i["id"] == "r2")["status"] = "resolved"
        transcript.save(data, jp, md)

    def test_decided_place(self):
        with files("sqlite", qs=self.questions()) as (jp, md, version):
            self.decide(jp, md)
            self.press(jp, version, self.LABEL, "r2")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r2"]["outcome"], "already_decided", rep.get("outcomes"))
            self.assertEqual(self.place_of(jp, "r2")["status"], "resolved")

    def test_speaker_card_returns_a_decided_place(self):
        with files("sqlite", qs=self.questions("speaker")) as (jp, md, version):
            self.decide(jp, md)
            self.press(jp, version, self.LABEL, "r2")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r2"]["outcome"], "left_for_human", rep.get("outcomes"))
            self.assertEqual(self.place_of(jp, "r2")["status"], "human")


class ListenMismatch(Base):
    """К-3/К-4: кнопка прослушивания против слов интервала — свой исход и уточнение под карточкой."""

    def test_pause_with_words(self):
        with files("sqlite", six.result(inside=True)) as (jp, md, version):
            self.press(jp, version, SILENCE, "r1")
            rep = core.apply_answers_files(jp, md)
            row = self.outcomes(rep)["r1"]
            self.assertEqual(row["outcome"], "listen_mismatch", rep.get("outcomes"))
            self.assertIn("ага", row.get("detail", ""))
            self.assertEqual(self.states(jp)["r1"].get("detail"), row.get("detail"))

    def test_asr_place_with_words(self):
        with files("sqlite", own.result()) as (jp, md, version):
            self.press(jp, version, QUIET, "r3")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r3"]["outcome"], "listen_mismatch", rep.get("outcomes"))

    def test_words_button_in_empty_pause_at_review_level(self):
        """Через карточку недостижимо (у пустой паузы нет кнопки «слова верны») — уровнем apply_review."""
        with files("sqlite") as (jp, md, _version):
            _data, report = transcript.apply_review(jp, md, [{"id": "r1", "listened": "words"}], by="human")
            self.assertEqual(report.get("listen_mismatch"), ["r1"])


class OutcomeDetailsOnCard(Base):
    """К-22: уточнение программы — то же, что в отчёте «Применить», видно под карточкой (только база)."""

    def test_detail_is_kept_until_a_new_answer(self):
        with files("sqlite") as (jp, md, version):
            self.save(jp, version, "r2", text="да")
            rep = core.apply_answers_files(jp, md)
            row = self.outcomes(rep)["r2"]
            self.assertEqual(row["outcome"], "hint", rep.get("outcomes"))
            self.assertTrue(row.get("detail"))
            self.assertEqual(self.states(jp)["r2"].get("detail"), row["detail"])
            self.assertNotIn(row["detail"], json.dumps(self.state(jp).get("application_events"), ensure_ascii=False))
            self.press(jp, version, AS_IS["label"], "r2")
            self.assertNotIn("detail", self.states(jp)["r2"])

    def test_cloud_manifest_has_no_details(self):
        """Ф-К-6: в облачный manifest уточнения не уходят — в них слова стенограммы и текст ответа."""
        with files("sqlite") as (jp, md, version):
            self.save(jp, version, "r2", text="да")
            core.apply_answers_files(jp, md)
            self.assertTrue(self.states(jp)["r2"].get("detail"))
            bundle = review_pack.cloud_bundle(jp)
            cloud = json.loads((six.Path(bundle["base"]) / "cloud" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(cloud["answer_states"]["r2"]["state"], "hint")
            self.assertNotIn("detail", cloud["answer_states"]["r2"])


class FileModeKeepsPreviousBehaviour(Base):
    """Файловый режим: новые уточнения не пишутся, прежние исходы и слова те же (решение владельца)."""

    def test_file_mode(self):
        with files("legacy_files") as (jp, md, version):
            self.save(jp, version, "r2", text="да")
            self.press(jp, version, REPLACE, "r4")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual({k: v["outcome"] for k, v in self.outcomes(rep).items()}, {"r2": "hint", "r4": "applied"})
            self.assertEqual(self.texts(jp), ["начало", "вот", "дальше", "тот"])
            self.assertNotIn("detail", self.states(jp)["r2"])
            self.assertNotIn("outcome_details", self.state(jp))


class ReplaceAfterFailedReplacement(Base):
    """К-15: сбой замены карточки после applied_open; ответ на старую редакцию — в истории, не за применённый."""

    def test_answer_on_old_revision_goes_to_history_as_not_applied(self):
        with files("sqlite", own.result(state="empty_unknown")) as (jp, md, version):
            self.save(jp, version, "r3", text="дальше пойдём")
            with mock.patch.object(review_pack, "upsert_cards", side_effect=OSError("диск занят")):
                rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r3"]["outcome"], "applied_open", rep.get("outcomes"))
            self.assertTrue(rep.get("cards_not_replaced"), rep)
            self.press(jp, version, WORDS, "r3")
            rep = core.apply_answers_files(jp, md)
            item = self.card(jp, "r3")
            self.assertEqual(item["rev"], 2, rep)
            rows = [h for h in item["history"] if h["rev"] == 1]
            self.assertEqual([(h["answer"].get("kind"), h["applied"]) for h in rows], [("free", True), ("option", False)])
            self.assertEqual([r["id"] for r in rep.get("answers_not_carried") or []], ["r3"])

    def test_recheck_under_lock(self):
        """Ф-К-10: кандидат перепроверяется под замком — без ждущего ответа замены нет."""
        with files("sqlite", own.result(state="empty_unknown")) as (jp, _md, _version):
            self.assertEqual(review_pack.open_after_answer_cards(jp, ["r3"], due=["r3"]), [])
            self.assertEqual(self.card(jp, "r3")["rev"], 1)


class ContradictionReplacementReason(cc.Base):
    """К-20: замена карточки противоречия после смены слов — причина в истории «слова изменились после показа»."""

    def test_reason(self):
        with files("sqlite") as (jp, md, version):
            cc.ChoiceTests.contradicted(self, jp, md, version)
            item = self.card(jp, "r4")
            option = next(n for n, o in enumerate(item["options"]) if o["label"].startswith(cc.ACCEPT))
            self.save(jp, version, "r4", rev=item["rev"], option=option)
            data = transcript.load(jp)
            next(w for s in data["segments"] for w in s["words"] if w["word_id"] == "w5")["w"] = "тота"
            transcript.save(data, jp, md)
            core.apply_answers_files(jp, md)
            self.assertEqual(self.card(jp, "r4")["history"][-1]["why"], "слова изменились после показа")


def page(body):
    html = (PROJECT / "cards.html").read_text(encoding="utf-8")
    script = 'global.location={hostname:"127.0.0.1"};\n' + html.split("  // ответы этой версии вопросов")[0].split(
        "<script>", 1)[1]
    lines = "  function reportLines" + html.split("  function reportLines", 1)[1].split("\n  }\n", 1)[0] + "\n  }\n"
    return subprocess.run([shutil.which("node")], input=script + lines + body + "\n})();\n",
                          capture_output=True, text=True, encoding="utf-8")


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class PageParts(unittest.TestCase):
    def run_page(self, body):
        result = page(body)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_check_state_title_conflict_line_and_confirm(self):
        self.run_page(r'''
          const a={kind:"option",option:1,ts:"t1",source:"cloud"}, one={id:"r1"};
          man = {answer_states:{r1:{state:"applied_check",ts:"t1",source:"cloud"}}, conflicts:[{id:"r1"}],
                 applied_answers:{r1:["option",1,null]}};
          if (cardStatus(one,a) !== "применён — проверьте: ответы ноутбука и облака расходились") throw Error(cardStatus(one,a));
          if (!holdsApplied(one,a)) throw Error("нет кнопок применённого ответа");
          if (!hasConflict(one) || !CONFLICT_TEXT.includes("расходились")) throw Error("строка конфликта");
          if (JSON.stringify(confirmAnswer(one)) !== '{"option":1}') throw Error(JSON.stringify(confirmAnswer(one)));
          man.answer_states.r1.state = "listen_mismatch";
          if (cardStatus(one,a) !== "не применён — ответ не совпадает со словами") throw Error(cardStatus(one,a));
        ''')

    def test_report_lines_name_failed_replacement(self):
        self.run_page(r'''
          const lines = reportLines({cards_not_replaced:[{id:"r1",why:"диск занят"}], answers_not_carried:[{id:"r1"}]},
                                    [{id:"r1", t:"00:01"}]);
          if (lines[0] !== "Нужно ваше действие:") throw Error(lines[0]);       // КН-15: группа действий (4.4)
          if (!lines[1].includes("новая редакция карточки не создана (диск занят)")) throw Error(lines[1]);
          if (!lines[2].includes("ответьте на карточке заново")) throw Error(lines[2]);
        ''')

    def test_failed_pass_chip(self):
        self.run_page(r'''
          if (chipText({asr_state:"error"}, "asr") !== "сбой распознавания") throw Error("сбой");
          if (chipText({asr_state:"disagreement"}, "asr") !== "модели расходятся") throw Error("расхождение");
          if (chipText({}, "asr") !== "модели расходятся") throw Error("старая карточка");
        ''')


if __name__ == "__main__":
    unittest.main()
