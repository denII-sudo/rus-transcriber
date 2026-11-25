"""РАР-6.1, PR 17-Б: пересборка пакета карточек не теряет ответы, ждущие «Применить».

Пересборка (`review_pack.build`, `core.build_review_pack`) меняет version пакета: прежние ответы, отметки
применения и снимки отмены уходят в прежнюю ревизию. Если в пакете есть ответ, ждущий «Применить» (сохранён,
подготовлена отмена, новый ответ поверх применённого), пересборка отказывает и называет карточки. Проверка
идёт под замком ответов пакета, ранняя проверка в `core` повторяется под ним. Неприменимые ответы
не блокируют, а перечисляются в отчёте; число применённых ответов, которые после пересборки нельзя будет
отменить со страницы, — `undo_lost`.

Только при хранении записей в базе (указание владельца: новое — без файловой записи); в файловом режиме
пересборка прежняя. Все данные синтетические.
"""
import copy
import unittest
from pathlib import Path
from unittest import mock

import test_pause_buttons_short_answers as six
from test_pause_buttons_short_answers import files

core = six.core
review_pack = six.review_pack
transcript = six.transcript

REPLACE = "Заменить на «тот»"          # вариант 1 карточки r4 (слово w5 «та»)


def new_questions():
    """Вопросы пересборки: другой текст — другая version (одинаковые вопросы в ту же секунду дают ту же)."""
    qs = copy.deepcopy(six.questions())
    qs[0]["question"] += " (пересборка)"
    return qs


def card_no(jp, aid):
    return [i["id"] for i in review_pack.manifest(jp)["items"]].index(aid) + 1


def clip_dirs(jp):
    root = review_pack.pack_dir(jp) / "clips"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


class Base(six.Case):
    def rebuild(self, jp):
        return review_pack.build(jp, new_questions())

    def refused(self, jp, *ids):
        """Пересборка отказала, ничего не изменив; -> текст отказа."""
        before, dirs = review_pack.answers_state(jp), clip_dirs(jp)
        with self.assertRaises(review_pack.RebuildRefused) as caught:
            self.rebuild(jp)
        self.assertEqual(review_pack.answers_state(jp), before, "отказ изменил состояние пакета")
        self.assertEqual(clip_dirs(jp), dirs, "папка клипов новой сборки осталась")
        text = str(caught.exception)
        for aid in ids:
            self.assertIn(f"№{card_no(jp, aid)} ({aid})", text)
        self.assertEqual([row["id"] for row in caught.exception.waiting], list(ids))
        return text

    def applied_r4(self, jp, md, version):
        self.press(jp, version, REPLACE, "r4")
        self.assertEqual(self.outcomes(self.apply(jp, md))["r4"]["outcome"], "applied")


class Groups(unittest.TestCase):
    """Таблица состояний: каждое состояние страницы — ровно в одной группе."""
    TABLE = {
        "waiting": ("saved", "revert_pending", "pending_new"),
        "applied": ("applied", "applied_open", "kept_previous", "applied_check"),
        "finished": ("reverted", "skipped", "already_decided", "left_for_human"),
        "unapplicable": ("negative", "hint", "pause_refused", "blocked_preflight", "apply_conflict",
                         "source_conflict", "not_found", "not_applied", "undo_conflict", "listen_mismatch"),
    }

    def test_each_state_row(self):
        for group, states in self.TABLE.items():
            for state in states:
                with self.subTest(state=state):
                    self.assertEqual(review_pack.rebuild_group(state), group)

    def test_every_outcome_is_classified(self):
        """Новый исход OUTCOME_REASONS без строки в таблице — ошибка этого теста, а не молчаливое «ждёт»."""
        for state in review_pack.OUTCOME_REASONS:
            with self.subTest(state=state):
                self.assertEqual(sum(state in group for group in review_pack.REBUILD_GROUPS.values()), 1)

    def test_unknown_state_waits(self):
        self.assertEqual(review_pack.rebuild_group("будущее_состояние"), "waiting")


class Refusal(Base):
    def test_saved_answer_refuses(self):
        with files("sqlite") as (jp, _md, version):
            self.press(jp, version, REPLACE, "r4")
            text = self.refused(jp, "r4")
            self.assertIn("«Применить сохранённые ответы»", text)

    def test_revert_pending_refuses(self):
        with files("sqlite") as (jp, md, version):
            self.applied_r4(jp, md, version)
            self.save(jp, version, "r4", kind="revert")
            self.assertEqual(review_pack.answer_states(jp)["r4"]["state"], "revert_pending")
            self.assertIn("«Не отменять»", self.refused(jp, "r4"))

    def test_pending_new_refuses(self):
        with files("sqlite") as (jp, md, version):
            self.applied_r4(jp, md, version)
            self.save(jp, version, "r4", text="этот")
            self.assertEqual(self.outcomes(self.apply(jp, md))["r4"]["outcome"], "pending_new")
            self.assertIn("«Подготовить отмену»", self.refused(jp, "r4"))

    def test_several_waiting_cards_named_in_card_order(self):
        with files("sqlite") as (jp, _md, version):
            self.press(jp, version, REPLACE, "r4")
            self.save(jp, version, "r2", text="вон")
            self.refused(jp, "r2", "r4")

    def test_answer_saved_while_clips_are_cut_is_refused_under_the_lock(self):
        """Ранняя проверка прошла; ответ сохранён, пока режутся клипы новой сборки. Проверка под замком
        видит его: отказ, ответ цел, папка клипов новой сборки удалена."""
        with files("sqlite") as (jp, _md, version):
            dirs = clip_dirs(jp)
            saved = []

            def cut(_src, t1, _t2, _out):
                if not saved:
                    saved.append(self.press(jp, version, REPLACE, "r4"))
                return t1

            with mock.patch.object(review_pack, "cut_clip", side_effect=cut), \
                    mock.patch.object(core, "_done_json", return_value={"json": str(jp)}):
                with self.assertRaises(core.CoreError) as caught:
                    core.build_review_pack("demo", new_questions())
            self.assertTrue(saved)
            self.assertIn("№4 (r4)", str(caught.exception))
            self.assertEqual(review_pack.answer_states(jp)["r4"]["state"], "saved")
            self.assertEqual(review_pack.manifest(jp)["version"], version)
            self.assertEqual(clip_dirs(jp), dirs, "папка клипов новой сборки осталась")

    def test_core_refuses_before_cutting_clips(self):
        with files("sqlite") as (jp, _md, version):
            self.press(jp, version, REPLACE, "r4")
            with mock.patch.object(review_pack, "cut_clip") as cut, \
                    mock.patch.object(core, "_done_json", return_value={"json": str(jp)}):
                with self.assertRaises(core.CoreError) as caught:
                    core.build_review_pack("demo", new_questions())
            cut.assert_not_called()
            self.assertIn("№4 (r4)", str(caught.exception))
            self.assertIn("«Применить сохранённые ответы»", str(caught.exception))


class Allowed(Base):
    def test_applied_answer_does_not_block_and_undo_lost_is_reported(self):
        with files("sqlite") as (jp, md, version):
            self.applied_r4(jp, md, version)
            rep = self.rebuild(jp)
            self.assertNotEqual(rep["version"], version)
            self.assertEqual(rep["undo_lost"], 1)
            self.assertEqual(rep["unapplicable"], [])
            self.assertEqual(self.word(jp, "w5")["w"], "тот", "применённое снято")

    def test_applied_without_snapshot_is_not_counted_as_undo_lost(self):
        with files("sqlite") as (jp, md, version):
            self.applied_r4(jp, md, version)
            state = review_pack.answers_state(jp)
            state.pop("undo_snapshots", None)
            review_pack._write_answers(jp, state)
            self.assertEqual(self.rebuild(jp)["undo_lost"], 0)

    def test_unapplicable_answer_does_not_block_and_is_listed(self):
        with files("sqlite") as (jp, md, version):
            self.save(jp, version, "r4", text="неверно")
            self.assertEqual(self.outcomes(self.apply(jp, md))["r4"]["outcome"], "negative")
            rep = self.rebuild(jp)
            self.assertEqual(rep["unapplicable"], [{"card": 4, "id": "r4", "state": "negative"}])
            self.assertEqual(rep["undo_lost"], 0)

    def test_finished_answer_does_not_block(self):
        with files("sqlite") as (jp, md, version):
            self.save(jp, version, "r4", kind="skip")
            self.assertEqual(self.outcomes(self.apply(jp, md))["r4"]["outcome"], "skipped")
            rep = self.rebuild(jp)
            self.assertEqual((rep["unapplicable"], rep["undo_lost"]), ([], 0))

    def test_answer_of_an_old_card_revision_is_not_waiting(self):
        """Ответ прежней редакции карточки (история 13а) к новому вопросу не применяется — он не ждёт."""
        with files("sqlite") as (jp, _md, version):
            self.press(jp, version, REPLACE, "r4")
            card = dict(next(q for q in six.questions() if q["id"] == "r4"), question="Другой вопрос")
            self.assertEqual([r["id"] for r in review_pack.upsert_cards(jp, [card], why="тест")["replaced"]],
                             ["r4"])
            self.assertNotIn("r4", review_pack.answer_states(jp))
            self.rebuild(jp)

    def test_fresh_pack_report_has_the_new_fields(self):
        with files("sqlite") as (jp, _md, _version):
            rep = self.rebuild(jp)
            self.assertEqual((rep["unapplicable"], rep["undo_lost"]), ([], 0))


class FilesUnchanged(Base):
    """Файловый режим (указание владельца): пересборка прежняя — без проверки и без новых полей отчёта."""

    def test_files_rebuild_is_as_before(self):
        with files("legacy_files") as (jp, _md, version):
            self.press(jp, version, REPLACE, "r4")
            rep = self.rebuild(jp)
            self.assertNotEqual(rep["version"], version)
            self.assertNotIn("undo_lost", rep)
            self.assertTrue(Path(review_pack.pack_dir(jp) / f"answers.{version}.json").exists(),
                            "ответы прежней версии не в архиве")


if __name__ == "__main__":
    unittest.main()
