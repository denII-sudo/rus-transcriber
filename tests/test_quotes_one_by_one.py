"""Цитаты по одной (РАР-7, этап 7.5а). Синтетика, без облака.

Запрет по каждой цитате, а не на всю запись: чистая цитата — в «», задевшая неразобранное место
печатается без кавычек с пометкой, после разбора кавычки возвращаются при пересборке. Правило «может ли
место держать цитату» — одна функция `workflow.place_holds_quote`; поле `word["seen"]` (PR 3в линии 1)
она не читает. Открытое место-пауза (слов внутри нет) задевает цитату, только если её слово — сосед паузы
(последнее слово до неё или первое после, `transcript.place_words(..., pause_neighbors=True)`, PR 6а; этап 7.5б).
"""
import copy
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import conspect
import llm_provider
import model_tasks
import paths
import recordings
import storage
import test_p3_workflow as p3
import transcript as T
import workflow


def words(texts, first, start):
    return [{"word_id": f"w{first + i}", "w": text, "s": start + i, "e": start + i + 0.5}
            for i, text in enumerate(texts)]


def make_data(review=None):
    """w1–w5 — «Спикер 1» (0–4 с), w6–w9 — «Спикер 2» (5–8 с); фраза «срок релиза» есть в обеих репликах."""
    return {"recording_id": "rec", "source_revision": 1, "speaker_names": {}, "review": review or [],
            "segments": [
                {"speaker": "Спикер 1", "words": words(["мы", "решили", "перенести", "срок", "релиза"], 1, 0)},
                {"speaker": "Спикер 2", "words": words(["согласен", "срок", "релиза", "точно"], 6, 5)}]}


A = ["w1", "w2", "w3", "w4", "w5"]
B = ["w6", "w7", "w8", "w9"]
LABELS = [{"label": "Участник 1", "word_ids": ["w1"]}, {"label": "Участник 2", "word_ids": ["w6"]}]
OPEN_W2 = {"id": "r1", "status": "open", "word_ids": ["w2"], "t1": 1.0, "t2": 1.0}
OPEN_BY_TIME = {"id": "r2", "status": "open", "t1": 3.0, "t2": 3.0}
OPEN_PAUSE = {"id": "r3", "status": "open", "t1": 4.7, "t2": 4.8}          # между w5 и w6: слов нет
OPEN_NOWHERE = {"id": "r4", "status": "open"}                              # ни слов, ни времени
RESOLVED = {"id": "r5", "status": "resolved", "word_ids": ["w2"], "t1": 1.0, "t2": 1.0}


def policy_obj(text, data, within=None):
    """Свёртка нового вида: как её собирает приём ответа модели."""
    obj, report = conspect.link(text, data, within=within, unlinked="mark", labels=True)
    obj.update(quote_policy=1, speaker_labels=copy.deepcopy(LABELS), provenance="model")
    return obj, report


class RuleTests(unittest.TestCase):
    def test_place_holds_quote_only_when_resolved_and_heard(self):
        self.assertTrue(workflow.place_holds_quote({"status": "resolved"}))
        for status in ("open", "human"):
            self.assertFalse(workflow.place_holds_quote({"status": status}))
        self.assertFalse(workflow.place_holds_quote({}))                      # статус не задан — открыто
        self.assertFalse(workflow._place_unresolved({"status": "resolved"}))
        self.assertTrue(workflow._place_unresolved({}))

    def test_open_places_uses_the_same_rule(self):
        review = {"review": [{"id": "a", "status": "open"}, {"id": "b"}, {"id": "c", "status": "human"},
                             {"id": "d", "status": "resolved"}]}
        self.assertEqual([item["id"] for item in workflow._open_places(review)], ["a", "b", "c"])

    def test_closed_unheard_is_false_until_pr_14(self):
        # поле «закрыто без прослушивания» появится с PR 14: до него функция всегда False
        self.assertFalse(workflow._closed_unheard({"status": "resolved", "unheard": True}))
        self.assertTrue(workflow.place_holds_quote({"status": "resolved"}))

    def test_closed_without_listening_blocks_with_its_own_reason(self):
        data = make_data([RESOLVED])
        with mock.patch.object(workflow, "_closed_unheard", return_value=True):
            self.assertFalse(workflow.place_holds_quote(RESOLVED))
            self.assertEqual(workflow.quote_blockers(data, ["w2", "w3"]), [{"id": "r5", "reason": "unheard"}])
            self.assertEqual(workflow.quote_blockers(data, ["w4"]), [])

    def test_blockers_by_word_ids_time_and_resolved(self):
        data = make_data([OPEN_W2, OPEN_BY_TIME, RESOLVED])
        self.assertEqual(workflow.quote_blockers(data, ["w2", "w3"]), [{"id": "r1", "reason": "open"}])
        self.assertEqual(workflow.quote_blockers(data, ["w4", "w5"]), [{"id": "r2", "reason": "open"}])
        self.assertEqual(workflow.quote_blockers(data, ["w6", "w7"]), [])        # чистая цитата рядом
        self.assertEqual(workflow.quote_blockers(make_data([RESOLVED]), ["w2", "w3"]), [])

    def test_place_with_word_ids_is_judged_by_the_words_not_by_its_time(self):
        # у места есть ссылки на слова: время места при этом может указывать совсем в другое место записи
        elsewhere = {"id": "r6", "status": "open", "word_ids": ["w2"], "t1": 50.0, "t2": 50.0}
        data = make_data([elsewhere])
        self.assertEqual(workflow.quote_blockers(data, ["w2", "w3"]), [{"id": "r6", "reason": "open"}])
        at_w4 = {"id": "r7", "status": "open", "word_ids": ["w2"], "t1": 3.0, "t2": 3.0}   # время w4, слова w2
        data = make_data([at_w4])
        self.assertEqual(workflow.quote_blockers(data, ["w4", "w5"]), [])
        self.assertEqual(workflow.quote_blockers(data, ["w2"]), [{"id": "r7", "reason": "open"}])

    def test_unlocated_open_place_blocks_every_quote(self):
        data = make_data([OPEN_NOWHERE])
        for ids in (["w2"], ["w7", "w8"]):
            self.assertEqual(workflow.quote_blockers(data, ids), [{"id": "r4", "reason": "unlocated"}])

    def test_quote_word_without_time_cannot_be_judged_against_a_timed_place(self):
        data = make_data([OPEN_BY_TIME])
        del data["segments"][0]["words"][1]["s"]                  # у w2 нет времени
        self.assertEqual(workflow.quote_blockers(data, ["w2"]), [{"id": "r2", "reason": "unlocated"}])
        self.assertEqual(workflow.quote_blockers(data, ["w7"]), [])

    def test_the_old_exception_is_gone(self):
        self.assertFalse(hasattr(workflow, "PAUSE_UNTIL_6A"))

    def test_a_pause_between_the_words_of_a_quote_blocks_it(self):
        # случай 12: открытая пауза между w5 и w6; цитата из слов по обе стороны паузы
        data = make_data([OPEN_PAUSE])
        self.assertEqual(workflow.quote_blockers(data, ["w5", "w6"]), [{"id": "r3", "reason": "open"}])
        self.assertEqual(workflow.quote_blockers(data, ["w4", "w5", "w6", "w7"]), [{"id": "r3", "reason": "open"}])

    def test_a_pause_next_to_the_edge_of_a_quote_blocks_it(self):
        data = make_data([OPEN_PAUSE])
        self.assertEqual(workflow.quote_blockers(data, ["w3", "w4", "w5"]), [{"id": "r3", "reason": "open"}])  # w5 — сосед
        self.assertEqual(workflow.quote_blockers(data, ["w6", "w7"]), [{"id": "r3", "reason": "open"}])        # w6 — сосед

    def test_a_pause_aside_does_not_touch_a_quote(self):
        data = make_data([OPEN_PAUSE])
        for ids in (["w1", "w2"], ["w2", "w3", "w4"], ["w7", "w8"], ["w7", "w8", "w9"]):
            self.assertEqual(workflow.quote_blockers(data, ids), [], ids)

    def test_a_missed_gap_pause_works_the_same_way(self):
        gap = dict(OPEN_PAUSE, reasons=["missed"])
        data = make_data([gap])
        self.assertEqual(workflow.quote_blockers(data, ["w5"]), [{"id": "r3", "reason": "open"}])
        self.assertEqual(workflow.quote_blockers(data, ["w6", "w7"]), [{"id": "r3", "reason": "open"}])
        self.assertEqual(workflow.quote_blockers(data, ["w2", "w3"]), [])
        self.assertEqual(workflow.quote_blockers(data, ["w8", "w9"]), [])

    def test_a_pause_at_the_start_or_the_end_of_the_record_has_one_neighbour(self):
        before_all = {"id": "r8", "status": "open", "t1": -3.0, "t2": -2.0}
        after_all = {"id": "r9", "status": "open", "t1": 20.0, "t2": 21.0}
        data = make_data([before_all, after_all])
        self.assertEqual(workflow.quote_blockers(data, ["w1", "w2"]), [{"id": "r8", "reason": "open"}])
        self.assertEqual(workflow.quote_blockers(data, ["w8", "w9"]), [{"id": "r9", "reason": "open"}])
        self.assertEqual(workflow.quote_blockers(data, ["w3", "w4"]), [])

    def test_a_resolved_pause_does_not_block_a_neighbour_quote(self):
        data = make_data([dict(OPEN_PAUSE, status="resolved")])
        self.assertEqual(workflow.quote_blockers(data, ["w5", "w6"]), [])

    def test_a_pause_blocker_follows_the_place_words_function(self):
        data = make_data([OPEN_PAUSE])
        with mock.patch.object(T, "place_words", return_value=(["w2"], True)) as called:
            self.assertEqual(workflow.quote_blockers(data, ["w2"]), [{"id": "r3", "reason": "open"}])
            self.assertEqual(workflow.quote_blockers(data, ["w5"]), [])
        self.assertEqual(called.call_args.kwargs, {"pause_neighbors": True})

    def test_a_place_that_cannot_be_tied_to_words_still_blocks_every_quote(self):
        def nothing(data, item, *, pause_neighbors=False):
            return None
        data = make_data([OPEN_PAUSE])
        with mock.patch.object(T, "place_words", nothing):
            self.assertEqual(workflow.quote_blockers(data, ["w1"]), [{"id": "r3", "reason": "unlocated"}])

    def test_seen_field_is_not_read(self):
        # PR 3в линии 1: «человек видел неизменённое слово». Это не подтверждение: слово только с seen
        # ведёт себя как слово без отметок. В main поля ещё нет — тест ставит его вручную.
        for review in ([OPEN_W2, OPEN_BY_TIME, OPEN_PAUSE, OPEN_NOWHERE, RESOLVED], [RESOLVED], []):
            plain = make_data(review)
            seen = make_data(review)
            for segment in seen["segments"]:
                for word in segment["words"]:
                    word["seen"] = True
            for ids in (["w2", "w3"], ["w4"], ["w7", "w8"]):
                self.assertEqual(workflow.quote_blockers(seen, ids), workflow.quote_blockers(plain, ids))

    def test_data_without_review_has_no_blockers(self):
        data = make_data()
        del data["review"]
        self.assertEqual(workflow.quote_blockers(data, ["w2"]), [])


class LinkTests(unittest.TestCase):
    def test_within_picks_the_single_place_in_the_section(self):
        data = make_data()
        obj, report = conspect.link("Решили, что «срок релиза» переносится.", data)
        self.assertEqual((len(report["unlinked"]), len(obj["quotes"])), (1, 0))      # без within — два места
        obj, report = conspect.link("Решили, что «срок релиза» переносится.", data, within=set(A))
        self.assertEqual((report["unlinked"], obj["quotes"]["1"]["parts"][0]["word_ids"]), ([], ["w4", "w5"]))

    def test_unlinked_quote_is_printed_plainly_with_a_mark(self):
        data = make_data()
        obj, report = conspect.link("Он сказал «этого нет в записи» и «срок релиза» дважды.", data,
                                    unlinked="mark")
        self.assertEqual(report["unlinked"], ["этого нет в записи", "срок релиза"])
        self.assertEqual(obj["template"],
                         "Он сказал этого нет в записи (не найдено в стенограмме) и "
                         "срок релиза (в разделе несколько таких мест) дважды.")
        self.assertEqual(obj["quotes"], {})
        self.assertEqual(obj["unlinked_count"], 2)

    def test_default_behaviour_is_unchanged(self):
        data = make_data()
        obj, report = conspect.link("Он сказал «этого нет в записи».", data)
        self.assertEqual(obj["template"], "Он сказал «этого нет в записи».")
        self.assertEqual(report["unlinked"], ["этого нет в записи"])
        self.assertNotIn("unlinked_count", obj)

    def test_label_before_a_quote_is_part_of_the_quote(self):
        data = make_data()
        text = "Участник 2: «решили перенести» и Участник 2 сказал «срок релиза»."
        obj, _ = conspect.link(text, data, within=set(A), unlinked="mark", labels=True)
        self.assertEqual(obj["template"], "{{q:1}} и Участник 2 сказал {{q:2}}.")
        self.assertEqual(obj["quotes"]["1"]["written_label"], "Участник 2")
        self.assertIsNone(obj["quotes"]["1"]["speaker"])
        self.assertIsNone(obj["quotes"]["2"]["written_label"])


    def test_unknown_name_before_a_quote_without_labels_is_cut_with_the_mark(self):
        # прежний режим (labels=False): имя, которого нет в стенограмме, — цитата не связывается
        data = make_data()
        obj, report = conspect.link("Сказал Иван: «срок релиза» и дальше.", data, within=set(A), unlinked="mark")
        self.assertEqual(report["unlinked"], ["срок релиза"])
        self.assertEqual(obj["template"], "Сказал срок релиза (не найдено в стенограмме) и дальше.")
        self.assertEqual((obj["quotes"], obj["unlinked_count"]), ({}, 1))
        plain, report = conspect.link("Сказал Иван: «срок релиза» и дальше.", data, within=set(A))
        self.assertEqual(plain["template"], "Сказал Иван: «срок релиза» и дальше.")
        self.assertEqual(report["unlinked"], ["срок релиза"])

    def test_known_speaker_name_before_a_quote_is_cut_when_labels_are_on(self):
        data = make_data()
        data["segments"][0]["speaker"] = "Анна"                          # имя без цифр: так его читает QUOTE
        obj, _ = conspect.link("Анна: «решили перенести» и «согласен».", data, unlinked="mark", labels=True)
        self.assertEqual(obj["template"], "{{q:1}} и {{q:2}}.")
        self.assertEqual(obj["quotes"]["1"]["written_label"], "Анна")
        self.assertIsNone(obj["quotes"]["2"]["written_label"])
        self.assertEqual([quote["parts"][0]["word_ids"] for quote in obj["quotes"].values()],
                         [["w2", "w3"], ["w6"]])
        obj, _ = conspect.link("Итог: «срок релиза» в разделе.", data, within=set(A), unlinked="mark", labels=True)
        self.assertEqual(obj["template"], "Итог: {{q:1}} в разделе.")      # «Итог:» — обычный текст, не имя
        self.assertIsNone(obj["quotes"]["1"]["written_label"])


class RenderTests(unittest.TestCase):
    def render(self, data, text):
        obj, _ = policy_obj(text, data, within=set(A) | set(B))
        return conspect.render(obj, data, update_last=False)

    def test_clean_quote_gets_the_author_from_the_program(self):
        data = make_data([RESOLVED])
        md, rep = self.render(data, "Итог: «решили перенести» и «согласен».")
        self.assertIn("Итог: Участник 1: «решили перенести» и Участник 2: «согласен».", md)
        self.assertIn("Цитат: 2 в кавычках, 0 без проверки", md)
        self.assertEqual(rep["author_corrected"], [])

    def test_open_place_quote_is_marked_and_neighbour_stays_clean(self):
        data = make_data([OPEN_W2])
        md, _ = self.render(data, "Итог: «решили перенести» и «согласен».")
        self.assertIn("Итог: Участник 1: решили перенести (по распознаванию, не проверено, 00:00:01) и "
                      "Участник 2: «согласен».", md)
        self.assertIn("Цитат: 1 в кавычках, 1 без проверки", md)

    def test_a_pause_marks_only_the_quotes_next_to_it(self):
        data = make_data([OPEN_PAUSE])           # пауза между w5 и w6: соседи — w5 и w6
        md, _ = self.render(data, "Итог: «решили перенести», «перенести срок релиза», «согласен» и «срок релиза точно».")
        body = md.split("Итог:")[1]
        self.assertIn("«решили перенести»", body)             # w2 w3 — в стороне
        self.assertIn("«срок релиза точно»", body)            # w7 w8 w9 — в стороне
        self.assertNotIn("«перенести срок релиза»", body)     # w3 w4 w5: w5 — сосед
        self.assertNotIn("«согласен»", body)                  # w6 — сосед
        self.assertIn("Цитат: 2 в кавычках, 2 без проверки", md)

    def test_unlocated_place_marks_every_quote(self):
        md, _ = self.render(make_data([OPEN_NOWHERE]), "Итог: «решили перенести» и «согласен».")
        self.assertNotIn("«", md.split("Итог:")[1])
        self.assertIn("Цитат: 0 в кавычках, 2 без проверки", md)

    def test_speaker_without_a_label_gets_the_header_name(self):
        # говорящий появился после приёма: метки у него нет — имя по правилу шапки
        data = make_data([RESOLVED])
        obj, _ = policy_obj("Итог: «решили перенести» и «согласен».", data, within=set(A) | set(B))
        obj["speaker_labels"] = [{"label": "Участник 1", "word_ids": ["w1"]}]
        md, rep = conspect.render(obj, data, update_last=False)
        self.assertIn("Участник 1: «решили перенести» и имя не указано (в стенограмме: Спикер 2): «согласен»", md)
        data["speaker_names"] = {"Спикер 2": "Борис"}
        md, _ = conspect.render(obj, data, update_last=False)
        self.assertIn("Участник 1: «решили перенести» и Борис: «согласен»", md)

    def test_wrong_label_is_replaced_and_reported(self):
        data = make_data()
        md, rep = self.render(data, "Итог: Участник 2: «решили перенести».")
        self.assertIn("Итог: Участник 1: «решили перенести».", md)
        self.assertEqual(rep["author_corrected"], [{"id": "1", "written": "Участник 2", "now": "Участник 1"}])

    def test_changed_word_after_linking_shows_new_text(self):
        data = make_data()
        obj, _ = policy_obj("Итог: «решили перенести».", data)
        data["segments"][0]["words"][1]["w"] = "постановили"
        md, rep = conspect.render(obj, data, update_last=False)
        self.assertIn("Участник 1: «постановили перенести»", md)
        self.assertEqual(rep["changed_since_link"], 1)

    def test_deleted_words_give_a_plain_mark(self):
        data = make_data()
        obj, _ = policy_obj("Итог: «решили перенести».", data)
        data["segments"][0]["words"] = [word for word in data["segments"][0]["words"]
                                        if word["word_id"] not in ("w2", "w3")]
        md, rep = conspect.render(obj, data, update_last=False)
        self.assertIn("Итог: решили перенести (место в стенограмме не найдено)", md)
        self.assertNotIn("«решили", md)
        self.assertEqual(len(rep["missing"]), 1)
        self.assertIn("Цитат: 0 в кавычках, 1 без проверки", md)

    def test_closed_without_listening_mark_and_header(self):
        data = make_data([RESOLVED])
        obj, _ = policy_obj("Итог: «решили перенести» и «согласен».", data)
        with mock.patch.object(workflow, "_closed_unheard", return_value=True):
            md, _ = conspect.render(obj, data, update_last=False)
        self.assertIn("решили перенести (по распознаванию, не прослушано, 00:00:01)", md)
        self.assertIn("Цитат: 1 в кавычках, 1 без проверки (из них не прослушано: 1)", md)

    def test_unlinked_at_accept_counts_in_the_header(self):
        md, _ = self.render(make_data(), "Итог: «нет такого» и «решили перенести».")
        self.assertIn("нет такого (не найдено в стенограмме)", md)
        self.assertIn("Цитат: 1 в кавычках, 1 без проверки", md)

    def test_old_conspect_renders_byte_for_byte(self):
        # свёртка без quote_policy: ни авторов, ни пометок, ни строки шапки — ровно как до 7.5а
        data = make_data([OPEN_W2, OPEN_NOWHERE])
        quote = {"parts": [{"t1": 1.0, "t2": 2.0, "word_ids": ["w2", "w3"]}], "author": "решили перенести",
                 "speaker": "Спикер 1", "base": "решили перенести", "base_speaker": "Спикер 1",
                 "last": "решили перенести", "last_speaker": "Спикер 1"}
        gone = {"parts": [{"t1": 9.0, "t2": 9.5, "word_ids": ["w99"]}], "author": "срок", "speaker": None,
                "base": "срок", "base_speaker": "Спикер 1", "last": "срок", "last_speaker": "Спикер 1"}
        obj = {"version": 2, "template": "[00:00:01]\nИтог: {{q:1}}, {{q:2}}.", "quotes": {"1": quote, "2": gone},
               "recording_id": "rec", "source_revision": 1, "analysis_revision": None, "provenance": "model",
               "speaker_labels": copy.deepcopy(LABELS)}
        md, rep = conspect.render(obj, data, update_last=False)
        self.assertEqual(md, (
            "# Конспект\n\nСоздан моделью; проверка человеком не подтверждена.\n\n"
            "Участники (условные метки; имена — по текущей стенограмме):\n\n"
            "- Участник 1 — имя не указано (в стенограмме: Спикер 1)\n"
            "- Участник 2 — имя не указано (в стенограмме: Спикер 2)\n\n"
            "Время в квадратных скобках — начало раздела в записи (часы:минуты:секунды).\n\n"
            "[00:00:01]\nИтог: Спикер 1: «решили перенести», «срок [место в стенограмме не найдено]»."))
        self.assertNotIn("author_corrected", rep)


class RerenderClosesStoreTests(unittest.TestCase):
    def test_store_is_closed_when_reading_fails(self):
        for failing in ("get_current", "load"):
            store = mock.MagicMock()
            binding = {"recording_id": "rec"}
            if failing == "get_current":
                store.get_current.side_effect = RuntimeError("чтение")
            with self.subTest(failing=failing), \
                    mock.patch.object(recordings, "open_for_document", return_value=(store, binding)), \
                    mock.patch.object(T, "load", side_effect=RuntimeError("чтение")):
                with self.assertRaises(RuntimeError):
                    conspect._rerender_locked(Path("x.transcript.json"))
                store.close.assert_called_once()


class RefreshConspectsTests(unittest.TestCase):
    """recordings.refresh_conspects: что пересобирается при смене только ревизии разбора."""

    @staticmethod
    def run_refresh(payloads, source_revision=5, review_revision=2):
        store = mock.MagicMock()
        store.db.execute.return_value.fetchone.return_value = None           # очередь файлов пуста
        store.get_current.return_value = {"transcript": {"revision": source_revision},
                                          "review": {"revision": review_revision}}
        store.list_conspects.return_value = [{"name": name, "payload": payload}
                                             for name, payload in payloads.items()]
        with mock.patch.object(recordings, "materialize_conspects"), \
                mock.patch.object(conspect, "rerender", return_value={}) as rerender:
            recordings.refresh_conspects(store, "rec", Path("doc.json"))
        return [call.args[1] for call in rerender.call_args_list]

    def test_old_conspect_is_not_rebuilt_by_a_review_only_change(self):
        old = {"assembly_revision": 5, "assembly_review_revision": 1}              # нет quote_policy
        self.assertEqual(self.run_refresh({"old": old}), [])
        self.assertEqual(self.run_refresh({"old": {"assembly_revision": 5}}), [])

    def test_policy_conspect_is_rebuilt_once_by_a_review_only_change(self):
        stale = {"assembly_revision": 5, "assembly_review_revision": 1, "quote_policy": 1}
        fresh = {"assembly_revision": 5, "assembly_review_revision": 2, "quote_policy": 1}
        self.assertEqual(self.run_refresh({"new": stale, "old": {"assembly_revision": 5}}), ["new"])
        self.assertEqual(self.run_refresh({"new": fresh}), [])                      # повтор — уже свежая
        self.assertEqual(self.run_refresh({"new": {"assembly_revision": 5, "quote_policy": 1}}), ["new"])

    def test_any_conspect_is_rebuilt_when_the_transcript_changed(self):
        self.assertEqual(self.run_refresh({"old": {"assembly_revision": 4}}), ["old"])
        self.assertEqual(self.run_refresh({"new": {"assembly_revision": 4, "assembly_review_revision": 2,
                                                   "quote_policy": 1}}), ["new"])


class SqliteModeTests(unittest.TestCase):
    """Модельная свёртка в базе: приём ответа, разбор человеком, пересборка без облака."""
    make_recording = p3.WorkflowTests.make_recording
    attach_recording = p3.WorkflowTests.attach_recording
    segment = staticmethod(p3.WorkflowTests.segment)
    ready_request = p3.WorkflowTests.ready_request
    conspect_files = p3.WorkflowTests.conspect_files

    TEXT = "Итог: Участник 2: «решили перенести», потом «согласен с этим»."
    OPEN = {"id": "r1", "status": "open", "t1": 1.0, "t2": 1.0}

    def setUp(self):
        patcher = mock.patch.object(model_tasks, "CURRENT_VERSION", 3)     # договор версии 3: метки и разделы
        patcher.start()
        self.addCleanup(patcher.stop)
        self.root = Path(tempfile.mkdtemp(prefix="quotes-one-by-one-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.store = storage.initialize(self.root / "state.sqlite3", initial_display_name="Тест")
        self.addCleanup(self.store.close)

    def prepared(self, review, segments=None, version=None):
        """version=1 — задание прежнего вида (без реплик и меток), как у записи, заданной до этапа 7.3."""
        request, recording = self.ready_request(
            review=review,
            segments=segments or [self.segment("Спикер 1", ["мы", "решили", "перенести", "срок", "релиза"]),
                                  self.segment("Спикер 2", ["согласен", "с", "этим", "предложением"], start=5)])
        with mock.patch.object(model_tasks, "CURRENT_VERSION", version or model_tasks.CURRENT_VERSION):
            task = workflow.prepare_model_task(self.store, request)
        return request, recording, task

    @staticmethod
    def answer(task, text):
        ids = [word["word_id"] for word in task["words"]]
        return {"source_revision": task["source_revision"], "sections": [{"text": text, "evidence": ids}]}

    def commit(self, recording, change):
        current = self.store.get_current(recording)
        review = copy.deepcopy(current["review"]["payload"])
        change(review)
        binding = recordings.document_binding(self.store, recording)
        self.store.commit_change_with_document(
            recording, {"source_revision": current["transcript"]["revision"],
                        "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            transcript=None, review=review, root_id=binding["root_id"], generated_at="2026-01-01T00:00:01Z")
        recordings.materialize_recording(self.store, recording)

    def refresh(self, recording):
        with mock.patch.object(paths, "STATE_DB", self.root / "state.sqlite3"):
            return recordings.refresh_conspects(self.store, recording, self.document_path(recording))

    def document_path(self, recording):
        binding = recordings.document_binding(self.store, recording)
        return binding["root"] / binding["paths"]["json"]

    def test_open_place_does_not_reject_the_answer_and_marks_one_quote(self):
        request, recording, task = self.prepared([self.OPEN])
        with mock.patch.object(llm_provider.OpenCodeTextClient, "generate") as cloud:
            workflow.accept_model_result(self.store, request, self.answer(task, self.TEXT))
        cloud.assert_not_called()
        payload, md = self.conspect_files(recording)
        self.assertEqual(payload["quote_policy"], 1)
        self.assertIn("Участник 1: решили перенести (по распознаванию, не проверено, 00:00:01)", md)
        self.assertIn("Участник 2: «согласен с этим»", md)
        self.assertIn("Цитат: 1 в кавычках, 1 без проверки", md)
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")

    def test_two_sections_number_quotes_through_and_sum_the_unlinked(self):
        request, recording, task = self.prepared([self.OPEN])
        ids = [word["word_id"] for word in task["words"]]
        answer = {"source_revision": task["source_revision"], "sections": [
            {"text": "Итог: «решили перенести», «срок релиза» и «нет такого».", "evidence": ids[:5]},
            {"text": "Потом «согласен с этим» и «тоже нет».", "evidence": ids[5:]}]}
        workflow.accept_model_result(self.store, request, answer)
        payload, md = self.conspect_files(recording)
        self.assertEqual(sorted(payload["quotes"]), ["1", "2", "3"])
        self.assertEqual([payload["quotes"][key]["parts"][0]["word_ids"] for key in ("1", "2", "3")],
                         [ids[1:3], ids[3:5], ids[5:8]])
        self.assertEqual(payload["unlinked_count"], 2)
        for number in (1, 2, 3):
            self.assertEqual(payload["template"].count("{{q:%d}}" % number), 1)
        self.assertLess(payload["template"].index("{{q:2}}"), payload["template"].index("{{q:3}}"))
        # «решили перенести» задевает открытое место (время 1 с); остальные две связанные чисты
        self.assertIn("решили перенести (по распознаванию, не проверено, 00:00:01)", md)
        self.assertIn("«срок релиза»", md)
        self.assertIn("«согласен с этим»", md)
        self.assertIn("нет такого (не найдено в стенограмме)", md)
        self.assertIn("тоже нет (не найдено в стенограмме)", md)
        self.assertIn("Цитат: 2 в кавычках, 3 без проверки", md)

    def test_accept_saves_the_review_revision_of_the_assembly(self):
        request, recording, task = self.prepared([self.OPEN])
        workflow.accept_model_result(self.store, request, self.answer(task, self.TEXT))
        payload, _ = self.conspect_files(recording)
        self.assertEqual(payload["assembly_review_revision"],
                         self.store.get_current(recording)["review"]["revision"])
        self.assertIsNotNone(payload["assembly_review_revision"])

    def test_quote_is_looked_for_among_the_words_of_its_own_section(self):
        # «срок релиза» есть в обеих репликах; в разделе первой реплики место одно — within его выбирает
        segments = [self.segment("Спикер 1", ["мы", "решили", "перенести", "срок", "релиза"]),
                    self.segment("Спикер 2", ["согласен", "срок", "релиза", "точно"], start=5)]
        request, recording, task = self.prepared([], segments)
        ids = [word["word_id"] for word in task["words"]]
        answer = {"source_revision": task["source_revision"], "sections": [
            {"text": "Итог: «срок релиза» сдвинут.", "evidence": ids[:5]},
            {"text": "Потом «срок релиза» подтвердили.", "evidence": ids[5:]}]}
        workflow.accept_model_result(self.store, request, answer)
        payload, md = self.conspect_files(recording)
        self.assertEqual([payload["quotes"][key]["parts"][0]["word_ids"] for key in ("1", "2")],
                         [ids[3:5], ids[6:8]])
        self.assertEqual(payload["unlinked_count"], 0)
        self.assertIn("Цитат: 2 в кавычках, 0 без проверки", md)

    def test_unlinked_quote_does_not_reject_the_answer(self):
        request, recording, task = self.prepared([])
        workflow.accept_model_result(self.store, request, self.answer(task, "Он сказал «такого нет вовсе»."))
        _, md = self.conspect_files(recording)
        self.assertIn("такого нет вовсе (не найдено в стенограмме)", md)
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")

    def test_part_with_open_place_is_not_rejected(self):
        request, recording, task = self.prepared([self.OPEN])
        ids = [word["word_id"] for word in task["words"]]
        before = workflow.get(self.store, request)
        workflow._validate_model_part(self.store, request, 0, {
            "source_revision": task["source_revision"], "sections": [{"text": self.TEXT, "evidence": ids}]}, before)

    def test_saved_version_1_task_keeps_the_whole_record_ban(self):
        # задание версии 1 (без реплик и меток) доделывается по-прежнему: цитата при открытом сомнении — отказ
        request, recording, task = self.prepared([self.OPEN], version=1)
        self.assertEqual(model_tasks.version(task), 1)
        answer = self.answer(task, "Тогда «согласен с этим» верно.")
        with self.assertRaises(ValueError):
            workflow.accept_model_result(self.store, request, answer)
        self.assertEqual(workflow.get(self.store, request)["state"], "waiting_model")
        self.assertEqual(self.store.list_conspects(recording), [])

    def test_version_1_open_place_bans_a_clean_linked_quote_on_accept_and_on_part(self):
        # запрет из-за открытого сомнения — сам по себе: цитата связывается однозначно и входит в свидетельства
        request, recording, task = self.prepared([self.OPEN], version=1)
        answer = self.answer(task, "Тогда «согласен с этим» верно.")
        with self.assertRaisesRegex(ValueError, "неразрешённые сомнения"):
            workflow.accept_model_result(self.store, request, answer)
        with self.assertRaisesRegex(ValueError, "неразрешённые сомнения"):
            workflow._validate_model_part(self.store, request, 0, answer)
        self.assertEqual(self.store.list_conspects(recording), [])

    def test_version_1_unlinked_quote_bans_without_any_open_place(self):
        # запрет из-за несвязанной цитаты — сам по себе: сомнений нет
        request, recording, task = self.prepared([], version=1)
        answer = self.answer(task, "Он сказал «такого нет вовсе».")
        with self.assertRaisesRegex(ValueError, "не найдены"):
            workflow.accept_model_result(self.store, request, answer)
        with self.assertRaisesRegex(ValueError, "неоднозначна"):
            workflow._validate_model_part(self.store, request, 0, answer)
        self.assertEqual(self.store.list_conspects(recording), [])

    def test_version_1_clean_quote_without_open_places_is_accepted(self):
        request, recording, task = self.prepared([], version=1)
        answer = self.answer(task, "Тогда «согласен с этим» верно.")
        workflow.accept_model_result(self.store, request, answer)
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")
        self.assertNotIn("quote_policy", self.conspect_files(recording)[0])

    def test_quotes_return_after_the_place_is_resolved_without_the_cloud(self):
        request, recording, task = self.prepared([self.OPEN])
        workflow.accept_model_result(self.store, request, self.answer(task, self.TEXT))

        def resolve(review):                       # меняется только разбор: слова стенограммы те же
            for item in review["review"]:
                item["status"] = "resolved"
        self.commit(recording, resolve)
        with mock.patch.object(llm_provider.OpenCodeTextClient, "generate") as cloud:
            report = self.refresh(recording)
        cloud.assert_not_called()
        self.assertEqual(len(report), 1)
        _, md = self.conspect_files(recording)
        self.assertIn("Участник 1: «решили перенести»", md)
        self.assertIn("Цитат: 2 в кавычках, 0 без проверки", md)
        self.assertNotIn("не проверено,", md)
        # повторный вызов без изменений ничего не пересобирает
        self.assertEqual(self.refresh(recording), {})


class FileModeTests(unittest.TestCase):
    """Файловый режим: apply_review → _refresh_conspects пересобирает свёртку нового вида."""

    def test_quotes_return_after_apply_review(self):
        tmp = Path(tempfile.mkdtemp(prefix="quotes-file-mode-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        json_path, md_path = tmp / "x.transcript.json", tmp / "x.transcript.md"
        segments = [{"speaker": "Анна", "label": "Анна", "start": 0.0, "end": 4.5, "text": "",
                     "words": [{"w": w, "s": float(i), "e": i + 0.5, "c": 0.9}
                               for i, w in enumerate(["мы", "решили", "перенести", "срок", "релиза"])]}]
        review = [{"id": "r1", "t": 1.0, "t1": 1.0, "t2": 1.0, "speaker": "Анна", "text": "решили",
                   "reasons": ["asr"], "status": "open"}]
        T.save({"file": "C:/rec/x.mp3", "duration": "00:01:00", "speakers": ["Анна"], "segments": segments,
                "review": review, "edits": []}, json_path, md_path)
        data = T.load(json_path)
        ids = [word["word_id"] for word in data["segments"][0]["words"]]
        obj, _ = conspect.link("Итог: «решили перенести» срок.", data, unlinked="mark", labels=True)
        obj.update(name="модель", job_json=str(json_path), quote_policy=1, provenance="model",
                   speaker_labels=[{"label": "Участник 1", "word_ids": [ids[0]]}])
        cj, cm = conspect.path_for(json_path, "модель")
        cj.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        conspect.rerender(json_path)
        self.assertIn("Итог: Участник 1: решили перенести (по распознаванию, не проверено, 00:00:01) срок.",
                      cm.read_text(encoding="utf-8"))
        _, report = T.apply_review(json_path, md_path, [{"id": "r1"}])
        self.assertNotIn("error", report.get("_conspects", {}))
        text = cm.read_text(encoding="utf-8")
        self.assertIn("Итог: Участник 1: «решили перенести» срок.", text)
        self.assertIn("Цитат: 1 в кавычках, 0 без проверки", text)


if __name__ == "__main__":
    unittest.main()
