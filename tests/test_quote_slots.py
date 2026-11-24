"""Слоты ненайденных цитат и порог исправления (РАР-7, этап 7.5в, шаг 1). Синтетика, без облака.

Ненайденная цитата при приёме ответа записывается в шаблон слотом `{{u:N}}`, а в свёртке — списком
`unlinked` (что за цитата, почему не найдена, в каком разделе и где в его тексте). Файл свёртки при этом
собирается так же, как раньше, когда пометка вписывалась прямо в шаблон. Порог «исправление доступно» —
одна функция `workflow.quotes_need_repair`. Вызова облака в шаге 1 нет.
"""
import copy
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import conspect
import core
import llm_provider
import model_tasks
import paths
import storage
import test_p3_workflow as p3
import test_quotes_one_by_one as base
import transcript as T
import workflow
from test_quotes_one_by_one import A, B, LABELS, OPEN_W2, make_data

HTML = Path(__file__).resolve().parent.parent / "workflow.html"


def slot_obj(text, data, within=None):
    """Свёртка нового вида, как её собирает приём ответа модели после шага 1."""
    obj, report = conspect.link(text, data, within=within, unlinked="slot", labels=True)
    obj.update(quote_policy=1, speaker_labels=copy.deepcopy(LABELS), provenance="model")
    return obj, report


class SlotLinkTests(unittest.TestCase):
    def test_unlinked_quotes_become_numbered_slots_with_their_places(self):
        text = "Он сказал «этого нет в записи» и «срок релиза» дважды."
        obj, report = conspect.link(text, make_data(), unlinked="slot")
        self.assertEqual(obj["template"], "Он сказал {{u:1}} и {{u:2}} дважды.")
        self.assertEqual([(item["id"], item["quote"], item["why"]) for item in obj["unlinked"]],
                         [("1", "этого нет в записи", "none"), ("2", "срок релиза", "several")])
        for item in obj["unlinked"]:
            self.assertEqual(text[item["at"][0]:item["at"][1]], "«%s»" % item["quote"])
        self.assertEqual(obj["unlinked_count"], 2)
        self.assertEqual(obj["quotes"], {})
        self.assertEqual(report["unlinked"], ["этого нет в записи", "срок релиза"])      # отчёт прежний

    def test_a_written_label_is_cut_but_the_place_is_the_quote_only(self):
        text = "Итог: Участник 2: «нет такого» конец."
        obj, _ = conspect.link(text, make_data(), unlinked="slot", labels=True)
        self.assertEqual(obj["template"], "Итог: {{u:1}} конец.")
        at = obj["unlinked"][0]["at"]
        self.assertEqual(text[at[0]:at[1]], "«нет такого»")

    def test_linked_and_unlinked_are_numbered_separately(self):
        text = "Участник 1: «решили перенести», потом «нет такого» и «согласен»."
        obj, _ = conspect.link(text, make_data(), within=set(A) | set(B), unlinked="slot", labels=True)
        self.assertEqual(obj["template"], "{{q:1}}, потом {{u:1}} и {{q:2}}.")
        self.assertEqual((sorted(obj["quotes"]), len(obj["unlinked"]), obj["unlinked_count"]), (["1", "2"], 1, 1))

    def test_no_unlinked_gives_an_empty_list(self):
        obj, _ = conspect.link("Итог: «решили перенести».", make_data(), within=set(A), unlinked="slot", labels=True)
        self.assertEqual((obj["unlinked"], obj["unlinked_count"]), ([], 0))

    def test_mark_and_reject_modes_do_not_change(self):
        data = make_data()
        text = "Он сказал «этого нет в записи»."
        marked, _ = conspect.link(text, data, unlinked="mark")
        self.assertEqual(marked["template"], "Он сказал этого нет в записи (не найдено в стенограмме).")
        self.assertNotIn("unlinked", marked)
        plain, _ = conspect.link(text, data)
        self.assertEqual(plain["template"], text)
        self.assertNotIn("unlinked", plain)
        self.assertNotIn("unlinked_count", plain)


class SlotRenderTests(unittest.TestCase):
    TEXT = "Итог: «нет такого» и «решили перенести», затем «срок релиза»."

    def test_slot_renders_exactly_the_text_that_used_to_be_written_into_the_template(self):
        data = make_data([base.RESOLVED])
        marked, _ = base.policy_obj(self.TEXT, data, within=set(A) | set(B))
        slotted, _ = slot_obj(self.TEXT, data, within=set(A) | set(B))
        self.assertNotIn("(не найдено", slotted["template"])
        old, _ = conspect.render(marked, data, update_last=False)
        new, _ = conspect.render(slotted, data, update_last=False)
        self.assertEqual(new, old)
        self.assertIn("Итог: нет такого (не найдено в стенограмме) и Участник 1: «решили перенести»", new)
        self.assertIn("затем срок релиза (в разделе несколько таких мест).", new)
        self.assertIn("Цитат: 1 в кавычках, 2 без проверки", new)

    def test_several_places_mark(self):
        data = make_data()
        obj, _ = slot_obj("Итог: «срок релиза» дважды.", data)
        md, _ = conspect.render(obj, data, update_last=False)
        self.assertIn("Итог: срок релиза (в разделе несколько таких мест) дважды.", md)

    def test_a_conspect_without_the_list_renders_byte_for_byte_as_before(self):
        # свёртка 7.5а: пометка уже в шаблоне, списка unlinked нет — собирается ровно как раньше
        data = make_data([OPEN_W2])
        quote = {"parts": [{"t1": 1.0, "t2": 2.0, "word_ids": ["w2", "w3"]}], "author": "решили перенести",
                 "speaker": None, "base": "решили перенести", "base_speaker": "Спикер 1",
                 "last": "решили перенести", "last_speaker": "Спикер 1", "written_label": None}
        obj = {"version": 2, "template": "[00:00:01]\nИтог: {{q:1}} и нет такого (не найдено в стенограмме).",
               "quotes": {"1": quote}, "recording_id": "rec", "source_revision": 1, "analysis_revision": None,
               "provenance": "model", "quote_policy": 1, "unlinked_count": 1,
               "speaker_labels": copy.deepcopy(LABELS)}
        md, rep = conspect.render(obj, data, update_last=False)
        self.assertEqual(md, (
            "# Конспект\n\nСоздан моделью; проверка человеком не подтверждена.\n\n"
            "Участники (условные метки; имена — по текущей стенограмме):\n\n"
            "- Участник 1 — имя не указано (в стенограмме: Спикер 1)\n"
            "- Участник 2 — имя не указано (в стенограмме: Спикер 2)\n\n"
            "Время в квадратных скобках — начало раздела в записи (часы:минуты:секунды).\n\n"
            "Цитат: 0 в кавычках, 2 без проверки\n\n"
            "[00:00:01]\nИтог: Участник 1: решили перенести (по распознаванию, не проверено, 00:00:01) "
            "и нет такого (не найдено в стенограмме)."))
        self.assertNotIn("unlinked", obj)

    def test_unknown_or_broken_slot_stays_as_it_is(self):
        data = make_data()
        obj, _ = slot_obj("Итог: «нет такого».", data)
        obj["template"] += " {{u:9}} {{u:2}}"
        obj["unlinked"].append({"id": "2", "quote": "без причины"})            # нет why: слот не раскрывается
        md, _ = conspect.render(obj, data, update_last=False)
        self.assertIn("нет такого (не найдено в стенограмме). {{u:9}} {{u:2}}", md)

    def test_one_pass_does_not_expand_marks_inside_the_inserted_text(self):
        data = make_data()
        obj, _ = slot_obj("Итог: «нет такого».", data)
        obj["unlinked"][0]["quote"] = "в тексте {{u:1}}"                         # слот внутри самой цитаты
        md, _ = conspect.render(obj, data, update_last=False)
        self.assertIn("Итог: в тексте {{u:1}} (не найдено в стенограмме).", md)

    def test_file_mode_rerender_writes_the_same_marks(self):
        tmp = Path(tempfile.mkdtemp(prefix="quote-slots-file-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        json_path, md_path = tmp / "x.transcript.json", tmp / "x.transcript.md"
        segments = [{"speaker": "Анна", "label": "Анна", "start": 0.0, "end": 4.5, "text": "",
                     "words": [{"w": w, "s": float(i), "e": i + 0.5, "c": 0.9}
                               for i, w in enumerate(["мы", "решили", "перенести", "срок", "релиза"])]}]
        T.save({"file": "C:/rec/x.mp3", "duration": "00:01:00", "speakers": ["Анна"], "segments": segments,
                "review": [], "edits": []}, json_path, md_path)
        data = T.load(json_path)
        ids = [word["word_id"] for word in data["segments"][0]["words"]]
        obj, _ = conspect.link("Итог: «решили перенести» и «нет такого».", data, unlinked="slot", labels=True)
        obj.update(name="модель", job_json=str(json_path), quote_policy=1, provenance="model",
                   speaker_labels=[{"label": "Участник 1", "word_ids": [ids[0]]}])
        cj, cm = conspect.path_for(json_path, "модель")
        cj.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        conspect.rerender(json_path)
        text = cm.read_text(encoding="utf-8")
        self.assertIn("Итог: Участник 1: «решили перенести» и нет такого (не найдено в стенограмме).", text)
        self.assertNotIn("{{u:", text)
        self.assertEqual(json.loads(cj.read_text(encoding="utf-8"))["unlinked"][0]["quote"], "нет такого")


class ThresholdTests(unittest.TestCase):
    def test_threshold_table(self):
        table = [((2, 5), False), ((3, 5), True), ((3, 14), True), ((3, 15), True), ((3, 16), False),
                 ((3, 3), True), ((5, 25), True), ((5, 26), False), ((0, 0), False), ((1, 1), False),
                 ((10, 100), False), ((20, 100), True), ((19, 100), False)]
        for (unlinked, total), expected in table:
            with self.subTest(unlinked=unlinked, total=total):
                self.assertIs(workflow.quotes_need_repair(unlinked, total), expected)

    def test_impossible_numbers_are_an_error(self):
        for args in ((4, 3), (-1, 5), (1, -5), (1.5, 5), (3, 5.0), (True, 5), ("3", 5), (None, 5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                workflow.quotes_need_repair(*args)

    def test_counts_take_only_unlinked_quotes_into_the_numerator(self):
        quotes = {str(n): {} for n in range(1, 6)}              # связанные, в том числе задевшие место разбора
        payload = {"quote_policy": 1, "quotes": quotes, "unlinked": [{}], "unlinked_count": 1}
        self.assertEqual(workflow.quote_counts(payload), (1, 6))
        self.assertFalse(workflow.quotes_need_repair(*workflow.quote_counts(payload)))
        many = {**payload, "unlinked": [{}] * 3, "unlinked_count": 3}
        self.assertEqual(workflow.quote_counts(many), (3, 8))
        self.assertTrue(workflow.quotes_need_repair(*workflow.quote_counts(many)))

    def test_counts_of_a_conspect_before_slots_use_the_saved_number(self):
        payload = {"quote_policy": 1, "quotes": {"1": {}, "2": {}}, "unlinked_count": 3}
        self.assertEqual(workflow.quote_counts(payload), (3, 5))
        self.assertEqual(workflow.quote_counts({"quote_policy": 1, "quotes": {}}), (0, 0))
        self.assertEqual(workflow.quote_counts({"quote_policy": 1, "quotes": {}, "unlinked_count": "3"}), (0, 0))

    def test_other_conspects_have_no_counts(self):
        self.assertIsNone(workflow.quote_counts({"quotes": {"1": {}}, "unlinked_count": 3}))
        self.assertIsNone(workflow.quote_counts({"quote_policy": 2, "quotes": {}}))


class SqliteSlotTests(unittest.TestCase):
    """Приём ответа в базе: слоты, список, файл свёртки, экран. Облако подменено и не вызывается."""
    make_recording = p3.WorkflowTests.make_recording
    attach_recording = p3.WorkflowTests.attach_recording
    segment = staticmethod(p3.WorkflowTests.segment)
    ready_request = p3.WorkflowTests.ready_request
    conspect_files = p3.WorkflowTests.conspect_files
    setUp = base.SqliteModeTests.setUp
    prepared = base.SqliteModeTests.prepared
    answer = staticmethod(base.SqliteModeTests.answer)
    OPEN = base.SqliteModeTests.OPEN

    def two_sections(self, review=()):
        request, recording, task = self.prepared(list(review))
        ids = [word["word_id"] for word in task["words"]]
        answer = {"source_revision": task["source_revision"], "sections": [
            {"text": "Итог: «решили перенести» и «нет такого».", "evidence": ids[:5]},
            {"text": "Потом «согласен с этим», «тоже нет» и «нет такого».", "evidence": ids[5:]}]}
        with mock.patch.object(llm_provider.OpenCodeTextClient, "generate") as cloud:
            workflow.accept_model_result(self.store, request, answer)
        cloud.assert_not_called()
        return request, recording, task

    def test_accept_stores_slots_with_sections_and_places(self):
        request, recording, _ = self.two_sections()
        payload, md = self.conspect_files(recording)
        self.assertEqual([(item["id"], item["quote"], item["why"], item["section"]) for item in payload["unlinked"]],
                         [("1", "нет такого", "none", 0), ("2", "тоже нет", "none", 1),
                          ("3", "нет такого", "none", 1)])
        for item in payload["unlinked"]:
            text = payload["evidence"][item["section"]]["text"]
            self.assertEqual(text[item["at"][0]:item["at"][1]], "«%s»" % item["quote"])
        self.assertEqual(payload["unlinked_count"], len(payload["unlinked"]))
        for number in (1, 2, 3):
            self.assertEqual(payload["template"].count("{{u:%d}}" % number), 1)
        self.assertEqual(len(re.findall(r"\{\{u:", payload["template"])), 3)
        indexes = [payload["template"].index("{{u:%d}}" % number) for number in (1, 2, 3)]
        self.assertEqual(indexes, sorted(indexes))
        self.assertNotIn("(не найдено", payload["template"])
        self.assertNotIn("{{u:", md)
        self.assertEqual(md.count("(не найдено в стенограмме)"), 3)
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")

    def test_the_file_is_the_same_as_when_the_mark_was_written_into_the_template(self):
        _, recording, _ = self.two_sections([self.OPEN])
        payload, md = self.conspect_files(recording)
        legacy = copy.deepcopy(payload)
        by_id = {item["id"]: item for item in legacy.pop("unlinked")}
        legacy["template"] = re.sub(
            r"\{\{u:(\d+)\}\}",
            lambda hit: "%s (%s)" % (by_id[hit.group(1)]["quote"], conspect.UNLINKED_MARKS[by_id[hit.group(1)]["why"]]),
            legacy["template"])
        snapshot = self.store.get_current(recording)
        data = {**snapshot["transcript"]["payload"], "recording_id": recording,
                "source_revision": snapshot["transcript"]["revision"],
                "review": snapshot["review"]["payload"].get("review", [])}
        expected, _ = conspect.render(legacy, data, update_last=False)
        self.assertEqual(md, expected + "\n")

    def reject_mark(self, mark):
        request, recording, task = self.prepared([])
        answer = self.answer(task, "Тогда %s и «согласен с этим»." % mark)
        with self.assertRaisesRegex(ValueError, "служебные маркеры"):
            workflow.accept_model_result(self.store, request, answer)
        with self.assertRaisesRegex(ValueError, "служебные маркеры"):
            workflow._validate_model_part(self.store, request, 0, answer)
        self.assertEqual(workflow.get(self.store, request)["state"], "waiting_model")
        self.assertEqual(self.store.list_conspects(recording), [])

    def test_slot_mark_in_the_answer_is_rejected_on_accept_and_on_part(self):
        self.reject_mark("{{u:1}}")

    def test_quote_mark_in_the_answer_is_still_rejected_on_accept_and_on_part(self):
        self.reject_mark("{{q:1}}")

    def test_slots_survive_a_rebuild_after_the_place_is_resolved(self):
        request, recording, _ = self.two_sections([self.OPEN])
        before, _ = self.conspect_files(recording)

        def resolve(review):
            for item in review["review"]:
                item["status"] = "resolved"
        base.SqliteModeTests.commit(self, recording, resolve)
        with mock.patch.object(llm_provider.OpenCodeTextClient, "generate") as cloud:
            base.SqliteModeTests.refresh(self, recording)
        cloud.assert_not_called()
        after, md = self.conspect_files(recording)
        self.assertEqual((after["unlinked"], after["template"]), (before["unlinked"], before["template"]))
        self.assertEqual(md.count("(не найдено в стенограмме)"), 3)

    document_path = base.SqliteModeTests.document_path
    commit = base.SqliteModeTests.commit
    refresh = base.SqliteModeTests.refresh

    def test_screen_gets_the_counts_for_every_state(self):
        # состояния строки на экране: нет ненайденных / есть, ниже порога / есть, порог достигнут / до 7.5в / без правила
        request, recording, task = self.two_sections()
        with mock.patch.object(paths, "STATE_DB", self.root / "state.sqlite3"):
            item = core.recording_conspects(recording)[0]
        self.assertEqual(item["quote_repair"], {"unlinked": 3, "total": 5, "needed": True})
        self.assertEqual(core.quote_repair_view({"quote_policy": 1, "quotes": {"1": {}, "2": {}}, "unlinked": []}),
                         {"unlinked": 0, "total": 2, "needed": False})
        self.assertEqual(core.quote_repair_view({"quote_policy": 1, "quotes": {"1": {}, "2": {}},
                                                 "unlinked": [{}], "unlinked_count": 1}),
                         {"unlinked": 1, "total": 3, "needed": False})
        self.assertEqual(core.quote_repair_view({"quote_policy": 1, "quotes": {"1": {}}, "unlinked_count": 3}),
                         {"unlinked": 3, "total": 4, "needed": True})          # собрана до 7.5в: только число
        self.assertIsNone(core.quote_repair_view({"quotes": {"1": {}}}))        # свёртка без правила «по одной»
        self.assertIsNone(core.quote_repair_view({"sections": [{"text": "Старый раздел."}]}))

    def test_screen_row_text_is_in_the_page(self):
        page = HTML.read_text(encoding="utf-8")
        # условие показа и текст строки — как в утверждённом виде («N из M», только при N > 0)
        self.assertIn("if (c.quote_repair && c.quote_repair.unlinked > 0) {", page)
        self.assertIn("`Не найдено в стенограмме цитат: ${c.quote_repair.unlinked} из ${c.quote_repair.total}.`", page)
        self.assertNotIn("api/quotes/repair", page)                # кнопки и вызова в шаге 1 нет


if __name__ == "__main__":
    unittest.main()
