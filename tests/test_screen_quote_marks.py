"""Экран и ответы агента показывают цитаты так же, как файл (О255-М1, О255-М3). Синтетика, без облака.

Локальный экран собирает разделы свёртки той же функцией, что и файл (conspect.render_body → conspect.render):
цитата из отрезка пропущенного наложения и цитата при открытом месте стоят без кавычек, с пометкой
«по распознаванию, не проверено» и меткой участника; сырой ответ модели (payload["evidence"]) не показывается.
Ответы conspect_link и conspect_link_selected сообщают о снятых кавычках ключом overlap_unchecked, как conspect_render.
"""
import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import conspect
import core
import core_screen
import paths
import recordings
import storage
import test_p3_workflow as p3
import test_quotes_one_by_one as base
import test_overlap_unmeasured as overlap
import workflow

SKIPPED = {"share": 0.35, "sec": 3.0, "spans": [[5.0, 8.0]]}      # слова «согласен с этим» (5–7 с) — внутри
MARK = "по распознаванию, не проверено"


class ScreenSectionsTests(unittest.TestCase):
    """Разделы свёртки на экране в режиме базы: тот же текст, что в файле свёртки."""
    segment = staticmethod(p3.WorkflowTests.segment)
    attach_recording = p3.WorkflowTests.attach_recording
    ready_request = p3.WorkflowTests.ready_request
    conspect_files = p3.WorkflowTests.conspect_files
    setUp = base.SqliteModeTests.setUp
    prepared = base.SqliteModeTests.prepared
    answer = staticmethod(base.SqliteModeTests.answer)
    TEXT = base.SqliteModeTests.TEXT
    OPEN = base.SqliteModeTests.OPEN
    quality = {}

    def make_recording(self, count=1, review=None, segments=None, speaker_names=None):
        """Как p3.WorkflowTests.make_recording, но с полем качества записи (там оно пустое)."""
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, doc_paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        ident = str(uuid.uuid4())
        data = {"file": str(self.source), "duration": "00:00:10", "speakers": ["Спикер 1", "Спикер 2"],
                "quality": copy.deepcopy(self.quality), "review": review or [], "review_done": False,
                "segments": segments}
        recordings.commit_worker_result(
            self.store, data, self.source, root_id, recordings.operation_context(self.store, "model"),
            recording_id=ident, operation_id=str(uuid.uuid4()),
            generated_at="2026-01-01T00:00:00Z", document_paths=doc_paths)
        recordings.materialize_recording(self.store, ident)
        return ident

    def screen_texts(self, recording):
        with mock.patch.object(paths, "STATE_DB", self.root / "state.sqlite3"):
            conspects = core.recording_conspects(recording)
        self.assertEqual(len(conspects), 1)
        return [section["text"] for section in conspects[0]["sections"]]

    def test_overlap_quote_is_marked_on_the_screen_as_in_the_file(self):
        self.quality = {"overlap_unmeasured_skipped": SKIPPED}
        request, recording, task = self.prepared([])
        workflow.accept_model_result(self.store, request, self.answer(task, self.TEXT))
        payload, md = self.conspect_files(recording)
        self.assertIn("«согласен с этим»", payload["evidence"][0]["text"])        # модель прислала в кавычках
        self.assertIn(f"Участник 2: согласен с этим ({MARK}, 00:00:05)", md)       # файл — без кавычек и с пометкой
        shown = self.screen_texts(recording)
        text = "\n\n".join(shown)
        self.assertIn(f"Участник 2: согласен с этим ({MARK}, 00:00:05)", text)
        self.assertIn("Участник 1: «решили перенести»", text)                      # вне отрезка — проверена, в «»
        self.assertNotIn("«согласен", text)
        self.assertNotIn("Итог: Участник 2: «решили", text)                       # метку модели заменил автор программы
        self.assertIn(text, md)                                                    # экран — то же тело, что в файле

    def test_open_place_quote_is_marked_on_the_screen_as_in_the_file(self):
        request, recording, task = self.prepared([self.OPEN])
        workflow.accept_model_result(self.store, request, self.answer(task, self.TEXT))
        _, md = self.conspect_files(recording)
        shown = "\n\n".join(self.screen_texts(recording))
        self.assertIn(f"Участник 1: решили перенести ({MARK}, 00:00:01)", shown)
        self.assertIn("Участник 2: «согласен с этим»", shown)
        self.assertNotIn("«решили", shown)
        self.assertIn(shown, md)

    def test_screen_follows_the_review_like_the_file_does(self):
        request, recording, task = self.prepared([self.OPEN])
        workflow.accept_model_result(self.store, request, self.answer(task, self.TEXT))
        self.assertIn(MARK, "\n\n".join(self.screen_texts(recording)))

        def close(review):
            for item in review["review"]:
                item["status"] = "resolved"
        base.SqliteModeTests.commit(self, recording, close)
        shown = "\n\n".join(self.screen_texts(recording))
        self.assertIn("Участник 1: «решили перенести»", shown)                      # место закрыто — кавычки вернулись
        self.assertNotIn(MARK, shown)

    def test_screen_does_not_write_and_does_not_change_the_saved_conspect(self):
        self.quality = {"overlap_unmeasured_skipped": SKIPPED}
        request, recording, task = self.prepared([])
        workflow.accept_model_result(self.store, request, self.answer(task, self.TEXT))
        before = copy.deepcopy(self.store.list_conspects(recording)[0]["payload"])
        revision = self.store.list_conspects(recording)[0]["current_revision"]
        self.screen_texts(recording)
        after = self.store.list_conspects(recording)[0]
        self.assertEqual(after["payload"], before)
        self.assertEqual(after["current_revision"], revision)

    def test_when_the_text_cannot_be_built_the_raw_model_answer_is_not_shown(self):
        self.quality = {"overlap_unmeasured_skipped": SKIPPED}
        request, recording, task = self.prepared([])
        workflow.accept_model_result(self.store, request, self.answer(task, self.TEXT))
        with mock.patch.object(conspect, "render_body", side_effect=KeyError("quotes")):
            shown = "\n\n".join(self.screen_texts(recording))
        self.assertIn("Не удалось собрать текст свёртки", shown)
        self.assertNotIn("согласен с этим", shown)

    def test_old_sections_field_and_plain_conspects_are_shown_as_before(self):
        self.assertEqual(core_screen._conspect_screen_sections({"sections": [{"text": "Старый раздел."}]},
                                                               data_of=lambda: 1 / 0),
                         [{"text": "Старый раздел."}])
        self.assertEqual(core_screen._conspect_screen_sections({"name": "x"}, data_of=lambda: 1 / 0), [])
        # evidence без шаблона и цитат (прежний вид): тексты как есть, стенограмма не читается
        self.assertEqual(core_screen._conspect_screen_sections({"evidence": [{"text": "Раздел."}]},
                                                               data_of=lambda: 1 / 0),
                         [{"text": "Раздел."}])


class AgentReportTests(unittest.TestCase):
    """conspect_link и conspect_link_selected говорят, что кавычки сняты из-за наложения (О255-М3)."""
    TEXT = "Итог: Спикер 1: «тета йота каппа»."      # фраза одна на запись: связывается без выбора
    PHRASE = "тета йота каппа"

    def test_conspect_link_reports_overlap_unchecked(self):
        for spans, expected in ((overlap.SPANS, 1), (None, 0)):
            with self.subTest(spans=spans), tempfile.TemporaryDirectory() as directory:
                jp = Path(directory) / "demo.transcript.json"
                jp.write_text(json.dumps(overlap.agent_data(spans)), encoding="utf-8")
                report = conspect.save_linked(jp, "агент", self.TEXT)
                md = Path(report["conspect_md"]).read_text(encoding="utf-8")
                if expected:
                    self.assertEqual(len(report["overlap_unchecked"]), 1)
                    self.assertIn(MARK, md)
                    self.assertNotIn("«" + self.PHRASE, md)
                    self.assertEqual(report["check"]["in_doubt"], 0)        # проверка идёт по уже собранному файлу
                else:
                    self.assertNotIn("overlap_unchecked", report)           # как у conspect_render: ключ только при снятии
                    self.assertIn("«" + self.PHRASE + "»", md)
                again = conspect.rerender(jp, "агент")["агент"]             # тот же ключ у conspect_render
                self.assertEqual("overlap_unchecked" in again, bool(expected))
                if expected:
                    self.assertEqual(again["overlap_unchecked"], report["overlap_unchecked"])

    def test_conspect_link_selected_reports_overlap_unchecked(self):
        text = overlap.AgentConspectTests.TEXT        # фраза дважды: место выбирает человек
        data = overlap.agent_data()
        selection = {"1": {"recording_id": "rec", "source_revision": 1, "word_ids": overlap.IN_SPAN}}
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            jp.write_text(json.dumps(data), encoding="utf-8")
            with mock.patch.object(core, "_done_json", return_value={"json": str(jp)}):
                snapshot, _ = core.prepare_selected_conspect("demo", "selected", text, selection)
                result = core.save_selected_conspect(snapshot, text, selection)
        self.assertEqual(len(result["overlap_unchecked"]), 1)


if __name__ == "__main__":
    unittest.main()
