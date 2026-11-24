"""Автор правки отрезка (`edit_span`) обязателен; правка отмечена как принятая через edit-span.

Без автора правка не принимается ни на одном слое (стенограмма, ядро, MCP, командная строка), и запись
не меняется. Автор `claude` — правка агента, `human` — текст продиктовал человек (только командной строкой).
Правка записывает журнал `edits` и отметки слов с `via: edit_span`, заметка закрытого места называет автора и
способ, сохранение одно. Помечаются только изменённые слова. Синтетика, оба режима хранения.

Различие режимов при показе: в `legacy_files` `*.transcript.json` — сами данные (видны `edits` и `via`), в
`sqlite` публичный документ их не показывает (`documents.py`), но снимок в БД хранит.
"""
import unittest
from unittest import mock

import test_human_confirmation as hc  # noqa: E402  (тот же sandbox и те же карточки)

import core  # noqa: E402
import recordings  # noqa: E402
import rt  # noqa: E402
import server  # noqa: E402
import transcript  # noqa: E402

BACKENDS = hc.BACKENDS
SPAN = (0.2, 0.5, "тесть")


def state(jp):
    return transcript.load(jp), jp.read_bytes()


class AuthorIsRequired(unittest.TestCase):
    def assert_refused(self, call, jp):
        before = state(jp)
        with mock.patch.object(recordings, "operation_for_document",
                               wraps=recordings.operation_for_document) as operation:
            call()
        operation.assert_not_called()          # отказ не открывает операцию (в том числе от имени человека)
        self.assertEqual(state(jp), before)    # ни ревизии, ни файла, ни журнала правок

    def test_transcript_layer(self):
        for backend in BACKENDS:
            with hc.files(backend) as (jp, md, _):
                for kwargs in ({}, {"by": None}, {"by": ""}, {"by": "model"}, {"by": "unknown"}):
                    with self.subTest(backend=backend, **kwargs):
                        def call():
                            with self.assertRaises(ValueError) as ctx:
                                transcript.edit_span(jp, md, *SPAN, **kwargs)
                            self.assertEqual(str(ctx.exception), transcript.EDIT_SPAN_AUTHOR_REQUIRED)
                        self.assert_refused(call, jp)

    def test_core_layer(self):
        for backend in BACKENDS:
            with hc.files(backend) as (jp, md, _):
                with mock.patch.object(core, "_done_json", return_value={"json": jp, "markdown": md}):
                    for kwargs in ({}, {"by": ""}, {"by": "model"}):
                        with self.subTest(backend=backend, **kwargs):
                            def call():
                                with self.assertRaises(core.CoreError) as ctx:
                                    core.edit_span("job", *SPAN, **kwargs)
                                self.assertEqual(str(ctx.exception), transcript.EDIT_SPAN_AUTHOR_REQUIRED)
                            self.assert_refused(call, jp)

    def test_mcp_layer(self):
        for backend in BACKENDS:
            with hc.files(backend) as (jp, md, _):
                with mock.patch.object(core, "_done_json", return_value={"json": jp, "markdown": md}):
                    with self.subTest(backend=backend, by="empty"):
                        self.assert_refused(lambda: self.assertEqual(
                            server.edit_span("job", *SPAN), transcript.EDIT_SPAN_AUTHOR_REQUIRED), jp)
                    with self.subTest(backend=backend, by="human"):      # прежний запрет авторства человека
                        self.assert_refused(lambda: self.assertIn(
                            "локальную карточку", server.edit_span("job", *SPAN, by="human")), jp)

    def test_command_line_layer(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), hc.files(backend) as (jp, md, _):
                with mock.patch.object(core, "_done_json", return_value={"json": jp, "markdown": md}):
                    args = rt.build().parse_args(["edit-span", "job", "0.2", "0.5", "тесть"])
                    self.assertIsNone(args.by)             # умолчания «human» больше нет

                    def call():
                        with self.assertRaises(core.CoreError) as ctx:
                            rt.run(args)
                        self.assertEqual(str(ctx.exception), transcript.EDIT_SPAN_AUTHOR_REQUIRED)
                    self.assert_refused(call, jp)

    def test_text_names_both_authors(self):
        self.assertIn('"claude"', transcript.EDIT_SPAN_AUTHOR_REQUIRED)
        self.assertIn('"human"', transcript.EDIT_SPAN_AUTHOR_REQUIRED)


class MarkAcceptedThroughEditSpan(unittest.TestCase):
    def test_journal_marks_notes_and_md(self):
        for backend in BACKENDS:
            for by, who in (("claude", "агент"), ("human", "человек")):
                with self.subTest(backend=backend, by=by), hc.files(backend) as (jp, md, _):
                    transcript.edit_span(jp, md, *SPAN, by=by)
                    data = transcript.load(jp)
                    entry = data["edits"][-1]
                    self.assertEqual((entry["by"], entry["via"]), (by, "edit_span"))
                    self.assertEqual(hc.word(jp, "w1")["provenance"]["text"]["via"], "edit_span")
                    note = hc.item(jp, "r1")["note"]
                    self.assertEqual(note, f"решено правкой отрезка span@0.20 ({who})")
                    self.assertNotIn("решено человеком", note)
                    self.assertIn(f"({who}, правка отрезка)", transcript.render_md({**data, "review_done": True}))
                    if backend == "sqlite":      # публичный документ в sqlite поле не показывает (не расширяется)
                        self.assertNotIn("edit_span", jp.read_text(encoding="utf-8"))
                    else:
                        self.assertIn("edit_span", jp.read_text(encoding="utf-8"))

    def test_one_save_per_call(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), hc.files(backend) as (jp, md, _):
                revision = transcript.load(jp).get("source_revision")
                with mock.patch.object(transcript, "save", wraps=transcript.save) as save:
                    transcript.edit_span(jp, md, *SPAN, by="claude")
                self.assertEqual(save.call_count, 1)
                if backend == "sqlite":
                    self.assertEqual(transcript.load(jp)["source_revision"], revision + 1)

    def test_prior_edit_reads_via_and_keeps_prefix_fallback(self):
        new = {"edited": "x1", "by": "claude", "provenance": {"text": {"by": "claude", "edit": "x1",
                                                                         "via": "edit_span"}}}
        old = {"edited": "span@0.20", "by": "claude"}
        card = {"edited": "r1", "by": "human", "provenance": {"text": {"by": "human", "edit": "r1"}}}
        self.assertEqual(transcript.prior_edit(new)["via"], "edit_span")
        self.assertEqual(transcript.prior_edit(old)["via"], "edit_span")
        self.assertEqual(transcript.prior_edit(card)["via"], "card")


class OnlyChangedWordsAreMarked(unittest.TestCase):
    def test_unchanged_word_keeps_its_marks(self):
        for backend in BACKENDS:
            for by in ("claude", "human"):
                with self.subTest(backend=backend, by=by), hc.files(backend) as (jp, md, _):
                    transcript.edit_span(jp, md, 0.2, 1.0, "тест словно", by=by)   # w1 прежнее, w2 изменено
                    w1, w2 = hc.word(jp, "w1"), hc.word(jp, "w2")
                    self.assertNotIn("by", w1)
                    self.assertNotIn("edited", w1)
                    self.assertNotIn("provenance", w1)
                    self.assertEqual((w2["w"], w2["by"], w2["edited"]), ("словно", by, "span@0.20"))
                    self.assertEqual(w2["provenance"]["text"]["via"], "edit_span")
                    self.assertFalse(hc.has_human_mark(w1))


    def test_speaker_change_with_same_text_is_marked(self):
        # Тот же текст, другой говорящий: слово помечается по говорящему, с via. Иначе смена говорящего
        # человеком не держала бы слово от агента (human_touched читает provenance.speaker).
        for backend in BACKENDS:
            for by in ("claude", "human"):
                with self.subTest(backend=backend, by=by), hc.files(backend) as (jp, md, _):
                    transcript.edit_span(jp, md, 0.2, 0.2, "тест", speaker="Вера", by=by)
                    w1 = hc.word(jp, "w1")
                    self.assertEqual(w1["provenance"]["speaker"],
                                     {"by": by, "edit": "span@0.20", "via": "edit_span"})
                    self.assertNotIn("text", w1["provenance"])
                    self.assertEqual(transcript.human_touched(w1), by == "human")
                    self.assertNotIn("provenance", hc.word(jp, "w2"))

class HumanSpanOverHumanCardAnswer(unittest.TestCase):
    def test_conflict_names_author_and_hints_reopen(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), hc.files(backend) as (jp, md, _):
                transcript.apply_review(jp, md, [{"id": "r1", "text": "другое"}], by="human")
                before = state(jp)
                with self.assertRaises(ValueError) as ctx:
                    transcript.edit_span(jp, md, 0.2, 0.5, "третье", by="human")
                self.assertIn("ответом человека на карточку", str(ctx.exception))
                self.assertIn("reopen", str(ctx.exception))
                self.assertEqual(state(jp), before)


if __name__ == "__main__":
    unittest.main()