"""Что считается подтверждением человека на карточках — в обоих режимах хранения.

Синтетическая стенограмма из двух слов («тест», «слово») у выдуманного Глеба. Правила:
- ответ человека на карточке главнее правки агента и старой правки без автора; прежняя запись
  остаётся в журнале, слово помнит, что распознала модель;
- поверх ответа другой карточки и поверх правки отрезка человека — конфликт с причиной: обе правки
  человеческие, выбирает человек;
- «Оставить человеку» только ставит статус: слова и говорящий не получают отметок человека;
- агент не меняет и не закрывает место, оставленное человеку.
"""
import contextlib
import copy
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import test_answer_outcomes as base  # noqa: E402  (тот же sandbox и те же правила исходов)

core, recordings, review_pack, settings, transcript = (base.core, base.recordings, base.review_pack,
                                                       base.settings, base.transcript)
import voices  # noqa: E402
import dbtemplate  # noqa: E402

BACKENDS = base.BACKENDS


def result_data():
    data = base.result_data()
    # r3 — второе место на том же слове, что и r1 (другая карточка того же слова).
    data["review"].append({"id": "r3", "status": "open", "text": "тест", "t": 0.2, "t1": 0.2, "t2": 0.5,
                           "speaker": "Глеб", "word_ids": ["w1"]})
    # r4 — сбой прохода распознавания: решается только прослушиванием.
    data["review"].append({"id": "r4", "status": "open", "text": "слово", "t": 0.7, "t1": 0.7, "t2": 1.0,
                           "speaker": "Глеб", "word_ids": ["w2"], "gap": [0.7, 1.0],
                           "asr_problem": {"state": "пропуск", "model": "m"}})
    return data


QUESTIONS = copy.deepcopy(base.QUESTIONS)
QUESTIONS[0]["options"].append({"label": "Говорит Вера", "decision": {"speaker": "Вера"}})
QUESTIONS.append({"id": "r4", "question": "Прослушайте", "options": [
    {"label": "Слова есть", "decision": {"listened": "words"}}]})


@contextlib.contextmanager
def files(backend, before=None):
    """Как base.answer_files, но before(jp, md) выполняется до сборки карточек: карточка видит
    стенограмму уже с прежней правкой (поздние правки отдельно ловит сверка ревизий)."""
    with tempfile.TemporaryDirectory(dir=base._TMP) as temp:
        root = Path(temp)
        db, audio = root / "state.sqlite3", root / "sample.wav"
        audio.write_bytes(b"synthetic")
        data = result_data()
        data["file"] = str(audio)
        layout = recordings.document_layout(audio)
        layout["root"].mkdir()
        store = None
        with mock.patch.object(settings.paths, "STATE_DB", db), \
                mock.patch.object(review_pack, "cut_clip", side_effect=lambda _s, t1, _t2, _out: t1):
            try:
                if backend == "sqlite":
                    store = dbtemplate.initialize(db, "Тест")
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
                if before:
                    before(jp, md)
                pack = review_pack.build(jp, copy.deepcopy(QUESTIONS))
                yield jp, md, pack["version"]
            finally:
                if store:
                    store.close()


def word(jp, wid):
    return next(w for s in transcript.load(jp)["segments"] for w in s["words"] if w.get("word_id") == wid)


def item(jp, rid):
    return next(i for i in transcript.load(jp)["review"] if i["id"] == rid)


def has_human_mark(w):
    return w.get("by") == "human" or any(isinstance(v, dict) and v.get("by") == "human"
                                         for v in (w.get("provenance") or {}).values())


def detail(rep, aid):
    return next((row.get("detail") or "") + " " + row["why"] for row in rep["outcomes"] if row["id"] == aid)


class HumanAnswerOverPriorEdit(base.AnswerOutcomeCase):
    """Кнопка и свой текст поверх прежней правки агента, другой карточки, отрезка человека, без автора."""

    def over(self, before, answer, words, outcome="applied"):
        reps = {}
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, before) as (jp, md, version):
                self.save(jp, version, **answer)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], outcome)
                self.assertEqual(self.words(jp), words)
                reps[backend] = (rep, transcript.load(jp))
        return reps

    def test_option_over_agent_edit_span(self):
        for _, data in self.over(lambda jp, md: transcript.edit_span(jp, md, 0.2, 0.5, "тесть", by="claude",
                                                                      resolve=False),
                                 {"option": 0}, ["другое", "слово"]).values():
            w = data["segments"][0]["words"][0]
            self.assertEqual(w.get("model"), "тест", "слово забыло, что распознала модель")
            self.assertEqual(w["provenance"]["text"]["by"], "human")
            log = [(e["id"], e["by"]) for e in data["edits"]]
            self.assertIn(("span@0.20", "claude"), log, "прежняя правка агента пропала из журнала")
            self.assertIn(("r1", "human"), log)

    def test_free_text_over_agent_edit_span(self):
        for _, data in self.over(lambda jp, md: transcript.edit_span(jp, md, 0.2, 0.5, "тесть", by="claude",
                                                                      resolve=False),
                                 {"text": "пятое"}, ["пятое", "слово"]).values():
            self.assertEqual(data["segments"][0]["words"][0].get("model"), "тест")
            self.assertIn("span@0.20", [e["id"] for e in data["edits"]])

    def test_option_over_agent_answer_of_other_card(self):
        def agent(jp, md):
            transcript.apply_review(jp, md, [{"id": "r3", "text": "тесть"}], by="claude")
        self.over(agent, {"option": 1}, ["третье", "слово"])

    def test_free_text_over_agent_answer_of_other_card(self):
        def agent(jp, md):
            transcript.apply_review(jp, md, [{"id": "r3", "text": "тесть"}], by="claude")
        self.over(agent, {"text": "пятое"}, ["пятое", "слово"])

    def test_answer_removes_word_inserted_by_agent(self):
        def agent(jp, md):
            _, rep = transcript.apply_word_edits(jp, md, [{"op": "insert", "s": 0.45, "e": 0.5, "text": "ну",
                                                           "speaker": "Глеб", "after_id": "w1", "card": "a1"}],
                                                 by="claude")
            self.assertFalse(rep["failed"], rep["failed"])
            data = transcript.load(jp)                       # место r1 теперь — «тест ну»
            inserted = next(w["word_id"] for s in data["segments"] for w in s["words"] if w["w"] == "ну")
            data["review"][0]["word_ids"] = ["w1", inserted]
            transcript.save(data, jp, md)
        # Файловый режим (legacy_files) уходит: итог тот же в обоих режимах, файловый путь «ответ поверх правки агента»
        # остаётся в test_partial_match_keeps_untouched_word.
        for backend in ("sqlite",):
            with self.subTest(backend=backend), files(backend, agent) as (jp, md, version):
                self.save(jp, version, option=0)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "applied")
                self.assertEqual(self.words(jp), ["другое", "слово"])

    def test_partial_match_keeps_untouched_word(self):
        # Агент правил оба слова; ответ человека меняет одно, второе остаётся правкой агента.
        def agent(jp, md):
            transcript.edit_span(jp, md, 0.2, 1.0, "тесть словно", by="claude", resolve=False)
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, agent) as (jp, md, version):
                self.save(jp, version, text="другое")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "applied")
                self.assertEqual(self.words(jp), ["другое", "словно"])
                self.assertEqual(word(jp, "w2")["provenance"]["text"]["by"], "claude")

    def test_old_edit_without_author_gives_way(self):
        # Старая запись: слово помечено правкой, но кто её сделал — неизвестно (ни by, ни журнала).
        def legacy(jp, md):
            data = transcript.load(jp)
            data["segments"][0]["words"][0].update(w="тесть", edited="r9")
            data["edits"] = []
            transcript.save(data, jp, md)
        self.over(legacy, {"option": 0}, ["другое", "слово"])
        self.over(legacy, {"text": "пятое"}, ["пятое", "слово"])

    def test_other_card_answer_of_human_is_a_conflict_with_reason(self):
        def human(jp, md):
            transcript.apply_review(jp, md, [{"id": "r3", "text": "тесть"}], by="human")
        # Исход свободного текста — общий для неудачной правки слов (not_applied); причина та же.
        for answer, outcome in (({"option": 0}, "apply_conflict"), ({"text": "пятое"}, "not_applied")):
            for rep, _ in self.over(human, answer, ["тесть", "слово"], outcome).values():
                self.assertIn("r3", detail(rep, "r1"), "причина не называет карточку прежней правки")

    def test_human_edit_span_is_a_conflict_with_reason(self):
        # Обе правки — человеческие: выбирает человек, одинаково в обоих режимах.
        def human(jp, md):
            transcript.edit_span(jp, md, 0.2, 0.5, "ручной", by="human", resolve=False)
        for answer, outcome in (({"option": 0}, "apply_conflict"), ({"text": "пятое"}, "not_applied")):
            for rep, _ in self.over(human, answer, ["ручной", "слово"], outcome).values():
                why = detail(rep, "r1")
                self.assertIn("правкой отрезка человека", why)
                self.assertIn("span@0.20", why)

    def test_repeated_answer_of_same_card_is_applied(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.save(jp, version, option=0)
                self.apply(jp, md)
                self.assertEqual(self.words(jp), ["другое", "слово"])

    def test_human_speaker_choice_only_marks_speaker(self):
        # Файловый режим (legacy_files) уходит: отметка голоса в слове одна для обоих режимов (voices._word_human).
        for backend in ("sqlite",):
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.save(jp, version, option=0)                 # только текст
                self.save(jp, version, aid="r2", option=0)
                self.apply(jp, md)
                self.assertFalse(voices._word_human(word(jp, "w1")), "текст без выбора голоса стал голосом")
            with self.subTest(backend=backend, speaker=True), files(backend) as (jp, md, version):
                self.save(jp, version, option=2)                 # «Говорит Вера»
                self.apply(jp, md)
                self.assertTrue(voices._word_human(word(jp, "w1")))
                self.assertFalse(voices._word_human(word(jp, "w2")))


class LeaveToHuman(base.AnswerOutcomeCase):
    def test_leave_to_human_confirms_nothing(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.save(jp, version, kind="human")
                rep = self.apply(jp, md)               # повтор: ни событий, ни ревизий, ни отметок
                self.assertEqual(self.outcomes(rep)["r1"], "left_for_human")
                self.assertEqual(item(jp, "r1")["status"], "human")
                w = word(jp, "w1")
                self.assertFalse(has_human_mark(w), f"«Оставить человеку» подтвердило слово: {w}")
                self.assertFalse(voices._word_human(w))

    def test_comment_reaches_note_of_place_already_left_to_human(self):
        def held(jp, md):
            transcript.apply_review(jp, md, [{"id": "r1", "status": "human", "note": "раньше"}], by="human")
        # Файловый режим (legacy_files) уходит: заметка места «человеку» одна для обоих режимов; файловый путь
        # «оставить человеку» остаётся в test_leave_to_human_confirms_nothing.
        for backend in ("sqlite",):
            with self.subTest(backend=backend), files(backend, held) as (jp, md, version):
                self.save(jp, version, kind="human", text="тут шум")
                rep = self.apply(jp, md)               # повтор: ни событий, ни ревизий, заметка та же
                self.assertEqual(self.outcomes(rep)["r1"], "left_for_human")
                it = item(jp, "r1")
                self.assertEqual(it["status"], "human")
                self.assertTrue(it["note"].startswith("раньше; "), it["note"])
                self.assertEqual(it["note"].count("тут шум"), 1, it["note"])
                self.assertFalse(has_human_mark(word(jp, "w1")))
            with self.subTest(backend=backend, without_text=True), files(backend, held) as (jp, md, version):
                self.save(jp, version, kind="human")
                self.apply(jp, md)
                self.assertEqual(item(jp, "r1")["note"], "раньше")

    def test_leave_to_human_on_listening_place(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.save(jp, version, aid="r4", kind="human")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r4"], "left_for_human")
                self.assertEqual(item(jp, "r4")["status"], "human")
                self.assertFalse((rep.get("review") or {}).get("conflicts"), rep.get("review"))
                self.assertFalse(has_human_mark(word(jp, "w2")))

    def test_place_closed_by_human_edit_span_stays_closed(self):
        reps = {}
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.save(jp, version, kind="human")
                self.apply(jp, md)
                transcript.edit_span(jp, md, 0.2, 0.5, "ручной", by="human")   # закрывает r1
                self.assertEqual(item(jp, "r1")["status"], "resolved")
                rep = self.apply(jp, md)
                self.assertEqual(item(jp, "r1")["status"], "resolved", "повтор вернул место человеку")
                reps[backend] = self.outcomes(rep)["r1"]
        # Различие режимов: в sqlite правку после сборки карточек ловит сверка ревизий раньше разбора мест.
        self.assertEqual(reps, {"legacy_files": "already_decided", "sqlite": "apply_conflict"})

    def test_already_decided_has_reason_and_page_label(self):
        self.assertIn("already_decided", review_pack.OUTCOME_REASONS)
        html = (base.PROJECT / "cards.html").read_text(encoding="utf-8")
        self.assertIn("already_decided:", html)
        self.assertIn("ничего не подтверждает", html)


class AgentKeepsOffHumanPlace(unittest.TestCase):
    """Место со статусом human агент не меняет и не закрывает — каждым из трёх путей."""

    @contextlib.contextmanager
    def held(self, backend):
        with files(backend) as (jp, md, _):
            _, rep = transcript.apply_review(jp, md, [{"id": "r1", "status": "human"}], by="human")
            self.assertEqual(item(jp, "r1")["status"], "human")
            self.assertFalse(has_human_mark(word(jp, "w1")))
            yield jp, md

    def assertUntouched(self, jp):
        self.assertEqual(item(jp, "r1")["status"], "human")
        self.assertEqual(word(jp, "w1")["w"], "тест")

    def test_apply_review(self):
        for backend in BACKENDS:
            for decision in ({"id": "r1", "text": "тесть"}, {"id": "r1"}, {"id": "r1", "speaker": "Вера"},
                             {"id": "r1", "status": "resolved"}):
                # Файловый режим (legacy_files) уходит: остаются «текст» и «закрыть место»; sqlite — все четыре решения.
                if backend == "legacy_files" and decision not in ({"id": "r1", "text": "тесть"},
                                                                   {"id": "r1", "status": "resolved"}):
                    continue
                with self.subTest(backend=backend, decision=decision), self.held(backend) as (jp, md):
                    _, rep = transcript.apply_review(jp, md, [decision], by="claude")
                    self.assertEqual([c.get("why") for c in rep.get("conflicts") or []],
                                     ["место оставлено человеку"])
                    self.assertUntouched(jp)

    def test_apply_word_edits(self):
        for backend in BACKENDS:
            for edit in ({"op": "replace", "t": 0.2, "w": "тест", "word_id": "w1", "text": "тесть"},
                         {"op": "relabel", "t": 0.2, "w": "тест", "word_id": "w1", "speaker": "Вера"},
                         {"op": "delete", "t": 0.2, "w": "тест", "word_id": "w1"},
                         {"op": "place_text", "t1": 0.2, "t2": 0.5, "text": "тесть", "word_ids": ["w1"]},
                         {"op": "insert", "s": 0.45, "e": 0.5, "text": "ну", "speaker": "Глеб", "after_id": "w1"}):
                # Файловый режим (legacy_files) уходит: остаются replace и insert; sqlite — все пять операций.
                if backend == "legacy_files" and edit["op"] not in ("replace", "insert"):
                    continue
                with self.subTest(backend=backend, op=edit["op"]), self.held(backend) as (jp, md):
                    _, rep = transcript.apply_word_edits(jp, md, [{**edit, "card": "a1"}], by="claude",
                                                         closed={"r1": "a1"})
                    self.assertIn("оставлено человеку", rep["failed"].get("a1", ""))
                    self.assertUntouched(jp)
                    self.assertEqual(TestWords.count(jp), 2)

    def test_apply_word_edits_does_not_close_held_place(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), self.held(backend) as (jp, md):
                _, rep = transcript.apply_word_edits(
                    jp, md, [{"op": "replace", "t": 0.7, "w": "слово", "word_id": "w2", "text": "словно",
                              "card": "a1"}], by="claude", closed={"r1": "a1"})
                self.assertFalse(rep["failed"])
                self.assertUntouched(jp)

    def test_edit_span(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), self.held(backend) as (jp, md):
                with self.assertRaisesRegex(ValueError, "оставлено человеку"):
                    transcript.edit_span(jp, md, 0.2, 0.5, "тесть", by="claude")
                self.assertUntouched(jp)
            with self.subTest(backend=backend, wider=True), self.held(backend) as (jp, md):
                with self.assertRaisesRegex(ValueError, "оставлено человеку"):
                    transcript.edit_span(jp, md, 0.1, 1.0, "тесть словно", by="claude")
                self.assertUntouched(jp)

    def test_human_may_still_decide_held_place(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), self.held(backend) as (jp, md):
                transcript.edit_span(jp, md, 0.2, 0.5, "ручной", by="human")
                self.assertEqual(item(jp, "r1")["status"], "resolved")

    def test_agent_over_human_word_is_refused_as_before(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, _):
                transcript.apply_review(jp, md, [{"id": "r2", "text": "иное"}], by="human")
                _, rep = transcript.apply_review(jp, md, [{"id": "r2", "text": "новое"}], by="claude")
                self.assertTrue(rep.get("conflicts"))
                self.assertEqual(word(jp, "w2")["w"], "иное")


class TestWords:
    @staticmethod
    def count(jp):
        return sum(len(s["words"]) for s in transcript.load(jp)["segments"])


if __name__ == "__main__":
    unittest.main()
