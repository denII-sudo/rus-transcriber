"""РАР-6.1, PR 13б: противоречие — карточка вместо отказа или молчаливой записи (В4).

Человек правкой отрезка меняет слово, которое он решил карточкой (правкой или «как есть»), или агент меняет
слово либо место человека: слова не меняются, место открыто, в открытом пакете появляется одна карточка этого
места с вариантами «Оставить подтверждённое», «Принять новое», «Как распознано». Выбор на ней — единственное
исключение из правила двух человеческих правок. Все данные синтетические.
"""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

import storage
import test_pause_buttons_short_answers as six
from test_pause_buttons_short_answers import BACKENDS, files

core = six.core
review_pack = six.review_pack
transcript = six.transcript

KEEP, ACCEPT, RAW = "Оставить подтверждённое: «", "Принять новое: «", "Как распознано: «"


class Base(six.Case):
    def card(self, jp, rid):
        return next((i for i in review_pack.manifest(jp)["items"] if i["id"] == rid), None)

    def choose(self, jp, version, rid, prefix):
        item = self.card(jp, rid)
        option = next(n for n, o in enumerate(item["options"]) if o["label"].startswith(prefix))
        return self.save(jp, version, rid, rev=item["rev"], option=option)

    def labels(self, jp, rid):
        return [o["label"] for o in self.card(jp, rid)["options"]]

    def answer_r4(self, jp, md, version):
        """Человек на карточке r4 выбрал «Заменить на «тот»»: слово w5 «та» → «тот» (правка карточки)."""
        self.press(jp, version, "Заменить на «тот»", "r4")
        core.apply_answers_files(jp, md)
        self.assertEqual(self.word(jp, "w5")["w"], "тот")

    def answer_r2_as_is(self, jp, md, version):
        """Человек на карточке r2 выбрал «Оставить как есть»: слово w2 «вот» подтверждено без правки."""
        self.press(jp, version, "Оставить как есть", "r2")
        core.apply_answers_files(jp, md)
        word = self.word(jp, "w2")
        self.assertNotIn("edited", word)
        self.assertEqual(word.get("by"), "human")

    def not_written(self, rep, rid):
        self.assertIs(rep["written"], False, rep)
        self.assertEqual(rep["contradiction"], rid, rep)
        self.assertTrue(rep["text"].startswith(f"не записано: противоречие, карточка {rid}"), rep["text"])
        self.assertNotIn("\n", rep["text"].split("не записано", 1)[0])

    def contradiction(self, jp, rid):
        place = self.place_of(jp, rid)
        self.assertIn("contradiction", place.get("reasons") or [], place)
        self.assertEqual(place["status"], "open", place)
        return place["contradiction"]


class ProposalInsteadOfRefusalTests(Base):
    """Тесты 11, 12, 22: противоречие не пишет слова, место открыто, карточка в том же пакете."""

    def test_human_span_over_card_edit(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.answer_r4(jp, md, version)
                rep = core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="human")
                self.not_written(rep, "r4")
                self.assertEqual(rep["card"], "updated")
                self.assertEqual(self.word(jp, "w5")["w"], "тот")
                c = self.contradiction(jp, "r4")
                self.assertEqual((c["confirmed"]["text"], c["confirmed"]["by"]), ("тот", "human"))
                self.assertEqual([(p["text"], p["by"], p["via"]) for p in c["proposals"]],
                                 [("тут", "human", "edit_span")])
                card = self.card(jp, "r4")
                self.assertEqual((card["rev"], review_pack.manifest(jp)["version"]), (2, version))
                self.assertIn("«тот»", card["question"])
                self.assertIn("«тут»", card["question"])

    def test_human_span_over_as_is(self):
        """Находка: «как есть» прежде молча перезаписывалось правкой отрезка человека."""
        # режимы хранения здесь не различаются: только sqlite (оба режима — в соседних тестах)
        for backend in ("sqlite",):
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.answer_r2_as_is(jp, md, version)
                rep = core.edit_span_files(jp, md, 2.0, 2.0, "вон", by="human")
                self.not_written(rep, "r2")
                self.assertEqual(self.word(jp, "w2")["w"], "вот")
                self.assertEqual(self.contradiction(jp, "r2")["confirmed"]["text"], "вот")

    def test_unchanged_text_is_not_a_contradiction(self):
        """Правка отрезка тем же текстом по подтверждённому слову — не противоречие (только «видел»)."""
        # режимы хранения здесь не различаются: только sqlite (оба режима — в соседних тестах)
        for backend in ("sqlite",):
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.answer_r4(jp, md, version)
                rep = core.edit_span_files(jp, md, 3.0, 3.0, "тот", by="human")
                self.assertNotIn("contradiction", rep)
                self.assertNotIn("contradiction", self.place_of(jp, "r4"))

    def test_agent_span_and_agent_review_over_human_words(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend, how="edit_span"), files(backend) as (jp, md, version):
                self.answer_r4(jp, md, version)
                rep = core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="claude")
                self.not_written(rep, "r4")
                self.assertEqual(self.word(jp, "w5")["w"], "тот")
                self.assertEqual([p["by"] for p in self.contradiction(jp, "r4")["proposals"]], ["claude"])
            with self.subTest(backend=backend, how="apply_review"), files(backend) as (jp, md, version):
                self.answer_r4(jp, md, version)
                rep = core.apply_review_files(jp, md, [{"id": "r4", "text": "тут"}], by="claude")
                self.assertEqual([c.get("kept_edits") for c in rep.get("conflicts") or []], [["human"]])
                self.assertEqual(rep.get("contradictions"), ["r4"])
                self.assertEqual(self.word(jp, "w5")["w"], "тот")
                self.assertEqual([p["via"] for p in self.contradiction(jp, "r4")["proposals"]], ["apply_review"])
                self.assertEqual(self.card(jp, "r4")["rev"], 2)

    def test_agent_review_over_place_left_to_human(self):
        # режимы хранения здесь не различаются: только sqlite (оба режима — в соседних тестах)
        for backend in ("sqlite",):
            with self.subTest(backend=backend), files(backend) as (jp, md, _v):
                transcript.apply_review(jp, md, [{"id": "r2", "status": "human"}], by="human")
                rep = core.apply_review_files(jp, md, [{"id": "r2", "text": "вон"}], by="claude")
                self.assertEqual(rep.get("contradictions"), ["r2"])
                self.assertEqual(self.word(jp, "w2")["w"], "вот")
                place = self.place_of(jp, "r2")
                self.assertEqual(place["status"], "human", place)
                self.assertEqual([p["text"] for p in place["contradiction"]["proposals"]], ["вон"])

    def test_human_card_over_agent_edit_applies_at_once(self):
        """Тест 13: человек карточкой по правке агента — сразу применяется, противоречия нет."""
        # правка агента — до сборки пакета: правка после сборки — другой отказ (слова карточки изменились, 13а)
        data = six.result()
        w5 = data["segments"][0]["words"][3]
        w5.update(w="ту", edited="span@3.00", by="claude",
                  provenance={"text": {"by": "claude", "edit": "span@3.00", "via": "edit_span"}})
        data["segments"][0]["text"] = " ".join(w["w"] for w in data["segments"][0]["words"])
        # режимы хранения здесь не различаются: только sqlite (оба режима — в соседних тестах)
        for backend in ("sqlite",):
            with self.subTest(backend=backend), files(backend, data) as (jp, md, version):
                self.press(jp, version, "Заменить на «тот»", "r4")
                rep = core.apply_answers_files(jp, md)
                self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.word(jp, "w5")["w"], "тот")
                self.assertNotIn("contradiction", self.place_of(jp, "r4"))

    def test_saving_does_not_close_the_place(self):
        """Тест 22: до выбора текст прежний; сохранение и правка статуса место не закрывают."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.answer_r4(jp, md, version)
                core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="human")
                data = transcript.load(jp)
                next(i for i in data["review"] if i["id"] == "r4")["status"] = "resolved"
                transcript.save(data, jp, md)
                self.assertEqual(self.place_of(jp, "r4")["status"], "open")
                self.assertIn("r4", [i["id"] for i in transcript.unresolved(transcript.load(jp))])

    def test_repeat_of_the_same_proposal_creates_nothing_and_second_is_a_new_option(self):
        """Тест 18: повтор того же предложения ничего не создаёт; второе предложение — новый вариант."""
        # режимы хранения здесь не различаются: только sqlite (оба режима — в соседних тестах)
        for backend in ("sqlite",):
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.answer_r4(jp, md, version)
                core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="human")
                before = (transcript.load(jp), review_pack.manifest(jp).get("pack_rev"))
                rep = core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="human")
                self.not_written(rep, "r4")
                self.assertEqual(rep["card"], "unchanged")
                self.assertEqual((transcript.load(jp), review_pack.manifest(jp).get("pack_rev")), before)
                core.edit_span_files(jp, md, 3.0, 3.0, "там", by="claude")
                self.assertEqual([p["text"] for p in self.contradiction(jp, "r4")["proposals"]], ["тут", "там"])
                self.assertEqual(sum(label.startswith(ACCEPT) for label in self.labels(jp, "r4")), 2)
                self.assertEqual(self.card(jp, "r4")["rev"], 3)

    def test_word_without_place_gets_a_new_place(self):
        """Тест 19: слово без места разбора — новое место только с причиной «contradiction»."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, _v):
                transcript.edit_span(jp, md, 0.5, 0.5, "начала", by="human")
                rep = core.edit_span_files(jp, md, 0.5, 0.5, "начало", by="claude")
                self.not_written(rep, "r5")
                self.assertEqual(rep["card"], "added")
                place = self.place_of(jp, "r5")
                self.assertEqual((place["reasons"], place["word_ids"], place["status"]),
                                 (["contradiction"], ["w1"], "open"))
                self.assertEqual(self.card(jp, "r5")["rev"], 1)

    def test_human_span_over_own_span_edit(self):
        """В-1, владелец: «Карточка противоречия (Recommended)». Человек правит отрезок поверх своей прежней
        правки отрезка (карточки не было): не прежний отказ, а противоречие на новом месте."""
        # режимы хранения здесь не различаются: только sqlite (оба режима — в соседних тестах)
        for backend in ("sqlite",):
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                transcript.edit_span(jp, md, 0.5, 0.5, "начала", by="human")
                rep = core.edit_span_files(jp, md, 0.45, 0.6, "начали", by="human")
                self.not_written(rep, "r5")
                self.assertEqual(self.word(jp, "w1")["w"], "начала")
                c = self.contradiction(jp, "r5")
                self.assertEqual((c["confirmed"]["via"], c["confirmed"]["card"]), ("edit_span", "span@0.50"))
                self.assertEqual([(p["by"], p["via"]) for p in c["proposals"]], [("human", "edit_span")])
                self.choose(jp, version, "r5", ACCEPT)
                core.apply_answers_files(jp, md)
                self.assertEqual(self.word(jp, "w1")["w"], "начали")

    def test_without_pack_the_card_waits_for_the_build(self):
        """Тест 20: пакета нет — место создаётся (card: pending); сборка сама добавляет его карточку."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, _v):
                self.answer_r4_by_review(jp, md)
                with mock.patch.object(review_pack, "manifest", return_value=None):
                    rep = core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="human")
                self.assertEqual(rep["card"], "pending")
                self.contradiction(jp, "r4")
                review_pack.build(jp, [])
                self.assertTrue(self.labels(jp, "r4")[0].startswith(KEEP))

    def answer_r4_by_review(self, jp, md):
        transcript.apply_review(jp, md, [{"id": "r4", "text": "тот"}], by="human")

    def test_card_failure_after_writing_the_proposal_is_in_the_result(self):
        """Р3: сбой постановки карточки после записи предложения — в результате (`cards_not_replaced`), а не
        исключением мимо MCP; повтор той же правки ставит карточку."""
        # режимы хранения здесь не различаются: только sqlite (оба режима — в соседних тестах)
        for error in (RuntimeError("занято"), OSError("диск занят"), storage.Conflict("запись изменилась")):
            with self.subTest(error=type(error).__name__, how="edit_span"), files("sqlite") as (jp, md, version):
                self.answer_r4(jp, md, version)
                with mock.patch.object(review_pack, "upsert_cards", side_effect=error):
                    rep = core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="human")
                self.not_written(rep, "r4")
                self.assertEqual(rep["card"], "not_replaced")
                self.assertEqual([row["id"] for row in rep["cards_not_replaced"]], ["r4"])
                self.assertEqual(self.card(jp, "r4")["rev"], 1)
                self.contradiction(jp, "r4")
                rep = core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="human")
                self.assertEqual(rep["card"], "updated")
                self.assertNotIn("cards_not_replaced", rep)
            with self.subTest(error=type(error).__name__, how="apply_review"), files("sqlite") as (jp, md, version):
                self.answer_r4(jp, md, version)
                with mock.patch.object(review_pack, "upsert_cards", side_effect=error):
                    rep = core.apply_review_files(jp, md, [{"id": "r4", "text": "тут"}], by="claude")
                self.assertEqual(rep.get("contradictions"), ["r4"])
                self.assertEqual([row["id"] for row in rep["cards_not_replaced"]], ["r4"])
                self.assertEqual(rep["cards"], {"r4": "not_replaced"})
                self.assertEqual(self.word(jp, "w5")["w"], "тот")
                self.contradiction(jp, "r4")


class ChoiceTests(Base):
    """Тесты 15, 16, 17, 23: варианты, подписи, выбор, слова изменились после показа, отмена выбора."""

    def contradicted(self, jp, md, version):
        self.answer_r4(jp, md, version)
        core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="claude")

    def test_options_and_signatures(self):
        with files("sqlite") as (jp, md, version):
            self.contradicted(jp, md, version)
            labels = self.labels(jp, "r4")
            self.assertEqual([next(p for p in (KEEP, ACCEPT, RAW) if label.startswith(p)) for label in labels],
                             [KEEP, ACCEPT, RAW])
            self.assertIn("«тот»", labels[0])
            self.assertIn("«тут»", labels[1])
            self.assertIn("агент", labels[1])
            self.assertIn("«та»", labels[2])
            self.assertIsNone(self.card(jp, "r4")["recommended"])
            self.assertTrue(self.card(jp, "r4").get("seen_fp"))
        with files("sqlite") as (jp, md, version):
            # «как распознано» совпадает с подтверждённым — варианта нет
            self.answer_r2_as_is(jp, md, version)
            core.edit_span_files(jp, md, 2.0, 2.0, "вон", by="human")
            self.assertEqual([next(p for p in (KEEP, ACCEPT, RAW) if label.startswith(p))
                              for label in self.labels(jp, "r2")], [KEEP, ACCEPT])

    def test_each_choice(self):
        cases = ((KEEP, "тот"), (ACCEPT, "тут"), (RAW, "та"))
        for backend in BACKENDS:
            for prefix, text in (cases if backend == "sqlite" else cases[1:2]):
                with self.subTest(backend=backend, choice=prefix), files(backend) as (jp, md, version):
                    self.contradicted(jp, md, version)
                    self.choose(jp, version, "r4", prefix)
                    rep = core.apply_answers_files(jp, md)
                    self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
                    self.assertEqual(self.word(jp, "w5")["w"], text)
                    place = self.place_of(jp, "r4")
                    self.assertEqual(place["status"], "resolved", place)
                    self.assertEqual(place["contradiction"]["proposals"], [])
                    self.assertEqual(len(place["contradiction"]["history"]), 1)
                    if prefix != KEEP:
                        self.assertEqual(self.word(jp, "w5")["provenance"]["text"].get("via"), "contradiction")

    def test_own_text_and_left_to_human(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend, how="own"), files(backend) as (jp, md, version):
                self.contradicted(jp, md, version)
                self.save(jp, version, "r4", rev=self.card(jp, "r4")["rev"], text="те")
                rep = core.apply_answers_files(jp, md)
                self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.word(jp, "w5")["w"], "те")
                self.assertEqual(self.place_of(jp, "r4")["contradiction"]["proposals"], [])
            if backend != "sqlite":
                continue                   # «Оставить человеку» от режима хранения не зависит
            with self.subTest(backend=backend, how="human"), files(backend) as (jp, md, version):
                self.contradicted(jp, md, version)
                self.save(jp, version, "r4", rev=self.card(jp, "r4")["rev"], kind="human")
                core.apply_answers_files(jp, md)
                place = self.place_of(jp, "r4")
                self.assertEqual(place["status"], "human")
                self.assertEqual(self.word(jp, "w5")["w"], "тот")

    def test_exception_only_for_the_contradiction_card(self):
        """Исключение в overwrite_refusal — только для карточки противоречия этого слова: ответ другой карточки
        по тому же слову по-прежнему конфликт двух человеческих правок (тест 14)."""
        # режимы хранения здесь не различаются: только sqlite (оба режима — в соседних тестах)
        for backend in ("sqlite",):
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                # w1 исправлен правкой отрезка человека; агент предлагает другое — новое место r5
                transcript.edit_span(jp, md, 0.5, 0.5, "начала", by="human")
                core.edit_span_files(jp, md, 0.5, 0.5, "начало", by="claude")
                data = transcript.load(jp)
                w1 = next(w for s in data["segments"] for w in s["words"] if w["word_id"] == "w1")
                self.assertIsNone(transcript.overwrite_refusal(w1, data, "human", "r5"))
                self.assertIn("правкой отрезка человека", transcript.overwrite_refusal(w1, data, "human", "r3"))
                self.assertIsNotNone(transcript.overwrite_refusal(w1, data, "claude", "r5"))
                self.choose(jp, version, "r5", ACCEPT)
                rep = core.apply_answers_files(jp, md)
                self.assertEqual(self.outcomes(rep)["r5"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.word(jp, "w1")["w"], "начало")

    def test_words_changed_after_showing(self):
        """Тест 17: слова места изменились после показа — ответ не применён, карточка — новая редакция."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.contradicted(jp, md, version)
                self.choose(jp, version, "r4", ACCEPT)
                rev = self.card(jp, "r4")["rev"]
                data = transcript.load(jp)
                next(w for s in data["segments"] for w in s["words"] if w["word_id"] == "w5")["w"] = "тота"
                transcript.save(data, jp, md)
                rep = core.apply_answers_files(jp, md)
                self.assertNotEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.word(jp, "w5")["w"], "тота")
                self.assertEqual(self.card(jp, "r4")["rev"], rev + 1)

    def test_free_text_when_words_changed(self):
        """Свой текст на карточке противоречия сверяется с показанными словами так же, как вариант."""
        # режимы хранения здесь не различаются: только sqlite (оба режима — в соседних тестах)
        with files("sqlite") as (jp, md, version):
            self.contradicted(jp, md, version)
            rev = self.card(jp, "r4")["rev"]
            self.save(jp, version, "r4", rev=rev, text="те")
            data = transcript.load(jp)
            next(w for s in data["segments"] for w in s["words"] if w["word_id"] == "w5")["w"] = "тота"
            transcript.save(data, jp, md)
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "source_conflict", rep.get("outcomes"))
            self.assertEqual(self.word(jp, "w5")["w"], "тота")
            self.assertEqual(self.card(jp, "r4")["rev"], rev + 1)

    def test_only_word_times_changed_the_choice_applies(self):
        """Р2: изменилось только время слов — человек видел те же слова, выбор применяется (отпечаток
        показанного — ID и текст слов, без времени)."""
        # режимы хранения здесь не различаются: только sqlite (оба режима — в соседних тестах)
        with files("sqlite") as (jp, md, version):
            self.contradicted(jp, md, version)
            self.choose(jp, version, "r4", ACCEPT)
            data = transcript.load(jp)
            w5 = next(w for s in data["segments"] for w in s["words"] if w["word_id"] == "w5")
            w5["s"], w5["e"] = 3.01, 3.19
            transcript.save(data, jp, md)
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
            self.assertEqual(self.word(jp, "w5")["w"], "тут")

    def test_two_word_place_contradiction_on_one_word(self):
        """Р1: место из двух слов, противоречие — по одному. Отпечаток показанного строится и сверяется по
        словам противоречия, выбор ложится только на них; второе слово места не меняется."""
        data = six.result()
        data["review"] = [i for i in data["review"] if i["id"] != "r3"]
        r4 = next(i for i in data["review"] if i["id"] == "r4")
        r4.update(t=2.5, t1=2.5, text="дальше та", word_ids=["w3", "w5"])
        qs = [q for q in six.questions() if q["id"] != "r3"]
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend, data, qs) as (jp, md, version):
                self.press(jp, version, "Оставить как есть", "r4")
                core.apply_answers_files(jp, md)
                rep = core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="human")
                self.not_written(rep, "r4")
                self.assertEqual(self.contradiction(jp, "r4")["words"], ["w5"])
                self.assertEqual(self.card(jp, "r4")["word_ids"], ["w5"])
                self.choose(jp, version, "r4", ACCEPT)
                rep = core.apply_answers_files(jp, md)
                self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.texts(jp), ["начало", "вот", "дальше", "тут"])
                place = self.place_of(jp, "r4")
                self.assertEqual((place["status"], place["contradiction"]["proposals"]), ("resolved", []))

    def test_undo_of_the_choice_returns_the_words(self):
        """Тест 23: «Отменить» применённый выбор возвращает прежние слова."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.contradicted(jp, md, version)
                self.choose(jp, version, "r4", ACCEPT)
                core.apply_answers_files(jp, md)
                self.assertEqual(self.word(jp, "w5")["w"], "тут")
                self.save(jp, version, "r4", rev=self.card(jp, "r4")["rev"], kind="revert")
                rep = core.apply_answers_files(jp, md)
                self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "reverted", rep.get("outcomes"))
                self.assertEqual(self.word(jp, "w5")["w"], "тот")


class HumanNoteTests(Base):
    """Тест 21. В-3, владелец: «Только на противоречии (Recommended)». Примечание человека — отдельно от
    ответа: в ответе, истории противоречия и журнале правок; не в стенограмме, не в .md, не в публичном
    документе и не в облачном пакете; ответом не разбирается."""

    NOTE = "да, уточнил у Анны после встречи"

    def test_note_is_kept_locally_and_goes_nowhere_else(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.answer_r4(jp, md, version)
                core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="claude")
                item = self.card(jp, "r4")
                option = next(n for n, o in enumerate(item["options"]) if o["label"].startswith(ACCEPT))
                self.save(jp, version, "r4", rev=item["rev"], option=option, human_note=self.NOTE)
                rep = core.apply_answers_files(jp, md)
                self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.word(jp, "w5")["w"], "тут")       # «да» в примечании — не ответ
                self.assertEqual(review_pack.answers_state(jp)["answers"]["r4"]["human_note"], self.NOTE)
                data = transcript.load(jp)
                place = next(i for i in data["review"] if i["id"] == "r4")
                self.assertEqual(place["contradiction"]["history"][-1]["human_note"], self.NOTE)
                self.assertEqual(data["edits"][-1].get("human_note"), self.NOTE)
                self.assertNotIn(self.NOTE, " ".join(w["w"] for s in data["segments"] for w in s["words"]))
                self.assertNotIn(self.NOTE, md.read_text(encoding="utf-8"))
                self.assertNotIn(self.NOTE, json.dumps(six.documents._public_review(data), ensure_ascii=False))
                bundle = review_pack.cloud_bundle(jp)
                cloud = (six.Path(bundle["base"]) / bundle["files"]["manifest.json"]).read_text(encoding="utf-8")
                self.assertNotIn(self.NOTE, cloud)
                self.assertNotIn("seen_fp", cloud)

    def test_note_on_another_card_is_not_kept(self):
        with files("sqlite") as (jp, _md, version):
            self.save(jp, version, "r2", option=0, human_note=self.NOTE)
            self.assertNotIn("human_note", review_pack.answers_state(jp)["answers"]["r2"])


class InterfaceTests(Base):
    """Тест 24 и условие архитектора: результат «не записано» читается без JSON; rt.py — код выхода 3."""

    def test_rt_exit_code_and_first_line(self):
        rep = {"written": False, "contradiction": "r4", "card": "updated",
               "text": "не записано: противоречие, карточка r4 — выберите на карточке"}
        import rt
        out = io.StringIO()
        with mock.patch.object(core, "edit_span", return_value=rep), redirect_stdout(out):
            code = rt.main(["edit-span", "job", "3", "3", "тут", "--by", "human"])
        self.assertEqual(code, 3)
        first, rest = out.getvalue().split("\n", 1)
        self.assertEqual(first, rep["text"])
        self.assertEqual(json.loads(rest)["contradiction"], "r4")

    def test_mcp_first_line(self):
        import server
        rep = {"written": False, "contradiction": "r4", "card": "updated",
               "text": "не записано: противоречие, карточка r4 — выберите на карточке"}
        with mock.patch.object(core, "edit_span", return_value=copy.deepcopy(rep)):
            text = server.edit_span("job", 3, 3, "тут", by="claude")
        self.assertTrue(text.startswith("не записано: противоречие, карточка r4"), text)
        self.assertEqual(json.loads(text.split("\n", 1)[1])["written"], False)


if __name__ == "__main__":
    unittest.main()
