import unittest

import model_tasks
import product_policy


class PolicyTests(unittest.TestCase):
    def test_ready_is_not_human_verified(self):
        result = product_policy.evaluate({}, execution="completed", document_sync="synced")
        self.assertEqual(result["deliverable"], "ready_for_request")
        self.assertEqual(result["human_verification"], "none")
        self.assertIn("результат не подтверждён человеком", result["limits"])

    def test_required_outputs_block(self):
        result = product_policy.evaluate(
            {"require_conspect": True, "require_named_speakers": True},
            execution="waiting_model", document_sync="pending", unnamed_speakers=1)
        self.assertEqual(result["deliverable"], "blocked")
        self.assertEqual(len(result["blockers"]), 4)

    def test_bad_policy_is_rejected(self):
        for policy in ({"unknown": True}, {"question_budget": True}, {"require_conspect": 1}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                product_policy.resolve(policy)

    def test_roles_do_not_automatically_change_quality_requirements(self):
        for role in ("simple", "advanced", "developer"):
            result = product_policy.evaluate(
                {"participation": role, "require_named_speakers": True},
                execution="completed", document_sync="synced", unnamed_speakers=1)
            self.assertEqual(result["deliverable"], "blocked")

    def test_exact_human_checked_result_does_not_become_ready_on_asr_completion(self):
        result = product_policy.evaluate(
            {"require_human_verification": True}, execution="completed", document_sync="synced")
        self.assertEqual(result["deliverable"], "blocked")

    def test_fully_confirmed_result_still_reports_uncertain_places(self):
        result = product_policy.evaluate(
            {}, execution="completed", document_sync="synced",
            human_verification="confirmed", unresolved=1)
        self.assertEqual(result["deliverable"], "usable_with_limits")


class ModelTaskTests(unittest.TestCase):
    def setUp(self):
        self.task = model_tasks.build("recording", 1, [{"word_id": "w1", "text": "текст"}])
        self.answer = {"source_revision": 1, "sections": [{"text": "Кратко", "evidence": ["w1"]}]}

    def test_valid_response_stays_model_authored(self):
        result = model_tasks.validate(self.task, self.answer, current_revision=1)
        self.assertEqual(result["actor_kind"], "model")
        self.assertEqual(result["human_verification"], "none")

    def test_stale_and_unknown_evidence_rejected(self):
        with self.assertRaises(ValueError):
            model_tasks.validate(self.task, self.answer, current_revision=2)
        self.answer["sections"][0]["evidence"] = ["missing"]
        with self.assertRaises(ValueError):
            model_tasks.validate(self.task, self.answer, current_revision=1)

    def test_boolean_revision_is_not_an_integer_revision(self):
        self.answer["source_revision"] = True
        with self.assertRaises(ValueError):
            model_tasks.validate(self.task, self.answer, current_revision=1)

    def test_unknown_fields_are_not_exposed(self):
        with self.assertRaises(ValueError):
            model_tasks.build("recording", 1, [{"word_id": "w1", "text": "x", "path": "private"}])
        with self.assertRaises(ValueError):
            model_tasks.build("recording", 1, [{"word_id": "w1", "text": None}])

    def test_saved_version_1_task_is_built_and_validated_as_before(self):
        # Так задание лежит в базе у запросов, созданных до версии 2: поля версии и реплик нет.
        saved = {"kind": "conspect", "recording_id": "recording", "source_revision": 1,
                 "review_revision": 1, "goal": "", "words": [{"word_id": "w1", "text": "текст", "uncertain": False}]}
        self.assertEqual(self.task, saved)
        self.assertEqual(model_tasks.version(saved), 1)
        self.assertEqual(model_tasks.canonical(saved), saved)
        self.assertEqual(model_tasks.validate(saved, self.answer, current_revision=1)["sections"],
                         self.answer["sections"])
        for extra in ({"turns": []}, {"version": 1}, {"version": 3}, {"version": True}, {"path": "private"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                model_tasks.validate({**saved, **extra}, self.answer, current_revision=1)
        with self.assertRaises(ValueError):       # номер реплики — поле версии 2
            model_tasks.build("recording", 1, [{"word_id": "w1", "text": "текст", "turn": 1}])


class ModelTaskVersion2Tests(unittest.TestCase):
    def setUp(self):
        self.turns = [{"turn": 1, "speaker": "Участник 1"}, {"turn": 2, "speaker": "Участник 2"},
                      {"turn": 3, "speaker": "Участник 1"}]
        self.words = [{"word_id": "w1", "text": "добрый", "turn": 1}, {"word_id": "w2", "text": "день.", "turn": 1},
                      {"word_id": "w3", "text": "да", "turn": 2, "uncertain": True},
                      {"word_id": "w4", "text": "начнём", "turn": 3}]
        self.task = model_tasks.build("recording", 1, self.words, "протокол", 2, turns=self.turns)
        self.answer = {"source_revision": 1, "sections": [
            {"text": "Участник 1 открыл встречу.", "evidence": ["w1", "w2", "w4"]},
            {"text": "Участник 2 согласился.", "evidence": ["w3"]}]}

    def test_task_has_explicit_version_turns_and_word_turn_numbers(self):
        self.assertEqual(self.task["version"], 2)
        self.assertEqual(model_tasks.version(self.task), 2)
        self.assertEqual(self.task["turns"], self.turns)
        self.assertEqual([word["turn"] for word in self.task["words"]], [1, 1, 2, 3])
        self.assertEqual(self.task["words"][2]["uncertain"], True)
        self.assertEqual(set(self.task), {"kind", "version", "recording_id", "source_revision",
                                          "review_revision", "goal", "turns", "words"})
        result = model_tasks.validate(self.task, self.answer, current_revision=1)
        self.assertEqual(result["actor_kind"], "model")
        self.assertEqual(result["human_verification"], "none")

    def test_speaker_is_only_a_conditional_label(self):
        model_tasks.build("recording", 1, [{"word_id": "w1", "text": "да", "turn": 1}],
                          turns=[{"turn": 1, "speaker": model_tasks.UNKNOWN_SPEAKER}])
        for speaker in ("Анна", "Спикер 1", "Участник", "Участник 0", "Участник 1 (Анна)", "", None, 1):
            with self.subTest(speaker=speaker), self.assertRaises(ValueError):
                model_tasks.build("recording", 1, [{"word_id": "w1", "text": "да", "turn": 1}],
                                  turns=[{"turn": 1, "speaker": speaker}])

    def test_unknown_fields_are_rejected_in_version_2(self):
        word = {"word_id": "w1", "text": "да", "turn": 1}
        turn = {"turn": 1, "speaker": "Участник 1"}
        for words, turns in (([{**word, "start": 1.5}], [turn]),
                             ([{**word, "speaker": "Анна"}], [turn]),
                             ([word], [{**turn, "name": "Анна"}]),
                             ([word], [{**turn, "start": 0}]),
                             ([{"word_id": "w1", "text": "да"}], [turn]),
                             ([{**word, "turn": True}], [turn])):
            with self.subTest(words=words, turns=turns), self.assertRaises(ValueError):
                model_tasks.build("recording", 1, words, turns=turns)
        for extra in ({"speaker_names": {}}, {"version": 3}, {"file": "private"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                model_tasks.validate({**self.task, **extra}, self.answer, current_revision=1)
        without_turns = {key: value for key, value in self.task.items() if key != "turns"}
        with self.assertRaises(ValueError):
            model_tasks.validate(without_turns, self.answer, current_revision=1)

    def test_words_must_follow_the_turn_list(self):
        word = lambda ident, turn: {"word_id": ident, "text": "да", "turn": turn}
        one, two = {"turn": 1, "speaker": "Участник 1"}, {"turn": 2, "speaker": "Участник 2"}
        for words, turns in (([word("a", 1)], [one, two]),                    # реплика без слов
                             ([word("a", 1), word("b", 2)], [one]),           # слово неизвестной реплики
                             ([word("a", 2), word("b", 1)], [one, two]),      # порядок нарушен
                             ([word("a", 1), word("b", 2), word("c", 1)], [one, two]),   # реплика разорвана
                             ([word("a", 1)], [one, one]),                    # повтор номера
                             ([word("a", 1)], [two, one])):
            with self.subTest(words=words, turns=turns), self.assertRaises(ValueError):
                model_tasks.build("recording", 1, words, turns=turns)

    def test_every_word_must_still_be_covered_by_evidence(self):
        self.answer["sections"][1]["evidence"] = ["w4"]
        self.answer["sections"][0]["evidence"] = ["w1", "w2"]
        with self.assertRaisesRegex(ValueError, "не все слова"):
            model_tasks.validate(self.task, self.answer, current_revision=1)
        self.answer["sections"][0]["evidence"] = ["w1", "w2", "missing"]
        with self.assertRaises(ValueError):
            model_tasks.validate(self.task, self.answer, current_revision=1)
        with self.assertRaises(ValueError):
            model_tasks.validate(self.task, self.answer, current_revision=2)


if __name__ == "__main__":
    unittest.main()
