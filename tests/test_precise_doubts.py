"""Точные пометки сомнений в задании модели (РАР-7, этап 7.3). Синтетика, sqlite, без облака.

Задание версии 3 помечает сомнительными (`uncertain: true`) только слова открытых мест:
по ссылкам на слова, по времени места, а у паузы (место со временем, в которое не попало
ни одно слово) — два соседних слова. Только место без ссылок и без годного времени, как
раньше, делает сомнительными все слова записи; число таких мест модель видит в поле
`unlocated_doubts`. Задания версий 1 и 2 сверяются по своим прежним правилам.
"""
import copy
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import core
import llm_provider
import model_tasks
import recordings
import storage
import workflow


def _word(ident, start, end):
    return {"word_id": ident, "w": ident, "s": start, "e": end}


def _snapshots(words, places, *, split=None):
    """Снимки без базы: одна или две реплики из слов, места разбора как есть."""
    split = len(words) if split is None else split
    segments = [{"start": 0, "end": 1, "speaker": "Спикер 1", "words": words[:split]}]
    if words[split:]:
        segments.append({"start": 1, "end": 2, "speaker": "Спикер 2", "words": words[split:]})
    return {"transcript": {"revision": 1, "payload": {"segments": segments}},
            "review": {"revision": 1, "payload": {"review": places}}}


REQUEST = {"recording_id": "synthetic-recording", "policy": {"goal": ""}}

# Кусочек стенограммы: слова с временем (секунды) и пауза 4–6 с без слов.
WORDS = [_word("a", 0.0, 1.0), _word("b", 1.0, 2.0), _word("c", 2.0, 3.0), _word("d", 3.0, 4.0),
         _word("e", 6.0, 7.0), _word("f", 7.0, 8.0)]


def _marked(places, words=WORDS, **kwargs):
    task = workflow._build_task(REQUEST, _snapshots(words, places, **kwargs))
    return [word["word_id"] for word in task["words"] if word["uncertain"]], task


class PreciseMarksTests(unittest.TestCase):
    def test_place_with_word_ids_marks_exactly_those_words(self):
        marked, task = _marked([{"id": "r1", "status": "open", "word_ids": ["b", "c"], "t1": 0, "t2": 8}])
        self.assertEqual(marked, ["b", "c"])
        self.assertEqual(task["unlocated_doubts"], 0)

    def test_place_with_time_marks_only_words_inside_it(self):
        # Правило: слово [s, e] сомнительно, если его начало s лежит в [t1 − 0,01; t2 + 0,01]
        # или слово перекрывает место по времени с положительной длиной (s < t2 и e > t1).
        marked, task = _marked([{"id": "r1", "status": "open", "t1": 1.0, "t2": 2.0}])
        self.assertEqual(marked, ["b", "c"])     # «a» лишь касается начала места концом — не помечено
        self.assertEqual(task["unlocated_doubts"], 0)
        self.assertEqual(task["version"], model_tasks.CURRENT_VERSION)

    def test_boundary_cases_of_the_intersection_rule(self):
        cases = (
            ((1.0, 2.0), ["b", "c"]),       # «a» кончается ровно на t1 — нет; «c» начинается ровно на t2 — да
            ((1.0, 1.995), ["b", "c"]),     # начало «c» в пределах допуска 0,01 после t2
            ((1.0, 1.98), ["b"]),           # начало «c» дальше допуска, перекрытия нет
            ((1.015, 2.0), ["b", "c"]),     # «b» начинается раньше t1, но перекрывает место
            ((2.5, 2.5), ["c"]),            # точка внутри слова
            ((3.0, 3.0), ["d"]),            # точка на стыке слов: слово, которое здесь начинается
            ((0.5, 0.6), ["a"]),            # место внутри слова
        )
        for (t1, t2), expected in cases:
            with self.subTest(t1=t1, t2=t2):
                marked, _task = _marked([{"id": "r1", "status": "open", "t1": t1, "t2": t2}])
                self.assertEqual(marked, expected)

    def test_word_overlapping_the_place_start_is_marked(self):
        words = [_word("x", 0.0, 1.5), _word("y", 1.5, 3.0)]
        marked, _task = _marked([{"id": "r1", "status": "open", "t1": 1.0, "t2": 1.2}], words)
        self.assertEqual(marked, ["x"])

    def test_pause_marks_the_two_neighbouring_words(self):
        marked, task = _marked([{"id": "r1", "status": "open", "t1": 4.2, "t2": 5.8}])
        self.assertEqual(marked, ["d", "e"])
        self.assertEqual(task["unlocated_doubts"], 0)

    def test_pause_at_the_edges_marks_only_the_existing_neighbour(self):
        marked, _task = _marked([{"id": "r1", "status": "open", "t1": 9.0, "t2": 12.0}])
        self.assertEqual(marked, ["f"])
        words = [_word("p", 5.0, 6.0), _word("q", 6.0, 7.0)]
        marked, _task = _marked([{"id": "r1", "status": "open", "t1": 0.0, "t2": 4.0}], words)
        self.assertEqual(marked, ["p"])

    def test_place_without_word_ids_and_without_time_marks_all_and_is_counted(self):
        for place in ({"id": "r1", "status": "open"}, {"id": "r1", "status": "open", "t1": 1.0},
                      {"id": "r1", "status": "open", "t1": 3.0, "t2": 2.0},
                      {"id": "r1", "status": "open", "t1": "1", "t2": 2.0},
                      {"id": "r1", "status": "open", "t1": True, "t2": 2.0},
                      {"id": "r1", "status": "open", "word_ids": [], "t1": float("nan"), "t2": 2.0}):
            with self.subTest(place=place):
                marked, task = _marked([place, {"id": "r2", "status": "open", "word_ids": ["a"]}])
                self.assertEqual(marked, [w["word_id"] for w in WORDS])
                self.assertEqual(task["unlocated_doubts"], 1)

    def test_place_with_time_but_no_timed_words_counts_as_unlocated(self):
        words = [{"word_id": "u", "w": "u"}, {"word_id": "v", "w": "v"}]
        marked, task = _marked([{"id": "r1", "status": "open", "t1": 1.0, "t2": 2.0}], words)
        self.assertEqual(marked, ["u", "v"])
        self.assertEqual(task["unlocated_doubts"], 1)

    def test_only_open_and_human_places_count(self):
        places = [{"id": "r1", "status": "resolved", "t1": 0.0, "t2": 8.0},
                  {"id": "r2", "status": "resolved"},
                  {"id": "r3", "status": "human", "t1": 7.0, "t2": 7.0},
                  {"id": "r4", "t1": 0.2, "t2": 0.2}]                 # без статуса — открыто
        marked, task = _marked(places)
        self.assertEqual(marked, ["a", "f"])
        self.assertEqual(task["unlocated_doubts"], 0)

    def test_marks_cross_turns(self):
        marked, task = _marked([{"id": "r1", "status": "open", "t1": 1.5, "t2": 2.5}], split=2)
        self.assertEqual(marked, ["b", "c"])
        self.assertEqual([turn["turn"] for turn in task["turns"]], [1, 2])


class Version3ContractTests(unittest.TestCase):
    def setUp(self):
        self.turns = [{"turn": 1, "speaker": "Участник 1"}]
        self.words = [{"word_id": "w1", "text": "да", "turn": 1, "uncertain": True}]
        self.task = model_tasks.build("recording", 1, self.words, "", 1, turns=self.turns, unlocated_doubts=2)
        self.answer = {"source_revision": 1, "sections": [{"text": "Итог.", "evidence": ["w1"]}]}

    def test_version_3_task_has_the_count_field(self):
        self.assertIn(3, model_tasks.VERSIONS)
        self.assertEqual(self.task["version"], 3)
        self.assertEqual(self.task["unlocated_doubts"], 2)
        self.assertEqual(model_tasks.version(self.task), 3)
        self.assertEqual(model_tasks.canonical(self.task), self.task)
        self.assertEqual(set(self.task), {"kind", "version", "recording_id", "source_revision", "review_revision",
                                          "goal", "turns", "unlocated_doubts", "words"})
        self.assertEqual(model_tasks.validate(self.task, self.answer, current_revision=1)["sections"],
                         self.answer["sections"])

    def test_versions_1_and_2_are_still_built_without_the_field(self):
        v2 = model_tasks.build("recording", 1, self.words, "", 1, turns=self.turns)
        self.assertEqual((v2["version"], model_tasks.version(v2)), (2, 2))
        self.assertNotIn("unlocated_doubts", v2)
        v1 = model_tasks.build("recording", 1, [{"word_id": "w1", "text": "да"}])
        self.assertEqual(model_tasks.version(v1), 1)
        with self.assertRaises(ValueError):       # поле только вместе с репликами (версия 3)
            model_tasks.build("recording", 1, [{"word_id": "w1", "text": "да"}], unlocated_doubts=0)
        with self.assertRaises(ValueError):
            model_tasks.canonical({**v2, "unlocated_doubts": 0})

    def test_bad_count_or_unknown_version_is_rejected(self):
        for value in (-1, True, 1.0, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model_tasks.canonical({**self.task, "unlocated_doubts": value})
        without = {key: value for key, value in self.task.items() if key != "unlocated_doubts"}
        with self.assertRaises(ValueError):
            model_tasks.canonical(without)
        for version in (5, 0, True, "3"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                model_tasks.version({**self.task, "version": version})

    def test_count_is_part_of_the_content_fingerprint(self):
        self.assertNotEqual(workflow.content_fingerprint({**self.task, "unlocated_doubts": 3}),
                            workflow.content_fingerprint(self.task))

    def test_part_task_keeps_version_and_count(self):
        part = workflow._part_task(self.task, (0, 1))
        self.assertEqual((part["version"], part["unlocated_doubts"]), (3, 2))


class Version3RequestTextTests(unittest.TestCase):
    def prompt(self, task):
        client = llm_provider.OpenCodeTextClient()
        with mock.patch.object(client, "_request", return_value={"data": {"text": "{}"}}) as call:
            client.generate(task)
        return call.call_args.args[1]["prompt"]

    def test_version_3_sends_only_the_count_in_addition_and_explains_the_marks(self):
        task = model_tasks.build("synthetic-recording-id", 3,
                                 [{"word_id": "w1", "text": "Добрый", "turn": 1, "uncertain": True},
                                  {"word_id": "w2", "text": "день.", "turn": 1}],
                                 "протокол", 2, turns=[{"turn": 1, "speaker": "Участник 1"}], unlocated_doubts=1)
        prompt = self.prompt(task)
        instructions, sent = prompt.split("Задание:\n", 1)
        self.assertEqual(json.loads(sent), {
            "kind": "conspect", "source_revision": 3, "review_revision": 2, "goal": "протокол",
            "turns": [{"turn": 1, "speaker": "Участник 1"}], "unlocated_doubts": 1,
            "words": [{"word_id": "w1", "text": "Добрый", "uncertain": True, "turn": 1},
                      {"word_id": "w2", "text": "день.", "uncertain": False, "turn": 1}]})
        self.assertIn("uncertain: true", instructions)
        self.assertIn("не проверено человеком", instructions)
        self.assertIn("не цитируй его дословно", instructions)
        self.assertIn("unlocated_doubts", instructions)
        self.assertNotIn("synthetic-recording-id", prompt)
        self.assertNotIn("recording_id", prompt)

    def test_version_2_request_text_has_no_new_rules(self):
        task = model_tasks.build("synthetic", 1, [{"word_id": "w1", "text": "слово", "turn": 1}], "", 1,
                                 turns=[{"turn": 1, "speaker": "Участник 1"}])
        prompt = self.prompt(task)
        self.assertNotIn("unlocated_doubts", prompt)
        self.assertNotIn("не цитируй его дословно", prompt)


class StoredTaskCompatibilityTests(unittest.TestCase):
    """Задания версии 2 сверяются по прежним правилам; новые — версии 3 (через базу)."""

    def setUp(self):
        # Договор версий 1–3 (ответ — перечень слов, блоки по 500 слов): так доделываются задания,
        # сохранённые до этапа 7.4. Договор версии 4 проверяет tests/test_sections_contract.py.
        patcher = mock.patch.object(model_tasks, "CURRENT_VERSION", 3)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.root = Path(tempfile.mkdtemp(prefix="precise-doubts-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.dbpath = self.root / "state.sqlite3"
        self.store = storage.initialize(self.dbpath, initial_display_name="Тест")
        self.addCleanup(self.store.close)
        self.sent = []

    def prepare_request(self, policy=None):
        """Запись из 4 слов (0–4 с) и открытое свежее место без ссылок на слова: 0–0,5 с (слово w0)."""
        policy = {"require_conspect": True, "allow_cloud_model": True, **(policy or {})}
        request = workflow.create(self.store, self.source, policy)
        workflow.claim_start(self.store, request)
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        recording_id = str(uuid.uuid4())
        data = {"file": str(self.source), "duration": "00:00:04", "speakers": ["Спикер 1", "Спикер 2"],
                "quality": {}, "review": [], "review_done": False,
                "segments": [{"start": 0, "end": 2, "speaker": "Спикер 1",
                              "words": [{"w": f"w{i}", "s": i, "e": i + 1} for i in range(2)]},
                             {"start": 2, "end": 4, "speaker": "Спикер 2",
                              "words": [{"w": f"w{i}", "s": i, "e": i + 1} for i in range(2, 4)]}]}
        recordings.commit_worker_result(
            self.store, data, self.source, root_id, recordings.operation_context(self.store, "model"),
            recording_id=recording_id, operation_id=str(uuid.uuid4()),
            generated_at="2026-01-01T00:00:00Z", document_paths=paths)
        recordings.materialize_recording(self.store, recording_id)
        with self.store._transaction():
            self.store.db.execute("UPDATE workflow_requests SET recording_id=?, state='processed' WHERE request_id=?",
                                  (recording_id, request))
        self.add_place(request, {"id": "fresh-1", "status": "open", "t1": 0.0, "t2": 0.5})
        return request

    def recording(self, request):
        return workflow.get(self.store, request)["recording_id"]

    def add_place(self, request, place):
        recording = self.recording(request)
        current = self.store.get_current(recording)
        review = copy.deepcopy(current["review"]["payload"])
        review["review"] = list(review.get("review", [])) + [place]
        binding = recordings.document_binding(self.store, recording)
        self.store.commit_change_with_document(
            recording, {"source_revision": current["transcript"]["revision"],
                        "review_revision": current["review"]["revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
            transcript=None, review=review, root_id=binding["root_id"], generated_at="2026-01-01T00:00:01Z")
        recordings.materialize_recording(self.store, recording)

    def show_card(self, request):
        recording = self.recording(request)
        head = recordings.recording_head(self.store, recording)
        self.store.commit_player_event(
            recording, {"source_revision": head["source_revision"], "review_revision": head["review_revision"]},
            str(uuid.uuid4()), recordings.operation_context(self.store, "human"), "card_visible")

    def client(self):
        test = self

        class Client:
            def __enter__(self): return self
            def __exit__(self, *args): return False

            def generate(self, task, **_kwargs):
                test.sent.append(task)
                return {"source_revision": task["source_revision"], "sections": [{
                    "text": "Итог блока.", "evidence": [word["word_id"] for word in task["words"]]}]}, "{}"
        return Client

    def execute(self, request, **kwargs):
        with mock.patch("llm_provider.OpenCodeTextClient", self.client()):
            return workflow.execute_model(self.store, request, **kwargs)

    def status(self, request):
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(self.dbpath)):
            return core.workflow_status(request)

    def v2_task(self, request):
        """Задание версии 2 из текущих снимков — так его построил бы код до этапа 7.3."""
        saved = workflow.get(self.store, request)
        return workflow._build_task(saved, self.store.get_current(saved["recording_id"]), task_version=2)

    def save_task(self, request, task):
        with self.store._transaction():
            self.store.db.execute("UPDATE workflow_requests SET model_task_json=? WHERE request_id=?",
                                  (storage._json(task), request))

    def test_new_task_is_version_3_with_precise_marks(self):
        request = self.prepare_request()
        self.assertEqual(self.execute(request)["state"], "completed")
        task = self.sent[0]
        self.assertEqual((task["version"], task["unlocated_doubts"]), (3, 0))
        self.assertEqual([word["uncertain"] for word in task["words"]], [True, False, False, False])

    def test_version_2_task_built_by_old_rules_marks_everything(self):
        request = self.prepare_request()
        task = self.v2_task(request)
        self.assertEqual(task["version"], 2)
        self.assertTrue(all(word["uncertain"] for word in task["words"]))

    def test_completed_version_2_conspect_stays_current_after_update(self):
        request = self.prepare_request()
        self.execute(request)
        self.save_task(request, self.v2_task(request))         # свёртка, сделанная до обновления
        for _ in range(2):
            readiness = self.status(request)["readiness"]
            self.assertEqual(readiness["execution"], "completed")
            self.assertNotIn("заказанный конспект отсутствует", readiness["blockers"])
            self.show_card(request)
        calls = len(self.sent)
        self.assertEqual(self.execute(request)["state"], "completed")
        self.assertEqual(len(self.sent), calls)

    def test_waiting_version_2_task_is_finished_unchanged(self):
        request = self.prepare_request()
        workflow.prepare_model_task(self.store, request)
        legacy = self.v2_task(request)
        self.save_task(request, legacy)
        self.show_card(request)
        self.assertEqual(workflow.prepare_model_task(self.store, request), legacy)
        self.assertEqual(self.execute(request)["state"], "completed")
        self.assertTrue(all(part["version"] == 2 and "unlocated_doubts" not in part for part in self.sent))
        self.assertEqual(workflow.get(self.store, request)["model_task"], legacy)
        self.assertEqual(workflow.prepare_model_task(self.store, request, restart=True)["version"], 3)

    def confirm(self, request):
        snapshot = workflow.verification_snapshot(self.store, request)
        workflow.confirm_verification(self.store, snapshot)
        self.assertEqual(self.status(request)["readiness"]["human_verification"], "confirmed")

    def test_verification_without_model_task_follows_version_3_marks(self):
        request = self.prepare_request({"require_human_verification": True})
        self.confirm(request)
        self.add_place(request, {"id": "fresh-2", "status": "open", "t1": 3.0, "t2": 3.5})
        self.assertEqual(self.status(request)["readiness"]["human_verification"], "none")

    def test_verification_with_saved_version_2_task_follows_old_marks(self):
        request = self.prepare_request({"require_human_verification": True})
        self.execute(request)
        self.save_task(request, self.v2_task(request))
        self.confirm(request)
        # По правилам версии 2 сомнительны все слова и до, и после: отметка не снимается.
        self.add_place(request, {"id": "fresh-2", "status": "open", "t1": 3.0, "t2": 3.5})
        self.assertEqual(self.status(request)["readiness"]["human_verification"], "confirmed")


if __name__ == "__main__":
    unittest.main()
