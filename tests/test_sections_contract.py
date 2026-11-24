"""Договор версии 4 (РАР-7, этап 7.4а): вся запись пунктами, ответ — границы разделов.

Модель видит запись целиком как пронумерованные пункты (реплика; реплика длиннее 120 слов
делится по концу предложения) и отвечает разделами «с пункта N по пункт M». Программа сама
раскладывает границы в прежнее поле `evidence` (список word_id), поэтому хранение не меняется.
Задания прежних версий доделываются по своим правилам. Синтетика, sqlite, без облака.
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

TURNS = [{"turn": 1, "speaker": "Участник 1"}, {"turn": 2, "speaker": "Участник 2"}]


def _words(count, turn, *, start=0, ends=(), doubtful=()):
    """Синтетические слова задания: «словоN», конец предложения после индексов из ends."""
    return [{"word_id": f"w{index}", "text": f"слово{index}" + ("." if index in ends else ""),
             "turn": turn, "uncertain": index in doubtful} for index in range(start, start + count)]


def _task(words, turns=TURNS[:1], unlocated=0, revision=3):
    return model_tasks.build("synthetic-recording-id", revision, words, "протокол", 2, turns=turns,
                             unlocated_doubts=unlocated, points=True)


def _sections(*ranges, also=None):
    sections = [{"from": first, "to": final, "text": f"Раздел {first}–{final}."} for first, final in ranges]
    for index, refs in (also or {}).items():
        sections[index]["also"] = refs
    return sections


class PointsTests(unittest.TestCase):
    def ranges(self, task):
        return [(point["start"], point["end"]) for point in task["points"]]

    def test_point_is_a_turn(self):
        task = _task(_words(3, 1) + _words(2, 2, start=3), TURNS)
        self.assertEqual(task["version"], 4)
        self.assertEqual([(p["point"], p["turn"]) for p in task["points"]], [(1, 1), (2, 2)])
        self.assertEqual(self.ranges(task), [(0, 3), (3, 5)])

    def test_long_turn_is_split_at_the_last_sentence_end_within_120_words(self):
        task = _task(_words(300, 1, ends=(49, 99, 199)))
        self.assertEqual(self.ranges(task), [(0, 100), (100, 200), (200, 300)])

    def test_long_turn_without_sentence_end_is_split_at_120_words(self):
        task = _task(_words(250, 1))
        self.assertEqual(self.ranges(task), [(0, 120), (120, 240), (240, 250)])

    def test_sentence_end_before_closing_quote_counts(self):
        words = _words(130, 1)
        words[89]["text"] = "сказал.»"
        self.assertEqual(self.ranges(_task(words)), [(0, 90), (90, 130)])

    def test_points_cover_every_word_once_in_order(self):
        words = _words(1000, 1, ends=(10, 300, 301, 777)) + _words(5, 2, start=1000)
        task = _task(words, TURNS)
        covered = [index for point in task["points"] for index in range(point["start"], point["end"])]
        self.assertEqual(covered, list(range(1005)))
        self.assertTrue(all(point["end"] - point["start"] <= model_tasks.MAX_POINT_WORDS for point in task["points"]))
        self.assertEqual([point["point"] for point in task["points"]], list(range(1, len(task["points"]) + 1)))

    def test_canonical_recomputes_points_and_rejects_tampering(self):
        task = _task(_words(5, 1))
        self.assertEqual(model_tasks.canonical(task), task)
        tampered = copy.deepcopy(task)
        tampered["points"][0]["end"] = 4
        for bad in (tampered, {key: value for key, value in task.items() if key != "points"},
                    {**task, "version": 3}):
            with self.subTest(bad=sorted(bad)), self.assertRaises(ValueError):
                model_tasks.canonical(bad)
        with self.assertRaises(ValueError):                      # пункты только при числе непривязанных мест
            model_tasks.build("r", 1, _words(1, 1), "", 1, turns=TURNS[:1], points=True)


class ModelViewTests(unittest.TestCase):
    def test_view_has_points_and_no_word_ids_recording_or_review_revision(self):
        task = _task(_words(3, 1, doubtful=(1, 2)) + _words(2, 2, start=3, doubtful=(4,)), TURNS, unlocated=1)
        view = model_tasks.model_view(task)
        self.assertEqual(view, {
            "kind": "conspect", "version": 4, "source_revision": 3, "goal": "протокол", "unlocated_doubts": 1,
            "points": [[1, "Участник 1", "слово0 ⟦слово1 слово2⟧"], [2, "Участник 2", "слово3 ⟦слово4⟧"]]})
        text = json.dumps(view, ensure_ascii=False)
        for hidden in ("w0", "word_id", "synthetic-recording-id", "review_revision", "turn"):
            self.assertNotIn(hidden, text)

    def test_doubt_brackets_inside_word_text_are_dropped(self):
        words = _words(2, 1)
        words[0]["text"] = "⟦странно⟧"
        self.assertEqual(model_tasks.model_view(_task(words))["points"][0][2], "странно слово1")

    def test_view_is_only_for_version_4(self):
        task = model_tasks.build("r", 1, _words(1, 1), "", 1, turns=TURNS[:1], unlocated_doubts=0)
        with self.assertRaises(ValueError):
            model_tasks.model_view(task)


class SectionValidationTests(unittest.TestCase):
    def setUp(self):
        # Пять пунктов по 10 слов: реплики 1–5.
        turns = [{"turn": n, "speaker": "Участник 1"} for n in range(1, 6)]
        words = [word for n in range(5) for word in _words(10, n + 1, start=10 * n)]
        self.task = _task(words, turns)

    def check(self, sections, revision=3):
        return model_tasks.validate(self.task, {"source_revision": revision, "sections": sections},
                                    current_revision=3)

    def test_contiguous_sections_are_laid_out_as_word_evidence(self):
        result = self.check(_sections((1, 2), (3, 5), also={0: [[4, 5]]}))
        self.assertEqual([len(section["evidence"]) for section in result["sections"]], [20, 30])
        self.assertEqual(result["sections"][0]["evidence"][:2], ["w0", "w1"])
        self.assertEqual(result["sections"][0]["also"], [[f"w{i}" for i in range(30, 50)]])
        self.assertEqual(result["sections"][1]["also"], [])
        self.assertEqual((result["actor_kind"], result["human_verification"]), ("model", "none"))
        self.assertEqual(self.check(_sections((1, 5)))["sections"][0]["evidence"],
                         [f"w{i}" for i in range(50)])

    def test_gaps_overlaps_and_bad_bounds_are_rejected(self):
        cases = {
            "пропуск": _sections((1, 2), (4, 5)),
            "пересечение": _sections((1, 3), (3, 5)),
            "конец не покрыт": _sections((1, 4)),
            "не с первого пункта": _sections((2, 5)),
            "обратный отрезок": [{"from": 3, "to": 2, "text": "x"}],
            "за концом записи": _sections((1, 6)),
            "перестановка": _sections((3, 5), (1, 2)),
            "логическое значение": [{"from": True, "to": 5, "text": "x"}],
            "дробь": [{"from": 1.0, "to": 5, "text": "x"}],
            "пустой текст": [{"from": 1, "to": 5, "text": " "}],
            "лишнее поле": [{"from": 1, "to": 5, "text": "x", "evidence": ["w0"]}],
            "нет границ": [{"text": "x"}],
            "нет разделов": [],
        }
        for name, sections in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.check(sections)
        with self.assertRaises(ValueError):
            self.check(_sections((1, 5)), revision=2)
        with self.assertRaises(ValueError):
            model_tasks.validate(self.task, {"source_revision": 3, "sections": _sections((1, 5)), "x": 1},
                                 current_revision=3)

    def test_error_names_the_section_and_the_expected_point(self):
        with self.assertRaises(ValueError) as error:
            self.check(_sections((1, 2), (4, 5)))
        self.assertIn("раздел 2", str(error.exception))
        self.assertIn("с пункта 3", str(error.exception))

    def test_bad_also_references_are_rejected(self):
        cases = {
            "больше трёх": [[3, 3], [4, 4], [5, 5], [3, 4]],
            "внутри своего раздела": [[2, 2]],
            "задевает свой раздел": [[2, 3]],
            "не пара": [[3]],
            "обратная": [[4, 3]],
            "за концом": [[5, 6]],
            "не список": "3-4",
        }
        for name, refs in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.check(_sections((1, 2), (3, 5), also={0: refs}))

    def test_long_reference_is_rejected(self):
        turns = [{"turn": n, "speaker": "Участник 1"} for n in range(1, 31)]
        task = _task([word for n in range(30) for word in _words(1, n + 1, start=n)], turns)
        answer = {"source_revision": 3, "sections": _sections((1, 1), (2, 30), also={0: [[2, 22]]})}
        with self.assertRaises(ValueError):
            model_tasks.validate(task, answer, current_revision=3)
        answer["sections"][0]["also"] = [[2, 21]]
        self.assertEqual(len(model_tasks.validate(task, answer, current_revision=3)["sections"][0]["also"][0]), 20)

    def test_section_of_several_points_is_limited_to_1500_words(self):
        turns = [{"turn": n, "speaker": "Участник 1"} for n in range(1, 18)]
        task = _task([word for n in range(17) for word in _words(100, n + 1, start=100 * n)], turns)
        too_long = {"source_revision": 3, "sections": _sections((1, 16), (17, 17))}
        with self.assertRaises(ValueError) as error:
            model_tasks.validate(task, too_long, current_revision=3)
        self.assertIn("1500", str(error.exception))
        fits = {"source_revision": 3, "sections": _sections((1, 15), (16, 17))}
        self.assertEqual(len(model_tasks.validate(task, fits, current_revision=3)["sections"]), 2)


class RequestTextTests(unittest.TestCase):
    def prompt(self, task, **kwargs):
        client = llm_provider.OpenCodeTextClient()
        with mock.patch.object(client, "_generate_text", return_value={"data": {"text": "{}"}}) as call:
            client.generate(task, **kwargs)
        return call.call_args.args[0]

    def test_request_sends_only_the_view_and_explains_the_contract(self):
        task = _task(_words(3, 1, doubtful=(1,)) + _words(2, 2, start=3), TURNS)
        instructions, sent = self.prompt(task).split("Задание:\n", 1)
        self.assertEqual(json.loads(sent), model_tasks.model_view(task))
        for phrase in ("\"from\"", "\"to\"", "сразу за последним пунктом предыдущего", "1500", "also",
                       "⟦…⟧", "не проверены человеком", "unlocated_doubts", "только этими метками"):
            self.assertIn(phrase, instructions)
        for hidden in ("w0", "word_id", "synthetic-recording-id", "review_revision", "evidence"):
            self.assertNotIn(hidden, sent)
        self.assertNotIn("evidence", instructions)

    def test_repair_shows_the_whole_previous_answer(self):
        long_answer = "x" * 20_000
        prompt = self.prompt(_task(_words(2, 1)), repair_error="раздел 2 начинается с пункта 4",
                             invalid_output=long_answer)
        self.assertIn(long_answer, prompt)
        self.assertIn("раздел 2 начинается с пункта 4", prompt)

    def test_request_for_an_hour_is_much_smaller_than_blocks(self):
        # Около 7 500 слов: реплики по 30 слов, предложения по 10 слов.
        words = []
        for turn in range(1, 251):
            words += _words(30, turn, start=len(words), ends=range(len(words) + 9, len(words) + 30, 10))
        turns = [{"turn": n, "speaker": f"Участник {1 + n % 2}"} for n in range(1, 251)]
        v4 = _task(words, turns)
        new = len(self.prompt(v4))
        v3 = model_tasks.build("synthetic-recording-id", 3, words, "протокол", 2, turns=turns, unlocated_doubts=0)
        old = sum(len(self.prompt(workflow._part_task(v3, bounds))) for bounds in workflow._part_bounds(v3))
        self.assertGreaterEqual(len(workflow._part_bounds(v3)), 15)     # прежде — 15–16 вызовов
        self.assertLess(new * 5, old)


class ConsentTextTests(unittest.TestCase):
    def test_screen_consent_is_the_text_approved_for_this_stage(self):
        # Текст утверждён владельцем дословно 23.11.2026 вместе с договором версии 4; фраза о частях —
        # решение владельца 24.11.2026 для этапа 7.4в.
        html = (Path(__file__).resolve().parent.parent / "workflow.html").read_text(encoding="utf-8")
        note = html.split('id="cloud-note">', 1)[1].split("</p>", 1)[0]
        self.assertEqual(" ".join(note.split()), (
            "Конспект готовит облачная модель OpenAI (через OpenCode). Отметка «нужен конспект» означает "
            "согласие отправить ей по этой записи: весь текст стенограммы с внесёнными исправлениями, деление "
            "на реплики с условными метками («Участник 1», «Участник 2»…), пометки сомнительных мест, номер "
            "версии стенограммы и вашу цель. Имена и сведения, произнесённые в записи или написанные в цели, "
            "отправляются вместе с текстом. Запись длиннее 10 000 слов отправляется частями; к каждой части "
            "прилагается изложение предыдущих, составленное моделью. Подписи говорящих именами, аудио, время слов, путь к файлу, "
            "голосовые профили и журналы не отправляются. Без отметки всё остаётся на этом компьютере."))


class SectionsWorkflowTests(unittest.TestCase):
    """Через базу: новое задание — версии 4, один вызов на запись, файл свёртки, путь ИИ-агента."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="sections-contract-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.dbpath = self.root / "state.sqlite3"
        self.store = storage.initialize(self.dbpath, initial_display_name="Тест")
        self.addCleanup(self.store.close)
        self.sent, self.answers = [], []

    def prepare_request(self, turn_words=(600, 600), policy=None):
        """Запись из реплик по turn_words слов (слово — секунда), предложения по 10 слов, без открытых мест."""
        policy = {"require_conspect": True, "allow_cloud_model": True, **(policy or {})}
        request = workflow.create(self.store, self.source, policy)
        workflow.claim_start(self.store, request)
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        recording_id = str(uuid.uuid4())
        segments, clock = [], 0
        for number, count in enumerate(turn_words):
            words = [{"w": f"слово{clock + i}" + ("." if (clock + i) % 10 == 9 else ""), "s": clock + i,
                      "e": clock + i + 1} for i in range(count)]
            segments.append({"start": clock, "end": clock + count, "speaker": f"Спикер {1 + number % 2}",
                             "words": words})
            clock += count
        data = {"file": str(self.source), "duration": "00:20:00", "speakers": ["Спикер 1", "Спикер 2"],
                "quality": {}, "review": [], "review_done": True, "segments": segments}
        recordings.commit_worker_result(
            self.store, data, self.source, root_id, recordings.operation_context(self.store, "model"),
            recording_id=recording_id, operation_id=str(uuid.uuid4()),
            generated_at="2026-01-01T00:00:00Z", document_paths=paths)
        recordings.materialize_recording(self.store, recording_id)
        with self.store._transaction():
            self.store.db.execute("UPDATE workflow_requests SET recording_id=?, state='processed' WHERE request_id=?",
                                  (recording_id, request))
        return request

    @staticmethod
    def answer(task):
        """Верный ответ для задания любой версии: версия 4 — два раздела по границам, иначе — перечень слов."""
        if model_tasks.version(task) == 4:
            last = len(task["points"])
            if last == 1:
                return {"source_revision": task["source_revision"],
                        "sections": [{"from": 1, "to": 1, "text": "Вся встреча."}]}
            middle = last // 2
            sections = [{"from": 1, "to": middle, "text": "Начало встречи."},
                        {"from": middle + 1, "to": last, "text": "Конец встречи.", "also": [[1, 1]]}]
            return {"source_revision": task["source_revision"], "sections": sections}
        return {"source_revision": task["source_revision"],
                "sections": [{"text": "Итог блока.", "evidence": [word["word_id"] for word in task["words"]]}]}

    def client(self, before_answer=None):
        test = self

        class Client:
            def __enter__(self): return self
            def __exit__(self, *args): return False

            def generate(self, task, **kwargs):
                test.sent.append((task, kwargs))
                if before_answer:
                    before_answer(len(test.sent))
                if test.answers:
                    answer = test.answers.pop(0)
                    return answer, json.dumps(answer, ensure_ascii=False)
                answer = test.answer(task)
                return answer, json.dumps(answer, ensure_ascii=False)
        return Client

    def execute(self, request, before_answer=None, **kwargs):
        with mock.patch("llm_provider.OpenCodeTextClient", self.client(before_answer)):
            return workflow.execute_model(self.store, request, **kwargs)

    def core_call(self, function, *args, **kwargs):
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(self.dbpath)):
            return function(*args, **kwargs)

    def conspect_md(self, request):
        recording = workflow.get(self.store, request)["recording_id"]
        binding = recordings.document_binding(self.store, recording)
        files = sorted((binding["root"]).rglob("*workflow-*.md"))
        self.assertEqual(len(files), 1)
        return files[0].read_text(encoding="utf-8")

    def test_whole_recording_goes_in_one_call_and_the_conspect_is_written(self):
        request = self.prepare_request()                            # 1200 слов: прежде — 3 блока
        result = self.execute(request)
        self.assertEqual((result["state"], result["part_count"], len(self.sent)), ("completed", 1, 1))
        task = self.sent[0][0]
        self.assertEqual((task["version"], len(task["words"]), len(task["points"])), (4, 1200, 10))
        saved = workflow.get(self.store, request)["model_result"]
        evidence = [ident for section in saved["sections"] for ident in section["evidence"]]
        self.assertEqual(evidence, [word["word_id"] for word in task["words"]])
        md = self.conspect_md(request)
        self.assertIn("[00:00:00]\nНачало встречи.", md)
        self.assertIn("[00:10:00]\nКонец встречи.", md)
        self.assertIn("См. также: [00:00:00]", md)                  # отсылка второго раздела к пункту 1
        self.assertEqual(self.execute(request)["state"], "completed")
        self.assertEqual(len(self.sent), 1)                         # повтор у готовой — без облака

    def test_invalid_answer_gets_one_repair_with_the_error(self):
        request = self.prepare_request()
        self.answers.append({"source_revision": 1, "sections": [{"from": 2, "to": 10, "text": "x"}]})
        self.assertEqual(self.execute(request)["state"], "completed")
        self.assertEqual(len(self.sent), 2)
        self.assertIn("с пункта 1", self.sent[1][1]["repair_error"])

    def test_card_shown_during_the_call_does_not_fail_the_step(self):
        request = self.prepare_request()
        recording = workflow.get(self.store, request)["recording_id"]

        def show_card(_call):
            head = recordings.recording_head(self.store, recording)
            self.store.commit_player_event(
                recording, {"source_revision": head["source_revision"], "review_revision": head["review_revision"]},
                str(uuid.uuid4()), recordings.operation_context(self.store, "human"), "card_visible")
        self.assertEqual(self.execute(request, before_answer=show_card)["state"], "completed")

    def test_agent_reads_pages_of_points_and_answers_with_bounds(self):
        request = self.prepare_request()
        first = self.core_call(core.workflow_model_task, request, 0, 4)
        self.assertEqual((first["version"], first["total_points"], first["total_words"], first["next_offset"]),
                         (4, 10, 1200, 4))
        self.assertEqual([point[0] for point in first["points"]], [1, 2, 3, 4])
        self.assertNotIn("words", first)
        self.assertNotIn("w0", json.dumps(first, ensure_ascii=False))
        last = self.core_call(core.workflow_model_task, request, 8, 4)
        self.assertEqual(([point[0] for point in last["points"]], last["next_offset"]), ([9, 10], None))
        task = workflow.get(self.store, request)["model_task"]
        accepted = self.core_call(core.workflow_model_result, request, self.answer(task))
        self.assertEqual(len(accepted["sections"]), 2)
        self.assertEqual(workflow.get(self.store, request)["state"], "completed")

    def test_agent_part_is_the_whole_view(self):
        request = self.prepare_request()
        part = self.core_call(core.workflow_model_part, request, 0)
        self.assertEqual((part["part_count"], part["task"]["version"], len(part["task"]["points"])), (1, 4, 10))
        self.assertNotIn("words", part["task"])
        task = workflow.get(self.store, request)["model_task"]
        done = self.core_call(core.workflow_model_part_result, request, 0, self.answer(task))
        self.assertEqual(done["state"], "completed")
        again = self.core_call(core.workflow_model_part_result, request, 0, self.answer(task))
        self.assertEqual(again["state"], "completed")

    def test_limit_is_checked_before_the_cloud(self):
        request = self.prepare_request()
        with (mock.patch("llm_provider.MAX_WORDS", 1000), mock.patch("llm_provider.MAX_PARTS", 1),
              self.assertRaises(llm_provider.ProviderError) as error):
            self.execute(request)
        self.assertEqual((error.exception.code, len(self.sent)), ("limit", 0))
        self.assertIn("workflow_model_task", str(error.exception))

    def test_waiting_version_3_task_is_finished_by_the_old_contract(self):
        request = self.prepare_request()
        with mock.patch.object(model_tasks, "CURRENT_VERSION", 3):
            legacy = workflow.prepare_model_task(self.store, request)      # сохранено до обновления
        self.assertEqual(legacy["version"], 3)
        self.assertEqual(workflow.prepare_model_task(self.store, request), legacy)
        self.assertEqual(self.execute(request)["state"], "completed")
        self.assertEqual([task["version"] for task, _kwargs in self.sent], [3, 3, 3, 3])     # блоки по 500 слов
        self.assertEqual(workflow.get(self.store, request)["model_task"], legacy)
        self.assertEqual(workflow.prepare_model_task(self.store, request, restart=True)["version"], 4)


if __name__ == "__main__":
    unittest.main()
