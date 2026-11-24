"""Длинная запись частями по паузе с изложением предыдущих частей (РАР-7, этап 7.4в).

Запись до 10 000 слов — как раньше: одно задание версии 4 без поля `parts`, запрос и хэш
побайтно прежние. Длиннее — N = ceil(слов / 10 000) частей (не больше 6), граница каждой —
самая длинная пауза между пунктами в окне ±10 % размера части у цели k·W/N, без превышения
10 000 слов в части; без пауз — граница пункта, ближайшая к цели. Части уходят по очереди,
к части k прилагается изложение частей 1…k−1 (тексты принятых разделов), сокращённое
детерминированно до 3 000 слов. Синтетика, sqlite, без облака.
"""
import hashlib
import json
import math
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
from test_sections_contract import TURNS, _task, _words

ROOT = Path(__file__).resolve().parent.parent
CONSENT_PARTS = ("Запись длиннее 10 000 слов отправляется частями; к каждой части прилагается изложение "
                 "предыдущих, составленное моделью")

# Задание версии 4 до 7.4в, снятое с main 55248e9 до правок: задание, хэш и текст запроса не меняются.
GOLDEN_TASK = json.loads(
    '{"goal": "протокол", "kind": "conspect", "points": [{"end": 3, "point": 1, "start": 0, "turn": 1}, '
    '{"end": 5, "point": 2, "start": 3, "turn": 2}], "recording_id": "synthetic-recording-id", '
    '"review_revision": 2, "source_revision": 3, "turns": [{"speaker": "Участник 1", "turn": 1}, '
    '{"speaker": "Участник 2", "turn": 2}], "unlocated_doubts": 0, "version": 4, "words": '
    '[{"text": "слово0", "turn": 1, "uncertain": false, "word_id": "w0"}, {"text": "слово1", "turn": 1, '
    '"uncertain": true, "word_id": "w1"}, {"text": "слово2", "turn": 1, "uncertain": false, "word_id": "w2"}, '
    '{"text": "слово3", "turn": 2, "uncertain": false, "word_id": "w3"}, {"text": "слово4.", "turn": 2, '
    '"uncertain": false, "word_id": "w4"}]}')
GOLDEN_HASH = "270df125fb3e297ad6083a3e5b7226740c64266df31e4abaa065455fdc9f71fd"
GOLDEN_FINGERPRINT = "55fc2935bff7d2aec338d0f32036216afa2f8652fcbc295cfa7f17c8eeeb3409"
GOLDEN_PROMPT_SHA = "80b741b7583f41fca1383dcf0253d089af6db54fa0f7a5bb10487b24bdad3219"
# Запись 5 000 + 4 990 слов через базу (prepare_request ниже без пауз): текст запроса на main 55248e9.
GOLDEN_9990_PROMPT_SHA = "f705968eccf495882db6798333c664c748d1281fbf01aba9ecf7d56a94f2227c"


def prompt_of(task, **kwargs):
    client = llm_provider.OpenCodeTextClient()
    with mock.patch.object(client, "_generate_text", return_value={"data": {"text": "{}"}}) as call:
        client.generate(task, **kwargs)
    return call.call_args.args[0]


def turn_points(turn_sizes):
    """Пункты версии 4 из реплик заданных размеров (каждая реплика ≤ 120 слов — один пункт)."""
    words = []
    for number, size in enumerate(turn_sizes, 1):
        words += _words(size, number, start=len(words))
    turns = [{"turn": n, "speaker": "Участник 1"} for n in range(1, len(turn_sizes) + 1)]
    return _task(words, turns)["points"], len(words)


def times_with_gaps(total, gaps):
    """Время слов: слово — секунда, перед словом с индексом из gaps — пауза в столько секунд."""
    result, clock = [], 0.0
    for index in range(total):
        clock += gaps.get(index, 0)
        result.append((clock, clock + 1))
        clock += 1
    return result


def part_answer(task, text="Раздел части."):
    """Верный ответ части (или всего задания): разделы подряд по 12 пунктов (не длиннее 1 500 слов)."""
    first, final = model_tasks.part_range(task)
    sections = [{"from": start, "to": min(start + 11, final), "text": f"{text} Пункты {start}–{min(start + 11, final)}."}
                for start in range(first, final + 1, 12)]
    return {"source_revision": task["source_revision"], "sections": sections}


class GoldenTests(unittest.TestCase):
    def test_short_task_request_and_hash_are_unchanged(self):
        task = _task(_words(3, 1, doubtful=(1,)) + _words(2, 2, start=3, ends=(4,)), TURNS)
        self.assertEqual(task, GOLDEN_TASK)
        self.assertNotIn("parts", task)
        self.assertEqual(model_tasks.canonical(GOLDEN_TASK), GOLDEN_TASK)
        self.assertEqual(storage._hash(task), GOLDEN_HASH)
        self.assertEqual(workflow.content_fingerprint(GOLDEN_TASK), GOLDEN_FINGERPRINT)
        self.assertEqual(hashlib.sha256(prompt_of(GOLDEN_TASK).encode()).hexdigest(), GOLDEN_PROMPT_SHA)
        self.assertEqual(workflow._part_bounds(GOLDEN_TASK), [(0, 5)])

    def test_saved_task_without_parts_is_one_part(self):
        self.assertEqual(len(workflow._part_bounds(GOLDEN_TASK)), 1)
        self.assertEqual(model_tasks.part_range(GOLDEN_TASK), (1, 2))


class PlanPartsTests(unittest.TestCase):
    def test_up_to_the_limit_there_are_no_parts(self):
        points, total = turn_points([100] * 100)
        self.assertIsNone(workflow._plan_parts(points, times_with_gaps(total, {})))

    def test_boundaries_on_the_longest_pause_within_ten_percent(self):
        points, total = turn_points([100] * 250)                 # 25 000 слов: 3 части, цели 8 333 и 16 667
        gaps = {7000: 20, 8000: 10, 8600: 5,                     # 20 с — вне окна ±833 слова
                16700: 8, 17400: 15}
        parts = workflow._plan_parts(points, times_with_gaps(total, gaps))
        self.assertEqual(parts, [[1, 80], [81, 174], [175, 250]])

    def test_without_pauses_the_nearest_boundary_to_the_target(self):
        points, total = turn_points([100] * 250)
        self.assertEqual(workflow._plan_parts(points, [None] * total), [[1, 83], [84, 167], [168, 250]])

    def test_55000_words_give_six_parts_each_within_the_limit(self):
        for sizes in ([100] * 550, [55_000]):                     # реплики и монолог без концов предложений
            points, total = turn_points(sizes) if sizes[0] <= 120 else self.monologue(sizes[0])
            parts = workflow._plan_parts(points, [None] * total)
            self.assertEqual(len(parts), 6)
            sizes_ = [points[b - 1]["end"] - points[a - 1]["start"] for a, b in parts]
            self.assertEqual(sum(sizes_), 55_000)
            self.assertTrue(all(size <= llm_provider.MAX_WORDS for size in sizes_), sizes_)
            self.assertEqual(parts, workflow._plan_parts(points, [None] * total))      # детерминировано

    @staticmethod
    def monologue(count):
        task = _task(_words(count, 1))
        return task["points"], count

    def test_more_than_six_parts_is_refused(self):
        points, total = turn_points([100] * 601)                 # 60 100 слов: 7 частей
        with self.assertRaises(ValueError):
            workflow._plan_parts(points, [None] * total)
        request = {"state": "processed"}
        problem = workflow._limit_problem(request, 60_100)
        self.assertIn("60000 слов", problem)
        self.assertIn("workflow_model_task", problem)
        self.assertIsNone(workflow._limit_problem(request, 55_000))


class PartTaskTests(unittest.TestCase):
    def setUp(self):
        words = []
        for turn in range(1, 7):
            words += _words(10, turn, start=len(words))
        turns = [{"turn": n, "speaker": f"Участник {1 + n % 2}"} for n in range(1, 7)]
        self.task = model_tasks.build("synthetic-recording-id", 3, words, "протокол", 2, turns=turns,
                                      unlocated_doubts=0, points=True, parts=[[1, 2], [3, 6]])

    def test_parts_are_checked_and_not_shown_to_the_model(self):
        self.assertEqual(model_tasks.canonical(self.task), self.task)
        for bad in ([[1, 2], [4, 6]], [[1, 6]], [[1, 2], [3, 5]], [[2, 2], [3, 6]], [[1, 3], [3, 6]]):
            with self.assertRaises(ValueError):
                model_tasks.canonical({**self.task, "parts": bad})
        self.assertNotIn("parts", model_tasks.model_view(self.task))
        self.assertEqual(workflow._part_bounds(self.task), [(0, 20), (20, 60)])

    def test_part_view_has_only_its_points_with_through_numbers(self):
        second = model_tasks.part_task(self.task, 2)
        self.assertEqual(model_tasks.part_range(second), (3, 6))
        view = model_tasks.model_view(second)
        self.assertEqual([point[0] for point in view["points"]], [3, 4, 5, 6])
        self.assertNotIn("parts", view)
        self.assertNotIn("part", view)

    def test_section_may_not_cross_the_part_boundary(self):
        first = model_tasks.part_task(self.task, 1)
        model_tasks.validate(first, part_answer(first), current_revision=3)
        crossing = {"source_revision": 3, "sections": [{"from": 1, "to": 3, "text": "x"}]}
        with self.assertRaises(ValueError):
            model_tasks.validate(first, crossing, current_revision=3)
        outside = {"source_revision": 3, "sections": [{"from": 1, "to": 2, "text": "x", "also": [[5, 5]]}]}
        with self.assertRaises(ValueError):
            model_tasks.validate(first, outside, current_revision=3)
        second = model_tasks.part_task(self.task, 2)
        early = {"source_revision": 3, "sections": [{"from": 1, "to": 6, "text": "x"}]}
        with self.assertRaises(ValueError):
            model_tasks.validate(second, early, current_revision=3)
        accepted = model_tasks.validate(second, part_answer(second), current_revision=3)
        self.assertEqual(accepted["sections"][0]["evidence"], [f"w{i}" for i in range(20, 60)])

    def test_part_request_names_the_part_and_carries_the_summary(self):
        second = model_tasks.part_task(self.task, 2)
        prompt = prompt_of(second, summary=[["Открыли встречу. Договорились о повестке."]])
        instructions, rest = prompt.split("Задание:\n", 1)
        self.assertIn("часть 2 из 2", instructions)
        self.assertIn("с пункта 3", instructions)
        block = ("Изложение предыдущих частей (составлено моделью; это не стенограмма, не цитируй его)")
        self.assertIn(block, prompt)
        self.assertIn("Открыли встречу. Договорились о повестке.", prompt)
        view_text = rest.split("\n", 1)[0]
        self.assertEqual(json.loads(view_text), model_tasks.model_view(second))
        self.assertNotIn("Открыли встречу", view_text)
        self.assertNotIn("Изложение предыдущих частей", prompt_of(model_tasks.part_task(self.task, 1)))


class SummaryTests(unittest.TestCase):
    @staticmethod
    def words_of(text):
        return text.split()

    def test_short_summary_is_unchanged(self):
        text = llm_provider.summary_block([["Первый раздел. Ещё фраза."], ["Второй."]])
        self.assertIn("Первый раздел. Ещё фраза.", text)
        self.assertNotIn("сокращено", text)
        self.assertNotIn("ранние части опущены", text)

    def test_earliest_parts_first_lose_all_but_heading_and_first_sentence(self):
        body = " ".join(["слово"] * 600)
        section = f"Заголовок раздела\nПервое предложение раздела. {body}."
        previous = [[section, section], [section, section], [section]]        # около 3 000+ слов
        text = llm_provider.summary_block(previous)
        self.assertIn("сокращено", text)
        lines = text.splitlines()
        part1 = lines[lines.index("Часть 1:") + 1:lines.index("Часть 2:")]
        self.assertTrue(all(line.endswith("Заголовок раздела. Первое предложение раздела.") for line in part1), part1)
        self.assertIn(body, text)                                            # поздние части целы
        self.assertLessEqual(self.count(text), llm_provider.MAX_SUMMARY_WORDS)

    def test_then_earliest_parts_are_omitted_without_cutting_words(self):
        sentence = " ".join(["слово"] * 400) + "."
        previous = [[sentence] * 2 for _ in range(5)]                         # каждая часть ~800 слов
        text = llm_provider.summary_block(previous)
        self.assertIn("ранние части опущены", text)
        self.assertIn("сокращено", text)
        self.assertLessEqual(self.count(text), llm_provider.MAX_SUMMARY_WORDS)
        self.assertNotIn("Часть 1:", text)
        self.assertIn("Часть 5:", text)
        allowed = set(self.words_of(sentence)) | set(self.words_of(text.splitlines()[0]))
        for line in text.splitlines()[1:]:
            if line.startswith("Часть ") or line == "ранние части опущены":
                continue
            self.assertTrue(set(self.words_of(line.lstrip("- "))) <= allowed, line[:80])

    @staticmethod
    def count(text):
        """Слова разделов в блоке изложения (без заголовка блока и строк «Часть N:»)."""
        lines = text.splitlines()[1:]
        return sum(len(line.lstrip("- ").split()) for line in lines
                   if line.startswith("- "))


class ConsentPartsTests(unittest.TestCase):
    def test_parts_phrase_is_the_same_in_every_copy(self):
        def flat(path):
            return " ".join((ROOT / path).read_text(encoding="utf-8").split())
        server = flat("server.py")
        instructions = server.split("INSTRUCTIONS = ", 1)[1].split('"""', 2)[1]
        start = server.split("def start_workflow", 1)[1].split('"""', 2)[1]
        html = flat("workflow.html").split('id="cloud-note">', 1)[1].split("</p>", 1)[0]
        for name, text in (("INSTRUCTIONS", instructions), ("start_workflow", start), ("workflow.html", html),
                           ("README.md", flat("README.md")),
                           ("П4", flat("docs/2026-11-17-p4-opencode-openai-handoff.md"))):
            self.assertIn(CONSENT_PARTS, text, name)


class LongWorkflowTests(unittest.TestCase):
    """Через базу. Предел части уменьшен до 1 000 слов, чтобы запись в 2 500 слов шла тремя частями."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="long-parts-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.dbpath = self.root / "state.sqlite3"
        self.store = storage.initialize(self.dbpath, initial_display_name="Тест")
        self.addCleanup(self.store.close)
        self.sent, self.failures = [], {}
        patcher = mock.patch("llm_provider.MAX_WORDS", 1000)
        patcher.start()
        self.addCleanup(patcher.stop)

    def prepare_request(self, turn_words=(100,) * 25, gaps=None):
        """Реплики по turn_words слов (слово — секунда), перед репликой с номером из gaps — пауза."""
        policy = {"require_conspect": True, "allow_cloud_model": True}
        request = workflow.create(self.store, self.source, policy)
        workflow.claim_start(self.store, request)
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        recording_id = str(uuid.uuid4())
        segments, clock, index = [], 0, 0
        for number, count in enumerate(turn_words):
            clock += (gaps or {}).get(number, 0)
            words = [{"w": f"слово{index + i}" + ("." if (index + i) % 10 == 9 else ""), "s": clock + i,
                      "e": clock + i + 1} for i in range(count)]
            segments.append({"start": clock, "end": clock + count, "speaker": f"Спикер {1 + number % 2}",
                             "words": words})
            clock += count
            index += count
        data = {"file": str(self.source), "duration": "01:00:00", "speakers": ["Спикер 1", "Спикер 2"],
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

    def client(self, before_answer=None, answer=None):
        test = self

        class Client:
            def __enter__(self): return self
            def __exit__(self, *args): return False

            def generate(self, task, **kwargs):
                test.sent.append((task, kwargs))
                if before_answer:
                    before_answer(len(test.sent))
                if len(test.sent) in test.failures:
                    raise llm_provider.ProviderError("offline", test.failures.pop(len(test.sent)))
                result = (answer or part_answer)(task)
                return result, json.dumps(result, ensure_ascii=False)
        return Client

    def execute(self, request, before_answer=None, answer=None, **kwargs):
        with mock.patch("llm_provider.OpenCodeTextClient", self.client(before_answer, answer)):
            return workflow.execute_model(self.store, request, **kwargs)

    def core_call(self, function, *args, **kwargs):
        with mock.patch("recordings.open_store", side_effect=lambda: storage.open_store(self.dbpath)):
            return function(*args, **kwargs)

    def test_golden_9990_words_go_in_one_unchanged_request(self):
        with mock.patch("llm_provider.MAX_WORDS", 10_000):
            request = self.prepare_request(turn_words=(5000, 4990))
            result = self.execute(request)
        self.assertEqual((result["state"], result["part_count"], len(self.sent)), ("completed", 1, 1))
        task = self.sent[0][0]
        self.assertNotIn("parts", task)
        self.assertNotIn("part", task)
        self.assertNotIn("summary", self.sent[0][1])
        self.assertEqual(hashlib.sha256(prompt_of(task).encode()).hexdigest(), GOLDEN_9990_PROMPT_SHA)
        self.assertEqual(workflow.get(self.store, request)["model_task"], task)

    def test_long_recording_goes_in_parts_with_the_summary_of_previous(self):
        request = self.prepare_request(gaps={9: 30, 17: 30})          # паузы перед репликами 10 и 18
        result = self.execute(request)
        self.assertEqual((result["state"], result["part_count"], len(self.sent)), ("completed", 3, 3))
        saved = workflow.get(self.store, request)["model_task"]
        self.assertEqual(saved["parts"], [[1, 9], [10, 17], [18, 25]])
        self.assertEqual([task["part"] for task, _ in self.sent], [1, 2, 3])
        self.assertNotIn("summary", self.sent[0][1])
        self.assertEqual(self.sent[1][1]["summary"], [["Раздел части. Пункты 1–9."]])
        self.assertEqual(self.sent[2][1]["summary"], [["Раздел части. Пункты 1–9."], ["Раздел части. Пункты 10–17."]])
        model = workflow.get(self.store, request)["model_result"]
        evidence = [ident for section in model["sections"] for ident in section["evidence"]]
        self.assertEqual(evidence, [word["word_id"] for word in saved["words"]])
        self.assertEqual(len(model["sections"]), 3)

    def test_failed_part_two_continues_from_it(self):
        request = self.prepare_request()
        self.failures[2] = "нет сети"
        with self.assertRaises(llm_provider.ProviderError) as error:
            self.execute(request)
        self.assertIn("сохранено блоков 1 из 3", str(error.exception))
        self.assertEqual(workflow.get(self.store, request)["state"], "failed")
        self.sent.clear()
        self.assertEqual(self.execute(request, retry_failed=True)["state"], "completed")
        self.assertEqual([task["part"] for task, _ in self.sent], [2, 3])
        self.assertEqual(self.sent[0][1]["summary"], [["Раздел части. Пункты 1–8."]])

    def test_revision_change_between_parts_needs_restart(self):
        request = self.prepare_request()

        def change(call):
            if call != 2:
                return
            recording = workflow.get(self.store, request)["recording_id"]
            current = self.store.get_current(recording)
            review = json.loads(json.dumps(current["review"]["payload"]))
            word = current["transcript"]["payload"]["segments"][0]["words"][0]["word_id"]
            review["review"] = review.get("review", []) + [
                {"id": str(uuid.uuid4()), "status": "open", "word_ids": [word], "t1": 0, "t2": 1}]
            binding = recordings.document_binding(self.store, recording)
            self.store.commit_change_with_document(
                recording, {"source_revision": current["transcript"]["revision"],
                            "review_revision": current["review"]["revision"]},
                str(uuid.uuid4()), recordings.operation_context(self.store, "human"),
                review=review, root_id=binding["root_id"], generated_at="2026-01-01T00:00:01Z")
        with self.assertRaises(llm_provider.ProviderError) as error:
            self.execute(request, before_answer=change)
        self.assertEqual(error.exception.code, "revision_changed")
        with self.assertRaisesRegex(storage.Conflict, "restart"):
            self.execute(request, retry_failed=True)
        self.sent.clear()
        self.assertEqual(self.execute(request, restart=True)["state"], "completed")
        self.assertEqual([task["part"] for task, _ in self.sent], [1, 2, 3])

    def test_summary_is_not_a_source_of_quotes_or_evidence(self):
        request = self.prepare_request()

        def answer(task):
            result = part_answer(task, text="Особая фраза изложения.")
            if task["part"] == 2 and not self.sent[-1][1].get("repair_error"):
                result["sections"][0]["text"] = "Сказано: «Особая фраза изложения»."
            return result
        self.assertEqual(self.execute(request, answer=answer)["state"], "completed")
        self.assertIn("Особая фраза изложения", json.dumps(self.sent[1][1]["summary"], ensure_ascii=False))
        self.assertIn("repair_error", self.sent[2][1])                # цитата из изложения отклонена
        saved = workflow.get(self.store, request)
        task = saved["model_task"]
        first, final = task["parts"][1]
        part_words = {word["word_id"] for point in task["points"][first - 1:final]
                      for word in task["words"][point["start"]:point["end"]]}
        second = saved["model_result"]["sections"][1]
        self.assertTrue(set(second["evidence"]) <= part_words)
        self.assertNotIn("«", second["text"])

    def test_more_than_six_parts_is_refused_before_the_cloud(self):
        request = self.prepare_request(turn_words=(100,) * 61)        # 6 100 слов при пределе 1 000
        with self.assertRaises(llm_provider.ProviderError) as error:
            self.execute(request)
        self.assertEqual((error.exception.code, len(self.sent)), ("limit", 0))
        self.assertIn("6000 слов", str(error.exception))
        self.assertEqual(workflow.get(self.store, request)["state"], "processed")

    def test_old_waiting_task_without_parts_keeps_the_old_limit(self):
        request = self.prepare_request()
        with mock.patch.object(workflow, "_plan_parts", return_value=None):
            legacy = workflow.prepare_model_task(self.store, request)    # сохранено до 7.4в
        self.assertNotIn("parts", legacy)
        self.assertEqual(workflow.prepare_model_task(self.store, request), legacy)   # содержание то же
        with self.assertRaises(llm_provider.ProviderError) as error:
            self.execute(request)
        self.assertEqual((error.exception.code, len(self.sent)), ("limit", 0))
        self.assertIn("(1000 слов)", str(error.exception))
        self.assertEqual(self.execute(request, restart=True)["part_count"], 3)

    def test_agent_gets_parts_with_their_points(self):
        request = self.prepare_request()
        task = workflow.prepare_model_task(self.store, request)
        for index in range(3):
            part = self.core_call(core.workflow_model_part, request, index)
            self.assertEqual(part["part_count"], 3)
            self.assertNotIn("words", part["task"])
            first, final = task["parts"][index]
            self.assertEqual([p[0] for p in part["task"]["points"]], list(range(first, final + 1)))
            done = self.core_call(core.workflow_model_part_result, request, index,
                                  part_answer(model_tasks.part_task(task, index + 1)))
        self.assertEqual(done["state"], "completed")


if __name__ == "__main__":
    unittest.main()
