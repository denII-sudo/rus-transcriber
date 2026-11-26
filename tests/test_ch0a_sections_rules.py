"""Ч-0а (РАР-7): правила разделов и отсылок also на ответах облачной пробы Ч-0 (26.11.2026).

Проба: модель ответила, но все четыре ответа проверка отвергла — отсылки also длиннее 20 пунктов
(модель ссылалась на целые разделы) и разделы больше 1500 слов стенограммы, а запрос исправления
называл только первое нарушение. Теперь: запрос однозначен и даёт число слов к концу каждого пункта;
проверка называет все нарушения разделов; also ответ не отвергает — урезается или отбрасывается,
что сделано — в also_adjusted. Ответы модели — tests/ch0_samples.py. Синтетика, без облака.
"""
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import ch0_samples as samples
import llm_provider
import model_tasks
import recordings
import storage
import workflow


def sized_task(point_words, parts=None):
    """Задание версии 4 с пунктами заданных размеров: пункт — реплика (не длиннее 120 слов)."""
    words, turns = [], []
    for number, size in enumerate(point_words, 1):
        turns.append({"turn": number, "speaker": f"Участник {1 + number % 5}"})
        words += [{"word_id": f"w{len(words) + i}", "text": f"слово{len(words) + i}", "turn": number,
                   "uncertain": False} for i in range(size)]
    task = model_tasks.build("synthetic-recording-id", 1, words, "протокол", 1, turns=turns, unlocated_doubts=0,
                             points=True, parts=parts)
    assert [point["end"] - point["start"] for point in task["points"]] == list(point_words)
    return task


def oversized(task, response):
    """Разделы ответа больше MAX_SECTION_WORDS слов стенограммы — независимый подсчёт по размерам пунктов."""
    points = task["points"]
    return [(number, section["from"], section["to"])
            for number, section in enumerate(response["sections"], 1)
            if section["from"] != section["to"]
            and points[section["to"] - 1]["end"] - points[section["from"] - 1]["start"] > model_tasks.MAX_SECTION_WORDS]


def long_refs(response):
    return sum(1 for section in response["sections"] for ref in section.get("also", [])
               if ref[1] - ref[0] + 1 > model_tasks.MAX_ALSO_POINTS)


class ProbeAnswersTests(unittest.TestCase):
    """Каждый из четырёх ответов либо принимается, либо ошибка называет все нарушения разделов."""

    def cases(self):
        short = sized_task(samples.SHORT_POINT_WORDS)
        long_part = model_tasks.part_task(sized_task(samples.LONG_POINT_WORDS, samples.LONG_PARTS), 1)
        return [("короткий, ответ", short, samples.SHORT_RESPONSE_1),
                ("короткий, исправление", short, samples.SHORT_RESPONSE_2),
                ("длинный, часть 1, ответ", long_part, samples.LONG_RESPONSE_1),
                ("длинный, часть 1, исправление", long_part, samples.LONG_RESPONSE_2)]

    def test_each_answer_is_accepted_or_every_violation_is_named(self):
        outcomes = {}
        for name, task, raw in self.cases():
            response = json.loads(raw)
            big = oversized(task, response)
            with self.subTest(name):
                if not big:
                    result = model_tasks.validate(task, response, current_revision=1)
                    self.assertEqual(len(result["sections"]), len(response["sections"]))
                    notes = result.get("also_adjusted", [])
                    self.assertEqual(sum(1 for note in notes if note["reason"] == "long"), long_refs(response))
                    for section in result["sections"]:
                        self.assertTrue(all(len(ids) <= sum(samples.SHORT_POINT_WORDS) for ids in section["also"]))
                    outcomes[name] = ("принят", len(notes))
                    continue
                with self.assertRaises(ValueError) as error:
                    model_tasks.validate(task, response, current_revision=1)
                message = str(error.exception)
                for number, first, final in big:
                    self.assertIn(f"раздел {number} (пункты {first}–{final}) охватывает", message)
                if len(big) > 1:
                    self.assertTrue(message.startswith(f"нарушений в ответе: {len(big)}, исправьте все"), message)
                self.assertNotIn("also", message)             # отсылки больше не отвергают ответ
                self.assertLessEqual(len(message), llm_provider.MAX_REPAIR_ERROR_CHARS)
                outcomes[name] = ("отвергнут", len(big))
        # Итог на образцах Ч-0 (зафиксирован, чтобы изменение правил было видно в тесте).
        self.assertEqual(outcomes, {"короткий, ответ": ("отвергнут", 1), "короткий, исправление": ("принят", 12),
                                    "длинный, часть 1, ответ": ("отвергнут", 3),
                                    "длинный, часть 1, исправление": ("отвергнут", 2)})

    def test_repair_request_carries_the_whole_error(self):
        task = model_tasks.part_task(sized_task(samples.LONG_POINT_WORDS, samples.LONG_PARTS), 1)
        with self.assertRaises(ValueError) as error:
            model_tasks.validate(task, json.loads(samples.LONG_RESPONSE_1), current_revision=1)
        client = llm_provider.OpenCodeTextClient()
        with mock.patch.object(client, "_generate_text", return_value={"data": {"text": "{}"}}) as call:
            client.generate(task, repair_error=str(error.exception), invalid_output=samples.LONG_RESPONSE_1)
        prompt = call.call_args.args[0]
        self.assertIn(str(error.exception), prompt)
        self.assertIn("нарушений в ответе: 3", prompt)


class RequestRulesTests(unittest.TestCase):
    def prompt(self, task, **kwargs):
        client = llm_provider.OpenCodeTextClient()
        with mock.patch.object(client, "_generate_text", return_value={"data": {"text": "{}"}}) as call:
            client.generate(task, **kwargs)
        return call.call_args.args[0]

    def test_rules_are_unambiguous(self):
        instructions = self.prompt(sized_task([10, 20, 30])).split("Задание:\n", 1)[0]
        for phrase in ("не больше 1500 слов стенограммы", "сумма слов его пунктов, а не длина твоего текста",
                       "Отсылка — короткий фрагмент, не длиннее 20 пунктов, а не весь раздел"):
            self.assertIn(phrase, instructions)

    def test_word_counts_to_the_end_of_each_point_are_given_outside_the_view(self):
        task = sized_task([10, 20, 30])
        instructions, sent = self.prompt(task).split("Задание:\n", 1)
        self.assertEqual(json.loads(sent), model_tasks.model_view(task))       # вид задания прежний
        self.assertIn("считая с пункта 1 (пункт:слов)", instructions)
        self.assertTrue(instructions.endswith(": 1:10 2:30 3:60\n"), instructions[-80:])

    def test_part_counts_start_at_the_part(self):
        task = sized_task(samples.LONG_POINT_WORDS, samples.LONG_PARTS)
        first, final = samples.LONG_PARTS[1]
        words = sum(samples.LONG_POINT_WORDS[first - 1:final])
        instructions = self.prompt(model_tasks.part_task(task, 2), summary=[["Раздел части 1."]]).split("Задание:\n")[0]
        self.assertIn(f"считая с пункта {first} (пункт:слов)", instructions)
        self.assertIn(f" {first}:{samples.LONG_POINT_WORDS[first - 1]} ", instructions)
        self.assertTrue(instructions.endswith(f" {final}:{words}\n"))

    def test_request_size_grows_modestly(self):
        task = sized_task(samples.SHORT_POINT_WORDS)
        line = llm_provider._word_counts(task)
        self.assertLess(len(line), 2500)                 # 261 пункт: около 2 КБ к запросу в 110 КБ
        self.assertTrue(line.rstrip().endswith(f"261:{sum(samples.SHORT_POINT_WORDS)}"))


class PartsReferencesTests(unittest.TestCase):
    """Отсылка из части 2 в часть 1 отбрасывается и в проверке части, и в итоговой сборке по всему заданию."""

    def test_reference_into_another_part_is_dropped_in_both_checks(self):
        task = sized_task([10] * 6, parts=[[1, 3], [4, 6]])
        part_answer = {"source_revision": 1, "sections": [{"from": 4, "to": 6, "text": "Часть 2.",
                                                           "also": [[1, 2]]}]}
        in_part = model_tasks.validate(model_tasks.part_task(task, 2), part_answer, current_revision=1)
        whole = {"source_revision": 1, "sections": [{"from": 1, "to": 3, "text": "Часть 1."},
                                                    *part_answer["sections"]]}
        assembled = model_tasks.validate(task, whole, current_revision=1)
        self.assertEqual(in_part["sections"][0]["also"], [])
        self.assertEqual(assembled["sections"][1]["also"], [])
        self.assertEqual([(n["reason"], n["ref"]) for n in assembled["also_adjusted"]], [("outside", [1, 2])])

    def test_reference_within_the_part_is_kept(self):
        task = sized_task([10] * 6, parts=[[1, 3], [4, 6]])
        whole = {"source_revision": 1, "sections": [{"from": 1, "to": 3, "text": "Часть 1."},
                                                    {"from": 4, "to": 5, "text": "А.", "also": [[6, 6]]},
                                                    {"from": 6, "to": 6, "text": "Б."}]}
        result = model_tasks.validate(task, whole, current_revision=1)
        self.assertEqual(len(result["sections"][1]["also"][0]), 10)
        self.assertNotIn("also_adjusted", result)


class ExecuteModelTests(unittest.TestCase):
    """Через базу: исправленный ответ пробы (12 длинных отсылок) теперь собирает свёртку, отметка — в итоге."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ch0a-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.store = storage.initialize(self.root / "state.sqlite3", initial_display_name="Тест")
        self.addCleanup(self.store.close)

    def prepare(self, point_words):
        request = workflow.create(self.store, self.source, {"require_conspect": True, "allow_cloud_model": True})
        workflow.claim_start(self.store, request)
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        recording_id, segments, clock = str(uuid.uuid4()), [], 0
        for number, count in enumerate(point_words):
            words = [{"w": f"слово{clock + i}", "s": clock + i, "e": clock + i + 1} for i in range(count)]
            segments.append({"start": clock, "end": clock + count, "speaker": f"Спикер {1 + number % 5}",
                             "words": words})
            clock += count + 1
        data = {"file": str(self.source), "duration": "01:00:00",
                "speakers": sorted({segment["speaker"] for segment in segments}), "quality": {}, "review": [],
                "review_done": True, "segments": segments}
        recordings.commit_worker_result(
            self.store, data, self.source, root_id, recordings.operation_context(self.store, "model"),
            recording_id=recording_id, operation_id=str(uuid.uuid4()), generated_at="2026-01-01T00:00:00Z",
            document_paths=paths)
        recordings.materialize_recording(self.store, recording_id)
        with self.store._transaction():
            self.store.db.execute("UPDATE workflow_requests SET recording_id=?, state='processed' WHERE request_id=?",
                                  (recording_id, request))
        return request

    def test_probe_repair_answer_completes_with_adjustments_reported(self):
        request = self.prepare(samples.SHORT_POINT_WORDS)
        sent = []

        class Client:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def generate(self, task, **kwargs):
                sent.append(task)
                answer = json.loads(samples.SHORT_RESPONSE_2)
                answer["source_revision"] = task["source_revision"]
                return answer, json.dumps(answer, ensure_ascii=False)

        with mock.patch("llm_provider.OpenCodeTextClient", Client):
            result = workflow.execute_model(self.store, request)
        self.assertEqual((result["state"], len(sent)), ("completed", 1))
        self.assertEqual(len(result["also_adjusted"]), 12)
        self.assertTrue(all(note["action"] == "truncated" for note in result["also_adjusted"]))
        stored = workflow.get(self.store, request)["model_result"]
        self.assertEqual(stored["also_adjusted"], result["also_adjusted"])
        self.assertTrue(list(self.root.rglob("*.conspect.md")))


if __name__ == "__main__":
    unittest.main()
