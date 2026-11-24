"""Правило «слово тронул человек» — одна функция transcript.human_touched.

Человек тронул слово, если он автор слова (by), его текста или его говорящего (provenance).
Агент такое слово не меняет ни по `apply_word_edits`, ни по `apply_review`. Синтетика, оба режима хранения.
"""
import unittest

import test_human_confirmation as hc  # noqa: E402  (тот же sandbox и те же карточки)

import transcript  # noqa: E402

BACKENDS = hc.BACKENDS


class HumanTouchedFunction(unittest.TestCase):
    def test_cases(self):
        cases = [
            ({"w": "а", "by": "human"}, True),
            ({"w": "а", "provenance": {"text": {"by": "human"}}}, True),
            ({"w": "а", "provenance": {"speaker": {"by": "human"}}}, True),
            ({"w": "а", "by": "claude"}, False),
            ({"w": "а", "by": "model", "provenance": {"text": {"by": "claude"}}}, False),
            ({"w": "а"}, False),
            ({"w": "а", "provenance": {}}, False),
            ({"w": "а", "provenance": None}, False),
            ({"w": "а", "provenance": "human"}, False),
            ({"w": "а", "provenance": {"speaker": "human"}}, False),
        ]
        for word, expected in cases:
            with self.subTest(word=word):
                self.assertIs(transcript.human_touched(word), expected)


def speaker_only(jp, md):
    """Слово w1: человек выбрал только говорящего — у слова нет `by`, есть provenance.speaker.by."""
    data = transcript.load(jp)
    word = next(w for s in data["segments"] for w in s["words"] if w["word_id"] == "w1")
    word.pop("by", None)
    word["provenance"] = {"speaker": {"by": "human"}}
    transcript.save(data, jp, md)


class AgentKeepsOffWordTouchedBySpeaker(unittest.TestCase):
    def test_apply_word_edits(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), hc.files(backend, speaker_only) as (jp, md, _):
                _, rep = transcript.apply_word_edits(
                    jp, md, [{"op": "replace", "t": 0.2, "w": "тест", "word_id": "w1", "text": "тесть",
                              "card": "a1"}], by="claude")
                self.assertIn("подтверждено человеком", rep["failed"].get("a1", ""))
                self.assertEqual(hc.word(jp, "w1")["w"], "тест")

    def test_apply_review(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), hc.files(backend, speaker_only) as (jp, md, _):
                _, rep = transcript.apply_review(jp, md, [{"id": "r1", "text": "тесть"}], by="claude")
                self.assertEqual([c.get("kept_edits") for c in rep.get("conflicts") or []], [["human"]])
                self.assertEqual(hc.word(jp, "w1")["w"], "тест")


if __name__ == "__main__":
    unittest.main()
