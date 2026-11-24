"""РАР-6.1, PR 6а: пауза не владеет соседним словом.

Место-пропуск («missed» в reasons) владеет только словами, которые НАЧАТЫ внутри него, по полуоткрытому
интервалу [t1, t2) в целых сотых. Одна функция на правило: transcript.in_pause / pause_words / place_words /
places_overlap / word_in_place. Все данные синтетические.
"""
import atexit
import copy
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="rt-pause-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import transcript  # noqa: E402
import workflow  # noqa: E402


def word(wid, text, s, e=None):
    return {"word_id": wid, "w": text, "s": s, "e": s + 0.2 if e is None else e}


def gap(t1, t2, **extra):
    item = {"id": "r1", "status": "open", "t": t1, "t1": t1, "t2": t2, "speaker": "Анна",
            "reasons": ["missed"], "text": ""}
    item.update(extra)
    return item


def place(t1, t2=None, **extra):
    item = {"id": "r2", "status": "open", "t": t1, "t1": t1, "t2": t1 if t2 is None else t2, "speaker": "Анна",
            "reasons": ["asr"], "text": "Ну"}
    item.update(extra)
    return item


class CentsTests(unittest.TestCase):
    def test_cents_values(self):
        for t, want in ((1.994, 199), (1.996, 200), (2.0, 200), (2.01, 201), (0.005, 1), (0, 0)):
            self.assertEqual(transcript._cents(t), want, t)

    def test_cents_equals_cents_of_package_rounding_on_every_half_cent(self):
        """Пакет хранит round(t, 2); сотые сырого времени обязаны совпасть с сотыми округлённого.

        Простое round(t * 100) расходится с round(t, 2) в x,xx5 (617 799 точек сетки 0–4 ч шагом 1 мс):
        проверяются все x,xx5 до 4 часов и миллисекунды рядом с ними.
        """
        bad = []
        for n in range(0, 1_440_000):
            base = n / 100
            for t in (base + 0.005, base + 0.004, base + 0.006):
                if transcript._cents(t) != transcript._cents(round(t, 2)):
                    bad.append(t)
            if len(bad) > 3:
                break
        self.assertEqual(bad, [])

    def test_cents_monotonic(self):
        previous = -1
        for ms in range(0, 20000):
            now = transcript._cents(ms / 1000)
            self.assertGreaterEqual(now, previous)
            previous = now


class InPauseTests(unittest.TestCase):
    def test_half_open_interval(self):
        self.assertTrue(transcript.in_pause(0.996, 1.0, 2.0))
        self.assertTrue(transcript.in_pause(1.0, 1.0, 2.0))
        self.assertTrue(transcript.in_pause(1.994, 1.0, 2.0))
        self.assertFalse(transcript.in_pause(1.996, 1.0, 2.0))
        self.assertFalse(transcript.in_pause(2.0, 1.0, 2.0))
        self.assertFalse(transcript.in_pause(2.01, 1.0, 2.0))
        self.assertFalse(transcript.in_pause(0.994, 1.0, 2.0))

    def test_is_gap_place(self):
        self.assertTrue(transcript.is_gap_place({"reasons": ["missed"]}))
        self.assertTrue(transcript.is_gap_place({"reasons": ["asr", "missed"]}))
        self.assertFalse(transcript.is_gap_place({"reasons": ["asr"]}))
        self.assertFalse(transcript.is_gap_place({"reasons": "missed"}))
        self.assertFalse(transcript.is_gap_place({}))

    def test_pause_words_in_transcript_order_and_skip_untimed(self):
        data = {"segments": [
            {"speaker": "Анна", "words": [word("w1", "а", 0.5), word("w2", "б", 1.0), {"w": "в"}]},
            {"speaker": "Борис", "words": [word("w3", "г", 1.5), word("w4", "д", 2.0)]}]}
        got = transcript.pause_words(data, 1.0, 2.0)
        self.assertEqual([(si, k, w["word_id"]) for si, k, w in got], [(0, 1, "w2"), (1, 0, "w3")])
        self.assertEqual(transcript.pause_words(data, 3.0, 4.0), [])


class PlaceWordsTests(unittest.TestCase):
    def setUp(self):
        self.data = {"segments": [{"speaker": "Анна", "words": [word("w1", "начало", 0.5, 1.05),
                                                                  word("w2", "Ну", 2.0, 2.2)]}]}

    def test_gap_ignores_word_ids_of_package(self):
        item = gap(1.0, 2.0, word_ids=["w2"])
        self.assertEqual(transcript.place_words(self.data, item), ([], True))

    def test_empty_gap_neighbors_only_on_request(self):
        item = gap(1.0, 2.0)
        self.assertEqual(transcript.place_words(self.data, item), ([], True))
        self.assertEqual(transcript.place_words(self.data, item, pause_neighbors=True), (["w1", "w2"], True))

    def test_gap_at_end_of_recording_has_only_word_before(self):
        self.assertEqual(transcript.place_words(self.data, gap(3.0, 4.0), pause_neighbors=True), (["w2"], True))

    def test_gap_at_start_of_recording_has_only_word_after(self):
        data = {"segments": [{"speaker": "Анна", "words": [word("w1", "Ну", 2.0)]}]}
        self.assertEqual(transcript.place_words(data, gap(0.0, 1.0), pause_neighbors=True), (["w1"], True))

    def test_gap_with_words_inside_returns_them_even_with_neighbors_flag(self):
        self.assertEqual(transcript.place_words(self.data, gap(0.4, 0.9), pause_neighbors=True), (["w1"], False))

    def test_equal_times_before_is_later_after_is_earlier(self):
        data = {"segments": [{"speaker": "Анна", "words": [word("a", "х", 0.5), word("b", "у", 0.5),
                                                            word("c", "з", 3.0), word("d", "и", 3.0)]}]}
        self.assertEqual(transcript.place_words(data, gap(1.0, 2.0), pause_neighbors=True)[0], ["b", "c"])

    def test_word_place_with_word_ids(self):
        self.assertEqual(transcript.place_words(self.data, place(2.0, word_ids=["w2"])), (["w2"], False))
        self.assertEqual(transcript.place_words(self.data, place(0.5, 2.0, word_ids=["w2", "w1"])),
                         (["w1", "w2"], False))

    def test_word_place_with_bad_word_ids(self):
        for ids in (["nope"], ["w1", "w1"], [1], ["w1", "nope"]):
            self.assertIsNone(transcript.place_words(self.data, place(2.0, word_ids=ids)), ids)

    def test_word_place_by_time_including_overlap(self):
        self.assertEqual(transcript.place_words(self.data, place(2.0)), (["w2"], False))
        self.assertEqual(transcript.place_words(self.data, place(2.1, 2.5)), (["w2"], False))
        self.assertEqual(transcript.place_words(self.data, place(2.205, 2.5)), ([], True))
        self.assertEqual(transcript.place_words(self.data, place(2.205, 2.5), pause_neighbors=True),
                         (["w2"], True))

    def test_bad_time_is_none(self):
        for t1, t2 in ((None, 2.0), (2.0, None), (float("nan"), 2.0), (2.0, float("inf")), ("1", 2.0),
                       (True, 2.0), (3.0, 2.0)):
            for reasons in (["missed"], ["asr"]):
                item = {"id": "r1", "t1": t1, "t2": t2, "reasons": reasons}
                self.assertIsNone(transcript.place_words(self.data, item), (t1, t2, reasons))

    def test_time_falls_back_to_t(self):
        self.assertEqual(transcript.place_words(self.data, {"id": "r1", "t": 2.0, "reasons": ["asr"]}),
                         (["w2"], False))

    def test_word_without_word_id_is_none(self):
        data = {"segments": [{"speaker": "Анна", "words": [{"w": "Ну", "s": 2.0, "e": 2.2}]}]}
        self.assertIsNone(transcript.place_words(data, place(2.0)))
        self.assertIsNone(transcript.place_words(data, gap(1.0, 3.0)))
        self.assertIsNone(transcript.place_words(data, gap(3.0, 4.0), pause_neighbors=True))

    def test_record_without_timed_words_is_none(self):
        data = {"segments": [{"speaker": "Анна", "words": [{"w": "Ну", "word_id": "w1"}]}]}
        self.assertIsNone(transcript.place_words(data, gap(1.0, 3.0)))
        self.assertIsNone(transcript.place_words(data, place(2.0)))

    def test_pure_function_does_not_change_data(self):
        before = copy.deepcopy(self.data)
        transcript.place_words(self.data, gap(1.0, 2.0), pause_neighbors=True)
        self.assertEqual(self.data, before)


class WordInPlaceTests(unittest.TestCase):
    def test_same_cases_as_before_the_move(self):
        w = transcript.word_in_place
        self.assertTrue(w(2.0, 2.2, 2.0, 2.0))
        self.assertTrue(w(1.995, 2.1, 2.0, 2.0))
        self.assertFalse(w(1.5, 1.9, 2.0, 2.0))          # касается лишь до места
        self.assertTrue(w(1.9, 2.3, 2.0, 2.1))           # перекрытие
        self.assertFalse(w(1.5, 2.0, 2.0, 2.5))          # конец слова ровно в t1
        self.assertTrue(w(2.505, 2.7, 2.0, 2.5))         # допуск справа

    def test_workflow_uses_the_same_function(self):
        self.assertIs(workflow._word_in_place, transcript.word_in_place)
        self.assertEqual(workflow.PLACE_TIME_TOLERANCE, transcript.WORD_PLACE_TOLERANCE)


class PlacesOverlapTests(unittest.TestCase):
    def test_pause_and_word_at_its_end_do_not_overlap(self):
        self.assertFalse(transcript.places_overlap(gap(2.0, 3.0), place(3.0)))
        self.assertFalse(transcript.places_overlap(place(3.0), gap(2.0, 3.0)))
        self.assertFalse(transcript.places_overlap(gap(2.0, 3.0), place(2.995)))

    def test_pause_and_word_inside_overlap(self):
        self.assertTrue(transcript.places_overlap(gap(2.0, 3.0), place(2.994)))
        self.assertTrue(transcript.places_overlap(gap(2.0, 3.0), place(2.5)))
        self.assertTrue(transcript.places_overlap(gap(2.0, 3.0), place(1.5, 2.5)))
        self.assertTrue(transcript.places_overlap(gap(2.0, 3.0), place(2.0)))
        self.assertFalse(transcript.places_overlap(gap(2.0, 3.0), place(1.99)))

    def test_adjacent_pauses_do_not_overlap(self):
        self.assertFalse(transcript.places_overlap(gap(1.0, 2.0), gap(2.0, 3.0)))
        self.assertTrue(transcript.places_overlap(gap(1.0, 2.0), gap(1.5, 3.0)))

    def test_two_word_places_keep_the_old_formula(self):
        self.assertTrue(transcript.places_overlap(place(3.0), place(3.005)))
        self.assertTrue(transcript.places_overlap(place(3.0), place(3.01)))
        self.assertFalse(transcript.places_overlap(place(3.0), place(3.02)))

    def test_bad_time_does_not_raise_and_does_not_overlap(self):
        bad = {"id": "r9", "t1": None, "t2": None, "reasons": ["missed"]}
        self.assertFalse(transcript.places_overlap(bad, gap(1.0, 2.0)))
        self.assertFalse(transcript.places_overlap(place(1.0), bad))


if __name__ == "__main__":
    unittest.main()
