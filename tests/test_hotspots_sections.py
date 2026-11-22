"""Сопоставление горячих мест и разделов на искусственных данных."""
import copy
import unittest

import hotspots


def section(start, end, name):
    return {"t1": start, "t2": end, "name": name}


def place(start, end, score, kind):
    return {"t1": start, "t2": end, "score": score, "kinds": [kind]}


class MatchSectionsTests(unittest.TestCase):
    def test_empty_and_outside(self):
        h = place(20, 22, .9, "a")
        self.assertEqual(hotspots.match_sections([], [], top_n=3), [])
        self.assertEqual(hotspots.match_sections([], [h], top_n=3), [])
        self.assertEqual(hotspots.match_sections([section(0, 10, "a")], [], top_n=3), [[]])
        self.assertEqual(hotspots.match_sections([section(0, 10, "a")], [h], top_n=3), [[]])

    def test_boundary_and_ties_choose_earlier_by_time(self):
        sections = [section(10, 20, "later"), section(0, 10, "earlier")]
        left = place(8, 11, .8, "left")
        right = place(9, 12, .8, "right")
        tie = place(9, 11, .8, "tie")
        touching = place(20, 21, .8, "touch")
        self.assertEqual(hotspots.match_sections(sections, [left, right, tie, touching], top_n=4),
                         [[right], [left, tie]])
        self.assertEqual(hotspots.match_sections([section(0, 10, "a"), section(0, 10, "b")],
                                                  [left], top_n=1), [[left], []])

    def test_top_n_global_stable_and_original_order(self):
        sections = [section(0, 10, "a"), section(10, 20, "b")]
        first = place(1, 2, .7, "first")
        second = place(3, 4, .9, "second")
        third = place(11, 12, .7, "third")
        outside = place(30, 31, 1.0, "outside")
        places = [first, second, third, outside]
        original = copy.deepcopy((sections, places))
        self.assertEqual(hotspots.match_sections(sections, places, top_n=3), [[first, second], []])
        self.assertEqual(hotspots.match_sections(sections, places, top_n=10), [[first, second], [third]])
        self.assertEqual(hotspots.match_sections(sections, places, top_n=0), [[], []])
        self.assertEqual((sections, places), original)

    def test_threshold_inclusive_and_gaps(self):
        sections = [section(0, 5, "a"), section(5, 5, "zero"),
                    section(None, None, "unknown"), section(10, 15, "b")]
        a = place(2, 3, .5, "a")
        gap = place(6, 9, .8, "gap")
        b = place(12, 13, .9, "b")
        low = place(1, 2, .4, "low")
        self.assertEqual(hotspots.match_sections(sections, [a, gap, b, low], min_score=.5),
                         [[a], [], [], [b]])

    def test_fractional_tie_and_distinct_overlap(self):
        sections = [section(.2, .4, "later"), section(0, .2, "earlier")]
        tied = place(.1, .3, .8, "tie")
        later = place(.1, .300001, .8, "later")
        self.assertEqual(hotspots.match_sections(sections, [tied, later], top_n=2),
                         [[later], [tied]])
        self.assertEqual(hotspots.match_sections(list(reversed(sections)), [tied], min_score=0),
                         [[tied], []])

    def test_equal_strength_uses_input_order_at_cutoff(self):
        sections = [section(0, 10, "a")]
        first = place(4, 5, .7, "first")
        second = place(1, 2, .7, "second")
        self.assertEqual(hotspots.match_sections(sections, [first, second], top_n=1), [[first]])
        self.assertEqual(hotspots.match_sections(sections, [second, first], top_n=1), [[second]])

    def test_every_place_is_validated_before_selection(self):
        good = place(1, 2, .8, "good")
        invalid = [
            None, {}, place("1", 2, .5, "bad"), place(1, None, .5, "bad"),
            place(float("nan"), 2, .5, "bad"), place(1, float("inf"), .5, "bad"),
            place(-1, 2, .5, "bad"), place(2, 2, .5, "bad"),
            place(3, 2, .5, "bad"), place(1, 2, "0.5", "bad"),
            place(1, 2, float("nan"), "bad"), place(1, 2, float("inf"), "bad"),
            place(1, 2, -.1, "bad"), place(1, 2, 1.1, "bad"),
            place(True, 2, .5, "bad"), place(1, 2, False, "bad"),
        ]
        for bad in invalid:
            for options in ({"top_n": 0}, {"top_n": 1}, {"min_score": .9}):
                with self.subTest(bad=bad, options=options):
                    with self.assertRaisesRegex(ValueError, "место 1"):
                        hotspots.match_sections([section(0, 10, "a")], [good, bad], **options)

    def test_valid_score_endpoints(self):
        zero = place(0, 1, 0, "zero")
        one = place(1, 2, 1, "one")
        self.assertEqual(hotspots.match_sections([section(0, 2, "a")], [zero, one], min_score=0),
                         [[zero, one]])

    def test_selection_mode_validation(self):
        for options in ({}, {"top_n": 1, "min_score": .5}, {"top_n": -1},
                        {"top_n": True}, {"min_score": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                hotspots.match_sections([], [], **options)


if __name__ == "__main__":
    unittest.main()
