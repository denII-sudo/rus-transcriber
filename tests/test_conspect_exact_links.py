"""Н28-3, Н28-4, Н28-5 (перепроверка Opus №248): ссылка свёртки на конкретные слова.

ID первой части разбитого слова значил и «эту часть», и «всё слово распознавания». Теперь `conspect.link` пишет у
части ссылки строение её слов (`exact`): пока ни одно слово цитаты не разбили и не склеили с соседом, текст цитаты —
ровно эти ID, корень не разворачивается и `merged_from` не мешает. Иначе — как у старой ссылки (Н-7: укороченный
текст не выдаётся за прежний). Ссылки без отметки — как раньше. Сценарии — ops
`research/2026-11-28-recheck-248-250/runs/s248.py`, `s248g.py`. Данные синтетические.
"""
import copy
import os
import shutil
import sys
import tempfile
import unittest
import atexit

if "paths" not in sys.modules:          # временная папка данных — до импорта программы
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-exact-links-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)

import conspect  # noqa: E402
import transcript  # noqa: E402

START = {"w1": (0.2, 0.6), "w2": (0.8, 1.4), "w3": (1.6, 2.0), "w4": (2.2, 2.6)}
NOT_FOUND = "[место в стенограмме не найдено]"


def base():
    words = [{"w": t, "s": START[i][0], "e": START[i][1], "word_id": i}
             for t, i in (("тест", "w1"), ("слово", "w2"), ("ещё", "w3"), ("раз", "w4"))]
    return {"recording_id": "rec1", "source_revision": 1,
            "segments": [{"start": 0.2, "end": 2.6, "speaker": "Глеб", "text": "", "words": words}], "review": []}


def rewrite(data, texts, text, item):
    words = data["segments"][0]["words"]
    pos = [i for i, w in enumerate(words) if w["w"] in texts]
    i, j = pos[0], pos[-1] + 1
    old, origin = words[i:j], []
    new, conflicts, _ = transcript._diff_words(old, text, item, refuse=lambda w: None, origin=origin)
    assert not conflicts
    transcript._relink_places(data, old, origin)
    words[i:j] = new


def split():
    d = base()
    rewrite(d, ["слово"], "как бы так", "a1")
    return d


def merged():
    d = base()
    rewrite(d, ["слово"], "как бы", "a1")
    rewrite(d, ["бы", "ещё"], "бысть", "a2")
    return d


def quote(text, at_link, at_render, strip_exact=False):
    obj, rep = conspect.link(f"Глеб: «{text}»", at_link)
    if strip_exact:                                  # ссылка, связанная до этой правки
        for q in obj["quotes"].values():
            for p in q["parts"]:
                p.pop("exact", None)
    md, rrep = conspect.render(copy.deepcopy(obj), at_render)
    return rep["linked"], md.strip().splitlines()[-1], rrep["changed_since_link"]


class NewLinksAfterMerge(unittest.TestCase):
    """Н28-3: новая ссылка на слова, оставшиеся после слияния части с соседом, находится."""

    def test_quotes_present_word_for_word_are_found(self):
        d = merged()
        for text in ("тест как", "как бысть", "тест как бысть раз"):
            with self.subTest(text=text):
                self.assertEqual(quote(text, d, d), (1, f"Глеб: «{text}»", 0))

    def test_after_further_edit_of_the_part(self):
        d = merged()
        rewrite(d, ["как"], "каков", "a3")
        self.assertEqual(quote("тест каков", d, d), (1, "Глеб: «тест каков»", 0))


class LinkAfterSplit(unittest.TestCase):
    def test_neighbor_of_same_root_merged_outside_quote(self):
        """Н28-4: слова цитаты не тронуты — цитата прежняя, без «не найдено»."""
        d = split()
        at_link = copy.deepcopy(d)
        rewrite(d, ["так", "ещё"], "такие", "a2")
        self.assertEqual(quote("тест как бы", at_link, d)[1], "Глеб: «тест как бы»")

    def test_edit_outside_quote_in_same_root_does_not_enter_quote(self):
        """Н28-5: правка соседнего слова того же корня не добавляет в кавычки чужих слов."""
        for new in ("там", "так вот"):
            with self.subTest(new=new):
                d = split()
                at_link = copy.deepcopy(d)
                rewrite(d, ["так"], new, "a2")
                self.assertEqual(quote("тест как бы", at_link, d)[1], "Глеб: «тест как бы»")

    def test_split_of_quoted_word_is_not_silently_shortened(self):
        """Н-7 для точной ссылки: слово цитаты разбили — короткий прежний текст не выдаётся, видно изменение."""
        d = split()
        at_link = copy.deepcopy(d)
        rewrite(d, ["бы"], "бы же", "a2")
        linked, md, changed = quote("тест как бы", at_link, d)
        self.assertNotEqual(md, "Глеб: «тест как бы»")
        self.assertTrue(changed or NOT_FOUND in md, md)

    def test_quoted_word_edited_one_to_one_is_shown_changed(self):
        d = split()
        at_link = copy.deepcopy(d)
        rewrite(d, ["бы"], "же", "a2")
        self.assertEqual(quote("тест как бы", at_link, d)[1:], ("Глеб: «тест как же»", 1))

    def test_merge_of_quoted_word_with_neighbor_keeps_old_behaviour(self):
        d = split()
        at_link = copy.deepcopy(d)
        rewrite(d, ["бы", "так"], "бытак", "a2")
        self.assertEqual(quote("тест как бы", at_link, d)[1:],
                         quote("тест как бы", at_link, d, strip_exact=True)[1:])


class OldLinks(unittest.TestCase):
    def test_link_made_before_merge_is_still_not_found(self):
        """Н-7 (№248): ссылка до слияния части с соседом — «не найдено», а не укороченный текст."""
        d = base()
        rewrite(d, ["слово"], "как бы", "a1")
        at_link = copy.deepcopy(d)
        rewrite(d, ["бы", "ещё"], "бысть", "a2")
        self.assertIn(NOT_FOUND, quote("тест как бы", at_link, d)[1])

    def test_link_without_mark_works_as_before(self):
        d = merged()
        self.assertIn(NOT_FOUND, quote("тест как", d, d, strip_exact=True)[1])

    def test_mark_records_roots_and_span(self):
        d = split()
        obj, _ = conspect.link("Глеб: «тест как бы»", d)
        exact = obj["quotes"]["1"]["parts"][0]["exact"]
        self.assertEqual(set(exact["roots"]), {"w1", "w2"})
        self.assertEqual(len(exact["roots"]["w2"]), 3)
        self.assertEqual(exact["span"][0], 0.2)


if __name__ == "__main__":
    unittest.main()
