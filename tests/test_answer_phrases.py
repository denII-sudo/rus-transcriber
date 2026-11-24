"""The answer phrase table preserves the exact closed lists used by freeform."""
import tempfile
import unittest
from pathlib import Path

import freeform


# PR 6б (В8): прежние строки agree разделены на короткие слова и подтверждения; ID и тексты прежние.
EXPECTED_SHORT_WORDS = {"да", "так", "ага", "угу", "нет"}
EXPECTED_CONFIRMS = {
    "верно", "правильно", "все верно", "всё верно", "да верно", "так и есть", "как есть", "да так",
}
EXPECTED_EVALUATION_STEMS = ("согла", "рекоменд", "предлож", "верно", "правильн")
EXPECTED_NEGATIVE_PREFIXES = ("нет", "это", "я", "нет-нет")
EXPECTED_NEGATIVE_INTENSIFIERS = ("совершенно", "совсем", "абсолютно")
EXPECTED_NEGATIVE_CORES = (
    "неверно", "не верно", "неправильно", "не правильно", "не согласен",
    "не согласна", "не согласны", "не согласен с этим", "не согласна с этим",
    "не согласны с этим",
)
EXPECTED_NEGATIVES = frozenset(
    " ".join(part for part in (prefix, intensifier, core) if part)
    for prefix in ("",) + EXPECTED_NEGATIVE_PREFIXES
    for intensifier in ("",) + EXPECTED_NEGATIVE_INTENSIFIERS
    for core in EXPECTED_NEGATIVE_CORES
)


class AnswerPhraseSnapshotTests(unittest.TestCase):
    def test_current_short_answer_and_negative_sets_are_preserved(self):
        self.assertEqual(freeform.SHORT_WORDS, EXPECTED_SHORT_WORDS)
        self.assertEqual(freeform.CONFIRMS, {x.replace("ё", "е") for x in EXPECTED_CONFIRMS})
        self.assertEqual(freeform.EVALUATION_STEMS, EXPECTED_EVALUATION_STEMS)
        self.assertEqual(freeform.NEGATIVES, EXPECTED_NEGATIVES)

    def test_installed_table_is_parsed_with_all_supported_kinds(self):
        phrases = freeform._read_answer_phrases()
        self.assertEqual(set(phrases), {
            "short_word", "confirm", "evaluation_stem", "negative_prefix", "negative_intensifier", "negative_core",
        })
        self.assertEqual(set(phrases["short_word"]), EXPECTED_SHORT_WORDS)
        self.assertEqual(set(phrases["confirm"]), EXPECTED_CONFIRMS)
        self.assertEqual(tuple(phrases["evaluation_stem"]), EXPECTED_EVALUATION_STEMS)
        self.assertEqual(tuple(phrases["negative_prefix"]), EXPECTED_NEGATIVE_PREFIXES)
        self.assertEqual(tuple(phrases["negative_intensifier"]), EXPECTED_NEGATIVE_INTENSIFIERS)
        self.assertEqual(tuple(phrases["negative_core"]), EXPECTED_NEGATIVE_CORES)

    def assert_table_error(self, name, data, message, line):
        with tempfile.TemporaryDirectory(prefix="answer-phrases-") as directory:
            path = Path(directory) / name
            if data is not None:
                path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, message) as caught:
                freeform._read_answer_phrases(path)
        self.assertIn(name, str(caught.exception))
        self.assertIn(f"строка {line}", str(caught.exception))

    def test_missing_table_is_a_clear_error(self):
        self.assert_table_error("missing.tsv", None, "не удалось прочитать", 1)

    def test_header_only_table_is_rejected_for_missing_categories(self):
        self.assert_table_error(
            "header-only.tsv", "id\tkind\ttext\tnote\n".encode("utf-8"),
            "обязательные категории: short_word, confirm, evaluation_stem, negative_prefix, "
            "negative_intensifier, negative_core", 2,
        )

    def test_each_required_category_must_be_present(self):
        rows = (
            ("A001", "short_word", "да", ""),
            ("A002", "confirm", "верно", ""),
            ("E001", "evaluation_stem", "согла", ""),
            ("N001", "negative_prefix", "нет", ""),
            ("N002", "negative_intensifier", "совсем", ""),
            ("N003", "negative_core", "неверно", ""),
        )
        for missing_kind in (
            "short_word", "confirm", "evaluation_stem", "negative_prefix", "negative_intensifier", "negative_core",
        ):
            with self.subTest(kind=missing_kind):
                table = "id\tkind\ttext\tnote\n" + "".join(
                    "\t".join(row) + "\n" for row in rows if row[1] != missing_kind
                )
                self.assert_table_error(
                    f"missing-{missing_kind}.tsv", table.encode("utf-8"),
                    f"обязательные категории: {missing_kind}", 7,
                )

    def test_bom_is_rejected(self):
        self.assert_table_error(
            "bom.tsv", b"\xef\xbb\xbfid\tkind\ttext\tnote\n",
            "не должен содержать BOM", 1,
        )

    def test_unknown_kind_is_rejected(self):
        self.assert_table_error(
            "unknown.tsv", "id\tkind\ttext\tnote\nA001\tunknown\tword\t\n".encode("utf-8"),
            "неизвестный kind", 2,
        )

    def test_empty_text_is_rejected(self):
        self.assert_table_error(
            "empty.tsv", "id\tkind\ttext\tnote\nA001\tshort_word\t \t\n".encode("utf-8"),
            "пустой text", 2,
        )

    def test_repeated_id_is_rejected(self):
        self.assert_table_error(
            "duplicate-id.tsv",
            "id\tkind\ttext\tnote\nA001\tshort_word\tда\t\nA001\tshort_word\tага\t\n".encode("utf-8"),
            "повторный id", 3,
        )

    def test_repeated_phrase_is_rejected(self):
        self.assert_table_error(
            "duplicate-text.tsv",
            "id\tkind\ttext\tnote\nA001\tshort_word\tда\t\nA002\tshort_word\tда\t\n".encode("utf-8"),
            "повторная фраза", 3,
        )


if __name__ == "__main__":
    unittest.main()
