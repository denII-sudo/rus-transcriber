"""The answer phrase table preserves the exact closed lists used by freeform."""
import tempfile
import unittest
from pathlib import Path

import freeform


EXPECTED_AGREE = {
    "да", "верно", "так", "правильно", "все верно", "всё верно", "да верно",
    "ага", "угу", "так и есть", "как есть", "да так",
}
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
    def test_current_agree_and_negative_sets_are_preserved(self):
        self.assertEqual(freeform.AGREE, EXPECTED_AGREE)
        self.assertEqual(freeform.NEGATIVES, EXPECTED_NEGATIVES)

    def test_installed_table_is_parsed_with_all_supported_kinds(self):
        phrases = freeform._read_answer_phrases()
        self.assertEqual(set(phrases), {
            "agree", "negative_prefix", "negative_intensifier", "negative_core",
        })
        self.assertEqual(set(phrases["agree"]), EXPECTED_AGREE)
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
            "empty.tsv", "id\tkind\ttext\tnote\nA001\tagree\t \t\n".encode("utf-8"),
            "пустой text", 2,
        )

    def test_repeated_id_is_rejected(self):
        self.assert_table_error(
            "duplicate-id.tsv",
            "id\tkind\ttext\tnote\nA001\tagree\tда\t\nA001\tagree\tага\t\n".encode("utf-8"),
            "повторный id", 3,
        )

    def test_repeated_phrase_is_rejected(self):
        self.assert_table_error(
            "duplicate-text.tsv",
            "id\tkind\ttext\tnote\nA001\tagree\tда\t\nA002\tagree\tда\t\n".encode("utf-8"),
            "повторная фраза", 3,
        )


if __name__ == "__main__":
    unittest.main()
