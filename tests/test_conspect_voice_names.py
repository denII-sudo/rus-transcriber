"""Шапка конспекта и автор цитаты: имена по голосу без подтверждения человека помечаются (правило 26)."""
import unittest

import conspect


class VoiceNamesInConspectTests(unittest.TestCase):
    def data(self, **extra):
        return {"speaker_names": {"Спикер 1": "Анна"},
                "segments": [{"speaker": "Спикер 1", "words": [{"word_id": "w1"}]}], **extra}

    OBJ = {"speaker_labels": [{"label": "Участник 1", "word_ids": ["w1"]}]}

    def test_header_marks_unconfirmed_voice_name(self):
        head = conspect._labels_header(self.OBJ, self.data(names_source="voices", names_confirmed=False), False)
        self.assertIn("- Участник 1 — Анна (по голосу, не подтверждено)", head)

    def test_header_plain_when_confirmed_or_not_from_voices(self):
        for extra in ({"names_source": "voices", "names_confirmed": True}, {"names_source": "rename"}, {}):
            head = conspect._labels_header(self.OBJ, self.data(**extra), False)
            self.assertIn("- Участник 1 — Анна\n", head, extra)
            self.assertNotIn("по голосу", head)

    def test_unnamed_speaker_is_not_marked(self):
        data = self.data(names_source="voices", names_confirmed=False, speaker_names={})
        self.assertNotIn("по голосу", conspect._labels_header(self.OBJ, data, False))

    def test_quote_author_marked(self):
        author = conspect._author_of({}, self.data(names_source="voices", names_confirmed=False))
        self.assertEqual(author("Спикер 1"), "Анна (по голосу, не подтверждено)")


if __name__ == "__main__":
    unittest.main()
