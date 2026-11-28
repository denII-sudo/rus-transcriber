"""Шапка свёрток и конспектов (РАР-7, 7.6а): событие, дата, кто составил, версия стенограммы, разобрано N из M."""
import os
import tempfile
import unittest
from pathlib import Path

import conspect


def data(**extra):
    return {"recording_id": "r1", "source_revision": 3,
            "segments": [{"speaker": "Спикер 1", "words": [{"word_id": "w1", "w": "да", "s": 1.0}]}],
            "review": [{"id": "a", "status": "resolved"}, {"id": "b", "status": "open"},
                       {"id": "c", "status": "human"}], **extra}


def model_obj(**extra):
    return {"template": "Текст.", "quotes": {}, "recording_id": "r1", "analysis_revision": 3,
            "quote_policy": 1, "unlinked": [], "speaker_labels": [{"label": "Участник 1", "word_ids": ["w1"]}],
            **extra}


def agent_obj(**extra):
    return {"template": "Текст агента.", "quotes": {}, "recording_id": "r1", "analysis_revision": 3, **extra}


class HeaderTests(unittest.TestCase):
    def test_model_header(self):
        md, _ = conspect.render(model_obj(), data(title="Планёрка", recorded_at="2026-11-20T10:00:00"))
        self.assertIn("Событие: Планёрка\nДата записи: 2026-11-20\nСоставлен: моделью OpenAI через OpenCode\n", md)
        self.assertIn("Стенограмма: версия 3\nМеста разбора: разобрано 1 из 3\n", md)
        self.assertEqual(md.count("Создан моделью"), 1)
        self.assertTrue(md.startswith("# Конспект\n\nСоздан моделью"))
        self.assertTrue(md.endswith("Текст."))

    def test_agent_header_has_no_model_wording(self):
        md, _ = conspect.render(agent_obj(), data(title="Планёрка", recorded_at="2026-11-20"))
        self.assertTrue(md.startswith("Событие: Планёрка\n"))
        self.assertIn("Составлен: ИИ-агентом\n", md)
        self.assertNotIn("OpenAI", md)
        self.assertTrue(md.endswith("Текст агента."))

    def test_event_falls_back_to_audio_name_and_file_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "встреча.mp3"
            audio.write_bytes(b"x")
            os.utime(audio, (1_700_000_000, 1_700_000_000))
            md, _ = conspect.render(agent_obj(), data(file=str(audio)))
        self.assertIn("Событие: встреча.mp3\nДата файла аудио: 2023-11-1", md)
        self.assertNotIn(tmp, md)

    def test_date_unknown(self):
        md, _ = conspect.render(agent_obj(), data(file=str(Path(tempfile.gettempdir()) / "нет-такого-файла.mp3")))
        self.assertIn("Дата записи: не установлена", md)

    def test_no_places(self):
        md, _ = conspect.render(agent_obj(), data(review=[]))
        self.assertIn("Места разбора: мест разбора нет", md)

    def test_rebuild_after_transcript_change(self):
        obj = agent_obj(assembly_revision=2)
        md, _ = conspect.render(obj, data())
        self.assertIn("Стенограмма: собрана по версии стенограммы 2, сейчас 3: изменено цитат 0, у говорящих 0", md)
        md2, _ = conspect.render(obj, data())          # вторая сборка: версия уже учтена
        self.assertIn("Стенограмма: версия 3\n", md2)

    def test_quotes_line_not_duplicated(self):
        obj = model_obj(template="Видно ⟦q1⟧.", quotes={}, unlinked=[], unlinked_count=1)
        md, _ = conspect.render(obj, data())
        self.assertEqual(md.count("Цитат:"), 1)


if __name__ == "__main__":
    unittest.main()
