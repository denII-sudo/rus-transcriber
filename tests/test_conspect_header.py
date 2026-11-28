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
        self.assertTrue(md.startswith(conspect.INFO_MARK + "\nСобытие: Планёрка\n"))
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


WORDS = "мы решили купить новый стол сегодня".split()


def spoken(speaker, revision=3):
    return {"recording_id": "r1", "source_revision": revision,
            "segments": [{"speaker": speaker, "words": [{"word_id": f"w{i}", "w": w, "s": float(i), "e": i + .9}
                                                        for i, w in enumerate(WORDS)]}],
            "review": [{"id": "a", "status": "resolved"}]}


class ReviewFindingsTests(unittest.TestCase):
    def linked(self, labels):
        obj, _ = conspect.link("Анна: «мы решили купить новый стол»", spoken("Анна"), labels=labels)
        obj.update(recording_id="r1", analysis_revision=3, assembly_revision=3)
        return obj

    def test_t260_3_speaker_change_counted_with_labels(self):
        for labels in (False, True):
            md, _ = conspect.render(self.linked(labels), spoken("Борис", 4))
            self.assertIn("изменено цитат 0, у говорящих 1", md, labels)

    def test_t260_3_unknown_speaker_is_not_zero(self):
        obj = self.linked(False)
        for quote in obj["quotes"].values():
            quote["base_speaker"] = quote["last_speaker"] = None
        md, _ = conspect.render(obj, spoken("Борис", 4))
        self.assertIn("у говорящих не определено", md)

    def test_t260_4_foreign_recording_shows_nothing_from_its_data(self):
        obj = agent_obj(recording_id="OTHER")
        md, _ = conspect.render(obj, data(title="Чужая запись", recorded_at="2026-11-20"))
        self.assertIn("Источник не подтверждён", md)
        self.assertNotIn("Чужая запись", md)
        self.assertNotIn("2026-11-20", md)
        self.assertNotIn("разобрано", md)

    def test_t260_4_conspect_without_recording_id_is_marked(self):
        obj = agent_obj()
        del obj["recording_id"]
        md, _ = conspect.render(obj, data(title="Планёрка"))
        self.assertIn("Событие: Планёрка", md)
        self.assertIn("Источник не подтверждён: у конспекта нет признака записи", md)

    def test_t260_1_missing_review_is_unknown_not_zero(self):
        without = data()
        del without["review"]
        md, _ = conspect.render(agent_obj(), without)
        self.assertIn("Места разбора: неизвестно", md)
        self.assertNotIn("мест разбора нет", md)

    def test_t260_5_own_header_is_not_duplicated(self):
        for obj in (agent_obj(), model_obj()):
            md, _ = conspect.render(obj, data(title="А"))
            again = dict(obj, template=md)
            md2, _ = conspect.render(again, data(title="А"))
            self.assertEqual(md2.count("Событие:"), 1, md2)
            relinked, _ = conspect.link(md, data(title="А"))
            self.assertNotIn("Событие:", relinked["template"])
            self.assertEqual(conspect.render(dict(relinked, recording_id="r1"), data(title="А"))[0].count("Составлен:"), 1)


    def test_t260_7_human_text_that_looks_like_the_header_survives(self):
        mine = "Событие: Мой план\nДата встречи: вторник\nСоставлен: мной\n\nОсновной текст."
        data_ = data(title="Планёрка")
        obj, _ = conspect.link(mine, data_)
        self.assertEqual(obj["template"], mine)
        md, _ = conspect.render(dict(agent_obj(), template=mine), data_)
        self.assertTrue(md.endswith("\n\n" + mine), md)
        self.assertEqual(md.count("Составлен: мной"), 1)
        # цикл render → link → render по-прежнему даёт одну шапку и сохраняет текст человека
        relinked, _ = conspect.link(md, data_)
        self.assertEqual(relinked["template"], mine)
        again, _ = conspect.render(dict(agent_obj(), template=relinked["template"]), data_)
        self.assertEqual(again.count("Составлен: ИИ-агентом"), 1)
        self.assertTrue(again.endswith("\n\n" + mine))

    def test_t260_3_missing_quote_with_changed_speaker_is_unknown(self):
        obj = conspect.link("Анна: «мы решили купить новый стол»", spoken("Анна"))[0]
        obj.update(recording_id="r1", analysis_revision=3, assembly_revision=3)
        changed = spoken("Борис", 4)
        for seg in changed["segments"]:
            for word in seg["words"]:
                word["word_id"] = "новый-" + word["word_id"]        # слова цитаты заменены: место не найдено
                word["w"] = "иное"
        md, rep = conspect.render(obj, changed)
        self.assertTrue(rep["missing"])
        self.assertIn("у говорящих не определено", md)
        self.assertNotIn("у говорящих 0", md)


if __name__ == "__main__":
    unittest.main()
