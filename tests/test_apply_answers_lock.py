"""РАР-6.1, PR 17-0: «Применить» и дописывание карточек не теряют друг друга в режиме базы.

Дописывание и замена карточек (`review_pack.upsert_cards`) и «Применить» пишут состояние пакета записи. В режиме
базы потерю ловят ожидаемые ревизии: проигравший получает `storage.Conflict`, ничего не теряется, повтор проходит.
Файловая часть PR 17-0 (замок ответов `answers.lock` и порядок замков «ответы, затем запись» в файловом
«Применить») убрана в Л-3.2 вместе с файловым режимом. Все данные синтетические.
"""
import contextlib
import threading
import time
import unittest
from unittest import mock

import storage
import test_pause_buttons_short_answers as six
from test_pause_buttons_short_answers import files

core = six.core
review_pack = six.review_pack

# место r5 (слово w1) есть в стенограмме, но не в пакете: его карточку дописывает второй поток
R5 = {"id": "r5", "status": "open", "t": 0.5, "t1": 0.5, "t2": 0.5, "speaker": "Анна",
      "reasons": ["conf"], "text": "начало", "word_ids": ["w1"]}
CARD_R5 = {"id": "r5", "question": "Проверьте", "options": [six.AS_IS]}
WAIT = 5


def data():
    return six.result(extra_review=[R5])


def wait_any(events, timeout=WAIT):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if any(e.is_set() for e in events):
            return True
        time.sleep(0.01)
    return False


class Base(six.Case):
    def assert_applied_kept(self, jp, rep):
        self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
        self.assertEqual(self.word(jp, "w5")["w"], "тот")
        store = review_pack.answers_state(jp)
        self.assertIn("r4", store.get("applied_answers") or {}, "отметка применения стёрта")
        self.assertIn("r4", store.get("undo_snapshots") or {}, "снимок отмены стёрт")
        self.assertEqual(review_pack.answer_states(jp)["r4"]["state"], "applied")


class RaceWithAppend(Base):
    """Дописывание карточки прочитало состояние пакета и режет клип; в это время человек нажал «Применить»."""

    def race(self, jp, md):
        """-> (отчёт «Применить», {"rep" | "error"} второго потока)."""
        read, done, wants = threading.Event(), threading.Event(), threading.Event()
        real_attach, real_lock = review_pack._attach_clip, review_pack._answers_lock
        result = {}

        def attach(*args, **kwargs):          # upsert_cards: состояние прочитано, запись — впереди
            read.set()
            wait_any([done, wants])           # «Применить» закончилось или ждёт замок ответов
            return real_attach(*args, **kwargs)

        @contextlib.contextmanager
        def answers_lock(path):
            if threading.current_thread() is threading.main_thread():
                wants.set()
            with real_lock(path):
                yield

        def append():
            try:
                result["rep"] = review_pack.upsert_cards(jp, [CARD_R5], why="тест")
            except Exception as ex:             # noqa: BLE001  (исход второго потока проверяет тест)
                result["error"] = ex

        worker = threading.Thread(target=append, daemon=True)
        with mock.patch.object(review_pack, "_attach_clip", side_effect=attach), \
                mock.patch.object(review_pack, "_answers_lock", side_effect=answers_lock):
            worker.start()
            self.assertTrue(read.wait(WAIT), "дописывание не дошло до резки клипа")
            try:
                rep = core.apply_answers_files(jp, md)
            finally:
                done.set()
            worker.join(WAIT * 2)
        self.assertFalse(worker.is_alive(), "второй поток завис")
        return rep, result

    def test_sqlite_loser_gets_conflict_nothing_lost_and_retry_works(self):
        with files("sqlite", data()) as (jp, md, version):
            self.press(jp, version, "Заменить на «тот»", "r4")
            rep, other = self.race(jp, md)
            self.assertIsInstance(other.get("error"), storage.Conflict, other)
            self.assert_applied_kept(jp, rep)
            again = review_pack.upsert_cards(jp, [CARD_R5], why="тест")
            self.assertEqual([row["id"] for row in again["added"]], ["r5"])
            self.assert_applied_kept(jp, rep)


if __name__ == "__main__":
    unittest.main()
