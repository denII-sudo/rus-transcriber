"""РАР-6.1, PR 17-0: «Применить» в файловом режиме держит замок ответов пакета.

Файловое «Применить» читает и пишет состояние пакета (`answers.json`: отметки применения, снимки отмены, исходы)
под `conspect.write_lock`. Дописывание и замена карточек (`review_pack.upsert_cards`), ответы и события страницы
пишут его под замком ответов (`answers.lock`). Без общего замка дописывание, прочитавшее состояние до «Применить» и
записавшее после, стирало отметку применения и снимок отмены: ответ выглядел неприменённым, отменить его было нельзя.

Порядок замков — «ответы, затем запись»: так их уже берёт `upsert_cards` (в файловом режиме `transcript.load` под
замком ответов берёт `write_lock`). Обратный порядок в «Применить» дал бы взаимную блокировку с дописыванием.
В режиме базы замок ответов в «Применить» не берётся: там потерю ловят ожидаемые ревизии, проигравший получает
`storage.Conflict`. Все данные синтетические.
"""
import contextlib
import threading
import time
import unittest
from unittest import mock

import conspect
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

    def test_files_apply_during_append_keeps_applied_marks(self):
        with files("legacy_files", data()) as (jp, md, version):
            self.press(jp, version, "Заменить на «тот»", "r4")
            rep, other = self.race(jp, md)
            self.assertNotIn("error", other)
            self.assertEqual([row["id"] for row in other["rep"]["added"]], ["r5"])
            self.assert_applied_kept(jp, rep)
            self.assertIn("r5", [it["id"] for it in review_pack.manifest(jp)["items"]])

    def test_sqlite_loser_gets_conflict_nothing_lost_and_retry_works(self):
        with files("sqlite", data()) as (jp, md, version):
            self.press(jp, version, "Заменить на «тот»", "r4")
            rep, other = self.race(jp, md)
            self.assertIsInstance(other.get("error"), storage.Conflict, other)
            self.assert_applied_kept(jp, rep)
            again = review_pack.upsert_cards(jp, [CARD_R5], why="тест")
            self.assertEqual([row["id"] for row in again["added"]], ["r5"])
            self.assert_applied_kept(jp, rep)


class LockOrder(Base):
    """Файловое «Применить»: каждая запись состояния пакета — под замком ответов и write_lock; замок ответов
    берётся первым и один раз (он файловый и не повторный: повтор в том же процессе завис бы)."""

    def test_every_state_write_is_under_both_locks_answers_first(self):
        depth = {"write": 0, "answers": 0}
        order, writes = [], []
        real_write_lock, real_answers_lock = conspect.write_lock, review_pack._answers_lock
        real_write_answers = review_pack._write_answers

        @contextlib.contextmanager
        def write_lock(jp):
            if not depth["write"]:
                order.append("write inside answers" if depth["answers"] else "write outside answers")
            with real_write_lock(jp):
                depth["write"] += 1
                try:
                    yield
                finally:
                    depth["write"] -= 1

        @contextlib.contextmanager
        def answers_lock(jp):
            if depth["answers"]:
                raise AssertionError("замок ответов взят повторно")
            with real_answers_lock(jp):
                depth["answers"] += 1
                try:
                    yield
                finally:
                    depth["answers"] -= 1

        def write_answers(jp, state):
            writes.append(dict(depth))
            return real_write_answers(jp, state)

        with files("legacy_files", data()) as (jp, md, version):
            self.press(jp, version, "Заменить на «тот»", "r4")
            self.save(jp, version, "r2", text="вон")            # свободный ответ: свой путь записи
            with mock.patch.object(conspect, "write_lock", side_effect=write_lock), \
                    mock.patch.object(review_pack, "_answers_lock", side_effect=answers_lock), \
                    mock.patch.object(review_pack, "_write_answers", side_effect=write_answers):
                rep = core._apply_answers_files_once(jp, md)
            self.assertEqual(self.outcomes(rep)["r4"]["outcome"], "applied", rep.get("outcomes"))
            self.assertTrue(writes, "«Применить» не записало состояние пакета")
            self.assertEqual([w for w in writes if not (w["write"] and w["answers"])], [], writes)
            self.assertEqual(order[:1], ["write inside answers"], order)

    def test_no_deadlock_with_append_holding_the_answers_lock(self):
        """Дописывание взяло замок ответов и сейчас возьмёт write_lock (transcript.load); «Применить» в это
        время не должно держать write_lock, ожидая замок ответов."""
        holds = threading.Event()
        real_manifest = review_pack.manifest
        result = {}

        def manifest(path):                   # _upsert_cards_locked: замок ответов взят, transcript.load впереди
            if threading.current_thread() is not threading.main_thread() and not holds.is_set():
                holds.set()
                time.sleep(0.5)               # «Применить» успевает дойти до замков
            return real_manifest(path)

        def run(name, fn):
            def body():
                try:
                    result[name] = fn()
                except Exception as ex:         # noqa: BLE001
                    result[name] = ex
            thread = threading.Thread(target=body, daemon=True)
            thread.start()
            return thread

        with files("legacy_files", data()) as (jp, md, version):
            self.press(jp, version, "Заменить на «тот»", "r4")
            with mock.patch.object(review_pack, "manifest", side_effect=manifest):
                append = run("append", lambda: review_pack.upsert_cards(jp, [CARD_R5], why="тест"))
                self.assertTrue(holds.wait(WAIT))
                apply = run("apply", lambda: core.apply_answers_files(jp, md))
                append.join(WAIT * 2)
                apply.join(WAIT * 2)
            self.assertFalse(append.is_alive() or apply.is_alive(), "взаимная блокировка «Применить» и дописывания")
            self.assertEqual([row["id"] for row in result["append"]["added"]], ["r5"], result)
            self.assert_applied_kept(jp, result["apply"])


if __name__ == "__main__":
    unittest.main()
