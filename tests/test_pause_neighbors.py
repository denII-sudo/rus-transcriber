"""РАР-6.1, PR 6а: пауза не владеет соседним словом.

Место-пропуск («missed» в reasons) владеет только словами, которые НАЧАТЫ внутри него, по полуоткрытому
интервалу [t1, t2) в целых сотых. Одна функция на правило: transcript.in_pause / pause_words / place_words /
places_overlap / word_in_place. Все данные синтетические.
"""
import atexit
import contextlib
import copy
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-pause-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import core  # noqa: E402
import quotes  # noqa: E402
import recordings  # noqa: E402
import review_pack  # noqa: E402
import settings  # noqa: E402
import transcript  # noqa: E402
import voices  # noqa: E402
import workflow  # noqa: E402

BACKENDS = ("legacy_files", "sqlite")


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


# ---------- записи в обоих режимах хранения ----------
AS_IS_REFUSAL = "в паузе нет слов — подтверждать «как есть» нечего; впишите услышанное в поле ответа («Имя: слова»)"
SPEAKER_REFUSAL = "в паузе нет слов — подписывать говорящего нечему; впишите услышанное в поле ответа («Имя: слова»)"
TEXT_REFUSAL = "вариант с текстом в пустую паузу не вписывается; впишите услышанное в поле ответа («Имя: слова»)"


def pause_result(inside=False, r1_status="open"):
    """Пауза r1 [1,0; 2,0) и место-слово r2 на «Ну» в 2,0 — сосед паузы (опора находок 25 и 38)."""
    words = [word("w1", "начало", 0.5, 0.9), word("w2", "Ну", 2.0, 2.2), word("w3", "дальше", 2.5, 2.9)]
    if inside:
        words.insert(1, word("w4", "ага", 1.4, 1.6))
    return {
        "file": "synthetic.wav", "duration": "00:00:04", "speakers": ["Анна"], "speaker_labels": ["speaker-1"],
        "timing": {"total_sec": 4}, "group": "demo", "quality": {"asr_incomplete_chunks": 0},
        "vocab_suggestions": [], "review_done": False, "edits": [],
        "segments": [{"start": 0.5, "end": 2.9, "label": "speaker-1", "speaker": "Анна",
                      "text": " ".join(w["w"] for w in words), "words": words}],
        "review": [gap(1.0, 2.0, id="r1", status=r1_status), place(2.0, id="r2", word_ids=["w2"])]}


def pause_questions(r1_decision=None):
    return [{"id": "r1", "question": "Проверьте", "recommended": 0,
             "options": [{"label": "Вариант", "decision": r1_decision or {}}]},
            {"id": "r2", "question": "Проверьте", "recommended": 0,
             "options": [{"label": "Оставить", "decision": {}}]}]


@contextlib.contextmanager
def pause_files(backend, data=None, questions=None):
    """Пара файлов стенограммы в выбранном режиме хранения и собранный пакет: (jp, md, версия)."""
    with tempfile.TemporaryDirectory(dir=_TMP) as temp:
        root = Path(temp)
        db, audio = root / "state.sqlite3", root / "sample.wav"
        audio.write_bytes(b"synthetic")
        data = copy.deepcopy(data or pause_result())
        data["file"] = str(audio)
        layout = recordings.document_layout(audio)
        layout["root"].mkdir()
        store = None
        with mock.patch.object(settings.paths, "STATE_DB", db), \
                mock.patch.object(review_pack, "cut_clip", side_effect=lambda _s, t1, _t2, _out: t1):
            try:
                if backend == "sqlite":
                    store = recordings.initialize_store(db, display_name="Тест")
                    root_id, targets = recordings.prepare_new_layout(store, layout["root"], layout["paths"])
                    rid = str(uuid.uuid4())
                    recordings.commit_worker_result(
                        store, data, audio, root_id, recordings.operation_context(store, "model"),
                        recording_id=rid, operation_id=str(uuid.uuid4()),
                        generated_at="2026-11-16T00:00:00Z", document_paths=targets)
                    recordings.materialize_recording(store, rid)
                    jp, md = layout["root"] / targets["json"], layout["root"] / targets["md"]
                else:
                    jp, md = layout["root"] / layout["paths"]["json"], layout["root"] / layout["paths"]["md"]
                    transcript.save(data, jp, md)
                pack = review_pack.build(jp, copy.deepcopy(pause_questions() if questions is None else questions))
                yield jp, md, pack.get("version")
            finally:
                if store:
                    store.close()


class PauseCase(unittest.TestCase):
    _ts = 0

    def save(self, jp, version, aid="r1", **row):
        PauseCase._ts += 1
        answer = {"id": aid, "version": version, "ts": f"2026-11-23T10:{PauseCase._ts // 60:02d}:"
                  f"{PauseCase._ts % 60:02d}", **row}
        return review_pack.add_answers(jp, [answer], "local", trusted_actor="human")

    @staticmethod
    def flat(jp):
        return [w for s in transcript.load(jp)["segments"] for w in s["words"]]

    def texts(self, jp):
        return [w["w"] for w in self.flat(jp)]

    def word(self, jp, wid):
        return next(w for w in self.flat(jp) if w.get("word_id") == wid)

    @staticmethod
    def place_of(jp, rid):
        return next(i for i in transcript.load(jp)["review"] if i["id"] == rid)

    @staticmethod
    def outcomes(rep):
        return {row["id"]: row for row in rep.get("outcomes") or []}

    def apply(self, jp, md):
        """Применить ответы как человек; повторный запуск ничего не меняет."""
        rep = core.apply_answers_files(jp, md)
        before = transcript.load(jp)
        again = core.apply_answers_files(jp, md)
        self.assertEqual(transcript.load(jp), before, "повторный запуск изменил стенограмму")
        self.assertEqual({k: v["outcome"] for k, v in self.outcomes(again).items()},
                         {k: v["outcome"] for k, v in self.outcomes(rep).items()})
        return rep

    def assertUntouchedNeighbors(self, jp):
        for wid in ("w1", "w2", "w3"):
            w = self.word(jp, wid)
            for key in ("by", "provenance", "seen", "edited"):
                self.assertNotIn(key, w, f"{wid}: появилось поле {key}")


class PauseAsIsTests(PauseCase):
    def test_empty_pause_refuses_as_is_speaker_and_text_with_reason(self):
        cases = (({}, AS_IS_REFUSAL), ({"speaker": "Анна"}, SPEAKER_REFUSAL), ({"text": "вот"}, TEXT_REFUSAL),
                 ({"speaker": "Анна", "text": "вот"}, TEXT_REFUSAL))
        for backend in BACKENDS:
            for decision, why in cases:
                with self.subTest(backend=backend, decision=decision), \
                        pause_files(backend, questions=pause_questions(decision)) as (jp, md, version):
                    before = transcript.load(jp)
                    self.save(jp, version, option=0)
                    rep = self.apply(jp, md)
                    row = self.outcomes(rep)["r1"]
                    self.assertEqual(row["outcome"], "apply_conflict")
                    self.assertIn(why, row.get("detail", ""))
                    self.assertEqual(self.texts(jp), ["начало", "Ну", "дальше"])
                    self.assertUntouchedNeighbors(jp)
                    self.assertEqual(self.place_of(jp, "r1")["status"], "open")
                    self.assertEqual(transcript.load(jp)["segments"], before["segments"])

    def test_apply_review_refuses_for_agent_and_human_without_changing_data(self):
        cases = (({"id": "r1"}, AS_IS_REFUSAL), ({"id": "r1", "speaker": "Анна"}, SPEAKER_REFUSAL),
                 ({"id": "r1", "text": "вот"}, TEXT_REFUSAL))
        for backend in BACKENDS:
            for decision, why in cases:
                for by in ("claude", "human"):
                    with self.subTest(backend=backend, decision=decision, by=by), \
                            pause_files(backend) as (jp, md, _version):
                        before = transcript.load(jp)
                        _, rep = transcript.apply_review(jp, md, [decision], by=by)
                        self.assertEqual(rep.get("conflicts"), [{"id": "r1", "why": why}])
                        self.assertFalse(rep.get("not_found"))
                        self.assertEqual(rep["successful_ids"], [])
                        self.assertEqual(transcript.load(jp), before)

    def test_pause_with_words_inside_confirms_only_its_own_words(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), \
                    pause_files(backend, pause_result(inside=True)) as (jp, md, version):
                self.save(jp, version, option=0)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "applied")
                self.assertEqual(self.place_of(jp, "r1")["status"], "resolved")
                inside = self.word(jp, "w4")
                self.assertEqual(inside.get("by"), "human")
                self.assertIn("text", inside.get("provenance") or {})
                self.assertUntouchedNeighbors(jp)

    def test_build_does_not_give_pause_the_neighbor_word(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), pause_files(backend) as (jp, md, _version):
                items = {i["id"]: i for i in review_pack.manifest(jp)["items"]}
                self.assertNotIn("word_ids", items["r1"])
                self.assertEqual(items["r2"]["word_ids"], ["w2"])

    def test_old_pack_with_neighbor_in_pause_word_ids_is_read_by_the_new_rule(self):
        """Старый пакет (так писал main) нёс у паузы ID соседнего слова; он не читается."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), pause_files(backend) as (jp, md, version):
                state = review_pack.answers_state(jp)
                next(i for i in state["manifest"]["items"] if i["id"] == "r1")["word_ids"] = ["w2"]
                review_pack._write_answers(jp, state)
                self.save(jp, version, option=0)
                decisions, _ = review_pack.decisions(jp)
                self.assertTrue(all("word_ids" not in d for d in decisions if d["id"] == "r1"))
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "apply_conflict")
                self.assertUntouchedNeighbors(jp)
                self.assertEqual(self.texts(jp), ["начало", "Ну", "дальше"])


class PauseWithWordsTests(PauseCase):
    def test_find_span_of_a_pause_never_reads_word_ids(self):
        data = pause_result()
        self.assertIsNone(transcript._find_span(data, gap(1.0, 2.0, word_ids=["w2"])))
        inside = pause_result(inside=True)
        self.assertEqual(transcript._find_span(inside, gap(1.0, 2.0, word_ids=["w2"])), (0, 1, 2))
        self.assertEqual(transcript._find_span(inside, place(2.0, word_ids=["w2"])), (0, 2, 3))

    def test_build_gives_a_pause_with_words_no_word_ids_either(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), pause_files(backend, pause_result(inside=True)) as (jp, _md, _v):
                items = {i["id"]: i for i in review_pack.manifest(jp)["items"]}
                self.assertNotIn("word_ids", items["r1"])
                self.assertEqual(items["r2"]["word_ids"], ["w2"])

    def test_old_pack_neighbor_edit_does_not_block_and_pause_word_does_in_sqlite(self):
        """Старый пакет с ID соседа в word_ids паузы: сверяются слова паузы, а не соседа."""
        for neighbor_edited in (True, False):
            with self.subTest(neighbor_edited=neighbor_edited), pause_files("sqlite") as (jp, md, version):
                state = review_pack.answers_state(jp)
                next(i for i in state["manifest"]["items"] if i["id"] == "r1")["word_ids"] = ["w2"]
                review_pack._write_answers(jp, state)
                if neighbor_edited:
                    transcript.edit_span(jp, md, 2.0, 2.2, "ну", by="human")
                else:
                    transcript.edit_span(jp, md, 1.4, 1.6, "ага", speaker="Анна", by="human")
                self.save(jp, version, text="Анна: угу")
                rep = self.apply(jp, md)
                want = "applied" if neighbor_edited else "apply_conflict"
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], want, rep.get("outcomes"))


class PauseSqliteGuardTests(PauseCase):
    def test_free_answer_on_empty_pause_is_applied(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), pause_files(backend) as (jp, md, version):
                self.save(jp, version, text="Анна: угу")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.texts(jp), ["начало", "угу", "Ну", "дальше"])

    def test_word_written_into_pause_after_build_blocks_the_answer_in_sqlite(self):
        with pause_files("sqlite") as (jp, md, version):
            transcript.edit_span(jp, md, 1.4, 1.6, "ага", speaker="Анна", by="human")
            self.assertEqual(self.texts(jp), ["начало", "ага", "Ну", "дальше"])
            before = copy.deepcopy(transcript.load(jp)["segments"])
            self.save(jp, version, text="Анна: угу")
            rep = self.apply(jp, md)
            self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "apply_conflict")
            self.assertTrue(rep.get("revision_conflicts"))
            self.assertEqual(transcript.load(jp)["segments"], before)

    def test_unrelated_edit_next_to_pause_does_not_block_in_sqlite(self):
        with pause_files("sqlite") as (jp, md, version):
            transcript.edit_span(jp, md, 2.5, 2.9, "далее", by="human")
            self.save(jp, version, text="Анна: угу")
            rep = self.apply(jp, md)
            self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "applied", rep.get("outcomes"))
            self.assertEqual(self.texts(jp), ["начало", "угу", "Ну", "далее"])

    def test_pause_snapshot_in_guard_is_empty_list_not_none(self):
        data = pause_result()
        manifest = {"items": [{"id": "r1", "t1": 1.0, "t2": 2.0, "reasons": ["missed"]}]}
        state = {"answers": {"r1": {"kind": "option", "option": 0}}}
        self.assertEqual(core._answer_revision_conflicts(data, manifest, state, copy.deepcopy(data), sqlite=True), {})
        changed = copy.deepcopy(data)
        changed["segments"][0]["words"].insert(1, word("w4", "ага", 1.4, 1.6))
        self.assertIn("r1", core._answer_revision_conflicts(changed, manifest, state, data, sqlite=True))
        bad = {"items": [{"id": "r1", "t1": None, "t2": 2.0, "reasons": ["missed"]}]}
        self.assertIn("r1", core._answer_revision_conflicts(data, bad, state, copy.deepcopy(data), sqlite=True))


class ListeningFingerprintTests(PauseCase):
    W1 = {"word_id": "w1", "w": "начало", "s": 0.5, "e": 1.05}
    W2 = {"word_id": "w2", "w": "Ну", "s": 2.0, "e": 2.2}

    def test_fingerprints_are_those_of_main(self):
        self.assertEqual(transcript._listening_fingerprint([self.W1, self.W2]),
                         "0e3f869fb15d6e7d381baf96545d807def1af13a8be803ca9f0d186855b47233")
        self.assertEqual(transcript._listening_fingerprint([]),
                         "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945")

    def test_listening_words_are_those_overlapping_the_interval(self):
        data = {"segments": [{"speaker": "Анна", "words": [self.W1, self.W2]}]}
        self.assertEqual([w["w"] for w in transcript._interval_words(data, 1.0, 2.0)], ["начало"])

    def test_resolved_listening_survives_a_save_by_the_new_code(self):
        for backend in BACKENDS:
            data = pause_result()
            data["segments"][0]["words"][0] = dict(self.W1)
            data["review"] = [{"id": "r9", "status": "resolved", "t": 1.0, "t1": 1.0, "t2": 2.0, "gap": [1.0, 2.0],
                               "speaker": None, "text": "", "reasons": ["asr", "missed"],
                               "asr_problem": {"model": "whisper", "state": "empty_unknown", "words_from": []},
                               "listening": {"by": "human", "result": "words", "at": "2026-11-23T10:00:00Z",
                                             "fingerprint": transcript._listening_fingerprint([self.W1])}}]
            with self.subTest(backend=backend), pause_files(backend, data, []) as (jp, md, _version):
                transcript.edit_span(jp, md, 2.5, 2.9, "далее", by="human")
                item = self.place_of(jp, "r9")
                self.assertEqual(item["status"], "resolved")
                self.assertEqual(item["listening"]["result"], "words")


class HeldPauseTests(PauseCase):
    def held(self, backend):
        return pause_files(backend, pause_result(r1_status="human"))

    def test_agent_may_edit_the_neighbor_of_a_held_pause(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), self.held(backend) as (jp, md, _version):
                _, rep = transcript.apply_review(jp, md, [{"id": "r2", "text": "ну"}], by="claude")
                self.assertFalse(rep.get("conflicts"), rep)
                self.assertEqual(self.word(jp, "w2")["w"], "ну")
                self.assertEqual(self.place_of(jp, "r1")["status"], "human")

    def test_agent_may_place_text_over_the_neighbor_but_not_inside_the_pause(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), self.held(backend) as (jp, md, _version):
                _, rep = transcript.apply_word_edits(
                    jp, md, [{"op": "place_text", "t1": 2.0, "t2": 2.5, "text": "ну дальше", "card": "a1",
                              "word_ids": ["w2", "w3"]}], by="claude")
                self.assertFalse(rep["failed"], rep)
                self.assertEqual(self.texts(jp), ["начало", "ну", "дальше"])
                _, rep = transcript.apply_word_edits(
                    jp, md, [{"op": "insert", "s": 1.4, "e": 1.5, "text": "ага", "speaker": "Анна",
                              "after_id": "w1", "card": "a2"}], by="claude")
                self.assertIn("оставлено человеку", rep["failed"].get("a2", ""))
                _, rep = transcript.apply_word_edits(
                    jp, md, [{"op": "place_text", "t1": 1.2, "t2": 1.3, "text": "ага", "speaker": "Анна",
                              "card": "a3"}], by="claude")
                self.assertIn("оставлено человеку", rep["failed"].get("a3", ""))
                self.assertEqual(self.texts(jp), ["начало", "ну", "дальше"])
                self.assertEqual(self.place_of(jp, "r1")["status"], "human")


class HeldSpanBoundaryTests(unittest.TestCase):
    def setUp(self):
        data = pause_result(r1_status="human")
        data["review"].append(place(0.5, 0.9, id="r3", status="human", word_ids=["w1"]))
        self.words, self.spans = transcript._held_places(data)
        self.data = data

    def test_held_pause_holds_only_its_own_words(self):
        pause = pause_result(inside=True, r1_status="human")
        held_words, _spans = transcript._held_places(pause)
        held = {w["word_id"] for seg in pause["segments"] for w in seg["words"] if id(w) in held_words}
        self.assertEqual(held, {"w4"})

    def test_time_boundaries_of_a_held_pause_are_half_open(self):
        only_pause = transcript._held_places(pause_result(r1_status="human"))[1]
        self.assertTrue(transcript._span_has_time(only_pause, 1.0))
        self.assertTrue(transcript._span_has_time(only_pause, 1.994))
        self.assertFalse(transcript._span_has_time(only_pause, 1.996))
        self.assertFalse(transcript._span_has_time(only_pause, 2.0))
        self.assertFalse(transcript._span_has_time(only_pause, 0.994))
        self.assertFalse(transcript._span_has_time(only_pause, None))

    def test_explicit_span_overlap_with_a_held_pause(self):
        only_pause = transcript._held_places(pause_result(r1_status="human"))[1]
        self.assertFalse(transcript._span_overlaps(only_pause, 2.0, 2.5))
        self.assertFalse(transcript._span_overlaps(only_pause, 0.5, 0.99))
        self.assertTrue(transcript._span_overlaps(only_pause, 0.5, 1.0))
        self.assertTrue(transcript._span_overlaps(only_pause, 1.9, 2.5))

    def test_word_place_keeps_closed_interval_with_tolerance(self):
        word_span = [s for s in self.spans if not s[2]]
        self.assertTrue(transcript._span_has_time(word_span, 0.5 - 0.01))
        self.assertTrue(transcript._span_overlaps(word_span, 0.9, 1.5))
        self.assertTrue(transcript._span_overlaps(word_span, 0.905, 1.5))
        self.assertFalse(transcript._span_overlaps(word_span, 0.93, 1.5))


class VoiceSuggestionTests(unittest.TestCase):
    @staticmethod
    def data(pause_words=()):
        words = [dict(word("w1", "начало", 0.5, 0.9)), {**word("w2", "Ну", 2.0, 3.0), "by": "human"},
                 *pause_words]
        return {"segments": [{"speaker": "Анна", "words": sorted(words, key=lambda w: w["s"])}], "edits": [],
                "review": [gap(1.0, 2.0, id="r1", status="resolved")]}

    def test_resolved_pause_does_not_offer_the_voice_of_its_neighbor(self):
        found = voices.legacy_speaker_places(self.data())
        self.assertEqual([p["id"] for p in found["places"]], [])
        self.assertEqual(found["skipped"]["no_words"], 1)

    def test_resolved_pause_offers_its_own_human_words(self):
        own = {**word("w4", "ага", 1.0, 1.9), "by": "human"}
        found = voices.legacy_speaker_places(self.data((own,)))
        self.assertEqual([p["id"] for p in found["places"]], ["r1"])
        self.assertEqual((found["places"][0]["w1"], found["places"][0]["w2"]), (1, 2))


# ---------- свободные ответы на паузу: узнавание соседнего слова, отказы, пересечение карточек ----------
def custom_result(words, review):
    """words — [(word_id, слово, начало, говорящий)]; соседние слова одного говорящего — одна реплика."""
    segments = []
    for wid, text, s, speaker in words:
        if not segments or segments[-1]["speaker"] != speaker:
            segments.append({"start": s, "end": s + 0.2, "label": "speaker-1", "speaker": speaker,
                             "text": "", "words": []})
        segments[-1]["words"].append(word(wid, text, s))
    for seg in segments:
        seg["text"] = " ".join(w["w"] for w in seg["words"])
        seg["end"] = seg["words"][-1]["e"]
    data = pause_result()
    data["segments"], data["review"] = segments, review
    data["speakers"] = sorted({w[3] for w in words})
    return data


def one_question(rid="r1"):
    return [{"id": rid, "question": "Проверьте", "recommended": 0, "options": [{"label": "Вариант", "decision": {}}]}]


class FreeAnswerCase(PauseCase):
    def run_answer(self, backend, data, text, questions=None, aid="r1"):
        """Ответ текстом на карточку aid; -> (отчёт, стенограмма) внутри открытых файлов не возвращается:
        возвращаются тексты слов, чтобы файлы можно было закрыть."""
        with pause_files(backend, data, questions or one_question(aid)) as (jp, md, version):
            self.save(jp, version, aid=aid, text=text)
            rep = self.apply(jp, md)
            return rep, self.texts(jp), transcript.load(jp)


class NeighborRecognitionTests(FreeAnswerCase):
    DATA = staticmethod(lambda: custom_result(
        [("w1", "начало", 0.5, "Анна"), ("w2", "Ну", 2.0, "Анна"), ("w3", "дальше", 2.5, "Анна")],
        [gap(1.0, 2.0, id="r1")]))

    def test_neighbor_after_the_pause_is_recognized(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                rep, texts, data = self.run_answer(backend, self.DATA(), "Анна: Вот. Ну")
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "applied")
                self.assertEqual(texts, ["начало", "Вот.", "Ну", "дальше"])
                neighbor = next(w for s in data["segments"] for w in s["words"] if w["word_id"] == "w2")
                for key in ("by", "provenance", "edited"):
                    self.assertNotIn(key, neighbor)

    def test_neighbor_before_the_pause_is_recognized(self):
        data = custom_result([("w1", "Ну", 0.5, "Анна"), ("w3", "дальше", 2.5, "Анна")], [gap(1.0, 2.0, id="r1")])
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                rep, texts, _ = self.run_answer(backend, data, "Анна: Ну вот")
                self.assertEqual(texts, ["Ну", "вот", "дальше"])

    def test_one_answer_token_is_one_transcript_word(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                rep, texts, _ = self.run_answer(backend, self.DATA(), "Анна: ну, ну")
                self.assertEqual(texts, ["начало", "ну,", "Ну", "дальше"])

    def test_other_speaker_or_far_neighbor_is_not_recognized(self):
        other = custom_result([("w1", "начало", 0.5, "Анна"), ("w2", "Ну", 2.0, "Борис"), ("w3", "дальше", 2.5, "Борис")],
                              [gap(1.0, 2.0, id="r1")])
        far = custom_result([("w1", "начало", 0.5, "Анна"), ("w2", "Ну", 4.5, "Анна")], [gap(1.0, 2.0, id="r1")])
        for backend in BACKENDS:
            with self.subTest(backend=backend, case="другой говорящий"):
                _, texts, _ = self.run_answer(backend, other, "Анна: Вот. Ну")
                self.assertEqual(texts, ["начало", "Вот.", "Ну", "Ну", "дальше"])
            with self.subTest(backend=backend, case="дальше предела"):
                _, texts, _ = self.run_answer(backend, far, "Анна: Вот. Ну")
                self.assertEqual(texts, ["начало", "Вот.", "Ну", "Ну"])

    def test_neighbor_before_the_pause_farther_than_the_limit_is_not_recognized(self):
        data = custom_result([("w1", "Ну", 0.5, "Анна"), ("w3", "дальше", 4.5, "Анна")], [gap(3.0, 4.0, id="r1")])
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                _, texts, _ = self.run_answer(backend, data, "Анна: Ну вот")
                self.assertEqual(texts, ["Ну", "Ну", "вот", "дальше"])

    def test_answer_equal_to_the_neighbor_goes_to_human_with_reason(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                rep, texts, data = self.run_answer(backend, self.DATA(), "Анна: Ну")
                self.assertEqual(texts, ["начало", "Ну", "дальше"])
                place_ = next(i for i in data["review"] if i["id"] == "r1")
                self.assertEqual(place_["status"], "human")
                self.assertIn("совпадает со словами стенограммы рядом с паузой", place_.get("note", ""))
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "not_applied")

    def test_rebuilt_pack_with_the_same_answer_does_not_duplicate(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), pause_files(backend, self.DATA(), one_question()) as (jp, md, version):
                self.save(jp, version, text="Анна: вот")
                self.apply(jp, md)
                self.assertEqual(self.texts(jp), ["начало", "вот", "Ну", "дальше"])
                again = review_pack.build(jp, one_question())
                self.save(jp, again["version"], text="Анна: вот")
                self.apply(jp, md)
                self.assertEqual(self.texts(jp), ["начало", "вот", "Ну", "дальше"])
                self.assertEqual(self.place_of(jp, "r1")["status"], "resolved")

    def test_open_pause_with_equal_words_is_closed_and_confirmed_by_human(self):
        """Ответ совпал со словами, уже стоящими в паузе: место закрывается, слова подтверждены, соседи — нет."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), \
                    pause_files(backend, pause_result(inside=True), one_question()) as (jp, md, version):
                self.save(jp, version, text="Анна: ага")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "applied", rep.get("outcomes"))
                self.assertEqual(self.texts(jp), ["начало", "ага", "Ну", "дальше"])
                self.assertEqual(self.place_of(jp, "r1")["status"], "resolved")
                self.assertEqual(self.word(jp, "w4").get("by"), "human")
                self.assertUntouchedNeighbors(jp)

    def test_closed_pause_with_equal_words_changes_nothing(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), \
                    pause_files(backend, pause_result(inside=True, r1_status="resolved"),
                                one_question()) as (jp, md, version):
                before = transcript.load(jp)
                self.save(jp, version, text="Анна: ага")
                self.apply(jp, md)
                after = transcript.load(jp)
                self.assertEqual(after["segments"], before["segments"])
                self.assertEqual(self.place_of(jp, "r1")["status"], "resolved")
                self.assertNotIn("by", self.word(jp, "w4"))

    def test_other_words_in_a_filled_pause_go_to_human_without_insert(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), \
                    pause_files(backend, pause_result(inside=True), one_question()) as (jp, md, version):
                self.save(jp, version, text="Анна: вот")
                rep = self.apply(jp, md)
                self.assertEqual(self.texts(jp), ["начало", "ага", "Ну", "дальше"])
                item = self.place_of(jp, "r1")
                self.assertEqual(item["status"], "human")
                self.assertIn("в паузе уже есть слова", item.get("note", ""))
                self.assertEqual(self.outcomes(rep)["r1"]["outcome"], "not_applied")


class PauseOverlapAnswersTests(FreeAnswerCase):
    def test_pause_and_word_next_to_it_are_answered_together(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), pause_files(backend) as (jp, md, version):
                self.save(jp, version, aid="r1", text="Анна: вот")
                self.save(jp, version, aid="r2", option=0)
                rep = self.apply(jp, md)
                rows = self.outcomes(rep)
                self.assertEqual((rows["r1"]["outcome"], rows["r2"]["outcome"]), ("applied", "applied"))
                self.assertFalse(rep.get("preflight_conflicts"))
                self.assertEqual((self.place_of(jp, "r1")["status"], self.place_of(jp, "r2")["status"]),
                                 ("resolved", "resolved"))
                self.assertEqual(self.texts(jp), ["начало", "вот", "Ну", "дальше"])

    def test_plan_does_not_close_the_neighbor_word_place_through_the_pause(self):
        import freeform
        data = pause_result()
        item = {"id": "r1", "t1": 1.0, "t2": 2.0, "speaker": "Анна", "reasons": ["missed"], "text": ""}
        _edits, _reports, closed, _conflicts = freeform.plan(
            {"items": [item]}, {"r1": {"text": "Анна: вот", "ts": "2026-11-23T10:00:00"}}, data)
        self.assertEqual(closed, {})

    def test_plan_closes_a_word_place_inside_the_pause(self):
        import freeform
        data = pause_result(inside=True)
        data["review"].append(place(1.4, id="r3", word_ids=["w4"]))
        item = {"id": "r1", "t1": 1.0, "t2": 2.0, "speaker": "Анна", "reasons": ["missed"], "text": ""}
        _edits, _reports, closed, _conflicts = freeform.plan(
            {"items": [item]}, {"r1": {"text": "Анна: ага", "ts": "2026-11-23T10:00:00"}}, data)
        self.assertEqual(closed, {"r3": "r1"})

    def test_three_answers_one_refusal_does_not_stop_the_others(self):
        data = pause_result()
        data["review"].append(gap(2.2, 2.5, id="r5"))
        questions = [
            {"id": "r1", "question": "Проверьте", "recommended": 0, "options": [{"label": "Как есть", "decision": {}}]},
            {"id": "r2", "question": "Проверьте", "recommended": 0,
             "options": [{"label": "ну", "decision": {"text": "ну"}}]},
            {"id": "r5", "question": "Проверьте", "recommended": 0, "options": [{"label": "Как есть", "decision": {}}]}]
        for backend in BACKENDS:
            with self.subTest(backend=backend), pause_files(backend, data, questions) as (jp, md, version):
                self.save(jp, version, aid="r1", option=0)
                self.save(jp, version, aid="r2", option=0)
                self.save(jp, version, aid="r5", text="Анна: эээ")
                rep = self.apply(jp, md)
                rows = {k: v["outcome"] for k, v in self.outcomes(rep).items()}
                self.assertEqual(rows, {"r1": "apply_conflict", "r2": "applied", "r5": "applied"})
                self.assertEqual(self.texts(jp), ["начало", "ну", "эээ", "дальше"])
                self.assertEqual(self.place_of(jp, "r1")["status"], "open")


class NameOnPauseTests(FreeAnswerCase):
    def test_name_alone_on_empty_pause_goes_to_human_with_reason(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                rep, texts, data = self.run_answer(backend, pause_result(), "Анна")
                self.assertEqual(texts, ["начало", "Ну", "дальше"])
                item = next(i for i in data["review"] if i["id"] == "r1")
                self.assertEqual(item["status"], "human")
                self.assertIn("в паузе нет слов: одно имя ничего не подписывает", item.get("note", ""))
                neighbor = next(w for s in data["segments"] for w in s["words"] if w["word_id"] == "w2")
                self.assertNotIn("by", neighbor)

    def test_name_relabels_only_the_words_of_the_pause(self):
        data = custom_result([("w1", "начало", 0.5, "Анна"), ("w4", "ага", 1.4, "Борис"), ("w2", "Ну", 2.0, "Борис")],
                             [gap(1.0, 2.0, id="r1")])
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                rep, _texts, result = self.run_answer(backend, data, "Анна")
                speakers = {w["word_id"]: s["speaker"] for s in result["segments"] for w in s["words"]}
                self.assertEqual(speakers, {"w1": "Анна", "w4": "Анна", "w2": "Борис"})


class OldPackFreeAnswersTests(FreeAnswerCase):
    @staticmethod
    def manifest_sha(jp):
        import hashlib
        import json
        return hashlib.sha256(json.dumps(review_pack.manifest(jp), sort_keys=True, ensure_ascii=False)
                              .encode("utf-8")).hexdigest()

    def test_old_pack_is_read_by_the_new_rule_and_not_rewritten(self):
        data = custom_result([("w1", "начало", 0.5, "Анна"), ("w2", "Ну", 2.0, "Анна"), ("w3", "дальше", 2.5, "Анна")],
                             [gap(1.0, 2.0, id="r1")])
        cases = (("как есть", {"option": 0}, ["начало", "Ну", "дальше"], "apply_conflict"),
                 ("имя", {"text": "Анна"}, ["начало", "Ну", "дальше"], "not_applied"),
                 ("слова", {"text": "Анна: Вот. Ну"}, ["начало", "Вот.", "Ну", "дальше"], "applied"))
        for backend in BACKENDS:
            for name, answer, want_texts, want_outcome in cases:
                with self.subTest(backend=backend, answer=name), pause_files(backend, data, one_question()) as (jp, md, version):
                    state = review_pack.answers_state(jp)
                    state["manifest"]["items"][0]["word_ids"] = ["w2"]
                    review_pack._write_answers(jp, state)
                    sha = self.manifest_sha(jp)
                    self.save(jp, version, **answer)
                    rep = self.apply(jp, md)
                    self.assertEqual(self.outcomes(rep)["r1"]["outcome"], want_outcome, rep.get("outcomes"))
                    self.assertEqual(self.texts(jp), want_texts)
                    self.assertUntouchedNeighbors(jp)
                    self.assertEqual(self.manifest_sha(jp), sha)


class DoubtMarksPauseTests(unittest.TestCase):
    """Тест 19 (ответ владельца на раздел 9, вариант А): открытая пустая пауза помечает ⟦…⟧ последнее слово до неё
    и первое после; слово «в 0,01 с от края» больше не слово паузы; цитата рядом с такой паузой сомнительна."""

    def make(self, words=None, speaker="Анна"):
        words = words or [word("w1", "раз", 0.5), word("w2", "два", 1.0), word("w3", "три", 3.0),
                          word("w4", "четыре", 3.5), word("w5", "пять", 4.0), word("w6", "шесть", 4.5)]
        text = " ".join(w["w"] for w in words)
        return {"segments": [{"speaker": speaker, "start": words[0]["s"], "end": words[-1]["e"], "text": text,
                              "words": words}], "review": []}

    def marks(self, item, data=None):
        data = data or self.make()
        data["review"] = [item]
        return transcript.doubt_marks(data)

    def test_empty_open_pause_marks_the_word_before_and_the_word_after(self):
        self.assertEqual(self.marks(gap(1.5, 3.0)), {0: {1: "r1", 2: "r1"}})

    def test_pause_left_for_human_is_marked_too(self):
        self.assertEqual(self.marks(gap(1.5, 3.0, status="human")), {0: {1: "r1", 2: "r1"}})

    def test_word_within_a_hundredth_of_the_edge_is_not_a_pause_word_any_more(self):
        words = [word("w1", "раз", 0.5), word("w2", "два", 1.0), word("w3", "три", 3.0), word("w4", "ещё", 3.01),
                 word("w5", "пять", 4.0)]
        # «три» — первое слово после паузы; «ещё» в 0,01 с от края раньше тоже помечалось, теперь нет
        self.assertEqual(self.marks(gap(1.5, 3.0), self.make(words)), {0: {1: "r1", 2: "r1"}})

    def test_pause_with_its_own_words_marks_only_them(self):
        words = [word("w1", "раз", 0.5), word("w2", "два", 1.0), word("w3", "полтора", 2.0), word("w4", "три", 3.0)]
        self.assertEqual(self.marks(gap(1.5, 3.0), self.make(words)), {0: {2: "r1"}})

    def test_neighbors_in_different_segments_are_marked_each_in_its_own(self):
        data = {"segments": [
            {"speaker": "Анна", "start": 0.5, "end": 1.2, "text": "раз два",
             "words": [word("w1", "раз", 0.5), word("w2", "два", 1.0)]},
            {"speaker": "Борис", "start": 3.0, "end": 3.7, "text": "три четыре",
             "words": [word("w3", "три", 3.0), word("w4", "четыре", 3.5)]}], "review": [gap(1.5, 3.0)]}
        self.assertEqual(transcript.doubt_marks(data), {0: {1: "r1"}, 1: {0: "r1"}})

    def test_pause_at_the_edges_of_the_recording_marks_one_word(self):
        self.assertEqual(self.marks(gap(5.0, 6.0)), {0: {5: "r1"}})
        self.assertEqual(self.marks(gap(0.0, 0.4)), {0: {0: "r1"}})

    def test_overlapping_speech_in_the_pause_marks_the_speakers_own_segment(self):
        data = {"segments": [
            {"speaker": "Борис", "start": 2.0, "end": 2.4, "text": "эх", "words": [word("w1", "эх", 2.0)]},
            {"speaker": "Анна", "start": 2.2, "end": 2.6, "text": "ага", "words": [word("w2", "ага", 2.2)]}],
            "review": [gap(1.5, 3.0, speaker="Анна")]}
        self.assertEqual(transcript.doubt_marks(data), {1: {0: "r1"}})
        data["review"] = [gap(1.5, 3.0, speaker="Борис")]
        self.assertEqual(transcript.doubt_marks(data), {0: {0: "r1"}})

    def test_neighbors_render_as_one_doubt_span(self):
        data = self.make()
        marks = self.marks(gap(1.5, 3.0), data)
        self.assertEqual(transcript._marked_text(data["segments"][0], marks[0]),
                         "раз ⟦два три⟧ четыре пять шесть")

    def test_resolved_pause_is_not_marked(self):
        for status in ("done", "applied", "accepted"):
            self.assertEqual(self.marks(gap(1.5, 3.0, status=status)), {}, status)

    def test_bad_time_marks_nothing_and_does_not_raise(self):
        for t1, t2 in ((None, 2.0), (float("nan"), 2.0), ("1", 2.0), (3.0, 2.0)):
            item = {"id": "r1", "status": "open", "t1": t1, "t2": t2, "reasons": ["missed"], "speaker": "Анна"}
            self.assertEqual(self.marks(item), {}, (t1, t2))

    def test_gap_among_other_reasons_is_a_pause(self):
        self.assertEqual(self.marks(gap(1.5, 3.0, reasons=["asr", "missed"])), {0: {1: "r1", 2: "r1"}})

    def test_word_place_keeps_the_old_rule(self):
        self.assertEqual(self.marks(place(3.0)), {0: {2: "r2"}})
        self.assertEqual(self.marks(place(2.99, 3.01)), {0: {2: "r2"}})

    def test_quote_next_to_an_open_empty_pause_is_in_doubt(self):
        data = self.make()
        data["review"] = [gap(1.5, 3.0)]
        for quote, want in (("«раз два»", ["r1"]), ("«три четыре»", ["r1"]), ("«два три»", ["r1"]),
                            ("«четыре пять шесть»", None)):
            with self.subTest(quote=quote):
                rep = quotes.check(f"Анна: {quote}", data)
                rows = rep["problems"] if want else []
                if want:
                    self.assertEqual(rows[0].get("doubt"), want, rep)
                else:
                    self.assertEqual(rep["summary"]["in_doubt"], 0, rep)
                    self.assertEqual(rep["summary"]["verbatim"], 1, rep)

    def test_quote_is_not_in_doubt_once_the_pause_is_resolved(self):
        data = self.make()
        data["review"] = [gap(1.5, 3.0, status="done")]
        rep = quotes.check("Анна: «два три»", data)
        self.assertEqual(rep["summary"]["in_doubt"], 0, rep)


if __name__ == "__main__":
    unittest.main()
