"""S24-1, S24-2: места записей из переноса старых файлов с неверной привязкой к словам (legacy_anchors).

Сценарий — как в воспроизведении (ops `research/2026-11-28-repro-s24/report.md`), на синтетике. Места старого файла
строит настоящий писатель старого режима `words.review_items` (у места-слова `t2` — начало последнего слова). Потом
человек вставил слова перед местом, а старый режим номера `seg/w1/w2` не пересчитывал. Перенос (удалён в У-1)
переводил номера в `word_ids` правилом из `anchor` ниже — копией его `anchor_review` и `_anchor_by_time`.

Проверяется:
- сверка по ревизии переноса находит чужие слова (S24-1) и неполные (S24-2) и не трогает верные места;
- «Применить», «Предпросмотр» ответа и решения агента на таких местах не вносятся, место уходит человеку с
  объяснением, слова и отметки человека не меняются;
- нормальное место, у которого `word_ids` после правки (`_relink_places`) начинаются позже `t2`, ложного отказа
  не получает: сверка идёт только по ревизии переноса, а у записи не из переноса её нет совсем;
- отчёт `rt.py legacy-anchor-check` считает места и закрытые ответом человека, `--summary` — без слов.
"""
import atexit
import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-anchors-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import core  # noqa: E402
import legacy_anchors as LA  # noqa: E402
import recordings  # noqa: E402
import review_pack  # noqa: E402
import settings  # noqa: E402
import transcript  # noqa: E402
import words  # noqa: E402

TIME_EPS = 0.05     # допуск удалённого переноса


def word(w, s, e, flags=()):
    return {"w": w, "s": s, "e": e, "word_id": uuid.uuid4().hex, **({"flags": list(flags)} if flags else {})}


def old_segments():
    """Старый файл до правок человека: помечены «подписали договор» (Анна), «в срок» (Борис), «хорошо» (Вера)."""
    return [
        {"start": 0.0, "end": 2.6, "speaker": "Анна", "label": "speaker-1",
         "words": [word("мы", 0.0, 0.4), word("подписали", 0.5, 1.0, ["low_conf"]),
                   word("договор", 1.1, 1.6, ["low_conf"]), word("аренды", 1.7, 2.1), word("вчера", 2.2, 2.6)]},
        {"start": 3.0, "end": 4.0, "speaker": "Борис", "label": "speaker-2",
         "words": [word("сделаем", 3.0, 3.3), word("в", 3.4, 3.5, ["low_conf"]), word("срок", 3.6, 4.0, ["low_conf"])]},
        {"start": 5.0, "end": 5.6, "speaker": "Вера", "label": "speaker-3",
         "words": [word("хорошо", 5.0, 5.6, ["low_conf"])]},
    ]


def human_edits(segments):
    """Правки человека в старом файле: «мы» → «мы с вами» (слова в пределах «мы»), у Бориса удалено слово перед
    местом. Номера мест старый режим не пересчитывал."""
    first = segments[0]["words"]
    first[0:1] = [word("мы", 0.0, 0.15), word("с", 0.15, 0.28), word("вами", 0.28, 0.4)]
    segments[1]["words"].pop(0)
    for seg in segments:
        seg["text"] = " ".join(w["w"] for w in seg["words"])


def anchor(segments, item):
    """Копия правила удалённого переноса (`anchor_review` на 270870c^): номера в диапазоне — как есть; иначе по
    времени со словами, кончающимися не позже t2 + 0,05."""
    si, w1, w2 = item["seg"], item["w1"], item["w2"]
    ws = segments[si]["words"] if 0 <= si < len(segments) else []
    if ws and 0 <= w1 < w2 <= len(ws):
        return [w["word_id"] for w in ws[w1:w2]]
    for seg in segments:
        if seg["speaker"] != item["speaker"]:
            continue
        inside = [w for w in seg["words"] if w["s"] >= item["t1"] - TIME_EPS and w["e"] <= item["t2"] + TIME_EPS]
        if inside:
            return [w["word_id"] for w in inside]
    return None


def migrated_data(audio):
    """Результат «как после переноса»: стенограмма после правок, места с word_ids по правилу переноса."""
    segments = old_segments()
    places = words.review_items(segments)
    human_edits(segments)
    for seg in segments:
        seg["text"] = " ".join(w["w"] for w in seg["words"])
        for w in seg["words"]:
            w.pop("flags", None)
    review = []
    for item in places:
        ids = anchor(segments, item)
        review.append({**{k: item[k] for k in ("id", "t", "t1", "t2", "speaker", "text", "reasons")},
                       "status": "open", "word_ids": ids})
    return {"file": str(audio), "duration": "00:00:06", "speakers": ["Анна", "Борис", "Вера"],
            "speaker_labels": ["speaker-1", "speaker-2", "speaker-3"], "group": "demo",
            "quality": {"migration": {"tool": "tests/test_legacy_anchors.py", "source": {"path": str(audio)}}},
            "segments": segments, "review": review, "review_done": False, "edits": []}


QUESTIONS = [{"id": rid, "question": "Проверьте", "recommended": 0,
              "options": [{"label": "Оставить как есть", "decision": {}},
                          {"label": "Другое", "decision": {"text": "другое"}}]} for rid in ("r1", "r2", "r3")]


@contextlib.contextmanager
def record(migrated=True, mutate=None):
    """Запись в базе с собранным пакетом: (jp, md, версия, путь базы, recording_id)."""
    with tempfile.TemporaryDirectory(dir=_TMP) as temp:
        root = Path(temp)
        db, audio = root / "state.sqlite3", root / "sample.wav"
        audio.write_bytes(b"synthetic")
        data = migrated_data(audio)
        if not migrated:
            data["quality"].pop("migration")
        if mutate:
            mutate(data)
        layout = recordings.document_layout(audio)
        layout["root"].mkdir()
        with mock.patch.object(settings.paths, "STATE_DB", db), \
                mock.patch.object(review_pack, "cut_clip", side_effect=lambda _s, t1, _t2, _out: t1):
            store = recordings.initialize_store(db, display_name="Тест")
            try:
                root_id, targets = recordings.prepare_new_layout(store, layout["root"], layout["paths"])
                rid = str(uuid.uuid4())
                recordings.commit_worker_result(
                    store, data, audio, root_id, recordings.operation_context(store, "model"),
                    recording_id=rid, operation_id=str(uuid.uuid4()),
                    generated_at="2026-11-28T00:00:00Z", document_paths=targets)
                recordings.materialize_recording(store, rid)
            finally:
                store.close()
            jp, md = layout["root"] / targets["json"], layout["root"] / targets["md"]
            pack = review_pack.build(jp, copy.deepcopy(QUESTIONS))
            yield jp, md, pack["version"], db, rid


def mismatches(db, rid):
    store = recordings.open_store(db, readonly=True)
    try:
        return LA.migration_mismatches(store, rid)
    finally:
        store.close()


def report(db, rid=None):
    store = recordings.open_store(db, readonly=True)
    try:
        return LA.check_store(store, [rid] if rid else None)
    finally:
        store.close()


class Case(unittest.TestCase):
    _ts = 0

    def answer(self, jp, version, aid, option=0):
        Case._ts += 1
        return review_pack.add_answers(jp, [{"id": aid, "version": version, "kind": "option", "option": option,
                                             "ts": f"2026-11-28T10:00:{Case._ts:02d}"}],
                                       "local", trusted_actor="human")

    @staticmethod
    def place(jp, rid):
        return next(r for r in transcript.load(jp)["review"] if r["id"] == rid)

    @staticmethod
    def human_marked(jp):
        return sorted(w["w"] for s in transcript.load(jp)["segments"] for w in s["words"]
                      if (w.get("provenance") or {}).get("text", {}).get("by") == "human")


class FindsMismatch(Case):
    def test_foreign_and_partial_found_correct_place_not(self):
        with record() as (jp, _md, _v, db, rid):
            found = mismatches(db, rid)
        self.assertEqual(set(found), {"r1", "r2"})
        self.assertEqual(found["r1"]["kind"], LA.FOREIGN)
        self.assertEqual(found["r1"]["bound_text"], "с вами")
        self.assertEqual(found["r1"]["by_time_text"], "подписали договор")
        self.assertEqual(found["r2"]["kind"], LA.PARTIAL)
        self.assertEqual(found["r2"]["bound_text"], "в")
        self.assertEqual(found["r2"]["by_time_text"], "в срок")

    def test_record_not_from_migration_is_not_checked(self):
        with record(migrated=False) as (_jp, _md, _v, db, rid):
            self.assertEqual(mismatches(db, rid), {})
            self.assertEqual((report(db, rid)["migrated"], report(db, rid)["places"]), (0, 0))


class ApplyGuard(Case):
    def test_keep_on_foreign_words_is_left_to_human(self):
        with record() as (jp, md, version, _db, _rid):
            before = transcript.load(jp)["segments"]
            self.answer(jp, version, "r1")
            rep = core.apply_answers_files(jp, md)
            row = next(r for r in rep["outcomes"] if r["id"] == "r1")
            self.assertEqual(row["outcome"], "apply_conflict")
            self.assertIn("перенесено из старых файлов", row["detail"])
            self.assertIn("«подписали договор»", row["detail"])
            self.assertEqual(transcript.load(jp)["segments"], before, "слова или отметки изменились")
            self.assertEqual(self.human_marked(jp), [])
            self.assertEqual(self.place(jp, "r1")["status"], "open")

    def test_partial_place_is_left_to_human(self):
        with record() as (jp, md, version, _db, _rid):
            self.answer(jp, version, "r2", option=1)
            rep = core.apply_answers_files(jp, md)
            row = next(r for r in rep["outcomes"] if r["id"] == "r2")
            self.assertEqual(row["outcome"], "apply_conflict")
            self.assertIn("«в срок»", row["detail"])
            self.assertNotIn("другое", [w["w"] for s in transcript.load(jp)["segments"] for w in s["words"]])

    def test_correct_place_of_same_record_applies(self):
        with record() as (jp, md, version, _db, _rid):
            self.answer(jp, version, "r3")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(next(r for r in rep["outcomes"] if r["id"] == "r3")["outcome"], "applied")
            self.assertEqual(self.human_marked(jp), ["хорошо"])

    def test_preview_says_the_same(self):
        with record() as (jp, md, version, _db, _rid):
            why = review_pack.answer_guard(jp, md, "r1", {"kind": "option", "option": 0, "version": version})
            self.assertIn("перенесено из старых файлов", why)
            self.assertIsNone(review_pack.answer_guard(jp, md, "r3", {"kind": "option", "option": 0,
                                                                      "version": version}))

    def test_agent_decision_refused_but_leave_to_human_passes(self):
        with record() as (jp, md, _version, _db, _rid):
            rep = core.apply_review_files(jp, md, [{"id": "r1", "text": "иное"}], by="claude")
            self.assertEqual([c["id"] for c in rep["conflicts"]], ["r1"])
            self.assertIn("перенесено из старых файлов", rep["conflicts"][0]["why"])
            self.assertNotIn("иное", [w["w"] for s in transcript.load(jp)["segments"] for w in s["words"]])
            rep = core.apply_review_files(jp, md, [{"id": "r1", "status": "human"}], by="claude")
            self.assertFalse(rep.get("conflicts"))
            self.assertEqual(self.place(jp, "r1")["status"], "human")


class ReviewFixes(Case):
    """Находки приёмки №265: O265-1 (= T265-1), O265-6 (= S265-4, T265-4), O265-7 (= S265-3, T265-5),
    O265-8 (= S265-5), выход для места после ремонта (O265-3)."""

    def words(self, jp):
        return [w["w"] for s in transcript.load(jp)["segments"] for w in s["words"]]

    def test_new_answer_over_applied_one_is_refused(self):
        """O265-1: «Другое» внесено на чужие слова до защиты; «как есть» поверх (К-В3) прежде снимало его и снова
        ставило отметку человека на чужие слова."""
        with record() as (jp, md, version, _db, _rid):
            self.answer(jp, version, "r1", option=1)
            with mock.patch.object(LA, "card_conflicts", return_value={}):
                core.apply_answers_files(jp, md)
            after = transcript.load(jp)["segments"]
            self.assertIn("другое", self.words(jp))
            self.assertIn("перенесено из старых файлов",
                          review_pack.answer_guard(jp, md, "r1", {"kind": "option", "option": 0, "version": version}))
            self.answer(jp, version, "r1", option=0)
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(next(r for r in rep["outcomes"] if r["id"] == "r1")["outcome"], "apply_conflict")
            self.assertEqual(transcript.load(jp)["segments"], after)

    def test_leave_to_human_from_card_passes(self):
        with record() as (jp, md, version, _db, _rid):
            Case._ts += 1
            review_pack.add_answers(jp, [{"id": "r1", "version": version, "kind": "human",
                                          "ts": f"2026-11-28T10:00:{Case._ts:02d}"}], "local", trusted_actor="human")
            self.assertIsNone(review_pack.answer_guard(jp, md, "r1", {"kind": "human", "version": version}))
            rep = core.apply_answers_files(jp, md)
            self.assertNotEqual(next(r for r in rep["outcomes"] if r["id"] == "r1")["outcome"], "apply_conflict")
            self.assertEqual(self.place(jp, "r1")["status"], "human")

    def test_text_names_transcript_not_audio_and_next_step(self):
        with record() as (jp, md, version, _db, _rid):
            self.answer(jp, version, "r1")
            detail = next(r for r in core.apply_answers_files(jp, md)["outcomes"] if r["id"] == "r1")["detail"]
        self.assertIn("по стенограмме здесь «подписали договор»", detail)
        self.assertIn("Этот ответ не внесён", detail)
        self.assertIn("ответить на неё нужно будет заново", detail)
        self.assertNotIn("звучит", detail)

    def test_repaired_place_is_not_held(self):
        with record() as (jp, md, version, _db, _rid):
            data = transcript.load(jp)
            next(r for r in data["review"] if r["id"] == "r1")[LA.REPAIRED] = True
            transcript.save(data, jp, md)
            self.answer(jp, version, "r1")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(next(r for r in rep["outcomes"] if r["id"] == "r1")["outcome"], "applied")

    def test_words_by_time_now_pass(self):
        """Решение агента со словами, которые стоят в это время сейчас (после правки слов места), не отказывается."""
        with record() as (jp, md, _version, _db, _rid):
            transcript.edit_span(jp, md, 0.45, 1.65, "подписали", by="claude", resolve=False)   # «договор» удалено
            data = transcript.load(jp)
            loc = transcript._find_span(data, {k: v for k, v in self.place(jp, "r1").items() if k != "word_ids"})
            now = [w["word_id"] for w in data["segments"][loc[0]]["words"][loc[1]:loc[2]]]
            rep = core.apply_review_files(jp, md, [{"id": "r1", "word_ids": now}], by="claude")
            self.assertFalse(rep.get("conflicts"))
            self.assertIn("r1", rep["successful_ids"])

    def test_recording_by_prefix_and_unknown(self):
        import rt
        with record() as (_jp, _md, _version, db, rid):
            out = report(db, rid[:8])
            self.assertEqual((out["places"], out["not_found"]), (2, []))
            buf = io.StringIO()
            with mock.patch("sys.stdout", buf), mock.patch("sys.stderr", io.StringIO()) as err:
                code = rt.main(["legacy-anchor-check", "--db", str(db), "--summary", "--recording", "ffffffff"])
        self.assertEqual(code, 1)
        self.assertIn("не найдены", err.getvalue())


class NoFalseRefusalAfterRelink(Case):
    def test_place_relinked_by_edit_is_checked_only_by_migration_revision(self):
        """Правка «хорошо» → «хорошо очень»: `_relink_places` переносит word_ids места на оба слова, «очень» начинается
        в 5,6 с, а t1/t2 места те же (5,0).
        Сверка нынешних данных со временем отказала бы; сверка по ревизии переноса — нет."""
        with record() as (jp, md, _version, db, rid):
            rep = core.apply_review_files(jp, md, [{"id": "r3", "text": "хорошо очень"}], by="human")
            self.assertIn("r3", rep["successful_ids"])
            now = transcript.load(jp)
            place = self.place(jp, "r3")
            self.assertEqual(len(place["word_ids"]), 2)
            self.assertIsNotNone(LA.place_mismatch(now, place), "наивная сверка по нынешним данным не ошиблась бы")
            self.assertNotIn("r3", mismatches(db, rid))
            rep = core.apply_review_files(jp, md, [{"id": "r3", "speaker": "Вера"}], by="human")
            self.assertFalse(rep.get("conflicts"))
            self.assertIn("r3", rep["successful_ids"])

    def test_record_not_from_migration_never_refused(self):
        """Запись не из переноса с word_ids места позже его t2 (как после правки) — ответ вносится."""
        def shifted(data):
            data["review"][2]["t2"] = data["review"][2]["t1"] = 4.9
        with record(migrated=False, mutate=shifted) as (jp, md, version, _db, _rid):
            self.answer(jp, version, "r3")
            rep = core.apply_answers_files(jp, md)
            self.assertEqual(next(r for r in rep["outcomes"] if r["id"] == "r3")["outcome"], "applied")


class Report(Case):
    def test_counts_and_closed_by_human_on_wrong_words(self):
        with record() as (jp, md, _version, db, rid):
            # как до этого исправления: «Оставить как есть» внесено на чужие слова
            transcript.apply_review(jp, md, [{"id": "r1"}], by="human")
            out = report(db)
        self.assertEqual((out["migrated"], out["places"], out["by_kind"]), (1, 2, {"foreign": 1, "partial": 1}))
        self.assertEqual((out["closed"], out["closed_with_human_marks"], out["human_marked_wrong_words"]), (1, 1, 2))
        row = next(r for r in out["records"][0]["places"] if r["id"] == "r1")
        self.assertEqual((row["status"], row["bound_text"]), ("resolved", "с вами"))
        self.assertFalse(out["writes"])

    def test_cli_summary_has_no_words(self):
        import rt
        with record() as (_jp, _md, _version, db, rid):
            buf = io.StringIO()
            with mock.patch("sys.stdout", buf):
                code = rt.main(["legacy-anchor-check", "--db", str(db), "--summary"])
        self.assertIn(code, (None, 0))
        text = buf.getvalue()
        out = json.loads(text[text.index("{"):])
        self.assertEqual(out["places"], 2)
        self.assertEqual(out["records"][0]["recording_id"], rid[:8])
        for secret in ("подписали", "договор", "срок", "вами"):
            self.assertNotIn(secret, text)


if __name__ == "__main__":
    unittest.main()
