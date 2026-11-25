"""РАР-6.1, PR 17а: следующая порция карточек дописывается в открытый пакет.

`core.build_review_pack` при собранном пакете той же записи дописывает карточки (`review_pack.append_cards`):
version прежняя, pack_rev + 1, ответы, отметки применения, снимки отмены и события прежних карточек остаются.
Место, у которого карточка уже есть, — `skipped` «уже в пакете»; карточка с неприменимым ответом получает новую
редакцию (решение владельца В-6), противоречие дописывается само. Клипы режутся до замка во временные имена
и переименовываются под замком. Полная пересборка — только `rebuild=True` (с проверкой 17-Б). Пакет — до 200
карточек (В-5). Только при хранении записей в базе; в файловом режиме без rebuild — отказ. Данные синтетические.
"""
import atexit
import os
import shutil
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="rt-append17a-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:          # временная папка данных — до импорта программы
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP

import copy  # noqa: E402
import unittest  # noqa: E402
from unittest import mock  # noqa: E402

import test_pause_buttons_short_answers as six  # noqa: E402
from test_pause_buttons_short_answers import files  # noqa: E402

core = six.core
review_pack = six.review_pack
REPLACE = "Заменить на «тот»"


def first():
    return [q for q in six.questions() if q["id"] in ("r1", "r2")]


def second():
    return [q for q in six.questions() if q["id"] in ("r3", "r4")]


def ids(jp):
    return [i["id"] for i in review_pack.manifest(jp)["items"]]


def parts(jp):
    root = review_pack.pack_dir(jp) / "clips"
    return sorted(p.name for p in root.rglob("*.part.mp3")) if root.exists() else []


class Base(six.Case):
    def append(self, jp, qs=None, **kw):
        with mock.patch.object(core, "_done_json", return_value={"json": str(jp)}):
            return core.build_review_pack("demo", second() if qs is None else qs, **kw)


class Append(Base):
    def test_second_portion_keeps_version_answers_and_events(self):
        with files("sqlite", qs=first()) as (jp, _md, version):
            self.save(jp, version, "r2", text="вон")
            review_pack.add_interaction_event(jp, "r2", version, "audio_play", "session-1")
            before = copy.deepcopy(review_pack.answers_state(jp))
            rep = self.append(jp)
            self.assertEqual((rep["version"], rep["pack_rev"], rep["written"]), (version, 1, True))
            self.assertEqual([r["id"] for r in rep["added"]], ["r3", "r4"])
            self.assertEqual(ids(jp), ["r1", "r2", "r3", "r4"])
            after = review_pack.answers_state(jp)
            self.assertEqual(after["answers"], before["answers"])
            self.assertEqual(after["interaction_events"], before["interaction_events"])
            self.assertEqual(review_pack.answer_states(jp)["r2"]["state"], "saved")
            self.assertEqual(parts(jp), [], "временные клипы остались")
            self.assertFalse([i["clip"] for i in review_pack.manifest(jp)["items"] if ".part." in (i["clip"] or "")])
            self.assertTrue(all(i.get("source_revision") is not None
                                for i in review_pack.manifest(jp)["items"][2:]))

    def test_applied_answer_survives_and_undo_works_after_append(self):
        with files("sqlite", qs=[q for q in six.questions() if q["id"] == "r4"]) as (jp, md, version):
            self.press(jp, version, REPLACE, "r4")
            self.assertEqual(self.outcomes(self.apply(jp, md))["r4"]["outcome"], "applied")
            snaps = copy.deepcopy(review_pack.answers_state(jp).get("undo_snapshots"))
            self.append(jp, [q for q in six.questions() if q["id"] == "r2"])
            self.assertEqual(review_pack.answers_state(jp).get("undo_snapshots"), snaps)
            self.assertEqual(review_pack.answer_states(jp)["r4"]["state"], "applied")
            self.save(jp, version, "r4", kind="revert")
            self.assertEqual(self.outcomes(self.apply(jp, md))["r4"]["outcome"], "reverted")
            self.assertEqual(self.word(jp, "w5")["w"], "та")

    def test_card_already_in_pack_is_skipped(self):
        with files("sqlite") as (jp, _md, version):
            self.press(jp, version, REPLACE, "r4")
            rep = self.append(jp)
            self.assertFalse(rep["written"])
            self.assertEqual({r["id"]: r["why"] for r in rep["skipped"]}, {"r3": "уже в пакете", "r4": "уже в пакете"})
            self.assertEqual(review_pack.answer_states(jp)["r4"]["state"], "saved")

    def test_card_with_unapplicable_answer_gets_a_new_revision(self):
        with files("sqlite") as (jp, _md, version):
            self.press(jp, version, REPLACE, "r4")
            real = review_pack.answer_states_of

            def states(store, man, *a, **kw):
                out = real(store, man, *a, **kw)
                if "r4" in out:
                    out["r4"] = dict(out["r4"], state="not_found")
                return out

            qs = [dict(q, question="Проверьте ещё раз") for q in second() if q["id"] == "r4"]
            with mock.patch.object(review_pack, "answer_states_of", side_effect=states):
                rep = self.append(jp, qs)
            self.assertEqual(rep["replaced"], [{"id": "r4", "rev": 2, "clip": True}])
            item = review_pack.manifest(jp)["items"][ids(jp).index("r4")]
            self.assertEqual(item["history"][-1]["rev"], 1)
            self.assertRegex(item["clip"] or "", r"r4.r2(.[0-9a-f]+)?.mp3$", item)

    def test_old_listening_card_without_listen_fp_is_not_refreshed(self):
        """Старая карточка «модели расходятся» без listen_fp (до К-5) с неприменимым ответом: тот же вопрос —
        не новая редакция и не «уже в пакете»."""
        with files("sqlite") as (jp, _md, version):
            store = review_pack.answers_state(jp)
            for item in store["manifest"]["items"]:
                item.pop("listen_fp", None) if item["id"] == "r3" else None
            review_pack._write_answers(jp, store)
            real = review_pack.answer_states_of
            with mock.patch.object(review_pack, "answer_states_of",
                                   side_effect=lambda s, m, *a, **k: {**real(s, m, *a, **k),
                                                                      "r3": {"state": "listen_mismatch", "card": 3}}):
                rep = self.append(jp, [q for q in six.questions() if q["id"] == "r3"])
            self.assertEqual((rep["unchanged"], rep["replaced"], rep["skipped"]), (["r3"], [], []))

    def test_pending_contradiction_is_appended_by_itself(self):
        """Противоречие r4 без карточки в пакете (так бывает при card: not_replaced, 13б) дописывается
        вместе с порцией, хотя агент его не назвал."""
        with files("sqlite", qs=[q for q in six.questions() if q["id"] == "r4"]) as (jp, md, version):
            self.press(jp, version, REPLACE, "r4")
            core.apply_answers_files(jp, md)
            core.edit_span_files(jp, md, 3.0, 3.0, "тут", by="human")
            store = review_pack.answers_state(jp)
            store["manifest"]["items"] = [i for i in store["manifest"]["items"] if i["id"] != "r4"]
            review_pack._write_answers(jp, store)
            rep = self.append(jp, [q for q in six.questions() if q["id"] == "r2"])
            self.assertEqual(sorted(r["id"] for r in rep["added"]), ["r2", "r4"])

    def test_limit_200(self):
        with files("sqlite", qs=first()) as (jp, _md, _version):
            with mock.patch.object(review_pack, "APPEND_LIMIT", 3):
                rep = self.append(jp)
            self.assertEqual([r["id"] for r in rep["added"]], ["r3"])
            self.assertIn("пакет полон", rep["skipped"][0]["why"])
            self.assertEqual(rep["skipped"][0]["id"], "r4")
        self.assertEqual(review_pack.APPEND_LIMIT, 200)


class Races(Base):
    def test_answer_saved_while_clips_are_cut_is_kept(self):
        with files("sqlite", qs=first()) as (jp, _md, version):
            saved = []

            def cut(_src, t1, _t2, _out):
                if not saved:
                    saved.append(self.save(jp, version, "r2", text="вон"))
                return t1

            with mock.patch.object(review_pack, "cut_clip", side_effect=cut):
                rep = self.append(jp)
            self.assertTrue(saved)
            self.assertTrue(rep["written"])
            self.assertEqual(review_pack.answer_states(jp)["r2"]["state"], "saved")

    def test_place_changed_while_clips_are_cut_is_skipped(self):
        with files("sqlite", qs=first()) as (jp, _md, _version):
            real_card = review_pack._card
            calls = []

            def card(T, data, it, q):
                calls.append(it["id"])
                out = real_card(T, data, it, q)
                if it["id"] == "r4" and calls.count("r4") == 2 and out[0]:
                    out[0]["text"] = "изменилось"
                return out

            with mock.patch.object(review_pack, "_card", side_effect=card):
                rep = self.append(jp)
            self.assertEqual([r["id"] for r in rep["added"]], ["r3"])
            self.assertEqual(rep["skipped"], [{"id": "r4", "why": "место изменилось во время сборки, повторите порцию"}])
            self.assertEqual(parts(jp), [])

    def test_conflict_from_apply_retries_once(self):
        with files("sqlite", qs=first()) as (jp, _md, _version):
            import storage
            real = review_pack._write_answers
            calls = []

            def write(path, state):
                calls.append(1)
                if len(calls) == 1:
                    raise storage.Conflict("stale review pack state")
                return real(path, state)

            with mock.patch.object(review_pack, "_write_answers", side_effect=write):
                rep = self.append(jp)
            self.assertEqual(len(calls), 2)
            self.assertTrue(rep["written"])
            self.assertEqual(ids(jp), ["r1", "r2", "r3", "r4"])

    def test_second_conflict_refuses_and_removes_own_temp_clips(self):
        with files("sqlite", qs=first()) as (jp, _md, version):
            import storage
            with mock.patch.object(review_pack, "_write_answers", side_effect=storage.Conflict("x")):
                with self.assertRaises(core.CoreError) as caught:
                    self.append(jp)
            self.assertIn("повторите", str(caught.exception))
            self.assertEqual(parts(jp), [])
            self.assertEqual(ids(jp), ["r1", "r2"])
            self.assertFalse([p for p in (review_pack.pack_dir(jp) / "clips").rglob("r3.*")], "клип остался")

    def test_clip_is_cut_outside_the_answers_lock(self):
        depth, seen = [0], []
        real_lock = review_pack._answers_lock
        import contextlib

        @contextlib.contextmanager
        def lock(path):
            with real_lock(path):
                depth[0] += 1
                try:
                    yield
                finally:
                    depth[0] -= 1

        with files("sqlite", qs=first()) as (jp, _md, _version):
            with mock.patch.object(review_pack, "_answers_lock", side_effect=lock), \
                    mock.patch.object(review_pack, "cut_clip",
                                      side_effect=lambda _s, t1, _t2, _o: seen.append(depth[0]) or t1):
                self.append(jp)
        self.assertEqual(seen, [0, 0])


class Entry(Base):
    def test_rebuild_true_rebuilds_with_the_guard(self):
        with files("sqlite") as (jp, _md, version):
            self.press(jp, version, REPLACE, "r4")
            with self.assertRaises(core.CoreError) as caught:
                self.append(jp, six.questions(), rebuild=True)
            self.assertIn("№4 (r4)", str(caught.exception))

    def test_other_recording_pack_is_built_anew(self):
        with files("sqlite", qs=first()) as (jp, _md, version):
            store = review_pack.answers_state(jp)
            store["manifest"]["recording_id"] = "другая"
            review_pack._write_answers(jp, store)
            rep = self.append(jp)
            self.assertNotEqual(rep["version"], version)
            self.assertIn("другой записи", rep["other_recording"])
            self.assertEqual(ids(jp), ["r3", "r4"])

    def test_files_mode_refuses_without_rebuild(self):
        with files("legacy_files", qs=first()) as (jp, _md, version):
            with self.assertRaises(core.CoreError) as caught:
                self.append(jp)
            self.assertIn("только при хранении записей в базе", str(caught.exception))
            self.assertIn("rebuild=true", str(caught.exception))
            self.assertNotEqual(self.append(jp, rebuild=True)["version"], version)

    def test_no_pack_builds(self):
        with files("sqlite", qs=first()) as (jp, _md, _version):
            with mock.patch.object(review_pack, "manifest", return_value=None), \
                    mock.patch.object(review_pack, "build", return_value={"built": 1}) as build:
                self.assertEqual(self.append(jp), {"built": 1})
            build.assert_called_once()


if __name__ == "__main__":
    unittest.main()
