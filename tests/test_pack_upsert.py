"""РАР-6.1, PR 13а: карточка в том же пакете (замена и дописывание без пересборки).

Замена карточки не меняет имя пакета (version) и ответы на другие карточки; номер редакции карточки (rev)
и счётчик пакета (pack_rev) растут; ответ на прежнюю редакцию отклоняется «вопрос обновлён». На этом же
механизме PR 17 дописывает порцию карточек. Все данные синтетические.
"""
import atexit
import contextlib
import copy
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock
from urllib.parse import quote, urlsplit

_TMP = tempfile.mkdtemp(prefix="rt-upsert13a-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import core  # noqa: E402
import recordings  # noqa: E402
import review_pack  # noqa: E402
import settings  # noqa: E402
import transcript  # noqa: E402

BACKENDS = ("legacy_files", "sqlite")
AS_IS = {"label": "Оставить как есть", "decision": {}}
FIRST = ("вот", "дальше", "та")


def result(n=3):
    """Одна реплика из n слов; у каждого слова своё место разбора r<k> с word_ids."""
    words = [{"word_id": f"w{k}", "w": FIRST[k - 1] if k <= len(FIRST) else f"слово{k}",
              "s": 0.5 * k, "e": 0.5 * k + 0.2} for k in range(1, n + 1)]
    review = [{"id": f"r{k}", "status": "open", "t": w["s"], "t1": w["s"], "t2": w["s"], "speaker": "Анна",
               "reasons": ["conf"], "text": w["w"], "word_ids": [w["word_id"]]} for k, w in enumerate(words, 1)]
    return {"file": "synthetic.wav", "duration": "00:00:30", "speakers": ["Анна"], "speaker_labels": ["speaker-1"],
            "timing": {"total_sec": 30}, "group": "demo", "quality": {"asr_incomplete_chunks": 0},
            "vocab_suggestions": [], "review_done": False, "edits": [],
            "segments": [{"start": words[0]["s"], "end": words[-1]["e"], "label": "speaker-1", "speaker": "Анна",
                          "text": " ".join(w["w"] for w in words), "words": words}],
            "review": review}


def question(rid, replace, ask="Проверьте"):
    return {"id": rid, "question": ask, "recommended": 0,
            "options": [AS_IS, {"label": f"Заменить на «{replace}»", "decision": {"text": replace}}]}


def fake_clip(_src, t1, _t2, out):
    """Клип без ffmpeg: в файле — его имя и время, чтобы видеть, что он не перезаписан."""
    Path(out).write_text(f"{Path(out).name}@{t1}", encoding="utf-8")
    return t1


@contextlib.contextmanager
def files(backend, data=None, qs=None):
    """Пара файлов стенограммы в выбранном режиме хранения и собранный пакет: (jp, md, версия)."""
    with tempfile.TemporaryDirectory(dir=_TMP) as temp:
        root = Path(temp)
        db, audio = root / "state.sqlite3", root / "sample.wav"
        audio.write_bytes(b"synthetic")
        data = copy.deepcopy(data or result())
        data["file"] = str(audio)
        layout = recordings.document_layout(audio)
        layout["root"].mkdir()
        store = None
        with mock.patch.object(settings.paths, "STATE_DB", db), \
                mock.patch.object(review_pack, "cut_clip", side_effect=fake_clip):
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
                qs = [question("r1", "вон"), question("r3", "тот")] if qs is None else qs
                pack = review_pack.build(jp, copy.deepcopy(qs))
                yield jp, md, pack["version"]
            finally:
                if store:
                    store.close()


class Case(unittest.TestCase):
    _ts = 0

    def save(self, jp, version, aid, **row):
        Case._ts += 1
        answer = {"id": aid, "version": version, "ts": f"2026-11-25T10:{Case._ts // 60:02d}:{Case._ts % 60:02d}",
                  **row}
        return review_pack.add_answers(jp, [answer], "local", trusted_actor="human")

    @staticmethod
    def item(jp, aid):
        return next(i for i in review_pack.manifest(jp)["items"] if i["id"] == aid)

    @staticmethod
    def texts(jp):
        return {w["word_id"]: w["w"] for s in transcript.load(jp)["segments"] for w in s["words"]}

    @staticmethod
    def outcomes(rep):
        return {row["id"]: row["outcome"] for row in rep.get("outcomes") or []}

    def writes(self):
        """Счётчик записей состояния пакета (одна на вызов upsert_cards)."""
        return mock.patch.object(review_pack, "_write_answers", wraps=review_pack._write_answers)


class ReplaceTests(Case):
    def test_replace_keeps_version_and_the_other_answer_and_both_apply(self):
        """Тест 1 (прототип 22.11): версия прежняя, rev 2, pack_rev + 1; ответ на другую карточку цел."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.save(jp, version, "r3", option=1)
                self.save(jp, version, "r1", option=0)
                rep = review_pack.upsert_cards(jp, [question("r1", "вам", "Новый вопрос")], why="тест")
                self.assertTrue(rep["written"])
                self.assertEqual(rep["replaced"], [{"id": "r1", "rev": 2, "clip": True}])
                man = review_pack.manifest(jp)
                self.assertEqual((man["version"], man["pack_rev"]), (version, 1))
                self.assertEqual((self.item(jp, "r1")["rev"], self.item(jp, "r1")["question"]), (2, "Новый вопрос"))
                answers = review_pack.answers_state(jp)["answers"]
                self.assertEqual(answers["r3"]["option"], 1)
                self.assertNotIn("r1", answers, "ответ прежней редакции не должен лечь на новый вопрос")
                self.assertEqual(self.save(jp, version, "r1", rev=2, option=1)["accepted"][0]["id"], "r1")
                out = self.outcomes(core.apply_answers_files(jp, md))
                self.assertEqual((out["r1"], out["r3"]), ("applied", "applied"))
                self.assertEqual((self.texts(jp)["w1"], self.texts(jp)["w3"]), ("вам", "тот"))

    def test_answer_to_previous_revision_is_refused(self):
        """Тест 2: ответ на прежнюю редакцию — «вопрос обновлён» в rejected с rev; без rev — только к rev 1."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, _md, version):
                review_pack.upsert_cards(jp, [question("r1", "вам")], why="тест")
                old = self.save(jp, version, "r1", rev=1, option=1)
                self.assertEqual(old["rejected"], [{"id": "r1", "why": review_pack.REV_STALE, "version": version,
                                                    "rev": 1}])
                self.assertEqual(self.save(jp, version, "r1", option=1)["rejected"][0]["why"], review_pack.REV_STALE)
                self.assertEqual(self.save(jp, version, "r1", rev=3, option=1)["rejected"][0]["why"],
                                 review_pack.REV_UNKNOWN)
                stored = review_pack.answers_state(jp)
                self.assertNotIn("r1", stored["answers"])
                self.assertEqual([(r["id"], r.get("rev")) for r in stored["rejected"]],
                                 [("r1", 1), ("r1", None), ("r1", 3)])
                # старая страница без rev: карточка не менялась — ответ принят
                self.assertEqual(self.save(jp, version, "r3", option=1)["accepted"][0]["id"], "r3")

    def test_applied_answer_moves_to_history_and_new_answer_is_not_held(self):
        """Тест 3: прежний ответ — в history (applied), снимка отмены и отметки применения нет; новый ответ
        сохраняется как обычный, а не «ждёт отмены прежнего». Применение поверх прежней правки человека —
        правило двух человеческих правок (13б), здесь не проверяется."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.save(jp, version, "r1", option=1)
                self.assertEqual(self.outcomes(core.apply_answers_files(jp, md))["r1"], "applied")
                self.assertIn("r1", review_pack.answers_state(jp).get("undo_snapshots") or {})
                review_pack.upsert_cards(jp, [question("r1", "вам", "Ещё раз")], why="слова изменились")
                item = self.item(jp, "r1")
                [past] = item["history"]
                self.assertEqual((past["rev"], past["answer"]["option"], past["applied"], past["why"]),
                                 (1, 1, True, "слова изменились"))
                self.assertNotIn("applied_answer", past["answer"])
                state = review_pack.answers_state(jp)
                self.assertNotIn("r1", state.get("undo_snapshots") or {})
                self.assertNotIn("r1", state.get("applied_answers") or {})
                self.assertNotIn("r1", review_pack.answer_states_of(state, review_pack.manifest(jp)))
                self.save(jp, version, "r1", rev=2, option=1)
                self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "saved")
                decisions, rep = review_pack.decisions(jp)
                self.assertEqual([(d["id"], d.get("text")) for d in decisions], [("r1", "вам")])
                self.assertEqual(rep["pending_new"], [])

    def test_old_revision_answer_left_by_a_crash_is_never_applied(self):
        """Условие архитектора (1): ответ старой редакции, оставшийся в ответах (сбой между записями), не
        применяется молча и не показывается как ответ на новый вопрос; конфликты прежней редакции — в history."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                self.save(jp, version, "r1", option=1)
                stale = copy.deepcopy(review_pack.answers_state(jp)["answers"]["r1"])
                self.save(jp, version, "r3", option=0)
                review_pack.add_answers(jp, [{"id": "r3", "version": version, "option": 1,
                                              "ts": "2026-11-25T11:00:00"}], "cloud", trusted_actor="human")
                self.assertEqual([c["id"] for c in review_pack.answers_state(jp)["conflicts"]], ["r3"])
                review_pack.upsert_cards(jp, [question("r1", "вам"), question("r3", "тут")], why="тест")
                state = review_pack.answers_state(jp)
                self.assertEqual(state["conflicts"], [])
                self.assertEqual([c["id"] for c in self.item(jp, "r3")["history"][0]["conflicts"]], ["r3"])
                self.assertEqual(self.item(jp, "r1")["history"][0]["answer"]["option"], 1)
                state["answers"]["r1"] = stale                    # как после обрыва между manifest и ответами
                free = {k: v for k, v in stale.items() if k != "option"}
                state["answers"]["r3"] = dict(free, id="r3", kind="free", text="тута")   # свой текст старой редакции
                review_pack._write_answers(jp, state)
                decisions, rep = review_pack.decisions(jp)
                self.assertEqual(decisions, [])
                self.assertIn("r1", rep["unanswered"])
                self.assertEqual(review_pack.free_answers(jp), {})
                self.assertNotIn("r1", review_pack.answer_states(jp))
                self.assertNotIn("r1", json.loads((review_pack.pack_dir(jp) / review_pack.cloud_bundle(jp)["files"]
                                                   ["manifest.json"]).read_text(encoding="utf-8"))["answers"])
                core.apply_answers_files(jp, md)
                self.assertEqual((self.texts(jp)["w1"], self.texts(jp)["w3"]), ("вот", "та"),
                                 "ответ старой редакции применился")

    def test_late_card_is_checked_against_its_own_transcript_revision(self):
        """Условие архитектора (б), (в): карточка, собранная позже пакета, сверяется при «Применить» с ревизией,
        по которой собрана; настоящая правка её слов ловится; своя ревизия старше пакетной не подменяет её."""
        def payload(text):
            return {"segments": [{"speaker": "Анна", "words": [{"word_id": "w1", "w": text}]}]}

        def conflicts(own, current, source, sources=None, answer_rev=2):
            man = {"source_revision": 1, "items": [{"id": "r1", "rev": 2, "source_revision": own, "word_ids": ["w1"]}]}
            answer = {"id": "r1", "kind": "option", "option": 0, **({"rev": answer_rev} if answer_rev else {})}
            return core._answer_revision_conflicts(payload(current), man, {"answers": {"r1": answer}},
                                                   payload(source), sqlite=True, sources=sources)
        # 1. заменённая карточка новее пакета: без ложного «слова изменились»
        self.assertEqual(conflicts(2, "тэ", "та", {2: payload("тэ")}), {})
        # 2. настоящая правка слов после замены ловится
        self.assertIn("r1", conflicts(2, "то", "та", {2: payload("тэ")}))
        self.assertIn("r1", conflicts(2, "тэ", "тэ"), "нет ревизии карточки — сверять не с чем")
        # 3. своя ревизия старше пакетной — ошибка данных: сверка по ревизии пакета, как раньше
        self.assertIn("r1", conflicts(0, "тэ", "та", {0: payload("тэ")}))
        self.assertEqual(conflicts(0, "та", "та", {0: payload("тэ")}), {})
        # ответ прежней редакции не сверяется вовсе: он не применяется
        self.assertEqual(conflicts(2, "тэ", "та", answer_rev=None), {})

    def test_listened_button_is_checked_against_the_card_revision(self):
        """Условие архитектора (б): «прослушал» на месте «модели расходятся» сверяется с ревизией карточки."""
        data = result()
        data["review"][1].update(reasons=["asr"], asr_problem={"model": "whisper", "state": "words_differ",
                                                                "words_from": []})
        with files("sqlite", data, [question("r1", "вон"), question("r2", "дальнейшем")]) as (jp, md, version):
            self.save(jp, version, "r1", option=1)
            core.apply_answers_files(jp, md)                  # ревизия стенограммы выросла
            self.save(jp, version, "r2", option=0)            # «Я прослушал: слова в стенограмме верны»

            def card_revision(revision):
                state = review_pack.answers_state(jp)
                next(i for i in state["manifest"]["items"] if i["id"] == "r2")["source_revision"] = revision
                review_pack._write_answers(jp, state)
                return review_pack.decisions(jp)
            pack_revision = review_pack.manifest(jp)["source_revision"]
            _out, rep = review_pack.decisions(jp)
            self.assertIn("r2", rep["source_conflict_ids"])
            current = transcript.load(jp)["source_revision"]
            out, rep = card_revision(current)
            self.assertNotIn("r2", rep["source_conflict_ids"])
            self.assertEqual([d["id"] for d in out], ["r2"])
            _out, rep = card_revision(pack_revision - 1)
            self.assertIn("r2", rep["source_conflict_ids"], "старшая ревизия карточки подменила ревизию пакета")

    def test_repeat_of_the_same_card_writes_nothing(self):
        """Тест 5: тот же вопрос — ничего не пишется, pack_rev прежний, ревизия sqlite прежняя."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, _md, _version):
                before = review_pack.answers_state(jp).get("_expected_versions")
                with self.writes() as write:
                    rep = review_pack.upsert_cards(jp, [question("r1", "вон")], why="тест")
                self.assertEqual((rep["written"], rep["unchanged"], rep["pack_rev"]), (False, ["r1"], 0))
                write.assert_not_called()
                self.assertEqual(review_pack.answers_state(jp).get("_expected_versions"), before)
                self.assertEqual(self.item(jp, "r1")["rev"], 1)


class AppendTests(Case):
    def test_append_fifty_cards_is_one_state_write_and_keeps_answers(self):
        """Тест 4 (задел PR 17): 50 новых карточек одним вызовом — одна запись состояния; ответы прежних целы."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), \
                    files(backend, result(52), [question("r1", "вон"), question("r2", "дальше-то")]) as (jp, _md, v):
                self.save(jp, v, "r1", option=1)
                with self.writes() as write:
                    rep = review_pack.upsert_cards(
                        jp, [question(f"r{k}", f"новое{k}") for k in range(3, 53)], why="следующая порция")
                self.assertEqual(write.call_count, 1)
                self.assertEqual((len(rep["added"]), rep["pack_rev"]), (50, 1))
                man = review_pack.manifest(jp)
                self.assertEqual([i["id"] for i in man["items"]], [f"r{k}" for k in range(1, 53)])
                self.assertEqual({i.get("rev") for i in man["items"]}, {1})
                self.assertEqual(review_pack.answers_state(jp)["answers"]["r1"]["option"], 1)
                states = review_pack.answer_states(jp)
                self.assertEqual(states["r1"]["card"], 1)
                # новая карточка первой редакции: ответ без rev (облако, import_answers) принимается
                self.assertEqual(self.save(jp, v, "r52", option=0)["accepted"][0]["id"], "r52")


class ClipTests(Case):
    def test_no_audio_or_failed_clip_gives_card_without_sound(self):
        """Тест 6: нет аудио или клип не нарезался → clip: null, ответ принимается; build без аудио — как раньше."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, _md, version):
                with mock.patch.object(review_pack, "cut_clip",
                                       side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
                    rep = review_pack.upsert_cards(jp, [question("r2", "дальнейшем")], why="тест")
                self.assertEqual(rep["added"], [{"id": "r2", "rev": 1, "clip": False}])
                Path(transcript.load(jp)["file"]).unlink()
                review_pack.upsert_cards(jp, [question("r1", "вам")], why="тест")
                for aid in ("r1", "r2"):
                    self.assertIsNone(self.item(jp, aid)["clip"])
                self.assertFalse(list(review_pack.pack_dir(jp).rglob("r2.r1*")), "остался обрывок клипа")
                self.assertEqual(self.save(jp, version, "r1", rev=2, option=1)["accepted"][0]["id"], "r1")
                bundle = review_pack.cloud_bundle(jp)
                self.assertNotIn(None, bundle["files"])
                with self.assertRaises(RuntimeError):
                    review_pack.build(jp, [question("r1", "вон")])

    def test_new_revision_clip_does_not_overwrite_previous(self):
        """Тест 7: клип новой редакции — отдельный файл; архивный manifest файлового режима читается."""
        with files("legacy_files") as (jp, _md, _version):
            pd = review_pack.pack_dir(jp)
            first = self.item(jp, "r1")["clip"]
            review_pack.upsert_cards(jp, [question("r1", "вам")], why="тест")
            second = self.item(jp, "r1")["clip"]
            self.assertEqual((first, second), ("clips/r1.mp3", "clips/r1.r2.mp3"))
            self.assertEqual(self.item(jp, "r1")["history"][0]["clip"], first)
            self.assertEqual((pd / second).read_text(encoding="utf-8"), "r1.r2.mp3@0.5")
            # пересборка (новая version): архив прежнего manifest и его клипы на месте
            old_version = review_pack.manifest(jp)["version"]
            review_pack.build(jp, [question("r1", "вон", "Другая сборка"), question("r3", "тот")])
            archived = review_pack._read(pd / f"manifest.{old_version}.json", None)
            self.assertEqual(next(i for i in archived["items"] if i["id"] == "r1")["clip"], second)
            self.assertTrue((pd / second).exists())
            # та же редакция в новой сборке: имя занято архивом — прежний файл не затирается
            review_pack.upsert_cards(jp, [question("r1", "вам", "Снова")], why="тест")
            third = self.item(jp, "r1")["clip"]
            self.assertNotEqual(third, second)
            self.assertTrue(third.startswith("clips/r1.r2."))
            self.assertEqual((pd / second).read_text(encoding="utf-8"), "r1.r2.mp3@0.5")

    def test_sqlite_clip_goes_to_the_pack_clip_folder(self):
        with files("sqlite") as (jp, _md, _version):
            man = review_pack.manifest(jp)
            self.assertTrue(man["clips_dir"].startswith("clips/"))
            review_pack.upsert_cards(jp, [question("r2", "дальнейшем")], why="тест")
            self.assertEqual(Path(self.item(jp, "r2")["clip"]).parent.as_posix(), man["clips_dir"])


class CloudTests(Case):
    def test_cloud_bundle_carries_no_local_paths(self):
        """Находка П-13а-1: путь к голосовому файлу на этом компьютере (поле ответа audio) не уходит в облако —
        ни из ответов, ни из applied_answers, ни из ответов в истории карточки; других путей ноутбука тоже нет."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                user = "secret-user-13a"
                voice = str(Path(_TMP) / "Users" / user / "voice" / "r1.webm")
                self.save(jp, version, "r1", kind="free", text="вон", audio=voice, heard=True)
                self.assertEqual(self.outcomes(core.apply_answers_files(jp, md))["r1"], "applied")
                review_pack.upsert_cards(jp, [question("r1", "вам", "Новый вопрос")], why="тест")
                self.assertEqual(self.item(jp, "r1")["history"][0]["answer"]["audio"], voice)   # на ноутбуке цел
                self.save(jp, version, "r3", kind="free", text="тот", audio=voice, heard=True)
                state = review_pack.answers_state(jp)
                # подпись применённого ответа в виде словаря с путём (на случай прежнего формата)
                state.setdefault("applied_answers", {})["r3"] = {"kind": "free", "text": "тот", "audio": voice}
                review_pack._write_answers(jp, state)
                bundle = review_pack.cloud_bundle(jp)
                raw = (review_pack.pack_dir(jp) / bundle["files"]["manifest.json"]).read_text(encoding="utf-8")
                for marker in (user, Path(_TMP).name):
                    self.assertNotIn(marker, raw, "в облачный пакет попал путь ноутбука")
                cloud = json.loads(raw)
                self.assertEqual((cloud["answers"]["r3"]["text"], cloud["answers"]["r3"]["heard"]), ("тот", True))
                self.assertNotIn("audio", cloud["answers"]["r3"])
                self.assertEqual(cloud["applied_answers"]["r3"], {"kind": "free", "text": "тот"})
                [past] = next(i for i in cloud["items"] if i["id"] == "r1")["history"]
                self.assertEqual(past["answer"]["text"], "вон")
                self.assertNotIn("audio", past["answer"])


class OldPackTests(Case):
    def test_pack_without_rev_is_read_without_rewrite(self):
        """Тест 8: пакет до 13а (без rev, pack_rev, clips_dir) читается как rev 1 / pack_rev 0 без перезаписи."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), files(backend) as (jp, md, version):
                def strip(man):
                    for key in ("pack_rev", "clips_dir"):
                        man.pop(key, None)
                    for it in man["items"]:
                        it.pop("rev", None)
                state = review_pack.answers_state(jp)
                strip(state["manifest"])
                review_pack._write_answers(jp, state)
                path = review_pack.pack_dir(jp) / "manifest.json"
                if backend == "legacy_files":
                    man = review_pack._read(path, {})
                    strip(man)
                    review_pack._write(path, man)
                    raw = path.read_bytes()
                self.assertEqual(self.save(jp, version, "r1", option=1)["accepted"][0]["id"], "r1")
                self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "saved")
                if backend == "legacy_files":
                    self.assertEqual(path.read_bytes(), raw, "чтение и ответ не должны переписывать manifest")
                self.assertNotIn("rev", review_pack.answers_state(jp)["answers"]["r1"])
                self.assertEqual(self.outcomes(core.apply_answers_files(jp, md))["r1"], "applied")
                rep = review_pack.upsert_cards(jp, [question("r3", "тут")], why="тест")
                self.assertEqual((rep["pack_rev"], self.item(jp, "r3")["rev"]), (1, 2))
                self.assertEqual(Path(self.item(jp, "r3")["clip"]).parent,
                                 Path(self.item(jp, "r1")["clip"]).parent)


class EventTests(Case):
    def test_interaction_event_checks_revision(self):
        with files("legacy_files") as (jp, _md, version):
            review_pack.upsert_cards(jp, [question("r1", "вам")], why="тест")
            for rev in (1, None):
                with self.subTest(rev=rev), self.assertRaisesRegex(ValueError, review_pack.REV_STALE):
                    review_pack.add_interaction_event(jp, "r1", version, "card_visible", "session-0001", rev)
            event = review_pack.add_interaction_event(jp, "r1", version, "card_visible", "session-0001", 2)
            self.assertEqual(event["event"]["rev"], 2)
            self.assertTrue(review_pack.add_interaction_event(jp, "r3", version, "card_visible", "session-0001")
                            ["stored"])


class ReviewPageTests(unittest.TestCase):
    """Тест 9: api/answers отдаёт pack_rev; api/voice, api/preview и api/event с устаревшим rev — отказ."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="review13a_", dir=_TMP)
        root = Path(cls.tmp.name)
        cls.pack = root / "demo.review"
        (cls.pack / "clips").mkdir(parents=True)
        (cls.pack / "clips" / "r1.r2.mp3").write_bytes(b"demo audio")
        option = [{"label": "да", "decision": {}}]
        (cls.pack / "manifest.json").write_text(json.dumps({
            "version": "v1", "pack_rev": 3, "audio": "demo.wav", "items": [
                {"id": "r1", "rev": 2, "t": "00:00:01", "clip": "clips/r1.r2.mp3", "text": "да", "t1": 0.0,
                 "t2": 0.0, "question": "Как правильно?", "options": option},
                {"id": "r2", "t": "00:00:02", "clip": None, "text": "нет", "t1": 1.0, "t2": 1.0,
                 "question": "Как правильно?", "options": option}]}), encoding="utf-8")
        status, data, temp = root / "status.json", root / "data", root / "temp"
        data.mkdir()
        temp.mkdir()
        env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=str(temp), TMP=str(temp),
                   PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        code = ("import sys, review_page; "
                "review_page.P.recognize_here=lambda files: {str(p): 'первый' for p in files}; "
                "sys.argv=['review_page.py', sys.argv[1], '--status', sys.argv[2], '--port', '19370']; "
                "review_page.main()")
        cls.proc = subprocess.Popen([sys.executable, "-B", "-c", code, str(root / "demo.transcript.json"),
                                     str(status)], cwd=str(PROJECT), env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if status.exists():
                try:
                    url = json.loads(status.read_text(encoding="utf-8")).get("url")
                    if url:
                        cls.url = urlsplit(url)
                        return
                except (OSError, ValueError):
                    pass
            if cls.proc.poll() is not None:
                break
            time.sleep(0.1)
        cls.tearDownClass()
        raise RuntimeError("тестовая страница не запустилась")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.tmp.cleanup()

    def request(self, path, method="GET", body=None, ctype="application/json"):
        conn = http.client.HTTPConnection("127.0.0.1", self.url.port, timeout=10)
        headers = {"Origin": f"http://127.0.0.1:{self.url.port}", "Content-Type": ctype} if method == "POST" else {}
        try:
            conn.request(method, self.url.path + path, body=body, headers=headers)
            response = conn.getresponse()
            return response.status, json.loads(response.read() or b"null")
        finally:
            conn.close()

    def post(self, path, obj):
        return self.request(path, "POST", json.dumps(obj).encode("utf-8"))

    def test_answers_report_pack_revision(self):
        status, body = self.request("api/answers")
        self.assertEqual((status, body["pack_version"], body["pack_rev"]), (200, "v1", 3))

    def test_stale_revision_is_refused_on_voice_preview_and_event(self):
        voice = self.pack / "voice"
        for rev in ("1", None):
            query = "api/voice?id=r1&version=v1" + (f"&rev={quote(rev)}" if rev else "")
            with self.subTest(route="voice", rev=rev):
                self.assertEqual(self.request(query, "POST", b"x", "audio/webm"), (400, {"error": "вопрос обновлён"}))
            with self.subTest(route="preview", rev=rev):
                obj = {"id": "r1", "version": "v1", "text": "да", **({"rev": int(rev)} if rev else {})}
                self.assertEqual(self.post("api/preview", obj), (400, {"error": "вопрос обновлён"}))
            with self.subTest(route="event", rev=rev):
                obj = {"id": "r1", "version": "v1", "type": "card_visible", "session": "session-0001",
                       **({"rev": int(rev)} if rev else {})}
                self.assertEqual(self.post("api/event", obj), (400, {"error": "вопрос обновлён"}))
        self.assertFalse(voice.exists() and any(voice.iterdir()), "голос записан до проверки редакции")
        status, body = self.request("api/voice?id=r1&version=v1&rev=2", "POST", b"x", "audio/webm")
        self.assertEqual((status, body["heard"]), (200, "первый"))
        self.assertEqual(self.request("api/voice?id=r2&version=v1", "POST", b"x", "audio/webm")[0], 200)


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class PageRedrawTests(unittest.TestCase):
    """Тест 10: при росте pack_rev страница перечитывает manifest и перерисовывает только изменённые карточки."""

    @staticmethod
    def part(html, start, end):
        return start + html.split(start, 1)[1].split(end, 1)[0]

    def run_node(self, body):
        html = (PROJECT / "cards.html").read_text(encoding="utf-8")
        script = html.split("<script>", 1)[1]
        marker = "  // ответы этой версии вопросов"
        prefix = script.split(marker)[0]
        layer = self.part(script, marker, "  async function cloudApi()")
        redraw = self.part(script, "  // Готовая карточка перерисовывается", "  function card(it) {")
        reload = self.part(script, "  function header() {", '\n\n  $("f-open").onclick')
        stubs = r'''
          global.location = {hostname: "127.0.0.1"};
          const el = {textContent: ""};
          global.document = {getElementById: () => el};
        '''
        source = stubs + prefix + layer + redraw + reload + body + "\n})();\n"
        result = subprocess.run([shutil.which("node"), "-e", source], capture_output=True, text=True,
                                encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_only_changed_cards_are_rebuilt_and_typed_text_survives(self):
        self.run_node(r'''
          let built = 0;
          const input = () => ({value: ""});
          var card = (it) => { built++; const inp = input(); return {it, inp, querySelector: () => inp}; };
          api = {voiceMode: null};
          man = {version: "v", pack_rev: 0, items: [{id: "r1"}, {id: "r2"}], answer_states: {}};
          answers = {};
          const one = cardFor(man.items[0]), two = cardFor(man.items[1]);
          one.inp.value = "набранный текст";
          if (cardFor(man.items[0]) !== one || cardFor(man.items[1]) !== two || built !== 2) throw Error("лишняя перерисовка");
          man = {...man, pack_rev: 1, items: [{id: "r1", rev: 2}, {id: "r2"}]};
          const fresh = cardFor(man.items[0]);
          if (fresh === one || fresh.inp.value !== "набранный текст") throw Error("текст не перешёл в новую редакцию");
          if (cardFor(man.items[1]) !== two || built !== 3) throw Error("неизменённая карточка перерисована");
          answers = {r2: {id: "r2", version: "v", kind: "skip"}};
          if (cardFor(man.items[1]) === two) throw Error("ответ не перерисовал карточку");
        ''')

    def test_load_rereads_manifest_when_pack_changes_and_drops_old_revision_answers(self):
        self.run_node(r'''
          const fresh = {version: "v", pack_rev: 1, items: [{id: "r1", rev: 2, options: []}, {id: "r2", options: []}]};
          const store = {pack_version: "v", pack_rev: 1, answer_states: {}, answers: {
            r1: {id: "r1", version: "v", rev: 1, kind: "skip", ts: "t1"},
            r2: {id: "r2", version: "v", kind: "skip", ts: "t2"}}};
          let manifests = 0;
          global.fetch = async url => { if (url === "manifest.json") manifests++;
            return {ok: true, json: async () => url === "manifest.json" ? fresh : store}; };
          man = {version: "v", pack_rev: 0, items: [{id: "r1", options: []}, {id: "r2", options: []}]};
          (async () => {
            const got = await localApi().load();
            if (manifests !== 1 || revOf(man.items[0]) !== 2) throw Error("manifest не перечитан");
            if (got.r1 || !got.r2) throw Error("ответ прежней редакции показан как ответ на новую: " + JSON.stringify(got));
            await localApi().load();
            if (manifests !== 1) throw Error("manifest перечитан без изменения пакета");
          })().catch(e => { console.error(e); process.exitCode = 1; });
        ''')


if __name__ == "__main__":
    unittest.main()
