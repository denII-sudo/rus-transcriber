"""Замена ответа, «Пропустить» и исход каждого ответа карточек — в обоих режимах хранения.

Синтетическая стенограмма из двух слов («тест», «слово») у выдуманного Глеба, две карточки.
Правило до появления полной замены: применённый ответ другим ответом не заменяется —
новый ответ сохраняется и ждёт, пока прежний не отменят. Пропуск и отрицание оставляют
применённый ответ. У каждого сохранённого ответа ровно один исход, и состояние, которое
показывает страница, совпадает с отчётом применения.
"""
import atexit
import contextlib
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
import http.client
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

_TMP = tempfile.mkdtemp(prefix="rt-outcomes-")
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


def result_data():
    return {
        "file": "synthetic.wav", "duration": "00:00:02", "speakers": ["Глеб", "Вера"],
        "speaker_labels": ["speaker-1"], "timing": {"total_sec": 4}, "group": "demo",
        "quality": {"asr_incomplete_chunks": 0}, "vocab_suggestions": [],
        "segments": [{"start": 0.2, "end": 1.0, "label": "speaker-1", "speaker": "Глеб", "text": "тест слово",
                      "words": [{"w": "тест", "s": 0.2, "e": 0.5, "word_id": "w1"},
                                {"w": "слово", "s": 0.7, "e": 1.0, "word_id": "w2"}]}],
        "review": [{"id": "r1", "status": "open", "text": "тест", "t": 0.2, "t1": 0.2, "t2": 0.5,
                    "speaker": "Глеб", "word_ids": ["w1"]},
                   {"id": "r2", "status": "open", "text": "слово", "t": 0.7, "t1": 0.7, "t2": 1.0,
                    "speaker": "Глеб", "word_ids": ["w2"]}],
        "review_done": False, "edits": [],
    }


QUESTIONS = [
    {"id": "r1", "question": "Проверьте", "recommended": 0, "options": [
        {"label": "Другое слово", "decision": {"text": "другое"}},
        {"label": "Третье слово", "decision": {"text": "третье"}}]},
    {"id": "r2", "question": "Проверьте", "recommended": 0, "options": [
        {"label": "Новое слово", "decision": {"text": "новое"}},
        {"label": "Иное слово", "decision": {"text": "иное"}}]},
]


@contextlib.contextmanager
def answer_files(backend):
    """Пара файлов стенограммы в выбранном режиме хранения и собранный пакет: (jp, md, версия)."""
    with tempfile.TemporaryDirectory(dir=_TMP) as temp:
        root = Path(temp)
        db, audio = root / "state.sqlite3", root / "sample.wav"
        audio.write_bytes(b"synthetic")
        data = result_data()
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
                pack = review_pack.build(jp, copy.deepcopy(QUESTIONS))
                yield jp, md, pack["version"]
            finally:
                if store:
                    store.close()


class AnswerOutcomeCase(unittest.TestCase):
    _ts = 0

    def save(self, jp, version, aid="r1", **row):
        AnswerOutcomeCase._ts += 1          # у каждого ответа своё, более позднее время
        answer = {"id": aid, "version": version, "ts": f"2026-11-23T10:{AnswerOutcomeCase._ts // 60:02d}:"
                  f"{AnswerOutcomeCase._ts % 60:02d}", **row}
        return review_pack.add_answers(jp, [answer], "local", trusted_actor="human")

    @staticmethod
    def words(jp):
        return [w["w"] for s in transcript.load(jp)["segments"] for w in s["words"]]

    @staticmethod
    def pack_state(jp):
        return review_pack.answers_state(jp)

    def apply(self, jp, md, repeat=True, rewrite=False):
        """Применить ответы и проверить общие правила исходов; повторный запуск ничего не меняет.
        repeat=False — только для «Оставить человеку»: его повторы — задача следующей части.
        rewrite=True — различие режимов: в legacy_files ответ с конфликтом пробуется снова, и файл
        стенограммы перезаписывается с тем же содержимым и новым номером source_revision."""
        rep = core.apply_answers_files(jp, md)
        self.check_outcomes(jp, rep)
        if not repeat:
            return rep
        transcript_before = transcript.load(jp)
        state_before = self.pack_state(jp)
        again = core.apply_answers_files(jp, md)
        self.check_outcomes(jp, again)
        transcript_after = transcript.load(jp)
        if rewrite:
            self.assertEqual(transcript_after.pop("source_revision"), transcript_before.pop("source_revision") + 1)
        self.assertEqual(transcript_after, transcript_before, "повторный запуск изменил стенограмму")
        self.assertEqual(self.pack_state(jp), state_before,
                         "повторный запуск дописал события, ревизии или отметки")
        self.assertEqual(self.outcomes(again), self.outcomes(rep))
        self.assertFalse(again.get("undo_conflicts"), "повторный запуск снова пытается отменить ответ")
        return rep

    @staticmethod
    def outcomes(rep):
        return {row["id"]: row["outcome"] for row in rep.get("outcomes") or []}

    def check_outcomes(self, jp, rep):
        rows = rep.get("outcomes")
        self.assertIsInstance(rows, list, "в отчёте нет карты исходов")
        ids = [row["id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)), "у ответа больше одного исхода")
        stored = set((self.pack_state(jp).get("answers") or {}))
        self.assertEqual(set(ids), stored, "не у каждого сохранённого ответа есть исход")
        for row in rows:
            self.assertIn(row["outcome"], review_pack.OUTCOME_REASONS)
            self.assertEqual(row["why"], review_pack.OUTCOME_REASONS[row["outcome"]])
        # То, что видит человек на странице, совпадает с отчётом применения.
        view = review_pack.answer_states(jp)
        self.assertEqual({aid: v["state"] for aid, v in view.items()}, self.outcomes(rep))


class ReplaceAppliedAnswer(AnswerOutcomeCase):
    def first(self, jp, md, version, first):
        self.save(jp, version, **first)
        rep = self.apply(jp, md)
        self.assertEqual(self.outcomes(rep)["r1"], "applied")
        return self.words(jp)

    def test_new_answer_over_applied_waits_and_keeps_previous(self):
        cases = {
            "кнопка → кнопка": ({"option": 0}, {"option": 1}),
            "кнопка → текст": ({"option": 0}, {"text": "пятое"}),
            "текст → кнопка": ({"text": "пятое"}, {"option": 1}),
            "текст → текст": ({"text": "пятое"}, {"text": "шестое"}),
            "кнопка → рекомендация": ({"option": 1}, {"kind": "recommended"}),
            "кнопка → оставить человеку": ({"option": 0}, {"kind": "human"}),
        }
        for backend in BACKENDS:
            for name, (first, second) in cases.items():
                with self.subTest(backend=backend, case=name), answer_files(backend) as (jp, md, version):
                    applied_words = self.first(jp, md, version, first)
                    self.assertNotEqual(applied_words[0], "тест")
                    before = transcript.load(jp)
                    self.save(jp, version, **second)
                    rep = self.apply(jp, md)
                    self.assertEqual(self.outcomes(rep)["r1"], "pending_new")
                    after = transcript.load(jp)
                    self.assertEqual(after["segments"], before["segments"], "прежняя правка снята")
                    self.assertEqual(after["review"], before["review"])
                    self.assertFalse(rep["reverted"])
                    self.assertFalse(rep.get("changed"))
                    events = [e for e in self.pack_state(jp).get("application_events") or [] if e["id"] == "r1"]
                    self.assertEqual([e["outcome"] for e in events].count("applied"), 1, "ложное «применён»")
                    self.assertIn("Подготовить отмену", rep["outcomes"][0]["why"])

    def test_without_snapshot_new_answer_also_waits(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.first(jp, md, version, {"option": 0})
                state = self.pack_state(jp)
                state.pop("undo_snapshots", None)
                review_pack._write_answers(jp, state)
                self.save(jp, version, option=1)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "pending_new")
                self.assertEqual(self.words(jp), ["другое", "слово"])
                self.assertFalse(rep.get("undo_conflicts"))

    def test_revert_then_choose_again(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.first(jp, md, version, {"option": 0})
                self.save(jp, version, option=1)
                self.assertEqual(self.outcomes(self.apply(jp, md))["r1"], "pending_new")
                self.save(jp, version, kind="revert")
                self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "revert_pending")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "reverted")
                self.assertEqual(self.words(jp), ["тест", "слово"])
                self.save(jp, version, option=1)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "applied")
                self.assertEqual(self.words(jp), ["третье", "слово"])

    def test_applied_then_negative_then_correct_text(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.first(jp, md, version, {"option": 0})
                before = transcript.load(jp)
                self.save(jp, version, text="неверно")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "kept_previous")
                self.assertEqual(transcript.load(jp)["segments"], before["segments"])
                self.save(jp, version, text="пятое")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "pending_new")
                self.assertEqual(self.words(jp), ["другое", "слово"])
                self.save(jp, version, kind="revert")
                self.assertEqual(self.outcomes(self.apply(jp, md))["r1"], "reverted")
                self.save(jp, version, text="пятое")
                self.assertEqual(self.outcomes(self.apply(jp, md))["r1"], "applied")
                self.assertEqual(self.words(jp), ["пятое", "слово"])


class SkipAnswer(AnswerOutcomeCase):
    def test_skip_does_not_replace_applied_answer(self):
        for backend in BACKENDS:
            for first in ({"option": 0}, {"text": "пятое"}):
                with self.subTest(backend=backend, first=first), answer_files(backend) as (jp, md, version):
                    self.save(jp, version, **first)
                    self.apply(jp, md)
                    applied = self.words(jp)
                    stored = copy.deepcopy(self.pack_state(jp)["answers"])
                    rep = self.save(jp, version, kind="skip", text="пропущено")
                    self.assertFalse(rep["accepted"][0].get("stored", True))
                    self.assertTrue(rep["accepted"][0].get("note"))
                    self.assertEqual(self.pack_state(jp)["answers"], stored)
                    rep = self.apply(jp, md)
                    self.assertEqual(self.outcomes(rep)["r1"], "applied")
                    self.assertEqual(self.words(jp), applied)
                    self.assertFalse(rep["reverted"])

    def test_old_pack_with_stored_skip_over_applied(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.save(jp, version, option=0)
                self.apply(jp, md)
                state = self.pack_state(jp)
                state["answers"]["r1"] = dict(state["answers"]["r1"], kind="skip", text="пропущено",
                                              ts="2026-11-23T23:00:00")
                state["answers"]["r1"].pop("option", None)
                review_pack._write_answers(jp, state)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "kept_previous")
                self.assertEqual(self.words(jp), ["другое", "слово"])
                self.assertFalse(rep["reverted"])
                self.assertFalse(rep.get("undo_conflicts"))

    def test_skip_before_apply(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.save(jp, version, kind="skip", text="пропущено")
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "skipped")
                self.assertEqual(self.words(jp), ["тест", "слово"])


class EveryAnswerHasOutcome(AnswerOutcomeCase):
    def test_several_answers_in_one_run(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.save(jp, version, option=0)
                self.apply(jp, md)
                self.save(jp, version, option=1)
                self.save(jp, version, aid="r2", option=1)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep), {"r1": "pending_new", "r2": "applied"})
                self.assertEqual(self.words(jp), ["другое", "иное"])

    def test_each_kind_of_answer(self):
        kinds = {"кнопка": ({"option": 0}, "applied"), "рекомендация": ({"kind": "recommended"}, "applied"),
                 "свой текст": ({"text": "пятое"}, "applied"), "отрицание": ({"text": "неверно"}, "negative"),
                 "оставить человеку": ({"kind": "human"}, "left_for_human"),
                 "пропуск": ({"kind": "skip", "text": "пропущено"}, "skipped"),
                 "отмена без применённого": ({"kind": "revert"}, "skipped")}
        for backend in BACKENDS:
            for name, (row, expected) in kinds.items():
                with self.subTest(backend=backend, kind=name), answer_files(backend) as (jp, md, version):
                    self.save(jp, version, **row)
                    self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "saved"
                                     if name not in ("пропуск", "отмена без применённого") else "skipped")
                    rep = self.apply(jp, md, repeat=name != "оставить человеку")
                    self.assertEqual(self.outcomes(rep)["r1"], expected)

    def test_failed_application_has_full_map(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                transcript.edit_span(jp, md, 0.2, 0.5, "ручной", by="human")
                self.save(jp, version, option=0)
                self.save(jp, version, aid="r2", option=0)
                rep = self.apply(jp, md, rewrite=backend == "legacy_files")
                self.assertEqual(self.outcomes(rep), {"r1": "apply_conflict", "r2": "applied"})
                self.assertEqual(self.words(jp), ["ручной", "новое"])

    def test_report_numbers_cards(self):
        with answer_files("legacy_files") as (jp, md, version):
            self.save(jp, version, aid="r2", option=0)
            rep = self.apply(jp, md)
            self.assertEqual(rep["outcomes"], [{"id": "r2", "card": 2, "outcome": "applied",
                                                "why": review_pack.OUTCOME_REASONS["applied"]}])


class PageErrors(unittest.TestCase):
    """Страница называет причину ошибки по-русски и не показывает текст исключения."""

    def test_safe_reason(self):
        import review_page
        import storage
        secret = r"C:\Users\someone\secret.json: чужой ответ «Анна»"
        cases = [
            (RuntimeError(secret), None),
            (KeyError(secret), None),
            (ValueError("версия пакета изменилась"), "версия пакета изменилась"),
            (core.CoreError("Ответы сохранены в БД; обновление документа не завершено"),
             "Ответы сохранены в БД; обновление документа не завершено"),
            (core.CoreError(f"Ответы сохранены, но конспекты не обновлены: {secret}"),
             "Ответы сохранены, но конспекты не обновлены"),
            (storage.Conflict("recording changed while applying answers"), None),
            (json.JSONDecodeError("x", secret, 0), None),
        ]
        for ex, expected in cases:
            with self.subTest(ex=type(ex).__name__):
                message = review_page.safe_error(ex)
                self.assertNotIn("secret", message)
                self.assertNotIn("Анна", message)
                self.assertNotIn("recording changed", message)
                if expected:
                    self.assertTrue(message.startswith(expected), message)
                else:
                    self.assertIn("код", message)
        self.assertIn("изменилась", review_page.safe_error(storage.Conflict("x")))

    def test_http_apply_error_names_reason_without_exception_text(self):
        with tempfile.TemporaryDirectory(dir=_TMP) as temp:
            root = Path(temp)
            pack = root / "demo.review"
            (pack / "clips").mkdir(parents=True)
            (pack / "manifest.json").write_text(json.dumps({"version": "v1", "audio": "demo.wav", "items": [
                {"id": "r1", "t": "00:00:01", "clip": "clips/r1.mp3", "text": "да", "speaker": "Участник",
                 "t1": 0.0, "t2": 0.0, "question": "Как?", "options": [{"label": "да", "decision": {}}]}]}),
                encoding="utf-8")
            status = root / "status.json"
            data, tmp = root / "data", root / "temp"
            data.mkdir()
            tmp.mkdir()
            env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=str(tmp), TMP=str(tmp),
                       PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
            code = ("import sys, core, review_page\n"
                    "def boom(*a, **k):\n"
                    "    raise RuntimeError('C:/secret/path чужой ответ')\n"
                    "core.apply_answers_files = boom\n"
                    "review_page.P.add_answers = boom\n"
                    "sys.argv = ['review_page.py', sys.argv[1], '--status', sys.argv[2], '--markdown', sys.argv[3],"
                    " '--port', '19370']\n"
                    "review_page.main()\n")
            proc = subprocess.Popen([sys.executable, "-B", "-c", code, str(root / "demo.transcript.json"),
                                     str(status), str(root / "demo.transcript.md")], cwd=str(PROJECT), env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                url = None
                for _ in range(100):
                    if status.exists():
                        try:
                            url = json.loads(status.read_text(encoding="utf-8")).get("url")
                        except (OSError, ValueError):
                            url = None
                        if url:
                            break
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                self.assertTrue(url, "тестовая страница не запустилась")
                u = urlsplit(url)
                for route, body in (("api/apply", {"version": "v1"}),
                                    ("api/answer", {"id": "r1", "version": "v1", "option": 0})):
                    with self.subTest(route=route):
                        conn = http.client.HTTPConnection("127.0.0.1", u.port, timeout=10)
                        try:
                            conn.request("POST", u.path + route, body=json.dumps(body).encode(), headers={
                                "Origin": f"http://127.0.0.1:{u.port}", "Content-Type": "application/json"})
                            response = conn.getresponse()
                            text = response.read().decode("utf-8")
                        finally:
                            conn.close()
                        self.assertEqual(response.status, 400)
                        error = json.loads(text)["error"]
                        self.assertNotIn("secret", error)
                        self.assertNotIn("чужой", error)
                        self.assertIn("код", error)
                        self.assertNotEqual(error, "не удалось обработать запрос")
            finally:
                proc.terminate()
                proc.wait(timeout=10)


class PageStateFromPython(unittest.TestCase):
    """Страница не считает подпись применённого ответа сама: состояние приходит из Python."""

    def test_cards_page_has_no_own_signature(self):
        html = (PROJECT / "cards.html").read_text(encoding="utf-8")
        self.assertNotIn("const signature", html)
        self.assertNotIn("applied_answers[it.id]", html)
        self.assertIn("answer_states", html)

    def test_cloud_bundle_carries_states(self):
        with answer_files("legacy_files") as (jp, md, version):
            review_pack.add_answers(jp, [{"id": "r1", "version": version, "option": 0}], "local",
                                    trusted_actor="human")
            core.apply_answers_files(jp, md)
            bundle = review_pack.cloud_bundle(jp)
            man = json.loads((Path(bundle["base"]) / "cloud" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(man["answer_states"]["r1"]["state"], "applied")


def without_keep_request(answer):
    """Ответ, возвращённый по «Не отменять», без отметки запроса — сверяется с прежним ответом целиком."""
    assert set(answer["keep_request"]) == {"ts", "source"}, answer
    return {k: v for k, v in answer.items() if k != "keep_request"}


def page_script(host="127.0.0.1"):
    """Начало скрипта cards.html (до функций ответов) — для проверки логики страницы в Node.js.
    host — адрес страницы: не 127.0.0.1/localhost означает облачную страницу."""
    html = (PROJECT / "cards.html").read_text(encoding="utf-8")
    return f'global.location={{hostname:"{host}"}};\n' + \
        html.split('  // ответы этой версии вопросов')[0].split('<script>', 1)[1]


def page_with_answers(host="127.0.0.1"):
    """Начало скрипта и слой ответов страницы (pick, merge, прослойки) — без запуска."""
    html = (PROJECT / "cards.html").read_text(encoding="utf-8")
    marker = "  // ответы этой версии вопросов"
    return page_script(host) + marker + html.split(marker, 1)[1].split("  async function cloudApi()", 1)[0]


def run_page(test, body, host="127.0.0.1"):
    script = page_script(host) + body + "\n})();\n"
    result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, encoding="utf-8")
    test.assertEqual(result.returncode, 0, result.stderr)
    return result.stdout


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class PageLabels(unittest.TestCase):
    # Состояния, которые знает только страница: ответ из облака, которого ноутбук ещё не видел.
    PAGE_ONLY = {"cloud_saved", "cloud_keep"}

    def test_every_outcome_has_its_own_label(self):
        out = run_page(self, "console.log(JSON.stringify(Object.keys(STATUS)));")
        self.assertEqual(set(json.loads(out)), set(review_pack.OUTCOME_REASONS) | self.PAGE_ONLY)

    def test_failed_undo_keeps_buttons_and_says_previous_is_in_place(self):
        run_page(self, r'''
          const a={kind:"revert",ts:"t2",source:"local"}, one={id:"r1"};
          man = {answer_states:{r1:{state:"undo_conflict",ts:"t2",source:"local",why:"отмена не выполнена"}}};
          if (!holdsApplied(one,a) || !appliedCard(one,a)) throw Error("кнопок нет: " + stateOf(one,a));
          if (cardStatus(one,a) !== "отмена не выполнена — прежний ответ на месте") throw Error(cardStatus(one,a));
        ''')


class SameSecondAnswers(AnswerOutcomeCase):
    def test_new_answer_in_same_second_does_not_take_old_state(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                transcript.edit_span(jp, md, 0.2, 0.5, "ручной", by="human")
                ts = "2026-11-23T11:00:00"
                review_pack.add_answers(jp, [{"id": "r1", "version": version, "option": 0, "ts": ts}], "local",
                                        trusted_actor="human")
                core.apply_answers_files(jp, md)
                self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "apply_conflict")
                review_pack.add_answers(jp, [{"id": "r1", "version": version, "option": 1, "ts": ts}], "local",
                                        trusted_actor="human")
                self.assertEqual(review_pack.answers_state(jp)["answers"]["r1"]["option"], 1)
                self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "saved")


class ReportDetails(AnswerOutcomeCase):
    """Подробности прежнего отчёта (конфликт отмены, отмена недоступна, конфликт карточек) видны человеку."""

    def lines(self, jp, rep):
        if not shutil.which("node"):
            self.skipTest("для проверки JavaScript нужен Node.js")
        items = review_pack.manifest(jp)["items"]
        out = run_page(self, "console.log(JSON.stringify(reportLines(%s, %s)));" % (
            json.dumps(rep, ensure_ascii=False), json.dumps(items, ensure_ascii=False)))
        return json.loads(out)

    def detail(self, rep, aid):
        return next(row for row in rep["outcomes"] if row["id"] == aid).get("detail") or ""

    def test_failed_undo_names_reason(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.save(jp, version, option=0)
                self.apply(jp, md)
                transcript.edit_span(jp, md, 0.7, 1.0, "ручной", by="human")
                self.save(jp, version, kind="revert")
                rep = core.apply_answers_files(jp, md)
                self.check_outcomes(jp, rep)
                self.assertEqual(self.outcomes(rep)["r1"], "undo_conflict")
                self.assertTrue(self.detail(rep, "r1"))
                self.assertTrue(any("Карточка 1" in line and self.detail(rep, "r1") in line
                                    for line in self.lines(jp, rep)))

    def test_applied_without_snapshot_says_undo_unavailable(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.save(jp, version, option=0)
                huge = {"segments": "x" * (17 * 1024 * 1024)}
                with mock.patch.object(transcript, "answer_snapshot", return_value=huge):
                    rep = core.apply_answers_files(jp, md)
                self.check_outcomes(jp, rep)
                self.assertEqual(self.outcomes(rep)["r1"], "applied")
                self.assertIn("r1", rep["undo_unavailable"])
                self.assertIn("нет снимка", self.detail(rep, "r1"))
                self.assertTrue(any("Карточка 1" in line and "нет снимка" in line for line in self.lines(jp, rep)))

    def test_preflight_conflict_names_other_card(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                state = review_pack.answers_state(jp)
                if backend == "sqlite":
                    state["manifest"]["items"][1]["word_ids"] = ["w1"]
                    review_pack._write_answers(jp, state)
                else:
                    path = review_pack.pack_dir(jp) / "manifest.json"
                    man = review_pack._read(path, {})
                    man["items"][1]["word_ids"] = ["w1"]
                    review_pack._write(path, man)
                self.save(jp, version, option=0)
                self.save(jp, version, aid="r2", option=0)
                rep = core.apply_answers_files(jp, md)
                self.check_outcomes(jp, rep)
                self.assertEqual(self.outcomes(rep), {"r1": "blocked_preflight", "r2": "blocked_preflight"})
                self.assertIn("карточкой 2", self.detail(rep, "r1"))
                self.assertIn("карточкой 1", self.detail(rep, "r2"))
                lines = self.lines(jp, rep)
                self.assertTrue(any(line.startswith("Карточка 1") and "карточкой 2" in line for line in lines))


class PreparedRevert(AnswerOutcomeCase):
    """Отмена сохранена, но «Применить» ещё не нажато: прежний ответ на месте; передумать — «Не отменять»."""

    def test_prepared_revert_state_and_keep(self):
        for backend in BACKENDS:
            for first in ({"option": 0}, {"text": "пятое"}):
                with self.subTest(backend=backend, first=first), answer_files(backend) as (jp, md, version):
                    self.save(jp, version, **first)
                    self.apply(jp, md)
                    applied = transcript.load(jp)
                    self.save(jp, version, kind="revert")
                    state = review_pack.answer_states(jp)["r1"]
                    self.assertEqual(state["state"], "revert_pending")
                    self.assertIn("Применить", state["why"])
                    rep = self.save(jp, version, kind="keep")
                    self.assertFalse(rep["rejected"])
                    self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "applied")
                    rep = self.apply(jp, md)
                    self.assertEqual(self.outcomes(rep)["r1"], "applied")
                    self.assertFalse(rep["reverted"])
                    self.assertEqual(transcript.load(jp)["segments"], applied["segments"])

    def test_keep_without_applied_answer_is_rejected(self):
        with answer_files("legacy_files") as (jp, md, version):
            rep = self.save(jp, version, kind="keep")
            self.assertEqual(rep["rejected"][0]["id"], "r1")
            self.assertNotIn("r1", review_pack.answers_state(jp).get("answers") or {})

    @unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
    def test_page_prepared_revert_holds_applied(self):
        run_page(self, r'''
          const a={kind:"revert",ts:"t2",source:"local"}, one={id:"r1"};
          man = {answer_states:{r1:{state:"revert_pending",ts:"t2",source:"local",why:"отмена подготовлена"}}};
          if (!holdsApplied(one,a) || !appliedCard(one,a)) throw Error("варианты как у открытого места");
          if (!cardStatus(one,a).startsWith("отмена подготовлена")) throw Error(cardStatus(one,a));
        ''')


class KeepRestoresWholeAnswer(AnswerOutcomeCase):
    """«Не отменять» возвращает прежний применённый ответ целиком: время, источник, услышанный текст."""

    def applied_with_heard(self, jp, md, version):
        review_pack.add_answers(jp, [{"id": "r1", "version": version, "option": 0, "text": "другое слово",
                                      "heard": True, "ts": "2026-11-23T09:00:00"}], "local", trusted_actor="human")
        self.apply(jp, md)
        return copy.deepcopy(review_pack.answers_state(jp)["answers"]["r1"])

    def test_keep_after_revert_restores_whole_answer(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                original = self.applied_with_heard(jp, md, version)
                self.save(jp, version, kind="revert")
                self.save(jp, version, kind="keep")
                self.assertEqual(without_keep_request(review_pack.answers_state(jp)["answers"]["r1"]), original)
                self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "applied")

    def test_keep_after_pending_and_revert_restores_whole_answer(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                original = self.applied_with_heard(jp, md, version)
                self.save(jp, version, option=1)
                self.save(jp, version, kind="revert")
                self.save(jp, version, kind="keep")
                self.assertEqual(without_keep_request(review_pack.answers_state(jp)["answers"]["r1"]), original)

    def test_copy_is_one_level_and_does_not_change_application(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                original = self.applied_with_heard(jp, md, version)
                self.save(jp, version, option=1)
                self.save(jp, version, text="пятое")
                self.save(jp, version, kind="revert")
                stored = review_pack.answers_state(jp)["answers"]["r1"]
                self.assertEqual(stored["applied_answer"], original)
                self.assertNotIn("applied_answer", stored["applied_answer"])
                # С копией внутри ответ применяется так же, как без неё.
                with_copy = review_pack.decisions(jp), review_pack.answer_states(jp)
                state = review_pack.answers_state(jp)
                state["answers"]["r1"].pop("applied_answer")
                review_pack._write_answers(jp, state)
                self.assertEqual((review_pack.decisions(jp), review_pack.answer_states(jp)), with_copy)
                rep = self.apply(jp, md)
                self.assertEqual(self.outcomes(rep)["r1"], "reverted")
                self.assertNotIn("applied_answer", json.dumps(rep, ensure_ascii=False))

    def test_cloud_bundle_does_not_carry_copy(self):
        with answer_files("legacy_files") as (jp, md, version):
            self.applied_with_heard(jp, md, version)
            self.save(jp, version, kind="revert")
            self.assertIn("applied_answer", review_pack.answers_state(jp)["answers"]["r1"])
            bundle = review_pack.cloud_bundle(jp)
            text = (Path(bundle["base"]) / "cloud" / "manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("\"applied_answer\"", text)

    def test_keep_from_cloud_restores_whole_answer(self):
        # «Не отменять» на облачной странице приходит на ноутбук через import_answers (источник cloud).
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                original = self.applied_with_heard(jp, md, version)
                self.save(jp, version, kind="revert")
                review_pack.add_answers(jp, [{"id": "r1", "version": version, "kind": "keep",
                                              "ts": "2026-11-23T23:00:00"}], "cloud")
                self.assertEqual(without_keep_request(review_pack.answers_state(jp)["answers"]["r1"]), original)
                self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "applied")

    def test_old_pack_without_copy_restores_by_signature_and_says_so(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.applied_with_heard(jp, md, version)
                self.save(jp, version, kind="revert")
                state = review_pack.answers_state(jp)
                state["answers"]["r1"].pop("applied_answer", None)
                review_pack._write_answers(jp, state)
                self.save(jp, version, kind="keep")
                answer = review_pack.answers_state(jp)["answers"]["r1"]
                self.assertTrue(answer.get("restored"))
                self.assertEqual((answer["kind"], answer["option"]), ("option", 0))
                self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "applied")


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class RevertAnswerText(unittest.TestCase):
    """Строка ответа «отменить» не противоречит заголовку карточки."""

    def test_revert_text_follows_state(self):
        run_page(self, r'''
          const a={kind:"revert",ts:"t2",source:"local"}, one={id:"r1", options: []};
          const cases = {revert_pending: "отмена подготовлена", undo_conflict: "отмена не выполнена", reverted: "ответ отменён"};
          for (const [state, start] of Object.entries(cases)) {
            man = {answer_states:{r1:{state, ts:"t2", source:"local"}}};
            const text = answerText(a, one);
            if (!text.startsWith(start)) throw Error(state + ": " + text);
            if (state !== "reverted" && text.includes("ответ отменён")) throw Error(state + " противоречит: " + text);
          }
          const restored = {kind:"option", option:0, restored:true, ts:"t3", source:"local"};
          if (!answerMeta(restored).includes("возвращён применённый ответ") || answerMeta(restored).includes("·"))
            throw Error("время возврата выдано за время ответа: " + answerMeta(restored));
        ''')


class KeepRequests(AnswerOutcomeCase):
    """«Не отменять» проходит ту же проверку времени, что и другие ответы; отклонённый запрос записан коротко."""

    KEEP_FIELDS = {"id", "kind", "ts", "source", "why"}

    def applied(self, jp, md, version):
        self.save(jp, version, option=0)
        self.apply(jp, md)

    def keep(self, jp, version, ts, source="cloud"):
        return review_pack.add_answers(jp, [{"id": "r1", "version": version, "kind": "keep", "ts": ts}], source)

    def rejected_keeps(self, jp):
        return [r for r in review_pack.answers_state(jp).get("rejected") or []
                if r.get("id") == "r1" and r.get("kind") == "keep"]

    def test_old_keep_does_not_override_newer_revert(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.applied(jp, md, version)
                review_pack.add_answers(jp, [{"id": "r1", "version": version, "kind": "revert",
                                              "ts": "2026-11-23T10:10:00"}], "local", trusted_actor="human")
                rep = self.keep(jp, version, "2026-11-23T10:05:00")
                self.assertEqual([r["id"] for r in rep["rejected"]], ["r1"])
                self.assertIn("более поздний", rep["rejected"][0]["why"])
                self.assertEqual(review_pack.answers_state(jp)["answers"]["r1"]["kind"], "revert")
                self.assertEqual(review_pack.answer_states(jp)["r1"]["state"], "revert_pending")
                self.assertEqual(self.outcomes(self.apply(jp, md))["r1"], "reverted")

    def test_rejected_keep_is_one_short_record_per_place(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.applied(jp, md, version)
                self.save(jp, version, kind="revert")
                self.apply(jp, md)                      # отмена прошла: применённого ответа нет
                self.keep(jp, version, "2026-11-23T22:00:00")
                self.keep(jp, version, "2026-11-23T23:00:00")
                rows = self.rejected_keeps(jp)
                self.assertEqual(len(rows), 1, rows)
                self.assertEqual(set(rows[0]), self.KEEP_FIELDS)
                self.assertEqual((rows[0]["ts"], rows[0]["source"]), ("2026-11-23T23:00:00", "cloud"))


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class CloudKeepPath(AnswerOutcomeCase):
    """Настоящий ход: «Не отменять» в облаке → import_answers → cloud_bundle → облачная страница."""

    KEEP_TS = "2026-11-23T23:00:00"

    def page(self, jp, body):
        bundle = review_pack.cloud_bundle(jp)
        text = (Path(bundle["base"]) / "cloud" / "manifest.json").read_text(encoding="utf-8")
        keep = {"id": "r1", "version": json.loads(text)["version"], "kind": "keep", "ts": self.KEEP_TS,
                "source": "cloud"}
        script = page_with_answers("cards.example") + "\n  man = " + text + ";\n  live = {r1: " + \
            json.dumps(keep) + "};\n  const it = man.items.find(x => x.id === \"r1\"), a = merge().r1;\n" + \
            body + "\n})();\n"
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True,
                                encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        return text

    def keep(self, jp, version, ts=None):
        return review_pack.add_answers(jp, [{"id": "r1", "version": version, "kind": "keep",
                                              "ts": ts or self.KEEP_TS}], "cloud")

    def test_accepted_keep_shows_program_state(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                review_pack.add_answers(jp, [{"id": "r1", "version": version, "option": 0, "text": "услышанное",
                                              "heard": True, "ts": "2026-11-23T09:00:00"}], "local",
                                        trusted_actor="human")
                self.apply(jp, md)
                self.save(jp, version, kind="revert")
                self.keep(jp, version)
                text = self.page(jp, r'''
                  if (stateOf(it, a) !== "applied") throw Error("состояние: " + stateOf(it, a));
                  if (a.kind === "keep" || answerText(a, it).includes("отмена")) throw Error("ответ: " + answerText(a, it));
                ''')
                man = json.loads(text)
                self.assertEqual(man["answer_states"]["r1"]["keep"],
                                 {"ts": self.KEEP_TS, "source": "cloud", "accepted": True})
                for leaked in ('"applied_answer"', '"keep_request"'):
                    self.assertNotIn(leaked, text)

    def test_rejected_keep_after_applied_revert_says_why(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.save(jp, version, option=0)
                self.apply(jp, md)
                self.save(jp, version, kind="revert")
                self.apply(jp, md)
                self.keep(jp, version)
                text = self.page(jp, r'''
                  if (stateOf(it, a) !== "reverted") throw Error("состояние: " + stateOf(it, a));
                  if (!programNote(it, a).includes("не принят")) throw Error("причина: " + programNote(it, a));
                ''')
                keep = json.loads(text)["answer_states"]["r1"]["keep"]
                self.assertEqual(keep, {"ts": self.KEEP_TS, "source": "cloud", "accepted": False})

    def test_old_rejected_keep_has_no_reason_under_newer_answer(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), answer_files(backend) as (jp, md, version):
                self.save(jp, version, option=0)
                self.apply(jp, md)
                review_pack.add_answers(jp, [{"id": "r1", "version": version, "kind": "revert",
                                              "ts": "2026-11-23T10:10:00"}], "local", trusted_actor="human")
                self.keep(jp, version, "2026-11-23T10:05:00")        # старее отмены — отклонён
                self.assertNotIn("не принят", review_pack.answer_states(jp)["r1"].get("detail", ""))
                self.apply(jp, md)
                self.keep(jp, version, "2026-11-23T10:20:00")        # отмена прошла — отклонён
                self.assertIn("не принят", review_pack.answer_states(jp)["r1"].get("detail", ""))
                review_pack.add_answers(jp, [{"id": "r1", "version": version, "option": 1,
                                              "ts": "2026-11-23T10:30:00"}], "local", trusted_actor="human")
                self.assertNotIn("не принят", review_pack.answer_states(jp)["r1"].get("detail", ""))


@unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
class CloudKeep(unittest.TestCase):
    """«Не отменять» на облачной странице до переноса на ноутбук: применённый ответ на месте."""

    SETUP = r'''
          const one={id:"r1", options:[{label:"Первое"},{label:"Второе"}]};
          const revert={id:"r1", kind:"revert", ts:"t2", source:"local"};
          const keep={id:"r1", kind:"keep", ts:"t3", source:"cloud"};
          man = {answer_states:{r1:{state:"revert_pending", ts:"t2", source:"local"}}};
        '''

    def test_cloud_keep_holds_applied_answer(self):
        run_page(self, self.SETUP + r'''
          if (stateOf(one, keep) !== "cloud_keep") throw Error("состояние: " + stateOf(one, keep));
          if (!holdsApplied(one, keep) || !appliedCard(one, keep)) throw Error("варианты открыты, кнопок нет");
          const texts = [cardStatus(one, keep), answerText(keep, one)].join(" | ");
          if (texts.includes("keep") || texts.includes("ответ отменён")) throw Error("служебное слово или неправда: " + texts);
          if (!texts.includes("применённый ответ")) throw Error("не сказано, что применённый ответ на месте: " + texts);
          man.answer_states.r1.why = "отмена подготовлена"; man.answer_states.r1.detail = "уточнение";
          if (programNote(one, keep) !== "") throw Error("причина прежнего ответа: " + programNote(one, keep));
          const b = revertButton(one, keep);
          if (b.kind !== "revert" || b.label !== "Подготовить отмену") throw Error("«Не отменять» повторно: " + JSON.stringify(b));
          const p = revertButton(one, revert);
          if (p.kind !== "keep" || p.label !== "Не отменять") throw Error("подготовленная отмена: " + JSON.stringify(p));
        ''', host="cards.example")

    def test_local_page_has_no_cloud_keep(self):
        run_page(self, self.SETUP + r'''
          if (stateOf(one, keep) === "cloud_keep") throw Error("на ноутбуке страничного состояния нет");
        ''')


if __name__ == "__main__":
    unittest.main()
