"""Ч-1: `legacy_read.read` — чтение старых файлов записи без записи — против замороженного итога файлового «Применить».

Случаи — образцы старого формата (tests/legacy_samples, #224) и их детерминированные правки (`CASES`). Итог
файлового «Применить» (`core.apply_answers_files` на копии каждого случая) заморожен в
`tests/legacy_read_expected.json`; его sha256 — `EXPECTED_SHA256`. Пока файловый код есть, тест пересчитывает итог
и сверяет с замороженным; после удаления файлового кода сверка идёт только с замороженным.

Пересобрать замороженный итог (только пока файловый код есть, при изменении образцов или CASES):
    python -B tests/test_legacy_read.py --freeze     (с RUS_TRANSCRIBER_DATA, TEMP и TMP на песочницу)
и записать новый sha256 в EXPECTED_SHA256.

Что проверяется: чтение ничего не меняет и не создаёт; «внесёт» ⇔ файловое «Применить» внесло ответ и слова места
стали теми, что показывает предпросмотр; «не внесёт» ⇔ не внесло, слова места те же; отказ «Применить» целиком
виден в чтении; модули с побочными эффектами (`paths`, `core`, `recordings`) чтение не импортирует.
"""
import atexit
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-legacy-read-data-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

import legacy_read  # noqa: E402
import legacy_sample_loader as samples  # noqa: E402

EXPECTED = HERE / "legacy_read_expected.json"
EXPECTED_SHA256 = "777de51a7cf9753eadd155e63f33a9cb964a29da191b53f12335b40f29a4e1f6"


def _edit_json(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_bytes(json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8"))


def _answer(name, aid, **fields):
    """Правка: у ответа aid поля fields (None — убрать поле)."""
    def change(side, jp):
        def edit(state):
            a = state["answers"][aid]
            for k, v in fields.items():
                if v is None:
                    a.pop(k, None)
                else:
                    a[k] = v
        _edit_json(side / f"{name}.review" / "answers.json", edit)
    return change


def _word(wid, text):
    def change(side, jp):
        def edit(data):
            for s in data["segments"]:
                for w in s["words"]:
                    if w.get("word_id") == wid:
                        w["w"] = text
        _edit_json(jp, edit)
    return change


def _word_id_dropped(wid):
    def change(side, jp):
        def edit(data):
            for s in data["segments"]:
                for w in s["words"]:
                    if w.get("word_id") == wid:
                        del w["word_id"]
        _edit_json(jp, edit)
    return change


def _pause(name, aid):
    """Место aid — пауза (причина «missed») в вопросах пакета и в стенограмме."""
    def change(side, jp):
        def edit_man(man):
            next(it for it in man["items"] if it["id"] == aid)["reasons"] = ["missed"]
        def edit_data(data):
            next(r for r in data["review"] if r["id"] == aid)["reasons"] = ["missed"]
        _edit_json(side / f"{name}.review" / "manifest.json", edit_man)
        _edit_json(jp, edit_data)
    return change


def _tried(name, aid, outcome):
    """Прошлое «Применить» уже пробовало ответ aid и не внесло его (событие с исходом outcome)."""
    import review_pack
    def change(side, jp):
        def edit(state):
            a = state["answers"][aid]
            state.setdefault("application_events", []).append({
                "id": aid, "version": a.get("version"), "diagnostic_id": None, "answer_source": a.get("source"),
                "answer_ts": a.get("ts"), "answer_sig_sha256": review_pack._sig_hash(a), "action": "apply_attempt",
                "outcome": outcome, "at": "2026-11-20T10:00:09"})
        _edit_json(side / f"{name}.review" / "answers.json", edit)
    return change


def _chain(*changes):
    def change(side, jp):
        for one in changes:
            one(side, jp)
    return change


# случай: (образец, правка копии или None). Имена записей образцов: built* — «встреча», waiting — «вторая».
CASES = {
    "built": ("built", None),
    "built_archive": ("built_archive", None),
    "waiting": ("waiting", None),
    # текст слова места r4 изменился после сборки карточки — файловый режим текст не сверяет, ответ вносится
    "built_words_changed": ("built", _word("w4", "то")),
    # слова места r4 нет (ID пропал) — ответ не вносится
    "built_word_missing": ("built", _word_id_dropped("w4")),
    # ответ без автора — «Применить» отказывает целиком
    "built_no_actor": ("built", _answer("встреча", "r4", actor_kind=None)),
    # пропуск — не ждёт
    "built_skip": ("built", _answer("встреча", "r4", kind="skip", option=None)),
    # отмена внесённого ответа со снимком — нужна проверка
    "built_revert": ("built", _answer("встреча", "r3", kind="revert", option=None)),
    # другой ответ поверх внесённого со снимком — замена, нужна проверка
    "built_replace": ("built", _answer("встреча", "r3", option=0, ts="2026-11-20T10:00:05")),
    # «оставить человеку» — вносится статусом
    "waiting_human": ("waiting", _answer("вторая", "r3", kind="human", option=None)),
    # свой ответ человека — нужна проверка
    "waiting_free": ("waiting", _answer("вторая", "r3", kind="free", option=None, text="дальнейшем")),
    # отрицание — не вносится
    "waiting_negative": ("waiting", _answer("вторая", "r3", kind="free", option=None, text="неверно")),
    # короткий свой ответ («нет» — не отрицание из списка): «Применить» даёт подсказку; чтение — нужна проверка
    "waiting_short": ("waiting", _answer("вторая", "r3", kind="free", option=None, text="нет")),
    # прошлое «Применить» не внесло ответ (страница: «не применён») — «Применить» пробует снова и вносит
    "built_tried": ("built", _tried("встреча", "r4", "blocked_preflight")),
    # место-пауза с «Оставить как есть» — нужна проверка (закрывается прослушиванием)
    "built_pause": ("built", _chain(_pause("встреча", "r4"), _answer("встреча", "r4", option=0))),
}


def make_case(case, root):
    """Файлы случая в новой папке root; -> (папка записи, путь стенограммы JSON)."""
    sample, change = CASES[case]
    root.mkdir(parents=True)
    audio = root / "запись.wav"
    audio.write_bytes(b"synthetic audio")
    side = root / "запись.rus-transcriber"
    jp = samples.load(sample, side, audio)
    if change:
        change(side, jp)
    return side, jp


def snapshot(folder):
    """{путь от папки: (sha256, mtime_ns)} — всё, что лежит в папке записи."""
    return {str(p.relative_to(folder)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def file_apply(case, root):
    """Итог файлового «Применить» на копии случая: {refused, outcomes {id: исход}, words [слова], status {id: статус}}."""
    import core
    import settings
    side, jp = make_case(case, root)
    md = jp.with_name(jp.name[:-len(".json")] + ".md")
    with mock.patch.object(settings.paths, "STATE_DB", root / "нет-базы.sqlite3"):
        try:
            rep = core.apply_answers_files(jp, md)
            refused = False
        except core.CoreError:
            rep, refused = {}, True
    data = json.loads(jp.read_text(encoding="utf-8"))
    return {"refused": refused,
            "outcomes": {o["id"]: o["outcome"] for o in rep.get("outcomes") or []},
            "words": {w["word_id"]: w["w"] for s in data["segments"] for w in s["words"] if w.get("word_id")},
            "status": {r["id"]: r.get("status") for r in data.get("review") or []}}


def freeze_all():
    with tempfile.TemporaryDirectory(prefix="rt-legacy-read-") as tmp:
        return {case: file_apply(case, Path(tmp) / case) for case in CASES}


def file_code_present():
    try:
        import core
    except ImportError:
        return False
    return hasattr(core, "apply_answers_files")


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="rt-legacy-read-")
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def expected(self):
        blob = EXPECTED.read_bytes()
        self.assertEqual(hashlib.sha256(blob).hexdigest(), EXPECTED_SHA256, "замороженный итог изменён")
        return json.loads(blob.decode("utf-8"))


class FrozenFileMode(Base):
    def test_frozen_result_is_recorded_and_matches_file_code(self):
        expected = self.expected()
        self.assertEqual(sorted(expected), sorted(CASES), "случай добавлен или убран без пересборки итога")
        if not file_code_present():
            self.skipTest("файлового «Применить» больше нет: сверка только с замороженным итогом")
        self.assertEqual(freeze_all(), expected, "файловое «Применить» даёт не тот итог, что заморожен")


class ReadAgainstFileMode(Base):
    def test_read_predicts_file_apply(self):
        expected = self.expected()
        seen = {"yes": 0, "no": 0, "check": 0}
        for case in CASES:
            with self.subTest(case=case):
                side, jp = make_case(case, self.root / case)
                before = snapshot(side)
                got = legacy_read.read(jp)
                self.assertEqual(snapshot(side), before, "чтение изменило или создало файлы")
                self.assertFalse(got["writes"])
                frozen = expected[case]
                pack = got["review_pack"]
                self.assertEqual(pack["apply_refused"] is not None, frozen["refused"])
                for row in pack["items"]:
                    aid, outcome = row["id"], frozen["outcomes"].get(row["id"])
                    ids = next(it for it in json.loads((side / Path(pack["folder"]).name / "manifest.json")
                                                       .read_text(encoding="utf-8"))["items"]
                               if it["id"] == aid)["word_ids"]
                    words = [frozen["words"].get(w) for w in ids]
                    after = None if None in words else " ".join(words)
                    if row["state"] == "applied":
                        self.assertIn(outcome, (None, "applied"), aid)
                        self.assertEqual(after, row["preview"]["after"], f"{aid}: внесённый ответ не в словах")
                    if "would_apply" not in row:
                        if not frozen["refused"] and row["state"] not in ("applied", "applied_open", "applied_check"):
                            self.assertNotIn(row["state"], legacy_read.TRIED, aid)
                            self.assertNotIn(outcome, ("applied", "applied_open", "left_for_human", "reverted"),
                                             f"{aid}: «Применить» внесло ответ, который чтение не предсказывало")
                        continue
                    seen[row["would_apply"]] += 1
                    if row["would_apply"] == "yes":
                        # «оставить человеку» вносится статусом места — исход left_for_human
                        self.assertIn(outcome, ("applied", "left_for_human"), aid)
                        self.assertEqual(after, row["preview"]["after"], aid)
                        if row["preview"].get("status"):
                            self.assertEqual(frozen["status"][aid], row["preview"]["status"], aid)
                    elif row["would_apply"] == "no":
                        self.assertNotEqual(outcome, "applied", aid)
                        self.assertEqual(after, row["preview"]["before"], f"{aid}: слова места изменились")
        # каждый исход чтения встречается хотя бы раз — иначе случаи ничего не проверяют
        self.assertTrue(all(seen.values()), seen)

    def test_cases_cover_expected_verdicts(self):
        """Исход чтения по случаям закреплён: правка кода чтения, меняющая их, видна здесь."""
        want = {"built": {"r4": "yes"}, "built_archive": {}, "waiting": {"r3": "yes"},
                "built_words_changed": {"r4": "yes"}, "built_word_missing": {"r4": "no"}, "built_no_actor": {"r4": "no"}, "built_skip": {},
                "built_revert": {"r3": "check", "r4": "yes"}, "built_replace": {"r3": "check", "r4": "yes"},
                "waiting_human": {"r3": "yes"}, "waiting_free": {"r3": "check"}, "waiting_negative": {"r3": "no"},
                "waiting_short": {"r3": "check"}, "built_tried": {"r4": "yes"}, "built_pause": {"r4": "check"}}
        for case, verdicts in want.items():
            with self.subTest(case=case):
                _side, jp = make_case(case, self.root / case)
                items = legacy_read.read(jp)["review_pack"]["items"]
                self.assertEqual({r["id"]: r["would_apply"] for r in items if "would_apply" in r}, verdicts)


class ReadOnly(Base):
    def test_no_lock_no_new_files_and_archives_listed(self):
        side, jp = make_case("built_archive", self.root / "a")
        (side / "встреча.review" / "answers.lock").unlink()
        before = snapshot(side)
        got = legacy_read.read(jp.with_name("встреча.transcript.md"))
        self.assertEqual(snapshot(side), before)
        self.assertFalse((side / "встреча.review" / "answers.lock").exists(), "чтение взяло замок ответов")
        archives = got["review_pack"]["archives"]
        self.assertEqual([(a["answers"], a["applied"], a["cards"]) for a in archives], [(2, 2, 3)])

    def test_transcript_summary_and_window(self):
        _side, jp = make_case("built", self.root / "b")
        got = legacy_read.read(jp, offset=0, limit=1)["transcript"]
        self.assertEqual((got["segments"], got["words"], got["lines_total"], len(got["lines"])), (1, 5, 1, 1))
        self.assertTrue(got["lines"][0].startswith("Анна: "))
        self.assertEqual(legacy_read.read(jp, offset=5)["transcript"]["lines"], [])

    def test_no_pack_and_bad_paths(self):
        _side, jp = make_case("built", self.root / "c")
        shutil.rmtree(jp.with_name("встреча.review"))
        self.assertIsNone(legacy_read.read(jp)["review_pack"])
        for bad in (jp.with_name("встреча.wav"), jp.with_name("нет.transcript.json")):
            with self.assertRaises(legacy_read.LegacyReadError):
                legacy_read.read(bad)

    def test_import_has_no_side_effect_modules(self):
        code = ("import sys, legacy_read; "
                "print([m for m in ('paths', 'core', 'recordings', 'storage', 'settings', 'jobslots') if m in sys.modules])")
        out = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, capture_output=True, text=True,
                             encoding="utf-8", env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")


class Interfaces(Base):
    """Инструмент MCP read_legacy_files и команда rt.py legacy-read отдают то же, что read, и ничего не пишут."""

    def test_mcp_tool(self):
        try:
            import migrate
            with mock.patch.object(migrate, "run", return_value={"status": "пропущено в тесте"}):
                import server
        except ImportError:
            self.skipTest("нет пакета mcp")
        side, jp = make_case("built", self.root / "m")
        before = snapshot(side)
        got = json.loads(server.read_legacy_files(str(jp), 0, 5))
        self.assertEqual(got, json.loads(json.dumps(legacy_read.read(jp, 0, 5), ensure_ascii=False)))
        self.assertEqual(snapshot(side), before)
        self.assertIn("error", json.loads(server.read_legacy_files(str(self.root / "нет.transcript.json"))))

    def test_rt_command(self):
        side, jp = make_case("waiting", self.root / "r")
        before = snapshot(side)
        data = self.root / "data"
        data.mkdir()
        env = {**os.environ, "RUS_TRANSCRIBER_DATA": str(data), "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        out = subprocess.run([sys.executable, "-B", str(PROJECT / "rt.py"), "legacy-read", str(jp), "--limit", "1"],
                             cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr)
        got = json.loads(out.stdout)
        self.assertFalse(got["writes"])
        self.assertEqual(got["review_pack"]["waiting"], {"yes": 1, "no": 0, "check": 0})
        self.assertEqual(snapshot(side), before)
        bad = subprocess.run([sys.executable, "-B", str(PROJECT / "rt.py"), "legacy-read", str(self.root / "x.txt")],
                             cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
        self.assertEqual(bad.returncode, 1)


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        blob = (json.dumps(freeze_all(), ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
        EXPECTED.write_bytes(blob)
        print(EXPECTED.name, hashlib.sha256(blob).hexdigest())
    else:
        unittest.main()
