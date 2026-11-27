"""Ч-1: `legacy_read.read` — чтение старых файлов записи без записи — против замороженного итога файлового «Применить».

Случаи — образцы старого формата (tests/legacy_samples, #224) и их детерминированные правки (`CASES`). Итог
файлового «Применить» (`core.apply_answers_files` на копии каждого случая) заморожен в
`tests/legacy_read_expected.json`; его sha256 — `EXPECTED_SHA256`. Пока файловый код есть, тест пересчитывает итог
и сверяет с замороженным; после удаления файлового кода сверка идёт только с замороженным.

Пересобрать замороженный итог (только пока файловый код есть, при изменении образцов или CASES):
    python -B tests/test_legacy_read.py --freeze     (с RUS_TRANSCRIBER_DATA, TEMP и TMP на песочницу)
и записать новый sha256 в EXPECTED_SHA256.

Что проверяется: чтение ничего не меняет и не создаёт; подсказка каждого ответа и исход файлового «Применить»
закреплены по случаям (`WANT`): `words_in_place` — слова карточки в файле те, что в предпросмотре (внесён ли ответ —
видно по исходу, внесение не обещано: Ф238-3), `not_applied` — ответ не попал в стенограмму и слова те же,
`check` — исход закреплён; отказ «Применить» целиком виден в чтении; документ базы и файлы неожиданного вида —
понятный отказ; модули с побочными эффектами (`paths`, `core`, `recordings`) чтение не импортирует.
"""
import atexit
import copy
import hashlib
import importlib.util
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
EXPECTED_SHA256 = "9b057a2c50f8004c1e755c3861b8e68228cca8073afcec44b59662034ac27584"


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


def _word_edit(wid, text, by):
    """Слово wid исправлено правкой отрезка автора by (след правки — как у transcript.edit_span)."""
    def change(side, jp):
        def edit(data):
            for s in data["segments"]:
                for w in s["words"]:
                    if w.get("word_id") == wid:
                        w.update({"w": text, "edited": "span@test", "by": by,
                                  "provenance": {"text": {"by": by, "edit": "span@test", "via": "edit_span"}}})
        _edit_json(jp, edit)
    return change


def _card(name, base, aid, card, place, answer):
    """Новая карточка aid — копия карточки base с полями card (None — убрать поле), её место в стенограмме — копия
    места base с полями place, и ждущий ответ человека с полями answer."""
    def fields(row, extra):
        row = {**copy.deepcopy(row), "id": aid}
        for k, v in extra.items():
            if v is None:
                row.pop(k, None)
            else:
                row[k] = v
        return row

    def change(side, jp):
        pd = side / f"{name}.review"

        def add_item(man):
            man["items"].append(fields(next(it for it in man["items"] if it["id"] == base), card))

        def add_answer(state):
            add_item(state["manifest"])
            a = copy.deepcopy(state["answers"][base])
            a.update({"id": aid, "ts": "2026-11-20T10:00:03", **answer})
            state["answers"][aid] = a

        def add_place(data):
            data["review"] = [r for r in data["review"] if r["id"] != aid]
            data["review"].append(fields(next(r for r in data["review"] if r["id"] == base), place))

        _edit_json(pd / "manifest.json", add_item)
        _edit_json(pd / "answers.json", add_answer)
        _edit_json(jp, add_place)
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
    # текст слова места r4 изменился после сборки карточки правкой без автора (старые файлы) — файловый режим текст
    # не сверяет, ответ вносится
    "built_words_changed": ("built", _word("w4", "то")),
    # слово места r4 исправил человек правкой отрезка — «Применить» не перезаписывает правку человека (отказ),
    # а слова карточки на месте: подсказка words_in_place не обещает внесения (Ф238-3)
    "built_human_edit": ("built", _word_edit("w4", "те", "human")),
    # то же правкой агента — ответ человека её перезаписывает
    "built_agent_edit": ("built", _word_edit("w4", "те", "claude")),
    # вторая ждущая карточка r5 на тех же словах, что r4 — «Применить» оставит обе человеку
    "built_overlap": ("built", _card("встреча", "r4", "r5", {}, {}, {})),
    # вторая ждущая карточка r5 старого вида без ID слов, по времени перекрывает r4 (Ф238-3)
    "built_old_card": ("built", _card("встреча", "r4", "r5",
                                      {"word_ids": None, "t2": 3.8, "text": "та конец"},
                                      {"word_ids": None, "t2": 3.8, "text": "та конец"}, {})),
    # повреждённый номер варианта (-1): прежнее «Применить» внесло последний вариант (Ф238-4)
    "built_bad_option": ("built", _answer("встреча", "r4", option=-1)),
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


def file_code_absent():
    """Почему файлового «Применить» больше нет (None — оно есть и работает без базы).

    После У-1/У-2 функция может остаться, но отказывать без базы ошибкой хранилища (`StorageError`, не
    `CoreError`): это тоже «файлового кода нет», а не падение теста (Ф238-8). Любая другая ошибка — падение."""
    try:
        import core
    except ImportError:
        return "нет модуля core"
    if not hasattr(core, "apply_answers_files"):
        return "нет core.apply_answers_files"
    with tempfile.TemporaryDirectory(prefix="rt-legacy-read-probe-") as tmp:
        try:
            file_apply("built", Path(tmp) / "probe")
        except Exception as ex:  # noqa: BLE001 — разбирается по имени класса ниже
            if type(ex).__name__ == "StorageError":
                return f"файловое «Применить» отказывает без базы: {ex}"
            raise
    return None


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
        absent = file_code_absent()
        if absent:
            self.skipTest(f"файлового «Применить» больше нет ({absent}): сверка только с замороженным итогом")
        self.assertEqual(freeze_all(), expected, "файловое «Применить» даёт не тот итог, что заморожен")


# исходы файлового «Применить», при которых ответ попал в стенограмму (словами, статусом места или отменой)
TOOK_EFFECT = ("applied", "applied_open", "applied_check", "left_for_human", "reverted")
W, N, C = legacy_read.WORDS_IN_PLACE, legacy_read.NOT_APPLIED, legacy_read.CHECK
# случай -> {карточка: (подсказка чтения, исход файлового «Применить» из замороженного итога)}. Исход закреплён и у
# words_in_place, и у check: подсказка не предсказывает его, а таблица показывает, что на деле сделало «Применить».
WANT = {
    "built": {"r4": (W, "applied")}, "built_archive": {}, "waiting": {"r3": (W, "applied")},
    "built_words_changed": {"r4": (W, "applied")}, "built_word_missing": {"r4": (N, "apply_conflict")},
    "built_no_actor": {"r4": (N, None)}, "built_skip": {},
    "built_revert": {"r3": (C, "reverted"), "r4": (W, "applied")},
    "built_replace": {"r3": (C, "applied"), "r4": (W, "applied")},
    "waiting_human": {"r3": (W, "left_for_human")}, "waiting_free": {"r3": (C, "applied")},
    "waiting_negative": {"r3": (N, "negative")}, "waiting_short": {"r3": (C, "hint")},
    "built_tried": {"r4": (W, "applied")}, "built_pause": {"r4": (C, "applied")},
    "built_human_edit": {"r4": (W, "apply_conflict")}, "built_agent_edit": {"r4": (W, "applied")},
    "built_overlap": {"r4": (C, "blocked_preflight"), "r5": (C, "blocked_preflight")},
    "built_old_card": {"r4": (W, "blocked_preflight"), "r5": (C, "blocked_preflight")},
    "built_bad_option": {"r4": (C, "applied")},
}


def _card_words(side, pack, aid, words):
    """Слова карточки aid в словаре {word_id: слово}: строка, или None — карточка без ID слов либо слова нет."""
    man = json.loads((side / Path(pack["folder"]).name / "manifest.json").read_text(encoding="utf-8"))
    ids = next(it for it in man["items"] if it["id"] == aid).get("word_ids") or []
    got = [words.get(w) for w in ids]
    return None if not ids or None in got else " ".join(got)


class ReadAgainstFileMode(Base):
    def test_read_against_file_apply(self):
        """Подсказка чтения против итога файлового «Применить» на тех же файлах.

        words_in_place ⇔ слова карточки в файле до «Применить» именно те, что показывает preview.before; если ответ
        внесён, слова стали preview.after. not_applied ⇔ ответ не попал в стенограмму и слова места те же. check —
        исход закреплён в WANT. Ответ без подсказки «Применить» не вносит."""
        expected = self.expected()
        seen = {W: 0, N: 0, C: 0}
        for case in CASES:
            with self.subTest(case=case):
                side, jp = make_case(case, self.root / case)
                before = snapshot(side)
                words_before = {w["word_id"]: w["w"] for s in json.loads(jp.read_text(encoding="utf-8"))["segments"]
                                for w in s["words"] if w.get("word_id")}
                got = legacy_read.read(jp)
                self.assertEqual(snapshot(side), before, "чтение изменило или создало файлы")
                self.assertFalse(got["writes"])
                frozen = expected[case]
                pack = got["review_pack"]
                self.assertEqual(pack["apply_refused"] is not None, frozen["refused"])
                self.assertEqual({r["id"]: (r["apply_hint"], frozen["outcomes"].get(r["id"]))
                                  for r in pack["items"] if "apply_hint" in r}, WANT[case])
                for row in pack["items"]:
                    aid, outcome = row["id"], frozen["outcomes"].get(row["id"])
                    was = _card_words(side, pack, aid, words_before)
                    now = _card_words(side, pack, aid, frozen["words"])
                    if row["state"] == "applied":
                        self.assertIn(outcome, (None, "applied"), aid)
                        self.assertEqual(now, row["preview"]["after"], f"{aid}: внесённый ответ не в словах")
                    hint = row.get("apply_hint")
                    if hint is None:
                        if not frozen["refused"] and row["state"] not in ("applied", "applied_open", "applied_check"):
                            self.assertNotIn(row["state"], legacy_read.TRIED, aid)
                            self.assertNotIn(outcome, TOOK_EFFECT,
                                             f"{aid}: «Применить» внесло ответ, у которого нет подсказки")
                        continue
                    seen[hint] += 1
                    if hint == W:
                        self.assertIsNotNone(was, aid)
                        self.assertEqual(was, row["preview"]["before"], f"{aid}: слова карточки не те, что в файле")
                        if outcome in ("applied", "left_for_human"):
                            self.assertEqual(now, row["preview"]["after"], aid)
                            if row["preview"].get("status"):
                                self.assertEqual(frozen["status"][aid], row["preview"]["status"], aid)
                        else:
                            self.assertEqual(now, was, f"{aid}: не внесено, а слова изменились")
                    elif hint == N:
                        self.assertNotIn(outcome, TOOK_EFFECT, aid)
                        self.assertEqual(now, was, f"{aid}: слова места изменились")
        # каждая подсказка встречается хотя бы раз — иначе случаи ничего не проверяют
        self.assertTrue(all(seen.values()), seen)
        self.assertEqual(sorted(WANT), sorted(CASES))

    def test_words_in_place_does_not_promise_apply(self):
        """Ф238-3: правка слова человеком — слова на месте, а «Применить» отказывает; подсказка говорит об этом."""
        side, jp = make_case("built_human_edit", self.root / "h")
        row = next(r for r in legacy_read.read(jp)["review_pack"]["items"] if r["id"] == "r4")
        self.assertEqual(row["apply_hint"], W)
        self.assertIn("не обещание", row["apply_hint_why"])
        self.assertIn("правки этих слов человеком", row["apply_hint_why"])
        self.assertEqual(self.expected()["built_human_edit"]["outcomes"]["r4"], "apply_conflict")


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

    def test_database_document_and_odd_shapes_refused_clearly(self):
        """Ф238-2: документ базы и файлы неожиданного вида — понятный отказ LegacyReadError, не AttributeError."""
        v4 = self.root / "d" / "v4"
        v4.mkdir(parents=True)
        doc = v4 / "встреча.transcript.json"
        doc.write_text(json.dumps({"passport": {"storage_backend": "sqlite"}, "transcript": {"segments": []}}),
                       encoding="utf-8")
        with self.assertRaisesRegex(legacy_read.LegacyReadError, "документ базы"):
            legacy_read.read(doc)
        side, jp = make_case("built", self.root / "e")
        _edit_json(jp, lambda data: data.update(segments="не список"))
        with self.assertRaisesRegex(legacy_read.LegacyReadError, "неожиданного вида"):
            legacy_read.read(jp)
        side, jp = make_case("built", self.root / "f")
        _edit_json(side / "встреча.review" / "answers.json", lambda st: st["answers"].update(r4="строка"))
        with self.assertRaisesRegex(legacy_read.LegacyReadError, "неожиданного вида"):
            legacy_read.read(jp)
        with self.assertRaisesRegex(legacy_read.LegacyReadError, "целые числа"):
            legacy_read.read(jp, offset="abc")

    def test_conspects_found_with_brackets_in_name_and_database_note(self):
        """Ф238-10: скобки в имени записи — конспекты находятся. Ф238-12: у записи с recording_id сказано, что
        действующее состояние перенесённой записи — в базе."""
        side, jp = make_case("built", self.root / "g")
        name = "встреча [1]"
        moved = {}
        for p in list(side.iterdir()):
            if p.name.startswith("встреча."):
                moved[p.name] = p.rename(side / (name + p.name[len("встреча"):]))
        jp = side / f"{name}.transcript.json"
        for suffix in (".итоги.conspect.json", ".итоги.conspect.md"):
            (side / (name + suffix)).write_text("{}", encoding="utf-8")
        (side / "встреча 1.итоги.conspect.json").write_text("{}", encoding="utf-8")   # чужая запись
        before = snapshot(side)
        got = legacy_read.read(jp)
        self.assertEqual(snapshot(side), before)
        self.assertEqual(sorted(Path(c["file"]).name for c in got["conspects"]),
                         [name + ".итоги.conspect.json", name + ".итоги.conspect.md"])
        self.assertIn("в базе", got["database_note"])
        _edit_json(jp, lambda data: data.pop("recording_id"))
        self.assertNotIn("database_note", legacy_read.read(jp))

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
        # пропуск — только без пакета mcp: ImportError не должен молча пропускать тест (Ф238-8)
        if importlib.util.find_spec("mcp") is None:
            self.skipTest("нет пакета mcp")
        import server
        side, jp = make_case("built", self.root / "m")
        before = snapshot(side)
        got = json.loads(server.read_legacy_files(str(jp), 0, 5))
        self.assertEqual(got, json.loads(json.dumps(legacy_read.read(jp, 0, 5), ensure_ascii=False)))
        self.assertEqual(snapshot(side), before)
        self.assertIn("error", json.loads(server.read_legacy_files(str(self.root / "нет.transcript.json"))))
        _edit_json(jp, lambda data: data.update(segments="не список"))
        self.assertIn("неожиданного вида", json.loads(server.read_legacy_files(str(jp)))["error"])
        self.assertIn("apply_hint", server.read_legacy_files.__doc__)
        self.assertNotIn("would_apply", server.read_legacy_files.__doc__)

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
        self.assertEqual(got["review_pack"]["apply_hints"], {W: 1, N: 0, C: 0})
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
