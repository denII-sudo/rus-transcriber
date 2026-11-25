"""Проверка находок аудита 25.11.2026 о делении речи по голосам: Д4, Д2, Д5 (только синтетика).

Д4 — блок голосов в worker.main перехватывает любое исключение и продолжает с quality["voices_error"]:
сбой искусственно вносится в каждую точку блока, смотрим, остаётся ли состояние согласованным и видит
ли человек предупреждение (статус, стенограмма, чтение записи). Д2 — одно имя на несколько кластеров:
метки сливаются; видит ли человек слияние. Д5 — автор реплики переназначается по роду глагола и сходству
голоса: на чём основано и помечается ли сомнением.

Модели не загружаются, настоящие записи, профили и стенограммы не читаются: звук — нули, голоса — выдуманные
векторы (Анна, Борис), запись — SQLite во временной папке. Тесты с пометкой expectedFailure показывают
найденное несоответствие правилу продукта («человек > голосовой профиль > распознавание > эвристики»,
«не выбирать автора спорной реплики за человека»): они не закрепляют дефект, а станут «неожиданными
успехами», когда исправление будет принято, и тогда пометку надо снять. Рабочий код тест не меняет.
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

if "paths" not in sys.modules:   # папка данных — временная ДО первого импорта программы (test_data_safety)
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-voices-d-data-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core
import recordings
import settings
import transcript
import voices
import words
import worker

SR = worker.SR
# Профили: Анна и Борис — ортогональные единичные векторы. Кластеры записи (по реплике):
#   «Спикер 1» (0–20 и 50–60 с) — голос Анны; «Спикер 2» (20–30 с) — другой человек, голос похож на Анну
#   (сходство 0,70) и совсем не похож на Бориса; «Спикер 3» (30–50 с) — голос Бориса.
VEC = {"Анна": np.array([1.0, 0.0, 0.0]), "Борис": np.array([0.0, 1.0, 0.0])}
SIMILAR = np.array([0.7, 0.0, 0.7141428428542850])      # единичный, сходство с Анной 0,70
RAW = [(0.0, 20.0, "S0"), (20.0, 30.0, "S1"), (30.0, 50.0, "S2"), (50.0, 60.0, "S0")]
CLUSTER_VEC = {"S0": VEC["Анна"], "S1": SIMILAR, "S2": VEC["Борис"]}
BEFORE_LABELS = ["Спикер 1", "Спикер 2", "Спикер 3", "Спикер 1"]   # метки pyannote по порядку появления
FAIL = "синтетический сбой"


class FakeEmbedder:
    """Вместо модели отпечатков: вектор реплики определяется её кластером в RAW."""
    seconds = 0.0
    fail_on_spans = False

    def __init__(self, _pipe):
        pass

    def spans(self, audio, spans, **_kwargs):
        if FakeEmbedder.fail_on_spans:
            raise RuntimeError(FAIL)
        out = []
        for s, e in spans:
            mid = (s + e) / 2
            out.append(next((CLUSTER_VEC[spk] for a, b, spk in RAW if a <= mid < b), None))
        return out


def profile(confusions=None):
    prof = {"version": voices.PROFILE_VERSION, "model": "synthetic", "sources": {},
            "speakers": {n: {"parts": {"earlier.wav": {"sec": 100.0, "vec": v}}} for n, v in VEC.items()}}
    if confusions is not None:
        prof["confusions"] = confusions
    return prof


def make_segments(turns, label_name):
    segs = []
    for t in turns:
        ws, x = [], t["s"]
        while x + 0.4 <= t["e"] + 1e-9:
            ws.append({"w": "слово", "s": round(x, 2), "e": round(x + 0.4, 2)})
            x += 0.5
        name = label_name.get(t["label"], t["label"])
        segs.append({"start": t["s"], "end": t["e"], "speaker": name, "label": t["label"],
                     "text": " ".join(w["w"] for w in ws), "words": ws})
    return segs


class Run:
    """Результат одного прогона worker.main: состояние, переданное пословной обработке, статус и стенограмма."""


def run_main(fault=None):
    """worker.main на синтетике. fault — точка искусственного сбоя в блоке голосов (None — без сбоя).
    Песочница — новая на каждый прогон: свои папки данных, TEMP и TMP; убирается после прогона."""
    root = Path(tempfile.mkdtemp(prefix="rt-voices-d-"))
    out = Run()
    out.root = root
    try:
        for sub in ("data", "tmp"):
            (root / sub).mkdir()
        db = root / "state.sqlite3"
        src = root / "audio.wav"
        src.write_bytes(b"synthetic placeholder")
        store = recordings.initialize_store(db, display_name="Инициатор")
        try:
            context = recordings.operation_context(store, "model")
        finally:
            store.close()
        status = root / "status.json"
        status.write_text(json.dumps({"recording_context": context}), encoding="utf-8")
        captured = {}
        prof = profile({"earlier.wav": {"нет-разделителя": 1}} if fault == "confusions" else None)
        ppath = "voices.json" if fault == "report" else Path("voices.json")   # у str нет .name

        def fake_run_words(audio, src_, cfg, st, raw, order, turns, full_ann, emb, label_name, label_of,
                           voice_ctx, allowed, genders_by_name, vocab, quality, make_asr, a):
            captured.update(turns=[dict(t) for t in turns], label_name=dict(label_name),
                            label_of=dict(label_of), voice_ctx=dict(voice_ctx), quality=quality)
            labels = list(dict.fromkeys(t["label"] for t in turns))
            data = {"speakers": [label_name.get(l, l) for l in labels], "speaker_labels": labels,
                    "quality": quality, "review": [], "review_done": False, "edits": [],
                    "segments": make_segments(turns, label_name), "_timing": {}}
            names = {l: label_name[l] for l in labels if l in label_name}
            if names:
                data.update(speaker_names=names, names_source="voices", names_confirmed=False)
            return data

        def failing(_name):
            def inner(*_a, **_k):
                raise RuntimeError(FAIL)
            return inner

        patches = [
            mock.patch.object(settings.paths, "STATE_DB", db),
            mock.patch.object(sys, "argv", ["worker.py", str(src), "--status", str(status),
                                            "--storage-backend", "sqlite"]),
            mock.patch.dict(os.environ, {"HF_TOKEN": "synthetic-token", "TEMP": str(root / "tmp"),
                                         "TMP": str(root / "tmp")}),
            mock.patch.object(worker, "wait_slot"),
            mock.patch.object(worker.tmpfiles, "sweep_stale"),
            mock.patch.object(worker, "load_config", return_value={"pipeline": "words", "voices": True}),
            mock.patch.object(worker, "resolve_vocab_details", return_value={
                "path": None, "vocab": {"participants": [{"name": "Анна"}, {"name": "Борис"}]},
                "group": None, "report": {}}),
            mock.patch.object(worker, "load_audio", return_value=np.zeros(60 * SR, dtype=np.float32)),
            mock.patch.object(worker, "diarize", return_value=(list(RAW), {}, None, None)),
            mock.patch.object(worker, "prepare_group_voice_profile", return_value=(ppath, prof, [])),
            mock.patch.object(worker, "run_words", fake_run_words),
            mock.patch.object(voices, "Embedder", failing("embedder") if fault == "embedder" else FakeEmbedder),
        ]
        if fault == "profile":
            patches.append(mock.patch.object(worker, "prepare_group_voice_profile", failing("profile")))
        if fault in ("centroids", "voice_vectors", "match", "references"):
            patches.append(mock.patch.object(voices, fault, failing(fault)))
        FakeEmbedder.fail_on_spans = fault == "spans"
        for p in patches:
            p.start()
        out.exit_code = None
        try:
            worker.main()
        except SystemExit as ex:      # worker при ошибке задачи пишет статус error и завершается с кодом 1
            out.exit_code = ex.code
        finally:
            for p in reversed(patches):
                p.stop()
            FakeEmbedder.fail_on_spans = False
        out.status = json.loads(status.read_text(encoding="utf-8"))
        out.captured = captured
        out.db = db
        if out.status.get("state") != "done":
            return out
        with mock.patch.object(settings.paths, "STATE_DB", db):
            out.data = transcript.load(out.status["json"])
            out.md = transcript.render_md(out.data)
            out.read_md = core.read_transcript("recording:" + out.status["recording_id"], "md")
            out.read_json = core.read_transcript("recording:" + out.status["recording_id"], "json")
        return out
    except BaseException:
        out.cleanup()
        raise


def _cleanup(self):
    shutil.rmtree(self.root, ignore_errors=True)


Run.cleanup = _cleanup


def is_unchanged(cap):
    """Состояние такое, будто блока голосов не было: метки pyannote, ни имён, ни связей."""
    return ([t["label"] for t in cap["turns"]] == BEFORE_LABELS and not cap["label_name"]
            and not cap["label_of"] and not cap["voice_ctx"])


def is_complete(cap):
    """Блок голосов отработал целиком: имена, связи кластеров с профилями и запись о них в quality."""
    return (bool(cap["label_name"]) and bool(cap["voice_ctx"].get("mapping"))
            and bool(cap["quality"].get("match")))


class FaultTests(unittest.TestCase):
    """Д4: искусственный сбой в каждой точке блока voices в worker.main (worker.py:1683–1757)."""

    def run_fault(self, fault):
        out = run_main(fault)
        self.addCleanup(out.cleanup)
        return out

    def test_control_without_fault_names_and_merge_are_complete(self):
        out = self.run_fault(None)
        self.assertEqual(out.status["state"], "done", out.status)
        self.assertTrue(is_complete(out.captured), out.captured)
        self.assertEqual(out.captured["label_name"], {"Спикер 1": "Анна", "Спикер 3": "Борис"})
        self.assertIsNone(out.status["quality"]["voices_error"])

    def test_faults_before_any_state_change_leave_state_untouched_and_run_continues(self):
        # точки, где ещё ничего не изменено: эти сбои не оставляют полусостояния (подтверждено прогоном)
        for fault in ("profile", "spans", "centroids", "voice_vectors", "match", "references"):
            with self.subTest(fault=fault):
                out = self.run_fault(fault)
                self.assertEqual(out.status["state"], "done", out.status)    # работа продолжается
                self.assertIn(FAIL, out.captured["quality"]["voices_error"])
                self.assertTrue(is_unchanged(out.captured), out.captured)

    def test_embedder_fault_is_recorded_but_then_stops_the_whole_job(self):
        """Д4, подтверждено прогоном: если не создаётся Embedder, блок голосов записывает voices_error, но
        пословная обработка создаёт Embedder заново вне try (worker.py:1760–1763) и задача завершается
        ошибкой, а не «продолжается без голосов». Сбой виден (state=error), а не тих."""
        out = self.run_fault("embedder")
        self.assertEqual(out.status["state"], "error", out.status)
        self.assertEqual(out.exit_code, 1)
        self.assertIn(FAIL, out.status["error"])

    @unittest.expectedFailure
    def test_fault_after_labels_merged_leaves_a_half_applied_state(self):
        """Д4 (подтверждено прогоном): подпорченная запись confusions в профиле — профиль её принимает
        (voices.validate_profile не проверяет confusions), а voices.confusions падает на разборе ключа
        уже ПОСЛЕ слияния кластеров и записи label_name/label_of (worker.py:1708–1714). Дальше идёт
        пословная обработка с именами, но без связи кластеров с профилями (voice_ctx пуст), без
        quality.match; метки в turns уже слиты. Ожидание: либо ничего не применено, либо всё."""
        out = self.run_fault("confusions")
        self.assertEqual(out.status["state"], "done", out.status)
        self.assertIn("voices_error", out.captured["quality"])
        self.assertTrue(is_unchanged(out.captured) or is_complete(out.captured), out.captured)

    @unittest.expectedFailure
    def test_fault_in_quality_report_leaves_names_without_match_record(self):
        """Д4 (подтверждено прогоном): сбой при составлении отчёта о сопоставлении (worker.py:1726–1730,
        здесь — искусственный, путь профиля не Path) оставляет применённые имена и слитые метки, а
        quality.match/profile_names/unmatched/cluster_sims не записаны: имена «по голосовым профилям»
        есть, а доказательств, какой кластер кому сопоставлен, в стенограмме нет."""
        out = self.run_fault("report")
        self.assertEqual(out.status["state"], "done", out.status)
        self.assertTrue(is_unchanged(out.captured) or is_complete(out.captured), out.captured)

    def test_error_is_in_status_and_saved_json_quality_and_recording_result(self):
        out = self.run_fault("spans")
        self.assertIn(FAIL, out.status["quality"]["voices_error"])             # transcription_status
        self.assertIn(FAIL, out.data["quality"]["voices_error"])               # JSON стенограммы
        with mock.patch.object(settings.paths, "STATE_DB", out.db):
            import migrate
            with mock.patch.object(migrate, "run"):
                import server
            shown = json.loads(server.read_recording_result(out.status["recording_id"]))
        self.assertIn(FAIL, shown["quality"]["voices_error"])                  # MCP read_recording_result

    @unittest.expectedFailure
    def test_error_is_visible_in_the_transcript_text_a_human_reads(self):
        """Д4 (подтверждено прогоном): в Markdown стенограммы нет ни слова о том, что проверка голосов
        отключена (transcript.render_md показывает group_doubt, names_outside_vocab и ошибки ASR, но не
        voices_error). Ожидание: в шапке — предупреждение, как у остальных «Внимание»."""
        out = self.run_fault("spans")
        header = out.md.split("**[", 1)[0].lower()
        self.assertIn("голос", header)

    @unittest.expectedFailure
    def test_error_is_visible_in_default_read_transcript(self):
        """Д4 (подтверждено прогоном): обычное чтение стенограммы агентом (core.read_transcript, режим по
        умолчанию) отбрасывает quality целиком и переносит в asr_summary только пять ключей ASR;
        в Markdown-режиме служебные разделы свёрнуты. voices_error не доходит."""
        out = self.run_fault("spans")
        self.assertTrue("голос" in out.read_md.split("**[", 1)[0].lower() or "voices_error" in out.read_json)


class MergeTests(unittest.TestCase):
    """Д2: одно имя на несколько кластеров (voices.match, worker.py:1707–1712)."""

    def test_match_gives_one_name_to_two_clusters_even_for_a_different_similar_person(self):
        cents = {"Спикер 1": VEC["Анна"], "Спикер 2": SIMILAR, "Спикер 3": VEC["Борис"]}
        got = voices.match(cents, VEC)
        self.assertEqual({lbl: nm for lbl, (nm, _) in got.items()},
                         {"Спикер 1": "Анна", "Спикер 2": "Анна", "Спикер 3": "Борис"})
        self.assertEqual(got["Спикер 2"][1], 0.7)     # сходство 0,70 при пороге MATCH_MIN = 0,50

    def test_main_merges_the_labels_and_records_the_pairing_only_in_quality_match(self):
        out = run_main(None)
        self.addCleanup(out.cleanup)
        cap = out.captured
        self.assertEqual([t["label"] for t in cap["turns"]], ["Спикер 1", "Спикер 1", "Спикер 3", "Спикер 1"])
        self.assertEqual({l: m["name"] for l, m in cap["quality"]["match"].items()},
                         {"Спикер 1": "Анна", "Спикер 2": "Анна", "Спикер 3": "Борис"})
        self.assertNotIn("Спикер 2", out.md)          # в тексте стенограммы кластера нет (слит)

    @unittest.expectedFailure
    def test_merge_of_two_clusters_is_visible_to_the_human(self):
        """Д2 (подтверждено прогоном): реплика другого человека (кластер «Спикер 2», сходство с Анной 0,70)
        становится репликами Анны без пометки: ни сомнительного места, ни строки в шапке, ни карточки;
        group_doubt срабатывает только когда узнано меньше половины речи. Запись о слиянии — лишь
        quality.match в JSON. Ожидание: слитый кластер назван там, где человек читает стенограмму."""
        out = run_main(None)
        self.addCleanup(out.cleanup)
        header = out.md.split("**[", 1)[0]
        self.assertIn("Спикер 2", header)
        self.assertEqual(out.data["review"], [])      # сейчас сомнений нет; ожидаем хотя бы одно


def gender_segment(speaker, text, sims):
    ws = [{"w": w, "s": 1.0 + 0.5 * i, "e": 1.4 + 0.5 * i, "c": 1.0, "sims": dict(sims)}
          for i, w in enumerate(text.split())]
    return {"start": 1.0, "end": ws[-1]["e"], "speaker": speaker, "label": "Спикер 2", "text": text, "words": ws}


GENDERS = {"Анна": "f", "Борис": "m"}   # как vocab.genders(): 'f' | 'm'
LABEL_OF = {"Анна": "Спикер 1", "Борис": "Спикер 2"}


class ReassignTests(unittest.TestCase):
    """Д5: переназначение автора по роду глагола и голосу (живой путь — worker._check_gender; прежний
    pipeline="segments" — worker.apply_gender и voices.reassign)."""

    def check(self, gap):
        seg = gender_segment("Борис", "я сказала это вчера", {"Анна": 0.50 + gap, "Борис": 0.50})
        quality = {"speaker_fixes": []}
        worker._check_gender([seg], GENDERS, {n: VEC[n] for n in GENDERS}, LABEL_OF, quality)
        return seg, quality

    def test_basis_is_a_verb_ending_in_asr_text_plus_voice_similarity_and_the_swap_is_recorded(self):
        seg, quality = self.check(0.10)
        self.assertEqual(seg["speaker"], "Анна")
        self.assertEqual(seg["label"], "Спикер 1")
        self.assertEqual(quality["speaker_fixes"][0]["kind"], "gender")       # в списке «Автоматические исправления»
        self.assertIn("сказала", quality["speaker_fixes"][0]["words"])
        self.assertTrue(seg["fix"].startswith("род"))
        self.assertEqual([i for i in words.review_items([seg]) if "gender" in i["reasons"]], [])  # сомнения нет

    @unittest.expectedFailure
    def test_near_tie_of_voices_must_not_swap_the_author(self):
        """Д5 (подтверждено прогоном): живая проверка рода (worker.py:928–949) переносит реплику к другому
        участнику уже при превосходстве сходства на 0,02, а сравнение сходств идёт по усреднённым сходствам
        слов; прежний путь (apply_gender) требует GENDER_MARGIN = 0,05. Основание — окончание в тексте
        распознавания: ошибка распознавания («сказала» вместо «сказал») меняет автора. Реплика не помечается
        сомнением. Ожидание по правилу «не выбирать автора спорной реплики за человека»: при почти равных
        голосах автор остаётся, реплика получает метку для человека."""
        seg, quality = self.check(0.02)
        self.assertEqual(seg["speaker"], "Борис")
        self.assertTrue(seg.get("gender_flag"))
        self.assertEqual(quality["speaker_fixes"], [])

    def test_legacy_apply_gender_flags_the_same_near_tie_instead_of_swapping(self):
        seg = gender_segment("Борис", "я сказала это вчера", {})
        seg["label"] = "Спикер 2"
        fixes, review = [], []
        refs = {n: VEC[n][:2] for n in GENDERS}
        worker.apply_gender([seg], [np.array([0.52, 0.50])], refs, LABEL_OF, GENDERS, fixes, review)
        self.assertEqual(seg["speaker"], "Борис")                              # разрыв 0,02 < GENDER_MARGIN
        self.assertEqual(fixes, [])
        self.assertEqual(len(review), 1)                                       # и сразу в список для проверки

    def test_voice_reassign_moves_only_beyond_own_bonus_and_records_the_fix(self):
        names = ["Анна", "Борис"]
        pv = {n: VEC[n][:2] for n in names}
        label_name = {"Спикер 1": "Анна", "Спикер 2": "Борис"}
        for gap, moved in ((0.04, False), (0.06, True)):
            with self.subTest(gap=gap):
                turn = {"s": 0.0, "e": 5.0, "label": "Спикер 2"}
                fixes = []
                out = voices.reassign([turn], [np.array([0.50 + gap, 0.50])], pv, dict(label_name),
                                      dict(LABEL_OF), fixes)
                self.assertEqual(out[0]["label"] == "Спикер 1", moved)
                self.assertEqual(len(fixes), int(moved))
                self.assertEqual("fix" in out[0], moved)

    @unittest.expectedFailure
    def test_reassign_failure_leaves_recorded_fixes_for_turns_that_are_thrown_away(self):
        """Д4, прежний pipeline="segments" (worker.py:1749): voices.reassign дописывает поправки в общий
        список по ходу и не откатывает их, если позже падает на другой реплике; worker при этом оставляет
        прежние turns, а quality.speaker_fixes уже содержит «исправления», которых в тексте нет.
        Подтверждено на функции; прогон main в этом режиме не делался."""
        pv = {n: VEC[n][:2] for n in ("Анна", "Борис")}
        turns = [{"s": 0.0, "e": 5.0, "label": "Спикер 2"}, {"s": 6.0, "e": 11.0, "label": "Спикер 2"}]
        embs = [np.array([0.60, 0.50]), np.array([1.0, 2.0, 3.0])]     # у второй неверная размерность
        fixes = []
        with self.assertRaises(ValueError):
            voices.reassign(turns, embs, pv, {"Спикер 1": "Анна", "Спикер 2": "Борис"}, dict(LABEL_OF), fixes)
        self.assertEqual(fixes, [])    # ожидание: ничего не записано; сейчас пишется поправка к 1-й реплике


if __name__ == "__main__":
    unittest.main()
