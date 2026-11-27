"""Проверка находок аудита 25.11.2026 о делении речи по голосам: Д4, Д2, Д5 (только синтетика).

Д4 — блок голосов в worker.main перехватывает любое исключение и продолжает с quality["voices_error"]:
сбой искусственно вносится в каждую точку блока, смотрим, остаётся ли состояние согласованным и видит
ли человек предупреждение (статус, стенограмма, чтение записи). Исправлено в PR «Д4, пункты 1–4»: блок
применяется одним шагом, предупреждение — в шапке стенограммы и в read_transcript, профиль проверяет confusions. Д2 — одно имя на несколько кластеров:
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
WARNING = "проверка голосов не выполнена или выполнена не полностью"   # шапка Markdown (transcript.render_md)


class FakeEmbedder:
    """Вместо модели отпечатков: вектор реплики определяется её кластером в RAW."""
    seconds = 0.0
    fail_on_spans = False
    created = 0          # сколько раз создавали (проверка «Embedder создаётся один раз»)
    failed_calls = 0     # сколько раз пытались создать при искусственном сбое создания

    def __init__(self, _pipe):
        FakeEmbedder.created += 1

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


def run_main(fault=None, pipeline="words", probe=None):
    """worker.main на синтетике. fault — точка искусственного сбоя в блоке голосов (None — без сбоя).
    pipeline="segments" — прежняя обработка: после блока голосов вызывается подменённый backends.make_asr,
    который кладёт quality из кадра main в probe и завершает задачу ошибкой (остальное в этом режиме не нужно).
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
                if _name == "embedder":
                    FakeEmbedder.failed_calls += 1
                raise RuntimeError(FAIL)
            return inner

        patches = [
            mock.patch.object(settings.paths, "STATE_DB", db),
            mock.patch.object(sys, "argv", ["worker.py", str(src), "--status", str(status)]),
            mock.patch.dict(os.environ, {"HF_TOKEN": "synthetic-token", "TEMP": str(root / "tmp"),
                                         "TMP": str(root / "tmp")}),
            mock.patch.object(worker, "wait_slot"),
            mock.patch.object(worker.tmpfiles, "sweep_stale"),
            mock.patch.object(worker, "load_config", return_value={"pipeline": pipeline, "voices": True}),
            mock.patch.object(worker, "resolve_vocab_details", return_value={
                "path": None, "vocab": {"participants": [{"name": "Анна"}, {"name": "Борис"}]},
                "group": None, "report": {}}),
            mock.patch.object(worker, "load_audio", return_value=np.zeros(60 * SR, dtype=np.float32)),
            mock.patch.object(worker, "diarize", return_value=(list(RAW), {}, None, None)),
            mock.patch.object(worker, "prepare_group_voice_profile", return_value=(ppath, prof, ["earlier.wav"])),
            mock.patch.object(worker, "run_words", fake_run_words),
            mock.patch.object(voices, "Embedder", failing("embedder") if fault == "embedder" else FakeEmbedder),
        ]
        if fault == "profile":
            patches.append(mock.patch.object(worker, "prepare_group_voice_profile", failing("profile")))
        if fault in ("centroids", "voice_vectors", "match", "references"):
            patches.append(mock.patch.object(voices, fault, failing(fault)))
        if fault == "reassign":     # прежний режим: reassign дописывает поправку и падает на следующей реплике
            def reassign_then_fail(turns, embs, pvecs, name_of, label_of, fixes, debug=None, new_label=None):
                fixes.append({"t": 0.0, "kind": "voice", "from": "Борис", "to": "Анна", "sim": [0.5, 0.6]})
                raise RuntimeError(FAIL)
            patches.append(mock.patch.object(voices, "reassign", reassign_then_fail))
        if pipeline == "segments":
            def probe_asr(_cfg):
                frame = sys._getframe(1).f_locals
                probe.update(quality=frame["quality"], turns=[dict(t) for t in frame["turns"]],
                             label_name=dict(frame["label_name"]), label_of=dict(frame["label_of"]),
                             voice_ctx=dict(frame["voice_ctx"]))
                raise RuntimeError("пробный выход после блока голосов")
            import backends
            patches.append(mock.patch.object(backends, "make_asr", probe_asr))
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
    """Д4: искусственный сбой в каждой точке блока voices в worker.main."""

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
        self.assertEqual(out.captured["quality"]["learned_from"], ["earlier.wav"])
        self.assertNotIn(WARNING, out.md)                   # без сбоя предупреждения в шапке нет
        self.assertNotIn("voices_error", out.read_json)

    def test_fault_at_any_step_leaves_the_state_as_without_voices_and_run_continues(self):
        """Д4 исправлено: блок голосов применяется одним шагом. Сбой в любой точке, в том числе поздней
        (confusions — после слияния кластеров, отчёт о сопоставлении — после сбора имён), оставляет
        метки pyannote, пустые имена и связи и пишет voices_error; прежде эти две точки оставляли
        половину: слитые метки и имена без сопоставления с профилем."""
        for fault in ("profile", "spans", "centroids", "voice_vectors", "match", "references",
                      "confusions", "report"):
            with self.subTest(fault=fault):
                out = self.run_fault(fault)
                self.assertEqual(out.status["state"], "done", out.status)    # работа продолжается
                self.assertTrue(out.captured["quality"]["voices_error"])
                self.assertTrue(is_unchanged(out.captured), out.captured)
                for key in ("match", "unmatched", "cluster_sims", "profile_names", "voice_profile", "group_doubt"):
                    self.assertNotIn(key, out.captured["quality"])
                self.assertNotIn("names_source", out.data)             # имён «по голосам» в стенограмме нет

    def test_disk_effect_of_profile_learning_is_recorded_even_when_a_later_step_fails(self):
        """prepare_group_voice_profile уже изменил профиль на диске (автопополнение из прежних записей, файл
        заменяется целиком под блокировкой группы) — результат фиксируется сразу, до шагов, которые могут упасть."""
        for fault in ("spans", "confusions", "report"):
            with self.subTest(fault=fault):
                out = self.run_fault(fault)
                self.assertEqual(out.captured["quality"]["learned_from"], ["earlier.wav"])

    def test_embedder_fault_is_the_job_error_and_embedder_is_created_once(self):
        """Embedder нужен пословной обработке всегда (опорные голоса, голос каждого слова): сбой создания —
        ошибка задачи с причиной, а не «голоса отключены»; повторного создания вне блока голосов больше нет."""
        FakeEmbedder.failed_calls = 0
        out = self.run_fault("embedder")
        self.assertEqual(FakeEmbedder.failed_calls, 1)       # прежде: один раз в блоке голосов и снова после него
        self.assertEqual(out.status["state"], "error", out.status)
        self.assertEqual(out.exit_code, 1)
        self.assertIn(FAIL, out.status["error"])
        self.assertNotIn("voices_error", (out.status.get("quality") or {}))
        FakeEmbedder.created = 0
        run_main(None).cleanup()
        self.assertEqual(FakeEmbedder.created, 1)

    def test_segments_pipeline_failure_after_fixes_discards_them_with_the_rest(self):
        """Прежний pipeline="segments": voices.reassign дописал поправку и упал — в quality.speaker_fixes
        поправок нет (прежде оставалось «исправление», которого в тексте нет), состояние «без голосов»."""
        probe = {}
        out = run_main("reassign", pipeline="segments", probe=probe)
        self.addCleanup(out.cleanup)
        self.assertEqual(probe["quality"]["speaker_fixes"], [])
        self.assertIn(FAIL, probe["quality"]["voices_error"])
        self.assertTrue(is_unchanged(probe), probe)

    def test_segments_pipeline_without_fault_applies_split_and_reassign_results(self):
        probe = {}
        out = run_main(None, pipeline="segments", probe=probe)
        self.addCleanup(out.cleanup)
        self.assertNotIn("voices_error", probe["quality"])
        self.assertEqual(probe["label_name"], {"Спикер 1": "Анна", "Спикер 3": "Борис"})
        self.assertIn("debug_sims", probe["quality"])

    def test_error_is_in_status_and_saved_json_quality_and_recording_result(self):
        out = self.run_fault("spans")
        self.assertIn(FAIL, out.status["quality"]["voices_error"])             # transcription_status
        self.assertIn(FAIL, out.data["quality"]["voices_error"])               # JSON стенограммы
        with mock.patch.object(settings.paths, "STATE_DB", out.db):
            import server
            shown = json.loads(server.read_recording_result(out.status["recording_id"]))
        self.assertIn(FAIL, shown["quality"]["voices_error"])                  # MCP read_recording_result

    def test_error_is_visible_in_the_transcript_text_a_human_reads(self):
        """Д4 исправлено: шапка Markdown (как у group_doubt) сообщает, что проверка голосов не выполнена;
        текст сбоя в Markdown не выводится (там может быть путь) — он в статусе и JSON."""
        out = self.run_fault("spans")
        header = out.md.split("**[", 1)[0]
        self.assertIn(WARNING, header)
        self.assertNotIn(FAIL, out.md)

    def test_error_is_visible_in_default_read_transcript(self):
        """Д4 исправлено: обычный read_transcript — в Markdown та же шапка (служебные разделы свёрнуты, шапка
        остаётся), в JSON — ключ voices_error рядом с asr_summary; остальной quality по-прежнему отброшен."""
        out = self.run_fault("spans")
        self.assertIn(WARNING, out.read_md.split("**[", 1)[0])
        shown = json.loads(out.read_json)
        self.assertIn(FAIL, shown["voices_error"])
        self.assertNotIn("quality", shown)


class ProfileConfusionsTests(unittest.TestCase):
    """Д4 исправлено: voices.validate_profile проверяет спутанные пары: ключ «а|б» и целые значения."""

    @staticmethod
    def with_confusions(confusions):
        prof = profile()
        prof["speakers"] = {n: {"parts": {"earlier.wav": {"sec": 100.0, "vec": [float(x) for x in v]}}}
                            for n, v in VEC.items()}
        prof["confusions"] = confusions
        return prof

    def test_valid_pairs_and_missing_or_empty_section_are_accepted(self):
        for confusions in ({}, {"earlier.wav": {}}, {"earlier.wav": {"Борис|Анна": 1, "Анна|Борис": 0}}):
            with self.subTest(confusions=confusions):
                self.assertEqual(voices.validate_profile(self.with_confusions(confusions))["confusions"], confusions)
        prof = self.with_confusions({})
        del prof["confusions"]
        voices.validate_profile(prof)

    def test_broken_pairs_are_refused_like_other_corrupt_profiles(self):
        bad = ([], {"earlier.wav": []}, {1: {"а|б": 1}}, {"earlier.wav": {"нет-разделителя": 1}},
               {"earlier.wav": {"|Анна": 1}}, {"earlier.wav": {"Борис|": 1}}, {"earlier.wav": {1: 1}},
               {"earlier.wav": {"Борис|Анна": 1.5}}, {"earlier.wav": {"Борис|Анна": True}},
               {"earlier.wav": {"Борис|Анна": "1"}}, {"earlier.wav": {"Борис|Анна": -1}})
        for confusions in bad:
            with self.subTest(confusions=confusions), self.assertRaises(voices.VoiceProfileError):
                voices.validate_profile(self.with_confusions(confusions))

    def test_recognition_continues_without_a_profile_with_broken_pairs(self):
        """Файл профиля с подпорченной записью читается как повреждённый: распознавание идёт без профиля,
        причина — в quality.voices_error, файл на диске не тронут."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "voices.json"
            prof = self.with_confusions({"earlier.wav": {"нет-разделителя": 1}})
            target.write_text(json.dumps(prof, default=lambda o: o.tolist()), encoding="utf-8")
            before = target.read_bytes()
            quality = {}
            loaded, ok = worker.load_group_profile_for_recognition(target, quality)
            self.assertFalse(ok)
            self.assertFalse(loaded["speakers"])
            self.assertIn("сохранение запрещено", quality["voices_error"])
            self.assertEqual(target.read_bytes(), before)


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


if __name__ == "__main__":
    unittest.main()
