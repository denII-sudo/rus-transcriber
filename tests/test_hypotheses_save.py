"""Сохранение гипотез моделей (РАР-4.2, шаг 3, PR Б2): настройка save_hypotheses, файл hyp-2, удаление.

Только синтетика: модели подменены (как в tests/test_run_words_golden.py), папка гипотез и соль стенда —
во временной папке, опрос компьютера и условий подменён. Главное условие: включённая настройка результат
расшифровки не меняет (отметка о записи — в статусе задачи), файл пишется только после сохранения результата.
"""
import copy
import json
import os
import shutil
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import backends
import hypotheses
import hypstore
import paths
import runinfo
import settings
import worker
from tests import test_run_words_golden as G

HW = {"status": "ok", "cpu": {"model": "synthetic"}, "devices": []}
COND = {"taken_utc": "synthetic", "cpu_load_percent": 1.0}


def _computed_once(real):
    """Настоящее вычисление сведений о сборке (запуски git, метаданные пакетов) — один раз на процесс.

    Сведения не зависят от теста, а каждое сохранение гипотез звало их заново (~70 % времени класса).
    Содержимое записи то же: значение считает настоящая функция, тесты получают его копию.
    """
    cache = {}

    def call(*args, **kwargs):
        key = repr((args, sorted(kwargs.items())))
        if key not in cache:
            cache[key] = real(*args, **kwargs)
        return copy.deepcopy(cache[key])
    return call


_CODE_REVISION_ONCE = _computed_once(runinfo.code_revision)
_PACKAGES_ONCE = _computed_once(runinfo.packages)


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="rt-hyp-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.hyp = self.root / "hypotheses"
        for patch in (mock.patch.object(paths, "HYPOTHESES", self.hyp),
                      mock.patch.object(paths, "BENCH", self.root / "bench"),
                      mock.patch.object(paths, "CACHE", self.root / "cache"),
                      mock.patch.object(runinfo, "hardware", return_value=HW),
                      mock.patch.object(runinfo, "conditions", return_value=COND),
                      mock.patch.object(runinfo, "code_revision", side_effect=_CODE_REVISION_ONCE),
                      mock.patch.object(runinfo, "packages", side_effect=_PACKAGES_ONCE)):
            patch.start()
            self.addCleanup(patch.stop)

    def files(self):
        return sorted(p for p in self.hyp.rglob("*.json") if not p.name.startswith("_")) if self.hyp.exists() else []


class SettingTests(unittest.TestCase):
    def test_setting_is_off_by_default_and_only_true_enables(self):
        self.assertNotIn("save_hypotheses", settings.DEFAULTS)
        self.assertIs(settings.CONFIG_TYPES["save_hypotheses"], bool)
        for cfg in ({}, {"save_hypotheses": False}, {"save_hypotheses": "true"}, {"save_hypotheses": 1}, None):
            self.assertFalse(hypstore.enabled(cfg))
        self.assertTrue(hypstore.enabled({"save_hypotheses": True}))

    def test_paths_constant_does_not_create_folder_on_import(self):
        self.assertEqual(paths.HYPOTHESES, paths.DATA_DIR / "hypotheses")


VOCAB = {"replace": {"борька": "Борис"}}         # словарь сценариев tests/test_run_words_golden.py


def no_seconds(obj):
    """Без замеров времени (*_sec, в том числе elapsed_sec попыток в диагностике): они меняются от прогона к прогону."""
    if isinstance(obj, dict):
        return {k: no_seconds(v) for k, v in obj.items() if not str(k).endswith("_sec")}
    if isinstance(obj, list):
        return [no_seconds(v) for v in obj]
    return obj


def finish(data, saved=None):
    """Как worker.main: служебное _hyp забирается до сохранения результата, файл пишется после.
    -> поля для статуса задачи."""
    return worker._finish_hypotheses(data.pop("_hyp", None), data["quality"], data["_timing"], VOCAB, saved)


class RunWordsWithSavingTests(Sandbox):
    def scenario(self, name, save=True, saved=None):
        """-> (данные расшифровки, журнал вызовов, поля статуса задачи)."""
        params = dict(G.SCENARIOS[name])
        params["cfg_extra"] = {**params.get("cfg_extra", {}), **({"save_hypotheses": True} if save else {})}
        data, calls = G.run_scenario(**params)
        return data, calls, finish(data, saved)

    def test_result_is_the_same_with_setting_on(self):
        for name in ("three_models", "whisper_doubt", "whisper_fails", "second_fails", "whisper_off",
                     "primary_chunk_error", "gpu_then_cpu"):
            with self.subTest(scenario=name):
                data, calls, state = self.scenario(name)
                self.assertEqual(state, {"hypotheses_saved": True})
                # настройка в cfg и отметка о записи в результат не попадают; вызовы моделей и ход статуса прежние
                self.assertEqual(G.digest(data, calls), G.EXPECTED[name])

    def test_setting_off_writes_nothing(self):
        params = dict(G.SCENARIOS["three_models"])
        data, calls = G.run_scenario(**params)
        self.assertNotIn("_hyp", data)
        self.assertEqual(finish(data), {})
        self.assertFalse(self.hyp.exists())

    def test_no_file_until_result_is_saved(self):
        """Файл пишет main после _save_worker_result: run_words сам ничего не пишет."""
        params = dict(G.SCENARIOS["three_models"])
        params["cfg_extra"] = {**params.get("cfg_extra", {}), "save_hypotheses": True}
        data, calls = G.run_scenario(**params)
        self.assertIn("_hyp", data)
        self.assertEqual(self.files(), [])

    def test_failed_task_after_combine_leaves_no_file(self):
        params = dict(G.SCENARIOS["three_models"])
        params["cfg_extra"] = {**params.get("cfg_extra", {}), "save_hypotheses": True}
        with mock.patch.object(worker, "_emotions", side_effect=RuntimeError("сбой после сведения")):
            with self.assertRaises(RuntimeError):
                G.run_scenario(**params)
        self.assertEqual(self.files(), [])
        self.assertFalse(self.hyp.exists())

    def test_main_writes_file_after_result_and_reports_in_status(self):
        src = (Path(worker.__file__)).read_text(encoding="utf-8")
        main = src[src.index("def main("):src.index("def _save_worker_result(")]
        pop, save, finish_at = (main.index('hyp = data.pop("_hyp", None)'), main.index("saved = _save_worker_result("),
                                main.index("_finish_hypotheses(hyp,"))
        self.assertLess(pop, save)
        self.assertLess(save, finish_at)
        self.assertIn("**hyp_state", main[finish_at:])

    def test_recording_id_from_saved_result(self):
        self.scenario("three_models", saved={"recording_id": "rec-synthetic-1"})
        self.scenario("whisper_off")
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in self.files()]
        ids = sorted(json.dumps(r["ids"]["recording_id"], ensure_ascii=False) for r in recs)
        self.assertIn('"rec-synthetic-1"', ids)
        other = next(r["ids"]["recording_id"] for r in recs if r["ids"]["recording_id"] != "rec-synthetic-1")
        self.assertEqual(other["status"], "not_available")

    def test_file_has_votes_of_every_model_before_vote_and_trace(self):
        self.scenario("three_models")
        (path,) = self.files()
        rec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(rec["schema"], "hyp-2")
        self.assertEqual(hypotheses.check(rec), [])
        self.assertEqual(path.parent.name, rec["ids"]["unit_id"])
        self.assertEqual(path.stem, rec["ids"]["run_id"])
        roles = {b["role"]: b for b in rec["roles"]}
        self.assertEqual(list(roles), ["primary", "second", "third"])
        primary = [w["w"] for w in roles["primary"]["words"]]
        self.assertIn("дела?", primary)                     # слово основной модели до голосования
        self.assertNotIn("дело", primary)
        self.assertIn("там", [w["w"] for w in roles["second"]["words"]])
        self.assertEqual(roles["third"]["word_times"]["status"], "not_available")     # Whisper — только текст
        self.assertTrue(all(c["heard"] for c in roles["third"]["chunks"]))
        self.assertEqual([c["removed_words"] for c in roles["third"]["chunks"]], [0, 4])
        decisions = [p["decision"] for p in rec["vote_trace"]["places"]]
        self.assertIn("replaced", decisions)
        self.assertIn("inserted", decisions)
        self.assertIn("disputed", decisions)
        self.assertEqual(rec["hardware"]["machine"], HW)
        self.assertEqual(rec["conditions"]["start"], COND)
        self.assertEqual(rec["config"]["vocab"]["entries"], 1)                # словарь — только сумма и число
        self.assertNotIn("борька", json.dumps(rec["config"], ensure_ascii=False))
        self.assertNotIn("synthetic.wav", path.read_text(encoding="utf-8"))   # имён файлов и путей нет
        self.assertTrue(runinfo.scrub(rec) == rec)

    def test_doubt_mode_marks_chunks_the_third_model_did_not_hear(self):
        self.scenario("whisper_doubt")
        (path,) = self.files()
        third = next(b for b in json.loads(path.read_text(encoding="utf-8"))["roles"] if b["role"] == "third")
        heard = [c["heard"] for c in third["chunks"]]
        self.assertIn(False, heard)
        self.assertIn(True, heard)
        self.assertTrue(all(c["raw_text"] is None for c in third["chunks"] if not c["heard"]))

    def test_missing_models_are_written_explicitly(self):
        self.scenario("second_fails")
        (path,) = self.files()
        rec = json.loads(path.read_text(encoding="utf-8"))
        second = next(b for b in rec["roles"] if b["role"] == "second")
        self.assertTrue(hypotheses.is_missing(second["words"]))
        self.assertEqual(hypotheses.check(rec), [])

    def test_failure_to_save_does_not_break_transcription(self):
        with mock.patch.object(hypstore, "write", side_effect=OSError("диск <путь> полон")):
            data, calls, state = self.scenario("three_models")
        self.assertEqual(state, {"hypotheses_saved": False, "hypotheses_error": "OSError"})   # без текста и пути
        self.assertEqual(G.digest(data, calls), G.EXPECTED["three_models"])
        self.assertEqual(self.files(), [])

    def test_no_none_inside_model_engine_timing(self):
        """Неизвестное внутри блоков model, engine, timing — not_measured, а не None (в words и chunks None —
        значение, например raw_text неслушанного куска)."""
        def nones(obj, where):
            if obj is None:
                return [where]
            if isinstance(obj, dict):
                return [p for k, v in obj.items() for p in nones(v, f"{where}.{k}")]
            if isinstance(obj, list):
                return [p for i, v in enumerate(obj) for p in nones(v, f"{where}[{i}]")]
            return []
        for name in ("three_models", "whisper_doubt", "whisper_fails", "second_fails", "whisper_off",
                     "primary_chunk_error", "gpu_then_cpu"):
            self.scenario(name)
        files = self.files()
        self.assertEqual(len(files), 7)
        for path in files:
            rec = json.loads(path.read_text(encoding="utf-8"))
            for block in rec["roles"]:
                for key in ("model", "engine", "timing"):
                    with self.subTest(role=block["role"], key=key):
                        self.assertEqual(nones(block[key], key), [])

    def test_two_runs_of_same_audio_lie_side_by_side(self):
        self.scenario("three_models")
        time.sleep(0.01)
        self.scenario("three_models")
        files = self.files()
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0].parent, files[1].parent)


class StoreTests(Sandbox):
    def put(self, unit="abcdef012345", run="20261124-100000-aaaaaa", age_days=0):
        p = hypstore.write({"schema": "hyp-2"}, unit, run, self.hyp)
        if age_days:
            t = time.time() - age_days * 86400
            os.utime(p, (t, t))
        return p

    def test_write_once_and_names_checked(self):
        self.put()
        with self.assertRaises(FileExistsError):
            self.put()
        for unit, run in (("../x", "20261124-100000-aaaaaa"), ("abcdef012345", "x"), ("ABCDEF012345", "20261124-100000-aaaaaa")):
            with self.assertRaises(ValueError):
                hypstore.write({}, unit, run, self.hyp)
        self.assertEqual(list(self.hyp.rglob("*.tmp")), [])

    def test_cleanup_report_then_apply_by_unit_and_age(self):
        a = self.put("aaaaaaaaaaaa", "20261124-100000-aaaaaa", age_days=40)
        b = self.put("bbbbbbbbbbbb", "20261124-100000-bbbbbb")
        (self.hyp / hypstore.ARTIFACTS_FILE).write_text("{}", encoding="utf-8")
        (self.hyp / hypstore.TESTBED_FILE).write_text("{}", encoding="utf-8")
        stray = self.hyp / "bbbbbbbbbbbb" / "notes.txt"
        stray.write_text("x", encoding="utf-8")
        rep = hypstore.cleanup()
        self.assertEqual((rep["files"], rep["units"], rep["applied"]), (2, 2, False))
        self.assertTrue(a.exists() and b.exists())
        self.assertEqual(hypstore.cleanup(older_than_days=30, apply=True)["files"], 1)
        self.assertFalse(a.exists())
        self.assertFalse(a.parent.exists())                       # пустая папка записи убрана
        self.assertTrue(b.exists())
        self.assertEqual(hypstore.cleanup(unit_id="bbbbbbbbbbbb", apply=True)["files"], 1)
        self.assertFalse(b.exists())
        self.assertTrue(stray.exists())                           # чужие файлы не трогаются
        self.assertTrue((self.hyp / hypstore.ARTIFACTS_FILE).exists())
        self.assertTrue((self.hyp / hypstore.TESTBED_FILE).exists())
        with self.assertRaises(ValueError):
            hypstore.cleanup(unit_id="..")

    def test_cleanup_does_not_follow_links(self):
        outside = self.root / "outside"
        (outside / "cccccccccccc").mkdir(parents=True)
        victim = outside / "cccccccccccc" / "20261124-100000-cccccc.json"
        victim.write_text("{}", encoding="utf-8")
        self.hyp.mkdir(parents=True, exist_ok=True)
        link = self.hyp / "cccccccccccc"
        try:
            os.symlink(outside / "cccccccccccc", link, target_is_directory=True)
        except (OSError, NotImplementedError):
            if os.name != "nt":
                self.skipTest("нет прав на создание ссылки")
            import subprocess        # junction в Windows создаётся без прав администратора
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside / "cccccccccccc")],
                           capture_output=True, check=False)
            if not link.exists():
                self.skipTest("не удалось создать junction")
        self.assertEqual(hypstore.cleanup(apply=True)["files"], 0)
        self.assertTrue(victim.exists())

    def test_weights_hash_is_cached_and_recomputed_on_change(self):
        model_dir = self.root / "cache" / "gigaam"
        model_dir.mkdir(parents=True)
        f = model_dir / "v3_e2e_rnnt.onnx"
        f.write_bytes(b"\x01" * (hypstore.MIN_WEIGHT_BYTES + 10))
        cache = {}
        calls = []
        real = runinfo.sha256_file
        with mock.patch.object(runinfo, "sha256_file", side_effect=lambda p: calls.append(p) or real(p)):
            first = hypstore.model_artifact("v3_e2e_rnnt", self.root / "cache", cache)
            hypstore.model_artifact("v3_e2e_rnnt", self.root / "cache", cache)
            self.assertEqual(len(calls), 1)                       # второй раз — из кэша
            f.write_bytes(b"\x02" * (hypstore.MIN_WEIGHT_BYTES + 11))
            second = hypstore.model_artifact("v3_e2e_rnnt", self.root / "cache", cache)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(first["files"][0]["sha256"], second["files"][0]["sha256"])
        self.assertEqual(first["files"][0]["file"], "gigaam/v3_e2e_rnnt.onnx")
        self.assertTrue(hypotheses.is_missing(hypstore.model_artifact("нет-такой", self.root / "cache", {})))

    def test_weight_files_are_those_the_engine_loads(self):
        """В общем хранилище gigaam-v3-onnx — только файлы своей модели; .ckpt PyTorch — только при движке torch;
        faster-whisper — своя сборка модели."""
        cache = self.root / "cache"
        snap = cache / "huggingface" / "hub"
        layout = {"models--istupakov--gigaam-v3-onnx/snapshots/s1": ["v3_e2e_ctc.int8.onnx", "v3_e2e_ctc.onnx",
                                                                     "v3_e2e_ctc_vocab.txt",
                                                                     "v3_e2e_rnnt_encoder.int8.onnx"],
                  "models--istupakov--gigaam-multilingual-large-ctc-onnx/snapshots/s2": ["multilingual_large_ctc.int8.onnx"],
                  "models--istupakov--gigaam-multilingual-ctc-onnx/snapshots/s3": ["multilingual_ctc.int8.onnx"],
                  "models--onnx-community--whisper-large-v3-turbo/snapshots/s4/onnx": ["encoder_model.onnx"],
                  "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/s5": ["model.bin"]}
        for d, names in layout.items():
            (snap / d).mkdir(parents=True)
            for n in names:
                (snap / d / n).write_bytes(b"0")
        (cache / "gigaam").mkdir()
        (cache / "gigaam" / "v3_e2e_rnnt.ckpt").write_bytes(b"0")

        def names(model, engine):
            return [rel for rel, _ in hypstore._model_files(model, cache, engine)]
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HOME", None)
            self.assertEqual(names("v3_e2e_ctc", "onnx-cpu"),
                             ["hf/istupakov/gigaam-v3-onnx/s1/v3_e2e_ctc.int8.onnx",
                              "hf/istupakov/gigaam-v3-onnx/s1/v3_e2e_ctc.onnx"])
            self.assertEqual(names("v3_e2e_rnnt", "openvino:GPU"),
                             ["hf/istupakov/gigaam-v3-onnx/s1/v3_e2e_rnnt_encoder.int8.onnx"])
            self.assertEqual(names("v3_e2e_rnnt", "torch:cpu"), ["gigaam/v3_e2e_rnnt.ckpt"])
            self.assertEqual(names("multilingual_ctc", "onnx-cpu"),
                             ["hf/istupakov/gigaam-multilingual-ctc-onnx/s3/multilingual_ctc.int8.onnx"])
            self.assertEqual(names("onnx-community/whisper-large-v3-turbo", "onnx-cpu"),
                             ["hf/onnx-community/whisper-large-v3-turbo/s4/onnx/encoder_model.onnx"])
            self.assertEqual(names("onnx-community/whisper-large-v3-turbo", "whisper-ct2:int8:t8"),
                             ["hf/mobiuslabsgmbh/faster-whisper-large-v3-turbo/s5/model.bin"])

    def test_testbed_is_cached_for_a_day(self):
        tid1, hw1 = hypstore.testbed(None, self.hyp.parent, now=1000.0)
        with mock.patch.object(runinfo, "hardware", side_effect=AssertionError("должно быть из кэша")):
            tid2, hw2 = hypstore.testbed(None, self.hyp.parent, now=1000.0 + 3600)
        self.assertEqual((tid1, hw1), (tid2, hw2))


DRIVE_PATH = "D" + ":" + "\\" + "записи" + "\\" + "встреча.wav"     # собрано из частей: не путь в тексте теста
UNC_PATH = "\\" + "\\" + "сервер" + "\\" + "общая"
MAC_PATH = "/" + "Users" + "/" + "тест" + "/a.wav"


class UnsafeRecordTests(Sandbox):
    """Перед записью: ни одна строка записи (и ключ) не похожа на абсолютный путь и не содержит имени
    исходного аудиофайла; иначе файла нет, в статусе — hypotheses_saved=false и вид ошибки без пути."""

    def record(self, **meta):
        roles = [{"role": "primary", "model": {"name": "v3_e2e_rnnt"}, "engine": {"name": "onnx-cpu"},
                  "timing": {}, "chunks": [], "words": [], "word_times": hypotheses.available()}]
        return hypotheses.build_record(roles, legacy={"combine_rule": "vote3_partial"}, combine={}, trace=None,
                                       meta=meta)

    def test_paths_and_audio_name_are_refused(self):
        home = str(Path.home())
        bad = [{"audio": {"x": DRIVE_PATH}}, {"audio": {"x": DRIVE_PATH.replace("\\", "/")}},
               {"program": {"x": UNC_PATH}}, {"environment": {"x": MAC_PATH}},
               {"environment": {"x": "/home/тест/a"}}, {"config": {"x": home + "\\модели"}},
               {"config": {"x": home.replace("\\", "/") + "/модели"}}, {"ids": {DRIVE_PATH: 1}},
               {"ids": {"note": "файл Встреча.WAV готов"}}]
        for meta in bad:
            with self.subTest(meta=list(meta)):
                with self.assertRaises(hypstore.UnsafeRecord) as cm:
                    hypstore.write(self.record(**meta), "abcdef012345", "20261124-100000-aaaaaa", self.hyp,
                                   forbid=("встреча.wav",))
                self.assertNotIn("встреча", str(cm.exception).lower())
                self.assertNotIn(home.lower(), str(cm.exception).lower())
                self.assertEqual(self.files(), [])

    def test_model_name_and_legacy_paths_are_refused(self):
        rec = self.record()
        rec["roles"][0]["model"]["name"] = DRIVE_PATH
        with self.assertRaises(hypstore.UnsafeRecord):
            hypstore.write(rec, "abcdef012345", "20261124-100000-aaaaaa", self.hyp)
        rec = self.record()
        rec["primary"] = {"model": MAC_PATH}
        with self.assertRaises(hypstore.UnsafeRecord):
            hypstore.write(rec, "abcdef012345", "20261124-100000-aaaaaa", self.hyp)
        self.assertEqual(self.files(), [])

    def test_ordinary_record_and_stem_of_audio_name_are_allowed(self):
        """Основа имени без расширения («встреча») не проверяется: она может быть словом стенограммы."""
        rec = self.record(ids={"note": "встреча прошла", "rel": "hf/istupakov/gigaam-v3-onnx/s/v3.onnx",
                               "time": "10:30", "ratio": "1:2"})
        path = hypstore.write(rec, "abcdef012345", "20261124-100000-aaaaaa", self.hyp, forbid=("встреча.wav",))
        self.assertTrue(path.is_file())

    def test_run_with_path_in_record_saves_nothing_and_reports_kind_only(self):
        with mock.patch.object(runinfo, "os_info", return_value={"name": str(Path.home() / "x")}):
            data, calls, state = RunWordsWithSavingTests.scenario(self, "three_models")
        self.assertEqual(state, {"hypotheses_saved": False, "hypotheses_error": "UnsafeRecord"})
        self.assertEqual(self.files(), [])
        with mock.patch.object(runinfo, "os_info", return_value={"name": "копия synthetic.WAV"}):
            data, calls, state = RunWordsWithSavingTests.scenario(self, "three_models")
        self.assertEqual(state["hypotheses_error"], "UnsafeRecord")
        self.assertEqual(self.files(), [])


class NotExportedTests(Sandbox):
    def test_export_with_jobs_does_not_take_hypotheses(self):
        import data_io
        g = paths.group_dir("hyp-export")
        g.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, g, ignore_errors=True)
        (g / "vocab.json").write_text('{"participants": []}', encoding="utf-8")
        real = paths.DATA_DIR / "hypotheses" / "dddddddddddd"
        real.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, paths.DATA_DIR / "hypotheses", ignore_errors=True)
        (real / "20261124-100000-dddddd.json").write_text('{"schema":"hyp-2","secret":"текст записи"}',
                                                          encoding="utf-8")
        r = data_io.export_data(dest_dir=str(self.root / "out"), group="hyp-export", with_jobs=True)
        with zipfile.ZipFile(r["zip"]) as z:
            names = z.namelist()
            blob = b"".join(z.read(n) for n in names)
        self.assertFalse([n for n in names if "hypothes" in n])
        self.assertNotIn("текст записи".encode("utf-8"), blob)

    def test_only_storage_modules_mention_the_folder(self):
        """Выгрузки, резервные копии, облако и модельные задания папку гипотез не читают."""
        root = Path(__file__).resolve().parent.parent
        allowed = {"hypotheses.py", "hypstore.py", "worker.py", "paths.py", "data_io.py", "core.py", "server.py",
                   "rt.py", "settings.py", "buildinfo.py", "backends.py", "words.py"}
        for p in sorted(root.glob("*.py")):
            if p.name in allowed:
                continue
            text = p.read_text(encoding="utf-8")
            with self.subTest(module=p.name):
                self.assertNotIn("HYPOTHESES", text)
                self.assertNotIn("hypstore", text)

    def test_check_repo_refuses_hypotheses_folder(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_repo", Path(__file__).resolve().parent.parent / "tools" / "check_repo.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertIn("hypotheses", mod.BAD_DIRS)


class CleanupToolTests(Sandbox):
    def test_core_report_and_apply(self):
        import core
        hypstore.write({}, "eeeeeeeeeeee", "20261124-100000-eeeeee", self.hyp)
        self.assertEqual(core.do_cleanup_hypotheses("", None, False)["files"], 1)
        lines = core.describe_cleanup_hypotheses("eeeeeeeeeeee")
        self.assertIn("файлов 1", lines[0])
        self.assertEqual(core.do_cleanup_hypotheses("", 0, True)["files"], 1)     # days=0 — всё сразу
        self.assertEqual(self.files(), [])
        with self.assertRaises(core.CoreError):
            core.do_cleanup_hypotheses("../..", None, False)

    def test_mcp_apply_needs_human_and_report_does_not(self):
        import asyncio
        import confirm
        try:
            import server
        except ImportError:
            self.skipTest("нет пакета mcp")
        hypstore.write({}, "ffffffffffff", "20261124-100000-ffffff", self.hyp)
        with mock.patch.object(confirm, "require", side_effect=AssertionError("отчёт без подтверждения")):
            rep = json.loads(asyncio.run(server.cleanup_hypotheses()))
        self.assertEqual((rep["files"], rep["applied"]), (1, False))
        with mock.patch.object(confirm, "require", side_effect=confirm.NotConfirmed("нет")) as req:
            self.assertEqual(asyncio.run(server.cleanup_hypotheses(apply=True)), "нет")
        req.assert_called_once()
        self.assertEqual(len(self.files()), 1)                     # без «да» ничего не удалено
        with mock.patch.object(confirm, "require", return_value="dialog"):
            asyncio.run(server.cleanup_hypotheses(apply=True))
        self.assertEqual(self.files(), [])


class TimedThirdTests(unittest.TestCase):
    """Слова со временами у третьей модели тем же вызовом: текст тот же, что у transcribe_texts."""

    def fake_asr(self):
        class TS(SimpleNamespace):
            pass

        def recognize(inputs, sample_rate=None, **kw):
            out = []
            for x in inputs:
                n = len(x)
                if n == 0:
                    raise ValueError("пусто")
                text = "" if n < 100 else f"слово{n} второе"
                toks = [] if not text else [" слово" + str(n), " второе"]
                out.append(TS(text=text, tokens=toks, timestamps=[0.1, 0.3][:len(toks)],
                              logprobs=[-0.1, -0.2][:len(toks)]))
            return out

        class Model:
            def __init__(self, ts):
                self.ts = ts

            def recognize(self, inputs, sample_rate=None, **kw):
                res = recognize(inputs, sample_rate, **kw)
                return res if self.ts else [r.text for r in res]

            def with_timestamps(self):
                return Model(True)

        asr = object.__new__(backends.OnnxASR)
        asr.model, asr.language, asr.batch = Model(False), "ru", 4
        return asr

    def test_text_and_diagnostics_equal_transcribe_texts(self):
        asr = self.fake_asr()
        pieces = [np.zeros(n, dtype=np.float32) for n in (16000, 50, 32000, 8000, 20000)]
        d1, d2 = [], []
        texts = asr.transcribe_texts(pieces, diagnostics=d1)
        timed_texts, results = backends.texts_timed(asr, pieces, diagnostics=d2)
        self.assertEqual(timed_texts, texts)

        def strip(recs):
            return [{**r, "attempts": [{k: v for k, v in a.items() if k != "elapsed_sec"} for a in r["attempts"]]}
                    for r in recs]
        self.assertEqual(strip(d2), strip(d1))
        self.assertEqual(len(results), len(pieces))

    def test_verified_models_are_gigaam_ctc_only(self):
        self.assertEqual(worker.TIMED_THIRD_VERIFIED,
                         frozenset({"multilingual_large_ctc", "multilingual_ctc", "v3_e2e_ctc"}))


class TimedThirdInRunWordsTests(Sandbox):
    """Проверенная третья модель: слова со временами в файле, результат расшифровки тот же."""

    def run_case(self, verified):
        fake = TimedThirdTests().fake_asr()
        fake.name = "onnx-cpu:fake"

        def factory(name, calls, **kw):
            return fake
        cfg = {"save_hypotheses": True, "asr_third_model": "multilingual_large_ctc", "asr_third_long_chunks": False}
        with mock.patch.object(G, "FakeTexts", factory), \
                mock.patch.object(worker, "TIMED_THIRD_VERIFIED", frozenset(verified)):
            data, calls = G.run_scenario(cfg_extra=cfg)
            self.assertEqual(finish(data), {"hypotheses_saved": True})
        return no_seconds(data)

    def test_same_result_with_and_without_timed_path(self):
        plain = self.run_case(())
        timed = self.run_case({"multilingual_large_ctc"})
        self.assertEqual(json.dumps(plain, ensure_ascii=False, default=str),
                         json.dumps(timed, ensure_ascii=False, default=str))
        files = self.files()
        self.assertEqual(len(files), 2)
        thirds = [next(b for b in json.loads(f.read_text(encoding="utf-8"))["roles"] if b["role"] == "third")
                  for f in files]
        self.assertEqual(sorted(t["word_times"]["status"] for t in thirds), ["available", "not_measured"])
        timed_block = next(t for t in thirds if t["word_times"]["status"] == "available")
        self.assertTrue(timed_block["words"])
        self.assertEqual(set(timed_block["words"][0]), {"w", "s", "e", "c"})
        other = next(t for t in thirds if t["word_times"]["status"] == "not_measured")
        self.assertIn("text_mismatch", other["word_times"]["reason"])


if __name__ == "__main__":
    unittest.main()
