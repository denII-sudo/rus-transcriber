"""Лаборатория опытов (lab/): песочница, файл опыта, возобновление, сводка без текстов.

Модели не загружаются: тяжёлые части подменены. Установщик папку lab/ не ставит, поэтому
в установленной копии (selfcheck.py) этот набор пропускается.
"""
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "lab"
HAVE_LAB = (LAB / "lab_run.py").is_file()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_snapshot import folder_snapshot  # noqa: E402
if HAVE_LAB:
    sys.path.insert(0, str(LAB))
    import lab_run  # noqa: E402
    import lab_sandbox  # noqa: E402
    import lab_spec  # noqa: E402
    import lab_summary  # noqa: E402

SECRET = "Тайнословие"           # «текст записи»: не должен попасть в сводку
NAME = "Неизвестнов"            # «имя участника»
EXPERIMENT = LAB / "experiments" / "models-arbitration.json"


def make_source(base):
    """Синтетическая настоящая папка данных: база с двумя вариантами одного звука и ещё одним звуком."""
    import storage
    data = base / "real-data"
    audio_dir = base / "audio"
    data.mkdir()
    audio_dir.mkdir()
    (data / "config.json").write_text('{"schema_version": 1, "asr_third_backend": "whisper-ct2"}',
                                      encoding="utf-8")
    group = data / "groups" / "grp"
    group.mkdir(parents=True)
    (group / "vocab.json").write_text('{"group": "grp", "participants": []}', encoding="utf-8")
    (group / "voices.json").write_text('{"version": 2, "speakers": {}, "sources": {}}', encoding="utf-8")
    a1, a2 = audio_dir / "one.wav", audio_dir / "two.wav"
    a1.write_bytes(b"RIFF-one-synthetic")
    a2.write_bytes(b"RIFF-two-synthetic")
    store = storage.initialize(data / "state.sqlite3")
    try:
        user = store.create_user("tester")
        inst = str(uuid.uuid4())
        store.bind_installation(inst, user)
        ctx = store.capture_context(inst, "human")
        (base / "docs").mkdir()
        root_id = store.register_root(base / "docs")
        for n, (audio, tag) in enumerate(((a1, "a"), (a1, "b"), (a2, ""))):
            transcript = {"duration": "00:01:00",
                          "quality": {"runtime": {"group": "grp", "timing": {
                              "asr_model": "v3_e2e_rnnt", "asr_sec": 6, "diarization_sec": 12}}},
                          "segments": [{"segment_id": f"s{n}", "speaker": NAME,
                                        "words": [{"word_id": f"w{n}", "w": SECRET}]}]}
            store.create_recording_with_document(
                str(uuid.uuid4()), os.path.normcase(os.path.abspath(audio)), transcript,
                {"review": [], "answers": {}}, str(uuid.uuid4()), ctx, root_id=root_id,
                generated_at=f"t{n}")
    finally:
        store.close()
    return data, audio_dir


class FakeEngine:
    """Подмена PipelineEngine: без моделей; считает вызовы, умеет падать на заданной паре."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.prepared, self.ran = [], []

    def base_config(self):
        import settings
        return dict(settings.DEFAULTS)

    def config_types(self):
        import settings
        return settings.CONFIG_TYPES

    def prepare_unit(self, unit, cfg):
        self.prepared.append(unit["unit_id"])
        return SimpleNamespace(unit=unit, duration=60.0, diarization_sha256="d" * 64,
                               shared_record={"label_name": {"Спикер 1": NAME}})

    def run_config(self, shared, cfg, conf):
        key = (shared.unit["unit_id"], conf["id"])
        self.ran.append(key)
        if conf["id"] in self.fail:
            raise RuntimeError(f"сбой {shared.unit['location']}")
        words = [{"w": SECRET, "s": 1.0, "e": 1.5, "c": 0.4, "flags": ["conf"]}]
        data = {"file": shared.unit["location"], "speakers": [NAME],
                "segments": [{"speaker": NAME, "text": SECRET, "words": words}],
                "review": [{"id": "r1", "text": SECRET, "speaker": NAME, "reasons": ["conf", "voice"]},
                           {"id": "r2", "text": "", "reasons": ["missed"]}],
                "quality": {"words": 1, "flagged_words": 1, "vote": {"agree3": 1, "disputed": 0},
                            "asr_diagnostics": {"primary": {"backend": "cpu", "chunks": [{"state": "ok"}]}},
                            "asr3_errors": [f"whisper-ct2:CPU: RuntimeError: {shared.unit['location']}"]}}
        hyps = {"schema": lab_run.HYP_SCHEMA, "primary": {"words": words}, "second": None,
                "third": {"chunks": [{"raw_text": SECRET, "text": SECRET}]}}
        timing = {"asr_sec": 3, "asr_model": "v3_e2e_rnnt", "diarization_sec": 12, "pair_sec": 4.0,
                  "asr3_fallbacks": [f"onnx-cpu:CPU: OSError: {shared.unit['location']}"]}
        calls = [{"requested": {"asr_model": "v3_e2e_rnnt"}, "actual": "cpu",
                  "fallbacks": [f"openvino:CPU: RuntimeError: нет {shared.unit['location']}"]}]
        return data, hyps, calls, timing


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class SandboxSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lab-safety-")
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        self.program = self.base / "program"
        self.data.mkdir()
        self.program.mkdir()
        self.protected = {"данные": self.data, "программа": self.program, "клон": ROOT}

    def tearDown(self):
        self.tmp.cleanup()

    def test_refuses_dangerous_places(self):
        for bad in (self.data, self.data / "lab", self.program / "lab", ROOT / "lab-sandbox", self.base):
            with self.subTest(bad=bad), self.assertRaises(lab_sandbox.SandboxError):
                lab_sandbox.check_sandbox(bad, self.protected)
        audio = self.base / "audio"
        with self.assertRaises(lab_sandbox.SandboxError):
            lab_sandbox.check_sandbox(audio / "lab", self.protected, [audio])

    def test_accepts_separate_folder(self):
        self.assertEqual(lab_sandbox.check_sandbox(self.base / "sandbox", self.protected),
                         lab_sandbox.real(self.base / "sandbox"))

    def test_case_insensitive_on_windows(self):
        if os.name != "nt":
            self.skipTest("регистр путей важен только в Windows")
        with self.assertRaises(lab_sandbox.SandboxError):
            lab_sandbox.check_sandbox(Path(str(self.data).upper()) / "x", self.protected)


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class ExperimentFile(unittest.TestCase):
    def test_shipped_experiment_parses_and_maps_roles(self):
        spec, sha = lab_spec.load(EXPERIMENT)
        self.assertEqual(len(sha), 64)
        ids = [c["id"] for c in spec["configurations"]]
        self.assertIn("a-current", ids)
        base = {"asr_model": "x", "asr_third_model": "onnx-community/whisper-large-v3-turbo",
                "emotions": True, "asr_third_mode": "all"}
        by_id = {c["id"]: lab_spec.effective_settings(base, c) for c in spec["configurations"]}
        self.assertEqual(by_id["a-current"]["asr_third_backend"], "whisper-ct2")
        self.assertEqual(by_id["c-no-third"]["asr_third_mode"], "off")
        self.assertEqual(by_id["b-ml3"]["asr_third_model"], "multilingual_large_ctc")
        self.assertIsNone(by_id["b-ml3"]["asr_third_quant_cpu"])
        self.assertEqual(by_id["d-ml2-whisper3"]["asr_second_model"], "multilingual_large_ctc")
        self.assertEqual(by_id["f-whisper-doubt"]["asr_third_mode"], "doubt")
        self.assertTrue(all(cfg["emotions"] is False for cfg in by_id.values()))
        hashes = {lab_spec.config_hash(c, by_id[c["id"]]) for c in spec["configurations"]}
        self.assertEqual(len(hashes), len(ids))

    def test_rejects_errors(self):
        good = {"schema": lab_spec.SCHEMA, "name": "x",
                "configurations": [{"id": "a", "roles": {"primary": {"model": "m"}, "second": {"model": "s"}}}]}
        lab_spec.parse(json.dumps(good))
        cases = {
            "shared": {"settings": {"diar_embeddings": "torch"}},
            "third without second": {"roles": {"primary": {"model": "m"}, "second": None,
                                               "third": {"engine": "whisper-ct2"}}},
            "same model": {"roles": {"primary": {"model": "m"}, "second": {"model": "m"}}},
            "unknown key": {"roles": {"primary": {"model": "m", "devise": "GPU"}}},
            "onnx on gpu": {"roles": {"primary": {"model": "m"}, "second": {"model": "s"},
                                      "third": {"engine": "onnx", "model": "q", "device": "GPU"}}},
            "bad id": {"id": "A B"},
            # вторая модель без времён слов и третья, совпадающая с другой ролью, конвейер пропускает
            "whisper second": {"roles": {"primary": {"model": "m"},
                                         "second": {"model": "onnx-community/whisper-large-v3-turbo"}}},
            "third same as second": {"roles": {"primary": {"model": "m", "quant": None},
                                               "second": {"model": "s", "backend": "onnx-cpu"},
                                               "third": {"engine": "onnx", "model": "s", "quant": None}}},
            "third same as primary": {"roles": {"primary": {"model": "m", "backend": "onnx-cpu", "quant": "int8"},
                                                "second": {"model": "s"},
                                                "third": {"engine": "onnx", "model": "m", "quant": "int8"}}},
            "long chunks not whisper": {"roles": {"primary": {"model": "m"}, "second": {"model": "s"},
                                                  "third": {"engine": "onnx", "model": "q",
                                                            "long_chunks": True}}},
        }
        for name, change in cases.items():
            bad = json.loads(json.dumps(good))
            bad["configurations"][0].update(change)
            with self.subTest(name), self.assertRaises(lab_spec.SpecError):
                lab_spec.parse(json.dumps(bad))
        # та же модель третьей, но другая точность или движок (или они берутся из config.json) — допустимо
        allowed = {
            "other quant": {"primary": {"model": "m", "backend": "onnx-cpu", "quant": None}, "second": {"model": "s"},
                            "third": {"engine": "onnx", "model": "m", "quant": "int8"}},
            "other engine": {"primary": {"model": "m", "backend": "openvino", "quant": None},
                             "second": {"model": "s"}, "third": {"engine": "onnx", "model": "m", "quant": None}},
            "from config": {"primary": {"model": "m"}, "second": {"model": "s"},
                            "third": {"engine": "onnx", "model": "s"}},
        }
        for name, roles in allowed.items():
            ok = json.loads(json.dumps(good))
            ok["configurations"][0]["roles"] = roles
            with self.subTest(name):
                lab_spec.parse(json.dumps(ok))
        import settings
        typo = json.loads(json.dumps(good))
        typo["configurations"][0]["settings"] = {"asr_thrid_mode": "all"}
        conf = lab_spec.parse(json.dumps(typo))["configurations"][0]
        with self.assertRaises(lab_spec.SpecError):
            lab_spec.effective_settings({}, conf, settings.CONFIG_TYPES)


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class PrepareAndRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lab-run-")
        self.base = Path(self.tmp.name)
        self.data, self.audio = make_source(self.base)
        self.program = self.base / "program"
        self.program.mkdir()
        self.sandbox = self.base / "sandbox"
        self.before = (folder_snapshot(self.data), folder_snapshot(self.audio))
        self.log = []

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self, voices="skip"):
        return lab_sandbox.prepare(self.sandbox, self.data, self.program, voices, env={},
                                   log=self.log.append)

    def assert_source_unchanged(self):
        self.assertEqual(folder_snapshot(self.data), self.before[0], "настоящая папка данных изменилась")
        self.assertEqual(folder_snapshot(self.audio), self.before[1], "папка с аудио изменилась")

    def test_prepare_copies_and_leaves_source_untouched(self):
        report = self.prepare("skip")
        self.assert_source_unchanged()
        self.assertEqual(report["units"], 2)
        self.assertEqual(report["recordings"], 3)
        s = lab_sandbox.real(self.sandbox)
        self.assertTrue((s / "source" / "state.sqlite3").is_file())
        self.assertFalse((s / "data" / "state.sqlite3").exists(), "копия базы не должна стать рабочей")
        self.assertTrue((s / "data" / "groups" / "grp" / "vocab.json").is_file())
        self.assertFalse((s / "data" / "groups" / "grp" / "voices.json").exists(), "профили не просили")
        db = sqlite3.connect((s / "source" / "state.sqlite3").as_uri() + "?mode=ro", uri=True)
        try:
            self.assertEqual(db.execute("SELECT locked FROM restore_guard").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM installations WHERE active=1").fetchone()[0], 0)
        finally:
            db.close()
        units = json.loads((s / "units.json").read_text(encoding="utf-8"))["units"]
        self.assertEqual(sorted(len(u["recordings"]) for u in units), [1, 2])
        self.assertTrue(all(u["group"] == "grp" for u in units))
        self.assertTrue(any("Голосовые профили не копировались" in m for m in self.log))

    def copied_method(self):
        marker = json.loads((lab_sandbox.real(self.sandbox) / lab_sandbox.MARKER).read_text(encoding="utf-8"))
        return next(c["method"] for c in marker["copied"] if c["relative"] == "source/state.sqlite3")

    def copied_user_name(self):
        import storage
        store = storage.open_store_readonly(lab_sandbox.real(self.sandbox) / "source" / "state.sqlite3")
        try:
            return store.db.execute("SELECT display_name FROM local_users").fetchone()[0]
        finally:
            store.close()

    def test_closed_database_is_copied_by_standard_readonly(self):
        import storage
        self.assertEqual(sorted(p.name for p in self.data.glob("state.sqlite3*")), ["state.sqlite3"])
        with mock.patch.object(storage, "open_store_readonly", wraps=storage.open_store_readonly) as spy:
            self.prepare()
        self.assertTrue(any(lab_sandbox.real(c.args[0]) == lab_sandbox.real(self.data / "state.sqlite3")
                            for c in spy.call_args_list), "штатное чтение источника")
        self.assertEqual(self.copied_method(), "readonly")
        self.assertFalse((lab_sandbox.real(self.sandbox) / "source" / "staging").exists())
        self.assert_source_unchanged()

    def test_live_writer_with_nonempty_wal_falls_back_to_bytes(self):
        import storage
        writer = storage.open_store(self.data / "state.sqlite3")
        try:
            writer.db.execute("UPDATE local_users SET display_name='ИЗ-ЖУРНАЛА'")
            self.assertGreater((self.data / "state.sqlite3-wal").stat().st_size, 0)
            before = folder_snapshot(self.data)
            self.prepare()
            self.assertEqual(folder_snapshot(self.data), before,
                             "при живом писателе папка данных не меняется (и -shm тоже)")
        finally:
            writer.close()
        self.assertEqual(self.copied_method(), "bytes")
        self.assertEqual(self.copied_user_name(), "ИЗ-ЖУРНАЛА", "данные из -wal попали в копию")
        self.assertFalse((lab_sandbox.real(self.sandbox) / "source" / "staging").exists())

    def test_leftover_wal_without_shm_leaves_no_new_file(self):
        import shutil
        import storage
        db = self.data / "state.sqlite3"
        keep = self.base / "kept"
        keep.mkdir()
        writer = storage.open_store(db)
        try:
            writer.db.execute("UPDATE local_users SET display_name='ПОСЛЕ-СБОЯ'")
            shutil.copyfile(db, keep / db.name)
            shutil.copyfile(self.data / "state.sqlite3-wal", keep / "state.sqlite3-wal")
        finally:
            writer.close()
        shutil.copyfile(keep / db.name, db)                      # состояние «после сбоя»: база + -wal, без -shm
        shutil.copyfile(keep / "state.sqlite3-wal", self.data / "state.sqlite3-wal")
        before = folder_snapshot(self.data)
        self.assertNotIn("state.sqlite3-shm", before)
        self.prepare()
        self.assertEqual(folder_snapshot(self.data), before, "новый -shm не появился, ничего не изменилось")
        self.assertEqual(self.copied_method(), "bytes")
        self.assertEqual(self.copied_user_name(), "ПОСЛЕ-СБОЯ")

    def test_prepare_with_voice_profiles_warns(self):
        self.prepare("copy")
        self.assertTrue((lab_sandbox.real(self.sandbox) / "data" / "groups" / "grp" / "voices.json").is_file())
        self.assertTrue(any("биометрия" in m for m in self.log))
        self.assert_source_unchanged()

    def test_prepare_refuses_nonempty_and_dangerous(self):
        self.sandbox.mkdir()
        (self.sandbox / "x.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(lab_sandbox.SandboxError):
            self.prepare()
        for bad in (self.data / "lab", self.audio / "lab", self.program / "lab"):
            with self.subTest(bad=bad), self.assertRaises(lab_sandbox.SandboxError):
                lab_sandbox.prepare(bad, self.data, self.program, "skip", env={}, log=self.log.append)
        self.assert_source_unchanged()
        self.assertFalse((self.data / "lab").exists())

    def run_lab(self, engine, **kw):
        sandbox, marker, units = lab_sandbox.load_sandbox(self.sandbox, env={})
        spec, sha = lab_spec.load(EXPERIMENT)
        return lab_run.run(sandbox, spec, sha, units, engine, log=self.log.append, **kw)

    def test_dry_run_resume_limit_and_summary(self):
        self.prepare()
        spec, _ = lab_spec.load(EXPERIMENT)
        n_conf = len(spec["configurations"])
        engine = FakeEngine()
        res = self.run_lab(engine, dry_run=True)
        self.assertEqual((res["pairs"], res["todo"]), (2 * n_conf, 2 * n_conf))
        self.assertEqual(engine.ran, [], "--dry-run ничего не считает")
        self.assertTrue(any("Оценка времени" in m for m in self.log))

        res = self.run_lab(engine, limit=3)
        self.assertEqual(res["computed"], 3)
        res = self.run_lab(FakeEngine(fail={"b-ml3"}))
        self.assertEqual((res["computed"], res["errors"]), (2 * n_conf - 3 - 1, 1))
        retry = FakeEngine()
        res = self.run_lab(retry)
        self.assertEqual(res["computed"], 1, "повторяется только пара с ошибкой")
        self.assertEqual(len(retry.ran), 1)
        again = FakeEngine()
        self.assertEqual(self.run_lab(again)["computed"], 0)
        self.assertEqual(again.ran, [])

        s = lab_sandbox.real(self.sandbox)
        summary_text = (s / "export" / "summary.json").read_text(encoding="utf-8")
        summary = json.loads(summary_text)
        self.assertEqual(summary["schema"], lab_summary.SUMMARY_SCHEMA)
        for forbidden in (SECRET, NAME, "one.wav", "two.wav", str(self.audio), str(self.base), "grp"):
            self.assertNotIn(forbidden, summary_text, forbidden)
        self.assertNotIn(str(self.audio).replace("\\", "\\\\"), summary_text)
        done = [p for p in summary["pairs"] if p["state"] == "done"]
        self.assertEqual(len(done), 2 * n_conf)
        m = done[0]["metrics"]
        self.assertEqual(m["review"]["by_reason"]["conf"], 1)
        self.assertEqual(m["review"]["by_reason"]["missed"], 1)
        self.assertEqual(m["review"]["total"], 2)
        self.assertEqual(m["asr3_errors"], ["whisper-ct2:CPU: RuntimeError"])
        self.assertEqual(m["make_asr"][0]["fallbacks"], ["openvino:CPU: RuntimeError"])
        self.assertEqual(m["hypotheses"], {"primary": 1, "third": 1})
        # тексты остаются в базе результатов песочницы
        db = lab_run.open_results(s / "results" / "lab.sqlite3")
        try:
            row = db.execute("SELECT result_json, hypotheses_json FROM pairs WHERE state='done' LIMIT 1").fetchone()
            self.assertIn(SECRET, row["result_json"])
            self.assertIn(SECRET, row["hypotheses_json"])
        finally:
            db.close()
        self.assert_source_unchanged()

    def test_changed_audio_is_not_processed(self):
        self.prepare()
        (self.audio / "two.wav").write_bytes(b"RIFF-changed")
        engine = FakeEngine()
        res = self.run_lab(engine, config_filter=["c-no-third"])
        self.assertEqual(res["computed"], 1)
        self.assertEqual(res["errors"], 1)


TRIAL_NAME = "Секретный отрывок 1"   # имя файла (с пробелами): не должно попасть в сводку и анализ
TRIAL_DIR = "папка с записью"


def write_wav(path, seconds=3.0, rate=16000):
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


class TrialEngine:
    """Подмена PipelineEngine для пробного запуска: пары — настоящий run_words с подменёнными
    моделями (тот же помощник, что в test_lab_analyze)."""

    def __init__(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_lab_analyze
        self.helper = test_lab_analyze
        self.ran = []

    def base_config(self):
        import settings
        return dict(settings.DEFAULTS)

    def config_types(self):
        import settings
        return settings.CONFIG_TYPES

    def prepare_unit(self, unit, cfg):
        return SimpleNamespace(unit=unit, src=Path(unit["location"]), duration=3.0, diarization_sha256="d" * 64,
                               shared_record={"duration_sec": 3.0})

    def run_config(self, shared, cfg, conf):
        self.ran.append((shared.unit["unit_id"], conf["id"]))
        _, data, hyps, _ = self.helper.run_pair(conf["id"])
        data["file"] = str(shared.src)
        return data, hyps, [], data["timing"]


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class TrialOnAudioFile(unittest.TestCase):
    """Пробный запуск на произвольном файле: prepare --audio/--only-audio, add-audio, run, summary, analyze."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lab-trial-")
        self.base = Path(self.tmp.name)
        self.data, self.audio = make_source(self.base)
        self.trial = write_wav(self.base / TRIAL_DIR / f"{TRIAL_NAME}.wav")
        self.program = self.base / "program"
        self.program.mkdir()
        self.sandbox = self.base / "sandbox"
        self.log = []
        self.before = (folder_snapshot(self.data), folder_snapshot(self.audio), folder_snapshot(self.trial.parent))
        self.experiment = self.base / "trial-experiment.json"
        self.experiment.write_text(json.dumps({
            "schema": lab_spec.SCHEMA, "name": "trial",
            "configurations": [{"id": "a-current", "roles": {
                                    "primary": {"model": "v3_e2e_rnnt"}, "second": {"model": "v3_e2e_ctc"},
                                    "third": {"engine": "whisper-ct2", "mode": "all"}}},
                               {"id": "c-no-third", "roles": {
                                   "primary": {"model": "v3_e2e_rnnt"}, "second": {"model": "v3_e2e_ctc"},
                                   "third": None}}]}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_lab(self, engine, **kw):
        sandbox, marker, units = lab_sandbox.load_sandbox(self.sandbox, env={})
        spec, sha = lab_spec.load(self.experiment)
        return lab_run.run(sandbox, spec, sha, units, engine, log=self.log.append, **kw)

    def test_only_audio_full_scenario_without_names_in_exports(self):
        import lab_analyze
        lab_sandbox.prepare(self.sandbox, self.data, self.program, "skip", env={}, log=self.log.append,
                            audio=[self.trial], only_audio=True)
        s = lab_sandbox.real(self.sandbox)
        self.assertFalse((s / "source" / "state.sqlite3").exists(), "--only-audio: база записей не копируется")
        units = json.loads((s / "units.json").read_text(encoding="utf-8"))["units"]
        self.assertEqual(len(units), 1)
        u = units[0]
        self.assertEqual((u["origin"], u["recordings"], u["group"]), ("audio_file", [], None))
        self.assertAlmostEqual(u["duration_sec"], 3.0, places=1)
        copy_path = s / "audio" / f"{TRIAL_NAME}.wav"
        self.assertEqual(lab_sandbox.real(u["location"]), copy_path, "считается копия в audio/ песочницы")
        self.assertEqual(lab_sandbox.real(u["source_location"]), lab_sandbox.real(self.trial))
        self.assertEqual(copy_path.read_bytes(), self.trial.read_bytes())
        self.assertTrue(any("единственная группа" in m for m in self.log), self.log)

        res = self.run_lab(TrialEngine(), dry_run=True)
        self.assertEqual((res["pairs"], res["todo"]), (2, 2))
        self.assertTrue(any("[файл вне базы]" in m for m in self.log))
        engine = TrialEngine()
        self.assertEqual(self.run_lab(engine)["computed"], 2)
        again = TrialEngine()
        self.assertEqual(self.run_lab(again)["computed"], 0, "возобновление: посчитанное не повторяется")

        sandbox, marker, units_doc = lab_sandbox.load_sandbox(self.sandbox, env={})
        analysis = lab_analyze.analyze(sandbox, units_doc, log=self.log.append)
        self.assertEqual(analysis["human"]["not_measured"], lab_analyze.NO_HUMAN)
        self.assertEqual(analysis["human"]["units_not_measured"], {u["unit_id"]: lab_analyze.NO_HUMAN})
        self.assertEqual(analysis["data"]["units_with_pairs"], 1)
        report = (s / "results" / "report.md").read_text(encoding="utf-8")
        self.assertIn("решений человека нет", report)
        summary = (s / "export" / "summary.json").read_text(encoding="utf-8")
        self.assertIn('"origin": "audio_file"', summary)
        for path in (s / "export" / "summary.json", s / "export" / "analysis.json"):
            text = path.read_text(encoding="utf-8")
            for forbidden in (TRIAL_NAME, f"{TRIAL_NAME}.wav", TRIAL_DIR, str(self.base),
                              json.dumps(str(self.base))[1:-1], "четыре", "шесть", "Неизвестнов"):
                self.assertNotIn(forbidden, text, f"{path.name}: {forbidden}")
        self.assertEqual(folder_snapshot(self.data), self.before[0])
        self.assertEqual(folder_snapshot(self.audio), self.before[1])
        self.assertEqual(folder_snapshot(self.trial.parent), self.before[2], "пробный файл только читается")

    def test_add_audio_to_existing_sandbox_and_run_only_it(self):
        lab_sandbox.prepare(self.sandbox, self.data, self.program, "skip", env={}, log=self.log.append)
        ids = lab_sandbox.add_audio_units(self.sandbox, [self.trial], group="grp", env={}, log=self.log.append)
        copy_path = lab_sandbox.real(self.sandbox) / "audio" / self.trial.name
        stamp = copy_path.stat().st_mtime_ns
        self.assertEqual(lab_sandbox.add_audio_units(self.sandbox, [self.trial], env={}, log=self.log.append), ids,
                         "тот же звук повторно не добавляется")
        self.assertEqual(copy_path.stat().st_mtime_ns, stamp, "повторное добавление не копирует заново")
        units = json.loads((lab_sandbox.real(self.sandbox) / "units.json").read_text(encoding="utf-8"))["units"]
        self.assertEqual(len(units), 3)
        self.assertEqual([u["group"] for u in units if u["unit_id"] == ids[0]], ["grp"])
        engine = TrialEngine()
        res = self.run_lab(engine, unit_filter=ids)
        self.assertEqual(res["computed"], 2)
        self.assertEqual({uid for uid, _ in engine.ran}, set(ids), "остальные записи не считаются")

        import lab_analyze
        sandbox, marker, units_doc = lab_sandbox.load_sandbox(self.sandbox, env={})
        analysis = lab_analyze.analyze(sandbox, units_doc, log=self.log.append)
        self.assertEqual(analysis["human"]["not_measured"], lab_analyze.NO_HUMAN)

        # копия изменилась после добавления — контрольная сумма перед прогоном не сходится, пара не считается
        self.assertEqual(folder_snapshot(self.trial.parent), self.before[2], "исходный файл только читается")
        write_wav(copy_path, seconds=2.0)
        sandbox, marker, units_doc = lab_sandbox.load_sandbox(self.sandbox, env={})
        spec, sha = lab_spec.load(self.experiment)
        spec["configurations"][1]["settings"] = {"review_missed": True}      # новая версия конфигурации
        res = lab_run.run(sandbox, spec, sha, units_doc, TrialEngine(), log=self.log.append,
                          unit_filter=ids, config_filter=["c-no-third"])
        self.assertEqual((res["computed"], res["errors"]), (0, 1))

    def test_foreign_file_with_same_name_is_not_overwritten(self):
        lab_sandbox.prepare(self.sandbox, self.data, self.program, "skip", env={}, log=self.log.append,
                            audio=[self.trial], only_audio=True)
        other = write_wav(self.base / "другая папка" / self.trial.name, seconds=1.0)
        copy_path = lab_sandbox.real(self.sandbox) / "audio" / self.trial.name
        before = copy_path.read_bytes()
        with self.assertRaises(lab_sandbox.SandboxError):
            lab_sandbox.add_audio_units(self.sandbox, [other], env={}, log=self.log.append)
        self.assertEqual(copy_path.read_bytes(), before, "чужой файл с тем же именем не перезаписан")
        units = json.loads((lab_sandbox.real(self.sandbox) / "units.json").read_text(encoding="utf-8"))["units"]
        self.assertEqual(len(units), 1)

    def test_trial_audio_safety(self):
        # песочница в папке исходного файла допустима: файл лишь копируется в audio/ песочницы
        inside = self.trial.parent / "лаб песочница"
        lab_sandbox.prepare(inside, self.data, self.program, "skip", env={}, log=self.log.append,
                            audio=[self.trial], only_audio=True)
        lab_sandbox.load_sandbox(inside, env={})
        # а с папками записей из базы — по-прежнему нет
        with self.assertRaises(lab_sandbox.SandboxError):
            lab_sandbox.prepare(self.audio / "lab", self.data, self.program, "skip", env={},
                                log=self.log.append, audio=[self.trial])
        with self.assertRaises(lab_sandbox.SandboxError):
            lab_sandbox.prepare(self.sandbox, self.data, self.program, "skip", env={}, log=self.log.append,
                                only_audio=True)
        with self.assertRaises(lab_sandbox.SandboxError):
            lab_sandbox.prepare(self.sandbox, self.data, self.program, "skip", env={}, log=self.log.append,
                                audio=[self.base / "нет.wav"], only_audio=True)
        lab_sandbox.prepare(self.sandbox, self.data, self.program, "skip", env={}, log=self.log.append,
                            audio=[self.trial], only_audio=True)
        with self.assertRaises(lab_sandbox.SandboxError):
            lab_sandbox.add_audio_units(self.sandbox, [self.trial], group="../x", env={}, log=self.log.append)
        # единица внешнего файла, указывающая не на audio/ песочницы, — отказ при загрузке
        upath = lab_sandbox.real(self.sandbox) / "units.json"
        doc = json.loads(upath.read_text(encoding="utf-8"))
        doc["units"][0]["location"] = str(self.trial)
        upath.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(lab_sandbox.SandboxError):
            lab_sandbox.load_sandbox(self.sandbox, env={})
        self.assertEqual(folder_snapshot(self.trial.parent)[self.trial.name], self.before[2][self.trial.name],
                         "исходный файл только читается")


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class SoundHash(unittest.TestCase):
    """Хэш звука записи лаборатории — то же правило, что у теста скорости и сохранённых гипотез."""

    def test_same_as_audio_identity(self):
        import numpy as np
        import hypstore
        import runinfo
        audio = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)
        ident = runinfo.audio_identity(Path(tempfile.gettempdir()) / "rt-no-such-file.wav",
                                       hypstore.pcm_bytes(audio), 16000)
        self.assertEqual(lab_run.pcm_sha256(audio), ident["excerpt"]["pcm_sha256"])
        self.assertTrue(ident["excerpt"]["pcm_sha256"].startswith(ident["unit_id"]))


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class CaptureWithVoteTrace(unittest.TestCase):
    """Сведение с протоколом (trace) идёт через words.vote_n мимо vote3: Capture записывает то же, что без него,
    и mark_alt на исключённых отрезках не затирает способ сведения."""

    def capture(self, trace, whisper_fails=True):
        import backends
        import words
        from tests.test_combine_audit_fixes import diag_with_failed_whisper
        from tests.test_combine_n_models import ok_diag, ten, text
        diag = diag_with_failed_whisper("primary", "second") if whisper_fails else ok_diag("primary", "second",
                                                                                           "whisper")
        cap = lab_run.Capture(words, backends)
        with cap:
            words.combine_hypotheses({"primary": ten(), "second": ten({4: "пятак"}), "whisper": text(ten({1: "двое"}))},
                                     diag, **({"trace": {}} if trace else {}))
        return cap.vote

    def test_same_capture_with_and_without_trace(self):
        for fails in (True, False):
            with self.subTest(whisper_fails=fails):
                plain, traced = self.capture(False, fails), self.capture(True, fails)
                self.assertEqual(plain["function"], "vote3")
                self.assertEqual(traced, plain)
        self.assertEqual(self.capture(True)["mark_alt_spans"], 1)


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class CaptureWithRealPipeline(unittest.TestCase):
    """Перехват гипотез вокруг настоящего worker.run_words (модели подменены, как в test_whisper_ct2)."""

    def test_hypotheses_of_all_passes_are_captured(self):
        data, hyps, calls, timing = self.run_pair()
        self.check_capture(data, hyps, calls, timing)

    def test_save_hypotheses_setting_does_not_change_lab_record(self):
        """save_hypotheses в config.json песочницы: лаборатория выключает её у себя, запись лаборатории та же."""
        import hypstore
        with mock.patch.object(hypstore, "write", side_effect=AssertionError("лаборатория не пишет hypotheses\\")):
            plain = self.run_pair()
            with_setting = self.run_pair({"save_hypotheses": True})
        self.check_capture(*with_setting)
        self.assertEqual(json.dumps(plain[1], ensure_ascii=False, sort_keys=True),
                         json.dumps(with_setting[1], ensure_ascii=False, sort_keys=True))
        self.assertNotIn("_hyp", with_setting[0])
        self.assertEqual(plain[2], with_setting[2])

    def run_pair(self, cfg_extra=None):
        import numpy as np
        import backends
        import words
        import worker

        class Main:
            def __init__(self, model):
                self.name, self.model_name = "fake-" + model, model

            def transcribe_timestamped(self, pieces, progress=None, diagnostics=None):
                diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
                word = " раз" if self.model_name == "primary" else " два"
                return [SimpleNamespace(tokens=[word], timestamps=[0.1], logprobs=[-0.1], text=word)
                        for _ in pieces]

        class CT2:
            name, compile_sec = "fake-ct2", None

            def __init__(self, *a, **kw):
                pass

            def transcribe_texts(self, pieces, progress=None, diagnostics=None):
                diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
                return ["три три" for _ in pieces]

        def fake_make_asr(cfg):
            return Main(cfg["asr_model"]), ["openvino:GPU: RuntimeError: нет устройства"]

        engine = object.__new__(lab_run.PipelineEngine)
        engine.allow_torch, engine.log = False, lambda m: None
        chunks = [{"a": 0.0, "b": 1.0, "own": (0.0, 1.0)}, {"a": 1.0, "b": 3.0, "own": (1.0, 3.0)}]
        shared = SimpleNamespace(
            unit={"unit_id": "u1"}, src=Path("synthetic.wav"), audio=np.zeros(48000, dtype=np.float32),
            duration=3.0, raw=[(0.0, 3.0, "s")], order={"s": "L"}, turns=[], full_ann=None,
            emb=SimpleNamespace(seconds=0.0), emb_shared_sec=0.0, label_name={}, label_of={},
            voice_ctx={"cents": {"L": [1]}}, allowed=[], genders_by_name={}, vocab=None,
            quality={"speaker_fixes": [], "vocab_fixes": []}, no_profile=False, group="g",
            t_diar=1.0, t_voice=0.5, diar_info={})
        cfg = {"asr_model": "primary", "asr_second_model": "second", "asr_third_model": "onnx-community/whisper-large-v3-turbo",
               "asr_third_device": "CPU", "asr_third_backend": "whisper-ct2", "asr_third_mode": "all",
               "asr_third_long_chunks": False, "asr_third_quant_cpu": "int8", "asr_third_batch": 4,
               "emotions": False, "review_missed": False, "interaction": False, **(cfg_extra or {})}
        conf = {"id": "a-current", "roles": {}}
        with mock.patch.object(backends, "make_asr", fake_make_asr), \
             mock.patch.object(backends, "WhisperCT2", CT2), \
             mock.patch.object(words, "make_chunks", return_value=chunks), \
             mock.patch.object(words, "assign_voices", return_value=0), \
             mock.patch.object(words, "build_replies", return_value=[]), \
             mock.patch.object(words, "overlap_regions", return_value=[]), \
             mock.patch.object(words, "doubt_flags", return_value=0), \
             mock.patch.object(words, "review_items", return_value=[]):
            orig_transcribe = words.transcribe
            result = engine.run_config(shared, cfg, conf)
            self.assertIs(words.transcribe, orig_transcribe, "подмены сняты после вызова")
        return result

    def check_capture(self, data, hyps, calls, timing):
        self.assertEqual([w["w"] for w in hyps["primary"]["words"]], ["раз", "раз"])
        self.assertEqual([w["w"] for w in hyps["second"]["words"]], ["два", "два"])
        self.assertEqual(hyps["primary"]["model"], "primary")
        self.assertEqual(len(hyps["primary"]["chunks"]), 2)
        self.assertEqual([c["raw_text"] for c in hyps["third"]["chunks"]], ["три три", "три три"])
        self.assertEqual(hyps["vote"]["function"], "vote3")
        self.assertEqual(hyps["vote"]["third_words"], ["три", "три", "три", "три"])
        import hypotheses            # общий сборщик программы: запись hyp-2, верхние ключи lab-hyp-1 те же
        self.assertEqual(hyps["schema"], lab_run.HYP_SCHEMA)
        self.assertEqual(hypotheses.check(hyps), [])
        self.assertEqual([b["role"] for b in hyps["roles"]], ["primary", "second", "third"])
        self.assertEqual(hyps["roles"][1]["words"], hyps["second"]["words"])
        self.assertTrue(hypotheses.is_missing(hyps["roles"][2]["word_times"]))
        self.assertEqual(hypotheses.read(json.loads(json.dumps(hyps)))["roles"][0]["role"], "primary")
        self.assertEqual([c["actual"] for c in calls], ["fake-primary", "fake-second"])
        self.assertIn("asr_sec", timing)
        self.assertIn("asr3_sec", timing)
        self.assertEqual(timing["diarization_sec"], 1)
        self.assertIsInstance(data["quality"]["vote"], dict)
        m = lab_summary.pair_metrics(data, hyps, calls, timing, 3.0)
        self.assertEqual(m["make_asr"][0]["fallbacks"], ["openvino:GPU: RuntimeError"])
        self.assertNotIn("раз", json.dumps(m, ensure_ascii=False))

    def test_torch_fallback_is_blocked(self):
        import backends
        import words
        original = backends.TorchASR
        cap = lab_run.Capture(words, backends)
        with cap:
            with self.assertRaises(RuntimeError):
                backends.TorchASR("v3_e2e_rnnt")
        self.assertIs(backends.TorchASR, original)


# ── числа опытов для базы теста скорости: выгрузка, копирование, загрузка ────────────────────────────

MARK = "МеткаТекстаЛабы"            # попадать в базу теста скорости не должна нигде
PASS_PATH = "C:\\Users\\Иван\\Секретная папка\\python.exe"
COND = {"power_source": "ac", "cpu_load_percent": 1.0, "memory": {"free_mb": 8000}, "other_heavy_jobs": 0}
PASSPORT = {
    "schema_version": 1, "lab_method": "lab-run-1", "testbed_id": "0123456789abcdef",
    "hardware": {"status": "ok", "cpu": {"model": "Synthetic CPU 9000", "cores": 6, "threads": 12,
                                         "features": ["avx2"]},
                 "devices": [{"name": "Synthetic GPU", "driver_version": "1.2.3"}]},
    "environment": {"os": {"system": "Windows", "build": "26100", "ubr": 1},
                    "python": {"version": "3.12.0", "implementation": "CPython", "executable": PASS_PATH},
                    "packages": {"key": {"onnxruntime": "1.20.0"}, "count": 80, "sha256": "ab" * 32},
                    "program_version": "2026.11.25.1",
                    "code": {"files_sha256": "cd" * 8, "git_revision": "e" * 40, "git_dirty": False}},
    "models_cache": {"hf": [{"repo": "onnx-community/whisper-large-v3-turbo", "ref_main": "r" * 40,
                             "snapshots": ["r" * 40], "files": 3, "size_mb": 100.0}],
                     "local": [{"file": "gigaam/v3_e2e_rnnt.onnx", "size_bytes": 5_000_000,
                                "modified_utc": "2026-11-01T00:00:00Z"}]},
    "offline": {"HF_HUB_OFFLINE": "1"}, "settings_base": {"asr_model": SECRET, "note": MARK},
}


class BenchEngine(FakeEngine):
    """FakeEngine, у которого общий этап даёт хэш звука (как настоящий, с №179)."""

    def prepare_unit(self, unit, cfg):
        self.prepared.append(unit["unit_id"])
        pcm = hashlib.sha256(("pcm|" + unit["audio_sha256"]).encode()).hexdigest()
        return SimpleNamespace(unit=unit, duration=60.0, diarization_sha256="d" * 64,
                               shared_record={"label_name": {"Спикер 1": NAME}, "pcm_sha256": pcm,
                                              "duration_sec": 60.0, "voices": {"profile_names": [NAME]}})


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class BenchExport(unittest.TestCase):
    """export-bench: числа опытов лаборатории -> файлы -> bench\\lab папки данных -> база теста скорости."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lab-bench-")
        self.base = Path(self.tmp.name)
        self.data, self.audio = make_source(self.base)
        self.program = self.base / "program"
        self.program.mkdir()
        self.sandbox = self.base / "sandbox"
        self.log = []
        lab_sandbox.prepare(self.sandbox, self.data, self.program, "skip", env={}, log=self.log.append)
        sandbox, self.marker, self.units = lab_sandbox.load_sandbox(self.sandbox, env={})
        self.s = sandbox
        self.spec, self.sha = lab_spec.load(EXPERIMENT)
        self.out = self.s / "export" / "bench"
        self.target = self.base / "data2"                 # «настоящая» папка данных для проверки копирования
        self.target.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_lab(self, engine=None, **kw):
        return lab_run.run(self.s, self.spec, self.sha, self.units, engine or BenchEngine(),
                           log=self.log.append, passport_fn=lambda: PASSPORT, conditions_fn=lambda: COND, **kw)

    def results_db(self):
        return lab_run.open_results(self.s / "results" / "lab.sqlite3")

    def export(self):
        db = self.results_db()
        try:
            return lab_summary.export_bench(db, self.units["units"], self.out)
        finally:
            db.close()

    def load(self, bench_dir):
        import benchdb
        return benchdb.apply_load(bench_dir)

    def rows(self, sql, *args):
        import benchdb
        con = sqlite3.connect(benchdb.db_path(self.target / "bench"))
        con.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in con.execute(sql, args)]
        finally:
            con.close()

    def dump(self):
        import benchdb
        con = sqlite3.connect(benchdb.db_path(self.target / "bench"))
        try:
            return "\n".join(con.iterdump())
        finally:
            con.close()

    def full_flow(self, **kw):
        res = self.run_lab(BenchEngine(**kw))
        rep = self.export()
        cp = lab_summary.copy_bench_files(self.out, self.target, apply=True)
        return res, rep, cp, self.load(self.target / "bench")

    def test_numbers_reach_the_speed_database_as_lab_rows(self):
        import benchdb
        n_conf = len(self.spec["configurations"])
        res, rep, cp, loaded = self.full_flow(fail={"b-ml3"})
        name = f"lab-{res['run_id']}.json"
        self.assertEqual(rep["written"], [name])
        self.assertEqual(cp["new"], [name])
        self.assertEqual(loaded["counts"], {"loaded": 1})
        self.assertTrue((self.target / "bench" / "lab" / name).is_file())
        self.assertFalse((self.target / "bench" / "runs").exists(), "замеры скорости лаборатория не трогает")
        run = self.rows("SELECT * FROM run")[0]
        self.assertEqual((run["kind"], run["method_version"], run["run_id"]), ("lab", "lab-1", "lab-" + res["run_id"]))
        self.assertEqual((run["status"], run["valid"]), ("done", "yes"))
        ms = self.rows("SELECT * FROM measurement ORDER BY measurement_id")
        self.assertEqual(len(ms), 2 * n_conf)
        self.assertEqual({m["mode"] for m in ms}, {"pipeline"})
        self.assertEqual(sum(1 for m in ms if m["status"] == "error"), 2)
        self.assertEqual({m["status_reason"] for m in ms if m["status"] == "error"}, {"RuntimeError"})
        for m in ms:
            for col in ("load_sec", "first_chunk_sec", "asr_sec", "rtfx", "wall_sec", "total_10min"):
                self.assertIsNone(m[col], f"{col}: у опыта лаборатории столбец скорости замера пуст")
            self.assertEqual(set(json.loads(m["not_measured_json"])),
                             {"load_sec", "first_chunk_sec", "asr_sec", "rtfx", "wall_sec", "total_10min"})
            self.assertEqual((m["family"], m["unit_pk"] != ""), ("lab", True))
        # числа в extra_json.lab — те же, что в metrics_json базы лаборатории, по списку разрешённых полей
        db = self.results_db()
        try:
            expected = sorted(json.dumps(benchdb.lab_numbers(json.loads(r["metrics_json"])), sort_keys=True)
                              for r in db.execute("SELECT metrics_json FROM pairs WHERE state='done'"))
        finally:
            db.close()
        got = sorted(json.dumps(json.loads(m["extra_json"])["lab"], sort_keys=True)
                     for m in ms if m["status"] == "done")
        self.assertEqual(got, expected)
        first = json.loads([m for m in ms if m["status"] == "done"][0]["extra_json"])["lab"]
        self.assertEqual((first["timing"]["pair_sec"], first["review"]["total"], first["review"]["by_reason"]["conf"]),
                         (4.0, 2, 1))
        self.assertEqual(first["asr3_errors"], ["whisper-ct2:CPU: RuntimeError"])
        # единица — хэш звука; хэш файла записи при этом сохранён
        units = self.rows("SELECT * FROM test_unit")
        self.assertEqual(len(units), 2)
        by_audio = {u["audio_sha256"]: u for u in self.units["units"]}
        for u in units:
            self.assertEqual(u["unit_id"], u["unit_pk"][:12])
            self.assertIn(u["audio_sha256"], by_audio)
            self.assertEqual(u["unit_pk"], hashlib.sha256(("pcm|" + u["audio_sha256"]).encode()).hexdigest())
            self.assertNotEqual(u["unit_id"], by_audio[u["audio_sha256"]]["unit_id"], "unit_id лаборатории — хэш файла")
        self.assertEqual(self.rows("SELECT testbed_id FROM testbed"), [{"testbed_id": "0123456789abcdef"}])
        self.assertEqual(len(self.rows("SELECT * FROM model_artifact")), 2)
        self.assertEqual(len(self.rows("SELECT * FROM exec_config WHERE family='lab'")), n_conf)

    def test_repeat_changes_nothing(self):
        res, rep, cp, loaded = self.full_flow()
        name = f"lab-{res['run_id']}.json"
        first_bytes = (self.out / name).read_bytes()
        before = self.dump()
        rep2 = self.export()
        self.assertEqual((rep2["written"], rep2["unchanged"]), ([], [name]))
        self.assertEqual((self.out / name).read_bytes(), first_bytes, "повторная выгрузка даёт те же байты")
        cp2 = lab_summary.copy_bench_files(self.out, self.target, apply=True)
        self.assertEqual((cp2["new"], cp2["same"], cp2["conflict"]), ([], [name], []))
        self.assertEqual(self.load(self.target / "bench")["counts"], {"known": 1})
        self.assertEqual(self.dump(), before, "повторные выгрузка, копирование и загрузка ничего не дописывают")

    def test_dry_copy_writes_nothing_and_foreign_content_is_not_overwritten(self):
        res = self.run_lab()
        self.export()
        name = f"lab-{res['run_id']}.json"
        plan = lab_summary.copy_bench_files(self.out, self.target, apply=False)
        self.assertEqual(plan["new"], [name])
        self.assertFalse((self.target / "bench").exists(), "без apply в папку данных ничего не пишется")
        dest = self.target / "bench" / "lab"
        dest.mkdir(parents=True)
        (dest / name).write_bytes(b"{}")
        cp = lab_summary.copy_bench_files(self.out, self.target, apply=True)
        self.assertEqual((cp["new"], cp["conflict"]), ([], [name]))
        self.assertEqual((dest / name).read_bytes(), b"{}", "чужое содержимое под тем же именем не перезаписано")
        self.assertEqual([p.name for p in dest.iterdir()], [name], "временных файлов не осталось")
        with self.assertRaises(ValueError):
            lab_summary.copy_bench_files(self.out, self.base / "нет-такой-папки", apply=True)

    def test_running_run_is_not_exported(self):
        self.run_lab()
        db = self.results_db()
        try:
            db.execute("INSERT INTO runs (run_id, started_utc, state, experiment_name) VALUES "
                       "('20261125-000000-ffffff', '2026-11-25T00:00:00Z', 'running', 'x')")
            db.commit()
        finally:
            db.close()
        rep = self.export()
        self.assertEqual(rep["running"], 1)
        self.assertEqual(len(rep["written"]), 1)
        self.assertFalse(any("ffffff" in p.name for p in self.out.iterdir()))

    def test_interrupted_run_is_exported_with_reservation(self):
        self.run_lab()
        db = self.results_db()
        try:
            db.execute("UPDATE runs SET state='interrupted'")
            db.commit()
        finally:
            db.close()
        self.export()
        cp = lab_summary.copy_bench_files(self.out, self.target, apply=True)
        self.assertEqual(self.load(self.target / "bench")["counts"], {"loaded": 1})
        run = self.rows("SELECT status, valid, valid_reason FROM run")[0]
        self.assertEqual((run["status"], run["valid"], run["valid_reason"]), ("interrupted", "with_reservations",
                                                                              "interrupted"))
        self.assertEqual(len(cp["new"]), 1)

    def test_no_texts_names_or_paths_anywhere(self):
        """Метки расставлены во всех текстовых местах базы лаборатории; в файле и в базе теста скорости их нет."""
        self.run_lab()
        db = self.results_db()
        try:
            row = db.execute("SELECT pair_id, metrics_json FROM pairs WHERE state='done' LIMIT 1").fetchone()
            m = json.loads(row["metrics_json"])
            m["transcript_text"] = MARK                                   # новое текстовое поле лаборатории
            m["timing"]["note"] = MARK
            m["timing"]["asr_model"] = f"{MARK} with spaces"              # не название из кода
            m["make_asr"][0]["requested"]["initial_prompt"] = MARK
            m["vote3_skipped"] = MARK + " слова"
            m["asr3_errors"].append(f"RuntimeError: {MARK} {self.audio}")
            m["asr2_error"] = f"ValueError: {MARK} слова"
            m["review"]["items_text"] = [MARK]
            db.execute("UPDATE pairs SET metrics_json=? WHERE pair_id=?", (json.dumps(m, ensure_ascii=False),
                                                                           row["pair_id"]))
            db.execute("UPDATE unit_runs SET shared_json=json_set(shared_json, '$.note', ?)", (MARK,))
            db.commit()
        finally:
            db.close()
        rep = self.export()
        self.assertEqual(rep["refused"], [])
        cp = lab_summary.copy_bench_files(self.out, self.target, apply=True)
        self.assertEqual(self.load(self.target / "bench")["counts"], {"loaded": 1})
        files = sorted(self.out.glob("*.json"))
        text = files[0].read_text(encoding="utf-8")
        scan = text + "\n" + self.dump()
        forbidden = (MARK, SECRET, NAME, "one.wav", "two.wav", str(self.audio), str(self.base), "grp",
                     "Иван", "Секретная", "Как сейчас", "real-data", "\\Users\\")
        for f in forbidden:
            self.assertNotIn(f, scan, f)
            self.assertNotIn(f.replace("\\", "\\\\"), scan, f)
        self.assertIn("<путь>", text, "путь из паспорта заменён, а не оставлен")
        self.assertEqual(cp["rejected"], [])
        # разрешённое осталось: тип ошибки без сообщения
        self.assertIn("RuntimeError", scan)

    def test_path_that_scrub_missed_refuses_the_file(self):
        """Вторая линия защиты: строка, похожая на путь, не доходит до файла, даже если очистка её пропустила."""
        import runinfo
        self.run_lab()
        with mock.patch.object(runinfo, "scrub", lambda value: value):
            rep = self.export()
        self.assertEqual(len(rep["refused"]), 1)
        self.assertIn("absolute_path", rep["refused"][0]["problems"])
        self.assertFalse(self.out.exists() and list(self.out.glob("*.json")), "файл с путём не записан")

    def test_testbed_id_of_sandbox_equals_the_data_folder_one(self):
        import runinfo
        (self.data / "bench").mkdir(exist_ok=True)
        (self.data / "bench" / "testbed.json").write_text('{"schema": 1, "salt": "соль-стенда"}', encoding="utf-8")
        second = self.base / "sandbox2"
        lab_sandbox.prepare(second, self.data, self.program, "skip", env={}, log=self.log.append)
        a = runinfo.testbed_id(self.data / "bench", machine="m")
        b = runinfo.testbed_id(lab_sandbox.real(second) / "data" / "bench", machine="m")
        self.assertEqual(a, b)

    def test_pcm_hash_is_the_speed_test_rule(self):
        """Хэш звука лаборатории (lab_run.pcm_sha256) = хэш звука hypstore одного и того же звука."""
        import numpy as np
        import hypstore
        audio = np.linspace(-1.0, 1.0, 8000, dtype=np.float32)
        self.assertEqual(lab_run.pcm_sha256(audio), hashlib.sha256(hypstore.pcm_bytes(audio)).hexdigest())

    def test_cli_export_bench(self):
        import experiment
        self.run_lab()
        base_args = ["export-bench", "--sandbox", str(self.sandbox)]
        self.assertEqual(experiment.main(base_args), 0)
        self.assertEqual(len(list(self.out.glob("lab-*.json"))), 1)
        self.assertEqual(experiment.main(base_args + ["--to-data", str(self.target)]), 0)
        self.assertFalse((self.target / "bench").exists(), "без --apply папка данных не меняется")
        self.assertEqual(experiment.main(base_args + ["--apply"]), 2, "--apply без --to-data — отказ")
        self.assertEqual(experiment.main(base_args + ["--to-data", str(self.target), "--apply"]), 0)
        self.assertEqual(len(list((self.target / "bench" / "lab").glob("lab-*.json"))), 1)
        self.assertEqual(experiment.main(base_args + ["--to-data", str(self.target / "нет"), "--apply"]), 2)


UNIT_A, UNIT_B = "aaaaaaaaaaaa", "bbbbbbbbbbbb"
RUN_1, RUN_2, RUN_3 = "20261125-100000-aaaaaa.json", "20261125-100001-bbbbbb.json", "20261125-100002-cccccc.json"


def make_hypotheses(data):
    """Синтетическая папка гипотез: три файла прогонов, кэши и посторонние имена (их копировать нельзя)."""
    hyp = data / "hypotheses"
    for unit, run in ((UNIT_A, RUN_1), (UNIT_A, RUN_2), (UNIT_B, RUN_3)):
        (hyp / unit).mkdir(parents=True, exist_ok=True)
        (hyp / unit / run).write_text(json.dumps({"schema": "hyp-2", "roles": [{"words": [{"w": SECRET}]}],
                                                  "ids": {"run_id": run[:-5]}}, ensure_ascii=False),
                                      encoding="utf-8")
    (hyp / "_artifacts.json").write_text('{"x": 1}', encoding="utf-8")
    (hyp / "_testbed.json").write_text('{"testbed_id": "0"}', encoding="utf-8")
    (hyp / "notes.txt").write_text("посторонний файл", encoding="utf-8")
    (hyp / "zz").mkdir()
    (hyp / "zz" / RUN_1).write_text("{}", encoding="utf-8")                 # папка не по маске
    (hyp / UNIT_B / "other.json").write_text("{}", encoding="utf-8")         # имя не по маске
    return hyp


def link_dir(link, target):
    """Ссылка на папку: symlink, а в Windows без прав — junction. -> False, если не удалось."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        if os.name != "nt":
            return False
        import subprocess
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False)
        return link.exists()


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class WorkSandbox(unittest.TestCase):
    """prepare --work (план лаборатории 3а, PR-А): только база записей и файлы гипотез, со сверкой SHA-256."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lab-work-")
        self.base = Path(self.tmp.name)
        self.data, self.audio = make_source(self.base)
        self.hyp = make_hypotheses(self.data)
        self.program = self.base / "program"
        self.program.mkdir()
        self.sandbox = self.base / "sandbox"
        self.before = (folder_snapshot(self.data), folder_snapshot(self.audio))
        self.log = []

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self, **kw):
        kw.setdefault("voice_profiles", "skip")
        return lab_sandbox.prepare(self.sandbox, self.data, self.program, env={}, log=self.log.append,
                                   work=True, **kw)

    def assert_source_unchanged(self):
        self.assertEqual(folder_snapshot(self.data), self.before[0], "настоящая папка данных изменилась")
        self.assertEqual(folder_snapshot(self.audio), self.before[1], "папка с аудио изменилась")

    def manifest(self):
        s = lab_sandbox.real(self.sandbox)
        return json.loads((s / "source" / lab_sandbox.HYP_MANIFEST).read_text(encoding="utf-8"))

    def copies(self):
        root = lab_sandbox.real(self.sandbox) / "source" / "hypotheses"
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) \
            if root.exists() else []

    def test_copies_database_and_run_files_only(self):
        report = self.prepare()
        self.assert_source_unchanged()
        s = lab_sandbox.real(self.sandbox)
        self.assertTrue((s / "source" / "state.sqlite3").is_file())
        expected = [f"{UNIT_A}/{RUN_1}", f"{UNIT_A}/{RUN_2}", f"{UNIT_B}/{RUN_3}"]
        self.assertEqual(self.copies(), expected)          # кэши, посторонние имена и папки не по маске — нет
        for rel in expected:
            self.assertEqual((s / "source" / "hypotheses" / rel).read_bytes(), (self.hyp / rel).read_bytes())
        man = self.manifest()
        self.assertEqual(man["schema"], lab_sandbox.HYP_MANIFEST_SCHEMA)
        self.assertEqual([f["relative"] for f in man["files"]], expected)
        for f in man["files"]:
            self.assertEqual(f["result"], "copied")
            self.assertEqual(f["sha256"], hashlib.sha256((self.hyp / f["relative"]).read_bytes()).hexdigest())
            self.assertEqual(f["bytes"], (self.hyp / f["relative"]).stat().st_size)
        self.assertEqual(man["counts"]["copied"], 3)
        text = json.dumps(man, ensure_ascii=False)
        self.assertNotIn(str(self.base), text)                # только относительные имена
        self.assertNotIn(SECRET, text)
        marker = json.loads((s / lab_sandbox.MARKER).read_text(encoding="utf-8"))
        self.assertIs(marker["work"], True)
        line = next(c for c in marker["copied"] if c["relative"] == "source/hypotheses")
        self.assertEqual((line["files"], line["method"]), (3, "sha256"))
        self.assertEqual(line["sha256"], hashlib.sha256((s / "source" / lab_sandbox.HYP_MANIFEST).read_bytes())
                         .hexdigest())
        self.assertEqual(sorted(c["relative"] for c in marker["copied"]), ["source/hypotheses", "source/state.sqlite3"])
        # настройки, соль, словари и профили не копируются; единиц опыта нет
        self.assertEqual([p.name for p in (s / "data").iterdir()], [])
        units = json.loads((s / "units.json").read_text(encoding="utf-8"))
        self.assertEqual((units["schema"], units["units"]), ("lab-units-1", []))
        self.assertEqual(report["recordings"], 3)
        self.assertEqual(report["hypotheses"]["copied"], 3)
        lab_sandbox.load_sandbox(self.sandbox, env={})        # песочница читается штатно

    def test_audio_is_not_opened_or_hashed(self):
        import builtins
        import runinfo
        real_open, real_sha = builtins.open, runinfo.sha256_file
        touched = []
        audio_key = os.path.normcase(str(lab_sandbox.real(self.audio)))

        def watch(path):
            if os.path.normcase(str(lab_sandbox.real(path))).startswith(audio_key):
                touched.append(Path(path).name)

        def spy_open(file, *a, **kw):
            if isinstance(file, (str, os.PathLike)):
                watch(file)
            return real_open(file, *a, **kw)

        def spy_sha(path, *a, **kw):
            watch(path)
            return real_sha(path, *a, **kw)
        with mock.patch("builtins.open", spy_open), mock.patch.object(runinfo, "sha256_file", spy_sha):
            self.prepare()
        self.assertEqual(touched, [])
        self.assert_source_unchanged()

    def test_links_are_not_followed(self):
        outside = self.base / "outside"
        (outside / "cccccccccccc").mkdir(parents=True)
        (outside / "cccccccccccc" / RUN_1).write_text("{}", encoding="utf-8")
        if not link_dir(self.hyp / "cccccccccccc", outside / "cccccccccccc"):
            self.skipTest("не удалось создать ссылку на папку")
        self.before = (folder_snapshot(self.data), folder_snapshot(self.audio))
        self.prepare()
        self.assertNotIn(f"cccccccccccc/{RUN_1}", self.copies())
        self.assertEqual(len(self.copies()), 3)

    def test_changed_deleted_and_new_files_during_copy_are_not_taken(self):
        real_copy = lab_sandbox._copy_file

        def racing(src, dst):
            real_copy(src, dst)
            name = Path(src).name
            if name == RUN_1:                              # файл изменился после копирования
                with open(src, "ab") as f:
                    f.write(b" ")
            elif name == RUN_2:                            # файл удалили после копирования
                os.unlink(src)
                (self.hyp / UNIT_B / "20261125-100009-dddddd.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(lab_sandbox, "_copy_file", racing):
            report = self.prepare()
        results = {f["relative"]: f["result"] for f in self.manifest()["files"]}
        self.assertEqual(results, {f"{UNIT_A}/{RUN_1}": "changed_during_copy",
                                   f"{UNIT_A}/{RUN_2}": "deleted_during_copy",
                                   f"{UNIT_B}/{RUN_3}": "copied"})
        self.assertEqual(self.copies(), [f"{UNIT_B}/{RUN_3}"])
        self.assertEqual(self.manifest()["appeared_during_copy"], 1)
        self.assertEqual(report["hypotheses"]["changed_during_copy"], 1)
        self.assertTrue(any("изменились при копировании: 1" in m for m in self.log))

    def test_copy_that_does_not_match_is_dropped(self):
        real_copy = lab_sandbox._copy_file

        def corrupt(src, dst):
            real_copy(src, dst)
            if Path(src).name == RUN_3:
                with open(dst, "ab") as f:
                    f.write(b"x")
        with mock.patch.object(lab_sandbox, "_copy_file", corrupt):
            self.prepare()
        results = {f["relative"]: f["result"] for f in self.manifest()["files"]}
        self.assertEqual(results[f"{UNIT_B}/{RUN_3}"], "copy_mismatch")
        self.assertNotIn(f"{UNIT_B}/{RUN_3}", self.copies())
        self.assert_source_unchanged()

    def test_missing_hypotheses_folder_is_not_created(self):
        import shutil
        shutil.rmtree(self.hyp)
        self.before = (folder_snapshot(self.data), folder_snapshot(self.audio))
        report = self.prepare()
        self.assertFalse(self.hyp.exists())
        self.assert_source_unchanged()
        self.assertEqual(self.manifest()["files"], [])
        self.assertEqual(report["hypotheses"]["copied"], 0)

    def test_live_writer_with_wal_leaves_data_folder_untouched(self):
        import storage
        writer = storage.open_store(self.data / "state.sqlite3")
        try:
            writer.db.execute("UPDATE local_users SET display_name='ИЗ-ЖУРНАЛА'")
            before = folder_snapshot(self.data)
            self.prepare()
            self.assertEqual(folder_snapshot(self.data), before, "при живом писателе папка данных не меняется")
        finally:
            writer.close()
        marker = json.loads((lab_sandbox.real(self.sandbox) / lab_sandbox.MARKER).read_text(encoding="utf-8"))
        self.assertEqual(next(c["method"] for c in marker["copied"] if c["relative"] == "source/state.sqlite3"),
                         "bytes")
        self.assertEqual(len(self.copies()), 3)

    def test_refuses_wrong_options_and_dangerous_places(self):
        for kw in ({"voice_profiles": "copy"}, {"audio": [self.audio / "one.wav"]},
                   {"audio": [self.audio / "one.wav"], "only_audio": True}, {"group": "grp"}):
            with self.subTest(kw=sorted(kw)), self.assertRaises(lab_sandbox.SandboxError):
                self.prepare(**kw)
        self.assertFalse(self.sandbox.exists())
        for bad in (self.data / "lab", self.hyp / "lab", self.audio / "lab", self.program / "lab"):
            with self.subTest(bad=bad), self.assertRaises(lab_sandbox.SandboxError):
                lab_sandbox.prepare(bad, self.data, self.program, "skip", env={}, log=self.log.append, work=True)
        self.assertFalse((self.data / "lab").exists())
        self.assertFalse((self.hyp / "lab").exists())
        self.assert_source_unchanged()

    def test_command_line(self):
        import experiment
        experiment.main(["prepare", "--sandbox", str(self.sandbox), "--voice-profiles", "skip", "--work",
                         "--source-data", str(self.data), "--program-dir", str(self.program)])
        self.assertEqual(len(self.copies()), 3)
        self.assert_source_unchanged()

    def test_ordinary_prepare_does_not_copy_hypotheses(self):
        lab_sandbox.prepare(self.sandbox, self.data, self.program, "skip", env={}, log=self.log.append)
        s = lab_sandbox.real(self.sandbox)
        self.assertFalse((s / "source" / "hypotheses").exists())
        self.assertNotIn("work", json.loads((s / lab_sandbox.MARKER).read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
