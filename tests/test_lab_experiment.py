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
if HAVE_LAB:
    sys.path.insert(0, str(LAB))
    import lab_run  # noqa: E402
    import lab_sandbox  # noqa: E402
    import lab_spec  # noqa: E402
    import lab_summary  # noqa: E402

SECRET = "Тайнословие"           # «текст записи»: не должен попасть в сводку
NAME = "Неизвестнов"            # «имя участника»
EXPERIMENT = LAB / "experiments" / "models-arbitration.json"


def snapshot(root):
    """Состояние папки: путь -> (размер, сумма, время изменения)."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            st = p.stat()
            out[p.relative_to(root).as_posix()] = (st.st_size, hashlib.sha256(p.read_bytes()).hexdigest(),
                                                   st.st_mtime_ns)
    return out


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
        }
        for name, change in cases.items():
            bad = json.loads(json.dumps(good))
            bad["configurations"][0].update(change)
            with self.subTest(name), self.assertRaises(lab_spec.SpecError):
                lab_spec.parse(json.dumps(bad))
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
        self.before = (snapshot(self.data), snapshot(self.audio))
        self.log = []

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self, voices="skip"):
        return lab_sandbox.prepare(self.sandbox, self.data, self.program, voices, env={},
                                   log=self.log.append)

    def assert_source_unchanged(self):
        self.assertEqual(snapshot(self.data), self.before[0], "настоящая папка данных изменилась")
        self.assertEqual(snapshot(self.audio), self.before[1], "папка с аудио изменилась")

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
            before = snapshot(self.data)
            self.prepare()
            self.assertEqual(snapshot(self.data), before,
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
        before = snapshot(self.data)
        self.assertNotIn("state.sqlite3-shm", before)
        self.prepare()
        self.assertEqual(snapshot(self.data), before, "новый -shm не появился, ничего не изменилось")
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
        self.before = (snapshot(self.data), snapshot(self.audio), snapshot(self.trial.parent))
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
        self.assertEqual(snapshot(self.data), self.before[0])
        self.assertEqual(snapshot(self.audio), self.before[1])
        self.assertEqual(snapshot(self.trial.parent), self.before[2], "пробный файл только читается")

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
        self.assertEqual(snapshot(self.trial.parent), self.before[2], "исходный файл только читается")
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
        self.assertEqual(snapshot(self.trial.parent)[self.trial.name], self.before[2][self.trial.name],
                         "исходный файл только читается")


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class CaptureWithRealPipeline(unittest.TestCase):
    """Перехват гипотез вокруг настоящего worker.run_words (модели подменены, как в test_whisper_ct2)."""

    def test_hypotheses_of_all_passes_are_captured(self):
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
        cfg = {"asr_model": "primary", "asr_second_model": "second", "asr_third_model": "whisper",
               "asr_third_device": "CPU", "asr_third_backend": "whisper-ct2", "asr_third_mode": "all",
               "asr_third_long_chunks": False, "asr_third_quant_cpu": "int8", "asr_third_batch": 4,
               "emotions": False, "review_missed": False, "interaction": False}
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
            data, hyps, calls, timing = engine.run_config(shared, cfg, conf)
            self.assertIs(words.transcribe, orig_transcribe, "подмены сняты после вызова")
        self.assertEqual([w["w"] for w in hyps["primary"]["words"]], ["раз", "раз"])
        self.assertEqual([w["w"] for w in hyps["second"]["words"]], ["два", "два"])
        self.assertEqual(hyps["primary"]["model"], "primary")
        self.assertEqual(len(hyps["primary"]["chunks"]), 2)
        self.assertEqual([c["raw_text"] for c in hyps["third"]["chunks"]], ["три три", "три три"])
        self.assertEqual(hyps["vote"]["function"], "vote3")
        self.assertEqual(hyps["vote"]["third_words"], ["три", "три", "три", "три"])
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


if __name__ == "__main__":
    unittest.main()
