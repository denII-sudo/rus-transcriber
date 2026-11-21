"""Анализ лаборатории (lab/lab_analyze.py): сверка пересведения с конвейером, попарное согласие,
«кто был прав» по решениям человека, обезличенная выгрузка, поведение без данных.

Пары считаются настоящим worker.run_words с подменёнными моделями (как в test_lab_experiment),
поэтому сверка сравнивает пересведение именно с конвейером. Модели не загружаются. Установщик
папку lab/ не ставит: в установленной копии набор пропускается.
"""
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
HAVE_LAB = (LAB / "lab_analyze.py").is_file()
if HAVE_LAB:
    sys.path.insert(0, str(LAB))
    import lab_analyze  # noqa: E402
    import lab_run  # noqa: E402
    import lab_sandbox  # noqa: E402
    import lab_spec  # noqa: E402
    import lab_summary  # noqa: E402

# слова моделей по двум кускам: [0–1 с] и [1–3 с]
OUTPUT = {
    "v3_e2e_rnnt": ([" раз", " два"], [" три", " четыре", " пять"]),
    "v3_e2e_ctc": ([" раз", " дво"], [" три", " четыре", " шесть"]),
    "multilingual_large_ctc": ([" раз", " два"], [" три", " черыре", " пять"]),
}
WHISPER = ("раз дво", "три четыре семь")
CONFS = {
    "a-current": {"primary": {"model": "v3_e2e_rnnt"}, "second": {"model": "v3_e2e_ctc"},
                  "third": {"engine": "whisper-ct2", "mode": "all"}},
    "c-no-third": {"primary": {"model": "v3_e2e_rnnt"}, "second": {"model": "v3_e2e_ctc"}, "third": None},
    "d-ml2-whisper3": {"primary": {"model": "v3_e2e_rnnt"}, "second": {"model": "multilingual_large_ctc"},
                       "third": {"engine": "whisper-ct2", "mode": "all"}},
    "f-whisper-doubt": {"primary": {"model": "v3_e2e_rnnt"}, "second": {"model": "v3_e2e_ctc"},
                        "third": {"engine": "whisper-ct2", "mode": "doubt"}},
}
WORDS_IN_TEXT = ("дво", "восемь", "четыре", "черыре", "шесть", "семь")
NAME = "Неизвестнов"


class Main:
    def __init__(self, model):
        self.name, self.model_name = "fake-" + model, model

    def transcribe_timestamped(self, pieces, progress=None, diagnostics=None):
        diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
        out = []
        for i, _ in enumerate(pieces):
            toks = OUTPUT[self.model_name][i]
            out.append(SimpleNamespace(tokens=list(toks), timestamps=[0.1 + 0.4 * k for k in range(len(toks))],
                                       logprobs=[-0.1] * len(toks), text="".join(toks)))
        return out


class CT2:
    name, compile_sec = "fake-ct2", None

    def __init__(self, *a, **kw):
        pass

    def transcribe_texts(self, pieces, progress=None, diagnostics=None):
        diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
        return [WHISPER[0] if len(p) <= 16000 else WHISPER[1] for p in pieces]


def run_pair(conf_id):
    """Одна пара настоящим конвейером. -> (metrics, data, hyps)."""
    import numpy as np
    import backends
    import words
    roles = CONFS[conf_id]
    cfg = {"asr_model": roles["primary"]["model"], "asr_second_model": roles["second"]["model"],
           "asr_third_model": "whisper", "asr_third_device": "CPU", "asr_third_backend": "whisper-ct2",
           "asr_third_mode": roles["third"]["mode"] if roles["third"] else "off",
           "asr_third_long_chunks": False, "asr_third_quant_cpu": "int8", "asr_third_batch": 4,
           "asr_third_max_repeat": 3, "emotions": False, "review_missed": False, "interaction": False}
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
    with mock.patch.object(backends, "make_asr", lambda c: (Main(c["asr_model"]), [])), \
         mock.patch.object(backends, "WhisperCT2", CT2), \
         mock.patch.object(words, "make_chunks", return_value=chunks), \
         mock.patch.object(words, "assign_voices", return_value=0):
        data, hyps, calls, timing = engine.run_config(shared, cfg, {"id": conf_id, "roles": roles})
    data["speakers"] = [NAME]
    return lab_summary.pair_metrics(data, hyps, calls, timing, 3.0), data, hyps, cfg


def make_sandbox(base, configs=tuple(CONFS), with_source=True):
    """Песочница с базой результатов и (по желанию) копией базы записей с решениями человека."""
    s = base / "sandbox"
    (s / "results").mkdir(parents=True)
    (s / "export").mkdir()
    rid = str(uuid.uuid4())
    audio = base / "audio" / "one.wav"
    units = {"schema": "lab-units-1", "missing_audio_recordings": [],
             "units": [{"unit_id": "u1", "audio_sha256": "u1" * 32, "location": str(audio), "size_bytes": 1,
                        "group": "grp", "duration_sec": 3.0,
                        "recordings": [{"recording_id": rid, "location": str(audio), "group": "grp"}]}]}
    (s / "units.json").write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
    (s / lab_sandbox.MARKER).write_text(json.dumps({
        "schema": lab_sandbox.MARKER_SCHEMA, "source_data": str(base / "real-data"),
        "program_dir": str(base / "program")}), encoding="utf-8")
    db = lab_run.open_results(s / "results" / "lab.sqlite3")
    try:
        db.execute("INSERT INTO runs (run_id, started_utc, state) VALUES ('r1', 't', 'finished')")
        for cid in configs:
            metrics, data, hyps, cfg = run_pair(cid)
            sha = cid + "-sha"
            db.execute("INSERT INTO configs VALUES (?,?,?,?,?,?,?)",
                       (cid, sha, cid, json.dumps(CONFS[cid]), "{}", json.dumps(cfg), "r1"))
            db.execute("INSERT INTO pairs (run_id, unit_id, config_id, config_sha256, state, started_utc, "
                       "finished_utc, diarization_sha256, metrics_json, result_json, hypotheses_json) "
                       "VALUES ('r1','u1',?,?,'done','t','t',?,?,?,?)",
                       (cid, sha, "d" * 64, lab_run._dumps(metrics), lab_run._dumps(data), lab_run._dumps(hyps)))
        db.commit()
    finally:
        db.close()
    if with_source:
        make_source_db(s / "source" / "state.sqlite3", rid, audio)
    return s, units


def make_source_db(path, rid, audio):
    """Копия базы записей: итоговая стенограмма с решениями человека и одной правкой ИИ-агента."""
    import storage
    path.parent.mkdir(parents=True)
    store = storage.initialize(path)
    try:
        user = store.create_user("tester")
        inst = str(uuid.uuid4())
        store.bind_installation(inst, user)
        ctx = store.capture_context(inst, "human")
        docs = path.parent.parent.parent / "docs-root"
        docs.mkdir()
        root_id = store.register_root(docs)

        def w(i, text, s, e, prov=None, by=None):
            word = {"word_id": f"w{i}", "w": text, "s": s, "e": e}
            if prov:
                word["provenance"] = {"text": {"by": prov[0], "edit": prov[1]}}
            if by:
                word["by"] = by
            return word
        words = [w(0, "Раз", 0.1, 0.5), w(1, "дво,", 0.5, 0.9, ("human", "r1"), "human"),
                 w(2, "три", 1.1, 1.5, ("human", "r3"), "human"), w(3, "четыре", 1.5, 1.9, ("claude", "r4"), "claude"),
                 w(4, "восемь.", 1.9, 2.3, ("human", "r2"), "human")]
        transcript = {"duration": "00:00:03", "segments": [{"segment_id": "s0", "speaker": NAME, "words": words}],
                      "edits": [{"id": "r1", "t": 0.5, "by": "human", "before": {"text": "два"}, "after": {"text": "дво"}},
                                {"id": "r2", "t": 1.9, "by": "human", "before": {"text": "пять"},
                                 "after": {"text": "восемь"}},
                                {"id": "r4", "t": 1.5, "by": "claude", "before": {"text": "x"}, "after": {"text": "четыре"}}]}
        review = {"review": [{"id": "r1", "t": 0.5, "t1": 0.5, "t2": 0.5, "status": "resolved"},
                             {"id": "r2", "t": 1.9, "t1": 1.9, "t2": 1.9, "status": "resolved"},
                             {"id": "r5", "t": 2.5, "t1": 2.5, "t2": 2.5, "status": "open"}], "answers": {}}
        store.create_recording_with_document(rid, os.path.normcase(os.path.abspath(audio)), transcript, review,
                                             str(uuid.uuid4()), ctx, root_id=root_id, generated_at="t")
    finally:
        store.close()


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class AnalyzeSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="lab-analyze-")
        cls.base = Path(cls.tmp.name)
        cls.sandbox, cls.units = make_sandbox(cls.base)
        cls.a = lab_analyze.analyze(cls.sandbox, cls.units, log=lambda m: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_revote_of_original_set_matches_pipeline(self):
        checks = self.a["revote_check"]
        self.assertEqual(len(checks), len(CONFS))
        for key, c in checks.items():
            with self.subTest(key):
                self.assertEqual(c["state"], "ok", c)
                self.assertGreater(c["words_checked"], 0)
        self.assertEqual(checks["u1|a-current"]["function"], "vote3")
        self.assertEqual(checks["u1|c-no-third"]["function"], "mark_alt")
        # vote3 «как сейчас»: «два»→«дво» заменено (CTC и Whisper согласны), «пять» — спор всех трёх
        vote = self.a["configs"]["a-current"]["all"]["vote"]
        self.assertEqual((vote["agree3"], vote["replaced"], vote["disputed"]), (3, 1, 1))
        self.assertEqual(self.a["configs"]["c-no-third"]["all"]["vote"]["disputed"], 2)

    def test_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory(prefix="lab-analyze-mism-") as tmp:
            s, units = make_sandbox(Path(tmp), configs=("a-current",), with_source=False)
            db = sqlite3.connect(str(s / "results" / "lab.sqlite3"))
            try:
                pair_id, text = db.execute("SELECT pair_id, result_json FROM pairs").fetchone()
                data = json.loads(text)
                data["quality"]["vote"]["disputed"] = 99          # «конвейер записал» другое число
                db.execute("UPDATE pairs SET result_json=? WHERE pair_id=?",
                           (json.dumps(data, ensure_ascii=False), pair_id))
                db.commit()
            finally:
                db.close()
            a = lab_analyze.analyze(s, units, log=lambda m: None)
            self.assertEqual(a["revote_check"]["u1|a-current"]["state"], "mismatch")
            self.assertIn("vote_stats", a["revote_check"]["u1|a-current"]["mismatch"])
            self.assertIn("НЕ совпало", (s / "results" / "report.md").read_text(encoding="utf-8"))

    def test_pairwise_agreement_known_example(self):
        t = self.a["pairwise"]["total"]
        ctc = t["primary:v3_e2e_rnnt~second:v3_e2e_ctc"]
        self.assertEqual((ctc["words"], ctc["agree"], ctc["places"]), (5, 3, 2))
        ml = t["primary:v3_e2e_rnnt~second:multilingual_large_ctc"]
        self.assertEqual((ml["agree"], ml["places"]), (4, 1))
        # судья faster-whisper против основной в «два» и «пять»: CTC там с основной ни разу, ML — оба раза
        sim = self.a["pairwise"]["similar_errors"]
        self.assertEqual(sim["primary:v3_e2e_rnnt~second:v3_e2e_ctc~third:whisper-ct2"]["second_sides_with_primary"], 0)
        self.assertEqual(sim["primary:v3_e2e_rnnt~second:multilingual_large_ctc~third:whisper-ct2"]
                         ["second_sides_with_primary"], 2)

    def test_combination_across_configs(self):
        rv = self.a["revote"]["units"]["u1"]
        cross = rv["primary:v3_e2e_rnnt|second:multilingual_large_ctc|third:whisper-ct2"]
        d_vote = self.a["configs"]["d-ml2-whisper3"]["all"]["vote"]
        self.assertEqual(cross["stats"], {k: d_vote[k] for k in ("agree3", "agree2", "replaced", "inserted", "disputed")})
        bad = rv["primary:v3_e2e_rnnt|second:multilingual_large_ctc|third:whisper-ct2:doubt"]
        self.assertEqual((bad["state"], bad["reason"]), ("invalid", "doubt_needs_own_second"))
        own = rv["primary:v3_e2e_rnnt|second:v3_e2e_ctc|third:whisper-ct2:doubt"]
        self.assertEqual(own["state"], "ok")
        self.assertEqual(self.a["revote"]["original_combo"]["c-no-third"],
                         "primary:v3_e2e_rnnt|second:v3_e2e_ctc")

    def test_who_was_right(self):
        h = self.a["human"]
        self.assertEqual(h["none_right"], {"decisions": 3, "none_right": 1}, "правка ИИ-агента не учитывается")
        src = h["sources"]
        self.assertEqual(src["second:v3_e2e_ctc"]["changed"], {"right": 1, "wrong": 1})
        self.assertEqual(src["primary:v3_e2e_rnnt"]["changed"], {"right": 0, "wrong": 2})
        self.assertEqual(src["primary:v3_e2e_rnnt"]["confirmed"], {"right": 1, "wrong": 0})
        cfg = h["configs"]
        self.assertEqual((cfg["a-current"]["shown"], cfg["a-current"]["spared_right"],
                          cfg["a-current"]["silent_error"]), (1, 2, 0))
        self.assertEqual((cfg["c-no-third"]["shown"], cfg["c-no-third"]["spared_right"]), (2, 1))
        self.assertEqual((cfg["d-ml2-whisper3"]["silent_error"], cfg["d-ml2-whisper3"]["spared_right"]), (2, 1))
        self.assertEqual(h["shared_error"]["second:multilingual_large_ctc"]["same_as_primary"], 2)
        self.assertEqual(h["units_stats"]["u1"]["agent_edits"], 1)

    def test_export_has_no_texts_or_paths(self):
        text = (self.sandbox / "export" / "analysis.json").read_text(encoding="utf-8")
        data = json.loads(text)
        self.assertEqual((data["schema"], data["version"]), (lab_analyze.ANALYSIS_SCHEMA, lab_analyze.ANALYSIS_VERSION))
        for bad in WORDS_IN_TEXT + (NAME, "one.wav", str(self.base), str(self.base).replace("\\", "\\\\"), "grp"):
            self.assertNotIn(bad, text, bad)
        report = (self.sandbox / "results" / "report.md").read_text(encoding="utf-8")
        for bad in WORDS_IN_TEXT + (NAME,):
            self.assertNotIn(bad, report, bad)
        examples = (self.sandbox / "results" / "examples.md").read_text(encoding="utf-8")
        self.assertIn("восемь", examples, "тексты — только в локальном файле примеров")

    def test_source_copy_is_not_changed(self):
        src = self.sandbox / "source"
        before = {p.name: p.stat().st_mtime_ns for p in src.iterdir() if p.is_file()}
        lab_analyze.analyze(self.sandbox, self.units, log=lambda m: None)
        after = {p.name: p.stat().st_mtime_ns for p in src.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_sandbox_loads_for_cli(self):
        s, marker, units = lab_sandbox.load_sandbox(self.sandbox, env={})
        self.assertEqual(units["units"][0]["unit_id"], "u1")


@unittest.skipUnless(HAVE_LAB, "папки lab/ нет (установленная копия программы)")
class AnalyzeWithoutData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lab-analyze-empty-")
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_results_db(self):
        s = self.base / "s"
        s.mkdir()
        a = lab_analyze.analyze(s, {"units": []}, log=lambda m: None)
        self.assertIn("not_measured", a)
        self.assertIn("Не измерено", (s / "results" / "report.md").read_text(encoding="utf-8"))
        self.assertTrue((s / "export" / "analysis.json").is_file())

    def test_no_source_db_and_single_config(self):
        s, units = make_sandbox(self.base, configs=("c-no-third",), with_source=False)
        a = lab_analyze.analyze(s, units, log=lambda m: None)
        self.assertIn("not_measured", a["human"])
        self.assertIn("not_measured", a["comparison"]["c-no-third"]["all"], "базовой нет — сравнение не измерено")
        self.assertEqual(a["revote_check"]["u1|c-no-third"]["state"], "ok")
        self.assertIn("не измерено", (s / "results" / "report.md").read_text(encoding="utf-8").lower())

    def test_fitness_flags_lost_chunk(self):
        words = [{"w": "а", "s": 1.0 + i * 0.1, "e": 1.05 + i * 0.1, "c": 1.0} for i in range(6)]
        p = {"hyps": {"primary": {"words": words, "chunks": [{"a": 0, "b": 3, "own": [0, 3]}]},
                      "second": {"words": [], "chunks": []}},
             "metrics": {"make_asr": [{"requested": {"asr_model": "m", "asr_device": "GPU"}, "actual": "cpu",
                                       "fallbacks": ["openvino:GPU: RuntimeError"]}]},
             "result": {}, "effective": {}}
        fit = lab_analyze.pair_fitness(p, lab_analyze._words_module())
        self.assertFalse(fit["fit"])
        self.assertIn("second:lost_chunks", fit["reasons"])
        self.assertIn("fallback:m", fit["reasons"])
        self.assertIn("device_mismatch:m", fit["reasons"])


if __name__ == "__main__":
    unittest.main()
