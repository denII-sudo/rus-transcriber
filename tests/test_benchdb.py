"""База прогонов теста скорости (benchdb.py, этап 2 задания docs/bench-database-spec.md).

Проверяется то, от чего зависит доверие к базе: загрузка повторяема и ничего не дублирует; чтение
отделено от записи (показ базы не создаёт и не меняет); файлы прогонов не трогаются; в базе нет путей,
имён и текстов записей; факты только дописываются; прогоны разных методик и исторические замеры не
сравниваются; сравнение называет ось, по которой отличаются два измерения.

Всё на синтетике во временной папке: файлы прогонов собираются через runinfo.write_run, как настоящие.
"""
import atexit
import copy
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-benchdb-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import benchdb  # noqa: E402
import paths  # noqa: E402
import runinfo  # noqa: E402

SECRET_NAME = "Совещание отдела кадров"
SECRET_WORD = "конфиденциальнейшееслово"
PCM = b"\x01\x00" * 16000


def new_bench():
    d = Path(tempfile.mkdtemp(prefix="b-", dir=_TMP))
    (d / "runs").mkdir()
    return d


def audio_identity(seed=b"a"):
    p = Path(_TMP) / f"{SECRET_NAME}.wav"
    p.write_bytes(b"container-" + seed)
    return runinfo.audio_identity(p, PCM + seed * 2, 16000, source_samples=48000)


def result(vid, family="rnnt", device="CPU", asr_sec=2.0, backend="cpu", device_actual=None, **extra):
    r = {"id": vid, "family": family, "device": device, "load_sec": 3.0, "first_chunk_sec": 0.5,
         "asr_sec": asr_sec, "chunks": 4, "backend_name": backend, "providers_actual": ["CPUExecutionProvider"],
         "compile_sec": None, "fallbacks": None, "chunk_words": [5, 6, 5, 4], "empty_chunks": 0,
         "peak_memory_mb": 512.0, "compile_cache_new_files": 0, "model_downloaded": False,
         "device_actual": device_actual or device, "device_mismatch": False, "unfit": None,
         "rtfx": round(5.0 / asr_sec, 1), "wall_sec": 9.0, "total_10min": 120.0,
         "reference_variant": "onnx-cpu", "similarity": 0.98, "words": 20, "missing": 0, "extra": 1,
         "changed": 1}
    r.update(extra)
    return r


def record(run_id, method="bench-2", power="ac", device_actual="CPU", seed=b"a", status="done",
           with_error_variant=False, packages=None, hardware=None, saved=None, config_written=False):
    variants = [{"id": "onnx-cpu", "label": "RNN-T, процессор", "family": "rnnt", "device": "CPU",
                 "asr_backend": "onnx-cpu", "asr_model": "v3_e2e_rnnt"},
                {"id": "ctc-cpu", "label": "CTC, процессор", "family": "ctc", "device": "CPU",
                 "asr_backend": "onnx-cpu", "asr_model": "v3_e2e_ctc"}]
    results = [result("onnx-cpu", asr_sec=2.0, device_actual=device_actual),
               result("ctc-cpu", family="ctc", asr_sec=1.0, vote={"contested": 1, "for_ctc": 0})]
    if with_error_variant:
        variants.append({"id": "ov-gpu", "label": "RNN-T, видеокарта", "family": "rnnt", "device": "GPU",
                         "asr_backend": "ov-gpu", "asr_model": "v3_e2e_rnnt"})
        results.append({"id": "ov-gpu", "family": "rnnt", "device": "GPU", "wall_sec": 1.0,
                        "error": [f"Не найден файл 'C:\\Users\\Иван\\{SECRET_NAME}\\model.onnx'"]})
    rec = {"schema_version": runinfo.SCHEMA_VERSION, "method_version": method, "kind": "bench",
           "run_id": run_id, "job_id": "20260101-000000-b00001", "started_utc": "2026-11-24T01:00:00Z",
           "status": status, "finished_utc": "2026-11-24T01:05:00Z", "duration_sec": 300.0,
           "params": {"seconds": 5, "chunk_max": 20.0, "only": "", "model": "v3_e2e_rnnt", "no_write": True,
                      "threshold": 0.99, "third_max_repeat": 3, "repeats": 1},
           "conditions": {"start": {"power_source": power, "cpu_load_percent": 3.0,
                                    "power": {"scheme": "balanced", "mode_ac": "best"}},
                          "end": {"power_source": power}},
           "audio": audio_identity(seed),
           "testbed": {"testbed_id": "0123456789abcdef",
                       "hardware": hardware or {"status": "ok", "cpu": {"model": "Synthetic CPU 9000", "cores": 6,
                                                                       "threads": 12, "features": ["avx2"]},
                                                "system": {"manufacturer": "Synthetic", "model": "Box"},
                                                "bios": {"version": "1.0"}, "memory": {"total_mb": 16384},
                                                "devices": [{"name": "Synthetic GPU", "driver_version": "1.2.3"}]}},
           "environment": {"os": {"system": "Windows", "build": "26100", "ubr": 1},
                           "python": {"version": "3.12.0", "implementation": "CPython"},
                           "packages": {"key": packages or {"onnxruntime": "1.20.0", "numpy": "2.0.0",
                                                            "huggingface-hub": "0.30.0"},
                                        "count": 80, "sha256": "ab" * 32},
                           "ffmpeg": None, "program_version": "2026.11.24.3",
                           "code": {"files_sha256": "cd" * 8, "git_revision": "e" * 40, "git_dirty": False},
                           "onnx_providers": ["CPUExecutionProvider"], "openvino_devices": ["CPU"]},
           "models": {"hf": [{"repo": "onnx-community/whisper-large-v3-turbo", "ref_main": "r" * 40,
                              "snapshots": ["r" * 40], "files": 3, "size_mb": 100.0}],
                      "local": [{"file": "gigaam/v3_e2e_rnnt.onnx", "size_bytes": 5_000_000,
                                 "modified_utc": "2026-11-01T00:00:00Z"}]},
           "variants": variants, "results": results,
           "reference_variants": {"rnnt": "onnx-cpu"}, "choice": {"main": "onnx-cpu", "ctc": "ctc-cpu",
                                                                  "whisper": None},
           "saved": saved, "config_written": config_written,
           "preflight": {"ok": power == "ac", "reasons": [] if power == "ac" else ["battery"]}}
    return rec


def write(bench, rec, texts=True):
    ok = texts and isinstance(rec.get("results"), list) and all(isinstance(r, dict) for r in rec["results"])
    t = {r["id"]: f"{SECRET_WORD} раз два три" for r in rec["results"]} if ok else None
    return runinfo.write_run(bench, rec, t)


def tree_hashes(root):
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def dump(bench):
    con = sqlite3.connect(benchdb.db_path(bench))
    try:
        return list(con.iterdump())
    finally:
        con.close()


def rows(bench, sql, *args):
    con = sqlite3.connect(benchdb.db_path(bench))
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def three_runs(bench):
    write(bench, record("20261101-100000-aaaaaa", method="bench-1"))
    write(bench, record("20261120-100000-bbbbbb", method="bench-2", with_error_variant=True,
                        saved={"asr_batch": 8, "asr_device": "CPU"}, config_written=True))
    write(bench, record("20261124-100000-cccccc", method="bench-2", device_actual="GPU"))


class SchemaTests(unittest.TestCase):

    def test_new_base_has_schema_version_and_no_user_text(self):
        bench = new_bench()
        benchdb.apply_load(bench)
        self.assertEqual(rows(bench, "SELECT value FROM meta WHERE key='schema_version'"), [("1",)])
        self.assertEqual(rows(bench, "SELECT value FROM meta WHERE key='key_version'"), [("1",)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM run"), [(0,)])
        self.assertTrue(benchdb.DB_NAME.startswith("bench_"))     # tools/check_repo.py не пускает bench_*

    def test_reopen_changes_nothing(self):
        bench = new_bench()
        benchdb.apply_load(bench)
        before = tree_hashes(bench)
        benchdb.apply_load(bench)
        self.assertEqual(tree_hashes(bench), before)

    def test_base_from_newer_schema_is_refused_without_write(self):
        bench = new_bench()
        benchdb.apply_load(bench)
        con = sqlite3.connect(benchdb.db_path(bench))
        con.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
        con.commit()
        con.close()
        write(bench, record("20261101-100000-aaaaaa"))
        before = tree_hashes(bench)
        with self.assertRaises(benchdb.BenchDbError):
            benchdb.apply_load(bench)
        self.assertEqual(tree_hashes(bench), before)

    def test_evaluation_table_exists_and_stays_empty(self):
        bench = new_bench()
        three_runs(bench)
        benchdb.apply_load(bench)
        self.assertEqual(rows(bench, "SELECT count(*) FROM evaluation"), [(0,)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM external_reference"), [(0,)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM reference"), [(0,)])
        cols = [c[1] for c in rows(bench, "PRAGMA table_info(evaluation)")]
        for name in ("metric", "value", "normalizer", "normalizer_version", "tool", "tool_version", "accuracy_key"):
            self.assertIn(name, cols)


class LoadTests(unittest.TestCase):

    def test_rows_of_three_runs(self):
        bench = new_bench()
        three_runs(bench)
        rep = benchdb.apply_load(bench)
        self.assertEqual(rep["counts"], {"loaded": 3})
        self.assertEqual(rows(bench, "SELECT count(*) FROM run"), [(3,)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM measurement"), [(7,)])   # 2 + 3 (с ошибкой) + 2
        self.assertEqual(rows(bench, "SELECT count(*) FROM test_unit"), [(1,)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM testbed"), [(1,)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM env_snapshot"), [(1,)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM raw_output"), [(3,)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM import_log WHERE outcome='loaded'"), [(3,)])
        by = dict(rows(bench, "SELECT run_id, method_version FROM run"))
        self.assertEqual(by["20261101-100000-aaaaaa"], "bench-1")
        self.assertEqual(by["20261120-100000-bbbbbb"], "bench-2")

    def test_failed_variant_is_a_measurement_with_status_and_clean_reason(self):
        bench = new_bench()
        three_runs(bench)
        benchdb.apply_load(bench)
        (status, reason, rtfx), = rows(bench, "SELECT status, status_reason, rtfx FROM measurement "
                                              "WHERE variant_id='ov-gpu'")
        self.assertEqual(status, "error")
        self.assertIsNone(rtfx)                                    # не измерено — NULL, а не ноль
        self.assertIn("<путь>", reason)
        self.assertNotIn(SECRET_NAME, reason)

    def test_error_run_is_recorded_as_invalid_without_measurements(self):
        bench = new_bench()
        rec = record("20261102-100000-dddddd", status="error")
        for k in ("variants", "results", "choice", "saved", "preflight", "finished_utc"):
            rec.pop(k, None)
        rec["error_type"] = "RuntimeError"
        rec.pop("audio")
        write(bench, rec, texts=False)
        benchdb.apply_load(bench)
        (status, valid, err), = rows(bench, "SELECT status, valid, error_type FROM run")
        self.assertEqual((status, valid, err), ("error", "no", "RuntimeError"))
        self.assertEqual(rows(bench, "SELECT count(*) FROM measurement"), [(0,)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM raw_output"), [(0,)])

    def test_valid_is_decided_by_status_and_preflight(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        write(bench, record("20261102-100000-bbbbbb", power="battery"))
        benchdb.apply_load(bench)
        got = dict(rows(bench, "SELECT run_id, valid FROM run"))
        self.assertEqual(got["20261101-100000-aaaaaa"], "yes")
        self.assertEqual(got["20261102-100000-bbbbbb"], "with_reservations")
        (reason,), = rows(bench, "SELECT valid_reason FROM run WHERE run_id='20261102-100000-bbbbbb'")
        self.assertIn("battery", reason)

    def test_decisions_come_from_saved_settings(self):
        bench = new_bench()
        three_runs(bench)
        benchdb.apply_load(bench)
        got = rows(bench, "SELECT run_id, setting, applied, chooser FROM decision ORDER BY setting")
        self.assertEqual(got, [("20261120-100000-bbbbbb", "asr_batch", 1, "bench.py"),
                               ("20261120-100000-bbbbbb", "asr_device", 1, "bench.py")])

    def test_decision_is_not_applied_when_config_was_not_written(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa", saved={"asr_batch": 8}, config_written=False))
        benchdb.apply_load(bench)
        self.assertEqual(rows(bench, "SELECT applied FROM decision"), [(0,)])

    def test_raw_output_is_only_a_reference_to_the_texts_file(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        benchdb.apply_load(bench)
        (name, sha, size), = rows(bench, "SELECT file_name, sha256, size_bytes FROM raw_output")
        f = bench / "runs" / name
        self.assertEqual(name, "20261101-100000-aaaaaa.texts.json")
        self.assertEqual(sha, hashlib.sha256(f.read_bytes()).hexdigest())
        self.assertEqual(size, f.stat().st_size)
        cols = [c[1] for c in rows(bench, "PRAGMA table_info(raw_output)")]
        self.assertNotIn("text", cols)

    def test_run_without_texts_file_has_no_raw_output_row(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"), texts=False)
        benchdb.apply_load(bench)
        self.assertEqual(rows(bench, "SELECT count(*) FROM raw_output"), [(0,)])

    def test_texts_files_are_not_loaded_as_runs(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        rep = benchdb.plan_load(bench)
        self.assertEqual([f["file"] for f in rep["files"]], ["20261101-100000-aaaaaa.json"])


class RepeatableLoadTests(unittest.TestCase):

    def test_second_load_does_not_change_the_base_at_all(self):
        bench = new_bench()
        three_runs(bench)
        benchdb.apply_load(bench)
        before_bytes = benchdb.db_path(bench).read_bytes()
        before = dump(bench)
        rep = benchdb.apply_load(bench)
        self.assertEqual(benchdb.db_path(bench).read_bytes(), before_bytes)
        self.assertEqual(dump(bench), before)
        self.assertEqual(rep["counts"], {"known": 3})

    def test_new_file_is_added_next_time_and_only_it(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        benchdb.apply_load(bench)
        write(bench, record("20261102-100000-bbbbbb"))
        rep = benchdb.apply_load(bench)
        self.assertEqual(rep["counts"], {"known": 1, "loaded": 1})
        self.assertEqual(rows(bench, "SELECT count(*) FROM run"), [(2,)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM test_unit"), [(1,)])      # справочник не дублируется
        self.assertEqual(rows(bench, "SELECT count(*) FROM testbed"), [(1,)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM env_snapshot"), [(1,)])

    def test_same_run_id_with_other_content_is_a_conflict_and_changes_no_data(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        benchdb.apply_load(bench)
        before = rows(bench, "SELECT * FROM run") + rows(bench, "SELECT * FROM measurement")
        # копия файла с правкой руками: тот же run_id в другом файле нельзя — берём то же имя, другое содержимое
        p = bench / "runs" / "20261101-100000-aaaaaa.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["duration_sec"] = 999.0
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        rep = benchdb.apply_load(bench)
        self.assertEqual(rep["counts"], {"conflict": 1})
        self.assertEqual(rows(bench, "SELECT * FROM run") + rows(bench, "SELECT * FROM measurement"), before)
        self.assertEqual(rows(bench, "SELECT outcome FROM import_log ORDER BY log_id"),
                         [("loaded",), ("conflict",)])
        again = benchdb.apply_load(bench)          # тот же изменённый файл — уже учтён, журнал не растёт
        self.assertEqual(again["counts"], {"known": 1})
        self.assertEqual(rows(bench, "SELECT count(*) FROM import_log"), [(2,)])

    def test_bad_files_are_rejected_and_others_still_loaded(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        (bench / "runs" / "broken.json").write_text("{not json", encoding="utf-8")
        newer = record("20261102-100000-bbbbbb")
        newer["schema_version"] = 99
        (bench / "runs" / "20261102-100000-bbbbbb.json").write_text(json.dumps(newer), encoding="utf-8")
        renamed = record("20261103-100000-cccccc")
        (bench / "runs" / "copy-of-run.json").write_text(json.dumps(renamed), encoding="utf-8")
        rep = benchdb.apply_load(bench)
        self.assertEqual(rep["counts"], {"loaded": 1, "rejected": 3})
        self.assertEqual(rows(bench, "SELECT count(*) FROM run"), [(1,)])
        again = benchdb.apply_load(bench)
        self.assertEqual(again["counts"], {"known": 4})
        self.assertEqual(rows(bench, "SELECT count(*) FROM import_log"), [(4,)])

    def test_unknown_method_is_loaded_but_not_hidden(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa", method="bench-7"))
        benchdb.apply_load(bench)
        self.assertEqual(rows(bench, "SELECT method_version FROM run"), [("bench-7",)])


class ReadIsSeparatedFromWriteTests(unittest.TestCase):

    def test_plan_does_not_create_the_base(self):
        bench = new_bench()
        three_runs(bench)
        before = tree_hashes(bench)
        rep = benchdb.plan_load(bench)
        self.assertEqual(rep["counts"], {"new": 3})
        self.assertFalse(benchdb.db_path(bench).exists())
        self.assertEqual(tree_hashes(bench), before)

    def test_plan_and_status_leave_an_existing_base_byte_identical(self):
        bench = new_bench()
        three_runs(bench)
        benchdb.apply_load(bench)
        write(bench, record("20261125-100000-eeeeee"))
        before = benchdb.db_path(bench).read_bytes()
        self.assertEqual(benchdb.plan_load(bench)["counts"], {"known": 3, "new": 1})
        st = benchdb.status(bench)
        self.assertEqual(st["by_method"], {"bench-1": 1, "bench-2": 2})
        self.assertEqual(st["not_loaded"], 1)
        self.assertEqual(benchdb.db_path(bench).read_bytes(), before)

    def test_status_without_base_says_so_and_creates_nothing(self):
        bench = new_bench()
        three_runs(bench)
        st = benchdb.status(bench)
        self.assertFalse(st["db_exists"])
        self.assertEqual(st["not_loaded"], 3)
        self.assertFalse(benchdb.db_path(bench).exists())

    def test_read_connection_cannot_write(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        benchdb.apply_load(bench)
        con = benchdb.open_read(bench)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                con.execute("INSERT INTO meta(key, value) VALUES ('x', 'y')")
        finally:
            con.close()

    def test_loading_never_touches_run_files(self):
        bench = new_bench()
        three_runs(bench)
        before = tree_hashes(bench / "runs")
        benchdb.apply_load(bench)
        benchdb.apply_load(bench)
        self.assertEqual(tree_hashes(bench / "runs"), before)

    def test_loading_does_not_open_units_or_testbed_files(self):
        bench = new_bench()
        three_runs(bench)
        (bench / "units.json").write_text('{"units": {"x": {"path": "C:\\\\secret\\\\a.wav"}}}', encoding="utf-8")
        (bench / "testbed.json").write_text('{"salt": "s"}', encoding="utf-8")
        before = {n: (bench / n).read_bytes() for n in ("units.json", "testbed.json")}
        benchdb.apply_load(bench)
        self.assertEqual({n: (bench / n).read_bytes() for n in before}, before)
        self.assertNotIn("secret", json.dumps(dump(bench), ensure_ascii=False))


class PrivacyTests(unittest.TestCase):

    def test_no_paths_names_or_texts_in_any_cell(self):
        bench = new_bench()
        three_runs(bench)
        benchdb.apply_load(bench)
        text = json.dumps(dump(bench), ensure_ascii=False)
        # в JSON одна обратная косая выглядит как две: «:\» — путь диска, «\\\\» — сетевой путь
        for bad in (":\\\\", "\\\\\\\\", "/Users/", SECRET_NAME, SECRET_WORD, "Иван", Path(_TMP).name):
            self.assertNotIn(bad, text, bad)

    def test_file_names_in_the_log_are_names_only(self):
        bench = new_bench()
        three_runs(bench)
        benchdb.apply_load(bench)
        for (name,) in rows(bench, "SELECT file_name FROM import_log"):
            self.assertNotIn("\\", name)
            self.assertNotIn("/", name)


class AppendOnlyTests(unittest.TestCase):

    def test_facts_cannot_be_updated_or_deleted(self):
        bench = new_bench()
        three_runs(bench)
        benchdb.apply_load(bench)
        con = sqlite3.connect(benchdb.db_path(bench))
        try:
            for table in ("run", "measurement", "testbed", "env_snapshot", "test_unit", "exec_config",
                          "model_artifact", "raw_output", "decision", "import_log"):
                with self.assertRaises(sqlite3.DatabaseError, msg=table):
                    con.execute(f"UPDATE {table} SET rowid = rowid")
                with self.assertRaises(sqlite3.DatabaseError, msg=table):
                    con.execute(f"DELETE FROM {table}")
        finally:
            con.close()


class CompareTests(unittest.TestCase):

    def load(self, **runs):
        bench = new_bench()
        for rid, kw in runs.items():
            write(bench, record(rid, **kw))
        benchdb.apply_load(bench)
        return bench

    def test_same_axes_are_comparable(self):
        bench = self.load(**{"20261101-100000-aaaaaa": {}, "20261102-100000-bbbbbb": {}})
        res = benchdb.compare(bench, "20261101-100000-aaaaaa:onnx-cpu", "20261102-100000-bbbbbb:onnx-cpu")
        self.assertEqual((res["verdict"], res["axes"]), ("comparable", []))

    def test_one_axis_differs_names_it(self):
        bench = self.load(**{"20261101-100000-aaaaaa": {}, "20261102-100000-bbbbbb": {"device_actual": "GPU"}})
        res = benchdb.compare(bench, "20261101-100000-aaaaaa:onnx-cpu", "20261102-100000-bbbbbb:onnx-cpu")
        self.assertEqual((res["verdict"], res["axes"]), ("by_axis", ["device"]))
        self.assertIn("устройств", res["text"])

    def test_package_driver_and_testbed_are_separate_axes(self):
        base = {"20261101-100000-aaaaaa": {}}
        bench = self.load(**base, **{"20261102-100000-bbbbbb": {"packages": {"onnxruntime": "1.21.0",
                                                                            "numpy": "2.0.0"}}})
        res = benchdb.compare(bench, "20261101-100000-aaaaaa:onnx-cpu", "20261102-100000-bbbbbb:onnx-cpu")
        self.assertEqual((res["verdict"], res["axes"]), ("by_axis", ["environment"]))
        hw = record("x")["testbed"]["hardware"]
        hw = copy.deepcopy(hw)
        hw["bios"]["version"] = "2.0"
        bench = self.load(**base, **{"20261102-100000-bbbbbb": {"hardware": hw}})
        res = benchdb.compare(bench, "20261101-100000-aaaaaa:onnx-cpu", "20261102-100000-bbbbbb:onnx-cpu")
        self.assertEqual((res["verdict"], res["axes"]), ("by_axis", ["testbed"]))

    def test_unrelated_package_does_not_change_the_environment_axis(self):
        bench = self.load(**{"20261101-100000-aaaaaa": {},
                             "20261102-100000-bbbbbb": {"packages": {"onnxruntime": "1.20.0", "numpy": "2.0.0",
                                                                    "huggingface-hub": "9.9.9"}}})
        res = benchdb.compare(bench, "20261101-100000-aaaaaa:onnx-cpu", "20261102-100000-bbbbbb:onnx-cpu")
        self.assertEqual(res["verdict"], "comparable")

    def test_more_than_one_axis_is_not_compared(self):
        bench = self.load(**{"20261101-100000-aaaaaa": {},
                             "20261102-100000-bbbbbb": {"device_actual": "GPU", "seed": b"z"}})
        res = benchdb.compare(bench, "20261101-100000-aaaaaa:onnx-cpu", "20261102-100000-bbbbbb:onnx-cpu")
        self.assertEqual(res["verdict"], "incomparable")
        self.assertEqual(sorted(res["axes"]), ["device", "unit"])

    def test_unmeasured_axis_makes_the_pair_incomparable(self):
        bench = self.load(**{"20261101-100000-aaaaaa": {}, "20261102-100000-bbbbbb": {"power": None}})
        res = benchdb.compare(bench, "20261101-100000-aaaaaa:onnx-cpu", "20261102-100000-bbbbbb:onnx-cpu")
        self.assertEqual(res["verdict"], "incomparable")
        self.assertIn("power", res["unmeasured"])

    def test_different_methods_are_never_compared(self):
        bench = self.load(**{"20261101-100000-aaaaaa": {"method": "bench-1"},
                             "20261102-100000-bbbbbb": {"method": "bench-2"}})
        res = benchdb.compare(bench, "20261101-100000-aaaaaa:onnx-cpu", "20261102-100000-bbbbbb:onnx-cpu")
        self.assertEqual(res["verdict"], "incomparable")
        self.assertIn("bench-1", res["text"])
        self.assertIn("bench-2", res["text"])
        # и ключ сравнимости у них разный, хотя остальные оси те же
        keys = rows(bench, "SELECT speed_key FROM measurement WHERE variant_id='onnx-cpu'")
        self.assertEqual(len({k for (k,) in keys}), 2)

    def test_unknown_measurement_is_reported(self):
        bench = self.load(**{"20261101-100000-aaaaaa": {}})
        res = benchdb.compare(bench, "20261101-100000-aaaaaa:onnx-cpu", "20261101-100000-aaaaaa:nope")
        self.assertEqual(res["verdict"], "unknown")


class HistoricalTests(unittest.TestCase):

    def file(self, entries, name="hist.json"):
        p = Path(_TMP) / f"{os.urandom(4).hex()}-{name}"
        p.write_text(json.dumps({"format": "bench-historical-1", "entries": entries}, ensure_ascii=False),
                     encoding="utf-8")
        return p

    ENTRIES = [{"entry_id": "h-001", "date": "2026-11-08", "machine": "Intel", "variant": "onnx-cpu",
                "family": "rnnt", "device": "CPU", "completeness": "V", "rtfx": 15.4, "load_sec": 142.0,
                "source": "условная ссылка 1"},
               {"entry_id": "h-002", "date": "2026-11-15", "machine": "AMD", "variant": "onnx-cpu",
                "family": "rnnt", "device": "CPU", "completeness": "B", "rtfx": 33.0, "source": "условная ссылка 2"}]

    def test_historical_rows_are_flagged_and_repeat_changes_nothing(self):
        bench = new_bench()
        f = self.file(self.ENTRIES)
        rep = benchdb.load_historical(bench, f, apply=True)
        self.assertEqual(rep["counts"], {"loaded": 1})
        self.assertEqual(rows(bench, "SELECT count(*), sum(is_historical) FROM run"), [(2, 2)])
        self.assertEqual(rows(bench, "SELECT DISTINCT method_version FROM run"), [("historical",)])
        self.assertEqual(sorted(rows(bench, "SELECT completeness FROM run")), [("B",), ("V",)])
        (rtfx, load, peak), = rows(bench, "SELECT rtfx, load_sec, peak_memory_mb FROM measurement "
                                          "WHERE run_id='hist-h-001'")
        self.assertEqual((rtfx, load, peak), (15.4, 142.0, None))
        before = dump(bench)
        again = benchdb.load_historical(bench, f, apply=True)
        self.assertEqual(again["counts"], {"known": 1})
        self.assertEqual(dump(bench), before)

    def test_historical_is_dry_run_by_default(self):
        bench = new_bench()
        rep = benchdb.load_historical(bench, self.file(self.ENTRIES), apply=False)
        self.assertEqual(rep["counts"], {"new": 1})
        self.assertFalse(benchdb.db_path(bench).exists())

    def test_changed_entry_with_known_id_is_a_conflict_not_an_overwrite(self):
        bench = new_bench()
        benchdb.load_historical(bench, self.file(self.ENTRIES), apply=True)
        changed = [dict(self.ENTRIES[0], rtfx=99.0), self.ENTRIES[1]]
        rep = benchdb.load_historical(bench, self.file(changed, "hist2.json"), apply=True)
        self.assertEqual(rep["counts"], {"loaded": 1})
        self.assertIn("1", rep["files"][0]["reason"])
        self.assertEqual(rows(bench, "SELECT rtfx FROM measurement WHERE run_id='hist-h-001'"), [(15.4,)])

    def test_historical_is_not_compared_with_new_runs(self):
        bench = new_bench()
        write(bench, record("20261124-100000-cccccc"))
        benchdb.apply_load(bench)
        benchdb.load_historical(bench, self.file(self.ENTRIES), apply=True)
        res = benchdb.compare(bench, "hist-h-001:onnx-cpu", "20261124-100000-cccccc:onnx-cpu")
        self.assertEqual(res["verdict"], "incomparable")
        self.assertIn("исторический", res["text"])

    def test_bad_historical_file_is_rejected(self):
        bench = new_bench()
        bad = Path(_TMP) / "bad-hist.json"
        bad.write_text(json.dumps({"format": "other", "entries": []}), encoding="utf-8")
        rep = benchdb.load_historical(bench, bad, apply=True)
        self.assertEqual(rep["counts"], {"rejected": 1})
        self.assertEqual(rows(bench, "SELECT count(*) FROM run"), [(0,)])

    def test_historical_text_is_scrubbed(self):
        bench = new_bench()
        e = dict(self.ENTRIES[0], source="лежит в 'C:\\Users\\Иван\\Документы\\замер.txt'")
        benchdb.load_historical(bench, self.file([e]), apply=True)
        text = json.dumps(dump(bench), ensure_ascii=False)
        self.assertNotIn("Иван", text)


class CommandTests(unittest.TestCase):

    def run_cli(self, bench, *argv):
        import io
        import contextlib
        import argparse
        ns = argparse.Namespace(bench_cmd=argv[0], apply="--apply" in argv,
                                historical=None, first=None, second=None)
        rest = [a for a in argv[1:] if a != "--apply"]
        if argv[0] == "compare":
            ns.first, ns.second = rest
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = benchdb.cli(ns, bench_dir=bench)
        return code, out.getvalue()

    def test_load_without_apply_only_reports(self):
        bench = new_bench()
        three_runs(bench)
        code, text = self.run_cli(bench, "load")
        self.assertEqual(code, 0)
        self.assertIn("--apply", text)
        self.assertFalse(benchdb.db_path(bench).exists())

    def test_load_with_apply_writes_and_repeats_quietly(self):
        bench = new_bench()
        three_runs(bench)
        self.assertEqual(self.run_cli(bench, "load", "--apply")[0], 0)
        self.assertEqual(rows(bench, "SELECT count(*) FROM run"), [(3,)])
        before = benchdb.db_path(bench).read_bytes()
        code, text = self.run_cli(bench, "load", "--apply")
        self.assertEqual(code, 0)
        self.assertEqual(benchdb.db_path(bench).read_bytes(), before)

    def test_status_and_compare_print_something_useful(self):
        bench = new_bench()
        three_runs(bench)
        self.run_cli(bench, "load", "--apply")
        code, text = self.run_cli(bench, "status")
        self.assertIn("bench-1", text)
        self.assertIn("bench-2", text)
        code, text = self.run_cli(bench, "compare", "20261120-100000-bbbbbb:onnx-cpu",
                                  "20261124-100000-cccccc:onnx-cpu")
        self.assertEqual(code, 0)
        self.assertIn("устройств", text)

    def test_newer_base_gives_nonzero_exit(self):
        bench = new_bench()
        benchdb.apply_load(bench)
        con = sqlite3.connect(benchdb.db_path(bench))
        con.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
        con.commit()
        con.close()
        write(bench, record("20261101-100000-aaaaaa"))
        code, text = self.run_cli(bench, "load", "--apply")
        self.assertNotEqual(code, 0)

    def test_rt_has_the_subcommand_and_the_help_says_how_to_remove_a_run(self):
        import rt
        a = rt.build().parse_args(["bench-db", "load", "--apply"])
        self.assertEqual((a.cmd, a.bench_cmd, a.apply), ("bench-db", "load", True))
        with mock.patch.object(benchdb, "cli", return_value=0) as cli:
            rt.run(a)
        cli.assert_called_once()
        self.assertIn("удалением файла базы", benchdb.HELP_APPEND_ONLY)
        self.assertIn("удалением файла базы", (Path(rt.__file__).read_text(encoding="utf-8")))


class BrokenStructureTests(unittest.TestCase):
    """Файл, прошедший kind, schema_version и run_id, но с неверной структурой, не роняет загрузку."""

    BREAKS = {
        "hardware_is_string": lambda r: r["testbed"].__setitem__("hardware", "oops"),
        "chunks_nan": lambda r: r["results"][0].__setitem__("chunks", float("nan")),
        "asr_sec_nan": lambda r: r["results"][0].__setitem__("asr_sec", float("nan")),
        "chunks_infinity": lambda r: r["results"][0].__setitem__("chunks", float("inf")),
        "started_is_dict": lambda r: r.__setitem__("started_utc", {"a": 1}),
        "reasons_is_number": lambda r: r["preflight"].__setitem__("reasons", 5),
        "duplicate_variant_id": lambda r: r["results"].append(dict(r["results"][0])),
        "results_is_string": lambda r: r.__setitem__("results", "x"),
        "models_is_string": lambda r: r.__setitem__("models", "x"),
        "environment_is_list": lambda r: r.__setitem__("environment", [1, 2]),
        "devices_not_list_of_dicts": lambda r: r["testbed"]["hardware"].__setitem__("devices", ["x", 3]),
        "huge_integer": lambda r: r["results"][0].__setitem__("chunks", 10 ** 30),
        "params_is_list": lambda r: r.__setitem__("params", [1]),
    }

    def setup(self, name):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        broken = record("20261102-100000-bbbbbb")
        self.BREAKS[name](broken)
        write(bench, broken)
        write(bench, record("20261103-100000-cccccc"))
        return bench

    def test_middle_broken_file_is_rejected_and_the_rest_loaded(self):
        for name in self.BREAKS:
            with self.subTest(name):
                bench = self.setup(name)
                plan = benchdb.plan_load(bench)                     # показ тоже не падает
                self.assertEqual(plan["counts"], {"new": 2, "rejected": 1}, plan)
                rep = benchdb.apply_load(bench)
                self.assertEqual(rep["counts"], {"loaded": 2, "rejected": 1}, rep)
                self.assertEqual(sorted(r for (r,) in rows(bench, "SELECT run_id FROM run")),
                                 ["20261101-100000-aaaaaa", "20261103-100000-cccccc"])
                (outcome, reason, run_id), = rows(bench, "SELECT outcome, reason, run_id FROM import_log "
                                                         "WHERE outcome='rejected'")
                self.assertTrue(reason)
                self.assertEqual(run_id, "20261102-100000-bbbbbb")
                self.assertEqual(benchdb.apply_load(bench)["counts"], {"known": 3})

    def test_failure_in_the_middle_of_a_file_rolls_back_only_that_file(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        write(bench, record("20261102-100000-bbbbbb"))
        write(bench, record("20261103-100000-cccccc"))
        real = benchdb._insert

        def failing(con, table, row, ignore=False):
            if table == "measurement" and row["run_id"] == "20261102-100000-bbbbbb":
                raise sqlite3.IntegrityError("синтетический сбой")
            return real(con, table, row, ignore)
        with mock.patch.object(benchdb, "_insert", failing):
            rep = benchdb.apply_load(bench)
        self.assertEqual(rep["counts"], {"loaded": 2, "rejected": 1})
        self.assertEqual(rows(bench, "SELECT count(*) FROM run WHERE run_id='20261102-100000-bbbbbb'"), [(0,)])
        self.assertEqual(rows(bench, "SELECT count(*) FROM measurement WHERE run_id='20261102-100000-bbbbbb'"),
                         [(0,)])                                    # строка прогона откатилась вместе с измерением
        self.assertEqual(rows(bench, "SELECT count(*) FROM run"), [(2,)])

    def test_locked_base_is_not_recorded_as_a_rejected_file(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        real = benchdb._insert

        def locked(con, table, row, ignore=False):
            if table == "measurement":
                raise sqlite3.OperationalError("database is locked")
            return real(con, table, row, ignore)
        with mock.patch.object(benchdb, "_insert", locked):
            with self.assertRaises(benchdb.BenchDbError):
                benchdb.apply_load(bench)
        self.assertEqual(rows(bench, "SELECT count(*) FROM import_log"), [(0,)])
        self.assertEqual(benchdb.apply_load(bench)["counts"], {"loaded": 1})

    def test_unreadable_file_does_not_stop_the_load(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        write(bench, record("20261102-100000-bbbbbb"))
        write(bench, record("20261103-100000-cccccc"))
        real = Path.read_bytes

        def read(self):
            if self.name == "20261102-100000-bbbbbb.json":
                raise PermissionError("синтетический отказ")
            return real(self)
        with mock.patch.object(Path, "read_bytes", read):
            rep = benchdb.apply_load(bench)
        self.assertEqual(rep["counts"], {"loaded": 2, "rejected": 1})
        self.assertEqual([i["reason"] for i in rep["files"] if i["action"] == "rejected"], ["файл не читается"])
        self.assertEqual(rows(bench, "SELECT count(*) FROM import_log"), [(2,)])     # нечитаемый не записан
        self.assertEqual(benchdb.apply_load(bench)["counts"], {"known": 2, "loaded": 1})

    def test_not_a_dict_foreign_kind_and_bad_run_id_are_rejected(self):
        bench = new_bench()
        runs = bench / "runs"
        (runs / "list.json").write_text("[1, 2]", encoding="utf-8")
        foreign = record("20261102-100000-bbbbbb")
        foreign["kind"] = "eval"
        (runs / "20261102-100000-bbbbbb.json").write_text(json.dumps(foreign), encoding="utf-8")
        bad = record("../escape")
        (runs / "x.json").write_text(json.dumps(bad), encoding="utf-8")
        nan = (runs / "nan.json")
        nan.write_text('{"kind": "bench", "schema_version": NaN, "run_id": "a1"}', encoding="utf-8")
        rep = benchdb.apply_load(bench)
        self.assertEqual(rep["counts"], {"rejected": 4})
        reasons = " | ".join(i["reason"] for i in rep["files"])
        for part in ("не объект", "вид файла", "run_id"):
            self.assertIn(part, reasons)

    def test_cli_never_shows_a_traceback(self):
        import contextlib
        import io
        import argparse
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        broken = record("20261102-100000-bbbbbb")
        self.BREAKS["hardware_is_string"](broken)
        write(bench, broken)
        for cmd, apply in (("status", False), ("load", False), ("load", True), ("status", False)):
            out, err = io.StringIO(), io.StringIO()
            ns = argparse.Namespace(bench_cmd=cmd, apply=apply, historical=None, first=None, second=None)
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = benchdb.cli(ns, bench_dir=bench)
            self.assertEqual(code, 0, out.getvalue() + err.getvalue())
            self.assertNotIn("Traceback", out.getvalue() + err.getvalue())
        out, err = io.StringIO(), io.StringIO()
        ns = argparse.Namespace(bench_cmd="load", apply=False, historical=None, first=None, second=None)
        with mock.patch.object(benchdb, "plan_load", side_effect=RuntimeError("синтетический сбой")), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = benchdb.cli(ns, bench_dir=bench)
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", out.getvalue() + err.getvalue())
        self.assertIn("RuntimeError", err.getvalue())


class RejectedIsNotForeverTests(unittest.TestCase):

    def test_rejected_file_is_rechecked_by_a_newer_loader_only(self):
        bench = new_bench()
        newer = record("20261101-100000-aaaaaa")
        newer["schema_version"] = 2
        (bench / "runs" / "20261101-100000-aaaaaa.json").write_text(json.dumps(newer), encoding="utf-8")
        self.assertEqual(benchdb.apply_load(bench)["counts"], {"rejected": 1})
        self.assertEqual(benchdb.apply_load(bench)["counts"], {"known": 1})       # тот же загрузчик — не заново
        with mock.patch.object(benchdb, "RUN_FORMATS", (1, 2)), mock.patch.object(benchdb, "LOADER_VERSION", "benchdb-next"):
            self.assertEqual(benchdb.plan_load(bench)["counts"], {"new": 1})
            self.assertEqual(benchdb.apply_load(bench)["counts"], {"loaded": 1})
        self.assertEqual(rows(bench, "SELECT loader_version, outcome FROM import_log ORDER BY log_id"),
                         [(benchdb.LOADER_VERSION, "rejected"), ("benchdb-next", "loaded")])    # журнал дописан
        self.assertEqual(rows(bench, "SELECT count(*) FROM run"), [(1,)])

    def test_same_loader_version_with_changed_rule_does_not_recheck(self):
        bench = new_bench()
        (bench / "runs" / "broken.json").write_text("{", encoding="utf-8")
        benchdb.apply_load(bench)
        before = benchdb.db_path(bench).read_bytes()
        benchdb.apply_load(bench)
        self.assertEqual(benchdb.db_path(bench).read_bytes(), before)

    def test_texts_file_that_appears_later_is_added_to_a_loaded_run(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        texts = bench / "runs" / "20261101-100000-aaaaaa.texts.json"
        saved = texts.read_bytes()
        texts.unlink()                                              # файл прогона скопирован раньше файла текстов
        benchdb.apply_load(bench)
        self.assertEqual(rows(bench, "SELECT count(*) FROM raw_output"), [(0,)])
        texts.write_bytes(saved)
        rep = benchdb.apply_load(bench)
        self.assertEqual(rep["counts"], {"known": 1})
        self.assertEqual(rows(bench, "SELECT file_name, sha256 FROM raw_output"),
                         [(texts.name, hashlib.sha256(saved).hexdigest())])
        before = benchdb.db_path(bench).read_bytes()
        benchdb.apply_load(bench)
        self.assertEqual(benchdb.db_path(bench).read_bytes(), before)             # второй раз — без записи

    def test_late_texts_written_by_another_loader_meanwhile_is_not_an_error(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        texts = bench / "runs" / "20261101-100000-aaaaaa.texts.json"
        saved = texts.read_bytes()
        texts.unlink()
        benchdb.apply_load(bench)
        texts.write_bytes(saved)
        real = benchdb._texts_ref

        def other_loader_first(bench_dir, parsed):                  # пока ждали блокировку, ссылку дописал другой
            ref = real(bench_dir, parsed)
            con = sqlite3.connect(benchdb.db_path(bench))
            con.execute("INSERT INTO raw_output(run_id, file_name, sha256, size_bytes) VALUES (?, ?, ?, ?)",
                        (ref["run_id"], ref["file_name"], ref["sha256"], ref["size_bytes"]))
            con.commit()
            con.close()
            return ref

        with mock.patch.object(benchdb, "_texts_ref", other_loader_first):
            rep = benchdb.apply_load(bench)
        self.assertEqual(rep["counts"], {"known": 1})
        self.assertEqual(rows(bench, "SELECT count(*) FROM raw_output"), [(1,)])


class ConnectionAndSchemaTests(unittest.TestCase):

    def test_refused_base_is_closed_at_once(self):
        bench = new_bench()
        benchdb.apply_load(bench)
        con = sqlite3.connect(benchdb.db_path(bench))
        con.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
        con.commit()
        con.close()
        with self.assertRaises(benchdb.BenchDbError):
            benchdb.apply_load(bench)
        os.remove(benchdb.db_path(bench))                           # на Windows не удалится, если соединение открыто

    def test_foreign_tables_are_refused_and_closed(self):
        bench = new_bench()
        con = sqlite3.connect(benchdb.db_path(bench))
        con.execute("CREATE TABLE other (x)")
        con.commit()
        con.close()
        before = benchdb.db_path(bench).read_bytes()
        with self.assertRaises(benchdb.BenchDbError):
            benchdb.apply_load(bench)
        self.assertEqual(benchdb.db_path(bench).read_bytes(), before)
        os.remove(benchdb.db_path(bench))

    def test_garbage_file_is_refused_by_both_openers_and_closed(self):
        bench = new_bench()
        benchdb.db_path(bench).write_bytes(b"this is not a sqlite database " * 20)
        for opener in (benchdb.open_read, benchdb._open_write):
            with self.assertRaises(benchdb.BenchDbError):
                opener(bench)
        os.remove(benchdb.db_path(bench))

    def test_empty_file_is_not_a_base_for_readers_and_is_filled_by_the_loader(self):
        bench = new_bench()
        benchdb.db_path(bench).write_bytes(b"")
        self.assertFalse(benchdb.status(bench)["db_exists"])
        self.assertEqual(benchdb.plan_load(bench)["counts"], {})
        write(bench, record("20261101-100000-aaaaaa"))
        self.assertEqual(benchdb.apply_load(bench)["counts"], {"loaded": 1})

    def test_schema_creation_is_atomic(self):
        bench = new_bench()
        with mock.patch.object(benchdb.runinfo, "utc_now", side_effect=RuntimeError("синтетический сбой")):
            with self.assertRaises(RuntimeError):
                benchdb.apply_load(bench)
        con = sqlite3.connect(benchdb.db_path(bench))
        try:
            self.assertEqual(con.execute("SELECT count(*) FROM sqlite_master").fetchone(), (0,))   # ни meta, ни таблиц
        finally:
            con.close()
        write(bench, record("20261101-100000-aaaaaa"))
        self.assertEqual(benchdb.apply_load(bench)["counts"], {"loaded": 1})

    def test_simultaneous_first_start_gives_one_base_and_no_false_refusals(self):
        import threading
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        barrier = threading.Barrier(4)
        results, errors = [], []

        def worker():
            barrier.wait()
            try:
                results.append(benchdb.apply_load(bench)["counts"])
            except Exception as ex:                                 # noqa: BLE001 — тест собирает любой сбой
                errors.append(repr(ex))
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(sorted(sum((list(c) for c in results), [])), ["known", "known", "known", "loaded"])
        self.assertEqual(rows(bench, "SELECT count(*) FROM run"), [(1,)])
        self.assertEqual(rows(bench, "SELECT value FROM meta WHERE key='schema_version'"), [("1",)])

    def test_newer_schema_stops_plan_and_historical_dry_run(self):
        bench = new_bench()
        benchdb.apply_load(bench)
        con = sqlite3.connect(benchdb.db_path(bench))
        con.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
        con.commit()
        con.close()
        write(bench, record("20261101-100000-aaaaaa"))
        hist = Path(_TMP) / f"{os.urandom(4).hex()}-h.json"
        hist.write_text(json.dumps({"format": "bench-historical-1", "entries": []}), encoding="utf-8")
        with self.assertRaises(benchdb.BenchDbError):
            benchdb.plan_load(bench)
        with self.assertRaises(benchdb.BenchDbError):
            benchdb.load_historical(bench, hist, apply=False)
        with self.assertRaises(benchdb.BenchDbError):
            benchdb.status(bench)
        os.remove(benchdb.db_path(bench))


class UncoveredBeforeTests(unittest.TestCase):
    """Поломки, которых прежние тесты не замечали."""

    def test_path_in_a_run_field_does_not_reach_the_base(self):
        bench = new_bench()
        rec = record("20261101-100000-aaaaaa")
        rec["params"]["model"] = "C:\\Users\\Иван\\модели\\rnnt.onnx"
        rec["variants"][0]["label"] = "лежит в \\\\сервер\\папка\\файл"
        rec["results"][0]["note"] = "/home/ivan/запись.wav"
        rec["choice"]["main"] = "D:\\Записи\\Секрет\\a.wav"
        write(bench, rec)
        benchdb.apply_load(bench)
        text = json.dumps(dump(bench), ensure_ascii=False)
        for bad in ("Иван", "сервер", "ivan", "Секрет", "D:\\\\"):
            self.assertNotIn(bad, text, bad)
        self.assertIn("<путь>", text)

    def axes(self, bench, run_id="20261101-100000-aaaaaa", variant="onnx-cpu"):
        (a,), = rows(bench, "SELECT axes_json FROM measurement WHERE run_id=? AND variant_id=?", run_id, variant)
        return json.loads(a)

    def test_axes_known_flags_and_values(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        benchdb.apply_load(bench)
        ax = self.axes(bench)
        self.assertTrue(all(ax[k]["known"] for k in benchdb.AXES))
        self.assertEqual(ax["device"]["value"], {"backend": "cpu", "providers": ["CPUExecutionProvider"],
                                                 "device": "CPU"})
        self.assertEqual(ax["unit"]["value"]["duration"], 1.0)
        self.assertEqual(ax["environment"]["value"]["packages"], {"numpy": "2.0.0", "onnxruntime": "1.20.0"})
        self.assertEqual(ax["environment"]["value"]["drivers"], [["Synthetic GPU", "1.2.3"]])
        self.assertEqual(ax["power"]["value"]["source"], "ac")
        self.assertEqual(ax["model"]["value"], {"name": "v3_e2e_rnnt", "revision": None})
        self.assertEqual(ax["config"]["value"]["requested"]["asr_backend"], "onnx-cpu")

    def test_axes_unknown_when_the_run_file_lacks_the_data(self):
        bench = new_bench()
        rec = record("20261101-100000-aaaaaa", hardware={"status": "not_measured", "reason": "нет"})
        rec["environment"]["packages"]["key"] = {}
        rec["results"][0].pop("backend_name")
        del rec["audio"]
        write(bench, rec)
        benchdb.apply_load(bench)
        ax = self.axes(bench)
        self.assertEqual({k for k in benchdb.AXES if not ax[k]["known"]}, {"unit", "device", "testbed", "environment"})
        self.assertTrue(ax["power"]["known"] and ax["model"]["known"] and ax["config"]["known"])

    def test_model_revision_is_taken_from_the_hf_cache_entry(self):
        bench = new_bench()
        rec = record("20261101-100000-aaaaaa")
        rec["variants"][0]["asr_model"] = "onnx-community/whisper-large-v3-turbo"
        write(bench, rec)
        benchdb.apply_load(bench)
        self.assertEqual(self.axes(bench)["model"]["value"]["revision"], "r" * 40)

    def test_status_counts_conflicts_and_rejected(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        (bench / "runs" / "broken.json").write_text("{", encoding="utf-8")
        benchdb.apply_load(bench)
        p = bench / "runs" / "20261101-100000-aaaaaa.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["duration_sec"] = 1.0
        p.write_text(json.dumps(data), encoding="utf-8")
        benchdb.apply_load(bench)
        st = benchdb.status(bench)
        self.assertEqual((st["conflicts"], st["rejected"], st["runs"], st["not_loaded"]), (1, 1, 1, 0))

    def test_texts_reference_needs_the_matching_name_and_a_readable_file(self):
        bench = new_bench()
        write(bench, record("20261101-100000-aaaaaa"))
        p = bench / "runs" / "20261101-100000-aaaaaa.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["texts_file"] = "other.texts.json"                    # имя не по run_id
        p.write_text(json.dumps(data), encoding="utf-8")
        write(bench, record("20261102-100000-bbbbbb"))
        (bench / "runs" / "20261102-100000-bbbbbb.texts.json").unlink()     # записан в файле прогона, но исчез
        rep = benchdb.apply_load(bench)
        self.assertEqual(rep["counts"], {"loaded": 2})
        self.assertEqual(rows(bench, "SELECT count(*) FROM raw_output"), [(0,)])

    def test_historical_foreign_format_is_rejected_and_logged(self):
        bench = new_bench()
        bad = Path(_TMP) / f"{os.urandom(4).hex()}-bad.json"
        bad.write_text(json.dumps({"format": "other", "entries": []}), encoding="utf-8")
        rep = benchdb.load_historical(bench, bad, apply=True)
        self.assertEqual(rep["counts"], {"rejected": 1})
        (name, outcome, reason), = rows(bench, "SELECT file_name, outcome, reason FROM import_log")
        self.assertEqual(outcome, "rejected")
        self.assertTrue(name.startswith("historical-"))
        self.assertNotIn(bad.name, name)
        self.assertEqual(benchdb.load_historical(bench, bad, apply=True)["counts"], {"known": 1})

    def test_status_uses_correct_plural_forms(self):
        for n, word in ((1, "прогон"), (2, "прогона"), (5, "прогонов"), (11, "прогонов"), (21, "прогон"), (22, "прогона")):
            self.assertEqual(benchdb.plural(n, ("прогон", "прогона", "прогонов")), word, n)


class FilesTests(unittest.TestCase):

    def test_module_is_utf8_without_bom_and_lf(self):
        raw = (Path(benchdb.__file__)).read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r", raw)
        raw.decode("utf-8")

    def test_module_is_registered_for_installation(self):
        import buildinfo
        self.assertIn("benchdb.py", buildinfo.root_files(Path(benchdb.__file__).parent))


if __name__ == "__main__":
    unittest.main()
