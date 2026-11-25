"""Паспорт прогона теста скорости (runinfo.py).

Проверяется то, от чего зависит сопоставимость прогонов и приватность: контрольные суммы звука,
отсутствие путей и имён в файле прогона, запись без перезаписи прежних прогонов, локальный
указатель файлов, устойчивость к сведениям, которые не удалось получить.

Всё на синтетике во временной папке. Сведения о реальной системе (питание, железо) могут быть
неизмеренными — тесты требуют только, чтобы сбор не падал.

Запуск: python -m unittest discover -s tests
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jobslots  # noqa: E402
import runinfo  # noqa: E402

PCM_1S = b"\x01\x00" * 16000          # одна секунда: моно, 16 кГц, s16le


class AudioIdentityTests(unittest.TestCase):
    def test_no_path_and_no_name(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "Совещание отдела.WAV"
            src.write_bytes(b"RIFF-synthetic")
            ident = runinfo.audio_identity(src, PCM_1S, 16000, source_samples=32000)
        dump = json.dumps(ident, ensure_ascii=False)
        self.assertNotIn("Совещание", dump)
        self.assertNotIn(Path(d).name, dump)
        self.assertEqual(ident["audio_sha256"], hashlib.sha256(b"RIFF-synthetic").hexdigest())
        self.assertEqual(ident["source_format"], "wav")
        self.assertEqual(ident["source_duration_sec"], 2.0)
        self.assertEqual(ident["excerpt"]["duration_sec"], 1.0)
        self.assertEqual(ident["excerpt"]["pcm_format"], "mono 16000 Hz s16le")
        self.assertEqual(ident["unit_id"], hashlib.sha256(PCM_1S).hexdigest()[:12])

    def test_same_sound_in_other_container_is_same_unit(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.wav", Path(d) / "b.ogg"
            a.write_bytes(b"container-one")
            b.write_bytes(b"container-two")
            ia = runinfo.audio_identity(a, PCM_1S, 16000)
            ib = runinfo.audio_identity(b, PCM_1S, 16000)
        self.assertEqual(ia["unit_id"], ib["unit_id"])
        self.assertNotEqual(ia["audio_sha256"], ib["audio_sha256"])
        self.assertIsNone(ia["source_duration_sec"])

    def test_missing_file_is_not_an_error(self):
        ident = runinfo.audio_identity(Path(tempfile.gettempdir()) / "rt-no-such-file.wav", PCM_1S, 16000)
        self.assertIsNone(ident["audio_sha256"])
        self.assertIsNone(ident["size_bytes"])
        self.assertEqual(len(ident["excerpt"]["pcm_sha256"]), 64)


class AudioFactsTests(unittest.TestCase):
    """Сведения о файле аудио: общее правило расшифровки и переноса (runinfo.audio_facts)."""

    def test_missing_file_gives_only_availability(self):
        facts = runinfo.audio_facts(Path(tempfile.gettempdir()) / "rt-no-such-file.wav", ffprobe="нет-ffprobe")
        self.assertEqual(facts, {"audio_available": False})

    def test_file_hash_without_ffprobe(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "a.wav"
            src.write_bytes(b"RIFF-synthetic")
            facts = runinfo.audio_facts(src, ffprobe="нет-ffprobe")
        self.assertEqual(facts, {"audio_available": True,
                                 "audio_sha256": hashlib.sha256(b"RIFF-synthetic").hexdigest()})

    def test_container_and_codec_from_ffprobe(self):
        out = json.dumps({"format": {"format_name": "mov,mp4,m4a"},
                          "streams": [{"codec_type": "video", "codec_name": "h264"},
                                      {"codec_type": "audio", "codec_name": "aac"}]})
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "a.m4a"
            src.write_bytes(b"synthetic")
            with mock.patch.object(runinfo.subprocess, "run", return_value=mock.Mock(stdout=out)) as run:
                facts = runinfo.audio_facts(src, ffprobe="ffprobe-synthetic")
        self.assertEqual(run.call_args[0][0][0], "ffprobe-synthetic")
        self.assertEqual(facts["container"], "mov,mp4,m4a")
        self.assertEqual(facts["codec"], "aac")
        self.assertEqual(facts["audio_sha256"], hashlib.sha256(b"synthetic").hexdigest())

    def test_ffprobe_garbage_is_not_invented(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "a.wav"
            src.write_bytes(b"synthetic")
            with mock.patch.object(runinfo.subprocess, "run", return_value=mock.Mock(stdout="не json")):
                facts = runinfo.audio_facts(src, ffprobe="ffprobe-synthetic")
        self.assertEqual(set(facts), {"audio_available", "audio_sha256"})


class EnvironmentTests(unittest.TestCase):
    def test_program_version_is_read_without_import(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "server.py").write_text('import nothing_like_this\nVERSION = "2000.1.2.3"   # версия\n'
                                               'raise SystemExit("сервер импортировать нельзя")\n', encoding="utf-8")
            self.assertEqual(runinfo.program_version(d), "2000.1.2.3")
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(runinfo.program_version(d))

    def test_code_revision_follows_file_content(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.py").write_text("x = 1\n", encoding="utf-8")
            (Path(d) / "notes.txt").write_text("не код\n", encoding="utf-8")
            first = runinfo.code_revision(d)
            (Path(d) / "notes.txt").write_text("другое\n", encoding="utf-8")
            self.assertEqual(runinfo.code_revision(d), first)          # не .py — ревизия та же
            (Path(d) / "a.py").write_text("x = 2\n", encoding="utf-8")
            second = runinfo.code_revision(d)
        self.assertEqual(first["files"], 1)
        self.assertEqual(len(first["files_sha256"]), 16)
        self.assertNotEqual(first["files_sha256"], second["files_sha256"])
        self.assertIsNone(first["git_revision"])                       # папка без Git

    def test_packages_are_stable_and_local_sources_are_hidden(self):
        first, second = runinfo.packages(), runinfo.packages()
        self.assertEqual(first, second)
        self.assertTrue(set(first["key"]) <= set(runinfo.KEY_PACKAGES))
        if first["count"]:
            self.assertEqual(len(first["sha256"]), 64)
        for src in (first["sources"] or {}).values():
            self.assertTrue(src["url"].startswith("http"))             # file:// содержал бы путь

    def test_environment_has_all_sections(self):
        with tempfile.TemporaryDirectory() as d:
            env = runinfo.environment(d)
        self.assertEqual(set(env), {"os", "python", "packages", "ffmpeg", "program_version", "code"})
        self.assertNotIn("product", env["os"])
        json.dumps(env)


class ConditionsTests(unittest.TestCase):
    GOOD = {"power_source": "ac", "cpu_load_percent": 3.0, "memory": {"free_mb": 9000}, "other_heavy_jobs": 0}

    def test_collecting_conditions_never_fails(self):
        cond = runinfo.conditions(load_interval=0.0)
        for key in ("taken_utc", "cpu_load_percent", "power_source", "battery_percent", "power", "memory",
                    "uptime_sec", "other_heavy_jobs"):
            self.assertIn(key, cond)
        self.assertIn(cond["power_source"], (None, "ac", "battery"))
        json.dumps(cond)

    def test_preflight_good_run(self):
        self.assertEqual(runinfo.preflight(self.GOOD, self.GOOD), {"ok": True, "reasons": []})

    def test_preflight_unknown_values_do_not_mark_the_run(self):
        unknown = {"power_source": None, "cpu_load_percent": None, "memory": {"free_mb": None},
                   "other_heavy_jobs": None}
        self.assertTrue(runinfo.preflight(unknown, unknown)["ok"])

    def test_preflight_reasons(self):
        cases = [({**self.GOOD, "power_source": "battery"}, None, False, "battery"),
                 (self.GOOD, {**self.GOOD, "power_source": "battery"}, False, "power_source_changed"),
                 ({**self.GOOD, "cpu_load_percent": 40.0}, None, False, "cpu_load"),
                 ({**self.GOOD, "memory": {"free_mb": 1000}}, None, False, "low_memory"),
                 ({**self.GOOD, "other_heavy_jobs": 1}, None, False, "other_heavy_jobs"),
                 (self.GOOD, {**self.GOOD, "other_heavy_jobs": 1}, False, "other_heavy_jobs"),
                 (self.GOOD, self.GOOD, True, "model_downloaded")]
        for start, end, downloaded, reason in cases:
            res = runinfo.preflight(start, end, downloaded)
            self.assertFalse(res["ok"], reason)
            self.assertIn(reason, res["reasons"])
            self.assertIn(reason, runinfo.REASONS_RU)

    def test_other_heavy_jobs_counts_locked_slots(self):
        with tempfile.TemporaryDirectory() as d:
            jobs = Path(d) / "jobs"
            self.assertEqual(runinfo.other_heavy_jobs(jobs, 2), 0)     # очереди ещё нет
            held = jobslots.acquire(jobs, 2, job_id="synthetic-1")
            try:
                self.assertEqual(runinfo.other_heavy_jobs(jobs, 2, own_slots=0), 1)
                self.assertEqual(runinfo.other_heavy_jobs(jobs, 2, own_slots=1), 0)
                second = jobslots.acquire(jobs, 2, job_id="synthetic-2")   # проба места не заняла
                try:
                    self.assertEqual(runinfo.other_heavy_jobs(jobs, 2, own_slots=1), 1)
                finally:
                    second.close()
            finally:
                held.close()
            self.assertEqual(runinfo.other_heavy_jobs(jobs, 2), 0)


class HardwareTests(unittest.TestCase):
    ANSWER = {"cpu": {"model": "Synthetic CPU 9000", "cores": 4, "threads": 8, "max_mhz": 3000},
              "system": {"manufacturer": "Synthetic", "model": "Model-1"},
              "bios": {"version": "1.0", "date": "2026-01-02"},
              "memory": {"modules": 2, "total_mb": 16384, "speed_mhz": 4800},
              "devices": [{"name": "Synthetic Graphics", "class": "DISPLAY", "driver_version": "1.2.3.4",
                           "driver_date": "2026-03-04"}, "мусор"]}

    def test_powershell_answer_is_parsed(self):
        hw = runinfo.hardware(run=lambda script: "\ufeff" + json.dumps(self.ANSWER))
        self.assertEqual(hw["status"], "ok")
        self.assertEqual(hw["cpu"]["model"], "Synthetic CPU 9000")
        self.assertIn("avx2", hw["cpu"]["features"])
        self.assertEqual([d["class"] for d in hw["devices"]], ["DISPLAY"])
        self.assertEqual(hw["bios"]["date"], "2026-01-02")

    def test_failures_are_not_measured_without_exception_text(self):
        def boom(script):
            raise RuntimeError(r"C:\Users\someone\secret")
        for run in (lambda s: "", lambda s: "не JSON", lambda s: "[1, 2]", lambda s: None, boom):
            hw = runinfo.hardware(run=run)
            self.assertEqual(hw["status"], "not_measured")
            self.assertNotIn("someone", json.dumps(hw, ensure_ascii=False))
            self.assertIn("features", hw["cpu"])

    def test_testbed_id_is_stable_and_salted(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a = runinfo.testbed_id(d1, machine="machine-A")
            self.assertEqual(a, runinfo.testbed_id(d1, machine="machine-A"))
            self.assertNotEqual(a, runinfo.testbed_id(d1, machine="machine-B"))
            self.assertNotEqual(a, runinfo.testbed_id(d2, machine="machine-A"))   # другая соль
            stored = (Path(d1) / "testbed.json").read_text(encoding="utf-8")
        self.assertEqual(len(a), 16)
        self.assertNotIn("machine-A", stored)


class ModelsCacheTests(unittest.TestCase):
    def _layout(self, d):
        cache = Path(d) / "cache"
        snap = cache / "huggingface" / "hub" / "models--org--synthetic-model" / "snapshots" / ("a" * 40)
        snap.mkdir(parents=True)
        (snap / "model.onnx").write_bytes(b"0" * 2048)
        refs = snap.parent.parent / "refs"
        refs.mkdir()
        (refs / "main").write_text("a" * 40 + "\n", encoding="ascii")
        (cache / "gigaam").mkdir()
        (cache / "gigaam" / "tokenizer.model").write_bytes(b"0" * 10)
        (cache / "gigaam" / "weights.ckpt").write_bytes(b"0" * runinfo.DOWNLOAD_BYTES)
        return cache

    def test_inventory_has_revisions_and_no_absolute_paths(self):
        with tempfile.TemporaryDirectory() as d:
            cache = self._layout(d)
            inv = runinfo.models_cache(cache, hf_home=cache / "huggingface")
            dump = json.dumps(inv, ensure_ascii=False)
            self.assertNotIn(Path(d).name, dump)
        self.assertEqual(inv["hf"], [{"repo": "org/synthetic-model", "ref_main": "a" * 40,
                                      "snapshots": ["a" * 40], "files": 1, "size_mb": 0.0}])
        self.assertEqual([x["file"] for x in inv["local"]], ["gigaam/weights.ckpt"])

    def test_empty_cache(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(runinfo.models_cache(d, hf_home=Path(d) / "нет"), {"hf": [], "local": []})
            self.assertEqual(runinfo.cache_counts(d, hf_home=Path(d) / "нет"),
                             {"openvino_files": 0, "model_files": 0, "model_bytes": 0})

    def test_cache_delta_shows_compile_and_download(self):
        with tempfile.TemporaryDirectory() as d:
            cache = self._layout(d)
            hf = cache / "huggingface"
            before = runinfo.cache_counts(cache, hf_home=hf)
            (cache / "openvino").mkdir()
            (cache / "openvino" / "1.blob").write_bytes(b"0" * 10)
            (cache / "gigaam" / "small.json").write_bytes(b"{}")
            compiled = runinfo.cache_counts(cache, hf_home=hf)
            (cache / "gigaam" / "new.ckpt").write_bytes(b"0" * runinfo.DOWNLOAD_BYTES)
            downloaded = runinfo.cache_counts(cache, hf_home=hf)
        self.assertEqual(runinfo.cache_delta(before, before),
                         {"compile_cache_new_files": 0, "model_downloaded": False})
        self.assertEqual(runinfo.cache_delta(before, compiled),
                         {"compile_cache_new_files": 1, "model_downloaded": False})
        self.assertTrue(runinfo.cache_delta(before, downloaded)["model_downloaded"])


class ScrubTests(unittest.TestCase):
    def test_paths_are_removed_and_urls_are_kept(self):
        cases = {r"FileNotFoundError: C:\Users\someone\models\x.onnx": "FileNotFoundError: <путь>",
                 r"не открыть 'C:\Users\someone\Мои записи\a b.onnx' (отказано)": "не открыть <путь> (отказано)",
                 r'load "D:/data/x y/model.bin" failed': "load <путь> failed",
                 r"сеть \\server\share\file недоступна": "сеть <путь>",
                 "open /home/user/models/x.bin failed": "open <путь> failed",
                 "архив https://example.com/org/repo/archive.zip#sha256=abc": None,
                 "dml: RuntimeError: DirectML недоступен (есть: ['CPUExecutionProvider'])": None,
                 "whisper-ct2:int8:t18": None}
        for text, expected in cases.items():
            self.assertEqual(runinfo.scrub(text), text if expected is None else expected)

    def test_structures_are_walked(self):
        data = {"a": [r"C:\Users\someone\x", {"b": "/home/user/y"}], "n": 5, "none": None}
        self.assertEqual(runinfo.scrub(data), {"a": ["<путь>", {"b": "<путь>"}], "n": 5, "none": None})


class RunFileTests(unittest.TestCase):
    def test_run_is_written_once_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            bench = Path(d) / "bench"
            first = runinfo.write_run(bench, {"run_id": "20260101-000000-aaaaaa", "n": 1}, {"onnx-cpu": "текст"})
            second = runinfo.write_run(bench, {"run_id": "20260101-000000-aaaaaa", "n": 2}, {"onnx-cpu": "иной"})
            self.assertEqual(first.parent, bench / "runs")
            self.assertNotEqual(first, second)
            one = json.loads(first.read_text(encoding="utf-8"))
            two = json.loads(second.read_text(encoding="utf-8"))
            texts = json.loads((first.parent / one["texts_file"]).read_text(encoding="utf-8"))
            self.assertEqual(sorted(p.name for p in first.parent.iterdir()),
                             sorted([first.name, one["texts_file"], second.name, two["texts_file"]]))
        self.assertEqual((one["n"], one["run_id"]), (1, "20260101-000000-aaaaaa"))
        self.assertEqual(two["n"], 2)
        self.assertEqual(two["run_id"], second.stem)
        self.assertEqual(texts["texts"], {"onnx-cpu": "текст"})
        self.assertEqual(texts["run_id"], one["run_id"])

    def test_run_without_texts(self):
        with tempfile.TemporaryDirectory() as d:
            path = runinfo.write_run(d, {"status": "error", "error_type": "RuntimeError"})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([p.name for p in path.parent.iterdir()], [path.name])
        self.assertIsNone(data["texts_file"])
        self.assertEqual(path.stem, data["run_id"])

    def test_run_file_has_no_paths_and_no_file_name(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "Планёрка 12 ноября.wav"
            src.write_bytes(b"synthetic")
            record = {"run_id": runinfo.new_run_id(), "audio": runinfo.audio_identity(src, PCM_1S, 16000),
                      "results": [{"id": "ov-gpu", "error": [rf"RuntimeError: не найден {src}"],
                                   "fallbacks": [rf"openvino:GPU: OSError: '{src}'"]}]}
            raw = runinfo.write_run(Path(d) / "bench", record, {"ov-gpu": ""}).read_bytes()
        text = raw.decode("utf-8")
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn("Планёрка", text)
        self.assertNotIn(Path(d).name, text)
        self.assertIn("<путь>", text)

    def test_units_index_keeps_the_path_locally(self):
        with tempfile.TemporaryDirectory() as d:
            bench = Path(d) / "bench"
            src = Path(d) / "запись.wav"
            src.write_bytes(b"synthetic")
            ident = runinfo.audio_identity(src, PCM_1S, 16000)
            other = runinfo.audio_identity(src, PCM_1S * 2, 16000)        # другой отрывок того же файла
            self.assertTrue(runinfo.update_units_index(bench, ident, src, "run-1"))
            self.assertTrue(runinfo.update_units_index(bench, other, src, "run-2"))
            data = json.loads((bench / "units.json").read_text(encoding="utf-8"))
        unit = data["units"][ident["audio_sha256"]]
        self.assertEqual((unit["path"], unit["name"], unit["runs"], unit["last_run_id"]),
                         (str(src), "запись.wav", 2, "run-2"))
        self.assertEqual(unit["unit_ids"], [ident["unit_id"], other["unit_id"]])
        self.assertLessEqual(unit["first_seen_utc"], unit["last_seen_utc"])

    def test_units_index_keeps_other_units_and_unreadable_file(self):
        with tempfile.TemporaryDirectory() as d:
            bench = Path(d) / "bench"
            a, b = Path(d) / "a.wav", Path(d) / "b.wav"
            a.write_bytes(b"one")
            b.write_bytes(b"two")
            ia, ib = runinfo.audio_identity(a, PCM_1S, 16000), runinfo.audio_identity(b, PCM_1S, 16000)
            runinfo.update_units_index(bench, ia, a)
            runinfo.update_units_index(bench, ib, b)
            data = json.loads((bench / "units.json").read_text(encoding="utf-8"))
            self.assertEqual(set(data["units"]), {ia["audio_sha256"], ib["audio_sha256"]})
            (bench / "units.json").write_text("{ испорчен", encoding="utf-8")
            self.assertFalse(runinfo.update_units_index(bench, ia, a))
            self.assertEqual((bench / "units.json").read_text(encoding="utf-8"), "{ испорчен")
            self.assertFalse(runinfo.update_units_index(bench, {"audio_sha256": None}, a))


if __name__ == "__main__":
    unittest.main()
