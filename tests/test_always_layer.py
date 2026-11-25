"""Слой «всегда» (план лаборатории, раздел 0.1; PR-0): поля quality.asr_run каждой новой записи, у всех.

Хэш звука, ссылка на файл гипотез, обезличенный стенд, снимок настроек распознавания, хэши файлов весов и
ключевые пакеты — без текстов и путей, без настройки. Только синтетика: модели подменены (как в
tests/test_run_words_golden.py), папки данных, кэш моделей и соль стенда — во временной папке.
"""
import atexit
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

if "paths" not in sys.modules:   # папка данных — временная ДО первого импорта программы (test_data_safety)
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-always-data-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hypotheses
import hypstore
import paths
import recordings
import runinfo
import storage
import worker
from tests import test_hypotheses_save as S
from tests import test_run_words_golden as G

CFG = {"asr_model": "v3_e2e_rnnt", "asr_backend": "onnx-cpu", "asr_device": "CPU", "asr_batch": 8,
       "asr_second_model": "v3_e2e_ctc", "asr_third_model": "whisper", "asr_third_mode": "all"}
VOCAB = {"replace": {"борька": "Борис"}}
FIELDS = {"layer", "sound", "hypotheses", "testbed_id", "config", "packages_all"}


def audio_of(seconds=3.0, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * worker.SR) + 13) * 0.3).astype(np.float32)


def quality_with_models():
    return {"asr_run": {"date_utc": "synthetic", "program": "synthetic", "packages": {},
                        "models": {"primary": {"id": "v3_e2e_rnnt", "backend": "onnx-cpu", "revision": None},
                                   "second": {"id": "v3_e2e_ctc", "backend": "onnx-cpu", "revision": None}}}}


class AlwaysLayerTests(S.Sandbox):
    def weights(self, name="v3_e2e_rnnt", fill=b"\x01"):
        d = self.root / "cache" / "onnx-models"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{name}.onnx"
        f.write_bytes(fill * (hypstore.MIN_WEIGHT_BYTES + 10))
        return f

    def test_fields_are_written_without_the_setting(self):
        audio = audio_of()
        quality = quality_with_models()
        run = worker._always_layer(audio, CFG, VOCAB, quality, None)
        self.assertIs(run, quality["asr_run"])
        self.assertTrue(FIELDS <= set(run), FIELDS - set(run))
        self.assertEqual(run["layer"], worker.ALWAYS_LAYER)
        self.assertEqual(run["hypotheses"]["state"], "off")
        self.assertFalse(self.hyp.exists())          # папка гипотез не создаётся, если настройка выключена

    def test_sound_hash_is_the_rule_of_the_bench_passport(self):
        audio = audio_of()
        run = worker._always_layer(audio, CFG, VOCAB, quality_with_models(), None)
        whole = hashlib.sha256(hypstore.pcm_bytes(audio)).hexdigest()
        self.assertEqual(run["sound"], {"pcm_sha256": whole, "pcm_format": "mono 16000 Hz s16le",
                                        "duration_sec": round(len(audio) / worker.SR, 2)})
        # тот же хэш, что паспорт теста скорости и файл гипотез (runinfo.audio_identity)
        ident = runinfo.audio_identity(Path("-"), hypstore.pcm_bytes(audio), worker.SR)
        self.assertEqual(ident["excerpt"]["pcm_sha256"], whole)
        # по блокам — тот же результат при любой длине блока и хвосте
        for block in (1, 1000, 4097, len(audio) + 5):
            with self.subTest(block=block):
                self.assertEqual(worker.pcm_sha256(audio, block=block), (whole, len(audio)))

    def test_link_names_the_file_of_hypotheses_written_after_the_result(self):
        """Как main: слой — до сохранения результата, файл гипотез — после; имя файла совпадает со ссылкой."""
        params = dict(G.SCENARIOS["three_models"])
        params["cfg_extra"] = {**params.get("cfg_extra", {}), "save_hypotheses": True}
        data, _calls = G.run_scenario(**params)
        hyp = data["_hyp"]
        run = worker._always_layer(hyp.audio, hyp.cfg, S.VOCAB, data["quality"], hyp)
        link = run["hypotheses"]
        self.assertEqual(link["state"], "expected")
        self.assertEqual(link["sound_id"], run["sound"]["pcm_sha256"][:12])
        self.assertRegex(link["sound_id"], hypstore.UNIT_RE)
        self.assertRegex(link["run_id"] + ".json", hypstore.RUN_RE)
        self.assertEqual(S.finish(data, {"recording_id": str(uuid.uuid4())}), {"hypotheses_saved": True})
        files = self.files()
        self.assertEqual([f"{p.parent.name}/{p.name}" for p in files],
                         [f"{link['sound_id']}/{link['run_id']}.json"])
        rec = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(rec["audio"]["pcm_sha256"], run["sound"]["pcm_sha256"])
        self.assertEqual(rec["ids"]["run_id"], link["run_id"])
        # стенд — тот же, что в файле гипотез и тесте скорости (одна соль bench\testbed.json)
        self.assertEqual(rec["hardware"]["testbed_id"], run["testbed_id"])
        self.assertEqual(run["testbed_id"], runinfo.testbed_id(paths.BENCH))
        self.assertRegex(run["testbed_id"], r"^[0-9a-f]{16}$")
        self.assertEqual(run["config"], rec["config"])

    def test_config_snapshot_is_that_of_hypotheses(self):
        run = worker._always_layer(audio_of(), CFG, VOCAB, quality_with_models(), None)
        self.assertEqual(run["config"], hypstore.config_snapshot(CFG, VOCAB))
        self.assertEqual(run["config"]["asr_model"], "v3_e2e_rnnt")
        self.assertNotIn("борька", json.dumps(run, ensure_ascii=False))     # словарь — только сумма и число

    def test_weights_are_hashed_once_and_cached_in_bench(self):
        f = self.weights()
        calls = []
        real = runinfo.sha256_file
        with mock.patch.object(runinfo, "sha256_file", side_effect=lambda p: calls.append(p) or real(p)):
            first = worker._always_layer(audio_of(), CFG, VOCAB, quality_with_models(), None)
            second = worker._always_layer(audio_of(), CFG, VOCAB, quality_with_models(), None)
        self.assertEqual(len(calls), 1)                                      # второй раз — из кэша
        art = first["models"]["primary"]["artifact"]
        self.assertEqual(art["files"], [{"file": "onnx-models/v3_e2e_rnnt.onnx", "size_bytes": f.stat().st_size,
                                         "sha256": real(f)}])
        self.assertEqual(second["models"]["primary"]["artifact"], art)
        self.assertTrue(hypotheses.is_missing(first["models"]["second"]["artifact"]))   # весов нет в кэше
        self.assertTrue((paths.BENCH / worker.MODEL_ARTIFACTS_FILE).is_file())
        self.assertFalse(self.hyp.exists())

    def test_key_packages_are_the_full_list(self):
        from importlib import metadata
        expected = {}
        for name in runinfo.KEY_PACKAGES:
            try:
                expected[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        self.assertIn("numpy", expected)                 # список шире прежних четырёх пакетов распознавания
        self.assertEqual(worker.asr_run_metadata()["packages"], expected)
        run = worker._always_layer(audio_of(), CFG, VOCAB, quality_with_models(), None)
        full = runinfo.packages()
        self.assertEqual(run["packages_all"]["count"], full["count"])
        self.assertEqual(run["packages_all"]["sha256"], full["sha256"])

    def test_no_paths_and_no_texts(self):
        self.weights()
        params = dict(G.SCENARIOS["three_models"])
        params["cfg_extra"] = {**params.get("cfg_extra", {}), "save_hypotheses": True}
        data, _calls = G.run_scenario(**params)
        hyp = data["_hyp"]
        run = worker._always_layer(hyp.audio, hyp.cfg, S.VOCAB, data["quality"], hyp)
        text = json.dumps({k: run[k] for k in (*FIELDS, "models")}, ensure_ascii=False)
        self.assertIsNone(hypstore._ABS_PATH_RE.search(text))
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("synthetic.wav", text)
        spoken = {w["w"] for seg in data["segments"] for w in seg.get("words", []) if len(w["w"]) > 3}
        self.assertTrue(spoken)
        for word in spoken:
            self.assertIsNone(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text), word)

    def test_failure_of_a_field_is_written_as_not_measured(self):
        quality = quality_with_models()
        with mock.patch.object(runinfo, "testbed_id", side_effect=OSError("диск <путь>")), \
                mock.patch.object(worker, "pcm_sha256", side_effect=MemoryError()), \
                mock.patch.object(hypstore, "config_snapshot", side_effect=ValueError("x")):
            run = worker._always_layer(audio_of(), CFG, VOCAB, quality, None)
        self.assertEqual(run["testbed_id"], {"status": "not_measured", "reason": "OSError"})
        self.assertEqual(run["sound"], {"status": "not_measured", "reason": "MemoryError"})
        self.assertEqual(run["config"], {"status": "not_measured", "reason": "ValueError"})
        self.assertEqual(run["layer"], worker.ALWAYS_LAYER)
        self.assertIn("packages_all", run)

    def test_link_without_sound_hash_keeps_run_id(self):
        params = dict(G.SCENARIOS["whisper_off"])
        params["cfg_extra"] = {**params.get("cfg_extra", {}), "save_hypotheses": True}
        data, _calls = G.run_scenario(**params)
        hyp = data["_hyp"]
        with mock.patch.object(worker, "pcm_sha256", side_effect=MemoryError()):
            run = worker._always_layer(hyp.audio, hyp.cfg, S.VOCAB, data["quality"], hyp)
        self.assertTrue(hypotheses.is_missing(run["hypotheses"]["sound_id"]))
        self.assertEqual(run["hypotheses"]["run_id"], hyp.run_id)
        S.finish(data)
        self.assertEqual([p.stem for p in self.files()], [hyp.run_id])

    def test_main_calls_the_layer_before_saving_and_guards_it(self):
        src = Path(worker.__file__).read_text(encoding="utf-8")
        main = src[src.index("def main("):src.index("def _save_worker_result(")]
        pop, layer, save = (main.index('hyp = data.pop("_hyp", None)'),
                            main.index("_always_layer_guarded(audio, cfg, vocab, quality, hyp)"),
                            main.index("saved = _save_worker_result("))
        self.assertLess(pop, layer)
        self.assertLess(layer, save)

    def test_failing_layer_does_not_break_transcription(self):
        """Сбой слоя целиком (не отдельного поля) расшифровку не роняет: в записи — вид ошибки без текста."""
        quality = {}
        with mock.patch.object(worker, "_always_layer", side_effect=RuntimeError("сбой <путь>")):
            self.assertIsNone(worker._always_layer_guarded(None, {}, None, quality, None))
        self.assertEqual(quality, {"asr_run": {"layer_error": "RuntimeError"}})


class StoredRecordTests(S.Sandbox):
    """Только хранение в базе: поля слоя доходят до ревизии 1 записи и читаются из неё.
    Песочница — та же, что у гипотез: кэш моделей, HF_HOME и соль стенда во временной папке (настоящие веса
    установленной программы не хэшируются)."""

    def setUp(self):
        super().setUp()
        self.source = self.root / "synthetic.wav"
        self.source.write_bytes(b"synthetic")
        self.store = storage.initialize(self.root / "state.sqlite3", initial_display_name="Тест")
        self.addCleanup(self.store.close)

    def test_layer_is_in_the_first_revision(self):
        quality = quality_with_models()
        worker._always_layer(audio_of(), CFG, VOCAB, quality, None)
        layout = recordings.document_layout(self.source, "")
        layout["root"].mkdir()
        root_id, doc_paths = recordings.prepare_new_layout(self.store, layout["root"], layout["paths"])
        ident = str(uuid.uuid4())
        data = {"file": str(self.source), "duration": "00:00:01", "speakers": ["Спикер 1"],
                "quality": quality, "review": [], "review_done": False,
                "segments": [{"start": 0, "end": 1, "speaker": "Спикер 1",
                              "words": [{"w": "слово", "s": 0, "e": 1}]}]}
        recordings.commit_worker_result(
            self.store, data, self.source, root_id, recordings.operation_context(self.store, "model"),
            recording_id=ident, operation_id=str(uuid.uuid4()),
            generated_at="2026-01-01T00:00:00Z", document_paths=doc_paths)
        stored = recordings.load_recording(self.store, ident)
        self.assertEqual(stored["source_revision"], 1)
        self.assertEqual(stored["quality"]["asr_run"], quality["asr_run"])
        self.assertTrue(FIELDS <= set(stored["quality"]["asr_run"]))


if __name__ == "__main__":
    unittest.main()
