"""Тест скорости: устройства выбираются отдельно для RNN-T, CTC и Whisper.

Регрессии:
  11.11.2026 — тест скорости сохранял только настройки GigaAM, а `asr_third_device`
  оставался умолчанием «GPU». Теперь каждая группа измеряется и записывается сама.
  Чистая установка — процессор; NPU Meteor Lake (без FP32) не тестируется и не предлагается.

Запуск: python -m unittest discover -s tests
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# как во всех модулях тестов: временная папка данных задаётся ДО первого импорта программы
_TMP = tempfile.mkdtemp(prefix="rt-bench-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench  # noqa: E402
import paths  # noqa: E402
import settings  # noqa: E402

NPU_METEOR = {"available": True, "fp32": False, "name": "Intel(R) AI Boost",
              "capabilities": ["FP16", "INT8"]}
NPU_NEW = {"available": True, "fp32": True, "name": "Intel(R) AI Boost",
           "capabilities": ["FP32", "FP16", "INT8"]}


class DeviceOfTests(unittest.TestCase):
    def test_openvino_gpu(self):
        self.assertEqual(bench.device_of({"asr_backend": "openvino", "asr_device": "GPU"}), "GPU")
        self.assertEqual(bench.device_of({"asr_backend": "openvino", "asr_device": "gpu"}), "GPU")

    def test_openvino_cpu(self):
        self.assertEqual(bench.device_of({"asr_backend": "openvino", "asr_device": "CPU"}), "CPU")

    def test_openvino_without_device_is_gpu(self):
        self.assertEqual(bench.device_of({"asr_backend": "openvino"}), "GPU")

    def test_cpu_backends(self):
        self.assertEqual(bench.device_of({"asr_backend": "onnx-cpu"}), "CPU")
        self.assertEqual(bench.device_of({"asr_backend": "torch"}), "CPU")

    def test_directml_is_gpu(self):
        self.assertEqual(bench.device_of({"asr_backend": "dml"}), "GPU")

    def test_explicit_npu(self):
        self.assertEqual(bench.device_of({"device": "npu", "asr_backend": "openvino"}), "NPU")


class VariantPolicyTests(unittest.TestCase):
    def test_meteor_lake_npu_is_not_offered(self):
        ids = [v["id"] for v in bench.select_variants("", npu=NPU_METEOR)]
        self.assertNotIn("ov-npu-pad22", ids)
        self.assertNotIn("ctc-npu-pad22", ids)
        self.assertIn("onnx-cpu", ids)
        self.assertIn("ov-gpu", ids)

    def test_order_is_cpu_then_gpu_then_npu(self):
        chosen = bench.select_variants("", npu=NPU_NEW)
        devs = [v["device"] for v in chosen]
        last_cpu = max(i for i, d in enumerate(devs) if d == "CPU")
        first_gpu = min(i for i, d in enumerate(devs) if d == "GPU")
        last_gpu = max(i for i, d in enumerate(devs) if d == "GPU")
        first_npu = min(i for i, d in enumerate(devs) if d == "NPU")
        self.assertLess(last_cpu, first_gpu)
        self.assertLess(last_gpu, first_npu)

    def test_new_npu_is_testable(self):
        ids = [v["id"] for v in bench.select_variants("", npu=NPU_NEW)]
        self.assertIn("ov-npu-pad22", ids)
        self.assertIn("ctc-npu-pad22", ids)

    def test_whisper_is_measured_separately(self):
        ids = [v["id"] for v in bench.select_variants("", npu=NPU_METEOR)]
        self.assertIn("whisper-gpu", ids)
        self.assertIn("whisper-cpu", ids)
        self.assertLess(ids.index("whisper-cpu"), ids.index("whisper-gpu"))

    def test_no_whisper(self):
        ids = [v["id"] for v in bench.select_variants("", npu=NPU_METEOR, with_whisper=False)]
        self.assertNotIn("whisper-gpu", ids)

    def test_only_npu_on_meteor_lake_is_refused(self):
        self.assertEqual(bench.select_variants("ov-npu-pad22", npu=NPU_METEOR), [])


class SavedSettingsTests(unittest.TestCase):
    def test_cpu_main_gpu_ctc_cpu_whisper(self):
        main = {"family": "rnnt", "device": "CPU", "asr_backend": "onnx-cpu", "asr_batch": 16}
        second = {"family": "ctc", "device": "GPU", "asr_backend": "openvino",
                  "asr_device": "GPU", "asr_model": "v3_e2e_ctc", "asr_batch": 8}
        whisper = {"family": "whisper", "device": "CPU", "asr_third_device": "CPU"}
        out = bench.saved_settings(main, second, whisper)
        self.assertEqual(out["asr_backend"], "onnx-cpu")
        self.assertEqual(out["asr_device"], "CPU")
        self.assertEqual(out["asr_batch"], 16)
        self.assertEqual(out["asr_second_backend"], "openvino")
        self.assertEqual(out["asr_second_device"], "GPU")
        self.assertEqual(out["asr_second_batch"], 8)
        self.assertEqual(out["asr_third_device"], "CPU")
        self.assertFalse(out["asr_allow_npu"])

    def test_gpu_main_cpu_ctc(self):
        main = {"family": "rnnt", "device": "GPU", "asr_backend": "openvino",
                "asr_device": "GPU", "pad_to_seconds": 22}
        second = {"family": "ctc", "device": "CPU", "asr_backend": "onnx-cpu",
                  "asr_model": "v3_e2e_ctc"}
        out = bench.saved_settings(main, second, None)
        self.assertEqual(out["asr_device"], "GPU")
        self.assertEqual(out["pad_to_seconds"], 22)
        self.assertEqual(out["asr_second_device"], "CPU")
        self.assertNotIn("asr_third_device", out)

    def test_npu_choice_enables_npu(self):
        main = {"family": "rnnt", "device": "NPU", "asr_backend": "openvino", "asr_device": "NPU"}
        out = bench.saved_settings(main)
        self.assertEqual(out["asr_device"], "NPU")
        self.assertTrue(out["asr_allow_npu"])

    def test_npu_second_pass_enables_npu(self):
        main = {"family": "rnnt", "device": "CPU", "asr_backend": "onnx-cpu"}
        second = {"family": "ctc", "device": "NPU", "asr_backend": "openvino",
                  "asr_device": "NPU", "asr_model": "v3_e2e_ctc"}
        out = bench.saved_settings(main, second)
        self.assertTrue(out["asr_allow_npu"])

    def test_cpu_choice_is_written_explicitly(self):
        # после теста скорости выбор этой машины должен быть виден в config.json,
        # даже если он совпал с общим умолчанием «процессор»
        self.addCleanup(paths.CONFIG_FILE.unlink, missing_ok=True)
        settings.update(explicit=["asr_device"], asr_device="CPU")
        saved = json.loads(paths.CONFIG_FILE.read_text(encoding="utf-8"))
        self.assertEqual(saved["asr_device"], "CPU")


class BestOfTests(unittest.TestCase):
    def test_npu_with_wrong_text_is_not_chosen(self):
        results = [
            {"family": "rnnt", "id": "onnx-cpu", "rtfx": 28.0, "similarity": 1.0, "text": "верный текст"},
            {"family": "rnnt", "id": "ov-npu-pad22", "rtfx": 96.8, "similarity": 0.56, "text": "неверный"},
        ]
        best = bench.best_of(results, "rnnt")
        self.assertEqual(best["id"], "onnx-cpu")

    def test_ctc_is_chosen_separately(self):
        results = [
            {"family": "rnnt", "id": "onnx-cpu", "rtfx": 28.0, "similarity": 1.0, "text": "текст"},
            {"family": "ctc", "id": "ctc-gpu", "rtfx": 50.0, "similarity": 0.99, "text": "текст"},
        ]
        self.assertEqual(bench.best_of(results, "rnnt")["id"], "onnx-cpu")
        self.assertEqual(bench.best_of(results, "ctc")["id"], "ctc-gpu")

    def test_empty_text_is_not_chosen(self):
        results = [{"family": "rnnt", "id": "onnx-cpu", "rtfx": 28.0, "similarity": 1.0, "text": ""}]
        self.assertIsNone(bench.best_of(results, "rnnt"))


class MergeVariantTests(unittest.TestCase):
    def test_batch_and_pad_come_from_variant(self):
        # регрессия: в config.json попадал batch=1, хотя лучшим выбран вариант с пакетом 8
        variants = [{"id": "onnx-cpu-b8", "family": "rnnt", "device": "CPU",
                     "asr_backend": "onnx-cpu", "asr_batch": 8}]
        result = {"id": "onnx-cpu-b8", "family": "rnnt", "device": "CPU",
                  "rtfx": 26.0, "text": "текст"}
        merged = bench.merge_variant(result, variants)
        self.assertEqual(merged["asr_batch"], 8)
        self.assertEqual(merged["rtfx"], 26.0)
        self.assertEqual(merged["text"], "текст")

    def test_no_result(self):
        self.assertIsNone(bench.merge_variant(None, []))


class FamilyRefTests(unittest.TestCase):
    def test_ctc_is_compared_to_ctc_cpu(self):
        results = [
            {"id": "onnx-cpu", "family": "rnnt", "device": "CPU", "text": "раз два три"},
            {"id": "ctc-cpu", "family": "ctc", "device": "CPU", "text": "раз два"},
            {"id": "ctc-gpu", "family": "ctc", "device": "GPU", "text": "раз два"},
        ]
        refs = bench.family_refs(results)
        self.assertEqual(refs["ctc"]["id"], "ctc-cpu")
        self.assertEqual(refs["rnnt"]["id"], "onnx-cpu")

    def test_torch_is_the_rnnt_reference_when_available(self):
        results = [
            {"id": "torch-cpu", "family": "torch", "device": "CPU", "text": "эталон"},
            {"id": "onnx-cpu", "family": "rnnt", "device": "CPU", "text": "почти эталон"},
        ]
        self.assertEqual(bench.family_refs(results)["rnnt"]["id"], "torch-cpu")

    def test_whisper_is_compared_to_main_pass(self):
        # регрессия 15.11.2026: Whisper сверялся с другим Whisper, и зацикленный текст
        # (повторы раздувают его длину) получал 100% и выигрывал у исправного варианта
        results = [
            {"id": "onnx-cpu", "family": "rnnt", "device": "CPU", "text": "раз два три"},
            {"id": "whisper-cpu", "family": "whisper", "device": "CPU", "text": "раз два три три три"},
            {"id": "whisper-ct2", "family": "whisper", "device": "CPU", "text": "раз два три"},
        ]
        refs = bench.family_refs(results)
        self.assertEqual(refs["whisper"]["id"], "onnx-cpu")
        self.assertEqual(refs["ctc"], None)

    def test_whisper_uses_torch_reference_when_available(self):
        results = [
            {"id": "torch-cpu", "family": "torch", "device": "CPU", "text": "эталон"},
            {"id": "onnx-cpu", "family": "rnnt", "device": "CPU", "text": "почти эталон"},
            {"id": "whisper-ct2", "family": "whisper", "device": "CPU", "text": "виспер"},
        ]
        self.assertEqual(bench.family_refs(results)["whisper"]["id"], "torch-cpu")

    def test_whisper_only_run_keeps_old_reference(self):
        # без основного прохода независимого эталона нет: сравнивать не с чем, кроме Whisper
        results = [
            {"id": "whisper-cpu", "family": "whisper", "device": "CPU", "text": "раз два"},
            {"id": "whisper-gpu", "family": "whisper", "device": "GPU", "text": "раз два"},
        ]
        self.assertEqual(bench.family_refs(results)["whisper"]["id"], "whisper-cpu")


class WhisperChoiceTests(unittest.TestCase):
    def test_looping_text_loses_despite_speed(self):
        # повторы делают текст длиннее: совпадение с основным проходом падает,
        # поэтому быстрый, но зацикленный вариант не должен выбираться
        results = [
            {"id": "whisper-cpu", "family": "whisper", "device": "CPU", "rtfx": 2.6,
             "similarity": 0.45, "text": "раз два три три три три"},
            {"id": "whisper-ct2", "family": "whisper", "device": "CPU", "rtfx": 4.5,
             "similarity": 0.83, "text": "раз два три"},
        ]
        self.assertEqual(bench.most_accurate(results, "whisper")["id"], "whisper-ct2")

    def test_similarity_beats_speed_on_equal_accuracy(self):
        results = [
            {"id": "slow", "family": "whisper", "device": "CPU", "rtfx": 2.0,
             "similarity": 0.80, "text": "раз два"},
            {"id": "fast", "family": "whisper", "device": "CPU", "rtfx": 5.0,
             "similarity": 0.80, "text": "раз два"},
        ]
        self.assertEqual(bench.most_accurate(results, "whisper")["id"], "fast")

    def test_failed_variant_is_skipped(self):
        results = [
            {"id": "broken", "family": "whisper", "device": "CPU", "error": ["нет пакета"]},
            {"id": "ok", "family": "whisper", "device": "CPU", "rtfx": 3.0,
             "similarity": 0.70, "text": "раз два"},
        ]
        self.assertEqual(bench.most_accurate(results, "whisper")["id"], "ok")

    def test_without_candidates_none(self):
        self.assertIsNone(bench.most_accurate([], "whisper"))


class HumanChoiceTests(unittest.TestCase):
    def test_word_stats(self):
        s = bench.word_stats("раз два три четыре", "раз три четыре")
        self.assertEqual(s["missing"], 1)
        self.assertEqual(s["words"], 3)

    def test_word_stats_counts_extra(self):
        s = bench.word_stats("раз два", "раз два три")
        self.assertEqual(s["extra"], 1)
        self.assertEqual(s["missing"], 0)

    def test_threshold_from_config_and_cli(self):
        self.assertEqual(bench.threshold_from({}, None), 0.97)
        self.assertEqual(bench.threshold_from({"bench_min_similarity": 0.99}, None), 0.99)
        self.assertEqual(bench.threshold_from({"bench_min_similarity": 0.99}, 0.95), 0.95)
        self.assertEqual(bench.threshold_from({}, 1.5), 1.0)

    def test_choice_summary_priorities(self):
        results = [
            {"id": "cpu", "family": "rnnt", "device": "CPU", "rtfx": 10, "similarity": 1.0, "text": "a"},
            {"id": "gpu", "family": "rnnt", "device": "GPU", "rtfx": 50, "similarity": 0.90, "text": "a"},
            {"id": "gpu-ok", "family": "rnnt", "device": "GPU", "rtfx": 40, "similarity": 0.98, "text": "a"},
        ]
        ch = bench.choice_summary(results, "rnnt", 0.97)
        self.assertEqual(ch["accurate"]["id"], "cpu")
        self.assertEqual(ch["fast_ok"]["id"], "gpu-ok")
        self.assertEqual(ch["fastest"]["id"], "gpu")

    def test_threshold_changes_best_of(self):
        results = [
            {"id": "cpu", "family": "rnnt", "rtfx": 10, "similarity": 0.995, "text": "a"},
            {"id": "gpu", "family": "rnnt", "rtfx": 50, "similarity": 0.98, "text": "a"},
        ]
        self.assertEqual(bench.best_of(results, "rnnt", 0.97)["id"], "gpu")
        self.assertEqual(bench.best_of(results, "rnnt", 0.99)["id"], "cpu")

    def test_cli_threshold_is_saved(self):
        main = {"family": "rnnt", "device": "CPU", "asr_backend": "onnx-cpu", "asr_batch": 1}
        out = bench.saved_settings(main, min_similarity=0.99)
        self.assertEqual(out["bench_min_similarity"], 0.99)


class DefaultsTests(unittest.TestCase):
    def test_quality_threshold_has_default(self):
        value = settings.DEFAULTS["bench_min_similarity"]
        self.assertTrue(0.9 <= value <= 1.0, value)

    def test_clean_install_is_cpu_everywhere(self):
        self.assertEqual(settings.DEFAULTS["asr_device"], "CPU")
        self.assertEqual(settings.DEFAULTS["asr_second_device"], "")   # как у основного, то есть CPU
        self.assertEqual(settings.DEFAULTS["asr_second_batch"], 0)     # как у основного
        self.assertEqual(settings.DEFAULTS["asr_third_device"], "CPU")
        self.assertEqual(settings.DEFAULTS["diar_ov_device"], "CPU")
        self.assertFalse(settings.DEFAULTS["asr_allow_npu"])

    def test_second_pass_has_own_keys(self):
        for key in ("asr_second_model", "asr_second_backend", "asr_second_device", "asr_second_batch"):
            self.assertIn(key, settings.DEFAULTS, key)

    def test_bench_saves_separate_groups(self):
        for key in ("asr_device", "asr_second_device", "asr_third_device", "asr_allow_npu"):
            self.assertIn(key, bench.CONFIG_KEYS, key)


class ExperimentalVariantTests(unittest.TestCase):
    """18.11.2026: среды исполнения v3 на CPU и GigaAM Multilingual — только по --only;
    эксперименты (int8, Multilingual) измеряются, но не выбираются и не пишутся в config.json."""

    def test_new_variants_are_not_in_default_run(self):
        ids = [v["id"] for v in bench.select_variants("", npu=NPU_NEW, providers=[])]
        for new in ("ov-cpu", "ctc-ov-cpu", "onnx-cpu-int8", "ctc-cpu-int8", "ml-cpu", "ml-large-cpu"):
            self.assertNotIn(new, ids, new)

    def test_groups_expand(self):
        ids = [v["id"] for v in bench.select_variants("multilingual", npu=NPU_METEOR)]
        self.assertIn("ml-large-cpu", ids)
        self.assertIn("onnx-cpu", ids)      # эталон RNN-T для голосования
        self.assertIn("ctc-cpu", ids)       # эталон CTC для голосования
        ids = [v["id"] for v in bench.select_variants("cpu-runtime", npu=NPU_METEOR)]
        self.assertEqual(set(ids), {"onnx-cpu", "ov-cpu", "onnx-cpu-int8",
                                    "ctc-cpu", "ctc-ov-cpu", "ctc-cpu-int8"})

    def test_group_variants_exist(self):
        known = {v["id"] for v in bench.VARIANTS}
        for name, group in bench.GROUPS.items():
            for vid in group.split(","):
                self.assertIn(vid, known, f"{name}: {vid}")

    def test_multilingual_runs_after_whisper_on_cpu(self):
        ids = [v["id"] for v in bench.select_variants("ml-large-cpu,whisper-ct2,onnx-cpu", npu=NPU_METEOR)]
        self.assertEqual(ids, ["onnx-cpu", "whisper-ct2", "ml-large-cpu"])

    def test_multilingual_models_have_onnx_names(self):
        import backends
        self.assertEqual(backends.ONNX_NAMES["multilingual_ctc"], "gigaam-multilingual-ctc")
        self.assertEqual(backends.ONNX_NAMES["multilingual_large_ctc"], "gigaam-multilingual-large-ctc")

    def test_experiment_is_never_chosen(self):
        results = [
            {"id": "onnx-cpu", "family": "rnnt", "device": "CPU", "rtfx": 30.0,
             "similarity": 1.0, "text": "раз два"},
            {"id": "onnx-cpu-int8", "family": "rnnt", "device": "CPU", "rtfx": 60.0,
             "similarity": 1.0, "text": "раз два", "experimental": True},
        ]
        self.assertEqual(bench.best_of(results, "rnnt")["id"], "onnx-cpu")
        self.assertEqual(bench.most_accurate(results, "rnnt")["id"], "onnx-cpu")
        self.assertEqual(bench.choice_summary(results, "rnnt", 0.97)["fastest"]["id"], "onnx-cpu")

    def test_openvino_cpu_can_be_chosen_and_saved(self):
        v = next(x for x in bench.VARIANTS if x["id"] == "ov-cpu")
        r = {"id": "ov-cpu", "family": "rnnt", "device": "CPU", "rtfx": 40.0,
             "similarity": 1.0, "text": "раз два"}
        best = bench.merge_variant(bench.best_of([r], "rnnt"), [v])
        saved = bench.saved_settings(best)
        self.assertEqual(saved["asr_backend"], "openvino")
        self.assertEqual(saved["asr_device"], "CPU")
        self.assertFalse(saved["asr_allow_npu"])

    def test_multilingual_is_never_saved(self):
        results = [{"id": "ml-large-cpu", "family": "multilingual", "device": "CPU", "rtfx": 20.0,
                    "similarity": 0.9, "text": "раз два", "experimental": True}]
        self.assertIsNone(bench.best_of(results, "multilingual", 0.0))
        self.assertIsNone(bench.most_accurate(results, "multilingual"))
        self.assertEqual(bench.saved_settings(None, None, None), {})

    def test_multilingual_is_compared_to_main_pass(self):
        results = [
            {"id": "onnx-cpu", "family": "rnnt", "device": "CPU", "text": "раз два три"},
            {"id": "onnx-cpu-int8", "family": "rnnt", "device": "CPU", "text": "раз два",
             "experimental": True},
            {"id": "ml-large-cpu", "family": "multilingual", "device": "CPU", "text": "раз два три",
             "experimental": True},
        ]
        refs = bench.family_refs(results)
        self.assertEqual(refs["multilingual"]["id"], "onnx-cpu")
        self.assertEqual(refs["rnnt"]["id"], "onnx-cpu")

    def test_reference_is_not_an_experiment(self):
        # int8 идёт раньше fp32 в списке — эталоном всё равно должен стать fp32
        results = [
            {"id": "ctc-cpu-int8", "family": "ctc", "device": "CPU", "text": "раз", "experimental": True},
            {"id": "ctc-cpu", "family": "ctc", "device": "CPU", "text": "раз два"},
        ]
        self.assertEqual(bench.family_refs(results)["ctc"]["id"], "ctc-cpu")


class TiebreakTests(unittest.TestCase):
    def test_third_model_sides_with_rnnt_or_ctc(self):
        main = "мама мыла раму вчера вечером"
        ctc = "мама мила раму вчера утром"
        third = "мама мыла раму вчера утром"
        vt = bench.tiebreak_stats(main, ctc, third)
        self.assertEqual(vt["contested"], 2)    # «мыла» и «вечером»
        self.assertEqual(vt["for_rnnt"], 1)     # «мыла» подтверждено
        self.assertEqual(vt["for_ctc"], 1)      # «вечером» → «утром»
        self.assertEqual(vt["unresolved"], 0)

    def test_all_three_disagree(self):
        vt = bench.tiebreak_stats("раз два три", "раз пять три", "раз шесть три")
        self.assertEqual(vt, {"contested": 1, "for_rnnt": 0, "for_ctc": 0, "unresolved": 1,
                              "inserted": 0})

    def test_case_and_punctuation_do_not_matter(self):
        # e2e-модели пишут с заглавной и знаками, Multilingual — строчными без знаков
        vt = bench.tiebreak_stats("Раз, два.", "раз пять", "раз два")
        self.assertEqual((vt["contested"], vt["for_rnnt"]), (1, 1))

    def test_empty_texts(self):
        vt = bench.tiebreak_stats("", "", "")
        self.assertEqual(vt["contested"], 0)

    def test_punctuation_tokens_are_not_words(self):
        # «—» отдельным токеном текста — не слово, как в расшифровке (words.result_words, worker.run_words)
        vt = bench.tiebreak_stats("— мы пошли гулять", "мы — пошли гулять", "мы пошли — гулять —")
        self.assertEqual(vt, {"contested": 0, "for_rnnt": 0, "for_ctc": 0, "unresolved": 0, "inserted": 0})


class ActualDeviceTests(unittest.TestCase):
    """Паспорт прогона: на чём вариант посчитался на самом деле (21.11.2026).

    Раньше в результат попадало только запрошенное устройство: «whisper-dml» после тихого отката
    make_asr считался на процессоре, а записывался как GPU."""

    def test_backend_names(self):
        cases = {"openvino:GPU": "GPU", "openvino:GPU:b8": "GPU", "openvino:GPU.0:int8": "GPU",
                 "openvino:CPU": "CPU", "openvino:NPU": "NPU", "dml": "GPU", "dml:b4:int8": "GPU",
                 "cpu": "CPU", "cpu:b16": "CPU", "cpu:int8": "CPU", "torch:cpu": "CPU",
                 "whisper-ov:GPU:fp16+dec-int8": "GPU", "whisper-ct2:int8:t18": "CPU"}
        for name, device in cases.items():
            self.assertEqual(bench.actual_device({"backend_name": name}), device, name)

    def test_unknown_or_missing_name(self):
        for name in (None, "", "что-то новое"):
            self.assertIsNone(bench.actual_device({"backend_name": name}))

    def test_session_providers_override_the_name(self):
        cpu_only = ["CPUExecutionProvider"]
        self.assertEqual(bench.actual_device({"backend_name": "openvino:GPU", "providers_actual": cpu_only}), "CPU")
        self.assertEqual(bench.actual_device({"backend_name": "dml", "providers_actual": cpu_only}), "CPU")
        self.assertEqual(bench.actual_device({"backend_name": "dml", "providers_actual":
                                              ["DmlExecutionProvider", "CPUExecutionProvider"]}), "GPU")

    def test_silent_fallback_is_marked(self):
        results = bench.mark_actual([
            {"id": "whisper-dml", "device": "GPU", "asr_sec": 60.0, "backend_name": "cpu:b4:int8"},
            {"id": "ov-gpu", "device": "GPU", "asr_sec": 4.0, "backend_name": "openvino:GPU"},
            {"id": "onnx-cpu", "device": "CPU", "asr_sec": 10.0, "backend_name": None},
            {"id": "ov-npu-pad22", "device": "NPU", "error": ["не запустился"]}])
        self.assertEqual([r.get("device_actual") for r in results], ["CPU", "GPU", None, None])
        self.assertEqual([r.get("device_mismatch") for r in results], [True, False, False, None])


class UnfitVariantTests(unittest.TestCase):
    """Вариант, чьи числа не относятся к тому, что он обещает, не выбирается (21.11.2026).

    Раньше «whisper-dml», тихо посчитанный на процессоре, мог быть выбран и записан в config.json
    как вариант видеокарты, а быстрый вариант, потерявший часть речи, отсекался только порогом."""

    @staticmethod
    def res(vid, family="rnnt", device="CPU", rtfx=10.0, similarity=1.0, **extra):
        return {"id": vid, "family": family, "device": device, "rtfx": rtfx, "asr_sec": 1.0,
                "similarity": similarity, "text": "раз два три", "chunks": 5, "empty_chunks": 0, **extra}

    def test_lost_chunks_are_counted_against_the_same_model(self):
        results = bench.mark_lost_chunks([
            self.res("torch-cpu", family="torch", empty_chunks=1),
            self.res("onnx-cpu", empty_chunks=1),              # пауза: пусто у всех — не потеря
            self.res("ov-gpu", device="GPU", empty_chunks=3),
            self.res("ctc-cpu", family="ctc", empty_chunks=0),
            self.res("whisper-ct2", family="whisper", chunks=4, empty_chunks=2),   # одна в семействе
            self.res("whisper-cpu", family="whisper", chunks=3, empty_chunks=0),   # другая нарезка
            {"id": "ov-npu-pad22", "family": "rnnt", "device": "NPU", "error": ["не запустился"]},
            self.res("ctc-gpu", family="ctc", device="GPU", empty_chunks=None)])
        self.assertEqual([r.get("lost_chunks") for r in results], [0, 0, 2, 0, 0, 0, None, None])

    def test_reasons(self):
        self.assertIsNone(bench.unfit_reason(self.res("onnx-cpu")))
        self.assertIsNone(bench.unfit_reason({"id": "старый результат без новых полей"}))
        swapped = self.res("whisper-dml", device="GPU", device_actual="CPU", device_mismatch=True, lost_chunks=2)
        lost = self.res("ov-gpu", device="GPU", lost_chunks=2)
        self.assertEqual(bench.unfit_reason(swapped), "device_mismatch")
        self.assertEqual(bench.unfit_reason(lost), "lost_chunks")
        self.assertEqual(bench.unfit_note(swapped), "посчитан на CPU вместо GPU — не выбирается")
        self.assertIn("потеряно кусков: 2 из 5", bench.unfit_note(lost))
        self.assertEqual(bench.unfit_note(self.res("onnx-cpu")), "")

    def test_fast_but_unfit_variant_is_never_chosen(self):
        honest = self.res("onnx-cpu", rtfx=15.0)
        swapped = self.res("ov-gpu", device="GPU", rtfx=60.0, device_actual="CPU", device_mismatch=True)
        lost = self.res("ov-gpu-pad5", device="GPU", rtfx=50.0, lost_chunks=1)
        results = [honest, swapped, lost]
        self.assertIs(bench.best_of(results, "rnnt", 0.97), honest)
        self.assertIs(bench.most_accurate(results, "rnnt"), honest)
        summary = bench.choice_summary(results, "rnnt", 0.97)
        self.assertEqual((summary["accurate"], summary["fast_ok"], summary["fastest"]), (honest, honest, honest))

    def test_nothing_is_chosen_when_every_variant_is_unfit(self):
        only = [self.res("whisper-dml", family="whisper", device="GPU", device_actual="CPU", device_mismatch=True)]
        self.assertIsNone(bench.best_of(only, "whisper", 0.0))
        self.assertIsNone(bench.most_accurate(only, "whisper"))
        self.assertEqual(bench.choice_summary(only, "whisper", 0.0),
                         {"accurate": None, "fast_ok": None, "fastest": None})

    def lost(self, *results):
        bench.mark_lost_chunks(list(results))
        return [(r.get("lost_chunks"), r.get("lost_few_words_chunks")) for r in results]

    def test_few_words_left_is_lost_speech(self):
        # реальный случай 21.11.2026: CTC на Arc оставил по слову в каждом из 5 кусков, пустых нет
        cpu = self.res("ctc-cpu", family="ctc", rtfx=20.0, chunk_words=[30, 31, 29, 32, 32])
        gpu = self.res("ctc-gpu", family="ctc", device="GPU", rtfx=80.0, similarity=0.05,
                       chunk_words=[1, 1, 1, 1, 1])
        b8 = self.res("ctc-gpu-b8", family="ctc", device="GPU", rtfx=90.0, similarity=0.05,
                      chunk_words=[1, 1, 1, 1, 1])
        self.assertEqual(self.lost(cpu, gpu, b8), [(0, 0), (5, 5), (5, 5)])
        self.assertEqual((gpu["lost_empty_chunks"], gpu["lost_few_words_kept"], gpu["lost_few_words_ref"]),
                         (0, 5, 154))
        self.assertEqual(bench.unfit_reason(gpu), "lost_chunks")
        note = bench.unfit_note(gpu)
        self.assertIn("потеряно кусков: 5 из 5", note)
        self.assertIn("слов в разы меньше", note)
        self.assertIn("осталось 5 из 154", note)
        self.assertNotIn("текст пуст", note)
        self.assertIs(bench.best_of([cpu, gpu, b8], "ctc", 0.0), cpu)     # даже без порога
        self.assertIs(bench.family_refs([gpu, b8, cpu])["ctc"], cpu)
        line = bench.lost_line({**gpu, "label": "CTC, видеокарта"})
        self.assertIn("в 5 из 5 кусков дал в разы меньше слов (осталось 5 из 154)", line)
        self.assertNotIn("пустой текст", line)

    def test_honest_variants_are_not_lost(self):
        # числа цифрами, мелкие расхождения; PyTorch — та же модель, что RNN-T
        self.assertEqual(self.lost(
            self.res("torch-cpu", family="torch", chunk_words=[31, 29, 26, 31, 12]),
            self.res("onnx-cpu", chunk_words=[30, 28, 25, 31, 11]),
            self.res("ov-gpu", device="GPU", chunk_words=[24, 27, 18, 30, 9])),
            [(0, 0), (0, 0), (0, 0)])

    def test_torch_is_a_reference_for_rnnt(self):
        self.assertEqual(self.lost(
            self.res("torch-cpu", family="torch", chunk_words=[30, 30, 30, 30, 30]),
            self.res("ov-gpu", device="GPU", chunk_words=[30, 10, 30, 2, 30])),
            [(0, 0), (2, 2)])

    def test_pause_and_short_chunks_are_not_lost(self):
        # пауза/музыка: мало или ноль слов у всех; короткий кусок ниже минимума опоры — шум
        self.assertEqual(self.lost(
            self.res("onnx-cpu", empty_chunks=1, chunk_words=[0, 2, 9, 30, 30]),
            self.res("ov-gpu", device="GPU", empty_chunks=1, chunk_words=[0, 1, 1, 29, 30])),
            [(0, 0), (0, 0)])

    def test_empty_and_few_words_are_told_apart(self):
        cpu = self.res("onnx-cpu", chunk_words=[30, 30, 30, 30, 30])
        gpu = self.res("ov-gpu", device="GPU", empty_chunks=1, chunk_words=[0, 3, 30, 30, 30])
        self.assertEqual(self.lost(cpu, gpu), [(0, 0), (2, 1)])
        note = bench.unfit_note(gpu)
        self.assertIn("потеряно кусков: 2 из 5", note)
        self.assertIn("текст пуст там, где у другого варианта он есть: 1", note)
        self.assertIn("слов в разы меньше, чем у другого варианта той же модели: 1, осталось 3 из 30", note)
        line = bench.lost_line({**gpu, "label": "RNN-T, видеокарта"})
        self.assertIn("вернул пустой текст для 1 из 5 кусков и в 1 из 5 кусков дал в разы меньше слов", line)

    def test_old_results_keep_the_empty_chunk_rule(self):
        # результат без chunk_words (старый или вариант не сообщил) — только правило пустых кусков
        old = self.res("ov-gpu", device="GPU")
        bad = self.res("ov-gpu-b8", device="GPU", chunk_words=[1, 1])          # длина не та — не верим
        new = self.res("onnx-cpu", chunk_words=[30, 30, 30, 30, 30])
        self.assertEqual(self.lost(old, bad, new), [(0, None), (0, None), (0, None)])
        old_lost = self.res("ov-gpu", device="GPU", empty_chunks=2)
        self.assertEqual(self.lost(old_lost, self.res("onnx-cpu")), [(2, None), (0, None)])
        self.assertIn("(текст пуст там, где у другого варианта он есть) — не выбирается", bench.unfit_note(old_lost))
        self.assertIn("вернул пустой текст для 2 из 5 кусков",
                      bench.lost_line({**old_lost, "label": "RNN-T, видеокарта"}))
        # прежний результат без разбивки по видам потери
        self.assertIn("вернул пустой текст для 2 из 5",
                      bench.lost_line({"label": "x", "chunks": 5, "lost_chunks": 2}))

    def test_different_cuts_and_whisper_are_not_compared_by_words(self):
        self.assertEqual(self.lost(
            self.res("ctc-cpu", family="ctc", chunk_words=[30, 30, 30, 30, 30]),
            self.res("ctc-x", family="ctc", chunks=4, chunk_words=[1, 1, 1, 1]),     # другая нарезка
            self.res("whisper-ct2", family="whisper", chunks=3, chunk_words=[90, 90, 90]),  # повторы
            self.res("whisper-dml", family="whisper", chunks=3, chunk_words=[30, 30, 30])),
            [(0, None), (0, None), (0, None), (0, None)])

    def lost_empty(self, *results):
        bench.mark_lost_chunks(list(results))
        return [r.get("lost_empty_chunks") for r in results]

    def test_empty_in_different_chunks_is_lost_for_both(self):
        # по количеству разница 0 (по одному пустому у обоих); покусочно потеря у каждого
        a = self.res("onnx-cpu", empty_chunks=1, chunk_words=[0, 30, 30, 30, 30])
        b = self.res("ov-gpu", device="GPU", empty_chunks=1, chunk_words=[30, 0, 30, 30, 30])
        self.assertEqual(self.lost(a, b), [(1, 0), (1, 0)])
        self.assertEqual(bench.unfit_reason(a), "lost_chunks")
        self.assertIn("текст пуст там, где у другого варианта он есть", bench.unfit_note(b))

    def test_pause_empty_for_all_is_not_lost(self):
        self.assertEqual(self.lost_empty(
            self.res("torch-cpu", family="torch", empty_chunks=2, chunk_words=[0, 30, 0, 30, 30]),
            self.res("onnx-cpu", empty_chunks=2, chunk_words=[0, 30, 0, 30, 30]),
            self.res("ov-gpu", device="GPU", empty_chunks=2, chunk_words=[0, 29, 0, 31, 30])),
            [0, 0, 0])

    def test_few_words_of_a_neighbour_on_a_pause_is_not_lost(self):
        # у соседа на паузе слова-галлюцинации ниже порога — пустой вариант прав
        below = bench.LOST_EMPTY_MIN_REF - 1
        self.assertEqual(self.lost_empty(
            self.res("onnx-cpu", empty_chunks=2, chunk_words=[0, 0, 30, 30, 30]),
            self.res("ov-gpu", device="GPU", chunk_words=[1, below, 30, 30, 30])),
            [0, 0])
        # от порога — уже речь, пустой кусок потерян
        self.assertEqual(self.lost_empty(
            self.res("onnx-cpu", empty_chunks=1, chunk_words=[0, 30, 30, 30, 30]),
            self.res("ov-gpu", device="GPU", chunk_words=[bench.LOST_EMPTY_MIN_REF, 30, 30, 30, 30])),
            [1, 0])

    def test_whisper_uses_the_per_chunk_empty_rule(self):
        # пустой кусок у Whisper — потеря; правило «слов в разы меньше» к Whisper не применяется
        ct2 = self.res("whisper-ct2", family="whisper", chunks=3, empty_chunks=1, chunk_words=[0, 30, 90])
        dml = self.res("whisper-dml", family="whisper", chunks=3, empty_chunks=1, chunk_words=[30, 0, 10])
        self.assertEqual(self.lost(ct2, dml), [(1, None), (1, None)])
        self.assertEqual((ct2["lost_empty_chunks"], dml["lost_empty_chunks"]), (1, 1))

    def test_old_results_without_chunk_words_count_empties(self):
        # нет chunk_words ни у кого: прежний подсчёт по количеству, даже если пустые в разных кусках
        self.assertEqual(self.lost_empty(
            self.res("onnx-cpu", empty_chunks=1),
            self.res("ov-gpu", device="GPU", empty_chunks=1),
            self.res("ov-gpu-b8", device="GPU", empty_chunks=3)),
            [0, 0, 2])

    def test_mixed_group(self):
        # у двух вариантов chunk_words есть, у старого нет
        a = self.res("onnx-cpu", empty_chunks=1, chunk_words=[0, 30, 30, 30, 30])
        b = self.res("ov-gpu", device="GPU", empty_chunks=1, chunk_words=[30, 0, 30, 30, 30])
        old = self.res("ov-gpu-b8", device="GPU", empty_chunks=2)
        self.assertEqual(self.lost_empty(a, b, old), [1, 1, 1])     # old — по количеству: 2 − 1
        # оба новых пусты в одном куске (между собой — пауза), а старый пустых не имеет:
        # проверка количеством против старого соседа сохраняет потерю
        a = self.res("onnx-cpu", empty_chunks=1, chunk_words=[0, 30, 30, 30, 30])
        b = self.res("ov-gpu", device="GPU", empty_chunks=1, chunk_words=[0, 30, 30, 30, 30])
        old = self.res("ov-gpu-b8", device="GPU", empty_chunks=0)
        self.assertEqual(self.lost_empty(a, b, old), [1, 1, 0])
        # новый вариант один в группе со старыми — у него тоже подсчёт по количеству
        a = self.res("onnx-cpu", empty_chunks=2, chunk_words=[0, 0, 30, 30, 30])
        old = self.res("ov-gpu", device="GPU", empty_chunks=1)
        self.assertEqual(self.lost(a, old), [(1, None), (0, None)])

    def test_empty_and_few_words_are_not_counted_twice(self):
        cpu = self.res("ctc-cpu", family="ctc", chunk_words=[30, 30, 30, 30, 30])
        gpu = self.res("ctc-gpu", family="ctc", device="GPU", empty_chunks=2, chunk_words=[0, 0, 2, 2, 30])
        self.assertEqual(self.lost(cpu, gpu), [(0, 0), (4, 2)])
        self.assertEqual(gpu["lost_chunks"], gpu["lost_empty_chunks"] + gpu["lost_few_words_chunks"])
        self.assertEqual(gpu["lost_empty_chunks"], 2)

    def test_empty_count_follows_chunk_words(self):
        # одно определение пустого куска: при годных chunk_words пустые считаются по ним, даже если
        # empty_chunks записан по тексту (кусок «.» — текст не пуст, но слов 0)
        punct = self.res("ov-gpu", device="GPU", empty_chunks=0, chunk_words=[0, 30, 30, 30, 30])
        cpu = self.res("onnx-cpu", chunk_words=[30, 30, 30, 30, 30])
        self.assertEqual(bench.empty_count(punct), 1)
        self.assertEqual(self.lost(punct, cpu), [(1, 0), (0, 0)])
        # старый результат без chunk_words — по empty_chunks, как раньше
        self.assertEqual(bench.empty_count(self.res("ov-gpu", empty_chunks=2)), 2)

    def test_whisper_false_phrases_on_a_pause_do_not_make_empty_variant_lost(self):
        # у соседа на паузе «титры» и зацикленная фраза: после фильтра конвейера они ниже порога Whisper
        junk = ["Продолжение следует...", "Субтитры сделал DimaTorzok. Спасибо за просмотр!",
                "Спасибо за внимание. " * 12]
        speech = " ".join(f"слово{i}" for i in range(60))
        counts = [bench.chunk_word_count(t, "whisper") for t in junk]
        self.assertEqual(counts[:2], [0, 0])
        self.assertLess(counts[2], bench.LOST_EMPTY_MIN_REF_WHISPER)
        self.assertGreaterEqual(counts[2], bench.LOST_EMPTY_MIN_REF)    # прежний порог 5 здесь ошибся бы
        noisy = self.res("whisper-ct2", family="whisper", chunks=4,
                         chunk_words=counts + [bench.chunk_word_count(speech, "whisper")])
        quiet = self.res("whisper-dml", family="whisper", chunks=4, empty_chunks=3, chunk_words=[0, 0, 0, 60])
        self.assertEqual(self.lost(noisy, quiet), [(0, None), (0, None)])

    def test_whisper_loss_next_to_normal_speech_is_caught(self):
        # явный случай: у соседа обычная речь (30–50 слов в куске) — пустой кусок Whisper потерян
        speech = [bench.chunk_word_count(" ".join(f"слово{i}" for i in range(n)) + ".", "whisper")
                  for n in (30, 50, 40)]
        self.assertEqual(speech, [30, 50, 40])
        full = self.res("whisper-ct2", family="whisper", chunks=3, chunk_words=speech)
        lost = self.res("whisper-dml", family="whisper", chunks=3, empty_chunks=2, chunk_words=[0, 0, 40])
        self.assertEqual(self.lost(full, lost), [(0, None), (2, None)])
        self.assertEqual(bench.unfit_reason(lost), "lost_chunks")
        edge = self.res("whisper-dml", family="whisper", chunks=3, chunk_words=[0, 0, 40])
        at_threshold = self.res("whisper-ct2", family="whisper", chunks=3,
                                chunk_words=[bench.LOST_EMPTY_MIN_REF_WHISPER,
                                             bench.LOST_EMPTY_MIN_REF_WHISPER - 1, 40])
        self.assertEqual(self.lost(at_threshold, edge), [(0, None), (1, None)])

    def test_old_whisper_results_keep_the_count_rule(self):
        # старые результаты Whisper без chunk_words — по количеству пустых, как раньше
        self.assertEqual(self.lost_empty(
            self.res("whisper-ct2", family="whisper", chunks=3, empty_chunks=0),
            self.res("whisper-dml", family="whisper", chunks=3, empty_chunks=2)),
            [0, 2])

    def test_unfit_variant_is_not_a_reference_when_a_fit_one_exists(self):
        lost = self.res("ctc-cpu", family="ctc", lost_chunks=1)
        fit = self.res("ctc-gpu", family="ctc", device="GPU")
        self.assertIs(bench.family_refs([lost, fit])["ctc"], fit)
        self.assertIs(bench.family_refs([lost])["ctc"], lost)        # другого нет — остаётся прежнее правило


class ChunkWordCountTests(unittest.TestCase):
    """Слова куска в тесте скорости считаются так же, как их получает рабочий конвейер (words.py)."""

    def test_punctuation_only_chunk_is_empty(self):
        for text in ("", "   ", ".", ". , —", "… ?!", "– -", None):
            self.assertEqual(bench.chunk_word_count(text), 0, text)
            self.assertEqual(bench.chunk_word_count(text, "whisper"), 0, text)
        self.assertEqual(bench.chunk_word_count("Ну — да, 25 раз… какой-то"), 5)

    def test_matches_pipeline_words_on_examples(self):
        # GigaAM: слово конвейера — токен с буквой или цифрой (words.result_words, WORD_RE);
        # токен без них приклеивается к соседнему слову и словом не считается
        import types
        import words as W
        for text in ("Привет, мир — как дела?", "раз . два …", "— начало", "ok 5 «цитата» ."):
            toks = [(" " + t) for t in text.split()]
            res = types.SimpleNamespace(tokens=toks, timestamps=[0.1 * i for i in range(len(toks))],
                                        logprobs=[0.0] * len(toks))
            pipeline = sum(1 for w in W.result_words(res, 0.0) if W.norm_word(w["w"]))
            self.assertEqual(bench.chunk_word_count(text), pipeline, text)
        # Whisper: конвейер берёт clean_whisper(текст).split() (worker.run_words)
        text = "Да, конечно. Продолжение следует... Спасибо. Спасибо. Спасибо. Спасибо. Спасибо."
        cleaned = W.clean_whisper(text)[0].split()
        self.assertEqual(bench.chunk_word_count(text, "whisper"), sum(1 for t in cleaned if W.norm_word(t)))
        self.assertEqual(bench.chunk_word_count(text, "whisper"), 5)
        self.assertEqual(bench.chunk_word_count(text), 9)          # у GigaAM фильтра Whisper нет

    def test_third_model_cleaning_follows_settings_and_covers_multilingual(self):
        loop = "да " * 8 + "конец"
        self.assertEqual(bench.third_max_repeat({}), 3)
        self.assertEqual(bench.third_max_repeat({"asr_third_max_repeat": 1}), 1)
        self.assertEqual(bench.third_max_repeat({"asr_third_max_repeat": "x"}), 3)
        self.assertEqual(bench.chunk_word_count(loop, "whisper"), 4)
        self.assertEqual(bench.chunk_word_count(loop, "whisper", 1), 2)
        self.assertEqual(bench.chunk_word_count(loop, "whisper", 0), 9)       # 0 — без ограничения
        # Multilingual — кандидат в третью модель: тот же фильтр, что в расшифровке
        self.assertEqual(bench.chunk_word_count(loop, "multilingual", 2), 3)
        self.assertEqual(bench.chunk_word_count("Продолжение следует...", "multilingual"), 0)
        for family in ("rnnt", "ctc", "torch"):                 # основной и второй проход не чистятся
            self.assertEqual(bench.chunk_word_count(loop, family, 1), 9)

    def test_multilingual_variant_that_lost_speech_is_still_unfit(self):
        # чистка лишь сокращает повторы: потерявший речь Multilingual по-прежнему исключается,
        # а сосед, зациклившийся в куске, не делает «потерявшим» исправный вариант
        speech = " ".join(f"слово{i}" for i in range(30))
        loop = "и " * 40
        def r(id_, texts):
            words = [bench.chunk_word_count(t, "multilingual") for t in texts]
            return {"id": id_, "family": "multilingual", "device": "CPU", "asr_sec": 1.0, "chunks": len(texts),
                    "empty_chunks": words.count(0), "chunk_words": words}
        full = r("ml-cpu", [speech, speech, "слово " + speech[:40]])
        lost = r("ml-large-gpu", ["слово", "", speech[:40]])
        bench.mark_lost_chunks([full, lost])
        self.assertEqual((full["lost_chunks"], lost["lost_chunks"]), (0, 2))
        self.assertEqual(bench.unfit_reason(lost), "lost_chunks")
        looped = r("ml-cpu-int8", [speech, loop, speech])
        fine = r("ml-cpu", [speech, " ".join(f"слово{i}" for i in range(12)), speech])
        bench.mark_lost_chunks([looped, fine])
        self.assertEqual(fine["lost_chunks"], 0)

    def test_run_one_reports_one_definition_of_empty(self):
        import contextlib
        import io
        import numpy as np
        from unittest import mock
        import backends
        tmp = Path(tempfile.mkdtemp(prefix="rt-bench-one-", dir=_TMP))
        npy = tmp / "audio.npy"
        np.save(npy, np.zeros(16000 * 45, dtype=np.float32))      # 20 + 20 + 5 с

        class FakeOnnx:
            name = "cpu"

            def __init__(self, *a, **kw):
                pass

            def transcribe(self, piece):
                return "раз"

            def transcribe_many(self, pieces):
                return ["", ". , —", "раз два три"][:len(pieces)]

        class FakeWhisper:
            name = "whisper-ct2"

            def __init__(self, *a, **kw):
                pass

            def transcribe_texts(self, pieces):
                return ["Продолжение следует...", " ".join(f"слово{i}" for i in range(40))][:len(pieces)]

        out = io.StringIO()
        with mock.patch.object(backends, "OnnxASR", FakeOnnx), \
                mock.patch.object(backends, "WhisperCT2", FakeWhisper), contextlib.redirect_stdout(out):
            bench.run_one({"id": "onnx-cpu", "family": "rnnt", "asr_backend": "onnx-cpu"}, str(npy),
                          "v3_e2e_rnnt", 20.0)
            bench.run_one({"id": "whisper-ct2", "family": "whisper", "asr_backend": "whisper-ct2"}, str(npy),
                          "v3_e2e_rnnt", 20.0)
            # max_repeat родителя (asr_third_max_repeat) доходит до подсчёта слов кусков
            FakeWhisper.transcribe_texts = lambda self, pieces: ["да " * 6, "слово"][:len(pieces)]
            bench.run_one({"id": "whisper-ct2", "family": "whisper", "asr_backend": "whisper-ct2"}, str(npy),
                          "v3_e2e_rnnt", 20.0, 2)
        got = [json.loads(line[len("@@RESULT@@"):]) for line in out.getvalue().splitlines()
               if line.startswith("@@RESULT@@")]
        self.assertEqual([(r["chunks"], r["chunk_words"], r["empty_chunks"]) for r in got],
                         [(3, [0, 0, 3], 2), (2, [0, 40], 1), (2, [2, 1], 0)])


class ConditionsReportTests(unittest.TestCase):
    """Блок отчёта «Условия прогона»: человек видит, чем прогон отличается от образцового."""

    RECORD = {"method_version": "bench-1", "audio": {"unit_id": "0123456789ab"},
              "conditions": {"start": {"power_source": "ac", "battery_percent": 100, "cpu_load_percent": 4.0,
                                       "power": {"scheme": "balanced"}, "memory": {"free_mb": 9000},
                                       "other_heavy_jobs": 0}},
              "environment": {"program_version": "2000.1.2.3",
                              "code": {"git_revision": "a" * 40, "git_dirty": False, "files_sha256": "b" * 16},
                              "packages": {"key": {"onnxruntime-openvino": "1.24.1", "openvino": "2025.4.1"}}},
              "testbed": {"hardware": {"cpu": {"model": "Synthetic CPU 9000"}}},
              "preflight": {"ok": True, "reasons": []}, "results": [], "run_file": "runs\\20260101-000000-aaaaaa.json"}

    def test_good_run(self):
        text = "\n".join(bench.conditions_lines(self.RECORD))
        for part in ("## Условия прогона", "от сети", "balanced", "4%", "9000 МБ", "Synthetic CPU 9000",
                     "2000.1.2.3", "aaaaaaaaaa", "onnxruntime-openvino 1.24.1", "bench-1", "0123456789ab",
                     "runs\\20260101-000000-aaaaaa.json", "нет имени и пути записи"):
            self.assertIn(part, text)
        self.assertNotIn("с оговорками", text)
        self.assertNotIn("Подмена устройства", text)

    def test_marked_run_and_swapped_device(self):
        record = {**self.RECORD,
                  "conditions": {"start": {"power_source": "battery", "battery_percent": 32}},
                  "preflight": {"ok": False, "reasons": ["battery", "other_heavy_jobs", "новая причина"]},
                  "results": [{"label": "Whisper, DirectML", "device": "GPU", "device_actual": "CPU",
                               "device_mismatch": True}]}
        text = "\n".join(bench.conditions_lines(record))
        self.assertIn("от батареи, заряд 32%", text)
        self.assertIn("Прогон с оговорками", text)
        self.assertIn("питание от батареи", text)
        self.assertIn("другая тяжёлая задача", text)
        self.assertIn("новая причина", text)
        self.assertIn("«Whisper, DirectML» запрошен на GPU, посчитан на CPU", text)

    def test_nothing_measured(self):
        text = "\n".join(bench.conditions_lines({}))
        for part in ("не определено", "не измерена", "не измерено", "не определён", "пакеты не определены"):
            self.assertIn(part, text)
        self.assertNotIn("Файл прогона", text)


class RunFileFromMainTests(unittest.TestCase):
    """bench.main() целиком, без моделей: дочерние процессы, звук и сведения о железе подменены.

    Проверяется, что каждый запуск оставляет свой файл прогона, прежние не перезаписываются, имени
    и пути записи в файле нет, а неудачный запуск тоже записан."""

    def setUp(self):
        import contextlib
        import io
        import types
        from unittest import mock
        import numpy as np
        import backends
        import runinfo
        import worker
        self.dir = Path(tempfile.mkdtemp(prefix="rt-bench-main-", dir=_TMP))
        self.audio = self.dir / "Синтетическая запись совещания.wav"
        # содержимое своё у каждого теста: указатель units.json ведёт счёт прогонов по контрольной сумме
        self.audio.write_bytes(b"synthetic-container-" + os.urandom(8))
        self.samples = np.linspace(-0.5, 0.5, 16000 * 8, dtype=np.float32)

        def child(cmd, **kw):
            if "--one" not in cmd:
                return self.real_run(cmd, **kw)
            self.cmds.append(cmd)
            v = json.loads(cmd[cmd.index("--one") + 1])
            res = {"load_sec": 1.0, "first_chunk_sec": 0.2, "asr_sec": 2.0 if v["family"] == "rnnt" else 1.0,
                   "chunks": 3, "family": v["family"], "device": v["device"], "backend_name": "cpu",
                   "providers_actual": ["CPUExecutionProvider"], "compile_sec": None, "fallbacks": None,
                   "empty_chunks": 0, "peak_memory_mb": 512.0, "compile_cache_new_files": 0,
                   "model_downloaded": False, "text": "раз два три четыре"}
            res.update(self.overrides.get(v["id"], {}))
            return types.SimpleNamespace(stdout="@@RESULT@@" + json.dumps(res, ensure_ascii=False), stderr="")

        self.real_run = bench.subprocess.run
        self.overrides = {}            # id варианта -> что его «процесс» сообщит иначе
        self.cmds = []                 # команды дочерних процессов
        self.load_audio = mock.Mock(return_value=self.samples)
        patches = [mock.patch.object(bench.subprocess, "run", side_effect=child),
                   mock.patch.object(worker, "load_audio", self.load_audio),
                   mock.patch.object(worker, "wait_slot", lambda st, cfg, path: None),
                   mock.patch.object(backends, "available_providers", lambda: ["CPUExecutionProvider"]),
                   mock.patch.object(backends, "openvino_devices", lambda: ["CPU"]),
                   mock.patch.object(backends, "npu_info", lambda: {"available": False, "fp32": False, "name": "",
                                                                    "capabilities": [], "reason": "нет NPU"}),
                   mock.patch.object(runinfo, "hardware", lambda: {"status": "ok",
                                                                   "cpu": {"model": "Synthetic CPU 9000"}}),
                   mock.patch.object(runinfo, "cpu_load_percent", lambda interval=1.0: 5.0)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.io, self.redirect = io, contextlib.redirect_stdout

    def run_main(self, job, *extra, only="onnx-cpu,ctc-cpu"):
        from unittest import mock
        status = self.dir / f"{job}.json"
        argv = ["bench.py", str(self.audio), "--only", only, "--no-write", "--seconds", "5",
                "--status", str(status), *extra]
        out = self.io.StringIO()
        with mock.patch.object(sys, "argv", argv), self.redirect(out):
            bench.main()
        return json.loads(status.read_text(encoding="utf-8")), out.getvalue()

    def test_third_model_text_is_cleaned_with_the_configured_max_repeat(self):
        # как в расшифровке (worker.run_words): asr_third_max_repeat из настроек, и чистится текст
        # любой третьей модели — не только Whisper, но и кандидата Multilingual
        from unittest import mock
        import settings
        import worker
        self.overrides["ml-cpu"] = {"text": "да да да раз два три четыре"}
        cfg = {**settings.DEFAULTS, "asr_third_max_repeat": 1}
        with mock.patch.object(worker, "load_config", lambda: cfg),                 mock.patch.object(bench, "tiebreak_stats", wraps=bench.tiebreak_stats) as tb:
            st, _ = self.run_main("20260101-000200-b00003", only="onnx-cpu,ctc-cpu,ml-cpu")
        self.assertEqual(tb.call_args.args[2], "да раз два три четыре")
        for cmd in self.cmds:          # подсчёт слов кусков в дочернем процессе — с тем же числом
            self.assertEqual(cmd[cmd.index("--max-repeat") + 1], "1")
        rec = json.loads(Path(st["run_file"]).read_text(encoding="utf-8"))
        self.assertEqual(rec["params"]["third_max_repeat"], 1)

    def test_every_run_leaves_its_own_file(self):
        import runinfo
        st1, report = self.run_main("20260101-000000-b00001")
        st2, _ = self.run_main("20260101-000100-b00002")
        f1, f2 = Path(st1["run_file"]), Path(st2["run_file"])
        self.assertEqual(f1.parent, paths.BENCH / "runs")
        self.assertNotEqual(f1, f2)
        raw = f1.read_text(encoding="utf-8")
        rec = json.loads(raw)
        self.assertEqual((st1["state"], st1["run_id"], rec["run_id"]), ("done", f1.stem, f1.stem))
        self.assertEqual((rec["schema_version"], rec["method_version"], rec["kind"], rec["status"]),
                         (runinfo.SCHEMA_VERSION, runinfo.METHOD_VERSION, "bench", "done"))
        self.assertEqual(rec["job_id"], "20260101-000000-b00001")
        # имени и пути записи в файле прогона нет, только контрольные суммы
        self.assertNotIn("Синтетическая", raw)
        self.assertNotIn(self.dir.name, raw)
        self.assertEqual(rec["audio"]["excerpt"]["duration_sec"], 5.0)
        self.assertEqual(rec["audio"]["source_duration_sec"], 8.0)
        self.assertEqual(rec["audio"]["source_format"], "wav")
        self.assertEqual(rec["params"]["only"], "onnx-cpu,ctc-cpu")
        self.assertEqual((rec["params"]["seconds"], rec["params"]["repeats"], rec["config_written"]), (5, 1, False))
        self.assertEqual(rec["environment"]["program_version"], runinfo.program_version(bench.HERE))
        self.assertEqual(rec["environment"]["onnx_providers"], ["CPUExecutionProvider"])
        self.assertEqual(rec["testbed"]["hardware"]["cpu"]["model"], "Synthetic CPU 9000")
        self.assertEqual(len(rec["testbed"]["testbed_id"]), 16)
        self.assertEqual(rec["conditions"]["start"]["cpu_load_percent"], 5.0)
        self.assertIsNotNone(rec["conditions"]["end"])
        self.assertIn("reasons", rec["preflight"])
        self.assertEqual([v["id"] for v in rec["variants"]], ["onnx-cpu", "ctc-cpu"])
        by_id = {r["id"]: r for r in rec["results"]}
        self.assertEqual(by_id["onnx-cpu"]["rtfx"], 2.5)
        for r in by_id.values():
            self.assertNotIn("text", r)
            self.assertEqual((r["device_actual"], r["device_mismatch"], r["backend_name"]), ("CPU", False, "cpu"))
            self.assertIn("wall_sec", r)
            self.assertIn("started_utc", r)
        self.assertEqual(by_id["ctc-cpu"]["reference_variant"], "ctc-cpu")
        # правило эталонов записано для всех семейств: Whisper и Multilingual сверяются с основным проходом
        self.assertEqual(rec["reference_variants"], {"rnnt": "onnx-cpu", "ctc": "ctc-cpu",
                                                     "whisper": "onnx-cpu", "multilingual": "onnx-cpu"})
        self.assertEqual(rec["choice"]["main"], "onnx-cpu")
        texts = json.loads((f1.parent / rec["texts_file"]).read_text(encoding="utf-8"))
        self.assertEqual(texts["texts"], {"onnx-cpu": "раз два три четыре", "ctc-cpu": "раз два три четыре"})
        # отчёт: блок условий со ссылкой на файл прогона; прежние файлы «последнего прогона» остались
        self.assertIn("## Условия прогона", report)
        self.assertIn(f"runs\\{Path(st2['run_file']).name}", (paths.BENCH / "bench_report.md").read_text(encoding="utf-8"))
        self.assertTrue((paths.BENCH / "bench_texts.json").is_file())
        # путь знает только локальный указатель
        units = json.loads((paths.BENCH / "units.json").read_text(encoding="utf-8"))["units"]
        unit = units[rec["audio"]["audio_sha256"]]
        self.assertEqual((unit["path"], unit["runs"], unit["last_run_id"]), (str(self.audio), 2, f2.stem))
        self.assertEqual(json.loads(f1.read_text(encoding="utf-8")), rec)        # второй прогон первый не тронул

    def test_swapped_device_and_lost_speech_are_not_chosen(self):
        # ov-gpu «быстрее всех», но его процесс сообщил, что считал на процессоре; пакетный вариант
        # ещё быстрее, но вернул пустой текст для куска, который обычный вариант распознал
        self.overrides = {"ov-gpu": {"backend_name": "cpu", "asr_sec": 0.5},
                          "onnx-cpu-b8": {"asr_sec": 0.8, "empty_chunks": 1}}
        st, report = self.run_main("20260101-000300-b00004", only="onnx-cpu,onnx-cpu-b8,ov-gpu")
        rec = json.loads(Path(st["run_file"]).read_text(encoding="utf-8"))
        by_id = {r["id"]: r for r in rec["results"]}
        self.assertEqual((st["best"], rec["choice"]["main"]), ("onnx-cpu", "onnx-cpu"))
        self.assertEqual(rec["saved"]["asr_backend"], "onnx-cpu")           # в config.json ушёл бы честный вариант
        self.assertEqual((by_id["ov-gpu"]["unfit"], by_id["ov-gpu"]["device_actual"]), ("device_mismatch", "CPU"))
        self.assertEqual((by_id["onnx-cpu-b8"]["unfit"], by_id["onnx-cpu-b8"]["lost_chunks"]), ("lost_chunks", 1))
        self.assertEqual((by_id["onnx-cpu"]["unfit"], by_id["onnx-cpu"]["lost_chunks"]), (None, 0))
        self.assertIn("посчитан на CPU вместо GPU — не выбирается", report)
        self.assertIn("потеряно кусков: 1 из 3", report)
        self.assertIn("**Подмена устройства:**", report)
        self.assertIn("**Потеря речи:**", report)
        self.assertIn("Основной проход: **RNN-T, процессор**", report)

    def test_few_words_left_is_named_as_lost_speech(self):
        # пакетный вариант быстрее, пустых кусков нет, но в каждом куске осталось по слову:
        # раньше его отсекал только порог совпадения, и причина в отчёте была «ниже порога»
        full = " ".join(["слово"] * 36)
        self.overrides = {"onnx-cpu": {"text": full, "chunk_words": [12, 11, 13]},
                          "onnx-cpu-b8": {"asr_sec": 0.8, "text": "слово слово слово", "chunk_words": [1, 1, 1]}}
        st, report = self.run_main("20260101-000400-b00005", only="onnx-cpu,onnx-cpu-b8")
        rec = json.loads(Path(st["run_file"]).read_text(encoding="utf-8"))
        by_id = {r["id"]: r for r in rec["results"]}
        self.assertEqual((st["best"], rec["choice"]["main"], rec["saved"]["asr_backend"]),
                         ("onnx-cpu", "onnx-cpu", "onnx-cpu"))
        b8 = by_id["onnx-cpu-b8"]
        self.assertEqual((b8["unfit"], b8["lost_chunks"], b8["lost_empty_chunks"], b8["lost_few_words_chunks"],
                          b8["lost_few_words_kept"], b8["lost_few_words_ref"], b8["chunk_words"]),
                         ("lost_chunks", 3, 0, 3, 3, 36, [1, 1, 1]))
        self.assertEqual((by_id["onnx-cpu"]["unfit"], by_id["onnx-cpu"]["lost_chunks"]), (None, 0))
        row = next(l for l in report.splitlines() if l.startswith("| ") and "слов в разы меньше" in l)
        self.assertIn("потеряно кусков: 3 из 3", row)
        self.assertIn("осталось 3 из 36", row)
        self.assertNotIn("ниже порога", row)
        self.assertIn("**Потеря речи:**", report)
        self.assertIn("в 3 из 3 кусков дал в разы меньше слов (осталось 3 из 36)", report)

    def test_failed_run_is_recorded_too(self):
        self.load_audio.side_effect = RuntimeError(f"не прочитан {self.audio}")
        before = set((paths.BENCH / "runs").glob("*.json")) if (paths.BENCH / "runs").is_dir() else set()
        with self.assertRaises(SystemExit):
            self.run_main("20260101-000200-b00003")
        new = [p for p in set((paths.BENCH / "runs").glob("*.json")) - before]
        self.assertEqual(len(new), 1)
        raw = new[0].read_text(encoding="utf-8")
        rec = json.loads(raw)
        self.assertEqual((rec["status"], rec["error_type"], rec["job_id"]),
                         ("error", "RuntimeError", "20260101-000200-b00003"))
        self.assertIsNone(rec["texts_file"])
        self.assertNotIn("Синтетическая", raw)
        self.assertIn("start", rec["conditions"])


if __name__ == "__main__":
    unittest.main()
