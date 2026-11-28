"""Горячие места разговора (hotspots.py): слой признаков и оценка на синтетических сигналах.

Звук — тоны известной частоты и громкости, участники выдуманы (Анна, Борис). Модели не
загружаются, реальные записи не используются. Проверяется устройство расчёта, а не
осмысленность оценок на настоящей речи.
"""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import core
import hotspots
import recordings
import settings
import transcript
import words
import worker

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db_record as DB  # noqa: E402  (У-2а: запись в базе вместо файла без базы)

SR = 16000
VOICES = {"Анна": (120.0, 0.05), "Борис": (220.0, 0.2)}   # обычные высота (Гц) и амплитуда


def tone(freq, amp, seconds, sr=SR):
    t = np.arange(int(round(sr * seconds))) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def segment(speaker, start, end):
    """Реплика со словами каждые 0,5 с (длиной 0,4 с)."""
    entries, t = [], start
    while t + 0.4 <= end + 1e-9:
        entries.append({"w": "слово", "s": round(t, 2), "e": round(t + 0.4, 2)})
        t += 0.5
    return {"start": start, "end": end, "speaker": speaker, "label": speaker,
            "text": " ".join(w["w"] for w in entries), "words": entries}


def conversation(raised=(220.0, 230.0)):
    """Пять минут: реплики по 10 с по очереди; 60–80 с — смены каждую секунду;
    150–159 с — наложения; raised — реплика Анны громкостью и высотой как обычно у Бориса."""
    plan, t, k = [], 0.0, 0
    while t < 300.0:
        length = 1.0 if 60.0 <= t < 80.0 else 10.0
        plan.append((t, t + length, "Анна" if k % 2 == 0 else "Борис"))
        t, k = t + length, k + 1
    audio = np.zeros(int(300 * SR), dtype=np.float32)
    for start, end, speaker in plan:
        freq, amp = VOICES["Борис" if (start, end) == raised else speaker]
        audio[int(start * SR):int(end * SR)] = tone(freq, amp, end - start)
    overlaps = [(150.0, 154.0), (155.0, 159.0)]
    raw = [(s, e, "SPEAKER_00" if who == "Анна" else "SPEAKER_01") for s, e, who in plan]
    turns, source = hotspots.cluster_turns(raw)
    layer = hotspots.build_layer(audio, SR, turns, overlaps, turns_source=source)
    data = {"file": "synthetic.wav", "duration": "00:05:00", "speakers": ["Анна", "Борис"],
            "quality": {"pipeline": "words", "interaction": layer},
            "segments": [segment(who, s, e) for s, e, who in plan],
            "review": [], "review_done": False, "edits": []}
    return data, audio, plan


def covering(found, start, end):
    return [h for h in found if h["t1"] < end and h["t2"] > start]


class LayerTests(unittest.TestCase):
    def test_tone_of_known_pitch_and_level(self):
        audio = np.concatenate([tone(220.0, 0.1, 3.0), np.zeros(2 * SR, dtype=np.float32), tone(110.0, 0.5, 3.0)])
        layer = hotspots.build_layer(audio, SR, [], [])
        windows = layer["windows"]
        self.assertEqual(layer["schema_version"], hotspots.SCHEMA_VERSION)
        self.assertEqual(layer["duration"], 8.0)
        self.assertEqual(windows["step"], 1.0)
        self.assertEqual(len(windows["rms_db"]), 8)
        # синус амплитуды A: RMS = A/√2 -> 20·lg(0,1/√2) = −23,0 дБ; 20·lg(0,5/√2) = −9,0 дБ
        self.assertEqual(windows["rms_db"][:3], [-23.0, -23.0, -23.0])
        self.assertEqual(windows["rms_db"][3:5], [hotspots.RMS_FLOOR_DB] * 2)
        self.assertEqual(windows["rms_db"][5:], [-9.0, -9.0, -9.0])
        self.assertEqual(windows["f0_hz"], [220, 220, 220, None, None, 110, 110, 110])

    def test_pitch_range_and_unvoiced_noise(self):
        for freq in (80.0, 150.0, 330.0, 440.0):
            with self.subTest(freq=freq):
                got = hotspots.build_layer(tone(freq, 0.1, 2.0), SR, [], [])["windows"]["f0_hz"]
                self.assertTrue(all(abs(v - freq) <= 1 for v in got), got)
        harmonic = sum(tone(150.0 * k, 0.1 / k, 2.0) for k in range(1, 8))
        self.assertEqual(hotspots.build_layer(harmonic, SR, [], [])["windows"]["f0_hz"], [150, 150])
        noise = (0.05 * np.random.default_rng(0).standard_normal(3 * SR)).astype(np.float32)
        self.assertEqual(hotspots.build_layer(noise, SR, [], [])["windows"]["f0_hz"], [None] * 3)

    def test_turns_and_overlaps_keep_only_numbers(self):
        class Annotation:
            def itertracks(self, yield_label=True):
                rows = [(2.0, 5.004, "SPEAKER_07"), (0.0, 2.5, "SPEAKER_03"), (4.0, 9.0, "SPEAKER_03")]
                return iter((SimpleNamespace(start=s, end=e), None, label) for s, e, label in rows)

        turns, source = hotspots.cluster_turns([(0.0, 9.0, "ignored")], Annotation())
        self.assertEqual(source, "overlapping")
        self.assertEqual(turns, [(0.0, 2.5, 0), (2.0, 5.004, 1), (4.0, 9.0, 0)])
        self.assertEqual(hotspots.cluster_turns([(3.0, 4.0, "b"), (0.0, 3.0, "a")], None),
                         ([(0.0, 3.0, 0), (3.0, 4.0, 1)], "exclusive"))

        overlaps = words.overlap_regions(Annotation())
        self.assertEqual([(round(s, 2), round(e, 2)) for s, e in overlaps], [(2.0, 2.5), (4.0, 5.0)])
        layer = hotspots.build_layer(np.zeros(9 * SR, dtype=np.float32), SR, turns, overlaps, turns_source=source)
        self.assertEqual(layer["overlaps"], {"s": [2.0, 4.0], "e": [2.5, 5.0]})
        self.assertEqual(layer["turns"], {"source": "overlapping", "s": [0.0, 2.0, 4.0],
                                          "e": [2.5, 5.0, 9.0], "cluster": [0, 1, 0]})

        def leaves(value):
            if isinstance(value, dict):
                return [leaf for item in value.values() for leaf in leaves(item)]
            if isinstance(value, list):
                return [leaf for item in value for leaf in leaves(item)]
            return [value]

        strings = [leaf for leaf in leaves(layer) if not (leaf is None or type(leaf) in (int, float))]
        self.assertEqual(strings, ["overlapping"], "в слое только числа и служебная отметка источника реплик")
        json.dumps(layer, allow_nan=False)      # так хранит SQLite: без NaN и бесконечностей

    def test_rejects_bad_audio(self):
        with self.assertRaises(ValueError):
            hotspots.build_layer(np.zeros((2, SR), dtype=np.float32), SR, [], [])
        with self.assertRaises(ValueError):
            hotspots.build_layer(np.full(SR, np.nan, dtype=np.float32), SR, [], [])
        empty = hotspots.build_layer(np.zeros(0, dtype=np.float32), SR, [], [])
        self.assertEqual(empty["windows"], {"step": 1.0, "rms_db": [], "f0_hz": []})

    def test_hour_of_signal_fits_size_budget(self):
        # Час синтетического сигнала (8 кГц — размер слоя от частоты не зависит, а тест легче):
        # голос звучит во всех окнах (худший случай для высоты), 1500 реплик, 400 наложений.
        sr = 8000
        audio = np.tile(tone(180.0, 0.1, 10.0, sr), 360)
        rng = np.random.default_rng(1)
        starts = np.sort(rng.uniform(0, 3590, 1500))
        turns = [(float(s), float(s) + float(d), int(k)) for s, d, k in
                 zip(starts, rng.uniform(0.3, 8.0, 1500), rng.integers(0, 4, 1500))]
        over = np.sort(rng.uniform(0, 3590, 400))
        overlaps = [(float(s), float(s) + float(d)) for s, d in zip(over, rng.uniform(0.1, 3.0, 400))]
        layer = hotspots.build_layer(audio, sr, turns, overlaps, turns_source="overlapping")
        self.assertEqual(len(layer["windows"]["rms_db"]), 3600)
        self.assertEqual(len(layer["turns"]["s"]), 1500)
        self.assertNotIn(None, layer["windows"]["f0_hz"])
        size = len(json.dumps(layer, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        self.assertLessEqual(size, 100_000, f"слой на час: {size} байт")


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.audio, cls.plan = conversation()
        cls.found = hotspots.find_hotspots(cls.data)

    def test_result_shape_for_time_join(self):
        self.assertTrue(self.found)
        for h in self.found:
            self.assertEqual(set(h), {"t1", "t2", "kinds", "score"})
            self.assertLess(h["t1"], h["t2"])
            self.assertTrue(0 < h["score"] <= 1)
            self.assertTrue(h["kinds"] and set(h["kinds"]) <= set(hotspots.KINDS))
        self.assertEqual(self.found, sorted(self.found, key=lambda h: h["t1"]))
        self.assertEqual(len(self.found), 3, self.found)

    def test_frequent_speaker_changes(self):
        hit = covering(self.found, 60, 80)
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0]["kinds"], [hotspots.KIND_TURNS])
        self.assertTrue(54 <= hit[0]["t1"] <= 62 and 78 <= hit[0]["t2"] <= 86, hit)
        # обычная очерёдность реплик по 10 с горячим местом не считается
        self.assertEqual(covering(self.found, 0, 50), [])
        self.assertEqual(covering(self.found, 90, 140), [])

    def test_overlaps_from_given_intervals(self):
        hit = covering(self.found, 150, 159)
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0]["kinds"], [hotspots.KIND_OVERLAP])
        self.assertTrue(144 <= hit[0]["t1"] <= 151 and 158 <= hit[0]["t2"] <= 165, hit)

    def test_level_is_relative_to_the_same_speaker(self):
        # Анна говорит с громкостью и высотой, обычными для Бориса: для неё это отклонение,
        hit = covering(self.found, 220, 230)
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0]["kinds"], [hotspots.KIND_LOUD, hotspots.KIND_PITCH])
        self.assertEqual((hit[0]["t1"], hit[0]["t2"]), (220.0, 230.0))
        # а те же абсолютные значения в репликах самого Бориса горячими не считаются.
        self.assertEqual(covering(self.found, 230, 300), [])
        scores = hotspots.window_scores(self.data)
        self.assertEqual(float(scores["kinds"][hotspots.KIND_LOUD][235]), 0.0)
        self.assertEqual(float(scores["kinds"][hotspots.KIND_LOUD][225]), 1.0)

    def test_speaker_comes_from_current_words(self):
        # Человек исправил, кто говорил: реплика 220–230 с принадлежит Борису — отклонения больше нет.
        edited = copy.deepcopy(self.data)
        for seg in edited["segments"]:
            if seg["start"] == 220.0:
                seg["speaker"] = seg["label"] = "Борис"
        self.assertEqual(covering(hotspots.find_hotspots(edited), 215, 235), [])
        # слой при этом тот же самый
        self.assertEqual(edited["quality"]["interaction"], self.data["quality"]["interaction"])

    def test_backchannel_inserts_count_as_changes(self):
        data = copy.deepcopy(self.data)
        for seg in data["segments"]:
            if seg["start"] == 100.0:      # реплика Анны 100–110 с, Борис поддакивает каждую секунду
                seg["inserts"] = [{"t": 100.5 + k, "speaker": "Борис", "text": "угу"} for k in range(9)]
        hit = covering(hotspots.find_hotspots(data), 100, 110)
        self.assertEqual([h["kinds"] for h in hit], [[hotspots.KIND_TURNS]])

    def test_min_score_filters(self):
        self.assertEqual(hotspots.find_hotspots(self.data, min_score=1.01), [])
        self.assertGreaterEqual(len(hotspots.find_hotspots(self.data, min_score=0.2)), 3)

    def test_empty_words_still_report_overlaps(self):
        data = copy.deepcopy(self.data)
        data["segments"] = []
        self.assertEqual([h["kinds"] for h in hotspots.find_hotspots(data)], [[hotspots.KIND_OVERLAP]])


class NoDataTests(unittest.TestCase):
    def reason(self, quality):
        return hotspots.no_data_reason({"quality": quality, "segments": []})

    def test_reasons(self):
        self.assertIsNone(self.reason({"interaction": hotspots.build_layer(np.zeros(SR, np.float32), SR, [], [])}))
        self.assertEqual(self.reason({})[0], "absent")
        self.assertEqual(hotspots.no_data_reason({"segments": []})[0], "absent")
        self.assertEqual(self.reason({"interaction_skipped": "disabled"})[0], "disabled")
        code, text = self.reason({"interaction_error": "ValueError: synthetic"})
        self.assertEqual(code, "error")
        self.assertIn("synthetic", text)
        self.assertEqual(self.reason({"interaction": {"schema_version": hotspots.SCHEMA_VERSION + 1}})[0],
                         "newer_schema")
        self.assertEqual(self.reason({"interaction": {"schema_version": 1, "windows": {}}})[0], "damaged")
        self.assertEqual(self.reason({"interaction": []})[0], "damaged")
        broken = hotspots.build_layer(np.zeros(2 * SR, np.float32), SR, [], [])
        broken["windows"]["f0_hz"].pop()
        self.assertEqual(self.reason({"interaction": broken})[0], "damaged")

    def test_find_raises_no_data(self):
        with self.assertRaises(hotspots.NoData) as caught:
            hotspots.find_hotspots({"quality": {}, "segments": []})
        self.assertEqual(caught.exception.code, "absent")


class CoreLegacyFileTests(unittest.TestCase):
    """Показ по запросу: стенограмма читается через transcript.load. У-2а-1: прежде файл без базы, теперь
    запись в базе (имя класса оставлено ради малого диффа), ассерты те же."""

    def run_core(self, data, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            jp, _, _ = DB.new_record(self, data, directory, "demo")
            with mock.patch.object(core, "_done_json", return_value={"json": str(jp)}):
                return core.conversation_hotspots("demo", **kwargs)

    def test_report_and_limit(self):
        data, _, _ = conversation()
        report = self.run_core(data)
        self.assertTrue(report["available"])
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["min_score"], hotspots.HOT_SCORE)
        self.assertEqual(report["note"], hotspots.NOTE)
        self.assertEqual([h["time"] for h in report["hotspots"] if h["t1"] == 220.0], ["00:03:40–00:03:50"])
        limited = self.run_core(data, limit=2)
        self.assertEqual((limited["total"], len(limited["hotspots"])), (3, 2))
        self.assertEqual(limited["hotspots"], sorted(limited["hotspots"], key=lambda h: h["t1"]))
        self.assertEqual(self.run_core(data, limit=0)["hotspots"], report["hotspots"])

    def test_no_layer_gives_clear_reason(self):
        data, _, _ = conversation()
        for quality, code in (({"pipeline": "words"}, "absent"),
                              ({"interaction_skipped": "disabled"}, "disabled"),
                              ({"interaction_error": "RuntimeError: synthetic"}, "error")):
            with self.subTest(code=code):
                report = self.run_core({**data, "quality": quality})
                self.assertEqual((report["available"], report["reason_code"], report["hotspots"]),
                                 (False, code, []))
                self.assertTrue(report["reason"].startswith("Нет данных: "))

    def test_bad_arguments(self):
        data, _, _ = conversation()
        for kwargs in ({"limit": -1}, {"min_score": 2}, {"limit": "много"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(core.CoreError):
                self.run_core(data, **kwargs)


class Status:
    def set(self, **kwargs):
        pass


class FakeAsr:
    name = "fake-primary"
    model_name = "fake"

    def transcribe_timestamped(self, pieces, progress=None, diagnostics=None):
        diagnostics.extend({"index": i, "state": "ok", "attempts": []} for i in range(len(pieces)))
        return [SimpleNamespace(tokens=[" тест"], timestamps=[0.1], logprobs=[-0.1], text="тест")
                for _ in pieces]


class RunWordsLayerTests(unittest.TestCase):
    """worker.run_words с подменёнными моделями: слой, настройка, ошибка слоя, замеры времени."""

    OVERLAPS = [(1.0, 1.6)]

    def run_words(self, cfg_extra=None, emotions=False, full_ann=None):
        cfg = {"asr_model": "primary", "asr_second_model": "", "asr_third_mode": "off",
               "emotions": emotions, "review_missed": False, **(cfg_extra or {})}
        args = SimpleNamespace(whisper="", whisper_max_repeat=-1, no_emotions=not emotions)
        quality = {"speaker_fixes": [], "vocab_fixes": []}
        audio = tone(220.0, 0.1, 3.0)
        chunks = [{"a": 0, "b": 3, "own": (0, 3)}]
        raw = [(0.0, 1.5, "SPEAKER_01"), (1.5, 3.0, "SPEAKER_00")]
        fake_emo = SimpleNamespace(load_model=lambda *a, **k: SimpleNamespace(get_probs=lambda path: {"neutral": 1.0}))
        with mock.patch.object(words, "make_chunks", return_value=chunks), \
                mock.patch.object(words, "assign_voices", return_value=0), \
                mock.patch.object(words, "build_replies", return_value=[]), \
                mock.patch.object(words, "overlap_regions", return_value=list(self.OVERLAPS)), \
                mock.patch.object(words, "doubt_flags", return_value=0), \
                mock.patch.object(words, "review_items", return_value=[]), \
                mock.patch.dict(sys.modules, {"gigaam": fake_emo, "soundfile": SimpleNamespace(write=None)}):
            data = worker.run_words(audio, "synthetic", cfg, Status(), raw, {"SPEAKER_00": "L0", "SPEAKER_01": "L1"},
                                    [], full_ann, None, {}, {}, {"cents": {"L0": [1]}}, None, {}, None,
                                    quality, lambda cfg: (FakeAsr(), None), args)
        return data, quality

    def test_layer_is_saved_by_default(self):
        data, quality = self.run_words()
        layer = quality["interaction"]
        self.assertIs(data["quality"], quality)
        self.assertEqual(layer["schema_version"], hotspots.SCHEMA_VERSION)
        self.assertEqual(layer["overlaps"], {"s": [1.0], "e": [1.6]})
        self.assertEqual(layer["turns"], {"source": "exclusive", "s": [0.0, 1.5], "e": [1.5, 3.0], "cluster": [0, 1]})
        self.assertEqual(layer["windows"]["rms_db"], [-23.0] * 3)
        self.assertEqual(layer["windows"]["f0_hz"], [220] * 3)
        self.assertNotIn("interaction_error", quality)
        self.assertNotIn("interaction_skipped", quality)
        self.assertEqual(quality["overlaps_sec"], 0.6)
        self.assertIn("interaction_sec", data["_timing"])
        self.assertNotIn("emotions_sec", data["_timing"], "эмоции выключены — времени этапа нет")
        self.assertTrue(settings.DEFAULTS["interaction"])
        self.assertIs(settings.CONFIG_TYPES["interaction"], bool)

    def test_setting_off_skips_layer(self):
        data, quality = self.run_words({"interaction": False})
        self.assertNotIn("interaction", quality)
        self.assertEqual(quality["interaction_skipped"], "disabled")
        self.assertEqual(hotspots.no_data_reason(data)[0], "disabled")

    def test_layer_failure_does_not_break_processing(self):
        with mock.patch.object(hotspots, "build_layer", side_effect=RuntimeError("synthetic layer failure")):
            data, quality = self.run_words()
        self.assertNotIn("interaction", quality)
        self.assertEqual(quality["interaction_error"], "RuntimeError: synthetic layer failure")
        self.assertEqual(quality["pipeline"], "words")
        self.assertEqual(data["segments"], [])
        self.assertIn("interaction_sec", data["_timing"])
        self.assertEqual(hotspots.no_data_reason(data)[0], "error")

    def test_broken_annotation_does_not_break_processing(self):
        class Broken:
            def itertracks(self, yield_label=True):
                raise RuntimeError("synthetic annotation failure")

        _, quality = self.run_words(full_ann=Broken())
        self.assertIn("synthetic annotation failure", quality["interaction_error"])

    def test_emotions_stage_is_timed(self):
        data, quality = self.run_words(emotions=True)
        self.assertNotIn("emotions_error", quality)
        self.assertIsInstance(data["_timing"]["emotions_sec"], float)
        self.assertIsInstance(data["_timing"]["interaction_sec"], float)


class SqliteEndToEndTests(unittest.TestCase):
    """Запись через worker.main в SQLite: слой хранится в снимке, читается по запросу,
    а в открытые документы записи не попадает."""

    def test_layer_is_stored_shown_on_request_and_kept_out_of_documents(self):
        import tmpfiles
        import voices

        class FakeEmbedder:
            seconds = 0

            def __init__(self, _pipe):
                pass

        data, audio, _ = conversation()
        data.update(speaker_labels=["Анна", "Борис"], group="demo", _timing={"interaction_sec": 0.1})
        layer = copy.deepcopy(data["quality"]["interaction"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
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
            args = ["worker.py", str(src), "--status", str(status), "--no-voices"]
            with mock.patch.object(settings.paths, "STATE_DB", db), \
                    mock.patch.object(sys, "argv", args), \
                    mock.patch.dict(os.environ, {"HF_TOKEN": "synthetic-token"}), \
                    mock.patch.object(worker, "wait_slot"), \
                    mock.patch.object(tmpfiles, "sweep_stale"), \
                    mock.patch.object(worker, "load_config", return_value={"pipeline": "words"}), \
                    mock.patch.object(worker, "resolve_vocab_details",
                                      return_value={"path": None, "vocab": {}, "group": None, "report": {}}), \
                    mock.patch.object(worker, "load_audio", return_value=audio), \
                    mock.patch.object(worker, "diarize",
                                      return_value=([(0.0, 300.0, "s1")], {}, None, None)), \
                    mock.patch.object(voices, "Embedder", FakeEmbedder), \
                    mock.patch.object(worker, "run_words", return_value=data):
                worker.main()
            result = json.loads(status.read_text(encoding="utf-8"))
            self.assertEqual(result["state"], "done", result)
            self.assertEqual(result["timing"]["interaction_sec"], 0.1)
            self.assertIsNone(result["quality"]["interaction_error"])
            with mock.patch.object(settings.paths, "STATE_DB", db), \
                    mock.patch.object(core, "job_status", side_effect=AssertionError("job is unavailable")):
                reference = "recording:" + result["recording_id"]
                self.assertEqual(transcript.load(result["json"])["quality"]["interaction"], layer)
                report = core.conversation_hotspots(reference)
                self.assertTrue(report["available"])
                self.assertEqual(report["total"], 3)
                self.assertEqual([h["kinds"] for h in report["hotspots"]],
                                 [[hotspots.KIND_TURNS], [hotspots.KIND_OVERLAP],
                                  [hotspots.KIND_LOUD, hotspots.KIND_PITCH]])
                # обычное чтение стенограммы слой не показывает
                for fmt in ("md", "json"):
                    self.assertNotIn("rms_db", core.read_transcript(reference, fmt))
                    self.assertNotIn("rms_db", core.read_transcript(reference, fmt, extra=True))
                import server
                shown = json.loads(server.read_recording_result(result["recording_id"]))
                self.assertEqual(shown["quality"]["interaction"], {"omitted": "см. conversation_hotspots"})
                self.assertEqual(json.loads(server.conversation_hotspots(reference, limit=1))["total"], 3)
            # открытые документы записи: ни слоя, ни его полей
            documents_root = Path(result["json"]).parent
            published = [p for p in documents_root.rglob("*") if p.is_file()]
            self.assertGreaterEqual(len(published), 2)
            for path in published:
                text = path.read_text(encoding="utf-8", errors="replace")
                for marker in ('"interaction"', "rms_db", "f0_hz", "schema_version\": 1, \"duration"):
                    self.assertNotIn(marker, text, f"{path.name}: {marker}")


if __name__ == "__main__":
    unittest.main()
