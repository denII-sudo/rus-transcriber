"""Оценка качества: эталон, WER, подпись голосов, пороги (на выдуманных данных)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import eval_quality as E  # noqa: E402

REF = """Стенограмма
[00:01] Анна: Мы пошли гулять в парк [смех] сегодня.
[00:05] Борис: Давай купим мороженое.
[1:00:10] Анна: Конец записи.
"""


def hyp_json(path, segs):
    path.write_text(json.dumps({"segments": [{"speaker": sp, "words": [{"w": w, "s": t + i * 0.4} for i, w in enumerate(text.split())]}
                                             for sp, t, text in segs], "timing": {"total_sec": 10, "realtime_factor": 1.0}}),
                    encoding="utf-8")


class T(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.addCleanup(__import__('shutil').rmtree, self.d, True)
        (self.d / "ref.txt").write_text(REF, encoding="utf-8")

    def test_reference_parsing(self):
        ref = E.load_reference(self.d / "ref.txt")
        self.assertEqual([r[0] for r in ref][:6], ["мы", "пошли", "гулять", "в", "парк", "сегодня"])   # [смех] не слово
        self.assertEqual(ref[0][1:], ("Анна", 1))
        self.assertEqual(ref[-1][2], 3610)                                                              # ч:мм:сс

    def test_perfect_and_errors(self):
        ref = E.load_reference(self.d / "ref.txt")
        hyp_json(self.d / "h.json", [("Анна", 1, "Мы пошли гулять в парк сегодня"), ("Борис", 5, "Давай купим мороженое"),
                                      ("Анна", 3610, "Конец записи")])
        hyp, _ = E.load_hypothesis(self.d / "h.json")
        r = E.score(ref, hyp)
        self.assertEqual((r["wer"], r["voice"]), (0.0, 100.0))
        hyp_json(self.d / "h2.json", [("Анна", 1, "Мы пошли гулять парк сегодня"), ("Анна", 5, "Давай купим мороженое вкусное"),
                                       ("Анна", 3610, "Конец записи")])
        r = E.score(ref, E.load_hypothesis(self.d / "h2.json")[0])
        self.assertEqual((r["S"], r["D"], r["I"]), (0, 1, 1))              # пропущено «в», вставлено «вкусное»
        self.assertAlmostEqual(r["wer"], 100 * 2 / 11, places=1)
        self.assertAlmostEqual(r["voice"], 70.0, places=1)                # Борис записан как Анна: верны 7 из 10 совпавших слов

    def test_yo_and_punctuation_do_not_count(self):
        self.assertEqual(E.norm("Ёлка,"), E.norm("елка"))

    def test_limit_cuts_both_sides(self):
        ref = E.load_reference(self.d / "ref.txt")
        hyp_json(self.d / "h.json", [("Анна", 1, "Мы пошли гулять в парк сегодня")])
        r = E.score(ref, E.load_hypothesis(self.d / "h.json")[0], limit=20)
        self.assertEqual(r["ref_words"], 9)                       # слова после 20 с не считаются
        self.assertAlmostEqual(r["wer"], 100 * 3 / 9, places=1)    # три слова второй реплики пропущены

    def test_empty_reference_is_an_error(self):
        (self.d / "bad.txt").write_text("просто текст", encoding="utf-8")
        with self.assertRaises(ValueError):
            E.score(E.load_reference(self.d / "bad.txt"), [("а", "Анна", 0)])

    def test_verdict_thresholds(self):
        case = {"max_wer": 10, "min_voice": 90, "baseline": {"total_sec": 100}}
        self.assertEqual(E.verdict(case, {"wer": 9.9, "voice": 90.0, "total_sec": 100}), ([], []))
        p, _ = E.verdict(case, {"wer": 10.1, "voice": 89.9, "total_sec": 100})
        self.assertEqual(len(p), 2)
        _, w = E.verdict(case, {"wer": 1, "voice": 99, "total_sec": 131})
        self.assertEqual(len(w), 1)

    def test_score_cli_exit_codes(self):
        hyp_json(self.d / "h.json", [("Анна", 1, "Мы пошли гулять в парк сегодня"), ("Борис", 5, "Давай купим мороженое"),
                                      ("Анна", 3610, "Конец записи")])
        args = ["score", str(self.d / "h.json"), str(self.d / "ref.txt")]
        self.assertEqual(E.main(args + ["--max-wer", "5"]), 0)
        hyp_json(self.d / "h.json", [("Анна", 1, "Совсем другой текст")])
        self.assertEqual(E.main(args + ["--max-wer", "5"]), 1)


    def fake_runs(self, seen, fail=False):
        """Подделка двух запусков: создание базы прогона (-c) и worker.py (пишет статус и выгрузку)."""
        from unittest import mock

        def fake_run(cmd, **kw):
            seen.setdefault("env", []).append(kw["env"])
            if "-c" in cmd:
                return mock.Mock(returncode=0, stdout=json.dumps({"actor_kind": "model"}) + "\n", stderr="")
            seen.setdefault("cmd", []).append(cmd)
            status = Path(cmd[cmd.index("--status") + 1])
            seen["context"] = json.loads(status.read_text(encoding="utf-8"))["recording_context"]
            if fail:
                return mock.Mock(returncode=1, stdout="", stderr="boom")
            audio = Path(cmd[2])
            out = audio.parent / f"{audio.stem}.rus-transcriber" / f"{audio.stem}.transcript.json"
            out.parent.mkdir()
            out.write_text(json.dumps({"passport": {"storage_backend": "sqlite"}, "transcript": {"segments": [
                {"speaker": "Спикер 1", "words": [{"w": "Мы", "s": 1}]}]}}), encoding="utf-8")
            status.write_text(json.dumps({"state": "done", "json": str(out),
                                          "timing": {"total_sec": 7, "realtime_factor": 2.0}}), encoding="utf-8")
            return mock.Mock(returncode=0, stdout="", stderr="")
        return fake_run

    def test_transcribe_runs_in_own_temp_data_dir_without_voices(self):
        from unittest import mock
        real = self.d / "real-data"
        (real / "groups" / "example").mkdir(parents=True)
        (real / "groups" / "example" / "vocab.json").write_text("{}", encoding="utf-8")
        (real / "groups" / "example" / "voices.json").write_text("{}", encoding="utf-8")
        (real / "config.json").write_text(json.dumps({"recording_backend": "sqlite", "device": "cpu"}),
                                          encoding="utf-8")
        audio = self.d / "rec.mp3"
        audio.write_bytes(b"x")
        before = sorted(p.name for p in self.d.iterdir())
        seen, data_dirs = {}, []

        def spy(cmd, **kw):
            data = Path(kw["env"]["RUS_TRANSCRIBER_DATA"])
            data_dirs.append(data)
            self.assertNotEqual(data, real)
            self.assertTrue((data / "groups" / "example" / "vocab.json").exists())
            self.assertFalse((data / "groups" / "example" / "voices.json").exists())   # биометрия не копируется
            self.assertNotIn("recording_backend", json.loads((data / "config.json").read_text(encoding="utf-8")))
            return fake(cmd, **kw)
        fake = self.fake_runs(seen)
        case = {"audio": str(audio), "names": "Анна, Борис", "group": "example", "whisper": "doubt"}
        with mock.patch.dict(E.os.environ, {"RUS_TRANSCRIBER_DATA": str(real)}), \
                mock.patch.object(E.subprocess, "run", spy):
            export, timing, kept = E.transcribe(case)
            E.transcribe(case)
        self.assertEqual(timing, {"total_sec": 7, "realtime_factor": 2.0})
        self.assertIsNone(kept)
        self.assertEqual(E.hypothesis_words(export), [("мы", "Спикер 1", 1)])
        self.assertEqual(seen["context"], {"actor_kind": "model"})
        first, second = seen["cmd"]
        self.assertNotEqual(first[first.index("--tag") + 1], second[second.index("--tag") + 1])
        self.assertNotEqual(Path(first[2]), audio)                   # прогон идёт по копии аудио
        self.assertIn("--no-voices", first)
        self.assertIn("--names", first)
        self.assertEqual(first[first.index("--group") + 1], "example")
        self.assertEqual(first[first.index("--whisper") + 1], "doubt")
        self.assertIn("--no-emotions", first)
        self.assertFalse(any(d.parent.exists() for d in data_dirs))  # временные папки удалены
        self.assertEqual(sorted(p.name for p in self.d.iterdir()), before)   # рядом с настоящим аудио — ничего
        self.assertEqual(sorted(p.name for p in real.rglob("*")), ["config.json", "example", "groups",
                                                                   "vocab.json", "voices.json"])

    def test_transcribe_failure_removes_temp_dir_unless_kept(self):
        from unittest import mock
        audio = self.d / "rec.mp3"
        audio.write_bytes(b"x")
        case = {"audio": str(audio)}
        for keep in (False, True):
            seen = {}
            with mock.patch.dict(E.os.environ, {"RUS_TRANSCRIBER_DATA": str(self.d / "real")}), \
                    mock.patch.object(E.subprocess, "run", self.fake_runs(seen, fail=True)), \
                    mock.patch.object(E.sys, "stderr"):
                with self.assertRaises(RuntimeError):
                    E.transcribe(case, keep=keep)
            tmp = Path(seen["env"][0]["RUS_TRANSCRIBER_DATA"]).parent
            self.assertEqual(tmp.exists(), keep)
            if keep:
                __import__("shutil").rmtree(tmp)

    def test_evaluate_without_voices_does_not_check_voice_threshold(self):
        from unittest import mock
        audio = self.d / "rec.mp3"
        audio.write_bytes(b"x")
        case = {"audio": str(audio), "reference": str(self.d / "ref.txt"), "max_wer": 100, "min_voice": 90}
        with mock.patch.dict(E.os.environ, {"RUS_TRANSCRIBER_DATA": str(self.d / "real")}), \
                mock.patch.object(E.subprocess, "run", self.fake_runs({})):
            res = E.evaluate(case)
        self.assertIsNone(res["voice"])
        self.assertEqual(res["total_sec"], 7)
        problems, warns = E.verdict(case, res)
        self.assertEqual(problems, [])
        self.assertEqual(len(warns), 1)

    def test_score_reads_store_export(self):
        export = {"passport": {"storage_backend": "sqlite"}, "transcript": {"segments": [
            {"speaker": "Анна", "words": [{"w": w, "s": 1} for w in "Мы пошли гулять в парк сегодня".split()]}]}}
        (self.d / "e.json").write_text(json.dumps(export), encoding="utf-8")
        hyp, _ = E.load_hypothesis(self.d / "e.json")
        self.assertEqual(hyp[0], ("мы", "Анна", 1))
        self.assertEqual(E.main(["score", str(self.d / "e.json"), str(self.d / "ref.txt"), "--limit", "3"]), 0)


if __name__ == "__main__":
    unittest.main()
