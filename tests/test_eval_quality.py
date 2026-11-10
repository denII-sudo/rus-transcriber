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


    def test_transcribe_builds_worker_command_and_finds_output(self):
        from unittest import mock
        audio = self.d / "rec.mp3"
        audio.write_bytes(b"x")
        out = self.d / "rec.rus-transcriber" / "rec.eval.transcript.json"
        out.parent.mkdir()
        out.write_text("{}", encoding="utf-8")
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return mock.Mock(returncode=0, stdout="", stderr="")
        case = {"audio": str(audio), "names": "Анна, Борис", "group": "example", "whisper": "doubt"}
        with mock.patch.object(E.subprocess, "run", fake_run):
            self.assertEqual(E.transcribe(case), out)
        cmd = seen["cmd"]
        self.assertIn("--names", cmd)
        self.assertEqual(cmd[cmd.index("--group") + 1], "example")
        self.assertEqual(cmd[cmd.index("--whisper") + 1], "doubt")
        self.assertIn("--no-emotions", cmd)
        with mock.patch.object(E.subprocess, "run", lambda *a, **k: mock.Mock(returncode=1, stdout="", stderr="boom")):
            with self.assertRaises(RuntimeError):
                E.transcribe(case)


if __name__ == "__main__":
    unittest.main()
