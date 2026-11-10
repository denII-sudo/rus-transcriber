"""Тесты ядра на выдуманных данных (Анна, Борис, Вера, Глеб): голосование трёх моделей, куски, Whisper-фильтры,
словарь и замены, род глаголов, разбор ответов человека, метки сомнений, проверка цитат, настройки.
Звука и моделей не нужно. Запуск: python -m unittest discover -s tests"""
import atexit
import shutil
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_OLD = os.environ.get("RUS_TRANSCRIBER_DATA")
_TMP = tempfile.mkdtemp(prefix="rt-core-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:                      # папка данных тестов — временная
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

import freeform  # noqa: E402
import gender  # noqa: E402
import paths  # noqa: E402
import quotes  # noqa: E402
import settings  # noqa: E402
import transcript  # noqa: E402
import vocab  # noqa: E402
import words as W  # noqa: E402

if _OLD is None:
    os.environ.pop("RUS_TRANSCRIBER_DATA", None)
else:
    os.environ["RUS_TRANSCRIBER_DATA"] = _OLD


def ws(text, t0=0.0, step=0.5, **kw):
    return [{"w": w, "s": round(t0 + i * step, 2), "e": round(t0 + i * step + 0.4, 2), "c": 1.0, **kw}
            for i, w in enumerate(text.split())]


class Voting(unittest.TestCase):
    def test_all_three_agree(self):
        main = ws("мы пошли гулять")
        st = W.vote3(main, ws("мы пошли гулять"), "мы пошли гулять".split())
        self.assertEqual(st["agree3"], 3)
        self.assertEqual([w["w"] for w in main], ["мы", "пошли", "гулять"])
        self.assertTrue(all(w["agree"] for w in main))

    def test_two_of_three_keep_main_word(self):
        main = ws("мы пошли гулять")
        W.vote3(main, ws("мы пошёл гулять"), "мы пошли гулять".split())     # CTC ошибся, Whisper согласен с main
        self.assertEqual([w["w"] for w in main], ["мы", "пошли", "гулять"])
        self.assertTrue(all(w["agree"] for w in main))

    def test_two_others_override_main(self):
        main = ws("мы пошли кулять")
        st = W.vote3(main, ws("мы пошли гулять"), "мы пошли гулять".split())
        self.assertEqual([w["w"] for w in main], ["мы", "пошли", "гулять"])
        self.assertEqual(st["replaced"], 1)
        self.assertEqual(main[2]["voted"], "ctc+whisper")
        self.assertEqual(main[2]["model"], "кулять")           # что говорил RNN-T — для журнала
        self.assertEqual(main[2]["s"], 1.0)                    # времена слова берутся у CTC

    def test_three_way_disagreement_is_flagged_not_decided(self):
        main = ws("мы видели берег")
        W.vote3(main, ws("мы видели бегер"), "мы видели берик".split())
        self.assertEqual(main[2]["w"], "берег")                 # слово main остаётся
        self.assertFalse(main[2]["agree"])
        self.assertEqual(main[2]["alt"], "бегер")
        self.assertEqual(main[2]["alt_whisper"], "берик")

    def test_yo_is_not_a_disagreement(self):
        main = ws("мы видели берег")
        W.vote3(main, ws("мы видели берёг"), "мы видели берёг".split())     # обе другие пишут «ё»
        self.assertEqual([w["w"] for w in main], ["мы", "видели", "берег"])
        self.assertNotIn("voted", main[2])
        self.assertTrue(all(w["agree"] for w in main))

    def test_insertion_when_both_others_hear_an_extra_word(self):
        main = ws("мы гулять")
        st = W.vote3(main, ws("мы пошли гулять"), "мы пошли гулять".split())
        self.assertEqual([w["w"] for w in main], ["мы", "пошли", "гулять"])
        self.assertEqual(st["inserted"], 1)

    def test_colloquial_variants_are_not_disagreements(self):
        main = ws("прям хорошо")
        W.vote3(main, ws("прямо хорошо"), "прямо хорошо".split())
        self.assertTrue(all(w["agree"] for w in main))

    def test_disagreement_times_only_where_models_differ(self):
        main, ctc = ws("мы пошли кулять домой"), ws("мы пошли гулять домой")
        times = W.disagreement_times(main, ctc)
        self.assertTrue(times and all(1.0 <= t <= 1.5 for t in times))
        self.assertEqual(W.disagreement_times(ws("а б в"), ws("а б в")), [])


class WhisperFilters(unittest.TestCase):
    def test_subtitle_junk_removed(self):
        text, n = W.clean_whisper("привет всем Продолжение следует... как дела")
        self.assertNotIn("следует", text)
        self.assertEqual(text, "привет всем как дела")
        self.assertEqual(n, 2)

    def test_loops_capped_at_max_repeat(self):
        text, removed = W.clean_whisper("да " * 10 + "конец", max_repeat=3)
        self.assertEqual(text.split().count("да"), 3)
        self.assertEqual(removed, 7)
        text2, removed2 = W.clean_whisper("да " * 10, max_repeat=0)
        self.assertEqual((len(text2.split()), removed2), (10, 0))

    def test_phrase_loop_capped(self):
        text, _ = W.clean_whisper("не знаю " * 8 + "ладно", max_repeat=2)
        self.assertEqual(text, "не знаю не знаю ладно")


class Chunks(unittest.TestCase):
    def test_chunks_cover_regions_and_respect_max(self):
        audio = np.zeros(60 * W.SR, dtype=np.float32)
        regions = [(1.0, 9.0), (10.0, 31.0), (32.0, 40.0), (50.0, 55.0)]
        for max_len, min_len in ((W.CHUNK_MAX, W.CHUNK_MIN), (28.0, 20.0)):
            ch = W.make_chunks(audio, regions, max_len=max_len, min_len=min_len)
            for c in ch:
                s, e = c["own"]
                self.assertLessEqual(e - s, max_len + 1e-6)
                self.assertLessEqual(c["a"], s)
                self.assertGreaterEqual(c["b"], e)
            for s, e in regions:                                 # вся речь попала в чьи-то «свои» границы
                cover = sum(max(0, min(e, ce) - max(s, cs)) for cs, ce in (c["own"] for c in ch))
                self.assertAlmostEqual(cover, e - s, places=1)

    def test_longer_limit_gives_fewer_chunks(self):
        audio = np.zeros(300 * W.SR, dtype=np.float32)
        regions = [(float(i * 7), float(i * 7 + 6)) for i in range(40)]
        self.assertLess(len(W.make_chunks(audio, regions, max_len=28.0, min_len=20.0)),
                        len(W.make_chunks(audio, regions)))


class Replies(unittest.TestCase):
    def test_same_speaker_merges_and_backchannel_becomes_insert(self):
        a = ws("давай пойдём", 0.0, spk="Анна")
        b = ws("угу", 1.1, spk="Борис", margin=0.5)
        c = ws("на улицу", 1.6, spk="Анна")
        r = W.build_replies(a + b + c)
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["spk"], "Анна")
        self.assertEqual(r[0]["inserts"][0]["speaker"], "Борис")
        self.assertEqual(W.reply_text(r[0]["words"]), "давай пойдём на улицу")

    def test_pause_splits_replies(self):
        r = W.build_replies(ws("привет", 0.0, spk="Анна") + ws("привет", 5.0, spk="Анна"))
        self.assertEqual(len(r), 2)

    def test_answer_to_question_is_not_backchannel(self):
        q = ws("правда?", 0.0, spk="Анна")
        a = ws("да", 0.6, spk="Борис")
        c = ws("тогда идём", 1.2, spk="Анна")
        r = W.build_replies(q + a + c)
        self.assertEqual([x["spk"] for x in r], ["Анна", "Борис", "Анна"])


class Doubts(unittest.TestCase):
    def test_flags(self):
        words = ws("раз два три")
        words[0]["agree"] = False
        words[1]["c"] = 0.2
        for w in words:
            w["spk"], w["margin"] = "Анна", 0.9
        W.doubt_flags(words, [])
        self.assertEqual(words[0].get("flags"), ["asr"])
        self.assertEqual(words[1].get("flags"), ["conf"])
        self.assertNotIn("flags", words[2])

    def test_missed_speech(self):
        turns = [{"s": 0.0, "e": 6.0, "label": "Спикер 1"}]
        got = W.missed_speech(ws("слово", 0.0), turns)
        self.assertTrue(got and got[0][0] >= 0.4 and got[0][1] == 6.0)


VOCAB = {"group": "example",
         "participants": [{"name": "Анна", "gender": "ж"}, {"name": "Борис", "gender": "м"}],
         "terms": ["Вереск", "Лисицын"],
         "replace": {"Борька": "Борис"},
         "replace_if": [{"from": "весна", "to": "Весна", "after": ["в", "на", "про"], "not_before": ["пришла"]},
                        {"from": "Верно", "to": "Верно!", "case": True, "after": ["клуб"]}]}


class VocabTests(unittest.TestCase):
    def setUp(self):
        self.fx = vocab.Fixer(VOCAB)

    def test_explicit_replace_whole_word_any_case(self):
        out, fixes = self.fx.apply("Борька пришёл, борька ушёл")
        self.assertEqual(out, "Борис пришёл, Борис ушёл")
        self.assertEqual(len(fixes), 2)
        self.assertEqual(self.fx.apply("Борькин дом")[0], "Борькин дом")      # часть слова не трогается

    def test_conditional_replace_only_under_condition(self):
        self.assertEqual(self.fx.apply("весна", prev="про", nxt="сегодня")[0], "Весна")
        self.assertEqual(self.fx.apply("весна", prev="пришла", nxt="рано")[0], "весна")   # не после предлога
        self.assertEqual(self.fx.apply("весна", prev="про", nxt="пришла")[0], "весна")    # not_before
        self.assertEqual(self.fx.apply("про весну и про весна")[0], "про весну и про Весна")

    def test_case_sensitive_condition(self):
        self.assertEqual(self.fx.apply("верно", prev="клуб")[0], "верно")
        self.assertEqual(self.fx.apply("Верно", prev="клуб")[0], "Верно!")

    def test_common_words_not_mangled_by_fuzzy_terms(self):
        for common in ("вереница", "версия", "лисица", "борис"):
            out, _ = self.fx.apply(common)
            self.assertEqual(out.lower(), common.lower() if common != "борис" else "борис")

    def test_case_forms_of_term_kept(self):
        self.assertEqual(self.fx.apply("Вереском")[0], "Вереском")

    def test_genders(self):
        self.assertEqual(vocab.genders(VOCAB), {"Анна": "f", "Борис": "m"})

    def test_find_precedence_explicit_then_near_audio_then_group(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "rec").mkdir()
            near = d / "rec" / vocab.FILE_NAME
            near.write_text(json.dumps({"group": "near"}), encoding="utf-8")
            expl = d / "explicit.json"
            expl.write_text(json.dumps({"group": "explicit"}), encoding="utf-8")
            grp = d / "group.json"
            grp.write_text(json.dumps({"group": "grp"}), encoding="utf-8")
            audio = d / "rec" / "a.mp3"
            self.assertEqual(vocab.find(audio, expl, grp)[1]["group"], "explicit")
            self.assertEqual(vocab.find(audio, None, grp)[1]["group"], "near")
            self.assertEqual(vocab.find(d / "other" / "a.mp3", None, grp)[1]["group"], "grp")
            self.assertEqual(vocab.find(d / "other" / "a.mp3", None, None), (None, {}))
            bad = d / "bad.json"
            bad.write_text("{oops", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                vocab.find(None, bad, None)


class Gender(unittest.TestCase):
    def test_evidence(self):
        self.assertEqual(gender.evidence("я вчера поняла, что опоздала")[0], "f")
        self.assertEqual(gender.evidence("я думал и был уверен")[0], "m")
        self.assertIsNone(gender.evidence("я поняла и я понял")[0])
        self.assertIsNone(gender.evidence("мы пошли домой")[0])

    def test_reported_speech_ignored(self):
        self.assertIsNone(gender.evidence("Борис сказал: «я бы не поехала»")[0])


class Answers(unittest.TestCase):
    names = ["Анна", "Борис", "Вера", "Глеб"]

    def test_classify(self):
        c = freeform.classify
        self.assertEqual(c("да", self.names), {"kind": "agree"})
        self.assertEqual(c("Борис", self.names), {"kind": "name", "speaker": "Борис"})
        self.assertEqual(c("не Анна, а Вера", self.names), {"kind": "name", "speaker": "Вера"})
        self.assertEqual(c("", self.names), {"kind": "skip"})
        self.assertEqual(c("это слово было «весна»", self.names)["kind"], "text")

    def test_split_replies_by_labels(self):
        r = freeform.split_replies("Борис: привет, как дела? Анна: хорошо, спасибо", self.names)
        self.assertEqual([x[0] for x in r], ["Борис", "Анна"])
        self.assertEqual(r[0][1], ["привет,", "как", "дела?"])

    def test_split_without_labels_is_one_reply(self):
        r = freeform.split_replies("просто текст без меток", self.names)
        self.assertEqual(r, [(None, ["просто", "текст", "без", "меток"])])


def _data():
    seg1 = {"speaker": "Анна", "start": 0.0, "text": "пошли гулять в парк", "words": ws("пошли гулять в парк", 0.0)}
    seg2 = {"speaker": "Борис", "start": 3.0, "text": "давай купим мороженое", "words": ws("давай купим мороженое", 3.0)}
    return {"file": "C:/rec/example.mp3", "duration": "00:00:05", "speakers": ["Анна", "Борис"],
            "segments": [seg1, seg2], "review": []}


class TranscriptMarks(unittest.TestCase):
    def _with_doubt(self, status="open"):
        d = _data()
        d["review"] = [{"id": "r1", "t1": 0.5, "t2": 1.0, "speaker": "Анна", "status": status}]
        return d

    def test_unresolved_words_are_marked(self):
        d = self._with_doubt()
        md = transcript.render_md(d, text_only=True)
        self.assertIn("⟦гулять в⟧", md)
        self.assertEqual(transcript.strip_marks(md).count("⟦"), 0)

    def test_resolved_places_are_not_marked(self):
        md = transcript.render_md(self._with_doubt("applied"), text_only=True)
        self.assertNotIn("⟦", md)
        self.assertEqual(len(transcript.unresolved(self._with_doubt("human"))), 1)

    def test_marks_go_to_the_segment_of_the_assigned_speaker(self):
        d = _data()
        d["segments"][1]["words"] = ws("давай купим мороженое", 0.4)       # наложение со словами Анны
        d["review"] = [{"id": "r1", "t1": 0.5, "t2": 0.9, "speaker": "Борис", "status": "open"}]
        marks = transcript.doubt_marks(d)
        self.assertEqual(list(marks), [1])


class Quotes(unittest.TestCase):
    def test_verbatim_close_and_invented(self):
        d = _data()
        text = ("Анна: «пошли гулять в парк». Борис: «давай купим мороженое». "
                "Борис: «поедем на море завтра утром».")
        res = quotes.check(text, d)
        self.assertEqual(res["summary"]["quotes"], 3)
        self.assertEqual(res["summary"]["verbatim"], 2)
        self.assertEqual(res["summary"]["not_found"], 1)

    def test_one_wrong_word_is_close_not_verbatim(self):
        d = _data()
        d["segments"][1]["words"] = ws("давай купим мороженое сегодня вечером", 3.0)
        d["segments"][1]["text"] = "давай купим мороженое сегодня вечером"
        res = quotes.check("Борис: «давай купим мороженое сегодня утром»", d)
        self.assertEqual((res["summary"]["verbatim"], res["summary"]["close"]), (0, 1))

    def test_wrong_speaker_detected(self):
        res = quotes.check("Борис: «пошли гулять в парк»", _data())
        self.assertEqual(res["summary"]["speaker_mismatch"], 1)

    def test_quote_in_doubt_place_is_reported(self):
        d = _data()
        d["review"] = [{"id": "r1", "t1": 0.5, "t2": 1.0, "speaker": "Анна", "status": "open"}]
        res = quotes.check("Анна: «пошли гулять в парк»", d)
        self.assertEqual(res["summary"]["in_doubt"], 1)


class Settings(unittest.TestCase):
    def setUp(self):
        import shutil
        shutil.rmtree(paths.GROUPS, ignore_errors=True)
        paths.GROUPS.mkdir(parents=True, exist_ok=True)
        self._cfg = paths.CONFIG_FILE.read_bytes() if paths.CONFIG_FILE.exists() else None
        paths.CONFIG_FILE.unlink(missing_ok=True)

    def tearDown(self):
        paths.CONFIG_FILE.unlink(missing_ok=True)
        if self._cfg is not None:
            paths.CONFIG_FILE.write_bytes(self._cfg)

    def test_overrides_only_differences_and_merge(self):
        self.assertEqual(settings.load_config(), settings.DEFAULTS)
        kept = settings.save_overrides({"asr_backend": "onnx-cpu", "asr_model": settings.DEFAULTS["asr_model"],
                                        "default_group": "example"})
        self.assertEqual(kept, {"asr_backend": "onnx-cpu", "default_group": "example"})
        saved = json.loads(paths.CONFIG_FILE.read_text(encoding="utf-8"))
        self.assertNotIn("asr_model", saved)
        self.assertEqual(saved["schema_version"], paths.DATA_SCHEMA)
        cfg = settings.load_config()
        self.assertEqual(cfg["asr_backend"], "onnx-cpu")
        self.assertEqual(cfg["asr_device"], settings.DEFAULTS["asr_device"])
        settings.update(asr_batch=16)
        self.assertEqual(settings.load_overrides()["asr_backend"], "onnx-cpu")      # прежние отличия сохранены

    def test_broken_config_falls_back_to_defaults(self):
        paths.CONFIG_FILE.write_text("{oops", encoding="utf-8")
        self.assertEqual(settings.load_config(), settings.DEFAULTS)

    def test_pick_group_rules(self):
        self.assertIsNone(paths.pick_group())                                     # групп нет — без словаря
        paths.voices_path("alpha").parent.mkdir(parents=True)
        paths.voices_path("alpha").write_text("{}", encoding="utf-8")
        self.assertEqual(paths.pick_group(), "alpha")                             # единственная
        paths.vocab_path("beta").parent.mkdir(parents=True)
        paths.vocab_path("beta").write_text("{}", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            paths.pick_group()                                                    # несколько — ошибка
        self.assertEqual(paths.pick_group(None, "beta"), "beta")                  # default_group
        self.assertEqual(paths.pick_group("alpha", "beta"), "alpha")              # явная сильнее

    def test_group_names_are_safe_folder_names(self):
        self.assertEqual(paths.safe_group("../x y"), "___x_y")
        self.assertEqual(paths.safe_group(""), "default")


if __name__ == "__main__":
    unittest.main()
