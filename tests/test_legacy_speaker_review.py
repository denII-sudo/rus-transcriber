"""Перенос старых подтверждений говорящего: только синтетические стенограммы во временной папке."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core
import paths
import review_pack
import transcript
import voices


def w(text, s, e, **extra):
    return {"w": text, "s": s, "e": e, "c": 0.9, **extra}


def legacy_data(audio):
    """Старый формат: `by: human` у слов без provenance, правки с before/after, сомнения r1…r6."""
    return {
        "file": str(audio), "duration": "00:00:15", "names_confirmed": True, "names_source": "rename", "speakers": ["Анна", "Борис"],
        "segments": [
            {"speaker": "Анна", "start": 0.0, "end": 3.0, "text": "принято как есть тут",
             "words": [w("принято", 0.0, 0.5, by="human"), w("как", 0.6, 0.8, by="human"),
                       w("есть", 0.9, 1.2, by="human"), w("тут", 1.3, 1.6)]},
            {"speaker": "Борис", "start": 4.0, "end": 6.0, "text": "сменили говорящего",
             "words": [w("сменили", 4.0, 4.6, by="human", edited="r2"), w("говорящего", 4.7, 5.4, by="human", edited="r2")]},
            {"speaker": "Анна", "start": 7.0, "end": 9.0, "text": "только текст",
             "words": [w("только", 7.0, 7.4, by="human", edited="r3"), w("текст", 7.5, 8.0, by="human", edited="r3")]},
            {"speaker": "Борис", "start": 10.0, "end": 12.0, "text": "уже отмечено",
             "words": [w("уже", 10.0, 10.4, by="human", provenance={"speaker": {"by": "human", "edit": "x"}}),
                       w("отмечено", 10.5, 11.0, by="human", provenance={"speaker": {"by": "human", "edit": "x"}})]},
            {"speaker": "Анна", "start": 13.0, "end": 13.5, "text": "раз", "words": [w("раз", 13.0, 13.4, by="human")]},
            {"speaker": "Борис", "start": 13.6, "end": 14.5, "text": "два", "words": [w("два", 13.7, 14.2, by="human")]},
            {"speaker": "Борис", "start": 15.0, "end": 15.4, "text": "да", "words": [w("да", 15.0, 15.4, by="human")]},
        ],
        "review": [
            {"id": "r1", "t": 0.0, "t1": 0.0, "t2": 1.2, "speaker": "Анна", "text": "принято как есть", "status": "resolved"},
            {"id": "r2", "t": 4.0, "t1": 4.0, "t2": 5.4, "speaker": "Анна", "text": "сменили говорящего", "status": "resolved"},
            {"id": "r3", "t": 7.0, "t1": 7.0, "t2": 8.0, "speaker": "Анна", "text": "толко текст", "status": "resolved"},
            {"id": "r4", "t": 10.0, "t1": 10.0, "t2": 11.0, "speaker": "Борис", "text": "уже отмечено", "status": "resolved"},
            {"id": "r5", "t": 13.0, "t1": 13.0, "t2": 14.2, "speaker": "Анна", "text": "раз два", "status": "resolved"},
            {"id": "r6", "t": 1.3, "t1": 1.3, "t2": 1.6, "speaker": "Анна", "text": "тут"},
            {"id": "r7", "t": 15.0, "t1": 15.0, "t2": 15.4, "speaker": "Борис", "text": "да", "status": "resolved"},
        ],
        "edits": [
            {"id": "r2", "t": 4.0, "by": "human", "before": {"text": "сменили говорящего", "speaker": "Анна"},
             "after": {"text": "сменили говорящего", "speaker": "Борис"}, "note": ""},
            {"id": "r3", "t": 7.0, "by": "human", "before": {"text": "толко текст", "speaker": "Анна"},
             "after": {"text": "только текст", "speaker": "Анна"}, "note": "", "kind": "words", "answer": "x"},
        ],
    }


class PlacesTests(unittest.TestCase):
    def test_places_by_evidence_and_skips(self):
        found = voices.legacy_speaker_places(legacy_data("a.wav"))
        places = {p["id"]: p for p in found["places"]}
        self.assertEqual(set(places), {"r1", "r2", "r3"})
        self.assertEqual(places["r1"]["evidence"], "место принято человеком как есть")
        self.assertEqual((places["r1"]["seg"], places["r1"]["w1"], places["r1"]["w2"]), (0, 0, 3))
        self.assertEqual(places["r2"]["evidence"], "смена говорящего правкой человека")
        self.assertEqual(places["r2"]["speaker"], "Борис")
        self.assertEqual(places["r3"]["evidence"], "правка текста человеком (говорящий не менялся)")
        # r7 короче HUMAN_MIN (0,8 с): в отпечаток не попадёт и не предлагается
        self.assertEqual(found["skipped"], {"too_short": 1, "already_confirmed": 1, "not_contiguous": 1, "no_words": 0,
                                            "merged": 0})
        self.assertEqual(found["seconds_by_speaker"], {"Анна": round(0.5 + 0.2 + 0.3 + 0.4 + 0.5, 2), "Борис": round(0.6 + 0.7, 2)})

    def test_nested_places_become_one_card(self):
        # r8 (слова 0–1) вложено в r1 (0–2) того же сегмента: одна карточка, а не две конфликтующие
        data = legacy_data("a.wav")
        data["review"].append({"id": "r8", "t": 0.0, "t1": 0.0, "t2": 0.8, "speaker": "Анна",
                               "text": "принято как", "status": "resolved"})
        found = voices.legacy_speaker_places(data)
        places = {p["id"]: p for p in found["places"]}
        self.assertEqual(set(places), {"r1", "r2", "r3"})
        self.assertEqual(places["r1"]["covers"], ["r8"])
        self.assertEqual((places["r1"]["w1"], places["r1"]["w2"]), (0, 3))
        self.assertEqual(found["skipped"]["merged"], 1)

    def test_partial_overlap_keeps_longer_place_without_widening(self):
        data = legacy_data("a.wav")
        data["segments"][0]["words"][3]["by"] = "human"           # «тут» тоже принят человеком
        data["review"].append({"id": "r8", "t": 0.6, "t1": 0.6, "t2": 1.6, "speaker": "Анна",
                               "text": "как есть тут", "status": "resolved"})
        data["review"][0]["t2"] = 0.8                              # r1: «принято как»
        found = voices.legacy_speaker_places(data)
        places = {p["id"]: p for p in found["places"]}
        self.assertNotIn("r1", places)
        self.assertEqual(places["r8"]["covers"], ["r1"])
        self.assertEqual((places["r8"]["w1"], places["r8"]["w2"]), (1, 4))

    def test_spans_report_counts_legacy_human_without_changing_selection(self):
        data = legacy_data("a.wav")
        report = {}
        spans = voices._spans(data, False, report)
        self.assertEqual(sum(1 for *_, human in spans if human), 1)          # только r4 с пословной отметкой
        self.assertEqual(report["legacy_human_without_speaker"], 10)
        self.assertIn("legacy_speaker_review",
                      voices.SELECTION_REASONS["legacy_human_without_speaker"])


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="legacy-speaker-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.audio = root / "встреча.wav"
        self.audio.write_bytes(b"synthetic audio")
        self.jp = root / "встреча.s2.transcript.json"
        self.md = root / "встреча.s2.transcript.md"
        # как настоящий старый файл: без word_id и recording_id, записан прежней версией программы
        self.jp.write_text(json.dumps(legacy_data(self.audio), ensure_ascii=False), encoding="utf-8")
        self.md.write_text("старый md\n", encoding="utf-8")
        for target, value in ((core, "_done_json"), (review_pack, "cut_clip"), (transcript, "_refresh_conspects")):
            replacement = {"_done_json": lambda job_id: {"json": str(self.jp), "markdown": str(self.md)},
                           "cut_clip": lambda *a, **k: 0.0, "_refresh_conspects": lambda *a, **k: None}[value]
            p = patch.object(target, value, side_effect=replacement)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(paths, "STATE_DB", root / "нет-базы.sqlite3")
        p.start()
        self.addCleanup(p.stop)

    def test_listing_does_not_change_files(self):
        before = self.jp.read_bytes()
        result = core.legacy_speaker_review("demo")
        self.assertEqual([p["id"] for p in result["places"]], ["r1", "r2", "r3"])
        self.assertIsNone(result["pack"])
        self.assertEqual(self.jp.read_bytes(), before)

    def test_pack_then_human_yes_sets_speaker_provenance_for_voice_learning(self):
        result = core.legacy_speaker_review("demo", build_pack=True)
        self.assertEqual(result["word_ids_added"], 13)          # все слова получили постоянные ID
        man = review_pack.manifest(self.jp)
        items = {i["id"]: i for i in man["items"]}
        self.assertEqual(set(items), {"r1", "r2", "r3"})
        yes = items["r1"]["options"][0]
        self.assertEqual(yes["label"], "Да, это говорит Анна")
        self.assertEqual(len(yes["decision"]["word_ids"]), 3)
        self.assertEqual(items["r2"]["options"][1]["decision"]["speaker"], "Анна")
        data = transcript.load(self.jp)
        self.assertTrue(all(x.get("word_id") for s in data["segments"] for x in s["words"]))
        decisions = [{"id": "r1", **yes["decision"]}, {"id": "r2", **items["r2"]["options"][0]["decision"]}]
        data, report = transcript.apply_review(self.jp, self.md, decisions, by="human")
        self.assertEqual(sorted(report["successful_ids"]), ["r1", "r2"])
        confirmed = [x for s in data["segments"] for x in s.get("words") or []
                     if (x.get("provenance") or {}).get("speaker", {}).get("by") == "human"]
        self.assertEqual(len(confirmed), 3 + 2 + 2)                            # r1, r2 и прежние r4
        self.assertEqual(data["segments"][1]["speaker"], "Борис")
        humans = [span for span in voices._spans(data, False) if span[3]]
        self.assertEqual({span[0] for span in humans}, {"Анна", "Борис"})
        again = core.legacy_speaker_review("demo")
        self.assertEqual([p["id"] for p in again["places"]], ["r3"])     # на r3 ещё не ответили

    def test_page_answers_applied_through_apply_answers(self):
        core.legacy_speaker_review("demo", build_pack=True)
        man = review_pack.manifest(self.jp)
        answers = [{"id": item["id"], "version": man["version"], "option": 0} for item in man["items"]]
        review_pack.add_answers(self.jp, answers, "local", trusted_actor="human")
        with patch.object(core, "job_json", return_value=(str(self.jp), str(self.md))):
            core.apply_answers("demo")
        data = transcript.load(self.jp)
        for rid in ("r1", "r2", "r3"):
            place_words = [x for s in data["segments"] for x in s["words"]
                           if (x.get("provenance") or {}).get("speaker", {}).get("edit") == rid]
            self.assertTrue(place_words, rid)
            self.assertTrue(all(x["provenance"]["speaker"]["by"] == "human" for x in place_words))
        self.assertEqual(core.legacy_speaker_review("demo")["places"], [])

    def apply_typed(self, rid, text):
        man = review_pack.manifest(self.jp)
        review_pack.add_answers(self.jp, [{"id": rid, "version": man["version"], "text": text}],
                                "local", trusted_actor="human")
        with patch.object(core, "job_json", return_value=(str(self.jp), str(self.md))):
            core.apply_answers("demo")
        return transcript.load(self.jp)

    def test_nested_places_answer_applies_without_conflict(self):
        data = json.loads(self.jp.read_text(encoding="utf-8"))
        data["review"].append({"id": "r8", "t": 0.0, "t1": 0.0, "t2": 0.8, "speaker": "Анна",
                               "text": "принято как", "status": "resolved"})
        self.jp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        core.legacy_speaker_review("demo", build_pack=True)
        man = review_pack.manifest(self.jp)
        self.assertEqual(sorted(i["id"] for i in man["items"]), ["r1", "r2", "r3"])
        answers = [{"id": item["id"], "version": man["version"], "option": 0} for item in man["items"]]
        review_pack.add_answers(self.jp, answers, "local", trusted_actor="human")
        with patch.object(core, "job_json", return_value=(str(self.jp), str(self.md))):
            core.apply_answers("demo")
        result = transcript.load(self.jp)
        first = result["segments"][0]["words"][:3]
        self.assertTrue(all(x["provenance"]["speaker"]["by"] == "human" for x in first))
        self.assertEqual(core.legacy_speaker_review("demo")["places"], [])

    def test_free_explanation_on_speaker_card_goes_to_human_not_into_text(self):
        result = core.legacy_speaker_review("demo", build_pack=True)
        self.assertTrue(result["pack"])
        items = {i["id"]: i for i in review_pack.manifest(self.jp)["items"]}
        self.assertEqual(items["r1"]["answer_mode"], "speaker")
        # сохранение может склеить соседние реплики одного говорящего — сравниваем слова по порядку
        before = [(s["speaker"], x["w"]) for s in transcript.load(self.jp)["segments"] for x in s["words"]]
        data = self.apply_typed("r1", "Анна говорит только слово «принято»")
        after = [(s["speaker"], x["w"]) for s in data["segments"] for x in s["words"]]
        self.assertEqual(after, before)
        r1 = next(r for r in data["review"] if r["id"] == "r1")
        self.assertEqual(r1["status"], "human")
        self.assertIn("подтверждения говорящего", r1.get("note") or "")

    def test_name_answer_on_speaker_card_changes_only_speaker(self):
        core.legacy_speaker_review("demo", build_pack=True)
        data = self.apply_typed("r1", "это Борис")
        words = [(s["speaker"], x["w"]) for s in data["segments"] for x in s["words"]]
        self.assertEqual([w for _, w in words][:4], ["принято", "как", "есть", "тут"])
        self.assertEqual([spk for spk, _ in words][:4], ["Борис", "Борис", "Борис", "Анна"])

    def test_explicit_label_answer_on_speaker_card_is_not_sent_to_human(self):
        core.legacy_speaker_review("demo", build_pack=True)
        data = self.apply_typed("r1", "Анна: принято как есть")
        words = [(s["speaker"], x["w"]) for s in data["segments"] for x in s["words"]][:4]
        self.assertEqual(words, [("Анна", "принято"), ("Анна", "как"), ("Анна", "есть"), ("Анна", "тут")])
        r1 = next(r for r in data["review"] if r["id"] == "r1")
        self.assertNotIn("подтверждения говорящего", r1.get("note") or "")

    def test_sqlite_record_is_refused(self):
        class Store:
            def close(self):
                pass
        with patch("recordings.open_for_document", return_value=(Store(), {})):
            with self.assertRaises(core.CoreError):
                core.legacy_speaker_review("demo")

    def test_rebuild_keeps_previous_manifest_and_its_clips(self):
        review_pack.build(self.jp, [{"id": "r6", "options": [{"label": "да", "decision": {}}]}])
        old = review_pack.manifest(self.jp)
        clip = review_pack.pack_dir(self.jp) / "clips" / "r6.mp3"
        clip.write_bytes(b"old clip")
        core.legacy_speaker_review("demo", build_pack=True)
        pd = review_pack.pack_dir(self.jp)
        archived = pd / f"manifest.{old['version']}.json"
        self.assertTrue(archived.is_file())
        self.assertEqual(json.loads(archived.read_text(encoding="utf-8"))["items"][0]["id"], "r6")
        self.assertTrue(clip.is_file())


class PreviewTests(unittest.TestCase):
    def test_preview_shows_human_seconds_change_and_warns_on_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = root / "groups"
            (groups / "demo-group").mkdir(parents=True)
            audio = root / "demo.wav"
            audio.write_bytes(b"x")
            data = legacy_data(audio)
            data["group"] = "demo-group"
            source = root / "demo.transcript.json"
            source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            key = str(audio.resolve())
            profile = {"version": voices.PROFILE_VERSION, "sources": {},
                       "speakers": {"Анна": {"parts": {key: {"vec": [0.1, 0.2], "sec": 9.0, "human_sec": 4.2}}}}}
            (groups / "demo-group" / "voices.json").write_text(json.dumps(profile), encoding="utf-8")
            with patch.object(core, "job_json", return_value=(str(source), None)), patch.object(paths, "GROUPS", groups):
                result = core.preview_voice_learning("demo")
            self.assertEqual(result["human_seconds_change"]["Анна"], {"before": 4.2, "after": 0.0})
            self.assertEqual(result["human_seconds_lost"], ["Анна"])
            lines = core.voice_action_lines({"group": "demo-group", "json": str(source), "report": result})
            self.assertTrue(any(line.startswith("ВНИМАНИЕ") and "Анна" in line for line in lines))
            self.assertTrue(any("4.2 → 0.0" in line for line in lines))

    def test_confirmation_text_fits_with_legacy_reason(self):
        # окно подтверждения обновления профиля ограничено 1100 знаками (core.prepare_voice_action);
        # отчёт как у настоящей записи после переноса старых подтверждений не должен его превышать
        counts = {"accepted_auto": 77, "accepted_human": 15, "foreign_overlap": 2, "gap_or_order": 32,
                  "legacy_human_without_speaker": 16, "overlap": 33, "too_short": 106, "unconfirmed_or_flagged": 173}
        report = {"existing_contributions": ["Мария", "Борис", "Ольга"],
                  "candidates": {"Мария": {"seconds": 107.63}, "Борис": {"seconds": 179.44}, "Ольга": {"seconds": 100.53}},
                  "human_seconds_change": {"Мария": {"before": 0.0, "after": 32.03}, "Ольга": {"before": 0.0, "after": 2.91}},
                  "human_seconds_lost": [], "group_selection": {"method": "transcript_or_vocab"},
                  "selection_explanation": [{"code": c, "description": voices.SELECTION_REASONS[c], "count": n}
                                            for c, n in counts.items()]}
        snapshot = {"group": "ip-demo-course-1", "json": "zapis1_2026-11-08_23-12-04.s2.transcript.json",
                    "vocab_path": "rus-transcriber.vocab.json", "vocab_hash": "0" * 64, "report": report}
        lines = core.voice_action_lines(snapshot)
        self.assertLessEqual(len("\n".join(lines)), 1100)
        self.assertTrue(all(len(line) <= 200 for line in lines))


if __name__ == "__main__":
    unittest.main()
