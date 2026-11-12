"""Синтетические проверки идентичности цитат и актуальности конспектов."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import conspect
import core
import quotes


def segment(speaker, words, start):
    entries = [{"w": word, "s": start + n * 0.2, "e": start + n * 0.2 + 0.1,
                "word_id": f"{speaker}-{start}-{n}"} for n, word in enumerate(words.split())]
    return {"speaker": speaker, "label": speaker, "start": start,
            "end": entries[-1]["e"], "text": words, "words": entries}


def recording(*segments):
    return {"recording_id": "synthetic-recording", "source_revision": 1,
            "segments": list(segments), "review": []}


class Stage4IdentityContract(unittest.TestCase):
    def test_explicit_speaker_selects_matching_occurrence(self):
        data = recording(segment("Анна", "один два три четыре", 0.0),
                         segment("Борис", "один два три четыре", 5.0))
        obj, report = conspect.link("Борис: «один два три четыре»", data)
        self.assertEqual(report["linked"], 1)
        self.assertEqual(obj["quotes"]["1"]["parts"][0]["t1"], 5.0)

    def test_ambiguous_occurrence_is_not_linked_automatically(self):
        data = recording(segment("Анна", "один два три четыре", 0.0),
                         segment("Борис", "один два три четыре", 5.0))
        obj, report = conspect.link("«один два три четыре»", data)
        self.assertEqual(report["linked"], 0)
        self.assertEqual(obj["quotes"], {})

    def test_changed_word_is_not_verbatim(self):
        data = recording(segment("Анна", "один два три четыре пять шесть семь восемь девять десять", 0.0))
        report = quotes.check("«один два три четыре пять шесть семь восемь девять другое»", data)
        self.assertEqual(report["summary"]["verbatim"], 0)
        self.assertTrue(report["problems"])

    def test_overlapping_speakers_are_not_mixed(self):
        data = recording(segment("Анна", "альфа бета", 0.0), segment("Борис", "гамма дельта", 0.0))
        obj, report = conspect.link("Анна: «альфа бета»", data)
        self.assertEqual(report["linked"], 1)
        base = obj["quotes"]["1"]["base"]
        self.assertNotIn("гамма", base)
        self.assertNotIn("дельта", base)

    def test_link_records_source_and_word_ids(self):
        data = recording(segment("Анна", "альфа бета", 0.0))
        obj, _ = conspect.link("Анна: «альфа бета»", data)
        self.assertEqual(obj["recording_id"], data["recording_id"])
        self.assertEqual(obj["source_revision"], data["source_revision"])
        self.assertEqual(obj["quotes"]["1"]["parts"][0]["word_ids"],
                         [w["word_id"] for w in data["segments"][0]["words"]])

    def test_repeated_phrase_of_same_speaker_is_ambiguous(self):
        data = recording(segment("Анна", "альфа бета", 0.0), segment("Анна", "альфа бета", 5.0))
        _, report = conspect.link("Анна: «альфа бета»", data)
        self.assertEqual(report["linked"], 0)

    def test_duplicate_word_ids_do_not_link_automatically(self):
        data = recording(segment("Анна", "альфа бета", 0.0))
        data["segments"][0]["words"][1]["word_id"] = data["segments"][0]["words"][0]["word_id"]
        _, report = conspect.link("Анна: «альфа бета»", data)
        self.assertEqual(report["linked"], 0)

    def test_rebuild_after_source_change_requires_analysis_review(self):
        data = recording(segment("Анна", "альфа бета", 0.0))
        obj, _ = conspect.link("Вывод. Анна: «альфа бета»", data)
        data["source_revision"] += 1
        _, report = conspect.render(obj, data)
        self.assertTrue(report.get("analysis_needs_review"))
        _, report_again = conspect.render(obj, data)
        self.assertTrue(report_again.get("analysis_needs_review"))

    def test_readonly_render_does_not_mutate_input(self):
        data = recording(segment("Анна", "альфа бета", 0.0))
        obj, _ = conspect.link("Анна: «альфа бета»", data)
        before_obj, before_data = copy.deepcopy(obj), copy.deepcopy(data)
        conspect.render(obj, data, update_last=False)
        self.assertEqual(obj, before_obj)
        self.assertEqual(data, before_data)

    def test_another_recording_never_substitutes_words(self):
        data = recording(segment("Анна", "альфа бета", 0.0))
        obj, _ = conspect.link("Анна: «альфа бета»", data)
        other = recording(segment("Анна", "гамма дельта", 0.0))
        other["recording_id"] = "other-recording"
        text, report = conspect.render(obj, other)
        self.assertIn("место в стенограмме не найдено", text)
        self.assertNotIn("гамма", text)
        self.assertIn("source_conflict", report)

    def test_missing_word_id_is_not_retargeted_by_time(self):
        data = recording(segment("Анна", "альфа бета", 0.0))
        obj, _ = conspect.link("Анна: «альфа бета»", data)
        data["segments"][0]["words"][0]["word_id"] = "new-word"
        text, report = conspect.render(obj, data)
        self.assertTrue(report["missing"])
        self.assertIn("место в стенограмме не найдено", text)

    def test_review_requires_explicit_confirmation(self):
        data = recording(segment("Анна", "альфа бета", 0.0))
        obj, _ = conspect.link("Анна: «альфа бета»", data)
        self.assertTrue(conspect.render(obj, data)[1]["analysis_needs_review"])
        conspect.confirm_analysis(obj, data)
        self.assertFalse(conspect.render(obj, data)[1]["analysis_needs_review"])
        data["source_revision"] += 1
        self.assertTrue(conspect.render(obj, data)[1]["analysis_needs_review"])

    def test_legacy_overlap_is_not_silently_substituted(self):
        data = recording(segment("Анна", "альфа бета", 0.0), segment("Борис", "гамма дельта", 0.0))
        obj = {"version": 1, "template": "{{q:1}}", "quotes": {"1": {"author": "альфа бета", "speaker": "Анна",
               "base": "альфа бета", "base_speaker": "Анна", "parts": [{"t1": 0.0, "t2": 0.2}]}}}
        text, report = conspect.render(obj, data, update_last=False)
        self.assertTrue(report["legacy_unverified"])
        self.assertTrue(report["missing"])
        self.assertNotIn("гамма", text)

    def test_unknown_speaker_is_not_discarded(self):
        data = recording(segment("Анна", "альфа бета", 0.0))
        result = quotes.check("Борис: «альфа бета»", data)
        self.assertEqual(result["problems"][0]["unknown_speaker"], "Борис")
        _, report = conspect.link("Борис: «альфа бета»", data)
        self.assertEqual(report["linked"], 0)

    def test_legacy_cannot_confirm_unknown_source(self):
        data = recording(segment("Анна", "альфа бета", 0.0))
        old = {"version": 1, "template": "", "quotes": {}}
        with self.assertRaises(ValueError):
            conspect.confirm_analysis(old, data)

    def test_confirm_another_recording_is_rejected(self):
        data = recording(segment("Анна", "альфа бета", 0.0))
        obj, _ = conspect.link("Анна: «альфа бета»", data)
        other = copy.deepcopy(data)
        other["recording_id"] = "other-recording"
        before = copy.deepcopy(obj)
        with self.assertRaises(ValueError):
            conspect.confirm_analysis(obj, other)
        self.assertEqual(obj, before)

    def test_reordered_linked_words_require_review_not_old_order_substitution(self):
        data = recording(segment("Анна", "альфа бета гамма", 0.0))
        obj, _ = conspect.link("Анна: «альфа бета гамма»", data)
        data["segments"][0]["words"].reverse()
        data["source_revision"] += 1
        _, report = conspect.render(obj, data, update_last=False)
        self.assertTrue(report["missing"] or report.get("order_conflicts"))

    def test_candidate_selection_requires_matching_recording_revision_and_ids(self):
        data = recording(segment("Анна", "альфа бета", 0.0), segment("Борис", "альфа бета", 5.0))
        text = "«альфа бета»"
        _, report = conspect.link(text, data)
        options = report["candidates"][0]["candidates"]
        selection = {"1": {"recording_id": data["recording_id"], "source_revision": data["source_revision"],
                            "word_ids": options[1]["word_ids"]}}
        obj, report = conspect.link(text, data, selections=selection)
        self.assertEqual(report["linked"], 1)
        self.assertEqual(obj["quotes"]["1"]["parts"][0]["t1"], 5.0)
        changed = copy.deepcopy(data)
        changed["source_revision"] += 1
        with self.assertRaises(ValueError):
            conspect.link(text, changed, selections=selection)

    def test_confirmation_rejects_concurrent_change(self):
        data = recording(segment("Анна", "альфа бета", 0.0))
        obj, _ = conspect.link("Анна: «альфа бета»", data)
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            cj, _ = conspect.path_for(jp, "demo")
            jp.write_text(json.dumps(data), encoding="utf-8")
            cj.write_text(json.dumps(obj), encoding="utf-8")
            with patch.object(core, "_done_json", return_value={"json": str(jp)}):
                snapshot = core.describe_conspect_confirmation("demo", "demo")
                data["source_revision"] += 1
                jp.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(core.CoreError):
                    core.confirm_conspect_analysis(snapshot)
                self.assertIsNone(json.loads(cj.read_text(encoding="utf-8"))["analysis_revision"])

    def test_selection_stale_source_does_not_create_files(self):
        data = recording(segment("Анна", "альфа бета", 0.0), segment("Борис", "альфа бета", 5.0))
        text = "«альфа бета»"
        _, report = conspect.link(text, data)
        selection = {"1": {"recording_id": data["recording_id"], "source_revision": data["source_revision"],
                            "word_ids": report["candidates"][0]["candidates"][1]["word_ids"]}}
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            jp.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(core, "_done_json", return_value={"json": str(jp)}):
                snapshot, _ = core.prepare_selected_conspect("demo", "selected", text, selection)
                data["source_revision"] += 1
                jp.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(core.CoreError):
                    core.save_selected_conspect(snapshot, text, selection)
                self.assertFalse(conspect.path_for(jp, "selected")[0].exists())

    def test_selected_conspect_can_be_reviewed_without_overwriting(self):
        data = recording(segment("Анна", "альфа бета", 0.0), segment("Борис", "альфа бета", 5.0))
        text = "«альфа бета»"
        _, report = conspect.link(text, data)
        selection = {"1": {"recording_id": data["recording_id"], "source_revision": data["source_revision"],
                            "word_ids": report["candidates"][0]["candidates"][1]["word_ids"]}}
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            jp.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(core, "_done_json", return_value={"json": str(jp)}):
                snapshot, _ = core.prepare_selected_conspect("demo", "selected", text, selection)
                result = core.save_selected_conspect(snapshot, text, selection)
                self.assertTrue(result["analysis_needs_review"])
                with self.assertRaises(core.CoreError):
                    core.save_selected_conspect(snapshot, text, selection)
                verified = core.describe_conspect_confirmation("demo", "selected")
                core.confirm_conspect_analysis(verified)
                obj = json.loads(Path(result["conspect_json"]).read_text(encoding="utf-8"))
                self.assertEqual(obj["analysis_revision"], data["source_revision"])

    def test_two_ambiguous_quotes_can_be_selected_for_one_conspect(self):
        """Желаемый сценарий: выбрать несколько мест, не перезаписывая уже созданный файл."""
        data = recording(segment("Анна", "альфа бета", 0.0), segment("Борис", "альфа бета", 5.0),
                         segment("Анна", "гамма дельта", 10.0), segment("Борис", "гамма дельта", 15.0))
        text = "«альфа бета» и «гамма дельта»"
        _, report = conspect.link(text, data)
        selections = {str(entry["quote_index"]): {"recording_id": data["recording_id"],
                      "source_revision": data["source_revision"], "word_ids": entry["candidates"][1]["word_ids"]}
                      for entry in report["candidates"]}
        self.assertEqual(len(selections), 2)
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            jp.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(core, "_done_json", return_value={"json": str(jp)}):
                snapshot, preview = core.prepare_selected_conspect("demo", "selected", text, selections)
                self.assertEqual(preview["linked"], 2)
                lines = core.selected_conspect_confirmation_lines("demo", "selected", snapshot, preview, selections)
                self.assertTrue(any("альфа бета" in line and "Борис" in line for line in lines))
                self.assertTrue(any("гамма дельта" in line and "Борис" in line for line in lines))
                import confirm
                body = "Выбор мест цитат\n\n" + "\n".join(lines)
                self.assertEqual(confirm.clean(body), body)
                result = core.save_selected_conspect(snapshot, text, selections)
                obj = json.loads(Path(result["conspect_json"]).read_text(encoding="utf-8"))
                self.assertEqual(len(obj["quotes"]), 2)

    def test_confirmation_lines_reject_truncation_and_indistinguishable_options(self):
        snapshot = {"source_revision": 1}
        selection = {"1": {"word_ids": ["w1"]}}
        report = {"candidates": [{"quote_index": 1, "quote": "а" * 121, "candidates": [
            {"word_ids": ["w1"], "speaker": "Анна", "t1": 1, "t2": 2},
            {"word_ids": ["w2"], "speaker": "Борис", "t1": 3, "t2": 4}]}]}
        with self.assertRaises(core.CoreError):
            core.selected_conspect_confirmation_lines("demo", "selected", snapshot, report, selection)
        report["candidates"][0]["quote"] = "альфа бета"
        report["candidates"][0]["candidates"][1].update(speaker="Анна", t1=1, t2=2)
        with self.assertRaises(core.CoreError):
            core.selected_conspect_confirmation_lines("demo", "selected", snapshot, report, selection)

    def test_three_long_choices_are_fully_visible(self):
        import confirm
        selections, candidates = {}, []
        for n in range(1, 4):
            selections[str(n)] = {"word_ids": [f"w{n}"]}
            candidates.append({"quote_index": n, "quote": "а" * 120, "candidates": [
                {"word_ids": [f"w{n}"], "speaker": "Б" * 50, "t1": n, "t2": n + 1}]})
        lines = core.selected_conspect_confirmation_lines("demo", "selected", {"source_revision": 1},
                                                           {"candidates": candidates}, selections)
        text = "Выбор мест цитат\n\n" + "\n".join(lines)
        self.assertEqual(confirm.clean(text), text)
        self.assertEqual(sum("а" * 120 in line for line in lines), 3)

    def test_selection_limit_rejects_four_before_reading_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            with patch.object(core, "_done_json", return_value={"json": str(jp)}):
                with self.assertRaises(core.CoreError):
                    core.prepare_selected_conspect("demo", "selected", "текст", {str(n): {} for n in range(4)})
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_one_stale_choice_blocks_entire_multi_selection(self):
        data = recording(segment("Анна", "альфа бета", 0.0), segment("Борис", "альфа бета", 5.0),
                         segment("Анна", "гамма дельта", 10.0), segment("Борис", "гамма дельта", 15.0))
        text = "«альфа бета» и «гамма дельта»"
        _, report = conspect.link(text, data)
        selections = {str(entry["quote_index"]): {"recording_id": data["recording_id"],
                      "source_revision": data["source_revision"], "word_ids": entry["candidates"][1]["word_ids"]}
                      for entry in report["candidates"]}
        selections["2"]["source_revision"] += 1
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            jp.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(core, "_done_json", return_value={"json": str(jp)}):
                with self.assertRaises(core.CoreError):
                    core.prepare_selected_conspect("demo", "selected", text, selections)
            self.assertFalse(conspect.path_for(jp, "selected")[0].exists())
