"""Синтетические проверки журнала записи конспекта."""
import subprocess
import sys
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import conspect
import core
import review_pack
import transcript


class ConspectRecovery(unittest.TestCase):
    def test_interrupted_replacements_can_be_rolled_back(self):
        for fail_at in (1, 2, 3):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as directory:
                jp = Path(directory) / "demo.transcript.json"
                cj, cm = conspect.path_for(jp, "demo")
                cj.write_bytes(b"old-json")
                cm.write_bytes(b"old-markdown")
                real_replace = conspect.os.replace
                calls = 0

                def fail_once(src, dst):
                    nonlocal calls
                    calls += 1
                    if calls == fail_at:
                        raise OSError("synthetic failure")
                    return real_replace(src, dst)

                with conspect.write_lock(jp), patch.object(conspect.os, "replace", side_effect=fail_once):
                    with self.assertRaises(OSError):
                        conspect.write_files(jp, [(cj, b"new-json"), (cm, b"new-markdown")])
                with conspect.write_lock(jp):
                    conspect.recover_pending(jp)
                self.assertEqual(cj.read_bytes(), b"old-json")
                self.assertEqual(cm.read_bytes(), b"old-markdown")

    def test_creation_interrupted_after_first_file_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            cj, cm = conspect.path_for(jp, "demo")
            original = conspect._replace_bytes

            def fail_markdown(path, value):
                if path == cm:
                    raise OSError("synthetic failure")
                return original(path, value)

            with conspect.write_lock(jp), patch.object(conspect, "_replace_bytes", side_effect=fail_markdown):
                with self.assertRaises(OSError):
                    conspect.write_files(jp, [(cj, b"json"), (cm, b"markdown")], create=True)
            with conspect.write_lock(jp):
                self.assertTrue(conspect.recover_pending(jp))
            self.assertFalse(cj.exists())
            self.assertFalse(cm.exists())

    def test_abrupt_process_exit_leaves_recoverable_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            cj, cm = conspect.path_for(jp, "demo")
            cj.write_bytes(b"old-json")
            cm.write_bytes(b"old-markdown")
            script = """
import os, sys
from pathlib import Path
import conspect
jp, cj, cm = map(Path, sys.argv[1:])
original = conspect._replace_bytes
def interrupt(path, value):
    if path == cm:
        os._exit(23)
    original(path, value)
conspect._replace_bytes = interrupt
with conspect.write_lock(jp):
    conspect.write_files(jp, [(cj, b'new-json'), (cm, b'new-markdown')])
"""
            result = subprocess.run([sys.executable, "-B", "-c", script, str(jp), str(cj), str(cm)],
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertTrue(conspect._journal_path(jp).exists())
            with conspect.write_lock(jp):
                self.assertTrue(conspect.recover_pending(jp))
            self.assertEqual(cj.read_bytes(), b"old-json")
            self.assertEqual(cm.read_bytes(), b"old-markdown")

    def test_external_change_blocks_recovery_without_modifying_files(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            cj, cm = conspect.path_for(jp, "demo")
            cj.write_bytes(b"old-json")
            cm.write_bytes(b"old-markdown")
            original = conspect._replace_bytes

            def fail_markdown(path, value):
                if path == cm:
                    raise OSError("synthetic failure")
                return original(path, value)

            with conspect.write_lock(jp), patch.object(conspect, "_replace_bytes", side_effect=fail_markdown):
                with self.assertRaises(OSError):
                    conspect.write_files(jp, [(cj, b"new-json"), (cm, b"new-markdown")])
            cj.write_bytes(b"external-edit")
            with conspect.write_lock(jp), self.assertRaises(ValueError):
                conspect.recover_pending(jp)
            self.assertEqual(cj.read_bytes(), b"external-edit")
            self.assertEqual(cm.read_bytes(), b"old-markdown")
            self.assertTrue(conspect._journal_path(jp).exists())

    def test_corrupt_journal_blocks_new_write_without_touching_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            cj, cm = conspect.path_for(jp, "demo")
            cj.write_bytes(b"old-json")
            cm.write_bytes(b"old-markdown")
            journal = conspect._journal_path(jp)
            journal.write_bytes(b"not json")
            with self.assertRaises(ValueError):
                conspect.save_linked(jp, "demo", "новый текст")
            self.assertEqual(cj.read_bytes(), b"old-json")
            self.assertEqual(cm.read_bytes(), b"old-markdown")
            self.assertEqual(journal.read_bytes(), b"not json")

    def test_interrupted_recovery_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            cj, cm = conspect.path_for(jp, "demo")
            cj.write_bytes(b"old-json")
            cm.write_bytes(b"old-markdown")
            original = conspect._replace_bytes

            def fail_markdown(path, value):
                if path == cm:
                    raise OSError("synthetic failure")
                return original(path, value)

            with conspect.write_lock(jp), patch.object(conspect, "_replace_bytes", side_effect=fail_markdown):
                with self.assertRaises(OSError):
                    conspect.write_files(jp, [(cj, b"new-json"), (cm, b"new-markdown")])
                with self.assertRaises(OSError):
                    conspect.recover_pending(jp)
            self.assertTrue(conspect._journal_path(jp).exists())
            with conspect.write_lock(jp):
                self.assertTrue(conspect.recover_pending(jp))
            self.assertEqual(cj.read_bytes(), b"old-json")
            self.assertEqual(cm.read_bytes(), b"old-markdown")

    def test_next_public_write_recovers_before_creating(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            jp.write_text(json.dumps({"recording_id": "demo", "source_revision": 1,
                                      "segments": [], "review": []}), encoding="utf-8")
            cj, cm = conspect.path_for(jp, "demo")
            original = conspect._replace_bytes

            def fail_markdown(path, value):
                if path == cm:
                    raise OSError("synthetic failure")
                return original(path, value)

            with patch.object(conspect, "_replace_bytes", side_effect=fail_markdown):
                with self.assertRaises(OSError):
                    conspect.save_linked(jp, "demo", "Первый текст")
            self.assertTrue(conspect._journal_path(jp).exists())
            result = conspect.save_linked(jp, "demo", "Второй текст")
            written = Path(result["conspect_md"]).read_text(encoding="utf-8")
            self.assertTrue(written.endswith("\n\nВторой текст"))
            self.assertNotIn("Первый текст", written)
            self.assertFalse(conspect._journal_path(jp).exists())
            self.assertEqual(json.loads(cj.read_text(encoding="utf-8"))["assembly_revision"], 1)

    def test_render_reports_three_distinct_states(self):
        data = {"recording_id": "synthetic", "source_revision": 4, "segments": []}
        obj = {"recording_id": "synthetic", "analysis_revision": 3, "template": "", "quotes": {}}
        _, report = conspect.render(obj, data)
        self.assertEqual(report["assembly_revision"], 4)
        self.assertEqual(obj["assembly_revision"], 4)
        self.assertEqual(report["analysis_revision"], 3)
        self.assertTrue(report["analysis_needs_review"])
        self.assertEqual(report["external_publication"], "not_tracked")

    def test_readonly_render_reports_but_does_not_record_attempted_revision(self):
        data = {"recording_id": "synthetic", "source_revision": 4, "segments": []}
        obj = {"recording_id": "synthetic", "analysis_revision": 2, "assembly_revision": 3,
               "template": "", "quotes": {}}
        _, report = conspect.render(obj, data, update_last=False)
        self.assertEqual(report["rendered_source_revision"], 4)
        self.assertEqual(report["assembly_revision"], 3)
        self.assertEqual(obj["assembly_revision"], 3)

    def test_saved_assembly_and_human_analysis_diverge_after_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            data = {"recording_id": "synthetic", "source_revision": 1, "segments": [], "review": []}
            jp.write_text(json.dumps(data), encoding="utf-8")
            created = conspect.save_linked(jp, "demo", "Конспект")
            self.assertEqual(created["assembly_revision"], 1)
            self.assertIsNone(created["analysis_revision"])
            with patch.object(core, "_done_json", return_value={"json": str(jp)}):
                snapshot = core.describe_conspect_confirmation("demo", "demo")
                core.confirm_conspect_analysis(snapshot)
            data["source_revision"] = 2
            jp.write_text(json.dumps(data), encoding="utf-8")
            report = conspect.rerender(jp, "demo")["demo"]
            saved = json.loads(Path(created["conspect_json"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["assembly_revision"], 2)
            self.assertEqual(saved["analysis_revision"], 1)
            self.assertTrue(report["analysis_needs_review"])
            self.assertEqual(report["external_publication"], "not_tracked")

    def test_second_process_waits_for_recording_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            script = ("import conspect,sys; "
                      "print('ready',flush=True); "
                      "exec('with conspect.write_lock(sys.argv[1]):\\n print(\"acquired\",flush=True)')")
            with conspect.write_lock(jp):
                process = subprocess.Popen([sys.executable, "-B", "-c", script, str(jp)],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self.assertEqual(process.stdout.readline().strip(), "ready")
                with self.assertRaises(subprocess.TimeoutExpired):
                    process.wait(timeout=0.5)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn("acquired", stdout)

    def test_concurrent_creators_do_not_overwrite_one_another(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            cj, cm = conspect.path_for(jp, "demo")
            script = """
import sys
from pathlib import Path
import conspect
jp, cj, cm = map(Path, sys.argv[1:4])
value = sys.argv[4].encode()
try:
    with conspect.write_lock(jp):
        conspect.recover_pending(jp)
        conspect.write_files(jp, [(cj, value), (cm, value)], create=True)
except FileExistsError:
    sys.exit(3)
"""
            processes = [subprocess.Popen([sys.executable, "-B", "-c", script, str(jp), str(cj),
                                            str(cm), value], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                         for value in ("one", "two")]
            results = [p.communicate(timeout=10) for p in processes]
            self.assertEqual(sorted(p.returncode for p in processes), [0, 3], results)
            self.assertIn(cj.read_bytes(), (b"one", b"two"))
            self.assertEqual(cj.read_bytes(), cm.read_bytes())

    def test_concurrent_transcript_renames_preserve_both_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            md = Path(directory) / "demo.transcript.md"
            data = {"file": "synthetic.wav", "duration": "00:03", "recording_id": "synthetic",
                    "source_revision": 1, "speakers": ["А", "Б"],
                    "speaker_labels": ["А", "Б"], "segments": [
                        {"speaker": "А", "label": "А", "start": 0, "end": 1, "text": "альфа"},
                        {"speaker": "Б", "label": "Б", "start": 2, "end": 3, "text": "бета"}]}
            jp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            script = """
import sys
import transcript
print('ready', flush=True)
transcript.rename(sys.argv[1], sys.argv[2], {sys.argv[3]: sys.argv[4]})
"""
            with conspect.write_lock(jp):
                processes = [subprocess.Popen([sys.executable, "-B", "-c", script, str(jp), str(md), a, b],
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                             for a, b in (("А", "Анна"), ("Б", "Борис"))]
                for process in processes:
                    self.assertEqual(process.stdout.readline().strip(), "ready")
            results = [process.communicate(timeout=10) for process in processes]
            for process, (_, stderr) in zip(processes, results):
                self.assertEqual(process.returncode, 0, stderr)
            result = json.loads(jp.read_text(encoding="utf-8"))
            self.assertEqual([s["speaker"] for s in result["segments"]], ["Анна", "Борис"])
            self.assertEqual(result["source_revision"], 3)

    def test_free_answer_planning_keeps_recording_locked(self):
        import freeform
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            md = Path(directory) / "demo.transcript.md"
            entered, release, acquired = threading.Event(), threading.Event(), threading.Event()
            errors = []

            def plan(*args):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("planning timeout")
                return [], [], {}, []

            def run_plan():
                try:
                    review_pack.apply_free(jp, md, {})
                except Exception as exc:
                    errors.append(exc)

            def contender():
                with conspect.write_lock(jp):
                    acquired.set()

            with patch.object(review_pack, "manifest", return_value={"items": []}), \
                    patch("transcript.load", return_value={"segments": [], "review": []}), \
                    patch.object(freeform, "plan", side_effect=plan):
                first = threading.Thread(target=run_plan)
                first.start()
                try:
                    self.assertTrue(entered.wait(5))
                    second = threading.Thread(target=contender)
                    second.start()
                    self.assertFalse(acquired.wait(0.3))
                finally:
                    release.set()
                    first.join(5)
                second.join(5)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertFalse(errors)
            self.assertTrue(acquired.is_set())

    def test_worker_creation_refuses_existing_or_newly_created_result(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            md = Path(directory) / "demo.transcript.md"
            data = {"file": "demo.wav", "duration": "00:01", "speakers": [], "segments": []}
            transcript.ensure_new_result(jp, md)
            md.write_bytes(b"existing markdown")
            with self.assertRaises(FileExistsError):
                transcript.save_new(data, jp, md)
            self.assertFalse(jp.exists())
            self.assertEqual(md.read_bytes(), b"existing markdown")
            with self.assertRaises(FileExistsError):
                transcript.ensure_new_result(jp, md)

    def test_two_worker_writers_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            jp = Path(directory) / "demo.transcript.json"
            md = Path(directory) / "demo.transcript.md"
            script = """
import sys
import transcript
data = {'file': sys.argv[3], 'duration': '00:01', 'speakers': [], 'segments': []}
try:
    transcript.ensure_new_result(sys.argv[1], sys.argv[2])
    print('ready', flush=True)
    transcript.save_new(data, sys.argv[1], sys.argv[2])
except FileExistsError:
    sys.exit(3)
"""
            with conspect.write_lock(jp):
                processes = [subprocess.Popen([sys.executable, "-B", "-c", script, str(jp), str(md), name],
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                             for name in ("one.wav", "two.wav")]
            results = [p.communicate(timeout=10) for p in processes]
            self.assertEqual(sorted(p.returncode for p in processes), [0, 3], results)
            saved = json.loads(jp.read_text(encoding="utf-8"))
            self.assertIn(saved["file"], ("one.wav", "two.wav"))
            self.assertIn(saved["file"], md.read_text(encoding="utf-8"))
