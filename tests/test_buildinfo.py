"""Build provenance uses only a synthetic installation and Git repository."""
import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import buildinfo

REPO = Path(__file__).resolve().parent.parent


def setUpModule():
    """An inherited GIT_DIR (hooks, rebase --exec) overrides git -C: never write to that repository."""
    environment = patch.dict(os.environ)
    environment.start()
    unittest.addModuleCleanup(environment.stop)
    for name in [n for n in os.environ if n.upper().startswith("GIT_")]:
        del os.environ[name]


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source"
        self.target = Path(self.tmp.name) / "target"
        self.source.mkdir()
        (self.source / "server.py").write_text('VERSION = "2099.1.1.1"\n', encoding="utf-8")
        (self.source / "cards.html").write_text("synthetic", encoding="utf-8")
        (self.source / "tools").mkdir()
        (self.source / "tools/check.py").write_text("# synthetic\n", encoding="utf-8")
        shutil.copytree(self.source, self.target)

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.source), *args], check=True,
                              capture_output=True, text=True, encoding="utf-8").stdout.strip()

    def init_git(self):
        self.git("init", "-q")
        self.git("config", "user.name", "Synthetic")
        self.git("config", "user.email", "synthetic@example.invalid")
        self.git("add", ".")
        self.git("commit", "-qm", "synthetic")

    def test_unknown_without_git(self):
        got = buildinfo.identity(self.source)
        self.assertEqual(got["source_state"], "unknown")
        self.assertIsNone(got["commit"])
        self.assertEqual(got["release"], "2099.1.1.1")

    def test_metadata_consumers_survive_missing_or_failing_buildinfo(self):
        import runinfo
        import selfcheck
        import worker
        def check_consumers():
            self.assertEqual(worker.asr_run_metadata()["build"]["source_state"], "unknown")
            self.assertEqual(runinfo.code_revision(self.source)["build"]["source_state"], "unknown")
            self.assertEqual(selfcheck.run(tests=False, repo=False, morph=False)["build"]["source_state"],
                             "unknown")
        with patch.object(buildinfo, "identity", side_effect=RuntimeError("synthetic failure")):
            check_consumers()
        with patch.dict(sys.modules, {"buildinfo": None}):
            check_consumers()

    def test_manifest_survives_install_without_git_and_detects_edit(self):
        self.init_git()
        sha = self.git("rev-parse", "HEAD")
        buildinfo.write_manifest(self.source, self.target)
        got = buildinfo.identity(self.target)
        self.assertEqual((got["commit"], got["source_state"], got["manifest_status"]),
                         (sha, "clean", "matched"))
        (self.target / "cards.html").write_text("modified", encoding="utf-8")
        self.assertEqual(buildinfo.identity(self.target)["manifest_status"], "modified")
        self.assertEqual(buildinfo.identity(self.target)["source_state"], "dirty")

    def test_untracked_source_is_dirty(self):
        self.init_git()
        (self.source / "new.py").write_text("# new", encoding="utf-8")
        self.assertEqual(buildinfo.identity(self.source)["source_state"], "dirty")

    def test_copy_mismatch_fails(self):
        (self.target / "cards.html").unlink()
        with self.assertRaises(ValueError):
            buildinfo.write_manifest(self.source, self.target)
        self.assertFalse((self.target / buildinfo.MANIFEST).exists())

    def test_installed_manifest_has_no_local_paths(self):
        self.init_git()
        data = buildinfo.write_manifest(self.source, self.target)
        self.assertNotIn(str(self.source), json.dumps(data))
        self.assertNotIn(str(self.target), json.dumps(data))
        self.assertEqual(buildinfo.identity(self.target)["payload_sha256"], data["payload_sha256"])

    def test_manifest_does_not_read_supplied_paths(self):
        data = buildinfo.write_manifest(self.source, self.target)
        data["files"]["../../private"] = "0" * 64
        data["payload_sha256"] = buildinfo.payload_hash(data["files"])
        (self.target / buildinfo.MANIFEST).write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(buildinfo.identity(self.target)["manifest_status"], "modified")

    def test_invalid_manifest_is_unknown(self):
        (self.target / buildinfo.MANIFEST).write_text('{"schema":1}', encoding="utf-8")
        self.assertEqual(buildinfo.identity(self.target)["manifest_status"], "invalid")
        self.assertIsNone(buildinfo.identity(self.target)["commit"])

    def test_ignored_data_and_worktrees_do_not_enter_payload(self):
        first = buildinfo.delivery_hashes(self.source)
        p = self.source / ".opencode/worktrees/task"
        p.mkdir(parents=True)
        (p / "server.py").write_text("private task", encoding="utf-8")
        (self.source / "config.json").write_text('{"private":true}', encoding="utf-8")
        self.assertEqual(buildinfo.delivery_hashes(self.source), first)

    def test_support_file_change_is_detected(self):
        buildinfo.write_manifest(self.source, self.target)
        (self.target / "tools/check.py").write_text("changed", encoding="utf-8")
        self.assertEqual(buildinfo.identity(self.target)["source_state"], "dirty")

    def test_installer_takes_root_list_from_buildinfo(self):
        text = (REPO / "install.ps1").read_text(encoding="utf-8-sig")
        step = text[text.index('Step "4/10"'):text.index('Step "5/10"')]
        self.assertNotRegex(step, r'@\(\s*"[A-Za-z0-9_]+\.py"', "hard-coded root file list is back")
        listing = step.index('(Join-Path $Source "buildinfo.py") --source $Source --target $Target --list-root')
        self.assertLess(step.index("$manifestPython = $basePyExe"), listing)
        checked = step[listing:step.index("foreach ($f in $rootFiles)")]
        self.assertIn("$LASTEXITCODE -ne 0", checked)
        self.assertIn("-not $rootFiles.Count", checked)
        self.assertIn("'^" + buildinfo.ROOT_NAME.pattern + "$'", checked)
        self.assertIn("Fail ", checked)
        docs = {name.replace("\\", "/") for name in
                re.findall(r'docs\\[A-Za-z0-9_.\\-]+\.md', text)}
        self.assertEqual(docs, set(buildinfo.DOC_FILES))
        support = re.search(r'foreach \(\$sub in @\((.*?)\)\)', text, re.S)
        self.assertIsNotNone(support)
        self.assertEqual(tuple(re.findall(r'"([^"]+)"', support.group(1))),
                         buildinfo.SUPPORT_DIRS)

    def test_repository_root_ships_tracked_program_files(self):
        if not (REPO / ".git").exists() or not shutil.which("git"):
            self.skipTest("no Git checkout (installed copy)")
        tracked = subprocess.run(["git", "-C", str(REPO), "ls-files"], check=True,
                                 capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
        expected = {n for n in tracked if n and "/" not in n and (
            n.lower().endswith(buildinfo.PROGRAM_SUFFIXES) or n in buildinfo.FIXED_ROOT_FILES)}
        # An untracked root file here would be installed too: keep scratch work out of the root.
        self.assertEqual(set(buildinfo.root_files(REPO)), expected)
        self.assertLessEqual({"server.py", "worker.py", "buildinfo.py", "cards.html", "workflow.html",
                              "answer_phrases.tsv", "README.md", "install.ps1"}, expected)

    def test_new_root_module_ships_without_list_entry(self):
        for name in ("new_module.py", "new_table.tsv", "new_page.html"):
            (self.source / name).write_text("# synthetic\n", encoding="utf-8")
            shutil.copy2(self.source / name, self.target / name)
        data = buildinfo.write_manifest(self.source, self.target)
        self.assertLessEqual({"new_module.py", "new_table.tsv", "new_page.html"}, set(data["files"]))
        self.assertEqual(buildinfo.identity(self.target)["manifest_status"], "matched")
        (self.target / "new_table.tsv").unlink()
        with self.assertRaises(ValueError):
            buildinfo.write_manifest(self.source, self.target)

    def test_non_program_root_files_do_not_ship(self):
        first = buildinfo.root_files(self.source)
        (self.source / "sub").mkdir()
        (self.source / "sub/inner.py").write_text("# synthetic", encoding="utf-8")
        (self.source / "folder.py").mkdir()
        for name in (".hidden.py", "заметка.html", "meeting.transcript.html", "Meeting.Transcript.tsv",
                     "notes.md", "config.json", "voices.json", "record.mp3"):
            (self.source / name).write_text("synthetic private", encoding="utf-8")
        self.assertEqual(buildinfo.root_files(self.source), first)
        self.assertEqual(set(buildinfo.root_files(self.source)), {"server.py", "cards.html"})
        self.assertFalse({n for n in buildinfo.delivery_hashes(self.source)
                          if "transcript" in n.lower() or n.startswith(("sub/", "."))})

    def test_linked_root_entry_is_refused_before_copy(self):
        real = buildinfo._link
        def linked_in(place):
            return lambda path: (path.name == "cards.html" and path.parent == place) or real(path)
        with patch.object(buildinfo, "_link", side_effect=linked_in(self.source)):
            self.assertIn("cards.html", buildinfo.root_files(self.source))
            with self.assertRaises(ValueError):
                buildinfo.list_root(self.source)
            with self.assertRaises(ValueError):
                buildinfo.delivery_hashes(self.source)
            code, out = self.list_root_cli("--source", str(self.source))
        self.assertEqual((code, out), (1, ""))
        with patch.object(buildinfo, "_link", side_effect=linked_in(self.target)):
            buildinfo.list_root(self.source)
            with self.assertRaises(ValueError):
                buildinfo.list_root(self.source, self.target)

    def list_root_cli(self, *args):
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["buildinfo.py", *args, "--list-root"]), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            try:
                buildinfo.main()
                code = 0
            except SystemExit as stop:
                code = stop.code
        return code, stdout.getvalue()

    def test_list_root_prints_ascii_names_one_per_line(self):
        (self.source / "new_module.py").write_text("# synthetic", encoding="utf-8")
        (self.source / "записка.py").write_text("# synthetic", encoding="utf-8")
        code, out = self.list_root_cli("--source", str(self.source), "--target", str(self.target))
        self.assertEqual(code, 0)
        self.assertTrue(out.isascii())
        self.assertEqual(tuple(out.splitlines()), ("cards.html", "new_module.py", "server.py"))
        self.assertTrue(all(buildinfo.ROOT_NAME.fullmatch(n) for n in out.splitlines()))
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        self.assertEqual(self.list_root_cli("--source", str(empty)), (1, ""))

    def test_stale_root_file_in_installation_is_kept_and_not_reported(self):
        stale = self.target / "stale_module.py"
        stale.write_text("# left by an older version", encoding="utf-8")
        data = buildinfo.write_manifest(self.source, self.target)
        self.assertNotIn("stale_module.py", data["files"])
        self.assertTrue(stale.exists())
        got = buildinfo.identity(self.target)
        self.assertEqual(got["manifest_status"], "matched")
        self.assertEqual(got["payload_sha256"], data["payload_sha256"])
        (self.target / "cards.html").unlink()
        self.assertEqual(buildinfo.identity(self.target)["manifest_status"], "modified")

    def test_buildinfo_uses_only_standard_library(self):
        tree = ast.parse((REPO / "buildinfo.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                names.add(node.module.split(".")[0])
        self.assertTrue(names)
        self.assertLessEqual(names, set(sys.stdlib_module_names), names - set(sys.stdlib_module_names))

    def test_uninstalled_docs_do_not_enter_payload(self):
        docs = self.source / "docs"
        docs.mkdir()
        (docs / "2026-11-20-code-audit.md").write_text("internal", encoding="utf-8")
        self.assertNotIn("docs/2026-11-20-code-audit.md",
                         buildinfo.delivery_hashes(self.source))

    def test_partial_support_install_matches_present_files(self):
        (self.target / "tools/check.py").unlink()
        data = buildinfo.write_manifest(self.source, self.target)
        self.assertNotIn("tools/check.py", data["files"])
        self.assertEqual(buildinfo.identity(self.target)["manifest_status"], "matched")

    def test_missing_runtime_file_still_fails(self):
        (self.target / "server.py").unlink()
        with self.assertRaises(ValueError):
            buildinfo.write_manifest(self.source, self.target)

    def test_kept_extra_support_file_is_not_deleted_and_is_reported(self):
        extra = self.target / "tools/local-note.txt"
        extra.write_text("synthetic local addition", encoding="utf-8")
        buildinfo.write_manifest(self.source, self.target)
        self.assertTrue(extra.exists())
        self.assertEqual(buildinfo.identity(self.target)["manifest_status"], "modified")

    def test_worktree_git_file_is_supported(self):
        self.init_git()
        linked = Path(self.tmp.name) / "linked"
        self.git("worktree", "add", "--detach", str(linked), "HEAD")
        self.assertTrue((linked / ".git").is_file())
        self.assertEqual(buildinfo.identity(linked)["commit"], self.git("rev-parse", "HEAD"))
        self.git("worktree", "remove", str(linked))


if __name__ == "__main__":
    unittest.main()
