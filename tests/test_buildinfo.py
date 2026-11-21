"""Build provenance uses only a synthetic installation and Git repository."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import buildinfo


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

    def test_installer_copy_list_matches_manifest_payload(self):
        text = (Path(__file__).resolve().parent.parent / "install.ps1").read_text(
            encoding="utf-8-sig")
        root = re.search(r'foreach \(\$f in @\((.*?)\)\) \{', text, re.S)
        self.assertIsNotNone(root)
        self.assertEqual(tuple(re.findall(r'"([^"]+)"', root.group(1))), buildinfo.ROOT_FILES)
        docs = {name.replace("\\", "/") for name in
                re.findall(r'docs\\[A-Za-z0-9_.\\-]+\.md', text)}
        self.assertEqual(docs, set(buildinfo.DOC_FILES))
        support = re.search(r'foreach \(\$sub in @\((.*?)\)\)', text, re.S)
        self.assertIsNotNone(support)
        self.assertEqual(tuple(re.findall(r'"([^"]+)"', support.group(1))),
                         buildinfo.SUPPORT_DIRS)

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
