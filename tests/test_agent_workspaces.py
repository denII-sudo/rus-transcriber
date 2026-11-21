"""Housekeeping is exercised only on disposable Git repositories, never on user worktrees."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "agent_workspaces", Path(__file__).resolve().parent.parent / "tools/agent_workspaces.py")
aw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aw)


class HousekeepingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.g("init", "-q", "-b", "main")
        self.g("config", "user.name", "Synthetic")
        self.g("config", "user.email", "synthetic@example.invalid")
        (self.repo / ".gitignore").write_text(".claude/worktrees/\n.opencode/worktrees/\n*.scratch\n",
                                               encoding="utf-8")
        (self.repo / "file.txt").write_text("base\n", encoding="utf-8")
        self.g("add", ".")
        self.g("commit", "-qm", "base")
        self.base = self.g("rev-parse", "HEAD")
        self.cwd = self.repo / ".claude/worktrees/task"
        self.g("worktree", "add", "-b", "feature/test", str(self.cwd), "HEAD")
        (self.cwd / "file.txt").write_text("task\n", encoding="utf-8")
        aw.git(self.cwd, "add", ".")
        aw.git(self.cwd, "commit", "-qm", "task")
        self.head = aw.git(self.cwd, "rev-parse", "HEAD")
        self.config = {"state": str(self.root / "registry"), "opencode": "synthetic-opencode",
                       "repos": [{"root": str(self.repo), "github": "synthetic/example", "excluded": []}]}
        self.m = aw.Manager(self.config)
        self.m.register("claude", "synthetic-session", str(self.cwd))
        self.m.release("claude", "synthetic-session")
        self.path = self.m.record_path("claude", "synthetic-session")
        # A verified merged PR is injected for deletion tests, never a real API.
        self.pr = patch.object(self.m, "merged_pr", return_value={"number": 1})
        self.pr.start()
        self.addCleanup(self.pr.stop)
        # Clean Git worktrees before TemporaryDirectory on Windows (Git files may be read-only).
        self.addCleanup(self.cleanup_git)

    def cleanup_git(self):
        if self.cwd.exists():
            subprocess.run(["git", "-C", str(self.repo), "worktree", "remove", "--force",
                            str(self.cwd)], capture_output=True)
        # TemporaryDirectory can remove Git object files after making them writable.
        for parent, _, files in os.walk(self.root):
            for name in files:
                try:
                    (Path(parent) / name).chmod(0o666)
                except OSError:
                    pass

    def g(self, *args):
        return aw.git(self.repo, *args)

    def test_dry_run_preserves_checkout_and_branch(self):
        self.assertEqual(self.m.sweep_one(self.path), "would-remove")
        self.assertTrue(self.cwd.exists())
        self.assertEqual(self.g("rev-parse", "feature/test"), self.head)

    def test_verified_squash_removes_only_registered_checkout(self):
        # Mark main as squash-merged; original feature commit is NOT its ancestor.
        self.g("merge", "--squash", "feature/test")
        self.g("commit", "-qm", "squash")
        self.assertEqual(self.m.sweep_one(self.path, True), "removed")
        self.assertFalse(self.cwd.exists())
        self.assertTrue(self.repo.exists())
        self.assertFalse(self.m.sandbox_path(self.cwd).exists())
        self.assertEqual(self.m.sweep_one(self.path, True), "already-removed")

    def test_dirty_file_is_kept(self):
        (self.cwd / "file.txt").write_text("unsaved", encoding="utf-8")
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")
        self.assertTrue(self.cwd.exists())

    def test_untracked_file_is_kept(self):
        (self.cwd / "new.txt").write_text("unsaved", encoding="utf-8")
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")

    def test_ignored_file_is_also_kept(self):
        (self.cwd / "secret.scratch").write_text("synthetic", encoding="utf-8")
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")

    def test_native_lock_is_respected_but_registration_works(self):
        self.g("worktree", "lock", str(self.cwd))
        self.m.register("claude", "synthetic-session", str(self.cwd))
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")
        self.g("worktree", "unlock", str(self.cwd))

    def test_registry_lock_waits_for_background_sweep(self):
        state = self.root / "registry"
        marker = state / "held.marker"
        script = (
            "import importlib.util,sys,time\n"
            "from pathlib import Path\n"
            "spec=importlib.util.spec_from_file_location('aw', sys.argv[1])\n"
            "aw=importlib.util.module_from_spec(spec); spec.loader.exec_module(aw)\n"
            "with aw.registry_lock(Path(sys.argv[2])):\n"
            " Path(sys.argv[3]).write_text('held')\n"
            " time.sleep(1.0)\n"
        )
        holder = subprocess.Popen([sys.executable, "-B", "-c", script, str(Path(aw.__file__)),
                                   str(state), str(marker)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(marker.exists())
            config_file = self.root / "config.json"
            config_file.write_text(json.dumps(self.config), encoding="utf-8")
            started = time.monotonic()
            out = subprocess.run(
                [sys.executable, "-B", str(Path(aw.__file__)), "--config", str(config_file),
                 "register", "--tool", "claude", "--session", "waiting-session",
                 "--cwd", str(self.cwd)],
                capture_output=True, text=True, encoding="utf-8", timeout=20)
            elapsed = time.monotonic() - started
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn('"registered"', out.stdout)
            self.assertGreaterEqual(elapsed, 0.5)
        finally:
            holder.communicate(timeout=20)

    def test_unknown_owner_not_released_is_kept(self):
        self.m.register("claude", "synthetic-session", str(self.cwd))
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")

    def test_alive_owner_is_kept_even_after_release(self):
        self.m.register("claude", "synthetic-session", str(self.cwd), owner=os.getpid())
        self.m.release("claude", "synthetic-session")
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")

    def test_existing_excluded_copy_is_not_adopted(self):
        self.config["repos"][0]["excluded"] = [str(self.cwd)]
        self.assertEqual(self.m.register("claude", "other-session", str(self.cwd))["status"], "ignored")
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")

    def test_main_cannot_be_registered(self):
        self.assertEqual(self.m.register("claude", "main", str(self.repo))["status"], "ignored")

    def test_changed_branch_is_kept(self):
        aw.git(self.cwd, "switch", "-c", "other")
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")

    def test_network_failure_is_not_a_merge(self):
        self.m.merged_pr.side_effect = RuntimeError("command failed")
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")
        self.assertTrue(self.cwd.exists())

    def test_registry_failure_does_not_delete(self):
        self.path.write_text("invalid", encoding="utf-8")
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")
        self.assertTrue(self.cwd.exists())

    def test_other_registered_owner_blocks_removal(self):
        self.m.register("claude", "other", str(self.cwd), owner=os.getpid())
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")
        self.assertTrue(self.cwd.exists())

    def test_sandbox_symlink_is_kept_without_traversal(self):
        sandbox = self.m.sandbox_path(self.cwd)
        link = sandbox / "temp" / "linked"
        if os.name == "nt":
            p = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(self.cwd)],
                               capture_output=True)
            if p.returncode:
                self.skipTest("junction unavailable")
        else:
            link.symlink_to(self.cwd, target_is_directory=True)
        try:
            with self.assertRaisesRegex(ValueError, "linked sandbox"):
                self.m.cleanup_sandbox(self.cwd)
            self.assertTrue((self.cwd / "file.txt").exists())
        finally:
            if os.name == "nt":
                link.rmdir()
            else:
                link.unlink()

    def test_task_cannot_silently_switch_to_other_copy(self):
        other = self.repo / ".claude/worktrees/other"
        self.g("worktree", "add", "-b", "feature/other", str(other), "HEAD")
        try:
            with self.assertRaises(ValueError):
                self.m.register("claude", "synthetic-session", str(other))
        finally:
            self.g("worktree", "remove", str(other))

    def test_recheck_after_network_calls(self):
        def race(*args):
            (self.cwd / "file.txt").write_text("raced", encoding="utf-8")
            return {"number": 1}
        self.m.merged_pr.side_effect = race
        self.assertEqual(self.m.sweep(True)[0]["status"], "kept-unverified")
        self.assertTrue(self.cwd.exists())

    def test_junction_or_symlink_is_rejected(self):
        link = self.repo / ".claude/worktrees/link"
        if os.name == "nt":
            p = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(self.cwd)],
                               capture_output=True)
            if p.returncode:
                self.skipTest("junction unavailable")
        else:
            link.symlink_to(self.cwd, target_is_directory=True)
        try:
            self.assertFalse(aw.safe_checkout(self.repo, link))
            self.assertEqual(self.m.register("claude", "linked", str(link))["status"], "ignored")
        finally:
            if os.name == "nt":
                link.rmdir()
            else:
                link.unlink()

    def test_merged_query_validates_exact_head_and_remote(self):
        self.pr.stop()
        real_command = aw.command
        pr = [{"number": 1, "headRefOid": "0" * 40, "isCrossRepository": False,
               "baseRefName": "main", "mergeCommit": {"oid": self.base}}]
        def fake(args, *a, **kw):
            if args[0] == "gh":
                return json.dumps(pr if "merged" in args else [])
            return real_command(args, *a, **kw)
        with patch.object(aw, "command", side_effect=fake):
            with self.assertRaisesRegex(ValueError, "no matching merged"):
                self.m.merged_pr(self.config["repos"][0], "feature/test", self.head)

    def test_opencode_plugin_create_resume_and_guards_without_model(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is needed only for plugin integration tests")
        bare = self.root / "remote.git"
        aw.command(["git", "init", "--bare", "-q", str(bare)])
        self.g("remote", "add", "origin", str(bare))
        self.g("push", "-u", "origin", "main")
        config_file = self.root / "config.json"
        config_file.write_text(json.dumps(self.config), encoding="utf-8")
        project = Path(__file__).resolve().parent.parent
        created = self.repo / ".opencode/worktrees/task-ses-synthetic"
        try:
            out = aw.command([node, project / "tests/workspace_plugin_probe.mjs",
                              project / "tools/opencode-workspaces/index.js", config_file,
                              sys.executable, project / "tools/agent_workspaces.py"], timeout=60)
            self.assertIn("passed", out)
        finally:
            if created.exists():
                self.g("worktree", "remove", str(created))

    def test_merge_lookup_rejects_new_remote_commits(self):
        self.pr.stop()
        pr = [{"number": 1, "headRefOid": self.head, "isCrossRepository": False,
               "baseRefName": "main", "mergeCommit": {"oid": self.base}}]
        def fake_command(args, *a, **kw):
            if args[0] == "gh":
                return json.dumps(pr if "merged" in args else [])
            raise AssertionError("unexpected external command")
        def fake_git(repo, *args):
            if args[0] == "ls-remote":
                return "f" * 40 + "\trefs/heads/feature/test"
            return ""
        with patch.object(aw, "command", side_effect=fake_command), patch.object(aw, "git", side_effect=fake_git):
            with self.assertRaisesRegex(ValueError, "remote branch changed"):
                self.m.merged_pr(self.config["repos"][0], "feature/test", self.head)

    def test_open_pr_prevents_removal_even_after_previous_merge(self):
        self.pr.stop()
        pr = [{"number": 1, "headRefOid": self.head, "isCrossRepository": False,
               "baseRefName": "main", "mergeCommit": {"oid": self.base}}]
        with patch.object(aw, "command", side_effect=lambda args: json.dumps(
                pr if "merged" in args else [{"number": 2}])):
            with self.assertRaisesRegex(ValueError, "open PR"):
                self.m.merged_pr(self.config["repos"][0], "feature/test", self.head)

    def test_opencode_busy_or_recent_sessions_are_kept(self):
        data = {"cwd": str(self.cwd), "session": "ses_busy"}
        s = {"id": "ses_busy", "location": {"directory": str(self.cwd)},
             "time": {"updated": 1, "busy": 2, "idle": 1}}
        with patch.object(self.m, "api", side_effect=lambda op, **kw: (
                {"data": [s], "cursor": {}} if op == "session.list" else {"data": s})):
            with self.assertRaisesRegex(ValueError, "active session"):
                self.m.opencode_sessions(data)

    def test_unknown_sandbox_is_not_adopted(self):
        sandbox = self.m.sandbox_path(self.cwd)
        (sandbox / "workspace-sandbox.marker").unlink()
        with self.assertRaisesRegex(ValueError, "sandbox not owned"):
            self.m.prepare_sandbox(self.cwd)

    def test_opencode_apply_moves_archives_evicts_then_removes(self):
        old = json.loads(self.path.read_text(encoding="utf-8"))
        old["removed"] = True
        aw.write_json(self.path, old)
        self.m.register("opencode", "ses_done", str(self.cwd), project="synthetic")
        record = self.m.record_path("opencode", "ses_done")
        session = {"id": "ses_done", "location": {"directory": str(self.cwd)},
                   "time": {"updated": 1, "busy": 0, "idle": 1}}
        events = []
        real_command = aw.command
        def fake_command(args, *a, **kw):
            if args[0] == "synthetic-opencode":
                self.assertEqual(args[1:3], ["api", "delete"])
                events.append("evicted")
                return ""
            return real_command(args, *a, **kw)
        def api(op, body=None, **kw):
            if op == "session.list":
                return {"data": [session], "cursor": {}}
            if op == "session.get":
                return {"data": session}
            if op == "session.move":
                events.append("moved")
                session["location"]["directory"] = body["directory"]
            elif op == "session.update":
                self.assertTrue(body["archived"])
                events.append("archived")
            elif op == "worktree.remove":
                self.assertEqual(events, ["moved", "archived", "evicted"])
                self.g("worktree", "remove", str(self.cwd))
                events.append("removed")
            else:
                raise AssertionError(op)
        with patch.object(self.m, "api", side_effect=api), patch.object(
                aw, "command", side_effect=fake_command):
            self.assertEqual(self.m.sweep_one(record, True), "removed")
        self.assertEqual(events[-1], "removed")
        self.assertFalse(self.cwd.exists())

    def test_explicitly_finished_empty_task_needs_no_pr(self):
        self.g("update-ref", "refs/remotes/origin/main", self.base)
        empty = self.repo / ".claude/worktrees/empty"
        self.g("worktree", "add", "-b", "feature/empty", str(empty), "main")
        try:
            self.m.register("claude", "empty", str(empty))
            self.m.release("claude", "empty")
            record = self.m.record_path("claude", "empty")
            self.assertEqual(self.m.sweep_one(record, True), "removed")
            self.m.merged_pr.assert_not_called()
            self.assertFalse(empty.exists())
        finally:
            if empty.exists():
                self.g("worktree", "remove", str(empty))


if __name__ == "__main__":
    unittest.main()
