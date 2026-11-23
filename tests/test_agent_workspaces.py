"""Housekeeping is exercised only on disposable Git repositories, never on user worktrees."""
import copy
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


def remove_tree(path):
    """Delete a disposable tree on Windows too: Git object files are read-only there."""
    for parent, _, files in os.walk(path):
        for name in files:
            try:
                (Path(parent) / name).chmod(0o666)
            except OSError:
                pass
    shutil.rmtree(path)


class HousekeepingTests(unittest.TestCase):
    """Every test works on its own copy of one synthetic repository sample.

    The sample (repository, task checkout, registry) is built once per class with real Git. Each
    test then gets a fresh copy at the same path, so absolute paths recorded inside Git and in the
    registry stay valid, and the copy is deleted after the test: nothing is shared between tests.
    """

    @classmethod
    def setUpClass(cls):
        holder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(holder.cleanup)
        cls.root = Path(holder.name) / "work"
        cls.sample = Path(holder.name) / "sample"
        cls.addClassCleanup(lambda: cls.sample.exists() and remove_tree(cls.sample))
        cls.root.mkdir()
        repo = cls.root / "repo"
        repo.mkdir()

        def g(*args):
            return aw.git(repo, *args)
        g("init", "-q", "-b", "main")
        g("config", "user.name", "Synthetic")
        g("config", "user.email", "synthetic@example.invalid")
        (repo / ".gitignore").write_text(".claude/worktrees/\n.opencode/worktrees/\n*.scratch\n",
                                         encoding="utf-8")
        (repo / "file.txt").write_text("base\n", encoding="utf-8")
        g("add", ".")
        g("commit", "-qm", "base")
        cls.base = g("rev-parse", "HEAD")
        cwd = repo / ".claude/worktrees/task"
        g("worktree", "add", "-b", "feature/test", str(cwd), "HEAD")
        (cwd / "file.txt").write_text("task\n", encoding="utf-8")
        aw.git(cwd, "add", ".")
        aw.git(cwd, "commit", "-qm", "task")
        cls.head = aw.git(cwd, "rev-parse", "HEAD")
        cls.sample_config = {"state": str(cls.root / "registry"), "opencode": "synthetic-opencode",
                             "repos": [{"root": str(repo), "github": "synthetic/example",
                                        "excluded": []}]}
        m = aw.Manager(cls.sample_config)
        m.register("claude", "synthetic-session", str(cwd))
        m.release("claude", "synthetic-session")
        shutil.move(str(cls.root), str(cls.sample))

    def setUp(self):
        shutil.copytree(self.sample, self.root, symlinks=True)
        self.addCleanup(remove_tree, self.root)
        self.repo = self.root / "repo"
        self.cwd = self.repo / ".claude/worktrees/task"
        self.config = copy.deepcopy(self.sample_config)
        self.m = aw.Manager(self.config)
        self.path = self.m.record_path("claude", "synthetic-session")
        # A verified merged PR is injected for deletion tests, never a real API.
        self.pr = patch.object(self.m, "merged_pr", return_value={"number": 1})
        self.pr.start()
        self.addCleanup(self.pr.stop)

    def g(self, *args):
        return aw.git(self.repo, *args)

    def test_dry_run_preserves_checkout_and_branch(self):
        self.assertEqual(self.m.sweep_one(self.path), "would-remove")
        self.assertTrue(self.cwd.exists())
        self.assertEqual(self.g("rev-parse", "feature/test"), self.head)

    def test_report_appends_modes_and_retains_only_recent_owned_entries(self):
        report = self.root / "reports" / "daily.log"
        now = time.time()
        results = self.m.sweep()
        aw.write_report(report, False, results, now - 31 * 86400)
        with report.open("a", encoding="utf-8") as f:
            f.write("unrecognized line kept\n")
        aw.write_report(report, False, results, now)
        aw.write_report(report, True, results, now + 1)
        lines = report.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "unrecognized line kept")
        self.assertEqual(len(lines), 3)
        entries = [json.loads(line[len(aw.REPORT_PREFIX):]) for line in lines[1:]]
        self.assertEqual([entry["mode"] for entry in entries], ["dry-run", "apply"])
        task = entries[0]["tasks"][0]
        self.assertEqual(task["branch"], "feature/test")
        self.assertEqual(task["worktree"], str(self.cwd))
        self.assertEqual(task["status"], "would-remove")
        self.assertNotIn("file.txt", "\n".join(lines))

    def test_report_hides_untrusted_reason(self):
        report = self.root / "reports" / "daily.log"
        aw.write_report(report, False, [{"branch": "feature/test", "cwd": str(self.cwd),
                                        "status": "kept-unverified",
                                        "reason": "secret-file-name and contents"}])
        text = report.read_text(encoding="utf-8")
        self.assertNotIn("secret-file-name", text)
        self.assertIn("verification failed", text)

    def test_invalid_task_record_is_reported_and_preserved(self):
        self.path.write_text("[]", encoding="utf-8")
        result = self.m.sweep()
        report = self.root / "reports" / "daily.log"
        aw.write_report(report, False, result)
        self.assertIn("invalid task record", report.read_text(encoding="utf-8"))
        self.assertTrue(self.cwd.exists())

    def test_sweep_cli_writes_report(self):
        config_file = self.root / "config.json"
        config_file.write_text(json.dumps(self.config), encoding="utf-8")
        report = self.root / "reports" / "daily.log"
        # No network calls: this checkout has local uncommitted changes.
        (self.cwd / "private-name.txt").write_text("private-content", encoding="utf-8")
        out = subprocess.run([sys.executable, "-B", str(Path(aw.__file__)), "--config",
                              str(config_file), "sweep", "--report", str(report)],
                             capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        text = report.read_text(encoding="utf-8")
        self.assertIn("unsaved or ignored files", text)
        self.assertNotIn("private-name.txt", text)
        self.assertNotIn("private-content", text)

    def row(self, rows, path):
        return next(r for r in rows if Path(r["worktree"]).exists() and
                    os.path.samefile(r["worktree"], path))

    def test_inventory_lists_main_clone_and_task_without_changing_anything(self):
        before_state = sorted((p.name, p.stat().st_mtime_ns) for p in self.m.state.glob("*.json"))
        before_branches = self.g("branch", "--list")
        with patch.object(aw, "pr_state", return_value="merged"), \
                patch.object(aw.Manager, "sweep_one", side_effect=AssertionError("unexpected sweep")):
            rows = self.m.inventory()
        main = self.row(rows, self.repo)
        task = self.row(rows, self.cwd)
        self.assertEqual((main["kind"], main["branch"], main["changes"], main["hint"]),
                         ("main-clone", "main", 0, "ok"))
        self.assertEqual((task["kind"], task["branch"], task["registration"], task["pr"], task["hint"]),
                         ("managed-root", "feature/test", "registered", "merged", "review-removal"))
        self.assertEqual(task["head"], self.head[:12])
        self.assertTrue(self.cwd.exists())
        self.assertEqual(self.g("branch", "--list"), before_branches)
        self.assertEqual(sorted((p.name, p.stat().st_mtime_ns) for p in self.m.state.glob("*.json")),
                         before_state)

    def test_inventory_keeps_dirty_locked_and_unknown(self):
        (self.cwd / "extra.txt").write_text("x", encoding="utf-8")
        with patch.object(aw, "pr_state", return_value="merged"):
            self.assertEqual(self.row(self.m.inventory(), self.cwd)["hint"], "keep")
            self.assertEqual(self.row(self.m.inventory(), self.cwd)["changes"], 1)
        (self.cwd / "extra.txt").unlink()
        self.g("worktree", "lock", str(self.cwd))
        self.addCleanup(lambda: subprocess.run(["git", "-C", str(self.repo), "worktree", "unlock",
                                                str(self.cwd)], capture_output=True))
        with patch.object(aw, "pr_state", return_value="merged"):
            row = self.row(self.m.inventory(), self.cwd)
        self.assertEqual((row["locked"], row["hint"]), (True, "keep"))
        self.g("worktree", "unlock", str(self.cwd))
        with patch.object(aw, "pr_state", side_effect=RuntimeError("command failed")):
            row = self.row(self.m.inventory(), self.cwd)
        self.assertEqual((row["pr"], row["hint"]), ("unknown", "keep"))
        self.assertEqual(self.row(self.m.inventory(github=False), self.cwd)["pr"], "-")

    def test_inventory_flags_main_clone_off_main_and_missing_checkout(self):
        self.g("switch", "-q", "-c", "stray")
        self.assertEqual(self.row(self.m.inventory(github=False), self.repo)["hint"], "check")
        shutil.rmtree(self.cwd, ignore_errors=True)
        rows = self.m.inventory(github=False)
        gone = next(r for r in rows if r["branch"] == "feature/test")
        self.assertEqual((gone["missing"], gone["changes"], gone["hint"]), (True, None, "check"))

    def test_inventory_shows_excluded_unregistered_and_outside_roots(self):
        other = self.repo / ".claude/worktrees/other"
        self.g("worktree", "add", "-q", "-b", "feature/other", str(other), "HEAD")
        outside = self.root / "elsewhere"
        self.g("worktree", "add", "-q", "-b", "feature/outside", str(outside), "HEAD")
        self.addCleanup(lambda: [subprocess.run(["git", "-C", str(self.repo), "worktree", "remove",
                                                 "--force", str(p)], capture_output=True)
                                 for p in (other, outside)])
        rows = self.m.inventory(github=False)
        self.assertEqual(self.row(rows, other)["registration"], "unregistered")
        self.assertEqual(self.row(rows, outside)["kind"], "outside-roots")
        self.config["repos"][0]["excluded"] = [str(other)]
        rows = aw.Manager(self.config).inventory(github=False)
        self.assertEqual(self.row(rows, other)["registration"], "excluded")

    def test_pr_state_matches_exact_head_only(self):
        repo = {"github": "synthetic/example"}

        def listing(rows):
            return patch.object(aw, "command", return_value=json.dumps(rows))
        merged = {"state": "MERGED", "headRefOid": "a" * 40, "isCrossRepository": False,
                  "baseRefName": "main"}
        with listing([]):
            self.assertEqual(aw.pr_state(repo, "b", "a" * 40), "none")
        with listing([merged]):
            self.assertEqual(aw.pr_state(repo, "b", "a" * 40), "merged")
            self.assertEqual(aw.pr_state(repo, "b", "c" * 40), "head-differs")
        with listing([merged, dict(merged, state="OPEN")]):
            self.assertEqual(aw.pr_state(repo, "b", "a" * 40), "open")
        with listing([dict(merged, state="CLOSED")]):
            self.assertEqual(aw.pr_state(repo, "b", "a" * 40), "closed")
        with listing([dict(merged, isCrossRepository=True)]):
            self.assertEqual(aw.pr_state(repo, "b", "a" * 40), "none")

    def test_inventory_report_shares_file_with_sweep_report(self):
        report = self.root / "reports" / "daily.log"
        now = time.time()
        rows = self.m.inventory(github=False)
        aw.write_inventory_report(report, rows, now - 31 * 86400)
        with report.open("a", encoding="utf-8") as f:
            f.write("unrecognized line kept\n")
        aw.write_report(report, False, self.m.sweep(), now - 31 * 86400)
        aw.write_inventory_report(report, rows, now - 86400)
        aw.write_report(report, False, self.m.sweep(), now)
        lines = report.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "unrecognized line kept")
        self.assertEqual([line.split(" ", 1)[0] for line in lines[1:]],
                         ["AgentWorkspaceInventory", "AgentWorkspaceReport"])
        entry = json.loads(lines[1][len(aw.INVENTORY_PREFIX):])
        self.assertEqual(entry["counts"], {"ok": 1, "keep": 1})
        self.assertNotIn("file.txt", "\n".join(lines))

    def test_inventory_cli_is_read_only_and_reports(self):
        config_file = self.root / "config.json"
        config_file.write_text(json.dumps(self.config), encoding="utf-8")
        report = self.root / "reports" / "inventory.log"
        (self.cwd / "private-name.txt").write_text("private-content", encoding="utf-8")
        out = subprocess.run([sys.executable, "-B", str(Path(aw.__file__)), "--config",
                              str(config_file), "inventory", "--no-github", "--report", str(report)],
                             capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        rows = json.loads(out.stdout)
        self.assertEqual(self.row(rows, self.cwd)["changes"], 1)
        text = report.read_text(encoding="utf-8")
        self.assertIn("feature/test", text)
        self.assertNotIn("private-name.txt", text)
        self.assertNotIn("private-content", text)
        self.assertTrue(self.cwd.exists())

    def test_claude_hooks_never_sweep(self):
        from io import StringIO
        config_file = self.root / "config.json"
        self.config["apply_on_claude_event"] = True
        config_file.write_text(json.dumps(self.config), encoding="utf-8")
        event = {"hook_event_name": "SessionEnd", "session_id": "synthetic-session"}
        args = ["agent_workspaces.py", "--config", str(config_file), "claude-hook"]
        with patch.object(sys, "argv", args), patch.object(sys, "stdin", StringIO(json.dumps(event))), \
                patch.object(aw.Manager, "sweep", side_effect=AssertionError("unexpected sweep")):
            self.assertEqual(aw.main(), 0)

    def test_verified_squash_removes_only_registered_checkout(self):
        # Mark main as squash-merged; original feature commit is NOT its ancestor.
        self.g("merge", "--squash", "feature/test")
        self.g("commit", "-qm", "squash")
        self.assertEqual(self.m.sweep_one(self.path, True), "removed")
        self.assertFalse(self.cwd.exists())
        self.assertTrue(self.repo.exists())
        self.assertFalse(self.m.sandbox_path(self.cwd).exists())
        self.assertEqual(self.m.sweep_one(self.path, True), "already-removed")

    def test_branch_delete_failure_records_removed_checkout(self):
        self.g("merge", "--squash", "feature/test")
        self.g("commit", "-qm", "squash")
        real_git = aw.git
        def fail_branch_delete(repo, *args):
            if args[:2] == ("branch", "-D"):
                raise RuntimeError("synthetic failure")
            return real_git(repo, *args)
        with patch.object(aw, "git", side_effect=fail_branch_delete):
            self.assertEqual(self.m.sweep_one(self.path, True), "removed-branch-kept")
        self.assertFalse(self.cwd.exists())
        self.assertTrue(json.loads(self.path.read_text(encoding="utf-8"))["removed"])
        self.assertEqual(self.m.sweep_one(self.path), "already-removed")
        self.assertEqual(self.g("rev-parse", "feature/test"), self.head)

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

    def moving_session(self):
        """An OpenCode session bound to the task checkout, plus a fresh checkout from origin/main."""
        self.g("update-ref", "refs/remotes/origin/main", self.base)
        self.m.register("opencode", "oc-session", str(self.cwd), project="synthetic-project")
        nxt = self.repo / ".opencode/worktrees/next"
        self.g("worktree", "add", "-q", "-b", "opencode/next", str(nxt), "origin/main")
        return self.m.record_path("opencode", "oc-session"), nxt

    def records(self):
        return [json.loads(p.read_text(encoding="utf-8")) for p in self.m.state.glob("*.json")]

    def test_moved_session_keeps_old_binding_and_registers_new(self):
        path, nxt = self.moving_session()
        (self.cwd / "unsaved.txt").write_text("old work", encoding="utf-8")
        self.assertEqual(self.m.register("opencode", "oc-session", str(nxt),
                                         project="synthetic-project")["status"], "registered")
        current = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((current["cwd"], current["branch"], current["base_head"]),
                         (str(nxt.resolve()), "opencode/next", self.base))
        moved = self.m.moved_path(path, str(self.cwd))
        kept = json.loads(moved.read_text(encoding="utf-8"))
        self.assertEqual((kept["cwd"], kept["branch"]), (str(self.cwd), "feature/test"))
        # The old checkout is still judged by its own record and keeps its unsaved work.
        with self.assertRaisesRegex(ValueError, "unsaved or ignored files"):
            self.m.sweep_one(moved, True)
        self.assertTrue((self.cwd / "unsaved.txt").exists())

    def test_claude_session_never_moves_to_another_checkout(self):
        _, nxt = self.moving_session()
        with self.assertRaisesRegex(ValueError, "cannot be silently replaced"):
            self.m.register("claude", "synthetic-session", str(nxt))
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["cwd"], str(self.cwd))

    def test_move_into_checkout_held_by_other_session_is_refused(self):
        path, nxt = self.moving_session()
        self.m.register("opencode", "other", str(nxt), project="synthetic-project")
        with self.assertRaisesRegex(ValueError, "checkout bound to another session"):
            self.m.register("opencode", "oc-session", str(nxt), project="synthetic-project")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["cwd"], str(self.cwd))
        self.assertFalse(self.m.moved_path(path, str(self.cwd)).exists())

    def test_move_into_dirty_or_diverged_checkout_is_refused(self):
        path, nxt = self.moving_session()
        (nxt / "someone.txt").write_text("work", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "new checkout is not clean"):
            self.m.register("opencode", "oc-session", str(nxt), project="synthetic-project")
        aw.git(nxt, "add", ".")
        aw.git(nxt, "commit", "-qm", "someone")
        with self.assertRaisesRegex(ValueError, "new checkout has own commits"):
            self.m.register("opencode", "oc-session", str(nxt), project="synthetic-project")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["cwd"], str(self.cwd))

    def test_move_back_reuses_own_checkout_with_one_binding_each(self):
        _, nxt = self.moving_session()
        first = self.repo / ".opencode/worktrees/first"
        self.g("worktree", "add", "-q", "-b", "opencode/first", str(first), "origin/main")
        (first / "file.txt").write_text("own work\n", encoding="utf-8")
        aw.git(first, "add", ".")
        aw.git(first, "commit", "-qm", "own work")
        path = self.m.record_path("opencode", "second-session")
        self.m.register("opencode", "second-session", str(first), project="synthetic-project")
        self.m.register("opencode", "second-session", str(nxt), project="synthetic-project")
        # Back to its own earlier checkout, although that one has its own commits.
        self.m.register("opencode", "second-session", str(first), project="synthetic-project")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["cwd"], str(first.resolve()))
        cwds = sorted(os.path.normcase(r["cwd"]) for r in self.records()
                      if r["session"] == "second-session")
        self.assertEqual(cwds, sorted(os.path.normcase(str(p.resolve())) for p in (first, nxt)))

    def test_removed_binding_is_not_silently_reused(self):
        record = json.loads(self.path.read_text(encoding="utf-8"))
        record["removed"] = True
        self.path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cannot be silently replaced"):
            self.m.register("claude", "synthetic-session", str(self.cwd))

    def test_refusal_names_fixed_reason_without_paths(self):
        _, nxt = self.moving_session()
        self.m.register("opencode", "other", str(nxt), project="synthetic-project")
        config_file = self.root / "config.json"
        config_file.write_text(json.dumps(self.config), encoding="utf-8")
        out = subprocess.run([sys.executable, "-B", str(Path(aw.__file__)), "--config",
                              str(config_file), "register", "--tool", "opencode",
                              "--session", "oc-session", "--cwd", str(nxt)],
                             capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(out.returncode, 1)
        self.assertIn("deferred (checkout bound to another session)", out.stderr)
        self.assertNotIn(str(self.root), out.stderr)
        self.assertEqual(aw.reason_code(OSError(str(self.root))), "error")
        self.assertEqual(aw.reason_code(ValueError("Expecting value: line 1 column 1")), "error")
        self.assertEqual(aw.reason_code(RuntimeError("command failed")), "command failed")

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
        # Simulate a newer remote-tracking main while the integration checkout
        # remains behind: the plugin must use the remote ref, not local HEAD.
        self.g("switch", "-q", "-c", "synthetic/new-base")
        (self.repo / "file.txt").write_text("new remote base\n", encoding="utf-8")
        self.g("commit", "-qam", "new base")
        self.g("push", "origin", "synthetic/new-base:main")
        self.g("fetch", "origin")
        self.g("switch", "-q", "main")
        self.assertNotEqual(self.g("rev-parse", "HEAD"), self.g("rev-parse", "origin/main"))
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
        (sandbox / "temp" / "someone.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sandbox not owned"):
            self.m.prepare_sandbox(self.cwd)
        (sandbox / "temp" / "someone.txt").unlink()
        (sandbox / "other").mkdir()
        with self.assertRaisesRegex(ValueError, "sandbox not owned"):
            self.m.prepare_sandbox(self.cwd)

    def test_empty_sandbox_made_by_shell_hook_is_taken_over(self):
        # The plugin's shell hook can create empty data/temp before registration succeeds;
        # cleanup of such a folder can lose nothing, so registration marks it as its own.
        sandbox = self.m.sandbox_path(self.cwd)
        (sandbox / "workspace-sandbox.marker").unlink()
        self.assertEqual(self.m.prepare_sandbox(self.cwd), sandbox)
        self.assertEqual((sandbox / "workspace-sandbox.marker").read_text(encoding="ascii"),
                         sandbox.name + "\n")

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


def setUpModule():
    """An inherited GIT_DIR (hooks, rebase --exec) overrides git -C: never write to that repository."""
    environment = patch.dict(os.environ)
    environment.start()
    unittest.addModuleCleanup(environment.stop)
    for name in [n for n in os.environ if n.upper().startswith("GIT_")]:
        del os.environ[name]


GUARDED_CHILD = '''
import importlib.util, subprocess, sys
project, module, work = sys.argv[1:]
sys.path.insert(0, project)
spec = importlib.util.spec_from_file_location("guarded", module)
guarded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guarded)
guarded.setUpModule()
for args in (["init", "-q"], ["config", "user.name", "Leaked"],
             ["config", "user.email", "leaked@example.invalid"],
             ["commit", "-q", "--allow-empty", "-m", "test"]):
    subprocess.run(["git", "-C", work, *args], check=True)
'''


class InheritedGitEnvironmentTests(unittest.TestCase):
    def test_tests_that_write_git_never_touch_an_inherited_repository(self):
        project = Path(__file__).resolve().parent.parent
        for name in ("test_agent_workspaces", "test_buildinfo", "test_install_update"):
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                victim, work = Path(tmp) / "victim", Path(tmp) / "work"
                victim.mkdir()
                work.mkdir()
                aw.git(victim, "init", "-q")
                aw.git(victim, "-c", "user.name=Victim", "-c", "user.email=victim@example.invalid",
                       "commit", "-q", "--allow-empty", "-m", "base")
                head = aw.git(victim, "rev-parse", "HEAD")
                config = (victim / ".git/config").read_bytes()
                environment = dict(os.environ, GIT_DIR=str(victim / ".git"),
                                   GIT_INDEX_FILE=str(victim / ".git/index"))
                done = subprocess.run(
                    [sys.executable, "-B", "-c", GUARDED_CHILD, str(project),
                     str(project / "tests" / f"{name}.py"), str(work)],
                    env=environment, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=60)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual(aw.git(victim, "rev-parse", "HEAD"), head)
                self.assertEqual((victim / ".git/config").read_bytes(), config)


if __name__ == "__main__":
    unittest.main()
