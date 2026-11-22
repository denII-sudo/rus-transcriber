"""Local, opt-in task registry and fail-closed post-merge worktree housekeeping.

No model calls. The configuration and registry live OUTSIDE the repository.
GitHub workflows and native Claude cleanup remain responsible for their own scope.
"""
import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from urllib.parse import quote


def command(args, cwd=None, timeout=30):
    p = subprocess.run([str(x) for x in args], cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if p.returncode:
        raise RuntimeError("command failed")  # Never log command bodies, paths or credentials.
    return p.stdout.strip()


def git(repo, *args):
    return command(["git", "--no-optional-locks", "-c", "core.quotePath=false", "-C", repo, *args])


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def registry_lock(state, timeout=20):
    state.mkdir(parents=True, exist_ok=True)
    with (state / "registry.lock").open("a+b") as f:
        # Locking a byte beyond EOF is supported; reading or writing it before
        # locking races with another process on Windows.
        f.seek(0)
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def linked(path):
    try:
        s = path.lstat()
        return path.is_symlink() or bool(getattr(s, "st_file_attributes", 0) & 0x400)
    except FileNotFoundError:
        return False


def safe_checkout(repo, cwd):
    """Only direct children of the two approved roots; reject junctions on the path."""
    repo, raw = Path(repo).resolve(), Path(cwd).absolute()
    if any(c in str(raw) for c in "\r\n\0"):
        return False
    # Inspect the path as written: resolving first would hide a junction and let it
    # masquerade as the directory it points to.
    for p in (raw, *raw.parents):
        if linked(p):
            return False
    cwd = raw.resolve()
    allowed = {repo / ".claude/worktrees", repo / ".opencode/worktrees"}
    parent = os.path.normcase(str(cwd.parent))
    if parent not in {os.path.normcase(str(p)) for p in allowed}:
        return False
    return os.path.normcase(str(cwd)) != os.path.normcase(str(repo))


def checkout_info(repo, cwd, allow_locked=False):
    if not safe_checkout(repo, cwd):
        raise ValueError("outside allowed checkout roots")
    root = Path(git(cwd, "rev-parse", "--show-toplevel")).resolve()
    if root != Path(cwd).resolve():
        raise ValueError("not a checkout root")
    common = Path(git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")).resolve()
    expected = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")).resolve()
    if common != expected:
        raise ValueError("different repository")
    branch = git(cwd, "symbolic-ref", "--quiet", "--short", "HEAD")
    if branch in ("main", "master") or branch.startswith("-"):
        raise ValueError("protected branch")
    inventory = git(repo, "worktree", "list", "--porcelain")
    entries = [e for e in inventory.split("\n\n") if e.strip()]
    entry = next((e for e in entries if e.splitlines()[0].removeprefix("worktree ") ==
                  str(root).replace("\\", "/")), None)
    if entry is None:
        # Native POSIX paths do not need slash conversion.
        entry = next((e for e in entries if e.splitlines()[0] == "worktree " + str(root)), None)
    if entry is None or any(line.startswith("prunable") or
                            (line.startswith("locked") and not allow_locked)
                            for line in entry.splitlines()):
        raise ValueError("checkout locked or not registered")
    return {"branch": branch, "head": git(cwd, "rev-parse", "HEAD")}


def process_token(pid):
    """Windows process creation time prevents accidental PID-reuse matches."""
    if not pid:
        return None
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
            return "alive"
        except ProcessLookupError:
            return None
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k.OpenProcess.restype = wintypes.HANDLE
    k.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_void_p)
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    h = k.OpenProcess(0x1000, False, int(pid))
    if not h:
        if ctypes.get_last_error() == 87:  # Invalid PID: process gone.
            return None
        raise RuntimeError("process status unavailable")
    try:
        values = [ctypes.c_ulonglong() for _ in range(4)]
        if not k.GetProcessTimes(h, *(ctypes.byref(x) for x in values)):
            raise RuntimeError("process status unavailable")
        return str(values[0].value)
    finally:
        k.CloseHandle(h)


def claude_owner():
    """Find the real Claude ancestor, not the short-lived shell running a hook."""
    if os.name != "nt":
        return None
    script = ("$p=$PID; for($i=0;$i -lt 12;$i++){"
              "$x=Get-CimInstance Win32_Process -Filter \"ProcessId=$p\";"
              "if(-not $x){break}; if($x.Name -like 'claude*'){"
              "[Console]::Write($x.ProcessId);break};$p=$x.ParentProcessId}")
    out = command(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
    return int(out) if out.isdigit() else None


class Manager:
    def __init__(self, config):
        self.config = config
        self.state = Path(config["state"])
        if any(linked(p) for p in (self.state, *self.state.parents)):
            raise ValueError("linked registry")
        self.state.mkdir(parents=True, exist_ok=True)

    def record_path(self, tool, session):
        key = hashlib.sha256((tool + ":" + session).encode()).hexdigest()
        return self.state / (key + ".json")

    def repo_for(self, cwd):
        return next((r for r in self.config["repos"] if safe_checkout(r["root"], cwd)), None)

    def sandbox_path(self, cwd):
        key = hashlib.sha256(
            str(Path(cwd).resolve()).casefold().replace("\\", "/").encode()).hexdigest()[:12]
        return self.state / "sandboxes" / key

    def prepare_sandbox(self, cwd):
        p = self.sandbox_path(cwd)
        if any(linked(x) for x in (p, p.parent)):
            raise ValueError("linked sandbox")
        marker = p / "workspace-sandbox.marker"
        if p.exists() and not marker.is_file():
            raise ValueError("sandbox not owned")
        if marker.exists() and (linked(marker) or marker.read_text(encoding="ascii").strip() != p.name):
            raise ValueError("sandbox not owned")
        p.mkdir(parents=True, exist_ok=True)
        marker.write_text(p.name + "\n", encoding="ascii")
        for sub in ("data", "temp"):
            if linked(p / sub):
                raise ValueError("linked sandbox")
            (p / sub).mkdir(exist_ok=True)
        return p

    def cleanup_sandbox(self, cwd):
        p = self.sandbox_path(cwd)
        if not p.exists():
            return
        if any(linked(x) for x in (p, p.parent)):
            raise ValueError("linked sandbox")
        marker = p / "workspace-sandbox.marker"
        if linked(marker) or marker.read_text(encoding="ascii").strip() != p.name:
            raise ValueError("sandbox not owned")
        for root, dirs, files in os.walk(p, followlinks=False):
            if any(linked(Path(root) / n) for n in dirs + files):
                raise ValueError("linked sandbox")
        # Only this positively identified disposable task sandbox; never user data.
        shutil.rmtree(p)

    def api(self, operation, body=None, **params):
        args = [self.config["opencode"], "api", operation]
        for k, v in params.items():
            args += ["--param", f"{k}={v}"]
        if body is not None:
            args += ["--data", json.dumps(body)]
        raw = command(args)
        return json.loads(raw) if raw else None

    def register(self, tool, session, cwd, owner=None, project=None):
        raw = Path(cwd).absolute()
        repo = self.repo_for(raw)
        cwd = raw.resolve()
        if not repo or str(cwd).casefold() in {
                str(Path(p).absolute()).casefold() for p in repo.get("excluded", [])}:
            return {"status": "ignored"}
        path = self.record_path(tool, session)
        old = {}
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old["cwd"] != str(cwd) or old.get("removed"):
                raise ValueError("task binding cannot be silently replaced")
        info = checkout_info(repo["root"], cwd, allow_locked=True)
        self.prepare_sandbox(cwd)
        base = old.get("base_head")
        if not old:
            try:
                candidate = git(repo["root"], "rev-parse", "origin/main")
                base = candidate if candidate == info["head"] else None
            except RuntimeError:
                base = None
        token = process_token(owner) if owner else None
        data = {"schema": 1, "repo": repo["root"], "cwd": str(cwd), "tool": tool,
                "session": session, "project": project, "owner": owner, "owner_token": token,
                "branch": info["branch"], "base_head": base,
                "released": False, "updated": time.time()}
        write_json(path, data)
        return {"status": "registered"}

    def release(self, tool, session):
        p = self.record_path(tool, session)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            data.update(released=True, updated=time.time())
            write_json(p, data)
        return {"status": "released"}

    def opencode_sessions(self, data):
        page = self.api("session.list", directory=data["cwd"], limit=100)
        if (page.get("cursor") or {}).get("next"):
            raise RuntimeError("too many sessions")
        sessions = page["data"]
        own = self.api("session.get", sessionID=data["session"])["data"]
        if own["location"]["directory"] == data["cwd"] and all(s["id"] != own["id"] for s in sessions):
            sessions.append(own)
        for s in sessions:
            t = s["time"]
            if t.get("busy", 0) > t.get("idle", 0):
                raise ValueError("active session")
            if not t.get("archived") and time.time() - t["updated"] / 1000 < 120:
                raise ValueError("recent session")
        return sessions

    def merged_pr(self, repo, branch, head):
        raw = command(["gh", "pr", "list", "--repo", repo["github"], "--head", branch,
                       "--state", "merged", "--limit", "100",
                       "--json", "number,headRefOid,isCrossRepository,baseRefName,mergeCommit"])
        matches = [p for p in json.loads(raw) if p["headRefOid"] == head
                   and not p["isCrossRepository"] and p["baseRefName"] == "main"
                   and p.get("mergeCommit", {}).get("oid")]
        if not matches:
            raise ValueError("no matching merged PR")
        opened = json.loads(command(["gh", "pr", "list", "--repo", repo["github"],
                                     "--head", branch, "--state", "open", "--json", "number"]))
        if opened:
            raise ValueError("open PR")
        # A deleted remote is not evidence. Fetch and verify the merge commit instead.
        git(repo["root"], "fetch", "origin")
        pr = matches[0]
        git(repo["root"], "merge-base", "--is-ancestor", pr["mergeCommit"]["oid"], "origin/main")
        remote = git(repo["root"], "ls-remote", "--heads", "origin", "refs/heads/" + branch)
        if remote and remote.split()[0] != head:
            raise ValueError("remote branch changed")
        return pr

    def sweep_one(self, path, apply=False):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("removed"):
            if apply:
                self.cleanup_sandbox(data["cwd"])
            return "already-removed"
        repo = next((r for r in self.config["repos"]
                     if os.path.normcase(str(Path(r["root"]).resolve())) ==
                     os.path.normcase(str(Path(data["repo"]).resolve()))), None)
        if not repo or str(Path(data["cwd"]).resolve()).casefold() in {
                str(Path(p).absolute()).casefold() for p in repo.get("excluded", [])}:
            raise ValueError("unmanaged checkout")
        cwd = Path(data["cwd"])
        if not cwd.exists():
            return "missing-kept"  # Do not prune metadata or delete a branch on missing paths.
        info = checkout_info(repo["root"], cwd)
        if info["branch"] != data["branch"]:
            raise ValueError("branch changed")
        if git(cwd, "status", "--porcelain", "--untracked-files=all", "--ignored"):
            raise ValueError("unsaved or ignored files")
        sessions = []
        if data["tool"] == "claude":
            if data.get("owner"):
                if process_token(data["owner"]) == data.get("owner_token"):
                    raise ValueError("active Claude process")
            elif not data.get("released"):
                raise ValueError("owner unknown")
        elif data["tool"] == "opencode":
            if not data.get("project"):
                raise ValueError("project unknown")
            sessions = self.opencode_sessions(data)
        else:
            raise ValueError("owner unknown")
        own_archived = False
        if data["tool"] == "opencode":
            own = self.api("session.get", sessionID=data["session"])["data"]
            own_archived = bool(own["time"].get("archived"))
        unchanged_finished = (info["head"] == data.get("base_head")
                              and (data.get("released") or own_archived))
        pr = None if unchanged_finished else self.merged_pr(repo, info["branch"], info["head"])
        if not apply:
            return "would-remove"
        # Recheck after network calls. Never remove while another registered owner is active.
        for other_path in self.state.glob("*.json"):
            if other_path == path:
                continue
            other = json.loads(other_path.read_text(encoding="utf-8"))
            if other.get("cwd") == data["cwd"] and not other.get("removed"):
                raise ValueError("another registered owner")
        if checkout_info(repo["root"], cwd) != info or git(
                cwd, "status", "--porcelain", "--untracked-files=all", "--ignored"):
            raise ValueError("checkout changed during verification")
        if data["tool"] == "opencode":
            self.opencode_sessions(data)
            for s in sessions:
                if s["location"]["directory"] == data["cwd"]:
                    self.api("session.move", {"directory": repo["root"]}, sessionID=s["id"])
                    self.api("session.update", {"archived": True}, sessionID=s["id"])
            # V2 can keep a loaded location after its worktree is removed; a
            # later global reload then fails on the missing directory.
            # Evict only our idle checkout before removal, never other locations.
            query = "/api/debug/location?location%5Bdirectory%5D=" + quote(data["cwd"], safe="")
            command([self.config["opencode"], "api", "delete", query])
            self.api("worktree.remove", {"projectID": data["project"],
                                        "directory": data["cwd"], "force": False})
        else:
            git(repo["root"], "worktree", "remove", str(cwd))
        data.update(removed=True, pr=pr["number"] if pr else None, updated=time.time())
        write_json(path, data)
        # Record successful checkout removal before attempting branch deletion.
        # A changed or undeletable branch remains intact and is reported separately.
        status = "removed"
        try:
            actual = git(repo["root"], "rev-parse", "refs/heads/" + info["branch"])
            if actual == info["head"]:
                git(repo["root"], "branch", "-D" if pr else "-d", info["branch"])
            else:
                status = "removed-branch-kept"
        except RuntimeError:
            status = "removed-branch-kept"
        self.cleanup_sandbox(data["cwd"])
        # Updating main is best effort and never resets a busy/modified checkout.
        if git(repo["root"], "branch", "--show-current") == "main" and not git(
                repo["root"], "status", "--porcelain", "--untracked-files=all"):
            try:
                git(repo["root"], "merge", "--ff-only", "origin/main")
            except RuntimeError:
                pass
        return status

    def sweep(self, apply=False):
        results = []
        for p in self.state.glob("*.json"):
            branch, cwd = None, None
            try:
                record = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(record, dict):
                    raise ValueError("invalid task record")
                branch = record.get("branch") if isinstance(record.get("branch"), str) else None
                cwd = record.get("cwd") if isinstance(record.get("cwd"), str) else None
                status = self.sweep_one(p, apply)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as ex:
                status = "kept-unverified"
                reason = str(ex) if isinstance(ex, ValueError) and type(ex) is ValueError else type(ex).__name__
                results.append({"task": p.stem, "branch": branch, "cwd": cwd,
                                "status": status, "reason": reason})
                continue
            results.append({"task": p.stem, "branch": branch, "cwd": cwd, "status": status})
            if apply and status == "already-removed" and time.time() - p.stat().st_mtime > 30 * 86400:
                p.unlink()  # Bounded retention for completed task metadata, never for checkouts.
        write_json(self.state / "last-sweep.result", {"time": time.time(), "results": results})
        return results


REPORT_PREFIX = "AgentWorkspaceReport "
REPORT_REASONS = {
    "outside allowed checkout roots", "not a checkout root", "different repository",
    "protected branch", "checkout locked or not registered", "linked sandbox",
    "sandbox not owned", "active session", "recent session", "no matching merged PR",
    "open PR", "remote branch changed", "unmanaged checkout", "branch changed",
    "unsaved or ignored files", "active Claude process", "owner unknown", "project unknown",
    "another registered owner", "checkout changed during verification",
    "invalid task record",
}
REPORT_STATUS_REASONS = {
    "would-remove": "checks passed; dry-run",
    "removed": "checks passed; removed",
    "removed-branch-kept": "checkout removed; branch retained",
    "already-removed": "previously removed",
    "missing-kept": "checkout missing; metadata kept",
}


def write_report(path, apply, results, now=None):
    """Append a single human-readable run; retain unknown lines and recent runs."""
    path = Path(path).absolute()
    if any(linked(p) for p in (path, *path.parents)):
        raise ValueError("linked report")
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time() if now is None else now
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
    kept = []
    for line in lines:
        if line.startswith(REPORT_PREFIX):
            try:
                previous = json.loads(line[len(REPORT_PREFIX):])
                timestamp = datetime.fromisoformat(previous["time"]).timestamp()
                if timestamp < now - 30 * 86400:
                    continue
            except (ValueError, TypeError, KeyError, OverflowError):
                pass
        kept.append(line)
    tasks = []
    for result in results:
        reason = result.get("reason")
        if reason and reason not in REPORT_REASONS:
            reason = "verification failed"
        tasks.append({"branch": result.get("branch"), "worktree": result.get("cwd"),
                      "status": result["status"],
                      "reason": reason or REPORT_STATUS_REASONS.get(result["status"], "unknown")})
    entry = {"time": datetime.fromtimestamp(now, timezone.utc).isoformat(),
             "mode": "apply" if apply else "dry-run", "tasks": tasks}
    line = REPORT_PREFIX + json.dumps(entry, ensure_ascii=False) + "\n"
    if len(kept) == len(lines):
        with path.open("a", encoding="utf-8", newline="") as f:
            f.write(line)
    else:
        tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            tmp.write_text("".join(kept) + line, encoding="utf-8", newline="")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("register")
    p.add_argument("--tool", choices=("claude", "opencode"), required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--cwd", required=True)
    p.add_argument("--project")
    p = sub.add_parser("release")
    p.add_argument("--tool", required=True)
    p.add_argument("--session", required=True)
    p = sub.add_parser("sweep")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--report", type=Path)
    sub.add_parser("claude-hook")
    a = parser.parse_args()
    try:
        config = json.loads(Path(a.config).read_text(encoding="utf-8"))
        manager = Manager(config)
        with registry_lock(manager.state):
            if a.action == "register":
                out = manager.register(a.tool, a.session, a.cwd, project=a.project)
            elif a.action == "release":
                out = manager.release(a.tool, a.session)
            elif a.action == "claude-hook":
                data = json.load(sys.stdin)
                if data.get("hook_event_name") == "SessionEnd":
                    out = manager.release("claude", data["session_id"])
                else:
                    out = manager.register("claude", data["session_id"], data["cwd"], claude_owner())
            else:
                out = manager.sweep(a.apply)
                if a.report:
                    write_report(a.report, a.apply, out)
        # Hooks must not inject registry paths or task metadata into the model context.
        if a.action != "claude-hook":
            print(json.dumps(out))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError):
        print("Workspace maintenance deferred; no unverified checkout removed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
