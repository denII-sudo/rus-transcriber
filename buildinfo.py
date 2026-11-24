"""Build identity without importing the application or disclosing local paths.

The install manifest is diagnostic provenance, not a cryptographic signature.
VERSION remains a literal in server.py; data schema versions are independent.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

MANIFEST = "build-info.json"
ROOT_FILES = (
    "server.py", "worker.py", "bench.py", "runinfo.py", "benchdb.py", "buildinfo.py", "backends.py",
    "paths.py", "recordings.py", "storage.py", "documents.py", "workflow.py",
    "product_policy.py", "model_tasks.py", "llm_provider.py", "download_hint.py", "transcript.py",
    "diar_fast.py", "vocab.py", "voices.py", "gender.py", "words.py", "hypotheses.py", "hypstore.py", "hotspots.py",
    "quotes.py", "conspect.py", "review_pack.py", "freeform.py", "answer_phrases.tsv", "jobslots.py",
    "settings.py", "migrate.py", "data_io.py", "core.py", "core_voices.py", "core_screen.py", "confirm.py",
    "tmpfiles.py", "selfcheck.py", "rt.py", "voice_io.py", "voice_review.py", "review_page.py",
    "workflow_page.py", "cards.html", "workflow.html", "README.md", "install.ps1",
)
DOC_FILES = (
    "docs/model-download-certificate.md", "docs/2026-11-17-p3-workflow-handoff.md",
    "docs/2026-11-17-p4-opencode-openai-handoff.md", "docs/2026-11-17-p5-screen-handoff.md",
)
SUPPORT_DIRS = ("tools", "tests", "examples")
SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
HASH = re.compile(r"[0-9a-f]{64}")


def _link(path):
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _git(here, *args):
    try:
        p = subprocess.run(["git", "--no-optional-locks", "-C", str(here), *args],
                           capture_output=True, text=True, encoding="utf-8", timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def release(here):
    try:
        tree = ast.parse((Path(here) / "server.py").read_text(encoding="utf-8"))
        value = next(ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and any(
                         isinstance(t, ast.Name) and t.id == "VERSION" for t in n.targets))
        return value if isinstance(value, str) else None
    except (OSError, SyntaxError, ValueError, StopIteration):
        return None


def file_hash(path):
    if _link(path):
        raise ValueError("linked delivery entry")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def delivery_hashes(here):
    """Only the agreed installation payload; never traverse worktrees or data."""
    here = Path(here)
    names = [n for n in ROOT_FILES + DOC_FILES if (here / n).is_file()]
    for sub in SUPPORT_DIRS:
        base = here / sub
        if _link(base):
            raise ValueError("linked delivery directory")
        if not base.is_dir():
            continue
        for root, dirs, files in os.walk(base, followlinks=False):
            parent = Path(root)
            if any(_link(parent / d) for d in dirs):
                raise ValueError("linked delivery directory")
            dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
            names.extend((parent / n).relative_to(here).as_posix()
                         for n in files if not n.endswith((".pyc", ".pyo")) and not n.startswith("."))
    return {n: file_hash(here / n) for n in sorted(names)}


def payload_hash(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _manifest(here):
    try:
        p = Path(here) / MANIFEST
        if _link(p) or p.stat().st_size > 2_000_000:
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        files = data["files"]
        if data["schema"] != 1 or not isinstance(files, dict) or not files:
            return None
        # Never read a path supplied by the manifest: compare with our own inventory.
        if any(not isinstance(k, str) or not isinstance(v, str) or not HASH.fullmatch(v)
               for k, v in files.items()):
            return None
        if data.get("commit") is not None and not SHA.fullmatch(str(data["commit"])):
            return None
        if data.get("source_state") not in ("clean", "dirty", "unknown"):
            return None
        if data.get("payload_sha256") != payload_hash(files):
            return None
        return data
    except (OSError, ValueError, TypeError, KeyError):
        return None


def identity(here):
    here = Path(here)
    out = {"release": release(here), "commit": None, "source_state": "unknown",
           "payload_sha256": None, "manifest_status": "missing"}
    try:
        files = delivery_hashes(here)
        out["payload_sha256"] = payload_hash(files) if files else None
    except (OSError, ValueError):
        out["manifest_status"] = "unreadable"
        return out
    # Do not accidentally inherit a parent repository when examining an installation.
    if (here / ".git").exists():
        commit = _git(here, "rev-parse", "HEAD")
        status = _git(here, "status", "--porcelain", "--untracked-files=all")
        if commit and SHA.fullmatch(commit):
            out.update(commit=commit, source_state=(
                "unknown" if status is None else "dirty" if status else "clean"),
                manifest_status="git")
        return out
    data = _manifest(here)
    if data:
        unchanged = files == data["files"] and out["release"] == data.get("release")
        out.update(commit=data.get("commit"),
                   source_state=data["source_state"] if unchanged else "dirty",
                   manifest_status="matched" if unchanged else "modified")
    elif (here / MANIFEST).exists():
        out["manifest_status"] = "invalid"
    return out


def write_manifest(source, target):
    source, target = Path(source), Path(target)
    wanted = delivery_hashes(source)
    copied = delivery_hashes(target)
    # A partial support-tree copy is allowed: the manifest describes what is actually
    # installed. Runtime and documented files are mandatory, and every included file
    # must still match the source byte for byte.
    files = {name: digest for name, digest in wanted.items() if name in copied}
    required = [name for name in ROOT_FILES + DOC_FILES if name in wanted]
    if (not files or any(name not in copied for name in required)
            or any(copied[name] != digest for name, digest in files.items())):
        raise ValueError("copied payload differs from source")
    info = identity(source)
    data = {"schema": 1, "release": release(source), "commit": info["commit"],
            "source_state": info["source_state"], "payload_sha256": payload_hash(files),
            "files": files}
    tmp = target / (MANIFEST + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target / MANIFEST)
    finally:
        tmp.unlink(missing_ok=True)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--target")
    a = p.parse_args()
    try:
        out = write_manifest(a.source, a.target) if a.target else identity(a.source)
    except (OSError, ValueError):
        p.exit(1, "Build identity: payload could not be verified.\n")
    print(json.dumps({k: v for k, v in out.items() if k != "files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
