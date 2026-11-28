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
import sys

MANIFEST = "build-info.json"
# Top-level program files are taken by suffix, so a new module needs no list entry here
# or in install.ps1. Service scripts belong in tools/, not in the repository root.
PROGRAM_SUFFIXES = (".py", ".html", ".tsv")
FIXED_ROOT_FILES = ("README.md", "install.ps1")
# ASCII only: install.ps1 reads the list from the console, whatever its code page.
ROOT_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
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


def root_files(here):
    """Program files at the top level only (never recursive), sorted.

    Hidden or non-ASCII names and transcripts (*.transcript.*, kept out by .gitignore)
    are not program files. A link with a program name is returned, so hashing or
    --list-root refuses it instead of silently shipping less.
    """
    names = []
    for path in Path(here).iterdir():
        name = path.name
        if not ROOT_NAME.fullmatch(name) or ".transcript." in name.lower():
            continue
        if not (name.lower().endswith(PROGRAM_SUFFIXES) or name in FIXED_ROOT_FILES):
            continue
        if _link(path) or path.is_file():
            names.append(name)
    return tuple(sorted(names))


def delivery_hashes(here, root=None):
    """Only the agreed installation payload; never traverse worktrees or data.

    root limits top-level program files to these names: an installation keeps root
    files the delivery does not ship, and they are neither hashed nor reported.
    """
    here = Path(here)
    names = [n for n in root_files(here) if root is None or n in root]
    names += [n for n in DOC_FILES if (here / n).is_file()]
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
    git = (here / ".git").exists()
    data = None if git else _manifest(here)
    try:
        # Our own inventory, only filtered by manifest names: no supplied path is read.
        files = delivery_hashes(here, root=set(data["files"]) if data else None)
        out["payload_sha256"] = payload_hash(files) if files else None
    except (OSError, ValueError):
        out["manifest_status"] = "unreadable"
        return out
    # Do not accidentally inherit a parent repository when examining an installation.
    if git:
        commit = _git(here, "rev-parse", "HEAD")
        status = _git(here, "status", "--porcelain", "--untracked-files=all")
        if commit and SHA.fullmatch(commit):
            out.update(commit=commit, source_state=(
                "unknown" if status is None else "dirty" if status else "clean"),
                manifest_status="git")
        return out
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
    copied = delivery_hashes(target, root=set(wanted))
    # A partial support-tree copy is allowed: the manifest describes what is actually
    # installed. Runtime and documented files are mandatory, and every included file
    # must still match the source byte for byte.
    files = {name: digest for name, digest in wanted.items() if name in copied}
    required = [name for name in root_files(source) + DOC_FILES if name in wanted]
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


def list_root(source, target=None):
    """Names for install.ps1 step 4; refuses links in the source or under the same name
    in the target, where Copy-Item would write through them."""
    names = root_files(source)
    places = [Path(source)] + ([Path(target)] if target else [])
    if any(_link(place / name) for place in places for name in names):
        raise ValueError("linked delivery entry")
    return names


REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT: junctions and other Windows links
NAME_LIMIT = 260


def _linked(path):
    """Link of any kind, including a junction on a Python without Path.is_junction."""
    if _link(path):
        return True
    try:
        return bool(path.lstat().st_file_attributes & REPARSE_POINT)
    except AttributeError:  # not Windows: st_file_attributes does not exist
        return False
    except OSError:
        return False


def _program_name(name):
    """The same rule root_files applies: a top-level program file, never data or a transcript."""
    return bool(ROOT_NAME.fullmatch(name)) and ".transcript." not in name.lower() and (
        name.lower().endswith(PROGRAM_SUFFIXES) or name in FIXED_ROOT_FILES)


def _inside(target, name):
    """Path of a delivery name under target, or None if it is not a plain delivery name.

    A manifest is installed data, not trusted input: only a root program file, one file
    directly under docs/ (.md), or a file under tools/, tests/ or examples/ is accepted, with
    no absolute path, drive, backslash, '.', '..' or empty part. An existing part of the
    path (a folder or the file itself) that is a link is refused, so nothing outside the
    program folder can be reached through it.
    """
    if not isinstance(name, str) or not name or len(name) > NAME_LIMIT:
        return None
    if "\\" in name or ":" in name or "\0" in name or name.startswith("/"):
        return None
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    if len(parts) == 1:
        if not _program_name(name):
            return None
    elif parts[0] == "docs":
        if len(parts) != 2 or not ROOT_NAME.fullmatch(parts[1]) or not parts[1].endswith(".md"):
            return None
    elif parts[0] not in SUPPORT_DIRS:
        return None
    here = Path(target)
    for part in parts:
        here = here / part
        if _linked(here):
            return None
    return here


def _regular_file(path):
    return path.is_file() and not _linked(path)


def _links_under(target):
    """Links inside the folders an update writes to (a link at the top counts as one)."""
    target = Path(target)
    found = []
    for sub in SUPPORT_DIRS + ("docs",):
        base = target / sub
        if _linked(base):
            found.append(sub)
            continue
        if not base.is_dir():
            continue
        for root, dirs, files in os.walk(base, followlinks=False):
            parent = Path(root)
            keep = []
            for d in sorted(dirs):
                if _linked(parent / d):
                    found.append((parent / d).relative_to(target).as_posix())
                else:
                    keep.append(d)
            dirs[:] = keep
            found.extend((parent / n).relative_to(target).as_posix()
                         for n in sorted(files) if _linked(parent / n))
    return sorted(found)


def stale_report(source, target):
    """What an update could remove from target, without changing anything.

    stale        files listed in the previous build-info.json, absent from the new source
                 and byte-identical to what was installed: provably ours, deletable;
    modified     the same, but changed (or not a plain file) since installation: kept;
    unknown      files under tools/, tests/, examples/ that the source does not ship and
                 the previous manifest never listed: possibly the user's own;
    root_unknown program files at the top level of target that the source does not ship
                 and the previous manifest never listed (shown only, never removed).
    links        links (symlink, junction) under tools/, tests/, examples/, docs/: an update
                 would write through them, so install.ps1 refuses to start.
    Without a valid previous manifest nothing is stale: files are only shown.
    Names are compared without regard to case (Windows keeps the old spelling of a file
    that a new version spells differently, so Foo.py -> foo.py must not look removed).
    """
    source, target = Path(source), Path(target)
    new = delivery_hashes(source)
    old = _manifest(target)
    old_files = old["files"] if old else {}
    report = {"manifest": old is not None, "stale": [], "modified": [], "unknown": [],
              "root_unknown": [], "links": _links_under(target)}
    new_keys = {n.casefold() for n in new}
    for name in sorted(set(old_files) - set(new)):
        if name.casefold() in new_keys:
            continue  # the same file under another spelling of its name: still shipped
        top = name.split("/")[0]
        if top in SUPPORT_DIRS and not (source / top).is_dir():
            continue  # the source ships no such folder at all: leave the installed one alone
        path = _inside(target, name)
        if path is None or not (path.exists() or path.is_symlink()):
            continue
        try:
            same = _regular_file(path) and file_hash(path) == old_files[name]
        except (OSError, ValueError):
            same = False
        report["stale" if same else "modified"].append(name)
    seen = new_keys | {n.casefold() for n in old_files}
    for sub in SUPPORT_DIRS:
        base = target / sub
        if not base.is_dir() or _linked(base):
            continue
        for root, dirs, files in os.walk(base, followlinks=False):
            parent = Path(root)
            dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not _linked(parent / d))
            for n in sorted(files):
                if n.endswith((".pyc", ".pyo")):
                    continue
                rel = (parent / n).relative_to(target).as_posix()
                # tools/tools/..., tests/tests/...: copies by an old installer, removed by it
                if rel.casefold() in seen or rel.startswith(f"{sub}/{sub}/"):
                    continue
                if _inside(target, rel) is not None and _regular_file(parent / n):
                    report["unknown"].append(rel)
    report["unknown"].sort()
    if target.is_dir():
        wanted = {n for n in new_keys if "/" not in n}
        known = {n.casefold() for n in old_files}
        report["root_unknown"] = [n for n in root_files(target)
                                  if n.casefold() not in wanted and n.casefold() not in known
                                  and _regular_file(target / n)]
    return report


def prune(source, target, what, shown):
    """Delete the files of stale_report under `what` ("stale" or "unknown") that were shown.

    `shown` is the report install.ps1 printed to the person before it copied anything:
    only names in both that report and the current one are deleted. A name that appeared
    since (the copy itself, a dot-file) is `skipped`, a shown name that is no longer a
    candidate is `gone`; neither is touched. Each name is checked again right before
    deletion; only plain files are removed, never folders (empty folders left under
    tools/, tests/, examples/ and docs/ are removed), never anything outside target.
    Returns (removed, failed, skipped, gone).
    """
    report = stale_report(source, target)
    shown_names = {n for n in shown.get(what, []) if isinstance(n, str)}
    current = set(report[what])
    removed, failed = [], []
    skipped = sorted(current - shown_names)
    gone = sorted(shown_names - current)
    for name in report[what]:
        if name not in shown_names:
            continue
        path = _inside(target, name)
        if path is None or not _regular_file(path):
            failed.append(name)
            continue
        try:
            path.unlink()
            removed.append(name)
        except OSError:
            failed.append(name)
            continue
        folder = path.parent
        while folder != Path(target) and folder.parent != Path(target) and not _linked(folder):
            try:
                folder.rmdir()  # only an empty folder goes; a non-empty one raises
            except OSError:
                break
            folder = folder.parent
    return removed, failed, skipped, gone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--target")
    p.add_argument("--list-root", action="store_true",
                   help="print top-level program files of --source, one per line")
    p.add_argument("--stale-report", action="store_true",
                   help="print JSON: files an update could remove from --target (changes nothing)")
    p.add_argument("--prune", choices=("stale", "unknown"),
                   help="delete the files under this heading of the report; print JSON")
    p.add_argument("--shown-stdin", action="store_true",
                   help="with --prune: the --stale-report JSON shown to the person, read from stdin; "
                        "only names present in it are deleted")
    a = p.parse_args()
    if a.stale_report or a.prune:
        if not a.target:
            p.exit(2, "Build identity: --target is required.\n")
        try:
            if a.prune:
                if not a.shown_stdin:
                    p.exit(2, "Build identity: --prune requires --shown-stdin.\n")
                shown = json.loads(sys.stdin.read())
                if not isinstance(shown, dict):
                    raise ValueError("shown list")
                removed, failed, skipped, gone = prune(a.source, a.target, a.prune, shown)
                out = {"removed": removed, "failed": failed, "skipped": skipped, "gone": gone}
            else:
                out = stale_report(a.source, a.target)
        except (OSError, ValueError):
            p.exit(1, "Build identity: stale files could not be listed.\n")
        print(json.dumps(out))  # ASCII only: install.ps1 reads it whatever the console code page
        return
    if a.list_root:
        try:
            names = list_root(a.source, a.target)
        except (OSError, ValueError):
            p.exit(1, "Build identity: program files could not be listed.\n")
        if not names:
            p.exit(1, "Build identity: no program files found.\n")
        print("\n".join(names))
        return
    try:
        out = write_manifest(a.source, a.target) if a.target else identity(a.source)
    except (OSError, ValueError):
        p.exit(1, "Build identity: payload could not be verified.\n")
    print(json.dumps({k: v for k, v in out.items() if k != "files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
