r"""Проверка перед коммитом: в отслеживаемых файлах нет личных данных.

    python tools/check_repo.py              # отслеживаемые файлы (git ls-files) или, без git, все файлы папки
    python tools/check_repo.py --staged     # только то, что собираетесь закоммитить (для pre-commit)
    python tools/check_repo.py --list-words # показать, какие слова считаются запрещёнными (без привязки к файлам)

Что проверяется:
  1. файлы, которых в репозитории быть не должно: *.voices.json, groups/, jobs/, config.json, bench*, аудио,
     *.transcript.*, cache/, .venv/;
  2. абсолютные пути с настоящим именем пользователя: C:\Users\<имя>\ (допустимы <вы>, Иван, %USERNAME%, user …);
  3. запрещённые слова — имена участников, названия и термины групп. Откуда берутся (список хранится ВНЕ
     репозитория): все groups\*\vocab.json и voices.json из папки данных (RUS_TRANSCRIBER_DATA или
     %APPDATA%\rus-transcriber-data) + необязательный файл со своими словами (по одному в строке): путь в
     RUS_FORBIDDEN_WORDS или forbidden_words.txt в папке данных. Слова короче 3 букв пропускаются; общеупотребительные
     слова, попавшие в список по ошибке, внесите в allow_words.txt рядом со списком.
Код возврата: 0 — чисто, 1 — найдены нарушения, 2 — не удалось получить список слов (проверьте на компьютере с данными).
"""
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAD_NAMES = ["*.voices.json", "voices.json", "config.json", "bench_*", "bench.json", "*.transcript.*",
             "*.mp3", "*.m4a", "*.wav", "*.ogg", "*.opus", "*.flac", "*.aac", "*.mp4", "*.mkv", "*.webm",
             "rus-transcriber.vocab.json"]
BAD_DIRS = {"groups", "jobs", "cache", ".venv", "profiles", "exports", "_deleted"}
OK_USERS = {"иван", "user", "username", "you", "name", "вы", "<вы>", "ваше_имя", "someone", "public"}
TEXT_MAX = 2_000_000
USER_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+([^\\/\s\"'`<>%]+)", re.I)


def data_dir():
    env = os.environ.get("RUS_TRANSCRIBER_DATA", "").strip()
    if env:
        return Path(env)
    if os.name == "nt":
        return Path.home() / "AppData" / "Roaming" / "rus-transcriber-data"
    return Path.home() / ".local" / "share" / "rus-transcriber-data"


def _words_from_json(d, out):
    for p in d.get("participants", []) or []:
        for k in ("name", "full_name"):
            if p.get(k):
                out.add(str(p[k]))
    out.update(map(str, d.get("terms", []) or []))
    for k, v in (d.get("replace") or {}).items():
        out.update((str(k), str(v)))
    for r in d.get("replace_if", []) or []:
        out.update(str(r[k]) for k in ("from", "to") if r.get(k))
    if d.get("group"):
        out.add(str(d["group"]))
    for n in (d.get("speakers") or {}):
        out.add(str(n))


def forbidden_words():
    words, allow = set(), set()
    dd = data_dir()
    if dd.is_dir():
        for g in (dd / "groups").glob("*") if (dd / "groups").is_dir() else []:
            words.add(g.name)
            for fn in ("vocab.json", "voices.json"):
                p = g / fn
                if p.is_file():
                    try:
                        _words_from_json(json.loads(p.read_text(encoding="utf-8-sig")), words)
                    except Exception:
                        pass
    extra = os.environ.get("RUS_FORBIDDEN_WORDS", "").strip()
    for p in ([Path(extra)] if extra else []) + [dd / "forbidden_words.txt"]:
        if p.is_file():
            words.update(l.strip() for l in p.read_text(encoding="utf-8-sig").splitlines() if l.strip() and not l.startswith("#"))
    ap = dd / "allow_words.txt"
    if ap.is_file():
        allow = {l.strip().lower() for l in ap.read_text(encoding="utf-8-sig").splitlines() if l.strip()}
    out = set()
    for w in words:
        for part in re.split(r"[\s,;]+", w):      # «Северный проект» -> два слова
            part = part.strip(" .\"'«»()")
            if len(part) >= 3 and part.lower() not in allow:
                out.add(part)
    return sorted(out, key=str.lower)


def tracked(staged):
    try:
        if staged:
            r = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8", check=True)
        else:
            r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
        return [l for l in r.stdout.splitlines() if l.strip()], True
    except (OSError, subprocess.CalledProcessError):
        files = []
        gi = [l.strip() for l in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
              if l.strip() and not l.startswith("#")] if (ROOT / ".gitignore").exists() else []
        # папка без git (например, установленная копия программы): окружение, кэши и личные данные
        # не являются файлами исходников, а .gitignore там может отсутствовать — не заходим в них вовсе
        skip_dirs = BAD_DIRS | {".git", "__pycache__"}
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for name in filenames:
                p = Path(dirpath) / name
                rel = p.relative_to(ROOT).as_posix()
                if any(fnmatch.fnmatch(rel, g.rstrip("/") + ("/*" if g.endswith("/") else "")) or
                       fnmatch.fnmatch(p.name, g.rstrip("/")) or
                       (g.endswith("/") and (rel.startswith(g) or f"/{g}" in f"/{rel}")) for g in gi):
                    continue
                files.append(rel)
        return files, False


def check(staged=False, require_words=True):
    files, is_git = tracked(staged)
    problems = []
    for rel in files:
        parts = rel.split("/")
        name = parts[-1]
        if any(fnmatch.fnmatch(name, pat) for pat in BAD_NAMES) or BAD_DIRS & set(parts[:-1]):
            problems.append((rel, "файл личных данных или результатов не должен быть в репозитории"))
    words = forbidden_words()
    wrx = re.compile(r"(?<![А-Яа-яЁёA-Za-z])(" + "|".join(re.escape(w) for w in words) + r")(?![А-Яа-яЁёA-Za-z])", re.I) if words else None
    for rel in files:
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size > TEXT_MAX:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for ln, line in enumerate(text.splitlines(), 1):
            for m in USER_PATH.finditer(line):
                if m.group(1).lower() not in OK_USERS and not m.group(1).startswith(("<", "%", "$")):
                    problems.append((f"{rel}:{ln}", f"путь с именем пользователя: {m.group(0)}"))
            if wrx:
                m = wrx.search(line)
                if m:
                    problems.append((f"{rel}:{ln}", f"запрещённое слово «{m.group(1)}»"))
    return problems, words, is_git


def main():
    if "--list-words" in sys.argv:
        for w in forbidden_words():
            print(w)
        return 0
    problems, words, is_git = check("--staged" in sys.argv)
    if not words:
        print("ВНИМАНИЕ: список запрещённых слов пуст (нет данных в папке данных и forbidden_words.txt). "
              "Проверка слов не выполнена — запустите на компьютере с данными.", file=sys.stderr)
    for where, why in problems:
        print(f"{where}: {why}")
    print(f"проверено ({'git' if is_git else 'папка без git'}), слов в списке: {len(words)}, нарушений: {len(problems)}")
    if problems:
        return 1
    return 2 if not words else 0


if __name__ == "__main__":
    sys.exit(main())
