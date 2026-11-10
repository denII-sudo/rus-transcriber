r"""Переносит данные rus-transcriber в обычную (не «пакетную») папку, например %APPDATA%\\rus-transcriber-data.

Запускать обычным Python из PowerShell при ЗАКРЫТОМ Claude Desktop:
    python move_data.py --dry-run      # только показать, что найдено и что будет сделано
    python move_data.py                # перенести в %APPDATA%\\rus-transcriber-data
    python move_data.py --dest D:\\rus-transcriber-data

Где ищет данные (если папка существует и это не папка назначения):
  - Packages\\Claude_*\\LocalCache\\{Local,Roaming}\\rus-transcriber-data   (скрытые папки Store-версии Claude)
  - %LOCALAPPDATA%\\rus-transcriber-data, %APPDATA%\\rus-transcriber-data, %USERPROFILE%\\rus-transcriber-data
  - любые, указанные через --src
Правила: копируются только отсутствующие файлы; при расхождении остаётся более новый, старый кладётся в
<назначение>\\_conflicts\; каждый файл проверяется контрольной суммой. После успешной проверки исходная папка
переименовывается в <имя>.moved-<дата> (ничего не удаляется; --keep-source оставляет её как есть).
Только стандартная библиотека Python.
"""
import argparse
import glob
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

NAME = "rus-transcriber-data"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def candidates(extra):
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    la = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    ra = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
    out = [Path(x) for x in extra]
    for pkg in glob.glob(str(la / "Packages" / "Claude_*")):
        for sub in ("Local", "Roaming"):
            out.append(Path(pkg) / "LocalCache" / sub / NAME)
    out += [la / NAME, ra / NAME, home / NAME]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="", help="куда переносить (по умолчанию %%APPDATA%%\\rus-transcriber-data)")
    ap.add_argument("--src", action="append", default=[], help="дополнительная папка-источник (можно несколько раз)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-source", action="store_true", help="не переименовывать исходные папки")
    a = ap.parse_args()

    ra = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    dest = Path(a.dest) if a.dest else ra / NAME
    if "Packages" in dest.parts:
        sys.exit(f"Назначение {dest} лежит внутри пакета приложения — выберите обычную папку.")
    dest_real = dest.resolve()
    srcs, seen = [], set()
    for c in candidates(a.src):
        try:
            if c.is_dir() and c.resolve() != dest_real and c.resolve() not in seen:
                seen.add(c.resolve())
                srcs.append(c)
        except OSError:
            pass
    if not srcs:
        print("Источников не найдено: переносить нечего.")
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_lines = []

    def log(m):
        print(m)
        log_lines.append(time.strftime("%Y-%m-%d %H:%M:%S ") + m)

    log(f"назначение: {dest}")
    for src in srcs:
        files = [p for p in src.rglob("*") if p.is_file() and not p.name.startswith(".migrate.lock")]
        log(f"источник: {src} ({len(files)} файлов)")
        if a.dry_run:
            continue
        copied = same = conflicts = 0
        for p in files:
            rel = p.relative_to(src)
            d = dest / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists():
                if sha(d) == sha(p):
                    same += 1
                    continue
                newer_src = p.stat().st_mtime > d.stat().st_mtime
                keep, other = (p, d) if newer_src else (d, p)
                cdir = dest / "_conflicts" / stamp / rel.parent
                cdir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(other, cdir / (rel.name + (".dest" if newer_src else ".src")))
                if newer_src:
                    tmp = d.with_name(d.name + ".migrating")
                    shutil.copy2(p, tmp)
                    tmp.replace(d)
                conflicts += 1
                log(f"расхождение {rel}: оставлен более новый ({'из источника' if newer_src else 'в назначении'}), "
                    f"второй в _conflicts\\{stamp}")
                continue
            tmp = d.with_name(d.name + ".migrating")
            shutil.copy2(p, tmp)
            if sha(tmp) != sha(p):
                tmp.unlink(missing_ok=True)
                sys.exit(f"Контрольная сумма не совпала: {p}. Источник не тронут, остановка.")
            tmp.replace(d)
            copied += 1
        # проверка: всё из источника теперь есть в назначении
        missing = [str(p.relative_to(src)) for p in files if not (dest / p.relative_to(src)).exists()]
        if missing:
            sys.exit(f"В назначении не хватает файлов: {missing[:5]} … Источник не тронут, остановка.")
        log(f"скопировано {copied}, уже было {same}, расхождений {conflicts}")
        if not a.keep_source:
            new = src.with_name(src.name + ".moved-" + stamp)
            try:
                src.rename(new)
                log(f"исходная папка переименована: {new}")
            except OSError as ex:
                log(f"переименовать не удалось ({ex}) — оставлена как есть")
    if a.dry_run:
        print("Пробный прогон: ничего не изменено.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "migration.log", "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print("Готово. Запустите Claude Desktop и попросите проверить get_settings.")


if __name__ == "__main__":
    main()
