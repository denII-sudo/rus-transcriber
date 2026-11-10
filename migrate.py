"""Одноразовый перенос данных из папки программы (прежняя раскладка) в DATA_DIR.

Что переносится:   profiles\\<g>.voices.json -> groups\\<g>\\voices.json      profiles\\vocab.json -> groups\\<группа словаря>\\vocab.json
                   config.json -> config.json (только отличия от умолчаний)   jobs\\ -> jobs\\   bench_texts.json, bench_report.md -> bench\\
Порядок: копирование -> проверка (хэш, чтение JSON) -> старые файлы УХОДЯТ В РЕЗЕРВНУЮ КОПИЮ
DATA_DIR\\_backup-<дата>\\ (не удаляются: удалить её можно вручную, когда всё проверено). Повторный запуск
безопасен: готовое не перезаписывается. Если идёт расшифровка, перенос откладывается. Итог — DATA_DIR\\migration.log.
Запуск вручную: python migrate.py [--dry-run]
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import paths
import settings

MARK = paths.DATA_DIR / ".migrated.json"
LOG = paths.DATA_DIR / "migration.log"


def _log(msg):
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _lock():
    f = open(paths.DATA_DIR / ".migrate.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def _legacy_items():
    """[(источник, назначение)] — только файлы."""
    H = paths.HERE
    items = []
    prof = paths.LEGACY_PROFILES
    vocab_group = None
    if (prof / "vocab.json").is_file():
        try:
            vocab_group = json.loads((prof / "vocab.json").read_text(encoding="utf-8-sig")).get("group")
        except Exception:
            vocab_group = None
        items.append((prof / "vocab.json", paths.vocab_path(vocab_group or "default")))
    if prof.is_dir():
        for p in sorted(prof.glob("*.voices.json")):
            g = p.name[:-len(".voices.json")]
            items.append((p, paths.voices_path(g)))
    for name, dst in (("bench_texts.json", paths.BENCH / "bench_texts.json"),
                      ("bench_report.md", paths.BENCH / "bench_report.md")):
        if (H / name).is_file():
            items.append((H / name, dst))
    jobs = paths.LEGACY_JOBS
    if jobs.is_dir():
        for p in sorted(jobs.rglob("*")):
            if p.is_file() and "slots" not in p.relative_to(jobs).parts and p.suffix not in (".pid", ".lock", ".tmp"):
                items.append((p, paths.JOBS / p.relative_to(jobs)))
    return items, vocab_group


def _busy():
    """Есть ли свежая незавершённая задача (старая или новая раскладка)."""
    now = time.time()
    for d in (paths.LEGACY_JOBS, paths.JOBS, *[o / "jobs" for o in paths.old_default_dirs()]):
        if not d.is_dir():
            continue
        for p in d.glob("*.json"):
            try:
                if now - p.stat().st_mtime < 900 and json.loads(p.read_text(encoding="utf-8")).get("state") in ("queued", "running"):
                    return p.name
            except Exception:
                continue
    return None


def pending():
    items, _ = _legacy_items()
    return bool(items) or paths.LEGACY_CONFIG.is_file() or bool(paths.old_default_dirs())


def _relocate(dry=False):
    """Данные из прежнего места по умолчанию -> DATA_DIR: копирование с проверкой, ничего не перезаписывается,
    исходная папка переименовывается в rus-transcriber-data.moved-<дата> (удаляете сами)."""
    done = []
    for src in paths.old_default_dirs():
        files = [p for p in src.rglob("*") if p.is_file() and p.name != ".migrate.lock"]
        if dry:
            done.append({"from": str(src), "files": len(files)})
            continue
        conflicts = 0
        for p in files:
            d = paths.DATA_DIR / p.relative_to(src)
            if d.exists():
                if _sha(d) != _sha(p):
                    conflicts += 1
                    _log(f"КОНФЛИКТ при переносе: {d} уже есть и отличается — оставлен новый, прежний лежит в {src.name}.moved")
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            tmp = d.with_name(d.name + ".migrating")
            shutil.copy2(p, tmp)
            if _sha(tmp) != _sha(p):
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"контрольная сумма не совпала: {p}")
            tmp.replace(d)
        moved = src.with_name(src.name + ".moved-" + time.strftime("%Y%m%d-%H%M%S"))
        src.rename(moved)
        _log(f"данные перенесены из {src} ({len(files)} файлов, конфликтов {conflicts}); прежняя папка: {moved}")
        done.append({"from": str(src), "files": len(files), "conflicts": conflicts, "old_folder": str(moved)})
    return done


def run(dry=False):
    if not pending():
        return {"status": "nothing"}
    lk = _lock()
    if lk is None:
        return {"status": "busy", "why": "перенос уже идёт в другом процессе"}
    try:
        b = _busy()
        if b:
            _log(f"перенос отложен: идёт задача {b}")
            return {"status": "deferred", "why": f"идёт задача {b}"}
        relocated = _relocate(dry)
        if not _legacy_items()[0] and not paths.LEGACY_CONFIG.is_file():
            return {"status": "relocated" if relocated else "nothing", "relocated": relocated}
        items, vgroup = _legacy_items()
        cfg_old = None
        if paths.LEGACY_CONFIG.is_file():
            try:
                cfg_old = json.loads(paths.LEGACY_CONFIG.read_text(encoding="utf-8-sig"))
            except Exception as ex:
                _log(f"старый config.json не прочитан ({ex}) — уйдёт в резервную копию без переноса")
        if dry:
            return {"status": "dry-run", "relocate": relocated, "files": [f"{s} -> {d}" for s, d in items], "config": bool(cfg_old)}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = paths.DATA_DIR / f"_backup-{stamp}"
        _log(f"старт переноса: {len(items)} файлов, config={'да' if cfg_old else 'нет'}")
        copied, conflicts = [], []
        # 1. копирование и проверка; ничего не удаляется
        for s, d in items:
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists():
                if _sha(d) != _sha(s):
                    conflicts.append(str(s))
                    _log(f"КОНФЛИКТ: {d} уже есть и отличается — оставлен как есть, исходник уйдёт в резерв")
                continue
            tmp = d.with_name(d.name + ".migrating")
            shutil.copy2(s, tmp)
            if _sha(tmp) != _sha(s):
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"контрольная сумма не совпала: {s}")
            if d.suffix == ".json" and not d.is_relative_to(paths.JOBS):
                json.loads(tmp.read_text(encoding="utf-8-sig"))
            tmp.replace(d)
            copied.append(d)
        if cfg_old is not None and not paths.CONFIG_FILE.exists():
            over = {k: v for k, v in cfg_old.items() if k != "schema_version"}
            if not over.get("default_group") and vgroup:
                over["default_group"] = vgroup
            kept = settings.save_overrides(over)
            _log(f"config.json: сохранены отличия от умолчаний: {sorted(kept)}")
        # 2. прежние файлы -> резервная копия (только после того, как всё скопировано и проверено)
        backup.mkdir(parents=True, exist_ok=True)
        moved = 0
        for s, _d in items:
            rel = s.relative_to(paths.HERE)
            t = backup / rel
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(s), str(t))
            moved += 1
        if cfg_old is not None or paths.LEGACY_CONFIG.is_file():
            shutil.move(str(paths.LEGACY_CONFIG), str(backup / "config.json"))
        for junk in list(paths.LEGACY_JOBS.rglob("*")) if paths.LEGACY_JOBS.is_dir() else []:
            if junk.is_file() and (junk.suffix in (".pid", ".lock", ".tmp") or "slots" in junk.parts):
                junk.unlink(missing_ok=True)                  # служебные остатки (идентификаторы процессов, замки)
        for d in (paths.LEGACY_PROFILES, paths.LEGACY_JOBS):      # пустые каталоги убираем
            try:
                for sub in sorted(d.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
                d.rmdir()
            except OSError:
                pass
        res = {"status": "done", "copied": len(copied), "moved_to_backup": moved, "backup": str(backup),
               "conflicts": conflicts, "default_group": vgroup, "relocated": relocated}
        MARK.write_text(json.dumps({**res, "time": stamp}, ensure_ascii=False, indent=1), encoding="utf-8")
        _log(f"готово: скопировано {len(copied)}, в резерв {moved}, резерв {backup}, конфликтов {len(conflicts)}")
        return res
    except Exception as ex:
        _log(f"ОШИБКА переноса, исходные файлы не тронуты: {ex}")
        raise
    finally:
        lk.close()


if __name__ == "__main__":
    print(json.dumps(run(dry="--dry-run" in sys.argv), ensure_ascii=False, indent=1))
