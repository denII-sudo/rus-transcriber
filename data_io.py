"""Перенос и обслуживание данных (библиотека): экспорт и импорт групп, удаление группы, очистка задач, проверка путей.

Командной строки здесь нет: она одна на всё — rt.py (`python rt.py export|import|delete-group|cleanup|audit`),
а решение человека для опасных действий принимают rt.py (терминал) и confirm.py (MCP).

Архив: manifest.json (schema_version, дата, группы, контрольные суммы) + groups/<группа>/{vocab,voices}.json
(+ jobs/*.json). Абсолютные пути заменяются именем файла. Архив содержит голосовые отпечатки — это биометрия:
хранить и пересылать как личные данные. Карантин профиля (голоса «тёзок», ждущие решения) не переносится.
"""
import hashlib
import io
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import paths

SCHEMA = paths.DATA_SCHEMA
APP = "rus-transcriber"
_ABS = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")
_MEMBER = re.compile(r"^(?:manifest\.json|groups/[^/\\]+/(?:vocab|voices)\.json|jobs/[^/\\]+\.json)$")
MAX_UNPACKED = 1 << 30


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _base(p):
    return re.split(r"[\\/]", str(p).rstrip("\\/"))[-1]


def normalize(obj):
    """Абсолютные пути (в ключах и значениях) -> имя файла; при совпадении имён ключу добавляется «#2»."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = _base(k) if isinstance(k, str) and _ABS.match(k) else k
            n = 2
            while nk in out:
                nk = f"{_base(k)}#{n}"
                n += 1
            out[nk] = normalize(v)
        return out
    if isinstance(obj, list):
        return [normalize(x) for x in obj]
    if isinstance(obj, str) and _ABS.match(obj):
        return _base(obj)
    return obj


def _dump(o):
    return json.dumps(o, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _speaker_seconds(voices):
    return {n: round(sum(pt.get("sec", 0) for pt in s.get("parts", {}).values()))
            for n, s in (voices.get("speakers") or {}).items()}


# ---------- экспорт ----------
def export_data(dest_dir=None, group=None, with_jobs=False):
    groups = [paths.safe_group(group)] if group else paths.list_groups()
    if not groups:
        raise RuntimeError("Нет групп для экспорта")
    files, warns, ginfo = {}, [], []
    for g in groups:
        gd = paths.group_dir(g)
        if not gd.is_dir():
            raise RuntimeError(f"Группа «{g}» не найдена (есть: {', '.join(paths.list_groups())})")
        info = {"name": g, "files": []}
        for fn in ("vocab.json", "voices.json"):
            p = gd / fn
            if not p.is_file():
                continue
            data = json.loads(p.read_text(encoding="utf-8-sig"))
            if fn == "voices.json":
                q = data.pop("quarantine", None)
                if q:
                    warns.append(f"{g}: карантин голосов не перенесён ({sum(len(v) for v in q.values())} записей)")
                data = normalize(data)
                data["schema_version"] = SCHEMA
                info["speakers_sec"] = _speaker_seconds(data)
            rel = f"groups/{g}/{fn}"
            files[rel] = _dump(data) if fn == "voices.json" else p.read_bytes()
            info["files"].append(rel)
        ginfo.append(info)
    if with_jobs:
        for p in sorted(paths.JOBS.glob("*.json")):
            try:
                files[f"jobs/{p.name}"] = _dump(normalize(json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                continue
    manifest = {"app": APP, "schema_version": SCHEMA, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "groups": ginfo, "with_jobs": bool(with_jobs), "path_normalization": "basename", "warnings": warns,
                "files": {k: {"sha256": _sha(v), "size": len(v)} for k, v in files.items()},
                "note": "Содержит голосовые отпечатки (биометрия) и имена участников — храните как личные данные."}
    dest = Path(dest_dir) if dest_dir else paths.DATA_DIR / "exports"
    dest.mkdir(parents=True, exist_ok=True)
    name = f"rus-transcriber-data-{groups[0] if len(groups) == 1 else 'all'}-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    out = dest / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
        for k, v in files.items():
            z.writestr(k, v)
    return {"zip": str(out), "groups": [g["name"] for g in ginfo], "files": len(files), "warnings": warns,
            "size": out.stat().st_size}


# ---------- импорт ----------
def _read_archive(zip_path):
    z = zipfile.ZipFile(zip_path)
    infos = z.infolist()
    if sum(i.file_size for i in infos) > MAX_UNPACKED:
        raise RuntimeError("Архив слишком большой после распаковки")
    for i in infos:
        n = i.filename
        if n.startswith("/") or ".." in n.split("/") or "\\" in n or not _MEMBER.match(n):
            raise RuntimeError(f"Недопустимое имя в архиве: {n!r}")
    try:
        man = json.loads(z.read("manifest.json"))
    except KeyError:
        raise RuntimeError("В архиве нет manifest.json — это не архив rus-transcriber") from None
    if man.get("app") != APP:
        raise RuntimeError("Архив не от rus-transcriber")
    if int(man.get("schema_version", 0)) > SCHEMA:
        raise RuntimeError(f"Архив новее программы (schema_version {man.get('schema_version')} > {SCHEMA}): обновите программу")
    blobs = {}
    for rel, meta in man.get("files", {}).items():
        if not _MEMBER.match(rel):
            raise RuntimeError(f"Недопустимое имя в манифесте: {rel!r}")
        b = z.read(rel)
        if _sha(b) != meta.get("sha256"):
            raise RuntimeError(f"Контрольная сумма не совпала: {rel} (архив повреждён)")
        blobs[rel] = b
    return man, blobs


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _merge_voices(ex, im, report):
    import voices
    if im.get("version") != voices.PROFILE_VERSION or im.get("model") != ex.get("model"):
        report["voices"] = "пропущено: другая версия или модель голосов (профили несовместимы)"
        return ex
    report["voices"] = "слито"
    report["added_speakers"], report["conflicts"] = [], []
    for name, sp in (im.get("speakers") or {}).items():
        cur = (ex.setdefault("speakers", {})).get(name)
        if cur is None:
            ex["speakers"][name] = sp
            report["added_speakers"].append(name)
            continue
        a = sum(pt.get("sec", 0) for pt in cur.get("parts", {}).values())
        b = sum(pt.get("sec", 0) for pt in sp.get("parts", {}).values())
        keep = "imported" if b > a else "existing"
        if keep == "imported":
            ex["speakers"][name] = sp
        report["conflicts"].append({"speaker": name, "kept": keep, "existing_sec": round(a), "imported_sec": round(b)})
    for k, v in (im.get("sources") or {}).items():
        ex.setdefault("sources", {}).setdefault(k, v)
    for k, v in (im.get("confusions") or {}).items():
        ex.setdefault("confusions", {}).setdefault(k, v)
    ex.setdefault("history", []).append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "from": "импорт архива",
                                         "speakers": sorted(im.get("speakers") or {})})
    ex["revision"] = int(ex.get("revision", 0)) + 1
    return ex


def import_data(zip_path, mode="merge", apply=False):
    if mode not in ("merge", "replace"):
        raise RuntimeError('mode: "merge" или "replace"')
    man, blobs = _read_archive(zip_path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = paths.DATA_DIR / f"_backup-import-{stamp}"
    result = {"mode": mode, "applied": bool(apply), "archive_created": man.get("created"), "groups": {},
              "jobs": "не импортируются" if man.get("with_jobs") else None, "backup": None}
    for g in [x["name"] for x in man.get("groups", [])]:
        if paths.safe_group(g) != g:
            raise RuntimeError(f"Недопустимое имя группы: {g!r}")
        rep = {}
        gd = paths.group_dir(g)
        vk, ok = f"groups/{g}/voices.json", f"groups/{g}/vocab.json"
        im_voices = json.loads(blobs[vk]) if vk in blobs else None
        existed = gd.is_dir() and any((gd / f).exists() for f in ("vocab.json", "voices.json"))
        rep["action"] = "новая группа" if not existed else ("замена" if mode == "replace" else "слияние")
        if existed and mode == "replace" and apply:
            (backup / "groups").mkdir(parents=True, exist_ok=True)
            shutil.copytree(gd, backup / "groups" / g)
            result["backup"] = str(backup)
        # словарь
        if ok in blobs:
            vp = gd / "vocab.json"
            if not vp.exists():
                rep["vocab"] = "добавлен"
                if apply:
                    vp.parent.mkdir(parents=True, exist_ok=True)
                    vp.write_bytes(blobs[ok])
            elif vp.read_bytes() == blobs[ok]:
                rep["vocab"] = "совпадает"
            elif mode == "replace":
                rep["vocab"] = "заменён"
                if apply:
                    vp.write_bytes(blobs[ok])
            else:
                rep["vocab"] = "расходится: оставлен существующий (импортируемый — rus-transcriber.vocab.imported.json рядом)"
                if apply:
                    (gd / "vocab.imported.json").write_bytes(blobs[ok])
        # голоса
        if im_voices is not None:
            pp = gd / "voices.json"
            if not pp.exists() or mode == "replace":
                rep["voices"] = "добавлены" if not pp.exists() else "заменены"
                rep["added_speakers"] = sorted(im_voices.get("speakers") or {})
                if apply:
                    _write_json(pp, im_voices)
            else:
                ex = json.loads(pp.read_text(encoding="utf-8"))
                merged = _merge_voices(ex, im_voices, rep)
                if apply and rep.get("voices") == "слито":
                    _write_json(pp, merged)
        result["groups"][g] = rep
    result["warnings"] = man.get("warnings", [])
    return result


# ---------- удаление группы, очистка, проверка ----------
def delete_group(group, confirm="", permanent=False):
    g = paths.safe_group(group)
    if confirm != group:
        raise RuntimeError(f"Подтверждение не совпало: для удаления передайте confirm=«{group}»")
    gd = paths.group_dir(g)
    if not gd.is_dir():
        raise RuntimeError(f"Группа «{group}» не найдена (есть: {', '.join(paths.list_groups())})")
    if permanent:
        shutil.rmtree(gd)
        return {"group": g, "deleted": "навсегда"}
    dst = paths.DATA_DIR / "_deleted" / f"{g}-{time.strftime('%Y%m%d-%H%M%S')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(gd), str(dst))
    return {"group": g, "deleted": "перенесена (можно вернуть)", "folder": str(dst)}


def cleanup_jobs(days=30, apply=False):
    cutoff = time.time() - max(0, float(days)) * 86400
    n = size = 0
    for p in paths.JOBS.iterdir():
        if not p.is_file() or p.suffix not in (".json", ".log", ".pid"):
            continue
        try:
            st = p.stat()
            if st.st_mtime >= cutoff:
                continue
            if p.suffix == ".json":
                state = json.loads(p.read_text(encoding="utf-8")).get("state")
                if state in ("queued", "running") and time.time() - st.st_mtime < 86400:
                    continue
            n += 1
            size += st.st_size
            if apply:
                p.unlink()
        except (OSError, ValueError):
            continue
    return {"older_than_days": days, "files": n, "bytes": size, "applied": bool(apply)}


def audit():
    """Файлы данных с абсолютными путями (кроме jobs, где они по смыслу есть)."""
    found = []
    for p in paths.GROUPS.rglob("*.json"):
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(k, str) and _ABS.match(k):
                        yield k
                    yield from walk(v)
            elif isinstance(o, list):
                for x in o:
                    yield from walk(x)
            elif isinstance(o, str) and _ABS.match(o):
                yield o
        try:
            hits = list(walk(json.loads(p.read_text(encoding="utf-8-sig"))))
        except Exception:
            continue
        if hits:
            found.append({"file": str(p.relative_to(paths.DATA_DIR)), "absolute_paths": len(hits)})
    return {"files_with_absolute_paths": found,
            "note": "в профилях это ключи учтённых записей: при экспорте заменяются именем файла"}
