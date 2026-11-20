"""Перенос записей старого файлового формата в SQLite-хранилище (схема v4).

Решения владельца (даты записей, основная версия, вид подтверждения голосов) задаются
в файле плана, а не угадываются здесь. Старые файлы (`*.transcript.*`, конспекты,
папки разбора) не изменяются: новые документы формируются в подпапке `v4` рядом.

    python tools/import_legacy.py check --plan plan.json [--report отчёт.json]
        проверка в памяти: ничего не пишет ни в БД, ни на диск (кроме отчёта)
    python tools/import_legacy.py apply --plan plan.json [--report отчёт.json]
        запись в БД текущей папки данных (RUS_TRANSCRIBER_DATA); БД создаётся заранее
        штатной операцией initialize_recording_store; повторный запуск пропускает уже перенесённое
    python tools/import_legacy.py mark-voices --plan plan.json [--apply]
        пометить, кем подтверждены фрагменты голосовых профилей; без --apply — только показать;
        перед изменением профиль копируется в voice-backups

Код возврата: 0 — без ошибок, 1 — есть записи с ошибками, 2 — неверный план или окружение.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOOL = "tools/import_legacy.py v1"
NAMESPACE = uuid.UUID("6b1d3c5e-8f0a-4c11-9d6e-2a7f4b9e0c31")
DOC_SUBDIR = "v4"
PLAN_VERSION = 1
METADATA_FIELDS = ("title", "recorded_at", "recorded_timezone", "container", "codec",
                   "audio_sha256", "audio_available")
TIME_EPS = 0.05
ANSWER_TS_FORMAT = "%Y-%m-%dT%H:%M:%S"     # как review_pack._now: UTC, сравнивается как строка
LEGACY_RUNTIME_KEYS = ("recording_id", "source_revision", "review_revision", "document_revision",
                       "document_sync")


class PlanError(ValueError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recording_id_for(source_path, source_sha):
    """Один и тот же исходный файл всегда даёт один и тот же recording_id (повтор безопасен)."""
    key = os.path.normcase(os.path.abspath(source_path)) + "|" + source_sha
    return str(uuid.uuid5(NAMESPACE, "legacy-recording:" + key))


def _norm_text(value):
    value = str(value or "").lower().replace("ё", "е")
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


# ---------------------------------------------------------------- план

def load_plan(path):
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as ex:
        raise PlanError(f"план не читается: {ex}") from ex
    if not isinstance(plan, dict) or plan.get("version") != PLAN_VERSION:
        raise PlanError(f"нужен план версии {PLAN_VERSION}")
    recs = plan.get("recordings")
    if not isinstance(recs, list) or not recs:
        raise PlanError("в плане нет записей")
    seen = set()
    for rec in recs:
        if not isinstance(rec, dict) or not rec.get("audio"):
            raise PlanError("у записи нет поля audio")
        variants = rec.get("variants")
        if not isinstance(variants, list) or not variants:
            raise PlanError(f"у записи {rec['audio']} нет вариантов")
        mains = [v for v in variants if v.get("main")]
        if len(mains) != 1:
            raise PlanError(f"у записи {rec['audio']} должна быть ровно одна основная версия")
        for v in variants:
            t = v.get("transcript")
            if not t or not str(t).endswith(".transcript.json"):
                raise PlanError(f"вариант без стенограммы .transcript.json: {rec['audio']}")
            key = os.path.normcase(os.path.abspath(t))
            if key in seen:
                raise PlanError(f"стенограмма указана дважды: {t}")
            seen.add(key)
            for field in ("conspects", "review_packs"):
                if not isinstance(v.get(field, []), list):
                    raise PlanError(f"{field} должен быть списком: {t}")
        meta = rec.get("metadata", {})
        if not isinstance(meta, dict) or set(meta) - set(METADATA_FIELDS):
            raise PlanError(f"неизвестные поля metadata у {rec['audio']}")
        if not isinstance(rec.get("field_sources", {}), dict):
            raise PlanError(f"field_sources должен быть объектом: {rec['audio']}")
    return plan


def basename_for(transcript_path):
    name = Path(transcript_path).name
    return name[:-len(".transcript.json")]


# ---------------------------------------------------------------- факты об аудио

def audio_facts(audio_path, ffprobe=None):
    """Только то, что следует из самого файла: наличие, SHA-256, контейнер и кодек (если есть ffprobe)."""
    p = Path(audio_path)
    if not p.is_file():
        return {"audio_available": False}
    facts = {"audio_available": True, "audio_sha256": sha256_file(p)}
    ffprobe = ffprobe or shutil.which("ffprobe")
    if ffprobe:
        try:
            out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=format_name:stream=codec_name,codec_type",
                                  "-of", "json", str(p)], capture_output=True, text=True, timeout=60,
                                 stdin=subprocess.DEVNULL)
            info = json.loads(out.stdout or "{}")
            fmt = (info.get("format") or {}).get("format_name")
            codec = next((s.get("codec_name") for s in info.get("streams") or [] if s.get("codec_type") == "audio"), None)
            if fmt:
                facts["container"] = fmt
            if codec:
                facts["codec"] = codec
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return facts


# ---------------------------------------------------------------- преобразование

def anchor_review(transcript, review):
    """Перевести ссылки сомнений «сегмент + номера слов» в word_ids.

    Верная ссылка (w1 < w2 <= число слов) переводится как есть. Устаревшая ссылка
    привязывается по времени, только если место однозначно (см. _anchor_by_time);
    иначе ссылка снимается, сомнение остаётся по времени с пометкой «место не найдено» —
    решает человек. Статус сомнения не меняется.
    """
    segments = transcript.get("segments") or []
    stats = {"по номерам": 0, "по времени": 0, "по времени, текст менялся": 0, "место не найдено": 0,
             "только по времени": 0}
    for item in review.get("review") or []:
        if "seg" not in item:
            stats["только по времени"] += 1
            continue
        si, w1, w2 = item.get("seg"), item.get("w1"), item.get("w2")
        words = (segments[si].get("words") or []) if isinstance(si, int) and 0 <= si < len(segments) else []
        if words and isinstance(w1, int) and isinstance(w2, int) and 0 <= w1 < w2 <= len(words):
            item["word_ids"] = [w["word_id"] for w in words[w1:w2]]
            stats["по номерам"] += 1
            continue
        found = _anchor_by_time(segments, item)
        legacy_ref = {"seg": si, "w1": w1, "w2": w2}
        if found is not None:
            seg_index, a, b, text_same = found
            item.update(seg=seg_index, w1=a, w2=b,
                        word_ids=[w["word_id"] for w in segments[seg_index]["words"][a:b]])
            item["migration"] = {"anchor": "time", "text_same": text_same, "legacy_ref": legacy_ref,
                                 "note": "привязано автоматически по времени" +
                                         ("" if text_same else "; текст места менялся после разбора")}
            stats["по времени" if text_same else "по времени, текст менялся"] += 1
        else:
            for key in ("seg", "w1", "w2", "word_ids"):
                item.pop(key, None)
            item["migration"] = {"anchor": "not_found", "note": "место не найдено — решает человек",
                                 "legacy_ref": legacy_ref}
            stats["место не найдено"] += 1
    return stats


def _anchor_by_time(segments, item):
    """Однозначно по времени: слова этого говорящего внутри [t1, t2] идут подряд и только в одном сегменте.

    Возвращает (сегмент, начало, конец, совпал_ли_текст) или None. Если внутри участка текст сомнения
    встречается несколько раз — место неоднозначно. Если не встречается ни разу — текст правили после
    разбора: привязывается весь участок с пометкой.
    """
    t1, t2 = item.get("t1"), item.get("t2")
    if not isinstance(t1, (int, float)) or not isinstance(t2, (int, float)) or t2 < t1:
        return None
    runs = []
    for si, seg in enumerate(segments):
        if item.get("speaker") and seg.get("speaker") and seg.get("speaker") != item.get("speaker"):
            continue
        words = seg.get("words") or []
        inside = [i for i, w in enumerate(words)
                  if isinstance(w.get("s"), (int, float)) and isinstance(w.get("e"), (int, float))
                  and w["s"] >= t1 - TIME_EPS and w["e"] <= t2 + TIME_EPS]
        if inside:
            runs.append((si, inside))
    if len(runs) != 1:
        return None
    si, inside = runs[0]
    a, b = inside[0], inside[-1] + 1
    if inside != list(range(a, b)):
        return None
    words = segments[si]["words"]
    target = _norm_text(item.get("text"))
    matches = [(x, y) for x in range(a, b) for y in range(x + 1, b + 1)
               if target and _norm_text(" ".join(str(w.get("w", "")) for w in words[x:y])) == target]
    if len(matches) > 1:
        return None
    if len(matches) == 1:
        return si, matches[0][0], matches[0][1], True
    return si, a, b, False


def import_answers(review, pack_answers, pack_name):
    """Ответы из папки разбора: только к существующим сомнениям, без подмены версии вопросов."""
    ids = {item["id"] for item in review.get("review") or []}
    answers = review.setdefault("answers", {})
    imported, skipped, converted = 0, 0, 0
    for ident, answer in (pack_answers or {}).items():
        if ident not in ids or not isinstance(answer, dict) or ident in answers:
            skipped += 1
            continue
        value = copy.deepcopy(answer)
        if "version" in value:
            value["legacy_version"] = value.pop("version")
        if isinstance(value.get("ts"), (int, float)) and not isinstance(value.get("ts"), bool):
            # старый формат — секунды с 1970 года; программа пишет и сравнивает строки
            # «ГГГГ-ММ-ДДTЧЧ:ММ:СС» в UTC (review_pack._now), поэтому — тот же вид, без долей секунды
            value["ts"] = time.strftime(ANSWER_TS_FORMAT, time.gmtime(value["ts"]))
            converted += 1
        if isinstance(value.get("source"), str) and "actor_kind" not in value:
            # кто отвечал, в старом формате не записано: метка источника, не утверждение об авторе
            value["legacy_author_label"] = "старая страница разбора: " + value["source"]
        value["id"] = ident
        value["origin"] = "legacy_review_pack:" + pack_name
        answers[ident] = value
        imported += 1
    return imported, skipped, converted


def build_snapshots(legacy, *, recording_id, metadata, field_sources, variant, source, answers_packs=()):
    """Чистое преобразование старого JSON в пару снимков нового формата (без БД и диска)."""
    import recordings
    import storage
    data = copy.deepcopy(legacy)
    # служебные поля, которые ставит штатное сохранение старого файла (transcript.save): в новом
    # формате их ведёт база; прежние значения сохраняются в сведениях о переносе
    legacy_runtime = {key: data.pop(key) for key in LEGACY_RUNTIME_KEYS if key in data}
    filled = {}
    for key in METADATA_FIELDS:
        if key in metadata and metadata[key] is not None and key not in data:
            data[key] = metadata[key]
            filled[key] = field_sources.get(key, "вычислено при переносе")
    rid, transcript, review = recordings.snapshots_from_worker(data, recording_id)
    stats = {"review": anchor_review(transcript, review)}
    packs = []
    for pack in answers_packs:
        n, skipped, converted = import_answers(review, pack.get("answers"), pack["name"])
        packs.append({"name": pack["name"], "imported": n, "skipped": skipped, "applied_at": pack.get("applied_at"),
                      "ts_converted": converted,
                      "ts_note": "время ответов переведено из секунд с 1970 года в UTC «ГГГГ-ММ-ДДTЧЧ:ММ:СС», "
                                 "доли секунды отброшены" if converted else ""})
    stats["answers"] = packs
    quality = transcript.setdefault("quality", {})
    quality["migration"] = {
        "tool": TOOL, "source": {**source, **({"legacy_runtime": legacy_runtime} if legacy_runtime else {})},
        "field_sources": filled,
        "variant": {k: variant.get(k) for k in ("tag", "order", "main", "main_note", "history") if k in variant},
        "review": stats["review"], "review_packs": packs,
    }
    storage.validate_pair(transcript, review)
    return rid, transcript, review, stats


def legacy_conspect_markdown(conspect_json):
    """Текст старого конспекта с цитатами в «»: заглушки {{q:N}} заменяются исходными цитатами."""
    quotes = conspect_json.get("quotes") or {}
    template = conspect_json.get("template") or ""
    missing = []

    def restore(match):
        q = quotes.get(match.group(1))
        if not isinstance(q, dict) or not q.get("author"):
            missing.append(match.group(1))
            return match.group(0)
        spk = q.get("speaker")
        return f"{spk}: «{q['author']}»" if spk else f"«{q['author']}»"

    text = re.sub(r"\{\{q:([^}]+)\}\}", restore, template)
    if missing:
        raise ValueError("в конспекте нет цитат для заглушек: " + ", ".join(missing[:5]))
    return text


def conspect_link_preview(template, transcript, review, recording_id):
    import conspect
    data = {**transcript, **review, "recording_id": recording_id, "source_revision": None}
    _, rep = conspect.link(template, data)
    return rep


# ---------------------------------------------------------------- чтение плана

def _read_variant(rec, variant):
    t = Path(variant["transcript"])
    legacy = json.loads(t.read_text(encoding="utf-8-sig"))
    source = {"path": str(t), "sha256": sha256_file(t), "size": t.stat().st_size}
    packs = []
    for pack_dir in variant.get("review_packs", []):
        pd = Path(pack_dir)
        # текущие ответы и архив прежних версий карточек (answers.<версия>.json): новые — первыми,
        # прежние добавляются только к сомнениям без более нового ответа
        files = [pd / "answers.json"] + sorted(pd.glob("answers.*.json"), key=lambda p: p.name, reverse=True)
        for f in files:
            if not f.is_file():
                continue
            ans = json.loads(f.read_text(encoding="utf-8-sig"))
            name = pd.name if f.name == "answers.json" else f"{pd.name}/{f.name}"
            packs.append({"name": name, "answers": ans.get("answers") or {}, "applied_at": ans.get("applied_at")})
    conspects = []
    for c in variant.get("conspects", []):
        cj = json.loads(Path(c).read_text(encoding="utf-8-sig"))
        conspects.append({"path": str(c), "name": cj.get("name"), "template": legacy_conspect_markdown(cj)})
    return legacy, source, packs, conspects


def _record_metadata(rec, facts):
    meta = dict(facts)
    meta.update({k: v for k, v in (rec.get("metadata") or {}).items() if v is not None})
    sources = {"audio_available": "наличие файла аудио", "audio_sha256": "вычислено по файлу аудио",
               "container": "ffprobe по файлу аудио", "codec": "ffprobe по файлу аудио"}
    sources.update(rec.get("field_sources") or {})
    return meta, sources


def run(plan, mode, ffprobe=None):
    """mode: check (в памяти) или apply (запись). Возвращает отчёт по каждой стенограмме."""
    import documents
    import recordings
    store = recordings.open_store() if mode == "apply" else None
    report = []
    try:
        for rec in plan["recordings"]:
            facts = audio_facts(rec["audio"], ffprobe)
            meta, sources = _record_metadata(rec, facts)
            for variant in sorted(rec["variants"], key=lambda v: v.get("order", 0)):
                entry = {"audio": rec["audio"], "transcript": variant["transcript"], "main": bool(variant.get("main"))}
                try:
                    legacy, source, packs, conspects = _read_variant(rec, variant)
                    rid = recording_id_for(variant["transcript"], source["sha256"])
                    entry["recording_id"] = rid
                    _, transcript, review, stats = build_snapshots(
                        legacy, recording_id=rid, metadata=meta, field_sources=sources,
                        variant=variant, source=source, answers_packs=packs)
                    entry.update(stats)
                    entry["conspects"] = [{"name": c["name"], **conspect_link_preview(c["template"], transcript, review, rid)}
                                          for c in conspects]
                    if mode == "apply":
                        entry.update(_apply_one(store, rec, variant, legacy, transcript, review, rid, meta, sources,
                                                source, packs, conspects, documents, recordings))
                    entry["status"] = entry.get("status", "ok")
                except Exception as ex:  # одна запись не должна останавливать отчёт по остальным
                    entry["status"] = "error"
                    entry["error"] = f"{type(ex).__name__}: {ex}"
                report.append(entry)
    finally:
        if store is not None:
            store.close()
    return report


def _apply_one(store, rec, variant, legacy, transcript, review, rid, meta, sources, source, packs, conspects,
               documents, recordings):
    import conspect
    if store.db.execute("SELECT 1 FROM recordings WHERE recording_id=?", (rid,)).fetchone():
        return {"status": "skipped", "note": "уже перенесено ранее"}
    tag = basename_for(variant["transcript"])[len(Path(rec["audio"]).stem):].lstrip(".")
    root = recordings.document_layout(rec["audio"], tag)["root"]
    root.mkdir(exist_ok=True)
    names = documents.paths_for_basename(basename_for(variant["transcript"]))
    wanted = {kind: f"{DOC_SUBDIR}/{name}" for kind, name in names.items()}
    root_id, doc_paths = recordings.prepare_new_layout(store, root, wanted)
    context = recordings.operation_context(store, "system")
    operation_id = str(uuid.uuid5(NAMESPACE, "legacy-import:" + rid))
    store.create_recording_with_document(
        rid, recordings.normalize_location(rec["audio"]), transcript, review, operation_id, context,
        root_id=root_id, generated_at=datetime.now(timezone.utc).isoformat(), document_paths=doc_paths)
    documents.materialize_pending(store, root_id, root)
    result = {"status": "imported", "documents": {k: str(root / v) for k, v in doc_paths.items()}, "conspects_saved": []}
    json_path = root / doc_paths["json"]
    for c in conspects:
        if not c["name"]:
            result["conspects_saved"].append({"source": c["path"], "error": "у конспекта нет имени"})
            continue
        saved = conspect.save_linked(json_path, c["name"], c["template"])
        result["conspects_saved"].append({"source": c["path"], "conspect_json": saved.get("conspect_json"),
                                          "unlinked": len(saved.get("unlinked") or [])})
    return result


# ---------------------------------------------------------------- голосовые профили

def mark_voices(plan, apply=False):
    """Пометить фрагменты профилей: кто подтвердил (решение владельца из плана), не меняя отпечатки."""
    import core
    import paths
    import settings
    import voices
    out = []
    for group, decision in (plan.get("voice_confirmation") or {}).items():
        by = decision.get("by")
        if by not in ("human", "program"):
            raise PlanError(f"voice_confirmation.{group}.by: human или program")
        path = paths.voices_path(group)
        with settings.group_voice_lock(group):
            prof = voices.load_profile(path)
            changed = 0
            for speaker in prof["speakers"].values():
                for part in speaker.get("parts", {}).values():
                    mark = {"by": by, "note": decision.get("note", ""), "decided": decision.get("decided", "")}
                    if part.get("confirmation") != mark:
                        part["confirmation"] = mark
                        changed += 1
            backup = None
            if apply and changed:
                backup = core.backup_voice_profile(path)
                voices.save_profile(path, prof)
        out.append({"group": group, "by": by, "parts_changed": changed, "applied": bool(apply and changed),
                    "backup": backup})
    return out


# ---------------------------------------------------------------- командная строка

def _summary(report):
    lines = []
    for e in report:
        rv = e.get("review") or {}
        cons = e.get("conspects") or []
        lines.append(f"{'*' if e['main'] else ' '} {e['status']:8} {Path(e['transcript']).name}: "
                     f"сомнения {rv}; ответы {[(p['name'], p['imported']) for p in e.get('answers', [])]}; "
                     f"конспекты {[(c.get('name'), c.get('linked'), len(c.get('unlinked') or []), len(c.get('candidates') or [])) for c in cons]}"
                     + (f"; ошибка: {e['error']}" if e.get("error") else ""))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["check", "apply", "mark-voices"])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--report")
    ap.add_argument("--apply", action="store_true", help="для mark-voices: записать изменения")
    a = ap.parse_args(argv)
    try:
        plan = load_plan(a.plan)
    except PlanError as ex:
        print(f"ошибка плана: {ex}", file=sys.stderr)
        return 2
    if a.command == "mark-voices":
        result = mark_voices(plan, apply=a.apply)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0
    try:
        report = run(plan, a.command)
    except Exception as ex:
        print(f"окружение не готово: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 2
    if a.report:
        Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(_summary(report))
    return 1 if any(e["status"] == "error" for e in report) else 0


if __name__ == "__main__":
    sys.exit(main())
