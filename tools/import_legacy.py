"""Перенос записей старого файлового формата в SQLite-хранилище (схема v4).

Решения владельца (даты записей, основная версия, вид подтверждения голосов) задаются
в файле плана, а не угадываются здесь. Старые файлы (`*.transcript.*`, конспекты,
папки разбора) не изменяются: новые документы формируются в подпапке `v4` рядом.

    python tools/import_legacy.py check --plan plan.json [--out папка-отчётов]
        проверка в памяти: ничего не пишет ни в БД, ни на диск (кроме отчётов)
    python tools/import_legacy.py apply --plan plan.json --data-dir <папка данных> --out папка-отчётов
        запись в БД папки данных программы (RUS_TRANSCRIBER_DATA; --data-dir должен с ней совпадать);
        БД создаётся заранее штатной операцией initialize_recording_store; повторный запуск пропускает
        уже перенесённое, а тот же recording_id из изменившейся стенограммы — конфликт, не перезапись.
        После записи — проверка набора и комплект inventory/plan/mapping/verification/report
        со статусом verified_copy или partial
    python tools/import_legacy.py mark-voices --plan plan.json [--apply]
        пометить, кем подтверждены фрагменты голосовых профилей; без --apply — только показать;
        перед изменением профиль копируется в voice-backups
    python tools/import_legacy.py repair-packs --plan plan.json [--out папка] [--apply --data-dir <папка данных>]
        перенести состояние пакета карточек в записи, перенесённые без него (М-1); без --apply — только отчёт;
        с --apply — сначала резерв базы в db-backups, затем одна новая ревизия разбора на годную запись
    python tools/import_legacy.py repair-packs --db-only [--out папка]
        то же без плана и без чтения файлов рядом с аудио: годность записей только по базе, ничего не пишет

Код возврата: 0 — без ошибок, 1 — есть записи с ошибками или отказами, 2 — неверный план, окружение или резерв.
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

TOOL = "tools/import_legacy.py v2"     # v2: состояние пакета карточек переносится целиком (М-1)
REPAIR_TOOL = "tools/import_legacy.py repair-packs v1"
NAMESPACE = uuid.UUID("6b1d3c5e-8f0a-4c11-9d6e-2a7f4b9e0c31")
MAX_INTERACTION_EVENTS = 2000           # как storage.MAX_INTERACTION_EVENTS и review_pack.MAX_INTERACTION_EVENTS
CLOSED_STATUSES = ("resolved", "human")
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


def recording_id_from(legacy, source_path, source_sha):
    """recording_id записи: старый сохраняется (uuid4.hex → тот же UUID), иначе вычисляется.

    -> (recording_id, правило). Правило попадает в таблицу соответствия mapping.json.
    """
    old = legacy.get("recording_id")
    if isinstance(old, str) and old:
        try:
            return str(uuid.UUID(old)), "сохранён старый recording_id"
        except ValueError:
            pass
    return recording_id_for(source_path, source_sha), "вычислен из пути и SHA-256 стенограммы (uuid5)"


def import_questions(review, transcript, manifests):
    """Вопросы карточек из манифестов папок разбора: новые версии первыми, один вопрос на сомнение.

    Ссылки word_ids берутся, только если все такие слова есть в стенограмме.
    """
    ids = {item["id"] for item in review.get("review") or []}
    word_ids = {w["word_id"] for s in transcript.get("segments") or [] for w in s.get("words") or []}
    questions = review.setdefault("questions", [])
    have = {q.get("id") for q in questions}
    added = 0
    for man in manifests:
        for item in man.get("items") or []:
            rid = item.get("id") if isinstance(item, dict) else None
            if rid not in ids or rid in have:
                continue
            q = {"id": rid, "question": str(item.get("question") or ""), "text": str(item.get("text") or ""),
                 "legacy_version": man.get("version"), "origin": "legacy_review_pack:" + man["name"]}
            refs = item.get("word_ids")
            if isinstance(refs, list) and refs and len(set(refs)) == len(refs) and set(refs) <= word_ids:
                q["word_ids"] = list(refs)
            questions.append(q)
            have.add(rid)
            added += 1
    if not questions:
        review.pop("questions", None)
    return added


def md_check(transcript_path, legacy):
    """Совпадает ли старый .transcript.md с рендером его JSON. Расхождение — повод посмотреть человеку:
    это может быть ручная правка MD или просто другая версия программы при рендере."""
    import transcript as T
    md = Path(str(transcript_path)[:-len(".json")] + ".md")
    if not md.is_file():
        return {"md": str(md), "exists": False}
    current = md.read_bytes()
    try:
        rendered = T.render_md(copy.deepcopy(legacy)).encode("utf-8")
    except Exception as ex:  # рендер старого файла может не поддерживаться — это сведения, не ошибка переноса
        return {"md": str(md), "exists": True, "sha256": hashlib.sha256(current).hexdigest(),
                "same": None, "note": f"рендер не выполнен: {type(ex).__name__}"}
    same = current.replace(b"\r\n", b"\n") == rendered.replace(b"\r\n", b"\n")   # переносы строк не смысл
    return {"md": str(md), "exists": True, "sha256": hashlib.sha256(current).hexdigest(),
            "rendered_sha256": hashlib.sha256(rendered).hexdigest(), "same": same}


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
    """Только то, что следует из самого файла: наличие, SHA-256, контейнер и кодек (если есть ffprobe).
    Общее правило с расшифровкой — runinfo.audio_facts."""
    import runinfo
    return runinfo.audio_facts(audio_path, ffprobe)


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


def _utc_text(value):
    """Время старого формата (секунды с 1970 года) — в строку UTC «ГГГГ-ММ-ДДTЧЧ:ММ:СС», как пишет и посимвольно
    сравнивает программа (review_pack._now); доли секунды отбрасываются. Строка и пустое значение — как есть."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return time.strftime(ANSWER_TS_FORMAT, time.gmtime(value))
    return value


def import_answers(review, pack_answers, pack_name, keep_version=False):
    """Ответы из папки разбора: только к существующим сомнениям.

    keep_version=False (ответы без перенесённого пакета, прежний перенос): версия пакета — в legacy_version,
    вопросы не подменяются. keep_version=True — ответы текущего пакета, который переносится целиком: версия
    остаётся version и равна question_version разбора (storage.validate_pair)."""
    ids = {item["id"] for item in review.get("review") or []}
    answers = review.setdefault("answers", {})
    imported, skipped, converted = 0, 0, 0
    for ident, answer in (pack_answers or {}).items():
        if ident not in ids or not isinstance(answer, dict) or ident in answers:
            skipped += 1
            continue
        value = copy.deepcopy(answer)
        if "version" in value and not keep_version:
            value["legacy_version"] = value.pop("version")
        if isinstance(value.get("ts"), (int, float)) and not isinstance(value.get("ts"), bool):
            # старый формат — секунды с 1970 года; программа пишет и сравнивает строки
            # «ГГГГ-ММ-ДДTЧЧ:ММ:СС» в UTC (review_pack._now), поэтому — тот же вид, без долей секунды
            value["ts"] = _utc_text(value["ts"])
            converted += 1
        if isinstance(value.get("source"), str) and "actor_kind" not in value:
            # кто отвечал, в старом формате не записано: метка источника, не утверждение об авторе
            value["legacy_author_label"] = "старая страница разбора: " + value["source"]
        value["id"] = ident
        value["origin"] = "legacy_review_pack:" + pack_name
        answers[ident] = value
        imported += 1
    return imported, skipped, converted


def _db_revision(legacy_revision, legacy_current):
    """Ревизия старой стенограммы, по которой собран пакет или карточка, — в ревизию базы. 1 — та же ревизия, что
    перенесена (первая ревизия базы); 0 — другая или неизвестная: такой ревизии в базе нет, и неприменённый ответ
    получит честный отказ «недоступна исходная ревизия», а не применится к изменившемуся тексту."""
    if legacy_revision is not None and legacy_revision == legacy_current:
        return 1
    return 0


def _pack_digest(packs, manifests):
    """Отпечаток файлов пакета (ответы и вопросы как прочитаны) — для операции ремонта и сверки."""
    rows = [[p["name"], p.get("raw", {"answers": p.get("answers"), "applied_at": p.get("applied_at")})] for p in packs]
    rows += [[m["name"], {k: v for k, v in m.items() if k not in ("name", "current")}] for m in manifests]
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def pack_state_missing(review):
    """Признак М-1 в снимке разбора: состояние пакета карточек не перенесено (нет версии вопросов и пакета)."""
    return review.get("question_version") is None and not (review.get("undo") or {}).get("review_pack")


def _prune_word_ids(item, word_ids):
    refs = item.get("word_ids")
    if refs is None:
        return False
    if isinstance(refs, list) and refs and len(set(refs)) == len(refs) and set(refs) <= word_ids:
        return False
    item.pop("word_ids")
    return True


def pack_state_from_legacy(review, transcript, packs, manifests, recording_id, legacy_runtime=None):
    """Состояние пакета карточек старого формата — в снимок разбора, как его читает recordings.read_review_pack_state
    (решение владельца по В-1 от 26.11.2026: «Перенос целиком»). Чистое преобразование, без БД и диска.

    packs — файлы ответов ({name, answers, applied_at, raw, current}), manifests — файлы вопросов ({name, …,
    current}); current — answers.json и manifest.json первой папки разбора варианта. Текущий пакет переносится
    целиком, если оба файла есть и версии совпадают. Не переносятся (решения Архитектора 26.11.2026): снимки отмены
    (Р-1: отпечаток состояния после переноса им не совпадёт, число — в отчёте) и клипы (Р-3: остаются в старой
    папке). Ответы и вопросы архивов и пакета, который нельзя перенести, — в undo.legacy_review_packs.files (Р-2): программа
    их не читает, и пересборку они не блокируют. Ответы закрытых мест без отметки «применён» остаются ждущими (Р-4
    решит владелец) — только считаются. -> сведения для отчёта."""
    import review_pack as RP
    word_ids = {w["word_id"] for s in transcript.get("segments") or [] for w in s.get("words") or []}
    review_ids = {item["id"] for item in review.get("review") or []}
    current = next((p for p in packs if p.get("current")), None)
    man = next((m for m in manifests if m.get("current")), None)
    stats = {"state": "none", "answers_current": 0, "answers_archived": 0, "ts_converted": 0}
    if not packs and not manifests:
        return stats
    raw = (current or {}).get("raw") or {}
    version = raw.get("version")
    if current is None:
        reason = "нет answers.json текущего пакета"
    elif man is None:
        reason = "нет manifest.json текущего пакета"
    elif not isinstance(version, str) or version != man.get("version"):
        reason = "версии answers.json и manifest.json текущего пакета не совпадают"
    else:
        reason = None
    full = reason is None
    had_undo = "undo" in review
    undo = review.setdefault("undo", {})
    archive = []
    if full:
        other_version = {aid: a for aid, a in (raw.get("answers") or {}).items()
                         if isinstance(a, dict) and "version" in a and a["version"] != version}
        mine = {aid: a for aid, a in (raw.get("answers") or {}).items() if aid not in other_version}
        n, skipped, converted = import_answers(review, mine, current["name"], keep_version=True)
        answers = review["answers"]
        if other_version:
            archive.append({"file": current["name"], "kind": "answers_other_version", "version": version,
                            "answers": copy.deepcopy(other_version)})
        manifest = copy.deepcopy({k: v for k, v in man.items() if k not in ("name", "current")})
        legacy_current = (legacy_runtime or {}).get("source_revision")
        pruned = 0
        for item in manifest.get("items") or []:
            pruned += _prune_word_ids(item, word_ids)
            if "source_revision" in item:          # карточка, заменённая или дописанная позже сборки (upsert_cards)
                item["legacy_source_revision"] = item["source_revision"]
                item["source_revision"] = _db_revision(item["source_revision"], legacy_current)
        manifest["legacy_recording_id"] = manifest.get("recording_id")
        manifest["recording_id"] = recording_id
        manifest["legacy_source_revision"] = manifest.get("source_revision")
        manifest["source_revision"] = _db_revision(manifest.get("source_revision"), legacy_current)
        manifest["legacy_job_json"] = manifest.pop("job_json", None)    # новый путь ставит запись документа
        questions, seen = [], set()
        for item in manifest.get("items") or []:
            if item.get("id") in review_ids and item.get("id") not in seen:
                seen.add(item["id"])
                questions.append(copy.deepcopy(item))
        review["question_version"] = version
        review["questions"] = questions
        review["conflicts"] = copy.deepcopy(raw.get("conflicts") or [])
        review["rejected"] = copy.deepcopy(raw.get("rejected") or [])
        applied_at = _utc_text(raw.get("applied_at"))
        applied, marks = {}, {"kept": 0, "from_true": 0, "dropped": 0, "from_applied_at": 0}
        if "applied_answers" in raw:
            for aid, sig in (raw.get("applied_answers") or {}).items():
                answer = answers.get(aid)
                if answer is not None and isinstance(sig, list):
                    applied[aid] = sig
                    marks["kept"] += 1
                elif answer is not None and sig is True and applied_at and (answer.get("ts") or "") <= applied_at:
                    # старая отметка «применён» без подписи: подпись ответа, который был до применения пакета
                    applied[aid] = RP._sig(answer)
                    marks["from_true"] += 1
                else:
                    marks["dropped"] += 1
        elif applied_at:
            # пакет до записи отметок: применёнными считались ответы не позже applied_at (review_pack._applied_map);
            # в базе состояние всегда с applied_answers, поэтому правило переносится явными подписями
            for aid, answer in answers.items():
                if (answer.get("ts") or "") <= applied_at:
                    applied[aid] = RP._sig(answer)
                    marks["from_applied_at"] += 1
        events = copy.deepcopy(raw.get("application_events") or [])
        for row in events:
            # время в журнале применения — как у ответа (по answer_ts событие находит свой ответ)
            for key in ("answer_ts", "at"):
                if isinstance(row, dict) and key in row:
                    row[key] = _utc_text(row[key])
        pack = {"manifest": manifest, "applied_answers": applied, "applied_at": applied_at, "undo_snapshots": {},
                "application_events": events,
                "application_events_dropped": raw.get("application_events_dropped", 0)}
        if raw.get("outcome_details"):
            pack["outcome_details"] = copy.deepcopy(raw["outcome_details"])
        if isinstance(raw.get("heard"), dict):
            pack["heard"] = copy.deepcopy(raw["heard"])
        undo["review_pack"] = pack
        review["applications"] = copy.deepcopy(events)
        evicted = 0
        if "interaction_events" in raw:
            seen_events = copy.deepcopy(raw.get("interaction_events") or [])
            evicted = max(0, len(seen_events) - MAX_INTERACTION_EVENTS)
            review["interaction_events"] = seen_events[evicted:]
            review["interaction_events_evicted"] = raw.get("interaction_events_dropped", 0) + evicted
        state = {"answers": answers, "applied_answers": applied, "application_events": events,
                 "undo_snapshots": {}, "rejected": review["rejected"],
                 "outcome_details": pack.get("outcome_details")}
        waiting = RP.rebuild_check(state, manifest)["waiting"]
        status = {item["id"]: item.get("status") or "open" for item in review.get("review") or []}
        stats.update(state="full", version=version, answers_current=n, answers_skipped=skipped,
                     answers_other_version=len(other_version), ts_converted=converted,
                     manifest_items=len(manifest.get("items") or []), questions=len(questions),
                     word_ids_pruned=pruned, source_revision=manifest["source_revision"],
                     applied=len(applied), applied_marks=marks,
                     undo_snapshots_dropped=len(raw.get("undo_snapshots") or {}),
                     application_events=len(events), interaction_events=len(review.get("interaction_events") or []),
                     interaction_events_evicted=evicted, heard=isinstance(raw.get("heard"), dict),
                     waiting=len(waiting),
                     waiting_closed_place=sum(status.get(w["id"]) in CLOSED_STATUSES for w in waiting))
    else:
        stats.update(state="archived_only", reason=reason)
    for p in packs:
        if full and p is current:
            continue
        body = p.get("raw") or {"answers": p.get("answers"), "applied_at": p.get("applied_at")}
        archive.append({"file": p["name"], "kind": "answers", "version": body.get("version"),
                        **{k: copy.deepcopy(body[k]) for k in ("answers", "applied_at", "applied_answers", "conflicts",
                                                               "rejected") if k in body}})
    for m in manifests:
        if full and m is man:
            continue
        archive.append({"file": m["name"], "kind": "manifest", "version": m.get("version"),
                        "questions": copy.deepcopy(m.get("items") or [])})
    if archive:
        # запись undo — объект (documents._public_review); программа это поле не читает
        undo["legacy_review_packs"] = {"files": archive}
    elif not undo and not had_undo:
        review.pop("undo", None)
    stats["answers_archived"] = sum(len(row.get("answers") or {}) for row in archive)
    stats["archived_files"] = len(archive)
    return stats


def build_snapshots(legacy, *, recording_id, metadata, field_sources, variant, source, answers_packs=(),
                    manifests=()):
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
    pack_stats = pack_state_from_legacy(review, transcript, list(answers_packs), list(manifests), rid, legacy_runtime)
    packs = []
    for pack in answers_packs:
        moved = pack_stats["state"] == "full" and pack.get("current")
        converted = pack_stats["ts_converted"] if moved else 0
        packs.append({"name": pack["name"], "imported": pack_stats["answers_current"] if moved else 0,
                      "archived": not moved, "applied_at": pack.get("applied_at"), "ts_converted": converted,
                      "ts_note": "время ответов переведено из секунд с 1970 года в UTC «ГГГГ-ММ-ДДTЧЧ:ММ:СС», "
                                 "доли секунды отброшены" if converted else ""})
    stats["answers"] = packs
    stats["questions"] = len(review.get("questions") or [])
    stats["pack_state"] = pack_stats
    quality = transcript.setdefault("quality", {})
    quality["migration"] = {
        "tool": TOOL, "source": {**source, **({"legacy_runtime": legacy_runtime} if legacy_runtime else {})},
        "baseline": {"legacy_source_revision": legacy_runtime.get("source_revision"),
                     "note": "первая ревизия базы — снимок стенограммы старого формата; прежняя история правок "
                             "доступна только в журнале edits и старых файлах"},
        "field_sources": filled,
        "variant": {k: variant.get(k) for k in ("tag", "order", "main", "main_note", "history") if k in variant},
        "review": stats["review"], "review_packs": packs, "questions_imported": stats["questions"],
        "pack_state": pack_stats,
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

def _read_pack_files(pack_dirs):
    """Файлы пакетов разбора: только answers*.json и manifest*.json (клипы и voice/ не читаются).

    Текущий пакет (current) — answers.json и manifest.json первой папки; архивы прежних версий карточек
    (answers.<версия>.json, manifest.<версия>.json) и другие папки — после него, новые первыми."""
    packs, manifests = [], []
    for index, pack_dir in enumerate(pack_dirs):
        pd = Path(pack_dir)
        files = [pd / "answers.json"] + sorted(pd.glob("answers.*.json"), key=lambda p: p.name, reverse=True)
        for f in files:
            if not f.is_file():
                continue
            ans = json.loads(f.read_text(encoding="utf-8-sig"))
            name = pd.name if f.name == "answers.json" else f"{pd.name}/{f.name}"
            packs.append({"name": name, "answers": ans.get("answers") or {}, "applied_at": ans.get("applied_at"),
                          "raw": ans, "current": index == 0 and f.name == "answers.json"})
    for index, pack_dir in enumerate(pack_dirs):
        pd = Path(pack_dir)
        files = [pd / "manifest.json"] + sorted(pd.glob("manifest.*.json"), key=lambda p: p.name, reverse=True)
        for f in files:
            if f.is_file():
                man = json.loads(f.read_text(encoding="utf-8-sig"))
                man["name"] = pd.name if f.name == "manifest.json" else f"{pd.name}/{f.name}"
                man["current"] = index == 0 and f.name == "manifest.json"
                manifests.append(man)
    return packs, manifests


def _read_variant(rec, variant):
    t = Path(variant["transcript"])
    legacy = json.loads(t.read_text(encoding="utf-8-sig"))
    source = {"path": str(t), "sha256": sha256_file(t), "size": t.stat().st_size}
    packs, manifests = _read_pack_files(variant.get("review_packs", []))
    conspects = []
    for c in variant.get("conspects", []):
        cj = json.loads(Path(c).read_text(encoding="utf-8-sig"))
        conspects.append({"path": str(c), "name": cj.get("name"), "template": legacy_conspect_markdown(cj)})
    return legacy, source, packs, conspects, manifests


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
                    legacy, source, packs, conspects, manifests = _read_variant(rec, variant)
                    rid, rule = recording_id_from(legacy, variant["transcript"], source["sha256"])
                    entry.update(recording_id=rid, recording_id_rule=rule, source_sha256=source["sha256"],
                                 legacy_recording_id=legacy.get("recording_id"),
                                 legacy_source_revision=legacy.get("source_revision"),
                                 md=md_check(variant["transcript"], legacy))
                    _, transcript, review, stats = build_snapshots(
                        legacy, recording_id=rid, metadata=meta, field_sources=sources,
                        variant=variant, source=source, answers_packs=packs, manifests=manifests)
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


def _set_pack_document(review, json_path):
    """Перенесённый manifest указывает на документ записи в базе (как пишет review_pack.build)."""
    manifest = ((review.get("undo") or {}).get("review_pack") or {}).get("manifest")
    if isinstance(manifest, dict):
        manifest["job_json"] = str(json_path)


def _apply_one(store, rec, variant, legacy, transcript, review, rid, meta, sources, source, packs, conspects,
               documents, recordings):
    """Запись одной стенограммы в БД и формирование её документов в подпапке v4."""
    import conspect
    if store.db.execute("SELECT 1 FROM recordings WHERE recording_id=?", (rid,)).fetchone():
        stored = store.get_current(rid)["transcript"]["payload"]
        stored_sha = ((stored.get("quality") or {}).get("migration") or {}).get("source", {}).get("sha256")
        if stored_sha == source["sha256"]:
            out = {"status": "skipped", "note": "уже перенесено ранее"}
            if packs and pack_state_missing(store.get_current(rid)["review"]["payload"]):
                # перенесено прежней версией без состояния пакета карточек (М-1): повторный apply его не трогает
                out.update(pack_state="missing", hint="состояние пакета карточек не перенесено: repair-packs")
            return out
        # тот же ID, другой вход: не перезаписывается молча
        return {"status": "conflict", "note": "запись с этим recording_id уже есть, но перенесена из другой "
                                              "версии стенограммы; повторный перенос не выполняется"}
    tag = basename_for(variant["transcript"])[len(Path(rec["audio"]).stem):].lstrip(".")
    root = recordings.document_layout(rec["audio"], tag)["root"]
    root.mkdir(exist_ok=True)
    names = documents.paths_for_basename(basename_for(variant["transcript"]))
    wanted = {kind: f"{DOC_SUBDIR}/{name}" for kind, name in names.items()}
    root_id, doc_paths = recordings.prepare_new_layout(store, root, wanted)
    _set_pack_document(review, root / doc_paths["json"])
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


# ---------------------------------------------------------------- проверка набора и отчёты

def _compare(legacy, data):
    """Перенесённое содержание против старого файла: текст, говорящие, сомнения, правки, имена."""
    words = lambda d: [w.get("w") for s in d.get("segments") or [] for w in s.get("words") or []]  # noqa: E731
    return {
        "слова": words(legacy) == words(data),
        "сегменты": [(s.get("speaker"), s.get("text")) for s in legacy.get("segments") or []] ==
                    [(s.get("speaker"), s.get("text")) for s in data.get("segments") or []],
        "сомнения": [(r.get("id"), r.get("status")) for r in legacy.get("review") or []] ==
                    [(r.get("id"), r.get("status")) for r in data.get("review") or []],
        "правки": legacy.get("edits") == data.get("edits"),
        "имена": (legacy.get("speaker_names"), legacy.get("names_confirmed")) ==
                 (data.get("speaker_names"), data.get("names_confirmed")),
    }


def verify_set(report, out_dir):
    """Проверка записанного набора: целостность БД, документы, содержание, исходный снимок, резерв."""
    import documents
    import paths
    import recordings
    import storage
    v = {"checks": {}, "records": []}
    try:
        storage.validate_copy(paths.STATE_DB)
        v["checks"]["база: integrity_check и foreign_key_check"] = True
    except Exception as ex:
        v["checks"]["база: integrity_check и foreign_key_check"] = False
        v["db_error"] = f"{type(ex).__name__}: {ex}"
    store = storage.open_store_readonly(paths.STATE_DB)
    try:
        roots = store.db.execute("SELECT root_id, path FROM allowed_roots").fetchall()
        bad_roots = [Path(p).name for rid_, p in roots if not documents.verify_files(store, rid_, p)["consistent"]]
        v["checks"]["документы согласованы с базой (verify_files)"] = not bad_roots
        if bad_roots:
            v["inconsistent_roots"] = bad_roots
    finally:
        store.close()
    store = recordings.open_store()
    try:
        for e in report:
            if e.get("status") not in ("imported", "skipped"):
                continue
            legacy = json.loads(Path(e["transcript"]).read_text(encoding="utf-8-sig"))
            data = recordings.load_recording(store, e["recording_id"])
            checks = _compare(legacy, data)
            baseline = store.get_revision(e["recording_id"], "transcript", 1)["payload"]
            checks["исходный снимок = старый файл"] = (
                ((baseline.get("quality") or {}).get("migration") or {}).get("source", {}).get("sha256") == e["source_sha256"])
            docs = e.get("documents") or {}
            readable = True
            for kind, path in docs.items():
                p = Path(path)
                try:
                    readable &= p.is_file() and (json.loads(p.read_text(encoding="utf-8")) is not None
                                                  if kind in ("json", "manifest") else bool(p.read_text(encoding="utf-8").strip()))
                except (OSError, ValueError):
                    readable = False
            checks["документы читаются без базы"] = readable if docs else None
            v["records"].append({"transcript": Path(e["transcript"]).name, "recording_id": e["recording_id"],
                                 "checks": checks})
    finally:
        store.close()
    rehearsal = Path(out_dir) / "rehearsal" / "state.sqlite3"
    rehearsal.parent.mkdir(parents=True, exist_ok=True)
    store = recordings.open_store()
    try:
        store.backup(rehearsal)
    finally:
        store.close()
    try:
        storage.validate_copy(rehearsal)
        v["checks"]["резервная копия набора создаётся и проходит проверку"] = True
    except Exception as ex:
        v["checks"]["резервная копия набора создаётся и проходит проверку"] = False
        v["rehearsal_error"] = f"{type(ex).__name__}: {ex}"
    v["checks"]["содержание всех записей совпадает со старыми файлами"] = all(
        ok is not False for r in v["records"] for ok in r["checks"].values())
    return v


def _status(mode, report, verification):
    if mode == "check":
        return "checked"
    if any(e.get("status") in ("error", "conflict") for e in report):
        return "partial"
    if verification and all(ok is not False for ok in verification["checks"].values()):
        return "verified_copy"
    return "partial"


def write_artifacts(out_dir, plan, report, mode, verification=None):
    """inventory / plan / mapping / verification / report — приватные пути, только на этом компьютере."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inventory = []
    for e in report:
        inventory.append({"transcript": e["transcript"], "sha256": e.get("source_sha256"), "md": e.get("md"),
                          "status": e.get("status")})
    mapping = [{"transcript": e["transcript"], "source_sha256": e.get("source_sha256"),
                "legacy_recording_id": e.get("legacy_recording_id"), "recording_id": e.get("recording_id"),
                "rule": e.get("recording_id_rule"), "legacy_source_revision": e.get("legacy_source_revision"),
                "database_revision": 1 if e.get("status") == "imported" else None} for e in report]
    status = _status(mode, report, verification)
    files = {"inventory.json": inventory, "plan.json": plan, "mapping.json": mapping, "report.json": report,
             "verification.json": {"status": status, "activation": "not_performed", "mode": mode, "tool": TOOL,
                                   **(verification or {})}}
    for name, value in files.items():
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {}
    for e in report:
        counts[e.get("status")] = counts.get(e.get("status"), 0) + 1
    md_diff = [Path(e["transcript"]).name for e in report if (e.get("md") or {}).get("same") is False]
    lines = [f"# Перенос старых записей — {mode}", "", f"Статус: **{status}**; активация: не выполнялась.", "",
             f"Стенограмм: {len(report)}; итоги: {counts}.", ""]
    if verification:
        lines += ["## Проверки", ""] + [f"- {k}: {'да' if ok else 'НЕТ'}" for k, ok in verification["checks"].items()] + [""]
    if md_diff:
        lines += ["## MD отличается от рендера JSON (посмотреть человеку)", ""] + [f"- {n}" for n in md_diff] + [""]
    problems = [e for e in report if e.get("status") in ("error", "conflict")]
    if problems:
        lines += ["## Ошибки и конфликты", ""] + [f"- {Path(e['transcript']).name}: {e.get('error') or e.get('note')}"
                                                 for e in problems] + [""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return status


# ---------------------------------------------------------------- ремонт: пакет карточек (М-1)

# Состояния записи в отчёте ремонта. Отказы дают код 1; остальное — штатный исход.
REPAIR_REFUSALS = frozenset(("not_migrated", "changed_after_import", "pack_files_changed", "error"))


class RepairBackupError(RuntimeError):
    """Резерв базы перед ремонтом не создан или не прошёл проверку: ни одна запись не меняется."""


def _import_operation(rid):
    return str(uuid.uuid5(NAMESPACE, "legacy-import:" + rid))


def _migrated_records(store):
    """{recording_id: SHA-256 старой стенограммы} записей, перенесённых П7 (метка в первой ревизии стенограммы)."""
    out = {}
    for row in store.db.execute("SELECT recording_id FROM recordings ORDER BY recording_id").fetchall():
        rid = row[0]
        mig = (store.get_revision(rid, "transcript", 1)["payload"].get("quality") or {}).get("migration")
        if mig:
            out[rid] = (mig.get("source") or {}).get("sha256")
    return out


def _db_check(store, rid):
    """Годность записи к ремонту по одной базе — без файлов рядом с аудио. -> сведения и status:
    already_repaired, changed_after_import, nothing_to_repair или candidate (решает сверка с файлами пакета)."""
    current = store.get_current(rid)
    transcript, review = current["transcript"]["payload"], current["review"]["payload"]
    mig = (store.get_revision(rid, "transcript", 1)["payload"].get("quality") or {}).get("migration") or {}
    status = {item["id"]: item.get("status") or "open" for item in review.get("review") or []}
    answers = review.get("answers") or {}
    out = {"recording_id": rid, "transcript_revision": current["transcript"]["revision"],
           "review_revision": current["review"]["revision"],
           "pack_files": len(mig.get("review_packs") or []), "answers": len(answers),
           "answers_on_closed_places": sum(status.get(aid) in CLOSED_STATUSES for aid in answers),
           "open_places": sum(s not in CLOSED_STATUSES for s in status.values())}
    import_op = _import_operation(rid)
    untouched = (current["transcript"]["revision"] == 1 and current["review"]["revision"] == 1 and
                 current["transcript"]["operation_id"] == import_op and current["review"]["operation_id"] == import_op)
    if not pack_state_missing(review):
        out["status"] = "already_repaired"
    elif not untouched:
        out["status"] = "changed_after_import"
        out["why"] = "запись менялась после переноса — ремонт не выполняется, нужен человек"
    elif not answers and not out["pack_files"]:
        out["status"] = "nothing_to_repair"
    else:
        out["status"] = "candidate"
    return out, transcript, review, mig


def _legacy_answers_questions(review_db, transcript, packs, manifests):
    """Что дал бы прежний перенос (до В-1) из этих файлов пакета: ответы и вопросы с происхождением
    legacy_review_pack. По совпадению с базой видно, что файлы пакета после переноса не менялись."""
    review = copy.deepcopy(review_db)
    review["answers"] = {k: v for k, v in (review.get("answers") or {}).items()
                         if not str(v.get("origin", "")).startswith("legacy_review_pack:")}
    review["questions"] = [q for q in review.get("questions") or []
                           if not str(q.get("origin", "")).startswith("legacy_review_pack:")]
    for pack in packs:
        import_answers(review, pack.get("answers"), pack["name"])
    import_questions(review, transcript, manifests)
    pick = lambda rows: {k: v for k, v in rows.items()  # noqa: E731
                         if str(v.get("origin", "")).startswith("legacy_review_pack:")}
    return pick(review.get("answers") or {}), [q for q in review.get("questions") or []
                                                if str(q.get("origin", "")).startswith("legacy_review_pack:")]


def _repair_review(store, rid, review_db, transcript, mig, packs, manifests):
    """Новый снимок разбора: прежние ответы и вопросы переноса заменяются состоянием пакета (pack_state_from_legacy).
    -> (снимок или None, сведения). None — сверка с файлами не прошла или переносить нечего."""
    import storage
    answers_db = {k: v for k, v in (review_db.get("answers") or {}).items()
                  if str(v.get("origin", "")).startswith("legacy_review_pack:")}
    questions_db = [q for q in review_db.get("questions") or []
                    if str(q.get("origin", "")).startswith("legacy_review_pack:")]
    expected_answers, expected_questions = _legacy_answers_questions(review_db, transcript, packs, manifests)
    if expected_answers != answers_db or expected_questions != questions_db:
        return None, {"status": "pack_files_changed",
                      "why": "ответы или вопросы в файлах пакета не совпадают с перенесёнными — файлы менялись после "
                             "переноса или план указывает другие папки; ремонт не выполняется"}
    review = copy.deepcopy(review_db)
    review["answers"] = {k: v for k, v in (review.get("answers") or {}).items() if k not in answers_db}
    review["questions"] = [q for q in review.get("questions") or [] if q not in questions_db]
    if not review["questions"]:
        review.pop("questions")
    legacy_runtime = (mig.get("source") or {}).get("legacy_runtime") or {}
    stats = pack_state_from_legacy(review, transcript, packs, manifests, rid, legacy_runtime)
    if stats["state"] == "none":
        return None, {"status": "nothing_to_repair", "pack_state": stats}
    binding = _binding(store, rid)
    _set_pack_document(review, binding["root"] / binding["paths"]["json"])
    review.setdefault("undo", {})["legacy_pack_repair"] = {"tool": REPAIR_TOOL,
                                                           "pack_files_sha256": _pack_digest(packs, manifests)}
    storage.validate_pair(transcript, review)
    return review, {"pack_state": stats}


def _binding(store, rid):
    import recordings
    return recordings.document_binding(store, rid)


def _repair_backup(store):
    """Резерв базы перед первой записью ремонта: <папка данных>/db-backups.

    Как у других штатных резервов (core.backup_store, recordings.ensure_review_events): повреждённая копия
    правки — не отказ, а список в отчёте (решение владельца 23.11.2026, storage.validate_copy). Store.backup
    сам проверяет копию через validate_copy, второй проверки нет. -> (путь резерва, повреждённые копии правок)"""
    import recordings
    try:
        folder = recordings.review_events_backup_dir(store)
        folder.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = folder / f"state-before-pack-repair-{stamp}.sqlite3"
        if target.exists():
            target = folder / f"state-before-pack-repair-{stamp}-{uuid.uuid4().hex[:8]}.sqlite3"
        checked = store.backup(target, allow_damaged_copies=True)
    except Exception as ex:
        raise RepairBackupError(f"резерв базы перед ремонтом не создан: {type(ex).__name__}: {ex}") from ex
    return str(target), list(checked.get("damaged_copies") or [])


def _repair_write(store, rid, review, digest):
    import documents
    import recordings
    documents.guard_before_write(store, rid)
    binding = _binding(store, rid)
    operation_id = str(uuid.uuid5(NAMESPACE, f"legacy-pack-repair:{rid}:{digest}"))
    context = recordings.operation_context(store, "system")
    result = store.commit_change_with_document(
        rid, {"source_revision": 1, "review_revision": 1}, operation_id, context, review=review,
        root_id=binding["root_id"], generated_at=datetime.now(timezone.utc).isoformat())
    completed = recordings.materialize_recording(store, rid)
    documents.require_written(store, rid, result["document_revision"], completed)
    return result


def repair_packs(plan=None, *, apply=False, db_only=False):
    """Ремонт записей, перенесённых без состояния пакета карточек (М-1). По умолчанию сухой: ничего не пишет.

    db_only — без плана и файлов рядом с аудио: только то, что видно по базе. С планом записи находятся по SHA-256
    старой стенограммы; файлы пакета сверяются с тем, что записал перенос; apply — резерв базы перед первой
    записью, затем одна новая ревизия разбора на годную запись (стенограмма не меняется) и документы v4.
    -> отчёт {mode, backup, records, summary}. RepairBackupError — резерв не удался, ничего не записано."""
    import recordings
    if apply and (db_only or plan is None):
        raise ValueError("--apply нужен план и чтение файлов пакета")
    store = recordings.open_store(readonly=not apply)
    report = {"mode": "apply" if apply else ("db-only" if db_only else "dry-run"), "tool": REPAIR_TOOL,
              "backup": None, "records": []}
    try:
        migrated = _migrated_records(store)
        if db_only:
            for rid in store.db.execute("SELECT recording_id FROM recordings ORDER BY recording_id").fetchall():
                rid = rid[0]
                if rid not in migrated:
                    report["records"].append({"recording_id": rid, "status": "not_migrated_in_db"})
                    continue
                report["records"].append(_db_check(store, rid)[0])
        else:
            by_sha = {}
            for rid, sha in migrated.items():
                by_sha.setdefault(sha, []).append(rid)
            for rec in plan["recordings"]:
                for variant in rec["variants"]:
                    entry = {"transcript": variant["transcript"]}
                    try:
                        entry.update(_repair_variant(store, rec, variant, by_sha, apply, report))
                    except RepairBackupError:
                        raise
                    except Exception as ex:    # одна запись не останавливает отчёт по остальным
                        entry.update(status="error", error=f"{type(ex).__name__}: {ex}")
                    report["records"].append(entry)
    finally:
        store.close()
    counts = {}
    for e in report["records"]:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    report["summary"] = counts
    return report


def _repair_variant(store, rec, variant, by_sha, apply, report):
    t = Path(variant["transcript"])
    sha = sha256_file(t)
    rids = by_sha.get(sha) or []
    if len(rids) != 1:
        return {"status": "not_migrated" if not rids else "error",
                **({"error": "несколько записей с этим источником"} if rids else {})}
    rid = rids[0]
    check, transcript, review_db, mig = _db_check(store, rid)
    if check["status"] != "candidate":
        return check
    packs, manifests = _read_pack_files(variant.get("review_packs", []))
    review, info = _repair_review(store, rid, review_db, transcript, mig, packs, manifests)
    out = {**check, **info}
    if review is None:
        return out
    if not apply:
        out["status"] = "would_repair"
        return out
    if report["backup"] is None:
        report["backup"], damaged = _repair_backup(store)
        if damaged:
            report["backup_damaged_copies"] = damaged
    result = _repair_write(store, rid, review, _pack_digest(packs, manifests))
    out.update(status="repaired", review_revision=result.get("review_revision", 2))
    return out


def _repair_summary(report):
    lines = []
    for e in report["records"]:
        ps = e.get("pack_state") or {}
        lines.append(f"{str(e.get('recording_id') or '-')[:8]} {e['status']:20} ответов {e.get('answers', '-')}"
                     f" (на закрытых местах {e.get('answers_on_closed_places', '-')}); пакет {ps.get('state', '-')}: "
                     f"текущих {ps.get('answers_current', '-')}, в архиве {ps.get('answers_archived', '-')}, "
                     f"применённых {ps.get('applied', '-')}, ждущих {ps.get('waiting', '-')} "
                     f"(из них на закрытых местах {ps.get('waiting_closed_place', '-')}), "
                     f"снимков отмены не перенесено {ps.get('undo_snapshots_dropped', '-')}"
                     + (f"; {e['why']}" if e.get("why") else "") + (f"; ошибка: {e['error']}" if e.get("error") else ""))
    lines.append(f"итог: {report['summary']}" + (f"; резерв: {report['backup']}" if report.get("backup") else ""))
    if report.get("backup_damaged_copies"):
        lines.append(f"в резерве повреждённых копий правок: {len(report['backup_damaged_copies'])} "
                     "(не отказ, как у других резервов; список — в repair-report.json)")
    return "\n".join(lines)


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


def _repair_main(a):
    import paths
    if a.db_only and (a.apply or a.plan):
        print("--db-only работает без плана и без --apply", file=sys.stderr)
        return 2
    plan = None
    if not a.db_only:
        if not a.plan:
            print("для repair-packs нужен --plan (или --db-only)", file=sys.stderr)
            return 2
        try:
            plan = load_plan(a.plan)
        except PlanError as ex:
            print(f"ошибка плана: {ex}", file=sys.stderr)
            return 2
    if a.apply and (not a.data_dir or Path(a.data_dir).resolve() != paths.DATA_DIR.resolve()):
        print(f"для repair-packs --apply нужен --data-dir, равный папке данных программы: {paths.DATA_DIR}",
              file=sys.stderr)
        return 2
    try:
        report = repair_packs(plan, apply=a.apply, db_only=a.db_only)
    except RepairBackupError as ex:
        print(str(ex), file=sys.stderr)
        return 2
    except Exception as ex:
        print(f"окружение не готово: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 2
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "repair-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(_repair_summary(report))
    return 1 if any(e["status"] in REPAIR_REFUSALS for e in report["records"]) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["check", "apply", "mark-voices", "repair-packs"])
    ap.add_argument("--plan", help="план переноса (обязателен, кроме repair-packs --db-only)")
    ap.add_argument("--out", help="папка комплекта отчётов (inventory, plan, mapping, verification, report)")
    ap.add_argument("--data-dir", help="для apply и repair-packs --apply: папка данных, в которую идёт запись; "
                                       "должна совпадать с рабочей")
    ap.add_argument("--apply", action="store_true", help="для mark-voices и repair-packs: записать изменения")
    ap.add_argument("--db-only", action="store_true",
                    help="для repair-packs: только по базе, без плана и файлов рядом с аудио; ничего не пишет")
    a = ap.parse_args(argv)
    if a.command == "repair-packs":
        return _repair_main(a)
    if not a.plan:
        print("нужен --plan", file=sys.stderr)
        return 2
    try:
        plan = load_plan(a.plan)
    except PlanError as ex:
        print(f"ошибка плана: {ex}", file=sys.stderr)
        return 2
    if a.command == "mark-voices":
        result = mark_voices(plan, apply=a.apply)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0
    if a.command == "apply":
        import paths
        if not a.data_dir or not a.out:
            print("для apply нужны --data-dir и --out", file=sys.stderr)
            return 2
        if Path(a.data_dir).resolve() != paths.DATA_DIR.resolve():
            print(f"--data-dir не совпадает с папкой данных программы: {paths.DATA_DIR}", file=sys.stderr)
            return 2
    try:
        report = run(plan, a.command)
        verification = verify_set(report, a.out) if a.command == "apply" else None
    except Exception as ex:
        print(f"окружение не готово: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 2
    status = write_artifacts(a.out, plan, report, a.command, verification) if a.out else _status(a.command, report, None)
    print(_summary(report))
    print(f"статус: {status}")
    return 0 if status in ("checked", "verified_copy") and not any(e["status"] == "error" for e in report) else 1


if __name__ == "__main__":
    sys.exit(main())
