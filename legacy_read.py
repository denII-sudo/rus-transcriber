"""Чтение старых файлов записи (прежний файловый режим) — только чтение, ничего не пишет (Ч-1).

Старая запись — файлы рядом с аудио: стенограмма `<имя>.transcript.json` (и `.md`), папка разбора
`<имя>.review/` (вопросы `manifest.json`, ответы человека `answers.json`, архивы прежних пакетов
`answers.<версия>.json`), конспекты `<имя>.<название>.conspect.json/md`. `read` показывает, что в них лежит
и что сделало бы с сохранёнными ответами файловое «Применить» — по его правилу слов карточки: слова карточки
на месте и стоят подряд (их текст файловый режим не сверял). Предсказание — для ответов, которые «Применить»
попробует внести: ждущих и тех, что прошлое «Применить» не внесло (оно пробует их снова). Где «Применить»
решает по проверкам, которые чтение не повторяет (прослушивание, пауза, противоречие, свой ответ человека,
отмена и замена по снимку), исход — «нужна проверка».

Ничего не создаёт и не меняет: файлы читаются целиком байтами, замок ответов (`answers.lock`) не берётся,
папка данных программы и база записей не открываются (модули `paths`, `core`, `recordings` не импортируются).
Вносить прочитанное в базу эта функция не умеет (решение Архитектора 27.11: только чтение).
"""
import hashlib
import json
from pathlib import Path

import freeform
import review_pack as RP
import transcript as T

SUFFIX_JSON = ".transcript.json"
SUFFIX_MD = ".transcript.md"
# ответ, который «Применить» попробует внести: ждёт, или прошлое «Применить» его не внесло (пробует снова)
TRIED = RP.REBUILD_GROUPS["waiting"] | RP.REBUILD_GROUPS["unapplicable"]
LINES_LIMIT = 40

YES, NO, CHECK = "yes", "no", "check"
CHECK_LISTEN = "нужна проверка: ответ прослушивания — «Применить» сверяет прослушанный отрезок"
CHECK_CONTRADICTION = "нужна проверка: место с противоречием — «Применить» сверяет слова, показанные на карточке"
CHECK_FREE = "нужна проверка: свой ответ человека вносится по словам (разбор ответа), чтение его не повторяет"
CHECK_UNDO = "нужна проверка: отмена по снимку — «Применить» сверяет, менялась ли стенограмма после ответа"
CHECK_REPLACE = "нужна проверка: замена внесённого ответа по снимку — «Применить» сначала снимает прежний"
CHECK_OVERLAP = "нужна проверка: те же слова у ответа другой карточки — «Применить» оставит обе человеку"
CHECK_PAUSE = ("нужна проверка: место-пауза — «Применить» закрывает её только прослушиванием или своим текстом "
               "и сверяет слова паузы")
NO_SNAPSHOT = "для безопасной отмены нет проверенного структурного снимка"
OTHER_RECORD = "пакет относится к другой стенограмме"
NO_ACTOR = ("«Применить» откажет целиком: есть ответ неизвестного происхождения, его нельзя применять как "
            "человеческий без явного подтверждения")


class LegacyReadError(ValueError):
    """Путь не указывает на читаемую старую стенограмму."""


def _sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def _json_file(p):
    """-> (объект, sha256 байтов) или (None, None), если файла нет; битый JSON — (None, sha256)."""
    try:
        blob = Path(p).read_bytes()
    except FileNotFoundError:
        return None, None
    try:
        return json.loads(blob.decode("utf-8")), _sha256(blob)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, _sha256(blob)


def _file_info(p):
    try:
        blob = Path(p).read_bytes()
    except FileNotFoundError:
        return None
    return {"file": str(p), "sha256": _sha256(blob), "bytes": len(blob)}


def resolve(path):
    """Путь к стенограмме JSON: дан `.transcript.json` или `.transcript.md` рядом с ним."""
    p = Path(path)
    if p.name.endswith(SUFFIX_MD):
        p = p.with_name(p.name[:-len(SUFFIX_MD)] + SUFFIX_JSON)
    if not p.name.endswith(SUFFIX_JSON):
        raise LegacyReadError(f"нужен файл старой стенограммы *{SUFFIX_JSON} (или *{SUFFIX_MD} рядом с ним)")
    if not p.is_file():
        raise LegacyReadError(f"файла стенограммы нет: {p}")
    return p


def _words_of(data, ids):
    """Слова с ID ids (как их найдёт «Применить»), или None, если найдены не все."""
    if not ids:
        return None
    by_id = {w.get("word_id"): w for s in data.get("segments") or [] for w in s.get("words") or []
             if w.get("word_id")}
    if any(i not in by_id for i in ids):
        return None
    return [by_id[i].get("w") for i in ids]


def _transcript(jp, data, sha, offset, limit):
    segments = data.get("segments") or []
    speakers = list(dict.fromkeys(s.get("speaker") for s in segments if s.get("speaker")))
    lines = [f"{s.get('speaker') or '?'}: {s.get('text') or ' '.join(w.get('w') or '' for w in s.get('words') or [])}"
             for s in segments]
    places = data.get("review") or []
    return {"file": str(jp), "sha256": sha, "recording_id": data.get("recording_id"),
            "source_revision": data.get("source_revision"), "segments": len(segments),
            "words": sum(len(s.get("words") or []) for s in segments), "speakers": speakers,
            "places": len(places), "places_open": sum(1 for r in places if r.get("status") == "open"),
            "lines_total": len(lines), "lines_offset": offset, "lines": lines[offset:offset + limit]}


def _refusal(state):
    """То же условие, что у файлового «Применить» (core._apply_answers_files_once) перед любым изменением."""
    applied = RP._applied_map(state)
    for aid, answer in (state.get("answers") or {}).items():
        if (answer.get("kind") not in ("skip", "repeat") and applied.get(aid) != RP._sig(answer) and
                answer.get("actor_kind") != "human" and
                (answer.get("confirmation_context") or {}).get("actor_kind") != "human" and
                not RP.legacy_decided(state, aid, answer)):
            return NO_ACTOR
    return None


def _preview(it, answer, data):
    """Что ответ внёс бы в слова места: {before, after} и при необходимости speaker / status."""
    ids = None if T.is_gap_place(it) else it.get("word_ids")
    words = _words_of(data, ids)
    before = " ".join(words) if words is not None else None
    kind = answer.get("kind")
    out = {"before": before, "after": before}
    if kind in ("option", "recommended"):
        options = it.get("options") or []
        k = answer.get("option")
        if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < len(options):
            return {"before": before, "after": None, "bad_option": True}
        d = options[k].get("decision") or {}
        if isinstance(d.get("text"), str):
            out["after"] = d["text"]
        for key in ("speaker", "status"):
            if d.get(key):
                out[key] = d[key]
    elif kind == "human":
        out["status"] = "human"
    elif kind == "free":
        out["after"] = (answer.get("text") or "").strip()
    return out


def _verdict(it, answer, ctx):
    """(yes|no|check, почему) для ответа, который «Применить» попробует внести, — что сделало бы с ним файловое
    «Применить». Порядок проверок — как в review_pack.decisions и core._apply_answers_files_locked."""
    aid = it.get("id")
    if ctx["refusal"]:
        return NO, ctx["refusal"]
    if ctx["other_record"]:
        return NO, OTHER_RECORD
    if RP.legacy_decided(ctx["state"], aid, answer):
        return NO, RP.OUTCOME_REASONS["already_decided"]
    relation = RP.over_applied(answer, ctx["applied"].get(aid))
    if relation == "revert":
        return (CHECK, CHECK_UNDO) if (ctx["state"].get("undo_snapshots") or {}).get(aid) else (NO, NO_SNAPSHOT)
    if relation == "pending_new":
        if RP.replaceable(ctx["state"], aid, answer, ctx["applied"].get(aid)):
            return CHECK, CHECK_REPLACE
        return NO, RP.OUTCOME_REASONS["pending_new"]
    if aid in ctx["conflicts"]:
        return NO, ctx["conflicts"][aid]
    kind = answer.get("kind")
    if kind == "free":
        if freeform.negative_answer(answer.get("text")):
            return NO, RP.OUTCOME_REASONS["negative"]
        return CHECK, CHECK_FREE
    if kind == "revert":
        return NO, RP.OUTCOME_REASONS["skipped"]
    place = ctx["places"].get(aid) or {}
    if "seen_fp" in it and T.contradiction_pending(place):
        return CHECK, CHECK_CONTRADICTION
    if kind in ("option", "recommended"):
        if ctx["previews"][aid].get("bad_option"):
            return NO, RP.OUTCOME_REASONS["not_applied"]
        if (it["options"][answer["option"]].get("decision") or {}).get("listened"):
            return CHECK, CHECK_LISTEN
        if T.is_gap_place(it) or T.is_gap_place(place):
            return CHECK, CHECK_PAUSE
    if aid in ctx["overlaps"]:
        return CHECK, CHECK_OVERLAP
    if kind == "human":
        return YES, RP.OUTCOME_REASONS["left_for_human"]
    if kind in ("option", "recommended"):
        return YES, RP.OUTCOME_REASONS["applied"]
    return CHECK, f"нужна проверка: вид ответа «{kind}» чтение не разбирает"


def _overlaps(items, waiting_ids):
    """Ждущие ответы карточек, чьи слова пересекаются (предпроверка «Применить» оставит их человеку)."""
    owners = {}
    for it in items:
        aid = it.get("id")
        if aid in waiting_ids and not T.is_gap_place(it):
            for wid in it.get("word_ids") or []:
                owners.setdefault(wid, set()).add(aid)
    return {aid for ids in owners.values() if len(ids) > 1 for aid in ids}


def _archives(pd):
    out = []
    for p in sorted(pd.glob("answers.*.json")):
        version = p.name[len("answers."):-len(".json")]
        state, sha = _json_file(p)
        state = state if isinstance(state, dict) else {}
        man, _ = _json_file(pd / f"manifest.{version}.json")
        out.append({"version": version, "file": str(p), "sha256": sha,
                    "answers": len(state.get("answers") or {}), "applied": len(RP._applied_map(state)),
                    "cards": len(man.get("items") or []) if isinstance(man, dict) else None})
    return out


def _pack(jp, data):
    pd = RP.pack_dir(jp)
    if not pd.is_dir():
        return None
    man, man_sha = _json_file(pd / "manifest.json")
    state, answers_sha = _json_file(pd / "answers.json")
    out = {"folder": str(pd), "manifest_sha256": man_sha, "answers_sha256": answers_sha, "archives": _archives(pd)}
    if not isinstance(man, dict):
        out["error"] = "manifest.json нет или он не читается"
        return out
    state = state if isinstance(state, dict) else {"answers": {}}
    state.setdefault("answers", {})
    items = [it for it in man.get("items") or [] if isinstance(it, dict)]
    answers = state["answers"]
    states = RP.answer_states_of(state, man)
    ctx = {"state": state, "applied": RP._applied_map(state), "refusal": _refusal(state),
           "other_record": bool(man.get("recording_id")) and man.get("recording_id") != data.get("recording_id"),
           "conflicts": RP.answer_revision_conflicts(data, man, state, None, sqlite=False),
           "places": {r.get("id"): r for r in data.get("review") or []},
           "overlaps": _overlaps(items, {aid for aid, row in states.items() if row["state"] in TRIED}),
           "previews": {it.get("id"): _preview(it, answers[it.get("id")], data) for it in items
                        if it.get("id") in answers}}
    rows, counts, by_state = [], {}, {}
    for n, it in enumerate(items, 1):
        aid = it.get("id")
        answer = answers.get(aid)
        row = {"id": aid, "card": n, "question": it.get("question"), "text": it.get("text"),
               "speaker": it.get("speaker"), "t": it.get("t"),
               "options": [o.get("label") for o in it.get("options") or []]}
        if answer is None:
            row["state"] = "none"
        elif not RP.answer_is_current(answer, it):
            row["state"], row["why"] = "stale", RP.REV_STALE
        else:
            shown = states.get(aid) or {"state": "saved", "why": RP.OUTCOME_REASONS["saved"]}
            row["state"], row["why"] = shown["state"], shown["why"]
            if shown.get("detail"):
                row["detail"] = shown["detail"]
        if answer is not None:
            row["answer"] = {k: answer.get(k) for k in ("kind", "option", "text", "actor_kind", "source", "ts")
                             if answer.get(k) is not None}
            row["preview"] = ctx["previews"][aid]
        if row["state"] in TRIED:
            row["would_apply"], row["why_apply"] = _verdict(it, answer, ctx)
            counts[row["would_apply"]] = counts.get(row["would_apply"], 0) + 1
        by_state[row["state"]] = by_state.get(row["state"], 0) + 1
        rows.append(row)
    out.update({"version": man.get("version"), "recording_id": man.get("recording_id"),
                "source_revision": man.get("source_revision"), "cards": len(items), "states": by_state,
                "waiting": {k: counts.get(k, 0) for k in (YES, NO, CHECK)}, "apply_refused": ctx["refusal"],
                "items": rows})
    return out


def read(path, offset=0, limit=LINES_LIMIT):
    """Что лежит в старых файлах записи и что внёс бы файловый режим. Ничего не пишет.

    path — `<имя>.transcript.json` или `.transcript.md`; offset/limit — окно строк стенограммы «Говорящий: текст».
    -> {transcript, md, review_pack (None — папки разбора нет), conspects, writes: False}."""
    jp = resolve(path)
    data, sha = _json_file(jp)
    if not isinstance(data, dict):
        raise LegacyReadError(f"стенограмма не читается как JSON: {jp}")
    offset, limit = max(0, int(offset)), max(0, int(limit))
    stem = jp.name[:-len(SUFFIX_JSON)]
    conspects = [info for p in sorted(jp.parent.glob(f"{stem}.*.conspect.*"))
                 if p.name.endswith((".conspect.json", ".conspect.md")) and (info := _file_info(p))]
    return {"transcript": _transcript(jp, data, sha, offset, limit),
            "md": _file_info(jp.with_name(stem + SUFFIX_MD)),
            "review_pack": _pack(jp, data), "conspects": conspects, "writes": False}
