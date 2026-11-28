"""Привязка мест разбора у записей, перенесённых из старых файлов (П7): сверка со временем (S24-1, S24-2).

Перенос старых файлов (`tools/import_legacy.py`, удалён в У-1) переводил ссылку места «реплика + номера слов»
(`seg/w1/w2`) в ID слов (`word_ids`). Два его допущения были неверны:
- S24-1: номера слов принимались без сверки со временем, хотя старый файловый режим их после правок человека не
  пересчитывал. Место получало ID чужих слов той же реплики, и пометки об этом нет;
- S24-2: по времени перенос искал слова, кончающиеся не позже `t2`, а в старых файлах `t2` — начало последнего
  слова. Последнее слово места выпадало из `word_ids`.

Нынешний код верит `word_ids` места. Сверять их с `t1/t2` можно только в ревизии переноса — первой ревизии
записи с `quality.migration` (`recordings.migration_info`): в ней правок ещё не было. Позже `_relink_places`
переносит `word_ids` на слова правки, а `t1/t2` места не меняет, поэтому сверка нынешних данных со временем
ложно отказала бы нормальным местам.

Правило слов по времени — то же, что у ответа (`transcript._find_span` без `word_ids`): начало слова в
`[t1 − 0,01; t2 + 0,01]`, реплика того же говорящего.

Модуль только читает. Ремонт данных (новая ревизия с верными `word_ids`) — отдельное решение владельца.
"""
import transcript as T

FOREIGN, PARTIAL, OTHER = "foreign", "partial", "other"
KIND_RU = {FOREIGN: "чужие слова", PARTIAL: "не все слова места", OTHER: "другие слова"}

# Что видит человек под карточкой после «Применить» (уточнение к исходу «не применён»). Простыми словами, без
# внутренних имён; {text} — слова, которые звучат в это время по записи в том виде, в каком её перенесли.
ANCHOR_MISMATCH = ("это место перенесено из старых файлов с ошибкой: карточка указывает не на те слова, что стоят в "
                   "стенограмме в это время (по стенограмме здесь «{text}»). Этот ответ не внесён и слов не изменил. "
                   "Сначала нужно исправить привязку места — скажите Claude; после этого карточка обновится, и "
                   "ответить на неё нужно будет заново")
# Отметка места после ремонта привязки (ставит ремонт): дальше место сверяется как обычное.
REPAIRED = "anchor_repaired"


def _by_id(data):
    return {w.get("word_id"): w for seg in data.get("segments") or [] for w in seg.get("words") or []
            if w.get("word_id")}


def _text(by_id, ids):
    return " ".join(str(by_id[i].get("w", "")) for i in ids if i in by_id)


def place_mismatch(data, place):
    """Сверка одного места с его временем в стенограмме data (ревизия переноса).

    -> None, если место не проверяется (пропуск, нет `word_ids` или конечного времени, какого-то ID нет в data)
    или его слова совпадают со словами по времени; иначе {kind, bound, by_time, bound_text, by_time_text}."""
    if T.is_gap_place(place):
        return None
    ids = place.get("word_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
        return None
    times = T._place_time(place)
    if times is None:
        return None
    by_id = _by_id(data)
    if any(x not in by_id for x in ids):
        return None
    loc = T._find_span(data, {k: v for k, v in place.items() if k != "word_ids"})
    by_time = [] if loc is None else \
        [w.get("word_id") for w in (data["segments"][loc[0]].get("words") or [])[loc[1]:loc[2]]]
    if set(ids) == set(by_time):
        return None
    t1, t2 = times
    tol = T.WORD_PLACE_TOLERANCE
    if any(not t1 - tol <= by_id[x]["s"] <= t2 + tol for x in ids):
        kind = FOREIGN
    elif set(ids) < set(by_time):
        kind = PARTIAL
    else:
        kind = OTHER
    return {"kind": kind, "bound": list(ids), "by_time": by_time,
            "bound_text": _text(by_id, ids), "by_time_text": _text(by_id, by_time)}


def migration_mismatches(store, recording_id):
    """{id места: расхождение} по ревизии переноса записи; {} — запись не из переноса или расхождений нет."""
    import recordings
    import storage
    if not recordings.migration_info(store, recording_id):
        return {}
    try:
        transcript = store.get_revision(recording_id, "transcript", 1)["payload"]
        review = store.get_revision(recording_id, "review", 1)["payload"]
    except (KeyError, ValueError, storage.StorageError):
        return {}
    out = {}
    for place in review.get("review") or []:
        if isinstance(place, dict) and place.get("id"):
            found = place_mismatch(transcript, place)
            if found is not None:
                out[place["id"]] = found
    return out


def card_conflicts(mismatches, data, cards):
    """{id: причина} для ответов, которые легли бы не на те слова.

    mismatches — migration_mismatches; data — нынешняя стенограмма с разбором; cards — {id: карточка или решение}.
    Ответ ложится на `word_ids` карточки, а без них — на `word_ids` места сейчас (transcript.apply_review). Отказ —
    если эти слова не совпадают со словами по времени ни в ревизии переноса, ни в нынешних данных. Нет ID ни там, ни
    там — место ищется по времени, отказа нет. Место с отметкой ремонта (REPAIRED) не проверяется: после ремонта его
    слова правят как обычно, и `_relink_places` переносит привязку на слова правок (O265-3)."""
    if not mismatches:
        return {}
    places = {r.get("id"): r for r in data.get("review") or [] if isinstance(r, dict)}
    out = {}
    for aid, card in cards.items():
        found = mismatches.get(aid)
        place = places.get(aid) or {}
        if found is None or place.get(REPAIRED):
            continue
        card = card or {}
        ids = card.get("word_ids") if "word_ids" in card else place.get("word_ids")
        if not ids or set(ids) == set(found["by_time"]):
            continue
        loc = T._find_span(data, {k: v for k, v in place.items() if k != "word_ids"}) if place else None
        now = [] if loc is None else \
            [w.get("word_id") for w in (data["segments"][loc[0]].get("words") or [])[loc[1]:loc[2]]]
        if now and set(ids) == set(now):
            continue
        out[aid] = ANCHOR_MISMATCH.format(text=found["by_time_text"] or "—")
    return out


def _marked_by_card(word, aid):
    provenance = (word or {}).get("provenance") or {}
    return any((provenance.get(dim) or {}).get("by") == "human" and (provenance.get(dim) or {}).get("edit") == aid
               for dim in ("text", "speaker"))


def record_report(store, recording_id, *, texts=True):
    """Места записи с расхождением и их состояние сейчас; None — запись не из переноса.

    human_marked — слов места с отметкой человека от этой карточки (ответ «как есть» или вариант),
    human_marked_wrong — из них чужих (их нет среди слов по времени)."""
    import recordings
    if not recordings.migration_info(store, recording_id):
        return None
    mismatches = migration_mismatches(store, recording_id)
    rows = []
    if mismatches:
        current = store.get_current(recording_id)
        words_now = _by_id(current["transcript"]["payload"])
        places_now = {r.get("id"): r for r in current["review"]["payload"].get("review") or []
                      if isinstance(r, dict)}
        for aid, found in mismatches.items():
            place = places_now.get(aid)
            wrong = set(found["bound"]) - set(found["by_time"])
            candidates = set(found["bound"]) | set((place or {}).get("word_ids") or [])
            marked = {x for x in candidates if _marked_by_card(words_now.get(x), aid)}
            row = {"id": aid, "kind": found["kind"], "kind_ru": KIND_RU[found["kind"]],
                   "status": (place or {}).get("status", "open") if place else None,
                   "closed": bool(place) and not T.place_unresolved(place),
                   "human_marked": len(marked), "human_marked_wrong": len(marked & wrong)}
            if texts:
                row.update(t=(place or {}).get("t"), speaker=(place or {}).get("speaker"),
                           bound_text=found["bound_text"], by_time_text=found["by_time_text"])
            rows.append(row)
    return {"recording_id": recording_id, "places": rows}


def check_store(store, recording_ids=None, *, texts=True):
    """Сводка по базе (или по названным записям: полный ID или его начало). Только чтение: ничего не пишет.
    Записи, которых нет (или начало подходит к нескольким), — в not_found: опечатка не выглядит как «расхождений нет»."""
    known = [r[0] for r in store.db.execute("SELECT recording_id FROM recordings ORDER BY recording_id")]
    not_found = []
    if recording_ids is None:
        recording_ids = known
    else:
        wanted, recording_ids = recording_ids, []
        for key in wanted:
            hits = [rid for rid in known if rid == key] or [rid for rid in known if rid.startswith(key)]
            if len(hits) == 1:
                recording_ids.append(hits[0])
            else:
                not_found.append(key)
    migrated, records = 0, []
    for rid in recording_ids:
        report = record_report(store, rid, texts=texts)
        if report is None:
            continue
        migrated += 1
        if report["places"]:
            records.append(report)
    places = [row for rec in records for row in rec["places"]]
    by_kind = {}
    for row in places:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
    return {"recordings_checked": len(recording_ids), "migrated": migrated,
            "recordings_with_mismatch": len(records), "places": len(places), "by_kind": by_kind,
            "closed": sum(1 for row in places if row["closed"]),
            "closed_with_human_marks": sum(1 for row in places if row["closed"] and row["human_marked"]),
            "human_marked_wrong_words": sum(row["human_marked_wrong"] for row in places),
            "records": records, "not_found": not_found, "writes": False}
