"""Запись расшифровки (.json и .md), переименование спикеров, разбор сомнительных мест.

Два слоя данных: слова модели (segments[].words: текст, время, уверенность, вторая гипотеза) и правки
(edits — журнал: кто, что, было/стало). Правка не стирает слой модели: исходные слова остаются
в поле model у исправленного места и в журнале.
"""
import json
import re
from pathlib import Path

REASON_RU = {"asr": "модели распознали по-разному", "conf": "модель не уверена в слове",
             "voice": "голоса близки — говорящий под вопросом", "overlap": "говорят одновременно",
             "gender": "род глагола не совпадает с полом говорящего"}


def hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


DOUBT_OPEN, DOUBT_CLOSE = "⟦", "⟧"
DOUBT_LEGEND = ("_⟦…⟧ — сомнительное место, ещё не решено: слова могут быть распознаны или подписаны неверно "
                "(подробно — review_items). В цитаты конспекта такие места не брать без проверки._")


def unresolved(data):
    """Места разбора, ещё не решённые: открытые и оставленные человеку."""
    return [i for i in data.get("review") or [] if i.get("status", "open") in ("open", "human")]


def doubt_marks(data):
    """{номер реплики: {номер слова: id места}} — слова нерешённых мест (по времени: индексы сдвигаются)."""
    out = {}
    for it in unresolved(data):
        t1, t2 = it.get("t1", it.get("t")), it.get("t2", it.get("t"))
        if t1 is None:
            continue
        cands = []
        for si, seg in enumerate(data["segments"]):
            idx = [k for k, w in enumerate(seg.get("words") or []) if t1 - 0.01 <= w["s"] <= t2 + 0.01]
            if idx:
                cands.append((seg["speaker"] != it.get("speaker"), si, idx))
        if cands:                      # при наложении речи — реплика того, кому место приписано
            _, si, idx = min(cands)
            for k in idx:
                out.setdefault(si, {})[k] = it["id"]
    return out


def strip_marks(text):
    """Текст без пометок ⟦…⟧ (для сравнения цитат и ответов)."""
    return (text or "").replace(DOUBT_OPEN, "").replace(DOUBT_CLOSE, "")


def _marked_text(seg, marks):
    ws = seg.get("words") or []
    if not marks or not ws or " ".join(w["w"] for w in ws) != seg["text"]:
        return seg["text"]
    out = []
    for k, w in enumerate(ws):
        x = w["w"]
        if k in marks:
            if k - 1 not in marks:
                x = DOUBT_OPEN + x
            if k + 1 not in marks:
                x = x + DOUBT_CLOSE
        out.append(x)
    return " ".join(out)


def _seg_line(seg, marks=None):
    tag = ""
    em = seg.get("emotion")
    if em and em["label"] != "нейтрально" and em["score"] >= 0.5:
        tag = f" _({em['label']}, {em['score']:.2f})_"
    text = _marked_text(seg, marks)
    if seg.get("overlap"):
        tag += " _(одновременно с соседней репликой)_"
    for ins in seg.get("inserts") or []:
        text += f" _[поверх — {ins['speaker']}: «{ins['text']}»]_"
    return f"**[{hms(seg['start'])}] {seg['speaker']}**{tag}: {text}"


def _item_line(it):
    why = ", ".join(REASON_RU.get(r, r) for r in it.get("reasons", []))
    s = f"- {hms(it['t'])} {it['speaker']}: «{it['text']}» — {why}"
    if it.get("alt_ctc"):
        s += f"; CTC: «{it['alt_ctc']}»"
    if it.get("alt_whisper"):
        s += f"; Whisper: «{it['alt_whisper']}»"
    if it.get("voices") and "voice" in it.get("reasons", []):
        top = list(it["voices"].items())[:2]
        s += "; голоса: " + ", ".join(f"{n} {v:.2f}" for n, v in top)
    if it.get("note"):
        s += f"; {it['note']}"
    return s


def render_md(data, text_only=False):
    """Стенограмма в markdown. Слова нерешённых мест — в ⟦…⟧. text_only — только шапка и реплики."""
    lines = [f"# Расшифровка: {Path(data['file']).name}", "",
             f"Длительность: {data['duration']} · Участников: {len(data['speakers'])}"
             + (f" ({', '.join(data['speakers'])})" if data.get("speaker_names") else ""), ""]
    q0 = data.get("quality") or {}
    if q0.get("group_doubt"):
        lines.append(f"_Внимание: {q0['group_doubt']['note']}._\n")
    if q0.get("names_outside_vocab"):
        lines.append(f"_Нет в словаре группы: {', '.join(q0['names_outside_vocab'])} — их голоса в профиль не "
                     "идут, пока их не добавят в participants словаря._\n")
    if data.get("names_source") == "voices" and not data.get("names_confirmed"):
        lines.append("_Имена подписаны автоматически по голосовым профилям; подтвердить или исправить — "
                     "rename_speakers._\n")
    if data.get("review") is not None:
        open_items = [i for i in data["review"] if i.get("status", "open") == "open"]
        human = [i for i in data["review"] if i.get("status") == "human"]
        if data.get("review_done"):
            lines.append(f"_Сомнительные места разобраны: правок {len(data.get('edits', []))}, "
                         f"осталось для человека {len(human)} (в конце)._\n")
        elif open_items:
            lines.append(f"_Сомнительных мест: {len(open_items)} — ещё не разобраны (review_items)._\n")
    marks = doubt_marks(data) if data.get("review") else {}
    if marks:
        lines.append(DOUBT_LEGEND + "\n")
    for si, seg in enumerate(data["segments"]):
        lines.append(_seg_line(seg, marks.get(si)))
        lines.append("")
    if text_only:
        return "\n".join(lines).rstrip() + "\n"
    q = data.get("quality") or {}
    fixes, vfix, old_review = q.get("speaker_fixes") or [], q.get("vocab_fixes") or [], q.get("review") or []
    if fixes or vfix or old_review or q.get("asr_votes"):
        lines += ["---", "", "## Автоматические исправления", ""]
        kinds = {"split": "чужой голос внутри реплики", "voice": "голос реплики", "gender": "род глагола и голос"}
        for f in fixes:
            lines.append(f"- {hms(f['t'])} — {kinds.get(f['kind'], f['kind'])}: {f['from']} → {f['to']}")
        for f in vfix:
            lines.append(f"- {hms(f['t'])} — словарь: " + ", ".join(f"«{a}» → «{b}»" for a, b in f["fixes"]))
        for f in q.get("asr_votes") or []:
            lines.append(f"- {hms(f['t'])} — две модели из трёх: " + (f"«{f['model']}» → «{f['to']}»" if f["model"]
                                                                    else f"добавлено «{f['to']}»"))
        if old_review:
            lines += ["", "## Проверить", ""]
            for r in old_review:
                lines.append(f"- {hms(r['t'])} {r['speaker']}: {r['reason']}")
        lines.append("")
    if data.get("review"):
        if data.get("review_done"):
            human = [i for i in data["review"] if i.get("status") == "human"]
            if human:
                lines += ["## Проверить человеку", ""] + [_item_line(i) for i in human] + [""]
            if data.get("edits"):
                lines += ["## Правки после распознавания", ""]
                for e in data["edits"]:
                    b, a = e["before"], e["after"]
                    ch = []
                    if b.get("text") != a.get("text"):
                        ch.append(f"«{b.get('text')}» → «{a.get('text')}»")
                    if b.get("speaker") != a.get("speaker"):
                        ch.append(f"{b.get('speaker')} → {a.get('speaker')}")
                    lines.append(f"- {hms(e['t'])} ({e.get('by', '?')}): " + "; ".join(ch)
                                 + (f" — {e['note']}" if e.get("note") else ""))
                lines.append("")
        else:
            lines += ["## Сомнительные места", ""] + [_item_line(i) for i in data["review"]] + [""]
    return "\n".join(lines)


def save(data, json_path, md_path):
    Path(json_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(md_path).write_text(render_md(data), encoding="utf-8")


def load(json_path):
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def rename(json_path, md_path, mapping):
    """mapping: {"Спикер 4": "Анна", ...}. Ключом может быть исходная метка или уже данное имя.
    Исходная метка сохраняется в поле label каждого сегмента, поэтому переименовывать можно повторно."""
    data = load(json_path)
    mapping = {str(k).strip(): str(v).strip() for k, v in mapping.items() if str(v).strip()}
    current = data.get("speaker_names") or {}            # исходная метка -> имя
    for label, name in list(current.items()):
        if name in mapping:                              # переименование уже данного имени
            current[label] = mapping.pop(name)
    for key, name in mapping.items():
        current[key] = name
    labels_all = {s.get("label", s["speaker"]) for s in data["segments"]}
    unknown = [k for k in current if k not in labels_all]
    old_names = {}
    for seg in data["segments"]:
        label = seg.setdefault("label", seg["speaker"])
        old_names[seg["speaker"]] = current.get(label, label)
        seg["speaker"] = current.get(label, label)
        for ins in seg.get("inserts") or []:
            lbl = ins.setdefault("label", ins["speaker"])
            ins["speaker"] = current.get(lbl, lbl)
    for it in data.get("review") or []:
        it["speaker"] = old_names.get(it["speaker"], it["speaker"])
    labels = data.get("speaker_labels") or data["speakers"]
    data["speaker_labels"] = labels
    data["speakers"] = [current.get(l, l) for l in labels]
    data["speaker_names"] = current
    data["names_confirmed"] = True          # такие расшифровки пополняют голосовые профили
    data["names_source"] = "rename"
    save(data, json_path, md_path)
    data["_conspects"] = _refresh_conspects(json_path)
    return data, unknown


def _refresh_conspects(json_path):
    """После любой правки стенограммы пересобираем связанные с ней конспекты (conspect.py)."""
    try:
        import conspect
        return conspect.rerender(json_path)
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}


# ---------- разбор сомнительных мест ----------
def _find_span(data, it):
    """Находит место по времени слов (индексы сдвигаются после правок)."""
    t1, t2 = it.get("t1", it["t"]), it.get("t2", it["t"])
    for si, seg in enumerate(data["segments"]):
        ws = seg.get("words") or []
        idx = [k for k, w in enumerate(ws) if t1 - 0.01 <= w["s"] <= t2 + 0.01]
        if idx:
            return si, idx[0], idx[-1] + 1
    return None


def _ctx_text(seg, w1, w2):
    ws = seg.get("words") or []
    if not ws:
        return seg["text"]
    return " ".join([w["w"] for w in ws[:w1]] + ["⟦" + " ".join(w["w"] for w in ws[w1:w2]) + "⟧"]
                    + [w["w"] for w in ws[w2:]])


def review_list(json_path, offset=0, limit=40, reasons=None, min_sec=0.0, brief=False, status="open"):
    """Места разбора. По умолчанию открытые, с контекстом: соседние реплики и реплика с выделенным местом ⟦…⟧.
    reasons — только места с одной из этих причин (asr, conf, voice, overlap, gender, missed);
    min_sec — только места не короче (с); brief — кратко, без контекста (id, время, говорящий, текст, причины,
    варианты); status — open | human | resolved | all."""
    data = load(json_path)
    allr = data.get("review") or []
    want = set(reasons or [])

    def ok(i):
        st = i.get("status", "open")
        if status != "all" and st != status:
            return False
        if want and not want & set(i.get("reasons") or []):
            return False
        dur = float(i.get("t2", i.get("t", 0))) - float(i.get("t1", i.get("t", 0)))
        return dur + 1e-9 >= float(min_sec or 0)
    items = [i for i in allr if ok(i)]
    by_reason = {}
    for i in items:
        for r in i.get("reasons") or []:
            by_reason[r] = by_reason.get(r, 0) + 1
    out = []
    for it in items[offset:offset + limit]:
        if brief:
            row = {"id": it["id"], "t": hms(it["t"]), "speaker": it.get("speaker"),
                   "text": str(it.get("text") or "")[:120], "reasons": it.get("reasons")}
            for k in ("alt_ctc", "alt_whisper"):
                if it.get(k):
                    row[k] = str(it[k])[:120]
            if status != "open":
                row["status"] = it.get("status", "open")
            out.append(row)
            continue
        loc = _find_span(data, it)
        row = {k: it.get(k) for k in ("id", "t", "speaker", "text", "alt_ctc", "alt_whisper", "alt_ctc_extra", "reasons", "voices",
                                      "conf_min", "note", "status") if it.get(k) is not None}
        row["t"] = hms(it["t"])
        if loc:
            si, w1, w2 = loc
            seg = data["segments"][si]
            row["reply"] = f"{seg['speaker']}: {_ctx_text(seg, w1, w2)}"[:700]
            if si > 0:
                p = data["segments"][si - 1]
                row["before"] = f"{p['speaker']}: {p['text']}"[-300:]
            if si + 1 < len(data["segments"]):
                n = data["segments"][si + 1]
                row["after"] = f"{n['speaker']}: {n['text']}"[:300]
        out.append(row)
    res = {"total": len(items), "offset": offset, "returned": len(out), "by_reason": by_reason, "items": out,
           "speakers": data.get("speakers")}
    if status == "open":
        res["total_open"] = len(items)
    if offset + limit < len(items):
        res["next_offset"] = offset + limit
    if not brief:
        res["hint"] = ("Решения: apply_review(job_id, decisions). Выбирать из вариантов (текст основной модели, "
                       "alt_ctc, alt_whisper, говорящие из voices); свой текст — только для явной ошибки окончания "
                       "или слова из словаря, с note. Много мест — brief=True и фильтры reasons/min_sec.")
    return res


def edit_span(json_path, md_path, t1, t2, text, speaker=None, note=None, by="claude", resolve=True):
    """Правка произвольного отрезка стенограммы: слова, начинающиеся в [t1, t2] с, заменяются текстом text
    (по словам: совпадающие сохраняют время), speaker — подписать отрезок этим говорящим (реплика делится).
    Пустой text — удалить слова отрезка. На отрезке без слов text вставляется (тогда speaker обязателен).
    resolve — места разбора, целиком лежащие в отрезке, считаются решёнными. -> (data, отчёт)"""
    data = load(json_path)
    t1, t2 = float(t1), float(t2)
    if t2 < t1:
        t1, t2 = t2, t1
    speakers = [s for s in (data.get("speakers") or []) if s]
    if speaker and speakers and speaker not in speakers:
        raise ValueError(f"нет такого говорящего: «{speaker}». Есть: {', '.join(speakers)} "
                         "(новое имя — сначала rename_speakers)")
    has_words = any(t1 - 0.01 <= w["s"] <= t2 + 0.01 for sg in data["segments"] for w in sg.get("words") or [])
    if not has_words and not (text or "").strip():
        raise ValueError("на отрезке нет слов — удалять нечего")
    if not has_words and not speaker:
        raise ValueError("на отрезке нет слов: для вставки укажите speaker")
    card = f"span@{t1:.2f}"
    e = {"op": "place_text", "t1": t1, "t2": t2, "text": (text or "").strip(), "card": card}
    if speaker:
        e["speaker"] = speaker
    closed = {}
    if resolve:
        for it in data.get("review") or []:
            a_, b_ = it.get("t1", it.get("t")), it.get("t2", it.get("t"))
            if a_ is not None and t1 - 0.01 <= a_ and b_ <= t2 + 0.01 and it.get("status", "open") != "resolved":
                closed[it["id"]] = card
    data2, rep = apply_word_edits(json_path, md_path, [e], by=by, closed=closed,
                                  log_note="правка отрезка" + (f": {note}" if note else ""))
    if card in rep["failed"]:
        raise ValueError(rep["failed"][card])
    if resolve:                               # закрытые правкой отрезка — даже если раньше оставлены человеку
        items = {i["id"]: i for i in data2.get("review") or []}
        for rid in closed:
            if items.get(rid):
                items[rid]["status"], items[rid]["note"] = "resolved", f"решено правкой отрезка {card}"
        data2["review_done"] = not any(i.get("status", "open") == "open" for i in data2.get("review") or [])
        save(data2, json_path, md_path)
    c = rep["cards"].get(card, {})
    return data2, {"card": card, "before": c.get("before"), "after": c.get("after"), "resolved": sorted(closed),
                   "conspects": rep.get("conspects")}


def reopen(json_path, md_path, ids):
    """Снова открывает места разбора (закрытые или оставленные человеку), чтобы решить их заново."""
    data = load(json_path)
    items = {i["id"]: i for i in data.get("review") or []}
    done, unknown = [], []
    for rid in ids:
        it = items.get(rid)
        if not it:
            unknown.append(rid)
            continue
        it["status"] = "open"
        it.pop("note", None)
        done.append(rid)
    data["review_done"] = not any(i.get("status", "open") == "open" for i in data.get("review") or [])
    save(data, json_path, md_path)
    return data, {"reopened": done, "unknown": unknown,
                  "open_left": sum(1 for i in data.get("review") or [] if i.get("status", "open") == "open")}


def _wkey(s):
    return re.sub(r"[^\w-]", "", s.lower().replace("ё", "е"))


def _diff_words(old, new_text, item_id):
    """Правка текста места по словам: меняются только отличающиеся слова, у остальных сохраняются время,
    уверенность и прежние правки (на время слов опираются ссылки конспекта). Слово, уже исправленное
    другой правкой, не перезаписывается — оно возвращается в списке конфликтов.
    -> (новые слова, [id правок, которые сохранены вместо этой])."""
    import difflib
    new = new_text.split()
    sm = difflib.SequenceMatcher(None, [_wkey(w["w"]) for w in old], [_wkey(x) for x in new], autojunk=False)
    out, conflicts = [], []

    def mk(text, src, s, e):
        return {"w": text, "s": s, "e": e, "c": 1.0, "edited": item_id,
                "model": " ".join(w.get("model", w.get("raw", w["w"])) for w in src) if src else ""}

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        src = old[i1:i2]
        if op == "equal":
            for w, x in zip(src, new[j1:j2]):          # пунктуация и регистр — из правки
                out.append(w if w["w"] == x else {**w, "w": x, "edited": w.get("edited", item_id)})
            continue
        prior = [w["edited"] for w in src if w.get("edited") and w["edited"] != item_id]
        if prior:
            conflicts += prior
            out.extend(src)
            continue
        if op == "delete":
            continue
        if op == "insert":
            t = out[-1]["e"] if out else (old[0]["s"] if old else 0.0)
            out.append(mk(" ".join(new[j1:j2]), [], t, t))
            continue
        if i2 - i1 == j2 - j1:                          # замена слово в слово
            out.extend(mk(x, [w], w["s"], w["e"]) for w, x in zip(src, new[j1:j2]))
        else:
            out.append(mk(" ".join(new[j1:j2]), src, src[0]["s"], src[-1]["e"]))
    return out, sorted(set(conflicts))


def _common_word(w):
    """Обычное слово русского языка (по pymorphy3, если установлен), а не искажение имени."""
    try:
        import freeform
        m = freeform._morph()
        return bool(m and len(w) > 2 and m.word_is_known(w.lower()))
    except Exception:
        return False


PROPER_TAGS = {"Name", "Surn", "Patr", "Geox", "Orgn", "Trad"}
STOP_WORDS = set("""если есть это что как так когда где вот ну да нет и а но или то там тут все всё он она оно они мы вы
я ты его её ее их мой моя наш ваш свой этот эта эти тот та те который какой какая почему зачем потому поэтому ещё
еще уже тоже также очень просто только даже вообще конечно может можно нужно надо будет было были была нам вам им
них для при про после перед через около сейчас потом тогда здесь ладно хорошо давайте давай спасибо пожалуйста
слушайте смотрите кстати например итак значит сначала затем кто чем чего кому мне меня тебе тебя себя себе""".split())


def vocab_candidate(word, data=None):
    """Годится ли слово в подсказки словаря: имя или название, а не служебное и не обычное слово.
    С pymorphy3: имя/фамилия/место/организация — да; известное обычное слово («Тетрадями») — нет; неизвестное — да.
    Без него: служебные слова — нет; слово, которое в стенограмме встречается со строчной буквы, — нет."""
    c = re.sub(r"[^\w-]", "", word or "").strip("-")
    if len(c) < 3 or not c[:1].isupper():
        return False
    if data and c in (data.get("speakers") or []):
        return False
    low = c.lower().replace("ё", "е")
    if low in STOP_WORDS or c.lower() in STOP_WORDS:
        return False
    try:
        import freeform
        m = freeform._morph()
    except Exception:
        m = None
    if m:
        try:
            parses = m.parse(c.lower())
            if any(PROPER_TAGS & set(p.tag.grammemes) for p in parses):
                return True
            if m.word_is_known(c.lower()):
                return False
            return True
        except Exception:
            pass
    if data:
        for seg in data.get("segments") or []:
            for w in seg.get("words") or []:
                x = re.sub(r"[^\w-]", "", w.get("w", ""))
                if x[:1].islower() and x.lower().replace("ё", "е") == low:
                    return False
    return True


def apply_word_edits(json_path, md_path, edits, by="human", closed=None, answers=None, log_note=None):
    """Правки по словам, а не по местам (свободные ответы человека, freeform.py; правка отрезка edit_span).
    edits: [{"op": "relabel", "t", "speaker"} | {"op": "replace", "t", "text"} |
            {"op": "insert", "s", "e", "text", "speaker", "after_t"?, "before_t"?} |
            {"op": "place_text", "t1", "t2", "text", "speaker"?} | {"op": "delete", "t", "w"} |
            {"op": "overlap_move", "t", "w", "after_t"} | {"op": "propose_delete", "t", "w"}, ... + "card", "note"?]
    Слово ищется по времени начала. Вставка ставится сразу за словом after_t (или перед before_t), в порядке
    ответа, а не по оценочному времени. overlap_move — чужое слово внутри реплики человека: переносится после
    реплики, помечается как одновременная речь. Правки одной карточки применяются целиком или никак: если
    какое-то слово не найдено или у вставки нет говорящего — карточка не применяется и идёт человеку с причиной.
    Реплика делится там, где внутри неё меняется говорящий. closed — {id места: карточка}, которые
    ответ человека покрыл (решено через другую карточку). answers — {карточка: текст ответа} для журнала.
    -> (data, отчёт)"""
    data = load(json_path)
    answers = answers or {}
    name_to_label = {v: k for k, v in (data.get("speaker_names") or {}).items()}
    flat = []
    for si, seg in enumerate(data["segments"]):
        for w in seg.get("words") or []:
            flat.append({"w": w, "spk": seg["speaker"], "seg": si})
    by_t = {}

    def reindex():
        by_t.clear()
        for k, f in enumerate(flat):
            by_t.setdefault(round(f["w"]["s"], 2), []).append(k)
    reindex()

    def find(t, text=None):
        ks = by_t.get(round(t, 2)) or []
        if text is not None:
            ks = [k for k in ks if flat[k]["w"]["w"] == text] or ks
        return ks[0] if ks else None

    log = data.setdefault("edits", [])
    pairs = {}
    snap = [(f["w"]["s"], f["w"]["w"], f["spk"]) for f in flat]
    rep = {"relabel": 0, "replace": 0, "insert": 0, "delete": 0, "overlap": 0, "not_found": [],
           "proposed_delete": [], "failed": {}, "cards": {}}
    groups = {}
    for e in edits:
        groups.setdefault(e.get("card", "?"), []).append(e)

    def check(eds):
        """Причина, по которой правки карточки нельзя применить целиком, или None."""
        for e in eds:
            op = e["op"]
            if op == "propose_delete":
                continue
            if op == "insert":
                if not e.get("speaker"):
                    return f"не удалось определить говорящего вставки «{e.get('text')}»"
                continue
            if op == "place_text":
                if not any(e["t1"] - 0.01 <= f["w"]["s"] <= e["t2"] + 0.01 for f in flat) and not e.get("speaker"):
                    return f"на отрезке {e['t1']}–{e['t2']} с нет слов, а говорящий не указан"
                continue
            if find(e["t"], e.get("w")) is None:
                return f"слово «{e.get('w')}» ({e['t']} с) не найдено — стенограмма изменилась"
        return None

    def put_after(t_anchor, card, item):
        """Позиция сразу за словом t_anchor и за уже вставленными этой карточкой словами после него."""
        k = find(t_anchor) if t_anchor is not None else None
        if k is None:
            return None
        k += 1
        while k < len(flat) and flat[k]["w"].get("inserted") and flat[k]["w"].get("edited") == card:
            k += 1
        flat.insert(k, item)
        return k

    for card, eds in groups.items():
        why = check(eds)
        if why:
            rep["failed"][card] = why
            continue
        before_words, after_words = [], []
        moves, inserts = [], []
        for e in eds:
            op = e["op"]
            if op == "propose_delete":
                rep["proposed_delete"].append({"card": card, "t": e["t"], "word": e.get("w")})
                continue
            if op == "insert":
                inserts.append(e)
                after_words.append(e["text"])
                continue
            if op == "overlap_move":
                moves.append(e)
                continue
            if op == "place_text":
                ks = [k for k, f in enumerate(flat) if e["t1"] - 0.01 <= f["w"]["s"] <= e["t2"] + 0.01]
                if not ks:                      # пустой отрезок: текст вставляется равномерно, говорящий задан
                    toks = e["text"].split()
                    step = (e["t2"] - e["t1"]) / (len(toks) + 1) if toks else 0
                    for q, x in enumerate(toks, 1):
                        inserts.append({"op": "insert", "s": round(e["t1"] + step * q, 3),
                                        "e": round(e["t1"] + step * q + step * 0.9, 3), "text": x,
                                        "speaker": e["speaker"], "card": card})
                    after_words += toks
                    continue
                old = [flat[k]["w"] for k in ks]
                new_ws, _ = _diff_words(old, e["text"], card)
                for w in new_ws:
                    w["edited"], w["by"] = card, by
                pos = ks[0]
                spk, si = e.get("speaker") or flat[pos]["spk"], flat[pos]["seg"]
                if spk != flat[pos]["spk"]:
                    pairs.setdefault(card, []).append([flat[pos]["spk"], spk])
                for k in sorted(ks, reverse=True):
                    flat.pop(k)
                for q, w in enumerate(new_ws):
                    flat.insert(pos + q, {"w": w, "spk": spk, "seg": si})
                reindex()
                before_words += [w["w"] for w in old]
                after_words += [w["w"] for w in new_ws]
                rep["replace"] += 1
                continue
            k = find(e["t"], e.get("w"))
            f = flat[k]
            if op == "delete":
                before_words.append(f["w"]["w"])
                flat.pop(k)
                reindex()
                rep["delete"] += 1
            elif op == "relabel":
                if f["spk"] != e["speaker"]:
                    pairs.setdefault(card, []).append([f["spk"], e["speaker"]])
                    before_words.append(f["w"]["w"])
                    f["spk"] = e["speaker"]
                    f["w"]["by"], f["w"]["edited"] = by, card
                    rep["relabel"] += 1
            elif op == "replace":
                if f["w"]["w"] != e["text"]:
                    before_words.append(f["w"]["w"])
                    f["w"].setdefault("model", f["w"].get("raw", f["w"]["w"]))
                    f["w"]["w"] = e["text"]
                    f["w"]["by"], f["w"]["edited"], f["w"]["c"] = by, card, 1.0
                    if e.get("note"):
                        f["w"]["typo"] = e["note"]
                    after_words.append(e["text"])
                    rep["replace"] += 1
        # вставки: сразу за опорным словом ответа (в порядке ответа), иначе перед следующим, иначе по времени
        for e in sorted(inserts, key=lambda e: e["s"]):
            w = {"w": e["text"], "s": e["s"], "e": max(e["s"], e.get("e", e["s"])), "c": 1.0, "by": by,
                 "edited": card, "inserted": True}
            item = {"w": w, "spk": e["speaker"], "seg": -1}
            pos = put_after(e.get("after_t"), card, item)
            if pos is None and e.get("before_t") is not None and find(e["before_t"]) is not None:
                pos = find(e["before_t"])
                while pos > 0 and flat[pos - 1]["w"].get("inserted") and flat[pos - 1]["w"].get("edited") == card:
                    pos -= 1
                pos += sum(1 for x in inserts if x is not e and x.get("before_t") == e["before_t"]
                           and x["s"] < e["s"])
                flat.insert(pos, item)
            elif pos is None:
                pos = next((k for k, f in enumerate(flat) if f["w"]["s"] > e["s"]), len(flat))
                flat.insert(pos, item)
            # реплика: соседа того же голоса
            nb = [flat[x] for x in (pos - 1, pos + 1) if 0 <= x < len(flat) and flat[x]["spk"] == e["speaker"]
                  and flat[x]["seg"] >= 0]
            item["seg"] = nb[0]["seg"] if nb else -1
            reindex()
            rep["insert"] += 1
        # наложение: чужое слово — после реплики человека целиком
        for e in moves:
            k = find(e["t"], e.get("w"))
            f = flat.pop(k)
            reindex()
            f["w"]["overlap"] = True
            f["w"]["by"] = f["w"].get("by") or "model"
            if put_after(e.get("after_t"), card, f) is None:
                flat.insert(k, f)
            reindex()
            rep["overlap"] += 1
            before_words.append(f["w"]["w"])
        if before_words or after_words or moves:
            ts = [x.get(k) for x in eds for k in ("t", "s", "t1", "t2", "after_t") if x.get(k) is not None
                  and x["op"] != "propose_delete"]
            rep["cards"][card] = {"t1": min(ts), "t2": max(ts)}
    # реплики заново: граница — смена говорящего или исходной реплики; одновременная речь — отдельной репликой
    old = data["segments"]
    segs = []
    for f in flat:
        cur = segs[-1] if segs else None
        ov = bool(f["w"].get("overlap"))
        if cur and cur["speaker"] == f["spk"] and cur.get("_ov", False) == ov and \
                (cur["_seg"] == f["seg"] or f["seg"] < 0 or cur["_seg"] < 0
                 or f["w"]["s"] - cur["words"][-1]["e"] < 1.5):
            cur["words"].append(f["w"])
            continue
        base = old[f["seg"]] if 0 <= f["seg"] < len(old) else {}
        ns = {k: v for k, v in base.items() if k not in ("words", "inserts", "emotion", "text", "start", "end",
                                                          "overlap")}
        ns.update(speaker=f["spk"], words=[f["w"]], _seg=f["seg"], _ov=ov,
                  label=name_to_label.get(f["spk"], base.get("label", f["spk"]) if base.get("speaker") == f["spk"] else f["spk"]))
        if base.get("speaker") == f["spk"] and base.get("emotion"):
            ns["emotion"] = base["emotion"]
        segs.append(ns)
    used_ins = set()
    for ns in segs:
        ns["start"], ns["end"] = min(w["s"] for w in ns["words"]), max(w["e"] for w in ns["words"])
        ns["text"] = " ".join(w["w"] for w in ns["words"])
        si = ns.pop("_seg")
        if ns.pop("_ov"):
            ns["overlap"] = True
        if 0 <= si < len(old) and old[si].get("inserts") and si not in used_ins:
            ns["inserts"] = old[si]["inserts"]
            used_ins.add(si)
    data["segments"] = segs

    def txt(rows):
        out, last = [], None
        for _, w, sp in rows:
            out.append((f"{sp}: " if sp != last else "") + w)
            last = sp
        return " ".join(out)

    def who(rows):
        return ", ".join(dict.fromkeys(sp for _, _, sp in rows if sp))

    final = [(w["s"], w["w"], sg["speaker"]) for sg in segs for w in sg["words"]]
    for card, rg in rep["cards"].items():
        a_, b_ = rg["t1"] - 0.01, rg["t2"] + 0.01
        brows = [r for r in snap if a_ <= r[0] <= b_]
        arows = [r for r in final if a_ <= r[0] <= b_ + 0.05]
        bt, at = txt(brows), txt(arows)
        log.append({"id": card, "t": rg["t1"], "by": by, "kind": "words",
                    "before": {"text": bt, "speaker": who(brows)}, "after": {"text": at, "speaker": who(arows)},
                    "answer": answers.get(card), "note": log_note or "ответ человека в карточке",
                    "pairs": sorted({tuple(p) for p in pairs.get(card, [])})})
        rg.update(before=bt, after=at)
    # места: закрытые ответом на другую карточку и сами карточки
    items = {i["id"]: i for i in data.get("review") or []}
    for rid, via in (closed or {}).items():
        if any(via == c for c in rep["failed"]):
            continue
        it = items.get(rid)
        if it and it.get("status", "open") != "human":
            it["status"], it["note"] = "resolved", f"решено человеком (через {via})"
    for card in groups:
        it = items.get(card)
        if not it:
            continue
        if card in rep["failed"]:
            it["status"] = "human"
            it["note"] = (it.get("note") + "; " if it.get("note") else "") + \
                f"ответ не применён: {rep['failed'][card]}" + (f": «{answers[card]}»" if answers.get(card) else "")
        else:
            it["status"] = "resolved"
    for d in rep["proposed_delete"]:
        it = items.get(d["card"])
        if it:
            it["status"] = "human"
            it["note"] = (it.get("note") + "; " if it.get("note") else "") + \
                f"человек не назвал «{d['word']}» — удалить?"
    # имена и названия из правок человека (не в начале фразы) — в подсказки словаря
    seen = {v.get("to") for v in data.get("vocab_suggestions") or []}
    for e in edits:
        if e["op"] not in ("insert", "replace", "place_text") or e.get("card") in rep["failed"]:
            continue
        toks = str(e.get("text", "")).split()
        for q, x in enumerate(toks):
            c = re.sub(r"[^\w-]", "", x)
            start = q == 0 and e["op"] != "replace" or (q > 0 and re.search(r"[.!?…:]$", toks[q - 1]))
            if not start and c not in seen and vocab_candidate(c, data):
                seen.add(c)
                data.setdefault("vocab_suggestions", []).append({"to": c, "from": None, "card": e.get("card"),
                                                                 "add": {"terms": [c]}})
    data["review_done"] = not any(i.get("status", "open") == "open" for i in data.get("review") or [])
    save(data, json_path, md_path)
    rep["open_left"] = sum(1 for i in data.get("review") or [] if i.get("status", "open") == "open")
    rep["conspects"] = _refresh_conspects(json_path)
    return data, rep


def apply_review(json_path, md_path, decisions, by="claude"):
    """decisions: [{"id": "r5", "speaker"?: имя, "text"?: новый текст места, "status"?: "resolved"|"human",
    "note"?: str}]. Без speaker и text — место принято как есть. status "human" — оставить человеку.
    Возвращает (data, отчёт)."""
    data = load(json_path)
    items = {i["id"]: i for i in data.get("review") or []}
    name_to_label = {v: k for k, v in (data.get("speaker_names") or {}).items()}
    report = {"applied": 0, "kept": 0, "human": 0, "unknown": [], "successful_ids": []}
    edits = data.setdefault("edits", [])
    # правки по убыванию времени, чтобы разрезание реплик не сдвигало следующие места
    for d in sorted(decisions, key=lambda d: -(items.get(d.get("id"), {}).get("t", 0))):
        it = items.get(d.get("id"))
        if not it:
            report["unknown"].append(d.get("id"))
            continue
        if d.get("status") == "human":
            it["status"], it["note"] = "human", d.get("note") or it.get("note")
            report["human"] += 1
            continue
        loc = _find_span(data, it)
        if loc is None:
            report.setdefault("not_found", []).append(it["id"])     # место исчезло (перекрыто другой правкой)
            continue
        same_spk = loc is not None and d.get("speaker") in (None, data["segments"][loc[0]]["speaker"])
        if loc is None or (not d.get("text") and same_spk):
            it["status"] = "resolved"
            report["kept"] += 1
            if loc is not None and by == "human":           # человек подтвердил место как есть
                for w in data["segments"][loc[0]]["words"][loc[1]:loc[2]]:
                    w["by"] = "human"
            report["successful_ids"].append(it["id"])
            continue
        si, w1, w2 = loc
        seg = data["segments"][si]
        ws = seg["words"]
        before = {"text": " ".join(w["w"] for w in ws[w1:w2]), "speaker": seg["speaker"]}
        if (not d.get("text") or d["text"] == before["text"]) and d.get("speaker") in (None, seg["speaker"]):
            it["status"] = "resolved"
            report["kept"] += 1
            report["successful_ids"].append(it["id"])
            continue
        new_text = d.get("text")
        if new_text and new_text != before["text"]:
            model = [w.get("raw", w["w"]) for w in ws[w1:w2]]
            new_ws, conflicts = _diff_words(ws[w1:w2], new_text, it["id"])
            if conflicts:
                report.setdefault("conflicts", []).append({"id": it["id"], "kept_edits": conflicts})
                continue
            ws[w1:w2] = new_ws
            w2 = w1 + len(new_ws)
            # одиночная замена слова на имя собственное — подсказка для словаря
            if len(model) == 1 and len(new_text.split()) == 1 and vocab_candidate(new_text, data) \
                    and re.sub(r"\W", "", new_text.lower()) != re.sub(r"\W", "", model[0].lower()):
                sug = {"from": re.sub(r"[^\w-]", "", model[0]), "to": re.sub(r"[^\w-]", "", new_text)}
                if _common_word(sug["from"]):
                    sug["common"] = True
                    sug["note"] = (f"«{sug['from']}» — обычное слово: в словарь лучше заменой с условием "
                                   "(replace_if), а не в replace")
                    sug["add"] = {"replace_if": [{"from": sug["from"], "to": sug["to"], "after": []}]}
                else:
                    sug["add"] = {"replace": {sug["from"]: sug["to"]}}
                data.setdefault("vocab_suggestions", []).append(sug)
        new_spk = d.get("speaker") or seg["speaker"]
        if by == "human":
            for w in ws[w1:w2]:
                w["by"] = "human"
        if new_spk != seg["speaker"]:
            parts = [ws[:w1], ws[w1:w2], ws[w2:]]
            spk = [seg["speaker"], new_spk, seg["speaker"]]
            new_segs = []
            for p, sp in zip(parts, spk):
                if not p:
                    continue
                ns = {**{k: v for k, v in seg.items() if k not in ("words", "inserts", "emotion")},
                      "start": p[0]["s"], "end": p[-1]["e"], "speaker": sp,
                      "label": name_to_label.get(sp, sp if sp != seg["speaker"] else seg.get("label", sp)),
                      "words": p, "text": " ".join(w["w"] for w in p)}
                new_segs.append(ns)
            if seg.get("inserts"):
                new_segs[0]["inserts"] = seg["inserts"]
            data["segments"][si:si + 1] = new_segs
        else:
            seg["text"] = " ".join(w["w"] for w in ws)
        edits.append({"id": it["id"], "t": it["t"], "by": by, "before": before,
                      "after": {"text": " ".join(w["w"] for w in (data["segments"][si]["words"] if new_spk == seg["speaker"]
                                                                   else ws)[w1:w2]) if new_text else before["text"],
                                "speaker": new_spk}, "note": d.get("note")})
        it["status"] = "resolved"
        report["applied"] += 1
        report["successful_ids"].append(it["id"])
    # соседние реплики одного говорящего после правок склеиваем
    merged = []
    for seg in data["segments"]:
        if merged and merged[-1]["speaker"] == seg["speaker"] and seg["start"] - merged[-1]["end"] < 1.5 \
                and seg.get("words") and merged[-1].get("words"):
            m = merged[-1]
            m["words"] = m["words"] + seg["words"]
            m["end"] = seg["end"]
            m["text"] = " ".join(w["w"] for w in m["words"])
            if seg.get("inserts"):
                m.setdefault("inserts", []).extend(seg["inserts"])
        else:
            merged.append(seg)
    data["segments"] = merged
    # решение применено, но место ещё раз показать человеку (например, ответы двух режимов разошлись)
    for d in decisions:
        it = items.get(d.get("id"))
        if it and d.get("human_after"):
            it["status"], it["note"] = "human", d.get("note") or it.get("note")
            report["human"] += 1
        elif it and d.get("reopen"):                 # ответ отменён — место снова ждёт разбора
            it["status"] = "open"
            it.pop("note", None)
    data["review_done"] = not any(i.get("status", "open") == "open" for i in data.get("review") or [])
    save(data, json_path, md_path)
    report["open_left"] = sum(1 for i in data.get("review") or [] if i.get("status", "open") == "open")
    report["vocab_suggestions"] = data.get("vocab_suggestions", [])
    report["conspects"] = _refresh_conspects(json_path)
    return data, report
