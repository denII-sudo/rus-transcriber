"""Пакет разбора: короткие клипы сомнительных мест + manifest.json с вопросами и вариантами,
и answers.json — ответы из локального и облачного режимов.

Папка пакета: <подпапка результатов>/<имя записи>[.tag].review/
    manifest.json   {"version", "job_id", "audio", "speakers", "items": [...]}
    clips/r5.mp3    клип места: 3,5 с до и 1,5 с после (mp3, моно, 64 кбит/с — играет везде, и на iPhone)
    answers.json    {"version", "answers": {id: ответ}, "conflicts": [...], "rejected": [...]}
    voice/          голосовые ответы (локальная страница и перенесённые из облака)
    cards.html      страница карточек (та же, что публикуется в облако)

Элемент manifest:
    id, t (чч:мм:сс), t1/t2 (время места в записи), clip, clip_t1/clip_t2 (место внутри клипа),
    speaker, text, reply (реплика с ⟦…⟧), before/after, reasons, voices,
    question, options [{label, decision: {speaker | text | status, note}}], recommended (номер варианта).
Ответ:
    {id, version, source: local|cloud, ts, kind: option|recommended|human|skip|free|repeat,
     option (номер), text (распознанный или набранный), audio (путь к голосовому ответу)}
Ответ на одно место из двух режимов: побеждает более поздний, расхождение — в conflicts (для человека).
Ответ с другой version (пакет пересобран) не принимается — в rejected.

Запуск отдельно (распознавание голосовых ответов, нужно отдельным процессом, чтобы сервер MCP не держал модель):
    python review_pack.py recognize <файл> [<файл> ...]   → JSON {файл: текст}
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PAD_BEFORE = 3.5     # клип начинается за 3,5 с до места (было 1,5)
PAD_AFTER = 1.5
SR = 16000
HERE = Path(__file__).resolve().parent


def pack_dir(json_path):
    jp = Path(json_path)
    stem = jp.name[:-len(".transcript.json")] if jp.name.endswith(".transcript.json") else jp.stem
    return jp.with_name(f"{stem}.review")


def _now():
    """Время ответа в UTC — как пишет облачная страница (иначе «более поздний» сравнивался бы по разным поясам)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _read(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in range(20):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            time.sleep(0.05 * (k + 1))
    raise RuntimeError(f"не удалось записать {p}")


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg не найден в PATH")
    return exe


def cut_clip(src, t1, t2, out):
    s = max(0.0, t1 - PAD_BEFORE)
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{s:.2f}", "-to", f"{t2 + PAD_AFTER:.2f}",
                    "-i", str(src), "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "64k", str(out)],
                   check=True)
    return s


# ---------- сборка пакета ----------
def build(json_path, items, base_url_note=None):
    """items: [{id, question, options: [{label, decision}], recommended}] — вопросы Claude по местам review.
    -> отчёт. Пересборка меняет version: ответы на прежние вопросы не применятся."""
    import transcript as T
    data = T.load(json_path)
    src = Path(data.get("file") or "")
    if not src.exists():
        raise RuntimeError(f"исходная запись не найдена: {src}")
    review = {it["id"]: it for it in data.get("review") or []}
    rows = {it["id"]: _context(T, data, it) for it in data.get("review") or []}
    pd = pack_dir(json_path)
    (pd / "clips").mkdir(parents=True, exist_ok=True)
    out, skipped = [], []
    for q in items:
        it = review.get(q.get("id"))
        if not it:
            skipped.append({"id": q.get("id"), "why": "нет такого места"})
            continue
        opts = q.get("options") or []
        if not 1 <= len(opts) <= 4:
            skipped.append({"id": it["id"], "why": "нужно 1–4 варианта"})
            continue
        t1, t2 = float(it.get("t1", it["t"])), float(it.get("t2", it["t"]))
        clip = pd / "clips" / f"{it['id']}.mp3"
        s0 = cut_clip(src, t1, t2, clip)
        row = rows.get(it["id"], {})
        out.append({
            "id": it["id"], "t": T.hms(it["t"]), "t1": round(t1, 2), "t2": round(t2, 2),
            "clip": f"clips/{clip.name}", "clip_t1": round(t1 - s0, 2), "clip_t2": round(t2 - s0, 2),
            "speaker": it.get("speaker"), "text": it.get("text"),
            "reply": row.get("reply"), "before": row.get("before"), "after": row.get("after"),
            "reasons": it.get("reasons"), "voices": it.get("voices"),
            "question": q.get("question") or "Как правильно?",
            "options": [{"label": str(o.get("label", "")), "decision": o.get("decision") or {}} for o in opts],
            "recommended": q.get("recommended"), "note": q.get("note")})
    keep = {f"{r['id']}.mp3" for r in out}
    for f in (pd / "clips").glob("*.mp3"):
        if f.name not in keep:
            f.unlink()
    body = json.dumps(out, ensure_ascii=False, sort_keys=True)
    version = time.strftime("%Y%m%d%H%M%S-") + hashlib.sha1(body.encode()).hexdigest()[:6]
    man = {"version": version, "job_json": str(json_path), "audio": src.name,
           "recording_id": data.get("recording_id"), "source_revision": data.get("source_revision"),
           "speakers": data.get("speakers"), "created": _now(), "items": out}
    _write(pd / "manifest.json", man)
    ans = _read(pd / "answers.json", {})
    if ans.get("version") != version:          # прежние ответы — к прежним вопросам; храним для истории
        if ans.get("answers"):
            _write(pd / f"answers.{ans.get('version', 'old')}.json", ans)
        _write(pd / "answers.json", {"version": version, "answers": {}, "conflicts": [], "rejected": []})
    shutil.copyfile(HERE / "cards.html", pd / "cards.html")
    return {"pack": str(pd), "version": version, "items": len(out), "skipped": skipped,
            "clips_bytes": sum(f.stat().st_size for f in (pd / "clips").glob("*.mp3"))}


CTX_BEFORE = 400      # символов соседних реплик до места
CTX_AFTER = 400
CTX_IN_REPLY = 700    # символов до и после места внутри его реплики; ⟦место⟧ всегда видно


def _context(T, data, it):
    """Реплика с выделенным местом ⟦…⟧ и соседние реплики (для любого места, и уже разобранного).
    Если реплика до места короче CTX_BEFORE, добавляется ещё одна предыдущая."""
    loc = T._find_span(data, it)
    if not loc and it.get("gap"):                 # пропущенная речь: реплики вокруг промежутка
        s, e = it["gap"]
        prev = [g for g in data["segments"] if g["end"] <= s + 0.05]
        nxt = [g for g in data["segments"] if g["start"] >= e - 0.05]
        row = {"reply": f"{it.get('speaker') or '?'}: ⟦… речь без распознанных слов, {round(e - s, 1)} с …⟧"}
        if prev:
            row["before"] = "\n".join(f"{g['speaker']}: {g['text']}" for g in prev[-2:])[-CTX_BEFORE:]
        if nxt:
            row["after"] = "\n".join(f"{g['speaker']}: {g['text']}" for g in nxt[:2])[:CTX_AFTER]
        return row
    if not loc:
        return {}
    si, w1, w2 = loc
    seg = data["segments"][si]
    ws = seg.get("words") or []
    if ws:
        head = " ".join(w["w"] for w in ws[:w1])
        tail = " ".join(w["w"] for w in ws[w2:])
        if len(head) > CTX_IN_REPLY:
            head = "…" + head[-CTX_IN_REPLY:].split(" ", 1)[-1]
        if len(tail) > CTX_IN_REPLY:
            tail = tail[:CTX_IN_REPLY].rsplit(" ", 1)[0] + "…"
        mid = "⟦" + " ".join(w["w"] for w in ws[w1:w2]) + "⟧"
        text = " ".join(x for x in (head, mid, tail) if x)
    else:
        text = T._ctx_text(seg, w1, w2)
    row = {"reply": f"{seg['speaker']}: {text}"}
    before, k = [], si - 1
    while k >= 0 and sum(len(x) for x in before) < CTX_BEFORE and len(before) < 6:
        pv = data["segments"][k]
        before.insert(0, f"{pv['speaker']}: {pv['text']}")
        k -= 1
    if before:
        b = "\n".join(before)
        row["before"] = ("…" + b[-CTX_BEFORE:]) if len(b) > CTX_BEFORE else b
    after, k = [], si + 1
    while k < len(data["segments"]) and sum(len(x) for x in after) < CTX_AFTER and len(after) < 6:
        nx = data["segments"][k]
        after.append(f"{nx['speaker']}: {nx['text']}")
        k += 1
    if after:
        a = "\n".join(after)
        row["after"] = (a[:CTX_AFTER] + "…") if len(a) > CTX_AFTER else a
    return row


def manifest(json_path):
    return _read(pack_dir(json_path) / "manifest.json", None)


# ---------- сопоставление голосового/текстового ответа с вариантами ----------
_ORD = [("перв", 0), ("один", 0), ("перый", 0), ("втор", 1), ("два", 1), ("трет", 2), ("три", 2),
        ("четв", 3), ("четыр", 3)]


def _norm(s):
    return re.sub(r"[^\wё ]+", " ", (s or "").lower().replace("ё", "е")).split()


def match(text, item):
    """-> {"kind": option|recommended|human|skip|repeat|free, "option": n?}"""
    w = _norm(text)
    j = " ".join(w)
    if not w:
        return {"kind": "skip", "why": "пустой ответ"}
    if re.search(r"повтор|еще раз|не расслыша|не понял", j):
        return {"kind": "repeat"}
    if re.search(r"человек|оставь|оставить|не знаю|не уверен", j):
        return {"kind": "human"}
    if re.search(r"пропус|дальше|следующ", j) and len(w) <= 3:
        return {"kind": "skip"}
    opts = item.get("options") or []
    if len(w) <= 4:
        for x in w:
            if x.isdigit() and 1 <= int(x) <= len(opts):
                return {"kind": "option", "option": int(x) - 1}
        for x in w:
            for stem, n in _ORD:
                if x.startswith(stem) and n < len(opts):
                    return {"kind": "option", "option": n}
        if re.search(r"рекоменд|как предлож|согла|верно|правильно|^да$|^ага$|^угу$", j) and \
                item.get("recommended") is not None:
            return {"kind": "recommended", "option": item["recommended"]}
    # совпадение по словам варианта (имена, слова текста): берём вариант с наибольшим числом
    # отличительных слов — тех, что есть только в нём
    sets = [set(_norm(o["label"])) for o in opts]
    best, hits = None, 0
    for n, s in enumerate(sets):
        own = s - set().union(*(x for k, x in enumerate(sets) if k != n)) if len(sets) > 1 else s
        h = sum(1 for x in w if any(x[:5] == y[:5] for y in own if len(y) > 2))
        if h > hits:
            best, hits = n, h
        elif h == hits and h:
            best = None
    if best is not None and hits and len(w) <= 6:
        return {"kind": "option", "option": best}
    return {"kind": "free"}


# ---------- ответы ----------
def _same(a, b):
    return (a.get("kind"), a.get("option"), (a.get("text") or "").strip() if a.get("kind") == "free" else None) == \
           (b.get("kind"), b.get("option"), (b.get("text") or "").strip() if b.get("kind") == "free" else None)


def add_answers(json_path, answers, source):
    """Сводит ответы в answers.json. answers: [{id, version, kind|option|text|audio, ts?}].
    Голосовые ответы без текста распознаются GigaAM (отдельный процесс) и сопоставляются с вариантами."""
    pd = pack_dir(json_path)
    man = manifest(json_path)
    if not man:
        raise RuntimeError("пакет разбора не собран: сначала build_review_pack")
    by_id = {it["id"]: it for it in man["items"]}
    store = _read(pd / "answers.json", {"version": man["version"], "answers": {}, "conflicts": [], "rejected": []})
    need_asr = [a for a in answers if a.get("audio") and not a.get("text")]
    if need_asr:
        heard = recognize_files([a["audio"] for a in need_asr])
        for a in need_asr:
            a["text"] = heard.get(str(a["audio"]), "")
            a["heard"] = True
    rep = {"accepted": [], "rejected": [], "conflicts": [], "free": []}
    for a in answers:
        aid = a.get("id")
        it = by_id.get(aid)
        if not it:
            rep["rejected"].append({"id": aid, "why": "нет в пакете"})
            continue
        if a.get("version") != man["version"]:
            rej = {"id": aid, "why": "ответ на прежнюю версию вопросов", "version": a.get("version")}
            rep["rejected"].append(rej)
            store.setdefault("rejected", []).append({**rej, "source": source, "ts": a.get("ts") or _now()})
            continue
        ans = {"id": aid, "version": a["version"], "source": source, "ts": a.get("ts") or _now()}
        if isinstance(a.get("text"), str):          # пометки сомнения ⟦…⟧ из скопированной стенограммы
            a["text"] = a["text"].replace("⟦", "").replace("⟧", "")
        for k in ("text", "audio", "heard", "by"):
            if a.get(k) is not None:
                ans[k] = a[k]
        if a.get("kind") in ("human", "skip", "recommended", "free", "repeat", "revert"):
            ans["kind"] = a["kind"]
            if a["kind"] == "recommended":
                ans["option"] = it.get("recommended")
        elif a.get("option") is not None:
            ans.update(kind="option", option=int(a["option"]))
        elif a.get("heard"):                     # голос: «второй», «да», имя из варианта…
            ans.update(match(a.get("text"), it))
        else:                                     # набранный текст: вариант — только при точном совпадении
            t = " ".join(_norm(a.get("text")))
            hit = [n for n, o in enumerate(it["options"]) if " ".join(_norm(o["label"])) == t and t]
            ans.update({"kind": "option", "option": hit[0]} if hit else {"kind": "free"})
        if ans["kind"] == "option" and not 0 <= ans["option"] < len(it["options"]):
            rep["rejected"].append({"id": aid, "why": "нет такого варианта"})
            continue
        if ans["kind"] in ("repeat", "skip") and not ans.get("text"):
            rep["accepted"].append({"id": aid, "kind": ans["kind"], "stored": False})
            continue
        old = store["answers"].get(aid)
        if old and old.get("source") != source and not _same(old, ans):
            c = {"id": aid, "was": old, "now": ans}
            store.setdefault("conflicts", []).append(c)
            rep["conflicts"].append({"id": aid, "was": _desc(old, it), "now": _desc(ans, it)})
        if old and old.get("ts", "") > ans["ts"]:          # более поздний ответ побеждает
            continue
        store["answers"][aid] = ans
        rep["accepted"].append({"id": aid, "kind": ans["kind"], "answer": _desc(ans, it)})
        if ans["kind"] == "free":
            rep["free"].append({"id": aid, "text": ans.get("text")})
    _write(pd / "answers.json", store)
    rep["answered"] = len(store["answers"])
    rep["total"] = len(man["items"])
    return rep


def _desc(a, it):
    k = a.get("kind")
    if k in ("option", "recommended") and a.get("option") is not None:
        return f"{a['option'] + 1}: {it['options'][a['option']]['label']}"
    return {"human": "оставить человеку", "skip": "пропущено", "revert": "ответ отменён",
            "free": f"свой ответ: {a.get('text')}"}.get(k, k)


def _sig(a):
    return [a.get("kind"), a.get("option"), (a.get("text") or "").strip() if a.get("kind") == "free" else None]


def _applied_map(store):
    """{id: подпись применённого ответа}. Для пакетов до появления этой записи — ответы не позже applied_at."""
    m = dict(store.get("applied_answers") or {})
    if "applied_answers" not in store and store.get("applied_at"):
        for aid, a in store.get("answers", {}).items():
            if (a.get("ts") or "") <= store["applied_at"]:
                m[aid] = _sig(a)
    return m


def _original(data, aid):
    """Текст и говорящий места до первой правки по нему (как распознала модель) и сейчас."""
    eds = [x for x in data.get("edits") or [] if x.get("id") == aid]
    if not eds:
        return None, None
    return eds[0]["before"], eds[-1]["after"]


def decisions(json_path):
    """Ответы пакета -> решения для apply_review. -> (decisions, отчёт).
    Уже применённый и не изменившийся ответ повторно не применяется. Если ответ на место изменили после
    применения (на карточке «Изменить ответ») — прежняя правка откатывается к тексту и говорящему модели
    и применяется новый ответ; ответ «отменить» только откатывает и снова открывает место."""
    import transcript as T
    pd = pack_dir(json_path)
    man = manifest(json_path)
    store = _read(pd / "answers.json", {"answers": {}})
    data = T.load(json_path)
    applied = _applied_map(store)
    by_id = {it["id"]: it for it in man["items"]}
    conflicted = {c["id"] for c in store.get("conflicts") or []}
    out, rep = [], {"applied_from_answers": 0, "human": [], "free": [], "skipped": [], "unanswered": [],
                    "unchanged": [], "changed": [], "reverted": []}
    for aid, it in by_id.items():
        a = store["answers"].get(aid)
        if not a:
            rep["unanswered"].append(aid)
            continue
        if aid in applied and applied[aid] == _sig(a):
            rep["unchanged"].append(aid)
            continue
        if aid in applied:
            rep.setdefault("undo_conflicts", []).append({"id": aid, "why":
                "для безопасной отмены нет проверенного структурного снимка"})
            continue
        src = f"разбор ({a['source']}, {a['ts']})"
        k = a["kind"]
        if k in ("option", "recommended"):
            d = dict(it["options"][a["option"]]["decision"] or {})
            note = f"{src}: {it['options'][a['option']]['label']}"
            if aid in conflicted:
                was = next((c["was"] for c in reversed(store["conflicts"]) if c["id"] == aid), {})
                note += (f" — ответы двух режимов расходились (раньше: {_desc(was, it)}), взят более поздний;"
                         " проверить человеку")
                d["human_after"] = True
                rep["human"].append(aid)
            d["note"] = (d.get("note") + "; " if d.get("note") else "") + note
            rep["applied_from_answers"] += 1
        elif k == "human":
            d = {"human_after": True, "note": f"{src}: оставлено человеку"
                 + (f" («{a['text']}»)" if a.get("text") else "")}
            rep["human"].append(aid)
        elif k == "free":
            # свободный ответ — правка услышанного в клипе: применяется по словам (apply_free, freeform.py)
            if aid in applied:
                d = {"human_after": True, "note": f"{src}: свой ответ изменён после применения — проверить: "
                                                  f"«{a.get('text')}»"}
                rep["human"].append(aid)
            else:
                rep["free"].append({"id": aid, "text": a.get("text")})
                continue
        elif k == "revert":
            d = {"reopen": True, "note": f"{src}: ответ отменён, место снова открыто"}
        else:
            rep["skipped"].append(aid)
            continue
        if aid in applied:                         # ответ меняли после применения: сначала откат к модели
            orig, cur = _original(data, aid)
            if orig and cur:
                if "speaker" not in d and orig.get("speaker") != cur.get("speaker"):
                    d["speaker"] = orig["speaker"]
                if "text" not in d and orig.get("text") != cur.get("text"):
                    d["text"] = orig["text"]
            rep["changed" if k != "revert" else "reverted"].append(aid)
            if k != "revert":
                d["note"] = d["note"] + " (прежний ответ отменён)"
        elif k == "revert":
            rep["skipped"].append(aid)
            continue
        out.append({"id": aid, **d})
    return out, rep


def free_answers(json_path):
    """Свободные ответы пакета, ещё не применённые: {id: ответ}."""
    pd = pack_dir(json_path)
    store = _read(pd / "answers.json", {"answers": {}})
    applied = _applied_map(store)
    return {aid: a for aid, a in store.get("answers", {}).items() if a.get("kind") == "free" and aid not in applied}


def apply_free(json_path, md_path, free):
    """Свободные ответы → правки слов (freeform.plan) → transcript.apply_word_edits. -> отчёт для человека."""
    import freeform as F
    import transcript as T
    man = manifest(json_path)
    data = T.load(json_path)
    vocab_words = []
    try:
        import vocab as V
        vp = (data.get("quality") or {}).get("vocab")
        vd = json.loads(Path(vp).read_text(encoding="utf-8-sig")) if vp and Path(vp).exists() else {}
        vocab_words = list(vd.get("terms") or []) + [p.get("name") for p in vd.get("participants") or [] if p.get("name")]
    except Exception:
        pass
    edits, reports, closed, conflicts = F.plan(man, free, data, vocab_words)
    conflict_ids = {cid for conflict in conflicts for cid in conflict.get("cards", [])}
    blocked = conflict_ids | {r["id"] for r in reports if r.get("human") or r.get("proposed_delete")}
    edits = [e for e in edits if e.get("card") not in blocked]
    closed = {rid: via for rid, via in closed.items() if via not in blocked}
    _, rep = T.apply_word_edits(json_path, md_path, edits, by="human", closed=closed,
                                answers={k: v.get("text") for k, v in free.items()})
    items = {r["id"]: r for r in data.get("review") or []}
    human = []
    for r in reports:
        if r["id"] in conflict_ids:
            human.append({"id": r["id"], "why": "ответы на карточки конфликтуют", "answer": r["answer"]})
            continue
        if r.get("proposed_delete"):
            human.append({"id": r["id"], "why": "удаление слов требует проверки", "answer": r["answer"]})
            continue
        if r.get("human"):
            human.append({"id": r["id"], "why": r["human"], "answer": r["answer"]})
    if human:                                     # не понятые ответы — человеку, с текстом ответа
        T.apply_review(json_path, md_path, [{"id": h["id"], "status": "human",
                                             "note": f"{h['why']}: «{h['answer']}»"} for h in human], by="human")
    return {"cards": [{"id": r["id"], "answer": r["answer"], "kind": r["kind"],
                       "was_now": rep["cards"].get(r["id"], {}).get("before", "") + " → " +
                       rep["cards"].get(r["id"], {}).get("after", "") if r["id"] in rep["cards"] else None,
                       "applied": r["applied"], "inserted": r["inserted"], "typos": r["typos"]} for r in reports],
            "closed_via": closed, "conflicts": conflicts, "left_to_human": human,
            "successful_ids": [r["id"] for r in reports if r["id"] in rep["cards"]
                               and r["id"] not in rep["failed"] and r["id"] not in conflict_ids
                               and not r.get("human")
                               and not any(d["card"] == r["id"] for d in rep["proposed_delete"])],
            "proposed_delete": rep["proposed_delete"], "not_found": rep["not_found"],
            "counts": {k: rep[k] for k in ("relabel", "replace", "insert")},
            "vocab_suggestions": [v for v in T.load(json_path).get("vocab_suggestions") or [] if v.get("card")],
            "conspects": rep.get("conspects")}


def cloud_bundle(json_path, artifact_url=None):
    """Готовит пакет к публикации в облако (приватный артефакт Claude): папка <пакет>/cloud с index.html
    (страница карточек с названием записи) и manifest.json, в который добавлены последние ответы с ноутбука,
    отметка «применено» и есть ли ещё клипы. Карточки работают в облаке сами по себе — на любом устройстве,
    где открыта задача; ответы копятся в базе страницы и переносятся сюда через import_answers.
    artifact_url — запомнить адрес страницы (cloud.json), чтобы обновлять ту же страницу из любой сессии.
    -> {"base": папка пакета, "files": {путь на странице: путь относительно base}, "remove": [клипы, которых
    больше нет], "artifact_url"}"""
    pd = pack_dir(json_path)
    man = manifest(json_path)
    if not man:
        raise RuntimeError("пакет разбора не собран")
    store = _read(pd / "answers.json", {"answers": {}})
    meta = _read(pd / "cloud.json", {})
    if artifact_url:
        meta["artifact_url"] = artifact_url
    clips = {it["clip"]: pd / it["clip"] for it in man["items"]}
    have = {k: v for k, v in clips.items() if v.exists()}
    out = dict(man, answers={k: v for k, v in store.get("answers", {}).items() if v.get("version") == man["version"]},
               applied=bool(store.get("applied_at")), applied_at=store.get("applied_at"),
               clips_removed=not have, artifact_url=meta.get("artifact_url"))
    out.pop("job_json", None)                        # пути ноутбука в облако не нужны
    cd = pd / "cloud"
    cd.mkdir(exist_ok=True)
    _write(cd / "manifest.json", out)
    stem = Path(man.get("audio") or "запись").stem
    title = re.sub(r"\s+", " ", re.sub(r"^_+", "", stem).replace("rus-transcriber", "")).strip(" ._-") or "запись"
    html = (HERE / "cards.html").read_text(encoding="utf-8")
    html = re.sub(r"<title>.*?</title>", f"<title>Разбор: {title}</title>", html, count=1)
    (cd / "index.html").write_text(html, encoding="utf-8")
    meta.update(version=man["version"], bundled=_now())
    _write(pd / "cloud.json", meta)
    # пути — относительно папки пакета (base): полный путь на каждый клип в ответе не нужен
    files = {"index.html": "cloud/index.html", "manifest.json": "cloud/manifest.json"}
    files.update({k: Path(v).relative_to(pd).as_posix() for k, v in have.items()})
    return {"base": str(pd), "files": files, "remove": sorted(set(clips) - set(have)),
            "artifact_url": meta.get("artifact_url"),
            "applied": out["applied"], "answered": len(out["answers"]), "items": len(man["items"])}


def mark_applied(json_path, ids=None):
    """Запоминает, какие ответы применены (подпись ответа на каждое место) и когда."""
    pd = pack_dir(json_path)
    store = _read(pd / "answers.json", {"answers": {}})
    m = _applied_map(store)
    for aid, a in store.get("answers", {}).items():
        if ids is None or aid in ids:
            if a.get("kind") == "revert":
                m.pop(aid, None)
            elif a.get("kind") != "skip":
                m[aid] = _sig(a)
    store["applied_answers"] = m
    store["applied_at"] = _now()
    _write(pd / "answers.json", store)


def drop_audio(json_path):
    """После применения: удалить голосовые ответы и клипы (на ноутбуке остаётся manifest и answers)."""
    pd = pack_dir(json_path)
    n = 0
    for sub in ("clips", "voice"):
        for f in (pd / sub).glob("*"):
            f.unlink()
            n += 1
    return n


# ---------- распознавание голосовых ответов ----------
_ASR = None


def _asr():
    global _ASR
    if _ASR is None:
        import onnx_asr
        import paths  # noqa: F401  (кэш моделей в папке программы)
        _ASR = onnx_asr.load_model("gigaam-v3-e2e-rnnt", providers=["CPUExecutionProvider"])
    return _ASR


def to_wav16(src, dst):
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)],
                   check=True)


def recognize_here(paths_):
    import soundfile as sf
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for p in paths_:
            try:
                w = Path(tmp) / "a.wav"
                to_wav16(p, w)
                audio, _ = sf.read(w, dtype="float32")
                r = _asr().recognize(audio, sample_rate=SR)
                out[str(p)] = (r if isinstance(r, str) else getattr(r, "text", str(r))).strip()
            except Exception as ex:
                out[str(p)] = ""
                print(f"[recognize] {p}: {ex}", file=sys.stderr, flush=True)
    return out


def recognize_files(paths_):
    """Распознаёт в отдельном процессе (модель не остаётся в памяти сервера)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "recognize", *map(str, paths_)]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(HERE),
                       creationflags=0x08000000 if os.name == "nt" else 0, timeout=600)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        raise RuntimeError(f"распознавание ответов не удалось: {r.stderr[-500:]}")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    if len(sys.argv) > 2 and sys.argv[1] == "recognize":
        print(json.dumps(recognize_here(sys.argv[2:]), ensure_ascii=False))
