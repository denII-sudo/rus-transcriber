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
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

PAD_BEFORE = 3.5     # клип начинается за 3,5 с до места (было 1,5)
PAD_AFTER = 1.5
SR = 16000
HERE = Path(__file__).resolve().parent
MAX_INTERACTION_EVENTS = 2000

# Исход каждого сохранённого ответа после «Применить» — он же состояние, которое видит человек на странице.
# Один словарь для отчёта применения, журнала и страницы карточек.
OUTCOME_REASONS = {
    "saved": "ответ сохранён, ещё не применён",
    "revert_pending": ("отмена подготовлена: нажмите «Применить сохранённые ответы»; до этого прежний ответ "
                       "на месте. Передумали — «Не отменять»"),
    "applied": "ответ применён к стенограмме",
    "reverted": "ответ отменён, место снова открыто",
    "pending_new": ("новый ответ сохранён, но не применён: применённый ответ этим не заменяется. "
                    "Чтобы заменить: «Подготовить отмену» → «Применить сохранённые ответы» → выберите ответ "
                    "заново → «Применить сохранённые ответы»"),
    "kept_previous": "применённый ответ оставлен: пропуск и отрицание его не меняют",
    "skipped": "пропущено, стенограмма не менялась",
    "negative": "отрицание: правка не применена, место осталось открытым",
    "left_for_human": "требует проверки человеком",
    "already_decided": "место уже решено; «Оставить человеку» ничего не изменило",
    "apply_conflict": "не применён: слова карточки изменились после её создания или конфликтуют с прежней правкой",
    "blocked_preflight": "не применён: ответ затрагивает те же слова, что и ответ другой карточки",
    "undo_conflict": "отмена не выполнена: место изменилось после применения или нет снимка для отмены",
    "source_conflict": "не применён: стенограмма изменилась после создания карточек или пакет от другой записи",
    "not_found": "не применён: место карточки не найдено в стенограмме",
    "not_applied": "не применён: автоматическое изменение не получилось",
}
# Уточнение к применённому ответу, у которого нет снимка для отмены (например, изменение слишком большое).
UNDO_UNAVAILABLE = "отменить его будет нельзя: нет снимка для отмены"
# ответ совпал со словами паузы, но вложенное «как есть» человека отказало (например, нужно прослушивание)
PAUSE_CONFIRM_REFUSAL = ("слова паузы совпали с ответом, но место не закрылось: {why}; "
                         "прослушайте паузу и подтвердите её отдельно")
# Исходы, после которых в журнал не пишется попытка применения: успех пишется своим событием,
# «пропущено» и «сохранено» ничего не пробовали менять.
_QUIET_OUTCOMES = {"applied", "reverted", "skipped", "saved"}


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
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=p.parent,
                                     prefix=p.name + ".", suffix=".tmp", delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        tmp = Path(f.name)
    try:
        for k in range(20):
            try:
                os.replace(tmp, p)
                return
            except PermissionError:
                time.sleep(0.05 * (k + 1))
        raise RuntimeError(f"не удалось записать {p}")
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def _answers_lock(json_path):
    """Сериализует read-modify-write ответов между страницей и отдельными MCP-процессами."""
    import jobslots
    lock_path = pack_dir(json_path) / "answers.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as f:
        while not jobslots._try_lock(f):
            time.sleep(0.05)
        yield


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
    """items: [{id, question, options: [{label, decision}], recommended, answer_mode?}] — вопросы Claude
    по местам review. answer_mode="speaker" — карточка подтверждения говорящего: свой ответ на ней
    применяется только как решение о говорящем (имя или «Имя: точные слова»), иначе — человеку.
    -> отчёт. Пересборка меняет version: ответы на прежние вопросы не применятся."""
    import transcript as T
    data = T.load(json_path)
    import recordings
    sqlite_mode = recordings.read_review_pack_state(json_path) is not None
    src = Path(data.get("file") or "")
    if not src.exists():
        raise RuntimeError(f"исходная запись не найдена: {src}")
    review = {it["id"]: it for it in data.get("review") or []}
    rows = {it["id"]: _context(T, data, it) for it in data.get("review") or []}
    pd = pack_dir(json_path)
    clips_dir = pd / "clips"
    if sqlite_mode:
        clips_dir = clips_dir / uuid.uuid4().hex
    clips_dir.mkdir(parents=True, exist_ok=True)
    out, skipped = [], []
    for q in items:
        it = review.get(q.get("id"))
        if not it:
            skipped.append({"id": q.get("id"), "why": "нет такого места"})
            continue
        opts = q.get("options") or []
        if it.get("asr_problem"):
            opts = [{"label": "Я прослушал: слова в стенограмме верны", "decision": {"listened": "words"}},
                    {"label": "Я прослушал: здесь тишина", "decision": {"listened": "silence"}}]
        else:
            opts = [{**o, "decision": {k: v for k, v in (o.get("decision") or {}).items() if k != "listened"}}
                    for o in opts]
        if not 1 <= len(opts) <= 4:
            skipped.append({"id": it["id"], "why": "нужно 1–4 варианта"})
            continue
        t1, t2 = float(it.get("t1", it["t"])), float(it.get("t2", it["t"]))
        clip = clips_dir / (f"{len(out)}.mp3" if sqlite_mode else f"{it['id']}.mp3")
        s0 = cut_clip(src, t1, t2, clip)
        row = rows.get(it["id"], {})
        loc = T._find_span(data, it)
        target_words = data["segments"][loc[0]]["words"][loc[1]:loc[2]] if loc else []
        ids = [w.get("word_id") for w in target_words]
        # Пропуск ссылается на слова временем паузы: ID соседнего слова в карточку не попадают (PR 6а).
        target = {"word_ids": ids} if ids and all(ids) and len(set(ids)) == len(ids) \
            and not T.is_gap_place(it) else {}
        question = q.get("question") or "Как правильно?"
        if it.get("asr_problem"):
            problem = it["asr_problem"]
            models = ((data.get("quality") or {}).get("asr_run") or {}).get("models") or {}
            names = [(models.get(key) or {}).get("id") or key for key in problem.get("words_from") or []]
            model = (models.get(problem.get("model")) or {}).get("id") or problem.get("model") or "ASR"
            question = (str(model) + ": " + str(problem.get("state")) +
                        ("; слова других проходов: " + ", ".join(names) if names else "") +
                        ". Прослушайте весь интервал и подтвердите только то, что вы действительно услышали.")
        manifest_item = {
            "id": it["id"], "t": T.hms(it["t"]), "t1": round(t1, 2), "t2": round(t2, 2),
            **target,
            "clip": clip.relative_to(pd).as_posix(), "clip_t1": round(t1 - s0, 2), "clip_t2": round(t2 - s0, 2),
            "speaker": it.get("speaker"), "text": it.get("text"),
            "reply": row.get("reply"), "before": row.get("before"), "after": row.get("after"),
            "reasons": it.get("reasons"), "voices": it.get("voices"),
            "question": question,
            "options": [{"label": str(o.get("label", "")), "decision": o.get("decision") or {}} for o in opts],
            "recommended": None if it.get("asr_problem") else q.get("recommended"), "note": q.get("note")}
        if q.get("answer_mode") == "speaker" and not it.get("asr_problem"):
            # подтверждение говорящего: свободный ответ не становится текстом реплики (freeform.card_edits)
            manifest_item["answer_mode"] = "speaker"
        automation_diagnostic = _automation_diagnostic(data, it)
        if automation_diagnostic:
            manifest_item["diagnostic_id"] = automation_diagnostic["id"]
            manifest_item["automation_diagnostic"] = automation_diagnostic
        out.append(manifest_item)
    body = json.dumps(out, ensure_ascii=False, sort_keys=True)
    version = time.strftime("%Y%m%d%H%M%S-") + hashlib.sha1(body.encode()).hexdigest()[:6]
    if not sqlite_mode:
        # прежние вопросы — часть истории ответов: manifest сохраняется рядом с архивом ответов,
        # а фрагменты, на которые ссылаются сохранённые манифесты, не удаляются
        old = _read(pd / "manifest.json", None)
        if isinstance(old, dict) and old.get("version") and old.get("version") != version:
            archived = pd / f"manifest.{old['version']}.json"
            if not archived.exists():
                _write(archived, old)
        keep = {f"{r['id']}.mp3" for r in out}
        for kept in pd.glob("manifest.*.json"):
            for row in (_read(kept, {}) or {}).get("items") or []:
                if isinstance(row, dict) and row.get("clip"):
                    keep.add(Path(row["clip"]).name)
        for f in clips_dir.glob("*.mp3"):
            if f.name not in keep:
                f.unlink()
    man = {"version": version, "job_json": str(json_path), "audio": src.name,
           "recording_id": data.get("recording_id"), "source_revision": data.get("source_revision"),
           "speakers": data.get("speakers"), "created": _now(), "items": out}
    ans = _read_answers(json_path, {})
    if sqlite_mode and ans.get("_expected_versions") != {
            "source_revision": data.get("source_revision"),
            "review_revision": data.get("review_revision")}:
        import storage
        raise storage.Conflict("recording changed while building review pack")
    if ans.get("version") != version:          # прежние ответы — к прежним вопросам; храним для истории
        if (not sqlite_mode and
                (ans.get("answers") or ans.get("interaction_events") or ans.get("application_events"))):
            _write(pd / f"answers.{ans.get('version', 'old')}.json", ans)
        expected_versions = ans.get("_expected_versions")
        ans = {"version": version, "answers": {}, "conflicts": [], "rejected": []}
        if expected_versions is not None:
            ans["_expected_versions"] = expected_versions
    ans["version"] = version
    ans["manifest"] = man
    _write_answers(json_path, ans)
    if not sqlite_mode:
        _write(pd / "manifest.json", man)
    shutil.copyfile(HERE / "cards.html", pd / "cards.html")
    return {"pack": str(pd), "version": version, "items": len(out), "skipped": skipped,
            "clips_bytes": sum(f.stat().st_size for f in clips_dir.glob("*.mp3"))}


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


def _automation_diagnostic(data, item):
    """Снимок основания автоматики, связанный с конкретным местом разбора."""
    diagnostic_id = item.get("diagnostic_id")
    if not diagnostic_id:
        return None
    quality = data.get("quality") or {}
    trace = quality.get("missed_speech_diagnostics") or {}
    candidate = next((row for row in trace.get("candidates") or []
                      if row.get("id") == diagnostic_id), None)
    if not candidate:
        return None
    turn = next((row for row in trace.get("speech_intervals") or []
                 if row.get("id") == candidate.get("turn_id")), None)
    return {"id": diagnostic_id, "kind": "missed_speech_gap",
            "algorithm": trace.get("algorithm"),
            "threshold_seconds": trace.get("threshold_seconds"),
            "candidate": dict(candidate),
            "speech_interval": dict(turn) if turn else None}


def manifest(json_path):
    import recordings
    state = recordings.read_review_pack_state(json_path)
    if state is not None:
        return state.get("manifest")
    return _read(pack_dir(json_path) / "manifest.json", None)


def _read_answers(json_path, default):
    import recordings
    state = recordings.read_review_pack_state(json_path)
    if state is not None:
        if state.get("version") is None:
            return {**default, "_expected_versions": state["_expected_versions"]}
        return state
    return _read(pack_dir(json_path) / "answers.json", default)


def answers_state(json_path, default=None):
    return _read_answers(json_path, {"answers": {}} if default is None else default)


def _write_answers(json_path, state):
    import recordings
    result = recordings.write_review_pack_state(json_path, state)
    if result is not None:
        return result
    return _write(pack_dir(json_path) / "answers.json", state)


def add_interaction_event(json_path, aid, version, event_type, session):
    """Сохраняет локальный факт показа карточки или действия с её клипом."""
    allowed = {"card_visible", "card_rendered", "audio_play", "audio_ended"}
    if event_type not in allowed:
        raise ValueError("неизвестное событие карточки")
    if not isinstance(session, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", session):
        raise ValueError("неверный идентификатор сессии страницы")
    pd = pack_dir(json_path)
    man = manifest(json_path)
    if not man or version != man.get("version"):
        raise ValueError("версия пакета изменилась")
    item = next((it for it in man.get("items") or [] if it.get("id") == aid), None)
    if not item:
        raise ValueError("карточка не найдена")
    with _answers_lock(json_path):
        store = _read_answers(json_path, {"version": man["version"], "answers": {}})
        events = store.setdefault("interaction_events", [])
        if event_type == "card_visible" and any(
                row.get("id") == aid and row.get("version") == version and
                row.get("type") == event_type and row.get("session") == session for row in events):
            return {"stored": False, "duplicate": True}
        event = {"id": aid, "version": version, "diagnostic_id": item.get("diagnostic_id"),
                 "type": event_type, "session": session, "source": "local", "ts": _now()}
        events.append(event)
        if len(events) > MAX_INTERACTION_EVENTS:
            removed = len(events) - MAX_INTERACTION_EVENTS
            del events[:removed]
            store["interaction_events_dropped"] = store.get("interaction_events_dropped", 0) + removed
        _write_answers(json_path, store)
    return {"stored": True, "event": event}


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
    import freeform
    if freeform.negative_answer(text):     # закрытый список отрицаний: ничего не подтверждает
        return {"kind": "free"}
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
        if re.search(r"рекоменд|как предлож|согла|(?<!на)верно|правильно|^да$|^ага$|^угу$", j) and \
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


def add_answers(json_path, answers, source, *, trusted_actor="unknown"):
    import recordings
    if trusted_actor not in ("human", "unknown"):
        raise ValueError("invalid answer actor")
    with recordings.operation_for_document(json_path, trusted_actor):
        with _answers_lock(json_path):
            return _add_answers(json_path, answers, source, trusted_actor)


def _answer_time(answer):
    """Время ответа для правила «более поздний побеждает». Исключение: у ответа, возвращённого по «Не отменять»,
    время прежнее, а решение принято в момент запроса — берётся более позднее из двух."""
    return max(answer.get("ts", ""), (answer.get("keep_request") or {}).get("ts", ""))


def _newer(old, ans):
    """Более поздний ответ побеждает: сохранённый ответ old новее пришедшего ans — ans не сохраняется."""
    return bool(old) and _answer_time(old) > ans["ts"]


def _reject_keep(store, ans, why):
    """Отклонённое «Не отменять» — одна короткая запись на место (последняя), без текста ответа."""
    rows = store.setdefault("rejected", [])
    rows[:] = [r for r in rows if not (r.get("id") == ans["id"] and r.get("kind") == "keep")]
    rows.append({"id": ans["id"], "kind": "keep", "ts": ans["ts"], "source": ans["source"], "why": why})


def _add_answers(json_path, answers, source, trusted_actor="unknown"):
    """Сводит ответы в answers.json. answers: [{id, version, kind|option|text|audio, ts?}].
    Голосовые ответы без текста распознаются GigaAM (отдельный процесс) и сопоставляются с вариантами."""
    import recordings
    context = recordings.context_for_document(json_path)
    pd = pack_dir(json_path)
    man = manifest(json_path)
    if not man:
        raise RuntimeError("пакет разбора не собран: сначала build_review_pack")
    by_id = {it["id"]: it for it in man["items"]}
    store = _read_answers(json_path, {"version": man["version"], "answers": {}, "conflicts": [], "rejected": []})
    need_asr = [a for a in answers if a.get("audio") and not a.get("text")]
    if need_asr:
        heard = recognize_files([a["audio"] for a in need_asr])
        for a in need_asr:
            a["text"] = heard.get(str(a["audio"]), "")
            a["heard"] = True
    rep = {"accepted": [], "rejected": [], "conflicts": [], "free": []}
    applied = _applied_map(store)
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
        if context is not None:
            ans.update(actor_kind=context["actor_kind"], author_ref=context["author_ref"],
                       initiated_by=context["initiated_by"])
        else:
            ans["actor_kind"] = trusted_actor
        if it.get("diagnostic_id"):
            ans["diagnostic_id"] = it["diagnostic_id"]
        if isinstance(a.get("text"), str):          # пометки сомнения ⟦…⟧ из скопированной стенограммы
            a["text"] = a["text"].replace("⟦", "").replace("⟧", "")
        for k in ("text", "audio", "heard", "by"):
            if a.get(k) is not None:
                ans[k] = a[k]
        if a.get("kind") == "keep":
            # «Не отменять»: вместо подготовленной отмены (или ждущего ответа) снова сохраняется применённый
            # ответ — целиком, из копии, которую хранит заменивший его ответ (время, источник, услышанное).
            previous = applied.get(aid)
            why = ("нет применённого ответа, к которому можно вернуться" if previous is None else
                   "есть более поздний ответ" if _newer(store["answers"].get(aid), ans) else None)
            if why:
                rep["rejected"].append({"id": aid, "why": why})
                _reject_keep(store, ans, why)
                continue
            kept = (store["answers"].get(aid) or {}).get("applied_answer")
            if kept and _sig(kept) == previous:
                # keep_request — по какому запросу ответ возвращён: страница сверяет с ним свой запрос.
                store["answers"][aid] = dict(copy.deepcopy(kept), keep_request={"ts": ans["ts"], "source": source})
                rep["accepted"].append({"id": aid, "kind": kept.get("kind"), "answer": _desc(kept, it),
                                        "restored": True})
                continue
            # Исключение для пакетов, записанных до появления копии: ответ собирается по подписи,
            # с отметкой restored и временем возврата (прежнее время неизвестно).
            kind, option, text = (list(previous) + [None, None, None])[:3]
            ans.pop("text", None)
            ans.update(kind=kind, restored=True, keep_request={"ts": ans["ts"], "source": source})
            if option is not None:
                ans["option"] = option
            if text is not None:
                ans["text"] = text
        elif a.get("kind") in ("human", "skip", "recommended", "free", "repeat", "revert"):
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
        if ans["kind"] in ("repeat", "skip") and over_applied(ans, applied.get(aid)) == "kept_previous":
            # Пропуск не заменяет применённый ответ: иначе «Применить» сняло бы прежнюю правку.
            rep["accepted"].append({"id": aid, "kind": ans["kind"], "stored": False,
                                    "note": "пропуск не меняет применённый ответ — он остаётся"})
            continue
        old = store["answers"].get(aid)
        if old and old.get("source") != source and not _same(old, ans):
            c = {"id": aid, "was": old, "now": ans}
            store.setdefault("conflicts", []).append(c)
            rep["conflicts"].append({"id": aid, "was": _desc(old, it), "now": _desc(ans, it)})
        if _newer(old, ans):
            continue
        if old and over_applied(ans, applied.get(aid)) not in (None, "applied"):
            # Ответ поверх применённого хранит копию применённого ответа целиком: по «Не отменять» он
            # возвращается как был. Копия переносится и через цепочку ждущих ответов.
            kept = old if _sig(old) == applied.get(aid) else old.get("applied_answer")
            if kept and _sig(kept) == applied.get(aid):
                ans["applied_answer"] = {k: copy.deepcopy(v) for k, v in kept.items()
                                         if k not in ("applied_answer", "keep_request")}
        store["answers"][aid] = ans
        rep["accepted"].append({"id": aid, "kind": ans["kind"], "answer": _desc(ans, it)})
        if ans["kind"] == "free":
            rep["free"].append({"id": aid, "text": ans.get("text")})
    _write_answers(json_path, store)
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


def over_applied(answer, applied_sig):
    """Правило замены применённого ответа (одно на сохранение, применение и страницу). -> None, если
    применённого ответа на месте нет; «applied» — это он и есть; «revert» — отменить его; «kept_previous» —
    пропуск или отрицание его не меняют; «pending_new» — новый ответ ждёт, пока прежний не отменят.
    Исключение до полной замены ответа: другой ответ (кнопка, свой текст, «Оставить человеку») применённый
    не заменяет — только явная отмена."""
    import freeform
    if applied_sig is None:
        return None
    kind = answer.get("kind")
    if applied_sig == _sig(answer):
        return "applied"
    if kind == "revert":
        return "revert"
    if kind in ("skip", "repeat") or (kind == "free" and freeform.negative_answer(answer.get("text"))):
        return "kept_previous"
    return "pending_new"


def _applied_map(store):
    """{id: подпись применённого ответа}. Для пакетов до появления этой записи — ответы не позже applied_at."""
    m = dict(store.get("applied_answers") or {})
    if "applied_answers" not in store and store.get("applied_at"):
        for aid, a in store.get("answers", {}).items():
            if (a.get("ts") or "") <= store["applied_at"]:
                m[aid] = _sig(a)
    return m


def decisions(json_path):
    """Ответы пакета -> решения для apply_review. -> (decisions, отчёт).
    Уже применённый и не изменившийся ответ повторно не применяется. Применённый ответ другим ответом не
    заменяется: новый ответ сохраняется и ждёт (pending_new), пока человек не отменит прежний; пропуск и
    отрицание оставляют прежний (kept_previous). Только ответ «отменить» откатывает правку по структурному
    снимку и снова открывает место; без снимка отмена блокируется конфликтом."""
    import freeform
    import transcript as T
    pd = pack_dir(json_path)
    man = manifest(json_path)
    store = _read_answers(json_path, {"answers": {}})
    data = T.load(json_path)
    applied = _applied_map(store)
    by_id = {it["id"]: it for it in man["items"]}
    conflicted = {c["id"] for c in store.get("conflicts") or []}
    out, rep = [], {"applied_from_answers": 0, "human": [], "free": [], "skipped": [], "unanswered": [],
                    "unchanged": [], "changed": [], "reverted": [], "kept_previous": [], "pending_new": [],
                    "source_conflict_ids": []}
    if man.get("recording_id") and man["recording_id"] != data.get("recording_id"):
        rep["source_conflict"] = "пакет относится к другой стенограмме"
        rep["source_conflict_ids"] = [aid for aid in by_id if store["answers"].get(aid)]
        return out, rep
    for aid, it in by_id.items():
        a = store["answers"].get(aid)
        if not a:
            rep["unanswered"].append(aid)
            continue
        relation = over_applied(a, applied.get(aid))
        if relation == "applied":
            rep["unchanged"].append(aid)
            continue
        src = f"разбор ({a['source']}, {a['ts']})"
        k = a["kind"]
        if k == "free" and freeform.negative_answer(a.get("text")):
            # отрицание не отменяет и ранее применённый ответ; подсказку даёт разбор свободного ответа
            rep["free"].append({"id": aid, "text": a.get("text")})
            if relation == "kept_previous":
                rep["kept_previous"].append(aid)
            continue
        if relation in ("kept_previous", "pending_new"):
            # Старый пакет мог сохранить пропуск поверх применённого ответа; другой ответ ждёт отмены.
            rep[relation].append(aid)
            continue
        if relation == "revert":
            snapshot = (store.get("undo_snapshots") or {}).get(aid)
            if not snapshot:
                rep.setdefault("undo_conflicts", []).append({"id": aid, "why":
                    "для безопасной отмены нет проверенного структурного снимка"})
                continue
            # Прежний ответ снимается структурно, а не по тексту отображения.
            out.append({"id": aid, "_undo_snapshot": snapshot})
            continue
        if k in ("option", "recommended"):
            d = dict(it["options"][a["option"]]["decision"] or {})
            if d.get("listened") and man.get("source_revision") != data.get("source_revision"):
                rep.setdefault("source_conflict", "стенограмма изменилась после создания карточек; пересоберите пакет")
                rep["source_conflict_ids"].append(aid)
                continue
            if "word_ids" in it and not T.is_gap_place(it):     # у пропуска старый пакет нёс ID соседа
                d["word_ids"] = it["word_ids"]
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
            # Только статус места: «Оставить человеку» ничего не подтверждает (слова без отметок человека).
            # Решённое место не возвращается человеку; кроме карточек подтверждения говорящего — они
            # строятся по уже решённым местам.
            d = {"status": "human", "note": f"{src}: оставлено человеку"
                 + (f" («{a['text']}»)" if a.get("text") else "")}
            if a.get("text"):
                d["comment"] = a["text"]            # комментарий человека — и к месту, уже оставленному ему
            if it.get("answer_mode") != "speaker":
                d["if_open"] = True
            rep["human"].append(aid)
        elif k == "free":
            # свободный ответ — правка услышанного в клипе: применяется по словам (apply_free, freeform.py)
            rep["free"].append({"id": aid, "text": a.get("text")})
            continue
        elif k == "revert":
            d = {"reopen": True, "note": f"{src}: ответ отменён, место снова открыто"}
        else:
            rep["skipped"].append(aid)
            continue
        if k == "revert":                          # отменять нечего: ответ не был применён
            rep["skipped"].append(aid)
            continue
        out.append({"id": aid, **d})
    return out, rep


def free_answers(json_path):
    """Свободные ответы пакета, ещё не применённые: {id: ответ}. Отрицание — и после применения (место
    остаётся с подсказкой, применённая правка не трогается)."""
    import freeform
    pd = pack_dir(json_path)
    store = _read_answers(json_path, {"answers": {}})
    applied = _applied_map(store)
    return {aid: a for aid, a in store.get("answers", {}).items() if a.get("kind") == "free" and
            (aid not in applied or freeform.negative_answer(a.get("text")))}


def apply_free(json_path, md_path, free, dry_run=False):
    """Свободные ответы → правки слов (freeform.plan) → transcript.apply_word_edits. -> отчёт для человека."""
    import conspect
    with conspect.write_lock(json_path):
        return _apply_free_locked(json_path, md_path, free, dry_run=dry_run)


def _apply_free_locked(json_path, md_path, free, dry_run=False):
    from collections import Counter
    import freeform as F
    import transcript as T
    man = manifest(json_path)
    data = T.load(json_path)
    missing, valid_free, missing_reasons = [], dict(free), {}
    if man.get("recording_id") and man["recording_id"] != data.get("recording_id"):
        missing.extend(valid_free)
        valid_free.clear()
    times = Counter(round(w["s"], 2) for s in data["segments"] for w in s.get("words") or [])
    for item in man["items"]:
        aid = item["id"]
        if aid not in valid_free:
            continue
        if "word_ids" in item and not T.is_gap_place(item):     # пропуск: слова — по времени паузы
            ids = item["word_ids"]
            all_words = [w for s in data["segments"] for w in s.get("words") or []]
            all_ids = Counter(w.get("word_id") for w in all_words)
            loc = T._find_span(data, item)
            bad = not loc or any(all_ids[wid] != 1 for wid in ids or [])
            if not bad:
                selected = [w for w in all_words if w.get("word_id") in ids]
                bad = any(not item["t1"] - 0.01 <= w["s"] <= item["t2"] + 0.01 for w in selected)
            if not bad:
                # Совпадающие времена вне выбранного места могут стать неоднозначными опорами выравнивания.
                window = F.window(all_words, item["t1"], item["t2"])
                local_times = Counter(round(w["s"], 2) for w in window)
                target_times = Counter(round(w["s"], 2) for w in selected)
                bad = any(count > 1 and target_times[t] != 1 for t, count in local_times.items())
            reason = "слова карточки отсутствуют, изменили время или их ID неоднозначны"
        else:
            words = [w for s in data["segments"] for w in s.get("words") or []]
            local = F.window(words, item["t1"], item["t2"])
            bad = any(times[round(w["s"], 2)] > 1 for w in local)
            reason = "старый пакет без ID: слова в окне имеют одинаковое время"
        if bad:
            missing.append(aid)
            valid_free.pop(aid)
            missing_reasons[aid] = reason
    vocab_words = []
    try:
        import vocab as V
        vp = (data.get("quality") or {}).get("vocab")
        vd = json.loads(Path(vp).read_text(encoding="utf-8-sig")) if vp and Path(vp).exists() else {}
        vocab_words = list(vd.get("terms") or []) + [p.get("name") for p in vd.get("participants") or [] if p.get("name")]
    except Exception:
        pass
    edits, reports, closed, conflicts = F.plan(man, valid_free, data, vocab_words)
    targets = {}
    for edit in edits:
        ids = list(edit.get("word_ids") or [])
        ids.extend(edit[k] for k in ("word_id", "after_id", "before_id") if edit.get(k))
        targets.setdefault(edit.get("card"), set()).update(ids)
    conflict_ids = {cid for conflict in conflicts for cid in conflict.get("cards", [])}
    blocked = conflict_ids | {r["id"] for r in reports if r.get("human") or r.get("proposed_delete")}
    edits = [e for e in edits if e.get("card") not in blocked]
    closed = {rid: via for rid, via in closed.items() if via not in blocked}
    if edits or closed:
        result_data, rep = T.apply_word_edits(json_path, md_path, edits, by="human", closed=closed,
                                    answers={k: v.get("text") for k, v in valid_free.items()}, dry_run=dry_run)
    else:
        result_data = data
        rep = {"cards": {}, "failed": {}, "proposed_delete": [], "not_found": [],
               "relabel": 0, "replace": 0, "insert": 0}
    items = {r["id"]: r for r in data.get("review") or []}
    confirmed, confirm_refusals = [], {}
    if not dry_run:
        # ответ совпал со словами, уже стоящими в паузе: открытое место закрывается тем же путём, что «как есть»
        still_open = {r["id"] for r in T.load(json_path).get("review") or []
                      if r.get("status", "open") in ("open", "human")}
        confirmed = [r["id"] for r in reports if r.get("confirm_pause") and r["id"] not in blocked
                     and r["id"] in still_open]
        if confirmed:
            _, nested = T.apply_review(json_path, md_path, [{"id": cid} for cid in confirmed], by="human")
            refused = {c.get("id"): c.get("why") for c in nested.get("conflicts") or []}
            done = set(nested.get("successful_ids") or [])
            not_closed = [cid for cid in confirmed if cid not in done]
            confirmed = [cid for cid in confirmed if cid in done]
            confirm_refusals = {cid: PAUSE_CONFIRM_REFUSAL.format(why=refused.get(cid) or "место не закрылось")
                                for cid in not_closed}
    human = [{"id": aid, "why": reason, "answer": free[aid].get("text")}
             for aid, reason in missing_reasons.items()]
    human += [{"id": cid, "why": why, "answer": free[cid].get("text")} for cid, why in confirm_refusals.items()]
    for r in reports:
        if r["id"] in conflict_ids:
            human.append({"id": r["id"], "why": "ответы на карточки конфликтуют", "answer": r["answer"]})
            continue
        if r.get("proposed_delete"):
            human.append({"id": r["id"], "why": "удаление слов требует проверки", "answer": r["answer"]})
            continue
        if r.get("human"):
            human.append({"id": r["id"], "why": r["human"], "answer": r["answer"]})
    negative_ids = {r["id"] for r in reports if r.get("kind") == "negative"}
    mark_human = [h for h in human if h["id"] not in negative_ids]
    if mark_human and not dry_run:                # не понятые ответы — человеку; отрицание оставляет место как было
        T.apply_review(json_path, md_path, [{"id": h["id"], "status": "human",
                                             "note": f"{h['why']}: «{h['answer']}»"} for h in mark_human], by="human")
    return {**({"target_ids": {aid: sorted(ids) for aid, ids in targets.items()}} if dry_run else {}),
            "cards": [{"id": r["id"], "answer": r["answer"], "kind": r["kind"],
                       "was_now": rep["cards"].get(r["id"], {}).get("before", "") + " → " +
                       rep["cards"].get(r["id"], {}).get("after", "") if r["id"] in rep["cards"] else None,
                       "applied": r["applied"], "inserted": r["inserted"], "typos": r["typos"]} for r in reports],
            "closed_via": closed, "conflicts": conflicts, "left_to_human": human,
            "successful_ids": [r["id"] for r in reports if (r["id"] in rep["cards"] or r["id"] in confirmed)
                               and r["id"] not in rep["failed"] and r["id"] not in conflict_ids
                               and not r.get("human")
                               and not any(d["card"] == r["id"] for d in rep["proposed_delete"])],
            "proposed_delete": rep["proposed_delete"], "not_found": rep["not_found"] + missing,
            "failed": rep["failed"],
            "counts": {k: rep[k] for k in ("relabel", "replace", "insert")},
            "vocab_suggestions": [v for v in (result_data if dry_run else T.load(json_path)).get("vocab_suggestions") or [] if v.get("card")],
            "conspects": rep.get("conspects")}


def preflight_answers(json_path, md_path, decisions_, free):
    """Найти пересекающиеся цели разных ответов до каких-либо изменений файлов."""
    import transcript as T
    ids = {d.get("id") for d in decisions_ if d.get("id")} | set(free)
    if len(ids) < 2:
        return []
    man = manifest(json_path) or {}
    items = {it["id"]: it for it in man.get("items") or []}
    data = T.load(json_path)
    # Планируем по одной карточке на одной исходной стенограмме: общий F.plan
    # может дедуплицировать правки и скрыть часть заявленных целей.
    planned = {}
    for aid, answer in free.items():
        preview = apply_free(json_path, md_path, {aid: answer}, dry_run=True)
        planned.update(preview.get("target_ids") or {})
    claims = {}
    for aid in ids:
        it = items.get(aid) or {}
        gap_card = T.is_gap_place(it)       # пропуск: цель — слова паузы; word_ids пакета не читаются
        own = set(T.place_words(data, it)[0] if T.place_words(data, it) else []) if gap_card \
            else set(it.get("word_ids") or [])
        target = own | set(planned.get(aid) or [])
        t1, t2 = it.get("t1"), it.get("t2")
        if not target and not gap_card and t1 is not None and t2 is not None:
            # Старые карточки адресуются временем: берём все слова в диапазоне,
            # а не выбранный эвристикой первый сегмент.
            target = {w.get("word_id") for s in data.get("segments") or [] for w in s.get("words") or []
                      if t1 - 0.01 <= w["s"] <= t2 + 0.01 and w.get("word_id")}
        claims[aid] = {"ids": target, "legacy": gap_card or not bool(it.get("word_ids")), "t1": t1, "t2": t2,
                       "gap": gap_card}
    conflicts = []
    ordered = sorted(ids)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            x, y = claims[a], claims[b]
            shared = x["ids"] & y["ids"]
            if x["gap"] or y["gap"]:
                legacy_overlap = T.places_overlap(items.get(a) or {}, items.get(b) or {})
            else:
                legacy_overlap = (x["legacy"] or y["legacy"]) and all(
                    v is not None for v in (x["t1"], x["t2"], y["t1"], y["t2"])) and \
                    x["t1"] <= y["t2"] + 0.01 and y["t1"] <= x["t2"] + 0.01
            if shared or legacy_overlap:
                conflicts.append({"cards": [a, b], "why": "ответы затрагивают общие слова или неоднозначный старый диапазон"})
    return conflicts


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
    store = _read_answers(json_path, {"answers": {}})
    meta = _read(pd / "cloud.json", {})
    if artifact_url:
        meta["artifact_url"] = artifact_url
    clips = {it["clip"]: pd / it["clip"] for it in man["items"]}
    have = {k: v for k, v in clips.items() if v.exists()}
    # Копия применённого ответа (applied_answer) и отметка запроса (keep_request) нужны только ноутбуку:
    # странице хватает answer_states (keep: ts, source, accepted).
    out = dict(man, answers={k: {f: x for f, x in v.items() if f not in ("applied_answer", "keep_request")}
                             for k, v in store.get("answers", {}).items() if v.get("version") == man["version"]},
               applied_answers=store.get("applied_answers") or {}, answer_states=answer_states_of(store, man),
               conflicts=[{"id": c["id"]} for c in store.get("conflicts") or []],
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


def mark_applied(json_path, ids=None, snapshots=None, outcomes=None):
    """Запоминает исходы применения без копии текста ответа.

    outcomes {id: исход}: «applied» отмечает ответ применённым (подпись ставится только после успеха),
    «reverted» снимает отметку и снимок только у ответа «отменить», остальные исходы пишут в журнал попытку
    применения — один раз на ответ: повторный запуск с тем же исходом событий не дописывает.
    Без outcomes (прежний вызов) ответы из ids отмечаются по своему виду."""
    store = _read_answers(json_path, {"answers": {}})
    man = manifest(json_path) or {}
    manifest_items = {it.get("id"): it for it in man.get("items") or []}
    m = _applied_map(store)
    answers = store.get("answers") or {}
    if outcomes is None:
        outcomes = {aid: "reverted" if a.get("kind") == "revert" else "applied"
                    for aid, a in answers.items() if (ids is None or aid in ids) and a.get("kind") != "skip"}
    changes = []
    for aid, outcome in outcomes.items():
        a = answers.get(aid)
        if not a or (ids is not None and outcome in ("applied", "reverted") and aid not in ids):
            continue
        if outcome == "reverted" and a.get("kind") == "revert":
            m.pop(aid, None)
            store.setdefault("undo_snapshots", {}).pop(aid, None)
            changes.append((aid, "reverted"))
        elif outcome == "applied" and a.get("kind") not in ("skip", "repeat", "revert"):
            m[aid] = _sig(a)
            if snapshots and aid in snapshots:
                store.setdefault("undo_snapshots", {})[aid] = snapshots[aid]
            else:
                store.setdefault("undo_snapshots", {}).pop(aid, None)
            changes.append((aid, "applied"))
    store["applied_answers"] = m
    if changes:
        store["applied_at"] = _now()
    events = store.setdefault("application_events", [])
    first_new_event = len(events)

    def event(aid, action, outcome):
        answer = answers.get(aid) or {}
        item = manifest_items.get(aid) or {}
        return {"id": aid, "version": answer.get("version") or man.get("version"),
                "diagnostic_id": answer.get("diagnostic_id") or item.get("diagnostic_id"),
                "answer_source": answer.get("source"), "answer_ts": answer.get("ts"),
                "answer_sig_sha256": _sig_hash(answer), "action": action, "outcome": outcome, "at": _now()}

    for aid, outcome in outcomes.items():
        if outcome in _QUIET_OUTCOMES or aid not in answers:
            continue
        last = _answer_event(events, aid, answers[aid])
        if last is not None and last.get("outcome") == outcome:
            continue                      # тот же исход того же ответа уже в журнале
        events.append(event(aid, "apply_attempt", outcome))
    for aid, outcome in changes:
        events.append(event(aid, outcome, outcome))
    import recordings
    draft = recordings.answer_draft_for(json_path)
    if draft is not None:
        for row in events[first_new_event:]:
            row["context"] = copy.deepcopy(draft["context"])
    if len(events) > MAX_INTERACTION_EVENTS:
        removed = len(events) - MAX_INTERACTION_EVENTS
        del events[:removed]
        store["application_events_dropped"] = store.get("application_events_dropped", 0) + removed
    _write_answers(json_path, store)


def _sig_hash(answer):
    """Хэш подписи ответа для журнала применения: сам ответ (текст своего ответа) в журнал не пишется."""
    raw = json.dumps(_sig(answer), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _answer_event(events, aid, answer):
    """Последнее событие применения именно этого ответа: то же место, источник, время и сам ответ (хэш подписи).
    Исключение: у событий, записанных без хэша подписи, сверяются только место, источник и время."""
    for row in reversed(events or []):
        if (row.get("id") == aid and row.get("answer_ts") == answer.get("ts") and
                row.get("answer_source") == answer.get("source") and
                ("answer_sig_sha256" not in row or row["answer_sig_sha256"] == _sig_hash(answer))):
            return row
    return None


def _answer_state(answer, applied_sig, event):
    kind = answer.get("kind")
    relation = over_applied(answer, applied_sig)
    if relation == "revert":
        # Отмена сохранена, но ещё не применялась: правка на месте, пока человек не нажмёт «Применить».
        return "undo_conflict" if (event or {}).get("outcome") == "undo_conflict" else "revert_pending"
    if relation is not None:
        return relation
    if event is not None and event.get("outcome") in OUTCOME_REASONS and event.get("outcome") != "applied":
        return event["outcome"]
    if kind in ("skip", "repeat", "revert"):
        return "skipped"
    return "saved"


def answer_states_of(store, man):
    """{id: {state, why, card, ts, source}} по сохранённому состоянию пакета — то, что показывает страница."""
    applied = _applied_map(store)
    events = store.get("application_events") or []
    snapshots = store.get("undo_snapshots") or {}
    cards = {it.get("id"): n for n, it in enumerate((man or {}).get("items") or [], 1)}
    rejected_keeps = {r.get("id"): r for r in store.get("rejected") or [] if r.get("kind") == "keep"}
    out = {}
    for aid, answer in (store.get("answers") or {}).items():
        state = _answer_state(answer, applied.get(aid), _answer_event(events, aid, answer))
        out[aid] = {"state": state, "why": OUTCOME_REASONS[state], "card": cards.get(aid),
                    "ts": answer.get("ts"), "source": answer.get("source")}
        details = []
        if aid in applied and aid not in snapshots and state in ("applied", "pending_new", "kept_previous"):
            details.append(UNDO_UNAVAILABLE)
        # Последний обработанный запрос «Не отменять»: по нему страница узнаёт, что её запрос принят или отклонён.
        accepted, rejected = answer.get("keep_request"), rejected_keeps.get(aid)
        if rejected and (not accepted or rejected["ts"] > accepted.get("ts", "")):
            out[aid]["keep"] = {"ts": rejected["ts"], "source": rejected["source"], "accepted": False}
            if rejected["ts"] > _answer_time(answer):       # причина — только под ответом, который старше запроса
                details.append("«Не отменять» не принят: " + rejected["why"])
        elif accepted:
            out[aid]["keep"] = {"ts": accepted.get("ts"), "source": accepted.get("source"), "accepted": True}
        if details:
            out[aid]["detail"] = " — ".join(details)
    return out


def answer_states(json_path):
    return answer_states_of(_read_answers(json_path, {"answers": {}}), manifest(json_path) or {})


def drop_audio(json_path):
    """После применения: удалить голосовые ответы и клипы (на ноутбуке остаётся manifest и answers)."""
    pd = pack_dir(json_path)
    n = 0
    for sub in ("clips", "voice"):
        root = pd / sub
        if not root.exists():
            continue
        for f in root.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
            n += 1
        try:
            root.rmdir()
        except OSError:
            pass
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
