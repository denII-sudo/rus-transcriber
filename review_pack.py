"""Пакет разбора: короткие клипы сомнительных мест + manifest.json с вопросами и вариантами,
и answers.json — ответы из локального и облачного режимов.

Папка пакета: <подпапка результатов>/<имя записи>[.tag].review/
    manifest.json   {"version", "pack_rev", "job_id", "audio", "speakers", "clips_dir", "items": [...]}
    clips/r5.mp3    клип места: 3,5 с до и 1,5 с после (mp3, моно, 64 кбит/с — играет везде, и на iPhone);
                    заменённая или дописанная карточка — clips/r5.r2.mp3 (upsert_cards)
    answers.json    {"version", "answers": {id: ответ}, "conflicts": [...], "rejected": [...]}
    voice/          голосовые ответы (локальная страница и перенесённые из облака)
    cards.html      страница карточек (та же, что публикуется в облако)

Элемент manifest:
    id, rev (редакция карточки, с 1), t (чч:мм:сс), t1/t2 (время места в записи), clip (null — без звука),
    clip_t1/clip_t2 (место внутри клипа), history (прежние редакции с их ответами),
    speaker, text, reply (реплика с ⟦…⟧), before/after, reasons, voices,
    question, options [{label, decision: {speaker | text | status, note}}], recommended (номер варианта).
Ответ:
    {id, version, rev (у карточки с rev > 1), source: local|cloud, ts, kind: option|recommended|human|skip|free|repeat,
     option (номер), text (распознанный или набранный), audio (путь к голосовому ответу)}
Ответ на одно место из двух режимов: побеждает более поздний, расхождение — в conflicts (для человека).
Ответ с другой version (пакет пересобран) не принимается — в rejected. version — имя пакета: upsert_cards его не
меняет, а растит pack_rev; ответ на прежнюю редакцию карточки (rev) не принимается — «вопрос обновлён».

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
    "applied_open": ("ответ применён, но место осталось открытым: прослушайте интервал и подтвердите "
                     "услышанное кнопкой «Я прослушал…»"),
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
    "pause_refused": ("не применён: паузу так не закрыть — нажмите кнопку прослушивания паузы "
                      "или впишите услышанное"),
    "hint": "не применён: короткий ответ не понят как слова, место осталось открытым — подсказка ниже",
}
# Уточнение к применённому ответу, у которого нет снимка для отмены (например, изменение слишком большое).
UNDO_UNAVAILABLE = "отменить его будет нельзя: нет снимка для отмены"
# ответ совпал со словами места (паузы) или подтвердил их, но вложенное «как есть» человека отказало
# (например, нужно прослушивание)
PAUSE_CONFIRM_REFUSAL = ("слова места совпали с ответом, но место не закрылось: {why}; "
                         "прослушайте паузу или место и подтвердите отдельно")
# годность кнопки прослушивания паузы — по словам её интервала, а не по ревизии всей стенограммы
LISTEN_STALE = "слова в интервале паузы изменились после создания карточек; пересоберите пакет"
CONTRADICTION_STALE = ("слова места изменились после показа карточки противоречия; ответ не применён, "
                       "карточка обновлена — выберите заново")
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
# Ответ на редакцию карточки, которая уже заменена (upsert_cards): страница показывает новую, текст остаётся в поле.
REV_STALE = "вопрос обновлён"
REV_UNKNOWN = "нет такой редакции вопроса"
# Поля карточки, по которым её замена считается тем же вопросом (повтор ничего не пишет). Контекст соседних
# реплик (reply/before/after) сюда не входит: правка соседнего места не делает вопрос другим.
_CARD_FP_FIELDS = ("question", "options", "recommended", "answer_mode", "word_ids", "listen_fp", "t1", "t2",
                   "speaker", "text")
# Копии для «Не отменять» нужны только живому ответу, не ответу в истории карточки.
_ANSWER_PRIVATE = ("applied_answer", "keep_request")
# В облако не уходит и путь к голосовому файлу на этом компьютере (в нём имя пользователя Windows):
# облачный голос идёт через audio_id, страница поле audio не читает.
# human_note — примечание человека на карточке противоречия (PR 13б): остаётся на этом компьютере
_ANSWER_LOCAL = _ANSWER_PRIVATE + ("audio", "human_note")


def card_rev(item):
    """Номер редакции карточки; у карточек до PR 13а его нет — первая редакция."""
    return (item or {}).get("rev", 1)


def rev_refusal(rev, item):
    """Почему ответ (событие, голос, предпросмотр) с номером редакции rev не относится к карточке item.
    Без номера (старая страница, облако до обновления, import_answers) — только к неизменённой карточке."""
    current = card_rev(item)
    if rev is None:
        return None if current == 1 else REV_STALE
    if isinstance(rev, bool) or not isinstance(rev, int):
        return REV_UNKNOWN
    if rev < current:
        return REV_STALE
    return REV_UNKNOWN if rev > current else None


def card_source_revision(item, man):
    """Ревизия стенограммы, по которой собрана карточка: своя, если карточка собрана позже пакета (upsert_cards),
    иначе — ревизия пакета (и у карточек до PR 13а). Своя ревизия не новее пакетной — ошибка данных или та же
    ревизия: проверяется по ревизии пакета, как раньше."""
    own, base = (item or {}).get("source_revision"), (man or {}).get("source_revision")
    if all(isinstance(x, int) and not isinstance(x, bool) for x in (own, base)) and own > base:
        return own
    return base


def answer_is_current(answer, item):
    """Сохранённый ответ относится к текущей редакции карточки (ответ без rev — к первой)."""
    return bool(answer) and answer.get("rev", 1) == card_rev(item)


# Состояние сбойного прохода (asr_problem.state) в вопросе карточки — по-русски; служебное имя не выводится.
ASR_STATE_RU = {"error": "сбой распознавания",
                "empty_unknown": "модель не выдала слов, неизвестно, была ли здесь речь",
                "filtered": "распознанный текст целиком удалён очисткой"}
ASR_STATE_UNKNOWN_RU = "результат неполон или неизвестен"


# Вопрос новой редакции карточки, чей ответ применён, а место осталось открытым (PR 13б, applied_open).
OPEN_AFTER_ANSWER = "Ответ применён, место осталось открытым. "


def open_after_answer_cards(json_path, ids):
    """Новая редакция карточек ids (механизм 13а, upsert_cards): вопрос прежний с OPEN_AFTER_ANSWER в начале.
    Повтор ничего не пишет: вопрос с этим началом уже стоит. -> id заменённых карточек."""
    items = {it.get("id"): it for it in (manifest(json_path) or {}).get("items") or []}
    cards = [{**{k: items[aid].get(k) for k in ("question", "options", "recommended", "answer_mode", "note")
                 if items[aid].get(k) is not None}, "id": aid, "open_after_answer": True}
             for aid in ids if aid in items]
    if not cards:
        return []
    rep = upsert_cards(json_path, cards, why="ответ применён, место осталось открытым")
    return [row["id"] for row in rep.get("replaced") or []]


def _when(at):
    """«2026-11-25T10:05:00Z» → «25.11 10:05» (UTC)."""
    at = str(at or "")
    return f"{at[8:10]}.{at[5:7]} {at[11:16]}" if len(at) >= 16 else ""


def _who(T, row):
    who = "человек" if row.get("by") == "human" else "агент"
    via = T.CONTRADICTION_VIA_RU.get(row.get("via")) or ("оставлено человеку" if row.get("via") == "held" else "")
    card = row.get("card") if row.get("via") in (None, "card") else None
    return ", ".join(x for x in (who, via + (f" {card}" if card else ""), _when(row.get("at"))) if x)


def _place_words(data, it):
    """Слова места: у противоречия — те, о которых предложение (их видел человек), иначе word_ids."""
    ids = (it.get("contradiction") or {}).get("words") or it.get("word_ids") or []
    by_id = {w.get("word_id"): w for seg in data.get("segments") or [] for w in seg.get("words") or []}
    return [by_id[wid] for wid in ids if wid in by_id]


def _contradiction_options(T, data, it):
    """Варианты карточки противоречия (PR 13б): «Оставить подтверждённое», «Принять новое» на каждое
    предложение (последние), «Как распознано» — только если отличается от обоих. Не больше четырёх."""
    record = it["contradiction"]
    confirmed = record.get("confirmed") or {}
    proposals = [p for p in record["proposals"] if (p.get("text") or "").strip()]
    words = _place_words(data, it)
    raw = " ".join(w.get("model") or w["w"] for w in words)
    raw_needed = bool(raw) and raw != confirmed.get("text") and raw not in [p["text"] for p in proposals]
    opts = [{"label": f"Оставить подтверждённое: «{confirmed.get('text', '')}» ({_who(T, confirmed)})",
             "decision": {"contradiction": "keep"}}]
    for p in proposals[-(2 if raw_needed else 3):]:
        opts.append({"label": f"Принять новое: «{p['text']}» ({_who(T, p)})",
                     "decision": {"text": p["text"], "contradiction": "accept", "proposal": p["key"],
                                  **({"speaker": p["speaker"]} if p.get("speaker") else {})}})
    if raw_needed:
        opts.append({"label": f"Как распознано: «{raw}»", "decision": {"text": raw, "contradiction": "raw"}})
    return opts


def _contradiction_question(T, data, it):
    record = it["contradiction"]
    last = record["proposals"][-1]
    current = " ".join(w["w"] for w in _place_words(data, it))
    return (f"Вы подтвердили «{(record.get('confirmed') or {}).get('text', '')}». "
            f"Предложено «{last.get('text', '')}» ({_who(T, last)}). Сейчас в стенограмме: «{current}». "
            "Что верно?")


def contradiction_cards(json_path, ids):
    """Карточки мест с противоречием в открытом пакете (механизм 13а, upsert_cards).
    -> {id: "added" | "updated" | "unchanged" | "pending"}; pending — пакета нет, карточка появится при сборке."""
    if not ids:
        return {}
    if not manifest(json_path):
        return {rid: "pending" for rid in ids}
    rep = upsert_cards(json_path, [{"id": rid} for rid in ids], why="противоречие")
    out = {rid: "unchanged" for rid in rep.get("unchanged") or []}
    out.update({row["id"]: "added" for row in rep.get("added") or []})
    out.update({row["id"]: "updated" for row in rep.get("replaced") or []})
    out.update({row["id"]: "pending" for row in rep.get("skipped") or []})
    return out


def _card(T, data, it, q):
    """Карточка пакета для места it по вопросу q — без клипа; одна для build и upsert_cards.
    -> (карточка, (t1, t2)) или (None, причина пропуска)."""
    opts = q.get("options") or []
    gap = T.gap_interval(it) if T.is_gap_place(it) else None
    contradiction = T.contradiction_pending(it)
    if contradiction:
        opts = _contradiction_options(T, data, it)
    elif gap:
        # Пропуск (В7): кнопки прослушивания вместо вариантов агента; «как есть» на паузе не показывается.
        opts = ([{"label": T.LISTEN_WORDS, "decision": {"listened": "words"}}] if T.pause_words(data, *gap)
                else []) + [{"label": T.LISTEN_SILENCE, "decision": {"listened": "silence"}},
                            {"label": T.LISTEN_INTERJECTIONS, "decision": {"listened": "interjections"}}]
    elif it.get("asr_problem"):
        opts = [{"label": "Я прослушал: слова в стенограмме верны", "decision": {"listened": "words"}},
                {"label": "Я прослушал: здесь тишина", "decision": {"listened": "silence"}}]
    else:
        opts = [{**o, "decision": {k: v for k, v in (o.get("decision") or {}).items() if k != "listened"}}
                for o in opts]
    if not 1 <= len(opts) <= 4:
        return None, "нужно 1–4 варианта"
    t1, t2 = float(it.get("t1", it["t"])), float(it.get("t2", it["t"]))
    row = _context(T, data, it)
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
        if problem.get("state") == "disagreement":
            question = "Модели распознали по-разному (" + ", ".join(dict.fromkeys([str(model), *names])) + ")"
        else:
            question = (str(model) + ": " + ASR_STATE_RU.get(problem.get("state"), ASR_STATE_UNKNOWN_RU) +
                        ("; слова других проходов: " + ", ".join(names) if names else ""))
        question += ". Прослушайте весь интервал и подтвердите только то, что вы действительно услышали."
    if contradiction:
        question = _contradiction_question(T, data, it)
    if q.get("open_after_answer") and not question.startswith(OPEN_AFTER_ANSWER):
        question = OPEN_AFTER_ANSWER + question
    manifest_item = {
        "id": it["id"], "rev": 1, "t": T.hms(it["t"]), "t1": round(t1, 2), "t2": round(t2, 2),
        **target,
        "clip": None, "clip_t1": None, "clip_t2": None,
        "speaker": it.get("speaker"), "text": it.get("text"),
        "reply": row.get("reply"), "before": row.get("before"), "after": row.get("after"),
        "reasons": it.get("reasons"), "voices": it.get("voices"),
        "question": question,
        "options": [{"label": str(o.get("label", "")), "decision": o.get("decision") or {}} for o in opts],
        "recommended": None if it.get("asr_problem") or gap or contradiction else q.get("recommended"),
        "note": q.get("note")}
    if contradiction:
        # варианты привязаны к словам, которые видел человек: ответ сверяется с ними при «Применить»
        manifest_item["seen_fp"] = T._listening_fingerprint(target_words)
    if gap:
        # отпечаток слов интервала на момент сборки: кнопка годна, пока они не изменились
        manifest_item["listen_fp"] = T._listening_fingerprint(T._interval_words(data, *gap))
    if q.get("answer_mode") == "speaker" and not it.get("asr_problem"):
        # подтверждение говорящего: свободный ответ не становится текстом реплики (freeform.card_edits)
        manifest_item["answer_mode"] = "speaker"
    automation_diagnostic = _automation_diagnostic(data, it)
    if automation_diagnostic:
        manifest_item["diagnostic_id"] = automation_diagnostic["id"]
        manifest_item["automation_diagnostic"] = automation_diagnostic
    return manifest_item, (t1, t2)


def _attach_clip(item, times, src, clip, pd, strict):
    """Нарезать клип карточки. strict (build) — ошибка наружу, как раньше; иначе карточка остаётся без звука."""
    t1, t2 = times
    try:
        if not src.is_file():
            raise RuntimeError(f"исходная запись не найдена: {src}")
        s0 = cut_clip(src, t1, t2, clip)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        if strict:
            raise
        Path(clip).unlink(missing_ok=True)
        return item
    item.update(clip=Path(clip).relative_to(pd).as_posix(), clip_t1=round(t1 - s0, 2), clip_t2=round(t2 - s0, 2))
    return item


def _card_fp(item):
    return json.dumps({k: item.get(k) for k in _CARD_FP_FIELDS}, ensure_ascii=False, sort_keys=True)


def build(json_path, items, base_url_note=None):
    """items: [{id, question, options: [{label, decision}], recommended, answer_mode?}] — вопросы Claude
    по местам review. answer_mode="speaker" — карточка подтверждения говорящего: свой ответ на ней
    применяется только как решение о говорящем (имя или «Имя: точные слова»), иначе — человеку.
    -> отчёт. Пересборка меняет version: ответы на прежние вопросы не применятся.
    Заменить или дописать карточки, не меняя version и не теряя ответов, — upsert_cards."""
    import transcript as T
    data = T.load(json_path)
    import recordings
    sqlite_mode = recordings.read_review_pack_state(json_path) is not None
    src = Path(data.get("file") or "")
    if not src.exists():
        raise RuntimeError(f"исходная запись не найдена: {src}")
    review = {it["id"]: it for it in data.get("review") or []}
    pd = pack_dir(json_path)
    clips_dir = pd / "clips"
    if sqlite_mode:
        clips_dir = clips_dir / uuid.uuid4().hex
    clips_dir.mkdir(parents=True, exist_ok=True)
    out, skipped = [], []
    listed = {q.get("id") for q in items}
    items = list(items) + [{"id": i["id"]} for i in data.get("review") or []
                           if T.contradiction_pending(i) and i.get("id") not in listed]
    for q in items:
        it = review.get(q.get("id"))
        if not it:
            skipped.append({"id": q.get("id"), "why": "нет такого места"})
            continue
        manifest_item, times = _card(T, data, it, q)
        if manifest_item is None:
            skipped.append({"id": it["id"], "why": times})
            continue
        clip = clips_dir / (f"{len(out)}.mp3" if sqlite_mode else f"{it['id']}.mp3")
        out.append(_attach_clip(manifest_item, times, src, clip, pd, strict=True))
    body = json.dumps(out, ensure_ascii=False, sort_keys=True)
    version = time.strftime("%Y%m%d%H%M%S-") + hashlib.sha1(body.encode()).hexdigest()[:6]
    if not sqlite_mode:
        # прежние вопросы — часть истории ответов: manifest сохраняется рядом с архивом ответов,
        # а фрагменты, на которые ссылаются сохранённые манифесты (и истории их карточек), не удаляются
        old = _read(pd / "manifest.json", None)
        if isinstance(old, dict) and old.get("version") and old.get("version") != version:
            archived = pd / f"manifest.{old['version']}.json"
            if not archived.exists():
                _write(archived, old)
        keep = {f"{r['id']}.mp3" for r in out}
        for kept in pd.glob("manifest.*.json"):
            for row in (_read(kept, {}) or {}).get("items") or []:
                if not isinstance(row, dict):
                    continue
                for ref in [row] + [h for h in row.get("history") or [] if isinstance(h, dict)]:
                    if ref.get("clip"):
                        keep.add(Path(ref["clip"]).name)
        for f in clips_dir.glob("*.mp3"):
            if f.name not in keep:
                f.unlink()
    man = {"version": version, "pack_rev": 0, "job_json": str(json_path), "audio": src.name,
           "recording_id": data.get("recording_id"), "source_revision": data.get("source_revision"),
           "speakers": data.get("speakers"), "created": _now(),
           "clips_dir": clips_dir.relative_to(pd).as_posix(), "items": out}
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


def upsert_cards(json_path, cards, *, why, actor="system"):
    """Заменить или дописать карточки в собранном пакете, не меняя version и не трогая ответы других карточек.
    cards — вопросы в виде build. Новая карточка дописывается в конец (rev 1). Заменённая получает rev + 1;
    прежняя редакция с её ответом уходит в history карточки, снимок отмены и отметка применения прежнего
    ответа снимаются: новый вопрос отвечается заново. Тот же вопрос (те же поля _CARD_FP_FIELDS) ничего не
    пишет. Клип — <id>.r<rev>.mp3, прежние не затираются; нет аудио или клип не нарезался — карточка без звука.
    Одна запись состояния на вызов; pack_rev + 1, если что-то изменилось.
    -> {"version", "pack_rev", "written", "added", "replaced", "unchanged", "skipped"}"""
    import recordings
    with recordings.operation_for_document(json_path, actor):
        with _answers_lock(json_path):
            return _upsert_cards_locked(json_path, cards, why)


def _pack_clips_dir(man, sqlite_mode):
    """Папка клипов пакета относительно папки пакета: из manifest, у пакетов до PR 13а — по клипам карточек."""
    if man.get("clips_dir"):
        return man["clips_dir"]
    for it in man.get("items") or []:
        if it.get("clip"):
            return Path(it["clip"]).parent.as_posix()
    return "clips/" + uuid.uuid4().hex if sqlite_mode else "clips"


def _free_clip_name(folder, aid, rev):
    """<id>.r<rev>.mp3; если такой файл уже есть (архивный manifest прежней версии), имя с уникальным хвостом."""
    clip = folder / f"{aid}.r{rev}.mp3"
    return clip if not clip.exists() else folder / f"{aid}.r{rev}.{uuid.uuid4().hex[:8]}.mp3"


def _history_row(old, answer, applied, conflicts, why):
    row = {"rev": card_rev(old), "question": old.get("question"), "options": copy.deepcopy(old.get("options")),
           "recommended": old.get("recommended"), "clip": old.get("clip"), "at": _now(), "why": why}
    if answer_is_current(answer, old):
        row["answer"] = {k: copy.deepcopy(v) for k, v in answer.items() if k not in _ANSWER_PRIVATE}
        row["applied"] = applied
    if conflicts:
        row["conflicts"] = conflicts
    return row


def _upsert_cards_locked(json_path, cards, why):
    import transcript as T
    man = manifest(json_path)
    if not man:
        raise RuntimeError("пакет разбора не собран: сначала build_review_pack")
    data = T.load(json_path)
    if man.get("recording_id") and man["recording_id"] != data.get("recording_id"):
        raise RuntimeError("пакет относится к другой стенограмме")
    store = _read_answers(json_path, {"version": man["version"], "answers": {}, "conflicts": [], "rejected": []})
    sqlite_mode = "_expected_versions" in store
    if sqlite_mode and store["_expected_versions"] != {"source_revision": data.get("source_revision"),
                                                        "review_revision": data.get("review_revision")}:
        import storage
        raise storage.Conflict("recording changed while updating review pack")
    pd = pack_dir(json_path)
    clips_rel = _pack_clips_dir(man, sqlite_mode)
    src = Path(data.get("file") or "")
    review = {it["id"]: it for it in data.get("review") or []}
    items = list(man.get("items") or [])
    index = {it.get("id"): n for n, it in enumerate(items)}
    applied = _applied_map(store)
    rep = {"added": [], "replaced": [], "unchanged": [], "skipped": []}
    seen = set()
    for q in cards:
        aid = q.get("id")
        it = review.get(aid)
        if not it:
            rep["skipped"].append({"id": aid, "why": "нет такого места"})
            continue
        if aid in seen:
            rep["skipped"].append({"id": aid, "why": "карточка повторяется в вызове"})
            continue
        seen.add(aid)
        new, times = _card(T, data, it, q)
        if new is None:
            rep["skipped"].append({"id": aid, "why": times})
            continue
        old = items[index[aid]] if aid in index else None
        if old is not None and _card_fp(old) == _card_fp(new):
            rep["unchanged"].append(aid)
            continue
        new["rev"] = card_rev(old) + 1 if old is not None else 1
        if data.get("source_revision") is not None:
            # слова карточки сверяются при «Применить» с ревизией, по которой она собрана, а не с ревизией пакета
            new["source_revision"] = data["source_revision"]
        folder = pd / clips_rel
        folder.mkdir(parents=True, exist_ok=True)
        _attach_clip(new, times, src, _free_clip_name(folder, aid, new["rev"]), pd, strict=False)
        if old is None:
            index[aid] = len(items)
            items.append(new)
            rep["added"].append({"id": aid, "rev": 1, "clip": bool(new["clip"])})
            continue
        answers = store.setdefault("answers", {})
        conflicts = [c for c in store.get("conflicts") or [] if c.get("id") == aid]
        new["history"] = list(old.get("history") or []) + [
            _history_row(old, answers.get(aid), aid in applied, conflicts, why)]
        answers.pop(aid, None)
        if conflicts:
            store["conflicts"] = [c for c in store["conflicts"] if c.get("id") != aid]
        if "applied_answers" in store:
            store["applied_answers"].pop(aid, None)
        (store.get("undo_snapshots") or {}).pop(aid, None)
        items[index[aid]] = new
        rep["replaced"].append({"id": aid, "rev": new["rev"], "clip": bool(new["clip"])})
    pack_rev = man.get("pack_rev", 0)
    if not rep["added"] and not rep["replaced"]:
        return {"version": man["version"], "pack_rev": pack_rev, "written": False, **rep}
    man = dict(man, items=items, pack_rev=pack_rev + 1, clips_dir=clips_rel, updated=_now())
    store["manifest"] = man
    if not sqlite_mode:
        # manifest первым: при сбое между записями прежний ответ уже в истории карточки, а оставшийся
        # в answers.json ответ прежней редакции не применяется (answer_is_current)
        _write(pd / "manifest.json", man)
        if not (pd / "cards.html").exists():
            shutil.copyfile(HERE / "cards.html", pd / "cards.html")
    _write_answers(json_path, store)
    return {"version": man["version"], "pack_rev": man["pack_rev"], "written": True, **rep}


CTX_BEFORE = 400     # символов соседних реплик до места
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


def add_interaction_event(json_path, aid, version, event_type, session, rev=None):
    """Сохраняет локальный факт показа карточки или действия с её клипом (rev — редакция карточки на странице)."""
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
    stale = rev_refusal(rev, item)
    if stale:
        raise ValueError(stale)
    with _answers_lock(json_path):
        store = _read_answers(json_path, {"version": man["version"], "answers": {}})
        events = store.setdefault("interaction_events", [])
        if event_type == "card_visible" and any(
                row.get("id") == aid and row.get("version") == version and row.get("rev", 1) == card_rev(item) and
                row.get("type") == event_type and row.get("session") == session for row in events):
            return {"stored": False, "duplicate": True}
        event = {"id": aid, "version": version, "diagnostic_id": item.get("diagnostic_id"),
                 "type": event_type, "session": session, "source": "local", "ts": _now()}
        if card_rev(item) != 1:
            event["rev"] = card_rev(item)
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
        # номер варианта выбирает вариант и рядом со словом подтверждения или оценки: «верно, второй», «да 2»
        for x in w:
            if x.isdigit() and 1 <= int(x) <= len(opts):
                return {"kind": "option", "option": int(x) - 1}
        for x in w:
            for stem, n in _ORD:
                if x.startswith(stem) and n < len(opts):
                    return {"kind": "option", "option": n}
    if freeform.classify(text, [])["kind"] in freeform.SHORT_KINDS:
        # голос = текст (В8): «да», «верно», «согласен» без номера разбирает таблица слов, рекомендацию они не выбирают
        return {"kind": "free"}
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
        stale = rev_refusal(a.get("rev"), it)
        if stale:
            # сверка по паре (id, rev): ответ на заменённую редакцию не ложится на новый вопрос
            rej = {"id": aid, "why": stale, "version": a.get("version"), "rev": a.get("rev")}
            rep["rejected"].append(rej)
            store.setdefault("rejected", []).append({**rej, "source": source, "ts": a.get("ts") or _now()})
            continue
        ans = {"id": aid, "version": a["version"], "source": source, "ts": a.get("ts") or _now()}
        if card_rev(it) != 1:
            ans["rev"] = card_rev(it)        # ответ без rev — к первой редакции (answer_is_current)
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
        note = a.get("human_note")
        if isinstance(note, str) and note.strip() and "seen_fp" in it:
            # примечание — только на карточке противоречия (В-3); ответом оно не разбирается
            ans["human_note"] = note.strip()[:2000]
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
        if old and not answer_is_current(old, it):
            old = None                     # ответ прежней редакции, оставшийся после сбоя замены, — не соперник
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
    places = {r.get("id"): r for r in data.get("review") or []}
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
        if not answer_is_current(a, it):       # ответ прежней редакции карточки к новому вопросу не применяется
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
        if "seen_fp" in it and T.contradiction_pending(places.get(aid) or {}) and \
                T._listening_fingerprint(_place_words(data, places[aid])) != it["seen_fp"]:
            rep.setdefault("source_conflict", CONTRADICTION_STALE)
            rep["source_conflict_ids"].append(aid)
            rep.setdefault("source_conflict_why", {})[aid] = CONTRADICTION_STALE
            continue
        if k in ("option", "recommended"):
            d = dict(it["options"][a["option"]]["decision"] or {})
            if d.get("listened") and "listen_fp" in it:
                # кнопка паузы (6б): годна, пока слова её интервала те же, что при сборке пакета
                gap = T.gap_interval(places.get(aid) or {})
                if not gap or T._listening_fingerprint(T._interval_words(data, *gap)) != it["listen_fp"]:
                    rep.setdefault("source_conflict", LISTEN_STALE)
                    rep["source_conflict_ids"].append(aid)
                    rep.setdefault("source_conflict_why", {})[aid] = LISTEN_STALE
                    continue
            elif d.get("listened") and card_source_revision(it, man) != data.get("source_revision"):
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
            if a.get("human_note"):
                d["human_note"] = a["human_note"]
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
    store = _read_answers(json_path, {"answers": {}})
    items = {it.get("id"): it for it in (manifest(json_path) or {}).get("items") or []}
    applied = _applied_map(store)
    return {aid: a for aid, a in store.get("answers", {}).items() if a.get("kind") == "free" and
            answer_is_current(a, items.get(aid)) and
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
                                    answers={k: v.get("text") for k, v in valid_free.items()}, dry_run=dry_run,
                                    listened_by="human")
    else:
        result_data = data
        rep = {"cards": {}, "failed": {}, "proposed_delete": [], "not_found": [],
               "relabel": 0, "replace": 0, "insert": 0}
    items = {r["id"]: r for r in data.get("review") or []}
    confirmed, confirm_refusals = [], {}
    if not dry_run:
        # ответ совпал со словами, уже стоящими в паузе: открытое место закрывается тем же путём, что «как есть»
        now_places = {r["id"]: r for r in T.load(json_path).get("review") or []}
        still_open = {rid for rid, r in now_places.items() if T.place_unresolved(r)}
        confirmed = [r["id"] for r in reports if r.get("confirm_as_is") and r["id"] not in blocked
                     and r["id"] in still_open]
        if confirmed:
            # «как есть» человека; у места-слова — по его ID слов из пакета, как у варианта (decisions)
            man_items = {it["id"]: it for it in man["items"]}
            # то же слово, вписанное человеком на месте «модели расходятся», — его прослушивание (PR 7);
            # «верно» и другие оценки им не считаются: они не говорят, что человек слышал
            same_word = {r["id"] for r in reports if r.get("kind") == "short_word"}
            asked = [{"id": cid, **({"word_ids": man_items[cid]["word_ids"]}
                                    if "word_ids" in man_items.get(cid, {})
                                    and not T.is_gap_place(man_items[cid]) else {}),
                      **({"listened": "words", "via": "card"} if cid in same_word and (
                          now_places[cid].get("asr_problem") or {}).get("state") == "disagreement" else {})}
                     for cid in confirmed]
            _, nested = T.apply_review(json_path, md_path, asked, by="human")
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
    hint_ids = {r["id"] for r in reports if r.get("kind") == "hint"}
    mark_human = [h for h in human if h["id"] not in negative_ids | hint_ids]
    # не понятые ответы — человеку; отрицание и подсказка на короткий ответ оставляют место открытым
    if mark_human and not dry_run:
        T.apply_review(json_path, md_path, [{"id": h["id"], "status": "human",
                                             "note": f"{h['why']}: «{h['answer']}»"} for h in mark_human], by="human")
    return {**({"target_ids": {aid: sorted(ids) for aid, ids in targets.items()}} if dry_run else {}),
            "cards": [{"id": r["id"], "answer": r["answer"], "kind": r["kind"],
                       "was_now": rep["cards"].get(r["id"], {}).get("before", "") + " → " +
                       rep["cards"].get(r["id"], {}).get("after", "") if r["id"] in rep["cards"] else None,
                       "applied": r["applied"], "inserted": r["inserted"], "typos": r["typos"]} for r in reports],
            "closed_via": closed, "conflicts": conflicts, "left_to_human": human, "hints": sorted(hint_ids),
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
    clips = {it["clip"]: pd / it["clip"] for it in man["items"] if it.get("clip")}
    have = {k: v for k, v in clips.items() if v.exists()}
    items = {it["id"]: it for it in man["items"]}
    # Копия применённого ответа (applied_answer) и отметка запроса (keep_request) нужны только ноутбуку:
    # странице хватает answer_states (keep: ts, source, accepted).
    # Отпечаток слов интервала паузы (listen_fp) нужен только сверке на ноутбуке (decisions): в облако не уходит.
    # Ответ прежней редакции карточки в облако не уходит: страница показывает его из истории карточки.
    out = dict(man, items=[_cloud_item(it) for it in man["items"]],
               answers={k: _cloud_answer(v) for k, v in store.get("answers", {}).items()
                        if v.get("version") == man["version"] and answer_is_current(v, items.get(k))},
               applied_answers={k: _cloud_answer(v) for k, v in (store.get("applied_answers") or {}).items()},
               answer_states=answer_states_of(store, man),
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


def _cloud_answer(a):
    """Ответ для облака: без служебных полей ноутбука и без локального пути к голосовому файлу.
    Подпись применённого ответа (applied_answers — список) идёт как есть."""
    return {f: x for f, x in a.items() if f not in _ANSWER_LOCAL} if isinstance(a, dict) else a


def _cloud_item(it):
    """Карточка для облака: без отпечатка слов паузы, служебных полей и путей ответов в истории."""
    out = {f: x for f, x in it.items() if f not in ("listen_fp", "seen_fp")}
    if it.get("history"):
        out["history"] = [dict(h, answer=_cloud_answer(h["answer"])) if isinstance(h.get("answer"), dict)
                          else dict(h) for h in it["history"]]
    return out


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
    if relation == "applied" and (event or {}).get("outcome") == "applied_open":
        return "applied_open"          # применён, место открыто (PR 7): подпись та же, исход последнего применения
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
    items = {it.get("id"): it for it in (man or {}).get("items") or []}
    cards = {aid: n for n, aid in enumerate(items, 1)}
    rejected_keeps = {r.get("id"): r for r in store.get("rejected") or [] if r.get("kind") == "keep"}
    out = {}
    for aid, answer in (store.get("answers") or {}).items():
        if not answer_is_current(answer, items.get(aid)):
            continue                      # ответ прежней редакции: страница показывает его в истории карточки
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
