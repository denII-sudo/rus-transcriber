"""Конспект, связанный со стенограммой: цитаты — ссылки на места стенограммы, текст подставляется из неё.

Файл конспекта — <имя записи>.<название>.conspect.json в подпапке результатов:
{
  "name": "конспект", "job_json": "...transcript.json",
  "template": "markdown с {{q:1}} на местах цитат",
  "quotes": {"1": {"parts": [{"t1": 12.34, "t2": 15.6}], "author": "текст как в конспекте",
                   "speaker": "Вера" | null,               — имя перед цитатой («Вера: «…»»)
                   "base": "текст стенограммы при связывании", "base_speaker": "...",
                   "last": "текст при последней сборке", "last_speaker": "..."}},
  "speaker_labels": [{"label": "Участник 1", "word_ids": [...]}]   — только у модельной свёртки с условными
                   метками говорящих: по словам метки сборка находит текущего говорящего и пишет шапку файла
}
Сборка (render): новые ссылки используют word_id, прежние — временное окно с проверкой неоднозначности.
  * текст места не менялся с момента связывания — выводится авторская версия цитаты (в ней могут быть
    [поправки в квадратных скобках]);
  * текст изменился (правка стенограммы) — выводится новый текст, место попадает в отчёт «обновлено»;
  * говорящий: имя автора остаётся, пока стенограмма его не меняла; если говорящий места в стенограмме
    изменился после связывания — выводится новый, место в отчёте;
  * место не найдено (слова удалены) — выводится авторская версия с пометкой, место в отчёте.
Связь односторонняя: правки в документе Claude Docs в стенограмму не возвращаются.
"""
import json
import re
import base64
import hashlib
import os
import threading
import time
import uuid
from contextlib import contextmanager
from collections import Counter
from pathlib import Path

import quotes as Q

PLACE = re.compile(r"\{\{q:(\d+)\}\}")
# Слот ненайденной цитаты (РАР-7, 7.5в): {{u:N}}; сборка раскрывает его по списку obj["unlinked"].
SLOT = re.compile(r"\{\{([qu]):(\d+)\}\}")
SLOT_MARKS = ("{{q:", "{{u:")           # служебные знаки шаблона: в тексте ответа модели их быть не может
_locks_guard = threading.Lock()
_thread_locks = {}
_held_locks = threading.local()


@contextmanager
def write_lock(json_path):
    """Общая блокировка записи; вложенные вызовы того же потока допускаются."""
    import jobslots
    key = os.path.normcase(str(Path(json_path).resolve()))
    with _locks_guard:
        lock = _thread_locks.setdefault(key, threading.RLock())
    with lock:
        held = getattr(_held_locks, "keys", None)
        if held is None:
            held = _held_locks.keys = set()
        if key in held:
            yield
            return
        path = Path(key + ".conspect.lock")
        with path.open("a+b") as f:
            while not jobslots._try_lock(f):
                time.sleep(0.1)
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)


def _digest(value):
    return None if value is None else hashlib.sha256(value).hexdigest()


def _read_bytes(path):
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _replace_bytes(path, value):
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("xb") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _journal_path(json_path):
    return Path(str(json_path) + ".conspect.pending")


def recover_pending(json_path):
    """Под блокировкой откатить прерванную запись, только если файлы не менялись извне."""
    journal = _journal_path(json_path)
    if not journal.exists():
        return False
    record = json.loads(journal.read_text(encoding="utf-8"))
    if record.get("version") != 1 or not isinstance(record.get("files"), list):
        raise ValueError("повреждён журнал конспекта; требуется ручное восстановление")
    entries = []
    parent = Path(json_path).parent.resolve()
    for item in record["files"]:
        name = item["name"]
        if not isinstance(name, str) or Path(name).name != name or not name.endswith((".conspect.json", ".conspect.md")):
            raise ValueError("неверный путь в журнале конспекта")
        path = parent / name
        old = None if item["old"] is None else base64.b64decode(item["old"], validate=True)
        current = _digest(_read_bytes(path))
        if current not in (item["old_hash"], item["new_hash"]) or _digest(old) != item["old_hash"]:
            raise ValueError("файлы конспекта изменились после сбоя; требуется ручное восстановление")
        entries.append((path, old))
    for path, old in entries:
        if old is None:
            if path.exists():
                path.unlink()
        else:
            _replace_bytes(path, old)
    journal.unlink()
    return True


def write_files(json_path, files, create=False):
    """Для SQLite-записи сохранить экспорт как консект-ревизию; legacy остаётся файловым."""
    import recordings
    store, binding = recordings.open_for_document(json_path)
    if store is not None:
        try:
            if len(files) == 1 and Path(files[0][0]).name.endswith(".conspect.json"):
                obj = json.loads(files[0][1].decode("utf-8"))
                import transcript
                md, _ = render(obj, transcript.load(json_path))
                md_path = Path(str(files[0][0])[:-len(".json")] + ".md")
                files = [*files, (md_path, md.encode("utf-8"))]
        finally:
            store.close()
        return recordings.save_conspect_files(json_path, files, create=create)
    return _write_files_raw(json_path, files, create=create)


def _write_files_raw(json_path, files, create=False):
    """Под блокировкой записать набор файлов; при сбое оставить журнал для отката."""
    journal = _journal_path(json_path)
    if journal.exists():
        raise ValueError("есть незавершённая запись конспекта")
    entries = []
    for path, value in files:
        path = Path(path)
        old = _read_bytes(path)
        if create and old is not None:
            raise FileExistsError("конспект уже существует")
        entries.append({"name": path.name, "old": None if old is None else base64.b64encode(old).decode("ascii"),
                        "old_hash": _digest(old), "new_hash": _digest(value)})
    _replace_bytes(journal, json.dumps({"version": 1, "files": entries}, ensure_ascii=False).encode("utf-8"))
    for path, value in files:
        _replace_bytes(Path(path), value)
    journal.unlink()


def _tokens(data):
    """[(слово, говорящий, время начала слова)] — по словам, а для старых расшифровок — равномерно по реплике.

    У слова без времени (перенесённые старые данные) время — None: цитата по нему связывается по тексту и ID."""
    out = []
    for seg in data["segments"]:
        ws = seg.get("words")
        if ws:
            for w in ws:
                toks = Q.WORD.findall(w["w"])
                for x in toks:
                    out.append((Q._n(x), seg["speaker"], _start(w), w.get("word_id")))
        else:
            toks = Q.WORD.findall(seg["text"])
            n = max(1, len(toks))
            for k, x in enumerate(toks):
                out.append((Q._n(x), seg["speaker"], round(seg["start"] + (seg["end"] - seg["start"]) * k / n, 3), None))
    return out


def _start(word):
    """Время начала слова или None, если его нет."""
    start = word.get("s")
    return start if type(start) in (int, float) else None


def _part_times(toks):
    """(t1, t2) отрезка — по словам, у которых есть время; (None, None), если времени нет ни у одного."""
    times = [tok[2] for tok in toks if tok[2] is not None]
    return (times[0], times[-1]) if times else (None, None)


def _span_text(data, t1, t2):
    """Текст и говорящий слов с t1 ≤ начало ≤ t2 (по словам или по реплике целиком)."""
    words, spk = [], {}
    for seg in data["segments"]:
        ws = seg.get("words")
        if ws:
            for w in ws:
                if _start(w) is not None and t1 - 0.005 <= w["s"] <= t2 + 0.005:
                    words.append(w["w"])
                    spk[seg["speaker"]] = spk.get(seg["speaker"], 0) + 1
        elif seg["start"] <= t2 and seg["end"] >= t1:
            toks = seg["text"].split()
            n = max(1, len(toks))
            for k, x in enumerate(toks):
                t = seg["start"] + (seg["end"] - seg["start"]) * k / n
                if t1 - 0.05 <= t <= t2 + 0.05:
                    words.append(x)
                    spk[seg["speaker"]] = spk.get(seg["speaker"], 0) + 1
    if not words:
        return None, None
    text = " ".join(words).strip(" ,;:—–-")
    return text, max(spk, key=spk.get)


def _clean(s):
    return " ".join(Q._n(x) for x in Q.WORD.findall(s))


def _shape(data, ids):
    """Строение слов цитаты при связывании (Н28-3…Н28-5): для корня каждого слова цитаты (`part_of` или само слово) —
    все ID его частей. Ссылка с этой отметкой сделана на конкретные слова: её текст — ровно эти ID."""
    words = [w for seg in data.get("segments") or [] for w in seg.get("words") or [] if w.get("word_id")]
    cited = [w for w in words if w["word_id"] in set(ids or [])]
    roots = {w.get("part_of") or w["word_id"] for w in cited}
    return {"roots": {root: sorted(w["word_id"] for w in words if (w.get("part_of") or w["word_id"]) == root)
                      for root in sorted(roots)},
            "span": [min(w["s"] for w in cited), max(w["e"] for w in cited)] if cited else None}


def _exact_holds(data, ids, exact):
    """Точная ссылка остаётся точной, пока ни одно слово цитаты не разбили после связывания: новая часть корня
    цитаты, лежащая во времени слов цитаты, — это разбиение её слова, и укороченный текст выдать нельзя (Н-7).
    Новая часть вне этого времени — разбиение соседнего слова того же корня: цитату не меняет (Н28-5); на самой
    границе (вставка нулевой длины) не различить — как у старой ссылки. Слово цитаты,
    вышедшее за время цитаты при связывании, вобрало соседнее слово: тоже не точная. Время — на момент связывания."""
    if not isinstance(exact, dict) or not exact.get("span"):
        return False
    known = {wid for parts in (exact.get("roots") or {}).values() for wid in parts}
    words = [w for seg in data.get("segments") or [] for w in seg.get("words") or [] if w.get("word_id")]
    cited = [w for w in words if w["word_id"] in set(ids or [])]
    if not cited:
        return True                                       # слов нет — «не найдено» решит общий разбор
    roots = set(exact.get("roots") or {})
    lo, hi = exact["span"]
    eps = 0.005
    if any(w["s"] < lo - eps or w["e"] > hi + eps for w in cited):
        return False
    return not any((w.get("part_of") or w["word_id"]) in roots and w["word_id"] not in known
                   and w["s"] <= hi + eps and w["e"] >= lo - eps for w in words)


def _id_text(data, ids, exact=None):
    """Разрешить точные ID, не прибегая к временным окнам или порядку хранения. ID слова распознавания, которое
    ответы заменили несколькими словами (Ф234-11б), покрывает и их, в порядке стенограммы: у них `part_of` с этим ID
    (корнем), в том числе после повторных ответов. Слово с самим этим ID могло быть удалено — ссылка покрывает
    оставшиеся. Часть слова ушла в слово вместе со словами другого корня (`merged_from`, Н-7) — ссылка на него не
    находится, как на main, где склеенное слово заменялось целиком: укороченный текст не выдаётся за прежний.
    exact — строение слов при связывании (`_shape`): ссылка сделана на конкретные слова, и пока строение их корней
    не выросло, корень не разворачивается и `merged_from` не мешает (Н28-3…Н28-5); иначе — как у старой ссылки."""
    if exact is not None and _exact_holds(data, ids, exact):
        return _plain_text(data, ids)
    parts, present, merged = {}, set(), set()
    for seg in data.get("segments") or []:
        for word in seg.get("words") or []:
            merged.update(word.get("merged_from") or [])
            if word.get("word_id"):
                present.add(word["word_id"])
                if word.get("part_of"):
                    parts.setdefault(word["part_of"], []).append(word["word_id"])
    if any(wid in merged for wid in ids or []):
        return None, None
    if ids and parts:
        rank = {wid: i for i, wid in enumerate(w.get("word_id") for seg in data.get("segments") or []
                                               for w in seg.get("words") or [])}
        ids = list(dict.fromkeys(x for wid in ids for x in sorted(
            ([wid] if wid in present or wid not in parts else []) + parts.get(wid, []),
            key=lambda x: rank.get(x, -1))))
    return _plain_text(data, ids)


def _plain_text(data, ids):
    """Текст и говорящий ровно этих ID в порядке ссылки (общая часть `_id_text`)."""
    by_id = {}
    order = []
    for seg in data.get("segments") or []:
        for word in seg.get("words") or []:
            wid = word.get("word_id")
            order.append((wid, seg["speaker"]))
            if wid:
                by_id.setdefault(wid, []).append((word["w"], seg["speaker"]))
    if not ids or len(set(ids)) != len(ids) or any(len(by_id.get(wid, [])) != 1 for wid in ids):
        return None, None
    found = [by_id[wid][0] for wid in ids]
    if len({speaker for _, speaker in found}) != 1:
        return None, None
    positions = [next(i for i, (wid, _) in enumerate(order) if wid == target) for target in ids]
    if positions != sorted(positions):
        return None, None
    speaker = found[0][1]
    if any(wid not in ids and spk == speaker for wid, spk in order[positions[0]:positions[-1] + 1]):
        return None, None
    return " ".join(word for word, _ in found).strip(" ,;:—–-"), found[0][1]


def _legacy_span(data, t1, t2):
    """Старое окно допустимо только без наложения/совпадений времени."""
    seen, speakers, legacy_segments = set(), set(), 0
    for seg in data.get("segments") or []:
        if not seg.get("words") and seg.get("start", 0) <= t2 and seg.get("end", 0) >= t1:
            legacy_segments += 1
            speakers.add(seg["speaker"])
        for word in seg.get("words") or []:
            if _start(word) is not None and t1 - 0.005 <= word["s"] <= t2 + 0.005:
                key = round(word["s"], 3)
                if key in seen:
                    return None, None
                seen.add(key)
                speakers.add(seg["speaker"])
    if len(speakers) > 1 or legacy_segments > 1 or (legacy_segments and seen):
        return None, None
    return _span_text(data, t1, t2)


# Метка говорящего, написанная вплотную перед цитатой («Участник 2: «…»»): часть цитаты, не текста раздела.
_LABEL_BEFORE = re.compile(r"(Участник\s+\d+|Говорящий не определён)\s*[:—–-]\s*$")
UNLINKED_MARKS = {"none": "не найдено в стенограмме", "several": "в разделе несколько таких мест"}


def link(conspect_md, data, min_score=0.75, selections=None, *, within=None, unlinked="reject", labels=False):
    """Связывает только однозначные точные цитаты. min_score оставлен для совместимости вызовов.

    Параметры правила «цитаты по одной» (РАР-7, 7.5; без них поведение прежнее):
    within — слова раздела (ID): цитата ищется только среди них;
    unlinked="mark" — несвязанная цитата остаётся в тексте без кавычек с пометкой (UNLINKED_MARKS),
    число таких цитат — в unlinked_count; по умолчанию "reject": текст не меняется;
    unlinked="slot" — как "mark", но вместо пометки в шаблоне стоит слот {{u:N}}, а в obj["unlinked"] —
    {"id", "quote", "why", "at"}: at — место «цитаты» с кавычками в исходном тексте (без метки перед ней);
    сборка (render) печатает ту же пометку, что и "mark";
    labels — метка говорящего вплотную перед цитатой считается частью цитаты: убирается из текста,
    в цитате остаётся written_label, автора пишет сборка.
    """
    unlinked_mode = unlinked                # ниже имя unlinked занимает список несвязанных цитат
    toks = _tokens(data)
    seq = [t[0] for t in toks]
    first = {}
    for i, w in enumerate(seq):
        first.setdefault(w, []).append(i)
    names = {seg["speaker"] for seg in data["segments"]}
    id_counts = Counter(w.get("word_id") for seg in data["segments"] for w in seg.get("words") or [])
    quotes, unlinked, candidates_report, slots = {}, [], [], []
    selections = selections or {}
    consumed = set()
    out, pos, n = [], 0, 0

    def mark_unlinked(m, q, why):
        nonlocal pos
        unlinked.append(q)
        if unlinked_mode in ("mark", "slot"):
            cut = label_cut(m)[0]
            out.append(conspect_md[pos:cut])
            if unlinked_mode == "slot":
                slots.append({"id": str(len(slots) + 1), "quote": q, "why": why, "at": [m.start(2) - 1, m.end()]})
                out.append("{{u:%d}}" % len(slots))
            else:
                out.append(f"{q} ({UNLINKED_MARKS[why]})")
            pos = m.end()

    def label_cut(m):
        """(откуда вырезать, написанная метка): имя перед цитатой или метка в конце текста перед ней."""
        if not labels:
            return (m.start(1) if m.group(1) else m.start(2) - 1), m.group(1)
        if m.group(1) in names:                 # имя говорящего перед цитатой; «Итог:» и т. п. — обычный текст
            return m.start(1), m.group(1)
        start = m.start(2) - 1
        found = _LABEL_BEFORE.search(conspect_md[pos:start])
        if found:
            return pos + found.start(), " ".join(found.group(1).split())
        return start, None

    for quote_index, m in enumerate(Q.QUOTE.finditer(conspect_md), 1):
        spk, q = m.group(1), m.group(2)
        why = "none"
        if spk and spk not in names and not labels:
            mark_unlinked(m, q, why)
            continue
        clean = re.sub(r"\[[^\]]*\]", "…", q)
        raw_parts = [p for p in re.split(r"…|\.\.\.", clean)]
        parts, ok = [], True
        # До отдельной реализации согласованного выбора частей не связывать
        # разорванную цитату с независимо найденными разными местами.
        if len(raw_parts) != 1 or not data.get("recording_id"):
            mark_unlinked(m, q, why)
            continue
        for p in raw_parts:
            toks_p = [Q._n(w) for w in Q.WORD.findall(p)]
            if not toks_p:
                continue
            # Приближённое совпадение показывается проверяющему, но не становится ссылкой автоматически.
            matches = Q.candidates(toks_p, toks, speaker=None if labels else spk, min_score=1.0)
            if within is not None:
                matches = [hit for hit in matches if all(t[3] in within for t in toks[hit[1]:hit[1] + hit[2]])]
            options = []
            for _, ci, cl in matches[:21]:
                selected_ids = list(dict.fromkeys(t[3] for t in toks[ci:ci + cl]))
                if None not in selected_ids and all(id_counts[wid] == 1 for wid in selected_ids):
                    t1, t2 = _part_times(toks[ci:ci + cl])
                    options.append({"word_ids": selected_ids, "speaker": toks[ci][1], "t1": t1, "t2": t2})
            if len(matches) > 1:
                candidates_report.append({"quote_index": quote_index, "quote": q, "candidates": options[:20],
                                          "truncated": len(matches) > 20})
            selection = selections.get(str(quote_index))
            if selection is not None:
                if len(matches) < 2 or not isinstance(selection, dict) or data.get("source_revision") is None or selection.get("recording_id") != data.get("recording_id") or \
                        selection.get("source_revision") != data.get("source_revision") or \
                        not any(option["word_ids"] == selection.get("word_ids") for option in options[:20]):
                    raise ValueError("выбранное место устарело или отсутствует")
                match_index = next(k for k, (_, ci, cl) in enumerate(matches)
                                   if list(dict.fromkeys(t[3] for t in toks[ci:ci + cl])) == selection["word_ids"])
                matches = [matches[match_index]]
                consumed.add(str(quote_index))
            if len(matches) != 1:
                ok = False
                why = "several" if len(matches) > 1 else "none"
                break
            r, i, L = matches[0]
            ids = list(dict.fromkeys(t[3] for t in toks[i:i + L]))
            if None in ids or any(id_counts[wid] != 1 for wid in ids):
                ok = False
                break
            t1, t2 = _part_times(toks[i:i + L])
            parts.append({"t1": t1, "t2": t2, "word_ids": ids, "exact": _shape(data, ids)})
        if not ok or not parts:
            mark_unlinked(m, q, why)
            continue
        n += 1
        key = str(n)
        t1, t2 = parts[0]["t1"], parts[-1]["t2"]
        base, base_spk = _id_text(data, parts[0]["word_ids"], parts[0]["exact"])
        joined = " … ".join(filter(None, (_id_text(data, p["word_ids"], p["exact"])[0] for p in parts)))
        quotes[key] = {"parts": parts, "author": q, "speaker": None if labels else spk, "base": joined,
                       "base_speaker": base_spk, "last": joined, "last_speaker": base_spk}
        # «Имя: «…»» — имя тоже часть ссылки, чтобы обновлялось вместе с говорящим
        start, written = label_cut(m)
        if labels:
            quotes[key]["written_label"] = written
        out.append(conspect_md[pos:start])
        out.append(f"{{{{q:{key}}}}}")
        pos = m.end()
    out.append(conspect_md[pos:])
    if set(selections) != consumed:
        raise ValueError("часть выбранных цитат отсутствует или не может быть связана")
    obj = {"version": 2, "template": "".join(out), "quotes": quotes,
           "recording_id": data.get("recording_id"), "source_revision": data.get("source_revision"),
           "analysis_revision": None}
    if unlinked_mode in ("mark", "slot"):
        obj["unlinked_count"] = len(unlinked)
    if unlinked_mode == "slot":
        obj["unlinked"] = slots
    return obj, {"linked": n, "unlinked": unlinked, "candidates": candidates_report,
                 "recording_id": data.get("recording_id"), "source_revision": data.get("source_revision")}


SECTION_TIME = re.compile(r"^\[\d{2,}:\d\d:\d\d\]$", re.M)


def _speaker_display(speaker, names):
    """Как назвать говорящего в шапке: текущее имя или честное «имя не указано»."""
    if not speaker:
        return "говорящий в стенограмме не указан"
    name = " ".join(str(names.get(speaker, speaker)).split())
    if not name or re.fullmatch(r"Спикер(\s*\d+)?", name):
        return f"имя не указано (в стенограмме: {name or speaker})"
    return name


def _labels_header(obj, data, needs_review):
    """Шапка модельного конспекта с условными метками говорящих («Участник N»).

    Собирается при каждой сборке файла по текущей стенограмме: метка привязана к словам
    (`speaker_labels[i].word_ids` — первые слова реплик этой метки), а не к имени на момент
    создания, поэтому после переименования говорящих в файле стоят текущие имена.
    У конспектов без `speaker_labels` шапки нет — они собираются как раньше.
    """
    labels = obj.get("speaker_labels")
    if not isinstance(labels, list):
        return ""
    names = data.get("speaker_names")
    names = names if isinstance(names, dict) else {}
    current = {word.get("word_id"): seg.get("speaker")
               for seg in data.get("segments") or [] for word in seg.get("words") or []}
    lines = []
    for item in labels:
        found = Counter(current[wid] for wid in item.get("word_ids") or [] if wid in current)
        shown = list(dict.fromkeys(_speaker_display(speaker, names) for speaker, _ in found.most_common()))
        text = "реплики не найдены в текущей стенограмме" if not shown else shown[0] + (
            f"; часть реплик в текущей стенограмме: {', '.join(shown[1:])}" if shown[1:] else "")
        lines.append(f"- {item.get('label')} — {text}")
    head = "# Конспект\n\nСоздан моделью" + ("; проверка человеком не подтверждена" if needs_review else "") + ".\n\n"
    if lines:
        head += "Участники (условные метки; имена — по текущей стенограмме):\n\n" + "\n".join(lines) + "\n\n"
    if SECTION_TIME.search(obj.get("template") or ""):
        head += "Время в квадратных скобках — начало раздела в записи (часы:минуты:секунды).\n\n"
    return head


def _clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _author_of(obj, data):
    """Функция «говорящий в стенограмме → как назвать автора цитаты» для свёртки «цитаты по одной».

    Условная метка («Участник N») — по словам метки в текущей стенограмме, как в шапке; говорящий без метки
    (появился после приёма) — по правилу шапки _speaker_display.
    """
    names = data.get("speaker_names")
    names = names if isinstance(names, dict) else {}
    current = {word.get("word_id"): seg.get("speaker")
               for seg in data.get("segments") or [] for word in seg.get("words") or []}
    by_speaker = {}
    for item in obj.get("speaker_labels") or []:
        for wid in item.get("word_ids") or []:
            if wid in current:
                by_speaker.setdefault(current[wid], item.get("label"))

    def author(speaker):
        return by_speaker.get(speaker) or _speaker_display(speaker, names)
    return author


def render(obj, data, update_last=True):
    """-> (markdown, отчёт). Отчёт: что обновилось из стенограммы с прошлой сборки и с момента связывания.

    Свёртка с quote_policy 1 («цитаты по одной», РАР-7, 7.5): автора пишет программа, цитата в «» только
    если ни одно место разбора её не задевает (workflow.quote_blockers), иначе она печатается без кавычек
    с пометкой; в шапке — строка «Цитат: …». Остальные свёртки собираются как раньше.
    """
    rep = {"changed_since_last": [], "changed_since_link": 0, "speaker_changed": [], "missing": []}
    rep["rendered_source_revision"] = data.get("source_revision")
    rep["analysis_revision"] = obj.get("analysis_revision")
    rep["external_publication"] = "not_tracked"
    rep["analysis_needs_review"] = (obj.get("analysis_revision") is None or data.get("source_revision") is None or
                                    obj.get("analysis_revision") != data.get("source_revision") or
                                    not obj.get("recording_id") or obj.get("recording_id") != data.get("recording_id"))
    source_conflict = bool(obj.get("recording_id") and obj.get("recording_id") != data.get("recording_id"))
    if source_conflict:
        rep["source_conflict"] = "конспект относится к другой записи"
    if not obj.get("recording_id"):
        rep["legacy_unverified"] = True

    policy = obj.get("quote_policy") == 1
    stats = {"quoted": 0, "marked": obj.get("unlinked_count", 0) if policy else 0, "unheard": 0}
    if policy:
        import workflow
        rep["author_corrected"] = []
        author_of = _author_of(obj, data)
        starts = {word.get("word_id"): word["s"] for seg in data.get("segments") or []
                  for word in seg.get("words") or [] if _start(word) is not None}

    slots = {item.get("id"): item for item in obj.get("unlinked") or [] if isinstance(item, dict)}

    def sub(m):
        if m.group(1) == "u":
            item = slots.get(m.group(2))
            if item is None or item.get("why") not in UNLINKED_MARKS or not isinstance(item.get("quote"), str):
                return m.group(0)               # неизвестный или негодный слот остаётся как есть
            return f"{item['quote']} ({UNLINKED_MARKS[item['why']]})"
        key = m.group(2)
        q = obj["quotes"].get(key)
        if not q:
            return m.group(0)
        texts = [(None, None) if source_conflict else _id_text(data, p["word_ids"], p.get("exact")) if "word_ids" in p
                 else _legacy_span(data, p["t1"], p["t2"]) for p in q["parts"]]
        if any(t is None for t, _ in texts):
            rep["missing"].append({"id": key, "author": q["author"]})
            if policy:
                stats["marked"] += 1
                return f"{q['author']} (место в стенограмме не найдено)"
            body = q["author"] + " [место в стенограмме не найдено]"
            return (f"{q['speaker']}: " if q.get("speaker") else "") + f"«{body}»"
        cur = " … ".join(t for t, _ in texts)
        cur_spk = max((s for _, s in texts), key=[s for _, s in texts].count)
        changed = _clean(cur) != _clean(q["base"])
        if changed:
            rep["changed_since_link"] += 1
        if _clean(cur) != _clean(q.get("last", q["base"])):
            rep["changed_since_last"].append({"id": key, "was": q.get("last"), "now": cur})
        body = cur if changed else q["author"]
        spk = q.get("speaker")
        if spk and cur_spk != q.get("base_speaker"):          # говорящий сменился в стенограмме
            if cur_spk != q.get("last_speaker"):
                rep["speaker_changed"].append({"id": key, "was": q.get("last_speaker"), "now": cur_spk,
                                               "quote": body[:80]})
            spk = cur_spk
        if update_last:
            q["last"], q["last_speaker"] = cur, cur_spk
        if policy:
            return one_by_one(key, q, body, cur_spk)
        return (f"{spk}: " if spk else "") + f"«{body}»"

    def one_by_one(key, q, body, speaker):
        """Цитата свёртки нового вида: автор от программы; кавычки — если места разбора её не задевают."""
        author = author_of(speaker)
        written = q.get("written_label")
        if written and written != author:
            rep["author_corrected"].append({"id": key, "written": written, "now": author})
        ids = [wid for part in q["parts"] for wid in part.get("word_ids", [])]
        blockers = workflow.quote_blockers(data, ids)
        if not blockers:
            stats["quoted"] += 1
            return f"{author}: «{body}»"
        stats["marked"] += 1
        unheard = all(blocker["reason"] == "unheard" for blocker in blockers)
        stats["unheard"] += unheard
        found = [starts[wid] for wid in ids if wid in starts]
        when = f", {_clock(min(found))}" if found else ""
        return f"{author}: {body} (по распознаванию, {'не прослушано' if unheard else 'не проверено'}{when})"

    body_md = SLOT.sub(sub, obj["template"])         # один проход: вставленный текст не раскрывается повторно
    header = _labels_header(obj, data, rep["analysis_needs_review"])
    if policy and (stats["quoted"] or stats["marked"]):
        header += (f"Цитат: {stats['quoted']} в кавычках, {stats['marked']} без проверки"
                   + (f" (из них не прослушано: {stats['unheard']})" if stats["unheard"] else "") + "\n\n")
    md = header + body_md
    if update_last and not source_conflict:
        obj["assembly_revision"] = data.get("source_revision")
    rep["assembly_revision"] = obj.get("assembly_revision")
    return md, rep


def confirm_analysis(obj, data):
    """Зафиксировать проверенную человеком ревизию; сам вызов не пишет файлы."""
    if not obj.get("recording_id") or obj.get("recording_id") != data.get("recording_id"):
        raise ValueError("невозможно подтвердить анализ другой или неизвестной записи")
    if data.get("source_revision") is None:
        raise ValueError("ревизия источника неизвестна")
    obj["analysis_revision"] = data["source_revision"]


# ---------- файлы ----------
def path_for(json_path, name):
    jp = Path(json_path)
    stem = jp.name[:-len(".transcript.json")] if jp.name.endswith(".transcript.json") else jp.stem
    safe = "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in name).strip() or "конспект"
    return jp.with_name(f"{stem}.{safe}.conspect.json"), jp.with_name(f"{stem}.{safe}.conspect.md")


def save_linked(json_path, name, conspect_md):
    with write_lock(json_path):
        recover_pending(json_path)
        return _save_linked_locked(json_path, name, conspect_md)


def _save_linked_locked(json_path, name, conspect_md):
    import recordings
    import transcript
    data = transcript.load(json_path)
    obj, rep = link(conspect_md, data)
    obj.update(name=name, job_json=str(json_path))
    cj, cm = path_for(json_path, name)
    md, rrep = render(obj, data)
    store, binding = recordings.open_for_document(json_path)
    sqlite_mode = store is not None
    if store is not None:
        store.close()
    write_files(json_path, [(cj, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")),
                            (cm, md.encode("utf-8"))], create=sqlite_mode)
    check = Q.check(md, data)["summary"]
    return {"conspect_json": str(cj), "conspect_md": str(cm), **rep,
            "analysis_needs_review": rrep["analysis_needs_review"],
            "assembly_revision": rrep["assembly_revision"], "analysis_revision": rrep["analysis_revision"],
            "external_publication": rrep["external_publication"], "check": check}


def rerender(json_path, name=None):
    """Пересобирает конспект(ы) записи из текущей стенограммы. -> {название: отчёт}."""
    with write_lock(json_path):
        recover_pending(json_path)
        return _rerender_locked(json_path, name)


def _rerender_locked(json_path, name=None):
    jp = Path(json_path)
    import transcript
    import recordings
    store, binding = recordings.open_for_document(jp)
    review_revision = None
    try:
        if store is not None:
            # Ревизия разбора читается до загрузки стенограммы: если разбор изменится между ними, в свёртке
            # останется старая ревизия и следующая проверка (recordings.refresh_conspects) её пересоберёт.
            review_revision = store.get_current(binding["recording_id"])["review"]["revision"]
        data = transcript.load(jp)
    except BaseException:
        if store is not None:
            store.close()
        raise
    if store is not None:
        try:
            out = {}
            for saved in store.list_conspects(binding["recording_id"]):
                obj = saved["payload"]
                if name and obj.get("name") != name:
                    continue
                md, rep = render(obj, data)
                obj["source_revision"] = data.get("source_revision")
                if obj.get("quote_policy") == 1:
                    obj["assembly_review_revision"] = review_revision
                paths = recordings.document_binding(store, binding["recording_id"])["root"]
                cj, cm = paths / saved["json_relative"], paths / saved["md_relative"]
                write_files(jp, [(cj, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")),
                                 (cm, md.encode("utf-8"))])
                chk = Q.check(md, data)
                out[obj.get("name", saved["name"])] = {
                    "conspect_md": str(cm), **rep, "check": chk["summary"],
                    "check_problems": chk["problems"][:20], "markdown_chars": len(md)}
            return out
        finally:
            store.close()
    stem = jp.name[:-len(".transcript.json")] if jp.name.endswith(".transcript.json") else jp.stem
    out = {}
    for cj in sorted(jp.parent.glob(f"{stem}.*.conspect.json")):
        obj = json.loads(cj.read_text(encoding="utf-8"))
        if name and obj.get("name") != name:
            continue
        md, rep = render(obj, data)
        cm = Path(str(cj)[:-len(".json")] + ".md")
        write_files(jp, [(cj, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")),
                         (cm, md.encode("utf-8"))])
        chk = Q.check(md, data)
        out[obj.get("name", cj.name)] = {"conspect_md": str(cm), **rep, "check": chk["summary"],
                                         "check_problems": chk["problems"][:20], "markdown_chars": len(md)}
    return out
