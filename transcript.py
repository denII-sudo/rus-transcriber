"""Запись расшифровки (.json и .md), переименование спикеров, разбор сомнительных мест.

Два слоя данных: слова модели (segments[].words: текст, время, уверенность, вторая гипотеза) и правки
(edits — журнал: кто, что, было/стало). Правка не стирает слой модели: исходные слова остаются
в поле model у исправленного места и в журнале.
"""
import copy
import hashlib
import inspect
import json
import re
import time
import uuid
from functools import wraps
from pathlib import Path


def recording_locked(func):
    """Сериализовать всю операцию с записью, включая вложенную пересборку."""
    @wraps(func)
    def wrapped(json_path, *args, **kwargs):
        import conspect
        import recordings
        bound = inspect.signature(func).bind_partial(json_path, *args, **kwargs)
        bound.apply_defaults()
        by = bound.arguments.get("by")
        actor_kind = "human" if by is None or by == "human" else ("model" if by == "claude" else "unknown")
        with conspect.write_lock(json_path):
            with recordings.operation_for_document(json_path, actor_kind):
                return func(json_path, *args, **kwargs)
    return wrapped


REASON_RU = {"asr": "модели распознали по-разному", "conf": "модель не уверена в слове",
             "voice": "голоса близки — говорящий под вопросом", "overlap": "говорят одновременно",
             "gender": "род глагола не совпадает с полом говорящего"}


def hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


# Почему вторая или третья модель не считалась (quality.asr2_skipped / asr3_skipped, worker.run_words)
SKIPPED_RU = {
    ("asr2_skipped", "same_as_primary"): "Вторая модель ({model}) совпадает с основной и не считалась: "
                                         "сверки и голосования моделей нет",
    ("asr2_skipped", "no_word_times"): "Вторая модель ({model}) не даёт времён слов и вторым номером не "
                                       "используется: сверки и голосования моделей нет",
    ("asr3_skipped", "same_as_primary"): "Третья модель ({model}) совпадает с основной (та же модель, движок "
                                         "и точность) и не считалась: голосования нет, сверены основная "
                                         "и вторая модели",
    ("asr3_skipped", "same_as_second"): "Третья модель ({model}) совпадает со второй (та же модель, движок "
                                        "и точность) и не считалась: голосования нет, сверены основная "
                                        "и вторая модели",
}

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


def is_whisper_model(model):
    """Модель — Whisper: в имени есть «whisper» (без учёта регистра)."""
    return "whisper" in str(model or "").lower()


# Возможности моделей распознавания — одно место для всего конвейера (worker.run_words, подписи
# стенограммы, выбор движков в backends.make_asr). Логика второго и третьего места спрашивает их,
# а не имя модели. Ключ — имя модели, как в config.json (asr_model, asr_second_model, asr_third_model).
#   label       — короткое название для людей (статус обработки, варианты в стенограмме, заметки);
#   word_times  — модель отдаёт времена слов: True, False или None (неизвестно, проверит сам проход);
#                 без них модель не может стоять вторым номером — голосование ставит её слова на место
#                 по времени и исключает сбойные отрезки по времени;
#   punctuation — текст с пунктуацией и регистром (сравнение слов от них не зависит: words.norm_word);
#   whisper     — семейство Whisper: окно 30 с (свои куски до 28 с, asr_third_long_chunks), чистка
#                 «субтитров» и зацикливания (words.clean_whisper), движки OpenVINO Whisper (whisper-ov);
#   ct2         — модель есть у движка faster-whisper (whisper-ct2: он грузит только large-v3-turbo);
#   torch       — модель есть в пакете gigaam (запасной движок PyTorch в make_asr).
ASR_MODELS = {
    "v3_e2e_rnnt": {"label": "RNN-T", "word_times": True, "punctuation": True, "torch": True},
    "v3_rnnt": {"label": "RNN-T", "word_times": True, "punctuation": False, "torch": True},
    "v3_e2e_ctc": {"label": "CTC", "word_times": True, "punctuation": True, "torch": True},
    "v3_ctc": {"label": "CTC", "word_times": True, "punctuation": False, "torch": True},
    # GigaAM Multilingual: CTC, текст строчными без пунктуации
    "multilingual_ctc": {"label": "Multilingual", "word_times": True, "punctuation": False, "torch": True},
    "multilingual_large_ctc": {"label": "Multilingual large", "word_times": True, "punctuation": False,
                               "torch": True},
}


def asr_model_caps(model):
    """Возможности модели распознавания (см. ASR_MODELS). Whisper узнаётся по имени; неизвестная
    модель — по имени, времена слов и пунктуация неизвестны, запасной движок PyTorch не исключён."""
    name = str(model or "")
    if is_whisper_model(name):
        return {"label": "Whisper", "word_times": False, "punctuation": True, "whisper": True,
                "ct2": name.lower().endswith("whisper-large-v3-turbo"), "torch": False}
    known = ASR_MODELS.get(name)
    if known:
        return {**known, "whisper": False, "ct2": False}
    return {"label": name, "word_times": None, "punctuation": None, "whisper": False, "ct2": False,
            "torch": True}


def model_label(model, backend=""):
    """Короткое название модели для людей: Whisper, CTC, Multilingual large или имя модели.
    Одно правило для подписи этапа обработки (worker) и вариантов в стенограмме. Движок Whisper
    (backend «whisper-…») — всегда Whisper: в старых результатах он подменял другую модель."""
    if str(backend or "").startswith("whisper-"):
        return "Whisper"
    return asr_model_caps(model)["label"]


third_model_label = model_label        # прежнее имя (PR №82): подпись третьей модели


def _third_name(data):
    """Подпись варианта третьей модели в стенограмме по timing результата; нет сведений — «Whisper»."""
    timing = data.get("timing") if isinstance(data, dict) else None
    timing = timing if isinstance(timing, dict) else {}
    model = timing.get("asr3_model")
    if not model:
        return "Whisper"
    backend = str(timing.get("asr3_device") or "").split(":")[0]
    return model_label(model, backend) or "Whisper"


def _second_name(data):
    """Подпись варианта второй модели (alt_ctc) по timing результата; нет сведений — «CTC»."""
    timing = data.get("timing") if isinstance(data, dict) else None
    timing = timing if isinstance(timing, dict) else {}
    model = timing.get("asr2_model")
    return (model_label(model) or "CTC") if model else "CTC"


def _item_line(it, third_name="Whisper", second_name="CTC"):
    why = ", ".join(REASON_RU.get(r, r) for r in it.get("reasons", []))
    s = f"- {hms(it['t'])} {it['speaker']}: «{it['text']}» — {why}"
    if it.get("alt_ctc"):
        s += f"; {second_name}: «{it['alt_ctc']}»"
    if it.get("alt_whisper"):
        s += f"; {third_name}: «{it['alt_whisper']}»"
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
    if q0.get("asr_incomplete_chunks"):
        lines.append(f"_Внимание: технически не распознано фрагментов: {q0['asr_incomplete_chunks']}; "
                     "завершение задачи не означает полноту стенограммы._\n")
    if q0.get("vote3_skipped"):
        lines.append("_Голосование трёх моделей отключено из-за неполного или неизвестного результата ASR._\n")
    if q0.get("vote3_partial"):
        p = q0["vote3_partial"]
        lines.append(f"_Голосование трёх моделей пропущено на отрезках с неполным результатом ASR: "
                     f"{p.get('spans')} (около {str(p.get('seconds')).replace('.', ',')} с); "
                     "там сверены только две модели._\n")
    for key in ("asr2_skipped", "asr3_skipped"):
        skipped = q0.get(key)
        note = SKIPPED_RU.get((key, skipped.get("reason"))) if isinstance(skipped, dict) else None
        if note:
            lines.append("_" + note.format(model=model_label(skipped.get("model"))) + "._\n")
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
        third_name, second_name = _third_name(data), _second_name(data)
        if data.get("review_done"):
            human = [i for i in data["review"] if i.get("status") == "human"]
            if human:
                lines += ["## Проверить человеку", ""] + [_item_line(i, third_name, second_name)
                                                         for i in human] + [""]
            if data.get("edits"):
                lines += ["## Правки после распознавания", ""]
                for e in data["edits"]:
                    b, a = e["before"], e["after"]
                    ch = []
                    if b.get("text") != a.get("text"):
                        ch.append(f"«{b.get('text')}» → «{a.get('text')}»")
                    if b.get("speaker") != a.get("speaker"):
                        ch.append(f"{b.get('speaker')} → {a.get('speaker')}")
                    who = e.get("by", "?")
                    if e.get("via") == "edit_span":
                        who = ("человек" if who == "human" else "агент") + ", правка отрезка"
                    lines.append(f"- {hms(e['t'])} ({who}): " + "; ".join(ch)
                                 + (f" — {e['note']}" if e.get("note") else ""))
                lines.append("")
        else:
            lines += ["## Сомнительные места", ""] + [_item_line(i, third_name, second_name)
                                                      for i in data["review"]] + [""]
    return "\n".join(lines)


def save(data, json_path, md_path):
    import conspect
    with conspect.write_lock(json_path):
        return _save_locked(data, json_path, md_path)


def ensure_new_result(json_path, md_path):
    """Под блокировкой проверить, что оба пути результата свободны."""
    import conspect
    with conspect.write_lock(json_path):
        if Path(json_path).exists() or Path(md_path).exists():
            raise FileExistsError("результат уже существует; выберите другую метку (tag)")


def save_new(data, json_path, md_path):
    """Создать результат worker, не заменяя существующие файлы."""
    import conspect
    with conspect.write_lock(json_path):
        ensure_new_result(json_path, md_path)
        return _save_locked(data, json_path, md_path)


def _listening_fingerprint(words):
    content = [[w.get("word_id"), w.get("w"), w.get("s"), w.get("e")] for w in words]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _interval_words(data, start, end):
    return [w for seg in data.get("segments") or [] for w in seg.get("words") or []
            if w.get("s", end) < end and w.get("e", w.get("s", end)) > start]


def _save_locked(data, json_path, md_path):
    # Идентификаторы не зависят от времени слова или его позиции в отображении.
    # Старые файлы получают их при первом сохранении; дубликаты после разрезания
    # реплик получают новые ID. Это не межпроцессная блокировка.
    data.setdefault("recording_id", uuid.uuid4().hex)
    for field, objects in (("segment_id", data.get("segments") or []),
                           ("word_id", [w for seg in data.get("segments") or []
                                        for w in seg.get("words") or []])):
        seen = set()
        for obj in objects:
            ident = obj.get(field)
            if not isinstance(ident, str) or not ident or ident in seen:
                ident = uuid.uuid4().hex
                obj[field] = ident
            seen.add(ident)
    for item in data.get("review") or []:
        listening = item.get("listening")
        if item.get("asr_problem") and not listening and item.get("status") == "resolved":
            item["status"] = "open"
            data["review_done"] = False
        if not listening or not item.get("gap"):
            continue
        start, end = item["gap"]
        current = _interval_words(data, start, end)
        if _listening_fingerprint(current) != listening.get("fingerprint"):
            item.pop("listening", None)
            item["status"] = "open"
            reminder = "Слова интервала изменились после прослушивания; нужна новая проверка."
            note = item.get("note") or ""
            if reminder not in note:
                item["note"] = f"{note.rstrip()}\n\n{reminder}".strip()
            data["review_done"] = False
    import recordings
    draft = recordings.answer_draft_for(json_path)
    if draft is not None:
        draft["data"] = copy.deepcopy(data)
        return {"draft": True}
    store, binding = recordings.open_for_document(json_path)
    if store is not None:
        try:
            recording_id = data.get("recording_id")
            if recording_id != binding["recording_id"]:
                raise RuntimeError("путь и recording_id не совпадают")
            expected = {"source_revision": data.get("source_revision"),
                        "review_revision": data.get("review_revision")}
            result = recordings.commit_recording_data(
                store, recording_id, data, expected)
            data.update(source_revision=result["source_revision"],
                        review_revision=result["review_revision"],
                        document_revision=result["document_revision"],
                        document_sync=result["document_sync"])
            if result["document_sync"] != "synced":
                completed = recordings.materialize_recording(store, recording_id)
                if (recording_id, result["document_revision"]) not in completed:
                    raise RuntimeError("SQLite commit succeeded but document materialization is pending")
            data["document_sync"] = "synced"
            return result
        finally:
            store.close()
    data["source_revision"] = data.get("source_revision", 0) + 1
    import conspect
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    md_bytes = render_md(data).encode("utf-8")
    md_target = Path(md_path)
    try:
        old_md = md_target.read_bytes()
    except FileNotFoundError:
        old_md = None
    try:
        conspect._replace_bytes(md_target, md_bytes)
        conspect._replace_bytes(Path(json_path), json_bytes)
    except Exception:
        try:
            if old_md is None:
                md_target.unlink(missing_ok=True)
            else:
                conspect._replace_bytes(md_target, old_md)
        except OSError:
            pass
        raise


def load(json_path):
    import conspect
    import recordings
    draft = recordings.answer_draft_for(json_path)
    if draft is not None:
        return copy.deepcopy(draft["data"])
    store, binding = recordings.open_for_document(json_path)
    if store is not None:
        try:
            return recordings.load_recording(store, binding["recording_id"])
        finally:
            store.close()
    with conspect.write_lock(json_path):
        return json.loads(Path(json_path).read_text(encoding="utf-8"))


def _undo_fingerprint(data):
    state = {k: data.get(k) for k in ("segments", "review", "edits")}
    if state["edits"] is not None:
        state["edits"] = [e for e in state["edits"] if not e.get("reverted")]
    raw = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def answer_snapshot(before, after):
    """Сохранить только структурные участки, изменённые ответом."""
    from difflib import SequenceMatcher

    def patches(key):
        old, new = before.get(key) or [], after.get(key) or []
        encode = lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        matcher = SequenceMatcher(None, [encode(x) for x in old], [encode(x) for x in new], autojunk=False)
        return [{"start": j1, "remove": j2 - j1, "restore": copy.deepcopy(old[i1:i2])}
                for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]

    return {"format": 2, "segment_patches": patches("segments"), "review_patches": patches("review"),
            "edits_before": len(before.get("edits") or []),
            "after": _undo_fingerprint(after)}


def _restore_patches(rows, patches):
    if not isinstance(patches, list):
        raise ValueError("неверный формат снимка")
    end = 0
    for p in patches:
        if not isinstance(p, dict):
            raise ValueError("неверный участок снимка")
        start, remove, restore = p.get("start"), p.get("remove"), p.get("restore")
        if type(start) is not int or type(remove) is not int or not isinstance(restore, list) \
                or start < end or remove < 0 or start + remove > len(rows):
            raise ValueError("неверные границы снимка")
        end = start + remove
    result = copy.deepcopy(rows)
    for p in reversed(patches):
        result[p["start"]:p["start"] + p["remove"]] = copy.deepcopy(p["restore"])
    return result


@recording_locked
def undo_answer(json_path, md_path, aid, snapshot):
    """Отменяет ответ только если состояние после него осталось неизменным."""
    data = load(json_path)
    if not snapshot or snapshot.get("after") != _undo_fingerprint(data):
        return {"id": aid, "conflict": "нет снимка или после ответа появились изменения"}
    edits = data.get("edits") or []
    n = snapshot.get("edits_before")
    if not isinstance(n, int) or not 0 <= n <= len(edits):
        return {"id": aid, "conflict": "повреждён снимок журнала"}
    try:
        if snapshot.get("format") == 2:
            segments = _restore_patches(data.get("segments") or [], snapshot.get("segment_patches"))
            review = _restore_patches(data.get("review") or [], snapshot.get("review_patches"))
        elif "format" not in snapshot and isinstance(snapshot.get("segments"), list) \
                and isinstance(snapshot.get("review"), list):
            segments, review = copy.deepcopy(snapshot["segments"]), copy.deepcopy(snapshot["review"])
        else:
            raise ValueError("неизвестный формат снимка")
    except (ValueError, TypeError):
        return {"id": aid, "conflict": "повреждён структурный снимок"}
    data["segments"], data["review"] = segments, review
    for edit in edits[n:]:
        if edit.get("by") == "human":
            edit["original_by"] = edit["by"]
            edit["by"] = "reverted"
        edit["reverted"] = True
    for item in data["review"]:
        if item.get("id") == aid:
            item["status"] = "open"
            item.pop("note", None)
    # Подсказки словаря, возникшие из отменённого ответа, больше не предлагаем.
    data["vocab_suggestions"] = [v for v in data.get("vocab_suggestions") or []
                                 if v.get("card") != aid]
    data["review_done"] = not any(i.get("status", "open") == "open" for i in data["review"])
    save(data, json_path, md_path)
    return {"id": aid, "reverted": True, "conspects": _refresh_conspects(json_path)}


@recording_locked
def rename(json_path, md_path, mapping, *, expected_versions=None):
    """mapping: {"Спикер 4": "Анна", ...}. Ключом может быть исходная метка или уже данное имя.
    Исходная метка сохраняется в поле label каждого сегмента, поэтому переименовывать можно повторно."""
    data = load(json_path)
    if expected_versions is not None and (
            data.get("source_revision") != expected_versions.get("source_revision") or
            data.get("review_revision") != expected_versions.get("review_revision")):
        raise ValueError("стенограмма изменилась после запроса подтверждения")
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
    import recordings
    if recordings.answer_draft_for(json_path) is not None:
        return {"deferred": True}
    try:
        import conspect
        return conspect.rerender(json_path)
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}


# ---------- разбор сомнительных мест ----------
def _find_span(data, it):
    """Находит место по времени слов (индексы сдвигаются после правок)."""
    if "word_ids" in it:
        target = it["word_ids"]
        if not isinstance(target, list) or not target or not all(isinstance(x, str) and x for x in target) \
                or len(target) != len(set(target)):
            return None
        for si, seg in enumerate(data["segments"]):
            ws = seg.get("words") or []
            ids = [w.get("word_id") for w in ws]
            for start in range(len(ids) - len(target) + 1):
                if ids[start:start + len(target)] == target:
                    return si, start, start + len(target)
        return None
    t1, t2 = it.get("t1", it["t"]), it.get("t2", it["t"])
    found = []
    for si, seg in enumerate(data["segments"]):
        ws = seg.get("words") or []
        idx = [k for k, w in enumerate(ws) if t1 - 0.01 <= w["s"] <= t2 + 0.01]
        if idx:
            found.append((si, idx[0], idx[-1] + 1))
    return next((loc for loc in found if data["segments"][loc[0]].get("speaker") == it.get("speaker")),
                found[0] if found else None)


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
                                      "conf_min", "note", "status", "asr_problem", "listening") if it.get(k) is not None}
        if row.get("asr_problem"):
            models = ((data.get("quality") or {}).get("asr_run") or {}).get("models") or {}
            problem = dict(row["asr_problem"])
            problem["model_id"] = (models.get(problem.get("model")) or {}).get("id")
            problem["words_from_models"] = [(models.get(key) or {}).get("id") or key
                                             for key in problem.get("words_from") or []]
            row["asr_problem"] = problem
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
        # какая модель дала варианты: ключи alt_ctc и alt_whisper — прежние, модели — любые
        res["alt_models"] = {"alt_ctc": _second_name(data), "alt_whisper": _third_name(data)}
    return res


def edit_span(json_path, md_path, t1, t2, text, speaker=None, note=None, by=None, resolve=True):
    """Правка произвольного отрезка стенограммы: слова, начинающиеся в [t1, t2] с, заменяются текстом text
    (по словам: совпадающие сохраняют время и прежние отметки), speaker — подписать отрезок этим говорящим
    (реплика делится). Пустой text — удалить слова отрезка. На отрезке без слов text вставляется (тогда
    speaker обязателен). by — автор правки, обязателен: "claude" или "human"; без него правка не принимается.
    resolve — места разбора, целиком лежащие в отрезке, считаются решёнными. -> (data, отчёт)"""
    check_edit_span_author(by)
    return _edit_span_locked(json_path, md_path, t1, t2, text, speaker, note, by, resolve)


@recording_locked
def _edit_span_locked(json_path, md_path, t1, t2, text, speaker=None, note=None, by="claude", resolve=True):
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
                if by != "human" and it.get("status") == "human":
                    raise ValueError(f"{HELD_FOR_HUMAN} ({it['id']}): агент его не меняет и не закрывает")
                closed[it["id"]] = card
    who = "человек" if by == "human" else "агент"
    data2, rep = apply_word_edits(json_path, md_path, [e], by=by, closed=closed, via="edit_span",
                                  log_note="правка отрезка" + (f": {note}" if note else ""),
                                  close_note=f"решено правкой отрезка {card} ({who})", close_held=by == "human")
    if card in rep["failed"]:
        reason = rep["failed"][card]
        raise ValueError(reason + (SPAN_REOPEN_HINT if "ответом человека на карточку" in reason else ""))
    c = rep["cards"].get(card, {})
    return data2, {"card": card, "before": c.get("before"), "after": c.get("after"), "resolved": sorted(closed),
                   "conspects": rep.get("conspects")}


@recording_locked
def reopen(json_path, md_path, ids, by="unknown"):
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
        it.pop("listening", None)
        done.append(rid)
    data["review_done"] = not any(i.get("status", "open") == "open" for i in data.get("review") or [])
    save(data, json_path, md_path)
    return data, {"reopened": done, "unknown": unknown,
                  "open_left": sum(1 for i in data.get("review") or [] if i.get("status", "open") == "open")}


def _wkey(s):
    return re.sub(r"[^\w-]", "", s.lower().replace("ё", "е"))


def _mark_provenance(word, dimension, by, edit, via=None):
    provenance = copy.deepcopy(word.get("provenance") or {})
    provenance[dimension] = {"by": by, "edit": edit, **({"via": via} if via else {})}
    word["provenance"] = provenance


HELD_FOR_HUMAN = "место оставлено человеку"
EDIT_SPAN_AUTHOR_REQUIRED = ('edit_span: укажите автора правки by — "claude" (правка агента, в том числе по поручению '
                             'человека) или "human" (текст продиктовал человек, только командной строкой); '
                             'без автора правка не принимается')
SPAN_REOPEN_HINT = "; чтобы решить заново — reopen_review (rt.py reopen)"


def check_edit_span_author(by):
    """Автор правки отрезка обязателен: только "claude" или "human". Проверяется до любой записи и до
    `recording_locked`, который считает пустого автора человеком."""
    if by not in ("claude", "human"):
        raise ValueError(EDIT_SPAN_AUTHOR_REQUIRED)


def human_touched(word):
    """Человек тронул слово: он автор слова, его текста или его говорящего (by или provenance).
    Единственное место этого правила; агент такое слово не меняет. Другой вопрос — у `prior_edit`
    (чья последняя правка), и другое правило — у `voices` (подтверждённый говорящий для голосов)."""
    provenance = word.get("provenance")
    return word.get("by") == "human" or isinstance(provenance, dict) and any(
        v.get("by") == "human" for v in provenance.values() if isinstance(v, dict))


def prior_edit(word, data=None):
    """Чья прежняя правка у слова и каким способом. -> None, если слово не правили, иначе
    {"id": правка, "by": автор или None, "via": "edit_span" | "card"}.
    Автор: `by` слова → `provenance` (текст, затем говорящий) → последняя запись журнала `edits` с тем же
    id → не определён (None; так бывает в старых записях). Способ: правка отрезка (`span@…`) или ответ
    на карточку. Одна функция для проверки и применения ответов и для правил правки отрезка."""
    edit_id = word.get("edited")
    if not edit_id:
        return None
    by = word.get("by")
    if not by:
        provenance = word.get("provenance") or {}
        by = next((v["by"] for v in (provenance.get("text"), provenance.get("speaker"))
                   if isinstance(v, dict) and v.get("by")), None)
    if not by and data:
        by = next((e["by"] for e in reversed(data.get("edits") or [])
                   if e.get("id") == edit_id and e.get("by")), None)
    if by == "unknown":
        by = None
    marked = next((v.get("via") for v in (word.get("provenance") or {}).values()
                   if isinstance(v, dict) and v.get("edit") == edit_id and v.get("via")), None)
    via = marked or ("edit_span" if str(edit_id).startswith("span@") else "card")
    return {"id": edit_id, "by": by, "via": via}


def overwrite_refusal(word, data, writer, card):
    """Можно ли правке `card` автора `writer` перезаписать слово. -> None (можно) или причина отказа,
    которая называет автора и способ прежней правки. Ответ человека главнее правки агента и старой
    правки без автора; поверх ответа другой карточки и правки отрезка человека — конфликт: обе правки
    человеческие, выбирает человек. Агент не перезаписывает ничью прежнюю правку."""
    prior = prior_edit(word, data)
    if prior is None or prior["id"] == card:
        return None
    if writer == "human" and prior["by"] != "human":
        return None
    span = prior["via"] == "edit_span"
    if prior["by"] == "human":
        how = "правкой отрезка человека" if span else "ответом человека на карточку"
    elif prior["by"]:
        how = "правкой отрезка агента" if span else "правкой агента по карточке"
    else:
        how = "прежней правкой без автора"
    return f"слово «{word.get('w')}» уже исправлено {how} ({prior['id']})"


def _held_places(data):
    """Места, оставленные человеку (статус human): (id() их слов, их отрезки). Агент их не меняет и не
    закрывает, пока нет правила о поручении такого места ИИ."""
    words, spans = set(), []
    all_words = [w for seg in data.get("segments") or [] for w in seg.get("words") or []]
    for it in data.get("review") or []:
        if it.get("status") != "human":
            continue
        t1, t2 = it.get("t1", it.get("t")), it.get("t2", it.get("t"))
        if t1 is not None and t2 is not None:
            spans.append((t1 - 0.01, t2 + 0.01))
        ids = it.get("word_ids")
        if ids:
            words.update(id(w) for w in all_words if w.get("word_id") in ids)
        elif t1 is not None and t2 is not None:
            words.update(id(w) for w in all_words if t1 - 0.01 <= w["s"] <= t2 + 0.01)
    return words, spans


def _diff_words(old, new_text, item_id, refuse=None):
    """Правка текста места по словам: меняются только отличающиеся слова, у остальных сохраняются время,
    уверенность и прежние правки (на время слов опираются ссылки конспекта). Слово с прежней правкой,
    которую нельзя перезаписать (refuse(слово) -> причина или None; по умолчанию — любая другая
    правка), не перезаписывается — оно возвращается в списке конфликтов.
    -> (новые слова, [id правок, которые сохранены вместо этой], [причины])."""
    import difflib
    if refuse is None:
        def refuse(w):
            return "слово уже исправлено прежней правкой" if w.get("edited") and w["edited"] != item_id else None
    new = new_text.split()
    sm = difflib.SequenceMatcher(None, [_wkey(w["w"]) for w in old], [_wkey(x) for x in new], autojunk=False)
    out, conflicts, reasons = [], [], []

    def mk(text, src, s, e):
        result = {"w": text, "s": s, "e": e, "c": 1.0, "edited": item_id,
                  "model": " ".join(w.get("model", w.get("raw", w["w"])) for w in src) if src else ""}
        if len(src) == 1 and src[0].get("word_id"):
            result["word_id"] = src[0]["word_id"]
        if len(src) == 1 and src[0].get("provenance"):
            result["provenance"] = copy.deepcopy(src[0]["provenance"])
        return result

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        src = old[i1:i2]
        if op == "equal":
            for w, x in zip(src, new[j1:j2]):          # пунктуация и регистр — из правки
                out.append(w if w["w"] == x else {**w, "w": x, "edited": w.get("edited", item_id)})
            continue
        refused = [(w["edited"], why) for w in src for why in [refuse(w)] if why]
        if refused:
            conflicts += [edit for edit, _ in refused]
            reasons += [why for _, why in refused if why not in reasons]
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
    return out, sorted(set(conflicts)), reasons


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


@recording_locked
def apply_word_edits(json_path, md_path, edits, by="human", closed=None, answers=None, log_note=None, dry_run=False,
                     via=None, close_note=None, close_held=False):
    """Правки по словам, а не по местам (свободные ответы человека, freeform.py; правка отрезка edit_span).
    edits: [{"op": "relabel", "t", "speaker"} | {"op": "replace", "t", "text"} |
            {"op": "insert", "s", "e", "text", "speaker", "after_t"?, "before_t"?} |
            {"op": "place_text", "t1", "t2", "text", "speaker"?} | {"op": "delete", "t", "w"} |
            {"op": "overlap_move", "t", "w", "after_t"} | {"op": "propose_delete", "t", "w"}, ... + "card", "note"?]
    При наличии word_id слово ищется по нему; старые временные ссылки допустимы только при однозначности.
    Вставка ставится сразу за after_id/after_t (или перед before_id/before_t), в порядке
    ответа, а не по оценочному времени. overlap_move — чужое слово внутри реплики человека: переносится после
    реплики, помечается как одновременная речь. Правки одной карточки применяются целиком или никак: если
    какое-то слово не найдено или у вставки нет говорящего — карточка не применяется и идёт человеку с причиной.
    Реплика делится там, где внутри неё меняется говорящий. closed — {id места: карточка}, которые
    ответ человека покрыл (решено через другую карточку). answers — {карточка: текст ответа} для журнала.
    via — способ правки ("edit_span"): пишется в журнал и в отметки слов, и тогда помечаются только изменённые
    слова. close_note — заметка закрытых мест (по умолчанию «решено человеком (через карточку)»); close_held —
    закрывать и места, оставленные человеку. -> (data, отчёт)"""
    data = load(json_path)
    original = copy.deepcopy(data)
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

    def find(t, text=None, strict=False, word_id=None):
        if word_id is not None:
            ks = [k for k, f in enumerate(flat) if f["w"].get("word_id") == word_id]
        else:
            ks = by_t.get(round(t, 2)) or [] if t is not None else []
            if len(ks) > 1:
                return None
        if text is not None:
            exact = [k for k in ks if flat[k]["w"]["w"] == text]
            ks = exact if strict or exact else ks
        return ks[0] if len(ks) == 1 else None

    def place_indices(e):
        ids = e.get("word_ids")
        if ids is not None:
            if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
                return None
            indexes = [find(None, word_id=wid) for wid in ids]
            if any(k is None for k in indexes) or indexes != list(range(indexes[0], indexes[0] + len(indexes))):
                return None
            return indexes
        indexes = [k for k, f in enumerate(flat) if e["t1"] - 0.01 <= f["w"]["s"] <= e["t2"] + 0.01]
        if any(len(by_t.get(round(flat[k]["w"]["s"], 2), [])) != 1 for k in indexes):
            return None
        return indexes

    log = data.setdefault("edits", [])
    pairs = {}
    snap = [(f["w"]["s"], f["w"]["w"], f["spk"]) for f in flat]
    rep = {"relabel": 0, "replace": 0, "insert": 0, "delete": 0, "overlap": 0, "not_found": [],
           "proposed_delete": [], "failed": {}, "cards": {}}
    groups = {}
    for e in edits:
        groups.setdefault(e.get("card", "?"), []).append(e)
    held_words, held_spans = _held_places(data) if by != "human" else (set(), [])

    def refuse(e):
        return lambda w: overwrite_refusal(w, data, by, e.get("card", "?"))

    def check(eds):
        """Причина, по которой правки карточки нельзя применить целиком, или None."""
        def held(k):
            return k is not None and id(flat[k]["w"]) in held_words

        def held_time(t):
            return t is not None and any(a <= t <= b for a, b in held_spans)

        removed = set()
        for e in eds:
            if e["op"] == "delete":
                k = find(e.get("t"), e.get("w"), strict=True, word_id=e.get("word_id"))
                if k is not None:
                    removed.add(k)
            elif e["op"] == "place_text":
                removed.update(place_indices(e) or [])

        for e in eds:
            op = e["op"]
            if op == "propose_delete":
                continue
            if op == "insert":
                if not e.get("speaker"):
                    return f"не удалось определить говорящего вставки «{e.get('text')}»"
                for prefix in ("after", "before"):
                    anchor_id, anchor_t = e.get(prefix + "_id"), e.get(prefix + "_t")
                    if anchor_id is not None or anchor_t is not None:
                        anchor = find(anchor_t, word_id=anchor_id)
                        if anchor is None or anchor in removed:
                            return "опора вставки отсутствует, удаляется или неоднозначна"
                        if held(anchor):
                            return HELD_FOR_HUMAN
                if held_time(e.get("s")):
                    return HELD_FOR_HUMAN
                continue
            if op == "place_text":
                ks = place_indices(e)
                if ks is None:
                    return "слова места отсутствуют, неоднозначны или изменили порядок"
                current = [flat[k]["w"] for k in ks]
                if not current and not e.get("speaker"):
                    return f"на отрезке {e['t1']}–{e['t2']} с нет слов, а говорящий не указан"
                if by != "human" and any(human_touched(flat[k]["w"]) for k in ks):
                    return "на отрезке есть подтверждённые человеком слова"
                if any(held(k) for k in ks) or (not ks and any(
                        a <= e["t2"] + 0.01 and e["t1"] - 0.01 <= b for a, b in held_spans)):
                    return HELD_FOR_HUMAN
                why = current and _diff_words(current, e["text"], e.get("card", "?"), refuse(e))[2]
                if why:
                    return "на отрезке есть конфликтующие прежние правки: " + "; ".join(why)
                continue
            k = find(e["t"], e.get("w"), strict=True, word_id=e.get("word_id"))
            if k is None:
                return f"слово «{e.get('w')}» ({e['t']} с) не найдено — стенограмма изменилась"
            if by != "human" and human_touched(flat[k]["w"]):
                return "слово подтверждено человеком"
            if held(k):
                return HELD_FOR_HUMAN
            if op == "overlap_move" and (e.get("after_id") is not None or e.get("after_t") is not None):
                anchor = find(e.get("after_t"), word_id=e.get("after_id"))
                if anchor is None or anchor in removed or anchor == k:
                    return "опора переноса отсутствует, удаляется или неоднозначна"
        return None

    def put_after(t_anchor, card, item, anchor_id=None):
        """Позиция сразу за словом t_anchor и за уже вставленными этой карточкой словами после него."""
        k = find(t_anchor, word_id=anchor_id) if t_anchor is not None or anchor_id is not None else None
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
                ks = place_indices(e)
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
                new_ws, _, _ = _diff_words(old, e["text"], card, refuse(e))
                old_text = {w.get("word_id"): w["w"] for w in old if w.get("word_id")}
                speaker_changed = bool(e.get("speaker") and any(flat[k]["spk"] != e["speaker"] for k in ks))
                for w in new_ws:
                    text_changed = not w.get("word_id") or old_text.get(w["word_id"]) != w["w"]
                    if via and not (text_changed or speaker_changed):
                        continue                  # правка отрезка: неизменённое слово сохраняет прежние отметки
                    w["edited"], w["by"] = card, by
                    if text_changed:
                        _mark_provenance(w, "text", by, card, via)
                    if speaker_changed:
                        _mark_provenance(w, "speaker", by, card, via)
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
            k = find(e["t"], e.get("w"), word_id=e.get("word_id"))
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
                    _mark_provenance(f["w"], "speaker", by, card, via)
                    rep["relabel"] += 1
            elif op == "replace":
                if f["w"]["w"] != e["text"]:
                    before_words.append(f["w"]["w"])
                    f["w"].setdefault("model", f["w"].get("raw", f["w"]["w"]))
                    f["w"]["w"] = e["text"]
                    f["w"]["by"], f["w"]["edited"], f["w"]["c"] = by, card, 1.0
                    _mark_provenance(f["w"], "text", by, card, via)
                    if e.get("note"):
                        f["w"]["typo"] = e["note"]
                    after_words.append(e["text"])
                    rep["replace"] += 1
        # вставки: сразу за опорным словом ответа (в порядке ответа), иначе перед следующим, иначе по времени
        for e in sorted(inserts, key=lambda e: e["s"]):
            w = {"w": e["text"], "s": e["s"], "e": max(e["s"], e.get("e", e["s"])), "c": 1.0, "by": by,
                 "edited": card, "inserted": True}
            for dimension in ("text", "speaker", "order"):
                _mark_provenance(w, dimension, by, card, via)
            item = {"w": w, "spk": e["speaker"], "seg": -1}
            pos = put_after(e.get("after_t"), card, item, e.get("after_id"))
            if pos is None and (e.get("before_t") is not None or e.get("before_id") is not None):
                pos = find(e.get("before_t"), word_id=e.get("before_id"))
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
            k = find(e["t"], e.get("w"), word_id=e.get("word_id"))
            f = flat.pop(k)
            reindex()
            f["w"]["overlap"] = True
            f["w"]["by"] = f["w"].get("by") or "model"
            _mark_provenance(f["w"], "order", by, card, via)
            if put_after(e.get("after_t"), card, f, e.get("after_id")) is None:
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
        log.append({"id": card, "t": rg["t1"], "by": by, "kind": "words", **({"via": via} if via else {}),
                    "before": {"text": bt, "speaker": who(brows)}, "after": {"text": at, "speaker": who(arows)},
                    "answer": answers.get(card), "note": log_note or "ответ человека в карточке",
                    "pairs": sorted({tuple(p) for p in pairs.get(card, [])})})
        rg.update(before=bt, after=at)
    # места: закрытые ответом на другую карточку и сами карточки
    items = {i["id"]: i for i in data.get("review") or []}
    for rid, card_via in (closed or {}).items():
        if any(card_via == c for c in rep["failed"]):
            continue
        it = items.get(rid)
        if it and (close_held or it.get("status", "open") != "human"):
            it["status"], it["note"] = "resolved", close_note or f"решено человеком (через {card_via})"
    for card in groups:
        it = items.get(card)
        if not it:
            continue
        if card in rep["failed"]:
            it["status"] = "human"
            why = f"ответ не применён: {rep['failed'][card]}" + (f": «{answers[card]}»" if answers.get(card) else "")
            if why not in (it.get("note") or ""):        # повтор той же неудачи заметку не удлиняет
                it["note"] = (it.get("note") + "; " if it.get("note") else "") + why
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
    if not dry_run and data != original:     # без изменений — без новой ревизии и перезаписи файлов
        save(data, json_path, md_path)
    rep["open_left"] = sum(1 for i in data.get("review") or [] if i.get("status", "open") == "open")
    rep["conspects"] = None if dry_run else _refresh_conspects(json_path)
    return data, rep


@recording_locked
def apply_review(json_path, md_path, decisions, by="claude"):
    """decisions: [{"id": "r5", "speaker"?: имя, "text"?: новый текст места, "status"?: "resolved"|"human",
    "note"?: str}]. Без speaker и text — место принято как есть. status "human" — оставить человеку.
    Возвращает (data, отчёт)."""
    data = load(json_path)
    original = copy.deepcopy(data)
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
            # Слова не получают отметок человека. Место, уже оставленное человеку, не меняется.
            # if_open («Оставить человеку» на карточке): решённое место тоже не меняется — повтор после
            # того, как место закрыли, ничего не пишет. Без него (программа отправляет непонятый ответ
            # человеку) решённое место снова уходит человеку, как раньше.
            status = it.get("status", "open")
            if status == "human":
                # Комментарий человека из нового ответа дописывается; повтор того же ничего не пишет.
                comment = d.get("comment")
                if comment and d.get("note") and comment not in (it.get("note") or ""):
                    it["note"] = (it["note"] + "; " if it.get("note") else "") + d["note"]
                continue
            if status != "open" and d.get("if_open"):
                report.setdefault("already_decided", []).append(it["id"])
                continue
            it["status"], it["note"] = "human", d.get("note") or it.get("note")
            report["human"] += 1
            continue
        if by != "human" and it.get("status") == "human":
            report.setdefault("conflicts", []).append({"id": it["id"], "why": HELD_FOR_HUMAN})
            continue
        if it.get("asr_problem") and "listened" not in d:
            report.setdefault("conflicts", []).append({"id": it["id"], "why": "нужно отдельное подтверждение прослушивания"})
            continue
        if "listened" in d:
            if by != "human" or not it.get("asr_problem") or d["listened"] not in ("words", "silence") \
                    or d.get("human_after"):
                report.setdefault("conflicts", []).append({"id": it["id"], "why": "нет явного подтверждения человека"})
                continue
            start, end = it.get("gap", [None, None])
            if start is None or end is None or end <= start:
                report.setdefault("conflicts", []).append({"id": it["id"], "why": "неверный интервал"})
                continue
            heard_words = _interval_words(data, start, end)
            if (d["listened"] == "silence" and heard_words) or (d["listened"] == "words" and not heard_words):
                report.setdefault("conflicts", []).append({"id": it["id"], "why":
                    "выбранный ответ противоречит словам стенограммы; сначала исправьте стенограмму"})
                continue
            it["listening"] = {"by": "human", "result": d["listened"],
                               "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "fingerprint": _listening_fingerprint(heard_words)}
            it["status"] = "resolved"
            report["kept"] += 1
            report["successful_ids"].append(it["id"])
            continue
        target = {**it, **({"word_ids": d["word_ids"]} if "word_ids" in d else {})}
        loc = _find_span(data, target)
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
                    _mark_provenance(w, "text", by, it["id"])
                    _mark_provenance(w, "speaker", by, it["id"])
            report["successful_ids"].append(it["id"])
            continue
        si, w1, w2 = loc
        seg = data["segments"][si]
        ws = seg["words"]
        before = {"text": " ".join(w["w"] for w in ws[w1:w2]), "speaker": seg["speaker"]}
        if by != "human" and ((d.get("text") and d["text"] != before["text"]) or
                              (d.get("speaker") and d["speaker"] != before["speaker"])):
            if any(human_touched(w) for w in ws[w1:w2]):
                report.setdefault("conflicts", []).append({"id": it["id"], "kept_edits": ["human"]})
                continue
        if (not d.get("text") or d["text"] == before["text"]) and d.get("speaker") in (None, seg["speaker"]):
            it["status"] = "resolved"
            report["kept"] += 1
            if by == "human" and d.get("text"):
                for w in ws[w1:w2]:
                    w["by"] = "human"
                    _mark_provenance(w, "text", by, it["id"])
            report["successful_ids"].append(it["id"])
            continue
        new_text = d.get("text")
        if new_text and new_text != before["text"]:
            model = [w.get("raw", w["w"]) for w in ws[w1:w2]]
            new_ws, conflicts, why = _diff_words(
                ws[w1:w2], new_text, it["id"], lambda w: overwrite_refusal(w, data, by, it["id"]))
            if conflicts:
                report.setdefault("conflicts", []).append({"id": it["id"], "kept_edits": conflicts,
                                                           "why": "; ".join(why)})
                continue
            original_text = {w.get("word_id"): w["w"] for w in ws[w1:w2] if w.get("word_id")}
            for w in new_ws:
                if not w.get("word_id") or original_text.get(w["word_id"]) != w["w"]:
                    _mark_provenance(w, "text", by, it["id"])
            ws[w1:w2] = new_ws
            w2 = w1 + len(new_ws)
            # одиночная замена слова на имя собственное — подсказка для словаря
            if len(model) == 1 and len(new_text.split()) == 1 and vocab_candidate(new_text, data) \
                    and re.sub(r"\W", "", new_text.lower()) != re.sub(r"\W", "", model[0].lower()):
                sug = {"from": re.sub(r"[^\w-]", "", model[0]), "to": re.sub(r"[^\w-]", "", new_text),
                       "card": it["id"]}
                if _common_word(sug["from"]):
                    sug["common"] = True
                    sug["note"] = (f"«{sug['from']}» — обычное слово: в словарь лучше заменой с условием "
                                   "(replace_if), а не в replace")
                    sug["add"] = {"replace_if": [{"from": sug["from"], "to": sug["to"], "after": []}]}
                else:
                    sug["add"] = {"replace": {sug["from"]: sug["to"]}}
                data.setdefault("vocab_suggestions", []).append(sug)
        new_spk = d.get("speaker") or seg["speaker"]
        if new_spk != seg["speaker"]:
            for w in ws[w1:w2]:
                _mark_provenance(w, "speaker", by, it["id"])
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
    if data != original:                     # без изменений — без новой ревизии и перезаписи файлов
        save(data, json_path, md_path)
    report["open_left"] = sum(1 for i in data.get("review") or [] if i.get("status", "open") == "open")
    report["vocab_suggestions"] = data.get("vocab_suggestions", [])
    report["conspects"] = _refresh_conspects(json_path)
    return data, report
