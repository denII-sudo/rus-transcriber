"""Запись расшифровки (.json и .md), переименование спикеров, разбор сомнительных мест.

Два слоя данных: слова модели (segments[].words: текст, время, уверенность, вторая гипотеза) и правки
(edits — журнал: кто, что, было/стало). Правка не стирает слой модели: исходные слова остаются
в поле model у исправленного места и в журнале.
"""
import copy
import hashlib
import inspect
import json
import math
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
             "gender": "род глагола не совпадает с полом говорящего",
             "contradiction": "противоречие: предложение расходится с решением человека"}


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


def place_open(item):
    """Место разбора открыто: его ещё никто не решал и не оставил человеку. Правило одно на программу (PR 7)."""
    return item.get("status", "open") == "open"


def place_unresolved(item):
    """Место разбора не решено: открыто или оставлено человеку."""
    return item.get("status", "open") in ("open", "human")


PLACE_STATUS_RU = {"open": "открыто", "resolved": "решено", "human": "оставлено человеку"}


def place_status_ru(item):
    return PLACE_STATUS_RU.get(item.get("status", "open"), "состояние неизвестно")


def unresolved(data):
    """Места разбора, ещё не решённые: открытые и оставленные человеку."""
    return [i for i in data.get("review") or [] if place_unresolved(i)]


def review_finished(data):
    """Разбор закончен: открытых мест нет (оставленные человеку не мешают)."""
    return not any(place_open(i) for i in data.get("review") or [])


def open_count(data):
    return sum(1 for i in data.get("review") or [] if place_open(i))


def doubt_marks(data):
    """{номер реплики: {номер слова: id места}} — слова нерешённых мест (по времени: индексы сдвигаются)."""
    out = {}
    for it in unresolved(data):
        if is_gap_place(it):
            # пропуск (ответ владельца, вариант А): слова паузы — начатые в ней; у пустой паузы помечены последнее
            # слово до неё и первое после, чтобы цитата рядом с непрослушанной паузой не считалась проверенной
            times = _place_time(it)
            if times is None:
                continue
            inside = pause_words(data, *times)
            if inside:
                # при наложении речи — реплика того, кому место приписано (как у мест-слов)
                best = min((data["segments"][si]["speaker"] != it.get("speaker"), si)
                           for si in {x[0] for x in inside})[1]
                found = [x for x in inside if x[0] == best]
            else:
                found = neighbors_of_pause(data, *times)
            for si, k, _w in found:
                out.setdefault(si, {})[k] = it["id"]
            continue
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


def place_current_text(data, item):
    """Слова места в стенограмме сейчас: по word_ids, а если какого-то уже нет — по времени места.
    У пропуска и места без слов — текст места. Одна функция для обоих отрисовщиков «Сомнений» (PR 7)."""
    if data and not is_gap_place(item):
        by_id = {w.get("word_id"): w for seg in data.get("segments") or [] for w in seg.get("words") or []}
        for place in (item, {k: v for k, v in item.items() if k != "word_ids"}):
            found = place_words(data, place)
            words = [str(by_id[i].get("w", "")) for i in (found[0] if found else None) or [] if i in by_id]
            if words:
                return " ".join(words)
    return str(item.get("text") or "")


def _item_line(it, third_name="Whisper", second_name="CTC", data=None, status=False):
    why = ", ".join(REASON_RU.get(r, r) for r in it.get("reasons", []))
    s = f"- {hms(it['t'])} {it['speaker']}: «{place_current_text(data, it)}» — {why}"
    if status:
        s += f"; {place_status_ru(it)}"
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
    if q0.get("voices_error"):
        lines.append("_Внимание: проверка голосов не выполнена или выполнена не полностью: голосовые профили "
                     "к этой записи не применены, имена по голосам не подписаны, метки «Спикер N» — только "
                     "деление pyannote. Причина — в статусе задачи и в JSON (quality.voices_error); имена "
                     "подтверждает человек (rename_speakers)._\n")
    if q0.get("group_doubt"):
        lines.append(f"_Внимание: {q0['group_doubt']['note']}._\n")
    if q0.get("names_outside_vocab"):
        lines.append(f"_Нет в словаре группы: {', '.join(q0['names_outside_vocab'])} — их голоса в профиль не "
                     "идут, пока их не добавят в participants словаря._\n")
    if data.get("names_source") == "voices" and not data.get("names_confirmed"):
        lines.append("_Имена подписаны автоматически по голосовым профилям; подтвердить или исправить — "
                     "rename_speakers._\n")
    if data.get("review") is not None:
        open_items = [i for i in data["review"] if place_open(i)]
        human = [i for i in data["review"] if i.get("status") == "human"]
        if data.get("review_done"):
            lines.append(f"_Сомнительные места разобраны: правок {len(data.get('edits', []))}, "
                         f"осталось для человека {len(human)} (в конце)._\n")
        elif open_items:
            lines.append(f"_Сомнительных мест: {len(open_items)} — ещё не разобраны (review_items)._\n")
    marks = doubt_marks(data) if data.get("review") else {}
    if marks:
        lines.append(DOUBT_LEGEND + "\n")
    before, tail = interjection_layout(data.get("segments") or [], data.get("review") or [])
    for si, seg in enumerate(data["segments"]):
        for mark in before.get(si, ()):
            lines += [mark, ""]
        lines.append(_seg_line(seg, marks.get(si)))
        lines.append("")
    for mark in tail:
        lines += [mark, ""]
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
                    who += _counts_text(e.get("changed"), e.get("seen"), len(e.get("seen_changed") or []))
                    lines.append(f"- {hms(e['t'])} ({who}): " + "; ".join(ch)
                                 + (f" — {e['note']}" if e.get("note") else ""))
                lines.append("")
        else:
            lines += ["## Сомнительные места", ""] + [_item_line(i, third_name, second_name, data, status=True)
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


def _ensure_ids(data):
    """ID записи, реплик и слов. Не зависят от времени слова или его позиции в отображении.
    Старые файлы получают их при первом сохранении; дубликаты после разрезания реплик получают новые ID.
    Отпечаток прослушивания берётся по ID, поэтому его ставят после этой функции (PR 7)."""
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


def disagreement_words(data, item):
    """ID слов расхождения места «модели расходятся» (asr_problem.state == "disagreement"): слово с вариантом
    другой модели и следующие за ним по alt_span, слово со вставкой другой модели (alt_ins). У результатов без
    этих полей — все слова места. [] — место не такое или его слова не найти. Чистая функция."""
    if (item.get("asr_problem") or {}).get("state") != "disagreement" or is_gap_place(item):
        return []
    found = place_words(data, item)
    ids = [wid for wid in (found[0] if found else None) or [] if wid]
    flat = [w for seg in data.get("segments") or [] for w in seg.get("words") or []]
    position = {w.get("word_id"): k for k, w in enumerate(flat)}
    inside, marked = set(ids), []
    for wid in ids:
        k = position.get(wid)
        if k is None:
            continue
        span = flat[k].get("alt_span")
        if type(span) is int and span > 0:
            marked += [w.get("word_id") for w in flat[k:k + span] if w.get("word_id") in inside]
        elif flat[k].get("alt_ins"):
            marked.append(wid)
    return list(dict.fromkeys(marked)) or ids


def _mark_listened_by_card(data, item):
    """Отметка «человек прослушал» по своему ответу на карточку. Отпечаток — по словам интервала с ID: зовите
    после _ensure_ids, иначе первое же сохранение выдаст новые ID и снимет отметку («дыра 1»)."""
    interval = gap_interval(item)
    if interval is None:
        return False
    item["listening"] = {"by": "human", "result": "words", "via": "card",
                         "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "fingerprint": _listening_fingerprint(_interval_words(data, *interval))}
    return True


def _save_locked(data, json_path, md_path):
    # Это не межпроцессная блокировка.
    _ensure_ids(data)
    for item in data.get("review") or []:
        listening = item.get("listening")
        if item.get("asr_problem") and not listening and item.get("status") == "resolved":
            item["status"] = "open"
            data["review_done"] = False
        if contradiction_pending(item) and item.get("status") == "resolved":
            # противоречие закрывает только выбор на его карточке (PR 13б)
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
                import documents
                completed = recordings.materialize_recording(store, recording_id)
                documents.require_written(store, recording_id, result["document_revision"], completed)
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


def confirm_snapshot(snapshot, before, after):
    """Снимок отмены ответа после подтверждения человека (К-23): слова и журнал правок те же, изменились только
    строки мест, которые снимок и так восстанавливает. Тогда снимок прежний, с отпечатком состояния после
    подтверждения: отмена вернёт место к состоянию до ответа. Иначе снимок не годится — None."""
    if not snapshot or snapshot.get("format") != 2:
        return None
    if before.get("segments") != after.get("segments") or (before.get("edits") or []) != (after.get("edits") or []):
        return None
    old, new = before.get("review") or [], after.get("review") or []
    if len(old) != len(new):
        return None
    try:
        covered = {i for p in snapshot["review_patches"] for i in range(p["start"], p["start"] + p["remove"])}
    except (KeyError, TypeError):
        return None
    if any(a != b and i not in covered for i, (a, b) in enumerate(zip(old, new))):
        return None
    return dict(snapshot, after=_undo_fingerprint(after))


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
    data["review_done"] = review_finished(data)
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


# ---------- правило «слова места» (РАР-6.1, PR 6а; реестр правил, строка 8) ----------
# Место-пропуск (причина «missed»: речи в записи нет, слов внутри не должно быть) владеет только словами,
# НАЧАТЫМИ внутри него: полуоткрытый интервал [t1, t2) в целых сотых. Слово рядом с паузой ей не принадлежит.
# Места-слова остаются по word_ids или по закрытому интервалу с допуском. Исключение, названное здесь и в
# описании PR: при применении ответа (_find_span) место-слово ищется по началу слова в одной реплике, а для
# цитат (word_in_place) — ещё и по перекрытию во всех репликах; это два разных правила, 6а их не сводит.
WORD_PLACE_TOLERANCE = 0.01   # допуск мест-слов, секунды (прежде workflow.PLACE_TIME_TOLERANCE)


def _cents(t):
    """Время в целых сотых. Пакет хранит round(t, 2), поэтому сначала так же округляем: round(t * 100)
    расходится с round(t, 2) на временах вида x,xx5."""
    return int(round(round(float(t), 2) * 100))


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _place_time(item):
    """(t1, t2) места, если оба — конечные числа и t1 <= t2; иначе None (время негодно)."""
    t1, t2 = item.get("t1", item.get("t")), item.get("t2", item.get("t"))
    return (t1, t2) if _finite_number(t1) and _finite_number(t2) and t1 <= t2 else None


def is_gap_place(item):
    """Место-пропуск: «missed» в причинах (диаризация и проход без слов: ["missed"], ["asr", "missed"])."""
    reasons = item.get("reasons")
    return isinstance(reasons, (list, tuple)) and "missed" in reasons


def gap_interval(item):
    """(начало, конец) интервала прослушивания места (поле gap), если оба — конечные числа и начало < конца."""
    gap = item.get("gap")
    if isinstance(gap, (list, tuple)) and len(gap) == 2 and all(_finite_number(x) for x in gap) and gap[0] < gap[1]:
        return gap[0], gap[1]
    return None


INTERJECTIONS_MARK = "_[междометия, не расшифровано] {t1}–{t2}_"


def interjection_marks(review_items):
    """[(начало, конец, id)] мест, закрытых кнопкой «только междометия», по времени. Одна для обоих отрисовщиков."""
    out = []
    for item in review_items or []:
        listening = item.get("listening")
        times = gap_interval(item)
        if isinstance(listening, dict) and listening.get("result") == "interjections" and times:
            out.append((times[0], times[1], str(item.get("id"))))
    return sorted(out)


def interjection_layout(segments, review_items):
    """Где стоят пометки «[междометия, не расшифровано]»: ({номер реплики: [строки перед ней]}, [строки в конце]).
    Пометка — перед первой репликой, начало которой не раньше начала паузы."""
    before, tail = {}, []
    starts = []
    for seg in segments:
        start = seg.get("start")
        if not _finite_number(start):
            words = [w.get("s") for w in seg.get("words") or [] if _finite_number(w.get("s"))]
            start = words[0] if words else None
        starts.append(start)
    for t1, t2, _rid in interjection_marks(review_items):
        line = INTERJECTIONS_MARK.format(t1=hms(t1), t2=hms(t2))
        at = next((si for si, start in enumerate(starts) if start is not None and start >= t1), None)
        if at is None:
            tail.append(line)
        else:
            before.setdefault(at, []).append(line)
    return before, tail


def in_pause(s, t1, t2):
    """Единственная запись правила «слово в паузе»: начало слова в [t1, t2), в целых сотых."""
    return _cents(t1) <= _cents(s) < _cents(t2)


def pause_words(data, t1, t2):
    """[(номер реплики, номер слова, слово)] слов, начатых в паузе [t1, t2), в порядке стенограммы.

    t1, t2 — конечные числа (проверяет вызывающий); слова без конечного начала пропускаются."""
    out = []
    for si, seg in enumerate(data.get("segments") or []):
        for k, w in enumerate(seg.get("words") or []):
            if _finite_number(w.get("s")) and in_pause(w["s"], t1, t2):
                out.append((si, k, w))
    return out


def neighbors_of_pause(data, t1, t2):
    """[(номер реплики, номер слова, слово)]: последнее слово с началом до паузы [t1, t2) и первое после неё
    (что есть; равные времена — по порядку стенограммы: «до» — позднее, «после» — раньше). Слова без конечного
    начала пропускаются. Одна запись для place_words(pause_neighbors=True) и пометок ⟦…⟧ (doubt_marks)."""
    timed = [(si, k, w) for si, seg in enumerate(data.get("segments") or [])
             for k, w in enumerate(seg.get("words") or []) if _finite_number(w.get("s"))]
    before = [x for x in timed if _cents(x[2]["s"]) < _cents(t1)]
    after = [x for x in timed if _cents(x[2]["s"]) >= _cents(t2)]
    key = lambda x: (x[2]["s"], x[0], x[1])
    return ([max(before, key=key)] if before else []) + ([min(after, key=key)] if after else [])


def word_in_place(start, end, t1, t2):
    """Правило пересечения слова [start, end] с местом-словом [t1, t2] (секунды): для цитат и пометок 7.3.

    Слово в месте, если его начало лежит в [t1 − допуск; t2 + допуск] (у мест распознавания t1 — начало
    первого слова, t2 — начало последнего) или если слово перекрывает место отрезком положительной длины
    (start < t2 и end > t1). Слово, которое лишь касается начала места своим концом (end == t1), не входит.
    """
    return (t1 - WORD_PLACE_TOLERANCE <= start <= t2 + WORD_PLACE_TOLERANCE
            or (start < t2 and end > t1))


def place_words(data, item, *, pause_neighbors=False):
    """Слова места: (ID слов в порядке стенограммы, это_пауза) или None, если место привязать нельзя.

    Договор с этапом 7.5 (plan-7-5.md, раздел 3). Пропуск: word_ids места не читаются, слова — pause_words.
    Иное место: его word_ids (какого-то нет в стенограмме, повтор, не строка — None) или, без word_ids,
    слова по word_in_place. это_пауза — внутри места нет слов; при pause_neighbors=True тогда возвращаются
    ID последнего слова до места и первого после. Чистая функция: данные не меняет.
    """
    times = _place_time(item)
    if times is None:
        return None
    t1, t2 = times
    gap_place = is_gap_place(item)
    ids = item.get("word_ids")
    order = []                                  # (слово, позиция в стенограмме)
    for seg in data.get("segments") or []:
        for w in seg.get("words") or []:
            order.append(w)
    if not gap_place and isinstance(ids, list) and ids:
        if not all(isinstance(x, str) and x for x in ids) or len(ids) != len(set(ids)):
            return None
        by_id = {w.get("word_id"): w for w in order if w.get("word_id")}
        if any(x not in by_id for x in ids):
            return None
        wanted = set(ids)
        return [w["word_id"] for w in order if w.get("word_id") in wanted], False
    timed = [(pos, w) for pos, w in enumerate(order) if _finite_number(w.get("s"))]
    if not timed:
        return None
    if gap_place:
        inside = [w for _pos, w in timed if in_pause(w["s"], t1, t2)]
    else:
        def end_of(w):
            e = w.get("e")
            return e if _finite_number(e) and e >= w["s"] else w["s"]
        inside = [w for _pos, w in timed if word_in_place(w["s"], end_of(w), t1, t2)]
    chosen = inside
    if not inside and pause_neighbors:
        if gap_place:
            chosen = [w for _si, _k, w in neighbors_of_pause(data, t1, t2)]
        else:
            before = [(w["s"], pos, w) for pos, w in timed if w["s"] < t1 - WORD_PLACE_TOLERANCE]
            after = [(w["s"], pos, w) for pos, w in timed if w["s"] > t2 + WORD_PLACE_TOLERANCE]
            chosen = ([max(before, key=lambda x: (x[0], x[1]))[2]] if before else []) \
                + ([min(after, key=lambda x: (x[0], x[1]))[2]] if after else [])
    if any(not isinstance(w.get("word_id"), str) or not w.get("word_id") for w in chosen):
        return None
    return [w["word_id"] for w in chosen], not inside


def places_overlap(x, y):
    """Единственное правило пересечения двух мест (карточек): True, если они делят время.

    Оба места-слова — прежняя формула с допуском. Иначе полуоткрытые интервалы в сотых: пропуск
    [t1, t2), место-слово [t1, t2 + 1 сотая) (начала его слов от t1 до t2 включительно). Место с негодным
    временем ни с чем не пересекается."""
    tx, ty = _place_time(x), _place_time(y)
    if tx is None or ty is None:
        return False
    gx, gy = is_gap_place(x), is_gap_place(y)
    if not gx and not gy:
        return tx[0] <= ty[1] + WORD_PLACE_TOLERANCE and ty[0] <= tx[1] + WORD_PLACE_TOLERANCE

    def interval(times, is_gap):
        return _cents(times[0]), _cents(times[1]) + (0 if is_gap else 1)
    a0, a1 = interval(tx, gx)
    b0, b1 = interval(ty, gy)
    return a0 < b1 and b0 < a1


# ---------- разбор сомнительных мест ----------
def _find_span(data, it):
    """Находит место по времени слов (индексы сдвигаются после правок).

    Пропуск — слова, начатые в его паузе [t1, t2); word_ids места не читаются (старые пакеты несли в них ID
    соседнего слова). Пустая пауза — None."""
    if is_gap_place(it):
        times = _place_time(it)
        if times is None:
            return None
        in_segment = {}
        for si, k, _w in pause_words(data, *times):
            in_segment.setdefault(si, []).append(k)
        locs = [(si, ks[0], ks[-1] + 1) for si, ks in in_segment.items()]
        return next((loc for loc in locs if data["segments"][loc[0]].get("speaker") == it.get("speaker")),
                    locs[0] if locs else None)
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
    pairs = [(w, sg["speaker"]) for sg in data["segments"] for w in sg.get("words") or []
             if t1 - 0.01 <= w["s"] <= t2 + 0.01]
    confirmed = _span_contradiction(data, pairs, (text or "").strip(), speaker, by, card) if pairs else None
    if confirmed is not None:
        _ensure_ids(data)
        rid, new = propose_contradiction(data, [w for w, _ in pairs], {
            "text": (text or "").strip(), "speaker": speaker, "by": by, "via": "edit_span",
            "speaker_now": pairs[0][1]}, confirmed)
        if new:
            save(data, json_path, md_path)
        return data, {"written": False, "contradiction": rid, "new": new,
                      "text": CONTRADICTION_NOT_WRITTEN.format(rid=rid)}
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
                                  close_note=f"решено правкой отрезка {card} ({who}{{counts}})", close_held=by == "human")
    if card in rep["failed"]:
        reason = rep["failed"][card]
        raise ValueError(reason + (SPAN_REOPEN_HINT if "ответом человека на карточку" in reason else ""))
    c = rep["cards"].get(card, {})
    # закрытыми называются места по сохранённым статусам: место со сбоем распознавания без прослушивания
    # при сохранении возвращается в open
    places = {i["id"]: i for i in data2.get("review") or []}
    resolved = sorted(rid for rid in closed if (places.get(rid) or {}).get("status") == "resolved")
    return data2, {"card": card, "before": c.get("before"), "after": c.get("after"), "resolved": resolved,
                   "seen_changed": rep.get("seen_changed", []), "conspects": rep.get("conspects")}


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
    data["review_done"] = review_finished(data)
    save(data, json_path, md_path)
    return data, {"reopened": done, "unknown": unknown,
                  "open_left": open_count(data)}


def _wkey(s):
    return re.sub(r"[^\w-]", "", s.lower().replace("ё", "е"))


def _mark_provenance(word, dimension, by, edit, via=None):
    provenance = copy.deepcopy(word.get("provenance") or {})
    provenance[dimension] = {"by": by, "edit": edit, **({"via": via} if via else {})}
    word["provenance"] = provenance


def _seen_key(word):
    return word.get("word_id") or f"t{round(word['s'], 2)}"


def _seen_snapshot(pairs):
    """Слова с отметкой «видел» до правки. pairs — (слово, говорящий). Отметка `word["seen"]` лежит вне
    `provenance`: она не делает слово «тронутым человеком» (`human_touched`) и в публичный документ sqlite не идёт."""
    return {_seen_key(w): {"word_id": w.get("word_id"), "t": w["s"], "w": w["w"], "spk": spk,
                           "seen_edit": w["seen"].get("edit")}
            for w, spk in pairs if isinstance(w.get("seen"), dict)}


def _seen_settle(snapshot, pairs):
    """После правки: у слова, чей текст или говорящий изменились (или которое исчезло), отметка снимается.
    -> список таких слов [{"word_id", "t", "w" (прежний текст), "seen_edit"}] — «видевшиеся» слова, изменённые правкой."""
    now = {_seen_key(w): (w, spk) for w, spk in pairs}
    changed = []
    for key, old in snapshot.items():
        cur = now.get(key)
        if cur and cur[0]["w"] == old["w"] and cur[1] == old["spk"]:
            continue
        if cur:
            cur[0].pop("seen", None)
        changed.append({"word_id": old["word_id"], "t": old["t"], "w": old["w"], "seen_edit": old["seen_edit"]})
    return changed


def _counts_text(changed, seen, seen_changed):
    """Числа правки отрезка для строки .md и заметки места: у человека «изменено N, видел M», у агента —
    сколько из изменённых слов человек видел. Пусто, если отметок «видел» не было."""
    if seen:
        return f"; изменено {changed}, видел {seen}"
    if seen_changed:
        return f"; изменено {changed}, из них видел человек {seen_changed}" if changed else \
            f"; видел человек: {seen_changed}"
    return ""


HELD_FOR_HUMAN = "место оставлено человеку"
# Отказы варианта карточки на пустой паузе (apply_review): слов нет — «как есть», подпись и замена не к чему.
PAUSE_EMPTY_AS_IS = ("в паузе нет слов — подтверждать «как есть» нечего; "
                     "впишите услышанное в поле ответа («Имя: слова»)")
PAUSE_EMPTY_SPEAKER = ("в паузе нет слов — подписывать говорящего нечему; "
                       "впишите услышанное в поле ответа («Имя: слова»)")
PAUSE_EMPTY_TEXT = ("вариант с текстом в пустую паузу не вписывается; "
                    "впишите услышанное в поле ответа («Имя: слова»)")


# Кнопки прослушивания пропуска (В7): у каждого места-пропуска с годным интервалом gap.
LISTEN_SILENCE = "Прослушал: слов нет"
LISTEN_INTERJECTIONS = "Прослушал: только междометия, не записывать"
LISTEN_WORDS = "Прослушал: слова в стенограмме верны"
PAUSE_BUTTONS_HINT = f" или нажмите «{LISTEN_SILENCE}» или «{LISTEN_INTERJECTIONS}», если слов нет"
PAUSE_HAS_WORDS = ("в паузе есть слова («{words}»): сначала исправьте их правкой отрезка или ответом "
                   "на их карточку")


def _empty_pause_refusal(decision, item=None):
    if decision.get("text"):
        why = PAUSE_EMPTY_TEXT
    else:
        why = PAUSE_EMPTY_SPEAKER if decision.get("speaker") else PAUSE_EMPTY_AS_IS
    # кнопки называются, только если они есть на карточке (у места годный интервал gap)
    return why + (PAUSE_BUTTONS_HINT if item is not None and gap_interval(item) else "")
EDIT_SPAN_AUTHOR_REQUIRED = ('edit_span: укажите автора правки by — "claude" (правка агента, в том числе по поручению '
                             'человека) или "human" (текст продиктовал человек, только командной строкой); '
                             'без автора правка не принимается')
SPAN_REOPEN_HINT = "; чтобы решить заново — reopen_review (rt.py reopen)"
SEVERAL_SPEAKERS = ("на отрезке несколько говорящих ({}): укажите, чей это текст — в карточке «Имя: текст», "
                    "в edit_span — speaker")


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
    if writer == "human" and _contradiction_covers(data, card, word):
        return None             # выбор человека на карточке противоречия этого слова (PR 13б)
    span = prior["via"] == "edit_span"
    if prior["by"] == "human":
        how = "правкой отрезка человека" if span else "ответом человека на карточку"
    elif prior["by"]:
        how = "правкой отрезка агента" if span else "правкой агента по карточке"
    else:
        how = "прежней правкой без автора"
    return f"слово «{word.get('w')}» уже исправлено {how} ({prior['id']})"


CONTRADICTION = "contradiction"
CONTRADICTION_NOT_WRITTEN = ("не записано: противоречие, карточка {rid} — слова не изменены; "
                             "выберите вариант на карточке этого места")
CONTRADICTION_VIA_RU = {"edit_span": "правкой отрезка", "apply_review": "решением агента", "card": "карточкой"}


def contradiction_pending(item):
    """У места противоречие, которое ждёт выбора человека на его карточке (PR 13б)."""
    return bool((item.get("contradiction") or {}).get("proposals"))


def _contradiction_covers(data, card, word):
    """Карточка card — место с ожидающим противоречием, в слова которого входит word."""
    item = next((i for i in data.get("review") or [] if i.get("id") == card), None)
    return bool(item and contradiction_pending(item) and word.get("word_id")
                and word["word_id"] in item["contradiction"].get("words", []))


def human_confirmation(word, data=None):
    """Как человек решил слово: правкой (карточкой или правкой отрезка) или «как есть» на карточке.
    -> {"card", "via"} или None. «Как есть» — `provenance` человека без `edited` (prior_edit его не видит:
    до PR 13б правка отрезка человека такое слово молча перезаписывала)."""
    prior = prior_edit(word, data)
    if prior is not None:
        return {"card": prior["id"], "via": prior["via"]} if prior["by"] == "human" else None
    for dimension in ("text", "speaker"):
        mark = (word.get("provenance") or {}).get(dimension)
        if isinstance(mark, dict) and mark.get("by") == "human" and mark.get("edit"):
            return {"card": mark["edit"], "via": mark.get("via") or "card"}
    return None


def _next_place_id(data):
    numbers = [int(m.group(1)) for i in data.get("review") or [] for m in [re.fullmatch(r"r(\d+)", str(i.get("id")))]
               if m]
    return f"r{max(numbers, default=0) + 1}"


def propose_contradiction(data, words, proposal, confirmed=None):
    """Записать предложение, противоречащее решению человека, на место этих слов (или на новое место).
    words — слова стенограммы (с word_id), proposal — {text, speaker?, by, via}; confirmed — что решил
    человек (по умолчанию текст слов сейчас). Слова не меняются; место открыто до выбора на карточке.
    Повтор того же предложения (слова, текст, говорящий, автор) ничего не меняет. -> (id места, новое?)"""
    ids = [w["word_id"] for w in words]
    places = data.setdefault("review", [])
    preferred = (confirmed or {}).get("card")       # место той карточки, которой человек решил слово
    item = next((i for i in places if i.get("id") == preferred and set(ids) <= set(i.get("word_ids") or [])),
                None) or next((i for i in places if set(ids) <= set(i.get("word_ids") or [])), None) or \
        next((i for i in places if set(ids) & set(i.get("word_ids") or [])), None)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    current = " ".join(w["w"] for w in words)
    if item is None:
        item = {"id": _next_place_id(data), "status": "open", "t": words[0]["s"], "t1": words[0]["s"],
                "t2": words[-1]["s"], "speaker": proposal.get("speaker_now"), "reasons": [CONTRADICTION],
                "text": current, "word_ids": ids}
        places.append(item)
        places.sort(key=lambda i: i.get("t", 0))
    record = item.setdefault(CONTRADICTION, {"words": ids, "proposals": [], "history": []})
    key = hashlib.sha1(json.dumps([sorted(ids), proposal.get("text"), proposal.get("speaker"), proposal.get("by")],
                                  ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    if any(p.get("key") == key for p in record["proposals"]):
        return item["id"], False
    if not record["proposals"]:
        record["words"] = ids
        record["confirmed"] = {"text": current, "by": "human", "at": now, **(confirmed or {})}
    record["proposals"].append({"text": proposal.get("text", ""), "by": proposal.get("by"),
                                "via": proposal.get("via"), "at": now, "key": key,
                                **({"speaker": proposal["speaker"]} if proposal.get("speaker") else {})})
    if CONTRADICTION not in (item.get("reasons") or []):
        item["reasons"] = list(item.get("reasons") or []) + [CONTRADICTION]
    if item.get("status", "open") == "resolved":
        item["status"] = "open"
    data["review_done"] = False
    return item["id"], True


def settle_contradiction(item, choice, by, proposal=None, note=None):
    """Выбор человека на карточке противоречия: итог — в историю, предложений больше нет.
    note — примечание человека (human_note): только здесь, в ответе и в журнале правок. -> записан ли итог"""
    record = item.get(CONTRADICTION) or {}
    if not record.get("proposals"):
        return False
    record.setdefault("history", []).append({
        "choice": choice, "by": by, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "confirmed": record.get("confirmed"), "proposals": record["proposals"],
        **({"proposal": proposal} if proposal else {}), **({"human_note": note} if note else {})})
    record["proposals"] = []
    return True


def note_contradiction(item, by, note):
    """К-2: «Оставить человеку» на карточке противоречия с примечанием — примечание в историю места. Предложения
    не снимаются: противоречие ждёт выбора. -> записано ли"""
    record = item.get(CONTRADICTION) or {}
    if by != "human" or not note or not record.get("proposals"):
        return False
    record.setdefault("notes", []).append({"choice": "human", "by": by, "human_note": note,
                                           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return True


def _propose_from_review(data, item, decision, by, report, confirmed=None):
    """Решение агента (apply_review) по словам или месту человека: вместо одного отказа — предложение на
    карточку противоречия этого места (PR 13б). Слова не меняются."""
    if not (decision.get("text") or decision.get("speaker")):
        return
    loc = _find_span(data, {**item, **({"word_ids": decision["word_ids"]} if "word_ids" in decision else {})})
    if loc is None:
        return
    seg = data["segments"][loc[0]]
    words = seg["words"][loc[1]:loc[2]]
    if not words:
        return
    _ensure_ids(data)
    if confirmed is None:
        confirmed = dict(next((m for m in map(lambda w: human_confirmation(w, data), words) if m), None) or {})
    rid, _ = propose_contradiction(data, words, {
        "text": decision.get("text") or " ".join(w["w"] for w in words), "speaker": decision.get("speaker"),
        "by": by, "via": "apply_review", "speaker_now": seg["speaker"]}, confirmed)
    report.setdefault("contradictions", []).append(rid)


def _span_contradiction(data, pairs, text, speaker, by, card):
    """Правка отрезка противоречит решению человека? pairs — (слово, говорящий) отрезка.
    -> что решил человек ({card?, via?}) или None. Агент: на отрезке есть слово, тронутое человеком
    (прежде — отказ). Человек: правка меняет текст или говорящего слова, которое он решил карточкой (правкой
    или «как есть») или другой правкой отрезка (прежде — отказ или, у «как есть», молчаливая запись)."""
    import difflib

    def decided(w):
        mark = human_confirmation(w, data)
        return mark if mark and mark["card"] != card else None      # в том числе своя правка отрезка (В-1)
    words = [w for w, _ in pairs]
    if by != "human":
        touched = [w for w in words if human_touched(w)]
        if not touched:
            return None
        mark = next((m for m in map(decided, touched) if m), None)
        return dict(mark) if mark else {}
    sm = difflib.SequenceMatcher(None, [_wkey(w["w"]) for w in words], [_wkey(x) for x in text.split()],
                                 autojunk=False)
    changed = [w for op, i1, i2, _, _ in sm.get_opcodes() if op != "equal" for w in words[i1:i2]]
    if speaker:
        changed += [w for w, spk in pairs if spk != speaker and w not in changed]
    mark = next((m for m in map(decided, changed) if m), None)
    return dict(mark) if mark else None


def _held_places(data):
    """Места, оставленные человеку (статус human): (id() их слов, их отрезки). Агент их не меняет и не
    закрывает, пока нет правила о поручении такого места ИИ.

    Отрезок — (a, b, пропуск?): у места-слова закрытый [t1 − допуск; t2 + допуск], у пропуска — [t1, t2) в
    сотых (_span_has_time, _span_overlaps). Слова пропуска — слова паузы; соседнее слово ему не принадлежит."""
    words, spans = set(), []
    all_words = [w for seg in data.get("segments") or [] for w in seg.get("words") or []]
    for it in data.get("review") or []:
        if it.get("status") != "human":
            continue
        if is_gap_place(it):
            times = _place_time(it)
            if times is not None:
                spans.append((times[0], times[1], True))
                words.update(id(w) for _si, _k, w in pause_words(data, *times))
            continue
        t1, t2 = it.get("t1", it.get("t")), it.get("t2", it.get("t"))
        if t1 is not None and t2 is not None:
            spans.append((t1 - WORD_PLACE_TOLERANCE, t2 + WORD_PLACE_TOLERANCE, False))
        ids = it.get("word_ids")
        if ids:
            words.update(id(w) for w in all_words if w.get("word_id") in ids)
        elif t1 is not None and t2 is not None:
            words.update(id(w) for w in all_words if t1 - WORD_PLACE_TOLERANCE <= w["s"] <= t2 + WORD_PLACE_TOLERANCE)
    return words, spans


def _span_has_time(spans, t):
    """Попадает ли время t (начало вставляемого слова) в один из отрезков _held_places."""
    return t is not None and any(in_pause(t, a, b) if gap else a <= t <= b for a, b, gap in spans)


def _span_overlaps(spans, t1, t2):
    """Пересекает ли явный отрезок правки [t1, t2] один из отрезков _held_places."""
    return any((_cents(t1) < _cents(b) and _cents(a) <= _cents(t2)) if gap
               else (a <= t2 + WORD_PLACE_TOLERANCE and t1 - WORD_PLACE_TOLERANCE <= b)
               for a, b, gap in spans)


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
                     via=None, close_note=None, close_held=False, listened_by=None, human_notes=None):
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
    слова. Человек (by="human") в правке с via ставит неизменённым словам отметку «видел» (`word["seen"]`, вне
    `provenance`): слово не становится подтверждённым. Правка, изменившая слово, отметку снимает; у агента такие
    слова идут в `seen_changed` отчёта и журнала.
    close_note — заметка закрытых мест (по умолчанию «решено человеком (через карточку)»; «{counts}» заменяется
    числами правки отрезка); close_held — закрывать и места, оставленные человеку.
    listened_by="human" — правки пришли ответом человека на свою карточку (review_pack): место «модели
    расходятся» этой карточки закрывается с отметкой прослушивания, если правка задела слово расхождения.
    Правка отрезка (via) и агент отметку не ставят. human_notes — {карточка: примечание человека} на карточке
    противоречия (К-2): в историю места и в журнал правок этой карточки; записанные — в отчёте notes_written.
    -> (data, отчёт)"""
    data = load(json_path)
    original = copy.deepcopy(data)
    answers = answers or {}
    human_notes = human_notes or {}
    name_to_label = {v: k for k, v in (data.get("speaker_names") or {}).items()}
    flat = []
    for si, seg in enumerate(data["segments"]):
        for w in seg.get("words") or []:
            flat.append({"w": w, "spk": seg["speaker"], "seg": si})
    by_t = {}
    seen_before = _seen_snapshot((f["w"], f["spk"]) for f in flat)
    counts, counts_text = {}, {}

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
            return _span_has_time(held_spans, t)

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
                if any(held(k) for k in ks) or (not ks and _span_overlaps(held_spans, e["t1"], e["t2"])):
                    return HELD_FOR_HUMAN
                names = list(dict.fromkeys(flat[k]["spk"] for k in ks))
                if len(names) > 1 and not e.get("speaker") and (e.get("text") or "").strip():
                    return SEVERAL_SPEAKERS.format(", ".join(names))   # иначе всё ушло бы первому говорящему
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

    # Слова расхождения мест «модели расходятся» — до правки: после неё удалённых слов уже нет (PR 7).
    listen = {}
    if listened_by == "human" and by == "human" and not via:
        places_before = {i.get("id"): i for i in data.get("review") or []}
        for card in groups:
            ids = set(disagreement_words(data, places_before[card])) if card in places_before else set()
            if ids:
                # слова, которые человек вписал своим ответом (в том числе тем же текстом): он их слышал
                aimed = {wid for e in groups[card] if e["op"] in ("place_text", "replace", "delete")
                         for wid in list(e.get("word_ids") or []) + [e.get("word_id")] if wid}
                listen[card] = ({f["w"]["word_id"]: f["w"]["w"] for f in flat if f["w"].get("word_id") in ids},
                                bool(aimed & ids))
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
                changed_new, kept = 0, set()
                for w in new_ws:
                    text_changed = not w.get("word_id") or old_text.get(w["word_id"]) != w["w"]
                    if via and not (text_changed or speaker_changed):
                        kept.add(_seen_key(w))
                        if by == "human":         # правка отрезка человека: неизменённое слово — только «видел»
                            w["seen"] = {"by": by, "edit": card, "via": via}
                        continue                  # прежние отметки слова не меняются
                    changed_new += 1
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
                if via:
                    c = counts.setdefault(card, {"changed": 0, "seen": 0})
                    c["changed"] += max(changed_new, sum(1 for w in old if _seen_key(w) not in kept))
                    c["seen"] += len(kept) if by == "human" else 0
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
    seen_changed = _seen_settle(seen_before, [(w, sg["speaker"]) for sg in segs for w in sg["words"]])
    if seen_changed and by != "human":
        rep["seen_changed"] = seen_changed

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
        card_seen = [x for x in rep.get("seen_changed", []) if a_ <= x["t"] <= b_]
        c = counts.get(card)
        counts_text[card] = _counts_text(c["changed"], c["seen"], len(card_seen)) if c else ""
        log.append({"id": card, "t": rg["t1"], "by": by, "kind": "words", **({"via": via} if via else {}),
                    **({"changed": c["changed"], "seen": c["seen"]} if c and (c["seen"] or card_seen) else {}),
                    **({"seen_changed": card_seen} if card_seen else {}),
                    "before": {"text": bt, "speaker": who(brows)}, "after": {"text": at, "speaker": who(arows)},
                    "answer": answers.get(card), "note": log_note or "ответ человека в карточке",
                    **({"human_note": human_notes[card]} if human_notes.get(card) else {}),
                    "pairs": sorted({tuple(p) for p in pairs.get(card, [])})})
        rg.update(before=bt, after=at)
    # места: закрытые ответом на другую карточку и сами карточки
    items = {i["id"]: i for i in data.get("review") or []}
    to_mark = []

    def touched(card, item):
        """Правка карточки задела слово расхождения: вписала его (и тем же текстом), заменила, удалила, слила
        или вставила слово внутри места."""
        before, aimed = listen[card]
        if aimed:
            return True
        now = {w.get("word_id"): w["w"] for sg in segs for w in sg["words"] if w.get("word_id") in before}
        if any(now.get(wid) != text for wid, text in before.items()):
            return True
        times = gap_interval(item) or _place_time(item)
        return bool(times) and any(not w.get("word_id") and w.get("edited") == card
                                   and times[0] - 0.01 <= w["s"] <= times[1] + 0.01
                                   for sg in segs for w in sg["words"])
    for rid, card_via in (closed or {}).items():
        if any(card_via == c for c in rep["failed"]):
            continue
        it = items.get(rid)
        if it and (close_held or it.get("status", "open") != "human"):
            it["status"], it["note"] = "resolved", (close_note or f"решено человеком (через {card_via})").replace(
                "{counts}", counts_text.get(card_via, ""))
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
            if card in listen and touched(card, it):
                to_mark.append(it)
            if by == "human" and not via:
                if settle_contradiction(it, "own", by, note=human_notes.get(card)) and human_notes.get(card):
                    rep.setdefault("notes_written", []).append(card)
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
    to_mark = [it for it in to_mark if it.get("status") == "resolved"]
    if to_mark:
        _ensure_ids(data)          # новые слова получают ID до отпечатка
        for it in to_mark:
            _mark_listened_by_card(data, it)
    data["review_done"] = review_finished(data)
    if not dry_run and data != original:     # без изменений — без новой ревизии и перезаписи файлов
        save(data, json_path, md_path)
    rep["open_left"] = open_count(data)
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
        if d.get("confirm_applied"):
            # К-23: человек подтвердил применённый ответ, который ему отдали на проверку: слова не меняются,
            # место закрывается его действием. Агент так не закрывает; решённое место не трогается.
            if by != "human":
                report.setdefault("conflicts", []).append({"id": it["id"], "why": "нет явного подтверждения человека"})
                continue
            if it.get("status") == "human":
                it["status"] = "resolved"
                it["note"] = d.get("note") or it.get("note")
            report["successful_ids"].append(it["id"])
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
            if note_contradiction(it, by, d.get("human_note")):
                report.setdefault("notes_written", []).append(it["id"])
            continue
        if by != "human" and it.get("status") == "human":
            report.setdefault("conflicts", []).append({"id": it["id"], "why": HELD_FOR_HUMAN})
            _propose_from_review(data, it, d, by, report, {"via": "held"})
            continue
        if it.get("asr_problem") and "listened" not in d:
            report.setdefault("conflicts", []).append({"id": it["id"], "why": "нужно отдельное подтверждение прослушивания"})
            continue
        if "listened" in d:
            gap_place = is_gap_place(it)

            def refuse(why, rid=it["id"], pause=gap_place):
                report.setdefault("conflicts", []).append({"id": rid, "why": why})
                if pause:                  # исход «нужна кнопка паузы» (core)
                    report.setdefault("pause_refused", []).append(rid)

            allowed = ("words", "silence", "interjections") if gap_place else ("words", "silence")
            if by != "human" or not (it.get("asr_problem") or gap_place) or d["listened"] not in allowed \
                    or d.get("human_after"):
                refuse("нет явного подтверждения человека")
                continue
            interval = gap_interval(it)
            if interval is None:
                refuse("неверный интервал")
                continue
            start, end = interval
            # отпечаток — по словам, перекрывающим интервал, уже с ID (их выдаёт и _save_locked): его пересверяет
            # _save_locked; без ID у старого результата первое же сохранение сняло бы отметку («дыра 1», PR 7)
            _ensure_ids(data)
            heard_words = _interval_words(data, start, end)
            # у пропуска проверка — по словам, начатым в паузе: хвост слова до паузы ей не принадлежит (6а)
            checked = [w for _, _, w in pause_words(data, start, end)] if gap_place else heard_words
            # кнопка противоречит словам интервала — свой исход listen_mismatch (К-3/К-4), не «нужна кнопка паузы»
            if gap_place and d["listened"] in ("silence", "interjections") and checked:
                refuse(PAUSE_HAS_WORDS.format(words=" ".join(str(w.get("w", "")) for w in checked)))
                report.setdefault("listen_mismatch", []).append(it["id"])
                continue
            if (d["listened"] == "silence" and checked) or (d["listened"] == "words" and not checked):
                refuse("выбранный ответ противоречит словам стенограммы; сначала исправьте стенограмму")
                report.setdefault("listen_mismatch", []).append(it["id"])
                continue
            it["listening"] = {"by": "human", "result": d["listened"],
                               **({"via": "card"} if d.get("via") == "card" else {}),
                               "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "fingerprint": _listening_fingerprint(heard_words)}
            it["status"] = "resolved"
            report["kept"] += 1
            report["successful_ids"].append(it["id"])
            continue
        if is_gap_place(it) and _place_time(it) is not None and not pause_words(data, *_place_time(it)):
            # В паузе нет слов: подтверждать, подписывать и заменять нечего. Соседнее слово ей не принадлежит.
            report.setdefault("conflicts", []).append({"id": it["id"], "why": _empty_pause_refusal(d, it)})
            report.setdefault("pause_refused", []).append(it["id"])
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
        seen_before = _seen_snapshot((w, seg["speaker"]) for w in ws[w1:w2])
        if by != "human" and ((d.get("text") and d["text"] != before["text"]) or
                              (d.get("speaker") and d["speaker"] != before["speaker"])):
            if any(human_touched(w) for w in ws[w1:w2]):
                report.setdefault("conflicts", []).append({"id": it["id"], "kept_edits": ["human"]})
                _propose_from_review(data, it, d, by, report)
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
                    _mark_provenance(w, "text", by, it["id"], CONTRADICTION if d.get(CONTRADICTION) else None)
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
        seen_changed = _seen_settle(seen_before, [(w, new_spk) for w in ws[w1:w2]]) if seen_before else []
        if seen_changed and by != "human":
            report.setdefault("seen_changed", []).extend(seen_changed)
        edits.append({"id": it["id"], "t": it["t"], "by": by, "before": before,
                      "after": {"text": " ".join(w["w"] for w in (data["segments"][si]["words"] if new_spk == seg["speaker"]
                                                                   else ws)[w1:w2]) if new_text else before["text"],
                                "speaker": new_spk}, "note": d.get("note"),
                      **({"human_note": d["human_note"]} if d.get("human_note") else {}),
                      **({"seen_changed": seen_changed} if seen_changed and by != "human" else {})})
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
    # выбор человека на карточке противоречия (PR 13б): итог — в историю места
    for d in decisions:
        it = items.get(d.get("id"))
        if by == "human" and it and d.get(CONTRADICTION) and it["id"] in report["successful_ids"]:
            if settle_contradiction(it, d[CONTRADICTION], by, d.get("proposal"), d.get("human_note")) \
                    and d.get("human_note"):
                report.setdefault("notes_written", []).append(it["id"])
    # решение применено, но место ещё раз показать человеку (например, ответы двух режимов разошлись)
    for d in decisions:
        it = items.get(d.get("id"))
        if it and d.get("human_after"):
            it["status"], it["note"] = "human", d.get("note") or it.get("note")
            report["human"] += 1
        elif it and d.get("reopen"):                 # ответ отменён — место снова ждёт разбора
            it["status"] = "open"
            it.pop("note", None)
    data["review_done"] = review_finished(data)
    if data != original:                     # без изменений — без новой ревизии и перезаписи файлов
        save(data, json_path, md_path)
    report["open_left"] = open_count(data)
    report["vocab_suggestions"] = data.get("vocab_suggestions", [])
    report["conspects"] = _refresh_conspects(json_path)
    return data, report
