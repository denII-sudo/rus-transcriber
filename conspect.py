"""Конспект, связанный со стенограммой: цитаты — ссылки на места стенограммы, текст подставляется из неё.

Файл конспекта — <имя записи>.<название>.conspect.json в подпапке результатов:
{
  "name": "конспект", "job_json": "...transcript.json",
  "template": "markdown с {{q:1}} на местах цитат",
  "quotes": {"1": {"parts": [{"t1": 12.34, "t2": 15.6}], "author": "текст как в конспекте",
                   "speaker": "Вера" | null,               — имя перед цитатой («Вера: «…»»)
                   "base": "текст стенограммы при связывании", "base_speaker": "...",
                   "last": "текст при последней сборке", "last_speaker": "..."}}
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
from collections import Counter
from pathlib import Path

import quotes as Q

PLACE = re.compile(r"\{\{q:(\d+)\}\}")


def _tokens(data):
    """[(слово, говорящий, время начала слова)] — по словам, а для старых расшифровок — равномерно по реплике."""
    out = []
    for seg in data["segments"]:
        ws = seg.get("words")
        if ws:
            for w in ws:
                toks = Q.WORD.findall(w["w"])
                for x in toks:
                    out.append((Q._n(x), seg["speaker"], w["s"], w.get("word_id")))
        else:
            toks = Q.WORD.findall(seg["text"])
            n = max(1, len(toks))
            for k, x in enumerate(toks):
                out.append((Q._n(x), seg["speaker"], round(seg["start"] + (seg["end"] - seg["start"]) * k / n, 3), None))
    return out


def _span_text(data, t1, t2):
    """Текст и говорящий слов с t1 ≤ начало ≤ t2 (по словам или по реплике целиком)."""
    words, spk = [], {}
    for seg in data["segments"]:
        ws = seg.get("words")
        if ws:
            for w in ws:
                if t1 - 0.005 <= w["s"] <= t2 + 0.005:
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


def _id_text(data, ids):
    """Разрешить точные ID, не прибегая к временным окнам или порядку хранения."""
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
            if t1 - 0.005 <= word["s"] <= t2 + 0.005:
                key = round(word["s"], 3)
                if key in seen:
                    return None, None
                seen.add(key)
                speakers.add(seg["speaker"])
    if len(speakers) > 1 or legacy_segments > 1 or (legacy_segments and seen):
        return None, None
    return _span_text(data, t1, t2)


def link(conspect_md, data, min_score=0.75, selections=None):
    """Связывает только однозначные точные цитаты. min_score оставлен для совместимости вызовов."""
    toks = _tokens(data)
    seq = [t[0] for t in toks]
    first = {}
    for i, w in enumerate(seq):
        first.setdefault(w, []).append(i)
    names = {seg["speaker"] for seg in data["segments"]}
    id_counts = Counter(w.get("word_id") for seg in data["segments"] for w in seg.get("words") or [])
    quotes, unlinked, candidates_report = {}, [], []
    selections = selections or {}
    consumed = set()
    out, pos, n = [], 0, 0
    for quote_index, m in enumerate(Q.QUOTE.finditer(conspect_md), 1):
        spk, q = m.group(1), m.group(2)
        if spk and spk not in names:
            unlinked.append(q)
            continue
        clean = re.sub(r"\[[^\]]*\]", "…", q)
        raw_parts = [p for p in re.split(r"…|\.\.\.", clean)]
        parts, ok = [], True
        # До отдельной реализации согласованного выбора частей не связывать
        # разорванную цитату с независимо найденными разными местами.
        if len([p for p in raw_parts if Q.WORD.findall(p)]) != 1 or not data.get("recording_id"):
            unlinked.append(q)
            continue
        for p in raw_parts:
            toks_p = [Q._n(w) for w in Q.WORD.findall(p)]
            if not toks_p:
                continue
            # Приближённое совпадение показывается проверяющему, но не становится ссылкой автоматически.
            matches = Q.candidates(toks_p, toks, speaker=spk, min_score=1.0)
            options = []
            for _, ci, cl in matches[:21]:
                selected_ids = list(dict.fromkeys(t[3] for t in toks[ci:ci + cl]))
                if None not in selected_ids and all(id_counts[wid] == 1 for wid in selected_ids):
                    options.append({"word_ids": selected_ids, "speaker": toks[ci][1],
                                    "t1": toks[ci][2], "t2": toks[ci + cl - 1][2]})
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
                break
            r, i, L = matches[0]
            ids = list(dict.fromkeys(t[3] for t in toks[i:i + L]))
            if None in ids or any(id_counts[wid] != 1 for wid in ids):
                ok = False
                break
            parts.append({"t1": toks[i][2], "t2": toks[i + L - 1][2], "word_ids": ids})
        if not ok or not parts:
            unlinked.append(q)
            continue
        n += 1
        key = str(n)
        t1, t2 = parts[0]["t1"], parts[-1]["t2"]
        base, base_spk = _id_text(data, parts[0]["word_ids"])
        joined = " … ".join(filter(None, (_id_text(data, p["word_ids"])[0] for p in parts)))
        quotes[key] = {"parts": parts, "author": q, "speaker": spk, "base": joined, "base_speaker": base_spk,
                       "last": joined, "last_speaker": base_spk}
        # «Имя: «…»» — имя тоже часть ссылки, чтобы обновлялось вместе с говорящим
        start = m.start(1) if spk else m.start(2) - 1
        out.append(conspect_md[pos:start])
        out.append(f"{{{{q:{key}}}}}")
        pos = m.end()
    out.append(conspect_md[pos:])
    if set(selections) != consumed:
        raise ValueError("часть выбранных цитат отсутствует или не может быть связана")
    obj = {"version": 2, "template": "".join(out), "quotes": quotes,
           "recording_id": data.get("recording_id"), "source_revision": data.get("source_revision"),
           "analysis_revision": None}
    return obj, {"linked": n, "unlinked": unlinked, "candidates": candidates_report,
                 "recording_id": data.get("recording_id"), "source_revision": data.get("source_revision")}


def render(obj, data, update_last=True):
    """-> (markdown, отчёт). Отчёт: что обновилось из стенограммы с прошлой сборки и с момента связывания."""
    rep = {"changed_since_last": [], "changed_since_link": 0, "speaker_changed": [], "missing": []}
    rep["analysis_needs_review"] = (obj.get("analysis_revision") is None or data.get("source_revision") is None or
                                    obj.get("analysis_revision") != data.get("source_revision") or
                                    not obj.get("recording_id") or obj.get("recording_id") != data.get("recording_id"))
    source_conflict = bool(obj.get("recording_id") and obj.get("recording_id") != data.get("recording_id"))
    if source_conflict:
        rep["source_conflict"] = "конспект относится к другой записи"
    if not obj.get("recording_id"):
        rep["legacy_unverified"] = True

    def sub(m):
        key = m.group(1)
        q = obj["quotes"].get(key)
        if not q:
            return m.group(0)
        texts = [(None, None) if source_conflict else _id_text(data, p["word_ids"]) if "word_ids" in p
                 else _legacy_span(data, p["t1"], p["t2"]) for p in q["parts"]]
        if any(t is None for t, _ in texts):
            rep["missing"].append({"id": key, "author": q["author"]})
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
        return (f"{spk}: " if spk else "") + f"«{body}»"

    md = PLACE.sub(sub, obj["template"])
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
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    obj, rep = link(conspect_md, data)
    obj.update(name=name, job_json=str(json_path))
    cj, cm = path_for(json_path, name)
    md, rrep = render(obj, data)
    cj.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    cm.write_text(md, encoding="utf-8")
    check = Q.check(md, data)["summary"]
    return {"conspect_json": str(cj), "conspect_md": str(cm), **rep,
            "analysis_needs_review": rrep["analysis_needs_review"], "check": check}


def rerender(json_path, name=None):
    """Пересобирает конспект(ы) записи из текущей стенограммы. -> {название: отчёт}."""
    jp = Path(json_path)
    data = json.loads(jp.read_text(encoding="utf-8"))
    stem = jp.name[:-len(".transcript.json")] if jp.name.endswith(".transcript.json") else jp.stem
    out = {}
    for cj in sorted(jp.parent.glob(f"{stem}.*.conspect.json")):
        obj = json.loads(cj.read_text(encoding="utf-8"))
        if name and obj.get("name") != name:
            continue
        md, rep = render(obj, data)
        cj.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        cm = Path(str(cj)[:-len(".json")] + ".md")
        cm.write_text(md, encoding="utf-8")
        chk = Q.check(md, data)
        out[obj.get("name", cj.name)] = {"conspect_md": str(cm), **rep, "check": chk["summary"],
                                         "check_problems": chk["problems"][:20], "markdown_chars": len(md)}
    return out
