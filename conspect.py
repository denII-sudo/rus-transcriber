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
Сборка (render): для каждой ссылки берутся слова стенограммы с t1 ≤ начало ≤ t2.
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
                    out.append((Q._n(x), seg["speaker"], w["s"]))
        else:
            toks = Q.WORD.findall(seg["text"])
            n = max(1, len(toks))
            for k, x in enumerate(toks):
                out.append((Q._n(x), seg["speaker"], round(seg["start"] + (seg["end"] - seg["start"]) * k / n, 3)))
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


def link(conspect_md, data, min_score=0.75):
    """Заменяет «цитаты» ссылками на стенограмму. -> (объект конспекта, отчёт)."""
    toks = _tokens(data)
    seq = [t[0] for t in toks]
    first = {}
    for i, w in enumerate(seq):
        first.setdefault(w, []).append(i)
    names = {seg["speaker"] for seg in data["segments"]}
    quotes, unlinked = {}, []
    out, pos, n = [], 0, 0
    for m in Q.QUOTE.finditer(conspect_md):
        spk, q = m.group(1), m.group(2)
        if spk and spk not in names:
            spk = None
        clean = re.sub(r"\[[^\]]*\]", "…", q)
        raw_parts = [p for p in re.split(r"…|\.\.\.", clean)]
        parts, ok = [], True
        for p in raw_parts:
            toks_p = [Q._n(w) for w in Q.WORD.findall(p)]
            if not toks_p:
                continue
            r, i, L = Q._best(toks_p, seq, first)
            if r < min_score or L <= 0:
                ok = False
                break
            parts.append({"t1": toks[i][2], "t2": toks[min(i + L, len(toks)) - 1][2]})
        if not ok or not parts:
            unlinked.append(q)
            continue
        n += 1
        key = str(n)
        t1, t2 = parts[0]["t1"], parts[-1]["t2"]
        base, base_spk = _span_text(data, t1, t2)
        joined = " … ".join(filter(None, (_span_text(data, p["t1"], p["t2"])[0] for p in parts)))
        quotes[key] = {"parts": parts, "author": q, "speaker": spk, "base": joined, "base_speaker": base_spk,
                       "last": joined, "last_speaker": base_spk}
        # «Имя: «…»» — имя тоже часть ссылки, чтобы обновлялось вместе с говорящим
        start = m.start(1) if spk else m.start(2) - 1
        out.append(conspect_md[pos:start])
        out.append(f"{{{{q:{key}}}}}")
        pos = m.end()
    out.append(conspect_md[pos:])
    obj = {"version": 1, "template": "".join(out), "quotes": quotes}
    return obj, {"linked": n, "unlinked": unlinked}


def render(obj, data, update_last=True):
    """-> (markdown, отчёт). Отчёт: что обновилось из стенограммы с прошлой сборки и с момента связывания."""
    rep = {"changed_since_last": [], "changed_since_link": 0, "speaker_changed": [], "missing": []}

    def sub(m):
        key = m.group(1)
        q = obj["quotes"].get(key)
        if not q:
            return m.group(0)
        texts = [_span_text(data, p["t1"], p["t2"]) for p in q["parts"]]
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
    return {"conspect_json": str(cj), "conspect_md": str(cm), **rep, "check": check}


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
