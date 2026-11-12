"""Проверка цитат конспекта по стенограмме: дословны ли «…» и тот ли говорящий.

Цитата — текст в «ёлочках». Говорящий берётся, если перед цитатой стоит «Имя:» (или «Имя (…):»).
Составная цитата «А… Б» сверяется по частям; [поправка в квадратных скобках] — допустимое отличие
(так в конспекте отмечаются явные ошибки распознавания) и в счёт не идёт.
Пометки ⟦…⟧ (нерешённые места стенограммы) словами не считаются; если цитата задевает нерешённое
место, в строке отчёта "doubt": [id мест] — такую цитату надо проверить по клипу или дождаться разбора.
«Дословно» — точное совпадение нормализованных слов одной непрерывной части без неоднозначности;
похожесть от 0.6 — «близко», ниже — «не найдено». Скобки и составные цитаты требуют проверки.
"""
import difflib
import re
from collections import Counter

WORD = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+")
QUOTE = re.compile(r"(?:([А-ЯЁ][а-яё]{1,15})\s*(?:\([^)]*\))?\s*[:—-]\s*)?«((?:[^«»]|«[^«»]*»){4,500})»")


def _n(w):
    return w.lower().replace("ё", "е")


def _tokens(data):
    """[(слово, говорящий, время, id нерешённого места | None)] — по словам, если они есть."""
    try:
        import transcript
        marks = transcript.doubt_marks(data) if data.get("review") else {}
    except Exception:
        marks = {}
    out = []
    for si, seg in enumerate(data["segments"]):
        ws = seg.get("words")
        if ws:
            m = marks.get(si) or {}
            for k, w in enumerate(ws):
                for x in WORD.findall(w["w"]):
                    out.append((_n(x), seg["speaker"], w["s"], m.get(k)))
        else:
            for w in WORD.findall(seg["text"]):
                out.append((_n(w), seg["speaker"], seg["start"], None))
    return out


def _best(part, seq, first_index):
    n = len(part)
    best = (0.0, 0, 0)
    starts = set()
    for w in set(part):
        starts.update(first_index.get(w, []))
    for i in sorted(starts):
        for L in (n - 1, n, n + 1, n + 2):
            if L <= 0:
                continue
            r = difflib.SequenceMatcher(None, part, seq[i:i + L], autojunk=False).ratio()
            if r > best[0]:
                best = (r, i, L)
    return best


def candidates(part, toks, speaker=None, min_score=0.6):
    """Лучшие различные совпадения в пределах одного говорящего, без выбора при равенстве."""
    if not part:
        return []
    seq = [t[0] for t in toks]
    scores = []
    n = len(part)
    for i in range(len(seq)):
        if seq[i] not in part:
            continue
        for length in range(max(1, n - 1), n + 3):
            span = toks[i:i + length]
            if len(span) != length or len({t[1] for t in span}) != 1:
                continue
            if speaker and span[0][1] != speaker:
                continue
            score = difflib.SequenceMatcher(None, part, seq[i:i + length], autojunk=False).ratio()
            if score >= min_score:
                scores.append((score, i, length))
    if not scores:
        return []
    best = max(x[0] for x in scores)
    return [(s, i, n) for s, i, n in scores if abs(s - best) < 1e-9]


def check(conspect_text, data, names=None):
    toks = _tokens(data)
    seq = [t[0] for t in toks]
    first = {}
    for i, w in enumerate(seq):
        first.setdefault(w, []).append(i)
    names = set(names or {seg["speaker"] for seg in data["segments"]})
    flat = re.sub(r"\s+", " ", conspect_text)
    rows = []
    for m in QUOTE.finditer(flat):
        spk, q = m.group(1), m.group(2)
        unknown_speaker = spk if spk and names and spk not in names else None
        if unknown_speaker:
            spk = None
        clean = re.sub(r"\[[^\]]*\]", "…", q)              # поправки в скобках — разрыв цитаты
        parts = [[_n(w) for w in WORD.findall(p)] for p in re.split(r"…|\.\.\.", clean)]
        parts = [p for p in parts if p]
        if not parts or sum(len(p) for p in parts) < 2:
            continue
        tot = sum(len(p) for p in parts)
        score, where, spks, doubt = 0.0, None, Counter(), set()
        ambiguous = False
        for p in parts:
            all_candidates = candidates(p, toks)
            named = [x for x in all_candidates if toks[x[1]][1] == spk] if spk else all_candidates
            matches = named or all_candidates
            if not matches:
                continue
            ambiguous = ambiguous or len(matches) > 1
            r, i, L = matches[0]
            score += r * len(p)
            if where is None or len(p) == max(len(x) for x in parts):
                where = (i, L)
            for t in toks[i:i + L]:
                spks[t[1]] += 1
                if t[3]:
                    doubt.add(t[3])
        score /= tot
        i, L = where if where is not None else (0, 0)
        found_spk = spks.most_common(1)[0][0] if spks else None
        rows.append({"quote": q, "speaker": spk, "score": round(score, 2),
                     "verdict": "дословно" if score == 1.0 and len(parts) == 1 and "[" not in q and not ambiguous else ("близко" if score >= 0.6 else "не найдено"),
                     "t": toks[i][2] if toks else 0, "transcript": " ".join(seq[max(0, i - 2):i + L + 2]),
                     "transcript_speaker": found_spk,
                     "speaker_ok": None if not spk else spk == found_spk,
                     **({"doubt": sorted(doubt)} if doubt else {}), **({"ambiguous": True} if ambiguous else {}),
                     **({"unknown_speaker": unknown_speaker} if unknown_speaker else {})})
    summary = {"quotes": len(rows),
               "verbatim": sum(r["verdict"] == "дословно" for r in rows),
               "close": sum(r["verdict"] == "близко" for r in rows),
               "not_found": sum(r["verdict"] == "не найдено" for r in rows),
               "with_speaker": sum(r["speaker"] is not None for r in rows),
               "speaker_mismatch": sum(r["speaker_ok"] is False for r in rows),
               "in_doubt": sum(bool(r.get("doubt")) for r in rows)}
    problems = [r for r in rows if r["verdict"] != "дословно" or r["speaker_ok"] is False or r.get("doubt") or r.get("ambiguous") or r.get("unknown_speaker")]
    return {"summary": summary, "problems": problems}
