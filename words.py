"""Пословная обработка: распознавание длинными кусками, голос для каждого слова, реплики — после.

Почему так (анализ записи 06.11): при нарезке по границам диаризации (267 кусков, четверть короче
секунды) на стыках терялись слова (139 пропусков), а в 37 сегментах из 224 говорили двое и больше.
Узнавание голоса узким местом не было — им были границы.

Шаги:
  1. куски для распознавания 12–20 с по речи из разметки pyannote; режем в паузах между репликами,
     а внутри сплошной речи — в самом тихом месте, с нахлёстом 0,5 с; слово принадлежит куску,
     в чьей «своей» части оно начинается;
  2. RNN-T (основной текст): слова со временами (шаг 40 мс) и уверенностью (вероятность токена);
  3. CTC по тем же кускам — вторая гипотеза; места расхождения идут в метки сомнения;
  4. голос: окна 1,5 с с шагом 0,25 с, сглаживание по 5 окнам (≈2,5 с звука на слово — лучший
     вариант в опыте анализа), слово оценивается по окну с центром в «начало слова + 0,15 с»;
     опорные голоса — профили участников (или центр кластера, если профиля нет); голосу, который
     на этом месте дал pyannote, — поправка OWN_BONUS;
  5. реплики — подряд идущие слова одного голоса с паузой < 1,5 с; «угу» поверх чужой речи —
     вставка, реплику не рвёт;
  6. метки сомнения: RNN-T ≠ CTC, низкая уверенность слова, малый отрыв первого голоса от второго
     (20 % самых малых), наложение голосов по pyannote, род глагола против пола говорящего.
"""
import bisect
import difflib
import re
import sys
import time

import numpy as np

SR = 16000
CHUNK_MIN, CHUNK_MAX, OVERLAP = 12.0, 20.0, 0.5
# Whisper всегда считает окно 30 с: куски GigaAM (до 20 с) он дополнял тишиной — для него куски длиннее
WHISPER_MIN, WHISPER_MAX = 20.0, 28.0
WIN, HOP, SMOOTH = 1.5, 0.25, 2          # окно, шаг, ±окон сглаживания
WORD_OFFSET = 0.15
OWN_BONUS = 0.05
REPLY_GAP = 1.5
MIN_VOICE_SEC = 3.0                      # голос, набравший меньше и не найденный pyannote, — ошибка
CONF_MIN = 0.5                           # слово с вероятностью ниже — сомнительное
DOUBT_QUANTILE = 0.20                    # доля слов с самым малым отрывом голосов
DOUBT_MAX_MARGIN = 0.10                  # но отрыв больше этого сомнением не считаем
PAIR_BONUS = 0.02      # за каждую поправку человеком этой пары голосов (не больше трёх)
MISSED_MIN = 1.0       # пропущенная речь: от 1 с голоса без слов
BACKCHANNEL = {"угу", "ага", "да", "мм", "м", "ну", "хм", "ммм", "ага-ага", "угу-угу", "да-да", "ясно",
               "понятно", "так", "хорошо", "ой", "о", "а"}
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]")
SAME = {("прям", "прямо"), ("что", "чё"), ("что", "че"), ("чего", "чё"), ("сейчас", "щас"), ("теперь", "щас"),
        ("ну", ""), ("угу", ""), ("а", ""), ("э", ""), ("ээ", ""), ("мм", ""), ("вот", ""), ("окей", "океи"),
        ("какоето", "какойто"), ("тоже", "тоже"), ("еще", "ещё")}


def _log(msg):
    print(f"[words] {msg}", file=sys.stderr, flush=True)


def norm_word(w):
    return re.sub(r"[^0-9a-zа-яё]", "", w.lower().replace("ё", "е"))


# ---------- 1. куски ----------
def speech_regions(tracks, pad=0.15, join=0.3):
    regs = []
    for s, e, _ in sorted(tracks):
        s, e = max(0.0, s - pad), e + pad
        if regs and s - regs[-1][1] < join:
            regs[-1][1] = max(regs[-1][1], e)
        else:
            regs.append([s, e])
    return regs


def _quiet(audio, lo, hi, frame=0.03):
    hop = int(frame * SR)
    a, b = int(lo * SR), int(hi * SR)
    seg = audio[a:b]
    n = len(seg) // hop
    if n < 2:
        return (lo + hi) / 2
    rms = np.sqrt((seg[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return (a + int(rms.argmin()) * hop + hop // 2) / SR


def make_chunks(audio, regions, max_len=CHUNK_MAX, min_len=CHUNK_MIN):
    """-> [{"a": начало звука, "b": конец звука, "own": (с, по)}] — own без нахлёста."""
    dur = len(audio) / SR
    cuts = []  # (own_s, own_e, inside_speech_left, inside_speech_right)
    cur_s, cur_e, left_inside = None, None, False
    for s, e in regions:
        if cur_s is None:
            cur_s, cur_e, left_inside = s, e, False
        elif e - cur_s <= max_len:
            cur_e = e
        else:
            cuts.append((cur_s, cur_e, left_inside, False))
            cur_s, cur_e, left_inside = s, e, False
        while cur_e - cur_s > max_len:       # сплошная речь — режем в тихом месте
            c = _quiet(audio, cur_s + min_len, cur_s + max_len)
            cuts.append((cur_s, c, left_inside, True))
            cur_s, left_inside = c, True
    if cur_s is not None:
        cuts.append((cur_s, cur_e, left_inside, False))
    out = []
    for s, e, li, ri in cuts:
        a = max(0.0, s - (OVERLAP / 2 if li else 0.0))
        b = min(dur, e + (OVERLAP / 2 if ri else 0.0))
        out.append({"a": a, "b": b, "own": (s, e)})
    return out


# ---------- 2–3. распознавание со временами ----------
def result_words(res, offset):
    """TimestampedResult -> [{"w", "s", "e", "c"}]; знаки препинания приклеиваются к слову."""
    toks = res.tokens or []
    ts = res.timestamps or [0.0] * len(toks)
    lp = res.logprobs or [0.0] * len(toks)
    words = []
    space = True   # в словаре GigaAM пробел бывает отдельным токеном: тогда новое слово — следующий токен
    for tok, t, p in zip(toks, ts, lp):
        if not tok.strip():
            space = True
            continue
        new = space or tok.startswith(" ") or not words
        space = tok.endswith(" ")
        piece = tok.strip() if new else tok.rstrip()
        if new and not WORD_RE.search(piece) and words:      # «—», «…» как отдельный токен
            words[-1]["w"] += (" " + piece) if piece in "—–-" else piece
            continue
        if new:
            words.append({"w": piece, "s": offset + t, "e": offset + t + 0.04, "lp": [p]})
        else:
            words[-1]["w"] += piece
            words[-1]["e"] = offset + t + 0.04
            words[-1]["lp"].append(p)
    for i, w in enumerate(words):
        if i + 1 < len(words):
            w["e"] = max(w["e"], min(words[i + 1]["s"], w["e"] + 0.4))
        w["c"] = round(float(np.exp(min(w.pop("lp")))), 3)
        w["s"], w["e"] = round(w["s"], 2), round(w["e"], 2)
    return words


def transcribe(asr, audio, chunks, progress=None):
    pieces = [audio[int(c["a"] * SR):int(c["b"] * SR)] for c in chunks]
    results = asr.transcribe_timestamped(pieces, progress=progress)
    words = []
    for c, r in zip(chunks, results):
        if r is None:
            continue
        s, e = c["own"]
        for w in result_words(r, c["a"]):
            if s - 1e-3 <= w["s"] < e + 1e-3:
                words.append(w)
    words.sort(key=lambda w: w["s"])
    # нахлёст: одно и то же слово на стыке двух кусков
    out = []
    for w in words:
        if out and norm_word(w["w"]) == norm_word(out[-1]["w"]) and w["s"] - out[-1]["s"] < 0.3:
            continue
        out.append(w)
    return out


def mark_alt(main, alt):
    """Сверяет основную гипотезу со второй. Каждому слову main: w["alt"] — текст второй гипотезы
    в месте расхождения (у первого слова расхождения), w["agree"] — совпало ли слово."""
    a = [norm_word(w["w"]) for w in main]
    b = [norm_word(w["w"]) for w in alt]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for w in main:
        w["agree"] = True
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        # CTC склеивает слова («сейчасже»), разговорные варианты («прям/прямо») — не расхождение
        ma, mb = "".join(a[i1:i2]), "".join(b[j1:j2])
        if ma == mb or (ma, mb) in SAME or (mb, ma) in SAME:
            continue
        alt_text = " ".join(x["w"] for x in alt[j1:j2])
        if op == "insert":       # вторая гипотеза слышит слова, которых нет в основной
            k = min(i1, len(main) - 1)
            if k >= 0 and main:
                main[k]["agree"] = False
                main[k].setdefault("alt_ins", alt_text)
            continue
        for k in range(i1, i2):
            main[k]["agree"] = False
        main[i1]["alt"] = alt_text
        main[i1]["alt_span"] = i2 - i1


def _ops(a, b):
    """-> (agree[i] для каждого слова a, список несовпадающих opcodes) с учётом склеек и разговорных вариантов."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    agree = [False] * len(a)
    bad = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i1, i2):
                agree[k] = True
            continue
        ma, mb = "".join(a[i1:i2]), "".join(b[j1:j2])
        if ma == mb or (ma, mb) in SAME or (mb, ma) in SAME:
            for k in range(i1, i2):
                agree[k] = True
            continue
        bad.append((op, i1, i2, j1, j2))
    return agree, bad


def _span_j(bad, i1, i2):
    """Диапазон слов второй гипотезы, соответствующий словам основной [i1, i2)."""
    js = [(j1, j2) for op, a1, a2, j1, j2 in bad if op != "insert" and a1 < i2 and a2 > i1]
    if not js:
        return None
    return min(j for j, _ in js), max(j for _, j in js)


def disagreement_times(main, ctc):
    """Времена, где RNN-T и CTC расходятся (только там Whisper может что-то решить в голосовании)."""
    a = [norm_word(w["w"]) for w in main]
    b = [norm_word(w["w"]) for w in ctc]
    _, bad = _ops(a, b)
    out = []
    for op, i1, i2, j1, j2 in bad:
        out += [w["s"] for w in main[i1:i2]] + [w["s"] for w in ctc[j1:j2]]
    return sorted(out)


# Известные «галлюцинации» Whisper на тишине и шуме (из субтитров, на которых он обучен)
_WH_JUNK = re.compile(r"(?i)\b(?:продолжение следует|субтитры (?:сделал|создавал|подготовил|делал)\w*"
                      r"|редактор субтитров|корректор|dimatorzok|подписывайтесь на (?:наш )?канал"
                      r"|спасибо за просмотр)\b[^.!?…]*[.!?…]*")


def clean_whisper(text, max_repeat=3):
    """Текст Whisper без зацикливания и «субтитров»: фраза из 1–8 слов, повторённая подряд больше
    max_repeat раз, сокращается до max_repeat повторов (0 — без ограничения). -> (текст, сколько слов убрано)."""
    n0 = len((text or "").split())
    toks = _WH_JUNK.sub(" ", text or "").split()
    if max_repeat and max_repeat > 0:
        keys = [norm_word(t) for t in toks]
        out, ko = [], []
        i = 0
        while i < len(toks):
            cut = False
            for n in range(1, 9):
                if len(ko) < n * max_repeat:
                    break
                pat = keys[i:i + n]
                if len(pat) < n or not any(pat):
                    continue
                if all(ko[len(ko) - n * (r + 1):len(ko) - n * r] == pat for r in range(max_repeat)):
                    i += n           # ещё один повтор сверх лимита — пропускаем
                    cut = True
                    break
            if not cut:
                out.append(toks[i])
                ko.append(keys[i])
                i += 1
        toks = out
    return " ".join(toks), n0 - len(toks)


def vote3(main, ctc, whisper):
    """Голосование «два из трёх»: RNN-T (main, со временами), CTC (со временами) и Whisper (только текст).

    Слово RNN-T принимается, если с ним согласна хотя бы одна из двух других моделей. Если обе другие
    согласны между собой и против RNN-T — берётся их вариант (слова и времена CTC), помечается voted.
    Слова, которые слышат обе другие модели, а RNN-T нет, вставляются (времена CTC).
    Где разошлись все три — метка asr и оба варианта (alt — CTC, alt_whisper — Whisper).
    Изменяет main на месте. -> статистика."""
    a = [norm_word(w["w"]) for w in main]
    b = [norm_word(w["w"]) for w in ctc]
    c = [norm_word(w) for w in whisper]
    agB, badB = _ops(a, b)
    agC, badC = _ops(a, c)
    stats = {"agree3": 0, "agree2": 0, "replaced": 0, "inserted": 0, "disputed": 0}
    for w in main:
        w["agree"] = True
    out = []
    # вставки «обе другие слышат, RNN-T — нет»
    insB = {i1: (j1, j2) for op, i1, i2, j1, j2 in badB if op == "insert"}
    insC = {i1: (j1, j2) for op, i1, i2, j1, j2 in badC if op == "insert"}
    i = 0
    while i <= len(main):
        if i in insB and i in insC:
            jb, jc = insB[i], insC[i]
            if "".join(b[jb[0]:jb[1]]) == "".join(c[jc[0]:jc[1]]) and b[jb[0]:jb[1]]:
                for x in ctc[jb[0]:jb[1]]:
                    out.append({**{k: x[k] for k in ("w", "s", "e")}, "c": 1.0, "voted": "ctc+whisper"})
                stats["inserted"] += jb[1] - jb[0]
        if i == len(main):
            break
        if agB[i] or agC[i]:
            stats["agree3" if agB[i] and agC[i] else "agree2"] += 1
            out.append(main[i])
            i += 1
            continue
        k = i
        while k < len(main) and not (agB[k] or agC[k]):
            k += 1
        sb, sc = _span_j(badB, i, k), _span_j(badC, i, k)
        tb = "".join(b[sb[0]:sb[1]]) if sb else ""
        tc = "".join(c[sc[0]:sc[1]]) if sc else ""
        span = main[i:k]
        if sb and sc and tb and tb == tc:          # CTC и Whisper согласны против RNN-T
            model_text = " ".join(w["w"] for w in span)
            for x in ctc[sb[0]:sb[1]]:
                out.append({**{kk: x[kk] for kk in ("w", "s", "e")}, "c": 1.0, "voted": "ctc+whisper",
                            "model": model_text})
            stats["replaced"] += len(span)
        else:
            for w in span:
                w["agree"] = False
            span[0]["alt"] = " ".join(x["w"] for x in ctc[sb[0]:sb[1]]) if sb else ""
            span[0]["alt_whisper"] = " ".join(whisper[sc[0]:sc[1]]) if sc else ""
            span[0]["alt_span"] = len(span)
            stats["disputed"] += len(span)
            out.extend(span)
        i = k
    main[:] = out
    return stats


# ---------- 4. голос по словам ----------
class LabelAt:
    def __init__(self, tracks):
        self.tr = sorted(tracks)
        self.starts = [t[0] for t in self.tr]

    def __call__(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        best = None
        for j in range(max(0, i - 3), min(len(self.tr), i + 2)):
            s, e, l = self.tr[j]
            if s <= t < e:
                return l
            d = min(abs(t - s), abs(t - e))
            if best is None or d < best[0]:
                best = (d, l)
        return best[1] if best and best[0] < 1.0 else None


def overlap_regions(annotation):
    """Где по неисключающей разметке pyannote говорят двое и больше."""
    if annotation is None:
        return []
    ev = []
    for seg, _, _ in annotation.itertracks(yield_label=True):
        ev += [(seg.start, 1), (seg.end, -1)]
    ev.sort()
    out, n, start = [], 0, None
    for t, d in ev:
        n += d
        if n >= 2 and start is None:
            start = t
        elif n < 2 and start is not None:
            if t - start >= 0.1:
                out.append((start, t))
            start = None
    return out


def _score_words(audio, words, idx, R, names, embedder, win, prior, progress=None, confusions=None):
    """Голос для слов words[idx] по скользящим окнам длины win (шаг HOP, сглаживание ±SMOOTH окон)."""
    dur = len(audio) / SR
    gmax = int((dur - win) / HOP)
    need, gi = set(), {}
    for i in idx:
        g = int(round((words[i]["s"] + WORD_OFFSET - win / 2) / HOP))
        g = min(max(g, 0), max(gmax, 0))
        gi[i] = g
        need.update(range(max(0, g - SMOOTH), min(gmax, g + SMOOTH) + 1))
    need = sorted(need)
    vecs = embedder.spans(audio, [(g * HOP, g * HOP + win) for g in need], min_len=win * 0.9, progress=progress)
    S = {g: R @ v for g, v in zip(need, vecs) if v is not None}
    for i in idx:
        w, g = words[i], gi[i]
        rows = [S[k] for k in range(g - SMOOTH, g + SMOOTH + 1) if k in S]
        if not rows:
            w["spk"], w["margin"] = None, 0.0
            continue
        sc = np.mean(rows, axis=0)
        w["sims"] = {n: round(float(x), 3) for n, x in zip(names, sc)}
        p = prior(w["s"] + WORD_OFFSET) if prior else None
        adj = sc + np.array([OWN_BONUS if n == p else 0.0 for n in names])
        order = np.argsort(-adj)
        if confusions and len(names) > 1:
            # пара, которую человек уже поправлял («профиль сказал Вера — это Анна»): склоняемся к
            # подтверждённому человеком и чаще отдаём место на проверку
            a_, b_ = names[order[0]], names[order[1]]
            n_ = confusions.get((a_, b_), 0)
            if n_ and adj[order[0]] - adj[order[1]] < 0.1:
                adj = adj + np.array([PAIR_BONUS * min(n_, 3) if n == b_ else 0.0 for n in names])
                order = np.argsort(-adj)
                w["pair_doubt"] = True
        w["spk"] = names[order[0]]
        w["margin"] = round(float(adj[order[0]] - adj[order[1]]), 3) if len(names) > 1 else 1.0
    return len(need)


def assign_voices(audio, words, refs, embedder, prior=None, info=None, progress=None, win_near_change=0.0,
                  confusions=None):
    """refs: {имя или метка: вектор}; prior(t) -> имя по pyannote или None.
    Пишет в каждое слово spk, margin, sims. -> число окон.
    win_near_change > 0 (например 1.0): слова в пределах 1 с от смены говорящего пересчитываются
    короткими окнами этой длины — у смены длинное окно захватывает два голоса сразу."""
    names = list(refs)
    R = np.stack([refs[n] for n in names])
    n_win = _score_words(audio, words, range(len(words)), R, names, embedder, WIN, prior, progress, confusions)
    if win_near_change and win_near_change < WIN:
        cuts = [(words[k - 1]["e"] + words[k]["s"]) / 2 for k in range(1, len(words))
                if words[k].get("spk") and words[k - 1].get("spk") and words[k]["spk"] != words[k - 1]["spk"]]
        near = [i for i, w in enumerate(words) if any(abs(w["s"] - c) <= 1.0 or abs(w["e"] - c) <= 1.0 for c in cuts)]
        if near:
            n_win += _score_words(audio, words, near, R, names, embedder, float(win_near_change), prior,
                                  confusions=confusions)
            for i in near:
                words[i]["short_win"] = True
            _log(f"у смен говорящего пересчитано окном {win_near_change} с: {len(near)} слов из {len(words)}")
    # голос, которого почти нет и которого не нашёл pyannote, — скорее ошибка: отдаём второму
    if info is not None and "found" in info:
        tot = {}
        for w in words:
            if w.get("spk"):
                tot[w["spk"]] = tot.get(w["spk"], 0.0) + w["e"] - w["s"]
        weak = {n for n, t in tot.items() if t < MIN_VOICE_SEC and n not in info["found"]}
        for w in words:
            if w.get("spk") in weak and w.get("sims"):
                alt = sorted(((v, n) for n, v in w["sims"].items() if n not in weak), reverse=True)
                if alt:
                    w["spk"] = alt[0][1]
                    w["margin"] = 0.0
        if weak:
            _log(f"почти без речи и не найдены pyannote, слова переданы другим: {sorted(weak)}")
    return n_win


# ---------- 5. реплики ----------
def _is_backchannel(ws, prev=None):
    """Короткое «угу / да / ага» поверх чужой речи — поддержка (вставка без своего имени в потоке).
    Но ответ на вопрос («…правда? — Да») несёт свой смысл: это отдельная реплика со своим именем."""
    if prev is not None and str(prev.get("w", "")).rstrip().endswith("?"):
        return False
    return len(ws) <= 2 and all(norm_word(w["w"]) in BACKCHANNEL for w in ws)


def build_replies(words):
    runs = []
    for w in words:
        if w.get("spk") is None:
            w["spk"] = runs[-1]["spk"] if runs else "?"
        if runs and runs[-1]["spk"] == w["spk"] and w["s"] - runs[-1]["words"][-1]["e"] < REPLY_GAP:
            runs[-1]["words"].append(w)
        else:
            runs.append({"spk": w["spk"], "words": [w]})
    # поддакивание поверх чужой речи — вставка; одиночное неуверенное слово между своими — к соседям
    out = []
    i = 0
    while i < len(runs):
        r = runs[i]
        if out and i + 1 < len(runs) and out[-1]["spk"] == runs[i + 1]["spk"] != r["spk"] \
                and r["words"][0]["s"] - out[-1]["words"][-1]["e"] < REPLY_GAP \
                and runs[i + 1]["words"][0]["s"] - r["words"][-1]["e"] < REPLY_GAP:
            if _is_backchannel(r["words"], out[-1]["words"][-1]):
                out[-1].setdefault("inserts", []).append(
                    {"t": r["words"][0]["s"], "speaker": r["spk"], "text": " ".join(w["w"] for w in r["words"])})
                out[-1]["words"] += runs[i + 1]["words"]
                i += 2
                continue
            if len(r["words"]) == 1 and r["words"][0].get("margin", 1) < 0.05:
                r["words"][0]["spk"] = out[-1]["spk"]
                r["words"][0]["voice_doubt"] = True
                out[-1]["words"] += r["words"] + runs[i + 1]["words"]
                i += 2
                continue
        if out and out[-1]["spk"] == r["spk"] and r["words"][0]["s"] - out[-1]["words"][-1]["e"] < REPLY_GAP:
            out[-1]["words"] += r["words"]
        else:
            out.append({**r, "words": list(r["words"])})
        i += 1
    return out


def reply_text(ws):
    return " ".join(w["w"] for w in ws)


# ---------- 6. метки сомнения ----------
def doubt_flags(words, overlaps):
    margins = sorted(w.get("margin", 1.0) for w in words if w.get("spk"))
    thr = min(DOUBT_MAX_MARGIN, margins[int(len(margins) * DOUBT_QUANTILE)] if margins else 0.0)
    ov_s = [s for s, _ in overlaps]
    for w in words:
        fl = []
        if not w.get("agree", True):
            fl.append("asr")
        if w.get("c", 1.0) < CONF_MIN:
            fl.append("conf")
        if w.get("margin", 1.0) <= thr or w.get("voice_doubt") or \
                (w.get("pair_doubt") and w.get("margin", 1.0) <= 2 * max(thr, 0.05)):
            fl.append("voice")
        t = w["s"] + 0.1
        i = bisect.bisect_right(ov_s, t) - 1
        if i >= 0 and overlaps[i][0] <= t < overlaps[i][1] and w.get("margin", 1.0) <= 2 * max(thr, 0.05):
            fl.append("overlap")   # наложение помечаем, только если и голоса при этом близки
        if fl:
            w["flags"] = fl
    return thr


def missed_speech(words, turns, min_gap=MISSED_MIN):
    """Где pyannote слышит речь, а ни одна модель не дала слов: [(начало, конец, метка)]."""
    starts = sorted((w["s"], w["e"]) for w in words)
    out = []
    for t in turns:
        s, e = t["s"], t["e"]
        if e - s < min_gap:
            continue
        cur = s
        for ws, we in starts:
            if we < s or ws > e:
                continue
            if ws - cur >= min_gap:
                out.append((round(cur, 2), round(ws, 2), t.get("label")))
            cur = max(cur, we)
        if e - cur >= min_gap:
            out.append((round(cur, 2), round(e, 2), t.get("label")))
    return out


def review_items(segments, min_gap_words=1, missed=None, name_of=None):
    """Сводит помеченные слова в места для разбора. Слова без флагов между двумя помеченными
    (не больше min_gap_words) входят в то же место."""
    items = []
    for si, seg in enumerate(segments):
        ws = seg.get("words") or []
        k = 0
        while k < len(ws):
            if not ws[k].get("flags"):
                k += 1
                continue
            j = k
            while True:
                nxt = next((m for m in range(j + 1, min(len(ws), j + 2 + min_gap_words)) if ws[m].get("flags")), None)
                if nxt is None:
                    break
                j = nxt
            span = ws[k:j + 1]
            reasons = sorted({f for w in span for f in w.get("flags", [])})
            alt = " ".join(w["alt"] for w in span if w.get("alt")) or None
            alt_wh = " ".join(w["alt_whisper"] for w in span if w.get("alt_whisper")) or None
            ins = " ".join(w["alt_ins"] for w in span if w.get("alt_ins")) or None
            sims = {}
            for w in span:
                for n, v in (w.get("sims") or {}).items():
                    sims[n] = sims.get(n, 0.0) + v / len(span)
            items.append({"id": f"r{len(items) + 1}", "seg": si, "w1": k, "w2": j + 1,
                          "t": span[0]["s"], "t1": span[0]["s"], "t2": span[-1]["s"], "speaker": seg["speaker"],
                          "text": reply_text(span), "alt_ctc": alt, "alt_whisper": alt_wh, "alt_ctc_extra": ins,
                          "reasons": reasons,
                          "voices": dict(sorted(((n, round(v, 2)) for n, v in sims.items()), key=lambda x: -x[1])),
                          "conf_min": round(min(w.get("c", 1.0) for w in span), 2)})
            k = j + 1
        for g in seg.get("gender_flag", []):
            items.append({"id": f"r{len(items) + 1}", "seg": si, "w1": 0, "w2": len(ws), "t": seg["start"],
                          "t1": ws[0]["s"] if ws else seg["start"], "t2": ws[-1]["s"] if ws else seg["start"],
                          "speaker": seg["speaker"], "text": seg["text"], "alt_ctc": None, "reasons": ["gender"],
                          "note": g})
    # пропущенная речь: pyannote слышит голос, а слов нет — спросить человека, что сказано
    for s, e, lbl in missed or []:
        spk = (name_of or {}).get(lbl, lbl)
        items.append({"id": f"r{len(items) + 1}", "t": s, "t1": s, "t2": e, "gap": [s, e], "speaker": spk,
                      "text": "", "alt_ctc": None, "reasons": ["missed"],
                      "note": f"речь {round(e - s, 1)} с без распознанных слов"})
    items.sort(key=lambda it: it["t"])
    for k, it in enumerate(items, 1):
        it["id"] = f"r{k}"
    return items
