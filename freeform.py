"""Свободные ответы человека в карточках разбора: ответ — правка того, что человек услышал в клипе.

Порядок старшинства: ответ человека > отпечаток голоса > распознавание (RNN-T/CTC/Whisper) > эвристики.

Для каждой карточки со свободным текстом:
  1. вид ответа: только имя («Борис», «не Анна, а Вера») / согласие («да», «верно») / правка текста;
  2. правка разбирается на реплики по меткам «Имя:» (или имя в начале фразы перед словом с заглавной);
     «…» в конце — «дальше без изменений»;
  3. окно стенограммы — все слова во времени клипа ±1 с;
  4. локальное выравнивание (Смит — Уотерман) слов ответа со словами окна; совпадение с вариантом другой
     модели (alt, alt_whisper) тоже опора; опор меньше 30 % слов ответа — «не удалось сопоставить»;
  5. правки слов: смена говорящего, замена (в том числе написание: «2.19.3» вместо «219.3»), вставка;
     вставка без «Имя:» — говорящего, чьи слова в ответе стоят перед ней (в начале ответа — после неё);
     слово модели внутри реплики человека, которого он не назвал: того же говорящего — удаляется,
     чужое — наложение речи, переносится после реплики целиком с пометкой; на стыке реплик — только
     предложение удалить; повтор (вставленное слово уже есть рядом у модели) — слово модели удаляется;
     правку нельзя применить целиком (неизвестен говорящий) — не применяется, место человеку;
     ответ шире окна — окно расширяется до всей фразы человека;
  6. опечатки: только обычные строчные кириллические слова — слово ответа, которого нет в словаре, а у модели
     в этом месте слово на 1–2 буквы иное, — берётся слово модели («приколько» → «прикольно»); изменение
     окончания принимается как есть; имена, названия, КАПС, латиница, числа, версии — как написал человек;
  7. карточки с пересекающимися окнами — по времени; две разные правки одного слова — побеждает карточка,
     чьё место покрывает слово, иначе более поздний ответ; конфликт — в отчёт;
     места других карточек, покрытые ответом, закрываются «решено человеком (через rN)».
Результат — правки слов для transcript.apply_word_edits.
"""
import itertools
import re

PAD_BEFORE, PAD_AFTER, SLACK = 3.5, 1.5, 1.0
GROW, MAX_GROW = 4.0, 2          # расширение окна, если ответ шире него (с), и сколько раз
DUP_SEC = 2.0                    # повтор: то же слово модели не дальше этого от вставки
MIN_ANCHORS = 0.3
AGREE = {"да", "верно", "так", "правильно", "все верно", "всё верно", "да верно", "ага", "угу", "так и есть",
         "как есть", "да так"}
WORD = re.compile(r"[\wЁё][\wЁё'-]*", re.U)

_MORPH = None


def _morph():
    global _MORPH
    if _MORPH is None:
        try:
            import pymorphy3
            _MORPH = pymorphy3.MorphAnalyzer()
        except Exception:
            _MORPH = False
    return _MORPH


def norm(w):
    return re.sub(r"[^\wё-]", "", (w or "").lower().replace("ё", "е")).strip("-")   # ⟦…⟧ тоже убирается


def tokens(text):
    """Слова с исходной пунктуацией вокруг: «Ой,» «извините,» — для вывода; norm — для сравнения."""
    out = []
    for raw in (text or "").replace("⟦", "").replace("⟧", "").split():   # пометки сомнения — не слова
        n = norm(raw)
        if n:
            out.append(raw)
    return out


def lev(a, b):
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 3:
        return 9
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def ending_only(a, b):
    """Отличие только в окончании (род, число, время): общая основа, хвосты до 3 букв."""
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k >= 3 and len(a) - k <= 3 and len(b) - k <= 3


ENDINGS = {"", "а", "я", "о", "е", "и", "ы", "у", "ю", "ь", "й", "л", "ла", "ло", "ли", "ть", "ти", "ем", "им",
           "ешь", "ишь", "ет", "ит", "ют", "ят", "ут", "ат", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ий", "ый",
           "ого", "его", "ому", "ему", "ом", "ам", "ям", "ями", "ами", "ах", "ях", "ов", "ев", "ей", "ся", "сь",
           "лась", "лся", "лось", "лись", "ал", "ала", "али", "ял", "яла", "яли", "ил", "ила", "или", "ел", "ела",
           "ели", "ыл", "ыла", "ыли", "ул", "ула", "ули", "ешься", "ется", "ишься", "ится", "ую", "юю"}


def inflection_only(a, b):
    """Отличие только окончанием-формой слова (заметила → заметил, стоим → стояли)."""
    m = _morph()
    if m:
        try:
            la = {p.normal_form for p in m.parse(a)}
            lb = {p.normal_form for p in m.parse(b)}
            if la & lb:
                return True
        except Exception:
            pass
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    for cut in range(k, max(2, k - 2), -1):
        if a[cut:] in ENDINGS and b[cut:] in ENDINGS and cut >= 3:
            return True
    return False


def similar(a, b):
    return a == b or ending_only(a, b) or (len(a) > 3 and len(b) > 3 and lev(a, b) <= max(1, min(len(a), len(b)) // 4))


# ---------- 1–2. вид ответа и реплики ----------
def classify(text, names):
    t = " ".join(norm(x) for x in (text or "").split() if norm(x))
    if not t:
        return {"kind": "skip"}
    if t in AGREE:
        return {"kind": "agree"}
    ws = t.split()
    low = {norm(n): n for n in names}
    named = [low[w] for w in ws if w in low]
    rest = [w for w in ws if w not in low and w not in {"это", "не", "а", "говорит", "был", "была", "тут", "здесь"}]
    if named and not rest:
        return {"kind": "name", "speaker": named[-1]}      # «не Анна, а Вера» — последнее имя
    return {"kind": "text"}


def split_replies(text, names):
    """-> [(говорящий | None, [слова])]. Метка — «Имя:» или имя в начале фразы перед словом с заглавной."""
    toks = (text or "").replace("…", "... ").split()
    nameset = {norm(n): n for n in names}
    out, cur, spk = [], [], None
    for i, raw in enumerate(toks):
        n = norm(raw)
        is_label = False
        if n in nameset:
            colon = raw.rstrip().endswith(":")
            prev_end = i == 0 or re.search(r"[.!?…:]$", toks[i - 1]) is not None
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if colon or (prev_end and nxt[:1].isupper() and not raw.rstrip().endswith(",")):
                is_label = True
        if is_label:
            if cur:
                out.append((spk, cur))
            spk, cur = nameset[n], []
            continue
        if re.fullmatch(r"\.{2,}|…", raw.strip()):
            continue
        raw = re.sub(r"(\.{3}|…)$", "", raw)
        if norm(raw):
            cur.append(raw)
    if cur:
        out.append((spk, cur))
    return out


# ---------- 3–4. окно и выравнивание ----------
def window(words, t1, t2):
    a, b = t1 - PAD_BEFORE - SLACK, t2 + PAD_AFTER + SLACK
    return [w for w in words if a <= w["s"] <= b]


def _alts(w):
    s = set()
    for k in ("alt", "alt_whisper", "model", "raw", "alt_place"):
        for x in str(w.get(k) or "").split():
            if norm(x):
                s.add(norm(x))
    return s


def align(ans, win):
    """Смит — Уотерман. ans: [нормы]; win: [слова окна]. -> пары [(i | None, j | None, вид)], где вид:
    match (то же слово или вариант модели), sub (замена), ins (только в ответе), del (только в стенограмме)."""
    n, m = len(ans), len(win)
    wn = [norm(w["w"]) for w in win]
    al = [_alts(w) for w in win]
    MATCH, NEAR, MIS, GAP = 3, 1, -1, -1

    def sc(i, j):
        a = ans[i]
        if a == wn[j] or a in al[j]:
            return MATCH
        if similar(a, wn[j]) or any(similar(a, x) for x in al[j]):
            return NEAR
        return MIS
    H = [[0] * (m + 1) for _ in range(n + 1)]
    best, bi, bj = 0, 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            v = max(0, H[i - 1][j - 1] + sc(i - 1, j - 1), H[i - 1][j] + GAP, H[i][j - 1] + GAP)
            H[i][j] = v
            if v > best:
                best, bi, bj = v, i, j
    pairs, i, j = [], bi, bj
    while i > 0 and j > 0 and H[i][j] > 0:
        s = sc(i - 1, j - 1)
        if H[i][j] == H[i - 1][j - 1] + s:
            pairs.append((i - 1, j - 1, "match" if s == MATCH else "sub"))
            i, j = i - 1, j - 1
        elif H[i][j] == H[i - 1][j] + GAP:
            pairs.append((i - 1, None, "ins"))
            i -= 1
        else:
            pairs.append((None, j - 1, "del"))
            j -= 1
    pairs.reverse()
    # слова ответа до и после выровненного куска — тоже правка (вставка у края), если их немного
    lead = [(k, None, "ins") for k in range(0, i)]
    tail = [(k, None, "ins") for k in range(bi, n)]
    return lead + pairs + tail


# ---------- 6. опечатки ----------
def _known(word, corpus):
    m = _morph()
    if m:
        try:
            return m.word_is_known(word)
        except Exception:
            pass
    return word in corpus


def core_of(raw):
    """Слово без пунктуации по краям; внутренняя (точки версий, дефисы) сохраняется: «2.19.3,» → «2.19.3»."""
    return re.sub(r"^[^\wЁё]+|[^\wЁё]+$", "", raw or "")


def human_spelling(human, model):
    """Слово модели в написании человека: основа — как у человека; знак после слова — человека, если он его
    поставил, иначе модели (человек в карточке часто пишет без знаков)."""
    hc, mc = core_of(human), core_of(model)
    if not hc or not mc:
        return model
    out = model.replace(mc, hc, 1)
    tail_h = human[len(human.rstrip(".,!?…;:»)\"")):] if human else ""
    if tail_h:
        out = out[:len(out) - (len(out) - len(out.rstrip(".,!?…;:»)\"")))] + tail_h
    return out


TYPO_OK = re.compile(r"[а-яё]+(?:-[а-яё]+)*")


def fix_typo(raw, w, corpus, vocab_words):
    """-> (слово, пометка | None). Слово человека, которого нет в словаре, при варианте модели на 1–2 буквы.
    Только обычные слова: строчные кириллические. Слова с заглавной буквы, КАПСОМ, латиницей, с цифрами,
    номера версий и слова из словаря группы человек написал сознательно — они не «исправляются»."""
    a = norm(raw)
    if not a or a in vocab_words or not TYPO_OK.fullmatch(core_of(raw)) or _known(a, corpus):
        return raw, None
    cands = [w["w"]] + [x for k in ("alt", "alt_whisper") for x in str(w.get(k) or "").split()]
    for c in cands:
        cn = norm(c)
        if cn and cn != a and not inflection_only(a, cn) and lev(a, cn) <= 2:
            out = re.sub(re.escape(re.sub(r"[^\wЁё-]", "", raw)), re.sub(r"[^\wЁё-]", "", c), raw, count=1) \
                if re.sub(r"[^\wЁё-]", "", raw) else c
            return out, f"опечатка «{re.sub(r'[^\wЁё-]', '', raw)}», исправлено по модели"
    return raw, None


# ---------- 5. правки по одной карточке ----------
def _place_span(place, ans_raw):
    """Ответ без опор («Пришли»): какой кусок места он заменяет — самый похожий по буквам."""
    import difflib
    target = "".join(norm(x) for x in ans_raw)
    best, span = -1.0, (0, len(place))
    for i in range(len(place)):
        for j in range(i + 1, len(place) + 1):
            s = "".join(norm(w["w"]) for w in place[i:j])
            r = difflib.SequenceMatcher(None, s, target, autojunk=False).ratio()
            if r > best + 1e-9 or (abs(r - best) <= 1e-9 and j - i > span[1] - span[0]):
                best, span = r, (i, j)
    return span


def card_edits(item, answer, words, names, corpus=frozenset(), vocab_words=frozenset()):
    """item — элемент manifest (id, t1, t2, text, speaker); answer — ответ (text, ts).
    -> {"kind", "edits": [...], "covered": (t_from, t_to), "anchors", "report": {...}}"""
    text = answer.get("text") or ""
    cid = item["id"]
    win = window(words, item["t1"], item["t2"])
    place = [w for w in words if item["t1"] - 0.01 <= w["s"] <= item["t2"] + 0.01]
    extra = " ".join(str(item.get(k) or "") for k in ("alt_ctc", "alt_whisper"))
    for w in place:                       # варианты моделей по месту — тоже опоры
        w["alt_place"] = extra
    c = classify(text, names)
    rep = {"id": cid, "answer": text, "kind": c["kind"], "applied": [], "inserted": [], "typos": [],
           "proposed_delete": [], "deleted": [], "overlap": [], "human": None}
    if c["kind"] == "agree":
        return {"kind": "agree", "edits": [], "covered": None, "report": rep}
    if c["kind"] == "name":
        eds = [{"op": "relabel", "t": w["s"], "w": w["w"], "speaker": c["speaker"], "card": cid, "in_place": True}
               for w in place if w.get("_spk") != c["speaker"]]
        rep["applied"] = [f"{' '.join(w['w'] for w in place)} → {c['speaker']}"]
        span = (place[0]["s"], place[-1]["s"]) if place else None
        return {"kind": "name", "edits": eds, "covered": span, "report": rep}
    replies = split_replies(text, names)
    if "missed" in (item.get("reasons") or []) and replies:
        # пропущенная речь: слов модели нет — весь ответ вставляется в этот промежуток
        t_a, t_b = float(item["t1"]), float(item["t2"])
        flat = [(spk or item.get("speaker"), x) for spk, ws in replies for x in ws]
        if any(not who for who, _ in flat):
            rep["kind"] = "missed"
            rep["human"] = "не удалось определить говорящего пропущенной речи (в ответе нет «Имя:»)"
            return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
        step = (t_b - t_a) / (len(flat) + 1)
        eds = []
        for q, (spk, x) in enumerate(flat, 1):
            s = round(t_a + step * q, 3)
            eds.append({"op": "insert", "s": s, "e": round(s + step * 0.9, 3), "text": x, "speaker": spk,
                        "card": cid, "in_place": True})
        rep["kind"] = "missed"
        rep["inserted"] = [f"{spk}: {' '.join(ws)}" for spk, ws in replies]
        return {"kind": "missed", "edits": eds, "covered": (t_a, t_b), "report": rep}
    ans_raw, ans_spk = [], []
    last = None
    for spk, ws in replies:
        if spk:
            last = spk
        for x in ws:
            ans_raw.append(x)
            ans_spk.append(spk or last)
    ans = [norm(x) for x in ans_raw]
    if not ans:
        return {"kind": "skip", "edits": [], "covered": None, "report": rep}
    pairs = align(ans, win)
    # ответ шире окна: слова ответа до (после) выровненного куска, а окно кончается у первой (последней) опоры —
    # окно расширяется, чтобы правка применялась ко всей фразе человека, а не только к месту разбора
    grow_a = grow_b = 0.0
    for _ in range(MAX_GROW):
        lead = sum(1 for _ in itertools.takewhile(lambda p: p[1] is None, pairs))
        tail = sum(1 for _ in itertools.takewhile(lambda p: p[1] is None, reversed(pairs)))
        js_ = [j for _, j, _ in pairs if j is not None]
        more_a = lead and js_ and js_[0] <= 1 and win and words and win[0] is not words[0] and \
            any(w["s"] < win[0]["s"] for w in words)
        more_b = tail and js_ and js_[-1] >= len(win) - 2 and win and any(w["s"] > win[-1]["s"] for w in words)
        if not (more_a or more_b):
            break
        grow_a += GROW if more_a else 0.0
        grow_b += GROW if more_b else 0.0
        win = window(words, item["t1"] - grow_a, item["t2"] + grow_b)
        pairs = align(ans, win)
    anchors = sum(1 for i, j, k in pairs if k == "match")
    labelled = any(s for s, _ in replies)
    # реплика с меткой, у которой нет ни одной опоры, — вставка пропущенной речи, в долю опор не входит
    rep_of = []
    for q, (spk, ws) in enumerate(replies):
        rep_of += [q] * len(ws)
    anchored_reps = {rep_of[i] for i, j, k in pairs if k == "match" and i is not None}
    countable = [i for i in range(len(ans)) if not (replies[rep_of[i]][0] and rep_of[i] not in anchored_reps)]
    in_place_j = {k for k, w in enumerate(win) if item["t1"] - 0.01 <= w["s"] <= item["t2"] + 0.01}
    anchored_j = {j for i, j, k in pairs if k == "match"}
    near_place = {k for k in in_place_j} | {k - 1 for k in in_place_j} | {k + 1 for k in in_place_j}
    if not labelled and place and (not anchored_j or anchored_j <= near_place) and len(ans) <= len(place) + 4 \
            and not (grow_a or grow_b):
        # ответ без меток про само место — новый текст места; без опор — самого похожего его куска
        if anchored_j:
            pids = [w["s"] for w in place]
            idx = sorted(pids.index(win[j]["s"]) for j in anchored_j if win[j]["s"] in pids)
            a_, b_ = (idx[0], idx[-1] + 1) if idx else (0, len(place))
            if len(ans) > b_ - a_:          # слов в ответе больше, чем покрыто опорами, — до краёв места
                a_, b_ = 0, len(place)
        else:
            a_, b_ = _place_span(place, ans_raw)
        sub = place[a_:b_]
        rep["kind"] = "place_text"
        rep["applied"] = [f"«{' '.join(w['w'] for w in sub)}» → «{' '.join(ans_raw)}»"]
        return {"kind": "place_text", "edits": [{"op": "place_text", "card": cid, "t1": sub[0]["s"],
                                                 "t2": sub[-1]["s"], "text": " ".join(ans_raw), "in_place": True}],
                "covered": (sub[0]["s"], sub[-1]["s"]), "report": rep}
    if not countable or sum(1 for i, j, k in pairs if k == "match" and i in set(countable)) < MIN_ANCHORS * len(countable) \
            or not anchored_reps:
        rep["human"] = "не удалось сопоставить ответ со стенограммой"
        return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
    edits = []
    in_place = lambda w: item["t1"] - 0.01 <= w["s"] <= item["t2"] + 0.01
    js = [j for i, j, k in pairs if j is not None]
    covered = (win[js[0]]["s"], win[js[-1]]["s"]) if js else None
    # опоры: пары «слово ответа — слово стенограммы»; говорящий реплики ответа — метка человека, а без меток —
    # говорящий слова стенограммы под опорой
    anch = [(n, i, j) for n, (i, j, k) in enumerate(pairs) if k in ("match", "sub")]
    rspk = {i: (ans_spk[i] or win[j]["_spk"]) for _, i, j in anch}
    paired_j = {j for _, j, _ in pairs if j is not None}
    problems = []

    def neighbours(n):
        """Ближайшие опоры до и после позиции n в выравнивании: (i, j) или None."""
        p = next(((i, j) for m, i, j in reversed(anch) if m < n), None)
        q = next(((i, j) for m, i, j in anch if m > n), None)
        return p, q

    def reply_end_t(i):
        """Время последнего слова стенограммы под опорой той же реплики ответа (для переноса наложения)."""
        same = [j for _, ii, j in anch if rep_of[ii] == rep_of[i] and rspk[ii] == rspk[i]]
        return win[max(same)]["s"] if same else None

    pending = []          # [(слово ответа, говорящий | None)]
    prev_j = None

    def flush(next_j):
        if not pending:
            return
        # говорящий вставки без «Имя:» — того, чьи слова стоят в ответе непосредственно перед ней; в начале
        # ответа — того, чьи слова идут сразу после; иначе определить нельзя
        spk_prev = rspk.get(next((i for m, i, j in reversed(anch) if j == prev_j), None)) if prev_j is not None else None
        spk_next = rspk.get(next((i for m, i, j in anch if j == next_j), None)) if next_j is not None else None
        t_a = win[prev_j]["e"] if prev_j is not None else max(0.0, win[next_j]["s"] - 0.05 * (len(pending) + 1))
        for q, (raw, spk) in enumerate(pending, 1):
            who = spk or spk_prev or spk_next
            if not who:
                problems.append(f"не удалось определить говорящего вставки «{raw}»")
            s = round(t_a + 0.01 * q, 3)
            edits.append({"op": "insert", "s": s, "e": s, "text": raw, "speaker": who, "card": cid,
                          "after_t": win[prev_j]["s"] if prev_j is not None else None,
                          "before_t": win[next_j]["s"] if next_j is not None else None, "in_place": False})
            rep["inserted"].append(f"{who or '?'}: {raw}")
            # повтор: такое же слово модели того же говорящего рядом, ни с чем не сопоставленное, — правка его
            # покрывает, в аудио оно одно; слово модели удаляется
            for jj, ww in enumerate(win):
                if jj not in paired_j and norm(ww["w"]) == norm(raw) and len(norm(raw)) > 1 and \
                        ww["_spk"] == who and abs(ww["s"] - t_a) <= DUP_SEC:
                    paired_j.add(jj)
                    edits.append({"op": "delete", "t": ww["s"], "w": ww["w"], "card": cid, "in_place": in_place(ww),
                                  "note": "повтор: слово модели покрыто правкой человека"})
                    rep["deleted"].append(ww["w"])
                    break
        pending.clear()

    for n, (i, j, k) in enumerate(pairs):
        if k == "ins":
            pending.append((ans_raw[i], ans_spk[i]))
            continue
        if k == "del":
            w = win[j]
            p, q = neighbours(n)
            if p is None or q is None:
                continue                               # за краем ответа — не трогаем
            same_reply = rep_of[p[0]] == rep_of[q[0]] and rspk[p[0]] == rspk[q[0]]
            if same_reply and w["_spk"] == rspk[p[0]]:
                # внутри реплики человека, тот же говорящий, в ответе слова нет — удаляется
                edits.append({"op": "delete", "t": w["s"], "w": w["w"], "card": cid, "in_place": in_place(w),
                              "note": "человек этого слова не назвал"})
                rep["deleted"].append(w["w"])
            elif same_reply:
                # чужое слово внутри реплики человека — наложение речи: после реплики целиком, с пометкой
                edits.append({"op": "overlap_move", "t": w["s"], "w": w["w"], "card": cid,
                              "after_t": reply_end_t(p[0]), "in_place": in_place(w)})
                rep["overlap"].append(f"{w['_spk']}: {w['w']}")
            else:
                # на стыке реплик ответа — только предложение удалить (решает человек)
                rep["proposed_delete"].append(w["w"])
                edits.append({"op": "propose_delete", "t": w["s"], "w": w["w"], "card": cid,
                              "in_place": in_place(w)})
            continue
        flush(j)
        w = win[j]
        spk = ans_spk[i]
        if k == "match" and norm(ans_raw[i]) != norm(w["w"]):          # совпало с вариантом другой модели
            edits.append({"op": "replace", "t": w["s"], "w": w["w"], "text": ans_raw[i], "card": cid,
                          "in_place": in_place(w), "note": "человек и вариант модели"})
            rep["applied"].append(f"«{w['w']}» → «{ans_raw[i]}» (как у другой модели)")
        elif k == "match":
            hl, wl = core_of(ans_raw[i]), core_of(w["w"])
            target = human_spelling(ans_raw[i], w["w"])
            if hl and wl and target != w["w"]:   # то же слово, иначе написано: регистр, «2.19.3», ё, знак после
                edits.append({"op": "replace", "t": w["s"], "w": w["w"], "text": target,
                              "card": cid, "in_place": in_place(w),
                              "note": "регистр" if hl.lower() == wl.lower() and hl != wl else "написание человека"})
                rep["applied"].append(f"«{w['w']}» → «{target}»")
        if k == "sub":
            new, note = fix_typo(ans_raw[i], w, corpus, vocab_words)
            if note:
                rep["typos"].append(f"«{ans_raw[i]}» → «{new}»")
            if norm(new) != norm(w["w"]) or core_of(new) != core_of(w["w"]):
                edits.append({"op": "replace", "t": w["s"], "w": w["w"], "text": new, "card": cid,
                              "in_place": in_place(w), "note": note})
                rep["applied"].append(f"«{w['w']}» → «{new}»")
        if spk and spk != w["_spk"]:
            edits.append({"op": "relabel", "t": w["s"], "w": w["w"], "speaker": spk, "card": cid,
                          "in_place": in_place(w)})
        prev_j = j
    flush(None)
    if problems:
        # правку нельзя применить целиком — не применяем частично: место человеку с причиной
        rep["human"] = "; ".join(dict.fromkeys(problems))
        rep["applied"], rep["inserted"], rep["deleted"], rep["overlap"] = [], [], [], []
        return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
    relab = {}
    for e in edits:
        if e["op"] == "relabel":
            relab.setdefault(e["speaker"], []).append(e["w"])
    for s_, ws_ in relab.items():
        rep["applied"].append(f"«{' '.join(ws_)}» → {s_}")
    return {"kind": "text", "edits": edits, "covered": covered, "anchors": anchors, "report": rep}


# ---------- 7. несколько карточек ----------
def plan(manifest, answers, data, vocab_words=()):
    """manifest — пакет; answers — {id: ответ} со свободным текстом (kind free/text) или «только имя».
    data — стенограмма. -> (правки слов, отчёт по карточкам, закрытые места {id: через}, конфликты)."""
    words = []
    for seg in data["segments"]:
        for w in seg.get("words") or []:
            words.append({**w, "_spk": seg["speaker"]})
    words.sort(key=lambda w: w["s"])
    names = [n for n in (data.get("speakers") or []) if n] or sorted({w["_spk"] for w in words})
    corpus = frozenset(norm(w["w"]) for w in words) | frozenset(
        norm(x) for w in words for k in ("alt", "alt_whisper") for x in str(w.get(k) or "").split())
    vocab_words = frozenset(norm(x) for x in vocab_words)
    rv_by = {r["id"]: r for r in data.get("review") or []}
    items = sorted(({**it, **{k: rv_by.get(it["id"], {}).get(k) for k in ("alt_ctc", "alt_whisper")
                              if rv_by.get(it["id"], {}).get(k)}}
                    for it in manifest["items"] if it["id"] in answers), key=lambda it: it["t1"])
    by_word, reports, results = {}, [], {}
    conflicts = []
    for it in items:
        a = answers[it["id"]]
        r = card_edits(it, a, words, names, corpus, vocab_words)
        results[it["id"]] = r
        reports.append(r["report"])
        for e in r["edits"]:
            key = (e["op"] if e["op"] in ("insert", "place_text") else "w",
                   round(e.get("t", e.get("s", e.get("t1", 0))), 2),
                   e["op"] if e["op"] != "relabel" else "relabel")
            old = by_word.get(key)
            if old is None:
                by_word[key] = (e, a.get("ts") or "")
                continue
            oe, ots = old
            same = (oe.get("text"), oe.get("speaker")) == (e.get("text"), e.get("speaker"))
            if same:
                continue
            win_new = (e.get("in_place") and not oe.get("in_place")) or \
                      (e.get("in_place") == oe.get("in_place") and (a.get("ts") or "") > ots)
            conflicts.append({"word": e.get("w") or e.get("text"), "t": e.get("t", e.get("s")),
                              "cards": [oe["card"], e["card"]], "kept": e["card"] if win_new else oe["card"]})
            if win_new:
                by_word[key] = (e, a.get("ts") or "")
    edits = [e for e, _ in by_word.values()]
    # места других карточек и сомнительные места стенограммы, покрытые ответом, — закрыты человеком
    closed = {}
    for cid, r in results.items():
        if not r.get("covered"):
            continue
        t_a, t_b = r["covered"]
        for rv in data.get("review") or []:
            if rv["id"] == cid or rv["id"] in answers:
                continue
            ws = [w for w in words if rv.get("t1", rv["t"]) - 0.01 <= w["s"] <= rv.get("t2", rv["t"]) + 0.01]
            inside = [w for w in ws if t_a - 0.01 <= w["s"] <= t_b + 0.01]
            if ws and len(inside) * 2 >= len(ws):
                closed.setdefault(rv["id"], cid)
    return edits, reports, closed, conflicts
