"""Свободные ответы человека в карточках разбора: ответ — правка того, что человек услышал в клипе.

Порядок старшинства: ответ человека > отпечаток голоса > распознавание (основная, вторая и третья модели) > эвристики.

Для каждой карточки со свободным текстом:
  1. вид ответа: только имя («Борис», «не Анна, а Вера») / короткий ответ по таблице answer_phrases.tsv
     («да», «верно», похожее на оценку «согласен» — таблица В8 в card_edits) / правка текста;
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
import csv
import io
import itertools
from pathlib import Path
import re

import transcript as T

PAD_BEFORE, PAD_AFTER, SLACK = 3.5, 1.5, 1.0
GROW, MAX_GROW = 4.0, 2          # расширение окна, если ответ шире него (с), и сколько раз
DUP_SEC = 2.0                    # повтор: то же слово модели не дальше этого от вставки
MIN_ANCHORS = 0.3

ANSWER_PHRASES_PATH = Path(__file__).with_name("answer_phrases.tsv")
_ANSWER_PHRASE_KINDS = (
    "short_word", "confirm", "evaluation_stem",
    "negative_prefix", "negative_intensifier", "negative_core",
)


def _read_answer_phrases(path=ANSWER_PHRASES_PATH):
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"{path.name}, строка 1: не удалось прочитать файл таблицы слов"
        ) from exc
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path.name}, строка 1: файл не должен содержать BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = raw[:exc.start].count(b"\n") + 1
        raise ValueError(
            f"{path.name}, строка {line}: файл должен быть в кодировке UTF-8"
        ) from exc

    reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"{path.name}, строка 1: отсутствует заголовок таблицы") from exc
    except csv.Error as exc:
        raise ValueError(f"{path.name}, строка 1: не удалось разобрать заголовок таблицы") from exc
    if header != ["id", "kind", "text", "note"]:
        raise ValueError(
            f"{path.name}, строка 1: ожидается заголовок id, kind, text, note"
        )

    phrases = {kind: [] for kind in _ANSWER_PHRASE_KINDS}
    seen_ids = set()
    seen_texts = set()          # (kind, фраза): повтор проверяется внутри категории («нет» — и слово, и приставка)
    try:
        for row in reader:
            line = reader.line_num
            if len(row) != 4:
                raise ValueError(
                    f"{path.name}, строка {line}: ожидалось 4 поля, разделённые табуляцией"
                )
            phrase_id, kind, phrase, _note = row
            if not phrase_id:
                raise ValueError(f"{path.name}, строка {line}: пустой id")
            if phrase_id in seen_ids:
                raise ValueError(f"{path.name}, строка {line}: повторный id {phrase_id}")
            if kind not in phrases:
                raise ValueError(f"{path.name}, строка {line}: неизвестный kind {kind}")
            if not phrase.strip():
                raise ValueError(f"{path.name}, строка {line}: пустой text")
            if (kind, phrase) in seen_texts:
                raise ValueError(f"{path.name}, строка {line}: повторная фраза")
            other = {"short_word": "confirm", "confirm": "short_word"}.get(kind)
            if other and (other, phrase) in seen_texts:
                raise ValueError(f"{path.name}, строка {line}: фраза «{phrase}» и в short_word, и в confirm")
            seen_ids.add(phrase_id)
            seen_texts.add((kind, phrase))
            phrases[kind].append(phrase)
    except csv.Error as exc:
        line = reader.line_num or 1
        raise ValueError(
            f"{path.name}, строка {line}: не удалось разобрать строку таблицы"
        ) from exc
    missing_kinds = [kind for kind, items in phrases.items() if not items]
    if missing_kinds:
        line = reader.line_num + 1
        kinds = ", ".join(missing_kinds)
        raise ValueError(
            f"{path.name}, строка {line}: отсутствуют обязательные категории: {kinds}"
        )
    return phrases


_ANSWER_PHRASES = _read_answer_phrases()

# Закрытый список отрицаний: ответ (или его часть до двоеточия) после norm — ровно [приставка] [усилитель] отрицание.
# Такой ответ ничего не меняет и не подтверждает. Других отрицаний нет: остальное разбирается как раньше.
NEGATIVE_PREFIXES = tuple(_ANSWER_PHRASES["negative_prefix"])
NEGATIVE_INTENSIFIERS = tuple(_ANSWER_PHRASES["negative_intensifier"])
NEGATIVE_CORES = tuple(_ANSWER_PHRASES["negative_core"])
NEGATIVES = frozenset(" ".join(x for x in (p, i, c) if x) for p in ("",) + NEGATIVE_PREFIXES
                      for i in ("",) + NEGATIVE_INTENSIFIERS for c in NEGATIVE_CORES)
WORD =re.compile(r"[\wЁё][\wЁё'-]*", re.U)

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


def _phrase_key(text):
    return " ".join(norm(x) for x in (text or "").split() if norm(x))


# Короткие ответы (В8): слово, которое может быть и словом стенограммы, и оценкой; подтверждение «слова верны»;
# начало слова, похожего на оценку (ответ вне таблицы, не длиннее EVALUATION_MAX_WORDS слов).
SHORT_WORDS = frozenset(_phrase_key(x) for x in _ANSWER_PHRASES["short_word"])
CONFIRMS = frozenset(_phrase_key(x) for x in _ANSWER_PHRASES["confirm"])
EVALUATION_STEMS = tuple(_phrase_key(x) for x in _ANSWER_PHRASES["evaluation_stem"])
EVALUATION_MAX_WORDS = 4
SHORT_KINDS = ("short_word", "confirm", "evaluation")


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
def _name_at(toks, pos, names):
    """Имя из целых слов; при совпадении префиксов выбирается самое длинное."""
    candidates = sorted(((tuple(norm(x) for x in n.split()), n) for n in names if n),
                        key=lambda pair: len(pair[0]), reverse=True)
    for parts, name in candidates:
        if tuple(norm(x) for x in toks[pos:pos + len(parts)]) == parts:
            return name, len(parts)
    return None, 0


NAME_NEGATIONS = {"не", "ни"}
# служебные слова ответа «только имя»: «это Вера», «не Анна, а Вера», «Вера говорит»
NAME_FILLER = {"это", "не", "а", "говорит", "был", "была", "тут", "здесь"}
# в ответе с отрицаемым именем служебными считаются и «нет», «ни», «и»: «нет, не Анна», «ни Анна, ни Борис»
NEGATED_NAME_FILLER = NAME_FILLER | {"нет", "ни", "и"}


def _negated_name(toks, pos):
    """Единое правило отрицания имени: имя, перед которым стоит «не» или «ни», не назначается.
    Его используют и ответ «только имя» (classify), и метка «Имя:» (split_replies, card_edits)."""
    return bool(pos) and norm(toks[pos - 1]) in NAME_NEGATIONS


def _name_mentions(toks, names):
    """-> [(позиция, число слов, имя, отрицается)] — все имена ответа слева направо."""
    out, i = [], 0
    while i < len(toks):
        name, count = _name_at(toks, i, names)
        if name:
            out.append((i, count, name, _negated_name(toks, i)))
            i += count
        else:
            i += 1
    return out


def _negated_label(toks, pos, names):
    name, count = _name_at(toks, pos, names)
    return bool(name and _negated_name(toks, pos) and toks[pos + count - 1].rstrip().endswith(":"))


def negative_answer(text):
    """Ответ из закрытого списка NEGATIVES; «неверно: слово» — тоже (проверяется часть до двоеточия)."""
    head = (text or "").split(":", 1)[0]
    return " ".join(norm(x) for x in head.split() if norm(x)) in NEGATIVES


def classify(text, names):
    t = " ".join(norm(x) for x in (text or "").split() if norm(x))
    if not t:
        return {"kind": "skip"}
    if negative_answer(text):
        return {"kind": "negative"}
    if t in CONFIRMS:
        return {"kind": "confirm"}
    if t in SHORT_WORDS:
        return {"kind": "short_word"}
    ws = t.split()
    mentions = _name_mentions(ws, names)
    if not mentions:
        # ответ вне таблицы, похожий на оценку («согласен», «рекомендую»): не слова стенограммы
        if len(ws) <= EVALUATION_MAX_WORDS and any(w.startswith(EVALUATION_STEMS) for w in ws):
            return {"kind": "evaluation"}
        return {"kind": "text"}
    filler = NEGATED_NAME_FILLER if any(neg for *_, neg in mentions) else NAME_FILLER
    covered = {k for pos, count, _, _ in mentions for k in range(pos, pos + count)}
    rest = [w for k, w in enumerate(ws) if k not in covered and w not in filler]
    named = [name for _, _, name, neg in mentions if not neg]
    if rest:
        return {"kind": "text"}
    if named:
        return {"kind": "name", "speaker": named[-1]}
    return {"kind": "unmatched"}      # все названные имена отрицаются — выбрать говорящего должен человек


# Н-9 (слово владельца 28.11: «Да, делить»): знак препинания между словами без пробела — конец слова.
# «Вот.Ну» → «Вот.» «Ну», «Ну,да» → «Ну,» «да». Слева нужны две буквы: «т.е.», «т.д.», «3.5» не делятся.
GLUED = re.compile(r"(?<=[^\W\d_]{2})([.,!?;:]+)(?=[^\W\d_])")


def unglue(text):
    """Пробел после знака препинания, приклеенного к следующему слову (см. GLUED)."""
    return GLUED.sub(lambda m: m.group(1) + " ", text or "")


def split_replies(text, names):
    """-> [(говорящий | None, [слова])]. Метка — «Имя:» или имя в начале фразы перед словом с заглавной.
    Слова, склеенные знаком препинания без пробела, делятся (unglue)."""
    toks = unglue(text).replace("…", "... ").split()
    out, cur, spk = [], [], None
    i = 0
    while i < len(toks):
        raw = toks[i]
        name, count = _name_at(toks, i, names)
        is_label = False
        if name:
            last = toks[i + count - 1]
            colon = last.rstrip().endswith(":")
            prev_end = i == 0 or re.search(r"[.!?…:]$", toks[i - 1]) is not None
            nxt = toks[i + count] if i + count < len(toks) else ""
            if not _negated_label(toks, i, names) and \
                    (colon or (prev_end and nxt[:1].isupper() and not last.rstrip().endswith(","))):
                is_label = True
        if is_label:
            if cur:
                out.append((spk, cur))
            spk, cur = name, []
            i += count
            continue
        if re.fullmatch(r"\.{2,}|…", raw.strip()):
            i += 1
            continue
        raw = re.sub(r"(\.{3}|…)$", "", raw)
        if norm(raw):
            cur.append(raw)
        i += 1
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


def _pause_answer(item, rep, flat, words, inside):
    """Ответ текстом на место-пропуск. flat — [(говорящий, слово)], inside — слова паузы (начатые в ней).

    Слова ответа, равные последним словам стенограммы до паузы (с начала ответа) и первым после неё (с конца),
    при том же говорящем и не дальше DUP_SEC от края паузы, — это соседние слова, а не речь из паузы:
    они не вставляются и не подписываются. Один токен ответа — одно слово стенограммы."""
    t_a, t_b = float(item["t1"]), float(item["t2"])
    cents, limit = T._cents, round(DUP_SEC * 100)
    before = [w for w in words if cents(w["s"]) < cents(t_a)]
    after = [w for w in words if cents(w["s"]) >= cents(t_b)]

    def same(token, w):
        return bool(norm(token[1])) and norm(token[1]) == norm(w["w"]) and token[0] == w["_spk"]

    k = m = 0
    for n in range(min(len(flat), len(before)), 0, -1):
        if cents(t_a) - cents(before[-n]["s"]) <= limit and all(same(t, w) for t, w in zip(flat[:n], before[-n:])):
            k = n
            break
    for n in range(min(len(flat) - k, len(after)), 0, -1):
        if cents(after[n - 1]["s"]) - cents(t_b) <= limit and all(same(t, w) for t, w in zip(flat[len(flat) - n:], after[:n])):
            m = n
            break
    middle = flat[k:len(flat) - m]
    known = flat[:k] + flat[len(flat) - m:]
    rep["kind"] = "missed"
    if not inside:
        if not middle:
            rep["human"] = ("ответ совпадает со словами стенограммы рядом с паузой («" +
                            " ".join(x for _, x in known) + "»): в паузу ничего не вписано")
            return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
        step = (t_b - t_a) / (len(middle) + 1)
        eds = []
        for q, (spk, x) in enumerate(middle, 1):
            s = round(t_a + step * q, 3)
            eds.append({"op": "insert", "s": s, "e": round(s + step * 0.9, 3), "text": x, "speaker": spk,
                        "card": item["id"], "in_place": True})
        rep["applied"] = [f"«{x}» — уже в стенограмме рядом с паузой, не вставлено" for _, x in known]
        # Н-9 (слово владельца 28.11: «вписать с предупреждением»): слово ответа на краю совпало с соседним
        # словом другого говорящего — вписываем, но человек видит, что слово окажется в стенограмме дважды.
        warn = []
        for edge, w in (("первое", before[-1] if before and k == 0 else None),
                        ("последнее", after[0] if after and m == 0 else None)):
            tok = middle[0] if edge == "первое" else middle[-1]
            if w is None or not norm(tok[1]) or norm(tok[1]) != norm(w["w"]) or tok[0] == w["_spk"]:
                continue
            if abs(cents(w["s"]) - cents(t_a if edge == "первое" else t_b)) > limit:
                continue
            fields = dict(edge=edge, word=tok[1], speaker=w["_spk"] or "без подписи",
                          sec=f"{float(w['s']):.2f}".replace(".", ","))
            warn.append((PAUSE_NEIGHBOR_DUP.format(**fields), PAUSE_NEIGHBOR_DUP_DONE.format(**fields)))
        if warn:
            rep["warnings"] = [x for x, _ in warn]            # предпросмотр: что произойдёт
            rep["warnings_done"] = [x for _, x in warn]       # отчёт «Применить»: что произошло
        rep["inserted"] = [f"{spk}: {' '.join(x for s2, x in grp)}"
                           for spk, grp in itertools.groupby(middle, key=lambda t: t[0])]
        return {"kind": "missed", "edits": eds, "covered": (t_a, t_b), "report": rep}
    here = " ".join(w["w"] for w in inside)
    if len(middle) == len(inside) and all(same(t, w) for t, w in zip(middle, inside)):
        # те же слова уже стоят в паузе: вставлять нечего; открытое место закроется как «как есть» (confirm_as_is)
        rep["applied"] = [f"уже вписано: «{here}»"]
        rep["confirm_as_is"] = True
        return {"kind": "missed", "edits": [], "covered": (t_a, t_b), "report": rep}
    rep["human"] = (f"в паузе уже есть слова («{here}»), ответ с ними не совпадает; исправьте их правкой "
                    "отрезка или ответом на карточку этих слов")
    return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}


PAUSE_NEIGHBOR_DUP_DONE = ("{edge} слово ответа «{word}» совпадало с соседним словом стенограммы ({speaker}, "
                           "{sec} с) и вписано второй раз; если это то же слово — уберите лишнее: скажите Claude "
                           "«исправь место этой карточки»")
PAUSE_NEIGHBOR_DUP = ("{edge} слово ответа «{word}» совпадает с соседним словом стенограммы ({speaker}, {sec} с): "
                      "оно будет вписано второй раз; если это то же слово — уберите его из ответа")


# Подсказки на короткий ответ (В8): место остаётся открытым, стенограмма не меняется.
HINT_PAUSE_CONFIRM = ("паузу коротким ответом не закрыть: нажмите «" + T.LISTEN_SILENCE + "» или «" +
                      T.LISTEN_INTERJECTIONS + "»; если в паузе сказано это слово — напишите «Имя: {answer}»")
HINT_EVALUATION = ("ответ похож на оценку, а не на услышанные слова: место осталось открытым; "
                   "выберите вариант или напишите «Имя: слова»")
HINT_WORD = ("короткий ответ «{answer}» не совпадает со словами места («{place}»): чтобы вписать его, "
             "напишите «Имя: {answer}»; чтобы подтвердить слова — «верно» или вариант «как есть»")
HINT_DISAGREE = "модели расходятся — короткое «{answer}» не говорит, чей вариант верен: "
HINT_DISAGREE_OPTION = "выберите вариант «{label}» или впишите услышанные слова"
HINT_DISAGREE_WORDS = "впишите услышанные слова"


def _short_answer(item, kind, text, place, is_gap):
    """Короткий ответ по таблице В8. -> "as_is" (подтвердить слова места), текст подсказки или None (это слова).
    В подсказке фраза человека как есть (ё и регистр сохраняются); сравнение со словами места — по ключу."""
    answer, key = " ".join(text.split()), _phrase_key(text)
    if kind == "evaluation":
        return HINT_EVALUATION
    if is_gap:
        return None if kind == "short_word" else HINT_PAUSE_CONFIRM.format(answer=answer)
    if kind == "short_word":
        here = " ".join(norm(w["w"]) for w in place if norm(w["w"]))
        if place and here == key:
            return "as_is"
        return HINT_WORD.format(answer=answer, place=" ".join(w["w"] for w in place) or "—")
    if "asr" in (item.get("reasons") or []):       # «модели расходятся»: «верно» не говорит, чей вариант
        label = next((str(o.get("label", "")) for o in item.get("options") or []
                      if not {k for k in (o.get("decision") or {}) if k != "note"}), None)
        return HINT_DISAGREE.format(answer=answer) + (
            HINT_DISAGREE_OPTION.format(label=label) if label else HINT_DISAGREE_WORDS)
    return "as_is"


def card_edits(item, answer, words, names, corpus=frozenset(), vocab_words=frozenset()):
    """item — элемент manifest (id, t1, t2, text, speaker); answer — ответ (text, ts).
    -> {"kind", "edits": [...], "covered": (t_from, t_to), "anchors", "report": {...}}"""
    text = answer.get("text") or ""
    cid = item["id"]
    is_gap = T.is_gap_place(item)             # пропуск: слова места — только начатые в паузе, word_ids не читаются
    times = T._place_time(item) if is_gap else None
    if is_gap and times is None:
        return {"kind": "unmatched", "edits": [], "covered": None, "report": {
            "id": cid, "answer": text, "kind": "unmatched", "applied": [], "inserted": [], "typos": [],
            "proposed_delete": [], "deleted": [], "overlap": [],
            "human": "у места нет годного времени: ответ не применён"}}
    target_ids = set() if is_gap else set(item.get("word_ids") or [])
    if is_gap:
        place = [w for w in words if T.in_pause(w["s"], *times)]
    else:
        place = ([w for w in words if w.get("word_id") in target_ids] if target_ids else
                 [w for w in words if item["t1"] - 0.01 <= w["s"] <= item["t2"] + 0.01])
    target_times = {round(w["s"], 2) for w in place} if target_ids else set()

    def card_window(a, b):
        return [w for w in window(words, a, b) if not target_ids or
                round(w["s"], 2) not in target_times or w.get("word_id") in target_ids]

    win = card_window(item["t1"], item["t2"])
    extra = " ".join(str(item.get(k) or "") for k in ("alt_ctc", "alt_whisper"))
    for w in place:                       # варианты моделей по месту — тоже опоры
        w["alt_place"] = extra
    c = classify(text, names)
    rep = {"id": cid, "answer": text, "kind": c["kind"], "applied": [], "inserted": [], "typos": [],
           "proposed_delete": [], "deleted": [], "overlap": [], "human": None}
    label_tokens = text.split()
    if any(_negated_label(label_tokens, i, names) for i in range(len(label_tokens))):
        rep["kind"] = "unmatched"
        rep["human"] = "отрицательная метка неоднозначна; укажите говорящего и слова отдельно"
        return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
    if c["kind"] == "negative":
        rep["human"] = "ответ — отказ, стенограмма не менялась; впишите услышанное слово или выберите вариант"
        return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
    if c["kind"] == "unmatched":
        rep["human"] = "указано только отрицание имени; выберите говорящего"
        return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
    if c["kind"] in SHORT_KINDS:
        if item.get("answer_mode") == "speaker":
            if c["kind"] != "evaluation":
                # Названное исключение: на карточке подтверждения говорящего короткий ответ — как прежнее
                # согласие, ничего не меняет; оценка разбирается как прежний текст.
                return {"kind": "agree", "edits": [], "covered": None, "report": rep}
        else:
            short = _short_answer(item, c["kind"], text, place, is_gap)
            if short == "as_is":
                # слова места верны: открытое место закроется тем же путём, что вариант «как есть»
                rep["confirm_as_is"] = True
                rep["applied"] = [f"подтверждено как есть: «{' '.join(w['w'] for w in place)}»"]
                return {"kind": c["kind"], "edits": [], "covered": None, "report": rep}
            if short:
                # подсказка: место остаётся открытым, к человеку не уходит
                rep["kind"], rep["human"] = "hint", short
                return {"kind": "hint", "edits": [], "covered": None, "report": rep}
    if c["kind"] == "name" and is_gap and not place:
        rep["human"] = ("в паузе нет слов: одно имя ничего не подписывает; "
                        "напишите «Имя: услышанные слова»")
        return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
    if c["kind"] == "name":
        eds = [{"op": "relabel", "t": w["s"], "w": w["w"], "word_id": w.get("word_id"), "speaker": c["speaker"], "card": cid, "in_place": True}
               for w in place if w.get("_spk") != c["speaker"]]
        rep["applied"] = [f"{' '.join(w['w'] for w in place)} → {c['speaker']}"]
        span = (place[0]["s"], place[-1]["s"]) if place else None
        return {"kind": "name", "edits": eds, "covered": span, "report": rep}
    for label in re.finditer(r"(?<!\w)Спикер\s+\d+\s*:", text, re.I):
        label_words = label.group().split()
        name, count = _name_at(label_words, 0, names)
        if not name or count != len(label_words):
            rep["kind"] = "unmatched"
            rep["human"] = "неизвестная метка говорящего; выберите имя"
            return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
    replies = split_replies(text, names)
    if len(replies) > 1 and replies[0][0] is None and replies[1][0] and \
            all(norm(x) == "нет" for x in replies[0][1]):
        # «нет, Анна: слово» — «нет» здесь скорее поправка к карточке, чем речь из записи; угадывать
        # не будем: ни «нет» в стенограмму, ни смену говорящего — место остаётся человеку
        rep["kind"] = "unmatched"
        rep["human"] = "перед меткой «Имя:» стоит только «нет»; напишите «Имя: точные слова» без «нет»"
        return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
    if item.get("answer_mode") == "speaker":
        # Карточка подтверждения говорящего: пояснение («Вера говорит только слово …») не должно
        # молча стать текстом реплики. Слова принимаются лишь в явной форме «Имя: точные слова».
        label = re.match(r"\s*([^:]+)\s*:", text)
        label_words = label.group(1).split() if label else []
        name, count = _name_at(label_words, 0, names)
        if not (name and count == len(label_words) and replies and all(spk for spk, _ in replies)):
            rep["kind"] = "speaker"
            rep["human"] = ("на карточке подтверждения говорящего свой ответ не понят как решение "
                            "о говорящем (нужно имя или «Имя: точные слова»)")
            return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
    if is_gap and replies:
        # пропущенная речь: слова паузы — только начатые в ней; слова ответа, уже стоящие рядом с паузой, не дублируются
        flat = [(spk or item.get("speaker"), x) for spk, ws in replies for x in ws]
        if any(not who for who, _ in flat):
            rep["kind"] = "missed"
            rep["human"] = "не удалось определить говорящего пропущенной речи (в ответе нет «Имя:»)"
            return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
        return _pause_answer(item, rep, flat, words, place)
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
    def ambiguous_window(ws):
        times = [round(w["s"], 2) for w in ws]
        return len(times) != len(set(times))
    if target_ids and ambiguous_window(win):
        rep["human"] = "неоднозначные одновременные слова в окне выравнивания"
        return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
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
        win = card_window(item["t1"] - grow_a, item["t2"] + grow_b)
        if target_ids and ambiguous_window(win):
            rep["human"] = "неоднозначные одновременные слова в расширенном окне"
            return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
        pairs = align(ans, win)
    anchors = sum(1 for i, j, k in pairs if k == "match")
    labelled = any(s for s, _ in replies)
    # реплика с меткой, у которой нет ни одной опоры, — вставка пропущенной речи, в долю опор не входит
    rep_of = []
    for q, (spk, ws) in enumerate(replies):
        rep_of += [q] * len(ws)
    anchored_reps = {rep_of[i] for i, j, k in pairs if k == "match" and i is not None}
    countable = [i for i in range(len(ans)) if not (replies[rep_of[i]][0] and rep_of[i] not in anchored_reps)]
    in_place_j = {k for k, w in enumerate(win) if (w.get("word_id") in target_ids if target_ids else
                   item["t1"] - 0.01 <= w["s"] <= item["t2"] + 0.01)}
    anchored_j = {j for i, j, k in pairs if k == "match"}
    near_place = {k for k in in_place_j} | {k - 1 for k in in_place_j} | {k + 1 for k in in_place_j}
    if not labelled and place and (not anchored_j or anchored_j <= near_place) and len(ans) <= len(place) + 4 \
            and not (grow_a or grow_b):
        # ответ без меток про само место — новый текст места; без опор — самого похожего его куска
        if anchored_j:
            idx = sorted(k for k, word in enumerate(place) for j in anchored_j if word is win[j])
            a_, b_ = (idx[0], idx[-1] + 1) if idx else (0, len(place))
            if len(ans) > b_ - a_:          # слов в ответе больше, чем покрыто опорами, — до краёв места
                a_, b_ = 0, len(place)
        else:
            a_, b_ = _place_span(place, ans_raw)
        sub = place[a_:b_]
        rep["kind"] = "place_text"
        rep["applied"] = [f"«{' '.join(w['w'] for w in sub)}» → «{' '.join(ans_raw)}»"]
        return {"kind": "place_text", "edits": [{"op": "place_text", "card": cid, "t1": sub[0]["s"],
                                                 "t2": sub[-1]["s"], "text": " ".join(ans_raw), "in_place": True,
                                                 **({"word_ids": [w["word_id"] for w in sub]} if target_ids else {})}],
                "covered": (sub[0]["s"], sub[-1]["s"]), "report": rep}
    if not countable or sum(1 for i, j, k in pairs if k == "match" and i in set(countable)) < MIN_ANCHORS * len(countable) \
            or not anchored_reps:
        rep["human"] = "не удалось сопоставить ответ со стенограммой"
        return {"kind": "unmatched", "edits": [], "covered": None, "report": rep}
    edits = []
    in_place = lambda w: w.get("word_id") in target_ids if target_ids else item["t1"] - 0.01 <= w["s"] <= item["t2"] + 0.01
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

    def reply_end_id(i):
        same = [j for _, ii, j in anch if rep_of[ii] == rep_of[i] and rspk[ii] == rspk[i]]
        return win[max(same)].get("word_id") if same else None

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
                          "before_t": win[next_j]["s"] if next_j is not None else None,
                          "after_id": win[prev_j].get("word_id") if prev_j is not None else None,
                          "before_id": win[next_j].get("word_id") if next_j is not None else None, "in_place": False})
            rep["inserted"].append(f"{who or '?'}: {raw}")
            # повтор: такое же слово модели того же говорящего рядом, ни с чем не сопоставленное, — правка его
            # покрывает, в аудио оно одно; слово модели удаляется
            for jj, ww in enumerate(win):
                if jj not in paired_j and norm(ww["w"]) == norm(raw) and len(norm(raw)) > 1 and \
                        ww["_spk"] == who and abs(ww["s"] - t_a) <= DUP_SEC:
                    paired_j.add(jj)
                    edits.append({"op": "delete", "t": ww["s"], "w": ww["w"], "word_id": ww.get("word_id"), "card": cid, "in_place": in_place(ww),
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
                edits.append({"op": "delete", "t": w["s"], "w": w["w"], "word_id": w.get("word_id"), "card": cid, "in_place": in_place(w),
                              "note": "человек этого слова не назвал"})
                rep["deleted"].append(w["w"])
            elif same_reply:
                # чужое слово внутри реплики человека — наложение речи: после реплики целиком, с пометкой
                edits.append({"op": "overlap_move", "t": w["s"], "w": w["w"], "word_id": w.get("word_id"), "card": cid,
                              "after_t": reply_end_t(p[0]), "after_id": reply_end_id(p[0]), "in_place": in_place(w)})
                rep["overlap"].append(f"{w['_spk']}: {w['w']}")
            else:
                # на стыке реплик ответа — только предложение удалить (решает человек)
                rep["proposed_delete"].append(w["w"])
                edits.append({"op": "propose_delete", "t": w["s"], "w": w["w"], "word_id": w.get("word_id"), "card": cid,
                              "in_place": in_place(w)})
            continue
        flush(j)
        w = win[j]
        spk = ans_spk[i]
        if k == "match" and norm(ans_raw[i]) != norm(w["w"]):          # совпало с вариантом другой модели
            edits.append({"op": "replace", "t": w["s"], "w": w["w"], "word_id": w.get("word_id"), "text": ans_raw[i], "card": cid,
                          "in_place": in_place(w), "note": "человек и вариант модели"})
            rep["applied"].append(f"«{w['w']}» → «{ans_raw[i]}» (как у другой модели)")
        elif k == "match":
            hl, wl = core_of(ans_raw[i]), core_of(w["w"])
            target = human_spelling(ans_raw[i], w["w"])
            if hl and wl and target != w["w"]:   # то же слово, иначе написано: регистр, «2.19.3», ё, знак после
                edits.append({"op": "replace", "t": w["s"], "w": w["w"], "word_id": w.get("word_id"), "text": target,
                              "card": cid, "in_place": in_place(w),
                              "note": "регистр" if hl.lower() == wl.lower() and hl != wl else "написание человека"})
                rep["applied"].append(f"«{w['w']}» → «{target}»")
        if k == "sub":
            new, note = fix_typo(ans_raw[i], w, corpus, vocab_words)
            if note:
                rep["typos"].append(f"«{ans_raw[i]}» → «{new}»")
            if norm(new) != norm(w["w"]) or core_of(new) != core_of(w["w"]):
                edits.append({"op": "replace", "t": w["s"], "w": w["w"], "word_id": w.get("word_id"), "text": new, "card": cid,
                              "in_place": in_place(w), "note": note})
                rep["applied"].append(f"«{w['w']}» → «{new}»")
        if spk and spk != w["_spk"]:
            edits.append({"op": "relabel", "t": w["s"], "w": w["w"], "word_id": w.get("word_id"), "speaker": spk, "card": cid,
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
                   e.get("word_id") or tuple(e.get("word_ids") or ()) or round(e.get("t", e.get("s", e.get("t1", 0))), 2),
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
    # Перекрывающиеся наборы word_ids могут иметь разные начала; сравнения
    # только ключей операций недостаточно, чтобы заметить общий целевой ID.
    claims = {}
    for cid, result in results.items():
        for edit in result["edits"]:
            ids = edit.get("word_ids") or ([edit["word_id"]] if edit.get("word_id") else [])
            for wid in ids:
                prior = claims.setdefault(wid, cid)
                if prior != cid and not any(set(c.get("cards", [])) == {prior, cid} for c in conflicts):
                    conflicts.append({"word_id": wid, "cards": [prior, cid], "why": "общая цель правок"})
    # места других карточек и сомнительные места стенограммы, покрытые ответом, — закрыты человеком
    closed = {}
    for cid, r in results.items():
        if any(it.get("id") == cid and it.get("word_ids") and not T.is_gap_place(it) for it in items):
            continue  # Временное покрытие не доказывает покрытие другой карточки по ID.
        if not r.get("covered"):
            continue
        t_a, t_b = r["covered"]
        pause_answer = r["kind"] == "missed"      # ответ на паузу покрывает слова, начатые в ней, и только их
        for rv in data.get("review") or []:
            if rv["id"] == cid or rv["id"] in answers:
                continue
            if T.is_gap_place(rv):
                rv_times = T._place_time(rv)
                ws = [w for w in words if rv_times and T.in_pause(w["s"], *rv_times)]
            else:
                ws = [w for w in words if rv.get("t1", rv["t"]) - 0.01 <= w["s"] <= rv.get("t2", rv["t"]) + 0.01]
            inside = [w for w in ws if (T.in_pause(w["s"], t_a, t_b) if pause_answer
                                        else t_a - 0.01 <= w["s"] <= t_b + 0.01)]
            if ws and len(inside) * 2 >= len(ws):
                closed.setdefault(rv["id"], cid)
    return edits, reports, closed, conflicts
