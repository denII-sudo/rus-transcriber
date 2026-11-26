"""РАР-5.1: порядок мест на разборе — три уровня и минуты вместо штук.

Чистые функции: не читают файлы, базу и сеть, ничего не пишут и не меняют вход. Статусы мест, пороги
отбора мест (`words.py`) и ответы не трогаются; ответа за человека не предлагается. Подключение к сборке
карточек — отдельный шаг (П51-2); здесь только оценка места (`assess`) и выбор порции (`plan`).

Решения владельца 22.11.2026 (`docs/rar-roadmap.md`, «Этап 5.1: решения владельца»):
- «обязательно» — технические сбои распознавания, места-пропуски от 2 с и расхождение моделей в корне
  у важного слова; доли 13/5/2 минуты: слова и пропуски, кто говорит в содержательных репликах, случайная
  проверка остального; неиспользованное время переходит дальше;
- пропуски от 2 с — все, даже сверх минут, с отметкой о превышении;
- одновременная речь: в минуты — только с важным словом, остальное — отдельной очередью со счётчиком.
Решения Архитектора 26.11.2026 по вопросам плана П-51: короткой записи не меньше 5 минут
(`MIN_MINUTES`); дособирание идёт тем же порядком, доли — только для первой порции.

Признак «расхождение в корне» и оценки секунд и вероятности правки — стартовые правила из замеров
одной установки (`docs/research/rar-5-0-review-effort-measure-2026-11.md`,
`docs/research/rar-i4-review-categories-2026-11.md`), на человеке не испытаны.
"""
import hashlib
import math
import re

PLAN_VERSION = 1

PER_HOUR_MINUTES = 20        # минут разбора на час записи по умолчанию
MIN_MINUTES = 5              # нижняя граница для коротких записей (решение 26.11.2026)
LONG_GAP_SECONDS = 2.0       # место-пропуск от этой длины — «обязательно» и показывается всегда
ROOT_PREFIX_SHARE = 0.6      # общее начало короче этой доли длины — корень другой (замер 5.0)

# Доли первой порции, минут из 20: слова и пропуски, кто говорит, случайная проверка.
SHARES = (("words", 13), ("speaker", 5), ("random", 2))

LEVELS = ("required", "wanted", "optional")
BLOCKS = ("required", "speaker", "wanted", "optional", "random")
QUEUE_MAIN, QUEUE_OVERLAP = "main", "overlap"

# Вид места -> (вероятность правки, секунд на место). Стартовые числа: v-rar-5-1, раздел 3.
ESTIMATES = {
    "tech": (0.5, 20),
    "long_gap": (0.61, 23),
    "short_gap": (0.14, 15),
    "root": (0.45, 20),
    "overlap": (0.4, 25),
    "gender": (0.33, 18),
    "voice_word": (0.34, 18),
    "voice": (0.26, 17),
    "dispute": (0.20, 16),
    "conf": (0.06, 13),
    "other": (0.10, 13),
}

NEGATIONS = frozenset({"не", "нет", "ни"})

# Заполнители: ядро и расширение замера И4 (основное правило); «нет» — не заполнитель.
FILLERS = frozenset({
    "аа", "ааа", "ага", "ага-ага", "вот", "да", "да-да", "м", "мда", "мм", "ммм", "ну", "ну-ну", "угу",
    "угу-угу", "хм", "хмм", "э", "эм", "эмм", "ээ", "эээ",
    "а", "ах", "значит", "ии", "короче", "ладно", "ну-да", "о", "ой", "ок", "океи", "окей", "оо", "ооо",
    "ох", "понятно", "слушай", "так", "типа", "хорошо", "ясно"})
_FILLER_FORMS = re.compile(r"(м+|хм+|э+м*|а{2,}|у+г+у+|а+г+а+|о{2,})")
FILLER_MAX_WORDS = 3

_NUMBER_WORDS = re.compile(
    r"(ноль|нуля|нулю|нулем|один|одна|одно|одни|одного|одной|одному|одним|одном|одну"
    r"|два|две|двух|двум|двумя|три|трех|трем|тремя|четыре|четырех|четырем|четырьмя"
    r"|(пят|шест|сем|девят|десят)(ь|и|ью)|восемь|восьми|восемью"
    r"|\w+надцат(ь|и|ью)|(двадцат|тридцат)(ь|и|ью)|(пять|шесть|семь|восемь)десят|(пят|шест|сем|восьм)идесяти"
    r"|сорок|сорока|девяносто|девяноста|сто|ста|(двест|трист|четырест)(и|а)|(пят|шест|сем|восем|девят)сот"
    r"|тысяч(а|и|у|ей|ами|ах|ам)?|(миллион|миллиард)(а|ов|ы|у|ом|ами|ах)?|полтора|полторы"
    r"|(перв|втор|трет|четверт|пят|шест|седьм|восьм|девят|десят)"
    r"(ый|ой|ая|ое|ые|ого|ому|ым|ом|ую|ых|ыми|ий|ья|ье|ьи|ьего|ьему|ьим|ьем|ью|ьих))")
_TOKEN = re.compile(r"\w+(?:-\w+)*")


def _norm(word):
    return word.lower().replace("ё", "е")


def _tokens(text):
    """[(исходное слово, нормализованное)] из строки; не строка — пусто."""
    if not isinstance(text, str):
        return []
    return [(m.group(0), _norm(m.group(0))) for m in _TOKEN.finditer(text)]


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def same_root(a, b):
    """Один корень по правилу замера 5.0: общее начало не короче ROOT_PREFIX_SHARE длины длинного слова."""
    a, b = _norm(a), _norm(b)
    longest = max(len(a), len(b))
    return longest == 0 or _common_prefix(a, b) >= ROOT_PREFIX_SHARE * longest


def is_filler(word):
    w = _norm(word)
    return w in FILLERS or bool(_FILLER_FORMS.fullmatch(w))


def _filler_text(tokens):
    return 0 < len(tokens) <= FILLER_MAX_WORDS and all(is_filler(raw) for raw, _ in tokens)


def _term_tokens(context):
    """Нормализованные слова имён и терминов словаря, которые передал вызывающий (от 3 букв)."""
    out = set()
    for term in (context or {}).get("terms") or ():
        for _, norm in _tokens(term):
            if len(norm) >= 3:
                out.add(norm)
    return frozenset(out)


def _is_term(norm, terms):
    if norm in terms:
        return True
    # падежные окончания: «Петрову» к термину «Петров» (термин от 4 букв, хвост не длиннее 3)
    return any(len(t) >= 4 and norm.startswith(t) and len(norm) - len(t) <= 3 for t in terms)


def important_word(raw, norm, terms, sentence_start=False):
    """Важное слово: цифры и числительные, отрицание, слово имени или термина группы, заглавная не в начале."""
    if any(ch.isdigit() for ch in norm) or _NUMBER_WORDS.fullmatch(norm):
        return "number"
    if norm in NEGATIONS:
        return "negation"
    if terms and _is_term(norm, terms):
        return "term"
    if raw[:1].isupper() and not sentence_start:
        return "capital"
    return None


def _variants(item):
    return [v for v in (item.get("alt_ctc"), item.get("alt_whisper")) if isinstance(v, str) and v.strip()]


def _root_pairs(text_tokens, variant_tokens):
    """Слова вариантов без пары одного корня в тексте места -> [(слово варианта, ближайшее слово текста)].
    Вариант содержит только слова со своим вариантом, поэтому слово текста без варианта расхождением не считается."""
    out = []
    for raw, norm in variant_tokens:
        if is_filler(raw) and any(is_filler(t) for t, _ in text_tokens):
            continue
        best = max(text_tokens, key=lambda tok: _common_prefix(norm, tok[1]), default=None)
        if best is None or not same_root(norm, best[1]):
            out.append(((raw, norm), best))
    return out


def _gap_seconds(item):
    gap = item.get("gap")
    if isinstance(gap, (list, tuple)) and len(gap) == 2 and all(_finite(x) for x in gap) and gap[0] < gap[1]:
        return float(gap[1] - gap[0])
    t1, t2 = item.get("t1"), item.get("t2")
    if _finite(t1) and _finite(t2) and t1 < t2:
        return float(t2 - t1)
    return None


def assess(item, context=None):
    """Оценка одного места -> {level, queue, pool, kind, importance, p_change, seconds, score, signs}.

    level: required | wanted | optional; queue: main | overlap (одновременная речь без важного слова,
    в минуты не входит); pool: words | speaker | other — из какой доли первой порции берётся место.
    context: {"terms": [имена и термины словаря группы]} — передаёт вызывающий. Вход не меняется."""
    if not isinstance(item, dict):
        raise TypeError("место для разбора — словарь")
    reasons = item.get("reasons")
    reasons = set(reasons) if isinstance(reasons, (list, tuple, set)) else set()
    problem = item.get("asr_problem") if isinstance(item.get("asr_problem"), dict) else None
    terms = _term_tokens(context)

    tech = bool(problem) and problem.get("state") != "disagreement"
    gap = "missed" in reasons and not tech
    gap_seconds = _gap_seconds(item) if gap else None
    long_gap = gap and gap_seconds is not None and gap_seconds >= LONG_GAP_SECONDS

    text_tokens = _tokens(item.get("text"))
    variant_tokens = [_tokens(v) for v in _variants(item)]
    w1 = item.get("w1")
    starts_reply = not (_finite(w1) and w1 > 0)

    def important_in(tokens, starts):
        return [kind for i, (raw, norm) in enumerate(tokens)
                if (kind := important_word(raw, norm, terms, sentence_start=(i == 0 and starts)))]

    important_any = important_in(text_tokens, starts_reply)
    for tokens in variant_tokens:
        important_any += important_in(tokens, True)
    pairs = [p for tokens in variant_tokens for p in _root_pairs(text_tokens, tokens)] if text_tokens else []
    root = bool(pairs)
    important_root = []
    for (raw, norm), best in pairs:
        kind = important_word(raw, norm, terms, sentence_start=True)
        if kind:
            important_root.append(kind)
        if best is not None:
            index = text_tokens.index(best)
            kind = important_word(best[0], best[1], terms, sentence_start=(index == 0 and starts_reply))
            if kind:
                important_root.append(kind)

    all_texts = [text_tokens] + variant_tokens
    filler = bool(text_tokens) and all(_filler_text(t) for t in all_texts if t)
    overlap = "overlap" in reasons
    voice = "voice" in reasons
    gender = "gender" in reasons
    word = bool(reasons & {"asr", "conf"}) or (bool(problem) and not tech)
    dispute = "asr" in reasons or (bool(problem) and not tech)
    conf_only = reasons == {"conf"} and not variant_tokens

    if tech:
        kind = "tech"
    elif gap:
        kind = "long_gap" if long_gap else "short_gap"
    elif overlap:
        kind = "overlap"
    elif root:
        kind = "root"
    elif gender:
        kind = "gender"
    elif voice:
        kind = "voice_word" if word else "voice"
    elif dispute:
        kind = "dispute"
    elif conf_only:
        kind = "conf"
    else:
        kind = "other"

    queue, pool = QUEUE_MAIN, "words"
    if tech or long_gap or (root and important_root) or (overlap and important_any):
        level = "required"
    elif overlap:
        level, queue, pool = "optional", QUEUE_OVERLAP, "other"
    elif filler or conf_only or gap or kind == "other":
        level, pool = "optional", "other"
    else:
        level = "wanted"
        if kind == "voice":
            pool = "speaker"

    if filler and not (tech or gap):
        importance = 0
    elif tech or long_gap or important_any:
        importance = 3
    elif kind == "voice":
        importance = 3 if len(text_tokens) >= 3 else 2
    elif root:
        importance = 2
    else:
        importance = 1
    p_change, seconds = ESTIMATES[kind]
    return {"level": level, "queue": queue, "pool": pool, "kind": kind, "importance": importance,
            "p_change": p_change, "seconds": seconds, "score": importance * p_change / seconds,
            "signs": {"tech": tech, "gap_seconds": gap_seconds, "long_gap": bool(long_gap), "root": root,
                      "important": sorted(set(important_any)), "important_root": sorted(set(important_root)),
                      "overlap": overlap, "voice": voice, "word": word, "gender": gender,
                      "conf_only": conf_only, "filler": filler}}


def budget_minutes(duration_seconds, per_hour=PER_HOUR_MINUTES, floor=MIN_MINUTES):
    """Минуты разбора по длительности записи: per_hour на час, не меньше floor.
    Длительность неизвестна или неверна — None: минуты не угадываются, их спрашивают у человека."""
    if not _finite(duration_seconds) or duration_seconds <= 0:
        return None
    return max(float(floor), per_hour * float(duration_seconds) / 3600.0)


def _time_key(item):
    t = item.get("t", item.get("t1"))
    return float(t) if _finite(t) else math.inf


def _random_key(seed, item_id):
    return hashlib.sha256(f"{seed}\x00{item_id}".encode("utf-8")).hexdigest()


def plan(items, minutes, seed="", context=None, first_portion=True):
    """Выбор порции мест на разбор.

    items — открытые места записи (вызывающий отбирает их сам); minutes — минуты разбора (> 0);
    seed — постоянный номер записи: случайная доля одна и та же при повторе. first_portion=False —
    дособирание: тот же порядок без долей и случайной проверки. Места-пропуски от 2 с берутся все,
    даже сверх минут. Одновременная речь без важного слова в минуты не входит.

    -> {"version", "selected": [{"id", "level", "block", "pick", "seconds"}] в порядке страницы (блоки,
    внутри блока — по времени записи), "remaining_by_level", "overlap_queue", "shown", "estimate_seconds",
    "budget_seconds", "overflow_seconds", "gaps_over_budget"}. shown + остаток + очередь = len(items)."""
    if not _finite(minutes) or minutes <= 0:
        raise ValueError("минуты разбора — положительное число; длительность неизвестна — спросить человека")
    items = list(items)
    ids = [it.get("id") if isinstance(it, dict) else None for it in items]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("у каждого места — непустой и неповторяющийся id")
    rank = {level: n for n, level in enumerate(LEVELS)}
    rows = []
    for it in items:
        a = assess(it, context)
        rows.append({"id": it["id"], "t": _time_key(it), "a": a,
                     "order": (rank[a["level"]], -a["score"], _time_key(it), it["id"])})
    budget = float(minutes) * 60.0
    left = budget
    taken = {}

    def take(row, block, pick):
        taken[row["id"]] = {"id": row["id"], "level": row["a"]["level"], "block": block, "pick": pick,
                            "seconds": row["a"]["seconds"], "_t": row["t"]}

    def fill(candidates, allowance, block_of, pick="plan"):
        """Берёт места по порядку, пока каждое помещается в allowance; -> неизрасходованные секунды."""
        for row in candidates:
            if row["id"] in taken:
                continue
            need = row["a"]["seconds"]
            if need <= allowance:
                take(row, block_of(row), pick)
                allowance -= need
        return allowance

    main = sorted((r for r in rows if r["a"]["queue"] == QUEUE_MAIN), key=lambda r: r["order"])
    for row in main:
        if row["a"]["signs"]["long_gap"]:
            take(row, "required", "plan")
            left -= row["a"]["seconds"]
    level_block = {"required": "required", "wanted": "wanted", "optional": "optional"}

    def block_of(row):
        a = row["a"]
        return "speaker" if a["level"] == "wanted" and a["pool"] == "speaker" else level_block[a["level"]]

    if first_portion:
        pools = {
            "words": ([r for r in main if r["a"]["pool"] == "words" and r["a"]["level"] != "optional"],
                      block_of, "plan"),
            "speaker": ([r for r in main if r["a"]["pool"] == "speaker"], block_of, "plan"),
            "random": (sorted((r for r in main if r["a"]["level"] != "required"),
                              key=lambda r: (_random_key(seed, r["id"]), r["id"])), lambda r: "random", "random"),
        }
        total = sum(n for _, n in SHARES)
        # пропуски от 2 с входят в долю слов; их превышение и неизрасходованное время переходят дальше
        carry = -(budget - left)
        for name, n in SHARES:
            candidates, blocks, pick = pools[name]
            allowance = budget * n / total + carry
            carry = fill(candidates, allowance, blocks, pick) if allowance > 0 else allowance
        left = budget - sum(t["seconds"] for t in taken.values())
    fill(main, max(0.0, left), block_of)

    block_rank = {b: n for n, b in enumerate(BLOCKS)}
    selected = sorted(taken.values(), key=lambda t: (block_rank[t["block"]], t["_t"], t["id"]))
    for t in selected:
        del t["_t"]
    remaining = {level: 0 for level in LEVELS}
    for row in main:
        if row["id"] not in taken:
            remaining[row["a"]["level"]] += 1
    estimate = float(sum(t["seconds"] for t in selected))
    overflow = max(0.0, estimate - budget)
    return {"version": PLAN_VERSION, "selected": selected, "remaining_by_level": remaining,
            "overlap_queue": sum(1 for r in rows if r["a"]["queue"] == QUEUE_OVERLAP),
            "shown": len(selected), "estimate_seconds": estimate, "budget_seconds": budget,
            "overflow_seconds": overflow, "gaps_over_budget": overflow > 0}
