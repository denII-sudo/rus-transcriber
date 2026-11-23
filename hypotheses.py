"""Формат гипотез моделей hyp-2 (РАР-4, этап 4.2, шаг 3): сборка записи и чтение записей обеих версий.

Гипотеза — то, что выдала каждая модель до голосования: слова основной и второй модели со временем и
уверенностью, текст кусков третьей и дополнительных моделей, сведения о сбойных кусках, правило и
протокол голосования, технические сведения (программа, окружение, компьютер, условия счёта).

Здесь только форма записи: модуль ничего не пишет на диск, не обращается к моделям и не опрашивает
компьютер — сведения ему передаёт вызывающий. Им пользуются лаборатория (lab/lab_run.py) и,
в следующем PR шага 3, расшифровка (worker.py, по настройке save_hypotheses).

hyp-2 — надмножество прежнего lab-hyp-1: верхние ключи combine_rule, primary, second, third, vote
остаются теми же, добавляются schema="hyp-2", roles (одинаковые блоки всех ролей), combine, vote_trace и
блоки сведений (meta). Отсутствие сведения записывается явно: {"status": "not_available" | "not_measured",
"reason": ...} — «не бывает у этой модели» и «не измерено», а не молчаливый пропуск.
"""
SCHEMA = "hyp-2"
LEGACY_SCHEMA = "lab-hyp-1"
KNOWN_SCHEMAS = (SCHEMA, LEGACY_SCHEMA)

# поля блока роли; у каждой роли присутствуют все
ROLE_FIELDS = ("role", "model", "engine", "timing", "chunks", "words", "word_times")
# блоки сведений о прогоне; если вызывающий их не передал — записываются как not_measured
META_FIELDS = ("audio", "program", "environment", "hardware", "conditions", "config", "ids")


def not_available(reason):
    """Сведения не бывают у этой модели (например, времена слов у Whisper)."""
    return {"status": "not_available", "reason": str(reason)}


def not_measured(reason):
    """Сведения бывают, но в этом прогоне не получены."""
    return {"status": "not_measured", "reason": str(reason)}


def available():
    """Признак «у слов этой модели есть времена и уверенность»."""
    return {"status": "available"}


def is_missing(value):
    """Явная отметка отсутствия сведений (not_available / not_measured)."""
    return isinstance(value, dict) and set(value) == {"status", "reason"} \
        and value["status"] in ("not_available", "not_measured")


def role_block(role, model=None, engine=None, timing=None, chunks=None, words=None, word_times=None):
    """Блок роли одинаковой формы для основной, второй, третьей и дополнительных моделей.

    model — имя и (если известны) артефакт; engine — движок, устройство, точность, откаты; timing — времена
    загрузки и прохода; chunks — куски и их состояния (у третьей модели с отметкой heard); words — слова
    {w, s, e, c} до голосования; word_times — признак, есть ли у слов времена (available / not_available /
    not_measured). Всё, что не передано, записывается как not_measured: поле не пропускается."""
    def field(value, reason):
        return not_measured(reason) if value is None else value
    return {"role": role,
            "model": field(model, "модель не передана"),
            "engine": field(engine, "движок не передан"),
            "timing": field(timing, "время не передано"),
            "chunks": field(chunks, "куски не переданы"),
            "words": field(words, "слова не переданы"),
            "word_times": field(word_times, "не определено")}


def build_record(roles, legacy=None, combine=None, trace=None, meta=None):
    """Запись гипотез hyp-2.

    roles — список аргументов role_block (словарей); legacy — верхние ключи lab-hyp-1 (combine_rule, primary,
    second, third, vote), которые читает лаборатория; combine — правило и сведения о сведении; trace —
    протокол голосования (words.vote_n, trace=...); meta — блоки сведений о прогоне из META_FIELDS
    (передаются целиком, ничего не опрашивается здесь). -> словарь, пригодный для json.dumps."""
    names = [r["role"] for r in roles]
    if len(set(names)) != len(names):
        raise ValueError(f"роли гипотез должны быть разными: {names!r}")
    meta = meta or {}
    unknown = set(meta) - set(META_FIELDS)
    if unknown:
        raise ValueError(f"неизвестные блоки сведений: {sorted(unknown)!r}")
    rec = {"schema": SCHEMA, **(legacy or {}), "schema": SCHEMA}
    for name in META_FIELDS:
        value = meta.get(name)
        rec[name] = not_measured("не передано") if value is None else value
    rec["roles"] = [role_block(**r) for r in roles]
    rec["combine"] = not_measured("не передано") if combine is None else combine
    rec["vote_trace"] = not_measured("протокол не вёлся") if trace is None else trace
    return rec


def read(obj):
    """Запись гипотез любой известной версии -> словарь или {} для чужой схемы и не словаря.

    hyp-2 и lab-hyp-1 читаются как есть: верхние ключи у них одни (combine_rule, primary, second, third,
    vote); блоки roles и сведений в lab-hyp-1 отсутствуют и не выдумываются. Записи без поля schema (базы
    лаборатории до отметки формата) тоже читаются как прежде."""
    if not isinstance(obj, dict) or obj.get("schema", SCHEMA) not in KNOWN_SCHEMAS:
        return {}
    return obj


def check(rec):
    """Проверка формы записи hyp-2. -> список замечаний (пустой — запись верна)."""
    bad = []
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
        return [f"schema не {SCHEMA}"]
    for name in META_FIELDS + ("combine", "vote_trace"):
        if name not in rec:
            bad.append(f"нет блока {name}")
    roles = rec.get("roles")
    if not isinstance(roles, list):
        return bad + ["roles — не список"]
    seen = set()
    for r in roles:
        if not isinstance(r, dict) or tuple(r) != ROLE_FIELDS:
            bad.append(f"блок роли {r.get('role') if isinstance(r, dict) else r!r}: поля не {ROLE_FIELDS!r}")
            continue
        if r["role"] in seen:
            bad.append(f"роль {r['role']!r} повторена")
        seen.add(r["role"])
    return bad
