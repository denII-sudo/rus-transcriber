"""Чистая оценка пригодности результата; технический успех не равен проверке человеком."""

DEFAULTS = {
    "require_conspect": False,
    # Облако только по явному согласию человека в конкретном запросе (решение владельца 18.11.2026).
    "allow_cloud_model": False,
    "require_named_speakers": False,
    "require_human_verification": False,
    "question_budget": 0,
    "participation": "simple",
    "goal": "",
    # Подсказки «кто на записи» из формы до расшифровки. Только для разбора на говорящих;
    # в задание модели и в облако не попадают (model_tasks берёт из политики лишь goal).
    "speaker_hints": None,
}

HINT_KEYS = frozenset(("count_mode", "count", "participants", "stray_voices", "group", "enroll_profile"))
_COUNT_MODES = ("unknown", "exact", "approx")


def normalize_speaker_hints(hints):
    """Проверить подсказки формы; None — подсказок нет (прежнее поведение)."""
    if hints is None:
        return None
    if not isinstance(hints, dict) or set(hints) - HINT_KEYS:
        raise ValueError("speaker_hints: неизвестные поля")
    mode = hints.get("count_mode", "unknown")
    if mode not in _COUNT_MODES:
        raise ValueError("speaker_hints.count_mode: unknown, exact или approx")
    count = hints.get("count", 0)
    if mode == "unknown":
        count = 0
    elif isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 30:
        raise ValueError("speaker_hints.count: целое число от 1 до 30")
    participants = hints.get("participants", [])
    if not isinstance(participants, list) or len(participants) > 30:
        raise ValueError("speaker_hints.participants: список не длиннее 30")
    clean, seen = [], set()
    for item in participants:
        if not isinstance(item, dict) or set(item) - {"name", "forms"}:
            raise ValueError("speaker_hints.participants: ожидаются name и forms")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80 or "," in name or ";" in name:
            raise ValueError("speaker_hints.participants: имя — текст до 80 знаков без запятых")
        name = name.strip()
        if name.casefold() in seen:
            raise ValueError("speaker_hints.participants: имя повторяется")
        seen.add(name.casefold())
        forms = item.get("forms", [])
        if not isinstance(forms, list) or len(forms) > 10 or any(
                not isinstance(f, str) or not f.strip() or len(f.strip()) > 80 for f in forms):
            raise ValueError("speaker_hints.participants: формы обращения — до 10 строк до 80 знаков")
        clean.append({"name": name, "forms": [f.strip() for f in forms]})
    stray = hints.get("stray_voices", False)
    enroll = hints.get("enroll_profile", True)
    if not isinstance(stray, bool) or not isinstance(enroll, bool):
        raise ValueError("speaker_hints: stray_voices и enroll_profile — логические значения")
    group = hints.get("group", "")
    if not isinstance(group, str) or len(group) > 80:
        raise ValueError("speaker_hints.group: текст до 80 знаков")
    return {"count_mode": mode, "count": count, "participants": clean,
            "stray_voices": stray, "group": group.strip(), "enroll_profile": enroll}


def diarization_bounds(hints):
    """(num_speakers, min_speakers, max_speakers) из подсказок: точно N; примерно N → N-1…N+1; запас голосов +1/+3."""
    if not hints or hints["count_mode"] == "unknown":
        return 0, 0, 0
    n = hints["count"]
    if hints["count_mode"] == "exact":
        return n, 0, 0
    return 0, max(1, n - 1), n + 1


def resolve(overrides=None):
    """Проверить и сохранить явную политику без угадывания её по роли человека."""
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides) - set(DEFAULTS):
        raise ValueError("неизвестные требования к результату")
    result = {**DEFAULTS, **overrides}
    for key in ("require_conspect", "allow_cloud_model", "require_named_speakers", "require_human_verification"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key}: требуется логическое значение")
    budget = result["question_budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= 100:
        raise ValueError("question_budget: требуется целое число от 0 до 100")
    if result["participation"] not in ("simple", "advanced", "developer"):
        raise ValueError("неизвестный режим участия")
    if not isinstance(result["goal"], str) or len(result["goal"]) > 2000:
        raise ValueError("цель должна быть текстом длиной не более 2000 символов")
    result["speaker_hints"] = normalize_speaker_hints(result["speaker_hints"])
    return result


def evaluate(policy, *, execution, document_sync, unresolved=0, unnamed_speakers=0,
             human_verification="none", conspect=False, incomplete=False):
    """Раздельные состояния и консервативный вердикт для конкретного снимка."""
    policy = resolve(policy)
    if execution not in {"queued", "running", "waiting_model", "interrupted", "failed", "completed"}:
        raise ValueError("неизвестное состояние выполнения")
    if document_sync not in {"not_started", "pending", "writing", "synced", "conflict", "error"}:
        raise ValueError("неизвестное состояние документов")
    if human_verification not in {"none", "partial", "confirmed"}:
        raise ValueError("неизвестное состояние проверки человеком")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (unresolved, unnamed_speakers)):
        raise ValueError("счётчики должны быть неотрицательными целыми числами")
    if not isinstance(conspect, bool) or not isinstance(incomplete, bool):
        raise ValueError("свидетельства должны быть логическими значениями")
    blockers, limits = [], []
    if execution != "completed":
        blockers.append("обработка не завершена")
    if document_sync != "synced":
        blockers.append("документы не синхронизированы")
    if policy["require_conspect"] and not conspect:
        blockers.append("заказанный конспект отсутствует")
    if policy["require_named_speakers"] and unnamed_speakers:
        blockers.append("не все говорящие определены")
    if policy["require_human_verification"] and human_verification != "confirmed":
        blockers.append("проверка человеком не подтверждена")
    if incomplete:
        limits.append("обработка может быть неполной")
    if unresolved:
        limits.append("есть неразрешённые сомнения")
    quality_limited = bool(limits)
    if human_verification != "confirmed":
        limits.append("результат не подтверждён человеком")
    return {
        "execution": execution,
        "deliverable": "blocked" if blockers else ("usable_with_limits" if quality_limited else "ready_for_request"),
        "review": {"unresolved": unresolved, "unnamed_speakers": unnamed_speakers},
        "human_verification": human_verification,
        "document_sync": document_sync,
        "blockers": blockers,
        "limits": limits,
    }
