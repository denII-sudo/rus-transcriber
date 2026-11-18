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
}


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
