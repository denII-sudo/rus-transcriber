"""Экран: история ревизий, конспекты записи, исправление цитат и возврат ревизии для локального экрана.

Перенесено из core.py без смены логики (разрезка core.py, перенос 2). Прежние имена `core.X` работают: core.py
в конце файла переотдаёт их отсюда. Из core.py вне своего раздела берётся только `core.CoreError` — как `core.X`
в момент вызова, а не копией. `import core` — последняя строка файла (круговой импорт: core.py в конце загружает
этот модуль). Проверяет tests/test_core_split.py.
"""
import time
import uuid
from pathlib import Path


def workflow_list(limit=20):
    """Последние сохранённые запросы для локального экрана; без тяжёлых чтений."""
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        store.current_identity()
        return workflow.list_recent(store, limit)
    finally:
        store.close()


def recording_revisions(recording_id, kind):
    """История ревизий записи (transcript/review) с происхождением; хэши снимков проверяются."""
    import recordings
    if kind not in ("transcript", "review"):
        raise core.CoreError("некорректный тип ревизии")
    store = recordings.open_store()
    try:
        store.current_identity()
        result = []
        for revision in store.list_revisions(recording_id, kind):
            data = store.get_revision(recording_id, kind, revision)
            context = data["context"]
            item = {"kind": kind, "revision": revision, "restored_from": data["restored_from"],
                    "actor_kind": context.get("actor_kind"),
                    "initiated_by": (context.get("initiated_by") or {}).get("user_name_snapshot")}
            row = store.db.execute("SELECT created_at FROM operations WHERE operation_id=?",
                                   (data["operation_id"],)).fetchone()
            item["created_at"] = row["created_at"] if row else None
            if kind == "review":
                item["source_revision"] = data["source_revision"]
            else:
                paired = store.db.execute(
                    "SELECT revision FROM review_revisions WHERE recording_id=? AND source_revision=? "
                    "ORDER BY revision DESC LIMIT 1", (recording_id, revision)).fetchone()
                item["restore_review_revision"] = paired["revision"] if paired else None
            result.append(item)
        return result
    finally:
        store.close()


def read_recording_revision(recording_id, kind, revision):
    """Прочитать сохранённую ревизию для просмотра истории; текущая ревизия не меняется."""
    import recordings
    if kind not in ("transcript", "review"):
        raise core.CoreError("некорректный тип ревизии")
    store = recordings.open_store()
    try:
        store.current_identity()
        data = store.get_revision(recording_id, kind, int(revision))
        context = data["context"]
        return {"kind": kind, "revision": data["revision"], "restored_from": data["restored_from"],
                "payload": data["payload"], "actor_kind": context.get("actor_kind"),
                "initiated_by": (context.get("initiated_by") or {}).get("user_name_snapshot")}
    finally:
        store.close()


def _conspect_screen_sections(payload):
    """Разделы конспекта для экрана. Прежнее поле `sections` отдаётся как есть; если его нет,
    берутся тексты из `evidence` (так исполнитель свёртки сохраняет разделы), без номеров слов."""
    sections = payload.get("sections")
    if isinstance(sections, list):
        return sections
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return []
    return [{"text": item["text"]} for item in evidence
            if isinstance(item, dict) and isinstance(item.get("text"), str)]


def quote_repair_view(payload):
    """Цитаты свёртки «по одной» для строки экрана: сколько не найдено в стенограмме и достигнут ли порог
    исправления (порог и счёт — workflow.quotes_need_repair, workflow.quote_counts). None — у остальных свёрток."""
    import workflow
    counts = workflow.quote_counts(payload)
    if counts is None:
        return None
    return {"unlinked": counts[0], "total": counts[1], "needed": workflow.quotes_need_repair(*counts)}


def quote_repair_state(store, name):
    """Для кнопки на экране: запрос свёртки, доступно ли исправление цитат и почему нет. Только чтение."""
    import storage
    import workflow
    if not name.startswith("workflow-"):
        return {"request_id": None, "available": False,
                "reason": "это не свёртка запроса; исправление цитат недоступно"}
    request_id = name[len("workflow-"):]
    try:
        problem = workflow.quote_repair_problem(store, workflow.get(store, request_id), screen=True)
    except (KeyError, ValueError, storage.StorageError):
        return {"request_id": request_id, "available": False, "reason": "запрос этой свёртки не найден"}
    return {"request_id": request_id, "available": problem is None, "reason": problem}


def workflow_repair_quotes_prepare(request_id):
    """Снимок для окна согласия на экране: что уйдёт в облако. Ничего не отправляет."""
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return workflow.quote_repair_prepare(store, request_id)
    finally:
        store.close()


def workflow_repair_quotes_info(request_id):
    """MCP без согласия: доступно ли исправление, почему нет, сколько уйдёт. Ничего не отправляет."""
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return workflow.quote_repair_info(store, request_id)
    finally:
        store.close()


def workflow_repair_quotes(request_id, *, via, snapshot=None, allow_cloud_model=False):
    """Один вызов облака по явному согласию человека (экран: со снимком; MCP: с allow_cloud_model=true)."""
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return workflow.repair_quotes(store, request_id, via=via, snapshot=snapshot,
                                      allow_cloud_model=allow_cloud_model)
    finally:
        store.close()


def recording_conspects(recording_id):
    """Конспекты записи для локального экрана: имя, ревизии, разделы и пути файлов."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        result = []
        for item in store.list_conspects(recording_id):
            root = store.db.execute("SELECT path FROM allowed_roots WHERE root_id=?",
                                    (item["root_id"],)).fetchone()
            locations = store.db.execute(
                "SELECT kind, relative_path FROM conspect_file_locations WHERE conspect_id=?",
                (item["conspect_id"],)).fetchall()
            files = {loc["kind"]: str(Path(root["path"]) / loc["relative_path"])
                     for loc in locations} if root else {}
            result.append({"conspect_id": item["conspect_id"], "name": item["name"],
                           "current_revision": item["current_revision"],
                           "source_revision": item["source_revision"],
                           "sections": _conspect_screen_sections(item["payload"]),
                           "quote_repair": (None if quote_repair_view(item["payload"]) is None else
                                            {**quote_repair_view(item["payload"]),
                                             **quote_repair_state(store, item["name"])}),
                           "files": files})
        return result
    finally:
        store.close()


def prepare_restore_snapshot(recording_id, source_revision, review_revision):
    """Снимок для возврата прежней ревизии парой; подтверждение человека привязано к снимку."""
    import recordings
    import storage
    store = recordings.open_store()
    try:
        store.current_identity()
        head = recordings.recording_head(store, recording_id)
        if head is None:
            raise KeyError(recording_id)
        expected = {"source_revision": head["source_revision"], "review_revision": head["review_revision"]}
        restored_review = store.get_revision(recording_id, "review", int(review_revision))
        if restored_review["source_revision"] != int(source_revision):
            raise storage.Conflict("пара ревизий несовместима")
        store.get_revision(recording_id, "transcript", int(source_revision))
        binding = recordings.document_binding(store, recording_id)
        return {"recording_id": recording_id, "source_revision": int(source_revision),
                "review_revision": int(review_revision), "expected_versions": expected,
                "document_revision": binding["document_revision"], "root_id": binding["root_id"],
                "document": str(binding["root"] / binding["paths"]["md"]),
                "context": recordings.operation_context(store, "human")}
    finally:
        store.close()


def restore_recording_revision(snapshot):
    """Вернуть прежнюю ревизию как новую текущую; история сохраняется, файлы формируются заново."""
    import documents
    import recordings
    if not isinstance(snapshot, dict) or "recording_id" not in snapshot:
        raise core.CoreError("неверный снимок возврата")
    current = prepare_restore_snapshot(snapshot["recording_id"], snapshot["source_revision"],
                                       snapshot["review_revision"])
    if current != snapshot:
        raise core.CoreError("состояние записи изменилось после запроса возврата")
    store = recordings.open_store()
    try:
        store.current_identity()
        documents.guard_before_write(store, snapshot["recording_id"])
        result = store.restore_with_document(
            snapshot["recording_id"], snapshot["source_revision"], snapshot["review_revision"],
            snapshot["expected_versions"], str(uuid.uuid4()), snapshot["context"],
            root_id=snapshot["root_id"],
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        completed = recordings.materialize_recording(store, snapshot["recording_id"])
        failure = documents.unsynced(store, snapshot["recording_id"], result["document_revision"], completed)
        result["materialized"] = failure is None
        result["outcome"] = "written" if failure is None else failure.status
        if failure is not None:
            result["reason"] = str(failure)
        return result
    except (OSError, ValueError, KeyError) as ex:
        raise core.CoreError(f"возврат не выполнен или формирование не завершено: {ex}") from ex
    finally:
        store.close()


import core  # noqa: E402  (последним: core.py в конце загружает этот модуль)
