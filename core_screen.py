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


RECORDING_LIST_LIMIT = 500    # больше строк экран не показывает: говорит «показаны N из M»

# Время последнего изменения записи: наибольшее время операций текущих ревизий текста и разбора.
_UPDATED_AT = ("MAX(COALESCE((SELECT o.created_at FROM transcript_revisions t JOIN operations o USING(operation_id) "
               "WHERE t.recording_id=r.recording_id AND t.revision=r.source_revision), ''), "
               "COALESCE((SELECT o.created_at FROM review_revisions v JOIN operations o USING(operation_id) "
               "WHERE v.recording_id=r.recording_id AND v.revision=r.review_revision), ''))")


def _json_flag(kind):
    """Значение json_type как логическое: 'true'/'false' или None, если поля нет или оно не логическое."""
    return {"true": True, "false": False}.get(kind)


def _recording_row(store, recording_id):
    """Строка списка записей: сводка без текста реплик, имён говорящих и путей. Поля стенограммы берутся из JSON
    в самой базе (json_extract), стенограмма целиком в Python не разбирается и хэш снимка здесь не проверяется —
    это делает открытие записи. Строку, которую не прочитать, помечает ошибкой, а не роняет весь список."""
    import json
    import re
    import sqlite3
    import storage
    import transcript
    item = {"recording_id": recording_id}
    try:
        row = store.db.execute(
            "SELECT r.location, r.source_revision, r.review_revision, " + _UPDATED_AT + " AS updated_at, "
            "json_extract(t.payload, '$.file') AS file, json_extract(t.payload, '$.title') AS title, "
            "json_extract(t.payload, '$.recorded_at') AS recorded_at, "
            "json_extract(t.payload, '$.duration') AS duration, "
            "json_type(t.payload, '$.names_confirmed') AS names_confirmed, "
            "json_type(t.payload, '$.quality.migration') AS migration, "
            "json_extract(t.payload, '$.quality.migration.variant.tag') AS variant_tag, "
            "json_type(t.payload, '$.quality.migration.variant.main') AS variant_main, "
            "json_extract(v.payload, '$.review') AS review, t.revision AS has_text, v.revision AS has_review "
            "FROM recordings r "
            "LEFT JOIN transcript_revisions t ON t.recording_id=r.recording_id AND t.revision=r.source_revision "
            "LEFT JOIN review_revisions v ON v.recording_id=r.recording_id AND v.revision=r.review_revision "
            "WHERE r.recording_id=?", (recording_id,)).fetchone()
        if row is None:
            raise storage.StorageError("записи больше нет в базе")
        if row["has_text"] is None:
            raise storage.StorageError("нет текущей ревизии текста")
        if row["has_review"] is None:
            raise storage.StorageError("нет текущей ревизии разбора")
        source = row["file"] if isinstance(row["file"], str) and row["file"] else row["location"]
        item.update(
            name=re.split(r"[\\/]", source)[-1],    # только имя файла: папку и путь к аудио экран не показывает
            title=row["title"] if isinstance(row["title"], str) else None,
            recorded_at=row["recorded_at"] if isinstance(row["recorded_at"], str) else None,
            duration=row["duration"] if isinstance(row["duration"], (str, int, float)) else None,
            updated_at=row["updated_at"] or None,
            source_revision=row["source_revision"], review_revision=row["review_revision"],
            names_confirmed=_json_flag(row["names_confirmed"]))
        if row["migration"] == "object":
            item["origin"] = "migrated"
            item["variant"] = {"tag": row["variant_tag"] if isinstance(row["variant_tag"], str) else None,
                               "main": _json_flag(row["variant_main"])}
        else:
            item["origin"] = "program"
            item["variant"] = None
        places = json.loads(row["review"]) if isinstance(row["review"], str) else []
        places = [place for place in places if isinstance(place, dict)] if isinstance(places, list) else []
        # правило «открыто» и «не решено» — общее для программы (transcript.place_*), без своей копии
        item["places"] = {"unresolved": sum(1 for place in places if transcript.place_unresolved(place)),
                          "open": sum(1 for place in places if transcript.place_open(place)),
                          "total": len(places)}
        outbox = store.db.execute(
            "SELECT state FROM document_outbox WHERE recording_id=? ORDER BY document_revision DESC LIMIT 1",
            (recording_id,)).fetchone()
        item["document_sync"] = outbox["state"] if outbox else None
        item["open_conflicts"] = store.db.execute(
            "SELECT (SELECT COUNT(*) FROM document_conflicts WHERE recording_id=? AND resolution_operation_id IS NULL) "
            "+ (SELECT COUNT(*) FROM conspect_conflicts k JOIN conspects c USING(conspect_id) "
            "WHERE c.recording_id=? AND k.resolution_operation_id IS NULL)",
            (recording_id, recording_id)).fetchone()[0]
        item["conspects"] = store.db.execute("SELECT COUNT(*) FROM conspects WHERE recording_id=?",
                                             (recording_id,)).fetchone()[0]
        request = store.db.execute(
            "SELECT request_id, state FROM workflow_requests WHERE recording_id=? "
            "ORDER BY updated_at DESC, request_id LIMIT 1", (recording_id,)).fetchone()
        item["request"] = dict(request) if request else None
        item["error"] = None
    except (sqlite3.Error, storage.StorageError, ValueError, TypeError) as ex:
        text = str(ex).lower()
        if isinstance(ex, storage.Busy) or "locked" in text or "busy" in text:
            raise    # занятая база — отказ всего списка с пояснением «подождите», а не «строка не читается»
        name = item.get("name")
        if name is None:    # JSON ревизии не читается: имя — из пути в базе (он хранится в нижнем регистре)
            location = store.db.execute("SELECT location FROM recordings WHERE recording_id=?",
                                        (recording_id,)).fetchone()
            name = re.split(r"[\\/]", location["location"])[-1] if location else None
        item = {"recording_id": recording_id, "name": name,
                "error": f"Запись не читается: {core.explain_error(ex)}"}
    return item


def recording_list():
    """Все записи базы для локального экрана, новые изменения сверху; ничего не пишет и не требует запроса.

    Сводка без текста, имён говорящих и путей к аудио (только имя файла). Не больше RECORDING_LIST_LIMIT строк."""
    import paths
    import recordings
    import settings
    try:
        backend = settings.recording_backend()
    except ValueError:
        backend = None
    if not Path(paths.STATE_DB).is_file():
        reason = "Базы записей ещё нет: показывать нечего."
        if backend != "sqlite":
            reason += (" Программа сейчас хранит записи в файлах вне базы: такие записи здесь не видны. "
                       "В базу они не переносятся; их файлы можно прочитать: rt.py legacy-read.")
        return {"available": False, "items": [], "total": 0, "truncated": False, "reason": reason,
                "backend": backend}
    store = recordings.open_store(readonly=True)    # только чтение (query_only)
    try:
        store.current_identity()
        store.db.execute("BEGIN")    # одна транзакция чтения: строки списка — из одного состояния базы
        try:
            total = store.db.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
            ids = [row["recording_id"] for row in store.db.execute(
                "SELECT r.recording_id FROM recordings r ORDER BY " + _UPDATED_AT + " DESC, r.recording_id LIMIT ?",
                (RECORDING_LIST_LIMIT,))]
            items = [_recording_row(store, recording_id) for recording_id in ids]
        finally:
            if store.db.in_transaction:
                store.db.execute("ROLLBACK")
        return {"available": True, "items": items, "total": total, "truncated": total > len(items),
                "reason": None, "backend": backend}
    finally:
        store.close()


def recording_view(recording_id):
    """Одна запись по устойчивому ID для экрана, без запроса: сводка, запросы записи и текущая пара ревизий."""
    import recordings
    import storage
    try:
        storage._uuid(recording_id)
    except ValueError:
        raise core.CoreError("Неверный номер записи.") from None
    store = recordings.open_store(readonly=True)    # только чтение (query_only)
    try:
        store.current_identity()
        if recordings.recording_head(store, recording_id) is None:
            raise core.CoreError("Такой записи в базе нет: обновите список.")
        header = _recording_row(store, recording_id)
        requests = [dict(row) for row in store.db.execute(
            "SELECT request_id, state, created_at, updated_at FROM workflow_requests WHERE recording_id=? "
            "ORDER BY updated_at DESC, request_id", (recording_id,))]
    finally:
        store.close()
    return {"recording": header, "requests": requests, "data": core.read_recording_result(recording_id)}


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


def _conspect_screen_sections(payload, data_of=None):
    """Разделы конспекта для экрана. Прежнее поле `sections` отдаётся как есть.

    Свёртка исполнителя (есть `evidence`, `template` и `quotes`) показывается так, как её печатает файл:
    conspect.render_body при текущей стенограмме — кавычки только у проверенных цитат, остальные с пометкой
    «по распознаванию, не проверено» и метками участников. Сырой ответ модели в `evidence` не показывается
    (О255-М1). data_of() — текущая стенограмма с разбором (читается один раз на запись). Если собрать не вышло,
    вместо текста — сообщение, а не сырой ответ. Без data_of и у остальных конспектов берутся тексты из `evidence`.
    """
    sections = payload.get("sections")
    if isinstance(sections, list):
        return sections
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return []
    if data_of is not None and isinstance(payload.get("template"), str) and isinstance(payload.get("quotes"), dict):
        import conspect
        import storage
        try:
            body = conspect.render_body(payload, data_of())
        except (KeyError, ValueError, TypeError, IndexError, storage.StorageError):
            return [{"text": "Не удалось собрать текст свёртки для показа. Сырой ответ модели не показан: в нём "
                             "цитаты стоят в кавычках без проверки. Откройте файл свёртки."}]
        return [{"text": part} for part in body.split("\n\n") if part.strip()]
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
    import core_conflicts
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        result = []
        loaded = []

        def data_of():       # стенограмма с разбором, как её читает пересборка файла; один раз на запись
            if not loaded:
                loaded.append(recordings.load_recording(store, recording_id))
            return loaded[0]
        for item in store.list_conspects(recording_id):
            root = store.db.execute("SELECT path FROM allowed_roots WHERE root_id=?",
                                    (item["root_id"],)).fetchone()
            locations = store.db.execute(
                "SELECT kind, relative_path FROM conspect_file_locations WHERE conspect_id=?",
                (item["conspect_id"],)).fetchall()
            # база хранит путь корня приведённым к нижнему регистру; для показа берём регистр с диска
            case_cache = {}
            files = {loc["kind"]: str(core_conflicts._real_case(Path(root["path"]) / loc["relative_path"],
                                                                case_cache))
                     for loc in locations} if root else {}
            result.append({"conspect_id": item["conspect_id"], "name": item["name"],
                           "current_revision": item["current_revision"],
                           "source_revision": item["source_revision"],
                           "sections": _conspect_screen_sections(item["payload"], data_of),
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
