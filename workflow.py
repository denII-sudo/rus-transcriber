"""Сохранённые запросы П3. Исполнение вызывается сервисом, не при импорте."""

import json
import math
import uuid
from pathlib import Path

import model_tasks
import product_policy
import recordings
import storage


def create(store, source, policy=None):
    """Принять запрос и зафиксировать инициатора до запуска worker."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise ValueError("исходный файл не найден")
    policy = product_policy.resolve(policy)
    request_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
    context = recordings.operation_context(store, "model")
    now = storage._now()
    with store._transaction():
        store.current_identity()
        store.db.execute(
            "INSERT INTO workflow_requests VALUES (?, ?, ?, ?, 'queued', NULL, NULL, NULL, NULL, 0, 1, NULL, ?, ?)",
            (request_id, str(source), storage._json(policy), storage._json(context), now, now))
        store.db.execute("INSERT INTO workflow_policy_versions VALUES (?, 1, ?, ?, ?)",
                         (request_id, storage._json(policy), storage._json(context), now))
        store.db.execute("INSERT INTO workflow_runs VALUES (?, ?, 'asr', 'queued', 'worker', ?, ?)",
                         (run_id, request_id, now, now))
        store.db.execute("INSERT INTO workflow_steps VALUES (?, ?, 'asr', 'queued', NULL, ?, ?)",
                         (str(uuid.uuid4()), run_id, now, now))
    return request_id


def list_recent(store, limit=20):
    """Короткий список последних запросов для локального экрана."""
    limit = max(0, min(int(limit), 100))
    return [dict(row) for row in store.db.execute(
        "SELECT request_id, state, source, recording_id, policy_revision, created_at, updated_at "
        "FROM workflow_requests ORDER BY updated_at DESC LIMIT ?", (limit,))]


def policy_head(store, request_id):
    """Текущая ревизия политики запроса: строка (policy_revision) или None, если запроса нет."""
    return store.db.execute("SELECT policy_revision FROM workflow_requests WHERE request_id=?",
                            (request_id,)).fetchone()


def get(store, request_id):
    storage._uuid(request_id)
    store.current_identity()
    row = store.db.execute("SELECT * FROM workflow_requests WHERE request_id=?", (request_id,)).fetchone()
    if row is None:
        raise KeyError(request_id)
    result = dict(row)
    for name in ("policy", "context", "model_task", "model_result"):
        value = result.pop(name + "_json")
        result[name] = storage._loads(value) if value is not None else None
    result["runs"] = [dict(item) for item in store.db.execute(
        "SELECT * FROM workflow_runs WHERE request_id=? ORDER BY created_at", (request_id,))]
    result["steps"] = [dict(item) for item in store.db.execute(
        "SELECT s.* FROM workflow_steps s JOIN workflow_runs r USING(run_id) "
        "WHERE r.request_id=? ORDER BY s.created_at", (request_id,))]
    result["policy_history"] = [
        {"revision": item["revision"], "policy": storage._loads(item["policy_json"]),
         "context": storage._loads(item["context_json"]), "created_at": item["created_at"]}
        for item in store.db.execute("SELECT * FROM workflow_policy_versions WHERE request_id=? ORDER BY revision",
                                     (request_id,))]
    result["assessment_history"] = [
        {"assessment": storage._loads(item["payload_json"]), "created_at": item["created_at"]}
        for item in store.db.execute(
            "SELECT payload_json, created_at FROM workflow_assessments WHERE request_id=? "
            "ORDER BY created_at DESC LIMIT 50", (request_id,))]
    return result


def change_policy(store, request_id, policy, expected_revision):
    """Сохранить новую политику без повторного запуска ASR и без перезаписи истории."""
    storage._uuid(request_id)
    policy = product_policy.resolve(policy)
    if type(expected_revision) is not int or expected_revision < 1:
        raise ValueError("требуется ожидаемая ревизия политики")
    context = recordings.operation_context(store, "unknown")
    with store._transaction():
        store.current_identity()
        row = store.db.execute("SELECT policy_revision, state, policy_json FROM workflow_requests WHERE request_id=?",
                               (request_id,)).fetchone()
        if row is None:
            raise KeyError(request_id)
        if row["policy_revision"] != expected_revision:
            raise storage.Conflict("политика уже изменилась")
        previous_policy = storage._loads(row["policy_json"])
        if not previous_policy["allow_cloud_model"] and policy["allow_cloud_model"]:
            raise storage.Conflict(
                "облачный режим нельзя включить для существующего запроса; создайте новый запрос "
                "с явным разрешением")
        if row["state"] == "waiting_model":
            raise storage.Conflict("сначала завершите текущее модельное задание")
        now, revision = storage._now(), expected_revision + 1
        store.db.execute("INSERT INTO workflow_policy_versions VALUES (?, ?, ?, ?, ?)",
                         (request_id, revision, storage._json(policy), storage._json(context), now))
        store.db.execute("UPDATE workflow_requests SET policy_json=?, policy_revision=?, updated_at=? "
                         "WHERE request_id=?", (storage._json(policy), revision, now, request_id))
    return revision


def verification_snapshot(store, request_id):
    """Снимок для отдельного подтверждения всей стенограммы человеком."""
    import documents
    request = get(store, request_id)
    if not request["recording_id"]:
        raise storage.Conflict("стенограмма ещё не сохранена")
    current = store.get_current(request["recording_id"])
    binding = recordings.document_binding(store, request["recording_id"])
    report = documents.verify_files(store, binding["root_id"], binding["root"])
    files = [item for item in report["entries"] if item["recording_id"] == request["recording_id"]
             and item["kind"] in ("md", "json", "manifest")]
    if len(files) != 3 or any(item["status"] != "ok" or item["outbox_state"] != "synced" for item in files):
        raise storage.Conflict("сначала необходимо синхронизировать документы стенограммы")
    return {"request_id": request_id, "recording_id": request["recording_id"],
            "source_revision": current["transcript"]["revision"],
            "review_revision": current["review"]["revision"],
            "document_revision": binding["document_revision"],
            "document": str(binding["root"] / binding["paths"]["md"]),
            "context": recordings.operation_context(store, "human")}


def confirm_verification(store, snapshot):
    """Фиксировать только после подтверждения из доверенного интерфейса."""
    if not isinstance(snapshot, dict) or "request_id" not in snapshot:
        raise ValueError("неверный снимок подтверждения")
    if verification_snapshot(store, snapshot["request_id"]) != snapshot:
        raise storage.Conflict("стенограмма или автор подтверждения изменились")
    context = snapshot["context"]
    if context["actor_kind"] != "human":
        raise ValueError("неизвестный автор подтверждения")
    ident = str(uuid.uuid5(uuid.UUID(snapshot["request_id"]), storage._hash(snapshot)))
    with store._transaction():
        store.current_identity()
        store._check_context_refs(context)
        head = recordings.recording_head(store, snapshot["recording_id"])
        latest = store.db.execute("SELECT MAX(document_revision) FROM document_versions WHERE recording_id=?",
                                  (snapshot["recording_id"],)).fetchone()[0]
        if (head is None or head["source_revision"] != snapshot["source_revision"]
                or head["review_revision"] != snapshot["review_revision"]
                or latest != snapshot["document_revision"]
                or recordings.operation_context(store, "human") != context):
            raise storage.Conflict("снимок подтверждения устарел")
        store.db.execute("INSERT OR IGNORE INTO workflow_verifications VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (ident, snapshot["request_id"], snapshot["recording_id"], snapshot["source_revision"],
                          snapshot["review_revision"], storage._json(context), storage._now()))
    return {"verification_id": ident, "source_revision": snapshot["source_revision"],
            "review_revision": snapshot["review_revision"], "human_verification": "confirmed"}


def abandon_snapshot(store, request_id):
    """Снимок отказа от незавершённого ASR без удаления данных и остановки процесса."""
    request = get(store, request_id)
    if request["state"] not in ("starting", "running", "interrupted") or request["recording_id"]:
        raise storage.Conflict("запрос уже завершён или результат сохранён")
    return {"request_id": request_id, "job_id": request["job_id"], "state": request["state"],
            "source": request["source"], "context": recordings.operation_context(store, "human")}


def abandon(store, snapshot):
    """Запретить прежнему worker сохранять результат; повторно запускать его нельзя."""
    if not isinstance(snapshot, dict) or "request_id" not in snapshot:
        raise ValueError("неверный снимок отказа")
    if abandon_snapshot(store, snapshot["request_id"]) != snapshot:
        raise storage.Conflict("состояние запроса изменилось")
    with store._transaction():
        store.current_identity()
        store._check_context_refs(snapshot["context"])
        row = store.db.execute("SELECT state, job_id, recording_id FROM workflow_requests WHERE request_id=?",
                               (snapshot["request_id"],)).fetchone()
        if (row is None or row["state"] != snapshot["state"] or row["job_id"] != snapshot["job_id"]
                or row["recording_id"] is not None
                or recordings.operation_context(store, "human") != snapshot["context"]):
            raise storage.Conflict("состояние запроса изменилось")
        now = storage._now()
        store.db.execute("UPDATE workflow_requests SET state='failed', updated_at=? WHERE request_id=?",
                         (now, snapshot["request_id"]))
        store.db.execute("UPDATE workflow_runs SET state='abandoned', updated_at=? "
                         "WHERE request_id=? AND kind='asr'", (now, snapshot["request_id"]))
        store.db.execute("UPDATE workflow_steps SET state='abandoned', result_json=?, updated_at=? WHERE run_id IN "
                         "(SELECT run_id FROM workflow_runs WHERE request_id=? AND kind='asr')",
                         (storage._json({"context": snapshot["context"]}), now, snapshot["request_id"]))
    return {"request_id": snapshot["request_id"], "state": "failed",
            "note": "Старый worker не сможет сохранить результат; процесс отдельно не останавливался."}


def claim_start(store, request_id):
    """Один раз зарезервировать job ID до запуска процесса; повтор не запускает worker."""
    storage._uuid(request_id)
    with store._transaction():
        store.current_identity()
        row = store.db.execute("SELECT state FROM workflow_requests WHERE request_id=?", (request_id,)).fetchone()
        if row is None:
            raise KeyError(request_id)
        if row["state"] != "queued":
            raise storage.Conflict("запрос уже запущен или завершён")
        job_id = "workflow-" + uuid.uuid4().hex
        now = storage._now()
        store.db.execute("UPDATE workflow_requests SET state='starting', job_id=?, updated_at=? WHERE request_id=?",
                         (job_id, now, request_id))
        store.db.execute("UPDATE workflow_runs SET state='running', updated_at=? WHERE request_id=?",
                         (now, request_id))
        store.db.execute("UPDATE workflow_steps SET state='running', updated_at=? WHERE run_id IN "
                         "(SELECT run_id FROM workflow_runs WHERE request_id=?)", (now, request_id))
    return job_id


def mark_launch_interrupted(store, request_id):
    """Исход запуска worker неизвестен: запрос и выполняемый ASR — interrupted, повтора нет."""
    with store._transaction():
        now = storage._now()
        store.db.execute("UPDATE workflow_requests SET state='interrupted', updated_at=? WHERE request_id=?",
                         (now, request_id))
        store.db.execute("UPDATE workflow_runs SET state='interrupted', updated_at=? "
                         "WHERE request_id=? AND kind='asr' AND state='running'",
                         (now, request_id))
        store.db.execute("UPDATE workflow_steps SET state='interrupted', updated_at=? WHERE run_id IN "
                         "(SELECT run_id FROM workflow_runs WHERE request_id=? AND kind='asr') "
                         "AND state='running'", (now, request_id))


def check_worker_result_allowed(store, request_id, context, source):
    """Внутри транзакции сохранения результата worker: запрос всё ещё ждёт результат этого запуска.

    Отказ, если запрос завершён или брошен, уже связан с записью, инициатор или источник другие."""
    row = store.db.execute(
        "SELECT source, context_json, state, recording_id FROM workflow_requests WHERE request_id=?",
        (request_id,)).fetchone()
    if (row is None or row["state"] not in ("starting", "running", "interrupted")
            or row["recording_id"] is not None
            or storage._loads(row["context_json"]) != context
            or recordings.normalize_location(row["source"]) != recordings.normalize_location(source)):
        raise RuntimeError("workflow request no longer permits this worker result")


def link_worker_result(store, request_id, recording_id):
    """В той же транзакции, что и сохранение записи: связать запрос с ней, завершить шаг ASR."""
    now = storage._now()
    store.db.execute("UPDATE workflow_requests SET recording_id=?, state='processed', updated_at=? "
                     "WHERE request_id=?", (recording_id, now, request_id))
    store.db.execute("UPDATE workflow_runs SET state='completed', updated_at=? "
                     "WHERE request_id=? AND kind='asr' AND state='running'",
                     (now, request_id))
    store.db.execute("UPDATE workflow_steps SET state='completed', result_json=?, updated_at=? "
                     "WHERE run_id IN (SELECT run_id FROM workflow_runs "
                     "WHERE request_id=? AND kind='asr') AND state='running'",
                     (storage._json({"recording_id": recording_id}), now, request_id))


def observe(store, request_id, job):
    """Зафиксировать наблюдаемый результат, не повторяя завершённое распознавание."""
    request = get(store, request_id)
    if request["state"] not in ("starting", "running", "interrupted"):
        return request
    if (request["state"] == "interrupted" and not request["recording_id"]
            and job and job.get("state") == "queued"):
        # A stale queued status file can remain if process creation failed.
        return request
    if request["state"] == "interrupted" and not request["recording_id"] and not job:
        return request
    if request["recording_id"] and (not job or job.get("state") in ("done", "error", "cancelled")):
        store.get_current(request["recording_id"])
        try:
            recordings.materialize_recording(store, request["recording_id"])
        except (storage.StorageError, OSError, ValueError):
            # Commit остаётся действительным; состояние очереди показывается отдельно.
            pass
        job = {"state": "done", "recording_id": request["recording_id"]}
    if not job:
        # Другой процесс ещё может запускать worker; неизвестный исход не повторяем.
        return request
    elif job.get("state") in ("queued", "running"):
        state, step = "running", "running"
    elif job.get("recording_id") and job.get("state") in ("done", "error"):
        # Файл статуса сам по себе не является авторитетной связью с результатом.
        state, step = (("processed", "completed") if request["recording_id"] == job["recording_id"]
                       else ("interrupted", "interrupted"))
    else:
        state, step = "failed", "failed"
    with store._transaction():
        store.current_identity()
        current = store.db.execute("SELECT state FROM workflow_requests WHERE request_id=?", (request_id,)).fetchone()
        if current["state"] not in ("starting", "running", "interrupted"):
            return get(store, request_id)
        now = storage._now()
        recording_id = job.get("recording_id") if state == "processed" else None
        store.db.execute("UPDATE workflow_requests SET state=?, recording_id=COALESCE(?, recording_id), "
                         "updated_at=? WHERE request_id=?",
                         (state, recording_id, now, request_id))
        store.db.execute("UPDATE workflow_runs SET state=?, updated_at=? WHERE request_id=?",
                         (step, now, request_id))
        store.db.execute("UPDATE workflow_steps SET state=?, result_json=?, updated_at=? WHERE run_id IN "
                         "(SELECT run_id FROM workflow_runs WHERE request_id=?)",
                         (step, storage._json({"recording_id": recording_id}) if recording_id else None,
                          now, request_id))
    return get(store, request_id)


def _speaker_labels(transcript):
    """Условные метки говорящих по порядку первого появления значения `speaker`.

    Учитываются только реплики со словами (другие в задание не попадают). Реплика без
    говорящего метки «Участник N» не получает: разные неизвестные — не один человек.
    """
    labels = {}
    for segment in transcript.get("segments", []):
        speaker = segment.get("speaker")
        if segment.get("words") and speaker and speaker not in labels:
            labels[speaker] = model_tasks.speaker_label(len(labels) + 1)
    return labels


def _task_turns(transcript):
    """Реплики задания версии 2: сегмент со словами — одна реплика с условной меткой."""
    labels = _speaker_labels(transcript)
    return [{"turn": number, "speaker": labels.get(segment.get("speaker"), model_tasks.UNKNOWN_SPEAKER)}
            for number, segment in enumerate(
                (segment for segment in transcript.get("segments", []) if segment.get("words")), 1)]


def _open_places(review):
    """Открытые места разбора: статус open (или не задан) и human — их ещё не закрыл человек."""
    return [item for item in review.get("review", []) if item.get("status", "open") in ("open", "human")]


# Допуск на округление времени: то же правило, по которому transcript._find_span находит
# слова места без ссылок при применении ответа человека.
PLACE_TIME_TOLERANCE = 0.01


def _seconds(value):
    return type(value) in (int, float) and math.isfinite(value)


def _place_time(item):
    """(t1, t2) места, если оба — конечные числа и t1 <= t2; иначе None (время негодно)."""
    t1, t2 = item.get("t1"), item.get("t2")
    return (t1, t2) if _seconds(t1) and _seconds(t2) and t1 <= t2 else None


def _word_in_place(start, end, t1, t2):
    """Правило пересечения слова [start, end] с местом [t1, t2] (секунды).

    Слово в месте, если его начало лежит в [t1 − допуск; t2 + допуск] (у мест распознавания
    t1 — начало первого слова, t2 — начало последнего) или если слово перекрывает место
    отрезком положительной длины (start < t2 и end > t1). Слово, которое лишь касается
    начала места своим концом (end == t1), в место не входит.
    """
    return (t1 - PLACE_TIME_TOLERANCE <= start <= t2 + PLACE_TIME_TOLERANCE
            or (start < t2 and end > t1))


def _precise_doubts(transcript, review):
    """Пометки версии 3: (ID сомнительных слов или None — «все слова», число непривязанных мест).

    Место со ссылками на слова помечает эти слова. Место без ссылок с годным временем —
    слова по правилу _word_in_place; если таких слов нет (пауза), — последнее слово до t1
    и первое после t2 (что есть). Место без ссылок и без годного времени, а также место
    со временем в записи, где у слов нет времени, а также место без ссылок, чьё время задевает
    промежуток со словом без времени, привязать нельзя: тогда, как в версии 2,
    сомнительны все слова записи, а число таких мест идёт в задание.
    """
    timed = []                                        # (начало, конец, порядок, ID) слов со временем
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            start, end = word.get("s"), word.get("e")
            if _seconds(start):
                timed.append((start, end if _seconds(end) and end >= start else start, len(timed),
                              word["word_id"]))
    # Промежутки вокруг слов без времени: от конца предыдущего слова со временем до начала следующего.
    # Где внутри такое слово, неизвестно, поэтому место без ссылок, задевающее промежуток, не привязывается.
    untimed, last_end, pending = [], -math.inf, False
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            start, end = word.get("s"), word.get("e")
            if _seconds(start):
                if pending:
                    untimed.append((last_end, start))
                    pending = False
                last_end = end if _seconds(end) and end >= start else start
            else:
                pending = True
    if pending:
        untimed.append((last_end, math.inf))
    marked, unlocated = set(), 0
    for item in _open_places(review):
        if item.get("word_ids"):
            marked.update(item["word_ids"])
            continue
        span = _place_time(item)
        if span is None or not timed:
            unlocated += 1
            continue
        t1, t2 = span
        if any(t1 - PLACE_TIME_TOLERANCE <= hi and t2 + PLACE_TIME_TOLERANCE >= lo for lo, hi in untimed):
            unlocated += 1
            continue
        inside = {ident for start, end, _order, ident in timed if _word_in_place(start, end, t1, t2)}
        if not inside:                                # пауза: никакое слово не попало во время места
            before = [word for word in timed if word[0] < t1]
            after = [word for word in timed if word[0] > t2]
            inside = ({max(before)[3]} if before else set()) | ({min(after)[3]} if after else set())
        marked |= inside
    return (None if unlocated else marked), unlocated


def _task_words(transcript, review, *, task_version=model_tasks.CURRENT_VERSION):
    """Слова задания из снимков: ID, текст, признак сомнения, номер реплики.

    Признак сомнения — по правилам версии task_version: в версиях 1 и 2 одно открытое место
    без ссылок на слова делает сомнительными все слова записи; в версии 3 — см. _precise_doubts.
    """
    if task_version >= 3:
        uncertain, _unlocated = _precise_doubts(transcript, review)
    else:
        open_items = _open_places(review)
        uncertain = (None if any(not item.get("word_ids") for item in open_items)
                     else {ident for item in open_items for ident in item.get("word_ids", [])})
    words, turn = [], 0
    for segment in transcript.get("segments", []):
        if segment.get("words"):
            turn += 1
        for word in segment.get("words", []):
            words.append({"word_id": word["word_id"], "text": word.get("w", word.get("text")),
                          "uncertain": uncertain is None or word["word_id"] in uncertain, "turn": turn})
    return words


def _task_doubts(transcript, review, task_version):
    """Что модель видит о сомнениях в задании версии task_version: пометки слов и (версия 3) число."""
    words = [(word["word_id"], word["uncertain"])
             for word in _task_words(transcript, review, task_version=task_version)]
    return words, (_precise_doubts(transcript, review)[1] if task_version >= 3 else None)


def _build_task(request, snapshots, *, task_version=None):
    """Задание из текущих снимков и цели запроса по правилам версии task_version (1–4).

    Без task_version — текущей версии (model_tasks.CURRENT_VERSION на момент вызова).
    """
    task_version = model_tasks.CURRENT_VERSION if task_version is None else task_version
    current, review = snapshots["transcript"], snapshots["review"]
    words = _task_words(current["payload"], review["payload"], task_version=task_version)
    if task_version == 1:
        return model_tasks.build(request["recording_id"], current["revision"],
                                 [{key: value for key, value in word.items() if key != "turn"} for word in words],
                                 request["policy"]["goal"], review["revision"])
    unlocated = _precise_doubts(current["payload"], review["payload"])[1] if task_version >= 3 else None
    return model_tasks.build(request["recording_id"], current["revision"], words, request["policy"]["goal"],
                             review["revision"], turns=_task_turns(current["payload"]), unlocated_doubts=unlocated,
                             points=task_version >= 4)


_REVISION_FIELDS = ("source_revision", "review_revision")


def content_fingerprint(task):
    """Отпечаток содержания задания: хэш всего, что видит модель, кроме номеров ревизий.

    Входят вид и версия задания, запись, цель, слова (ID, текст, признак сомнения, номер реплики,
    порядок), реплики с условными метками говорящих и (версия 3) число непривязанных мест. Считается из самого задания, поэтому
    подходит и к заданиям, сохранённым раньше. Неизвестные поля задания отклоняются.
    """
    checked = model_tasks.canonical(task)
    return storage._hash({key: value for key, value in checked.items() if key not in _REVISION_FIELDS})


def content_matches(request, task, snapshots):
    """Совпадает ли то, что модель видит в task, с заданием из текущих снимков и цели запроса.

    Номера ревизий не сравниваются: показ карточки, сборка пакета и сохранённый неприменённый
    ответ меняют только номер ревизии разбора. Ничего не пишет.
    """
    try:
        fresh = _build_task(request, snapshots, task_version=model_tasks.version(task))
        return content_fingerprint(fresh) == content_fingerprint(task)
    except ValueError:
        return False


def verification_is_current(store, request_id, recording_id, snapshots):
    """Действует ли отметка «проверено человеком» для текущих снимков. Только чтение.

    Отметка привязана к ревизии стенограммы: любая правка текста её снимает, как раньше.
    Новая ревизия разбора отметку не снимает, если сомнения (что модель увидела бы
    сомнительным) те же, — так и при показе карточки, сборке пакета, неприменённом ответе.

    Обе стороны сравнения считаются по правилам одной версии: версии сохранённого задания
    запроса (то, что модель видела или увидит при доигрывании), а без задания — текущей.
    Поэтому отметка у свёртки, сделанной до этапа 7.3, ведёт себя как прежде.
    """
    transcript = snapshots["transcript"]
    row = store.db.execute("SELECT model_task_json FROM workflow_requests WHERE request_id=?",
                           (request_id,)).fetchone()
    task_version = model_tasks.CURRENT_VERSION
    if row is not None and row["model_task_json"] is not None:
        try:
            task_version = model_tasks.version(storage._loads(row["model_task_json"]))
        except ValueError:
            pass
    expected = _task_doubts(transcript["payload"], snapshots["review"]["payload"], task_version)
    rows = store.db.execute("SELECT DISTINCT review_revision FROM workflow_verifications WHERE request_id=? "
                            "AND recording_id=? AND source_revision=? ORDER BY review_revision DESC",
                            (request_id, recording_id, transcript["revision"])).fetchall()
    for row in rows:
        if row["review_revision"] == snapshots["review"]["revision"]:
            return True
        try:
            old = store.get_revision(recording_id, "review", row["review_revision"])
        except (KeyError, ValueError, storage.StorageError):
            continue
        if old["source_revision"] == transcript["revision"] and _task_doubts(
                transcript["payload"], old["payload"], task_version) == expected:
            return True
    return False


class _HeadMoved(storage.Conflict):
    """Ревизия записи сменилась между чтением снимков и записью; повтор перечитает их."""


def _retry_head_moved(action):
    for _ in range(3):
        try:
            return action()
        except _HeadMoved:
            continue
    raise storage.Conflict("стенограмма или разбор меняются слишком часто; повторите позже")


RESTART_HINT = ("Начните модельный шаг заново: workflow_execute_model(restart=true) или "
                "workflow_model_task(restart=true), в командной строке — rt.py workflow-execute --restart "
                "или rt.py workflow-task --restart. Расшифровка не повторяется; через workflow_execute_model "
                "текст стенограммы снова уйдёт в облако.")


def prepare_model_task(store, request_id, *, restart=False):
    """Задание модели. У готовой свёртки возвращает сохранённое задание и ничего не меняет.

    restart=True — явная команда «начать заново»: новый модельный шаг того же запроса из ошибки,
    готовой свёртки или ожидания; прежние блоки не используются, расшифровка не повторяется.
    """
    request = get(store, request_id)
    if request["state"] == "completed" and request["model_task"] is not None and not restart:
        return request["model_task"]
    allowed = ("processed", "waiting_model", "completed")
    if restart and request["model_task"] is not None:
        allowed += ("failed",)
    if request["state"] not in allowed:
        if request["state"] == "failed" and request["model_task"] is not None:
            raise storage.Conflict("модельный шаг завершился с ошибкой. " + RESTART_HINT)
        raise storage.Conflict("стенограмма ещё не готова")
    return _retry_head_moved(lambda: _prepare_model_task(store, request, restart))


def _prepare_model_task(store, request, restart):
    request_id = request["request_id"]
    snapshots = store.get_current(request["recording_id"])
    current = snapshots["transcript"]
    saved = request["model_task"]
    if (request["state"] == "waiting_model" and not restart and isinstance(saved, dict)
            and content_matches(request, saved, snapshots)):
        # То же содержание — сохранённое задание не переписывается, даже если сменились номера
        # ревизий (показ карточки, пакет, неприменённый ответ): принятые блоки (ключ — хэш задания)
        # не теряют связи с ним. Так же, по своим правилам пометок и договору, доигрываются задания
        # версий 1–3. Новое — всегда текущей версии (4).
        if request["model_run_id"] is not None:
            return saved
        task = saved
    elif request["state"] == "waiting_model" and not restart and isinstance(saved, dict):
        # Содержание сменилось, пока задание ждало: переписать его можно только явным «начать заново».
        raise storage.Conflict("содержание стенограммы или сомнений изменилось после выдачи задания. " + RESTART_HINT)
    else:
        task = _build_task(request, snapshots)
    with store._transaction():
        store.current_identity()
        head = recordings.recording_head(store, request["recording_id"])
        if head is None:
            raise storage.Conflict("запись не найдена")
        if (head["source_revision"] != current["revision"]
                or head["review_revision"] != snapshots["review"]["revision"]):
            raise _HeadMoved("ревизия стенограммы изменилась")
        new_generation = restart or request["state"] in ("processed", "completed")
        cursor = store.db.execute("UPDATE workflow_requests SET state='waiting_model', model_task_json=?, "
                                  "model_result_json=NULL, model_attempts=CASE WHEN ? THEN 0 ELSE model_attempts END, "
                                  "updated_at=? WHERE request_id=? AND state=?",
                                  (storage._json(task), int(new_generation), storage._now(),
                                   request_id, request["state"]))
        if cursor.rowcount != 1:
            raise storage.Conflict("состояние запроса изменилось")
        run = store.db.execute("SELECT run_id FROM workflow_runs WHERE request_id=? AND kind='model'",
                               (request_id,)).fetchone()
        if run is None or new_generation:
            run_id, now = str(uuid.uuid4()), storage._now()
            # Начать заново из ожидания: прежний ожидающий запуск и его шаг закрываются здесь же,
            # иначе ожидающих стало бы два и ни блок, ни итог нового запуска не были бы приняты.
            store.db.execute("UPDATE workflow_steps SET state='superseded', updated_at=? WHERE run_id IN "
                             "(SELECT run_id FROM workflow_runs WHERE request_id=? AND kind='model' "
                             "AND state='waiting_model') AND state='waiting_model'", (now, request_id))
            store.db.execute("UPDATE workflow_runs SET state='superseded', updated_at=? "
                             "WHERE request_id=? AND kind='model' AND state='waiting_model'", (now, request_id))
            store.db.execute("INSERT INTO workflow_runs VALUES (?, ?, 'model', 'waiting_model', "
                             "'opencode/openai/gpt-6-luna', ?, ?)",
                             (run_id, request_id, now, now))
            store.db.execute("INSERT INTO workflow_steps VALUES (?, ?, 'model', 'waiting_model', NULL, ?, ?)",
                             (str(uuid.uuid4()), run_id, now, now))
            store.db.execute("UPDATE workflow_requests SET model_run_id=? WHERE request_id=?",
                             (run_id, request_id))
    return task


def accept_model_result(store, request_id, result):
    """Не более одного исправленного ответа после ошибочного."""
    before = get(store, request_id)
    try:
        return _accept_model_result(store, request_id, result)
    except ValueError:
        _record_model_error(store, request_id, before)
        raise


def _record_model_error(store, request_id, before):
    if before["state"] != "waiting_model":
        return
    with store._transaction():
        store.current_identity()
        row = store.db.execute(
            "SELECT state, model_task_json, model_attempts FROM workflow_requests WHERE request_id=?",
            (request_id,)).fetchone()
        if (row and row["state"] == "waiting_model" and row["model_task_json"] is not None
                and storage._loads(row["model_task_json"]) == before["model_task"]):
            attempts = row["model_attempts"] + 1
            state = "failed" if attempts >= 2 else "waiting_model"
            now = storage._now()
            store.db.execute("UPDATE workflow_requests SET model_attempts=?, state=?, updated_at=? "
                             "WHERE request_id=?", (attempts, state, now, request_id))
            if state == "failed":
                store.db.execute("UPDATE workflow_runs SET state='failed', updated_at=? "
                                 "WHERE request_id=? AND kind='model' AND state='waiting_model'", (now, request_id))
                store.db.execute("UPDATE workflow_steps SET state='failed', updated_at=? WHERE run_id IN "
                                 "(SELECT run_id FROM workflow_runs WHERE request_id=? AND kind='model' "
                                 "AND state='failed') AND state='waiting_model'",
                                 (now, request_id))


MODEL_PART_WORDS = 500
_SENTENCE_END = (".", "!", "?", "…")
_CLOSING = "»\"”’')]"


def _part_bounds(task):
    """Границы блоков задания: список (начало, конец) по индексам слов. Единственное место деления.

    Версия 1 — ровно по MODEL_PART_WORDS слов, как раньше. Версии 2 и 3 — блок не длиннее предела
    и кончается на границе реплики; реплика длиннее предела делится по последнему в пределах
    блока концу предложения, а если такого нет — по пределу. Блоки идут подряд без пропусков.
    Версия 4 — один блок на всю запись: модель отвечает границами разделов, а не перечнем слов.
    """
    words, limit = task["words"], MODEL_PART_WORDS
    if model_tasks.version(task) == 4:
        return [(0, len(words))] if words else []
    if model_tasks.version(task) == 1:
        return [(start, min(start + limit, len(words))) for start in range(0, len(words), limit)]
    bounds, start, end, index = [], 0, 0, 0
    while index < len(words):
        stop = index
        while stop < len(words) and words[stop]["turn"] == words[index]["turn"]:
            stop += 1
        if stop - start > limit:              # реплика [index, stop) не помещается в текущий блок
            if end > start:
                bounds.append((start, end))
                start = end
            while stop - start > limit:       # сама реплика длиннее предела
                cut = start + limit
                for candidate in range(start + limit, start, -1):
                    if words[candidate - 1]["text"].rstrip().rstrip(_CLOSING).endswith(_SENTENCE_END):
                        cut = candidate
                        break
                bounds.append((start, cut))
                start = cut
        end = index = stop
    if end > start:
        bounds.append((start, end))
    return bounds


def _part_task(task, bounds):
    """Задание одного блока той же версии, что и целое. У версии 4 блок один — само задание."""
    if model_tasks.version(task) == 4:
        return task
    words = task["words"][bounds[0]:bounds[1]]
    turns = None
    if model_tasks.version(task) != 1:
        used = {word["turn"] for word in words}
        turns = [turn for turn in task["turns"] if turn["turn"] in used]
    return model_tasks.build(task["recording_id"], task["source_revision"], words,
                             task["goal"], task["review_revision"], turns=turns,
                             unlocated_doubts=task.get("unlocated_doubts"))


def section_time_spans(sections, transcript):
    """Отрезок времени каждого раздела свёртки по временам слов его `evidence`. Ничего не пишет.

    sections — разделы сохранённой свёртки: [{"text": ..., "evidence": [word_id, ...]}, ...]
    (поле `evidence` объекта конспекта). transcript — снимок стенограммы (payload с `segments`).
    Возвращает по одному {"t1": начало, "t2": конец} на раздел, в том же порядке, в секундах
    от начала записи; None, если ни у одного слова раздела в снимке нет времени. Слова,
    которых в снимке нет (стенограмма изменилась), пропускаются.
    """
    times = {}
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            start = word.get("s")
            if "word_id" in word and type(start) in (int, float):
                end = word.get("e")
                times[word["word_id"]] = (start, end if type(end) in (int, float) and end >= start else start)
    spans = []
    for section in sections:
        found = [times[ident] for ident in section.get("evidence", []) if ident in times]
        spans.append({"t1": min(item[0] for item in found) if found else None,
                      "t2": max(item[1] for item in found) if found else None})
    return spans


def _also_line(section, transcript):
    """Строка «См. также» с временем начала каждой отсылки раздела (версия 4); иначе пусто.

    Добавляется локально, в облако не передаётся; отсылка без времени слов не показывается.
    """
    spans = section_time_spans([{"evidence": ids} for ids in section.get("also", [])], transcript)
    times = [f"[{_clock(span['t1'])}]" for span in spans if span["t1"] is not None]
    return "\n\nСм. также: " + ", ".join(times) if times else ""


def _clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _speaker_anchors(transcript):
    """Привязка условных меток к словам: [{"label", "word_ids"}] — первое слово каждой реплики метки.

    По этим словам `conspect.render` при каждой сборке файла находит текущего говорящего,
    поэтому расшифровка меток переживает переименование. Имён здесь нет.
    """
    labels, anchors = _speaker_labels(transcript), {}
    for segment in transcript.get("segments", []):
        if segment.get("words"):
            label = labels.get(segment.get("speaker"), model_tasks.UNKNOWN_SPEAKER)
            anchors.setdefault(label, []).append(segment["words"][0]["word_id"])
    return [{"label": label, "word_ids": ids} for label, ids in anchors.items()]


def _workflow_conspect_id(request):
    return str(uuid.uuid5(uuid.UUID(request["recording_id"]), "conspect:workflow-" + request["request_id"]))


def _conspect_files(store, ident):
    """Записанные файлы свёртки и их подтверждённые суммы; общая основа двух сверок ниже."""
    return store.db.execute("SELECT f.relative_path, f.confirmed_hash, f.confirmed_revision, r.path "
                            "FROM conspect_file_locations f "
                            "JOIN allowed_roots r USING(root_id) WHERE f.conspect_id=?", (ident,)).fetchall()


def conspect_file_problem(store, request):
    """Почему файлы свёртки запроса нельзя перезаписать, или None. Только чтение.

    Та же сверка, что после записи: файл на месте отличается от подтверждённой суммы или по нему
    открыт конфликт. Пропавший файл программа создаст заново — это не помеха. Проверка до вызова
    облака не исключает правку в редакторе между ней и записью (известное ограничение).
    """
    import documents
    if not request["recording_id"]:
        return None
    ident = _workflow_conspect_id(request)
    opened = store.db.execute("SELECT relative_path FROM conspect_conflicts WHERE conspect_id=? "
                              "AND resolution_operation_id IS NULL ORDER BY rowid", (ident,)).fetchone()
    if opened is not None:
        return (f"по файлу свёртки {opened['relative_path']} открыт конфликт с правкой вне программы. "
                "Решите его на экране программы (rt.py screen), затем повторите.")
    for item in _conspect_files(store, ident):
        if item["confirmed_hash"] is None:
            continue
        try:
            path, exists = documents._read_only_target(Path(item["path"]), item["relative_path"])
            if exists and documents._file_digest(path)[0] != item["confirmed_hash"]:
                return (f"файл свёртки {item['relative_path']} изменён вне программы; новая свёртка "
                        "перезаписала бы правку. Верните прежний текст или сохраните свою версию под другим "
                        "именем, затем повторите.")
        except (OSError, ValueError):
            return (f"файл свёртки {item['relative_path']} сейчас нельзя прочитать: он занят другой "
                    "программой или недоступен. Освободите его и повторите.")
    return None


def _ensure_conspect_materialized(store, request):
    import documents
    ident = _workflow_conspect_id(request)
    row = store.db.execute("SELECT current_revision FROM conspects WHERE conspect_id=?", (ident,)).fetchone()
    if row is None:
        raise storage.StorageError("сохранённый конспект отсутствует")
    completed = recordings.materialize_conspects(store, request["recording_id"], conspect_id=ident)
    if (ident, row["current_revision"]) in completed:
        return
    state = store.db.execute("SELECT state FROM conspect_outbox WHERE conspect_id=? AND revision=?",
                             (ident, row["current_revision"])).fetchone()
    files = _conspect_files(store, ident)
    if state and state["state"] == "synced" and len(files) == 2:
        try:
            for item in files:
                path, exists = documents._read_only_target(Path(item["path"]), item["relative_path"])
                if (not exists or item["confirmed_revision"] != row["current_revision"]
                        or documents._file_digest(path)[0] != item["confirmed_hash"]):
                    break
            else:
                return
        except (OSError, ValueError):
            pass
    raise storage.StorageError("конспект сохранён в БД; формирование файлов не завершено")


def model_part(store, request_id, part_index):
    """Один ограниченный блок зафиксированного задания; не отбрасывает остальные слова."""
    if type(part_index) is not int or part_index < 0:
        raise ValueError("неверный номер блока")
    request = get(store, request_id)
    if request["state"] not in ("processed", "waiting_model"):
        if request["state"] == "failed" and request["model_task"] is not None:
            raise storage.Conflict("модельный шаг завершился с ошибкой. " + RESTART_HINT)
        raise storage.Conflict("нет ожидающего модельного задания")
    task = prepare_model_task(store, request_id)
    bounds = _part_bounds(task)
    if part_index >= len(bounds):
        raise ValueError("блок отсутствует")
    return {"task": _part_task(task, bounds[part_index]),
            "part_index": part_index, "part_count": len(bounds)}


RUN_REPLACED = "модельный шаг начат заново другим вызовом; ответ прежнего запуска не сохранён"


def accept_model_part(store, request_id, part_index, result, *, expected_run_id=None):
    """Сохранить блок; итог формируется только после покрытия всех блоков.

    expected_run_id — запуск, которому выдано задание (execute_model): если его уже заменили
    (начать заново), блок не сохраняется. Без него — текущий запуск (ИИ-агент номера запуска не знает)."""
    before = get(store, request_id)
    return _retry_head_moved(lambda: _accept_model_part(store, request_id, part_index, result, before,
                                                        expected_run_id))


def _accept_model_part(store, request_id, part_index, result, before, expected_run_id=None):
    if expected_run_id is not None and before["model_run_id"] != expected_run_id:
        raise storage.Conflict(RUN_REPLACED)
    try:
        task, total, accepted, snapshots = _validate_model_part(store, request_id, part_index, result, before)
    except ValueError:
        _record_model_error(store, request_id, before)
        raise
    if model_tasks.version(task) == 4:
        # Блок версии 4 хранится ответом модели как есть (границы разделов): итог из блоков
        # проверяется заново тем же договором, а разложенный в word_id вид его не прошёл бы.
        accepted = result
    if before["state"] == "completed":
        row = store.db.execute(
            "SELECT result_json FROM workflow_model_parts WHERE request_id=? AND task_hash=? AND part_index=?",
            (request_id, storage._hash({"task": task, "run_id": before["model_run_id"]}), part_index)).fetchone()
        if row is None or storage._loads(row["result_json"]) != accepted:
            raise storage.Conflict("повтор блока содержит другой ответ")
        _ensure_conspect_materialized(store, before)
        return {"part_index": part_index, "received": total, "part_count": total, "state": "completed"}
    with store._transaction():
        store.current_identity()
        row = store.db.execute("SELECT state, model_task_json, model_run_id FROM workflow_requests WHERE request_id=?",
                               (request_id,)).fetchone()
        head = recordings.recording_head(store, before["recording_id"])
        run = store.db.execute("SELECT run_id FROM workflow_runs WHERE request_id=? AND kind='model' "
                               "AND state='waiting_model'", (request_id,)).fetchall()
        if expected_run_id is not None and row["model_run_id"] != expected_run_id:
            raise storage.Conflict(RUN_REPLACED)
        if (row["state"] != "waiting_model" or storage._loads(row["model_task_json"]) != task
                or len(run) != 1 or row["model_run_id"] != run[0]["run_id"]):
            raise storage.Conflict("задание или ревизия изменились")
        if (head["source_revision"] != snapshots["transcript"]["revision"]
                or head["review_revision"] != snapshots["review"]["revision"]):
            raise _HeadMoved("ревизия изменилась после проверки содержания")
        task_hash = storage._hash({"task": task, "run_id": run[0]["run_id"]})
        old = store.db.execute("SELECT result_json FROM workflow_model_parts WHERE request_id=? "
                               "AND task_hash=? AND part_index=?", (request_id, task_hash, part_index)).fetchone()
        if old and storage._loads(old["result_json"]) != accepted:
            raise storage.Conflict("повтор блока содержит другой ответ")
        if not old:
            store.db.execute("INSERT INTO workflow_model_parts VALUES (?, ?, ?, ?, ?)",
                             (request_id, task_hash, part_index, storage._json(accepted), storage._now()))
        rows = store.db.execute("SELECT part_index, result_json FROM workflow_model_parts "
                                "WHERE request_id=? AND task_hash=? ORDER BY part_index",
                                (request_id, task_hash)).fetchall()
    if len(rows) != total:
        return {"part_index": part_index, "received": len(rows), "part_count": total, "state": "waiting_model"}
    accepted = _assemble_parts(store, request_id, task, task_hash, expected_run_id=expected_run_id)
    return {"part_index": part_index, "received": total, "part_count": total,
            "state": "completed", "result": accepted}


def saved_part_count(store, *, request_id, task_hash):
    """Сколько блоков этого задания и шага сохранено."""
    return store.db.execute("SELECT COUNT(*) FROM workflow_model_parts WHERE request_id=? AND task_hash=?",
                            (request_id, task_hash)).fetchone()[0]


def _assemble_parts(store, request_id, task, task_hash, *, expected_run_id=None):
    """Итог из всех сохранённых блоков; облако не вызывается. expected_run_id — только для своего запуска."""
    if expected_run_id is not None and get(store, request_id)["model_run_id"] != expected_run_id:
        raise storage.Conflict(RUN_REPLACED)
    rows = store.db.execute("SELECT result_json FROM workflow_model_parts WHERE request_id=? AND task_hash=? "
                            "ORDER BY part_index", (request_id, task_hash)).fetchall()
    sections = [section for row in rows for section in storage._loads(row["result_json"])["sections"]]
    return accept_model_result(store, request_id, {"source_revision": task["source_revision"],
                                                    "sections": sections})


def _validate_model_part(store, request_id, part_index, result, before=None):
    """Чистая проверка ответа, пригодная для repair без изменения счётчиков состояния."""
    before = before or get(store, request_id)
    if before["state"] not in ("waiting_model", "completed") or before["model_task"] is None:
        raise storage.Conflict("нет ожидающего модельного задания")
    task = before["model_task"]
    bounds = _part_bounds(task)
    total = len(bounds)
    if type(part_index) is not int or not 0 <= part_index < total:
        raise ValueError("неверный номер блока")
    current = None
    if before["state"] != "completed":
        # Сверяется содержание, а не номера ревизий: см. content_matches.
        current = store.get_current(before["recording_id"])
        if not content_matches(before, task, current):
            raise storage.Conflict("содержание стенограммы или сомнений изменилось")
    subtask = _part_task(task, bounds[part_index])
    # Ответ сверяется с заданием, которое видела модель (в том числе её source_revision); слова
    # задания совпадают со словами текущей стенограммы — это и проверено отпечатком содержания.
    accepted = model_tasks.validate(subtask, result, current_revision=task["source_revision"])
    if before["state"] != "completed":
        import conspect
        data = {**current["transcript"]["payload"], "recording_id": before["recording_id"],
                "source_revision": current["transcript"]["revision"]}
        open_review = any(item.get("status", "open") in ("open", "human")
                          for item in current["review"]["payload"].get("review", []))
        for section in accepted["sections"]:
            if "{{q:" in section["text"]:
                raise ValueError("служебные маркеры цитат в ответе не допускаются")
            obj, report = conspect.link(section["text"], data)
            if report["unlinked"] or (obj["quotes"] and open_review):
                raise ValueError("цитата неоднозначна или есть неразрешённые сомнения")
            if any(not set(part["word_ids"]) <= set(section["evidence"])
                   for quote in obj["quotes"].values() for part in quote["parts"]):
                raise ValueError("цитата не входит в свидетельства раздела")
    return task, total, accepted, current


def _fail_model_run(store, request_id, code, message, *, invalid_count=0, expected_run_id=None):
    """Ошибка модельного шага. expected_run_id — запуск вызова: если его уже заменили, ничего не меняется."""
    with store._transaction():
        store.current_identity()
        row = store.db.execute("SELECT state, model_run_id FROM workflow_requests WHERE request_id=?",
                               (request_id,)).fetchone()
        if row is None or row["state"] != "waiting_model":
            return
        if expected_run_id is not None and row["model_run_id"] != expected_run_id:
            return
        now = storage._now()
        store.db.execute("UPDATE workflow_requests SET state='failed', model_attempts=model_attempts+?, "
                         "updated_at=? WHERE request_id=?", (invalid_count, now, request_id))
        store.db.execute("UPDATE workflow_runs SET state='failed', updated_at=? "
                         "WHERE run_id=? AND kind='model' AND state='waiting_model'", (now, row["model_run_id"]))
        store.db.execute("UPDATE workflow_steps SET state='failed', result_json=?, updated_at=? "
                         "WHERE run_id=? AND state='waiting_model'",
                         (storage._json({"error_code": code, "message": message}), now, row["model_run_id"]))


def _resume_model_run(store, request):
    if request["state"] != "failed" or not request["model_task"]:
        raise storage.Conflict("нет завершившегося с ошибкой модельного задания для продолжения")
    if request["model_task"]["goal"] != request["policy"]["goal"]:
        raise storage.Conflict("цель изменилась; продолжение старого модельного задания запрещено. " + RESTART_HINT)
    current = store.get_current(request["recording_id"])
    if not content_matches(request, request["model_task"], current):
        raise storage.Conflict("содержание стенограммы или разбора изменилось (слова, говорящие или сомнения); "
                               "продолжить прежний модельный шаг нельзя. " + RESTART_HINT)
    with store._transaction():
        store.current_identity()
        now = storage._now()
        store.db.execute("UPDATE workflow_requests SET state='waiting_model', model_attempts=0, updated_at=? "
                         "WHERE request_id=? AND state='failed'", (now, request["request_id"]))
        store.db.execute("UPDATE workflow_runs SET state='waiting_model', updated_at=? "
                         "WHERE request_id=? AND kind='model' AND state='failed'",
                         (now, request["request_id"]))
        store.db.execute("UPDATE workflow_steps SET state='waiting_model', result_json=NULL, updated_at=? "
                         "WHERE run_id=? AND state='failed'", (now, request["model_run_id"]))


def _completed_result(store, request):
    """Ответ «готово» — только когда итог сохранён в базе и файлы свёртки записаны."""
    import llm_provider
    try:
        _ensure_conspect_materialized(store, request)
    except (OSError, ValueError, storage.StorageError) as ex:
        raise llm_provider.ProviderError(
            "files_not_written",
            "свёртка сохранена в базе, но её файлы не записаны: файл изменён вне программы, занят или "
            "недоступен. Решите конфликт на экране программы (rt.py screen) или освободите файл и повторите "
            "workflow_execute_model — в облако повторно ничего не уйдёт.") from ex
    total = len(_part_bounds(request["model_task"]))
    return {"request_id": request["request_id"], "state": "completed",
            "model": f"{llm_provider.PROVIDER_ID}/{llm_provider.MODEL_ID}",
            "part_count": total, "completed_parts": total}


def _limit_problem(request, words):
    import llm_provider
    if words == 0:
        return "в стенограмме нет слов: свёртку делать не из чего. В облако ничего не отправлено."
    if words <= llm_provider.MAX_WORDS:
        return None
    exits = ("свёртку без облака может сделать ИИ-агент (workflow_model_task — задание по страницам, "
             "затем workflow_model_result); или снимите заказ конспекта (workflow_change_policy с "
             "require_conspect=false), стенограмма останется.")
    if request["state"] == "failed":
        exits = "сначала начните шаг заново для ИИ-агента: workflow_model_task(restart=true); затем " + exits
    return (f"задание длиннее предела облачного исполнителя ({llm_provider.MAX_WORDS} слов): {words} слов. "
            "В облако ничего не отправлено, состояние запроса не изменилось. Выходы: " + exits)


def execute_model(store, request_id, *, retry_failed=False, restart=False):
    """Автономно выполнить все ещё не сохранённые блоки одним ограниченным OpenCode provider.

    Повтор у готовой свёртки облако не вызывает: дописывает файлы или честно говорит, почему не может.
    Все блоки сохранены — повтор собирает итог без облака. restart — явное «начать заново».
    Предел размера и файл свёртки проверяются до любого изменения состояния и до вызова облака.
    """
    import llm_provider
    if retry_failed and restart:
        raise ValueError("retry_failed и restart вместе не допускаются: продолжить или начать заново — что-то одно")
    request = get(store, request_id)
    if not request["policy"]["require_conspect"]:
        raise storage.Conflict("для запроса не заказан конспект")
    if not request["policy"]["allow_cloud_model"]:
        raise storage.Conflict("облачный модельный исполнитель запрещён политикой запроса")
    state = request["state"]
    if state == "completed" and request["model_task"] is not None and not restart:
        return _completed_result(store, request)
    if restart or state == "processed":
        if not request["recording_id"] or (state == "failed" and request["model_task"] is None):
            raise storage.Conflict("модельное задание ещё не готово к исполнению")
        words = sum(len(segment.get("words", [])) for segment in
                    store.get_current(request["recording_id"])["transcript"]["payload"].get("segments", []))
    elif state == "waiting_model" or (state == "failed" and retry_failed):
        words = len(request["model_task"]["words"])
    elif state == "failed" and request["model_task"] is not None:
        raise storage.Conflict("модельный шаг завершился с ошибкой: продолжите его (retry_failed=true) или "
                               "начните заново. " + RESTART_HINT)
    else:
        raise storage.Conflict("модельное задание ещё не готово к исполнению")
    problem = _limit_problem(request, words)
    if problem:
        raise llm_provider.ProviderError("limit", problem)
    problem = conspect_file_problem(store, request)
    if problem:
        raise llm_provider.ProviderError("conspect_file_changed", problem + " В облако ничего не отправлено.")
    if restart or state == "processed":
        task = prepare_model_task(store, request_id, restart=restart)
        request = get(store, request_id)
    elif state == "failed":
        _resume_model_run(store, request)
        request = get(store, request_id)
        task = request["model_task"]
    else:
        task = request["model_task"]
    total = len(_part_bounds(task))
    run_id = request["model_run_id"]                 # свой запуск: ответы и ошибки — только в него
    task_hash = storage._hash({"task": task, "run_id": run_id})
    completed = {row["part_index"] for row in store.db.execute(
        "SELECT part_index FROM workflow_model_parts WHERE request_id=? AND task_hash=?",
        (request_id, task_hash))}
    invalid_count = 0
    cleanup = []                                     # предупреждения клиента: неубранные сессии OpenCode

    def progress():
        saved = saved_part_count(store, request_id=request_id, task_hash=task_hash)
        return (f"сохранено блоков {saved} из {total}; продолжить с места остановки — "
                "workflow_execute_model(retry_failed=true) или rt.py workflow-execute --retry-failed")
    try:
        if len(completed) < total:                  # все блоки сохранены — клиент облака не нужен
            with llm_provider.OpenCodeTextClient() as client:
                cleanup = getattr(client, "cleanup_warnings", cleanup)
                for part_index in range(total):
                    if part_index in completed:
                        continue
                    part = model_part(store, request_id, part_index)
                    if get(store, request_id)["model_task"] != task:
                        raise storage.Conflict("задание переписано: содержание стенограммы изменилось")
                    repair_error = invalid_output = None
                    for attempt in range(2):
                        result, raw = client.generate(part["task"], repair_error=repair_error,
                                                      invalid_output=invalid_output)
                        try:
                            _validate_model_part(store, request_id, part_index, result)
                        except ValueError as ex:
                            invalid_count += 1
                            if attempt == 1:
                                raise llm_provider.ProviderError(
                                    "invalid_response", "ответ модели не прошёл проверку после одной попытки исправления")
                            repair_error, invalid_output = str(ex), raw
                            continue
                        accept_model_part(store, request_id, part_index, result, expected_run_id=run_id)
                        break
        if (get(store, request_id)["state"] == "waiting_model"
                and saved_part_count(store, request_id=request_id, task_hash=task_hash) == total):
            _assemble_parts(store, request_id, task, task_hash,      # блоки сохранены раньше, итога не было
                            expected_run_id=run_id)
    except llm_provider.ProviderError as ex:
        if get(store, request_id)["model_run_id"] != run_id:
            raise llm_provider.ProviderError(ex.code, f"{ex}; {RUN_REPLACED}") from ex
        message = f"{ex}; {progress()}"
        _fail_model_run(store, request_id, ex.code, message, invalid_count=invalid_count, expected_run_id=run_id)
        raise llm_provider.ProviderError(ex.code, message) from ex
    except (OSError, ValueError, KeyError, storage.StorageError) as ex:
        after = get(store, request_id)
        if after["model_run_id"] != run_id:
            raise llm_provider.ProviderError("revision_changed", RUN_REPLACED) from ex
        if after["state"] == "completed":
            return _completed_result(store, after)
        if isinstance(ex, storage.Conflict):
            code = "revision_changed"
            message = ("содержание стенограммы или разбора изменилось, пока модель отвечала (слово, говорящий, "
                       "применённый ответ на сомнение); сохранённые блоки устарели. " + RESTART_HINT)
        else:
            code, message = "internal_error", f"модельный шаг остановлен; {progress()}"
        _fail_model_run(store, request_id, code, message, invalid_count=invalid_count, expected_run_id=run_id)
        raise llm_provider.ProviderError(code, message) from ex
    finally:
        _record_cleanup_warnings(store, run_id, cleanup)
    after = get(store, request_id)
    if after["state"] == "completed":
        return _completed_result(store, after)
    message = f"модельный шаг не завершён: {progress()}"
    if after["model_run_id"] != run_id:
        raise llm_provider.ProviderError("revision_changed", RUN_REPLACED)
    _fail_model_run(store, request_id, "internal_error", message, invalid_count=invalid_count,
                    expected_run_id=run_id)
    raise llm_provider.ProviderError("internal_error", message)


def _record_cleanup_warnings(store, run_id, warnings):
    """Предупреждения исполнителя (например, OpenCode не удалил временную сессию) — в шаги этого запуска.

    Текст предупреждения задаёт llm_provider: без пути и идентификатора сессии. Ошибка записи не меняет исход шага.
    """
    if not warnings:
        return
    try:
        with store._transaction():
            for row in store.db.execute("SELECT step_id, result_json FROM workflow_steps WHERE run_id=?",
                                        (run_id,)).fetchall():
                result = json.loads(row["result_json"]) if row["result_json"] else {}
                if not isinstance(result, dict):
                    result = {"result": result}
                result["warnings"] = list(dict.fromkeys([*result.get("warnings", []), *warnings]))
                store.db.execute("UPDATE workflow_steps SET result_json=? WHERE step_id=?",
                                 (storage._json(result), row["step_id"]))
    except Exception as ex:        # запись предупреждения не должна менять исход шага
        import sys
        print(f"workflow: предупреждение исполнителя не записано: {type(ex).__name__}", file=sys.stderr)


def _accept_model_result(store, request_id, result):
    return _retry_head_moved(lambda: _accept_model_result_once(store, request_id, result))


def _accept_model_result_once(store, request_id, result):
    """Принять итог. Связка ревизий при совпадающем содержании, но сменившихся номерах:

    model_task и model_result хранят ревизии, которые видела модель (ответ сверяется с ними);
    свёртка сохраняется с текущей ревизией стенограммы (conspect_revisions.source_revision и
    source_revision в её JSON), цитаты связываются с текущими словами.
    """
    import conspect
    request = get(store, request_id)
    if request["state"] == "completed" and request["model_result"] is not None:
        accepted = model_tasks.validate(request["model_task"], result,
                                        current_revision=request["model_task"]["source_revision"])
        if accepted != request["model_result"]:
            raise storage.Conflict("повтор содержит другой ответ")
        _ensure_conspect_materialized(store, request)
        return accepted
    if request["state"] != "waiting_model" or request["model_task"] is None:
        raise storage.Conflict("нет ожидающего модельного задания")
    snapshots = store.get_current(request["recording_id"])
    current = snapshots["transcript"]
    if not content_matches(request, request["model_task"], snapshots):
        raise storage.Conflict("содержание стенограммы или сомнений изменилось; подготовьте новое задание")
    accepted = model_tasks.validate(request["model_task"], result,
                                    current_revision=request["model_task"]["source_revision"])
    text = "\n\n".join(section["text"] for section in accepted["sections"])
    if "{{q:" in text:
        raise ValueError("служебные маркеры цитат в ответе не допускаются")
    labelled = model_tasks.version(request["model_task"]) != 1
    if labelled:
        # Локальные добавления к ответу модели, в облако не передаются. Время начала раздела
        # (по словам его evidence) стоит отдельной строкой перед текстом раздела и попадает
        # в шаблон. Расшифровку условных меток строит conspect.render при каждой сборке файла.
        if _task_turns(current["payload"]) != request["model_task"]["turns"]:
            raise storage.Conflict("реплики стенограммы не совпадают с заданием; подготовьте новое задание")
        spans = section_time_spans(accepted["sections"], current["payload"])
        text = "\n\n".join(
            (f"[{_clock(span['t1'])}]\n" if span["t1"] is not None else "") + section["text"]
            + _also_line(section, current["payload"])
            for section, span in zip(accepted["sections"], spans))
    data = {**current["payload"], "recording_id": request["recording_id"],
            "source_revision": current["revision"]}
    for section in accepted["sections"]:
        section_obj, section_report = conspect.link(section["text"], data)
        if section_report["unlinked"]:
            raise ValueError("цитаты неоднозначны или не найдены в стенограмме")
        evidence = set(section["evidence"])
        if any(not set(part["word_ids"]) <= evidence
               for quote in section_obj["quotes"].values() for part in quote["parts"]):
            raise ValueError("цитата не входит в свидетельства раздела")
    obj, report = conspect.link(text, data)
    if report["unlinked"]:
        raise ValueError("цитаты неоднозначны или не найдены в стенограмме")
    if obj["quotes"] and any(item.get("status", "open") in ("open", "human")
                             for item in snapshots["review"]["payload"].get("review", [])):
        raise ValueError("цитаты недоступны, пока остаются неразрешённые сомнения")
    name = "workflow-" + request_id
    binding = recordings.document_binding(store, request["recording_id"])
    json_path, md_path = conspect.path_for(binding["root"] / binding["paths"]["json"], name)
    obj.update(name=name, provenance="model", evidence=accepted["sections"])
    if labelled:
        obj["speaker_labels"] = _speaker_anchors(current["payload"])
    md, _ = conspect.render(obj, data)
    payload = json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")
    # С метками говорящих шапку (заголовок, строку об авторстве, расшифровку меток) собирает
    # conspect.render — она же восстановит её при пересборке. Без меток (задание версии 1) — как раньше.
    if not labelled:
        md = "# Конспект\n\nСоздан моделью; проверка человеком не подтверждена.\n\n" + md
    md_bytes = (md + "\n").encode("utf-8")
    conspect_id = str(uuid.uuid5(uuid.UUID(request["recording_id"]), "conspect:" + name))
    previous = store.db.execute("SELECT current_revision FROM conspects WHERE conspect_id=?",
                                (conspect_id,)).fetchone()
    expected = previous["current_revision"] if previous else None
    operation_id = str(uuid.uuid5(uuid.UUID(request_id), "model-conspect:" + str((expected or 0) + 1)))
    with store._transaction():
        store.current_identity()
        head = recordings.recording_head(store, request["recording_id"])
        row = store.db.execute("SELECT state, model_task_json FROM workflow_requests WHERE request_id=?",
                               (request_id,)).fetchone()
        if row["state"] != "waiting_model" or storage._loads(row["model_task_json"]) != request["model_task"]:
            raise storage.Conflict("задание или ревизия изменились")
        if (head["source_revision"] != current["revision"]
                or head["review_revision"] != snapshots["review"]["revision"]):
            raise _HeadMoved("ревизия изменилась после проверки содержания")
        saved = store.save_conspect(
            request["recording_id"], conspect_id, name, current["revision"], expected, obj, payload, md_bytes,
            json_path.relative_to(binding["root"]).as_posix(), md_path.relative_to(binding["root"]).as_posix(),
            binding["root_id"], operation_id, request["context"])
        store.db.execute("UPDATE workflow_requests SET state='completed', model_attempts=model_attempts+1, "
                         "model_result_json=?, updated_at=? "
                         "WHERE request_id=?", (storage._json(accepted), storage._now(), request_id))
        now = storage._now()
        store.db.execute("UPDATE workflow_runs SET state='completed', updated_at=? WHERE request_id=? "
                         "AND kind='model' AND state='waiting_model'",
                         (now, request_id))
        store.db.execute("UPDATE workflow_steps SET state='completed', updated_at=? WHERE run_id IN "
                         "(SELECT run_id FROM workflow_runs WHERE request_id=? AND kind='model' "
                         "AND state='completed') AND state='waiting_model'",
                         (now, request_id))
    completed = recordings.materialize_conspects(store, request["recording_id"], conspect_id=conspect_id)
    if (conspect_id, saved["conspect_revision"]) not in completed:
        raise storage.StorageError("конспект сохранён в БД; формирование файлов не завершено")
    return accepted
