"""Сохранённые запросы П3. Исполнение вызывается сервисом, не при импорте."""

import copy
import itertools
import json
import math
import re
import uuid
from fractions import Fraction
from pathlib import Path

import model_tasks
import product_policy
import recordings
import storage
import transcript


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


def _place_unresolved(item):
    """Место разбора ещё не закрыто человеком: статус open (или не задан) или human.

    Единственная запись этого правила для цитат и признаков сомнения в workflow.py. Копии его остались в
    core.py и transcript.py (до PR 7 линии 1): там их не трогаем.
    """
    return item.get("status", "open") in ("open", "human")


def _open_places(review):
    """Открытые места разбора: их ещё не закрыл человек (см. _place_unresolved)."""
    return [item for item in review.get("review", []) if _place_unresolved(item)]


# Правило слов места живёт в transcript.py (РАР-6.1, PR 6а): здесь только имена для прежних вызовов.
PLACE_TIME_TOLERANCE = transcript.WORD_PLACE_TOLERANCE
_seconds = transcript._finite_number


def _place_time(item):
    """(t1, t2) места, если оба — конечные числа и t1 <= t2; иначе None. Только t1/t2, как до PR 6а
    (у transcript._place_time есть ещё запасное «t»: это правило линии 2 6а не меняет)."""
    t1, t2 = item.get("t1"), item.get("t2")
    return (t1, t2) if _seconds(t1) and _seconds(t2) and t1 <= t2 else None


_word_in_place = transcript.word_in_place


def _closed_unheard(item):
    """Место закрыто без прослушивания (ИИ по поручению): цитату оно не держит, в шапке видно «не прослушано».

    Поле появится с PR 14 (его имя задаёт линия 1): до него функция всегда возвращает False.
    """
    return False


def place_holds_quote(item):
    """Может ли место разбора держать цитату в кавычках. Единственная запись этого правила (РАР-7, 7.5).

    Не держит открытое место и место, закрытое без прослушивания. Отметки слов (`word["seen"]` — человек
    видел слово, это не подтверждение) правило не читает.
    """
    return not _place_unresolved(item) and not _closed_unheard(item)


def quote_blockers(data, word_ids, overlap_ids=None):
    """Места разбора, из-за которых цитата из слов word_ids не держится в кавычках.

    data — стенограмма с ключом review (как у transcript.load): список мест. -> [{"id", "reason"}], reason:
    open — место не решено; unheard — закрыто без прослушивания; unlocated — место или слова цитаты
    привязать к словам нельзя, и оно задевает все цитаты; overlap_skipped — слово цитаты лежит в отрезке
    наложения, где место overlap не поставлено, потому что наложений в записи больше
    words.OVERLAP_UNMEASURED_MAX_SHARE (quality.overlap_unmeasured_skipped.spans, Н28-1; id — «overlap_skipped:N»,
    N — номер отрезка). Слово, которое тронул человек (transcript.human_touched), отрезок не задевает. Пусто —
    цитата чистая.

    Место со ссылками на слова (и не пропуск) — эти слова. Остальные места — слова по transcript.place_words
    (PR 6а) с pause_neighbors=True: у места-паузы (слов внутри нет) это последнее слово до неё и первое после,
    так что пауза между словами цитаты или вплотную к её краю цитату задевает, а пауза в стороне — нет.

    overlap_ids — слова, по которым проверяются отрезки наложений; по умолчанию word_ids. Сборка конспекта передаёт
    в word_ids объединение сохранённых и напечатанных ID (место могло сослаться на корневой ID, которого в
    стенограмме уже нет, С255-2), а в overlap_ids — только напечатанные слова (О255-2).
    """
    wanted = set(word_ids)
    quote_untimed = any(
        word.get("word_id") in wanted and not _seconds(word.get("s"))
        for segment in data.get("segments") or [] for word in segment.get("words") or [])
    blockers = []
    for item in data.get("review") or []:
        if place_holds_quote(item):
            continue
        reason = "open" if _place_unresolved(item) else "unheard"
        if item.get("word_ids") and not transcript.is_gap_place(item):
            hit = bool(wanted & set(item["word_ids"]))
        else:
            # Время места — только t1/t2, как до PR 6а (place_words читает и запасное «t»: цитат это не касается).
            found = (transcript.place_words(data, item, pause_neighbors=True)
                     if _place_time(item) is not None else None)
            if found is None or quote_untimed:
                reason, hit = "unlocated", True   # место или время слова цитаты неизвестны — не судим
            else:
                hit = bool(wanted & set(found[0]))
        if hit:
            blockers.append({"id": item.get("id"), "reason": reason})
    return blockers + overlap_blockers(data, word_ids if overlap_ids is None else overlap_ids)


def overlap_blockers(data, word_ids):
    """Часть quote_blockers по отрезкам пропущенных наложений (Н28-1): [{"id", "reason"}], reason —
    overlap_skipped (id «overlap_skipped:N») или unlocated (у слова цитаты нет времени, id «overlap_skipped»).
    Её же читает сборка конспекта агента (conspect.render без quote_policy, Р255-1)."""
    spans = transcript.overlap_skipped_spans(data)
    if not spans:
        return []
    wanted = set(word_ids)
    hits = {transcript.overlap_skipped_word(data, word, spans)
            for segment in data.get("segments") or [] for word in segment.get("words") or []
            if word.get("word_id") in wanted} - {None}
    out = [{"id": "overlap_skipped", "reason": "unlocated"}] if "overlap_skipped" in hits else []
    return out + [{"id": h, "reason": "overlap_skipped"}
                  for h in sorted(hits - {"overlap_skipped"}, key=lambda h: int(h.rsplit(":", 1)[1]))]


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


def _word_times(transcript):
    """Время слов в порядке _task_words: (начало, конец) в секундах или None. В задание не попадает."""
    result = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            start, end = word.get("s"), word.get("e")
            timed = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (start, end))
            result.append((float(start), float(end)) if timed else None)
    return result


def _pause_after(points, times, number):
    """Пауза в секундах между пунктом number (с 1) и следующим; None — у слов нет времени."""
    before, after = times[points[number - 1]["end"] - 1], times[points[number]["start"]]
    return None if before is None or after is None else after[0] - before[1]


def _try_parts(points, times, total, count, limit):
    """Деление на count частей не длиннее limit слов или None, если так разделить нельзя."""
    size = total / count
    cuts, previous = [], 0
    for k in range(1, count):
        target = k * size
        options = [number for number in range(1, len(points))
                   if previous < points[number - 1]["end"] <= previous + limit
                   and total - points[number - 1]["end"] <= (count - k) * limit]
        if not options:
            return None

        def distance(number):
            return abs(points[number - 1]["end"] - target), points[number - 1]["end"]
        near = [number for number in options if abs(points[number - 1]["end"] - target) <= size / 10]
        timed = [(_pause_after(points, times, number), number) for number in near]
        timed = [(pause, number) for pause, number in timed if pause is not None]
        if timed:
            longest = max(pause for pause, _number in timed)
            cut = min((number for pause, number in timed if pause == longest), key=distance)
        else:
            cut = min(options, key=distance)
        cuts.append(cut)
        previous = points[cut - 1]["end"]
    return [[first, final] for first, final in zip([1] + [cut + 1 for cut in cuts], cuts + [len(points)])]


def _plan_parts(points, times):
    """Части записи длиннее предела одного запроса: [[первый пункт, последний пункт], ...] или None.

    W — число слов, N = ceil(W / MAX_WORDS). Граница k ищется у цели k·W/N среди границ пунктов:
    самая длинная пауза между пунктами в окне ±10 % размера части (W/N), но так, чтобы ни одна
    часть не превысила MAX_WORDS слов; при равных паузах — ближе к цели. Если в окне нет пауз
    (у слов нет времени), берётся граница пункта, ближайшая к цели, при том же условии. Пункт —
    реплика или её кусок не длиннее 120 слов, поэтому делится и монолог. Если при N частях
    условие не выполнить, пробуется N + 1. Больше MAX_PARTS частей — ValueError. Детерминировано.
    """
    import llm_provider
    limit, most = llm_provider.MAX_WORDS, llm_provider.MAX_PARTS
    total = points[-1]["end"] if points else 0
    if total <= limit:
        return None
    for count in range(math.ceil(total / limit), most + 1):
        parts = _try_parts(points, times, total, count, limit)
        if parts is not None:
            return parts
    raise ValueError(f"запись в {total} слов не делится на {most} частей не длиннее {limit} слов")


def _task_doubts(transcript, review, task_version):
    """Что модель видит о сомнениях в задании версии task_version: пометки слов и (версия 3) число."""
    words = [(word["word_id"], word["uncertain"])
             for word in _task_words(transcript, review, task_version=task_version)]
    return words, (_precise_doubts(transcript, review)[1] if task_version >= 3 else None)


_PLAN = object()


def _build_task(request, snapshots, *, task_version=None, parts=_PLAN):
    """Задание из текущих снимков и цели запроса по правилам версии task_version (1–4).

    Без task_version — текущей версии (model_tasks.CURRENT_VERSION на момент вызова). У версии 4
    без parts деление на части вычисляется (_plan_parts); запись до предела одного запроса и
    запись, которую не разделить, получают задание без частей. parts=None — без частей, список —
    эти части (сверка содержания с сохранённым заданием).
    """
    task_version = model_tasks.CURRENT_VERSION if task_version is None else task_version
    current, review = snapshots["transcript"], snapshots["review"]
    words = _task_words(current["payload"], review["payload"], task_version=task_version)
    if task_version == 1:
        return model_tasks.build(request["recording_id"], current["revision"],
                                 [{key: value for key, value in word.items() if key != "turn"} for word in words],
                                 request["policy"]["goal"], review["revision"])
    unlocated = _precise_doubts(current["payload"], review["payload"])[1] if task_version >= 3 else None

    def build(parts=None):
        return model_tasks.build(request["recording_id"], current["revision"], words, request["policy"]["goal"],
                                 review["revision"], turns=_task_turns(current["payload"]),
                                 unlocated_doubts=unlocated, points=task_version >= 4, parts=parts)
    task = build()
    if task_version >= 4:
        if parts is _PLAN:
            try:
                parts = _plan_parts(task["points"], _word_times(current["payload"]))
            except ValueError:
                parts = None            # длиннее предела облака: одно задание для ИИ-агента, облако откажет
        if parts is not None:
            task = build(parts)
    return task


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
        # Деление на части — то же, что у сохранённого задания: границы зависят от пауз, а не от текста.
        fresh = _build_task(request, snapshots, task_version=model_tasks.version(task), parts=task.get("parts"))
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
    Версия 4 — один блок на всю запись: модель отвечает границами разделов, а не перечнем слов;
    у задания с частями (запись длиннее предела одного запроса) — блок на каждую часть.
    """
    words, limit = task["words"], MODEL_PART_WORDS
    if model_tasks.version(task) == 4:
        if "parts" in task:
            points = task["points"]
            return [(points[first - 1]["start"], points[final - 1]["end"]) for first, final in task["parts"]]
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
    """Задание одного блока той же версии, что и целое. У версии 4 без частей блок один — само задание,
    с частями — задание части (model_tasks.part_task): модель видит только её пункты."""
    if model_tasks.version(task) == 4:
        if "parts" not in task:
            return task
        return model_tasks.part_task(task, _part_bounds(task).index(tuple(bounds)) + 1)
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


def _quotes_one_by_one(task):
    """Новые модельные свёртки (задание с метками говорящих, версии 2 и выше) собираются «цитаты по одной»
    (РАР-7, 7.5): заданные до этого версии 1 — по-прежнему с запретом на всю запись."""
    return model_tasks.version(task) != 1


# Исправление ненайденных цитат (РАР-7, 7.5в; решение владельца 24.11.2026 к В7 плана 7.5: «Да, как предложено»).
REPAIR_MIN_UNLINKED = 3
REPAIR_MIN_SHARE = Fraction(1, 5)       # 20 % всех цитат свёртки; дробь, чтобы граница не зависела от округления


def quotes_need_repair(unlinked, total):
    """Достаточно ли ненайденных цитат для исправления: не меньше 3 и не меньше 20 % всех цитат свёртки.

    unlinked — цитаты «не найдено в стенограмме» и «в разделе несколько таких мест»; total — все цитаты свёртки,
    в том числе связанные (задевшие место разбора цитаты в кавычки не берутся, но связаны и входят в total).
    Единственная запись порога: экран и MCP спрашивают эту функцию.
    """
    for number in (unlinked, total):
        if type(number) is not int or number < 0:
            raise ValueError("число цитат — целое, не меньше нуля")
    if total < unlinked:
        raise ValueError("ненайденных цитат не может быть больше, чем всех")
    return unlinked >= REPAIR_MIN_UNLINKED and unlinked >= REPAIR_MIN_SHARE * total


def quote_counts(payload):
    """(ненайденных цитат, всех цитат) свёртки «цитаты по одной»; None — у остальных свёрток.

    Свёртка с перечнем `unlinked` считает по нему; собранная до 7.5в (перечня нет) — по сохранённому числу.
    """
    if payload.get("quote_policy") != 1:
        return None
    listed = payload.get("unlinked")
    if isinstance(listed, list):
        unlinked = len(listed)
    else:
        saved = payload.get("unlinked_count")
        unlinked = saved if type(saved) is int and saved >= 0 else 0
    quotes = payload.get("quotes")
    return unlinked, (len(quotes) if isinstance(quotes, dict) else 0) + unlinked


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
        open_review = any(_place_unresolved(item) for item in current["review"]["payload"].get("review", []))
        for section in accepted["sections"]:
            if any(mark in section["text"] for mark in conspect.SLOT_MARKS):
                raise ValueError("служебные маркеры цитат в ответе не допускаются")
            if _quotes_one_by_one(task):
                continue      # цитаты по одной: несвязанная или задевшая сомнение цитата ответ не отклоняет
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
    result = {"request_id": request["request_id"], "state": "completed",
              "model": f"{llm_provider.PROVIDER_ID}/{llm_provider.MODEL_ID}",
              "part_count": total, "completed_parts": total}
    adjusted = (request.get("model_result") or {}).get("also_adjusted")
    if adjusted:
        # Отсылки «См. также», урезанные или отброшенные при приёме ответа (Ч-0а): ответ из-за них не отвергнут.
        result["also_adjusted"] = adjusted
    return result


def _limit_problem(request, words, *, limit=None, unsplit=False):
    """Отказ по размеру до вызова облака или None. limit — предел задания (по умолчанию — записи
    частями, MAX_WORDS × MAX_PARTS; у задания без частей, сохранённого раньше, — MAX_WORDS).
    unsplit — запись в пределе, но не делится на части не длиннее MAX_WORDS слов."""
    import llm_provider
    if words == 0:
        return "в стенограмме нет слов: свёртку делать не из чего. В облако ничего не отправлено."
    limit = llm_provider.MAX_WORDS * llm_provider.MAX_PARTS if limit is None else limit
    if words <= llm_provider.MAX_WORDS or (words <= limit and not unsplit):
        return None
    exits = ("свёртку без облака может сделать ИИ-агент (workflow_model_task — задание по страницам, "
             "затем workflow_model_result); или снимите заказ конспекта (workflow_change_policy с "
             "require_conspect=false), стенограмма останется.")
    if request["state"] == "failed":
        exits = "сначала начните шаг заново для ИИ-агента: workflow_model_task(restart=true); затем " + exits
    reason = (f"задание длиннее предела облачного исполнителя ({limit} слов): {words} слов"
              if words > limit else
              f"запись в {words} слов не делится на {llm_provider.MAX_PARTS} частей не длиннее "
              f"{llm_provider.MAX_WORDS} слов по границам пунктов")
    return reason + ". В облако ничего не отправлено, состояние запроса не изменилось. Выходы: " + exits


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
        snapshots = store.get_current(request["recording_id"])
        transcript = snapshots["transcript"]["payload"]
        words = sum(len(segment.get("words", [])) for segment in transcript.get("segments", []))
        limit = None
        unsplit = False
        if llm_provider.MAX_WORDS < words <= llm_provider.MAX_WORDS * llm_provider.MAX_PARTS:
            points = model_tasks._points(_task_words(transcript, snapshots["review"]["payload"]))
            try:
                _plan_parts(points, _word_times(transcript))
            except ValueError:
                unsplit = True
    elif state == "waiting_model" or (state == "failed" and retry_failed):
        task = request["model_task"]
        words = len(task["words"])
        unsplit = False
        # Задание без частей (версии 1–4, сохранённое раньше) — прежний предел одного задания.
        limit = None if model_tasks.version(task) == 4 and "parts" in task else llm_provider.MAX_WORDS
        if limit is None and any(end - start > llm_provider.MAX_WORDS for start, end in _part_bounds(task)):
            limit = llm_provider.MAX_WORDS
    elif state == "failed" and request["model_task"] is not None:
        raise storage.Conflict("модельный шаг завершился с ошибкой: продолжите его (retry_failed=true) или "
                               "начните заново. " + RESTART_HINT)
    else:
        raise storage.Conflict("модельное задание ещё не готово к исполнению")
    problem = _limit_problem(request, words, limit=limit, unsplit=unsplit)
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
                    extra = {}
                    if part_index and "part" in part["task"]:
                        # Изложение — тексты разделов уже принятых частей (проверенный ответ модели).
                        extra["summary"] = [
                            [section["text"] for section in storage._loads(row["result_json"])["sections"]]
                            for row in store.db.execute(
                                "SELECT result_json FROM workflow_model_parts WHERE request_id=? AND task_hash=? "
                                "AND part_index<? ORDER BY part_index", (request_id, task_hash, part_index))]
                    for attempt in range(2):
                        result, raw = client.generate(part["task"], repair_error=repair_error,
                                                      invalid_output=invalid_output, **extra)
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
    if any(mark in text for mark in conspect.SLOT_MARKS):
        raise ValueError("служебные маркеры цитат в ответе не допускаются")
    labelled = _quotes_one_by_one(request["model_task"])     # одно правило: версия задания, не равная 1
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
    if _quotes_one_by_one(request["model_task"]):
        # Цитаты по одной: ответ не отклоняется из-за цитаты. Каждая ищется среди слов своего раздела;
        # не найденная или неоднозначная печатается без кавычек с пометкой, а задевшая место разбора
        # получает пометку при каждой сборке (conspect.render). Разбор нужен сборке — он идёт в data.
        data["review"] = snapshots["review"]["payload"].get("review", [])
        pieces, quotes, unlinked = [], {}, []
        for index, (section, span) in enumerate(zip(accepted["sections"], spans)):
            section_obj, _report = conspect.link(section["text"], data, within=set(section["evidence"]),
                                                 unlinked="slot", labels=True)
            shift = {"q": len(quotes), "u": len(unlinked)}          # номера цитат и слотов — сквозные
            template = conspect.SLOT.sub(
                lambda hit: "{{%s:%d}}" % (hit.group(1), int(hit.group(2)) + shift[hit.group(1)]),
                section_obj["template"])
            quotes.update({str(int(key) + shift["q"]): quote for key, quote in section_obj["quotes"].items()})
            unlinked.extend({**item, "id": str(int(item["id"]) + shift["u"]), "section": index}
                            for item in section_obj["unlinked"])
            pieces.append((f"[{_clock(span['t1'])}]\n" if span["t1"] is not None else "") + template
                          + _also_line(section, current["payload"]))
        obj = {"version": 2, "template": "\n\n".join(pieces), "quotes": quotes,
               "recording_id": data["recording_id"], "source_revision": data["source_revision"],
               "analysis_revision": None, "quote_policy": 1, "unlinked": unlinked,
               "unlinked_count": len(unlinked), "assembly_review_revision": snapshots["review"]["revision"]}
    else:
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
        if obj["quotes"] and any(_place_unresolved(item)
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


# ---------- Исправление ненайденных цитат по просьбе человека (РАР-7, этап 7.5в, шаг 2) ----------
# Один вызов облака на одно нажатие (экран) или одно обращение (MCP) с согласием человека. Сама программа его
# не запускает: ни приём ответа, ни пересборка, ни разбор. Условия — quote_repair_problem (единственная запись),
# порог — quotes_need_repair, задание — quote_repair_task, применение — _apply_quote_repair (по одной цитате).
REPAIR_MAX_QUOTES = 60
REPAIR_WHO = "opencode/openai/gpt-6-luna"
REPAIR_OLD = ("свёртка собрана до этой возможности; исправление недоступно. Новая свёртка — "
              "workflow_execute_model(restart=true): весь текст снова уйдёт в облако.")
REPAIR_OLD_SCREEN = ("свёртка собрана до этой возможности; исправление недоступно. Новую свёртку может собрать "
                     "ИИ-агент заново: весь текст снова уйдёт в облако.")
REPAIR_CHANGED = "что уйдёт в облако, изменилось: посмотрите снова"
REPAIR_RUNNING = "исправление цитат по этому запросу уже идёт; дождитесь ответа"
REPAIR_LIVE_MARGIN_SEC = 120         # к таймауту вызова и запуска сервера OpenCode (llm_provider): живой вызов
_LEADING_LABEL = re.compile(r"^(Участник\s+\d+|Говорящий не определён)\s*[:—–-]\s*")


def _repair_conspect(store, request):
    """Текущая запись свёртки запроса или None."""
    for saved in store.list_conspects(request["recording_id"]):
        if saved["name"] == "workflow-" + request["request_id"]:
            return saved
    return None


def _repair_labels(payload, transcript):
    """{номер реплики со словами: условная метка} по меткам свёртки. Имён здесь нет: говорящий без метки
    (появился после приёма ответа) и без говорящего — UNKNOWN_SPEAKER."""
    speaker_of = {word.get("word_id"): segment.get("speaker")
                  for segment in transcript.get("segments", []) for word in segment.get("words", [])}
    by_speaker = {}
    for item in payload.get("speaker_labels") or []:
        for word_id in item.get("word_ids") or []:
            if word_id in speaker_of:
                by_speaker.setdefault(speaker_of[word_id], item.get("label"))
    labels, turn = {}, 0
    for segment in transcript.get("segments", []):
        if segment.get("words"):
            turn += 1
            label = by_speaker.get(segment["speaker"]) if segment.get("speaker") else None
            ok = isinstance(label, str) and model_tasks.SPEAKER_LABEL.fullmatch(label)
            labels[turn] = label if ok else model_tasks.UNKNOWN_SPEAKER
    return labels


def _repair_task_from(payload, snapshots):
    """(задание или None, сведения) по свёртке и текущим снимкам. Только разделы со слотами; пункты — из текущей
    стенограммы (с исправлениями и сомнениями ⟦…⟧), метки — условные. Имён, времени, ID слов, записи, цели
    и номеров ревизий в задании нет. Не поместившееся и потерявшее слова — в left."""
    import llm_provider
    transcript = snapshots["transcript"]["payload"]
    words = _task_words(transcript, snapshots["review"]["payload"])
    labels = _repair_labels(payload, transcript)
    by_section = {}
    for item in payload.get("unlinked") or []:
        by_section.setdefault(item["section"], []).append(item)
    sections, indexes, sent, used, left, full = [], [], [], 0, 0, False
    for index in sorted(by_section):
        items = sorted(by_section[index], key=lambda item: item["at"][0])
        evidence = set(payload["evidence"][index].get("evidence") or [])
        chosen = [word for word in words if word["word_id"] in evidence]
        if not chosen:
            left += len(items)                      # места этих цитат удалены из стенограммы
            continue
        fits = len(sent) + len(items) <= REPAIR_MAX_QUOTES and used + len(chosen) <= llm_provider.MAX_WORDS
        if full or (not fits and sections):
            full = True
            left += len(items)
            continue
        if not fits:                                # первый раздел сам не помещается: берутся первые цитаты
            left += max(0, len(items) - REPAIR_MAX_QUOTES)
            items = items[:REPAIR_MAX_QUOTES]
        points = [[labels[turn], model_tasks._point_text(list(group))]
                  for turn, group in itertools.groupby(chosen, key=lambda word: word["turn"])]
        sections.append({"text": payload["evidence"][index]["text"], "quotes": [
            {"id": item["id"], "quote": item["quote"], "why": item["why"]} for item in items], "points": points})
        indexes.append(index)
        sent += items
        used += len(chosen)
    meta = {"sections": indexes, "ids": [item["id"] for item in sent], "words": used, "left": left,
            "slots": {item["id"]: {"quote": item["quote"], "section": item["section"], "at": list(item["at"])}
                      for item in sent}}
    if not sections:
        return None, meta
    return model_tasks.quote_repair({"kind": model_tasks.REPAIR_KIND, "version": 1, "sections": sections}), meta


def quote_repair_task(store, request):
    """(задание, сведения) исправления для запроса; Conflict, если отправлять нечего. Только чтение."""
    saved = _repair_conspect(store, request)
    if saved is None:
        raise storage.Conflict(REPAIR_OLD)
    snapshots = store.get_current(request["recording_id"])
    task, meta = _repair_task_from(saved["payload"], snapshots)
    if task is None:
        raise storage.Conflict("места этих цитат удалены из стенограммы: отправлять нечего")
    meta.update(conspect_id=saved["conspect_id"], conspect_revision=saved["current_revision"],
                source_revision=snapshots["transcript"]["revision"], review_revision=snapshots["review"]["revision"])
    return task, meta


def quote_repair_problem(store, request, *, screen=False):
    """Почему исправление цитат недоступно, или None. Только чтение; единственная запись условий.

    Экран, MCP и исполнение спрашивают эту функцию; ни одна проверка здесь облако не вызывает.
    screen=True — для человека на экране: без имён инструментов агента.
    """
    old = REPAIR_OLD_SCREEN if screen else REPAIR_OLD
    if (not request["policy"]["require_conspect"] or request["state"] != "completed"
            or request["model_task"] is None or not request["recording_id"]):
        return "готовой модельной свёртки у этого запроса нет"
    if not _quotes_one_by_one(request["model_task"]):
        return old
    if not request["policy"]["allow_cloud_model"]:
        return ("облако запрещено политикой этого запроса; для облачной обработки нужен новый запрос "
                "с разрешением")
    saved = _repair_conspect(store, request)
    payload = saved["payload"] if saved else {}
    if payload.get("quote_policy") != 1 or not isinstance(payload.get("unlinked"), list):
        return old
    unlinked, total = quote_counts(payload)
    if not quotes_need_repair(unlinked, total):
        return (f"не найдено {unlinked} из {total} цитат: исправление доступно от {REPAIR_MIN_UNLINKED} "
                f"и от {int(REPAIR_MIN_SHARE * 100)} %")
    problem = conspect_file_problem(store, request)
    if problem:
        return problem
    try:
        quote_repair_task(store, request)
    except storage.Conflict as ex:
        return str(ex)
    return None


def _repair_snapshot(request, saved, task, meta):
    unlinked, total = quote_counts(saved["payload"])
    return {"request_id": request["request_id"], "name": saved["name"], "unlinked": unlinked, "total": total,
            "sections": len(meta["sections"]), "quotes": len(meta["ids"]), "words": meta["words"],
            "left": meta["left"], "task_hash": storage._hash(task), "conspect_revision": meta["conspect_revision"],
            "source_revision": meta["source_revision"], "review_revision": meta["review_revision"]}


def quote_repair_prepare(store, request_id):
    """Снимок того, что уйдёт в облако, для окна согласия на экране. Conflict с причиной, если недоступно."""
    request = get(store, request_id)
    problem = quote_repair_problem(store, request, screen=True)
    if problem:
        raise storage.Conflict(problem)
    task, meta = quote_repair_task(store, request)
    return _repair_snapshot(request, _repair_conspect(store, request), task, meta)


def quote_repair_info(store, request_id):
    """Для MCP без согласия: доступно ли исправление, почему нет, сколько уйдёт. Облако не вызывается."""
    request = get(store, request_id)
    problem = quote_repair_problem(store, request)
    info = {"request_id": request_id, "available": problem is None, "reason": problem}
    saved = _repair_conspect(store, request) if request["recording_id"] else None
    counts = quote_counts(saved["payload"]) if saved else None
    if counts:
        info.update(unlinked=counts[0], total=counts[1], quotes=[
            {"id": item["id"], "quote": item["quote"], "why": item["why"], "section": item["section"]}
            for item in saved["payload"].get("unlinked") or []])
    if problem is None:
        task, meta = quote_repair_task(store, request)
        info.update(sections=len(meta["sections"]), words=meta["words"], left=meta["left"])
    return info


def _repair_clean(answer):
    """Ответ модели -> цитата для связывания: без наших служебных знаков ⟦ ⟧, крайних «» и метки говорящего."""
    text = " ".join(answer.replace("⟦", "").replace("⟧", "").split())
    for _ in range(3):
        text = text.strip("«»\" ")
        text = _LEADING_LABEL.sub("", text, count=1)
    return text


def _repair_link(clean, data, within):
    """Связанная цитата (объект свёртки) или None: отрывок должен однозначно найтись среди слов раздела."""
    import conspect
    if (not clean or len(clean) > model_tasks.MAX_REPAIR_QUOTE_CHARS
            or any(mark in clean for mark in ("…", "...", "«", "»", "[", "]", *conspect.SLOT_MARKS))):
        return None
    try:
        obj, report = conspect.link(f"«{clean}»", data, within=within, unlinked="reject", labels=True)
    except ValueError:
        return None
    if report["unlinked"] or list(obj["quotes"]) != ["1"]:
        return None
    return obj["quotes"]["1"]


def _apply_quote_repair(payload, data, ids, answers):
    """Чистая функция: (новая свёртка, счёт). Каждая цитата — отдельно, одна негодная не мешает остальным.

    Строка, найденная в своём разделе, — новая цитата: слот {{u:N}} становится {{q:K}}, в тексте раздела
    «старая» заменена на «новая». null — кавычки сняты: цитата остаётся обычным текстом без пометки.
    Замены идут с конца раздела, позиции оставшихся слотов сдвигаются на разницу длин.
    """
    import conspect
    new = copy.deepcopy(payload)
    by_id = {item["id"]: item for item in new["unlinked"]}
    todo = [by_id[ident] for ident in ids if ident in by_id and ident in answers]
    todo.sort(key=lambda item: (item["section"], item["at"][0]), reverse=True)
    counts = {"fixed": 0, "unquoted": 0, "failed": 0}
    last = max((int(key) for key in new["quotes"]), default=0)
    for item in todo:
        section = new["evidence"][item["section"]]
        text, (start, end), answer = section["text"], item["at"], answers[item["id"]]
        if text[start:end] != "«%s»" % item["quote"]:
            counts["failed"] += 1                    # место цитаты в тексте не то, что записано: не трогаем
            continue
        if answer is None:
            replacement, token = item["quote"], item["quote"]
            counts["unquoted"] += 1
        else:
            clean = _repair_clean(answer)
            quote = _repair_link(clean, data, set(section.get("evidence") or []))
            if quote is None:
                counts["failed"] += 1
                continue
            last += 1
            found = conspect._LABEL_BEFORE.search(text[:start])
            quote["written_label"] = " ".join(found.group(1).split()) if found else None
            new["quotes"][str(last)] = quote
            replacement, token = f"«{clean}»", "{{q:%d}}" % last
            counts["fixed"] += 1
        new["template"] = new["template"].replace("{{u:%s}}" % item["id"], token, 1)
        section["text"] = text[:start] + replacement + text[end:]
        delta = len(replacement) - (end - start)
        for other in new["unlinked"]:
            if other is not item and other["section"] == item["section"] and other["at"][0] >= end:
                other["at"] = [other["at"][0] + delta, other["at"][1] + delta]
        new["unlinked"] = [other for other in new["unlinked"] if other is not item]
    new["unlinked_count"] = len(new["unlinked"])
    return new, counts


def _repair_rows(store, run_id, state, extra):
    """Состояние запуска и шага исправления; в result_json дописываются поля extra. Вызывать в транзакции."""
    row = store.db.execute("SELECT step_id, result_json FROM workflow_steps WHERE run_id=?", (run_id,)).fetchone()
    details = storage._loads(row["result_json"]) if row["result_json"] else {}
    details.update(extra)
    now = storage._now()
    store.db.execute("UPDATE workflow_steps SET state=?, result_json=?, updated_at=? WHERE step_id=?",
                     (state, storage._json(details), now, row["step_id"]))
    store.db.execute("UPDATE workflow_runs SET state=?, updated_at=? WHERE run_id=?", (state, now, run_id))


def _repair_failed(store, run_id, code, message):
    try:
        with store._transaction():
            store.current_identity()
            _repair_rows(store, run_id, "failed", {"error_code": code, "message": message})
    except Exception as ex:     # запись отказа не должна скрыть первую ошибку
        import sys
        print(f"workflow: отказ исправления цитат не записан: {type(ex).__name__}", file=sys.stderr)


def _save_repair(store, request, run_id, meta, answers):
    """Применить ответ к ТЕКУЩЕЙ свёртке и записать новую ревизию вместе с шагом. -> (счёт, ревизия, id, записано)."""
    import conspect
    now = get(store, request["request_id"])
    if now["state"] != "completed" or now["model_task"] != request["model_task"]:
        raise storage.Conflict("модельный шаг начат заново; исправление не записано")
    snapshots = store.get_current(request["recording_id"])
    current = snapshots["transcript"]
    if current["revision"] != meta["source_revision"]:
        raise storage.Conflict("стенограмма изменилась, пока модель отвечала; исправление не записано")
    saved = _repair_conspect(store, request)
    live = {item["id"]: item for item in (saved["payload"].get("unlinked") or [])} if saved else {}
    if any(live.get(ident, {}).get(key) != value
           for ident, slot in meta["slots"].items() for key, value in slot.items()):
        raise storage.Conflict("свёртка изменилась, пока модель отвечала; исправление не записано")
    data = {**current["payload"], "recording_id": request["recording_id"], "source_revision": current["revision"],
            "review": snapshots["review"]["payload"].get("review", [])}
    new, counts = _apply_quote_repair(saved["payload"], data, meta["ids"], answers)
    counts["left"] = meta["left"]
    if counts["fixed"] + counts["unquoted"] == 0:
        with store._transaction():
            store.current_identity()
            _repair_rows(store, run_id, "completed", {**counts, "conspect_revision": saved["current_revision"]})
        return counts, saved["current_revision"], saved["conspect_id"], False
    new["source_revision"] = current["revision"]
    new["assembly_review_revision"] = snapshots["review"]["revision"]
    md, _ = conspect.render(new, data)
    payload = json.dumps(new, ensure_ascii=False, indent=1).encode("utf-8")
    binding = recordings.document_binding(store, request["recording_id"])
    with store._transaction():
        store.current_identity()
        head = recordings.recording_head(store, request["recording_id"])
        if head["source_revision"] != current["revision"]:
            raise storage.Conflict("стенограмма изменилась, пока модель отвечала; исправление не записано")
        result = store.save_conspect(
            request["recording_id"], saved["conspect_id"], saved["name"], current["revision"],
            saved["current_revision"], new, payload, (md + "\n").encode("utf-8"), saved["json_relative"],
            saved["md_relative"], binding["root_id"], str(uuid.uuid5(uuid.UUID(run_id), "quote-repair")),
            recordings.operation_context(store, "model"))
        _repair_rows(store, run_id, "completed", {**counts, "conspect_revision": result["conspect_revision"]})
    return counts, result["conspect_revision"], saved["conspect_id"], True


def repair_quotes(store, request_id, *, via, snapshot=None, allow_cloud_model=False):
    """Один вызов облака: исправить ненайденные цитаты готовой свёртки. via — «screen» или «mcp».

    allow_cloud_model — согласие человека на эту отправку: без настоящего True отказ до чтения базы.
    Отказы до вызова (согласие, условия, устаревший снимок, уже идущий вызов) облако не вызывают и запуска не
    пишут. Отказы после вызова (формат ответа, смена стенограммы или свёртки) пишут шаг failed, свёртку не
    меняют; повтора нет. Один живой вызов на запрос: проверяется в базе, в той же транзакции, что и запись
    запуска (экран и MCP — разные процессы).
    """
    import datetime
    import llm_provider
    if allow_cloud_model is not True:
        raise ValueError("отправка в облако — только по явному согласию человека (allow_cloud_model=True)")
    if via not in ("screen", "mcp") or (via == "screen" and snapshot is None):
        raise ValueError("исправление запускает экран (со снимком) или MCP")
    request = get(store, request_id)
    problem = quote_repair_problem(store, request, screen=(via == "screen"))
    if problem:
        raise storage.Conflict(problem)
    task, meta = quote_repair_task(store, request)
    if snapshot is not None and snapshot != _repair_snapshot(request, _repair_conspect(store, request), task, meta):
        raise storage.Conflict(REPAIR_CHANGED)
    run_id, now = str(uuid.uuid4()), storage._now()
    with store._transaction():
        store.current_identity()
        row = store.db.execute("SELECT current_revision FROM conspects WHERE conspect_id=?",
                               (meta["conspect_id"],)).fetchone()
        if row is None or row["current_revision"] != meta["conspect_revision"]:
            raise storage.Conflict(REPAIR_CHANGED)
        # Живой запуск этого запроса (моложе таймаута вызова с запасом) — отказ до клиента; старше — оборванный.
        alive = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            seconds=llm_provider.REQUEST_TIMEOUT_SEC + llm_provider.STARTUP_TIMEOUT_SEC + REPAIR_LIVE_MARGIN_SEC)
        for running in store.db.execute("SELECT updated_at FROM workflow_runs WHERE request_id=? AND "
                                        "kind='quote_repair' AND state='running'", (request_id,)).fetchall():
            if datetime.datetime.fromisoformat(running["updated_at"]) >= alive:
                raise storage.Conflict(REPAIR_RUNNING)
        # Запуск, оставшийся running от оборванного процесса, больше не ждётся: interrupted.
        store.db.execute("UPDATE workflow_steps SET state='interrupted', updated_at=? WHERE state='running' AND "
                         "run_id IN (SELECT run_id FROM workflow_runs WHERE request_id=? AND kind='quote_repair')",
                         (now, request_id))
        store.db.execute("UPDATE workflow_runs SET state='interrupted', updated_at=? "
                         "WHERE request_id=? AND kind='quote_repair' AND state='running'", (now, request_id))
        store.db.execute("INSERT INTO workflow_runs VALUES (?, ?, 'quote_repair', 'running', ?, ?, ?)",
                         (run_id, request_id, REPAIR_WHO, now, now))
        store.db.execute("INSERT INTO workflow_steps VALUES (?, ?, 'quote_repair', 'running', ?, ?, ?)",
                         (str(uuid.uuid4()), run_id,
                          storage._json({"initiated_via": via, "context": recordings.operation_context(store, "model"),
                                         "meta": meta, "task": task}), now, now))
    cleanup = []
    try:
        with llm_provider.OpenCodeTextClient() as client:
            cleanup = getattr(client, "cleanup_warnings", cleanup)
            result, _raw = client.repair_quotes(task)
        try:
            answers = model_tasks.validate_quote_repair(task, result)
        except ValueError as ex:
            raise llm_provider.ProviderError(
                "invalid_response", f"ответ модели не прошёл проверку формата ({ex}); свёртка не изменена") from ex
        counts, revision, conspect_id, changed = _save_repair(store, request, run_id, meta, answers)
    except llm_provider.ProviderError as ex:
        _repair_failed(store, run_id, ex.code, str(ex))
        raise
    except storage.Conflict as ex:
        _repair_failed(store, run_id, "revision_changed", str(ex))
        raise llm_provider.ProviderError("revision_changed", str(ex)) from ex
    except Exception as ex:
        _repair_failed(store, run_id, "internal_error", f"исправление цитат остановлено: {type(ex).__name__}")
        raise
    finally:
        _record_cleanup_warnings(store, run_id, cleanup)
    if changed:
        completed = recordings.materialize_conspects(store, request["recording_id"], conspect_id=conspect_id)
        if (conspect_id, revision) not in completed:
            raise storage.StorageError("сохранено в базе; файлы свёртки не записаны: решите конфликт на экране")
    return {"request_id": request_id, **counts, "conspect_revision": revision}
