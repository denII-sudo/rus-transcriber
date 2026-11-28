"""Изолированный прототип хранения П1а. Не подключён к рабочему приложению."""

import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 4
TRANSCRIPT_FIELDS = frozenset(("file", "duration", "speakers", "speaker_names",
                               "names_source", "names_confirmed", "segments",
                               "quality", "edits", "vocab_suggestions", "title", "recorded_at", "recorded_timezone",
                               "container", "codec", "audio_sha256", "audio_available", "speaker_naming"))
REVIEW_FIELDS = frozenset(("review", "review_done", "questions", "answers",
                           "conflicts", "rejected", "applications", "undo",
                           "interaction_events", "interaction_events_evicted", "question_version"))
ACTOR_KINDS = frozenset(("human", "model", "system", "unknown"))
KEEP_ANNOTATION = object()
MAX_INTERACTION_EVENTS = 2000
# События карточек (показ, проигрывание, прослушивание места) — отдельная таблица только для добавления (Х1):
# событие не создаёт ревизию разбора. Разбор прежнего формата хранит их списком interaction_events; такой список
# только читается, новые события в него не пишутся. Таблицы нет в базе, созданной до Х1, — её создаёт шаг
# обновления базы или страховка recordings.ensure_review_events (резерв, потом пустая таблица).
REVIEW_EVENTS_DDL = """CREATE TABLE review_events (
            recording_id TEXT NOT NULL REFERENCES recordings(recording_id),
            seq INTEGER NOT NULL CHECK(seq >= 1),
            event_json TEXT NOT NULL,
            review_revision INTEGER NOT NULL CHECK(review_revision >= 1),
            origin TEXT NOT NULL CHECK(origin IN ('page', 'migrated')),
            operation_id TEXT NOT NULL REFERENCES operations(operation_id),
            created_at TEXT NOT NULL,
            PRIMARY KEY(recording_id, seq))"""
REVIEW_EVENTS_COLUMNS = frozenset(("recording_id", "seq", "event_json", "review_revision", "origin",
                                   "operation_id", "created_at"))
MAX_REVIEW_EVENT_BYTES = 16384
# Гипотезы моделей (Г-1, РАР-4.2): запись hyp-2 одного прогона расшифровки — часть записи. Удалить запись, не
# удалив её гипотезы, нельзя: внешний ключ (сторож будущего удаления записи, tests/test_hypotheses_db.py).
# Таблицы нет в базе, созданной до Г-1, — её создаёт recordings.ensure_hypotheses (резерв, потом пустая таблица).
# Машинные данные: операций и ревизий не создают; удаляются только очисткой гипотез после подтверждения человеком.
HYPOTHESES_DDL = """CREATE TABLE hypotheses (
            run_id TEXT PRIMARY KEY,
            recording_id TEXT NOT NULL REFERENCES recordings(recording_id),
            sound_id TEXT NOT NULL CHECK(length(sound_id) = 12),
            job_id TEXT,
            format TEXT NOT NULL,
            record_json TEXT NOT NULL,
            size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
            created_at TEXT NOT NULL)"""
HYPOTHESES_INDEX_DDL = "CREATE INDEX hypotheses_by_recording ON hypotheses(recording_id)"
HYPOTHESES_COLUMNS = frozenset(("run_id", "recording_id", "sound_id", "job_id", "format", "record_json",
                                "size_bytes", "created_at"))
HYPOTHESIS_RUN_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
HYPOTHESIS_SOUND_RE = re.compile(r"^[0-9a-f]{12}$")


class StorageError(Exception):
    pass


class Conflict(StorageError):
    pass


class UnsupportedSchema(StorageError):
    pass


class Busy(Conflict):
    pass


class ReviewEventsMissing(StorageError):
    """В базе нет таблицы событий карточек: база создана до Х1 и ещё не обновлена."""


def _is_busy(exc):
    code = getattr(exc, "sqlite_errorcode", None)
    return ((isinstance(code, int) and (code & 0xff) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED))
            or "locked" in str(exc).lower() or "busy" in str(exc).lower())


def check_local_path(path):
    """Не принимать UNC и диски Windows, не являющиеся локальными fixed-дисками."""
    try:
        path = Path(path).absolute()
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("invalid path") from exc
    if os.name == "nt":
        import ctypes
        anchor = resolved.anchor
        if anchor.startswith("\\\\") or not anchor:
            raise ValueError("network paths are not supported")
        if ctypes.windll.kernel32.GetDriveTypeW(str(anchor)) != 3:
            raise ValueError("only local fixed drives are supported")
    return path


def _uuid(value):
    if not isinstance(value, str):
        raise ValueError("expected UUID string")
    try:
        canonical = str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError("invalid UUID") from exc
    if value != canonical:
        raise ValueError("UUID must be canonical")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _loads(value):
    try:
        return json.loads(value)
    except (TypeError, ValueError) as exc:
        raise StorageError("invalid stored JSON") from exc


def _hash_key(kind):
    """Ключ хэша документа для вида цели реестра файлов."""
    return kind if kind == "archive_manifest" else kind.removeprefix("archive_")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ref(ref):
    if not isinstance(ref, dict) or set(ref) != {"user_id", "user_name_snapshot"}:
        raise ValueError("invalid author reference")
    ident, name = ref["user_id"], ref["user_name_snapshot"]
    if ident is None:
        if name is not None:
            raise ValueError("unknown author must have null name")
    else:
        _uuid(ident)
        if not isinstance(name, str):
            raise ValueError("known author needs a name snapshot")


def _context(ctx):
    if not isinstance(ctx, dict) or set(ctx) != {
            "actor_kind", "author_ref", "initiated_by", "origin_installation_id"}:
        raise ValueError("invalid actor context")
    if ctx["actor_kind"] not in ACTOR_KINDS:
        raise ValueError("invalid actor kind")
    _ref(ctx["author_ref"])
    _ref(ctx["initiated_by"])
    if ctx["actor_kind"] != "human" and ctx["author_ref"]["user_id"] is not None:
        raise ValueError("automatic action cannot have a human author")
    if ctx["origin_installation_id"] is not None:
        _uuid(ctx["origin_installation_id"])


NAMING_SOURCES = frozenset(("form", "voice", "transcript", "model", "agent", "human"))


def validate_speaker_naming(naming):
    """Поле `speaker_naming`: исходная метка -> {name, source, confirmed, revision[, unknown]}."""
    if not isinstance(naming, dict):
        raise ValueError("speaker_naming must be an object")
    for label, item in naming.items():
        if not isinstance(label, str) or not label or not isinstance(item, dict):
            raise ValueError("invalid speaker_naming entry")
        if not set(item) <= {"name", "source", "confirmed", "revision", "unknown"}:
            raise ValueError("unknown speaker_naming fields")
        if not isinstance(item.get("name"), str) or not item["name"]:
            raise ValueError("speaker_naming name must be text")
        if item.get("source") not in NAMING_SOURCES:
            raise ValueError("invalid speaker_naming source")
        if not isinstance(item.get("confirmed"), bool):
            raise ValueError("speaker_naming confirmed must be boolean")
        if "unknown" in item and not isinstance(item["unknown"], bool):
            raise ValueError("speaker_naming unknown must be boolean")
        rev = item.get("revision")
        if rev is not None and (isinstance(rev, bool) or not isinstance(rev, int) or rev < 0):
            raise ValueError("invalid speaker_naming revision")
    return naming


def _snapshot(value, fields):
    if not isinstance(value, dict) or not set(value) <= fields:
        raise ValueError("unknown snapshot fields")
    _json(value)
    if fields is TRANSCRIPT_FIELDS:
        if "speaker_naming" in value:
            validate_speaker_naming(value["speaker_naming"])
        segments = value.get("segments", [])
        if not isinstance(segments, list):
            raise ValueError("segments must be a list")
        segment_ids, word_ids = set(), set()
        for segment in segments:
            if not isinstance(segment, dict) or not isinstance(segment.get("segment_id"), str):
                raise ValueError("segment_id required")
            sid = segment["segment_id"]
            if not sid or sid in segment_ids:
                raise ValueError("duplicate or empty segment_id")
            segment_ids.add(sid)
            if "speaker" in segment and segment["speaker"] is not None and not isinstance(segment["speaker"], str):
                raise ValueError("speaker must be text or null")
            for start_key, end_key in (("start", "end"),):
                _validate_times(segment, start_key, end_key)
            words = segment.get("words", [])
            if not isinstance(words, list):
                raise ValueError("words must be a list")
            for word in words:
                if not isinstance(word, dict) or not isinstance(word.get("word_id"), str):
                    raise ValueError("word_id required")
                wid = word["word_id"]
                if not wid or wid in word_ids:
                    raise ValueError("duplicate or empty word_id")
                word_ids.add(wid)
                if "w" in word and not isinstance(word["w"], str):
                    raise ValueError("word text must be text")
                _validate_times(word, "s", "e")
    else:
        items = value.get("review", [])
        if not isinstance(items, list):
            raise ValueError("review must be a list")
        ids = set()
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                raise ValueError("review id required")
            if item["id"] in ids:
                raise ValueError("duplicate review id")
            ids.add(item["id"])
        events = value.get("interaction_events", [])
        evicted = value.get("interaction_events_evicted", 0)
        if not isinstance(events, list) or len(events) > MAX_INTERACTION_EVENTS:
            raise ValueError("too many interaction events")
        if type(evicted) is not int or evicted < 0:
            raise ValueError("invalid evicted count")


def _validate_times(value, start_key, end_key):
    for key in (start_key, end_key):
        if key in value and (type(value[key]) not in (int, float) or
                             not math.isfinite(value[key]) or value[key] < 0):
            raise ValueError("invalid time")
    if start_key in value and end_key in value and value[start_key] > value[end_key]:
        raise ValueError("time interval reversed")


def _validate_embedded_provenance(value):
    if isinstance(value, dict):
        if "actor_kind" in value:
            kind = value["actor_kind"]
            if kind not in ACTOR_KINDS:
                raise ValueError("invalid actor kind")
            author = value.get("author_ref")
            if author is not None:
                _ref(author)
                if kind != "human" and author["user_id"] is not None:
                    raise ValueError("automatic action cannot have a human author")
        for key, item in value.items():
            if key in ("author_ref", "initiated_by"):
                _ref(item)
            else:
                _validate_embedded_provenance(item)
    elif isinstance(value, list):
        for item in value:
            _validate_embedded_provenance(item)


def validate_pair(transcript, review):
    """Проверить поддерживаемые явные ссылки одного кортежа снимков."""
    _snapshot(transcript, TRANSCRIPT_FIELDS)
    _snapshot(review, REVIEW_FIELDS)
    word_ids = {word["word_id"] for segment in transcript.get("segments", [])
                for word in segment.get("words", [])}
    review_ids = {item["id"] for item in review.get("review", [])}
    answers = review.get("answers", {})
    if not isinstance(answers, dict) or not set(answers) <= review_ids:
        raise ValueError("answers must reference existing review items")
    if any(not isinstance(answer, dict) for answer in answers.values()):
        raise ValueError("answers must be objects")
    for item in review.get("review", []):
        if "word_ids" in item:
            refs = item["word_ids"]
            if (not isinstance(refs, list) or any(not isinstance(x, str) for x in refs)
                    or len(set(refs)) != len(refs) or not set(refs) <= word_ids):
                raise ValueError("review references unknown or duplicate words")
        _validate_times(item, "t1", "t2")
    questions = review.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("questions must be a list")
    question_ids = set()
    for question in questions:
        if not isinstance(question, dict) or question.get("id") not in review_ids:
            raise ValueError("question references missing review item")
        if question["id"] in question_ids:
            raise ValueError("duplicate question")
        question_ids.add(question["id"])
        refs = question.get("word_ids", [])
        if (not isinstance(refs, list) or any(not isinstance(x, str) for x in refs) or
                len(set(refs)) != len(refs) or not set(refs) <= word_ids):
            raise ValueError("question references unknown or duplicate words")
    version = review.get("question_version")
    if version is not None and not isinstance(version, str):
        raise ValueError("invalid question version")
    for ident, answer in answers.items():
        if "id" in answer and answer["id"] != ident:
            raise ValueError("answer ID mismatch")
        if "version" in answer and (version is None or answer["version"] != version):
            raise ValueError("answer question version mismatch")
    applications = review.get("applications", [])
    if not isinstance(applications, list):
        raise ValueError("applications must be a list")
    for action in applications:
        if not isinstance(action, dict):
            raise ValueError("invalid application")
        if "answer_id" in action and action["answer_id"] not in answers:
            raise ValueError("application references missing answer")
        if "context" in action:
            _context(action["context"])
    _validate_embedded_provenance(transcript)
    _validate_embedded_provenance(review)


def _connect_readonly(path):
    """Соединение только для чтения; следов не оставляет только у закрытой базы.

    База всегда в режиме WAL. Обычное mode=ro на закрытой базе создаёт -wal/-shm и не может
    убрать их при закрытии. Если рядом нет ни -wal, ни -shm, все записи уже в основном файле
    и никто его не использует: открываем с immutable=1, и файлы не появляются. Если они есть,
    свежие данные могут лежать в -wal, а immutable=1 его игнорирует, поэтому открываем
    обычным mode=ro. Это не бесследно (читатель WAL всегда отмечается в -shm):
      - есть -wal и -shm (пишут в базу или остались после сбоя): меняется содержимое -shm;
      - есть только -wal: создаётся новый -shm.
    Основной файл и -wal не меняются, данные записей не затрагиваются.
    """
    resolved = check_local_path(path).resolve()
    busy = any(Path(str(resolved) + suffix).exists() for suffix in ("-wal", "-shm"))
    return _connect(resolved, "ro" if busy else "ro&immutable=1")


def _connect(path, mode):
    uri = check_local_path(path).resolve().as_uri() + "?mode=" + mode
    try:
        db = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5)
    except sqlite3.OperationalError as exc:
        if _is_busy(exc):
            raise Busy("database is busy") from exc
        raise StorageError("cannot open database") from exc
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
    except sqlite3.DatabaseError as exc:
        db.close()
        if _is_busy(exc):
            raise Busy("database is busy") from exc
        raise UnsupportedSchema("invalid or unsupported database") from exc
    return db


def _initialize_at(path, initial_display_name=None):
    """Явно создать новую пустую БД; существующий файл не заменяется."""
    if initial_display_name is not None and not isinstance(initial_display_name, str):
        raise ValueError("display name must be text")
    path = Path(path)
    with path.open("xb"):
        pass
    db = _connect(path, "rw")
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.executescript("""
        BEGIN IMMEDIATE;
        CREATE TABLE store_meta (store_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL);
        CREATE TABLE local_users (user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE installations (installation_id TEXT PRIMARY KEY,
            current_user_id TEXT NOT NULL REFERENCES local_users(user_id),
            active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)));
        CREATE TABLE recordings (recording_id TEXT PRIMARY KEY, location TEXT NOT NULL,
            source_revision INTEGER NOT NULL, review_revision INTEGER NOT NULL,
            FOREIGN KEY(recording_id, source_revision)
                REFERENCES transcript_revisions(recording_id, revision) DEFERRABLE INITIALLY DEFERRED,
            FOREIGN KEY(recording_id, review_revision)
                REFERENCES review_revisions(recording_id, revision) DEFERRABLE INITIALLY DEFERRED);
        CREATE TABLE operations (operation_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
            parameters_hash TEXT NOT NULL, context_json TEXT NOT NULL, result_json TEXT NOT NULL,
            author_user_id TEXT REFERENCES local_users(user_id),
            initiator_user_id TEXT REFERENCES local_users(user_id),
            installation_id TEXT REFERENCES installations(installation_id), created_at TEXT NOT NULL);
        CREATE TABLE transcript_revisions (recording_id TEXT NOT NULL REFERENCES recordings(recording_id),
            revision INTEGER NOT NULL, parent_revision INTEGER, payload TEXT NOT NULL,
            payload_hash TEXT NOT NULL, operation_id TEXT NOT NULL REFERENCES operations(operation_id),
            restored_from INTEGER,
            PRIMARY KEY(recording_id, revision),
            FOREIGN KEY(recording_id, parent_revision) REFERENCES transcript_revisions(recording_id, revision),
            FOREIGN KEY(recording_id, restored_from) REFERENCES transcript_revisions(recording_id, revision));
        CREATE TABLE review_revisions (recording_id TEXT NOT NULL REFERENCES recordings(recording_id),
            revision INTEGER NOT NULL, parent_revision INTEGER, source_revision INTEGER NOT NULL,
            payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
            operation_id TEXT NOT NULL REFERENCES operations(operation_id),
            restored_from INTEGER,
            PRIMARY KEY(recording_id, revision),
            FOREIGN KEY(recording_id, parent_revision) REFERENCES review_revisions(recording_id, revision),
            FOREIGN KEY(recording_id, source_revision) REFERENCES transcript_revisions(recording_id, revision),
            FOREIGN KEY(recording_id, restored_from) REFERENCES review_revisions(recording_id, revision));
        CREATE TABLE document_versions (
            recording_id TEXT NOT NULL REFERENCES recordings(recording_id),
            document_revision INTEGER NOT NULL,
            source_revision INTEGER NOT NULL,
            review_revision INTEGER NOT NULL,
            md BLOB NOT NULL, json BLOB NOT NULL, manifest BLOB NOT NULL, archive_manifest BLOB NOT NULL,
            hashes TEXT NOT NULL, operation_id TEXT NOT NULL REFERENCES operations(operation_id),
            PRIMARY KEY(recording_id, document_revision),
            FOREIGN KEY(recording_id, source_revision) REFERENCES transcript_revisions(recording_id, revision),
            FOREIGN KEY(recording_id, review_revision) REFERENCES review_revisions(recording_id, revision));
        CREATE TABLE allowed_roots (
            root_id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE,
            requires_remap INTEGER NOT NULL DEFAULT 0 CHECK(requires_remap IN (0,1)),
            role TEXT NOT NULL DEFAULT 'documents');
        CREATE TABLE restore_guard (id INTEGER PRIMARY KEY CHECK(id=1),
            locked INTEGER NOT NULL CHECK(locked IN (0,1)));
        INSERT INTO restore_guard VALUES (1, 0);
        CREATE TABLE document_outbox (
            recording_id TEXT NOT NULL, document_revision INTEGER NOT NULL,
            root_id TEXT NOT NULL REFERENCES allowed_roots(root_id),
            state TEXT NOT NULL CHECK(state IN ('pending', 'writing', 'synced', 'conflict', 'error')),
            generation INTEGER NOT NULL, expected_previous_hashes TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, error_code TEXT,
            PRIMARY KEY(recording_id, document_revision),
            FOREIGN KEY(recording_id, document_revision)
                REFERENCES document_versions(recording_id, document_revision));
        CREATE TABLE file_locations (
            root_id TEXT NOT NULL REFERENCES allowed_roots(root_id), relative_path TEXT NOT NULL,
            canonical_path TEXT NOT NULL UNIQUE,
            recording_id TEXT NOT NULL REFERENCES recordings(recording_id), kind TEXT NOT NULL,
            confirmed_revision INTEGER, confirmed_hash TEXT, generation INTEGER NOT NULL DEFAULT 0,
            conflict INTEGER NOT NULL DEFAULT 0 CHECK(conflict IN (0,1)),
            PRIMARY KEY(root_id, relative_path));
        CREATE TABLE document_conflicts (
            conflict_id TEXT PRIMARY KEY, recording_id TEXT NOT NULL, document_revision INTEGER NOT NULL,
            kind TEXT NOT NULL, relative_path TEXT NOT NULL, observed_bytes BLOB,
            observed_hash TEXT NOT NULL, observed_size INTEGER NOT NULL,
            resolution_operation_id TEXT REFERENCES operations(operation_id),
            FOREIGN KEY(recording_id, document_revision)
                REFERENCES document_outbox(recording_id, document_revision));
        CREATE TABLE materialization_events (
            event_id TEXT PRIMARY KEY, recording_id TEXT NOT NULL, document_revision INTEGER NOT NULL,
            kind TEXT NOT NULL, event_type TEXT NOT NULL, created_at TEXT NOT NULL,
            FOREIGN KEY(recording_id, document_revision)
                REFERENCES document_outbox(recording_id, document_revision));
        CREATE TABLE conspects (
            conspect_id TEXT PRIMARY KEY, recording_id TEXT NOT NULL REFERENCES recordings(recording_id),
            name TEXT NOT NULL, root_id TEXT NOT NULL REFERENCES allowed_roots(root_id),
            json_relative TEXT NOT NULL, md_relative TEXT NOT NULL, current_revision INTEGER NOT NULL,
            UNIQUE(recording_id, name),
            FOREIGN KEY(conspect_id, current_revision)
                REFERENCES conspect_revisions(conspect_id, revision) DEFERRABLE INITIALLY DEFERRED);
        CREATE TABLE conspect_revisions (
            conspect_id TEXT NOT NULL REFERENCES conspects(conspect_id),
            recording_id TEXT NOT NULL REFERENCES recordings(recording_id),
            revision INTEGER NOT NULL, parent_revision INTEGER, source_revision INTEGER NOT NULL,
            payload TEXT NOT NULL, payload_hash TEXT NOT NULL, json BLOB NOT NULL, md BLOB NOT NULL,
            json_hash TEXT NOT NULL, md_hash TEXT NOT NULL,
            operation_id TEXT NOT NULL REFERENCES operations(operation_id),
            context_json TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY(conspect_id, revision),
            FOREIGN KEY(conspect_id, parent_revision) REFERENCES conspect_revisions(conspect_id, revision),
            FOREIGN KEY(conspect_id) REFERENCES conspects(conspect_id),
            FOREIGN KEY(recording_id, source_revision)
                REFERENCES transcript_revisions(recording_id, revision));
        CREATE TABLE conspect_outbox (
            conspect_id TEXT NOT NULL, revision INTEGER NOT NULL,
            root_id TEXT NOT NULL REFERENCES allowed_roots(root_id),
            state TEXT NOT NULL CHECK(state IN ('pending','writing','synced','conflict','error')),
            expected_previous_hashes TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            error_code TEXT, PRIMARY KEY(conspect_id, revision),
            FOREIGN KEY(conspect_id, revision) REFERENCES conspect_revisions(conspect_id, revision));
        CREATE TABLE conspect_file_locations (
            root_id TEXT NOT NULL REFERENCES allowed_roots(root_id), relative_path TEXT NOT NULL,
            canonical_path TEXT NOT NULL UNIQUE,
            conspect_id TEXT NOT NULL REFERENCES conspects(conspect_id), kind TEXT NOT NULL,
            confirmed_revision INTEGER, confirmed_hash TEXT, generation INTEGER NOT NULL DEFAULT 0,
            conflict INTEGER NOT NULL DEFAULT 0 CHECK(conflict IN (0,1)),
            PRIMARY KEY(root_id, relative_path));
        CREATE TABLE conspect_conflicts (
            conflict_id TEXT PRIMARY KEY, conspect_id TEXT NOT NULL, revision INTEGER NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('json','md')), relative_path TEXT NOT NULL,
            observed_bytes BLOB, observed_hash TEXT NOT NULL, observed_size INTEGER NOT NULL,
            resolution_operation_id TEXT REFERENCES operations(operation_id),
            FOREIGN KEY(conspect_id, revision) REFERENCES conspect_outbox(conspect_id, revision));
        CREATE TABLE workflow_requests (
            request_id TEXT PRIMARY KEY, source TEXT NOT NULL, policy_json TEXT NOT NULL,
            context_json TEXT NOT NULL, state TEXT NOT NULL, job_id TEXT UNIQUE,
            recording_id TEXT REFERENCES recordings(recording_id), model_task_json TEXT,
            model_result_json TEXT, model_attempts INTEGER NOT NULL DEFAULT 0,
            policy_revision INTEGER NOT NULL DEFAULT 1, model_run_id TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE workflow_policy_versions (
            request_id TEXT NOT NULL REFERENCES workflow_requests(request_id),
            revision INTEGER NOT NULL, policy_json TEXT NOT NULL, context_json TEXT NOT NULL,
            created_at TEXT NOT NULL, PRIMARY KEY(request_id, revision));
        CREATE TABLE workflow_assessments (
            request_id TEXT NOT NULL REFERENCES workflow_requests(request_id), assessment_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(request_id, assessment_hash));
        CREATE TABLE workflow_verifications (
            verification_id TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES workflow_requests(request_id),
            recording_id TEXT NOT NULL REFERENCES recordings(recording_id), source_revision INTEGER NOT NULL,
            review_revision INTEGER NOT NULL, context_json TEXT NOT NULL, created_at TEXT NOT NULL,
            FOREIGN KEY(recording_id, source_revision) REFERENCES transcript_revisions(recording_id, revision),
            FOREIGN KEY(recording_id, review_revision) REFERENCES review_revisions(recording_id, revision));
        CREATE TABLE workflow_model_parts (
            request_id TEXT NOT NULL REFERENCES workflow_requests(request_id),
            task_hash TEXT NOT NULL, part_index INTEGER NOT NULL, result_json TEXT NOT NULL,
            created_at TEXT NOT NULL, PRIMARY KEY(request_id, task_hash, part_index));
        CREATE TABLE workflow_question_slots (
            request_id TEXT NOT NULL REFERENCES workflow_requests(request_id), slot INTEGER NOT NULL,
            question_id TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY(request_id, slot), UNIQUE(request_id, question_id));
        CREATE TABLE workflow_runs (
            run_id TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES workflow_requests(request_id),
            kind TEXT NOT NULL, state TEXT NOT NULL, executor TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE workflow_steps (
            step_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES workflow_runs(run_id),
            kind TEXT NOT NULL, state TEXT NOT NULL, result_json TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(run_id, kind));
        """ + REVIEW_EVENTS_DDL + ";\n" + HYPOTHESES_DDL + ";\n" + HYPOTHESES_INDEX_DDL + ";")
        db.execute("INSERT INTO store_meta VALUES (?, ?)", (str(uuid.uuid4()), SCHEMA_VERSION))
        if initial_display_name is not None:
            user_id, installation_id = str(uuid.uuid4()), str(uuid.uuid4())
            now = _now()
            db.execute("INSERT INTO local_users VALUES (?, ?, ?, ?)",
                       (user_id, initial_display_name, now, now))
            db.execute("INSERT INTO installations VALUES (?, ?, 1)", (installation_id, user_id))
        db.execute("COMMIT")
    except BaseException:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    return open_store(path)


def initialize(path, *, initial_display_name=None):
    """Подготовить новую БД отдельно и опубликовать без замены существующего файла.

    Если передано initial_display_name, локальный пользователь и установка
    создаются в той же транзакции, что и схема.
    """
    if initial_display_name is not None and not isinstance(initial_display_name, str):
        raise ValueError("display name must be text")
    path = check_local_path(path)
    if path.exists():
        raise FileExistsError(path)
    staged = path.with_name(path.name + ".incomplete-" + uuid.uuid4().hex)
    try:
        created = _initialize_at(staged, initial_display_name)
        created.close()
        validate_copy(staged)
        _release_wal_files(staged)
        os.link(staged, path)
    except BaseException:
        _remove_unpublished(staged)       # свой незаконченный файл; каждый повтор иначе оставлял бы ещё один
        raise
    staged.unlink()
    return open_store(path)


def _release_wal_files(path):
    """Убрать служебные -wal/-shm временного имени до публикации файла.

    validate_copy обычно не оставляет служебных файлов (см. _connect_readonly), но страхуемся:
    если они уже были, соединение только для чтения не может их удалить, и они остались бы
    рядом под временным именем. Последнее закрываемое соединение для записи само переносит
    журнал в файл и удаляет их; журнал SQLite открывает только при первом обращении к базе,
    поэтому нужно одно чтение.
    """
    db = _connect(path, "rw")
    try:
        db.execute("SELECT count(*) FROM sqlite_master").fetchone()
    finally:
        db.close()


def _remove_unpublished(staged):
    """Удалить неопубликованный временный файл (резерва или новой базы) и его служебные -wal/-shm/-journal.

    Только имена этого вызова (`<назначение>.incomplete-<uuid>`). Ошибка удаления (файл держит другая программа)
    не заменяет исходную ошибку: файл тогда остаётся и готовым резервом или базой не считается.
    """
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            staged.with_name(staged.name + suffix).unlink(missing_ok=True)
        except OSError:
            pass


def _has_table(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _check_schema(db):
    required = {"store_meta", "local_users", "installations", "recordings", "operations",
                "transcript_revisions", "review_revisions", "document_versions", "allowed_roots",
                "restore_guard", "document_outbox", "document_conflicts", "file_locations",
                "materialization_events", "conspects", "conspect_revisions", "conspect_outbox",
                "conspect_file_locations", "conspect_conflicts", "workflow_requests",
                "workflow_runs", "workflow_steps", "workflow_policy_versions", "workflow_assessments",
                "workflow_verifications", "workflow_model_parts", "workflow_question_slots"}
    try:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not required <= names:
            raise UnsupportedSchema("required tables missing")
        columns = {
            "store_meta": {"store_id", "schema_version"},
            "recordings": {"recording_id", "source_revision", "review_revision"},
            "operations": {"operation_id", "context_json", "parameters_hash", "result_json"},
            "transcript_revisions": {"recording_id", "revision", "payload", "payload_hash"},
            "review_revisions": {"recording_id", "revision", "source_revision", "payload", "payload_hash"},
            "document_versions": {"recording_id", "document_revision", "source_revision", "review_revision",
                                  "md", "json", "manifest", "archive_manifest", "hashes"},
            "document_outbox": {"recording_id", "document_revision", "root_id", "state",
                                "generation", "expected_previous_hashes"},
            "file_locations": {"root_id", "relative_path", "canonical_path", "confirmed_revision",
                               "confirmed_hash", "generation", "conflict"},
            "conspects": {"conspect_id", "recording_id", "name", "root_id", "json_relative",
                          "md_relative", "current_revision"},
            "conspect_revisions": {"conspect_id", "recording_id", "revision", "parent_revision", "source_revision",
                                   "payload", "payload_hash", "json", "md", "json_hash", "md_hash"},
            "conspect_outbox": {"conspect_id", "revision", "root_id", "state",
                                "expected_previous_hashes", "attempts"},
            "conspect_file_locations": {"root_id", "relative_path", "canonical_path", "conspect_id",
                                        "kind", "confirmed_revision", "confirmed_hash", "generation", "conflict"},
            "conspect_conflicts": {"conflict_id", "conspect_id", "revision", "kind", "relative_path",
                                   "observed_bytes", "observed_hash", "observed_size", "resolution_operation_id"},
            "workflow_requests": {"request_id", "source", "policy_json", "context_json", "state",
                                  "job_id", "recording_id", "model_task_json", "model_result_json",
                                  "model_attempts", "policy_revision", "model_run_id", "created_at", "updated_at"},
            "workflow_policy_versions": {"request_id", "revision", "policy_json", "context_json", "created_at"},
            "workflow_assessments": {"request_id", "assessment_hash", "payload_json", "created_at"},
            "workflow_verifications": {"verification_id", "request_id", "recording_id", "source_revision",
                                       "review_revision", "context_json", "created_at"},
            "workflow_model_parts": {"request_id", "task_hash", "part_index", "result_json", "created_at"},
            "workflow_question_slots": {"request_id", "slot", "question_id", "created_at"},
            "workflow_runs": {"run_id", "request_id", "kind", "state", "executor", "created_at", "updated_at"},
            "workflow_steps": {"step_id", "run_id", "kind", "state", "result_json", "created_at", "updated_at"},
        }
        # таблица событий карточек необязательна: базы до Х1 её не имеют (номер схемы тот же, 4)
        if "review_events" in names:
            columns["review_events"] = REVIEW_EVENTS_COLUMNS
        # таблица гипотез необязательна: базы до Г-1 её не имеют (номер схемы тот же, 4)
        if "hypotheses" in names:
            columns["hypotheses"] = HYPOTHESES_COLUMNS
        for table, expected in columns.items():
            if not expected <= {row[1] for row in db.execute(f"PRAGMA table_info({table})")}:
                raise UnsupportedSchema("required columns missing")
        rows = db.execute("SELECT schema_version FROM store_meta").fetchall()
        if len(rows) != 1 or rows[0][0] != SCHEMA_VERSION:
            raise UnsupportedSchema("unsupported schema")
    except sqlite3.DatabaseError as exc:
        if _is_busy(exc):
            raise Busy("database is busy") from exc
        raise UnsupportedSchema("invalid or unsupported database") from exc


def open_store(path):
    """Открыть существующую совместимую БД, не создавать и не мигрировать."""
    db = _connect(path, "rw")
    try:
        _check_schema(db)
        db.execute("PRAGMA synchronous=FULL")
        return Store(db)
    except BaseException:
        db.close()
        raise


def open_store_readonly(path):
    """Открыть существующую совместимую БД только для чтения."""
    db = _connect_readonly(path)
    try:
        _check_schema(db)
        db.execute("PRAGMA query_only=ON")
        return Store(db)
    except BaseException:
        db.close()
        raise


def damaged_conflict_copies(db):
    """Копии правок, у которых не совпал размер или сумма. Отсутствие копии (файл больше предела) — не повреждение."""
    found = []
    for scope, table in (("document", "document_conflicts"), ("conspect", "conspect_conflicts")):
        for row in db.execute("SELECT conflict_id, relative_path, observed_bytes, observed_hash, observed_size "
                              f"FROM {table} ORDER BY rowid"):
            content = row["observed_bytes"]
            if content is not None and (len(content) != row["observed_size"] or
                                        _hash_bytes(content) != row["observed_hash"]):
                found.append({"scope": scope, "conflict_id": row["conflict_id"],
                              "relative_path": row["relative_path"]})
    return found


def validate_copy(path, *, allow_damaged_copies=False):
    """Проверка копии без записи и запуска восстановленных очередей.

    allow_damaged_copies — только для резерва и восстановления (решение владельца 23.11.2026): повреждённая
    копия правки не отказ, а список damaged_copies в отчёте. Любое другое нарушение — отказ, как прежде.
    """
    db = _connect_readonly(path)
    try:
        _check_schema(db)
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise StorageError("integrity check failed")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise StorageError("foreign key check failed")
        checker = Store(db)
        for table in ("transcript_revisions", "review_revisions"):
            for row in db.execute(f"SELECT payload, payload_hash FROM {table}"):
                payload = _loads(row["payload"])
                if _hash(payload) != row["payload_hash"]:
                    raise StorageError("snapshot hash mismatch")
                checker._check_snapshot_refs(payload)
        for row in db.execute("SELECT conspect_id, revision FROM conspect_revisions"):
            checker.get_conspect(row["conspect_id"], row["revision"])
        for row in db.execute("SELECT * FROM operations"):
            context = _loads(row["context_json"])
            _context(context)
            if (context["author_ref"]["user_id"] != row["author_user_id"] or
                    context["initiated_by"]["user_id"] != row["initiator_user_id"] or
                    context["origin_installation_id"] != row["installation_id"]):
                raise StorageError("operation author reference mismatch")
        if _has_table(db, "review_events"):
            joined = 0
            for row in db.execute("SELECT e.event_json, e.origin, o.kind AS operation_kind FROM review_events e "
                                  "JOIN operations o ON o.operation_id=e.operation_id"):
                joined += 1
                if not isinstance(_loads(row["event_json"]), dict):
                    raise StorageError("review event must be an object")
                if row["origin"] == "page" and row["operation_kind"] != "review_event":
                    raise StorageError("review event operation mismatch")
            if joined != db.execute("SELECT count(*) FROM review_events").fetchone()[0]:
                raise StorageError("review event without operation")
        if _has_table(db, "hypotheses"):
            for row in db.execute("SELECT record_json, size_bytes FROM hypotheses"):
                if not isinstance(_loads(row["record_json"]), dict):
                    raise StorageError("hypothesis record must be an object")
                if len(row["record_json"].encode("utf-8")) != row["size_bytes"]:
                    raise StorageError("hypothesis record size mismatch")
        for row in db.execute("SELECT r.payload AS review, t.payload AS transcript FROM review_revisions r "
                              "JOIN transcript_revisions t ON t.recording_id=r.recording_id "
                              "AND t.revision=r.source_revision"):
            validate_pair(_loads(row["transcript"]), _loads(row["review"]))
        for row in db.execute("SELECT r.source_revision, v.source_revision AS reviewed "
                              "FROM recordings r JOIN review_revisions v ON v.recording_id=r.recording_id "
                              "AND v.revision=r.review_revision"):
            if row["source_revision"] != row["reviewed"]:
                raise StorageError("incompatible current revisions")
        for row in db.execute("SELECT * FROM document_versions"):
            reviewed = db.execute("SELECT source_revision, payload FROM review_revisions "
                                  "WHERE recording_id=? AND revision=?",
                                  (row["recording_id"], row["review_revision"])).fetchone()
            source = db.execute("SELECT payload FROM transcript_revisions WHERE recording_id=? AND revision=?",
                                (row["recording_id"], row["source_revision"])).fetchone()
            if reviewed is None or source is None or reviewed["source_revision"] != row["source_revision"]:
                raise StorageError("document has incompatible snapshots")
            validate_pair(_loads(source["payload"]), _loads(reviewed["payload"]))
            hashes = _loads(row["hashes"])
            for key in ("md", "json", "manifest", "archive_manifest"):
                if hashlib.sha256(row[key]).hexdigest() != hashes[key]:
                    raise StorageError("document hash mismatch")
            passport = _loads(row["json"])["passport"]
            for key in ("recording_id", "document_revision", "source_revision", "review_revision"):
                if passport[key] != row[key]:
                    raise StorageError("document passport mismatch")
            manifest = _loads(row["manifest"])
            archive_manifest = _loads(row["archive_manifest"])
            if manifest["passport"] != passport or archive_manifest["passport"] != passport:
                raise StorageError("manifest passport mismatch")
            expected = {hashes["md"], hashes["json"]}
            if ({item["sha256"] for item in manifest["files"].values()} != expected or
                    {item["sha256"] for item in archive_manifest["files"].values()} != expected):
                raise StorageError("manifest hash mismatch")
        damaged = damaged_conflict_copies(db)
        if damaged and not allow_damaged_copies:
            prefix = "conspect " if damaged[0]["scope"] == "conspect" else ""
            raise StorageError(prefix + "conflict copy hash mismatch")
        for row in db.execute("SELECT * FROM document_outbox"):
            if row["generation"] != row["document_revision"]:
                raise StorageError("outbox generation mismatch")
            previous = {}
            if row["document_revision"] > 1:
                previous_row = db.execute("SELECT hashes FROM document_versions WHERE recording_id=? "
                                          "AND document_revision=?",
                                          (row["recording_id"], row["document_revision"] - 1)).fetchone()
                if previous_row is None:
                    raise StorageError("missing prior document")
                previous = _loads(previous_row["hashes"])
            expected = {kind: previous.get(kind) for kind in ("md", "json", "manifest")}
            if _loads(row["expected_previous_hashes"]) != expected:
                raise StorageError("outbox expected hash mismatch")
        for row in db.execute("SELECT f.*, r.path AS root_path FROM file_locations f "
                              "JOIN allowed_roots r ON r.root_id=f.root_id"):
            canonical = os.path.normcase(os.path.abspath(os.path.join(row["root_path"], row["relative_path"])))
            if canonical != row["canonical_path"]:
                raise StorageError("file target mismatch")
            if row["confirmed_revision"] is not None:
                doc = db.execute("SELECT hashes FROM document_versions WHERE recording_id=? AND document_revision=?",
                                 (row["recording_id"], row["confirmed_revision"])).fetchone()
                key = _hash_key(row["kind"])
                if doc is None or _loads(doc["hashes"]).get(key) != row["confirmed_hash"]:
                    raise StorageError("confirmed file hash mismatch")
        for row in db.execute(
                "SELECT f.*, r.path AS root_path, c.root_id AS conspect_root, "
                "c.json_relative, c.md_relative FROM conspect_file_locations f "
                "JOIN allowed_roots r ON r.root_id=f.root_id JOIN conspects c USING(conspect_id)"):
            if (row["kind"] not in ("json", "md") or row["root_id"] != row["conspect_root"] or
                    row["relative_path"] != row[row["kind"] + "_relative"]):
                raise StorageError("conspect file registry mismatch")
            canonical = os.path.normcase(os.path.abspath(os.path.join(row["root_path"], row["relative_path"])))
            if canonical != row["canonical_path"]:
                raise StorageError("conspect file target mismatch")
            if row["confirmed_revision"] is not None:
                saved = checker.get_conspect(row["conspect_id"], row["confirmed_revision"])
                if saved[row["kind"] + "_hash"] != row["confirmed_hash"]:
                    raise StorageError("confirmed conspect file hash mismatch")
        for row in db.execute("SELECT conspect_id, root_id, json_relative, md_relative FROM conspects"):
            registered = db.execute(
                "SELECT kind, relative_path FROM conspect_file_locations WHERE conspect_id=? AND root_id=?",
                (row["conspect_id"], row["root_id"])).fetchall()
            if {(item["kind"], item["relative_path"]) for item in registered} != {
                    ("json", row["json_relative"]), ("md", row["md_relative"])}:
                raise StorageError("incomplete conspect file registry")
        for row in db.execute("SELECT o.*, c.root_id AS conspect_root FROM conspect_outbox o "
                              "JOIN conspects c USING(conspect_id)"):
            if row["root_id"] != row["conspect_root"]:
                raise StorageError("conspect outbox root mismatch")
            previous = {}
            if row["revision"] > 1:
                prior = checker.get_conspect(row["conspect_id"], row["revision"] - 1)
                previous = {"json": prior["json_hash"], "md": prior["md_hash"]}
            if _loads(row["expected_previous_hashes"]) != previous:
                raise StorageError("conspect outbox expected hash mismatch")
        return {"integrity": "ok", "foreign_keys": "ok", "schema_version": SCHEMA_VERSION,
                "inactive": bool(db.execute("SELECT locked FROM restore_guard WHERE id=1").fetchone()[0]),
                **({"damaged_copies": damaged} if damaged else {})}
    except StorageError:
        raise
    except sqlite3.DatabaseError as exc:
        if _is_busy(exc):
            raise Busy("database is busy") from exc
        raise StorageError("invalid or unreadable database") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise StorageError("invalid or unreadable database") from exc
    finally:
        db.close()


class Store:
    def __init__(self, db):
        self.db = db
        rows = db.execute("PRAGMA database_list").fetchall()
        self.path = Path(rows[0]["file"]).resolve() if rows and rows[0]["file"] else None

    def close(self):
        self.db.close()

    def current_identity(self):
        """Вернуть единственную активную локальную installation/user привязку."""
        if self.db.execute("SELECT locked FROM restore_guard WHERE id=1").fetchone()[0]:
            raise Conflict("restored copy requires explicit rebind")
        rows = self.db.execute(
            "SELECT i.installation_id, u.user_id, u.display_name "
            "FROM installations i JOIN local_users u ON u.user_id=i.current_user_id "
            "WHERE i.active=1").fetchall()
        if len(rows) != 1:
            raise StorageError("store must have exactly one active installation")
        return dict(rows[0])

    def backup(self, destination, *, allow_damaged_copies=False):
        """SQLite backup в новый файл; восстановленная копия неактивна."""
        if self.db.in_transaction:
            raise Conflict("cannot back up an open transaction")
        destination = check_local_path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        staged = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
        with staged.open("xb"):
            pass
        try:
            target = _connect(staged, "rw")
            try:
                self.db.backup(target)
                target.execute("BEGIN IMMEDIATE")
                target.execute("UPDATE restore_guard SET locked=1 WHERE id=1")
                target.execute("UPDATE allowed_roots SET requires_remap=1")
                target.execute("UPDATE installations SET active=0")
                target.execute("UPDATE workflow_requests SET state='interrupted', job_id=NULL "
                               "WHERE state IN ('queued', 'starting', 'running')")
                target.execute("UPDATE workflow_runs SET state='interrupted' WHERE state IN ('queued', 'running')")
                target.execute("UPDATE workflow_steps SET state='interrupted' WHERE state IN ('queued', 'running')")
                target.execute("COMMIT")
            except BaseException:
                if target.in_transaction:
                    target.execute("ROLLBACK")
                raise
            finally:
                target.close()
            report = validate_copy(staged, allow_damaged_copies=allow_damaged_copies)
            _release_wal_files(staged)
            os.link(staged, destination)
        except BaseException:
            # Ф204-2 (слово владельца 26.11.2026): до публикации убрать только свой временный файл этого вызова;
            # назначение, база, другие .incomplete-* и резервы не трогаются; не удалось удалить — файл остаётся
            _remove_unpublished(staged)
            raise
        staged.unlink()
        return report

    def rebind_restored_copy(self, root_mapping, installation_id, user_id):
        """Явно переназначить все корни и новую установку в подготовленной копии."""
        import documents
        _uuid(installation_id)
        _uuid(user_id)
        if not isinstance(root_mapping, dict):
            raise ValueError("complete root mapping required")
        existing = self.db.execute("SELECT root_id, path FROM allowed_roots").fetchall()
        if set(root_mapping) != {row["root_id"] for row in existing}:
            raise ValueError("all roots must be remapped")
        paths = {key: os.path.normcase(str(documents.checked_root(value)))
                 for key, value in root_mapping.items()}
        def overlaps(a, b):
            try:
                shared = os.path.commonpath((a, b))
            except ValueError:
                return False
            return shared == a or shared == b
        values = list(paths.values())
        if any(overlaps(a, b) for i, a in enumerate(values) for b in values[i + 1:]):
            raise ValueError("restored roots overlap")
        old_roots = [row["path"] for row in existing]
        for new in paths.values():
            for old in old_roots:
                if overlaps(new, old):
                    raise ValueError("old and new roots must not overlap")
        with self._transaction():
            if not self.db.execute("SELECT locked FROM restore_guard WHERE id=1").fetchone()[0]:
                raise Conflict("store is not an inactive restore")
            if self.db.execute("SELECT 1 FROM installations WHERE installation_id=?",
                               (installation_id,)).fetchone():
                raise Conflict("new installation ID required")
            if not self.db.execute("SELECT 1 FROM local_users WHERE user_id=?", (user_id,)).fetchone():
                raise ValueError("unknown user")
            for root_id, path in paths.items():
                self.db.execute("UPDATE allowed_roots SET path=?, requires_remap=0 WHERE root_id=?",
                                (path, root_id))
                for location in self.db.execute("SELECT relative_path FROM file_locations WHERE root_id=?",
                                                (root_id,)).fetchall():
                    canonical = os.path.normcase(os.path.abspath(os.path.join(path, location["relative_path"])))
                    self.db.execute("UPDATE file_locations SET canonical_path=?, confirmed_revision=NULL, "
                                    "confirmed_hash=NULL, generation=0 WHERE root_id=? AND relative_path=?",
                                    (canonical, root_id, location["relative_path"]))
                for location in self.db.execute(
                        "SELECT relative_path FROM conspect_file_locations WHERE root_id=?",
                        (root_id,)).fetchall():
                    canonical = os.path.normcase(os.path.abspath(os.path.join(path, location["relative_path"])))
                    self.db.execute(
                        "UPDATE conspect_file_locations SET canonical_path=?, confirmed_revision=NULL, "
                        "confirmed_hash=NULL, generation=0 WHERE root_id=? AND relative_path=?",
                        (canonical, root_id, location["relative_path"]))
            self.db.execute("UPDATE document_outbox SET state='pending', error_code=NULL WHERE state!='conflict'")
            self.db.execute("UPDATE conspect_outbox SET state='pending', error_code=NULL WHERE state!='conflict'")
            self.db.execute("INSERT INTO installations VALUES (?, ?, 1)", (installation_id, user_id))
            self.db.execute("UPDATE restore_guard SET locked=0 WHERE id=1")

    @contextmanager
    def _transaction(self):
        if self.db.in_transaction:
            name = "sp_" + uuid.uuid4().hex
            self.db.execute("SAVEPOINT " + name)
            try:
                yield
                self.db.execute("RELEASE SAVEPOINT " + name)
            except BaseException:
                self.db.execute("ROLLBACK TO SAVEPOINT " + name)
                self.db.execute("RELEASE SAVEPOINT " + name)
                raise
            return
        try:
            self.db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if _is_busy(exc):
                raise Busy("database is busy") from exc
            raise StorageError("cannot begin transaction") from exc
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def create_user(self, display_name):
        if not isinstance(display_name, str):
            raise ValueError("name must be text")
        ident = str(uuid.uuid4())
        with self._transaction():
            if self.db.execute("SELECT locked FROM restore_guard WHERE id=1").fetchone()[0]:
                raise Conflict("restored copy is inactive")
            now = _now()
            self.db.execute("INSERT INTO local_users VALUES (?, ?, ?, ?)", (ident, display_name, now, now))
        return ident

    def rename_user(self, user_id, display_name):
        _uuid(user_id)
        if not isinstance(display_name, str):
            raise ValueError("name must be text")
        with self._transaction():
            if self.db.execute("SELECT locked FROM restore_guard WHERE id=1").fetchone()[0]:
                raise Conflict("restored copy is inactive")
            cur = self.db.execute("UPDATE local_users SET display_name=?, updated_at=? WHERE user_id=?",
                                  (display_name, _now(), user_id))
            if cur.rowcount != 1:
                raise KeyError(user_id)

    def bind_installation(self, installation_id, user_id):
        _uuid(installation_id)
        _uuid(user_id)
        with self._transaction():
            if self.db.execute("SELECT locked FROM restore_guard WHERE id=1").fetchone()[0]:
                raise Conflict("restore copy requires explicit rebind")
            if not self.db.execute("SELECT 1 FROM local_users WHERE user_id=?", (user_id,)).fetchone():
                raise ValueError("unknown user")
            self.db.execute("INSERT INTO installations VALUES (?, ?, 1) "
                            "ON CONFLICT(installation_id) DO UPDATE SET current_user_id=excluded.current_user_id, active=1",
                            (installation_id, user_id))

    def register_root(self, path):
        """Явно зарегистрировать существующий корень; без создания каталогов."""
        import documents
        root = documents.checked_root(path)
        normalized = os.path.normcase(str(root))
        root_id = str(uuid.uuid4())
        with self._transaction():
            if self.db.execute("SELECT locked FROM restore_guard WHERE id=1").fetchone()[0]:
                raise Conflict("restore copy requires explicit rebind")
            existing = self.db.execute("SELECT path FROM allowed_roots").fetchall()
            for row in existing:
                registered = os.path.normcase(row["path"])
                if registered == normalized:
                    raise Conflict("root already registered")
                try:
                    shared = os.path.commonpath((normalized, registered))
                except ValueError:
                    continue
                if shared in (normalized, registered):
                    raise Conflict("root overlaps an existing root")
            self.db.execute("INSERT INTO allowed_roots(root_id, path) VALUES (?, ?)",
                            (root_id, normalized))
        return root_id

    def get_or_register_root(self, path):
        """Переиспользовать точный разрешённый корень или зарегистрировать новый."""
        import documents
        root = documents.checked_root(path)
        normalized = os.path.normcase(str(root))
        row = self.db.execute("SELECT root_id FROM allowed_roots WHERE path=?", (normalized,)).fetchone()
        if row is not None:
            if self.db.execute("SELECT locked FROM restore_guard WHERE id=1").fetchone()[0]:
                raise Conflict("restore copy requires explicit rebind")
            return row["root_id"]
        try:
            return self.register_root(root)
        except Conflict:
            # Другой процесс мог зарегистрировать тот же корень между чтением и вставкой.
            row = self.db.execute("SELECT root_id FROM allowed_roots WHERE path=?", (normalized,)).fetchone()
            if row is not None:
                return row["root_id"]
            raise

    def capture_context(self, installation_id, actor_kind):
        _uuid(installation_id)
        if actor_kind not in ACTOR_KINDS:
            raise ValueError("invalid actor kind")
        row = self.db.execute("SELECT u.user_id, u.display_name FROM installations i "
                              "JOIN local_users u ON u.user_id=i.current_user_id "
                              "WHERE i.installation_id=? AND i.active=1", (installation_id,)).fetchone()
        if row is None:
            raise KeyError(installation_id)
        known = {"user_id": row["user_id"], "user_name_snapshot": row["display_name"]}
        unknown = {"user_id": None, "user_name_snapshot": None}
        return {"actor_kind": actor_kind, "author_ref": known if actor_kind == "human" else unknown,
                "initiated_by": unknown if actor_kind == "human" else known,
                "origin_installation_id": installation_id}

    def _check_context_refs(self, ctx, *, require_active=True):
        if self.db.execute("SELECT locked FROM restore_guard WHERE id=1").fetchone()[0]:
            raise Conflict("restored copy is inactive")
        for field in ("author_ref", "initiated_by"):
            ident = ctx[field]["user_id"]
            if ident is not None and not self.db.execute(
                    "SELECT 1 FROM local_users WHERE user_id=?", (ident,)).fetchone():
                raise ValueError("unknown user reference")
        ident = ctx["origin_installation_id"]
        if ident is not None:
            query = "SELECT 1 FROM installations WHERE installation_id=?"
            if require_active:
                query += " AND active=1"
            if not self.db.execute(query, (ident,)).fetchone():
                raise ValueError("unknown installation reference")

    def _check_document_conflict(self, recording_id):
        if self.db.execute("SELECT 1 FROM document_outbox WHERE recording_id=? AND state='conflict'",
                           (recording_id,)).fetchone():
            raise Conflict("unresolved document conflict")

    def resolve_document_conflict(self, conflict_id, operation_id, context, *, without_copy=False):
        """Сохранённые внешние байты остаются в БД; повтор сверит их хэш перед заменой.

        without_copy — файл был больше предела и копии нет: решение принимается только по просьбе вызывающего,
        который проверил, что файл уже равен версии программы. Позже писатель такой файл не заменит: без копии
        он видит «испорченную копию» и останавливается."""
        _uuid(conflict_id)
        _uuid(operation_id)
        _context(context)
        if context["actor_kind"] != "human":
            raise ValueError("conflict resolution requires a human action")
        digest = _hash({"conflict_id": conflict_id, "context": context})
        with self._transaction():
            replay = self._replay(operation_id, "resolve_document_conflict", digest)
            if replay is not None:
                return replay
            self._check_context_refs(context)
            row = self.db.execute("SELECT * FROM document_conflicts WHERE conflict_id=?",
                                  (conflict_id,)).fetchone()
            if row is None:
                raise KeyError(conflict_id)
            if row["kind"].startswith("archive_"):
                raise Conflict("immutable archive conflict requires separate resolution")
            if row["observed_bytes"] is None:
                if not without_copy:
                    raise Conflict("external file too large to preserve; manual resolution required")
            elif (len(row["observed_bytes"]) != row["observed_size"] or
                    _hash_bytes(row["observed_bytes"]) != row["observed_hash"]):
                raise StorageError("document conflict copy hash mismatch")
            if row["resolution_operation_id"] is not None:
                raise Conflict("conflict already resolved")
            result = {"recording_id": row["recording_id"], "document_revision": row["document_revision"]}
            self._insert_operation(operation_id, "resolve_document_conflict", digest, context, result)
            self.db.execute("UPDATE document_conflicts SET resolution_operation_id=? WHERE conflict_id=?",
                            (operation_id, conflict_id))
            self.db.execute("UPDATE file_locations SET conflict=0 WHERE recording_id=? AND relative_path=?",
                            (row["recording_id"], row["relative_path"]))
            remaining = self.db.execute(
                "SELECT 1 FROM document_conflicts WHERE recording_id=? AND document_revision=? "
                "AND resolution_operation_id IS NULL", (row["recording_id"], row["document_revision"])).fetchone()
            if remaining is None:
                self.db.execute("UPDATE document_outbox SET state='pending', error_code=NULL "
                                "WHERE recording_id=? AND document_revision=? AND state='conflict'",
                                (row["recording_id"], row["document_revision"]))
            return result

    def resolve_conspect_conflict(self, conflict_id, operation_id, context, *, without_copy=False):
        """Зафиксировать решение человека; повторно заменять только те же внешние байты.

        without_copy — как у resolve_document_conflict: копии нет (файл больше предела), файл уже равен версии."""
        _uuid(conflict_id)
        _uuid(operation_id)
        _context(context)
        if context["actor_kind"] != "human":
            raise ValueError("conflict resolution requires a human action")
        digest = _hash({"conflict_id": conflict_id, "context": context})
        with self._transaction():
            replay = self._replay(operation_id, "resolve_conspect_conflict", digest)
            if replay is not None:
                return replay
            self._check_context_refs(context)
            row = self.db.execute("SELECT * FROM conspect_conflicts WHERE conflict_id=?",
                                  (conflict_id,)).fetchone()
            if row is None:
                raise KeyError(conflict_id)
            if row["observed_bytes"] is None:
                if not without_copy:
                    raise Conflict("external file too large to preserve; manual resolution required")
            elif (len(row["observed_bytes"]) != row["observed_size"] or
                    _hash_bytes(row["observed_bytes"]) != row["observed_hash"]):
                raise StorageError("conspect conflict copy hash mismatch")
            if row["resolution_operation_id"] is not None:
                raise Conflict("conflict already resolved")
            result = {"conspect_id": row["conspect_id"], "revision": row["revision"]}
            self._insert_operation(operation_id, "resolve_conspect_conflict", digest, context, result)
            self.db.execute("UPDATE conspect_conflicts SET resolution_operation_id=? WHERE conflict_id=?",
                            (operation_id, conflict_id))
            remaining = self.db.execute(
                "SELECT 1 FROM conspect_conflicts WHERE conspect_id=? AND revision=? "
                "AND resolution_operation_id IS NULL", (row["conspect_id"], row["revision"])).fetchone()
            if remaining is None:
                self.db.execute("UPDATE conspect_outbox SET state='pending', error_code=NULL "
                                "WHERE conspect_id=? AND revision=? AND state='conflict'",
                                (row["conspect_id"], row["revision"]))
            return result

    def _check_snapshot_refs(self, value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("author_ref", "initiated_by"):
                    _ref(item)
                    ident = item["user_id"]
                    if ident is not None and not self.db.execute(
                            "SELECT 1 FROM local_users WHERE user_id=?", (ident,)).fetchone():
                        raise ValueError("unknown snapshot user reference")
                elif key == "origin_installation_id":
                    if item is not None:
                        _uuid(item)
                        if not self.db.execute("SELECT 1 FROM installations WHERE installation_id=?",
                                               (item,)).fetchone():
                            raise ValueError("unknown snapshot installation reference")
                else:
                    self._check_snapshot_refs(item)
        elif isinstance(value, list):
            for item in value:
                self._check_snapshot_refs(item)

    def _replay(self, operation_id, kind, digest):
        row = self.db.execute("SELECT kind, parameters_hash, result_json FROM operations "
                              "WHERE operation_id=?", (operation_id,)).fetchone()
        if row:
            if row["kind"] != kind or row["parameters_hash"] != digest:
                raise Conflict("operation_id reused with different parameters")
            return _loads(row["result_json"])
        return None

    def _insert_operation(self, operation_id, kind, digest, context, result):
        self.db.execute("INSERT INTO operations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (operation_id, kind, digest, _json(context), _json(result),
                         context["author_ref"]["user_id"], context["initiated_by"]["user_id"],
                         context["origin_installation_id"], _now()))

    def create_recording(self, recording_id, location, transcript, review, operation_id, context,
                         *, _document_spec=None):
        _uuid(recording_id)
        _uuid(operation_id)
        _context(context)
        if not isinstance(location, str) or not location:
            raise ValueError("location required")
        _snapshot(transcript, TRANSCRIPT_FIELDS)
        _snapshot(review, REVIEW_FIELDS)
        validate_pair(transcript, review)
        params = {"recording_id": recording_id, "location": location, "transcript": transcript,
                  "review": review, "context": context, "document_spec": _document_spec}
        digest = _hash(params)
        with self._transaction():
            replay = self._replay(operation_id, "create", digest)
            if replay is not None:
                return replay
            self._check_context_refs(context)
            self._check_document_conflict(recording_id)
            self._check_snapshot_refs(transcript)
            self._check_snapshot_refs(review)
            if self.db.execute("SELECT 1 FROM recordings WHERE recording_id=?", (recording_id,)).fetchone():
                raise Conflict("recording already exists")
            result = {"recording_id": recording_id, "source_revision": 1, "review_revision": 1}
            self._insert_operation(operation_id, "create", digest, context, result)
            self.db.execute("INSERT INTO recordings VALUES (?, ?, 1, 1)", (recording_id, location))
            self.db.execute("INSERT INTO transcript_revisions VALUES (?, 1, NULL, ?, ?, ?, NULL)",
                            (recording_id, _json(transcript), _hash(transcript), operation_id))
            self.db.execute("INSERT INTO review_revisions VALUES (?, 1, NULL, 1, ?, ?, ?, NULL)",
                            (recording_id, _json(review), _hash(review), operation_id))
            return result

    def commit_change(self, recording_id, expected_versions, operation_id, context,
                      *, transcript=None, review=None, _restore_from=None, _document_spec=None):
        _uuid(recording_id)
        _uuid(operation_id)
        _context(context)
        if (not isinstance(expected_versions, dict) or
                set(expected_versions) != {"source_revision", "review_revision"} or
                any(type(v) is not int or v < 1 for v in expected_versions.values())):
            raise ValueError("expected source and review revisions required")
        if transcript is None and review is None:
            raise ValueError("no change")
        if transcript is not None:
            _snapshot(transcript, TRANSCRIPT_FIELDS)
            if review is None:
                raise ValueError("source change requires compatible review snapshot")
        if review is not None:
            _snapshot(review, REVIEW_FIELDS)
        params = {"recording_id": recording_id, "expected_versions": expected_versions,
                  "context": context, "transcript": transcript, "review": review,
                  "restore_from": _restore_from, "document_spec": _document_spec}
        digest = _hash(params)
        with self._transaction():
            replay = self._replay(operation_id, "change", digest)
            if replay is not None:
                return replay
            self._check_context_refs(context)
            self._check_document_conflict(recording_id)
            self._check_snapshot_refs(transcript)
            self._check_snapshot_refs(review)
            row = self.db.execute("SELECT source_revision, review_revision FROM recordings "
                                  "WHERE recording_id=?", (recording_id,)).fetchone()
            if row is None:
                raise KeyError(recording_id)
            if dict(row) != expected_versions:
                raise Conflict("stale revisions")
            current_transcript = (transcript if transcript is not None else
                                  self.get_revision(recording_id, "transcript",
                                                    row["source_revision"])["payload"])
            validate_pair(current_transcript, review)
            if _restore_from is not None:
                old_source, old_review = _restore_from
                old_t = self.get_revision(recording_id, "transcript", old_source)
                old_r = self.get_revision(recording_id, "review", old_review)
                if (old_r["source_revision"] != old_source or old_t["payload"] != transcript
                        or old_r["payload"] != review):
                    raise Conflict("restore tuple is incompatible")
            source = row["source_revision"] + (transcript is not None)
            rev = row["review_revision"] + (review is not None)
            result = {"recording_id": recording_id, "source_revision": source, "review_revision": rev}
            self._insert_operation(operation_id, "change", digest, context, result)
            if transcript is not None:
                self.db.execute("INSERT INTO transcript_revisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (recording_id, source, row["source_revision"], _json(transcript),
                                 _hash(transcript), operation_id,
                                 _restore_from[0] if _restore_from else None))
            if review is not None:
                self.db.execute("INSERT INTO review_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (recording_id, rev, row["review_revision"], source, _json(review),
                                 _hash(review), operation_id,
                                 _restore_from[1] if _restore_from else None))
            self.db.execute("UPDATE recordings SET source_revision=?, review_revision=? WHERE recording_id=?",
                            (source, rev, recording_id))
            return result

    def restore_as_new_revision(self, recording_id, source_revision, review_revision,
                                expected_versions, operation_id, context):
        old_t = self.get_revision(recording_id, "transcript", source_revision)
        old_r = self.get_revision(recording_id, "review", review_revision)
        if old_r["source_revision"] != source_revision:
            raise Conflict("restore tuple is incompatible")
        return self.commit_change(recording_id, expected_versions, operation_id, context,
                                  transcript=old_t["payload"], review=old_r["payload"],
                                  _restore_from=(source_revision, review_revision))

    def _insert_prepared_document(self, recording_id, revision, source_revision, review_revision,
                                  prepared, root_id, operation_id):
        latest = self.db.execute("SELECT COALESCE(MAX(document_revision), 0) FROM document_versions "
                                 "WHERE recording_id=?", (recording_id,)).fetchone()[0]
        if latest + 1 != revision:
            raise Conflict("document revision changed")
        if revision > 1:
            prev = self.db.execute("SELECT root_id FROM document_outbox WHERE recording_id=? "
                                   "AND document_revision=?", (recording_id, revision - 1)).fetchone()
            if prev is None or prev["root_id"] != root_id:
                raise Conflict("document generations require a single registered root")
        self.db.execute("INSERT INTO document_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (recording_id, revision, source_revision, review_revision,
                         prepared["md"], prepared["json"], prepared["manifest"],
                         prepared["archive_manifest"], _json(prepared["hashes"]), operation_id))
        self._enqueue_document(recording_id, revision, root_id, prepared["hashes"], prepared["paths"])

    def _enqueue_document(self, recording_id, revision, root_id, hashes, paths):
        import documents
        previous = self.get_document(recording_id, revision - 1)["hashes"] if revision > 1 else {}
        archives = documents.archive_paths(recording_id, revision)
        entries = [(kind, path) for kind, path in paths.items()]
        entries += [("archive_" + kind, path) for kind, path in archives.items()]
        root = self.db.execute("SELECT path FROM allowed_roots WHERE root_id=?", (root_id,)).fetchone()
        if root is None:
            raise ValueError("unknown root")
        for kind, path in entries:
            canonical = os.path.normcase(os.path.abspath(os.path.join(root["path"], path)))
            if self.db.execute("SELECT 1 FROM conspect_file_locations WHERE canonical_path=?",
                               (canonical,)).fetchone():
                raise Conflict("file target collides with conspect")
            self.db.execute("INSERT OR IGNORE INTO file_locations(root_id,relative_path,canonical_path,recording_id,kind) "
                            "VALUES (?, ?, ?, ?, ?)", (root_id, path, canonical, recording_id, kind))
            row = self.db.execute("SELECT recording_id,kind FROM file_locations "
                                  "WHERE root_id=? AND relative_path=?", (root_id, path)).fetchone()
            if row is None or row["recording_id"] != recording_id or row["kind"] != kind:
                raise Conflict("file target collision")
        self.db.execute("INSERT INTO document_outbox(recording_id, document_revision, root_id, state, "
                        "generation, expected_previous_hashes) VALUES (?, ?, ?, 'pending', ?, ?)",
                        (recording_id, revision, root_id, revision,
                         _json({kind: previous.get(kind) for kind in paths})))

    def _document_result(self, operation_id, result):
        row = self.db.execute("SELECT d.document_revision, o.state FROM document_versions d "
                              "JOIN document_outbox o USING(recording_id, document_revision) "
                              "WHERE d.operation_id=?", (operation_id,)).fetchone()
        if row is None:
            raise StorageError("operation has no queued document")
        return {**result, "document_revision": row["document_revision"],
                "document_sync": row["state"]}

    def create_recording_with_document(self, recording_id, location, transcript, review,
                                       operation_id, context, *, root_id, generated_at,
                                       document_paths=None):
        import documents
        _uuid(root_id)
        spec = {"root_id": root_id, "generated_at": generated_at,
                "document_paths": document_paths}
        params = {"recording_id": recording_id, "location": location, "transcript": transcript,
                  "review": review, "context": context, "document_spec": spec}
        digest = _hash(params)
        replay = self._replay(operation_id, "create", digest)
        if replay is not None:
            return self._document_result(operation_id, replay)
        prepared = documents.prepare(recording_id, 1, 1, 1, transcript, review, context, context, generated_at,
                                     document_paths=document_paths)
        with self._transaction():
            replay = self._replay(operation_id, "create", digest)
            if replay is not None:
                return self._document_result(operation_id, replay)
            result = self.create_recording(recording_id, location, transcript, review,
                                           operation_id, context, _document_spec=spec)
            self._insert_prepared_document(recording_id, 1, 1, 1, prepared, root_id, operation_id)
            return self._document_result(operation_id, result)

    def commit_change_with_document(self, recording_id, expected_versions, operation_id, context,
                                    *, transcript=None, review=None, root_id, generated_at,
                                    _restore_from=None):
        import documents
        _uuid(root_id)
        spec = {"root_id": root_id, "generated_at": generated_at}
        params = {"recording_id": recording_id, "expected_versions": expected_versions,
                  "context": context, "transcript": transcript, "review": review,
                  "restore_from": _restore_from, "document_spec": spec}
        digest = _hash(params)
        replay = self._replay(operation_id, "change", digest)
        if replay is not None:
            return self._document_result(operation_id, replay)
        old_t = self.get_revision(recording_id, "transcript", expected_versions["source_revision"])
        old_r = self.get_revision(recording_id, "review", expected_versions["review_revision"])
        if old_r["source_revision"] != old_t["revision"]:
            raise Conflict("snapshot tuple is incompatible")
        if transcript is not None and review is None:
            raise ValueError("source change requires review")
        if transcript is None and review is None:
            raise ValueError("no change")
        next_t = transcript if transcript is not None else old_t["payload"]
        next_r = review if review is not None else old_r["payload"]
        source_rev = old_t["revision"] + (transcript is not None)
        review_rev = old_r["revision"] + (review is not None)
        source_ctx = context if transcript is not None else old_t["context"]
        review_ctx = context if review is not None else old_r["context"]
        origin = (self.get_revision(recording_id, "transcript", _restore_from[0])
                  if _restore_from else old_t)
        while origin["restored_from"] is not None:
            origin = self.get_revision(recording_id, "transcript", origin["restored_from"])
        content_ctx = context if transcript is not None and _restore_from is None else origin["context"]
        revision = self.db.execute("SELECT COALESCE(MAX(document_revision),0)+1 "
                                   "FROM document_versions WHERE recording_id=?", (recording_id,)).fetchone()[0]
        annotation = (_loads(self.get_document(recording_id, revision - 1)["json"])["annotation"]
                      if revision > 1 else None)
        document_paths = (documents.current_paths_from_document(
            self.get_document(recording_id, revision - 1), recording_id) if revision > 1 else None)
        prepared = documents.prepare(recording_id, source_rev, review_rev, revision,
                                     next_t, next_r, source_ctx, review_ctx, generated_at,
                                     source_content_context=content_ctx, annotation=annotation,
                                     document_paths=document_paths)
        with self._transaction():
            replay = self._replay(operation_id, "change", digest)
            if replay is not None:
                return self._document_result(operation_id, replay)
            if annotation is not None:
                self._check_context_refs(annotation["context"], require_active=False)
                self.get_revision(recording_id, "transcript", annotation["source_revision"])
            result = self.commit_change(recording_id, expected_versions, operation_id, context,
                                        transcript=transcript, review=review, _restore_from=_restore_from,
                                        _document_spec=spec)
            self._insert_prepared_document(recording_id, revision, source_rev, review_rev,
                                           prepared, root_id, operation_id)
            return self._document_result(operation_id, result)

    def restore_with_document(self, recording_id, source_revision, review_revision, expected_versions,
                              operation_id, context, *, root_id, generated_at):
        old_t = self.get_revision(recording_id, "transcript", source_revision)
        old_r = self.get_revision(recording_id, "review", review_revision)
        if old_r["source_revision"] != source_revision:
            raise Conflict("restore tuple is incompatible")
        return self.commit_change_with_document(
            recording_id, expected_versions, operation_id, context, transcript=old_t["payload"],
            review=old_r["payload"], root_id=root_id, generated_at=generated_at,
            _restore_from=(source_revision, review_revision))

    def commit_player_event(self, recording_id, expected_versions, operation_id, context, event_type):
        """Прототип П1: событие плеера списком в новой ревизии разбора (прежний формат).

        Рабочий код его не вызывает: события карточек пишет append_review_event в таблицу событий (Х1).
        Остаётся для испытаний, которым нужна ревизия разбора без изменения содержания."""
        _uuid(recording_id)
        _uuid(operation_id)
        _context(context)
        if event_type not in ("card_visible", "audio_play", "audio_ended", "heartbeat"):
            raise ValueError("unsupported player event")
        if (not isinstance(expected_versions, dict) or
                set(expected_versions) != {"source_revision", "review_revision"} or
                any(type(v) is not int or v < 1 for v in expected_versions.values())):
            raise ValueError("expected source and review revisions required")
        digest = _hash({"recording_id": recording_id, "expected_versions": expected_versions,
                        "context": context, "event_type": event_type})
        with self._transaction():
            replay = self._replay(operation_id, "player_event", digest)
            if replay is not None:
                return replay
            self._check_context_refs(context)
            self._check_document_conflict(recording_id)
            row = self.db.execute("SELECT source_revision, review_revision FROM recordings "
                                  "WHERE recording_id=?", (recording_id,)).fetchone()
            if row is None:
                raise KeyError(recording_id)
            if dict(row) != expected_versions:
                raise Conflict("stale revisions")
            if event_type == "heartbeat":
                result = {"recording_id": recording_id, "source_revision": row["source_revision"],
                          "review_revision": row["review_revision"]}
                self._insert_operation(operation_id, "player_event", digest, context, result)
                return result
            transcript = self.get_revision(recording_id, "transcript", row["source_revision"])["payload"]
            review = self.get_revision(recording_id, "review", row["review_revision"])["payload"]
            events = review.setdefault("interaction_events", [])
            if not isinstance(events, list):
                raise ValueError("invalid interaction events")
            events.append({"event_type": event_type, "context": context})
            if len(events) > MAX_INTERACTION_EVENTS:
                removed = len(events) - MAX_INTERACTION_EVENTS
                del events[:removed]
                review["interaction_events_evicted"] = review.get("interaction_events_evicted", 0) + removed
            validate_pair(transcript, review)
            result = {"recording_id": recording_id, "source_revision": row["source_revision"],
                      "review_revision": row["review_revision"] + 1}
            self._insert_operation(operation_id, "player_event", digest, context, result)
            self.db.execute("INSERT INTO review_revisions VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                            (recording_id, result["review_revision"], row["review_revision"],
                             row["source_revision"], _json(review), _hash(review), operation_id))
            self.db.execute("UPDATE recordings SET review_revision=? WHERE recording_id=?",
                            (result["review_revision"], recording_id))
            return result

    def has_review_events(self):
        """Есть ли в базе таблица событий карточек (Х1)."""
        return _has_table(self.db, "review_events")

    def create_review_events_table(self):
        """Создать пустую таблицу событий в базе до Х1. Ничего не переносит и не переписывает; резерв перед этим
        делает вызывающий (recordings.ensure_review_events). Повторный вызов ничего не меняет."""
        with self._transaction():
            if not self.has_review_events():
                self.db.execute(REVIEW_EVENTS_DDL)

    def append_review_event(self, recording_id, event, operation_id, context, *,
                            expected_versions=None, review=None):
        """Добавить событие карточки в таблицу событий без ревизии разбора.

        review — новый снимок разбора, когда событие меняет прослушанные отрезки (heard): тогда в той же
        транзакции пишется ревизия разбора (как commit_change, с проверкой expected_versions)."""
        _uuid(recording_id)
        _uuid(operation_id)
        _context(context)
        if not isinstance(event, dict):
            raise ValueError("review event must be an object")
        encoded = _json(event)
        if len(encoded.encode("utf-8")) > MAX_REVIEW_EVENT_BYTES:
            raise ValueError("review event is too large")
        if review is not None and expected_versions is None:
            raise ValueError("expected source and review revisions required")
        digest = _hash({"recording_id": recording_id, "event": event, "context": context,
                        "expected_versions": expected_versions, "review": review})
        with self._transaction():
            if not self.has_review_events():
                raise ReviewEventsMissing("review events table missing")
            replay = self._replay(operation_id, "review_event", digest)
            if replay is not None:
                return replay
            self._check_context_refs(context)
            if review is not None:
                # ревизия разбора — своей операцией, ключ выводится из ключа события: повтор события её не удвоит
                self.commit_change(recording_id, expected_versions,
                                   str(uuid.uuid5(uuid.UUID(operation_id), "review-change")), context, review=review)
            row = self.db.execute("SELECT review_revision FROM recordings WHERE recording_id=?",
                                  (recording_id,)).fetchone()
            if row is None:
                raise KeyError(recording_id)
            seq = self.db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM review_events WHERE recording_id=?",
                                  (recording_id,)).fetchone()[0]
            result = {"recording_id": recording_id, "seq": seq, "review_revision": row["review_revision"]}
            self._insert_operation(operation_id, "review_event", digest, context, result)
            self.db.execute("INSERT INTO review_events VALUES (?, ?, ?, ?, 'page', ?, ?)",
                            (recording_id, seq, encoded, row["review_revision"], operation_id, _now()))
            return result

    def review_events(self, recording_id):
        """События карточек записи из таблицы, по порядку записи; в базе до Х1 — пустой список."""
        if not self.has_review_events():
            return []
        return [_loads(row["event_json"]) for row in self.db.execute(
            "SELECT event_json FROM review_events WHERE recording_id=? ORDER BY seq", (recording_id,))]

    def has_hypotheses(self):
        """Есть ли в базе таблица гипотез моделей (Г-1)."""
        return _has_table(self.db, "hypotheses")

    def create_hypotheses_table(self):
        """Создать пустую таблицу гипотез в базе до Г-1. Ничего не переносит и не переписывает; резерв перед этим
        делает вызывающий (recordings.ensure_hypotheses). Повторный вызов ничего не меняет."""
        with self._transaction():
            if not self.has_hypotheses():
                self.db.execute(HYPOTHESES_DDL)
                self.db.execute(HYPOTHESES_INDEX_DDL)

    def add_hypotheses(self, run_id, recording_id, sound_id, record, *, job_id=None, fmt="hyp-2"):
        """Записать гипотезы одного прогона расшифровки записи. Один раз: повтор run_id — Conflict.
        -> {"run_id", "recording_id", "size_bytes"}."""
        if not isinstance(run_id, str) or not HYPOTHESIS_RUN_RE.match(run_id):
            raise ValueError("invalid hypotheses run_id")
        _uuid(recording_id)
        if not isinstance(sound_id, str) or not HYPOTHESIS_SOUND_RE.match(sound_id):
            raise ValueError("invalid hypotheses sound_id")
        if not isinstance(record, dict):
            raise ValueError("hypotheses record must be an object")
        if job_id is not None and not isinstance(job_id, str):
            raise ValueError("job_id must be a string")
        # как прежний файл hyp-2: без сортировки ключей; NaN и бесконечность не запрещены
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        size = len(encoded.encode("utf-8"))
        with self._transaction():
            if not self.has_hypotheses():
                raise StorageError("hypotheses table missing")
            if self.db.execute("SELECT 1 FROM recordings WHERE recording_id=?", (recording_id,)).fetchone() is None:
                raise KeyError(recording_id)
            if self.db.execute("SELECT 1 FROM hypotheses WHERE run_id=?", (run_id,)).fetchone():
                raise Conflict("hypotheses run already stored")
            self.db.execute("INSERT INTO hypotheses VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (run_id, recording_id, sound_id, job_id, str(fmt), encoded, size, _now()))
        return {"run_id": run_id, "recording_id": recording_id, "size_bytes": size}

    @staticmethod
    def _hypotheses_where(recording_id=None, sound_id=None, created_before=None):
        clauses, params = [], []
        if recording_id is not None:
            _uuid(recording_id)
            clauses.append("recording_id=?")
            params.append(recording_id)
        if sound_id is not None:
            if not isinstance(sound_id, str) or not HYPOTHESIS_SOUND_RE.match(sound_id):
                raise ValueError("invalid hypotheses sound_id")
            clauses.append("sound_id=?")
            params.append(sound_id)
        if created_before is not None:
            clauses.append("created_at<?")
            params.append(created_before.astimezone(timezone.utc).isoformat())
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def hypotheses_summary(self, recording_id=None, sound_id=None, created_before=None):
        """Сколько гипотез подходит под отбор: {"runs", "recordings", "bytes"}. Та же выборка, что удаляет
        delete_hypotheses: человек видит ровно то, что удалится. Без таблицы — нули."""
        where, params = self._hypotheses_where(recording_id, sound_id, created_before)
        if not self.has_hypotheses():
            return {"runs": 0, "recordings": 0, "bytes": 0}
        row = self.db.execute("SELECT count(*), count(DISTINCT recording_id), COALESCE(sum(size_bytes), 0) "
                              "FROM hypotheses" + where, params).fetchone()
        return {"runs": row[0], "recordings": row[1], "bytes": row[2]}

    def delete_hypotheses(self, recording_id=None, sound_id=None, created_before=None, with_recordings=False):
        """Удалить гипотезы по отбору (все, записи, звука, созданные раньше created_before) сразу и навсегда.
        -> сводка удалённого, как hypotheses_summary до удаления; with_recordings=True — (сводка, перечень записей
        как hypotheses_recordings до удаления, в той же транзакции)."""
        where, params = self._hypotheses_where(recording_id, sound_id, created_before)
        with self._transaction():
            summary = self.hypotheses_summary(recording_id, sound_id, created_before)
            listed = self.hypotheses_recordings(recording_id, sound_id, created_before) if with_recordings else None
            if summary["runs"]:
                self.db.execute("DELETE FROM hypotheses" + where, params)
        return (summary, listed) if with_recordings else summary

    _RECORDING_LABEL_SQL = (
        "r.location, json_extract(t.payload, '$.title') AS title, json_extract(t.payload, '$.file') AS file, "
        "json_extract(t.payload, '$.recorded_at') AS recorded_at, "
        "(SELECT o.created_at FROM transcript_revisions f JOIN operations o USING(operation_id) "
        " WHERE f.recording_id=r.recording_id ORDER BY f.revision LIMIT 1) AS processed_at ")
    _RECORDING_LABEL_JOIN = (
        "LEFT JOIN transcript_revisions t ON t.recording_id=r.recording_id AND t.revision=r.source_revision ")

    @staticmethod
    def _recording_label(row):
        source = row["file"] if isinstance(row["file"], str) and row["file"] else row["location"]
        title = row["title"] if isinstance(row["title"], str) and row["title"].strip() else None
        return {"title": title,
                "file_name": re.split(r"[\\/]", source)[-1] if isinstance(source, str) else None,
                "recorded_at": row["recorded_at"] if isinstance(row["recorded_at"], str) else None,
                "processed_at": row["processed_at"]}

    def recording_label(self, recording_id):
        """Подпись записи для человека: {"title", "file_name", "recorded_at", "processed_at"} или None, если записи
        нет. file_name — только имя файла аудио, без папки; processed_at — время первой ревизии текста. Текст
        стенограммы и пути не отдаются."""
        _uuid(recording_id)
        row = self.db.execute("SELECT " + self._RECORDING_LABEL_SQL + "FROM recordings r " + self._RECORDING_LABEL_JOIN
                              + "WHERE r.recording_id=?", (recording_id,)).fetchone()
        return self._recording_label(row) if row is not None else None

    def hypotheses_recordings(self, recording_id=None, sound_id=None, created_before=None):
        """Записи, у которых есть гипотезы под отбор (та же выборка, что hypotheses_summary), с подписью
        recording_label: [{"recording_id", "runs", "bytes", "title", "file_name", "recorded_at", "processed_at"}]
        по времени обработки. Без таблицы — []."""
        where, params = self._hypotheses_where(recording_id, sound_id, created_before)
        if not self.has_hypotheses():
            return []
        rows = self.db.execute(
            "SELECT h.recording_id, count(*) AS runs, COALESCE(sum(h.size_bytes), 0) AS bytes, "
            + self._RECORDING_LABEL_SQL + "FROM (SELECT * FROM hypotheses" + where + ") h "
            "JOIN recordings r ON r.recording_id=h.recording_id " + self._RECORDING_LABEL_JOIN
            + "GROUP BY h.recording_id ORDER BY processed_at, h.recording_id", params).fetchall()
        return [{"recording_id": row["recording_id"], "runs": row["runs"], "bytes": row["bytes"],
                 **self._recording_label(row)} for row in rows]

    def hypotheses_rows(self):
        """Все прогоны гипотез: [{"run_id", "recording_id", "sound_id", "record_json"}] по run_id (лаборатория)."""
        if not self.has_hypotheses():
            return []
        return [dict(row) for row in self.db.execute(
            "SELECT run_id, recording_id, sound_id, record_json FROM hypotheses ORDER BY run_id")]

    def save_conspect(self, recording_id, conspect_id, name, source_revision, expected_revision,
                      payload, json_bytes, md_bytes, json_relative, md_relative, root_id,
                      operation_id, context):
        """Сохранить ревизию конспекта и точные экспортные bytes одной транзакцией."""
        import documents
        _uuid(recording_id)
        _uuid(conspect_id)
        _uuid(root_id)
        _uuid(operation_id)
        _context(context)
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("invalid conspect name")
        if type(source_revision) is not int or source_revision < 1:
            raise ValueError("invalid source revision")
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 1):
            raise ValueError("invalid expected conspect revision")
        if not isinstance(payload, dict):
            raise ValueError("conspect payload must be an object")
        _json(payload)
        if not isinstance(json_bytes, bytes) or not isinstance(md_bytes, bytes):
            raise ValueError("conspect exports must be bytes")
        try:
            export_payload = json.loads(json_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("invalid conspect JSON export") from exc
        if export_payload != payload:
            raise ValueError("conspect JSON export does not match payload")
        json_parts = documents._relative_parts(json_relative)
        md_parts = documents._relative_parts(md_relative)
        json_relative, md_relative = Path(*json_parts).as_posix(), Path(*md_parts).as_posix()
        if json_relative == md_relative:
            raise ValueError("conspect export paths must be distinct")
        params = {"recording_id": recording_id, "conspect_id": conspect_id, "name": name,
                  "source_revision": source_revision, "expected_revision": expected_revision,
                  "payload": payload, "json_hash": _hash_bytes(json_bytes), "md_hash": _hash_bytes(md_bytes),
                  "json_relative": json_relative, "md_relative": md_relative, "root_id": root_id,
                  "context": context}
        digest = _hash(params)
        with self._transaction():
            replay = self._replay(operation_id, "conspect", digest)
            if replay is not None:
                return replay
            self._check_context_refs(context)
            self._check_document_conflict(recording_id)
            self._check_snapshot_refs(payload)
            source = self.db.execute("SELECT source_revision FROM recordings WHERE recording_id=?",
                                     (recording_id,)).fetchone()
            if source is None:
                raise KeyError(recording_id)
            if source["source_revision"] != source_revision:
                raise Conflict("stale conspect source revision")
            root = self.db.execute("SELECT path FROM allowed_roots WHERE root_id=?", (root_id,)).fetchone()
            if root is None:
                raise ValueError("unknown conspect root")
            existing = self.db.execute("SELECT * FROM conspects WHERE conspect_id=?",
                                       (conspect_id,)).fetchone()
            by_name = self.db.execute("SELECT conspect_id FROM conspects WHERE recording_id=? AND name=?",
                                      (recording_id, name)).fetchone()
            if existing is None:
                if by_name is not None or expected_revision is not None:
                    raise Conflict("conspect identity or expected revision mismatch")
                revision, parent = 1, None
            else:
                if existing["recording_id"] != recording_id or existing["name"] != name:
                    raise Conflict("conspect identity mismatch")
                if self.db.execute("SELECT 1 FROM conspect_outbox "
                                   "WHERE conspect_id=? AND state='conflict'", (conspect_id,)).fetchone():
                    raise Conflict("unresolved conspect conflict")
                if by_name is None or by_name["conspect_id"] != conspect_id:
                    raise Conflict("conspect name collision")
                if existing["current_revision"] != expected_revision:
                    raise Conflict("stale conspect revision")
                if existing["root_id"] != root_id or existing["json_relative"] != json_relative \
                        or existing["md_relative"] != md_relative:
                    raise Conflict("conspect targets are immutable")
                revision, parent = expected_revision + 1, expected_revision
            json_path = os.path.normcase(os.path.abspath(os.path.join(root["path"], json_relative)))
            md_path = os.path.normcase(os.path.abspath(os.path.join(root["path"], md_relative)))
            for kind, path in (("json", json_path), ("md", md_path)):
                if self.db.execute(
                        "SELECT 1 FROM file_locations WHERE canonical_path=? UNION ALL "
                        "SELECT 1 FROM conspect_file_locations WHERE canonical_path=? "
                        "AND NOT (conspect_id=? AND kind=?)",
                        (path, path, conspect_id, kind)).fetchone():
                    raise Conflict("conspect target collision")
            result = {"recording_id": recording_id, "conspect_id": conspect_id,
                      "conspect_revision": revision, "source_revision": source_revision}
            self._insert_operation(operation_id, "conspect", digest, context, result)
            if existing is None:
                self.db.execute("INSERT INTO conspects VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (conspect_id, recording_id, name, root_id, json_relative, md_relative, revision))
            else:
                self.db.execute("UPDATE conspects SET current_revision=? WHERE conspect_id=?",
                                (revision, conspect_id))
            self.db.execute(
                "INSERT INTO conspect_revisions(conspect_id,recording_id,revision,parent_revision,"
                "source_revision,payload,payload_hash,json,md,json_hash,md_hash,operation_id,context_json,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (conspect_id, recording_id, revision, parent, source_revision, _json(payload), _hash(payload),
                 json_bytes, md_bytes, _hash_bytes(json_bytes), _hash_bytes(md_bytes), operation_id,
                 _json(context), _now()))
            previous = {}
            if parent is not None:
                prior = self.get_conspect(conspect_id, parent)
                previous = {"json": prior["json_hash"], "md": prior["md_hash"]}
            for kind, relative, canonical in (("json", json_relative, json_path), ("md", md_relative, md_path)):
                self.db.execute(
                    "INSERT OR IGNORE INTO conspect_file_locations(root_id,relative_path,canonical_path,"
                    "conspect_id,kind) VALUES (?,?,?,?,?)",
                    (root_id, relative, canonical, conspect_id, kind))
                registered = self.db.execute(
                    "SELECT conspect_id,kind FROM conspect_file_locations WHERE root_id=? AND relative_path=?",
                    (root_id, relative)).fetchone()
                if registered is None or registered["conspect_id"] != conspect_id or registered["kind"] != kind:
                    raise Conflict("conspect target collision")
            self.db.execute("INSERT INTO conspect_outbox(conspect_id,revision,root_id,state,"
                            "expected_previous_hashes) VALUES (?,?,?,'pending',?)",
                            (conspect_id, revision, root_id, _json(previous)))
            return result

    def get_conspect(self, conspect_id, revision=None):
        _uuid(conspect_id)
        head = self.db.execute("SELECT * FROM conspects WHERE conspect_id=?", (conspect_id,)).fetchone()
        if head is None:
            raise KeyError(conspect_id)
        revision = head["current_revision"] if revision is None else revision
        row = self.db.execute("SELECT * FROM conspect_revisions WHERE conspect_id=? AND revision=?",
                              (conspect_id, revision)).fetchone()
        if row is None:
            raise KeyError((conspect_id, revision))
        result = dict(row)
        if result["recording_id"] != head["recording_id"]:
            raise StorageError("conspect recording mismatch")
        result["payload"] = _loads(result["payload"])
        if _hash(result["payload"]) != result["payload_hash"]:
            raise StorageError("conspect payload hash mismatch")
        if (_hash_bytes(result["json"]) != result["json_hash"] or
                _hash_bytes(result["md"]) != result["md_hash"]):
            raise StorageError("conspect export hash mismatch")
        if _loads(result["json"]) != result["payload"]:
            raise StorageError("conspect JSON and payload mismatch")
        result["context"] = _loads(result["context_json"])
        _context(result["context"])
        operation = self.db.execute("SELECT context_json FROM operations WHERE operation_id=?",
                                    (result["operation_id"],)).fetchone()
        if operation is None or _loads(operation["context_json"]) != result["context"]:
            raise StorageError("conspect context mismatch")
        return {**dict(head), **result}

    def list_conspects(self, recording_id):
        _uuid(recording_id)
        return [self.get_conspect(row["conspect_id"]) for row in self.db.execute(
            "SELECT conspect_id FROM conspects WHERE recording_id=? ORDER BY name", (recording_id,))]

    def save_document(self, recording_id, expected_versions, generated_at, operation_id, context,
                      *, annotation=KEEP_ANNOTATION, root_id=None):
        """Сохранить точные байты документа; публикация файлов здесь не выполняется."""
        import documents
        _uuid(recording_id)
        _uuid(operation_id)
        _context(context)
        if (not isinstance(expected_versions, dict) or
                set(expected_versions) != {"source_revision", "review_revision"} or
                any(type(v) is not int or v < 1 for v in expected_versions.values())):
            raise ValueError("expected source and review revisions required")
        if not isinstance(generated_at, str) or not generated_at:
            raise ValueError("generated_at required")
        if root_id is not None:
            _uuid(root_id)
        annotation_param = {"mode": "inherit"} if annotation is KEEP_ANNOTATION else {
            "mode": "replace", "value": annotation}
        digest = _hash({"recording_id": recording_id, "expected_versions": expected_versions,
                        "generated_at": generated_at, "context": context, "annotation": annotation_param,
                        "root_id": root_id})
        replay = self._replay(operation_id, "document", digest)
        if replay is not None:
            return replay
        source = self.get_revision(recording_id, "transcript", expected_versions["source_revision"])
        review = self.get_revision(recording_id, "review", expected_versions["review_revision"])
        if review["source_revision"] != source["revision"]:
            raise Conflict("document snapshot tuple is incompatible")
        latest = self.db.execute("SELECT COALESCE(MAX(document_revision), 0) FROM document_versions "
                                 "WHERE recording_id=?", (recording_id,)).fetchone()[0]
        revision = latest + 1
        if annotation is KEEP_ANNOTATION:
            annotation = (_loads(self.get_document(recording_id, latest)["json"])["annotation"]
                          if latest else None)
        origin = source
        while origin["restored_from"] is not None:
            origin = self.get_revision(recording_id, "transcript", origin["restored_from"])
        prepared = documents.prepare(recording_id, source["revision"], review["revision"], revision,
                                     source["payload"], review["payload"], source["context"],
                                     review["context"], generated_at,
                                     source_content_context=origin["context"], annotation=annotation,
                                     document_paths=(documents.current_paths_from_document(
                                         self.get_document(recording_id, latest), recording_id)
                                         if latest else None))
        with self._transaction():
            replay = self._replay(operation_id, "document", digest)
            if replay is not None:
                return replay
            self._check_context_refs(context)
            self._check_document_conflict(recording_id)
            if annotation is not None:
                self._check_context_refs(annotation["context"], require_active=False)
                self.get_revision(recording_id, "transcript", annotation["source_revision"])
                self.get_revision(recording_id, "transcript", annotation["source_revision"])
            head = self.db.execute("SELECT source_revision, review_revision FROM recordings "
                                   "WHERE recording_id=?", (recording_id,)).fetchone()
            if head is None or dict(head) != expected_versions:
                raise Conflict("stale document snapshot")
            latest = self.db.execute("SELECT COALESCE(MAX(document_revision), 0) FROM document_versions "
                                     "WHERE recording_id=?", (recording_id,)).fetchone()[0]
            if latest + 1 != revision:
                raise Conflict("document revision changed")
            result = {"recording_id": recording_id, "document_revision": revision}
            self._insert_operation(operation_id, "document", digest, context, result)
            self.db.execute("INSERT INTO document_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (recording_id, revision, source["revision"], review["revision"],
                             prepared["md"], prepared["json"], prepared["manifest"],
                             prepared["archive_manifest"], _json(prepared["hashes"]), operation_id))
            if root_id is not None:
                if revision > 1:
                    previous = self.db.execute(
                        "SELECT root_id FROM document_outbox WHERE recording_id=? AND document_revision=?",
                        (recording_id, revision - 1)).fetchone()
                    if previous is None or previous["root_id"] != root_id:
                        raise Conflict("document generations require a single registered root")
                self._enqueue_document(recording_id, revision, root_id, prepared["hashes"], prepared["paths"])
            return result

    def get_document(self, recording_id, document_revision):
        """Прочитать сохранённые исходные байты без нового рендера."""
        _uuid(recording_id)
        if type(document_revision) is not int or document_revision < 1:
            raise ValueError("invalid document revision")
        row = self.db.execute("SELECT * FROM document_versions WHERE recording_id=? AND document_revision=?",
                              (recording_id, document_revision)).fetchone()
        if row is None:
            raise KeyError((recording_id, document_revision))
        result = dict(row)
        result["hashes"] = _loads(result["hashes"])
        for field in ("md", "json", "manifest", "archive_manifest"):
            if hashlib.sha256(result[field]).hexdigest() != result["hashes"][field]:
                raise StorageError("document hash mismatch")
        return result

    def get_revision(self, recording_id, kind, revision):
        _uuid(recording_id)
        if kind not in ("transcript", "review") or type(revision) is not int or revision < 1:
            raise ValueError("invalid revision")
        table = "transcript_revisions" if kind == "transcript" else "review_revisions"
        row = self.db.execute(f"SELECT r.*, o.context_json FROM {table} r "
                              "JOIN operations o ON o.operation_id=r.operation_id "
                              "WHERE r.recording_id=? AND r.revision=?", (recording_id, revision)).fetchone()
        if row is None:
            raise KeyError((recording_id, kind, revision))
        result = dict(row)
        result["payload"] = _loads(result["payload"])
        if _hash(result["payload"]) != result["payload_hash"]:
            raise StorageError("snapshot hash mismatch")
        result["context"] = _loads(result.pop("context_json"))
        return result

    def get_current(self, recording_id):
        """Прочитать согласованную пару текущих снимков в одной read-транзакции."""
        _uuid(recording_id)
        if self.db.in_transaction:
            raise Conflict("read requires its own transaction")
        try:
            self.db.execute("BEGIN")
        except sqlite3.OperationalError as exc:
            if _is_busy(exc):
                raise Busy("database is busy") from exc
            raise StorageError("cannot begin read transaction") from exc
        try:
            head = self.db.execute("SELECT source_revision, review_revision FROM recordings "
                                   "WHERE recording_id=?", (recording_id,)).fetchone()
            if head is None:
                raise KeyError(recording_id)
            transcript = self.get_revision(recording_id, "transcript", head["source_revision"])
            review = self.get_revision(recording_id, "review", head["review_revision"])
            if review["source_revision"] != transcript["revision"]:
                raise StorageError("incompatible current snapshots")
            try:
                validate_pair(transcript["payload"], review["payload"])
            except ValueError as exc:
                raise StorageError("invalid stored snapshots") from exc
            self.db.execute("COMMIT")
            return {"transcript": transcript, "review": review}
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def list_revisions(self, recording_id, kind):
        _uuid(recording_id)
        if kind not in ("transcript", "review"):
            raise ValueError("invalid revision kind")
        table = "transcript_revisions" if kind == "transcript" else "review_revisions"
        return [r[0] for r in self.db.execute(
            f"SELECT revision FROM {table} WHERE recording_id=? ORDER BY revision", (recording_id,))]
