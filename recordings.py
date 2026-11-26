"""Runtime-фасад и адаптер между форматом приложения и SQLite-снимками.

Создание/открытие БД остаётся явным; этот модуль не выбирает backend сам.
"""

import copy
import contextvars
import errno
import json
import os
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import documents
import storage

_OPERATION_CONTEXT = contextvars.ContextVar("recording_operation_context", default=None)
_ANSWER_DRAFT = contextvars.ContextVar("recording_answer_draft", default=None)
# Runtime writers for SQLite-owned recordings share the facade below; legacy files remain separate.
SQLITE_RUNTIME_READY = True


def answer_draft_for(document_path):
    draft = _ANSWER_DRAFT.get()
    target = os.path.normcase(os.path.abspath(document_path))
    return draft if draft is not None and draft["path"] == target else None


@contextmanager
def answer_draft(document_path, data, state, context):
    if _ANSWER_DRAFT.get() is not None:
        raise RuntimeError("nested answer draft is not supported")
    draft = {"path": os.path.normcase(os.path.abspath(document_path)),
             "data": copy.deepcopy(data), "state": copy.deepcopy(state), "context": copy.deepcopy(context)}
    token = _ANSWER_DRAFT.set(draft)
    try:
        yield draft
    finally:
        _ANSWER_DRAFT.reset(token)


def merge_answer_draft(data, state):
    """Объединить подготовленную стенограмму и пакет ответов перед одним commit."""
    result = copy.deepcopy(data)
    result["question_version"] = state.get("version")
    manifest = state.get("manifest")
    result["questions"] = copy.deepcopy(manifest.get("items") or []) if manifest else []
    for field in ("answers", "conflicts", "rejected"):
        result[field] = copy.deepcopy(state.get(field) or ({} if field == "answers" else []))
    undo = copy.deepcopy(result.get("undo") or {})
    # прослушанные отрезки записи (раздел 10): состояние без поля прежние не снимает
    heard = state.get("heard") if state.get("heard") is not None else (undo.get("review_pack") or {}).get("heard")
    undo["review_pack"] = {field: copy.deepcopy(state.get(field)) for field in (
        "manifest", "applied_answers", "applied_at", "undo_snapshots",
        "application_events", "application_events_dropped")}
    if state.get("outcome_details"):          # уточнения последнего «Применить» к ответам (К-22)
        undo["review_pack"]["outcome_details"] = copy.deepcopy(state["outcome_details"])
    if heard is not None:
        undo["review_pack"]["heard"] = copy.deepcopy(heard)
    result["undo"] = undo
    # события карточек (Х1) черновик не меняет: прежний список разбора переходит как есть, новые — в таблице
    result["applications"] = copy.deepcopy(state.get("application_events") or [])
    # Как при записи в БД: исчезнувшие слова не остаются в карточках документа (manifest не трогаем).
    _prune_missing_word_ids(result, result)
    return result


def initialize_store(path=None, *, display_name=""):
    """Явно создать runtime-БД и первоначальную локальную identity без миграции."""
    if not isinstance(display_name, str):
        raise ValueError("display name must be text")
    if path is None:
        import paths
        path = paths.STATE_DB
    return storage.initialize(path, initial_display_name=display_name)


def open_store(path=None, readonly=False):
    """Открыть существующую runtime-БД; не создавать, мигрировать или инициализировать.

    readonly=True — соединение только для чтения (storage.open_store_readonly): для тех, кто базу не меняет.
    """
    if path is None:
        import paths
        path = paths.STATE_DB
    return storage.open_store_readonly(path) if readonly else storage.open_store(path)


def document_backend_hint(document_path):
    """`passport.storage_backend` файла стенограммы (`sqlite` — выгрузка записи базы) или None."""
    try:
        payload = json.loads(Path(document_path).read_text(encoding="utf-8"))
        return (payload.get("passport") or {}).get("storage_backend")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return None


def open_for_document(document_path, path=None, readonly=False):
    """Открыть Store только если путь зарегистрирован как текущая цель SQLite-записи."""
    import settings
    sqlite_selected = settings.recording_backend() == "sqlite"
    if path is None:
        import paths
        path = paths.STATE_DB
    hint = document_backend_hint(document_path)
    if not Path(path).is_file():
        if hint == "sqlite" or sqlite_selected:
            raise storage.StorageError("SQLite database is missing for this recording")
        return None, None
    try:
        store = open_store(path, readonly=readonly)
    except storage.StorageError:
        if hint == "sqlite" or sqlite_selected:
            raise
        return None, None
    try:
        binding = resolve_document(store, document_path)
        if binding is None:
            message = "SQLite document is not registered in the active store"
            if sqlite_selected and hint != "sqlite":
                # старая стенограмма после переключения на базу (М-2): подсказать перенесённую запись или
                # причину, почему её нет; запись по старому пути не подменяется — от пути зависят папка
                # разбора и файлы рядом с аудио
                try:
                    message += "; " + find_migrated(store, document_path)["hint"]
                except (OSError, ValueError, KeyError, sqlite3.Error, storage.StorageError):
                    pass
            store.close()
            if hint == "sqlite" or sqlite_selected:
                raise storage.Conflict(message)
            return None, None
        store.current_identity()  # fail closed for inactive/ambiguous restored stores
        return store, binding
    except BaseException:
        store.close()
        raise


def normalize_location(location):
    if not isinstance(location, (str, os.PathLike)) or not str(location):
        raise ValueError("recording location required")
    return os.path.normcase(os.path.abspath(Path(location).expanduser()))


def find_recordings(store, source_path):
    """Найти все tag одного исходника независимо от job."""
    location = normalize_location(source_path)
    return [dict(row) for row in store.db.execute(
        "SELECT recording_id, location, source_revision, review_revision FROM recordings "
        "WHERE location=? ORDER BY recording_id", (location,))]


# ---------- записи, перенесённые из старых файлов (П7, `tools/import_legacy.py`) ----------
# Связь записи со старой стенограммой — только `quality.migration` первой ревизии стенограммы: путь и SHA-256
# старого файла на момент переноса. Функции ниже только читают базу и старый файл.
def migration_info(store, recording_id):
    """`quality.migration` первой ревизии стенограммы записи или {} (запись не из переноса)."""
    try:
        first = store.get_revision(recording_id, "transcript", 1)["payload"]
    except (KeyError, ValueError, storage.StorageError):
        return {}
    migration = (first.get("quality") or {}).get("migration")
    return migration if isinstance(migration, dict) else {}


def migration_link(store, recording_id):
    """Сведения о старом файле записи (`path`, `sha256`, `size`) или {}."""
    link = migration_info(store, recording_id).get("source") or {}
    return link if isinstance(link, dict) and isinstance(link.get("path"), str) and link["path"] else {}


def same_location(a, b):
    """Пути равны без учёта регистра (в базе `location` хранится через normcase, в старых задачах — как есть)."""
    return os.path.normcase(str(a)).lower() == os.path.normcase(str(b)).lower()


def migrated_copies(store, legacy_path):
    """Записи базы, перенесённые из этого старого файла: [(recording_id, migration.source)]."""
    target = Path(legacy_path).resolve()
    out = []
    for (recording_id,) in store.db.execute("SELECT recording_id FROM recordings ORDER BY recording_id").fetchall():
        link = migration_link(store, recording_id)
        if link and same_location(Path(link["path"]).resolve(), target):
            out.append((recording_id, link))
    return out


def _legacy_file_facts(legacy_path):
    """(SHA-256 файла или None, если его нет; recording_id из файла или None)."""
    import hashlib
    p = Path(legacy_path)
    if not p.is_file():
        return None, None
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        old = json.loads(p.read_bytes().decode("utf-8-sig")).get("recording_id")
    except (OSError, UnicodeDecodeError, ValueError, AttributeError):
        old = None
    return digest.hexdigest(), old if isinstance(old, str) and old else None


RECORDING_REF = "recording:"


def _refs(copies):
    return ", ".join(RECORDING_REF + rid for rid, _link in copies)


_MIGRATION_HINTS = {
    "found": "эта стенограмма перенесена в базу: {refs} — используйте его",
    "not_migrated": "эта стенограмма не перенесена в базу; перенесите её (tools/import_legacy.py, "
                    "docs/legacy-migration.md) или верните режим legacy_files",
    "changed": "старый файл изменился после переноса в базу; запись в базе — {refs}; выберите её явно",
    "file_missing": "старого файла нет на диске; записи в базе по этому пути — {refs}; выберите запись явно",
    # файла нет и по его пути ничего не переносили: переносить нечего, но запись могли перенести из другой папки
    "not_found": "старого файла нет на диске и записей по этому пути в базе нет; если стенограмму перенесли "
                 "из другой папки, найдите запись в списке записей и укажите recording:<id>",
    "ambiguous": "по этому файлу в базе несколько записей — {refs}; выберите запись явно",
}


def find_migrated(store, legacy_path):
    """Единственная запись базы, перенесённая из этого старого файла, — только чтение.

    Сначала прежний `recording_id` из файла (перенос его сохраняет), иначе перебор записей по
    `quality.migration.source.path`. Подходит только запись с тем же SHA-256, что у файла сейчас.
    -> {"state": found | not_migrated | not_found | changed | file_missing | ambiguous, "recording_id" (для found),
        "candidates": [recording_id…] по пути, "hint": текст для человека}.
    """
    sha, old_id = _legacy_file_facts(legacy_path)
    copies = None
    if sha is not None and old_id:
        try:
            candidate = str(uuid.UUID(old_id))
        except ValueError:
            candidate = None
        link = migration_link(store, candidate) if candidate else {}
        if link and link.get("sha256") == sha and same_location(Path(link["path"]).resolve(),
                                                                Path(legacy_path).resolve()):
            copies = [(candidate, link)]
    if copies is None:
        copies = migrated_copies(store, legacy_path)
    if not copies:
        state, matching = ("not_found" if sha is None else "not_migrated"), []
    elif sha is None:
        state, matching = "file_missing", copies
    else:
        matching = [c for c in copies if c[1].get("sha256") == sha]
        state = {0: "changed", 1: "found"}.get(len(matching), "ambiguous")
    shown = matching if state in ("found", "ambiguous") else copies
    result = {"state": state, "candidates": [rid for rid, _link in copies],
              "hint": _MIGRATION_HINTS[state].format(refs=_refs(shown))}
    if state == "found":
        result["recording_id"] = matching[0][0]
        others = [c for c in copies if c[0] != result["recording_id"]]
        if others:
            result["hint"] += f"; другие перенесённые копии этого файла — {_refs(others)}"
    return result


def main_variant_of(store, recording_id):
    """ID основного варианта той же записи (то же аудио, `migration.variant.main`) или None."""
    row = store.db.execute("SELECT location FROM recordings WHERE recording_id=?", (recording_id,)).fetchone()
    if row is None:
        return None
    mains = [rid for (rid,) in store.db.execute(
        "SELECT recording_id FROM recordings WHERE location=? ORDER BY recording_id", (row[0],)).fetchall()
        if (migration_info(store, rid).get("variant") or {}).get("main")]
    return mains[0] if len(mains) == 1 else None


def document_binding(store, recording_id):
    """Вернуть зарегистрированные корень и текущие sidecar-цели записи."""
    row = store.db.execute(
        "SELECT o.root_id, r.path AS root_path, o.document_revision "
        "FROM document_outbox o JOIN allowed_roots r USING(root_id) "
        "WHERE o.recording_id=? ORDER BY o.document_revision DESC LIMIT 1",
        (recording_id,)).fetchone()
    if row is None:
        raise KeyError(recording_id)
    document = store.get_document(recording_id, row["document_revision"])
    return {"root_id": row["root_id"], "root": Path(row["root_path"]),
            "document_revision": row["document_revision"],
            "paths": documents.current_paths_from_document(document, recording_id)}


def recording_head(store, recording_id):
    """Текущие ревизии записи: строка (source_revision, review_revision) или None, если записи нет."""
    return store.db.execute("SELECT source_revision, review_revision FROM recordings WHERE recording_id=?",
                            (recording_id,)).fetchone()


def document_sync_state(store, recording_id, document_revision):
    """Состояние формирования файлов этой ревизии документа (очередь outbox): synced, pending…"""
    return store.db.execute(
        "SELECT state FROM document_outbox WHERE recording_id=? AND document_revision=?",
        (recording_id, document_revision)).fetchone()["state"]


def resolve_document(store, document_path):
    """Разрешить только зарегистрированный текущий JSON/MD/manifest, не угадывая по имени."""
    target = os.path.normcase(os.path.abspath(Path(document_path).expanduser()))
    rows = store.db.execute(
        "SELECT recording_id, kind, canonical_path FROM file_locations "
        "WHERE kind IN ('md','json','manifest') AND canonical_path=?",
        (target,)).fetchall()
    if len(rows) != 1:
        return None
    row = rows[0]
    binding = document_binding(store, row["recording_id"])
    expected = os.path.normcase(os.path.abspath(binding["root"] / binding["paths"][row["kind"]]))
    if expected != target:
        return None
    return {"recording_id": row["recording_id"], "kind": row["kind"], **binding}


def operation_context(store, actor_kind):
    """Захватить автора/инициатора из единственной активной локальной identity."""
    identity = store.current_identity()
    return store.capture_context(identity["installation_id"], actor_kind)


@contextmanager
def operation_scope(store, actor_kind):
    """Передать происхождение текущей операции вложенным transcript API."""
    key = os.path.normcase(str(store.path)) if store.path is not None else id(store)
    token = _OPERATION_CONTEXT.set((key, operation_context(store, actor_kind)))
    try:
        yield
    finally:
        _OPERATION_CONTEXT.reset(token)


@contextmanager
def operation_for_document(document_path, actor_kind):
    store, binding = open_for_document(document_path)
    if store is None:
        yield None
        return
    try:
        with operation_scope(store, actor_kind):
            yield binding
    finally:
        store.close()


def current_operation_context(store):
    current = _OPERATION_CONTEXT.get()
    key = os.path.normcase(str(store.path)) if store.path is not None else id(store)
    if current is not None and current[0] == key:
        return current[1]
    return operation_context(store, "unknown")


def context_for_document(document_path):
    store, binding = open_for_document(document_path)
    if store is None:
        return None
    try:
        return current_operation_context(store)
    finally:
        store.close()


def materialize_recording(store, recording_id):
    binding = document_binding(store, recording_id)
    return documents.materialize_pending(store, binding["root_id"], binding["root"], recording_id=recording_id)


def read_review_pack_state(document_path):
    """Прочитать ответы/события review-пакета из SQLite; None означает legacy_files."""
    draft = answer_draft_for(document_path)
    if draft is not None:
        return copy.deepcopy(draft["state"])
    store, binding = open_for_document(document_path)
    if store is None:
        return None
    try:
        current = store.get_current(binding["recording_id"])
        review = current["review"]["payload"]
        pack = (review.get("undo") or {}).get("review_pack") or {}
        applications = review.get("applications") or []
        # уточнения последнего «Применить» к ответам (К-22) — только если есть: прежние записи без поля не меняются
        details = {"outcome_details": copy.deepcopy(pack["outcome_details"])} if pack.get("outcome_details") else {}
        # прослушанные отрезки записи (раздел 10) — только если есть: без поля действует переход 10.6
        if pack.get("heard") is not None:
            details["heard"] = copy.deepcopy(pack["heard"])
        return {
            **details,
            "_expected_versions": {"source_revision": current["transcript"]["revision"],
                                   "review_revision": current["review"]["revision"]},
            "version": review.get("question_version"),
            "manifest": copy.deepcopy(pack.get("manifest")),
            "answers": copy.deepcopy(review.get("answers") or {}),
            "conflicts": copy.deepcopy(review.get("conflicts") or []),
            "rejected": copy.deepcopy(review.get("rejected") or []),
            "applied_answers": copy.deepcopy(pack.get("applied_answers") or {}),
            "applied_at": pack.get("applied_at"),
            "undo_snapshots": copy.deepcopy(pack.get("undo_snapshots") or {}),
            "application_events": copy.deepcopy(pack.get("application_events") or []),
            "application_events_dropped": pack.get("application_events_dropped", 0),
            # события карточек: прежний список разбора (базы до переноса Х1у), потом таблица событий (Х1)
            "interaction_events": (copy.deepcopy(review.get("interaction_events") or []) +
                                   store.review_events(binding["recording_id"])),
            "interaction_events_dropped": review.get("interaction_events_evicted", 0),
        }
    finally:
        store.close()


# Поля состояния пакета, изменение которых не меняет документ: пишутся ревизией разбора без пересборки документа.
_EVENT_FIELDS = frozenset(("application_events", "application_events_dropped", "outcome_details", "heard"))
# События карточек пишет только append_review_event (таблица событий, Х1): запись состояния пакета их не трогает.
_INTERACTION_FIELDS = frozenset(("interaction_events", "interaction_events_dropped"))
REVIEW_EVENTS_NOT_UPGRADED = ("база не обновлена: показы карточек и проигрывания не записываются, отметки "
                              "«прослушано» записываются — не удалось сделать резерв базы перед обновлением")
# Ф204-4: причина отказа резерва — категория без путей и без текста исключения (в тексте OSError есть пути)
BACKUP_NO_SPACE = "нет места на диске"
BACKUP_LOCKED = "файл занят другой программой или нет доступа"
BACKUP_NO_LINKS = "том не поддерживает жёсткие ссылки"
BACKUP_INVALID_COPY = "копия базы не прошла проверку: в базе есть расхождение, резерв без его исправления невозможен"
_BACKUP_ADVICE = {BACKUP_NO_SPACE: "Освободите место на диске, затем закройте и снова откройте страницу карточек"}
_BACKUP_RETRY = "Закройте и снова откройте страницу карточек, чтобы попробовать снова"
_REVIEW_EVENTS_BACKUP_FAILED = {}   # ключ базы -> причина отказа; до конца процесса страницы резерв не пробуется
_REVIEW_EVENTS_BACKUP_DONE = {}     # ключ базы -> готовый резерв (Ф204-3): повтор только создаёт таблицу


def backup_failure_reason(ex):
    """Короткая причина отказа резерва для человека: нет места, файл занят, нет жёстких ссылок, проверка копии."""
    if isinstance(ex, OSError):
        code, win = ex.errno, getattr(ex, "winerror", None)
        if code == errno.ENOSPC or win in (39, 112):
            return BACKUP_NO_SPACE
        if code in (errno.EXDEV, errno.EMLINK, getattr(errno, "ENOTSUP", None), getattr(errno, "EOPNOTSUPP", None)) \
                or win in (1, 50, 1142):
            return BACKUP_NO_LINKS
        if code in (errno.EACCES, errno.EPERM) or win in (5, 32, 33):
            return BACKUP_LOCKED
        return f"ошибка файла ({type(ex).__name__})"
    if isinstance(ex, sqlite3.Error):
        if getattr(ex, "sqlite_errorcode", None) == getattr(sqlite3, "SQLITE_FULL", 13):
            return BACKUP_NO_SPACE
        return f"ошибка SQLite ({type(ex).__name__})"
    if isinstance(ex, storage.StorageError):
        return BACKUP_INVALID_COPY
    return type(ex).__name__


def review_events_warning(reason):
    """Предупреждение страницы, когда резерв перед созданием таблицы событий не удался."""
    return f"{REVIEW_EVENTS_NOT_UPGRADED}. Причина: {reason}. {_BACKUP_ADVICE.get(reason, _BACKUP_RETRY)}"


def _review_from_state(store, binding, state):
    """Снимок разбора из состояния пакета: (expected_versions, review, изменённые поля)."""
    current = store.get_current(binding["recording_id"])
    expected = state.get("_expected_versions")
    actual = {"source_revision": current["transcript"]["revision"],
              "review_revision": current["review"]["revision"]}
    if expected != actual:
        raise storage.Conflict("stale review pack state")
    review = copy.deepcopy(current["review"]["payload"])
    previous_state = {
        "version": review.get("question_version"),
        "manifest": copy.deepcopy((review.get("undo") or {}).get("review_pack", {}).get("manifest")),
        "answers": copy.deepcopy(review.get("answers") or {}),
        "conflicts": copy.deepcopy(review.get("conflicts") or []),
        "rejected": copy.deepcopy(review.get("rejected") or []),
        "applied_answers": copy.deepcopy((review.get("undo") or {}).get(
            "review_pack", {}).get("applied_answers") or {}),
        "applied_at": (review.get("undo") or {}).get("review_pack", {}).get("applied_at"),
        "undo_snapshots": copy.deepcopy((review.get("undo") or {}).get(
            "review_pack", {}).get("undo_snapshots") or {}),
        "application_events": copy.deepcopy((review.get("undo") or {}).get(
            "review_pack", {}).get("application_events") or []),
        "application_events_dropped": (review.get("undo") or {}).get(
            "review_pack", {}).get("application_events_dropped", 0),
        "outcome_details": copy.deepcopy((review.get("undo") or {}).get(
            "review_pack", {}).get("outcome_details")) or None,
        "heard": copy.deepcopy((review.get("undo") or {}).get("review_pack", {}).get("heard")),
    }
    review["question_version"] = state.get("version")
    if "manifest" in state:
        manifest = copy.deepcopy(state["manifest"])
        review["questions"] = copy.deepcopy(manifest.get("items") or []) if manifest else []
    _prune_missing_word_ids(current["transcript"]["payload"], review)
    for field in ("answers", "conflicts", "rejected"):
        if field in state:
            review[field] = copy.deepcopy(state[field])
    undo = copy.deepcopy(review.get("undo") or {})
    pack = copy.deepcopy(undo.get("review_pack") or {})
    pack["manifest"] = copy.deepcopy(state.get("manifest"))
    pack["applied_answers"] = copy.deepcopy(state.get("applied_answers") or {})
    pack["applied_at"] = state.get("applied_at")
    pack["undo_snapshots"] = copy.deepcopy(state.get("undo_snapshots") or {})
    pack["application_events"] = copy.deepcopy(state.get("application_events") or [])
    pack["application_events_dropped"] = state.get("application_events_dropped", 0)
    if state.get("outcome_details"):
        pack["outcome_details"] = copy.deepcopy(state["outcome_details"])
    else:
        pack.pop("outcome_details", None)
    if state.get("heard") is not None:         # состояние без поля прослушанное не снимает
        pack["heard"] = copy.deepcopy(state["heard"])
    undo["review_pack"] = pack
    review["undo"] = undo
    if review.get("applications") is None:
        review["applications"] = []
    if not isinstance(review.get("applications"), list):
        raise ValueError("stored applications must be a list")
    if "application_events" in state:
        review["applications"] = copy.deepcopy(state["application_events"])
    # уточнения к ответам (К-22) документ не меняют: как события, пишутся без пересборки документа
    changed = {key for key in (set(previous_state) | set(state)) - {"_expected_versions"} - _INTERACTION_FIELDS
               if previous_state.get(key) != state.get(key)}
    return expected, review, changed


def write_review_pack_state(document_path, state):
    """Сохранить ответный пакет в review-снимок и обновить пользовательский документ.

    События карточек (interaction_events) здесь не пишутся: их добавляет append_review_event (Х1)."""
    draft = answer_draft_for(document_path)
    if draft is not None:
        draft["state"] = copy.deepcopy(state)
        return {"draft": True}
    store, binding = open_for_document(document_path)
    if store is None:
        return None
    try:
        expected, review, changed = _review_from_state(store, binding, state)
        context = current_operation_context(store)
        if changed and changed <= _EVENT_FIELDS:
            result = store.commit_change(
                binding["recording_id"], expected, str(uuid.uuid4()), context, review=review)
            return result
        documents.guard_before_write(store, binding["recording_id"])
        result = store.commit_change_with_document(
            binding["recording_id"], expected, str(uuid.uuid4()), context, review=review,
            root_id=binding["root_id"], generated_at=datetime.now(timezone.utc).isoformat())
        completed = materialize_recording(store, binding["recording_id"])
        documents.require_written(store, binding["recording_id"], result["document_revision"], completed)
        return result
    finally:
        store.close()


def review_events_backup_dir(store):
    """Папка резервов базы: <папка данных>/db-backups рядом с базой."""
    return Path(store.path).parent / "db-backups"


def ensure_review_events(store):
    """Страховка Х1 для базы, где обновление базы не прошло: резерв, потом пустая таблица событий.

    Ничего не переносит, не переписывает и не удаляет: это делает только обновление базы (Х1у). True — таблица
    есть. False — резерв не удался: таблица не создана, и до перезапуска процесса резерв больше не пробуем
    (иначе полная копия базы на каждое событие); причина — в _REVIEW_EVENTS_BACKUP_FAILED и одной строкой в stderr
    (журнал страницы). Занятость базы (Busy) — не отказ: следующее событие попробует снова. Резерв удался, а таблица
    не создалась — следующее событие только создаёт таблицу, пока файл резерва на месте (Ф204-3)."""
    if store.has_review_events():
        return True
    key = _review_events_key(store)
    if key in _REVIEW_EVENTS_BACKUP_FAILED:
        return False
    done = _REVIEW_EVENTS_BACKUP_DONE.get(key)
    if done is None or not done.is_file():
        try:
            folder = review_events_backup_dir(store)
            folder.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            target = folder / f"state-before-review-events-{stamp}.sqlite3"
            if target.exists():
                target = folder / f"state-before-review-events-{stamp}-{uuid.uuid4().hex[:8]}.sqlite3"
            store.backup(target, allow_damaged_copies=True)
        except storage.Busy:
            raise
        except (OSError, storage.StorageError, sqlite3.Error) as ex:
            reason = backup_failure_reason(ex)
            _REVIEW_EVENTS_BACKUP_FAILED[key] = reason
            # без путей и текста исключения: только категория и тип
            print(f"[review-events] резерв базы перед обновлением не удался: {reason} ({type(ex).__name__})",
                  file=sys.stderr, flush=True)
            return False
        _REVIEW_EVENTS_BACKUP_DONE[key] = target
    store.create_review_events_table()
    return True


def _review_events_key(store):
    """Путь базы и её store_id: восстановленная на том же месте другая база не берёт чужой резерв."""
    row = store.db.execute("SELECT store_id FROM store_meta").fetchone()
    return os.path.normcase(str(store.path)), row[0] if row else None


def append_review_event(document_path, event, state=None):
    """Записать событие карточки в таблицу событий (Х1): без ревизии разбора и без пересборки документа.

    state — состояние пакета, когда событие меняет прослушанные отрезки (heard): тогда в той же транзакции
    пишется ревизия разбора. Если таблицы нет и резерв перед её созданием не удался — событие не записывается,
    прослушанные отрезки записываются ревизией разбора, ответ несёт предупреждение. None — запись не в SQLite."""
    if answer_draft_for(document_path) is not None:
        raise RuntimeError("card events are not written inside an answer draft")
    store, binding = open_for_document(document_path)
    if store is None:
        return None
    try:
        recording_id = binding["recording_id"]
        context = current_operation_context(store)
        expected = review = None
        if state is not None:
            expected, review, changed = _review_from_state(store, binding, state)
            if not changed <= _EVENT_FIELDS:
                raise ValueError("card event may change only heard spans")
            if not changed:
                expected = review = None
        if not ensure_review_events(store):
            if review is not None:
                store.commit_change(recording_id, expected, str(uuid.uuid4()), context, review=review)
            reason = _REVIEW_EVENTS_BACKUP_FAILED.get(_review_events_key(store), "")
            return {"stored": False, "warning": review_events_warning(reason), "heard_saved": review is not None}
        result = store.append_review_event(recording_id, event, str(uuid.uuid4()), context,
                                           expected_versions=expected, review=review)
        return {"stored": True, **result}
    finally:
        store.close()


def save_conspect_files(document_path, files, *, create=False):
    """Зафиксировать пару файлов конспекта как ревизию SQLite-записи."""
    store, binding = open_for_document(document_path)
    if store is None:
        return None
    try:
        json_entries = [(Path(path), value) for path, value in files
                        if Path(path).name.endswith(".conspect.json")]
        md_entries = [(Path(path), value) for path, value in files
                      if Path(path).name.endswith(".conspect.md")]
        if len(json_entries) != 1 or len(md_entries) != 1 or len(files) != 2:
            raise ValueError("SQLite conspect writes require one JSON/Markdown pair")
        json_path, json_bytes = json_entries[0]
        md_path, md_bytes = md_entries[0]
        try:
            payload = json.loads(json_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid conspect JSON") from exc
        recording_id = binding["recording_id"]
        if payload.get("recording_id") != recording_id:
            raise storage.Conflict("conspect belongs to another recording")
        name = payload.get("name")
        conspect_id = str(uuid.uuid5(uuid.UUID(recording_id), "conspect:" + str(name)))
        root = binding["root"]
        json_relative = json_path.resolve().relative_to(root).as_posix()
        md_relative = md_path.resolve().relative_to(root).as_posix()
        head = store.db.execute("SELECT current_revision FROM conspects WHERE conspect_id=?",
                                (conspect_id,)).fetchone()
        if create and head is not None:
            raise storage.Conflict("conspect already exists")
        expected_revision = head["current_revision"] if head else None
        current = store.get_current(recording_id)
        source_revision = payload.get("source_revision")
        if source_revision is None:
            source_revision = current["transcript"]["revision"]
            payload["source_revision"] = source_revision
            json_bytes = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
        result = store.save_conspect(
            recording_id, conspect_id, name, source_revision, expected_revision,
             payload, json_bytes, md_bytes, json_relative, md_relative, binding["root_id"],
             str(uuid.uuid4()), current_operation_context(store))
        completed = materialize_conspects(store, recording_id, conspect_id=conspect_id)
        if (conspect_id, result["conspect_revision"]) not in completed:
            raise storage.StorageError(
                "Свёртка записана в базу, но её файлы ещё не обновлены. Повторите действие позже: программа "
                "допишет файлы при следующей записи (или запустите sync_recording_documents).")
        return result
    finally:
        store.close()


def conspect_for_document(document_path, name):
    """Прочитать authoritative conspect revision, не полагаясь на materialized JSON."""
    store, binding = open_for_document(document_path)
    if store is None:
        return None
    try:
        for saved in store.list_conspects(binding["recording_id"]):
            if saved["name"] == name:
                return {
                    "recording_id": binding["recording_id"],
                    "conspect_id": saved["conspect_id"],
                    "revision": saved["current_revision"],
                    "source_revision": saved["source_revision"],
                    "payload": saved["payload"],
                    "json": saved["json"],
                    "md": saved["md"],
                    "json_hash": saved["json_hash"],
                    "md_hash": saved["md_hash"],
                }
        return None
    finally:
        store.close()


def materialize_conspects(store, recording_id, *, conspect_id=None):
    """Записать очередь свёрток записи. Повреждённая копия останавливает только свою свёртку.

    conspect_id — чья это операция: отказ бросается лишь за неё, без него — за любую свёртку записи.
    """
    import conspect
    binding = document_binding(store, recording_id)
    json_path = binding["root"] / binding["paths"]["json"]
    with conspect.write_lock(json_path):
        root = documents._authorized_root(store, binding["root_id"], binding["root"])
        with documents._file_lock(root):
            conspect.recover_pending(json_path)
            completed, damaged = _materialize_conspects_locked(store, recording_id)
    for ident, exc in damaged.items():
        if conspect_id in (None, ident):
            raise exc
    return completed


def refresh_conspects(store, recording_id, document_path):
    """Довести очередь и пересобрать только конспекты с устаревшей опорой."""
    import conspect
    materialize_conspects(store, recording_id)
    if store.db.execute(
            "SELECT 1 FROM conspect_outbox o JOIN conspects c USING(conspect_id) "
            "WHERE c.recording_id=? AND o.state!='synced' LIMIT 1", (recording_id,)).fetchone():
        raise storage.Conflict("conspect materialization is not synchronized")
    current = store.get_current(recording_id)
    source_revision, review_revision = current["transcript"]["revision"], current["review"]["revision"]
    report = {}
    for saved in store.list_conspects(recording_id):
        payload = saved["payload"]
        # Свёртка «цитаты по одной» (quote_policy 1) показывает цитаты по текущему разбору: она устаревает и
        # когда изменилась только ревизия разбора (место закрыто, слова те же). Остальные — по стенограмме.
        stale = (payload.get("assembly_revision") != source_revision
                 or (payload.get("quote_policy") == 1
                     and payload.get("assembly_review_revision") != review_revision))
        if stale:
            report.update(conspect.rerender(document_path, saved["name"]))
    return report


def _materialize_conspects_locked(store, recording_id):
    import conspect
    completed, damaged = [], {}
    rows = store.db.execute(
        "SELECT o.conspect_id, o.revision, o.root_id, o.expected_previous_hashes, "
        "r.path AS root_path, c.json_relative, c.md_relative "
        "FROM conspect_outbox o JOIN conspects c USING(conspect_id) "
        "JOIN allowed_roots r USING(root_id) WHERE c.recording_id=? AND o.state='pending' "
        "ORDER BY c.name, o.revision", (recording_id,)).fetchall()
    for row in rows:
        if row["conspect_id"] in damaged:
            continue    # следующая ревизия не обгоняет остановленную
        if store.db.execute(
                "SELECT 1 FROM conspect_outbox WHERE conspect_id=? AND revision<? AND state='conflict'",
                (row["conspect_id"], row["revision"])).fetchone():
            continue
        saved = store.get_conspect(row["conspect_id"], row["revision"])
        root = Path(row["root_path"])
        targets = {"json": (row["json_relative"], saved["json"], saved["json_hash"]),
                   "md": (row["md_relative"], saved["md"], saved["md_hash"])}
        expected = storage._loads(row["expected_previous_hashes"])
        conflicts = []
        overrides = []
        for kind, (relative, desired_bytes, desired_hash) in targets.items():
            target, exists = documents._read_only_target(root, relative)
            current_hash = documents._file_hash(target) if exists else None
            resolved = store.db.execute(
                "SELECT observed_hash, observed_size, observed_bytes FROM conspect_conflicts WHERE conspect_id=? AND revision=? "
                "AND kind=? AND relative_path=? AND resolution_operation_id IS NOT NULL ORDER BY rowid DESC LIMIT 1",
                (row["conspect_id"], row["revision"], kind, relative)).fetchone()
            verdict = documents.file_verdict(current_hash, desired_hash, expected.get(kind), resolved)
            if verdict == "gone_after_confirmation":
                raise storage.Conflict("external conspect file disappeared after confirmation")
            if verdict == "external":    # правки ищутся по всем файлам свёртки сразу, копии решений — после них
                conflicts.append(documents._FileConflict(kind, target))
            overrides.append((resolved, current_hash, desired_hash, relative))
        if conflicts:
            with store._transaction():
                store.db.execute("UPDATE conspect_outbox SET state='conflict', error_code='external_edit' "
                                 "WHERE conspect_id=? AND revision=?",
                                 (row["conspect_id"], row["revision"]))
                for conflict in conflicts:
                    relative = targets[conflict.kind][0]
                    store.db.execute(
                        "INSERT INTO conspect_conflicts VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                        (str(uuid.uuid4()), row["conspect_id"], row["revision"], conflict.kind,
                         relative, conflict.content, conflict.digest, conflict.size))
                    store.db.execute("UPDATE conspect_file_locations SET conflict=1 "
                                     "WHERE conspect_id=? AND kind=?", (row["conspect_id"], conflict.kind))
            continue
        # Проверять только запись, которая действительно состоится, а не открытые конфликты
        # или исторические решения. Защита общая для экрана, MCP и синхронизации.
        try:
            for resolved, current_hash, desired_hash, relative in overrides:
                documents._check_override_copy(resolved, current_hash, desired_hash, relative)
        except documents.DamagedCopy as exc:
            damaged[row["conspect_id"]] = exc
            continue
        paths = {kind: root / relative for kind, (relative, _, _) in targets.items()}
        conspect._write_files_raw(
            Path(document_binding(store, recording_id)["root"] /
                 document_binding(store, recording_id)["paths"]["json"]),
            [(paths["json"], saved["json"]), (paths["md"], saved["md"])])
        with store._transaction():
            for kind, (_, _, desired_hash) in targets.items():
                store.db.execute(
                    "UPDATE conspect_file_locations SET confirmed_revision=?, confirmed_hash=?, "
                    "generation=?, conflict=0 WHERE conspect_id=? AND kind=?",
                    (row["revision"], desired_hash, row["revision"], row["conspect_id"], kind))
            store.db.execute("UPDATE conspect_outbox SET state='synced', attempts=attempts+1, "
                             "error_code=NULL WHERE conspect_id=? AND revision=?",
                             (row["conspect_id"], row["revision"]))
        completed.append((row["conspect_id"], row["revision"]))
    return completed, damaged


def load_recording(store, recording_id):
    """Прочитать текущую согласованную пару и ссылки документов без job-файла."""
    current = store.get_current(recording_id)
    transcript = copy.deepcopy(current["transcript"]["payload"])
    review = copy.deepcopy(current["review"]["payload"])
    quality = transcript.get("quality")
    if isinstance(quality, dict):
        runtime = quality.pop("runtime", None)
        if isinstance(runtime, dict):
            transcript.update(runtime)
    data = {**transcript, **review,
            "recording_id": recording_id,
            "source_revision": current["transcript"]["revision"],
            "review_revision": current["review"]["revision"]}
    binding = document_binding(store, recording_id)
    data.update(document_revision=binding["document_revision"],
                document_sync=document_sync_state(store, recording_id, binding["document_revision"]),
                json=str(binding["root"] / binding["paths"]["json"]),
                markdown=str(binding["root"] / binding["paths"]["md"]),
                manifest=str(binding["root"] / binding["paths"]["manifest"]))
    return data


_WORKER_FIELDS = frozenset({
    "file", "duration", "title", "recorded_at", "recorded_timezone", "container", "codec",
    "audio_sha256", "audio_available", "speakers", "speaker_names", "vocab_suggestions",
    "names_source", "names_confirmed", "segments", "quality", "edits",
    "review", "review_done", "questions", "answers", "conflicts",
    "rejected", "applications", "undo", "interaction_events",
    "interaction_events_evicted", "question_version", "speaker_labels",
    "timing", "_timing", "group", "no_profile",
})
_TRANSCRIPT_FIELDS = (
    "file", "duration", "title", "recorded_at", "recorded_timezone",
    "container", "codec", "audio_sha256", "audio_available", "speakers",
    "speaker_names", "names_source", "names_confirmed", "segments",
    "quality", "edits", "vocab_suggestions",
)
_REVIEW_FIELDS = (
    "review", "review_done", "questions", "answers", "conflicts", "rejected",
    "applications", "undo", "interaction_events", "interaction_events_evicted",
    "question_version",
)
_RUNTIME_METADATA = ("speaker_labels", "timing", "_timing", "group", "no_profile")


def document_layout(source_path, tag=""):
    """Рассчитать sidecar и привычные имена без создания каталогов."""
    source = Path(source_path).expanduser()
    if not isinstance(tag, str):
        raise ValueError("tag must be text")
    safe_tag = "".join(char for char in tag if char.isalnum() or char in "-_")
    basename = source.stem + (("." + safe_tag) if safe_tag else "")
    root = source.parent / (source.stem + ".rus-transcriber")
    return {"root": root, "paths": documents.paths_for_basename(basename)}


def prepare_new_layout(store, root_path, document_paths):
    """Зарегистрировать корень и отказать коллизии до запуска тяжёлого worker."""
    root = documents.checked_root(root_path)
    if not isinstance(document_paths, dict) or set(document_paths) != {"md", "json", "manifest"}:
        raise ValueError("invalid document paths")
    normalized = {}
    for kind, relative in document_paths.items():
        parts = documents._relative_parts(relative)
        normalized[kind] = Path(*parts).as_posix()
    if len(set(normalized.values())) != 3:
        raise ValueError("document paths must be distinct")
    root_id = store.get_or_register_root(root)
    for relative in normalized.values():
        target, exists = documents._read_only_target(root, relative)
        canonical = os.path.normcase(os.path.abspath(target))
        if store.db.execute("SELECT 1 FROM file_locations WHERE canonical_path=?", (canonical,)).fetchone():
            raise storage.Conflict("file target already registered")
        if store.db.execute("SELECT 1 FROM conspect_file_locations WHERE canonical_path=?", (canonical,)).fetchone():
            raise storage.Conflict("file target collides with conspect")
        if exists:
            raise FileExistsError(target)
    return root_id, normalized


def snapshots_from_worker(data, recording_id=None):
    """Проверить и разделить результат worker без молчаливой потери полей.

    Runtime-only metadata не становится частью открытого документа: оно сохраняется
    под quality.runtime в снимке. Доменное содержимое и review остаются раздельными.
    """
    if not isinstance(data, dict):
        raise ValueError("worker result must be an object")
    unknown = set(data) - _WORKER_FIELDS
    if unknown:
        raise ValueError("unsupported worker fields: " + ", ".join(sorted(unknown)))

    value = copy.deepcopy(data)
    recording_id = recording_id or str(uuid.uuid4())
    storage._uuid(recording_id)
    transcript = {key: value[key] for key in _TRANSCRIPT_FIELDS if key in value}
    quality = transcript.setdefault("quality", {})
    if not isinstance(quality, dict):
        raise ValueError("worker quality must be an object")
    runtime = {key: value[key] for key in _RUNTIME_METADATA if key in value}
    if runtime:
        quality["runtime"] = runtime

    _ensure_ids(transcript, recording_id)
    review = {key: value[key] for key in _REVIEW_FIELDS if key in value}
    review.setdefault("review", [])
    review.setdefault("answers", {})
    storage.validate_pair(transcript, review)
    return recording_id, transcript, review


def commit_worker_result(store, data, location, root_id, context, *,
                         recording_id, operation_id, generated_at, document_paths=None):
    """Атомарно сохранить новую пару снимков и поставить документы в outbox.

    Store, корень и контекст должны быть подготовлены вызывающим слоем явно.
    Публикация файлов выполняется отдельно через documents.materialize_pending.
    """
    recording_id, transcript, review = snapshots_from_worker(data, recording_id)
    result = store.create_recording_with_document(
        recording_id, normalize_location(location), transcript, review, operation_id, context,
        root_id=root_id, generated_at=generated_at, document_paths=document_paths,
    )
    return {"recording_id": recording_id, **result}


def commit_recording_data(store, recording_id, data, expected_versions, operation_id=None,
                          context=None, root_id=None, generated_at=None, precommit=None):
    """Сохранить изменённое transcript/review-представление через storage."""
    if not isinstance(data, dict) or data.get("recording_id") != recording_id:
        raise ValueError("recording identity mismatch")
    if (not isinstance(expected_versions, dict)
            or set(expected_versions) != {"source_revision", "review_revision"}
            or data.get("source_revision") != expected_versions["source_revision"]
            or data.get("review_revision") != expected_versions["review_revision"]):
        raise storage.Conflict("stale recording snapshot")
    candidate = copy.deepcopy(data)
    _prune_missing_word_ids(candidate, candidate)
    for key in ("recording_id", "source_revision", "review_revision", "document_revision",
                "document_sync", "json", "markdown", "manifest"):
        candidate.pop(key, None)
    _, transcript, review = snapshots_from_worker(candidate, recording_id)
    current = store.get_current(recording_id)
    transcript_changed = transcript != current["transcript"]["payload"]
    review_changed = review != current["review"]["payload"]
    if not transcript_changed and not review_changed:
        if (current["transcript"]["revision"] != expected_versions["source_revision"] or
                current["review"]["revision"] != expected_versions["review_revision"]):
            raise storage.Conflict("stale recording snapshot")
        binding = document_binding(store, recording_id)
        return {"recording_id": recording_id, **expected_versions,
                "document_revision": binding["document_revision"],
                "document_sync": document_sync_state(store, recording_id, binding["document_revision"]),
                "unchanged": True}
    binding = document_binding(store, recording_id)
    operation_id = operation_id or str(uuid.uuid4())
    context = context or current_operation_context(store)
    root_id = root_id or binding["root_id"]
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    documents.guard_before_write(store, recording_id)     # ручная правка файла: до записи в базу, вне транзакции
    # Изменение source всегда получает новый согласованный review-снимок.
    with store._transaction():
        if precommit is not None:
            precommit(store)
        return store.commit_change_with_document(
            recording_id, expected_versions, operation_id, context,
            transcript=transcript if transcript_changed else None,
            review=review if (review_changed or transcript_changed) else None,
            root_id=root_id, generated_at=generated_at)


def _prune_missing_word_ids(transcript, review):
    """Убрать только исчезнувшие ссылки из производных списков разбора."""
    ids = {word.get("word_id") for segment in transcript.get("segments") or []
           for word in segment.get("words") or [] if word.get("word_id")}
    for field in ("review", "questions"):
        for item in review.get(field) or []:
            refs = item.get("word_ids")
            if isinstance(refs, list):
                remaining = [ident for ident in refs if ident in ids]
                if remaining:
                    item["word_ids"] = remaining
                else:
                    item.pop("word_ids", None)


def _ensure_ids(transcript, recording_id):
    segments = transcript.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("worker segments must be a list")
    seen_segments = set()
    seen_words = set()
    namespace = uuid.UUID(recording_id)
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError("worker segment must be an object")
        ident = segment.get("segment_id")
        if not isinstance(ident, str) or not ident or ident in seen_segments:
            ident = str(uuid.uuid5(namespace, f"segment:{segment_index}"))
            segment["segment_id"] = ident
        seen_segments.add(ident)
        words = segment.get("words", [])
        if not isinstance(words, list):
            raise ValueError("worker words must be a list")
        for word_index, word in enumerate(words):
            if not isinstance(word, dict):
                raise ValueError("worker word must be an object")
            ident = word.get("word_id")
            if not isinstance(ident, str) or not ident or ident in seen_words:
                ident = str(uuid.uuid5(namespace, f"word:{segment_index}:{word_index}"))
                word["word_id"] = ident
            seen_words.add(ident)


# --- База записей установки (шаг «База записей» install.ps1, `rt.py db-init`) ---
# Явная операция, которую запускает человек (установщиком или командой); сервер и обычные инструменты базу
# по-прежнему не создают. Это не перенос: создаётся пустая база, файлы записей не читаются и не меняются.


def _config_readable():
    """config.json нет или он читается как словарь; иначе дописывание выбора стёрло бы остальные настройки."""
    import paths
    p = Path(paths.CONFIG_FILE)
    if not p.exists():
        return True
    try:
        return isinstance(json.loads(p.read_text(encoding="utf-8-sig")), dict)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def prepare_installation_store(check_only=False):
    """Базы нет — создать её штатно и выбрать `sqlite`; база есть — не трогать, только выбрать `sqlite`.

    Существующая база открывается только чтением и проверяется (схема, единственная активная установка);
    не открывается — `ok=False`, ничего не чинится и не создаётся. Ключ `case`: `created` — база создана и
    выбрана; `selected` — база была, выбрано хранение в ней; `ready` — база была и уже выбрана; `error` —
    база записей не готова (установщик: код 2), причина в `message`. check_only=True — только сказать,
    что было бы сделано.
    """
    import paths
    import settings
    db = Path(paths.STATE_DB)
    result = {"database": str(db), "check_only": bool(check_only), "changed": []}

    def done(case, ok, message):
        result.update(case=case, ok=ok, message=message)
        try:
            result["recording_backend"] = settings.recording_backend()
        except ValueError:
            result["recording_backend"] = None
        return result

    def failed(what, ex):
        return done("error", False, f"База записей не готова: {what} ({type(ex).__name__}: {ex}).")

    if not _config_readable():
        return done("error", False, f"База записей не готова: {paths.CONFIG_FILE} не читается; исправьте или "
                    "удалите файл и запустите `rt.py db-init` снова. Ничего не изменено.")
    try:
        selected = settings.recording_backend() == "sqlite"
    except ValueError:
        selected = False                  # неверное значение — будет заменено выбором базы
    created = False
    if db.exists():
        try:
            store = open_store(db, readonly=True)
            try:
                store.current_identity()
            finally:
                store.close()
        except (OSError, ValueError, sqlite3.Error, storage.StorageError) as ex:
            return failed(f"{db} не открывается; база не изменена, новая не создана", ex)
        if selected:
            return done("ready", True, f"База записей есть: {db}. Записи хранятся в базе.")
        if check_only:
            return done("selected", True, f"База записей есть: {db}; будет выбрано хранение в ней.")
    else:
        if check_only:
            return done("created", True, f"Будет создана база записей: {db}.")
        try:
            initialize_store(db, display_name="").close()
        except (OSError, ValueError, sqlite3.Error, storage.StorageError) as ex:
            return failed(f"не удалось создать {db}", ex)
        result["changed"].append("state.sqlite3")
        created = True
    try:
        settings.set_recording_backend("sqlite")
    except (OSError, ValueError, RuntimeError, sqlite3.Error, storage.StorageError) as ex:
        return failed(f"база {db} есть, но выбрать её не удалось; запустите `rt.py db-init` снова", ex)
    result["changed"].append("config.json")
    if created:
        return done("created", True, f"База записей создана: {db}. Записи будут храниться в базе.")
    return done("selected", True, f"База записей есть: {db}. Выбрано хранение записей в ней; база не изменена.")
