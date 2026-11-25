"""Источник обучения голосов: файл стенограммы старого вида или запись в базе SQLite.

Для записи в базе данные берутся только из зафиксированной ревизии (`recordings.load_recording`), а не из файла
выгрузки и не из черновика ответов. Жетон ревизии `sqlite:<store_id>:<recording_id>:<ревизия>:<хэш>` заменяет
SHA-256 файла в подтверждении: изменилась ревизия или база — нужен новый просмотр.

Связь со старым файлом, перенесённым в базу (`tools/import_legacy.py`), вычисляется при чтении: запись связана
со старым ключом профиля, если `quality.migration.source.path` её первой ревизии указывает на этот файл. Других
способов угадать связь нет. Модуль только читает: базу открывает только на чтение (`readonly=True`),
в профили и рядом с аудио ничего не пишет.
"""
import hashlib
import os
from pathlib import Path

RECORDING_PREFIX = "recording:"
SQLITE_FINGERPRINT_KEYS = ("store_id", "recording_id", "transcript_revision", "transcript_payload_hash",
                           "audio_size", "audio_mtime")


class SourceError(Exception):
    """Источник нельзя безопасно открыть; текст — для человека."""


def file_hash(path):
    p = Path(path)
    if not p.is_file():
        return "absent"
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_path(a, b):
    """Пути равны без учёта регистра (в базе `location` хранится через normcase, в профиле — как дал resolve)."""
    return os.path.normcase(str(a)).lower() == os.path.normcase(str(b)).lower()


def matching_keys(mapping, key):
    return [k for k in (mapping or {}) if same_path(k, key)]


def existing_audio_key(prof, audio_key):
    """Ключ вклада этого аудио, если он уже есть в профиле под другим написанием; иначе сам audio_key."""
    for speaker in (prof.get("speakers") or {}).values():
        for k in matching_keys(speaker.get("parts"), audio_key):
            return k
    for section in ("quarantine", "identity", "confusions", "audio_quality"):
        for k in matching_keys(prof.get(section), audio_key):
            return k
    return audio_key


class Source:
    """Один источник обучения. `key` — ключ в prof["sources"], `name` — подпись в истории и окне."""

    def __init__(self, kind, json_path, data, key, name, token, ref, **extra):
        self.kind, self.json_path, self.data = kind, Path(json_path), data
        self.key, self.name, self.token, self.ref = key, name, token, ref
        self.recording_id = extra.get("recording_id")
        self.store_id = extra.get("store_id")
        self.revision = extra.get("revision")
        self.payload_hash = extra.get("payload_hash")
        self.legacy_keys = list(extra.get("legacy_keys") or [])
        self.migrated_from = extra.get("migrated_from")
        self.migration_sha256 = extra.get("migration_sha256")
        self.redirected_from = extra.get("redirected_from")
        self.revision_time = float(extra.get("revision_time") or 0.0)   # для выбора самой свежей копии аудио

    def __str__(self):
        return self.ref

    def fingerprint(self, audio_path):
        if self.kind == "file":
            import voices
            return voices.source_fingerprint(self.json_path, audio_path)
        st = Path(audio_path).stat()
        return {"kind": "sqlite", "store_id": self.store_id, "recording_id": self.recording_id,
                "transcript_revision": self.revision, "transcript_payload_hash": self.payload_hash,
                "audio_size": int(st.st_size), "audio_mtime": round(float(st.st_mtime), 3)}

    def describe(self):
        """Сведения для отчёта и окна подтверждения: без путей, только имена файлов."""
        out = {"kind": self.kind, "name": self.name}
        if self.kind == "sqlite":
            out.update(recording=self.ref, revision=self.revision)
            if self.migrated_from:
                out["migrated_from"] = self.migrated_from
            if self.redirected_from:
                out["redirected_from"] = self.redirected_from
        return out


def _store_id(store):
    row = store.db.execute("SELECT store_id FROM store_meta").fetchone()
    if row is None:
        raise SourceError("у базы нет идентификатора")
    return row[0]


def _migration_link(store, recording_id):
    import recordings
    return recordings.migration_link(store, recording_id)


def _revision_time(store, recording_id, revision):
    """Время операции, создавшей ревизию стенограммы, в секундах; 0.0 — если не разобрать."""
    from datetime import datetime
    row = store.db.execute("SELECT o.created_at FROM transcript_revisions r JOIN operations o "
                           "ON o.operation_id=r.operation_id WHERE r.recording_id=? AND r.revision=?",
                           (recording_id, revision)).fetchone()
    try:
        return datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")).timestamp() if row else 0.0
    except (TypeError, ValueError, OverflowError, OSError):
        return 0.0


def _sqlite_source(store, recording_id, redirected_from=None):
    import recordings
    data = recordings.load_recording(store, recording_id)
    revision = data["source_revision"]
    payload_hash = store.get_revision(recording_id, "transcript", revision)["payload_hash"]
    store_id = _store_id(store)
    link = _migration_link(store, recording_id)
    legacy_keys = [str(Path(link["path"]).resolve())] if link else []
    name = Path(data["json"]).name
    return Source("sqlite", data["json"], data, RECORDING_PREFIX + recording_id, name,
                  f"sqlite:{store_id}:{recording_id}:{revision}:{payload_hash}", RECORDING_PREFIX + recording_id,
                  recording_id=recording_id, store_id=store_id, revision=revision, payload_hash=payload_hash,
                  legacy_keys=legacy_keys, migrated_from=Path(link["path"]).name if link else None,
                  migration_sha256=link.get("sha256") if link else None,
                  revision_time=_revision_time(store, recording_id, revision), redirected_from=redirected_from)


def record_source(store, recording_id):
    """Источник по записи в уже открытой базе (только чтение)."""
    return _sqlite_source(store, recording_id)


def linked_profile_keys(sources, source):
    """Старые ключи профиля (sources), из которых перенесена эта запись в базе."""
    return [k for old in source.legacy_keys for k in matching_keys(sources, old)]


def linked_status(stored, source):
    """Состояние старого ключа, связанного с записью в базе: unchanged / legacy_baseline_pending / changed.

    unchanged — старый отпечаток совпал с перенесённым файлом, а запись после переноса не правили (ревизия 1);
    legacy_baseline_pending — у старого ключа только mtime, а старый файл на месте и не менялся после переноса:
    исходное состояние можно принять, но только подтверждённым обновлением; всё прочее — changed (нужен просмотр).
    """
    if source.revision != 1 or not source.migration_sha256:
        return "changed"
    if isinstance(stored, dict):
        return "unchanged" if stored.get("transcript_sha256") == source.migration_sha256 else "changed"
    if isinstance(stored, (int, float)) and not isinstance(stored, bool) and source.legacy_keys:
        return ("legacy_baseline_pending" if file_hash(source.legacy_keys[0]) == source.migration_sha256
                else "changed")
    return "changed"


def _open_store_readonly():
    import recordings
    import storage
    try:
        store = recordings.open_store(readonly=True)
    except storage.StorageError as ex:
        raise SourceError(f"база недоступна: {ex}") from ex
    try:
        store.current_identity()
    except BaseException:
        store.close()
        raise
    return store


def folder_sources(folder):
    """Записи базы, чьё аудио лежит в папке folder (охват прежнего просмотра папки). -> (источники, не открылись).

    Только чтение. База недоступна — SourceError; запись, которая не открывается, считается во втором числе.
    """
    import storage
    target = os.path.abspath(str(folder))
    try:
        store = _open_store_readonly()
    except storage.StorageError as ex:
        raise SourceError(f"база недоступна: {ex}") from ex
    out, failed = [], 0
    try:
        rows = store.db.execute("SELECT recording_id, location FROM recordings ORDER BY recording_id").fetchall()
        for recording_id, location in rows:
            if not same_path(os.path.dirname(str(location)), target):
                continue
            try:
                out.append(_sqlite_source(store, recording_id))
            except (KeyError, ValueError, OSError, SourceError, storage.StorageError):
                failed += 1
        return out, failed
    finally:
        store.close()


def migrated_copies(store, legacy_path):
    """Записи базы, перенесённые из этого старого файла: [(recording_id, migration.source)]."""
    import recordings
    return recordings.migrated_copies(store, legacy_path)


def _redirect(legacy_path):
    """Старая стенограмма при работе с базой: перейти к её единственной копии в базе."""
    import recordings
    store = recordings.open_store(readonly=True)
    try:
        store.current_identity()
        # тот же отбор, что у старых номеров задач (recordings.find_migrated): путь и SHA-256 файла сейчас
        found = recordings.find_migrated(store, legacy_path)
        if found["state"] == "changed":
            raise SourceError("старый файл изменился после переноса в базу; выберите запись в базе явно "
                              "(recording:<id>)")
        if found["state"] != "found":
            raise SourceError("у этой старой записи нет единственной копии в базе "
                              f"(найдено {len(found['candidates'])}); выберите запись в базе явно (recording:<id>)")
        return _sqlite_source(store, found["recording_id"], redirected_from=Path(legacy_path).name)
    finally:
        store.close()


def _file_source(json_path, data):
    p = Path(json_path)
    return Source("file", p, data, str(p.resolve()), p.name, file_hash(p), str(p))


def open_document(json_path):
    """Источник по пути стенограммы (из job_json, rt.py или воркера)."""
    import json
    import recordings
    import settings
    import storage
    p = Path(json_path)
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError) as ex:
        raise SourceError(f"стенограмма не читается: {ex}") from ex
    if not isinstance(data, dict):
        raise SourceError("стенограмма должна быть JSON-объектом")
    passport = data.get("passport")
    export = isinstance(passport, dict) and passport.get("storage_backend") == "sqlite"
    try:
        store, binding = recordings.open_for_document(p, readonly=True)
    except storage.Conflict:
        if export or settings.recording_backend() != "sqlite":
            raise
        return _redirect(p)
    if store is None:
        return _file_source(p, data)
    try:
        return _sqlite_source(store, binding["recording_id"])
    finally:
        store.close()


def open_ref(ref):
    """Источник по `recording:<id>` или пути стенограммы."""
    import recordings
    import storage
    if isinstance(ref, str) and ref.startswith(RECORDING_PREFIX):
        recording_id = ref[len(RECORDING_PREFIX):]
        store = recordings.open_store(readonly=True)
        try:
            store.current_identity()
            return _sqlite_source(store, recording_id)
        except (KeyError, ValueError) as ex:
            raise SourceError(f"запись в базе не найдена: {ex}") from ex
        except storage.StorageError as ex:
            raise SourceError(f"запись в базе недоступна: {ex}") from ex
        finally:
            store.close()
    return open_document(ref)
