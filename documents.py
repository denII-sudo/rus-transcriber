"""Чистая подготовка синтетического комплекта документов П1б, без записи файлов."""

import hashlib
import html
import json
import os
import re
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

import storage

FORMAT_VERSION = 2
RENDERER_VERSION = 1
MAX_CONFLICT_BYTES = 8 * 1024 * 1024


class _FileConflict(storage.Conflict):
    def __init__(self, kind, path):
        self.kind = kind
        self.path = path
        _not_reparse(path)
        with path.open("rb") as source:
            self.content = source.read(MAX_CONFLICT_BYTES + 1)
        self.size = path.stat().st_size
        if len(self.content) > MAX_CONFLICT_BYTES or self.size > MAX_CONFLICT_BYTES:
            self.content = None
            self.digest = _file_hash(path)
        else:
            self.digest = _digest(self.content)
            self.size = len(self.content)
        super().__init__("document conflict")


class DamagedCopy(storage.StorageError):
    """Копия, разрешающая перезапись, непригодна: останавливается только её объект, не соседи."""


def _not_reparse(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("symlink or reparse point not allowed")


def checked_root(path):
    """Проверить существующий абсолютный корень и все его компоненты."""
    root = Path(path)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("absolute root without traversal required")
    storage.check_local_path(root)
    for parent in reversed((root, *root.parents)):
        _not_reparse(parent)
    if not root.is_dir():
        raise ValueError("root must be a directory")
    return root.resolve(strict=True)


def _relative_parts(relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(x in ("..", ".", "") for x in path.parts):
        raise ValueError("invalid relative path")
    return path.parts


def _target(root, relative):
    parts = _relative_parts(relative)
    current = root
    for part in parts[:-1]:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        _not_reparse(current)
        if not current.is_dir():
            raise ValueError("target parent is not a directory")
    target = current / parts[-1]
    if os.path.lexists(target):
        _not_reparse(target)
        if not target.is_file():
            raise ValueError("target is not a regular file")
    return target


def _read_only_target(root, relative):
    """Разрешить цель без создания каталогов; вернуть путь и признак существования."""
    parts = _relative_parts(relative)
    current = root
    for part in parts[:-1]:
        current = current / part
        if not os.path.lexists(current):
            return root.joinpath(*parts), False
        _not_reparse(current)
        if not current.is_dir():
            raise ValueError("target parent is not a directory")
    target = current / parts[-1]
    if not os.path.lexists(target):
        return target, False
    _not_reparse(target)
    if not target.is_file():
        raise ValueError("target is not a regular file")
    return target, True


@contextmanager
def _file_lock(root):
    """OS-блокировка одной очереди корня; не считать наличие файла блокировкой."""
    path = _target(root, ".rus-transcriber-materializer.lock")
    with open(path, "a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _file_digest(path):
    _not_reparse(path)
    if not path.is_file():
        raise ValueError("target is not a regular file")
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _file_hash(path):
    if not os.path.lexists(path):
        return None
    return _file_digest(path)[0]


def _write_atomic(target, data, *, immutable=False, conflict_kind=None):
    temp = target.parent / (".rt-" + uuid.uuid4().hex + ".tmp")
    try:
        with open(temp, "xb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        if immutable:
            try:
                os.link(temp, target)
            except FileExistsError:
                if _file_hash(target) != _digest(data):
                    raise _FileConflict(conflict_kind or "archive_unknown", target)
        else:
            os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def _record_conflict(store, root_id, rid, revision, exc):
    root = store.db.execute("SELECT path FROM allowed_roots WHERE root_id=?", (root_id,)).fetchone()["path"]
    relative = Path(os.path.relpath(exc.path, root)).as_posix()
    with store._transaction():
        store.db.execute("INSERT INTO document_conflicts VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                         (str(uuid.uuid4()), rid, revision, exc.kind, relative, exc.content, exc.digest, exc.size))
        store.db.execute("UPDATE file_locations SET conflict=1 WHERE root_id=? AND relative_path=?",
                         (root_id, relative))
        store.db.execute("UPDATE document_outbox SET state='conflict', error_code='conflict' "
                         "WHERE recording_id=? AND document_revision=?", (rid, revision))


def _check_override_copy(override, current, desired, relative):
    """Под блокировкой писателя проверить копию, разрешающую именно эту перезапись.

    Старые решения не участвуют; новый внешний файл обработается как новый конфликт.
    Уже записанная версия программы не требует пригодной копии для повторного подтверждения.
    """
    if override is None or current == desired or current != override["observed_hash"]:
        return
    content = override["observed_bytes"]
    if (not isinstance(content, (bytes, bytearray, memoryview)) or len(content) != override["observed_size"]
            or _digest(content) != override["observed_hash"]):
        raise DamagedCopy(
            f"Сохранённая копия правки файла «{relative}» повреждена или отсутствует, поэтому файл не заменён. "
            "Повтор без изменений не поможет. Скопируйте файл в другое место, затем измените его, не увеличивая: "
            "удалите один любой знак (в пустой файл добавьте один) и сохраните. Если файл останется не больше "
            "8 МиБ, при повторе записи программа сохранит новую копию и покажет новый конфликт, в котором файл "
            "можно заменить; если станет больше, копия не сохранится и заменить его на экране будет нельзя. "
            "Если файл совпадёт с версией программы, заменять его уже не нужно.")


def _authorized_root(store, root_id, root_path):
    """Проверить, что корень зарегистрирован, активен и совпадает с переданным путём."""
    storage._uuid(root_id)
    root = checked_root(root_path)
    row = store.db.execute("SELECT path, requires_remap FROM allowed_roots WHERE root_id=?", (root_id,)).fetchone()
    locked = store.db.execute("SELECT locked FROM restore_guard WHERE id=1").fetchone()[0]
    if locked or row is None or row["requires_remap"] or os.path.normcase(row["path"]) != os.path.normcase(str(root)):
        raise ValueError("root not explicitly authorized")
    return root


def materialize_pending(store, root_id, root_path, *, fault=None, recording_id=None):
    """Обработать очередь зарегистрированного, явно разрешённого корня по порядку.

    Повреждённая копия останавливает только свою запись. recording_id — чья это операция:
    отказ вызывающему бросается лишь за неё, без него — за любую запись корня.
    """
    root = _authorized_root(store, root_id, root_path)
    fault = fault or (lambda boundary, recording_id, revision: None)
    completed = []
    with _file_lock(root):
        rows = store.db.execute(
            "SELECT recording_id, document_revision, state, generation, expected_previous_hashes FROM document_outbox "
            "WHERE root_id=? AND state!='synced' ORDER BY recording_id, document_revision",
            (root_id,)).fetchall()
        blocked = set()
        damaged = {}
        for row in rows:
            rid, rev = row["recording_id"], row["document_revision"]
            if rid in blocked or row["state"] == "conflict":
                blocked.add(rid)
                continue
            with store._transaction():
                store.db.execute("UPDATE document_outbox SET state='writing', attempts=attempts+1, "
                                 "error_code=NULL WHERE recording_id=? AND document_revision=?", (rid, rev))
            try:
                fault("writing", rid, rev)
                doc = store.get_document(rid, rev)
                paths = current_paths_from_document(doc, rid)
                expected_hashes = storage._loads(row["expected_previous_hashes"])
                if row["generation"] != rev:
                    raise storage.StorageError("outbox generation mismatch")
                archives = archive_paths(rid, rev)
                for kind in ("md", "json", "manifest"):
                    value = doc["archive_manifest"] if kind == "manifest" else doc[kind]
                    path = _target(root, archives[kind])
                    current = _file_hash(path)
                    if current is not None and current != _digest(value):
                        raise _FileConflict("archive_" + kind, path)
                    if current is None:
                        _write_atomic(path, value, immutable=True, conflict_kind="archive_" + kind)
                    fault("archive_" + kind, rid, rev)
                for kind, filename in paths.items():
                    path = _target(root, filename)
                    current = _file_hash(path)
                    desired = _digest(doc[kind])
                    expected = expected_hashes[kind]
                    override = store.db.execute(
                        "SELECT observed_hash, observed_size, observed_bytes FROM document_conflicts WHERE recording_id=? "
                        "AND document_revision=? AND kind=? AND relative_path=? AND resolution_operation_id IS NOT NULL "
                        "ORDER BY rowid DESC LIMIT 1", (rid, rev, kind, filename)).fetchone()
                    if override is not None and current not in (desired, override["observed_hash"]):
                        if current is None:
                            raise storage.Conflict("external document disappeared after confirmation")
                        raise _FileConflict(kind, path)
                    if override is None and current not in (None, desired, expected):
                        raise _FileConflict(kind, path)
                    if current != desired:
                        _check_override_copy(override, current, desired, filename)
                        _write_atomic(path, doc[kind])
                    fault("replace_" + kind, rid, rev)
                fault("before_ack", rid, rev)
                with store._transaction():
                    for kind, filename in paths.items():
                        store.db.execute("UPDATE file_locations SET confirmed_revision=?, confirmed_hash=?, "
                                         "generation=?, conflict=0 WHERE root_id=? AND relative_path=?",
                                         (rev, doc["hashes"][kind], rev, root_id, filename))
                    for kind, filename in archives.items():
                        hash_key = "archive_manifest" if kind == "manifest" else kind
                        store.db.execute("UPDATE file_locations SET confirmed_revision=?, confirmed_hash=?, "
                                         "generation=?, conflict=0 WHERE root_id=? AND relative_path=?",
                                         (rev, doc["hashes"][hash_key], rev, root_id, filename))
                    store.db.execute("UPDATE document_outbox SET state='synced', error_code=NULL "
                                     "WHERE recording_id=? AND document_revision=? AND state='writing'",
                                     (rid, rev))
                completed.append((rid, rev))
                fault("after_ack", rid, rev)
            except _FileConflict as exc:
                _record_conflict(store, root_id, rid, rev, exc)
                blocked.add(rid)
            except DamagedCopy as exc:
                with store._transaction():
                    store.db.execute("UPDATE document_outbox SET state='error', error_code='damaged_copy' "
                                     "WHERE recording_id=? AND document_revision=? AND state!='synced'", (rid, rev))
                blocked.add(rid)
                damaged[rid] = exc
            except storage.Conflict:
                with store._transaction():
                    store.db.execute("UPDATE document_outbox SET state='error', error_code='unclassified_conflict' "
                                     "WHERE recording_id=? AND document_revision=? AND state!='synced'", (rid, rev))
                blocked.add(rid)
            except Exception:
                with store._transaction():
                    store.db.execute("UPDATE document_outbox SET state='error', error_code='io_or_interrupted' "
                                     "WHERE recording_id=? AND document_revision=? AND state!='synced'", (rid, rev))
                raise
        latest = store.db.execute(
            "SELECT o.recording_id, o.document_revision FROM document_outbox o "
            "WHERE o.root_id=? AND o.state='synced' AND NOT EXISTS "
            "(SELECT 1 FROM document_outbox x WHERE x.recording_id=o.recording_id "
            "AND (x.state!='synced' OR x.document_revision>o.document_revision))", (root_id,)).fetchall()
        for row in latest:
            rid, rev = row["recording_id"], row["document_revision"]
            try:
                doc = store.get_document(rid, rev)
                for kind, filename in current_paths_from_document(doc, rid).items():
                    target = _target(root, filename)
                    observed = _file_hash(target)
                    if observed is None:
                        _write_atomic(target, doc[kind])
                        with store._transaction():
                            store.db.execute("INSERT INTO materialization_events VALUES (?, ?, ?, ?, ?, ?)",
                                             (str(uuid.uuid4()), rid, rev, kind, "missing_restored", storage._now()))
                    elif observed != doc["hashes"][kind]:
                        raise _FileConflict(kind, target)
            except _FileConflict as exc:
                _record_conflict(store, root_id, rid, rev, exc)
            except Exception:
                with store._transaction():
                    store.db.execute("UPDATE document_outbox SET state='error', error_code='io_or_interrupted' "
                                     "WHERE recording_id=? AND document_revision=? AND state='synced'", (rid, rev))
                raise
    for rid, exc in damaged.items():
        if recording_id in (None, rid):
            raise exc
    return completed


VERIFY_STATUSES = ("ok", "missing", "modified", "unconfirmed", "unresolved_conflict",
                   "invalid_target", "unreadable", "registry_mismatch", "registry_incomplete",
                   "unknown_document")


def verify_files(store, root_id, root_path):
    """Сверить file_locations, версии документов и файлы корня; только чтение, без починки."""
    owns_transaction = not store.db.in_transaction
    if owns_transaction:
        try:
            store.db.execute("BEGIN")
        except storage.sqlite3.OperationalError as exc:
            if storage._is_busy(exc):
                raise storage.Busy("database is busy") from exc
            raise storage.StorageError("cannot begin verification snapshot") from exc
    try:
        return _verify_files_snapshot(store, root_id, root_path)
    except storage.sqlite3.DatabaseError as exc:
        if storage._is_busy(exc):
            raise storage.Busy("database is busy") from exc
        raise storage.StorageError("cannot read verification snapshot") from exc
    finally:
        if owns_transaction and store.db.in_transaction:
            store.db.execute("ROLLBACK")


def _verify_files_snapshot(store, root_id, root_path):
    root = _authorized_root(store, root_id, root_path)
    states = _latest_outbox_states(store, root_id)
    entries = []
    seen = set()
    for loc in store.db.execute("SELECT * FROM file_locations WHERE root_id=? "
                                "ORDER BY recording_id, relative_path", (root_id,)).fetchall():
        entries.append(_verify_location(store, root, loc, states.get(loc["recording_id"])))
        seen.add((loc["recording_id"], loc["relative_path"]))
    entries += _registry_gaps(store, root_id, seen, states)
    counts = {status: 0 for status in VERIFY_STATUSES}
    for entry in entries:
        counts[entry["status"]] += 1
    return {"root_id": root_id, "root_path": str(root),
            "consistent": all(entry["status"] == "ok" for entry in entries),
            "checked": len(entries), "counts": counts, "entries": entries}


def _latest_outbox_states(store, root_id):
    states = {}
    for row in store.db.execute("SELECT recording_id, state FROM document_outbox WHERE root_id=? "
                                "ORDER BY recording_id, document_revision", (root_id,)):
        states[row["recording_id"]] = row["state"]
    return states


def _verify_location(store, root, loc, outbox_state):
    entry = {"recording_id": loc["recording_id"], "kind": loc["kind"],
             "relative_path": loc["relative_path"], "document_revision": loc["confirmed_revision"],
             "generation": loc["generation"], "outbox_state": outbox_state, "status": "ok",
             "detail": None, "expected_sha256": None, "observed_sha256": None, "observed_size": None}
    canonical = os.path.normcase(os.path.abspath(os.path.join(str(root), loc["relative_path"])))
    if canonical != loc["canonical_path"]:
        entry["status"] = "registry_mismatch"
        entry["detail"] = "canonical path does not match root and relative path"
        return entry
    revision = loc["confirmed_revision"]
    if revision is not None:
        doc = store.db.execute("SELECT hashes FROM document_versions WHERE recording_id=? "
                               "AND document_revision=?", (loc["recording_id"], revision)).fetchone()
        if doc is None:
            entry["status"] = "unknown_document"
            entry["detail"] = "confirmed revision is missing from document versions"
            return entry
        try:
            hashes = storage._loads(doc["hashes"])
        except storage.StorageError:
            entry["status"] = "registry_mismatch"
            entry["detail"] = "stored document hashes are unreadable"
            return entry
        expected = hashes.get(storage._hash_key(loc["kind"]))
        entry["expected_sha256"] = expected
        if expected is None or expected != loc["confirmed_hash"]:
            entry["status"] = "registry_mismatch"
            entry["detail"] = "confirmed hash does not match document version"
            return entry
    try:
        target, exists = _read_only_target(root, loc["relative_path"])
    except ValueError as exc:
        entry["status"] = "invalid_target"
        entry["detail"] = str(exc)
        return entry
    if exists:
        try:
            entry["observed_sha256"], entry["observed_size"] = _file_digest(target)
        except OSError as exc:
            entry["status"] = "unreadable"
            entry["detail"] = exc.strerror or type(exc).__name__
            return entry
    if loc["conflict"]:
        entry["status"] = "unresolved_conflict"
        entry["detail"] = "registry target is marked as an unresolved conflict"
        return entry
    if revision is None:
        entry["status"] = "unconfirmed"
        entry["detail"] = "file has no acknowledged document revision"
        return entry
    if not exists:
        entry["status"] = "missing"
        entry["detail"] = "file is absent from disk"
        return entry
    if entry["observed_sha256"] != loc["confirmed_hash"]:
        entry["status"] = "modified"
        entry["detail"] = "file content differs from the confirmed revision"
    return entry


def _registry_gaps(store, root_id, seen, states):
    gaps = []
    for row in store.db.execute("SELECT recording_id, document_revision FROM document_outbox "
                                "WHERE root_id=? ORDER BY recording_id, document_revision",
                                (root_id,)).fetchall():
        rid, revision = row["recording_id"], row["document_revision"]
        saved = store.db.execute(
            "SELECT json FROM document_versions WHERE recording_id=? AND document_revision=?",
            (rid, revision)).fetchone()
        try:
            paths = current_paths_from_document({"json": saved["json"]}, rid) if saved else current_paths(rid)
        except storage.StorageError:
            paths = current_paths(rid)
        expected = list(paths.items())
        expected += [("archive_" + kind, path) for kind, path in archive_paths(rid, revision).items()]
        for kind, relative in expected:
            if (rid, relative) in seen:
                continue
            gaps.append({"recording_id": rid, "kind": kind, "relative_path": relative,
                         "document_revision": revision, "generation": revision,
                         "outbox_state": states.get(rid), "status": "registry_incomplete",
                         "detail": "no file_locations row for document version",
                         "expected_sha256": None, "observed_sha256": None, "observed_size": None})
    return gaps


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _md(value):
    value = html.escape(str(value), quote=True).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", value)


def _label(ref):
    if ref["user_id"] is None:
        return "неизвестно"
    return _md(ref["user_name_snapshot"]) if ref["user_name_snapshot"] else "имя не указано"


def _public_ref(ref):
    storage._ref(ref)
    return {"user_id": ref["user_id"], "user_name_snapshot": ref["user_name_snapshot"]}


def _public_provenance(value):
    if not isinstance(value, dict):
        raise ValueError("invalid provenance")
    result = {}
    for dimension, entry in value.items():
        if dimension not in ("text", "speaker", "order") or not isinstance(entry, dict):
            raise ValueError("unsupported provenance dimension")
        item = _select_scalars(entry, {"by": (str,), "edit": (str,),
                                       "actor_kind": (str,), "legacy_author_label": (str,)})
        if "actor_kind" in item and item["actor_kind"] not in storage.ACTOR_KINDS:
            raise ValueError("invalid provenance actor")
        for key in ("author_ref", "initiated_by"):
            if key in entry:
                item[key] = _public_ref(entry[key])
        if (item.get("actor_kind") in ("model", "system", "unknown") and
                item.get("author_ref", {}).get("user_id") is not None):
            raise ValueError("automatic provenance cannot have a human author")
        result[dimension] = item
    return result


def _public_transcript(transcript):
    """Явное подмножество без диагностики, профилей и произвольных вложений."""
    result = _select_scalars(transcript, {"file": (str,), "duration": (int, float, str), "title": (str,),
        "recorded_at": (str,), "recorded_timezone": (str,), "container": (str,), "codec": (str,),
        "audio_sha256": (str,), "audio_available": (bool,), "names_source": (str,),
        "names_confirmed": (bool,)})
    names = transcript.get("speaker_names", {})
    if not isinstance(names, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                          for k, v in names.items()):
        raise ValueError("invalid speaker names")
    result["speaker_names"] = dict(names)
    speakers = transcript.get("speakers", [])
    if not isinstance(speakers, list) or any(not isinstance(s, str) for s in speakers):
        raise ValueError("invalid speakers")
    result["speakers"] = list(speakers)
    quality = transcript.get("quality")
    if quality is not None:
        if not isinstance(quality, dict):
            raise ValueError("invalid quality")
        result["quality"] = _select_scalars(
            quality, {"asr_incomplete_chunks": (int,), "vote3_skipped": (bool, str)})
        if result["quality"].get("asr_incomplete_chunks", 0) < 0:
            raise ValueError("invalid incomplete count")
        # words.combine_hypotheses пишет причину строкой; логическое значение — прежний вид поля
        skipped = result["quality"].get("vote3_skipped")
        if isinstance(skipped, str) and skipped not in ("whisper_incomplete", "asr_incomplete"):
            raise ValueError("invalid public field: vote3_skipped")
        if "vote3_partial" in quality:      # голосование пропущено только на части отрезков
            partial = quality["vote3_partial"]
            if not isinstance(partial, dict):
                raise ValueError("invalid public field: vote3_partial")
            partial = _select_scalars(partial, {"reason": (str,), "spans": (int,), "seconds": (int, float)})
            if (len(partial) != 3 or partial["reason"] not in ("whisper_incomplete", "asr_incomplete")
                    or partial["spans"] < 0 or partial["seconds"] < 0):
                raise ValueError("invalid public field: vote3_partial")
            result["quality"]["vote3_partial"] = partial
    result["segments"] = []
    for segment in transcript.get("segments", []):
        public = _select_scalars(segment, {"segment_id": (str,), "speaker": (str, type(None)),
                                          "start": (int, float), "end": (int, float),
                                          "text": (str,), "overlap": (bool,)})
        public["words"] = []
        for word in segment.get("words", []):
            entry = _select_scalars(word, {"word_id": (str,), "w": (str,), "s": (int, float),
                                           "e": (int, float)})
            if "provenance" in word:
                entry["provenance"] = _public_provenance(word["provenance"])
            public["words"].append(entry)
        result["segments"].append(public)
    return result


def _select_scalars(value, fields):
    result = {}
    for key, types in fields.items():
        if key in value:
            if type(value[key]) not in types:
                raise ValueError("invalid public field: " + key)
            result[key] = value[key]
    return result


def _public_review(review):
    result = _select_scalars(review, {"review_done": (bool, type(None))})
    result["review"] = []
    for item in review.get("review", []):
        public = _select_scalars(item, {"id": (str,), "status": (str,), "text": (str,),
                                        "t": (int, float), "t1": (int, float), "t2": (int, float)})
        for key in ("reasons", "word_ids"):
            if key in item:
                if not isinstance(item[key], list) or any(not isinstance(x, str) for x in item[key]):
                    raise ValueError("invalid public list: " + key)
                public[key] = list(item[key])
        if "gap" in item:
            gap = item["gap"]
            if (not isinstance(gap, list) or len(gap) != 2 or
                    any(type(x) not in (int, float) for x in gap) or gap[0] > gap[1]):
                raise ValueError("invalid review gap")
            public["gap"] = list(gap)
        result["review"].append(public)
    result["answers"] = {}
    for ident, answer in review.get("answers", {}).items():
        public = _select_scalars(answer, {"id": (str,), "version": (str,), "kind": (str,),
                                          "text": (str,), "source": (str,), "actor_kind": (str,),
                                          "option": (int, type(None)), "ts": (str,),
                                          "legacy_author_label": (str,)})
        if "actor_kind" in public and public["actor_kind"] not in storage.ACTOR_KINDS:
            raise ValueError("invalid answer actor")
        for key in ("author_ref", "initiated_by"):
            if key in answer:
                public[key] = _public_ref(answer[key])
        if "confirmation_context" in answer:
            storage._context(answer["confirmation_context"])
            public["confirmation_context"] = answer["confirmation_context"]
        if (public.get("actor_kind") in ("model", "system", "unknown") and
                public.get("author_ref", {}).get("user_id") is not None):
            raise ValueError("automatic answer cannot have a human author")
        result["answers"][ident] = public
    actions = review.get("applications", [])
    if not isinstance(actions, list):
        raise ValueError("applications must be a list")
    result["applications"] = []
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("invalid application")
        public = _select_scalars(action, {"id": (str,), "kind": (str,), "answer_id": (str,),
                                          "version": (str,), "action": (str,), "outcome": (str,),
                                          "answer_source": (str, type(None)), "answer_ts": (str, type(None)),
                                          "at": (str,)})
        if "answer_id" in public and public["answer_id"] not in result["answers"]:
            raise ValueError("application references missing answer")
        if "context" in action:
            storage._context(action["context"])
            public["context"] = action["context"]
        result["applications"].append(public)
    if "question_version" in review:
        if not isinstance(review["question_version"], str):
            raise ValueError("invalid question version")
        result["question_version"] = review["question_version"]
    if "questions" in review:
        result["questions"] = []
        for question in review["questions"]:
            public = _select_scalars(question, {"id": (str,), "question": (str,), "text": (str,)})
            if "word_ids" in question:
                public["word_ids"] = list(question["word_ids"])
            result["questions"].append(public)
    if "undo" in review:
        if not isinstance(review["undo"], dict):
            raise ValueError("invalid undo")
        result["undo"] = {}
        for ident, value in review["undo"].items():
            if not isinstance(ident, str) or not isinstance(value, dict):
                raise ValueError("invalid undo entry")
            public = _select_scalars(value, {"source_revision": (int,), "review_revision": (int,),
                                            "answer_id": (str,), "kind": (str,)})
            for key in ("author_ref", "initiated_by"):
                if key in value:
                    public[key] = _public_ref(value[key])
            if "context" in value:
                storage._context(value["context"])
                public["context"] = value["context"]
            result["undo"][ident] = public
    return result


def archive_paths(recording_id, document_revision):
    storage._uuid(recording_id)
    if type(document_revision) is not int or document_revision < 1:
        raise ValueError("invalid document revision")
    base = f"history/{recording_id}/document-{document_revision}"
    return {"md": base + "/transcript.md", "json": base + "/transcript.json",
            "manifest": base + "/manifest.json"}


def current_paths(recording_id, basename=None):
    storage._uuid(recording_id)
    basename = recording_id if basename is None else _safe_basename(basename)
    return paths_for_basename(basename)


def _safe_basename(basename):
    if (not isinstance(basename, str) or not basename or basename in (".", "..")
            or Path(basename).name != basename or "/" in basename or "\\" in basename
            or "\0" in basename or any(char in basename for char in '<>:"|?*')
            or basename.endswith((" ", "."))):
        raise ValueError("invalid document basename")
    return basename


def paths_for_basename(basename):
    basename = _safe_basename(basename)
    return {"md": basename + ".transcript.md", "json": basename + ".transcript.json",
            "manifest": basename + ".manifest.json"}


def current_paths_from_document(document, recording_id):
    """Прочитать сохранённые цели из passport, с поддержкой формата П1."""
    raw = document.get("json")
    if not isinstance(raw, (bytes, bytearray)):
        raise storage.StorageError("invalid stored document JSON")
    try:
        payload = json.loads(raw.decode("utf-8"))
        passport = payload["passport"]
        paths = passport.get("current_paths")
    except (UnicodeDecodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise storage.StorageError("invalid stored document paths") from exc
    if paths is None:
        if passport.get("format_version", 1) >= 2:
            raise storage.StorageError("missing stored document paths")
        return current_paths(recording_id)
    if not isinstance(paths, dict) or set(paths) != {"md", "json", "manifest"}:
        raise storage.StorageError("invalid stored document paths")
    checked = {}
    for kind, relative in paths.items():
        try:
            parts = _relative_parts(relative)
        except (TypeError, ValueError) as exc:
            raise storage.StorageError("invalid stored document paths") from exc
        checked[kind] = Path(*parts).as_posix()
    if len(set(checked.values())) != 3:
        raise storage.StorageError("duplicate stored document paths")
    return checked


def _author_refs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("author_ref", "initiated_by"):
                yield _public_ref(item)
            else:
                yield from _author_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _author_refs(item)


def prepare(recording_id, source_revision, review_revision, document_revision,
            transcript, review, source_context, review_context, generated_at,
            *, source_content_context=None, annotation=None, document_paths=None):
    """Вернуть bytes MD/JSON/manifest без I/O; время передаёт вызывающий сервис."""
    storage._uuid(recording_id)
    for version in (source_revision, review_revision, document_revision):
        if type(version) is not int or version < 1:
            raise ValueError("invalid revision")
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("generated_at required")
    storage.validate_pair(transcript, review)
    storage._context(source_context)
    storage._context(review_context)
    if source_content_context is None:
        source_content_context = source_context
    storage._context(source_content_context)
    if annotation is not None:
        if (not isinstance(annotation, dict) or
                set(annotation) != {"text", "source_revision", "context"} or
                not isinstance(annotation["text"], str) or
                type(annotation["source_revision"]) is not int or
                annotation["source_revision"] < 1):
            raise ValueError("invalid annotation")
        storage._context(annotation["context"])
    paths = current_paths(recording_id) if document_paths is None else document_paths
    if not isinstance(paths, dict) or set(paths) != {"md", "json", "manifest"}:
        raise ValueError("invalid document paths")
    paths = {kind: Path(*_relative_parts(value)).as_posix() for kind, value in paths.items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError("document paths must be distinct")
    archives = archive_paths(recording_id, document_revision)
    public_transcript = _public_transcript(transcript)
    names = public_transcript["speaker_names"]
    counts = {}
    segments_by_speaker = {}
    for speaker in set(public_transcript["speakers"]) | set(names):
        counts[speaker] = 0
        segments_by_speaker[speaker] = 0
    for segment in transcript.get("segments", []):
        speaker = segment.get("speaker")
        key = speaker if isinstance(speaker, str) and speaker else None
        counts[key] = counts.get(key, 0) + len(segment.get("words", []))
        segments_by_speaker[key] = segments_by_speaker.get(key, 0) + 1
    total_words = sum(counts.values())
    statistics = {"algorithm": "stored-word-count-v1",
                  "speakers": [{"speaker": key, "display_name": names.get(key, key) if key else None,
                                "words": count, "segments": segments_by_speaker[key],
                                "word_share": count / total_words if total_words else None}
                               for key, count in sorted(counts.items(), key=lambda pair: pair[0] or "")]}
    passport = {"recording_id": recording_id, "source_revision": source_revision,
                "review_revision": review_revision, "document_revision": document_revision,
                "format_version": FORMAT_VERSION, "renderer_version": RENDERER_VERSION,
                "generated_at": generated_at, "storage_backend": "sqlite",
                "current_paths": paths, "archive_paths": archives,
                "source": {key: public_transcript.get(key) for key in
                           ("file", "duration", "recorded_at", "recorded_timezone", "container",
                            "codec", "audio_sha256", "audio_available")}}
    payload = {"passport": passport, "transcript": public_transcript,
               "review": _public_review(review),
               "source_action": source_context, "review_action": review_context,
               "source_content_action": source_content_context,
               "annotation": annotation, "statistics": statistics}
    lines = [f"# {_md(transcript.get('title') or 'Стенограмма')}", "", f"Запись: `{recording_id}`",
             f"Ревизия текста: {source_revision}; разбора: {review_revision}; "
             f"документа: {document_revision}", f"Сформирован: {_md(generated_at)}", "",
             f"Источник: {_md(transcript.get('file', 'не указан'))}; "
             f"длительность: {_md(transcript.get('duration', 'не указана'))}.",
             f"Дата записи: {_md(transcript.get('recorded_at', 'не установлена'))}; "
             f"часовой пояс: {_md(transcript.get('recorded_timezone', 'не установлен'))}.",
             f"Контейнер: {_md(transcript.get('container', 'не указан'))}; "
             f"кодек: {_md(transcript.get('codec', 'не указан'))}; "
             f"SHA-256: {_md(transcript.get('audio_sha256', 'не указан'))}; "
             f"доступность аудио: {_md(transcript.get('audio_available', 'неизвестна'))}.", "",
             "Техническая диагностика, голосовые ответы и полная история здесь не представлены.", "",
             "## Происхождение", "",
             f"Последнее действие с текстом: {_md(source_context['actor_kind'])}; "
             f"автор действия: {_label(source_context['author_ref'])}; "
             f"инициатор: {_label(source_context['initiated_by'])}.",
             f"Действие над исходным снимком содержания: {_md(source_content_context['actor_kind'])}; "
             f"автор действия: {_label(source_content_context['author_ref'])}; "
             f"инициатор: {_label(source_content_context['initiated_by'])}.",
             f"Последнее действие с разбором: {_md(review_context['actor_kind'])}; "
             f"автор действия: {_label(review_context['author_ref'])}; "
             f"инициатор: {_label(review_context['initiated_by'])}.", "",
             "Авторы действий не обязательно являются авторами всего содержания или говорящими.",
             ""]
    quality = public_transcript.get("quality", {})
    if quality.get("asr_incomplete_chunks"):
        lines.extend([f"Внимание: технически не распознано фрагментов: {quality['asr_incomplete_chunks']}.", ""])
    if quality.get("vote3_skipped"):
        lines.extend(["Внимание: голосование трёх моделей было пропущено.", ""])
    if quality.get("vote3_partial"):
        partial = quality["vote3_partial"]
        lines.extend([f"Внимание: голосование трёх моделей пропущено на отрезках с неполным результатом "
                      f"распознавания: {partial['spans']} (около {str(partial['seconds']).replace('.', ',')} с).", ""])
    refs = {}
    for ref in _author_refs(payload):
        if ref["user_id"] is not None:
            refs[(ref["user_id"], ref["user_name_snapshot"])] = ref
    lines += ["Упомянутые авторы и инициаторы действий (исторические имена):"]
    if not refs:
        lines.append("- Неизвестны.")
    for _, ref in sorted(refs.items()):
        lines.append(f"- {_label(ref)} (`{ref['user_id']}`)")
    lines += ["", "## Аннотация", ""]
    if annotation is None:
        lines.append("Не сформирована.")
    else:
        ctx = annotation["context"]
        status = ("актуальность не подтверждена: опорная ревизия отличается"
                  if annotation["source_revision"] != source_revision
                  else "на указанной ревизии; проверка человеком не заявлена")
        lines.extend([_md(annotation["text"]), "",
                      f"Опорная ревизия: {annotation['source_revision']}; {status}.",
                      f"Исполнитель: {_md(ctx['actor_kind'])}; "
                      f"автор действия: {_label(ctx['author_ref'])}; "
                      f"инициатор: {_label(ctx['initiated_by'])}."])
    lines += ["", "## Участники", ""]
    for row in statistics["speakers"]:
        lines.append(f"- {_md(row['display_name']) if row['display_name'] else 'говорящий неизвестен'}")
    if transcript.get("names_source") == "voices" and not transcript.get("names_confirmed"):
        lines.append("Имена определены автоматически и не подтверждены.")
    lines += ["", "## Статистика", "",
             "Число сохранённых слов; имена не доказывают число физических участников.", ""]
    if not counts:
        lines += ["Слова не указаны.", ""]
    for row in statistics["speakers"]:
        name = _md(row["display_name"]) if row["display_name"] else "говорящий неизвестен"
        share = f"{row['word_share']:.1%}" if row["word_share"] is not None else "неизвестна"
        lines.append(f"- {name}: слов {row['words']}, реплик {row['segments']}, доля слов {share}")
    lines += ["", "## Текст", ""]
    lines += ["Сомнительные места и возможные пропуски перечислены в разделе «Сомнения».", ""]
    for segment in transcript.get("segments", []):
        speaker = segment.get("speaker")
        speaker = _md(names.get(speaker, speaker)) if speaker else "говорящий неизвестен"
        words = segment.get("words", [])
        text = segment.get("text")
        if not isinstance(text, str):
            text = " ".join(str(w.get("w", "")) for w in words)
        start = segment.get("start")
        prefix = f"[{_md(start)}] " if start is not None else ""
        anchor = hashlib.sha256(segment["segment_id"].encode("utf-8")).hexdigest()[:24]
        lines.append(f'<a id="segment-{anchor}"></a>')
        lines.append(f"**{prefix}{speaker}:** {_md(text)}")
        lines.append("")
    for item in review.get("review", []):
        if "gap" in item:
            lines.append(f"_Возможный пропуск речи: {_md(item['gap'][0])}–{_md(item['gap'][1])}; "
                         f"место {_md(item['id'])}._")
            lines.append("")
    lines += ["## Сомнения", ""]
    items = review.get("review", [])
    if not items:
        lines.append("Не указаны.")
    for item in items:
        lines.append(f"- {_md(item['id'])}: {_md(item.get('status', 'open'))}; "
                     f"{_md(item.get('text', 'описание не указано'))}")
        if "gap" in item:
            lines.append(f"  Возможный пропуск речи: {_md(item['gap'][0])}–{_md(item['gap'][1])}.")
    md = ("\n".join(lines) + "\n").encode("utf-8")
    data = _json_bytes(payload)
    manifest = {"passport": passport, "source_action": source_context,
                "source_content_action": source_content_context, "review_action": review_context,
                "annotation_context": annotation["context"] if annotation else None,
                "mentioned_authors": [refs[key] for key in sorted(refs)], "files": {
        paths["md"]: {"sha256": _digest(md), "size": len(md)},
        paths["json"]: {"sha256": _digest(data), "size": len(data)}}}
    manifest_bytes = _json_bytes(manifest)
    archive_manifest = _json_bytes({**{k: v for k, v in manifest.items() if k != "files"}, "files": {
        "transcript.md": {"sha256": _digest(md), "size": len(md)},
        "transcript.json": {"sha256": _digest(data), "size": len(data)}}})
    return {"passport": passport, "paths": paths, "archive_paths": archives, "md": md, "json": data,
            "manifest": manifest_bytes, "archive_manifest": archive_manifest,
            "hashes": {"md": _digest(md), "json": _digest(data),
                       "manifest": _digest(manifest_bytes),
                       "archive_manifest": _digest(archive_manifest)}}


def prepare_conspect(title, text, recording_id, document_revision):
    """Синтетический документ с относительной ссылкой на архивную версию."""
    if not isinstance(title, str) or not isinstance(text, str):
        raise ValueError("invalid conspect")
    target = archive_paths(recording_id, document_revision)["md"]
    return (f"# {_md(title)}\n\n{text}\n\n"
            f"Источник: [стенограмма, версия {document_revision}]({target})\n").encode("utf-8")
