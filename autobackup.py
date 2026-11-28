"""БД-4(б): автоматический резерв базы записей раз в рабочий день с прореживанием И-1а (решения владельца 28.11.2026).

Когда. При первом запуске программы (сервер MCP, экран) за рабочий день — до первой записи процесса и до переходов
схемы. Рабочий день момента — местная дата от «время − 4 ч»: записи до 4:00 относятся к предыдущему дню («До 4:00 —
предыдущий», слово владельца). Снимок SQLite backup API — сразу (на базе 119 МБ около секунды); пометка копии
неактивной, validate_copy, сводка, публикация и прореживание — в фоновом потоке.

Дата и пропуск. Время последней записи — самое позднее время по столбцам времени базы (вариант А, без смены схемы);
копия называется рабочим днём этого времени: state-ГГГГ-ММ-ДД.sqlite3. Время равно времени самой свежей копии —
шага нет. Тот же рабочий день, позже — замена копии этого дня (прежняя переименовывается в .replaced-<32 hex>, новая
публикуется, опись, прежняя удаляется). Новый рабочий день — шаг прореживания.

Хранение. Не больше MAX_COPIES копий и не больше db_backup_max_gb ГБ (ГБ = 2**30 байт, как в Проводнике; по
умолчанию 1). Защищены всегда: самая старая и три самые свежие. Прореживание от старых (И-1а): удаляется вторая по
старшинству, дальше через одну к новым, пока следующая не попадёт в три свежие; затем при каждом запуске — вопрос
человеку, а до ответа удаляется четвёртая по свежести (скользят три свежие). Если и четыре защищённые не помещаются
в предел — они хранятся сверх предела, человеку сразу сообщается о пределе (ответ владельца на В-1).

Что удаляется. Только копии из своей описи auto/manifest.json в папке <папка данных>/db-backups/auto, с именем
строго по шаблону, не ссылки и не junction, с совпавшими размером и SHA-256 (разрешение владельца — ответ 5).
Резервы других видов, чужие файлы в auto, база и её служебные файлы не удаляются никогда. Порядок — публикация новой
копии, потом удаление вытесняемой: сбой между ними не теряет ни одной копии.

Защита от вытеснения испорченными (3.4). Сводка копии: число строк основных таблиц и время последней записи. Если
по сравнению с прежней копией что-то уменьшилось или время откатилось (при тех же установках — не после
восстановления), копия подозрительная: она кладётся в auto/suspect, ничего не удаляется, человек предупреждён.

Неудачи (3.1) — в auto/status.json (вид и время, без путей и содержимого); полоса экрана и self_check держат их до
следующего удачного шага. Повреждённая база (quick_check, dbguard) — шага нет и ничего не удаляется.
"""
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dbguard
import storage

MAX_COPIES = 10
GIB = 2 ** 30
DAY_START = timedelta(hours=4)
FRESH = 3                                         # защищённых самых свежих копий
STALE_DAYS = 3
NAME_RE = re.compile(r"^state-(\d{4}-\d{2}-\d{2})\.sqlite3$")
ASIDE_RE = re.compile(r"^state-\d{4}-\d{2}-\d{2}\.sqlite3\.replaced-[0-9a-f]{32}$")
SUSPECT_RE = re.compile(r"^state-\d{4}-\d{2}-\d{2}-[0-9a-f]{8}\.sqlite3$")
COUNT_TABLES = ("recordings", "transcript_revisions", "review_revisions", "operations", "document_versions",
                "conspect_revisions", "review_events")
TIME_COLUMNS = (("operations", "created_at"), ("conspect_revisions", "created_at"),
                ("review_events", "created_at"), ("materialization_events", "created_at"),
                ("hypotheses", "created_at"), ("local_users", "created_at"), ("local_users", "updated_at"),
                ("workflow_requests", "created_at"), ("workflow_requests", "updated_at"),
                ("workflow_policy_versions", "created_at"), ("workflow_assessments", "created_at"),
                ("workflow_verifications", "created_at"), ("workflow_model_parts", "created_at"),
                ("workflow_question_slots", "created_at"), ("workflow_runs", "created_at"),
                ("workflow_runs", "updated_at"), ("workflow_steps", "created_at"), ("workflow_steps", "updated_at"))
FAILURES = {
    "damaged": "база не прошла проверку целостности — резерв не делался, старые копии не тронуты",
    "manifest": "опись автоматических резервов не читается или не совпадает с файлами — резервы остановлены, "
                "ничего не удалено; нужна помощь",
    "space": "не хватило места на диске для новой копии",
    "snapshot": "не удалось снять копию базы",
    "verify": "новая копия не прошла проверку — она не сохранена, прежние копии на месте",
    "publish": "не удалось сохранить новую копию",
    "delete": "не удалось удалить вытесняемую копию (файл занят или защищён) — она осталась",
    "suspect": "новая копия подозрительная: в ней меньше данных или более раннее время, чем в прежней. Она отложена "
               "отдельно, старые копии не удалялись. Если вы ничего не удаляли и не восстанавливали — база могла "
               "потерять данные: проверьте записи и попросите помощи",
    "error": "неожиданный сбой авторезерва",
}


class ManifestError(storage.StorageError):
    """Опись auto/manifest.json не читается или не совпадает с файлами."""


class Failure(Exception):
    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


def _utc(value):
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    if moment.tzinfo is None:                     # время без пояса в базе — UTC (storage._now пишет с поясом)
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def workday(moment, tz=None):
    """Рабочий день момента: местная дата от «время − 4 ч» по стенным часам. 03:59 → предыдущий день, 04:00 → тот же.
    tz — пояс (для тестов); по умолчанию пояс компьютера. -> 'ГГГГ-ММ-ДД'."""
    local = _utc(moment).astimezone(tz)
    return (local.replace(tzinfo=None) - DAY_START).date().isoformat()


def _now(now=None):
    return _utc(now) if now is not None else datetime.now(timezone.utc)


def _columns(db, table):
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def last_write(db):
    """Самое позднее время записи по столбцам времени базы (вариант А), как время, не как строка. None — записей нет."""
    best = None
    for table, column in TIME_COLUMNS:
        if column not in _columns(db, table):
            continue
        for (value,) in db.execute(f"SELECT {column} FROM {table}"):
            moment = _utc(value)
            if moment is not None and (best is None or moment > best):
                best = moment
    return best


def summary(db):
    """Сводка копии (3.4): числа строк основных таблиц, время последней записи, установки (хэш), схема."""
    counts = {}
    for table in COUNT_TABLES:
        if _columns(db, table):
            counts[table] = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    installs = sorted(row[0] for row in db.execute("SELECT installation_id FROM installations"))
    moment = last_write(db)
    return {"counts": counts, "last_write": moment.isoformat() if moment else None,
            "installations": hashlib.sha256("\n".join(installs).encode()).hexdigest(),
            "schema_version": db.execute("SELECT schema_version FROM store_meta").fetchone()[0]}


def suspicious(previous, current):
    """Новая копия подозрительна: при тех же установках уменьшилось число строк или откатилось время."""
    if previous.get("installations") != current.get("installations"):
        return False                              # восстановление или новая установка: уменьшение законно
    if any(current["counts"].get(table, 0) < number for table, number in previous.get("counts", {}).items()):
        return True
    before, after = _utc(previous.get("last_write")), _utc(current.get("last_write"))
    return before is not None and (after is None or after < before)


def _sha256(path):
    return storage._file_sha256(path)


def _is_link(path):
    """Ссылка, junction или точка повторной обработки: такие пути программа не удаляет и по ним не ходит."""
    path = Path(path)
    if os.path.islink(path) or (hasattr(path, "is_junction") and path.is_junction()):
        return True
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & 0x400)               # FILE_ATTRIBUTE_REPARSE_POINT


def _write_json(path, data):
    """Свой файл описи или состояния: временный рядом, сброс на диск, замена (os.replace)."""
    temp = path.with_name(f".{path.stem}-{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class Folder:
    """Папка авторезервов <папка базы>/db-backups/auto: опись, состояние, замок."""

    def __init__(self, store_path):
        self.store_path = Path(store_path)
        self.path = self.store_path.parent / "db-backups" / "auto"
        self.manifest_path = self.path / "manifest.json"
        self.status_path = self.path / "status.json"
        self.suspect_path = self.path / "suspect"
        self._lock = None

    def prepare(self):
        for folder in (self.path.parent, self.path):
            folder.mkdir(exist_ok=True)
            if _is_link(folder):
                raise ManifestError("backup folder is a link")

    # замок: один процесс делает шаг, другой его пропускает
    def try_lock(self):
        from jobslots import _try_lock
        handle = (self.path / ".lock").open("a+b")
        if not _try_lock(handle):
            handle.close()
            return False
        self._lock = handle
        return True

    def unlock(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    # состояние: неудачи, вопрос, отметка рабочего дня; битое — начинается заново (это не данные)
    def load_status(self):
        try:
            data = json.loads(self.status_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_status(self, status):
        _write_json(self.status_path, status)

    def record_failure(self, kind, now=None):
        try:
            self.prepare()
            status = self.load_status()
            status["last_failure"] = {"kind": kind, "at": _now(now).isoformat()}
            self.save_status(status)
        except (OSError, storage.StorageError):
            pass

    # опись: источник правды о своих копиях
    def load_manifest(self):
        if not self.manifest_path.exists():
            if any(NAME_RE.match(p.name) or ASIDE_RE.match(p.name) for p in self.path.iterdir()):
                raise ManifestError("backup files without manifest")
            return {"version": 1, "copies": [], "thin_pointer": 1, "thin_done": False, "suspect": []}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            copies = data["copies"]
            if data.get("version") != 1 or not isinstance(copies, list):
                raise ValueError("version")
            for entry in copies:
                if not NAME_RE.match(entry["name"]) or not isinstance(entry["size"], int) or \
                        len(entry["sha256"]) != 64 or _utc(entry["last_write"]) is None:
                    raise ValueError("entry")
            data.setdefault("thin_pointer", 1)
            data.setdefault("thin_done", False)
            data.setdefault("suspect", [])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ManifestError("backup manifest is unreadable") from exc
        return data

    def save_manifest(self, manifest):
        manifest["copies"].sort(key=lambda entry: _utc(entry["last_write"]))
        _write_json(self.manifest_path, manifest)

    def delete_copy(self, name, size, sha256):
        """Удалить свою копию: имя по шаблону, внутри auto, не ссылка, размер и SHA-256 как в описи."""
        if not (NAME_RE.match(name) or ASIDE_RE.match(name)):
            raise ManifestError("refusing to delete a foreign name")
        target = self.path / name
        if _is_link(self.path) or _is_link(target):
            raise ManifestError("refusing to delete a link")
        if os.path.normcase(os.path.realpath(target.parent)) != os.path.normcase(os.path.realpath(self.path)):
            raise ManifestError("refusing to delete outside the backup folder")
        if target.stat().st_size != size or _sha256(target) != sha256:
            raise ManifestError("copy does not match the manifest")
        target.unlink()


def _newest(manifest):
    copies = [entry for entry in manifest["copies"] if not entry.get("pending")]
    return max(copies, key=lambda entry: _utc(entry["last_write"])) if copies else None


def recover(folder, manifest):
    """Довести или откатить прерванные шаги по описи (сбой процесса между шагами публикации и замены)."""
    changed = False
    for entry in list(manifest["copies"]):
        target = folder.path / entry["name"]
        if entry.get("pending"):                  # новая копия: опубликована ли
            if target.exists() and _sha256(target) == entry["sha256"]:
                entry.pop("pending")
            else:
                manifest["copies"].remove(entry)
            changed = True
            continue
        replacing = entry.get("replacing")
        if replacing:
            aside, new = folder.path / replacing["aside"], replacing["new"]
            if not ASIDE_RE.match(replacing["aside"]):
                raise ManifestError("bad replacement record")
            if target.exists() and _sha256(target) == new["sha256"]:
                manifest["copies"][manifest["copies"].index(entry)] = new
                folder.save_manifest(manifest)
                if aside.exists():
                    folder.delete_copy(replacing["aside"], entry["size"], entry["sha256"])
            elif aside.exists() and not target.exists() and _sha256(aside) == entry["sha256"]:
                os.rename(aside, target)          # вернуть прежнюю копию под её имя
                entry.pop("replacing")
            elif target.exists() and _sha256(target) == entry["sha256"]:
                entry.pop("replacing")
            else:
                raise ManifestError("interrupted replacement cannot be resolved")
            changed = True
            continue
        if not target.is_file() or target.stat().st_size != entry["size"]:
            raise ManifestError("listed copy is missing or changed")
    if changed:
        folder.save_manifest(manifest)


def _protected(copies):
    """Индексы, которые не удаляются никогда: самая старая, три свежие, опора (самая свежая с проверкой
    восстановлением — проверка 3.2 будет следующим шагом линии; сейчас таких копий нет)."""
    count = len(copies)
    keep = {0} | {index for index in range(max(0, count - FRESH), count)}
    verified = [index for index, entry in enumerate(copies) if entry.get("restore_verified")]
    if verified:
        keep.add(verified[-1])
    return keep


def choose_victim(manifest):
    """Какую копию удалить по И-1а, или None (удалять нечего: остались защищённые). -> (индекс, скольжение)."""
    copies = manifest["copies"]
    count, keep = len(copies), _protected(copies)
    while not manifest["thin_done"] and manifest["thin_pointer"] < count - FRESH:
        index = manifest["thin_pointer"]
        manifest["thin_pointer"] += 1
        if index not in keep:
            return index, False
    manifest["thin_done"] = True
    index = count - FRESH - 1
    if index <= 0 or index in keep:
        return None, True
    return index, True


def rotate(folder, manifest, limit_bytes):
    """Удалять по одной, пока копий не больше MAX_COPIES и сумма не больше предела. -> вопрос человеку или None."""
    question = None
    while True:
        copies = manifest["copies"]
        total = sum(entry["size"] for entry in copies)
        if len(copies) <= MAX_COPIES and total <= limit_bytes:
            return question
        before = (manifest["thin_pointer"], manifest["thin_done"])
        index, sliding = choose_victim(manifest)
        if index is None:
            return {"kind": "small_limit"}
        if sliding:
            question = {"kind": "limit"}
        victim = copies[index]
        if index in _protected(copies):           # утверждение в самом коде, не только в тестах
            raise AssertionError("защищённая копия не удаляется")
        try:
            folder.delete_copy(victim["name"], victim["size"], victim["sha256"])
        except OSError as exc:
            manifest["thin_pointer"], manifest["thin_done"] = before     # копия осталась: проход не сдвигается
            folder.save_manifest(manifest)
            raise Failure("delete") from exc
        # после удаления по индексу p следующая «через одну» — индекс p + 1 в укороченном списке (указатель уже там)
        copies.pop(index)
        folder.save_manifest(manifest)


def _limit_bytes(config):
    value = config.get("db_backup_max_gb", 1.0)
    if not isinstance(value, (int, float)) or value <= 0:
        value = 1.0
    return int(value * GIB)


def _question_view(question, manifest, config):
    copies = manifest["copies"]
    if not question or not copies:
        return None
    average = sum(entry["size"] for entry in copies) / len(copies)
    return {"kind": question["kind"], "count": len(copies), "oldest_workday": copies[0]["workday"],
            "limit_gb": round(_limit_bytes(config) / GIB, 2),
            "needed_gb": round(average * MAX_COPIES / GIB + 0.05, 1)}


def _store_path(path):
    if path is None:
        import paths
        path = paths.STATE_DB
    return Path(path)


def start_daily(path=None, *, now=None, tz=None, background=True, config=None):
    """Шаг авторезерва при запуске процесса. Никогда не бросает: итог — словарь с полем state.

    Синхронно: проверка базы, замок, опись, решение и снимок. Остальное — в потоке (background=False — здесь же)."""
    path = _store_path(path)
    if not path.is_file():
        return {"state": "absent"}
    if config is None:
        import settings
        config = settings.load_config()
    if config.get("db_backup_auto", True) is not True:
        return {"state": "off"}
    folder = Folder(path)
    checked = dbguard.ensure_checked(path)
    if checked is not None and checked["ok"] is False:
        folder.record_failure("damaged", now)
        return {"state": "damaged"}
    if checked is None or checked["ok"] is None:
        return {"state": "busy"}                  # база занята: попробует следующий запуск
    try:
        folder.prepare()
        if not folder.try_lock():
            return {"state": "locked"}
    except (OSError, storage.StorageError):
        folder.record_failure("manifest", now)
        return {"state": "failed", "kind": "manifest"}
    try:
        plan = _decide(folder, path, now, tz)
    except Failure as failure:
        folder.unlock()
        folder.record_failure(failure.kind, now)
        return {"state": "failed", "kind": failure.kind}
    except Exception as exc:                      # запуск программы не должен падать из-за резерва
        folder.unlock()
        folder.record_failure("manifest" if isinstance(exc, ManifestError) else "error", now)
        return {"state": "failed", "kind": "manifest" if isinstance(exc, ManifestError) else "error"}
    if plan["state"] != "snapshot":
        folder.unlock()
        return plan

    def finish():
        try:
            return _finish(folder, plan, config, now, tz)
        finally:
            folder.unlock()
    if not background:
        return finish()
    threading.Thread(target=finish, name="autobackup", daemon=True).start()
    return {"state": "started", "name": plan["name"]}


def _decide(folder, path, now, tz):
    status = folder.load_status()
    manifest = folder.load_manifest()
    recover(folder, manifest)
    today = workday(_now(now), tz)
    if status.get("checked_workday") == today:
        return {"state": "done_today"}
    store = storage.open_store(path)              # на запись: снимок под обычными блокировками SQLite
    try:
        live = last_write(store.db)
        newest = _newest(manifest)
        if live is None or (newest is not None and _utc(newest["last_write"]) == live):
            _mark_checked(folder, today, success=True, now=now)
            return {"state": "unchanged"}
        size = path.stat().st_size
        free = shutil.disk_usage(folder.path).free
        if free < size + size // 10 + 64 * 2 ** 20:
            raise Failure("space")
        name = f"state-{workday(live, tz)}.sqlite3"
        staged = folder.path / f"{name}.incomplete-{uuid.uuid4().hex}"
        try:
            store.snapshot_into(staged)
        except (OSError, storage.StorageError, sqlite3.Error) as exc:
            raise Failure("snapshot") from exc
    finally:
        store.close()
    return {"state": "snapshot", "staged": staged, "name": name, "today": today,
            "manifest": manifest}


def _mark_checked(folder, today, *, success, now=None, question=None, keep_question=False):
    status = folder.load_status()
    status["checked_workday"] = today
    if success:
        status["last_success"] = {"at": _now(now).isoformat(), "workday": today}
        status.pop("last_failure", None)
    if not keep_question:
        if question:
            status["question"] = question
        else:
            status.pop("question", None)
    folder.save_status(status)


def _finish(folder, plan, config, now, tz):
    staged, manifest, today = plan["staged"], plan["manifest"], plan["today"]
    try:
        try:
            storage.finish_snapshot(staged, allow_damaged_copies=True)
        except (OSError, storage.StorageError, sqlite3.Error) as exc:
            raise Failure("verify") from exc
        copy = storage.open_store_readonly(staged)
        try:
            facts = summary(copy.db)
        finally:
            copy.close()
        moment = _utc(facts["last_write"])
        entry = {"name": f"state-{workday(moment, tz)}.sqlite3", "workday": workday(moment, tz),
                 "last_write": facts["last_write"], "taken_at": _now(now).isoformat(),
                 "size": staged.stat().st_size, "sha256": _sha256(staged), "summary": facts,
                 "restore_verified": False}
        previous = _newest(manifest)
        if previous is not None and suspicious(previous.get("summary", {}), facts):
            _keep_suspect(folder, manifest, staged, entry)
            _mark_checked(folder, today, success=False, now=now, keep_question=True)
            folder.record_failure("suspect", now)
            return {"state": "suspect"}
        if previous is not None and _utc(previous["last_write"]) == moment:
            storage._remove_unpublished(staged)   # пока шла проверка, ничего нового: свой временный файл
            _mark_checked(folder, today, success=True, now=now, keep_question=True)
            return {"state": "unchanged"}
        if previous is not None and previous["workday"] == entry["workday"]:
            _replace(folder, manifest, previous, staged, entry)
            outcome = "replaced"
        else:
            _publish(folder, manifest, staged, entry)
            outcome = "added"
        question = rotate(folder, manifest, _limit_bytes(config))
        _mark_checked(folder, today, success=True, now=now, question=_question_view(question, manifest, config))
        return {"state": outcome, "name": entry["name"], "question": question}
    except Failure as failure:
        storage._remove_unpublished(staged)
        folder.record_failure(failure.kind, now)
        return {"state": "failed", "kind": failure.kind}
    except Exception as exc:
        storage._remove_unpublished(staged)
        kind = "manifest" if isinstance(exc, ManifestError) else "error"
        folder.record_failure(kind, now)
        print(f"[autobackup] неудача: {kind} ({type(exc).__name__})", file=sys.stderr, flush=True)
        return {"state": "failed", "kind": kind}


def _publish(folder, manifest, staged, entry):
    """Новая копия: запись в описи «ждёт» → публикация без замены → запись действующая."""
    target = folder.path / entry["name"]
    manifest["copies"].append({**entry, "pending": True})
    folder.save_manifest(manifest)
    try:
        renamed = storage._publish_new(staged, target)
    except OSError as exc:
        manifest["copies"] = [item for item in manifest["copies"] if not item.get("pending")]
        folder.save_manifest(manifest)
        raise Failure("publish") from exc
    if not renamed:
        staged.unlink()
    for item in manifest["copies"]:
        item.pop("pending", None)
    folder.save_manifest(manifest)


def _replace(folder, manifest, previous, staged, entry):
    """Замена копии того же рабочего дня (9.3): прежняя → .replaced-<hex>, новая → имя дня, опись, прежняя удаляется.
    Сбой между шагами: прежняя копия лежит под .replaced-* и в описи действующая, recover доведёт или вернёт."""
    aside = f"{previous['name']}.replaced-{uuid.uuid4().hex}"
    previous["replacing"] = {"aside": aside, "new": entry}
    folder.save_manifest(manifest)
    target = folder.path / previous["name"]
    try:
        os.rename(target, folder.path / aside)
    except OSError as exc:
        previous.pop("replacing")
        folder.save_manifest(manifest)
        raise Failure("publish") from exc
    try:
        renamed = storage._publish_new(staged, target)
    except OSError as exc:
        os.rename(folder.path / aside, target)
        previous.pop("replacing")
        folder.save_manifest(manifest)
        raise Failure("publish") from exc
    if not renamed:
        staged.unlink()
    index = manifest["copies"].index(previous)
    if previous.get("restore_verified"):          # опору не удаляем: остаётся под своим вторым именем
        kept = {key: value for key, value in previous.items() if key != "replacing"}
        manifest["copies"][index] = entry
        manifest["copies"].append({**kept, "name": aside})
        folder.save_manifest(manifest)
        return
    manifest["copies"][index] = entry
    folder.save_manifest(manifest)
    try:
        folder.delete_copy(aside, previous["size"], previous["sha256"])
    except OSError as exc:
        raise Failure("delete") from exc


def _keep_suspect(folder, manifest, staged, entry):
    folder.suspect_path.mkdir(exist_ok=True)
    name = f"state-{entry['workday']}-{uuid.uuid4().hex[:8]}.sqlite3"
    try:
        renamed = storage._publish_new(staged, folder.suspect_path / name)
    except OSError as exc:
        raise Failure("publish") from exc
    if not renamed:
        staged.unlink()
    manifest["suspect"].append({**entry, "name": name})
    folder.save_manifest(manifest)


def answer(choice, limit_gb=None, path=None):
    """Ответ человека на вопрос о пределе (экран). raise — поднять предел до limit_gb ГБ; thin — проредить ещё
    раз (новый проход того же правила); keep — оставить как есть (вопрос повторится при следующем запуске)."""
    folder = Folder(_store_path(path))
    if choice == "keep":
        return {"answered": "keep"}
    if choice not in ("raise", "thin"):
        raise ValueError("неизвестный ответ")
    folder.prepare()
    if not folder.try_lock():
        raise storage.Busy("резерв сейчас делается — повторите через минуту")
    try:
        if choice == "raise":
            import settings
            current = _limit_bytes(settings.load_config()) / GIB
            if not isinstance(limit_gb, (int, float)) or isinstance(limit_gb, bool) or limit_gb <= current:
                raise ValueError(f"новый предел должен быть больше нынешнего ({current:g} ГБ)")
            settings.update(db_backup_max_gb=float(limit_gb))
        else:
            manifest = folder.load_manifest()
            manifest["thin_done"], manifest["thin_pointer"] = False, 1
            folder.save_manifest(manifest)
        status = folder.load_status()
        status.pop("question", None)
        folder.save_status(status)
        return {"answered": choice}
    finally:
        folder.unlock()


def status_view(path=None):
    """Для экрана, self_check и журнала: {"lines": [...], "question": {...}|None, "last_success": ...}; без путей."""
    path = _store_path(path)
    folder = Folder(path)
    if not folder.path.is_dir():
        return {"lines": [], "question": None, "last_success": None, "copies": 0}
    status = folder.load_status()
    lines = []
    failure = status.get("last_failure")
    if failure:
        lines.append("Автоматический резерв базы: " + FAILURES.get(failure.get("kind"), FAILURES["error"]) +
                     f" (с {str(failure.get('at', ''))[:16].replace('T', ' ')} UTC).")
    question = status.get("question")
    if question:
        if question["kind"] == "small_limit":
            lines.append(f"Резервы базы не помещаются в предел {question['limit_gb']:g} ГБ даже вчетвером: самая "
                         "старая и три свежие копии хранятся сверх предела. Поднимите предел.")
        else:
            lines.append(f"Резервы базы достигли предела {question['limit_gb']:g} ГБ: хранится {question['count']} "
                         f"копий, самая старая — за {question['oldest_workday']}. Поднять предел (для 10 копий нужно "
                         f"около {question['needed_gb']:g} ГБ), проредить ещё раз (интервал между старыми копиями "
                         "станет вдвое больше) или оставить как есть — спрошу при следующем запуске. Отвечает человек "
                         "на экране программы.")
    try:
        copies = len(folder.load_manifest()["copies"])
    except (ManifestError, OSError):
        copies = None
    return {"lines": lines, "question": question, "last_success": status.get("last_success"), "copies": copies}


def self_check_report(path=None, now=None, tz=None):
    """Часть самопроверки: последний удачный шаг, последняя неудача, вопрос; давность резерва (3.1)."""
    view = status_view(path)
    out = {"copies": view["copies"], "last_success": view["last_success"], "lines": list(view["lines"]),
           "question": view["question"]}
    path = _store_path(path)
    if path.is_file() and view["copies"]:
        try:
            newest = _newest(Folder(path).load_manifest())
            store = storage.open_store_readonly(path)
            try:
                live = last_write(store.db)
            finally:
                store.close()
            if newest is not None and live is not None:
                gap = (datetime.fromisoformat(workday(live, tz)) - datetime.fromisoformat(newest["workday"])).days
                if gap > STALE_DAYS:
                    out["lines"].append(f"Автоматического резерва нет уже {gap} дн. с изменениями базы: последняя "
                                        f"копия — за {newest['workday']}. Перезапустите программу; если не поможет — "
                                        "нужна помощь.")
        except (storage.StorageError, OSError, ValueError):
            pass
    out["ok"] = True                              # авторезерв не делает самопроверку проваленной: строки — для человека
    return out


def start_at_launch(who):
    """Вызов при запуске сервера и экрана: шаг и строки журнала (stderr), без путей. Не бросает."""
    try:
        result = start_daily()
        view = status_view()
    except Exception as exc:                      # запуск программы важнее резерва
        print(f"[{who}] авторезерв базы: сбой {type(exc).__name__}", file=sys.stderr, flush=True)
        return None
    if result.get("state") not in ("absent", "off", "done_today", "unchanged"):
        print(f"[{who}] авторезерв базы: {result.get('state')}", file=sys.stderr, flush=True)
    for line in view["lines"]:
        print(f"[{who}] {line}", file=sys.stderr, flush=True)
    return result
