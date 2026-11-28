"""Конфликты внешней правки, резерв базы, результаты записи и подтверждение ответов. Перенесено из core.py
без смены логики (разрезка core.py, перенос 3).

Прежние имена `core.X` работают: core.py в конце файла переотдаёт их отсюда. Всё, что раздел берёт из core.py вне
своего раздела (`CoreError` и др.), берётся как `core.X` в момент вызова, а не копируется. Функции и константа раздела,
которые тесты подменяют через core (`_anchors`, `_program_version`, `CONFLICT_DIFF_WORK`), и внутри раздела
вызываются как `core.X`: так подмена действует и на внутренние вызовы. Статус задач (`live`, `status`, `job_json`
и др.) остался в core.py. `import core` — последняя строка файла (круговой импорт: core.py в конце загружает этот
модуль). Проверяет tests/test_core_split.py.
"""
import bisect
import collections
import difflib
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


CONFLICT_SCOPES = ("document", "conspect")
CONFLICT_PREVIEW_CHARS = 20000        # знаков копии в предпросмотре
CONFLICT_SHOWN_LINES, CONFLICT_SHOWN_CHARS = 200, 500   # строк различий и знаков в строке на показ
CONFLICT_DIFF_CELLS = 90000    # сколько пар строк сравнивать точно на одном участке; для большего ищутся опоры
CONFLICT_DIFF_WORK = 2000000   # бюджет уточнения сверх первичного поиска опор, пропорционального входу

_KNOWN_CONFLICTS = {
    "stale revisions": "Запись изменилась после показа. Обновите её и повторите действие.",
    "unresolved document conflict": "Сначала решите конфликт внешней правки документа.",
    "unresolved conspect conflict": "Сначала решите конфликт внешней правки свёртки.",
    "conflict already resolved": "Этот конфликт уже решён. Обновите список.",
    "restored copy is inactive": "Восстановленная копия базы не активирована.",
    "restore copy requires explicit rebind": "Восстановленная копия требует явного подключения папок.",
    "external file too large to preserve; manual resolution required":
        "Файл слишком велик для сохранения копии: нужен отдельный ручной разбор.",
}


def say_error(ex):
    """Текст отказа для человека и агента: известные английские коды хранилища переведены, остальное как было.

    Русские исходы записи (documents.NotWritten, documents.NotSynced) уже со словами и идут без изменений."""
    import storage
    if isinstance(ex, storage.Conflict):
        return _KNOWN_CONFLICTS.get(str(ex), str(ex))
    return str(ex)


def explain_error(ex):
    """Русское пояснение сбоя хранилища или базы для экрана; неизвестный сбой — с исходной причиной."""
    import storage
    text = str(ex)
    low = text.lower()
    if isinstance(ex, storage.Busy) or (isinstance(ex, sqlite3.OperationalError)
                                       and ("locked" in low or "busy" in low)):
        return "База записей сейчас занята другой операцией. Подождите немного и повторите."
    if isinstance(ex, storage.Conflict):
        known = _KNOWN_CONFLICTS
        return known.get(text, text if re.search("[А-Яа-яЁё]", text)
                         else f"Действие не выполнено: состояние записи не позволяет его выполнить (причина: {text}).")
    if isinstance(ex, storage.StorageError) and re.search("[А-Яа-яЁё]", text):
        return text    # собственный отказ программы уже по-русски: прежний текст, не «ошибка базы»
    if "cannot open database" in low or "unable to open database" in low:
        return ("Не удалось открыть базу записей: проверьте, что папка данных на месте и база не занята другой "
                f"программой (причина: {text}).")
    if "database is busy" in low:
        return "База записей сейчас занята другой операцией. Подождите немного и повторите."
    if "hash mismatch" in low:
        return ("Контрольная сумма сохранённых данных не совпала: строка в базе повреждена "
                f"(причина: {text}). Такие данные программа не использует.")
    if isinstance(ex, (sqlite3.Error, storage.StorageError)):
        return f"Ошибка базы записей (причина: {text})."
    if isinstance(ex, OSError):
        return f"Не удалось обратиться к файлу или папке (причина: {text})."
    return f"Непредвиденная ошибка программы (причина: {text})."


def _real_case(path, cache):
    """Путь в том регистре, в каком он лежит на диске (база хранит его приведённым к нижнему); только для показа.

    Что не нашлось на диске, остаётся как есть."""
    path = Path(path)
    parts = path.parts
    if not path.is_absolute() or len(parts) < 2:
        return path
    current, shown = Path(parts[0]), [parts[0][:1].upper() + parts[0][1:]]
    for number, part in enumerate(parts[1:]):
        if current not in cache:
            try:
                names = os.listdir(current)
            except OSError:
                names = []
            folded = {}
            for name in names:              # при нескольких совпадениях — первое, как раньше при переборе
                folded.setdefault(name.casefold(), name)
            cache[current] = (set(names), folded)
        exact, folded = cache[current]
        found = part if part in exact else folded.get(part.casefold())
        if found is None:
            shown.extend(parts[1 + number:])
            break
        shown.append(found)
        current = current / found
    return Path(*shown)


_CONFLICT_COLUMNS = ("f.kind, f.relative_path, f.observed_bytes IS NOT NULL AS preserved, f.observed_hash, "
                     "f.observed_size, f.resolution_operation_id IS NOT NULL AS resolved, f.rowid AS seq, "
                     "o.root_id, (SELECT path FROM allowed_roots WHERE root_id=o.root_id) AS root_path, "
                     "rec.location")
_CONFLICT_SQL = {
    "document": ("SELECT f.conflict_id, f.recording_id, f.document_revision AS revision, NULL AS conspect_id, "
                 "NULL AS name, " + _CONFLICT_COLUMNS + " FROM document_conflicts f "
                 "JOIN document_outbox o ON o.recording_id=f.recording_id "
                 "AND o.document_revision=f.document_revision "
                 "JOIN recordings rec ON rec.recording_id=f.recording_id"),
    "conspect": ("SELECT f.conflict_id, c.recording_id, f.revision, f.conspect_id, c.name, "
                 + _CONFLICT_COLUMNS + " FROM conspect_conflicts f "
                 "JOIN conspects c ON c.conspect_id=f.conspect_id "
                 "JOIN conspect_outbox o ON o.conspect_id=f.conspect_id AND o.revision=f.revision "
                 "JOIN recordings rec ON rec.recording_id=c.recording_id")}


def _conflict_rows(store, scope, where="", args=()):
    """Конфликты одной области без самих байт: при просмотре списка копии не читаются."""
    rows = []
    for found in store.db.execute(_CONFLICT_SQL[scope] + where + " ORDER BY f.rowid", args):
        row = dict(found, scope=scope, preserved=bool(found["preserved"]), resolved=bool(found["resolved"]))
        row["object"] = (scope, row["conspect_id"] or row["recording_id"])
        rows.append(row)
    return rows


def _conflict_row(store, recording_id, conflict_id, scope):
    import storage
    storage._uuid(recording_id)
    storage._uuid(conflict_id)
    if scope not in CONFLICT_SCOPES:
        raise core.CoreError("Неизвестный вид конфликта.")
    rows = _conflict_rows(store, scope, " WHERE f.conflict_id=?", (conflict_id,))
    if not rows or rows[0]["recording_id"] != recording_id:
        raise core.CoreError("Конфликт не найден для этой записи.")
    return rows[0]


def _observed_bytes(store, row):
    """Сохранённая копия человека целиком или None, если файл был слишком велик."""
    import documents
    found = store.db.execute(f"SELECT observed_bytes FROM {row['scope']}_conflicts WHERE conflict_id=?",
                             (row["conflict_id"],)).fetchone()[0]
    if found is None:
        return None
    found = bytes(found)
    if len(found) != row["observed_size"] or documents._digest(found) != row["observed_hash"]:
        raise core.CoreError("Сохранённая копия повреждена: её размер или контрольная сумма не совпали.")
    return found


def _program_version(store, row):
    """Последняя версия программы для файлов объекта конфликта: (ревизия, {вид файла: байты})."""
    import storage
    try:
        if row["scope"] == "document":
            revision = store.db.execute("SELECT max(document_revision) FROM document_versions WHERE recording_id=?",
                                        (row["recording_id"],)).fetchone()[0]
            saved = store.get_document(row["recording_id"], revision)
            return revision, {kind: bytes(saved[kind]) for kind in ("md", "json", "manifest")}
        saved = store.get_conspect(row["conspect_id"])
        return saved["revision"], {kind: bytes(saved[kind]) for kind in ("md", "json")}
    except storage.Busy as ex:
        raise core.CoreError(explain_error(ex)) from ex
    except sqlite3.Error as ex:
        raise core.CoreError(explain_error(ex)) from ex
    except (storage.StorageError, ValueError, KeyError, TypeError) as ex:
        raise core.CoreError("Сохранённая версия программы для этого файла повреждена или не читается "
                        f"(причина: {ex}): заменять файл нечем. Копию вашей правки можно скачать.") from ex


def _file_at(store, row, relative):
    """Файл на диске без записи и без создания папок: (missing|unavailable|ok, sha256 или None)."""
    import documents
    import storage
    try:
        root = documents._authorized_root(store, row["root_id"], row["root_path"])
        path, exists = documents._read_only_target(root, relative)
        return ("ok", documents._file_digest(path)[0]) if exists else ("missing", None)
    except FileNotFoundError:
        return "missing", None
    except (OSError, ValueError, storage.StorageError):
        return "unavailable", None


def _queue_unfinished(store, row):
    """Есть ли у объекта конфликта запись файлов, которую программа не довела до конца."""
    table, column = (("document_outbox", "recording_id") if row["scope"] == "document"
                     else ("conspect_outbox", "conspect_id"))
    return store.db.execute(f"SELECT 1 FROM {table} WHERE {column}=? AND state!='synced'",
                            (row["object"][1],)).fetchone() is not None


def _shown_file(row, cache):
    """Полный путь файла для показа: в регистре, в каком он лежит на диске."""
    return _real_case(Path(row["root_path"]) / row["relative_path"], cache)


def _copy_name(row):
    """Имя, под которым скачивается копия человека."""
    path = Path(row["relative_path"])
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", f"{path.stem} (моя правка){path.suffix}")


def _missing_text(row, then):
    """Файла нет: что сделать, чтобы он появился на месте, и под каким именем."""
    name = Path(row["relative_path"]).name
    folder = _shown_file(row, {}).parent
    return (f"Файла «{name}» на прежнем месте нет. Скачайте копию («Скачать мою правку»: она сохранится как "
            f"«{_copy_name(row)}»), переименуйте её в «{name}» и положите в папку {folder}, {then}")


def _same_object(row, rows, cache):
    """Конфликты того же объекта; rows группируются один раз на построение списка."""
    key = ("same_object", row["object"])
    if key not in cache:
        for other in rows:
            cache.setdefault(("same_object", other["object"]), []).append(other)
    return cache[key]


def _listed_program_version(store, row, cache):
    """Версия программы для списка: (ревизия, {вид файла: хэш}); читается один раз на объект, ошибка
    запоминается так же, как успех. Кеш держит только хэши, не содержимое: построение списка не копит
    файлы всех объектов в памяти. Замена файла и запись по-прежнему читают версию заново (_program_version)."""
    import documents
    key = ("program", row["object"])
    if key not in cache:
        try:
            revision, files = core._program_version(store, row)
            cache[key] = ((revision, {kind: documents._digest(data) for kind, data in files.items()}), None)
        except core.CoreError as ex:
            cache[key] = (None, ex)
    found, error = cache[key]
    if error is not None:
        raise error
    return found


def _conflict_item(store, row, rows, cache):
    """Строка общего списка; rows — все конфликты той же области. Повреждённая строка не роняет список.
    cache живёт одно построение списка: в нём регистр путей, группы объектов и версии программы."""
    import storage
    item = {"conflict_id": row["conflict_id"], "scope": row["scope"], "recording_id": row["recording_id"],
            "title": _real_case(row["location"], cache).name, "name": row["name"], "kind": row["kind"],
            "relative_path": row["relative_path"], "file": str(_shown_file(row, cache)),
            "observed_size": row["observed_size"], "preserved": row["preserved"], "status": "open",
            "can_replace": False, "can_retry": False, "problem": None,
            "copy_problem": None, "next_effect": None, "next_step": None}
    try:
        same_object = _same_object(row, rows, cache)
        status = "resolved" if row["resolved"] else "open"
        last_of_file = row["seq"] == max(other["seq"] for other in same_object
                                         if other["relative_path"] == row["relative_path"])
        equal_to_program = False
        if not row["resolved"] and not row["kind"].startswith("archive_"):
            program = _listed_program_version(store, row, cache)[1]   # повреждена — строка «повреждена»
            if not row["preserved"]:    # копии нет: конфликт закрывается, только если файл уже равен версии программы
                equal_to_program = _file_at(store, row, row["relative_path"])[1] == program.get(row["kind"])
        if row["resolved"] and last_of_file and _queue_unfinished(store, row):
            program = _listed_program_version(store, row, cache)[1].get(row["kind"])
            if program is not None and _file_at(store, row, row["relative_path"])[1] != program:
                status = "not_written"
        item.update(status=status,
                    can_replace=status == "open" and (row["preserved"] or equal_to_program)
                    and not row["kind"].startswith("archive_"),
                    can_retry=status == "not_written" and not any(not other["resolved"] for other in same_object))
        if status in ("open", "not_written"):    # у истории копии не читаются: их проверит скачивание
            item["copy_problem"] = _copy_problem(store, row)
            if item["copy_problem"]:
                item["can_replace"] = False
        if status == "not_written":
            item["next_effect"], item["next_step"] = _next_step(store, row, same_object, item["copy_problem"])
    except (core.CoreError, storage.StorageError, sqlite3.Error, ValueError, KeyError, TypeError, OSError) as ex:
        item.update(status="damaged", can_replace=False, can_retry=False,
                    problem=str(ex) if isinstance(ex, core.CoreError) else
                    "Эту строку не удалось проверить, замена и повтор для неё выключены. " + explain_error(ex))
    return item


def list_all_conflicts():
    """Открытые и решённые конфликты всей базы для экрана; ничего не пишет и не требует запроса."""
    import paths
    import recordings
    import settings
    try:
        backend = settings.recording_backend()
    except ValueError:
        backend = None
    if not Path(paths.STATE_DB).is_file():
        reason = ("Базы записей ещё нет: конфликтов нет, показывать нечего.")
        if backend != "sqlite":
            reason += (" Программа сейчас хранит записи в файлах вне базы: такие файлы она перезаписывает "
                       "без вопроса и затирает вашу правку молча, защиты этого раздела для них нет.")
        return {"available": False, "items": [], "reason": reason, "backend": backend}
    store = recordings.open_store()
    try:
        store.current_identity()
        by_scope = {scope: _conflict_rows(store, scope) for scope in CONFLICT_SCOPES}
        cache = {}
        return {"available": True, "reason": None, "backend": backend,
                "items": [_conflict_item(store, row, rows, cache) for rows in by_scope.values() for row in rows]}
    finally:
        store.close()


def _refusal(row, state):
    """Почему заменять нельзя (None — можно)."""
    if row["resolved"]:
        return "Этот конфликт уже решён: заменять нечего. Копию можно скачать."
    if row["kind"].startswith("archive_"):
        return ("Это архивная копия прежней версии: программа записывает её один раз и не меняет, поэтому "
                "заменить её здесь нельзя. Свою версию можно скачать. Решить такой конфликт на этом экране нельзя.")
    if not row["preserved"] and state != "program":
        return ("Файл больше 8 МиБ, поэтому программа не сохранила его копию и не заменяет его: он остаётся "
                "на диске как есть. Заменить его здесь нельзя. Когда файл снова станет равен версии программы "
                "(её копия в истории записи), здесь можно будет закрыть конфликт: файл при этом не изменится.")
    if state == "missing":
        return _missing_text(row, "обновите список, и замена станет доступна.")
    if state == "unavailable":
        return ("Файл сейчас нельзя прочитать: он занят другой программой, недоступен или на его месте папка "
                "либо ссылка. Освободите его и обновите список.")
    return None


def _also_replaced(store, row, program):
    """Другие файлы объекта, которые это подтверждение запишет заново по прежнему решению:
    они равны решённой копии и не равны версии программы."""
    import documents
    decided = {}
    for other in _conflict_rows(store, row["scope"]):
        if (other["object"] == row["object"] and other["revision"] == row["revision"] and other["resolved"]
                and other["kind"] != row["kind"] and other["kind"] in program):
            decided[other["kind"]] = other    # последнее решение вида: так выбирает и запись файлов
    found = []
    for kind, other in decided.items():
        state, digest = _file_at(store, other, other["relative_path"])
        if (state == "ok" and digest == other["observed_hash"] and digest != documents._digest(program[kind])
                and not _copy_problem(store, other)):
            found.append(other["relative_path"])
    return sorted(found)


def _waits_for(row, rows):
    """Другие открытые конфликты того же объекта: пока они есть, подтверждение файл не меняет."""
    return sorted({Path(other["relative_path"]).name for other in rows
                   if other["object"] == row["object"] and other["conflict_id"] != row["conflict_id"]
                   and not other["resolved"]})


def _copy_problem(store, row):
    """Почему сохранённая копия непригодна; None — цела или её не сохраняли."""
    if not row["preserved"]:
        return None
    try:
        _observed_bytes(store, row)
    except core.CoreError as ex:
        return str(ex)
    return None


def _set_stops(store, row, rows, program, row_state):
    """Файлы с принятым решением, на которых запись встанет, не дойдя до файла row: (все такие, из них изменённые).

    Каждый сверяется тем же правилом, что и при записи (`documents.file_verdict`): изменён после решения — запись
    откроет по нему новый конфликт; повреждена копия решения — встанет без конфликта. row_state — состояние файла row.
    Порядок отказов документа — `documents.stopped_by`: файл раньше row останавливает запись всегда, позже — только
    если row не изменён после решения (изменённый row даёт новый конфликт раньше). У свёртки правки ищутся по всем
    файлам сразу, порядка нет. Файл, исчезнувший после решения, здесь не назван (как и раньше)."""
    import documents
    decided = {}
    for other in rows:
        if (other["object"] != row["object"] or not other["resolved"] or other["revision"] > row["revision"]
                or other["kind"] not in program):
            continue
        same_file = (other["revision"], other["relative_path"]) == (row["revision"], row["relative_path"])
        if same_file and not row["resolved"]:
            continue    # новое решение заменит прежнее решение по этому же файлу
        decided[(other["revision"], other["relative_path"])] = other
    table, column, revision = (("document_outbox", "recording_id", "document_revision") if row["scope"] == "document"
                               else ("conspect_outbox", "conspect_id", "revision"))
    found = []    # (решение, изменён ли файл после решения)
    own = (row["revision"], row["relative_path"])
    for other in decided.values():
        state, digest = _file_at(store, other, other["relative_path"])
        program_hash = documents._digest(program[other["kind"]])
        if state != "ok" or digest == program_hash:
            continue
        if not store.db.execute(f"SELECT 1 FROM {table} WHERE {column}=? AND {revision}=? AND state!='synced'",
                                (row["object"][1], other["revision"])).fetchone():
            continue
        if digest == other["observed_hash"]:    # принятые байты: встанет, только если копия решения повреждена
            if _copy_problem(store, other):
                found.append((other, False))
        elif ((other["revision"], other["relative_path"]) != own    # свой файл: его состояние — row_state
              and documents.file_verdict(digest, program_hash, None, other) == "external"):
            found.append((other, True))
    if row["scope"] == "document":
        verdicts = {other["kind"]: "external" if changed else "damaged_copy" for other, changed in found
                    if other["revision"] == row["revision"] and (other["revision"], other["relative_path"]) != own}
        verdicts[row["kind"]] = "accepted" if row_state == "same" else "refused"
        allowed = set(documents.stopped_by(verdicts, row["kind"]))
        found = [(other, changed) for other, changed in found
                 if other["revision"] != row["revision"] or (other["revision"], other["relative_path"]) == own
                 or other["kind"] in allowed]
    return (sorted({Path(other["relative_path"]).name for other, _ in found}),
            sorted({Path(other["relative_path"]).name for other, changed in found if changed}))


def _stops_words(stops, changed):
    """Чем остановится запись, словами: повреждённые копии и файлы, изменённые после решения."""
    damaged = [name for name in stops if name not in changed]
    parts = []
    if damaged:
        parts.append("повреждённая копия файла " + ", ".join(f"«{name}»" for name in damaged))
    if changed:
        parts.append("файл " + ", ".join(f"«{name}»" for name in changed) + ": его изменили после принятого решения, "
                     "и программа покажет по нему новый конфликт")
    return "; ".join(parts)


def _effect(row, state, waits, stops):
    """Что случится с файлом при записи по решению:
    replace — заменён; close — уже равен версии программы; recopy — изменён, будет новый конфликт;
    wait — ждёт другого конфликта; stopped — сейчас записать нельзя."""
    if state == "program":
        return "close"
    if waits:
        return "wait"
    if state in ("missing", "unavailable"):
        return "stopped"
    if state == "changed" and (row["scope"] == "conspect" or not stops):
        return "recopy"    # у свёртки внешние правки ищутся раньше проверки копий
    return "stopped" if stops else "replace"


def replacement_effect(recording_id, conflict_id, kind):
    """Для окна MCP: что произойдёт после «Да»; если заменять нельзя — отказ до окна. Ничего не записывает."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        row = _conflict_row(store, recording_id, conflict_id, kind)
        problem = _copy_problem(store, row)
        if problem:
            raise core.CoreError(problem + " Заменить такой файл нельзя: нужен отдельный ручной разбор.")
        if _file_at(store, row, row["relative_path"])[0] == "missing":
            raise core.CoreError(_missing_text(row, "затем повторите."))
        facts = _conflict_facts(store, row)
        if not facts["can_replace"]:
            raise core.CoreError(facts["reason"])
        blocked = bool(_unresolvable_wait(store, row, _conflict_rows(store, kind)))
        return {"effect": facts["effect"], "waits": facts["waits_for"], "stops": facts["stops_for"],
                "stops_changed": facts["stops_changed"],
                "changed": facts["file_state"] == "changed", "blocked": blocked}
    finally:
        store.close()


def _conflict_facts(store, row):
    """Что известно о конфликте без чтения текста: файл, версия программы, можно ли заменить, снимок."""
    import documents
    archive = row["kind"].startswith("archive_")
    revision, program, problem = None, None, None
    if not archive:
        try:
            revision, program = core._program_version(store, row)
        except core.CoreError as ex:
            problem = str(ex)
    program_hash = None if program is None else documents._digest(program[row["kind"]])
    found, digest = _file_at(store, row, row["relative_path"])
    state = (found if found != "ok" else "program" if digest == program_hash
             else "same" if digest == row["observed_hash"] else "changed")
    reason = _refusal(row, state) or problem
    also = [] if reason else _also_replaced(store, row, program)
    rows = [] if reason else _conflict_rows(store, row["scope"])
    waits = [] if reason else _waits_for(row, rows)
    stops, changed = ([], []) if reason else _set_stops(store, row, rows, program, state)
    return {"program": None if program is None else program[row["kind"]], "file_state": state,
            "also_replaces": also, "waits_for": waits, "stops_for": stops, "stops_changed": changed,
            "effect": None if reason else _effect(row, state, waits, stops), "can_replace": reason is None, "reason": reason,
            "snapshot": None if reason else {
                "scope": row["scope"], "recording_id": row["recording_id"], "conflict_id": row["conflict_id"],
                "observed_hash": row["observed_hash"], "program_revision": revision,
                "program_hash": program_hash, "also_replaces": also, "waits_for": waits,
                "stops_for": stops}}    # причины остановки в снимок не входят: состав снимка прежний


def _lines(text):
    """Строки по CRLF, CR, LF (не str.splitlines: она прячет U+2028 и разрыв страницы)."""
    found = re.split(r"\r\n|\r|\n", text)
    return found[:-1] if found[-1] == "" else found


def _decode_copy(data):
    """Текст копии для сравнения: (текст, кодировка, точно ли). Кодировку без признака не угадываем."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), "utf-16", True
        except UnicodeDecodeError:
            pass
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="surrogateescape"), "unknown", False
    if "\x00" in text:    # нулевые знаки в тексте — признак UTF-16 без метки: кодировку не угадываем
        return text, "unknown", False
    return text, "utf-8", True


def _show_text(text):
    """Текст для показа: недекодируемые байты и нулевые знаки — знаком замены."""
    return re.sub("[\udc80-\udcff\x00]", "�", text)


def _anchors(a, b):
    """Строки, которые встречаются по одному разу и в a, и в b, в одном порядке: опоры сравнения."""
    count_a, count_b = collections.Counter(a), collections.Counter(b)
    place = {line: j for j, line in enumerate(b) if count_b[line] == 1}
    pairs = [(i, place[line]) for i, line in enumerate(a) if count_a[line] == 1 and line in place]
    tails, ends, back = [], [], []
    for k, (_, j) in enumerate(pairs):
        at = bisect.bisect_left(tails, j)
        back.append(ends[at - 1] if at else None)
        if at == len(tails):
            tails.append(j)
            ends.append(k)
        else:
            tails[at], ends[at] = j, k
    chain, k = [], ends[-1] if ends else None
    while k is not None:
        chain.append(pairs[k])
        k = back[k]
    return chain[::-1]


def _diff_lines(a, b, *, details=None):
    """Строки a, которых нет на своём месте в b (gone), и строки b, которых нет в a (came).

    Сравнение по порядку: перестановка и перенос строк — потеря. Значение — номер строки напротив
    при замене одной строки другой или None. Общий бюджет — первичный поиск опор по входу плюс
    CONFLICT_DIFF_WORK для уточнения. Что в него не уложилось, считается целиком потерей и
    появлением (оценка сверху); details сообщает, что результат огрублён."""
    gone, came = {}, {}
    budget = core.CONFLICT_DIFF_WORK + 5 * (len(a) + len(b))
    if details is not None:
        details["approximate"] = False
    work = [(0, len(a), 0, len(b))]
    while work:
        a1, a2, b1, b2 = work.pop()
        while a1 < a2 and b1 < b2 and a[a1] == b[b1]:
            a1, b1 = a1 + 1, b1 + 1
        while a1 < a2 and b1 < b2 and a[a2 - 1] == b[b2 - 1]:
            a2, b2 = a2 - 1, b2 - 1
        cells = (a2 - a1) * (b2 - b1)
        if a1 < a2 and b1 < b2 and cells <= CONFLICT_DIFF_CELLS and cells <= budget:
            budget -= cells
            matcher = difflib.SequenceMatcher(None, a[a1:a2], b[b1:b2], autojunk=False)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                paired = min(i2 - i1, j2 - j1)
                for k in range(i2 - i1):
                    gone[a1 + i1 + k] = b1 + j1 + k if k < paired else None
                for k in range(j2 - j1):
                    came[b1 + j1 + k] = a1 + i1 + k if k < paired else None
            continue
        anchors = []
        if a1 < a2 and b1 < b2:
            cost = 5 * ((a2 - a1) + (b2 - b1))     # поиск опор тоже работа, и вложенные опоры не бесплатны
            if cost <= budget:
                anchors = core._anchors(a[a1:a2], b[b1:b2])
            budget -= cost
        if not anchors:
            if details is not None and a1 < a2 and b1 < b2:
                details["approximate"] = True
            gone.update((i, None) for i in range(a1, a2))
            came.update((j, None) for j in range(b1, b2))
            continue
        last_i, last_j = 0, 0
        for i, j in anchors:
            work.append((a1 + last_i, a1 + i, b1 + last_j, b1 + j))
            last_i, last_j = i + 1, j + 1
        work.append((a1 + last_i, a2, b1 + last_j, b2))
    return gone, came


_ZERO_WIDTH = {ord(c): None for c in "​‌‍⁠"}
_WIDE_SPACES = {ord(c): " " for c in "       "}


def _invisible_note(one, other):
    """Если строки различаются только невидимыми знаками, сказать об этом словами; иначе пустая строка."""
    kinds = []
    for label, fix in (("метка порядка байтов (BOM)", lambda t: t.replace("﻿", "")),
                       ("знаки нулевой ширины", lambda t: t.translate(_ZERO_WIDTH)),
                       ("неразрывные пробелы вместо обычных", lambda t: t.translate(_WIDE_SPACES)),
                       ("пробелы или табуляция в конце строки", lambda t: t.rstrip(" \t"))):
        fixed = fix(one), fix(other)
        if fixed != (one, other):
            kinds.append(label)
        one, other = fixed
    return "Различие только в невидимых знаках: " + ", ".join(kinds) + "." if kinds and one == other else ""


def _view(line, partner):
    """Строка для показа и признак, что она сокращена. Длинную строку показываем вокруг места различия."""
    line = _show_text(line)
    if len(line) <= CONFLICT_SHOWN_CHARS:
        return line, False
    if partner is None:
        return line[:CONFLICT_SHOWN_CHARS] + "…", True
    same = len(os.path.commonprefix([line, _show_text(partner)]))
    start = max(0, min(same - 80, len(line) - CONFLICT_SHOWN_CHARS))
    end = start + CONFLICT_SHOWN_CHARS
    return ("…" if start else "") + line[start:end] + ("…" if end < len(line) else ""), True


def _present(lines, marks, other):
    """Первые строки различий для показа, пометки к ним и признак, что показано всё."""
    order = sorted(marks)
    shown, notes, whole = [], [], len(order) <= CONFLICT_SHOWN_LINES
    for index in order[:CONFLICT_SHOWN_LINES]:
        partner = None if marks[index] is None else other[marks[index]]
        text, cut = _view(lines[index], partner)
        shown.append(text)
        notes.append("" if partner is None else _invisible_note(lines[index], partner))
        whole = whole and not cut
    return shown, notes, whole


def _eol_names(text):
    plain = text.replace("\r\n", "")
    return [name for name, found in (("CRLF", "\r\n" in text), ("CR", "\r" in plain),
                                     ("LF", "\n" in plain)) if found]


def _file_notes(mine, theirs, encoding):
    """Слова о том, что в самих файлах отличается и не видно в строках: кодировка, BOM, концы строк."""
    notes = []
    if encoding == "utf-16":
        notes.append("Копия записана в кодировке UTF-16: программа показала её расшифрованной. "
                     "Скачивается копия без изменений, байт в байт.")
    elif encoding == "unknown":
        notes.append("Копия записана не в UTF-8 (или это UTF-16 без метки: в тексте есть нулевые знаки), "
                     "и по ней нельзя точно определить кодировку, поэтому показ неточный: непонятные знаки заменены на «�». Скачивается копия без изменений, байт в байт.")
    if theirs is not None:
        if mine.startswith("﻿") != theirs.startswith("﻿"):
            notes.append("В начале файла есть невидимая метка BOM: " + (
                "она есть в вашей копии и её нет в версии программы." if mine.startswith("﻿")
                else "она есть в версии программы и её нет в вашей копии."))
        ends, theirs_ends = _eol_names(mine), _eol_names(theirs)
        if ends != theirs_ends and mine and theirs:
            notes.append("Концы строк различаются: в вашей копии " + (", ".join(ends) or "нет") +
                         ", в версии программы " + (", ".join(theirs_ends) or "нет") + ".")
        if mine and theirs and mine.endswith(("\n", "\r")) != theirs.endswith(("\n", "\r")):
            notes.append("Различается конец файла: в одной из версий последняя строка не закрыта переводом строки.")
    return notes


def preview_conflict(recording_id, conflict_id, kind="document"):
    """Показать копию человека и сравнение с последней версией программы; ничего не записывает.

    kind — область конфликта: document или conspect."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        row = _conflict_row(store, recording_id, conflict_id, kind)
        facts = _conflict_facts(store, row)
        reason, can_replace, snapshot, copy_problem = facts["reason"], facts["can_replace"], facts["snapshot"], None
        try:
            observed = _observed_bytes(store, row)
        except core.CoreError as ex:    # копия в базе повреждена: показываем причину, заменять нельзя
            observed, copy_problem = None, str(ex)
            reason, can_replace, snapshot = copy_problem, False, None
        program, encoding, newlines_only, notes = facts["program"], None, False, []
        gone = came = {}
        mine = theirs = []
        text = None
        comparison = {"approximate": False}
        if observed is not None:
            text, encoding, _ = _decode_copy(observed)
            program_text = None if program is None else program.decode("utf-8", errors="surrogateescape")
            if program_text is not None:
                mine, theirs = _lines(text), _lines(program_text)
                gone, came = _diff_lines(mine, theirs, details=comparison)
                newlines_only = text != program_text and mine == theirs
            notes = _file_notes(text, program_text, encoding)
            if comparison["approximate"]:
                notes.append("Сравнение огрублено: число строк — оценка сверху, а не точное число изменений. "
                             "В списки могли попасть совпадающие строки. Сохранённая копия доступна целиком.")
        waits = _unresolvable_wait(store, row, _conflict_rows(store, row["scope"])) if can_replace else None
        if waits:
            notes.append(waits)
        removed, removed_notes, whole_removed = _present(mine, gone, theirs)
        added, added_notes, whole_added = _present(theirs, came, mine)
        shown_text = None if text is None else _show_text(text)
        return {"conflict_id": conflict_id, "kind": row["kind"], "relative_path": row["relative_path"],
                "observed_hash": row["observed_hash"], "observed_size": row["observed_size"],
                "preserved": row["preserved"],
                "preview": None if shown_text is None else shown_text[:CONFLICT_PREVIEW_CHARS],
                "truncated": shown_text is not None and len(shown_text) > CONFLICT_PREVIEW_CHARS,
                "file": str(_shown_file(row, {})), "file_state": facts["file_state"],
                "removed": removed, "added": added, "removed_total": len(gone), "added_total": len(came),
                "removed_notes": removed_notes, "added_notes": added_notes,
                "shown_all": whole_removed and whole_added, "newlines_only": newlines_only,
                "encoding": encoding, "file_notes": notes, "copy_problem": copy_problem,
                "comparison_approximate": comparison["approximate"],
                "also_replaces": facts["also_replaces"] if can_replace else [],
                "waits_for": facts["waits_for"] if can_replace else [], "waits_blocked": bool(waits),
                "stops_for": facts["stops_for"] if can_replace else [],
                "stops_changed": facts["stops_changed"] if can_replace else [],
                "effect": facts["effect"] if can_replace else None,
                "can_replace": can_replace, "reason": reason, "snapshot": snapshot}
    finally:
        store.close()


def conflict_copy(recording_id, conflict_id, kind):
    """Сохранённые байты копии человека целиком и имя для скачивания; ничего не меняет и не читает файл."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        row = _conflict_row(store, recording_id, conflict_id, kind)
        data = _observed_bytes(store, row)
        if data is None:
            raise core.CoreError("Копия не сохранена: файл был больше 8 МиБ. Заберите его прямо из папки записи.")
        return {"data": data, "name": _copy_name(row)}
    finally:
        store.close()


def _is_journal_stop(ex):
    """Запись свёртки остановлена журналом: файлы свёртки менялись после сбоя, и программа не продолжает сама."""
    return isinstance(ex, ValueError) and "ручное восстановление" in str(ex)


def _journal_blockers(store, row):
    """Файлы, из-за которых журнал свёртки не даёт продолжить запись: (имя, хэши журнала, имя журнала).

    Журнал один на запись, поэтому мешать может и файл другой свёртки из той же папки."""
    import conspect
    import documents
    import recordings
    found = []
    try:
        binding = recordings.document_binding(store, row["recording_id"])
        root = documents._authorized_root(store, binding["root_id"], binding["root"])
        json_relative = Path(binding["paths"]["json"])
        journal, exists = documents._read_only_target(root, str(json_relative) + ".conspect.pending")
        if not exists:
            return found
        record = json.loads(journal.read_text(encoding="utf-8"))
        if record.get("version") != 1 or not isinstance(record.get("files"), list):
            return found
        for item in record["files"]:
            name = item["name"]
            if (not isinstance(name, str) or Path(name).name != name
                    or not name.endswith((".conspect.json", ".conspect.md"))):
                return []
            relative = (json_relative.parent / name).as_posix()
            path, exists = documents._read_only_target(root, relative)
            digest = conspect._digest(path.read_bytes()) if exists else None
            hashes = (item["old_hash"], item["new_hash"])
            if digest not in hashes:
                found.append((relative, hashes, journal.name, binding["root_id"]))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return found


def _journal_reason(store, row):
    """Почему журнал остановил запись: какой файл мешает и есть ли выход на этом экране."""
    head = ("Запись файлов свёртки остановлена журналом: после прошлого сбоя файлы свёртки изменились, и программа "
            "не продолжает сама, чтобы не затереть вашу правку. Сначала сохраните свою новую правку отдельно "
            "(скопируйте файл в другое место). ")
    blockers = _journal_blockers(store, row)
    if not blockers:
        return head + ("Какой файл мешает, определить не удалось, и выхода на этом экране нет: нужно "
                       "ручное восстановление рядом с файлами записи.")
    rows = _conflict_rows(store, row["scope"])
    parts = []
    for relative, hashes, journal, root_id in blockers:
        name = Path(relative).name
        copies = []
        for other in reversed(rows):
            if (other["recording_id"] == row["recording_id"] and other["root_id"] == root_id
                    and os.path.normcase(str(Path(other["relative_path"]))) == os.path.normcase(str(Path(relative)))
                    and other["preserved"] and other["observed_hash"] in hashes):
                try:
                    if _observed_bytes(store, other) is not None:
                        copies.append(other)
                except core.CoreError:
                    pass
        if copies:
            parts.append(f"Мешает файл «{name}» (журнал «{journal}»): верните на его место его сохранённую копию "
                         f"(скачайте её как «{_copy_name(copies[0])}» и переименуйте в «{name}») и нажмите «Повторить».")
        else:
            parts.append(f"Мешает файл «{name}» (журнал «{journal}»): он изменён после сбоя, а подходящей его копии "
                         "у программы нет, поэтому выхода на этом экране нет: нужно ручное восстановление "
                         "этого файла и журнала вне экрана.")
    return head + " ".join(parts)


def _failure_reason(store, row, ex):
    """Настоящая причина, по которой файл не записан, когда она известна по ошибке записи."""
    import documents
    import storage
    if _is_journal_stop(ex):
        return _journal_reason(store, row)
    if isinstance(ex, documents.DamagedCopy):
        return str(ex) + " После изменения файла нажмите «Повторить»."
    if isinstance(ex, storage.Busy) or (isinstance(ex, sqlite3.OperationalError)
                                        and ("locked" in str(ex).lower() or "busy" in str(ex).lower())):
        return ("База записей была занята другой операцией. Решение принято; подождите немного "
                "и нажмите «Повторить».")
    if isinstance(ex, OSError):
        return ("Файл не удалось записать: возможно, он открыт в другой программе или недоступен "
                f"(причина: {ex}). Закройте программы, где он открыт, и нажмите «Повторить».")
    return ("При записи файла программа остановилась с ошибкой "
            f"(причина: {type(ex).__name__}: {ex}). Нажмите «Повторить»; если ошибка повторится, сообщите о ней.")


def _file_words(row, digest):
    """Что сказать о файле по его сверенному содержимому, а не по догадке о том, дошла ли запись."""
    if digest == row["observed_hash"]:
        return " Файл по-прежнему такой, как ваша правка: версия программы в него не записана."
    return (" Файл сейчас не совпадает ни с вашей сохранённой правкой, ни с последней версией программы. "
            "Причину изменения по этой проверке установить нельзя. Сверьте его содержимое.")


def _unresolvable_wait(store, row, rows):
    """Открытый конфликт того же объекта, который на экране не решить, но запись этого файла его ждёт."""
    stuck = sorted({Path(other["relative_path"]).name for other in rows
                    if other["object"] == row["object"] and other["conflict_id"] != row["conflict_id"]
                    and not other["resolved"] and (not other["preserved"] or other["kind"].startswith("archive_")
                                                   or _copy_problem(store, other))})
    if not stuck:
        return None
    names = ", ".join(f"«{name}»" for name in stuck)
    return (f"Запись этого файла ждёт решения по другому файлу той же записи: {names}. Но решить его на этом экране "
            "нельзя (файл больше 8 МиБ, это архив или его копия повреждена), нужен отдельный ручной разбор. До него "
            "этот файл останется как есть, даже если решение о замене принято.")


def _next_step(store, row, rows, problem):
    """Карточка принятого, но не записанного решения: (что будет с файлом, слова человеку)."""
    import documents
    program = core._program_version(store, row)[1]
    found, digest = _file_at(store, row, row["relative_path"])
    state = found if found != "ok" else "same" if digest == row["observed_hash"] else "changed"
    waits = _waits_for(row, rows)
    stops, changed = _set_stops(store, row, rows, program, state)
    effect = _effect(row, state, waits, stops)
    waiting = ""
    if waits:
        waiting = _unresolvable_wait(store, row, rows) or (
            "Запись этого файла ждёт решения по другому файлу той же записи (он среди ждущих решения).")
    if problem:
        if state == "missing":
            text = ("Файла на прежнем месте нет, а его сохранённая копия повреждена, скачать её нельзя: "
                    "нужен ручной разбор вне экрана.")
        elif state == "same":
            text = ("Сохранённая копия повреждена, поэтому файл заменён не будет"
                    + (", даже когда ожидание другого файла кончится. " if waits else ". ") + documents.DAMAGED_EXIT)
        elif state == "changed":
            text = ("Сохранённая копия повреждена, но файл уже изменён. Заменён он не будет: когда запись дойдёт "
                    "до него, программа покажет новый конфликт, а если файл не больше 8 МиБ, сохранит его новую копию.")
        else:
            text = "Сохранённая копия повреждена, а файл сейчас не читается: освободите его и обновите список."
        return effect, f"{waiting} {text}".strip()
    if waits:
        return effect, (waiting + " Если тот файл будет решён, программа допишет этот, если вы его с тех пор "
                        "не меняли; если меняли, появится новый конфликт.")
    if state == "missing":
        return effect, _missing_text(row, "и нажмите «Повторить».")
    if state == "unavailable":
        return effect, "Файл сейчас нельзя прочитать, записать его программа не может. Освободите его и повторите."
    if effect == "recopy":
        return effect, ("Файл изменён после принятого решения, поэтому заменён не будет: при повторе программа "
                        "покажет новый конфликт, а если файл не больше 8 МиБ, сохранит его новую копию.")
    if effect == "stopped" and changed:
        return effect, ("Этот файл сейчас не будет записан: запись встанет раньше — " + _stops_words(stops, changed)
                        + ". Выход указан в карточке того файла.")
    if effect == "stopped":
        return effect, ("Этот файл не будет записан, пока не разобрана повреждённая копия: "
                        + ", ".join(f"«{name}»" for name in stops) + ". Выход указан в карточке того файла.")
    return effect, ("Принятое решение программа может довести и сама, при любой следующей записи файлов этой записи, "
                    "не спрашивая снова, если файл не меняли и другие файлы записи не помешают.")


def _file_outcome(store, row, failure):
    """Итог записи по самому файлу: что в нём лежит сейчас."""
    import documents
    state, digest = _file_at(store, row, row["relative_path"])
    if digest == documents._digest(core._program_version(store, row)[1][row["kind"]]):
        return {"outcome": "replaced", "new_conflict_id": None, "reason": None}
    rows = [other for other in _conflict_rows(store, row["scope"]) if other["object"] == row["object"]]
    fresh = [other for other in rows if not other["resolved"] and other["seq"] > row["seq"]
             and other["relative_path"] == row["relative_path"]]
    if fresh:
        return {"outcome": "changed", "new_conflict_id": fresh[-1]["conflict_id"], "reason": None}
    waiting = any(not other["resolved"] for other in rows)
    problem = _copy_problem(store, row) if waiting or state == "missing" else None
    if problem:
        reason = _next_step(store, row, rows, problem)[1]
    elif waiting:
        reason = _unresolvable_wait(store, row, rows) or (
            "Запись этого файла ждёт решения по другому файлу той же записи (он среди ждущих решения). "
            "Если тот файл будет решён, программа допишет этот, если вы его с тех пор не меняли; "
            "если меняли, появится новый конфликт.")
    elif state == "missing":
        reason = _missing_text(row, "и нажмите «Повторить».")
    elif state == "unavailable":
        reason = ("Файл сейчас нельзя прочитать: он занят другой программой, недоступен или на его месте папка "
                  "либо ссылка. Освободите его и нажмите «Повторить».")
    elif failure is not None:
        reason = _failure_reason(store, row, failure) + _file_words(row, digest)
    else:
        reason = ("Программа не дописала этот файл: очередь записи остановилась, возможно, на другой записи "
                  "из той же папки. Нажмите «Повторить»; если не помогает, закройте программы, где открыты "
                  "файлы этой записи." + _file_words(row, digest))
    return {"outcome": "not_written", "new_conflict_id": None, "reason": reason}


def _write_and_report(store, row):
    """Довести запись файлов объекта и сообщить итог по самому файлу, а не по очереди."""
    import recordings
    failure = None
    try:
        if row["scope"] == "document":
            recordings.materialize_recording(store, row["recording_id"])
        else:
            recordings.materialize_conspects(store, row["recording_id"], conspect_id=row["conspect_id"])
    except Exception as ex:    # решение уже записано, и ответ всегда по факту: что лежит в файле
        failure = ex
    try:
        return _file_outcome(store, row, failure)
    except Exception as ex:    # итог проверить нельзя, но решение записано: об этом и отвечаем, а не обрываем ответ
        return {"outcome": "not_written", "new_conflict_id": None,
                "reason": "Решение принято, но проверить, что лежит в файле, не удалось: "
                          + (str(ex) if isinstance(ex, core.CoreError) else explain_error(ex))
                          + " Откройте список позже: если файл не заменён, появится кнопка «Повторить»."}


def _replace_file(store, row, snapshot=None):
    """Записать решение «заменить файл версией программы» и довести запись. Файл проверяется до решения."""
    import recordings
    import storage
    facts = _conflict_facts(store, row)
    if not facts["can_replace"]:
        raise core.CoreError(facts["reason"])
    if snapshot is not None and facts["snapshot"] != snapshot:
        raise core.CoreError("Показанное сравнение устарело: конфликт или версия программы изменились. "
                        "Откройте сравнение заново.")
    try:     # без копии (файл больше 8 МиБ) закрывается лишь конфликт, файл уже равен версии программы и не меняется
        result = getattr(store, f"resolve_{row['scope']}_conflict")(
            row["conflict_id"], str(uuid.uuid4()), recordings.operation_context(store, "human"),
            **({} if row["preserved"] else {"without_copy": True}))
    except (storage.Busy, sqlite3.Error) as ex:
        raise core.CoreError("Решение не записано, файл не тронут. " + explain_error(ex)) from ex
    except (OSError, ValueError, storage.StorageError) as ex:
        raise core.CoreError("Решение не записано, файл не тронут. " + explain_error(ex)) from ex
    return {**result, **_write_and_report(store, row)}


def replace_conflict_file(snapshot):
    """Экран: заменить файл, только если человек видел сравнение (snapshot из preview_conflict) и оно не устарело."""
    import recordings
    if not isinstance(snapshot, dict):
        raise core.CoreError("Нет сравнения, которое видел человек: откройте его и проверьте, что пропадёт.")
    store = recordings.open_store()
    try:
        store.current_identity()
        row = _conflict_row(store, snapshot.get("recording_id"), snapshot.get("conflict_id"),
                            snapshot.get("scope"))
        return _replace_file(store, row, snapshot)
    finally:
        store.close()


def retry_conflict_file(recording_id, conflict_id, kind):
    """Повторить только запись файлов по уже принятому решению: без новых решений и ревизий."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        row = _conflict_row(store, recording_id, conflict_id, kind)
        item = _conflict_item(store, row, _conflict_rows(store, kind), {})
        if item["status"] == "damaged":
            raise core.CoreError(item["problem"])
        if item["status"] == "open":
            raise core.CoreError("Конфликт ещё не решён: повторять нечего. Сначала выберите, что делать с файлом.")
        if item["status"] == "resolved":
            raise core.CoreError("Повторять нечего: файл уже записан, или вы вернули на место свою копию, "
                            "или по этому файлу есть более новый конфликт.")
        return _write_and_report(store, row)
    finally:
        store.close()


def _damaged_copies_warning(copies):
    """Предупреждение о повреждённых копиях. Первая строка без путей: её получает и агент."""
    if not copies:
        return []
    return [f"Повреждённых сохранённых копий правок: {len(copies)}. В резерве они такие же повреждённые: "
            "вернуть по ним правку нельзя. Если файл с правкой ещё цел, сохраните его отдельно.",
            *(f"- «{item['relative_path']}» — конфликт {item['conflict_id']}" for item in copies)]


def _not_written_warning(rows):
    """Предупреждение о ревизиях, которые восстановление могло записать, но не записало (очередь pending).

    Конфликты и сбои, пришедшие из резерва, сюда не входят: они видны на экране конфликтов, как и раньше.
    Первая строка без путей и имён: её получает и агент."""
    if not rows:
        return []
    count = len({row["recording_id"] for row in rows})
    return [f"Восстановлено, но не все файлы записей сформированы: записей — {count}, ревизий — {len(rows)}. "
            "Если файл занят другой программой или защищён от записи, закройте его или снимите защиту; затем "
            "запустите sync_recording_documents для записи или повторите действие с ней — файлы допишутся.",
            *(f"- запись {row['recording_id']}, ревизия {row['document_revision']}: " +
              ("файл занят другой программой или защищён от записи" if row["error_code"] == "busy"
               else "ждёт предыдущую ревизию этой записи") for row in rows)]


def backup_store(destination):
    """Понятная операция резерва состояния: новый неактивный файл БД и отчёт проверки копии."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        report = store.backup(destination, allow_damaged_copies=True)
        warning = _damaged_copies_warning(report.get("damaged_copies"))
        return {"destination": str(destination), "report": report, **({"warning": warning} if warning else {})}
    finally:
        store.close()


def inspect_backup(backup_path):
    """Прочитать резервную копию только для чтения: корни, пользователи, число записей, повреждённые копии."""
    import storage
    store = storage.open_store_readonly(str(backup_path))
    try:
        roots = [dict(row) for row in store.db.execute(
            "SELECT root_id, path, requires_remap FROM allowed_roots ORDER BY path")]
        users = [dict(row) for row in store.db.execute(
            "SELECT user_id, display_name FROM local_users ORDER BY display_name")]
        recordings_count = store.db.execute("SELECT count(*) FROM recordings").fetchone()[0]
        damaged = storage.damaged_conflict_copies(store.db)
        return {"path": str(backup_path), "roots": roots, "users": users,
                "recordings": recordings_count, "damaged_copies": damaged,
                "warning": _damaged_copies_warning(damaged),
                "restore_guard_locked": bool(store.db.execute(
                    "SELECT locked FROM restore_guard WHERE id=1").fetchone()[0])}
    finally:
        store.close()


_SIDE_SUFFIXES = ("-wal", "-shm", "-journal")     # служебные файлы SQLite рядом с базой: набор с ней одно целое
_HELD_ERRORS = (32, 33)                            # ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION
_SQLITE_HEADER = b"SQLite format 3\x00"
_REPLACE_SUPPORTED = os.name == "nt"               # замена открытой базы проверяется переименованием: только Windows
_BUSY_HINT = ("закройте программу (сервер в приложении ИИ-агента, экран rt.py screen, идущую обработку) "
              "и повторите")


def _side_files(path):
    """Служебные файлы SQLite, которые сейчас лежат рядом с базой path (в том числе битые ссылки)."""
    return [Path(str(path) + suffix) for suffix in _SIDE_SUFFIXES if os.path.lexists(str(path) + suffix)]


def _held_elsewhere(path):
    """Windows: файл сейчас открыт другой программой (или этим процессом). Файл не меняется: проба — открытие
    только на чтение без совместного доступа, сразу закрывается. Иная ошибка открытия — не «занят»: её покажет
    следующий шаг."""
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.CreateFileW(str(path), 0x80000000, 0, None, 3, 0x80, None)   # GENERIC_READ, OPEN_EXISTING
    if handle is None or handle == wintypes.HANDLE(-1).value:
        return ctypes.get_last_error() in _HELD_ERRORS
    kernel32.CloseHandle(handle)
    return False


def _check_destination_free(destination, replace):
    """До любых изменений: можно ли занять место базы. Ничего не пишет и не создаёт."""
    present = os.path.lexists(destination)
    orphans = _side_files(destination)
    if not present and orphans:
        # без основного файла чужой -wal наложился бы на опубликованную базу (S-2): не трогаем, отказ
        raise core.CoreError("рядом с местом восстановления лежат служебные файлы базы без самой базы: "
                             + ", ".join(str(item) for item in orphans) + ". Рядом с новой базой их оставить "
                             "нельзя. Ничего не изменено; укажите другой путь или перенесите эти файлы вручную.")
    if not present or not replace:
        return
    if not _REPLACE_SUPPORTED:
        # на POSIX переименование открытой базы проходит молча: занятость так не проверить (до отдельного решения)
        raise core.CoreError("замена существующей базы при восстановлении поддерживается только в Windows; "
                             "ничего не изменено. Укажите новый путь.")
    if any(_held_elsewhere(item) for item in (destination, *orphans)):
        raise core.CoreError("прежнюю базу сейчас использует другая программа: " + _BUSY_HINT
                             + ". Ничего не изменено.")


def _copy_backup(backup, staged):
    """Шаг 1: копия резерва через SQLite backup API из соединения только на чтение. Резерв со своим -wal рядом
    копируется вместе с подтверждённым в -wal; резерв без служебных файлов открывается неизменяемым и следов
    не оставляет. Путь резерва на тип диска не проверяется (как и прежним копированием файла)."""
    import storage
    uri = backup.resolve().as_uri() + ("?mode=ro" if _side_files(backup) else "?mode=ro&immutable=1")
    with staged.open("xb"):
        pass
    source = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5)
    try:
        target = storage._connect(staged, "rw")
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def _settle_previous(destination):
    """Шаг 2: прежнюю базу не копировать, а открыть на запись и закрыть: SQLite сам вносит подтверждённое из -wal
    в основной файл и убирает служебные файлы. -> None, если база открылась, иначе причина (база не читается: её
    набор файлов откладывается как есть). Остались служебные файлы у читаемой базы — её держит другая программа:
    отказ, ничего не переименовано."""
    import storage
    busy = "прежняя база занята другой программой: " + _BUSY_HINT + ". Ничего не изменено."
    try:
        with open(destination, "rb") as file:
            header = file.read(len(_SQLITE_HEADER))
    except OSError as ex:
        raise core.CoreError(f"прежнюю базу не удалось прочитать ({ex}). Ничего не изменено.") from ex
    if header != _SQLITE_HEADER:
        # пустой или чужой файл SQLite не открываем: при открытии он удалил бы лежащий рядом -wal
        return "прежний файл базы пуст или не является базой SQLite"
    try:
        db = storage._connect(destination, "rw")
    except storage.Busy as ex:
        raise core.CoreError(busy) from ex
    except (storage.StorageError, ValueError) as ex:
        return f"прежняя база не открывается ({ex})"
    unreadable = None
    try:
        db.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.DatabaseError as ex:
        if storage._is_busy(ex):
            raise core.CoreError(busy) from ex
        unreadable = f"прежняя база не читается ({ex})"
    finally:
        db.close()
    if unreadable is None and _side_files(destination):
        raise core.CoreError("прежнюю базу держит другая программа (её служебные файлы не закрылись): "
                             + _BUSY_HINT + ". Ничего не изменено.")
    return unreadable


def _move_aside(destination):
    """Шаг 3: переименовать прежнюю базу целиком в before-restore-*: сначала основной файл (в Windows открытый
    файл не переименовывается — это и проверка «никто не держит»), затем её служебные файлы с тем же именем, чтобы
    SQLite нашёл их у отложенной базы. Набор не разлучается и не удаляется. -> (новое имя, [(было, стало)])."""
    name = destination.name + ".before-restore-" + time.strftime("%Y%m%d-%H%M%S")
    if any(os.path.lexists(destination.with_name(name + suffix)) for suffix in ("", *_SIDE_SUFFIXES)):
        name += "-" + uuid.uuid4().hex[:8]
    saved = destination.with_name(name)
    moved = []
    try:
        os.rename(destination, saved)        # в Windows не заменяет существующий файл
        moved.append((destination, saved))
        for side in _side_files(destination):
            target = Path(str(saved) + side.name[len(destination.name):])
            os.rename(side, target)
            moved.append((side, target))
    except OSError as ex:
        left = _move_back(moved)
        if left:
            raise core.CoreError(_not_returned(left)) from ex
        raise core.CoreError("не удалось отложить прежнюю базу (возможно, её держит другая программа): "
                             + _BUSY_HINT + f". Прежняя база на месте ({ex}).") from ex
    return saved, moved


def _move_back(moved):
    """Шаг 5: вернуть набор на место — служебные файлы раньше основного (пока основного нет, базу никто не
    откроет без её -wal). -> пары, которые вернуть не удалось."""
    left = []
    for original, renamed in reversed(moved):
        try:
            os.rename(renamed, original)
        except OSError:
            left.append((original, renamed))
    return left


def _not_returned(left):
    return ("восстановление не выполнено, и прежнюю базу не удалось вернуть на место. Ничего не удалено: "
            + "; ".join(f"«{renamed}» должен называться «{original}»" for original, renamed in left)
            + ". Закройте программу и переименуйте эти файлы обратно вручную.")


def _publish(staged, destination):
    """Шаг 4: опубликовать подготовленный файл без замены (жёсткая ссылка, как в Store.backup)."""
    os.link(staged, destination)


def restore_from_backup(backup_path, destination, root_mapping, user_id, *, replace=True):
    """Мастер восстановления: копия резерва получает новые корни и установку, документы
    формируются в новых корнях из сохранённых bytes. Живая база подменяется только
    при replace=True и только в Windows: её набор файлов (база, -wal, -shm) переименовывается рядом
    в before-restore-*, ничего не удаляется; из резерва ничего не удаляется (БД-1, S-2)."""
    import documents
    import storage
    backup = Path(backup_path).expanduser()
    if not backup.is_file():
        raise core.CoreError("файл резерва не найден")
    destination = Path(destination).expanduser()
    if destination.exists() and destination.resolve() == backup.resolve():
        raise core.CoreError("нельзя восстанавливать поверх самого резерва")
    if not isinstance(root_mapping, dict) or not root_mapping:
        raise core.CoreError("нужно переназначить все документные корни")
    mapping = {str(key): str(value) for key, value in root_mapping.items()}
    for root_id, path in mapping.items():
        if not Path(path).is_dir():
            raise core.CoreError(f"путь документного корня не существует: {path}")
    _check_destination_free(destination, replace)
    staged = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
    try:
        _copy_backup(backup, staged)
    except (OSError, sqlite3.Error, storage.StorageError, ValueError) as ex:
        raise core.CoreError(f"не удалось скопировать резерв в {staged}: {ex}") from ex
    try:
        store = storage.open_store(staged)
        try:
            installation_id = str(uuid.uuid4())
            store.rebind_restored_copy(mapping, installation_id, user_id)
            materialized = []
            for root_id, path in mapping.items():
                materialized.extend(documents.materialize_pending(store, root_id, Path(path)))
            # PR 5б-2: занятый или защищённый от записи файл больше не обрывает восстановление — называем, что осталось
            not_written = [dict(row) for row in store.db.execute(
                "SELECT recording_id, document_revision, error_code FROM document_outbox "
                "WHERE state = 'pending' ORDER BY recording_id, document_revision")]
            validation = storage.validate_copy(staged, allow_damaged_copies=True)
        finally:
            store.close()
        storage._release_wal_files(staged)
    except (storage.StorageError, ValueError) as ex:
        raise core.CoreError(f"восстановление не выполнено; подготовленный файл остался: {staged} "
                        f"({ex})") from ex
    if _side_files(staged):
        # опубликуется только основной файл: подтверждённое в -wal подготовленной копии пропало бы
        raise core.CoreError("восстановление не выполнено: у подготовленной копии не закрылись служебные файлы; "
                             f"подготовленный файл остался: {staged}")
    previous_saved_as = None
    warning = []
    moved = []
    if os.path.lexists(destination):
        if not replace:
            raise core.CoreError(f"файл {destination} уже существует; укажите другой путь")
        _check_destination_free(destination, replace)
        unreadable = _settle_previous(destination)
        saved, moved = _move_aside(destination)
        previous_saved_as = str(saved)
        if unreadable:
            warning.append(f"{unreadable}; она отложена как есть вместе со служебными файлами: "
                           + ", ".join(str(renamed) for _, renamed in moved))
    try:
        occupied = ([destination] if os.path.lexists(destination) else []) + _side_files(destination)
        if occupied:
            raise FileExistsError("на месте базы появились файлы: " + ", ".join(str(item) for item in occupied))
        _publish(staged, destination)
    except OSError as ex:
        left = _move_back(moved)
        if left:
            raise core.CoreError(_not_returned(left) + f" Подготовленный файл остался: {staged}") from ex
        raise core.CoreError(f"не удалось опубликовать восстановленную базу ({ex}); прежняя база на месте, "
                             f"подготовленный файл остался: {staged}") from ex
    try:
        staged.unlink()       # второе имя того же файла: данные остаются под именем базы
    except OSError:
        warning.append(f"временное имя восстановленной базы не убралось: {staged} — это то же содержимое, "
                       "что и у базы; не открывайте его, удалите после закрытия программы")
    warning = _not_written_warning(not_written) + _damaged_copies_warning(validation.get("damaged_copies")) + warning
    return {"destination": str(destination), "installation_id": installation_id,
            "previous_saved_as": previous_saved_as, "validation": validation,
            "materialized": materialized, "not_written": not_written, **({"warning": warning} if warning else {})}


def find_recording_results(source_path):
    """Найти SQLite-записи исходника независимо от файлов истории задач."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        return recordings.find_recordings(store, source_path)
    finally:
        store.close()


def read_recording_result(recording_id):
    """Прочитать текущую SQLite-запись по устойчивому ID без job."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        result = recordings.load_recording(store, recording_id)
        result["recording_ref"] = "recording:" + recording_id
        return result
    finally:
        store.close()


def _recording_reference(reference):
    import storage
    try:
        result = read_recording_result(reference[len("recording:"):])
        return {"state": "done", "json": result["json"], "markdown": result["markdown"],
                "recording_id": result["recording_id"], "document_sync": result["document_sync"]}
    except (OSError, ValueError, KeyError, storage.StorageError) as ex:
        raise core.CoreError(f"SQLite-запись недоступна: {ex}") from ex


def sync_recording_documents(recording_id):
    """Повторить безопасную материализацию после commit или перезапуска."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        recordings.materialize_recording(store, recording_id)
        binding = recordings.document_binding(store, recording_id)
        path = binding["root"] / binding["paths"]["json"]
        conspects = recordings.refresh_conspects(store, recording_id, path)
        return {"recording_id": recording_id, "document_revision": binding["document_revision"],
                "document_sync": recordings.document_sync_state(store, recording_id, binding["document_revision"]),
                "conspects": conspects}
    finally:
        store.close()


def prepare_answer_confirmation(recording_id, answer_ids):
    """Снимок и точное содержимое ответов для явного подтверждения человеком."""
    import recordings
    import review_pack
    import unicodedata
    if (not isinstance(answer_ids, list) or not 1 <= len(answer_ids) <= 8 or
            any(not isinstance(ident, str) or not ident or len(ident) > 80 for ident in answer_ids) or
            len(set(answer_ids)) != len(answer_ids)):
        raise core.CoreError("выберите от 1 до 8 разных ответов")
    store = recordings.open_store()
    try:
        store.current_identity()
        binding = recordings.document_binding(store, recording_id)
        path = binding["root"] / binding["paths"]["json"]
        state = recordings.read_review_pack_state(path)
        manifest = state.get("manifest") or {}
        items = {item["id"]: item for item in manifest.get("items") or []}
        applied = review_pack._applied_map(state)
        conflicted = {item.get("id") for item in state.get("conflicts") or []}
        lines = []
        for ident in answer_ids:
            answer = state["answers"].get(ident)
            item = items.get(ident)
            if not answer or not item or answer.get("actor_kind") == "human":
                raise core.CoreError("ответ отсутствует или уже имеет человеческое происхождение")
            if ident in conflicted or ident in applied:
                raise core.CoreError("конфликтующий или ранее применённый ответ нужно проверить в локальной карточке")
            kind = answer.get("kind")
            question = item.get("question") or "Как правильно?"
            source_text = item.get("text") or ""
            if kind in ("option", "recommended"):
                option = answer.get("option")
                options = item.get("options") or []
                if type(option) is not int or not 0 <= option < len(options):
                    raise core.CoreError("вариант ответа недействителен")
                text = options[option].get("label", "")
                decision = json.dumps(options[option].get("decision") or {}, ensure_ascii=False, sort_keys=True)
            elif kind == "free":
                raise core.CoreError("свободный ответ неизвестного происхождения нужно ввести заново в карточке")
            elif kind in ("revert", "human"):
                text = "отменить ответ" if kind == "revert" else "оставить человеку"
                decision = kind
            else:
                raise core.CoreError("этот тип ответа нельзя подтвердить")
            values = (ident, question, source_text, text, decision)
            if (any(not isinstance(value, str) for value in values) or not text or
                    any(unicodedata.category(ch).startswith("C") for value in values for ch in value)):
                raise core.CoreError("ответ нельзя полностью и однозначно показать для подтверждения")
            detail = [f"{ident}: {question}; место: {source_text}",
                      f"Ответ: {text}; действие: {decision}"]
            if any(len(line) > 200 for line in detail):
                raise core.CoreError("ответ слишком длинный для полного подтверждения; используйте карточку")
            lines.extend(detail)
        if sum(len(line) + 1 for line in lines) > 900:
            raise core.CoreError("ответы не помещаются в окно подтверждения; выберите меньше")
        encoded = _answer_confirmation_hash(state, answer_ids)
        return {"recording_id": recording_id, "answer_ids": list(answer_ids),
                "state_hash": hashlib.sha256(encoded).hexdigest(), "lines": lines}
    except (OSError, ValueError, KeyError) as ex:
        raise core.CoreError(f"не удалось подготовить подтверждение ответов: {ex}") from ex
    finally:
        store.close()


def _answer_confirmation_hash(state, answer_ids):
    ids = set(answer_ids)
    manifest = state.get("manifest") or {}
    relevant_items = [item for item in manifest.get("items") or [] if item.get("id") in ids]
    conflicts = [row for row in state.get("conflicts") or []
                 if row.get("id") in ids or ids.intersection(row.get("cards") or [])]
    value = {"version": state.get("version"), "answers": {
                 aid: state.get("answers", {}).get(aid) for aid in sorted(ids)},
             "items": relevant_items,
             "applied_answers": {aid: (state.get("applied_answers") or {}).get(aid) for aid in sorted(ids)},
             "conflicts": conflicts}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def confirm_review_answers(snapshot):
    """Зафиксировать отдельного подтверждающего, не меняя автора исходного ответа."""
    import recordings
    import review_pack
    current = prepare_answer_confirmation(snapshot["recording_id"], snapshot["answer_ids"])
    if current != snapshot:
        raise core.CoreError("ответы изменились после запроса подтверждения")
    store = recordings.open_store()
    try:
        binding = recordings.document_binding(store, snapshot["recording_id"])
        path = binding["root"] / binding["paths"]["json"]
    finally:
        store.close()
    with recordings.operation_for_document(path, "human"):
        state = recordings.read_review_pack_state(path)
        encoded = _answer_confirmation_hash(state, snapshot["answer_ids"])
        if hashlib.sha256(encoded).hexdigest() != snapshot["state_hash"]:
            raise core.CoreError("ответы изменились после запроса подтверждения")
        context = recordings.context_for_document(path)
        for ident in snapshot["answer_ids"]:
            state["answers"][ident]["confirmation_context"] = context
        review_pack._write_answers(path, state)
    return {"recording_id": snapshot["recording_id"], "confirmed": snapshot["answer_ids"]}


def list_conspect_conflicts(recording_id):
    import recordings
    import storage
    storage._uuid(recording_id)
    store = recordings.open_store()
    try:
        store.current_identity()
        return [dict(row) for row in store.db.execute(
            "SELECT f.conflict_id, f.conspect_id, c.name, f.revision, f.kind, f.relative_path, "
            "f.observed_hash, f.observed_size, (f.observed_bytes IS NOT NULL) AS preserved "
            "FROM conspect_conflicts f JOIN conspects c USING(conspect_id) "
            "WHERE c.recording_id=? AND f.resolution_operation_id IS NULL ORDER BY f.rowid",
            (recording_id,))]
    finally:
        store.close()


def list_document_conflicts(recording_id):
    import recordings
    import storage
    storage._uuid(recording_id)
    store = recordings.open_store()
    try:
        store.current_identity()
        return [dict(row) for row in store.db.execute(
            "SELECT conflict_id, document_revision, kind, relative_path, observed_hash, observed_size, "
            "(observed_bytes IS NOT NULL) AS preserved FROM document_conflicts "
            "WHERE recording_id=? AND resolution_operation_id IS NULL ORDER BY rowid", (recording_id,))]
    finally:
        store.close()


def _resolve_for_agent(scope, recording_id, conflict_id):
    """MCP: после подтверждения в окне заменить файл версией программы. Ответ и ошибки прежние;
    новое одно: если файла на месте нет, отказ до записи решения."""
    import documents
    import recordings
    import storage
    if scope == "document":
        owner = "SELECT recording_id FROM document_conflicts WHERE conflict_id=?"
    else:
        owner = ("SELECT c.recording_id FROM conspect_conflicts f JOIN conspects c USING(conspect_id) "
                 "WHERE f.conflict_id=?")
    store = recordings.open_store()
    try:
        store.current_identity()
        found = store.db.execute(owner, (conflict_id,)).fetchone()
        if found is None or found["recording_id"] != recording_id:
            raise core.CoreError("конфликт не найден для этой записи")
        row = _conflict_row(store, recording_id, conflict_id, scope)
        if _file_at(store, row, row["relative_path"])[0] == "missing":
            raise core.CoreError(_missing_text(row, "затем повторите."))
        extra = {}
        if not row["preserved"]:       # копии нет: закрыть можно, только если файл уже равен версии программы
            facts = _conflict_facts(store, row)
            if not facts["can_replace"]:
                raise core.CoreError(facts["reason"])
            extra = {"without_copy": True}
        if scope == "document":
            result = store.resolve_document_conflict(
                conflict_id, str(uuid.uuid4()), recordings.operation_context(store, "human"), **extra)
            completed = recordings.materialize_recording(store, recording_id)
            failure = documents.unsynced(store, recording_id, result["document_revision"], completed)
            result["materialized"] = failure is None
            if failure is not None:       # решение принято, файл не дописан: исход словами, а не голое false
                result["outcome"], result["reason"] = failure.status, str(failure)
        else:
            result = store.resolve_conspect_conflict(
                conflict_id, str(uuid.uuid4()), recordings.operation_context(store, "human"), **extra)
            completed = recordings.materialize_conspects(store, recording_id, conspect_id=result["conspect_id"])
            result["materialized"] = (result["conspect_id"], result["revision"]) in completed
        return result
    except (OSError, ValueError, storage.StorageError) as ex:
        raise core.CoreError(f"конфликт не разрешён или формирование не завершено: {ex}") from ex
    finally:
        store.close()


def resolve_document_conflict(recording_id, conflict_id):
    return _resolve_for_agent("document", recording_id, conflict_id)


def resolve_conspect_conflict(recording_id, conflict_id):
    return _resolve_for_agent("conspect", recording_id, conflict_id)


import core  # noqa: E402  (последним: core.py в конце загружает этот модуль)
