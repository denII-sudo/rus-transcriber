"""БД-4(а): проверка целостности базы записей (PRAGMA quick_check) один раз на процесс и запрет записи при провале.

quick_check проверяет строение файла (страницы, деревья, свободные списки), без проверки индексов против таблиц —
поэтому он заметно быстрее integrity_check (синтетика 119 МБ: около 0,25 с). Глубже копию проверяет validate_copy,
но только у резервов. Проверка идёт при первом открытии главной базы (paths.STATE_DB) на запись в этом процессе и
при запуске экрана; self_check проверяет заново. Ключ — путь и идентификатор файла: восстановленная база — другой
файл, её проверка начинается заново.

Провал — запись в главную базу отклоняется (StorageError по-русски), чтение работает: человек может посмотреть
и выгрузить. Занятость базы и иная ошибка открытия — не провал: проверка повторится при следующем открытии.
Другие файлы баз (резервы, базы тестов и лаборатории) этот модуль не трогает.
"""
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import storage

LIMIT = 5                        # сколько первых сообщений quick_check хранить: для причины достаточно
_LOCK = threading.Lock()
_RESULTS = {}                    # ключ файла -> итог проверки (только провал или успех; занятость не хранится)

DAMAGED_TEXT = ("База записей не прошла проверку целостности (quick_check): файл базы повреждён. Запись в неё "
                "остановлена, чтобы не испортить данные сильнее; чтение работает. Не удаляйте файлы базы "
                "(state.sqlite3 и рядом -wal, -shm). Восстановите базу из резерва (экран «Резерв состояния» или "
                "python rt.py db-restore РЕЗЕРВ.sqlite3) или сохраните копию папки данных и попросите помощи; "
                "после восстановления перезапустите программу.")


class Damaged(storage.StorageError):
    """Главная база не прошла quick_check: запись в неё отклоняется."""


def _main_db():
    import paths
    return Path(paths.STATE_DB)


def _key(path):
    """Путь, идентификатор файла и время его создания (где оно есть: номер файла может достаться новому файлу).
    None — файла нет."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.normcase(os.path.abspath(str(path))), st.st_dev, st.st_ino,
            getattr(st, "st_birthtime_ns", None) or getattr(st, "st_birthtime", None))


def is_main_db(path):
    """path — главная база программы (paths.STATE_DB); None означает её же."""
    if path is None:
        return True
    return os.path.normcase(os.path.abspath(str(path))) == os.path.normcase(os.path.abspath(str(_main_db())))


def quick_check(path):
    """Проверить файл базы только чтением. -> {"ok": True|False|None, "messages": [...], "seconds": float}.

    ok=None — проверить не удалось (база занята, файл не открылся): это не повреждение."""
    started = time.perf_counter()
    try:
        db = storage._connect_readonly(path)
    except storage.Busy:
        return {"ok": None, "reason": "busy", "messages": [], "seconds": round(time.perf_counter() - started, 3)}
    except storage.UnsupportedSchema as exc:   # первое же чтение отказало: «file is not a database» и подобное
        cause = exc.__cause__
        return {"ok": False, "messages": [str(cause or exc)[:300]],
                "seconds": round(time.perf_counter() - started, 3)}
    except (storage.StorageError, ValueError, OSError) as exc:
        return {"ok": None, "reason": "open", "messages": [type(exc).__name__],
                "seconds": round(time.perf_counter() - started, 3)}
    try:
        rows = [row[0] for row in db.execute(f"PRAGMA quick_check({LIMIT})").fetchall()]
        ok = rows == ["ok"]
        result = {"ok": ok, "messages": [] if ok else [str(item)[:300] for item in rows[:LIMIT]]}
    except sqlite3.DatabaseError as exc:
        if storage._is_busy(exc):
            result = {"ok": None, "reason": "busy", "messages": []}
        else:                                  # «file is not a database», «malformed»: строение файла нарушено
            result = {"ok": False, "messages": [str(exc)[:300]]}
    finally:
        db.close()
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def ensure_checked(path=None):
    """Итог проверки главной базы для этого процесса: проверить, если этот файл ещё не проверялся.

    -> итог quick_check с полем checked_at или None (файла нет). Не главную базу не проверяет (None)."""
    if not is_main_db(path):
        return None
    path = _main_db() if path is None else Path(path)
    key = _key(path)
    if key is None:
        return None
    with _LOCK:
        known = _RESULTS.get(key)
        if known is not None:
            return dict(known)
        result = quick_check(path)
        result["checked_at"] = datetime.now(timezone.utc).isoformat()
        if result["ok"] is not None:           # занятость не запоминаем: следующее открытие проверит снова
            _RESULTS[key] = result
        return dict(result)


def require_writable(path=None):
    """Отказ Damaged, если главная база не прошла проверку в этом процессе. Другие файлы — без проверки."""
    result = ensure_checked(path)
    if result is not None and result["ok"] is False:
        raise Damaged(DAMAGED_TEXT)


def state(path=None):
    """Что известно процессу о главной базе, без новой проверки: None — ещё не проверялась или файла нет."""
    if not is_main_db(path):
        return None
    key = _key(_main_db() if path is None else Path(path))
    with _LOCK:
        known = _RESULTS.get(key) if key is not None else None
        return dict(known) if known is not None else None


def view(path=None):
    """Для экрана и агента: {"checked": bool, "ok": ..., "message": текст или None} без путей."""
    known = state(path)
    if known is None:
        return {"checked": False, "ok": None, "message": None}
    return {"checked": True, "ok": known["ok"], "seconds": known.get("seconds"),
            "message": DAMAGED_TEXT if known["ok"] is False else None}


def self_check_report():
    """Часть самопроверки: свежий quick_check главной базы (не из памяти процесса)."""
    path = _main_db()
    if not path.is_file():
        return {"ok": True, "state": "absent", "note": "базы записей нет — проверять нечего"}
    result = quick_check(path)
    if result["ok"] is None and result.get("reason") == "busy":
        return {"ok": True, "state": "not_checked", "seconds": result["seconds"],
                "note": "база занята — проверка не выполнена, повторите позже"}
    if result["ok"] is None:
        return {"ok": False, "state": "unreadable", "seconds": result["seconds"],
                "note": "файл базы не открылся для проверки (права доступа, путь или диск)"}
    out = {"ok": result["ok"], "state": "ok" if result["ok"] else "damaged", "seconds": result["seconds"]}
    if not result["ok"]:
        out["messages"] = result["messages"]
        out["note"] = DAMAGED_TEXT
    return out


def _forget_all():
    """Для тестов: забыть итоги проверок этого процесса."""
    with _LOCK:
        _RESULTS.clear()
