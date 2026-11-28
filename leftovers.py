"""Остатки после обрыва (БД-9а; находки S-21, О-7): только найти и показать. Ничего не удаляется и не открывается.

Ищутся только имена, которые создаёт сама программа, строго по шаблону (не по префиксу):
  * `<имя>.incomplete-<32 знака hex>` и его `-wal`/`-shm`/`-journal` — подготовленная копия базы, которую не довели
    до публикации: создание базы (`storage.initialize`), резерв (`Store.backup`), восстановление
    (`core_conflicts.restore_from_backup`). Ищутся рядом с базой и в папке резервов `db-backups`, без вложенных папок;
  * `.rt-<32 знака hex>.tmp` — недописанный временный файл документа (`documents._write_atomic`). Ищется в
    зарегистрированных корнях документов и их вложенных папках.
Показываются файлы старше суток (моложе могут быть в работе прямо сейчас), с размером и временем изменения; число
более свежих — отдельно. Исключение — второе имя рабочей базы (ниже): оно показывается сразу.

Файлы не открываются ни SQLite, ни на чтение содержимого: только сведения каталога (`os.lstat`, `os.scandir`).
Это важно для `.incomplete-*`, который после публикации восстановленной базы мог остаться вторым именем того же
файла (жёсткой ссылкой, Р258-7): открыть его SQLite, пока работает программа, значит дать одному файлу два разных
`-wal` и испортить базу. Такой файл помечается особо. По ссылкам и точкам повторной обработки (junction, symlink)
обход не переходит. В остатках — содержимое базы со стенограммами: если папка в облаке, оно уходит наружу.
Удалять остатки может только человек (вопрос 7 плана БД, ответа владельца пока нет); программа их не удаляет.
"""
import os
import re
import stat
import time
from datetime import datetime
from pathlib import Path


INCOMPLETE_RE = re.compile(r"^.+\.incomplete-[0-9a-f]{32}(?:-wal|-shm|-journal)?$")
TEMP_DOCUMENT_RE = re.compile(r"^\.rt-[0-9a-f]{32}\.tmp$")
MIN_AGE_SECONDS = 24 * 3600
MAX_ENTRIES = 50000          # предел просмотренных записей каталогов за один вызов: обход корней не затягивается

HINT_COPY = ("незавершённая копия базы (обрыв при создании базы, резерве или восстановлении): программа её не "
             "использует; внутри — содержимое базы со стенограммами")
HINT_COPY_SIDE = "служебный файл незавершённой копии базы: программа его не использует"
HINT_SECOND_NAME = ("не копия, а второе имя рабочей базы (осталось после восстановления): не открывайте его ничем, "
                    "пока работает программа, — иначе база испортится; удалить можно после закрытия программы")
HINT_LINKED = ("у файла есть второе имя в другом месте (жёсткая ссылка): не открывайте его, пока не выяснено, "
               "чьё это второе имя")
HINT_TEMP_DOCUMENT = "недописанный временный файл документа: программа его не использует"


def _is_link(info):
    """Ссылка или точка повторной обработки (junction и подобные): по ним обход не идёт."""
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _lstat(path):
    try:
        return os.lstat(path)
    except OSError:
        return None


class _Scan:
    def __init__(self, now, min_age, database):
        self.now = now
        self.min_age = min_age
        self.database = database
        self.items = []
        self.younger = 0
        self.entries = 0
        self.truncated = False
        self.skipped = []

    def _entries(self, folder):
        try:
            with os.scandir(folder) as found:
                for entry in found:
                    if self.entries >= MAX_ENTRIES:
                        self.truncated = True
                        return
                    self.entries += 1
                    yield entry
        except FileNotFoundError:
            return
        except OSError as ex:
            self.skipped.append({"path": str(folder), "reason": type(ex).__name__})

    def flat(self, folder, pattern, kind):
        info = _lstat(folder)
        if info is None or _is_link(info) or not stat.S_ISDIR(info.st_mode):
            return
        for entry in self._entries(folder):
            if pattern.match(entry.name):
                self._consider(Path(entry.path), kind)

    def tree(self, root, pattern, kind):
        info = _lstat(root)
        if info is None:
            return
        if _is_link(info) or not stat.S_ISDIR(info.st_mode):
            self.skipped.append({"path": str(root), "reason": "not a plain folder"})
            return
        pending = [Path(root)]
        while pending and not self.truncated:
            folder = pending.pop()
            for entry in self._entries(folder):
                info = _lstat(entry.path)
                if info is None or _is_link(info):
                    continue
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif pattern.match(entry.name):
                    self._consider(Path(entry.path), kind)

    def _consider(self, path, kind):
        info = _lstat(path)
        if info is None or _is_link(info) or not stat.S_ISREG(info.st_mode):
            return
        item = {"path": str(path), "kind": kind, "size": info.st_size,
                "modified": datetime.fromtimestamp(info.st_mtime).astimezone().isoformat(timespec="seconds"),
                "age_days": round(max(0.0, self.now - info.st_mtime) / 86400, 1)}
        second_name = False
        if kind == "database_copy" and not path.name.endswith(("-wal", "-shm", "-journal")):
            item["hard_links"] = info.st_nlink
            second_name = self._same_as_database(info)
            item["second_name_of_database"] = second_name
            item["hint"] = HINT_SECOND_NAME if second_name else (HINT_LINKED if info.st_nlink > 1 else HINT_COPY)
        else:
            item["hint"] = HINT_COPY_SIDE if kind == "database_copy" else HINT_TEMP_DOCUMENT
        if not second_name and self.now - info.st_mtime < self.min_age:
            self.younger += 1
            return
        self.items.append(item)

    def _same_as_database(self, info):
        if self.database is None or info.st_nlink < 2:
            return False
        database = _lstat(self.database)
        return (database is not None and stat.S_ISREG(database.st_mode)
                and (database.st_dev, database.st_ino) == (info.st_dev, info.st_ino) and info.st_ino != 0)


def find(database, roots=(), *, backups=None, now=None, min_age=MIN_AGE_SECONDS):
    """Остатки рядом с базой database, в папке резервов (по умолчанию `<папка базы>/db-backups`) и в корнях
    документов roots. -> {"items", "count", "bytes", "younger", "checked", "truncated", "skipped"}."""
    database = Path(database) if database is not None else None
    now = time.time() if now is None else now
    scan = _Scan(now, min_age, database)
    checked = []
    if database is not None:
        folders = [database.parent, Path(backups) if backups is not None else database.parent / "db-backups"]
        for folder in folders:
            checked.append(str(folder))
            scan.flat(folder, INCOMPLETE_RE, "database_copy")
    for root in roots:
        checked.append(str(root))
        scan.tree(root, TEMP_DOCUMENT_RE, "document_temp")
    items = sorted(scan.items, key=lambda item: item["path"])
    return {"items": items, "count": len(items), "bytes": sum(item["size"] for item in items),
            "younger": scan.younger, "checked": checked, "truncated": scan.truncated, "skipped": scan.skipped}
