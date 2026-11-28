"""В-2: независимая копия автоматического резерва базы на другом диске (решение владельца 28.11.2026).

Что. Папку задаёт человек настройкой db_backup_mirror_dir (экран программы или config.json); по умолчанию пусто —
копия выключена; агент её не ставит. Внутри этой папки программа ведёт свою подпапку rus-transcriber-db-backups:
туда кладётся самая свежая проверенная автокопия из <папка данных>/db-backups/auto. Старые копии там удаляются по
тому же правилу, что в auto (И-1а: самая старая и три свежие защищены, не больше 10 копий и db_backup_max_gb ГБ) —
слова владельца «по тому же правилу».

Как. Автокопия уже прошла validate_copy и записана в опись с SHA-256; на другой диск переносятся её байты
(временный файл, сброс на диск, сверка SHA-256 с описью), потом публикация без замены. SQLite файлов на другом диске
не открывает: так работает и съёмный диск (вопрос 8), на котором программа баз не открывает. Опись и замена копии
того же рабочего дня, восстановление после сбоя и удаление — те же функции autobackup, что для auto.

Место. Папка должна существовать (её создаёт человек), быть локальной (не сеть; съёмный диск можно), не ссылкой и не
junction, не внутри папки данных и не содержать её. Тот же том, что у папки данных, — предупреждение «копия не
независима»; папка в OneDrive — предупреждение. Папка не найдена (диск отключён) — шаг откладывается, неудача
показывается, если так 3 рабочих дня подряд. Подпапка принадлежит одной папке данных: метка в описи auto и в описи
подпапки; чужая подпапка не трогается.

Что удаляется. Только копии из описи своей подпапки: имя по шаблону, не ссылка, размер и SHA-256 как в описи.
Остальное в выбранной папке (чужие файлы, ручные резервы) не читается и не удаляется.
"""
import hashlib
import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

import autobackup
import storage

SETTING = "db_backup_mirror_dir"
SUBFOLDER = "rus-transcriber-db-backups"
FAILURES = {
    "place": "папка не подходит: нужна существующая локальная папка (можно на съёмном диске), не сетевая, не ссылка "
             "и не внутри папки данных программы — укажите другую",
    "foreign": "в выбранной папке уже лежат резервы другой папки данных программы — они не тронуты; укажите "
               "другую папку",
    "manifest": "опись резервов на другом диске не читается или не совпадает с файлами — копирование остановлено, "
                "ничего не удалено; нужна помощь",
    "space": "не хватило места на другом диске",
    "copy": "не удалось скопировать резерв на другой диск (или копия не совпала с исходной) — прежние копии там "
            "на месте",
    "publish": "не удалось сохранить копию на другом диске",
    "delete": "не удалось удалить вытесняемую копию на другом диске (файл занят или защищён) — она осталась",
    "error": "неожиданный сбой копирования на другой диск",
}
WARNINGS = {
    "same_volume": "Копия резерва на «другом диске» лежит на том же диске, что и база: от поломки диска она не "
                   "защитит. Укажите папку на другом диске.",
    "cloud": "Копия резерва на другом диске лежит в папке OneDrive: файлы базы уйдут в облако. Если это не нужно — "
             "укажите другую папку.",
}


class Failure(Exception):
    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


class Unavailable(Exception):
    """Выбранная папка не найдена: диск отключён. Шаг откладывается."""


class MirrorFolder(autobackup.Folder):
    """Своя подпапка в выбранной человеком папке: опись и копии, как в auto. Состояние и замок — у папки auto."""

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / SUBFOLDER
        self.manifest_path = self.path / "manifest.json"
        self.status_path = None
        self.suspect_path = None
        self._lock = None

    def prepare(self):
        self.path.mkdir(exist_ok=True)
        if autobackup._is_link(self.root) or autobackup._is_link(self.path):
            raise Failure("place")


def configured(config):
    value = config.get(SETTING, "")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _real(path):
    return os.path.normcase(os.path.realpath(path))


def _inside(child, parent):
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:                            # разные диски Windows
        return False


def _cloud_roots():
    roots = []
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        value = os.environ.get(name)
        if value:
            roots.append(_real(value))
    return roots


def check_place(value, data_dir):
    """Проверить выбранную папку. -> (папка, предупреждения). Unavailable — папки нет; Failure("place") — не годится."""
    root = Path(value)
    if not root.is_absolute():
        raise Failure("place")
    real_root, real_data = _real(root), _real(data_dir)
    if _inside(real_root, real_data) or _inside(real_data, real_root):
        raise Failure("place")
    if not root.is_dir():
        if root.exists():
            raise Failure("place")
        raise Unavailable()
    try:
        storage.check_local_path(root, allow_removable=True)
    except ValueError as exc:
        raise Failure("place") from exc
    if autobackup._is_link(root):
        raise Failure("place")
    warnings = []
    if os.stat(root).st_dev == os.stat(data_dir).st_dev:      # Windows: серийный номер тома; Linux: устройство
        warnings.append("same_volume")
    if any(_inside(real_root, cloud) for cloud in _cloud_roots()):
        warnings.append("cloud")
    return root, warnings


def _copy_bytes(source, staged, expected):
    """Байты проверенной автокопии во временный файл на другом диске; сброс на диск; сверка SHA-256 с описью."""
    digest = hashlib.sha256()
    with source.open("rb") as src, staged.open("xb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    if digest.hexdigest() != expected or storage._file_sha256(staged) != expected:
        raise Failure("copy")


def _owner(folder, manifest):
    """Метка папки данных в описи auto: по ней подпапка на другом диске признаётся своей."""
    if not manifest.get("mirror_owner"):
        manifest["mirror_owner"] = uuid.uuid4().hex
        folder.save_manifest(manifest)
    return manifest["mirror_owner"]


def _save(folder, update):
    status = folder.load_status()
    mirror = status.get("mirror") if isinstance(status.get("mirror"), dict) else {}
    for key, value in update.items():
        if value is None:
            mirror.pop(key, None)
        else:
            mirror[key] = value
    status["mirror"] = mirror
    folder.save_status(status)


def run(folder, config, now=None, tz=None):
    """Шаг копии на другой диск. Вызывать под замком папки auto (autobackup). Никогда не бросает. -> {"mirror": …}"""
    value = configured(config)
    if value is None:
        status = folder.load_status()
        if status.pop("mirror", None) is not None:
            folder.save_status(status)
        return {"mirror": "off"}
    today = autobackup.workday(autobackup._now(now), tz)
    at = autobackup._now(now).isoformat()
    try:
        root, warnings = check_place(value, folder.store_path.parent)
    except Unavailable:
        since = (folder.load_status().get("mirror") or {}).get("unavailable_since")
        _save(folder, {"unavailable_since": since or today})
        return {"mirror": "unavailable"}
    except Failure as failure:
        _save(folder, {"last_failure": {"kind": failure.kind, "at": at}, "unavailable_since": None})
        return {"mirror": "failed", "kind": failure.kind}
    mirror = MirrorFolder(root)
    staged = None
    try:
        manifest = folder.load_manifest()
        source = autobackup._newest(manifest)
        mirror.prepare()
        theirs = mirror.load_manifest()
        owner = _owner(folder, manifest)
        if theirs.get("owner") not in (None, owner) or (theirs.get("owner") is None and theirs["copies"]):
            raise Failure("foreign")
        theirs["owner"] = owner
        autobackup.recover(mirror, theirs)
        newest = autobackup._newest(theirs)
        question = None
        if source is None or (newest is not None and (
                newest["sha256"] == source["sha256"] or
                autobackup._utc(newest["last_write"]) >= autobackup._utc(source["last_write"]))):
            outcome = "current"
            if not mirror.manifest_path.exists():
                mirror.save_manifest(theirs)
        else:
            if shutil.disk_usage(mirror.path).free < source["size"] + 64 * 2 ** 20:
                raise Failure("space")
            staged = mirror.path / f"{source['name']}.incomplete-{uuid.uuid4().hex}"
            try:
                _copy_bytes(folder.path / source["name"], staged, source["sha256"])
            except OSError as exc:
                raise Failure("copy") from exc
            entry = {key: source[key] for key in ("name", "workday", "last_write", "taken_at", "size", "sha256",
                                                  "summary", "restore_verified") if key in source}
            entry["mirrored_at"] = at
            if newest is not None and newest["workday"] == entry["workday"]:
                autobackup._replace(mirror, theirs, newest, staged, entry)
                outcome = "replaced"
            else:
                autobackup._publish(mirror, theirs, staged, entry)
                outcome = "added"
            staged = None
            question = autobackup.rotate(mirror, theirs, autobackup._limit_bytes(config))
        _save(folder, {"last_success": {"at": at, "workday": today}, "last_failure": None,
                       "unavailable_since": None, "warnings": warnings or None,
                       "question": autobackup._question_view(question, theirs, config)})
        return {"mirror": outcome}
    except (Failure, autobackup.Failure) as failure:
        kind = failure.kind if failure.kind in FAILURES else "error"
    except autobackup.ManifestError:
        kind = "manifest"
    except Exception:                             # запуск программы важнее копии
        kind = "error"
    if staged is not None:
        storage._remove_unpublished(staged)       # свой незаконченный файл этого шага
    try:
        _save(folder, {"last_failure": {"kind": kind, "at": at}, "unavailable_since": None})
    except (OSError, storage.StorageError):
        pass
    return {"mirror": "failed", "kind": kind}


def reset_thinning(config):
    """Ответ «проредить ещё раз»: новый проход и в подпапке на другом диске, если она доступна."""
    value = configured(config)
    if value is None:
        return
    mirror = MirrorFolder(Path(value))
    if not mirror.manifest_path.is_file():
        return
    manifest = mirror.load_manifest()
    manifest["thin_done"], manifest["thin_pointer"] = False, 1
    mirror.save_manifest(manifest)


def view(status, now=None, tz=None):
    """Строки для экрана и self_check (без путей) и вопрос о пределе копий на другом диске."""
    mirror = status.get("mirror")
    if not isinstance(mirror, dict):
        return [], None
    lines = []
    failure = mirror.get("last_failure")
    if failure:
        lines.append("Копия резерва на другом диске: " + FAILURES.get(failure.get("kind"), FAILURES["error"]) +
                     f" (с {str(failure.get('at', ''))[:16].replace('T', ' ')} UTC).")
    since = mirror.get("unavailable_since")
    if since:
        today = autobackup.workday(autobackup._now(now), tz)
        try:
            gap = (datetime.fromisoformat(today) - datetime.fromisoformat(since)).days
        except ValueError:
            gap = 0
        if gap >= autobackup.STALE_DAYS:
            lines.append(f"Копия резерва на другом диске не делается {gap} дн.: папка не найдена (диск отключён?). "
                         "Подключите диск и перезапустите программу или укажите другую папку.")
    for warning in mirror.get("warnings") or ():
        if warning in WARNINGS:
            lines.append(WARNINGS[warning])
    question = mirror.get("question")
    if question:
        if question["kind"] == "small_limit":
            lines.append(f"Резервы на другом диске не помещаются в предел {question['limit_gb']:g} ГБ даже вчетвером: "
                         "самая старая и три свежие копии хранятся сверх предела. Поднимите предел.")
        else:
            lines.append(f"Резервы на другом диске достигли предела {question['limit_gb']:g} ГБ: хранится "
                         f"{question['count']} копий, самая старая — за {question['oldest_workday']}. Предел общий с "
                         "резервами на этом диске — ответ ниже относится к обоим.")
    return lines, question


def self_check_part(status):
    mirror = status.get("mirror")
    if not isinstance(mirror, dict):
        return {"enabled": False}
    return {"enabled": True, "last_success": mirror.get("last_success"),
            "last_failure": mirror.get("last_failure"), "unavailable_since": mirror.get("unavailable_since")}


def set_dir(value, path=None):
    """Экран: человек указывает папку (пусто — выключить). Проверка места до записи настройки; затем шаг в фоне.
    -> {"dir": …, "warnings": [...], "step": started|locked|off}"""
    import settings
    store_path = autobackup._store_path(path)
    value = (value or "").strip()
    if value:
        try:
            root, warnings = check_place(value, store_path.parent)
        except Unavailable:
            raise ValueError("такой папки нет — создайте её на другом диске и укажите снова") from None
        except Failure:
            raise ValueError(FAILURES["place"]) from None
        value = str(root)
    else:
        warnings = []
    settings.update(**{SETTING: value})
    shown = [WARNINGS[warning] for warning in warnings]
    if not value:
        folder = autobackup.Folder(store_path)
        if folder.path.is_dir() and folder.try_lock():
            try:
                status = folder.load_status()
                if status.pop("mirror", None) is not None:
                    folder.save_status(status)
            finally:
                folder.unlock()
        return {"dir": "", "warnings": [], "step": "off"}
    if not store_path.is_file():
        return {"dir": value, "warnings": shown, "step": "later"}     # базы ещё нет: копия появится после резерва
    return {"dir": value, "warnings": shown, "step": start_now(store_path)}


def start_now(store_path):
    """Шаг копии сразу (после смены папки): под замком auto, в фоне."""
    import settings
    folder = autobackup.Folder(store_path)
    try:
        folder.prepare()
        if not folder.try_lock():
            return "locked"
    except (OSError, storage.StorageError):
        return "locked"

    def step():
        try:
            run(folder, settings.load_config())
        finally:
            folder.unlock()
    threading.Thread(target=step, name="automirror", daemon=True).start()
    return "started"


def screen_view(path=None):
    """Экран: выбранная папка (путь показывается только на своём экране) и последняя удачная копия."""
    import settings
    store_path = autobackup._store_path(path)
    value = configured(settings.load_config())
    folder = autobackup.Folder(store_path)
    status = folder.load_status() if folder.path.is_dir() else {}
    part = self_check_part(status)
    return {"dir": value or "", "last_success": part.get("last_success")}
