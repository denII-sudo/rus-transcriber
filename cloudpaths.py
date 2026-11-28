"""Облачные и сетевые папки (БД-8а): где лежит путь — на локальном диске, в облачной папке или на сетевом диске.

Только читает признаки, ничего не меняет: атрибуты и тег ссылки каталога и его родителей (запись каталога в родителе
через FindFirstFileW, содержимое не перечисляется, файлы не открываются, заглушки не скачиваются), реестр Windows,
переменные среды и файл настроек Dropbox. Ответ — вид (`cloud` / `network` / `local` / `unknown`), имя клиента и
признак; самого пути в ответе нет, в журнал он не пишется.

Признаки облака, от дешёвого к дорогому (любой из них — облако):
  1. тег ссылки Cloud Files (0x9000_x01A) или флаги RECALL_ON_DATA_ACCESS / RECALL_ON_OPEN / PINNED / UNPINNED
     у самого каталога или любого родителя (так помечены папки OneDrive, нового Яндекс Диска и других клиентов);
  2. путь под корнем синхронизации из реестра (SyncRootManager);
  3. путь под известной папкой клиента: OneDrive (среда и реестр), Яндекс Диск (реестр; у прежнего клиента тега нет —
     О-8), Dropbox (info.json);
  4. CfGetSyncRootInfoByPath (cldapi.dll) — только для каталога и только если 1–3 ничего не нашли;
  5. имя папки как у клиента облака (OneDrive, Яндекс.Диск, Dropbox, Google Drive…) — слабый признак: классические
     клиенты по-другому надёжно не определяются.
Сеть — путь UNC (\\\\сервер\\ресурс) или сетевой диск (GetDriveTypeW = 4).

Политика владельца (28.11.2026, ответ 2 — «Предупреждать»): папку данных и резервы в облаке программа не отвергает,
а предупреждает. Сетевой диск для базы по-прежнему отвергает storage.check_local_path.

Вне Windows признаков пока нет (ответ `unknown`, без предупреждения): ветка Linux (mountinfo, FUSE, папки клиентов) —
к пилоту на Linux (`opus-plan-3`, раздел 2). Результат кэшируется на процесс по пути.
"""
import json
import ntpath
import os
from pathlib import Path, PureWindowsPath

CLOUD, NETWORK, LOCAL, UNKNOWN = "cloud", "network", "local", "unknown"

_REPARSE_POINT = 0x400
_CLOUD_FLAGS = 0x40000 | 0x80000 | 0x100000 | 0x400000   # RECALL_ON_OPEN, PINNED, UNPINNED, RECALL_ON_DATA_ACCESS
_DRIVE_REMOTE = 4

# Имя клиента по фрагменту имени провайдера в реестре или имени папки (в нижнем регистре)
_CLIENTS = (("onedrive", "OneDrive"), ("yandex", "Яндекс Диск"), ("яндекс", "Яндекс Диск"), ("dropbox", "Dropbox"),
            ("googledrive", "Google Drive"), ("google drive", "Google Drive"), ("drivefs", "Google Drive"),
            ("icloud", "iCloud"), ("mail.ru", "Облако Mail.ru"), ("облако mail", "Облако Mail.ru"))
# Имена папок клиентов по умолчанию (компонент пути начинается с них)
_NAME_HINTS = (("onedrive", "OneDrive"), ("yandexdisk", "Яндекс Диск"), ("yandex.disk", "Яндекс Диск"),
               ("яндекс.диск", "Яндекс Диск"), ("яндекс диск", "Яндекс Диск"), ("dropbox", "Dropbox"),
               ("google drive", "Google Drive"), ("googledrive", "Google Drive"), ("мой диск", "Google Drive"),
               ("my drive", "Google Drive"), ("icloud drive", "iCloud"), ("iclouddrive", "iCloud"),
               ("облако mail.ru", "Облако Mail.ru"), ("cloud mail.ru", "Облако Mail.ru"))

_cache = {}


def clear_cache():
    _cache.clear()


def _client_name(text, default=None):
    low = (text or "").casefold()
    for part, name in _CLIENTS:
        if part in low:
            return name
    return default


def _norm(path):
    return str(PureWindowsPath(path)).rstrip("\\/").casefold()


def _under(path, root):
    a, b = _norm(path), _norm(ntpath.expandvars(str(root)))    # в реестре бывает %USERPROFILE%\…
    return bool(b) and (a == b or a.startswith(b + "\\"))


# ---------- системные вызовы Windows (в тестах подменяются) ----------
def _find_entry(path):
    """(атрибуты, тег ссылки) записи каталога/файла в его родителе или None, если её нет. Объект не открывается."""
    import ctypes
    from ctypes import wintypes

    class _WFD(ctypes.Structure):
        _fields_ = [("dwFileAttributes", wintypes.DWORD), ("ftCreationTime", wintypes.FILETIME),
                    ("ftLastAccessTime", wintypes.FILETIME), ("ftLastWriteTime", wintypes.FILETIME),
                    ("nFileSizeHigh", wintypes.DWORD), ("nFileSizeLow", wintypes.DWORD),
                    ("dwReserved0", wintypes.DWORD), ("dwReserved1", wintypes.DWORD),
                    ("cFileName", wintypes.WCHAR * 260), ("cAlternateFileName", wintypes.WCHAR * 14)]

    text = str(path)
    if "*" in text or "?" in text:
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.FindFirstFileW.restype = wintypes.HANDLE
    k32.FindFirstFileW.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(_WFD))
    k32.FindClose.argtypes = (wintypes.HANDLE,)
    data = _WFD()
    handle = k32.FindFirstFileW(text, ctypes.byref(data))
    if handle is None or handle == wintypes.HANDLE(-1).value:
        return None
    k32.FindClose(handle)
    tag = data.dwReserved0 if data.dwFileAttributes & _REPARSE_POINT else 0
    return data.dwFileAttributes, tag


def _drive_type(anchor):
    import ctypes
    return ctypes.windll.kernel32.GetDriveTypeW(str(anchor))


def _cf_sync_root(directory):
    """True — каталог внутри зарегистрированного корня синхронизации (Cloud Files), False — нет, None — проверить нельзя
    (нет cldapi.dll). Запрашиваются только сведения о корне (CF_SYNC_ROOT_INFO_BASIC)."""
    import ctypes
    from ctypes import wintypes
    try:
        cld = ctypes.WinDLL("cldapi")
        fn = cld.CfGetSyncRootInfoByPath
    except (OSError, AttributeError):
        return None
    fn.restype = ctypes.c_long
    fn.argtypes = (wintypes.LPCWSTR, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))
    buf = ctypes.create_string_buffer(4096)
    returned = wintypes.DWORD(0)
    return fn(str(directory), 0, buf, len(buf), ctypes.byref(returned)) == 0


def _reg_values(hive, key, value):
    import winreg
    try:
        with winreg.OpenKey(hive, key) as k:
            found, _ = winreg.QueryValueEx(k, value)
    except OSError:
        return None
    return found if isinstance(found, str) and found else None


def _reg_subkeys(hive, key):
    import winreg
    out = []
    try:
        with winreg.OpenKey(hive, key) as k:
            i = 0
            while True:
                try:
                    out.append(winreg.EnumKey(k, i))
                except OSError:
                    break
                i += 1
    except OSError:
        pass
    return out


def _sync_roots():
    """[(клиент, путь)] корней синхронизации из SyncRootManager; клиент — по имени ключа до первого «!»."""
    import winreg
    base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\SyncRootManager"
    out = []
    for name in _reg_subkeys(winreg.HKEY_LOCAL_MACHINE, base):
        provider = name.split("!")[0]
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base + "\\" + name + r"\UserSyncRoots") as k:
                j = 0
                while True:
                    try:
                        _sid, val, _t = winreg.EnumValue(k, j)
                    except OSError:
                        break
                    j += 1
                    if isinstance(val, str) and val:
                        out.append((_client_name(provider, "облако"), val))
        except OSError:
            continue
    return out


def _client_folders():
    """[(клиент, путь)] известных папок клиентов: среда и реестр OneDrive, реестр Яндекс Диска, info.json Dropbox."""
    import winreg
    out = []
    for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        if os.environ.get(var, "").strip():
            out.append(("OneDrive", os.environ[var].strip()))
    accounts = r"Software\Microsoft\OneDrive\Accounts"
    for name in _reg_subkeys(winreg.HKEY_CURRENT_USER, accounts):
        folder = _reg_values(winreg.HKEY_CURRENT_USER, accounts + "\\" + name, "UserFolder")
        if folder:
            out.append(("OneDrive", folder))
    for key in (r"Software\Yandex\Yandex.Disk.2", r"Software\Yandex\Yandex.Disk"):
        folder = _reg_values(winreg.HKEY_CURRENT_USER, key, "RootFolder")
        if folder:
            out.append(("Яндекс Диск", folder))
    # от домашней папки, а не %APPDATA%: в Claude Desktop из Microsoft Store она подменяется (см. paths.py)
    for base in (Path.home() / "AppData" / "Roaming", Path.home() / "AppData" / "Local"):
        try:
            info = json.loads((base / "Dropbox" / "info.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(info, dict):
            for account in info.values():
                if isinstance(account, dict) and isinstance(account.get("path"), str) and account["path"]:
                    out.append(("Dropbox", account["path"]))
    return out


# ---------- классификация ----------
def _result(kind, client=None, basis=None):
    return {"kind": kind, "client": client, "basis": basis}


def _absolute(path):
    p = Path(path)
    try:
        return p.resolve()
    except (OSError, RuntimeError):
        return p.absolute()


def _classify_windows(path):
    anchor = path.anchor
    if anchor.startswith("\\\\"):
        return _result(NETWORK, basis="unc")
    if not anchor:
        return _result(UNKNOWN)
    try:
        if _drive_type(anchor) == _DRIVE_REMOTE:
            return _result(NETWORK, basis="drive")
    except OSError:
        pass
    chain = [p for p in (path, *path.parents) if str(p) != p.anchor]
    existing_dir = None
    tagged = False
    for item in chain:
        try:
            entry = _find_entry(item)
        except OSError:
            entry = None
        if entry is None:
            continue
        attributes, tag = entry
        if existing_dir is None and attributes & 0x10:      # FILE_ATTRIBUTE_DIRECTORY
            existing_dir = item
        if (tag & 0xFFFF0FFF) == 0x9000001A or attributes & _CLOUD_FLAGS:
            tagged = True
            break
    for source, basis in ((_sync_roots, "sync_root"), (_client_folders, "client_folder")):
        try:
            roots = source()
        except OSError:
            roots = []
        for client, root in roots:
            if _under(path, root):
                return _result(CLOUD, client, "tag" if tagged else basis)
    if tagged:
        return _result(CLOUD, _name_client(path), "tag")
    if existing_dir is not None:
        try:
            if _cf_sync_root(existing_dir):
                return _result(CLOUD, _name_client(path), "cf_api")
        except OSError:
            pass
    client = _name_client(path)
    if client:
        return _result(CLOUD, client, "name")
    return _result(LOCAL)


def _name_client(path):
    for part in PureWindowsPath(path).parts[1:]:
        low = part.casefold()
        for hint, name in _NAME_HINTS:
            if low.startswith(hint):
                return name
    return None


def classify(path):
    """Где лежит путь: {"kind": cloud|network|local|unknown, "client": имя клиента или None, "basis": признак}.

    Путь может ещё не существовать (новый файл резерва): тогда смотрятся существующие родители. Сбой отдельного
    признака не прерывает проверку: остальные признаки всё равно проверяются."""
    if os.name != "nt":
        return _result(UNKNOWN)
    resolved = _absolute(path)
    key = _norm(resolved)
    if key not in _cache:
        try:
            _cache[key] = _classify_windows(resolved)
        except Exception:                # детектор не должен ломать вызывающего: неизвестно — без предупреждения
            _cache[key] = _result(UNKNOWN)
    return dict(_cache[key])


_WHAT = {"data": "Папка данных программы (база записей, голосовые профили)",
         "backups": "Папка резервов базы (db-backups)",
         "backup": "Файл резерва базы"}


def warning(path, what):
    """Предупреждение человеку по-русски или None, если путь локальный или проверить нельзя. Пути в тексте нет."""
    info = classify(path)
    subject = _WHAT[what]
    if info["kind"] == CLOUD:
        where = f"облачной папке {info['client']}" if info["client"] else "облачной папке"
        if info["basis"] == "name":
            where = f"папке, которая по имени похожа на облачную ({info['client']})"
        tail = ("клиент синхронизации может разлучить базу и её журнал (-wal) и повредить базу записей. "
                "Перенесите папку данных на локальный диск вне облака (переменная RUS_TRANSCRIBER_DATA или "
                "установщик с -DataDir) — см. README, «Папка данных».") if what == "data" else (
                "клиент синхронизации может выгрузить незаконченную копию или разлучить её с журналом. "
                "Держите резервы на локальном диске вне облака, а в облако кладите уже готовую копию.")
        return f"Внимание: {subject} лежит в {where}: {tail}"
    if info["kind"] == NETWORK:
        return (f"Внимание: {subject} лежит на сетевом диске: база SQLite в режиме WAL работает только на одном "
                "компьютере, программа не откроет там базу и не создаст резерв. Выберите локальный диск.")
    return None


def data_dir_warnings(data_dir):
    """Предупреждения о папке данных и папке резервов db-backups (если она указывает в другое место)."""
    out = []
    main = warning(data_dir, "data")
    if main:
        out.append(main)
    backups = Path(data_dir) / "db-backups"
    if os.path.lexists(backups):
        other = warning(backups, "backups")
        if other and classify(backups) != classify(data_dir):
            out.append(other)
    return out
