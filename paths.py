"""Все тяжёлые и временные данные — внутри папки программы, чтобы удалялось одной папкой.

    <папка программы>\\cache\\huggingface   модели pyannote и onnx-asr   (HF_HOME)
    <папка программы>\\cache\\torch         кэш PyTorch                   (TORCH_HOME)
    <папка программы>\\cache\\gigaam        модели GigaAM для PyTorch
    <папка программы>\\cache\\openvino      скомпилированные под Arc/NPU модели
    <папка программы>\\cache\\pip           кэш установщика                (PIP_CACHE_DIR)

Данные пользователя лежат отдельно от программы, в DATA_DIR (не вложена в папку программы):
    RUS_TRANSCRIBER_DATA, иначе <домашняя папка>\\AppData\\Roaming\\rus-transcriber-data
    Путь строится от домашней папки, а не от переменных %APPDATA% / %LOCALAPPDATA%: в Claude Desktop из Microsoft
    Store они подменяются скрытой папкой внутри пакета Claude (Packages\\Claude_*\\LocalCache\\...). Проверка:
    папка должна быть видна в Проводнике по адресу %APPDATA%\\rus-transcriber-data; если нет, задайте RUS_TRANSCRIBER_DATA.
    DATA_DIR\\config.json                 только отличия от умолчаний (settings.py)
    DATA_DIR\\groups\\<группа>\\vocab.json, voices.json   словарь и голосовой профиль группы — один переносимый каталог
    DATA_DIR\\jobs                        статусы и логи задач
    DATA_DIR\\bench                       тексты и отчёт теста скорости
В папку программы кроме cache и .venv ничего не пишется. Все пути выбираются здесь.

Если переменная окружения уже задана (например, HF_HOME на другом диске), она не меняется.
Импортировать до torch / pyannote / huggingface_hub.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DATA_SCHEMA = 1


def _data_dir():
    env = os.environ.get("RUS_TRANSCRIBER_DATA", "").strip()
    if env:
        base = Path(env).expanduser()
    elif os.name == "nt":
        base = Path.home() / "AppData" / "Roaming" / "rus-transcriber-data"
    else:
        base = Path.home() / ".local" / "share" / "rus-transcriber-data"
    base = base.resolve()
    if base == HERE or HERE in base.parents:
        raise RuntimeError(f"Папка данных {base} лежит внутри папки программы {HERE}: задайте RUS_TRANSCRIBER_DATA "
                           "снаружи (при обновлении программы вложенные данные потерялись бы)")
    return base


# Задана ли папка данных явно — запоминается ОДИН раз при загрузке модуля. Позже переменную могут снять или
# подменить (тесты, дочерний процесс), но решение «это не стандартное место» уже принято: иначе настоящая папка
# данных выглядела бы «прежним местом по умолчанию» и переносилась бы в другую (случай 09.11.2026).
EXPLICIT = bool(os.environ.get("RUS_TRANSCRIBER_DATA", "").strip())
DATA_DIR = _data_dir()
GROUPS = DATA_DIR / "groups"
JOBS = DATA_DIR / "jobs"
BENCH = DATA_DIR / "bench"
CONFIG_FILE = DATA_DIR / "config.json"
for _d in (GROUPS, JOBS, BENCH):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# прежняя раскладка внутри папки программы — только источник миграции (migrate.py)
LEGACY_PROFILES = HERE / "profiles"
LEGACY_JOBS = HERE / "jobs"
LEGACY_CONFIG = HERE / "config.json"


def old_default_dirs():
    """Прежнее место по умолчанию (%LOCALAPPDATA%\\rus-transcriber-data, в том числе внутри пакета Store-версии
    Claude) — только если папка данных не задана явно; источник переноса в DATA_DIR."""
    if EXPLICIT:                 # папка данных задана явно — «прежнего места по умолчанию» нет
        return []
    out = []
    for p in (Path(os.environ.get("LOCALAPPDATA") or "") / "rus-transcriber-data",
              Path.home() / "AppData" / "Local" / "rus-transcriber-data",
              Path(os.environ.get("APPDATA") or "") / "rus-transcriber-data",
              Path.home() / "rus-transcriber-data"):
        if str(p) != "rus-transcriber-data":
            try:
                if p.is_dir() and p.resolve() != DATA_DIR and p.resolve() not in out:
                    out.append(p.resolve())
            except OSError:
                pass
    return out


def safe_group(group):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (group or "default"))


def group_dir(group):
    return GROUPS / safe_group(group)


def voices_path(group):
    return group_dir(group) / "voices.json"


def vocab_path(group):
    return group_dir(group) / "vocab.json"


def list_groups():
    """Группы, у которых есть словарь или голосовой профиль."""
    try:
        return sorted(d.name for d in GROUPS.iterdir()
                      if d.is_dir() and ((d / "vocab.json").exists() or (d / "voices.json").exists()))
    except OSError:
        return []


def pick_group(explicit=None, default=None):
    """Группа без словаря рядом с записью: явная → default_group из config.json → единственная в DATA_DIR.
    Нет групп — None (расшифровка без словаря). Несколько и ничего не выбрано — понятная ошибка."""
    for g in (explicit, default):
        if g:
            return safe_group(g)
    gs = list_groups()
    if len(gs) > 1:
        raise RuntimeError("В данных несколько групп (" + ", ".join(gs) + "): укажите group в start_transcription, "
                           "положите rus-transcriber.vocab.json рядом с записью или задайте default_group в config.json")
    return gs[0] if gs else None
GIGAAM_DIR = CACHE / "gigaam"
OPENVINO_DIR = CACHE / "openvino"

for _var, _sub in (("HF_HOME", "huggingface"), ("TORCH_HOME", "torch"), ("PIP_CACHE_DIR", "pip")):
    os.environ.setdefault(_var, str(CACHE / _sub))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Проверка HTTPS-сертификатов средствами Windows, а не встроенным OpenSSL Python.
# Windows сама достраивает актуальную цепочку сертификатов; OpenSSL иногда выбирает
# цепочку с просроченным промежуточным сертификатом и падает с CERTIFICATE_VERIFY_FAILED
# (так было при скачивании моделей GigaAM с сервера Сбера).
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
