"""Настройки: значения по умолчанию заданы в коде, DATA_DIR\\config.json хранит только отличия пользователя
(результат теста скорости, свои пределы) и накладывается поверх них."""
import json
import math
import sys
import time
from contextlib import contextmanager

import paths

# Чистая установка — везде процессор. Тест скорости на каждом компьютере сам решает,
# не быстрее ли видеокарта, и записывает свои отличия в DATA_DIR\config.json. Эти значения
# уходят в Git как общие для всех; личный выбор машины в Git не попадает.
DEFAULTS = {"asr_backend": "auto", "asr_device": "CPU", "asr_model": "v3_e2e_rnnt",
            "asr_batch": 1, "asr_allow_npu": False,
            "pad_to_seconds": 0, "emotions": True,
            # слой признаков разговора для «горячих мест» (hotspots.py): считается при обработке,
            # показывается только по запросу
            "interaction": True,
            # вторая модель GigaAM и третья модель Whisper: голосование «два из трёх».
            # У каждой группы своё устройство: RNN-T, CTC и Whisper могут считаться на разном железе.
            "asr_second_model": "v3_e2e_ctc", "asr_second_backend": "auto",
            "asr_second_device": "", "asr_second_batch": 0,   # пусто/0 — как у основного прохода
            "asr_third_model": "onnx-community/whisper-large-v3-turbo", "asr_third_device": "CPU",
            "asr_third_mode": "all", "asr_third_quant": "fp16", "asr_third_quant_cpu": "int8",
            "asr_third_batch": 4, "asr_third_long_chunks": True, "asr_third_max_repeat": 3,
            "diar_embeddings": "auto", "diar_ov_device": "CPU", "diar_chunks_per_batch": 16,
            "diar_segmentation_batch": 0, "torch_threads": 0,
             "pipeline": "words", "max_heavy_jobs": 2, "default_group": "", "voice_auto_learn_default": True,
             "recording_backend": "legacy_files",
            "voice_quality_policy_default": "warn",
            # порог совпадения текста с эталоном при выборе устройства в тесте скорости.
            # 0.97 — обычный компромисс; для работы, где важнее точность, поднимите до 0.99.
            "bench_min_similarity": 0.97,
             "confirm_mode": "auto", "confirm_timeout_sec": 180}

# Настройки дополнительных возможностей, у которых нет обязательного значения по умолчанию.
# Неизвестные ключи остаются в файле для совместимости, но в эффективную конфигурацию не попадают.
EXTRA_CONFIG_TYPES = {
    "asr_quant": str, "asr_quant_cpu": str, "asr_language": str, "asr_third_backend": str,
    "word_win_near_change": float,
    "review_missed": bool, "voices": bool, "voices_debug": bool, "review_local": bool,
}
CONFIG_TYPES = {**{key: type(value) for key, value in DEFAULTS.items()}, **EXTRA_CONFIG_TYPES}


def _valid_config_value(key, value):
    expected = CONFIG_TYPES.get(key)
    if expected is None:
        return True
    if expected is bool:
        return type(value) is bool
    if expected is int:
        return type(value) is int
    if expected is float:
        return type(value) in (int, float) and math.isfinite(value)
    return isinstance(value, expected)


def load_overrides():
    """Только то, что записано в DATA_DIR\\config.json (без служебных ключей)."""
    p = paths.CONFIG_FILE
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError) as ex:
        print(f"config.json повреждён, беру настройки по умолчанию: {ex}", file=sys.stderr)
        return {}
    return {k: v for k, v in d.items() if k != "schema_version"} if isinstance(d, dict) else {}


def load_config():
    cfg = dict(DEFAULTS)
    for key, value in load_overrides().items():
        if key not in CONFIG_TYPES:
            continue
        if not _valid_config_value(key, value):
            print(f"config.json: неверный тип параметра {key}; используется значение по умолчанию",
                  file=sys.stderr)
            continue
        cfg[key] = value
    return cfg


def save_overrides(overrides, explicit=()):
    """Записывает отличия от умолчаний (ключи, которых нет в DEFAULTS, сохраняются как есть).

    explicit — ключи, которые нужно сохранить, даже если они совпадают с умолчанием: так выбор
    этой машины (например, «процессор») остаётся явным и не меняется при обновлении программы.
    """
    with _config_lock():
        return _save_overrides_locked(overrides, explicit)


def _config_lock():
    import jobslots
    @contextmanager
    def lock():
        path = paths.JOBS / "config.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as f:
            while not jobslots._try_lock(f):
                time.sleep(0.05)
            yield
    return lock()


def _save_overrides_locked(overrides, explicit=()):
    import os
    import tempfile
    usable = {}
    for key, value in overrides.items():
        if not _valid_config_value(key, value):
            print(f"config.json: неверный тип параметра {key}; значение не сохранено", file=sys.stderr)
            continue
        usable[key] = value
    keep = set(explicit)
    diff = {k: v for k, v in usable.items()
            if k != "schema_version" and (k in keep or DEFAULTS.get(k, object()) != v)}
    out = {"schema_version": paths.DATA_SCHEMA, **diff}
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=paths.DATA_DIR,
                                     prefix="config.", suffix=".tmp", delete=False) as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        tmp = f.name
    try:
        os.replace(tmp, paths.CONFIG_FILE)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return diff


def update(explicit=(), **kv):
    """Меняет отдельные настройки, остальные отличия сохраняются.

    explicit — что записать явно, даже если значение совпадает с умолчанием (выбор теста скорости).
    Ключи, уже записанные в config.json, остаются в нём, даже если совпадают с умолчанием:
    иначе выбор этой машины молча сменился бы вместе с умолчаниями новой версии программы.
    """
    invalid = [key for key, value in kv.items() if not _valid_config_value(key, value)]
    if invalid:
        raise ValueError("неверный тип настройки: " + ", ".join(sorted(invalid)))
    with _config_lock():
        cur = load_overrides()
        keep = set(cur) | set(explicit)
        cur.update(kv)
        return _save_overrides_locked(cur, explicit=keep)


def user_display_name(store):
    """Имя текущего пользователя установки хранится только в SQLite, не в config.json."""
    return store.current_identity()["display_name"]


def set_user_display_name(store, display_name):
    """Переименовать локального пользователя, сохраняя прежние снимки авторства."""
    identity = store.current_identity()
    store.rename_user(identity["user_id"], display_name)
    return store.current_identity()["display_name"]


def recording_backend():
    value = load_config().get("recording_backend", "legacy_files")
    if value not in ("legacy_files", "sqlite"):
        raise ValueError("recording_backend must be legacy_files or sqlite")
    return value


def set_recording_backend(value):
    if value not in ("legacy_files", "sqlite"):
        raise ValueError("recording_backend must be legacy_files or sqlite")
    import recordings
    if value == "sqlite" and not recordings.SQLITE_RUNTIME_READY:
        raise RuntimeError("SQLite backend is unavailable until all recording writers are routed")
    if value == "sqlite":
        store = recordings.open_store()
        try:
            store.current_identity()
        finally:
            store.close()
    return update(recording_backend=value)


def group_voice_settings(group, initialize=False):
    """Настройка группы фиксируется при первом использовании, затем не зависит от общего умолчания.

    Политика качества (`warn` или `reject_clipping`) без явного значения наследует общее умолчание;
    источник значения показывается, чтобы отличие умолчания от выбора человека было явным.
    """
    if not group or paths.safe_group(group) != group:
        raise ValueError("недопустимое имя группы")
    p = paths.group_dir(group) / "voice-settings.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or type(data.get("auto_learn")) is not bool:
                raise ValueError("неверный формат")
            policy = data.get("audio_quality_policy")
            if policy is None:
                policy, source = load_config().get("voice_quality_policy_default", "warn"), "default"
                if policy not in ("warn", "reject_clipping"):
                    raise ValueError("неверное глобальное умолчание качества")
            elif policy in ("warn", "reject_clipping"):
                source = "group"
            else:
                raise ValueError("неверная политика качества")
        except (OSError, ValueError) as ex:
            raise ValueError("настройки голосов группы повреждены; автопополнение запрещено") from ex
        return {**data, "audio_quality_policy": policy, "audio_quality_policy_source": source}
    value = load_config().get("voice_auto_learn_default", True)
    if type(value) is not bool:
        raise ValueError("неверное глобальное умолчание автопополнения")
    data = {"auto_learn": value, "reason": "default"}
    if initialize:
        save_group_voice_settings(group, data)
    policy = load_config().get("voice_quality_policy_default", "warn")
    if policy not in ("warn", "reject_clipping"):
        raise ValueError("неверное глобальное умолчание качества")
    return {**data, "audio_quality_policy": policy, "audio_quality_policy_source": "default"}


def save_group_voice_settings(group, data):
    if not group or paths.safe_group(group) != group or type(data.get("auto_learn")) is not bool:
        raise ValueError("неверные настройки группы")
    if data.get("audio_quality_policy") not in (None, "warn", "reject_clipping"):
        raise ValueError("неверная политика качества")
    p = paths.group_dir(group) / "voice-settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=p.parent, delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        tmp = f.name
    os.replace(tmp, p)
    return data


@contextmanager
def group_voice_lock(group):
    """Межпроцессная блокировка общего профиля и настройки группы."""
    if not group or paths.safe_group(group) != group:
        raise ValueError("недопустимое имя группы")
    import jobslots
    directory = paths.JOBS / "voice-locks"
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / (group + ".lock"), "a+") as f:
        deadline = time.monotonic() + 60
        while not jobslots._try_lock(f):
            if time.monotonic() >= deadline:
                raise TimeoutError("профиль занят другим процессом; повторите позже")
            time.sleep(0.1)
        yield
