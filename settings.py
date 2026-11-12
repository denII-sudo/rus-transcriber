"""Настройки: значения по умолчанию заданы в коде, DATA_DIR\\config.json хранит только отличия пользователя
(результат теста скорости, свои пределы) и накладывается поверх них."""
import json
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
            # порог совпадения текста с эталоном при выборе устройства в тесте скорости.
            # 0.97 — обычный компромисс; для работы, где важнее точность, поднимите до 0.99.
            "bench_min_similarity": 0.97,
            "confirm_mode": "auto", "confirm_timeout_sec": 180}


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
    cfg.update(load_overrides())
    return cfg


def save_overrides(overrides, explicit=()):
    """Записывает отличия от умолчаний (ключи, которых нет в DEFAULTS, сохраняются как есть).

    explicit — ключи, которые нужно сохранить, даже если они совпадают с умолчанием: так выбор
    этой машины (например, «процессор») остаётся явным и не меняется при обновлении программы.
    """
    keep = set(explicit)
    diff = {k: v for k, v in overrides.items()
            if k != "schema_version" and (k in keep or DEFAULTS.get(k, object()) != v)}
    out = {"schema_version": paths.DATA_SCHEMA, **diff}
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = paths.CONFIG_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(paths.CONFIG_FILE)
    return diff


def update(explicit=(), **kv):
    """Меняет отдельные настройки, остальные отличия сохраняются.

    explicit — что записать явно, даже если значение совпадает с умолчанием (выбор теста скорости).
    """
    cur = load_overrides()
    cur.update(kv)
    return save_overrides(cur, explicit=explicit)


def group_voice_settings(group, initialize=False):
    """Настройка группы фиксируется при первом использовании, затем не зависит от общего умолчания."""
    if not group or paths.safe_group(group) != group:
        raise ValueError("недопустимое имя группы")
    p = paths.group_dir(group) / "voice-settings.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or type(data.get("auto_learn")) is not bool:
                raise ValueError("неверный формат")
            return data
        except (OSError, ValueError) as ex:
            raise ValueError("настройки голосов группы повреждены; автопополнение запрещено") from ex
    value = load_config().get("voice_auto_learn_default", True)
    if type(value) is not bool:
        raise ValueError("неверное глобальное умолчание автопополнения")
    data = {"auto_learn": value, "reason": "default"}
    if initialize:
        save_group_voice_settings(group, data)
    return data


def save_group_voice_settings(group, data):
    if not group or paths.safe_group(group) != group or type(data.get("auto_learn")) is not bool:
        raise ValueError("неверные настройки группы")
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
        while not jobslots._try_lock(f):
            time.sleep(0.1)
        yield
