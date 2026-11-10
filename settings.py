"""Настройки: значения по умолчанию заданы в коде, DATA_DIR\\config.json хранит только отличия пользователя
(результат теста скорости, свои пределы) и накладывается поверх них."""
import json
import sys

import paths

DEFAULTS = {"asr_backend": "auto", "asr_device": "GPU", "asr_model": "v3_e2e_rnnt",
            "pad_to_seconds": 0, "emotions": True,
            "diar_embeddings": "auto", "diar_ov_device": "GPU", "diar_chunks_per_batch": 16,
            "diar_segmentation_batch": 0, "torch_threads": 0,
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


def save_overrides(overrides):
    """Записывает отличия от умолчаний (ключи, которых нет в DEFAULTS, сохраняются как есть)."""
    diff = {k: v for k, v in overrides.items() if k != "schema_version" and DEFAULTS.get(k, object()) != v}
    out = {"schema_version": paths.DATA_SCHEMA, **diff}
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = paths.CONFIG_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(paths.CONFIG_FILE)
    return diff


def update(**kv):
    """Меняет отдельные настройки, остальные отличия сохраняются."""
    cur = load_overrides()
    cur.update(kv)
    return save_overrides(cur)
