"""Файл опыта лаборатории: разбор, проверка и перевод ролей моделей в настройки программы.

Модуль чистый: не импортирует программу и ничего не пишет. Формат файла описан в lab/README.md.
Конфигурация опыта задаёт модели по ролям (основная, вторая, третья) и отдельные настройки;
всё, что не задано, берётся из config.json, скопированного в песочницу, — то есть из выбора
теста скорости этой машины. Результат перевода — обычный словарь настроек, который получает
рабочий конвейер (worker.run_words).
"""
import copy
import hashlib
import json
import re

SCHEMA = "lab-experiment-1"

# Эти настройки относятся к общим для всех конфигураций этапам (разделение говорящих, голоса)
# или выключены лабораторией: в конфигурации их менять нельзя, иначе общий этап не был бы общим.
SHARED_KEYS = frozenset({
    "diar_embeddings", "diar_ov_device", "diar_chunks_per_batch", "diar_segmentation_batch",
    "torch_threads", "voices", "voices_debug", "default_group", "pipeline", "recording_backend",
    "emotions", "max_heavy_jobs", "word_win_near_change",
})
# Настройки, которые лаборатория ставит всегда: эмоции не нужны для опыта и могут потребовать
# скачивания модели; конвейер — только пословный.
FORCED = {"emotions": False, "pipeline": "words"}

PRIMARY_KEYS = {"model", "backend", "device", "batch", "quant"}
SECOND_KEYS = {"model", "backend", "device", "batch"}
THIRD_KEYS = {"engine", "model", "device", "quant", "mode", "max_repeat", "long_chunks"}
THIRD_ENGINES = ("whisper-ct2", "whisper-onnx", "onnx")
THIRD_MODES = ("all", "doubt")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class SpecError(ValueError):
    """Файл опыта неверен; текст объясняет, что исправить."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_keys(where, value, allowed, required=()):
    if not isinstance(value, dict):
        raise SpecError(f"{where}: ожидается объект")
    extra = sorted(set(value) - set(allowed))
    if extra:
        raise SpecError(f"{where}: неизвестные поля {', '.join(extra)}")
    for key in required:
        if not value.get(key):
            raise SpecError(f"{where}: нужно поле {key}")


def parse(text):
    """Текст файла опыта -> проверенный словарь. Ошибка формата — SpecError."""
    try:
        data = json.loads(text)
    except ValueError as ex:
        raise SpecError(f"файл опыта не разобран как JSON: {ex}") from ex
    if not isinstance(data, dict):
        raise SpecError("файл опыта: ожидается объект")
    if data.get("schema") != SCHEMA:
        raise SpecError(f"файл опыта: поле schema должно быть «{SCHEMA}»")
    _check_keys("файл опыта", data, {"schema", "name", "description", "configurations",
                                     "estimate_sec_per_audio_hour", "allow_torch_fallback"},
                required=("name",))
    confs = data.get("configurations")
    if not isinstance(confs, list) or not confs:
        raise SpecError("файл опыта: нужен непустой список configurations")
    est = data.get("estimate_sec_per_audio_hour", {})
    if not isinstance(est, dict) or any(not isinstance(v, (int, float)) or v < 0 for v in est.values()):
        raise SpecError("estimate_sec_per_audio_hour: ожидаются неотрицательные числа")
    if not isinstance(data.get("allow_torch_fallback", False), bool):
        raise SpecError("allow_torch_fallback: ожидается true или false")
    seen = set()
    for n, conf in enumerate(confs, 1):
        where = f"конфигурация №{n}"
        _check_keys(where, conf, {"id", "label", "roles", "settings", "note"}, required=("id", "roles"))
        cid = conf["id"]
        if not isinstance(cid, str) or not ID_RE.match(cid):
            raise SpecError(f"{where}: id — латиница в нижнем регистре, цифры, «-» и «_», до 40 знаков")
        if cid in seen:
            raise SpecError(f"{where}: id «{cid}» повторяется")
        seen.add(cid)
        where = f"конфигурация «{cid}»"
        roles = conf["roles"]
        _check_keys(where + ": roles", roles, {"primary", "second", "third"}, required=("primary",))
        _check_keys(where + ": primary", roles["primary"], PRIMARY_KEYS, required=("model",))
        second = roles.get("second")
        third = roles.get("third")
        if second is not None:
            _check_keys(where + ": second", second, SECOND_KEYS, required=("model",))
            if second["model"] == roles["primary"]["model"]:
                raise SpecError(f"{where}: вторая модель совпадает с основной — конвейер её пропустит")
        if third is not None:
            _check_keys(where + ": third", third, THIRD_KEYS, required=("engine",))
            if second is None:
                raise SpecError(f"{where}: третья модель без второй не работает (конвейер голосует, "
                                "только когда есть вторая гипотеза)")
            if third["engine"] not in THIRD_ENGINES:
                raise SpecError(f"{where}: third.engine — одно из {', '.join(THIRD_ENGINES)}")
            if third["engine"] == "onnx":
                if not third.get("model"):
                    raise SpecError(f"{where}: для third.engine=onnx нужно поле model")
                if str(third.get("device", "CPU")).upper() != "CPU":
                    raise SpecError(f"{where}: third.engine=onnx считается только на CPU "
                                    "(на GPU конвейер пробует путь Whisper для видеокарты)")
            if third.get("mode", "all") not in THIRD_MODES:
                raise SpecError(f"{where}: third.mode — all или doubt")
        settings = conf.get("settings", {})
        if not isinstance(settings, dict):
            raise SpecError(f"{where}: settings — объект")
        bad = sorted(set(settings) & (SHARED_KEYS | set(FORCED)))
        if bad:
            raise SpecError(f"{where}: настройки {', '.join(bad)} относятся к общим этапам или "
                            "задаются лабораторией; в конфигурации их менять нельзя")
    return data


def load(path):
    from pathlib import Path
    text = Path(path).read_text(encoding="utf-8-sig")
    return parse(text), sha256_text(text)


def effective_settings(base, conf, config_types=None):
    """Настройки конвейера для конфигурации: config.json песочницы + роли + settings + обязательные.

    config_types — settings.CONFIG_TYPES: неизвестный ключ в settings — ошибка опечатки."""
    cfg = copy.deepcopy(dict(base))
    roles = conf["roles"]
    p = roles["primary"]
    cfg["asr_model"] = p["model"]
    for key, target in (("backend", "asr_backend"), ("device", "asr_device"), ("batch", "asr_batch")):
        if key in p:
            cfg[target] = p[key]
    if "quant" in p:                      # None — fp32; действует и на второй проход (так устроен worker)
        cfg["asr_quant"] = cfg["asr_quant_cpu"] = p["quant"]
    s = roles.get("second")
    if s is None:
        cfg["asr_second_model"] = ""
    else:
        cfg["asr_second_model"] = s["model"]
        for key, target in (("backend", "asr_second_backend"), ("device", "asr_second_device"),
                            ("batch", "asr_second_batch")):
            if key in s:
                cfg[target] = s[key]
    t = roles.get("third")
    if t is None:
        cfg["asr_third_mode"] = "off"
    else:
        engine = t["engine"]
        cfg["asr_third_mode"] = t.get("mode", "all")
        if engine == "whisper-ct2":
            cfg["asr_third_backend"] = "whisper-ct2"
        else:
            cfg["asr_third_backend"] = ""
        if t.get("model"):
            cfg["asr_third_model"] = t["model"]
        if engine == "onnx":
            cfg["asr_third_device"] = "CPU"
            cfg["asr_third_quant_cpu"] = t.get("quant")       # None — fp32
            cfg["asr_third_long_chunks"] = bool(t.get("long_chunks", False))
        else:
            if "device" in t:
                cfg["asr_third_device"] = t["device"]
            if "quant" in t:
                cfg["asr_third_quant"] = cfg["asr_third_quant_cpu"] = t["quant"]
            if "long_chunks" in t:
                cfg["asr_third_long_chunks"] = bool(t["long_chunks"])
        if "max_repeat" in t:
            cfg["asr_third_max_repeat"] = int(t["max_repeat"])
    for key, value in (conf.get("settings") or {}).items():
        if config_types is not None and key not in config_types:
            raise SpecError(f"конфигурация «{conf['id']}»: неизвестная настройка {key}")
        cfg[key] = value
    cfg.update(FORCED)
    return cfg


def config_hash(conf, cfg):
    """Контрольная сумма конфигурации: роли, её настройки и итоговые настройки конвейера.

    Изменились роли или config.json песочницы — сумма другая, пара считается заново."""
    body = {"roles": conf["roles"], "settings": conf.get("settings") or {}, "effective": cfg}
    return sha256_text(canonical(body))


def role_keys(conf, cfg):
    """Короткие ключи этапов для оценки времени: {"primary": ..., "second": ..., "third": ...}."""
    roles = conf["roles"]
    out = {"primary": "primary:" + str(cfg.get("asr_model"))}
    out["second"] = "second:" + str(cfg["asr_second_model"]) if roles.get("second") else None
    t = roles.get("third")
    if t is None:
        out["third"] = None
    else:
        name = "whisper-ct2" if t["engine"] == "whisper-ct2" else str(cfg.get("asr_third_model"))
        out["third"] = "third:" + name
        out["third_mode"] = cfg.get("asr_third_mode", "all")
    return out
