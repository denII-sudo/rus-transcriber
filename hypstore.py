"""Сохранение гипотез моделей на диск (РАР-4, этап 4.2, шаг 3): файл hyp-2 в папке данных и его удаление.

Включается настройкой save_hypotheses (по умолчанию выключена). Гипотезы содержат полный текст записи,
поэтому лежат только в папке данных: <папка данных>\\hypotheses\\<unit_id>\\<run_id>.json, где unit_id —
первые 12 знаков SHA-256 звука (моно 16 кГц s16le, как в паспорте теста скорости runinfo.audio_identity),
run_id — как runinfo.new_run_id. Связь с задачей и записью (job_id, recording_id, source_revision) —
внутри файла; путей к аудио и имён файлов нет.

Что пишет модуль: файлы прогонов и два кэша в той же папке — _artifacts.json (хэши файлов моделей,
пересчёт только при смене размера или времени изменения) и _testbed.json (конфигурация компьютера,
опрос через PowerShell долгий; пересчёт раз в сутки). Обезличенный testbed_id берётся тем же способом,
что в тесте скорости (соль в <папка данных>\\bench\\testbed.json — создаётся, если её ещё нет).
Что удаляет: только файлы прогонов внутри hypotheses\\ и только через cleanup (отчёт без применения,
затем apply=True после подтверждения человека в MCP и rt.py). Ссылкам (junction/symlink) не следует.

Выгрузки (data_io.export_data), резервные копии, облако и модельные задания этот каталог не читают.
"""
import json
import os
import re
import time
from pathlib import Path

import hypotheses as H

ARTIFACTS_FILE = "_artifacts.json"
TESTBED_FILE = "_testbed.json"
TESTBED_MAX_AGE_SEC = 24 * 3600
UNIT_RE = re.compile(r"^[0-9a-f]{12}$")
RUN_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}\.json$")
WEIGHT_SUFFIXES = (".onnx", ".onnx_data", ".data", ".bin", ".safetensors", ".pt", ".ckpt", ".xml")
MIN_WEIGHT_BYTES = 1_000_000          # мелкие служебные файлы (конфигурации, словари) не хэшируются


def enabled(cfg):
    """Сохранять ли гипотезы: только явное true в настройках."""
    return isinstance(cfg, dict) and cfg.get("save_hypotheses") is True


def root():
    import paths
    return paths.HYPOTHESES


def pcm_bytes(audio):
    """Звук (float32, моно 16 кГц) -> байты s16le — тем же правилом, что в bench.py."""
    import numpy as np
    return np.clip(np.rint(np.asarray(audio, dtype=np.float32) * 32767.0), -32768, 32767).astype("<i2").tobytes()


def _inside(base, p):
    """Путь p внутри base (после разрешения ссылок)."""
    try:
        Path(os.path.realpath(p)).relative_to(Path(os.path.realpath(base)))
        return True
    except ValueError:
        return False


def _write_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _read_json(path, default):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, type(default)) else default
    except (OSError, ValueError):
        return default


# ── артефакты моделей ────────────────────────────────────────────────────────────────────────────

def _norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _own_file(name, filename):
    """Файл принадлежит модели с коротким именем: v3_e2e_ctc.int8.onnx, v3_e2e_rnnt_encoder.onnx — да;
    у v3_e2e_ctc файл v3_e2e_rnnt_encoder.onnx — нет (в одном хранилище gigaam-v3-onnx лежат несколько моделей)."""
    low, n = filename.lower(), name.lower()
    return low.startswith(n + ".") or low.startswith(n + "_")


def _model_files(model, cache_dir, engine=None):
    """Файлы весов модели в кэше программы. -> [(относительное имя, путь)].

    Имя с «/» (Whisper) — снимок Hugging Face этого репозитория целиком. Короткое имя GigaAM
    (v3_e2e_ctc, multilingual_large_ctc) — файлы, имя которых начинается с имени модели, в снимках
    istupakov/gigaam-* (варианты v3 — в общем gigaam-v3-onnx) и в onnx-models; веса PyTorch (gigaam/*.ckpt) —
    только при движке torch, при других движках они не используются. engine=None — все места."""
    cache_dir = Path(cache_dir)
    name = str(model or "")
    if not name:
        return []
    hub = Path(os.environ.get("HF_HOME") or (cache_dir / "huggingface")) / "hub"
    exact = "/" in name
    short = _norm(name.rsplit("/", 1)[-1])
    if not short:
        return []
    torch = engine is None or str(engine).startswith("torch")
    onnx = engine is None or not str(engine).startswith("torch")
    roots = []
    if hub.is_dir() and (exact or onnx):
        for d in sorted(hub.glob("models--*")):
            repo = d.name[len("models--"):].replace("--", "/")
            if exact and str(engine or "").startswith("whisper-ct2"):
                # faster-whisper грузит свою сборку той же модели: …/faster-whisper-large-v3-turbo
                hit = repo.rsplit("/", 1)[-1] == "faster-" + name.rsplit("/", 1)[-1]
            elif exact:
                hit = repo == name
            else:
                tail = _norm(repo.rsplit("/", 1)[-1])
                hit = tail.startswith("gigaam") and (short in tail or (name.startswith("v3_")
                                                                          and tail == "gigaamv3onnx"))
            if hit:
                roots.append((d / "snapshots", "hf/" + repo, exact))
    if not exact:
        if onnx:
            roots.append((cache_dir / "onnx-models", "onnx-models", False))
        if torch:
            roots.append((cache_dir / "gigaam", "gigaam", False))
    found = []
    for base, label, whole in roots:
        if not base.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(base, followlinks=False):
            for fn in sorted(files):
                p = Path(dirpath) / fn
                rel = p.relative_to(base).as_posix()
                if (whole or _own_file(name, fn)) and p.suffix.lower() in WEIGHT_SUFFIXES:
                    found.append((f"{label}/{rel}", p))
    return found


def model_artifact(model, cache_dir, cache, engine=None):
    """Артефакт модели: файлы весов (имя в кэше, размер, время изменения, SHA-256). cache — словарь
    _artifacts.json, дополняется на месте. Хэш считается один раз на файл; пересчёт — при смене размера
    или времени изменения. -> словарь или not_measured."""
    import runinfo
    files = []
    for rel, p in _model_files(model, cache_dir, engine):
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_size < MIN_WEIGHT_BYTES:
            continue
        known = cache.get(rel)
        if not (isinstance(known, dict) and known.get("size") == st.st_size
                and known.get("mtime_ns") == st.st_mtime_ns and known.get("sha256")):
            digest = runinfo.sha256_file(p)
            if digest is None:
                continue
            known = cache[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
        files.append({"file": rel, "size_bytes": known["size"], "sha256": known["sha256"]})
    if not files:
        return H.not_measured("файлы модели не найдены в кэше программы")
    return {"files": files}


# ── сведения о прогоне ───────────────────────────────────────────────────────────────────────────

def testbed(here, data_dir, run=None, now=None):
    """Конфигурация компьютера (runinfo.hardware) с кэшем в hypotheses\\_testbed.json. -> (testbed_id, hardware)."""
    import paths
    import runinfo
    tid = runinfo.testbed_id(paths.BENCH)
    p = Path(data_dir) / TESTBED_FILE
    cached = _read_json(p, {})
    now = time.time() if now is None else now
    if cached.get("testbed_id") == tid and now - float(cached.get("taken_ts") or 0) < TESTBED_MAX_AGE_SEC \
            and isinstance(cached.get("hardware"), dict):
        return tid, cached["hardware"]
    hw = runinfo.hardware(run=run)
    try:
        _write_json_atomic(p, {"schema": 1, "testbed_id": tid, "taken_ts": now, "hardware": hw})
    except OSError:
        pass
    return tid, hw


CONFIG_KEYS = ("asr_model", "asr_backend", "asr_device", "asr_quant", "asr_quant_cpu", "asr_batch", "asr_allow_npu",
               "pad_to_seconds", "asr_language", "asr_second_model", "asr_second_backend", "asr_second_device",
               "asr_second_batch", "asr_third_model", "asr_third_backend", "asr_third_device", "asr_third_mode",
               "asr_third_quant", "asr_third_quant_cpu", "asr_third_batch", "asr_third_long_chunks",
               "asr_third_max_repeat", "torch_threads", "pipeline", "word_win_near_change")


def config_snapshot(cfg, vocab=None):
    """Параметры, влияющие на распознавание и сведение. Словарь группы — только сумма и число слов."""
    import hashlib
    import words as W
    out = {k: cfg.get(k) for k in CONFIG_KEYS}
    out["chunking"] = {"max_len": getattr(W, "CHUNK_MAX", None), "min_len": getattr(W, "CHUNK_MIN", None),
                       "overlap": getattr(W, "OVERLAP", None), "whisper_max": getattr(W, "WHISPER_MAX", None),
                       "whisper_min": getattr(W, "WHISPER_MIN", None)}
    out["thresholds"] = {"conf_min": getattr(W, "CONF_MIN", None)}
    if vocab:
        try:
            text = json.dumps(vocab, ensure_ascii=False, sort_keys=True, default=str)
            out["vocab"] = {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            "entries": len(vocab) if hasattr(vocab, "__len__") else None}
        except (TypeError, ValueError):
            out["vocab"] = H.not_measured("словарь не сериализуется")
    else:
        out["vocab"] = H.not_available("словаря нет")
    return out


class UnsafeRecord(ValueError):
    """В записи гипотез строка, похожая на путь, или имя исходного аудиофайла. Текст — без самих строк."""


# Абсолютный путь: буква диска (C:\ или C:/), сетевой путь (\\сервер), папки пользователей macOS и Linux.
_ABS_PATH_RE = re.compile(r"(?i)(?<![a-z0-9])[a-z]:[\\/]|\\\\[^\\\s]|/users/|/home/")


def check_record(record, forbid=()):
    """Строки и ключи записи без путей и имён исходных файлов. forbid — имена файлов с расширением
    (основа имени без расширения не проверяется: она может совпасть со словом стенограммы).
    -> список видов нарушений (без самих строк): absolute_path, home_dir, audio_name."""
    home = str(Path.home()).lower()
    homes = {h for h in (home, home.replace("\\", "/")) if len(h) > 3}
    names = {str(n).lower() for n in forbid if n and str(n).strip()}
    found = []

    def look(text):
        low = text.lower()
        if _ABS_PATH_RE.search(low):
            found.append("absolute_path")
        if any(h in low for h in homes):
            found.append("home_dir")
        if any(n in low for n in names):
            found.append("audio_name")

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                look(str(k))
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        elif isinstance(obj, str):
            look(obj)
    walk(record)
    return found


def write(record, unit_id, run_id, data_dir=None, forbid=()):
    """Записать файл прогона один раз, атомарно. -> путь. Существующий файл не перезаписывается.
    Перед записью — check_record: при строке, похожей на путь, или имени исходного аудио (forbid) файла нет."""
    base = Path(data_dir) if data_dir is not None else root()
    if not UNIT_RE.match(unit_id) or not RUN_RE.match(run_id + ".json"):
        raise ValueError("неверный unit_id или run_id")
    problems = check_record(record, forbid)
    if problems:
        raise UnsafeRecord(f"запись гипотез не сохранена: {', '.join(sorted(set(problems)))}")
    folder = base / unit_id
    folder.mkdir(parents=True, exist_ok=True)
    if not _inside(base, folder):
        raise ValueError("папка гипотез вне hypotheses")
    path = folder / (run_id + ".json")
    if path.exists():
        raise FileExistsError("файл прогона уже есть")
    _write_json_atomic(path, record)
    return path


# ── удаление ─────────────────────────────────────────────────────────────────────────────────────

def _run_files(base, unit_id=None):
    base = Path(base)
    if not base.is_dir():
        return []
    out = []
    for unit in sorted(os.listdir(base)):
        folder = base / unit
        if not UNIT_RE.match(unit) or (unit_id and unit != unit_id):
            continue
        if folder.is_symlink() or not folder.is_dir() or not _inside(base, folder):
            continue
        for fn in sorted(os.listdir(folder)):
            p = folder / fn
            if RUN_RE.match(fn) and p.is_file() and not p.is_symlink() and _inside(base, p):
                out.append(p)
    return out


def cleanup(unit_id=None, older_than_days=None, apply=False, data_dir=None, now=None):
    """Удаление файлов гипотез: всех, одной записи (unit_id) или старше older_than_days дней по времени
    изменения файла. apply=False — только отчёт. Кэши _artifacts.json и _testbed.json не трогаются.
    -> {"files", "bytes", "units", "applied"}."""
    if unit_id is not None and not UNIT_RE.match(str(unit_id)):
        raise ValueError("unit_id — 12 знаков 0-9a-f")
    base = Path(data_dir) if data_dir is not None else root()
    cutoff = None
    # 0 дней — все файлы сразу, без сравнения времени: иначе только что записанный файл может оказаться
    # «новее» текущего времени из-за разной точности часов файловой системы и time.time()
    if older_than_days is not None and float(older_than_days) > 0:
        cutoff = (time.time() if now is None else now) - max(0.0, float(older_than_days)) * 86400
    n = size = 0
    units = set()
    for p in _run_files(base, unit_id):
        try:
            st = p.stat()
        except OSError:
            continue
        if cutoff is not None and st.st_mtime > cutoff:
            continue
        n += 1
        size += st.st_size
        units.add(p.parent.name)
        if apply:
            p.unlink()
            try:
                p.parent.rmdir()          # пустая папка записи убирается; непустая остаётся
            except OSError:
                pass
    return {"unit_id": unit_id, "older_than_days": older_than_days, "files": n, "bytes": size,
            "units": len(units), "applied": bool(apply)}
