"""Гипотезы моделей (РАР-4, этап 4.2): запись hyp-2 прогона расшифровки — в базе записей, частью записи (Г-1).

Включается настройкой save_hypotheses (по умолчанию выключена). Гипотезы содержат полный текст записи и
хранятся только в базе записей (state.sqlite3 папки данных), в таблице hypotheses: строка на прогон, с номером
записи (recording_id), звука (sound_id — первые 12 знаков SHA-256 звука, моно 16 кГц s16le, как в паспорте теста
скорости runinfo.audio_identity), задачи и прогона (run_id — как runinfo.new_run_id). Внутри записи те же
ids.job_id и ids.recording_id; source_revision нет: запись расшифровки новая, её ревизия 1 и есть итог прогона.
Путей к аудио и имён файлов нет. Удаление записи (когда оно появится) обязано удалить её гипотезы: внешний ключ
таблицы не даёт удалить запись с гипотезами.

Что пишет модуль: строку гипотез в базу (в базе до Г-1 — сначала резерв базы и пустая таблица,
recordings.ensure_hypotheses) и два кэша в папке <папка данных>\\hypotheses — _artifacts.json (хэши файлов
моделей, пересчёт только при смене размера или времени изменения) и _testbed.json (конфигурация компьютера,
опрос через PowerShell долгий; пересчёт раз в сутки). Обезличенный testbed_id берётся тем же способом, что в
тесте скорости (соль в <папка данных>\\bench\\testbed.json — создаётся, если её ещё нет).

Прежние файлы прогонов (<папка данных>\\hypotheses\\<unit_id>\\<run_id>.json, до Г-1) читаются и удаляются
очисткой, новые не пишутся; их перенос в базу — Г-2. Удаляет модуль только через cleanup (отчёт без применения,
затем apply=True после подтверждения человека в MCP и rt.py) — строки базы и прежние файлы прогонов, сразу
и навсегда. Ссылкам (junction/symlink) не следует.

Выгрузки (data_io.export_data), облако и модельные задания гипотезы не читают. Резерв базы (свой, для
восстановления) содержит их вместе с записью.
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


def _store_path(store_path=None):
    if store_path is not None:
        return Path(store_path)
    import paths
    return paths.STATE_DB


def write_db(record, unit_id, run_id, recording_id, forbid=(), job_id=None, store_path=None):
    """Записать гипотезы прогона в базу записей, частью записи recording_id. Один раз на run_id.
    Перед записью — check_record: при строке, похожей на путь, или имени исходного аудио (forbid) записи нет.
    В базе до Г-1 сначала резерв базы и пустая таблица (recordings.ensure_hypotheses).
    -> {"run_id", "recording_id", "size_bytes"}."""
    import recordings
    if not UNIT_RE.match(str(unit_id)) or not RUN_RE.match(f"{run_id}.json"):
        raise ValueError("неверный unit_id или run_id")
    if not isinstance(recording_id, str) or not recording_id:
        raise ValueError("гипотезы сохраняются только для записи в базе: нет номера записи")
    problems = check_record(record, forbid)
    if problems:
        raise UnsafeRecord(f"запись гипотез не сохранена: {', '.join(sorted(set(problems)))}")
    store = recordings.open_store(_store_path(store_path))
    try:
        recordings.ensure_hypotheses(store)
        return store.add_hypotheses(run_id, recording_id, unit_id, record, job_id=job_id,
                                    fmt=str(record.get("schema") or "hyp-2"))
    finally:
        store.close()


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


def _file_recording_id(p):
    """ids.recording_id прежнего файла прогона или None (файл не читается, номера нет)."""
    data = _read_json(p, {})
    ids = data.get("ids")
    rid = ids.get("recording_id") if isinstance(ids, dict) else None
    return rid if isinstance(rid, str) and rid else None


def _cutoff(older_than_days, now):
    # 0 дней — всё сразу, без сравнения времени: иначе только что записанное может оказаться «новее»
    # текущего времени из-за разной точности часов файловой системы и time.time()
    if older_than_days is not None and float(older_than_days) > 0:
        return (time.time() if now is None else now) - max(0.0, float(older_than_days)) * 86400
    return None


def recording_entry(recording_id, label, runs=0, size=0):
    """Строка перечня записей для человека: название (title, иначе имя файла аудио без папки), дата записи
    (recorded_at), иначе дата обработки (первая ревизия текста), номер записи. Ни путей, ни текста стенограммы."""
    label = label or {}
    name = label.get("title") or label.get("file_name") or "без названия"
    if label.get("recorded_at"):
        date, kind = label["recorded_at"][:10], "recorded"
    elif label.get("processed_at"):
        date, kind = label["processed_at"][:10], "processed"
    else:
        date, kind = None, None
    return {"recording_id": recording_id, "name": name, "date": date, "date_kind": kind, "runs": runs,
            "bytes": size}


def entry_text(entry):
    """«Название — записано ГГГГ-ММ-ДД (номер …)»: подпись записи в окне подтверждения и отчёте."""
    date = {"recorded": "записано ", "processed": "обработано "}.get(entry.get("date_kind"), "") + (
        entry.get("date") or "дата неизвестна")
    return f"«{entry['name']}» — {date} (номер {entry['recording_id']})"


def _database_part(recording_id, unit_id, cutoff, apply, store_path, with_recordings=False):
    """Строки гипотез базы под отбор: сводка, при apply — удаление. Базы нет — нули. with_recordings=True —
    (сводка, перечень записей recording_entry той же выборки по дате, названию и номеру; при apply — снятый до
    удаления в той же транзакции; при отборе по записи без строк базы — подпись этой записи с нулями)."""
    import datetime as dt
    import storage
    path = _store_path(store_path)
    if not path.is_file():
        empty = {"runs": 0, "recordings": 0, "bytes": 0}
        return (empty, []) if with_recordings else empty
    before = dt.datetime.fromtimestamp(cutoff, dt.timezone.utc) if cutoff is not None else None
    store = storage.open_store(path) if apply else storage.open_store_readonly(path)
    try:
        if not with_recordings:
            if apply:
                return store.delete_hypotheses(recording_id, unit_id, before)
            return store.hypotheses_summary(recording_id, unit_id, before)
        if apply:
            database, rows = store.delete_hypotheses(recording_id, unit_id, before, with_recordings=True)
        else:
            database = store.hypotheses_summary(recording_id, unit_id, before)
            rows = store.hypotheses_recordings(recording_id, unit_id, before)
        listed = sorted((recording_entry(row["recording_id"], row, row["runs"], row["bytes"]) for row in rows),
                        key=lambda e: (e["date"] or "", e["name"], e["recording_id"]))     # как читает человек
        if recording_id is not None and not listed:
            label = store.recording_label(recording_id)
            if label is not None:
                listed = [recording_entry(recording_id, label)]
        return database, listed
    finally:
        store.close()


def _check_recording_id(recording_id):
    import uuid
    try:
        ok = str(uuid.UUID(str(recording_id))) == recording_id
    except ValueError:
        ok = False
    if not ok:
        raise ValueError("recording_id — номер записи базы (UUID)")


def cleanup(unit_id=None, older_than_days=None, apply=False, data_dir=None, now=None, recording_id=None,
            store_path=None):
    """Удаление гипотез моделей: всех, одной записи (recording_id — номер записи базы), одного звука (unit_id —
    12 знаков) или старше older_than_days дней. Строки базы и прежние файлы прогонов (у файла — по ids.recording_id
    внутри и времени изменения); кэши _artifacts.json и _testbed.json не трогаются. apply=False — только отчёт:
    те же числа, что удалит apply=True. data_dir — папка гипотез (по умолчанию <папка данных>\\hypotheses),
    store_path — база записей (по умолчанию paths.STATE_DB). -> {"database": {"runs", "recordings", "bytes"},
    "recordings": [recording_entry…], "files", "bytes", "units", "applied", ...}: files/bytes/units — прежние
    файлы; "recordings" — записи, чьи строки базы попали в отбор (при отборе по записи — она сама)."""
    if unit_id is not None and not UNIT_RE.match(str(unit_id)):
        raise ValueError("unit_id — 12 знаков 0-9a-f")
    if recording_id is not None:
        _check_recording_id(recording_id)
    base = Path(data_dir) if data_dir is not None else root()
    cutoff = _cutoff(older_than_days, now)
    database, listed = _database_part(recording_id, unit_id, cutoff, apply, store_path, with_recordings=True)
    n = size = 0
    units = set()
    for p in _run_files(base, unit_id):
        try:
            st = p.stat()
        except OSError:
            continue
        if cutoff is not None and st.st_mtime > cutoff:
            continue
        if recording_id is not None and _file_recording_id(p) != recording_id:
            continue
        n += 1
        size += st.st_size
        units.add(p.parent.name)
        if apply:
            p.unlink()
            try:
                p.parent.rmdir()          # пустая папка звука убирается; непустая остаётся
            except OSError:
                pass
    return {"unit_id": unit_id, "recording_id": recording_id, "older_than_days": older_than_days,
            "database": database, "recordings": listed, "files": n, "bytes": size, "units": len(units),
            "applied": bool(apply)}


def summary(cfg=None, data_dir=None, store_path=None):
    """Объём гипотез для показа (get_settings, self_check): числа те же, что у cleanup() без отбора.
    -> {"enabled", "database": {"runs", "recordings", "bytes", "share_of_database"}, "old_files": {...}}.
    Ничего не меняет; база недоступна — "database": {"error": вид ошибки}."""
    out = {"enabled": enabled(cfg)}
    try:
        report = cleanup(data_dir=data_dir, store_path=store_path)
    except Exception as ex:          # показ объёма не должен ронять настройки и самопроверку
        return {**out, "database": {"error": type(ex).__name__}}
    database = dict(report["database"])
    try:
        total = _store_path(store_path).stat().st_size
    except OSError:
        total = 0
    database["share_of_database"] = round(database["bytes"] / total, 4) if total else None
    return {**out, "database": database,
            "old_files": {"files": report["files"], "bytes": report["bytes"], "units": report["units"]}}
