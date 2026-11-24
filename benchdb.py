"""База прогонов теста скорости: файл SQLite и загрузка файлов прогонов (этап 2, docs/bench-database-spec.md).

База — производный указатель: истина лежит в файлах <папка данных>\\bench\\runs\\*.json (их пишет bench.py
через runinfo.write_run). База собирается из них и при любой беде собирается заново в новый файл.

Правила:
  * только стандартная библиотека; остальная программа не импортируется (кроме runinfo — очистка путей);
  * чтение отделено от записи: plan_load, status, compare открывают базу только на чтение и не создают её;
    пишет одна функция — apply_load (и load_historical с apply=True);
  * загрузка повторяема: файл с уже учтённой контрольной суммой пропускается без записи;
  * база только дописывается: на таблицах фактов триггеры запрещают UPDATE и DELETE; код ничего не удаляет;
  * файлы прогонов не открываются на запись; units.json и testbed.json не читаются (в них пути и соль);
  * в базе нет путей, имён файлов записей и текстов: тексты вариантов остаются в <run_id>.texts.json,
    в базе о них только имя, контрольная сумма и размер (raw_output);
  * числа разных методик (bench-1, bench-2, historical) не сравниваются.
"""
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

import runinfo

BENCH_DB_SCHEMA = 1                    # номер схемы базы; не то же, что runinfo.SCHEMA_VERSION (формат файла)
KEY_VERSION = 1                        # версия правила ключа сопоставимости (раздел 7 задания)
LOADER_VERSION = "benchdb-2"     # отклонённые прежним загрузчиком файлы проверяются заново
DB_NAME = "bench_runs.sqlite3"
RUN_FORMATS = (1,)                     # какие значения schema_version файла прогона загрузчик понимает
HISTORICAL_FORMAT = "bench-historical-1"
HISTORICAL_METHOD = "historical"

HELP_APPEND_ONLY = ("База только дописывается: лишний или ошибочный прогон убирается удалением файла базы "
                    "и повторной загрузкой из runs/*.json; код ничего не удаляет.")

# Оси сравнимости измерения скорости (раздел 7). Методика — отдельно: её отличие вывода не допускает.
AXES = ("unit", "model", "device", "config", "testbed", "environment", "power")
AXIS_RU = {"unit": "тестовая единица", "model": "модель", "device": "устройство и среда исполнения",
           "config": "конфигурация", "testbed": "стенд", "environment": "окружение", "power": "питание"}
# Пакеты, от которых зависит скорость; остальные (huggingface-hub и др.) окружение сравнимости не меняют.
COMPUTE_PACKAGES = ("onnxruntime", "onnxruntime-openvino", "onnxruntime-directml", "openvino", "torch",
                    "torchaudio", "ctranslate2", "faster-whisper", "onnx-asr", "gigaam", "numpy", "onnx")
REQUESTED_FIELDS = ("asr_backend", "quant", "asr_quant", "asr_quant_cpu", "pad_to_seconds", "asr_batch",
                    "asr_device", "allow_npu", "asr_third_quant", "asr_third_quant_cpu")
NUM_FIELDS = ("load_sec", "first_chunk_sec", "asr_sec", "rtfx", "wall_sec", "total_10min", "compile_sec",
              "peak_memory_mb")
INT_FIELDS = ("chunks", "empty_chunks", "compile_cache_new_files")
BOOL_FIELDS = ("model_downloaded", "device_mismatch")
_RUN_NAME = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")      # имя, которое даёт runinfo.new_run_id
_ID = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_.-]{0,79}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS testbed (
    testbed_pk TEXT PRIMARY KEY, testbed_id TEXT, label TEXT, hardware_json TEXT, is_historical INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS env_snapshot (
    env_pk TEXT PRIMARY KEY, os_build TEXT, python_version TEXT, packages_sha256 TEXT, key_packages_json TEXT,
    program_version TEXT, code_revision TEXT, code_dirty INTEGER, environment_json TEXT);
CREATE TABLE IF NOT EXISTS model_artifact (
    model_pk TEXT PRIMARY KEY, kind TEXT, name TEXT, ref_main TEXT, snapshots_json TEXT, files INTEGER,
    size_mb REAL, size_bytes INTEGER, extra_json TEXT);
CREATE TABLE IF NOT EXISTS test_unit (
    unit_pk TEXT PRIMARY KEY, unit_id TEXT, audio_sha256 TEXT, offset_sec REAL, duration_sec REAL,
    source_format TEXT, source_duration_sec REAL, size_bytes INTEGER, pcm_format TEXT);
CREATE TABLE IF NOT EXISTS reference (
    reference_id INTEGER PRIMARY KEY, unit_pk TEXT, kind TEXT, sha256 TEXT, words INTEGER, supersedes INTEGER,
    created_utc TEXT);
CREATE TABLE IF NOT EXISTS exec_config (
    config_pk TEXT PRIMARY KEY, variant_id TEXT, family TEXT, asr_backend TEXT, asr_model TEXT, device TEXT,
    chunk_max REAL, third_max_repeat INTEGER, config_json TEXT);
CREATE TABLE IF NOT EXISTS run (
    run_id TEXT PRIMARY KEY, kind TEXT, started_utc TEXT, finished_utc TEXT, duration_sec REAL, status TEXT,
    error_type TEXT, method_version TEXT, run_schema_version INTEGER, db_schema_version INTEGER,
    loader_version TEXT, job_id TEXT, testbed_pk TEXT, env_pk TEXT, unit_pk TEXT, params_json TEXT,
    conditions_json TEXT, preflight_json TEXT, choice_json TEXT, valid TEXT, valid_reason TEXT,
    is_historical INTEGER DEFAULT 0, completeness TEXT, source_note TEXT, source_file_sha256 TEXT,
    texts_file TEXT);
CREATE TABLE IF NOT EXISTS run_model (run_id TEXT, model_pk TEXT, PRIMARY KEY (run_id, model_pk));
CREATE TABLE IF NOT EXISTS measurement (
    measurement_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, variant_id TEXT NOT NULL,
    unit_pk TEXT NOT NULL DEFAULT '', repeat_no INTEGER NOT NULL DEFAULT 0, mode TEXT NOT NULL DEFAULT 'steady',
    config_pk TEXT, family TEXT, device_requested TEXT, device_actual TEXT, device_mismatch INTEGER,
    backend_name TEXT, providers_actual_json TEXT, fallbacks_json TEXT, compile_sec REAL,
    compile_cache_new_files INTEGER, model_downloaded INTEGER, load_sec REAL, first_chunk_sec REAL,
    asr_sec REAL, rtfx REAL, wall_sec REAL, total_10min REAL, chunks INTEGER, empty_chunks INTEGER,
    chunk_words_json TEXT, peak_memory_mb REAL, unfit TEXT, status TEXT, status_reason TEXT,
    not_measured_json TEXT, speed_key TEXT, axes_json TEXT, extra_json TEXT,
    UNIQUE (run_id, variant_id, unit_pk, repeat_no, mode));
CREATE TABLE IF NOT EXISTS raw_output (
    run_id TEXT, file_name TEXT, sha256 TEXT, size_bytes INTEGER, PRIMARY KEY (run_id, file_name));
CREATE TABLE IF NOT EXISTS evaluation (
    evaluation_id INTEGER PRIMARY KEY, measurement_id INTEGER, reference_id INTEGER, metric TEXT, value REAL,
    counts_json TEXT, normalizer TEXT, normalizer_version TEXT, tool TEXT, tool_version TEXT, params_json TEXT,
    accuracy_key TEXT, computed_utc TEXT, supersedes INTEGER);
CREATE TABLE IF NOT EXISTS decision (
    decision_id INTEGER PRIMARY KEY, run_id TEXT, setting TEXT, old_value_json TEXT, new_value_json TEXT,
    applied INTEGER, chooser TEXT, UNIQUE (run_id, setting));
CREATE TABLE IF NOT EXISTS external_reference (
    ref_id INTEGER PRIMARY KEY, source TEXT, published TEXT, summary_json TEXT);
CREATE TABLE IF NOT EXISTS import_log (
    log_id INTEGER PRIMARY KEY, file_name TEXT, sha256 TEXT NOT NULL, size_bytes INTEGER, run_id TEXT,
    loaded_utc TEXT, loader_version TEXT, outcome TEXT, reason TEXT, UNIQUE (sha256, loader_version));
CREATE INDEX IF NOT EXISTS measurement_key ON measurement (speed_key);
CREATE INDEX IF NOT EXISTS measurement_run ON measurement (run_id, variant_id);
"""
FACT_TABLES = ("testbed", "env_snapshot", "model_artifact", "test_unit", "reference", "exec_config", "run",
               "run_model", "measurement", "raw_output", "evaluation", "decision", "external_reference",
               "import_log")


class BenchDbError(Exception):
    """База не может быть открыта или записана (новая схема, не база SQLite, нет файла для чтения)."""


class RunRejected(Exception):
    """Файл не принят: текст — причина без путей."""


def db_path(bench_dir):
    return Path(bench_dir) / DB_NAME


def _exists(bench_dir):
    """Файл базы есть и не пуст: пустой файл — база, которую ещё не успели создать (или бросили на полпути)."""
    p = db_path(bench_dir)
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def plural(n, forms):
    """Склонение по числу: forms = («прогон», «прогона», «прогонов»)."""
    n = abs(int(n))
    if 11 <= n % 100 <= 14:
        return forms[2]
    return forms[0] if n % 10 == 1 else forms[1] if 2 <= n % 10 <= 4 else forms[2]


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def digest(x, n=32):
    return hashlib.sha256(canon(x).encode("utf-8")).hexdigest()[:n]


def _js(x):
    return None if x is None else json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


# ── соединения ───────────────────────────────────────────────────────────────────────────────────

def open_read(bench_dir):
    """Только чтение: запись через это соединение невозможна. Нет базы — BenchDbError."""
    if not _exists(bench_dir):
        raise BenchDbError("базы прогонов нет")
    con = None
    try:
        con = sqlite3.connect(db_path(bench_dir).resolve().as_uri() + "?mode=ro", uri=True)
        con.execute("SELECT 1 FROM meta LIMIT 1")
    except sqlite3.DatabaseError as ex:
        if con is not None:
            con.close()
        raise BenchDbError(f"файл не читается как база прогонов: {type(ex).__name__}") from None
    return con


def _check_version(con):
    row = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if not row or str(row[0]) != str(BENCH_DB_SCHEMA):
        raise BenchDbError(f"схема базы {row[0] if row else '?'} не совпадает со схемой программы "
                           f"{BENCH_DB_SCHEMA}: соберите базу заново в новый файл из runs/*.json")


def _has_meta(con):
    return con.execute("SELECT 1 FROM sqlite_master WHERE name='meta'").fetchone() is not None


def _rollback(con):
    try:
        con.execute("ROLLBACK")
    except sqlite3.Error:
        pass


def _create_schema(con):
    for stmt in SCHEMA.split(";"):
        if stmt.strip():
            con.execute(stmt)
    for t in FACT_TABLES:
        for op in ("UPDATE", "DELETE"):
            con.execute(f"CREATE TRIGGER no_{op.lower()}_{t} BEFORE {op} ON {t} "
                        f"BEGIN SELECT RAISE(ABORT, 'база прогонов только дописывается'); END")
    for k, v in (("schema_version", str(BENCH_DB_SCHEMA)), ("key_version", str(KEY_VERSION)),
                 ("created_utc", runinfo.utc_now()), ("loader_version", LOADER_VERSION)):
        con.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (k, v))


def _open_write(bench_dir):
    """Соединение на запись. Схема создаётся одной транзакцией под блокировкой: одновременный первый запуск
    нескольких процессов не даёт полусозданной базы. При любом отказе соединение закрывается."""
    con = None
    try:
        con = sqlite3.connect(db_path(bench_dir), isolation_level=None, timeout=10)
        con.execute("PRAGMA busy_timeout=5000")
        if not _has_meta(con):
            con.execute("BEGIN IMMEDIATE")
            try:
                if not _has_meta(con):           # пока ждали блокировку, другой процесс мог создать схему
                    if con.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]:
                        raise BenchDbError("в файле базы чужие таблицы: он не создан этой программой")
                    _create_schema(con)
                con.execute("COMMIT")
            except BaseException:
                _rollback(con)
                raise
        _check_version(con)
    except BaseException as ex:
        if con is not None:
            con.close()
        if isinstance(ex, sqlite3.DatabaseError):
            raise BenchDbError(f"файл не открывается как база прогонов: {type(ex).__name__}") from None
        raise
    return con


# ── разбор файла прогона (без базы) ──────────────────────────────────────────────────────────────

def _num(v, key, nm):
    if isinstance(v, dict) and v.get("status") == "not_measured":
        nm[key] = str(v.get("reason") or "")[:120]
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, int) and abs(v) >= 2 ** 63:
        raise OverflowError("число не помещается в целое SQLite")
    return v


def _axes(rec, variant, r, unit, tb_pk):
    """Оси сравнимости измерения: {ось: {"value": ..., "known": bool}}. Неизвестное не равно известному."""
    hw = _get(rec, "testbed", "hardware") or {}
    pk = _get(rec, "environment", "packages", "key") or {}
    cond = _get(rec, "conditions", "start") or {}
    name = variant.get("asr_model") or variant.get("asr_third_model") or f"default:{variant.get('family')}"
    revision = None
    for m in _get(rec, "models", "hf") or []:
        repo = str(m.get("repo") or "").lower()
        if repo and (repo == str(name).lower() or repo.endswith("/" + str(name).lower())):
            revision = m.get("ref_main")
    drivers = sorted([str(d.get("name")), str(d.get("driver_version"))]
                     for d in (hw.get("devices") or []) if isinstance(d, dict))
    backend, dev = r.get("backend_name"), r.get("device_actual")
    return {
        "unit": {"value": {"pcm": _get(unit, "pcm"), "offset": _get(unit, "offset"),
                           "duration": _get(unit, "duration")}, "known": bool(_get(unit, "pcm"))},
        # revision None — ревизия модели неизвестна (локальные файлы GigaAM): равенство значит «то же имя»
        "model": {"value": {"name": name, "revision": revision}, "known": True},
        "device": {"value": {"backend": backend, "providers": r.get("providers_actual"), "device": dev},
                   "known": bool(backend and dev)},
        "config": {"value": {"family": variant.get("family") or r.get("family"),
                             "requested": {k: variant.get(k) for k in REQUESTED_FIELDS if k in variant},
                             "chunk_max": _get(rec, "params", "chunk_max"),
                             "third_max_repeat": _get(rec, "params", "third_max_repeat")}, "known": True},
        "testbed": {"value": tb_pk, "known": bool(tb_pk) and hw.get("status") != "not_measured"},
        "environment": {"value": {"packages": {k: pk[k] for k in COMPUTE_PACKAGES if k in pk},
                                  "os": {"build": _get(rec, "environment", "os", "build"),
                                         "ubr": _get(rec, "environment", "os", "ubr")},
                                  "drivers": drivers}, "known": bool(pk)},
        "power": {"value": {"source": cond.get("power_source"), "scheme": cond.get("power")},
                  "known": bool(cond.get("power_source"))},
    }


def _speed_key(method, axes):
    return digest({"key_version": KEY_VERSION, "method": method,
                   "axes": {a: (axes[a]["value"] if axes[a]["known"] else None) for a in AXES},
                   "known": {a: axes[a]["known"] for a in AXES}})


def _valid(rec):
    status = rec.get("status")
    if status != "done":
        return "no", f"status={status}"
    pre = rec.get("preflight")
    if not isinstance(pre, dict):
        return "with_reservations", "no_preflight"
    if pre.get("ok"):
        return "yes", None
    return "with_reservations", ",".join(str(x) for x in pre.get("reasons") or []) or "preflight_not_ok"


def _no_constant(name):
    raise ValueError("NaN или Infinity в файле")


def _check_structure(rec):
    """Типы полей, от которых зависит разбор. Неверная структура — RunRejected с названием поля."""
    def kind(path, v, types):
        if v is None:
            return
        if isinstance(v, bool) or not isinstance(v, types):
            raise RunRejected(f"неверный тип поля {path}")

    for key in ("testbed", "environment", "audio", "params", "models", "conditions", "preflight", "choice", "saved"):
        kind(key, rec.get(key), dict)
    for key in ("started_utc", "finished_utc", "status", "error_type", "job_id", "method_version", "texts_file"):
        kind(key, rec.get(key), str)
    kind("duration_sec", rec.get("duration_sec"), (int, float))
    kind("variants", rec.get("variants"), list)
    kind("results", rec.get("results"), list)
    hw = _get(rec, "testbed", "hardware")
    kind("testbed.hardware", hw, dict)
    if isinstance(hw, dict):
        kind("testbed.hardware.devices", hw.get("devices"), list)
        for d in hw.get("devices") or []:
            if not isinstance(d, dict):
                raise RunRejected("неверный тип поля testbed.hardware.devices[]")
    env = rec.get("environment") or {}
    for key in ("os", "python", "packages", "code"):
        kind(f"environment.{key}", env.get(key), dict)
    kind("environment.packages.key", _get(env, "packages", "key"), dict)
    cond = rec.get("conditions") or {}
    for key in ("start", "end"):
        kind(f"conditions.{key}", cond.get(key), dict)
    kind("conditions.start.power", _get(cond, "start", "power"), dict)
    kind("preflight.reasons", _get(rec, "preflight", "reasons"), list)
    mods = rec.get("models") or {}
    for key in ("hf", "local"):
        kind(f"models.{key}", mods.get(key), list)
    kind("audio.excerpt", _get(rec, "audio", "excerpt"), dict)
    for name in ("variants", "results"):
        for item in rec.get(name) or []:
            if not isinstance(item, dict):
                raise RunRejected(f"неверный тип элемента {name}")
    seen = set()
    for r in rec.get("results") or []:
        rid = r.get("id")
        if rid is None:
            continue
        if not isinstance(rid, str) or not rid:
            raise RunRejected("неверный id варианта в results")
        if rid in seen:
            raise RunRejected("повтор id варианта в results")
        seen.add(rid)


def parse_run(data, file_name=None):
    """Раскладывает файл прогона по таблицам. Чистая функция: базу не открывает. Плохой файл — RunRejected."""
    try:
        rec = json.loads(data.decode("utf-8-sig"), parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as ex:
        raise RunRejected("в файле NaN или Infinity" if "NaN" in str(ex) else "файл не JSON") from None
    if not isinstance(rec, dict):
        raise RunRejected("в файле не объект JSON")
    if rec.get("kind") != "bench":
        raise RunRejected("вид файла не bench")
    if rec.get("schema_version") not in RUN_FORMATS:
        raise RunRejected(f"неизвестная версия формата файла прогона: {rec.get('schema_version')!r}")
    run_id = rec.get("run_id")
    if not isinstance(run_id, str) or not _ID.match(run_id):
        raise RunRejected("нет допустимого run_id")
    if file_name is not None and file_name != f"{run_id}.json":
        raise RunRejected("имя файла не совпадает с run_id")
    _check_structure(rec)
    try:
        return _build(runinfo.scrub(rec), run_id)
    except (TypeError, ValueError, AttributeError, KeyError, OverflowError, RecursionError) as ex:
        raise RunRejected(f"структура файла неверна ({type(ex).__name__})") from None


def _build(rec, run_id):
    """Раскладка проверенного файла по таблицам."""
    method = str(rec.get("method_version") or "unknown")
    out = {"run_id": run_id, "models": [], "configs": [], "measurements": [], "decisions": []}

    tb = rec.get("testbed") if isinstance(rec.get("testbed"), dict) else None
    tb_pk = ""
    if tb:
        tb_pk = digest({"id": tb.get("testbed_id"), "hardware": tb.get("hardware")})
        hw = tb.get("hardware") if isinstance(tb.get("hardware"), dict) else {}
        out["testbed"] = {"testbed_pk": tb_pk, "testbed_id": tb.get("testbed_id"), "label": None,
                          "hardware_json": _js(hw), "is_historical": 0}
    env = rec.get("environment") if isinstance(rec.get("environment"), dict) else None
    env_pk = ""
    if env:
        env_pk = digest(env)
        os_ = env.get("os") or {}
        out["env"] = {"env_pk": env_pk, "os_build": f"{os_.get('build')}.{os_.get('ubr')}" if os_.get("build") else None,
                      "python_version": _get(env, "python", "version"),
                      "packages_sha256": _get(env, "packages", "sha256"),
                      "key_packages_json": _js(_get(env, "packages", "key")),
                      "program_version": env.get("program_version"),
                      "code_revision": _get(env, "code", "git_revision") or _get(env, "code", "files_sha256"),
                      "code_dirty": (None if _get(env, "code", "git_dirty") is None
                                     else int(bool(_get(env, "code", "git_dirty")))),
                      "environment_json": _js(env)}
    mods = rec.get("models") if isinstance(rec.get("models"), dict) else {}
    for m in mods.get("hf") or []:
        if isinstance(m, dict):
            pk = digest(["hf", m.get("repo"), m.get("ref_main"), m.get("snapshots"), m.get("files"), m.get("size_mb")])
            out["models"].append({"model_pk": pk, "kind": "hf", "name": m.get("repo"), "ref_main": m.get("ref_main"),
                                  "snapshots_json": _js(m.get("snapshots")), "files": m.get("files"),
                                  "size_mb": m.get("size_mb"), "size_bytes": None, "extra_json": None})
    for m in mods.get("local") or []:
        if isinstance(m, dict):
            pk = digest(["local", m.get("file"), m.get("size_bytes")])
            out["models"].append({"model_pk": pk, "kind": "local", "name": m.get("file"), "ref_main": None,
                                  "snapshots_json": None, "files": None, "size_mb": None,
                                  "size_bytes": m.get("size_bytes"),
                                  "extra_json": _js({"modified_utc": m.get("modified_utc")})})
    audio = rec.get("audio") if isinstance(rec.get("audio"), dict) else None
    unit_pk, unit = "", {}
    if audio:
        ex = audio.get("excerpt") if isinstance(audio.get("excerpt"), dict) else {}
        unit_pk = ex.get("pcm_sha256") or audio.get("unit_id") or ""
        unit = {"pcm": unit_pk or None, "offset": ex.get("offset_sec"), "duration": ex.get("duration_sec")}
        if unit_pk:
            out["unit"] = {"unit_pk": unit_pk, "unit_id": audio.get("unit_id"), "audio_sha256": audio.get("audio_sha256"),
                           "offset_sec": ex.get("offset_sec"), "duration_sec": ex.get("duration_sec"),
                           "source_format": audio.get("source_format"),
                           "source_duration_sec": audio.get("source_duration_sec"),
                           "size_bytes": audio.get("size_bytes"), "pcm_format": ex.get("pcm_format")}
    params = rec.get("params") if isinstance(rec.get("params"), dict) else {}
    variants = {v.get("id"): v for v in rec.get("variants") or [] if isinstance(v, dict)}
    for r in rec.get("results") or []:
        if not isinstance(r, dict) or not r.get("id"):
            continue
        v = variants.get(r["id"], {})
        cfg = {"variant": v, "chunk_max": params.get("chunk_max"), "seconds": params.get("seconds"),
               "third_max_repeat": params.get("third_max_repeat")}
        cfg_pk = digest(cfg)
        out["configs"].append({"config_pk": cfg_pk, "variant_id": r["id"], "family": v.get("family") or r.get("family"),
                               "asr_backend": v.get("asr_backend"), "asr_model": v.get("asr_model"),
                               "device": v.get("device") or r.get("device"), "chunk_max": params.get("chunk_max"),
                               "third_max_repeat": params.get("third_max_repeat"), "config_json": _js(cfg)})
        nm = {}
        row = {"run_id": run_id, "variant_id": r["id"], "unit_pk": unit_pk, "repeat_no": 0, "mode": "steady",
               "config_pk": cfg_pk, "family": r.get("family") or v.get("family"),
               "device_requested": r.get("device") or v.get("device"), "device_actual": r.get("device_actual"),
               "backend_name": r.get("backend_name"), "providers_actual_json": _js(r.get("providers_actual")),
               "fallbacks_json": _js(r.get("fallbacks")), "chunk_words_json": _js(r.get("chunk_words")),
               "unfit": r.get("unfit") if isinstance(r.get("unfit"), str) else _js(r.get("unfit"))}
        for k in NUM_FIELDS:
            row[k] = _num(r.get(k), k, nm)
        for k in INT_FIELDS:
            n = _num(r.get(k), k, nm)
            row[k] = None if n is None else int(n)
        for k in BOOL_FIELDS:
            row[k] = None if r.get(k) is None else int(bool(r.get(k)))
        if "asr_sec" in r and "error" not in r:
            row["status"], row["status_reason"] = "done", None
        elif "error" in r:
            err = r["error"]
            reason = " ".join(str(x) for x in err) if isinstance(err, list) else str(err)
            row["status"], row["status_reason"] = "error", reason[:200]
        else:
            row["status"], row["status_reason"] = "skipped", "нет результата"
        if r.get("cert_error"):
            row["status_reason"] = ((row["status_reason"] or "") + " [сертификат]").strip()
        used = set(NUM_FIELDS) | set(INT_FIELDS) | set(BOOL_FIELDS) | {
            "id", "family", "device", "text", "error", "cert_error", "backend_name", "providers_actual", "fallbacks",
            "chunk_words", "unfit", "device_actual", "label"}
        row["not_measured_json"] = _js(nm or None)
        row["extra_json"] = _js({k: x for k, x in r.items() if k not in used} or None)
        axes = _axes(rec, v, r, unit, tb_pk)
        row["axes_json"] = _js(axes)
        row["speed_key"] = _speed_key(method, axes)
        out["measurements"].append(row)
    saved = rec.get("saved") if isinstance(rec.get("saved"), dict) else {}
    for setting, value in sorted(saved.items()):
        out["decisions"].append({"run_id": run_id, "setting": str(setting), "old_value_json": None,
                                 "new_value_json": _js(value), "applied": int(bool(rec.get("config_written"))),
                                 "chooser": "bench.py"})
    valid, reason = _valid(rec)
    out["run"] = {"run_id": run_id, "kind": "bench", "started_utc": rec.get("started_utc"),
                  "finished_utc": rec.get("finished_utc"), "duration_sec": rec.get("duration_sec"),
                  "status": rec.get("status"), "error_type": rec.get("error_type"), "method_version": method,
                  "run_schema_version": rec.get("schema_version"), "db_schema_version": BENCH_DB_SCHEMA,
                  "loader_version": LOADER_VERSION, "job_id": rec.get("job_id"), "testbed_pk": tb_pk,
                  "env_pk": env_pk, "unit_pk": unit_pk, "params_json": _js(params),
                  "conditions_json": _js(rec.get("conditions")), "preflight_json": _js(rec.get("preflight")),
                  "choice_json": _js(rec.get("choice")), "valid": valid, "valid_reason": reason,
                  "is_historical": 0, "completeness": None, "source_note": None,
                  "texts_file": rec.get("texts_file")}
    out["texts_file"] = rec.get("texts_file")
    return out


# ── загрузка ─────────────────────────────────────────────────────────────────────────────────────

def _insert(con, table, row, ignore=False):
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    con.execute(f"INSERT {'OR IGNORE ' if ignore else ''}INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))


def _log(con, name, sha, size, run_id, outcome, reason):
    _insert(con, "import_log", {"file_name": name, "sha256": sha, "size_bytes": size, "run_id": run_id,
                                "loaded_utc": runinfo.utc_now(), "loader_version": LOADER_VERSION,
                                "outcome": outcome, "reason": (reason or "")[:300] or None})


def _known(con, sha):
    """Файл уже учтён: загружен или в конфликте (любым загрузчиком), отклонён — только этим же загрузчиком:
    после обновления загрузчика отклонённый файл проверяется заново, журнал при этом только дописывается."""
    if con is None:
        return None
    row = con.execute("SELECT outcome, run_id FROM import_log WHERE sha256=? AND (outcome != 'rejected' "
                      "OR loader_version=?) ORDER BY log_id LIMIT 1", (sha, LOADER_VERSION)).fetchone()
    return {"outcome": row[0], "run_id": row[1]} if row else None


def _run_exists(con, run_id):
    return con is not None and con.execute("SELECT 1 FROM run WHERE run_id=?", (run_id,)).fetchone() is not None


def _classify(con, path):
    """-> (запись об одном файле, разобранный прогон или None). Базу только читает."""
    p = Path(path)
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    item = {"file": p.name, "sha256": sha, "size_bytes": len(data), "run_id": None, "action": None, "reason": None}
    earlier = _known(con, sha)
    if earlier:
        item.update(action="known", reason=f"уже учтён ({earlier['outcome']})", run_id=earlier["run_id"],
                    known_outcome=earlier["outcome"])
        return item, None
    try:
        parsed = parse_run(data, p.name)
    except RunRejected as ex:
        stem = p.name[:-len(".json")]
        item.update(action="rejected", reason=str(ex), run_id=stem if _RUN_NAME.match(stem) else None)
        return item, None
    item["run_id"] = parsed["run_id"]
    if _run_exists(con, parsed["run_id"]):
        item.update(action="conflict", reason="run_id уже в базе, содержимое файла другое; данные не изменены")
        return item, None
    item["action"] = "new"
    return item, parsed


def _run_files(bench_dir):
    d = Path(bench_dir) / "runs"
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.json") if p.is_file() and not p.name.endswith(".texts.json"))


def _counts(items):
    out = {}
    for it in items:
        out[it["action"]] = out.get(it["action"], 0) + 1
    return out


def _ro_or_none(bench_dir):
    try:
        return open_read(bench_dir)
    except BenchDbError:
        if _exists(bench_dir):
            raise
        return None


def plan_load(bench_dir):
    """Что будет сделано с файлами прогонов. Базу только читает, не создаёт."""
    con = _ro_or_none(bench_dir)
    try:
        if con is not None:
            _check_version(con)
        items = []
        for p in _run_files(bench_dir):
            try:
                item, _ = _classify(con, p)
            except OSError:
                item = {"file": p.name, "sha256": None, "run_id": None, "action": "rejected",
                        "reason": "файл не читается"}
            items.append(item)
    finally:
        if con is not None:
            con.close()
    return {"files": items, "counts": _counts(items)}


def _texts_ref(bench_dir, parsed):
    name = parsed.get("texts_file")
    if name != f"{parsed['run_id']}.texts.json":
        return None
    f = Path(bench_dir) / "runs" / name
    try:
        data = f.read_bytes()
    except OSError:
        return None
    return {"run_id": parsed["run_id"], "file_name": name, "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def _store(con, bench_dir, parsed):
    for key, table in (("testbed", "testbed"), ("env", "env_snapshot"), ("unit", "test_unit")):
        if key in parsed:
            _insert(con, table, parsed[key], ignore=True)
    for m in parsed["models"]:
        _insert(con, "model_artifact", m, ignore=True)
        _insert(con, "run_model", {"run_id": parsed["run_id"], "model_pk": m["model_pk"]}, ignore=True)
    for c in parsed["configs"]:
        _insert(con, "exec_config", c, ignore=True)
    _insert(con, "run", parsed["run"])
    for m in parsed["measurements"]:
        _insert(con, "measurement", m)
    for d in parsed["decisions"]:
        _insert(con, "decision", d)
    ref = _texts_ref(bench_dir, parsed)
    if ref:
        _insert(con, "raw_output", ref)


def _unreadable(p):
    return {"file": p.name, "sha256": None, "size_bytes": None, "run_id": None, "action": "rejected",
            "reason": "файл не читается"}


def _late_texts(con, bench_dir, item):
    """Файл текстов появился позже файла прогона: ссылку на него дописываем уже загруженному прогону."""
    run_id = item.get("run_id")
    if item.get("known_outcome") != "loaded" or not run_id:
        return
    row = con.execute("SELECT texts_file FROM run WHERE run_id=?", (run_id,)).fetchone()
    if not row or row[0] != f"{run_id}.texts.json":
        return
    if con.execute("SELECT 1 FROM raw_output WHERE run_id=?", (run_id,)).fetchone():
        return
    ref = _texts_ref(bench_dir, {"run_id": run_id, "texts_file": row[0]})
    if not ref:
        return
    con.execute("BEGIN IMMEDIATE")
    try:
        if not con.execute("SELECT 1 FROM raw_output WHERE run_id=?", (run_id,)).fetchone():
            _insert(con, "raw_output", ref)
        con.execute("COMMIT")
    except BaseException as ex:
        _rollback(con)
        if isinstance(ex, sqlite3.OperationalError):
            raise BenchDbError(f"запись не удалась: {type(ex).__name__}") from None
        raise


def _apply_one(con, bench_dir, p, item):
    """Один файл — одна транзакция: сбой разбора или вставки откатывает только его, файл записывается как
    отклонённый, загрузка идёт дальше. Занятая база (OperationalError) — не вина файла: загрузка останавливается."""
    con.execute("BEGIN IMMEDIATE")
    try:
        item, parsed = _classify(con, p)           # заново внутри транзакции: пока ждали, мог загрузить другой
        if item["action"] == "new":
            parsed["run"]["source_file_sha256"] = item["sha256"]
            _store(con, bench_dir, parsed)
            item["action"] = "loaded"
            _log(con, item["file"], item["sha256"], item["size_bytes"], item["run_id"], "loaded", None)
        elif item["action"] in ("rejected", "conflict"):
            _log(con, item["file"], item["sha256"], item["size_bytes"], item["run_id"], item["action"], item["reason"])
        con.execute("COMMIT")
        return item
    except OSError:
        _rollback(con)
        return _unreadable(p)
    except sqlite3.OperationalError as ex:
        _rollback(con)
        raise BenchDbError(f"запись не удалась: {type(ex).__name__}") from None
    except (ValueError, TypeError, OverflowError, sqlite3.Error) as ex:
        _rollback(con)
        item.update(action="rejected", reason=f"сбой разбора или записи ({type(ex).__name__})")
        con.execute("BEGIN IMMEDIATE")
        try:
            _log(con, item["file"], item["sha256"], item["size_bytes"], item["run_id"], "rejected", item["reason"])
            con.execute("COMMIT")
        except BaseException:
            _rollback(con)
            raise
        return item
    except BaseException:
        _rollback(con)
        raise


def apply_load(bench_dir):
    """Единственное место записи для файлов прогонов. Файл — своя транзакция; учтённые пропускаются."""
    con = _open_write(bench_dir)
    items = []
    try:
        for p in _run_files(bench_dir):
            try:
                item, _ = _classify(con, p)
            except OSError:
                items.append(_unreadable(p))
                continue
            if item["action"] == "known":              # учтённый файл: транзакции нет, пишется только запоздавшая ссылка
                _late_texts(con, bench_dir, item)
                items.append(item)
                continue
            items.append(_apply_one(con, bench_dir, p, item))
    finally:
        con.close()
    return {"files": items, "counts": _counts(items)}


# ── прошлые замеры (раздел 11) ───────────────────────────────────────────────────────────────────

def _historical_rows(entry):
    entry = runinfo.scrub(entry)
    eid = entry.get("entry_id")
    if not isinstance(eid, str) or not _ID.match(eid):
        raise RunRejected("у записи нет допустимого entry_id")
    run_id = f"hist-{eid}"
    entry_sha = digest(entry)
    machine = str(entry.get("machine") or "")[:60] or None
    tb_pk = digest({"historical_machine": machine})
    nm = {}
    row = {"run_id": run_id, "variant_id": str(entry.get("variant") or "unknown"), "unit_pk": "",
           "repeat_no": 0, "mode": "steady", "config_pk": None, "family": entry.get("family"),
           "device_requested": entry.get("device"), "status": "done", "status_reason": None,
           "speed_key": digest({"historical": eid}), "axes_json": _js({}),
           "not_measured_json": None, "extra_json": _js({k: entry[k] for k in ("similarity", "note")
                                                       if k in entry} or None)}
    for k in NUM_FIELDS:
        row[k] = _num(entry.get(k), k, nm)
    run = {"run_id": run_id, "kind": "historical", "started_utc": f"{str(entry.get('date') or '')[:10]}T00:00:00Z"
           if entry.get("date") else None, "status": "done", "method_version": HISTORICAL_METHOD,
           "run_schema_version": None, "db_schema_version": BENCH_DB_SCHEMA, "loader_version": LOADER_VERSION,
           "testbed_pk": tb_pk, "env_pk": "", "unit_pk": "", "valid": "with_reservations",
           "valid_reason": "historical", "is_historical": 1, "params_json": _js({"entry_sha256": entry_sha}),
           "completeness": str(entry.get("completeness") or "")[:8] or None,
           "source_note": str(entry.get("source") or "")[:200] or None}
    testbed = {"testbed_pk": tb_pk, "testbed_id": None, "label": machine, "hardware_json": None, "is_historical": 1}
    return run_id, run, row, testbed, entry_sha


def _hist_state(con, run_id, entry_sha):
    """new — записи нет; same — есть такая же; changed — есть, но другого содержания (не перезаписывается)."""
    if con is None:
        return "new"
    row = con.execute("SELECT params_json FROM run WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        return "new"
    return "same" if (json.loads(row[0] or "{}").get("entry_sha256") == entry_sha) else "changed"


def load_historical(bench_dir, file_path, apply=False):
    """Вносит исторические замеры из файла bench-historical-1. Без apply базу не создаёт и не меняет."""
    data = Path(file_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    name = f"historical-{sha[:12]}"
    item = {"file": name, "sha256": sha, "size_bytes": len(data), "run_id": None, "action": None, "reason": None}
    con = None
    try:
        con = _open_write(bench_dir) if apply else _ro_or_none(bench_dir)
        if con is not None and not apply:
            _check_version(con)
        earlier = _known(con, sha)
        if earlier:
            item.update(action="known", reason=f"уже учтён ({earlier['outcome']})")
            return {"files": [item], "counts": {"known": 1}}
        try:
            doc = json.loads(data.decode("utf-8-sig"))
            if not isinstance(doc, dict) or doc.get("format") != HISTORICAL_FORMAT \
                    or not isinstance(doc.get("entries"), list):
                raise RunRejected(f"формат не {HISTORICAL_FORMAT}")
            built = [_historical_rows(e) for e in doc["entries"] if isinstance(e, dict)]
        except (RunRejected, ValueError, UnicodeDecodeError, TypeError, AttributeError, KeyError,
                OverflowError, RecursionError) as ex:
            item.update(action="rejected", reason=str(ex) if isinstance(ex, RunRejected)
                        else "файл не JSON" if isinstance(ex, (ValueError, UnicodeDecodeError, RecursionError))
                        else f"структура файла неверна ({type(ex).__name__})")
            if apply:
                con.execute("BEGIN IMMEDIATE")
                _log(con, name, sha, len(data), None, "rejected", item["reason"])
                con.execute("COMMIT")
            return {"files": [item], "counts": {"rejected": 1}}
        state = {b[0]: _hist_state(con, b[0], b[4]) for b in built}
        n_new = sum(1 for v in state.values() if v == "new")
        n_changed = sum(1 for v in state.values() if v == "changed")
        item["reason"] = (f"записей: {len(built)}, новых: {n_new}, уже были: {len(built) - n_new - n_changed}, "
                          f"изменённых (прежние не перезаписаны): {n_changed}")
        if not apply:
            item["action"] = "new"
            return {"files": [item], "counts": {"new": 1}}
        con.execute("BEGIN IMMEDIATE")
        try:
            for run_id, run, row, testbed, entry_sha in built:
                if _hist_state(con, run_id, entry_sha) != "new":
                    continue
                _insert(con, "testbed", testbed, ignore=True)
                run["source_file_sha256"] = sha
                _insert(con, "run", run)
                _insert(con, "measurement", row)
            _log(con, name, sha, len(data), None, "loaded", item["reason"])
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        item["action"] = "loaded"
        return {"files": [item], "counts": {"loaded": 1}}
    finally:
        if con is not None:
            con.close()


# ── чтение ───────────────────────────────────────────────────────────────────────────────────────

def status(bench_dir):
    """Сколько прогонов в базе по методикам и сколько файлов ещё не загружено. Только чтение."""
    plan = plan_load(bench_dir)
    out = {"db_exists": _exists(bench_dir), "by_method": {}, "runs": 0,
           "not_loaded": plan["counts"].get("new", 0), "conflicts": 0, "rejected": 0}
    if out["db_exists"]:
        con = open_read(bench_dir)
        try:
            _check_version(con)
            out["by_method"] = dict(con.execute("SELECT method_version, count(*) FROM run GROUP BY 1 ORDER BY 1"))
            out["runs"] = sum(out["by_method"].values())
            out["conflicts"] = con.execute("SELECT count(*) FROM import_log WHERE outcome='conflict'").fetchone()[0]
            out["rejected"] = con.execute("SELECT count(*) FROM import_log WHERE outcome='rejected'").fetchone()[0]
        finally:
            con.close()
    return out


def _measurement(con, ref):
    run_id, _, variant = ref.partition(":")
    return con.execute("SELECT r.method_version, r.is_historical, m.axes_json, m.run_id, m.variant_id "
                       "FROM measurement m JOIN run r USING (run_id) WHERE m.run_id=? AND m.variant_id=? "
                       "ORDER BY m.measurement_id LIMIT 1", (run_id, variant)).fetchone()


def compare(bench_dir, first, second):
    """Сравнивает два измерения скорости («run_id:вариант») по осям раздела 7. Только чтение."""
    con = open_read(bench_dir)
    try:
        _check_version(con)
        a, b = _measurement(con, first), _measurement(con, second)
    finally:
        con.close()
    if a is None or b is None:
        which = first if a is None else second
        return {"verdict": "unknown", "axes": [], "unmeasured": [], "text": f"измерение не найдено: {which}"}
    if a[1] or b[1]:
        return {"verdict": "incomparable", "axes": [], "unmeasured": [],
                "text": "несравнимы: исторический замер не сравнивается со значениями новой методики"}
    if a[0] != b[0]:
        return {"verdict": "incomparable", "axes": [], "unmeasured": [],
                "text": f"несравнимы: методика {a[0]} и {b[0]}; числа разных методик не сравниваются"}
    ax, bx = json.loads(a[2]), json.loads(b[2])
    unmeasured = [k for k in AXES if not ax[k]["known"] or not bx[k]["known"]]
    if unmeasured:
        names = ", ".join(AXIS_RU[k] for k in unmeasured)
        return {"verdict": "incomparable", "axes": [], "unmeasured": unmeasured,
                "text": f"несравнимы: не измерено: {names}"}
    diff = [k for k in AXES if canon(ax[k]["value"]) != canon(bx[k]["value"])]
    if not diff:
        return {"verdict": "comparable", "axes": [], "unmeasured": [], "text": "сравнимы: все оси совпадают"}
    names = ", ".join(AXIS_RU[k] for k in diff)
    if len(diff) == 1:
        return {"verdict": "by_axis", "axes": diff, "unmeasured": [],
                "text": f"сравнение по оси «{names}»: остальные оси совпадают"}
    return {"verdict": "incomparable", "axes": diff, "unmeasured": [],
            "text": f"несравнимы: отличаются оси: {names}; вывода не делается"}


# ── команда rt.py bench-db ───────────────────────────────────────────────────────────────────────

def _print_report(rep, apply):
    c = rep["counts"]
    label = {"new": "новых", "loaded": "загружено", "known": "уже учтено", "conflict": "конфликтов",
             "rejected": "отклонено"}
    print("Файлов: " + (", ".join(f"{label[k]} {c[k]}" for k in label if k in c) or "нет"))
    for it in rep["files"]:
        if it["action"] in ("new", "conflict", "rejected", "loaded") and it.get("reason"):
            print(f"  {it['file']}: {it['action']} — {it['reason']}")
    if not apply and c.get("new"):
        print("Ничего не записано. Чтобы загрузить: rt.py bench-db load --apply")


def cli(a, bench_dir=None):
    """Выполняет подкоманду rt.py bench-db; возвращает код выхода. bench_dir — подмена для тестов."""
    if bench_dir is None:
        import paths
        bench_dir = paths.BENCH
    try:
        if a.bench_cmd == "status":
            st = status(bench_dir)
            print(f"База: {'есть' if st['db_exists'] else 'нет (создаётся командой load --apply)'}")
            for m, n in st["by_method"].items():
                print(f"  методика {m}: {n} {plural(n, ('прогон', 'прогона', 'прогонов'))}")
            print(f"Не загружено файлов: {st['not_loaded']}; конфликтов: {st['conflicts']}; "
                  f"отклонено: {st['rejected']}")
            print(HELP_APPEND_ONLY)
        elif a.bench_cmd == "load":
            if a.historical:
                rep = load_historical(bench_dir, a.historical, apply=a.apply)
            else:
                rep = apply_load(bench_dir) if a.apply else plan_load(bench_dir)
            _print_report(rep, a.apply)
            print(HELP_APPEND_ONLY)
        elif a.bench_cmd == "compare":
            print(compare(bench_dir, a.first, a.second)["text"])
        else:       # rt.py не пропускает другие значения; ветка для прямого вызова benchdb.cli
            print("Подкоманды: status, load, compare")
            return 2
    except BenchDbError as ex:
        print(f"Ошибка: {ex}", file=sys.stderr)
        return 1
    except Exception as ex:         # noqa: BLE001 — человеку не показывается трейсбек, только вид сбоя
        print(f"Ошибка: непредвиденный сбой ({type(ex).__name__}); база и файлы прогонов не изменены "
              f"сверх завершённых транзакций", file=sys.stderr)
        return 1
    return 0
