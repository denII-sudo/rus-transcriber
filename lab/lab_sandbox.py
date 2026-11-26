"""Песочница лаборатории: проверка места и подготовка копии (команда prepare).

Что делает prepare (подробно — lab/README.md, раздел «Что копируется»):
  * база записей state.sqlite3 копируется штатной резервной копией программы
    (storage.Store.backup). Если рядом с базой нет -wal и -shm, она открывается штатно
    storage.open_store_readonly (неизменяемой, без служебных файлов); иначе база и -wal
    читаются побайтно во временную папку песочницы и копия делается оттуда (copy_database:
    штатное чтение при служебных файлах меняет -shm). Копия неактивна: программа по ней не
    пишет файлы рядом с аудио, пока её явно не перепривязать;
  * config.json, словари групп (groups/<группа>/vocab.json) и соль стенда
    (bench/testbed.json) копируются как есть;
  * голосовые профили (voices.json, voice-settings.json) — это биометрия — копируются только
    по явному выбору --voice-profiles copy;
  * аудиофайлы записей из базы только читаются (контрольная сумма); внешние файлы (--audio,
    add-audio) копируются в audio/ песочницы.

Режим обычной работы (prepare --work): копируются только база записей и файлы прогонов гипотез
(source/hypotheses/, опись со сверкой SHA-256 до и после — source/hypotheses-manifest.json); аудио,
настройки, словари и профили не копируются и не читаются, единиц опыта нет.

Настоящая папка данных не меняется ни при каких условиях: в неё ничего не пишется.
Модуль не импортирует paths.py (у его импорта побочные эффекты), только storage, runinfo и hypstore.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
REPO_DIR = LAB_DIR.parent
MARKER = "LAB-SANDBOX.json"
MARKER_SCHEMA = "lab-sandbox-1"
UNITS_FILE = "units.json"
GROUP_FILES_ALWAYS = ("vocab.json",)
GROUP_FILES_VOICES = ("voices.json", "voice-settings.json")


class SandboxError(RuntimeError):
    """Опасное или неверное место песочницы; работа не начинается."""


def _import_repo_module(name):
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    return __import__(name)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_data_dirs(env=None):
    """Возможные места настоящей папки данных: явная переменная и место по умолчанию (как paths.py)."""
    env = os.environ if env is None else env
    out = []
    explicit = (env.get("RUS_TRANSCRIBER_DATA") or "").strip()
    if explicit:
        out.append(Path(explicit).expanduser())
    if os.name == "nt":
        out.append(Path.home() / "AppData" / "Roaming" / "rus-transcriber-data")
    else:
        out.append(Path.home() / ".local" / "share" / "rus-transcriber-data")
    return out


def default_program_dir(env=None):
    env = os.environ if env is None else env
    base = (env.get("LOCALAPPDATA") or "").strip()
    return (Path(base) if base else Path.home() / "AppData" / "Local") / "rus-transcriber"


def real(path):
    """Полный путь с раскрытыми ссылками и junction (os.path.realpath делает это и в Windows)."""
    return Path(os.path.realpath(os.path.abspath(os.path.expanduser(str(path)))))


def _key(path):
    return os.path.normcase(str(path)).rstrip("\\/")


def overlaps(a, b):
    """Совпадают ли папки или одна лежит внутри другой (без учёта регистра в Windows)."""
    ka, kb = _key(real(a)), _key(real(b))
    if ka == kb:
        return True
    sep = os.sep
    return ka.startswith(kb + sep) or kb.startswith(ka + sep)


def check_sandbox(sandbox, protected, audio_dirs=()):
    """Отказ, если песочница совпадает с охраняемой папкой, лежит в ней или содержит её.

    protected — {"описание": путь}; audio_dirs — папки с исходными записями."""
    s = real(sandbox)
    if s.parent == s:
        raise SandboxError(f"песочница не может быть корнем диска: {s}")
    for what, path in protected.items():
        if path and overlaps(s, path):
            raise SandboxError(f"песочница {s} совпадает с папкой «{what}» ({real(path)}), "
                               "лежит внутри неё или содержит её — выберите отдельную папку")
    for d in audio_dirs:
        if d and overlaps(s, d):
            raise SandboxError(f"песочница {s} пересекается с папкой исходных записей — "
                               "выберите папку вне папок с аудио")
    return s


def protected_dirs(source_data, program_dir, env=None):
    out = {"источник данных (настоящая папка данных)": source_data,
           "папка установленной программы": program_dir,
           "клон репозитория с кодом лаборатории": REPO_DIR}
    for n, d in enumerate(default_data_dirs(env), 1):
        out[f"папка данных программы (вариант {n})"] = d
    return out


def _parse_hms(value):
    try:
        parts = [int(x) for x in str(value).split(":")]
        sec = 0
        for p in parts:
            sec = sec * 60 + p
        return float(sec)
    except (TypeError, ValueError):
        return None


NUMERIC_TIMING = ("asr_sec", "asr2_sec", "asr3_sec", "asr3_load_sec", "diarization_sec", "voices_sec",
                  "word_voice_sec", "embeddings_sec", "total_sec", "interaction_sec")
TEXT_TIMING = ("asr_model", "asr_backend", "asr2_model", "asr3_model", "asr3_device", "asr3_mode",
               "asr3_pieces")


def read_recordings(db):
    """Записи из базы: [{recording_id, location, group, no_profile, tag, duration_sec, timing}].

    Читается текущая ревизия стенограммы; из неё берутся только служебные сведения."""
    rows = db.execute(
        "SELECT r.recording_id AS rid, r.location AS location, t.payload AS payload "
        "FROM recordings r JOIN transcript_revisions t "
        "ON t.recording_id = r.recording_id AND t.revision = r.source_revision "
        "ORDER BY r.location, r.recording_id").fetchall()
    out = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            payload = {}
        quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
        runtime = quality.get("runtime") if isinstance(quality.get("runtime"), dict) else {}
        timing = runtime.get("timing") if isinstance(runtime.get("timing"), dict) else {}
        variant = ((quality.get("migration") or {}).get("variant") or {}) \
            if isinstance(quality.get("migration"), dict) else {}
        out.append({
            "recording_id": row["rid"], "location": row["location"],
            "group": runtime.get("group"), "no_profile": bool(runtime.get("no_profile")),
            "tag": variant.get("tag") if isinstance(variant, dict) else None,
            "duration_sec": _parse_hms(payload.get("duration")),
            "timing": {**{k: timing[k] for k in NUMERIC_TIMING if isinstance(timing.get(k), (int, float))},
                       **{k: str(timing[k]) for k in TEXT_TIMING if timing.get(k) is not None}},
        })
    return out


def build_units(recordings, sha256_file):
    """Записи -> тестовые единицы: одна единица на одинаковый звук (контрольная сумма файла)."""
    units, missing, by_location = {}, [], {}
    for rec in recordings:
        loc = rec["location"]
        if loc not in by_location:
            p = Path(loc)
            by_location[loc] = sha256_file(p) if p.is_file() else None
        sha = by_location[loc]
        if not sha:
            missing.append(rec["recording_id"])
            continue
        unit = units.setdefault(sha, {"unit_id": sha[:16], "audio_sha256": sha, "location": loc,
                                      "size_bytes": Path(loc).stat().st_size, "recordings": []})
        unit["recordings"].append(rec)
    for unit in units.values():
        groups = [r["group"] for r in unit["recordings"] if r["group"]]
        unit["group"] = max(set(groups), key=groups.count) if groups else None
        unit["group_conflict"] = len(set(groups)) > 1
        unit["no_profile"] = any(r["no_profile"] for r in unit["recordings"]) and not groups
        durations = [r["duration_sec"] for r in unit["recordings"] if r["duration_sec"]]
        unit["duration_sec"] = max(durations) if durations else None
    return sorted(units.values(), key=lambda u: u["location"].lower()), missing


def _copy(src, dst, copied, what, root):
    """Копия файла: источник открывается только на чтение, существующий файл не заменяется."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in, open(dst, "xb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    copied.append({"what": what, "relative": dst.relative_to(root).as_posix(), "bytes": dst.stat().st_size})


def _read_stable(path, attempts=5):
    """Байты файла, не менявшегося во время чтения (сверка размера и времени до и после)."""
    for _ in range(attempts):
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) == \
                (len(data), after.st_mtime_ns):
            return data, (after.st_size, after.st_mtime_ns)
        time.sleep(0.5)
    raise SandboxError("база записей меняется во время копирования: закройте программу и повторите")


def stage_database(src_db, staging, attempts=5):
    """Побайтная копия базы и журнала -wal в staging; источник только читается.

    Пара «база + журнал» должна относиться к одному моменту: если за время чтения что-то
    изменилось, чтение повторяется. -> путь копии базы."""
    wal = src_db.with_name(src_db.name + "-wal")
    for _ in range(attempts):
        db_bytes, db_state = _read_stable(src_db)
        wal_bytes, wal_state = _read_stable(wal) if wal.is_file() else (None, None)
        if src_db.stat().st_mtime_ns == db_state[1] and \
                (wal_state is None or (wal.is_file() and wal.stat().st_mtime_ns == wal_state[1])):
            break
    else:
        raise SandboxError("база записей меняется во время копирования: закройте программу и повторите")
    staging.mkdir(parents=True, exist_ok=False)
    target = staging / "state.sqlite3"
    target.write_bytes(db_bytes)
    if wal_bytes is not None:
        (staging / "state.sqlite3-wal").write_bytes(wal_bytes)
    return target


AUDIO_ORIGIN = "audio_file"      # единица из указанного файла, а не из базы записей


def audio_duration(path):
    """Длительность звука в секундах по самому файлу: ffprobe, для WAV — без него. None — не узнать."""
    import subprocess
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                           capture_output=True, timeout=60,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if p.returncode == 0:
            value = float(p.stdout.decode("ascii", "replace").strip())
            if value > 0:
                return round(value, 2)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        import wave
        with wave.open(str(path), "rb") as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:
        return None


def _valid_group(group):
    value = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in group)
    if not group or value != group or len(group) > 64:
        raise SandboxError("--group: имя группы — буквы, цифры, «-» и «_», как папка groups/<группа>")
    return group


def _group_hint(s, group, log):
    """Предупредить заранее о том, как конвейер выберет словарь и профиль для нового файла."""
    groups_dir = s / "data" / "groups"
    known = sorted(d.name for d in groups_dir.iterdir()
                   if d.is_dir() and ((d / "vocab.json").exists() or (d / "voices.json").exists())) \
        if groups_dir.is_dir() else []
    if group:
        if group not in known:
            log(f"  ВНИМАНИЕ: группы «{group}» в песочнице нет — словаря и голосового профиля не будет.")
        return
    default = None
    try:
        default = json.loads((s / "data" / "config.json").read_text(encoding="utf-8-sig")).get("default_group")
    except (OSError, ValueError, AttributeError):
        pass
    if default:
        log(f"  Группа не задана: как при обычной расшифровке нового файла возьмётся default_group "
            f"из config.json («{default}»), если рядом с файлом нет своего словаря.")
    elif len(known) == 1:
        log(f"  Группа не задана: как при обычной расшифровке возьмётся единственная группа «{known[0]}», "
            "если рядом с файлом нет своего словаря.")
    elif known:
        log("  ВНИМАНИЕ: группа не задана, а групп несколько и default_group нет — программа при "
            "расшифровке такого файла откажет, прогон этой единицы завершится ошибкой; укажите --group.")
    else:
        log("  Групп нет: файл посчитается без словаря и голосовых профилей.")


AUDIO_DIR = "audio"              # копии внешних файлов внутри песочницы


def _copy_audio(src, dest_dir, sha, sha256_file):
    """Копия внешнего файла в audio/ песочницы под тем же именем. Источник только читается.

    Файл с тем же именем и той же суммой — уже скопирован, используется; с другой суммой — отказ
    (чужой файл не перезаписывается). -> (путь копии, скопирован ли сейчас)."""
    dest = dest_dir / src.name
    if dest.exists():
        if dest.is_file() and sha256_file(dest) == sha:
            return dest, False
        raise SandboxError(f"в {AUDIO_DIR}/ песочницы уже есть другой файл с именем «{src.name}»; "
                           "он не перезаписывается — переименуйте исходный файл или уберите прежний")
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / (dest.name + ".copying")
    with open(src, "rb") as f_in, open(tmp, "xb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    if sha256_file(tmp) != sha:
        tmp.unlink()
        raise SandboxError(f"файл {src.name} изменился во время копирования; повторите")
    os.link(tmp, dest)        # не заменяет существующий файл, если он появился за это время
    tmp.unlink()
    return dest, True


def is_own_audio(s, unit):
    """Единица из внешнего файла, чья копия лежит в audio/ этой песочницы."""
    if unit.get("origin") != AUDIO_ORIGIN:
        return False
    loc = real(unit["location"])
    return _key(loc.parent) == _key(real(s) / AUDIO_DIR)


def add_audio_units(sandbox, files, group=None, env=None, log=print, _marker=None):
    """Добавить единицы опыта из указанных аудиофайлов (их нет в базе записей). -> список unit_id.

    Файл копируется в audio/ песочницы, дальше лаборатория работает с копией; исходный файл
    только читается, его папка с песочницей сравнивается не должна (это лишь источник копии).
    Длительность — по самому звуку. Уже добавленный звук (та же сумма) повторно не добавляется."""
    s = real(sandbox)
    marker = _marker or json.loads((s / MARKER).read_text(encoding="utf-8"))
    runinfo = _import_repo_module("runinfo")
    if group:
        _valid_group(group)
    paths = []
    for f in files:
        p = real(f)
        if not p.is_file():
            raise SandboxError(f"аудиофайл не найден: {p}")
        paths.append(p)
    check_sandbox(s, protected_dirs(marker["source_data"], marker["program_dir"], env))
    units_path = s / UNITS_FILE
    doc = json.loads(units_path.read_text(encoding="utf-8")) if units_path.is_file() else \
        {"schema": "lab-units-1", "units": [], "missing_audio_recordings": []}
    known = {u["audio_sha256"]: u for u in doc["units"]}
    added = []
    for p in paths:
        sha = runinfo.sha256_file(p)
        if not sha:
            raise SandboxError(f"аудиофайл не читается: {p}")
        if sha in known:
            log(f"Звук {p.name} уже есть в песочнице как единица {known[sha]['unit_id']}; заново не копируется.")
            added.append(known[sha]["unit_id"])
            continue
        copy_path, fresh = _copy_audio(p, s / AUDIO_DIR, sha, runinfo.sha256_file)
        unit = {"unit_id": sha[:16], "audio_sha256": sha, "location": str(copy_path),
                "source_location": str(p), "size_bytes": copy_path.stat().st_size,
                "origin": AUDIO_ORIGIN, "group": group or None, "group_conflict": False, "no_profile": False,
                "duration_sec": audio_duration(copy_path), "recordings": []}
        doc["units"].append(unit)
        known[sha] = unit
        added.append(unit["unit_id"])
        dur = f"{unit['duration_sec'] / 60:.1f} мин" if unit["duration_sec"] else "длительность не узнана"
        log(f"Добавлена единица {unit['unit_id']} из файла {p.name} ({dur}): "
            + ("скопирован" if fresh else "уже был скопирован") + f" в {AUDIO_DIR}/ песочницы, считается копия; "
            "исходный файл только прочитан. Решений человека для неё нет.")
    _group_hint(s, group, log)
    tmp = units_path.with_name(units_path.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(units_path)
    return added


def _side_files(db):
    return [p for p in (db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")) if p.exists()]


def _file_state(path):
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def copy_database(src_db, dest, staging):
    """Неактивная резервная копия базы записей в песочницу. -> (отчёт проверки копии, способ).

    «readonly» — штатно: storage.open_store_readonly + Store.backup. Применяется, только когда
    рядом с базой нет ни -wal, ни -shm (программа базу не держит): тогда SQLite открывает её как
    неизменяемую и рядом ничего не создаёт. Если за время копирования база или служебные файлы
    изменились (программа начала писать), копия выбрасывается и делается побайтно.
    «bytes» — запасной путь, когда служебные файлы есть: штатное чтение в этом случае меняет
    рядом файл -shm (обновляет его содержимое, а при одном лишь -wal создаёт новый -shm),
    поэтому база и -wal читаются побайтно во временную папку песочницы, и резервная копия
    делается уже оттуда."""
    storage = _import_repo_module("storage")
    if not _side_files(src_db):
        before = _file_state(src_db)
        store = storage.open_store_readonly(src_db)
        try:
            report = store.backup(dest)
        finally:
            store.close()
        if not _side_files(src_db) and _file_state(src_db) == before:
            return report, "readonly"
        dest.unlink()                    # снимок мог оказаться несогласованным: делаем заново
    staged = stage_database(src_db, staging)
    try:
        store = storage.open_store(staged)
        try:
            report = store.backup(dest)
        finally:
            store.close()
    finally:
        if staging.exists():
            shutil.rmtree(staging)       # побайтная копия больше не нужна
    return report, "bytes"


HYP_DIR = "hypotheses"                       # папка гипотез в папке данных и в source/ песочницы
HYP_MANIFEST = "hypotheses-manifest.json"    # опись копии гипотез: source/hypotheses-manifest.json
HYP_MANIFEST_SCHEMA = "lab-hyp-manifest-1"
HYP_COPIED = "copied"
HYP_RESULTS = (HYP_COPIED, "copy_mismatch", "changed_during_copy", "deleted_during_copy")


def describe_hypotheses(src_dir):
    """Опись файлов прогонов гипотез: {относительное имя: {bytes, mtime_ns, sha256}}.

    Только файлы прогонов по маскам hypstore (UNIT_RE/RUN_RE): кэши _artifacts.json, _testbed.json и прочее
    не берутся; ссылкам (symlink, junction) обход не следует — тем же правилом, что hypstore._run_files.
    Папка только читается; если её нет, она не создаётся."""
    hypstore = _import_repo_module("hypstore")
    runinfo = _import_repo_module("runinfo")
    src_dir = Path(src_dir)
    out = {}
    for p in hypstore._run_files(src_dir):
        try:
            st = p.stat()
        except OSError:
            continue
        digest = runinfo.sha256_file(p)
        if digest is None:
            continue
        out[p.relative_to(src_dir).as_posix()] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                                                  "sha256": digest}
    return out


def _copy_file(src, dst):
    """Побайтная копия: источник открывается только на чтение, существующий файл не заменяется."""
    with open(src, "rb") as f_in, open(dst, "xb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def copy_hypotheses(src_dir, dest_dir, manifest_path):
    """Копия файлов гипотез со сверкой SHA-256 до и после. -> опись (она же пишется в manifest_path).

    1. Опись источника до копирования (размер, время изменения, SHA-256 каждого файла прогона).
    2. Копия каждого файла сверяется с суммой описи; не сошлась — копия выбрасывается (copy_mismatch).
    3. Опись источника после копирования: файл изменился (размер, время или сумма) — копия выбрасывается
       (changed_during_copy); исчез — тоже (deleted_during_copy). Файлы гипотез пишутся один раз, поэтому
       любое изменение — повод не брать файл. Файлы, появившиеся за время копирования, не берутся.
    В описи только относительные имена внутри hypotheses/, путей настоящей папки данных нет."""
    runinfo = _import_repo_module("runinfo")
    src_dir, dest_dir = Path(src_dir), Path(dest_dir)
    before = describe_hypotheses(src_dir)
    results = {}
    for rel, info in sorted(before.items()):
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            _copy_file(src_dir / rel, target)
        except FileNotFoundError:
            results[rel] = "deleted_during_copy"
            continue
        results[rel] = HYP_COPIED if runinfo.sha256_file(target) == info["sha256"] else "copy_mismatch"
    after = describe_hypotheses(src_dir)
    for rel, info in before.items():
        if results[rel] != HYP_COPIED:
            continue
        if rel not in after:
            results[rel] = "deleted_during_copy"
        elif after[rel] != info:
            results[rel] = "changed_during_copy"
    for rel, result in results.items():
        target = dest_dir / rel
        if result != HYP_COPIED and target.exists():
            target.unlink()
    files = [{"relative": rel, "bytes": before[rel]["bytes"], "sha256": before[rel]["sha256"],
              "result": results[rel]} for rel in sorted(before)]
    manifest = {"schema": HYP_MANIFEST_SCHEMA, "created_utc": utc_now(), "files": files,
                "appeared_during_copy": len(set(after) - set(before)),
                "counts": {k: sum(1 for f in files if f["result"] == k) for k in HYP_RESULTS}}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def add_database_hypotheses(db_copy, dest_dir, manifest_path, manifest):
    """Гипотезы из копии базы песочницы (таблица hypotheses, Г-1) — файлами в dest_dir, как прежние файлы:
    <sound_id>/<run_id>.json, тот же текст записи hyp-2; в опись — с origin=database и суммой SHA-256 файла.

    Источник — только копия базы в песочнице (настоящая база здесь не читается). Прогон, который уже взят
    прежним файлом (тот же относительный путь), не повторяется: в описи один раз. Базы до Г-1 (без таблицы)
    ничего не добавляют. -> обновлённая опись (она же переписывается в manifest_path)."""
    runinfo = _import_repo_module("runinfo")
    storage = _import_repo_module("storage")
    dest_dir = Path(dest_dir)
    for f in manifest["files"]:
        f.setdefault("origin", "files")
    taken = {f["relative"] for f in manifest["files"] if f["result"] == HYP_COPIED}
    store = storage.open_store_readonly(db_copy)
    try:
        rows = store.hypotheses_rows()
    finally:
        store.close()
    duplicates = 0
    for row in rows:
        rel = f"{row['sound_id']}/{row['run_id']}.json"
        if rel in taken:
            duplicates += 1
            continue
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = row["record_json"].encode("utf-8")
        with open(target, "xb") as out:
            out.write(data)
        manifest["files"].append({"relative": rel, "bytes": len(data), "sha256": runinfo.sha256_file(target),
                                  "result": HYP_COPIED, "origin": "database"})
        taken.add(rel)
    manifest["files"].sort(key=lambda f: f["relative"])
    manifest["from_database"] = len(rows) - duplicates
    manifest["database_duplicates_of_files"] = duplicates
    manifest["counts"] = {k: sum(1 for f in manifest["files"] if f["result"] == k) for k in HYP_RESULTS}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def prepare(sandbox, source_data, program_dir, voice_profiles, env=None, log=print, audio=(), group=None,
            only_audio=False, work=False):
    """Создать песочницу. -> отчёт (что скопировано, сколько единиц, каких записей нет).

    audio — файлы для единиц вне базы записей; only_audio — только они: база записей не копируется
    и не нужна (пробный запуск на отрывке). work — песочница обычной работы (анализ по сохранённым
    гипотезам и ответам человека, без прогона моделей): копируются только база записей и файлы гипотез;
    аудио, голосовые профили, словари и настройки не копируются и не читаются, единиц опыта нет."""
    if voice_profiles not in ("copy", "skip"):
        raise SandboxError("--voice-profiles: copy или skip")
    if only_audio and not audio:
        raise SandboxError("--only-audio: укажите хотя бы один --audio")
    if work and (audio or only_audio or group):
        raise SandboxError("--work: без --audio, --only-audio и --group — модели в этом режиме не запускаются")
    if work and voice_profiles != "skip":
        raise SandboxError("--work: голосовые профили не нужны, укажите --voice-profiles skip")
    if group:
        _valid_group(group)
    source_data = real(source_data)
    audio = [real(a) for a in audio]
    for a in audio:
        if not a.is_file():
            raise SandboxError(f"аудиофайл не найден: {a}")
    # папка внешнего файла с песочницей не сравнивается: файл лишь копируется в audio/
    check_sandbox(sandbox, protected_dirs(source_data, program_dir, env))
    src_db = source_data / "state.sqlite3"
    if not only_audio and not src_db.is_file():
        raise SandboxError(f"в папке данных нет базы записей state.sqlite3: {source_data}")
    storage = _import_repo_module("storage")
    runinfo = _import_repo_module("runinfo")

    s = real(sandbox)
    if s.exists() and any(s.iterdir()):
        raise SandboxError(f"папка песочницы уже существует и не пуста: {s}. "
                           "Для новой подготовки укажите новую папку")
    copied, report, copy_method = [], None, None
    if not only_audio:
        (s / "source").mkdir(parents=True, exist_ok=True)
        report, copy_method = copy_database(src_db, s / "source" / "state.sqlite3", s / "source" / "staging")
        store = storage.open_store_readonly(s / "source" / "state.sqlite3")
        try:
            recordings = read_recordings(store.db)
        finally:
            store.close()
        audio_dirs = sorted({str(real(Path(r["location"]).parent)) for r in recordings})
        try:
            check_sandbox(s, protected_dirs(source_data, program_dir, env), audio_dirs)
        except SandboxError:
            shutil.rmtree(s / "source")      # только что созданные лабораторией файлы
            raise
        copied.append({"what": "база записей (резервная копия, неактивная)", "relative": "source/state.sqlite3",
                       "bytes": (s / "source" / "state.sqlite3").stat().st_size, "method": copy_method})
    hyp = None
    if work:
        # после базы: прежний файл гипотез (до Г-1) писался после записи, поэтому гипотеза без записи в копии
        # базы возможна — это отметка анализа «записи нет в копии», а не ошибка
        hyp = copy_hypotheses(source_data / HYP_DIR, s / "source" / HYP_DIR, s / "source" / HYP_MANIFEST)
        # с Г-1 гипотезы — в базе: выкладка из копии базы песочницы в те же файлы и ту же опись
        hyp = add_database_hypotheses(s / "source" / "state.sqlite3", s / "source" / HYP_DIR,
                                      s / "source" / HYP_MANIFEST, hyp)
        kept = [f for f in hyp["files"] if f["result"] == HYP_COPIED]
        copied.append({"what": "гипотезы моделей (полный текст записей; только на этом компьютере)",
                       "relative": f"source/{HYP_DIR}", "bytes": sum(f["bytes"] for f in kept), "files": len(kept),
                       "sha256": runinfo.sha256_file(s / "source" / HYP_MANIFEST), "method": "sha256"})
    s.mkdir(parents=True, exist_ok=True)
    data = s / "data"
    data.mkdir(exist_ok=True)
    # в режиме work настройки, соль стенда, словари и профили не копируются: анализу обычной работы не нужны
    if not work and (source_data / "config.json").is_file():
        _copy(source_data / "config.json", data / "config.json", copied, "настройки config.json", s)
    if not work and (source_data / "bench" / "testbed.json").is_file():
        _copy(source_data / "bench" / "testbed.json", data / "bench" / "testbed.json", copied,
              "соль обезличенного идентификатора стенда", s)
    groups = source_data / "groups"
    names = GROUP_FILES_ALWAYS + (GROUP_FILES_VOICES if voice_profiles == "copy" else ())
    if not work and groups.is_dir():
        for g in sorted(groups.iterdir()):
            if not g.is_dir():
                continue
            for name in names:
                if (g / name).is_file():
                    what = "голосовой профиль (биометрия)" if name in GROUP_FILES_VOICES else "словарь группы"
                    _copy(g / name, data / "groups" / g.name / name, copied, what, s)

    units, missing = [], []
    if not only_audio and not work:
        # единицы — по копии базы (записи прочитаны из неё выше); в режиме work аудио не читается
        units, missing = build_units(recordings, runinfo.sha256_file)
    (s / UNITS_FILE).write_text(json.dumps({"schema": "lab-units-1", "units": units,
                                            "missing_audio_recordings": missing},
                                           ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for sub in ("tmp", "results", "export", "cache"):
        (s / sub).mkdir(exist_ok=True)
    marker = {"schema": MARKER_SCHEMA, "created_utc": utc_now(), "source_data": str(source_data),
              "program_dir": str(real(program_dir)), "repo_dir": str(REPO_DIR),
              "voice_profiles": voice_profiles, "copied": copied, "only_audio": bool(only_audio),
              "backup_report": report if isinstance(report, (dict, list)) else None}
    if work:
        marker["work"] = True       # нет поля — обычная песочница (старые песочницы читаются как раньше)
    (s / MARKER).write_text(json.dumps(marker, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if audio:
        add_audio_units(s, audio, group, env=env, log=log, _marker=marker)
        units = json.loads((s / UNITS_FILE).read_text(encoding="utf-8"))["units"]
    out = {"sandbox": str(s), "copied": copied, "units": len(units),
           "recordings": sum(len(u["recordings"]) for u in units),
           "audio_hours": round(sum(u["duration_sec"] or 0 for u in units) / 3600, 2),
           "missing_audio_recordings": len(missing), "voice_profiles": voice_profiles}
    if work:
        out["hypotheses"] = dict(hyp["counts"], appeared_during_copy=hyp["appeared_during_copy"],
                                 from_database=hyp["from_database"])
        out["recordings"] = len(recordings)          # записей в копии базы; единиц опыта в этом режиме нет
    log("Песочница подготовлена: " + str(s))
    log("Скопировано (только на этом компьютере, наружу ничего не передаётся):")
    for item in copied:
        how = {"readonly": "; штатное чтение только на чтение",
               "bytes": "; база была занята или с журналом — прочитана побайтно"}.get(item.get("method"), "")
        log(f"  - {item['what']}: {item['relative']} ({item['bytes']} байт{how})")
    if work:
        c = hyp["counts"]
        log(f"  Гипотезы моделей: скопировано файлов {c[HYP_COPIED]}, каждый сверен по SHA-256 до и после; "
            f"не взяты — изменились при копировании: {c['changed_during_copy']}, удалены при копировании: "
            f"{c['deleted_during_copy']}, копия не сошлась: {c['copy_mismatch']}; появились за время "
            f"копирования и не взяты: {hyp['appeared_during_copy']}. Опись — source/{HYP_MANIFEST}.")
        log("  Режим обычной работы (--work): аудио не читалось и не копировалось; настройки, словари и "
            "голосовые профили не копировались; модели не запускаются, единиц опыта нет. Копия гипотез — "
            "полный текст записей: удалите песочницу после опыта.")
        log(f"Записей в базе: {len(recordings)}.")
        return out
    if voice_profiles == "copy":
        log("  ВНИМАНИЕ: скопированы голосовые профили — это биометрия участников. Они лежат только "
            "в песочнице; удалить их можно вместе с папкой песочницы.")
    else:
        log("  Голосовые профили не копировались (--voice-profiles skip): имена говорящих не подставятся.")
    if only_audio:
        log("  База записей не копировалась (--only-audio): в песочнице только указанные файлы, "
            "решений человека для них нет.")
    log(f"Записей в базе: {out['recordings']}, разных звуков (единиц опыта): {out['units']}, "
        f"около {out['audio_hours']} ч звука; аудио не найдено у записей: {out['missing_audio_recordings']}.")
    return out


def load_sandbox(sandbox, env=None):
    """Проверить готовую песочницу перед прогоном. -> (путь, маркер, единицы)."""
    s = real(sandbox)
    marker_path = s / MARKER
    if not marker_path.is_file():
        raise SandboxError(f"{s} — не песочница лаборатории (нет {MARKER}); сначала выполните prepare")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != MARKER_SCHEMA:
        raise SandboxError("песочница подготовлена другой версией лаборатории")
    units = json.loads((s / UNITS_FILE).read_text(encoding="utf-8"))
    for u in units["units"]:
        if u.get("origin") == AUDIO_ORIGIN and not is_own_audio(s, u):
            raise SandboxError(f"единица {u['unit_id']} из внешнего файла должна читаться из {AUDIO_DIR}/ "
                               "песочницы; добавьте файл заново командой add-audio")
    # записи из базы: песочница не должна пересекаться с их папками; копии внешних файлов —
    # собственная подпапка audio/ песочницы, для них эта проверка не нужна
    audio_dirs = sorted({str(real(Path(u["location"]).parent)) for u in units["units"] if not is_own_audio(s, u)})
    check_sandbox(s, protected_dirs(marker["source_data"], marker["program_dir"], env), audio_dirs)
    if (s / "data" / "state.sqlite3").exists():
        raise SandboxError("в data/ песочницы появилась state.sqlite3: лаборатория её не создаёт; "
                           "уберите файл вручную, прежде чем продолжать")
    return s, marker, units
