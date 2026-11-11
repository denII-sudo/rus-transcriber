"""Ядро rus-transcriber: задачи (запуск, статус, отмена), чтение стенограмм, настройки, данные.

Здесь нет ничего, что знало бы про MCP, Claude или любой другой клиент. Слои управления — server.py (MCP) и
rt.py (командная строка) — только вызывают эти функции и показывают результат. Любой третий слой (HTTP, окно,
другой протокол) строится так же.

Соглашения: функции возвращают обычные структуры (dict, list, str); понятная пользователю ошибка — CoreError.
Опасные действия (выгрузка, загрузка, удаление, очистка, перенос) здесь только исполняются; решение человека
получает слой управления (confirm.py для MCP, ввод в терминале для rt.py).
"""
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402  (дочерние процессы унаследуют HF_HOME и др.)
import settings  # noqa: E402
import migrate  # noqa: E402
import data_io  # noqa: E402

JOBS = paths.JOBS
JOBS.mkdir(parents=True, exist_ok=True)

ACTIVE = ("queued", "running")
FINISHED = ("done", "error", "cancelled")


class CoreError(Exception):
    """Ошибка, текст которой можно показать человеку как есть."""


# ---------- процессы и статусы задач ----------
def alive(pid):
    """Жив ли процесс (без psutil; os.kill на Windows процесс бы завершил)."""
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))       # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259           # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def pid_of(job_id):
    try:
        return int((JOBS / f"{job_id}.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def job_status(job_id):
    """Статус задачи (dict) или None, если такой задачи нет. Если процесс пропал без итога — state=error."""
    p = JOBS / f"{job_id}.json"
    if not p.exists():
        return None
    for _ in range(3):        # файл могут как раз заменять (os.replace в worker) — пробуем ещё раз
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            break
        except (json.JSONDecodeError, OSError):
            time.sleep(0.1)
    else:
        return {"state": "running", "stage": "обновляется"}
    if d.get("state") in ACTIVE:
        pid = pid_of(job_id)
        if pid and not alive(pid):
            time.sleep(0.5)          # процесс мог только что записать итог
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
            if d.get("state") in ACTIVE:
                d.update(state="error", stage="процесс завершился без итога",
                         error=f"процесс задачи пропал — подробности в jobs\\{job_id}.log")
    return d


def _need(job_id):
    s = job_status(job_id)
    if not s:
        raise CoreError(f"Задача {job_id} не найдена")
    return s


def launch(script, args, src, kind):
    """Запускает фоновый процесс-задачу (worker.py и др.), возвращает job_id."""
    job_id = time.strftime("%Y%m%d-%H%M%S-") + kind[0] + uuid.uuid4().hex[:5]
    status = JOBS / f"{job_id}.json"
    status.write_text(json.dumps({"state": "queued", "kind": kind, "file": str(src)}, ensure_ascii=False),
                      encoding="utf-8")
    cmd = [sys.executable, str(HERE / script), str(src), "--status", str(status), *args]
    log = open(JOBS / f"{job_id}.log", "w", encoding="utf-8")
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            cwd=str(HERE), creationflags=flags, env=env)
    (JOBS / f"{job_id}.pid").write_text(str(proc.pid), encoding="utf-8")
    return job_id


def cancel_job(job_id):
    """Останавливает идущую или ждущую задачу вместе с дочерними процессами; готовые результаты не трогает."""
    s = _need(job_id)
    if s.get("state") in FINISHED:
        return {"job_id": job_id, "state": s.get("state"), "note": "задача уже не идёт"}
    pid = pid_of(job_id)
    killed = False
    if pid and alive(pid):
        if os.name == "nt":
            r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True,
                               creationflags=0x08000000)
            killed = r.returncode == 0
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
            killed = True
    for _ in range(10):
        if not pid or not alive(pid):
            break
        time.sleep(0.3)
    st = {**s, "state": "cancelled", "stage": "отменено", "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    (JOBS / f"{job_id}.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"job_id": job_id, "state": "cancelled", "was": s.get("stage"), "process_killed": killed}


def start_transcription(audio_path, num_speakers=0, emotions=True, vocab_path="", tag="", voices=True,
                        names="", whisper="", whisper_max_repeat=-1, group=""):
    src = Path(audio_path).expanduser()
    if not src.exists():
        raise CoreError(f"Файл не найден: {src}")
    args = (["--speakers", str(num_speakers)] if num_speakers else []) + ([] if emotions else ["--no-emotions"])
    if vocab_path:
        args += ["--vocab", vocab_path]
    if group:
        args += ["--group", group]
    if tag:
        args += ["--tag", tag]
    if not voices:
        args.append("--no-voices")
    if whisper:
        if whisper not in ("all", "doubt", "off"):
            raise CoreError('whisper: "all", "doubt" или "off"')
        args += ["--whisper", whisper]
    if whisper_max_repeat >= 0:
        args += ["--whisper-max-repeat", str(int(whisper_max_repeat))]
    names_list = [n.strip() for n in names.replace(";", ",").split(",") if n.strip()]
    if names_list:
        args += ["--names", ",".join(names_list)]
    return {"job_id": launch("worker.py", args, src, "transcribe"), "file": str(src)}


def start_benchmark(audio_path, seconds=180, only="", chunk_max=20.0, min_similarity=None):
    src = Path(audio_path).expanduser()
    if not src.exists():
        raise CoreError(f"Файл не найден: {src}")
    args = ["--seconds", str(seconds), "--chunk-max", str(chunk_max)] + (["--only", only] if only else [])
    if min_similarity is not None and float(min_similarity) > 0:
        args += ["--min-similarity", str(min_similarity)]
    return {"job_id": launch("bench.py", args, src, "benchmark")}


def live(s, job_id):
    """Статус с вычисленным: сколько идёт задача и текущий этап, давно ли обновлялся файл статуса."""
    s = dict(s)
    now = time.time()
    t0, ts = s.pop("started_ts", None), s.pop("stage_started_ts", None)
    if s.get("state") in ACTIVE:
        if t0:
            s["elapsed_sec"] = round(now - t0)
        if ts:
            s["stage_elapsed_sec"] = round(now - ts)
        try:
            s["since_update_sec"] = round(now - (JOBS / f"{job_id}.json").stat().st_mtime)
        except OSError:
            pass
    return s


def stage_key(s):
    """Состояние и этап без счётчика в конце («поиск речи 12/321») — смена этапа, а не прогресс."""
    return s.get("state"), re.sub(r"\s*\d+/\d+\s*$", "", str(s.get("stage") or ""))


def status(job_id, wait_sec=0, sleep=time.sleep):
    """Статус задачи; при wait_sec>0 ждёт (до 50 с) смены этапа или конца. sleep можно подменить (асинхронно
    это делает server.py через poll())."""
    s = _need(job_id)
    k0 = stage_key(s)
    deadline = time.time() + min(max(int(wait_sec or 0), 0), 50)
    while s.get("state") in ACTIVE and time.time() < deadline:
        sleep(1)
        s2 = job_status(job_id)
        if s2 and s2.get("stage") != "обновляется":
            s = s2
        if stage_key(s) != k0:
            break
    return live(s, job_id)


def list_jobs(limit=10):
    rows = []
    for p in sorted(JOBS.glob("*.json"), reverse=True)[:limit]:
        s = job_status(p.stem) or {}
        rows.append({"job_id": p.stem, "kind": s.get("kind"), "state": s.get("state"),
                     "stage": s.get("stage"), "progress": s.get("progress"), "file": s.get("file")})
    return rows


def job_json(job_id):
    """(путь к json, путь к md) готовой расшифровки или (None, None)."""
    s = job_status(job_id)
    if not s or s.get("state") != "done" or "json" not in s:
        return None, None
    return s["json"], s.get("markdown")


# ---------- чтение стенограммы ----------
def _text_only(s, key):
    """Только реплики с пометками ⟦…⟧ у нерешённых мест; служебные разделы — одной строкой."""
    import transcript
    data = transcript.load(s["json"])
    marks = transcript.doubt_marks(data) if data.get("review") else {}
    if key == "markdown":
        full = transcript.render_md(data)
        body = transcript.render_md(data, text_only=True).rstrip()
        rest = [l[3:] for l in full.split("\n") if l.startswith("## ")]
        if rest:
            body += f"\n\n[без разделов: {', '.join(rest)} — {len(full) - len(body)} симв.; extra=True — полностью]"
        return body
    for k in ("review", "edits", "quality", "timing"):
        data.pop(k, None)
    segs = []
    for si, seg in enumerate(data.get("segments", [])):
        x = {k: v for k, v in seg.items() if k not in ("words", "text_raw")}
        if marks.get(si):
            x["doubts"] = sorted(set(marks[si].values()))
        segs.append(x)
    data["segments"] = segs
    return json.dumps(data, ensure_ascii=False)


def read_transcript(job_id, fmt="md", offset=0, max_chars=60000, extra=False):
    s = _need(job_id)
    if s.get("state") != "done":
        raise CoreError(f"Ещё не готово: {s.get('stage')} {s.get('progress', 0)}%")
    key = "report" if "report" in s else ("json" if fmt == "json" else "markdown")
    if key != "report" and not extra and s.get("json"):
        text = _text_only(s, key)
    else:
        text = Path(s[key]).read_text(encoding="utf-8")
    part = text[offset:offset + max_chars]
    if offset + max_chars < len(text):
        part += f"\n\n[...продолжение: offset={offset + max_chars}, всего символов {len(text)}]"
    return part


def _done_json(job_id):
    s = job_status(job_id)
    if not s or s.get("state") != "done" or "json" not in s:
        raise CoreError("Расшифровка не найдена или ещё не готова")
    return s


REASONS = ("asr", "conf", "voice", "overlap", "gender", "missed")


def review_items(job_id, offset=0, limit=40, reasons="", min_sec=0.0, brief=False, status="open"):
    """Сомнительные места готовой расшифровки (список для смыслового разбора). reasons — через запятую."""
    import transcript
    s = _done_json(job_id)
    want = [r.strip() for r in (reasons or "").split(",") if r.strip()]
    bad = [r for r in want if r not in REASONS]
    if bad:
        raise CoreError(f"неизвестные причины: {', '.join(bad)}; есть: {', '.join(REASONS)}")
    if status not in ("open", "human", "resolved", "all"):
        raise CoreError("status: open, human, resolved или all")
    limit = max(1, min(int(limit), 200 if brief else 60))
    return transcript.review_list(s["json"], max(0, int(offset)), limit, want, float(min_sec or 0), bool(brief), status)


def edit_span(job_id, t1, t2, text, speaker="", note="", by="claude"):
    """Правка отрезка t1–t2 (с): текст и говорящий; без пересборки карточек."""
    import transcript
    s = _done_json(job_id)
    if by not in ("claude", "human"):
        raise CoreError('by: "claude" или "human"')
    try:
        _, rep = transcript.edit_span(s["json"], s["markdown"], t1, t2, text, speaker or None, note or None, by)
    except ValueError as ex:
        raise CoreError(str(ex))
    return rep


def reopen_review(job_id, ids):
    """Снова открыть места разбора (через запятую или списком)."""
    import transcript
    s = _done_json(job_id)
    ids = [x.strip() for x in (ids.split(",") if isinstance(ids, str) else ids) if str(x).strip()]
    if not ids:
        raise CoreError("укажите id мест, например r57,r88")
    _, rep = transcript.reopen(s["json"], s["markdown"], ids)
    return rep


# ---------- настройки, группы, профили ----------
def settings_info():
    return {"data_dir": str(paths.DATA_DIR), "config_file": str(paths.CONFIG_FILE),
            "overrides": settings.load_overrides(), "effective": settings.load_config(),
            "groups": paths.list_groups()}


def voice_profiles():
    """Сводка по голосовым профилям групп; сами отпечатки не выводятся."""
    out = []
    for p in sorted(paths.GROUPS.glob("*/voices.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as ex:
            out.append({"group": p.parent.name, "error": str(ex)})
            continue
        conf = {}
        for pairs in (d.get("confusions") or {}).values():
            for k, v in pairs.items():
                conf[k.replace("|", " → ")] = conf.get(k.replace("|", " → "), 0) + v
        out.append({"group": p.parent.name, "revision": d.get("revision"),
                    "speakers": {n: {"records": len(s.get("parts", {})),
                                     "seconds": round(sum(pt.get("auto_sec", pt["sec"]) + pt.get("human_sec", 0)
                                                          for pt in s.get("parts", {}).values())),
                                     "human_seconds": round(sum(pt.get("human_sec", 0) for pt in s.get("parts", {}).values()), 1)}
                                 for n, s in d.get("speakers", {}).items()},
                    "corrected_pairs": conf,
                    "warnings": (d.get("warnings") or [])[-5:],
                    "quarantine": [{"name": n, "recording": Path(a).name, **{k: v for k, v in q.items() if k != "json"}}
                                   for a, qa in (d.get("quarantine") or {}).items() for n, q in qa.items()],
                    "identity": {Path(a).name: v for a, v in (d.get("identity") or {}).items()},
                    "sources": list(d.get("sources", {}))})
    return {"profiles": out, "groups": paths.list_groups(),
            "default_group": settings.load_config().get("default_group")}


# ---------- данные: выполнение (решение человека принимает слой управления) ----------
def do_migrate(dry_run=True):
    out = {"data_dir": str(paths.DATA_DIR), "old_data_dirs": [str(p) for p in paths.old_default_dirs()],
           "env": {k: os.environ.get(k) for k in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "RUS_TRANSCRIBER_DATA")},
           "home": str(Path.home())}
    try:
        out["result"] = migrate.run(dry=dry_run)
    except Exception as ex:
        out["error"] = f"{type(ex).__name__}: {ex}"
    try:
        out["log_tail"] = migrate.LOG.read_text(encoding="utf-8").splitlines()[-12:]
    except OSError:
        out["log_tail"] = []
    return out


def _data(fn, *a):
    try:
        return fn(*a)
    except RuntimeError as ex:
        raise CoreError(str(ex)) from ex


def do_export(group="", dest_dir="", with_jobs=False):
    return _data(data_io.export_data, dest_dir or None, group or None, with_jobs)


def do_import(zip_path, mode="merge", apply=False):
    return _data(data_io.import_data, zip_path, mode, apply)


def do_delete_group(group, confirm="", permanent=False):
    return _data(data_io.delete_group, group, confirm, permanent)


def do_cleanup(days=30, apply=False):
    return data_io.cleanup_jobs(days, apply)


# ---------- описание действия для человека (без выполнения) ----------
def describe_export(group="", dest_dir="", with_jobs=False):
    groups = [group] if group else paths.list_groups()
    return ["Выгрузка архива с ГОЛОСОВЫМИ ОТПЕЧАТКАМИ (биометрия) и словарями.",
            f"Группы: {', '.join(groups) or '—'}",
            f"Статусы задач: {'да' if with_jobs else 'нет'}",
            f"Куда: {dest_dir or str(paths.DATA_DIR / 'exports')}"]


def describe_import(zip_path, mode="merge"):
    rep = do_import(zip_path, mode, False)            # только отчёт, ничего не меняется
    lines = [f"Загрузка архива {Path(zip_path).name}, режим: {mode}."]
    for g, r in rep.get("groups", {}).items():
        lines.append(f"Группа «{g}»: {r.get('action')}; словарь: {r.get('vocab', '—')}; голоса: {r.get('voices', '—')}")
    if mode == "replace":
        lines.append("Прежняя версия заменяемой группы уйдёт в резервную копию.")
    return lines


def describe_delete(group, permanent=False):
    g = paths.safe_group(group)
    gd = paths.group_dir(g)
    if not gd.is_dir():
        raise CoreError(f"Группа «{group}» не найдена (есть: {', '.join(paths.list_groups())})")
    n = sum(1 for _ in gd.rglob("*") if _.is_file())
    return [f"Удаление группы «{g}» (файлов: {n}).",
            "НАВСЕГДА: восстановить будет нельзя." if permanent else "Папка будет перенесена в _deleted (можно вернуть)."]


def describe_cleanup(days=30):
    r = do_cleanup(days, False)
    return [f"Удаление статусов и логов задач старше {days} дн.: файлов {r['files']}, {r['bytes'] // 1024} КБ."]


def describe_migrate():
    r = do_migrate(True)
    res = r.get("result")
    return ["Перенос данных в папку данных из прежних мест.", f"Папка данных: {r['data_dir']}",
            f"План: {json.dumps(res, ensure_ascii=False)[:300] if res is not None else r.get('error', '—')}"]
