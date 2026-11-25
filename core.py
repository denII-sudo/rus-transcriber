"""Ядро rus-transcriber: задачи (запуск, статус, отмена), чтение стенограмм, настройки, данные.

Здесь нет ничего, что знало бы про MCP, Claude или любой другой клиент. Слои управления — server.py (MCP) и
rt.py (командная строка) — только вызывают эти функции и показывают результат. Любой третий слой (HTTP, окно,
другой протокол) строится так же.

Соглашения: функции возвращают обычные структуры (dict, list, str); понятная пользователю ошибка — CoreError.
Опасные действия (выгрузка, загрузка, удаление, очистка, перенос) здесь только исполняются; решение человека
получает слой управления (confirm.py для MCP, ввод в терминале для rt.py).
"""
import bisect
import collections
import difflib
import json
import hashlib
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
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


def process_start_time(pid):
    """Метка запуска процесса (не зависит от часов), чтобы отличить его от процесса с тем же PID позже.

    Windows: время создания из GetProcessTimes; Linux: starttime из /proc. None — узнать нельзя.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
            if not k.GetProcessTimes(h, ctypes.byref(created), ctypes.byref(exited),
                                     ctypes.byref(kernel), ctypes.byref(user)):
                return None
            return (created.dwHighDateTime << 32) | created.dwLowDateTime
        finally:
            k.CloseHandle(h)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
        return int(stat[stat.rindex(")") + 2:].split()[19])   # поле 22: starttime
    except (OSError, ValueError, IndexError):
        return None


def _pid_record(job_id):
    """(pid, метка запуска или None) из PID-файла задачи; None, если файла нет."""
    if not _valid_job_id(job_id):
        return None
    try:
        parts = _job_path(job_id, ".pid").read_text(encoding="utf-8").split()
        pid = int(parts[0])
        start = int(parts[1]) if len(parts) > 1 else None
    except (OSError, ValueError, IndexError):
        return None
    return pid, start


def pid_of(job_id):
    rec = _pid_record(job_id)
    return rec[0] if rec else None


def job_process_alive(job_id):
    """True — процесс задачи жив и это именно он; False — его нет (или PID занят другим процессом);
    None — процесс с этим PID жив, но принадлежность задаче подтвердить нельзя (старый PID-файл)."""
    rec = _pid_record(job_id)
    if not rec:
        return False
    pid, start = rec
    if not alive(pid):
        return False
    if start is None:
        return None
    current = process_start_time(pid)
    if current is None:
        return None
    return current == start


def _valid_job_id(job_id):
    return isinstance(job_id, str) and bool(
        re.fullmatch(r"(?:\d{8}-\d{6}-[a-z][0-9a-f]{5}|workflow-[0-9a-f]{32})", job_id))


def _job_path(job_id, suffix):
    if not _valid_job_id(job_id):
        raise CoreError("Недопустимый идентификатор задачи")
    root = JOBS.resolve()
    path = (JOBS / f"{job_id}{suffix}").resolve()
    if not path.is_relative_to(root):
        raise CoreError("Путь задачи находится вне каталога jobs")
    return path


def job_status(job_id):
    """Статус задачи (dict) или None, если такой задачи нет. Если процесс пропал без итога — state=error."""
    if not _valid_job_id(job_id):
        return None
    p = _job_path(job_id, ".json")
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
        if pid_of(job_id) and job_process_alive(job_id) is False:
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


def launch(script, args, src, kind, *, initial_state=None, job_id=None):
    """Запускает фоновый процесс-задачу (worker.py и др.), возвращает job_id."""
    job_id = job_id or time.strftime("%Y%m%d-%H%M%S-") + kind[0] + uuid.uuid4().hex[:5]
    status = _job_path(job_id, ".json")
    with status.open("x", encoding="utf-8") as stream:
        json.dump({**(initial_state or {}), "state": "queued", "kind": kind, "file": str(src)},
                  stream, ensure_ascii=False)
    cmd = [sys.executable, str(HERE / script), str(src), "--status", str(status), *args]
    log = open(JOBS / f"{job_id}.log", "w", encoding="utf-8")
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            cwd=str(HERE), creationflags=flags, env=env)
    import jobslots
    pid_path = _job_path(job_id, ".pid")
    try:
        with jobslots.file_lock(_job_path(job_id, ".status.lock")):
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=JOBS,
                                             prefix=job_id + ".", suffix=".pid.tmp", delete=False) as f:
                start = process_start_time(proc.pid)
                f.write(str(proc.pid) if start is None else f"{proc.pid} {start}")
                tmp = Path(f.name)
            try:
                os.replace(tmp, pid_path)
            finally:
                tmp.unlink(missing_ok=True)
            current = job_status(job_id) or {}
            if current.get("state") in FINISHED:
                pid_path.unlink(missing_ok=True)
    except OSError as ex:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        pid_path.unlink(missing_ok=True)
        raise CoreError(f"Не удалось сохранить PID задачи; worker мог остаться запущенным: {ex}") from ex
    return job_id


def cancel_job(job_id):
    """Останавливает идущую или ждущую задачу вместе с дочерними процессами; готовые результаты не трогает."""
    s = _need(job_id)
    if s.get("state") in FINISHED:
        return {"job_id": job_id, "state": s.get("state"), "note": "задача уже не идёт"}
    pid = pid_of(job_id)
    killed = False
    owned = job_process_alive(job_id) if pid else False
    if owned is None:
        raise CoreError("Не удалось подтвердить, что процесс с PID из файла задачи — это её worker; "
                        "процесс не завершён, статус задачи не изменён")
    if owned is False:
        pid = None          # процесса задачи нет; PID мог достаться другой программе — её не трогаем
    if pid:
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
    if pid and alive(pid):
        raise CoreError("Процесс не остановлен; статус задачи не изменён")
    import jobslots
    status_path = _job_path(job_id, ".json")
    with jobslots.file_lock(_job_path(job_id, ".status.lock")):
        current = job_status(job_id) or s
        if current.get("state") in FINISHED:
            return {"job_id": job_id, "state": current["state"], "note": "задача завершилась во время отмены"}
        st = {**current, "state": "cancelled", "stage": "отменено",
              "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=JOBS, delete=False) as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
            tmp = Path(f.name)
        try:
            os.replace(tmp, status_path)
        finally:
            tmp.unlink(missing_ok=True)
    return {"job_id": job_id, "state": "cancelled", "was": s.get("stage"), "process_killed": killed}


def start_transcription(audio_path, num_speakers=0, emotions=True, vocab_path="", tag="", voices=True,
                        names="", whisper="", whisper_max_repeat=-1, group="", *,
                        workflow_job_id=None, workflow_context=None, workflow_request_id=None):
    src = Path(audio_path).expanduser()
    if not src.exists():
        raise CoreError(f"Файл не найден: {src}")
    args = (["--speakers", str(num_speakers)] if num_speakers else []) + ([] if emotions else ["--no-emotions"])
    backend = settings.recording_backend()
    initial_state = None
    if backend == "sqlite":
        import recordings
        import storage
        if not recordings.SQLITE_RUNTIME_READY:
            raise CoreError("SQLite backend ещё не доступен: не все операции записи переведены")
        try:
            store = recordings.open_store()
            try:
                store.current_identity()
                initial_state = {"recording_context": workflow_context or recordings.operation_context(store, "model")}
                if workflow_request_id is not None:
                    initial_state["workflow_request_id"] = workflow_request_id
            finally:
                store.close()
        except (OSError, storage.StorageError) as ex:
            raise CoreError("SQLite-хранилище не инициализировано или недоступно") from ex
        args += ["--storage-backend", "sqlite"]
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
    if initial_state is not None:
        if workflow_job_id is None:
            job_id = launch("worker.py", args, src, "transcribe", initial_state=initial_state)
        else:
            job_id = launch("worker.py", args, src, "transcribe", initial_state=initial_state,
                            job_id=workflow_job_id)
    else:
        job_id = launch("worker.py", args, src, "transcribe")
    return {"job_id": job_id, "file": str(src)}


def start_benchmark(audio_path, seconds=180, only="", chunk_max=20.0, min_similarity=None):
    src = Path(audio_path).expanduser()
    if not src.exists():
        raise CoreError(f"Файл не найден: {src}")
    args = ["--seconds", str(seconds), "--chunk-max", str(chunk_max)] + (["--only", only] if only else [])
    if min_similarity is not None and float(min_similarity) > 0:
        args += ["--min-similarity", str(min_similarity)]
    return {"job_id": launch("bench.py", args, src, "benchmark")}


def start_workflow(audio_path, policy=None):
    """Создать сохранённый запрос и один раз запустить SQLite-worker без обучения голосов."""
    import recordings
    import workflow
    if settings.recording_backend() != "sqlite":
        raise CoreError("Для сохранённого процесса требуется явно выбранный SQLite backend")
    store = recordings.open_store()
    try:
        request_id = workflow.create(store, audio_path, policy)
        job_id = workflow.claim_start(store, request_id)
        request = workflow.get(store, request_id)
        try:
            start_transcription(request["source"], voices=False, workflow_job_id=job_id,
                                workflow_context=request["context"], workflow_request_id=request_id)
        except Exception:
            workflow.mark_launch_interrupted(store, request_id)
            return {"request_id": request_id, "job_id": job_id, "state": "interrupted",
                    "note": "Исход запуска неизвестен; автоматический повтор отключён."}
        return {"request_id": request_id, "job_id": job_id}
    finally:
        store.close()


def workflow_status(request_id):
    import documents
    import product_policy
    import recordings
    import storage
    import transcript
    import workflow
    store = recordings.open_store()
    try:
        request = workflow.get(store, request_id)
        if request["state"] in ("starting", "running", "interrupted") and request["job_id"]:
            request = workflow.observe(store, request_id, job_status(request["job_id"]))
        recording_id = request["recording_id"]
        sync, unresolved, incomplete, has_conspect, unnamed = "not_started", 0, False, False, 1
        source_revision, review_revision, human_verification = None, None, "none"
        if recording_id:
            current = store.get_current(recording_id)
            source_revision, review_revision = current["transcript"]["revision"], current["review"]["revision"]
            # Отметка снимается при правке текста или сомнений, но не от одного номера ревизии разбора.
            if workflow.verification_is_current(store, request_id, recording_id, current):
                human_verification = "confirmed"
            review = current["review"]["payload"]
            unresolved = sum(1 for item in (review.get("review") or []) if transcript.place_unresolved(item))
            quality = current["transcript"]["payload"].get("quality") or {}
            diagnostics = quality.get("asr_diagnostics")
            incomplete = bool(quality.get("asr_incomplete_chunks")) or not isinstance(diagnostics, dict) or not diagnostics
            if isinstance(diagnostics, dict):
                incomplete = incomplete or any(
                    chunk.get("state") in ("error", "empty_unknown") or chunk.get("filtered_all")
                    for stage in diagnostics.values() if isinstance(stage, dict)
                    for chunk in (stage.get("chunks") or []) if isinstance(chunk, dict))
            transcript_snapshot = current["transcript"]["payload"]
            segments = transcript_snapshot.get("segments") or []
            names = transcript_snapshot.get("speaker_names") or {}
            labels = {segment.get("label", segment.get("speaker")) for segment in segments}
            unnamed = (sum(1 for label in labels if not label or not names.get(label)
                           or names[label] == label) if labels and transcript_snapshot.get("names_confirmed") else 1)
            row = store.db.execute("SELECT state FROM document_outbox WHERE recording_id=? "
                                   "ORDER BY document_revision DESC LIMIT 1", (recording_id,)).fetchone()
            sync = row["state"] if row else "not_started"
            if sync == "synced":
                binding = recordings.document_binding(store, recording_id)
                verified = documents.verify_files(store, binding["root_id"], binding["root"])
                entries = [entry for entry in verified["entries"]
                           if entry["recording_id"] == recording_id and entry["kind"] in ("md", "json", "manifest")]
                if len(entries) != 3 or any(entry["status"] != "ok" for entry in entries):
                    sync = "error"
            if request["model_result"] is not None and request["model_task"] is not None:
                conspect_id = str(uuid.uuid5(uuid.UUID(recording_id), "conspect:workflow-" + request_id))
                row = store.db.execute(
                    "SELECT c.current_revision, r.source_revision, o.state FROM conspects c "
                    "JOIN conspect_revisions r ON r.conspect_id=c.conspect_id AND r.revision=c.current_revision "
                    "JOIN conspect_outbox o ON o.conspect_id=r.conspect_id AND o.revision=r.revision "
                    "WHERE c.conspect_id=?", (conspect_id,)).fetchone()
                # Свёртка актуальна, пока совпадает содержание её задания (слова, говорящие, сомнения,
                # цель), а не номера ревизий: показ карточки или пакет её устаревшей не делают.
                has_conspect = bool(row and workflow.content_matches(request, request["model_task"], current))
                if row and row["state"] != "synced":
                    sync = row["state"]
                elif row:
                    locations = store.db.execute(
                        "SELECT f.relative_path, f.confirmed_hash, r.path FROM conspect_file_locations f "
                        "JOIN allowed_roots r USING(root_id) WHERE f.conspect_id=?", (conspect_id,)).fetchall()
                    try:
                        if len(locations) != 2:
                            sync = "error"
                        else:
                            for location in locations:
                                target, exists = documents._read_only_target(
                                    Path(location["path"]), location["relative_path"])
                                if not exists or documents._file_digest(target)[0] != location["confirmed_hash"]:
                                    sync = "error"
                    except (OSError, ValueError):
                        sync = "error"
        execution = {"starting": "running", "processed": "completed",
                     "model_received": "waiting_model"}.get(request["state"], request["state"])
        if execution == "processed":
            execution = "completed"
        if request["policy"]["require_conspect"] and not has_conspect and execution == "completed":
            execution = "waiting_model"
        readiness = product_policy.evaluate(request["policy"], execution=execution, document_sync=sync,
                                            unresolved=unresolved, unnamed_speakers=unnamed,
                                            incomplete=incomplete, conspect=has_conspect,
                                            human_verification=human_verification)
        assessment = {"policy_revision": request["policy_revision"], "recording_id": recording_id,
                      "source_revision": source_revision, "review_revision": review_revision,
                      "readiness": readiness}
        with store._transaction():
            store.current_identity()
            policy_head = workflow.policy_head(store, request_id)
            if policy_head is None or policy_head["policy_revision"] != request["policy_revision"]:
                raise storage.Conflict("политика изменилась во время оценки")
            if recording_id:
                head = recordings.recording_head(store, recording_id)
                if (head is None or head["source_revision"] != source_revision
                        or head["review_revision"] != review_revision):
                    raise storage.Conflict("состояние записи изменилось во время оценки")
            store.db.execute("INSERT OR IGNORE INTO workflow_assessments VALUES (?, ?, ?, ?)",
                             (request_id, storage._hash(assessment), storage._json(assessment), storage._now()))
        public_request = dict(request)
        public_request["model_task"] = bool(request["model_task"])
        public_request["model_result"] = bool(request["model_result"])
        public_request["model_failure"] = None
        if request["model_task"] and request["state"] == "failed":
            failure = store.db.execute(
                "SELECT s.result_json FROM workflow_steps s JOIN workflow_runs r USING(run_id) "
                "WHERE r.request_id=? AND r.kind='model' AND s.state='failed' "
                "ORDER BY s.updated_at DESC LIMIT 1", (request_id,)).fetchone()
            if failure and failure["result_json"]:
                details = storage._loads(failure["result_json"])
                if isinstance(details, dict):
                    public_request["model_failure"] = {
                        "code": details.get("error_code", "model_failed"),
                        "message": details.get("message", "модельный шаг завершился с ошибкой"),
                    }
        return {"request": public_request, "readiness": readiness}
    finally:
        store.close()


def workflow_model_task(request_id, offset=0, limit=1000, *, restart=False):
    """Страница задания модели. restart — явное «начать заново», только с первой страницы.

    Задание версии 4 выдаётся в том виде, какой видит модель (пункты без ID слов), страницами
    по пунктам: offset и limit считают пункты, total_points — их число.
    """
    import model_tasks
    import recordings
    import workflow
    if (type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 2000):
        raise ValueError("неверные границы страницы")
    if restart and offset != 0:
        raise ValueError("начать заново можно только с первой страницы (offset=0)")
    store = recordings.open_store()
    try:
        task = workflow.prepare_model_task(store, request_id, restart=restart)
        if model_tasks.version(task) == 4:
            view = model_tasks.model_view(task)
            total = len(view["points"])
            return {**view, "points": view["points"][offset:offset + limit], "offset": offset,
                    "total_points": total, "total_words": len(task["words"]),
                    "next_offset": offset + limit if offset + limit < total else None}
        total = len(task["words"])
        page = {**task, "words": task["words"][offset:offset + limit], "offset": offset,
                "total_words": total, "next_offset": offset + limit if offset + limit < total else None}
        if "turns" in task:      # на странице — только реплики её слов
            used = {word["turn"] for word in page["words"]}
            page["turns"] = [turn for turn in task["turns"] if turn["turn"] in used]
        return page
    finally:
        store.close()


def workflow_model_result(request_id, result):
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return workflow.accept_model_result(store, request_id, result)
    finally:
        store.close()


def workflow_model_part(request_id, part_index):
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        part = workflow.model_part(store, request_id, part_index)
        import model_tasks
        if model_tasks.version(part["task"]) == 4:      # модель и ИИ-агент видят пункты, без ID слов
            part = {**part, "task": model_tasks.model_view(part["task"])}
        return part
    finally:
        store.close()


def workflow_model_part_result(request_id, part_index, result):
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return workflow.accept_model_part(store, request_id, part_index, result)
    finally:
        store.close()


def workflow_execute_model(request_id, *, retry_failed=False, restart=False):
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return workflow.execute_model(store, request_id, retry_failed=retry_failed, restart=restart)
    finally:
        store.close()


def workflow_change_policy(request_id, policy, expected_revision):
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return {"policy_revision": workflow.change_policy(store, request_id, policy, expected_revision)}
    finally:
        store.close()


def workflow_questions(request_id):
    """Показать ограниченный пакет добровольных уточнений без автоматического ответа."""
    import recordings
    import storage
    import transcript
    import workflow
    store = recordings.open_store()
    try:
        request = workflow.get(store, request_id)
        budget = request["policy"]["question_budget"]
        if not request["recording_id"] or not budget:
            return {"questions": [], "budget": budget, "recording_id": request["recording_id"]}
        current = store.get_current(request["recording_id"])
        items = [item for item in (current["review"]["payload"].get("review") or [])
                 if transcript.place_unresolved(item) and isinstance(item.get("id"), str)]
        with store._transaction():
            store.current_identity()
            head = recordings.recording_head(store, request["recording_id"])
            policy_head = workflow.policy_head(store, request_id)
            if (head["source_revision"] != current["transcript"]["revision"]
                    or head["review_revision"] != current["review"]["revision"]
                    or policy_head["policy_revision"] != request["policy_revision"]):
                raise storage.Conflict("запись или политика изменились")
            slots = store.db.execute("SELECT slot, question_id FROM workflow_question_slots "
                                     "WHERE request_id=? ORDER BY slot", (request_id,)).fetchall()
            seen = {slot["question_id"] for slot in slots}
            next_slot = max((slot["slot"] for slot in slots), default=0)
            for item in items:
                if next_slot >= budget:
                    break
                if item["id"] not in seen:
                    next_slot += 1
                    store.db.execute("INSERT INTO workflow_question_slots VALUES (?, ?, ?, ?)",
                                     (request_id, next_slot, item["id"], storage._now()))
                    seen.add(item["id"])
            slots = store.db.execute("SELECT question_id FROM workflow_question_slots "
                                     "WHERE request_id=? AND slot<=? ORDER BY slot",
                                     (request_id, budget)).fetchall()
        by_id = {item["id"]: item for item in items}
        selected = [by_id[slot["question_id"]] for slot in slots if slot["question_id"] in by_id]
        questions = [{"id": item.get("id"), "question": "Проверьте текст и говорящего в этом месте.",
                      "text": item.get("text"), "t1": item.get("t1"), "t2": item.get("t2"),
                      "word_ids": item.get("word_ids") or []} for item in selected]
        return {"recording_id": request["recording_id"], "source_revision": current["transcript"]["revision"],
                "review_revision": current["review"]["revision"], "budget": budget,
                "questions": questions, "remaining": max(0, len(items) - len(questions)),
                "note": "Ответы человека обрабатываются отдельно через существующий интерфейс разбора."}
    finally:
        store.close()


def workflow_verification_snapshot(request_id):
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return workflow.verification_snapshot(store, request_id)
    finally:
        store.close()


def workflow_confirm_verification(snapshot):
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return workflow.confirm_verification(store, snapshot)
    finally:
        store.close()


def workflow_abandon_snapshot(request_id):
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return workflow.abandon_snapshot(store, request_id)
    finally:
        store.close()


def workflow_abandon(snapshot):
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return workflow.abandon(store, snapshot)
    finally:
        store.close()


# ---------- экран: перенесён в core_screen.py (имена прежние, переотдача — в конце файла) ----------


# ---------- конфликты, резерв базы, результаты записи, подтверждение ответов: перенесены в core_conflicts.py (имена прежние, переотдача — в конце файла) ----------


# ---------- статус задач (оставлен в core.py: тесты подменяют job_json и job_status через core) ----------
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
    job_files = [p for p in JOBS.glob("*.json") if not p.name.endswith(".voices_debug.json")]
    for p in sorted(job_files, reverse=True)[:limit]:
        s = job_status(p.stem) or {}
        rows.append({"job_id": p.stem, "kind": s.get("kind"), "state": s.get("state"),
                     "stage": s.get("stage"), "progress": s.get("progress"), "file": s.get("file")})
    return rows


def job_json(job_id):
    """(путь к json, путь к md) готовой расшифровки или (None, None)."""
    if isinstance(job_id, str) and job_id.startswith("recording:"):
        s = _recording_reference(job_id)
        return s["json"], s["markdown"]
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
    quality = data.get("quality") or {}
    summary = {k: quality[k] for k in ("asr_incomplete_chunks", "vote3_skipped", "vote3_partial",
                                       "asr2_skipped", "asr3_skipped")
               if quality.get(k)}
    if summary:
        data["asr_summary"] = summary
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
    s = (_recording_reference(job_id) if isinstance(job_id, str) and job_id.startswith("recording:")
         else _need(job_id))
    if s.get("state") != "done":
        raise CoreError(f"Ещё не готово: {s.get('stage')} {s.get('progress', 0)}%")
    key = "report" if "report" in s else ("json" if fmt == "json" else "markdown")
    if key != "report" and not extra and s.get("json"):
        text = _text_only(s, key)
    else:
        if s.get("document_sync", "synced") != "synced":
            raise CoreError("Документы записи ещё не синхронизированы")
        if isinstance(job_id, str) and job_id.startswith("recording:"):
            import recordings
            store = recordings.open_store()
            try:
                binding = recordings.document_binding(store, s["recording_id"])
                doc = store.get_document(s["recording_id"], binding["document_revision"])
                text = doc["json" if key == "json" else "md"].decode("utf-8")
            finally:
                store.close()
        else:
            text = Path(s[key]).read_text(encoding="utf-8")
    part = text[offset:offset + max_chars]
    if offset + max_chars < len(text):
        part += f"\n\n[...продолжение: offset={offset + max_chars}, всего символов {len(text)}]"
    return part


def _done_json(job_id):
    s = (_recording_reference(job_id) if isinstance(job_id, str) and job_id.startswith("recording:")
         else job_status(job_id))
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


def conversation_hotspots(job_id, limit=20, min_score=0.0):
    """Горячие места разговора готовой расшифровки — только по запросу (hotspots.py).

    Оценка считается при чтении по текущим словам стенограммы и слою quality["interaction"];
    в обоих хранилищах данные читаются через transcript.load. Если слоя нет — available=False
    с причиной. limit — сколько мест показать (самые выраженные, в порядке времени); 0 — все.
    min_score — нижняя граница оценки (0 — стартовый порог hotspots.HOT_SCORE).
    """
    import hotspots
    import storage
    import transcript
    s = _done_json(job_id)
    try:
        limit, min_score = int(limit), float(min_score or 0)
    except (TypeError, ValueError) as ex:
        raise CoreError("limit — целое число, min_score — число от 0 до 1") from ex
    if limit < 0 or not 0 <= min_score <= 1:
        raise CoreError("limit — не меньше 0, min_score — от 0 до 1")
    try:
        data = transcript.load(s["json"])
    except (OSError, ValueError, KeyError, storage.StorageError) as ex:
        raise CoreError(f"Стенограмма недоступна: {type(ex).__name__}") from ex
    threshold = min_score or hotspots.HOT_SCORE
    try:
        found = hotspots.find_hotspots(data, min_score=threshold)
    except hotspots.NoData as ex:
        return {"available": False, "reason_code": ex.code, "reason": f"Нет данных: {ex}", "hotspots": []}
    shown = found
    if limit and len(found) > limit:
        shown = sorted(sorted(found, key=lambda h: -h["score"])[:limit], key=lambda h: h["t1"])
    return {"available": True, "total": len(found), "min_score": threshold,
            "hotspots": [{**h, "time": f"{transcript.hms(h['t1'])}–{transcript.hms(h['t2'])}"} for h in shown],
            "note": hotspots.NOTE}


def build_review_pack(job_id, items=None):
    """Подготовка карточек без зависимости от MCP или конкретной модели."""
    import review_pack as P
    import transcript
    jp = _done_json(job_id)["json"]
    if items is None:
        data = transcript.load(jp)
        items = [{"id": it["id"], "question": "Проверьте текст и говорящего. При необходимости введите свой ответ.",
                  "options": [{"label": "Оставить как есть", "decision": {}}]}
                 for it in data.get("review", []) if transcript.place_unresolved(it)]
    try:
        P.check_rebuild(jp)             # быстрый отказ до резки клипов; build повторяет проверку под замком
        return P.build(jp, items)
    except P.RebuildRefused as ex:
        raise CoreError(str(ex)) from ex


def start_review_page(job_id, port=8765):
    import review_pack as P
    job = _done_json(job_id)
    jp = job["json"]
    if not P.manifest(jp):
        raise CoreError("Сначала подготовьте карточки: review-build / build_review_pack")
    if not 1 <= int(port) <= 65516:
        raise CoreError("port: от 1 до 65516")
    args = ["--port", str(int(port))]
    if job.get("markdown"):
        args += ["--markdown", job["markdown"]]
    jid = launch("review_page.py", args, jp, "page")
    for _ in range(40):
        s = job_status(jid) or {}
        if s.get("url"):
            return {"job_id": jid, "url": s["url"], "hint": "Откройте адрес в браузере на этом компьютере"}
        if s.get("state") == "error":
            raise CoreError(s.get("error") or "Не удалось запустить страницу")
        time.sleep(0.25)
    raise CoreError(f"Страница ещё запускается; проверьте status {jid}")


def _merge_free_reports(parts):
    """Свести отчёты по свободным ответам одного вызова, сохранив причины по каждой карточке."""
    if not parts:
        return {}
    merged = dict(parts[0])
    for key in ("successful_ids", "not_found", "cards", "conflicts", "left_to_human",
                "proposed_delete", "vocab_suggestions", "hints"):
        merged[key] = [x for part in parts for x in part.get(key) or []]
    for key in ("closed_via", "failed", "conspects"):
        merged[key] = {k: v for part in parts for k, v in (part.get(key) or {}).items()}
    merged["counts"] = {k: sum((part.get("counts") or {}).get(k, 0) for part in parts)
                        for k in ("relabel", "replace", "insert")}
    return merged


def describe_conspect_confirmation(job_id, name):
    """Снимок для подтверждения: хэши обоих файлов связывают разрешение с их содержимым."""
    import hashlib
    import conspect
    jp = Path(_done_json(job_id)["json"])
    import recordings
    store, binding = recordings.open_for_document(jp)
    if store is not None:
        try:
            saved = recordings.conspect_for_document(jp, name)
            if saved is None:
                raise CoreError("конспект не найден")
            document = store.get_document(
                binding["recording_id"], binding["document_revision"])
            current = store.get_current(binding["recording_id"])
            obj = saved["payload"]
            if obj.get("recording_id") != binding["recording_id"]:
                raise CoreError("конспект относится к другой записи")
            return {"job_id": job_id, "name": name, "recording_id": binding["recording_id"],
                    "source_revision": current["transcript"]["revision"],
                    "source_hash": hashlib.sha256(document["json"]).hexdigest(),
                    "conspect_hash": saved["json_hash"]}
        finally:
            store.close()
    cj, _ = conspect.path_for(jp, name)
    try:
        source_bytes, conspect_bytes = jp.read_bytes(), cj.read_bytes()
        data, obj = json.loads(source_bytes), json.loads(conspect_bytes)
        if not obj.get("recording_id") or obj.get("recording_id") != data.get("recording_id"):
            raise CoreError("конспект относится к другой или неизвестной записи")
        if data.get("source_revision") is None:
            raise CoreError("ревизия источника неизвестна")
        return {"job_id": job_id, "name": name, "recording_id": data["recording_id"],
                "source_revision": data["source_revision"],
                "source_hash": hashlib.sha256(source_bytes).hexdigest(),
                "conspect_hash": hashlib.sha256(conspect_bytes).hexdigest()}
    except (OSError, ValueError, KeyError) as ex:
        raise CoreError("не удалось прочитать стенограмму или конспект") from ex


def preview_conspect_candidates(job_id, text):
    import conspect
    import transcript
    jp = Path(_done_json(job_id)["json"])
    try:
        data = transcript.load(jp)
        _, report = conspect.link(text, data)
        return report
    except (OSError, ValueError) as ex:
        raise CoreError("не удалось подготовить кандидатов цитат") from ex


MAX_CONSPECT_SELECTIONS = 3


def selected_conspect_confirmation_lines(job_id, name, snapshot, report, selections):
    """Полный ограниченный текст для человека; обрезка важной информации запрещена."""
    import math

    def visible(value, limit):
        if not isinstance(value, str) or not value or len(value) > limit or not value.isprintable():
            raise CoreError("выбор нельзя полностью и безопасно показать человеку")
        return value

    def time_text(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 864000:
            raise CoreError("недопустимое время выбранного места")
        return str(value)

    visible(job_id, 80)
    visible(name, 80)
    revision = snapshot.get("source_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or not 0 <= revision <= 2**53:
        raise CoreError("недопустимая ревизия источника")
    lines = [f"Новый конспект: {name}", f"Задача: {job_id}, ревизия: {revision}"]
    shown = set()
    for entry in report.get("candidates") or []:
        key = str(entry["quote_index"])
        selection = selections.get(key)
        if selection is None:
            continue
        try:
            option = next(x for x in entry["candidates"] if x["word_ids"] == selection["word_ids"])
            quote = visible(entry["quote"], 120)
            speaker = visible(option["speaker"], 50)
            start, end = time_text(option["t1"]), time_text(option["t2"])
            if len(key) > 8 or not key.isascii() or not key.isdecimal():
                raise CoreError("недопустимый номер цитаты")
            label = (speaker, start, end)
            if entry.get("truncated") or sum(
                    (visible(x["speaker"], 50), time_text(x["t1"]), time_text(x["t2"])) == label
                    for x in entry["candidates"]) != 1:
                raise CoreError("кандидатов нельзя однозначно различить по отображаемым данным")
            lines.append(f"Цитата {key} «{quote}»: {speaker}, {start}–{end} с")
            shown.add(key)
        except (KeyError, TypeError, StopIteration) as ex:
            raise CoreError("выбранный кандидат отсутствует") from ex
    if shown != set(selections):
        raise CoreError("не все выбранные цитаты можно показать человеку")
    lines.append("Создать новый конспект с этими выбранными местами?")
    if any(len(line) > 220 for line in lines) or len("\n".join(lines)) > 1200:
        raise CoreError("список выбора слишком длинный для подтверждения")
    return lines


def prepare_selected_conspect(job_id, name, text, selections):
    import hashlib
    import conspect
    jp = Path(_done_json(job_id)["json"])
    cj, cm = conspect.path_for(jp, name)
    import recordings
    existing = recordings.conspect_for_document(jp, name)
    if existing is not None or cj.exists() or cm.exists():
        raise CoreError("конспект с таким названием уже существует; выберите другое название")
    if not isinstance(selections, dict) or not 1 <= len(selections) <= MAX_CONSPECT_SELECTIONS:
        raise CoreError(f"за одно подтверждение выберите от 1 до {MAX_CONSPECT_SELECTIONS} цитат")
    try:
        import transcript
        data = transcript.load(jp)
        source = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        obj, report = conspect.link(text, data, selections=selections)
        if len(selections) > report["linked"]:
            raise ValueError("не все выбранные цитаты связаны")
        payload = json.dumps({"text": text, "selections": selections}, ensure_ascii=False, sort_keys=True).encode("utf-8")
        snapshot = {"job_id": job_id, "name": name, "source_hash": hashlib.sha256(source).hexdigest(),
                    "request_hash": hashlib.sha256(payload).hexdigest(), "recording_id": data.get("recording_id"),
                    "source_revision": data.get("source_revision")}
        selected_conspect_confirmation_lines(job_id, name, snapshot, report, selections)
        return snapshot, report
    except (OSError, ValueError, TypeError, StopIteration, KeyError) as ex:
        raise CoreError("выбранные цитаты устарели или недействительны") from ex


def save_selected_conspect(snapshot, text, selections):
    import conspect
    jp = Path(_done_json(snapshot["job_id"])["json"])
    with conspect.write_lock(jp):
        conspect.recover_pending(jp)
        return _save_selected_conspect_locked(snapshot, text, selections)


def _save_selected_conspect_locked(snapshot, text, selections):
    import conspect
    import transcript
    current, _ = prepare_selected_conspect(snapshot["job_id"], snapshot["name"], text, selections)
    if current != snapshot:
        raise CoreError("стенограмма или выбор изменились после подтверждения")
    jp = Path(_done_json(snapshot["job_id"])["json"])
    data = transcript.load(jp)
    obj, report = conspect.link(text, data, selections=selections)
    obj.update(name=snapshot["name"], job_json=str(jp))
    md, render_report = conspect.render(obj, data)
    if prepare_selected_conspect(snapshot["job_id"], snapshot["name"], text, selections)[0] != snapshot:
        raise CoreError("стенограмма или выбор изменились после подтверждения")
    cj, cm = conspect.path_for(jp, snapshot["name"])
    try:
        conspect.write_files(jp, [(cj, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")),
                                  (cm, md.encode("utf-8"))], create=True)
    except OSError as ex:
        raise CoreError("не удалось полностью сохранить конспект; проверьте файлы результата") from ex
    return {**report, "conspect_json": str(cj), "conspect_md": str(cm),
            "analysis_needs_review": render_report["analysis_needs_review"],
            "assembly_revision": render_report["assembly_revision"],
            "analysis_revision": render_report["analysis_revision"],
            "external_publication": render_report["external_publication"]}


def confirm_conspect_analysis(snapshot):
    """Записать подтверждение только если файлы не изменились после показа человеку."""
    import conspect
    import recordings
    jp = Path(_done_json(snapshot["job_id"])["json"])
    with conspect.write_lock(jp):
        conspect.recover_pending(jp)
        with recordings.operation_for_document(jp, "human"):
            return _confirm_conspect_analysis_locked(snapshot)


def _confirm_conspect_analysis_locked(snapshot):
    import conspect
    import recordings
    import transcript
    current = describe_conspect_confirmation(snapshot["job_id"], snapshot["name"])
    if current != snapshot:
        raise CoreError("конспект или стенограмма изменились после запроса подтверждения")
    jp = Path(_done_json(snapshot["job_id"])["json"])
    saved = recordings.conspect_for_document(jp, snapshot["name"])
    if saved is not None:
        obj = saved["payload"]
        data = transcript.load(jp)
        conspect.confirm_analysis(obj, data)
        md, _ = conspect.render(obj, data)
        cj, cm = conspect.path_for(jp, snapshot["name"])
        conspect.write_files(jp, [
            (cj, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")),
            (cm, md.encode("utf-8")),
        ])
        if describe_conspect_confirmation(snapshot["job_id"], snapshot["name"])["source_revision"] \
                != snapshot["source_revision"]:
            raise CoreError("конспект или стенограмма изменились после запроса подтверждения")
        return {"name": snapshot["name"], "source_revision": snapshot["source_revision"],
                "analysis_revision": snapshot["source_revision"], "analysis_needs_review": False,
                "external_publication": "not_tracked"}
    cj, _ = conspect.path_for(jp, snapshot["name"])
    obj = json.loads(cj.read_text(encoding="utf-8"))
    data = json.loads(jp.read_text(encoding="utf-8"))
    conspect.confirm_analysis(obj, data)
    # Повторно проверяем снимок; внешние писатели могут не соблюдать блокировку.
    if describe_conspect_confirmation(snapshot["job_id"], snapshot["name"]) != snapshot:
        raise CoreError("конспект или стенограмма изменились после запроса подтверждения")
    conspect.write_files(jp, [(cj, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))])
    return {"name": snapshot["name"], "source_revision": snapshot["source_revision"],
            "analysis_revision": snapshot["source_revision"], "analysis_needs_review": False,
            "external_publication": "not_tracked"}


def apply_answers(job_id):
    """Применить сохранённые ответы человека; аудио не удаляется."""
    jp, md = job_json(job_id)
    if not jp:
        raise CoreError("Расшифровка не найдена или ещё не готова")
    return apply_answers_files(jp, md)


def apply_answers_files(jp, md):
    """Применить ответы к указанной паре файлов готовой задачи.

    После применения (и после фиксации в sqlite) карточки ответов с исходом applied_open получают новую
    редакцию (PR 13б): ответ применён, место осталось открытым — следующий вопрос человеку. Замена идёт
    после применения, когда его замки отпущены: `review_pack.upsert_cards` сам берёт операцию документа и
    замок ответов пакета — без вложения.

    Замки применения. Файловый режим: замок ответов пакета (`review_pack._answers_lock`), внутри него
    `conspect.write_lock` — на всё чтение, применение и запись состояния пакета (PR 17-0). Тот же порядок
    «ответы, затем запись» у `upsert_cards`: под замком ответов `transcript.load` берёт `write_lock`; обратный
    порядок дал бы взаимную блокировку. Режим базы: только `write_lock`; потерю ловят ожидаемые ревизии
    (проигравший получает `storage.Conflict`)."""
    report = _apply_answers_files_once(jp, md)
    return _replace_open_cards(jp, report)


def _card_errors():
    """Сбои постановки карточки в открытый пакет после записи стенограммы (PR 13б): не отменяют записанное,
    а попадают в отчёт (`cards_not_replaced`). storage.Conflict — запись сменилась между фиксацией и заменой."""
    import storage
    return OSError, RuntimeError, ValueError, storage.StorageError


def _replace_open_cards(jp, report):
    """Новая редакция карточек с исходом applied_open. Сбой замены не отменяет применённые ответы: он в
    отчёте (`cards_not_replaced`), и следующее «Применить» снова даёт этому ответу applied_open и замену."""
    import review_pack
    import transcript
    outcomes = report.get("outcomes") or []
    # слова места противоречия изменились после показа: карточка — новая редакция с текущими словами
    stale = [row["id"] for row in outcomes if row.get("outcome") == "source_conflict"]
    if stale:
        places = {i.get("id"): i for i in transcript.load(jp).get("review") or []}
        stale = [aid for aid in stale if transcript.contradiction_pending(places.get(aid) or {})]
        try:
            cards = review_pack.contradiction_cards(jp, stale, why="слова изменились после показа")
            report["replaced_cards"] = [aid for aid, state in cards.items() if state == "updated"]
        except _card_errors() as ex:
            report["cards_not_replaced"] = [{"id": aid, "why": str(ex)} for aid in stale]
    ids = [row["id"] for row in outcomes if row.get("outcome") == "applied_open"]
    # К-15: прежняя замена не состоялась, а человек уже ответил на старую редакцию — ответ ждёт поверх
    # применённого. Отбор без замка — с запасом; под замком ответов upsert_cards перепроверяет условие.
    due = [aid for aid in review_pack.open_after_answer_ids(review_pack.answers_state(jp)) if aid not in ids]
    ids += due
    if not ids:
        return report
    try:
        replaced = review_pack.open_after_answer_cards(jp, ids, due=due, report=report)
    except _card_errors() as ex:
        report["cards_not_replaced"] = (report.get("cards_not_replaced") or []) + \
            [{"id": aid, "why": str(ex)} for aid in ids]
        return report
    report["replaced_cards"] = (report.get("replaced_cards") or []) + replaced
    return report


def _apply_answers_files_once(jp, md):
    import conspect
    import recordings
    store, binding = recordings.open_for_document(jp)
    import review_pack
    state = review_pack.answers_state(jp, {"answers": {}})
    applied = review_pack._applied_map(state)
    if any(answer.get("kind") not in ("skip", "repeat") and
           applied.get(aid) != review_pack._sig(answer) and
           answer.get("actor_kind") != "human" and
           (answer.get("confirmation_context") or {}).get("actor_kind") != "human"
           for aid, answer in state.get("answers", {}).items()):
        if store is not None:
            store.close()
        raise CoreError("Ответ неизвестного происхождения нельзя применять как человеческий; "
                        "необходимо явное подтверждение")
    if store is not None:
        try:
            with conspect.write_lock(jp):
                import transcript
                initial = recordings.load_recording(store, binding["recording_id"])
                state = recordings.read_review_pack_state(jp)
                expected = state["_expected_versions"]
                context = recordings.operation_context(store, "human")
                manifest = state.get("manifest") or {}
                source = _answer_source_snapshot(store, binding["recording_id"], manifest)
                sources = _answer_item_sources(store, binding["recording_id"], manifest)
                guard = _answer_revision_conflicts(initial, manifest, state, source, sqlite=True,
                                                   sources=sources)
                with recordings.answer_draft(jp, initial, state, context) as draft:
                    draft["revision_conflicts"] = guard
                    report = _apply_answers_files_locked(jp, md)
                    candidate = recordings.merge_answer_draft(draft["data"], draft["state"])
                if candidate == initial:
                    return _answers_synced(store, binding["recording_id"], jp, report)
                def verify_guard(transaction_store):
                    import storage
                    head = recordings.recording_head(transaction_store, binding["recording_id"])
                    if head is None or dict(head) != expected:
                        raise storage.Conflict("recording changed while applying answers")
                    current = transaction_store.get_revision(
                        binding["recording_id"], "transcript", expected["source_revision"])["payload"]
                    original = _answer_source_snapshot(transaction_store, binding["recording_id"], manifest)
                    originals = _answer_item_sources(transaction_store, binding["recording_id"], manifest)
                    if _answer_revision_conflicts(current, manifest, state, original, sqlite=True,
                                                  sources=originals) != guard:
                        raise storage.Conflict("answer source changed while applying answers")

                result = recordings.commit_recording_data(
                    store, binding["recording_id"], candidate, expected,
                    context=context, precommit=verify_guard)
                if result.get("unchanged"):     # commit ничего не записал: как при candidate == initial
                    return _answers_synced(store, binding["recording_id"], jp, report)
                completed = recordings.materialize_recording(store, binding["recording_id"])
                _require_answers_written(store, binding["recording_id"], result["document_revision"], completed)
                return _answers_refreshed(store, binding["recording_id"], jp, report)
        finally:
            store.close()
    # файловый режим: состояние пакета пишут и дописывание карточек, и страница — под замком ответов; без него
    # дописывание, прочитавшее состояние до «Применить», стирало бы отметки применения и снимки отмены (PR 17-0)
    with review_pack._answers_lock(jp), conspect.write_lock(jp):
        import transcript
        manifest = review_pack.manifest(jp) or {}
        state = review_pack.answers_state(jp, {"answers": {}})
        guard = _answer_revision_conflicts(transcript.load(jp), manifest, state, None, sqlite=False)
        return _apply_answers_files_locked(jp, md, guard)


def _answers_synced(store, recording_id, jp, report):
    """Записывать нечего: довести текущую ревизию документа, если она ещё не записана на диск."""
    import recordings
    latest = recordings.document_binding(store, recording_id)
    if recordings.document_sync_state(store, recording_id, latest["document_revision"]) != "synced":
        completed = recordings.materialize_recording(store, recording_id)
        _require_answers_written(store, recording_id, latest["document_revision"], completed)
    return _answers_refreshed(store, recording_id, jp, report)


def _require_answers_written(store, recording_id, revision, completed):
    """Ответы уже в базе; если файлы не дописаны, исход словами (тип CoreError, как у прежнего отказа)."""
    import documents
    failure = documents.unsynced(store, recording_id, revision, completed)
    if failure is not None:
        raise CoreError(str(failure)) from failure


def _answers_refreshed(store, recording_id, jp, report):
    import recordings
    report["document_sync"] = "synced"
    try:
        report["conspects"] = recordings.refresh_conspects(store, recording_id, jp)
    except Exception as ex:
        raise CoreError(f"Ответы сохранены, но конспекты не обновлены: {ex}") from ex
    return report


def _answer_source_snapshot(store, recording_id, manifest):
    import storage
    try:
        return store.get_revision(recording_id, "transcript", manifest.get("source_revision"))["payload"]
    except (KeyError, ValueError, storage.StorageError):
        return None


def _answer_item_sources(store, recording_id, manifest):
    """{ревизия: стенограмма} для карточек, заменённых или дописанных по другой ревизии (upsert_cards)."""
    import review_pack
    out = {}
    for item in manifest.get("items") or []:
        revision = review_pack.card_source_revision(item, manifest)
        if revision != manifest.get("source_revision") and revision not in out:
            out[revision] = _answer_source_snapshot(store, recording_id, {"source_revision": revision})
    return out


def _answer_revision_conflicts(current, manifest, state, source, *, sqlite, sources=None):
    import review_pack
    import transcript
    applied = review_pack._applied_map(state)
    answers = state.get("answers") or {}
    base_source = source

    def snapshot(word, speaker):
        return (word.get("word_id"), word.get("w"), word.get("edited"),
                word.get("provenance") or {}, speaker)

    def select(data, ids):
        rows = [(word, segment.get("speaker"))
                for segment in (data or {}).get("segments") or []
                for word in segment.get("words") or []]
        positions = [i for i, (word, _) in enumerate(rows) if word.get("word_id") in ids]
        if (len(set(ids)) != len(ids) or len(positions) != len(ids) or not positions or
                positions != list(range(positions[0], positions[0] + len(ids))) or
                [rows[i][0].get("word_id") for i in positions] != ids):
            return None
        return [snapshot(*rows[i]) for i in positions]

    def select_by_time(data, item):
        # Тем же правилом, что и ответ: _find_span по началу слова с допуском 0,01 с.
        loc = transcript._find_span(data, item)
        if loc is None:
            return None
        si, w1, w2 = loc
        segment = data["segments"][si]
        return [snapshot(word, segment.get("speaker"))
                for word in (segment.get("words") or [])[w1:w2]]

    def select_pause(data, item):
        # Слова пропуска — начатые в его паузе (transcript.pause_words); пустая пауза — пустой снимок, не None.
        times = transcript._place_time(item)
        if times is None:
            return None
        return [snapshot(word, data["segments"][si].get("speaker"))
                for si, _k, word in transcript.pause_words(data, *times)]

    conflicts = {}
    for item in manifest.get("items") or []:
        aid = item.get("id")
        answer = answers.get(aid)
        gap = transcript.is_gap_place(item)
        ids = None if gap else item.get("word_ids")      # у пропуска word_ids пакета не читаются
        if not review_pack.answer_is_current(answer, item) or aid in applied:
            continue
        # Карточка, заменённая или дописанная позже сборки пакета, сверяется с ревизией, по которой собрана.
        revision = review_pack.card_source_revision(item, manifest)
        if sqlite and revision != manifest.get("source_revision"):
            source = (sources or {}).get(revision)
        else:
            source = base_source
        if sqlite and source is None:
            conflicts[aid] = "недоступна исходная ревизия карточки"
            continue
        if not ids:
            # Старый вид карточки: ответ ищет место по времени. Сравниваем ровно те
            # слова, которые он перезапишет, включая говорящего и все измерения provenance.
            if sqlite and (gap or "word_ids" not in item):
                pick = select_pause if gap else select_by_time
                actual = pick(current, item)
                original = pick(source, item)
                if actual is None or original is None or actual != original:
                    conflicts[aid] = "слова карточки изменились или недоступна исходная ревизия"
            continue
        actual = select(current, ids)
        original = select(source, ids) if sqlite else actual
        if actual is None or original is None or actual != original:
            conflicts[aid] = "слова карточки изменились или недоступна исходная ревизия"
    return conflicts


def _apply_answers_files_locked(jp, md, revision_conflicts=None):
    import review_pack as P
    import transcript
    import recordings
    draft = recordings.answer_draft_for(jp)
    if revision_conflicts is None and draft is not None:
        revision_conflicts = draft.get("revision_conflicts")
    revision_conflicts = dict(revision_conflicts or {})
    import freeform
    state_before = P.answers_state(jp, {"answers": {}})
    applied_before = P._applied_map(state_before)
    # места, где ответы ноутбука и облака расходились (К-23): применённый более поздний ответ ждёт проверки
    conflicted = {c.get("id") for c in state_before.get("conflicts") or []}
    conflict_was = {c.get("id"): c.get("was") for c in state_before.get("conflicts") or []}
    decisions, rep = P.decisions(jp)
    free = P.free_answers(jp)
    if draft is not None:
        items = {item["id"]: item for item in (P.manifest(jp) or {}).get("items") or []}
        already_applied = P._applied_map(draft["state"])
        now = transcript.load(jp)
        places = {r.get("id"): r for r in now.get("review") or []}
        for aid, answer in free.items():
            place = places.get(aid) or {}
            if transcript.is_gap_place(place):
                # пропуск: слова карточки — слова паузы; word_ids старого пакета (ID соседа) не читаются (PR 6а)
                found = transcript.place_words(now, place)
                ids = found[0] if found else None
            else:
                ids = (items.get(aid) or {}).get("word_ids")
            if not ids or aid in already_applied or aid in (revision_conflicts or {}):
                continue
            preview = P.apply_free(jp, md, {aid: answer}, dry_run=True)
            targets = set((preview.get("target_ids") or {}).get(aid) or [])
            if targets - set(ids):
                revision_conflicts[aid] = "свободный ответ затрагивает слова вне карточки"
    requested_ids = {d.get("id") for d in decisions if d.get("id")} | set(free)
    preflight = P.preflight_answers(jp, md, decisions, free)
    revision_conflicts = revision_conflicts or {}
    blocked = {aid for conflict in preflight for aid in conflict["cards"]} | set(revision_conflicts)
    blocked_free = {aid: answer for aid, answer in free.items() if aid in blocked}
    decisions = [d for d in decisions if d.get("id") not in blocked]
    free = {aid: answer for aid, answer in free.items() if aid not in blocked}
    if revision_conflicts:
        rep["revision_conflicts"] = [{"id": aid, "why": why} for aid, why in revision_conflicts.items()]
    if preflight:
        rep["preflight_conflicts"] = preflight
        rep["human"] = list(dict.fromkeys((rep.get("human") or []) + sorted(blocked)))
        rep["changed"] = [aid for aid in rep.get("changed") or [] if aid not in blocked]
    # Отменяется только ответ «отменить»: замены применённого ответа другим до полной замены нет.
    undo = [d for d in decisions if "_undo_snapshot" in d]
    decisions = [d for d in decisions if "_undo_snapshot" not in d]
    undone, undo_failed = [], set()
    for d in undo:
        result = transcript.undo_answer(jp, md, d["id"], d["_undo_snapshot"])
        if result.get("reverted"):
            undone.append(d["id"])
            rep["reverted"].append(d["id"])
            P.mark_applied(jp, {d["id"]}, outcomes={d["id"]: "reverted"})
        else:
            rep.setdefault("undo_conflicts", []).append(result)
            undo_failed.add(d["id"])
    snapshots, report = {}, {}
    snapshot_bytes = 0

    def remember_snapshot(aid, before, after):
        nonlocal snapshot_bytes
        snapshot = transcript.answer_snapshot(before, after)
        size = len(json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))
        # Даже компактные снимки крупных изменений могут быть большими.
        # Ответ без снимка останется применённым, но его
        # автоматическая отмена будет недоступна и об этом сообщается.
        if snapshot_bytes + size > 16 * 1024 * 1024:
            rep.setdefault("undo_unavailable", []).append(aid)
            return
        snapshots[aid] = snapshot
        snapshot_bytes += size

    if decisions:
        state = transcript.load(jp)
        order = {it["id"]: it.get("t", 0) for it in state.get("review") or []}
        for decision in sorted(decisions, key=lambda d: -order.get(d.get("id"), 0)):
            before = transcript.load(jp)
            after, result = transcript.apply_review(jp, md, [decision], by="human")
            for key, value in result.items():
                if isinstance(value, list) and key != "vocab_suggestions":
                    report.setdefault(key, []).extend(value)
                elif key in ("applied", "kept", "human"):
                    report[key] = report.get(key, 0) + value
                else:
                    report[key] = value
            aid = decision.get("id")
            if aid in result.get("successful_ids", []) and decision.get("confirm_applied"):
                # подтверждение человека (К-23): слова те же, снимок прежний — отмена вернёт место к состоянию
                # до применённого ответа, а не к состоянию после него; отпечаток «после» — с подтверждением
                kept = transcript.confirm_snapshot(
                    (P.answers_state(jp, {"answers": {}}).get("undo_snapshots") or {}).get(aid), before, after)
                if kept is not None:
                    snapshots[aid] = kept
                P.mark_applied(jp, {aid}, snapshots={aid: kept} if kept is not None else None,
                               outcomes={aid: "applied"})
            elif aid in result.get("successful_ids", []):
                # Ответ, применённый с отметкой «проверить человеку» (human_after, К-23), тоже учитывается как
                # применённый: подпись и снимок отмены ставятся, исход — applied_check.
                remember_snapshot(aid, before, after)
                # Без снимка ответ всё равно применён: отметка ставится, отмена сообщит, что снимка нет.
                P.mark_applied(jp, {aid}, snapshots={aid: snapshots[aid]} if aid in snapshots else None,
                               outcomes={aid: "applied"})
    free_items = []
    if blocked_free:
        free_items.append({"conflicts": preflight,
                           "left_to_human": [{"id": aid, "why": revision_conflicts.get(
                                               aid, "конфликт с другой карточкой до применения"),
                                               "answer": answer.get("text")} for aid, answer in blocked_free.items()]})
    if free:
        # Свободные ответы нередко затрагивают одни и те же слова. Применяем их
        # отдельно, чтобы у каждого успешного ответа был обратимый структурный
        # снимок; это снимает консервативный отказ от пакетной отмены.
        order = {it["id"]: it.get("t1", 0) for it in (P.manifest(jp) or {}).get("items") or []}
        for aid in sorted(free, key=lambda x: -order.get(x, 0)):
            before = transcript.load(jp)
            part = P.apply_free(jp, md, {aid: free[aid]})
            free_items.append(part)
            if aid in part.get("successful_ids", []):
                remember_snapshot(aid, before, transcript.load(jp))
                P.mark_applied(jp, {aid}, snapshots={aid: snapshots[aid]} if aid in snapshots else None,
                               outcomes={aid: "applied"})
    if free_items:
        rep["free_text"] = _merge_free_reports(free_items)
    successful = set(report.get("successful_ids", []))
    successful.update(rep.get("free_text", {}).get("successful_ids", []))
    successful.difference_update(blocked)          # заблокированные preflight не применялись
    rep["applied_from_answers"] = sum(1 for d in decisions if d.get("id") in successful)
    snapshots = {aid: snapshot for aid, snapshot in snapshots.items() if aid in successful}
    successful.update(undone)
    apply_conflicts = {row.get("id") for row in report.get("conflicts") or [] if row.get("id")}
    free_report = rep.get("free_text") or {}
    free_conflicts = {aid for row in free_report.get("conflicts") or [] for aid in row.get("cards") or []}
    free_not_found = set(free_report.get("not_found") or [])
    pause_refused = set(report.get("pause_refused") or [])      # пропуск без кнопки паузы (6б)
    listen_mismatch = set(report.get("listen_mismatch") or [])  # кнопка прослушивания против слов (К-3/К-4)
    free_hints = set(free_report.get("hints") or [])             # короткий ответ — подсказка, место открыто
    outcomes = {}
    human_after = set(rep.get("human_after") or [])
    for aid in requested_ids:
        if aid in successful:
            checked = aid in human_after or (aid in free and aid in conflicted)
            outcomes[aid] = "reverted" if aid in undone else "applied_check" if checked else "applied"
        elif aid in revision_conflicts:
            outcomes[aid] = "apply_conflict"
        elif aid in blocked:
            outcomes[aid] = "blocked_preflight"
        elif aid in undo_failed:
            outcomes[aid] = "undo_conflict"
        elif aid in (report.get("already_decided") or []):
            outcomes[aid] = "already_decided"
        elif aid in rep.get("human", []) or aid in human_after:
            # human_after без успеха — например, кнопка прослушивания: при расхождении ответов её не применяют
            outcomes[aid] = "left_for_human"
        elif aid in listen_mismatch:
            outcomes[aid] = "listen_mismatch"
        elif aid in pause_refused:
            outcomes[aid] = "pause_refused"
        elif aid in free_hints:
            outcomes[aid] = "hint"
        elif aid in apply_conflicts or aid in free_conflicts:
            outcomes[aid] = "apply_conflict"
        elif aid in free_not_found or aid in (report.get("not_found") or []):
            outcomes[aid] = "not_found"
        else:
            outcomes[aid] = "not_applied"
        if aid in free and freeform.negative_answer(free[aid].get("text")) and \
                outcomes[aid] in ("not_applied", "left_for_human", "not_found"):
            outcomes[aid] = ("kept_previous" if P.over_applied(free[aid], applied_before.get(aid)) == "kept_previous"
                             else "negative")
    # Исход есть у каждого сохранённого ответа пакета, а не только у тех, что применялись в этот раз.
    state_now = P.answers_state(jp, {"answers": {}})
    stored = state_now.get("answers") or {}
    events_now = state_now.get("application_events") or []
    # ответ прежней редакции карточки (остался после сбоя замены) не применялся и исхода не получает
    current_items = {item.get("id"): item for item in (P.manifest(jp) or {}).get("items") or []}
    stored = {aid: a for aid, a in stored.items() if P.answer_is_current(a, current_items.get(aid))}
    undo_conflict_ids = {row.get("id") for row in rep.get("undo_conflicts") or []}
    for aid in stored:
        if aid in rep.get("source_conflict_ids", []):
            outcomes[aid] = "source_conflict"
        elif aid in outcomes:
            continue
        elif aid in rep.get("unchanged", []):
            # неизменившийся ответ сохраняет отметку «проверьте» (К-23), пока человек её не снимет: иначе
            # место human дало бы applied_open и замену карточки с потерей отмены
            last = P._last_place_event(events_now, aid)
            outcomes[aid] = "applied_check" if (last or {}).get("outcome") == "applied_check" else "applied"
        elif aid in rep.get("kept_previous", []):
            outcomes[aid] = "kept_previous"
        elif aid in rep.get("pending_new", []):
            outcomes[aid] = "pending_new"
        elif aid in undo_conflict_ids:
            outcomes[aid] = "undo_conflict"
    # Итог — после всех ответов: ответ применён, а его место открыто (нужно прослушивание, или слова интервала
    # изменил другой ответ или правка) — исход applied_open, не applied (PR 7). Ответ остаётся применённым:
    # подпись и снимок для отмены не снимаются; повтор «Применить» такое место сам не закроет.
    places_now = {r.get("id"): r for r in transcript.load(jp).get("review") or []}
    for aid, outcome in outcomes.items():
        if outcome == "applied" and aid in places_now and transcript.place_unresolved(places_now[aid]):
            outcomes[aid] = "applied_open"
    failed_outcomes = {aid: outcome for aid, outcome in outcomes.items()
                       if outcome not in ("applied", "reverted")}
    if failed_outcomes:
        P.mark_applied(jp, set(), outcomes=failed_outcomes)
    # Оставшиеся (пропуск, отмена без применённого ответа, давняя отмена) — по сохранённому состоянию.
    states = P.answer_states(jp, outcome_details=False)
    for aid in stored:
        if aid not in outcomes:
            outcomes[aid] = states[aid]["state"]
    # Уточнения к исходу — то, что прежний отчёт показывал отдельными списками.
    details = {aid: [states[aid]["detail"]] if states[aid].get("detail") else [] for aid in stored}
    fresh = {aid: [] for aid in stored}      # уточнения этого «Применить» — их же видит карточка (К-22)

    def add_detail(aid, text):
        if aid in details and text and text not in details[aid]:
            details[aid].append(text)
            if text != P.UNDO_UNAVAILABLE:
                fresh[aid].append(text)

    for row in rep.get("undo_conflicts") or []:
        add_detail(row.get("id"), row.get("why") or row.get("conflict"))
    for conflict in preflight or []:
        for aid in conflict.get("cards") or []:
            others = [states[x]["card"] or x for x in conflict["cards"] if x != aid and x in states]
            add_detail(aid, "с карточкой " + ", ".join(str(x) for x in others) if others else conflict.get("why"))
    for aid, why in revision_conflicts.items():
        add_detail(aid, why)
    items_now = {item.get("id"): item for item in (P.manifest(jp) or {}).get("items") or []}
    for aid in stored:
        if outcomes[aid] == "applied_check":
            was = conflict_was.get(aid)
            add_detail(aid, "раньше на это место был другой ответ" +
                       (f" ({P._desc(was, items_now[aid])})" if was and aid in items_now else "") +
                       "; ответы ноутбука и облака расходились")
    for aid in stored:
        if outcomes[aid] == "applied_open":
            note = (places_now.get(aid) or {}).get("note") or ""
            add_detail(aid, "слова интервала изменились после ответа" if "нужна новая проверка" in note else
                       "ответ не исправил слова, в которых модели расходятся, или место требует прослушивания")
    for row in free_report.get("left_to_human") or []:
        add_detail(row.get("id"), row.get("why"))
    for aid, why in (free_report.get("failed") or {}).items():
        add_detail(aid, why)
    for row in report.get("conflicts") or []:
        add_detail(row.get("id"), row.get("why") or ("слово уже исправлено прежней правкой"
                                                      if row.get("kept_edits") else None))
    for aid in rep.get("source_conflict_ids") or []:
        add_detail(aid, (rep.get("source_conflict_why") or {}).get(aid) or rep.get("source_conflict"))

    def row(aid):
        out = {"id": aid, "card": states[aid]["card"], "outcome": outcomes[aid],
               "why": P.OUTCOME_REASONS[outcomes[aid]]}
        if outcomes[aid] == "applied" and aid in (rep.get("undo_unavailable") or []):
            add_detail(aid, P.UNDO_UNAVAILABLE)
        extra = [text for text in details[aid] if text != out["why"]]
        if extra:
            out["detail"] = "; ".join(extra)
        return out

    rep["outcomes"] = [row(aid) for aid in sorted(stored, key=lambda x: (states[x]["card"] or 0, x))]
    # уточнения — под карточку (только при хранении в базе; в файловом режиме ничего не пишется)
    P.remember_outcome_details(jp, {aid: (stored[aid].get("ts"), stored[aid].get("source"),
                                          [text for text in fresh[aid] if text != P.OUTCOME_REASONS[outcomes[aid]]])
                                    for aid in stored})
    if undone:
        rep["voice_profile_warning"] = {
            "answer_ids": undone,
            "message": ("Если эта запись ранее пополнила голосовой профиль, её прежний вклад мог остаться. "
                        "Профиль не изменён: проверьте вклад и при необходимости отдельно подтвердите "
                        "его обновление или отзыв."),
        }
    return {**rep, "review": report}


def edit_span(job_id, t1, t2, text, speaker="", note="", by=None):
    """Правка отрезка t1–t2 (с): текст и говорящий; без пересборки карточек. Автор by обязателен."""
    import transcript
    try:
        transcript.check_edit_span_author(by)
    except ValueError as ex:
        raise CoreError(str(ex))
    s = _done_json(job_id)
    return edit_span_files(s["json"], s["markdown"], t1, t2, text, speaker or None, note or None, by)


def edit_span_files(jp, md, t1, t2, text, speaker=None, note=None, by=None):
    """Правка отрезка пары файлов. Противоречие решению человека (PR 13б) — не ошибка, а результат
    {"written": false, "contradiction": "rN", "card": "added|updated|unchanged|pending|not_replaced",
    "text": «не записано: …»}: слова не меняются, карточка места встаёт в открытый пакет. Сбой постановки
    карточки — card: not_replaced и `cards_not_replaced`; повтор той же правки её ставит."""
    import review_pack
    import transcript
    try:
        _, rep = transcript.edit_span(jp, md, t1, t2, text, speaker, note, by)
    except ValueError as ex:
        raise CoreError(str(ex))
    rid = rep.get("contradiction")
    if rid:
        try:
            rep["card"] = review_pack.contradiction_cards(jp, [rid]).get(rid, "pending")
        except _card_errors() as ex:
            rep["card"] = "not_replaced"
            rep["cards_not_replaced"] = [{"id": rid, "why": str(ex)}]
    return rep


def apply_review_files(jp, md, decisions, by="claude"):
    """Решения по местам разбора (transcript.apply_review); предложения агента по словам или месту человека
    становятся карточками противоречия в открытом пакете (PR 13б): `contradictions`, `cards`."""
    import review_pack
    import transcript
    _, rep = transcript.apply_review(jp, md, decisions, by=by)
    ids = list(dict.fromkeys(rep.get("contradictions") or []))
    if ids:
        rep["contradictions"] = ids
        try:
            rep["cards"] = review_pack.contradiction_cards(jp, ids)
        except _card_errors() as ex:
            # решения уже записаны: сбой постановки карточек — в отчёт; повтор решения или сборка их поставят
            rep["cards"] = {rid: "not_replaced" for rid in ids}
            rep["cards_not_replaced"] = [{"id": rid, "why": str(ex)} for rid in ids]
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


def legacy_speaker_review(job_id, build_pack=False):
    """Перенос старых подтверждений говорящего: найти места или собрать по ним карточки.

    По умолчанию только показывает места, где по истории правок видно решение человека
    о говорящем (voices.legacy_speaker_places). build_pack=True собирает по этим местам
    карточки разбора; подтверждение — ответ человека на локальной странице, применяемый
    штатным apply_answers (он ставит пословную отметку provenance.speaker). Автоматически
    ничего не подтверждается. Только для стенограмм старого файлового формата.
    """
    import recordings
    import review_pack
    import transcript
    import voices
    s = _done_json(job_id)
    jp, md = s["json"], s["markdown"]
    store, _ = recordings.open_for_document(jp)
    if store is not None:
        store.close()
        raise CoreError("для записи в SQLite перенос старых подтверждений не применяется")
    data = transcript.load(jp)
    found = voices.legacy_speaker_places(data)
    result = {"json": jp, **found, "pack": None}
    if not build_pack or not found["places"]:
        return result
    # пословные ID: ответ найдёт место точно, даже если номера слов сдвинулись. Старый файл получает
    # их при первом штатном сохранении (transcript.save присваивает постоянные ID всем словам)
    added = sum(1 for seg in data["segments"] for word in seg.get("words") or [] if not word.get("word_id"))
    if added:
        transcript.save(data, jp, md)
        data = transcript.load(jp)
        found = voices.legacy_speaker_places(data)
        result.update(found)
    names = sorted({seg.get("speaker") for seg in data["segments"]
                    if seg.get("speaker") and not str(seg["speaker"]).startswith("Спикер")})
    items = []
    for place in found["places"]:
        ids = [w["word_id"] for w in data["segments"][place["seg"]]["words"][place["w1"]:place["w2"]]]
        options = [{"label": f"Да, это говорит {place['speaker']}", "decision": {"word_ids": ids}}]
        options += [{"label": f"Нет, это {name}", "decision": {"word_ids": ids, "speaker": name}}
                    for name in names if name != place["speaker"]]
        # answer_mode: свой ответ на такой карточке — решение о говорящем, а не текст реплики
        items.append({"id": place["id"], "answer_mode": "speaker",
                      "question":(f"Подтвердите говорящего ({place['evidence']}). Ответ «да» войдёт в "
                                   "голосовой профиль как подтверждение человека."),
                      "options": options})
    result["word_ids_added"] = added
    result["pack"] = review_pack.build(jp, items)
    return result


# ---------- настройки, группы, профили ----------
def settings_info():
    import recordings
    import storage
    result = {"data_dir": str(paths.DATA_DIR), "config_file": str(paths.CONFIG_FILE),
              "overrides": settings.load_overrides(), "effective": settings.load_config(),
              "recording_backend": settings.recording_backend(), "groups": paths.list_groups(),
              # старая раскладка данных переносится только явной командой migrate_data
              "legacy_migration_pending": bool(migrate.pending())}
    if paths.STATE_DB.is_file():
        try:
            store = recordings.open_store()
            try:
                result["recording_identity"] = store.current_identity()
            finally:
                store.close()
        except (OSError, storage.StorageError) as ex:
            result["recording_store_error"] = str(ex)
    return result


# ---------- голоса: перенесены в core_voices.py (имена прежние, переотдача — в конце файла) ----------


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


def do_cleanup_hypotheses(unit_id="", older_than_days=None, apply=False):
    try:
        return data_io.cleanup_hypotheses(unit_id or None, older_than_days, apply)
    except ValueError as ex:
        raise CoreError(str(ex)) from ex


# ---------- описание действия для человека (без выполнения) ----------
def describe_export(group="", dest_dir="", with_jobs=False):
    groups = [paths.safe_group(group)] if group else paths.list_groups()
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


def describe_cleanup_hypotheses(unit_id="", older_than_days=None):
    r = do_cleanup_hypotheses(unit_id, older_than_days, False)
    scope = f"записи {unit_id}" if unit_id else "всех записей"
    age = f", старше {older_than_days} дн." if older_than_days is not None else ""
    return [f"Удаление сохранённых гипотез моделей (полный текст записей) {scope}{age}: "
            f"файлов {r['files']}, записей {r['units']}, {r['bytes'] // 1024} КБ.",
            "Стенограммы, конспекты, аудио и статусы задач не затрагиваются. Удалённое не восстановить."]


def describe_migrate():
    r = do_migrate(True)
    res = r.get("result")
    return ["Перенос данных в папку данных из прежних мест.", f"Папка данных: {r['data_dir']}",
            f"План: {json.dumps(res, ensure_ascii=False)[:300] if res is not None else r.get('error', '—')}"]


# ---------- голоса: перенесены в core_voices.py, имена прежние ----------
from core_voices import (  # noqa: E402,F401
    REFRESH_MEANINGS,
    _speaker_voice_summary,
    _voice_hash,
    _voice_job_map,
    _voice_vocab_context,
    backup_voice_profile,
    backup_voice_profiles,
    prepare_voice_action,
    preview_voice_learning,
    revoke_voice_source,
    set_voice_auto_learn,
    set_voice_quality_policy,
    start_voice_update,
    voice_action_lines,
    voice_profiles,
    voice_quality,
    voice_refresh_plan,
)


# ---------- экран: перенесён в core_screen.py, имена прежние ----------
from core_screen import (  # noqa: E402,F401
    _conspect_screen_sections,
    prepare_restore_snapshot,
    quote_repair_state,
    quote_repair_view,
    read_recording_revision,
    recording_conspects,
    recording_revisions,
    restore_recording_revision,
    workflow_list,
    workflow_repair_quotes,
    workflow_repair_quotes_info,
    workflow_repair_quotes_prepare,
)


# ---------- конфликты внешней правки и др.: перенесены в core_conflicts.py, имена прежние ----------
from core_conflicts import (  # noqa: E402,F401
    CONFLICT_DIFF_CELLS,
    CONFLICT_DIFF_WORK,
    CONFLICT_PREVIEW_CHARS,
    CONFLICT_SCOPES,
    CONFLICT_SHOWN_CHARS,
    CONFLICT_SHOWN_LINES,
    _CONFLICT_COLUMNS,
    _CONFLICT_SQL,
    _KNOWN_CONFLICTS,
    _WIDE_SPACES,
    _ZERO_WIDTH,
    _also_replaced,
    _anchors,
    _answer_confirmation_hash,
    _conflict_facts,
    _conflict_item,
    _conflict_row,
    _conflict_rows,
    _copy_name,
    _copy_problem,
    _damaged_copies_warning,
    _decode_copy,
    _diff_lines,
    _effect,
    _eol_names,
    _failure_reason,
    _file_at,
    _file_notes,
    _file_outcome,
    _file_words,
    _invisible_note,
    _is_journal_stop,
    _journal_blockers,
    _journal_reason,
    _lines,
    _listed_program_version,
    _missing_text,
    _next_step,
    _not_written_warning,
    _observed_bytes,
    _present,
    _program_version,
    _queue_unfinished,
    _real_case,
    _recording_reference,
    _refusal,
    _replace_file,
    _resolve_for_agent,
    _same_object,
    _set_stops,
    _show_text,
    _shown_file,
    _stops_words,
    _unresolvable_wait,
    _view,
    _waits_for,
    _write_and_report,
    backup_store,
    confirm_review_answers,
    conflict_copy,
    explain_error,
    find_recording_results,
    inspect_backup,
    list_all_conflicts,
    list_conspect_conflicts,
    list_document_conflicts,
    prepare_answer_confirmation,
    preview_conflict,
    read_recording_result,
    replace_conflict_file,
    replacement_effect,
    resolve_conspect_conflict,
    resolve_document_conflict,
    restore_from_backup,
    retry_conflict_file,
    say_error,
    sync_recording_documents,
)
