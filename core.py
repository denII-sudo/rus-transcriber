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
            unresolved = sum(1 for item in (review.get("review") or [])
                             if item.get("status", "open") in ("open", "human"))
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
    import workflow
    store = recordings.open_store()
    try:
        request = workflow.get(store, request_id)
        budget = request["policy"]["question_budget"]
        if not request["recording_id"] or not budget:
            return {"questions": [], "budget": budget, "recording_id": request["recording_id"]}
        current = store.get_current(request["recording_id"])
        items = [item for item in (current["review"]["payload"].get("review") or [])
                 if item.get("status", "open") in ("open", "human") and isinstance(item.get("id"), str)]
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


# ---------- конфликты внешней правки: копия человека и явная замена файла ----------
CONFLICT_SCOPES = ("document", "conspect")
CONFLICT_PREVIEW_CHARS = 20000        # знаков копии в предпросмотре
CONFLICT_SHOWN_LINES, CONFLICT_SHOWN_CHARS = 200, 500   # строк различий и знаков в строке на показ
CONFLICT_DIFF_CELLS = 90000    # сколько пар строк сравнивать точно на одном участке; для большего ищутся опоры
CONFLICT_DIFF_WORK = 2000000   # бюджет уточнения сверх первичного поиска опор, пропорционального входу

_KNOWN_CONFLICTS = {
    "stale revisions": "Запись изменилась после показа. Обновите её и повторите действие.",
    "unresolved document conflict": "Сначала решите конфликт внешней правки документа.",
    "unresolved conspect conflict": "Сначала решите конфликт внешней правки свёртки.",
    "conflict already resolved": "Этот конфликт уже решён. Обновите список.",
    "restored copy is inactive": "Восстановленная копия базы не активирована.",
    "restore copy requires explicit rebind": "Восстановленная копия требует явного подключения папок.",
    "external file too large to preserve; manual resolution required":
        "Файл слишком велик для сохранения копии: нужен отдельный ручной разбор.",
}


def say_error(ex):
    """Текст отказа для человека и агента: известные английские коды хранилища переведены, остальное как было.

    Русские исходы записи (documents.NotWritten, documents.NotSynced) уже со словами и идут без изменений."""
    import storage
    if isinstance(ex, storage.Conflict):
        return _KNOWN_CONFLICTS.get(str(ex), str(ex))
    return str(ex)


def explain_error(ex):
    """Русское пояснение сбоя хранилища или базы для экрана; неизвестный сбой — с исходной причиной."""
    import storage
    text = str(ex)
    low = text.lower()
    if isinstance(ex, storage.Busy) or (isinstance(ex, sqlite3.OperationalError)
                                       and ("locked" in low or "busy" in low)):
        return "База записей сейчас занята другой операцией. Подождите немного и повторите."
    if isinstance(ex, storage.Conflict):
        known = _KNOWN_CONFLICTS
        return known.get(text, text if re.search("[А-Яа-яЁё]", text)
                         else f"Действие не выполнено: состояние записи не позволяет его выполнить (причина: {text}).")
    if isinstance(ex, storage.StorageError) and re.search("[А-Яа-яЁё]", text):
        return text    # собственный отказ программы уже по-русски: прежний текст, не «ошибка базы»
    if "cannot open database" in low or "unable to open database" in low:
        return ("Не удалось открыть базу записей: проверьте, что папка данных на месте и база не занята другой "
                f"программой (причина: {text}).")
    if "database is busy" in low:
        return "База записей сейчас занята другой операцией. Подождите немного и повторите."
    if "hash mismatch" in low:
        return ("Контрольная сумма сохранённых данных не совпала: строка в базе повреждена "
                f"(причина: {text}). Такие данные программа не использует.")
    if isinstance(ex, (sqlite3.Error, storage.StorageError)):
        return f"Ошибка базы записей (причина: {text})."
    if isinstance(ex, OSError):
        return f"Не удалось обратиться к файлу или папке (причина: {text})."
    return f"Непредвиденная ошибка программы (причина: {text})."


def _real_case(path, cache):
    """Путь в том регистре, в каком он лежит на диске (база хранит его приведённым к нижнему); только для показа.

    Что не нашлось на диске, остаётся как есть."""
    path = Path(path)
    parts = path.parts
    if not path.is_absolute() or len(parts) < 2:
        return path
    current, shown = Path(parts[0]), [parts[0][:1].upper() + parts[0][1:]]
    for number, part in enumerate(parts[1:]):
        if current not in cache:
            try:
                names = os.listdir(current)
            except OSError:
                names = []
            folded = {}
            for name in names:              # при нескольких совпадениях — первое, как раньше при переборе
                folded.setdefault(name.casefold(), name)
            cache[current] = (set(names), folded)
        exact, folded = cache[current]
        found = part if part in exact else folded.get(part.casefold())
        if found is None:
            shown.extend(parts[1 + number:])
            break
        shown.append(found)
        current = current / found
    return Path(*shown)


_CONFLICT_COLUMNS = ("f.kind, f.relative_path, f.observed_bytes IS NOT NULL AS preserved, f.observed_hash, "
                     "f.observed_size, f.resolution_operation_id IS NOT NULL AS resolved, f.rowid AS seq, "
                     "o.root_id, (SELECT path FROM allowed_roots WHERE root_id=o.root_id) AS root_path, "
                     "rec.location")
_CONFLICT_SQL = {
    "document": ("SELECT f.conflict_id, f.recording_id, f.document_revision AS revision, NULL AS conspect_id, "
                 "NULL AS name, " + _CONFLICT_COLUMNS + " FROM document_conflicts f "
                 "JOIN document_outbox o ON o.recording_id=f.recording_id "
                 "AND o.document_revision=f.document_revision "
                 "JOIN recordings rec ON rec.recording_id=f.recording_id"),
    "conspect": ("SELECT f.conflict_id, c.recording_id, f.revision, f.conspect_id, c.name, "
                 + _CONFLICT_COLUMNS + " FROM conspect_conflicts f "
                 "JOIN conspects c ON c.conspect_id=f.conspect_id "
                 "JOIN conspect_outbox o ON o.conspect_id=f.conspect_id AND o.revision=f.revision "
                 "JOIN recordings rec ON rec.recording_id=c.recording_id")}


def _conflict_rows(store, scope, where="", args=()):
    """Конфликты одной области без самих байт: при просмотре списка копии не читаются."""
    rows = []
    for found in store.db.execute(_CONFLICT_SQL[scope] + where + " ORDER BY f.rowid", args):
        row = dict(found, scope=scope, preserved=bool(found["preserved"]), resolved=bool(found["resolved"]))
        row["object"] = (scope, row["conspect_id"] or row["recording_id"])
        rows.append(row)
    return rows


def _conflict_row(store, recording_id, conflict_id, scope):
    import storage
    storage._uuid(recording_id)
    storage._uuid(conflict_id)
    if scope not in CONFLICT_SCOPES:
        raise CoreError("Неизвестный вид конфликта.")
    rows = _conflict_rows(store, scope, " WHERE f.conflict_id=?", (conflict_id,))
    if not rows or rows[0]["recording_id"] != recording_id:
        raise CoreError("Конфликт не найден для этой записи.")
    return rows[0]


def _observed_bytes(store, row):
    """Сохранённая копия человека целиком или None, если файл был слишком велик."""
    import documents
    found = store.db.execute(f"SELECT observed_bytes FROM {row['scope']}_conflicts WHERE conflict_id=?",
                             (row["conflict_id"],)).fetchone()[0]
    if found is None:
        return None
    found = bytes(found)
    if len(found) != row["observed_size"] or documents._digest(found) != row["observed_hash"]:
        raise CoreError("Сохранённая копия повреждена: её размер или контрольная сумма не совпали.")
    return found


def _program_version(store, row):
    """Последняя версия программы для файлов объекта конфликта: (ревизия, {вид файла: байты})."""
    import storage
    try:
        if row["scope"] == "document":
            revision = store.db.execute("SELECT max(document_revision) FROM document_versions WHERE recording_id=?",
                                        (row["recording_id"],)).fetchone()[0]
            saved = store.get_document(row["recording_id"], revision)
            return revision, {kind: bytes(saved[kind]) for kind in ("md", "json", "manifest")}
        saved = store.get_conspect(row["conspect_id"])
        return saved["revision"], {kind: bytes(saved[kind]) for kind in ("md", "json")}
    except storage.Busy as ex:
        raise CoreError(explain_error(ex)) from ex
    except sqlite3.Error as ex:
        raise CoreError(explain_error(ex)) from ex
    except (storage.StorageError, ValueError, KeyError, TypeError) as ex:
        raise CoreError("Сохранённая версия программы для этого файла повреждена или не читается "
                        f"(причина: {ex}): заменять файл нечем. Копию вашей правки можно скачать.") from ex


def _file_at(store, row, relative):
    """Файл на диске без записи и без создания папок: (missing|unavailable|ok, sha256 или None)."""
    import documents
    import storage
    try:
        root = documents._authorized_root(store, row["root_id"], row["root_path"])
        path, exists = documents._read_only_target(root, relative)
        return ("ok", documents._file_digest(path)[0]) if exists else ("missing", None)
    except FileNotFoundError:
        return "missing", None
    except (OSError, ValueError, storage.StorageError):
        return "unavailable", None


def _queue_unfinished(store, row):
    """Есть ли у объекта конфликта запись файлов, которую программа не довела до конца."""
    table, column = (("document_outbox", "recording_id") if row["scope"] == "document"
                     else ("conspect_outbox", "conspect_id"))
    return store.db.execute(f"SELECT 1 FROM {table} WHERE {column}=? AND state!='synced'",
                            (row["object"][1],)).fetchone() is not None


def _shown_file(row, cache):
    """Полный путь файла для показа: в регистре, в каком он лежит на диске."""
    return _real_case(Path(row["root_path"]) / row["relative_path"], cache)


def _copy_name(row):
    """Имя, под которым скачивается копия человека."""
    path = Path(row["relative_path"])
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", f"{path.stem} (моя правка){path.suffix}")


def _missing_text(row, then):
    """Файла нет: что сделать, чтобы он появился на месте, и под каким именем."""
    name = Path(row["relative_path"]).name
    folder = _shown_file(row, {}).parent
    return (f"Файла «{name}» на прежнем месте нет. Скачайте копию («Скачать мою правку»: она сохранится как "
            f"«{_copy_name(row)}»), переименуйте её в «{name}» и положите в папку {folder}, {then}")


def _same_object(row, rows, cache):
    """Конфликты того же объекта; rows группируются один раз на построение списка."""
    key = ("same_object", row["object"])
    if key not in cache:
        for other in rows:
            cache.setdefault(("same_object", other["object"]), []).append(other)
    return cache[key]


def _listed_program_version(store, row, cache):
    """Версия программы для списка: (ревизия, {вид файла: хэш}); читается один раз на объект, ошибка
    запоминается так же, как успех. Кеш держит только хэши, не содержимое: построение списка не копит
    файлы всех объектов в памяти. Замена файла и запись по-прежнему читают версию заново (_program_version)."""
    import documents
    key = ("program", row["object"])
    if key not in cache:
        try:
            revision, files = _program_version(store, row)
            cache[key] = ((revision, {kind: documents._digest(data) for kind, data in files.items()}), None)
        except CoreError as ex:
            cache[key] = (None, ex)
    found, error = cache[key]
    if error is not None:
        raise error
    return found


def _conflict_item(store, row, rows, cache):
    """Строка общего списка; rows — все конфликты той же области. Повреждённая строка не роняет список.
    cache живёт одно построение списка: в нём регистр путей, группы объектов и версии программы."""
    import storage
    item = {"conflict_id": row["conflict_id"], "scope": row["scope"], "recording_id": row["recording_id"],
            "title": _real_case(row["location"], cache).name, "name": row["name"], "kind": row["kind"],
            "relative_path": row["relative_path"], "file": str(_shown_file(row, cache)),
            "observed_size": row["observed_size"], "preserved": row["preserved"], "status": "open",
            "can_replace": False, "can_retry": False, "problem": None,
            "copy_problem": None, "next_effect": None, "next_step": None}
    try:
        same_object = _same_object(row, rows, cache)
        status = "resolved" if row["resolved"] else "open"
        last_of_file = row["seq"] == max(other["seq"] for other in same_object
                                         if other["relative_path"] == row["relative_path"])
        equal_to_program = False
        if not row["resolved"] and not row["kind"].startswith("archive_"):
            program = _listed_program_version(store, row, cache)[1]   # повреждена — строка «повреждена»
            if not row["preserved"]:    # копии нет: конфликт закрывается, только если файл уже равен версии программы
                equal_to_program = _file_at(store, row, row["relative_path"])[1] == program.get(row["kind"])
        if row["resolved"] and last_of_file and _queue_unfinished(store, row):
            program = _listed_program_version(store, row, cache)[1].get(row["kind"])
            if program is not None and _file_at(store, row, row["relative_path"])[1] != program:
                status = "not_written"
        item.update(status=status,
                    can_replace=status == "open" and (row["preserved"] or equal_to_program)
                    and not row["kind"].startswith("archive_"),
                    can_retry=status == "not_written" and not any(not other["resolved"] for other in same_object))
        if status in ("open", "not_written"):    # у истории копии не читаются: их проверит скачивание
            item["copy_problem"] = _copy_problem(store, row)
            if item["copy_problem"]:
                item["can_replace"] = False
        if status == "not_written":
            item["next_effect"], item["next_step"] = _next_step(store, row, same_object, item["copy_problem"])
    except (CoreError, storage.StorageError, sqlite3.Error, ValueError, KeyError, TypeError, OSError) as ex:
        item.update(status="damaged", can_replace=False, can_retry=False,
                    problem=str(ex) if isinstance(ex, CoreError) else
                    "Эту строку не удалось проверить, замена и повтор для неё выключены. " + explain_error(ex))
    return item


def list_all_conflicts():
    """Открытые и решённые конфликты всей базы для экрана; ничего не пишет и не требует запроса."""
    import paths
    import recordings
    import settings
    try:
        backend = settings.recording_backend()
    except ValueError:
        backend = None
    if not Path(paths.STATE_DB).is_file():
        reason = ("Базы записей ещё нет: конфликтов нет, показывать нечего.")
        if backend != "sqlite":
            reason += (" Программа сейчас хранит записи в файлах вне базы: такие файлы она перезаписывает "
                       "без вопроса и затирает вашу правку молча, защиты этого раздела для них нет.")
        return {"available": False, "items": [], "reason": reason, "backend": backend}
    store = recordings.open_store()
    try:
        store.current_identity()
        by_scope = {scope: _conflict_rows(store, scope) for scope in CONFLICT_SCOPES}
        cache = {}
        return {"available": True, "reason": None, "backend": backend,
                "items": [_conflict_item(store, row, rows, cache) for rows in by_scope.values() for row in rows]}
    finally:
        store.close()


def _refusal(row, state):
    """Почему заменять нельзя (None — можно)."""
    if row["resolved"]:
        return "Этот конфликт уже решён: заменять нечего. Копию можно скачать."
    if row["kind"].startswith("archive_"):
        return ("Это архивная копия прежней версии: программа записывает её один раз и не меняет, поэтому "
                "заменить её здесь нельзя. Свою версию можно скачать. Решить такой конфликт на этом экране нельзя.")
    if not row["preserved"] and state != "program":
        return ("Файл больше 8 МиБ, поэтому программа не сохранила его копию и не заменяет его: он остаётся "
                "на диске как есть. Заменить его здесь нельзя. Когда файл снова станет равен версии программы "
                "(её копия в истории записи), здесь можно будет закрыть конфликт: файл при этом не изменится.")
    if state == "missing":
        return _missing_text(row, "обновите список, и замена станет доступна.")
    if state == "unavailable":
        return ("Файл сейчас нельзя прочитать: он занят другой программой, недоступен или на его месте папка "
                "либо ссылка. Освободите его и обновите список.")
    return None


def _also_replaced(store, row, program):
    """Другие файлы объекта, которые это подтверждение запишет заново по прежнему решению:
    они равны решённой копии и не равны версии программы."""
    import documents
    decided = {}
    for other in _conflict_rows(store, row["scope"]):
        if (other["object"] == row["object"] and other["revision"] == row["revision"] and other["resolved"]
                and other["kind"] != row["kind"] and other["kind"] in program):
            decided[other["kind"]] = other    # последнее решение вида: так выбирает и запись файлов
    found = []
    for kind, other in decided.items():
        state, digest = _file_at(store, other, other["relative_path"])
        if (state == "ok" and digest == other["observed_hash"] and digest != documents._digest(program[kind])
                and not _copy_problem(store, other)):
            found.append(other["relative_path"])
    return sorted(found)


def _waits_for(row, rows):
    """Другие открытые конфликты того же объекта: пока они есть, подтверждение файл не меняет."""
    return sorted({Path(other["relative_path"]).name for other in rows
                   if other["object"] == row["object"] and other["conflict_id"] != row["conflict_id"]
                   and not other["resolved"]})


def _copy_problem(store, row):
    """Почему сохранённая копия непригодна; None — цела или её не сохраняли."""
    if not row["preserved"]:
        return None
    try:
        _observed_bytes(store, row)
    except CoreError as ex:
        return str(ex)
    return None


def _damaged_stops(store, row, rows, program, row_state):
    """Файлы с повреждённой копией принятого решения, на которых запись встанет, не дойдя до файла row.

    row_state — состояние файла row. Файлы документа сверяются все до первой замены в порядке `documents.WRITE_ORDER`:
    копия раньше row в этом порядке останавливает запись всегда, позже — только если row не изменён после решения
    (изменённый файл даёт новый конфликт раньше, чем сверка дойдёт до копии)."""
    import documents
    decided = {}
    for other in rows:
        if (other["object"] != row["object"] or not other["resolved"] or other["revision"] > row["revision"]
                or other["kind"] not in program):
            continue
        same_file = (other["revision"], other["relative_path"]) == (row["revision"], row["relative_path"])
        if same_file and not row["resolved"]:
            continue    # новое решение заменит прежнее решение по этому же файлу
        decided[(other["revision"], other["relative_path"])] = other
    table, column, revision = (("document_outbox", "recording_id", "document_revision") if row["scope"] == "document"
                               else ("conspect_outbox", "conspect_id", "revision"))
    found = set()
    for other in decided.values():
        if row["scope"] == "document" and other["revision"] == row["revision"] and row_state != "same":
            order = documents.WRITE_ORDER
            if order.index(other["kind"]) > order.index(row["kind"]):
                continue
        state, digest = _file_at(store, other, other["relative_path"])
        if state != "ok" or digest != other["observed_hash"] or digest == documents._digest(program[other["kind"]]):
            continue
        if store.db.execute(f"SELECT 1 FROM {table} WHERE {column}=? AND {revision}=? AND state!='synced'",
                            (row["object"][1], other["revision"])).fetchone() and _copy_problem(store, other):
            found.add(Path(other["relative_path"]).name)
    return sorted(found)


def _effect(row, state, waits, stops):
    """Что случится с файлом при записи по решению:
    replace — заменён; close — уже равен версии программы; recopy — изменён, будет новый конфликт;
    wait — ждёт другого конфликта; stopped — сейчас записать нельзя."""
    if state == "program":
        return "close"
    if waits:
        return "wait"
    if state in ("missing", "unavailable"):
        return "stopped"
    if state == "changed" and (row["scope"] == "conspect" or not stops):
        return "recopy"    # у свёртки внешние правки ищутся раньше проверки копий
    return "stopped" if stops else "replace"


def replacement_effect(recording_id, conflict_id, kind):
    """Для окна MCP: что произойдёт после «Да»; если заменять нельзя — отказ до окна. Ничего не записывает."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        row = _conflict_row(store, recording_id, conflict_id, kind)
        problem = _copy_problem(store, row)
        if problem:
            raise CoreError(problem + " Заменить такой файл нельзя: нужен отдельный ручной разбор.")
        if _file_at(store, row, row["relative_path"])[0] == "missing":
            raise CoreError(_missing_text(row, "затем повторите."))
        facts = _conflict_facts(store, row)
        if not facts["can_replace"]:
            raise CoreError(facts["reason"])
        blocked = bool(_unresolvable_wait(store, row, _conflict_rows(store, kind)))
        return {"effect": facts["effect"], "waits": facts["waits_for"], "stops": facts["stops_for"],
                "changed": facts["file_state"] == "changed", "blocked": blocked}
    finally:
        store.close()


def _conflict_facts(store, row):
    """Что известно о конфликте без чтения текста: файл, версия программы, можно ли заменить, снимок."""
    import documents
    archive = row["kind"].startswith("archive_")
    revision, program, problem = None, None, None
    if not archive:
        try:
            revision, program = _program_version(store, row)
        except CoreError as ex:
            problem = str(ex)
    program_hash = None if program is None else documents._digest(program[row["kind"]])
    found, digest = _file_at(store, row, row["relative_path"])
    state = (found if found != "ok" else "program" if digest == program_hash
             else "same" if digest == row["observed_hash"] else "changed")
    reason = _refusal(row, state) or problem
    also = [] if reason else _also_replaced(store, row, program)
    rows = [] if reason else _conflict_rows(store, row["scope"])
    waits = [] if reason else _waits_for(row, rows)
    stops = [] if reason else _damaged_stops(store, row, rows, program, state)
    return {"program": None if program is None else program[row["kind"]], "file_state": state,
            "also_replaces": also, "waits_for": waits, "stops_for": stops,
            "effect": None if reason else _effect(row, state, waits, stops), "can_replace": reason is None, "reason": reason,
            "snapshot": None if reason else {
                "scope": row["scope"], "recording_id": row["recording_id"], "conflict_id": row["conflict_id"],
                "observed_hash": row["observed_hash"], "program_revision": revision,
                "program_hash": program_hash, "also_replaces": also, "waits_for": waits,
                "stops_for": stops}}


def _lines(text):
    """Строки по CRLF, CR, LF (не str.splitlines: она прячет U+2028 и разрыв страницы)."""
    found = re.split(r"\r\n|\r|\n", text)
    return found[:-1] if found[-1] == "" else found


def _decode_copy(data):
    """Текст копии для сравнения: (текст, кодировка, точно ли). Кодировку без признака не угадываем."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), "utf-16", True
        except UnicodeDecodeError:
            pass
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="surrogateescape"), "unknown", False
    if "\x00" in text:    # нулевые знаки в тексте — признак UTF-16 без метки: кодировку не угадываем
        return text, "unknown", False
    return text, "utf-8", True


def _show_text(text):
    """Текст для показа: недекодируемые байты и нулевые знаки — знаком замены."""
    return re.sub("[\udc80-\udcff\x00]", "�", text)


def _anchors(a, b):
    """Строки, которые встречаются по одному разу и в a, и в b, в одном порядке: опоры сравнения."""
    count_a, count_b = collections.Counter(a), collections.Counter(b)
    place = {line: j for j, line in enumerate(b) if count_b[line] == 1}
    pairs = [(i, place[line]) for i, line in enumerate(a) if count_a[line] == 1 and line in place]
    tails, ends, back = [], [], []
    for k, (_, j) in enumerate(pairs):
        at = bisect.bisect_left(tails, j)
        back.append(ends[at - 1] if at else None)
        if at == len(tails):
            tails.append(j)
            ends.append(k)
        else:
            tails[at], ends[at] = j, k
    chain, k = [], ends[-1] if ends else None
    while k is not None:
        chain.append(pairs[k])
        k = back[k]
    return chain[::-1]


def _diff_lines(a, b, *, details=None):
    """Строки a, которых нет на своём месте в b (gone), и строки b, которых нет в a (came).

    Сравнение по порядку: перестановка и перенос строк — потеря. Значение — номер строки напротив
    при замене одной строки другой или None. Общий бюджет — первичный поиск опор по входу плюс
    CONFLICT_DIFF_WORK для уточнения. Что в него не уложилось, считается целиком потерей и
    появлением (оценка сверху); details сообщает, что результат огрублён."""
    gone, came = {}, {}
    budget = CONFLICT_DIFF_WORK + 5 * (len(a) + len(b))
    if details is not None:
        details["approximate"] = False
    work = [(0, len(a), 0, len(b))]
    while work:
        a1, a2, b1, b2 = work.pop()
        while a1 < a2 and b1 < b2 and a[a1] == b[b1]:
            a1, b1 = a1 + 1, b1 + 1
        while a1 < a2 and b1 < b2 and a[a2 - 1] == b[b2 - 1]:
            a2, b2 = a2 - 1, b2 - 1
        cells = (a2 - a1) * (b2 - b1)
        if a1 < a2 and b1 < b2 and cells <= CONFLICT_DIFF_CELLS and cells <= budget:
            budget -= cells
            matcher = difflib.SequenceMatcher(None, a[a1:a2], b[b1:b2], autojunk=False)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                paired = min(i2 - i1, j2 - j1)
                for k in range(i2 - i1):
                    gone[a1 + i1 + k] = b1 + j1 + k if k < paired else None
                for k in range(j2 - j1):
                    came[b1 + j1 + k] = a1 + i1 + k if k < paired else None
            continue
        anchors = []
        if a1 < a2 and b1 < b2:
            cost = 5 * ((a2 - a1) + (b2 - b1))     # поиск опор тоже работа, и вложенные опоры не бесплатны
            if cost <= budget:
                anchors = _anchors(a[a1:a2], b[b1:b2])
            budget -= cost
        if not anchors:
            if details is not None and a1 < a2 and b1 < b2:
                details["approximate"] = True
            gone.update((i, None) for i in range(a1, a2))
            came.update((j, None) for j in range(b1, b2))
            continue
        last_i, last_j = 0, 0
        for i, j in anchors:
            work.append((a1 + last_i, a1 + i, b1 + last_j, b1 + j))
            last_i, last_j = i + 1, j + 1
        work.append((a1 + last_i, a2, b1 + last_j, b2))
    return gone, came


_ZERO_WIDTH = {ord(c): None for c in "​‌‍⁠"}
_WIDE_SPACES = {ord(c): " " for c in "       "}


def _invisible_note(one, other):
    """Если строки различаются только невидимыми знаками, сказать об этом словами; иначе пустая строка."""
    kinds = []
    for label, fix in (("метка порядка байтов (BOM)", lambda t: t.replace("﻿", "")),
                       ("знаки нулевой ширины", lambda t: t.translate(_ZERO_WIDTH)),
                       ("неразрывные пробелы вместо обычных", lambda t: t.translate(_WIDE_SPACES)),
                       ("пробелы или табуляция в конце строки", lambda t: t.rstrip(" \t"))):
        fixed = fix(one), fix(other)
        if fixed != (one, other):
            kinds.append(label)
        one, other = fixed
    return "Различие только в невидимых знаках: " + ", ".join(kinds) + "." if kinds and one == other else ""


def _view(line, partner):
    """Строка для показа и признак, что она сокращена. Длинную строку показываем вокруг места различия."""
    line = _show_text(line)
    if len(line) <= CONFLICT_SHOWN_CHARS:
        return line, False
    if partner is None:
        return line[:CONFLICT_SHOWN_CHARS] + "…", True
    same = len(os.path.commonprefix([line, _show_text(partner)]))
    start = max(0, min(same - 80, len(line) - CONFLICT_SHOWN_CHARS))
    end = start + CONFLICT_SHOWN_CHARS
    return ("…" if start else "") + line[start:end] + ("…" if end < len(line) else ""), True


def _present(lines, marks, other):
    """Первые строки различий для показа, пометки к ним и признак, что показано всё."""
    order = sorted(marks)
    shown, notes, whole = [], [], len(order) <= CONFLICT_SHOWN_LINES
    for index in order[:CONFLICT_SHOWN_LINES]:
        partner = None if marks[index] is None else other[marks[index]]
        text, cut = _view(lines[index], partner)
        shown.append(text)
        notes.append("" if partner is None else _invisible_note(lines[index], partner))
        whole = whole and not cut
    return shown, notes, whole


def _eol_names(text):
    plain = text.replace("\r\n", "")
    return [name for name, found in (("CRLF", "\r\n" in text), ("CR", "\r" in plain),
                                     ("LF", "\n" in plain)) if found]


def _file_notes(mine, theirs, encoding):
    """Слова о том, что в самих файлах отличается и не видно в строках: кодировка, BOM, концы строк."""
    notes = []
    if encoding == "utf-16":
        notes.append("Копия записана в кодировке UTF-16: программа показала её расшифрованной. "
                     "Скачивается копия без изменений, байт в байт.")
    elif encoding == "unknown":
        notes.append("Копия записана не в UTF-8 (или это UTF-16 без метки: в тексте есть нулевые знаки), "
                     "и по ней нельзя точно определить кодировку, поэтому показ неточный: непонятные знаки заменены на «�». Скачивается копия без изменений, байт в байт.")
    if theirs is not None:
        if mine.startswith("﻿") != theirs.startswith("﻿"):
            notes.append("В начале файла есть невидимая метка BOM: " + (
                "она есть в вашей копии и её нет в версии программы." if mine.startswith("﻿")
                else "она есть в версии программы и её нет в вашей копии."))
        ends, theirs_ends = _eol_names(mine), _eol_names(theirs)
        if ends != theirs_ends and mine and theirs:
            notes.append("Концы строк различаются: в вашей копии " + (", ".join(ends) or "нет") +
                         ", в версии программы " + (", ".join(theirs_ends) or "нет") + ".")
        if mine and theirs and mine.endswith(("\n", "\r")) != theirs.endswith(("\n", "\r")):
            notes.append("Различается конец файла: в одной из версий последняя строка не закрыта переводом строки.")
    return notes


def preview_conflict(recording_id, conflict_id, kind="document"):
    """Показать копию человека и сравнение с последней версией программы; ничего не записывает.

    kind — область конфликта: document или conspect."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        row = _conflict_row(store, recording_id, conflict_id, kind)
        facts = _conflict_facts(store, row)
        reason, can_replace, snapshot, copy_problem = facts["reason"], facts["can_replace"], facts["snapshot"], None
        try:
            observed = _observed_bytes(store, row)
        except CoreError as ex:    # копия в базе повреждена: показываем причину, заменять нельзя
            observed, copy_problem = None, str(ex)
            reason, can_replace, snapshot = copy_problem, False, None
        program, encoding, newlines_only, notes = facts["program"], None, False, []
        gone = came = {}
        mine = theirs = []
        text = None
        comparison = {"approximate": False}
        if observed is not None:
            text, encoding, _ = _decode_copy(observed)
            program_text = None if program is None else program.decode("utf-8", errors="surrogateescape")
            if program_text is not None:
                mine, theirs = _lines(text), _lines(program_text)
                gone, came = _diff_lines(mine, theirs, details=comparison)
                newlines_only = text != program_text and mine == theirs
            notes = _file_notes(text, program_text, encoding)
            if comparison["approximate"]:
                notes.append("Сравнение огрублено: число строк — оценка сверху, а не точное число изменений. "
                             "В списки могли попасть совпадающие строки. Сохранённая копия доступна целиком.")
        waits = _unresolvable_wait(store, row, _conflict_rows(store, row["scope"])) if can_replace else None
        if waits:
            notes.append(waits)
        removed, removed_notes, whole_removed = _present(mine, gone, theirs)
        added, added_notes, whole_added = _present(theirs, came, mine)
        shown_text = None if text is None else _show_text(text)
        return {"conflict_id": conflict_id, "kind": row["kind"], "relative_path": row["relative_path"],
                "observed_hash": row["observed_hash"], "observed_size": row["observed_size"],
                "preserved": row["preserved"],
                "preview": None if shown_text is None else shown_text[:CONFLICT_PREVIEW_CHARS],
                "truncated": shown_text is not None and len(shown_text) > CONFLICT_PREVIEW_CHARS,
                "file": str(_shown_file(row, {})), "file_state": facts["file_state"],
                "removed": removed, "added": added, "removed_total": len(gone), "added_total": len(came),
                "removed_notes": removed_notes, "added_notes": added_notes,
                "shown_all": whole_removed and whole_added, "newlines_only": newlines_only,
                "encoding": encoding, "file_notes": notes, "copy_problem": copy_problem,
                "comparison_approximate": comparison["approximate"],
                "also_replaces": facts["also_replaces"] if can_replace else [],
                "waits_for": facts["waits_for"] if can_replace else [], "waits_blocked": bool(waits),
                "stops_for": facts["stops_for"] if can_replace else [],
                "effect": facts["effect"] if can_replace else None,
                "can_replace": can_replace, "reason": reason, "snapshot": snapshot}
    finally:
        store.close()


def conflict_copy(recording_id, conflict_id, kind):
    """Сохранённые байты копии человека целиком и имя для скачивания; ничего не меняет и не читает файл."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        row = _conflict_row(store, recording_id, conflict_id, kind)
        data = _observed_bytes(store, row)
        if data is None:
            raise CoreError("Копия не сохранена: файл был больше 8 МиБ. Заберите его прямо из папки записи.")
        return {"data": data, "name": _copy_name(row)}
    finally:
        store.close()


def _is_journal_stop(ex):
    """Запись свёртки остановлена журналом: файлы свёртки менялись после сбоя, и программа не продолжает сама."""
    return isinstance(ex, ValueError) and "ручное восстановление" in str(ex)


def _journal_blockers(store, row):
    """Файлы, из-за которых журнал свёртки не даёт продолжить запись: (имя, хэши журнала, имя журнала).

    Журнал один на запись, поэтому мешать может и файл другой свёртки из той же папки."""
    import conspect
    import documents
    import recordings
    found = []
    try:
        binding = recordings.document_binding(store, row["recording_id"])
        root = documents._authorized_root(store, binding["root_id"], binding["root"])
        json_relative = Path(binding["paths"]["json"])
        journal, exists = documents._read_only_target(root, str(json_relative) + ".conspect.pending")
        if not exists:
            return found
        record = json.loads(journal.read_text(encoding="utf-8"))
        if record.get("version") != 1 or not isinstance(record.get("files"), list):
            return found
        for item in record["files"]:
            name = item["name"]
            if (not isinstance(name, str) or Path(name).name != name
                    or not name.endswith((".conspect.json", ".conspect.md"))):
                return []
            relative = (json_relative.parent / name).as_posix()
            path, exists = documents._read_only_target(root, relative)
            digest = conspect._digest(path.read_bytes()) if exists else None
            hashes = (item["old_hash"], item["new_hash"])
            if digest not in hashes:
                found.append((relative, hashes, journal.name, binding["root_id"]))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return found


def _journal_reason(store, row):
    """Почему журнал остановил запись: какой файл мешает и есть ли выход на этом экране."""
    head = ("Запись файлов свёртки остановлена журналом: после прошлого сбоя файлы свёртки изменились, и программа "
            "не продолжает сама, чтобы не затереть вашу правку. Сначала сохраните свою новую правку отдельно "
            "(скопируйте файл в другое место). ")
    blockers = _journal_blockers(store, row)
    if not blockers:
        return head + ("Какой файл мешает, определить не удалось, и выхода на этом экране нет: нужно "
                       "ручное восстановление рядом с файлами записи.")
    rows = _conflict_rows(store, row["scope"])
    parts = []
    for relative, hashes, journal, root_id in blockers:
        name = Path(relative).name
        copies = []
        for other in reversed(rows):
            if (other["recording_id"] == row["recording_id"] and other["root_id"] == root_id
                    and os.path.normcase(str(Path(other["relative_path"]))) == os.path.normcase(str(Path(relative)))
                    and other["preserved"] and other["observed_hash"] in hashes):
                try:
                    if _observed_bytes(store, other) is not None:
                        copies.append(other)
                except CoreError:
                    pass
        if copies:
            parts.append(f"Мешает файл «{name}» (журнал «{journal}»): верните на его место его сохранённую копию "
                         f"(скачайте её как «{_copy_name(copies[0])}» и переименуйте в «{name}») и нажмите «Повторить».")
        else:
            parts.append(f"Мешает файл «{name}» (журнал «{journal}»): он изменён после сбоя, а подходящей его копии "
                         "у программы нет, поэтому выхода на этом экране нет: нужно ручное восстановление "
                         "этого файла и журнала вне экрана.")
    return head + " ".join(parts)


def _failure_reason(store, row, ex):
    """Настоящая причина, по которой файл не записан, когда она известна по ошибке записи."""
    import documents
    import storage
    if _is_journal_stop(ex):
        return _journal_reason(store, row)
    if isinstance(ex, documents.DamagedCopy):
        return str(ex) + " После изменения файла нажмите «Повторить»."
    if isinstance(ex, storage.Busy) or (isinstance(ex, sqlite3.OperationalError)
                                        and ("locked" in str(ex).lower() or "busy" in str(ex).lower())):
        return ("База записей была занята другой операцией. Решение принято; подождите немного "
                "и нажмите «Повторить».")
    if isinstance(ex, OSError):
        return ("Файл не удалось записать: возможно, он открыт в другой программе или недоступен "
                f"(причина: {ex}). Закройте программы, где он открыт, и нажмите «Повторить».")
    return ("При записи файла программа остановилась с ошибкой "
            f"(причина: {type(ex).__name__}: {ex}). Нажмите «Повторить»; если ошибка повторится, сообщите о ней.")


def _file_words(row, digest):
    """Что сказать о файле по его сверенному содержимому, а не по догадке о том, дошла ли запись."""
    if digest == row["observed_hash"]:
        return " Файл по-прежнему такой, как ваша правка: версия программы в него не записана."
    return (" Файл сейчас не совпадает ни с вашей сохранённой правкой, ни с последней версией программы. "
            "Причину изменения по этой проверке установить нельзя. Сверьте его содержимое.")


def _unresolvable_wait(store, row, rows):
    """Открытый конфликт того же объекта, который на экране не решить, но запись этого файла его ждёт."""
    stuck = sorted({Path(other["relative_path"]).name for other in rows
                    if other["object"] == row["object"] and other["conflict_id"] != row["conflict_id"]
                    and not other["resolved"] and (not other["preserved"] or other["kind"].startswith("archive_")
                                                   or _copy_problem(store, other))})
    if not stuck:
        return None
    names = ", ".join(f"«{name}»" for name in stuck)
    return (f"Запись этого файла ждёт решения по другому файлу той же записи: {names}. Но решить его на этом экране "
            "нельзя (файл больше 8 МиБ, это архив или его копия повреждена), нужен отдельный ручной разбор. До него "
            "этот файл останется как есть, даже если решение о замене принято.")


def _next_step(store, row, rows, problem):
    """Карточка принятого, но не записанного решения: (что будет с файлом, слова человеку)."""
    import documents
    program = _program_version(store, row)[1]
    found, digest = _file_at(store, row, row["relative_path"])
    state = found if found != "ok" else "same" if digest == row["observed_hash"] else "changed"
    waits = _waits_for(row, rows)
    stops = _damaged_stops(store, row, rows, program, state)
    effect = _effect(row, state, waits, stops)
    waiting = ""
    if waits:
        waiting = _unresolvable_wait(store, row, rows) or (
            "Запись этого файла ждёт решения по другому файлу той же записи (он среди ждущих решения).")
    if problem:
        if state == "missing":
            text = ("Файла на прежнем месте нет, а его сохранённая копия повреждена, скачать её нельзя: "
                    "нужен ручной разбор вне экрана.")
        elif state == "same":
            text = ("Сохранённая копия повреждена, поэтому файл заменён не будет"
                    + (", даже когда ожидание другого файла кончится. " if waits else ". ") + documents.DAMAGED_EXIT)
        elif state == "changed":
            text = ("Сохранённая копия повреждена, но файл уже изменён. Заменён он не будет: когда запись дойдёт "
                    "до него, программа покажет новый конфликт, а если файл не больше 8 МиБ, сохранит его новую копию.")
        else:
            text = "Сохранённая копия повреждена, а файл сейчас не читается: освободите его и обновите список."
        return effect, f"{waiting} {text}".strip()
    if waits:
        return effect, (waiting + " Если тот файл будет решён, программа допишет этот, если вы его с тех пор "
                        "не меняли; если меняли, появится новый конфликт.")
    if state == "missing":
        return effect, _missing_text(row, "и нажмите «Повторить».")
    if state == "unavailable":
        return effect, "Файл сейчас нельзя прочитать, записать его программа не может. Освободите его и повторите."
    if effect == "recopy":
        return effect, ("Файл изменён после принятого решения, поэтому заменён не будет: при повторе программа "
                        "покажет новый конфликт, а если файл не больше 8 МиБ, сохранит его новую копию.")
    if effect == "stopped":
        return effect, ("Этот файл не будет записан, пока не разобрана повреждённая копия: "
                        + ", ".join(f"«{name}»" for name in stops) + ". Выход указан в карточке того файла.")
    return effect, ("Принятое решение программа может довести и сама, при любой следующей записи файлов этой записи, "
                    "не спрашивая снова, если файл не меняли и другие файлы записи не помешают.")


def _file_outcome(store, row, failure):
    """Итог записи по самому файлу: что в нём лежит сейчас."""
    import documents
    state, digest = _file_at(store, row, row["relative_path"])
    if digest == documents._digest(_program_version(store, row)[1][row["kind"]]):
        return {"outcome": "replaced", "new_conflict_id": None, "reason": None}
    rows = [other for other in _conflict_rows(store, row["scope"]) if other["object"] == row["object"]]
    fresh = [other for other in rows if not other["resolved"] and other["seq"] > row["seq"]
             and other["relative_path"] == row["relative_path"]]
    if fresh:
        return {"outcome": "changed", "new_conflict_id": fresh[-1]["conflict_id"], "reason": None}
    waiting = any(not other["resolved"] for other in rows)
    problem = _copy_problem(store, row) if waiting or state == "missing" else None
    if problem:
        reason = _next_step(store, row, rows, problem)[1]
    elif waiting:
        reason = _unresolvable_wait(store, row, rows) or (
            "Запись этого файла ждёт решения по другому файлу той же записи (он среди ждущих решения). "
            "Если тот файл будет решён, программа допишет этот, если вы его с тех пор не меняли; "
            "если меняли, появится новый конфликт.")
    elif state == "missing":
        reason = _missing_text(row, "и нажмите «Повторить».")
    elif state == "unavailable":
        reason = ("Файл сейчас нельзя прочитать: он занят другой программой, недоступен или на его месте папка "
                  "либо ссылка. Освободите его и нажмите «Повторить».")
    elif failure is not None:
        reason = _failure_reason(store, row, failure) + _file_words(row, digest)
    else:
        reason = ("Программа не дописала этот файл: очередь записи остановилась, возможно, на другой записи "
                  "из той же папки. Нажмите «Повторить»; если не помогает, закройте программы, где открыты "
                  "файлы этой записи." + _file_words(row, digest))
    return {"outcome": "not_written", "new_conflict_id": None, "reason": reason}


def _write_and_report(store, row):
    """Довести запись файлов объекта и сообщить итог по самому файлу, а не по очереди."""
    import recordings
    failure = None
    try:
        if row["scope"] == "document":
            recordings.materialize_recording(store, row["recording_id"])
        else:
            recordings.materialize_conspects(store, row["recording_id"], conspect_id=row["conspect_id"])
    except Exception as ex:    # решение уже записано, и ответ всегда по факту: что лежит в файле
        failure = ex
    try:
        return _file_outcome(store, row, failure)
    except Exception as ex:    # итог проверить нельзя, но решение записано: об этом и отвечаем, а не обрываем ответ
        return {"outcome": "not_written", "new_conflict_id": None,
                "reason": "Решение принято, но проверить, что лежит в файле, не удалось: "
                          + (str(ex) if isinstance(ex, CoreError) else explain_error(ex))
                          + " Откройте список позже: если файл не заменён, появится кнопка «Повторить»."}


def _replace_file(store, row, snapshot=None):
    """Записать решение «заменить файл версией программы» и довести запись. Файл проверяется до решения."""
    import recordings
    import storage
    facts = _conflict_facts(store, row)
    if not facts["can_replace"]:
        raise CoreError(facts["reason"])
    if snapshot is not None and facts["snapshot"] != snapshot:
        raise CoreError("Показанное сравнение устарело: конфликт или версия программы изменились. "
                        "Откройте сравнение заново.")
    try:     # без копии (файл больше 8 МиБ) закрывается лишь конфликт, файл уже равен версии программы и не меняется
        result = getattr(store, f"resolve_{row['scope']}_conflict")(
            row["conflict_id"], str(uuid.uuid4()), recordings.operation_context(store, "human"),
            **({} if row["preserved"] else {"without_copy": True}))
    except (storage.Busy, sqlite3.Error) as ex:
        raise CoreError("Решение не записано, файл не тронут. " + explain_error(ex)) from ex
    except (OSError, ValueError, storage.StorageError) as ex:
        raise CoreError("Решение не записано, файл не тронут. " + explain_error(ex)) from ex
    return {**result, **_write_and_report(store, row)}


def replace_conflict_file(snapshot):
    """Экран: заменить файл, только если человек видел сравнение (snapshot из preview_conflict) и оно не устарело."""
    import recordings
    if not isinstance(snapshot, dict):
        raise CoreError("Нет сравнения, которое видел человек: откройте его и проверьте, что пропадёт.")
    store = recordings.open_store()
    try:
        store.current_identity()
        row = _conflict_row(store, snapshot.get("recording_id"), snapshot.get("conflict_id"),
                            snapshot.get("scope"))
        return _replace_file(store, row, snapshot)
    finally:
        store.close()


def retry_conflict_file(recording_id, conflict_id, kind):
    """Повторить только запись файлов по уже принятому решению: без новых решений и ревизий."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        row = _conflict_row(store, recording_id, conflict_id, kind)
        item = _conflict_item(store, row, _conflict_rows(store, kind), {})
        if item["status"] == "damaged":
            raise CoreError(item["problem"])
        if item["status"] == "open":
            raise CoreError("Конфликт ещё не решён: повторять нечего. Сначала выберите, что делать с файлом.")
        if item["status"] == "resolved":
            raise CoreError("Повторять нечего: файл уже записан, или вы вернули на место свою копию, "
                            "или по этому файлу есть более новый конфликт.")
        return _write_and_report(store, row)
    finally:
        store.close()


def _damaged_copies_warning(copies):
    """Предупреждение о повреждённых копиях. Первая строка без путей: её получает и агент."""
    if not copies:
        return []
    return [f"Повреждённых сохранённых копий правок: {len(copies)}. В резерве они такие же повреждённые: "
            "вернуть по ним правку нельзя. Если файл с правкой ещё цел, сохраните его отдельно.",
            *(f"- «{item['relative_path']}» — конфликт {item['conflict_id']}" for item in copies)]


def _not_written_warning(rows):
    """Предупреждение о ревизиях, которые восстановление могло записать, но не записало (очередь pending).

    Конфликты и сбои, пришедшие из резерва, сюда не входят: они видны на экране конфликтов, как и раньше.
    Первая строка без путей и имён: её получает и агент."""
    if not rows:
        return []
    count = len({row["recording_id"] for row in rows})
    return [f"Восстановлено, но не все файлы записей сформированы: записей — {count}, ревизий — {len(rows)}. "
            "Если файл занят другой программой или защищён от записи, закройте его или снимите защиту; затем "
            "запустите sync_recording_documents для записи или повторите действие с ней — файлы допишутся.",
            *(f"- запись {row['recording_id']}, ревизия {row['document_revision']}: " +
              ("файл занят другой программой или защищён от записи" if row["error_code"] == "busy"
               else "ждёт предыдущую ревизию этой записи") for row in rows)]


def backup_store(destination):
    """Понятная операция резерва состояния: новый неактивный файл БД и отчёт проверки копии."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        report = store.backup(destination, allow_damaged_copies=True)
        warning = _damaged_copies_warning(report.get("damaged_copies"))
        return {"destination": str(destination), "report": report, **({"warning": warning} if warning else {})}
    finally:
        store.close()


def inspect_backup(backup_path):
    """Прочитать резервную копию только для чтения: корни, пользователи, число записей, повреждённые копии."""
    import storage
    store = storage.open_store_readonly(str(backup_path))
    try:
        roots = [dict(row) for row in store.db.execute(
            "SELECT root_id, path, requires_remap FROM allowed_roots ORDER BY path")]
        users = [dict(row) for row in store.db.execute(
            "SELECT user_id, display_name FROM local_users ORDER BY display_name")]
        recordings_count = store.db.execute("SELECT count(*) FROM recordings").fetchone()[0]
        damaged = storage.damaged_conflict_copies(store.db)
        return {"path": str(backup_path), "roots": roots, "users": users,
                "recordings": recordings_count, "damaged_copies": damaged,
                "warning": _damaged_copies_warning(damaged),
                "restore_guard_locked": bool(store.db.execute(
                    "SELECT locked FROM restore_guard WHERE id=1").fetchone()[0])}
    finally:
        store.close()


def restore_from_backup(backup_path, destination, root_mapping, user_id, *, replace=True):
    """Мастер восстановления: копия резерва получает новые корни и установку, документы
    формируются в новых корнях из сохранённых bytes. Живая база подменяется только
    при replace=True, её прежний файл сохраняется рядом; из резерва ничего не удаляется."""
    import shutil
    import documents
    import storage
    backup = Path(backup_path).expanduser()
    if not backup.is_file():
        raise CoreError("файл резерва не найден")
    destination = Path(destination).expanduser()
    if destination.exists() and destination.resolve() == backup.resolve():
        raise CoreError("нельзя восстанавливать поверх самого резерва")
    if not isinstance(root_mapping, dict) or not root_mapping:
        raise CoreError("нужно переназначить все документные корни")
    mapping = {str(key): str(value) for key, value in root_mapping.items()}
    for root_id, path in mapping.items():
        if not Path(path).is_dir():
            raise CoreError(f"путь документного корня не существует: {path}")
    staged = destination.with_name(destination.name + ".incomplete-" + uuid.uuid4().hex)
    try:
        shutil.copyfile(backup, staged)
    except OSError as ex:
        raise CoreError(f"не удалось скопировать резерв в {staged}: {ex}") from ex
    try:
        store = storage.open_store(staged)
        try:
            installation_id = str(uuid.uuid4())
            store.rebind_restored_copy(mapping, installation_id, user_id)
            materialized = []
            for root_id, path in mapping.items():
                materialized.extend(documents.materialize_pending(store, root_id, Path(path)))
            # PR 5б-2: занятый или защищённый от записи файл больше не обрывает восстановление — называем, что осталось
            not_written = [dict(row) for row in store.db.execute(
                "SELECT recording_id, document_revision, error_code FROM document_outbox "
                "WHERE state = 'pending' ORDER BY recording_id, document_revision")]
            validation = storage.validate_copy(staged, allow_damaged_copies=True)
        finally:
            store.close()
    except (storage.StorageError, ValueError) as ex:
        raise CoreError(f"восстановление не выполнено; подготовленный файл остался: {staged} "
                        f"({ex})") from ex
    previous_saved_as = None
    if destination.exists():
        if not replace:
            raise CoreError(f"файл {destination} уже существует; укажите другой путь")
        previous_saved_as = str(destination.with_name(
            destination.name + ".before-restore-" + time.strftime("%Y%m%d-%H%M%S")))
        shutil.copy2(destination, previous_saved_as)
    try:
        os.replace(staged, destination)
    except OSError as ex:
        raise CoreError(f"не удалось опубликовать восстановленную базу (возможно, программа "
                        f"запущена — закройте её и повторите); файл остался: {staged}") from ex
    warning = _not_written_warning(not_written) + _damaged_copies_warning(validation.get("damaged_copies"))
    return {"destination": str(destination), "installation_id": installation_id,
            "previous_saved_as": previous_saved_as, "validation": validation,
            "materialized": materialized, "not_written": not_written, **({"warning": warning} if warning else {})}


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


def find_recording_results(source_path):
    """Найти SQLite-записи исходника независимо от файлов истории задач."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        return recordings.find_recordings(store, source_path)
    finally:
        store.close()


def read_recording_result(recording_id):
    """Прочитать текущую SQLite-запись по устойчивому ID без job."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        result = recordings.load_recording(store, recording_id)
        result["recording_ref"] = "recording:" + recording_id
        return result
    finally:
        store.close()


def _recording_reference(reference):
    import storage
    try:
        result = read_recording_result(reference[len("recording:"):])
        return {"state": "done", "json": result["json"], "markdown": result["markdown"],
                "recording_id": result["recording_id"], "document_sync": result["document_sync"]}
    except (OSError, ValueError, KeyError, storage.StorageError) as ex:
        raise CoreError(f"SQLite-запись недоступна: {ex}") from ex


def sync_recording_documents(recording_id):
    """Повторить безопасную материализацию после commit или перезапуска."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        recordings.materialize_recording(store, recording_id)
        binding = recordings.document_binding(store, recording_id)
        path = binding["root"] / binding["paths"]["json"]
        conspects = recordings.refresh_conspects(store, recording_id, path)
        return {"recording_id": recording_id, "document_revision": binding["document_revision"],
                "document_sync": recordings.document_sync_state(store, recording_id, binding["document_revision"]),
                "conspects": conspects}
    finally:
        store.close()


def prepare_answer_confirmation(recording_id, answer_ids):
    """Снимок и точное содержимое ответов для явного подтверждения человеком."""
    import recordings
    import review_pack
    import unicodedata
    if (not isinstance(answer_ids, list) or not 1 <= len(answer_ids) <= 8 or
            any(not isinstance(ident, str) or not ident or len(ident) > 80 for ident in answer_ids) or
            len(set(answer_ids)) != len(answer_ids)):
        raise CoreError("выберите от 1 до 8 разных ответов")
    store = recordings.open_store()
    try:
        store.current_identity()
        binding = recordings.document_binding(store, recording_id)
        path = binding["root"] / binding["paths"]["json"]
        state = recordings.read_review_pack_state(path)
        manifest = state.get("manifest") or {}
        items = {item["id"]: item for item in manifest.get("items") or []}
        applied = review_pack._applied_map(state)
        conflicted = {item.get("id") for item in state.get("conflicts") or []}
        lines = []
        for ident in answer_ids:
            answer = state["answers"].get(ident)
            item = items.get(ident)
            if not answer or not item or answer.get("actor_kind") == "human":
                raise CoreError("ответ отсутствует или уже имеет человеческое происхождение")
            if ident in conflicted or ident in applied:
                raise CoreError("конфликтующий или ранее применённый ответ нужно проверить в локальной карточке")
            kind = answer.get("kind")
            question = item.get("question") or "Как правильно?"
            source_text = item.get("text") or ""
            if kind in ("option", "recommended"):
                option = answer.get("option")
                options = item.get("options") or []
                if type(option) is not int or not 0 <= option < len(options):
                    raise CoreError("вариант ответа недействителен")
                text = options[option].get("label", "")
                decision = json.dumps(options[option].get("decision") or {}, ensure_ascii=False, sort_keys=True)
            elif kind == "free":
                raise CoreError("свободный ответ неизвестного происхождения нужно ввести заново в карточке")
            elif kind in ("revert", "human"):
                text = "отменить ответ" if kind == "revert" else "оставить человеку"
                decision = kind
            else:
                raise CoreError("этот тип ответа нельзя подтвердить")
            values = (ident, question, source_text, text, decision)
            if (any(not isinstance(value, str) for value in values) or not text or
                    any(unicodedata.category(ch).startswith("C") for value in values for ch in value)):
                raise CoreError("ответ нельзя полностью и однозначно показать для подтверждения")
            detail = [f"{ident}: {question}; место: {source_text}",
                      f"Ответ: {text}; действие: {decision}"]
            if any(len(line) > 200 for line in detail):
                raise CoreError("ответ слишком длинный для полного подтверждения; используйте карточку")
            lines.extend(detail)
        if sum(len(line) + 1 for line in lines) > 900:
            raise CoreError("ответы не помещаются в окно подтверждения; выберите меньше")
        encoded = _answer_confirmation_hash(state, answer_ids)
        return {"recording_id": recording_id, "answer_ids": list(answer_ids),
                "state_hash": hashlib.sha256(encoded).hexdigest(), "lines": lines}
    except (OSError, ValueError, KeyError) as ex:
        raise CoreError(f"не удалось подготовить подтверждение ответов: {ex}") from ex
    finally:
        store.close()


def _answer_confirmation_hash(state, answer_ids):
    ids = set(answer_ids)
    manifest = state.get("manifest") or {}
    relevant_items = [item for item in manifest.get("items") or [] if item.get("id") in ids]
    conflicts = [row for row in state.get("conflicts") or []
                 if row.get("id") in ids or ids.intersection(row.get("cards") or [])]
    value = {"version": state.get("version"), "answers": {
                 aid: state.get("answers", {}).get(aid) for aid in sorted(ids)},
             "items": relevant_items,
             "applied_answers": {aid: (state.get("applied_answers") or {}).get(aid) for aid in sorted(ids)},
             "conflicts": conflicts}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def confirm_review_answers(snapshot):
    """Зафиксировать отдельного подтверждающего, не меняя автора исходного ответа."""
    import recordings
    import review_pack
    current = prepare_answer_confirmation(snapshot["recording_id"], snapshot["answer_ids"])
    if current != snapshot:
        raise CoreError("ответы изменились после запроса подтверждения")
    store = recordings.open_store()
    try:
        binding = recordings.document_binding(store, snapshot["recording_id"])
        path = binding["root"] / binding["paths"]["json"]
    finally:
        store.close()
    with recordings.operation_for_document(path, "human"):
        state = recordings.read_review_pack_state(path)
        encoded = _answer_confirmation_hash(state, snapshot["answer_ids"])
        if hashlib.sha256(encoded).hexdigest() != snapshot["state_hash"]:
            raise CoreError("ответы изменились после запроса подтверждения")
        context = recordings.context_for_document(path)
        for ident in snapshot["answer_ids"]:
            state["answers"][ident]["confirmation_context"] = context
        review_pack._write_answers(path, state)
    return {"recording_id": snapshot["recording_id"], "confirmed": snapshot["answer_ids"]}


def list_conspect_conflicts(recording_id):
    import recordings
    import storage
    storage._uuid(recording_id)
    store = recordings.open_store()
    try:
        store.current_identity()
        return [dict(row) for row in store.db.execute(
            "SELECT f.conflict_id, f.conspect_id, c.name, f.revision, f.kind, f.relative_path, "
            "f.observed_hash, f.observed_size, (f.observed_bytes IS NOT NULL) AS preserved "
            "FROM conspect_conflicts f JOIN conspects c USING(conspect_id) "
            "WHERE c.recording_id=? AND f.resolution_operation_id IS NULL ORDER BY f.rowid",
            (recording_id,))]
    finally:
        store.close()


def list_document_conflicts(recording_id):
    import recordings
    import storage
    storage._uuid(recording_id)
    store = recordings.open_store()
    try:
        store.current_identity()
        return [dict(row) for row in store.db.execute(
            "SELECT conflict_id, document_revision, kind, relative_path, observed_hash, observed_size, "
            "(observed_bytes IS NOT NULL) AS preserved FROM document_conflicts "
            "WHERE recording_id=? AND resolution_operation_id IS NULL ORDER BY rowid", (recording_id,))]
    finally:
        store.close()


def _resolve_for_agent(scope, recording_id, conflict_id):
    """MCP: после подтверждения в окне заменить файл версией программы. Ответ и ошибки прежние;
    новое одно: если файла на месте нет, отказ до записи решения."""
    import documents
    import recordings
    import storage
    if scope == "document":
        owner = "SELECT recording_id FROM document_conflicts WHERE conflict_id=?"
    else:
        owner = ("SELECT c.recording_id FROM conspect_conflicts f JOIN conspects c USING(conspect_id) "
                 "WHERE f.conflict_id=?")
    store = recordings.open_store()
    try:
        store.current_identity()
        found = store.db.execute(owner, (conflict_id,)).fetchone()
        if found is None or found["recording_id"] != recording_id:
            raise CoreError("конфликт не найден для этой записи")
        row = _conflict_row(store, recording_id, conflict_id, scope)
        if _file_at(store, row, row["relative_path"])[0] == "missing":
            raise CoreError(_missing_text(row, "затем повторите."))
        extra = {}
        if not row["preserved"]:       # копии нет: закрыть можно, только если файл уже равен версии программы
            facts = _conflict_facts(store, row)
            if not facts["can_replace"]:
                raise CoreError(facts["reason"])
            extra = {"without_copy": True}
        if scope == "document":
            result = store.resolve_document_conflict(
                conflict_id, str(uuid.uuid4()), recordings.operation_context(store, "human"), **extra)
            completed = recordings.materialize_recording(store, recording_id)
            failure = documents.unsynced(store, recording_id, result["document_revision"], completed)
            result["materialized"] = failure is None
            if failure is not None:       # решение принято, файл не дописан: исход словами, а не голое false
                result["outcome"], result["reason"] = failure.status, str(failure)
        else:
            result = store.resolve_conspect_conflict(
                conflict_id, str(uuid.uuid4()), recordings.operation_context(store, "human"), **extra)
            completed = recordings.materialize_conspects(store, recording_id, conspect_id=result["conspect_id"])
            result["materialized"] = (result["conspect_id"], result["revision"]) in completed
        return result
    except (OSError, ValueError, storage.StorageError) as ex:
        raise CoreError(f"конфликт не разрешён или формирование не завершено: {ex}") from ex
    finally:
        store.close()


def resolve_document_conflict(recording_id, conflict_id):
    return _resolve_for_agent("document", recording_id, conflict_id)


def resolve_conspect_conflict(recording_id, conflict_id):
    return _resolve_for_agent("conspect", recording_id, conflict_id)


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
                 for it in data.get("review", []) if it.get("status", "open") in ("open", "human")]
    return P.build(jp, items)


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
                "proposed_delete", "vocab_suggestions"):
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
    """Применить ответы к указанной паре файлов готовой задачи."""
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
                guard = _answer_revision_conflicts(initial, manifest, state, source, sqlite=True)
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
                    if _answer_revision_conflicts(current, manifest, state, original, sqlite=True) != guard:
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
    with conspect.write_lock(jp):
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


def _answer_revision_conflicts(current, manifest, state, source, *, sqlite):
    import review_pack
    import transcript
    applied = review_pack._applied_map(state)
    answers = state.get("answers") or {}

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
        if not answer or aid in applied:
            continue
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
    applied_before = P._applied_map(P.answers_state(jp, {"answers": {}}))
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
            if aid in result.get("successful_ids", []) and aid not in rep.get("human", []):
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
    successful.difference_update(rep.get("human", []))
    rep["applied_from_answers"] = sum(1 for d in decisions if d.get("id") in successful)
    snapshots = {aid: snapshot for aid, snapshot in snapshots.items() if aid in successful}
    successful.update(undone)
    apply_conflicts = {row.get("id") for row in report.get("conflicts") or [] if row.get("id")}
    free_report = rep.get("free_text") or {}
    free_conflicts = {aid for row in free_report.get("conflicts") or [] for aid in row.get("cards") or []}
    free_not_found = set(free_report.get("not_found") or [])
    outcomes = {}
    for aid in requested_ids:
        if aid in successful:
            outcomes[aid] = "reverted" if aid in undone else "applied"
        elif aid in revision_conflicts:
            outcomes[aid] = "apply_conflict"
        elif aid in blocked:
            outcomes[aid] = "blocked_preflight"
        elif aid in undo_failed:
            outcomes[aid] = "undo_conflict"
        elif aid in (report.get("already_decided") or []):
            outcomes[aid] = "already_decided"
        elif aid in rep.get("human", []):
            outcomes[aid] = "left_for_human"
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
    stored = P.answers_state(jp, {"answers": {}}).get("answers") or {}
    undo_conflict_ids = {row.get("id") for row in rep.get("undo_conflicts") or []}
    for aid in stored:
        if aid in rep.get("source_conflict_ids", []):
            outcomes[aid] = "source_conflict"
        elif aid in outcomes:
            continue
        elif aid in rep.get("unchanged", []):
            outcomes[aid] = "applied"
        elif aid in rep.get("kept_previous", []):
            outcomes[aid] = "kept_previous"
        elif aid in rep.get("pending_new", []):
            outcomes[aid] = "pending_new"
        elif aid in undo_conflict_ids:
            outcomes[aid] = "undo_conflict"
    failed_outcomes = {aid: outcome for aid, outcome in outcomes.items()
                       if outcome not in ("applied", "reverted")}
    if failed_outcomes:
        P.mark_applied(jp, set(), outcomes=failed_outcomes)
    # Оставшиеся (пропуск, отмена без применённого ответа, давняя отмена) — по сохранённому состоянию.
    states = P.answer_states(jp)
    for aid in stored:
        if aid not in outcomes:
            outcomes[aid] = states[aid]["state"]
    # Уточнения к исходу — то, что прежний отчёт показывал отдельными списками.
    details = {aid: [states[aid]["detail"]] if states[aid].get("detail") else [] for aid in stored}

    def add_detail(aid, text):
        if aid in details and text and text not in details[aid]:
            details[aid].append(text)

    for row in rep.get("undo_conflicts") or []:
        add_detail(row.get("id"), row.get("why") or row.get("conflict"))
    for conflict in preflight or []:
        for aid in conflict.get("cards") or []:
            others = [states[x]["card"] or x for x in conflict["cards"] if x != aid and x in states]
            add_detail(aid, "с карточкой " + ", ".join(str(x) for x in others) if others else conflict.get("why"))
    for aid, why in revision_conflicts.items():
        add_detail(aid, why)
    for row in free_report.get("left_to_human") or []:
        add_detail(row.get("id"), row.get("why"))
    for aid, why in (free_report.get("failed") or {}).items():
        add_detail(aid, why)
    for row in report.get("conflicts") or []:
        add_detail(row.get("id"), row.get("why") or ("слово уже исправлено прежней правкой"
                                                      if row.get("kept_edits") else None))
    for aid in rep.get("source_conflict_ids") or []:
        add_detail(aid, rep.get("source_conflict"))

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
