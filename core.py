"""Ядро rus-transcriber: задачи (запуск, статус, отмена), чтение стенограмм, настройки, данные.

Здесь нет ничего, что знало бы про MCP, Claude или любой другой клиент. Слои управления — server.py (MCP) и
rt.py (командная строка) — только вызывают эти функции и показывают результат. Любой третий слой (HTTP, окно,
другой протокол) строится так же.

Соглашения: функции возвращают обычные структуры (dict, list, str); понятная пользователю ошибка — CoreError.
Опасные действия (выгрузка, загрузка, удаление, очистка, перенос) здесь только исполняются; решение человека
получает слой управления (confirm.py для MCP, ввод в терминале для rt.py).
"""
import json
import hashlib
import os
import re
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
            if store.db.execute(
                    "SELECT 1 FROM workflow_verifications WHERE request_id=? AND recording_id=? "
                    "AND source_revision=? AND review_revision=? LIMIT 1",
                    (request_id, recording_id, source_revision, review_revision)).fetchone():
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
                has_conspect = bool(row and row["source_revision"] == current["transcript"]["revision"]
                                    and request["model_task"]["review_revision"] == current["review"]["revision"]
                                    and request["model_task"]["goal"] == request["policy"]["goal"])
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


def workflow_model_task(request_id, offset=0, limit=1000):
    import recordings
    import workflow
    if (type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 2000):
        raise ValueError("неверные границы страницы")
    store = recordings.open_store()
    try:
        task = workflow.prepare_model_task(store, request_id)
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
        return workflow.model_part(store, request_id, part_index)
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


def workflow_execute_model(request_id, *, retry_failed=False):
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        return workflow.execute_model(store, request_id, retry_failed=retry_failed)
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


# ---------- П5: локальный экран и эксплуатация ----------
def workflow_list(limit=20):
    """Последние сохранённые запросы для локального экрана; без тяжёлых чтений."""
    import recordings
    import workflow
    store = recordings.open_store()
    try:
        store.current_identity()
        return workflow.list_recent(store, limit)
    finally:
        store.close()


def recording_revisions(recording_id, kind):
    """История ревизий записи (transcript/review) с происхождением; хэши снимков проверяются."""
    import recordings
    if kind not in ("transcript", "review"):
        raise CoreError("некорректный тип ревизии")
    store = recordings.open_store()
    try:
        store.current_identity()
        result = []
        for revision in store.list_revisions(recording_id, kind):
            data = store.get_revision(recording_id, kind, revision)
            context = data["context"]
            item = {"kind": kind, "revision": revision, "restored_from": data["restored_from"],
                    "actor_kind": context.get("actor_kind"),
                    "initiated_by": (context.get("initiated_by") or {}).get("user_name_snapshot")}
            row = store.db.execute("SELECT created_at FROM operations WHERE operation_id=?",
                                   (data["operation_id"],)).fetchone()
            item["created_at"] = row["created_at"] if row else None
            if kind == "review":
                item["source_revision"] = data["source_revision"]
            else:
                paired = store.db.execute(
                    "SELECT revision FROM review_revisions WHERE recording_id=? AND source_revision=? "
                    "ORDER BY revision DESC LIMIT 1", (recording_id, revision)).fetchone()
                item["restore_review_revision"] = paired["revision"] if paired else None
            result.append(item)
        return result
    finally:
        store.close()


def read_recording_revision(recording_id, kind, revision):
    """Прочитать сохранённую ревизию для просмотра истории; текущая ревизия не меняется."""
    import recordings
    if kind not in ("transcript", "review"):
        raise CoreError("некорректный тип ревизии")
    store = recordings.open_store()
    try:
        store.current_identity()
        data = store.get_revision(recording_id, kind, int(revision))
        context = data["context"]
        return {"kind": kind, "revision": data["revision"], "restored_from": data["restored_from"],
                "payload": data["payload"], "actor_kind": context.get("actor_kind"),
                "initiated_by": (context.get("initiated_by") or {}).get("user_name_snapshot")}
    finally:
        store.close()


def _conspect_screen_sections(payload):
    """Разделы конспекта для экрана. Прежнее поле `sections` отдаётся как есть; если его нет,
    берутся тексты из `evidence` (так исполнитель свёртки сохраняет разделы), без номеров слов."""
    sections = payload.get("sections")
    if isinstance(sections, list):
        return sections
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return []
    return [{"text": item["text"]} for item in evidence
            if isinstance(item, dict) and isinstance(item.get("text"), str)]


def recording_conspects(recording_id):
    """Конспекты записи для локального экрана: имя, ревизии, разделы и пути файлов."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        result = []
        for item in store.list_conspects(recording_id):
            root = store.db.execute("SELECT path FROM allowed_roots WHERE root_id=?",
                                    (item["root_id"],)).fetchone()
            locations = store.db.execute(
                "SELECT kind, relative_path FROM conspect_file_locations WHERE conspect_id=?",
                (item["conspect_id"],)).fetchall()
            files = {loc["kind"]: str(Path(root["path"]) / loc["relative_path"])
                     for loc in locations} if root else {}
            result.append({"conspect_id": item["conspect_id"], "name": item["name"],
                           "current_revision": item["current_revision"],
                           "source_revision": item["source_revision"],
                           "sections": _conspect_screen_sections(item["payload"]),
                           "files": files})
        return result
    finally:
        store.close()


def prepare_restore_snapshot(recording_id, source_revision, review_revision):
    """Снимок для возврата прежней ревизии парой; подтверждение человека привязано к снимку."""
    import recordings
    import storage
    store = recordings.open_store()
    try:
        store.current_identity()
        head = recordings.recording_head(store, recording_id)
        if head is None:
            raise KeyError(recording_id)
        expected = {"source_revision": head["source_revision"], "review_revision": head["review_revision"]}
        restored_review = store.get_revision(recording_id, "review", int(review_revision))
        if restored_review["source_revision"] != int(source_revision):
            raise storage.Conflict("пара ревизий несовместима")
        store.get_revision(recording_id, "transcript", int(source_revision))
        binding = recordings.document_binding(store, recording_id)
        return {"recording_id": recording_id, "source_revision": int(source_revision),
                "review_revision": int(review_revision), "expected_versions": expected,
                "document_revision": binding["document_revision"], "root_id": binding["root_id"],
                "document": str(binding["root"] / binding["paths"]["md"]),
                "context": recordings.operation_context(store, "human")}
    finally:
        store.close()


def restore_recording_revision(snapshot):
    """Вернуть прежнюю ревизию как новую текущую; история сохраняется, файлы формируются заново."""
    import recordings
    if not isinstance(snapshot, dict) or "recording_id" not in snapshot:
        raise CoreError("неверный снимок возврата")
    current = prepare_restore_snapshot(snapshot["recording_id"], snapshot["source_revision"],
                                       snapshot["review_revision"])
    if current != snapshot:
        raise CoreError("состояние записи изменилось после запроса возврата")
    store = recordings.open_store()
    try:
        store.current_identity()
        result = store.restore_with_document(
            snapshot["recording_id"], snapshot["source_revision"], snapshot["review_revision"],
            snapshot["expected_versions"], str(uuid.uuid4()), snapshot["context"],
            root_id=snapshot["root_id"],
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        completed = recordings.materialize_recording(store, snapshot["recording_id"])
        result["materialized"] = (snapshot["recording_id"], result["document_revision"]) in completed
        return result
    except (OSError, ValueError, KeyError) as ex:
        raise CoreError(f"возврат не выполнен или формирование не завершено: {ex}") from ex
    finally:
        store.close()


# ---------- конфликты внешней правки: копия человека и явная замена файла ----------
CONFLICT_SCOPES = ("document", "conspect")
CONFLICT_PREVIEW_CHARS = 20000        # знаков копии в предпросмотре
CONFLICT_SHOWN_LINES, CONFLICT_SHOWN_CHARS = 200, 500   # строк различий и знаков в строке на показ
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
    if documents._digest(found) != row["observed_hash"]:
        raise CoreError("Сохранённая копия повреждена: её контрольная сумма не совпала.")
    return found


def _program_version(store, row):
    """Последняя версия программы для файлов объекта конфликта: (ревизия, {вид файла: байты})."""
    if row["scope"] == "document":
        revision = store.db.execute("SELECT max(document_revision) FROM document_versions WHERE recording_id=?",
                                    (row["recording_id"],)).fetchone()[0]
        saved = store.get_document(row["recording_id"], revision)
        return revision, {kind: bytes(saved[kind]) for kind in ("md", "json", "manifest")}
    saved = store.get_conspect(row["conspect_id"])
    return saved["revision"], {kind: bytes(saved[kind]) for kind in ("md", "json")}


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


def _conflict_item(store, row, rows):
    """Строка общего списка; rows — все конфликты той же области."""
    import documents
    same_object = [other for other in rows if other["object"] == row["object"]]
    status = "resolved" if row["resolved"] else "open"
    last_of_file = row["seq"] == max(other["seq"] for other in same_object
                                     if other["relative_path"] == row["relative_path"])
    if row["resolved"] and last_of_file and _queue_unfinished(store, row):
        program = _program_version(store, row)[1].get(row["kind"])
        if program is not None and _file_at(store, row, row["relative_path"])[1] != documents._digest(program):
            status = "not_written"
    return {"conflict_id": row["conflict_id"], "scope": row["scope"], "recording_id": row["recording_id"],
            "title": Path(row["location"]).name, "name": row["name"], "kind": row["kind"],
            "relative_path": row["relative_path"], "file": str(Path(row["root_path"]) / row["relative_path"]),
            "observed_size": row["observed_size"], "preserved": row["preserved"], "status": status,
            "can_replace": status == "open" and row["preserved"] and not row["kind"].startswith("archive_"),
            "can_retry": status == "not_written" and not any(not other["resolved"] for other in same_object)}


def list_all_conflicts():
    """Открытые и решённые конфликты всей базы для экрана; ничего не пишет и не требует запроса."""
    import paths
    import recordings
    if not Path(paths.STATE_DB).is_file():
        return {"available": False, "items": [],
                "reason": "Базы записей ещё нет: конфликтов нет, показывать нечего."}
    store = recordings.open_store()
    try:
        store.current_identity()
        by_scope = {scope: _conflict_rows(store, scope) for scope in CONFLICT_SCOPES}
        return {"available": True, "reason": None,
                "items": [_conflict_item(store, row, rows) for rows in by_scope.values() for row in rows]}
    finally:
        store.close()


def _refusal(row, state):
    """Почему заменять нельзя (None — можно)."""
    if row["resolved"]:
        return "Этот конфликт уже решён: заменять нечего. Копию можно скачать."
    if row["kind"].startswith("archive_"):
        return ("Это архивная копия прежней версии: программа записывает её один раз и не меняет, поэтому "
                "заменить её здесь нельзя. Свою версию можно скачать.")
    if not row["preserved"]:
        return ("Файл больше 8 МиБ, поэтому программа не сохранила его копию и не заменяет его: он остаётся "
                "на диске как есть. Разберите его вручную.")
    if state == "missing":
        return ("Файла на прежнем месте нет. Положите на его место скачанную копию («Скачать мою правку»), "
                "обновите список, и замена станет доступна.")
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
        if state == "ok" and digest == other["observed_hash"] and digest != documents._digest(program[kind]):
            found.append(other["relative_path"])
    return sorted(found)


def _conflict_facts(store, row):
    """Что известно о конфликте без чтения текста: файл, версия программы, можно ли заменить, снимок."""
    import documents
    archive = row["kind"].startswith("archive_")
    revision, program = (None, None) if archive else _program_version(store, row)
    program_hash = None if archive else documents._digest(program[row["kind"]])
    found, digest = _file_at(store, row, row["relative_path"])
    state = (found if found != "ok" else "program" if digest == program_hash
             else "same" if digest == row["observed_hash"] else "changed")
    reason = _refusal(row, state)
    also = [] if reason else _also_replaced(store, row, program)
    return {"program": None if archive else program[row["kind"]], "file_state": state, "also_replaces": also,
            "can_replace": reason is None, "reason": reason,
            "snapshot": None if reason else {
                "scope": row["scope"], "recording_id": row["recording_id"], "conflict_id": row["conflict_id"],
                "observed_hash": row["observed_hash"], "program_revision": revision,
                "program_hash": program_hash, "also_replaces": also}}


def _lines(data):
    """Строки по CRLF, CR, LF (не str.splitlines: она прячет U+2028 и разрыв страницы)."""
    found = re.split(rb"\r\n|\r|\n", data)
    return found[:-1] if found[-1] == b"" else found


def _extra(mine, other):
    """Строки mine, которых нет в other (с учётом повторов), по порядку. Порядок строк не сравнивается."""
    left = {}
    for line in other:
        left[line] = left.get(line, 0) + 1
    extra = []
    for line in mine:
        if left.get(line):
            left[line] -= 1
        else:
            extra.append(line)
    return extra


def _shown(lines):
    """Строки для показа и признак, что показано всё: число строк и их длина ограничены."""
    text = [line.decode("utf-8", errors="replace") for line in lines[:CONFLICT_SHOWN_LINES]]
    whole = len(text) == len(lines) and all(len(line) <= CONFLICT_SHOWN_CHARS for line in text)
    return [line[:CONFLICT_SHOWN_CHARS] for line in text], whole


def preview_conflict(recording_id, conflict_id, kind="document"):
    """Показать копию человека и сравнение с последней версией программы; ничего не записывает.

    kind — область конфликта: document или conspect."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        row = _conflict_row(store, recording_id, conflict_id, kind)
        facts = _conflict_facts(store, row)
        observed, program = _observed_bytes(store, row), facts["program"]
        text = None if observed is None else observed.decode("utf-8", errors="replace")
        removed = added = []
        newlines_only = False
        if observed is not None and program is not None:
            mine, theirs = _lines(observed), _lines(program)
            removed, added = _extra(mine, theirs), _extra(theirs, mine)
            newlines_only = observed != program and mine == theirs
        shown_removed, whole_removed = _shown(removed)
        shown_added, whole_added = _shown(added)
        return {"conflict_id": conflict_id, "kind": row["kind"], "relative_path": row["relative_path"],
                "observed_hash": row["observed_hash"], "observed_size": row["observed_size"],
                "preserved": row["preserved"],
                "preview": None if text is None else text[:CONFLICT_PREVIEW_CHARS],
                "truncated": text is not None and len(text) > CONFLICT_PREVIEW_CHARS,
                "file": str(Path(row["root_path"]) / row["relative_path"]), "file_state": facts["file_state"],
                "removed": shown_removed, "added": shown_added, "removed_total": len(removed),
                "added_total": len(added), "shown_all": whole_removed and whole_added,
                "newlines_only": newlines_only, "also_replaces": facts["also_replaces"],
                "can_replace": facts["can_replace"], "reason": facts["reason"], "snapshot": facts["snapshot"]}
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
        path = Path(row["relative_path"])
        name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", f"{path.stem} (моя правка){path.suffix}")
        return {"data": data, "name": name}
    finally:
        store.close()


def _write_and_report(store, row):
    """Довести запись файлов объекта и сообщить итог по самому файлу, а не по очереди."""
    import documents
    import recordings
    write = recordings.materialize_recording if row["scope"] == "document" else recordings.materialize_conspects
    try:
        write(store, row["recording_id"])
    except Exception:    # решение уже записано, и ответ всегда по факту: что лежит в файле
        pass
    state, digest = _file_at(store, row, row["relative_path"])
    if digest == documents._digest(_program_version(store, row)[1][row["kind"]]):
        return {"outcome": "replaced", "new_conflict_id": None, "reason": None}
    rows = [other for other in _conflict_rows(store, row["scope"]) if other["object"] == row["object"]]
    fresh = [other for other in rows if not other["resolved"] and other["seq"] > row["seq"]
             and other["relative_path"] == row["relative_path"]]
    if fresh:
        return {"outcome": "changed", "new_conflict_id": fresh[-1]["conflict_id"], "reason": None}
    if any(not other["resolved"] for other in rows):
        reason = ("Запись этого файла ждёт решения по другому файлу той же записи: решите его, "
                  "и программа допишет и этот.")
    elif state == "missing":
        reason = ("Файла на прежнем месте нет. Положите на его место скачанную копию («Скачать мою правку») "
                  "и нажмите «Повторить».")
    elif state == "unavailable":
        reason = ("Файл сейчас нельзя прочитать: он занят другой программой, недоступен или на его месте папка "
                  "либо ссылка. Освободите его и нажмите «Повторить».")
    else:
        reason = ("Программа не смогла записать файл: возможно, он открыт в другой программе, или запись "
                  "остановилась на другой записи из той же папки. Закройте файлы и нажмите «Повторить».")
    return {"outcome": "not_written", "new_conflict_id": None, "reason": reason}


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
    try:
        result = getattr(store, f"resolve_{row['scope']}_conflict")(
            row["conflict_id"], str(uuid.uuid4()), recordings.operation_context(store, "human"))
    except (OSError, ValueError, storage.StorageError) as ex:
        raise CoreError("Решение не записано, файл не тронут: конфликт уже решён или база занята.") from ex
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
        status = _conflict_item(store, row, _conflict_rows(store, kind))["status"]
        if status == "open":
            raise CoreError("Конфликт ещё не решён: повторять нечего. Сначала выберите, что делать с файлом.")
        if status == "resolved":
            raise CoreError("Повторять нечего: файл уже записан, или вы вернули на место свою копию, "
                            "или по этому файлу есть более новый конфликт.")
        return _write_and_report(store, row)
    finally:
        store.close()


def backup_store(destination):
    """Понятная операция резерва состояния: новый неактивный файл БД и отчёт проверки копии."""
    import recordings
    store = recordings.open_store()
    try:
        store.current_identity()
        report = store.backup(destination)
        return {"destination": str(destination), "report": report}
    finally:
        store.close()


def inspect_backup(backup_path):
    """Прочитать резервную копию только для чтения: корни, пользователи, число записей."""
    import storage
    store = storage.open_store_readonly(str(backup_path))
    try:
        roots = [dict(row) for row in store.db.execute(
            "SELECT root_id, path, requires_remap FROM allowed_roots ORDER BY path")]
        users = [dict(row) for row in store.db.execute(
            "SELECT user_id, display_name FROM local_users ORDER BY display_name")]
        recordings_count = store.db.execute("SELECT count(*) FROM recordings").fetchone()[0]
        return {"path": str(backup_path), "roots": roots, "users": users,
                "recordings": recordings_count,
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
            validation = storage.validate_copy(staged)
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
    return {"destination": str(destination), "installation_id": installation_id,
            "previous_saved_as": previous_saved_as, "validation": validation,
            "materialized": materialized}


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
    """MCP: после подтверждения в окне заменить файл версией программы; итог — по самому файлу."""
    import recordings
    import storage
    store = recordings.open_store()
    try:
        store.current_identity()
        result = _replace_file(store, _conflict_row(store, recording_id, conflict_id, scope))
        return {**result, "materialized": result["outcome"] == "replaced"}
    except (OSError, ValueError, storage.StorageError) as ex:
        raise CoreError(f"Конфликт не разрешён: {ex}") from ex
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
                with recordings.answer_draft(jp, initial, state, context) as draft:
                    report = _apply_answers_files_locked(jp, md)
                    candidate = recordings.merge_answer_draft(draft["data"], draft["state"])
                if candidate == initial:
                    latest = recordings.document_binding(store, binding["recording_id"])
                    if recordings.document_sync_state(store, binding["recording_id"],
                                                      latest["document_revision"]) != "synced":
                        completed = recordings.materialize_recording(store, binding["recording_id"])
                        if (binding["recording_id"], latest["document_revision"]) not in completed:
                            raise CoreError("Ответы сохранены в БД; обновление документа не завершено")
                    report["document_sync"] = "synced"
                    try:
                        report["conspects"] = recordings.refresh_conspects(store, binding["recording_id"], jp)
                    except Exception as ex:
                        raise CoreError(f"Ответы сохранены, но конспекты не обновлены: {ex}") from ex
                    return report
                result = recordings.commit_recording_data(
                    store, binding["recording_id"], candidate, expected,
                    context=context)
                completed = recordings.materialize_recording(store, binding["recording_id"])
                if (binding["recording_id"], result["document_revision"]) not in completed:
                    raise CoreError("Ответы сохранены в БД; обновление документа не завершено")
                report["document_sync"] = "synced"
                try:
                    report["conspects"] = recordings.refresh_conspects(store, binding["recording_id"], jp)
                except Exception as ex:
                    raise CoreError(f"Ответы сохранены, но конспекты не обновлены: {ex}") from ex
                return report
        finally:
            store.close()
    with conspect.write_lock(jp):
        return _apply_answers_files_locked(jp, md)


def _apply_answers_files_locked(jp, md):
    import review_pack as P
    import transcript
    decisions, rep = P.decisions(jp)
    free = P.free_answers(jp)
    requested_ids = {d.get("id") for d in decisions if d.get("id")} | set(free)
    preflight = P.preflight_answers(jp, md, decisions, free)
    blocked = {aid for conflict in preflight for aid in conflict["cards"]}
    blocked_free = {aid: answer for aid, answer in free.items() if aid in blocked}
    decisions = [d for d in decisions if d.get("id") not in blocked]
    free = {aid: answer for aid, answer in free.items() if aid not in blocked}
    if preflight:
        rep["preflight_conflicts"] = preflight
        rep["human"] = list(dict.fromkeys((rep.get("human") or []) + sorted(blocked)))
        rep["changed"] = [aid for aid in rep.get("changed") or [] if aid not in blocked]
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
    decisions = [d for d in decisions if d.get("id") not in undo_failed]
    free = {aid: answer for aid, answer in free.items() if aid not in undo_failed}
    # Изменённый ответ тоже сначала отменяется, но в отчёте это изменение, а не отмена.
    rep["reverted"] = [x for x in rep.get("reverted") or [] if x not in rep.get("changed", [])]
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
                if aid in snapshots:
                    P.mark_applied(jp, {aid}, snapshots={aid: snapshots[aid]},
                                   outcomes={aid: "applied"})
    free_items = []
    if blocked_free:
        free_items.append({"conflicts": preflight,
                           "left_to_human": [{"id": aid, "why": "конфликт с другой карточкой до применения",
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
                if aid in snapshots:
                    P.mark_applied(jp, {aid}, snapshots={aid: snapshots[aid]},
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
        elif aid in blocked:
            outcomes[aid] = "blocked_preflight"
        elif aid in undo_failed:
            outcomes[aid] = "undo_conflict"
        elif aid in rep.get("human", []):
            outcomes[aid] = "left_for_human"
        elif aid in apply_conflicts or aid in free_conflicts:
            outcomes[aid] = "apply_conflict"
        elif aid in free_not_found or aid in (report.get("not_found") or []):
            outcomes[aid] = "not_found"
        else:
            outcomes[aid] = "not_applied"
    failed_outcomes = {aid: outcome for aid, outcome in outcomes.items()
                       if outcome not in ("applied", "reverted")}
    if failed_outcomes:
        P.mark_applied(jp, set(), outcomes=failed_outcomes)
    if undone:
        rep["voice_profile_warning"] = {
            "answer_ids": undone,
            "message": ("Если эта запись ранее пополнила голосовой профиль, её прежний вклад мог остаться. "
                        "Профиль не изменён: проверьте вклад и при необходимости отдельно подтвердите "
                        "его обновление или отзыв."),
        }
    return {**rep, "review": report}


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


def _voice_vocab_context(data, json_path, group_override=None):
    """Разрешает группу и только точечные словари, сверяя их с привязкой стенограммы."""
    import vocab
    import voices

    try:
        binding = voices.group_binding(data)
        if binding["conflict"]:
            raise vocab.ResolutionError("Группа расшифровки противоречит quality.vocab",
                                        {"binding": binding, "candidates": []})
        if binding["vocab_status"] == "invalid":
            raise vocab.ResolutionError("Ссылка quality.vocab повреждена или невалидна",
                                        {"binding": binding, "candidates": []})
        if group_override:
            if not isinstance(group_override, str) or paths.safe_group(group_override) != group_override:
                raise CoreError("недопустимое имя выбранной группы")
            if binding["group"] and paths.safe_group(binding["group"]) != group_override:
                raise vocab.ResolutionError("Выбранная группа противоречит привязке расшифровки",
                                            {"binding": binding, "candidates": []})
            canonical = paths.vocab_path(group_override)
            path, data_vocab, group, resolution, binding = vocab.resolve_group_vocab(
                data.get("file"), json_path, data, group_override, canonical)
            if not paths.group_dir(group_override).is_dir():
                raise CoreError("выбранная группа не существует")
        else:
            candidates = vocab.related_candidates(
                data.get("file"), json_path, binding.get("vocab_path"))
            group = binding.get("group")
            has_existing = any(Path(path).expanduser().is_file() for path, _ in candidates)
            if not group and has_existing:
                _, _, group, _ = vocab.resolve_candidates(candidates)
            if group:
                candidates.append((paths.vocab_path(group), "group_fallback"))
            path, data_vocab, group, resolution = vocab.resolve_candidates(
                candidates, expected_group=group)
            resolution["binding"] = binding
        if not group:
            raise CoreError("группа расшифровки неизвестна; укажите явную группу для предпросмотра")
        return {"group": group, "path": path, "vocab": data_vocab, "binding": binding,
                "resolution": resolution,
                "manual_override": bool(group_override and not binding.get("group")),
                "override_requested": bool(group_override)}
    except vocab.ResolutionError as ex:
        raise CoreError(f"не удалось безопасно разрешить группу и словарь: {ex}") from ex


def preview_voice_learning(job_id, group_override=None):
    """Предварительная оценка по разметке, без моделей, биометрических векторов и изменения файлов."""
    import voices
    import vocab
    jp, _ = job_json(job_id)
    if not jp:
        raise CoreError("готовая расшифровка не найдена")
    import recordings
    store, binding = recordings.open_for_document(jp)
    if store is not None:
        store.close()
        raise CoreError("Обновление голосового профиля по SQLite-записи пока не поддерживается")
    try:
        data = json.loads(Path(jp).read_text(encoding="utf-8"))
        if data.get("no_profile"):
            raise CoreError("для этой расшифровки профили групп отключены")
        context = _voice_vocab_context(data, jp, group_override)
        group = context["group"]
        group_vocab = context["vocab"]
        path = paths.voices_path(group)
        try:
            profile = voices.load_profile(path, convert=False)
        except voices.VoiceProfileError as ex:
            raise CoreError(f"голосовой профиль группы несовместим или повреждён: {ex}. "
                            "Просмотр и изменение не выполняются; перенесите профиль отдельно.") from ex
        audio = data.get("file")
        if not audio:
            raise CoreError("в расшифровке не указан исходный аудиофайл")
        key = str(Path(audio).resolve())
        existing = sorted(name for name, speaker in profile.get("speakers", {}).items()
                          if key in speaker.get("parts", {}))
        report = {}
        eligible = voices._confirmed(data) or voices.has_human_edits(data)
        spans = voices._spans(data, voices._confirmed(data), report) if eligible else []
        if not eligible:
            report["unconfirmed_source"] = 1
        participants = (group_vocab or {}).get("participants") or []
        allowed_names = {p["name"] for p in participants if p.get("name")}
        outside_vocab = sorted({name for name, _, _, _ in spans if allowed_names and name not in allowed_names})
        if allowed_names:
            spans = [item for item in spans if item[0] in allowed_names]
        candidates = {}
        human_seconds = {}
        for name, start, end, human in spans:
            item = candidates.setdefault(name, {"windows": 0, "seconds": 0.0, "human_windows": 0})
            item["windows"] += 1
            item["seconds"] += end - start
            item["human_windows"] += int(human)
            if human:
                human_seconds[name] = human_seconds.get(name, 0.0) + (end - start)
        for item in candidates.values():
            item["seconds"] = round(item["seconds"], 2)
        # «человеческие» секунды вклада этой записи: сейчас в профиле → после замены по текущей разметке
        human_change, human_warnings = {}, []
        for name in sorted(set(existing) | set(human_seconds)):
            part = profile.get("speakers", {}).get(name, {}).get("parts", {}).get(key) or {}
            before_h = round(float(part.get("human_sec") or 0.0), 2)
            after_h = round(human_seconds.get(name, 0.0), 2)
            if before_h or after_h:
                human_change[name] = {"before": before_h, "after": after_h}
            if before_h > 0 and after_h == 0:
                human_warnings.append(name)
        return {"group": group, "existing_contributions": existing, "candidates": candidates,
                "human_seconds_change": human_change, "human_seconds_lost": human_warnings,
                "selection_report": report, "selection_explanation": voices.explain_selection_report(report),
                "audio_available": Path(audio).is_file(),
                "group_selection": {"group": group,
                                    "method": ("manual_override" if context["manual_override"] else
                                               "explicit_match" if context["override_requested"] else "transcript_or_vocab"),
                                    "binding_group": context["binding"].get("group")},
                "vocab": {"path": str(context["path"]) if context["path"] else None,
                          "group": group_vocab.get("group") if context["path"] else None,
                          "sha256": _voice_hash(context["path"]) if context["path"] else "absent"},
                "vocab_resolution": context["resolution"],
                "names_outside_vocab": outside_vocab,
                "replacement_available": True,
                "note": "Только разметка; модель, качество голоса и карантин не проверялись. "
                        "Окончательный результат обновления зависит от модели и карантина."}
    except (OSError, ValueError, KeyError, TypeError, vocab.ResolutionError) as ex:
        raise CoreError(f"не удалось построить предварительный просмотр: {ex}") from ex


def _voice_hash(path):
    p = Path(path)
    if not p.is_file():
        return "absent"
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_voice_action(job_id, group_override=None):
    """Снимок предварительного плана для привязки подтверждения к конкретным файлам."""
    jp, _ = job_json(job_id)
    if not jp:
        raise CoreError("готовая расшифровка не найдена")
    try:
        before = _voice_hash(jp)
        report = preview_voice_learning(job_id, group_override)
        group = report["group"]
        if paths.safe_group(group) != group:
            raise CoreError("недопустимое имя группы")
        with settings.group_voice_lock(group):
            profile = paths.voices_path(group)
            profile_hash = _voice_hash(profile)
            source_data = json.loads(Path(jp).read_text(encoding="utf-8"))
            audio = source_data.get("file")
            audio_hash = _voice_hash(audio) if audio else "absent"
            report = preview_voice_learning(job_id, group_override)
            vocab_path = report["vocab"]["path"]
            vocab_hash = report["vocab"]["sha256"]
            if (before != _voice_hash(jp) or report["group"] != group or
                    profile_hash != _voice_hash(profile) or
                    (vocab_path and _voice_hash(vocab_path) != vocab_hash) or
                    (audio and audio_hash != _voice_hash(audio))):
                raise CoreError("исходные файлы изменились; повторите предварительный просмотр")
        snapshot = {"job_id": job_id, "json": str(Path(jp).resolve()), "group": group,
                    "group_override": group_override or None,
                    "manual_group_override": report["group_selection"]["method"] == "manual_override",
                    "vocab_path": vocab_path, "vocab_hash": vocab_hash,
                    "source_hash": before, "profile_hash": profile_hash, "audio_hash": audio_hash,
                    "report": report}
        if any(len(line) > 200 for line in voice_action_lines(snapshot)) or len("\n".join(voice_action_lines(snapshot))) > 1100:
            raise CoreError("список участников слишком длинный для безопасного подтверждения; действие не запущено")
        return snapshot
    except (OSError, ValueError) as ex:
        raise CoreError(f"не удалось подготовить подтверждение: {ex}") from ex


def voice_action_lines(snapshot, revoke=False):
    report = snapshot["report"]
    candidates = report["candidates"]
    lines = [f"Группа: {snapshot['group']}", f"Расшифровка: {Path(snapshot['json']).name}",
             "Существующие вклады: " + (", ".join(report["existing_contributions"]) or "нет"),
             "Кандидаты по разметке: " + (", ".join(f"{n}: {v['seconds']} с" for n, v in candidates.items()) or "нет")]
    change = report.get("human_seconds_change") or {}
    if change:
        lines.append("Подтверждено человеком, с: " +
                     ", ".join(f"{n} {v['before']} → {v['after']}" for n, v in change.items()))
    if report.get("human_seconds_lost"):
        lines.append("ВНИМАНИЕ: замена обнулит подтверждённое человеком у: " + ", ".join(report["human_seconds_lost"]) +
                     ". Сначала перенесите старые подтверждения (legacy_speaker_review).")
    selection = report.get("group_selection") or {}
    if selection.get("method") == "manual_override":
        lines.append(f"Группа выбрана вручную: {snapshot['group']} (в расшифровке привязка отсутствует).")
    elif selection.get("method") == "explicit_match":
        lines.append(f"Группа выбрана явно и совпадает с привязкой: {snapshot['group']}.")
    if snapshot.get("vocab_path"):
        lines.append(f"Словарь группы: {Path(snapshot['vocab_path']).name} "
                     f"(SHA-256 {snapshot.get('vocab_hash', '')[:12]}…).")
    elif snapshot.get("group_override"):
        lines.append("Словарь выбранной группы отсутствует; обновление не запущено.")
    else:
        lines.append("Словарь группы не найден; ограничение участников словарём недоступно.")
    if report.get("selection_explanation"):
        lines.append("Предварительный отбор (результат модели может отличаться):")
        lines.extend(f"{item['description']}: {item['count']}" for item in report["selection_explanation"])
    lines.extend([("Удалить перечисленные вклады источника; другие источники не меняются." if revoke else
                   "Новый отпечаток может заменить вклад или попасть в карантин; без подходящего отпечатка вклад сохраняется."),
                  "Будет создана резервная копия профиля. Подтвердите только после проверки."])
    return lines


def backup_voice_profile(path):
    """Сохранить точную копию биометрического файла до изменения, не заменяя старые копии."""
    path = Path(path)
    if not path.exists():
        return None
    folder = path.parent / "voice-backups"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ("voices-" + uuid.uuid4().hex + ".json")
    with target.open("xb") as f:
        f.write(path.read_bytes())
    return str(target)


def backup_voice_profiles(group=""):
    """Сделать отдельную резервную копию профиля одной группы или всех групп.

    Копии лежат в `voice-backups` внутри группы, не шифруются, не входят в экспорт и
    автоматически не удаляются. Векторы и имена не возвращаются.
    """
    import voices
    if group:
        if group not in paths.list_groups():
            raise CoreError("группа не найдена")
        groups = [group]
    else:
        groups = paths.list_groups()
    backups = []
    for name in groups:
        path = paths.voices_path(name)
        try:
            with settings.group_voice_lock(name):
                target = backup_voice_profile(path)
        except (OSError, ValueError) as ex:
            raise CoreError(f"не удалось сохранить резерв группы «{name}»: {ex}") from ex
        if target:
            backups.append({"group": name, "backup": target})
    return {"backups": backups, "count": len(backups),
            "note": "Резерв — биометрия в открытом виде; храните и удаляйте по решению владельца."}


def voice_quality(group=""):
    """Политика качества и сохранённые измерения по профилям; аудио не декодируется."""
    import voices
    if group and group not in paths.list_groups():
        raise CoreError("группа не найдена")
    groups = [group] if group else paths.list_groups()
    out = []
    for name in groups:
        try:
            policy = settings.group_voice_settings(name)
        except ValueError as ex:
            out.append({"group": name, "settings_error": str(ex)})
            continue
        entry = {"group": name,
                 "audio_quality_policy": policy.get("audio_quality_policy", "warn"),
                 "policy_source": policy.get("audio_quality_policy_source", "default")}
        path = paths.voices_path(name)
        if not path.exists():
            entry.update({"profile": "absent", "measurements": [], "summary": {"measured_sources": 0}})
            out.append(entry)
            continue
        try:
            data = voices.load_profile(path, convert=False)
        except voices.VoiceProfileError as ex:
            entry.update({"profile_error": str(ex)})
            out.append(entry)
            continue
        measurements = []
        for source, q in (data.get("audio_quality") or {}).items():
            if not isinstance(q, dict):
                continue
            measurements.append({
                "recording": Path(source).name, "from": q.get("from"),
                "at": q.get("at"), "available": q.get("available"),
                "duration_sec": q.get("duration_sec"), "peak_dbfs": q.get("peak_dbfs"),
                "rms_dbfs": q.get("rms_dbfs"), "clipping_ratio": q.get("clipping_ratio"),
                "silence_ratio": q.get("silence_ratio"), "snr_db_estimate": q.get("snr_db_estimate"),
                "warnings": q.get("warnings") or [],
                "warning_notes": voices.explain_audio_quality(q)})
        entry.update({
            "profile": "ok", "measurements": measurements[-50:],
            "summary": {"measured_sources": len(measurements),
                        "clipping_sources": sum(1 for m in measurements if "clipping" in (m["warnings"] or [])),
                        "warning_sources": sum(1 for m in measurements if m["warnings"])},
            "recent_quality_warnings": [w for w in (data.get("warnings") or [])
                                        if w.get("kind") == "audio_quality"][-5:]})
        out.append(entry)
    return {"groups": out,
            "note": "Измерения приблизительны. Политика warn только предупреждает; "
                    "reject_clipping не учитывает источник с клиппингом. Тихая речь не отбраковывается."}


def set_voice_quality_policy(policy, group=""):
    """Политика качества: warn (по умолчанию) или reject_clipping; без group — для новых групп."""
    import voices
    try:
        policy = voices.validate_quality_policy(policy)
    except ValueError as ex:
        raise CoreError(str(ex)) from ex
    if not group:
        settings.update(explicit=("voice_quality_policy_default",), voice_quality_policy_default=policy)
        return {"default_for_new_groups": policy,
                "note": "Группы без явной политики наследуют это значение; уже выбранная политика группы не меняется."}
    if group not in paths.list_groups():
        raise CoreError("группа не найдена")
    try:
        with settings.group_voice_lock(group):
            current = settings.group_voice_settings(group)
            data = {k: v for k, v in current.items() if k != "audio_quality_policy_source"}
            data["audio_quality_policy"] = policy
            saved = settings.save_group_voice_settings(group, data)
    except (ValueError, OSError) as ex:
        raise CoreError(str(ex)) from ex
    return {"group": group, "audio_quality_policy": saved.get("audio_quality_policy", policy),
            "auto_learn": saved.get("auto_learn")}


REFRESH_MEANINGS = {
    "unchanged": "Источник учтён, отпечаток совпадает.",
    "changed": "Отпечаток стенограммы или метаданные аудио отличаются — нужен просмотр и подтверждённое обновление.",
    "legacy_baseline_pending": "Отпечатка нет; при следующем обучении текущее состояние будет принято как исходное.",
    "not_recorded": "Источник есть в профиле, но без записи происхождения.",
    "transcript_missing": "Файл стенограммы не найден.",
    "transcript_unreadable": "Стенограмма не читается.",
    "audio_missing": "Исходное аудио не найдено или не указано.",
}


def _voice_job_map():
    """{путь стенограммы: id задачи} по готовым задачам; нужен только для подсказки обновления."""
    out = {}
    try:
        files = sorted(paths.JOBS.glob("*.json"))
    except OSError:
        return out
    for p in files:
        try:
            status = json.loads(p.read_text(encoding="utf-8"))
            jp = status.get("json")
            if status.get("state") == "done" and jp:
                out[str(Path(jp).resolve())] = p.stem
        except (OSError, ValueError, TypeError):
            continue
    return out


def voice_refresh_plan(group=""):
    """Что изменилось после прежнего обучения; только чтение, без моделей и записи профиля."""
    import voices
    if group and group not in paths.list_groups():
        raise CoreError("группа не найдена")
    groups = [group] if group else paths.list_groups()
    jobs = _voice_job_map()
    out = []
    for name in groups:
        path = paths.voices_path(name)
        if not path.exists():
            out.append({"group": name, "profile": "absent", "sources": [], "summary": {"needs_update": 0}})
            continue
        try:
            data = voices.load_profile(path, convert=False)
        except voices.VoiceProfileError as ex:
            out.append({"group": name, "profile_error": str(ex)})
            continue
        entries = []
        for key, stored in sorted((data.get("sources") or {}).items()):
            jp = Path(key)
            audio_path, audio_key = None, None
            entry = {"transcript": jp.name, "recording": None, "job_id": jobs.get(str(jp.resolve())),
                     "status": "transcript_missing", "needs_update": True,
                     "transcript_changed": None, "audio_changed": None,
                     "comparison_basis": "unavailable", "legacy_mtime_changed": None,
                     "needs_review_reasons": []}
            if jp.is_file():
                try:
                    source_data = json.loads(jp.read_text(encoding="utf-8"))
                    audio_path = source_data.get("file")
                except (OSError, ValueError, TypeError):
                    entry["status"] = "transcript_unreadable"
                else:
                    if not audio_path:
                        entry["status"] = "audio_missing"
                    else:
                        audio_key = str(Path(audio_path).resolve())
                        entry["recording"] = Path(audio_path).name
                        if not Path(audio_path).is_file():
                            entry["status"] = "audio_missing"
                        else:
                            current = voices.source_fingerprint(jp, audio_path)
                            status = voices.source_fingerprint_status(stored, current, jp.stat().st_mtime)
                            entry["status"] = {"unchanged": "unchanged", "changed": "changed",
                                               "legacy_unchanged": "legacy_baseline_pending",
                                               "legacy_changed": "changed",
                                               "absent": "not_recorded"}.get(status, "changed")
                            if status in ("legacy_changed", "legacy_unchanged"):
                                entry["comparison_basis"] = "legacy_mtime"
                                entry["legacy_mtime_changed"] = status == "legacy_changed"
                            elif isinstance(stored, dict):
                                entry["comparison_basis"] = "fingerprint"
                                transcript_keys = ("transcript_sha256", "transcript_size")
                                audio_keys = ("audio_size", "audio_mtime")
                                if all(stored.get(k) is not None for k in transcript_keys):
                                    entry["transcript_changed"] = any(stored[k] != current[k] for k in transcript_keys)
                                if all(stored.get(k) is not None for k in audio_keys):
                                    entry["audio_changed"] = any(stored[k] != current[k] for k in audio_keys)
            versions = []
            if audio_key:
                for speaker in (data.get("speakers") or {}).values():
                    part = (speaker.get("parts") or {}).get(audio_key)
                    if part:
                        versions.append(part.get("selection_version"))
            current_selection = bool(versions) and all(v == voices.SELECTION_VERSION for v in versions)
            entry["selection_version_current"] = current_selection
            if entry["status"] == "changed":
                if entry["comparison_basis"] == "legacy_mtime":
                    entry["needs_review_reasons"].append("время стенограммы отличается от сохранённого старого значения")
                elif entry["comparison_basis"] == "unavailable":
                    entry["needs_review_reasons"].append("прежние данные источника не позволяют выполнить сравнение")
                else:
                    entry["needs_review_reasons"].append("отпечаток стенограммы или метаданные аудио отличаются")
            if versions and not current_selection:
                entry["needs_review_reasons"].append("вклад без текущей версии отбора")
            if entry["status"] in ("transcript_missing", "transcript_unreadable", "audio_missing"):
                entry["needs_review_reasons"].append(REFRESH_MEANINGS[entry["status"]])
            if entry["status"] == "not_recorded":
                entry["needs_review_reasons"].append("нет отпечатка источника")
            if entry["status"] == "legacy_baseline_pending":
                entry["needs_review_reasons"].append("нет постоянного отпечатка источника")
            if audio_key and audio_key in (data.get("audio_quality") or {}):
                q = data["audio_quality"][audio_key]
                entry["audio_quality_warnings"] = (q or {}).get("warnings") or []
                entry["audio_quality_notes"] = voices.explain_audio_quality(q)
                if entry["audio_quality_warnings"]:
                    entry["needs_review_reasons"].append("есть предупреждения качества аудио")
            entry["needs_update"] = bool(entry["needs_review_reasons"])
            entry["meaning"] = REFRESH_MEANINGS.get(entry["status"], entry["status"])
            if entry["status"] == "changed" and entry["comparison_basis"] == "legacy_mtime":
                entry["meaning"] = ("Время стенограммы отличается от старого значения; сравнить содержимое "
                                    "стенограммы и прежнее аудио невозможно. Нужен просмотр.")
            elif entry["status"] == "changed" and entry["comparison_basis"] == "unavailable":
                entry["meaning"] = "Прежние данные источника не позволяют выполнить сравнение; нужен просмотр."
            entries.append(entry)
        summary = {}
        for item in entries:
            summary[item["status"]] = summary.get(item["status"], 0) + 1
        summary["needs_update"] = sum(1 for item in entries if item["needs_update"])
        out.append({"group": name, "profile": "ok", "revision": data.get("revision"),
                    "sources": entries, "summary": summary})
    return {"groups": out,
            "needs_update": sum(item.get("summary", {}).get("needs_update", 0) for item in out),
            "note": "План только показывает различия. Автоматическая замена вкладов не выполняется; "
                    "обновляйте через preview_voice_profiles и update_voice_profiles отдельной задачи."}


def start_voice_update(snapshot):
    group, jp = snapshot["group"], Path(snapshot["json"])
    with settings.group_voice_lock(group):
        if _voice_hash(jp) != snapshot["source_hash"] or _voice_hash(paths.voices_path(group)) != snapshot["profile_hash"]:
            raise CoreError("стенограмма или профиль изменились; повторите просмотр и подтверждение")
        data = json.loads(jp.read_text(encoding="utf-8"))
        audio = data.get("file")
        if not audio or not Path(audio).is_file():
            raise CoreError("аудиофайл недоступен")
        if _voice_hash(audio) != snapshot["audio_hash"]:
            raise CoreError("аудиофайл изменился; повторите просмотр и подтверждение")
        current = _voice_vocab_context(data, jp, snapshot.get("group_override"))
        current_path = str(current["path"]) if current["path"] else None
        current_vocab_hash = _voice_hash(current["path"]) if current["path"] else "absent"
        if (current["group"] != group or current_path != snapshot.get("vocab_path") or
                current_vocab_hash != snapshot.get("vocab_hash")):
            raise CoreError("словарь или привязка группы изменились; повторите просмотр и подтверждение")
        args = ["--learn-only", str(jp), "--voice-source-hash", snapshot["source_hash"],
                "--voice-profile-hash", snapshot["profile_hash"], "--voice-audio-hash", snapshot["audio_hash"],
                "--voice-group", group, "--voice-vocab-path", snapshot.get("vocab_path") or "",
                "--voice-vocab-hash", snapshot.get("vocab_hash") or "absent"]
        if snapshot.get("group_override"):
            args.extend(["--voice-group-override", snapshot["group_override"]])
        return {"job_id": launch("worker.py", args, audio, "learn"),
                "hint": "Ход — transcription_status(job_id)"}


def revoke_voice_source(snapshot):
    """Отозвать вклад источника после подтверждения; другие записи не затрагиваются."""
    group, jp = snapshot["group"], Path(snapshot["json"])
    path = paths.voices_path(group)
    with settings.group_voice_lock(group):
        if _voice_hash(jp) != snapshot["source_hash"] or _voice_hash(path) != snapshot["profile_hash"]:
            raise CoreError("стенограмма или профиль изменились; повторите предварительный просмотр и подтверждение")
        if not path.exists():
            raise CoreError("профиль отсутствует")
        data = json.loads(jp.read_text(encoding="utf-8"))
        key = str(Path(data["file"]).resolve())
        import voices
        try:
            profile = voices.load_profile(path, convert=False)
        except voices.VoiceProfileError as ex:
            raise CoreError(f"голосовой профиль группы несовместим или повреждён: {ex}. "
                            "Отзыв не выполнен; перенесите профиль отдельно.") from ex
        names = sorted(n for n, s in profile.get("speakers", {}).items() if key in s.get("parts", {}))
        if not names or names != snapshot["report"]["existing_contributions"]:
            raise CoreError("вклады изменились или отсутствуют; повторите просмотр")
        backup = backup_voice_profile(path)
        for name in names:
            profile["speakers"][name]["parts"].pop(key)
        profile.get("confusions", {}).pop(key, None)
        profile.get("sources", {}).pop(str(jp.resolve()), None)
        profile.get("source_selection_versions", {}).pop(str(jp.resolve()), None)
        profile["revision"] = int(profile.get("revision", 0)) + 1
        profile.setdefault("history", []).append({"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                   "from": jp.name, "kind": "source_revoked", "speakers": names})
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            json.dump(profile, f, ensure_ascii=False)
            tmp = f.name
        os.replace(tmp, path)
        return {"group": group, "revoked": names, "revision": profile["revision"], "backup": backup}


def set_voice_auto_learn(enabled, group=""):
    if type(enabled) is not bool:
        raise CoreError("значение должно быть true или false")
    if not group:
        try:
            for existing in paths.list_groups():
                with settings.group_voice_lock(existing):
                    settings.group_voice_settings(existing, initialize=True)
        except (ValueError, OSError) as ex:
            raise CoreError(f"не удалось сохранить настройки существующих групп: {ex}") from ex
        settings.update(explicit=("voice_auto_learn_default",), voice_auto_learn_default=enabled)
        return {"default_for_new_groups": enabled}
    if group not in paths.list_groups():
        raise CoreError("группа не найдена")
    try:
        with settings.group_voice_lock(group):
            current = settings.group_voice_settings(group)
            data = {k: v for k, v in current.items() if k != "audio_quality_policy_source"}
            data.update(auto_learn=enabled, reason="manual")
            return {"group": group, **settings.save_group_voice_settings(group, data)}
    except (ValueError, OSError) as ex:
        raise CoreError(str(ex)) from ex


def voice_profiles():
    """Сводка по голосовым профилям групп; сами отпечатки не выводятся."""
    import voices
    out = []
    for p in sorted(paths.GROUPS.glob("*/voices.json")):
        try:
            d = voices.load_profile(p, convert=False)
        except voices.VoiceProfileError as ex:
            out.append({"group": p.parent.name, "error": str(ex), "profile_error": str(ex)})
            continue
        conf = {}
        for pairs in (d.get("confusions") or {}).values():
            for k, v in pairs.items():
                conf[k.replace("|", " → ")] = conf.get(k.replace("|", " → "), 0) + v
        try:
            policy = settings.group_voice_settings(p.parent.name)
        except ValueError:
            policy = {"auto_learn": False, "reason": "settings_error"}
        vocab_file = p.parent / "vocab.json"
        try:
            vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
            expected = [x["name"] for x in vocab.get("participants", []) if x.get("name")]
        except (OSError, ValueError, TypeError):
            expected = []
        out.append({"group": p.parent.name, "revision": d.get("revision"),
                    "auto_learn": policy, "sufficiency": voices.profile_sufficiency(d, expected),
                    "weight_diagnostics": voices.voice_weight_diagnostics(d),
                    "audio_quality_policy": policy.get("audio_quality_policy", "warn"),
                    "audio_quality_policy_source": policy.get("audio_quality_policy_source", "default"),
                    "speakers": {n: _speaker_voice_summary(s) for n, s in d.get("speakers", {}).items()},
                    "corrected_pairs": conf,
                    "warnings": (d.get("warnings") or [])[-5:],
                    "quality_summary": {"measured_sources": len(d.get("audio_quality") or {}),
                                        "clipping_sources": sum(
                                            1 for q in (d.get("audio_quality") or {}).values()
                                            if "clipping" in ((q or {}).get("warnings") or []))},
                    "quarantine": [{"name": n, "recording": Path(a).name, **{k: v for k, v in q.items() if k != "json"}}
                                   for a, qa in (d.get("quarantine") or {}).items() for n, q in qa.items()],
                    "identity": {Path(a).name: v for a, v in (d.get("identity") or {}).items()},
                    "sources": list(d.get("sources", {}))})
    return {"profiles": out, "groups": paths.list_groups(),
            "default_group": settings.load_config().get("default_group")}


def _speaker_voice_summary(speaker):
    """Длительность и вес раздельно: старый `sec` не подменяет неизмеренную длительность."""
    seconds = weighted = human = 0.0
    measured = unmeasured = 0
    for part in (speaker.get("parts") or {}).values():
        sec = part.get("sec")
        if isinstance(sec, (int, float)) and not isinstance(sec, bool) and sec >= 0:
            weighted += float(sec)
        a, h = part.get("auto_sec"), part.get("human_sec")
        if (isinstance(a, (int, float)) and not isinstance(a, bool)
                and isinstance(h, (int, float)) and not isinstance(h, bool) and a >= 0 and h >= 0):
            seconds += float(a) + float(h)
            human += float(h)
            measured += 1
        else:
            unmeasured += 1
    return {"records": len(speaker.get("parts") or {}),
            "seconds": round(seconds, 1),
            "weighted_seconds": round(weighted, 1),
            "human_seconds": round(human, 1),
            "measured_records": measured,
            "unmeasured_records": unmeasured}


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


def describe_migrate():
    r = do_migrate(True)
    res = r.get("result")
    return ["Перенос данных в папку данных из прежних мест.", f"Папка данных: {r['data_dir']}",
            f"План: {json.dumps(res, ensure_ascii=False)[:300] if res is not None else r.get('error', '—')}"]
