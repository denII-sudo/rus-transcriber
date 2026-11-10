"""Очередь тяжёлых задач: одновременно — не больше max_heavy_jobs (config.json, по умолчанию 2) задач,
которые сильно грузят процессор и видеокарту (расшифровка, тест скорости, пересчёт профилей голосов).

Места — файлы jobs/slots/slot<N>.lock с блокировкой ОС: место освобождается само, когда процесс
завершается или его отменяют (cancel_job), даже аварийно. Задача без места ждёт в состоянии queued.
"""
import os
import time
from pathlib import Path


def _try_lock(f):
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def acquire(jobs_dir, limit, status=None, job_id=None, poll=2.0):
    """Ждёт свободное место и держит его до конца процесса. -> открытый файл места (не закрывать)."""
    d = Path(jobs_dir) / "slots"
    d.mkdir(parents=True, exist_ok=True)
    limit = max(1, int(limit or 1))
    shown = None
    while True:
        for n in range(limit):
            p = d / f"slot{n}.lock"
            f = open(p, "a+")
            if _try_lock(f):
                (d / f"slot{n}.owner").write_text(f"{job_id or os.getpid()}", encoding="utf-8")
                return f
            f.close()
        if status is not None:
            owners = []
            for n in range(limit):
                try:
                    owners.append((d / f"slot{n}.owner").read_text(encoding="utf-8").strip())
                except OSError:
                    pass
            msg = f"в очереди: заняты {limit} из {limit} мест ({', '.join(owners)})"
            if msg != shown:
                status.set(state="queued", stage=msg, progress=0)
                shown = msg
        time.sleep(poll)
