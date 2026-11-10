"""Временные папки программы: создаются через scratch(), удаляются при выходе процесса, а остатки от
принудительно завершённых процессов (taskkill /F, отключение питания) убирает sweep_stale().

Раньше mkdtemp() без очистки оставлял в %TEMP% папки emo_*, bench_* (с копией всего звука), gigaam_*, vr_*.
"""
import atexit
import shutil
import tempfile
import time
from pathlib import Path

# Только префиксы, которые создаёт сама программа. Папки данных (rt-data-*) сюда не входят.
PREFIXES = ("emo_", "bench_", "gigaam_", "vr_")
_made = []


def _rm(p):
    shutil.rmtree(p, ignore_errors=True)


def scratch(prefix):
    """Новая временная папка; удаляется при завершении процесса (и вручную через drop())."""
    if prefix not in PREFIXES:
        raise ValueError(f"неизвестный префикс временной папки: {prefix}")
    p = Path(tempfile.mkdtemp(prefix=prefix))
    _made.append(p)
    return p


def drop(p):
    _rm(p)
    if p in _made:
        _made.remove(p)


@atexit.register
def _cleanup_all():
    for p in list(_made):
        _rm(p)
    _made.clear()


def sweep_stale(max_age_hours=12, now=None):
    """Удаляет папки программы в %TEMP% старше max_age_hours. Возвращает число удалённых папок.
    Берёт только папки с префиксами программы; задачи длиннее 12 часов не бывает (слот тяжёлых задач один)."""
    now = time.time() if now is None else now
    n = 0
    try:
        base = Path(tempfile.gettempdir())
        for d in base.iterdir():
            if d.is_dir() and d.name.startswith(PREFIXES):
                try:
                    if now - d.stat().st_mtime > max_age_hours * 3600:
                        _rm(d)
                        n += 1
                except OSError:
                    pass
    except OSError:
        pass
    return n
