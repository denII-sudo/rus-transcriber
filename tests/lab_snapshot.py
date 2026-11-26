"""Общий помощник тестов лаборатории: снимок папки (не тест: имя не test*.py, paths не импортирует)."""
import hashlib
from pathlib import Path


def folder_snapshot(root, skip=()):
    """{путь: (sha256, размер, mtime_ns)} всех файлов под root, кроме верхних папок skip."""
    out = {}
    for f in sorted(Path(root).rglob("*")):
        rel = f.relative_to(root)
        if f.is_file() and rel.parts[0] not in skip:
            st = f.stat()
            out[rel.as_posix()] = (hashlib.sha256(f.read_bytes()).hexdigest(), st.st_size, st.st_mtime_ns)
    return out
