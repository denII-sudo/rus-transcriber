"""Запись базы «как после переноса П7» для тестов переадресации и голосов. Не тест: имя не test*.py.

Инструмент переноса (`tools/import_legacy.py`) удалён в У-1, а записи, перенесённые им раньше, остаются в базах
установок. Программа узнаёт такую запись только по `quality.migration` первой ревизии стенограммы:
`source` (путь, SHA-256 и размер старого файла на момент переноса) и `variant` (метка и основной ли вариант) —
это читают `recordings.migration_info`, `voice_sources` и экран. Помощник строит пару снимков штатным
`recordings.snapshots_from_worker` и дописывает только эти поля; привязку сомнений и пакетов карточек, которые
делал перенос, он не повторяет: тестам с пустым разбором она не нужна.

Модули программы импортируются внутри функций: тест задаёт временную папку данных до первого импорта.
"""
import copy
import hashlib
import os
import uuid

TOOL = "tests/migrated_record.py"
NAMESPACE = uuid.UUID("6b1d3c5e-8f0a-4c11-9d6e-2a7f4b9e0c31")     # тот же, что был у переноса П7
# служебные поля старого файла, которые при переносе уходили в сведения о переносе (их ведёт база)
LEGACY_RUNTIME_KEYS = ("recording_id", "source_revision", "review_revision", "document_revision",
                       "document_sync")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recording_id_for(source_path, source_sha):
    """Как у переноса: один и тот же старый файл даёт один и тот же recording_id."""
    key = os.path.normcase(os.path.abspath(source_path)) + "|" + source_sha
    return str(uuid.uuid5(NAMESPACE, "legacy-recording:" + key))


def recording_id_from(legacy, source_path, source_sha):
    """Как у переноса: прежний recording_id из файла сохраняется, иначе вычисляется из пути и SHA-256."""
    old = legacy.get("recording_id")
    if isinstance(old, str) and old:
        try:
            return str(uuid.UUID(old))
        except ValueError:
            pass
    return recording_id_for(source_path, source_sha)


def source_of(path):
    """Сведения о старом файле, как их сохранял перенос."""
    return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}


def build_snapshots(legacy, *, recording_id, variant, source):
    """Пара снимков (стенограмма, разбор) записи, перенесённой из `legacy`; -> (recording_id, transcript, review)."""
    import recordings
    import storage
    data = copy.deepcopy(legacy)
    for key in LEGACY_RUNTIME_KEYS:
        data.pop(key, None)
    rid, transcript, review = recordings.snapshots_from_worker(data, recording_id)
    transcript.setdefault("quality", {})["migration"] = {
        "tool": TOOL, "source": dict(source),
        "variant": {k: variant[k] for k in ("tag", "main") if k in variant}}
    storage.validate_pair(transcript, review)
    return rid, transcript, review
