"""Синтетическая запись в базе для тестов (У-2а): тесты, которые раньше писали стенограмму файлом без базы
(файловый режим по умолчанию), готовят её штатным путём программы — фиксация результата в базе и выпуск
файлов стенограммы рядом с синтетическим аудио.

Режим хранения выбирается не умолчанием, а самой записью: документ, зарегистрированный в базе, открывается
в режиме базы при любом `recording_backend` (`recordings.open_for_document`). Поэтому тест проходит и до,
и после смены умолчания (У-2б).

База — своя у каждого теста, в отдельной временной папке: на время теста `paths.STATE_DB` подменяется (как в
`test_answer_outcomes.answer_files`), база папки данных не создаётся
(`test_recordings.test_default_state_database_path_is_declared_not_created`). Если тест уже подменил
`paths.STATE_DB` сам, запись идёт в его базу."""
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402
import recordings  # noqa: E402

_DEFAULT_DB = paths.STATE_DB            # путь базы папки данных до подмен тестами


def isolate(test):
    """База теста: прежняя подмена теста, иначе своя временная папка с подменой до конца теста (папка записи
    может жить меньше теста — `with TemporaryDirectory()` внутри него)."""
    if paths.STATE_DB != _DEFAULT_DB:
        return paths.STATE_DB
    folder = Path(tempfile.mkdtemp(prefix="db-record-"))
    test.addCleanup(shutil.rmtree, folder, ignore_errors=True)
    db = folder / "state.sqlite3"
    patcher = mock.patch.object(paths, "STATE_DB", db)
    patcher.start()
    test.addCleanup(patcher.stop)          # снимается раньше удаления папки (обратный порядок)
    return db


def open_db(db=None):
    """Store по пути базы (по умолчанию — база папки данных теста); нет файла — создаётся штатно."""
    db = Path(db or paths.STATE_DB)
    if db.is_file():
        return recordings.open_store(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    return recordings.initialize_store(db, display_name="Тест")


def new_record(test, data, folder, name="x", *, actor="model"):
    """Записать `data` новой записью базы теста `test` рядом с синтетическим аудио `folder/<name>.wav`.

    Возвращает (json_path, md_path, recording_id). `data["file"]` заменяется путём синтетического аудио:
    запись в базе привязана к нему. Папка `folder` должна существовать."""
    folder = Path(folder)
    audio = folder / f"{name}.wav"
    if not audio.exists():
        audio.write_bytes(b"synthetic " + uuid.uuid4().hex.encode())
    data = dict(data)
    data["file"] = str(audio)
    layout = recordings.document_layout(audio)
    layout["root"].mkdir(exist_ok=True)
    store = open_db(isolate(test))
    try:
        root_id, targets = recordings.prepare_new_layout(store, layout["root"], layout["paths"])
        rid = str(uuid.uuid4())
        recordings.commit_worker_result(
            store, data, audio, root_id, recordings.operation_context(store, actor),
            recording_id=rid, operation_id=str(uuid.uuid4()),
            generated_at="2026-11-27T00:00:00Z", document_paths=targets)
        recordings.materialize_recording(store, rid)
    finally:
        store.close()
    return layout["root"] / targets["json"], layout["root"] / targets["md"], rid


def is_db_document(json_path):
    """Документ зарегистрирован в базе папки данных теста."""
    store, binding = recordings.open_for_document(os.fspath(json_path), readonly=True)
    if store is None:
        return False
    store.close()
    return binding is not None


def put(json_path, data):
    """Заменить содержимое записи на `data` новой ревизией поверх текущей — то, чем в файловом режиме было
    переписывание `*.transcript.json` тестом. ID записи, ревизии и путь аудио берутся из текущей записи."""
    import transcript
    current = transcript.load(json_path)
    data = dict(data)
    data.update({k: current.get(k) for k in ("recording_id", "source_revision", "review_revision", "file")})
    transcript.save(data, json_path, Path(json_path).with_suffix(".md"))


# Пустая синтетическая стенограмма: тестам, которым нужна только зарегистрированная запись (пути и блокировка).
EMPTY = {"file": "x", "duration": "00:00:02", "speakers": [], "segments": []}
