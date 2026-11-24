"""Шаблон пустой базы SQLite для тестов, которым схема нужна как фон, а не как предмет проверки.

`recordings.initialize_store` создаёт базу заново: временный файл, схема, `validate_copy`, перенос журнала, `os.link`
и ещё одно открытие (около 50 мс на базу). Тесты, которые только кладут записи в чистую базу (экран П5, пакет ответов,
подтверждение человека), берут здесь копию шаблона: он строится один раз на процесс, во временной папке этого процесса
(у каждой группы запускателя `tests/run_parallel.py` свой процесс, значит и свой шаблон; общего файла нет), и копируется
побайтно. Имя файла не подходит под `test*.py`: `discover` его не берёт.

Копия — то же, что свежая база: та же схема, те же строки (свой пользователь и установка), WAL. Отличие одно: все копии
одного шаблона имеют одинаковые `store_id`, `installation_id` и `user_id` (у свежих баз они разные). Тесты, которые сравнивают
идентификаторы разных баз или создают две базы подряд, и тесты самой схемы, `initialize`, миграций и резерва по-прежнему
создают базу с нуля — этот помощник им не нужен (его работу проверяет `test_db_template.py`).
"""
import atexit
import shutil
import tempfile
from pathlib import Path

import recordings
import storage

_state = {"dir": None, "templates": {}}


def _template(display_name):
    if _state["dir"] is None:
        _state["dir"] = tempfile.mkdtemp(prefix="rt-dbtemplate-")
        atexit.register(shutil.rmtree, _state["dir"], True)
    template = _state["templates"].get(display_name)
    if template is None:
        template = Path(_state["dir"]) / f"template-{len(_state['templates'])}.sqlite3"
        recordings.initialize_store(template, display_name=display_name).close()     # без -wal/-shm: закрыто последним
        _state["templates"][display_name] = template
    return template


def initialize(path, display_name="Тест"):
    """Как `recordings.initialize_store(path, display_name=…)`: новая база по пути, вернуть открытый Store.

    Существующий файл не заменяется (`FileExistsError`, как у `storage.initialize`)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    shutil.copyfile(_template(display_name), path)
    return storage.open_store(path)
