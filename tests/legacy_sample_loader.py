"""Образцы старого файлового формата для тестов переноса (Л-3.1). Не тест: имя не test*.py.

Образец — один JSON-файл в tests/legacy_samples/: все файлы папки записи рядом с аудио (стенограмма, её MD,
папка разбора с ответами и вопросами) как тексты. Их один раз собрала программа штатными вызовами файлового
режима (review_pack.build, add_answers, apply_answers_files, add_interaction_event, transcript.save) на
синтетической записи; разовый скрипт — в описании PR Л-3.1 (после Л-5 он уже не запустится). Настоящих путей в
образцах нет: путь к аудио — метка {AUDIO}, путь к стенограмме — {JOBJSON}; `load` подставляет пути записи теста.
Имена файлов внутри образца содержат «.transcript.», а на диске репозитория таких файлов быть не должно
(tools/check_repo.py), поэтому образец — один файл с другим именем.

Образцы не правят руками: хэши ниже проверяет tests/test_legacy_samples.py.
"""
import json
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent / "legacy_samples"

# sha256 байтов каждого образца (UTF-8 без BOM, переносы LF)
HASHES = {
    "built.json": "edf4f108fddbbac5165b40bde887c26d0d7c83d750e9117c6a8343ecd31cd669",
    "built_archive.json": "b5614fdb3acec5ea80b69209f7d4d45eeb0b64a163be13718e77fd34df79d4ec",
    "saved.json": "047641360dff5df66228717a503f6f8927cd7cf034f9ead5388c4afbb33c5616",
    "waiting.json": "aeaa1034d2452bfc0b451eb6272d96b701ba1fbd65068ac58d39da65c58f1e81",
}


def load(name, side, audio):
    """Записать файлы образца `name` в папку `side`; вернуть путь стенограммы (JSON).

    - built: запись, где ответ r3 применён, а r4 ждёт «Применить»; в папке разбора одна редакция пакета;
    - built_archive: та же запись после применения r4 и пересборки пакета — в папке разбора есть архив редакции;
    - waiting: запись «вторая», где r3 ждёт «Применить» на той же ревизии стенограммы, что у пакета;
    - saved: один файл стенограммы, сохранённый программой поверх старого (есть recording_id и ID слов).
    Уже лежащие в папке одноимённые файлы заменяются.
    """
    data = json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    side = Path(side)
    jp = side / data["transcript"]
    tokens = {"{AUDIO}": str(audio), "{JOBJSON}": str(jp)}
    for rel, text in data["files"].items():
        target = side / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        for token, value in tokens.items():
            # внутри JSON путь записан с экранированием (\\ и \u), в остальных файлах — как есть
            text = text.replace(token, json.dumps(value, ensure_ascii=False)[1:-1] if rel.endswith(".json") else value)
        target.write_bytes(text.encode("utf-8"))
    return jp
