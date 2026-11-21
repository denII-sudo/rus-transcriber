"""Схема ограниченного модельного задания и проверка ссылок на слова.

Версия 1 (без поля `version`) — только слова; так выглядят задания, сохранённые до версии 2,
они собираются и проверяются по-прежнему. Версия 2 (`"version": 2`) добавляет реплики:
у слова номер реплики `turn`, у задания список `turns` с условной меткой говорящего.
Имена говорящих в задание не попадают: метка — только «Участник N» или UNKNOWN_SPEAKER.
"""
import re


CURRENT_VERSION = 2
SPEAKER_LABEL = re.compile(r"Участник [1-9][0-9]{0,3}")
UNKNOWN_SPEAKER = "Говорящий не определён"


def speaker_label(number):
    """Условная метка говорящего по его порядковому номеру в записи."""
    if type(number) is not int or not 1 <= number <= 9999:
        raise ValueError("некорректный номер участника")
    return f"Участник {number}"


def version(task):
    """1 — задание без поля версии, 2 — с репликами; иное отклоняется."""
    if not isinstance(task, dict):
        raise ValueError("некорректное задание")
    if "version" not in task:
        return 1
    if type(task["version"]) is not int or task["version"] != CURRENT_VERSION:
        raise ValueError("неизвестная версия задания")
    return CURRENT_VERSION


def _turns(turns):
    if not isinstance(turns, list):
        raise ValueError("реплики должны быть списком")
    result, previous = [], 0
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) != {"turn", "speaker"}:
            raise ValueError("реплика должна содержать только номер и условную метку говорящего")
        number, speaker = turn["turn"], turn["speaker"]
        if type(number) is not int or number <= previous:
            raise ValueError("номера реплик должны возрастать")
        if not isinstance(speaker, str) or not (speaker == UNKNOWN_SPEAKER or SPEAKER_LABEL.fullmatch(speaker)):
            raise ValueError("говорящий задаётся только условной меткой")
        previous = number
        result.append({"turn": number, "speaker": speaker})
    return result


def build(recording_id, source_revision, words, goal="", review_revision=1, *, turns=None):
    """Сформировать задание из слов и их устойчивых идентификаторов.

    Без `turns` — версия 1 (только слова). С `turns` — версия 2: каждое слово несёт номер
    своей реплики, реплики идут по порядку, пустых и неизвестных реплик нет.
    """
    if not isinstance(recording_id, str) or not recording_id:
        raise ValueError("требуется идентификатор записи")
    if isinstance(source_revision, bool) or not isinstance(source_revision, int) or source_revision < 1:
        raise ValueError("требуется ревизия источника")
    if type(review_revision) is not int or review_revision < 1:
        raise ValueError("требуется ревизия сомнений")
    if not isinstance(words, list):
        raise ValueError("слова должны быть списком")
    if not isinstance(goal, str) or len(goal) > 2000:
        raise ValueError("некорректная цель")
    checked_turns = None if turns is None else _turns(turns)
    required = {"word_id", "text"} if turns is None else {"word_id", "text", "turn"}
    allowed = required | {"uncertain"}
    result, seen = [], set()
    for word in words:
        if (not isinstance(word, dict) or not required <= set(word) or set(word) - allowed
                or not isinstance(word.get("uncertain", False), bool)):
            raise ValueError("слово должно содержать только ID, текст и признак сомнения"
                             if turns is None else
                             "слово должно содержать только ID, текст, признак сомнения и номер реплики")
        ident, text = word["word_id"], word["text"]
        if not isinstance(ident, str) or not ident or ident in seen or not isinstance(text, str):
            raise ValueError("некорректный или повторный ID слова")
        seen.add(ident)
        item = {"word_id": ident, "text": text, "uncertain": word.get("uncertain", False)}
        if turns is not None:
            if type(word["turn"]) is not int:
                raise ValueError("некорректный номер реплики у слова")
            item["turn"] = word["turn"]
        result.append(item)
    if turns is None:
        return {"kind": "conspect", "recording_id": recording_id,
                "source_revision": source_revision, "review_revision": review_revision, "goal": goal, "words": result}
    # Слова идут по репликам подряд: каждая реплика списка встречается ровно одним отрезком слов.
    order = []
    for item in result:
        if not order or order[-1] != item["turn"]:
            order.append(item["turn"])
    if order != [turn["turn"] for turn in checked_turns]:
        raise ValueError("слова не соответствуют списку реплик")
    return {"kind": "conspect", "version": CURRENT_VERSION, "recording_id": recording_id,
            "source_revision": source_revision, "review_revision": review_revision, "goal": goal,
            "turns": checked_turns, "words": result}


def canonical(task):
    """Пересобрать задание своей версии; расхождение означает неизвестные поля."""
    if version(task) == 1:
        checked = build(task.get("recording_id"), task.get("source_revision"), task.get("words"),
                        task.get("goal"), task.get("review_revision"))
    else:
        if "turns" not in task:
            raise ValueError("в задании версии 2 нет списка реплик")
        checked = build(task.get("recording_id"), task.get("source_revision"), task.get("words"),
                        task.get("goal"), task.get("review_revision"), turns=task["turns"])
    if task != checked:
        raise ValueError("неизвестные поля задания")
    return checked


def validate(task, response, *, current_revision):
    """Проверить ссылки и ревизию; совпадение ссылок не доказывает смысл ответа."""
    checked = canonical(task)
    if type(current_revision) is not int or current_revision != checked["source_revision"]:
        raise ValueError("ревизия источника изменилась")
    if not isinstance(response, dict) or set(response) != {"source_revision", "sections"}:
        raise ValueError("неверный формат ответа")
    if (type(response["source_revision"]) is not int or response["source_revision"] != current_revision
            or not isinstance(response["sections"], list)):
        raise ValueError("неверная ревизия или разделы")
    known = {word["word_id"] for word in checked["words"]}
    if not response["sections"]:
        raise ValueError("ответ не содержит разделов")
    sections, covered = [], set()
    for section in response["sections"]:
        if not isinstance(section, dict) or set(section) != {"text", "evidence"}:
            raise ValueError("неверный формат раздела")
        text, evidence = section["text"], section["evidence"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("пустой раздел")
        if (not isinstance(evidence, list) or not evidence
                or any(not isinstance(i, str) or i not in known for i in evidence)
                or len(set(evidence)) != len(evidence)):
            raise ValueError("неверные ссылки на слова")
        covered.update(evidence)
        sections.append({"text": text, "evidence": list(evidence)})
    if not known or covered != known:
        raise ValueError("не все слова источника охвачены ссылками ответа")
    return {"source_revision": current_revision, "sections": sections,
            "actor_kind": "model", "human_verification": "none"}
