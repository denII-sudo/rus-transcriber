"""Схема ограниченного модельного задания и проверка ссылок на слова.

Версия 1 (без поля `version`) — только слова; так выглядят задания, сохранённые до версии 2,
они собираются и проверяются по-прежнему. Версия 2 (`"version": 2`) добавляет реплики:
у слова номер реплики `turn`, у задания список `turns` с условной меткой говорящего.
Версия 3 (`"version": 3`) — те же поля и `unlocated_doubts`: число открытых мест, которые не
удалось привязать к словам ни по ссылкам, ни по времени. Версии различаются и правилами
пометки `uncertain` (их применяет workflow при сборке), сами поля слов те же.
Версия 4 (`"version": 4`) — те же поля и `points`: пронумерованные пункты (реплика, а реплика
длиннее MAX_POINT_WORDS слов делится по концу предложения). Модель видит запись целиком как
пункты (`model_view`) и отвечает границами разделов «с пункта N по пункт M» вместо перечня
слов; программа сама раскладывает границы в прежнее поле `evidence` (список word_id).
Запись длиннее предела одного запроса (этап 7.4в) делится на части: служебное поле `parts` —
список пар [первый пункт, последний пункт] со сквозной нумерацией, не меньше двух частей. Поле
есть только у таких заданий; у записи до предела его нет, и задание побайтно прежнее. Задание
части — то же задание с полем `part` (номер части с 1): модель видит только пункты своей части,
разделы ответа не выходят за её границы. Поля `parts` и `part` модели не показываются.
Имена говорящих в задание не попадают: метка — только «Участник N» или UNKNOWN_SPEAKER.
"""
import re


CURRENT_VERSION = 4
VERSIONS = (2, 3, 4)    # версии с полем `version`; версия 1 — без него
MAX_POINT_WORDS = 120       # версия 4: пункт не длиннее; длинная реплика делится по концу предложения
MAX_SECTION_WORDS = 1500    # версия 4: раздел из нескольких пунктов не длиннее (около 12 минут речи)
MAX_ALSO = 3                # версия 4: отсылок к другим местам записи у раздела не больше
MAX_ALSO_POINTS = 20        # версия 4: отсылка не длиннее стольких пунктов
_SENTENCE_END = (".", "!", "?", "…")
_CLOSING = "»\"”’')]"
_DOUBT_OPEN, _DOUBT_CLOSE = "⟦", "⟧"
SPEAKER_LABEL = re.compile(r"Участник [1-9][0-9]{0,3}")
UNKNOWN_SPEAKER = "Говорящий не определён"


def speaker_label(number):
    """Условная метка говорящего по его порядковому номеру в записи."""
    if type(number) is not int or not 1 <= number <= 9999:
        raise ValueError("некорректный номер участника")
    return f"Участник {number}"


def version(task):
    """1 — без поля версии, 2 — с репликами, 3 — с числом непривязанных мест, 4 — с пунктами; иное отклоняется."""
    if not isinstance(task, dict):
        raise ValueError("некорректное задание")
    if "version" not in task:
        return 1
    if type(task["version"]) is not int or task["version"] not in VERSIONS:
        raise ValueError("неизвестная версия задания")
    return task["version"]


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


def _points(words):
    """Пункты версии 4 по словам с репликами: [{"point", "turn", "start", "end"}], end — не включая.

    Пункт — реплика. Реплика длиннее MAX_POINT_WORDS слов делится по последнему в пределах этого
    числа концу предложения, а если его нет — ровно по пределу. Пункты идут подряд без пропусков.
    """
    result, index = [], 0
    while index < len(words):
        stop = index
        while stop < len(words) and words[stop]["turn"] == words[index]["turn"]:
            stop += 1
        while index < stop:
            cut = min(index + MAX_POINT_WORDS, stop)
            if cut < stop:
                cut = next((k for k in range(cut, index, -1)
                            if words[k - 1]["text"].rstrip().rstrip(_CLOSING).endswith(_SENTENCE_END)), cut)
            result.append({"point": len(result) + 1, "turn": words[index]["turn"], "start": index, "end": cut})
            index = cut
    return result


def build(recording_id, source_revision, words, goal="", review_revision=1, *, turns=None,
          unlocated_doubts=None, points=False, parts=None):
    """Сформировать задание из слов и их устойчивых идентификаторов.

    Без `turns` — версия 1 (только слова). С `turns` — версия 2: каждое слово несёт номер
    своей реплики, реплики идут по порядку, пустых и неизвестных реплик нет. С `turns` и
    `unlocated_doubts` (целое ≥ 0) — версия 3. С ними же и points=True — версия 4: задание
    дополнительно несёт пункты, вычисленные из слов (см. _points); с parts — и деление на части.
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
    if parts is not None and not points:
        raise ValueError("части есть только в задании с пунктами")
    if points and unlocated_doubts is None:
        raise ValueError("пункты есть только в задании с репликами и числом непривязанных мест")
    if unlocated_doubts is not None:
        if turns is None:
            raise ValueError("число непривязанных мест есть только в задании с репликами")
        if type(unlocated_doubts) is not int or unlocated_doubts < 0:
            raise ValueError("некорректное число непривязанных мест")
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
    number = 2 if unlocated_doubts is None else 4 if points else 3
    task = {"kind": "conspect", "version": number, "recording_id": recording_id,
            "source_revision": source_revision, "review_revision": review_revision, "goal": goal,
            "turns": checked_turns, "words": result}
    if unlocated_doubts is not None:
        task["unlocated_doubts"] = unlocated_doubts
    if points:
        task["points"] = _points(result)
    if parts is not None:
        task["parts"] = _parts(parts, len(task["points"]))
    return task


def _parts(parts, last):
    """Части версии 4: пары [первый, последний] пункт подряд от 1 до last, не меньше двух."""
    if not isinstance(parts, list) or len(parts) < 2:
        raise ValueError("части задания — список не менее чем из двух пар номеров пунктов")
    result, following = [], 1
    for part in parts:
        if (not isinstance(part, list) or len(part) != 2 or any(type(x) is not int for x in part)
                or part[0] != following or part[1] < part[0]):
            raise ValueError("части задания должны идти подряд по пунктам без пропусков и пересечений")
        result.append([part[0], part[1]])
        following = part[1] + 1
    if following != last + 1:
        raise ValueError("части задания должны покрывать все пункты")
    return result


def part_task(task, number):
    """Задание части number (с 1) задания версии 4 с частями: то же задание с полем `part`."""
    checked = canonical(task)
    if "parts" not in checked or "part" in task:
        raise ValueError("задание не делится на части")
    if type(number) is not int or not 1 <= number <= len(checked["parts"]):
        raise ValueError("неверный номер части")
    return {**checked, "part": number}


def part_range(task):
    """Первый и последний пункт, которые видит модель: своей части или всего задания версии 4."""
    if "part" in task:
        first, final = task["parts"][task["part"] - 1]
        return first, final
    return 1, len(task["points"])


def canonical(task):
    """Пересобрать задание своей версии; расхождение означает неизвестные поля."""
    if isinstance(task, dict) and "part" in task:
        return part_task({key: value for key, value in task.items() if key != "part"}, task["part"])
    if version(task) == 1:
        checked = build(task.get("recording_id"), task.get("source_revision"), task.get("words"),
                        task.get("goal"), task.get("review_revision"))
    else:
        number = version(task)
        if "turns" not in task:
            raise ValueError(f"в задании версии {number} нет списка реплик")
        if number >= 3 and "unlocated_doubts" not in task:
            raise ValueError(f"в задании версии {number} нет числа непривязанных мест")
        checked = build(task.get("recording_id"), task.get("source_revision"), task.get("words"),
                        task.get("goal"), task.get("review_revision"), turns=task["turns"],
                        unlocated_doubts=task["unlocated_doubts"] if number >= 3 else None,
                        points=number == 4, parts=task.get("parts") if number == 4 else None)
    if task != checked:
        raise ValueError("неизвестные поля задания")
    return checked


def _point_text(words):
    """Текст пункта для модели: слова через пробел, подряд идущие сомнительные — в ⟦…⟧."""
    parts = []
    for index, word in enumerate(words):
        text = word["text"].replace(_DOUBT_OPEN, "").replace(_DOUBT_CLOSE, "")
        if word["uncertain"]:
            if index == 0 or not words[index - 1]["uncertain"]:
                text = _DOUBT_OPEN + text
            if index + 1 == len(words) or not words[index + 1]["uncertain"]:
                text += _DOUBT_CLOSE
        parts.append(text)
    return " ".join(parts)


def model_view(task):
    """Что видит модель в задании версии 4: пункты [номер, условная метка, текст] и общие поля.

    Идентификаторы слов, запись, номер ревизии разбора, время и деление на части сюда не входят;
    у задания части — только её пункты (номера сквозные).
    """
    checked = canonical(task)
    if version(checked) != 4:
        raise ValueError("вид с пунктами есть только у задания версии 4")
    speakers = {turn["turn"]: turn["speaker"] for turn in checked["turns"]}
    words = checked["words"]
    first, final = part_range(checked)
    return {"kind": checked["kind"], "version": 4, "source_revision": checked["source_revision"],
            "goal": checked["goal"], "unlocated_doubts": checked["unlocated_doubts"],
            "points": [[point["point"], speakers[point["turn"]], _point_text(words[point["start"]:point["end"]])]
                       for point in checked["points"][first - 1:final]]}


def _validate_sections(checked, response, current_revision):
    """Ответ версии 4: разделы с границами пунктов «from»–«to», подряд и без пересечений.

    У задания части разделы и отсылки also — только внутри её пунктов: раздел не переходит границу части.
    """
    points, words = checked["points"], checked["words"]
    start, last = part_range(checked)
    if not isinstance(response, dict) or set(response) != {"source_revision", "sections"}:
        raise ValueError("неверный формат ответа: нужны ровно поля source_revision и sections")
    if (type(response["source_revision"]) is not int or response["source_revision"] != current_revision
            or not isinstance(response["sections"], list)):
        raise ValueError("неверная ревизия или разделы")
    if not response["sections"] or not points:
        raise ValueError("ответ не содержит разделов")

    def ids(first, final):
        return [word["word_id"] for point in points[first - 1:final] for word in words[point["start"]:point["end"]]]

    sections, following = [], start
    for number, section in enumerate(response["sections"], 1):
        if (not isinstance(section, dict) or not {"from", "to", "text"} <= set(section)
                or set(section) - {"from", "to", "text", "also"}):
            raise ValueError(f"раздел {number}: нужны поля from, to, text и необязательное also")
        first, final, text = section["from"], section["to"], section["text"]
        if type(first) is not int or type(final) is not int or not start <= first <= final <= last:
            raise ValueError(f"раздел {number}: границы — целые номера пунктов от {start} до {last}, "
                             "from не больше to")
        if first != following:
            raise ValueError(f"раздел {number} начинается с пункта {first}, а должен — с пункта {following}, "
                             "сразу за концом предыдущего раздела (без пропусков и пересечений)")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"раздел {number}: пустой текст")
        evidence = ids(first, final)
        if first != final and len(evidence) > MAX_SECTION_WORDS:
            raise ValueError(f"раздел {number} (пункты {first}–{final}) длиннее {MAX_SECTION_WORDS} слов: "
                             "разделите его на несколько")
        also = section.get("also", [])
        if not isinstance(also, list) or len(also) > MAX_ALSO:
            raise ValueError(f"раздел {number}: also — список не более чем из {MAX_ALSO} отсылок")
        refs = []
        for ref in also:
            if (not isinstance(ref, list) or len(ref) != 2 or any(type(x) is not int for x in ref)
                    or not start <= ref[0] <= ref[1] <= last or ref[1] - ref[0] + 1 > MAX_ALSO_POINTS):
                within = "" if "part" not in checked else f" от {start} до {last} (внутри своей части)"
                raise ValueError(f"раздел {number}: отсылка also — пара [from, to] номеров пунктов{within}, "
                                 f"не длиннее {MAX_ALSO_POINTS} пунктов")
            if ref[0] <= final and ref[1] >= first:
                raise ValueError(f"раздел {number}: отсылка also должна вести вне своего раздела")
            refs.append(ids(ref[0], ref[1]))
        sections.append({"text": text, "evidence": evidence, "also": refs})
        following = final + 1
    if following != last + 1:
        raise ValueError(f"разделы кончаются пунктом {following - 1}, а последний пункт — {last}: "
                         "последний раздел должен кончаться последним пунктом")
    return {"source_revision": current_revision, "sections": sections,
            "actor_kind": "model", "human_verification": "none"}


def validate(task, response, *, current_revision):
    """Проверить ссылки и ревизию; совпадение ссылок не доказывает смысл ответа.

    Версия 4 — границы разделов по пунктам; итог в том же виде, что у прежних версий
    (`evidence` — word_id основного отрезка), плюс `also` — word_id каждой отсылки списком.
    """
    checked = canonical(task)
    if type(current_revision) is not int or current_revision != checked["source_revision"]:
        raise ValueError("ревизия источника изменилась")
    if version(checked) == 4:
        return _validate_sections(checked, response, current_revision)
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
