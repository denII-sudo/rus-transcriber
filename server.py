"""MCP-сервер для приложений ИИ-агентов (например, Claude Desktop) и других MCP-клиентов: тонкий слой над core.py.

Вся работа с задачами, стенограммами, настройками и данными лежит в core.py (там нет ничего про MCP);
опасные действия подтверждает человек через confirm.py. Здесь — только описание инструментов для клиента.
Тяжёлая обработка идёт в отдельном процессе, поэтому длинные записи не упираются в таймаут вызова.
"""
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:  # MCP SDK 2.x
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp import FastMCP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402  (импортирует paths первым: дочерние процессы унаследуют HF_HOME и др.)
import paths  # noqa: E402
import settings  # noqa: E402
import confirm  # noqa: E402
import storage  # noqa: E402

JOBS = core.JOBS
def _status(job_id):
    if isinstance(job_id, str) and job_id.startswith("recording:"):
        return core._recording_reference(job_id)
    return core.job_status(job_id)
_launch = core.launch
_job_json = core.job_json
# Перенос старой раскладки данных при старте больше не выполняется: только явной командой
# migrate_data (с отчётом без применения и подтверждением человека) или python migrate.py.

INSTRUCTIONS = """\
rus-transcriber: расшифровка речи на этом компьютере. При заказанном конспекте П4 автоматически отправляет
только текст стенограммы с внесёнными исправлениями, пометки сомнительных мест, деление на реплики и условные
метки говорящих («Участник 1»…), номер версии стенограммы и цель в OpenAI GPT-6 Luna через OpenCode (задание,
начатое до обновления программы, доделывается в прежнем виде — с ID слов и номерами ревизий). Подписи
говорящих именами, аудио, время слов, биометрия,
пути и журналы не передаются; метки расшифровываются именами только в файле конспекта на этом компьютере.
Имена и сведения, произнесённые в записи или написанные в цели, уходят вместе с текстом. Запись длиннее
10 000 слов отправляется частями; к каждой части прилагается изложение предыдущих, составленное моделью.
Ответ остаётся модельным предложением и проходит локальную проверку; он не является подтверждением человека. Рабочий процесс:

П3/P4: start_workflow → workflow_status. Облако только по явному согласию человека: allow_cloud_model
по умолчанию false; передавайте true, лишь если человек прямо согласился на отправку текста стенограммы
в OpenAI для этого запроса. Без согласия заказанный конспект ждёт исполнителя и ничего не отправляется.
Не пытайтесь заменять модельный ответ или подтверждать его от имени человека.
Ненайденные цитаты готовой свёртки модель исправляет только по просьбе человека: workflow_repair_quotes
(allow_cloud_model=true — лишь при его прямом согласии на эту отправку) или кнопка на экране П5.
П5: для человека есть локальный экран (rt.py screen); он же показывает историю, конфликты и резерв.
Реальные и старые данные не подключать до отдельной миграции.

1. Запуск: start_transcription(audio_path, names="Анна, Борис", group="…"). names — участники записи, если известны
   (голоса ищутся только среди них); num_speakers — только если число голосов известно точно; group нужен, когда в
   папке данных несколько групп (voice_profiles покажет их). emotions=false ускоряет, если эмоции не нужны.
2. Ожидание: transcription_status(job_id, wait_sec=50) — вызов сам ждёт и отвечает раньше при смене этапа; вызывайте
   подряд без пауз и без других опросов. Отменить — cancel_job. Одновременно идут не больше двух тяжёлых задач.
3. Чтение: read_transcript(job_id) — реплики без служебных разделов. ⟦…⟧ у слов — нерешённые сомнительные места:
   в цитаты и выводы без проверки не брать. Стенограмму не копируйте в разговор целиком и не переписывайте вручную:
   путь к файлу есть в transcription_status.
4. Сомнительные места решает человек: review_items(job_id) → варианты (много мест — сначала brief=True и фильтры
   reasons/min_sec); build_review_pack → start_review_page (обычный локальный браузер, ИИ-агент не нужен)
   → apply_answers. Облачные карточки необязательны; их ответы сначала передаются через import_answers;
   прямые решения — apply_review. Приоритет: слово человека > голосовой профиль > распознавание > эвристики.
   Не выбирайте за человека, кто говорит в спорном месте. Остатки после разбора (слово вне места, неверное
   число) — edit_span(job_id, t1, t2, text, speaker, by="claude") — правка агента, автор by обязателен; повторно решить место — reopen_review.
5. Голоса: rename_speakers подтверждает имена в расшифровке; preview_voice_profiles и
   update_voice_profiles пересчитывают план. Параметр group_override допустим только при неизвестной
   привязке и будет показан человеку до штатного подтверждения; resolve_voice — решение о тёзке.
   Автопополнение зависит от настройки группы и отключается при достаточности; новые голоса по-прежнему обнаруживаются.
   voice_refresh_plan показывает изменённые источники без записи; voice_quality и set_voice_quality_policy —
   диагностика и политика warn/reject_clipping; backup_voice_profiles делает открытый резерв биометрии.
6. Конспект пишете вы: цитаты «…» берите только из стенограммы; check_quotes проверяет их, conspect_link связывает
   с местами стенограммы, conspect_render собирает документ.
7. После обновления программы — self_check (тесты, проверка файлов на личные данные, морфология).
8. Данные: get_settings (папка данных, группы, настройки). export_group, import_group (apply=true), delete_group,
   cleanup_jobs (apply=true), cleanup_hypotheses (apply=true), migrate_data (dry_run=false) меняют или выгружают личные данные — только по прямой
   просьбе человека, после отчёта в режиме «без применения». Каждое такое действие дополнительно подтверждает сам
   человек в окне на экране (или командой `python rt.py approve` в терминале): подтвердить за него нельзя, ответ
   «Человек не подтвердил» значит отказ — не повторяйте вызов без новой просьбы. Архив export_group содержит
   голосовые отпечатки (биометрию): не читайте его и не передавайте в разговор.
"""

VERSION = "2026.11.24.11"   # версия программы; видна клиентам в ответе initialize (serverInfo.version)

try:
    mcp = FastMCP("rus-transcriber", instructions=INSTRUCTIONS, version=VERSION)
except TypeError:   # mcp 1.x: параметра version нет
    mcp = FastMCP("rus-transcriber", instructions=INSTRUCTIONS)


async def _gated(action, args, lines_fn, run_fn, *, require_human=False):
    """Выполняет run_fn; опасное действие (action не None) — только после подтверждения человеком (confirm.py).
    Всё блокирующее (окно подтверждения, работа с файлами) идёт в потоке, чтобы не задерживать другие вызовы."""
    def work():
        try:
            if action:
                method = confirm.require(action, args, lines_fn())
                if require_human and method not in ("dialog", "terminal"):
                    raise confirm.NotConfirmed("Требуется подтверждение человека; без него действие не выполнено.")
            return json.dumps(run_fn(), ensure_ascii=False, indent=1)
        except confirm.NotConfirmed as ex:
            return str(ex)
        except core.CoreError as ex:
            return f"Ошибка: {core.say_error(ex)}"
        except (storage.StorageError, ValueError, KeyError, OSError) as ex:
            return f"Ошибка: {core.say_error(ex)}"
    return await asyncio.to_thread(work)


@mcp.prompt()
def transcribe_recording(audio_path: str, participants: str = "") -> str:
    """Расшифровать запись: запуск, ожидание, чтение и краткий итог."""
    names = f' names="{participants}"' if participants else ""
    return (f"Расшифруй запись {audio_path}. Вызови start_transcription(audio_path=...{names}), затем жди через "
            "transcription_status(job_id, wait_sec=50) до состояния done. Прочитай read_transcript(job_id) и коротко "
            "сообщи: сколько участников и кто, длительность, сколько сомнительных мест (⟦…⟧), что делать дальше. "
            "Стенограмму целиком в разговор не копируй.")


@mcp.prompt()
def review_doubts(job_id: str) -> str:
    """Разобрать сомнительные места расшифровки вместе с человеком."""
    return (f"Разбери сомнительные места расшифровки {job_id}: review_items(job_id), затем предложи человеку варианты "
            "(карточки, если клиент умеет показывать страницы; иначе по несколько мест списком). Решения человека "
            "применяй через apply_review или import_answers → apply_answers. Сам не выбирай, кто говорит в спорном "
            "месте. В конце сообщи, сколько мест решено и сколько осталось.")


@mcp.prompt()
def make_conspect(job_id: str, focus: str = "") -> str:
    """Написать конспект по расшифровке с проверенными цитатами."""
    f = f" Акцент: {focus}." if focus else ""
    return (f"Напиши конспект расшифровки {job_id}.{f} Читай read_transcript(job_id) частями. Цитаты «…» бери "
            "только из стенограммы и не из мест ⟦…⟧. Проверь check_quotes, затем conspect_link и conspect_render.")


@mcp.tool()
def start_workflow(audio_path: str, require_conspect: bool = False,
                   allow_cloud_model: bool = False,
                   require_named_speakers: bool = False, require_human_verification: bool = False,
                   participation: str = "simple", goal: str = "", question_budget: int = 0) -> str:
    """Начать процесс на новой SQLite-схеме. allow_cloud_model=true — только при явном согласии человека
    на отправку текста стенограммы (без аудио, путей и биометрии) в OpenAI для конспекта; по умолчанию false.
    Запись длиннее 10 000 слов отправляется частями; к каждой части прилагается изложение предыдущих,
    составленное моделью."""
    try:
        result = core.start_workflow(audio_path, {
            "require_conspect": require_conspect,
            "allow_cloud_model": allow_cloud_model,
            "require_named_speakers": require_named_speakers,
            "require_human_verification": require_human_verification,
            "participation": participation, "goal": goal, "question_budget": question_budget,
        })
        return json.dumps(result, ensure_ascii=False)
    except (core.CoreError, storage.StorageError, ValueError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def workflow_status(request_id: str) -> str:
    """Состояние сохранённого запроса и отдельные ограничения готовности."""
    try:
        return json.dumps(core.workflow_status(request_id), ensure_ascii=False)
    except (core.CoreError, storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def workflow_model_task(request_id: str, offset: int = 0, limit: int = 1000, restart: bool = False) -> str:
    """Получить страницу задания. Прочитайте все страницы до next_offset=null. Задание версии 4: points —
    пункты [номер, условная метка говорящего, текст], слова в ⟦…⟧ распознаны под сомнением и не проверены
    человеком (не цитируйте их дословно); offset и limit считают пункты, всего — total_points. Ответ
    (workflow_model_result): {"source_revision": как в задании, "sections": [{"from": пункт, "to": пункт,
    "text": текст раздела, "also": [[с пункта, по пункт], ...]}]} — разделы подряд с пункта 1 до последнего,
    без пропусков и пересечений; раздел из нескольких пунктов не длиннее 1500 слов; also необязательно — до
    3 отсылок вне своего раздела, каждая до 20 пунктов. В заданиях версий 1–3 (начатых до обновления) —
    слова с word_id, ответ — evidence. В задании версии 2 у слова
    есть номер реплики turn, а в turns — условные метки говорящих («Участник N») для реплик этой страницы:
    называйте говорящих только этими метками, имён в задании нет. uncertain: true — распознавание слова
    под сомнением и не проверено человеком: не цитируйте его дословно и не опирайтесь на него как на точное;
    в задании версии 3 unlocated_doubts — число сомнительных мест, не привязанных к словам (если оно больше
    нуля, помечены все слова). У готовой свёртки возвращается
    сохранённое задание, ничего не меняется. restart=true (только с offset=0) — начать модельный шаг
    заново: после ошибки или для новой свёртки; прежние блоки не используются."""
    try:
        return json.dumps(core.workflow_model_task(request_id, offset, limit, restart=restart), ensure_ascii=False)
    except (storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def workflow_questions(request_id: str) -> str:
    """Ограниченный политикой список добровольных уточнений; не выбирает ответы."""
    try:
        return json.dumps(core.workflow_questions(request_id), ensure_ascii=False)
    except (storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
async def workflow_abandon_request(request_id: str) -> str:
    """Отказаться от незавершённого запуска с неизвестным исходом после подтверждения человеком."""
    try:
        snapshot = core.workflow_abandon_snapshot(request_id)
        if confirm.mode() == "off":
            return "Ошибка: для отказа от результата требуется подтверждение человека"
    except (storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)
    return await _gated(
        "Отказ от незавершённого запуска", snapshot,
        lambda: [f"Источник: {snapshot['source']}", f"Задача: {snapshot['job_id']}",
                 "Результат старого запуска больше нельзя будет сохранить.",
                 "Процесс не останавливается; для новой попытки потребуется новый запрос."],
        lambda: core.workflow_abandon(snapshot), require_human=True)


@mcp.tool()
async def workflow_mark_reviewed(request_id: str) -> str:
    """Отдельно подтвердить, что человек проверил всю текущую стенограмму."""
    try:
        snapshot = core.workflow_verification_snapshot(request_id)
        if confirm.mode() == "off":
            return "Ошибка: для подтверждения проверки нужен канал подтверждения человека"
    except (storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)
    return await _gated(
        "Подтверждение проверки стенограммы", snapshot,
        lambda: [f"Стенограмма: {snapshot['document']}",
                 f"Ревизия: {snapshot['source_revision']}; ревизия разбора: {snapshot['review_revision']}",
                 "Подтвердите только если самостоятельно проверили всю стенограмму.",
                 "Это не подтверждение конспекта или точности модели."],
        lambda: core.workflow_confirm_verification(snapshot), require_human=True)


@mcp.tool()
def workflow_change_policy(request_id: str, policy: dict, expected_revision: int) -> str:
    """Изменить требования с сохранением предыдущей политики, без повторного ASR."""
    try:
        return json.dumps(core.workflow_change_policy(request_id, policy, expected_revision), ensure_ascii=False)
    except (storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def workflow_execute_model(request_id: str, retry_failed: bool = False, restart: bool = False) -> str:
    """Продолжить автономное модельное задание без повторного распознавания. retry_failed=true — продолжить
    шаг после ошибки (только недостающие блоки); restart=true — начать шаг заново: текст стенограммы снова
    уйдёт в облако. Вместе их не передавать. У готовой свёртки повтор облако не вызывает: дописывает файлы
    или называет причину. Предел размера и ручная правка файла свёртки проверяются до вызова облака."""
    import llm_provider
    try:
        return json.dumps(core.workflow_execute_model(request_id, retry_failed=retry_failed, restart=restart),
                          ensure_ascii=False)
    except (core.CoreError, storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)
    except llm_provider.ProviderError as ex:
        return json.dumps({"error_code": ex.code, "error": str(ex)}, ensure_ascii=False)


@mcp.tool()
def workflow_repair_quotes(request_id: str, allow_cloud_model: bool = False) -> str:
    """Попросить облачную модель исправить цитаты готовой модельной свёртки, не найденные в стенограмме.
    Доступно, когда таких цитат не меньше 3 и не меньше 20 % всех цитат свёртки и политика запроса разрешает
    облако. allow_cloud_model=false (по умолчанию) ничего не отправляет: сообщает, доступно ли исправление,
    почему нет, сколько цитат, разделов и слов уйдёт. allow_cloud_model=true передавайте, только если человек
    прямо согласился на эту отправку: в OpenAI GPT-6 Luna через OpenCode уйдут только разделы свёртки с
    ненайденными цитатами — текст раздела, сами цитаты и текущий текст стенограммы этих мест (с исправлениями,
    делением на реплики, условными метками «Участник N» и пометками ⟦…⟧); имена и сведения, произнесённые там,
    уходят вместе с текстом. Подписи говорящих именами, аудио, время слов, пути, биометрия, журналы, цель и
    остальной текст не передаются. Один вызов на одно обращение, без автоповтора; программа сама его не
    запускает. Ответ — модельное предложение, а не подтверждение человека: каждая новая цитата проверяется
    локально, не найденная остаётся с пометкой. Повторяйте только по новой просьбе человека."""
    import llm_provider
    try:
        if type(allow_cloud_model) is not bool:
            raise ValueError("allow_cloud_model — логическое значение true или false")
        if not allow_cloud_model:
            return json.dumps(core.workflow_repair_quotes_info(request_id), ensure_ascii=False)
        return json.dumps(core.workflow_repair_quotes(request_id, via="mcp", allow_cloud_model=True),
                          ensure_ascii=False)
    except (core.CoreError, storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)
    except llm_provider.ProviderError as ex:
        return json.dumps({"error_code": ex.code, "error": str(ex)}, ensure_ascii=False)


@mcp.tool()
def workflow_model_part(request_id: str, part_index: int) -> str:
    """Получить один блок задания. У задания версии 4 блок один — вся запись пунктами, как в
    workflow_model_task. У версий 1–3 блок не длиннее 500 слов и кончается на границе реплики;
    реплика длиннее 500 слов делится по концу предложения."""
    try:
        return json.dumps(core.workflow_model_part(request_id, part_index), ensure_ascii=False)
    except (storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def workflow_model_part_result(request_id: str, part_index: int, result: dict) -> str:
    """Сохранить модельный ответ на блок; документ создаётся после получения всех блоков."""
    try:
        return json.dumps(core.workflow_model_part_result(request_id, part_index, result), ensure_ascii=False)
    except (storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def workflow_model_result(request_id: str, result: dict) -> str:
    """Передать модельный ответ; он не становится подтверждением человека."""
    try:
        return json.dumps(core.workflow_model_result(request_id, result), ensure_ascii=False)
    except (storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def workflow_list(limit: int = 20) -> str:
    """Последние сохранённые запросы workflow: состояния без тяжёлых чтений."""
    try:
        return json.dumps(core.workflow_list(limit), ensure_ascii=False)
    except (storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def recording_history(recording_id: str, kind: str = "transcript") -> str:
    """История ревизий SQLite-записи (kind: transcript или review) с авторством; без изменения текущей."""
    try:
        return json.dumps(core.recording_revisions(recording_id, kind), ensure_ascii=False)
    except (core.CoreError, storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def read_recording_revision(recording_id: str, kind: str, revision: int) -> str:
    """Прочитать сохранённую ревизию записи для просмотра истории; текущая ревизия не меняется."""
    try:
        return json.dumps(core.read_recording_revision(recording_id, kind, revision), ensure_ascii=False)
    except (core.CoreError, storage.StorageError, ValueError, KeyError, OSError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
async def restore_recording_revision(recording_id: str, source_revision: int, review_revision: int) -> str:
    """Вернуть прежнюю ревизию парой (текст + разбор) как новую текущую после подтверждения человека."""
    try:
        snapshot = core.prepare_restore_snapshot(recording_id, source_revision, review_revision)
        if confirm.mode() == "off":
            return "Ошибка: для возврата ревизии нужен канал подтверждения человека"
    except (core.CoreError, storage.StorageError, ValueError, KeyError, OSError) as ex:
        return f"Ошибка: {core.say_error(ex)}"
    return await _gated(
        "Возврат прежней ревизии как текущей", snapshot,
        lambda: [f"Стенограмма: {snapshot['document']}",
                 f"Возвращается пара: текст {snapshot['source_revision']}, разбор {snapshot['review_revision']}.",
                 f"Текущая пара {snapshot['expected_versions']['source_revision']}/"
                 f"{snapshot['expected_versions']['review_revision']} станет предыдущей в истории.",
                 "История сохраняется; файлы документа формируются заново."],
        lambda: core.restore_recording_revision(snapshot), require_human=True)


def _backup_for_agent(destination):
    """Агенту — число и номера повреждённых копий: пути файлов остаются экрану и rt.py."""
    result = core.backup_store(destination)
    copies = result["report"].get("damaged_copies")
    if copies:
        result["report"]["damaged_copies"] = {"count": len(copies),
                                              "conflict_ids": [item["conflict_id"] for item in copies]}
        result["warning"] = result["warning"][:1]
    return result


@mcp.tool()
async def backup_state(destination: str) -> str:
    """Создать проверенную неактивную копию базы состояния; аудио и папки документов она не заменяет.
    Если в ответе есть warning (повреждённые копии правок), передайте его человеку."""
    return await _gated(
        "Создание резерва состояния БД", {"destination": destination},
        lambda: [f"Новый файл: {destination}",
                 "Копия неактивна: установки отключены, незавершённые запуски прерываются.",
                 "Существующие файлы не изменяются и не удаляются.",
                 "Если в базе есть повреждённые копии правок, резерв всё равно будет создан с предупреждением."],
        lambda: _backup_for_agent(destination), require_human=True)


@mcp.tool()
def cancel_job(job_id: str) -> str:
    """Отменяет задачу (расшифровку, тест скорости, пересчёт профилей), идущую или ждущую в очереди:
    процесс завершается вместе с дочерними, место в очереди освобождается. Готовые прежние результаты
    (расшифровки, профили) не трогаются: файлы задачи пишутся только в её конце."""
    try:
        return json.dumps(core.cancel_job(job_id), ensure_ascii=False)
    except core.CoreError as ex:
        return str(ex)


@mcp.tool()
def start_transcription(audio_path: str, num_speakers: int = 0, emotions: bool = True,
                        vocab_path: str = "", tag: str = "", voices: bool = True, names: str = "",
                        whisper: str = "", whisper_max_repeat: int = -1, group: str = "") -> str:
    """Запускает расшифровку русской записи с разделением по спикерам и эмоциями.

    audio_path: полный путь к файлу на этом компьютере (mp3, m4a, wav, ogg, mp4 и т.д.).
    num_speakers: точное число голосов в записи, если известно (0 — определить автоматически).
    names: имена участников этой записи через запятую («Анна, Вера, Борис, Глеб»), если известны: имена по
      голосу ищутся только среди них; голосов — не меньше их числа и с запасом в один посторонний
      (точное число — только через num_speakers). Имя без голосового профиля останется «Спикер N».
    emotions: определять эмоцию по голосу для каждой реплики.
    whisper: третья модель голосования «два из трёх» (asr_third_model в config.json; по умолчанию Whisper,
      имя параметра прежнее): "all" — вся запись; "doubt" — только куски, где основная и вторая модели
      расходятся (в остальных голосование решено без неё; быстрее, но то, в чём обе ошиблись одинаково,
      третья модель не исправит — этого не исправляет и "all"); "off" — без неё (самое быстрое, нет
      голосования); пусто — как в config.json (asr_third_mode, по умолчанию "all").
    whisper_max_repeat: только для третьей модели Whisper — сколько раз подряд она может повторить одну
      фразу (защита от зацикливания); 0 — без ограничения; -1 — как в config.json (по умолчанию 3). Её
      «субтитры» на тишине убираются всегда. Текст другой третьей модели не чистится.
    vocab_path: словарь группы; пусто — rus-transcriber.vocab.json рядом с записью (или до двух папок
      выше), иначе словарь группы в папке данных (DATA_DIR\\groups\\<группа>\\vocab.json).
    group: группа (имя папки в данных; см. voice_profiles): её словарь и голосовые профили; сильнее словаря
      рядом с записью. Пусто — группа из словаря рядом с записью, иначе default_group в config.json, иначе
      единственная группа; если групп несколько и ни одна не выбрана, расшифровка не стартует.
    tag: метка в имени результата (<имя>.<tag>.transcript.md), чтобы не перезаписать прежнюю расшифровку.
    voices: голосовые профили — имена по голосу, разрезание реплик с чужим голосом, проверка коротких
      реплик и рода глаголов. Профили пополняются сами из расшифровок в папке записи с подтверждёнными
      именами (rename_speakers).
    Одновременно идут не больше max_heavy_jobs тяжёлых задач (config.json в папке данных, по умолчанию 2), остальные ждут
    в состоянии queued. Отменить — cancel_job(job_id).
    Возвращает job_id. Обработка идёт в фоне; проверяйте transcription_status.
    """
    try:
        r = core.start_transcription(audio_path, num_speakers, emotions, vocab_path, tag, voices, names,
                                     whisper, whisper_max_repeat, group)
    except core.CoreError as ex:
        return str(ex)
    return json.dumps({**r, "hint": "Ждите через transcription_status(job_id, wait_sec=50): вызов сам ждёт и отвечает раньше при смене этапа — вызывать подряд без паузы не нужно."}, ensure_ascii=False)


@mcp.tool()
def voice_profiles() -> str:
    """Голосовые профили на этом компьютере: группы, участники, из скольких записей и секунд собраны.
    Сами отпечатки голосов не выводятся."""
    return json.dumps(core.voice_profiles(), ensure_ascii=False, indent=2)


@mcp.tool()
def start_benchmark(audio_path: str, seconds: int = 180, only: str = "", chunk_max: float = 20.0,
                    min_similarity: float = 0.0) -> str:
    """Сравнивает варианты распознавания на этом компьютере по скорости и точности на отрывке
    записи и сохраняет лучший выбор в config.json этой машины. Запускать один раз после установки
    и после обновления драйверов.

    Порядок: процессор → видеокарта Intel Arc (OpenVINO) → NPU. NPU Meteor Lake (Core Ultra
    Series 1) пропускается: он не поддерживает FP32 и считает GigaAM неверно. NPU новее
    проверяется и выбирается только при совпадении текста с эталоном. Основной проход RNN-T,
    второй проход CTC и Whisper проверяются отдельно и получают свои настройки.

    audio_path: путь к типичной записи с русской речью. seconds: длина отрывка для теста.
    only: id вариантов через запятую. RNN-T: torch-cpu (эталон), onnx-cpu, onnx-cpu-b8, onnx-cpu-b16,
    ov-gpu, ov-gpu-b8, ov-gpu-pad5, ov-gpu-pad22, dml-gpu, ov-npu-pad22. CTC:
    ctc-cpu, ctc-cpu-b8, ctc-gpu, ctc-gpu-b8, ctc-npu-pad22. Whisper: whisper-gpu, whisper-cpu.
    Пусто — основной набор: RNN-T, CTC и Whisper на процессоре и видеокарте.
    chunk_max: максимальная длина фрагмента в секундах (по умолчанию 20 — как в рабочем конвейере).
    min_similarity: порог совпадения текста с эталоном, 0..1. 0 — взять `bench_min_similarity`
    из config.json (по умолчанию 0.97). Поднимите до 0.99, если точность важнее скорости:
    тогда тест выберет более точный вариант, даже если он медленнее.
    """
    try:
        r = core.start_benchmark(audio_path, seconds, only, chunk_max, min_similarity or None)
    except core.CoreError as ex:
        return str(ex)
    return json.dumps({**r, "hint": "Тест занимает 5–30 минут (первая компиляция под "
                       "видеокарту долгая). Проверяйте transcription_status."}, ensure_ascii=False)


@mcp.tool()
async def rename_speakers(job_id: str, names: dict[str, str]) -> str:
    """Подписывает спикеров именами в готовой расшифровке (переписывает .md и .json).

    names: соответствие меток именам, например {"Спикер 1": "Вера", "Спикер 4": "Анна"}.
    Ключом можно указать и уже данное имя, чтобы исправить его. Исходные метки сохраняются
    в .json (поле label), поэтому переименовывать можно сколько угодно раз.
    """
    import recordings
    import transcript
    import unicodedata
    try:
        s = _status(job_id)
        if not s or s.get("state") != "done" or "json" not in s:
            return "Расшифровка не найдена или ещё не готова"
        store, binding = recordings.open_for_document(s["json"])
        sqlite_mode = store is not None
        if store is not None:
            store.close()

        def run(expected=None):
            try:
                data, unknown = transcript.rename(s["json"], s["markdown"], names,
                                                  expected_versions=expected)
            except (ValueError, storage.StorageError) as ex:
                raise core.CoreError(core.say_error(ex)) from ex
            msg = {"speakers": data["speakers"], "mapping": data["speaker_names"], "markdown": s["markdown"]}
            if data.get("_conspects"):
                msg["conspects"] = data["_conspects"]
            if unknown:
                msg["warning"] = f"таких меток нет в расшифровке: {unknown}"
            return msg

        if (not isinstance(names, dict) or not 1 <= len(names) <= 6 or
                any(not isinstance(k, str) or not isinstance(v, str) or not k.strip() or not v.strip() or
                    len(k) > 80 or len(v) > 80 or
                    any(unicodedata.category(ch).startswith("C") for ch in k + v)
                    for k, v in names.items())):
            return "Имена нельзя полностью и однозначно показать для подтверждения"
        lines = [f"{key} → {value}" for key, value in names.items()]
        if sum(len(line) + 1 for line in lines) > 900:
            return "Список имён слишком длинный для подтверждения"
        if not sqlite_mode:
            return await _gated("Подписать говорящих", {"job_id": job_id, "names": names},
                                lambda: ["Подтвердить имена говорящих?"] + lines,
                                lambda: run(), require_human=True)
        before = transcript.load(s["json"])
        expected = {"source_revision": before["source_revision"], "review_revision": before["review_revision"]}
        recording_id = before["recording_id"]
        return await _gated("Подписать говорящих",
                            {"job_id": job_id, "recording_id": recording_id, "names": names, **expected},
                            lambda: [f"Запись: {recording_id}; ревизия текста: {expected['source_revision']}",
                                     "Подтвердить имена говорящих?"] + lines,
                             lambda: run(expected), require_human=True)
    except (OSError, ValueError, storage.StorageError, core.CoreError) as ex:
        return f"Ошибка: {core.say_error(ex)}"


@mcp.tool()
def review_items(job_id: str, offset: int = 0, limit: int = 40, reasons: str = "", min_sec: float = 0.0,
                 brief: bool = False, status: str = "open") -> str:
    """Сомнительные места готовой расшифровки (пословная обработка) — для смыслового разбора ИИ-агентом.

    Метки: asr — модели распознали по-разному (alt_ctc — вариант второй модели, alt_whisper — третьей;
    какие это модели — alt_models, по умолчанию GigaAM CTC и Whisper); conf — модель не уверена;
    voice — голоса близки (voices — сходство с участниками); overlap — говорят одновременно;
    gender — род глагола при «я» не совпадает с полом говорящего; missed — речь без слов модели.
    У каждого места: реплика с выделенным местом ⟦…⟧, соседние реплики (before/after).
    Фильтры: reasons — через запятую (например "voice,gender"); min_sec — места не короче (с);
    brief=True — кратко, без контекста (до 200 мест за раз, иначе до 60); status — open (по умолчанию),
    human, resolved или all. В ответе by_reason — сколько мест по каждой причине, next_offset — продолжение.
    Много мест — сначала brief=True, потом подробно только нужные. Разбирать по смыслу (вопрос педагога —
    ответ ребёнка, обращение по имени, род глагола), выбирая из имеющихся вариантов; решения — apply_review."""
    try:
        return json.dumps(core.review_items(job_id, offset, limit, reasons, min_sec, brief, status),
                          ensure_ascii=False, indent=None if brief else 1)
    except core.CoreError as ex:
        return str(ex)


@mcp.tool()
def conversation_hotspots(job_id: str, limit: int = 20, min_score: float = 0.0) -> str:
    """Горячие места разговора — отрезки, где разговор был оживлён или напряжён. Только по просьбе человека:
    сами по себе в стенограмму, конспект и ответы не добавляются.

    В ответе hotspots: t1, t2 (секунды), time, score (0…1) и kinds — что измерено: «частые смены говорящих»,
    «наложения», «громче обычного», «тон выше обычного» (громкость и высота — относительно обычных для
    того же говорящего в этой записи). Это эвристика для навигации (где развернуть стенограмму или
    послушать запись), а не оценка эмоций или поведения людей и не доказательство: не пересказывайте её
    как «человек злился/волновался», пороги стартовые, на настоящей речи не проверялись.
    limit — сколько мест показать (самые выраженные, по времени; 0 — все), min_score — нижняя граница оценки.
    available=false — данных нет, причина в reason: запись обработана до появления слоя признаков, настройка
    interaction была выключена или расчёт слоя не удался; старые записи не пересчитываются.
    job_id — задача или "recording:<id>" записи в SQLite."""
    try:
        return json.dumps(core.conversation_hotspots(job_id, limit, min_score), ensure_ascii=False, indent=1)
    except core.CoreError as ex:
        return str(ex)


@mcp.tool()
def edit_span(job_id: str, t1: float, t2: float, text: str, speaker: str = "", note: str = "",
              by: str = "") -> str:
    """Правка произвольного отрезка стенограммы — для остатков, которые не исправил разбор, без пересборки
    карточек. Слова, начинающиеся в [t1, t2] (секунды записи), заменяются текстом text по словам (совпадающие
    сохраняют время); speaker — подписать отрезок этим участником (реплика делится); пустой text — удалить
    слова отрезка; на отрезке без слов text вставляется (speaker обязателен); на отрезке слов нескольких говорящих
    speaker тоже обязателен. Места разбора внутри отрезка закрываются (в ответе resolved — только закрытые по
    сохранённой записи; seen_changed — слова, которые человек видел в своей правке отрезка и которые вы
    изменили). Пишется в журнал правок. by обязателен: "claude" — правка агента (в том числе по поручению
    человека); "human" через MCP не принимается; без by правка не принимается.
    note — пояснение для журнала. Время слов — read_transcript(fmt="json", extra=True) или review_items."""
    try:
        if by == "human":
            return "Правку с авторством человека вносите через локальную карточку с подтверждением"
        return json.dumps(core.edit_span(job_id, t1, t2, text, speaker, note, by), ensure_ascii=False, indent=1)
    except (core.CoreError, storage.StorageError, ValueError) as ex:
        return core.say_error(ex)


@mcp.tool()
def reopen_review(job_id: str, ids: str) -> str:
    """Снова открывает места разбора (решённые или оставленные человеку), например "r57,r88", — чтобы
    решить их заново через apply_review. Карточки пересобирать не нужно."""
    try:
        return json.dumps(core.reopen_review(job_id, ids), ensure_ascii=False, indent=1)
    except (core.CoreError, storage.StorageError) as ex:      # запись ревизии: исход недописанного файла словами
        return core.say_error(ex)


@mcp.tool()
def legacy_speaker_review(job_id: str, build_pack: bool = False) -> str:
    """Старые подтверждения говорящего (до пословных отметок): места, где по истории правок видно решение
    человека: смена говорящего, место, принятое как есть, или правка текста (её говорящий подтверждается
    отдельно); места короче 0,8 с в отпечаток не попадут и не предлагаются.
    build_pack=false — только список мест и секунды по участникам; build_pack=true — карточки «подтвердите
    говорящего» по этим местам (затем start_review_page → ответы человека → apply_answers). Сам ничего не
    подтверждает. После применения ответов — preview_voice_profiles и update_voice_profiles."""
    try:
        return json.dumps(core.legacy_speaker_review(job_id, build_pack), ensure_ascii=False, indent=1)
    except (core.CoreError, storage.StorageError) as ex:      # build_pack пишет запись
        return core.say_error(ex)


@mcp.tool()
def apply_review(job_id: str, decisions: list[dict]) -> str:
    """Применяет решения по сомнительным местам и пишет их в журнал правок (edits).

    decisions: [{"id": "r5", "speaker": "Анна"}, {"id": "r7", "text": "Вереск"},
                {"id": "r9"} (принять как есть), {"id": "r12", "status": "human", "note": "неразборчиво"}].
    speaker — переподписать место (реплика при необходимости делится); text — заменить текст места
    (слова модели сохраняются в поле model и в журнале); status "human" — оставить человеку.
    Если решение изменило слова, которые человек видел в своей правке отрезка, они названы в seen_changed ответа.
    Место «пропущена речь» (пауза) владеет только словами, начатыми внутри неё: соседнее слово ему не принадлежит.
    «Как есть», подпись говорящего и текст для пустой паузы отклоняются с причиной в conflicts — впишите услышанное
    ответом «Имя: слова» (apply_answers).
    После разбора .md содержит список «Проверить человеку» и журнал правок."""
    import transcript
    s = _status(job_id)
    if not s or s.get("state") != "done" or "json" not in s:
        return "Расшифровка не найдена или ещё не готова"
    _, report = transcript.apply_review(s["json"], s["markdown"], decisions)
    return json.dumps(report, ensure_ascii=False, indent=2)


@mcp.tool()
async def check_quotes(job_id: str, conspect_text: str = "", conspect_path: str = "") -> str:
    """Проверяет цитаты «…» конспекта по стенограмме задачи: дословны ли они и тот ли говорящий.

    conspect_text — текст конспекта (markdown или обычный), либо conspect_path — путь к .md/.txt на этом
    компьютере. Говорящий проверяется у цитат вида «Имя: «…»». Поправки и составные цитаты требуют проверки.
    Возвращает сводку и список цитат, которые не дословны или приписаны не тому."""
    import quotes
    import transcript
    s = _status(job_id)
    if not s or s.get("state") != "done" or "json" not in s:
        return "Расшифровка не найдена или ещё не готова"
    try:
        path = Path(conspect_path).expanduser().resolve() if conspect_path and not conspect_text else None
        if path:
            if path.suffix.lower() not in (".md", ".txt") or not path.is_file():
                return "Путь должен указывать на существующий .md или .txt файл"
            if path.stat().st_size > 2 * 1024 * 1024:
                return "Файл конспекта слишком велик (лимит 2 МиБ)"
    except OSError as ex:
        return f"Ошибка чтения конспекта: {ex}"
    if len(conspect_text) > 2 * 1024 * 1024:
        return "Текст конспекта слишком велик (лимит 2 МиБ)"

    def run():
        text = conspect_text or (path.read_text(encoding="utf-8-sig") if path else "")
        if not text.strip():
            return "Передайте conspect_text или conspect_path"
        res = quotes.check(text, transcript.load(s["json"]))
        for r in res["problems"]:
            r["t"] = transcript.hms(r["t"]) if r["t"] is not None else None   # у слов места нет времени
        return json.dumps(res, ensure_ascii=False, indent=1)

    if path and not path.is_relative_to(paths.DATA_DIR.resolve()):
        return await _gated("Чтение внешнего файла для проверки цитат",
                            {"path": str(path), "size": path.stat().st_size},
                            lambda: [f"Файл: {path}", f"Размер: {path.stat().st_size} байт",
                                     "Содержимое будет прочитано только для проверки цитат."],
                            run, require_human=True)
    try:
        return await asyncio.to_thread(run)
    except (OSError, ValueError) as ex:
        return f"Ошибка чтения конспекта: {ex}"


@mcp.tool()
def conspect_link(job_id: str, conspect_text: str, name: str = "конспект") -> str:
    """Связывает конспект со стенограммой (единый источник): каждая цитата «…» заменяется ссылкой на место
    стенограммы по ID слов; связываются только однозначные точные цитаты.

    conspect_text — конспект в markdown (как он пойдёт в документ). name — название (несколько конспектов
    к одной записи различаются названием). Создаёт в папке результатов <имя>.<name>.conspect.json и .md.
    Возвращает, сколько цитат связано, какие не найдены, и проверку цитат собранного конспекта.
    Составные, приблизительные и неоднозначные цитаты остаются несвязанными."""
    import conspect
    s = _status(job_id)
    if not s or s.get("state") != "done" or "json" not in s:
        return "Расшифровка не найдена или ещё не готова"
    return json.dumps(conspect.save_linked(s["json"], name, conspect_text), ensure_ascii=False, indent=1)


@mcp.tool()
def conspect_render(job_id: str, name: str = "конспект", include_markdown: bool = True) -> str:
    """Пересобирает связанный конспект из текущей стенограммы и возвращает markdown (для документа)
    и отчёт: какие цитаты обновились из стенограммы, у каких сменился говорящий, какие места пропали,
    плюс проверку цитат. Вызывается и сам — после apply_review и rename_speakers."""
    import conspect
    s = _status(job_id)
    if not s or s.get("state") != "done" or "json" not in s:
        return "Расшифровка не найдена или ещё не готова"
    rep = conspect.rerender(s["json"], name)
    if include_markdown:
        for k, r in rep.items():
            if isinstance(r, dict) and r.get("conspect_md"):
                r["markdown"] = Path(r["conspect_md"]).read_text(encoding="utf-8")
    return json.dumps(rep, ensure_ascii=False, indent=1)


@mcp.tool()
async def conspect_mark_reviewed(job_id: str, name: str = "конспект") -> str:
    """Подтвердить проверку смысла конспекта человеком на текущей ревизии источника.

    Требуется отдельное подтверждение человека; после изменения данных разрешение недействительно.
    Не повторять отказанный вызов без новой просьбы человека.
    """
    try:
        snapshot = core.describe_conspect_confirmation(job_id, name)
    except core.CoreError as ex:
        return f"Ошибка: {core.say_error(ex)}"
    return await _gated("Подтверждение проверки конспекта", snapshot,
                        lambda: [f"Конспект: {name}", f"Задача: {job_id}",
                                 f"Ревизия источника: {snapshot['source_revision']}",
                                 "Подтвердите только если сами проверили смысл и цитаты по текущей стенограмме."],
                         lambda: core.confirm_conspect_analysis(snapshot), require_human=True)


@mcp.tool()
def conspect_candidates(job_id: str, conspect_text: str) -> str:
    """Без записи показать неоднозначные цитаты и возможные места с ID и ревизией."""
    try:
        return json.dumps(core.preview_conspect_candidates(job_id, conspect_text), ensure_ascii=False, indent=1)
    except core.CoreError as ex:
        return f"Ошибка: {core.say_error(ex)}"


@mcp.tool()
async def conspect_link_selected(job_id: str, conspect_text: str, selections: dict,
                                 name: str = "конспект") -> str:
    """Создать новый конспект с выбранными человеком местами; нужно подтверждение человека.

    selections: {номер цитаты: {recording_id, source_revision, word_ids}} из conspect_candidates.
    Существующий конспект не перезаписывается; отказ без новой просьбы не повторять.
    """
    try:
        snapshot, report = core.prepare_selected_conspect(job_id, name, conspect_text, selections)
        lines = core.selected_conspect_confirmation_lines(job_id, name, snapshot, report, selections)
    except core.CoreError as ex:
        return f"Ошибка: {core.say_error(ex)}"
    return await _gated("Выбор мест цитат", snapshot, lambda: lines,
                        lambda: core.save_selected_conspect(snapshot, conspect_text, selections),
                        require_human=True)


@mcp.tool()
async def transcription_status(job_id: str, wait_sec: int = 50) -> str:
    """Статус задачи (расшифровки или теста): этап, процент, по готовности — пути к результатам.

    wait_sec (0–50, по умолчанию 50): если задача идёт, ответ ждёт до этого времени и приходит раньше,
    как только сменится этап или задача закончится. Поэтому вызывать подряд не нужно — один вызов
    заменяет частые проверки; для долгих этапов (скачивание модели) просто повторяйте вызов.
    В ответе для идущей задачи: elapsed_sec — с начала задачи, stage_elapsed_sec — с начала этапа,
    since_update_sec — давно ли обновлялся статус, done — счётчик внутри этапа, download — скачивание
    модели (МБ, скорость МБ/с)."""
    import asyncio
    s = core.job_status(job_id)
    if not s:
        return f"Задача {job_id} не найдена"
    k0 = core.stage_key(s)
    deadline = time.time() + min(max(int(wait_sec or 0), 0), 50)
    while s.get("state") in core.ACTIVE and time.time() < deadline:
        await asyncio.sleep(1)
        s2 = core.job_status(job_id)
        if s2 and s2.get("stage") != "обновляется":
            s = s2
        if core.stage_key(s) != k0:
            break
    return json.dumps(core.live(s, job_id), ensure_ascii=False, indent=2)


@mcp.tool()
def read_transcript(job_id: str, fmt: str = "md", offset: int = 0, max_chars: int = 60000,
                    extra: bool = False) -> str:
    """Текст готовой расшифровки. fmt: 'md' (читаемый) или 'json' (с таймкодами и вероятностями эмоций).
    extra=False (по умолчанию) — только реплики: без разделов «Сомнительные места», «Проверить человеку»,
    «Автоматические исправления» и журнала правок (это до половины объёма); в json — без слов, мест разбора
    и журнала. Слова нерешённых мест в репликах взяты в ⟦…⟧: распознаны или подписаны неуверенно — в цитаты
    конспекта без проверки не брать; подробно о месте — review_items; в json у реплики "doubts": [id мест].
    Полный текст — extra=True.
    Чтобы передать стенограмму целиком (файлом человеку, в артефакт, в проект — только по просьбе), не
    читайте её сюда: путь к файлу есть в transcription_status, device_stage_files копирует файл в рабочую
    папку сессии, дальше файл передаётся по пути. Читать нужно только то, что анализируется (конспект).
    Для отчёта теста скорости fmt игнорируется. Длинные тексты читайте частями через offset."""
    try:
        return core.read_transcript(job_id, fmt, offset, max_chars, extra)
    except core.CoreError as ex:
        return str(ex)


@mcp.tool()
def get_settings() -> str:
    """Текущие настройки: папка данных (data_dir), отличия пользователя (overrides — файл config_file) и
    действующие значения (effective) с учётом умолчаний программы."""
    return json.dumps(core.settings_info(), ensure_ascii=False, indent=2)


@mcp.tool()
def initialize_recording_store(display_name: str = "") -> str:
    """Явно создать новую пустую SQLite-БД и локальную identity; существующие файлы не мигрируются."""
    import recordings
    try:
        store = recordings.initialize_store(display_name=display_name)
        try:
            identity = store.current_identity()
        finally:
            store.close()
        return json.dumps({"initialized": True, "database": str(paths.STATE_DB),
                           "identity": identity, "recording_backend": settings.recording_backend()},
                          ensure_ascii=False, indent=2)
    except (OSError, ValueError, storage.StorageError) as ex:
        return json.dumps({"initialized": False, "error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def find_recording_results(source_path: str) -> str:
    """Найти SQLite-результаты исходного файла независимо от истории фоновых задач."""
    try:
        return json.dumps(core.find_recording_results(source_path), ensure_ascii=False)
    except (OSError, ValueError, storage.StorageError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def read_recording_result(recording_id: str) -> str:
    """Прочитать SQLite-результат по recording_id без сохранённого job."""
    try:
        result = core.read_recording_result(recording_id)
        quality = result.get("quality")
        if isinstance(quality, dict) and "interaction" in quality:
            # слой признаков разговора (массивы чисел) показывается только по запросу
            quality["interaction"] = {"omitted": "см. conversation_hotspots"}
        return json.dumps(result, ensure_ascii=False)
    except (OSError, ValueError, KeyError, storage.StorageError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def sync_recording_documents(recording_id: str) -> str:
    """Повторить формирование файлов SQLite-записи, не затирая внешние изменения."""
    try:
        return json.dumps(core.sync_recording_documents(recording_id), ensure_ascii=False)
    except (OSError, ValueError, KeyError, storage.StorageError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def list_conspect_conflicts(recording_id: str) -> str:
    """Показать неразрешённые внешние правки конспектов SQLite-записи."""
    try:
        return json.dumps(core.list_conspect_conflicts(recording_id), ensure_ascii=False)
    except (OSError, ValueError, storage.StorageError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
async def request_review_answer_verification(recording_id: str, answer_ids: list[str]) -> str:
    """Явно подтвердить выбранные ответы неизвестного происхождения перед применением."""
    try:
        snapshot = core.prepare_answer_confirmation(recording_id, answer_ids)
        return await _gated(
            "Подтвердить ответы разбора",
            {"recording_id": recording_id, "answer_ids": answer_ids, "state_hash": snapshot["state_hash"]},
            lambda: [f"Запись: {recording_id}",
                     "Подтвердить следующие ответы как проверенные человеком:"] + snapshot["lines"],
            lambda: core.confirm_review_answers(snapshot), require_human=True)
    except (OSError, ValueError, storage.StorageError, core.CoreError) as ex:
        return f"Ошибка: {core.say_error(ex)}"


@mcp.tool()
def list_document_conflicts(recording_id: str) -> str:
    """Показать неразрешённые внешние правки документов SQLite-записи."""
    try:
        return json.dumps(core.list_document_conflicts(recording_id), ensure_ascii=False)
    except (OSError, ValueError, storage.StorageError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


def _replace_window(what, plan):
    """Действие и строки окна замены: обещать только то, что произойдёт после «Да»."""
    effect = plan["effect"]
    if effect == "close":
        return (f"Закрыть конфликт: файл {what} уже равен версии программы",
                ["Файл на диске не изменится; решение нельзя отменить.",
                 "Прежняя копия правки остаётся в базе программы."])
    if effect == "recopy":
        return (f"Отметить прежнюю копию решённой: изменённый файл {what} не будет заменён",
                ["Файл изменили после того, как копия была сохранена. Программа покажет новый конфликт; "
                 "новую копию она сохранит, если файл не больше 8 МиБ.",
                 "Прежняя копия остаётся в базе программы; решение по ней нельзя отменить."])
    kept = ["Копия правки остаётся в базе программы; в стенограмму правка не переносится.",
            "Строки различий это окно не показывает; их показывает локальный экран.",
            "Пока идёт замена, не сохраняйте этот файл в других программах: правка может пропасть без копии."]
    if effect == "replace":
        return (f"Заменить файл {what} версией программы",
                ["Заменить файл на диске версией программы: правка в нём пропадёт.", *kept])
    names = ", ".join(f"«{name}»" for name in (plan["waits"] if effect == "wait" else plan["stops"]))
    cause = (f"Он ждёт решения по другому файлу: {names}. " if effect == "wait"
             else f"Запись остановит повреждённая копия файла {names}. ")
    if plan["blocked"]:
        cause += "Тот файл здесь и на экране решить нельзя: нужен ручной разбор. "
    later = ("этот изменённый файл не заменит, а покажет новый конфликт." if plan["changed"]
             else "заменит этот сама, если он не менялся, и правка в нём пропадёт.")
    return (f"Записать решение о замене файла {what}: сейчас файл не изменится",
            [cause + "Решение нельзя отменить. Если тот файл будет решён, программа " + later, *kept])


@mcp.tool()
async def resolve_document_conflict(recording_id: str, conflict_id: str) -> str:
    """После подтверждения человеком заменить файл документа версией программы. Правка человека пропадает
    из файла; её копия остаётся в базе и в стенограмму не переносится. Без файла на месте — отказ до решения."""
    try:
        row = next((item for item in core.list_document_conflicts(recording_id)
                    if item["conflict_id"] == conflict_id), None)
        if row is None:
            return "Конфликт не найден или уже разрешён"
        if row["kind"].startswith("archive_"):
            return "Изменён неизменяемый архив; требуется ручной разбор"
        if not row["preserved"]:
            return "Внешний файл слишком велик для сохранения в БД; требуется ручной разбор"
        action, effect = _replace_window(
            "документа", core.replacement_effect(recording_id, conflict_id, "document"))
        return await _gated(
            action,
            {"recording_id": recording_id, "conflict_id": conflict_id, "hash": row["observed_hash"]},
            lambda: [*effect,
                     f"Запись: {recording_id}; файл: {row['relative_path']}",
                     f"Размер копии: {row['observed_size']} байт; SHA-256: {row['observed_hash']}"],
            lambda: core.resolve_document_conflict(recording_id, conflict_id), require_human=True)
    except (OSError, ValueError, storage.StorageError, core.CoreError) as ex:
        return f"Ошибка: {core.say_error(ex)}"


@mcp.tool()
async def resolve_conspect_conflict(recording_id: str, conflict_id: str) -> str:
    """После подтверждения человеком заменить файл конспекта версией программы. Правка человека пропадает
    из файла; её копия остаётся в базе и в стенограмму не переносится. Без файла на месте — отказ до решения."""
    try:
        conflicts = core.list_conspect_conflicts(recording_id)
        row = next((item for item in conflicts if item["conflict_id"] == conflict_id), None)
        if row is None:
            return "Конфликт не найден или уже разрешён"
        if not row["preserved"]:
            return "Внешний файл слишком велик для сохранения в БД; требуется ручной разбор"
        action, effect = _replace_window(
            "конспекта", core.replacement_effect(recording_id, conflict_id, "conspect"))
        return await _gated(
            action,
            {"recording_id": recording_id, "conflict_id": conflict_id, "hash": row["observed_hash"]},
            lambda: [*effect,
                     f"Запись: {recording_id}; конспект: {row['name']}",
                     f"Файл: {row['relative_path']}",
                     f"Размер копии: {row['observed_size']} байт; SHA-256: {row['observed_hash']}"],
            lambda: core.resolve_conspect_conflict(recording_id, conflict_id), require_human=True)
    except (OSError, ValueError, storage.StorageError, core.CoreError) as ex:
        return f"Ошибка: {core.say_error(ex)}"


@mcp.tool()
def set_recording_display_name(display_name: str) -> str:
    """Изменить локальное имя в SQLite; оно не записывается в config.json."""
    import recordings
    try:
        store = recordings.open_store()
        try:
            name = settings.set_user_display_name(store, display_name)
        finally:
            store.close()
        return json.dumps({"display_name": name}, ensure_ascii=False)
    except (OSError, ValueError, storage.StorageError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
def set_recording_backend(backend: str) -> str:
    """Выбрать legacy_files или подготовленный SQLite backend; старые записи не перемещаются."""
    try:
        settings.set_recording_backend(backend)
        return json.dumps({"recording_backend": settings.recording_backend()}, ensure_ascii=False)
    except (ValueError, RuntimeError, OSError, storage.StorageError) as ex:
        return json.dumps({"error": core.say_error(ex)}, ensure_ascii=False)


@mcp.tool()
async def self_check(tests: bool = True) -> str:
    """Самопроверка программы на этом компьютере — после обновления или при подозрении на сбой: тесты (в отдельном
    процессе, с отдельной временной папкой данных — настоящие данные не трогаются), проверка файлов программы на
    личные данные (tools/check_repo.py: только где нарушение, сами слова не показываются) и морфология
    (pymorphy3 для подсказок словаря). Только проверки, ничего не меняет. tests=False — без тестов (быстро).
    Тесты идут несколько минут, предел — 12 мин. Итог: ok, summary и подробности по каждой части."""
    import selfcheck
    r = await asyncio.to_thread(selfcheck.run, VERSION, bool(tests), True, True)
    return json.dumps(r, ensure_ascii=False, indent=1)


@mcp.tool()
async def migrate_data(dry_run: bool = True) -> str:
    """Перенос данных в папку данных: из папки программы (profiles, config.json, jobs, bench_*) и из прежних мест
    по умолчанию. dry_run=true — только показать, что будет перенесено; dry_run=false — после подтверждения
    человеком в окне на экране. Возвращает итог, прежние папки и хвост
    migration.log. Резервные копии не удаляются."""
    if dry_run:
        return json.dumps(core.do_migrate(True), ensure_ascii=False, indent=1)
    return await _gated("Перенос данных", {"dry_run": False}, core.describe_migrate,
                        lambda: core.do_migrate(False), require_human=True)


@mcp.tool()
async def export_group(group: str = "", dest_dir: str = "", with_jobs: bool = False) -> str:
    """Выгружает группу (словарь + голосовой профиль) в zip для переноса на другой компьютер или резервной копии.
    group — имя группы (пусто — все); dest_dir — куда положить архив (пусто — exports в папке данных);
    with_jobs — добавить статусы задач. В архиве абсолютные пути заменены именами файлов, карантин голосов не
    переносится, есть manifest.json с контрольными суммами. Архив содержит голосовые отпечатки (биометрия):
    в разговор он не передаётся, возвращаются только путь и сводка. Делайте только по просьбе человека: на экране
    появится окно, где человек сам подтверждает выгрузку (подтвердить за него нельзя)."""
    return await _gated("Выгрузка группы", {"group": group, "dest": dest_dir, "jobs": with_jobs},
                        lambda: core.describe_export(group, dest_dir, with_jobs),
                        lambda: core.do_export(group, dest_dir, with_jobs), require_human=True)


@mcp.tool()
async def import_group(zip_path: str, mode: str = "merge", apply: bool = False) -> str:
    """Загружает архив из export_group. mode: "merge" — добавить недостающих участников и словарь, при споре
    остаётся профиль с большим числом секунд речи (отчёт: conflicts); "replace" — заменить группу (прежняя уходит
    в резервную копию _backup-import-<дата>). apply=false (по умолчанию) — только отчёт, ничего не меняется;
    apply=true — выполнить: на экране появится окно с описанием изменений, и выполнится только если человек
    нажмёт «Да». Проверяются версия схемы, имена файлов и контрольные суммы. Сначала покажите человеку отчёт."""
    if not apply:
        return await _gated(None, None, None, lambda: core.do_import(zip_path, mode, False))
    return await _gated("Загрузка архива", {"zip": zip_path, "mode": mode},
                        lambda: core.describe_import(zip_path, mode),
                        lambda: core.do_import(zip_path, mode, True), require_human=True)


@mcp.tool()
async def delete_group(group: str, confirm: str, permanent: bool = False) -> str:
    """Удаляет группу (словарь и голосовой профиль). confirm должен точно совпасть с именем группы. По умолчанию
    папка переносится в _deleted (можно вернуть); permanent=true удаляет навсегда. Вызывать только по прямой
    просьбе человека, назвавшего группу; permanent=true — только если он прямо сказал «навсегда». Человек
    дополнительно подтверждает удаление в окне на экране; без его «Да» ничего не удаляется."""
    def lines():
        canonical = paths.safe_group(group)
        if confirm != canonical:
            raise core.CoreError(f"Подтверждение не совпало: для удаления передайте confirm=«{canonical}»")
        return core.describe_delete(group, permanent)
    return await _gated("Удаление группы", {"group": group, "permanent": permanent}, lines,
                        lambda: core.do_delete_group(group, confirm, permanent), require_human=True)


@mcp.tool()
async def cleanup_jobs(days: int = 30, apply: bool = False) -> str:
    """Чистит статусы, логи и pid задач старше days дней в папке данных (идущие задачи не трогает).
    apply=false — только подсчёт; apply=true — после подтверждения человеком в окне на экране."""
    if not apply:
        return await _gated(None, None, None, lambda: core.do_cleanup(days, False))
    return await _gated("Очистка журналов задач", {"days": days}, lambda: core.describe_cleanup(days),
                        lambda: core.do_cleanup(days, True))


@mcp.tool()
async def cleanup_hypotheses(unit_id: str = "", older_than_days: float = -1, apply: bool = False) -> str:
    """Удаляет сохранённые гипотезы моделей (настройка save_hypotheses; в них полный текст записей):
    всех, одной записи (unit_id) или старше older_than_days дней (-1 — без ограничения по возрасту).
    apply=false — только подсчёт; apply=true — после подтверждения человеком в окне на экране.
    Выключение save_hypotheses уже сохранённые файлы не удаляет."""
    days = None if older_than_days < 0 else older_than_days
    if not apply:
        return await _gated(None, None, None, lambda: core.do_cleanup_hypotheses(unit_id, days, False))
    return await _gated("Удаление гипотез моделей", {"unit_id": unit_id, "older_than_days": days},
                        lambda: core.describe_cleanup_hypotheses(unit_id, days),
                        lambda: core.do_cleanup_hypotheses(unit_id, days, True))


def _local_voice_tool():
    """Дополнительные голосовые инструменты: включить review_local и перезапустить MCP-клиент.
    Локальная веб-страница доступна независимо от этой настройки."""
    on = False
    try:
        on = bool(settings.load_config().get("review_local"))
    except Exception:
        pass
    return mcp.tool() if on else (lambda f: f)


# ---------- пакет разбора: локальный и облачный режимы ----------
@mcp.tool()
def build_review_pack(job_id: str, items: list[dict] | None = None) -> str:
    """Собирает пакет разбора сомнительных мест: клипы (3,5 с до места, 1,5 с после) (mp3 моно 64 кбит/с).
    Manifest хранится в файле для legacy или в SQLite для SQLite-записи. Целиком запись никуда не уходит.

    items — вопросы клиента по местам из review_items; без items — нейтральные вопросы:
      [{"id": "r5", "question": "Кто говорит «…»?", "options": [
          {"label": "Анна", "decision": {"speaker": "Анна"}},
          {"label": "Вера",   "decision": {"speaker": "Вера"}},
          {"label": "как есть", "decision": {}}], "recommended": 0}]
    decision — то, что получит apply_review при выборе варианта (speaker / text / status, note).
    2–4 варианта. Пересборка меняет version: ответы на прежние вопросы не применятся."""
    try:
        return json.dumps(core.build_review_pack(job_id, items), ensure_ascii=False, indent=1)
    except (core.CoreError, storage.StorageError) as ex:      # состояние пакета пишется в запись
        return core.say_error(ex)


@mcp.tool()
def review_pack_info(job_id: str, include_manifest: bool = False) -> str:
    """Состояние пакета разбора: папка, версия, места, сколько ответов (по источникам), расхождения,
    файлы клипов (для публикации в облако). include_manifest — вернуть manifest целиком."""
    import review_pack as P
    jp, _ = _job_json(job_id)
    if not jp:
        return "Расшифровка не найдена или ещё не готова"
    man = P.manifest(jp)
    if not man:
        return "Пакет разбора не собран: build_review_pack"
    pd = P.pack_dir(jp)
    ans = P.answers_state(jp, {"answers": {}})
    sqlite_mode = "_expected_versions" in ans
    src = {}
    for a in ans.get("answers", {}).values():
        src[a.get("source")] = src.get(a.get("source"), 0) + 1
    out = {"pack": str(pd), "version": man["version"], "items": len(man["items"]),
           "answered": len(ans.get("answers", {})), "by_source": src,
           "conflicts": len(ans.get("conflicts") or []), "rejected": len(ans.get("rejected") or []),
           "manifest_storage": "sqlite" if sqlite_mode else "file",
           "files": ["cards.html"] + ([] if sqlite_mode else ["manifest.json"]) +
                    [it["clip"] for it in man["items"]]}
    if include_manifest:
        out["manifest"] = man
    return json.dumps(out, ensure_ascii=False, indent=1)


@mcp.tool()
def start_review_page(job_id: str, port: int = 8765) -> str:
    """Открывает локальную страницу карточек разбора (http://127.0.0.1:<порт>/, только этот компьютер):
    клипы, варианты касанием, озвучка вопроса, голосовой ответ с микрофона (распознаётся GigaAM).
    Ответы — в answers.json пакета (source=local). Возвращает job_id; адрес — в transcription_status."""
    try:
        return json.dumps(core.start_review_page(job_id, port), ensure_ascii=False)
    except core.CoreError as ex:
        return str(ex)


@_local_voice_tool()
def start_voice_review(job_id: str, seconds: float = 5.0, ids: str = "") -> str:
    """Голосовой разбор на ноутбуке без экрана: озвучка вопроса и вариантов (голос Windows, Irina),
    клип, запись ответа после сигнала (seconds), распознавание GigaAM и сопоставление с вариантами.
    Понимает «первый/второй…», имя из варианта, «да» (рекомендация), «повтори», «пропусти»,
    «оставь человеку»; остальное — свой ответ. ids — только эти места (через запятую), иначе все без
    ответа. Возвращает job_id; ход — transcription_status."""
    jp, _ = _job_json(job_id)
    if not jp:
        return "Расшифровка не найдена или ещё не готова"
    args = ["--seconds", str(float(seconds))] + (["--ids", ids] if ids else [])
    return json.dumps({"job_id": _launch("voice_review.py", args, jp, "voice")}, ensure_ascii=False)


@mcp.tool()
def review_pack_cloud(job_id: str, artifact_url: str = "") -> str:
    """Готовит карточки пакета к публикации в облако: <пакет>/cloud/index.html и manifest.json (с последними
    ответами, отметкой «применено»), плюс клипы. Возвращает base — папку пакета на этом компьютере и files —
    {путь на странице: путь относительно base} (файл здесь = base + путь) для публикации артефактом (ИИ-агент
    забирает их и публикует), remove — клипы, которые надо
    убрать со страницы, и artifact_url страницы, если она уже публиковалась (тогда обновлять её же).
    artifact_url — запомнить адрес опубликованной страницы для этого пакета."""
    import review_pack as P
    jp, _ = _job_json(job_id)
    if not jp:
        return "Расшифровка не найдена или ещё не готова"
    return json.dumps(P.cloud_bundle(jp, artifact_url or None), ensure_ascii=False, indent=1)


@mcp.tool()
def preview_voice_profiles(job_id: str, group_override: str = "") -> str:
    """Показать кандидатов по разметке. group_override — только если группа расшифровки неизвестна."""
    try:
        return json.dumps(core.preview_voice_learning(job_id, group_override or None),
                          ensure_ascii=False, indent=1)
    except core.CoreError as ex:
        return f"Ошибка: {core.say_error(ex)}"


@mcp.tool()
async def update_voice_profiles(job_id: str, group_override: str = "") -> str:
    """Пересчитывает голосовые профили по этой расшифровке после ответов человека (ответ человека главнее
    отпечатка): отрезки, подписанные человеком, идут в профиль с весом ×3 (кроме наложений и отрезков
    короче 0,8 с), из подтверждённых расшифровок — только слова без пометок сомнения. Замена существующих
    вкладов требует отдельного подтверждения после предварительного просмотра; создаётся резерв. Поправки «A → B» копятся как
    спутанные пары: на них сервер чаще ставит метку сомнения и склоняется к подтверждённому. Если профиль
    участника сильно изменился (сходство со старым < 0,8) — предупреждение. Защита от тёзок: профиль —
    той группы, с которой сделана расшифровка; вклад, непохожий на собранный голос имени, идёт в карантин
    (в ответе quarantine) до решения человека — resolve_voice. group_override допустим только при неизвестной
    группе; явный выбор и словарь будут показаны до штатного подтверждения. Возвращает job_id."""
    try:
        snapshot = await asyncio.to_thread(core.prepare_voice_action, job_id, group_override or None)
    except core.CoreError as ex:
        return f"Ошибка: {core.say_error(ex)}"
    return await _gated("Обновление голосового профиля", snapshot,
                        lambda: core.voice_action_lines(snapshot), lambda: core.start_voice_update(snapshot),
                        require_human=True)


@mcp.tool()
async def revoke_voice_profiles(job_id: str) -> str:
    """Отозвать вклады одного источника после просмотра и подтверждения человека, с резервной копией."""
    try:
        snapshot = await asyncio.to_thread(core.prepare_voice_action, job_id)
        if not snapshot["report"]["existing_contributions"]:
            return "Вкладов этой записи в профиле нет."
    except core.CoreError as ex:
        return f"Ошибка: {core.say_error(ex)}"
    return await _gated("Отзыв голосового вклада", snapshot,
                        lambda: core.voice_action_lines(snapshot, revoke=True),
                        lambda: core.revoke_voice_source(snapshot), require_human=True)


@mcp.tool()
async def set_voice_auto_learn(enabled: bool, group: str = "") -> str:
    """Настроить автопополнение группы; без group — умолчание для групп при первом использовании.
    Изменение требует отдельного подтверждения человеком."""
    return await _gated("Настройка автопополнения голосов", {"group": group, "enabled": enabled},
                        lambda: [f"Группа: {group or 'умолчание новых групп'}", f"Автопополнение: {enabled}"],
                        lambda: core.set_voice_auto_learn(enabled, group), require_human=True)


@mcp.tool()
def voice_quality(group: str = "") -> str:
    """Политика качества аудио и сохранённые измерения по группам (без декодирования аудио).
    warn — только предупреждать; reject_clipping — не учитывать источник с клиппингом. Тихая речь
    и оценка SNR сами по себе не отбраковываются. Измерения приблизительны."""
    try:
        return json.dumps(core.voice_quality(group), ensure_ascii=False, indent=1)
    except core.CoreError as ex:
        return f"Ошибка: {core.say_error(ex)}"


@mcp.tool()
async def set_voice_quality_policy(policy: str, group: str = "") -> str:
    """Выбрать политику качества: warn (по умолчанию) или reject_clipping. Без group — умолчание
    для групп без явной политики. Изменение требует отдельного подтверждения человеком."""
    if policy not in ("warn", "reject_clipping"):
        return 'policy: "warn" или "reject_clipping"'
    return await _gated("Политика качества голосов", {"group": group, "policy": policy},
                        lambda: [f"Группа: {group or 'умолчание новых групп'}", f"Политика качества: {policy}",
                                 "reject_clipping отбраковывает только источник с клиппингом; "
                                 "тихая речь не отбраковывается."],
                        lambda: core.set_voice_quality_policy(policy, group), require_human=True)


@mcp.tool()
def voice_refresh_plan(group: str = "") -> str:
    """Показать, у каких источников изменились стенограмма/аудио или остался старый отбор.
    Только чтение: профиль не меняется. Обновление — отдельно через preview_voice_profiles
    и update_voice_profiles для задачи из поля job_id."""
    try:
        return json.dumps(core.voice_refresh_plan(group), ensure_ascii=False, indent=1)
    except core.CoreError as ex:
        return f"Ошибка: {core.say_error(ex)}"


@mcp.tool()
async def backup_voice_profiles(group: str = "") -> str:
    """Сохранить резервные копии профилей одной группы или всех групп в voice-backups.
    Это биометрия в открытом виде; копии не входят в экспорт и автоматически не удаляются."""
    try:
        return await _gated("Резервное копирование голосовых профилей", {"group": group},
                            lambda: [f"Группа: {group or 'все группы'}",
                                     "Резерв — биометрия в открытом виде.",
                                     "Старые резервы не заменяются и не удаляются."],
                            lambda: core.backup_voice_profiles(group), require_human=True)
    except core.CoreError as ex:
        return f"Ошибка: {core.say_error(ex)}"


@mcp.tool()
def resolve_voice(name: str, recording: str, decision: str, profile: str = "") -> str:
    """Решение человека по тёзке из карантина профиля (см. voice_profiles → quarantine): голос с именем
    name в записи recording (имя файла расшифровки или записи, можно часть) не похож на собранный профиль.
    decision: "same" — это тот же человек: решение сохраняется; обновление профиля запускается отдельно;
    "different" — другой человек с тем же именем: в этот профиль запись по этому имени больше не идёт.
    profile — группа (имя из voice_profiles), если групп несколько. Возвращает итог решения."""
    if decision not in ("same", "different"):
        return 'decision: "same" или "different"'
    if confirm.mode() == "off":
        return "Ошибка: для решения по голосовому профилю требуется подтверждение человека"
    g = profile.replace(".voices.json", "").strip() if profile else ""
    files = [paths.voices_path(g)] if g else sorted(paths.GROUPS.glob("*/voices.json"))
    hits = []
    for pf in files:
        if not pf.exists():
            continue
        d = json.loads(pf.read_text(encoding="utf-8"))
        for akey, qa in (d.get("quarantine") or {}).items():
            q = qa.get(name)
            if q and (recording.lower() in akey.lower() or recording.lower() in q.get("from", "").lower()):
                hits.append((pf, d, akey, q))
    if not hits:
        return f"В карантине нет «{name}» для «{recording}» (voice_profiles покажет, что там есть)"
    if len(hits) > 1:
        return json.dumps({"ambiguous": [{"group": h[0].parent.name, "recording": h[2], "from": h[3].get("from")}
                                         for h in hits]}, ensure_ascii=False)
    pf, d, akey, q = hits[0]
    try:
        confirm.require("Решение по голосовому профилю", {"name": name, "recording": akey, "decision": decision},
                        [f"Группа: {pf.parent.name}", f"Участник: {name}", f"Решение: {decision}"])
    except confirm.NotConfirmed as ex:
        return str(ex)
    import settings
    with settings.group_voice_lock(pf.parent.name):
        d = json.loads(pf.read_text(encoding="utf-8"))
        current = (d.get("quarantine") or {}).get(akey, {}).get(name)
        if current != q:
            return "Карантин изменился; повторите просмотр и подтверждение."
        d.setdefault("identity", {}).setdefault(akey, {})[name] = decision
        d["quarantine"][akey].pop(name, None)
        if not d["quarantine"][akey]:
            d["quarantine"].pop(akey)
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=pf.parent, delete=False) as f:
            json.dump(d, f, ensure_ascii=False)
            tmp = f.name
        os.replace(tmp, pf)
    if decision == "different":
        remaining = sorted(n for n, s in (d.get("speakers") or {}).items()
                           if akey in (s.get("parts") or {}))
        return json.dumps({"group": pf.parent.name, "name": name, "recording": akey,
                           "done": "в этот профиль запись по этому имени больше не идёт",
                           "remaining_contributions": remaining,
                           "revoke_removes_all_source_contributions": bool(remaining),
                           "warning": ("Прежний вклад этой записи остался. Штатный "
                                       "revoke_voice_profiles удалит вклады этой записи у всех "
                                       "участников, не только у выбранного имени; сначала проверьте "
                                       "preview_voice_profiles." if remaining else
                                       "Вкладов этой записи в профиле нет.")},
                          ensure_ascii=False)
    return json.dumps({"group": pf.parent.name, "name": name, "recording": akey,
                       "done": "личность подтверждена; профиль пока не изменён",
                       "hint": "Для пересчёта вызовите update_voice_profiles для задачи этой расшифровки и подтвердите план."},
                      ensure_ascii=False)


@mcp.tool()
def import_answers(job_id: str, answers: list[dict], source: str = "cloud") -> str:
    """Сводит ответы в answers.json пакета. answers: [{"id", "version", "option" (с 0) | "kind"
    (human/skip/recommended/free) | "text", "audio" (путь к голосовому ответу на этом компьютере), "ts"}].
    Голосовые ответы распознаются GigaAM и сопоставляются с вариантами. Ответ на прежнюю version не
    принимается; на одно место из двух режимов побеждает более поздний, расхождение — в conflicts."""
    import review_pack as P
    if any(isinstance(answer, dict) and answer.get("audio") for answer in answers):
        return "Импорт ответов не принимает пути к аудиофайлам; используйте локальный голосовой разбор"
    jp, _ = _job_json(job_id)
    if not jp:
        return "Расшифровка не найдена или ещё не готова"
    return json.dumps(P.add_answers(jp, answers, source), ensure_ascii=False, indent=1)


@mcp.tool()
def apply_answers(job_id: str, drop_audio: bool = False) -> str:
    """Применяет ответы пакета: выбранный вариант → его decision (apply_review, by=human); «оставить
    человеку» → в «Проверить человеку»; свой ответ текстом — правка того, что человек услышал в клипе:
    разбор по репликам «Имя: …», выравнивание со словами окна клипа, смена говорящего, замены, вставки
    пропущенной речи, опечатки по вариантам моделей; места соседних карточек, покрытые ответом, закрываются.
    Ответ на паузу: слова, уже стоящие рядом с ней в стенограмме, узнаются и не вставляются второй раз; если ответ
    совпал только с ними, пауза остаётся человеку с причиной (not_applied, detail). Если ответ совпал со словами,
    уже стоящими в открытой паузе, место закрывается как «как есть» человеком и эти слова подтверждаются; если
    место требует отдельного подтверждения прослушивания, оно остаётся человеку (not_applied, detail).
    Не понятые ответы — человеку. Связанные конспекты пересобираются.
    drop_audio — после применения удалить клипы и голосовые ответы пакета на этом компьютере."""
    import review_pack as P
    import transcript
    jp, md = _job_json(job_id)
    if not jp:
        return "Расшифровка не найдена или ещё не готова"
    rep = core.apply_answers(job_id)
    if drop_audio:
        rep["audio_removed"] = P.drop_audio(jp)
    return json.dumps(rep, ensure_ascii=False, indent=1)


def _speak(text):
    """Озвучивает текст голосом Windows (Irina или другой русский) на этом компьютере."""
    import voice_io
    voice_io.speak(text)
    return json.dumps({"spoken": len(text), "voice": voice_io.voice()[1]}, ensure_ascii=False)


def _play_clip(job_id, item_id):
    """Проигрывает на этом компьютере клип места из пакета разбора."""
    import review_pack as P
    import voice_io
    jp, _ = _job_json(job_id)
    man = P.manifest(jp) if jp else None
    it = next((x for x in (man or {}).get("items", []) if x["id"] == item_id), None)
    if not it:
        return "Нет такого места в пакете"
    voice_io.play(P.pack_dir(jp) / it["clip"])
    return json.dumps({"played": it["clip"]}, ensure_ascii=False)


def _listen_reply(seconds, job_id, item_id):
    """Сигнал, запись ответа с микрофона (seconds), распознавание GigaAM. С job_id и item_id — ещё и
    сопоставление с вариантами места и запись ответа в пакет (source=local)."""
    import review_pack as P
    import voice_io
    jp, _ = _job_json(job_id) if job_id else (None, None)
    folder = (P.pack_dir(jp) / "voice") if jp else (JOBS / "voice")
    safe_item_id = "reply"
    if jp and item_id:
        man = P.manifest(jp)
        it = next((row for row in (man or {}).get("items", []) if row.get("id") == item_id), None)
        if it is None:
            return "Нет такого места в пакете"
        safe_item_id = str(it["id"])
    wav = folder / f"{safe_item_id}-listen-{int(time.time())}.wav"
    if not wav.resolve().is_relative_to(folder.resolve()):
        return "Недопустимый путь для аудиоответа"
    voice_io.beep()
    voice_io.record(wav, min(max(float(seconds), 1.0), 30.0))
    text = P.recognize_files([wav]).get(str(wav), "")
    out = {"heard": text, "audio": str(wav)}
    if jp and item_id:
        man = P.manifest(jp)
        out["answer"] = P.add_answers(jp, [{"id": item_id, "version": man["version"], "text": text,
                                            "audio": str(wav), "heard": True}], "local", trusted_actor="human")
    return json.dumps(out, ensure_ascii=False, indent=1)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as ex:
        import traceback
        return json.dumps({"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-800:]},
                          ensure_ascii=False)


@_local_voice_tool()
def speak(text: str) -> str:
    """Озвучивает текст голосом Windows (Irina или другой русский) на этом компьютере."""
    return _safe(_speak, text)


@_local_voice_tool()
def play_clip(job_id: str, item_id: str) -> str:
    """Проигрывает на этом компьютере клип места из пакета разбора."""
    return _safe(_play_clip, job_id, item_id)


@_local_voice_tool()
def listen_reply(seconds: float = 5.0, job_id: str = "", item_id: str = "") -> str:
    """Сигнал, запись ответа с микрофона (seconds), распознавание GigaAM. С job_id и item_id — ещё и
    сопоставление с вариантами места и запись ответа в пакет (source=local)."""
    return _safe(_listen_reply, seconds, job_id, item_id)


@mcp.tool()
def list_jobs(limit: int = 10) -> str:
    """Последние задачи и их состояние."""
    return json.dumps(core.list_jobs(limit), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
