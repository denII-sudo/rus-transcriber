"""MCP-сервер для Claude Desktop и других MCP-клиентов: тонкий слой над core.py.

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
import migrate  # noqa: E402
import confirm  # noqa: E402

JOBS = core.JOBS
_status = core.job_status        # короткие имена для инструментов разбора и конспекта ниже
_launch = core.launch
_job_json = core.job_json
try:
    migrate.run()          # один раз: данные из папки программы -> DATA_DIR (с резервной копией)
except Exception as _ex:   # старт сервера из-за этого не рвём
    print(f"миграция данных не выполнена: {_ex}", file=sys.stderr)

INSTRUCTIONS = """\
rus-transcriber: расшифровка русской речи (в том числе детских разговоров) на этом компьютере. Модели считаются
локально; в разговор попадает только то, что вы прочитаете инструментами. Рабочий процесс:

1. Запуск: start_transcription(audio_path, names="Анна, Борис", group="…"). names — участники записи, если известны
   (голоса ищутся только среди них); num_speakers — только если число голосов известно точно; group нужен, когда в
   папке данных несколько групп (voice_profiles покажет их). emotions=false ускоряет, если эмоции не нужны.
2. Ожидание: transcription_status(job_id, wait_sec=50) — вызов сам ждёт и отвечает раньше при смене этапа; вызывайте
   подряд без пауз и без других опросов. Отменить — cancel_job. Одновременно идут не больше двух тяжёлых задач.
3. Чтение: read_transcript(job_id) — реплики без служебных разделов. ⟦…⟧ у слов — нерешённые сомнительные места:
   в цитаты и выводы без проверки не брать. Стенограмму не копируйте в разговор целиком и не переписывайте вручную:
   путь к файлу есть в transcription_status.
4. Сомнительные места решает человек: review_items(job_id) → варианты (много мест — сначала brief=True и фильтры
   reasons/min_sec); build_review_pack → start_review_page (обычный локальный браузер, Claude не нужен)
   → apply_answers. Облачные карточки необязательны; их ответы сначала передаются через import_answers;
   прямые решения — apply_review. Приоритет: слово человека > голосовой профиль > распознавание > эвристики.
   Не выбирайте за человека, кто говорит в спорном месте. Остатки после разбора (слово вне места, неверное
   число) — edit_span(job_id, t1, t2, text, speaker) по словам человека; повторно решить место — reopen_review.
5. Голоса: rename_speakers подтверждает имена («Спикер 1» → имя) и пополняет профиль; update_voice_profiles —
   пересчёт по ответам; resolve_voice — только по решению человека о тёзке из карантина (voice_profiles → quarantine).
6. Конспект пишете вы: цитаты «…» берите только из стенограммы; check_quotes проверяет их, conspect_link связывает
   с местами стенограммы, conspect_render собирает документ.
7. После обновления программы — self_check (тесты, проверка файлов на личные данные, морфология).
8. Данные: get_settings (папка данных, группы, настройки). export_group, import_group (apply=true), delete_group,
   cleanup_jobs (apply=true), migrate_data (dry_run=false) меняют или выгружают личные данные — только по прямой
   просьбе человека, после отчёта в режиме «без применения». Каждое такое действие дополнительно подтверждает сам
   человек в окне на экране (или командой `python rt.py approve` в терминале): подтвердить за него нельзя, ответ
   «Человек не подтвердил» значит отказ — не повторяйте вызов без новой просьбы. Архив export_group содержит
   голосовые отпечатки (биометрию): не читайте его и не передавайте в разговор.
"""

VERSION = "2026.11.12.1"   # версия программы; видна клиентам в ответе initialize (serverInfo.version)

try:
    mcp = FastMCP("rus-transcriber", instructions=INSTRUCTIONS, version=VERSION)
except TypeError:   # mcp 1.x: параметра version нет
    mcp = FastMCP("rus-transcriber", instructions=INSTRUCTIONS)


async def _gated(action, args, lines_fn, run_fn):
    """Выполняет run_fn; опасное действие (action не None) — только после подтверждения человеком (confirm.py).
    Всё блокирующее (окно подтверждения, работа с файлами) идёт в потоке, чтобы не задерживать другие вызовы."""
    def work():
        try:
            if action:
                confirm.require(action, args, lines_fn())
            return json.dumps(run_fn(), ensure_ascii=False, indent=1)
        except confirm.NotConfirmed as ex:
            return str(ex)
        except core.CoreError as ex:
            return f"Ошибка: {ex}"
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
    whisper: третья модель (Whisper) для голосования «два из трёх»: "all" — вся запись; "doubt" — только
      куски, где GigaAM RNN-T и CTC расходятся (в остальных голосование решено без неё; быстрее, но
      ниже точность там, где обе GigaAM ошиблись одинаково, не исправит и "all"); "off" — без неё (самое
      быстрое, нет голосования); пусто — как в config.json (asr_third_mode, по умолчанию "all").
    whisper_max_repeat: сколько раз подряд Whisper может повторить одну фразу — защита от зацикливания;
      0 — без ограничения; -1 — как в config.json (по умолчанию 3). Его «субтитры» на тишине убираются всегда.
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
def rename_speakers(job_id: str, names: dict[str, str]) -> str:
    """Подписывает спикеров именами в готовой расшифровке (переписывает .md и .json).

    names: соответствие меток именам, например {"Спикер 1": "Вера", "Спикер 4": "Анна"}.
    Ключом можно указать и уже данное имя, чтобы исправить его. Исходные метки сохраняются
    в .json (поле label), поэтому переименовывать можно сколько угодно раз.
    """
    import transcript
    s = _status(job_id)
    if not s:
        return f"Задача {job_id} не найдена"
    if s.get("state") != "done" or "json" not in s:
        return "Расшифровка ещё не готова"
    data, unknown = transcript.rename(s["json"], s["markdown"], names)
    msg = {"speakers": data["speakers"], "mapping": data["speaker_names"], "markdown": s["markdown"]}
    if data.get("_conspects"):
        msg["conspects"] = data["_conspects"]
    if unknown:
        msg["warning"] = f"таких меток нет в расшифровке: {unknown}"
    return json.dumps(msg, ensure_ascii=False, indent=2)


@mcp.tool()
def review_items(job_id: str, offset: int = 0, limit: int = 40, reasons: str = "", min_sec: float = 0.0,
                 brief: bool = False, status: str = "open") -> str:
    """Сомнительные места готовой расшифровки (пословная обработка) — для смыслового разбора Claude.

    Метки: asr — RNN-T и CTC распознали по-разному (alt_ctc — вариант CTC); conf — модель не уверена;
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
def edit_span(job_id: str, t1: float, t2: float, text: str, speaker: str = "", note: str = "",
              by: str = "claude") -> str:
    """Правка произвольного отрезка стенограммы — для остатков, которые не исправил разбор, без пересборки
    карточек. Слова, начинающиеся в [t1, t2] (секунды записи), заменяются текстом text по словам (совпадающие
    сохраняют время); speaker — подписать отрезок этим участником (реплика делится); пустой text — удалить
    слова отрезка; на отрезке без слов text вставляется (speaker обязателен). Места разбора внутри отрезка
    закрываются. Пишется в журнал правок. by — "human", если текст продиктовал человек, иначе "claude".
    note — пояснение для журнала. Время слов — read_transcript(fmt="json", extra=True) или review_items."""
    try:
        return json.dumps(core.edit_span(job_id, t1, t2, text, speaker, note, by), ensure_ascii=False, indent=1)
    except core.CoreError as ex:
        return str(ex)


@mcp.tool()
def reopen_review(job_id: str, ids: str) -> str:
    """Снова открывает места разбора (решённые или оставленные человеку), например "r57,r88", — чтобы
    решить их заново через apply_review. Карточки пересобирать не нужно."""
    try:
        return json.dumps(core.reopen_review(job_id, ids), ensure_ascii=False, indent=1)
    except core.CoreError as ex:
        return str(ex)


@mcp.tool()
def apply_review(job_id: str, decisions: list[dict]) -> str:
    """Применяет решения по сомнительным местам и пишет их в журнал правок (edits).

    decisions: [{"id": "r5", "speaker": "Анна"}, {"id": "r7", "text": "Вереск"},
                {"id": "r9"} (принять как есть), {"id": "r12", "status": "human", "note": "неразборчиво"}].
    speaker — переподписать место (реплика при необходимости делится); text — заменить текст места
    (слова модели сохраняются в поле model и в журнале); status "human" — оставить человеку.
    После разбора .md содержит список «Проверить человеку» и журнал правок."""
    import transcript
    s = _status(job_id)
    if not s or s.get("state") != "done" or "json" not in s:
        return "Расшифровка не найдена или ещё не готова"
    _, report = transcript.apply_review(s["json"], s["markdown"], decisions)
    return json.dumps(report, ensure_ascii=False, indent=2)


@mcp.tool()
def check_quotes(job_id: str, conspect_text: str = "", conspect_path: str = "") -> str:
    """Проверяет цитаты «…» конспекта по стенограмме задачи: дословны ли они и тот ли говорящий.

    conspect_text — текст конспекта (markdown или обычный), либо conspect_path — путь к .md/.txt на этом
    компьютере. Говорящий проверяется у цитат вида «Имя: «…»». Поправки и составные цитаты требуют проверки.
    Возвращает сводку и список цитат, которые не дословны или приписаны не тому."""
    import quotes
    import transcript
    s = _status(job_id)
    if not s or s.get("state") != "done" or "json" not in s:
        return "Расшифровка не найдена или ещё не готова"
    text = conspect_text or (Path(conspect_path).expanduser().read_text(encoding="utf-8-sig") if conspect_path else "")
    if not text.strip():
        return "Передайте conspect_text или conspect_path"
    res = quotes.check(text, transcript.load(s["json"]))
    for r in res["problems"]:
        r["t"] = transcript.hms(r["t"])
    return json.dumps(res, ensure_ascii=False, indent=1)


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
        return f"Ошибка: {ex}"
    return await _gated("Подтверждение проверки конспекта", snapshot,
                        lambda: [f"Конспект: {name}", f"Задача: {job_id}",
                                 f"Ревизия источника: {snapshot['source_revision']}",
                                 "Подтвердите только если сами проверили смысл и цитаты по текущей стенограмме."],
                        lambda: core.confirm_conspect_analysis(snapshot))


@mcp.tool()
def conspect_candidates(job_id: str, conspect_text: str) -> str:
    """Без записи показать неоднозначные цитаты и возможные места с ID и ревизией."""
    try:
        return json.dumps(core.preview_conspect_candidates(job_id, conspect_text), ensure_ascii=False, indent=1)
    except core.CoreError as ex:
        return f"Ошибка: {ex}"


@mcp.tool()
async def conspect_link_selected(job_id: str, conspect_text: str, selections: dict,
                                 name: str = "конспект") -> str:
    """Создать новый конспект с выбранными человеком местами; нужно подтверждение человека.

    selections: {номер цитаты: {recording_id, source_revision, word_ids}} из conspect_candidates.
    Существующий конспект не перезаписывается; отказ без новой просьбы не повторять.
    """
    try:
        snapshot, report = core.prepare_selected_conspect(job_id, name, conspect_text, selections)
        lines = [f"Новый конспект: {name}", f"Задача: {job_id}, ревизия: {snapshot['source_revision']}"]
        for entry in report["candidates"]:
            selection = selections.get(str(entry["quote_index"]))
            if selection:
                option = next(x for x in entry["candidates"] if x["word_ids"] == selection["word_ids"])
                lines.append(f"Цитата «{entry['quote'][:80]}»: {option['speaker']}, {option['t1']}–{option['t2']} с")
        lines.append("Создать новый конспект с этими выбранными местами?")
    except (core.CoreError, KeyError, StopIteration) as ex:
        return f"Ошибка: {ex}"
    return await _gated("Выбор мест цитат", snapshot, lambda: lines,
                        lambda: core.save_selected_conspect(snapshot, conspect_text, selections))


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
async def self_check(tests: bool = True) -> str:
    """Самопроверка программы на этом компьютере — после обновления или при подозрении на сбой: тесты (в отдельном
    процессе, с отдельной временной папкой данных — настоящие данные не трогаются), проверка файлов программы на
    личные данные (tools/check_repo.py: только где нарушение, сами слова не показываются) и морфология
    (pymorphy3 для подсказок словаря). Только проверки, ничего не меняет. tests=False — без тестов (быстро).
    Тесты идут 15–60 с. Итог: ok, summary и подробности по каждой части."""
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
                        lambda: core.do_migrate(False))


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
                        lambda: core.do_export(group, dest_dir, with_jobs))


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
                        lambda: core.do_import(zip_path, mode, True))


@mcp.tool()
async def delete_group(group: str, confirm: str, permanent: bool = False) -> str:
    """Удаляет группу (словарь и голосовой профиль). confirm должен точно совпасть с именем группы. По умолчанию
    папка переносится в _deleted (можно вернуть); permanent=true удаляет навсегда. Вызывать только по прямой
    просьбе человека, назвавшего группу; permanent=true — только если он прямо сказал «навсегда». Человек
    дополнительно подтверждает удаление в окне на экране; без его «Да» ничего не удаляется."""
    def lines():
        if confirm != group:
            raise core.CoreError(f"Подтверждение не совпало: для удаления передайте confirm=«{group}»")
        return core.describe_delete(group, permanent)
    return await _gated("Удаление группы", {"group": group, "permanent": permanent}, lines,
                        lambda: core.do_delete_group(group, confirm, permanent))


@mcp.tool()
async def cleanup_jobs(days: int = 30, apply: bool = False) -> str:
    """Чистит статусы, логи и pid задач старше days дней в папке данных (идущие задачи не трогает).
    apply=false — только подсчёт; apply=true — после подтверждения человеком в окне на экране."""
    if not apply:
        return await _gated(None, None, None, lambda: core.do_cleanup(days, False))
    return await _gated("Очистка журналов задач", {"days": days}, lambda: core.describe_cleanup(days),
                        lambda: core.do_cleanup(days, True))


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
    """Собирает пакет разбора сомнительных мест: клипы (3,5 с до места, 1,5 с после) (mp3 моно 64 кбит/с) и manifest.json
    в папке <имя>.review рядом с расшифровкой. Целиком запись никуда не уходит.

    items — вопросы клиента по местам из review_items; без items — нейтральные вопросы:
      [{"id": "r5", "question": "Кто говорит «…»?", "options": [
          {"label": "Анна", "decision": {"speaker": "Анна"}},
          {"label": "Вера",   "decision": {"speaker": "Вера"}},
          {"label": "как есть", "decision": {}}], "recommended": 0}]
    decision — то, что получит apply_review при выборе варианта (speaker / text / status, note).
    2–4 варианта. Пересборка меняет version: ответы на прежние вопросы не применятся."""
    try:
        return json.dumps(core.build_review_pack(job_id, items), ensure_ascii=False, indent=1)
    except core.CoreError as ex:
        return str(ex)


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
    ans = P._read(pd / "answers.json", {"answers": {}})
    src = {}
    for a in ans.get("answers", {}).values():
        src[a.get("source")] = src.get(a.get("source"), 0) + 1
    out = {"pack": str(pd), "version": man["version"], "items": len(man["items"]),
           "answered": len(ans.get("answers", {})), "by_source": src,
           "conflicts": len(ans.get("conflicts") or []), "rejected": len(ans.get("rejected") or []),
           "files": ["cards.html", "manifest.json"] + [it["clip"] for it in man["items"]]}
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
    {путь на странице: путь относительно base} (файл здесь = base + путь) для публикации артефактом (Claude
    забирает их и публикует), remove — клипы, которые надо
    убрать со страницы, и artifact_url страницы, если она уже публиковалась (тогда обновлять её же).
    artifact_url — запомнить адрес опубликованной страницы для этого пакета."""
    import review_pack as P
    jp, _ = _job_json(job_id)
    if not jp:
        return "Расшифровка не найдена или ещё не готова"
    return json.dumps(P.cloud_bundle(jp, artifact_url or None), ensure_ascii=False, indent=1)


@mcp.tool()
def update_voice_profiles(job_id: str) -> str:
    """Пересчитывает голосовые профили по этой расшифровке после ответов человека (ответ человека главнее
    отпечатка): отрезки, подписанные человеком, идут в профиль с весом ×3 (кроме наложений и отрезков
    короче 0,8 с), из подтверждённых расшифровок — только слова без пометок сомнения; прежний вклад этой
    записи заменяется целиком, поэтому опровергнутое и отменённое уходит. Поправки «A → B» копятся как
    спутанные пары: на них сервер чаще ставит метку сомнения и склоняется к подтверждённому. Если профиль
    участника сильно изменился (сходство со старым < 0,8) — предупреждение. Защита от тёзок: профиль —
    той группы, с которой сделана расшифровка; вклад, непохожий на собранный голос имени, идёт в карантин
    (в ответе quarantine) до решения человека — resolve_voice. Возвращает job_id."""
    jp, _ = _job_json(job_id)
    if not jp:
        return "Расшифровка не найдена или ещё не готова"
    data = json.loads(Path(jp).read_text(encoding="utf-8"))
    audio = data.get("file")
    if not audio:
        return "В расшифровке нет пути к записи"
    jid = _launch("worker.py", ["--learn-only", jp], audio, "learn")
    return json.dumps({"job_id": jid, "hint": "Ход — transcription_status(job_id)"}, ensure_ascii=False)


@mcp.tool()
def resolve_voice(name: str, recording: str, decision: str, profile: str = "") -> str:
    """Решение человека по тёзке из карантина профиля (см. voice_profiles → quarantine): голос с именем
    name в записи recording (имя файла расшифровки или записи, можно часть) не похож на собранный профиль.
    decision: "same" — это тот же человек: вклад записи добавляется в профиль (запускается пересчёт);
    "different" — другой человек с тем же именем: в этот профиль запись по этому имени больше не идёт.
    profile — группа (имя из voice_profiles), если групп несколько. Возвращает job_id
    пересчёта или итог."""
    if decision not in ("same", "different"):
        return 'decision: "same" или "different"'
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
    d.setdefault("identity", {}).setdefault(akey, {})[name] = decision
    d["quarantine"][akey].pop(name, None)
    if not d["quarantine"][akey]:
        d["quarantine"].pop(akey)
    pf.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    if decision == "different":
        return json.dumps({"group": pf.parent.name, "name": name, "recording": akey,
                           "done": "в этот профиль запись по этому имени больше не идёт"}, ensure_ascii=False)
    jid = _launch("worker.py", ["--learn-only", q["json"]], akey, "learn")
    return json.dumps({"job_id": jid, "hint": "пересчёт профиля — transcription_status(job_id, wait_sec=50)"},
                      ensure_ascii=False)


@mcp.tool()
def import_answers(job_id: str, answers: list[dict], source: str = "cloud") -> str:
    """Сводит ответы в answers.json пакета. answers: [{"id", "version", "option" (с 0) | "kind"
    (human/skip/recommended/free) | "text", "audio" (путь к голосовому ответу на этом компьютере), "ts"}].
    Голосовые ответы распознаются GigaAM и сопоставляются с вариантами. Ответ на прежнюю version не
    принимается; на одно место из двух режимов побеждает более поздний, расхождение — в conflicts."""
    import review_pack as P
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
    wav = folder / f"{item_id or 'reply'}-listen-{int(time.time())}.wav"
    voice_io.beep()
    voice_io.record(wav, min(max(float(seconds), 1.0), 30.0))
    text = P.recognize_files([wav]).get(str(wav), "")
    out = {"heard": text, "audio": str(wav)}
    if jp and item_id:
        man = P.manifest(jp)
        out["answer"] = P.add_answers(jp, [{"id": item_id, "version": man["version"], "text": text,
                                            "audio": str(wav), "heard": True}], "local")
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
