# Лаборатория опытов: прогон моделей по имеющимся записям

Первый рабочий вариант лаборатории из направления РАР-4
([rar-roadmap.md](../docs/rar-roadmap.md), раздел «РАР-4»). Задача опыта — собрать данные для
вопроса «какие модели нужны для сведения и насколько меньше станет мест для разбора
человеком»: скорость этапов, итоги голосования, места для разбора по причинам и гипотезы
каждой модели. Анализ собранного — отдельный шаг (по формату ниже).

Папку `lab/` установщик (`install.ps1`) не ставит. Лаборатория запускается обычной командой
Python, без ИИ-агента и без MCP-сервера.

## Как запустить

Интерпретатор — из установленной программы (только в нём есть пакеты), код — из этого клона:

```powershell
$py = "$env:LOCALAPPDATA\rus-transcriber\.venv\Scripts\python.exe"
cd <папка клона>

# 1. Песочница: новая пустая папка вне папки данных, программы, клона и папок с аудио
& $py -B lab\experiment.py prepare --sandbox D:\lab\models-1 --voice-profiles copy

# 2. План и оценка времени, без счёта
& $py -B lab\experiment.py run --sandbox D:\lab\models-1 --dry-run

# 3. Прогон (можно прерывать Ctrl+C и запускать снова — посчитанное пропускается)
& $py -B lab\experiment.py run --sandbox D:\lab\models-1
& $py -B lab\experiment.py run --sandbox D:\lab\models-1 --configs c-no-third,f-whisper-doubt --limit 4

# 4. Пересобрать обезличенную сводку (прогон делает это сам в конце)
& $py -B lab\experiment.py summary --sandbox D:\lab\models-1
```

Ключи `run`: `--experiment` — файл опыта (по умолчанию
`lab/experiments/models-arbitration.json`); `--configs` — конфигурации через запятую (id или
начало id); `--units` — единицы через запятую (начало `unit_id`, его показывает `--dry-run`);
`--limit N` — не больше N пар за запуск; `--dry-run` — только план; `--program-dir` — папка
установленной программы, если не та, что при подготовке.

Перед прогоном: закрыть программу и не запускать расшифровки (лаборатория не занимает место в
очереди тяжёлых задач программы, а только отмечает в паспорте, была ли занята очередь),
питание от сети. Переменная `HF_TOKEN` не нужна: модели берутся из кэша без сети.

`prepare` по умолчанию берёт ту же папку данных, что программа (`RUS_TRANSCRIBER_DATA` или
`%APPDATA%\rus-transcriber-data`); другая — ключом `--source-data`. Папка программы — ключ
`--program-dir` (по умолчанию `%LOCALAPPDATA%\rus-transcriber`).

## Безопасность и что копируется

Песочница — обязательный параметр. Работа не начинается, если папка песочницы совпадает с
настоящей папкой данных, папкой установленной программы, клоном репозитория или папкой с
исходными записями, лежит внутри одной из них или содержит её (ссылки и junction
раскрываются, регистр в Windows не учитывается). Для новой подготовки нужна новая пустая папка.

| Что | Откуда | Куда в песочнице | Зачем |
|---|---|---|---|
| База записей `state.sqlite3` | папка данных | `source/state.sqlite3` | список записей, группы, прежние времена, решения человека — для анализа |
| `config.json` | папка данных | `data/config.json` | выбор моделей и устройств этой машины — основа конфигураций |
| Словари групп `vocab.json` | `groups/<группа>/` | `data/groups/<группа>/` | словарь и участники, как в обычной обработке |
| Голосовые профили `voices.json`, `voice-settings.json` — **биометрия** | `groups/<группа>/` | `data/groups/<группа>/` | только при `--voice-profiles copy`: имена говорящих и метки «голос» как в обычной обработке |
| Соль стенда `bench/testbed.json` | папка данных | `data/bench/` | тот же обезличенный идентификатор стенда, что в тесте скорости |

База копируется так: файл базы и её журнал `-wal` читаются побайтно во временную папку
песочницы (если файлы меняются во время чтения — повтор; при постоянных изменениях — отказ с
просьбой закрыть программу), из этой копии делается штатная резервная копия программы
(`Store.backup`), временная копия удаляется. Сам SQLite настоящую базу не открывает: даже
открытие «только на чтение» оставляет рядом файл `state.sqlite3-shm`, а папка данных не
должна меняться. Резервная копия неактивна (заблокирована до явной перепривязки), поэтому
программа по ней не пишет документы рядом с аудио. Копия лежит в `source/`, а не в `data/`:
у конвейера в песочнице нет рабочей базы записей.

Аудиофайлы только читаются (контрольная сумма при подготовке и перед прогоном; изменившийся
файл не обрабатывается). В настоящую папку данных лаборатория не пишет ничего; она лишь
проверяет в паспорте, заняты ли места очереди `jobs/slots` (проба блокировки без записи).
Из папки программы читаются только модели кэша `cache/` и интерпретатор; компиляция OpenVINO
идёт в `cache/openvino` песочницы. Всё работает без сети: `HF_HUB_OFFLINE=1`, откат на
PyTorch (он может скачивать модели) запрещён, эмоции выключены. Ничего не уходит за
пределы компьютера; выгрузка сводки — отдельное решение человека.

Неизменность источника закреплена тестом `tests/test_lab_experiment.py` (содержимое и время
изменения каждого файла папки данных и папки с аудио до и после подготовки и прогона).

## Устройство прогона

- **Единица опыта** — звук: записи базы с одинаковой контрольной суммой файла (варианты одной
  записи) считаются одним звуком. Группа берётся из стенограммы записи
  (`quality.runtime.group`); имена участников (`names`) не передаются — голос ищется среди
  всех участников словаря группы.
- **Общие этапы на звук** выполняются один раз и общие для всех его конфигураций в этом
  запуске: чтение звука (`worker.load_audio`), словарь (`worker.resolve_vocab_details`),
  разделение говорящих (`worker.diarize`), голоса. Голоса готовятся теми же функциями, что в
  `worker.main` для пословного конвейера, но профиль только читается
  (`worker.load_group_profile_for_recognition`): автопополнение выключено, чтобы прогон
  не зависел от порядка и ничего не менял. Эти ~40 строк повторяют порядок вызовов из
  `worker.main` (там они не выделены в функцию) — это единственное место, где лаборатория
  повторяет шаги, а не вызывает готовую функцию.
- **Каждая конфигурация** — вызов рабочего `worker.run_words` с настройками конфигурации:
  проходы моделей, голосование `vote3` или `mark_alt`, словарь, голос каждого слова, места
  для разбора, слой признаков разговора — тот же код, что в обычной расшифровке.
- **Гипотезы** перехватываются на время вызова, без правки конвейера: `words.transcribe`
  (слова основного и второго прохода до голосования), `words.clean_whisper` (текст третьей
  модели по кускам до и после чистки), `words.vote3` / `words.mark_alt` (какое сведение и
  с каким списком слов третьей модели), `backends.make_asr` (что запрошено и что
  получилось, откаты).
- **Результат** пишется в отдельную базу результатов `results/lab.sqlite3`, а не новой
  ревизией в копию базы записей: ревизия в базе записей регистрирует файлы стенограммы рядом с
  аудио (при открытии копии программой они могли бы появиться рядом с настоящими записями),
  сталкивается с уже зарегистрированными путями и смешивает опыт с работой человека.
  Сопоставление конфигураций на одной записи — по `unit_id`; связь с записями и решениями
  человека в копии базы — через `recording_ids` в `units.json`.
- **Возобновление.** Пара «единица × конфигурация» считается посчитанной, если в базе есть
  строка `state='done'` с тем же `config_sha256`. Сумма меняется при правке ролей или
  настроек конфигурации и при другом `config.json` песочницы — тогда пара считается заново
  (прежняя строка остаётся). Пары с ошибкой повторяются при следующем запуске.
- **Скорость** отдельно не меряется: записываются времена этапов, которые считает конвейер
  (`timing`), и условия запуска из `runinfo` (паспорт прогона; условия до и после каждой пары
  и проверка готовности `runinfo.preflight`).

## Конфигурации готового опыта

Файл `lab/experiments/models-arbitration.json`. Не заданные в ролях среды исполнения и
устройства берутся из `config.json` песочницы.

| id | Основная | Вторая | Третья | Зачем |
|---|---|---|---|---|
| `a-current` | GigaAM v3 RNN-T | GigaAM v3 CTC | faster-whisper (`whisper-ct2`), вся запись | как сейчас, опорная |
| `b-ml3` | RNN-T | CTC | GigaAM Multilingual large CTC, fp32, процессор | другая третья модель |
| `c-no-third` | RNN-T | CTC | — | нужен ли третий проход |
| `d-ml2-whisper3` | RNN-T | Multilingual large CTC | faster-whisper | вторая — независимая модель |
| `e-ml2-no-third` | RNN-T | Multilingual large CTC | — | независимая вторая без третьей |
| `f-whisper-doubt` | RNN-T | CTC | faster-whisper только в кусках расхождения RNN-T и CTC | дешёвая третья |

Особенности: Multilingual третьей считается кусками GigaAM (до 22 с), а не длинными кусками
Whisper (`long_chunks: false`); её текст строчный и без пунктуации, как и в тесте скорости.
Multilingual второй сохраняет времена слов; там, где голосование берёт вариант второй модели,
в стенограмму попадают её слова без пунктуации. В диагностике конвейера ключ третьей модели
называется `whisper` при любой модели.

Невозможно без правки `worker.py` (и потому нет в опыте): четыре и больше модели в одном
голосовании; основная модель без времён слов; голосование по другому правилу; Multilingual
третьей на видеокарте (для третьей модели на GPU конвейер пробует путь Whisper для
видеокарты). Пересвести сохранённые гипотезы любого набора моделей можно при анализе:
слова RNN-T, CTC и Multilingual со временами и тексты Whisper и Multilingual по кускам
сохраняются, функции `words.vote3` и `words.mark_alt` можно вызвать на них напрямую.

## Формат файла опыта

```json
{"schema": "lab-experiment-1", "name": "...", "description": "...",
 "allow_torch_fallback": false,
 "estimate_sec_per_audio_hour": {"third:multilingual_large_ctc:all": 900},
 "configurations": [
   {"id": "a-current", "label": "...", "note": "...",
    "roles": {"primary": {"model": "v3_e2e_rnnt", "backend": "onnx-cpu", "device": "CPU", "batch": 1, "quant": null},
              "second": {"model": "v3_e2e_ctc", "backend": "auto", "device": "CPU", "batch": 1},
              "third": {"engine": "whisper-ct2", "mode": "all"}},
    "settings": {"review_missed": true}}]}
```

- `id` — латиница в нижнем регистре, цифры, `-`, `_`, до 40 знаков; уникален.
- `primary`: `model` обязательно → `asr_model`; `backend` → `asr_backend`, `device` →
  `asr_device`, `batch` → `asr_batch`, `quant` → `asr_quant` и `asr_quant_cpu` (действует и на
  второй проход — так устроен конвейер; `null` — fp32).
- `second`: `null` — без второй модели (`asr_second_model: ""`); иначе `model` →
  `asr_second_model`, `backend`, `device`, `batch` → `asr_second_*`. Модель не может
  совпадать с основной.
- `third`: `null` — без третьей (`asr_third_mode: "off"`); требует второй модели. `engine`:
  `whisper-ct2` (faster-whisper; при сбое конвейер сам откатывается на ONNX-Whisper),
  `whisper-onnx` (ONNX-Whisper; `device` CPU или GPU, `quant`), `onnx` (любая модель onnx-asr
  по имени `model`, только CPU, `quant` → `asr_third_quant_cpu`, `null` — fp32). Общие поля:
  `mode` (`all` или `doubt`) → `asr_third_mode`, `max_repeat` → `asr_third_max_repeat`,
  `long_chunks` → `asr_third_long_chunks`.
- `settings` — прочие настройки `settings.py` (опечатка в имени — ошибка). Нельзя менять
  настройки общих этапов и заданные лабораторией: `diar_*`, `torch_threads`, `voices`,
  `voices_debug`, `default_group`, `pipeline`, `recording_backend`, `emotions`,
  `max_heavy_jobs`, `word_win_near_change`.
- Лаборатория всегда ставит `emotions: false`, `pipeline: "words"`.
- `estimate_sec_per_audio_hour` — свои оценки секунд счёта на час звука по ключам
  `primary:<модель>`, `second:<модель>`, `third:<модель или whisper-ct2>:<режим>`,
  `diarization`, `voices`, `word_voice`.

## Формат результатов

Песочница:

```
<песочница>/
  LAB-SANDBOX.json        метка песочницы: источник, папка программы, что скопировано
  units.json              единицы опыта (локально: в нём пути к аудио)
  source/state.sqlite3    неактивная резервная копия базы записей
  data/                   папка данных конвейера в песочнице (config.json, groups/, bench/, jobs/)
  results/lab.sqlite3     база результатов (тексты и гипотезы — только здесь)
  export/summary.json     обезличенная сводка
  tmp/, cache/openvino/   временные файлы и кэш компиляции
```

### `units.json` (схема `lab-units-1`)

`{"schema", "units": [...], "missing_audio_recordings": [recording_id, ...]}`. Единица:
`unit_id` (первые 16 знаков `audio_sha256`), `audio_sha256`, `location` (путь к аудио),
`size_bytes`, `group`, `group_conflict` (варианты с разными группами), `no_profile`,
`duration_sec` (из стенограммы), `recordings` — варианты: `recording_id`, `location`,
`group`, `no_profile`, `tag`, `duration_sec`, `timing` (прежние времена обработки этой записи:
`asr_sec`, `asr2_sec`, `asr3_sec`, `diarization_sec`, `voices_sec`, `word_voice_sec`, … и
названия `asr_model`, `asr2_model`, `asr3_model`, `asr3_device`, `asr3_mode`).

### `results/lab.sqlite3` (схема `lab-results-1`, в таблице `meta`)

Строки только добавляются (кроме отметок окончания запуска и звука).

- `runs` — запуск: `run_id`, `started_utc`, `finished_utc`, `state` (`running`, `finished`,
  `interrupted`), `experiment_name`, `experiment_sha256` (сумма текста файла опыта),
  `experiment_json`, `passport_json`, `args_json`.
- `configs` — конфигурация в версии: `config_id`, `config_sha256`, `label`, `roles_json`,
  `settings_json` (поле `settings` конфигурации), `effective_json` (все настройки, с которыми
  вызван конвейер), `first_run_id`.
- `unit_runs` — общий этап звука в запуске: `run_id`, `unit_id`, `state` (`running`, `done`,
  `error`), `error_kind`, `error` (хвост трассировки), `shared_json`: `duration_sec`,
  `diarization_sha256`, `diarization_segments` (`[начало, конец, метка pyannote]`),
  `diarization_info`, `turns`, `label_name` (метка → имя), `voices` (`voice_profile`,
  `profile_names`, `match`, `unmatched`, `voices_error`, `voices_off`), `timing`
  (`diarization_sec`, `voices_sec`, `embeddings_sec`). Векторов голосов нет.
- `pairs` — пара: `pair_id`, `run_id`, `unit_id`, `config_id`, `config_sha256`, `state`
  (`done` или `error`), `started_utc`, `finished_utc`, `error_kind`, `error`,
  `diarization_sha256` (у пар одного звука из разных запусков можно проверить, что разделение
  говорящих одинаково), `conditions_json` (`before`, `after` — `runinfo.conditions`;
  `preflight` — `runinfo.preflight`), `metrics_json` (ниже), `result_json`, `hypotheses_json`.

`result_json` — данные расшифровки, которые вернул `worker.run_words`, дополненные так же,
как в `worker.main`: `segments` (слова с `w`, `s`, `e`, `c`, `flags`, `alt`, `alt_whisper`,
`voted`, `model`, …), `review` (места для разбора: `reasons`, `t1`, `t2`, `text`, `alt_ctc`,
`alt_whisper`, `asr_problem`, …), `quality` (`vote`, `vote3_skipped`, `asr_votes`,
`asr_diagnostics`, `asr_run`, `asr_incomplete_chunks`, `asr2_error`, `asr3_errors`,
`missed_speech`, `missed_speech_diagnostics`, `interaction`, `words`, `flagged_words`, …),
`speakers`, `speaker_names`, `file`, `duration`, `group` и `timing`.

`timing` пары: времена конвейера (`asr_sec` — основной проход без загрузки модели,
`asr_chunks`, `asr_backend`, `asr_model`; `asr2_sec` — второй вместе с загрузкой, `asr2_model`;
`asr3_sec`, `asr3_load_sec`, `asr3_model`, `asr3_device` — `движок:устройство:точность`,
`asr3_name`, `asr3_mode`, `asr3_pieces` — `посчитано/всего` кусков, `asr3_removed_words`,
`asr3_fallbacks`, `asr3_compile_sec`; `word_voice_sec`, `word_windows`, `interaction_sec`) и
лаборатории: `diarization_sec`, `voices_sec`, `diarization` (шаги pyannote) — общие для звука;
`embeddings_sec` — общий этап плюс эта пара; `pair_sec` — время вызова `run_words`;
`total_sec` = общие этапы + `pair_sec` (сколько заняла бы обычная обработка с этой
конфигурацией, без эмоций); `realtime_factor` = длительность / `total_sec`;
`asr_fallbacks` — откаты основного прохода.

`hypotheses_json` (схема `lab-hyp-1`):

```
{"schema": "lab-hyp-1",
 "primary": {"model", "backend", "chunks": [{"a", "b", "own": [с, по]}], "words": [{"w", "s", "e", "c"}]},
 "second":  тот же вид или null,
 "third":   {"model", "backend", "name", "mode", "pieces",
             "chunks": [{"chunk_index", "audio_span", "own_span", "state", "raw_text", "text", "removed_words"}]} или null,
 "vote":    {"function": "vote3" | "mark_alt", "third_words": [слова, поданные в vote3] | null} или null}
```

Слова — до голосования и словаря; `c` — уверенность слова; времена в секундах от начала
записи. Слово попадает в проход, только если начинается в своей части куска (`own`), как в
конвейере. У третьей модели `raw_text` — ответ модели, `text` — после `clean_whisper`;
в режиме `doubt` кусков меньше, а в `vote.third_words` на месте непосчитанных кусков стоят
слова RNN-T (так конвейер подаёт их в голосование). `state` куска — из диагностики
(`ok`, `empty_unknown`, `error`).

### `metrics_json` (схема `lab-metrics-1`) и сводка

`metrics_json` — только числа и названия: `duration_sec`, `words`, `flagged_words`,
`review` (`total`, `by_reason` — `asr`, `conf`, `voice`, `missed`, `overlap`, `gender`;
место с несколькими причинами считается в каждой; `other_reasons`, `multi_reason_items`,
`asr_problem_items`), `review_per_hour`, `vote` (`agree3`, `agree2`, `replaced`, `inserted`,
`disputed`), `vote3_skipped`, `asr_votes`, `asr_incomplete_chunks`, `asr_diagnostics`
(по ролям `primary`, `second`, `whisper`: `backend`, `chunks`, `states` — число кусков по
состояниям), `asr2_error`, `asr3_errors` (только `движок:устройство: ТипОшибки`),
`missed_speech`, `voice_doubt_threshold`, `overlaps_sec`, `timing` (числовые поля выше и
названия моделей и сред), `fallbacks.asr3`, `make_asr` (каждый вызов: `requested` —
`asr_model`, `asr_backend`, `asr_device`, `asr_batch`, `asr_quant`, `asr_quant_cpu`,
`asr_language`; `actual` — фактическая среда; `providers`; `fallbacks` — типы ошибок
откатов; `error`), `hypotheses` (число слов по ролям).

`export/summary.json` (схема `lab-summary-1`) — пригодная для выгрузки сводка: `runs`
(паспорт `runinfo`: окружение, железо, обезличенный `testbed_id`, кэш моделей, работа без
сети, настройки `asr_*`/`diar_*`), `configs` (`config_id` → версии с `config_sha256`, `label`,
`roles`), `units` (`unit_id`, `audio_sha256`, `duration_sec`, `variants`), `pairs` (`run_id`,
`unit_id`, `config_id`, `config_sha256`, `state`, времена, `error_kind`, `conditions`,
`metrics`). В сводке нет текстов, гипотез, имён, путей, имён файлов и групп; тест проверяет
это на синтетических данных. Выгрузка — только по решению человека.

## Оценка времени

`--dry-run` показывает план и оценку по каждой паре. Оценка берётся по порядку: медиана
посчитанных пар этой песочницы; прежние времена обработки из копии базы (для моделей, которые
уже стояли в обработке); `estimate_sec_per_audio_hour` файла опыта; грубые значения по
умолчанию (помечаются «грубо»). Для набора около 1,75 ч звука и шести конфигураций грубая
оценка — 4–6 часов счёта на процессоре, из них больше половины — faster-whisper в
конфигурациях `a` и `d`; разделение говорящих считается один раз на звук. Уточнённое число
даст `--dry-run` по истории обработки.

## Что не умеет и упрощения относительно РАР-4

- Нет «границы этапов» 4.2: гипотезы получаются перехватом вызовов, а не штатной функцией
  «проход → гипотеза»; формат гипотез лаборатории свой (`lab-hyp-1`).
- Пересведение сохранённых гипотез и попарное согласие моделей — задача анализа (этап 4.3);
  прогон только собирает данные.
- База результатов своя (`lab.sqlite3`), не база прогонов из
  [bench-database-spec.md](../docs/bench-database-spec.md): нет стенда и окружения отдельными
  сущностями, повторов замера, полосы шума и правила «один вариант — один процесс». Все
  конфигурации одного звука считаются в одном процессе подряд, поэтому холодный старт модели
  (загрузка) у первой пары и у следующих может отличаться.
- Разделение говорящих общее только внутри одного запуска: если прогон прерван посреди звука,
  следующий запуск разделит этот звук заново (несколько минут); совпадение проверяется по
  `diarization_sha256`.
- Эмоции не считаются. Автопополнение голосовых профилей выключено. Имена участников записи
  (`names`) не передаются.
- Эталонов нет: точность слов (WER) не считается; решения человека — в копии базы.
- Проверено только на синтетических данных с подменой моделей (см. отчёт о проверках);
  настоящий прогон — отдельное решение владельца.
