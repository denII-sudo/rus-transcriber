# Аудит кода rus-transcriber: обновление 2026-11-20

> **Исторический снимок, не описание текущей версии.** Отчёт относится к указанному
> ниже коммиту и не обновлялся по последующим изменениям проекта. Его выводы,
> номера строк и рекомендации требуют самостоятельной проверки по текущему коду;
> утверждения об отсутствии дефектов или секретов не являются гарантией безопасности.
> В исходном отчёте есть противоречия: статусы M4 и M6 в разных разделах расходятся,
> а F3 обозначает разные вещи в разделах 4 и 6. Утверждение об отсутствии CI-файлов
> также нельзя применять к текущему репозиторию. Предыдущий аудит позднее перенесён
> в `docs/history/2026-11-18-code-audit.md`. Основной текст ниже сохранён как источник.
>
> **Архив (перенесён в `docs/history/` 23.11.2026).** Сверка с `main` `b916181` чтением кода, без запусков: ещё верны F1 (`server.resolve_voice` оставляет временный файл при сбое записи), F2 (`server.check_quotes` не перепроверяет размер файла после подтверждения) и F5 (`paths.py` молча пропускает ошибку создания папок); M12 и L6 не перепроверялись; остальное закрыто или устарело (номера строк, «CI в репозитории нет»).

## Метаданные аудита

| Параметр | Значение |
| --- | --- |
| Дата | 20.11.2026 |
| Коммит HEAD | `f0462eb` |
| Метод | статический анализ: чтение публичной поверхности + pattern-скан (grep: subprocess, eval/exec/compile/pickle, network, assert, `except: pass`, TODO, секреты, icacls) + верификация находок предыдущего аудита по текущему коду |
| Динамические прогоны | не выполнялись |
| Реальные данные | не использовались |
| Изменения рабочего кода | не вносились; создан только этот файл |

Модель: идентификатор не раскрывается (провайдер OpenCode).

## 1. Кратко

**Новые находки: 2 (низко, F1–F2) + наблюдения (F3–F5).** Предыдущий аудит
(docs/2026-11-18-code-audit.md, 29 пунктов) перепроверен по текущему коду —
большинство закрыто (статусы раздела 3). Открыты до решений: M7 (отложен владельцем),
M12 (проверка ранее по worker.py не перечитывалась), M4 (отложен), остаточный риск M13
(см. F3), M6-частично, M11-остаток («default» незапрещённый — по задумке).

Изменения с момента аудита 17–18.11.2026: коммит `e697ba7` «fix: address verified
audit findings» (проверка + частичные исправления), коммит `909cd0d` — перенос данных
только по явной команде (`migrate_data` с подтверждением человека; сервер больше не
вызывает `migrate.run()` при старте), коммит `3bbe791` — перенос старых подтверждений
говорящего через карточки; остальное — CI (`9dac4eb`, `647ff03`, `69622f3`, `ada1aa8`),
документация, chore.

Основной вывод: публичный репозиторий — программа под Windows; публичная поверхность:
MCP-сервер (`server.py`, ≈40 инструментов и промптов), консоль (`rt.py`), установщик
(`install.ps1`), инструменты (`tools/`), HTTP-страницы (`review_page.py`,
`workflow_page.py`). Граница «человек/модель» соблюдена на действиях с биометрией и
авторством; секретов и токенов в репозитории не обнаружено (grep + чтение).

## 2. Оценка сложности и охват

- 87 файлов Python (36 модулей корня + 46 тестов + 5 в `tools/`), 4 PowerShell, 2 HTML;
  суммарно ~1.8 МБ.
- Точки входа: `server.py` (MCP-слой), `rt.py` (консоль), `install.ps1`, `tools/*.py`
  (9 инструментов), `review_page.py`/`workflow_page.py` (HTTP-страницы на 127.0.0.1),
  `llm_provider.py` (OpenCode API + OpenAI), `selfcheck.py`, `bench.py`.
- Перечитанные заново полностью: server.py, core.py, storage.py, data_io.py,
  review_pack.py, settings.py, paths.py, confirm.py, install.ps1 (1–1027),
  llm_provider.py (1–40, 200–347), documents.py, tmpfiles.py, selfcheck.py, tools/*
  (check_repo, import_legacy, move_data, move_data.ps1, verify_document_files,
  eval_quality), review_page.py, workflow_page.py, .github/CODEOWNERS.
  По предыдущему аудиту (перечтение не повторялось, код по диффам сверен):
  worker.py, transcript.py, jobslots.py, recordings.py, conspect.py, workflow.py,
  vocab.py, voices.py, words.py, quotes.py, freeform.py, gender.py, diar_fast.py,
  backends.py, migrate.py, model_tasks.py, download_hint.py, product_policy.py,
  bench.py, rt.py, voice_io.py, voice_review.py.
- Pattern-скан: все `pass`-ветки в production проверены — безопасные cleanup/except
  (напр., server.py:955 — `except Exception: pass` при чтении флага настроек).

## 3. Статус находок предыдущего аудита

| ID | Уровень | Статус | Подтверждение по текущему коду |
| --- | --- | --- | --- |
| C1 | Критично | **Закрыт** | `core._valid_job_id` (core.py:128-130) + `core._job_path` (133-140) + `is_relative_to(JOBS.resolve())`; `job_status` → None (145-146); `server._status` (server.py:30-33) |
| H1 | Высоко | **Закрыт** | `server.rename_speakers` — `require_human=True` обе ветки (server.py:454-465); `core.apply_answers_files` — проверка `actor_kind == "human"` + `confirmation_context` (core.py:1554-1562); `core.apply_answers_files` SQLite-ветка (1563-1602) |
| H2 | Высоко | **Закрыт** | `require_human=True`: export_group (server.py:904-906), import_group (918-920), delete_group (934-935), update_voice_profiles (1071-1073), revoke_voice_profiles (1085-1087), set_voice_auto_learn (1094-1096), set_voice_quality_policy (1116-1120), backup_voice_profiles (1139-1143), backup_state (317-322), migrate_data (892-893); прямые проверки `mode() == "off"`: workflow_abandon_request (189-190), workflow_mark_reviewed (207-208), restore_recording_revision (300-301), resolve_voice (1157-1158), conspect_link_selected (667-669) |
| H3 | Высоко | **Закрыт** | `review_pack._answers_lock` (81-90) — OS-блокировка `answers.lock` + `jobslots._try_lock`; `review_pack._write_answers` (316-321); MCP `import_answers` — блокировка видима из любого процесса (server.py:1222) |
| H4 | Высоко | **Закрыт** | `server._listen_reply` (1278-1286): `item_id` ищется в manifest, имя из manifest, `wav.resolve().is_relative_to(folder.resolve())` |
| M1 | Средне | **Закрыт** | `server.check_quotes` — подтверждение + лимит 2 МиБ (server.py:562-571); `server.import_answers` — пути `audio` запрещены (1223-1224); `export_group`/`import_group` — имена через `safe_group` (data_io.py:86, 304, 332) |
| M2 | Средне | **Закрыт** | `core.cancel_job` (244-261): после `taskkill` — `alive(pid)` + отказ «Процесс не остановлен»; перечитка статуса под `jobslots.file_lock(.status.lock)`; запись через `NamedTemporaryFile` + `os.replace` |
| M3 | Средне | **Закрыт** | `server.delete_group` — `confirm` сравнивается с `safe_group(group)` (server.py:929-932) |
| M4 | Средне | **Закрыт** | `data_io.import_data` — `settings.group_voice_lock(g)` (338-340); `data_io._write_json/_write_bytes` — уникальный temp + `os.replace` (174-193) |
| M5 | Средне | **Закрыт** | `settings._save_overrides_locked` — `NamedTemporaryFile` в DATA_DIR + lock + `os.replace` (settings.py:120-125, 230) |
| M6 | Средне | **Закрыт** | `settings.group_voice_lock` — 60 с deadline (settings.py:243-246) |
| M7 | Средне | **Открыт** | Владелец не отдельно разрешил; `migrate.run()` при старте уже не вызывается (server.py:36-37). Оставляем решение владельца |
| M8 | Средне | **Закрыт** | `confirm.clean` — стрипит Cc/Cf/Cs (confirm.py:58-69) |
| M9 | Средне | **Закрыт** | `review_pack.drop_audio` — `rmtree` + `rmdir` (849-867) |
| M10 | Средне | **Закрыт** | `data_io.cleanup_jobs` — `active_job` через `core.job_process_alive` (378-391) |
| M11 | Средне | **Почти** | `paths.safe_group` (87-95): зарезервированные имена Windows без учёта регистра и расширений, ≤64 символов; остаток — «default» (по задумке: обычное имя, в `check_repo.GENERIC_GROUP_NAMES`) |
| M12 | Средне | **Открыт** | Проверка прежнего аудита по worker.py не перечитывалась (worker.py:1366-1379). Приоритетно — отдельный проход |
| M13 | Средне | **Частично** | Закрыто: `core.launch` — метка запуска процесса (`process_start_time`, core.py:60-90, 192-203); `core.cancel_job` — проверка принадлежности (223-228). Остаток: состояние «потерянного worker» (F3) |
| M14 | Средне | **Закрыт** | Сессия 13 (e697ba7): legacy-запись стенограммы атомарно (transcript.py) |
| M15 | Средне | **Закрыт** | Сессия 13 (e697ba7): `core._apply_answers_files_locked` + `P.mark_applied` (1607-1731) |
| L1 | Низко | **Закрыт** | `settings.load_config`/`update` — типы + сообщения об ошибке (settings.py:44-80, 132-139) |
| L2 | Низко | **Закрыт** | `tmpfiles.sweep_stale` — внутренний скан mtime (tmpfiles.py:72-82) |
| L3 | Низко | **Закрыт** | `selfcheck.run_tests` — sandbox с `RUS_TRANSCRIBER_DATA`/`TEMP`/`TMP` + `PYTHONDONTWRITEBYTECODE` (selfcheck.py:56-83) |
| L4 | Низко | **Закрыт** | `llm_provider` — проверка владельца слушающего (`listener_pids` + tree, llm_provider.py:60-113, 244-255) |
| L5 | Низко | **Закрыт** | `confirm._write` — уникальный temp + `os.replace` (confirm.py:90-101) |
| L6 | Низко | **Частично** | `core.launch` (207-214): при ошибке записи PID — `proc.kill()` + отсечение сирот; если kill/wait падает — сирота возможна |
| L7 | Низко | **Закрыт** | Сессия 13 (e697ba7) |
| L8 | Низко | **Закрыт** | `core.prepare_answer_confirmation` — хэш из answers + manifest, без interaction_events (core.py:1012-1069) |
| L9 | Низко | **Закрыт** | `data_io.export_data` пропускает `*.voices_debug.json` (120-121); `core.list_jobs` фильтрует (core.py:940) |

## 4. Новые находки (текущий код)

| ID | Уровень | Где | Суть |
| --- | --- | --- | --- |
| F1 | Низко | `server.py:1193-1196` (`resolve_voice`) | При ошибке `json.dump` или `os.replace` в `NamedTemporaryFile(delete=False)` в папке профиля остаётся файл `voices.json.<rand>.tmp` (унифицировать с `data_io._write_bytes` — там try/finally `tmp.unlink`) |
| F2 | Низко | `server.py:566` vs `server.py:574` (`check_quotes`) | Лимит 2 МиБ проверяется до подтверждения (`path.stat().st_size`), но `run()` читает `path.read_text` без повторной проверки — TOCTOU при росте файла; человек подтверждает один размер, содержимое может отличаться |
| F3 | Наблюдение | `server.py:337-372` (`start_transcription`) | Агент может запустить расшифровку по любому локальному пути `audio_path` (нет привязки к DATA_DIR). По задумке инструмента «полный путь к файлу на этом компьютере»; результат виден через `transcription_status` |
| F4 | Наблюдение | `worker.py:993`, `worker.py:1089` | Worker читает `HF_TOKEN` из окружения (по задумке; не возвращает его в ответы); `worker.py:340` — текст ошибки упоминает слово HF_TOKEN |
| F5 | Наблюдение | `paths.py:56-60` | При старте `mkdir` для GROUPS/JOBS/BENCH подавляет `OSError` (`pass`) — ошибки появятся позже в случайных местах; мелкая юзабильность |

Итог: **2 низкие (F1–F2) + 3 наблюдения (F3–F5)**. Отсутствие критично-высоких
подтверждено чтением поверхности и перечиткой прошлого аудита.

## 5. Проверено и не является ошибкой

- **О границе «человек/модель»:** все действия с биометрией/авторством в MCP требуют
  `require_human=True` или явных проверок режима `confirm_mode` (H2 закрыт).
  `import_answers(source="cloud")` — архив без путей аудио; `apply_answers_files`
  отказывает ответам неизвестного происхождения (H1 закрыт).
- **О путях и секретах:** grep по `hf_…`-значениям, `Bearer`, `HF_TOKEN` — только чтение
  токена из окружения worker.py:993 + чтение сохранённого токена из настроек приложения в
  install.ps1:910-911 + плейсхолдер `ВСТАВЬТЕ_СЮДА_hf_ТОКЕН` (install.ps1:911, 961).
  Пароль OpenCode — случайный `token_urlsafe(32)` (llm_provider.py:218), только на
  127.0.0.1, не записывается в логи (llm_provider.py:217, 229, 246-248).
- **О shell/exec:** нигде в корне и tools нет `shell=True` (grep по subprocess);
  `eval`/`exec`/`compile`/`pickle` — только `ast.literal_eval` (diar_fast.py:85,
  download_hint.py:45, worker.py:60); `re.compile` — обычные паттерны.
- **О записи по файлам:** `os.replace` используется только в контекстах atomic-write
  (кроме core.cancel_job — закрыт M2; и L6-остаток выше).
- **Обновлённые «проверено» из прошлого аудита:** `job_status("..\\..\\x")` → None;
  `read_transcript` — через `_need` и `job_json` с валидацией job_id;
  `check_quotes` — подтверждение снаружи; `data_io._read_archive` — имена членов, size,
  manifest, SHA-256; `documents.checked_root` — без `..` и reparse-точек;
  `storage.save_conspect` — пути через `documents._relative_parts` + collision checks.
- **О .github/:** только CODEOWNERS; CI-файлы (`tests.yml`, `automerge.yml`) в публичном
  репозитории нет (по истории коммитов — в апстрим/приватном репозитории).
- **О install.ps1:** `icacls` — по задумке; запись токена в настройки приложения ИИ-агента
  — по задумке (с проверкой whoami и доступа к pyannote).

## 6. План: что стоит сделать (приоритеты)

1. **F1, F2** — низкие, можно в любой момент (исправление: try/finally унификация
   с `data_io._write_bytes`; повторная проверка размера файла в `check_quotes`).
2. **Решение по M7** (возобновление миграции) — открыт владельцу.
3. **M12** — отдельный проход по worker.py/transcript.py.
4. **M13-остаток (F3)** — после F3 закрыть полностью: в `job_status`, если процесс мёртв
   и есть активный `.pid`, задача помечается `error` («потерянный worker»).
5. **M4, M6, M11, M13-часть, L6-часть** — по мере касания соответствующих модулей
   (сейчас закрыты кодом; M4/M6 — решения владельца оставляются открытыми).

## 7. Ограничения

- Не перечитывались заново: перечень в разделе 2 — по предыдущему аудиту; изменения с
  момента аудита — см. раздел 1 (список коммитов).
- Динамические прогоны, реальные данные, пентест, тестирование сторонних библиотек —
  не выполнялись (не просили).
- Аудит не гарантирует отсутствие других дефектов: проверены модули перечня; остальные —
  по предыдущему аудиту.
