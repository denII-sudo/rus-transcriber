# Источник 04. Python-фреймворки интерфейса

Отчёт поискового субагента (модель Sonnet), 21.11.2026. Приведён как получен; не перепроверен,
кроме оговорённого в [README.md](../README.md). Обращения «вы», «твой» — к заказчику поиска.

## Сравнение Python-фреймворков для адаптивного офлайн-интерфейса расшифровщика

### 0. Как собраны данные и что не проверено

- **Версии, даты релизов, лицензии, `requires_python`, зависимости.** Взяты из PyPI JSON API: `https://pypi.org/pypi/<пакет>/json`, например [nicegui](https://pypi.org/pypi/nicegui/json) и [reflex](https://pypi.org/pypi/reflex/json). Запросы выполнены 22.11.2026.
- **Звёзды, открытые issues, дата последнего коммита, флаг archived.** Взяты из GitHub API, например [nicegui](https://api.github.com/repos/zauberzeug/nicegui).
- **Число пакетов и «вес».** Получено через `pip install --dry-run --report` под Windows и Python 3.12. Размеры суммируют сжатые wheel-файлы с PyPI. Это не размер после установки, он обычно в 2–3 раза больше.
- **«Не грузит ли с CDN».** Я скачал wheel-пакеты и просканировал их на внешние адреса. Найденная строка с адресом означает только «в коде есть ссылка». Загрузку в браузере я не проверял, и это везде оговорено.
- **Не проверялось:**
  - живая работа на телефоне, измеренная скорость при десятках тысяч слов, отсутствие сети в реальном браузере;
  - возможно устаревшее: версии и звёзды на момент запроса, содержимое документации на момент чтения.
- **Где прочитана только страница-аннотация или обрывок, это отмечено словом «аннотация».** Часть страниц WebFetch отдавал усечёнными.

**Находка по текущему решению (гипотеза, кода проекта я не видел).** Стандартный `http.server` не поддерживает HTTP Range, issue [cpython#86809](https://github.com/python/cpython/issues/86809) остаётся открытым (статус на 09.08.2026). Без Range перемотка и проигрыш отрезков в `<audio>` работают плохо. Starlette поддерживает Range в `FileResponse` с версии 0.39.0 (по [истории issue #950](https://github.com/Kludex/starlette/issues/950) и сводке поиска, первоисточник — changelog Starlette, я его целиком не читал). Любой ASGI-вариант ниже (NiceGUI, FastAPI, FastHTML, Flet-web) это даёт сам.

---

### 1. Таблица A: живость, лицензия, вес

Колонка «выпусков 2026» считает и предрелизы. Значения для PySide6 и pywebview в звёздах не показательны.

| Фреймворк | Версия / дата релиза | Звёзды / открытые issues | Лицензия | Python | Зависимости и вес (Win, py3.12) | Признаки заброшенности |
|---|---|---|---|---|---|---|
| **NiceGUI** | 4.0.0 / 04.11.2026 | 16 354 / 34 | MIT | ≥3.11 | 49 пакетов, 16,3 МБ (сам wheel 8,2 МБ) | Нет. 21 выпуск за 2026, коммит 20.11.2026 |
| **Reflex** | 0.11.0 / 12.11.2026 | 29 043 / 365 | Apache-2.0 | ≥3.11 | 42 пакета, 9 МБ. Плюс Bun/Node при сборке фронтенда | Нет. 71 выпуск за 2026 (много alpha) |
| **Flet** | 1.1.2 / 06.11.2026 | 17 583 / 209 | Apache-2.0 | ≥3.10 | `flet[all]` 54 пакета, 47,7 МБ; `flet[web]` 28 пакетов, 35,6 МБ (flet-web 30 МБ) | Нет. 48 выпусков за 2026 |
| **Streamlit** | 1.67.0 / 05.11.2026 | 46 467 / 1195 | Apache-2.0 | ≥3.11 | 34 пакета, 86,8 МБ (pyarrow 29, numpy 12,6, pydeck 11) | Нет |
| **Gradio** | 6.32.0 / 12.11.2026 (есть 7.0.0.dev0) | 44 117 / 118 | Apache-2.0 | ≥3.10 | 53 пакета, 75,3 МБ (gradio 31) | Нет |
| **Panel** | 1.10.0 / 23.10.2026 (1.11 rc) | 5 828 / 766 | BSD-3 | ≥3.12 | 31 пакет, 63,7 МБ (+Bokeh, numpy, pandas) | Нет |
| **Solara** | 1.66.3 / 10.11.2026 | 2 180 / 348 | MIT | ≥3.8 | 71 пакет, 61,6 МБ (стек Jupyter: ipykernel, ipyvue, ipyvuetify) | Нет |
| **Textual** | 8.2.8 / 30.06.2026 | 37 703 / 419 | MIT | ≥3.9 | с textual-serve 22 пакета, 4,2 МБ | Репозиторий живой (коммит 09.10.2026), но выпусков нет с июня. В шаблоне баг-репорта автор пишет, что отвечает медленно |
| **textual-serve** | 1.1.3 / 01.11.2025 | 411 / 23 | MIT | ≥3.9 | входит в строку выше | Последний релиз ≈ год назад |
| **textual-web** | 0.8.0 / 30.08.2024 | 1 477 / 23 | MIT | ≥3.8 | — | Последний релиз 2024, фактически заброшен |
| **Toga (BeeWare)** | 0.5.7 / 01.10.2026 | 5 431 / 326 | BSD-3 | ≥3.11 | на Windows 11 пакетов, 8,5 МБ (pythonnet, WebView2) | Нет, но 5 выпусков за 2026 |
| **Kivy** | 2.3.2 / 08.11.2026 | 19 028 / 890 | MIT | ≥3.8 | на Windows 15 пакетов, 23,4 МБ (angle, sdl2, pywin32) | Предыдущий релиз 2.3.1 был 26.12.2024, то есть перерыв почти два года |
| **KivyMD** | 2.0.0 / 06.08.2026 | 2 465 / 61 | MIT | ≥3.7 | — | Предыдущий релиз 1.2.0 был 21.01.2024 |
| **PySide6** | 6.12.0 / 08.10.2026 | звёзды репозитория-зеркала не показательны | LGPL-3.0 / GPL-2.0 / GPL-3.0 | 3.10–3.15 | Essentials 82 МБ + Addons 40,5 МБ (Windows). Мета-пакет `PySide6` тянет ещё WebEngine 131,7 МБ и Pdf | Нет |
| **pywebview** | 6.2.1 / 15.04.2026 | 6 176 / 14 | BSD-3 | ≥3.8 | 8 пакетов, 1,2 МБ | Нет, но релизы редкие (2 за 2026) |
| **FastHTML** | 0.14.27 / 21.11.2026 | 7 063 / 75 | Apache-2.0 | ≥3.10 | 24 пакета, 2,6 МБ | Нет. 44 выпуска за 2026, но версия 0.x |
| **FastAPI + Jinja2 + uvicorn** | FastAPI 0.165.0 / 18.11.2026 | 104 997 / 49 | MIT | ≥3.11 | 17 пакетов, 3,8 МБ | Нет |
| **Starlette** | 1.10.0 / 24.10.2026 | 12 747 / 36 | BSD-3 | ≥3.11 | входит в строку выше | Нет |
| **Litestar** | 2.25.0 / 18.10.2026 | 8 565 / 359 | MIT | ≥3.8 | с jinja и standard 33 пакета, 7 МБ | Нет, но релизы реже (7 за 2026) |
| **Shiny for Python** | 1.9.0 / 29.10.2026 | 1 771 / 468 | MIT | ≥3.10 | 32 пакета, 12,5 МБ | Нет |
| **Dash** | 4.5.1 / 19.11.2026 | 24 501 / 427 | MIT | ≥3.10 | 29 пакетов, 15 МБ | Нет |
| **Rio** (`rio-ui`) | 0.12.4 / 29.10.2026 | 3 649 / 77 | Apache-2.0 | ≥3.11 | — | Стадия Beta, 5 выпусков за 2026 |
| **Mesop** | 1.3.7 / 26.09.2026 | 6 572 | Apache-2.0 | ≥3.10 | — | **Заброшен.** README: [«no longer maintained as of September 30, 2026»](https://github.com/mesop-dev/mesop) |
| **FastUI** | 0.9.0 / 28.10.2025 | 8 920 | MIT | ≥3.10 | — | **Заброшен.** README: [«this project is inactive»](https://github.com/pydantic/FastUI), репозиторий archived |

Остальные, найденные самостоятельно (данные PyPI и GitHub):

- **ReactPy**: стабильная 1.1.0 от 24.11.2024, ветка 2.0 — только беты до июля 2026.
- **Taipy**: 4.1.1 от 16.02.2026, требует Python <3.13.
- **trame** (Kitware): 4.0.0 от 09.09.2026, научная визуализация.
- **Hyperdiv**: 0.1.9 от 14.10.2025, 971 звезда, неактивен.
- **pytauri**: 0.8.0 от 01.09.2025, но коммиты до 21.11.2026.
- **pyview-web**: 0.9.0 от 03.07.2026, 142 звезды.
- **marimo**: 0.25.2 от 19.10.2026, 23 673 звезды, Apache-2.0.
- **Slint**: Python-привязка 1.19.0b1 от 16.11.2026, помечена Beta.
- **Dear PyGui**: 2.3.1 от 01.05.2026, MIT.
- **wxPython**: 4.3.2 от 13.10.2026.
- **Flask**: 3.1.3 от 19.02.2026, один релиз за 2026.
- **Quart**: 0.23.1, требует Python ≥3.13.
- **Django**: 6.1.3 от 03.11.2026; **django-htmx**: 1.29.0 от 05.08.2026.
- **htmx**: [v4.0.2 от 18.11.2026, 4.0.0 вышел 28.08.2026, ветка 2.x тоже поддерживается (v2.0.11 от 01.11.2026)](https://github.com/bigskysoftware/htmx/releases).
- **Air** (FastAPI+HTMX): 0.48.1 от 03.04.2026, на GitHub стоит флаг archived.
- **datastar-py**: 1.0.3 от 27.09.2026.
- **htmy**: 0.13.6 от 23.10.2026.
- **jinja2-fragments**: 1.12.0 от 08.04.2026.
- **sse-starlette**: 3.7.0 от 07.11.2026.
- Старые и неактивные: justpy (2023), remi (2022), PyWebIO (2025-04), Eel (2025-06, archived), flask-sock (2023).

---

### 2. Таблица B: возможности для задачи

Условные обозначения: «+» подтверждено источником, «±» частично или с оговоркой, «−» нет или плохо, «?» не нашёл.

| Фреймворк | Офлайн без CDN | Телефон «из коробки» | Аудио и отрезки | Горячие клавиши | Ход долгой задачи | Доступ по сети и защита | Упаковка | Мобильные сборки | Длинные списки |
|---|---|---|---|---|---|---|---|---|---|
| **NiceGUI** | **+** | ± (Quasar + Tailwind) | ± | ± | + | ± | + | ± (только native/PWA, PWA — [закрытый запрос](https://github.com/zauberzeug/nicegui/issues/246)) | ± |
| **Reflex** | ± | + (Radix) | ± | ? | + | ± | ± | − | ± |
| **Flet** | ± (`no_cdn`, есть открытая ошибка) | + | ± | ± | + | ± | + | **+** | ± |
| **Streamlit** | ± | ± | ± (`start_time`, `end_time`) | ± (`st.button(shortcut=…)`) | ± (`@st.fragment(run_every)`) | ± | ± | − | − |
| **Gradio** | − (Google Fonts, см. ниже) | ± | − | ? | + | ± | ± | − | − |
| **Panel** | ± (`inline`/`server`, офлайн не тестируется) | ± | ± | ? | + | ± | ± | − | ± |
| **Solara** | ± | ± | ± | ? | + | ± | ± | − | ± |
| **Textual (+serve)** | ± (Google Font в шаблоне) | − | − | + | + | ± | ± | − | ± |
| **Toga** | + (нативный) | ± (Android/iOS) | **−** (виджета нет) | ± | ± | − | + (Briefcase) | **+** | ? |
| **Kivy/KivyMD** | + | ± | ± | ± | ± | − | + (buildozer) | + | ± |
| **PySide6/QML** | + | − (нативные окна, не браузер) | **+** | + | + | − | ± (PyInstaller, 120+ МБ) | ± (технопревью) | **+** |
| **pywebview** | + | − | зависит от веб-части | зависит | зависит | − | + | ± (Android заявлен) | зависит |
| **FastHTML** | **−** (по умолчанию всё с jsDelivr) | своя вёрстка | своя | своя | + (SSE, WebSocket) | своя | ± | − | своя |
| **FastAPI/Starlette + Jinja2 + htmx** | **+** (всё можно положить в `static/`) | своя вёрстка | + (Range есть) | своя | + (SSE) | своя | ± | − | своя |
| **Litestar + HTMX-плагин** | + | своя | + | своя | ± | своя | ± | − | своя |
| **Shiny for Python** | ± | + (Bootstrap) | − | − | + | ± | ± | − | − |
| **Dash** | + (`serve_locally=True`) | ± | − | − | ± | ± | ± | − | ± |
| **Rio** | + (шрифты Roboto в wheel) | + (заявлено) | ± (есть `media_player`) | + (`key_event_listener`) | + | ± | ± | − | ? |

---

### 3. По каждому фреймворку

#### NiceGUI 4.0.0 (Python + FastAPI/Starlette + Vue/Quasar)

- **Офлайн.** Я просканировал wheel. Quasar, socket.io и Tailwind лежат локально, в `static/` есть 15 шрифтов woff2 (`quasar.umd.prod.js` 604 КБ, `tailwindcss.min.js` 287 КБ). Внешние адреса найдены только в двух необязательных элементах: `ui.aggrid` (jsDelivr) и `ui.plotly`. Шаблон страницы грузит все скрипты со своего пути `/_nicegui/…/static/`. Режим On Air (внешний ретранслятор) использует CDN, но он не нужен ([Configuration & Deployment](https://nicegui.io/documentation/section_configuration_deployment)).
- **Работа с агентами.** В wheel лежит `nicegui/llms.md`, то есть справка для ИИ идёт в комплекте.
- **Стабильность API.** Это минус для разработки агентами. Релиз 4.0 удалил API, устаревшие в 3.x, и содержит «тихие» изменения поведения (имена props больше не мигрируют, httpx заменён на httpx2). Релиз 3.0 раньше изменил схему auto-index. См. [заметки о релизе 4.0.0](https://github.com/zauberzeug/nicegui/releases/tag/v4.0.0).
- **Минимальные браузеры в 4.0:** Chrome/Edge 121, Safari 17.2. Старые планшеты и телефоны, например iOS ниже 17.2, отпадут.
- **Аудио.** [`ui.audio`](https://nicegui.io/documentation/audio) умеет `play()`, `pause()`, `seek(seconds)`, события через `.on('play'|'pause'|'ended')`. Проигрыша ограниченного отрезка нет. Остановку по концу отрезка придётся делать через `timeupdate` на JS (`js_handler`), иначе каждое событие идёт круг через сервер.
- **Горячие клавиши.** [`ui.keyboard`](https://nicegui.io/documentation/keyboard): события keydown/keyup, модификаторы, по умолчанию игнорируются фокус на input/select/button/textarea. Это нужно учесть для карточек с кнопками. Событие идёт через сервер, на локальной сети это допустимо.
- **Ход задачи.** Через Socket.IO. Server-push идёт по умолчанию, `binding_refresh_interval=0.1 с`. Параметры [`ui.run`](https://nicegui.io/documentation/run): `reconnect_timeout=3.0`, `message_history_length=1000`.
- **Доступ по сети — важно.** `host` по умолчанию `'0.0.0.0'`, то есть программа слушает всю сеть. Это надо явно переопределить на `127.0.0.1` или добавить авторизацию. Встроенной авторизации страница не описывает, есть `storage_secret` и SessionMiddleware. HTTPS: `ssl_certfile`/`ssl_keyfile`.
- **Упаковка.** `nicegui-pack` (обёртка PyInstaller); режим `native=True` использует pywebview (нужен Chromium 121+ или WebKit 17.2+).
- **Известные проблемы.**
  - Закрытый [issue #6018](https://github.com/zauberzeug/nicegui/issues/6018): тихая перезагрузка страницы при каждом переподключении WebSocket после первого ack, на деле потеря состояния на телефоне при засыпании. Исправлено в 2026 (issue закрыт).
  - [#6170](https://github.com/zauberzeug/nicegui/issues/6170): навсегда висящее окно «Connection lost».
  - [#5760](https://github.com/zauberzeug/nicegui/issues/5760): «Message too large for WebSocket» на офлайн-ПК (причина не выяснена, разработчик подозревает повреждённую установку).
  - Утечки памяти исправлялись в 2026 ([#6022](https://github.com/zauberzeug/nicegui/issues/6022), [#5803](https://github.com/zauberzeug/nicegui/issues/5803)).
- **Длинные списки.** Весь интерфейс — дерево Python-объектов на сервере на каждого клиента. Десятки тысяч слов как отдельных элементов недопустимы (измерений не нашёл, вывод из архитектуры). [`ui.table`](https://nicegui.io/documentation/table) поддерживает пагинацию и prop `virtual-scroll`. Для стенограммы надо делать один HTML-блок или пользовательский компонент на Vue/JS ([custom Vue components](https://nicegui.io/documentation/section_configuration_deployment)).
- **Мобильные сборки.** Готовых нет. Есть только режим «браузер по сети» и PWA как запрос.

#### Reflex 0.11.0 (Python → React)

- **Установка.** Требует Python ≥3.11. Документация упоминает bun и требует Visual C++ на Windows, на Windows [рекомендуется WSL](https://reflex.dev/docs/getting-started/installation.md) (аннотация страницы; про Node там ничего нет). В заметках к предрелизу 0.9.9a1 сказано, что нужна Node ≥22.22.0 ([newreleases.io](https://newreleases.io/project/github/reflex-dev/reflex/release/v0.9.9a1)); для 0.11 я это не подтвердил.
- **Фронтенд.** Зависимости фронтенда скачиваются при `reflex init`. Это противоречит установке без Node и работе без сети, хотя `reflex export` собирает статический фронтенд ([self-hosting](https://reflex.dev/docs/hosting/self-hosting/)).
- **Большие таблицы.** [issue #6179](https://github.com/reflex-dev/reflex/issues/6179): подвисание при 5–10 тысячах строк. Мейнтейнер советует пагинацию или ag-grid из платного пакета reflex-enterprise.
- **Активные ошибки.** Открыты [#7768](https://github.com/reflex-dev/reflex/issues/7768) (потеря изменений состояния) и [#7771](https://github.com/reflex-dev/reflex/issues/7771) (утечка блокировок).
- **Версия 0.x** с массовыми выпусками, ломающие изменения вероятны.

#### Flet 1.1.2 (Flutter)

- **Офлайн.** По умолчанию Pyodide, CanvasKit и шрифты грузятся с CDN, отключается через `no_cdn`/`--no-cdn`/`FLET_WEB_NO_CDN` ([документация](https://flet.dev/docs/publish/web/static-website)). Открытый [issue #7071](https://github.com/flet-dev/flet/issues/7071): веб-клиент всегда грузит `rive_native.js` с jsDelivr даже при `no_cdn=True`. В wheel `flet-web` я нашёл ссылки на `www.gstatic.com/flutter-canvaskit` и `fonts.gstatic.com`.
- **Скачивание клиента.** [Issue #7104](https://github.com/flet-dev/flet/issues/7104) (закрыт): собранное через `pyinstaller` приложение на чистом Windows скачивало настольную оболочку с GitHub при первом запуске. Исправление обещано в 1.1.3; `flet pack` обходит проблему.
- **Рендеринг.** Flutter рисует интерфейс на одном холсте. Доступность в вебе [выключена по умолчанию](https://docs.flutter.dev/ui/accessibility/web-accessibility). Работу поиска по странице и выделения текста я не проверял, документация это не описывает. Для чтения стенограммы это риск.
- **Аудио.** [`flet-audio`](https://flet.dev/docs/services/audio): `play(position)`, `pause()`, `seek()`, `on_position_change`. Позиция обновляется примерно раз в секунду, что мало для подсветки слов. На Linux нужен полный клиент, автовоспроизведение не работает в Chrome/Edge.
- **Клавиши.** [`page.on_keyboard_event`](https://flet.dev/docs/cookbook/keyboard-shortcuts), страница ограничений для веба ничего не говорит. Открыты запросы на глобальные сочетания ([#2052](https://github.com/flet-dev/flet/issues/2052)).
- **Производительность.** Открытый [issue #5941](https://github.com/flet-dev/flet/issues/5941): сильные тормоза при большом ListView из пользовательских компонентов, регрессия относительно 0.28.
- **Мобильные сборки.** Заявлены (Android, iOS, desktop, web из одного кода). Я не проверял качество.

#### Streamlit 1.67.0

- **Модель.** Скрипт перезапускается целиком при действии. Частичный перезапуск есть через [`@st.fragment`, в том числе `run_every`](https://docs.streamlit.io/develop/concepts/architecture/fragments), но фрагменты не видят изменений значений без Session State.
- **Аудио.** [`st.audio`](https://docs.streamlit.io/develop/api-reference/media/st.audio) принимает `start_time` и `end_time`, то есть воспроизведение отрезка есть «из коробки». О событиях воспроизведения и перемотке страница молчит.
- **Горячие клавиши.** [`st.button(shortcut=…)`](https://docs.streamlit.io/develop/api-reference/widgets/st.button): буквы, цифры, F-клавиши, Enter/Esc/стрелки. Клавиши C и R зарезервированы, знаки препинания не поддерживаются.
- **Сеть.** `browser.gatherUsageStats` по умолчанию `true` ([config.toml](https://docs.streamlit.io/develop/api-reference/configuration/config.toml)), есть XSRF и CORS-защита. Шрифты темы по умолчанию — URL в настройках; пример в документации с Google Fonts. В бандле есть строка `fonts.gstatic.com/s/i/short-term/release/materialsymbolsrounded/` (загрузка не проверена).
- **Вес** 86,8 МБ (pyarrow, numpy, pydeck) избыточен для задачи.
- **Сотни карточек с формами.** Модель перезапуска для них неудобна (вывод из архитектуры, измерений не нашёл).
- **Мобильные проблемы.** [Обсуждение про обрывы WebSocket на мобильных в 1.38.0](https://discuss.streamlit.io/t/web-socket-disconnection-in-1-38-0-problematic-in-mobile/80854) и открытый [#17265](https://github.com/streamlit/streamlit/issues/17265).

#### Gradio 6.32.0

- **Офлайн.** В чанке фронтенда есть `fonts.googleapis.com/css2?family=IBM+Plex+Mono…Source+Sans+Pro`, также ссылки на cdn.bokeh.org, cdn.plot.ly, `gradio.s3…/share.html`. Старая [issue #1450](https://github.com/gradio-app/gradio/issues/1450) (закрыта 04.06.2026) описывает пустую страницу без интернета из-за Google Fonts, разработчики писали о загрузке шрифтов с Google CDN. Вывод по текущей версии не проверялся.
- **Аналитика.** `GRADIO_ANALYTICS_ENABLED` по умолчанию `True` ([environment-variables](https://www.gradio.app/guides/environment-variables)); `GRADIO_SERVER_NAME` по умолчанию 127.0.0.1.
- **Аудио.** [`gr.Audio`](https://www.gradio.app/docs/gradio/audio): события play/pause/stop/stream, параметр `playback_position`. Программной перемотки и отрезков в документации нет.
- Сделан для демо ML. Для кастомных экранов неудобен.

#### Panel 1.10.0

- Требует Python ≥3.12. На Bokeh-сервере. [Issue #7429](https://github.com/holoviz/panel/issues/7429) «Off-line / air-gapped Panel: implement testing» открыт. Есть открытые [#7550](https://github.com/holoviz/panel/issues/7550) (require.js не отдаётся офлайн при `config.inline=True`) и [#6161](https://github.com/holoviz/panel/issues/6161) (свой CDN). В wheel — ссылки на cdn.holoviz.org, Plotly и Bokeh CDN. Подходит для научных панелей, а не для текстового редактора.

#### Solara 1.66.3

- Построен на ipywidgets и Vuetify 2 (ipyvue, ipyvuetify). В коде `telemetry.py` есть отправка в Mixpanel (флаг `mixpanel_enable`, значение по умолчанию я не выяснял). `plain.html` ссылается на CDN (Vuetify 2.2.26, Vue 2.6.10, require.js). Присутствует `cdn_helper.py`. Тяжёлая цепочка зависимостей (71 пакет, ipykernel, debugpy, jedi). Открытый [issue #75](https://github.com/widgetti/solara/issues/75) про заблокированный CDN. Подходит слабо.

#### Textual и textual-serve

- Терминальный интерфейс. В браузере он показывается в эмуляции терминала. `app_index.html` ссылается на Google Fonts (Roboto Mono). Для телефона, аудио по клику на слово и карточек с клипами не подходит. textual-web заброшен (2024), textual-serve обновлялся в 11.2025, открыт [issue про зависание после аудио](https://github.com/Textualize/textual-serve/issues/22).

#### Toga/BeeWare 0.5.7

- Нативные виджеты. [Аудио/видео виджета нет](https://toga.beeware.org/en/stable/reference/api/index.html). Есть WebView. Виртуализация Table/DetailedList в документации не описана. Платформы: Windows, macOS, Linux, Android, iOS, Web (частично). Таблица виджетов показывает, что Table поддержан частично на Android, на iOS нет. Для описанной задачи слишком мало виджетов.

#### Kivy 2.3.2 и KivyMD 2.0.0

- Свой графический движок. Релизы редкие (см. таблицу A). Плюс: мобильные и настольные сборки. Минусы: не веб, чтение длинного текста, доступность и выделение текста для расшифровки неудобны (вывод из архитектуры, измерений не нашёл).

#### PySide6 6.12.0 (Qt/QML)

- Лучший вариант для настольной программы: [QMediaPlayer](https://doc.qt.io/qt-6/qmediaplayer.html) даёт `setPosition` (мс), `positionChanged`, `playbackRate`, `play`/`pause`. Системные ListView в QML виртуализируются (из знаний о Qt, в этом исследовании не проверялось).
- **Минусы.** Не даёт браузерного доступа с телефона. Вес Essentials 82 МБ + Addons 40,5 МБ только для Windows. Лицензия LGPL требует внимания при поставке (динамическая линковка). Android: [инструмент `pyside6-android-deploy`](https://doc.qt.io/qtforpython/deployment/deployment-pyside6-android-deploy.html) существует, в документации 6.7 назван Technical Preview; статус в 6.12 я не подтвердил (страница отдалась усечённой). iOS не нашёл.

#### pywebview 6.2.1

- Не фреймворк, а оболочка для окна с веб-страницей ([guide](https://pywebview.flowrl.com/guide/): Windows, macOS, Linux GTK/Qt, Android; об iOS ничего). Её уже использует NiceGUI в `native=True`. Подходит как настольная упаковка любого веб-варианта, но не решает адаптивность.

#### FastHTML 0.14.27

- Python-обёртка над Starlette с HTML-компонентами и htmx. В коде: htmx 2.0.7, `fasthtml.js`, surreal, css-scope-inline подключаются с `cdn.jsdelivr.net` **по умолчанию**; расширения (ws, sse) тоже с CDN ([core.py в wheel](https://pypi.org/project/python-fasthtml/); параметр `default_hdrs`/`def_hdrs(htmx, htmx4, surreal)` из [документации](https://www.fastht.ml/docs/api/core.html)). Для офлайна нужно положить файлы в `static/` и отключить заголовки по умолчанию.
- Есть WebSocket (`ws`), SSE (`EventStream`), `static_route`. Версия 0.x с частыми выпусками.
- В wheel лежит `skill.py`, то есть ориентация на ИИ-агентов; содержание не читал.
- Запасной слой над Starlette. Выигрыш невелик, зато добавляется зависимость от молодого API.

#### FastAPI/Starlette + Jinja2 + htmx (+ sse-starlette)

- **Офлайн.** Всё зависит от вас, но риск CDN исчезает, если положить `htmx.min.js` и CSS в `static/`.
- **htmx.** [4.0.0 вышел 28.08.2026](https://four.htmx.org/announcements/2026-08-28-htmx-4.0.0-is-released), ветка 2.x поддерживается бессрочно, 4.0 ставится под тегом `next`. Основное изменение — переход на `fetch()`, наследование атрибутов теперь явное, есть [миграционное руководство](https://four.htmx.org/migration-guide-htmx-2) и расширение `htmx-2-compat`. Для нового проекта разумно закрепить одну версию. Лицензия htmx в GitHub API значится как NOASSERTION, проверьте файл лицензии при поставке.
- **Аудио.** Starlette отдаёт файлы с Range. Отрезок: `currentTime` + `timeupdate` ([MDN](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Audio_and_video_delivery/Cross-browser_audio_basics)).
- **Длинная стенограмма.** CSS `content-visibility: auto` пропускает отрисовку невидимых блоков, при этом работают поиск по странице и доступность; Baseline 2024 ([MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/content-visibility)). Для блока абзацев без JS-виртуализации это сильное решение.
- **Ход задачи.** SSE (sse-starlette 3.7.0) или htmx-расширения.
- **Безопасность.** Адрес с секретом остаётся возможным. Обычный `http://192.168.x.x` не считается безопасным контекстом ([MDN](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts)), поэтому service worker/PWA без HTTPS недоступны.
- **Главный минус.** Адаптивную вёрстку, карточки и горячие клавиши придётся писать вручную. Это ровно то место, где текущий интерфейс получился плохим.

#### Litestar 2.25.0

- Есть [HTMX-плагин](https://docs.litestar.dev/latest/usage/htmx.html) (`HTMXRequest`, `HTMXTemplate`, заголовки ответа). Документация о собственной интеграции SSE/WebSocket с htmx молчит. Выгода перед Starlette невелика, сообщество меньше.

#### Shiny for Python 1.9.0

- Реактивная модель, Bootstrap. В bootstrap.min.css встречается Google Fonts (Open Sans) — вероятно, из тем; реальную загрузку не проверял. Аудиоэлемента и горячих клавиш в нём нет. [Обзорная страница](https://shiny.posit.co/py/docs/overview.html) не говорит про офлайн и прогресс (аннотация).

#### Dash 4.5.1

- [`serve_locally=True` по умолчанию, при отсутствии интернета — единственный вариант](https://dash.plotly.com/external-resources). Модель обратных вызовов, ориентирована на графики. Аудио и горячих клавиш из коробки нет.

#### Rio 0.12.4

- В wheel 75 файлов компонентов, среди них `media_player`, `key_event_listener`, `list_view`, `webview`, `pdf_viewer`, шрифты Roboto лежат в пакете. Стадия Beta. Для режима окна `extra "window"` требует `pyside6<=6.7.3` и Python <3.13. `fastapi<0.162` закреплён сверху. Документация переехала на rio-ui.dev. Подтверждения виртуализации списков не нашёл.

#### Остальные

- **Mesop** и **FastUI** заброшены (ссылки в таблице A). **Air** и **Eel** помечены archived на GitHub.
- **Taipy** (Python <3.13), **Hyperdiv**, **justpy**, **remi**, **PyWebIO** — неактивны или с устаревшими релизами.
- **marimo** — реактивный блокнот с режимом приложения ([README](https://github.com/marimo-team/marimo)), для сложных экранов с произвольной клавиатурой и аудио не предназначен.
- **Slint** — Python-привязка в Beta; тройная лицензия: [Royalty-free, GPLv3 или коммерческая](https://github.com/slint-ui/slint).
- **Dear PyGui**, **wxPython** — только настольные.
- **pytauri** требует Rust (или готовый `pytauri-wheel`), может [сочетаться с NiceGUI/FastAPI без Node.js](https://github.com/pytauri/pytauri). До 1.0, последний релиз 09.2025.
- **Django + htmx** (Django 6.1.3, django-htmx 1.29.0) — избыточно для одного пользователя на локальном SQLite.
- **Quart** требует Python ≥3.13.
- **trame**, **Vizro**, **Anvil app server** (AGPL) ориентированы на другое.

---

### 4. Три лучших кандидата

#### 1. NiceGUI 4.0 (с отдельным JS-компонентом для стенограммы)

**Почему.**
- Всё на Python, в wheel лежат все скрипты и шрифты для базового набора, то есть работает без CDN и без Node ([проверено сканом wheel](https://pypi.org/project/nicegui/4.0.0/)).
- Лучшая живость среди «серверных» UI: 21 релиз за 2026, всего 34 открытых issue, MIT.
- В комплекте `llms.md`.
- Готовы: прогресс в реальном времени (socket.io), `ui.keyboard`, `ui.audio` с `seek`, лёгкий переход на встроенное окно (`native=True`) и упаковку (`nicegui-pack`).
- Quasar даёт адаптивную сетку и готовые виджеты (диалоги, табы, таблицы).

**Явные минусы.**
- Состояние живёт на сервере в дереве объектов. Десятки тысяч слов отдельными элементами нельзя; потребуется свой компонент или блок HTML, и тогда часть кода всё равно будет на JS.
- Нестабильность API: 3.0 и 4.0 ломали совместимость, агенту придётся сверяться с версией.
- Обрывы WebSocket на телефонах давали потерю состояния (#6018 исправлен, #6170 и другие подобные случаи встречались), нужна проверка на реальном телефоне.
- `host='0.0.0.0'` по умолчанию, защиту нужно добавлять самим.
- Минимальные браузеры Chrome 121 / Safari 17.2.
- Аудио-отрезок и быстрые клавиши в карточках требуют JS-обработчиков, чтобы не гонять каждое событие через сервер.

#### 2. Starlette/FastAPI + Jinja2 + htmx (локально положенный) + небольшие модули JS

**Почему.**
- Минимум зависимостей (17 пакетов, 3,8 МБ), стабильные основы, полный контроль над офлайном.
- Range для аудио решает перемотку, SSE даёт ход обработки.
- `content-visibility: auto` покрывает длинный текст без своей виртуализации, `<audio>` + `timeupdate` — отрезки.
- Ближе всего к тому, что уже есть (серверный HTML и немного JS), с понятными агенту технологиями.
- FastHTML возможен как тонкий слой, но добавляет молодой API (0.x), и по умолчанию тянет скрипты с jsDelivr.

**Явные минусы.**
- Это не готовый UI-набор: адаптивность, карточки, диалоги, горячие клавиши, состояние формы придётся писать и поддерживать (причина нынешней «плохой» оценки интерфейса может повториться).
- htmx сейчас на развилке 2.x и 4.x с несовместимостями по умолчаниям, версию придётся закрепить.
- Нужно самим проектировать защиту доступа по сети.

#### 3. Flet 1.1 (условно)

**Почему.** Единственный из просмотренных, кто даёт один код для desktop, web и мобильных сборок и имеет аудио-сервис `flet-audio` с `seek`. Активный проект (48 выпусков за 2026), Apache-2.0, ленивые списки Flutter.

**Явные минусы.**
- Холст вместо DOM: нативный поиск, выделение и доступность в вебе не работают так же, как у HTML (не проверял, исходя из [документации Flutter](https://docs.flutter.dev/ui/accessibility/web-accessibility)), что важно для читаемой стенограммы.
- Полностью офлайн веб-клиент не подтверждён: открыт [#7071](https://github.com/flet-dev/flet/issues/7071), `flet-web` весит 30 МБ.
- Позиция аудио обновляется примерно раз в секунду, подсветка слов по времени будет грубой.
- Известны тормоза больших ListView ([#5941](https://github.com/flet-dev/flet/issues/5941)).
- До 1.0 были массовые ломающие изменения, 0.x → 1.x миграции.
- Пока не доказано, что это лучше, чем нативное окно PySide6 для настольной версии.

**Запасной вариант вне тройки: PySide6/QML.** Если требование «телефон через браузер» ослабнет, это самый надёжный выбор для аудио по миллисекундам и виртуализации длинных списков. Он требует 120+ МБ и не открывается в браузере.

---

### 5. Что точно не подходит и почему

| Вариант | Причина |
|---|---|
| **Mesop** | Официально не поддерживается с 30.09.2026 ([README](https://github.com/mesop-dev/mesop)) |
| **FastUI** | Проект неактивен, репозиторий в архиве ([README](https://github.com/pydantic/FastUI)) |
| **textual-web** | Последний релиз 30.08.2024 |
| **Textual и textual-serve** | Терминал; нет аудио, плохо для телефона; textual-serve не обновлялся с 11.2025 |
| **Gradio** | Сделан для ML-демо; в коде есть Google Fonts, аналитика включена по умолчанию, отрезков аудио нет, вес 75 МБ |
| **Streamlit** | Перезапуск скрипта на каждое действие, вес 86,8 МБ с pandas и pyarrow, телеметрия по умолчанию. Плюс: `st.audio(end_time)` и `st.button(shortcut)` |
| **Panel** | Python ≥3.12, офлайн-режим не тестируется ([#7429](https://github.com/holoviz/panel/issues/7429)), ориентирован на научные панели |
| **Solara** | Стек Jupyter (71 пакет), Vuetify 2, телеметрия Mixpanel в коде |
| **Reflex** | Нужны Bun/Node при сборке (условие «без Node у пользователя» нарушено), на Windows рекомендуют WSL, тормоза больших таблиц |
| **Toga** | Нет аудио-виджета, мало виджетов |
| **Kivy/KivyMD** | Не браузер; длинный текст и доступность слабые (вывод из архитектуры) |
| **Dash, Shiny** | Не заброшены и работают офлайн, но нет аудио и горячих клавиш, нужен собственный JS |
| **Rio** | Beta, в режиме окна закреплён `pyside6<=6.7.3` и Python <3.13, узкие версии FastAPI |
| **Taipy** | Python <3.13, ориентирован на другое |
| **Quart** | Требует Python ≥3.13 |
| **Django** | Избыточен для локальной программы с одним пользователем |
| **Air, Eel, Hyperdiv, justpy, remi, PyWebIO** | Archived или без релизов давно |

---

### 6. Что я не смог проверить

- Реальную скорость NiceGUI, Flet и Reflex на десятках тысяч слов и сотнях карточек: измерений в источниках не нашёл.
- Работу веб-клиента Flet с `no_cdn=True` без сети: по [#7071](https://github.com/flet-dev/flet/issues/7071) почти работает, но остаётся один внешний запрос.
- Загрузку шрифтов Streamlit, Gradio и Shiny в браузере: только факт наличия адресов в wheel.
- Требования Reflex 0.11 к версии Node: данные из предрелиза 0.9.9a1.
- Текущий статус Android-инструмента PySide6 в 6.12 (страница усечена).
- Вес «после установки» и поведение под Linux/macOS: измерял только сжатые wheel под Windows и Python 3.12.
- Страницы Flet (раздел адаптивности) и часть документации Qt не загрузились или отданы усечёнными.
