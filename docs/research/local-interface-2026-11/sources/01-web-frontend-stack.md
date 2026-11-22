# Источник 01. Веб-стек и библиотеки компонентов

Отчёт поискового субагента (модель Sonnet), 21.11.2026. Приведён как получен; не перепроверен,
кроме оговорённого в [README.md](../README.md). Обращения «вы», «твой» — к заказчику поиска.

## Сравнение веб-фронтенда для локального приложения расшифровки (исследование, ничего не менялось)

### 0. Как читать (ограничения источников)

- Версии и лицензии взяты из ответов `https://registry.npmjs.org/<пакет>/latest`. В таблицах ссылка «npm» ведёт на такой ответ. Ответы пересказаны инструментом чтения страниц, поэтому возможны неточности пересказа. Дат выпуска реестр не дал, так что «активность в 2026» подтверждена только там, где ниже указана дата из анонса или новости.
- Размер «unpacked» — это размер пакета на диске, а не размер в браузере. Размер min+gzip я не нашёл ни для одной библиотеки и не выдумываю. Исключение: у Datastar и htmx в читанных документах размера нет.
- Про то, как хорошо ИИ-агенты пишут код под каждую технологию, независимых измерений почти нет. Есть только академические бенчмарки по React, Vue и чистому HTML и самореклама авторов Svelte (см. раздел 5). Остальные оценки «для агентов» — мой вывод, помечен как «вывод».
- Многие из читанных страниц — агрегаторы, новостные пересказы и блоги. Это помечено ниже словами «пересказ» или «не проверено в первоисточнике».
- Я не читал первоисточник по многим пунктам: даты и активность Alpine, Unpoly, Lit, Bootstrap, Pico, Open Props, Quasar, Vuetify, Mantine, MUI не проверены, известны только текущие версии. Состояние Quasar (есть ли «Quasar 3») не подтверждено. Для Vuetify и Quasar не нашёл независимых свидетельств активности в 2026 году, кроме номеров версий.

### 1. Без сборки

| Вариант | Версия, лицензия, размер | Node/сборка | Без CDN | Реальное время | Особенности |
|---|---|---|---|---|---|
| **htmx** | 2.0.11, 0BSD, 0,9 МБ на диске ([npm](https://registry.npmjs.org/htmx.org/latest)). 4.0.0 вышел 28.08.2026 ([анонс](https://four.htmx.org/announcements/2026-08-28-htmx-4.0.0-is-released)) | Не нужен. Один файл `htmx.min.js`, его кладут в проект ([docs](https://htmx.org/docs/)) | Да. Docs советуют не использовать CDN в production ([docs](https://htmx.org/docs/)) | SSE — отдельным расширением ([docs](https://htmx.org/docs/)); в 4.x это `hx-sse` ([анонс](https://four.htmx.org/announcements/2026-08-28-htmx-4.0.0-is-released)) | Две линии. 2.x остаётся `latest` в npm, 4.0 лежит под тегом `next` до начала 2027; 2.x обещано поддерживать «бессрочно» (там же). Переход 2→4 ломает: наследование атрибутов стало явным, переименованы события и атрибуты (там же). Для аудио: `hx-preserve` сохраняет элемент при подмене ([docs](https://htmx.org/docs/)) |
| **Alpine.js** | 3.17.5, MIT, 0,7 МБ ([npm](https://registry.npmjs.org/alpinejs/latest)) | Не нужен | Да | Нет своего | Лёгкая интерактивность в разметке. Не заменяет приложение с большим клиентским состоянием (вывод) |
| **Datastar** | На странице «Getting started» CDN-ссылки указывают 1.0.5 ([docs](https://data-star.dev/guide/getting_started)). Тег `latest` старого npm-пакета `@starfederation/datastar` показывает 1.0.0-beta.11 ([npm](https://registry.npmjs.org/@starfederation/datastar/latest)), то есть пакет устарел или переименован. Лицензия на прочитанной странице не указана; MIT — только по реестру старого пакета | Не нужен (подключается тегом `script`, [docs](https://data-star.dev/guide/getting_started)) | Да (подключается тегом `script`) | SSE — основной транспорт; сервер шлёт `datastar-patch-elements` и `datastar-patch-signals` ([docs](https://data-star.dev/guide/getting_started), [поиск](https://github.com/starfederation/datastar/releases)) | Есть Python-SDK (`datastar_py`) ([docs](https://data-star.dev/guide/getting_started)). Релиз 1.0 уже включал ломающие изменения (двоеточие вместо дефиса в `data-on:click`; DELETE без тела) — пересказ в результатах поиска. Мало времени «в бою» — вывод |
| **Unpoly** | 3.15.0, MIT, 7,7 МБ на диске (в основном исходники) ([npm](https://registry.npmjs.org/unpoly/latest)) | Не нужен | Да | Не проверял | `[up-keep]` сохраняет `<audio>` при частичных обновлениях; медиа не должно быть прямым потомком `body` ([docs](https://unpoly.com/up-keep)). Активность и размер не проверены |
| **Lit** (веб-компоненты) | 3.3.3, BSD-3, 0,1 МБ ([npm](https://registry.npmjs.org/lit/latest)) | Можно без сборки через ES-модули и import map; для декораторов и TypeScript нужна сборка (вывод) | Да, при локальной копии | Не встроено | Библиотека, а не набор экранов; всё остальное придётся писать. Для агентов — слабее, чем React/Svelte (вывод) |
| **Preact + htm** | Preact 11.2.0, MIT, 0,6 МБ ([npm](https://registry.npmjs.org/preact/latest)) | Работает без сборки; docs описывают htm и import map ([docs](https://preactjs.com/guide/v10/getting-started)) | Да, при локальной копии (пример docs берёт модули с esm.sh — это CDN, их нужно вендорить) | Не встроено | Компонентная модель без сборщика. Страница в docs про v10; номер 11.2.0 взят из реестра |
| **Vue 3 без сборки** | Vue 3.5.46, MIT, 2,6 МБ ([npm](https://registry.npmjs.org/vue/latest)) | Возможен ES-модульный режим с компилятором шаблонов в браузере | Да, при локальной копии | Не встроено | Страницу quick-start прочитать не удалось (домен не открылся), подробностей про режим без сборки подтвердить не могу. Отдельно: в статье WebCompass Vue «consistently challenging» для моделей ([аннотация](https://arxiv.org/abs/2604.18224)) |

### 2. Со сборкой

Общее: Vite 8.3.8 требует Node `^20.19.0 || >=22.12.0` ([npm](https://registry.npmjs.org/vite/latest)). Vite 8 стал стабильным 12.03.2026 (пересказ в [поиске](https://vite.dev/blog/announcing-vite8)). Сборка даёт статические файлы. Для установки у пользователя Node не нужен, если собранный `dist` лежит в репозитории или релизе (вывод).

| Вариант | Версия, лицензия | Что важно |
|---|---|---|
| **Svelte 5** | 5.59.1, MIT, 3,1 МБ ([npm](https://registry.npmjs.org/svelte/latest)) | Компилятор, в браузер уходит мало кода. Есть официальные llms.txt и MCP-сервер с проверкой кода для агентов ([svelte.dev/docs/ai](https://svelte.dev/docs/ai)) |
| **SvelteKit** | 3.4.0, MIT; требует vite ^8.0.12, svelte ^5.57.1, typescript ^6 ([npm](https://registry.npmjs.org/@sveltejs/kit/latest)) | Статика через adapter-static (4.0.1, [npm](https://registry.npmjs.org/@sveltejs/adapter-static/latest)): нужен `prerender`, для одностраничного режима — `fallback` ([docs](https://svelte.dev/docs/kit/adapter-static)). Мажорные версии часто: SvelteKit 3 ещё летом 2026 был в preview с множеством ломающих правок (пересказ [releases.sh](https://releases.sh/collections/frontend-frameworks/digest/2026-07-06)). Для нашего случая SvelteKit избыточен: достаточно Vite + Svelte без SvelteKit (вывод) |
| **Vue 3 + Quasar** | Quasar 2.35.6, MIT, 14 МБ ([npm](https://registry.npmjs.org/quasar/latest)) | Полный каркас приложения. «Quasar 3» в поиске не найден; статус не подтверждён |
| **Vue 3 + Vuetify** | 4.3.4, MIT, 78 МБ на диске; peer: vue ^3.5 ([npm](https://registry.npmjs.org/vuetify/latest)) | Material-стиль. Для сравнения: PrimeVue 5.0.5 в реестре имеет лицензию «SEE LICENSE IN LICENSE.md» и зависимость `@primeui/license-manager` ([npm](https://registry.npmjs.org/primevue/latest)), тогда как файл `LICENSE.md` на GitHub — MIT ([GitHub](https://github.com/primefaces/primevue/blob/master/LICENSE.md)). Расхождение не разрешил; до проверки PrimeVue не брать |
| **React** | 19.3.1, MIT ([npm](https://registry.npmjs.org/react/latest)) | Больше всего обучающих данных (довод авторов [Web-Bench](https://arxiv.org/pdf/2505.07473), их мнение) |
| **Mantine** | 9.8.2, MIT, 10,7 МБ; peer: react ^19.2 ([npm](https://registry.npmjs.org/@mantine/core/latest)) | Полный набор компонентов, хуки; адаптивные свойства и хуки (хуки по моим знаниям, не проверял в docs) |
| **MUI** | 9.5.0, MIT, 8,5 МБ ([npm](https://registry.npmjs.org/@mui/material/latest)) | Material-стиль, объёмный |
| **shadcn/ui** | Не нашёл отдельной версии: это не пакет, а копируемый в проект код поверх Tailwind и Radix (по моим знаниям, не проверял) | Для Svelte аналог — shadcn-svelte на основе Bits UI 2.22.0, MIT, peer svelte ^5.33 ([npm](https://registry.npmjs.org/bits-ui/latest)) |
| **Skeleton (Svelte)** | 5.0.6, MIT; peer tailwindcss ^4 ([npm](https://registry.npmjs.org/@skeletonlabs/skeleton/latest)) | Дизайн-система под Tailwind. Про активность не проверял |
| **SolidJS** | 2.0.3, MIT ([npm](https://registry.npmjs.org/solid-js/latest)); 2.0 признан стабильным 16.11.2026, после 23 RC; 1.x остаётся отдельной веткой (пересказ [AlternativeTo](https://alternativeto.net/news/2026/11/solid-2-0-reaches-stable-after-five-years-rebuilding-async-into-its-reactive-core/)) | Ломающие изменения (асинхронность в ядре, переименования). Сведения про SolidStart противоречат друг другу ([поиск](https://www.infoq.com/news/2026/09/solid-start-v2/)). Экосистема компонентов меньше (вывод) |
| **Ionic** | `@ionic/core` 9.0.13, MIT, 21 МБ ([npm](https://registry.npmjs.org/@ionic/core/latest)); Ionic 9 выпущен 19.08.2026 ([блог](https://ionic.io/blog/?p=6843)) | Ориентирован на мобильные приложения и Capacitor; для компьютерного экрана вторичен (вывод) |

### 3. Библиотеки компонентов и стили

| Вариант | Версия, лицензия | Адаптивность и особенности |
|---|---|---|
| **Web Awesome Free** | `@awesome.me/webawesome` 3.15.0, MIT, 21 МБ ([npm](https://registry.npmjs.org/@awesome.me/webawesome/latest)). Core 3.10.0 от июня 2026 ([блог Font Awesome](https://blog.fontawesome.com/how-does-web-awesome-stack-up-against-shoelace/)) | Веб-компоненты работают в любом фреймворке и без сборки. Есть `wa-page`, `wa-scroller`, утилиты вёрстки `wa-stack`, `wa-cluster`, `wa-split` (там же). Свой хостинг: пакет или `npm pack`, папка `dist-cdn`, `setBasePath()` ([docs](https://webawesome.com/docs/)). Pro (платно) добавляет combobox, выбор даты, графики; цена в источнике не указана (там же). Заявление о доступности общее, уровень WCAG не назван ([страница](https://webawesome.com/docs/resources/accessibility)) |
| **Shoelace** | 2.20.1, MIT ([npm](https://registry.npmjs.org/@shoelace-style/shoelace/latest)) | «Sunset»: последний релиз март 2025, новых версий не будет ([блог Font Awesome](https://blog.fontawesome.com/how-does-web-awesome-stack-up-against-shoelace/)). Не брать для нового проекта |
| **Material Web** | 2.5.0, Apache-2.0 ([npm](https://registry.npmjs.org/@material/web/latest)) | Режим поддержки с 10.06.2024: новых компонентов не планируется, команду Google перевели на другое; в июле 2025 мейнтейнеры сказали, что проект «ещё поддерживается» ([обсуждение](https://github.com/material-components/material-web/discussions/5642)). Не брать |
| **Pico CSS** | 2.1.1, MIT, 19,6 МБ ([npm](https://registry.npmjs.org/@picocss/pico/latest)) | CSS без JS, семантическая разметка, подходит под htmx. Для сложных виджетов (карточки, модальные панели с фокусом) компонентов нет (вывод) |
| **Open Props** | 1.7.23, MIT ([npm](https://registry.npmjs.org/open-props/latest)) | Только переменные CSS, не компоненты |
| **Tailwind** | 4.3.5, MIT ([npm](https://registry.npmjs.org/tailwindcss/latest)) | Через npm нужен Node; есть автономный исполняемый файл без Node ([docs](https://tailwindcss.com/docs/installation/tailwind-cli)). Состав релизов по платформам не проверял |
| **Bootstrap 5** | 5.3.9, MIT ([npm](https://registry.npmjs.org/bootstrap/latest)) | Сетка и готовые адаптивные классы, подходит как «без сборки» вариант (вывод). Дату выхода и активность не проверял |
| **Framework7** | 9.3.0, MIT ([npm](https://registry.npmjs.org/framework7/latest)); одна из публикаций называет 30.10.2026 датой выхода 9.3.0 ([пересказ](https://forum.framework7.io/t/amazing-job-framework-maintained-ios19/22889)) | Стиль нативного мобильного приложения (iOS 26 и Material). На экран ноутбука не рассчитан (вывод) |
| **Konsta UI** | 5.5.0, MIT; зависит от tailwind-merge ([npm](https://registry.npmjs.org/konsta/latest)); сайт указывает выход 5.5.0 28.09.2026 ([konstaui.com](https://konstaui.com)) | Работает внутри родительского фреймворка (Ionic, Framework7), сам он не каркас. Для нас второстепенный |

### 4. PWA и доступ по http в домашней сети

- **Secure context.** Надёжным считается `https`, `localhost`, `127.0.0.0/8`, `::1`, `.localhost`. Адрес вида `http://192.168.x.x` — НЕ secure context ([MDN](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts)). Сервис-воркеры в нём не регистрируются.
- **Установка.** Для Chromium нужны HTTPS (или localhost), манифест с `name`, `icons` 192 и 512 px, `start_url`, `display`. Сервис-воркер для самой установки не обязателен. Firefox на компьютере PWA не устанавливает ([MDN](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Making_PWAs_installable)).
- **iOS.** С iOS 26 любой сайт можно добавить на экран «Домой» как веб-приложение, даже без манифеста; манифест, если есть, используется (пересказ [firt.dev](https://firt.dev/notes/pwa-ios/) и результатов поиска). Требует ли iOS HTTPS для установки или сервис-воркера, я не подтвердил.
- **Микрофон.** `getUserMedia` на iOS поддерживается с 13.0 ([firt.dev](https://firt.dev/notes/pwa-ios/)). То, что он требует secure context, — по общим знаниям: на странице MDN я прочитал только про сервис-воркеры. Если голосовой ввод понадобится, по http в локальной сети он не заработает (вывод).
- **Как получить HTTPS в домашней сети.**
  - Локальный центр сертификации (mkcert): на iPhone корневой сертификат нужно установить и вручную включить «полное доверие» в «Основные - Об этом устройстве - Доверие сертификатов» (результаты [поиска](https://dzone.com/articles/ios-pwa-offline-caching)).
  - Tailscale: `tailscale cert` выдаёт сертификат Let's Encrypt для имени `*.ts.net`; имена машин попадают в публичные журналы Certificate Transparency; сертификат живёт 90 дней, продлевать вручную ([Tailscale](https://tailscale.com/kb/1153/enabling-https)).
  - Флаг Chrome `--unsafely-treat-insecure-origin-as-secure` годится для разработки; на Android он требует root или режима разработчика ([пересказ в поиске](https://chromium.org/Home/chromium-security/deprecating-powerful-features-on-insecure-origins)).
- **Кэш офлайн.** Для приложения, чьи данные живут на локальном сервере, офлайн-кэш почти бесполезен: без сервера нет ни записей, ни аудио (вывод). Реальная польза PWA — значок и полноэкранный режим.
- **SSE.** По HTTP/1.1 браузер держит максимум около 6 соединений на домен на весь браузер, общих для вкладок; по HTTP/2 лимит около 100 потоков ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/EventSource)). Значит, нужен один SSE-поток на вкладку; HTTP/2 требует TLS. Автопереподключение и `Last-Event-ID` — часть спецификации (там же).
- **Аудио на iPhone.** Для воспроизведения и перемотки сервер должен отвечать на `Range` статусом 206 с корректным `Content-Range` ([WebKit](https://bugs.webkit.org/show_bug.cgi?id=229413), результаты поиска). Это требование к Python-серверу, не к фронтенду.

### 5. Готовые компоненты под задачу

| Задача | Вариант | Данные |
|---|---|---|
| Дорожка и регионы | **wavesurfer.js** | 8.1.0, BSD-3, без runtime-зависимостей ([npm](https://registry.npmjs.org/wavesurfer.js/latest)). Плагины: Regions, Timeline, Minimap, Envelope, Record, Spectrogram, Hover. Для длинных файлов декодирование в браузере может падать по памяти; нужны заранее подготовленные пики (генерирует `audiowaveform`), только с ними работает потоковое воспроизведение ([README](https://github.com/katspaugh/wavesurfer.js)). Страница документации частично про v7. Про сенсорное управление в прочитанном ничего нет |
| Дорожка и регионы | **peaks.js** (BBC) | 4.0.0, LGPL-3.0, зависит от eventemitter3 ([npm](https://registry.npmjs.org/peaks.js/latest)). Данные из `audiowaveform` (`.dat` или `.json`); мышь, касание, колесо, клавиатура; сегменты и точки; README говорит, что разработка перенесена на Codeberg ([GitHub](https://github.com/bbc/peaks.js)). LGPL требует аккуратности при поставке (вывод, юридически не проверял) |
| Виртуальная прокрутка | **TanStack Virtual** | 3.19.11, MIT; адаптеры Vue, Svelte, Angular, Solid, Lit ([npm](https://registry.npmjs.org/@tanstack/virtual-core/latest), [docs](https://tanstack.com/virtual/latest/docs/framework)). Переменная высота — через измерение после показа (там же) |
| Виртуальная прокрутка | **virtua** | 0.55.0 (пока 0.x), MIT; React, Vue, Solid, Svelte, Angular ([npm](https://registry.npmjs.org/virtua/latest)) |
| Без библиотеки | **CSS `content-visibility: auto`** | Baseline 2024 (с сентября 2024). Скрытое остаётся в DOM, поиск по странице, табуляция и выделение работают; нужно `contain-intrinsic-size` ([MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/content-visibility)). Для стенограммы на десятки тысяч слов это проще, чем настоящая виртуализация, и сохраняет Ctrl+F (вывод). Группировать слова в абзацы или реплики — делает сервер |
| Diff | **jsdiff** | 9.0.0, BSD-3 ([npm](https://registry.npmjs.org/diff/latest)); `diffWords` и `diffWordsWithSpace`, опционально `Intl.Segmenter` ([README, пересказ](https://github.com/kpdecker/jsdiff)) |
| Diff | **diff-match-patch** | 1.0.5, Apache-2.0 ([npm](https://registry.npmjs.org/diff-match-patch/latest)); есть «семантическая очистка», но проект малоактивен по стороннему и недатированному источнику (там же). Не брать без причины |
| Diff на сервере | Python `difflib` | Моё предложение без источника: сравнение версий можно считать на сервере и отдавать готовую разметку, тогда в браузере библиотека не нужна |
| Редактор стенограммы со словами | **react-transcript-editor** (BBC, Draft.js) | Помечен как «work in progress», 92 открытых вопроса на момент индексации; рабочего состояния в 2026 году не подтвердил ([npm](https://www.npmjs.com/package/react-transcript-editor)). Зависимость от Draft.js не проверял. Не брать |
| Клик по слову, воспроизведение отрезка, подсветка | своя разметка | Мой вывод: слово — элемент с `data-start/end`, один общий `<audio>`, остановка на конце отрезка. Готового решения под наш случай не нашёл |

### 6. Как агенты пишут код (доказательная база слабая)

- **Svelte.** Авторы Svelte и студия Mainmatter сообщают: модели учились на Svelte 4 и часто смешивают паттерны React и Svelte; типичные ошибки — `$effect` вместо `$derived`, доступ к состоянию через `.$`. Против этого есть официальный MCP, `svelte-autofixer`, llms.txt и «skills» ([Mainmatter](https://mainmatter.com/blog/2026/07/28/agentic-engineering-with-svelte/), [svelte.dev/docs/ai](https://svelte.dev/docs/ai)). Это самореклама, независимых цифр нет.
- **Бенчмарки.** Авторы Web-Bench считают, что у React больше всего обучающих данных; Vue и Svelte местами показывают лучшие результаты у некоторых моделей ([arXiv](https://arxiv.org/pdf/2505.07473), конкретных цифр не видел). WebCompass (2026): на React, Vue и чистом HTML/CSS/JS вариант без фреймворка стабильно даёт лучшие оценки в генерации и правке, у Vue «consistently challenging» ([поиск](https://arxiv.org/pdf/2604.18224), [аннотация](https://arxiv.org/abs/2604.18224)). DesignBench: модели заметно хуже во фреймворках, чем на чистом HTML/CSS ([arXiv](https://arxiv.org/abs/2506.06251)). Про htmx, Datastar, Solid, Quasar данных нет.
- **Стабильность API (факты).**
  - htmx: ломающая 4.0 при поддерживаемой 2.x ([анонс](https://four.htmx.org/announcements/2026-08-28-htmx-4.0.0-is-released)).
  - Datastar: ломающие правки при выходе 1.0 (поиск).
  - SvelteKit: версия 3 в 2026 году, Vite 8, TypeScript 6; вместе это заметное «шатание» инструментария за один год ([пересказ](https://releases.sh/collections/frontend-frameworks/digest/2026-07-06)).
  - Solid 2.0 с ломающими изменениями (AlternativeTo).
  - Material Web и Shoelace больше не развиваются.

### 7. Сводная таблица

| Вариант | Node/сборка | Работает без CDN | Адаптивность «из коробки» | Зрелость на конец 2026 | Подходит для нашей задачи |
|---|---|---|---|---|---|
| htmx 2 + Alpine | Нет | Да | Нет (даёт компонентная библиотека) | Высокая (2.x), 4.x в `next` | Для списка, истории, диффа: да. Для карточек с горячими клавишами и жестами: громоздко (вывод) |
| Datastar | Нет | Да | Нет | 1.0.x, молодой | Хорош для хода выполнения по SSE; риск по зрелости |
| Unpoly | Нет | Да | Нет | Версия 3.15, активность не проверена | Запасной вариант |
| Lit + Web Awesome | Нет | Да | Есть (`wa-page`, `wa-split`, утилиты) | Web Awesome 3.15 | Хорошо как слой компонентов |
| Preact + htm | Нет | Да (вендорить) | Нет | Preact 11.2 | Рабочий компромисс |
| Svelte 5 + Bits UI/Skeleton + Tailwind | Сборка в CI | Да | Через Tailwind | Высокая, но частые мажорные версии вокруг | Лучший по размеру и читаемости; ставку делать на MCP/autofixer |
| React + Mantine | Сборка в CI | Да | Есть (хуки и компоненты) | Версии 19.3 / 9.8 | Лучший по знакомству агентов (вывод) |
| Vue + Quasar/Vuetify | Сборка в CI | Да | Есть | Vuetify 4.3, Quasar 2.35 | Слабее для агентов по WebCompass |
| Solid 2 | Сборка | Да | Нет | Только что вышел | Не брать сейчас |
| Ionic / Framework7 / Konsta | Сборка | Да | Мобильный стиль | Активны (Ionic 9, F7 9.3, Konsta 5.5) | Не подходят для экрана ноутбука (вывод) |
| Shoelace / Material Web | — | — | — | Не развиваются | Не брать |

### 8. Три лучших набора

**1. Svelte 5 (Vite, без SvelteKit) + Bits UI или Skeleton + Tailwind + wavesurfer.js + `content-visibility` + diff на сервере или jsdiff**
- Плюсы:
  - Сложные экраны (карточки, горячие клавиши, жесты, общий аудиоплеер) пишутся в одном реактивном стиле без перерисовки страницы.
  - Готовый набор для агентов: официальный MCP и autofixer.
  - Результат — статические файлы; Node нужен только при сборке.
  - Tailwind можно собирать автономным исполняемым файлом.
- Минусы:
  - Агенты по умолчанию склонны писать код Svelte 4 (Mainmatter).
  - Вокруг много ломающих обновлений за год (SvelteKit 3, Vite 8, TypeScript 6); лучше зафиксировать версии.
  - Сборка в CI и коммит `dist` — отдельная дисциплина.
  - Менее распространён, чем React.

**2. React 19 + Mantine (или shadcn/ui) + TanStack Virtual (при необходимости) + wavesurfer.js + jsdiff**
- Плюсы: больше всего обучающих данных, готовые адаптивные компоненты и хуки, виртуализация с официальным адаптером.
- Минусы:
  - Тяжелее (Mantine 10,7 МБ на диске, MUI аналогично).
  - Экосистема ходит за мажорными версиями.
  - Больше шаблонного кода.
  - Тоже нужна сборка.

**3. Без сборки: Python-сервер (Jinja-шаблоны) + htmx 2 (SSE-расширение) + Alpine.js + Web Awesome Free (локальная копия) + wavesurfer.js (локальный ES-модуль) + `content-visibility`**
- Плюсы:
  - Нет Node ни у пользователя, ни у агентов.
  - Все файлы лежат в репозитории и читаются.
  - Хорошо подходит для списка записей, хода выполнения, истории версий и диффа (diff считает сервер).
  - Web Awesome даёт доступные адаптивные компоненты.
- Минусы:
  - Горячие клавиши, жесты и состояние плеера между подменами придётся держать в Alpine и своём JS; `hx-preserve` обязателен.
  - Два «мира» (разметка сервера и JS) — больше места для расхождений.
  - htmx 4 расходится с 2.x; нужно закрепить 2.x.
  - Размер Web Awesome 21 МБ на диске: надо копировать только `dist-cdn`.
  - Pro-компоненты платные (графики, combobox).

### 9. Что не подходит и почему

- **Shoelace:** больше не развивается ([Font Awesome](https://blog.fontawesome.com/how-does-web-awesome-stack-up-against-shoelace/)).
- **Material Web:** режим поддержки без новых компонентов ([GitHub](https://github.com/material-components/material-web/discussions/5642)).
- **peaks.js:** LGPL, зависит от `audiowaveform`, развитие перенесено на Codeberg; wavesurfer проще лицензионно (BSD-3).
- **react-transcript-editor:** «work in progress», активность не подтверждена.
- **diff-match-patch:** по сведениям, малоактивен; проще jsdiff.
- **Solid 2:** только что стал стабильным, ломающие изменения, неясна судьба SolidStart.
- **Ionic, Framework7, Konsta:** ориентированы на мобильное приложение, экран ноутбука — не их задача (вывод).
- **PrimeVue:** расхождение между лицензией в реестре и на GitHub; сначала выяснить.
- **Datastar как основной выбор:** молодой, ломающие правки при выходе 1.0, мало материалов для агентов (вывод). Но как транспорт SSE — интересен.
- **PWA с офлайн-кэшем:** смысла почти нет, пока данные только на сервере.

### 10. Что стоит проверить отдельно (я не подтвердил)

1. Можно ли отдавать HTTPS из Python-сервера при установке (локальный центр сертификации, mkcert, Caddy), и приемлем ли ручной шаг доверия на iPhone.
2. Нужен ли iOS HTTPS для установки на экран «Домой» и для сервис-воркера (я не нашёл прямого подтверждения).
3. Фактическую активность и даты выпусков для Alpine, Unpoly, Lit, Bootstrap, Pico, Quasar, Vuetify, Mantine, MUI, Skeleton и состав релизов автономного Tailwind CLI для Windows, Linux и macOS.
4. Поведение wavesurfer.js с готовыми пиками на часовой записи на телефоне и поддержку касания.
5. Размеры собранных бандлов (min+gzip) — прототипом.
