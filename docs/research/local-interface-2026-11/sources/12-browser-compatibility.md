# Источник 12. Совместимость шести браузеров

Отчёт поискового субагента (модель Sonnet), 22.11.2026. Приведён как получен; не перепроверен,
кроме оговорённого в [README.md](../README.md). Обращения «вы», «твой» — к заказчику поиска.

ОТЧЁТ: совместимость браузеров для локального веб-интерфейса (на 22.11.2026)

Пометки: [Д] — документация или первоисточник, [Ф] — форумы, блоги, вторичные сайты, [?] — мой вывод или знание без подтверждения в этом поиске. Часть справки Яндекса (yandex.ru/support) не открылась: сайт обрывал соединение. Поэтому по Яндексу многое взято из выдержек поиска, а не из полных страниц. Файлы не создавались.

1. ДВИЖКИ И ОТСТАВАНИЕ

- Firefox обычный: 160, выпущен 10.11.2026 [Д]. Бета 161 выйдет 24.11 [Д, whattrainisitnow].
- Firefox ESR: актуальная ветка 153 (для ESR-канала с 21.07.2026). ESR 140 ещё получает исправления безопасности и автоматически обновляется до 153 [Д, Mozilla Enterprise]. Дата окончания поддержки ESR 140 не найдена. В списке на whattrainisitnow также значится 115, зачем — не выяснено.
- Отставание ESR 153 от 160: семь релизов, примерно полгода [?]. В ESR 153 нет, по заметкам MDN к версиям 155, 158 и 160 [Д]:
  - распознавания речи Web Speech на устройстве (появилось в 160, только на компьютере, не на Android);
  - JPEG XL (158);
  - `attr()` для любых свойств (155);
  - прочей мелочи.
  Для твоего списка критичного нет.
- Firefox для Android: канала ESR я не нашёл, версия идёт вместе с обычным Firefox [?].
- Chrome: по постам Chrome Releases стабильная версия 158 (17.11.2026), 159 в раннем доступе. Тела постов не прочитаны, вывод по заголовкам и датам [Д/?]. По Ф-источнику, с 08.09.2026 Chrome перешёл на цикл в две недели; не проверено.
- Яндекс Браузер: актуальная сборка 26.11.x (Windows, 20.11.2026) [Ф, softexia и RuStore]. Движок:
  - в июле 2026 версии 26.6 работали на Chromium 150 [Ф, testmuai];
  - в списке User-Agent на whatmyuseragent встречаются Chrome/154 рядом с YaBrowser 26.11.1.119, но пара не подтверждена [Ф];
  - официальных заметок о версии Chromium нет.
  Вывод: отставание от Chrome порядка 3–5 релизов [?]. Ru.wikipedia пишет «на 2–3 релиза» [Ф]. Точную версию смотри в `yandex://about` или в `navigator.userAgent`. Токен `YaBrowser/` есть на всех платформах [Ф].
- Яндекс Браузер на iOS: движок WebKit от Apple, токена Chrome в User-Agent нет [Ф, testmuai]. Любой браузер на iOS ведёт себя как Safari, то есть это отдельная платформа поддержки.

2. ВОЗМОЖНОСТИ

Всё из этого списка есть в Chromium 120+ и Firefox 125+ (по памяти; номера версий не перепроверял) [?]:
- container queries, `:has()`, dialog, Popover, CSS nesting, `dvh`;
- Pointer Events и touch-action;
- EventSource;
- fetch со стримингом ответа.

Что проверено:
- View Transitions в Firefox: только в рамках одного документа, с 144 (14.10.2025), без «types» [Д, web.dev]. В ESR 153 есть.
- content-visibility: в Firefox с 125, у Chromium с 85 [?]. MDN: у `auto` скрытое содержимое остаётся доступным для поиска по странице [Д]. В Firefox поиск, по свидетельствам, работает [Ф]. Открыта ошибка: ссылки `#:~:text=` не подсвечивают текст внутри `auto` (Bugzilla 1952515) [Д]. Сам я это не проверял, лучше проверить на живой странице. Для 10⁴–10⁵ слов лучше разбивать стенограмму на блоки, а не полагаться на подсветку через text fragments.
- Запись по fetch с потоком (streaming upload): в Firefox только в Nightly [Д, MDN 158]. Не использовать.
- MediaRecorder:
  - Chrome и Яндекс пишут `audio/webm;codecs=opus`; mp4 только если есть кодировщик ОС [Ф];
  - Firefox пишет ogg/opus, webm с версии 63 [Ф]; `audio/mp4` в Firefox был отвергнут в баге 2021 года [Ф];
  - Яндекс на iOS (WebKit) — только mp4/AAC [?].
  Нужен перебор `isTypeSupported()` по списку.
- Web Speech (распознавание):
  - Chrome: облако Google, через `webkitSpeechRecognition` [Ф];
  - Firefox до 159: нет, ошибка network [Ф];
  - Firefox 160: на устройстве, только компьютер, на Android нет [Д];
  - Яндекс: данных не нашёл.
  На ESR 153 нет. Для диктовки не полагаться; встроенная диктовка клавиатуры телефона работает независимо.
- Автозапуск звука: во всех браузерах нужен жест пользователя для звука [?].
- Media Session, Range/206, mp3/m4a/ogg/webm: подробно не искал, по памяти ожидаются без проблем, кроме iOS [?]. Для Яндекса (кодеки AAC/MP3 в сборке) стоит проверить на живом устройстве.
- Установка PWA:
  - Firefox на компьютере: нет. Mozilla только обсуждает «web apps» как вкладку на панели задач, Windows первой [Ф, Mozilla Connect, 08.2025].
  - Firefox Android: не проверял.
  - Chrome и Яндекс: есть. Для установки обычно нужен secure context [?].

3. HTTP В ДОМАШНЕЙ СЕТИ

- Адрес `http://192.168.x.x` — не secure context [Д, MDN: доверенными считаются localhost, 127.0.0.0/8, ::1].
  - Отсутствует `navigator.mediaDevices`, значит запись с микрофона невозможна.
  - Также недоступны Async Clipboard и Service Worker (значит и PWA) [Д, MDN].
  - Проверка: `window.isSecureContext`.
  - Сам MediaRecorder без getUserMedia бесполезен, поэтому запись голоса на телефоне будет только после HTTPS или исключения.
- Chrome: флаг `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, список адресов, допускается порт [Д, chromium.org; Android: старые сведения Ф]. Есть предупреждение «unsupported flag» [Д]. Командная строка требует `--user-data-dir`.
- Chrome 154 (октябрь 2026): HTTPS по умолчанию с предупреждением, но частные адреса (192.168.x.x) и одноуровневые имена исключены [Д/Ф, Google Blog, Android Authority 21.10.2026]. Окончательное включение не подтверждено постом Google.
- Firefox на компьютере: `about:config` → `dom.securecontext.allowlist` (список хостов через запятую). Порты не поддерживаются [Ф, Bugzilla, Mozilla Connect]. Есть открытая ошибка: после добавления хоста перестали грузиться картинки [Ф].
- Firefox Android: `about:config` в release недоступен, только Beta и Nightly [Ф, Mozilla Support]. Значит исключения на телефоне нет без Beta.
- Яндекс на компьютере: как в Chromium есть флаг `browser://flags` [?]. Официально есть корпоративная политика `OverrideSecurityRestrictionsOnInsecureOrigin` [Д, Яндекс]. В описании сказано только про значок «небезопасно» в умной строке, про микрофон не сказано. На Android в Яндексе подтверждения нет.

4. ДОВЕРИЕ СОБСТВЕННОМУ ЦС

- Firefox на компьютере: свой список. Включать `security.enterprise_roots.enabled` (читает хранилище Windows) или импортировать ЦС в «Настройки → Сертификаты → Центры сертификации» [Д, Mozilla KB, текст страницы не открылся, цитирую по выдержке]. Статья говорит, что на Windows, macOS и Android Firefox по умолчанию использует сторонние ЦС из хранилища ОС. Как это совмещается со старыми сведениями «по умолчанию выключено» (Bugzilla 1314010, Cisco), не выяснил. Проверь значение в `about:config`.
- Firefox Android: скрытая настройка «Use third party CA certificates»: пять нажатий на логотип в «О Firefox» → Secret Settings (с версии 90) [Ф, AdGuard]. Сообщения о системном и пользовательском хранилище противоречат друг другу [Ф]. Нужно проверить установкой.
- Chrome: использует хранилище ОС, локальное доверие учитывается на Windows, при TCP добавляется и убирается доверие [Д, FAQ Chrome Root Store]. Для Android — по памяти работает через пользовательское хранилище [?].
- Яндекс:
  - Блокировки «Нейропротекта» происходят на самоподписанные сертификаты и сертификаты, поставленные программами (антивирусы и т. п.) [Д, справка Яндекса].
  - Официальной инструкции для добавления своего ЦС не нашёл.
  - В бета-справке сказано: можно отказаться от сайта или внести сертификат в список надёжных [Ф, вторично].
  - Русские корневые сертификаты, по информации mos.ru и ряда блогов, встроены в браузер [Ф].
  - Хранилище Windows используется в Chromium-браузерах [Ф, comss].
  - На Android не выяснено, видит ли Яндекс пользовательские ЦС.
  Итог: без проверки на твоём железе нельзя утверждать, что собственный ЦС сработает. Тест обязателен, на телефоне и на компьютере.

5. ПРИВАТНОСТЬ

Яндекс Браузер, что нашёл:
- Нейропротект полностью не отключается, только ограничивается [Д]. Переключатели: «Отправлять информацию о подозрительных файлах и сайтах в Яндекс» и «Проверять безопасность посещаемых сайтов и загружаемых файлов» [Д, справка Яндекса, через поиск].
- Проверка сайтов идёт по списку заражённых страниц (по данным Яндекса, более 80 000). Отправляется ли адрес или только хеш — не выяснил. Проверка файлов отправляет характеристики файла на сервер, только Windows [Д].
- Объяснение Яндекса 2015 года: при включённой статистике использования собираются в том числе адреса посещённых страниц [Ф, Roem.ru; страница не открылась, по выдержке]. Это старо и могло измениться.
- Ничего не нашёл про адреса домашней сети и 127.0.0.1: исключаются они или нет, неизвестно.
- Алиса AI, пересказ, переводчик: считать облачными, поскольку без сети не работают [Ф, косвенно]. Данные не найдены.
- Турбо Яндекс прекратил поддерживать в апреле 2025 [Ф, ru.wikipedia].
- [?] Рекомендация: в тестовом профиле выключить статистику, ассистента, защиту отправки, не держать секреты в URL.

Chrome: Standard Safe Browsing проверяет по 4-байтовым префиксам хешей через прокси OHTTP (Fastly), Enhanced отправляет больше [Д, Google Security Blog, 14.03.2024]. По частным адресам данных нет.
Firefox: проверка по локальной базе хешей, большая часть запросов не уходит на сервер [Ф, блог 2016, возможно устарело]. В заметках к ESR 153 упомянут Safe Browsing V5 [Д]. По LAN данных нет.

6. ИЗВЕСТНЫЕ ОСОБЕННОСТИ

- SSE: HTTP/1.1 даёт 6 соединений на хост во всех трёх, при нескольких вкладках лимит быстро выбирается [?]. Без HTTPS HTTP/2 в браузерах не работает, это одна из причин делать HTTPS.
- Firefox закрывает EventSource после пробуждения компьютера (Bugzilla 1251117, открыта) [Д]. Firefox Android: отключение WebSocket при смене приложения на 147/149 — WORKSFORME, Chrome при блокировке экрана тоже разрывает (Bugzilla 1944288) [Д]. Нужны автопереподключение, Last-Event-ID, пересинхронизация по visibilitychange и pageshow.
- Android, клавиатура: диктовка и Gboard дают keydown с keyCode 229 и `insertCompositionText` в input, а не обычные keydown/keyup [Д, Bugzilla 1875787, 1343451]. В Firefox Android был баг: события клавиш не приходят при назначенном oninput (Bugzilla 1609407, статус не проверен). Не вешать горячие клавиши на текстовые поля, работать с `beforeinput` и `input`.
- Блокировщики рекламы и расширения могут резать запросы и SSE; режима Турбо больше нет [Ф].

7. ИТОГОВАЯ ТАБЛИЦА (д — да, н — нет, о — с оговоркой)

Столбцы: Fx ESR / Chrome / Яндекс ПК / Fx Android / Яндекс Android / Chrome Android.

| Возможность | Fx ESR | Chrome | Я ПК | Fx And | Я And | Ch And |
|---|---|---|---|---|---|---|
| container, :has, nesting, dialog, popover, dvh | д | д | д | д | д | д |
| View Transitions | д (без types) | д | д | д [?] | д | д |
| content-visibility: auto | д | д | д | д [?] | д | д |
| SSE / fetch-stream ответа | д | д | д | д | д | д |
| Запись голоса (MediaRecorder) на http LAN | н | н | н | н | н | н |
| Запись при HTTPS, формат | ogg/opus | webm/opus | webm/opus | ogg/opus [?] | webm/opus [?] | webm/opus |
| Web Speech распознавание | н | о (облако) | ? | н | ? | о |
| Clipboard API на http LAN | н | н | н | н | н | н |
| PWA | н | д | д | о | д | д |
| Исключение из secure context | about:config | флаг | флаг/политика | н | ? | флаг [?] |
| Свой ЦС | о | д | о (проверить) | о | ? | д [?] |

На что нельзя полагаться (при поддержке шести браузеров):
- микрофон, Clipboard, PWA и любые secure-context-API на http в LAN: нужен HTTPS или исключения;
- Web Speech, на ESR 153 его нет;
- один формат записи: перебирать `isTypeSupported()` и принимать webm и ogg на сервере;
- fetch с потоком запроса (в Firefox только Nightly);
- keydown и горячие клавиши в текстовых полях на Android;
- непрерывность SSE на мобильных;
- поиск через text fragments в `content-visibility:auto` на Firefox;
- неподтверждённое поведение Яндекса с собственным ЦС и локальными адресами.

Минимальные версии, которые разумно объявить [?]: Firefox ESR 153 (допустимо 140), Firefox Android не ниже текущего −3 релиза; Chrome 120+; Яндекс Браузер с Chromium не ниже 120 (проверять по Chrome/ в User-Agent, номер сборки Яндекса, соответствующий Chromium 120, не нашёл). Яндекс на iOS — как Safari, отдельно, без гарантий.

ИСТОЧНИКИ
- Firefox ESR 153 заметки: https://www.firefox.com/firefox/153.0esr/releasenotes/ ; Enterprise 153: https://support.mozilla.org/kb/firefox-enterprise-153-release-notes
- Whattrainisitnow: https://whattrainisitnow.com/
- MDN Firefox 155/158/160: https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Releases/160 (также /158, /155)
- web.dev View Transitions: https://web.dev/blog/same-document-view-transitions-are-now-baseline-newly-available
- MDN content-visibility: https://developer.mozilla.org/en-US/docs/Web/CSS/content-visibility ; Bugzilla 1952515, 1761043, 1764512
- MDN Secure contexts: https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Secure_Contexts/features_restricted_to_secure_contexts
- Chromium insecure origins: https://chromium.org/Home/chromium-security/deprecating-powerful-features-on-insecure-origins ; https://groups.google.com/a/chromium.org/g/security-dev/c/LlwAeKmvHr4
- Chrome HTTPS по умолчанию: https://blog.google/security/https-by-defau/ ; https://androidauthority.com/google-chrome-https-security-warnings-default-3725570
- Firefox allowlist: https://connect.mozilla.org/t5/ideas/support-ports-in-dom-securecontext-allowlist/idi-p/116011 ; Bugzilla 1308951, 1918915
- Firefox Android about:config: https://support.mozilla.org/en-US/questions/1523679
- Firefox ЦС: https://support.mozilla.org/en-US/kb/setting-certificate-authorities-firefox ; Firefox Android secret setting: https://kb.adguard.com/en/android/solving-problems/firefox-certificates
- Chrome Root Store FAQ: https://chromium.googlesource.com/chromium/src/+/main/net/data/ssl/chrome_root_store/faq.md
- Яндекс, политика: https://yandex.ru/support/browser-corporate/policy/override-security-restrictions-on-insecure-origin.html
- Яндекс, сертификаты: https://yandex.com/support/browser/ru/security/untrusted-sertificates ; Нейропротект: https://yandex.ru/support/browser/ru/security/protect-panel ; https://yandex.ru/support/browser/security/virus-protection
- Яндекс, версии и движок: https://www.testmuai.com/latest-version/yandex-user-agents/ ; https://whatmyuseragent.com/browser/browser-yandex-browser ; https://ru.wikipedia.org/wiki/Яндекс_Браузер ; https://en.wikipedia.org/wiki/Yandex_Browser
- Roem.ru 2015: https://roem.ru/17-07-2015/200620/yandeksbot-hodit-po-privatnym-ssylkam/
- Chrome Releases: https://chromereleases.googleblog.com/2026/11/stable-channel-update-for-desktop_018223192.html
- Google Safe Browsing real-time: https://security.googleblog.com/2024/03/blog-post.html
- Bugzilla: 1251117, 1944288, 1386317, 1875787, 1609407, 1343451 (bugzilla.mozilla.org/show_bug.cgi?id=…)
