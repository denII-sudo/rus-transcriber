# Источник 06. Системная и сторонняя диктовка вместо своей записи

Отчёт поискового субагента (модель Sonnet), 22.11.2026. Приведён как получен; не перепроверен,
кроме оговорённого в [README.md](../README.md). Обращения «вы», «твой» — к заказчику поиска.

Метки: [Д] — документация или первоисточник, [Ф] — форумы и отзывы, [?] — мой вывод. Поиск шёл через веб-поисковик и выборочное чтение страниц. Многое по Gboard, Яндекс Клавиатуре и Samsung подтвердить не удалось, это отмечено «НЕ ПОДТВЕРЖДЕНО».

### 1. Системный голосовой ввод (русский)

**Gboard (Android).**
- [Д] Google называет «расширенный голосовой ввод» с обработкой на устройстве только для английского, французского, немецкого, итальянского, японского и испанского. Работает он на Pixel 6 и новее, русского в списке нет.
- [Д] Новая функция Rambler (Pixel 11) заявляет русский. Офлайн у неё ограничен, а полные возможности требуют сети.
- НЕ ПОДТВЕРЖДЕНО: офлайн-пакет Gboard для русского. Официального списка языков нет. Офлайн-распознавание появилось в 2019 году только на Pixel и только для американского английского. [Ф] Пользователи GrapheneOS не смогли получить офлайн для русского.
- [Ф] По умолчанию звук уходит в Google. Это следует из описания приложения в App Store и общего устройства сервиса. Подробностей про имена, термины и пунктуацию на русском не нашлось.

**Яндекс Клавиатура.** Справка описывает только включение микрофона. Где идёт распознавание, НЕ ПОДТВЕРЖДЕНО, политику конфиденциальности я не читал. **Samsung Keyboard:** по запросу ничего не нашлось.

**iOS.** [Д] На странице Apple Feature Availability русский (Россия) стоит и в «Dictation», и в «On-Device and Modeless Dictation». То есть диктовка на устройстве для русского заявлена. Остаётся проверить на конкретном iPhone в режиме без сети.

**Windows 11, Win+H.** [Д] Русский поддержан. Распознавание онлайн, на Azure Speech services, офлайн-режима нет. Локально работает только «Fluid dictation» на Copilot+ PC, и то лишь для английского. macOS Dictation и Linux я не проверял.

**Поведение в веб-полях.** Единого баг-репорта «Gboard + Chrome Android + textarea» не нашлось.
- [Ф] Известны дубли слов при диктовке на Pixel 6 (Android 12).
- [Д] Рассылка W3C editing 2018 года объясняет, что Gboard ведёт composition по каждому слову.
- [Д] Задача w3c/editing #511 (ноябрь 2025): на Android Chrome `getTargetRanges()` не ловит исходный диапазон при старте диктовки.
- [?] Практический вывод: слушать `input` и `change`, не завязываться на `composition*` и `beforeinput`. Обычный `textarea` надёжнее `contenteditable`.

### 2. Web Speech API

- [Д] Режим on-device (`processLocally`, `available()`, `install()`) — Chrome 135+, только Windows, Mac, Linux. Языковой пакет около 60 МБ. Android и WebView в первоначальных платформах не названы (blink-dev, Intent to Ship).
- [Д] В обсуждении blink-dev в мае 2026 сказано, что Android и ChromeOS on-device Web Speech не поддерживают. Таблица caniuse показывает Chrome для Android без поддержки `processLocally` (по этой таблице Firefox и десктопный Chrome поддерживают, но даты и версии в ней выглядят странно, я бы перепроверил).
- [Д] Значение по умолчанию у `processLocally` — `false`: браузер сам решает, отправлять ли звук на сервер. MDN не называет, какие браузеры что делают.
- [Ф] Chrome отправляет звук в облако Google, Safari — в Apple. Первоисточника именно по Chrome не нашёл.
- [Д] На устройстве без нужного пакета `start()` даёт ошибку `service-not-allowed`.
- Поддержка русского в on-device режиме НЕ ПОДТВЕРЖДЕНА, список языков не нашёл. Safari: `processLocally` нет.
- [?] На Android Chrome Web Speech фактически означает облако Google, значит для чувствительных данных не годится.

### 3. Сторонние офлайн-программы

- **FUTO Voice Input.** [Д] Русский поддержан. Работает на Whisper, Parakeet и asr4all, офлайн. Лицензия «FUTO Source First License 1.0», это не открытая лицензия. Работает как голосовой режим в сторонних клавиатурах (HeliBoard, FlorisBoard, AnySoftKeyboard, Unexpected Keyboard). Gboard и Samsung Keyboard не поддерживаются. Встроенный вариант есть в FUTO Keyboard, она платная.
- **Sayboard (Vosk), WhisperIME, Transcribro.** Есть в F-Droid, но про русский и качество подтверждения не нашёл.
- **sherpa-onnx.** Готовой клавиатуры с GigaAM не нашёл. Есть Android-приложение ProtocolVoice (GigaAM v3 e2e CTC int8, 305 МБ) с открытым кодом на GitHub, но это приложение для расшифровки, не клавиатура.
- **Handy.** [Д] Лицензия MIT, Windows/macOS/Linux, полностью локальный, Whisper и Parakeet V3. GigaAM в README не упомянут. [Ф] Есть русский форк Echo.
- **Whispering, superwhisper, nerd-dictation:** независимых данных по русскому не нашёл. Сравнение по русскому у superwhisper — только от конкурента (aidictation.com), ему доверять нельзя.

### 4. GigaAM на телефоне и в браузере

- [Д] Лицензия v3 — MIT (карточка Hugging Face и репозиторий GitHub). Размер около 220–240 млн параметров, архитектура Conformer. Варианты: `ctc`, `rnnt`, `e2e_ctc`, `e2e_rnnt` (последние две — с пунктуацией и нормализацией).
- [Д] WER у v3 по данным авторов: средний 9,2% (CTC) и 8,4% (RNNT). У Whisper на их наборах 25,1%.
- [Д] **Важно про лицензию:** старая модель GigaAM от 2024-10-24 в sherpa-onnx помечена «только некоммерческое использование». У v2 (2025-04-19) такого предупреждения нет. Проверь лицензию именно той сборки, которую будешь брать.
- [Д] В sherpa-onnx готовые модели: GigaAM v2 CTC (226 МБ int8) и v2 transducer (около 231 МБ). Готовой модели v3 в официальном списке нет.
- [Ф] Сторонние конвертации v3: ONNX int8, около 305 МБ. GGUF: Q4_K_M 184 МБ, Q8_0 274 МБ, F16 452 МБ.
- Скорость на телефонах среднего класса НЕ ПОДТВЕРЖДЕНА. Есть только самоотчёт проекта gigastt: RTF около 0,10 на десктопе.
- В браузере (onnxruntime-web, WebGPU или WASM) демо с GigaAM не нашёл. [?] Модель 225–305 МБ технически загрузить можно. Но готовых решений, потоковой обработки и замеров нет, пришлось бы писать самим.
- Готовых приложений с GigaAM на Android нашёл одно: ProtocolVoice (см. выше).

### 5. Как это делают подобные продукты

По этому вопросу содержательных данных не нашёл. Отзывов по инструментам разметки и правки стенограмм не получилось собрать.

### 6. Сравнение трёх путей

| | (а) Текстовое поле + диктовка | (б) Запись на телефоне, распознавание на домашнем компьютере | (в) Модель на телефоне |
|---|---|---|---|
| Качество на русском | Зависит от клавиатуры и телефона; на именах и терминах — плохо проверяемо | Лучшее по управляемости: GigaAM/Whisper уже есть в проекте | GigaAM v3 хорош, но на телефоне и в браузере не проверен |
| Приватность | Android: чаще всего облако Google. iOS: заявлено на устройстве. Win+H: облако | Полностью локально | Полностью локально |
| Сложность | Минимальная | Уже реализована | Высокая: нет готового браузерного решения |
| HTTPS | Клавиатурной диктовке не нужен | Нужен (микрофон в браузере требует безопасного контекста) | Нужен для микрофона, плюс загрузка 200–300 МБ |
| Риски | Облако, дубли слов, разное поведение на устройствах | Нужен HTTPS или туннель | Размер, память, скорость, лицензия модели |

[Д] Для микрофона в браузере нужна защищённая страница: HTTPS, `file://` или `localhost`. На обычном `http://` по LAN-адресу `navigator.mediaDevices` просто `undefined`. Клавиатурной диктовке это не нужно: клавиатура работает на уровне системы, вне страницы.

### Рекомендация

1. Основным оставить путь (б): своя запись и распознавание на домашнем компьютере. Он уже работает и данные не уходят в облако. Если не хватает HTTPS, решать это отдельно (например, локальный сертификат или туннель).
2. Путь (а) добавить как необязательное дополнение: обычное текстовое поле всегда доступно, а как человек диктует в него, решает он сам. В интерфейсе предупредить, что на Android диктовка Gboard обычно идёт через облако Google. Самое безопасное — iPhone с диктовкой на устройстве (русский заявлен Apple) или FUTO Voice Input с HeliBoard (русский, офлайн, но лицензия Source First).
3. Путь (в) пока не брать: нет браузерных демо и замеров на телефонах, а лицензии у разных сборок разные. Вернуться к нему можно, если появится готовая сборка GigaAM v3 в sherpa-onnx для Android с проверкой на нужных телефонах.
4. Не использовать Web Speech API в Chrome Android: по умолчанию это облако, а локальный режим там не поддерживается.

Перед решением стоит проверить на самих телефонах: есть ли русский офлайн в Gboard (Настройки > Система > Языки и ввод > Gboard > Голосовой ввод), работает ли диктовка iOS без сети, и куда ходит трафик Яндекс Клавиатуры.

### Источники

- Gboard Help, расширенный голосовой ввод: https://support.google.com/gboard/answer/11197787?hl=en
- Gboard Help, Rambler: https://support.google.com/gboard/answer/17468539?hl=en
- Apple, Feature Availability: https://www.apple.com/ios/feature-availability/
- Microsoft, голосовой ввод Windows 11: https://support.microsoft.com/en-us/windows/use-voice-typing-to-talk-instead-of-type-on-your-pc-fec94565-c4bd-329d-e59a-af033fa5689f
- MDN, SpeechRecognition.processLocally: https://developer.mozilla.org/docs/Web/API/SpeechRecognition/processLocally
- MDN, Using the Web Speech API: https://developer.mozilla.org/docs/Web/API/Web_Speech_API/Using_the_Web_Speech_API
- caniuse, processLocally: https://caniuse.com/mdn-api_speechrecognition_processlocally
- blink-dev, Intent to Ship: On-device Web Speech API: https://groups.google.com/a/chromium.org/g/blink-dev/c/VNOok2dbmHM
- blink-dev, On-Device Recognition Quality: https://groups.google.com/a/chromium.org/g/blink-dev/c/P8P-x7AnC6I
- MDN, getUserMedia: https://developer.mozilla.org/docs/Web/API/MediaDevices/getUserMedia
- Chrome Developers, Web Speech API: https://developer.chrome.com/blog/voice-driven-web-apps-introduction-to-the-web-speech-api
- FUTO Voice Input: https://github.com/futo-org/voice-input
- Handy: https://github.com/cjpais/Handy
- GigaAM, GitHub: https://github.com/salute-developers/GigaAM
- GigaAM v3, Hugging Face: https://huggingface.co/ai-sage/GigaAM-v3
- sherpa-onnx, NeMo CTC (русский): https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/nemo/russian.html
- sherpa-onnx, NeMo transducer: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html
- ProtocolVoice, модели: https://huggingface.co/protocolvoice/asr-models/blob/main/README.md
- GigaAM v3 GGUF (transcribe.cpp): https://huggingface.co/handy-computer/gigaam-v3-e2e-rnnt-gguf
- gigastt: https://github.com/ekhodzitsky/gigastt
- W3C editing, Gboard и composition: https://lists.w3.org/Archives/Public/public-editing-tf/2018Feb/0000.html
- GrapheneOS, офлайн-голоса Gboard: https://discuss.grapheneos.org/d/1487-gboard-doesnt-download-offline-voices/1
- Яндекс Клавиатура, голосовой ввод: https://yandex.ru/support/keyboard-android/ru/voice-input
