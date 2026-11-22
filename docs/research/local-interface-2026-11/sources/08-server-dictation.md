# Источник 08. Быстрая диктовка на сервере

Отчёт поискового субагента (модель Sonnet), 22.11.2026. Приведён как получен; не перепроверен,
кроме оговорённого в [README.md](../README.md). Обращения «вы», «твой» — к заказчику поиска.

## Диктовка по домашней сети: GigaAM и альтернативы (исследование на 22.11.2026)

Метки: [Д] — документация или первоисточник, [Ф] — форумы и отзывы, [?] — мой вывод.
Все скорости ниже — самоотчёты авторов на разном железе. Для вашего ноутбука их надо мерить самим.

### 1. GigaAM

- **Состав.** [Д] v2 и v3 (v3 вышла 11.2025). Для v3 есть ctc, rnnt, e2e_ctc и e2e_rnnt; e2e даёт пунктуацию и нормализацию текста. Лицензия MIT. В 2026 добавлены мультиязычные модели и пословные метки времени. [GitHub GigaAM](https://github.com/salute-developers/GigaAM)
- **Поток.** [Д] У авторов потокового режима нет, в README он не упомянут. Фрагмент `.transcribe` — до 25 с, длиннее — `transcribe_longform` с внешним VAD (pyannote).
- **Псевдопоток.** [Д] Это Silero VAD плюс распознавание законченных фраз. Так сделано в `onnx-asr`, где есть `load_vad("silero")`; авторы предупреждают, что параметры VAD придётся подбирать. [onnx-asr](https://pypi.org/project/onnx-asr/)
- **Сторонний потоковый RNN-T.** [Д/?] Проект gigastt (Rust, ONNX INT8) держит состояние декодера между кусками и отдаёт частичные результаты по WebSocket. Заявлено: RTF около 0,10, первая частичная фраза за 0,8–1,7 с, около 400 МБ ОЗУ на сессию. Замер сделан на Apple M1, не на x86. [gigastt](https://docs.rs/crate/gigastt/2.3.0/source/README.md). Это самодельный поток; проверенным решением от авторов GigaAM он не является.
- **Скорость на CPU.**
  - [Д] Таблица onnx-asr на Ryzen 9800X3D: GigaAM v3 CTC даёт 59× реального времени, RNN-T — 43×, Parakeet v3 — 36×. На слабом ARM Cortex-A53 — около 1,5×.
  - [Ф] В Habr-замере (источник пишет «июнь», год не указан) RTF на CPU равен 0,047 у GigaAM v3 ONNX и 1,02 у Whisper turbo int8, то есть Whisper не успевает за речью. Модель CPU там не названа. [Habr](https://habr.com/ru/articles/1042574/)
  - [?] На ноутбучном Core i5/i7 реалистичен RTF 0,05–0,15. Для фразы в 5 с это 0,3–0,7 с распознавания.
- **Дробление на куски.** [Ф] В том же замере 4-секундные куски по VAD почти вдвое ухудшили WER у Whisper (с 7,3 до 13,7%). Для GigaAM этой цифры нет; по замеру на YouTube, куски по VAD ухудшили его результат примерно на 1,4 п.п. относительно Whisper на полном окне. [?] Режьте по паузам не короче 0,5–0,7 с, а не по фиксированным 3–4 с.
- **Подсказки и словарь.**
  - В GigaAM их нет. [Д]
  - sherpa-onnx: hotwords работают только у transducer-моделей и только с `modified_beam_search`; CTC не поддерживается. Единицы — bpe или cjkchar, про кириллицу в документации ничего нет, нужна проверка. [sherpa hotwords](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html)
  - NeMo: есть CTC-WS (подходит и для CTC) и GPU-PB (CTC, RNN-T, TDT). [NeMo Word Boosting](https://docs.nvidia.com/nemo/speech/latest/asr/asr_customization/word_boosting.html)
  - gigastt: смещение жадного декодера к заданным написаниям (по умолчанию выключено). [lexicon.rs](https://docs.rs/crate/gigastt-core/2.3.0/source/src/lexicon.rs)
  - [?] В GigaAM через ONNX это придётся писать самим. Дешёвая замена — постобработка по словарю замен и нечёткое сопоставление с именами из карточки.

### 2. Альтернативы с настоящим потоком

- **T-one (Т-Банк).** [Д] Conformer на 71 млн параметров, CTC, куски по 300 мс, Apache 2.0. Заточена под телефонию. Заявленный WER: колл-центр 8,63 против 10,22 у GigaAM-RNNT v2; CommonVoice 5,32 против 2,68 у GigaAM. Скорость на CPU не указана. Есть KenLM-декодер. Пунктуации нет. [HF t-tech/T-one](https://huggingface.co/t-tech/T-one). В sherpa-onnx есть сборка от 08.09.2025 [[sherpa T-one](https://csukuangfj.github.io/sherpa/onnx/pretrained_models/online-ctc/t-one-ctc-models.html)]. В onnx-asr поток для неё пока не поддержан.
- **Vosk / sherpa streaming zipformer small ru.** [Д] Zipformer2 от alphacep, Apache 2.0. WER на CommonVoice 13,7 по текущей карточке (в старой версии было 19,3). Намного слабее GigaAM. [HF vosk-model-small-streaming-ru](https://huggingface.co/alphacep/vosk-model-small-streaming-ru). Плюс: это transducer, значит hotwords в sherpa работают.
- **Parakeet TDT v3.** [Д] 600 млн параметров, 25 европейских языков, включая русский, с пунктуацией. Лицензия CC-BY-4.0. Русского WER в моих источниках нет. На CPU около 36× реального времени (таблица onnx-asr). Своего потока у ONNX-версии нет. [onnx-asr](https://pypi.org/project/onnx-asr/)
- **Whisper-варианты.** [Д] WhisperLiveKit реализует потоковые политики SimulStreaming и LocalAgreement на faster-whisper. [Ф] Измеренной задержки на CPU для русского я не нашёл. По Habr-замеру turbo на CPU RTF около 1, то есть для потока на ноутбуке не годится. [WhisperLiveKit](https://pypi.org/project/whisperlivekit/)
- **Moonshine.** [Д] Русского языка нет: английский плюс арабский, китайский, японский, корейский, украинский, вьетнамский (на 09.2025). [arXiv 2509.02523](https://arxiv.org/abs/2509.02523)

### 3. Готовые серверы и протоколы

- **Wyoming.** [Д] JSONL плюс PCM. Потоковый обмен: `audio-chunk` → `transcript-start`, `transcript-chunk`, `transcript-stop`, у сервера есть флаг `supports_transcript_streaming`. [Wyoming](https://github.com/rhasspy/wyoming/)
- **Speaches.** [Д] Это бывший faster-whisper-server. `POST /v1/audio/transcriptions` с потоком через SSE; WebSocket `/v1/realtime` с `intent=transcription` заявлен. [Ф] В декабре 2025 был GitHub-issue о несовместимости realtime с клиентом Home Assistant. [Speaches](https://speaches-ai.github.io/speaches)
- **sherpa-onnx WebSocket.** [Д] Бинарный PCM на вход, JSON с частичными и окончательными результатами на выход, есть браузерная страница. [Docs](https://k2-fsa.github.io/sherpa/onnx/python/streaming-websocket-server.html)
- **[?] Что взять за образец.** Два уровня. Для файлов и коротких фраз — `/v1/audio/transcriptions` в стиле OpenAI: чужие клиенты его знают. Для потока — простой WebSocket: бинарный PCM16 16 кГц, JSON `{type: partial|final, text, ...}`, как у sherpa. Wyoming стоит добавлять только при интеграции с Home Assistant.

### 4. Клиент в браузере

- **Нужен HTTPS.** [Д] `getUserMedia` работает только в защищённом контексте; `http://192.168.x.x` не подходит, `localhost` подходит. [MDN](https://developer.mozilla.org/en-US/docs/Web/API/Navigator.mediaDevices). Флаг Chrome `unsafely-treat-insecure-origin-as-secure` хрупок, на Android нужен root или режим разработчика. [Chromium](https://chromium.org/Home/chromium-security/deprecating-powerful-features-on-insecure-origins). Рабочий путь: HTTPS через `tailscale cert` (имя из tailnet, доверенный сертификат) или свой корневой сертификат, установленный на телефоне. [Ф] [Tailscale на iPhone](https://flaviocopes.com/tailscale-dev-server-phone/). Про mkcert на iOS подтверждения не нашёл.
- **iOS.** [Ф] В установленном на экран «домой» веб-приложении `mediaDevices` бывает `undefined`; тестируйте в обычном Safari. [WebKit 180551](https://bugs.webkit.org/show_bug.cgi?id=180551)
- **MediaRecorder или AudioWorklet.**
  - [Ф] На iOS у AudioWorklet бывают сбои, а число сэмплов в секунду может не совпадать с заданным. Надёжнее MediaRecorder (в Safari это mp4/aac, не webm) и перекодирование на сервере (ffmpeg). [Apple forum](https://developer.apple.com/forums/thread/768347), [WebKit 294623](https://bugs.webkit.org/show_bug.cgi?id=294623)
  - [?] На десктопе и Android AudioWorklet с PCM по WebSocket даёт меньшую задержку и проще для потока.
- **Заблокированный экран.** [Ф] Захват микрофона обычно прекращается. Считайте диктовку «с включённым экраном».
- **Интерфейс.** [?] «Нажал и держи» подходит для 1–15 слов: конец речи известен, VAD не нужен, ошибки обрыва исключены. «Старт/стоп» нужен для длинной диктовки. Промежуточный текст показывайте серым и заменяйте окончательным по `final`.

### 5. Программы диктовки

- **Handy.** [Д/Ф] Silero VAD, Whisper и Parakeet v3, по умолчанию push-to-talk; для длинной речи удобнее переключатель. Отзывы в основном производные. [Handy README](https://raw.githubusercontent.com/cjpais/Handy/main/README.md)
- **VoiceInk и superwhisper.** [Д/Ф] Личный словарь, замены, глобальные горячие клавиши, режимы с ИИ-постобработкой. Задержки в отзывах — маркетинг конкурентов (200 мс – 1 с после отпускания). У superwhisper жалуются на Windows-версию. [Обзор](https://spokenly.app/blog/superwhisper-review)
- **Whispering, nerd-dictation, FUTO, Talon.** Поиск ничего не дал. По памяти: nerd-dictation — Vosk с нажатием клавиши; Talon — голосовое управление с командами. [?] Подтверждённых данных нет, не опирайтесь.
- **[?] Что перенять.** Push-to-talk для коротких фраз, VAD для длинных, словарь замен и личный словарь, пунктуация (e2e-модель), удаление слов-паразитов правилами. Показывать результат нужно как черновик, который человек правит.

### 6. Ресурсы

- **Память.** [Д] Одна INT8-модель GigaAM v3 около 225 МБ на диске, сессия около 400 МБ ОЗУ (gigastt). Холодный старт около 1 с (gigastt, заявлено). [?] На вашей машине холодный старт может быть заметно дольше из-за чтения диска; мерить.
- **[?] Схема.** Держите отдельный малый процесс диктовки с постоянно загруженной моделью (ONNX CPU, 1–2 потока), а тяжёлую расшифровку и разделение по говорящим оставьте в очереди с низким приоритетом (`BELOW_NORMAL` в Windows). Тогда диктовка не ждёт ночную обработку. Если окно разделения по говорящим занимает все ядра, диктовка получит задержку; ограничьте тяжёлой задаче число потоков.

### 7. Рекомендация

- **(1) Короткий ответ, 1–15 слов.** Кнопка «нажал и держи». Звук целиком уходит по HTTPS (WebSocket или POST), сервер перекодирует (ffmpeg), прогоняет GigaAM v3 e2e (CTC или RNN-T) и возвращает текст. Потока не нужно: распознаётся одна фраза. После этого: сопоставление имён из карточки с результатом (нечёткое, по словарю) и постобработка. Оценка задержки на ноутбучном CPU: 0,3–1 с после отпускания [?].
- **(2) Длинная диктовка.** Старт/стоп, PCM 16 кГц через AudioWorklet (на iOS — MediaRecorder). На сервере Silero VAD режет по паузам ≥0,5 с и отправляет фразы в GigaAM e2e; в браузере сначала серый черновик, потом окончательный. Задержка после паузы около 0,5–1,5 с [?], «по мере речи» между паузами не получится. Настоящий поток (T-one или gigastt) стоит рассматривать, только если эта схема окажется слишком медленной.

**Что измерить первым.** Для рекомендованного конвейера (VAD и GigaAM) нужны четыре замера:
1. RTF и холодный старт GigaAM v3 e2e CTC и RNN-T на вашем CPU (INT8 и FP32), с фоновой тяжёлой задачей и без неё.
2. WER и точность имён на 30–50 реальных коротких ответах (1–15 слов).
3. Качество и задержка VAD-нарезки на длинной речи (пороги паузы 0,3 / 0,5 / 0,8 с).
4. Работа микрофона на реальном телефоне через HTTPS (Safari и Chrome): формат, сбои AudioWorklet, поведение при блокировке экрана.

**Не найдено.** Русский WER для Parakeet v3; задержка Whisper-стриминга на русском CPU; задержка T-one на CPU; поддержка кириллицы в hotwords sherpa-onnx; надёжные данные про Whispering, nerd-dictation, FUTO и Talon; работа mkcert на iOS.
