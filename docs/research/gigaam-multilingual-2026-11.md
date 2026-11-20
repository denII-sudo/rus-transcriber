# GigaAM Multilingual и среды исполнения v3 на CPU, ноябрь 2026

Краткая заметка: что даёт GigaAM Multilingual проекту и как это проверять тестом скорости.
Это не сертификат точности: выводы о пользе на реальных записях делаются только по прогону
`bench.py` на машине владельца.

## Что это за модель

- Выпущена в июне 2026: `multilingual_ctc` (220M) и `multilingual_large_ctc` (600M),
  предобучение 2 млн ч на 70+ языках, дообучение 50 тыс. ч. Только CTC.
- Текст строчными, без пунктуации, числа словами, английские слова в русской речи —
  кириллицей («пинк», «синергия тв»). Поэтому заменить основной проход `v3_e2e_rnnt`
  она не может.
- В onnx-asr поддерживается с версии 0.12.0: `gigaam-multilingual-ctc`,
  `gigaam-multilingual-large-ctc`.

## Точность на русском (внешние замеры, одинаковая нормализация)

| Russian LibriSpeech, WER | v3 RNN-T | v3 CTC | ML large | ML 220M | v3 e2e RNN-T | Whisper turbo |
|---|---|---|---|---|---|---|
| onnx-asr `docs/comparison.md` | 4.39% | 4.72% | 5.55% | 8.43% | 7.60% | 10.10% |

Multilingual хуже v3 без пунктуации. Распространённое «точнее» — это сравнение с e2e-моделями,
у которых WER завышают пунктуация и цифры. Кандидат для проекта — третья модель голосования
вместо Whisper: на русском она точнее Whisper, быстрее и требует меньше памяти.

## Проверка в облачном контейнере 19.11.2026

Условия: Linux, 4 ядра Intel Xeon (AVX-512, VNNI, AMX), onnx-asr 0.12.0,
onnxruntime-openvino 1.24.1, OpenVINO 2025.4.1; публичные фрагменты Golos, 62 с.
**Скорость на этом железе не переносится на проверенные машины проекта:** Intel Core Ultra 5
125H (Meteor Lake: AVX2 и AVX-VNNI, без AVX-512 и AMX) и AMD Ryzen 5 6600H (Zen 3+: без
AVX-512 и VNNI). Здесь int8 ускоряют AVX-512 VNNI и AMX.

| Вариант | Скорость | Совпадение |
|---|---|---|
| RNN-T: ORT CPU / OpenVINO CPU / int8 | 28.0× / 34.5× / 40.5× | 100 / 100 / 97.6% |
| CTC: ORT CPU / OpenVINO CPU / int8 | 33.0× / 42.5× / 45.5× | 100 / 100 / 96.9% |
| ML 220M: fp32 / int8 | 25.1× / 32.8× | 78.5% к RNN-T |
| ML large: fp32 / OpenVINO CPU / int8 | 11.2× / 15.5× / 21.4–26.6× | 80.0% к RNN-T |

- Низкое совпадение Multilingual с RNN-T в основном вызвано оформлением (числа словами,
  английский кириллицей), а не ошибками распознавания.
- На этом отрывке ML large подтвердил RNN-T в 2 из 3 спорных слов и услышал 3 слова,
  которые RNN-T пропустил, а CTC услышал. Отрывок короткий и чистый — это проверка
  работоспособности, а не вывод о качестве.
- Тексты ML large в fp32, OpenVINO и int8 совпали.
- Разброс скорости между двумя прогонами одного варианта — около 10%.

## Найденная проблема и обход

ONNX Runtime 1.24 не загружает fp32-модели с весами в отдельном файле `*.onnx.data`
(ML large), если файл — символическая ссылка кэша Hugging Face: «External data path ...
escapes model directory». В `backends.py` такие модели (`EXTERNAL_DATA_MODELS`)
скачиваются обычными файлами в `cache/onnx-models/<имя>` и загружаются оттуда.
fp32-веса ML large занимают около 2,3 ГБ.

## Как проверить на своей машине

```powershell
.\.venv\Scripts\python.exe bench.py "C:\путь\к\записи.mp3" --seconds 180 --only cpu-runtime,multilingual --no-write
```

- Группа `cpu-runtime` проверяет, быстрее ли OpenVINO на процессоре, чем обычный ONNX Runtime.
  Это путь по умолчанию при `asr_backend: auto`, который раньше тестом не измерялся.
- Группа `multilingual` содержит раздел отчёта «Третья модель в голосовании» в сравнении
  с Whisper (`whisper-ct2`).
- На Intel с Arc — группа `multilingual-gpu`. На AMD dml-варианты требуют замены пакета onnxruntime.
- Модели int8 и Multilingual в `config.json` не пишутся. Включение Multilingual третьей
  моделью — отдельная задача после замера.

## Дополнительно из исследования 18.11.2026

Цифры ниже взяты из внешних источников и на машинах проекта не проверены.

**Другие внешние замеры точности на русском.**
- gigastt (int8, Golos Crowd / Golos Farfield): ML large — 4.44 / 5.70%, ML 220M — 6.15 / 8.28%.
  Официальные цифры v3 на Golos Crowd: RNN-T — 2.4%, CTC — 2.8%. Наборы и обвязка разные,
  поэтому сравнение ориентировочное.
- Статья на Хабре «Голосовые в текст на одном ядре»: на одном ядре ML large тратит около 5,6 с
  на минуту речи против 2,2 с у v3 RNN-T, память — 1138 МБ, точность близкая. Статья прочитана
  только через пересказ поиска.

**Скорость (onnx-asr, Ryzen 7 9800X3D, CPU).** ML 220M — 58,5×, ML large — 30,4×,
v3 CTC — 58,5×, v3 RNN-T — 42,4×. Для 6600H, по соотношению с замером v3 CTC в
`amd-bench-baseline-2026-11.md`, это оценочно около 38× и 20× — не измерение.

**Ресурсы.**
- Диск: int8 — 214 МиБ (220M) и 564 МиБ (large); fp32 — около 0,9 ГБ и 2,3 ГБ.
- Память ML large int8 — 0,66–0,8 ГиБ (gigastt); для сравнения, Whisper large-v3 int8 — 2,5–3 ГиБ.
- Новых пакетов не нужно: достаточно onnx-asr 0.12.0 или новее (`install.ps1` ставит последнюю).

**Железо.**
- CPU с AVX2 — работает.
- Intel Arc через OpenVINO — ожидаемо как v3 CTC: архитектура та же. Не проверено.
- NPU для Multilingual не проверялся: для GigaAM он отключён, включать только тестом скорости.
- AMD Radeon — только DirectML после замены пакета onnxruntime.
- NVIDIA CUDA/TensorRT — быстро, но это не целевая платформа.

**Другие среды исполнения v3 на CPU с AVX2.**
- OpenVINO EP на CPU — уже в пакете, проверяется группой `cpu-runtime`.
- int8 через onnx-asr (`asr_quant_cpu`) — экономит память. На 9800X3D ускорения нет, на Zen 3
  без VNNI маловероятно; на Intel с AVX-VNNI нужен замер.
- OpenVINO напрямую (формат IR и квантизация NNCF) — много нового кода, выигрыш не доказан.
- sherpa-onnx, gigastt (Rust) — внутри тот же ONNX Runtime, энкодер быстрее не станет.
- PyTorch — медленнее onnx-asr (39× против 58× у v3 CTC на 9800X3D).
- Пакеты по 8/16 на AMD уже измерены — медленнее поштучного распознавания.

**Следующий шаг после замера на машине владельца.** Если Multilingual large в голосовании
не хуже Whisper, можно сделать его выбираемым значением `asr_third_model`: отдельной
настройкой в `config.json`, без изменения умолчаний `settings.py`. Нужно проверить
нормализацию в `vote3` для чисел словами и английского кириллицей.

**Источники.**
[salute-developers/GigaAM](https://github.com/salute-developers/GigaAM) (README, `evaluation.md`),
[istupakov/onnx-asr](https://github.com/istupakov/onnx-asr) (`docs/comparison.md`, `docs/benchmarks.md`),
[ekhodzitsky/gigastt](https://github.com/ekhodzitsky/gigastt) (`docs/benchmarks.md`, `docs/models.md`),
[статья arXiv 2607.10371](https://arxiv.org/abs/2607.10371),
[Хабр, статья 1102072](https://habr.com/ru/articles/1102072/).
