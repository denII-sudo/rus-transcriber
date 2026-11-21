"""Сравнивает варианты распознавания на вашей записи и записывает лучший в config.json.

    .venv\\Scripts\\python.exe bench.py "C:\\путь\\к\\записи.mp3" --seconds 180

Порядок проверки: сначала процессор, потом видеокарта, потом NPU (если он есть и поддерживает
FP32). NPU Meteor Lake (Core Ultra Series 1) пропускается: он не умеет FP32 и считает GigaAM
неверно. Основной проход RNN-T, второй проход CTC и Whisper проверяются и сохраняются отдельно:
на одном компьютере они могут оказаться на разном железе.

Каждый вариант запускается в отдельном процессе с таймаутом: если, например, NPU
компилирует модель слишком долго или падает, тест просто идёт дальше.
Точность каждого варианта сравнивается с эталоном (оригинальный GigaAM на CPU).
Whisper — другой движок: он сравнивается с текстом основного прохода, а порог точности
к нему не применяется, иначе зацикленный вариант выигрывал бы по скорости.
"""
import argparse
import difflib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):  # Windows: лог в файл иначе пишется в cp1251 и падает на «→», «×»
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402,F401
import tmpfiles  # noqa: E402
import download_hint  # noqa: E402
import runinfo  # noqa: E402
SR = 16000

RNNT, CTC, WHISPER, TORCH = "rnnt", "ctc", "whisper", "torch"
# GigaAM Multilingual — кандидат в третью модель голосования; только измеряется, в config.json не пишется
ML = "multilingual"
RUN_ORDER = {TORCH: 0, RNNT: 1, CTC: 2, WHISPER: 3, ML: 4}
DEVICE_ORDER = {"CPU": 0, "GPU": 1, "NPU": 2}

VARIANTS = [
    {"id": "torch-cpu", "family": TORCH, "device": "CPU", "asr_backend": "torch",
     "label": "GigaAM PyTorch, процессор (эталон)"},

    # основной проход: RNN-T
    {"id": "onnx-cpu", "family": RNNT, "device": "CPU", "asr_backend": "onnx-cpu",
     "label": "RNN-T, процессор"},
    {"id": "onnx-cpu-b8", "family": RNNT, "device": "CPU", "asr_backend": "onnx-cpu", "asr_batch": 8,
     "label": "RNN-T, процессор, пакеты по 8"},
    {"id": "onnx-cpu-b16", "family": RNNT, "device": "CPU", "asr_backend": "onnx-cpu", "asr_batch": 16,
     "label": "RNN-T, процессор, пакеты по 16"},
    {"id": "ov-gpu", "family": RNNT, "device": "GPU", "asr_backend": "openvino", "asr_device": "GPU",
     "pad_to_seconds": 0, "label": "RNN-T, видеокарта Arc"},
    {"id": "ov-gpu-b8", "family": RNNT, "device": "GPU", "asr_backend": "openvino", "asr_device": "GPU",
     "asr_batch": 8, "label": "RNN-T, Arc, пакеты по 8"},
    {"id": "ov-gpu-pad5", "family": RNNT, "device": "GPU", "asr_backend": "openvino", "asr_device": "GPU",
     "pad_to_seconds": 5, "label": "RNN-T, Arc, куски кратно 5 с"},
    {"id": "ov-gpu-pad22", "family": RNNT, "device": "GPU", "asr_backend": "openvino", "asr_device": "GPU",
     "pad_to_seconds": 22, "label": "RNN-T, Arc, фиксированные 22 с"},
    {"id": "ov-npu-pad22", "family": RNNT, "device": "NPU", "asr_backend": "openvino", "asr_device": "NPU",
     "pad_to_seconds": 22, "allow_npu": True, "label": "RNN-T, NPU, фиксированные 22 с"},
    {"id": "dml-gpu", "family": RNNT, "device": "GPU", "asr_backend": "dml",
     "label": "RNN-T, DirectML"},

    # второй проход: CTC (свои устройство и пакет)
    {"id": "ctc-cpu", "family": CTC, "device": "CPU", "asr_backend": "onnx-cpu",
     "asr_model": "v3_e2e_ctc", "label": "CTC, процессор"},
    {"id": "ctc-cpu-b8", "family": CTC, "device": "CPU", "asr_backend": "onnx-cpu",
     "asr_model": "v3_e2e_ctc", "asr_batch": 8, "label": "CTC, процессор, пакеты по 8"},
    {"id": "ctc-dml", "family": CTC, "device": "GPU", "asr_backend": "dml",
     "asr_model": "v3_e2e_ctc", "label": "CTC, DirectML"},
    {"id": "ctc-gpu", "family": CTC, "device": "GPU", "asr_backend": "openvino", "asr_device": "GPU",
     "asr_model": "v3_e2e_ctc", "label": "CTC, видеокарта Arc"},
    {"id": "ctc-gpu-b8", "family": CTC, "device": "GPU", "asr_backend": "openvino", "asr_device": "GPU",
     "asr_model": "v3_e2e_ctc", "asr_batch": 8, "label": "CTC, Arc, пакеты по 8"},
    {"id": "ctc-npu-pad22", "family": CTC, "device": "NPU", "asr_backend": "openvino", "asr_device": "NPU",
     "asr_model": "v3_e2e_ctc", "pad_to_seconds": 22, "allow_npu": True,
     "label": "CTC, NPU, фиксированные 22 с"},

    # третья модель: Whisper (проверяется отдельно от GigaAM)
    {"id": "whisper-gpu", "family": WHISPER, "device": "GPU", "asr_backend": "whisper-ov",
     "asr_third_device": "GPU", "asr_third_quant": "fp16", "label": "Whisper, видеокарта Arc (fp16)"},
    {"id": "whisper-cpu", "family": WHISPER, "device": "CPU", "asr_backend": "whisper-cpu",
     "asr_third_device": "CPU", "asr_third_quant_cpu": "int8", "label": "Whisper, процессор (int8)"},
    {"id": "whisper-ct2", "family": WHISPER, "device": "CPU", "asr_backend": "whisper-ct2",
     "label": "Whisper, faster-whisper (int8)"},
    {"id": "whisper-dml", "family": WHISPER, "device": "GPU", "asr_backend": "dml",
     "asr_quant": "int8", "label": "Whisper, DirectML (int8)"},

    # Другие среды исполнения GigaAM v3 на процессоре. Только по --only (в DEFAULT_SET не входят).
    # OpenVINO EP на CPU — путь по умолчанию при asr_backend=auto, но раньше тестом не измерялся;
    # он может быть выбран и сохранён, как любой вариант своего семейства.
    {"id": "ov-cpu", "family": RNNT, "device": "CPU", "asr_backend": "openvino", "asr_device": "CPU",
     "label": "RNN-T, процессор через OpenVINO"},
    {"id": "ctc-ov-cpu", "family": CTC, "device": "CPU", "asr_backend": "openvino", "asr_device": "CPU",
     "asr_model": "v3_e2e_ctc", "label": "CTC, процессор через OpenVINO"},
    # int8 — эксперимент: квантизация основного/второго прохода в config.json не сохраняется,
    # поэтому такой вариант не выбирается, а только показывается рядом с fp32
    {"id": "onnx-cpu-int8", "family": RNNT, "device": "CPU", "asr_backend": "onnx-cpu",
     "quant": "int8", "experimental": True, "label": "RNN-T, процессор, int8 (эксперимент)"},
    {"id": "ctc-cpu-int8", "family": CTC, "device": "CPU", "asr_backend": "onnx-cpu",
     "asr_model": "v3_e2e_ctc", "quant": "int8", "experimental": True,
     "label": "CTC, процессор, int8 (эксперимент)"},

    # GigaAM Multilingual (CTC, без пунктуации) — кандидат в третью модель вместо Whisper.
    # Сверяется с основным проходом, как Whisper; в config.json не пишется ни при каком результате.
    {"id": "ml-cpu", "family": ML, "device": "CPU", "asr_backend": "onnx-cpu",
     "asr_model": "multilingual_ctc", "experimental": True,
     "label": "Multilingual 220M, процессор"},
    {"id": "ml-cpu-int8", "family": ML, "device": "CPU", "asr_backend": "onnx-cpu",
     "asr_model": "multilingual_ctc", "quant": "int8", "experimental": True,
     "label": "Multilingual 220M, процессор, int8"},
    {"id": "ml-large-cpu", "family": ML, "device": "CPU", "asr_backend": "onnx-cpu",
     "asr_model": "multilingual_large_ctc", "experimental": True,
     "label": "Multilingual large 600M, процессор"},
    {"id": "ml-large-cpu-int8", "family": ML, "device": "CPU", "asr_backend": "onnx-cpu",
     "asr_model": "multilingual_large_ctc", "quant": "int8", "experimental": True,
     "label": "Multilingual large 600M, процессор, int8"},
    {"id": "ml-large-ov-cpu", "family": ML, "device": "CPU", "asr_backend": "openvino", "asr_device": "CPU",
     "asr_model": "multilingual_large_ctc", "experimental": True,
     "label": "Multilingual large 600M, процессор через OpenVINO"},
    {"id": "ml-large-gpu", "family": ML, "device": "GPU", "asr_backend": "openvino", "asr_device": "GPU",
     "asr_model": "multilingual_large_ctc", "experimental": True,
     "label": "Multilingual large 600M, видеокарта Arc"},
    {"id": "ml-large-dml", "family": ML, "device": "GPU", "asr_backend": "dml",
     "asr_model": "multilingual_large_ctc", "experimental": True,
     "label": "Multilingual large 600M, DirectML"},
]

DEFAULT_SET = ("torch-cpu,onnx-cpu,onnx-cpu-b8,onnx-cpu-b16,"
               "ov-gpu,ov-gpu-pad5,ov-gpu-pad22,"
               "ctc-cpu,ctc-cpu-b8,ctc-gpu,ctc-gpu-b8")
DEFAULT_WHISPER = "whisper-gpu,whisper-cpu,whisper-ct2,whisper-dml"
DEFAULT_NPU = "ov-npu-pad22,ctc-npu-pad22"
# Наборы для --only: имя группы раскрывается в список вариантов. Эталоны (onnx-cpu, ctc-cpu)
# включены, чтобы совпадение и голосование считались относительно обычного fp32-прохода.
GROUPS = {
    "cpu-runtime": "onnx-cpu,ov-cpu,onnx-cpu-int8,ctc-cpu,ctc-ov-cpu,ctc-cpu-int8",
    "multilingual": "onnx-cpu,ctc-cpu,whisper-ct2,ml-cpu,ml-cpu-int8,ml-large-cpu,"
                    "ml-large-cpu-int8,ml-large-ov-cpu",
    "multilingual-gpu": "onnx-cpu,ctc-cpu,ml-large-cpu,ml-large-gpu,ml-large-dml",
}

CONFIG_KEYS = ("asr_backend", "asr_device", "pad_to_seconds", "asr_model", "asr_batch",
               "asr_allow_npu", "asr_second_backend", "asr_second_device", "asr_second_batch",
               "asr_third_device", "asr_third_quant", "asr_third_quant_cpu", "asr_third_backend",
               "bench_min_similarity")


def device_of(variant):
    """Куда пойдёт модель этого варианта."""
    dev = variant.get("device")
    if dev:
        return str(dev).strip().upper()
    backend = variant.get("asr_backend")
    if backend == "openvino":
        return "CPU" if str(variant.get("asr_device", "GPU")).strip().upper() == "CPU" else "GPU"
    return "GPU" if backend == "dml" else "CPU"


def select_variants(only="", npu=None, with_whisper=True, providers=None):
    """Варианты в порядке CPU → GPU → NPU; NPU без FP32 (Meteor Lake) не добавляется.

    Провайдер DirectML есть только у пакета onnxruntime-directml, который установщик
    сносит (взаимно исключаем с onnxruntime-openvino). Поэтому в дефолтном наборе dml-
    варианты пропускаются без запуска, если провайдера нет: попытка заведомо падает и
    только тратит время. При явном `--only dml…` вариант сохраняется и падает с честной
    ошибкой в отчёте — так человек видит, что именно не работает и почему.
    """
    if npu is None:
        from backends import npu_info
        npu = npu_info()
    npu_ok = bool(npu and npu.get("available") and npu.get("fp32"))
    explicit_only = bool(only)
    if providers is None and not explicit_only:
        from backends import available_providers as _ap
        providers = _ap()
    dml_ok = providers is None or "DmlExecutionProvider" in providers
    ids = []
    for x in (only or DEFAULT_SET).split(","):
        x = x.strip()
        ids += [y.strip() for y in GROUPS[x].split(",")] if x in GROUPS else ([x] if x else [])
    if not only:
        if with_whisper:
            ids += [x.strip() for x in DEFAULT_WHISPER.split(",")]
        if npu_ok:
            ids += [x.strip() for x in DEFAULT_NPU.split(",")]
    by_id = {v["id"]: v for v in VARIANTS}
    out, seen = [], set()
    for i in ids:
        v = by_id.get(i)
        if not v or v["id"] in seen:
            continue
        if v["device"] == "NPU" and not npu_ok:
            continue
        if (not explicit_only and v.get("asr_backend") == "dml" and not dml_ok):
            continue
        out.append(v)
        seen.add(v["id"])
    # сначала все варианты на процессоре, потом на видеокарте, потом на NPU
    out.sort(key=lambda v: (DEVICE_ORDER[v["device"]], RUN_ORDER[v["family"]], v.get("order", 0)))
    return out


def npu_note(npu):
    name = npu.get("name") or "NPU"
    if not npu.get("available"):
        return "NPU: не найден — проверяются только процессор и видеокарта."
    if not npu.get("fp32"):
        return f"NPU {name}: пропущен — {npu.get('reason') or 'не поддерживает FP32'}."
    return f"NPU {name}: будет проверен после процессора и видеокарты."


def actual_device(result):
    """На чём вариант посчитался на самом деле — по имени бэкенда, которое сообщил дочерний процесс.

    Запрошенное устройство известно из варианта, но бэкенд умеет тихо откатиться на процессор
    (make_asr). None — понять не удалось."""
    name = str(result.get("backend_name") or "").strip()
    if not name:
        return None
    parts = name.split(":")
    head = parts[0].lower()
    provs = result.get("providers_actual") or []
    if head == "openvino":
        if provs and "OpenVINOExecutionProvider" not in provs:
            return "CPU"
        dev = parts[1].upper() if len(parts) > 1 else "GPU"
        return next((d for d in ("CPU", "GPU", "NPU") if dev.startswith(d)), None)
    if head == "dml":
        return "CPU" if provs and "DmlExecutionProvider" not in provs else "GPU"
    if head == "whisper-ov":
        dev = parts[1].upper() if len(parts) > 1 else ""
        return next((d for d in ("CPU", "GPU", "NPU") if dev.startswith(d)), None)
    if head in ("cpu", "onnx-cpu", "torch", "whisper-ct2"):
        return "CPU"
    return None


def mark_actual(results):
    """Дописывает фактическое устройство и признак подмены: запрошено одно — посчитано на другом."""
    for r in results:
        if "asr_sec" not in r:
            continue
        r["device_actual"] = actual_device(r)
        r["device_mismatch"] = bool(r["device_actual"] and r["device_actual"] != r.get("device"))
    return results


# Кусок «в разы беднее» опоры (21.11.2026: CTC на Arc оставил по одному слову из ~30 в каждом
# куске — пустых не было, и потерю отсекал только порог совпадения). Опора куска — вариант группы
# с наибольшим числом слов в нём. Потеря — когда у варианта слов не больше трети опоры: исправные
# варианты одной модели расходятся на единицы процентов (e2e-модели пишут числа цифрами:
# «двадцать пять» → «25» сокращает два слова до одного, но не весь кусок втрое), а сбой даёт
# расхождение в десятки раз.
LOST_WORDS_FACTOR = 3
# Минимум слов у опоры: на коротком куске (обрывок фразы, пауза с одним-двумя словами) разница
# «3 против 1» — шум, а не потеря речи; кусок обычной речи длиной 15–20 с даёт 30–50 слов.
LOST_WORDS_MIN_REF = 10
# Правило слов — только для моделей GigaAM: Whisper при зацикливании повторяет фразы, и раздутый
# повторами кусок сделал бы «потерявшим речь» исправный вариант. Для Whisper — правило пустых кусков.
WORD_RULE_FAMILIES = (RNNT, CTC, ML)
# Пустой кусок — потеря, если у другого варианта той же группы в этом куске не меньше стольких
# слов. Сравнение покусочное: прежний подсчёт по количеству не видел, что A пуст в куске 1, а B —
# в куске 2 (по одному пустому у обоих, разница 0). Порог ниже LOST_WORDS_MIN_REF: пустота —
# полная потеря, а не «в разы меньше», и короткий кусок с несколькими словами речи тоже должен
# считаться. Но одно-два слова у соседа на паузе — обычно галлюцинация (GigaAM даёт на шуме
# одиночные «а», «м»; Whisper — короткие «титры» вроде «Продолжение следует»), и пустой вариант
# там прав. Явный случай — у соседа нормальный текст (кусок речи 15–20 с даёт 30–50 слов) —
# порог не задевает: он остаётся потерей, как при прежнем правиле.
LOST_EMPTY_MIN_REF = 5


def chunk_words_of(result):
    """Число слов по кускам из результата или None, если вариант его не сообщил (старый результат)."""
    words = result.get("chunk_words")
    if not isinstance(words, list) or len(words) != result.get("chunks") \
            or not all(isinstance(n, int) and n >= 0 for n in words):
        return None
    return words


def mark_lost_chunks(results):
    """Потерянные куски: речь, которую другой вариант той же модели на той же нарезке распознал.

    Пустой кусок сам по себе не сбой: пауза или музыка пусты у всех. Сбой — когда на той же
    нарезке другой вариант семейства текст получил, а этот нет (так GigaAM на видеокарте терял
    часть речи при высокой скорости), или получил в разы меньше слов (LOST_WORDS_FACTOR).
    Сравниваются варианты одного семейства с одинаковым числом кусков; PyTorch-эталон — та же
    модель, что RNN-T.

    lost_chunks = lost_empty_chunks + lost_few_words_chunks; один кусок не попадает в обе части:
    пустой — это 0 слов, «в разы меньше» — хотя бы одно слово.

    Пустые куски (все семейства, и Whisper: пустой кусок — не раздувание повторами) считаются
    покусочно по chunk_words: кусок с 0 слов, где у другого варианта группы их не меньше
    LOST_EMPTY_MIN_REF. Вариант без годного chunk_words (старый результат, длина не совпала) или
    без другого такого варианта в группе считается по-прежнему, по количеству: пустых больше, чем
    у варианта группы, у которого их меньше всего. Смешанная группа: у варианта с chunk_words
    к покусочному числу добавляется проверка количеством против вариантов без chunk_words —
    берётся большее, чтобы старый сосед с текстом не перестал выявлять потерю.

    Куски «слов в разы меньше» — только GigaAM и только у вариантов, сообщивших chunk_words
    (у старых результатов это правило не применяется); lost_few_words_kept / lost_few_words_ref —
    сколько слов осталось в этих кусках и сколько было у опоры."""
    groups = {}
    for r in results:
        if "asr_sec" not in r or r.get("empty_chunks") is None:
            continue
        family = RNNT if r.get("family") == TORCH else r.get("family")
        groups.setdefault((family, r.get("chunks")), []).append(r)
    for (family, chunks), group in groups.items():
        least = min(r["empty_chunks"] for r in group)
        words_of = {id(r): chunk_words_of(r) for r in group}
        counted = [(r, words_of[id(r)]) for r in group if words_of[id(r)] is not None]
        uncounted = [r for r in group if words_of[id(r)] is None]
        least_uncounted = min((r["empty_chunks"] for r in uncounted), default=None)
        ref = [max(col) for col in zip(*(w for _, w in counted))] if len(counted) > 1 else None
        for r in group:
            words = words_of[id(r)]
            if words is None or ref is None:
                lost = r["empty_chunks"] - least               # прежнее правило: по количеству
            else:
                # у самого варианта в пустом куске 0 слов, поэтому опора куска — другой вариант
                lost = sum(1 for n, top in zip(words, ref) if n == 0 and top >= LOST_EMPTY_MIN_REF)
                if least_uncounted is not None:
                    lost = max(lost, r["empty_chunks"] - least_uncounted)
            r["lost_empty_chunks"] = lost
            r["lost_chunks"] = lost
        if ref is None or family not in WORD_RULE_FAMILIES:
            continue
        for r, words in counted:
            few = [(n, top) for n, top in zip(words, ref)
                   if n > 0 and top >= LOST_WORDS_MIN_REF and n * LOST_WORDS_FACTOR <= top]
            r["lost_few_words_chunks"] = len(few)
            r["lost_few_words_kept"] = sum(n for n, _ in few)
            r["lost_few_words_ref"] = sum(top for _, top in few)
            r["lost_chunks"] += len(few)
    return results


def lost_parts(result):
    """Части пояснения о потере речи: «пустой текст» и «слов в разы меньше» — для отчёта."""
    total = result.get("lost_chunks") or 0
    few = result.get("lost_few_words_chunks") or 0
    empty = result.get("lost_empty_chunks")
    empty = total - few if empty is None else empty       # старый результат: только пустые куски
    parts = []
    if empty > 0:
        parts.append(("empty", empty))
    if few > 0:
        parts.append(("few", few))
    return parts


def unfit_reason(result):
    """Почему отработавший вариант нельзя выбирать: его числа не относятся к тому, что он обещает.

    device_mismatch — посчитан не на запрошенном устройстве (тихий откат на процессор): скорость
    не принадлежит варианту, а в config.json записалось бы устройство, которое не работает.
    lost_chunks — потерял речь, которую другой вариант той же модели распознал."""
    if result.get("device_mismatch"):
        return "device_mismatch"
    if (result.get("lost_chunks") or 0) > 0:
        return "lost_chunks"
    return None


def unfit_note(result):
    """Пояснение для отчёта к варианту, который не выбирается; пусто, если вариант годен."""
    reason = unfit_reason(result)
    if reason == "device_mismatch":
        return f"посчитан на {result.get('device_actual')} вместо {result.get('device')} — не выбирается"
    if reason == "lost_chunks":
        how, parts = [], lost_parts(result)
        for kind, n in parts:
            if kind == "empty":
                how.append("текст пуст там, где у другого варианта он есть" + (f": {n}" if len(parts) > 1 else ""))
            else:
                how.append(f"слов в разы меньше, чем у другого варианта той же модели: {n}, осталось "
                           f"{result.get('lost_few_words_kept')} из {result.get('lost_few_words_ref')}")
        return (f"потеряно кусков: {result.get('lost_chunks')} из {result.get('chunks')} "
                f"({'; '.join(how)}) — не выбирается")
    return ""


def lost_line(result):
    """Строка блока «Условия прогона» о потере речи вариантом."""
    chunks = result.get("chunks")
    how = []
    for kind, n in lost_parts(result):
        if kind == "empty":
            how.append(f"вернул пустой текст для {n} из {chunks} кусков")
        else:
            how.append(f"в {n} из {chunks} кусков дал в разы меньше слов (осталось "
                       f"{result.get('lost_few_words_kept')} из {result.get('lost_few_words_ref')})")
    return (f"- **Потеря речи:** «{result.get('label')}» {' и '.join(how)} там, где другой вариант "
            "той же модели речь распознал, — вариант не выбирается.")


def conditions_lines(record):
    """Блок отчёта «Условия прогона»: что записано в файл прогона и чем прогон отличается от образцового."""
    cond = (record.get("conditions") or {}).get("start") or {}
    env = record.get("environment") or {}
    hw = (record.get("testbed") or {}).get("hardware") or {}
    code = env.get("code") or {}
    power = {"ac": "от сети", "battery": "от батареи"}.get(cond.get("power_source"), "не определено")
    if cond.get("battery_percent") is not None and cond.get("power_source") == "battery":
        power += f", заряд {cond['battery_percent']}%"
    scheme = (cond.get("power") or {}).get("scheme") or "не определена"
    load = cond.get("cpu_load_percent")
    free = (cond.get("memory") or {}).get("free_mb")
    others = cond.get("other_heavy_jobs")
    key = (env.get("packages") or {}).get("key") or {}
    pk = ", ".join(f"{n} {key[n]}" for n in ("onnxruntime-openvino", "onnxruntime-directml", "onnxruntime",
                                             "openvino", "onnx-asr", "faster-whisper", "torch") if n in key)
    rev = (code.get("git_revision") or "")[:10] or code.get("files_sha256") or "—"
    lines = ["", "## Условия прогона",
             f"- Питание: {power}; схема питания: {scheme}.",
             "- Загрузка процессора перед стартом: " + (f"{load:g}%" if load is not None else "не измерена")
             + "; свободно памяти: " + (f"{free} МБ" if free is not None else "не измерено")
             + "; других тяжёлых задач: " + (str(others) if others is not None else "не определено") + ".",
             f"- Процессор: {(hw.get('cpu') or {}).get('model') or 'не определён'}.",
             f"- Программа {env.get('program_version') or '—'}, код {rev}"
             + (" (есть незакоммиченные изменения)" if code.get("git_dirty") else "")
             + "; " + (pk or "пакеты не определены") + ".",
             f"- Методика {record.get('method_version')}; звук {(record.get('audio') or {}).get('unit_id') or '—'} "
             "(начало контрольной суммы отрывка); повтор один, разброс не измерялся."]
    reasons = (record.get("preflight") or {}).get("reasons") or []
    if reasons:
        lines.append("- **Прогон с оговорками** — с другими прогонами сравнивать осторожно:")
        lines += [f"  - {runinfo.REASONS_RU.get(x, x)}" for x in reasons]
    for r in record.get("results") or []:
        if r.get("device_mismatch"):
            lines.append(f"- **Подмена устройства:** «{r.get('label')}» запрошен на {r.get('device')}, посчитан на "
                         f"{r.get('device_actual')} — его скорость не относится к {r.get('device')}, "
                         "вариант не выбирается.")
        elif (r.get("lost_chunks") or 0) > 0:
            lines.append(lost_line(r))
    if record.get("run_file"):
        lines.append(f"- Файл прогона: `{record['run_file']}` (в нём нет имени и пути записи).")
    return lines


def norm(t):
    return re.sub(r"[^\w ]+", "", t.lower().replace("ё", "е")).split()


def similarity(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b), autojunk=False).ratio()


def threshold_from(cfg, cli=None):
    """Порог совпадения с эталоном: --min-similarity или bench_min_similarity из config.json."""
    try:
        value = float(cli if cli is not None else (cfg or {}).get("bench_min_similarity", 0.97))
    except (TypeError, ValueError):
        value = 0.97
    return min(max(value, 0.0), 1.0)


def word_stats(ref_text, hyp_text):
    """Сколько слов потеряно, лишних и заменено относительно эталона (для выбора человеком)."""
    a, b = norm(ref_text), norm(hyp_text)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    missing = extra = changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "delete":
            missing += i2 - i1
        elif tag == "insert":
            extra += j2 - j1
        elif tag == "replace":
            changed += min(i2 - i1, j2 - j1)
            missing += max(0, (i2 - i1) - (j2 - j1))
            extra += max(0, (j2 - j1) - (i2 - i1))
    return {"words": len(b), "missing": missing, "extra": extra, "changed": changed}


def time_for(load_sec, rtfx, audio_sec=600.0):
    """Сколько примерно займёт типичная запись (по умолчанию 10 минут) с учётом загрузки модели."""
    if not rtfx:
        return None
    return float(load_sec or 0) + audio_sec / float(rtfx)


def fmt_seconds(sec):
    if sec is None:
        return "—"
    return f"{sec:.0f} с" if sec < 90 else f"{sec / 60:.1f} мин"


def choice_summary(results, family, threshold):
    """Три ориентира для человека: точнее всего, быстрее всего в пределах порога, самое быстрое вообще."""
    usable = [r for r in results
              if r.get("family") == family and r.get("rtfx") and r.get("text", "").strip()
              and not r.get("experimental") and not unfit_reason(r)]
    if not usable:
        return {"accurate": None, "fast_ok": None, "fastest": None}
    accurate = max(usable, key=lambda r: (r.get("similarity", 0), r["rtfx"]))
    ok = [r for r in usable if r.get("similarity", 0) >= threshold]
    return {"accurate": accurate,
            "fast_ok": max(ok, key=lambda r: r["rtfx"]) if ok else None,
            "fastest": max(usable, key=lambda r: r["rtfx"])}



def run_one(variant, npy, model, chunk_max):
    """Выполняется в дочернем процессе: грузит бэкенд, распознаёт куски, печатает JSON."""
    from backends import OnnxASR, TorchASR, WhisperCT2, WhisperOV, make_asr
    from worker import split_at_pauses
    audio = np.load(npy)
    family = variant.get("family", RNNT)
    if family == WHISPER:            # Whisper всё равно считает окно 30 с
        chunk_max = max(float(chunk_max), 28.0)
    chunks = split_at_pauses(audio, 0.0, len(audio) / SR, max_len=chunk_max,
                             search=min(4.0, chunk_max / 2))
    pieces = [audio[int(s * SR):int(e * SR)] for s, e in chunks]
    if not pieces:
        raise RuntimeError("не удалось нарезать запись на куски")
    model = variant.get("asr_model", model)
    from backends import actual_providers
    cache0 = runinfo.cache_counts(paths.CACHE)       # до загрузки: по разнице видно компиляцию и скачивание
    fallbacks = []
    t0 = time.time()
    if variant["asr_backend"] == "torch":
        asr = TorchASR(model)
    elif family == WHISPER:
        third = variant.get("asr_third_model", "onnx-community/whisper-large-v3-turbo")
        if variant["asr_backend"] == "whisper-ov":
            asr = WhisperOV(third, device="GPU", quant=variant.get("asr_third_quant", "fp16"),
                            language="ru", batch=4)
        elif variant["asr_backend"] == "dml":
            # Whisper через общий OnnxASR с провайдером DirectML (для машин без OpenVINO GPU)
            asr, fallbacks = make_asr({"asr_model": third, "asr_backend": "dml", "asr_device": "GPU",
                                       "asr_quant": variant.get("asr_quant", "int8"),
                                       "asr_quant_cpu": variant.get("asr_quant", "int8"),
                                       "asr_language": "ru", "asr_batch": 4})
        elif variant["asr_backend"] == "whisper-ct2":
            asr = WhisperCT2()
        else:
            asr, fallbacks = make_asr({"asr_model": third, "asr_backend": "onnx-cpu", "asr_device": "CPU",
                                       "asr_quant_cpu": variant.get("asr_third_quant_cpu", "int8"),
                                       "asr_language": "ru", "asr_batch": 4})
    else:
        prov = {"onnx-cpu": "cpu"}.get(variant["asr_backend"], variant["asr_backend"])
        asr = OnnxASR(model, device=variant.get("asr_device", "CPU"), provider=prov,
                      pad_to=variant.get("pad_to_seconds", 0), batch=variant.get("asr_batch", 1),
                      quant=variant.get("quant") or None,
                      allow_npu=bool(variant.get("allow_npu", False)))
    # прогрев: первая компиляция/загрузка не должна попадать в замер скорости
    t_first = time.time()
    if family == WHISPER:
        asr.transcribe_texts(pieces[:1])
    else:
        asr.transcribe(pieces[0])
    t_first = time.time() - t_first
    load = time.time() - t0
    cache1 = runinfo.cache_counts(paths.CACHE)       # вне замеряемых интервалов
    t1 = time.time()
    if family == WHISPER:
        texts = asr.transcribe_texts(pieces)
    else:
        texts = asr.transcribe_many(pieces)
    work = time.time() - t1
    # что получилось фактически: имя бэкенда после прогона (пакет мог откатиться на поштучное),
    # провайдеры сессии, откаты make_asr — по ним видно, что «GPU» на самом деле посчитан на процессоре
    fact = {"backend_name": getattr(asr, "name", None),
            "providers_actual": actual_providers(getattr(asr, "model", None)) or None,
            "compile_sec": getattr(asr, "compile_sec", None),
            "fallbacks": [str(x)[:200] for x in fallbacks] or None,
            "empty_chunks": sum(1 for t in texts if not str(t).strip()),
            "chunk_words": [len(norm(str(t))) for t in texts],   # для правила «слов в разы меньше»
            "peak_memory_mb": runinfo.peak_memory_mb(),
            **runinfo.cache_delta(cache0, cache1)}
    print("@@RESULT@@" + json.dumps({"load_sec": round(load, 1), "first_chunk_sec": round(t_first, 1),
                                     "asr_sec": round(work, 3), "chunks": len(pieces),
                                     "family": family, "device": device_of(variant), **fact,
                                     "text": " ".join(texts)}, ensure_ascii=False))


def best_of(results, family, min_similarity=0.97):
    """Самый быстрый вариант семейства в пределах порога; эксперименты не выбираются никогда,
    как и варианты с подменой устройства или потерянными кусками (unfit_reason)."""
    ok = [r for r in results
          if r.get("family") == family and "rtfx" in r and r.get("text", "").strip()
          and r.get("similarity", 1) >= min_similarity and not r.get("experimental")
          and not unfit_reason(r)]
    return max(ok, key=lambda r: r["rtfx"]) if ok else None


def most_accurate(results, family):
    """Самый близкий к эталону вариант семейства; скорость важна только при равенстве.

    Для Whisper порог точности GigaAM не применяется: движок другой, поэтому даже исправный
    текст редко совпадает с RNN-T на 97%. Выбирается наибольшее совпадение, а зацикленный
    или потерявший речь вариант проигрывает по этому же совпадению. Вариант с подменой
    устройства или потерянными кусками не выбирается вовсе (unfit_reason).
    """
    usable = [r for r in results
              if r.get("family") == family and "rtfx" in r and r.get("text", "").strip()
              and not r.get("experimental") and not unfit_reason(r)]
    return max(usable, key=lambda r: (r.get("similarity", 0), r["rtfx"])) if usable else None


def merge_variant(result, variants):
    """Результат замера + параметры варианта: saved_settings должен видеть batch, pad и backend."""
    if not result:
        return None
    variant = next((v for v in variants if v["id"] == result.get("id")), {})
    return {**variant, **result}


def family_refs(results):
    """Эталон для каждого семейства.

    CTC — та же модель GigaAM, поэтому сверяется со своим вариантом на процессоре:
    так видно искажение от GPU/NPU, а не разницу моделей.

    Whisper — другой движок и другая модель, его текст и так отличается от GigaAM. Сверять
    Whisper с другим Whisper нельзя: зацикленный текст одного варианта (повторы раздувают
    его длину) становится «эталоном» и занижает оценку исправного движка. Поэтому Whisper
    сверяется с основным проходом RNN-T (а если он измерялся — с PyTorch-эталоном).
    Если основного прохода нет (прогон только Whisper-вариантов), остаётся прежнее правило:
    первый вариант этого семейства на процессоре.
    """
    refs = {}
    for fam in (RNNT, CTC, WHISPER, ML):
        cands = [r for r in results if r.get("family") == fam and r.get("text", "").strip()]
        # вариант с подменой устройства или потерянными кусками — плохой эталон, если есть годный
        cands = [r for r in cands if not unfit_reason(r)] or cands
        # эталон семейства — обычный вариант (fp32), а не эксперимент: иначе int8 сверялся бы сам с собой
        plain = [r for r in cands if not r.get("experimental")] or cands
        cpu = [r for r in plain if str(r.get("device", "")).upper() == "CPU"]
        refs[fam] = (cpu or plain)[0] if plain else None
    torch = [r for r in results if r.get("family") == TORCH and r.get("text", "").strip()]
    if torch:                      # PyTorch — исходная модель: лучший эталон точности для RNN-T
        refs[RNNT] = torch[0]
    if refs.get(RNNT):
        refs[WHISPER] = refs[RNNT]
        refs[ML] = refs[RNNT]      # Multilingual — другая модель, как и Whisper: сверка с основным проходом
    return refs


def tiebreak_stats(main_text, ctc_text, third_text):
    """Чем третья модель помогла бы голосованию «два из трёх» (words.vote3) на этих текстах.

    contested — слова RNN-T, с которыми CTC не согласен (только там третья модель что-то решает);
    for_rnnt — из них третья модель согласна с RNN-T; for_ctc — согласна с CTC (vote3 заменил бы
    слово); unresolved — разошлись все три (останется спорным для человека).
    Это согласие моделей между собой, а не точность: эталонной расшифровки человеком здесь нет.
    Числа e2e-моделей пишут цифрами, а Multilingual — словами, поэтому числа попадают в unresolved.
    """
    import copy
    import words as W
    main = [{"w": w, "s": 0.0, "e": 0.0} for w in (main_text or "").split()]
    ctc = [{"w": w, "s": 0.0, "e": 0.0} for w in (ctc_text or "").split()]
    third = (third_text or "").split()
    a = [W.norm_word(w["w"]) for w in main]
    ag_ctc, _ = W._ops(a, [W.norm_word(w["w"]) for w in ctc])
    ag_third, _ = W._ops(a, [W.norm_word(w) for w in third])
    contested = sum(1 for x in ag_ctc if not x)
    for_rnnt = sum(1 for x, y in zip(ag_ctc, ag_third) if not x and y)
    voted = copy.deepcopy(main)
    st = W.vote3(voted, copy.deepcopy(ctc), third)
    return {"contested": contested, "for_rnnt": for_rnnt, "for_ctc": st["replaced"],
            "unresolved": st["disputed"], "inserted": st["inserted"]}



def saved_settings(main, second=None, whisper=None, model="v3_e2e_rnnt", min_similarity=None):
    """Какие отличия от умолчаний записать в config.json этой машины."""
    out = {}
    if main:
        out.update({
            "asr_model": main.get("asr_model", model),
            "asr_backend": main.get("asr_backend", "onnx-cpu"),
            "asr_device": device_of(main),
            "pad_to_seconds": main.get("pad_to_seconds", 0),
            "asr_batch": max(1, int(main.get("asr_batch", 1) or 1)),
        })
    if second:
        out.update({
            "asr_second_model": second.get("asr_model", "v3_e2e_ctc"),
            "asr_second_backend": second.get("asr_backend", "onnx-cpu"),
            "asr_second_device": device_of(second),
            "asr_second_batch": max(1, int(second.get("asr_batch", 1) or 1)),
        })
    if whisper:
        dev = device_of(whisper)
        out["asr_third_device"] = dev
        out["asr_third_quant"] = "fp16"
        out["asr_third_quant_cpu"] = "int8"
        # выбор движка третьей модели: faster-whisper vs onnx; пусто — старое поведение
        out["asr_third_backend"] = "whisper-ct2" if whisper.get("asr_backend") == "whisper-ct2" else ""
    # пишем только если основной/второй проход тестировались: иначе прогон
    # одного семейства (например, только Whisper) затёр бы выбор полного теста
    if main or second:
        out["asr_allow_npu"] = any(x and device_of(x) == "NPU" for x in (main, second))
    if min_similarity is not None:
        # порог, заданный ключом --min-similarity, сохраняем: иначе следующий запуск его забудет
        out["bench_min_similarity"] = round(min(max(float(min_similarity), 0.0), 1.0), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--seconds", type=int, default=180, help="сколько секунд записи взять для теста")
    ap.add_argument("--timeout", type=int, default=900, help="лимит на один вариант, с")
    ap.add_argument("--only", default="",
                    help="через запятую id вариантов или группы: " + ", ".join(GROUPS))
    ap.add_argument("--no-write", action="store_true", help="не менять config.json")
    ap.add_argument("--no-whisper", action="store_true", help="не проверять Whisper")
    ap.add_argument("--no-npu", action="store_true", help="не проверять NPU, даже если он поддерживает FP32")
    ap.add_argument("--status", default="")
    ap.add_argument("--one", default="", help=argparse.SUPPRESS)
    ap.add_argument("--npy", default="", help=argparse.SUPPRESS)
    ap.add_argument("--model", default="v3_e2e_rnnt")
    ap.add_argument("--chunk-max", type=float, default=20.0,
                    help="макс. длина фрагмента, с (как в рабочем конвейере: 12–20 с)")
    ap.add_argument("--min-similarity", type=float, default=None,
                    help="порог совпадения текста с эталоном 0..1 (по умолчанию bench_min_similarity "
                         "из config.json; выше порог — строже к точности)")
    a = ap.parse_args()

    if a.one:
        run_one(json.loads(a.one), a.npy, a.model, a.chunk_max)
        return

    from worker import Status, load_audio, load_config, wait_slot
    import paths, settings
    from backends import available_providers, npu_info, openvino_devices
    st = Status(a.status)
    st.data["file"] = a.audio
    cfg = load_config()
    wait_slot(st, cfg, a.status)
    threshold = threshold_from(cfg, a.min_similarity)
    # Паспорт прогона (runinfo.py): всё, что нужно, чтобы потом сопоставить этот прогон с другими.
    # Имени и пути записи в нём нет — только контрольные суммы; путь знает локальный указатель units.json.
    t_run = time.time()
    record = {"schema_version": runinfo.SCHEMA_VERSION, "method_version": runinfo.METHOD_VERSION,
              "kind": "bench", "run_id": runinfo.new_run_id(), "job_id": Path(a.status).stem if a.status else None,
              "started_utc": runinfo.utc_now(), "status": "running",
              "params": {"seconds": a.seconds, "chunk_max": a.chunk_max, "only": a.only, "timeout": a.timeout,
                         "model": a.model, "no_write": a.no_write, "no_whisper": a.no_whisper,
                         "no_npu": a.no_npu, "min_similarity_cli": a.min_similarity, "threshold": threshold,
                         "repeats": 1}}
    try:
        slot_limit = cfg.get("max_heavy_jobs", 2)
        cond_start = runinfo.conditions(paths.JOBS, slot_limit, own_slots=1)   # до всякой своей нагрузки
        record["conditions"] = {"start": cond_start, "end": None}
        providers = available_providers()
        npu = npu_info()
        if a.no_npu:
            npu = {**npu, "fp32": False, "reason": "проверка NPU отключена ключом --no-npu"}
        st.set(stage="подготовка теста", progress=1, openvino_devices=openvino_devices(),
               onnx_providers=providers, npu=npu)
        src = Path(a.audio).expanduser()
        full = load_audio(src)
        audio = full[: a.seconds * SR]
        dur = len(audio) / SR
        pcm = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2").tobytes()
        identity = runinfo.audio_identity(src, pcm, SR, source_samples=len(full))
        record.update(audio=identity,
                      testbed={"testbed_id": runinfo.testbed_id(paths.BENCH), "hardware": runinfo.hardware()},
                      environment={**runinfo.environment(HERE), "onnx_providers": providers,
                                   "openvino_devices": openvino_devices(), "npu": npu},
                      models=runinfo.models_cache(paths.CACHE))
        tmp = tmpfiles.scratch("bench_")
        npy = tmp / "audio.npy"
        np.save(npy, audio)

        variants = select_variants(a.only, npu=npu, with_whisper=not a.no_whisper,
                                   providers=providers)
        dml_note = None
        if not a.only and "DmlExecutionProvider" not in providers:
            dml_note = ("DirectML: провайдер onnxruntime-directml не установлен (установщик ставит "
                        "onnxruntime-openvino, пакеты взаимно исключают друг друга) — dml-варианты "
                        "пропущены без запуска. Чтобы их проверить, замените пакет onnxruntime "
                        "и вызовите bench с --only dml-gpu,whisper-dml.")
        if not variants:
            raise RuntimeError("не выбрано ни одного варианта: проверьте --only и доступные устройства")
        results = []
        for n, v in enumerate(variants):
            st.set(stage=f"тест: {v['label']}", progress=int(100 * n / len(variants)))
            cmd = [sys.executable, str(Path(__file__).resolve()), "--one",
                   json.dumps(v, ensure_ascii=False), "--npy", str(npy), "--model", a.model,
                   "--chunk-max", str(a.chunk_max)]
            r = {"id": v["id"], "label": v["label"], "family": v["family"], "device": device_of(v)}
            if v.get("experimental"):
                r["experimental"] = True
            r["started_utc"] = runinfo.utc_now()
            t_var = time.time()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=a.timeout, cwd=str(HERE))
                line = next((l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
                if line:
                    r.update(json.loads(line[len("@@RESULT@@"):]))
                    r["rtfx"] = round(dur / max(r["asr_sec"], 1e-6), 1)
                else:
                    tail = (p.stderr or p.stdout).strip()
                    r["error"] = tail.splitlines()[-1:] or ["нет вывода"]
                    if download_hint.is_certificate_error(tail):
                        r["cert_error"] = True
            except subprocess.TimeoutExpired:
                r["error"] = [f"дольше {a.timeout} с (скорее всего, долгая компиляция модели)"]
            r["wall_sec"] = round(time.time() - t_var, 1)
            results.append(r)

        mark_actual(results)
        mark_lost_chunks(results)
        for r in results:
            if "asr_sec" in r:
                r["unfit"] = unfit_reason(r)       # в файл прогона: почему вариант не участвовал в выборе
        refs = family_refs(results)
        whisper_only = (any(r.get("family") == WHISPER for r in results)
                        and not any(r.get("family") == RNNT for r in results))
        for r in results:
            if "text" not in r:
                continue
            ref = refs.get(r.get("family"))
            r["reference_variant"] = ref["id"] if ref is not None else None
            r["similarity"] = 1.0 if ref is None or ref is r \
                else round(similarity(ref["text"], r["text"]), 3)
            stats = word_stats(ref["text"], r["text"]) if ref is not None else \
                {"words": len(norm(r["text"])), "missing": 0, "extra": 0, "changed": 0}
            r.update(stats)
            r["total_10min"] = time_for(r.get("load_sec"), r.get("rtfx"))
        # польза кандидата в третью модель: считается, только если есть и RNN-T, и CTC
        vote_ok = bool(refs.get(RNNT) and refs.get(CTC))
        if vote_ok:
            import words as W
            for r in results:
                if r.get("family") in (WHISPER, ML) and r.get("text", "").strip():
                    third = W.clean_whisper(r["text"])[0] if r["family"] == WHISPER else r["text"]
                    r["vote"] = tiebreak_stats(refs[RNNT]["text"], refs[CTC]["text"], third)

        main_family = CTC if "ctc" in str(a.model).lower() else RNNT
        best_main = best_of(results, main_family, threshold)
        if best_main is None and main_family == RNNT:
            best_main = best_of(results, CTC, threshold)   # запасной путь: RNN-T не прошёл порог
        best_ctc = best_of(results, CTC, threshold) if main_family == RNNT else None
        # Whisper — другой движок: порог точности GigaAM к нему не применяется,
        # иначе ни один исправный вариант не прошёл бы, а зацикленный выигрывал бы по скорости
        best_whisper = most_accurate(results, WHISPER)
        whisper_note = ("Whisper сверяется с текстом основного прохода, а не с другим Whisper: иначе "
                        "раздутый повторами текст одного движка завышает его оценку и занижает оценку второго.")
        if whisper_only:
            whisper_note = ("Основной проход не тестировался: Whisper сравнивается с другим Whisper, "
                            "поэтому оценка может быть искажена. Для честного сравнения запустите тест "
                            "с вариантами RNN-T.")

        lines = [f"# Тест распознавания: {Path(a.audio).name}, {int(dur)} с, фрагменты до {a.chunk_max:g} с",
                 "",
                 f"Устройства OpenVINO: {', '.join(openvino_devices()) or 'не найдены'}",
                 npu_note(npu),
                 *([dml_note] if dml_note else []),
                 f"Порог качества при выборе: **{threshold * 100:.0f}%** (`bench_min_similarity`; "
                 f"выше — строже к точности). Время «на 10 мин» включает загрузку модели.",
                 whisper_note,
                 "",
                 "| Вариант | Скорость | 10 мин записи | Совпадение | "
                 "Слова (−потеряно/+лишних/~заменено) | Итог |",
                 "|---|---|---|---|---|---|"]
        for r in results:
            if "rtfx" not in r:
                lines.append(f"| {r['label']} | — | — | — | — | не работает: {r['error'][0][:120]} |")
                continue
            sim = f"{r['similarity'] * 100:.1f}%" if "similarity" in r else "—"
            words = f"{r.get('words', 0)}"
            if r.get("missing") or r.get("extra") or r.get("changed"):
                words += f" (−{r.get('missing', 0)}/+{r.get('extra', 0)}/~{r.get('changed', 0)})"
            mark = ""
            if best_main is r:
                mark = "**выбран для основного прохода**"
            elif best_ctc is r:
                mark = "**выбран для CTC**"
            elif best_whisper is r:
                mark = "**выбран для Whisper**"
            if r.get("family") not in (WHISPER, ML) and r.get("similarity", 1) < threshold:
                # порог — про точность GigaAM v3; у Whisper и Multilingual шкала другая
                mark = f"ниже порога {threshold * 100:.0f}%"
            if unfit_reason(r):
                mark = unfit_note(r)               # числа варианта не относятся к тому, что он обещает
            if r.get("experimental"):
                mark = (mark + "; " if mark else "") + "эксперимент, не сохраняется"
            lines.append(f"| {r['label']} | {r['rtfx']}× | {fmt_seconds(r.get('total_10min'))} | "
                         f"{sim} | {words} | {mark} |")

        voters = [r for r in results if r.get("vote")]
        if voters:
            lines += ["", "## Третья модель в голосовании «два из трёх»",
                      f"Спорные слова — где CTC (`{refs[CTC]['id']}`) не согласен с RNN-T "
                      f"(`{refs[RNNT]['id']}`): только там третья модель что-то решает. «За RNN-T» — "
                      "подтвердила слово основного прохода, «за CTC» — голосование заменило бы слово, "
                      "«не решено» — разошлись все три, слово останется спорным для человека; «добавлено» — "
                      "слова, которые RNN-T пропустил, а CTC и третья модель услышали одинаково. "
                      "Это согласие моделей между собой, а не проверка точности по расшифровке человека. "
                      "Числа e2e-модели пишут цифрами, Multilingual — словами: они попадают в «не решено».",
                      "",
                      "| Третья модель | Скорость | 10 мин записи | Совпадение с RNN-T | Спорных | "
                      "За RNN-T | За CTC | Не решено | Добавлено |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for r in voters:
                vt = r["vote"]
                n = max(vt["contested"], 1)
                lines.append(f"| {r['label']} | {r['rtfx']}× | {fmt_seconds(r.get('total_10min'))} | "
                             f"{r.get('similarity', 0) * 100:.1f}% | {vt['contested']} | "
                             f"{vt['for_rnnt']} ({vt['for_rnnt'] * 100 / n:.0f}%) | "
                             f"{vt['for_ctc']} ({vt['for_ctc'] * 100 / n:.0f}%) | "
                             f"{vt['unresolved']} ({vt['unresolved'] * 100 / n:.0f}%) | {vt['inserted']} |")
        elif any(r.get("family") == ML and r.get("text", "").strip() for r in results):
            lines += ["", "Multilingual: голосование не оценено — нужны и RNN-T, и CTC "
                          "(добавьте onnx-cpu,ctc-cpu в --only или используйте группу multilingual)."]

        if any(r.get("cert_error") for r in results):
            lines += ["", "**Ошибки сертификата:** сервер моделей не прошёл проверку TLS. "
                          f"Проверка и разовая ручная загрузка: {download_hint.DOC}; "
                          "команды: `.venv\\Scripts\\python.exe download_hint.py`."]

        best_main_v = merge_variant(best_main, variants)
        best_ctc_v = merge_variant(best_ctc, variants)
        best_whisper_v = merge_variant(best_whisper, variants)
        saved = saved_settings(best_main_v, best_ctc_v, best_whisper_v, a.model,
                               a.min_similarity) if (best_main_v or best_whisper_v) else {}
        if best_main:
            lines += ["", f"Основной проход: **{best_main['label']}**"]
            lines.append(f"Второй проход CTC: **{best_ctc['label']}**" if best_ctc
                         else "Второй проход CTC: не тестировался")
            lines.append(f"Whisper: **{best_whisper['label']}**" if best_whisper
                         else "Whisper: не тестировался или не прошёл порог точности")
        elif best_whisper:
            lines += ["", f"Whisper: **{best_whisper['label']}**"]
        if saved and not a.no_write:
            # результат теста — это выбор этой машины, поэтому пишем его явно,
            # даже если он совпал с общим умолчанием (например, процессор)
            settings.update(explicit=saved.keys(), **saved)
            lines.append("Настройки этой машины сохранены в config.json папки данных: "
                         + ", ".join(f"{k}={v}" for k, v in saved.items()))
        elif a.no_write:
            lines.append("Настройки не записаны (--no-write).")

        def describe(r):
            if not r:
                return "—"
            return (f"**{r['label']}** — {r['rtfx']}×, совпадение {r['similarity'] * 100:.1f}%, "
                    f"на 10 мин записи ≈ {fmt_seconds(r.get('total_10min'))}")

        lines += ["", "## Как выбрать",
                  f"Порог качества: {threshold * 100:.0f}%. Он задаётся в config.json ключом "
                  f"`bench_min_similarity` (или ключом `--min-similarity`). Поднимите порог, если "
                  f"точность важнее скорости: тогда тест выберет более точный вариант даже в ущерб "
                  f"скорости процессора.", ""]
        families = [(main_family, "Основной проход")]
        if main_family != CTC:
            families.append((CTC, "Второй проход CTC"))
        families.append((WHISPER, "Whisper"))
        for fam, title in families:
            # у Whisper своя шкала: сравниваем с основным проходом и не режем по порогу GigaAM
            ch = choice_summary(results, fam, 0.0 if fam == WHISPER else threshold)
            if not ch["accurate"]:
                continue
            lines.append(f"- {title}, точность важнее: {describe(ch['accurate'])}")
            if ch["fast_ok"] and ch["fast_ok"] is not ch["accurate"]:
                lines.append(f"- {title}, скорость в пределах порога: {describe(ch['fast_ok'])}")
            if ch["fastest"] is not ch["fast_ok"]:
                lines.append(f"- {title}, самое быстрое вообще: {describe(ch['fastest'])} — ниже порога, "
                             f"для работы, где важна точность, не подходит")

        paths.BENCH.mkdir(parents=True, exist_ok=True)
        texts = {r["id"]: r.get("text", "") for r in results}
        for r in results:
            r.pop("text", None)
        # Файл прогона — история: он не перезаписывается, в отличие от bench_report.md и bench_texts.json,
        # которые по-прежнему хранят только последний прогон.
        cond_end = runinfo.conditions(paths.JOBS, slot_limit, own_slots=1, load_interval=0.0)
        record["conditions"]["end"] = cond_end
        record.update(status="done", finished_utc=runinfo.utc_now(), duration_sec=round(time.time() - t_run, 1),
                      variants=variants, results=results,
                      reference_variants={fam: ref["id"] for fam, ref in refs.items() if ref},
                      choice={"main": best_main["id"] if best_main else None,
                              "ctc": best_ctc["id"] if best_ctc else None,
                              "whisper": best_whisper["id"] if best_whisper else None},
                      saved=saved or None, config_written=bool(saved and not a.no_write),
                      preflight=runinfo.preflight(cond_start, cond_end,
                                                  downloaded=any(r.get("model_downloaded") for r in results)))
        run_path = None
        try:
            run_path = runinfo.write_run(paths.BENCH, record, texts)
            runinfo.update_units_index(paths.BENCH, identity, src, run_path.stem)
        except OSError as ex:
            print(f"[bench] файл прогона не записан: {type(ex).__name__}", file=sys.stderr, flush=True)
        lines += conditions_lines({**record, "run_file": f"runs\\{run_path.name}" if run_path else None})

        # тексты всех вариантов — чтобы глазами сравнить, где именно они расходятся
        (paths.BENCH / "bench_texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
        report = paths.BENCH / "bench_report.md"
        report.write_text("\n".join(lines), encoding="utf-8")
        st.set(state="done", stage="готово", progress=100, report=str(report),
               best=best_main["id"] if best_main else None, results=results, saved=saved or None,
               run_id=run_path.stem if run_path else None, run_file=str(run_path) if run_path else None,
               preflight=record["preflight"])
        print("\n".join(lines))
    except Exception as ex:
        import traceback
        st.set(state="error", stage="ошибка", error=f"{type(ex).__name__}: {ex}",
               trace=traceback.format_exc()[-3000:])
        try:        # неудачный прогон тоже записывается; текст ошибки — в задаче, сюда только её вид
            record.update(status="error", error_type=type(ex).__name__, finished_utc=runinfo.utc_now(),
                          duration_sec=round(time.time() - t_run, 1))
            runinfo.write_run(paths.BENCH, record)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
