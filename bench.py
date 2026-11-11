"""Сравнивает варианты распознавания на вашей записи и записывает лучший в config.json.

    .venv\\Scripts\\python.exe bench.py "C:\\путь\\к\\записи.mp3" --seconds 180

Порядок проверки: сначала процессор, потом видеокарта, потом NPU (если он есть и поддерживает
FP32). NPU Meteor Lake (Core Ultra Series 1) пропускается: он не умеет FP32 и считает GigaAM
неверно. Основной проход RNN-T, второй проход CTC и Whisper проверяются и сохраняются отдельно:
на одном компьютере они могут оказаться на разном железе.

Каждый вариант запускается в отдельном процессе с таймаутом: если, например, NPU
компилирует модель слишком долго или падает, тест просто идёт дальше.
Точность каждого варианта сравнивается с эталоном (оригинальный GigaAM на CPU).
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
SR = 16000

RNNT, CTC, WHISPER, TORCH = "rnnt", "ctc", "whisper", "torch"
RUN_ORDER = {TORCH: 0, RNNT: 1, CTC: 2, WHISPER: 3}
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
]

DEFAULT_SET = ("torch-cpu,onnx-cpu,onnx-cpu-b8,onnx-cpu-b16,"
               "ov-gpu,ov-gpu-pad5,ov-gpu-pad22,"
               "ctc-cpu,ctc-cpu-b8,ctc-gpu,ctc-gpu-b8")
DEFAULT_WHISPER = "whisper-gpu,whisper-cpu"
DEFAULT_NPU = "ov-npu-pad22,ctc-npu-pad22"

CONFIG_KEYS = ("asr_backend", "asr_device", "pad_to_seconds", "asr_model", "asr_batch",
               "asr_allow_npu", "asr_second_backend", "asr_second_device", "asr_second_batch",
               "asr_third_device", "asr_third_quant", "asr_third_quant_cpu",
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


def select_variants(only="", npu=None, with_whisper=True):
    """Варианты в порядке CPU → GPU → NPU; NPU без FP32 (Meteor Lake) не добавляется."""
    if npu is None:
        from backends import npu_info
        npu = npu_info()
    npu_ok = bool(npu and npu.get("available") and npu.get("fp32"))
    ids = [x.strip() for x in (only or DEFAULT_SET).split(",") if x.strip()]
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
              if r.get("family") == family and r.get("rtfx") and r.get("text", "").strip()]
    if not usable:
        return {"accurate": None, "fast_ok": None, "fastest": None}
    accurate = max(usable, key=lambda r: (r.get("similarity", 0), r["rtfx"]))
    ok = [r for r in usable if r.get("similarity", 0) >= threshold]
    return {"accurate": accurate,
            "fast_ok": max(ok, key=lambda r: r["rtfx"]) if ok else None,
            "fastest": max(usable, key=lambda r: r["rtfx"])}



def run_one(variant, npy, model, chunk_max):
    """Выполняется в дочернем процессе: грузит бэкенд, распознаёт куски, печатает JSON."""
    from backends import OnnxASR, TorchASR, WhisperOV, make_asr
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
    t0 = time.time()
    if variant["asr_backend"] == "torch":
        asr = TorchASR(model)
    elif family == WHISPER:
        third = variant.get("asr_third_model", "onnx-community/whisper-large-v3-turbo")
        if variant["asr_backend"] == "whisper-ov":
            asr = WhisperOV(third, device="GPU", quant=variant.get("asr_third_quant", "fp16"),
                            language="ru", batch=4)
        else:
            asr, _ = make_asr({"asr_model": third, "asr_backend": "onnx-cpu", "asr_device": "CPU",
                               "asr_quant_cpu": variant.get("asr_third_quant_cpu", "int8"),
                               "asr_language": "ru", "asr_batch": 4})
    else:
        prov = {"onnx-cpu": "cpu"}.get(variant["asr_backend"], variant["asr_backend"])
        asr = OnnxASR(model, device=variant.get("asr_device", "CPU"), provider=prov,
                      pad_to=variant.get("pad_to_seconds", 0), batch=variant.get("asr_batch", 1),
                      allow_npu=bool(variant.get("allow_npu", False)))
    # прогрев: первая компиляция/загрузка не должна попадать в замер скорости
    t_first = time.time()
    if family == WHISPER:
        asr.transcribe_texts(pieces[:1])
    else:
        asr.transcribe(pieces[0])
    t_first = time.time() - t_first
    load = time.time() - t0
    t1 = time.time()
    if family == WHISPER:
        texts = asr.transcribe_texts(pieces)
    else:
        texts = asr.transcribe_many(pieces)
    work = time.time() - t1
    print("@@RESULT@@" + json.dumps({"load_sec": round(load, 1), "first_chunk_sec": round(t_first, 1),
                                     "asr_sec": round(work, 3), "chunks": len(pieces),
                                     "family": family, "device": device_of(variant),
                                     "text": " ".join(texts)}, ensure_ascii=False))


def best_of(results, family, min_similarity=0.97):
    ok = [r for r in results
          if r.get("family") == family and "rtfx" in r and r.get("text", "").strip()
          and r.get("similarity", 1) >= min_similarity]
    return max(ok, key=lambda r: r["rtfx"]) if ok else None


def merge_variant(result, variants):
    """Результат замера + параметры варианта: saved_settings должен видеть batch, pad и backend."""
    if not result:
        return None
    variant = next((v for v in variants if v["id"] == result.get("id")), {})
    return {**variant, **result}


def family_refs(results):
    """Эталон для каждого семейства — его же вариант на процессоре.

    CTC и Whisper — другие модели, их текст и так отличается от RNN-T; сверять их с RNN-T
    значит забраковать любой исправный вариант. Сверяем устройство с тем же семейством на CPU:
    так видно именно искажение от GPU/NPU, а не разницу моделей.
    """
    refs = {}
    for fam in (RNNT, CTC, WHISPER):
        cands = [r for r in results if r.get("family") == fam and r.get("text", "").strip()]
        cpu = [r for r in cands if str(r.get("device", "")).upper() == "CPU"]
        refs[fam] = (cpu or cands)[0] if cands else None
    torch = [r for r in results if r.get("family") == TORCH and r.get("text", "").strip()]
    if torch:                      # PyTorch — исходная модель: лучший эталон точности для RNN-T
        refs[RNNT] = torch[0]
    return refs



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
    ap.add_argument("--only", default="", help="через запятую id вариантов")
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
    try:
        npu = npu_info()
        if a.no_npu:
            npu = {**npu, "fp32": False, "reason": "проверка NPU отключена ключом --no-npu"}
        st.set(stage="подготовка теста", progress=1, openvino_devices=openvino_devices(),
               onnx_providers=available_providers(), npu=npu)
        audio = load_audio(Path(a.audio).expanduser())[: a.seconds * SR]
        dur = len(audio) / SR
        tmp = tmpfiles.scratch("bench_")
        npy = tmp / "audio.npy"
        np.save(npy, audio)

        variants = select_variants(a.only, npu=npu, with_whisper=not a.no_whisper)
        if not variants:
            raise RuntimeError("не выбрано ни одного варианта: проверьте --only и доступные устройства")
        results = []
        for n, v in enumerate(variants):
            st.set(stage=f"тест: {v['label']}", progress=int(100 * n / len(variants)))
            cmd = [sys.executable, str(Path(__file__).resolve()), "--one",
                   json.dumps(v, ensure_ascii=False), "--npy", str(npy), "--model", a.model,
                   "--chunk-max", str(a.chunk_max)]
            r = {"id": v["id"], "label": v["label"], "family": v["family"], "device": device_of(v)}
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=a.timeout, cwd=str(HERE))
                line = next((l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
                if line:
                    r.update(json.loads(line[len("@@RESULT@@"):]))
                    r["rtfx"] = round(dur / max(r["asr_sec"], 1e-6), 1)
                else:
                    r["error"] = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["нет вывода"]
            except subprocess.TimeoutExpired:
                r["error"] = [f"дольше {a.timeout} с (скорее всего, долгая компиляция модели)"]
            results.append(r)

        refs = family_refs(results)
        for r in results:
            if "text" not in r:
                continue
            ref = refs.get(r.get("family"))
            r["similarity"] = 1.0 if ref is None or ref is r \
                else round(similarity(ref["text"], r["text"]), 3)
            stats = word_stats(ref["text"], r["text"]) if ref is not None else \
                {"words": len(norm(r["text"])), "missing": 0, "extra": 0, "changed": 0}
            r.update(stats)
            r["total_10min"] = time_for(r.get("load_sec"), r.get("rtfx"))

        main_family = CTC if "ctc" in str(a.model).lower() else RNNT
        best_main = best_of(results, main_family, threshold)
        if best_main is None and main_family == RNNT:
            best_main = best_of(results, CTC, threshold)   # запасной путь: RNN-T не прошёл порог
        best_ctc = best_of(results, CTC, threshold) if main_family == RNNT else None
        best_whisper = best_of(results, WHISPER, threshold)

        lines = [f"# Тест распознавания: {Path(a.audio).name}, {int(dur)} с, фрагменты до {a.chunk_max:g} с",
                 "",
                 f"Устройства OpenVINO: {', '.join(openvino_devices()) or 'не найдены'}",
                 npu_note(npu),
                 f"Порог качества при выборе: **{threshold * 100:.0f}%** (`bench_min_similarity`; "
                 f"выше — строже к точности). Время «на 10 мин» включает загрузку модели.",
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
            if r.get("similarity", 1) < threshold:
                mark = f"ниже порога {threshold * 100:.0f}%"
            lines.append(f"| {r['label']} | {r['rtfx']}× | {fmt_seconds(r.get('total_10min'))} | "
                         f"{sim} | {words} | {mark} |")

        best_main_v = merge_variant(best_main, variants)
        best_ctc_v = merge_variant(best_ctc, variants)
        best_whisper_v = merge_variant(best_whisper, variants)
        saved = saved_settings(best_main_v, best_ctc_v, best_whisper_v, a.model,
                               a.min_similarity) if best_main_v else {}
        if best_main:
            lines += ["", f"Основной проход: **{best_main['label']}**"]
            lines.append(f"Второй проход CTC: **{best_ctc['label']}**" if best_ctc
                         else "Второй проход CTC: не тестировался")
            lines.append(f"Whisper: **{best_whisper['label']}**" if best_whisper
                         else "Whisper: не тестировался или не прошёл порог точности")
            if saved and not a.no_write:
                # результат теста — это выбор этой машины, поэтому пишем его явно,
                # даже если он совпал с общим умолчанием (например, процессор)
                settings.update(explicit=saved.keys(), **saved)
                lines.append("Настройки этой машины сохранены в config.json папки данных: "
                             + ", ".join(f"{k}={v}" for k, v in saved.items()))
            else:
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
            ch = choice_summary(results, fam, threshold)
            if not ch["accurate"]:
                continue
            lines.append(f"- {title}, точность важнее: {describe(ch['accurate'])}")
            if ch["fast_ok"] and ch["fast_ok"] is not ch["accurate"]:
                lines.append(f"- {title}, скорость в пределах порога: {describe(ch['fast_ok'])}")
            if ch["fastest"] is not ch["fast_ok"]:
                lines.append(f"- {title}, самое быстрое вообще: {describe(ch['fastest'])} — ниже порога, "
                             f"для работы, где важна точность, не подходит")

        # тексты всех вариантов — чтобы глазами сравнить, где именно они расходятся
        paths.BENCH.mkdir(parents=True, exist_ok=True)
        (paths.BENCH / "bench_texts.json").write_text(json.dumps(
            {r["id"]: r.get("text", "") for r in results}, ensure_ascii=False, indent=2), encoding="utf-8")
        report = paths.BENCH / "bench_report.md"
        report.write_text("\n".join(lines), encoding="utf-8")
        for r in results:
            r.pop("text", None)
        st.set(state="done", stage="готово", progress=100, report=str(report),
               best=best_main["id"] if best_main else None, results=results, saved=saved or None)
        print("\n".join(lines))
    except Exception as ex:
        import traceback
        st.set(state="error", stage="ошибка", error=f"{type(ex).__name__}: {ex}",
               trace=traceback.format_exc()[-3000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
