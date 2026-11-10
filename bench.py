"""Сравнивает варианты распознавания на вашей записи и записывает лучший в config.json.

    .venv\\Scripts\\python.exe bench.py "C:\\путь\\к\\записи.mp3" --seconds 180

Каждый вариант запускается в отдельном процессе с таймаутом: если, например, NPU
компилирует модель слишком долго или падает, тест просто идёт дальше.
Точность каждого варианта сравнивается с эталоном (оригинальный GigaAM на CPU).
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
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

VARIANTS = [
    {"id": "torch-cpu", "asr_backend": "torch", "label": "GigaAM PyTorch, процессор (эталон)"},
    {"id": "onnx-cpu", "asr_backend": "onnx-cpu", "label": "RNN-T, процессор"},
    {"id": "onnx-cpu-b8", "asr_backend": "onnx-cpu", "asr_batch": 8, "label": "RNN-T, процессор, пакеты по 8"},
    {"id": "onnx-cpu-b16", "asr_backend": "onnx-cpu", "asr_batch": 16, "label": "RNN-T, процессор, пакеты по 16"},
    {"id": "ov-gpu", "asr_backend": "openvino", "asr_device": "GPU", "pad_to_seconds": 0,
     "label": "RNN-T, видеокарта Arc"},
    {"id": "ov-gpu-b8", "asr_backend": "openvino", "asr_device": "GPU", "asr_batch": 8,
     "label": "RNN-T, Arc, пакеты по 8"},
    {"id": "ov-gpu-pad5", "asr_backend": "openvino", "asr_device": "GPU", "pad_to_seconds": 5,
     "label": "RNN-T, Arc, куски кратно 5 с"},
    {"id": "ov-gpu-pad22", "asr_backend": "openvino", "asr_device": "GPU", "pad_to_seconds": 22,
     "label": "RNN-T, Arc, фиксированные 22 с"},
    {"id": "ov-npu-pad22", "asr_backend": "openvino", "asr_device": "NPU", "pad_to_seconds": 22,
     "label": "RNN-T, NPU, фиксированные 22 с"},
    {"id": "dml-gpu", "asr_backend": "dml", "label": "RNN-T, DirectML"},
    {"id": "ctc-cpu", "asr_backend": "onnx-cpu", "asr_model": "v3_e2e_ctc", "label": "CTC, процессор"},
    {"id": "ctc-cpu-b8", "asr_backend": "onnx-cpu", "asr_model": "v3_e2e_ctc", "asr_batch": 8,
     "label": "CTC, процессор, пакеты по 8"},
    {"id": "ctc-gpu", "asr_backend": "openvino", "asr_device": "GPU", "asr_model": "v3_e2e_ctc",
     "label": "CTC, видеокарта Arc"},
    {"id": "ctc-gpu-b8", "asr_backend": "openvino", "asr_device": "GPU", "asr_model": "v3_e2e_ctc",
     "asr_batch": 8, "label": "CTC, Arc, пакеты по 8"},
]
DEFAULT_SET = "onnx-cpu,onnx-cpu-b8,onnx-cpu-b16,ov-gpu,ov-gpu-b8,ctc-cpu,ctc-cpu-b8,ctc-gpu,ctc-gpu-b8"
CONFIG_KEYS = ("asr_backend", "asr_device", "pad_to_seconds", "asr_model", "asr_batch")


def norm(t):
    return re.sub(r"[^\w ]+", "", t.lower().replace("ё", "е")).split()


def similarity(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b), autojunk=False).ratio()


def run_one(variant, npy, model, chunk_max):
    """Выполняется в дочернем процессе: грузит бэкенд, распознаёт куски, печатает JSON."""
    from backends import OnnxASR, TorchASR
    from worker import split_at_pauses
    audio = np.load(npy)
    chunks = split_at_pauses(audio, 0.0, len(audio) / SR, max_len=chunk_max, search=min(4.0, chunk_max / 2))
    pieces = [audio[int(s * SR):int(e * SR)] for s, e in chunks]
    model = variant.get("asr_model", model)
    t0 = time.time()
    if variant["asr_backend"] == "torch":
        asr = TorchASR(model)
    else:
        prov = {"onnx-cpu": "cpu"}.get(variant["asr_backend"], variant["asr_backend"])
        asr = OnnxASR(model, device=variant.get("asr_device", "CPU"), provider=prov,
                      pad_to=variant.get("pad_to_seconds", 0), batch=variant.get("asr_batch", 1))
    # прогрев: первая компиляция/загрузка не должна попадать в замер скорости
    t_first = time.time()
    asr.transcribe(pieces[0])
    t_first = time.time() - t_first
    load = time.time() - t0
    t1 = time.time()
    texts = asr.transcribe_many(pieces)
    work = time.time() - t1
    print("@@RESULT@@" + json.dumps({"load_sec": round(load, 1), "first_chunk_sec": round(t_first, 1),
                                     "asr_sec": round(work, 3), "chunks": len(pieces),
                                     "text": " ".join(texts)}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--seconds", type=int, default=180, help="сколько секунд записи взять для теста")
    ap.add_argument("--timeout", type=int, default=900, help="лимит на один вариант, с")
    ap.add_argument("--only", default="", help="через запятую id вариантов")
    ap.add_argument("--no-write", action="store_true", help="не менять config.json")
    ap.add_argument("--status", default="")
    ap.add_argument("--one", default="", help=argparse.SUPPRESS)
    ap.add_argument("--npy", default="", help=argparse.SUPPRESS)
    ap.add_argument("--model", default="v3_e2e_rnnt")
    ap.add_argument("--chunk-max", type=float, default=6.0,
                    help="макс. длина фрагмента, с (в живом разговоре реплики короткие)")
    a = ap.parse_args()

    if a.one:
        run_one(json.loads(a.one), a.npy, a.model, a.chunk_max)
        return

    from worker import Status, load_audio, load_config, wait_slot
    import paths, settings
    from backends import available_providers, openvino_devices
    st = Status(a.status)
    st.data["file"] = a.audio
    wait_slot(st, load_config(), a.status)
    try:
        st.set(stage="подготовка теста", progress=1,
               openvino_devices=openvino_devices(), onnx_providers=available_providers())
        audio = load_audio(Path(a.audio).expanduser())[: a.seconds * SR]
        dur = len(audio) / SR
        tmp = tmpfiles.scratch("bench_")
        npy = tmp / "audio.npy"
        np.save(npy, audio)

        wanted = (a.only or DEFAULT_SET).split(",")
        variants = [v for v in VARIANTS if v["id"] in wanted]
        results = []
        for n, v in enumerate(variants):
            st.set(stage=f"тест: {v['label']}", progress=int(100 * n / len(variants)))
            cmd = [sys.executable, str(Path(__file__).resolve()), "--one",
                   json.dumps(v, ensure_ascii=False), "--npy", str(npy), "--model", a.model,
                   "--chunk-max", str(a.chunk_max)]
            r = {"id": v["id"], "label": v["label"]}
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

        ref = next((r for r in results if r["id"] == "torch-cpu" and "text" in r), None) \
            or next((r for r in results if r["id"] == "onnx-cpu" and "text" in r), None)
        for r in results:
            if ref and "text" in r:
                r["similarity"] = round(similarity(ref["text"], r["text"]), 3)

        ok = [r for r in results if "rtfx" in r and r.get("similarity", 1) >= 0.97]
        best = max(ok, key=lambda r: r["rtfx"]) if ok else None

        lines = [f"# Тест распознавания: {Path(a.audio).name}, {int(dur)} с, фрагменты до {a.chunk_max:g} с", "",
                 f"Устройства OpenVINO: {', '.join(openvino_devices()) or 'не найдены'}", "",
                 "| Вариант | Скорость (× реального времени) | Загрузка, с | Совпадение с эталоном | Итог |",
                 "|---|---|---|---|---|"]
        for r in results:
            if "rtfx" in r:
                sim = f"{r['similarity'] * 100:.1f}%" if "similarity" in r else "—"
                mark = "**лучший**" if best is r else ("ниже порога точности" if r.get("similarity", 1) < 0.97 else "")
                lines.append(f"| {r['label']} | {r['rtfx']}× | {r['load_sec']} | {sim} | {mark} |")
            else:
                lines.append(f"| {r['label']} | — | — | — | не работает: {r['error'][0][:120]} |")
        if best:
            chosen = next(v for v in VARIANTS if v["id"] == best["id"])
            lines += ["", f"Выбран: **{best['label']}**"]
            if not a.no_write:
                base = {"asr_device": "GPU", "pad_to_seconds": 0, "asr_model": a.model, "asr_batch": 1}
                settings.update(**{k: chosen.get(k, base.get(k)) for k in CONFIG_KEYS})
                lines.append("Настройки сохранены в config.json папки данных")
        # тексты всех вариантов — чтобы глазами сравнить, где именно они расходятся
        paths.BENCH.mkdir(parents=True, exist_ok=True)
        (paths.BENCH / "bench_texts.json").write_text(json.dumps(
            {r["id"]: r.get("text", "") for r in results}, ensure_ascii=False, indent=2), encoding="utf-8")
        report = paths.BENCH / "bench_report.md"
        report.write_text("\n".join(lines), encoding="utf-8")
        for r in results:
            r.pop("text", None)
        st.set(state="done", stage="готово", progress=100, report=str(report),
               best=best["id"] if best else None, results=results)
        print("\n".join(lines))
    except Exception as ex:
        import traceback
        st.set(state="error", stage="ошибка", error=f"{type(ex).__name__}: {ex}",
               trace=traceback.format_exc()[-3000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
