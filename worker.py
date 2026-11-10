"""Обработка одной записи: диаризация (pyannote) + распознавание (GigaAM) + эмоции (GigaAM-Emo).

Запуск вручную:
    .venv\\Scripts\\python.exe worker.py "C:\\путь\\к\\записи.mp3"
    .venv\\Scripts\\python.exe worker.py запись.m4a --speakers 3 --no-emotions

Устройство и модель берутся из config.json (его заполняет bench.py).
Результат кладётся рядом с аудио: <имя>.transcript.md и <имя>.transcript.json
Нужна переменная окружения HF_TOKEN и ffmpeg в PATH.
"""
import argparse
import json
import re
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):  # Windows: лог в файл иначе пишется в cp1251 и падает на «→», «×»
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tmpfiles  # noqa: E402
import paths  # noqa: E402  (кэши моделей — в папке программы)
import transcript  # noqa: E402
from transcript import hms  # noqa: E402

HERE = Path(__file__).resolve().parent
from settings import DEFAULTS, load_config  # noqa: E402,F401  (настройки: умолчания в коде + DATA_DIR\config.json)

SR = 16000
MAX_CHUNK = 22.0   # GigaAM распознаёт куски до ~25 с
MERGE_GAP = 1.0    # склеиваем реплики одного спикера, если пауза меньше
MIN_SEG = 0.3      # совсем короткие обрывки отбрасываем
EMO_MIN = 2.0      # эмоцию считаем только на фрагментах от 2 с
GENDER_MARGIN = 0.05  # род глагола переносит реплику, только если и голос ближе к новому участнику

EMO_RU = {"angry": "раздражение", "sad": "грусть", "neutral": "нейтрально", "positive": "позитив"}


def wait_slot(st, cfg, status_path):
    """Не больше max_heavy_jobs тяжёлых задач одновременно (jobslots.py); время в очереди в задачу не входит."""
    import jobslots
    job_id = Path(status_path).stem if status_path else None
    t = time.time()
    st._slot = jobslots.acquire(paths.JOBS, cfg.get("max_heavy_jobs", 2), status=st, job_id=job_id)
    now = round(time.time(), 1)
    st.set(state="running", stage="старт", progress=0, started_ts=now,
           queued_sec=round(time.time() - t, 1) if time.time() - t > 3 else None)


def resolve_vocab(src, a, cfg):
    """(путь словаря, словарь, группа). Порядок: --vocab → явная --group (её словарь) → словарь рядом с записью
    (и двумя папками выше) → группа по умолчанию (default_group в config.json или единственная в DATA_DIR)."""
    import vocab as vocab_mod
    if a.vocab:
        vp, vocab = vocab_mod.find(src, a.vocab, None)
    elif a.group:
        vp, vocab = vocab_mod.find(None, None, paths.vocab_path(a.group))
    else:
        vp, vocab = vocab_mod.find(src, None, None)
    group = vocab.get("group") or a.group
    if not vp:
        group = group or paths.pick_group(None, cfg.get("default_group"))
        if group:
            vp, vocab = vocab_mod.find(None, None, paths.vocab_path(group))
            group = vocab.get("group") or group
    return vp, vocab, a.group or group


def _hf_downloads():
    """Недокачанные файлы моделей Hugging Face: (число файлов, байт)."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), "hub")
    n = size = 0
    for p in Path(hub).glob("models--*/blobs/*.incomplete"):
        try:
            size += p.stat().st_size
            n += 1
        except OSError:
            pass
    return n, size


class Status:
    """Файл статуса задачи. Кроме этапа и процента пишет время начала задачи и этапа (сервер по ним
    считает, сколько идёт этап), а фоновый поток раз в 5 с отмечает скачивание моделей (МБ и скорость)
    — чтобы долгий этап не выглядел зависшим."""

    def __init__(self, path):
        import threading
        self.path = Path(path) if path else None
        self.data = {}
        if self.path and self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        now = time.time()
        self.data.update(state="running", stage="старт", progress=0, started_ts=round(now, 1),
                         stage_started_ts=round(now, 1))
        self._lock = threading.Lock()
        if self.path:
            threading.Thread(target=self._watch, daemon=True).start()

    def _write(self):
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        for k in range(20):      # Windows: замена не проходит, пока сервер читает файл, — повторяем
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.05 * (k + 1))
        if self.data.get("state") in ("done", "error"):
            raise RuntimeError(f"не удалось записать статус {self.path}")

    def set(self, **kw):
        with self._lock:
            base = lambda x: re.sub(r"\s*\d+/\d+\s*$", "", str(x or ""))
            if "stage" in kw and base(kw["stage"]) != base(self.data.get("stage")):
                kw["stage_started_ts"] = round(time.time(), 1)
                self.data.pop("done", None)          # счётчик «n/m» относится к прежнему этапу
                self.data.pop("download", None)
            self.data.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"[{self.data['updated']}] {self.data.get('stage')} {self.data.get('progress')}%",
                  file=sys.stderr, flush=True)
            self._write()

    def _watch(self):
        prev = None
        while self.data.get("state") not in ("done", "error"):
            time.sleep(5)
            try:
                n, size = _hf_downloads()
            except Exception:
                continue
            now = time.time()
            if n:
                rate = (size - prev[0]) / (now - prev[1]) / 1e6 if prev and size >= prev[0] else None
                dl = {"files": n, "mb": round(size / 1e6), "mb_s": round(rate, 1) if rate is not None else None}
                prev = (size, now)
            else:
                dl, prev = None, None
            if dl != self.data.get("download") and self.data.get("state") not in ("done", "error"):
                with self._lock:
                    if dl:
                        self.data["download"] = dl
                    else:
                        self.data.pop("download", None)
                    self._write()


def hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def load_audio(src):
    """Любой формат → mono 16 кГц float32 через ffmpeg."""
    import soundfile as sf
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg не найден в PATH. Установите: winget install Gyan.FFmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "full.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-ac", "1", "-ar", str(SR), str(wav)], check=True)
        audio, _ = sf.read(wav, dtype="float32")
    return audio


DIAR_STEPS_RU = {"segmentation": "поиск речи", "speaker_counting": "подсчёт голосов",
                 "embeddings": "отпечатки голосов", "discrete_diarization": "сборка разметки"}


class DiarHook:
    """Получает события pyannote: показывает ход в статусе и замеряет время каждого шага."""

    def __init__(self, status):
        self.st, self.t0, self.last_update = status, time.time(), 0.0
        self.order, self.last_seen = [], {}

    def __call__(self, step_name, step_artifact=None, file=None, total=None, completed=None, **kw):
        now = time.time()
        if step_name not in self.last_seen:
            self.order.append(step_name)
        self.last_seen[step_name] = now
        if total and completed is not None and now - self.last_update > 2:
            self.last_update = now
            pct = 5 + int(25 * (self.order.index(step_name) + completed / max(total, 1)) / 4)
            self.st.set(stage=f"разделение по спикерам: {DIAR_STEPS_RU.get(step_name, step_name)} "
                              f"{completed}/{total}", progress=min(pct, 30))

    def durations(self):
        out, prev = {}, self.t0
        for name in self.order:
            out[name] = round(self.last_seen[name] - prev, 1)
            prev = self.last_seen[name]
        return out


def diarize(audio, num_speakers, token, cfg, status, max_speakers=0, min_speakers=0):
    import torch
    from pyannote.audio import Pipeline

    threads = int(cfg.get("torch_threads", 0) or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    name = "pyannote/speaker-diarization-community-1"
    try:
        pipe = Pipeline.from_pretrained(name, token=token)          # pyannote 4.x
    except TypeError:
        pipe = Pipeline.from_pretrained(name, use_auth_token=token)  # pyannote 3.x
    if pipe is None:
        raise RuntimeError("Не удалось загрузить pyannote: проверьте HF_TOKEN и что условия модели "
                           f"{name} приняты на huggingface.co")
    pipe.to(torch.device("cpu"))
    seg_bs = int(cfg.get("diar_segmentation_batch", 0) or 0)
    if seg_bs > 0 and hasattr(pipe, "segmentation_batch_size"):
        pipe.segmentation_batch_size = seg_bs

    info = {"torch_threads": torch.get_num_threads(),
            "segmentation_batch": getattr(pipe, "segmentation_batch_size", None)}
    try:
        import diar_fast
        diar_fast.install(pipe, mode=cfg.get("diar_embeddings", "auto"),
                          ov_device=cfg.get("diar_ov_device", "GPU"),
                          chunks_per_batch=int(cfg.get("diar_chunks_per_batch", 16)),
                          cache_dir=paths.OPENVINO_DIR, info=info)
    except Exception as ex:
        info["fast_error"] = f"{type(ex).__name__}: {ex}"[:300]
        print(f"[diar] ускорение не установлено: {ex}", file=sys.stderr, flush=True)

    hook = DiarHook(status)
    kwargs = {"num_speakers": num_speakers} if num_speakers else (
        {"max_speakers": max_speakers} if max_speakers else {})
    if min_speakers and not num_speakers:
        kwargs["min_speakers"] = min_speakers
    out = pipe({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR}, hook=hook, **kwargs)
    info["steps_sec"] = hook.durations()
    ann = (getattr(out, "exclusive_speaker_diarization", None)
           or getattr(out, "speaker_diarization", None) or out)
    full = getattr(out, "speaker_diarization", None)   # с наложениями — для меток «говорят одновременно»
    return [(seg.start, seg.end, spk) for seg, _, spk in ann.itertracks(yield_label=True)], info, pipe, full


def build_turns(raw):
    turns = []
    for s, e, spk in sorted(raw):
        if turns and turns[-1][2] == spk and s - turns[-1][1] < MERGE_GAP:
            turns[-1][1] = max(turns[-1][1], e)
        else:
            turns.append([s, e, spk])
    return [t for t in turns if t[1] - t[0] >= MIN_SEG]


def split_at_pauses(audio, s, e, max_len=MAX_CHUNK, search=4.0, frame=0.03):
    """Режет длинную реплику на куски ≤ max_len по самым тихим местам, а не посреди слова."""
    out = []
    hop = int(frame * SR)
    while e - s > max_len:
        lo, hi = int((s + max_len - search) * SR), int((s + max_len) * SR)
        seg = audio[lo:hi]
        n = len(seg) // hop
        if n < 2:
            cut = s + max_len
        else:
            rms = np.sqrt((seg[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
            cut = (lo + int(rms.argmin()) * hop + hop // 2) / SR
        out.append((s, cut))
        s = cut
    out.append((s, e))
    return out


def apply_gender(segments, seg_embs, refs, label_of, genders_by_name, fixes, review):
    """Род глагола при «я» против пола подписанного участника.
    Переподписываем, только если и голос ближе к участнику нужного пола (на GENDER_MARGIN). Иначе — в список для проверки (ошибка распознавания
    окончания или чужая речь, например пересказ)."""
    import gender
    for k, seg in enumerate(segments):
        g_need = genders_by_name.get(seg["speaker"])
        if not g_need:
            continue
        g, words = gender.evidence(seg["text"])
        if not g or g == g_need:
            continue
        v = seg_embs[k]
        cands = [n for n, gg in genders_by_name.items() if gg == g and n in refs]
        if v is None or not cands or seg["speaker"] not in refs:
            review.append({"t": seg["start"], "speaker": seg["speaker"], "reason": f"род «{', '.join(words)}» "
                           f"не совпадает с полом говорящего; голос проверить нельзя"})
            continue
        cur = float(refs[seg["speaker"]] @ v)
        best = max(cands, key=lambda n: float(refs[n] @ v))
        bs = float(refs[best] @ v)
        if bs >= cur + GENDER_MARGIN:
            fixes.append({"t": seg["start"], "kind": "gender", "from": seg["speaker"], "to": best,
                          "words": words, "sim": [round(cur, 2), round(bs, 2)]})
            seg["fix"] = f"род «{', '.join(words)}» и голос: {seg['speaker']}→{best}"
            seg["speaker"], seg["label"] = best, label_of[best]
        else:
            review.append({"t": seg["start"], "speaker": seg["speaker"],
                           "reason": f"род «{', '.join(words)}» не совпадает, но голос похож на "
                                     f"{seg['speaker']} ({cur:.2f} против {bs:.2f} у {best}): "
                                     f"вероятно, ошибка окончания или чужая речь"})


class _SkipSegmentSteps(Exception):
    """Пословной обработке шаги разрезания и перепроверки сегментов не нужны."""


def run_words(audio, src, cfg, st, raw, order, turns, full_ann, emb, label_name, label_of, voice_ctx,
              allowed, genders_by_name, vocab, quality, make_asr, a):
    """Пословная обработка (words.py). -> данные расшифровки (без file/duration/timing)."""
    import gender
    import vocab as vocab_mod
    import voices
    import words as W

    timing = {}
    # опорные голоса: профили опознанных участников; участники из словаря с профилем, но без кластера
    # (pyannote мог слить ребёнка с другим); крупные неопознанные кластеры — по центру кластера
    pvecs = voice_ctx.get("pvecs") or {}
    cents = voice_ctx.get("cents") or {}
    if not cents:
        embs = emb.spans(audio, [(t["s"], t["e"]) for t in turns])
        cents = voices.centroids(turns, embs)
    refs = {}
    found = set()
    for lbl, nm in label_name.items():
        refs[nm] = pvecs.get(nm, cents.get(lbl))
        found.add(nm)
    for nm, v in pvecs.items():
        if (not allowed or nm in allowed) and nm not in refs:
            refs[nm] = v
    size = {}
    for t in turns:
        size[t["label"]] = size.get(t["label"], 0.0) + t["e"] - t["s"]
    for lbl, c in cents.items():
        if lbl not in label_name and size.get(lbl, 0) >= 10.0:
            refs[lbl] = c
            found.add(lbl)
    refs = {k: v for k, v in refs.items() if v is not None}
    mapping = voice_ctx.get("mapping") or {}

    def name_of_label(lbl):   # метка pyannote -> имя (в т.ч. для кластеров, слитых в одного человека)
        return (mapping.get(lbl) or (label_name.get(lbl),))[0] or lbl

    prior = W.LabelAt([(s, e, name_of_label(order[spk])) for s, e, spk in raw])

    # распознавание длинными кусками
    regions = W.speech_regions(raw)
    chunks = W.make_chunks(audio, regions)
    st.set(stage="загрузка модели распознавания", progress=36)
    asr, asr_errors = make_asr(cfg)
    if not hasattr(asr, "transcribe_timestamped"):
        raise RuntimeError(f"бэкенд {asr.name} не отдаёт времена слов; задайте \"pipeline\": \"segments\"")
    t1 = time.time()
    last = [0.0]

    def prog(lo, hi, name):
        def f(done, total):
            if time.time() - last[0] > 1 or done == total:
                last[0] = time.time()
                st.set(stage=name, progress=lo + int((hi - lo) * done / total), done=f"{done}/{total}")
        return f

    st.set(stage="распознавание", progress=38, asr_backend=asr.name, asr_fallbacks=asr_errors or None)
    words = W.transcribe(asr, audio, chunks, progress=prog(38, 62, "распознавание"))
    timing.update(asr_sec=round(time.time() - t1), asr_chunks=len(chunks), asr_backend=asr.name,
                  asr_model=getattr(asr, "model_name", cfg.get("asr_model")))
    pieces = [audio[int(c["a"] * SR):int(c["b"] * SR)] for c in chunks]
    alt = None
    second = cfg.get("asr_second_model", "v3_e2e_ctc")
    if second and second != cfg.get("asr_model"):
        try:
            t2 = time.time()
            del asr
            asr2, _ = make_asr({**cfg, "asr_model": second})
            alt = W.transcribe(asr2, audio, chunks, progress=prog(62, 68, "второй проход распознавания (CTC)"))
            del asr2
            timing.update(asr2_sec=round(time.time() - t2), asr2_model=second)
        except Exception as ex:
            quality["asr2_error"] = f"{type(ex).__name__}: {ex}"[:300]
            print(f"[words] второй проход не выполнен: {ex}", file=sys.stderr, flush=True)
    wh_words = None
    third = cfg.get("asr_third_model", "onnx-community/whisper-large-v3-turbo")
    wmode = (getattr(a, "whisper", "") or cfg.get("asr_third_mode", "all")).lower()
    wrep = a.whisper_max_repeat if getattr(a, "whisper_max_repeat", -1) >= 0 else int(cfg.get("asr_third_max_repeat", 3))
    if wmode == "off":
        third = None
    # куски для Whisper — свои, до 28 с (он всегда считает окно 30 с); в режиме doubt — только те,
    # где GigaAM RNN-T и CTC расходятся: в остальных голосование «два из трёх» решено без него
    wchunks = W.make_chunks(audio, regions, max_len=W.WHISPER_MAX, min_len=W.WHISPER_MIN) \
        if cfg.get("asr_third_long_chunks", True) else chunks
    selected = list(range(len(wchunks)))
    if third and alt is not None and wmode == "doubt":
        dt = W.disagreement_times(words, alt)
        selected = [k for k, c in enumerate(wchunks) if any(c["own"][0] <= x < c["own"][1] for x in dt)]
    wpieces = [audio[int(wchunks[k]["a"] * SR):int(wchunks[k]["b"] * SR)] for k in selected]
    if third and alt is not None:
        # Whisper: на видеокарте Arc (OpenVINO, fp16), при сбое — процессор (int8)
        dev = str(cfg.get("asr_third_device", "GPU")).upper()
        tries = ([("whisper-ov", dev, cfg.get("asr_third_quant", "fp16"))] if dev in ("GPU", "NPU") else []) \
            + [("onnx-cpu", "CPU", cfg.get("asr_third_quant_cpu", "int8"))]
        for backend, device, quant in tries:
            try:
                t3 = time.time()
                st.set(stage=f"третья модель: загрузка Whisper ({device})", progress=68)
                if backend == "whisper-ov":
                    from backends import WhisperOV
                    asr3, errs3 = WhisperOV(third, device=device, quant=quant, language="ru",
                                            batch=int(cfg.get("asr_third_batch", 4))), None
                else:
                    asr3, errs3 = make_asr({**cfg, "asr_model": third, "asr_backend": backend, "asr_device": device,
                                            "asr_quant": quant, "asr_quant_cpu": quant, "asr_language": "ru",
                                            "asr_batch": int(cfg.get("asr_third_batch", 4))})
                t_load = time.time() - t3
                asr3_meta = type("M", (), {"compile_sec": getattr(asr3, "compile_sec", None),
                                           "name": getattr(asr3, "name", None)})
                texts = asr3.transcribe_texts(wpieces, progress=prog(68, 74, f"третья модель (Whisper, {device})"))
                del asr3
                by_k = dict(zip(selected, texts))
                wh_words, removed = [], 0
                for k, c in enumerate(wchunks):
                    if k in by_k:
                        tx, n = W.clean_whisper(by_k[k], wrep)
                        removed += n
                        wh_words += tx.split()
                    else:            # RNN-T и CTC здесь согласны: Whisper голосования не изменил бы
                        s0, e0 = c["own"]
                        wh_words += [w["w"] for w in words if s0 <= w["s"] < e0]
                timing.update(asr3_mode=wmode, asr3_pieces=f"{len(selected)}/{len(wchunks)}",
                              asr3_removed_words=removed or None)
                timing.update(asr3_sec=round(time.time() - t3), asr3_load_sec=round(t_load, 1),
                              asr3_model=third, asr3_device=f"{backend}:{device}:{quant}",
                              asr3_fallbacks=errs3 or None,
                              asr3_compile_sec=getattr(asr3_meta, "compile_sec", None),
                              asr3_name=getattr(asr3_meta, "name", None))
                break
            except Exception as ex:
                quality.setdefault("asr3_errors", []).append(f"{backend}:{device}: {type(ex).__name__}: {ex}"[:300])
                print(f"[words] Whisper ({backend}:{device}) не выполнен: {ex}", file=sys.stderr, flush=True)
    if alt is not None and wh_words is not None:
        quality["vote"] = W.vote3(words, alt, wh_words)
        prev = None
        for wd in words:                      # что заменило голосование — в список исправлений
            if wd.get("voted") and wd.get("model", "") != "":
                if prev is not None and prev["model"] == wd["model"] and prev["t2"] >= wd["s"] - 1.0:
                    prev["to"] += " " + wd["w"]
                    prev["t2"] = wd["s"]
                else:
                    prev = {"t": wd["s"], "t2": wd["s"], "model": wd["model"], "to": wd["w"]}
                    quality.setdefault("asr_votes", []).append(prev)
            elif wd.get("voted"):
                quality.setdefault("asr_votes", []).append({"t": wd["s"], "t2": wd["s"], "model": "", "to": wd["w"]})
                prev = None
            else:
                prev = None
    elif alt is not None:
        W.mark_alt(words, alt)

    # словарь — по словам, исходное слово сохраняется
    if vocab:
        fixer = vocab_mod.Fixer(vocab)
        for k, w in enumerate(words):
            new, fx = fixer.apply(w["w"], prev=words[k - 1]["w"] if k else None,
                                  nxt=words[k + 1]["w"] if k + 1 < len(words) else None)
            if fx:
                w["raw"], w["w"] = w["w"], new
                quality["vocab_fixes"].append({"t": w["s"], "fixes": fx})

    # голос для каждого слова
    st.set(stage="голос для каждого слова", progress=75)
    t3 = time.time()
    nwin = W.assign_voices(audio, words, refs, emb, prior=prior, info={"found": found},
                           win_near_change=float(cfg.get("word_win_near_change", 0) or 0),
                           confusions=voice_ctx.get("confusions"),
                           progress=prog(75, 85, "голос для каждого слова"))
    timing.update(word_voice_sec=round(time.time() - t3, 1), word_windows=nwin)
    replies = W.build_replies(words)

    # реплики -> сегменты
    used = set(label_name)
    for nm in refs:
        if nm not in label_of and not str(nm).startswith("Спикер"):
            k = 1
            while f"Спикер {k}" in used:
                k += 1
            label_of[nm] = f"Спикер {k}"
            label_name[f"Спикер {k}"] = nm
            used.add(f"Спикер {k}")
    segments = []
    for r in replies:
        ws = r["words"]
        spk = r["spk"]
        seg = {"start": ws[0]["s"], "end": ws[-1]["e"], "label": label_of.get(spk, spk), "speaker": spk,
               "text": W.reply_text(ws),
               "words": [{k: w[k] for k in ("w", "s", "e", "c", "raw", "alt", "alt_whisper", "alt_span", "alt_ins", "voted",
                                         "model", "margin")
                          if k in w} for w in ws]}
        for wd, w in zip(seg["words"], ws):
            wd["sims"] = w.get("sims")
            wd["_flags_src"] = w
        if r.get("inserts"):
            seg["inserts"] = [{**i, "label": label_of.get(i["speaker"], i["speaker"])} for i in r["inserts"]]
        segments.append(seg)

    # род глагола против пола: голос близок к участнику нужного пола — переносим, иначе метка
    for seg in segments:
        need = genders_by_name.get(seg["speaker"])
        g, ev = gender.evidence(seg["text"])
        if not need or not g or g == need:
            continue
        avg = {}
        for wd in seg["words"]:
            for n, v in (wd.get("sims") or {}).items():
                avg[n] = avg.get(n, 0.0) + v / len(seg["words"])
        cands = sorted(((avg.get(n, -1), n) for n, gg in genders_by_name.items() if gg == g and n in refs),
                       reverse=True)
        cur = avg.get(seg["speaker"], 0.0)
        if cands and cands[0][0] >= cur + 0.02:
            quality["speaker_fixes"].append({"t": seg["start"], "kind": "gender", "from": seg["speaker"],
                                             "to": cands[0][1], "words": ev,
                                             "sim": [round(cur, 2), round(cands[0][0], 2)]})
            seg["fix"] = f"род «{', '.join(ev)}» и голос: {seg['speaker']}→{cands[0][1]}"
            seg["speaker"] = cands[0][1]
            seg["label"] = label_of.get(seg["speaker"], seg["speaker"])
        else:
            alt_n = cands[0][1] if cands else "участник другого пола"
            seg.setdefault("gender_flag", []).append(
                f"«{', '.join(ev)}»: голос ближе к {seg['speaker']} ({cur:.2f}), чем к {alt_n} "
                f"({cands[0][0]:.2f})" if cands else f"«{', '.join(ev)}»: нет участника нужного пола")

    # метки сомнения
    overlaps = W.overlap_regions(full_ann)
    thr = W.doubt_flags(words, overlaps)
    for seg in segments:
        for wd in seg["words"]:
            src_w = wd.pop("_flags_src")
            if src_w.get("flags"):
                wd["flags"] = src_w["flags"]
            if not wd.get("flags"):
                wd.pop("sims", None)
    missed = W.missed_speech(words, turns) if cfg.get("review_missed", True) else []
    items = W.review_items(segments, missed=missed, name_of=label_name)
    quality["missed_speech"] = [{"t1": s, "t2": e} for s, e, _ in missed]
    for seg in segments:
        for wd in seg["words"]:
            wd.pop("sims", None)

    # эмоции — по репликам от 2 с (если модель доступна)
    if cfg.get("emotions", True) and not a.no_emotions:
        try:
            import gigaam
            import soundfile as sf
            emo = gigaam.load_model("emo", download_root=str(paths.GIGAAM_DIR))
            tmp = tmpfiles.scratch("emo_") / "piece.wav"
            for seg in segments:
                if seg["end"] - seg["start"] >= EMO_MIN:
                    s0, e0 = seg["start"], min(seg["end"], seg["start"] + 15)
                    sf.write(tmp, audio[int(s0 * SR):int(e0 * SR)], SR)
                    p = emo.get_probs(str(tmp))
                    top = max(p, key=p.get)
                    seg["emotion"] = {"label": EMO_RU.get(top, top), "score": round(float(p[top]), 2)}
        except Exception as ex:
            print(f"Эмоции отключены: {ex}", file=sys.stderr, flush=True)
            quality["emotions_error"] = f"{type(ex).__name__}: {ex}"[:300]

    labels = list(dict.fromkeys(s["label"] for s in segments))
    names = {l: label_name[l] for l in labels if l in label_name}
    quality.update(pipeline="words", voice_doubt_threshold=thr, overlaps_sec=round(sum(e - s for s, e in overlaps), 1),
                   flagged_words=sum(1 for w in words if w.get("flags")), words=len(words))
    data = {"speakers": [label_name.get(l, l) for l in labels], "speaker_labels": labels,
            "quality": quality, "review": items, "review_done": False, "edits": [], "segments": segments,
            "_timing": timing}
    if names:
        data.update(speaker_names=names, names_source="voices", names_confirmed=False)
    return data


def learn_only(a, st, cfg, src):
    """Пересчёт профилей голосов по одной расшифровке (после ответов человека): вклад записи заменяется."""
    try:
        if json.loads(Path(a.learn_only).read_text(encoding="utf-8")).get("no_profile"):
            st.set(state="done", stage="готово", progress=100, kind="learn", learned=0,
                   note="расшифровка сделана с общим словарём без group — профили групп не пополняются; "
                        "чтобы пополнять профиль, укажите group и расшифруйте запись заново")
            return
    except (OSError, ValueError):
        pass
    try:
        import voices
        import vocab as vocab_mod
        from pyannote.audio import Pipeline
        token = os.environ.get("HF_TOKEN", "").strip()
        st.set(stage="профили: загрузка модели голосов", progress=10)
        name = "pyannote/speaker-diarization-community-1"
        try:
            pipe = Pipeline.from_pretrained(name, token=token)
        except TypeError:
            pipe = Pipeline.from_pretrained(name, use_auth_token=token)
        emb = voices.Embedder(pipe)
        _, vocab, vgroup = resolve_vocab(src, a, cfg)
        allowed = [p["name"] for p in vocab.get("participants", []) if p.get("name")]
        # профиль — той группы, с которой сделана расшифровка, а не текущего словаря по умолчанию
        tg = voices.group_of(json.loads(Path(a.learn_only).read_text(encoding="utf-8")))
        group = tg or vgroup or "default"
        if tg and vgroup and tg != vgroup:
            allowed = []                           # участники словаря — другой группы: не ограничиваем
        ppath = voices.profile_path(group)
        prof = voices.load_profile(ppath)
        st.set(stage="профили: пересчёт по ответам человека", progress=40)
        learned = voices.learn(prof, Path(a.learn_only).parent, emb, load_audio, names=allowed or None,
                               only=[a.learn_only], group=group)
        voices.save_profile(ppath, prof)
        last = prof.get("history", [])[-1] if learned else None
        st.set(state="done", stage="готово", progress=100, kind="learn", learned=learned, profile=ppath.name,
               revision=prof.get("revision"), last=last,
               quarantine={n: q for qa in (prof.get("quarantine") or {}).values() for n, q in qa.items()
                           if last and q.get("from") == last.get("from")},
               warnings=[w for w in prof.get("warnings", []) if last and w.get("from") == last.get("from")])
    except Exception as ex:
        import traceback
        st.set(state="error", stage="ошибка", error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc()[-1500:])
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--speakers", type=int, default=0, help="число участников, если известно")
    ap.add_argument("--no-emotions", action="store_true")
    ap.add_argument("--status", default="", help="файл статуса (для MCP-сервера)")
    ap.add_argument("--vocab", default="", help="путь к словарю группы (rus-transcriber.vocab.json)")
    ap.add_argument("--group", default="", help="группа (папка DATA_DIR\\groups\\<группа>): её словарь и голосовой профиль")
    ap.add_argument("--tag", default="", help="метка в имени результата: <имя>.<метка>.transcript.md")
    ap.add_argument("--no-voices", action="store_true", help="без голосовых профилей и проверки голосов")
    ap.add_argument("--names", default="", help="имена участников записи через запятую: голоса только среди них")
    ap.add_argument("--learn-only", default="", help="только пересчитать профили голосов по этой расшифровке")
    ap.add_argument("--whisper", default="", choices=["", "all", "doubt", "off"],
                    help="третья модель Whisper: all — вся запись, doubt — только куски, где GigaAM RNN-T и CTC "
                         "расходятся, off — без неё")
    ap.add_argument("--whisper-max-repeat", type=int, default=-1,
                    help="сколько раз подряд Whisper может повторить одну фразу (зацикливание); 0 — без ограничения")
    a = ap.parse_args()

    st = Status(a.status)
    src = Path(a.audio).expanduser().resolve()
    st.data["file"] = str(src)
    cfg = load_config()
    wait_slot(st, cfg, a.status)
    tmpfiles.sweep_stale()          # остатки прежних принудительно завершённых задач (слот у нас — чужих тяжёлых задач нет)
    t0 = time.time()
    if a.learn_only:
        return learn_only(a, st, cfg, src)
    try:
        if not src.exists():
            raise FileNotFoundError(f"Файл не найден: {src}")
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("Не задан HF_TOKEN")

        from backends import make_asr
        import vocab as vocab_mod

        tag = "".join(ch for ch in a.tag if ch.isalnum() or ch in "-_")
        stem = src.stem + (f".{tag}" if tag else "")
        out_dir = src.parent / f"{src.stem}.rus-transcriber"   # все файлы результата — в подпапке рядом с аудио
        out_dir.mkdir(exist_ok=True)
        md_path = out_dir / (stem + ".transcript.md")
        json_path = out_dir / (stem + ".transcript.json")

        vocab_path, vocab, vgroup = resolve_vocab(src, a, cfg)
        # общий словарь без "group" (и группа не задана явно): словарь применяется, профили голосов групп — нет
        no_profile = bool(vocab_path) and not vgroup
        genders_by_name = vocab_mod.genders(vocab)
        allowed = list(genders_by_name) or [p["name"] for p in vocab.get("participants", []) if p.get("name")]
        names = [n.strip() for n in (a.names or "").split(",") if n.strip()]
        all_names = list(allowed)
        if names:                                  # участники этой записи известны заранее
            genders_by_name = {n: g for n, g in genders_by_name.items() if n in names}
            allowed = names
        names_outside = [n for n in names if all_names and n not in all_names]

        st.set(stage="конвертация звука", progress=2)
        audio = load_audio(src)
        duration = len(audio) / SR

        st.set(stage="разделение по спикерам", progress=5, duration=hms(duration))
        # участники известны: не меньше их числа и запас на один посторонний голос; точное число — только --speakers
        max_spk = len(allowed) + 1 if allowed and not a.speakers else 0
        min_spk = len(names) if names and not a.speakers else 0
        raw, diar_info, pipe, full_ann = diarize(audio, a.speakers, token, cfg, st, max_speakers=max_spk,
                                                 min_speakers=min_spk)
        order = {}
        for _, _, spk in sorted(raw):
            order.setdefault(spk, f"Спикер {len(order) + 1}")
        turns = [{"s": s, "e": e, "label": order[spk]} for s, e, spk in build_turns(raw)]
        t_diar = time.time() - t0

        # ---- голоса: профиль, разрезание реплик, короткие реплики, имена ----
        quality = {"vocab": str(vocab_path) if vocab_path else None, "speaker_fixes": [], "vocab_fixes": [],
                   "review": []}
        label_name, label_of, refs, emb, t_voice = {}, {}, {}, None, 0.0
        voice_ctx = {}
        pipeline = cfg.get("pipeline", "words")
        if not a.no_voices and cfg.get("voices", True):
            tv = time.time()
            try:
                import voices
                st.set(stage="голоса: профили участников", progress=30)
                emb = voices.Embedder(pipe)
                if no_profile:
                    group, ppath, learned = None, None, 0
                    prof = voices.load_profile(paths.DATA_DIR / "__no_profile__.json")     # пустой, не сохраняется
                    quality["voices_off"] = ("общий словарь без поля group: профили голосов групп не применяются "
                                             "и не пополняются (имена — rename_speakers)")
                else:
                    group = vgroup or "default"
                    ppath = voices.profile_path(group)
                    prof = voices.load_profile(ppath)
                    learned = voices.learn(prof, src.parent, emb, load_audio, names=all_names or None, group=group)
                    if learned:
                        voices.save_profile(ppath, prof)
                st.set(stage="голоса: проверка реплик", progress=33)
                embs = emb.spans(audio, [(t["s"], t["e"]) for t in turns])
                if cfg.get("voices_debug"):   # отпечатки реплик для подбора порогов (биометрия: только локально)
                    dbg = paths.JOBS / f"{stem}.voices_debug.json"
                    dbg.write_text(json.dumps({"turns": [{**t, "vec": None if v is None else [round(float(x), 4) for x in v]}
                                                         for t, v in zip(turns, embs)]}, ensure_ascii=False), encoding="utf-8")
                cents = voices.centroids(turns, embs)
                pvecs = voices.voice_vectors(prof, exclude_audio=src)
                mapping = voices.match(cents, pvecs, allowed or None)
                refs, label_of = voices.references(cents, turns, mapping)
                # кластеры одного человека сводим к одной метке
                for t in turns:
                    nm = mapping.get(t["label"], (None,))[0]
                    if nm and label_of.get(nm) != t["label"]:
                        t["label"] = label_of[nm]
                label_name = {lbl: nm for nm, lbl in label_of.items()}
                voice_ctx.update(pvecs=pvecs, cents=cents, mapping=mapping,
                                 confusions=voices.confusions(prof, exclude_audio=src))
                # голоса записи почти не похожи на группу — возможно, это другая группа с тем же словарём
                dur = {}
                for tt in turns:
                    dur[tt["label"]] = dur.get(tt["label"], 0) + tt["e"] - tt["s"]
                tot = sum(dur.values()) or 1.0
                share = sum(d for l, d in dur.items() if l in mapping) / tot
                if len(pvecs) >= 2 and share < 0.5:
                    quality["group_doubt"] = {
                        "group": group, "matched_share": round(share, 2),
                        "note": f"голоса записи почти не совпадают с профилями группы «{group}» (узнано "
                                f"{share:.0%} речи) — возможно, другая группа: нужен её словарь с другим group"}
                quality.update(voice_profile=ppath.name if ppath else None, profile_names=sorted(pvecs), learned_from=learned,
                               match={lbl: {"name": nm, "sim": sim} for lbl, (nm, sim) in mapping.items()},
                               unmatched=[l for l in cents if l not in mapping],
                               cluster_sims={l: {n: round(float(cents[l] @ v), 3) for n, v in pvecs.items()}
                                             for l in cents})
                if pipeline == "words":
                    raise _SkipSegmentSteps
                turns = voices.scan_split(audio, turns, refs, label_name, label_of, emb, quality["speaker_fixes"])
                embs = emb.spans(audio, [(t["s"], t["e"]) for t in turns])
                debug = []
                used = {t["label"] for t in turns}

                def new_label(name):
                    k = 1
                    while f"Спикер {k}" in used:
                        k += 1
                    lbl = f"Спикер {k}"
                    used.add(lbl)
                    label_of[name], label_name[lbl] = lbl, name
                    refs.setdefault(name, pvecs[name])
                    return lbl

                pv = {n: v for n, v in pvecs.items() if not allowed or n in allowed}
                turns = voices.reassign(turns, embs, pv, label_name, label_of, quality["speaker_fixes"], debug,
                                        new_label=new_label)
                quality["debug_sims"] = debug
            except _SkipSegmentSteps:
                pass
            except Exception as ex:
                quality["voices_error"] = f"{type(ex).__name__}: {ex}"[:300]
                print(f"[voices] проверка голосов отключена: {ex}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            t_voice = time.time() - tv
        turns = [t for t in turns if t["e"] - t["s"] >= MIN_SEG]

        if pipeline == "words":
            if emb is None:
                import voices
                emb = voices.Embedder(pipe)
            data = run_words(audio, src, cfg, st, raw, order, turns, full_ann, emb, label_name, label_of,
                             voice_ctx, allowed, genders_by_name, vocab, quality, make_asr, a)
            timing = {"total_sec": round(time.time() - t0), "diarization_sec": round(t_diar),
                      "voices_sec": round(t_voice, 1), **data.pop("_timing"), "diarization": diar_info,
                      "embeddings_sec": round(emb.seconds, 1),
                      "realtime_factor": round(duration / max(time.time() - t0, 1e-6), 2)}
            data.update(file=str(src), duration=hms(duration), timing=timing,
                        group=None if no_profile else (vgroup or "default"))
            if no_profile:
                data["no_profile"] = True
            if names_outside:
                quality["names_outside_vocab"] = names_outside
            transcript.save(data, json_path, md_path)
            st.set(state="done", stage="готово", progress=100, markdown=str(md_path), json=str(json_path),
                   timing=timing, quality={"pipeline": "words", "vocab": quality["vocab"],
                                           "names": data.get("speaker_names") or None,
                                           "review_items": len(data.get("review", [])),
                                           "emotions_error": quality.get("emotions_error"),
                                           "speaker_fixes": len(quality["speaker_fixes"]),
                                           "vocab_fixes": sum(len(v["fixes"]) for v in quality["vocab_fixes"]),
                                           "voices_error": quality.get("voices_error"),
                                           "group_doubt": quality.get("group_doubt"),
                                           "names_outside_vocab": quality.get("names_outside_vocab")},
                   hint="Разбор сомнительных мест: review_items(job_id), затем apply_review(job_id, decisions)")
            return

        st.set(stage="загрузка модели распознавания", progress=36)
        asr, asr_errors = make_asr(cfg)
        emo = None
        if cfg.get("emotions", True) and not a.no_emotions:
            try:
                import gigaam
                emo = gigaam.load_model("emo", download_root=str(paths.GIGAAM_DIR))
            except Exception as ex:
                print(f"Эмоции отключены: {ex}", file=sys.stderr, flush=True)
        st.set(stage="распознавание", progress=38, asr_backend=asr.name,
               asr_fallbacks=asr_errors or None)

        chunks = [(i, cs, ce) for i, t in enumerate(turns) for cs, ce in split_at_pauses(audio, t["s"], t["e"])]
        texts = {i: [] for i in range(len(turns))}
        emos = {i: [] for i in range(len(turns))}
        emo_tmp = tmpfiles.scratch("emo_") / "piece.wav"
        t1 = time.time()
        pieces = [audio[int(cs * SR):int(ce * SR)] for _, cs, ce in chunks]
        last = [0.0]

        def progress(done, total):
            if time.time() - last[0] > 1 or done == total:
                last[0] = time.time()
                st.set(stage="распознавание", progress=38 + int(55 * done / total), done=f"{done}/{total}")

        recognized = asr.transcribe_many(pieces, progress=progress)
        t_asr = time.time() - t1
        for k, (i, cs, ce) in enumerate(chunks):
            piece = pieces[k]
            if recognized[k]:
                texts[i].append(recognized[k])
            if emo is not None and ce - cs >= EMO_MIN:
                try:
                    import soundfile as sf
                    sf.write(emo_tmp, piece, SR)
                    emos[i].append((ce - cs, emo.get_probs(str(emo_tmp))))
                except Exception as ex:
                    print(f"Эмоции отключены: {ex}", file=sys.stderr, flush=True)
                    emo = None
            if emo is not None and k % 10 == 0:
                st.set(stage="эмоции", progress=93 + int(5 * (k + 1) / len(chunks)))

        fixer = vocab_mod.Fixer(vocab) if vocab else None
        segments, kept = [], []
        for i, t in enumerate(turns):
            text = " ".join(texts[i]).strip()
            if not text:
                continue
            seg = {"start": round(t["s"], 2), "end": round(t["e"], 2), "label": t["label"],
                   "speaker": label_name.get(t["label"], t["label"]), "text": text}
            if t.get("fix"):
                fx_text = t["fix"]
                for lbl, nm in sorted(label_name.items(), key=lambda x: -len(x[0])):
                    fx_text = fx_text.replace(lbl, nm)
                seg["fix"] = fx_text
            if fixer:
                new, fx = fixer.apply(text)
                if fx:
                    seg["text_raw"], seg["text"] = text, new
                    quality["vocab_fixes"].append({"t": seg["start"], "fixes": fx})
            if emos[i]:
                total = sum(w for w, _ in emos[i])
                avg = {}
                for w, p in emos[i]:
                    for lbl, v in p.items():
                        avg[lbl] = avg.get(lbl, 0) + float(v) * w / total
                top = max(avg, key=avg.get)
                seg["emotion"] = {"label": EMO_RU.get(top, top), "score": round(avg[top], 2),
                                  "probs": {k2: round(v, 3) for k2, v in avg.items()}}
            segments.append(seg)
            kept.append(t)

        # ---- род глагола против пола говорящего ----
        if label_name and genders_by_name and emb is not None:
            try:
                seg_embs = emb.spans(audio, [(t["s"], t["e"]) for t in kept])
                apply_gender(segments, seg_embs, refs, label_of, genders_by_name,
                             quality["speaker_fixes"], quality["review"])
            except Exception as ex:
                quality["gender_error"] = f"{type(ex).__name__}: {ex}"[:300]
        for f in quality["speaker_fixes"]:   # метки кластеров -> имена для читаемости
            for key in ("from", "to"):
                f[key] = label_name.get(f[key], f[key])

        labels = list(dict.fromkeys(s["label"] for s in segments))
        speakers = [label_name.get(l, l) for l in labels]
        timing = {"total_sec": round(time.time() - t0), "diarization_sec": round(t_diar),
                  "voices_sec": round(t_voice, 1), "asr_sec": round(t_asr), "asr_chunks": len(chunks),
                  "diarization": diar_info, "asr_backend": asr.name,
                  "asr_model": getattr(asr, "model_name", cfg.get("asr_model")),
                  "realtime_factor": round(duration / max(time.time() - t0, 1e-6), 2)}
        if emb is not None:
            timing["embeddings_sec"] = round(emb.seconds, 1)
        data = {"file": str(src), "duration": hms(duration), "speakers": speakers, "speaker_labels": labels,
                "timing": timing, "quality": quality, "segments": segments,
                "group": None if no_profile else (vgroup or "default")}
        if no_profile:
            data["no_profile"] = True
        if names_outside:
            quality["names_outside_vocab"] = names_outside
        if label_name:
            data.update(speaker_names={l: label_name[l] for l in labels if l in label_name},
                        names_source="voices", names_confirmed=False)
        transcript.save(data, json_path, md_path)

        st.set(state="done", stage="готово", progress=100, markdown=str(md_path),
               json=str(json_path), timing=timing,
               quality={"vocab": quality["vocab"], "names": label_name or None,
                        "speaker_fixes": len(quality["speaker_fixes"]),
                        "vocab_fixes": sum(len(v["fixes"]) for v in quality["vocab_fixes"]),
                        "review": len(quality["review"]),
                        "voices_error": quality.get("voices_error")})
    except Exception as ex:
        st.set(state="error", stage="ошибка", error=f"{type(ex).__name__}: {ex}",
               trace=traceback.format_exc()[-3000:])
        sys.exit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
