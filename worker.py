"""Обработка одной записи: диаризация (pyannote) + распознавание (GigaAM) + эмоции (GigaAM-Emo).

Запуск вручную:
    .venv\\Scripts\\python.exe worker.py "C:\\путь\\к\\записи.mp3"
    .venv\\Scripts\\python.exe worker.py запись.m4a --speakers 3 --no-emotions

Устройство и модель берутся из config.json (его заполняет bench.py).
Новая запись сохраняется только в базу записей (state.sqlite3 папки данных; создаёт её установщик
или `rt.py db-init`), её документы — рядом с аудио в папке <имя>.rus-transcriber: <имя>.transcript.md
и <имя>.transcript.json. Без базы worker отказывает до тяжёлой работы.
Автор записи: программа передаёт его в файле статуса (--status); при ручном запуске без --status
worker берёт его из базы сам, как программа в момент запуска.
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
import ast
import uuid
from datetime import datetime, timezone
from importlib import metadata
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
import download_hint  # noqa: E402
import transcript  # noqa: E402
import words as W  # noqa: E402  (пословная обработка: нарезка, сведение гипотез, голос по словам)
from transcript import hms  # noqa: E402

HERE = Path(__file__).resolve().parent
from settings import DEFAULTS, load_config  # noqa: E402,F401  (настройки: умолчания в коде + DATA_DIR\config.json)
import settings

SR = 16000
MAX_CHUNK = 22.0   # GigaAM распознаёт куски до ~25 с
MERGE_GAP = 1.0    # склеиваем реплики одного спикера, если пауза меньше
MIN_SEG = 0.3      # совсем короткие обрывки отбрасываем
EMO_MIN = 2.0      # эмоцию считаем только на фрагментах от 2 с
GENDER_MARGIN = 0.05  # род глагола переносит реплику, только если и голос ближе к новому участнику

EMO_RU = {"angry": "раздражение", "sad": "грусть", "neutral": "нейтрально", "positive": "позитив"}

def asr_run_metadata():
    """Компактное описание запуска без загрузки сервера и обращения к сети."""
    try:
        import buildinfo
        build = buildinfo.identity(HERE)
    except Exception:
        build = {"release": None, "commit": None, "source_state": "unknown",
                 "payload_sha256": None, "manifest_status": "unknown"}
    version = None
    try:
        tree = ast.parse((HERE / "server.py").read_text(encoding="utf-8"))
        version = next(ast.literal_eval(n.value) for n in tree.body
                       if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VERSION"
                                                            for t in n.targets))
    except (OSError, SyntaxError, ValueError, StopIteration):
        pass
    import runinfo
    packages = {}
    for name in runinfo.KEY_PACKAGES:      # те же ключевые пакеты, что в тесте скорости и файле гипотез
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "program": version, "build": build, "packages": packages, "models": {}}


def wait_slot(st, cfg, status_path):
    """Не больше max_heavy_jobs тяжёлых задач одновременно (jobslots.py); время в очереди в задачу не входит."""
    import jobslots
    job_id = Path(status_path).stem if status_path else None
    t = time.time()
    st._slot = jobslots.acquire(paths.JOBS, cfg.get("max_heavy_jobs", 2), status=st, job_id=job_id)
    now = round(time.time(), 1)
    st.set(state="running", stage="старт", progress=0, started_ts=now,
           queued_sec=round(time.time() - t, 1) if time.time() - t > 3 else None)


def resolve_vocab_details(src, a, cfg, transcript_path=None, group_override=None, transcript_data=None):
    """Разрешает словарь только по ограниченным путям и возвращает отчёт без содержимого словарей.

    transcript_data — уже прочитанные данные источника (у записи в базе файл transcript_path — выгрузка
    другого вида, привязка группы и словаря берётся из ревизии)."""
    import vocab as vocab_mod

    binding = None
    if transcript_path:
        if transcript_data is None:
            try:
                transcript_data = json.loads(Path(transcript_path).read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError, json.JSONDecodeError) as ex:
                raise RuntimeError("Не удалось прочитать исходную расшифровку для проверки группы") from ex
        binding = vocab_mod.transcript_binding(transcript_data)
        if binding["conflict"]:
            raise vocab_mod.ResolutionError(
                "Группа в расшифровке противоречит группе словаря quality.vocab",
                {"binding": binding, "candidates": []})

    explicit_group = getattr(a, "group", "") or ""
    explicit_vocab = getattr(a, "vocab", "") or ""
    expected = explicit_group or (binding or {}).get("group")
    if explicit_group and (binding or {}).get("group") and \
            paths.safe_group(explicit_group) != paths.safe_group(binding["group"]):
        raise vocab_mod.ResolutionError(
            "Явная группа противоречит группе исходной расшифровки",
            {"binding": binding, "candidates": []})

    if group_override:
        if not transcript_path or transcript_data is None:
            raise vocab_mod.ResolutionError(
                "Ручной выбор группы возможен только для конкретной расшифровки",
                {"candidates": []})
        if paths.safe_group(group_override) != group_override:
            raise vocab_mod.ResolutionError("Недопустимое имя выбранной группы", {"candidates": []})
        if explicit_group and paths.safe_group(explicit_group) != group_override:
            raise vocab_mod.ResolutionError("Параметры группы противоречат друг другу", {"candidates": []})
        vp, data, group, report, _ = vocab_mod.resolve_group_vocab(
            src, transcript_path, transcript_data, group_override, paths.vocab_path(group_override))
        return {"path": vp, "vocab": data, "group": group, "report": report}

    if explicit_vocab:
        candidates = [(Path(explicit_vocab).expanduser(), "explicit_vocab")]
        vp, data, group, report = vocab_mod.resolve_candidates(
            candidates, expected_group=expected, allow_ungrouped=True)
        if report["candidates"][0]["status"] == "missing":
            raise vocab_mod.ResolutionError("Явно указанный словарь не найден", report)
        if explicit_group:
            group = explicit_group
        report["binding"] = binding
        return {"path": vp, "vocab": data, "group": group, "report": report}

    if explicit_group:
        candidates = [(paths.vocab_path(explicit_group), "explicit_group_vocab")]
        vp, data, group, report = vocab_mod.resolve_candidates(
            candidates, expected_group=explicit_group, allow_ungrouped=True)
        report["binding"] = binding
        return {"path": vp, "vocab": data, "group": explicit_group, "report": report}

    candidates = vocab_mod.related_candidates(
        src, transcript_path, (binding or {}).get("vocab_path"))

    # Уникальная/настроенная группа — fallback только для новой расшифровки.
    # При обучении старой записи текущий default_group не доказывает её принадлежность.
    if transcript_path:
        expected = (binding or {}).get("group")
        has_existing = any(Path(path).expanduser().is_file() for path, _ in candidates)
        if not expected and has_existing:
            _, _, expected, _ = vocab_mod.resolve_candidates(candidates)
        if expected:
            candidates.append((paths.vocab_path(expected), "group_fallback"))
        vp, data, group, report = vocab_mod.resolve_candidates(
            candidates, expected_group=expected)
        report["binding"] = binding
        return {"path": vp, "vocab": data, "group": group, "report": report}

    # Sidecar может установить группу, но не выбирается автоматически вместо второго
    # словаря той же группы: оба точечных источника попадут в отчёт и дадут неоднозначность.
    existing = any(Path(path).expanduser().is_file() for path, _ in candidates)
    if existing:
        _, _, group, _ = vocab_mod.resolve_candidates(candidates)
        if group:
            candidates.append((paths.vocab_path(group), "group_fallback"))
        vp, data, group, report = vocab_mod.resolve_candidates(candidates, expected_group=group)
        return {"path": vp, "vocab": data, "group": group, "report": report}

    group = paths.pick_group(None, cfg.get("default_group"))
    if group:
        candidates.append((paths.vocab_path(group), "group_fallback"))
    vp, data, group, report = vocab_mod.resolve_candidates(
        candidates, expected_group=group)
    return {"path": vp, "vocab": data, "group": group, "report": report}


def resolve_vocab(src, a, cfg, transcript_path=None):
    """Совместимый краткий интерфейс: (путь словаря, словарь, группа)."""
    result = resolve_vocab_details(src, a, cfg, transcript_path=transcript_path)
    return result["path"], result["vocab"], result["group"]


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
        import jobslots
        import tempfile
        with jobslots.file_lock(self.path.with_suffix(".status.lock")):
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=self.path.name + ".", suffix=".tmp", delete=False) as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
                tmp = Path(f.name)
            try:
                for k in range(20):      # Windows: замена не проходит, пока читатель держит файл — повторяем
                    try:
                        os.replace(tmp, self.path)
                        if self.data.get("state") in ("done", "error", "cancelled"):
                            self.path.with_suffix(".pid").unlink(missing_ok=True)
                        return
                    except PermissionError:
                        time.sleep(0.05 * (k + 1))
                if self.data.get("state") in ("done", "error"):
                    raise RuntimeError(f"не удалось записать статус {self.path}")
            finally:
                tmp.unlink(missing_ok=True)

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
                          ov_device=cfg.get("diar_ov_device", "CPU"),
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


def link_missed_diagnostics(items, trace):
    """Связать review-предупреждения с точными кандидатами трассы по округлённым границам."""
    warning_ids = {}
    for candidate in trace.get("candidates") or []:
        if candidate.get("decision") == "warning":
            key = (round(candidate["start"], 2), round(candidate["end"], 2))
            warning_ids.setdefault(key, []).append(candidate["id"])
    for item in items:
        if "missed" not in (item.get("reasons") or []):
            continue
        key = (round(item.get("t1", item["t"]), 2), round(item.get("t2", item["t"]), 2))
        candidates = warning_ids.get(key) or []
        if candidates:
            item["diagnostic_id"] = candidates.pop(0)
    return items


def _progress(st):
    """Ход этапа в статусе: не чаще раза в секунду и в конце этапа. Счётчик общий для всех проходов."""
    last = [0.0]

    def prog(lo, hi, name):
        def f(done, total):
            if time.time() - last[0] > 1 or done == total:
                last[0] = time.time()
                st.set(stage=name, progress=lo + int((hi - lo) * done / total), done=f"{done}/{total}")
        return f
    return prog


def _register_pass(quality, name, model_id, backend, chunks):
    """Сведения прохода модели: движок и диагностика кусков, модель в описании запуска."""
    quality["asr_diagnostics"][name] = {"backend": backend, "chunks": chunks}
    quality["asr_run"]["models"][name] = {"id": model_id, "backend": backend, "revision": None}


def _failed_records(chunks, attempts, selected=None):
    """Сведения о проходе, который не дал результата: каждый кусок — ошибка с этими попытками.
    selected — номера кусков, отданных модели (Whisper в режиме doubt); тогда пишется и chunk_index."""
    out = []
    for i, k in enumerate(range(len(chunks)) if selected is None else selected):
        rec = {"index": i}
        if selected is not None:
            rec["chunk_index"] = k
        rec.update(state="error", audio_span=[chunks[k]["a"], chunks[k]["b"]], own_span=list(chunks[k]["own"]),
                   attempts=[dict(x) for x in attempts])
        out.append(rec)
    return out


def timestamped_pass(name, asr, audio, chunks, quality, model_id, progress):
    """Проход модели со временами слов по кускам -> гипотеза: слова {"w", "s", "e", "c"}.
    Сведения о сбоях кусков — в quality["asr_diagnostics"][name]."""
    diagnostics = []
    _register_pass(quality, name, model_id, asr.name, diagnostics)
    return W.transcribe(asr, audio, chunks, progress=progress, diagnostics=diagnostics)


def primary_pass(audio, chunks, cfg, st, quality, make_asr, prog, timing):
    """Основной проход (GigaAM RNN-T): его слова, времена и пунктуация — основа расшифровки.
    Сбой загрузки модели останавливает обработку."""
    st.set(stage="загрузка модели распознавания", progress=36)
    asr, asr_errors = make_asr(cfg)
    if not hasattr(asr, "transcribe_timestamped"):
        raise RuntimeError(f"бэкенд {asr.name} не отдаёт времена слов; задайте \"pipeline\": \"segments\"")
    t1 = time.time()
    st.set(stage="распознавание", progress=38, asr_backend=asr.name, asr_fallbacks=asr_errors or None)
    quality.setdefault("asr_diagnostics", {})
    quality.setdefault("asr_run", asr_run_metadata())
    words = timestamped_pass("primary", asr, audio, chunks, quality,
                             getattr(asr, "model_name", cfg.get("asr_model")), prog(38, 62, "распознавание"))
    timing.update(asr_sec=round(time.time() - t1), asr_chunks=len(chunks), asr_backend=asr.name,
                  asr_model=getattr(asr, "model_name", cfg.get("asr_model")))
    return words


def second_pass(audio, chunks, cfg, quality, make_asr, prog, timing):
    """Второй проход (по умолчанию GigaAM CTC) по тем же кускам -> слова со временами или None
    (не задан, совпадает с основной, не даёт времён слов или сбой).

    Вторая модель может быть любой, если она отдаёт времена слов (transcript.asr_model_caps): по ним
    голосование ставит её слова на место и исключает сбойные отрезки. Модель без времён (Whisper)
    вторым номером не считается — одна ясная строка в журнале и quality["asr2_skipped"], без подмены
    модели; модель, совпадающая с основной, новой гипотезы не дала бы — тоже asr2_skipped.

    Совпадение второй с основной — по имени, как раньше (в отличие от третьей, _third_same_model):
    точность вторая наследует у основной (asr_quant, asr_quant_cpu), отличаться может лишь движок
    или устройство, а решение владельца 22.11.2026 касалось только третьей модели."""
    second = cfg.get("asr_second_model", "v3_e2e_ctc")
    if not second:
        return None
    label = transcript.model_label(second)
    if second == cfg.get("asr_model"):
        quality["asr2_skipped"] = {"reason": "same_as_primary", "model": second}
        print(f"[words] вторая модель {second} совпадает с основной: второй проход пропущен — он не дал бы "
              "новой гипотезы; сверки и голосования моделей не будет", file=sys.stderr, flush=True)
        return None
    if transcript.asr_model_caps(second)["word_times"] is False:
        quality["asr2_skipped"] = {"reason": "no_word_times", "model": second}
        print(f"[words] вторая модель {second} ({label}) не даёт времён слов, а второй модели они нужны "
              "(голосование ставит её слова по времени): второй проход пропущен, сверки и голосования "
              "моделей не будет. Такую модель ставьте третьей (asr_third_model), второй — модель со "
              "временами слов (например v3_e2e_ctc)", file=sys.stderr, flush=True)
        return None
    try:
        t2 = time.time()
        from backends import backend_for_device, normalize_device
        main_dev = normalize_device(cfg.get("asr_device", "CPU"))
        sec_dev = normalize_device(cfg.get("asr_second_device") or main_dev)
        sec_backend = str(cfg.get("asr_second_backend") or "auto")
        if sec_backend in WHISPER_ONLY_ENGINES:
            # движки только для Whisper второй модели не подходят: make_asr считал бы её общим
            # onnx-путём под чужим именем движка
            print(f"[words] вторая модель {second}: движок {sec_backend} умеет только Whisper — "
                  "выбираю движок по устройству", file=sys.stderr, flush=True)
            sec_backend = "auto"
        if sec_backend == "auto":
            # своё устройство — свой бэкенд; если устройство то же, доверяем основному
            sec_backend = (cfg.get("asr_backend", "auto") if sec_dev == main_dev
                           else backend_for_device(sec_dev))
        second_cfg = {**cfg, "asr_model": second, "asr_backend": sec_backend,
                      "asr_device": sec_dev,
                      "asr_batch": int(cfg.get("asr_second_batch") or cfg.get("asr_batch", 1))}
        asr2, _ = make_asr(second_cfg)
        alt = timestamped_pass("second", asr2, audio, chunks, quality, getattr(asr2, "model_name", second),
                               prog(62, 68, f"второй проход распознавания ({label})"))
        del asr2
        timing.update(asr2_sec=round(time.time() - t2), asr2_model=second)
        return alt
    except Exception as ex:
        quality["asr2_error"] = f"{type(ex).__name__}: {ex}"[:300]
        print(f"[words] второй проход ({label}) не выполнен: {ex}", file=sys.stderr, flush=True)
        _register_pass(quality, "second", second, None,
                       _failed_records(chunks, [{"result": "error", "error_type": type(ex).__name__}]))
        return None


def _engine_key(name):
    """Имя движка загруженной модели -> (движок, устройство, точность) без размера пакета.
    «cpu:b4:int8» -> ("cpu", "int8"), «openvino:GPU:b8» -> ("openvino", "GPU"), «cpu» (fp32) -> ("cpu",).
    Формат имени — backends.OnnxASR.name; размер пакета на загруженную модель не влияет."""
    return tuple(p for p in str(name or "").split(":") if p and not re.fullmatch(r"b\d+", p))


def _third_engine_key(third, cfg):
    """Движок, на котором будет загружена третья модель не из семейства Whisper: только onnx-cpu
    с точностью asr_third_quant_cpu (whisper_pass) -> ключ как у _engine_key; для Whisper — None."""
    if transcript.asr_model_caps(third)["whisper"]:
        return None
    quant = cfg.get("asr_third_quant_cpu", "int8")
    return ("cpu", quant) if quant else ("cpu",)


def _third_same_model(third, cfg, quality):
    """Третья модель совпадает с основной или второй -> причина пропуска (голос без новой информации).

    «Совпадает» — то же имя модели и та же фактическая загрузка: движок, устройство и точность
    (квантование), с которыми основная или вторая модель действительно загружена (её движок
    в quality["asr_diagnostics"], с учётом откатов), против движка третьей (onnx-cpu,
    asr_third_quant_cpu). Размер пакета не в счёт. Такая третья повторила бы гипотезу: совпадая
    с основной, всегда поддерживала бы её и снимала бы метки расхождения со второй; совпадая со
    второй, удваивала бы её голос. Тогда третий проход не считается, сводятся две модели (как при
    asr_third_mode=off). Если отличается хоть что-то — модель, движок, устройство или точность —
    третья считается (решение владельца 22.11.2026)."""
    if quality is None:
        return None
    mine = _third_engine_key(third, cfg)
    if mine is None:
        return None
    diag = quality.get("asr_diagnostics") or {}
    reason = None
    for role, key, why in (("primary", "asr_model", "same_as_primary"),
                           ("second", "asr_second_model", "same_as_second")):
        if cfg.get(key) and third == cfg.get(key) and \
                _engine_key((diag.get(role) or {}).get("backend")) == mine:
            reason = why
            break
    if reason is None:
        return None
    quality["asr3_skipped"] = {"reason": reason, "model": third, "engine": ":".join(mine)}
    print(f"[words] третья модель {third} совпадает с " + ("основной" if reason == "same_as_primary" else "второй")
          + f" и загружалась бы так же ({':'.join(mine)}): третий проход пропущен — он не дал бы нового "
          "голоса; сверяются основная и вторая модели", file=sys.stderr, flush=True)
    return reason


def whisper_plan(audio, regions, chunks, cfg, a, words, alt, quality=None):
    """Что считает третья модель: модель (None — выключена или совпадает с основной либо второй),
    режим, предел повторов и куски. Правило нарезки — по возможностям модели (transcript.asr_model_caps)."""
    third = cfg.get("asr_third_model", "onnx-community/whisper-large-v3-turbo")
    wmode = (getattr(a, "whisper", "") or cfg.get("asr_third_mode", "all")).lower()
    wrep = a.whisper_max_repeat if getattr(a, "whisper_max_repeat", -1) >= 0 else int(cfg.get("asr_third_max_repeat", 3))
    if wmode == "off":
        third = None
    if third and alt is not None and _third_same_model(third, cfg, quality):
        third = None
    # куски для Whisper — свои, до 28 с (он всегда считает окно 30 с); другой модели (GigaAM CTC,
    # Multilingual) — куски основного прохода: длинные куски Whisper ей не нужны. В режиме doubt —
    # только куски, где основная и вторая модели расходятся: в остальных голосование решено без третьей
    long_chunks = bool(cfg.get("asr_third_long_chunks", True))
    if third and long_chunks and not transcript.asr_model_caps(third)["whisper"]:
        print(f"[words] третья модель {third}: длинные куски Whisper (asr_third_long_chunks) к ней не "
              "применяются — куски как у основного прохода", file=sys.stderr, flush=True)
        long_chunks = False
    wchunks = W.make_chunks(audio, regions, max_len=W.WHISPER_MAX, min_len=W.WHISPER_MIN) \
        if long_chunks else chunks
    selected = list(range(len(wchunks)))
    if third and alt is not None and wmode == "doubt":
        dt = W.disagreement_times(words, alt)
        selected = [k for k, c in enumerate(wchunks) if any(c["own"][0] <= x < c["own"][1] for x in dt)]
    pieces = [audio[int(wchunks[k]["a"] * SR):int(wchunks[k]["b"] * SR)] for k in selected]
    return {"model": third, "mode": wmode, "max_repeat": wrep, "chunks": wchunks, "selected": selected,
            "pieces": pieces}


def _third_label(third, backend=""):
    """Короткое название третьей модели для статуса обработки: Whisper, Multilingual large или имя модели
    (правило общее со стенограммой — transcript.third_model_label)."""
    return transcript.third_model_label(third, backend)


def _load_whisper(backend, device, quant, third, cfg, make_asr):
    """Третья модель выбранным движком -> (модель, откаты движка)."""
    if backend == "whisper-ov":
        from backends import WhisperOV
        return WhisperOV(third, device=device, quant=quant, language="ru",
                         batch=int(cfg.get("asr_third_batch", 4))), None
    if backend == "whisper-ct2":
        from backends import WhisperCT2
        return WhisperCT2(cpu_threads=int(cfg.get("torch_threads") or 0)), None
    return make_asr({**cfg, "asr_model": third, "asr_backend": backend, "asr_device": device,
                     "asr_quant": quant, "asr_quant_cpu": quant, "asr_language": "ru",
                     "asr_batch": int(cfg.get("asr_third_batch", 4))})


def _retry_whisper_on_cpu(failed_indices, texts, whisper_diag, pieces, third, cfg, st, make_asr, prog):
    """Повтор ошибочных фрагментов на onnx-cpu; сведения о попытках дописываются к фрагменту."""
    try:
        st.set(stage="третья модель: повтор ошибочных фрагментов на CPU", progress=74)
        cpu, _ = make_asr({**cfg, "asr_model": third, "asr_backend": "onnx-cpu",
                           "asr_device": "CPU", "asr_quant": cfg.get("asr_third_quant_cpu", "int8"),
                           "asr_quant_cpu": cfg.get("asr_third_quant_cpu", "int8"),
                           "asr_language": "ru", "asr_batch": int(cfg.get("asr_third_batch", 4))})
        cpu_diag = []
        cpu_texts = cpu.transcribe_texts([pieces[i] for i in failed_indices], diagnostics=cpu_diag,
                                        progress=prog(74, 74, f"повтор {_third_label(third)} на CPU"))
        if len(cpu_texts) != len(failed_indices) or len(cpu_diag) != len(failed_indices):
            raise ValueError("неверное количество результатов CPU")
        for pos, i in enumerate(failed_indices):
            rec, replacement = whisper_diag[i], cpu_diag[pos]
            for attempt in replacement["attempts"]:
                attempt["backend"] = "onnx-cpu:CPU"
                attempt["batch"] = [failed_indices[j] for j in attempt["batch"]]
            rec["attempts"].extend(replacement["attempts"])
            rec["state"] = replacement["state"]
            texts[i] = cpu_texts[pos]
        del cpu
    except Exception as ex:
        for i in failed_indices:
            whisper_diag[i]["attempts"].append(
                {"result": "error", "backend": "onnx-cpu:CPU",
                 "error_type": type(ex).__name__, "fallback": True})


def _whisper_words(plan, texts, whisper_diag, words, capture=None):
    """Текст третьей модели по кускам -> слова гипотезы и число убранных. Чистка «субтитров» и
    зацикливания (words.clean_whisper) — только для Whisper; текст другой модели берётся как есть."""
    by_k = dict(zip(plan["selected"], texts))
    diag_by_k = dict(zip(plan["selected"], whisper_diag))
    whisper = transcript.asr_model_caps(plan["model"])["whisper"]
    wh_words, removed = [], 0
    for k, c in enumerate(plan["chunks"]):
        if k in by_k:
            tx, n = W.third_text(by_k[k], plan["max_repeat"], whisper)
            rec = diag_by_k[k]
            rec.update(chunk_index=k, audio_span=[c["a"], c["b"]], own_span=list(c["own"]),
                       raw_words=len(by_k[k].split()), cleaned_words=len(tx.split()),
                       removed_words=n)
            if by_k[k].strip() and not tx.strip():
                rec["filtered_all"] = True
            removed += n
            # токен из одних знаков («—») — не слово: в голосовании он не совпал бы
            # ни с одним словом GigaAM (те знаков отдельно не дают, words.result_words)
            wh_words += [t for t in tx.split() if W.norm_word(t)]
            if capture is not None:
                capture.append({"chunk_index": k, "audio_span": [c["a"], c["b"]], "own_span": list(c["own"]),
                                "heard": True, "state": rec.get("state"), "raw_text": by_k[k], "text": tx,
                                "removed_words": n})
        else:            # основная и вторая здесь согласны: третья голосования не изменила бы
            s0, e0 = c["own"]
            wh_words += [w["w"] for w in words if s0 <= w["s"] < e0]
            if capture is not None:     # подставлено из основной модели — не голос третьей
                capture.append({"chunk_index": k, "audio_span": [c["a"], c["b"]], "own_span": list(c["own"]),
                                "heard": False, "state": None, "raw_text": None, "text": None,
                                "removed_words": None})
    return wh_words, removed


WHISPER_ONLY_ENGINES = ("whisper-ct2", "whisper-ov")     # движки, умеющие только Whisper
# Третьи модели, у которых при сохранении гипотез (save_hypotheses) слова со временами берутся тем же
# вызовом (backends.texts_timed): только те, у кого текст совпал с transcribe_texts побайтно при сравнении
# на настоящей модели. У остальных — прежний путь, в гипотезах word_times: not_measured.
# Сравнение 24.11.2026 (синтетическая речь 86 с, 6 кусков, onnx-cpu int8, модели из кэша): тексты и
# диагностика кусков совпали у всех трёх моделей. Другая точность или движок — прежний путь.
TIMED_THIRD_VERIFIED = frozenset({"multilingual_large_ctc", "multilingual_ctc", "v3_e2e_ctc"})


def whisper_pass(plan, words, cfg, st, quality, make_asr, prog, timing, capture=None):
    """Третий проход (любая модель, только текст; по умолчанию Whisper) -> слова-строки или None,
    если не удался ни один движок. Имя функции и ключ «whisper» в данных — прежние.

    Движки по очереди: faster-whisper (если выбран тестом скорости, asr_third_backend=whisper-ct2),
    видеокарта Arc (OpenVINO, fp16), процессор (int8). Ошибочные фрагменты повторяются на onnx-cpu:
    для Arc (GPU) это повтор на CPU, для faster-whisper (whisper-ct2) — возврат к прежнему движку.
    Для самого onnx-cpu повтор тем же движком бессмыслен. Пустой текст ошибкой не считается.
    Если третья модель — не Whisper (например multilingual_large_ctc), движки faster-whisper и OpenVINO
    Whisper пропускаются (они умеют только Whisper), считает общий onnx-cpu. faster-whisper грузит только
    large-v3-turbo: для другого Whisper он тоже пропускается (transcript.asr_model_caps, ключ ct2),
    а OpenVINO Whisper и onnx-cpu грузят выбранную модель по имени.

    capture — словарь для сохранения гипотез (hypstore) или None (прежнее поведение): в него пишутся
    куски третьей модели (слушала / подставлено из основной, сырой текст и после чистки) и, для моделей
    из TIMED_THIRD_VERIFIED, слова со временами того же вызова."""
    from backends import normalize_device
    third, pieces = plan["model"], plan["pieces"]
    dev = normalize_device(cfg.get("asr_third_device", "CPU"))
    if dev not in ("GPU", "CPU"):
        print(f"[words] {_third_label(third)}: устройство {dev} не поддерживается, считаю на CPU",
              file=sys.stderr, flush=True)
        dev = "CPU"
    tries = ([("whisper-ct2", "CPU", "int8")] if cfg.get("asr_third_backend", "") == "whisper-ct2" else []) \
        + ([("whisper-ov", dev, cfg.get("asr_third_quant", "fp16"))] if dev == "GPU" else []) \
        + [("onnx-cpu", "CPU", cfg.get("asr_third_quant_cpu", "int8"))]
    caps = transcript.asr_model_caps(third)
    if not caps["whisper"]:
        # whisper-ct2 и whisper-ov грузят только Whisper и имя модели игнорируют: для другой модели
        # (например multilingual_large_ctc) они молча подменили бы её Whisper
        skipped = [t[0] for t in tries if t[0] in WHISPER_ONLY_ENGINES]
        tries = [t for t in tries if t[0] not in WHISPER_ONLY_ENGINES]
        if skipped:
            print(f"[words] третья модель {third}: " + (
                f"движок {skipped[0]} пропущен — он умеет только Whisper" if len(skipped) == 1 else
                f"движки {', '.join(skipped)} пропущены — они умеют только Whisper")
                + "; считаю общим движком onnx-cpu", file=sys.stderr, flush=True)
    elif not caps["ct2"] and tries[0][0] == "whisper-ct2":
        # faster-whisper здесь грузит только large-v3-turbo: другой Whisper он подменил бы
        # (решение владельца 22.11.2026 — движок пропускается). OpenVINO Whisper (whisper-ov)
        # грузит модель по имени, поэтому для любого Whisper остаётся в очереди
        tries = tries[1:]
        print(f"[words] третья модель {third}: движок whisper-ct2 пропущен — он умеет только "
              "Whisper large-v3-turbo; считаю движком с выбранной моделью", file=sys.stderr, flush=True)
    outer_attempts = []
    for backend, device, quant in tries:
        try:
            t3 = time.time()
            label = _third_label(third, backend)
            st.set(stage=f"третья модель: загрузка {label} ({device})", progress=68)
            asr3, errs3 = _load_whisper(backend, device, quant, third, cfg, make_asr)
            t_load = time.time() - t3
            compile_sec, name3 = getattr(asr3, "compile_sec", None), getattr(asr3, "name", None)
            whisper_diag = []
            _register_pass(quality, "whisper", third, asr3.name, whisper_diag)
            timed = None
            if capture is not None and third in TIMED_THIRD_VERIFIED and backend == "onnx-cpu" and quant == "int8" \
                    and not caps["whisper"] and hasattr(getattr(asr3, "model", None), "with_timestamps"):
                from backends import texts_timed
                texts, timed = texts_timed(asr3, pieces, progress=prog(68, 74, f"третья модель ({label}, {device})"),
                                           diagnostics=whisper_diag)
            else:
                texts = asr3.transcribe_texts(pieces, progress=prog(68, 74, f"третья модель ({label}, {device})"),
                                              diagnostics=whisper_diag)
            del asr3
            for rec in whisper_diag:
                rec["attempts"][:0] = [dict(attempt) for attempt in outer_attempts]
                for attempt in rec["attempts"]:
                    attempt.setdefault("backend", backend + ":" + device)
            failed_indices = [i for i, rec in enumerate(whisper_diag) if rec["state"] == "error"]
            if failed_indices and backend != "onnx-cpu":
                _retry_whisper_on_cpu(failed_indices, texts, whisper_diag, pieces, third, cfg, st, make_asr, prog)
            chunk_capture = [] if capture is not None else None
            wh_words, removed = _whisper_words(plan, texts, whisper_diag, words, chunk_capture)
            if capture is not None:
                capture.update(chunks=chunk_capture, engine={"backend": backend, "device": device, "quant": quant,
                                                             "name": name3, "fallbacks": errs3 or None})
                if timed is not None:
                    tw = []
                    for k, r in zip(plan["selected"], timed):
                        if r is None:
                            continue
                        c = plan["chunks"][k]
                        tw += [w for w in W.result_words(r, c["a"]) if c["own"][0] <= w["s"] < c["own"][1]]
                    # повтор на CPU заменяет текст ошибочных кусков без времён: тогда времена не полные
                    capture["timed_words"] = tw if not failed_indices else None
            timing.update(asr3_mode=plan["mode"], asr3_pieces=f"{len(plan['selected'])}/{len(plan['chunks'])}",
                          asr3_removed_words=removed or None)
            timing.update(asr3_sec=round(time.time() - t3), asr3_load_sec=round(t_load, 1),
                          asr3_model=third, asr3_device=f"{backend}:{device}:{quant}",
                          asr3_fallbacks=errs3 or None, asr3_compile_sec=compile_sec, asr3_name=name3)
            return wh_words
        except Exception as ex:
            outer_attempts.append({"result": "error", "backend": backend + ":" + device,
                                   "error_type": type(ex).__name__})
            quality.setdefault("asr3_errors", []).append(f"{backend}:{device}: {type(ex).__name__}: {ex}"[:300])
            print(f"[words] {_third_label(third, backend)} ({backend}:{device}) не выполнен: {ex}",
                  file=sys.stderr, flush=True)
    _register_pass(quality, "whisper", third, None, _failed_records(plan["chunks"], outer_attempts, plan["selected"]))
    return None


REF_MIN_SEC = 3.0   # неопознанный кластер с центром и не меньше стольких секунд речи — опорный голос (Т-1-Н1)


def _voice_references(audio, turns, emb, label_name, voice_ctx, allowed):
    """Опорные голоса -> (refs, found): профили опознанных участников; участники из словаря с профилем,
    но без кластера (pyannote мог слить ребёнка с другим); неопознанные кластеры от REF_MIN_SEC с речи — по
    центру (центр есть, только если у кластера есть реплики от 1,5 с: voices.centroids). Кластеры без опоры
    (короче или без центра) называет _without_reference: их слова остаются под меткой разметки."""
    import voices
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
        if lbl not in label_name and size.get(lbl, 0) >= REF_MIN_SEC:
            refs[lbl] = c
            found.add(lbl)
    return {k: v for k, v in refs.items() if v is not None}, found


def _without_reference(turns, label_name, refs):
    """Метки разметки, у которых нет опорного голоса: не опознаны (нет в label_name) и не стали опорой."""
    return {t["label"] for t in turns} - set(label_name) - set(refs)


def _apply_vocab(words, vocab, quality):
    """Словарь группы — по словам; исходное слово сохраняется в raw."""
    if not vocab:
        return
    import vocab as vocab_mod
    fixer = vocab_mod.Fixer(vocab)
    for k, w in enumerate(words):
        new, fx = fixer.apply(w["w"], prev=words[k - 1]["w"] if k else None,
                              nxt=words[k + 1]["w"] if k + 1 < len(words) else None)
        if fx:
            w["raw"], w["w"] = w["w"], new
            quality["vocab_fixes"].append({"t": w["s"], "fixes": fx})


def _segments_from_replies(replies, refs, label_name, label_of):
    """Реплики -> сегменты расшифровки. Голосам без метки (из профиля) даются метки «Спикер N»."""
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
    return segments


def _check_gender(segments, genders_by_name, refs, label_of, quality):
    """Род глагола против пола: голос близок к участнику нужного пола — переносим, иначе метка."""
    import gender
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


def _asr_problem_items(evidence, segments, note):
    """Места для прослушивания, где результат прохода неполон или неизвестен.
    evidence — [(начало, конец, проход, причина, проходы со словами в этом месте)]."""
    items = []
    for start, end, name, reason, sources in evidence:
        has_words = any(start <= w.get("s", -1) < end for seg in segments for w in seg.get("words") or [])
        items.append({"id": "", "t": start, "t1": start, "t2": end, "gap": [start, end],
                      "speaker": None, "text": "", "reasons": ["asr"] if has_words else ["asr", "missed"],
                      "asr_problem": {"model": name, "state": reason, "words_from": sources},
                      "note": note})
    return items


def _review_places(words, segments, turns, full_ann, cfg, label_name, evidence, quality):
    """Метки сомнения и места для разбора -> (места, порог сомнения голоса, наложения голосов)."""
    overlaps = W.overlap_regions(full_ann)
    thr = W.doubt_flags(words, overlaps)
    for seg in segments:
        for wd in seg["words"]:
            src_w = wd.pop("_flags_src")
            if src_w.get("flags"):
                wd["flags"] = src_w["flags"]
            if not wd.get("flags"):
                wd.pop("sims", None)
    missed_enabled = bool(cfg.get("review_missed", True))
    missed, missed_diagnostics = W.missed_speech_diagnostics(
        words, turns, enabled=missed_enabled)
    items = W.review_items(segments, missed=missed, name_of=label_name)
    link_missed_diagnostics(items, missed_diagnostics)
    for item in items:
        if item.get("alt_whisper"):
            start, end = item.get("t1", item["t"]), item.get("t2", item["t"])
            item["gap"] = [start, max(end + 0.04, start + 0.04)]
            item["asr_problem"] = {"model": "whisper", "state": "disagreement", "words_from": ["primary", "second"]}
    third = transcript.model_label(cfg.get("asr_third_model", "onnx-community/whisper-large-v3-turbo"))
    items += _asr_problem_items([(s, e, "whisper", reason, sources) for s, e, reason, sources in evidence["whisper"]],
                                segments, f"Результат {third or 'третьей модели'} неполон или неизвестен; "
                                          "прослушайте фрагмент.")
    items += _asr_problem_items(evidence["other"], segments,
                                "Результат прохода неполон или неизвестен; прослушайте фрагмент.")
    items.sort(key=lambda item: item["t"])
    for index, item in enumerate(items, 1):
        item["id"] = f"r{index}"
    quality["missed_speech"] = [{"t1": s, "t2": e} for s, e, _ in missed]
    quality["missed_speech_diagnostics"] = missed_diagnostics
    for seg in segments:
        for wd in seg["words"]:
            wd.pop("sims", None)
    return items, thr, overlaps


def _interaction_layer(audio, raw, full_ann, overlaps, cfg, quality, timing):
    """Слой признаков разговора для «горячих мест» (hotspots.py): только числа по времени,
    без текста, имён и отпечатков голоса. Ошибка слоя обработку не останавливает."""
    t_layer = time.time()
    if cfg.get("interaction", True):
        try:
            import hotspots
            layer_turns, turns_source = hotspots.cluster_turns(raw, full_ann)
            quality["interaction"] = hotspots.build_layer(audio, SR, layer_turns, overlaps,
                                                          turns_source=turns_source)
        except Exception as ex:
            quality.pop("interaction", None)
            quality["interaction_error"] = f"{type(ex).__name__}: {ex}"[:300]
            print(f"[words] слой признаков разговора не рассчитан: {ex}", file=sys.stderr, flush=True)
    else:
        quality["interaction_skipped"] = "disabled"
    timing["interaction_sec"] = round(time.time() - t_layer, 1)


def _emotions(audio, segments, cfg, a, quality, timing):
    """Эмоции — по репликам от 2 с (если модель доступна)."""
    t_emo = time.time()
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
            err = f"{type(ex).__name__}: {ex}"[:300]
            quality["emotions_error"] = (err + download_hint.certificate_hint(ex))[:600]
        timing["emotions_sec"] = round(time.time() - t_emo, 1)


class _HypCapture:
    """Сбор гипотез для сохранения (hypstore, настройка save_hypotheses). Результат расшифровки не меняет:
    добавляется только строка гипотез записи в базе и поля hypotheses_saved/hypotheses_error в статусе задачи."""

    def __init__(self, audio, cfg, a, src=None):
        import runinfo
        self.cfg, self.a, self.audio = cfg, a, audio
        self.forbid = tuple(n for n in {Path(str(src)).name if src else "",
                                        Path(str(getattr(a, "audio", "") or "")).name} if n)
        self.third, self.trace, self.before = {}, {}, None
        self.run_id = None      # имя файла заранее: слой «всегда» пишет его в запись до сохранения гипотез
        self.conditions_start = _safe(lambda: runinfo.conditions(paths.JOBS, cfg.get("max_heavy_jobs", 2), 1,
                                                                 load_interval=1.0))
        self.conditions_end = None

    def before_vote(self, words, alt, wh_words, plan, quality):
        import copy
        import runinfo
        self.before = {"primary": copy.deepcopy(words), "second": copy.deepcopy(alt),
                       "whisper": list(wh_words) if wh_words is not None else None,
                       "plan": {"model": plan.get("model"), "mode": plan.get("mode"),
                                "max_repeat": plan.get("max_repeat"),
                                "selected": len(plan.get("selected") or []), "chunks": len(plan.get("chunks") or [])},
                       "diagnostics": copy.deepcopy(quality.get("asr_diagnostics") or {})}
        self.conditions_end = _safe(lambda: runinfo.conditions(paths.JOBS, self.cfg.get("max_heavy_jobs", 2), 1,
                                                               load_interval=0))

    def save(self, quality, timing, vocab, recording_id=None):
        """Запись гипотез в базу после сохранения результата задачи. Сбой записи задачу не роняет.
        -> поля для статуса задачи: hypotheses_saved и при сбое hypotheses_error (вид ошибки без текста и пути)."""
        try:
            saved = _save_hypotheses(self, quality, timing, vocab, recording_id)
            print(f"[words] гипотезы моделей сохранены в базу (прогон {saved['run_id']}, "
                  f"{saved['size_bytes'] // 1024} КБ)", file=sys.stderr, flush=True)
            return {"hypotheses_saved": True}
        except Exception as ex:
            print(f"[words] гипотезы моделей не сохранены: {type(ex).__name__}", file=sys.stderr, flush=True)
            return {"hypotheses_saved": False, "hypotheses_error": type(ex).__name__}


def _finish_hypotheses(hyp, quality, timing, vocab, saved):
    """Гипотезы — только у задачи, результат которой уже сохранён в базу (main, после _save_worker_result):
    им нужен номер записи. Отменённая или упавшая задача гипотез не оставляет.
    -> поля для статуса задачи ({} — настройка выключена)."""
    if hyp is None:
        return {}
    return hyp.save(quality, timing, vocab, (saved or {}).get("recording_id"))


ALWAYS_LAYER = "always-1"
MODEL_ARTIFACTS_FILE = "model_artifacts.json"   # кэш хэшей весов слоя «всегда» в <папка данных>\bench


def pcm_sha256(audio, block=1 << 20):
    """SHA-256 звука по правилу паспорта теста скорости (моно 16 кГц s16le, hypstore.pcm_bytes), по блокам:
    без второй полной копии звука в памяти. Тот же результат, что sha256(hypstore.pcm_bytes(audio))."""
    import hashlib
    import hypstore
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    h = hashlib.sha256()
    for i in range(0, len(arr), block):
        h.update(hypstore.pcm_bytes(arr[i:i + block]))
    return h.hexdigest(), len(arr)


def _always_layer(audio, cfg, vocab, quality, hyp):
    """Слой «всегда» (план лаборатории, раздел 0.1): в quality.asr_run каждой новой записи, у всех, без
    настройки — хэш звука, ссылка на гипотезы прогона, обезличенный стенд, снимок настроек распознавания,
    хэши файлов весов и ключевые пакеты. Только числа, хэши и имена моделей, без текстов и путей.
    Сбой любого поля пишется как not_measured и расшифровку не роняет."""
    import hypotheses as H
    import hypstore
    import runinfo
    run = quality.setdefault("asr_run", {})
    run["layer"] = ALWAYS_LAYER
    sound = _safe(lambda: pcm_sha256(audio))
    if isinstance(sound, tuple):
        digest, samples = sound
        run["sound"] = {"pcm_sha256": digest, "pcm_format": f"mono {SR} Hz s16le",
                        "duration_sec": round(samples / SR, 2)}
    else:
        run["sound"] = sound
    if hyp is None:
        run["hypotheses"] = {"state": "off", "reason": "save_hypotheses выключена"}
    else:
        # прогон, гипотезы которого будут записаны в базу после сохранения результата (строка таблицы hypotheses
        # с этим run_id, Г-1); запись могла не удаться — строку ищут по run_id и проверяют её наличие (сбой виден
        # в статусе задачи, hypotheses_error)
        hyp.run_id = runinfo.new_run_id()
        run["hypotheses"] = {"state": "expected",
                             "sound_id": sound[0][:12] if isinstance(sound, tuple)
                             else H.not_measured("хэш звука не посчитан"),
                             "run_id": hyp.run_id}
    run["testbed_id"] = _safe(lambda: runinfo.testbed_id(paths.BENCH))
    run["config"] = _safe(lambda: hypstore.config_snapshot(cfg, vocab))
    full = _safe(runinfo.packages)
    if isinstance(full, dict) and "key" in full:
        run["packages_all"] = {"count": full.get("count"), "sha256": full.get("sha256"),
                               "sources": full.get("sources")}
    else:
        run["packages_all"] = full
    cache_path = paths.BENCH / MODEL_ARTIFACTS_FILE
    arts = hypstore._read_json(cache_path, {})
    before = json.dumps(arts, sort_keys=True)
    for entry in (run.get("models") or {}).values():
        if isinstance(entry, dict):
            model_id, backend = entry.get("id"), entry.get("backend")
            entry["artifact"] = _safe(lambda: hypstore.model_artifact(model_id, paths.CACHE, arts, backend)) \
                if model_id else H.not_available("модель не названа")
    if json.dumps(arts, sort_keys=True) != before:
        try:
            hypstore._write_json_atomic(cache_path, arts)
        except OSError:
            pass
    return run


def _always_layer_guarded(audio, cfg, vocab, quality, hyp):
    """Слой «всегда» в main: сбой слоя целиком расшифровку не роняет, в записи остаётся вид ошибки."""
    try:
        return _always_layer(audio, cfg, vocab, quality, hyp)
    except Exception as ex:
        quality.setdefault("asr_run", {})["layer_error"] = type(ex).__name__
        return None


def _safe(fn):
    try:
        return fn()
    except Exception as ex:
        return {"status": "not_measured", "reason": type(ex).__name__}


def _no_none(obj):
    """None внутри блока (и сам блок None) -> not_measured: в файле гипотез неизвестное записано явно."""
    import hypotheses as H
    if obj is None:
        return H.not_measured("не записано")
    if isinstance(obj, dict):
        return {k: _no_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_no_none(v) for v in obj]
    return obj


def _save_hypotheses(cap, quality, timing, vocab, recording_id=None):
    """Запись гипотез hyp-2 в базу (hypstore.write_db) после сохранения результата задачи, частью записи
    recording_id. -> {"run_id", "recording_id", "size_bytes"}."""
    import platform
    import hypotheses as H
    import hypstore
    import runinfo
    b = cap.before
    if b is None:
        raise RuntimeError("гипотезы до голосования не собраны")
    if not recording_id:
        raise ValueError("гипотезы сохраняются только для записи в базе: нет номера записи")
    base = hypstore.root()           # папка кэшей _artifacts.json и _testbed.json
    base.mkdir(parents=True, exist_ok=True)
    pcm = hypstore.pcm_bytes(cap.audio)
    src = Path(cap.a.audio).expanduser() if getattr(cap.a, "audio", "") else None
    ident = runinfo.audio_identity(src if src else Path("-"), pcm, SR)
    ex = ident.pop("excerpt")
    unit_id = ident.pop("unit_id")
    audio_block = {**ident, "pcm_sha256": ex["pcm_sha256"], "pcm_format": ex["pcm_format"],
                   "duration_sec": ex["duration_sec"]}
    run_id = cap.run_id or runinfo.new_run_id()
    diag = b["diagnostics"]
    arts = hypstore._read_json(base / hypstore.ARTIFACTS_FILE, {})

    def artifact(model, engine):
        return {"name": model, "artifact": _safe(lambda: hypstore.model_artifact(model, paths.CACHE, arts, engine))}

    def chunks_of(name):
        return [{k: rec.get(k) for k in ("index", "audio_span", "own_span", "state")}
                for rec in (diag.get(name) or {}).get("chunks", [])]

    def absent(why):
        return {k: H.not_available(why) for k in ("model", "engine", "timing", "words", "word_times")}

    roles = [{"role": "primary", "model": artifact(timing.get("asr_model"), timing.get("asr_backend")),
              "engine": {"name": timing.get("asr_backend"), "fallbacks": quality.get("asr_fallbacks")},
              "timing": {"pass_sec": timing.get("asr_sec"), "chunks": timing.get("asr_chunks")},
              "chunks": chunks_of("primary"), "words": b["primary"], "word_times": H.available()}]
    if b["second"] is not None:
        roles.append({"role": "second", "model": artifact(timing.get("asr2_model"), (diag.get("second") or {}).get("backend")),
                      "engine": {"name": (diag.get("second") or {}).get("backend")},
                      "timing": {"pass_sec": timing.get("asr2_sec")},
                      "chunks": chunks_of("second"), "words": b["second"], "word_times": H.available()})
    else:
        roles.append({"role": "second", "chunks": chunks_of("second"), **absent("вторая модель не считалась")})
    third_model, t3 = b["plan"]["model"], cap.third
    if third_model and b["whisper"] is not None:
        if transcript.asr_model_caps(third_model)["whisper"]:
            wt = H.not_available("Whisper: времена слов меняют декодирование, сохраняется только текст")
            tw = wt
        elif t3.get("timed_words") is not None:
            wt, tw = H.available(), t3["timed_words"]
        elif third_model in TIMED_THIRD_VERIFIED:
            wt = tw = H.not_measured("повтор кусков на CPU: времена не полные")
        else:
            wt = tw = H.not_measured("text_mismatch: совпадение текста со временами для этой модели не проверено")
        roles.append({"role": "third", "model": artifact(third_model, (t3.get("engine") or {}).get("name")),
                      "engine": {**(t3.get("engine") or {}), "mode": b["plan"]["mode"],
                                 "pieces": timing.get("asr3_pieces"), "fallbacks": timing.get("asr3_fallbacks")},
                      "timing": {"pass_sec": timing.get("asr3_sec"), "load_sec": timing.get("asr3_load_sec"),
                                 "compile_sec": timing.get("asr3_compile_sec")},
                      "chunks": t3.get("chunks") or chunks_of("whisper"), "words": tw, "word_times": wt})
    else:
        roles.append({"role": "third", "chunks": chunks_of("whisper"),
                      **absent("третья модель не считалась или не дала результата")})
    for block in roles:      # неизвестное в model/engine/timing — not_measured, а не None (words, chunks — как есть)
        for key in ("model", "engine", "timing"):
            block[key] = _no_none(block.get(key))
    tid, hw = hypstore.testbed(HERE, base)
    meta = {"audio": audio_block,
            "program": {"version": runinfo.program_version(HERE), "code": _safe(lambda: runinfo.code_revision(HERE))},
            "environment": {"os": _safe(runinfo.os_info), "python": {"version": platform.python_version()},
                            "packages": _safe(runinfo.packages), "asr_run": quality.get("asr_run")},
            "hardware": {"testbed_id": tid, "machine": hw},
            "conditions": {"start": cap.conditions_start, "after_passes": cap.conditions_end,
                           "peak_memory_mb": _safe(runinfo.peak_memory_mb)},
            "config": hypstore.config_snapshot(cap.cfg, vocab),
            "ids": {"run_id": run_id, "unit_id": unit_id,
                    "job_id": Path(cap.a.status).stem if getattr(cap.a, "status", "") else None,
                    "recording_id": recording_id,
                    "written_utc": runinfo.utc_now()}}
    combine = {"function": "vote_n", "rule": cap.trace.get("rule") or dict(W.TWO_OF_THREE),
               "partial": quality.get("vote3_partial"), "skipped": quality.get("vote3_skipped"),
               "third_words": b["whisper"], "third_plan": b["plan"]}
    rec = H.build_record(roles, legacy={"combine_rule": "vote3_partial"}, combine=combine,
                         trace=cap.trace or None, meta=meta)
    saved = hypstore.write_db(rec, unit_id, run_id, recording_id, forbid=cap.forbid, job_id=meta["ids"]["job_id"])
    try:
        hypstore._write_json_atomic(base / hypstore.ARTIFACTS_FILE, arts)
    except OSError:
        pass
    return saved


def run_words(audio, src, cfg, st, raw, order, turns, full_ann, emb, label_name, label_of, voice_ctx,
              allowed, genders_by_name, vocab, quality, make_asr, a):
    """Пословная обработка (words.py). -> данные расшифровки (без file/duration/timing).

    Этапы: опорные голоса; проходы моделей по кускам, каждый даёт гипотезу (primary_pass,
    second_pass, whisper_pass; сведения о сбоях — в quality["asr_diagnostics"]); сведение гипотез
    (words.combine_hypotheses); словарь; голос для каждого слова; реплики; род глагола; места
    для разбора; слой признаков разговора; эмоции."""

    timing = {}
    import hypstore
    hyp = _HypCapture(audio, cfg, a, src) if hypstore.enabled(cfg) else None
    refs, found = _voice_references(audio, turns, emb, label_name, voice_ctx, allowed)
    unref = _without_reference(turns, label_name, refs)
    if unref or not refs:
        # голос этих слов назначает программа по разметке pyannote, без сверки отпечатков (Т-1-Н5, Т-1-Н1)
        quality["voice_refs"] = {"refs": len(refs), "without_reference": sorted(unref), "assigned_by": "diarization"}
    mapping = voice_ctx.get("mapping") or {}

    def name_of_label(lbl):   # метка pyannote -> имя (в т.ч. для кластеров, слитых в одного человека)
        return (mapping.get(lbl) or (label_name.get(lbl),))[0] or lbl

    prior = W.LabelAt([(s, e, name_of_label(order[spk])) for s, e, spk in raw])

    # проходы моделей по длинным кускам -> гипотезы
    regions = W.speech_regions(raw)
    chunks = W.make_chunks(audio, regions)
    prog = _progress(st)
    words = primary_pass(audio, chunks, cfg, st, quality, make_asr, prog, timing)
    alt = second_pass(audio, chunks, cfg, quality, make_asr, prog, timing)
    plan = whisper_plan(audio, regions, chunks, cfg, a, words, alt, quality)
    wh_words = whisper_pass(plan, words, cfg, st, quality, make_asr, prog, timing,
                            **({"capture": hyp.third} if hyp else {})) \
        if plan["model"] and alt is not None else None
    if hyp:
        hyp.before_vote(words, alt, wh_words, plan, quality)

    # сведение гипотез -> слова с пометками расхождений, решения голосования, сведения о сбоях
    fused = W.combine_hypotheses({"primary": words, "second": alt, "whisper": wh_words},
                                 quality["asr_diagnostics"], **({"trace": hyp.trace} if hyp else {}))
    quality.update(fused["quality"])

    _apply_vocab(words, vocab, quality)

    # голос для каждого слова
    st.set(stage="голос для каждого слова", progress=75)
    t3 = time.time()
    nwin = W.assign_voices(audio, words, refs, emb, prior=prior, info={"found": found}, unreferenced=unref,
                           win_near_change=float(cfg.get("word_win_near_change", 0) or 0),
                           confusions=voice_ctx.get("confusions"),
                           progress=prog(75, 85, "голос для каждого слова"))
    timing.update(word_voice_sec=round(time.time() - t3, 1), word_windows=nwin)
    segments = _segments_from_replies(W.build_replies(words), refs, label_name, label_of)
    _check_gender(segments, genders_by_name, refs, label_of, quality)
    items, thr, overlaps = _review_places(words, segments, turns, full_ann, cfg, label_name,
                                          fused["evidence"], quality)
    _interaction_layer(audio, raw, full_ann, overlaps, cfg, quality, timing)
    _emotions(audio, segments, cfg, a, quality, timing)

    labels = list(dict.fromkeys(s["label"] for s in segments))
    names = {l: label_name[l] for l in labels if l in label_name}
    quality.update(pipeline="words", voice_doubt_threshold=thr, overlaps_sec=round(sum(e - s for s, e in overlaps), 1),
                   flagged_words=sum(1 for w in words if w.get("flags")), words=len(words))
    data = {"speakers": [label_name.get(l, l) for l in labels], "speaker_labels": labels,
            "quality": quality, "review": items, "review_done": False, "edits": [], "segments": segments,
            "_timing": timing}
    if names:
        data.update(speaker_names=names, names_source="voices", names_confirmed=False)
    if hyp:
        data["_hyp"] = hyp      # служебное, как _timing: main забирает и пишет файл после сохранения результата
    return data


def load_group_profile_for_recognition(path, quality):
    """Использовать диаризацию без профиля, если существующий файл небезопасен для чтения."""
    import voices
    try:
        return voices.load_profile(path), True
    except voices.VoiceProfileError as ex:
        quality["voices_error"] = str(ex)
        quality["voice_auto_learn"] = {"enabled": False, "reason": "invalid_profile"}
        return voices.empty_profile(), False


class _ProfileBackupFailed(Exception):
    """Резерв профиля перед автозаписью не сохранён; профиль не записывается."""


def prepare_group_voice_profile(group, src, embedder, names, quality):
    """Подготовить профиль и безопасно выполнить разрешённое автоматическое пополнение."""
    import voices
    ppath = voices.profile_path(group)
    with settings.group_voice_lock(group):
        prof, profile_ok = load_group_profile_for_recognition(ppath, quality)
        selection_report = {}
        quality_report = {}
        learned = []
        if profile_ok:
            try:
                sqlite_mode = settings.recording_backend() == "sqlite"
            except (ValueError, OSError):
                sqlite_mode = False

            def run_learn():
                try:
                    policy = settings.group_voice_settings(group).get("audio_quality_policy", "warn")
                except (ValueError, OSError):
                    policy = "warn"
                return voices.learn(prof, src.parent, embedder, load_audio, names=names or None, group=group,
                                    selection_report=selection_report, quality_policy=policy,
                                    quality_report=quality_report, sqlite_mode=sqlite_mode,
                                    stop_when=lambda: voices.profile_sufficiency(prof, names)["sufficient"])

            def save():
                # при работе с базой (решение владельца 25.11.2026) — резерв перед автозаписью; не вышел — не пишем
                if sqlite_mode:
                    try:
                        voices.backup_profile(ppath)
                    except OSError as ex:
                        raise _ProfileBackupFailed(str(ex)) from ex
                voices.save_profile(ppath, prof)

            try:
                learned, quality["voice_auto_learn"] = voices.auto_learn_group(prof, group, names, run_learn, save)
            except _ProfileBackupFailed as ex:
                # профиль на диске не тронут; распознавание идёт по нему, а не по несохранённому пересчёту
                prof, _ = load_group_profile_for_recognition(ppath, quality)
                learned = []
                quality["voice_auto_learn"] = {"enabled": True, "reason": "backup_failed", "error": str(ex)}
                selection_report.clear()
                selection_report["profile_backup_failed"] = 1
        quality["voice_selection_report"] = selection_report
        quality["voice_selection_explanation"] = voices.explain_selection_report(selection_report)
        quality["voice_audio_quality"] = quality_report
    return ppath, prof, learned


def _open_voice_source(ref):
    import storage
    import voice_sources
    try:
        return voice_sources.open_ref(ref)
    except voice_sources.SourceError as ex:
        raise RuntimeError(str(ex)) from ex
    except (storage.StorageError, OSError, KeyError, ValueError) as ex:
        raise RuntimeError(f"источник для голосового профиля недоступен: {type(ex).__name__}: {ex}") from ex


def _voice_source_token(ref):
    """Жетон источника: SHA-256 файла или ревизия записи в базе (sqlite:<база>:<запись>:<ревизия>:<хэш>)."""
    return _open_voice_source(ref).token


def _check_confirmed_voice_inputs(a, src, group, profile_path, resolution, core):
    source_hash = getattr(a, "voice_source_hash", "")
    profile_hash = getattr(a, "voice_profile_hash", "")
    audio_hash = getattr(a, "voice_audio_hash", "")
    expected_group = getattr(a, "voice_group", "")
    authorized = bool(source_hash and profile_hash and audio_hash and expected_group)
    if not authorized:
        return False
    if (group != expected_group or _voice_source_token(a.learn_only) != source_hash or
            core._voice_hash(profile_path) != profile_hash or core._voice_hash(src) != audio_hash):
        raise RuntimeError("стенограмма, аудио или профиль изменились после подтверждения; нужен новый просмотр")

    expected_path = getattr(a, "voice_vocab_path", None)
    expected_hash = getattr(a, "voice_vocab_hash", None)
    if expected_path is not None:
        expected_path = str(Path(expected_path).resolve()) if expected_path else None
        actual_path = str(Path(resolution["path"]).resolve()) if resolution["path"] else None
        if expected_path != actual_path:
            raise RuntimeError("выбранный словарь изменился после подтверждения; нужен новый просмотр")
    if expected_hash:
        actual_hash = core._voice_hash(resolution["path"]) if resolution["path"] else "absent"
        if actual_hash != expected_hash:
            raise RuntimeError("словарь изменился после подтверждения; нужен новый просмотр")
    return True


def learn_only(a, st, cfg, src):
    """Подтверждаемое обновление только указанной расшифровки."""
    try:
        source = _open_voice_source(a.learn_only)
        transcript_path = source.json_path      # у записи в базе — её выгрузка (поиск словаря рядом)
        transcript_data = source.data
        if source.kind == "sqlite" and not (getattr(a, "voice_source_hash", "") and
                                            getattr(a, "voice_profile_hash", "") and
                                            getattr(a, "voice_audio_hash", "") and getattr(a, "voice_group", "")):
            raise RuntimeError("обновление голосового профиля по записи в базе — только после просмотра и "
                               "подтверждения человеком")
        override = getattr(a, "voice_group_override", "") or ""
        if transcript_data.get("no_profile"):
            if override:
                raise RuntimeError("для этой расшифровки профили групп отключены; выбор группы это не меняет")
            st.set(state="done", stage="готово", progress=100, kind="learn", learned=0,
                   note="расшифровка сделана без профилей групп — группа не может включить обучение")
            return

        import voices
        resolution = resolve_vocab_details(src, a, cfg, transcript_path=transcript_path,
                                           transcript_data=transcript_data,
                                           group_override=override or None)
        binding = voices.group_binding(transcript_data)
        group = override or binding["group"] or resolution["group"]
        if not group:
            raise RuntimeError("Группа исходной расшифровки неизвестна; требуется явный выбор")
        if paths.safe_group(group) != group:
            raise RuntimeError("Недопустимое имя группы")
        profile_path = voices.profile_path(group)
        authorized = bool(getattr(a, "voice_source_hash", "") and
                          getattr(a, "voice_profile_hash", "") and
                          getattr(a, "voice_audio_hash", "") and getattr(a, "voice_group", ""))
        if override and not authorized:
            raise RuntimeError("ручной выбор группы разрешён только по подтверждённому hash-bound preview")
        if override and (getattr(a, "voice_group", "") != group or
                         not getattr(a, "voice_vocab_hash", "") or
                         not getattr(a, "voice_vocab_path", "")):
            raise RuntimeError("снимок ручного выбора группы неполон")

        import core
        with settings.group_voice_lock(group):
            # Повторно обнаруживаем sidecar/reference уже под блокировкой до загрузки модели.
            resolution = resolve_vocab_details(src, a, cfg, transcript_path=transcript_path,
                                               transcript_data=transcript_data,
                                               group_override=override or None)
            binding = voices.group_binding(transcript_data)
            if (override or binding["group"] or resolution["group"]) != group:
                raise RuntimeError("группа изменилась после подтверждения; нужен новый просмотр")
            _check_confirmed_voice_inputs(a, src, group, profile_path, resolution, core)
            if override and resolution["report"].get("selection_reason") != "selected_group_vocab":
                raise RuntimeError("не подтверждён словарь выбранной группы")

            vocab_data = resolution["vocab"]
            allowed = [p["name"] for p in vocab_data.get("participants", []) if p.get("name")]
            st.set(stage="профили: загрузка модели голосов", progress=10)
            from pyannote.audio import Pipeline
            token = os.environ.get("HF_TOKEN", "").strip()
            name = "pyannote/speaker-diarization-community-1"
            try:
                pipe = Pipeline.from_pretrained(name, token=token)
            except TypeError:
                pipe = Pipeline.from_pretrained(name, use_auth_token=token)
            emb = voices.Embedder(pipe)

            prof = voices.load_profile(profile_path)
            st.set(stage="профили: пересчёт по ответам человека", progress=40)
            selection_report = {}
            quality_report = {}
            try:
                quality_policy = settings.group_voice_settings(group).get("audio_quality_policy", "warn")
            except (ValueError, OSError):
                quality_policy = "warn"
            before_provenance = prof.get("provenance_revision", 0)
            learned = voices.learn(prof, transcript_path.parent, emb, load_audio, names=allowed or None,
                                   only=[source], group=group, selection_report=selection_report,
                                   allow_replace=authorized, quality_policy=quality_policy,
                                   quality_report=quality_report)
            provenance_updated = prof.get("provenance_revision", 0) != before_provenance

            # Входы и разрешение словаря могут устареть даже во время вычисления.
            after_resolution = resolve_vocab_details(src, a, cfg, transcript_path=transcript_path,
                                                     transcript_data=transcript_data,
                                                     group_override=override or None)
            _check_confirmed_voice_inputs(a, src, group, profile_path, after_resolution, core)
            after_path = str(after_resolution["path"]) if after_resolution["path"] else None
            before_path = str(resolution["path"]) if resolution["path"] else None
            if after_resolution["group"] != group or after_path != before_path:
                raise RuntimeError("привязка или словарь изменились во время пересчёта; запись отменена")

            backup = None
            if learned or provenance_updated:
                if authorized:
                    backup = core.backup_voice_profile(profile_path)
                voices.save_profile(profile_path, prof)
        last = prof.get("history", [])[-1] if learned else None
        st.set(state="done", stage="готово", progress=100, kind="learn", learned=learned,
               profile=profile_path.name, revision=prof.get("revision"), last=last,
               selection_report=selection_report,
               selection_explanation=voices.explain_selection_report(selection_report), backup=backup,
               provenance_updated=provenance_updated, audio_quality=quality_report,
               vocab_resolution=resolution["report"],
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
    ap.add_argument("--voice-source-hash", default=""); ap.add_argument("--voice-profile-hash", default="")
    ap.add_argument("--voice-audio-hash", default="")
    ap.add_argument("--voice-group", default="")
    ap.add_argument("--voice-group-override", default="")
    ap.add_argument("--voice-vocab-path", default="")
    ap.add_argument("--voice-vocab-hash", default="")
    ap.add_argument("--whisper", default="", choices=["", "all", "doubt", "off"],
                    help="третья модель голосования (asr_third_model, по умолчанию Whisper): all — вся запись, "
                         "doubt — только куски, где основная и вторая модели расходятся, off — без неё")
    ap.add_argument("--whisper-max-repeat", type=int, default=-1,
                    help="только для третьей модели Whisper: сколько раз подряд она может повторить одну фразу "
                         "(зацикливание); 0 — без ограничения")
    a = ap.parse_args()
    import recordings
    if not recordings.SQLITE_RUNTIME_READY:
        raise RuntimeError("SQLite backend unavailable until all recording writers are routed")

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
        # новая запись — только в базу; документы — в подпапке рядом с аудио
        layout = recordings.document_layout(src, tag)
        out_dir = layout["root"]
        out_dir.mkdir(exist_ok=True)
        store = recordings.open_store()
        try:
            root_id, document_paths = recordings.prepare_new_layout(store, out_dir, layout["paths"])
            context = st.data.get("recording_context")
            if context is None and not a.status:
                # ручной запуск: автора берём из базы, как core.start_transcription в момент запуска
                context = recordings.operation_context(store, "model")
        finally:
            store.close()
        json_path = out_dir / document_paths["json"]
        md_path = out_dir / document_paths["md"]
        if not isinstance(context, dict) or context.get("actor_kind") != "model":
            raise RuntimeError("SQLite job is missing its captured initiator context")
        sqlite_target = {"root_id": root_id, "paths": document_paths, "context": context}

        resolution = resolve_vocab_details(src, a, cfg)
        vocab_path, vocab, vgroup = resolution["path"], resolution["vocab"], resolution["group"]
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
        quality = {"vocab": str(vocab_path) if vocab_path else None,
                   "vocab_resolution": resolution["report"], "speaker_fixes": [], "vocab_fixes": [],
                   "review": []}
        label_name, label_of, refs, emb, t_voice = {}, {}, {}, None, 0.0
        voice_ctx = {}
        pipeline = cfg.get("pipeline", "words")
        if pipeline == "words":
            # пословной обработке отпечатки нужны всегда (опорные голоса, голос каждого слова): сбой создания
            # модели — ошибка задачи, а не «голоса отключены»; создаём один раз, до блока голосов
            import voices
            emb = voices.Embedder(pipe)
        if not a.no_voices and cfg.get("voices", True):
            tv = time.time()
            # Блок голосов — всё или ничего: результат собирается во временных переменных (рабочие копии реплик,
            # свои словари меток) и применяется одним шагом в конце, где только присваивания. Сбой на любом шаге
            # оставляет состояние «без голосов»: метки pyannote, имён нет, отчёта о сопоставлении нет, и пишет
            # quality["voices_error"]. На диске профиль и настройки группы меняет prepare_group_voice_profile
            # (автопополнение из прежних записей, под блокировкой группы, файл заменяется целиком): это не зависит
            # от шагов ниже, поэтому его результат (learned_from) записывается сразу, а не вместе с отчётом.
            try:
                import voices
                st.set(stage="голоса: профили участников", progress=30)
                if emb is None:        # прежний режим segments: без отпечатков он идёт дальше без голосов
                    emb = voices.Embedder(pipe)
                if no_profile:
                    group, ppath, learned = None, None, 0
                    prof = voices.empty_profile()     # пустой, не сохраняется
                    quality["voices_off"] = ("общий словарь без поля group: профили голосов групп не применяются "
                                             "и не пополняются (имена — rename_speakers)")
                else:
                    group = vgroup or "default"
                    ppath, prof, learned = prepare_group_voice_profile(group, src, emb, all_names, quality)
                quality["learned_from"] = learned     # уже записанное на диск: фиксируем, даже если ниже будет сбой
                st.set(stage="голоса: проверка реплик", progress=33)
                work = [dict(t) for t in turns]       # рабочие копии: turns меняются только при применении
                embs = emb.spans(audio, [(t["s"], t["e"]) for t in work])
                if cfg.get("voices_debug"):   # отпечатки реплик для подбора порогов (биометрия: только локально)
                    dbg = paths.JOBS / f"{stem}.voices_debug.json"
                    dbg.write_text(json.dumps({"turns": [{**t, "vec": None if v is None else [round(float(x), 4) for x in v]}
                                                         for t, v in zip(work, embs)]}, ensure_ascii=False), encoding="utf-8")
                cents = voices.centroids(work, embs)
                pvecs = voices.voice_vectors(prof, exclude_audio=src)
                mapping = voices.match(cents, pvecs, allowed or None)
                new_refs, new_label_of = voices.references(cents, work, mapping)
                # кластеры одного человека сводим к одной метке
                for t in work:
                    nm = mapping.get(t["label"], (None,))[0]
                    if nm and new_label_of.get(nm) != t["label"]:
                        t["label"] = new_label_of[nm]
                new_label_name = {lbl: nm for nm, lbl in new_label_of.items()}
                new_ctx = dict(pvecs=pvecs, cents=cents, mapping=mapping,
                               confusions=voices.confusions(prof, exclude_audio=src))
                report = {}
                # голоса записи почти не похожи на группу — возможно, это другая группа с тем же словарём
                dur = {}
                for tt in work:
                    dur[tt["label"]] = dur.get(tt["label"], 0) + tt["e"] - tt["s"]
                tot = sum(dur.values()) or 1.0
                share = sum(d for l, d in dur.items() if l in mapping) / tot
                if len(pvecs) >= 2 and share < 0.5:
                    report["group_doubt"] = {
                        "group": group, "matched_share": round(share, 2),
                        "note": f"голоса записи почти не совпадают с профилями группы «{group}» (узнано "
                                f"{share:.0%} речи) — возможно, другая группа: нужен её словарь с другим group"}
                report.update(voice_profile=ppath.name if ppath else None, profile_names=sorted(pvecs),
                             match={lbl: {"name": nm, "sim": sim} for lbl, (nm, sim) in mapping.items()},
                             unmatched=[l for l in cents if l not in mapping],
                             cluster_sims={l: {n: round(float(cents[l] @ v), 3) for n, v in pvecs.items()}
                                           for l in cents})
                new_fixes = []
                if pipeline != "words":
                    work = voices.scan_split(audio, work, new_refs, new_label_name, new_label_of, emb, new_fixes)
                    embs = emb.spans(audio, [(t["s"], t["e"]) for t in work])
                    debug = []
                    used = {t["label"] for t in work}

                    def new_label(name):
                        k = 1
                        while f"Спикер {k}" in used:
                            k += 1
                        lbl = f"Спикер {k}"
                        used.add(lbl)
                        new_label_of[name], new_label_name[lbl] = lbl, name
                        new_refs.setdefault(name, pvecs[name])
                        return lbl

                    pv = {n: v for n, v in pvecs.items() if not allowed or n in allowed}
                    work = voices.reassign(work, embs, pv, new_label_name, new_label_of, new_fixes, debug,
                                           new_label=new_label)
                    report["debug_sims"] = debug
                # применение одним шагом: только присваивания и обновление словарей, сбоя здесь быть не может
                turns, label_name, label_of, refs, voice_ctx = work, new_label_name, new_label_of, new_refs, new_ctx
                quality["speaker_fixes"].extend(new_fixes)
                quality.update(report)
            except Exception as ex:
                quality["voices_error"] = f"{type(ex).__name__}: {ex}"[:300]
                print(f"[voices] проверка голосов отключена: {ex}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            t_voice = time.time() - tv
        turns = [t for t in turns if t["e"] - t["s"] >= MIN_SEG]

        if pipeline == "words":
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
            hyp = data.pop("_hyp", None)
            _always_layer_guarded(audio, cfg, vocab, quality, hyp)
            saved = _save_worker_result(data, src, st, sqlite_target, json_path, md_path)
            hyp_state = _finish_hypotheses(hyp, quality, timing, vocab, saved)
            model = _execute_workflow_model(st)
            st.set(state="done", stage="готово", progress=100, markdown=str(md_path), json=str(json_path),
                   **saved, **model, **hyp_state,
                   timing=timing, quality={"pipeline": "words", "vocab": quality["vocab"],
                                           "names": data.get("speaker_names") or None,
                                            "review_items": len(data.get("review", [])),
                                            "asr_incomplete_chunks": quality.get("asr_incomplete_chunks"),
                                            "vote3_skipped": quality.get("vote3_skipped"),
                                            "vote3_partial": quality.get("vote3_partial"),
                                           "emotions_error": quality.get("emotions_error"),
                                           "interaction_error": quality.get("interaction_error"),
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
                quality["emotions_error"] = (
                    f"{type(ex).__name__}: {ex}"[:300]
                    + download_hint.certificate_hint(ex)
                )[:600]
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
        saved = _save_worker_result(data, src, st, sqlite_target, json_path, md_path)
        model = _execute_workflow_model(st)

        st.set(state="done", stage="готово", progress=100, markdown=str(md_path),
               json=str(json_path), **saved, **model, timing=timing,
               quality={"vocab": quality["vocab"], "names": label_name or None,
                        "speaker_fixes": len(quality["speaker_fixes"]),
                        "vocab_fixes": sum(len(v["fixes"]) for v in quality["vocab_fixes"]),
                        "review": len(quality["review"]),
                        "voices_error": quality.get("voices_error")})
    except Exception as ex:
        st.set(state="error", stage="ошибка", error=f"{type(ex).__name__}: {ex}",
               trace=traceback.format_exc()[-3000:])
        sys.exit(1)


def _save_worker_result(data, src, status, sqlite_target, json_path, md_path):
    import runinfo
    # сведения о самом файле — как у записей после переноса (tools/import_legacy.py): наличие, хэш файла,
    # контейнер и кодек; по хэшу файла запись связывается с лабораторией и сохранёнными гипотезами
    data.update(runinfo.audio_facts(src))
    if not sqlite_target:
        raise RuntimeError("Новая запись сохраняется только в базу: нет раскладки записи базы")
    import recordings
    store = recordings.open_store()
    try:
        workflow_request_id = getattr(status, "data", {}).get("workflow_request_id")
        if workflow_request_id:
            request_uuid = uuid.UUID(workflow_request_id)
            recording_id = str(uuid.uuid5(request_uuid, "worker-recording"))
            operation_id = str(uuid.uuid5(request_uuid, "worker-result"))
        else:
            recording_id, operation_id = str(uuid.uuid4()), str(uuid.uuid4())
        context = sqlite_target["context"]
        with store._transaction():
            if workflow_request_id:
                import workflow
                workflow.check_worker_result_allowed(store, workflow_request_id, context, src)
            result = recordings.commit_worker_result(
                store, data, src, sqlite_target["root_id"], context,
                recording_id=recording_id, operation_id=operation_id,
                generated_at=datetime.now(timezone.utc).isoformat(),
                document_paths=sqlite_target["paths"])
            if workflow_request_id:
                workflow.link_worker_result(store, workflow_request_id, recording_id)
        status.set(recording_id=recording_id, source_revision=result["source_revision"],
                   review_revision=result["review_revision"], document_revision=result["document_revision"],
                   document_sync=result["document_sync"])
        import documents
        completed = recordings.materialize_recording(store, recording_id)
        documents.require_written(store, recording_id, result["document_revision"], completed)
        return {"recording_id": recording_id, "source_revision": result["source_revision"],
                "review_revision": result["review_revision"], "document_revision": result["document_revision"],
                "document_sync": "synced"}
    finally:
        store.close()


def _execute_workflow_model(status):
    """После commit ASR выполнить заказанный конспект; ошибки не скрывают готовую стенограмму."""
    request_id = getattr(status, "data", {}).get("workflow_request_id")
    if not request_id:
        return {}
    try:
        import recordings
        import workflow
        store = recordings.open_store()
        try:
            request = workflow.get(store, request_id)
            if not request["policy"]["require_conspect"]:
                return {}
            if not request["policy"]["allow_cloud_model"]:
                return {"model_state": "waiting_model", "model_error": "облачный исполнитель запрещён политикой"}
            result = workflow.execute_model(store, request_id)
            return {"model_state": result["state"], "model": result["model"]}
        finally:
            store.close()
    except Exception as ex:
        return {"model_state": "failed", "model_error": f"{type(ex).__name__}: {ex}"[:1000]}


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
