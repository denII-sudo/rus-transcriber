"""Прогон опыта: для каждой единицы (звука) и конфигурации вызывается рабочий конвейер.

Устройство (подробно — lab/README.md):
  * разделение говорящих (worker.diarize) и подготовка голосов выполняются один раз на звук и
    общие для всех его конфигураций; затем для каждой конфигурации вызывается worker.run_words —
    тот же код, что в обычной расшифровке, со своими настройками моделей;
  * гипотезы проходов перехватываются в лаборатории (words.transcribe, words.clean_whisper,
    words.vote3, words.mark_alt) — конвейер для этого не меняется;
  * результат пары пишется в отдельную базу результатов песочницы results/lab.sqlite3;
    уже посчитанные пары пропускаются (возобновление).

Модуль не импортирует программу при загрузке: setup_environment задаёт папку данных
песочницы, кэш моделей установленной программы и работу без сети, и только после этого
PipelineEngine импортирует worker и остальные модули.
"""
import copy
import hashlib
import json
import os
import secrets
import sqlite3
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import lab_sandbox
import lab_spec
import lab_summary

RESULTS_SCHEMA = "lab-results-1"
HYP_SCHEMA = "hyp-2"       # формат гипотез программы (hypotheses.SCHEMA); lab-hyp-1 читается как прежде
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1"}
# Грубые оценки секунд счёта на час звука, если нет ни истории, ни посчитанных пар.
DEFAULT_ESTIMATES = {"diarization": 400, "voices": 60, "word_voice": 150, "primary": 150,
                     "second": 120, "third": 1800}


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── окружение ────────────────────────────────────────────────────────────────────────────────────

def setup_environment(sandbox, program_dir):
    """Окружение процесса до импорта программы. -> словарь прежних значений (для restore_environment)."""
    sandbox, program_dir = Path(sandbox), Path(program_dir)
    cache = program_dir / "cache"
    values = {"RUS_TRANSCRIBER_DATA": str(sandbox / "data"),
              "TEMP": str(sandbox / "tmp"), "TMP": str(sandbox / "tmp"),
              "HF_HOME": str(cache / "huggingface"), "TORCH_HOME": str(cache / "torch"),
              **OFFLINE_ENV}
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    (sandbox / "tmp").mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(sandbox / "tmp")
    if str(lab_sandbox.REPO_DIR) not in sys.path:
        sys.path.insert(0, str(lab_sandbox.REPO_DIR))
    return old


def restore_environment(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    tempfile.tempdir = None


# ── база результатов ─────────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, started_utc TEXT NOT NULL, finished_utc TEXT, state TEXT NOT NULL,
    experiment_name TEXT, experiment_sha256 TEXT, experiment_json TEXT,
    passport_json TEXT, args_json TEXT);
CREATE TABLE IF NOT EXISTS configs (
    config_id TEXT NOT NULL, config_sha256 TEXT NOT NULL, label TEXT, roles_json TEXT NOT NULL,
    settings_json TEXT NOT NULL, effective_json TEXT NOT NULL, first_run_id TEXT,
    PRIMARY KEY (config_id, config_sha256));
CREATE TABLE IF NOT EXISTS unit_runs (
    run_id TEXT NOT NULL, unit_id TEXT NOT NULL, started_utc TEXT NOT NULL, finished_utc TEXT,
    state TEXT NOT NULL, error_kind TEXT, error TEXT, shared_json TEXT,
    PRIMARY KEY (run_id, unit_id));
CREATE TABLE IF NOT EXISTS pairs (
    pair_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, unit_id TEXT NOT NULL,
    config_id TEXT NOT NULL, config_sha256 TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('done', 'error')),
    started_utc TEXT NOT NULL, finished_utc TEXT NOT NULL, error_kind TEXT, error TEXT,
    diarization_sha256 TEXT, conditions_json TEXT, metrics_json TEXT,
    result_json TEXT, hypotheses_json TEXT);
CREATE INDEX IF NOT EXISTS pairs_key ON pairs (unit_id, config_id, config_sha256, state);
"""


def open_results(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA_SQL)
    row = db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
    if row is None:
        db.execute("INSERT INTO meta VALUES ('schema', ?)", (RESULTS_SCHEMA,))
        db.commit()
    elif row["value"] != RESULTS_SCHEMA:
        db.close()
        raise RuntimeError(f"база результатов другой версии ({row['value']}); нужна новая песочница")
    return db


def is_done(db, unit_id, config_id, config_sha):
    return db.execute("SELECT 1 FROM pairs WHERE unit_id=? AND config_id=? AND config_sha256=? "
                      "AND state='done' LIMIT 1", (unit_id, config_id, config_sha)).fetchone() is not None


def _dumps(value):
    return json.dumps(value, ensure_ascii=False, default=_json_default, separators=(",", ":"))


def _json_default(value):
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return None            # векторы голосов и звук в базу результатов не пишутся
    except ImportError:
        pass
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(type(value).__name__)


# ── перехват гипотез ─────────────────────────────────────────────────────────────────────────────

def _f(x):
    return round(float(x), 3)


def combine_rule(words_mod):
    """Правило сведения гипотез в этой версии words.py (отметка в hypotheses_json).

    vote3_partial — сбойный кусок второй или третьей модели исключает из голосования только свой
    отрезок (PR №76); whole_record — прежнее правило: любой сбойный кусок отключал vote3 на всей
    записи. В базах, записанных до появления отметки, её нет (lab_analyze различает по записи)."""
    return "vote3_partial" if hasattr(words_mod, "_vote3_outside_spans") else "whole_record"


class Capture:
    """Подмены на время одного вызова run_words: записывают гипотезы, конвейер не меняют.

    words.transcribe — слова основного и второго прохода (до голосования); words.clean_whisper —
    текст третьей модели по кускам до и после чистки; words.vote3 / words.mark_alt — какое
    сведение выполнено и с каким списком слов третьей модели; backends.make_asr — что запрошено
    и что получено (откаты); backends.TorchASR — запрет отката на PyTorch (он может скачивать)."""

    def __init__(self, words_mod, backends_mod, allow_torch=False):
        self.W, self.B = words_mod, backends_mod
        self.allow_torch = allow_torch
        self.passes, self.clean, self.vote, self.calls = [], [], None, []
        self._saved = []

    def make_asr(self, cfg):
        keys = ("asr_model", "asr_backend", "asr_device", "asr_batch", "asr_quant", "asr_quant_cpu",
                "asr_language")
        rec = {"requested": {k: cfg.get(k) for k in keys}}
        try:
            asr, errors = self.B.make_asr(cfg)
        except Exception as ex:
            rec["error"] = type(ex).__name__
            self.calls.append(rec)
            raise
        rec["actual"] = getattr(asr, "name", None)
        rec["fallbacks"] = [str(e)[:300] for e in errors or []]
        try:
            rec["providers"] = list(self.B.actual_providers(getattr(asr, "model", None)) or [])
        except Exception:
            rec["providers"] = []
        self.calls.append(rec)
        return asr, errors

    def __enter__(self):
        W, B = self.W, self.B
        orig_tr, orig_clean, orig_vote, orig_mark = W.transcribe, W.clean_whisper, W.vote3, W.mark_alt
        cap = self

        def transcribe(asr, audio, chunks, *args, **kwargs):
            out = orig_tr(asr, audio, chunks, *args, **kwargs)
            role = ("primary", "second")[len(cap.passes)] if len(cap.passes) < 2 else f"extra{len(cap.passes)}"
            cap.passes.append({"role": role, "model": getattr(asr, "model_name", None),
                               "backend": getattr(asr, "name", None),
                               "chunks": [{"a": _f(c["a"]), "b": _f(c["b"]),
                                           "own": [_f(c["own"][0]), _f(c["own"][1])]} for c in chunks],
                               "words": [{k: (w[k] if k == "w" else float(w[k])) for k in ("w", "s", "e", "c")
                                          if k in w} for w in out]})
            return out

        def clean_whisper(text, *args, **kwargs):
            res = orig_clean(text, *args, **kwargs)
            cap.clean.append({"raw_text": text, "text": res[0], "removed_words": res[1]})
            return res

        def vote3(main, ctc, whisper):
            cap.vote = {"function": "vote3", "third_words": list(whisper)}
            return orig_vote(main, ctc, whisper)

        def mark_alt(main, alt):
            # при частичном исключении (words.combine_hypotheses, vote3_partial) после vote3 вне
            # отрезков идёт mark_alt на каждом исключённом отрезке: запись vote3 не затирается
            if cap.vote and cap.vote.get("function") == "vote3":
                cap.vote["mark_alt_spans"] = cap.vote.get("mark_alt_spans", 0) + 1
            else:
                cap.vote = {"function": "mark_alt", "third_words": None}
            return orig_mark(main, alt)

        class NoTorch:
            def __init__(self, *a, **kw):
                raise RuntimeError("лаборатория: откат на PyTorch отключён (он может скачивать модели)")

        self._saved = [(W, "transcribe", orig_tr), (W, "clean_whisper", orig_clean), (W, "vote3", orig_vote),
                       (W, "mark_alt", orig_mark)]
        W.transcribe, W.clean_whisper, W.vote3, W.mark_alt = transcribe, clean_whisper, vote3, mark_alt
        if not self.allow_torch:
            self._saved.append((B, "TorchASR", B.TorchASR))
            B.TorchASR = NoTorch
        return self

    def __exit__(self, *exc):
        for mod, name, value in reversed(self._saved):
            setattr(mod, name, value)
        self._saved = []
        return False

    def hypotheses(self, quality, timing):
        """Гипотезы по ролям: слова со временем (основная, вторая), куски текста (третья)."""
        roles = {p["role"]: {k: v for k, v in p.items() if k != "role"} for p in self.passes}
        third = None
        wdiag = ((quality.get("asr_diagnostics") or {}).get("whisper") or {}).get("chunks")
        if wdiag is not None or self.clean:
            chunks = []
            for i, rec in enumerate(wdiag or []):
                c = self.clean[i] if i < len(self.clean) else {}
                chunks.append({"chunk_index": rec.get("chunk_index", rec.get("index")),
                               "audio_span": rec.get("audio_span"), "own_span": rec.get("own_span"),
                               "state": rec.get("state"), "raw_text": c.get("raw_text"),
                               "text": c.get("text"), "removed_words": c.get("removed_words")})
            third = {"model": timing.get("asr3_model"), "backend": timing.get("asr3_device"),
                     "name": timing.get("asr3_name"), "mode": timing.get("asr3_mode"),
                     "pieces": timing.get("asr3_pieces"), "chunks": chunks}
        legacy = {"combine_rule": combine_rule(self.W), "primary": roles.get("primary"),
                  "second": roles.get("second"), "third": third, "vote": self.vote}
        return self.record(roles, third, legacy, timing)

    def record(self, roles, third, legacy, timing):
        """Запись hyp-2 общим сборщиком программы (hypotheses.build_record); верхние ключи lab-hyp-1 те же."""
        import hypotheses as H
        blocks = []
        for name, p in roles.items():
            blocks.append({"role": name, "model": {"name": p.get("model")}, "engine": {"name": p.get("backend")},
                           "chunks": p.get("chunks"), "words": p.get("words"), "word_times": H.available()})
        if third is not None:
            blocks.append({"role": "third", "model": {"name": third.get("model")},
                           "engine": {"name": third.get("backend"), "engine_name": third.get("name"),
                                      "mode": third.get("mode"), "pieces": third.get("pieces")},
                           "timing": {k: timing.get(k) for k in ("asr3_sec", "asr3_load_sec", "asr3_compile_sec")},
                           "chunks": third.get("chunks"),
                           "words": H.not_measured("лаборатория хранит у третьей модели текст кусков (chunks)"),
                           "word_times": H.not_available("лаборатория: у третьей модели только текст")})
        return H.build_record(blocks, legacy=legacy)


# ── рабочий движок ───────────────────────────────────────────────────────────────────────────────

class LabStatus:
    """Замена файла статуса задачи: печатает смену этапа и помнит последние значения."""

    def __init__(self, prefix="", log=None):
        self.data, self.prefix = {}, prefix
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._stage = None

    def set(self, **kw):
        self.data.update(kw)
        stage = kw.get("stage")
        if stage and str(stage).split(" ")[0:3] != str(self._stage or "").split(" ")[0:3]:
            self._stage = stage
            self.log(f"[{time.strftime('%H:%M:%S')}] {self.prefix}{stage}")


class PipelineEngine:
    """Вызовы рабочего конвейера. Создавать после setup_environment."""

    def __init__(self, sandbox, allow_torch=False, log=print):
        import paths
        expected = lab_sandbox.real(Path(sandbox) / "data")
        if lab_sandbox._key(paths.DATA_DIR) != lab_sandbox._key(expected):
            raise RuntimeError(f"папка данных программы {paths.DATA_DIR} — не папка песочницы; прогон остановлен")
        self.paths = paths
        self.sandbox = Path(sandbox)
        self.allow_torch = allow_torch
        self.log = log

    def point_caches(self, program_dir):
        """Модели — из кэша установленной программы (только чтение), компиляция OpenVINO — в песочницу."""
        import backends
        cache = Path(program_dir) / "cache"
        ov = self.sandbox / "cache" / "openvino"
        ov.mkdir(parents=True, exist_ok=True)
        self.paths.CACHE, self.paths.GIGAAM_DIR, self.paths.OPENVINO_DIR = cache, cache / "gigaam", ov
        backends.GIGAAM_DIR, backends.OPENVINO_DIR = cache / "gigaam", ov

    def base_config(self):
        import settings
        return settings.load_config()

    def config_types(self):
        import settings
        return settings.CONFIG_TYPES

    def prepare_unit(self, unit, cfg):
        """Общие этапы звука: звук, словарь, разделение говорящих, голоса. Повторяет шаги
        worker.main для пословного конвейера, кроме автопополнения профиля (профиль только читается)."""
        import vocab as vocab_mod
        import voices
        import worker
        src = Path(unit["location"])
        st = LabStatus(f"{unit['unit_id']}: ", self.log)
        t0 = time.time()
        a = SimpleNamespace(group=unit.get("group") or "", vocab="", names="")
        resolution = worker.resolve_vocab_details(src, a, cfg)
        vocab_path, vocab, vgroup = resolution["path"], resolution["vocab"], resolution["group"]
        no_profile = bool(vocab_path) and not vgroup
        genders_by_name = vocab_mod.genders(vocab)
        allowed = list(genders_by_name) or [p["name"] for p in vocab.get("participants", []) if p.get("name")]
        st.set(stage="конвертация звука")
        audio = worker.load_audio(src)
        duration = len(audio) / worker.SR
        st.set(stage="разделение по спикерам")
        token = os.environ.get("HF_TOKEN", "").strip() or None
        max_spk = len(allowed) + 1 if allowed else 0
        raw, diar_info, pipe, full_ann = worker.diarize(audio, 0, token, cfg, st, max_speakers=max_spk,
                                                        min_speakers=0)
        order = {}
        for _, _, spk in sorted(raw):
            order.setdefault(spk, f"Спикер {len(order) + 1}")
        turns = [{"s": s, "e": e, "label": order[spk]} for s, e, spk in worker.build_turns(raw)]
        t_diar = time.time() - t0

        quality = {"vocab": str(vocab_path) if vocab_path else None, "vocab_resolution": resolution["report"],
                   "speaker_fixes": [], "vocab_fixes": [], "review": []}
        label_name, label_of, voice_ctx, emb = {}, {}, {}, None
        tv = time.time()
        if cfg.get("voices", True):
            try:
                st.set(stage="голоса: профили участников")
                emb = voices.Embedder(pipe)
                if no_profile:
                    prof, ppath = voices.empty_profile(), None
                    quality["voices_off"] = "общий словарь без поля group: профили голосов групп не применяются"
                else:
                    ppath = voices.profile_path(vgroup or "default")
                    prof, _ = worker.load_group_profile_for_recognition(ppath, quality)
                quality["voice_auto_learn"] = {"enabled": False, "reason": "lab_read_only"}
                embs = emb.spans(audio, [(t["s"], t["e"]) for t in turns])
                cents = voices.centroids(turns, embs)
                pvecs = voices.voice_vectors(prof, exclude_audio=src)
                mapping = voices.match(cents, pvecs, allowed or None)
                refs, label_of = voices.references(cents, turns, mapping)
                for t in turns:
                    nm = mapping.get(t["label"], (None,))[0]
                    if nm and label_of.get(nm) != t["label"]:
                        t["label"] = label_of[nm]
                label_name = {lbl: nm for nm, lbl in label_of.items()}
                voice_ctx.update(pvecs=pvecs, cents=cents, mapping=mapping,
                                 confusions=voices.confusions(prof, exclude_audio=src))
                quality.update(voice_profile=ppath.name if ppath else None, profile_names=sorted(pvecs),
                               learned_from=[],
                               match={lbl: {"name": nm, "sim": sim} for lbl, (nm, sim) in mapping.items()},
                               unmatched=[l for l in cents if l not in mapping],
                               cluster_sims={l: {n: round(float(cents[l] @ v), 3) for n, v in pvecs.items()}
                                             for l in cents})
            except Exception as ex:
                quality["voices_error"] = f"{type(ex).__name__}: {ex}"[:300]
                self.log(f"[voices] проверка голосов отключена: {ex}")
        t_voice = time.time() - tv
        turns = [t for t in turns if t["e"] - t["s"] >= worker.MIN_SEG]
        if emb is None:
            emb = voices.Embedder(pipe)
        segs = [[round(float(s), 3), round(float(e), 3), str(spk)] for s, e, spk in sorted(raw)]
        diar_sha = hashlib.sha256(lab_spec.canonical(segs).encode("utf-8")).hexdigest()
        return SimpleNamespace(
            unit=unit, src=src, audio=audio, duration=duration, raw=raw, order=order, turns=turns,
            full_ann=full_ann, emb=emb, emb_shared_sec=emb.seconds, label_name=label_name,
            label_of=label_of, voice_ctx=voice_ctx, allowed=allowed, genders_by_name=genders_by_name,
            vocab=vocab, quality=quality, no_profile=no_profile, group=None if no_profile else (vgroup or "default"),
            t_diar=t_diar, t_voice=t_voice, diar_info=diar_info, diarization_sha256=diar_sha,
            shared_record={"duration_sec": round(duration, 2), "diarization_sha256": diar_sha,
                           "diarization_segments": segs, "diarization_info": diar_info,
                           "turns": len(turns), "label_name": label_name,
                           "voices": {k: quality.get(k) for k in ("voice_profile", "profile_names", "match",
                                                                   "unmatched", "voices_error", "voices_off")},
                           "timing": {"diarization_sec": round(t_diar, 1), "voices_sec": round(t_voice, 1),
                                      "embeddings_sec": round(emb.seconds, 1)}})

    def run_config(self, shared, cfg, conf):
        """Один вызов worker.run_words. -> (данные расшифровки, гипотезы, вызовы make_asr, времена)."""
        import backends
        import words
        import worker
        st = LabStatus(f"{shared.unit['unit_id']} × {conf['id']}: ", self.log)
        a = SimpleNamespace(whisper="", whisper_max_repeat=-1, no_emotions=True)
        emb = shared.emb
        emb_before = emb.seconds
        cap = Capture(words, backends, allow_torch=self.allow_torch)
        t0 = time.time()
        with cap:
            data = worker.run_words(shared.audio, shared.src, cfg, st, shared.raw, shared.order,
                                    copy.deepcopy(shared.turns), shared.full_ann, emb,
                                    copy.deepcopy(shared.label_name), copy.deepcopy(shared.label_of),
                                    shared.voice_ctx, list(shared.allowed), dict(shared.genders_by_name),
                                    shared.vocab, copy.deepcopy(shared.quality), cap.make_asr, a)
        wall = time.time() - t0
        total = shared.t_diar + shared.t_voice + wall
        timing = {"total_sec": round(total), "pair_sec": round(wall, 1), "diarization_sec": round(shared.t_diar),
                  "voices_sec": round(shared.t_voice, 1), **data.pop("_timing"),
                  "diarization": shared.diar_info,
                  "embeddings_sec": round(shared.emb_shared_sec + emb.seconds - emb_before, 1),
                  "realtime_factor": round(shared.duration / max(total, 1e-6), 2),
                  "asr_fallbacks": st.data.get("asr_fallbacks")}
        data.update(file=str(shared.src), duration=worker.hms(shared.duration), timing=timing,
                    group=shared.group)
        if shared.no_profile:
            data["no_profile"] = True
        hyps = cap.hypotheses(data.get("quality") or {}, timing)
        return data, hyps, cap.calls, timing


# ── план, оценка времени ─────────────────────────────────────────────────────────────────────────

def _rate(sec, duration):
    return sec / duration * 3600.0 if sec and duration else None


def estimate_rates(units, db, experiment):
    """Секунды счёта на час звука по ключам этапов: посчитанные пары > история базы > файл опыта."""
    found = {}

    def add(key, value):
        if value:
            found.setdefault(key, []).append(value)

    if db is not None:
        for row in db.execute("SELECT metrics_json, config_id, config_sha256 FROM pairs WHERE state='done'"):
            m = json.loads(row["metrics_json"] or "{}")
            t, d = m.get("timing") or {}, m.get("duration_sec")
            add("diarization", _rate(t.get("diarization_sec"), d))
            add("voices", _rate(t.get("voices_sec"), d))
            add("word_voice", _rate(t.get("word_voice_sec"), d))
            add("primary:" + str(t.get("asr_model")), _rate(t.get("asr_sec"), d))
            if t.get("asr2_model"):
                add("second:" + str(t["asr2_model"]), _rate(t.get("asr2_sec"), d))
            if t.get("asr3_model") and t.get("asr3_sec"):
                name = "whisper-ct2" if str(t.get("asr3_device", "")).startswith("whisper-ct2") else t["asr3_model"]
                add(f"third:{name}:{t.get('asr3_mode', 'all')}", _rate(t["asr3_sec"], d))
    if not found:
        for u in units:
            for r in u.get("recordings") or []:
                t, d = r.get("timing") or {}, r.get("duration_sec")
                add("diarization", _rate(t.get("diarization_sec"), d))
                add("voices", _rate(t.get("voices_sec"), d))
                add("word_voice", _rate(t.get("word_voice_sec"), d))
                if t.get("asr_model"):
                    add("primary:" + t["asr_model"], _rate(t.get("asr_sec"), d))
                if t.get("asr2_model"):
                    add("second:" + t["asr2_model"], _rate(t.get("asr2_sec"), d))
                if t.get("asr3_model") and t.get("asr3_sec"):
                    name = "whisper-ct2" if t.get("asr3_device", "").startswith("whisper-ct2") else t["asr3_model"]
                    add(f"third:{name}:{t.get('asr3_mode', 'all')}", _rate(t["asr3_sec"], d))
    rates = {k: sorted(v)[len(v) // 2] for k, v in found.items()}
    return rates, dict(DEFAULT_ESTIMATES, **(experiment.get("estimate_sec_per_audio_hour") or {}))


def pair_estimate(keys, rates, defaults):
    """-> (секунд на час звука, были ли использованы грубые значения по умолчанию)."""
    rough = False

    def pick(key, fallback_key):
        nonlocal rough
        if key in rates:
            return rates[key]
        if key in defaults:
            return defaults[key]
        rough = True
        return defaults.get(fallback_key, DEFAULT_ESTIMATES[fallback_key])

    total = pick("word_voice", "word_voice") + pick(keys["primary"], "primary")
    if keys.get("second"):
        total += pick(keys["second"], "second")
    if keys.get("third"):
        mode = keys.get("third_mode", "all")
        k = f"{keys['third']}:{mode}"
        if k in rates or k in defaults:
            total += pick(k, "third")
        else:
            total += pick(f"{keys['third']}:all", "third") * (0.4 if mode == "doubt" else 1.0)
    return total, rough


# ── прогон ───────────────────────────────────────────────────────────────────────────────────────

def select(items, wanted, key):
    if not wanted:
        return list(items)
    out = [x for x in items if any(str(x[key]).startswith(w) for w in wanted)]
    unknown = [w for w in wanted if not any(str(x[key]).startswith(w) for x in items)]
    if unknown:
        raise SystemExit("не найдено: " + ", ".join(unknown))
    return out


def plan(sandbox, units, experiment, base_cfg, config_types, db, unit_filter=(), config_filter=()):
    confs = select(experiment["configurations"], config_filter, "id")
    chosen = select(units, unit_filter, "unit_id")
    prepared = []
    for conf in confs:
        cfg = lab_spec.effective_settings(base_cfg, conf, config_types)
        prepared.append((conf, cfg, lab_spec.config_hash(conf, cfg)))
    rows = []
    for u in chosen:
        for conf, cfg, sha in prepared:
            rows.append({"unit": u, "conf": conf, "cfg": cfg, "sha": sha,
                         "done": db is not None and is_done(db, u["unit_id"], conf["id"], sha)})
    return prepared, rows


def print_plan(rows, units_all, experiment, db, limit, log=print):
    rates, defaults = estimate_rates(units_all, db, experiment)
    pending = [r for r in rows if not r["done"]]
    todo = pending[:limit] if limit else pending
    total, rough_any, shared = 0.0, False, set()
    log(f"Опыт «{experiment['name']}»: пар всего {len(rows)}, посчитано {len(rows) - len(pending)}, "
        f"к счёту {len(todo)}" + (f" (ограничение --limit {limit})" if limit else ""))
    for r in rows:
        u, conf = r["unit"], r["conf"]
        hours = (u.get("duration_sec") or 0) / 3600.0
        mark = "готово" if r["done"] else ("в плане" if r in todo else "позже")
        est = ""
        if r in todo:
            sec, rough = pair_estimate(lab_spec.role_keys(conf, r["cfg"]), rates, defaults)
            sec *= hours
            if u["unit_id"] not in shared:
                shared.add(u["unit_id"])
                sec += hours * (rates.get("diarization", defaults["diarization"]) +
                                rates.get("voices", defaults["voices"]))
            total += sec
            rough_any = rough_any or rough
            est = f"~{sec / 60:.1f} мин" + (" (грубо)" if rough else "")
        name = Path(u["location"]).name
        missing = "" if Path(u["location"]).is_file() else " [аудио не найдено]"
        if u.get("origin") == lab_sandbox.AUDIO_ORIGIN:
            missing += " [файл вне базы]"
        log(f"  {u['unit_id']}  {name}{missing}  {hours * 60:5.1f} мин  {conf['id']:<16} {mark:<8} {est}")
    log(f"Оценка времени счёта: около {total / 3600:.1f} ч" +
        (" (часть этапов — по грубым значениям по умолчанию)" if rough_any else "") +
        "; источник оценок: " + ("посчитанные пары" if db is not None and
                                 db.execute("SELECT 1 FROM pairs WHERE state='done'").fetchone()
                                 else "история обработки в копии базы и файл опыта"))
    return todo, total


def run(sandbox, experiment, experiment_sha, units_doc, engine, program_dir=None, unit_filter=(),
        config_filter=(), limit=0, dry_run=False, passport_fn=None, conditions_fn=None, log=print,
        args=None):
    """Главный цикл. engine — PipelineEngine или его подмена в тестах."""
    sandbox = Path(sandbox)
    units = units_doc["units"]
    db = open_results(sandbox / "results" / "lab.sqlite3")
    try:
        base_cfg = engine.base_config()
        prepared, rows = plan(sandbox, units, experiment, base_cfg, engine.config_types(), db,
                              unit_filter, config_filter)
        todo, _ = print_plan(rows, units, experiment, db, limit, log=log)
        if dry_run:
            return {"dry_run": True, "pairs": len(rows), "todo": len(todo)}
        if not todo:
            log("Все пары уже посчитаны.")
            return {"computed": 0, "errors": 0}
        run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)
        passport = passport_fn() if passport_fn else None
        db.execute("INSERT INTO runs (run_id, started_utc, state, experiment_name, experiment_sha256, "
                   "experiment_json, passport_json, args_json) VALUES (?,?,?,?,?,?,?,?)",
                   (run_id, utc_now(), "running", experiment["name"], experiment_sha,
                    _dumps(experiment), _dumps(passport), _dumps(args or {})))
        for conf, cfg, sha in prepared:
            db.execute("INSERT OR IGNORE INTO configs VALUES (?,?,?,?,?,?,?)",
                       (conf["id"], sha, conf.get("label"), _dumps(conf["roles"]),
                        _dumps(conf.get("settings") or {}), _dumps(cfg), run_id))
        db.commit()
        computed = errors = 0
        state = "finished"
        try:
            by_unit = {}
            for r in todo:
                by_unit.setdefault(r["unit"]["unit_id"], []).append(r)
            for unit_id, unit_rows in by_unit.items():
                unit = unit_rows[0]["unit"]
                db.execute("INSERT INTO unit_runs (run_id, unit_id, started_utc, state) VALUES (?,?,?,?)",
                           (run_id, unit_id, utc_now(), "running"))
                db.commit()
                try:
                    src = Path(unit["location"])
                    if not src.is_file():
                        raise FileNotFoundError("аудиофайл записи не найден")
                    if engine_audio_sha(engine, src) != unit["audio_sha256"]:
                        raise RuntimeError("аудиофайл изменился после подготовки песочницы")
                    # общий этап — по настройкам config.json песочницы: настройки разделения
                    # говорящих и голосов конфигурация менять не может (lab_spec.SHARED_KEYS)
                    shared = engine.prepare_unit(unit, dict(base_cfg, **lab_spec.FORCED))
                except Exception as ex:
                    log(f"{unit_id}: общий этап не выполнен: {type(ex).__name__}: {ex}")
                    db.execute("UPDATE unit_runs SET finished_utc=?, state='error', error_kind=?, error=? "
                               "WHERE run_id=? AND unit_id=?",
                               (utc_now(), type(ex).__name__, traceback.format_exc()[-3000:], run_id, unit_id))
                    db.commit()
                    errors += len(unit_rows)
                    continue
                db.execute("UPDATE unit_runs SET shared_json=? WHERE run_id=? AND unit_id=?",
                           (_dumps(shared.shared_record), run_id, unit_id))
                db.commit()
                for r in unit_rows:
                    conf, cfg, sha = r["conf"], r["cfg"], r["sha"]
                    started = utc_now()
                    before = conditions_fn() if conditions_fn else None
                    try:
                        data, hyps, calls, timing = engine.run_config(shared, cfg, conf)
                        after = conditions_fn() if conditions_fn else None
                        metrics = lab_summary.pair_metrics(data, hyps, calls, timing, shared.duration)
                        cond = {"before": before, "after": after,
                                "preflight": _preflight(before, after)}
                        db.execute("INSERT INTO pairs (run_id, unit_id, config_id, config_sha256, state, "
                                   "started_utc, finished_utc, diarization_sha256, conditions_json, "
                                   "metrics_json, result_json, hypotheses_json) "
                                   "VALUES (?,?,?,?,'done',?,?,?,?,?,?,?)",
                                   (run_id, unit_id, conf["id"], sha, started, utc_now(),
                                    shared.diarization_sha256, _dumps(cond), _dumps(metrics),
                                    _dumps(data), _dumps(hyps)))
                        db.commit()
                        computed += 1
                        log(f"{unit_id} × {conf['id']}: готово за {timing.get('pair_sec')} с, "
                            f"мест для разбора {metrics['review']['total']}")
                    except KeyboardInterrupt:
                        raise
                    except Exception as ex:
                        db.execute("INSERT INTO pairs (run_id, unit_id, config_id, config_sha256, state, "
                                   "started_utc, finished_utc, error_kind, error, diarization_sha256) "
                                   "VALUES (?,?,?,?,'error',?,?,?,?,?)",
                                   (run_id, unit_id, conf["id"], sha, started, utc_now(), type(ex).__name__,
                                    traceback.format_exc()[-3000:], shared.diarization_sha256))
                        db.commit()
                        errors += 1
                        log(f"{unit_id} × {conf['id']}: ошибка {type(ex).__name__}: {ex}")
                db.execute("UPDATE unit_runs SET finished_utc=?, state='done' WHERE run_id=? AND unit_id=?",
                           (utc_now(), run_id, unit_id))
                db.commit()
                del shared
        except KeyboardInterrupt:
            state = "interrupted"
            log("Прервано: посчитанные пары сохранены, следующий запуск продолжит с оставшихся.")
        finally:
            db.execute("UPDATE runs SET finished_utc=?, state=? WHERE run_id=?", (utc_now(), state, run_id))
            db.commit()
            lab_summary.export_summary(db, units, experiment, sandbox / "export" / "summary.json")
        log(f"Посчитано пар: {computed}, с ошибкой: {errors}. Сводка: {sandbox / 'export' / 'summary.json'}")
        return {"computed": computed, "errors": errors, "run_id": run_id, "state": state}
    finally:
        db.close()


def engine_audio_sha(engine, path):
    fn = getattr(engine, "audio_sha256", None)
    if fn is not None:
        return fn(path)
    import runinfo
    return runinfo.sha256_file(path)


def _preflight(before, after):
    if not before:
        return None
    try:
        import runinfo
        return runinfo.preflight(before, after)
    except Exception:
        return None
