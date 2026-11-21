"""Числа по паре «запись × конфигурация» и обезличенная сводка для выгрузки.

Сводка строится только из перечисленных здесь числовых полей и названий из кода и файла
опыта (модели, среды исполнения, причины мест, типы ошибок). Тексты, слова гипотез, имена,
пути и имена файлов в неё не попадают: записи обозначаются контрольной суммой звука.
Формат — lab/README.md, раздел «Обезличенная сводка».
"""
import json
import re
from pathlib import Path

SUMMARY_SCHEMA = "lab-summary-1"
METRICS_SCHEMA = "lab-metrics-1"
REASONS = ("asr", "conf", "voice", "missed", "overlap", "gender")
TIMING_NUMBERS = ("total_sec", "pair_sec", "diarization_sec", "voices_sec", "embeddings_sec",
                  "asr_sec", "asr_chunks", "asr2_sec", "asr3_sec", "asr3_load_sec", "asr3_compile_sec",
                  "asr3_removed_words", "word_voice_sec", "word_windows", "interaction_sec",
                  "realtime_factor")
TIMING_NAMES = ("asr_backend", "asr_model", "asr2_model", "asr3_model", "asr3_device", "asr3_name",
                "asr3_mode", "asr3_pieces")
# поля «откатов» и ошибок: «бэкенд:устройство: ТипОшибки: сообщение» -> без сообщения
_ERR = re.compile(r"^\s*([^:]*(?::[A-Za-z0-9_+-]+)*):\s*([A-Za-z_][A-Za-z0-9_.]*)")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.:+/@-]{1,120}$")


def error_kind(text):
    """«openvino:GPU: RuntimeError: …» -> «openvino:GPU: RuntimeError»; иначе только тип."""
    text = str(text or "")
    m = _ERR.match(text)
    if m:
        return f"{m.group(1)}: {m.group(2)}"[:120]
    head = text.split(":", 1)[0].strip()
    return head[:60] if _SAFE_NAME.match(head) else "ошибка"


def safe_name(value):
    """Название из кода (модель, среда исполнения) или None, если похоже на путь или текст."""
    if value is None:
        return None
    value = str(value)
    if "\\" in value or re.match(r"^[A-Za-z]:", value) or not _SAFE_NAME.match(value):
        return None
    return value


def chunk_states(diag):
    out = {}
    for rec in (diag or {}).get("chunks") or []:
        state = str(rec.get("state"))
        out[state] = out.get(state, 0) + 1
    return out


def review_counts(items):
    by_reason = {r: 0 for r in REASONS}
    other = 0
    multi = 0
    asr_problem = 0
    for item in items or []:
        reasons = item.get("reasons") or []
        if len(reasons) > 1:
            multi += 1
        if item.get("asr_problem"):
            asr_problem += 1
        for r in reasons:
            if r in by_reason:
                by_reason[r] += 1
            else:
                other += 1
    return {"total": len(items or []), "by_reason": by_reason, "other_reasons": other,
            "multi_reason_items": multi, "asr_problem_items": asr_problem}


def pair_metrics(data, hyps, calls, timing, duration_sec):
    """Числа по паре: только счётчики, времена и названия моделей и сред исполнения."""
    quality = data.get("quality") or {}
    diag = quality.get("asr_diagnostics") or {}
    hours = (duration_sec or 0) / 3600.0
    review = review_counts(data.get("review"))
    out = {
        "schema": METRICS_SCHEMA,
        "duration_sec": round(duration_sec, 2) if duration_sec else None,
        "words": quality.get("words"),
        "flagged_words": quality.get("flagged_words"),
        "review": review,
        "review_per_hour": round(review["total"] / hours, 1) if hours else None,
        "vote": {k: v for k, v in (quality.get("vote") or {}).items() if isinstance(v, (int, float))} or None,
        "vote3_skipped": quality.get("vote3_skipped"),
        "asr_votes": len(quality.get("asr_votes") or []),
        "asr_incomplete_chunks": quality.get("asr_incomplete_chunks"),
        "asr_diagnostics": {role: {"backend": safe_name(d.get("backend")), "chunks": len(d.get("chunks") or []),
                                   "states": chunk_states(d)}
                            for role, d in diag.items() if isinstance(d, dict)},
        "asr2_error": error_kind(quality["asr2_error"]) if quality.get("asr2_error") else None,
        "asr3_errors": [error_kind(e) for e in quality.get("asr3_errors") or []],
        "missed_speech": len(quality.get("missed_speech") or []),
        "voice_doubt_threshold": quality.get("voice_doubt_threshold"),
        "overlaps_sec": quality.get("overlaps_sec"),
        "timing": {**{k: timing[k] for k in TIMING_NUMBERS if isinstance(timing.get(k), (int, float))},
                   **{k: safe_name(timing[k]) for k in TIMING_NAMES if timing.get(k) is not None}},
        "fallbacks": {"asr3": [error_kind(e) for e in timing.get("asr3_fallbacks") or []]},
        "make_asr": [{"requested": {k: safe_name(v) if isinstance(v, str) else v
                                    for k, v in (c.get("requested") or {}).items()},
                      "actual": safe_name(c.get("actual")),
                      "providers": [safe_name(p) for p in c.get("providers") or []],
                      "fallbacks": [error_kind(e) for e in c.get("fallbacks") or []],
                      "error": c.get("error")}
                     for c in calls or []],
        "hypotheses": {role: (len(h.get("words") or []) if role != "third"
                              else sum(len((ch.get("text") or "").split()) for ch in h.get("chunks") or []))
                       for role, h in (hyps or {}).items() if role in ("primary", "second", "third") and h},
    }
    return out


def export_summary(db, units, experiment=None, out_path=None):
    """Собрать обезличенную сводку из базы результатов. -> словарь (и файл, если задан путь)."""
    unit_ids = {u["unit_id"]: u for u in units}
    runs = [dict(r) for r in db.execute(
        "SELECT run_id, started_utc, finished_utc, state, experiment_name, experiment_sha256, passport_json "
        "FROM runs ORDER BY started_utc")]
    pairs = []
    for row in db.execute("SELECT run_id, unit_id, config_id, config_sha256, state, started_utc, finished_utc, "
                          "error_kind, conditions_json, metrics_json FROM pairs ORDER BY pair_id"):
        pairs.append({"run_id": row["run_id"], "unit_id": row["unit_id"], "config_id": row["config_id"],
                      "config_sha256": row["config_sha256"], "state": row["state"],
                      "started_utc": row["started_utc"], "finished_utc": row["finished_utc"],
                      "error_kind": row["error_kind"],
                      "conditions": json.loads(row["conditions_json"] or "null"),
                      "metrics": json.loads(row["metrics_json"] or "null")})
    configs = {}
    for row in db.execute("SELECT DISTINCT config_id, config_sha256, label, roles_json FROM configs"):
        configs.setdefault(row["config_id"], []).append(
            {"config_sha256": row["config_sha256"], "label": row["label"],
             "roles": json.loads(row["roles_json"])})
    summary = {
        "schema": SUMMARY_SCHEMA,
        "note": "числа без текстов, имён и путей; записи — по контрольной сумме звука",
        "experiment": ({"name": experiment.get("name")} if experiment else None),
        "runs": [{"run_id": r["run_id"], "started_utc": r["started_utc"], "finished_utc": r["finished_utc"],
                  "state": r["state"], "experiment_name": r["experiment_name"],
                  "experiment_sha256": r["experiment_sha256"],
                  "passport": json.loads(r["passport_json"] or "null")} for r in runs],
        "configs": configs,
        "units": [{"unit_id": uid, "audio_sha256": u["audio_sha256"],
                   "duration_sec": u.get("duration_sec"), "variants": len(u.get("recordings") or []),
                   "origin": "audio_file" if u.get("origin") == "audio_file" else "database"}
                  for uid, u in unit_ids.items()],
        "pairs": pairs,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(out_path)
    return summary
