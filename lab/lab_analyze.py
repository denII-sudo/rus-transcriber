"""Анализ результатов опыта лаборатории (команда analyze).

Читает только песочницу: базу результатов results/lab.sqlite3, units.json и копию базы
записей source/state.sqlite3 (только чтение). Пишет только в песочницу:
  * results/report.md   — отчёт для человека: таблицы и короткие выводы, без текстов слов;
  * results/examples.md — примеры со словами (только на этом компьютере);
  * export/analysis.json — обезличенные числа: идентификаторы конфигураций, unit_id,
    названия моделей и числа; без текстов, имён и путей.

Сведение пересчитывается штатной функцией words.combine_hypotheses (она же выбирает vote3,
частичное исключение сбойных отрезков или mark_alt) по гипотезам и диагностике кусков пары,
а выравнивание гипотез — words._ops, то есть тем же правилом, что в конвейере. Точные
определения показателей — lab/README.md, раздел «Анализ».
"""
import copy
import difflib
import json
import sqlite3
import statistics
import sys
from pathlib import Path

import lab_sandbox
import lab_summary

ANALYSIS_SCHEMA = "lab-analysis-1"
ANALYSIS_VERSION = 3          # 3: раздел work — обычная работа по гипотезам и ответам человека (lab_work)
BASELINE = "a-current"
NO_HUMAN = "решений человека нет: звук добавлен из файла вне базы записей"
# состояния сбойных кусков (words.combine_hypotheses): исключают из голосования свой отрезок
# или, у основной модели, отключают голосование целиком
BAD_STATES = ("error", "empty_unknown")
# отметка прежнего правила сведения в hypotheses_json (lab_run.combine_rule)
OLD_RULE = "whole_record"
# пригодность пары (bench-database-spec.md, раздел 8.3): пустой кусок при словах у соседа
LOST_MIN_WORDS = 5            # у GigaAM-соседа на том же куске
LOST_MIN_WORDS_THIRD = 10     # у основной модели на участке куска третьей модели
FEW_WORDS_RATIO = 0.5         # «слов в разы меньше»
FEW_WORDS_MIN = 20            # ...если у соседа хотя бы столько слов
# тенденция (bench-database-spec.md, раздел 8.6)
TREND_MIN_UNITS = 4
TREND_SHARE = 2 / 3
TREND_MIN_REL = 0.10
TREND_MIN_PLACES = 10
GAP_TOL = 0.1                 # с: допуск при наложении места без слов на решение человека
STAGES = ("diarization_sec", "voices_sec", "asr_sec", "asr2_sec", "asr3_sec", "word_voice_sec",
          "interaction_sec", "pair_sec", "total_sec")
STAGE_NAMES = {"diarization_sec": "разделение говорящих", "voices_sec": "голоса (профили)",
               "asr_sec": "основной проход (без загрузки)", "asr2_sec": "второй проход (с загрузкой)",
               "asr3_sec": "третий проход (с загрузкой)", "word_voice_sec": "голос для каждого слова",
               "interaction_sec": "признаки разговора", "pair_sec": "вызов конвейера целиком",
               "total_sec": "всего (общие этапы + конвейер)"}
SHORT = {"v3_e2e_rnnt": "RNN-T", "v3_e2e_ctc": "CTC", "v3_rnnt": "RNN-T v3", "v3_ctc": "CTC v3",
         "multilingual_large_ctc": "ML large", "multilingual_ctc": "ML", "whisper-ct2": "faster-whisper",
         "onnx-community/whisper-large-v3-turbo": "Whisper ONNX"}


def _words_module():
    if str(lab_sandbox.REPO_DIR) not in sys.path:
        sys.path.insert(0, str(lab_sandbox.REPO_DIR))
    import words
    return words


def _hyps(text):
    """Гипотезы пары: запись hyp-2 или прежняя lab-hyp-1 (верхние ключи у них одни); чужая — пустая."""
    _words_module()                       # путь к программе в sys.path
    import hypotheses
    return hypotheses.read(_loads(text, {})) if text else {}


def _loads(text, default=None):
    try:
        return json.loads(text) if text else default
    except (TypeError, ValueError):
        return default


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _per_hour(value, hours):
    return round(value / hours, 1) if hours and value is not None else None


def _share(part, whole):
    return round(part / whole, 4) if whole else None


def nm(text):
    """Безопасное название модели или роли для выгрузки (без путей и текста)."""
    return lab_summary.safe_name(text) or "?"


# ── загрузка пар ─────────────────────────────────────────────────────────────────────────────────

def load_pairs(db):
    """-> (пары по (unit_id, config_id), порядок конфигураций, сведения о конфигурациях, учёт).

    Берётся последняя версия каждой конфигурации (config_sha256 последней посчитанной пары) и
    последняя пара в состоянии done на каждую единицу. Пары с ошибкой и старых версий только
    считаются."""
    configs, order = {}, []
    for row in db.execute("SELECT rowid, config_id, config_sha256, label, roles_json, effective_json "
                          "FROM configs ORDER BY rowid"):
        if row["config_id"] not in order:
            order.append(row["config_id"])
        configs[(row["config_id"], row["config_sha256"])] = {
            "label": row["label"], "roles": _loads(row["roles_json"], {}),
            "effective": _loads(row["effective_json"], {})}
    rows = db.execute("SELECT pair_id, run_id, unit_id, config_id, config_sha256, state, error_kind, "
                      "diarization_sha256, metrics_json, result_json, hypotheses_json FROM pairs "
                      "ORDER BY pair_id").fetchall()
    latest = {}
    for r in rows:
        if r["state"] == "done":
            latest[r["config_id"]] = r["config_sha256"]
    pairs, errors, stale = {}, {}, 0
    for r in rows:
        cid = r["config_id"]
        if cid not in order:
            order.append(cid)
        if r["state"] != "done":
            errors[(r["unit_id"], cid)] = r["error_kind"] or "ошибка"
            continue
        if r["config_sha256"] != latest.get(cid):
            stale += 1
            continue
        metrics = _loads(r["metrics_json"], {}) or {}
        conf = configs.get((cid, r["config_sha256"]), {})
        pairs[(r["unit_id"], cid)] = {
            "pair_id": r["pair_id"], "unit_id": r["unit_id"], "config_id": cid,
            "config_sha256": r["config_sha256"], "diarization_sha256": r["diarization_sha256"],
            "metrics": metrics, "result": _loads(r["result_json"], {}) or {},
            "hyps": _hyps(r["hypotheses_json"]), "effective": conf.get("effective") or {},
            "roles": conf.get("roles") or {},
            "hours": (_num(metrics.get("duration_sec")) or 0) / 3600.0}
    errors = {k: v for k, v in errors.items() if k not in pairs}
    labels = {cid: next((c["label"] for (i, _), c in configs.items() if i == cid), None) for cid in order}
    return pairs, order, labels, {"errors": errors, "stale": stale}


# ── места для разбора, голосование, куски ────────────────────────────────────────────────────────

def review_categories(items):
    """Места для разбора по видам. Место с несколькими причинами в by_reason считается в каждой,
    в видах asr — один раз."""
    out = {"total": 0, "by_reason": {r: 0 for r in lab_summary.REASONS}, "other_reasons": 0,
           "asr_only": 0, "asr_mixed": 0, "asr_incomplete": 0, "no_asr": 0}
    for it in items or []:
        reasons = list(it.get("reasons") or [])
        out["total"] += 1
        for r in reasons:
            if r in out["by_reason"]:
                out["by_reason"][r] += 1
            else:
                out["other_reasons"] += 1
        problem = it.get("asr_problem") or {}
        if "asr" not in reasons:
            out["no_asr"] += 1
        elif problem and problem.get("state") != "disagreement":
            out["asr_incomplete"] += 1
        elif reasons == ["asr"]:
            out["asr_only"] += 1
        else:
            out["asr_mixed"] += 1
    return out


def chunk_states(result):
    """Состояния кусков по ролям конвейера: {роль: {состояние: число}}; у третьей — и filtered."""
    diag = (result.get("quality") or {}).get("asr_diagnostics") or {}
    out = {}
    for role, d in diag.items():
        if not isinstance(d, dict):
            continue
        st = {}
        for rec in d.get("chunks") or []:
            key = str(rec.get("state"))
            st[key] = st.get(key, 0) + 1
            if rec.get("filtered_all"):
                st["filtered"] = st.get("filtered", 0) + 1
        out[role] = st
    return out


def stage_diagnostics(result, stage):
    """Диагностика кусков одного прохода из записи конвейера (quality.asr_diagnostics[stage]).

    -> (диагностика, None) или (None, причина). Состояние куска берётся только из диагностики:
    в hypotheses_json у кусков основной и второй моделей поля state нет, и «ok» не подставляется.
    Пригодна, если у каждого куска есть state, а у сбойного — own_span (по нему
    words.combine_hypotheses исключает отрезок)."""
    diag = (result.get("quality") or {}).get("asr_diagnostics")
    if not isinstance(diag, dict):
        return None, "no_asr_diagnostics"
    d = diag.get(stage)
    if not isinstance(d, dict) or not isinstance(d.get("chunks"), list):
        return None, f"no_{stage}_diagnostics"
    for rec in d["chunks"]:
        if not isinstance(rec, dict) or not rec.get("state"):
            return None, f"{stage}_chunk_without_state"
        bad = rec["state"] in BAD_STATES or (stage == "whisper" and rec.get("filtered_all"))
        if bad and not (isinstance(rec.get("own_span"), list) and len(rec["own_span"]) == 2):
            return None, f"{stage}_chunk_without_own_span"
    return {"chunks": d["chunks"]}, None


def pair_diagnostics(result, stages):
    """Диагностика нужных проходов пары для words.combine_hypotheses. -> (словарь, None) или (None, причина)."""
    out = {}
    for stage in stages:
        d, why = stage_diagnostics(result, stage)
        if d is None:
            return None, why
        out[stage] = d
    return out, None


def combine(W, primary, second, whisper, diagnostics):
    """Сведение штатным words.combine_hypotheses — тем же правилом, что в worker.run_words.

    Слова копируются: сохранённые гипотезы не меняются. -> {"main": слова основной после
    сведения, "function": "vote3" | "mark_alt" | None, "stats", "skipped", "partial"}."""
    main = [dict(w) for w in primary]
    fused = W.combine_hypotheses({"primary": main,
                                  "second": [dict(w) for w in second] if second is not None else None,
                                  "whisper": list(whisper) if whisper is not None else None}, diagnostics)
    q = fused["quality"]
    fn = "vote3" if q.get("vote") is not None else ("mark_alt" if second is not None else None)
    return {"main": main, "function": fn, "stats": q.get("vote"), "skipped": q.get("vote3_skipped"),
            "partial": q.get("vote3_partial")}


def third_tokens(third, W):
    """Слова третьей модели в том виде, в каком конвейер подаёт их в vote3 (режим all):
    куски по порядку, текст после clean_whisper, токены без букв и цифр отброшены."""
    toks = []
    for ch in sorted(third.get("chunks") or [], key=lambda c: c.get("chunk_index") or 0):
        text = ch.get("text")
        if text:
            toks += [t for t in text.split() if W.norm_word(t)]
    return toks


def estimate_places(words, gap=1):
    """Оценка мест «спор моделей»: подряд идущие спорные слова, между которыми не больше gap
    согласных (правило words.review_items), без деления на реплики и без других причин."""
    flagged = [not w.get("agree", True) for w in words]
    n, k, places = len(words), 0, 0
    while k < n:
        if not flagged[k]:
            k += 1
            continue
        j = k
        while True:
            nxt = next((m for m in range(j + 1, min(n, j + 2 + gap)) if flagged[m]), None)
            if nxt is None:
                break
            j = nxt
        places += 1
        k = j + 1
    return places


def flag_check(main, result, W):
    """Пометка `asr` у слов стенограммы против «несогласия» слов после сведения.
    -> (слов сверено, слов не найдено, расхождений пометки)."""
    checked = unmatched = flag_mism = 0
    pool = {}
    for w in main:
        pool.setdefault((round(float(w["s"]), 2), W.norm_word(w["w"])), []).append(not w.get("agree", True))
    for seg in result.get("segments") or []:
        for wd in seg.get("words") or []:
            key = (round(float(wd.get("s", -1)), 2), W.norm_word(str(wd.get("raw", wd.get("w", "")))))
            lst = pool.get(key)
            if not lst:
                unmatched += 1
                continue
            checked += 1
            if lst.pop(0) != ("asr" in (wd.get("flags") or [])):
                flag_mism += 1
    return checked, unmatched, flag_mism


def old_rule_record(p, W, new, prim, alt):
    """Сделана ли запись конвейера прежним правилом сведения (до PR №76: любой сбойный кусок
    второй или третьей модели отключал голосование трёх на всей записи).

    Отметка lab_run.combine_rule в гипотезах решает сразу: нынешнее правило — не прежнее. Без
    отметки (база записана до её появления) прежнее правило признаётся, только если запись
    в точности такая, какой её дало бы прежнее правило: нынешнее частично исключает отрезки
    (vote3_partial), а записаны mark_alt по всей записи с vote3_skipped той же причины, без
    итогов голосования, и пометки `asr` у всех слов совпали со сверкой mark_alt по всей записи."""
    h, q = p["hyps"], p["result"].get("quality") or {}
    if h.get("combine_rule") not in (None, OLD_RULE):
        return False
    if new["function"] != "vote3" or not new["partial"]:
        return False
    if (h.get("vote") or {}).get("function") != "mark_alt" or q.get("vote") or q.get("vote3_partial"):
        return False
    if q.get("vote3_skipped") != new["partial"].get("reason"):
        return False
    whole = [dict(w) for w in prim]
    W.mark_alt(whole, [dict(w) for w in alt])
    checked, _, mism = flag_check(whole, p["result"], W)
    return checked > 0 and mism == 0


def own_revote(p, W):
    """Пересведение исходного набора пары нынешним правилом (words.combine_hypotheses) и сверка
    с тем, что записал конвейер."""
    h, result = p["hyps"], p["result"]
    q = result.get("quality") or {}
    program = (q.get("asr_run") or {}).get("program")
    base = {"pipeline_program": program if isinstance(program, str) else None,
            "combine_rule": h.get("combine_rule")}
    prim = (h.get("primary") or {}).get("words")
    if prim is None:
        return {"state": "no_hypotheses", **base}
    alt = (h.get("second") or {}).get("words") if h.get("second") else None
    third, vote = h.get("third"), h.get("vote") or {}
    finished = bool(third) and third.get("pieces") is not None
    # диагностика — та же, что конвейер подал в combine_hypotheses: все записанные проходы и
    # обязательно те, чьи гипотезы есть
    stored = q.get("asr_diagnostics") if isinstance(q.get("asr_diagnostics"), dict) else {}
    need = {"primary"} | ({"second"} if alt is not None else set()) | ({"whisper"} if finished else set())
    stages = [s for s in ("primary", "second", "whisper") if s in need or s in stored]
    diag, why = pair_diagnostics(result, stages)
    if diag is None:
        return {"state": "no_diagnostics", "detail": why, "primary_words": len(prim), **base}
    tokens, rebuilt = None, None
    if finished and alt is not None:
        if third.get("mode") == "doubt":
            tokens = vote.get("third_words") if vote.get("function") == "vote3" else None
        else:
            rebuilt = tokens = third_tokens(third, W)
    whisper = tokens if finished else None
    if finished and alt is not None and tokens is None:
        # слов третьей модели нет (режим doubt, конвейер не голосовал). Если и нынешнее правило
        # не голосовало бы, итог от них не зависит (mark_alt); иначе пересчитать нельзя.
        if combine(W, prim, alt, [], diag)["function"] == "vote3":
            return {"state": "not_reconstructible", "detail": "doubt_third_words_unknown",
                    "primary_words": len(prim), **base}
        whisper = []
    new = combine(W, prim, alt, whisper, diag)
    mism = []
    if new["function"] != vote.get("function"):
        mism.append("function")
    if (q.get("vote3_skipped") or None) != new["skipped"]:
        mism.append("vote3_skipped")
    if (q.get("vote3_partial") or None) != new["partial"]:
        mism.append("vote3_partial")
    if new["function"] == "vote3":
        rec = {k: v for k, v in (q.get("vote") or {}).items() if isinstance(v, (int, float))}
        if rec != new["stats"]:
            mism.append("vote_stats")
        if rebuilt is not None and vote.get("third_words") is not None and rebuilt != vote["third_words"]:
            mism.append("third_words")
    checked, unmatched, flag_mism = flag_check(new["main"], result, W)
    if flag_mism:
        mism.append("word_flags")
    state = "ok"
    if mism:
        state = "old_rule" if old_rule_record(p, W, new, prim, alt) else "mismatch"
    main = new["main"]
    out = {"state": state, "mismatch": mism, "function": new["function"], "stats": new["stats"],
           "skipped": new["skipped"], "partial": new["partial"],
           "disputed_words": sum(1 for w in main if not w.get("agree", True)),
           "places_estimate": estimate_places(main), "words_checked": checked,
           "words_unmatched": unmatched, "word_flag_mismatch": flag_mism, "primary_words": len(prim), **base}
    if state == "old_rule":
        out["recorded"] = {"function": vote.get("function"), "skipped": q.get("vote3_skipped")}
    return out


def vote_summary(p, own):
    """Итоги голосования пары: нынешним правилом — из записи конвейера (vote3) или пересчётом
    (mark_alt; запись прежним правилом). Без пересчёта — как записал конвейер."""
    q = p["result"].get("quality") or {}
    prim = len(((p["hyps"].get("primary") or {}).get("words")) or [])
    if own.get("state") == "old_rule" and own.get("function") == "vote3":
        return {"function": "vote3", "source": "recomputed", "primary_words": prim, **own["stats"]}
    rec = {k: v for k, v in (q.get("vote") or {}).items() if isinstance(v, (int, float))}
    if rec:
        return {"function": "vote3", "source": "pipeline", "primary_words": prim, **rec}
    if own.get("function") == "mark_alt":
        d = own["disputed_words"]
        return {"function": "mark_alt", "source": "recomputed", "primary_words": prim, "agree3": None,
                "agree2": prim - d, "replaced": 0, "inserted": 0, "disputed": d}
    return {"function": None, "source": None, "primary_words": prim}


# ── пригодность пары ─────────────────────────────────────────────────────────────────────────────

def _count_in(words, span):
    s, e = span
    return sum(1 for w in words if s - 1e-3 <= float(w["s"]) < e + 1e-3)


def pair_fitness(p, W):
    """Признаки технического сбоя сборки в паре. -> {"fit": bool, "reasons": [коды], "details": {...}}."""
    h, m, result = p["hyps"], p["metrics"], p["result"]
    reasons, det = [], {}
    for role, st in chunk_states(result).items():
        if st.get("error"):
            reasons.append(f"{role}:error_chunks")
            det[f"{role}_error_chunks"] = st["error"]
    prim = (h.get("primary") or {}).get("words") or []
    sec = (h.get("second") or {}).get("words") if h.get("second") else None
    chunks = (h.get("primary") or {}).get("chunks") or []
    if sec is not None:
        lost = {"primary": 0, "second": 0}
        for ch in chunks:
            a, b = _count_in(prim, ch["own"]), _count_in(sec, ch["own"])
            if a == 0 and b >= LOST_MIN_WORDS:
                lost["primary"] += 1
            if b == 0 and a >= LOST_MIN_WORDS:
                lost["second"] += 1
        for role, n in lost.items():
            if n:
                reasons.append(f"{role}:lost_chunks")
                det[f"{role}_lost_chunks"] = n
        if len(prim) >= FEW_WORDS_MIN and len(sec) < FEW_WORDS_RATIO * len(prim):
            reasons.append("second:few_words")
        if len(sec) >= FEW_WORDS_MIN and len(prim) < FEW_WORDS_RATIO * len(sec):
            reasons.append("primary:few_words")
    third = h.get("third")
    if third and third.get("pieces") is not None:
        lost3, total3 = 0, 0
        for ch in third.get("chunks") or []:
            if ch.get("state") == "error" or not ch.get("own_span"):
                continue
            n3 = len([t for t in (ch.get("text") or "").split() if W.norm_word(t)])
            total3 += n3
            if n3 == 0 and _count_in(prim, ch["own_span"]) >= LOST_MIN_WORDS_THIRD:
                lost3 += 1
        if lost3:
            reasons.append("whisper:lost_chunks")
            det["whisper_lost_chunks"] = lost3
        if third.get("mode") == "all" and len(prim) >= FEW_WORDS_MIN and total3 < FEW_WORDS_RATIO * len(prim):
            reasons.append("whisper:few_words")
    for call in m.get("make_asr") or []:
        req = call.get("requested") or {}
        if call.get("fallbacks") or call.get("error"):
            reasons.append(f"fallback:{req.get('asr_model')}")
        dev = str(req.get("asr_device") or "").upper()
        actual = str(call.get("actual") or "").upper()
        if dev in ("GPU", "NPU") and actual and dev not in actual and "DML" not in actual:
            reasons.append(f"device_mismatch:{req.get('asr_model')}")
    if (m.get("fallbacks") or {}).get("asr3") or m.get("asr3_errors"):
        reasons.append("fallback:third")
    want = str(p["effective"].get("asr_third_backend") or "")
    got = str((m.get("timing") or {}).get("asr3_device") or "")
    if third and want == "whisper-ct2" and got and not got.startswith("whisper-ct2"):
        reasons.append("engine_mismatch:third")
    if m.get("asr2_error"):
        reasons.append("second:error")
    reasons = sorted(set(reasons))
    return {"fit": not reasons, "reasons": reasons, "details": det}


# ── источники гипотез и пересведение ─────────────────────────────────────────────────────────────

def _diag_fields(result, stage):
    """Диагностика кусков прохода, давшего источник: она идёт в combine_hypotheses той роли,
    в которой источник участвует в сочетании. diag None — диагностики нет (diag_missing — почему)."""
    d, why = stage_diagnostics(result, stage)
    return {"diag": d, "diag_missing": why}


def pair_sources(p, W):
    """Гипотезы пары как источники для сведения: ключ «роль:модель» -> описание."""
    h, result = p["hyps"], p["result"]
    out = {}
    prim = h.get("primary")
    if prim and prim.get("words") is not None:
        out["primary:" + nm(prim.get("model"))] = {
            "kind": "timed", "role": "primary", "model": nm(prim.get("model")), "words": prim["words"],
            "chunks": prim.get("chunks") or [], **_diag_fields(result, "primary")}
    sec = h.get("second")
    if sec and sec.get("words") is not None:
        out["second:" + nm(sec.get("model"))] = {
            "kind": "timed", "role": "second", "model": nm(sec.get("model")), "words": sec["words"],
            "chunks": sec.get("chunks") or [], **_diag_fields(result, "second")}
    third = h.get("third")
    if third and third.get("pieces") is not None:
        engine = str(third.get("backend") or "")
        model = "whisper-ct2" if engine.startswith("whisper-ct2") else nm(third.get("model"))
        if third.get("mode") == "doubt":
            vote = h.get("vote") or {}
            tokens = vote.get("third_words") if vote.get("function") == "vote3" else None
            out[f"third:{model}:doubt"] = {
                "kind": "text", "role": "third", "model": model, "mode": "doubt", "tokens": tokens,
                "bound_second": next((k for k in out if k.startswith("second:")), None),
                **_diag_fields(result, "whisper")}
        else:
            out[f"third:{model}"] = {"kind": "text", "role": "third", "model": model, "mode": "all",
                                     "tokens": third_tokens(third, W), **_diag_fields(result, "whisper")}
    for src in out.values():
        src["pair"] = p
    return out


def _signature(src):
    if src["kind"] == "timed":
        return [(w["w"], round(float(w["s"]), 2)) for w in src["words"]]
    return src.get("tokens")


def unit_sources(unit_pairs, order, W):
    """Источники единицы: первая пара по порядку конфигураций (базовая — первой) даёт источник,
    остальные сверяются с ним. -> (источники, сверка одинаковости)."""
    ordered = sorted(unit_pairs, key=lambda p: (p["config_id"] != BASELINE,
                                                order.index(p["config_id"]) if p["config_id"] in order else 99))
    canon, consistency = {}, {}
    for p in ordered:
        for key, src in pair_sources(p, W).items():
            if key not in canon:
                canon[key] = src
                consistency[key] = {"pairs": 1, "same": 1, "different": []}
                continue
            c = consistency[key]
            c["pairs"] += 1
            if _signature(src) == _signature(canon[key]) and \
                    src["pair"]["diarization_sha256"] == canon[key]["pair"]["diarization_sha256"]:
                c["same"] += 1
            else:
                c["different"].append(p["config_id"])
    return canon, consistency


def as_text_tokens(src, W, max_repeat):
    """Источник как список слов третьей модели. Модель со временами слов подаётся как текст по
    своим кускам через words.third_text — тем же правилом, что в конвейере: чистка «субтитров» и
    зацикливания только для Whisper (transcript.asr_model_caps), текст другой модели — как есть."""
    if src["kind"] == "text":
        return src.get("tokens")
    import transcript              # путь к программе добавлен _words_module
    whisper = transcript.asr_model_caps(src.get("model"))["whisper"]
    toks = []
    for ch in src.get("chunks") or []:
        s, e = ch["own"]
        text = " ".join(w["w"] for w in src["words"] if s - 1e-3 <= float(w["s"]) < e + 1e-3)
        tx, _ = W.third_text(text, max_repeat, whisper)
        toks += [t for t in tx.split() if W.norm_word(t)]
    return toks


def combo_validity(P, X, Y):
    """Можно ли свести эти источники вместе. -> None или причина (код)."""
    srcs = [s for s in (P, X, Y) if s is not None]
    if len({s["pair"]["diarization_sha256"] for s in srcs}) > 1:
        return "diarization_differs"
    base = _signature(P)
    for s in srcs[1:]:
        hp = (s["pair"]["hyps"].get("primary") or {}).get("words")
        if hp is not None and [(w["w"], round(float(w["s"]), 2)) for w in hp] != base:
            return "primary_differs"
    if Y is not None and Y.get("mode") == "doubt":
        # в режиме doubt третья модель считала только куски, где основная расходится со СВОЕЙ
        # второй; на непосчитанных кусках в vote3 поданы слова основной. С другой второй моделью
        # набор кусков был бы другим — такое сочетание не сводится.
        own = pair_second(Y)
        if X is None or X["kind"] != "timed" or own is None or _signature(X) != _signature(own):
            return "doubt_needs_own_second"
        if Y.get("tokens") is None:
            return "doubt_not_reconstructible"
    return None


def pair_second(src):
    sec = src["pair"]["hyps"].get("second")
    return {"kind": "timed", "words": sec["words"]} if sec and sec.get("words") is not None else None


def combo_key(*keys):
    return "|".join(k for k in keys if k)


def combo_label(key):
    parts = []
    for k in key.split("|"):
        role, model = k.split(":", 1)
        doubt = model.endswith(":doubt")
        model = model[:-6] if doubt else model
        name = SHORT.get(model, model)
        if role == "third":
            name += " (текст" + (", только куски расхождения" if doubt else "") + ")"
        parts.append(name)
    return " + ".join(parts)


def revote_unit(canon, W, max_repeat):
    """Все сочетания «основная + вторая» и «основная + вторая + третья» по источникам единицы."""
    prims = [k for k in canon if k.startswith("primary:")]
    others = [k for k in canon if not k.startswith("primary:")]
    out = {}
    for pk in prims:
        P = canon[pk]
        for xk in others:
            X = canon[xk]
            if X.get("mode") == "doubt":
                continue
            alt = X["words"] if X["kind"] == "timed" else [{"w": t} for t in (X.get("tokens") or [])]
            key = combo_key(pk, xk)
            bad = combo_validity(P, X, None)
            if bad:
                out[key] = {"state": "invalid", "reason": bad}
                continue
            if X["kind"] == "timed":
                if P["diag"] is None or X["diag"] is None:
                    out[key] = {"state": "invalid", "reason": "no_diagnostics"}
                    continue
                diag = {"primary": P["diag"], "second": X["diag"]}
            else:
                # вторая без времён слов (только пересведением): без третьей правило — mark_alt по
                # всей записи, от сбойных кусков оно не зависит; отрезки по времени здесь не исключить
                diag = {}
            out[key] = _combo_result(combine(W, P["words"], alt, None, diag), P, [X],
                                     text_second=X["kind"] != "timed")
            for yk in others:
                Y = canon[yk]
                if yk == xk or X["kind"] != "timed" or Y["model"] == X["model"]:
                    continue
                key3 = combo_key(pk, xk, yk)
                bad = combo_validity(P, X, Y)
                if not bad and Y["diag"] is None:
                    bad = "no_diagnostics"
                if bad:
                    out[key3] = {"state": "invalid", "reason": bad}
                    continue
                tokens = as_text_tokens(Y, W, max_repeat)
                out[key3] = _combo_result(combine(W, P["words"], X["words"], tokens, dict(diag, whisper=Y["diag"])),
                                          P, [X, Y], approx=Y["kind"] == "timed")
    return out


def _combo_result(res, P, others, text_second=False, approx=False):
    main = res["main"]
    disputed = sum(1 for w in main if not w.get("agree", True))
    return {"state": "ok", "function": res["function"], "stats": res["stats"], "skipped": res["skipped"],
            "partial": res["partial"], "disputed_words": disputed, "places_estimate": estimate_places(main),
            "primary_words": len(P["words"]), "text_second": text_second, "approx_third": approx,
            "fit": all(lab_fit(s["pair"]) for s in [P] + others)}


def lab_fit(pair):
    return pair.get("_fitness", {}).get("fit", True)


# ── попарное согласие ────────────────────────────────────────────────────────────────────────────

def norm_list(src, W):
    if src["kind"] == "timed":
        return [W.norm_word(w["w"]) for w in src["words"]]
    return [W.norm_word(t) for t in src.get("tokens") or []]


def pairwise_unit(canon, W):
    """Доля слов A, с которыми согласна B (words._ops), и число мест расхождения."""
    keys = [k for k in canon if not k.endswith(":doubt")]
    lists = {k: norm_list(canon[k], W) for k in keys}
    out = {}
    for a in keys:
        for b in keys:
            if a == b:
                continue
            agree, bad = W._ops(lists[a], lists[b])
            out[f"{a}~{b}"] = {"words": len(agree), "agree": sum(agree), "places": len(bad)}
    # «похожие ошибки»: где судья R расходится с основной, согласна ли с основной вторая X
    similar = {}
    prims = [k for k in keys if k.startswith("primary:")]
    for pk in prims:
        P = lists[pk]
        ag = {k: W._ops(P, lists[k])[0] for k in keys if k != pk}
        for xk in [k for k in ag if k.startswith("second:")]:
            for rk in ag:
                if xk == rk or canon[xk]["model"] == canon[rk]["model"]:
                    continue
                dis = [i for i, ok in enumerate(ag[rk]) if not ok]
                similar[f"{pk}~{xk}~{rk}"] = {"referee_disagrees": len(dis),
                                              "second_sides_with_primary": sum(1 for i in dis if ag[xk][i])}
    return out, similar


# ── решения человека ─────────────────────────────────────────────────────────────────────────────

def open_readonly(path):
    """База результатов песочницы только на чтение; без служебных файлов — неизменяемой.

    Копию базы записей анализ открывает штатно (storage.open_store_readonly, load_human);
    здесь — база результатов lab.sqlite3: у неё своя схема, штатная проверка схемы не подходит.
    Правило то же, что у storage: при -wal/-shm/-journal рядом — обычное mode=ro."""
    path = Path(path)
    uri = path.resolve().as_uri() + "?mode=ro"
    if not any(path.with_name(path.name + s).exists() for s in ("-wal", "-shm", "-journal")):
        uri += "&immutable=1"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    return db


def _human(mark):
    return isinstance(mark, dict) and mark.get("by") == "human"


def recording_decisions(transcript, review, W):
    """Решения человека в одной записи (только автор «человек», не ИИ-агент и не модель).

    -> {"H": нормализованные слова итоговой стенограммы, "words": сами слова,
        "decisions": [...], "stats": {...}}"""
    words = sorted((w for seg in transcript.get("segments") or [] for w in seg.get("words") or []
                    if isinstance(w, dict)), key=lambda w: float(w.get("s", 0) or 0))
    Hw = [w for w in words if W.norm_word(str(w.get("w", "")))]
    H = [W.norm_word(str(w["w"])) for w in Hw]
    stats = {"speaker_only": 0, "unmatched_edits": 0, "agent_edits": 0}
    groups, loose = {}, []
    for i, w in enumerate(Hw):
        prov = w.get("provenance") if isinstance(w.get("provenance"), dict) else {}
        if _human(prov.get("text")):
            groups.setdefault(str(prov["text"].get("edit")), []).append(i)
        elif w.get("by") == "human":
            if "text" not in prov and _human(prov.get("speaker")):
                stats["speaker_only"] += 1
            elif w.get("edited"):
                groups.setdefault(str(w["edited"]), []).append(i)
            else:
                loose.append(i)
    owner = {i: c for c, idx in groups.items() for i in idx}
    changed = True
    while changed and loose:           # слова «человек» без своей метки текста — к соседнему решению
        changed = False
        for i in list(loose):
            c = owner.get(i - 1) or owner.get(i + 1)
            if c is not None:
                groups[c].append(i)
                owner[i] = c
                loose.remove(i)
                changed = True
    run = []
    for i in loose + [None]:
        if run and (i is None or i != run[-1] + 1):
            groups[f"legacy-{run[0]}"] = run
            run = []
        if i is not None:
            run.append(i)
    edits = [e for e in transcript.get("edits") or [] if isinstance(e, dict)]
    human_edit_ids = {str(e.get("id")) for e in edits if e.get("by") == "human"}
    stats["agent_edits"] = sum(1 for e in edits if e.get("by") not in ("human", "reverted"))
    items = {str(it.get("id")): it for it in (review or {}).get("review") or [] if isinstance(it, dict)}
    decisions = []
    for card, idx in groups.items():
        h1, h2 = min(idx), max(idx) + 1
        decisions.append({"card": card, "kind": "changed" if card in human_edit_ids else "confirmed",
                          "h1": h1, "h2": h2, "a": float(Hw[h1]["s"]), "b": float(Hw[h2 - 1].get("e", Hw[h2 - 1]["s"])),
                          "final": H[h1:h2], "text": " ".join(str(w["w"]) for w in Hw[h1:h2])})

    def position(t):
        return next((i for i, w in enumerate(Hw) if float(w["s"]) >= t), len(Hw))

    for e in edits:
        if e.get("by") != "human" or str(e.get("id")) in groups:
            continue
        after = (e.get("after") or {}).get("text") if isinstance(e.get("after"), dict) else e.get("after")
        before = (e.get("before") or {}).get("text") if isinstance(e.get("before"), dict) else e.get("before")
        if not "".join(W.norm_word(x) for x in str(after or "").split()) and str(before or "").strip():
            t = float(e.get("t") or 0)
            it = items.get(str(e.get("id"))) or {}
            pos = position(t)
            decisions.append({"card": str(e.get("id")), "kind": "deleted", "h1": pos, "h2": pos, "a": t,
                              "b": max(t, float(it.get("t2", t) or t)) + 0.5, "final": [], "text": "",
                              "before": str(before)})
        else:
            stats["unmatched_edits"] += 1
    taken = {d["card"] for d in decisions}
    for card, it in items.items():
        lis = it.get("listening")
        if card in taken or not _human(lis):
            continue
        gap = it.get("gap") or [it.get("t1"), it.get("t2")]
        try:
            g0, g1 = float(gap[0]), float(gap[1])
        except (TypeError, ValueError, IndexError):
            continue
        inside = [i for i, w in enumerate(Hw) if g0 <= float(w["s"]) < g1]
        if lis.get("result") == "silence" or not inside:
            pos = position(g0)
            decisions.append({"card": card, "kind": "listened", "h1": pos, "h2": pos, "a": g0, "b": g1,
                              "final": [], "text": ""})
        else:
            h1, h2 = min(inside), max(inside) + 1
            decisions.append({"card": card, "kind": "listened", "h1": h1, "h2": h2, "a": g0, "b": g1,
                              "final": H[h1:h2], "text": " ".join(str(w["w"]) for w in Hw[h1:h2])})
    status = {}
    for it in items.values():
        s = str(it.get("status") or "open")
        status[s] = status.get(s, 0) + 1
    stats["review_items"] = status
    decisions.sort(key=lambda d: d["a"])
    return {"H": H, "words": Hw, "decisions": decisions, "stats": stats}


def load_human(db_path, units, W):
    """Решения человека по единицам. -> ({unit_id: {"variants", "decisions", "stats"}}, причина «не измерено»)."""
    db_path = Path(db_path)
    if not db_path.is_file():
        return None, "нет копии базы записей source/state.sqlite3"
    # копия базы записей — штатным чтением программы: проверка схемы и правило «без -wal/-shm —
    # неизменяемая». Копию делает Store.backup, служебных файлов рядом с ней нет.
    import lab_sandbox
    store = lab_sandbox._import_repo_module("storage").open_store_readonly(db_path)
    db = store.db
    try:
        out = {}
        for u in units:
            variants = []
            for rec in u.get("recordings") or []:
                row = db.execute(
                    "SELECT t.payload AS tp, v.payload AS rp FROM recordings r "
                    "JOIN transcript_revisions t ON t.recording_id=r.recording_id AND t.revision=r.source_revision "
                    "LEFT JOIN review_revisions v ON v.recording_id=r.recording_id AND v.revision=r.review_revision "
                    "WHERE r.recording_id=?", (rec["recording_id"],)).fetchone()
                if row is None:
                    continue
                variants.append(recording_decisions(_loads(row["tp"], {}) or {}, _loads(row["rp"], {}) or {}, W))
            # варианты одной записи: одно и то же место, решённое в двух вариантах одинаково, —
            # одно решение; решённое по-разному — спорное, в сопоставление не идёт
            merged, dups = [], 0
            for vi, var in enumerate(variants):
                for d in var["decisions"]:
                    d = dict(d, variant=vi)
                    clash = [m for m in merged if m["variant"] != vi and d["a"] < m["b"] and m["a"] < d["b"]]
                    if not clash:
                        merged.append(d)
                    elif all(equal_norm(m["final"], d["final"], W) for m in clash):
                        dups += 1
                    else:
                        for m in clash:
                            m["conflict"] = True
            kept = [d for d in merged if not d.get("conflict")]
            stats = {"variants": len(variants), "decisions": len(kept),
                     "conflicting_between_variants": sum(1 for d in merged if d.get("conflict")),
                     "duplicates_between_variants": dups}
            for key in ("speaker_only", "unmatched_edits", "agent_edits"):
                stats[key] = sum(v["stats"][key] for v in variants)
            out[u["unit_id"]] = {"variants": variants, "decisions": kept, "stats": stats}
        return out, None
    finally:
        store.close()


def equal_norm(a, b, W):
    ma, mb = "".join(a), "".join(b)
    return ma == mb or (ma, mb) in W.SAME or (mb, ma) in W.SAME


def map_range(ops, h1, h2):
    """Индексы слов гипотезы, выровненные с [h1, h2) итоговой стенограммы."""
    js = []
    for op, i1, i2, j1, j2 in ops:
        # совпадение и замена равной длины — слово к слову; иначе участок замены целиком
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            lo, hi = max(i1, h1), min(i2, h2)
            if lo < hi:
                js.append((j1 + lo - i1, j1 + hi - i1))
        elif op == "insert":
            if h1 < i1 < h2 or h1 == h2 == i1:
                js.append((j1, j2))
        elif (i1 < h2 and i2 > h1) or (h1 == h2 and i1 < h1 < i2):
            js.append((j1, j2))
    if not js:
        return None
    return min(j for j, _ in js), max(j for _, j in js)


def output_words(result, W):
    """Итоговые слова конфигурации: [(норма, есть ли пометка, есть ли пометка asr)] по времени."""
    ws = sorted((w for seg in result.get("segments") or [] for w in seg.get("words") or []),
                key=lambda w: float(w.get("s", 0)))
    out = []
    for w in ws:
        n = W.norm_word(str(w.get("w", "")))
        if n:
            flags = w.get("flags") or []
            out.append((n, bool(flags), "asr" in flags, str(w.get("w"))))
    gaps = []
    for it in result.get("review") or []:
        if "missed" in (it.get("reasons") or []) or it.get("asr_problem"):
            gap = it.get("gap") or [it.get("t1"), it.get("t2")]
            try:
                gaps.append((float(gap[0]), float(gap[1]), "asr" in (it.get("reasons") or [])))
            except (TypeError, ValueError, IndexError):
                pass
    return out, gaps


def human_analysis(human, pairs_by_unit, canon_by_unit, order, W):
    """«Кто был прав» и что было бы с решёнными местами при каждой конфигурации."""
    src_tot, cfg_tot, none_right, shared = {}, {}, {"decisions": 0, "none_right": 0}, {}
    per_unit, examples = {}, []
    for uid, hu in human.items():
        if not hu["decisions"]:
            continue
        canon = canon_by_unit.get(uid) or {}
        lists = {k: [x for x in norm_list(s, W) if x] for k, s in canon.items()}
        raw = {k: ([w["w"] for w in s["words"] if W.norm_word(w["w"])] if s["kind"] == "timed"
                   else [t for t in s.get("tokens") or [] if W.norm_word(t)]) for k, s in canon.items()}
        ops = {}

        def aligned(vi, key, M):
            if (vi, key) not in ops:
                ops[(vi, key)] = difflib.SequenceMatcher(None, hu["variants"][vi]["H"], M,
                                                         autojunk=False).get_opcodes()
            return ops[(vi, key)]

        outs = {p["config_id"]: output_words(p["result"], W) for p in pairs_by_unit.get(uid, [])}
        urow = {"decisions": len(hu["decisions"]), "by_kind": {}, "sources": {}, "configs": {}, "none_right": 0}
        for d in hu["decisions"]:
            urow["by_kind"][d["kind"]] = urow["by_kind"].get(d["kind"], 0) + 1
            ex = {"unit_id": uid, "t": round(d["a"], 2), "kind": d["kind"], "human": d["text"], "sources": {},
                  "configs": {}}
            right_any = False
            texts = {}
            for key, M in lists.items():
                if canon[key].get("tokens") is None and canon[key]["kind"] == "text":
                    continue
                rng = map_range(aligned(d["variant"], key, M), d["h1"], d["h2"])
                got = M[rng[0]:rng[1]] if rng else []
                texts[key] = got
                ok = equal_norm(got, d["final"], W)
                right_any = right_any or ok
                row = src_tot.setdefault(key, {})
                cell = row.setdefault(d["kind"], {"right": 0, "wrong": 0})
                cell["right" if ok else "wrong"] += 1
                us = urow["sources"].setdefault(key, {"right": 0, "wrong": 0})
                us["right" if ok else "wrong"] += 1
                ex["sources"][key] = " ".join(raw[key][rng[0]:rng[1]]) if rng else ""
            none_right["decisions"] += 1
            if not right_any:
                none_right["none_right"] += 1
                urow["none_right"] += 1
            pk = next((k for k in texts if k.startswith("primary:")), None)
            if pk is not None and not equal_norm(texts[pk], d["final"], W):
                for key, got in texts.items():
                    if key == pk:
                        continue
                    cell = shared.setdefault(key, {"primary_wrong": 0, "same_as_primary": 0, "right": 0})
                    cell["primary_wrong"] += 1
                    if equal_norm(got, texts[pk], W):
                        cell["same_as_primary"] += 1
                    elif equal_norm(got, d["final"], W):
                        cell["right"] += 1
            for cid, (O, gaps) in outs.items():
                M = [o[0] for o in O]
                rng = map_range(aligned(d["variant"], "cfg:" + cid, M), d["h1"], d["h2"])
                seg = O[rng[0]:rng[1]] if rng else []
                shown = any(o[1] for o in seg) or any(g0 < d["b"] + GAP_TOL and d["a"] - GAP_TOL < g1
                                                      for g0, g1, _ in gaps)
                by_asr = any(o[2] for o in seg) or any(g0 < d["b"] + GAP_TOL and d["a"] - GAP_TOL < g1 and asr
                                                       for g0, g1, asr in gaps)
                ok = equal_norm([o[0] for o in seg], d["final"], W)
                verdict = "shown" if shown else ("spared_right" if ok else "silent_error")
                cell = cfg_tot.setdefault(cid, {"decisions": 0, "shown": 0, "shown_by_asr": 0,
                                                "spared_right": 0, "silent_error": 0, "output_right": 0})
                cell["decisions"] += 1
                cell[verdict] += 1
                cell["shown_by_asr"] += int(by_asr)
                cell["output_right"] += int(ok)
                uc = urow["configs"].setdefault(cid, {"shown": 0, "spared_right": 0, "silent_error": 0})
                uc[verdict] += 1
                ex["configs"][cid] = {"text": " ".join(o[3] for o in seg), "verdict": verdict}
            examples.append(ex)
        per_unit[uid] = urow
    return {"sources": src_tot, "configs": cfg_tot, "none_right": none_right, "shared_error": shared,
            "units": per_unit}, examples


# ── сводка по конфигурациям ──────────────────────────────────────────────────────────────────────

def unit_row(p, own, fit):
    m = p["metrics"]
    rev = review_categories(p["result"].get("review"))
    timing = m.get("timing") or {}
    q = p["result"].get("quality") or {}
    # отметки сведения — нынешним правилом: пересчётом, если он есть; иначе как записал конвейер
    if own.get("state") in ("ok", "old_rule"):
        skipped, partial = own.get("skipped"), own.get("partial")
    else:
        skipped, partial = q.get("vote3_skipped"), q.get("vote3_partial")
    return {"hours": round(p["hours"], 4), "words": m.get("words"), "review": rev,
            "vote": vote_summary(p, own), "vote3_skipped": skipped, "vote3_partial": partial,
            "chunk_states": chunk_states(p["result"]),
            "timing": {k: timing.get(k) for k in STAGES if _num(timing.get(k)) is not None},
            "fit": fit["fit"], "unfit_reasons": fit["reasons"],
            "revote_check": own.get("state")}


def _add_partial(acc, partial):
    """Добавить отметку vote3_partial записи к сумме: записей, отрезков, секунд, по причинам."""
    if not isinstance(partial, dict):
        return
    acc["units"] += 1
    acc["spans"] += int(partial.get("spans") or 0)
    acc["seconds"] += float(partial.get("seconds") or 0)
    reason = str(partial.get("reason"))
    acc["by_reason"][reason] = acc["by_reason"].get(reason, 0) + 1


def aggregate(rows):
    """Сумма по единицам (не среднее процентов)."""
    hours = sum(r["hours"] for r in rows)
    agg = {"units": len(rows), "hours": round(hours, 3)}
    rev = {"total": 0, "asr_only": 0, "asr_mixed": 0, "asr_incomplete": 0, "no_asr": 0, "other_reasons": 0,
           "by_reason": {r: 0 for r in lab_summary.REASONS}}
    vote = {"agree3": 0, "agree2": 0, "replaced": 0, "inserted": 0, "disputed": 0, "primary_words": 0}
    fn_units, skipped, states, timing, tcount = {}, {}, {}, {}, {}
    partial = {"units": 0, "spans": 0, "seconds": 0.0, "by_reason": {}}
    checks = {}
    for r in rows:
        for k in rev:
            if k == "by_reason":
                for rr, n in r["review"]["by_reason"].items():
                    rev["by_reason"][rr] = rev["by_reason"].get(rr, 0) + n
            else:
                rev[k] += r["review"].get(k, 0)
        v = r["vote"]
        fn_units[str(v.get("function"))] = fn_units.get(str(v.get("function")), 0) + 1
        if v.get("function"):
            for k in vote:
                vote[k] += v.get(k) or 0
        if r["vote3_skipped"]:
            skipped[r["vote3_skipped"]] = skipped.get(r["vote3_skipped"], 0) + 1
        _add_partial(partial, r.get("vote3_partial"))
        checks[str(r["revote_check"])] = checks.get(str(r["revote_check"]), 0) + 1
        for role, st in r["chunk_states"].items():
            dst = states.setdefault(role, {})
            for s, n in st.items():
                dst[s] = dst.get(s, 0) + n
        for k, val in r["timing"].items():
            timing[k] = timing.get(k, 0) + val
            tcount[k] = tcount.get(k, 0) + 1
    agg["review"] = rev
    agg["review_per_hour"] = {k: _per_hour(v, hours) for k, v in rev.items() if k != "by_reason"}
    agg["review_per_hour"]["by_reason"] = {k: _per_hour(v, hours) for k, v in rev["by_reason"].items()}
    agg["vote"] = vote
    base = vote["primary_words"]
    agg["vote_share"] = {k: _share(vote[k], base) for k in ("agree3", "agree2", "replaced", "disputed")}
    agg["vote_function_units"] = fn_units
    agg["vote3_skipped"] = {"units": sum(skipped.values()), "by_reason": skipped}
    partial["seconds"] = round(partial["seconds"], 1)
    agg["vote3_partial"] = partial
    agg["revote_check"] = checks
    agg["chunk_states"] = states
    # время на час — только по единицам, где этап измерен у всех; иначе «не измерено»
    agg["timing_per_hour"] = {k: (round(timing[k] / hours, 1) if tcount.get(k) == len(rows) and hours
                                  else None) for k in STAGES}
    agg["timing_partial"] = sorted(k for k in STAGES if 0 < tcount.get(k, 0) < len(rows))
    return agg


def trend(base_vals, cand_vals, lower_better=True, min_abs=0):
    """Сравнение на общих единицах по правилу bench-database-spec.md, раздел 8.6."""
    n = len(base_vals)
    tb, tc = sum(base_vals), sum(cand_vals)
    diffs = [c - b for b, c in zip(base_vals, cand_vals)]
    better = sum(1 for d in diffs if (d < 0 if lower_better else d > 0))
    worse = sum(1 for d in diffs if (d > 0 if lower_better else d < 0))
    out = {"units": n, "base_total": tb, "cand_total": tc, "diff_total": tc - tb,
           "rel": round((tc - tb) / tb, 4) if tb else None,
           "units_better": better, "units_worse": worse, "units_equal": n - better - worse}
    if n < TREND_MIN_UNITS:
        out["verdict"] = "мало данных"
    elif better >= TREND_SHARE * n and tb and (tb - tc if lower_better else tc - tb) >= max(TREND_MIN_REL * tb, min_abs):
        out["verdict"] = "тенденция: лучше"
    elif worse >= TREND_SHARE * n and tb and (tc - tb if lower_better else tb - tc) >= max(TREND_MIN_REL * tb, min_abs):
        out["verdict"] = "тенденция: хуже"
    else:
        out["verdict"] = "различия нет"
    return out


def compare(rows_by_cfg, cid, baseline, only_fit):
    base, cand = rows_by_cfg.get(baseline, {}), rows_by_cfg.get(cid, {})
    common = sorted(u for u in base if u in cand and (not only_fit or (base[u]["fit"] and cand[u]["fit"])))
    if not common:
        return {"units": 0, "not_measured": "нет общих посчитанных единиц с базовой конфигурацией"}
    hours = sum(base[u]["hours"] for u in common)
    out = {"units": len(common), "hours": round(hours, 3)}
    for name, get in (("review_total", lambda r: r["review"]["total"]),
                      ("review_asr_only", lambda r: r["review"]["asr_only"]),
                      ("review_asr_all", lambda r: r["review"]["asr_only"] + r["review"]["asr_mixed"]
                       + r["review"]["asr_incomplete"])):
        t = trend([get(base[u]) for u in common], [get(cand[u]) for u in common], True, TREND_MIN_PLACES)
        t["per_hour_base"] = _per_hour(t["base_total"], hours)
        t["per_hour_cand"] = _per_hour(t["cand_total"], hours)
        t["per_unit"] = {u: get(cand[u]) - get(base[u]) for u in common}
        out[name] = t
    tt = [u for u in common if _num(base[u]["timing"].get("total_sec")) and _num(cand[u]["timing"].get("total_sec"))]
    if len(tt) == len(common):
        ratios = [cand[u]["timing"]["total_sec"] / base[u]["timing"]["total_sec"] for u in tt]
        faster = sum(1 for r in ratios if r < 1)
        slower = sum(1 for r in ratios if r > 1)
        med = statistics.median(ratios)
        verdict = "мало данных" if len(tt) < TREND_MIN_UNITS else (
            "тенденция: быстрее" if faster >= TREND_SHARE * len(tt) and med <= 1 - TREND_MIN_REL else
            "тенденция: медленнее" if slower >= TREND_SHARE * len(tt) and med >= 1 + TREND_MIN_REL else
            "различия нет")
        out["total_sec"] = {"units": len(tt), "median_ratio": round(med, 3), "units_faster": faster,
                            "units_slower": slower, "verdict": verdict,
                            "note": "полоса шума не измерена: повторов нет",
                            "per_hour_base": _per_hour(sum(base[u]["timing"]["total_sec"] for u in tt), hours),
                            "per_hour_cand": _per_hour(sum(cand[u]["timing"]["total_sec"] for u in tt), hours)}
    else:
        out["total_sec"] = {"not_measured": "время записано не у всех общих единиц"}
    return out


def aggregate_combos(per_unit, hours_by_unit, only_fit):
    out = {}
    for uid, combos in per_unit.items():
        for key, r in combos.items():
            a = out.setdefault(key, {"units": 0, "hours": 0.0, "disputed_words": 0, "places_estimate": 0,
                                     "primary_words": 0, "functions": {}, "vote3_skipped": {},
                                     "vote3_partial": {"units": 0, "spans": 0, "seconds": 0.0, "by_reason": {}},
                                     "invalid": {}, "excluded_unfit": 0, "text_second": False,
                                     "approx_third": False})
            if r["state"] != "ok":
                a["invalid"][r["reason"]] = a["invalid"].get(r["reason"], 0) + 1
                continue
            if only_fit and not r["fit"]:
                a["excluded_unfit"] += 1
                continue
            a["units"] += 1
            a["hours"] += hours_by_unit.get(uid, 0)
            a["disputed_words"] += r["disputed_words"]
            a["places_estimate"] += r["places_estimate"]
            a["primary_words"] += r["primary_words"]
            a["functions"][str(r["function"])] = a["functions"].get(str(r["function"]), 0) + 1
            if r["skipped"]:
                a["vote3_skipped"][r["skipped"]] = a["vote3_skipped"].get(r["skipped"], 0) + 1
            _add_partial(a["vote3_partial"], r.get("partial"))
            a["text_second"] = a["text_second"] or r["text_second"]
            a["approx_third"] = a["approx_third"] or r["approx_third"]
    for a in out.values():
        a["hours"] = round(a["hours"], 3)
        a["vote3_partial"]["seconds"] = round(a["vote3_partial"]["seconds"], 1)
        a["places_per_hour"] = _per_hour(a["places_estimate"], a["hours"])
        a["disputed_per_hour"] = _per_hour(a["disputed_words"], a["hours"])
        a["disputed_share"] = _share(a["disputed_words"], a["primary_words"])
    return out


# ── главный вход ─────────────────────────────────────────────────────────────────────────────────

def analyze(sandbox, units_doc, baseline=BASELINE, log=print):
    """Посчитать анализ по песочнице. -> словарь analysis (тот же, что в export/analysis.json)."""
    import lab_run
    W = _words_module()
    sandbox = Path(sandbox)
    units = units_doc.get("units") or []
    import lab_work
    work, work_examples = lab_work.analyze_work(sandbox, W)
    db_path = sandbox / "results" / "lab.sqlite3"
    if not db_path.is_file():
        analysis = {"schema": ANALYSIS_SCHEMA, "version": ANALYSIS_VERSION,
                    "not_measured": "нет базы результатов results/lab.sqlite3: прогон не выполнялся",
                    "work": work}
        write_outputs(sandbox, analysis, [], {"work_examples": work_examples})
        return analysis
    db = open_readonly(db_path)
    try:
        row = db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if row is None or row["value"] != lab_run.RESULTS_SCHEMA:
            raise RuntimeError("база результатов другой версии или повреждена: анализ невозможен")
        pairs, order, labels, acc = load_pairs(db)
    finally:
        db.close()
    max_repeat = 3
    for p in pairs.values():
        if p["config_id"] == baseline and _num(p["effective"].get("asr_third_max_repeat")) is not None:
            max_repeat = int(p["effective"]["asr_third_max_repeat"])
    rows_by_cfg, checks, fitness, pairs_by_unit = {}, {}, {}, {}
    for (uid, cid), p in sorted(pairs.items()):
        fit = pair_fitness(p, W)
        p["_fitness"] = fit
        own = own_revote(p, W)
        rows_by_cfg.setdefault(cid, {})[uid] = unit_row(p, own, fit)
        checks[f"{uid}|{cid}"] = {k: own.get(k) for k in ("state", "mismatch", "function", "skipped",
                                                           "partial", "detail", "recorded",
                                                           "combine_rule", "pipeline_program",
                                                           "disputed_words", "places_estimate",
                                                           "words_checked", "words_unmatched",
                                                           "word_flag_mismatch")}
        checks[f"{uid}|{cid}"]["pipeline_asr_places"] = (rows_by_cfg[cid][uid]["review"]["asr_only"]
                                                         + rows_by_cfg[cid][uid]["review"]["asr_mixed"])
        fitness[f"{uid}|{cid}"] = fit
        pairs_by_unit.setdefault(uid, []).append(p)
    canon_by_unit, consistency, revote, pairwise, similar = {}, {}, {}, {}, {}
    hours_by_unit = {}
    for uid, plist in pairs_by_unit.items():
        canon, cons = unit_sources(plist, order, W)
        canon_by_unit[uid] = canon
        consistency[uid] = cons
        hours_by_unit[uid] = max(p["hours"] for p in plist)
        revote[uid] = revote_unit(canon, W, max_repeat)
        pairwise[uid], similar[uid] = pairwise_unit(canon, W)
    configs = {}
    for cid in order:
        rows = rows_by_cfg.get(cid, {})
        configs[cid] = {"label_known": labels.get(cid) is not None,
                        "all": aggregate(list(rows.values())) if rows else {"not_measured": "нет посчитанных пар"},
                        "fit_only": (aggregate([r for r in rows.values() if r["fit"]])
                                     if any(r["fit"] for r in rows.values()) else
                                     {"not_measured": "нет пригодных пар"}),
                        "units": rows}
    comparison = {cid: {"all": compare(rows_by_cfg, cid, baseline, False),
                        "fit_only": compare(rows_by_cfg, cid, baseline, True)}
                  for cid in order if cid != baseline}
    # единицы из файлов вне базы записей (add-audio, prepare --audio): решений человека у них нет
    no_human = {u["unit_id"]: NO_HUMAN for u in units if u.get("origin") == "audio_file"}
    db_units = [u for u in units if u.get("origin") != "audio_file"]
    with_pairs_db = [u for u in db_units if u["unit_id"] in pairs_by_unit]
    if not with_pairs_db and pairs_by_unit:
        human, reason = None, NO_HUMAN
    else:
        human, reason = load_human(sandbox / "source" / "state.sqlite3", db_units, W)
    examples = []
    if human is None:
        human_part = {"not_measured": reason}
        if no_human:
            human_part["units_not_measured"] = no_human
    else:
        human_part, examples = human_analysis(human, pairs_by_unit, canon_by_unit, order, W)
        human_part["units_stats"] = {uid: h["stats"] for uid, h in human.items()}
        human_part["warning"] = ("выборка мала и сделана по старым вариантам записей: числа абсолютные, "
                                 "это повод для опыта, а не вывод")
        if no_human:
            human_part["units_not_measured"] = no_human
    pw_total, sim_total = {}, {}
    for uid in pairwise:
        for k, v in pairwise[uid].items():
            t = pw_total.setdefault(k, {"units": 0, "words": 0, "agree": 0, "places": 0, "hours": 0.0})
            t["units"] += 1
            t["hours"] += hours_by_unit[uid]
            for f in ("words", "agree", "places"):
                t[f] += v[f]
        for k, v in similar[uid].items():
            t = sim_total.setdefault(k, {"units": 0, "referee_disagrees": 0, "second_sides_with_primary": 0})
            t["units"] += 1
            for f in ("referee_disagrees", "second_sides_with_primary"):
                t[f] += v[f]
    for t in pw_total.values():
        t["agree_share"] = _share(t["agree"], t["words"])
        t["places_per_hour"] = _per_hour(t["places"], t["hours"])
        t["hours"] = round(t["hours"], 3)
    for t in sim_total.values():
        t["share"] = _share(t["second_sides_with_primary"], t["referee_disagrees"])
    original = {}
    for (uid, cid), p in pairs.items():
        keys = sorted(pair_sources(p, W), key=lambda k: ("primary", "second", "third").index(k.split(":")[0]))
        original.setdefault(cid, combo_key(*keys))
    analysis = {
        "schema": ANALYSIS_SCHEMA, "version": ANALYSIS_VERSION, "baseline": baseline,
        "definitions": "lab/README.md, раздел «Анализ»",
        "data": {"units_in_sandbox": len(units), "units_with_pairs": len(pairs_by_unit),
                 "hours_with_pairs": round(sum(hours_by_unit.values()), 3), "pairs_done": len(pairs),
                 "pairs_error_without_done": len(acc["errors"]), "pairs_old_versions": acc["stale"],
                 "errors": {f"{u}|{c}": lab_summary.error_kind(k) for (u, c), k in acc["errors"].items()},
                 "config_order": order},
        "configs": configs, "comparison": comparison,
        "pairwise": {"total": pw_total, "similar_errors": sim_total, "units": pairwise},
        "sources_consistency": {uid: {k: {"pairs": v["pairs"], "same": v["same"], "different": v["different"]}
                                      for k, v in c.items()} for uid, c in consistency.items()},
        "revote": {"original_combo": original, "max_repeat": max_repeat,
                   "all": aggregate_combos(revote, hours_by_unit, False),
                   "fit_only": aggregate_combos(revote, hours_by_unit, True), "units": revote},
        "revote_check": checks,
        "human": human_part,
        "fitness": fitness,
        "work": work,
    }
    write_outputs(sandbox, analysis, examples, {"labels": labels, "work_examples": work_examples})
    return analysis


# ── вывод ────────────────────────────────────────────────────────────────────────────────────────

def _atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_outputs(sandbox, analysis, examples, extra):
    _atomic(sandbox / "export" / "analysis.json", json.dumps(analysis, ensure_ascii=False, indent=1) + "\n")
    _atomic(sandbox / "results" / "report.md", render_report(analysis, extra.get("labels") or {}))
    import lab_work
    text = render_examples(examples)
    if extra.get("work_examples") is not None and isinstance(analysis.get("work"), dict) \
            and not analysis["work"].get("not_measured"):
        text = text.rstrip("\n") + "\n\n" + "\n".join(lab_work.render_examples(extra["work_examples"]))
    _atomic(sandbox / "results" / "examples.md", text)


def _fmt(x, nd=1):
    if x is None:
        return "не измерено"
    if isinstance(x, float):
        return f"{x:.{nd}f}".replace(".", ",")
    return str(x)


def _pct(x):
    return "не измерено" if x is None else f"{x * 100:.1f} %".replace(".", ",")


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def src_label(key):
    return combo_label(key)


def render_report(a, labels):
    L = ["# Анализ опыта моделей", "",
         "Отчёт только для этого компьютера. Тексты слов — в соседнем файле `examples.md`. Определения "
         "показателей — `lab/README.md`, раздел «Анализ». «Не измерено» — показатель не считался, а не ноль.", ""]
    if a.get("not_measured"):
        return "\n".join(L + [f"**Не измерено:** {a['not_measured']}.", ""] + _work_section(a))
    d = a["data"]
    base = a["baseline"]
    L += ["## Данные", "",
          f"- Единиц (звуков) в песочнице: {d['units_in_sandbox']}, с посчитанными парами: {d['units_with_pairs']}, "
          f"звука в них: {_fmt(d['hours_with_pairs'], 2)} ч.",
          f"- Пар учтено: {d['pairs_done']}; с ошибкой и без удачного пересчёта: {d['pairs_error_without_done']}; "
          f"старых версий конфигураций (не учтены): {d['pairs_old_versions']}.",
          f"- Базовая конфигурация для сравнения: `{base}`.", ""]
    L += _conclusions(a)
    order = d["config_order"]
    for mode, title in (("all", "все пары"), ("fit_only", "без технически сбойных пар")):
        L += [f"## Места для разбора на час звука ({title})", "",
              "Сумма мест по единицам, делённая на сумму часов. «Только спор» — у места одна причина: "
              "модели разошлись; «спор и другое» — есть и другие причины; «неполный проход» — место "
              "из-за ошибки или пустого куска модели.", ""]
        rows = []
        for cid in order:
            ag = a["configs"][cid][mode]
            if ag.get("not_measured"):
                rows.append([f"`{cid}`", ag["not_measured"]] + [""] * 8)
                continue
            ph = ag["review_per_hour"]
            rows.append([f"`{cid}`", ag["units"], _fmt(ag["hours"], 2), _fmt(ph["total"]), _fmt(ph["asr_only"]),
                         _fmt(ph["asr_mixed"]), _fmt(ph["asr_incomplete"]), _fmt(ph["by_reason"]["conf"]),
                         _fmt(ph["by_reason"]["voice"]), _fmt(ph["by_reason"]["missed"])])
        L += _table(["конфигурация", "единиц", "часов", "всего", "только спор", "спор и другое",
                     "неполный проход", "низкая уверенность", "близкие голоса", "возможный пропуск"], rows)
    units_all = sorted({u for cid in order for u in (a["configs"][cid].get("units") or {})})
    if units_all:
        L += ["## Места для разбора по единицам", "",
              "В ячейке: мест всего / из них спор моделей (только спор и спор с другими причинами); "
              "звёздочка — пара с признаками технического сбоя; «—» — пара не посчитана.", ""]
        rows = []
        for uid in units_all:
            row = [f"`{uid}`"]
            for cid in order:
                r = (a["configs"][cid].get("units") or {}).get(uid)
                if r is None:
                    row.append("—")
                    continue
                rv = r["review"]
                row.append(f"{rv['total']} / {rv['asr_only'] + rv['asr_mixed']}" + ("" if r["fit"] else " *"))
            rows.append(row)
        L += _table(["единица"] + [f"`{c}`" for c in order], rows)
    L += ["## Итоги голосования", "",
          "Доли — от числа слов основной модели (сумма по единицам). У конфигураций без третьей модели "
          "согласие двух моделей пересчитано по сохранённым гипотезам. Отметки сведения — нынешним "
          "правилом (`words.combine_hypotheses`): «голосование отключилось» — на всей записи (сбой "
          "куска основной модели или вне сбойных отрезков нет слов); «частично» — сбойные куски второй "
          "или третьей модели исключены из голосования только на своих отрезках, там сверены две "
          "модели. При частичном исключении счётчики голосования — только вне исключённых отрезков.", ""]
    rows = []
    for cid in order:
        ag = a["configs"][cid]["all"]
        if ag.get("not_measured"):
            continue
        s = ag["vote_share"]
        sk = ag["vote3_skipped"]
        rows.append([f"`{cid}`", ag["vote"]["primary_words"], _pct(s["agree3"]), _pct(s["agree2"]),
                     _pct(s["replaced"]), ag["vote"]["inserted"], _pct(s["disputed"]),
                     f"{sk['units']} из {ag['units']}" + (" (" + ", ".join(f"{k}: {v}" for k, v in
                                                            sk["by_reason"].items()) + ")" if sk["units"] else ""),
                     _partial_text(ag["vote3_partial"], ag["units"])])
    L += _table(["конфигурация", "слов основной", "согласны все три", "согласна одна", "заменено",
                 "вставлено слов", "спорных слов", "голосование отключилось",
                 "частично: записей (отрезков, секунд)"], rows)
    old = {cid: a["configs"][cid]["all"].get("revote_check", {}).get("old_rule", 0) for cid in order
           if not a["configs"][cid]["all"].get("not_measured")}
    if any(old.values()):
        L += ["Записей конвейера, сделанных прежним правилом сведения (до частичного исключения): "
              + ", ".join(f"`{c}` — {n}" for c, n in old.items() if n) + ". Отметки и итоги голосования "
              "в них пересчитаны нынешним правилом; места для разбора в таблицах выше — как записал "
              "конвейер, то есть прежним правилом (оценка нынешним — в «Сверке», столбец «мест спора "
              "(оценка)»).", ""]
    L += ["## Состояние кусков по моделям", "",
          "`ok` — есть текст, `empty_unknown` — пустой ответ, `error` — ошибка, `filtered` — у третьей "
          "модели весь текст убран фильтром ложных фраз.", ""]
    rows = []
    for cid in order:
        ag = a["configs"][cid]["all"]
        for role, st in sorted((ag.get("chunk_states") or {}).items()):
            rows.append([f"`{cid}`", role, st.get("ok", 0), st.get("empty_unknown", 0), st.get("error", 0),
                         st.get("filtered", 0)])
    L += _table(["конфигурация", "роль", "ok", "пустые", "ошибки", "отфильтрованы"], rows)
    L += ["## Время машины на час звука, с", "",
          "Общие этапы (разделение говорящих, голоса) считаются один раз на звук и одинаковы у всех "
          "конфигураций. Повторов нет, полоса шума не измерена.", ""]
    rows = []
    for cid in order:
        ag = a["configs"][cid]["all"]
        if ag.get("not_measured"):
            continue
        rows.append([f"`{cid}`"] + [_fmt(ag["timing_per_hour"][k], 0) for k in STAGES])
    L += _table(["конфигурация"] + [STAGE_NAMES[k] for k in STAGES], rows)
    L += [f"## Сравнение с `{base}` на одних и тех же единицах", "",
          f"Разность — сумма мест конфигурации минус сумма мест базовой на общих единицах. Тенденция — "
          f"в одну сторону не меньше чем на {round(TREND_SHARE * 100)} % единиц и суммарно не меньше "
          f"{round(TREND_MIN_REL * 100)} % и {TREND_MIN_PLACES} мест; меньше {TREND_MIN_UNITS} единиц — "
          "«мало данных» (bench-database-spec.md, раздел 8.6).", ""]
    rows = []
    for cid, cmp in a["comparison"].items():
        for mode, title in (("all", "все"), ("fit_only", "без сбойных")):
            c = cmp[mode]
            if c.get("not_measured"):
                rows.append([f"`{cid}`", title, c["not_measured"], "", "", ""])
                continue
            r, ra = c["review_total"], c["review_asr_all"]
            t = c["total_sec"]
            rows.append([f"`{cid}`", title, c["units"],
                         f"{r['diff_total']:+d} ({_fmt(r['per_hour_base'])} → {_fmt(r['per_hour_cand'])} на час); "
                         f"лучше {r['units_better']}, хуже {r['units_worse']}, равно {r['units_equal']}; {r['verdict']}",
                         f"{ra['diff_total']:+d}; лучше {ra['units_better']}, хуже {ra['units_worse']}; {ra['verdict']}",
                         t.get("not_measured") or f"медиана ×{_fmt(t['median_ratio'], 2)}; быстрее "
                         f"{t['units_faster']}, медленнее {t['units_slower']}; {t['verdict']}"])
    L += _table(["конфигурация", "пары", "единиц", "мест всего", "мест спора моделей", "полное время"], rows)
    L += _pairwise_section(a)
    L += _revote_section(a)
    L += _human_section(a)
    L += _fitness_section(a)
    L += _work_section(a)
    return "\n".join(L)


def _work_section(a):
    import lab_work
    work = a.get("work")
    return lab_work.render_section(work) if isinstance(work, dict) else []


def _conclusions(a):
    out = ["## Коротко", ""]
    base = a["baseline"]
    rows = {cid: c["all"] for cid, c in a["configs"].items() if not c["all"].get("not_measured")}
    if rows:
        best = min(rows, key=lambda c: rows[c]["review_per_hour"]["total"] or 0)
        out.append(f"- Меньше всего мест для разбора на час: `{best}` — "
                   f"{_fmt(rows[best]['review_per_hour']['total'])}"
                   + (f"; у базовой `{base}` — {_fmt(rows[base]['review_per_hour']['total'])}." if base in rows else "."))
        for cid, ag in rows.items():
            rev = ag["review"]
            if rev["total"]:
                out.append(f"- `{cid}`: спор моделей даёт {_pct(_share(rev['asr_only'] + rev['asr_mixed'], rev['total']))} "
                           f"мест, только спор — {_pct(_share(rev['asr_only'], rev['total']))}.")
    for cid, cmp in a["comparison"].items():
        c = cmp["all"]
        if not c.get("not_measured"):
            out.append(f"- `{cid}` против `{base}`: мест всего {c['review_total']['diff_total']:+d} на "
                       f"{c['units']} единицах ({c['review_total']['verdict']}).")
    bad = [k for k, v in a["revote_check"].items() if v["state"] == "mismatch"]
    if bad:
        out.append(f"- **Внимание:** пересведение не совпало с конвейером в {len(bad)} парах — числа "
                   "пересведения в этих единицах недостоверны (раздел «Сверка»).")
    old = [k for k, v in a["revote_check"].items() if v["state"] == "old_rule"]
    if old:
        out.append(f"- Запись конвейера сделана прежним правилом сведения в {len(old)} парах: это смена "
                   "правила, а не ошибка данных; итоги сведения посчитаны нынешним правилом, места для "
                   "разбора конвейера в них — прежним (раздел «Сверка»).")
    nodiag = [k for k, v in a["revote_check"].items() if v["state"] in ("no_diagnostics", "not_reconstructible")]
    if nodiag:
        out.append(f"- Не пересчитано нынешним правилом: {len(nodiag)} пар (нет диагностики кусков или слов "
                   "третьей модели) — в них отметки сведения взяты из записи конвейера (раздел «Сверка»).")
    unfit = [k for k, v in a["fitness"].items() if not v["fit"]]
    if unfit:
        out.append(f"- Пар с признаками технического сбоя: {len(unfit)} (раздел «Пригодность»).")
    h = a["human"]
    if h.get("not_measured"):
        out.append(f"- Решения человека: не измерено — {h['not_measured']}.")
    else:
        nr = h["none_right"]
        out.append(f"- Решений человека сопоставлено: {nr['decisions']}; ни одна модель не дала решённого "
                   f"текста в {nr['none_right']} из них (абсолютные числа, выборка мала).")
    return out + [""]


def _pairwise_section(a):
    L = ["## Попарное согласие моделей", "",
         "Доля слов модели A, с которыми согласна модель B (выравнивание `words._ops`, как в конвейере: "
         "склейки и разговорные варианты не считаются расхождением). Места расхождения — участки, где "
         "модели разошлись, на час звука.", ""]
    rows = []
    for k, t in sorted(a["pairwise"]["total"].items()):
        x, y = k.split("~")
        rows.append([src_label(x), src_label(y), t["units"], t["words"], _pct(t["agree_share"]),
                     _fmt(t["places_per_hour"])])
    L += _table(["A", "B", "единиц", "слов A", "согласна B", "мест расхождения на час"], rows)
    L += ["«Похожие ошибки» (гипотеза РАР-4): среди слов основной модели, с которыми не согласен судья R, "
          "доля тех, где вторая X согласна с основной. Если у X того же семейства доля выше, чем у "
          "независимой, — X чаще повторяет основную там, где третья видит иное. Это согласие, не точность.", ""]
    rows = []
    for k, t in sorted(a["pairwise"]["similar_errors"].items()):
        p, x, r = k.split("~")
        rows.append([src_label(x), src_label(r), t["units"], t["referee_disagrees"],
                     t["second_sides_with_primary"], _pct(t["share"])])
    L += _table(["вторая X", "судья R", "единиц", "R против основной, слов", "из них X с основной", "доля"], rows)
    return L


def _revote_section(a):
    rv = a["revote"]
    L = ["## Пересведение сохранённых гипотез", "",
         "Для каждой единицы сведение пересчитано штатной `words.combine_hypotheses` (то же правило, что "
         "в конвейере: `vote3`, частичное исключение сбойных отрезков или `mark_alt`) с диагностикой кусков "
         "пар-источников для всех сочетаний сохранённых гипотез. «Спорных слов» — слова, которые ушли бы "
         "человеку как спор моделей; «мест (оценка)» — те же слова, сведённые в места правилом "
         "`review_items` без деления на реплики и без других причин. Числа сочетаний сравнимы между "
         "собой только на одинаковом наборе единиц (столбец «единиц»).", ""]
    orig = {v: k for k, v in rv["original_combo"].items()}
    rows = []
    for key, t in sorted(rv["all"].items(), key=lambda kv: (kv[0].count("|"), kv[0])):
        notes = []
        if key in orig:
            notes.append(f"как `{orig[key]}`")
        if t["text_second"]:
            notes.append("вторая без времён слов — только пересведением")
        if t["approx_third"]:
            notes.append("третья — слова модели со временами, поданные текстом (приближённо)")
        if t["invalid"]:
            notes.append("не сводится: " + ", ".join(f"{INVALID_TEXT.get(r, r)} ({n})" for r, n in t["invalid"].items()))
        if t["vote3_skipped"]:
            notes.append("голосование отключилось бы: " + ", ".join(f"{r} ({n})" for r, n in t["vote3_skipped"].items()))
        if t["vote3_partial"]["units"]:
            notes.append("частично: " + _partial_text(t["vote3_partial"], t["units"]))
        fo = rv["fit_only"].get(key) or {}
        rows.append([combo_label(key), t["units"], _fmt(t["hours"], 2), t["disputed_words"],
                     _pct(t["disputed_share"]), _fmt(t["places_per_hour"]),
                     _fmt(fo.get("places_per_hour")), "; ".join(notes)])
    L += _table(["сочетание", "единиц", "часов", "спорных слов", "доля спорных", "мест (оценка) на час",
                 "то же без сбойных", "примечание"], rows)
    L += ["### Сверка пересведения с конвейером", "",
          "Пересведение исходного набора каждой пары нынешним правилом (`words.combine_hypotheses` с "
          "диагностикой кусков пары) сравнивается с тем, что записал конвейер: функция сведения, "
          "причины отключения и частичного исключения, итоги голосования, слова третьей модели и "
          "пометка `asr` у каждого слова стенограммы. Несовпадение — признак недостоверности "
          "пересведения этой единицы. «Прежнее правило» — не ошибка: запись сделана до частичного "
          "исключения отрезков, и это подтверждено сверкой с прежним итогом; дальше она считается "
          "нынешним правилом. «Нет диагностики» — пара не пересчитана (не выдаётся за совпадение).", ""]
    rows = []
    for k, v in sorted(a["revote_check"].items()):
        uid, cid = k.split("|")
        diff = ", ".join(v.get("mismatch") or []) or "—"
        if v["state"] in ("no_diagnostics", "not_reconstructible") and v.get("detail"):
            diff = v["detail"]
        rows.append([f"`{uid}`", f"`{cid}`", CHECK_TEXT.get(v["state"], v["state"]), diff,
                     v.get("words_checked"), v.get("words_unmatched"),
                     v.get("places_estimate"), v.get("pipeline_asr_places")])
    L += _table(["единица", "конфигурация", "итог", "расхождения (или причина)", "слов сверено",
                 "слов не найдено (вставки)", "мест спора (оценка)", "мест спора у конвейера"], rows)
    return L


INVALID_TEXT = {"diarization_differs": "разное разделение говорящих в парах",
                "primary_differs": "основная модель дала в парах разные слова",
                "doubt_needs_own_second": "режим doubt сводится только со своей второй моделью",
                "doubt_not_reconstructible": "в режиме doubt голосование отключилось, нарезка непосчитанных кусков не сохранена",
                "no_diagnostics": "у пары нет сохранённой диагностики кусков"}
CHECK_TEXT = {"ok": "совпало", "mismatch": "НЕ совпало", "no_hypotheses": "нет гипотез",
              "not_reconstructible": "не восстановить",
              "old_rule": "запись конвейера сделана прежним правилом сведения",
              "no_diagnostics": "нет диагностики кусков — не пересчитано"}


def _partial_text(part, units):
    """«N из M (отрезков K, X с)» для отметок vote3_partial."""
    if not part or not part.get("units"):
        return f"0 из {units}"
    return f"{part['units']} из {units} (отрезков {part['spans']}, {_fmt(float(part['seconds']))} с)"


def _human_section(a):
    h = a["human"]
    L = ["## Кто был прав: решения человека", ""]
    extra = []
    if h.get("units_not_measured"):
        extra = ["Единицы без решений человека (звук из файла вне базы записей), не измерено: "
                 + ", ".join(f"`{u}`" for u in sorted(h["units_not_measured"])) + ".", ""]
    if h.get("not_measured"):
        return L + [f"Не измерено: {h['not_measured']}.", ""] + extra
    L += extra
    L += [f"**Предупреждение:** {h['warning']}.", "",
          "Решение человека — слова итоговой стенограммы старого варианта записи с отметкой автора "
          "«человек» (подтверждено как есть или исправлено), удаление слов человеком и ответ человека "
          "после прослушивания. Правки ИИ-агента и модели не учитываются. Слова каждой модели "
          "выравниваются с итоговой стенограммой целиком (`difflib`, как в конвейере), «угадала» — "
          "текст места совпал с решением после нормализации (`norm_word`, пары `SAME`).", ""]
    nr = h["none_right"]
    L.append(f"Решений сопоставлено: {nr['decisions']}; ни одна модель не угадала: {nr['none_right']}.")
    L.append("")
    rows = []
    for key, kinds in sorted(h["sources"].items()):
        for kind, c in sorted(kinds.items()):
            rows.append([src_label(key), KIND_TEXT.get(kind, kind), c["right"], c["wrong"]])
    L += _table(["модель", "вид решения", "угадала", "не угадала"], rows)
    rows = []
    for key, c in sorted(h["shared_error"].items()):
        rows.append([src_label(key), c["primary_wrong"], c["same_as_primary"], c["right"]])
    if rows:
        L += ["Где основная модель не угадала: повторила ли другая модель её текст (похожая ошибка) или "
              "дала решение человека.", ""]
        L += _table(["модель", "основная не угадала", "тот же текст, что основная", "угадала"], rows)
    rows = []
    for cid, c in sorted(h["configs"].items()):
        rows.append([f"`{cid}`", c["decisions"], c["shown"], c["shown_by_asr"], c["spared_right"],
                     c["silent_error"], c["output_right"]])
    L += ["Что было бы с решёнными местами при каждой конфигурации: «показано» — место попало бы "
          "человеку (любая пометка); «снято верно» — пометки нет и итог совпал с решением (человеку "
          "не пришлось бы смотреть); «ошибка без пометки» — пометки нет, а итог неверен.", ""]
    L += _table(["конфигурация", "решений", "показано", "из них спором моделей", "снято верно",
                 "ошибка без пометки", "итог совпал"], rows)
    return L


KIND_TEXT = {"changed": "исправлено", "confirmed": "подтверждено как есть", "deleted": "удалено",
             "listened": "после прослушивания"}


def _fitness_section(a):
    L = ["## Пригодность: признаки технического сбоя", "",
         "Пара помечается, если у модели есть ошибочные куски, пустой кусок там, где у соседней модели "
         f"не меньше {LOST_MIN_WORDS} слов (у третьей — {LOST_MIN_WORDS_THIRD} слов основной), слов меньше "
         f"половины соседа (при {FEW_WORDS_MIN}+ словах), откаты, подмена устройства или движка, ошибка "
         "второго прохода. Итоги выше даны с такими парами и без них. Одинаковость одной модели в "
         "разных конфигурациях — в `analysis.json`, поле `sources_consistency`.", ""]
    rows = [[f"`{k.split('|')[0]}`", f"`{k.split('|')[1]}`", ", ".join(v["reasons"])]
            for k, v in sorted(a["fitness"].items()) if not v["fit"]]
    if not rows:
        return L + ["Признаков сбоя не найдено.", ""]
    return L + _table(["единица", "конфигурация", "признаки"], rows)


def render_examples(examples):
    L = ["# Примеры решений человека (со словами)", "",
         "Только для этого компьютера: файл содержит текст записей. Не выгружать.", ""]
    if not examples:
        return "\n".join(L + ["Примеров нет.", ""])
    for ex in examples[:500]:
        L.append(f"## `{ex['unit_id']}` {ex['t']} с — {KIND_TEXT.get(ex['kind'], ex['kind'])}")
        L.append(f"- человек: «{ex['human']}»")
        for k, t in ex["sources"].items():
            L.append(f"- {src_label(k)}: «{t}»")
        for cid, c in ex["configs"].items():
            L.append(f"- `{cid}`: «{c['text']}» — {c['verdict']}")
        L.append("")
    return "\n".join(L)
