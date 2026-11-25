"""Анализ обычной работы (шаг 3а лаборатории): сохранённые гипотезы моделей и ответы человека на карточки.

Читает только песочницу prepare --work: копию базы записей source/state.sqlite3 (штатно,
storage.open_store_readonly) и копию файлов гипотез source/hypotheses/ (hypotheses.read). Модели не
запускаются, настоящая папка данных не открывается. Подробности — lab/README.md, раздел «Анализ», 6.

Связь записи и гипотез — по recording_id (ids.recording_id файла), проверка — хэшем файла audio_sha256,
группировка — хэшем звука (sound_id = pcm_sha256[:12]); слово unit_id в новых полях не используется:
file_unit — audio_sha256[:16] (смысл unit_id лаборатории), sound_id — смысл unit_id папки hypstore.
Состояния ответов — только из review_pack.answer_states_of; своих правил учёта ответов здесь нет.
"""
import json
import statistics
from pathlib import Path

import lab_sandbox

GAP_TOL = 0.1                 # с: допуск начала слова гипотезы относительно окна места
MIN_PLACES, MIN_RECORDINGS = 30, 3      # ниже — пометка «мало данных»; доли показываются всегда (DECISIONS 281547f)
FEW = "мало данных: меньше 30 мест или 3 записей"

# исход ответа (review_pack.OUTCOME_REASONS) -> строка учёта. Годится в основной счёт только «eligible»;
# новый исход без строки роняет тест tests/test_lab_work.py, а не попадает в счёт молча
OUTCOME_ROW = {
    "applied": "eligible", "applied_open": "eligible",
    "saved": "saved_not_applied", "pending_new": "saved_not_applied", "kept_previous": "saved_not_applied",
    "revert_pending": "reverted", "reverted": "reverted", "undo_conflict": "reverted",
    "skipped": "skipped", "negative": "skipped",
    "left_for_human": "left_for_human", "already_decided": "left_for_human",
    "apply_conflict": "not_applied", "blocked_preflight": "not_applied", "source_conflict": "not_applied",
    "not_found": "not_applied", "not_applied": "not_applied", "pause_refused": "not_applied", "hint": "not_applied",
    "listen_mismatch": "not_applied",
    # К-23: применён, но на место был другой ответ и человек его ещё не проверил — в основной счёт не идёт
    "applied_check": "applied_unchecked",
}
ROWS = ("eligible", "applied_unchecked", "unanswered", "previous_revision", "saved_not_applied", "reverted",
        "skipped", "left_for_human", "not_applied", "speaker_only", "not_human", "no_decision_in_transcript",
        "overlapped", "conflicting_between_variants", "recording_excluded")
CELLS = ("agree_corrected", "agree_confirmed", "disagree_corrected", "disagree_confirmed", "agreement_unknown")
REC_STATES = ("ok", "no_hypotheses", "several_files", "no_file_hash", "other_file", "mismatch")
ROLES = ("primary", "second", "third")


def _repo(name):
    return lab_sandbox._import_repo_module(name)


def _loads(text):
    try:
        value = json.loads(text) if text else None
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _words_of(transcript):
    return sorted((w for seg in transcript.get("segments") or [] for w in seg.get("words") or []
                   if isinstance(w, dict)), key=lambda w: float(w.get("s", 0) or 0))


def _norm(words, W):
    return [n for n in (W.norm_word(str(x)) for x in words) if n]


# ── чтение копии ─────────────────────────────────────────────────────────────────────────────────

def read_recordings(db_path):
    """Записи копии базы: [{recording_id, first, current, review}] — ревизия 1 стенограммы, текущая
    стенограмма и текущий снимок разбора."""
    store = _repo("storage").open_store_readonly(db_path)
    try:
        rows = store.db.execute(
            "SELECT r.recording_id AS rid, t1.payload AS first, t.payload AS cur, v.payload AS rev "
            "FROM recordings r "
            "JOIN transcript_revisions t1 ON t1.recording_id=r.recording_id AND t1.revision=1 "
            "JOIN transcript_revisions t ON t.recording_id=r.recording_id AND t.revision=r.source_revision "
            "LEFT JOIN review_revisions v ON v.recording_id=r.recording_id AND v.revision=r.review_revision "
            "ORDER BY r.recording_id").fetchall()
        return [{"recording_id": row["rid"], "first": _loads(row["first"]), "current": _loads(row["cur"]),
                 "review": _loads(row["rev"])} for row in rows]
    finally:
        store.close()


def read_hypotheses(hyp_dir, manifest):
    """Файлы гипотез, взятые при копировании (опись prepare --work). -> {recording_id: [запись]}, без номера."""
    H = _repo("hypotheses")
    by_rec, orphan = {}, 0
    for f in manifest.get("files") or []:
        if f.get("result") != lab_sandbox.HYP_COPIED:
            continue
        p = Path(hyp_dir) / f["relative"]
        try:
            rec = H.read(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            rec = {}
        rid = (rec.get("ids") or {}).get("recording_id") if rec else None
        if isinstance(rid, str) and rid:
            by_rec.setdefault(rid, []).append(rec)
        else:
            orphan += 1
    return by_rec, orphan


# ── связь и сверка ───────────────────────────────────────────────────────────────────────────────

def _role(rec, name):
    return next((r for r in rec.get("roles") or [] if isinstance(r, dict) and r.get("role") == name), None)


def _role_words(rec, name):
    r = _role(rec, name)
    words = r.get("words") if r else None
    return words if isinstance(words, list) else None


def reconcile(rec, first, W):
    """Пересведение гипотез штатным words.combine_hypotheses и сверка с ревизией 1 записи: функция сведения,
    vote3_skipped, vote3_partial, числа голосования и пометка asr у слов. -> список расхождений."""
    import lab_analyze
    prim = _role_words(rec, "primary")
    if prim is None:
        return ["no_primary"]
    alt = _role_words(rec, "second")
    combine = rec.get("combine") if isinstance(rec.get("combine"), dict) else {}
    third = combine.get("third_words") if isinstance(combine.get("third_words"), list) else None
    q = first.get("quality") or {}
    diag = q.get("asr_diagnostics") if isinstance(q.get("asr_diagnostics"), dict) else {}
    new = lab_analyze.combine(W, prim, alt, third, diag)
    recorded_fn = "vote3" if q.get("vote") is not None else ("mark_alt" if alt is not None else None)
    mism = []
    if new["function"] != recorded_fn:
        mism.append("function")
    if (q.get("vote3_skipped") or None) != new["skipped"]:
        mism.append("vote3_skipped")
    if (q.get("vote3_partial") or None) != new["partial"]:
        mism.append("vote3_partial")
    if new["function"] == "vote3":
        stats = {k: v for k, v in (q.get("vote") or {}).items() if isinstance(v, (int, float))}
        if stats != new["stats"]:
            mism.append("vote_stats")
    checked, _unmatched, flag_mism = lab_analyze.flag_check(new["main"], first, W)
    if flag_mism or not checked:
        mism.append("word_flags")
    return mism


def link(rec_row, hyps, W):
    """Запись -> (состояние связи, файл гипотез или None). Связь по recording_id, проверка хэшем файла."""
    files = hyps.get(rec_row["recording_id"]) or []
    if not files:
        return "no_hypotheses", None
    if len(files) > 1:
        return "several_files", None
    rec = files[0]
    sha = rec_row["first"].get("audio_sha256")
    if not sha:
        return "no_file_hash", rec
    if (rec.get("audio") or {}).get("audio_sha256") != sha:
        return "other_file", rec
    return ("mismatch" if reconcile(rec, rec_row["first"], W) else "ok"), rec


# ── окно места и тексты ролей ────────────────────────────────────────────────────────────────────

def place_window(item, first_words):
    """Окно места по времени слов ревизии 1 (правки могли сдвинуть время, ревизия 1 — нет).
    -> (начало, конец, способ: words | time | pause)."""
    ids = [x for x in item.get("word_ids") or [] if x]
    by_id = {w.get("word_id"): w for w in first_words}
    found = [by_id[x] for x in ids if x in by_id]
    if ids and len(found) == len(ids):
        return (min(float(w["s"]) for w in found), max(float(w.get("e", w["s"])) for w in found), "words")
    gap = item.get("gap") if isinstance(item.get("gap"), (list, tuple)) else None
    t1, t2 = (gap[0], gap[1]) if gap and len(gap) == 2 else (item.get("t1"), item.get("t2"))
    try:
        return float(t1), float(t2), ("time" if ids else "pause")
    except (TypeError, ValueError):
        return None, None, "pause"


def _in(w, a, b):
    s = float(w.get("s", -1))
    return a - GAP_TOL <= s < b if b > a else abs(s - a) <= GAP_TOL


def _third_heard(rec, a, b):
    """Слышала ли третья модель всё окно: куски, чей own_span пересекает окно, с heard не false."""
    r = _role(rec, "third")
    if not r or not isinstance(r.get("chunks"), list) or not r["chunks"]:
        return False
    hit = [c for c in r["chunks"] if isinstance(c, dict) and isinstance(c.get("own_span"), list)
           and len(c["own_span"]) == 2 and float(c["own_span"][0]) < max(b, a + GAP_TOL)
           and float(c["own_span"][1]) > a]
    return bool(hit) and all(c.get("heard", True) is not False for c in hit)


def _trace(rec):
    t = rec.get("vote_trace")
    return t if isinstance(t, dict) and isinstance(t.get("places"), list) else None


def models_agree(rec, a, b):
    """Согласны ли модели на окне: голосование шло всеми тремя ролями, третья слышала окно, исключённых
    отрезков в окне нет и мест не единогласия (vote_trace.places) в окне нет. None — неизвестно."""
    trace = _trace(rec)
    if trace is None or trace.get("disabled") or set(trace.get("roles") or []) < {"primary", "second", "whisper"}:
        return None
    if not _third_heard(rec, a, b):
        return None
    for s, e in trace.get("merged_skip_spans") or []:
        if float(s) < b + GAP_TOL and float(e) > a - GAP_TOL:
            return None
    lo, hi = a - GAP_TOL, (b if b > a else a + GAP_TOL)
    return not any(isinstance(p, dict) and p.get("t") is not None and lo <= float(p["t"]) < hi
                   for p in trace["places"])


def role_texts(rec, a, b):
    """Текст каждой роли на окне: основная и вторая — по времени слов; третья — по протоколу голосования
    (времён слов у неё нет) и только если слышала окно. -> {роль: [слова] | None}."""
    out = {}
    for name in ("primary", "second"):
        ws = _role_words(rec, name)
        out[name] = None if ws is None else [str(w.get("w", "")) for w in ws if isinstance(w, dict) and _in(w, a, b)]
    trace = _trace(rec)
    if out["primary"] is None or trace is None or not _third_heard(rec, a, b) \
            or "whisper" not in (trace.get("roles") or []):
        out["third"] = None
        return out
    prim = [w for w in _role_words(rec, "primary") if isinstance(w, dict) and _in(w, a, b)]
    places = {round(float(p["t"]), 3): p for p in trace["places"]
              if isinstance(p, dict) and p.get("t") is not None and p.get("decision") != "inserted"}
    inserted = [p for p in trace["places"] if isinstance(p, dict) and p.get("decision") == "inserted"
                and p.get("t") is not None and a - GAP_TOL <= float(p["t"]) < b]
    third, i = [], 0
    while i < len(prim):
        w = prim[i]
        for p in inserted:
            if float(p["t"]) <= float(w["s"]) and not p.get("_used"):
                third += str((p.get("variants") or {}).get("whisper", "")).split()
                p["_used"] = True
        p = places.get(round(float(w["s"]), 3))
        if p is None or (p.get("decision") == "kept" and "whisper" in (p.get("support") or [])):
            third.append(str(w.get("w", "")))
            i += 1
            continue
        third += str((p.get("variants") or {}).get("whisper", "")).split()
        i += max(1, len(str(p.get("main") or "").split()))
    for p in inserted:
        if not p.pop("_used", False):
            third += str((p.get("variants") or {}).get("whisper", "")).split()
    out["third"] = third
    return out


def min_confidence(rec, a, b):
    """Наименьшая уверенность c основной и второй моделей на окне (М2). None — слов нет."""
    cs = []
    for name in ("primary", "second"):
        for w in _role_words(rec, name) or []:
            if isinstance(w, dict) and _in(w, a, b) and isinstance(w.get("c"), (int, float)):
                cs.append(float(w["c"]))
    return min(cs) if cs else None


# ── места и ответы ───────────────────────────────────────────────────────────────────────────────

def _pack_state(review):
    """Ответы и события пакета из снимка разбора — тем же порядком, что recordings.read_review_pack_state."""
    pack = (review.get("undo") or {}).get("review_pack") or {}
    store = {"answers": review.get("answers") or {}, "rejected": review.get("rejected") or [],
             "applied_answers": pack.get("applied_answers") or {}, "applied_at": pack.get("applied_at"),
             "undo_snapshots": pack.get("undo_snapshots") or {},
             "application_events": pack.get("application_events") or []}
    if "applied_answers" not in pack:
        store.pop("applied_answers")
    return store, pack.get("manifest") or {}


def _text_decision(answer, item):
    """Касается ли ответ текста (а не только говорящего)."""
    if item.get("answer_mode") == "speaker":
        return False
    kind = answer.get("kind")
    if kind in ("option", "recommended"):
        opts = item.get("options") or []
        n = answer.get("option")
        dec = (opts[n].get("decision") or {}) if isinstance(n, int) and 0 <= n < len(opts) else {}
        return bool(set(dec) - {"speaker"})
    return kind == "free"


def classify_answer(aid, item, answer, states, RP):
    """Строка учёта места карточки. -> (строка, состояние ответа)."""
    if answer is None or not RP.answer_is_current(answer, item):
        old = [h for h in item.get("history") or [] if isinstance(h, dict) and h.get("answer")]
        return ("previous_revision" if old or answer is not None else "unanswered"), None
    if answer.get("kind") == "revert":
        return "reverted", None
    info = states.get(aid) or {}
    state = info.get("state")
    row = OUTCOME_ROW.get(state, "not_applied")
    if row != "eligible":
        return row, state
    if answer.get("kind") == "human":
        return "left_for_human", state
    if answer.get("actor_kind") != "human":
        return "not_human", state
    if not _text_decision(answer, item):
        return "speaker_only", state
    return "eligible", state


def recording_places(rec_row, hyp, W):
    """Места одной связанной записи: [{source: card|edit, row, reasons, window, agree, cell, ...}]."""
    import lab_analyze
    RP = _repo("review_pack")
    first_words = _words_of(rec_row["first"])
    store, man = _pack_state(rec_row["review"])
    states = RP.answer_states_of(store, man)
    items = {str(it.get("id")): it for it in man.get("items") or [] if isinstance(it, dict)}
    decisions = lab_analyze.recording_decisions(rec_row["current"], rec_row["review"], W)["decisions"]
    by_card = {}
    for d in decisions:
        by_card.setdefault(str(d["card"]), []).append(d)
    out = []

    def judge(row, a, b, final, kind):
        program = _norm([w.get("w") for w in first_words if _in(w, a, b)], W)
        corrected = kind == "deleted" or (kind == "changed" and not lab_analyze.equal_norm(program, final, W))
        agree = models_agree(hyp, a, b)
        texts = role_texts(hyp, a, b)
        human = final if kind != "deleted" else []
        matched = [r for r in ROLES if texts.get(r) is not None
                   and lab_analyze.equal_norm(_norm(texts[r], W), human, W)]
        row.update(window=[round(a, 2), round(b, 2)], corrected=corrected, agree=agree,
                   cell=("agreement_unknown" if agree is None else
                         f"{'agree' if agree else 'disagree'}_{'corrected' if corrected else 'confirmed'}"),
                   conf_min=min_confidence(hyp, a, b), matched=matched, texts=texts, program=program,
                   human=human)
        return row

    for aid, item in items.items():
        row = {"source": "card", "reasons": list(item.get("reasons") or []),
               "rev": RP.card_rev(item), "card_source_revision": RP.card_source_revision(item, man)}
        kind, state = classify_answer(aid, item, (store["answers"] or {}).get(aid), states, RP)
        row.update(row=kind, state=state)
        a, b, how = place_window(item, first_words)
        row["window_by"] = how
        mine = by_card.get(aid) or []
        if kind == "eligible":
            if a is None or not mine:
                row["row"] = "no_decision_in_transcript"
            elif any(d.get("card") != aid and d["a"] < b - GAP_TOL and a + GAP_TOL < d["b"] for d in decisions):
                row["row"] = "overlapped"
            else:
                d = mine[-1]
                judge(row, a, b, d["final"], d["kind"])
        out.append(row)
    for d in decisions:            # правки человека вне карточек — отдельная строка (ответ 3а-2)
        if str(d["card"]) in items:
            continue
        row = {"source": "edit", "row": "eligible", "reasons": [], "window_by": "edit"}
        out.append(judge(row, d["a"], max(d["b"], d["a"]), d["final"], d["kind"]))
    return out


# ── сводка ───────────────────────────────────────────────────────────────────────────────────────

def _table(rows):
    t = {c: 0 for c in CELLS}
    for r in rows:
        t[r["cell"]] += 1
    return t


def _shares(table, places, recordings):
    """Доли всегда, рядом число мест и записей; ниже 30 мест / 3 записей — пометка FEW (процент не скрывается)."""
    def share(x, y):
        return round(x / (x + y), 3) if x + y else None
    out = {"corrected_when_agree": share(table["agree_corrected"], table["agree_confirmed"]),
           "corrected_when_disagree": share(table["disagree_corrected"], table["disagree_confirmed"]),
           "places": places, "recordings": recordings}
    if places < MIN_PLACES or recordings < MIN_RECORDINGS:
        out["few_data"] = FEW
    return out


def _spread(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "median": round(statistics.median(vals), 3), "min": round(vals[0], 3),
            "max": round(vals[-1], 3)}


def analyze_work(sandbox, W):
    """Раздел «Обычная работа». -> (раздел для analysis.json без текстов, примеры со словами для examples.md)."""
    s = Path(sandbox)
    man_path = s / "source" / lab_sandbox.HYP_MANIFEST
    db_path = s / "source" / "state.sqlite3"
    if not man_path.is_file():
        return {"not_measured": "песочница не prepare --work: нет описи гипотез"}, []
    if not db_path.is_file():
        return {"not_measured": "нет копии базы записей source/state.sqlite3"}, []
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    hyps, orphan = read_hypotheses(s / "source" / lab_sandbox.HYP_DIR, manifest)
    recs = read_recordings(db_path)
    rec_states = {k: 0 for k in REC_STATES}
    linked, sounds, all_rows, hours = [], {}, [], 0.0
    known = {r["recording_id"] for r in recs}
    for rec_row in recs:
        state, hyp = link(rec_row, hyps, W)
        rec_states[state] += 1
        audio = (hyp or {}).get("audio") or {}
        pcm = audio.get("pcm_sha256") if isinstance(audio.get("pcm_sha256"), str) else None
        sha = rec_row["first"].get("audio_sha256")
        entry = {"file_unit": sha[:16] if isinstance(sha, str) else None,
                 "sound_id": pcm[:12] if pcm else None, "state": state}
        if state != "ok":
            linked.append(entry)
            continue
        dur = audio.get("duration_sec")
        hours += float(dur) / 3600 if isinstance(dur, (int, float)) else 0.0
        rows = recording_places(rec_row, hyp, W)
        for r in rows:
            r["recording"] = rec_row["recording_id"]
            r["sound_id"] = entry["sound_id"]
        sounds.setdefault(entry["sound_id"], []).append(rows)
        all_rows += rows
        linked.append(dict(entry, places=sum(1 for r in rows if r["source"] == "card")))
    # тот же звук в разных записях: одно место, решённое по-разному, — спорное, в счёт не идёт
    for groups in sounds.values():
        if len(groups) < 2:
            continue
        for rows in groups:
            for r in rows:
                if r["row"] != "eligible" or "cell" not in r:
                    continue
                for other in groups:
                    if other is rows:
                        continue
                    for o in other:
                        if o.get("cell") and o["source"] == r["source"] and \
                                o["window"][0] < r["window"][1] and r["window"][0] < o["window"][1] and \
                                o["corrected"] != r["corrected"]:
                            r["row"] = o["row"] = "conflicting_between_variants"
    counted = {"card": [r for r in all_rows if r["source"] == "card" and r["row"] == "eligible" and "cell" in r],
               "edit": [r for r in all_rows if r["source"] == "edit" and r["row"] == "eligible" and "cell" in r]}
    rows_count = {k: 0 for k in ROWS}
    for r in all_rows:
        if r["source"] == "card":
            rows_count[r["row"]] += 1
    m1, shares, m2 = {}, {}, {}
    for src, rows in counted.items():
        table = _table(rows)
        recs_n = len({r["recording"] for r in rows})
        m1[src] = table
        shares[src] = _shares(table, len(rows), recs_n)
        m2[src] = {c: _spread(r["conf_min"] for r in rows if r["cell"] == c) for c in CELLS}
    by_reason = {}
    for r in counted["card"]:
        for reason in r["reasons"] or ["none"]:
            by_reason.setdefault(reason, []).append(r)
    every = counted["card"] + counted["edit"]
    matched = {k: 0 for k in (*ROLES, "none")}
    for r in every:
        for role in r["matched"]:
            matched[role] += 1
        if not r["matched"]:
            matched["none"] += 1
    found = [r for r in every if r["cell"] == "agree_corrected"]
    section = {
        "definitions": "lab/README.md, раздел «Анализ», 6",
        "recordings": {"total": len(recs), "states": rec_states, "hours_linked": round(hours, 3),
                       "hypotheses_without_record": orphan + sum(1 for rid in hyps if rid not in known),
                       "units": linked},
        "places": {"cards": sum(rows_count.values()), "rows": rows_count,
                   "window_by": {k: sum(1 for r in counted["card"] if r["window_by"] == k)
                                 for k in ("words", "time", "pause")},
                   "edits_without_card": len(counted["edit"])},
        "m1": {"cards": m1["card"], "edits": m1["edit"], "by_reason": {k: _table(v) for k, v in sorted(by_reason.items())},
               "shares": shares},
        "m2": m2,
        "roles_matched_human": matched,
        "recordings_with_agree_corrected": len({r["recording"] for r in found}),
        "warning": ("ответы человека есть только на местах карточек и правок: это не точность слов; "
                    "числа абсолютные, доли — от 30 мест и 3 записей"),
    }
    examples = [{"source": r["source"], "cell": r["cell"], "window": r["window"], "reasons": r["reasons"],
                 "texts": r["texts"], "program": r["program"], "human": r["human"], "conf_min": r["conf_min"]}
                for r in every if r["cell"] in ("agree_corrected", "disagree_corrected")]
    return section, examples


def render_section(work):
    """Раздел 6 отчёта results/report.md — без текстов слов."""
    lines = ["## 6. Обычная работа: гипотезы и ответы человека", ""]
    if work.get("not_measured"):
        return lines + [f"Не измерено: {work['not_measured']}.", ""]
    rec, places = work["recordings"], work["places"]
    lines += [f"Записей в копии базы: {rec['total']}; связаны с гипотезами и сверены: {rec['states']['ok']} "
              f"(около {rec['hours_linked']} ч звука). Не связаны: "
              + ", ".join(f"{k} — {v}" for k, v in rec["states"].items() if k != "ok" and v) + ".", "",
              "Учёт мест карточек:", ""]
    lines += [f"- {k}: {v}" for k, v in places["rows"].items() if v]
    lines += [f"- правки человека вне карточек (отдельная строка): {places['edits_without_card']}", "",
              "М1 — согласие моделей против решения человека (абсолютные числа):", "",
              "| Источник | " + " | ".join(CELLS) + " |", "|---" * (len(CELLS) + 1) + "|"]
    for src in ("cards", "edits"):
        lines.append(f"| {src} | " + " | ".join(str(work["m1"][src][c]) for c in CELLS) + " |")
    lines += [""]
    for src, sh in work["m1"]["shares"].items():
        pct = ", ".join(f"{k} = {'—' if sh[k] is None else format(sh[k], '.1%')}"
                        for k in ("corrected_when_agree", "corrected_when_disagree"))
        lines.append(f"Доли ({src}): {pct}; мест {sh['places']}, записей {sh['recordings']}"
                     + (f" — {sh['few_data']}" if sh.get("few_data") else ""))
    lines += ["", "М2 — наименьшая уверенность основной и второй моделей на месте (по клеткам М1):", ""]
    for src, cells in work["m2"].items():
        for c, v in cells.items():
            if v.get("n"):
                lines.append(f"- {src}, {c}: мест {v['n']}, медиана {v['median']}, от {v['min']} до {v['max']}")
    lines += ["", "Чья гипотеза совпала с решением человека: "
              + ", ".join(f"{k} — {v}" for k, v in work["roles_matched_human"].items()) + ".",
              f"Записей хотя бы с одним «модели согласны, но человек исправил»: "
              f"{work['recordings_with_agree_corrected']}.", "", work["warning"] + ".", ""]
    return lines


def render_examples(examples):
    """Примеры мест «исправил» со словами всех ролей — только этот компьютер (results/examples.md)."""
    lines = ["## Обычная работа: места, которые исправил человек", ""]
    if not examples:
        return lines + ["Примеров нет.", ""]
    for e in examples:
        t = e["texts"]
        lines += [f"- {e['source']}, {e['cell']}, {e['window'][0]}–{e['window'][1]} с, причины "
                  f"{', '.join(e['reasons']) or '—'}, уверенность {e['conf_min']}",
                  f"  - основная: {' '.join(t.get('primary') or []) if t.get('primary') is not None else '—'}",
                  f"  - вторая: {' '.join(t.get('second') or []) if t.get('second') is not None else '—'}",
                  f"  - третья: {' '.join(t.get('third') or []) if t.get('third') is not None else 'не слушала'}",
                  f"  - программа: {' '.join(e['program'])}", f"  - человек: {' '.join(e['human'])}"]
    return lines + [""]
