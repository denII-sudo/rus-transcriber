"""Голоса: отпечатки фрагментов, профили участников, проверка и исправление подписей.

Отпечаток — та же модель WeSpeaker ResNet34, что внутри pyannote (pipe._embedding), поэтому
профили, собранные из прошлых записей, сравнимы с кластерами новой записи.

Профили лежат в DATA_DIR\\groups\\<group>\\voices.json (paths.voices_path) и с этого компьютера не уходят:
это биометрия участников. Словарь группы (vocab.py) передаётся отдельно, без них.
Профиль пополняется сам: перед расшифровкой берутся расшифровки из папки записи, в которых имена
подтверждены (переименованы через rename_speakers), и ещё не учтённые.
"""
import json
import bisect
import copy
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
CAP = 10.0            # от длинных фрагментов берём середину не длиннее
MIN_EMB = 0.6         # короче — отпечаток слишком шумный
LEARN_MIN = 2.0       # для профиля берём реплики от 2 с
MATCH_MIN = 0.50      # кластер записи считается голосом участника при сходстве не ниже
MATCH_MARGIN = 0.08   # и отрыве от второго по сходству участника
WIN, HOP = 1.5, 0.75  # окна для поиска чужого голоса внутри реплики
SCAN_MIN = 3.0        # проверяем внутри реплик не короче
SPLIT_MARGIN = 0.12   # окно чужое, если сходство с другим голосом выше своего на столько
SPLIT_MIN_RUN = 2     # минимум подряд чужих окон (≈2,25 с)
REASSIGN_MIN = 1.0    # пореплично по профилям сверяем реплики от 1 с
OWN_BONUS = 0.05      # поправка в пользу кластера, к которому реплику отнёс pyannote
PROFILE_VERSION = 3   # 3: отпечатки без дополнения тишиной


def _log(msg):
    print(f"[voices] {msg}", file=sys.stderr, flush=True)


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 and np.isfinite(n) else None


class Embedder:
    def __init__(self, pipe, batch=32):
        self.emb = getattr(pipe, "_embedding", None)
        if self.emb is None:
            raise RuntimeError("в pyannote нет модели отпечатков (_embedding)")
        self.batch = batch
        self.seconds = 0.0

    def spans(self, audio, spans, cap=CAP, min_len=MIN_EMB, progress=None):
        """spans: [(start, end)] в секундах -> список нормированных векторов или None.
        progress(done, total) — после каждого пакета (для статуса задачи)."""
        import torch
        t0 = time.time()
        items = []
        for k, (s, e) in enumerate(spans):
            if e - s < min_len:
                continue
            if e - s > cap:
                mid = (s + e) / 2
                s, e = mid - cap / 2, mid + cap / 2
            a, b = int(max(0, s) * SR), int(min(len(audio) / SR, e) * SR)
            if b - a >= int(min_len * SR):
                items.append((k, a, b))
        out = [None] * len(spans)
        # Пакетом считаем только фрагменты одинаковой длины: дополнение тишиной искажает отпечаток
        # (маски у этой модели его не убирают — проверено на записи 06.11). Разной длины — по одному.
        groups, done_n = {}, 0
        for k, a, b in items:
            groups.setdefault(b - a, []).append((k, a, b))
        for n, grp in groups.items():
            for i in range(0, len(grp), self.batch):
                part = grp[i:i + self.batch]
                w = np.stack([audio[a:b] for _, a, b in part])[:, None, :].astype(np.float32)
                vecs = np.asarray(self.emb(torch.from_numpy(w)))
                for j, (k, _, _) in enumerate(part):
                    out[k] = _norm(np.asarray(vecs[j], dtype=np.float32))
                done_n += len(part)
                if progress:
                    progress(done_n, len(items))
        self.seconds += time.time() - t0
        return out


def centroids(turns, embs, min_len=1.5):
    """{label: вектор} по репликам записи, вес — длительность."""
    acc = {}
    for t, v in zip(turns, embs):
        if v is None or t["e"] - t["s"] < min_len:
            continue
        w = min(t["e"] - t["s"], CAP)
        acc.setdefault(t["label"], []).append((w, v))
    out = {}
    for lbl, items in acc.items():
        c = _norm(sum(w * v for w, v in items))
        if c is None:
            continue
        # отбрасываем явные чужие куски и пересчитываем
        good = [(w, v) for w, v in items if float(v @ c) >= 0.25] or items
        out[lbl] = _norm(sum(w * v for w, v in good))
    return out


# ---------- профили ----------
def profile_path(group):
    import paths
    return paths.voices_path(group)


class VoiceProfileError(ValueError):
    """Существующий профиль несовместим или недоступен; перезаписывать его нельзя."""


def empty_profile():
    return {"version": PROFILE_VERSION, "model": "wespeaker-resnet34 (pyannote community-1)", "speakers": {}, "sources": {}}


def validate_profile(d, convert=True):
    """Проверить структуру и отпечатки; convert=False сохраняет списки для записи JSON."""
    try:
        if not isinstance(d, dict) or d.get("version") != PROFILE_VERSION:
            raise VoiceProfileError("неподдерживаемая версия голосового профиля; сохранение запрещено")
        if not isinstance(d.get("speakers"), dict) or not isinstance(d.get("sources"), dict):
            raise ValueError("неверная структура")
        dimension = None
        for speaker in d["speakers"].values():
            if not isinstance(speaker, dict) or not isinstance(speaker.get("parts"), dict):
                raise ValueError("неверные вклады")
            for part in speaker["parts"].values():
                vec = np.asarray(part["vec"], dtype=np.float32)
                weight = part["sec"]
                if (vec.ndim != 1 or not vec.size or not np.all(np.isfinite(vec))
                        or not np.any(vec)
                        or isinstance(weight, bool) or not isinstance(weight, (int, float))
                        or not math.isfinite(weight) or weight < 0):
                    raise ValueError("неверный отпечаток или вес")
                if dimension is not None and vec.size != dimension:
                    raise ValueError("различающиеся размеры отпечатков")
                dimension = vec.size
                if convert:
                    part["vec"] = vec
        return d
    except VoiceProfileError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as ex:
        raise VoiceProfileError("голосовой профиль повреждён; сохранение запрещено") from ex


def load_profile(path, convert=True):
    """Прочитать профиль; пустой создаётся только для отсутствующего файла."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as ex:
        if p.is_symlink():
            raise VoiceProfileError("ссылка на голосовой профиль недоступна; сохранение запрещено") from ex
        return empty_profile()
    except (OSError, UnicodeError) as ex:
        raise VoiceProfileError("голосовой профиль не прочитан; сохранение запрещено") from ex
    try:
        data = json.loads(text)
    except (ValueError, UnicodeError) as ex:
        raise VoiceProfileError("голосовой профиль повреждён; сохранение запрещено") from ex
    return validate_profile(data, convert=convert)


def save_profile(path, prof):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = dict(prof)
    d["schema_version"] = 1
    d["speakers"] = {n: {"parts": {k: {**{kk: vv for kk, vv in part.items() if kk != "vec"},
                                       "vec": [round(float(x), 6) for x in part["vec"]]}
                                   for k, part in s.get("parts", {}).items()}}
                     for n, s in prof["speakers"].items()}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def voice_vectors(prof, exclude_audio=None):
    """{имя: вектор} по всем учтённым записям, кроме exclude_audio (своя запись не подсказывает сама себе)."""
    ex = str(Path(exclude_audio).resolve()).lower() if exclude_audio else None
    out = {}
    for name, s in prof.get("speakers", {}).items():
        parts = [pt for src, pt in s.get("parts", {}).items() if src.lower() != ex]
        if parts:
            v = _norm(sum(pt["vec"] * pt["sec"] for pt in parts))
            if v is not None:
                out[name] = v
    return out


def source_fingerprint(transcript_path, audio_path):
    """Отпечаток происхождения источника: содержимое стенограммы и размер/mtime аудио.

    mtime стенограммы остаётся для совместимости со старыми записями, но одного mtime мало:
    содержимое может измениться при том же времени. Отпечаток не даёт права автоматически
    заменять существующий вклад — для этого по-прежнему нужен просмотр и подтверждение.
    """
    jp, src = Path(transcript_path), Path(audio_path)
    data = jp.read_bytes()
    st = src.stat()
    return {"transcript_sha256": hashlib.sha256(data).hexdigest(),
            "transcript_size": len(data),
            "audio_size": int(st.st_size),
            "audio_mtime": round(float(st.st_mtime), 3)}


def source_fingerprint_status(stored, current, transcript_mtime):
    """Сравнить запись sources с текущим состоянием: unchanged/changed/legacy_*/absent."""
    if stored is None:
        return "absent"
    if isinstance(stored, dict):
        return "unchanged" if all(stored.get(k) == current.get(k) for k in FINGERPRINT_KEYS) else "changed"
    if isinstance(stored, (int, float)) and not isinstance(stored, bool) and math.isfinite(stored):
        same = math.isclose(float(stored), float(transcript_mtime), rel_tol=0, abs_tol=1e-6)
        return "legacy_unchanged" if same else "legacy_changed"
    return "changed"


def audio_quality(audio):
    """Приблизительные измерения записи без изменения отбора и без моделей.

    Возвращает длительность, пик, RMS, долю клиппинга, долю тишины и грубую оценку SNR
    по кадрам 30 мс. Тихая речь не считается плохой: это только предупреждение.
    Нулевой или нечитаемый сигнал не прерывает обучение.
    """
    try:
        a = np.asarray(audio, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return {"available": False, "reason": "сигнал не прочитан", "warnings": []}
    if not a.size:
        return {"available": False, "reason": "пустой сигнал", "warnings": ["silent"]}
    if not np.all(np.isfinite(a)):
        return {"available": False, "reason": "сигнал содержит нечисловые значения", "warnings": []}
    duration = a.size / SR
    peak = float(np.max(np.abs(a)))
    rms = float(np.sqrt(np.mean(a * a)))
    clipping_ratio = float(np.mean(np.abs(a) >= CLIP_LEVEL))
    frame = max(1, int(round(0.03 * SR)))
    frames = a.size // frame
    snr = None
    silence_ratio = 1.0
    if frames:
        blocks = a[:frames * frame].reshape(frames, frame)
        levels = np.sqrt(np.mean(blocks * blocks, axis=1))
        silence_ratio = float(np.mean(levels <= 10 ** (-50 / 20)))
        noise = float(np.percentile(levels, 10))
        speech = float(np.percentile(levels, 90))
        if noise > 1e-7 and speech > 0:
            snr = round(20 * math.log10(speech / noise), 1)
    warnings = []
    if clipping_ratio >= 0.0005:
        warnings.append("clipping")
    if peak <= 1e-7:
        warnings.append("silent")
    elif peak < 10 ** (QUIET_PEAK_DBFS / 20):
        warnings.append("very_quiet")
    if snr is not None and snr < NOISY_SNR_DB:
        warnings.append("noisy_estimate")

    def dbfs(value):
        return round(20 * math.log10(max(value, 1e-8)), 1)

    return {"available": True, "duration_sec": round(duration, 2), "peak_dbfs": dbfs(peak),
            "rms_dbfs": dbfs(rms), "clipping_ratio": round(clipping_ratio, 6), "clipped": "clipping" in warnings,
            "silence_ratio": round(silence_ratio, 3), "snr_db_estimate": snr,
            "warnings": warnings, "approximate": True}


def explain_audio_quality(quality):
    """Короткие русские пояснения к кодам предупреждений качества."""
    return [AUDIO_QUALITY_NOTES.get(code, code) for code in (quality or {}).get("warnings") or []]


def validate_quality_policy(policy):
    if policy not in AUDIO_QUALITY_POLICIES:
        raise ValueError("политика качества: warn или reject_clipping")
    return policy


def trim_quality_map(prof, limit=200):
    """Ограничить историю измерений качества последними записями (векторы не затрагиваются)."""
    items = prof.get("audio_quality")
    if isinstance(items, dict) and len(items) > limit:
        for key in list(items)[:-limit]:
            items.pop(key, None)


def voice_weight_diagnostics(prof):
    """Доли численных весов частей без векторов; не измерение влияния или качества узнавания."""
    result = {}
    for name, speaker in (prof.get("speakers") or {}).items():
        parts = speaker.get("parts") or {}
        values = []
        invalid = False
        for source, part in sorted(parts.items()):
            weight = part.get("sec")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                invalid = True
                break
            values.append((source, float(weight)))
        total = sum(weight for _, weight in values)
        if invalid or not math.isfinite(total):
            result[name] = {"available": False, "reason": "некорректный вес записи", "records": len(parts)}
        elif not total:
            result[name] = {"available": False, "reason": "сумма весов равна нулю", "records": len(parts)}
        else:
            result[name] = {"available": True, "total_weight": round(total, 2), "records": len(parts),
                            "parts": [{"recording_index": i, "recording": Path(source).name,
                                       "weight": round(weight, 2), "weight_share_percent": round(100 * weight / total, 2)}
                                      for i, (source, weight) in enumerate(values, 1)]}
    return result


def _confirmed(data):
    """Имена в расшифровке подтверждены (rename_speakers), а не угаданы по голосу."""
    if data.get("names_confirmed"):
        return True
    return bool(data.get("speaker_names")) and not data.get("names_source")


HUMAN_WEIGHT = 3.0    # отрезок, подписанный человеком, весит втрое больше обычного
HUMAN_MIN = 0.8       # отрезки человека — от 0,8 с чистой речи
LEARN_MAX_GAP = 0.5   # экспериментальный предел паузы между словами в одном отрезке
LEARN_CAP = 8.0       # окно обучения не пересекает границы отобранных слов
SELECTION_VERSION = 2  # границы слов, разрыв пауз и исключение пересечений чужой речи
DRIFT_MIN = 0.8       # сходство нового профиля со старым ниже — профиль «уплыл», показать человеку
# защита от тёзок — разных людей с одним именем (другая группа, новый участник):
SAME_MIN = 0.70       # вклад записи похож на профиль имени меньше — в карантин, а не в профиль
SAME_MARGIN = 0.05    # или вклад ближе к другому участнику на столько
GATE_MIN_SEC = 20.0   # проверка — когда у имени уже есть столько секунд из других записей
GATE_PART_SEC = 4.0   # и вклад записи не короче (короткие отрезки слишком шумные для вывода)

# Качество записи: сейчас только измерения и предупреждения. Автоматическая отбраковка
# по клиппингу включается решением человека для группы; тихая речь и оценка SNR сами по себе
# ничего не отбраковывают. SNR без чистого эталона приблизителен.
AUDIO_QUALITY_POLICIES = ("warn", "reject_clipping")
CLIP_LEVEL = 0.999
CLIP_REJECT_RATIO = 0.005
QUIET_PEAK_DBFS = -30.0
NOISY_SNR_DB = 10.0
AUDIO_QUALITY_NOTES = {
    "clipping": "есть клиппинг (перегрузка)",
    "silent": "сигнал почти отсутствует",
    "very_quiet": "очень тихая запись",
    "noisy_estimate": "оценка шума высокая",
}
FINGERPRINT_KEYS = ("transcript_sha256", "transcript_size", "audio_size", "audio_mtime")


SELECTION_REASONS = {
    "unknown_speaker": "Реплика пропущена: участник не определён",
    "overlap": "Слово пропущено: отмечено наложение голосов",
    "foreign_overlap": "Окно пропущено: пересекается со временем другого говорящего",
    "unconfirmed_or_flagged": "Слово пропущено: автор не подтверждён или есть пометка сомнения",
    # коротко: строка входит в окно подтверждения (core.prepare_voice_action ограничивает его длину)
    "legacy_human_without_speaker": ("Пропущено: старая пометка «человек» без пословной отметки "
                                     "(перенести — legacy_speaker_review)"),
    "invalid_time": "Слово пропущено: некорректные границы времени",
    "word_too_long": "Слово пропущено: длительность больше максимального окна",
    "gap_or_order": "Участок разделён из-за паузы, порядка или предельной длины",
    "accepted_human": "Кандидат принят: говорящий подтверждён человеком",
    "accepted_auto": "Кандидат принят: слово без пометок в подтверждённой расшифровке",
    "too_short": "Окно пропущено: слишком короткое",
    "legacy_without_word_times": "Реплика пропущена: нет времён отдельных слов",
    "unreadable_transcript": "Расшифровка пропущена: файл не прочитан",
    "other_group": "Расшифровка пропущена: другая группа",
    "profile_disabled": "Расшифровка пропущена: профили для неё отключены",
    "unconfirmed_source": "Расшифровка пропущена: нет подтверждённых имён или действующих правок",
    "missing_audio": "Расшифровка пропущена: исходное аудио недоступно",
    "already_processed": "Расшифровка пропущена: эта версия уже учтена",
    "source_changed_requires_preview": "Источник изменился после прежнего учёта; нужен просмотр и подтверждение",
    "audio_quality_rejected": "Источник пропущен: политика группы отбраковала клиппинг",
    "source_needs_review": "Прежний источник не подтверждён текущими правилами отбора; требуется проверка",
    "existing_requires_preview": "Вклад сохранён без изменений: требуется отдельное подтверждение обновления",
    "no_usable_vectors": "Новый вклад не создан: нет подходящих фрагментов или пригодных отпечатков",
    "identity_different": "Кандидат пропущен: человек указал другого участника с тем же именем",
    "quarantined": "Новый вклад помещён в карантин; прежний, если был, сохранён",
    "replaced": "Вклад записи заменён после подтверждения",
    "added": "Добавлен вклад записи",
    "no_profile_change": "Изменений вклада нет",
    "processing_error": "Источник пропущен из-за ошибки; изменения этого источника отменены",
}


def explain_selection_report(report):
    """Читаемые описания счётчиков; единицы разных причин могут различаться."""
    return [{"code": key, "description": SELECTION_REASONS.get(key, "Неизвестная причина"), "count": count}
            for key, count in sorted(report.items())]


def profile_sufficiency(prof, expected_names):
    """Оценка полноты, не оценка точности: 90 с и три записи для каждого известного участника."""
    names = set(expected_names or ()) | set(prof.get("speakers", {}))
    result = {}
    for name in sorted(names):
        parts = prof.get("speakers", {}).get(name, {}).get("parts", {})
        verified_seconds = legacy_seconds = 0.0
        verified_records = legacy_records = 0
        unverified_records = 0
        legacy_unmeasured = 0
        for part in parts.values():
            verified = part.get("selection_version") == SELECTION_VERSION
            if not verified:
                unverified_records += 1
            # Старый взвешенный sec не равен длительности речи; без раздельных полей не засчитываем.
            a, h = part.get("auto_sec"), part.get("human_sec")
            if not isinstance(a, (int, float)) or not isinstance(h, (int, float)):
                if not verified:
                    legacy_unmeasured += 1
                continue
            duration = a + h
            if math.isfinite(duration) and duration >= 20 and a >= 0 and h >= 0:
                if verified:
                    verified_seconds += min(duration, 60.0)
                    verified_records += 1
                else:
                    legacy_seconds += min(duration, 60.0)
                    legacy_records += 1
            elif not verified and (not math.isfinite(duration) or a < 0 or h < 0):
                legacy_unmeasured += 1
        result[name] = {"seconds": round(verified_seconds + legacy_seconds, 1),
                        "recordings": verified_records + legacy_records,
                        "verified_seconds": round(verified_seconds, 1), "verified_recordings": verified_records,
                        "legacy_seconds": round(legacy_seconds, 1), "legacy_recordings": legacy_records,
                        "unverified_records": unverified_records, "legacy_unmeasured_records": legacy_unmeasured,
                        "sufficient": verified_seconds >= 90 and verified_records >= 3}
    return {"sufficient": bool(result) and all(v["sufficient"] for v in result.values()), "speakers": result}


def auto_learn_group(prof, group, expected_names, learn_fn, save_fn):
    """Обычное автопополнение; распознавание и ручное обновление вызываются отдельно."""
    import settings
    try:
        policy = settings.group_voice_settings(group, initialize=True)
    except (ValueError, OSError) as ex:
        return [], {"enabled": False, "reason": "settings_error", "error": str(ex)}
    if not policy["auto_learn"]:
        return [], {"enabled": False, "reason": policy.get("reason", "manual")}
    sufficiency = profile_sufficiency(prof, expected_names)
    learned = []
    if not sufficiency["sufficient"]:
        before_provenance = prof.get("provenance_revision", 0)
        learned = learn_fn()
        if learned or prof.get("provenance_revision", 0) != before_provenance:
            save_fn()
        sufficiency = profile_sufficiency(prof, expected_names)
    if sufficiency["sufficient"]:
        data = {"auto_learn": False, "reason": "sufficient"}
        if policy.get("audio_quality_policy_source") == "group":
            data["audio_quality_policy"] = policy["audio_quality_policy"]
        settings.save_group_voice_settings(group, data)
        return learned, {"enabled": False, "reason": "sufficient"}
    return learned, {"enabled": True, "reason": policy.get("reason", "default")}


def group_binding(data):
    """Диагностика источников группы; stale-ссылка видна, конфликт не маскируется."""
    import vocab
    return vocab.transcript_binding(data)


def group_of(data, _cache=None):
    """Группа расшифровки: встроенное поле или прямая ссылка quality.vocab."""
    binding = group_binding(data)
    if binding["conflict"]:
        raise ValueError("Группа расшифровки противоречит группе словаря quality.vocab")
    return binding["group"]


def _word_human(word):
    """Человек подтвердил именно говорящего; старое общее `by` этого не доказывает."""
    speaker = (word.get("provenance") or {}).get("speaker")
    return isinstance(speaker, dict) and speaker.get("by") == "human" and not speaker.get("reverted")


def _legacy_human_only(word):
    """Старое общее `by: human` без пословной отметки говорящего (до правил 13.11.2026)."""
    return word.get("by") == "human" and not isinstance((word.get("provenance") or {}).get("speaker"), dict)


def legacy_speaker_places(data):
    """Места, где по старой истории правок видно решение человека о говорящем.

    Только чтение. Основания: (1) правка человека, сменившая говорящего (before.speaker !=
    after.speaker или pairs); (2) место разбора, принятое человеком как есть (слова с
    `by: human` без ссылки на правку внутри [t1, t2]); (3) правка человеком только текста —
    говорящего она не доказывает, поэтому место тоже предлагается подтвердить отдельно.
    Слова, уже имеющие пословную отметку говорящего, пропускаются. Каждое место — непрерывный
    участок одного сегмента; места короче HUMAN_MIN в отпечаток не попадут и не предлагаются.
    """
    review = {item.get("id"): item for item in data.get("review") or [] if item.get("id")}
    edits = {}
    for edit in data.get("edits") or []:
        if edit.get("by") == "human" and edit.get("id") in review and not edit.get("reverted"):
            edits[edit["id"]] = edit
    places, skipped = [], {"too_short": 0, "already_confirmed": 0, "not_contiguous": 0, "no_words": 0,
                           "merged": 0}
    for rid, item in review.items():
        edit = edits.get(rid)
        if edit is not None:
            before, after = edit.get("before") or {}, edit.get("after") or {}
            changed = isinstance(before, dict) and isinstance(after, dict) and before.get("speaker") != after.get("speaker")
            evidence = ("смена говорящего правкой человека" if changed or edit.get("pairs") else
                        "правка текста человеком (говорящий не менялся)")
            def wanted(w, rid=rid):
                return w.get("edited") == rid or (w.get("by") == "human" and not w.get("edited"))
        elif item.get("status") == "resolved":
            evidence = "место принято человеком как есть"
            def wanted(w):
                return w.get("by") == "human" and not w.get("edited")
        else:
            continue
        t1, t2 = item.get("t1", item.get("t")), item.get("t2", item.get("t"))
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (t1, t2)):
            skipped["no_words"] += 1
            continue
        import transcript
        found = []
        pause = transcript.is_gap_place(item)       # слова пропуска — начатые в паузе, соседнее слово не его
        for si, seg in enumerate(data.get("segments") or []):
            for wi, w in enumerate(seg.get("words") or []):
                if not isinstance(w.get("s"), (int, float)):
                    continue
                inside = (transcript.in_pause(w["s"], t1, t2) if pause
                          else t1 - transcript.WORD_PLACE_TOLERANCE <= w["s"] <= t2 + transcript.WORD_PLACE_TOLERANCE)
                if inside and wanted(w):
                    found.append((si, wi))
        if not found:
            skipped["no_words"] += 1
            continue
        segs = {si for si, _ in found}
        indexes = [wi for _, wi in found]
        if len(segs) != 1 or indexes != list(range(indexes[0], indexes[-1] + 1)):
            skipped["not_contiguous"] += 1
            continue
        si = segs.pop()
        seg = data["segments"][si]
        words = seg["words"][indexes[0]:indexes[-1] + 1]
        if any(not _legacy_human_only(w) for w in words):
            skipped["already_confirmed"] += 1
            continue
        if max(w["e"] for w in words) - words[0]["s"] < HUMAN_MIN:
            skipped["too_short"] += 1
            continue
        places.append({"id": rid, "seg": si, "w1": indexes[0], "w2": indexes[-1] + 1,
                       "t1": round(words[0]["s"], 2), "t2": round(max(w["e"] for w in words), 2),
                       "speaker": seg.get("speaker"), "words": len(words),
                       "seconds": round(sum(max(0.0, w["e"] - w["s"]) for w in words), 2),
                       "evidence": evidence, "review_status": item.get("status")})
    places = _merge_overlapping_places(data, places, skipped)
    return {"places": places, "skipped": skipped,
            "seconds_by_speaker": {name: round(sum(p["seconds"] for p in places if p["speaker"] == name), 2)
                                   for name in sorted({p["speaker"] for p in places})}}


def _merge_overlapping_places(data, places, skipped):
    """Места одного сегмента с общими словами — одна карточка.

    Иначе ответы на вложенные места (r19: слова 0–2, r20: 0–3) затрагивают общие слова и
    оба блокируются проверкой перед применением. Из пересекающихся остаётся место с бо́льшим
    числом слов (для вложенных — внешнее): карточка адресует слова своего места разбора,
    поэтому границы не расширяются. Поглощённые места перечислены в `covers`, их число —
    в skipped["merged"]. При частичном пересечении слова поглощённого места вне оставшегося
    в карточку не входят.
    """
    order = {id(place): n for n, place in enumerate(places)}
    kept = []
    for place in sorted(places, key=lambda p: (-p["words"], order[id(p)])):
        host = next((k for k in kept if k["seg"] == place["seg"]
                     and place["w1"] < k["w2"] and k["w1"] < place["w2"]), None)
        if host is None:
            kept.append(place)
            continue
        skipped["merged"] += 1
        host.setdefault("covers", []).append(place["id"])
        if place["evidence"] not in host["evidence"].split("; "):
            host["evidence"] += "; " + place["evidence"]
    return sorted(kept, key=lambda p: order[id(p)])


def _spans(data, confirmed, report=None):
    """Отрезки «говорящий — время» для профиля: (имя, начало, конец, человек?).
    Человек: слова с подтверждённым автором, кроме наложений.
    Модель: только в подтверждённых расшифровках и только слова без пометок сомнения.
    report, если передан, получает счётчики причин без аудио и биометрии."""
    out = []
    def note(reason):
        if report is not None:
            report[reason] = report.get(reason, 0) + 1
    foreign = {}
    for seg in data["segments"]:
        words = seg.get("words") or []
        intervals = [(w.get("s"), w.get("e")) for w in words] if words else [(seg.get("start"), seg.get("end"))]
        for s, e in intervals:
            if isinstance(s, (int, float)) and isinstance(e, (int, float)) and math.isfinite(s) and math.isfinite(e) and s < e:
                foreign.setdefault(seg.get("speaker"), []).append((s, e))
    indexes = {}
    for speaker, intervals in foreign.items():
        intervals.sort()
        starts, maxima = [], []
        for start, end in intervals:
            starts.append(start)
            maxima.append(max(end, maxima[-1] if maxima else end))
        indexes[speaker] = (starts, maxima)

    def intersects_foreign(name, start, end):
        for speaker, (starts, maxima) in indexes.items():
            if speaker == name:
                continue
            index = bisect.bisect_left(starts, end)
            if index and maxima[index - 1] > start:
                return True
        return False
    for seg in data["segments"]:
        name = seg["speaker"]
        if str(name).startswith("Спикер"):
            note("unknown_speaker")
            continue
        run, kind = [], None
        for w in (seg.get("words") or []) + [None]:
            if w is not None:
                overlap = seg.get("overlap") or w.get("overlap") or "overlap" in (w.get("flags") or [])
                human = _word_human(w) and not overlap
                if not human and not overlap and _legacy_human_only(w):
                    note("legacy_human_without_speaker")
                auto = confirmed and not overlap and not w.get("flags") and not human and not w.get("inserted")
                k = "h" if human else ("a" if auto else None)
                if overlap:
                    note("overlap")
                elif not k:
                    note("unconfirmed_or_flagged")
                valid = isinstance(w.get("s"), (int, float)) and isinstance(w.get("e"), (int, float))
                valid = valid and math.isfinite(w["s"]) and math.isfinite(w["e"]) and 0 <= w["s"] < w["e"]
                if not valid:
                    k = None
                    note("invalid_time")
                elif w["e"] - w["s"] > LEARN_CAP:
                    k = None
                    note("word_too_long")
            else:
                k = None
            separated = bool(run and k and (w["s"] < run[-1]["s"] or w["s"] - run[-1]["e"] > LEARN_MAX_GAP
                                             or max(w["e"], *(x["e"] for x in run)) - run[0]["s"] > LEARN_CAP))
            if separated:
                note("gap_or_order")
            if run and (k != kind or separated):
                s, e = run[0]["s"], max(x["e"] for x in run)
                if intersects_foreign(name, s, e):
                    note("foreign_overlap")
                elif (kind == "h" and e - s >= HUMAN_MIN) or (kind == "a" and e - s >= LEARN_MIN):
                    out.append((name, s, e, kind == "h"))
                    note("accepted_human" if kind == "h" else "accepted_auto")
                else:
                    note("too_short")
                run = []
            if k:
                run.append(w)
                kind = k
        if not seg.get("words"):
            note("legacy_without_word_times")
    return out


def has_human_edits(data):
    """Есть ли в журнале действующая (не отменённая) человеческая правка."""
    return any(ed.get("by") == "human" and not ed.get("reverted")
               for ed in data.get("edits") or [])


def _human_pairs(data):
    """Поправки человека «профиль/модель сказали A — это B» из журнала правок. -> {(A, B): n}."""
    out = {}
    for ed in data.get("edits") or []:
        if ed.get("by") != "human" or ed.get("reverted"):
            continue
        a, b = (ed.get("before") or {}).get("speaker"), (ed.get("after") or {}).get("speaker")
        if a and b and a != b and "," not in a and "," not in b:
            out[(a, b)] = out.get((a, b), 0) + 1
        for line in (ed.get("pairs") or []):
            out[tuple(line)] = out.get(tuple(line), 0) + 1
    return out


def confusions(prof, exclude_audio=None):
    """{(сказал профиль, подтвердил человек): n} по всем записям, кроме exclude_audio."""
    ex = str(Path(exclude_audio).resolve()).lower() if exclude_audio else None
    out = {}
    for akey, pairs in (prof.get("confusions") or {}).items():
        if ex and akey.lower() == ex:
            continue
        for k, n in pairs.items():
            a, b = k.split("|", 1)
            out[(a, b)] = out.get((a, b), 0) + n
    return out


def _source_needs_review(prof, audio_key):
    for speaker in prof.get("speakers", {}).values():
        for source, part in speaker.get("parts", {}).items():
            if source.lower() == audio_key.lower() and part.get("selection_version") != SELECTION_VERSION:
                return True
    return any(source.lower() == audio_key.lower() and bool(items)
               for source, items in (prof.get("quarantine") or {}).items())


def _is_sqlite_export(data):
    passport = data.get("passport") if isinstance(data, dict) else None
    return isinstance(passport, dict) and passport.get("storage_backend") == "sqlite"


def learn(prof, folder, embedder, load_audio, names=None, limit=6, only=None, group=None, selection_report=None,
          stop_when=None, allow_replace=False, quality_policy="warn", quality_report=None):
    """Пополняет профиль из расшифровок в папке. -> список учтённых файлов.
    Вклад каждой записи хранится отдельно (ключ — путь к аудио). До реализации подтверждаемого
    предпросмотра существующий вклад не заменяется и не отзывается автоматически. На одну запись берётся самая
    свежая расшифровка, в которой есть подтверждённые имена или правки человека.
    В профиль идут: отрезки, подписанные человеком (вес ×3), и — в подтверждённых расшифровках — слова
    без пометок сомнения. Поправки «A → B» человеком копятся как спутанные пары (confusions).
    Если профиль участника после пересчёта сильно отличается от прежнего (сходство < DRIFT_MIN),
    это записывается в warnings. only — учесть только эти файлы .transcript.json.
    Защита от тёзок: расшифровка другой группы (group) пропускается; вклад записи, непохожий на уже
    собранный голос этого имени (SAME_MIN, SAME_MARGIN), в профиль не идёт, а ждёт решения человека
    в prof["quarantine"] (resolve_voice: тот же человек или другой).
    quality_policy: warn — только измерить и предупредить; reject_clipping — не учитывать источник
    с клиппингом. quality_report получает измерение последнего источника и признак отбраковки."""
    quality_policy = quality_policy if quality_policy in AUDIO_QUALITY_POLICIES else "warn"
    if quality_report is not None:
        quality_report["policy"] = quality_policy
    learned = []
    def note(reason):
        if selection_report is not None:
            selection_report[reason] = selection_report.get(reason, 0) + 1
    files = sorted(set(Path(folder).glob("*.transcript.json")) | set(Path(folder).glob("*.rus-transcriber/*.transcript.json")))
    if only:
        files = [Path(p) for p in only]
    best, conf_best = {}, {}
    for jp in files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            note("unreadable_transcript")
            continue
        if _is_sqlite_export(data):
            note("sqlite_source_not_supported")
            continue
        g = group_of(data)
        if group and g and g != group:       # расшифровка другой группы: её «Анна» — не наша
            note("other_group")
            continue
        if data.get("no_profile"):           # общий словарь без group: в профили групп не идёт
            note("profile_disabled")
            continue
        has_human = has_human_edits(data)
        if not (_confirmed(data) or has_human):
            note("unconfirmed_source")
            continue
        src = Path(data.get("file", ""))
        if not src.exists():
            note("missing_audio")
            continue
        akey = str(src.resolve())
        mt = jp.stat().st_mtime
        if akey not in best or mt > best[akey][1]:
            best[akey] = (jp, mt, data, src)
        if _confirmed(data) and (akey not in conf_best or mt > conf_best[akey][1]):
            conf_best[akey] = (jp, mt, data)
    if only and not allow_replace:  # при подтверждённом пересчёте используются только показанные данные
        for jp in sorted(set(Path(files[0]).parent.glob("*.transcript.json")) |
                         set(Path(files[0]).parent.parent.glob("*.transcript.json"))) if files else []:
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
                if _is_sqlite_export(data):
                    continue
                akey = str(Path(data.get("file", "")).resolve())
                if akey in best and _confirmed(data):
                    mt = jp.stat().st_mtime
                    if akey not in conf_best or mt > conf_best[akey][1]:
                        conf_best[akey] = (jp, mt, data)
            except Exception:
                continue
    prof.setdefault("history", [])
    prof.setdefault("warnings", [])
    prof.setdefault("confusions", {})
    for akey, (jp, mt, data, src) in best.items():
        if len(learned) >= limit or (stop_when is not None and stop_when()):
            break
        key = str(jp.resolve())
        current_fp = source_fingerprint(jp, src)
        status = source_fingerprint_status(prof["sources"].get(key), current_fp, mt)
        if not only:
            if status in ("unchanged", "legacy_unchanged"):
                if status == "legacy_unchanged":
                    # Старый источник уже был учтён до появления отпечатков. Принимаем текущее
                    # состояние как исходное, но не пересчитываем вклад автоматически.
                    prof["sources"][key] = current_fp
                    prof["provenance_revision"] = int(prof.get("provenance_revision", 0)) + 1
                    prof.setdefault("history", []).append({
                        "at": time.strftime("%Y-%m-%d %H:%M:%S"), "from": jp.name,
                        "kind": "source_fingerprint_accepted",
                        "note": "старый источник принят как исходный; вклад не пересчитывался"})
                version = (prof.get("source_selection_versions") or {}).get(key)
                note("already_processed" if version == SELECTION_VERSION and not _source_needs_review(prof, akey)
                     else "source_needs_review")
                continue
            if status == "changed":
                note("source_changed_requires_preview")
                continue
            if status == "legacy_changed" and any(akey in s.get("parts", {}) for s in prof["speakers"].values()):
                note("source_changed_requires_preview")
                continue
        before_update = copy.deepcopy(prof)
        before_report = dict(selection_report) if selection_report is not None else None
        try:
            existing = any(akey in s.get("parts", {}) for s in prof["speakers"].values())
            if existing and not allow_replace:
                note("existing_requires_preview")
                continue
            confirmed = _confirmed(data)
            spans = _spans(data, confirmed, selection_report)
            if not confirmed and akey in conf_best:
                # имена этой расшифровки не подтверждены: слова модели — из подтверждённой расшифровки той же
                # записи, отрезки человека — из этой (самой свежей)
                hum = [s for s in spans if s[3]]
                auto = [s for s in _spans(conf_best[akey][2], True, selection_report) if not s[3]
                        and not any(s[1] < h[2] and h[1] < s[2] for h in hum)]   # человек главнее
                spans = hum + auto
            spans = [s for s in spans if not names or s[0] in names]
            t0 = time.time()
            before = voice_vectors(prof)
            per = {}
            if spans:
                audio = load_audio(src)
                quality = audio_quality(audio)
                if quality_report is not None:
                    quality_report.update(quality)
                if (quality_policy == "reject_clipping" and quality.get("available")
                        and quality.get("clipping_ratio", 0.0) >= CLIP_REJECT_RATIO):
                    note("audio_quality_rejected")
                    if quality_report is not None:
                        quality_report["rejected"] = True
                        quality_report["reject_reason"] = "clipping"
                    continue
                vecs = embedder.spans(audio, [(s, e) for _, s, e, _ in spans], cap=LEARN_CAP)
                for (name, s, e, human), v in zip(spans, vecs):
                    if v is not None:
                        sec = min(e - s, LEARN_CAP)
                        per.setdefault(name, []).append((sec * (HUMAN_WEIGHT if human else 1.0), v, human, sec))
            if not per:
                note("no_usable_vectors")
                continue
            ident = prof.setdefault("identity", {}).get(akey, {})
            quar = prof.setdefault("quarantine", {})
            others = voice_vectors(prof, exclude_audio=src)
            other_sec = {n: sum(pt["sec"] for source, pt in sp.get("parts", {}).items() if source.lower() != akey.lower())
                         for n, sp in prof["speakers"].items()}
            same_sims = {}
            changed_names, quarantined_names = [], []
            for name, items in per.items():
                if ident.get(name) == "different":
                    note("identity_different")
                    continue                       # человек решил: в этой записи другой человек с этим именем
                c = _norm(sum(w * v for w, v, _, _ in items))
                good = [it for it in items if it[2] or float(it[1] @ c) >= 0.3] or items
                c = _norm(sum(w * v for w, v, _, _ in good))
                if c is None:
                    continue
                part_sec = sum(sec for _, _, _, sec in good)
                if ident.get(name) != "same" and name in others and other_sec.get(name, 0) >= GATE_MIN_SEC \
                        and part_sec >= GATE_PART_SEC:
                    sim = float(c @ others[name])
                    same_sims[name] = round(sim, 3)
                    near = max(((float(c @ v), n) for n, v in others.items() if n != name), default=(-1.0, None))
                    if sim < SAME_MIN or near[0] - sim >= SAME_MARGIN:
                        note("quarantined")
                        quarantined_names.append(name)
                        quar.setdefault(akey, {})[name] = {
                            "similarity": round(sim, 3), "closest_other": near[1],
                            "closest_other_sim": round(near[0], 3), "sec": round(part_sec, 1),
                            "from": jp.name, "json": str(jp), "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                        prof["warnings"].append({
                            "at": time.strftime("%Y-%m-%d %H:%M:%S"), "speaker": name, "similarity": round(sim, 3),
                            "from": jp.name, "kind": "namesake",
                            "note": (f"голос «{name}» в этой записи не похож на собранный профиль"
                                     + (f" (ближе к «{near[1]}»)" if near[0] - sim >= SAME_MARGIN else "")
                                     + " — тёзка или ошибка подписи; в профиль не добавлено, решение — resolve_voice")})
                        continue
                was_present = akey in prof["speakers"].get(name, {}).get("parts", {})
                prof["speakers"].setdefault(name, {"parts": {}})["parts"][akey] = {
                    "vec": c, "sec": round(sum(sec for _, _, h, sec in good if not h)
                                           + HUMAN_WEIGHT * sum(sec for _, _, h, sec in good if h), 1),
                    "human_sec": round(sum(sec for _, _, h, sec in good if h), 1),
                    "auto_sec": round(sum(sec for _, _, h, sec in good if not h), 1),
                    "from": jp.name, "confirmed": confirmed, "selection_version": SELECTION_VERSION,
                    "source_revision": data.get("source_revision")}
                note("replaced" if was_present else "added")
                changed_names.append(name)
                if akey in quar:
                    quar[akey].pop(name, None)
                    if not quar[akey]:
                        quar.pop(akey)
            if not changed_names and not quarantined_names:
                note("no_profile_change")
                continue
            prof.setdefault("audio_quality", {})[akey] = {
                "at": time.strftime("%Y-%m-%d %H:%M:%S"), "from": jp.name,
                **{k: v for k, v in quality.items() if k != "warnings"},
                "warnings": quality.get("warnings") or []}
            trim_quality_map(prof)
            if quality.get("warnings"):
                prof["warnings"].append({
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"), "speaker": "вся запись",
                    "from": jp.name, "kind": "audio_quality",
                    "quality_warnings": list(quality["warnings"]),
                    "note": "Качество аудио: " + ", ".join(explain_audio_quality(quality))
                            + ". Это измерение, а не автоматический отказ; решение — за человеком."})
            pairs = _human_pairs(data)
            prof["confusions"][akey] = {f"{a}|{b}": n for (a, b), n in pairs.items()}
            after = voice_vectors(prof)
            drift = {}
            for name in after:
                if name in before:
                    sim = float(before[name] @ after[name])
                    drift[name] = round(sim, 3)
                    if sim < DRIFT_MIN:
                        prof["warnings"].append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "speaker": name,
                                                 "similarity": round(sim, 3), "from": jp.name,
                                                 "note": "профиль сильно изменился — проверить отрезки этой записи"})
            prof["history"].append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "from": jp.name,
                                    "speakers": {n: {"human_sec": prof["speakers"][n]["parts"][akey]["human_sec"],
                                                     "auto_sec": prof["speakers"][n]["parts"][akey]["auto_sec"]}
                                                 for n in changed_names},
                                    "drift": drift, "pairs": {f"{a}|{b}": n for (a, b), n in pairs.items()},
                                    "same_sim": same_sims,
                                    **({"quarantine": sorted(quar[akey])} if quar.get(akey) else {})})
            prof["history"] = prof["history"][-200:]
            prof["warnings"] = prof["warnings"][-50:]
            prof["revision"] = int(prof.get("revision", 0)) + 1
            learned.append(jp.name)
            _log(f"профиль пополнен из {jp.name}: {sorted(per)} за {time.time() - t0:.1f} с")
            prof["sources"][key] = current_fp
            prof.setdefault("source_selection_versions", {})[key] = SELECTION_VERSION
        except Exception as ex:
            prof.clear()
            prof.update(before_update)
            if selection_report is not None:
                selection_report.clear()
                selection_report.update(before_report)
            note("processing_error")
            _log(f"{jp.name} пропущен: {type(ex).__name__}: {ex}")
    return learned


def match(cents, vecs, allowed=None):
    """Каждый кластер записи сравнивается с голосами профиля отдельно: имя даётся при сходстве
    ≥ MATCH_MIN и отрыве от второго ≥ MATCH_MARGIN. Несколько кластеров могут получить одно имя
    (pyannote иногда делит голос надвое); отсутствующий в записи участник никому не навязывается.
    -> {label: (имя, сходство)}"""
    names = [n for n in vecs if not allowed or n in allowed]
    out = {}
    for lbl, c in cents.items():
        sims = sorted(((float(c @ vecs[n]), n) for n in names), reverse=True)
        if not sims:
            continue
        best, second = sims[0], (sims[1][0] if len(sims) > 1 else -1.0)
        if best[0] >= MATCH_MIN and best[0] - second >= MATCH_MARGIN:
            out[lbl] = (best[1], round(best[0], 3))
    return out


def references(cents, turns, mapping):
    """Опорный голос каждого опознанного участника — центр его самого большого кластера в этой записи.
    -> ({имя: вектор}, {имя: label})"""
    size = {}
    for t in turns:
        size[t["label"]] = size.get(t["label"], 0.0) + t["e"] - t["s"]
    refs, main = {}, {}
    for lbl, (name, _) in mapping.items():
        if lbl in cents and (name not in main or size.get(lbl, 0) > size.get(main[name], 0)):
            main[name], refs[name] = lbl, cents[lbl]
    return refs, main


# ---------- проверка внутри записи ----------
def _quiet_point(audio, t, radius=0.4, frame=0.03):
    hop = int(frame * SR)
    a, b = int(max(0, t - radius) * SR), int(min(len(audio) / SR, t + radius) * SR)
    seg = audio[a:b]
    n = len(seg) // hop
    if n < 2:
        return t
    rms = np.sqrt((seg[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return (a + int(rms.argmin()) * hop + hop // 2) / SR


def scan_split(audio, turns, refs, name_of, label_of, embedder, fixes):
    """Ищет внутри длинных реплик куски с другим опознанным голосом и разрезает реплику.
    refs: {имя: вектор}; name_of: {label: имя}; label_of: {имя: основной label}."""
    names = list(refs)
    if len(names) < 2:
        return turns
    C = np.stack([refs[n] for n in names])
    out = []
    for t in turns:
        dur = t["e"] - t["s"]
        if dur < SCAN_MIN or name_of.get(t["label"]) not in refs:
            out.append(t)
            continue
        starts = np.arange(t["s"], t["e"] - WIN + 1e-6, HOP)
        vecs = embedder.spans(audio, [(s, s + WIN) for s in starts], min_len=WIN * 0.9)
        own = names.index(name_of[t["label"]])
        tags = []
        for v in vecs:
            if v is None:
                tags.append(own)
                continue
            sims = C @ v
            j = int(sims.argmax())
            tags.append(j if j != own and sims[j] - sims[own] >= SPLIT_MARGIN else own)
        # серии чужих окон
        runs, k = [], 0
        while k < len(tags):
            if tags[k] != own:
                j = k
                while j + 1 < len(tags) and tags[j + 1] == tags[k]:
                    j += 1
                if j - k + 1 >= SPLIT_MIN_RUN:
                    runs.append((k, j, tags[k]))
                k = j + 1
            else:
                k += 1
        if not runs:
            out.append(t)
            continue
        cur = t["s"]
        for k0, k1, lab in runs:
            a = _quiet_point(audio, max(t["s"], starts[k0] + HOP / 2))
            b = _quiet_point(audio, min(t["e"], starts[k1] + WIN - HOP / 2))
            a = cur if a - cur < 0.5 else a          # обрывок короче 0,5 с не оставляем
            b = t["e"] if t["e"] - b < 0.5 else b
            if b - a < 1.0 or a < cur:
                continue
            if a > cur:
                out.append({**t, "s": cur, "e": a})
            out.append({**t, "s": a, "e": b, "label": label_of[names[lab]],
                        "fix": f"голос внутри реплики: {name_of[t['label']]}→{names[lab]}"})
            fixes.append({"t": round(a, 1), "kind": "split", "from": name_of[t["label"]], "to": names[lab],
                          "sec": round(b - a, 1)})
            cur = b
        if t["e"] > cur:
            out.append({**t, "s": cur, "e": t["e"]})
    return out


def reassign(turns, embs, pvecs, name_of, label_of, fixes, debug=None, new_label=None):
    """Пореплично сверяет голос с профилями участников. Своему кластеру даётся поправка OWN_BONUS:
    реплика переходит к другому участнику, только если её голос ближе к его профилю с запасом.
    Подобрано на отрывке 06.11 (профиль — только по 05.11): совпадение подписей с выверенной
    стенограммой по словам 93,7% -> 95,1%. Участник, которого pyannote не выделил в отдельный кластер
    (слил с другим), получает новую метку через new_label(имя).
    debug: сюда пишется сходство каждой реплики с каждым голосом."""
    names = list(pvecs)
    if len(names) < 2:
        return turns
    C = np.stack([pvecs[n] for n in names])
    out = []
    for t, v in zip(turns, embs):
        own = name_of.get(t["label"])
        if v is None:
            out.append(t)
            continue
        sims = C @ v
        if debug is not None:
            debug.append({"t": round(t["s"], 2), "dur": round(t["e"] - t["s"], 2), "speaker": own or t["label"],
                          "sims": {n: round(float(x), 3) for n, x in zip(names, sims)}})
        if t["e"] - t["s"] < REASSIGN_MIN or t.get("fix"):
            out.append(t)
            continue
        score = sims + np.array([OWN_BONUS if n == own else 0.0 for n in names])
        j = int(score.argmax())
        best = names[j]
        if own is None:
            srt = np.sort(sims)
            if sims[j] < MATCH_MIN or srt[-1] - srt[-2] < 0.05:
                out.append(t)
                continue
        if best != own:
            lbl = label_of.get(best) or (new_label(best) if new_label else None)
            if lbl:
                fixes.append({"t": round(t["s"], 1), "kind": "voice", "from": own or t["label"], "to": best,
                              "sim": [round(float(sims[names.index(own)]), 2) if own in names else None,
                                      round(float(sims[j]), 2)]})
                t = {**t, "label": lbl, "fix": f"голос реплики: {own or t['label']}→{best}"}
        out.append(t)
    return out


def merge_adjacent(turns, gap=0.3):
    out = []
    for t in sorted(turns, key=lambda x: x["s"]):
        if out and out[-1]["label"] == t["label"] and t["s"] - out[-1]["e"] < gap and not t.get("fix") \
                and not out[-1].get("fix"):
            out[-1] = {**out[-1], "e": max(out[-1]["e"], t["e"])}
        else:
            out.append(dict(t))
    return out
