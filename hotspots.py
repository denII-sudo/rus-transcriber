"""Горячие места разговора: слой признаков (считается при обработке) и оценка (при чтении).

Горячее место — отрезок, где разговор оживлён или напряжён: говорящие часто сменяются,
речь накладывается, голос громче или выше обычного для этого говорящего в этой записи.
Это оценка мест разговора для навигации, а не эмоций людей и не доказательство.

Слой (quality["interaction"] стенограммы) — только числа, привязанные ко времени: интервалы
наложений, реплики диаризации с номером кластера, громкость и высота голоса по окнам.
В нём нет текста, имён и отпечатков голоса. Модели не используются: только numpy.

Оценка считается при чтении по текущим словам стенограммы: правки человека (кто говорил)
учитываются без пересчёта слоя.
"""
import math

import numpy as np

SCHEMA_VERSION = 1
STEP_SEC = 1.0            # шаг окон слоя, с

# ---- высота голоса (F0): автокорреляция кадра с поправкой на окно ----
F0_MIN_HZ, F0_MAX_HZ = 65.0, 500.0
FRAME_SEC, HOP_SEC = 0.048, 0.016
VOICING_MIN = 0.5         # нормированный пик автокорреляции; ниже — кадр считается невокализованным
SILENCE_DB = -55.0        # кадр тише этого уровня не оценивается
OCTAVE_COST = 0.02        # штраф длинным периодам: из равных пиков берётся основной тон, а не его доля
MIN_VOICED_FRAMES = 5     # меньше вокализованных кадров в окне — высота окна пуста (None)
RMS_FLOOR_DB = -100.0     # громкость цифровой тишины
_PITCH_BLOCK = 2048       # кадров за один вызов FFT (ограничивает память)

# ---- оценка: СТАРТОВЫЕ значения без эталона. Они подобраны из общих соображений и на
# синтетических сигналах; на размеченных разговорах не проверялись и не настраивались.
# Менять их следует по результатам прослушивания, а не «на глаз». ----
CONTEXT_SEC = 10.0                    # окрестность, в которой считаются смены говорящих и доля наложений
TURN_RATE_LO, TURN_RATE_HI = 2.0, 6.0  # смен говорящего за 10 с: ниже — 0, выше — 1
OVERLAP_LO, OVERLAP_HI = 0.05, 0.35    # доля окрестности, занятая наложениями: ниже — 0, выше — 1
Z_LO, Z_HI = 1.0, 3.0                  # отклонение вверх от обычного уровня говорящего, в устойчивых разбросах
LOUD_MIN_SCALE_DB = 1.5               # нижняя граница разброса громкости, дБ (ровная запись не даёт ложных пиков)
PITCH_MIN_SCALE_ST = 1.0              # нижняя граница разброса высоты, полутоны
MIN_COVERAGE = 0.5                    # доля окна, занятая словами: меньше — громкость и высота не оцениваются
MIN_DOMINANCE = 0.7                   # доля слов окна у одного говорящего: меньше — окно «ничьё»
MAX_OVERLAP_FOR_VOICE = 0.25          # в окне с наложением два голоса: громкость и высоту не сравниваем
MIN_BASELINE_WINDOWS = 20             # меньше окон у говорящего — обычный уровень не определяется
INSERT_SEC = 0.3                      # условная длительность вставки-поддакивания (у неё только время начала)
HOT_SCORE = 0.5                       # окно горячее, если общая оценка не ниже
KIND_SCORE = 0.3                      # вид указывается, если его частная оценка в отрезке не ниже
MERGE_GAP_SEC = 3.0                   # горячие окна с разрывом не больше — один отрезок
MIN_LEN_SEC = 2.0                     # отрезки короче отбрасываются

# Виды названы по тому, что измерено, а не словами об эмоциях.
KIND_TURNS = "частые смены говорящих"
KIND_OVERLAP = "наложения"
KIND_LOUD = "громче обычного"
KIND_PITCH = "тон выше обычного"
KINDS = (KIND_TURNS, KIND_OVERLAP, KIND_LOUD, KIND_PITCH)

NOTE = ("Горячее место — эвристика для навигации по записи: где разговор был оживлён или напряжён "
        "(смены говорящих, наложения, громкость и высота голоса относительно обычных для говорящего). "
        "Это не оценка эмоций или поведения людей и не доказательство; пороги стартовые, без эталона.")


class NoData(ValueError):
    """Слоя признаков нет или он не читается; code — причина для программы, текст — для человека."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- слой признаков
def window_rms_db(audio, sr, step=STEP_SEC):
    """Громкость по окнам шага step: RMS в дБ относительно полной шкалы (1.0 -> 0 дБ)."""
    x = np.asarray(audio, dtype=np.float32)
    size = int(round(sr * step))
    if size <= 0:
        raise ValueError("шаг окна слишком мал")
    count = int(math.ceil(len(x) / size))
    power = np.zeros(count)
    for i in range(count):      # по одному окну: без временных массивов размером со всю запись
        part = x[i * size:(i + 1) * size]
        power[i] = np.mean(np.square(part, dtype=np.float64))
    return np.maximum(10.0 * np.log10(np.maximum(power, 1e-12)), RMS_FLOOR_DB)


def frame_pitch(audio, sr):
    """Высота голоса по кадрам -> (время середины кадра, с; F0, Гц или nan).

    Автокорреляция кадра (окно Ханна) делится на автокорреляцию окна; период — локальный
    максимум в пределах F0_MIN_HZ…F0_MAX_HZ, уточнённый параболой.
    """
    x = np.asarray(audio, dtype=np.float32)
    frame = int(round(FRAME_SEC * sr))
    hop = int(round(HOP_SEC * sr))
    lag_min = max(2, int(sr / F0_MAX_HZ))
    lag_max = int(math.ceil(sr / F0_MIN_HZ))
    if frame <= lag_max + 2 or len(x) < frame:
        return np.zeros(0), np.zeros(0)
    count = 1 + (len(x) - frame) // hop
    nfft = 1 << (frame + lag_max + 2 - 1).bit_length()   # без заворота автокорреляции
    window = np.hanning(frame).astype(np.float32)
    wac = np.fft.irfft(np.abs(np.fft.rfft(window, nfft)) ** 2, nfft)[:lag_max + 2]
    wac = wac / wac[0]
    lags = np.arange(lag_min, lag_max + 1)
    cost = OCTAVE_COST * np.log2(lags / lag_min)
    offsets = np.arange(frame)
    f0 = np.full(count, np.nan)
    for start in range(0, count, _PITCH_BLOCK):
        idx = np.arange(start, min(count, start + _PITCH_BLOCK))
        frames = x[idx[:, None] * hop + offsets]
        frames = frames - frames.mean(axis=1, keepdims=True)
        level = 10.0 * np.log10(np.maximum(np.mean(np.square(frames, dtype=np.float64), axis=1), 1e-12))
        spec = np.fft.rfft(frames * window, nfft, axis=1)
        ac = np.fft.irfft(spec.real ** 2 + spec.imag ** 2, nfft, axis=1)[:, :lag_max + 2]
        r0 = ac[:, 0]
        loud = (r0 > 0) & (level > SILENCE_DB)
        r = ac / np.where(r0 > 0, r0, 1.0)[:, None] / wac
        mid = r[:, lag_min:lag_max + 1]
        is_peak = (mid > r[:, lag_min - 1:lag_max]) & (mid >= r[:, lag_min + 1:lag_max + 2])
        cand = np.where(is_peak, mid - cost, -np.inf)
        best = cand.argmax(axis=1)
        rows = np.arange(len(idx))
        found = is_peak[rows, best]
        lag = best + lag_min
        a, b, c = r[rows, lag - 1], r[rows, lag], r[rows, lag + 1]
        denom = a - 2.0 * b + c
        shift = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / np.where(denom == 0, 1.0, denom), 0.0)
        period = lag + np.clip(shift, -1.0, 1.0)
        voiced = loud & found & (b >= VOICING_MIN)
        f0[idx] = np.where(voiced, sr / period, np.nan)
    centers = (np.arange(count) * hop + frame / 2.0) / sr
    return centers, f0


def window_f0(audio, sr, count, step=STEP_SEC):
    """Высота голоса по окнам: медиана F0 вокализованных кадров окна, иначе nan."""
    out = np.full(count, np.nan)
    centers, f0 = frame_pitch(audio, sr)
    if not len(f0):
        return out
    bounds = np.searchsorted(centers, np.arange(count + 1) * step, side="left")
    for i in range(count):
        part = f0[bounds[i]:bounds[i + 1]]
        part = part[np.isfinite(part)]
        if len(part) >= MIN_VOICED_FRAMES:
            out[i] = np.median(part)
    return out


def cluster_turns(raw, annotation=None):
    """Реплики диаризации с номером кластера вместо метки -> (реплики [(s, e, номер)], источник).

    annotation — разметка pyannote с наложениями (itertracks); если её нет, берётся raw —
    список (начало, конец, метка) исключающей разметки. Номер — порядок первого появления
    кластера во времени: метки pyannote и имена участников в слой не попадают.
    """
    rows, source = [], "exclusive"
    if annotation is not None:
        rows = [(float(seg.start), float(seg.end), label)
                for seg, _, label in annotation.itertracks(yield_label=True)]
        source = "overlapping"
    if not rows:
        rows, source = [(float(s), float(e), label) for s, e, label in raw or []], "exclusive"
    rows.sort(key=lambda row: (row[0], row[1]))
    index = {}
    turns = []
    for s, e, label in rows:
        turns.append((s, e, index.setdefault(label, len(index))))
    return turns, source


def build_layer(audio, sr, turns, overlaps, *, turns_source="", step=STEP_SEC):
    """Слой признаков разговора для quality["interaction"].

    audio — моно, float; turns — [(начало, конец, номер кластера)] (см. cluster_turns);
    overlaps — [(начало, конец)] наложений. Всё округляется: времена до 0,01 с, громкость
    до 0,1 дБ, высота до 1 Гц. Значения — только числа и None (JSON без NaN).
    """
    x = np.asarray(audio)
    if x.ndim != 1:
        raise ValueError("ожидается одноканальный звук")
    if sr <= 0 or step <= 0:
        raise ValueError("частота и шаг должны быть положительными")
    duration = len(x) / float(sr)
    rms = window_rms_db(x, sr, step)
    if not np.isfinite(rms).all():
        raise ValueError("звук содержит нечисловые значения")
    f0 = window_f0(x, sr, len(rms), step)

    def clip(t):
        return round(min(max(float(t), 0.0), duration), 2)

    turn_rows = sorted((clip(s), clip(e), int(c)) for s, e, c in turns or [])
    turn_rows = [row for row in turn_rows if row[1] > row[0]]
    over_rows = sorted((clip(s), clip(e)) for s, e in overlaps or [])
    over_rows = [row for row in over_rows if row[1] > row[0]]
    return {
        "schema_version": SCHEMA_VERSION,
        "duration": round(duration, 2),
        "overlaps": {"s": [r[0] for r in over_rows], "e": [r[1] for r in over_rows]},
        "turns": {"source": str(turns_source), "s": [r[0] for r in turn_rows],
                  "e": [r[1] for r in turn_rows], "cluster": [r[2] for r in turn_rows]},
        "windows": {"step": float(step),
                    "rms_db": [round(float(v), 1) for v in rms],
                    "f0_hz": [None if not np.isfinite(v) else int(round(float(v))) for v in f0]},
    }


# ---------------------------------------------------------------- чтение слоя
def _num(value):
    return type(value) in (int, float) and math.isfinite(value)


def _layer(data):
    quality = (data or {}).get("quality")
    quality = quality if isinstance(quality, dict) else {}
    layer = quality.get("interaction")
    if layer is None:
        if quality.get("interaction_error"):
            raise NoData("error", "при обработке записи слой признаков рассчитать не удалось: "
                         + str(quality["interaction_error"])[:300])
        if quality.get("interaction_skipped") == "disabled":
            raise NoData("disabled", "при обработке записи настройка interaction была выключена")
        raise NoData("absent", "запись обработана до появления слоя признаков или прежним путём "
                               "(pipeline = segments); старые записи не пересчитываются")
    if not isinstance(layer, dict) or not _num(layer.get("schema_version")):
        raise NoData("damaged", "слой признаков повреждён: нет версии схемы")
    if layer["schema_version"] > SCHEMA_VERSION:
        raise NoData("newer_schema", "слой признаков записан более новой версией программы; обновите программу")
    windows, overlaps = layer.get("windows"), layer.get("overlaps")
    ok = (layer["schema_version"] == SCHEMA_VERSION and isinstance(windows, dict) and isinstance(overlaps, dict)
          and _num(windows.get("step")) and windows["step"] > 0
          and isinstance(windows.get("rms_db"), list) and isinstance(windows.get("f0_hz"), list)
          and len(windows["rms_db"]) == len(windows["f0_hz"])
          and all(_num(v) for v in windows["rms_db"])
          and all(v is None or _num(v) for v in windows["f0_hz"])
          and isinstance(overlaps.get("s"), list) and isinstance(overlaps.get("e"), list)
          and len(overlaps["s"]) == len(overlaps["e"])
          and all(_num(v) for v in overlaps["s"]) and all(_num(v) for v in overlaps["e"]))
    if not ok:
        raise NoData("damaged", "слой признаков повреждён: неожиданная структура")
    return layer


def no_data_reason(data):
    """None, если слой есть и читается; иначе (код причины, пояснение для человека)."""
    try:
        _layer(data)
    except NoData as ex:
        return ex.code, str(ex)
    return None


# ---------------------------------------------------------------- оценка
def _spread(intervals, count, step):
    """Сколько секунд каждого окна занято интервалами [(s, e)]."""
    out = np.zeros(count)
    end = count * step
    for s, e in intervals:
        s, e = max(float(s), 0.0), min(float(e), end)
        if e <= s:
            continue
        first, last = int(s // step), min(int(e // step), count - 1)
        for i in range(first, last + 1):
            out[i] += max(0.0, min(e, (i + 1) * step) - max(s, i * step))
    return out


def _speech_events(data):
    """Текущие слова стенограммы как события (начало, конец, говорящий) по времени."""
    events = []
    for seg in (data or {}).get("segments") or []:
        if not isinstance(seg, dict):
            continue
        speaker = seg.get("speaker") or seg.get("label")
        words = [w for w in seg.get("words") or [] if isinstance(w, dict) and _num(w.get("s"))]
        if speaker and words:
            for w in words:
                s = float(w["s"])
                e = float(w["e"]) if _num(w.get("e")) else s
                events.append((s, max(e, s + 0.05), speaker))
        elif speaker and _num(seg.get("start")) and _num(seg.get("end")):
            events.append((float(seg["start"]), float(seg["end"]), speaker))
        for ins in seg.get("inserts") or []:     # поддакивание другого голоса внутри реплики
            if isinstance(ins, dict) and ins.get("speaker") and _num(ins.get("t")):
                events.append((float(ins["t"]), float(ins["t"]) + INSERT_SEC, ins["speaker"]))
    events.sort(key=lambda ev: (ev[0], ev[1]))
    return events


def _context(values, half):
    """Сумма по окну и half соседям с каждой стороны -> (сумма, число окон в окрестности)."""
    count = len(values)
    csum = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(count)
    lo, hi = np.maximum(idx - half, 0), np.minimum(idx + half + 1, count)
    return csum[hi] - csum[lo], hi - lo


def _robust_z(values, min_scale):
    """Отклонение от медианы в устойчивых разбросах (1,4826·MAD, но не меньше min_scale)."""
    median = np.median(values)
    scale = max(1.4826 * float(np.median(np.abs(values - median))), min_scale)
    return (values - median) / scale


def _smooth(z):
    """Среднее по окну и двум соседям; пропуски (nan) не участвуют и остаются пропусками."""
    ok = np.isfinite(z)
    v = np.where(ok, z, 0.0)
    total, weight = v.copy(), ok.astype(float)
    total[1:] += v[:-1]
    weight[1:] += ok[:-1]
    total[:-1] += v[1:]
    weight[:-1] += ok[1:]
    return np.where(ok, total / np.maximum(weight, 1.0), np.nan)


def _ramp(values, lo, hi):
    return np.clip((np.nan_to_num(values, nan=lo) - lo) / (hi - lo), 0.0, 1.0)


def window_scores(data):
    """Частные и общая оценки по окнам слоя -> {"step", "score", "kinds": {вид: оценки}}.

    Говорящий окна берётся из текущих слов стенограммы (правки человека учитываются);
    обычный уровень громкости и высоты — медиана и устойчивый разброс того же говорящего
    в этой записи. NoData, если слоя нет.
    """
    layer = _layer(data)
    windows = layer["windows"]
    step = float(windows["step"])
    rms = np.array(windows["rms_db"], dtype=float)
    f0 = np.array([np.nan if v is None else v for v in windows["f0_hz"]], dtype=float)
    count = len(rms)
    empty = {"step": step, "score": np.zeros(0), "kinds": {kind: np.zeros(0) for kind in KINDS}}
    if not count:
        return empty

    events = _speech_events(data)
    speakers = list(dict.fromkeys(ev[2] for ev in events))
    speech = np.zeros((max(len(speakers), 1), count))
    for k, name in enumerate(speakers):
        speech[k] = _spread([(s, e) for s, e, who in events if who == name], count, step)
    changes = np.zeros(count)
    previous = None
    for s, _, who in events:
        if previous is not None and who != previous and 0 <= s < count * step:
            changes[int(s // step)] += 1
        previous = who

    half = max(1, int(round(CONTEXT_SEC / step / 2)))
    change_sum, width = _context(changes, half)
    turn_rate = change_sum / (width * step) * 10.0          # смен говорящего за 10 с
    overlap = _spread(zip(layer["overlaps"]["s"], layer["overlaps"]["e"]), count, step)
    overlap_sum, _ = _context(overlap, half)
    overlap_share = overlap_sum / (width * step)

    total = speech.sum(axis=0)
    dominant = speech.argmax(axis=0)
    share = np.where(total > 0, speech.max(axis=0) / np.where(total > 0, total, 1.0), 0.0)
    usable = ((total / step >= MIN_COVERAGE) & (share >= MIN_DOMINANCE)
              & (overlap / step <= MAX_OVERLAP_FOR_VOICE))
    z_loud = np.full(count, np.nan)
    z_pitch = np.full(count, np.nan)
    for k in range(len(speakers)):
        own = usable & (dominant == k)
        if own.sum() >= MIN_BASELINE_WINDOWS:
            z_loud[own] = _robust_z(rms[own], LOUD_MIN_SCALE_DB)
        voiced = own & np.isfinite(f0) & (f0 > 0)
        if voiced.sum() >= MIN_BASELINE_WINDOWS:
            z_pitch[voiced] = _robust_z(12.0 * np.log2(f0[voiced]), PITCH_MIN_SCALE_ST)

    # Отклонение окна ограничено средним с соседями: одиночный всплеск (кашель, стук) гасится,
    # а соседнее окно другого говорящего не «заражается» чужим громким окном.
    with np.errstate(invalid="ignore"):
        z_loud = np.minimum(z_loud, _smooth(z_loud))
        z_pitch = np.minimum(z_pitch, _smooth(z_pitch))
    kinds = {KIND_TURNS: _ramp(turn_rate, TURN_RATE_LO, TURN_RATE_HI),
             KIND_OVERLAP: _ramp(overlap_share, OVERLAP_LO, OVERLAP_HI),
             KIND_LOUD: _ramp(z_loud, Z_LO, Z_HI),
             KIND_PITCH: _ramp(z_pitch, Z_LO, Z_HI)}
    # Общая оценка: 1 − произведение (1 − частная). Один сильный признак даёт столько же,
    # сколько его частная оценка; несколько признаков вместе усиливают друг друга.
    miss = np.ones(count)
    for values in kinds.values():
        miss *= 1.0 - values
    return {"step": step, "score": 1.0 - miss, "kinds": kinds}


def find_hotspots(data, *, min_score=HOT_SCORE):
    """Горячие места стенограммы -> [{"t1", "t2", "kinds", "score"}] по порядку времени.

    data — стенограмма, как её отдаёт transcript.load: текущие слова с говорящими и слой
    quality["interaction"]. t1, t2 — секунды от начала записи; kinds — что измерено
    (см. KINDS); score — 0…1, наибольшая оценка окна в отрезке. Пустой список — горячих
    мест нет. NoData — слоя нет (причина в исключении, см. no_data_reason).
    """
    scored = window_scores(data)
    step, score = scored["step"], scored["score"]
    count = len(score)
    if not count:
        return []
    duration = _layer(data).get("duration")
    end_time = float(duration) if _num(duration) and duration > 0 else count * step
    hot = np.flatnonzero(score >= float(min_score))
    runs = []
    gap = int(round(MERGE_GAP_SEC / step))
    for i in hot:
        if runs and i - runs[-1][1] - 1 <= gap:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    out = []
    for first, last in runs:
        t1, t2 = first * step, min((last + 1) * step, max(end_time, first * step))
        if t2 - t1 < MIN_LEN_SEC:
            continue
        part = {kind: float(values[first:last + 1].max()) for kind, values in scored["kinds"].items()}
        kinds = [kind for kind in KINDS if part[kind] >= KIND_SCORE] or [max(KINDS, key=lambda kind: part[kind])]
        out.append({"t1": round(t1, 2), "t2": round(t2, 2), "kinds": kinds,
                    "score": round(float(score[first:last + 1].max()), 2)})
    return out


def match_sections(sections, places, *, top_n=None, min_score=None):
    """Распределить места по разделам: список списков в порядке входных разделов.

    sections — отрезки {"t1", "t2", ...} из workflow.section_time_spans;
    places — места {"t1", "t2", "score", "kinds", ...} из find_hotspots.
    Дополнительные поля сохраняются: места возвращаются без изменения.
    Выберите ровно один режим: top_n — N сильнейших мест всей записи (включая
    места вне разделов), min_score — все места с оценкой не ниже порога.
    При равной силе раньше выбирается место, идущее раньше во входном списке.

    Место относится только к одному разделу с наибольшей длиной пересечения;
    при равенстве с учётом погрешности float (четыре ULP границ) — к разделу
    с более ранним началом, а при одинаковом начале — к первому во входе.
    Время места — конечные числа 0 <= t1 < t2, score — конечное число от 0 до 1;
    все места проверяются до отбора, иначе ValueError с номером места.
    Касание границ не считается пересечением. Разделы
    без времени или нулевой длины остаются пустыми. Места внутри каждого
    раздела идут в исходном порядке; входные данные не изменяются.
    """
    if (top_n is None) == (min_score is None):
        raise ValueError("укажите ровно один режим: top_n или min_score")
    if top_n is not None and (type(top_n) is not int or top_n < 0):
        raise ValueError("top_n должен быть неотрицательным целым числом")
    if min_score is not None and not _num(min_score):
        raise ValueError("min_score должен быть конечным числом")

    for i, place in enumerate(places):
        if not isinstance(place, dict):
            raise ValueError(f"место {i}: ожидается словарь")
        start, end, score = place.get("t1"), place.get("t2"), place.get("score")
        if not _num(start) or not _num(end) or not 0 <= start < end:
            raise ValueError(f"место {i}: ожидаются конечные времена 0 <= t1 < t2")
        if not _num(score) or not 0 <= score <= 1:
            raise ValueError(f"место {i}: score должен быть конечным числом от 0 до 1")

    if top_n is not None:
        selected = sorted(range(len(places)), key=lambda i: -places[i]["score"])[:top_n]
        selected.sort()
    else:
        selected = [i for i, place in enumerate(places) if place["score"] >= min_score]

    result = [[] for _ in sections]
    for i in selected:
        place = places[i]
        best_index, best_overlap, best_start, best_tolerance = None, 0, None, 0
        for j, section in enumerate(sections):
            start, end = section.get("t1"), section.get("t2")
            if not _num(start) or not _num(end) or end <= start:
                continue
            overlap = min(end, place["t2"]) - max(start, place["t1"])
            if overlap <= 0:
                continue
            tolerance = 4 * max(math.ulp(v) for v in (start, end, place["t1"], place["t2"]))
            tied = best_index is not None and abs(overlap - best_overlap) <= max(tolerance, best_tolerance)
            if best_index is None or (tied and start < best_start) or (not tied and overlap > best_overlap):
                best_index, best_overlap, best_start, best_tolerance = j, overlap, start, tolerance
        if best_index is not None:
            result[best_index].append(place)
    return result
