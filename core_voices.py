"""Голоса: профили, качество, пополнение. Перенесено из core.py без смены логики (разрезка core.py, перенос 1).

Прежние имена `core.X` работают: core.py в конце файла переотдаёт их отсюда. Всё, что раздел берёт из core.py вне
своего раздела (`CoreError`, `job_json`, `launch`), берётся как `core.X` в момент вызова, а не копируется: так
подмены `patch.object(core, "job_json")` в тестах действуют. `import core` — последняя строка файла (круговой
импорт: core.py в конце загружает этот модуль). Проверяет tests/test_core_split.py.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402
import settings  # noqa: E402


def _voice_vocab_context(data, json_path, group_override=None):
    """Разрешает группу и только точечные словари, сверяя их с привязкой стенограммы."""
    import vocab
    import voices

    try:
        binding = voices.group_binding(data)
        if binding["conflict"]:
            raise vocab.ResolutionError("Группа расшифровки противоречит quality.vocab",
                                        {"binding": binding, "candidates": []})
        if binding["vocab_status"] == "invalid":
            raise vocab.ResolutionError("Ссылка quality.vocab повреждена или невалидна",
                                        {"binding": binding, "candidates": []})
        if group_override:
            if not isinstance(group_override, str) or paths.safe_group(group_override) != group_override:
                raise core.CoreError("недопустимое имя выбранной группы")
            if binding["group"] and paths.safe_group(binding["group"]) != group_override:
                raise vocab.ResolutionError("Выбранная группа противоречит привязке расшифровки",
                                            {"binding": binding, "candidates": []})
            canonical = paths.vocab_path(group_override)
            path, data_vocab, group, resolution, binding = vocab.resolve_group_vocab(
                data.get("file"), json_path, data, group_override, canonical)
            if not paths.group_dir(group_override).is_dir():
                raise core.CoreError("выбранная группа не существует")
        else:
            candidates = vocab.related_candidates(
                data.get("file"), json_path, binding.get("vocab_path"))
            group = binding.get("group")
            has_existing = any(Path(path).expanduser().is_file() for path, _ in candidates)
            if not group and has_existing:
                _, _, group, _ = vocab.resolve_candidates(candidates)
            if group:
                candidates.append((paths.vocab_path(group), "group_fallback"))
            path, data_vocab, group, resolution = vocab.resolve_candidates(
                candidates, expected_group=group)
            resolution["binding"] = binding
        if not group:
            raise core.CoreError("группа расшифровки неизвестна; укажите явную группу для предпросмотра")
        return {"group": group, "path": path, "vocab": data_vocab, "binding": binding,
                "resolution": resolution,
                "manual_override": bool(group_override and not binding.get("group")),
                "override_requested": bool(group_override)}
    except vocab.ResolutionError as ex:
        raise core.CoreError(f"не удалось безопасно разрешить группу и словарь: {ex}") from ex


def _open_voice_source(ref):
    """Источник обучения (файл или запись в базе) с отказом, понятным человеку."""
    import storage
    import voice_sources
    try:
        return voice_sources.open_ref(str(ref))
    except voice_sources.SourceError as ex:
        raise core.CoreError(str(ex)) from ex
    except (storage.StorageError, OSError, KeyError, ValueError) as ex:
        raise core.CoreError(f"источник для голосового профиля недоступен: {type(ex).__name__}: {ex}") from ex


def _voice_job_source_ref(job_id):
    """Ссылка на источник по номеру задачи или `recording:<id>`.

    `recording:<id>` идёт в voice_sources напрямую: там база открывается только на чтение, а общий
    core.job_json открыл бы её на запись.
    """
    import voice_sources
    if isinstance(job_id, str) and job_id.startswith(voice_sources.RECORDING_PREFIX):
        return job_id
    jp, _ = core.job_json(job_id)
    if not jp:
        raise core.CoreError("готовая расшифровка не найдена")
    return jp


def _source_contributions(profile, source, key):
    """Пометки с переноса, которые заменит обновление, и вклады из других вариантов стенограммы."""
    confirmation, other = [], set()
    same = {source.name, source.migrated_from} - {None}
    for name, speaker in sorted((profile.get("speakers") or {}).items()):
        part = (speaker.get("parts") or {}).get(key)
        if not part:
            continue
        if part.get("confirmation") is not None:
            confirmation.append(name)
        if part.get("from") and part["from"] not in same:
            other.add(part["from"])
    return confirmation, sorted(other)


def preview_voice_learning(job_id, group_override=None):
    """Предварительная оценка по разметке, без моделей, биометрических векторов и изменения файлов."""
    import voices
    import vocab
    import voice_sources
    source = _open_voice_source(_voice_job_source_ref(job_id))
    try:
        data = source.data
        if data.get("no_profile"):
            raise core.CoreError("для этой расшифровки профили групп отключены")
        context = _voice_vocab_context(data, source.json_path, group_override)
        group = context["group"]
        group_vocab = context["vocab"]
        path = paths.voices_path(group)
        try:
            profile = voices.load_profile(path, convert=False)
        except voices.VoiceProfileError as ex:
            raise core.CoreError(f"голосовой профиль группы несовместим или повреждён: {ex}. "
                            "Просмотр и изменение не выполняются; перенесите профиль отдельно.") from ex
        audio = data.get("file")
        if not audio:
            raise core.CoreError("в расшифровке не указан исходный аудиофайл")
        key = str(Path(audio).resolve())
        if source.kind == "sqlite":
            key = voice_sources.existing_audio_key(profile, key)
        existing = sorted(name for name, speaker in profile.get("speakers", {}).items()
                          if key in speaker.get("parts", {}))
        confirmation_replaced, other_variants = _source_contributions(profile, source, key)
        report = {}
        eligible = voices._confirmed(data) or voices.has_human_edits(data)
        spans = voices._spans(data, voices._confirmed(data), report) if eligible else []
        if not eligible:
            report["unconfirmed_source"] = 1
        participants = (group_vocab or {}).get("participants") or []
        allowed_names = {p["name"] for p in participants if p.get("name")}
        outside_vocab = sorted({name for name, _, _, _ in spans if allowed_names and name not in allowed_names})
        if allowed_names:
            spans = [item for item in spans if item[0] in allowed_names]
        candidates = {}
        human_seconds = {}
        for name, start, end, human in spans:
            item = candidates.setdefault(name, {"windows": 0, "seconds": 0.0, "human_windows": 0})
            item["windows"] += 1
            item["seconds"] += end - start
            item["human_windows"] += int(human)
            if human:
                human_seconds[name] = human_seconds.get(name, 0.0) + (end - start)
        for item in candidates.values():
            item["seconds"] = round(item["seconds"], 2)
        # «человеческие» секунды вклада этой записи: сейчас в профиле → после замены по текущей разметке
        human_change, human_warnings = {}, []
        for name in sorted(set(existing) | set(human_seconds)):
            part = profile.get("speakers", {}).get(name, {}).get("parts", {}).get(key) or {}
            before_h = round(float(part.get("human_sec") or 0.0), 2)
            after_h = round(human_seconds.get(name, 0.0), 2)
            if before_h or after_h:
                human_change[name] = {"before": before_h, "after": after_h}
            if before_h > 0 and after_h == 0:
                human_warnings.append(name)
        return {"group": group, "existing_contributions": existing, "candidates": candidates,
                "human_seconds_change": human_change, "human_seconds_lost": human_warnings,
                "selection_report": report, "selection_explanation": voices.explain_selection_report(report),
                "audio_available": Path(audio).is_file(),
                "group_selection": {"group": group,
                                    "method": ("manual_override" if context["manual_override"] else
                                               "explicit_match" if context["override_requested"] else "transcript_or_vocab"),
                                    "binding_group": context["binding"].get("group")},
                "vocab": {"path": str(context["path"]) if context["path"] else None,
                          "group": group_vocab.get("group") if context["path"] else None,
                          "sha256": _voice_hash(context["path"]) if context["path"] else "absent"},
                "vocab_resolution": context["resolution"],
                "names_outside_vocab": outside_vocab,
                "source": source.describe(),
                "confirmation_replaced": confirmation_replaced,
                "other_variant_contributions": other_variants,
                "replacement_available": True,
                "note": "Только разметка; модель, качество голоса и карантин не проверялись. "
                        "Окончательный результат обновления зависит от модели и карантина."}
    except (OSError, ValueError, KeyError, TypeError, vocab.ResolutionError) as ex:
        raise core.CoreError(f"не удалось построить предварительный просмотр: {ex}") from ex


def _voice_hash(path):
    p = Path(path)
    if not p.is_file():
        return "absent"
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_voice_action(job_id, group_override=None):
    """Снимок предварительного плана для привязки подтверждения к конкретным файлам."""
    jp = _voice_job_source_ref(job_id)
    try:
        before = _open_voice_source(jp).token
        report = preview_voice_learning(job_id, group_override)
        group = report["group"]
        if paths.safe_group(group) != group:
            raise core.CoreError("недопустимое имя группы")
        with settings.group_voice_lock(group):
            profile = paths.voices_path(group)
            profile_hash = _voice_hash(profile)
            source = _open_voice_source(jp)
            audio = source.data.get("file")
            audio_hash = _voice_hash(audio) if audio else "absent"
            report = preview_voice_learning(job_id, group_override)
            vocab_path = report["vocab"]["path"]
            vocab_hash = report["vocab"]["sha256"]
            if (before != source.token or before != _open_voice_source(jp).token or report["group"] != group or
                    profile_hash != _voice_hash(profile) or
                    (vocab_path and _voice_hash(vocab_path) != vocab_hash) or
                    (audio and audio_hash != _voice_hash(audio))):
                raise core.CoreError("исходные файлы изменились; повторите предварительный просмотр")
        snapshot = {"job_id": job_id, "json": str(source.json_path.resolve()), "group": group,
                    "source_ref": source.ref, "source_kind": source.kind,
                    "group_override": group_override or None,
                    "manual_group_override": report["group_selection"]["method"] == "manual_override",
                    "vocab_path": vocab_path, "vocab_hash": vocab_hash,
                    "source_hash": before, "profile_hash": profile_hash, "audio_hash": audio_hash,
                    "report": report}
        lines = voice_action_lines(snapshot)
        if any(len(line) > VOICE_ACTION_LINE_LIMIT for line in lines) or len("\n".join(lines)) > VOICE_ACTION_TEXT_LIMIT:
            raise core.CoreError("список участников слишком длинный для безопасного подтверждения; действие не запущено")
        return snapshot
    except (OSError, ValueError) as ex:
        raise core.CoreError(f"не удалось подготовить подтверждение: {ex}") from ex


VOICE_ACTION_LINE_LIMIT = 200    # пределы окна подтверждения: confirm.py показывает текст целиком
VOICE_ACTION_TEXT_LIMIT = 1100


def _short(name, limit=60):
    name = str(name)
    return name if len(name) <= limit else name[:limit - 1] + "…"


def voice_action_lines(snapshot, revoke=False):
    report = snapshot["report"]
    candidates = report["candidates"]
    lines = [f"Группа: {snapshot['group']}", f"Расшифровка: {Path(snapshot['json']).name}"]
    source = report.get("source") or {}
    if source.get("kind") == "sqlite":
        line = f"Источник: запись в базе, ревизия {source.get('revision')}"
        if source.get("migrated_from"):
            line += f", перенесена из {_short(source['migrated_from'])}; её старый ключ в профиле снимается"
        if source.get("redirected_from"):
            line += "; найдена по старому номеру задачи"
        lines.append(line + ".")
    lines += ["Существующие вклады: " + (", ".join(report["existing_contributions"]) or "нет"),
              "Кандидаты по разметке: " + (", ".join(f"{n}: {v['seconds']} с" for n, v in candidates.items()) or "нет")]
    other_variants = report.get("other_variant_contributions") or []
    if other_variants:
        # вклад хранится по ключу аудио: отзыв снимает и вклад другой копии стенограммы того же аудио
        lines.append(("Снимет и вклад другого варианта стенограммы: " if revoke else
                      "Заменит вклад другого варианта стенограммы: ") +
                     ", ".join(_short(name) for name in other_variants) + ".")
    if report.get("confirmation_replaced") and not revoke:
        lines.append("Пометка «кем подтверждены голоса» заменится подсчётом секунд (прежняя — в истории): " +
                     ", ".join(report["confirmation_replaced"]) + ".")
    change = report.get("human_seconds_change") or {}
    if change:
        lines.append("Подтверждено человеком, с: " +
                     ", ".join(f"{n} {v['before']} → {v['after']}" for n, v in change.items()))
    if report.get("human_seconds_lost"):
        lines.append("ВНИМАНИЕ: замена обнулит подтверждённое человеком у: " + ", ".join(report["human_seconds_lost"]) +
                     ". Сначала перенесите старые подтверждения (legacy_speaker_review).")
    selection = report.get("group_selection") or {}
    if selection.get("method") == "manual_override":
        lines.append(f"Группа выбрана вручную: {snapshot['group']} (в расшифровке привязка отсутствует).")
    elif selection.get("method") == "explicit_match":
        lines.append(f"Группа выбрана явно и совпадает с привязкой: {snapshot['group']}.")
    if snapshot.get("vocab_path"):
        lines.append(f"Словарь группы: {Path(snapshot['vocab_path']).name} "
                     f"(SHA-256 {snapshot.get('vocab_hash', '')[:12]}…).")
    elif snapshot.get("group_override"):
        lines.append("Словарь выбранной группы отсутствует; обновление не запущено.")
    else:
        lines.append("Словарь группы не найден; ограничение участников словарём недоступно.")
    revoke_tail = ("Удалить перечисленные вклады этого аудио; другие записи не меняются." if other_variants else
                   "Удалить перечисленные вклады источника; другие источники не меняются.")
    tail = [(revoke_tail if revoke else
             "Новый отпечаток может заменить вклад или попасть в карантин; без подходящего отпечатка вклад сохраняется."),
            "Будет создана резервная копия профиля. Подтвердите только после проверки."]
    explanation = report.get("selection_explanation") or []
    detail = (["Предварительный отбор (результат модели может отличаться):"] +
              [f"{item['description']}: {item['count']}" for item in explanation]) if explanation else []
    if detail and len("\n".join(lines + detail + tail)) > VOICE_ACTION_TEXT_LIMIT:
        # обязательные строки важнее подробностей отбора: подробности остаются в предпросмотре
        detail = [f"Предварительный отбор: причин {len(explanation)}, подробно — в предпросмотре "
                  "(результат модели может отличаться)."]
    return lines + detail + tail


def backup_voice_profile(path):
    """Сохранить точную копию биометрического файла до изменения, не заменяя старые копии."""
    import voices
    return voices.backup_profile(path)


def backup_voice_profiles(group=""):
    """Сделать отдельную резервную копию профиля одной группы или всех групп.

    Копии лежат в `voice-backups` внутри группы, не шифруются, не входят в экспорт и
    автоматически не удаляются. Векторы и имена не возвращаются.
    """
    import voices
    if group:
        if group not in paths.list_groups():
            raise core.CoreError("группа не найдена")
        groups = [group]
    else:
        groups = paths.list_groups()
    backups = []
    for name in groups:
        path = paths.voices_path(name)
        try:
            with settings.group_voice_lock(name):
                target = backup_voice_profile(path)
        except (OSError, ValueError) as ex:
            raise core.CoreError(f"не удалось сохранить резерв группы «{name}»: {ex}") from ex
        if target:
            backups.append({"group": name, "backup": target})
    return {"backups": backups, "count": len(backups),
            "note": "Резерв — биометрия в открытом виде; храните и удаляйте по решению владельца."}


def voice_quality(group=""):
    """Политика качества и сохранённые измерения по профилям; аудио не декодируется."""
    import voices
    if group and group not in paths.list_groups():
        raise core.CoreError("группа не найдена")
    groups = [group] if group else paths.list_groups()
    out = []
    for name in groups:
        try:
            policy = settings.group_voice_settings(name)
        except ValueError as ex:
            out.append({"group": name, "settings_error": str(ex)})
            continue
        entry = {"group": name,
                 "audio_quality_policy": policy.get("audio_quality_policy", "warn"),
                 "policy_source": policy.get("audio_quality_policy_source", "default")}
        path = paths.voices_path(name)
        if not path.exists():
            entry.update({"profile": "absent", "measurements": [], "summary": {"measured_sources": 0}})
            out.append(entry)
            continue
        try:
            data = voices.load_profile(path, convert=False)
        except voices.VoiceProfileError as ex:
            entry.update({"profile_error": str(ex)})
            out.append(entry)
            continue
        measurements = []
        for source, q in (data.get("audio_quality") or {}).items():
            if not isinstance(q, dict):
                continue
            measurements.append({
                "recording": Path(source).name, "from": q.get("from"),
                "at": q.get("at"), "available": q.get("available"),
                "duration_sec": q.get("duration_sec"), "peak_dbfs": q.get("peak_dbfs"),
                "rms_dbfs": q.get("rms_dbfs"), "clipping_ratio": q.get("clipping_ratio"),
                "silence_ratio": q.get("silence_ratio"), "snr_db_estimate": q.get("snr_db_estimate"),
                "warnings": q.get("warnings") or [],
                "warning_notes": voices.explain_audio_quality(q)})
        entry.update({
            "profile": "ok", "measurements": measurements[-50:],
            "summary": {"measured_sources": len(measurements),
                        "clipping_sources": sum(1 for m in measurements if "clipping" in (m["warnings"] or [])),
                        "warning_sources": sum(1 for m in measurements if m["warnings"])},
            "recent_quality_warnings": [w for w in (data.get("warnings") or [])
                                        if w.get("kind") == "audio_quality"][-5:]})
        out.append(entry)
    return {"groups": out,
            "note": "Измерения приблизительны. Политика warn только предупреждает; "
                    "reject_clipping не учитывает источник с клиппингом. Тихая речь не отбраковывается."}


def set_voice_quality_policy(policy, group=""):
    """Политика качества: warn (по умолчанию) или reject_clipping; без group — для новых групп."""
    import voices
    try:
        policy = voices.validate_quality_policy(policy)
    except ValueError as ex:
        raise core.CoreError(str(ex)) from ex
    if not group:
        settings.update(explicit=("voice_quality_policy_default",), voice_quality_policy_default=policy)
        return {"default_for_new_groups": policy,
                "note": "Группы без явной политики наследуют это значение; уже выбранная политика группы не меняется."}
    if group not in paths.list_groups():
        raise core.CoreError("группа не найдена")
    try:
        with settings.group_voice_lock(group):
            current = settings.group_voice_settings(group)
            data = {k: v for k, v in current.items() if k != "audio_quality_policy_source"}
            data["audio_quality_policy"] = policy
            saved = settings.save_group_voice_settings(group, data)
    except (ValueError, OSError) as ex:
        raise core.CoreError(str(ex)) from ex
    return {"group": group, "audio_quality_policy": saved.get("audio_quality_policy", policy),
            "auto_learn": saved.get("auto_learn")}


REFRESH_MEANINGS = {
    "unchanged": "Источник учтён, отпечаток совпадает.",
    "changed": "Отпечаток стенограммы или метаданные аудио отличаются — нужен просмотр и подтверждённое обновление.",
    "legacy_baseline_pending": "Отпечатка нет; при следующем обучении текущее состояние будет принято как исходное.",
    "not_recorded": "Источник есть в профиле, но без записи происхождения.",
    "transcript_missing": "Файл стенограммы не найден.",
    "transcript_unreadable": "Стенограмма не читается.",
    "audio_missing": "Исходное аудио не найдено или не указано.",
    "recording_missing": "Запись в базе не найдена или база недоступна.",
}


def _voice_job_map():
    """{путь стенограммы: id задачи} по готовым задачам; нужен только для подсказки обновления."""
    out = {}
    try:
        files = sorted(paths.JOBS.glob("*.json"))
    except OSError:
        return out
    for p in files:
        try:
            status = json.loads(p.read_text(encoding="utf-8"))
            jp = status.get("json")
            if status.get("state") == "done" and jp:
                out[str(Path(jp).resolve())] = p.stem
        except (OSError, ValueError, TypeError):
            continue
    return out


def voice_refresh_plan(group=""):
    """Что изменилось после прежнего обучения; только чтение, без моделей и записи профиля."""
    import voices
    if group and group not in paths.list_groups():
        raise core.CoreError("группа не найдена")
    groups = [group] if group else paths.list_groups()
    jobs = _voice_job_map()
    try:
        sqlite_mode = settings.recording_backend() == "sqlite"
    except (ValueError, OSError):
        sqlite_mode = False
    store = _refresh_store() if sqlite_mode else None
    try:
        return _voice_refresh_groups(groups, jobs, store, sqlite_mode)
    finally:
        if store is not None:
            store.close()


def _refresh_store():
    """База только на чтение (только при хранении в базе); недоступна — None."""
    import voice_sources
    try:
        return voice_sources._open_store_readonly()
    except Exception:   # noqa: BLE001 — план только показывает: без базы записи получают recording_missing
        return None


def _refresh_entry(transcript=None, job_id=None, status="transcript_missing"):
    return {"transcript": transcript, "recording": None, "job_id": job_id,
            "status": status, "needs_update": True,
            "transcript_changed": None, "audio_changed": None,
            "comparison_basis": "unavailable", "legacy_mtime_changed": None,
            "needs_review_reasons": [], "migrated_to": None}


def _refresh_recording_entry(store, key, stored, profile):
    """Строка плана для ключа recording:<id>: сравнение с текущей ревизией записи в базе."""
    import voices
    import voice_sources
    entry = _refresh_entry(job_id=key, status="recording_missing")
    if store is None:
        return entry, None
    try:
        source = voice_sources.record_source(store, key[len(voice_sources.RECORDING_PREFIX):])
    except Exception:   # noqa: BLE001 — запись удалена или не читается: план это показывает, а не падает
        return entry, None
    entry["transcript"] = source.name
    audio_path = source.data.get("file")
    if not audio_path:
        entry["status"] = "audio_missing"
        return entry, None
    audio_key = voice_sources.existing_audio_key(profile, str(Path(audio_path).resolve()))
    entry["recording"] = Path(audio_path).name
    if not Path(audio_path).is_file():
        entry["status"] = "audio_missing"
        return entry, audio_key
    current = source.fingerprint(audio_path)
    status = voices.source_fingerprint_status(stored, current, 0.0)
    entry["status"] = {"unchanged": "unchanged", "absent": "not_recorded"}.get(status, "changed")
    if isinstance(stored, dict):
        entry["comparison_basis"] = "fingerprint"
        transcript_keys = ("store_id", "recording_id", "transcript_revision", "transcript_payload_hash")
        entry["transcript_changed"] = any(stored.get(k) != current[k] for k in transcript_keys)
        entry["audio_changed"] = any(stored.get(k) != current[k] for k in ("audio_size", "audio_mtime"))
    return entry, audio_key


def _refresh_migrated_copy(store, key, stored, profile, entry, audio_key):
    """Старый ключ при работе с базой: статус — по её единственной копии (связь — quality.migration.source)."""
    import recordings
    import voice_sources
    try:
        copies = recordings.migrated_copies(store, key)
        if len(copies) != 1:
            return audio_key
        source = voice_sources.record_source(store, copies[0][0])
    except Exception:   # noqa: BLE001 — копия не читается: остаётся статус по старому файлу
        return audio_key
    entry["migrated_to"] = source.name
    entry["job_id"] = source.ref
    entry["status"] = voice_sources.linked_status(stored, source)
    entry["comparison_basis"] = "migrated_copy"
    entry["transcript_changed"] = entry["audio_changed"] = entry["legacy_mtime_changed"] = None
    audio_path = source.data.get("file")
    if audio_path:
        entry["recording"] = Path(audio_path).name
        audio_key = voice_sources.existing_audio_key(profile, str(Path(audio_path).resolve()))
    return audio_key


def _voice_refresh_groups(groups, jobs, store, sqlite_mode):
    import voices
    import voice_sources
    out = []
    for name in groups:
        path = paths.voices_path(name)
        if not path.exists():
            out.append({"group": name, "profile": "absent", "sources": [], "summary": {"needs_update": 0}})
            continue
        try:
            data = voices.load_profile(path, convert=False)
        except voices.VoiceProfileError as ex:
            out.append({"group": name, "profile_error": str(ex)})
            continue
        entries = []
        for key, stored in sorted((data.get("sources") or {}).items()):
            if sqlite_mode and key.startswith(voice_sources.RECORDING_PREFIX):
                entry, audio_key = _refresh_recording_entry(store, key, stored, data)
            else:
                jp = Path(key)
                audio_path, audio_key = None, None
                entry = _refresh_entry(jp.name, jobs.get(str(jp.resolve())))
                if jp.is_file():
                    try:
                        source_data = json.loads(jp.read_text(encoding="utf-8"))
                        audio_path = source_data.get("file")
                    except (OSError, ValueError, TypeError):
                        entry["status"] = "transcript_unreadable"
                    else:
                        if not audio_path:
                            entry["status"] = "audio_missing"
                        else:
                            audio_key = str(Path(audio_path).resolve())
                            entry["recording"] = Path(audio_path).name
                            if not Path(audio_path).is_file():
                                entry["status"] = "audio_missing"
                            else:
                                current = voices.source_fingerprint(jp, audio_path)
                                status = voices.source_fingerprint_status(stored, current, jp.stat().st_mtime)
                                entry["status"] = {"unchanged": "unchanged", "changed": "changed",
                                                   "legacy_unchanged": "legacy_baseline_pending",
                                                   "legacy_changed": "changed",
                                                   "absent": "not_recorded"}.get(status, "changed")
                                if status in ("legacy_changed", "legacy_unchanged"):
                                    entry["comparison_basis"] = "legacy_mtime"
                                    entry["legacy_mtime_changed"] = status == "legacy_changed"
                                elif isinstance(stored, dict):
                                    entry["comparison_basis"] = "fingerprint"
                                    transcript_keys = ("transcript_sha256", "transcript_size")
                                    audio_keys = ("audio_size", "audio_mtime")
                                    if all(stored.get(k) is not None for k in transcript_keys):
                                        entry["transcript_changed"] = any(stored[k] != current[k]
                                                                          for k in transcript_keys)
                                    if all(stored.get(k) is not None for k in audio_keys):
                                        entry["audio_changed"] = any(stored[k] != current[k] for k in audio_keys)
                if sqlite_mode and store is not None:
                    audio_key = _refresh_migrated_copy(store, key, stored, data, entry, audio_key)
            versions = []
            if audio_key:
                for speaker in (data.get("speakers") or {}).values():
                    part = (speaker.get("parts") or {}).get(audio_key)
                    if part:
                        versions.append(part.get("selection_version"))
            current_selection = bool(versions) and all(v == voices.SELECTION_VERSION for v in versions)
            entry["selection_version_current"] = current_selection
            if entry["status"] == "changed":
                if entry["comparison_basis"] == "migrated_copy":
                    entry["needs_review_reasons"].append("запись в базе правили после переноса или старый файл "
                                                         "отличается от перенесённого")
                elif entry["comparison_basis"] == "legacy_mtime":
                    entry["needs_review_reasons"].append("время стенограммы отличается от сохранённого старого значения")
                elif entry["comparison_basis"] == "unavailable":
                    entry["needs_review_reasons"].append("прежние данные источника не позволяют выполнить сравнение")
                else:
                    entry["needs_review_reasons"].append("отпечаток стенограммы или метаданные аудио отличаются")
            if versions and not current_selection:
                entry["needs_review_reasons"].append("вклад без текущей версии отбора")
            if entry["status"] in ("transcript_missing", "transcript_unreadable", "audio_missing",
                                   "recording_missing"):
                entry["needs_review_reasons"].append(REFRESH_MEANINGS[entry["status"]])
            if entry["status"] == "not_recorded":
                entry["needs_review_reasons"].append("нет отпечатка источника")
            if entry["status"] == "legacy_baseline_pending":
                entry["needs_review_reasons"].append("нет постоянного отпечатка источника")
            if audio_key and audio_key in (data.get("audio_quality") or {}):
                q = data["audio_quality"][audio_key]
                entry["audio_quality_warnings"] = (q or {}).get("warnings") or []
                entry["audio_quality_notes"] = voices.explain_audio_quality(q)
                if entry["audio_quality_warnings"]:
                    entry["needs_review_reasons"].append("есть предупреждения качества аудио")
            entry["needs_update"] = bool(entry["needs_review_reasons"])
            entry["meaning"] = REFRESH_MEANINGS.get(entry["status"], entry["status"])
            if entry["status"] == "changed" and entry["comparison_basis"] == "legacy_mtime":
                entry["meaning"] = ("Время стенограммы отличается от старого значения; сравнить содержимое "
                                    "стенограммы и прежнее аудио невозможно. Нужен просмотр.")
            elif entry["status"] == "changed" and entry["comparison_basis"] == "unavailable":
                entry["meaning"] = "Прежние данные источника не позволяют выполнить сравнение; нужен просмотр."
            elif (sqlite_mode and entry["comparison_basis"] == "legacy_mtime"
                  and entry["status"] == "legacy_baseline_pending"):
                entry["meaning"] = ("Отпечатка нет; при работе с базой старый файл в обучение не идёт: нужна его "
                                    "единственная копия в базе и подтверждённое обновление.")
            elif entry["comparison_basis"] == "migrated_copy":
                entry["meaning"] = {
                    "unchanged": "Источник учтён; его запись в базе после переноса не менялась.",
                    "legacy_baseline_pending": "Отпечатка нет; старый файл не менялся после переноса. Исходное "
                                               "состояние примет подтверждённое обновление записи в базе.",
                }.get(entry["status"], "Запись в базе правили после переноса или старый файл изменился — нужен "
                                       "просмотр и подтверждённое обновление.")
            entries.append(entry)
        summary = {}
        for item in entries:
            summary[item["status"]] = summary.get(item["status"], 0) + 1
        summary["needs_update"] = sum(1 for item in entries if item["needs_update"])
        out.append({"group": name, "profile": "ok", "revision": data.get("revision"),
                    "sources": entries, "summary": summary})
    return {"groups": out,
            "needs_update": sum(item.get("summary", {}).get("needs_update", 0) for item in out),
            "note": "План только показывает различия. Автоматическая замена вкладов не выполняется; "
                    "обновляйте через preview_voice_profiles и update_voice_profiles отдельной задачи."}


def _snapshot_source(snapshot):
    """Источник подтверждённого снимка, открытый заново (запись в базе — по recording:<id>)."""
    return _open_voice_source(snapshot.get("source_ref") or snapshot["json"])


def start_voice_update(snapshot):
    group = snapshot["group"]
    with settings.group_voice_lock(group):
        source = _snapshot_source(snapshot)
        if source.token != snapshot["source_hash"] or _voice_hash(paths.voices_path(group)) != snapshot["profile_hash"]:
            raise core.CoreError("стенограмма или профиль изменились; повторите просмотр и подтверждение")
        data = source.data
        audio = data.get("file")
        if not audio or not Path(audio).is_file():
            raise core.CoreError("аудиофайл недоступен")
        if _voice_hash(audio) != snapshot["audio_hash"]:
            raise core.CoreError("аудиофайл изменился; повторите просмотр и подтверждение")
        current = _voice_vocab_context(data, source.json_path, snapshot.get("group_override"))
        current_path = str(current["path"]) if current["path"] else None
        current_vocab_hash = _voice_hash(current["path"]) if current["path"] else "absent"
        if (current["group"] != group or current_path != snapshot.get("vocab_path") or
                current_vocab_hash != snapshot.get("vocab_hash")):
            raise core.CoreError("словарь или привязка группы изменились; повторите просмотр и подтверждение")
        learn_ref = source.ref if source.kind == "sqlite" else str(source.json_path)
        args = ["--learn-only", learn_ref, "--voice-source-hash", snapshot["source_hash"],
                "--voice-profile-hash", snapshot["profile_hash"], "--voice-audio-hash", snapshot["audio_hash"],
                "--voice-group", group, "--voice-vocab-path", snapshot.get("vocab_path") or "",
                "--voice-vocab-hash", snapshot.get("vocab_hash") or "absent"]
        if snapshot.get("group_override"):
            args.extend(["--voice-group-override", snapshot["group_override"]])
        return {"job_id": core.launch("worker.py", args, audio, "learn"),
                "hint": "Ход — transcription_status(job_id)"}


def revoke_voice_source(snapshot):
    """Отозвать вклад источника после подтверждения; другие записи не затрагиваются."""
    group = snapshot["group"]
    path = paths.voices_path(group)
    with settings.group_voice_lock(group):
        source = _snapshot_source(snapshot)
        if source.token != snapshot["source_hash"] or _voice_hash(path) != snapshot["profile_hash"]:
            raise core.CoreError("стенограмма или профиль изменились; повторите предварительный просмотр и подтверждение")
        if not path.exists():
            raise core.CoreError("профиль отсутствует")
        data = source.data
        key = str(Path(data["file"]).resolve())
        import voices
        import voice_sources
        try:
            profile = voices.load_profile(path, convert=False)
        except voices.VoiceProfileError as ex:
            raise core.CoreError(f"голосовой профиль группы несовместим или повреждён: {ex}. "
                            "Отзыв не выполнен; перенесите профиль отдельно.") from ex
        if source.kind == "sqlite":
            key = voice_sources.existing_audio_key(profile, key)
        names = sorted(n for n, s in profile.get("speakers", {}).items() if key in s.get("parts", {}))
        if not names or names != snapshot["report"]["existing_contributions"]:
            raise core.CoreError("вклады изменились или отсутствуют; повторите просмотр")
        backup = backup_voice_profile(path)
        for name in names:
            profile["speakers"][name]["parts"].pop(key)
        profile.get("confusions", {}).pop(key, None)
        source_keys = [source.key] + [k for old in source.legacy_keys
                                      for k in voice_sources.matching_keys(profile.get("sources"), old)]
        for source_key in source_keys:
            profile.get("sources", {}).pop(source_key, None)
            profile.get("source_selection_versions", {}).pop(source_key, None)
        profile["revision"] = int(profile.get("revision", 0)) + 1
        entry = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "from": source.name, "kind": "source_revoked",
                 "speakers": names}
        if source.kind == "sqlite":
            entry["source"] = source.key
        profile.setdefault("history", []).append(entry)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            json.dump(profile, f, ensure_ascii=False)
            tmp = f.name
        try:
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return {"group": group, "revoked": names, "revision": profile["revision"], "backup": backup}


def set_voice_auto_learn(enabled, group=""):
    if type(enabled) is not bool:
        raise core.CoreError("значение должно быть true или false")
    if not group:
        try:
            for existing in paths.list_groups():
                with settings.group_voice_lock(existing):
                    settings.group_voice_settings(existing, initialize=True)
        except (ValueError, OSError) as ex:
            raise core.CoreError(f"не удалось сохранить настройки существующих групп: {ex}") from ex
        settings.update(explicit=("voice_auto_learn_default",), voice_auto_learn_default=enabled)
        return {"default_for_new_groups": enabled}
    if group not in paths.list_groups():
        raise core.CoreError("группа не найдена")
    try:
        with settings.group_voice_lock(group):
            current = settings.group_voice_settings(group)
            data = {k: v for k, v in current.items() if k != "audio_quality_policy_source"}
            data.update(auto_learn=enabled, reason="manual")
            return {"group": group, **settings.save_group_voice_settings(group, data)}
    except (ValueError, OSError) as ex:
        raise core.CoreError(str(ex)) from ex


def voice_profiles():
    """Сводка по голосовым профилям групп; сами отпечатки не выводятся."""
    import voices
    out = []
    for p in sorted(paths.GROUPS.glob("*/voices.json")):
        try:
            d = voices.load_profile(p, convert=False)
        except voices.VoiceProfileError as ex:
            out.append({"group": p.parent.name, "error": str(ex), "profile_error": str(ex)})
            continue
        conf = {}
        for pairs in (d.get("confusions") or {}).values():
            for k, v in pairs.items():
                conf[k.replace("|", " → ")] = conf.get(k.replace("|", " → "), 0) + v
        try:
            policy = settings.group_voice_settings(p.parent.name)
        except ValueError:
            policy = {"auto_learn": False, "reason": "settings_error"}
        vocab_file = p.parent / "vocab.json"
        try:
            vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
            expected = [x["name"] for x in vocab.get("participants", []) if x.get("name")]
        except (OSError, ValueError, TypeError):
            expected = []
        out.append({"group": p.parent.name, "revision": d.get("revision"),
                    "auto_learn": policy, "sufficiency": voices.profile_sufficiency(d, expected),
                    "weight_diagnostics": voices.voice_weight_diagnostics(d),
                    "audio_quality_policy": policy.get("audio_quality_policy", "warn"),
                    "audio_quality_policy_source": policy.get("audio_quality_policy_source", "default"),
                    "speakers": {n: _speaker_voice_summary(s) for n, s in d.get("speakers", {}).items()},
                    "corrected_pairs": conf,
                    "warnings": (d.get("warnings") or [])[-5:],
                    "quality_summary": {"measured_sources": len(d.get("audio_quality") or {}),
                                        "clipping_sources": sum(
                                            1 for q in (d.get("audio_quality") or {}).values()
                                            if "clipping" in ((q or {}).get("warnings") or []))},
                    "quarantine": [{"name": n, "recording": Path(a).name, **{k: v for k, v in q.items() if k != "json"}}
                                   for a, qa in (d.get("quarantine") or {}).items() for n, q in qa.items()],
                    "identity": {Path(a).name: v for a, v in (d.get("identity") or {}).items()},
                    "sources": list(d.get("sources", {}))})
    return {"profiles": out, "groups": paths.list_groups(),
            "default_group": settings.load_config().get("default_group")}


def _speaker_voice_summary(speaker):
    """Длительность и вес раздельно: старый `sec` не подменяет неизмеренную длительность."""
    seconds = weighted = human = 0.0
    measured = unmeasured = 0
    for part in (speaker.get("parts") or {}).values():
        sec = part.get("sec")
        if isinstance(sec, (int, float)) and not isinstance(sec, bool) and sec >= 0:
            weighted += float(sec)
        a, h = part.get("auto_sec"), part.get("human_sec")
        if (isinstance(a, (int, float)) and not isinstance(a, bool)
                and isinstance(h, (int, float)) and not isinstance(h, bool) and a >= 0 and h >= 0):
            seconds += float(a) + float(h)
            human += float(h)
            measured += 1
        else:
            unmeasured += 1
    return {"records": len(speaker.get("parts") or {}),
            "seconds": round(seconds, 1),
            "weighted_seconds": round(weighted, 1),
            "human_seconds": round(human, 1),
            "measured_records": measured,
            "unmeasured_records": unmeasured}


import core  # noqa: E402  (последним: core.py в конце загружает этот модуль)
