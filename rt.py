"""Командная строка rus-transcriber: те же возможности, что у MCP-сервера, без Claude и без любого MCP-клиента.

    python rt.py transcribe запись.mp3 --names "Анна, Борис" --group моя-группа
    python rt.py status ЗАДАЧА --wait 50        # ждёт смены этапа или конца (до 50 с)
    python rt.py jobs [--limit 10]
    python rt.py read ЗАДАЧА [--fmt md|json] [--extra] [--offset 0]
    python rt.py review ЗАДАЧА [--offset 0 --limit 40] [--reasons voice,gender] [--min-sec 1] [--brief] [--status open]
    python rt.py hotspots ЗАДАЧА [--limit 20] [--min-score 0.5]  # горячие места разговора (эвристика, по запросу)
    python rt.py edit-span ЗАДАЧА T1 T2 "текст" [--speaker Имя] [--note ...] [--by human]
    python rt.py reopen ЗАДАЧА r57,r88
    python rt.py cancel ЗАДАЧА
    python rt.py settings | profiles | audit    # audit: абсолютные пути в данных
    python rt.py voice-quality [--group ИМЯ]  # политика и измерения качества аудио
    python rt.py voice-refresh-plan [--group ИМЯ]  # изменённые источники, только чтение
    python rt.py voice-quality-policy warn|reject_clipping [--group ИМЯ]
    python rt.py voice-backup [--group ИМЯ]    # открытый резерв профилей в voice-backups
    python rt.py voice-preview ЗАДАЧА [--group ИМЯ]  # --group только при неизвестной группе
    python rt.py voice-update ЗАДАЧА [--group ИМЯ]   # перед запуском спрашивает подтверждение
    python rt.py screen [--port 8766]          # локальный экран П5 в браузере (запуск, состояние, резерв)
    python rt.py workflow-list [--limit 20]    # последние запросы workflow
    python rt.py db-backup НОВЫЙ-ФАЙЛ.sqlite3  # проверенная неактивная копия базы состояния
    python rt.py db-restore РЕЗЕРВ.sqlite3 [--dest ПУТЬ] [--root ID=ПУТЬ]
        # мастер восстановления: новые корни и установка, документы формируются заново,
        # прежняя база сохраняется рядом; спрашивает подтверждение на клавиатуре
    python rt.py self-check [--no-tests]       # тесты, проверка файлов на личные данные, морфология

    Данные (опасные действия: спрашивают согласие на клавиатуре, только в настоящем терминале):
    python rt.py export [--group g] [--dest папка] [--with-jobs]
    python rt.py import архив.zip [--replace] [--apply]       # без --apply только отчёт
    python rt.py delete-group имя [--permanent]
    python rt.py cleanup [--days 30] [--apply]
    python rt.py migrate [--apply]
    python rt.py approve                                       # разрешить запрос, оставленный MCP-сервером

Код возврата: 0 — готово, 1 — ошибка, 2 — отказ человека или нет терминала для подтверждения.
Вывод — JSON (удобно для скриптов); сообщения для человека идут в stderr.
"""
import argparse
import json
import sys
from pathlib import Path

import core
import confirm
import paths


def out(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=1))


def say(msg):
    print(msg, file=sys.stderr)


def human_ok(lines, must_type=None):
    """Согласие человека на клавиатуре. Без настоящего терминала (скрипт, другой процесс, модель с оболочкой)
    подтверждения нет — действие не выполняется."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        say("Нужен настоящий терминал: подтвердить действие может только человек на клавиатуре.")
        return False
    say("\n".join(lines))
    try:
        if must_type:
            return input(f"Для подтверждения введите «{must_type}»: ").strip() == must_type
        return input("Выполнить? [да/нет]: ").strip().lower() in ("да", "д", "y", "yes")
    except EOFError:
        return False


def log(action, ok):
    confirm._log(action, "разрешено" if ok else "отказ", "cli")


def gated(action, lines, run, must_type=None):
    ok = human_ok(lines, must_type)
    log(action, ok)
    if not ok:
        say("Не выполнено.")
        return 2
    out(run())
    return 0


def ask(prompt):
    """Ответ человека в настоящем терминале; без него восстановление не продолжается."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise core.CoreError("мастеру восстановления нужен настоящий терминал")
    try:
        return input(prompt).strip()
    except EOFError:
        raise core.CoreError("ввод прерван; восстановление не выполнено")


def cmd_db_restore(a):
    """Мастер: резерв → новые корни и установка → проверка → файлы → замена базы."""
    info = core.inspect_backup(a.backup)
    say(f"Резерв: {info['path']}; записей: {info['recordings']}; "
        f"корней: {len(info['roots'])}; пользователей: {len(info['users'])}.")
    if not info["roots"]:
        raise core.CoreError("в резерве нет документных корней; восстанавливать нечего")
    if not info["restore_guard_locked"]:
        raise core.CoreError("это не подготовленный резерв (копия не помечена как неактивная)")
    mapping = {}
    for item in a.root:
        root_id, _, path = item.partition("=")
        if not root_id.strip() or not path.strip():
            raise core.CoreError("формат --root: ID_корня=ПУТЬ")
        mapping[root_id.strip()] = path.strip()
    for root in info["roots"]:
        if root["root_id"] in mapping:
            continue
        default = root["path"] + "-restored"
        answer = ask(f"Новый путь для корня «{root['path']}» (Enter — {default}): ")
        mapping[root["root_id"]] = answer or default
    for root_id, path in mapping.items():
        if not Path(path).is_dir():
            if ask(f"Папка «{path}» не существует; создать для восстановленных документов? "
                   f"[да/нет]: ").lower() in ("да", "д", "y", "yes"):
                Path(path).mkdir(parents=True, exist_ok=True)
            else:
                raise core.CoreError("пути документных корней должны существовать")
    users = info["users"]
    if len(users) == 1:
        user_id = users[0]["user_id"]
    else:
        for index, user in enumerate(users, 1):
            say(f"{index}. {user['display_name'] or '(без имени)'}")
        choice = int(ask("Кому принадлежит восстановленная база? Введите номер: ")) - 1
        if not 0 <= choice < len(users):
            raise core.CoreError("некорректный номер пользователя")
        user_id = users[choice]["user_id"]
    destination = Path(a.dest) if a.dest else paths.STATE_DB
    lines = [f"Резерв: {a.backup}", f"Восстановленная база: {destination}",
             f"Корни документов: " + "; ".join(f"{k} → {v}" for k, v in mapping.items()),
             "Документы будут сформированы в новых корнях из сохранённых байтов.",
             "Прежняя база (если есть) будет сохранена рядом, из резерва ничего не удаляется."]
    return gated("Восстановление базы из резерва", lines,
                 lambda: core.restore_from_backup(a.backup, destination, mapping, user_id))


def cmd_approve(a):
    reqs = confirm.pending_requests()
    if not reqs:
        say("Нет запросов на подтверждение.")
        return 0
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        say("Нужен настоящий терминал: подтвердить действие может только человек на клавиатуре.")
        return 2
    for key, r in reqs.items():
        say(f"\n--- {r['action']} ---\n{r['text']}")
        ans = input("Разрешить? [да/нет]: ").strip().lower()
        if ans in ("да", "д", "y", "yes"):
            confirm.approve(key)
            confirm._log(r["action"], "разрешено", "terminal")
            say("Разрешено на 5 минут: повторите вызов в клиенте с теми же параметрами.")
        else:
            confirm.reject(key)
            confirm._log(r["action"], "отказ", "terminal")
            say("Отказано.")
    return 0


def build():
    ap = argparse.ArgumentParser(prog="rt.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcribe"); t.add_argument("audio")
    t.add_argument("--names", default=""); t.add_argument("--group", default=""); t.add_argument("--speakers", type=int, default=0)
    t.add_argument("--no-emotions", action="store_true"); t.add_argument("--no-voices", action="store_true")
    t.add_argument("--tag", default=""); t.add_argument("--vocab", default=""); t.add_argument("--whisper", default="")
    wf = sub.add_parser("workflow-start"); wf.add_argument("audio")
    wf.add_argument("--conspect", action="store_true")
    wf.add_argument("--cloud-model", action="store_true",
                    help="согласие отправить в облачную модель OpenAI для конспекта текст стенограммы: слова, "
                         "деление на реплики и условные метки говорящих («Участник 1»…), без имён")
    wf.add_argument("--no-cloud-model", action="store_true",
                    help="облако запрещено (так и по умолчанию; ключ оставлен для совместимости)")
    wf.add_argument("--named-speakers", action="store_true")
    wf.add_argument("--human-verified", action="store_true")
    wf.add_argument("--participation", choices=["simple", "advanced", "developer"], default="simple")
    wf.add_argument("--goal", default=""); wf.add_argument("--question-budget", type=int, default=0)
    ws = sub.add_parser("workflow-status"); ws.add_argument("request")
    wt = sub.add_parser("workflow-task"); wt.add_argument("request")
    wt.add_argument("--offset", type=int, default=0); wt.add_argument("--limit", type=int, default=1000)
    wr = sub.add_parser("workflow-result"); wr.add_argument("request"); wr.add_argument("result_file")
    wmp = sub.add_parser("workflow-part"); wmp.add_argument("request"); wmp.add_argument("index", type=int)
    wmr = sub.add_parser("workflow-part-result"); wmr.add_argument("request"); wmr.add_argument("index", type=int)
    wmr.add_argument("result_file")
    we = sub.add_parser("workflow-execute"); we.add_argument("request")
    we.add_argument("--retry-failed", action="store_true")
    wp = sub.add_parser("workflow-policy"); wp.add_argument("request"); wp.add_argument("policy_file")
    wp.add_argument("expected_revision", type=int)
    wv = sub.add_parser("workflow-confirm"); wv.add_argument("request")
    wa = sub.add_parser("workflow-abandon"); wa.add_argument("request")
    wq = sub.add_parser("workflow-questions"); wq.add_argument("request")
    wl = sub.add_parser("workflow-list"); wl.add_argument("--limit", type=int, default=20)
    sc = sub.add_parser("screen"); sc.add_argument("--port", type=int, default=8766)
    sc.add_argument("--hours", type=float, default=6.0)
    s = sub.add_parser("status"); s.add_argument("job"); s.add_argument("--wait", type=int, default=0)
    j = sub.add_parser("jobs"); j.add_argument("--limit", type=int, default=10)
    r = sub.add_parser("read"); r.add_argument("job"); r.add_argument("--fmt", default="md", choices=["md", "json"])
    r.add_argument("--extra", action="store_true"); r.add_argument("--offset", type=int, default=0)
    r.add_argument("--max-chars", type=int, default=60000)
    v = sub.add_parser("review"); v.add_argument("job"); v.add_argument("--offset", type=int, default=0); v.add_argument("--limit", type=int, default=40)
    v.add_argument("--reasons", default=""); v.add_argument("--min-sec", type=float, default=0.0)
    v.add_argument("--brief", action="store_true"); v.add_argument("--status", default="open")
    hs = sub.add_parser("hotspots"); hs.add_argument("job"); hs.add_argument("--limit", type=int, default=20)
    hs.add_argument("--min-score", type=float, default=0.0)
    es = sub.add_parser("edit-span"); es.add_argument("job"); es.add_argument("t1", type=float); es.add_argument("t2", type=float)
    es.add_argument("text"); es.add_argument("--speaker", default=""); es.add_argument("--note", default="")
    es.add_argument("--by", default="human", choices=["human", "claude"])
    rb = sub.add_parser("review-build"); rb.add_argument("job"); rb.add_argument("--items", help="JSON-файл с вопросами; без него — нейтральные карточки")
    rp = sub.add_parser("review-page"); rp.add_argument("job"); rp.add_argument("--port", type=int, default=8765)
    ra = sub.add_parser("review-apply"); ra.add_argument("job")
    cc = sub.add_parser("conspect-confirm"); cc.add_argument("job"); cc.add_argument("--name", default="конспект")
    cp = sub.add_parser("conspect-candidates"); cp.add_argument("job"); cp.add_argument("text_file")
    cs = sub.add_parser("conspect-select"); cs.add_argument("job"); cs.add_argument("text_file"); cs.add_argument("selection_file"); cs.add_argument("--name", default="конспект")
    ro = sub.add_parser("reopen"); ro.add_argument("job"); ro.add_argument("ids")
    c = sub.add_parser("cancel"); c.add_argument("job")
    sub.add_parser("settings"); sub.add_parser("profiles"); sub.add_parser("approve"); sub.add_parser("audit")
    vl = sub.add_parser("voice-auto-learn"); vl.add_argument("state", choices=["on", "off"]); vl.add_argument("--group", default="")
    vp = sub.add_parser("voice-preview"); vp.add_argument("job")
    vp.add_argument("--group", default="", help="явный выбор только для расшифровки без привязки")
    vu = sub.add_parser("voice-update"); vu.add_argument("job")
    vu.add_argument("--group", default="", help="явный выбор только для расшифровки без привязки")
    vr = sub.add_parser("voice-revoke"); vr.add_argument("job")
    vq = sub.add_parser("voice-quality"); vq.add_argument("--group", default="")
    vqp = sub.add_parser("voice-quality-policy"); vqp.add_argument("policy", choices=["warn", "reject_clipping"])
    vqp.add_argument("--group", default="")
    vrp = sub.add_parser("voice-refresh-plan"); vrp.add_argument("--group", default="")
    vbk = sub.add_parser("voice-backup"); vbk.add_argument("--group", default="")
    dbk = sub.add_parser("db-backup"); dbk.add_argument("dest")
    dbr = sub.add_parser("db-restore"); dbr.add_argument("backup")
    dbr.add_argument("--dest", default="", help="куда восстановить; по умолчанию текущая база состояния")
    dbr.add_argument("--root", action="append", default=[], metavar="ID=ПУТЬ",
                     help="новый путь документного корня; недостающие мастер спросит в терминале")
    sc = sub.add_parser("self-check"); sc.add_argument("--no-tests", action="store_true")
    e = sub.add_parser("export"); e.add_argument("--group", default=""); e.add_argument("--dest", default=""); e.add_argument("--with-jobs", action="store_true")
    i = sub.add_parser("import"); i.add_argument("zip"); i.add_argument("--replace", action="store_true"); i.add_argument("--apply", action="store_true")
    d = sub.add_parser("delete-group"); d.add_argument("name"); d.add_argument("--permanent", action="store_true")
    k = sub.add_parser("cleanup"); k.add_argument("--days", type=float, default=30); k.add_argument("--apply", action="store_true")
    m = sub.add_parser("migrate"); m.add_argument("--apply", action="store_true")
    return ap


def run(a):
    c = a.cmd
    if c == "workflow-start":
        out(core.start_workflow(a.audio, {"require_conspect": a.conspect,
                                         "allow_cloud_model": bool(a.cloud_model and not a.no_cloud_model),
                                         "require_named_speakers": a.named_speakers,
                                         "require_human_verification": a.human_verified,
                                         "participation": a.participation, "goal": a.goal,
                                         "question_budget": a.question_budget}))
    elif c == "workflow-status":
        out(core.workflow_status(a.request))
    elif c == "workflow-task":
        out(core.workflow_model_task(a.request, a.offset, a.limit))
    elif c == "workflow-result":
        out(core.workflow_model_result(a.request, json.loads(Path(a.result_file).read_text(encoding="utf-8"))))
    elif c == "workflow-part":
        out(core.workflow_model_part(a.request, a.index))
    elif c == "workflow-part-result":
        out(core.workflow_model_part_result(a.request, a.index, json.loads(Path(a.result_file).read_text(encoding="utf-8"))))
    elif c == "workflow-execute":
        out(core.workflow_execute_model(a.request, retry_failed=a.retry_failed))
    elif c == "workflow-policy":
        out(core.workflow_change_policy(a.request, json.loads(Path(a.policy_file).read_text(encoding="utf-8")),
                                        a.expected_revision))
    elif c == "workflow-questions":
        out(core.workflow_questions(a.request))
    elif c == "workflow-list":
        out(core.workflow_list(a.limit))
    elif c == "screen":
        import workflow_page
        workflow_page.main(["--port", str(a.port), "--hours", str(a.hours)])
    elif c == "db-backup":
        report = core.backup_store(a.dest)
        print(f"Резерв создан: {report['destination']}; проверка: {report['report'].get('integrity')}. "
              "Копия неактивна; для использования нужен rebind_restored_copy.")
    elif c == "db-restore":
        return cmd_db_restore(a)
    elif c == "workflow-confirm":
        snapshot = core.workflow_verification_snapshot(a.request)
        return gated("Подтверждение проверки стенограммы",
                     [f"Стенограмма: {snapshot['document']}", f"Ревизия: {snapshot['source_revision']}",
                      f"Ревизия разбора: {snapshot['review_revision']}",
                      "Подтвердите только если самостоятельно проверили всю стенограмму."],
                     lambda: core.workflow_confirm_verification(snapshot))
    elif c == "workflow-abandon":
        snapshot = core.workflow_abandon_snapshot(a.request)
        return gated("Отказ от незавершённого запуска",
                     [f"Источник: {snapshot['source']}", f"Задача: {snapshot['job_id']}",
                      "Старый worker больше не сможет сохранить результат; процесс не останавливается."],
                     lambda: core.workflow_abandon(snapshot))
    elif c == "transcribe":
        out(core.start_transcription(a.audio, a.speakers, not a.no_emotions, a.vocab, a.tag, not a.no_voices,
                                     a.names, a.whisper, -1, a.group))
    elif c == "status":
        out(core.status(a.job, a.wait))
    elif c == "jobs":
        out(core.list_jobs(a.limit))
    elif c == "read":
        print(core.read_transcript(a.job, a.fmt, a.offset, a.max_chars, a.extra))
    elif c == "review":
        out(core.review_items(a.job, a.offset, a.limit, a.reasons, a.min_sec, a.brief, a.status))
    elif c == "hotspots":
        out(core.conversation_hotspots(a.job, a.limit, a.min_score))
    elif c == "review-build":
        items = None
        if a.items:
            items = json.loads(Path(a.items).read_text(encoding="utf-8-sig"))
            if not isinstance(items, list):
                raise core.CoreError("Файл вопросов должен содержать JSON-массив")
        out(core.build_review_pack(a.job, items))
    elif c == "review-page":
        out(core.start_review_page(a.job, a.port))
    elif c == "review-apply":
        out(core.apply_answers(a.job))
    elif c == "conspect-confirm":
        snapshot = core.describe_conspect_confirmation(a.job, a.name)
        return gated("Подтверждение проверки конспекта", [f"Конспект: {a.name}", f"Задача: {a.job}",
                     f"Ревизия источника: {snapshot['source_revision']}",
                     "Подтвердите только после самостоятельной проверки смысла и цитат."],
                     lambda: core.confirm_conspect_analysis(snapshot))
    elif c == "conspect-candidates":
        out(core.preview_conspect_candidates(a.job, Path(a.text_file).read_text(encoding="utf-8-sig")))
    elif c == "conspect-select":
        text = Path(a.text_file).read_text(encoding="utf-8-sig")
        selections = json.loads(Path(a.selection_file).read_text(encoding="utf-8-sig"))
        snapshot, report = core.prepare_selected_conspect(a.job, a.name, text, selections)
        lines = core.selected_conspect_confirmation_lines(a.job, a.name, snapshot, report, selections)
        return gated("Выбор места цитаты", lines, lambda: core.save_selected_conspect(snapshot, text, selections))
    elif c == "edit-span":
        out(core.edit_span(a.job, a.t1, a.t2, a.text, a.speaker, a.note, a.by))
    elif c == "reopen":
        out(core.reopen_review(a.job, a.ids))
    elif c == "cancel":
        out(core.cancel_job(a.job))
    elif c == "settings":
        out(core.settings_info())
    elif c == "profiles":
        out(core.voice_profiles())
    elif c == "voice-preview":
        out(core.preview_voice_learning(a.job, a.group or None))
    elif c in ("voice-update", "voice-revoke"):
        snapshot = core.prepare_voice_action(a.job, a.group or None) if c == "voice-update" \
            else core.prepare_voice_action(a.job)
        if c == "voice-revoke" and not snapshot["report"]["existing_contributions"]:
            raise core.CoreError("вкладов этой записи в профиле нет")
        return gated("Отзыв голосового вклада" if c == "voice-revoke" else "Обновление голосового профиля",
                     core.voice_action_lines(snapshot, revoke=c == "voice-revoke"),
                     lambda: core.revoke_voice_source(snapshot) if c == "voice-revoke" else core.start_voice_update(snapshot))
    elif c == "voice-auto-learn":
        return gated("Настройка автопополнения голосов", [f"Группа: {a.group or 'умолчание новых групп'}",
                     f"Автопополнение: {a.state}"],
                     lambda: core.set_voice_auto_learn(a.state == "on", a.group))
    elif c == "voice-quality":
        out(core.voice_quality(a.group))
    elif c == "voice-quality-policy":
        return gated("Политика качества голосов", [f"Группа: {a.group or 'умолчание новых групп'}",
                     f"Политика: {a.policy}",
                     "reject_clipping отбраковывает только клиппинг; тихая речь не отбраковывается."],
                     lambda: core.set_voice_quality_policy(a.policy, a.group))
    elif c == "voice-refresh-plan":
        out(core.voice_refresh_plan(a.group))
    elif c == "voice-backup":
        return gated("Резервное копирование голосовых профилей", [f"Группа: {a.group or 'все группы'}",
                     "Резерв — биометрия в открытом виде; старые копии не заменяются и не удаляются."],
                     lambda: core.backup_voice_profiles(a.group))
    elif c == "audit":
        import data_io
        out(data_io.audit())
    elif c == "approve":
        return cmd_approve(a)
    elif c == "self-check":
        import selfcheck
        r = selfcheck.run(None, not a.no_tests)
        out(r)
        return 0 if r["ok"] else 1
    elif c == "export":
        return gated("Выгрузка группы", core.describe_export(a.group, a.dest, a.with_jobs),
                     lambda: core.do_export(a.group, a.dest, a.with_jobs))
    elif c == "import":
        mode = "replace" if a.replace else "merge"
        if not a.apply:
            out(core.do_import(a.zip, mode, False))
        else:
            return gated("Загрузка архива", core.describe_import(a.zip, mode), lambda: core.do_import(a.zip, mode, True))
    elif c == "delete-group":
        return gated("Удаление группы", core.describe_delete(a.name, a.permanent),
                     lambda: core.do_delete_group(a.name, a.name, a.permanent), must_type=a.name)
    elif c == "cleanup":
        if not a.apply:
            out(core.do_cleanup(a.days, False))
        else:
            return gated("Очистка журналов задач", core.describe_cleanup(a.days), lambda: core.do_cleanup(a.days, True))
    elif c == "migrate":
        if not a.apply:
            out(core.do_migrate(True))
        else:
            return gated("Перенос данных", core.describe_migrate(), lambda: core.do_migrate(False))
    return 0


def main(argv=None):
    a = build().parse_args(argv)
    try:
        return run(a)
    except core.CoreError as ex:
        say(f"Ошибка: {ex}")
        return 1
    except Exception as ex:
        import llm_provider
        if isinstance(ex, llm_provider.ProviderError):
            say(f"Ошибка модельного исполнителя [{ex.code}]: {ex}")
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
