"""Командная строка rus-transcriber: те же возможности, что у MCP-сервера, без Claude и без любого MCP-клиента.

    python rt.py transcribe запись.mp3 --names "Анна, Борис" --group моя-группа
    python rt.py status ЗАДАЧА --wait 50        # ждёт смены этапа или конца (до 50 с)
    python rt.py jobs [--limit 10]
    python rt.py read ЗАДАЧА [--fmt md|json] [--extra] [--offset 0]
    python rt.py review ЗАДАЧА [--offset 0 --limit 40] [--reasons voice,gender] [--min-sec 1] [--brief] [--status open]
    python rt.py edit-span ЗАДАЧА T1 T2 "текст" [--speaker Имя] [--note ...] [--by human]
    python rt.py reopen ЗАДАЧА r57,r88
    python rt.py cancel ЗАДАЧА
    python rt.py settings | profiles | audit    # audit: абсолютные пути в данных
    python rt.py self-check [--no-tests]        # тесты, проверка файлов на личные данные, морфология

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
    s = sub.add_parser("status"); s.add_argument("job"); s.add_argument("--wait", type=int, default=0)
    j = sub.add_parser("jobs"); j.add_argument("--limit", type=int, default=10)
    r = sub.add_parser("read"); r.add_argument("job"); r.add_argument("--fmt", default="md", choices=["md", "json"])
    r.add_argument("--extra", action="store_true"); r.add_argument("--offset", type=int, default=0)
    r.add_argument("--max-chars", type=int, default=60000)
    v = sub.add_parser("review"); v.add_argument("job"); v.add_argument("--offset", type=int, default=0); v.add_argument("--limit", type=int, default=40)
    v.add_argument("--reasons", default=""); v.add_argument("--min-sec", type=float, default=0.0)
    v.add_argument("--brief", action="store_true"); v.add_argument("--status", default="open")
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
    sc = sub.add_parser("self-check"); sc.add_argument("--no-tests", action="store_true")
    e = sub.add_parser("export"); e.add_argument("--group", default=""); e.add_argument("--dest", default=""); e.add_argument("--with-jobs", action="store_true")
    i = sub.add_parser("import"); i.add_argument("zip"); i.add_argument("--replace", action="store_true"); i.add_argument("--apply", action="store_true")
    d = sub.add_parser("delete-group"); d.add_argument("name"); d.add_argument("--permanent", action="store_true")
    k = sub.add_parser("cleanup"); k.add_argument("--days", type=float, default=30); k.add_argument("--apply", action="store_true")
    m = sub.add_parser("migrate"); m.add_argument("--apply", action="store_true")
    return ap


def run(a):
    c = a.cmd
    if c == "transcribe":
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
    elif c == "review-build":
        items = None
        if a.items:
            from pathlib import Path
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
        lines = [f"Новый конспект: {a.name}", f"Задача: {a.job}, ревизия: {snapshot['source_revision']}"]
        for entry in report["candidates"]:
            selected = selections.get(str(entry["quote_index"]))
            if selected:
                option = next(x for x in entry["candidates"] if x["word_ids"] == selected["word_ids"])
                lines.append(f"Цитата «{entry['quote'][:80]}»: {option['speaker']}, {option['t1']}–{option['t2']} с")
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


if __name__ == "__main__":
    sys.exit(main())
