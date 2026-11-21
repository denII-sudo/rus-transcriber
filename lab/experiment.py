"""Лаборатория опытов rus-transcriber: прогон набора моделей по имеющимся записям в песочнице.

    <python> lab\\experiment.py prepare --sandbox D:\\lab-1 --voice-profiles copy
    <python> lab\\experiment.py run --sandbox D:\\lab-1 --dry-run
    <python> lab\\experiment.py run --sandbox D:\\lab-1 [--configs a-current,c-no-third] [--units 3fa1] [--limit 5]
    <python> lab\\experiment.py summary --sandbox D:\\lab-1
    <python> lab\\experiment.py analyze --sandbox D:\\lab-1

<python> — интерпретатор установленной программы: %LOCALAPPDATA%\\rus-transcriber\\.venv\\Scripts\\python.exe.
Код конвейера берётся из этого клона, модели — из кэша установленной программы, без сети.
Подробности — lab/README.md.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lab_analyze  # noqa: E402
import lab_run  # noqa: E402
import lab_sandbox  # noqa: E402
import lab_spec  # noqa: E402
import lab_summary  # noqa: E402

DEFAULT_EXPERIMENT = HERE / "experiments" / "models-arbitration.json"
# что искать в кэше моделей для предупреждения в --dry-run (только имена папок, без загрузки)
MODEL_HINTS = {"v3_e2e_rnnt": "gigaam-v3-e2e-rnnt", "v3_e2e_ctc": "gigaam-v3-e2e-ctc",
               "v3_rnnt": "gigaam-v3-rnnt", "v3_ctc": "gigaam-v3-ctc",
               "multilingual_ctc": "gigaam-multilingual-ctc",
               "multilingual_large_ctc": "gigaam-multilingual-large-ctc",
               "whisper-ct2": "faster-whisper-large-v3-turbo",
               "onnx-community/whisper-large-v3-turbo": "whisper-large-v3-turbo",
               "pyannote": "speaker-diarization-community-1"}


def _utf8_console():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def cached_model_names(program_dir):
    cache = Path(program_dir) / "cache"
    names = []
    for root in (cache / "huggingface" / "hub", cache / "onnx-models"):
        try:
            names += [p.name.lower() for p in root.iterdir() if p.is_dir()]
        except OSError:
            pass
    return names


def model_warnings(experiment, base_cfg, program_dir):
    names = cached_model_names(program_dir)
    if not names:
        return [f"кэш моделей не найден в {Path(program_dir) / 'cache'}: прогон без сети не сможет загрузить модели"]
    need = {"pyannote"}
    for conf in experiment["configurations"]:
        cfg = lab_spec.effective_settings(base_cfg, conf)
        roles = conf["roles"]
        need.add(cfg["asr_model"])
        if roles.get("second"):
            need.add(cfg["asr_second_model"])
        t = roles.get("third")
        if t:
            need.add("whisper-ct2" if t["engine"] == "whisper-ct2" else cfg.get("asr_third_model"))
    out = []
    for model in sorted(n for n in need if n):
        hint = MODEL_HINTS.get(model, str(model).split("/")[-1].lower())
        if not any(hint in n for n in names):
            out.append(f"модель «{model}» не найдена в кэше по имени папки (ищется «{hint}»); "
                       "этапы с ней завершатся ошибкой загрузки (скачивания не будет)")
    return out


def cmd_prepare(a):
    sources = [Path(a.source_data)] if a.source_data else lab_sandbox.default_data_dirs()
    source = next((p for p in sources if (p / "state.sqlite3").is_file()), None)
    if source is None:
        raise SystemExit("не найдена папка данных с state.sqlite3; укажите --source-data")
    program = Path(a.program_dir) if a.program_dir else lab_sandbox.default_program_dir()
    lab_sandbox.prepare(a.sandbox, source, program, a.voice_profiles)


def _passport(program_dir, cfg):
    import paths
    import runinfo
    return runinfo.scrub({
        "schema_version": 1, "lab_method": "lab-run-1",
        "environment": runinfo.environment(lab_sandbox.REPO_DIR),
        "hardware": runinfo.hardware(),
        "testbed_id": runinfo.testbed_id(paths.BENCH),
        "models_cache": runinfo.models_cache(Path(program_dir) / "cache",
                                             hf_home=Path(program_dir) / "cache" / "huggingface"),
        "offline": {k: os.environ.get(k) for k in lab_run.OFFLINE_ENV},
        "settings_base": {k: v for k, v in cfg.items()
                          if k.startswith(("asr_", "diar_")) or k in ("torch_threads", "pipeline",
                                                                       "interaction", "review_missed")},
    })


def cmd_run(a):
    sandbox, marker, units_doc = lab_sandbox.load_sandbox(a.sandbox)
    program = Path(a.program_dir or marker["program_dir"])
    experiment, sha = lab_spec.load(a.experiment)
    old_env = lab_run.setup_environment(sandbox, program)
    try:
        engine = lab_run.PipelineEngine(sandbox, allow_torch=experiment.get("allow_torch_fallback", False))
        engine.point_caches(program)
        base = engine.base_config()
        for w in model_warnings(experiment, base, program):
            print("ПРЕДУПРЕЖДЕНИЕ: " + w)
        if not a.dry_run and not (program / "cache").is_dir():
            raise SystemExit(f"нет кэша моделей установленной программы: {program / 'cache'}")

        def conditions():
            import runinfo
            return runinfo.conditions(jobs_dir=Path(marker["source_data"]) / "jobs",
                                      slot_limit=base.get("max_heavy_jobs", 2), load_interval=1.0)

        args = {"units": a.units, "configs": a.configs, "limit": a.limit, "experiment_file":
                Path(a.experiment).name}
        lab_run.run(sandbox, experiment, sha, units_doc, engine, program_dir=program,
                    unit_filter=_split(a.units), config_filter=_split(a.configs), limit=a.limit,
                    dry_run=a.dry_run, passport_fn=lambda: _passport(program, base),
                    conditions_fn=conditions, args=args)
    finally:
        lab_run.restore_environment(old_env)


def cmd_summary(a):
    sandbox, marker, units_doc = lab_sandbox.load_sandbox(a.sandbox)
    db = lab_run.open_results(sandbox / "results" / "lab.sqlite3")
    try:
        out = sandbox / "export" / "summary.json"
        lab_summary.export_summary(db, units_doc["units"], None, out)
        print(f"Сводка без текстов и имён: {out}")
    finally:
        db.close()


def cmd_analyze(a):
    sandbox, marker, units_doc = lab_sandbox.load_sandbox(a.sandbox)
    analysis = lab_analyze.analyze(sandbox, units_doc, baseline=a.baseline)
    print(f"Отчёт (только этот компьютер): {sandbox / 'results' / 'report.md'}")
    print(f"Примеры со словами (только этот компьютер): {sandbox / 'results' / 'examples.md'}")
    print(f"Обезличенные числа: {sandbox / 'export' / 'analysis.json'}")
    if analysis.get("not_measured"):
        print("Не измерено: " + analysis["not_measured"])
    bad = [k for k, v in (analysis.get("revote_check") or {}).items() if v.get("state") == "mismatch"]
    if bad:
        print(f"ВНИМАНИЕ: пересведение не совпало с конвейером в {len(bad)} парах — см. раздел «Сверка» отчёта")


def _split(value):
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def main(argv=None):
    _utf8_console()
    ap = argparse.ArgumentParser(description="Лаборатория опытов: прогон моделей по записям в песочнице")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="создать песочницу: копия базы, настроек и словарей")
    p.add_argument("--sandbox", required=True, help="новая пустая папка песочницы")
    p.add_argument("--voice-profiles", required=True, choices=("copy", "skip"),
                   help="копировать ли голосовые профили (биометрия): copy — как в обычной обработке, "
                        "skip — без них")
    p.add_argument("--source-data", default="", help="настоящая папка данных (по умолчанию — как у программы)")
    p.add_argument("--program-dir", default="", help="папка установленной программы (кэш моделей)")
    p.set_defaults(fn=cmd_prepare)
    r = sub.add_parser("run", help="посчитать пары «запись × конфигурация»")
    r.add_argument("--sandbox", required=True)
    r.add_argument("--experiment", default=str(DEFAULT_EXPERIMENT), help="файл опыта (JSON)")
    r.add_argument("--units", default="", help="единицы через запятую (начало unit_id)")
    r.add_argument("--configs", default="", help="конфигурации через запятую (id или начало id)")
    r.add_argument("--limit", type=int, default=0, help="посчитать не больше N пар за запуск")
    r.add_argument("--dry-run", action="store_true", help="только план и оценка времени, без счёта")
    r.add_argument("--program-dir", default="", help="папка установленной программы (по умолчанию — из песочницы)")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("summary", help="пересобрать обезличенную сводку export/summary.json")
    s.add_argument("--sandbox", required=True)
    s.set_defaults(fn=cmd_summary)
    n = sub.add_parser("analyze", help="анализ результатов: отчёт results/report.md и числа export/analysis.json")
    n.add_argument("--sandbox", required=True)
    n.add_argument("--baseline", default=lab_analyze.BASELINE, help="базовая конфигурация для сравнения")
    n.set_defaults(fn=cmd_analyze)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (lab_sandbox.SandboxError, lab_spec.SpecError) as ex:
        print(f"ОТКАЗ: {ex}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
