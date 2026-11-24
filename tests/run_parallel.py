"""Параллельный запуск набора тестов: модули делятся на группы, каждая группа — свой процесс unittest.

Запуск из корня клона (или откуда угодно: путь к репозиторию берётся от этого файла):

    python -B tests/run_parallel.py            # N = min(4, число ядер) групп
    python -B tests/run_parallel.py -v         # подробный вывод, строки групп с метками [g1] … [g4]
    python -B tests/run_parallel.py --shuffle  # случайная раскладка по группам; число (seed) печатается в итоге
    python -B tests/run_parallel.py --shuffle 12345   # повторить ту же раскладку
    python -B tests/run_parallel.py --sequential      # прежний путь: один процесс, `unittest discover -s tests`

Что гарантирует запуск (иначе итог — FAILED и код выхода 1):
  * каждый модуль набора попал ровно в одну группу (сверка до запуска);
  * список модулей и число тестов в каждой группе совпали с тем, что находит `discover` (отдельный процесс);
  * у каждой группы есть итоговая строка unittest (убитая, зависшая или вылетевшая группа — отказ);
  * дерево репозитория после запуска то же, что до него (`git status --porcelain --ignored`).
Остаток файлов во временной папке группы — предупреждение (`--strict-tmp` делает его отказом).

У каждой группы свои каталоги `data` и `tmp` в новой песочнице: `RUS_TRANSCRIBER_DATA`, `TEMP`, `TMP`.
Переменные окружения, патчи и модульные кэши между группами не делятся (разные процессы); общее вне процесса
(порты, файлы, git) разобрано в аудите плана ускорения тестов; для git задано GIT_OPTIONAL_LOCKS=0.

Имя файла не подходит под `test*.py`, поэтому `discover` его не подхватывает. Таблица длительностей модулей —
`tests/parallel_durations.json` (только для баланса: устаревшая таблица портит скорость, не результат);
обновить: `--update-table`.
"""
import argparse
import fnmatch
import json
import os
import random
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
PATTERN = "test*.py"                  # как у `unittest discover` по умолчанию
MAX_JOBS = 4
GROUP_TIMEOUT = 900                   # секунд на группу; предел CI — 20 минут на всё задание
SUMMARY_RAN = re.compile(r"^Ran (\d+) tests? in ([\d.]+)s\s*$", re.M)
SUMMARY_STATUS = re.compile(r"^(OK|FAILED)(?: \((.*)\))?\s*$", re.M)
DURATION_LINE = re.compile(r"^\s*([\d.]+)s\s+\S+ \(([\w.]+)\)\s*$", re.M)


# ---------- модули и раскладка ----------

def list_modules(tests_dir):
    """Имена модулей набора — как их находит `discover -s tests` (файлы test*.py в корне папки)."""
    tests_dir = Path(tests_dir)
    for sub in tests_dir.iterdir():
        if sub.is_dir() and (sub / "__init__.py").exists():
            raise SystemExit(f"в {tests_dir.name}/ есть пакет {sub.name}: запускатель его не поддерживает")
    return sorted(p.stem for p in tests_dir.iterdir()
                  if p.is_file() and p.suffix == ".py" and fnmatch.fnmatch(p.name, PATTERN))


def load_table(path):
    """-> (секунды по модулям, списки модулей, которые держать в одной группе)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, []
    modules = {k: float(v) for k, v in (data.get("modules") or {}).items() if isinstance(v, (int, float))}
    return modules, [list(g) for g in (data.get("together") or []) if isinstance(g, list)]


def plan_groups(modules, seconds, jobs, together=(), seed=None):
    """Делит модули на группы по ожидаемому времени. -> список списков имён (пустых групп нет).

    Без seed: самый тяжёлый «кусок» — в самую лёгкую группу (устойчивая раскладка). С seed: куски перемешаны,
    раскладка и порядок внутри групп случайны, но по тому же правилу «в самую лёгкую» — баланс сохраняется."""
    known = [seconds[m] for m in modules if m in seconds]
    default = (sum(known) / len(known)) if known else 1.0
    weight = {m: seconds.get(m, default) for m in modules}
    left, units = set(modules), []
    for glue in together:
        part = [m for m in glue if m in left]
        if part:
            units.append(part)
            left -= set(part)
    units += [[m] for m in sorted(left)]
    jobs = max(1, min(jobs, len(units)))
    groups, load = [[] for _ in range(jobs)], [0.0] * jobs
    rng = random.Random(seed) if seed is not None else None
    if rng:
        rng.shuffle(units)
    else:
        units.sort(key=lambda u: (-sum(weight[m] for m in u), u[0]))
    for unit in units:
        lightest = min(load)
        pick = [i for i, v in enumerate(load) if v == lightest]
        i = rng.choice(pick) if rng else pick[0]
        groups[i] += unit
        load[i] += sum(weight[m] for m in unit)
    if rng:
        for g in groups:
            rng.shuffle(g)
    else:
        groups = [sorted(g) for g in groups]
    return [g for g in groups if g]


def verify_plan(modules, groups):
    """Сверка «каждый модуль ровно один раз». -> список замечаний (пустой — всё верно)."""
    seen, problems = {}, []
    for i, group in enumerate(groups, 1):
        for m in group:
            seen.setdefault(m, []).append(i)
    for m in modules:
        if m not in seen:
            problems.append(f"модуль {m} не попал ни в одну группу")
        elif len(seen[m]) > 1:
            problems.append(f"модуль {m} в нескольких группах: {seen[m]}")
    for m in seen:
        if m not in modules:
            problems.append(f"в группах есть модуль {m}, которого нет в наборе")
    return problems


# ---------- итог unittest ----------

def parse_summary(text):
    """Итог одного процесса unittest: {ran, seconds, ok, failures, errors, skipped, expected, unexpected} или None."""
    ran = SUMMARY_RAN.findall(text)
    status = SUMMARY_STATUS.findall(text)
    if not ran or not status:
        return None
    n, secs = ran[-1]
    word, detail = status[-1]
    items = {}
    for part in (detail or "").split(","):
        key, _, val = part.strip().partition("=")
        if val.isdigit():
            items[key.strip()] = int(val)
    return {"ran": int(n), "seconds": float(secs), "ok": word == "OK",
            "failures": items.get("failures", 0), "errors": items.get("errors", 0),
            "skipped": items.get("skipped", 0), "expected": items.get("expected failures", 0),
            "unexpected": items.get("unexpected successes", 0)}


def module_durations(text):
    """Секунды по модулям из блока `--durations 0`."""
    out = {}
    for secs, test_id in DURATION_LINE.findall(text.split("Slowest test durations", 1)[-1]):
        mod = test_id.split(".")[0]
        out[mod] = out.get(mod, 0.0) + float(secs)
    return out


# ---------- процессы ----------

def group_env(sandbox, tests_dir):
    data, temp = Path(sandbox) / "data", Path(sandbox) / "tmp"
    data.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=str(temp), TMP=str(temp),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               GIT_OPTIONAL_LOCKS="0")
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(tests_dir) + (os.pathsep + old if old else "")
    return env, temp


def spawn(argv, cwd, env):
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {"start_new_session": True}
    return subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, **kwargs)


def kill_tree(proc):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=60)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


class Run:
    """Один процесс unittest: свой каталог, вывод собирается потоком (и печатается сразу при verbose)."""

    def __init__(self, label, argv, cwd, tests_dir, root, modules, verbose, say):
        self.label, self.modules, self.argv, self.cwd = label, modules, argv, cwd
        self.sandbox = Path(root) / label
        self.env, self.tmp = group_env(self.sandbox, tests_dir)
        self.verbose, self.say = verbose, say
        self.lines, self.proc, self.thread = [], None, None
        self.started = self.ended = None
        self.timed_out = False
        self.summary = None
        self.leftover = []

    def start(self):
        self.started = time.time()
        self.proc = spawn(self.argv, self.cwd, self.env)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            self.lines.append(line)
            if self.verbose:
                self.say(f"[{self.label}] {line}")

    def wait(self, deadline):
        try:
            self.proc.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            self.timed_out = True
            kill_tree(self.proc)
            self.proc.wait()
        self.thread.join(timeout=30)
        try:
            self.proc.stdout.close()
        except (OSError, ValueError):
            pass
        self.ended = time.time()
        self.summary = parse_summary(self.output)
        try:
            self.leftover = sorted(p.name for p in self.tmp.iterdir())
        except OSError:
            self.leftover = []

    @property
    def output(self):
        return "\n".join(self.lines) + "\n"

    @property
    def seconds(self):
        return (self.ended or time.time()) - (self.started or time.time())


def tree_state(repo):
    """Снимок дерева репозитория (отслеживаемые изменения, неотслеживаемые и игнорируемые файлы) или None."""
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-c", "core.quotePath=false", "-C", str(repo), "status",
                            "--porcelain", "--ignored", "--untracked-files=all"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return sorted(r.stdout.splitlines()) if r.returncode == 0 else None


COUNT_CODE = (
    "import json, unittest\n"
    "counts = {}\n"
    "def walk(suite):\n"
    "    for t in suite:\n"
    "        if isinstance(t, unittest.TestSuite):\n"
    "            walk(t)\n"
    "        else:\n"
    "            mod = t._testMethodName if type(t).__name__ == '_FailedTest' else type(t).__module__\n"
    "            counts[mod] = counts.get(mod, 0) + 1\n"
    "walk(unittest.defaultTestLoader.discover('tests'))\n"
    "print('@@COUNTS@@' + json.dumps(counts))\n")


def read_counts(run):
    for line in run.lines:
        if line.startswith("@@COUNTS@@"):
            try:
                return json.loads(line[len("@@COUNTS@@"):])
            except ValueError:
                return None
    return None


# ---------- запуск ----------

def run(argv=None, *, repo=REPO, out=None, mutate=None):
    """Весь запуск. -> код выхода: 0 — всё хорошо, 1 — падения или нарушения запуска, 2 — неверная раскладка."""
    out = out or sys.stdout
    lock = threading.Lock()

    def say(text=""):
        with lock:
            print(text, file=out, flush=True)

    ap = argparse.ArgumentParser(description="Параллельный запуск тестов (по группам модулей)")
    ap.add_argument("-v", "--verbose", action="store_true", help="подробный вывод unittest с метками групп")
    ap.add_argument("--jobs", type=int, help=f"число групп (по умолчанию min({MAX_JOBS}, число ядер))")
    ap.add_argument("--shuffle", nargs="?", const="random", metavar="SEED",
                    help="случайная раскладка по группам; без числа — seed выбирается и печатается")
    ap.add_argument("--sequential", action="store_true", help="один процесс: unittest discover -s tests")
    ap.add_argument("--timeout", type=int, default=GROUP_TIMEOUT, help="секунд на группу (по умолчанию %(default)s)")
    ap.add_argument("--table", help="таблица длительностей модулей (по умолчанию tests/parallel_durations.json)")
    ap.add_argument("--update-table", action="store_true", help="записать длительности модулей этого запуска в таблицу")
    ap.add_argument("--no-count", action="store_true", help="не сверять число тестов с discover")
    ap.add_argument("--strict-tmp", action="store_true", help="остаток во временной папке группы — отказ")
    args = ap.parse_args(argv)

    repo = Path(repo)
    tests_dir = repo / "tests"
    table_path = Path(args.table) if args.table else tests_dir / "parallel_durations.json"
    modules = list_modules(tests_dir)
    seed = None
    if args.shuffle is not None:
        seed = secrets.randbelow(10 ** 9) if args.shuffle == "random" else int(args.shuffle)
    jobs = args.jobs or min(MAX_JOBS, os.cpu_count() or 1)
    problems = []
    py = sys.executable

    seconds, together = load_table(table_path)
    if args.sequential:
        groups = []
    else:
        groups = plan_groups(modules, seconds, jobs, together, seed)
        if mutate:
            groups = mutate(groups)
        plan_problems = verify_plan(modules, groups)
        if plan_problems:
            for p in plan_problems:
                say("РАСКЛАДКА: " + p)
            say("----------------------------------------------------------------------")
            say("Ran 0 tests in 0.000s\n")
            say(f"FAILED (run problems={len(plan_problems)})")
            return 2

    root = Path(tempfile.mkdtemp(prefix="rt-par-"))
    before = tree_state(repo)
    extra = (["-v"] if args.verbose else []) + (["--durations", "0"] if args.update_table else [])
    runs = []
    if args.sequential:
        runs.append(Run("seq", [py, "-B", "-m", "unittest", "discover", "-s", "tests", *extra], repo, tests_dir, root,
                        modules, args.verbose, say))
    else:
        for i, g in enumerate(groups, 1):
            runs.append(Run(f"g{i}", [py, "-B", "-m", "unittest", *extra, *g], repo, tests_dir, root, g,
                            args.verbose, say))
    counter = None
    if not args.sequential and not args.no_count:
        counter = Run("count", [py, "-B", "-c", COUNT_CODE], repo, tests_dir, root, [], False, say)
    started = time.time()
    if seed is not None:
        say(f"раскладка: seed={seed} (повтор: python -B tests/run_parallel.py --jobs {len(groups)} --shuffle {seed})")
    say(f"запуск: {'последовательно, один процесс' if args.sequential else f'{len(runs)} групп(ы) параллельно'}"
        f", модулей {len(modules)}")
    for r in runs:
        r.start()
        say(f"{r.label}: начата" + ("" if args.sequential else f", модулей {len(r.modules)}"))
    if counter:
        counter.start()
    deadline = started + args.timeout
    for r in runs:
        r.wait(deadline)
        s = r.summary
        say(f"{r.label}: " + ("не уложилась в %d с, убита" % args.timeout if r.timed_out else
                              "нет итоговой строки unittest" if not s else
                              f"Ran {s['ran']} за {s['seconds']:.1f} с, {'OK' if s['ok'] else 'FAILED'}")
            + f" (по часам {r.seconds:.1f} с)")
    if counter:
        counter.wait(time.time() + args.timeout)
    wall = time.time() - started

    # сверки
    totals = dict.fromkeys(("ran", "failures", "errors", "skipped", "expected", "unexpected"), 0)
    for r in runs:
        if r.timed_out:
            problems.append(f"{r.label}: превышен срок {args.timeout} с, группа убита")
        elif not r.summary:
            problems.append(f"{r.label}: нет итоговой строки unittest (процесс вылетел или вывод оборван)")
        else:
            for k in totals:
                totals[k] += r.summary[k]
            if r.leftover:
                text = f"{r.label}: во временной папке группы остались {len(r.leftover)} объектов (первые: " \
                       f"{', '.join(r.leftover[:3])})"
                if args.strict_tmp:
                    problems.append(text)
                else:
                    say("предупреждение: " + text)
    if counter:
        counts = read_counts(counter)
        if counts is None:
            problems.append("не удалось посчитать тесты через discover: " + (counter.output.strip()[-300:] or "нет вывода"))
        else:
            if sorted(counts) != modules:
                problems.append("список модулей не совпал с найденным discover: только в discover "
                                f"{sorted(set(counts) - set(modules))}, только у запускателя {sorted(set(modules) - set(counts))}")
            for r in runs:
                want = sum(counts.get(m, 0) for m in r.modules)
                if r.summary and r.summary["ran"] != want:
                    problems.append(f"{r.label}: запущено {r.summary['ran']} тестов, по discover у её модулей {want}")
    after = tree_state(repo)
    if before is not None and after is not None and before != after:
        changed = sorted(set(before) ^ set(after))
        problems.append("дерево репозитория изменилось за время запуска: " + "; ".join(changed[:5]))

    bad = [r for r in runs if r.timed_out or not r.summary or not r.summary["ok"]]
    if args.update_table and not bad and not problems:
        merged = dict(seconds)
        for r in runs:
            for m, v in module_durations(r.output).items():
                merged[m] = round(v, 1)
        table_path.write_text(json.dumps({"modules": dict(sorted(merged.items())), "together": together},
                                         ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        say(f"таблица длительностей записана: {table_path.name}")
    for r in bad:
        say(f"===== {r.label}: полный вывод упавшей группы =====")
        say(r.output.rstrip("\n"))
        say(f"===== конец вывода {r.label} =====")
    for p in problems:
        say("НАРУШЕНИЕ: " + p)

    parts = []
    if totals["failures"] or bad:
        parts.append(f"failures={totals['failures']}")
    if totals["errors"]:
        parts.append(f"errors={totals['errors']}")
    if problems:
        parts.append(f"run problems={len(problems)}")
    failed = bool(bad or problems or totals["failures"] or totals["errors"] or totals["unexpected"])
    for key, label in (("skipped", "skipped"), ("expected", "expected failures"), ("unexpected", "unexpected successes")):
        if totals[key]:
            parts.append(f"{label}={totals[key]}")
    say("----------------------------------------------------------------------")
    say(f"Ran {totals['ran']} tests in {wall:.3f}s\n")
    say(("FAILED" if failed else "OK") + (f" ({', '.join(parts)})" if parts else ""))
    if not args.sequential:
        say(f"групп {len(runs)}" + (f", seed {seed}" if seed is not None else "") +
            "; суммарное время групп " + f"{sum(r.seconds for r in runs):.0f} с")
    shutil.rmtree(root, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(run())
