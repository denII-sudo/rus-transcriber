"""Параллельный запуск набора тестов: модули делятся на группы, каждая группа — свой процесс unittest.

Запуск из корня клона (или откуда угодно: путь к репозиторию берётся от этого файла):

    python -B tests/run_parallel.py            # N = min(4, число ядер) групп
    python -B tests/run_parallel.py -v         # подробный вывод, строки групп с метками [g1] … [g4]
    python -B tests/run_parallel.py --shuffle  # случайный порядок модулей внутри групп (состав групп прежний); seed печатается
    python -B tests/run_parallel.py --shuffle 12345   # повторить тот же порядок
    python -B tests/run_parallel.py --sequential      # прежний путь: один процесс, `unittest discover -s tests`

Что гарантирует запуск (иначе итог — FAILED и код выхода 1):
  * каждый модуль набора попал ровно в одну группу (сверка до запуска);
  * список модулей и число тестов в каждой группе совпали с тем, что находит `discover` (отдельный процесс);
  * у каждой группы есть итоговая строка unittest (убитая, зависшая или вылетевшая группа — отказ);
  * дерево репозитория после запуска то же, что до него (`git status --porcelain --ignored`); без Git (установленная
    копия) эту сверку выполнить нельзя: об этом печатается одна строка «предупреждение: …», код выхода не меняется.
Остаток файлов во временной папке группы — предупреждение (`--strict-tmp` делает его отказом).

У каждой группы свои каталоги `data` и `tmp` в новой песочнице: `RUS_TRANSCRIBER_DATA`, `TEMP`, `TMP`, а также кэши
моделей `HF_HOME` и `TORCH_HOME` (тесты не видят кэш установленной программы). Успешный запуск песочницу убирает;
неудачный, исключение или Ctrl-C (группы при этом убиваются) оставляют её, путь печатается последней строкой итога.
Группа с ненулевым кодом выхода считается упавшей, даже если unittest напечатал OK.
Переменные окружения, патчи и модульные кэши между группами не делятся (разные процессы); общее вне процесса
(порты, файлы, git) разобрано в аудите плана ускорения тестов; для git задано GIT_OPTIONAL_LOCKS=0.

Имя файла не подходит под `test*.py`, поэтому `discover` его не подхватывает. Таблица длительностей модулей —
`tests/parallel_durations.json` (только для баланса: устаревшая таблица портит скорость, не результат);
обновить: `--update-table` (после прогона) или `--table-from ЖУРНАЛ` (из сохранённого вывода с `--durations`).

Запись в файловом режиме (`legacy_files`) будет убрана. В таблице два столбца: `modules` — секунды сейчас,
`after_legacy_removal` — оценка после ухода этого режима; раскладка строится по их среднему, поэтому ровна и до, и
после. Оценка считается при пересчёте таблицы по тексту тестов (`classify_tests`): тест, гоняющий оба режима, даёт
после ухода долю SQLITE_COST/(1+SQLITE_COST) своего времени; тест только для файлового режима — 0; остальные — как были.
Когда `legacy_files` из тестов исчезнет, пересчёт сам даст «после» = «сейчас» (одна команда, без ручной правки).
`--plan` печатает раскладку и ожидаемую нагрузку групп в обоих состояниях, ничего не запуская.
"""
import argparse
import ast
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
# класс или модуль, пропущенный целиком (SkipTest в setUpClass/setUpModule): unittest не засчитывает их тесты в «Ran»
SETUP_SKIPPED = re.compile(r"(setUpClass|setUpModule) \(([\w.]+)\) \.\.\. skipped")
DURATION_LINE = re.compile(r"(?:^|\s)(\d+\.\d+)s\s+\S+ \(([\w.]+)\)\s*$", re.M)
DATA_SUBDIR = "data"                  # папка данных группы внутри её TEMP
SQLITE_COST = 4.5                     # цена варианта в режиме sqlite к варианту legacy_files (К-2 плана: 0,42 против 0,09 с)


# ---------- модули и раскладка ----------

def list_modules(tests_dir):
    """Имена модулей набора — как их находит `discover -s tests` (файлы test*.py в корне папки)."""
    tests_dir = Path(tests_dir)
    for sub in tests_dir.iterdir():
        if sub.is_dir() and (sub / "__init__.py").exists():
            raise SystemExit(f"в {tests_dir.name}/ есть пакет {sub.name}: запускатель его не поддерживает")
    return sorted(p.stem for p in tests_dir.iterdir()
                  if p.is_file() and p.suffix == ".py" and fnmatch.fnmatch(p.name, PATTERN))


def _read_table(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _seconds(section):
    return {k: float(v) for k, v in (section or {}).items() if isinstance(v, (int, float))}


def load_table(path):
    """-> (секунды по модулям, списки модулей, которые держать в одной группе)."""
    data = _read_table(path)
    return _seconds(data.get("modules")), [list(g) for g in (data.get("together") or []) if isinstance(g, list)]


def load_after(path):
    """Оценка секунд по модулям после ухода режима legacy_files (пусто, если в таблице её нет)."""
    return _seconds(_read_table(path).get("after_legacy_removal"))


def plan_groups(modules, seconds, jobs, together=(), seed=None, after=None):
    """Делит модули на группы по ожидаемому времени. -> список списков имён (пустых групп нет).

    Два состояния набора: «сейчас» (`seconds`) и «после ухода legacy_files» (`after`, если дана). Кусок ставится в ту
    группу, где после него наибольшая нагрузка в любом из двух состояний меньше всего (при одном состоянии это
    правило «в самую лёгкую группу»). Куски идут от самого тяжёлого, а затем перестановками и переносами убирается
    остаточный перекос в обоих состояниях — состав групп детерминирован и не зависит от seed. Seed перемешивает только
    порядок модулей внутри каждой группы (скрытая зависимость соседей по набору проявится, а нагрузка групп не меняется)."""
    known = [seconds[m] for m in modules if m in seconds]
    default = (sum(known) / len(known)) if known else 1.0
    now = {m: seconds.get(m, default) for m in modules}
    then = {m: (after or {}).get(m, now[m]) for m in modules}
    left, units = set(modules), []
    for glue in together:
        part = [m for m in glue if m in left]
        if part:
            units.append(part)
            left -= set(part)
    units += [[m] for m in sorted(left)]
    jobs = max(1, min(jobs, len(units)))
    weight = {tuple(u): (sum(now[m] for m in u), sum(then[m] for m in u)) for u in units}
    units.sort(key=lambda u: (-max(weight[tuple(u)]), u[0]))
    groups, load = [[] for _ in range(jobs)], [[0.0, 0.0] for _ in range(jobs)]
    for unit in units:
        a, b = weight[tuple(unit)]
        score = [max(load[i][0] + a, load[i][1] + b) for i in range(jobs)]
        i = score.index(min(score))
        groups[i].append(unit)
        load[i][0] += a
        load[i][1] += b

    def cost(loads):
        return (max(max(x[0] for x in loads), max(x[1] for x in loads)),
                sum(x[0] ** 2 + x[1] ** 2 for x in loads))

    for _ in range(200):                         # доводка: перенос или обмен кусков, пока что-то улучшается
        loads = [[sum(weight[tuple(u)][0] for u in g), sum(weight[tuple(u)][1] for u in g)] for g in groups]
        best, move = cost(loads), None
        for gi in range(jobs):
            for ui, u in enumerate(groups[gi]):
                wu = weight[tuple(u)]
                for gj in range(jobs):
                    if gj == gi:
                        continue
                    trial = [x[:] for x in loads]
                    trial[gi] = [trial[gi][0] - wu[0], trial[gi][1] - wu[1]]
                    trial[gj] = [trial[gj][0] + wu[0], trial[gj][1] + wu[1]]
                    if len(groups[gi]) > 1 and cost(trial) < best:
                        best, move = cost(trial), ("move", gi, ui, gj, None)
                    for vi, v in enumerate(groups[gj]):
                        wv = weight[tuple(v)]
                        swap = [x[:] for x in loads]
                        swap[gi] = [swap[gi][0] - wu[0] + wv[0], swap[gi][1] - wu[1] + wv[1]]
                        swap[gj] = [swap[gj][0] + wu[0] - wv[0], swap[gj][1] + wu[1] - wv[1]]
                        if cost(swap) < best:
                            best, move = cost(swap), ("swap", gi, ui, gj, vi)
        if not move:
            break
        kind, gi, ui, gj, vi = move
        if kind == "move":
            groups[gj].append(groups[gi].pop(ui))
        else:
            groups[gi][ui], groups[gj][vi] = groups[gj][vi], groups[gi][ui]
    result = [g for g in (sorted(m for unit in grp for m in unit) for grp in groups) if g]
    if seed is not None:
        rng = random.Random(seed)
        for g in result:
            rng.shuffle(g)
    return result


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


def test_durations(text):
    """Секунды по тестам {модуль.Класс.тест: с} из блоков `--durations 0` (метки групп и времени перед строкой — не помеха)."""
    out = {}
    for secs, test_id in DURATION_LINE.findall(text.split("Slowest test durations", 1)[-1]):
        out[test_id] = out.get(test_id, 0.0) + float(secs)
    return out


def module_durations(text):
    """Секунды по модулям из блока `--durations 0`."""
    out = {}
    for test_id, secs in test_durations(text).items():
        mod = test_id.split(".")[0]
        out[mod] = out.get(mod, 0.0) + secs
    return out


def classify_tests(tests_dir):
    """Тесты, связанные с режимом legacy_files, по тексту тестов. -> {модуль.Класс.тест: "both" | "legacy"}.

    "both" — тест гоняет оба режима (`BACKENDS` или оба названия режимов в теле, в том числе через помощник
    класса или модуля); "legacy" — только файловый режим (название в теле или `backend = "legacy_files"` у класса)."""
    result = {}
    for path in sorted(Path(tests_dir).glob(PATTERN)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue

        def facts(fn):
            strings = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
            calls = {n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            calls |= {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            both = "BACKENDS" in names or ("legacy_files" in strings and "sqlite" in strings)
            return both, "legacy_files" in strings, calls

        module_funcs = {n.name: facts(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            pinned = any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "backend" for t in n.targets)
                         and isinstance(n.value, ast.Constant) and n.value.value == "legacy_files" for n in cls.body)
            funcs = {n.name: facts(n) for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            both_by_helper = {name for name, (b, _l, _c) in {**module_funcs, **funcs}.items() if b}
            for name, (b, legacy, calls) in funcs.items():
                if not name.startswith("test"):
                    continue
                if b or (calls & both_by_helper):
                    result[f"{path.stem}.{cls.name}.{name}"] = "both"
                elif legacy or pinned:
                    result[f"{path.stem}.{cls.name}.{name}"] = "legacy"
    return result


def build_table(test_times, tests_dir, together=()):
    """Таблица длительностей из секунд по тестам: `modules` (сейчас) и `after_legacy_removal` (оценка)."""
    kinds = classify_tests(tests_dir)
    now, after = {}, {}
    for test_id, secs in test_times.items():
        mod = test_id.split(".")[0]
        keep = {"both": SQLITE_COST / (1 + SQLITE_COST), "legacy": 0.0}.get(kinds.get(test_id), 1.0)
        now[mod] = now.get(mod, 0.0) + secs
        after[mod] = after.get(mod, 0.0) + secs * keep
    return {"modules": {m: round(v, 1) for m, v in sorted(now.items())},
            "after_legacy_removal": {m: round(after[m], 1) for m in sorted(after)},
            "together": [list(g) for g in together]}


def keep_old_seconds(table, seconds, after, modules):
    """Модуль набора, которого не было в журнале, остаётся с прежними числами; модуль, которого в наборе уже нет
    (удалён или переименован), из таблицы уходит."""
    for m in set(seconds) & set(modules) - set(table["modules"]):
        table["modules"][m] = round(seconds[m], 1)
        table["after_legacy_removal"][m] = round(after.get(m, seconds[m]), 1)
    for section in ("modules", "after_legacy_removal"):
        table[section] = {m: v for m, v in table[section].items() if m in modules}
    return table


def write_table(path, table):
    Path(path).write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


# ---------- процессы ----------

def group_env(sandbox, tests_dir):
    # данные — внутри временной папки: tests/test_data_safety проверяет, что папка данных лежит под TEMP
    temp = Path(sandbox) / "tmp"
    data = temp / DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    # кэши моделей — тоже в песочнице: тесты не должны видеть кэш установленной программы (защита в глубину к №163)
    cache = Path(sandbox) / "cache"
    env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=str(temp), TMP=str(temp),
               HF_HOME=str(cache / "huggingface"), TORCH_HOME=str(cache / "torch"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               GIT_OPTIONAL_LOCKS="0")
    env.pop("HF_HUB_CACHE", None)
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
        self.returncode = None
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
        if self.thread.is_alive():                 # внучатый процесс держит конец трубы: добить дерево, не ждать вечно
            kill_tree(self.proc)
            self.thread.join(timeout=10)
        self.returncode = self.proc.returncode
        closer = threading.Thread(target=self._close_stdout, daemon=True)
        closer.start()
        closer.join(timeout=10)                    # close() в Windows может зависнуть, пока кто-то читает трубу
        self.ended = time.time()
        self.summary = parse_summary(self.output)
        try:
            self.leftover = sorted(p.name for p in self.tmp.iterdir() if p.name != DATA_SUBDIR)
        except OSError:
            self.leftover = []

    def _close_stdout(self):
        try:
            self.proc.stdout.close()
        except (OSError, ValueError):
            pass

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
    "counts, classes = {}, {}\n"
    "def walk(suite):\n"
    "    for t in suite:\n"
    "        if isinstance(t, unittest.TestSuite):\n"
    "            walk(t)\n"
    "        else:\n"
    "            failed = type(t).__name__ == '_FailedTest'\n"
    "            mod = t._testMethodName if failed else type(t).__module__\n"
    "            counts[mod] = counts.get(mod, 0) + 1\n"
    "            if not failed:\n"
    "                cls = mod + '.' + type(t).__qualname__\n"
    "                classes[cls] = classes.get(cls, 0) + 1\n"
    "walk(unittest.defaultTestLoader.discover('tests'))\n"
    "print('@@COUNTS@@' + json.dumps(counts))\n"
    "print('@@CLASSES@@' + json.dumps(classes))\n")


def skipped_whole(text):
    """[(setUpClass|setUpModule, имя)] — классы и модули, пропущенные целиком: из подробного вывода группы."""
    return SETUP_SKIPPED.findall(text)


def shown_output(r, verbose):
    """Вывод группы для людей: без строк «… ok» подробного режима, который включён ради сверки (иначе — как есть)."""
    if verbose:
        return r.output
    return "\n".join(line for line in r.lines if not line.endswith(" ... ok")) + "\n"


def read_counts(run, marker="@@COUNTS@@"):
    for line in run.lines:
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker):])
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
                    help="случайный порядок модулей внутри групп (состав групп не меняется); без числа — seed выбирается и печатается")
    ap.add_argument("--sequential", action="store_true", help="один процесс: unittest discover -s tests")
    ap.add_argument("--timeout", type=int, default=GROUP_TIMEOUT, help="секунд на группу (по умолчанию %(default)s)")
    ap.add_argument("--table", help="таблица длительностей модулей (по умолчанию tests/parallel_durations.json)")
    ap.add_argument("--update-table", action="store_true", help="записать длительности модулей этого запуска в таблицу")
    ap.add_argument("--table-from", metavar="ЖУРНАЛ", help="пересчитать таблицу из сохранённого вывода запуска с "
                                                            "--durations (в том числе журнал CI) и выйти")
    ap.add_argument("--durations", action="store_true", help="группы печатают секунды по тестам (для --table-from)")
    ap.add_argument("--plan", action="store_true", help="напечатать раскладку и нагрузку групп, ничего не запуская")
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
    after = load_after(table_path)
    if args.table_from:
        times = test_durations(Path(args.table_from).read_text(encoding="utf-8", errors="replace"))
        if not times:
            print("в журнале нет блока «Slowest test durations»: запустите с --durations", file=out)
            return 2
        table = build_table(times, tests_dir, together)
        keep_old_seconds(table, seconds, after, modules)
        write_table(table_path, table)
        print(f"таблица пересчитана из журнала: {len(table['modules'])} модулей, {sum(table['modules'].values()):.0f} с "
              f"сейчас, {sum(table['after_legacy_removal'].values()):.0f} с после ухода legacy_files (оценка)", file=out)
        return 0
    if args.plan:
        plan = plan_groups(modules, seconds, jobs, together, seed, after)

        def load(g, table):
            return sum(table.get(m, seconds.get(m, 0.0)) for m in g)
        for i, g in enumerate(plan, 1):
            print(f"g{i}: модулей {len(g)}, сейчас {load(g, seconds):.0f} с, после ухода legacy_files "
                  f"{load(g, after):.0f} с (оценка)", file=out)
        print(f"самая длинная группа: сейчас {max(load(g, seconds) for g in plan):.0f} с, "
              f"после {max(load(g, after) for g in plan):.0f} с", file=out)
        return 0
    if args.sequential:
        groups = []
    else:
        groups = plan_groups(modules, seconds, jobs, together, seed, after)
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
    tree_before = tree_state(repo)
    extra = (["-v"] if args.verbose else []) + (["--durations", "0"] if args.update_table or args.durations else [])
    runs, counter, clean = [], None, False
    try:
        if args.sequential:
            runs.append(Run("seq", [py, "-B", "-m", "unittest", "discover", "-s", "tests", *extra], repo, tests_dir, root,
                            modules, args.verbose, say))
        else:
            for i, g in enumerate(groups, 1):
                runs.append(Run(f"g{i}", [py, "-B", "-m", "unittest", *([] if args.verbose else ["-v"]), *extra, *g], repo, tests_dir, root, g,
                                args.verbose, say))
        if not args.sequential and not args.no_count:
            counter = Run("count", [py, "-B", "-c", COUNT_CODE], repo, tests_dir, root, [], False, say)
        started = time.time()
        if seed is not None:
            say(f"порядок модулей в группах: seed={seed} (повтор: python -B tests/run_parallel.py --jobs {len(groups)} --shuffle {seed})")
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
                if r.returncode != 0 and r.summary["ok"]:
                    problems.append(f"{r.label}: итог unittest OK, но процесс вышел с кодом {r.returncode}")
                if r.leftover:
                    text = f"{r.label}: во временной папке группы остались {len(r.leftover)} объектов (первые: " \
                           f"{', '.join(r.leftover[:3])})"
                    if args.strict_tmp:
                        problems.append(text)
                    else:
                        say("предупреждение: " + text)
        if counter:
            counts, classes = read_counts(counter), read_counts(counter, "@@CLASSES@@")
            if counts is None or classes is None:
                problems.append("не удалось посчитать тесты через discover: " + (counter.output.strip()[-300:] or "нет вывода"))
            else:
                if sorted(counts) != modules:
                    problems.append("список модулей не совпал с найденным discover: только в discover "
                                    f"{sorted(set(counts) - set(modules))}, только у запускателя {sorted(set(modules) - set(counts))}")
                for r in runs:
                    want = sum(counts.get(m, 0) for m in r.modules)
                    skipped = 0                     # тесты классов и модулей, пропущенных целиком, в «Ran» не входят
                    for kind, name in skipped_whole(r.output):
                        skipped += counts.get(name, 0) if kind == "setUpModule" else classes.get(name, 0)
                    if r.summary and r.summary["ran"] + skipped != want:
                        problems.append(f"{r.label}: запущено {r.summary['ran']} тестов"
                                        + (f" и ещё {skipped} в пропущенных целиком классах или модулях" if skipped else "")
                                        + f", по discover у её модулей {want}")
        tree_after = tree_state(repo)
        if tree_before is None or tree_after is None:
            say("предупреждение: сверка «дерево репозитория не изменилось» не выполнена (нет Git или это не его копия)")
        elif tree_before != tree_after:
            changed = sorted(set(tree_before) ^ set(tree_after))
            problems.append("дерево репозитория изменилось за время запуска: " + "; ".join(changed[:5]))

        bad = [r for r in runs if r.timed_out or not r.summary or not r.summary["ok"] or r.returncode != 0]
        if args.update_table and not bad and not problems:
            times = {}
            for r in runs:
                times.update(test_durations(r.output))
            table = build_table(times, tests_dir, together)
            keep_old_seconds(table, seconds, after, modules)
            write_table(table_path, table)
            say(f"таблица длительностей записана: {table_path.name}")
        for r in bad:
            say(f"===== {r.label}: полный вывод упавшей группы =====")
            say(shown_output(r, args.verbose).rstrip("\n"))
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
        clean = not failed
        return 1 if failed else 0
    finally:
        for r in runs + ([counter] if counter else []):    # Ctrl-C или исключение: группы не остаются работать
            if r.proc is not None and r.proc.poll() is None:
                kill_tree(r.proc)
        if clean:
            shutil.rmtree(root, ignore_errors=True)
        else:                                              # AGENTS.md: песочница после неудачного прогона остаётся
            say(f"песочница оставлена для разбора: {root}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(run())
