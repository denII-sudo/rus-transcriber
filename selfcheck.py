"""Самопроверка программы на этом компьютере: тесты, проверка файлов на личные данные, морфология.

Только проверки, произвольных команд нет. Тесты идут в отдельном процессе с отдельной временной папкой данных
(настоящие данные не трогаются), проверка файлов только читает. В ответе нет запрещённых слов: только где
нарушение и какого оно вида.

    python selfcheck.py            # то же из командной строки
"""
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS_TIMEOUT = 720            # 12 мин: предел, о котором пишет шаг 11 install.ps1 (решение владельца 24.11.2026)
JOBS_ENV = "RT_SELFCHECK_JOBS"  # число процессов с тестами: 1 — по-старому, одним процессом; без значения — по ядрам (до 4)
GROUP_MARGIN = 60              # запас: группы запускателя должны закончиться (и быть убиты) раньше предела самопроверки
MORPH_SAMPLE = ["Если", "Есть", "Тетрадями", "Quartzline", "Москва"]
MORPH_EXPECT = ["Quartzline", "Москва"]


def parse_unittest(text):
    """Итог unittest из его вывода: число тестов, падения, ошибки, пропуски, имена упавших.

    expected_failures — тесты с @unittest.expectedFailure (записанные известные ограничения),
    они не падения; unexpected_successes — такой тест вдруг прошёл (unittest считает это неуспехом,
    имён в обычном выводе не печатает: их показывает запуск тестов с -v).
    """
    ran = re.search(r"Ran (\d+) tests? in ([\d.]+)s", text)
    res = {"ran": int(ran.group(1)) if ran else 0, "seconds": float(ran.group(2)) if ran else None,
           "ok": bool(re.search(r"^OK\b", text, re.M)), "failures": 0, "errors": 0, "skipped": 0,
           "expected_failures": 0, "unexpected_successes": 0}
    m = re.search(r"^(?:OK|FAILED) \(([^)]*)\)", text, re.M)
    if m:
        # Название поля — целиком, с пробелами: «expected failures=2» не должно попасть в failures
        for k, v in re.findall(r"([a-z][a-z ]*)=(\d+)", m.group(1)):
            k = k.strip().replace(" ", "_")
            if k in res:
                res[k] = int(v)
    failed = []
    for block in re.split(r"^=+\n", text, flags=re.M)[1:]:
        head = block.splitlines()[0] if block.strip() else ""
        hm = re.match(r"(FAIL|ERROR): (\S+) \(([^)]+)\)", head)
        if not hm:
            continue
        body = block.split("\n" + "-" * 70, 1)[-1]
        last = [l for l in body.strip().splitlines() if l.strip() and not l.startswith(("Ran ", "OK", "FAILED"))]
        why = next((l for l in reversed(last) if re.match(r"\s*\w*(Error|Exception)\b", l)), last[-1] if last else "")
        failed.append({"kind": hm.group(1), "test": f"{hm.group(3)}", "why": why.strip()[:300]})
    slow = re.findall(r"^\s*([\d.]+)s\s+(\S+) \(([^)]+)\)", text.split("Slowest test durations", 1)[-1], re.M) \
        if "Slowest test durations" in text else []
    if slow:
        res["slowest"] = [{"sec": float(t), "test": full} for t, _, full in slow[:5]]
    res["failed"] = failed[:10]
    if len(failed) > 10:
        res["failed_more"] = len(failed) - 10
    return res


def parse_parallel(text):
    """Итог tests/run_parallel.py: счёт — из его итоговых строк (последние «Ran…» и «OK/FAILED»), упавшие тесты —
    из полного вывода упавших групп, которые запускатель печатает выше итога; нарушения запуска — отдельным списком."""
    cut = text.rfind("\nRan ")
    head, tail = (text[:cut], text[cut:]) if cut >= 0 else ("", text)
    res = parse_unittest(tail)
    res["failed"] = parse_unittest(head)["failed"][:10]
    res.pop("failed_more", None)
    problems = [l.strip()[:300] for l in text.splitlines() if l.startswith(("НАРУШЕНИЕ:", "РАСКЛАДКА:"))]
    if problems:
        res["run_problems"] = problems[:5]
    groups = re.search(r"^групп (\d+)", tail, re.M)
    if groups:
        res["groups"] = int(groups.group(1))
    return res


def _jobs_setting():
    """Сколько процессов с тестами: None — один (по настройке или на одном ядре), 0 — по ядрам, N ≥ 2 — столько групп."""
    raw = os.environ.get(JOBS_ENV, "").strip()
    if raw.isdigit():
        return None if int(raw) <= 1 else int(raw)
    return None if (os.cpu_count() or 1) < 2 else 0


def _kill_tree(proc):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=60)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_tree(cmd, cwd, env, timeout):
    """Запуск с выводом; при истечении срока убивается всё дерево процессов (запускатель и его группы). -> (код, текст)."""
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", **kwargs)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=30)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        raise
    return proc.returncode, out


def _run_tests_parallel(timeout, jobs):
    """Набор по группам модулей (tests/run_parallel.py). -> (результат, None) или (None, причина): тогда — последовательно."""
    launcher = HERE / "tests" / "run_parallel.py"
    if not launcher.is_file():
        return None, "нет tests/run_parallel.py"
    cmd = [sys.executable, "-B", str(launcher), "--timeout", str(max(60, timeout - GROUP_MARGIN))]
    if jobs:
        cmd += ["--jobs", str(jobs)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        code, text = _run_tree(cmd, HERE, env, timeout)
    except subprocess.TimeoutExpired:
        return {"ran": 0, "ok": False, "error": f"тесты не уложились в {timeout // 60} мин ({timeout} с)"}, None
    except OSError as ex:
        return None, f"не удалось запустить параллельный запуск: {ex}"
    res = parse_parallel(text)
    res["exit_code"] = code
    res["ok"] = bool(res["ok"]) and code == 0
    res["mode"] = "parallel"
    if not res["ran"]:                                       # запускатель не дошёл до итога: раскладка, запуск, вывод
        return None, f"параллельный запуск не дал итога (код {code}): {text.strip()[-300:]}"
    return res, None


def _run_tests_sequential(timeout):
    with tempfile.TemporaryDirectory(prefix="rt-selfcheck-") as sandbox:
        root = Path(sandbox)
        data, temp = root / "data", root / "tmp"
        data.mkdir()
        temp.mkdir()
        env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=str(temp), TMP=str(temp),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        try:
            cmd = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"]
            if sys.version_info >= (3, 12):
                cmd += ["--durations", "5"]                # самые медленные тесты — в ответ
            # stdin=DEVNULL: тестам не нужен ввод; из терминала (например, шаг 11 установщика)
            # тест, ждущий ответа человека, иначе зависал бы на скрытом вопросе
            r = subprocess.run(cmd, cwd=str(HERE), env=env, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                               creationflags=0x08000000 if os.name == "nt" else 0)
            res = parse_unittest(r.stderr + "\n" + r.stdout)
            res["exit_code"] = r.returncode
            res["mode"] = "sequential"
            if not res["ran"]:
                res["output_tail"] = (r.stderr or r.stdout)[-800:]
        except subprocess.TimeoutExpired:
            res = {"ran": 0, "ok": False, "error": f"тесты не уложились в {timeout // 60} мин ({timeout} с)"}
        except OSError as ex:
            res = {"ran": 0, "ok": False, "error": f"не удалось запустить тесты: {ex}"}
    return res


def run_tests(timeout=TESTS_TIMEOUT):
    """Тесты программы. По умолчанию — параллельно, группами модулей (tests/run_parallel.py, до 4 процессов; у каждой
    группы своя песочница). RT_SELFCHECK_JOBS=1 или один процессор — прежний путь: один процесс `unittest discover`.
    Если параллельный запуск не дошёл до итога (нет файла, не запустился, раскладка), тесты идут последовательно
    на оставшееся время, причина — в поле fallback. Падения тестов в параллельном запуске откатом не «лечатся»."""
    t0 = time.time()
    res, why = None, None
    jobs = _jobs_setting()
    if jobs is not None:
        res, why = _run_tests_parallel(timeout, jobs)
    if res is None:
        left = timeout - (time.time() - t0)
        if why and left < 120:
            res = {"ran": 0, "ok": False, "error": f"параллельный запуск не удался, на последовательный не осталось времени: {why}"[:300]}
        else:
            res = _run_tests_sequential(int(left) if why else timeout)
            if why:
                res["fallback"] = why[:400]
    res["wall_sec"] = round(time.time() - t0, 1)
    return res


def mask(problems):
    """Нарушения без самих слов: где и какого вида."""
    out = []
    for where, why in problems:
        if why.startswith("запрещённое слово"):
            why = "запрещённое слово (само слово не показывается)"
        elif why.startswith("путь с именем пользователя"):
            why = "путь с именем пользователя"
        out.append({"where": where, "why": why})
    return out


def run_check_repo():
    p = HERE / "tools" / "check_repo.py"
    if not p.exists():
        return {"ok": False, "error": "нет tools/check_repo.py"}
    try:
        spec = importlib.util.spec_from_file_location("rt_check_repo", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        problems, words, is_git = m.check(False)
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}"[:300]}
    if not words:
        # на домашней установке приватный список запрещённых слов недоступен: это не поломка.
        # Проверка путей и формата выполнена; проверку слов владелец делает на своей машине.
        return {"ok": not problems, "partial": True, "words": 0, "violations": len(problems),
                "where": mask(problems)[:20], "mode": "git" if is_git else "папка без git",
                "note": "список запрещённых слов пуст — проверка слов не выполнена (проверка путей выполнена)"}
    return {"ok": not problems, "words": len(words), "violations": len(problems),
            "where": mask(problems)[:20], "mode": "git" if is_git else "папка без git"}


def run_morph():
    try:
        import pymorphy3  # noqa: F401
    except Exception as ex:
        return {"ok": False, "pymorphy3": False, "error": f"pymorphy3 не установлен: {ex}"[:200]}
    try:
        import transcript
        got = [w for w in MORPH_SAMPLE if transcript.vocab_candidate(w)]
    except Exception as ex:
        return {"ok": False, "pymorphy3": True, "error": f"{type(ex).__name__}: {ex}"[:300]}
    return {"ok": got == MORPH_EXPECT, "pymorphy3": True, "got": got, "expected": MORPH_EXPECT}


def run(version=None, tests=True, repo=True, morph=True):
    out = {"version": version, "python": platform.python_version(), "program": HERE.name}
    try:
        import buildinfo
        out["build"] = buildinfo.identity(HERE)
    except Exception:
        out["build"] = {"release": None, "commit": None, "source_state": "unknown",
                        "payload_sha256": None, "manifest_status": "unknown"}
    if tests:
        out["tests"] = run_tests()
    if repo:
        out["check_repo"] = run_check_repo()
    if morph:
        out["morphology"] = run_morph()
    parts = [k for k in ("tests", "check_repo", "morphology") if k in out]
    out["ok"] = all(out[k].get("ok") for k in parts)
    bad = [k for k in parts if not out[k].get("ok")]
    out["summary"] = "всё в порядке" if out["ok"] else "не в порядке: " + ", ".join(bad)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    r = run()
    print(json.dumps(r, ensure_ascii=False, indent=1))
    sys.exit(0 if r["ok"] else 1)
