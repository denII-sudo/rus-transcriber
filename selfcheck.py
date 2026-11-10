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
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS_TIMEOUT = 600
MORPH_SAMPLE = ["Если", "Есть", "Тетрадями", "Quartzline", "Москва"]
MORPH_EXPECT = ["Quartzline", "Москва"]


def parse_unittest(text):
    """Итог unittest из его вывода: число тестов, падения, ошибки, пропуски, имена упавших."""
    ran = re.search(r"Ran (\d+) tests? in ([\d.]+)s", text)
    res = {"ran": int(ran.group(1)) if ran else 0, "seconds": float(ran.group(2)) if ran else None,
           "ok": bool(re.search(r"^OK\b", text, re.M)), "failures": 0, "errors": 0, "skipped": 0}
    m = re.search(r"^(?:OK|FAILED) \(([^)]*)\)", text, re.M)
    if m:
        for k, v in re.findall(r"(\w+)=(\d+)", m.group(1)):
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


def run_tests(timeout=TESTS_TIMEOUT):
    data = tempfile.mkdtemp(prefix="rt-data-")           # своя папка данных: тесты не видят настоящую
    env = dict(os.environ, RUS_TRANSCRIBER_DATA=data, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    try:
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
        if sys.version_info >= (3, 12):
            cmd += ["--durations", "5"]                # самые медленные тесты — в ответ
        r = subprocess.run(cmd, cwd=str(HERE), env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                           creationflags=0x08000000 if os.name == "nt" else 0)
        res = parse_unittest(r.stderr + "\n" + r.stdout)
        res["exit_code"] = r.returncode
        if not res["ran"]:
            res["output_tail"] = (r.stderr or r.stdout)[-800:]
    except subprocess.TimeoutExpired:
        res = {"ran": 0, "ok": False, "error": f"тесты не уложились в {timeout} с"}
    except OSError as ex:
        res = {"ran": 0, "ok": False, "error": f"не удалось запустить тесты: {ex}"}
    finally:
        shutil.rmtree(data, ignore_errors=True)
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
    return {"ok": bool(words) and not problems, "words": len(words), "violations": len(problems),
            "where": mask(problems)[:20], "mode": "git" if is_git else "папка без git",
            **({"note": "список запрещённых слов пуст — проверка слов не выполнена"} if not words else {})}


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
