r"""Оценка качества расшифровки по эталону: WER и точность подписи голосов, пороги, история прогонов.

Эталон — текстовый файл со строками  [мм:сс] Имя: текст  (или [ч:мм:сс]); остальные строки игнорируются,
[пометки в квадратных скобках] внутри реплики не учитываются.

    python tools/eval_quality.py score расшифровка.transcript.json эталон.txt [--limit 300] [--max-wer 16] [--min-voice 93]
    python tools/eval_quality.py init имя запись.mp3 эталон.txt --names "Анна, Борис" [--group g] [--limit 300] [--keep-temp]
    python tools/eval_quality.py run [--case имя] [--keep-temp]   # прогон worker.py по случаям и сравнение с порогами
    python tools/eval_quality.py list

Случаи хранятся вне репозитория: <папка данных>\eval\cases.json (там пути к вашим записям и эталонам), история
прогонов — <папка данных>\eval\history.jsonl. `init` запускает расшифровку один раз и записывает базовые значения;
пороги = базовое значение с запасом (--margin-wer 1.0, --margin-voice 1.5 процентных пункта). `run` и `init`
требуют HF_TOKEN в окружении и запускать их надо питоном из .venv программы. Код возврата: 0 — в порогах,
1 — хуже порогов, 2 — ошибка запуска.
Зачем: любая правка в распознавании, голосовании, чанках или профилях проверяется на одних и тех же записях,
а не на глаз. Время (total_sec) сравнивается с базой: рост больше чем на 30 % — предупреждение.

Где идёт прогон (`run`, `init` без --transcript): в своей временной папке на каждый прогон. Туда копируются
аудио, словарь рядом с ним, config.json (без выбора хранения) и словари групп (vocab.json); создаётся новая
база записей. Рабочая база, настоящий голосовой профиль и папка с настоящим аудио не меняются. Голосовые
профили (биометрия) не копируются, поэтому прогон идёт без голосов (--no-voices): точность голосов не
измеряется и порог голосов не проверяется. Папка удаляется после прогона, в том числе неудачного;
--keep-temp оставляет её для разбора (путь печатается).
`score` принимает готовую стенограмму обоих видов: выгрузку записи базы и прежний файл.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"\[(?:(\d+):)?(\d+):(\d+)\]\s*([^:\n]+?):\s*(.*)")


def data_dir():
    env = os.environ.get("RUS_TRANSCRIBER_DATA", "").strip()
    if env:
        return Path(env)
    if os.name == "nt":
        return Path.home() / "AppData" / "Roaming" / "rus-transcriber-data"
    return Path.home() / ".local" / "share" / "rus-transcriber-data"


EVAL_DIR = data_dir() / "eval"


def norm(w):
    return re.sub(r"[^0-9a-zа-яё]", "", w.lower().replace("ё", "е"))


def load_reference(path):
    """-> [(слово, говорящий, секунды)]"""
    out = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        h, mi, se = int(m[1] or 0), int(m[2]), int(m[3])
        t = h * 3600 + mi * 60 + se
        for w in re.sub(r"\[[^\]]*\]", " ", m[5]).split():
            if norm(w):
                out.append((norm(w), m[4].strip(), t))
    return out


def load_hypothesis(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return hypothesis_words(d), d


def hypothesis_words(d):
    """-> [(слово, говорящий, секунды)] из выгрузки записи базы ({"transcript": {...}}) или прежнего файла."""
    body = d["transcript"] if isinstance(d.get("transcript"), dict) else d
    out = []
    for s in body["segments"]:
        for w in s.get("words") or []:
            for x in w["w"].split():
                if norm(x):
                    out.append((norm(x), s["speaker"], w["s"]))
    return out


def score(ref, hyp, limit=None):
    """WER (подстановки/пропуски/вставки) и доля слов с верным говорящим среди совпавших."""
    if limit:
        ref = [r for r in ref if r[2] <= limit]
        hyp = [h for h in hyp if h[2] <= limit]
    if not ref:
        raise ValueError("в эталоне нет слов (проверьте формат строк «[мм:сс] Имя: текст» и --limit)")
    sm = difflib.SequenceMatcher(None, [r[0] for r in ref], [h[0] for h in hyp], autojunk=False)
    S = D = I = ok = tot = 0
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            for k in range(a2 - a1):
                tot += 1
                ok += ref[a1 + k][1] == hyp[b1 + k][1]
        elif op == "replace":
            S += min(a2 - a1, b2 - b1)
            D += max(0, (a2 - a1) - (b2 - b1))
            I += max(0, (b2 - b1) - (a2 - a1))
        elif op == "delete":
            D += a2 - a1
        else:
            I += b2 - b1
    return {"wer": round(100 * (S + D + I) / len(ref), 2), "S": S, "D": D, "I": I, "ref_words": len(ref),
            "voice": round(100 * ok / tot, 2) if tot else 0.0, "voice_words": tot}


def code_id():
    h = hashlib.sha1()
    for p in sorted(ROOT.glob("*.py")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()[:10]


def _cases_path():
    return EVAL_DIR / "cases.json"


def load_cases():
    p = _cases_path()
    return json.loads(p.read_text(encoding="utf-8"))["cases"] if p.exists() else []


def save_cases(cases):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    _cases_path().write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")


# создать базу в папке данных прогона и захватить автора — как core.start_transcription в момент запуска
_PREPARE = (
    "import json, recordings\n"
    "report = recordings.prepare_installation_store()\n"
    "if not report['ok']:\n"
    "    raise SystemExit(report.get('message'))\n"
    "store = recordings.open_store()\n"
    "try:\n"
    "    print(json.dumps(recordings.operation_context(store, 'model')))\n"
    "finally:\n"
    "    store.close()\n")


def _run(cmd, env):
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def _prepare_data_dir(tmp):
    """Папка данных прогона: config.json без выбора хранения и словари групп. Голосовых профилей нет."""
    data = tmp / "data"
    data.mkdir()
    real = data_dir()
    config = real / "config.json"
    if config.is_file():
        cfg = json.loads(config.read_text(encoding="utf-8-sig"))
        cfg.pop("recording_backend", None)          # хранение в базе выбирает создание новой базы
        (data / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    groups = real / "groups"
    if groups.is_dir():
        for g in sorted(groups.iterdir()):
            if (g / "vocab.json").is_file():
                (data / "groups" / g.name).mkdir(parents=True)
                shutil.copy2(g / "vocab.json", data / "groups" / g.name / "vocab.json")
    return data


def transcribe(case, tag="eval", keep=False):
    """Прогон worker.py в своей временной папке. -> (выгрузка записи базы, timing, оставленная папка или None).

    Папка удаляется и после неудачи; keep=True оставляет её для разбора."""
    audio = Path(case["audio"])
    if not audio.is_file():
        raise RuntimeError(f"нет аудио: {audio}")
    tag = "".join(ch for ch in tag if ch.isalnum() or ch in "-_")
    tag = f"{tag or 'eval'}-{uuid.uuid4().hex}"
    tmp = Path(tempfile.mkdtemp(prefix="rt-eval-"))
    try:
        data = _prepare_data_dir(tmp)
        (tmp / "tmp").mkdir()
        (tmp / "audio").mkdir()
        local = tmp / "audio" / audio.name
        shutil.copy2(audio, local)
        near = audio.parent / "rus-transcriber.vocab.json"     # словарь рядом с записью (vocab.FILE_NAME)
        if near.is_file():
            shutil.copy2(near, tmp / "audio" / near.name)
        env = {**os.environ, "PYTHONUTF8": "1", "RUS_TRANSCRIBER_DATA": str(data),
               "TEMP": str(tmp / "tmp"), "TMP": str(tmp / "tmp")}
        r = _run([sys.executable, "-B", "-c", _PREPARE], env)
        lines = (r.stdout or "").strip().splitlines()
        if r.returncode != 0 or not lines:
            raise RuntimeError(f"база прогона не создана (код {r.returncode}): {(r.stderr or r.stdout)[-600:]}")
        status = tmp / "status.json"
        status.write_text(json.dumps({"recording_context": json.loads(lines[-1])}), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "worker.py"), str(local), "--tag", tag, "--status", str(status),
               "--no-voices"]
        if case.get("names"):
            cmd += ["--names", case["names"]]
        if case.get("group"):
            cmd += ["--group", case["group"]]
        if not case.get("emotions", False):
            cmd.append("--no-emotions")
        if case.get("whisper"):
            cmd += ["--whisper", case["whisper"]]
        r = _run(cmd, env)
        try:
            st = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        out = Path(st["json"]) if st.get("json") else None
        if r.returncode != 0 or st.get("state") != "done" or out is None or not out.is_file():
            raise RuntimeError(f"расшифровка не удалась (код {r.returncode}): "
                               f"{st.get('error') or (r.stderr or r.stdout)[-600:]}")
        export = json.loads(out.read_text(encoding="utf-8"))
        return export, st.get("timing") or {}, (tmp if keep else None)
    finally:
        if keep:
            print(f"Папка прогона оставлена для разбора: {tmp}", file=sys.stderr)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            if tmp.exists():
                print(f"Папку прогона не удалось удалить: {tmp}", file=sys.stderr)


def evaluate(case, transcript_json=None, tag="eval", keep=False):
    if transcript_json:
        hyp, d = load_hypothesis(transcript_json)
        t, source, voices = d.get("timing", {}), str(transcript_json), True
    else:
        d, t, kept = transcribe(case, tag, keep)
        hyp, source, voices = hypothesis_words(d), (str(kept) if kept else ""), False
    res = score(load_reference(case["reference"]), hyp, case.get("limit_sec"))
    if not voices:                    # прогон без голосовых профилей: подписи голосов не сравниваются
        res.update(voice=None, voice_words=0)
    res.update(total_sec=t.get("total_sec"), rtf=t.get("realtime_factor"), transcript=source, code=code_id())
    return res


def verdict(case, res):
    problems, warns = [], []
    if case.get("max_wer") is not None and res["wer"] > case["max_wer"]:
        problems.append(f"WER {res['wer']}% выше порога {case['max_wer']}%")
    if case.get("min_voice") is not None and res["voice"] is None:
        warns.append("голоса не проверялись: прогон без голосовых профилей")
    elif case.get("min_voice") is not None and res["voice"] < case["min_voice"]:
        problems.append(f"точность голосов {res['voice']}% ниже порога {case['min_voice']}%")
    base = (case.get("baseline") or {}).get("total_sec")
    if base and res.get("total_sec") and res["total_sec"] > base * 1.3:
        warns.append(f"время {res['total_sec']} с против базовых {base} с (+{round(100 * res['total_sec'] / base - 100)}%)")
    return problems, warns


def log_history(name, res):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "case": name,
                            **{k: v for k, v in res.items() if k != "transcript"}}, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("transcript"); s.add_argument("reference")
    s.add_argument("--limit", type=float, default=None); s.add_argument("--max-wer", type=float); s.add_argument("--min-voice", type=float)
    i = sub.add_parser("init"); i.add_argument("name"); i.add_argument("audio"); i.add_argument("reference")
    i.add_argument("--names", default=""); i.add_argument("--group", default=""); i.add_argument("--limit", type=float, default=None)
    i.add_argument("--margin-wer", type=float, default=1.0); i.add_argument("--margin-voice", type=float, default=1.5)
    i.add_argument("--transcript", default="", help="взять готовую расшифровку вместо запуска")
    i.add_argument("--keep-temp", action="store_true", help="оставить временную папку прогона для разбора")
    r = sub.add_parser("run"); r.add_argument("--case", default="")
    r.add_argument("--keep-temp", action="store_true", help="оставить временную папку прогона для разбора")
    sub.add_parser("list")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "score":
            res = score(load_reference(a.reference), load_hypothesis(a.transcript)[0], a.limit)
            print(json.dumps(res, ensure_ascii=False))
            bad = (a.max_wer is not None and res["wer"] > a.max_wer) or (a.min_voice is not None and res["voice"] < a.min_voice)
            return 1 if bad else 0
        if a.cmd == "list":
            for c in load_cases():
                b = c.get("baseline") or {}
                print(f"{c['name']}: база WER {b.get('wer')}% / голоса {b.get('voice')}% / {b.get('total_sec')} с; "
                      f"пороги WER ≤ {c.get('max_wer')}%, голоса ≥ {c.get('min_voice')}%")
            return 0
        if a.cmd == "init":
            case = {"name": a.name, "audio": a.audio, "reference": a.reference, "names": a.names, "group": a.group,
                    "limit_sec": a.limit, "emotions": False}
            res = evaluate(case, a.transcript or None, keep=a.keep_temp)
            case["baseline"] = {k: res[k] for k in ("wer", "voice", "total_sec", "rtf", "code")}
            case["max_wer"] = round(res["wer"] + a.margin_wer, 1)
            case["min_voice"] = None if res["voice"] is None else round(res["voice"] - a.margin_voice, 1)
            cases = [c for c in load_cases() if c["name"] != a.name] + [case]
            save_cases(cases)
            log_history(a.name, res)
            print(json.dumps({"case": a.name, **{k: case[k] for k in ("baseline", "max_wer", "min_voice")}}, ensure_ascii=False))
            return 0
        cases = [c for c in load_cases() if not a.case or c["name"] == a.case]
        if not cases:
            print("Нет случаев: создайте через init (cases.json в папке данных, подпапка eval).", file=sys.stderr)
            return 2
        worst = 0
        for c in cases:
            res = evaluate(c, keep=a.keep_temp)
            problems, warns = verdict(c, res)
            log_history(c["name"], res)
            voice = "не проверялись" if res["voice"] is None else f"{res['voice']}%"
            print(f"{c['name']}: WER {res['wer']}% (порог ≤ {c.get('max_wer')}), голоса {voice} "
                  f"(порог ≥ {c.get('min_voice')}), {res.get('total_sec')} с, код {res['code']}")
            for p in problems:
                print("  НАРУШЕНИЕ:", p)
            for w in warns:
                print("  внимание:", w)
            worst = max(worst, 1 if problems else 0)
        return worst
    except (RuntimeError, ValueError, OSError) as ex:
        print(f"Ошибка: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
