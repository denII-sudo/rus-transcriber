"""Подтверждение опасных действий человеком — вне разговора с моделью.

Зачем. Параметры вроде confirm=… или apply=true модель подставляет сама, поэтому они не доказывают согласия
человека. Здесь решение принимается каналом, к которому у модели нет доступа:

  1. Окно на экране компьютера (Windows): текст окна формирует программа по своим данным (что именно будет
     сделано), а не модель; кнопку «Разрешить» нажимает человек. Нет ответа за confirm_timeout_sec секунд — отказ.
  2. Если окно показать нельзя (нет экрана, не Windows) — запрос в файл и команда человеку в терминале:
         python rt.py approve
     Она показывает запрос и спрашивает согласие на клавиатуре (только в настоящем терминале). После «да» тот же
     вызов, повторённый с теми же параметрами, выполняется один раз (разрешение живёт 5 минут).

Настройка в config.json (папка данных): "confirm_mode": "auto" (по умолчанию: окно, иначе терминал), "dialog"
(только окно), "terminal" (только терминал), "off" (без подтверждения — только для автоматических прогонов
и тестов; на живой машине не включайте). Каждое решение записывается в confirmations.log в папке данных.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import paths
import settings

TITLE = "rus-transcriber: подтвердите действие"
PENDING = paths.DATA_DIR / "confirm" / "pending.json"
APPROVED = paths.DATA_DIR / "confirm" / "approved.json"
LOG = paths.DATA_DIR / "confirmations.log"
TTL = 300           # сколько секунд живут запрос в терминал и выданное разрешение
MODES = ("auto", "dialog", "terminal", "off")


class NotConfirmed(Exception):
    """Человек не подтвердил (или подтверждение ещё не получено). Текст показывается клиенту как результат."""


def mode():
    m = settings.load_config().get("confirm_mode", "auto")
    return m if m in MODES else "auto"      # неизвестное значение — строгий режим, а не «выключено»


def timeout_sec():
    try:
        return max(10, int(settings.load_config().get("confirm_timeout_sec", 180)))
    except (TypeError, ValueError):
        return 180


def key_of(action, args):
    """Отпечаток действия с параметрами: разрешение действует только на то, что человек видел."""
    blob = json.dumps({"a": action, "p": args}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def clean(text, limit=1400):
    """Текст для окна: без управляющих символов, строки и общая длина ограничены."""
    lines = []
    for raw in str(text).splitlines():
        s = "".join(ch for ch in raw if ch == "\t" or ch >= " ")[:220]
        lines.append(s)
    out = "\n".join(lines)
    return out if len(out) <= limit else out[:limit] + "\n…"


def _log(action, decision, how):
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action,
                                "decision": decision, "how": how}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _read(p):
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


# ---------- канал 1: окно на экране ----------
_WIN_SCRIPT = (
    "import ctypes, sys\n"
    "text = sys.stdin.buffer.read().decode('utf-8')\n"
    "title = sys.argv[1]\n"
    # MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2 («Нет» по умолчанию) | MB_SETFOREGROUND | MB_TOPMOST
    "r = ctypes.windll.user32.MessageBoxW(0, text, title, 0x4 | 0x30 | 0x100 | 0x10000 | 0x40000)\n"
    "sys.exit(0 if r == 6 else 1)\n"
)


def dialog_available():
    return os.name == "nt"


def ask_dialog(text, timeout):
    """True — человек нажал «Да»; False — «Нет», закрыл окно или не ответил за timeout секунд."""
    body = clean(text) + "\n\nРазрешить? (Нет — действие не выполняется.)"
    try:
        r = subprocess.run([sys.executable, "-c", _WIN_SCRIPT, TITLE], input=body.encode("utf-8"),
                           timeout=timeout, creationflags=0x08000000, capture_output=True)
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        return False
    return r.returncode == 0


# ---------- канал 2: терминал человека ----------
def _prune(d):
    now = time.time()
    return {k: v for k, v in d.items() if isinstance(v, dict) and v.get("expires", 0) > now}


def request_terminal(key, action, text):
    d = _prune(_read(PENDING))
    d[key] = {"action": action, "text": clean(text), "expires": time.time() + TTL}
    _write(PENDING, d)


def consume(key):
    """Есть ли действующее разрешение на это действие; разрешение одноразовое."""
    d = _prune(_read(APPROVED))
    if key not in d:
        return False
    d.pop(key)
    _write(APPROVED, d)
    return True


def pending_requests():
    return _prune(_read(PENDING))


def approve(key):
    """Вызывается rt.py approve после «да» человека в терминале."""
    p = _prune(_read(PENDING))
    if key not in p:
        return False
    a = _prune(_read(APPROVED))
    a[key] = {"action": p[key]["action"], "expires": time.time() + TTL}
    _write(APPROVED, a)
    p.pop(key)
    _write(PENDING, p)
    return True


def reject(key):
    p = _prune(_read(PENDING))
    p.pop(key, None)
    _write(PENDING, p)


# ---------- вход для слоя управления ----------
def require(action, args, summary_lines):
    """Возвращает способ подтверждения ('dialog', 'terminal', 'off') или бросает NotConfirmed.

    action — имя действия; args — его параметры (по ним строится отпечаток); summary_lines — что будет сделано,
    словами программы: именно это видит человек."""
    m = mode()
    text = "\n".join(summary_lines)
    if m == "off":
        _log(action, "разрешено без подтверждения (confirm_mode=off)", "off")
        return "off"
    key = key_of(action, args)
    if consume(key):
        _log(action, "разрешено", "terminal")
        return "terminal"
    if m in ("auto", "dialog") and dialog_available():
        if ask_dialog(f"{action}\n\n{text}", timeout_sec()):
            _log(action, "разрешено", "dialog")
            return "dialog"
        _log(action, "отказ или нет ответа", "dialog")
        raise NotConfirmed("Человек не подтвердил действие (нажал «Нет», закрыл окно или не ответил за "
                           f"{timeout_sec()} с). Ничего не выполнено. Не повторяйте вызов без новой просьбы человека.")
    if m == "dialog":
        raise NotConfirmed("Окно подтверждения здесь показать нельзя (confirm_mode=dialog). Ничего не выполнено.")
    request_terminal(key, action, text)
    _log(action, "запрос в терминал", "terminal")
    raise NotConfirmed("Нужно подтверждение человека. Попросите его открыть терминал в папке программы и выполнить "
                       "`python rt.py approve` — он увидит, что именно будет сделано, и ответит на клавиатуре. "
                       "После этого повторите вызов с теми же параметрами (разрешение одноразовое, 5 минут). "
                       "Подтвердить за человека нельзя.")
