"""Паспорт прогона теста скорости: машина, окружение, условия запуска, звук и модели.

Модуль только читает: ничего не меняет в системе и не загружает модели. Нужен, чтобы два
прогона можно было сопоставить — см. docs/bench-database-spec.md, термины — docs/bench-glossary.md.

Правила:
  * только стандартная библиотека; остальная программа не импортируется (кроме jobslots — по месту);
  * сведение, которое не удалось получить, — None или {"status": "not_measured", "reason": ...};
    прогон из-за этого не падает, а причина записывается без текста исключения (в нём бывают пути);
  * в паспорт не попадают имя компьютера, имя пользователя, пути и имена файлов. Путь к записи
    хранит только локальный указатель units.json (update_units_index); в выгрузку он не идёт;
  * файлы прогонов не перезаписываются (write_run).

Windows — основная платформа; на остальных часть сведений остаётся неизмеренной.
"""
import ast
import base64
import hashlib
import json
import os
import platform
import re
import secrets
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

SCHEMA_VERSION = 1
# Методика замера bench.py на момент введения паспорта: один проход по отрывку, первый кусок —
# прогрев, load_sec включает прогрев. Меняется методика — меняется номер; числа разных версий
# между собой не сравниваются.
METHOD_VERSION = "bench-1"

KEY_PACKAGES = ("onnx-asr", "onnxruntime", "onnxruntime-openvino", "onnxruntime-directml", "openvino",
                "torch", "torchaudio", "ctranslate2", "faster-whisper", "pyannote-audio", "gigaam",
                "numpy", "onnx", "huggingface-hub", "soundfile", "tokenizers")

# стартовые пороги готовности к замеру (docs/bench-database-spec.md, раздел 8.1); прогон они
# не останавливают, а только помечают
LOAD_WARN_PERCENT = 15.0
FREE_RAM_WARN_MB = 4096
DOWNLOAD_BYTES = 1_000_000

POWER_SCHEMES = {"381b4222-f694-41f0-9685-ff5bb260df2e": "balanced",
                 "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "high_performance",
                 "a1841308-3541-4fab-bc81-f71556f20b4a": "power_saver",
                 "e9a42b02-d5df-448d-aa00-03f14749eb61": "ultimate_performance"}
POWER_MODES = {"00000000-0000-0000-0000-000000000000": "balanced",
               "961cc777-2547-4f9d-8174-7d86181b8a7a": "best_efficiency",
               "ded574b5-45a0-4f42-8737-46345c09c238": "best_performance"}

REASONS_RU = {"battery": "питание от батареи: Windows замедляет программы, числа не сравнимы с замерами от сети",
              "power_source_changed": "источник питания сменился во время прогона",
              "cpu_load": "процессор был занят другими программами перед стартом",
              "low_memory": "мало свободной памяти перед стартом",
              "other_heavy_jobs": "одновременно шла другая тяжёлая задача программы",
              "model_downloaded": "во время прогона скачивалась модель: время загрузки завышено"}

_GUID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def utc_now(ts=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def not_measured(reason):
    return {"status": "not_measured", "reason": str(reason)[:120]}


def _run(cmd, timeout=10, oem=False):
    """Вывод команды или None. Окно консоли не показывается: тест может идти из-под MCP-сервера."""
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    enc = "oem" if oem and os.name == "nt" else "utf-8"
    try:
        return p.stdout.decode(enc, errors="replace")
    except LookupError:
        return p.stdout.decode("utf-8", errors="replace")


def _write_json(path, data):
    """Запись через временный файл рядом: читатель не увидит половину файла."""
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ── звук ─────────────────────────────────────────────────────────────────────────────────────────

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def audio_identity(path, pcm_bytes, sample_rate, source_samples=None, offset_sec=0.0):
    """Что за звук измерялся — без имени и пути. pcm_bytes — отрывок: моно, s16le, sample_rate.

    Контрольная сумма файла доказывает те же байты, сумма отрывка — тот же звук и после смены
    контейнера. unit_id — короткий идентификатор тестовой единицы (отрывка)."""
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        size = None
    pcm = hashlib.sha256(pcm_bytes).hexdigest()
    return {"unit_id": pcm[:12],
            "audio_sha256": sha256_file(p),
            "size_bytes": size,
            "source_format": p.suffix.lower().lstrip(".") or None,
            "source_duration_sec": (round(source_samples / sample_rate, 2)
                                    if source_samples is not None else None),
            "excerpt": {"offset_sec": round(float(offset_sec), 2),
                        "duration_sec": round(len(pcm_bytes) / 2 / sample_rate, 2),
                        "pcm_sha256": pcm,
                        "pcm_format": f"mono {int(sample_rate)} Hz s16le"}}


# ── программа и пакеты ───────────────────────────────────────────────────────────────────────────

def program_version(here):
    """VERSION из server.py без его импорта (импорт сервера имеет побочные эффекты)."""
    try:
        tree = ast.parse((Path(here) / "server.py").read_text(encoding="utf-8"))
        return next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VERSION"
                                                         for t in n.targets))
    except (OSError, SyntaxError, ValueError, StopIteration):
        return None


def code_revision(here):
    """Ревизия кода: сумма файлов *.py (есть и у установленной копии без Git) и коммит, если он есть."""
    here = Path(here)
    h, n = hashlib.sha256(), 0
    for p in sorted(here.glob("*.py")):
        try:
            data = p.read_bytes()
        except OSError:
            continue
        h.update(p.name.encode("utf-8") + b"\0" + hashlib.sha256(data).digest())
        n += 1
    out = {"files_sha256": h.hexdigest()[:16] if n else None, "files": n,
           "git_revision": None, "git_dirty": None}
    if (here / ".git").exists():
        rev = _run(["git", "-C", str(here), "rev-parse", "HEAD"])
        if rev and rev.strip():
            out["git_revision"] = rev.strip()
            dirty = _run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=all"])
            out["git_dirty"] = bool(dirty.strip()) if dirty is not None else None
    try:
        import buildinfo
        out["build"] = buildinfo.identity(here)
    except Exception:
        out["build"] = {"release": None, "commit": None, "source_state": "unknown",
                        "payload_sha256": None, "manifest_status": "unknown"}
    if out["git_revision"] is None and out["build"]["manifest_status"] in ("matched", "modified"):
        out["git_revision"] = out["build"]["commit"]
        out["git_dirty"] = {"clean": False, "dirty": True}.get(out["build"]["source_state"])
    return out


def packages(key=KEY_PACKAGES):
    """Версии ключевых пакетов и контрольная сумма полного списка «имя==версия»."""
    found, sources = {}, {}
    try:
        for d in metadata.distributions():
            name = str(d.metadata["Name"] or "").strip().lower().replace("_", "-").replace(".", "-")
            if not name:
                continue
            found[name] = str(d.version)
            if name in key:
                try:
                    direct = json.loads(d.read_text("direct_url.json") or "null")
                except (OSError, ValueError):
                    direct = None
                # пакет, поставленный не из индекса (gigaam — архив ветки): адрес и сумма архива.
                # Локальные адреса (file://) не записываются: в них пути.
                if isinstance(direct, dict) and str(direct.get("url", "")).startswith("http"):
                    sources[name] = {"url": direct["url"],
                                     "hash": (direct.get("archive_info") or {}).get("hash")}
    except Exception:
        pass
    lines = sorted(f"{n}=={v}" for n, v in found.items())
    return {"key": {k: found[k] for k in key if k in found},
            "sources": sources or None,
            "count": len(lines),
            "sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest() if lines else None}


def os_info():
    out = {"system": platform.system(), "release": platform.release(),
           "version": platform.version(), "machine": platform.machine()}
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as k:
                # ProductName не берём: в Windows 11 там по-прежнему написано «Windows 10»
                for field, value in (("display_version", "DisplayVersion"), ("build", "CurrentBuild"),
                                     ("ubr", "UBR")):
                    try:
                        out[field] = winreg.QueryValueEx(k, value)[0]
                    except OSError:
                        pass
        except Exception:
            pass
    return out


def ffmpeg_version():
    out = _run(["ffmpeg", "-version"], timeout=5)
    return out.splitlines()[0].strip()[:120] if out and out.strip() else None


def environment(here):
    return {"os": os_info(),
            "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
            "packages": packages(),
            "ffmpeg": ffmpeg_version(),
            "program_version": program_version(here),
            "code": code_revision(here)}


# ── условия запуска ──────────────────────────────────────────────────────────────────────────────

def power_status():
    """Питание: от сети или от батареи. Вне Windows не измеряется."""
    out = {"power_source": None, "battery_percent": None, "battery_present": None}
    if os.name != "nt":
        return out
    try:
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                        ("system", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]

        s = _Status()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            out["power_source"] = {0: "battery", 1: "ac"}.get(s.ac)
            out["battery_percent"] = int(s.percent) if s.percent <= 100 else None
            out["battery_present"] = None if s.flag == 255 else not bool(s.flag & 128)
    except Exception:
        pass
    return out


def power_scheme():
    """Схема питания (powercfg) и режим питания Windows (ползунок) — идентификаторы и известные имена."""
    out = {"scheme_guid": None, "scheme": None, "mode_ac_guid": None, "mode_dc_guid": None,
           "mode_ac": None, "mode_dc": None}
    if os.name != "nt":
        return out
    text = _run(["powercfg", "/getactivescheme"], timeout=5, oem=True)
    m = _GUID.search(text or "")
    if m:
        out["scheme_guid"] = m.group(0).lower()
        out["scheme"] = POWER_SCHEMES.get(out["scheme_guid"], "other")
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes") as k:
            for field, value in (("mode_ac", "ActiveOverlayAcPowerScheme"),
                                 ("mode_dc", "ActiveOverlayDcPowerScheme")):
                try:
                    guid = str(winreg.QueryValueEx(k, value)[0]).strip().lower()
                except OSError:
                    continue
                out[field + "_guid"] = guid
                out[field] = POWER_MODES.get(guid, "other")
    except Exception:
        pass
    return out


def memory_status():
    """Память: всего и свободно, МБ."""
    try:
        if os.name == "nt":
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_ext", ctypes.c_ulonglong)]

            m = _Mem()
            m.length = ctypes.sizeof(_Mem)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return {"total_mb": int(m.total // 2 ** 20), "free_mb": int(m.avail // 2 ** 20)}
        else:
            info = {}
            for line in Path("/proc/meminfo").read_text(encoding="ascii", errors="replace").splitlines():
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.split()[0]) if v.split() else 0
            return {"total_mb": info["MemTotal"] // 1024, "free_mb": info["MemAvailable"] // 1024}
    except Exception:
        pass
    return {"total_mb": None, "free_mb": None}


def uptime_sec():
    """Сколько секунд прошло с загрузки системы."""
    try:
        if os.name == "nt":
            import ctypes
            f = ctypes.windll.kernel32.GetTickCount64
            f.restype = ctypes.c_ulonglong
            return int(f() // 1000)
        return int(float(Path("/proc/uptime").read_text(encoding="ascii").split()[0]))
    except Exception:
        return None


def _cpu_times():
    """(занято, всего) в условных единицах с момента загрузки."""
    if os.name == "nt":
        import ctypes

        class _FT(ctypes.Structure):
            _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

        idle, kernel, user = _FT(), _FT(), _FT()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise OSError("GetSystemTimes")
        i, k, u = ((x.high << 32) | x.low for x in (idle, kernel, user))
        return k + u - i, k + u          # время ядра включает простой
    parts = [int(x) for x in Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()[1:]]
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
    return sum(parts) - idle, sum(parts)


def cpu_load_percent(interval=1.0):
    """Средняя загрузка всех ядер за interval секунд."""
    try:
        b0, t0 = _cpu_times()
        time.sleep(max(0.0, interval))
        b1, t1 = _cpu_times()
        return round(100.0 * (b1 - b0) / (t1 - t0), 1) if t1 > t0 else None
    except Exception:
        return None


def other_heavy_jobs(jobs_dir, limit, own_slots=0):
    """Сколько мест очереди тяжёлых задач занято другими процессами (jobslots).

    Место занято, если его файл заблокирован; своё место тоже выглядит занятым, поэтому
    own_slots вычитается. Проба берёт блокировку на мгновение и тут же отпускает."""
    try:
        import jobslots
        busy = 0
        slots = Path(jobs_dir) / "slots"
        for n in range(max(1, int(limit or 1))):
            p = slots / f"slot{n}.lock"
            if not p.is_file():
                continue
            with open(p, "a+") as f:
                if not jobslots._try_lock(f):
                    busy += 1
        return max(0, busy - int(own_slots or 0))
    except Exception:
        return None


def conditions(jobs_dir=None, slot_limit=None, own_slots=0, load_interval=1.0):
    """Снимок условий: питание, режим, загрузка, память, время с загрузки, соседние задачи."""
    out = {"taken_utc": utc_now()}
    out["cpu_load_percent"] = cpu_load_percent(load_interval)     # первым: остальные вызовы сами грузят
    out.update(power_status())
    out["power"] = power_scheme()
    out["memory"] = memory_status()
    out["uptime_sec"] = uptime_sec()
    out["other_heavy_jobs"] = (other_heavy_jobs(jobs_dir, slot_limit, own_slots)
                               if jobs_dir is not None else None)
    return out


def preflight(start, end=None, downloaded=False):
    """Готовность к замеру по записанным условиям. Прогон не останавливает: возвращает пометку."""
    reasons = []
    if start.get("power_source") == "battery" or (end or {}).get("power_source") == "battery":
        reasons.append("battery")
    if end and start.get("power_source") and end.get("power_source") \
            and start["power_source"] != end["power_source"]:
        reasons.append("power_source_changed")
    load = start.get("cpu_load_percent")
    if load is not None and load > LOAD_WARN_PERCENT:
        reasons.append("cpu_load")
    free = (start.get("memory") or {}).get("free_mb")
    if free is not None and free < FREE_RAM_WARN_MB:
        reasons.append("low_memory")
    if (start.get("other_heavy_jobs") or 0) > 0 or ((end or {}).get("other_heavy_jobs") or 0) > 0:
        reasons.append("other_heavy_jobs")
    if downloaded:
        reasons.append("model_downloaded")
    return {"ok": not reasons, "reasons": reasons}


def peak_memory_mb():
    """Пик памяти текущего процесса, МБ."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                            ("peak_wset", ctypes.c_size_t), ("wset", ctypes.c_size_t),
                            ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
                            ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
                            ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]

            c = _Counters()
            c.cb = ctypes.sizeof(_Counters)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            fn = getattr(k32, "K32GetProcessMemoryInfo")
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            if fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return round(c.peak_wset / 2 ** 20, 1)
            return None
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (2 ** 20 if sys.platform == "darwin" else 1024), 1)
    except Exception:
        return None


# ── стенд ────────────────────────────────────────────────────────────────────────────────────────

_HARDWARE_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
function D($d) { if ($d) { ([datetime]$d).ToString('yyyy-MM-dd') } else { $null } }
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$cs = Get-CimInstance Win32_ComputerSystem
$bios = Get-CimInstance Win32_BIOS
$mem = @(Get-CimInstance Win32_PhysicalMemory)
$drv = @(Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceClass='DISPLAY' OR DeviceClass='COMPUTEACCELERATOR'")
[pscustomobject]@{
  cpu = [pscustomobject]@{ model = "$($cpu.Name)".Trim(); cores = $cpu.NumberOfCores
                           threads = $cpu.NumberOfLogicalProcessors; max_mhz = $cpu.MaxClockSpeed }
  system = [pscustomobject]@{ manufacturer = "$($cs.Manufacturer)".Trim(); model = "$($cs.Model)".Trim() }
  bios = [pscustomobject]@{ version = "$($bios.SMBIOSBIOSVersion)".Trim(); date = (D $bios.ReleaseDate) }
  memory = [pscustomobject]@{ modules = $mem.Count
                              total_mb = [int](($mem | Measure-Object Capacity -Sum).Sum / 1MB)
                              speed_mhz = ($mem | Select-Object -First 1).ConfiguredClockSpeed }
  devices = @($drv | ForEach-Object { [pscustomobject]@{ name = "$($_.DeviceName)".Trim(); 'class' = "$($_.DeviceClass)"
                                                         driver_version = "$($_.DriverVersion)"
                                                         driver_date = (D $_.DriverDate) } })
} | ConvertTo-Json -Depth 4 -Compress
"""


def _powershell(script, timeout=45):
    """Запуск сценария Windows PowerShell 5.1 без профиля; сценарий передаётся закодированным."""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return _run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], timeout=timeout)


def cpu_features():
    """Наборы команд процессора, от которых зависит скорость: AVX2, AVX-512, VNNI (что удаётся узнать)."""
    out = {"avx2": None, "avx512f": None, "vnni": None}
    try:
        if os.name == "nt":
            import ctypes
            has = ctypes.windll.kernel32.IsProcessorFeaturePresent
            out["avx2"], out["avx512f"] = bool(has(40)), bool(has(41))
        else:
            flags = set()
            for line in Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace").splitlines():
                if line.startswith("flags"):
                    flags = set(line.partition(":")[2].split())
                    break
            if flags:
                out = {"avx2": "avx2" in flags, "avx512f": "avx512f" in flags,
                       "vnni": bool({"avx_vnni", "avx512_vnni"} & flags)}
    except Exception:
        pass
    return out


def hardware(run=None):
    """Железо и драйверы графики и NPU. run — подмена запуска PowerShell (для тестов)."""
    features = cpu_features()
    if os.name != "nt" and run is None:
        cpu_model = None
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace").splitlines():
                if line.startswith("model name"):
                    cpu_model = line.partition(":")[2].strip()
                    break
        except OSError:
            pass
        return {"status": "partial", "cpu": {"model": cpu_model or platform.processor() or None,
                                             "cores": None, "threads": os.cpu_count(), "max_mhz": None,
                                             "features": features},
                "memory": {"total_mb": memory_status()["total_mb"]}, "devices": []}
    try:
        raw = (run or _powershell)(_HARDWARE_PS)
    except Exception as ex:
        return {**not_measured(type(ex).__name__), "cpu": {"features": features}}
    if not raw or not raw.strip():
        return {**not_measured("PowerShell не вернул сведения"), "cpu": {"features": features}}
    try:
        data = json.loads(raw.strip().lstrip("﻿"))
        if not isinstance(data, dict):
            raise ValueError
    except ValueError:
        return {**not_measured("ответ PowerShell не разобран"), "cpu": {"features": features}}
    cpu = data.get("cpu") if isinstance(data.get("cpu"), dict) else {}
    return {"status": "ok",
            "cpu": {**cpu, "features": features},
            "system": data.get("system"),
            "bios": data.get("bios"),
            "memory": data.get("memory"),
            "devices": [d for d in (data.get("devices") or []) if isinstance(d, dict)]}


def _machine_guid():
    try:
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography", 0,
                                winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as k:
                return str(winreg.QueryValueEx(k, "MachineGuid")[0])
        return Path("/etc/machine-id").read_text(encoding="ascii").strip()
    except Exception:
        return None


def testbed_id(bench_dir, machine=None):
    """Обезличенный идентификатор стенда: хэш идентификатора машины с локальной солью.

    Соль лежит в <папка данных>\\bench\\testbed.json и никуда не выгружается; без неё по
    идентификатору стенда машину не узнать. Папка данных, перенесённая на другой компьютер,
    даст там другой идентификатор."""
    p = Path(bench_dir) / "testbed.json"
    salt = None
    try:
        salt = json.loads(p.read_text(encoding="utf-8")).get("salt")
    except (OSError, ValueError, AttributeError):
        pass
    if not salt:
        salt = secrets.token_hex(16)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            _write_json(p, {"schema": 1, "salt": salt, "created_utc": utc_now()})
        except OSError:
            pass
    machine = machine if machine is not None else (_machine_guid() or "")
    return hashlib.sha256(f"{salt}|{machine}".encode("utf-8")).hexdigest()[:16]


# ── модели и кэши ────────────────────────────────────────────────────────────────────────────────

def _tree_stats(root):
    """(число файлов, байт) в папке; ссылки на файлы считаются по цели."""
    files = size = 0
    for base, _dirs, names in os.walk(root):
        for name in names:
            try:
                size += os.stat(os.path.join(base, name)).st_size
                files += 1
            except OSError:
                pass
    return files, size


def _hub_dir(cache_dir, hf_home=None):
    return Path(hf_home or os.environ.get("HF_HOME") or (Path(cache_dir) / "huggingface")) / "hub"


def models_cache(cache_dir, hf_home=None):
    """Какие модели лежат в кэше: репозиторий, ревизия снимка, объём. Пути не записываются."""
    out = {"hf": [], "local": []}
    hub = _hub_dir(cache_dir, hf_home)
    try:
        repos = sorted(d for d in hub.glob("models--*") if d.is_dir())
    except OSError:
        repos = []
    for d in repos:
        ref = None
        try:
            ref = (d / "refs" / "main").read_text(encoding="ascii").strip() or None
        except OSError:
            pass
        try:
            snaps = sorted(x.name for x in (d / "snapshots").iterdir() if x.is_dir())
        except OSError:
            snaps = []
        files, size = _tree_stats(d / "snapshots")
        out["hf"].append({"repo": d.name[len("models--"):].replace("--", "/"), "ref_main": ref,
                          "snapshots": snaps, "files": files, "size_mb": round(size / 1e6, 1)})
    for sub in ("gigaam", "onnx-models"):
        root = Path(cache_dir) / sub
        for base, _dirs, names in os.walk(root):
            for name in sorted(names):
                p = Path(base) / name
                try:
                    st = p.stat()
                except OSError:
                    continue
                if st.st_size >= DOWNLOAD_BYTES:        # мелкие служебные файлы не перечисляем
                    out["local"].append({"file": p.relative_to(Path(cache_dir)).as_posix(),
                                         "size_bytes": st.st_size, "modified_utc": utc_now(st.st_mtime)})
    return out


def cache_counts(cache_dir, hf_home=None):
    """Счётчики кэшей до и после загрузки модели: по разнице видно компиляцию и скачивание."""
    ov_files, _ = _tree_stats(Path(cache_dir) / "openvino")
    model_files = model_bytes = 0
    for root in (_hub_dir(cache_dir, hf_home), Path(cache_dir) / "gigaam", Path(cache_dir) / "onnx-models"):
        f, b = _tree_stats(root)
        model_files += f
        model_bytes += b
    return {"openvino_files": ov_files, "model_files": model_files, "model_bytes": model_bytes}


def cache_delta(before, after):
    """Что изменилось в кэшах за время загрузки модели одного варианта."""
    return {"compile_cache_new_files": max(0, after["openvino_files"] - before["openvino_files"]),
            "model_downloaded": (after["model_bytes"] - before["model_bytes"]) >= DOWNLOAD_BYTES}


# ── запись прогона ───────────────────────────────────────────────────────────────────────────────

def new_run_id(now=None):
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime(now)) + "-" + secrets.token_hex(3)


# Пути в тексте ошибок: в кавычках — до закрывающей кавычки; путь Windows без кавычек — до конца
# строки (в нём бывают пробелы, границу не угадать); путь POSIX — до пробела. Адреса https:// не трогаются.
_PATHS = re.compile(r"""(['"])(?:[A-Za-z]:[\\/]|\\\\|/)[^'"\r\n]*\1"""
                    r"""|(?<![A-Za-z])[A-Za-z]:[\\/][^\r\n"'<>|]*"""
                    r"""|(?<![\w\\])\\\\[^\r\n"'<>|]+"""
                    r"""|(?<![\w/:.])/(?:home|Users|tmp|var|mnt|root|opt|usr|etc)/[^\s"']*""")


def scrub(value):
    """Заменяет пути в строках на «<путь>» во всей структуре: в файле прогона путей быть не должно."""
    if isinstance(value, str):
        return _PATHS.sub("<путь>", value)
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


def write_run(bench_dir, record, texts=None):
    """Пишет <bench>\\runs\\<run_id>.json и рядом тексты вариантов. Прежние прогоны не перезаписывает.

    Тексты — сырой вывод: лежат отдельным файлом, в выгрузку не идут. -> путь файла прогона."""
    d = Path(bench_dir) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    run_id = str(record.get("run_id") or new_run_id())
    while (d / f"{run_id}.json").exists() or (d / f"{run_id}.texts.json").exists():
        run_id = new_run_id()
    # пути могли попасть в тексты ошибок и откатов; сырой текст вариантов не трогается
    record = {**scrub(record), "run_id": run_id, "texts_file": None}
    if texts is not None:
        _write_json(d / f"{run_id}.texts.json", {"schema_version": SCHEMA_VERSION, "run_id": run_id,
                                                 "texts": texts})
        record["texts_file"] = f"{run_id}.texts.json"
    path = d / f"{run_id}.json"
    _write_json(path, record)
    return path


def update_units_index(bench_dir, identity, audio_path, run_id=None):
    """Локальный указатель «контрольная сумма звука → где лежит файл»: <bench>\\units.json.

    Нужен, чтобы по файлу прогона (в нём путей нет) найти запись и после очистки заданий.
    Только для этого компьютера: в выгрузку не идёт. Нечитаемый указатель не перезаписывается.
    -> True, если указатель обновлён."""
    key = identity.get("audio_sha256")
    if not key:
        return False
    p = Path(bench_dir) / "units.json"
    data = {"schema_version": SCHEMA_VERSION, "units": {}}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("units"), dict):
                raise ValueError
        except (OSError, ValueError):
            return False
    now = utc_now()
    src = Path(audio_path)
    unit = data["units"].get(key) if isinstance(data["units"].get(key), dict) else {}
    excerpts = [x for x in unit.get("unit_ids", []) if isinstance(x, str)]
    if identity.get("unit_id") and identity["unit_id"] not in excerpts:
        excerpts.append(identity["unit_id"])
    data["units"][key] = {"path": str(src), "name": src.name, "size_bytes": identity.get("size_bytes"),
                          "first_seen_utc": unit.get("first_seen_utc") or now, "last_seen_utc": now,
                          "runs": int(unit.get("runs") or 0) + 1, "last_run_id": run_id,
                          "unit_ids": excerpts}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_json(p, data)
    except OSError:
        return False
    return True
