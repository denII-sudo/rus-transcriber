"""Голос на ноутбуке без дополнительных пакетов: озвучка (Windows SAPI, голос Irina или другой русский),
проигрывание клипа и запись с микрофона (Windows MCI через winmm), сигнал перед записью.

    speak(text)                 — озвучить и дождаться конца
    play(path)                  — проиграть mp3/wav и дождаться конца
    record(path, seconds)       — записать wav 16 кГц моно
Silero (по желанию) не используется: голос Windows есть всегда и не требует скачивания модели.
"""
import ctypes
import os
import time
from pathlib import Path

_VOICE = None


def _winmm(cmd):
    buf = ctypes.create_unicode_buffer(512)
    err = ctypes.windll.winmm.mciSendStringW(cmd, buf, 511, 0)
    if err:
        msg = ctypes.create_unicode_buffer(512)
        ctypes.windll.winmm.mciGetErrorStringW(err, msg, 511)
        raise RuntimeError(f"MCI: {msg.value} ({cmd.split()[0]})")
    return buf.value


def voice():
    """SAPI-голос: Irina, иначе любой русский, иначе по умолчанию. -> (объект, имя)."""
    global _VOICE
    if _VOICE is None:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        v = win32com.client.Dispatch("SAPI.SpVoice")
        name = v.Voice.GetDescription()
        best = None
        for tok in v.GetVoices():
            d = tok.GetDescription()
            if "irina" in d.lower():
                best = tok
                break
            if best is None and ("russian" in d.lower() or "рус" in d.lower()):
                best = tok
        if best is not None:
            v.Voice = best
            name = best.GetDescription()
        v.Rate = 1
        _VOICE = (v, name)
    return _VOICE


def speak(text):
    v, _ = voice()
    v.Speak(str(text), 0)          # 0 — синхронно


def play(path):
    p = str(Path(path).resolve())
    alias = f"clip{int(time.time() * 1000) % 100000}"
    kind = "mpegvideo" if p.lower().endswith((".mp3", ".m4a", ".mp4")) else "waveaudio"
    _winmm(f'open "{p}" type {kind} alias {alias}')
    try:
        _winmm(f"play {alias} wait")
    finally:
        try:
            _winmm(f"close {alias}")
        except RuntimeError:
            pass


def beep():
    try:
        import winsound
        winsound.Beep(880, 180)
    except Exception:
        pass


def _mci_record(p, seconds, fmt):
    _winmm("open new type waveaudio alias rec")
    try:
        if fmt:
            _winmm("set rec bitspersample 16 samplespersec 16000 channels 1")
        _winmm("record rec")
        time.sleep(max(0.5, float(seconds)))
        _winmm("stop rec")
        _winmm(f'save rec "{p}"')
    finally:
        try:
            _winmm("close rec")
        except RuntimeError:
            pass


def _dshow_mic():
    """Имя микрофона для ffmpeg (DirectShow): первое аудиоустройство из списка."""
    import re
    import shutil
    import subprocess
    r = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow",
                        "-i", "dummy"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=0x08000000)
    for line in r.stderr.splitlines():
        m = re.search(r'"([^"]+)"\s*\(audio\)', line)
        if m:
            return m.group(1)
    names, audio = [], False                       # старый формат вывода ffmpeg
    for line in r.stderr.splitlines():
        if "DirectShow audio devices" in line:
            audio = True
        elif "DirectShow video devices" in line:
            audio = False
        elif audio:
            m = re.search(r'\]\s+"([^"]+)"', line)
            if m and "Alternative name" not in line:
                names.append(m.group(1))
    return names[0] if names else None


def _ffmpeg_record(p, seconds):
    import shutil
    import subprocess
    mic = _dshow_mic()
    if not mic:
        raise RuntimeError("микрофон не найден (ни MCI, ни DirectShow): проверьте, что он включён и Windows "
                           "разрешает классическим приложениям доступ к микрофону")
    r = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-f", "dshow",
                        "-i", f"audio={mic}", "-t", f"{float(seconds):.1f}", "-ac", "1", "-ar", "16000", p],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=0x08000000)
    if r.returncode != 0:
        raise RuntimeError(f"запись через ffmpeg ({mic}) не удалась: {r.stderr[-300:]}")
    return mic


def record(path, seconds=5.0):
    """Записывает wav. Сначала Windows MCI (16 кГц моно, затем формат по умолчанию), если не вышло —
    ffmpeg через DirectShow с первым микрофоном. -> путь."""
    p = str(Path(path).resolve())
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for fmt in (True, False):
        try:
            _mci_record(p, seconds, fmt)
            if Path(p).exists() and Path(p).stat().st_size > 1000:
                return p
        except RuntimeError as ex:
            errors.append(str(ex))
    try:
        _ffmpeg_record(p, seconds)
        return p
    except RuntimeError as ex:
        errors.append(str(ex))
    raise RuntimeError("не удалось записать с микрофона: " + " | ".join(errors))


def available():
    return os.name == "nt"
