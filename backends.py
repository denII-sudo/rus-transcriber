"""Бэкенды распознавания GigaAM для Windows-ноутбука на Intel Core Ultra.

  openvino  — onnx-asr + ONNX Runtime OpenVINO EP: устройство GPU (Intel Arc iGPU), NPU или CPU
  dml       — onnx-asr + DirectML (любая DX12-видеокарта, запасной вариант для GPU)
  onnx-cpu  — onnx-asr на процессоре
  torch     — оригинальный пакет gigaam на PyTorch CPU (эталон, самый медленный)
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import tmpfiles  # noqa: E402
import paths  # noqa: F401  (задаёт HF_HOME и др. до загрузки моделей)
from paths import GIGAAM_DIR, OPENVINO_DIR

SR = 16000
HERE = Path(__file__).resolve().parent

# имена одной и той же модели в двух пакетах
ONNX_NAMES = {"v3_e2e_rnnt": "gigaam-v3-e2e-rnnt", "v3_e2e_ctc": "gigaam-v3-e2e-ctc",
              "v3_rnnt": "gigaam-v3-rnnt", "v3_ctc": "gigaam-v3-ctc"}


def _ort_windows_openvino_libs():
    """На Windows библиотеки OpenVINO нужно явно добавить в PATH до создания сессии."""
    if os.name != "nt":
        return
    try:
        import onnxruntime.tools.add_openvino_win_libs as w
        w.add_openvino_libs_to_path()
    except Exception:
        try:
            import openvino  # noqa: F401  (пакет openvino сам регистрирует свои dll)
        except Exception:
            pass


def available_providers():
    try:
        _ort_windows_openvino_libs()
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return []


def openvino_devices():
    try:
        import openvino as ov
        return ov.Core().available_devices
    except Exception:
        return []


def _as_text(res):
    if isinstance(res, str):
        return res
    for attr in ("text", "transcription"):
        if hasattr(res, attr):
            return getattr(res, attr)
        if isinstance(res, dict) and attr in res:
            return res[attr]
    return str(res)


class OnnxASR:
    def __init__(self, model="v3_e2e_rnnt", device="GPU", provider="openvino", pad_to=0.0, batch=1,
                 quant=None, language=None):
        import onnx_asr
        _ort_windows_openvino_libs()
        have = available_providers()
        if provider == "openvino":
            if "OpenVINOExecutionProvider" not in have:
                raise RuntimeError(f"OpenVINO EP недоступен (есть: {have}). Нужен пакет onnxruntime-openvino")
            cache = OPENVINO_DIR
            cache.mkdir(parents=True, exist_ok=True)
            providers = [("OpenVINOExecutionProvider", {"device_type": device, "cache_dir": str(cache)}),
                         "CPUExecutionProvider"]
        elif provider == "dml":
            if "DmlExecutionProvider" not in have:
                raise RuntimeError(f"DirectML недоступен (есть: {have}). Нужен пакет onnxruntime-directml")
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.name = f"{provider}:{device if provider == 'openvino' else ''}".rstrip(":")
        self.model_name = model
        self.pad_to = float(pad_to or 0)
        self.batch = max(1, int(batch or 1))
        if self.batch > 1:
            self.name += f":b{self.batch}"
        self.language = language
        kw = {"quantization": quant} if quant else {}
        self.model = onnx_asr.load_model(ONNX_NAMES.get(model, model), providers=providers, **kw)
        if quant:
            self.name += f":{quant}"

    def _pad(self, audio):
        if self.pad_to > 0:
            # дополняем тишиной до кратной длины, чтобы GPU не перекомпилировал модель под каждый размер
            step = int(self.pad_to * SR)
            need = -len(audio) % step
            if need:
                audio = np.concatenate([audio, np.zeros(need, dtype=np.float32)])
        return audio.astype(np.float32)

    def transcribe(self, audio):
        return _as_text(self.model.recognize(self._pad(audio), sample_rate=SR)).strip()

    def transcribe_many(self, pieces, progress=None):
        """Распознаёт список фрагментов пакетами по self.batch. Внутри пакета фрагменты похожей длины
        (сортировка по длине), чтобы почти не тратить время на дополнение тишиной. Порядок сохраняется."""
        out = [""] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
        done = g = 0
        while g < len(order):
            idx = order[g:g + self.batch]
            g += len(idx)  # шаг берётся заново: после отката на поштучное self.batch станет 1
            if self.batch > 1:
                try:
                    res = self.model.recognize([self._pad(pieces[i]) for i in idx], sample_rate=SR)
                    if not isinstance(res, (list, tuple)) or len(res) != len(idx):
                        raise TypeError(f"пакет вернул {type(res).__name__}")
                    for i, r in zip(idx, res):
                        out[i] = _as_text(r).strip()
                except Exception as ex:
                    print(f"[asr] пакетное распознавание не работает ({ex}), перехожу на поштучное",
                          file=sys.stderr, flush=True)
                    self.batch = 1
                    self.name = self.name.rsplit(":b", 1)[0]
                    for i in idx:
                        out[i] = self.transcribe(pieces[i])
            else:
                for i in idx:
                    out[i] = self.transcribe(pieces[i])
            done += len(idx)
            if progress:
                progress(done, len(pieces))
        return out


    def transcribe_timestamped(self, pieces, progress=None):
        """Как transcribe_many, но возвращает TimestampedResult (токены, времена, вероятности)."""
        model = self.model.with_timestamps()
        out = [None] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
        bs = max(1, min(self.batch, 8))  # куски длинные (до 20 с): пакет меньше, чтобы не раздувать память
        done = g = 0
        while g < len(order):
            idx = order[g:g + bs]
            g += len(idx)
            try:
                res = model.recognize([pieces[i].astype(np.float32) for i in idx], sample_rate=SR)
                if not isinstance(res, (list, tuple)) or len(res) != len(idx):
                    raise TypeError(f"пакет вернул {type(res).__name__}")
                for i, r in zip(idx, res):
                    out[i] = r
            except Exception as ex:
                if bs == 1:
                    print(f"[asr] кусок не распознан: {ex}", file=sys.stderr, flush=True)
                    continue
                print(f"[asr] пакетное распознавание со временами не работает ({ex}), перехожу на поштучное",
                      file=sys.stderr, flush=True)
                bs = 1
                for i in idx:
                    try:
                        out[i] = model.recognize(pieces[i].astype(np.float32), sample_rate=SR)
                    except Exception as ex2:
                        print(f"[asr] кусок не распознан: {ex2}", file=sys.stderr, flush=True)
            done += len(idx)
            if progress:
                progress(done, len(pieces))
        return out


    def transcribe_texts(self, pieces, progress=None):
        """Только текст (для Whisper: времён слов он не отдаёт). Язык — self.language."""
        kw = {"language": self.language} if self.language else {}
        out = [""] * len(pieces)
        bs = max(1, min(self.batch, 4))
        for g in range(0, len(pieces), bs):
            idx = list(range(g, min(g + bs, len(pieces))))
            try:
                res = self.model.recognize([pieces[i].astype(np.float32) for i in idx], sample_rate=SR, **kw)
                for i, r in zip(idx, res):
                    out[i] = _as_text(r).strip()
            except Exception as ex:
                print(f"[asr] пакет не распознан ({ex}), поштучно", file=sys.stderr, flush=True)
                for i in idx:
                    try:
                        out[i] = _as_text(self.model.recognize(pieces[i].astype(np.float32), sample_rate=SR, **kw)).strip()
                    except Exception as ex2:
                        print(f"[asr] кусок не распознан: {ex2}", file=sys.stderr, flush=True)
            if progress:
                progress(min(g + bs, len(pieces)), len(pieces))
        return out


class WhisperOV:
    """Whisper: тяжёлый кодировщик — напрямую в OpenVINO (видеокарта Arc, NPU или CPU), лёгкий декодер —
    onnxruntime на процессоре. Модуль OpenVINO внутри onnxruntime (onnxruntime-openvino 1.24) на этом
    компьютере не загружается (ошибка 127 — не совпадает с установленным openvino 2026), а у fp16-модели
    оптимизатор onnxruntime падает на слиянии LayerNorm; поэтому кодировщик fp16 идёт в OpenVINO,
    декодер — int8 (входы/выходы float32) в onnxruntime с базовой оптимизацией."""

    def __init__(self, model="onnx-community/whisper-large-v3-turbo", device="GPU", quant="fp16",
                 dec_quant="int8", language="ru", batch=1):
        import time as _t
        import onnxruntime as ort
        import openvino as ov
        import onnx_asr.loader as L
        import onnx_asr.models.whisper as WM
        from huggingface_hub import hf_hub_download
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        res = L.create_asr_resolver(model, None)
        files = res.resolve_model(quantization=dec_quant)
        sfx = f"_{quant}" if quant else ""
        enc_path = Path(files["decoder"]).parent / f"encoder_model{sfx}.onnx"
        if not enc_path.exists():
            enc_path = Path(hf_hub_download(model, f"onnx/encoder_model{sfx}.onnx"))
            try:
                hf_hub_download(model, f"onnx/encoder_model{sfx}.onnx_data")
            except Exception:
                pass

        class _Stub:        # кодировщик onnxruntime не нужен — его заменяет OpenVINO
            def get_providers(self):
                return ["CPUExecutionProvider"]

            def get_provider_options(self):
                return {"CPUExecutionProvider": {}}

        real = WM.rt

        class _RT:
            def __getattr__(self, n):
                return getattr(real, n)

            def InferenceSession(self, path, *a, **kw):
                return _Stub() if "encoder_model" in Path(str(path)).name else real.InferenceSession(path, *a, **kw)

        mgr = L.Manager(so, ["CPUExecutionProvider"], None, None, None)
        WM.rt = _RT()
        try:
            asr = res.model_type(files, mgr._create_preprocessor,
                                 {"sess_options": so, "providers": ["CPUExecutionProvider"]})
        finally:
            WM.rt = real
        self.adapter = mgr._create_asr_adapter(asr)
        self.asr = asr
        dec_in = {x.name: x.type for x in asr._decoder.get_inputs()}
        dec_dtype = np.float16 if "float16" in dec_in.get("encoder_hidden_states", "") else np.float32

        core = ov.Core()
        cache = OPENVINO_DIR / "whisper"
        cache.mkdir(parents=True, exist_ok=True)
        try:
            core.set_property({"CACHE_DIR": str(cache)})
        except Exception:
            pass
        t0 = _t.time()
        m = core.read_model(str(enc_path))
        self.batch = B = max(1, int(batch or 1))
        inp = m.inputs[0]
        mels = inp.get_partial_shape()[1]
        mels = mels.get_length() if mels.is_static else int(asr.config.get("num_mel_bins", 128))
        m.reshape({inp.get_any_name(): [B, mels, 3000]})
        self.enc = core.compile_model(m, device)
        self.compile_sec = round(_t.time() - t0, 1)
        in_dtype = np.float16 if "f16" in str(self.enc.inputs[0].get_element_type()) else np.float32
        self.language = language
        self.name = f"whisper-ov:{device}:{quant}+dec-{dec_quant}"
        self.model_name = model
        from onnxruntime import OrtValue
        req = self.enc.create_infer_request()

        def _encode(waveforms, waveforms_len):
            feats, _ = asr._preprocessor(waveforms, waveforms_len)
            feats = np.asarray(feats)
            n = feats.shape[0]
            if feats.shape[-1] != 3000:
                pad = np.zeros(feats.shape[:-1] + (3000,), dtype=feats.dtype)
                pad[..., :min(3000, feats.shape[-1])] = feats[..., :3000]
                feats = pad
            if n < B:
                feats = np.concatenate([feats, np.zeros((B - n,) + feats.shape[1:], dtype=feats.dtype)])
            out = req.infer({0: feats.astype(in_dtype)})[self.enc.outputs[0]][:n]
            return OrtValue.ortvalue_from_numpy(np.ascontiguousarray(out, dtype=dec_dtype))

        asr._encode = _encode     # декодер onnx-asr берёт выход кодировщика как OrtValue на процессоре

    def transcribe_texts(self, pieces, progress=None):
        kw = {"language": self.language} if self.language else {}
        out = [""] * len(pieces)
        for g in range(0, len(pieces), self.batch):
            idx = list(range(g, min(g + self.batch, len(pieces))))
            try:
                res = self.adapter.recognize([pieces[i].astype(np.float32) for i in idx], sample_rate=SR, **kw)
                for i, r in zip(idx, res):
                    out[i] = _as_text(r).strip()
            except Exception as ex:
                print(f"[whisper] пакет не распознан: {ex}", file=sys.stderr, flush=True)
            if progress:
                progress(min(g + self.batch, len(pieces)), len(pieces))
        return out


class TorchASR:
    def __init__(self, model="v3_e2e_rnnt"):
        import gigaam
        self.name = "torch:cpu"
        self.model = gigaam.load_model(model, download_root=str(GIGAAM_DIR))
        self._tmp = tmpfiles.scratch("gigaam_")

    def transcribe(self, audio):
        import soundfile as sf
        p = self._tmp / "piece.wav"
        sf.write(p, audio, SR)
        return _as_text(self.model.transcribe(str(p))).strip()

    def transcribe_many(self, pieces, progress=None):
        out = []
        for k, p in enumerate(pieces, 1):
            out.append(self.transcribe(p))
            if progress:
                progress(k, len(pieces))
        return out


def make_asr(cfg):
    """Создаёт бэкенд по config.json; при ошибке откатывается к следующему по списку."""
    model = cfg.get("asr_model", "v3_e2e_rnnt")
    backend = cfg.get("asr_backend", "auto")
    device = cfg.get("asr_device", "GPU")
    pad = cfg.get("pad_to_seconds", 0)
    batch = cfg.get("asr_batch", 1)
    if backend == "auto":
        order = [("openvino", device), ("dml", None), ("onnx-cpu", None), ("torch", None)]
    else:
        order = [(backend, device), ("onnx-cpu", None), ("torch", None)]
    if "whisper" in str(model):          # у пакета gigaam (PyTorch) Whisper нет
        order = [o for o in order if o[0] != "torch"]
    errors = []
    for b, d in order:
        try:
            if b == "torch":
                return TorchASR(model), errors
            prov = {"onnx-cpu": "cpu"}.get(b, b)
            quant = cfg.get("asr_quant") if b != "onnx-cpu" else cfg.get("asr_quant_cpu", cfg.get("asr_quant"))
            return OnnxASR(model, device=d or "CPU", provider=prov, pad_to=pad if b != "onnx-cpu" else 0,
                           batch=batch, quant=quant, language=cfg.get("asr_language")), errors
        except Exception as ex:
            errors.append(f"{b}{':' + d if d else ''}: {type(ex).__name__}: {ex}")
            print(f"[asr] {errors[-1]} → пробую следующий вариант", file=sys.stderr, flush=True)
    raise RuntimeError("Ни один бэкенд распознавания не запустился:\n" + "\n".join(errors))
