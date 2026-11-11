"""Ускорение pyannote (community-1, WeSpeaker ResNet34) для процессора и Intel Arc.

Что делает оригинал: каждый 10-секундный кусок записи прогоняется через ResNet отпечатков
голоса столько раз, сколько в куске «местных» спикеров (обычно 3) — с одним и тем же звуком,
меняется только маска «кто когда говорит». Маска применяется лишь в слое pool в самом конце.

Что делаем мы:
  1. считаем тяжёлую часть (fbank + свёрточные слои, resnet.forward_frames) один раз на кусок,
     а маски накладываем потом (resnet.forward_embedding) — математически то же самое;
  2. по желанию считаем тяжёлую часть на видеокарте Arc (или NPU) через OpenVINO —
     все куски одной длины, поэтому модель компилируется один раз под фиксированный размер;
  3. на первом пакете сверяем отпечатки с оригинальным pyannote; при расхождении или любой
     ошибке откатываемся на следующий вариант, а в крайнем случае — на оригинальный код.
"""
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

MIN_COSINE = 0.999      # порог совпадения с оригиналом на активных (кусок, спикер)
CHECK_PAIRS = 24        # сколько пар (кусок, спикер) сверять


def _log(msg):
    print(f"[diar] {msg}", file=sys.stderr, flush=True)


def device_name(name):
    """Имя устройства OpenVINO — только заглавными: «cpu» из config.json иначе не находится
    («Device with "cpu" name is not registered»), и ускорение молча отключается."""
    return ":".join(part.strip().upper() for part in str(name or "GPU").split(":"))


class TorchTrunk:
    name = "cpu"

    def __init__(self, resnet):
        self.resnet = resnet

    def __call__(self, fbank):
        return self.resnet.forward_frames(fbank.clone())


class OVTrunk:
    """resnet.forward_frames, скомпилированный OpenVINO под фиксированный размер (B, T, F)."""

    def __init__(self, resnet, device="GPU", precision=None, cache_dir=None):
        self.resnet, self.device, self.precision = resnet, device, precision
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.compiled = None
        self.shape = None
        self.name = f"openvino:{device}" + (f":{precision}" if precision else "")

    def _key(self):
        h = hashlib.sha1()
        for p in self.resnet.parameters():
            h.update(p.detach().cpu().numpy().tobytes()[:4096])
        return h.hexdigest()[:12]

    def build(self, B, T, F):
        import openvino as ov

        class Frames(torch.nn.Module):
            def __init__(self, r):
                super().__init__()
                self.r = r

            def forward(self, x):
                return self.r.forward_frames(x.clone())

        core = ov.Core()
        xml = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(self.cache_dir)})
            xml = self.cache_dir / f"wespeaker_{self._key()}_B{B}_T{T}_F{F}.xml"
        if xml is not None and xml.exists():
            model = str(xml)
        else:
            with torch.no_grad():
                ovm = ov.convert_model(Frames(self.resnet).eval(), example_input=torch.zeros(B, T, F))
            ovm.reshape([B, T, F])
            if xml is not None:
                ov.save_model(ovm, str(xml), compress_to_fp16=False)
                model = str(xml)
            else:
                model = ovm
        cfg = {"INFERENCE_PRECISION_HINT": self.precision} if self.precision else {}
        t0 = time.time()
        self.compiled = core.compile_model(model, self.device, cfg)
        self.shape = (B, T, F)
        _log(f"OpenVINO {self.device} скомпилирован за {time.time() - t0:.1f} с, размер {self.shape}")

    def __call__(self, fbank):
        B, T, F = fbank.shape
        if self.compiled is None or self.shape[1:] != (T, F):
            self.build(max(B, self.shape[0] if self.shape else B), T, F)
        bb = self.shape[0]
        x = fbank.detach().cpu().numpy().astype(np.float32)
        if B < bb:  # последний неполный пакет дополняем нулями
            x = np.concatenate([x, np.zeros((bb - B, T, F), dtype=np.float32)])
        out = self.compiled(x)[0]
        return torch.from_numpy(np.asarray(out)[:B].copy())


def _iter_chunks(pipe, file, binary_segmentations, exclude_overlap):
    """Те же маски, что в pyannote SpeakerDiarization.get_embeddings (4.0.x)."""
    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, num_speakers = binary_segmentations.data.shape
    if exclude_overlap:
        min_num_samples = pipe._embedding.min_num_samples
        num_samples = duration * pipe._embedding.sample_rate
        min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)
        clean = binary_segmentations.data * (
            1.0 * (np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2))
    else:
        min_num_frames = -1
        clean = binary_segmentations.data
    for c, (chunk, _) in enumerate(binary_segmentations):              # как в оригинале
        waveform, _ = pipe._audio.crop(file, chunk, mode="pad")       # (1, num_samples)
        masks = np.nan_to_num(binary_segmentations.data[c], nan=0.0).astype(np.float32)
        cmasks = np.nan_to_num(clean[c], nan=0.0).astype(np.float32)
        used = [cm if np.sum(cm) > min_num_frames else m for m, cm in zip(masks.T, cmasks.T)]
        yield waveform, np.stack(used)                                 # (S, num_frames)


def install(pipe, mode="auto", ov_device="GPU", chunks_per_batch=16, cache_dir=None, info=None):
    """Подменяет pipe.get_embeddings быстрым вариантом. info — dict, куда пишется, что выбрано."""
    info = info if info is not None else {}
    info["embeddings_backend"] = "original"
    ov_device = device_name(ov_device)
    if mode == "original":
        return info
    emb = getattr(pipe, "_embedding", None)
    model = getattr(emb, "model_", None)
    resnet = getattr(model, "resnet", None)
    if not (resnet is not None and hasattr(resnet, "forward_frames") and hasattr(resnet, "forward_embedding")
            and hasattr(model, "compute_fbank")):
        _log("модель эмбеддингов не WeSpeaker ResNet — оставляю оригинальный код pyannote")
        return info

    candidates = []
    if mode in ("auto", "openvino"):
        candidates += [OVTrunk(resnet, ov_device, None, cache_dir), OVTrunk(resnet, ov_device, "f32", cache_dir)]
    if mode in ("auto", "openvino", "cpu"):
        candidates.append(TorchTrunk(resnet))
    original = pipe.get_embeddings

    def embed_batch(trunk, waves, masks):
        # waves: (B, 1, N); masks: (B, S, frames) -> (B, S, dim)
        with torch.inference_mode():
            fbank = model.compute_fbank(waves)
            frames = trunk(fbank)
            outs = [resnet.forward_embedding(frames, weights=masks[:, s, :])[1] for s in range(masks.shape[1])]
        return torch.stack(outs, dim=1).cpu().numpy()

    def check(trunk, waves, masks):
        """Сверка с оригиналом pyannote на активных парах (кусок, спикер)."""
        fast = embed_batch(trunk, waves, masks)
        B, S, _ = masks.shape
        pairs = [(b, s) for b in range(B) for s in range(S) if float(masks[b, s].sum()) > 0][:CHECK_PAIRS]
        if not pairs:
            return 1.0
        w = torch.stack([waves[b] for b, _ in pairs])               # (P, 1, N)
        m = torch.stack([masks[b, s] for b, s in pairs])            # (P, frames)
        ref = emb(w, masks=m)                                       # оригинальный путь pyannote
        got = np.stack([fast[b, s] for b, s in pairs])
        cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1) + 1e-9)
        return float(cos.min())

    def fast_get_embeddings(file, binary_segmentations, exclude_overlap=False, hook=None):
        t0 = time.time()
        try:
            num_chunks, _, num_speakers = binary_segmentations.data.shape
            n_batches = math.ceil(num_chunks / chunks_per_batch)
            if hook is not None:
                hook("embeddings", None, total=n_batches, completed=0)
            it = _iter_chunks(pipe, file, binary_segmentations, exclude_overlap)
            result, trunk, done = [], None, 0
            while True:
                batch = [x for _, x in zip(range(chunks_per_batch), it)]
                if not batch:
                    break
                waves = torch.stack([w for w, _ in batch])                    # (B, 1, N)
                masks = torch.from_numpy(np.stack([m for _, m in batch]))     # (B, S, frames)
                if trunk is None:
                    for cand in candidates:
                        try:
                            score = check(cand, waves, masks)
                            info.setdefault("checks", {})[cand.name] = round(score, 5)
                            _log(f"{cand.name}: совпадение с оригиналом {score:.5f}")
                            if score >= MIN_COSINE:
                                trunk = cand
                                break
                        except Exception as ex:
                            info.setdefault("checks", {})[cand.name] = f"ошибка: {type(ex).__name__}: {ex}"[:300]
                            _log(f"{cand.name} не подошёл: {type(ex).__name__}: {ex}")
                    if trunk is None:
                        raise RuntimeError("ни один быстрый вариант не прошёл сверку")
                    info["embeddings_backend"] = trunk.name
                result.append(embed_batch(trunk, waves, masks))
                done += 1
                if hook is not None:
                    hook("embeddings", None, total=n_batches, completed=done)
            info["embeddings_sec"] = round(time.time() - t0, 1)
            return np.concatenate(result).astype(np.float32)               # (chunks, S, dim)
        except Exception as ex:
            _log(f"быстрый путь отключён ({type(ex).__name__}: {ex}), считаю оригинальным pyannote")
            info["embeddings_backend"] = "original"
            info["fast_error"] = f"{type(ex).__name__}: {ex}"[:300]
            t1 = time.time()
            out = original(file, binary_segmentations, exclude_overlap=exclude_overlap, hook=hook)
            info["embeddings_sec"] = round(time.time() - t1, 1)
            return out

    pipe.get_embeddings = fast_get_embeddings
    return info
