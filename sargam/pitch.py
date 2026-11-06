"""Frame-level f0 extraction. One entry point, swappable backends."""
from __future__ import annotations

import numpy as np

from .config import Config
from .model import F0Track


def _hop(cfg: Config, sr: int) -> int:
    return int(round(sr * cfg.hop_ms / 1000.0))


def _rms(y: np.ndarray, sr: int, hop: int, n_frames: int) -> np.ndarray:
    import librosa

    r = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop, center=True)[0]
    if len(r) < n_frames:
        r = np.pad(r, (0, n_frames - len(r)), mode="edge")
    return r[:n_frames].astype(np.float64)


def extract_f0_pyin(y: np.ndarray, sr: int, cfg: Config) -> F0Track:
    import librosa

    hop = _hop(cfg, sr)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y.astype(np.float32), fmin=cfg.fmin, fmax=cfg.fmax, sr=sr, hop_length=hop,
        frame_length=2048, fill_na=np.nan, center=True,
    )
    n = len(f0)
    times = np.arange(n) * hop / sr
    conf = np.where(np.isfinite(f0), voiced_prob, 0.0).astype(np.float64)
    return F0Track(times=times, f0_hz=f0.astype(np.float64), conf=conf, hop_s=hop / sr, rms=_rms(y, sr, hop, n))


BACKENDS = ("pyin", "crepe", "rmvpe", "fcpe")


def extract_f0(y: np.ndarray, sr: int, cfg: Config, backend: str | None = None) -> F0Track:
    backend = backend or cfg.pitch_backend
    if backend == "pyin":
        return extract_f0_pyin(y, sr, cfg)
    if backend == "crepe":
        return extract_f0_crepe(y, sr, cfg)
    if backend == "rmvpe":
        return extract_f0_rmvpe(y, sr, cfg)
    if backend == "fcpe":
        return extract_f0_fcpe(y, sr, cfg)
    raise ValueError(f"unknown pitch backend {backend!r} (choose from {', '.join(BACKENDS)})")


def _torch_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_f0_crepe(y: np.ndarray, sr: int, cfg: Config) -> F0Track:
    """torchcrepe backend (`uv add torchcrepe`). Uses the Apple GPU (MPS) or CUDA when available."""
    import librosa
    import torch
    import torchcrepe

    y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
    hop = int(16000 * cfg.hop_ms / 1000)
    audio = torch.from_numpy(y16)[None]
    f0, periodicity = torchcrepe.predict(
        audio, 16000, hop, cfg.fmin, cfg.fmax, model="full", return_periodicity=True, batch_size=1024,
        device=_torch_device(),
    )
    # light smoothing of the periodicity and f0 as recommended by torchcrepe, then back to numpy
    periodicity = torchcrepe.filter.median(periodicity, 3)
    f0 = torchcrepe.filter.median(f0, 3)
    f0 = f0[0].cpu().numpy().astype(np.float64)
    conf = periodicity[0].cpu().numpy().astype(np.float64)
    f0 = np.where(conf >= 0.05, f0, np.nan)
    n = len(f0)
    times = np.arange(n) * hop / 16000
    return F0Track(times=times, f0_hz=f0, conf=conf, hop_s=hop / 16000, rms=_rms(y16, 16000, hop, n))


def track_from_salience(sal: np.ndarray, y16: np.ndarray, cfg: Config) -> F0Track:
    """F0Track from a (frames, 360) RMVPE salience at 10 ms hop. conf = peak salience; frames whose peak is below
    the decoder floor are NaN, the pipeline's voicing threshold does the rest."""
    from . import rmvpe

    f0, peak = rmvpe.decode(sal)
    f0 = np.where((f0 >= cfg.fmin * 0.9) & (f0 <= cfg.fmax * 1.1), f0, np.nan)
    n = len(f0)
    times = np.arange(n) * rmvpe.HOP / rmvpe.SR
    return F0Track(times=times, f0_hz=f0, conf=peak.astype(np.float64), hop_s=rmvpe.HOP / rmvpe.SR,
                   rms=_rms(y16, rmvpe.SR, rmvpe.HOP, n))


def extract_f0_rmvpe(y: np.ndarray, sr: int, cfg: Config) -> F0Track:
    """RMVPE (trained for vocals in polyphonic music). Weights download once to data/models/. Fixed 10 ms hop."""
    import librosa

    from . import rmvpe

    y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=rmvpe.SR) if sr != rmvpe.SR else y.astype(np.float32)
    return track_from_salience(rmvpe.salience(y16, rmvpe.SR), y16, cfg)


_FCPE = {}


def extract_f0_fcpe(y: np.ndarray, sr: int, cfg: Config) -> F0Track:
    """torchfcpe (bundled weights). conf = peak of the 360-bin latent, decoded with its local-argmax decoder."""
    import librosa
    import torch
    from torchfcpe import spawn_bundled_infer_model

    device = _torch_device()
    if device not in _FCPE:
        _FCPE[device] = spawn_bundled_infer_model(device=device)
    m = _FCPE[device]
    y16 = np.clip(librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000), -1.0, 1.0)  # FCPE complains outside [-1, 1]
    hop = m.get_hop_size()
    with torch.no_grad():
        wav = torch.from_numpy(y16)[None, :, None].to(device)
        latent = m.model.forward(m.wav2mel(wav, 16000))
        cents = m.model.latent2cents_local_decoder(latent, mask=False)
        f0 = m.model.cent_to_f0(cents)[0, :, 0].cpu().numpy().astype(np.float64)
        conf = latent.max(dim=-1)[0][0].cpu().numpy().astype(np.float64)
    f0 = np.where((conf > 0.006) & (f0 >= cfg.fmin * 0.9) & (f0 <= cfg.fmax * 1.1), f0, np.nan)
    n = len(f0)
    return F0Track(times=np.arange(n) * hop / 16000, f0_hz=f0, conf=conf, hop_s=hop / 16000, rms=_rms(y16, 16000, hop, n))


def save_track(path, track: F0Track) -> None:
    np.savez(path, times=track.times, f0_hz=track.f0_hz, conf=track.conf, hop_s=track.hop_s,
             rms=track.rms if track.rms is not None else np.array([]))


def load_track(path) -> F0Track:
    z = np.load(path)
    rms = z["rms"] if z["rms"].size else None
    return F0Track(times=z["times"], f0_hz=z["f0_hz"], conf=z["conf"], hop_s=float(z["hop_s"]), rms=rms)
