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


def extract_f0(y: np.ndarray, sr: int, cfg: Config, backend: str | None = None) -> F0Track:
    backend = backend or cfg.pitch_backend
    if backend == "pyin":
        return extract_f0_pyin(y, sr, cfg)
    if backend == "crepe":
        return extract_f0_crepe(y, sr, cfg)
    raise ValueError(f"unknown pitch backend {backend!r}")


def extract_f0_crepe(y: np.ndarray, sr: int, cfg: Config) -> F0Track:  # pragma: no cover - optional
    """torchcrepe backend (install with `uv add torchcrepe torch`)."""
    import librosa
    import torch
    import torchcrepe

    y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
    hop = int(16000 * cfg.hop_ms / 1000)
    audio = torch.from_numpy(y16)[None]
    f0, periodicity = torchcrepe.predict(
        audio, 16000, hop, cfg.fmin, cfg.fmax, model="full", return_periodicity=True, batch_size=512,
        device="cpu",
    )
    f0 = f0[0].numpy().astype(np.float64)
    conf = periodicity[0].numpy().astype(np.float64)
    f0 = np.where(conf >= 0.05, f0, np.nan)
    n = len(f0)
    times = np.arange(n) * hop / 16000
    return F0Track(times=times, f0_hz=f0, conf=conf, hop_s=hop / 16000, rms=_rms(y16, 16000, hop, n))
