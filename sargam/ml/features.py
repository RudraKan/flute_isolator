"""Frame features shared by synthetic training data and real pitch tracks.

Everything is on a fixed 10 ms grid and relative to Sa, so the model learns sargam directly:
  rel_cents  (T,) pitch in cents above Sa, NaN where unvoiced
  energy     (T,) loudness in [0, 1] (0 = 60 dB below the song's loud level)
The model turns rel_cents into a salience image itself (see model.SalienceRenderer); a real
salience (e.g. RMVPE's 360 bins rolled to Sa) can be fed in its place later.
"""
from __future__ import annotations

import numpy as np

HOP_S = 0.01
REL_LO, REL_HI = -18, 30  # semitones relative to Sa covered by the salience image
BINS_PER_SEMI = 5  # 20-cent bins, same resolution as RMVPE/CREPE
N_BINS = (REL_HI - REL_LO) * BINS_PER_SEMI
CLS_LO, CLS_HI = -12, 29  # note classes: lower Sa .. upper-upper Ni; class 0 = no note
N_CLASSES = CLS_HI - CLS_LO + 2
ENERGY_FLOOR_DB = 60.0


def rel_to_class(rel: np.ndarray | int) -> np.ndarray | int:
    return np.clip(rel, CLS_LO, CLS_HI) - CLS_LO + 1


def class_to_rel(c: np.ndarray | int) -> np.ndarray | int:
    return c - 1 + CLS_LO


def normalise_energy_db(db: np.ndarray, voiced: np.ndarray | None = None) -> np.ndarray:
    ref_pool = db[voiced] if voiced is not None and voiced.sum() > 20 else db
    ref = np.percentile(ref_pool, 95) if len(ref_pool) else 0.0
    return np.clip((db - ref + ENERGY_FLOOR_DB) / ENERGY_FLOOR_DB, 0.0, 1.0).astype(np.float32)


def _to_grid(times: np.ndarray, values: np.ndarray, n_out: int) -> np.ndarray:
    """Nearest-frame resampling onto the 10 ms grid (pyin's hop is 9.98 ms, which drifts over a song)."""
    grid = np.arange(n_out) * HOP_S
    idx = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    prev = np.clip(idx - 1, 0, len(times) - 1)
    use_prev = np.abs(times[prev] - grid) < np.abs(times[idx] - grid)
    return values[np.where(use_prev, prev, idx)]


def track_features(track, sa_hz: float, voicing_thr: float, fix_octaves: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(rel_cents, energy) on the 10 ms grid from an F0Track and the song's Sa frequency."""
    from ..config import Config
    from ..notes import fix_octave_errors

    times = np.asarray(track.times, dtype=np.float64)
    n_out = int(np.floor(times[-1] / HOP_S)) + 1 if len(times) else 0
    f0 = np.asarray(track.f0_hz, dtype=np.float64)
    conf = np.asarray(track.conf, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 1200.0 * np.log2(f0 / sa_hz)
    rel = np.where(np.isfinite(rel) & (conf >= voicing_thr), rel, np.nan)
    rel = _to_grid(times, rel, n_out)
    if fix_octaves:
        rel, _ = fix_octave_errors(rel, HOP_S, Config())
    if track.rms is not None and len(track.rms) == len(times):
        db = 20 * np.log10(np.asarray(track.rms, dtype=np.float64) + 1e-6)
        energy = normalise_energy_db(_to_grid(times, db, n_out), np.isfinite(rel))
    else:
        energy = np.where(np.isfinite(rel), 0.8, 0.1).astype(np.float32)
    return rel.astype(np.float32), energy


def salience_numpy(rel_cents: np.ndarray, sigma_cents: float = 25.0) -> np.ndarray:
    """(T, N_BINS) Gaussian bump at each voiced frame's pitch (reference implementation of the torch renderer)."""
    centers = REL_LO * 100.0 + (np.arange(N_BINS) + 0.5) * (100.0 / BINS_PER_SEMI)
    r = np.nan_to_num(rel_cents, nan=-1e6)[:, None]
    return np.exp(-0.5 * ((r - centers[None]) / sigma_cents) ** 2).astype(np.float32)
