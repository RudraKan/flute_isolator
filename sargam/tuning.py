"""Global tuning offset (cents relative to A440) from the vocal f0 track and the mix."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from .config import Config
from .model import F0Track, TuningEstimate


def _circ_diff(a: float, b: float, period: float = 100.0) -> float:
    d = (a - b + period / 2) % period - period / 2
    return abs(d)


def _smooth_in_runs(c: np.ndarray, ok: np.ndarray, k: int) -> np.ndarray:
    """Moving average over k frames inside voiced runs (vibrato averages out)."""
    out = c.copy()
    n = len(c)
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= k:
            out[i:j] = uniform_filter1d(c[i:j], size=k, mode="nearest")
        i = j
    return out


def tuning_from_track(track: F0Track, cfg: Config) -> tuple[float, float]:
    """Returns (offset_cents in [-50, 50), peak sharpness = share of mass in the peak bin)."""
    f0 = np.asarray(track.f0_hz, dtype=np.float64)
    conf = np.asarray(track.conf, dtype=np.float64)
    ok = np.isfinite(f0) & (conf >= cfg.voicing_thr)
    if ok.sum() < 20:
        return 0.0, 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        cents_all = 1200.0 * np.log2(f0 / 440.0)
    cents_all = np.where(ok, cents_all, 0.0)
    k = max(1, int(round(cfg.tuning_smooth_ms / 1000.0 / track.hop_s)))
    smooth = _smooth_in_runs(cents_all, ok, k)
    c = smooth[ok]
    raw = cents_all[ok]
    stable = np.ones(len(c))
    d = np.abs(np.diff(raw, prepend=raw[0]))
    stable[d > 30] = 0.2
    w = conf[ok] * stable
    frac = np.mod(c, 100.0)
    hist, _ = np.histogram(frac, bins=100, range=(0, 100), weights=w)
    hist = gaussian_filter1d(hist, cfg.tuning_smooth_bins, mode="wrap")
    k = int(np.argmax(hist))
    y0, y1, y2 = hist[(k - 1) % 100], hist[k], hist[(k + 1) % 100]
    denom = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    off = (k + delta) % 100
    if off >= 50:
        off -= 100
    sharp = float(hist[k] / hist.sum()) if hist.sum() > 0 else 0.0
    return float(off), sharp


def tuning_from_mix(mix: np.ndarray, sr: int) -> float | None:
    try:
        import librosa

        t = librosa.estimate_tuning(y=mix.astype(np.float32), sr=sr, bins_per_octave=120)
        if t is None or not np.isfinite(t):
            return None
        return float(t) * 100.0
    except Exception:
        return None


def estimate_tuning_cents(track: F0Track, mix: np.ndarray | None, sr: int, cfg: Config) -> TuningEstimate:
    """The vocal histogram is what gets quantised, so it is the estimate; the mix value (biased by
    harmonics in librosa's estimator) is only a diagnostic, used when the vocal track is nearly empty."""
    voc, sharp = tuning_from_track(track, cfg)
    n_ok = int((np.isfinite(track.f0_hz) & (np.asarray(track.conf) >= cfg.voicing_thr)).sum())
    mx = tuning_from_mix(mix, sr) if mix is not None else None
    if n_ok < 20 and mx is not None:
        return TuningEstimate(offset_cents=mx, source="mix (no vocal)", spread_cents=0.0)
    spread = _circ_diff(mx, voc) if mx is not None else 0.0
    src = "vocal" if sharp >= cfg.tuning_min_sharpness else "vocal (flat histogram)"
    return TuningEstimate(offset_cents=voc, source=src, spread_cents=spread)


def refine_sa_hz(track: F0Track, tuning_cents: float, sa_midi: int, cfg: Config) -> float:
    """Median f0 of frames within +-50 cents of Sa (any octave), folded to sa_midi's octave."""
    f0 = np.asarray(track.f0_hz, dtype=np.float64)
    conf = np.asarray(track.conf, dtype=np.float64)
    ok = np.isfinite(f0) & (conf >= cfg.voicing_thr)
    base = 440.0 * 2 ** ((sa_midi - 69) / 12.0 + tuning_cents / 1200.0)
    if not ok.any():
        return base
    cents = 1200.0 * np.log2(f0[ok] / base)
    fold = (cents + 600) % 1200 - 600
    near = np.abs(fold) <= 50
    if near.sum() < 10:
        return base
    return float(base * 2 ** (np.median(fold[near]) / 1200.0))
