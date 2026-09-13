import numpy as np

from sargam.config import Config
from sargam.model import F0Track
from sargam.tuning import estimate_tuning_cents, refine_sa_hz, tuning_from_track

cfg = Config()


def track_from_midis(midis, offset_cents, n_per=50):
    f0 = np.concatenate([np.full(n_per, 440 * 2 ** ((m - 69) / 12 + offset_cents / 1200)) for m in midis])
    f0 = f0 * 2 ** (np.random.default_rng(0).normal(0, 4, len(f0)) / 1200)
    n = len(f0)
    return F0Track(times=np.arange(n) * 0.01, f0_hz=f0, conf=np.full(n, 0.9), hop_s=0.01)


def test_tuning_from_track_recovers_offset():
    for off in (0, 30, -40, 49):
        tr = track_from_midis([60, 62, 64, 65, 67, 69], off)
        est, _ = tuning_from_track(tr, cfg)
        assert abs(est - off) < 6, (off, est)


def test_estimate_without_mix():
    tr = track_from_midis([60, 64, 67], 20)
    e = estimate_tuning_cents(tr, None, 22050, cfg)
    assert e.source == "vocal" and abs(e.offset_cents - 20) < 6


def test_refine_sa_hz():
    tr = track_from_midis([60, 62, 64, 60], 25)
    hz = refine_sa_hz(tr, 25.0, 60, cfg)
    assert abs(1200 * np.log2(hz / (261.63 * 2 ** (25 / 1200)))) < 8
