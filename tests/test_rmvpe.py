import numpy as np
import pytest

from sargam import rmvpe
from sargam.config import Config
from sargam.pitch import track_from_salience


def _bump(n_frames: int, centre_bin: float, height: float = 0.9) -> np.ndarray:
    bins = np.arange(rmvpe.N_BINS)
    return np.tile(height * np.exp(-0.5 * ((bins - centre_bin) / 1.5) ** 2), (n_frames, 1))


def test_bin_mapping():
    assert rmvpe.bin_to_hz(0) == pytest.approx(31.70, abs=0.01)
    # 60 bins of 20 cents = one octave
    assert rmvpe.bin_to_hz(160) / rmvpe.bin_to_hz(100) == pytest.approx(2.0)


def test_decode_local_average_is_sub_bin_accurate():
    sal = np.concatenate([_bump(5, 100.3), np.full((3, rmvpe.N_BINS), 0.01), _bump(2, 358.6)])
    f0, peak = rmvpe.decode(sal)
    cents_err = 1200 * np.log2(f0[:5] / rmvpe.bin_to_hz(100.3))
    assert np.all(np.abs(cents_err) < 2.0)
    assert np.all(np.isnan(f0[5:8]))  # peak 0.01 is below the 0.03 decoder floor
    assert np.allclose(peak[:5], peak[0]) and peak[0] > 0.85
    # bump at the top edge: the window is clipped by zero padding, still finite and close
    assert np.all(np.isfinite(f0[8:]))
    assert abs(1200 * np.log2(f0[8] / rmvpe.bin_to_hz(358.6))) < 15


def test_decode_empty():
    f0, peak = rmvpe.decode(np.zeros((0, rmvpe.N_BINS)))
    assert f0.shape == (0,) and peak.shape == (0,)


def test_track_from_salience_limits_range_and_sets_hop():
    sal = np.concatenate([_bump(4, 150.0), _bump(4, 10.0)])  # ~180 Hz, then ~35 Hz (below fmin)
    y16 = np.zeros(8 * rmvpe.HOP, dtype=np.float32)
    tr = track_from_salience(sal, y16, Config())
    assert tr.hop_s == pytest.approx(0.01)
    assert len(tr) == 8 and tr.times[1] == pytest.approx(0.01)
    assert np.all(np.isfinite(tr.f0_hz[:4])) and np.all(np.isnan(tr.f0_hz[4:]))
    assert tr.rms is not None and len(tr.rms) == 8


def test_backend_defaults_resolution():
    c = Config(pitch_backend="rmvpe").resolved()
    assert (c.voicing_thr, c.switch_penalty, c.simplify_min_ms) == (0.7, 16.0, 100.0)
    p = Config(pitch_backend="pyin").resolved()
    assert (p.voicing_thr, p.switch_penalty, p.simplify_min_ms) == (0.5, 12.0, 0.0)
    explicit = Config(pitch_backend="rmvpe", voicing_thr=0.4, simplify_min_ms=-1.0).resolved()
    assert (explicit.voicing_thr, explicit.switch_penalty, explicit.simplify_min_ms) == (0.4, 16.0, -1.0)


@pytest.mark.slow
def test_rmvpe_weights_on_synth_tone():
    """Runs the real network (downloads 181 MB once). 220 Hz then 330 Hz harmonic tone, then silence."""
    from sargam.pitch import extract_f0

    sr = 22050
    t = np.arange(int(3.0 * sr)) / sr
    f = np.where(t < 1.5, 220.0, 330.0)
    ph = 2 * np.pi * np.cumsum(f) / sr
    y = (0.3 * sum(np.sin(k * ph) / k for k in range(1, 6))).astype(np.float32)
    y[int(2.5 * sr):] = 0.0
    tr = extract_f0(y, sr, Config(), backend="rmvpe")
    assert len(tr) == pytest.approx(300, abs=2)
    cents = lambda f0, ref: 1200 * np.log2(f0 / ref)  # noqa: E731
    assert np.all(np.abs(cents(tr.f0_hz[20:130], 220.0)) < 10)
    assert np.all(np.abs(cents(tr.f0_hz[170:230], 330.0)) < 10)
    assert np.all(np.isnan(tr.f0_hz[265:]))
    assert tr.conf[50] > 0.8
