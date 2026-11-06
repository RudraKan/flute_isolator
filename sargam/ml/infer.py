"""Run a trained NoteNet over a whole song (overlapping windows of the training length)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .decode import DecodeParams, decode, to_note_events

WINDOW = 1024
CONTEXT = 128  # frames discarded at each inner window edge


def default_checkpoint() -> Path:
    from ..audio import DATA_DIR

    return Path(os.environ.get("SARGAM_NOTENET", DATA_DIR / "models" / "notenet.pt"))


def _device() -> str:
    import torch

    if os.environ.get("SARGAM_DEVICE"):
        return os.environ["SARGAM_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def _load(path: str, device: str):
    from .model import load_checkpoint

    return load_checkpoint(path, device)


def predict_logits(model, rel_cents: np.ndarray, energy: np.ndarray, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(cls_logits (T, C), onset_logits (T,)) for a full-length feature sequence."""
    import torch

    device = device or next(model.parameters()).device.type
    T = len(rel_cents)
    hop = WINDOW - 2 * CONTEXT
    starts = list(range(0, max(1, T - 2 * CONTEXT), hop)) or [0]
    xs, es = [], []
    for s in starts:
        r = np.full(WINDOW, np.nan, np.float32)
        e = np.zeros(WINDOW, np.float32)
        seg = slice(s, min(T, s + WINDOW))
        r[: seg.stop - seg.start] = rel_cents[seg]
        e[: seg.stop - seg.start] = energy[seg]
        xs.append(r)
        es.append(e)
    cls_out = None
    on_out = np.zeros(T, np.float32)
    with torch.no_grad():
        for b in range(0, len(xs), 16):
            x = torch.from_numpy(np.stack(xs[b : b + 16])).to(device)
            e = torch.from_numpy(np.stack(es[b : b + 16])).to(device)
            c, o = model(x, e)
            c, o = c.float().cpu().numpy(), o.float().cpu().numpy()
            if cls_out is None:
                cls_out = np.zeros((T, c.shape[-1]), np.float32)
            for k in range(len(c)):
                s = starts[b + k]
                lo = 0 if s == 0 else CONTEXT
                hi = min(WINDOW, T - s)
                if s + WINDOW < T:
                    hi = min(hi, WINDOW - CONTEXT)
                if hi > lo:
                    cls_out[s + lo : s + hi] = c[k, lo:hi]
                    on_out[s + lo : s + hi] = o[k, lo:hi]
    return cls_out, on_out


def transcribe_features(model, rel_cents, energy, sa_midi: int, params: DecodeParams = DecodeParams()):
    cls_logits, onset_logits = predict_logits(model, rel_cents, energy)
    return to_note_events(decode(cls_logits, onset_logits, params), sa_midi, rel_cents)


def transcribe_track(track, sa_hz: float, sa_midi: int, voicing_thr: float, checkpoint: str | Path | None = None,
                     min_note_ms: float = 40.0):
    """NoteEvents for a pitch track, Sa given. Used by pipeline.analyze when notes_engine == 'notenet'.
    min_note_ms is the simplification knob: shorter decoded notes merge into their neighbour."""
    from .features import track_features

    path = Path(checkpoint or default_checkpoint())
    if not path.exists():
        raise FileNotFoundError(f"no NoteNet checkpoint at {path}; install one with `sargam model install best.pt` or set SARGAM_NOTENET")
    model = _load(str(path), _device())
    rel, energy = track_features(track, sa_hz, voicing_thr)
    params = DecodeParams(min_note_frames=max(1, int(round(min_note_ms / 10.0))))
    return transcribe_features(model, rel, energy, sa_midi, params)
