"""Frame posteriors -> notes. Numpy only, so it runs inside training (eval) and in the app."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import HOP_S, class_to_rel


@dataclass
class DecodeParams:
    switch_cost: float = 3.0  # log-prob cost of changing class, reduced by onset evidence
    onset_weight: float = 2.0
    split_onset: float = 0.5  # onset peak inside a same-class run that splits it into a repeated note
    min_note_frames: int = 4
    min_split_frames: int = 6


@dataclass
class DecodedNote:
    start_f: int
    end_f: int  # exclusive
    rel: int  # semitones above Sa
    conf: float


def viterbi_classes(logp: np.ndarray, onset: np.ndarray, p: DecodeParams) -> np.ndarray:
    """Most likely class path where switching class costs switch_cost - onset_weight * log-odds of an onset."""
    T, C = logp.shape
    o = np.clip(onset, 1e-4, 1 - 1e-4)
    switch = -p.switch_cost + p.onset_weight * np.log(o / (1 - o)).clip(-5, 5) * 0.5
    score = logp[0].copy()
    back = np.zeros((T, C), dtype=np.int32)
    for t in range(1, T):
        best = int(np.argmax(score))
        stay = score
        move = score[best] + switch[t]
        take_move = move > stay
        back[t] = np.where(take_move, best, np.arange(C))
        score = np.where(take_move, move, stay) + logp[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _peaks(x: np.ndarray, thr: float) -> np.ndarray:
    if len(x) < 3:
        return np.zeros(0, dtype=np.int64)
    m = (x[1:-1] >= thr) & (x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:])
    return np.where(m)[0] + 1


def decode(cls_logits: np.ndarray, onset_logits: np.ndarray, p: DecodeParams = DecodeParams()) -> list[DecodedNote]:
    """cls_logits (T, C), onset_logits (T,) -> notes in frame units."""
    z = cls_logits - cls_logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    onset = 1.0 / (1.0 + np.exp(-onset_logits))
    path = viterbi_classes(logp, onset, p)
    probs = np.exp(logp)
    # runs of one class
    notes: list[DecodedNote] = []
    T = len(path)
    i = 0
    while i < T:
        j = i
        while j < T and path[j] == path[i]:
            j += 1
        c = int(path[i])
        if c != 0:
            # repeated notes: split the run at strong onset peaks away from its edges
            cuts = [k for k in _peaks(onset[i:j], p.split_onset) + i if k - i >= p.min_split_frames and j - k >= p.min_split_frames]
            bounds = [i] + cuts + [j]
            for a, b in zip(bounds[:-1], bounds[1:]):
                notes.append(DecodedNote(a, b, int(class_to_rel(c)), float(probs[a:b, c].mean())))
        i = j
    # drop fragments shorter than min_note_frames: glue to the neighbour they touch, else delete
    out: list[DecodedNote] = []
    for n in notes:
        if n.end_f - n.start_f >= p.min_note_frames:
            out.append(n)
        elif out and out[-1].end_f == n.start_f:
            out[-1].end_f = n.end_f
    return out


def to_note_events(decoded: list[DecodedNote], sa_midi: int, rel_cents: np.ndarray | None = None):
    from ..model import NoteEvent

    events = []
    for d in decoded:
        dev = 0.0
        if rel_cents is not None:
            seg = rel_cents[d.start_f : d.end_f]
            seg = seg[np.isfinite(seg)]
            if len(seg):
                dev = float(np.median(seg) - 100.0 * d.rel)
        events.append(NoteEvent(start=float(d.start_f * HOP_S), end=float(d.end_f * HOP_S), midi=int(sa_midi + d.rel),
                                cents_dev=float(dev), confidence=float(d.conf)))
    return events


def symbols(decoded: list[DecodedNote]) -> list[str]:
    """Collapsed plain-note symbols (same convention as swara.collapse_tokens)."""
    from ..swara import rel_to_symbol

    out: list[str] = []
    for d in decoded:
        s = rel_to_symbol(d.rel)
        if not out or out[-1] != s:
            out.append(s)
    return out
