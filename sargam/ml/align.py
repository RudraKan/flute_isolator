"""Forced alignment of a reference notation to a song, for training on real (audio, notation) pairs.

Notation pages carry no timing. Given the model's frame posteriors and the written note sequence, a left-to-right
HMM finds when each written note is sung: states alternate gap / note (gap = no-note class, skippable between
legato notes), and a written note can be skipped at a cost (pages sometimes drop a repeat). Frames whose aligned
label the model finds very unlikely are masked out of the loss rather than trusted.
"""
from __future__ import annotations

import numpy as np

from .features import rel_to_class

IGNORE = -100


def _log_softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def align(cls_logits: np.ndarray, truth_rels: list[int], skip_cost: float = 12.0, gap_floor: float = -3.0,
          min_post: float = 0.02, min_frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """(labels (T,) class per frame or IGNORE, onsets (T,) 1 at each aligned note start)."""
    T = len(cls_logits)
    K = len(truth_rels)
    logp = _log_softmax(cls_logits.astype(np.float64))
    note_cls = np.array([rel_to_class(r) for r in truth_rels], dtype=np.int64)
    S = 2 * K + 1  # even = gap before note k (k = s // 2), odd = note k
    emit_gap = np.maximum(logp[:, 0], gap_floor)  # spurious sound in a gap should not drag the path
    NEG = -1e18
    score = np.full(S, NEG)
    score[0] = emit_gap[0]
    if K:
        score[1] = logp[0, note_cls[0]]
    back = np.zeros((T, S), dtype=np.int8)  # how many states we advanced
    odd = np.arange(S) % 2 == 1
    for t in range(1, T):
        cand = np.stack([
            score,
            np.concatenate([[NEG], score[:-1]]),
            np.where(odd, np.concatenate([[NEG, NEG], score[:-2]]), NEG),  # note -> next note, no gap
            np.where(odd, np.concatenate([[NEG] * 4, score[:-4]]) - skip_cost, NEG),  # skip one written note
        ])
        arg = cand.argmax(axis=0)
        score = cand[arg, np.arange(S)]
        emit = np.empty(S)
        emit[0::2] = emit_gap[t]
        emit[1::2] = logp[t, note_cls]
        score = score + emit
        back[t] = np.array([0, 1, 2, 4], dtype=np.int8)[arg]
    s = int(np.argmax(score[-3:]) + S - 3) if S >= 3 else int(np.argmax(score))
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    labels = np.zeros(T, dtype=np.int64)
    onsets = np.zeros(T, dtype=np.float32)
    is_note = path % 2 == 1
    labels[is_note] = note_cls[path[is_note] // 2]
    starts = np.where(is_note & np.concatenate([[True], path[1:] != path[:-1]]))[0]
    for a in starts:
        b = a
        while b < T and path[b] == path[a]:
            b += 1
        if b - a < min_frames:  # squeezed in to avoid a skip: not evidence of where the note is
            labels[a:b] = IGNORE
        else:
            onsets[a] = 1.0
    post = np.exp(logp[np.arange(T), np.maximum(labels, 0)])
    labels[(post < min_post) & (labels != IGNORE)] = IGNORE
    return labels, onsets


def truth_rels_from_text(text: str) -> list[int]:
    """Written notes (not collapsed: repeated notes stay separate) in semitones above Sa."""
    from ..swara import parse_notation

    _, phrases = parse_notation(text, "nw")
    return [t.degree + 12 * t.octave for p in phrases for t in p if t.kind == "note"]

