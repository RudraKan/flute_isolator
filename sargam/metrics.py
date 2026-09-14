"""Transcription metrics against a reference sargam text."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .model import parse_pitch_name
from .swara import Token, collapse_tokens, parse_notation


def levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _strip_octave(sym: str) -> str:
    return sym.replace(".", "").replace("'", "")


def aligned_f1(pred: list[str], truth: list[str]) -> tuple[float, float, float]:
    """Needleman-Wunsch alignment: match +1, octave-only mismatch +0.5, mismatch -1, gap -1."""
    n, m = len(pred), len(truth)
    if n == 0 or m == 0:
        return 0.0, 0.0, 0.0
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = -i
    for j in range(1, m + 1):
        score[0][j] = -j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred[i - 1] == truth[j - 1]:
                s = 1.0
            elif _strip_octave(pred[i - 1]) == _strip_octave(truth[j - 1]):
                s = 0.5
            else:
                s = -1.0
            score[i][j] = max(score[i - 1][j - 1] + s, score[i - 1][j] - 1, score[i][j - 1] - 1)
    # traceback to count matches
    i, j, matched = n, m, 0.0
    while i > 0 and j > 0:
        if pred[i - 1] == truth[j - 1]:
            s = 1.0
        elif _strip_octave(pred[i - 1]) == _strip_octave(truth[j - 1]):
            s = 0.5
        else:
            s = -1.0
        if score[i][j] == score[i - 1][j - 1] + s:
            matched += max(s, 0.0)
            i, j = i - 1, j - 1
        elif score[i][j] == score[i - 1][j] - 1:
            i -= 1
        else:
            j -= 1
    p, r = matched / n, matched / m
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class Report:
    tonic_pred: str | None
    tonic_truth: str | None
    tonic_ok: bool | None
    tonic_relation: str | None
    ser: float
    ser_no_octave: float
    precision: float
    recall: float
    f1: float
    scale_jaccard: float
    n_pred: int
    n_truth: int

    def table(self) -> str:
        rel = f" ({self.tonic_relation})" if self.tonic_relation else ""
        return "\n".join(
            [
                f"tonic: pred {self.tonic_pred} truth {self.tonic_truth} -> {'OK' if self.tonic_ok else 'MISS' if self.tonic_ok is False else 'n/a'}{rel}",
                f"sargam error rate (SER): {self.ser:.3f}  octave-insensitive {self.ser_no_octave:.3f}   ({self.n_pred} pred vs {self.n_truth} truth symbols)",
                f"aligned precision {self.precision:.2f}  recall {self.recall:.2f}  F1 {self.f1:.2f}",
                f"scale (degree set) jaccard: {self.scale_jaccard:.2f}",
            ]
        )


def _relation(pred_pc: int, truth_pc: int) -> str | None:
    d = (pred_pc - truth_pc) % 12
    return {0: None, 7: "fifth (pred is Pa of truth)", 5: "fifth (pred is ma of truth)", 9: "relative (pred is Dha of truth)", 3: "relative (pred is ga of truth)"}.get(d, f"{d} semitones off")


def compare_tokens(pred_tokens: list[Token], truth_tokens: list[Token], pred_tonic: str | None, truth_tonic: str | None, keep_octave: bool = True) -> Report:
    pred = collapse_tokens(pred_tokens, keep_octave)
    truth = collapse_tokens(truth_tokens, keep_octave)
    ser = levenshtein(pred, truth) / max(1, len(truth))
    pred_o = collapse_tokens(pred_tokens, False)
    truth_o = collapse_tokens(truth_tokens, False)
    ser_no_octave = levenshtein(pred_o, truth_o) / max(1, len(truth_o))
    p, r, f = aligned_f1(pred, truth)
    ps = {t.degree for t in pred_tokens if t.kind == "note"}
    ts = {t.degree for t in truth_tokens if t.kind == "note"}
    jac = len(ps & ts) / len(ps | ts) if ps | ts else 1.0
    ok, rel = None, None
    if pred_tonic and truth_tonic:
        pp, tp = parse_pitch_name(pred_tonic), parse_pitch_name(truth_tonic)
        ok, rel = pp == tp, _relation(pp, tp)
    return Report(pred_tonic, truth_tonic, ok, rel, ser, ser_no_octave, p, r, f, jac, len(pred), len(truth))


def tokens_from_json(data: dict) -> list[Token]:
    """Prediction tokens from notes.json with ornaments/graces expanded to plain notes (reference style)."""
    sa = data["tonic"]["sa_midi"]
    out: list[Token] = []
    for n in data["notes"]:
        pre = list(n.get("ornament_before") or [])
        if n.get("grace_before") is not None:
            pre.append(n["grace_before"])
        for m in pre:
            rel = m - sa
            out.append(Token("note", rel % 12, rel // 12))
        out.append(Token("note", n["degree"], n["octave"]))
    return out


def compare_files(notes_json: str | Path, truth_txt: str | Path) -> Report:
    from .swara import Token

    data = json.loads(Path(notes_json).read_text())
    pred_tokens = tokens_from_json(data)
    truth_tonic, truth_phrases = parse_notation(Path(truth_txt).read_text())
    truth_tokens = [t for p in truth_phrases for t in p]
    return compare_tokens(pred_tokens, truth_tokens, data["tonic"]["name"], truth_tonic)
