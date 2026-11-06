"""Synthetic training windows: corpus melodies -> expressive pitch contours with tracker artefacts.

The notation corpus gives real melodies (Sa-relative notes with dot lengths) but no audio. We render each melody
the way a singer or flautist would perform it and a pitch tracker would hear it, and keep the written notes as
frame labels. What the model must learn is exactly what the notation writer does: ignore vibrato, scoops,
slides, kann/murki ornaments, reverb tails and octave glitches, keep quick written passing notes, split
repeated notes at syllable breaks.

Every knob is randomised per window (domain randomisation) so the model does not overfit one singer or tracker.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass

import numpy as np

from .features import CLS_HI, CLS_LO, HOP_S, normalise_energy_db, rel_to_class

SIL = 0


@dataclass
class Window:
    rel_cents: np.ndarray  # (T,) float32, NaN = unvoiced
    energy: np.ndarray  # (T,) float32 in [0, 1]
    cls: np.ndarray  # (T,) int64 note class (0 = no note)
    onset: np.ndarray  # (T,) float32 1 at written note starts


def load_melodies(jsonl_path, kinds=("sargam", "piano"), split: str = "train", val_frac: float = 0.05) -> list[dict]:
    """Songs as lists of lines of [rel, units, chunk_start]. Split by a stable hash of the slug."""
    out = []
    with open(jsonl_path) as f:
        for rec in map(json.loads, f):
            if rec["kind"] not in kinds:
                continue
            is_val = (zlib.crc32(rec["slug"].encode()) % 1000) < val_frac * 1000
            if (split == "val") != is_val:
                continue
            lines = [ln["notes"] for ln in rec["lines"] if len(ln["notes"]) >= 2]
            lines = [l for l in lines if all(-17 <= n[0] <= 34 for n in l)]
            if lines:
                out.append({"slug": rec["slug"], "kind": rec["kind"], "lines": lines})
    return out


def _u(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class Performer:
    """Samples one 'performance style' and renders melodies with it."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        r = rng
        self.unit_s = _u(r, (0.09, 0.32))
        self.dur_sd = _u(r, (0.05, 0.25))
        self.chunk_gap_p = _u(r, (0.2, 0.9))
        self.consonant_p = _u(r, (0.1, 0.6))
        self.repeat_gap_p = _u(r, (0.6, 1.0))
        self.trans_s = (0.015, _u(r, (0.04, 0.14)))
        self.meend_p = _u(r, (0.0, 0.15))
        self.overshoot_p = _u(r, (0.0, 0.5))
        self.scoop_p = _u(r, (0.0, 0.7))
        self.fall_p = _u(r, (0.0, 0.5))
        self.vib_p = _u(r, (0.0, 0.9))
        self.vib_depth = (5.0, _u(r, (15, 80)))
        self.vib_rate = (4.5, _u(r, (5.5, 7.5)))
        self.kann_p = _u(r, (0.0, 0.12))
        self.murki_p = _u(r, (0.0, 0.06))
        self.intonation_sd = _u(r, (3, 20))
        self.drift_sd = _u(r, (0, 15))
        self.jitter_sd = _u(r, (1, 9))
        self.tuning = _u(r, (-25, 25))
        noisy = _u(r, (0.0, 2.0))  # tracker / separation quality
        self.dropout_p = 0.15 * noisy
        self.octave_p = 0.02 * noisy
        self.fifth_p = 0.004 * noisy
        self.bleed_p = 0.02 * noisy
        self.tail_p = _u(r, (0.0, 0.6)) * min(1.0, noisy)
        self.spurious_p = 0.25 * noisy
        self.level_sd = _u(r, (1, 5))
        self.trans_unvoiced_p = _u(r, (0.0, 0.7))  # trackers lose voicing during fast pitch changes
        self.onset_unvoiced_p = _u(r, (0.0, 0.6))  # consonant at the start of a syllable
        self.gap_floor_db = _u(r, (12, 45))  # how far the separated stem drops in long silences
        self.dip_db = (2.0, _u(r, (6, 22)))  # syllable-break dips are shallow in real stems
        self.tremolo_db = _u(r, (0, 4))

    # ------------------------------------------------------------ timeline

    def schedule(self, lines: list[list[list[int]]], kind: str) -> list[dict]:
        """Notes with written start/end (s) and how they connect to the next note."""
        r = self.rng
        notes: list[dict] = []
        t = _u(r, (0.0, 0.6))
        for li, line in enumerate(lines):
            for k, (rel, units, chunk_start) in enumerate(line):
                if kind == "piano":
                    units = int(r.choice([1, 1, 1, 2, 2, 3]))
                dur = max(0.06, units * self.unit_s * float(np.exp(r.normal(0, self.dur_sd))))
                gap = 0.0
                if k == len(line) - 1:
                    gap = _u(r, (0.2, 1.6)) if li < len(lines) - 1 else 0.3
                else:
                    nxt = line[k + 1]
                    if nxt[2] and r.random() < self.chunk_gap_p:
                        gap = _u(r, (0.04, 0.35))
                    elif nxt[0] == rel and r.random() < self.repeat_gap_p:
                        gap = _u(r, (0.02, 0.09))
                    elif r.random() < self.consonant_p:
                        gap = _u(r, (0.02, 0.08))
                if gap < 0.1:  # a consonant break is part of the note's written length
                    gap = min(gap, 0.6 * dur)
                    notes.append({"rel": int(rel), "start": t, "end": t + dur - gap, "gap_after": gap})
                    t += dur
                else:  # a breath comes on top of it
                    notes.append({"rel": int(rel), "start": t, "end": t + dur, "gap_after": gap})
                    t += dur + gap
        return notes

    # ------------------------------------------------------------ rendering

    def render(self, notes: list[dict], scale_pcs: set[int]) -> Window:
        r = self.rng
        T = int(np.ceil(notes[-1]["end"] / HOP_S)) + int(0.4 / HOP_S)
        tt = np.arange(T) * HOP_S
        pitch = np.full(T, np.nan)
        cls = np.zeros(T, dtype=np.int64)
        onset = np.zeros(T, dtype=np.float32)
        level = np.full(T, -48.0)

        def fr(t):
            return int(round(t / HOP_S))

        for i, n in enumerate(notes):
            i0, i1 = fr(n["start"]), max(fr(n["start"]) + 1, fr(n["end"]))
            target = 100.0 * n["rel"] + r.normal(0, self.intonation_sd)
            seg_t = tt[i0:i1] - n["start"]
            seg = np.full(i1 - i0, target)
            L = len(seg)
            dur = L * HOP_S
            # vibrato on long notes
            if dur > 0.3 and r.random() < self.vib_p:
                depth, rate = _u(r, self.vib_depth), _u(r, self.vib_rate)
                delay = _u(r, (0.05, 0.3))
                ramp = _smoothstep((seg_t - delay) / 0.15)
                seg += depth * ramp * np.sin(2 * np.pi * rate * seg_t + r.uniform(0, 2 * np.pi))
            prev = notes[i - 1] if i else None
            legato = prev is not None and prev["gap_after"] == 0.0
            if not legato and r.random() < self.scoop_p:  # approach from below after a breath
                depth, length = _u(r, (40, 250)), _u(r, (0.03, 0.15))
                seg -= depth * (1 - _smoothstep(seg_t / length))
            if n["gap_after"] > 0 and r.random() < self.fall_p:  # pitch falls off at the release
                depth, length = _u(r, (30, 200)), _u(r, (0.03, 0.09))
                seg -= depth * _smoothstep((seg_t - (dur - length)) / length)
            # ornaments that the notation does not write: kann (grace) or murki at the note start
            if dur > 0.15 and r.random() < self.kann_p:
                up = self._neighbour(n["rel"], scale_pcs, +1)
                k = min(L // 3, int(_u(r, (0.03, 0.07)) / HOP_S))
                seg[:k] = 100.0 * up + r.normal(0, 8)
            elif dur > 0.25 and r.random() < self.murki_p:
                up, down = self._neighbour(n["rel"], scale_pcs, +1), self._neighbour(n["rel"], scale_pcs, -1)
                pos = 0
                for p in (up, n["rel"], down):
                    k = int(_u(r, (0.035, 0.06)) / HOP_S)
                    seg[pos : pos + k] = 100.0 * p
                    pos += k
            pitch[i0:i1] = seg
            cls[i0:i1] = rel_to_class(n["rel"]) if CLS_LO <= n["rel"] <= CLS_HI else SIL
            onset[i0] = 1.0
            lv = r.normal(-8, self.level_sd)
            level[i0:i1] = lv - _u(r, (0, 5)) * seg_t / max(dur, 1e-3) + self.tremolo_db * np.sin(2 * np.pi * _u(r, (1, 4)) * seg_t)
            # legato transition from the previous note: a smooth ramp centred on the written boundary
            if legato and i0 > 0 and np.isfinite(pitch[i0 - 1]):
                a = 100.0 * prev["rel"]
                b = target
                meend = r.random() < self.meend_p
                tau = _u(r, (0.12, 0.4)) if meend else _u(r, self.trans_s)
                h = max(1, int(tau / 2 / HOP_S))
                j0, j1 = max(fr(prev["start"]) + 1, i0 - h), min(i1 - 1, i0 + h)
                if j1 > j0:
                    x = (np.arange(j0, j1) - j0) / (j1 - j0)
                    shape = x if meend else _smoothstep(x)
                    base = pitch[j0:j1] - np.where(np.arange(j0, j1) < i0, a, b)  # keep vibrato etc.
                    pitch[j0:j1] = a + (b - a) * shape + 0.5 * base
                    if r.random() < self.overshoot_p and j1 < i1:
                        frac, decay = _u(r, (0.05, 0.35)), _u(r, (0.03, 0.08))
                        x2 = (np.arange(j1, i1) - j1) * HOP_S
                        pitch[j1:i1] += (b - a) * frac * np.exp(-x2 / decay) * np.minimum(1, x2 / 0.02 + 0.5)
        self._fill_gaps_level(level, notes)
        level = np.convolve(np.pad(level, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        # slow drift, tuning residual and frame jitter
        drift = np.cumsum(r.normal(0, 1, T))
        drift = drift - np.convolve(drift, np.ones(101) / 101, mode="same")
        drift = drift / (np.std(drift) + 1e-6) * self.drift_sd
        jitter = np.convolve(r.normal(0, self.jitter_sd, T), np.ones(3) / 3, mode="same")
        pitch = pitch + drift + jitter + self.tuning
        self._tracker_artefacts(pitch, cls, level, notes)
        level += r.normal(0, 1.0, T)
        energy = normalise_energy_db(level, np.isfinite(pitch))
        return Window(pitch.astype(np.float32), energy, cls, onset)

    def _fill_gaps_level(self, level, notes) -> None:
        r = self.rng
        T = len(level)
        loud = float(np.median([level[int(n["start"] / HOP_S)] for n in notes]))
        floor = loud - self.gap_floor_db
        level[: int(notes[0]["start"] / HOP_S)] = floor
        for i, n in enumerate(notes):
            g0 = max(1, int(round(n["end"] / HOP_S)))
            g1 = int(round(notes[i + 1]["start"] / HOP_S)) if i + 1 < len(notes) else T
            if g1 <= g0:
                continue
            if n["gap_after"] < 0.1 and i + 1 < len(notes):
                level[g0:g1] = min(level[g0 - 1], level[min(T - 1, g1)]) - _u(r, self.dip_db)
            else:
                level[g0:g1] = floor + r.normal(0, 2)

    def _neighbour(self, rel: int, scale_pcs: set[int], direction: int) -> int:
        for step in range(1, 4):
            cand = rel + direction * step
            if cand % 12 in scale_pcs:
                return cand
        return rel + 2 * direction

    def _tracker_artefacts(self, pitch, cls, level, notes) -> None:
        r = self.rng
        T = len(pitch)
        for i, n in enumerate(notes):
            i0, i1 = int(n["start"] / HOP_S), int(n["end"] / HOP_S)
            L = i1 - i0
            if L < 4:
                continue
            legato = i > 0 and notes[i - 1]["gap_after"] == 0.0
            if legato and r.random() < self.trans_unvoiced_p:
                pitch[max(0, i0 - int(r.integers(1, 6))) : i0 + int(r.integers(1, 6))] = np.nan
            elif not legato and r.random() < self.onset_unvoiced_p:
                pitch[i0 : i0 + min(L // 3, int(r.integers(1, 7)))] = np.nan
            for _ in range(2):
                if r.random() < self.dropout_p:  # unvoiced holes inside a note
                    k = int(_u(r, (0.02, 0.12)) / HOP_S)
                    s = int(r.integers(i0, max(i0 + 1, i1 - k)))
                    pitch[s : s + k] = np.nan
            if r.random() < self.octave_p:  # octave glitches, mostly near the onset
                k = max(2, int(_u(r, (0.03, 0.25)) / HOP_S))
                s = i0 if r.random() < 0.6 else int(r.integers(i0, max(i0 + 1, i1 - k)))
                pitch[s : min(i1, s + k)] += 1200.0 * r.choice([-1, 1])
            if r.random() < self.fifth_p:
                k = max(2, int(_u(r, (0.03, 0.15)) / HOP_S))
                pitch[i0 : i0 + k] += 1900.0
            if r.random() < self.bleed_p:  # a backing harmony line wins for a moment
                k = max(3, int(_u(r, (0.05, 0.2)) / HOP_S))
                s = int(r.integers(i0, max(i0 + 1, i1 - k)))
                pitch[s : s + k] += 100.0 * r.choice([3, 4, -3, -4, 7, -5])
        # reverb tails and bleed in the silences (labelled no-note)
        voiced = np.isfinite(pitch)
        for n in notes:
            g0 = int(n["end"] / HOP_S)
            glen = int(n["gap_after"] / HOP_S)
            if glen < 5:
                continue
            if r.random() < self.tail_p and 0 < g0 < len(pitch) and np.isfinite(pitch[g0 - 1]):
                k = min(glen - 1, int(_u(r, (0.05, 0.25)) / HOP_S), len(pitch) - g0)
                pitch[g0 : g0 + k] = pitch[g0 - 1] + r.normal(0, 5, k)
                level[g0 : g0 + k] = np.minimum(level[g0 : g0 + k] + 6, level[g0 - 1] - np.linspace(3, 15, k))
            if glen > 20 and r.random() < self.spurious_p:
                k = min(glen - 5, int(_u(r, (0.05, 0.5)) / HOP_S))
                s = g0 + int(r.integers(2, max(3, glen - k)))
                k = min(k, len(pitch) - s)
                if k <= 0:
                    continue
                base = 100.0 * (n["rel"] + r.choice([-5, -3, 0, 4, 7, 12]))
                pitch[s : s + k] = base + r.normal(0, 30, k).cumsum() / np.sqrt(np.arange(1, k + 1))
                level[s : s + k] = np.maximum(level[s : s + k], level[s : s + k] + r.uniform(3, 12))
        del voiced, T


def make_window(melodies: list[dict], rng: np.random.Generator, n_frames: int = 1024) -> Window:
    """One random training window of exactly n_frames."""
    song = melodies[int(rng.integers(len(melodies)))]
    perf = Performer(rng)
    lines = song["lines"]
    li = int(rng.integers(len(lines)))
    chosen: list[list[list[int]]] = []
    budget = n_frames * HOP_S + 2.0
    est = 0.0
    while est < budget:
        line = lines[li % len(lines)]
        chosen.append(line)
        est += sum(max(1, n[1]) for n in line) * perf.unit_s + 0.8
        li += 1
    notes = perf.schedule(chosen, song["kind"])
    scale_pcs = {n[0] % 12 for line in chosen for n in line}
    w = perf.render(notes, scale_pcs)
    T = len(w.cls)
    if T >= n_frames:
        s = int(rng.integers(0, T - n_frames + 1))
        sl = slice(s, s + n_frames)
        w = Window(w.rel_cents[sl], w.energy[sl], w.cls[sl], w.onset[sl])
    else:
        pad = n_frames - T
        w = Window(
            np.concatenate([w.rel_cents, np.full(pad, np.nan, np.float32)]),
            np.concatenate([w.energy, np.zeros(pad, np.float32)]),
            np.concatenate([w.cls, np.zeros(pad, np.int64)]),
            np.concatenate([w.onset, np.zeros(pad, np.float32)]),
        )
    return w
