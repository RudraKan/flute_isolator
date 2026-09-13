"""Sargam -> audio synthesizer.

Used by the tests (ground truth without any dataset) and by `sargam audition`
(render transcribed notes as a sine track next to the vocal stem).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import NoteEvent, hz_to_midi
from .swara import Token, parse_notation


@dataclass
class SynthNote:
    start: float
    end: float
    f_start: float
    f_end: float  # != f_start for a glide
    truth: NoteEvent | None = None


@dataclass
class SynthResult:
    vocal: np.ndarray
    mix: np.ndarray
    sr: int
    truth: list[NoteEvent]
    sa_midi: int
    sa_hz: float
    tuning_cents: float
    beat_s: float
    phrase_starts: list[float] = field(default_factory=list)


def _adsr(n: int, sr: int, a_ms=20.0, d_ms=50.0, sustain=0.85, r_ms=60.0) -> np.ndarray:
    a, d, r = (int(sr * x / 1000) for x in (a_ms, d_ms, r_ms))
    a, d, r = min(a, n // 3), min(d, n // 3), min(r, n // 3)
    env = np.full(n, sustain, dtype=np.float64)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if d > 0:
        env[a : a + d] = np.linspace(1, sustain, d)
    if r > 0:
        env[n - r :] = np.linspace(env[n - r - 1] if n - r - 1 >= 0 else sustain, 0, r)
    return env


def additive_tone(
    f_track: np.ndarray, sr: int, harmonics: int = 6, rolloff: float = 1.3, jitter: float = 0.0
) -> np.ndarray:
    """Additive synthesis from an instantaneous-frequency track (Hz per sample)."""
    phase = 2 * np.pi * np.cumsum(f_track) / sr
    y = np.zeros_like(f_track)
    for h in range(1, harmonics + 1):
        y += (1.0 / h**rolloff) * np.sin(h * phase)
    y *= _adsr(len(y), sr)
    return y


def render_notes(
    notes: list[SynthNote],
    sr: int,
    total_s: float,
    vibrato_cents: float = 0.0,
    vibrato_hz: float = 5.5,
    harmonics: int = 6,
) -> np.ndarray:
    y = np.zeros(int(total_s * sr) + sr, dtype=np.float64)
    for n in notes:
        i0, i1 = int(n.start * sr), int(n.end * sr)
        L = i1 - i0
        if L <= 0:
            continue
        t = np.arange(L) / sr
        f = np.linspace(n.f_start, n.f_end, L) if n.f_end != n.f_start else np.full(L, n.f_start)
        if vibrato_cents > 0:
            # vibrato ramps in over the first 80 ms so onsets stay clean
            ramp = np.clip(t / 0.08, 0, 1)
            f = f * 2 ** (ramp * vibrato_cents * np.sin(2 * np.pi * vibrato_hz * t) / 1200)
        y[i0:i1] += additive_tone(f, sr, harmonics=harmonics)
    return y


def synthesize(
    sargam: str,
    sa_hz: float = 261.63,
    bpm: float = 120.0,
    sr: int = 22050,
    vibrato_cents: float = 0.0,
    vibrato_hz: float = 5.5,
    glide_frac: float = 0.5,
    ornament_ms: float = 50.0,
    grace_ms: float = 50.0,
    harmonics: int = 6,
    drone_db: float | None = None,
    pad: list[tuple[int, ...]] | None = None,
    pad_db: float = -18.0,
    pad_beats: int = 2,
    noise_db: float | None = -45.0,
    phrase_gap_beats: float = 1.0,
    seed: int = 0,
) -> SynthResult:
    """Render a sargam string. Each note token = 1 beat (+1 per sustain dash), rest = 1 beat,
    phrases separated by `phrase_gap_beats` of silence.

    pad: list of chord shapes as semitone offsets from Sa, cycled every `pad_beats` beats,
    rendered as a soft sustained pad in the mix only (the classic relative-minor trap).
    """
    _, phrases = parse_notation(sargam)
    if not phrases:
        raise ValueError("no notation found")
    beat = 60.0 / bpm
    sa_midi_f = hz_to_midi(sa_hz)
    sa_midi = int(round(sa_midi_f))
    tuning_cents = (sa_midi_f - sa_midi) * 100.0

    def f_of(deg: int, octv: int) -> float:
        return sa_hz * 2 ** ((deg + 12 * octv) / 12.0)

    def midi_of(deg: int, octv: int) -> int:
        return sa_midi + deg + 12 * octv

    synth_notes: list[SynthNote] = []
    truth: list[NoteEvent] = []
    phrase_starts: list[float] = []
    t = beat * 0.5  # lead-in

    for pi, toks in enumerate(phrases):
        phrase_starts.append(t)
        i = 0
        while i < len(toks):
            tok: Token = toks[i]
            if tok.kind == "rest":
                t += beat
                i += 1
                continue
            if tok.kind == "sustain":  # dangling sustain: treat as rest
                t += beat
                i += 1
                continue
            beats = 1
            j = i + 1
            while j < len(toks) and toks[j].kind == "sustain":
                beats += 1
                j += 1
            dur = beats * beat
            t0 = t
            orn_midis: list[int] | None = None
            grace_midi: int | None = None
            if tok.ornament:
                orn_midis = []
                for d, o in tok.ornament:
                    synth_notes.append(SynthNote(t0, t0 + ornament_ms / 1000, f_of(d, o), f_of(d, o)))
                    orn_midis.append(midi_of(d, o))
                    t0 += ornament_ms / 1000
            if tok.grace:
                d, o = tok.grace
                synth_notes.append(SynthNote(t0, t0 + grace_ms / 1000, f_of(d, o), f_of(d, o)))
                grace_midi = midi_of(d, o)
                t0 += grace_ms / 1000
            t_end = t + dur
            f0 = f_of(tok.degree, tok.octave)
            ev = NoteEvent(
                start=t0, end=t_end, midi=midi_of(tok.degree, tok.octave),
                grace_before=grace_midi, ornament_before=orn_midis, phrase_id=pi,
            )
            if tok.glide_to:
                d2, o2 = tok.glide_to
                f1 = f_of(d2, o2)
                hold_end = t0 + (t_end - t0) * (1 - glide_frac)
                synth_notes.append(SynthNote(t0, hold_end, f0, f0))
                synth_notes.append(SynthNote(hold_end, t_end, f0, f1))
                ev.kind = "glide"
                ev.glide_to_midi = midi_of(d2, o2)
            else:
                synth_notes.append(SynthNote(t0, t_end, f0, f0))
            truth.append(ev)
            t = t_end
            i = j
        t += phrase_gap_beats * beat

    total = t + beat
    vocal = render_notes(synth_notes, sr, total, vibrato_cents, vibrato_hz, harmonics)
    peak = np.max(np.abs(vocal)) or 1.0
    vocal = 0.5 * vocal / peak

    mix = vocal.copy()
    rng = np.random.default_rng(seed)
    n = len(mix)
    tt = np.arange(n) / sr
    if drone_db is not None:
        g = 10 ** (drone_db / 20)
        mix += g * (np.sin(2 * np.pi * sa_hz / 2 * tt) + 0.6 * np.sin(2 * np.pi * sa_hz * 1.5 / 2 * tt))
    if pad:
        g = 10 ** (pad_db / 20)
        seg = int(pad_beats * beat * sr)
        for k, i0 in enumerate(range(0, n, seg)):
            chord = pad[k % len(pad)]
            L = min(seg, n - i0)
            env = _adsr(L, sr, a_ms=30, d_ms=30, sustain=0.9, r_ms=30)
            for off in chord:
                f = sa_hz / 2 * 2 ** (off / 12.0)
                mix[i0 : i0 + L] += g * env * np.sin(2 * np.pi * f * tt[:L])
    if noise_db is not None:
        mix += 10 ** (noise_db / 20) * rng.standard_normal(n)
        vocal = vocal + 10 ** ((noise_db - 6) / 20) * rng.standard_normal(n)

    return SynthResult(
        vocal=vocal.astype(np.float32), mix=mix.astype(np.float32), sr=sr, truth=truth,
        sa_midi=sa_midi, sa_hz=sa_hz, tuning_cents=tuning_cents, beat_s=beat,
        phrase_starts=phrase_starts,
    )


def render_note_events(notes: list[NoteEvent], sr: int, total_s: float, a4_hz: float = 440.0) -> np.ndarray:
    """Sine rendering of transcribed NoteEvents (for auditioning)."""
    sn = []
    for n in notes:
        f0 = a4_hz * 2 ** ((n.midi + n.cents_dev / 100 - 69) / 12)
        f1 = f0
        if n.kind == "glide" and n.glide_to_midi is not None:
            f1 = a4_hz * 2 ** ((n.glide_to_midi - 69) / 12)
            mid = n.start + 0.5 * (n.end - n.start)
            sn.append(SynthNote(n.start, mid, f0, f0))
            sn.append(SynthNote(mid, n.end, f0, f1))
        else:
            sn.append(SynthNote(n.start, n.end, f0, f0))
    y = render_notes(sn, sr, total_s, harmonics=3)
    peak = np.max(np.abs(y)) or 1.0
    return (0.4 * y / peak).astype(np.float32)
