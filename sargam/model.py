"""Shared data model for the pipeline. Pure dataclasses, no audio deps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class F0Track:
    """Frame-level pitch track. f0_hz is NaN where unvoiced."""

    times: np.ndarray
    f0_hz: np.ndarray
    conf: np.ndarray
    hop_s: float
    rms: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.times)


@dataclass
class NoteEvent:
    start: float
    end: float
    midi: int  # absolute 12-TET pitch after tuning correction
    cents_dev: float = 0.0  # mean deviation from the semitone centre
    degree: int = -1  # 0..11 relative to Sa (filled by assign_degrees)
    octave: int = 0  # relative to Sa's octave
    confidence: float = 0.0
    kind: Literal["note", "glide"] = "note"
    glide_to_midi: int | None = None
    grace_before: int | None = None  # midi of a kann swar
    ornament_before: list[int] | None = None  # midis of a murki cluster
    flags: set[str] = field(default_factory=set)
    phrase_id: int = -1

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Phrase:
    start: float
    end: float
    notes: list[NoteEvent]


@dataclass
class TuningEstimate:
    offset_cents: float
    source: str
    spread_cents: float


@dataclass
class TonicEstimate:
    pc: int
    name: str
    sa_hz: float
    sa_midi: int
    confidence: float
    margin: float
    label: str
    ambiguity: str | None
    scores: np.ndarray
    components: dict[str, np.ndarray]
    overridden: bool = False
    mode: str | None = None  # 'major' / 'minor' from the key profile, informational


@dataclass
class ScaleEstimate:
    name: str
    distance: int
    degrees_present: list[int]
    variants: dict[str, int]
    mixed: dict[str, bool]
    degree_mass: np.ndarray


@dataclass
class KeyChangeWarning:
    start_s: float
    end_s: float
    local_tonic_pc: int
    margin: float


@dataclass
class Flute:
    name: str
    sa_midi: int


@dataclass
class FluteRecommendation:
    sa_pc: int
    sa_midi: int
    rel_range: tuple[int, int]
    rel_range_str: str
    fits: bool
    unplayable_fraction: float
    out_of_range_phrases: list[int]
    recommended: Flute | None
    transposition_semitones: int
    advice: str


PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_ALIASES = {"DB": 1, "EB": 3, "GB": 6, "AB": 8, "BB": 10}


def pitch_name(pc: int) -> str:
    return PITCH_NAMES[pc % 12]


def parse_pitch_name(name: str) -> int:
    """'F#' -> 6, 'Gb' -> 6, 'c' -> 0."""
    key = name.strip().upper().replace("♯", "#").replace("♭", "B")
    if key in _FLAT_ALIASES:
        return _FLAT_ALIASES[key]
    for i, n in enumerate(PITCH_NAMES):
        if key == n:
            return i
    raise ValueError(f"unknown pitch name: {name!r}")


def midi_to_hz(midi: float, a4_hz: float = 440.0) -> float:
    return a4_hz * 2.0 ** ((midi - 69) / 12.0)


def hz_to_midi(hz: float, a4_hz: float = 440.0) -> float:
    return 69.0 + 12.0 * np.log2(hz / a4_hz)
