"""Tempo / beat grid from the full mix (used for sustain marks and diagnostics)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BeatGrid:
    bpm: float
    beat_times: np.ndarray

    @property
    def period(self) -> float:
        return 60.0 / self.bpm if self.bpm > 0 else 0.0


def track_beats(mix: np.ndarray, sr: int) -> BeatGrid | None:
    try:
        import librosa

        tempo, beats = librosa.beat.beat_track(y=mix.astype(np.float32), sr=sr, units="time")
        bpm = float(np.atleast_1d(tempo)[0])
        if bpm <= 0 or len(beats) < 4:
            return None
        return BeatGrid(bpm=bpm, beat_times=np.asarray(beats))
    except Exception:
        return None
