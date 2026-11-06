"""Wires the stages together. `analyze` works on arrays (testable without Demucs); `run` on a file."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import load_mono
from .beats import BeatGrid, track_beats
from .config import Config
from .flute import fold_octaves, recommend_flute
from .model import F0Track, Flute, FluteRecommendation, KeyChangeWarning, NoteEvent, Phrase, ScaleEstimate, TonicEstimate, TuningEstimate
from .notes import assign_degrees, choose_sa_octave, segment_notes, simplify_notes, split_phrases
from .pitch import extract_f0, load_track, save_track
from .tonic import Profiles, build_profiles, detect_key_changes, estimate_scale, estimate_tonic, snap_rare_degrees
from .tuning import estimate_tuning_cents, refine_sa_hz


@dataclass
class Result:
    source: str
    cfg: Config
    sr: int
    duration_s: float
    track: F0Track
    tuning: TuningEstimate
    notes: list[NoteEvent]
    phrases: list[Phrase]
    tonic: TonicEstimate
    scale: ScaleEstimate
    flute: FluteRecommendation
    beats: BeatGrid | None
    key_changes: list[KeyChangeWarning]
    profiles: Profiles
    separation_backend: str = ""
    log: list[str] = field(default_factory=list)


def analyze(
    vocal: np.ndarray,
    mix: np.ndarray | None,
    bass: np.ndarray | None,
    sr: int,
    cfg: Config = Config(),
    sa_override: str | None = None,
    flutes: list[Flute] | None = None,
    use_beats: bool = True,
    fold: bool = False,
    source: str = "",
    track: F0Track | None = None,
    other: np.ndarray | None = None,
) -> Result:
    log: list[str] = []
    cfg = cfg.resolved()
    if track is None:
        track = extract_f0(vocal, sr, cfg)
    tuning = estimate_tuning_cents(track, mix, sr, cfg)
    log.append(f"tuning {tuning.offset_cents:+.1f} cents ({tuning.source}, spread {tuning.spread_cents:.1f})")
    notes = segment_notes(track, tuning.offset_cents, cfg)
    phrases = split_phrases(notes, cfg.phrase_gap_s)
    profiles = build_profiles(notes, phrases, track, tuning.offset_cents, mix, bass, sr, cfg, other=other)
    tonic = estimate_tonic(profiles, cfg, override=sa_override)
    sa_midi = choose_sa_octave(notes, tonic.pc)
    tonic.sa_midi = sa_midi
    tonic.sa_hz = refine_sa_hz(track, tuning.offset_cents, sa_midi, cfg)
    scale = estimate_scale(profiles, tonic.pc, cfg)
    if cfg.snap_rare:
        k = snap_rare_degrees(notes, scale, tonic.pc, cfg)
        if k:
            log.append(f"snapped {k} short off-scale notes onto the detected scale")
    if cfg.simplify_min_ms > 0:
        before = len(notes)
        notes = simplify_notes(notes, cfg.simplify_min_ms, cfg.simplify_gap_ms)
        phrases = split_phrases(notes, cfg.phrase_gap_s)
        log.append(f"simplified {before} -> {len(notes)} notes (fragments < {cfg.simplify_min_ms:.0f} ms merged)")
    key_changes = detect_key_changes(notes, phrases, tonic.pc, cfg, profiles)
    flute = recommend_flute(notes, tonic.pc, flutes, cfg, sa_midi=sa_midi)
    if fold and not flute.fits:
        fold_octaves(notes, sa_midi, cfg)
        flute = recommend_flute(notes, tonic.pc, flutes, cfg, sa_midi=sa_midi)
        log.append("folded out-of-range notes by octaves")
    assign_degrees(notes, sa_midi)
    beats = track_beats(mix, sr) if (use_beats and mix is not None) else None
    duration = float(len(vocal) / sr)
    return Result(
        source=source, cfg=cfg, sr=sr, duration_s=duration, track=track, tuning=tuning, notes=notes, phrases=phrases,
        tonic=tonic, scale=scale, flute=flute, beats=beats, key_changes=key_changes, profiles=profiles, log=log,
    )


def run(
    path: str | Path,
    cfg: Config = Config(),
    sa_override: str | None = None,
    flutes: list[Flute] | None = None,
    use_beats: bool = True,
    fold: bool = False,
    model: str = "htdemucs",
    no_separate: bool = False,
    progress=None,
) -> Result:
    path = Path(path)
    say = progress or (lambda *_: None)
    say(f"loading {path.name}")
    mix = load_mono(path, cfg.sr)
    if no_separate:
        vocal, bass, other, backend = mix, None, None, "none"
    else:
        from .separate import separate

        say("separating vocals (Demucs)")
        stems = separate(path, cfg.sr, model=model)
        vocal, bass, other, backend = stems.vocals, stems.bass, stems.other, stems.backend
        say(f"separated with {backend}")
    say(f"tracking pitch ({cfg.pitch_backend})")
    track = None
    if not no_separate:
        from .audio import cache_dir

        f0_path = cache_dir(path) / f"f0_{cfg.pitch_backend}.npz"
        if f0_path.exists():
            track = load_track(f0_path)
        else:
            track = extract_f0(vocal, cfg.sr, cfg)
            save_track(f0_path, track)
    res = analyze(vocal, mix, bass, cfg.sr, cfg, sa_override, flutes, use_beats, fold, source=str(path), other=other, track=track)
    res.separation_backend = backend
    return res
