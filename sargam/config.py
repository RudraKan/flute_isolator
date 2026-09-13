"""All tunable parameters in one frozen dataclass."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # audio / pitch
    sr: int = 22050
    hop_ms: float = 10.0
    fmin: float = 80.0
    fmax: float = 1000.0
    voicing_thr: float = 0.5
    pitch_backend: str = "pyin"

    # tuning
    tuning_smooth_bins: float = 5.0
    tuning_disagree_cents: float = 20.0
    tuning_smooth_ms: float = 150.0  # vibrato averages out over this window
    tuning_min_sharpness: float = 0.02  # peak bin share of the 100-bin vocal histogram

    # segmentation: preprocessing
    bridge_gap_frames: int = 5
    bridge_max_cents: float = 300.0
    medfilt_frames: int = 5
    octave_med_window_ms: float = 500.0

    # segmentation: viterbi
    sigma_cents: float = 35.0
    switch_penalty: float = 12.0
    per_semitone_penalty: float = 0.2
    max_jump_semitones: int = 24
    silence_penalty: float = 4.0
    unvoiced_in_note_cost: float = 1.5
    voiced_in_sil_cost: float = 8.0

    # segmentation: post-processing
    min_note_ms: float = 60.0
    merge_gap_ms: float = 80.0
    rms_dip_db: float = 6.0
    rms_dip_min_ms: float = 20.0
    rms_dip_max_ms: float = 80.0
    repeat_split_min_ms: float = 250.0
    vibrato_elem_max_ms: float = 100.0
    grace_max_ms: float = 80.0
    grace_next_min_ms: float = 150.0
    ornament_window_ms: float = 250.0
    ornament_max_span_cents: float = 500.0
    glide_slope_cents_per_s: float = 400.0
    glide_min_ms: float = 120.0
    slide_in_max_ms: float = 150.0  # shorter glides landing on the next note are approach slides
    min_note_conf: float = 0.3
    phrase_gap_s: float = 0.35

    # tonic scoring weights
    w_dur: float = 0.20
    w_end: float = 0.15
    w_key: float = 0.30  # key-profile correlation on the separated accompaniment
    w_key_unseparated: float = 0.15  # same component when only the mix is available
    w_bass: float = 0.15
    w_mix: float = 0.07
    w_frame: float = 0.08
    w_onset: float = 0.05
    key_temp: float = 0.08
    softmax_temp: float = 0.03
    margin_high: float = 0.06
    margin_medium: float = 0.025
    degree_absent_thr: float = 0.02
    mixed_thr: float = 0.25
    drone_peak_thr: float = 0.35
    snap_rare: bool = True  # snap short, low-confidence notes on unused degrees to the scale
    snap_degree_max_mass: float = 0.06  # a degree with less mass than this may be an artefact degree...
    snap_degree_max_conf: float = 0.55  # ...if its notes are collectively this unconfident
    snap_max_ms: float = 450.0
    snap_max_conf: float = 0.6
    keychange_window_s: float = 30.0
    keychange_hop_s: float = 15.0
    keychange_margin: float = 0.04

    # flute
    flute_low: int = -5  # .P
    flute_high: int = 24  # S''
    flute_comfort_high: int = 19  # P'
    unplayable_tol: float = 0.02

    # rendering
    phrases_per_block: int = 4


DEFAULT = Config()
