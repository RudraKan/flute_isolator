"""L0 tests: synthetic cents/conf arrays -> notes.py, no audio."""
import numpy as np
import pytest

from sargam.config import Config
from sargam.model import F0Track, NoteEvent
from sargam.notes import (
    assign_degrees, choose_sa_octave, fix_octave_errors, segment_naive, segment_notes, split_phrases,
)

HOP = 0.01
cfg = Config()


def make_track(spec, vibrato_cents=0.0, vibrato_hz=6.0, noise_cents=3.0, seed=0, rms_dips=()):
    """spec: list of (duration_s, midi or None for silence, optional dict(glide_to=midi)).
    Returns an F0Track with 10 ms hop."""
    rng = np.random.default_rng(seed)
    f0, conf, rms = [], [], []
    t = 0.0
    for item in spec:
        dur, midi = item[0], item[1]
        opts = item[2] if len(item) > 2 else {}
        n = int(round(dur / HOP))
        tt = np.arange(n) * HOP
        if midi is None:
            f0.extend([np.nan] * n)
            conf.extend([0.0] * n)
            rms.extend([opts.get("rms", 1e-4)] * n)
        else:
            m = np.full(n, float(midi))
            if "glide_to" in opts:
                hold = int(n * 0.5)
                m[hold:] = np.linspace(midi, opts["glide_to"], n - hold)
            c = 100 * (m - 69) + rng.normal(0, noise_cents, n)
            if vibrato_cents:
                c += vibrato_cents * np.sin(2 * np.pi * vibrato_hz * (t + tt))
            f0.extend(440 * 2 ** (c / 1200))
            conf.extend([0.9] * n)
            rms.extend([0.1] * n)
        t += dur
    rms = np.array(rms)
    for t0, t1 in rms_dips:
        rms[int(t0 / HOP) : int(t1 / HOP)] *= 0.2  # -14 dB
    n = len(f0)
    return F0Track(times=np.arange(n) * HOP, f0_hz=np.array(f0), conf=np.array(conf), hop_s=HOP, rms=rms)


def midis(notes):
    return [n.midi for n in notes]


def test_plain_sequence_both_segmenters():
    tr = make_track([(0.5, 60), (0.5, 62), (0.5, 64), (0.3, None), (0.5, 65), (0.5, 67)])
    for fn in (segment_notes, segment_naive):
        notes = fn(tr, 0.0, cfg)
        assert midis(notes) == [60, 62, 64, 65, 67], fn.__name__
    notes = segment_notes(tr, 0.0, cfg)
    assert abs(notes[0].start - 0.0) < 0.03 and abs(notes[0].end - 0.5) < 0.03
    assert abs(notes[3].start - 1.8) < 0.03


def test_tuning_offset_applied():
    tr = make_track([(0.5, 60), (0.5, 62)])
    # everything 40 cents sharp: without correction rounds the same, with -40 also the same; check +60 shift
    tr.f0_hz = tr.f0_hz * 2 ** (60 / 1200)
    assert midis(segment_notes(tr, 60.0, cfg)) == [60, 62]
    assert midis(segment_notes(tr, 0.0, cfg)) == [61, 63]


def test_vibrato_does_not_flicker():
    tr = make_track([(0.6, 60), (0.6, 64), (0.6, 62)], vibrato_cents=60, vibrato_hz=6)
    notes = segment_notes(tr, 0.0, cfg)
    assert midis(notes) == [60, 64, 62]


def test_short_dropout_bridged_long_gap_splits():
    # tracker dropout with no energy dip -> bridged into one note
    tr = make_track([(0.4, 60), (0.03, None, {"rms": 0.1}), (0.4, 60)])
    assert midis(segment_notes(tr, 0.0, cfg)) == [60]
    # same gap but with a real energy dip -> re-articulated note
    tr = make_track([(0.4, 60), (0.03, None), (0.4, 60)])
    assert midis(segment_notes(tr, 0.0, cfg)) == [60, 60]
    tr = make_track([(0.4, 60), (0.2, None), (0.4, 60)])
    assert midis(segment_notes(tr, 0.0, cfg)) == [60, 60]


def test_repeated_note_split_on_rms_dip():
    tr = make_track([(1.0, 60)], rms_dips=[(0.48, 0.52)])
    notes = segment_notes(tr, 0.0, cfg)
    assert midis(notes) == [60, 60]
    assert abs(notes[1].start - 0.5) < 0.05


def test_octave_jump_repaired():
    tr = make_track([(0.6, 60)])
    i0, i1 = 20, 30
    tr.f0_hz[i0:i1] *= 2  # 100 ms octave error in the middle
    notes = segment_notes(tr, 0.0, cfg)
    assert midis(notes) == [60]
    assert "octave_fixed" in notes[0].flags


def test_fix_octave_errors_mask():
    tr = make_track([(0.6, 60)])
    cents = 100 * (np.log2(tr.f0_hz / 440) * 12)
    cents[20:30] += 1200
    fixed, mask = fix_octave_errors(cents, HOP, cfg)
    assert mask[20:30].all() and not mask[:20].any()
    assert np.allclose(fixed[20:30], cents[20:30] - 1200)


def test_grace_note_attached():
    tr = make_track([(0.4, 60), (0.05, 62), (0.4, 60)])
    notes = segment_notes(tr, 0.0, cfg)
    assert midis(notes) == [60, 60]
    assert notes[1].grace_before == 62


def test_murki_cluster_attached():
    tr = make_track([(0.4, 60), (0.05, 62), (0.05, 60), (0.05, 58), (0.5, 60)])
    notes = segment_notes(tr, 0.0, cfg)
    assert midis(notes) == [60, 60]
    assert notes[1].ornament_before == [62, 60, 58]


def test_glide_detected_as_meend():
    tr = make_track([(0.5, 60, {"glide_to": 64}), (0.5, 64)])
    notes = segment_notes(tr, 0.0, cfg)
    assert midis(notes) == [60, 64]
    assert notes[0].kind == "glide" and notes[0].glide_to_midi == 64


def test_short_notes_merged_or_dropped():
    tr = make_track([(0.5, 60), (0.03, 67), (0.5, 62)])
    # a lone 30 ms blip contiguous with both neighbours merges into the closer one, no new note
    assert midis(segment_notes(tr, 0.0, cfg)) == [60, 62]


def test_split_phrases_and_degrees():
    notes = [NoteEvent(0, 0.5, 66), NoteEvent(0.5, 1.0, 68), NoteEvent(1.6, 2.0, 61), NoteEvent(2.0, 2.4, 78)]
    phrases = split_phrases(notes, 0.35)
    assert [len(p.notes) for p in phrases] == [2, 2]
    assert notes[2].phrase_id == 1
    sa = choose_sa_octave(notes, 6)
    assert sa == 66
    assign_degrees(notes, sa)
    assert [(n.degree, n.octave) for n in notes] == [(0, 0), (2, 0), (7, -1), (0, 1)]


def test_empty_track():
    tr = make_track([(0.5, None)])
    assert segment_notes(tr, 0.0, cfg) == []


def test_drifting_note_is_not_a_glide():
    # a 140 ms Pa whose pitch sags by ~100 cents between Dha and Ga must stay a note (expressive intonation)
    tr = make_track([(0.3, 69), (0.14, 67, {"glide_to": 66}), (0.3, 64)])
    notes = segment_notes(tr, 0.0, cfg)
    assert midis(notes) == [69, 67, 64]
    assert all(n.kind == "note" for n in notes)
