"""L1: synthesized audio -> pyin -> full analysis. Exact recovery expected."""
import numpy as np
import pytest

from sargam.config import Config
from sargam.model import PITCH_NAMES, midi_to_hz
from sargam.pipeline import analyze
from sargam.swara import DEGREE_SYMBOLS, collapse_tokens, parse_notation, note_to_token
from sargam.synth import synthesize
from sargam.tonic import PENTATONIC_TEMPLATES, SCALE_TEMPLATES

cfg = Config()
SCALES = {**SCALE_TEMPLATES, **PENTATONIC_TEMPLATES}


def scale_phrase(name: str) -> str:
    degs = sorted(SCALES[name])
    sym = [DEGREE_SYMBOLS[d] for d in degs]
    asc = " ".join(sym + ["S'"])
    desc = " ".join(["S'"] + sym[::-1])
    return f"{asc} -\n{desc} - -\n{sym[-1]} {sym[-2]} {sym[1]} S - - -\n"


def run_synth(text, sa_pc=6, octave=4, **kw):
    sa_hz = midi_to_hz(12 * (octave + 1) + sa_pc) * 2 ** (kw.pop("detune_cents", 0) / 1200)
    r = synthesize(text, sa_hz=sa_hz, bpm=150, **kw)
    res = analyze(r.vocal, r.mix, None, r.sr, cfg, use_beats=False)
    return r, res


def got_symbols(res):
    return collapse_tokens([note_to_token(n, res.tonic.sa_midi, 0) for n in res.notes])


def truth_symbols(text):
    _, phrases = parse_notation(text)
    return collapse_tokens([t for p in phrases for t in p])


@pytest.mark.parametrize("scale", ["Bilawal", "Kafi", "Bhairavi", "Kalyan", "Bhupali", "Asavari", "Khamaj", "Malkauns"])
def test_scale_recovery(scale):
    text = scale_phrase(scale)
    r, res = run_synth(text, sa_pc=6)
    assert res.tonic.pc == 6, (res.tonic.name, res.tonic.label)
    assert res.scale.name == scale
    assert got_symbols(res) == truth_symbols(text)


@pytest.mark.parametrize("sa_pc", range(12))
def test_all_tonics(sa_pc):
    text = scale_phrase("Kafi")
    r, res = run_synth(text, sa_pc=sa_pc, octave=4 if sa_pc < 6 else 3)
    assert res.tonic.pc == sa_pc, res.tonic.name
    assert res.tonic.sa_midi == r.sa_midi
    assert got_symbols(res) == truth_symbols(text)


@pytest.mark.parametrize("detune", [-30, 40])
def test_detuned_sa(detune):
    text = scale_phrase("Bilawal")
    r, res = run_synth(text, sa_pc=2, detune_cents=detune)
    assert abs(res.tuning.offset_cents - detune) < 10
    assert res.tonic.pc == 2
    assert abs(1200 * np.log2(res.tonic.sa_hz / r.sa_hz)) < 12
    assert got_symbols(res) == truth_symbols(text)


def test_relative_minor_trap_with_pad():
    # Asavari melody over a pad of the relative-major triad (g D n = degrees 3, 8, 10 -> the major chord on komal Ga)
    text = "S R g m P d n S' -\nS' n d P m g R S - -\nd P m g R S - - -\n"
    r, res = run_synth(text, sa_pc=9, pad=[(3, 7, 10), (3, 7, 10), (8, 0, 3)], pad_db=-12)
    assert res.tonic.pc == 9, (res.tonic.name, res.tonic.ambiguity)


def test_fifth_trap_pa_held_longer():
    text = "S R G m P - - -\nP - - D P m G S -\nP - - - m G R S -\n"
    r, res = run_synth(text, sa_pc=4)
    assert res.tonic.pc == 4, (res.tonic.name, res.tonic.ambiguity)


def test_vibrato_no_extra_notes():
    text = "S R G m P D N S' -\nS' N D P m G R S - -\n"
    r, res = run_synth(text, sa_pc=6, vibrato_cents=60, vibrato_hz=6)
    assert got_symbols(res) == truth_symbols(text)
    assert len(res.notes) == len(r.truth)


def test_octaves_and_flute_fit():
    text = ".P .D .N S R G m P D N S' R' G' m' P' -\nP' m' G' R' S' N D P m G R S - -\n"
    r, res = run_synth(text, sa_pc=0)
    assert res.tonic.pc == 0 and res.tonic.sa_midi == r.sa_midi
    assert got_symbols(res) == truth_symbols(text)
    assert res.flute.fits and res.flute.rel_range_str == ".P .. P'"


def test_out_of_range_flagged():
    text = "S R G P D S' R' G' P' D' S'' R'' G'' - -\nS - - -\n"
    r, res = run_synth(text, sa_pc=0)
    assert res.tonic.pc == 0
    assert not res.flute.fits


def test_phrases_split_at_gaps():
    text = "S R G -\nP D N -\nS' - - -\n"
    r, res = run_synth(text, sa_pc=6)
    assert len(res.phrases) == 3
