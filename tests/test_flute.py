import pytest

from sargam.config import Config
from sargam.flute import DEFAULT_FLUTES, fold_octaves, parse_flutes, recommend_flute
from sargam.model import NoteEvent

cfg = Config()


def notes_from_rels(rels, sa=66):
    out = []
    for i, r in enumerate(rels):
        out.append(NoteEvent(start=i * 0.5, end=(i + 1) * 0.5, midi=sa + r, phrase_id=i // 4))
    return out


def test_parse_flutes():
    fl = parse_flutes("C, G bass, F#, E=330, Bb")
    assert [(f.name, f.sa_midi) for f in fl] == [("C", 60), ("G bass", 55), ("F#", 66), ("E", 64), ("A#", 70)]
    assert DEFAULT_FLUTES["F#"] == 66 and DEFAULT_FLUTES["F# bass"] == 54
    with pytest.raises(ValueError):
        parse_flutes("H")


def test_recommend_match_and_range():
    notes = notes_from_rels([-5, -3, 0, 2, 4, 7, 12, 19])
    rec = recommend_flute(notes, 6, parse_flutes("C,F#,G"), cfg)
    assert rec.fits and rec.recommended.name == "F#" and rec.transposition_semitones == 0
    assert rec.rel_range_str == ".P .. P'"
    assert rec.sa_midi == 66


def test_recommend_nearest_when_no_match():
    notes = notes_from_rels([0, 2, 4, 7])
    rec = recommend_flute(notes, 6, parse_flutes("C,G"), cfg)
    assert rec.recommended.name == "G" and rec.transposition_semitones == 1
    assert "1 semitone higher" in rec.advice


def test_out_of_range_reported_and_folded():
    notes = notes_from_rels([0, 2, 4, 7, 26, 27, 0, -8])
    rec = recommend_flute(notes, 6, None, cfg, sa_midi=66)
    assert not rec.fits and rec.out_of_range_phrases == [1]
    fold_octaves(notes, 66, cfg)
    rels = [n.midi - 66 for n in notes]
    assert rels == [0, 2, 4, 7, 14, 15, 0, 4]
    assert "folded" in notes[4].flags


def test_sa_octave_chosen_for_register():
    # female singer: melody around F#4..F#5 -> sa_midi 66; male an octave lower -> 54
    hi = recommend_flute(notes_from_rels([0, 4, 7, 12], sa=66), 6, None, cfg)
    lo = recommend_flute(notes_from_rels([0, 4, 7, 12], sa=54), 6, None, cfg)
    assert hi.sa_midi == 66 and lo.sa_midi == 54
