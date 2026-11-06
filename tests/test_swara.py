import pytest

from sargam.model import NoteEvent, Phrase
from sargam.swara import (
    DEGREE_SYMBOLS, ParseError, Token, collapse_tokens, parse_line, parse_notation,
    parse_word, rel_to_symbol, render_notation, render_phrase, swara_symbol, tokens_to_symbols,
)


def test_symbols_all_degrees_octaves():
    assert [swara_symbol(d) for d in range(12)] == DEGREE_SYMBOLS
    assert swara_symbol(7, -1) == ".P"
    assert swara_symbol(0, 1) == "S'"
    assert swara_symbol(0, 2) == "S''"
    assert swara_symbol(9, -2) == "..D"
    assert rel_to_symbol(-5) == ".P"
    assert rel_to_symbol(12) == "S'"
    assert rel_to_symbol(-12) == ".S"
    assert rel_to_symbol(23) == "N'"


@pytest.mark.parametrize(
    "word",
    ["S", ".P", "S'", "S''", "..N", "{RSn}S", "(R)S", "S~G", "{RSn}(R)S~P'", "S-", "S---", "_", "S~.N"],
)
def test_round_trip_word(word):
    toks = parse_word(word)
    assert tokens_to_symbols(toks).replace(" ", "") == word


def test_glued_atoms_in_ornament():
    toks = parse_word("{R.nS'}S")
    assert toks[0].ornament == [(2, 0), (10, -1), (0, 1)]
    assert toks[0].pitch == (0, 0)


def test_comma_lower_octave():
    assert parse_word(",P")[0].pitch == (7, -1)


@pytest.mark.parametrize("bad", ["X", "S'.", "{RS", "(R", "Re", "hello", "S~"])
def test_parse_errors(bad):
    with pytest.raises(ParseError):
        parse_word(bad)


def test_parse_line_lyrics_rejected():
    assert parse_line("there was a farmer had a dog") is None
    assert parse_line("Sa Re Ga Ma") is None
    toks = parse_line("S R G m P - D N S'")
    assert [t.kind for t in toks].count("sustain") == 1
    assert len(toks) == 9


def test_parse_notation_header_and_lines():
    text = """Some Song (Film)
SCALE OF THE FLUTE IS F# BASS/MIDDLE

la la la la
S R G m P D N S'
tum hi ho
.P .D .N S R G
"""
    sa, phrases = parse_notation(text)
    assert sa == "F#"
    assert len(phrases) == 2
    assert [t.pitch for t in phrases[1]] == [(7, -1), (9, -1), (11, -1), (0, 0), (2, 0), (4, 0)]


def test_parse_notation_flat_alias_header():
    sa, _ = parse_notation("Scale: Gb\nS R G\n")
    assert sa == "F#"


def test_collapse_tokens():
    toks = parse_line("S S - R {GR}S (R)S~G _ G")
    assert collapse_tokens(toks) == ["S", "R", "S", "G"]


def _n(start, end, midi, **kw):
    return NoteEvent(start=start, end=end, midi=midi, **kw)


def test_render_phrase_basic_and_detail():
    sa = 66  # F#
    notes = [
        _n(0.0, 0.5, 66),
        _n(0.5, 1.0, 68, grace_before=66),
        _n(1.0, 1.5, 70, ornament_before=[68, 66, 64]),
        _n(1.5, 2.0, 71, kind="glide", glide_to_midi=73),
        _n(2.0, 2.2, 61),
        _n(2.2, 2.4, 78),
    ]
    assert render_phrase(notes, sa) == "S (S)R {RS.n}G m~P .P S'"
    # plain rendering expands graces/ornaments into ordinary notes (reference style) and drops glide marks
    assert render_phrase(notes, sa, detail=0) == "S S R R S .n G m .P S'"


def test_render_sustain_from_beats_and_median():
    sa = 60
    notes = [_n(0.0, 0.5, 60), _n(0.5, 2.5, 62), _n(2.5, 3.0, 64)]
    assert render_phrase(notes, sa, beat_period=0.5) == "S R - - - G"
    assert render_phrase(notes, sa, beat_period=None, median_dur=0.5) == "S R - - - G"
    assert render_phrase(notes, sa) == "S R G"


def test_render_notation_blocks_and_header():
    sa = 60
    phrases = [Phrase(i, i + 1, [_n(i, i + 0.5, 60), _n(i + 0.5, i + 1, 62)]) for i in range(5)]
    out = render_notation(phrases, sa, phrases_per_block=2, header_lines=["SCALE OF THE FLUTE IS C"])
    lines = out.splitlines()
    assert lines[0] == "SCALE OF THE FLUTE IS C"
    assert lines[1] == ""
    assert lines[2] == "S R" and lines[3] == "S R" and lines[4] == "" and lines[5] == "S R"
    sa_name, parsed = parse_notation(out)
    assert sa_name == "C" and len(parsed) == 5


def test_dot_duration_dialect():
    toks = parse_line("G..m.. R..G..R.S.R.. .P.S\u2019.")
    syms = [str(t) for t in toks]
    assert syms == ["G", "-", "m", "-", "R", "-", "G", "-", "R", "S", "R", "-", ".P", "S'"]
    assert parse_line("S. R. G.") is not None  # short dialect line still parses
    assert parse_line("a line of words with dots... here.") is None


def test_notationsworld_dialect():
    from sargam.swara import nw_symbol, parse_nw_line

    toks, used_m = parse_nw_line("G.G.R.S.n. n.S.R.G.")
    notes = [t for t in toks if t.kind == "note"]
    assert [(t.degree, t.octave) for t in notes[3:6]] == [(0, 0), (11, -1), (11, -1)]  # n = lower Ni, not komal
    assert not used_m
    toks, _ = parse_nw_line("G…P..R.R…D(k).S’..")
    notes = [t for t in toks if t.kind == "note"]
    assert [(t.degree, t.octave) for t in notes] == [(4, 0), (7, 0), (2, 0), (2, 0), (8, 0), (0, 1)]
    assert sum(t.kind == "sustain" for t in toks) == 2 + 1 + 2 + 0 + 1  # '…' = three dots
    assert parse_nw_line("Na hai ye panaa") is None
    assert parse_nw_line("Hmm…. hmm…") is None
    for d in range(12):
        for o in (-1, 0, 1):
            sym = nw_symbol(d, o)
            got = [t for t in parse_nw_line(f"{sym}. {sym}.")[0] if t.kind == "note"][0]
            if d == 5 and o == -1:
                continue  # lower shuddh Ma has no site symbol; rendered as '.m'
            assert (got.degree, got.octave) == (d, o), sym


def test_notationsworld_ma_convention():
    page = "SCALE OF THE SONG IS F# MAJOR\nlyrics here\nG..M..P.. G.M.R.\n"
    sa, phrases = parse_notation(page, "nw")
    assert sa == "F#"
    assert {t.degree for t in phrases[0] if t.kind == "note"} == {4, 5, 7, 2}  # no 'm' on the page: M is shuddh
    sa, phrases = parse_notation(page + "G.m.M.P.\n", "nw")
    assert [t.degree for t in phrases[1] if t.kind == "note"] == [4, 5, 6, 7]
