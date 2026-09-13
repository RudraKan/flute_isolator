from sargam.metrics import aligned_f1, compare_tokens, levenshtein
from sargam.swara import parse_line


def test_levenshtein():
    assert levenshtein(list("kitten"), list("sitting")) == 3
    assert levenshtein([], ["S"]) == 1


def test_aligned_f1_octave_partial_credit():
    p, r, f = aligned_f1(["S", "R", "G'"], ["S", "R", "G"])
    assert p == r == (2.5 / 3)


def test_compare_tokens_report():
    pred = parse_line("S R G m P - D N S'")
    truth = parse_line("S R G m P D N S'")
    rep = compare_tokens(pred, truth, "F#", "F#")
    assert rep.tonic_ok and rep.ser == 0.0 and rep.f1 == 1.0 and rep.scale_jaccard == 1.0
    rep2 = compare_tokens(parse_line("S R g"), truth, "C#", "F#")
    assert rep2.tonic_ok is False and "fifth" in rep2.tonic_relation
    assert 0 < rep2.ser <= 1
