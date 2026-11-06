import json

import numpy as np
import pytest

from sargam.ml.align import IGNORE, align
from sargam.ml.decode import DecodeParams, decode, symbols
from sargam.ml.features import N_CLASSES, class_to_rel, rel_to_class
from sargam.ml.synth_data import load_melodies, make_window


def _logits(path_classes: list[tuple[int, int]], onsets_at=()) -> tuple[np.ndarray, np.ndarray]:
    """Confident logits for runs of (class, n_frames); onset logits high at the given frames."""
    T = sum(n for _, n in path_classes)
    cl = np.full((T, N_CLASSES), -6.0, np.float32)
    t = 0
    for c, n in path_classes:
        cl[t : t + n, c] = 6.0
        t += n
    on = np.full(T, -6.0, np.float32)
    on[list(onsets_at)] = 6.0
    return cl, on


def test_decode_runs_split_and_short_fragments():
    S, R, G = rel_to_class(0), rel_to_class(2), rel_to_class(4)
    cl, on = _logits([(0, 10), (S, 30), (R, 2), (G, 40), (0, 10), (G, 40)], onsets_at=[10, 42, 92, 112])
    dec = decode(cl, on, DecodeParams(min_note_frames=4))
    assert [d.rel for d in dec] == [0, 4, 4, 4]  # 2-frame R glued to S; G re-struck by onset at 112 splits
    assert dec[0].start_f == 10 and dec[1].start_f == 42
    assert symbols(dec) == ["S", "G"]


def test_align_places_written_notes():
    S, R, G = rel_to_class(0), rel_to_class(2), rel_to_class(4)
    cl, _ = _logits([(0, 20), (S, 50), (R, 50), (0, 15), (R, 30), (G, 60), (0, 20)])
    lab, on = align(cl, [0, 2, 2, 4])
    starts = list(np.where(on > 0)[0])
    assert starts == [20, 70, 135, 165]
    assert (lab[20:70] == S).all() and (lab[:20] == 0).all() and not (lab == IGNORE).any()


def test_align_skips_a_note_the_singer_did_not_sing():
    S, G = rel_to_class(0), rel_to_class(4)
    cl, _ = _logits([(0, 10), (S, 40), (G, 40), (0, 10)])
    lab, on = align(cl, [0, 2, 4])  # R written but not sung
    assert int(on.sum()) == 2 and (lab[50:90] == G).all()


@pytest.fixture
def tiny_corpus(tmp_path):
    rec = {"url": "u", "slug": "tiny", "kind": "sargam", "title": "t", "tonic": "C", "tonic_source": "header", "mode": None,
           "lines": [{"lyric": None, "raw": "", "notes": [[4, 2, 1], [5, 2, 0], [2, 2, 1], [4, 1, 0], [2, 1, 0], [0, 1, 0], [2, 2, 0]]},
                     {"lyric": None, "raw": "", "notes": [[7, 2, 1], [9, 2, 0], [11, 1, 0], [12, 3, 0], [-1, 2, 1], [0, 2, 0]]}]}
    p = tmp_path / "songs.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    return p


def test_synthetic_windows_are_valid(tiny_corpus):
    mel = load_melodies(tiny_corpus, val_frac=0.0)
    for seed in range(200):
        w = make_window(mel, np.random.default_rng(seed), 512)
        assert w.rel_cents.shape == w.energy.shape == w.cls.shape == w.onset.shape == (512,)
        assert 0 <= w.cls.min() and w.cls.max() < N_CLASSES
        assert np.isfinite(w.energy).all() and 0 <= w.energy.min() and w.energy.max() <= 1
        labelled = {int(class_to_rel(c)) for c in np.unique(w.cls) if c}
        assert labelled <= {4, 5, 2, 0, 7, 9, 11, 12, -1}  # only written notes are ever labels


def test_model_and_windowed_inference_cover_long_input():
    torch = pytest.importorskip("torch")
    from sargam.ml.infer import predict_logits
    from sargam.ml.model import NoteNet

    model = NoteNet(channels=4, d_model=32, layers=1, heads=2).eval()
    T = 2500  # several overlapping windows
    rel = np.where(np.arange(T) % 50 < 40, 400.0, np.nan).astype(np.float32)
    cl, on = predict_logits(model, rel, np.full(T, 0.8, np.float32), "cpu")
    assert cl.shape == (T, N_CLASSES) and on.shape == (T,)
    assert np.isfinite(cl).all() and not (cl == 0).all(axis=1).any()  # every frame written by some window
    with torch.no_grad():
        c, o = model(torch.from_numpy(rel[None, :1024]), torch.full((1, 1024), 0.8))
    assert c.shape == (1, 1024, N_CLASSES) and o.shape == (1, 1024)


def test_corpus_page_parsing():
    from sargam.corpus import estimate_key, line_text, parse_piano_page, parse_sargam_page

    html = """<h1>Song – Sargam And Flute Notes</h1><div class="entry-content x">
    <p>SCALE OF THE SONG IS F# MAJOR</p><p>Tum se hi</p><p>G…P..R.R…G..P..P.D…</p>
    <p>Dil charkhe ki</p><p>S..S..R..S..n…d..</p><p>Also Read This</p><p>S.R.G.</p></div>"""
    rec = parse_sargam_page("https://x/song-sargam-notes.html", html)
    assert rec["tonic"] == "F#" and rec["mode"] == "major" and len(rec["lines"]) == 2
    assert rec["lines"][0]["lyric"] == "Tum se hi"
    assert rec["lines"][1]["notes"][-2:] == [[-1, 3, 0], [-3, 2, 0]]  # n = lower Ni, d = lower Dha
    assert line_text(rec["lines"][1]["notes"]) == "S..S..R..S..n...d.."
    piano = """<div class="entry-content">""" + "".join(f"<p>{l}</p>" for l in ["C4 D4 E4 F4 G4", "G4 A4 B4 C5", "E4 D4 C4 C4"]) + "</div>"
    prec = parse_piano_page("https://x/a-piano-notes.html", piano)
    assert prec["tonic"] == "C" and prec["lines"][0]["notes"][0][0] == 0
    assert estimate_key([57, 59, 60, 62, 64, 65, 67, 69, 69, 57, 64, 57]) == (9, "minor")


def test_lookup_finds_song_regardless_of_sa(tmp_path):
    from sargam.lookup import suggest

    def rec(slug, rels):
        return {"url": f"https://x/{slug}.html", "slug": slug, "kind": "sargam", "title": slug, "tonic": "C",
                "tonic_source": "header", "mode": None, "lines": [{"lyric": None, "raw": "", "notes": [[r, 1, 0] for r in rels]}]}

    target = [0, 2, 4, 5, 4, 2, 0, 7, 9, 7, 5, 4, 2, 4, 0, -1, 0, 2, 7, 4]
    others = [[0, 4, 7, 12, 7, 4, 0, 4, 7, 12], [12, 11, 9, 7, 5, 4, 2, 0, 2, 4], [0, 0, 7, 7, 9, 9, 7, 5, 5, 4, 4, 2, 2, 0]]
    p = tmp_path / "songs.jsonl"
    p.write_text("".join(json.dumps(rec(f"s{i}", r)) + "\n" for i, r in enumerate([*others, target])))
    noisy = [x + 2 for x in target[:8] + [3] + target[8:]]  # transposed, one wrong note, repeats
    noisy = [n for x in noisy for n in (x, x)]
    top = suggest(noisy, k=2, corpus_path=p)
    assert top[0]["title"] == "s3" and top[0]["margin"] > 1.0
