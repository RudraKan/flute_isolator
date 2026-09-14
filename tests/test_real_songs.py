"""L2: real songs with a reference notation under data/songs/<slug>/{audio.*, truth.txt, meta.json}.
Run with:  uv run pytest -m slow -s tests/test_real_songs.py
"""
import json
from pathlib import Path

import pytest

from tests.conftest import load_meta, song_dirs
from sargam.config import Config
from sargam.metrics import compare_tokens, tokens_from_json
from sargam.pipeline import run
from sargam.render import to_json, write_outputs
from sargam.swara import Token, parse_notation

SONGS = song_dirs()
pytestmark = pytest.mark.slow

RESULTS: list[str] = []


@pytest.mark.skipif(not SONGS, reason="no songs with truth.txt under data/songs")
@pytest.mark.parametrize("song", SONGS, ids=[d.name for d in SONGS])
def test_song(song: Path):
    meta = load_meta(song)
    audio = next(p for p in song.glob("audio.*"))
    res = run(audio, Config(), no_separate=bool(meta.get("no_separate", False)))
    write_outputs(res, song / "out", "audio")
    data = to_json(res)
    truth_sa, truth_phrases = parse_notation((song / "truth.txt").read_text())
    truth_sa = meta.get("sa", truth_sa)
    pred_tokens = tokens_from_json(data)
    truth_tokens = [t for p in truth_phrases for t in p]
    rep = compare_tokens(pred_tokens, truth_tokens, data["tonic"]["name"], truth_sa)
    line = (
        f"{song.name:>16} tonic {rep.tonic_pred}/{rep.tonic_truth} {'OK' if rep.tonic_ok else 'MISS'} "
        f"margin {res.tonic.margin:.3f} {res.tonic.label:<6} SER {rep.ser:.2f} F1 {rep.f1:.2f} scale-jac {rep.scale_jaccard:.2f} "
        f"scale {res.scale.name}"
    )
    RESULTS.append(line)
    print("\n" + line)
    (song / "out" / "report.txt").write_text(rep.table() + "\n")
    assert rep.tonic_ok, rep.table()
