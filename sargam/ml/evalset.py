"""Real-song evaluation set: model-ready features + reference notation, packed per (song, pitch backend).

`export` runs the normal pipeline (stems and pitch tracks are cached) to get Sa, then stores the 10 ms features
and the collapsed reference symbols in DATA_DIR/eval/<slug>.<backend>.npz. Those files are small, need no audio
stack to read, and ship in the PACE bundle so training can report real-song accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio import DATA_DIR


def eval_dir() -> Path:
    d = DATA_DIR / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def labelled_songs() -> list[Path]:
    """Song folders with a reference notation and audio: data/songs/<slug>/ and library items from the web UI."""
    out = []
    for root in (DATA_DIR / "songs", DATA_DIR / "library"):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if (d / "truth.txt").exists() and _audio_in(d):
                out.append(d)
    return out


def _audio_in(d: Path) -> Path | None:
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac", ".mp4", ".ogg", ".webm") and not p.name.startswith("."):
            return p
    return None


def export(backends: tuple[str, ...] = ("pyin",), progress=print) -> list[Path]:
    from dataclasses import replace

    from ..config import Config
    from ..metrics import tokens_from_json
    from ..pipeline import run
    from ..render import to_json
    from ..swara import collapse_tokens, parse_notation
    from .features import track_features

    written = []
    for song in labelled_songs():
        audio = _audio_in(song)
        truth_sa, phrases = parse_notation((song / "truth.txt").read_text(), "nw")
        truth = collapse_tokens([t for p in phrases for t in p])
        for backend in backends:
            cfg = replace(Config(), pitch_backend=backend)
            try:
                res = run(audio, cfg, sa_override=truth_sa)
            except Exception as e:  # noqa: BLE001
                progress(f"skip {song.name} ({backend}): {e}")
                continue
            thr = res.cfg.voicing_thr
            rel, energy = track_features(res.track, res.tonic.sa_hz, thr)
            baseline = collapse_tokens(tokens_from_json(to_json(res)))
            out = eval_dir() / f"{song.name}.{backend}.npz"
            np.savez_compressed(
                out, rel_cents=rel, energy=energy, sa_midi=res.tonic.sa_midi, sa_hz=res.tonic.sa_hz,
                truth=np.array(" ".join(truth)), baseline=np.array(" ".join(baseline)), backend=np.array(backend),
            )
            written.append(out)
            progress(f"wrote {out.name}: {len(truth)} truth symbols, Sa {truth_sa} {res.tonic.sa_hz:.1f} Hz")
    return written


def load(path: Path) -> dict:
    z = np.load(path)
    return {
        "name": path.stem, "rel_cents": z["rel_cents"], "energy": z["energy"], "sa_midi": int(z["sa_midi"]),
        "truth": str(z["truth"]).split(), "baseline": str(z["baseline"]).split(),
    }
