import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"


def song_dirs() -> list[Path]:
    root = DATA / "songs"
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        audio = [p for p in d.glob("audio.*") if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac", ".ogg")]
        if audio and (d / "truth.txt").exists():
            out.append(d)
    return out


def load_meta(d: Path) -> dict:
    m = d / "meta.json"
    return json.loads(m.read_text()) if m.exists() else {}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs real songs under data/songs")
