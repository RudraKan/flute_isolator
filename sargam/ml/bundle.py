"""Build the zip that goes to PACE: training code, corpus, real-song eval set, cluster scripts.

The zip holds derived data only (parsed notation and pitch features, no audio). It still contains notation scraped
from notationsworld, so keep it in your private PACE storage and do not publish it.
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

from ..audio import DATA_DIR

ROOT = Path(__file__).resolve().parents[2]
CODE_SKIP = {"__pycache__", "static"}


def make_bundle(out_dir: Path) -> Path:
    corpus = DATA_DIR / "corpus" / "songs.jsonl"
    if not corpus.exists():
        raise FileNotFoundError("no corpus yet: run `sargam data crawl` then `sargam data build`")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sargam-pace-{time.strftime('%Y%m%d-%H%M')}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((ROOT / "sargam").rglob("*")):
            if f.is_file() and not (CODE_SKIP & set(f.relative_to(ROOT).parts)) and f.suffix == ".py":
                z.write(f, Path("sargam_pace") / f.relative_to(ROOT))
        for f in sorted((ROOT / "pace").iterdir()):
            if f.is_file():
                z.write(f, Path("sargam_pace") / "pace" / f.name)
        z.write(corpus, "sargam_pace/data/corpus/songs.jsonl")
        for f in sorted((DATA_DIR / "eval").glob("*.npz")):
            z.write(f, Path("sargam_pace/data/eval") / f.name)
        init = DATA_DIR / "models" / "notenet.pt"
        if init.exists():  # lets a cluster run fine-tune from the current local model (--init)
            z.write(init, "sargam_pace/data/models/notenet.pt")
    return path
