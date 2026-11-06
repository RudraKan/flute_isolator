"""Find which notation-corpus song a transcription is: TF-IDF over interval n-grams.

Intervals (not Sa-relative notes) make the match independent of the Sa a page chose; collapsing repeats makes it
independent of rhythm. Used by the web UI to suggest a reference page, which turns a song into a training pair.
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache

N = 4  # intervals per gram (five notes)


def _grams(rels: list[int]) -> Counter:
    seq: list[int] = []
    for r in rels:
        if not seq or seq[-1] != r:
            seq.append(r)
    iv = [b - a for a, b in zip(seq, seq[1:])]
    return Counter(tuple(iv[i : i + N]) for i in range(len(iv) - N + 1))


@lru_cache(maxsize=1)
def _index(path: str, mtime: float):
    from pathlib import Path

    from .corpus import load_songs

    songs = load_songs(Path(path), kinds=("sargam", "piano"))
    entries = []
    for s in songs:
        g = _grams([n[0] for ln in s["lines"] for n in ln["notes"]])
        if g:
            entries.append((s, g, math.sqrt(sum(g.values()))))
    df = Counter(k for _, g, _ in entries for k in g)
    return entries, df


def suggest(rels: list[int], k: int = 3, corpus_path=None) -> list[dict]:
    """Top-k corpus songs for a note sequence (semitones above Sa, in time order)."""
    from .corpus import corpus_dir

    path = corpus_path or corpus_dir() / "songs.jsonl"
    if not path.exists():
        return []
    entries, df = _index(str(path), path.stat().st_mtime)
    q = _grams(rels)
    D = len(entries)
    scored = []
    for s, g, norm in entries:
        sc = sum(min(c, g[key]) * math.log(D / (1 + df[key])) for key, c in q.items() if key in g)
        if sc > 0:
            scored.append((sc / (norm + 1), s))
    scored.sort(key=lambda x: -x[0])
    top = scored[:k]
    runner_up = scored[k][0] if len(scored) > k else 0.0
    out = []
    for i, (sc, s) in enumerate(top):
        nxt = top[i + 1][0] if i + 1 < len(top) else runner_up
        margin = round(sc / nxt, 2) if nxt > 0 else 99.0  # 99 = nothing else matches at all
        out.append({"title": s["title"], "url": s["url"], "kind": s["kind"], "tonic": s["tonic"], "score": round(sc, 2),
                    "margin": margin})
    return out
