"""Bansuri model and flute recommendation."""
from __future__ import annotations

import re

from .config import Config
from .model import Flute, FluteRecommendation, NoteEvent, hz_to_midi, parse_pitch_name, pitch_name
from .swara import rel_to_symbol

# bare letter = "medium" flute (Sa in octave 4); "X bass" = octave 3
DEFAULT_FLUTES: dict[str, int] = {**{f"{pitch_name(pc)} bass": 48 + pc for pc in range(12)}, **{pitch_name(pc): 60 + pc for pc in range(12)}}


def parse_flutes(spec: str | None) -> list[Flute]:
    if not spec:
        return []
    out: list[Flute] = []
    for raw in re.split(r"[,;]", spec):
        s = raw.strip()
        if not s:
            continue
        if "=" in s:
            name, hz = s.split("=", 1)
            out.append(Flute(name.strip(), int(round(hz_to_midi(float(hz))))))
            continue
        m = re.match(r"^([A-Ga-g][#b♯♭]?)\s*(bass|medium|middle)?$", s)
        if not m:
            raise ValueError(f"cannot parse flute {s!r} (use e.g. 'F#', 'G bass', 'E=330')")
        pc = parse_pitch_name(m.group(1))
        reg = (m.group(2) or "").lower()
        octave = 3 if reg == "bass" else 4
        label = f"{pitch_name(pc)}{' bass' if reg == 'bass' else ''}"
        out.append(Flute(label, 12 * (octave + 1) + pc))
    return out


def _mass(notes: list[NoteEvent], pred) -> float:
    tot = sum(n.duration for n in notes) or 1.0
    return sum(n.duration for n in notes if pred(n)) / tot


def best_sa_octave(notes: list[NoteEvent], tonic_pc: int, cfg: Config) -> tuple[int, float, float, float]:
    """(sa_midi, below, above, high) maximising in-range duration mass."""
    best = None
    for k in range(3, 7):
        sa = tonic_pc + 12 * k
        below = _mass(notes, lambda n: n.midi - sa < cfg.flute_low)
        above = _mass(notes, lambda n: n.midi - sa > cfg.flute_high)
        high = _mass(notes, lambda n: cfg.flute_comfort_high < n.midi - sa <= cfg.flute_high)
        score = 1 - below - above - 0.5 * high
        if best is None or score > best[0]:
            best = (score, sa, below, above, high)
    _, sa, below, above, high = best
    return sa, below, above, high


def recommend_flute(notes: list[NoteEvent], tonic_pc: int, owned: list[Flute] | None, cfg: Config, sa_midi: int | None = None) -> FluteRecommendation:
    if not notes:
        return FluteRecommendation(tonic_pc, tonic_pc + 60, (0, 0), "-", True, 0.0, [], None, 0, "no notes")
    if sa_midi is None:
        sa_midi, below, above, high = best_sa_octave(notes, tonic_pc, cfg)
    else:
        below = _mass(notes, lambda n: n.midi - sa_midi < cfg.flute_low)
        above = _mass(notes, lambda n: n.midi - sa_midi > cfg.flute_high)
    rels = [n.midi - sa_midi for n in notes]
    lo, hi = min(rels), max(rels)
    unplayable = below + above
    fits = unplayable <= cfg.unplayable_tol
    bad_phrases = sorted({n.phrase_id for n in notes if not (cfg.flute_low <= n.midi - sa_midi <= cfg.flute_high)})
    rng = f"{rel_to_symbol(lo)} .. {rel_to_symbol(hi)}"

    recommended, transposition = None, 0
    owned = owned or []
    match = [f for f in owned if f.sa_midi % 12 == tonic_pc]
    if match:
        recommended = min(match, key=lambda f: abs(f.sa_midi - sa_midi))
    elif owned:
        def cdist(f: Flute) -> int:
            d = (f.sa_midi - tonic_pc) % 12
            return d if d <= 6 else d - 12
        recommended = min(owned, key=lambda f: (abs(cdist(f)), abs(f.sa_midi - sa_midi)))
        transposition = cdist(recommended)

    parts = [f"Sa = {pitch_name(tonic_pc)}: use a {pitch_name(tonic_pc)} flute. Melody spans {rng}."]
    if recommended is not None:
        if transposition == 0:
            parts.append(f"Your {recommended.name} flute plays it in the original key.")
        else:
            direction = "higher" if transposition > 0 else "lower"
            parts.append(
                f"No owned flute in {pitch_name(tonic_pc)}; nearest is your {recommended.name} "
                f"({abs(transposition)} semitone{'s' if abs(transposition) != 1 else ''} {direction} than the recording). Sargam is unchanged."
            )
    if not fits:
        pct = round(100 * unplayable)
        parts.append(
            f"{pct}% of the melody falls outside the bansuri range ({rel_to_symbol(cfg.flute_low)} .. {rel_to_symbol(cfg.flute_high)}); "
            f"phrases {', '.join(str(p + 1) for p in bad_phrases)} need an octave shift (see --fold-octaves)."
        )
    return FluteRecommendation(
        sa_pc=tonic_pc, sa_midi=sa_midi, rel_range=(lo, hi), rel_range_str=rng, fits=fits, unplayable_fraction=unplayable,
        out_of_range_phrases=bad_phrases, recommended=recommended, transposition_semitones=transposition, advice=" ".join(parts),
    )


def fold_octaves(notes: list[NoteEvent], sa_midi: int, cfg: Config) -> list[NoteEvent]:
    """Shift out-of-range notes by whole octaves into the playable range (flag 'folded')."""
    for n in notes:
        rel = n.midi - sa_midi
        while rel > cfg.flute_high:
            rel -= 12
            n.flags.add("folded")
        while rel < cfg.flute_low:
            rel += 12
            n.flags.add("folded")
        n.midi = sa_midi + rel
        if n.glide_to_midi is not None and "folded" in n.flags:
            n.glide_to_midi += (n.midi - (sa_midi + (n.glide_to_midi - sa_midi)))  # keep interval
    return notes
