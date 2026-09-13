"""Text / JSON / MIDI output and diagnostics."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .model import pitch_name
from .pipeline import Result
from .swara import DEGREE_SYMBOLS, render_notation, swara_symbol
from .tonic import describe_scores


def header_lines(res: Result) -> list[str]:
    t = res.tonic
    lines = [f"SCALE OF THE FLUTE IS {t.name}"]
    mode = f", key {t.name} {t.mode}" if t.mode else ""
    lines.append(f"Sa = {t.name} ({t.sa_hz:.1f} Hz)  scale: {res.scale.name}{mode}")
    conf = f"tonic confidence: {t.label} (margin {t.margin:.3f})"
    if t.overridden:
        conf = "tonic set by --sa; " + conf
    lines.append(conf)
    if res.beats is not None:
        lines.append(f"tempo: {res.beats.bpm:.0f} bpm (one '-' per extra beat)")
    lines.append(res.flute.advice)
    if t.ambiguity and t.label == "low" and not t.overridden:
        order = np.argsort(-t.scores)
        alt = pitch_name(int(order[1]))
        lines.append(f"WARNING: tonic ambiguous ({t.ambiguity}); if it sounds wrong try --sa {alt}")
    for kc in res.key_changes:
        lines.append(f"WARNING: possible key change to {pitch_name(kc.local_tonic_pc)} around {kc.start_s:.0f}-{kc.end_s:.0f}s")
    return lines


def to_text(res: Result, detail: int = 1, with_times: bool = False) -> str:
    beat = res.beats.period if res.beats is not None else None
    body = render_notation(res.phrases, res.tonic.sa_midi, beat, detail, res.cfg.phrases_per_block, header_lines(res))
    if not with_times:
        return body
    out = body.splitlines()
    k = 0
    for i, line in enumerate(out):
        if i < len(header_lines(res)) + 1 or not line.strip():
            continue
        if k < len(res.phrases):
            out[i] = f"[{res.phrases[k].start:6.1f}s] {line}"
            k += 1
    return "\n".join(out) + "\n"


def to_json(res: Result) -> dict:
    t = res.tonic
    notes = []
    for n in res.notes:
        d = asdict(n)
        d["flags"] = sorted(n.flags)
        d["symbol"] = swara_symbol(n.degree, n.octave)
        d["start"], d["end"] = round(n.start, 3), round(n.end, 3)
        d["cents_dev"], d["confidence"] = round(n.cents_dev, 1), round(n.confidence, 3)
        notes.append(d)
    return {
        "source": res.source,
        "duration_s": round(res.duration_s, 2),
        "separation_backend": res.separation_backend,
        "tuning_cents": round(res.tuning.offset_cents, 1),
        "tonic": {
            "name": t.name, "pc": t.pc, "sa_hz": round(t.sa_hz, 2), "sa_midi": t.sa_midi, "confidence": round(t.confidence, 3),
            "margin": round(t.margin, 4), "label": t.label, "ambiguity": t.ambiguity, "overridden": t.overridden, "mode": t.mode,
            "scores": {pitch_name(i): round(float(s), 4) for i, s in enumerate(t.scores)},
            "components": {k: [round(float(x), 4) for x in v] for k, v in t.components.items()},
        },
        "scale": {
            "name": res.scale.name, "distance": res.scale.distance, "degrees_present": res.scale.degrees_present,
            "degree_mass": {DEGREE_SYMBOLS[i]: round(float(m), 4) for i, m in enumerate(res.scale.degree_mass)},
            "variants": res.scale.variants, "mixed": res.scale.mixed,
        },
        "flute": {
            "sa_pc": res.flute.sa_pc, "sa_midi": res.flute.sa_midi, "range": res.flute.rel_range_str, "fits": res.flute.fits,
            "unplayable_fraction": round(res.flute.unplayable_fraction, 3), "recommended": res.flute.recommended.name if res.flute.recommended else None,
            "transposition_semitones": res.flute.transposition_semitones, "advice": res.flute.advice,
        },
        "tempo_bpm": round(res.beats.bpm, 1) if res.beats else None,
        "key_change_warnings": [asdict(k) for k in res.key_changes],
        "phrases": [{"id": i, "start": round(p.start, 3), "end": round(p.end, 3), "n_notes": len(p.notes)} for i, p in enumerate(res.phrases)],
        "notes": notes,
        "log": res.log,
    }


def to_midi(res: Result, path: str | Path, a4_hz: float = 440.0) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=res.beats.bpm if res.beats else 120.0)
    inst = pretty_midi.Instrument(program=73, name="flute")  # GM flute
    # global pitch bend so the MIDI plays at the song's actual tuning
    bend_cents = res.tuning.offset_cents
    for n in res.notes:
        vel = int(60 + 60 * min(1.0, max(0.0, n.confidence)))
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=int(n.midi), start=float(n.start), end=float(max(n.end, n.start + 0.05))))
    if abs(bend_cents) > 1:
        inst.pitch_bends.append(pretty_midi.PitchBend(pitch=int(np.clip(bend_cents / 200.0 * 8192, -8192, 8191)), time=0.0))
    pm.instruments.append(inst)
    pm.write(str(path))


def diagnostics(res: Result) -> str:
    lines = ["tonic candidates:"]
    lines += ["  " + s for s in describe_scores(res.tonic, top=4)]
    lines.append("degree mass relative to Sa:")
    D = res.scale.degree_mass
    for i, m in enumerate(D):
        bar = "#" * int(round(40 * m))
        lines.append(f"  {DEGREE_SYMBOLS[i]:>2} {m:5.2f} {bar}")
    lines.append(f"notes: {len(res.notes)}  phrases: {len(res.phrases)}  voiced: {float(np.isfinite(res.track.f0_hz).mean()):.0%}")
    lines += res.log
    return "\n".join(lines)


def write_outputs(res: Result, out_dir: str | Path, stem: str, detail: int = 1) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "text": out / f"{stem}.sargam.txt",
        "json": out / f"{stem}.notes.json",
        "midi": out / f"{stem}.melody.mid",
    }
    paths["text"].write_text(to_text(res, detail))
    paths["json"].write_text(json.dumps(to_json(res), indent=1))
    to_midi(res, paths["midi"])
    return paths
