"""Command line interface: sargam transcribe / audition / compare."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from .config import Config

app = typer.Typer(help="Transcribe songs into bansuri sargam notation.", no_args_is_help=True)
console = Console()


def _cfg(**overrides) -> Config:
    from dataclasses import replace

    return replace(Config(), **{k: v for k, v in overrides.items() if v is not None})


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., exists=True, help="Song file (mp3/wav/m4a/flac...)"),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory"),
    sa: str | None = typer.Option(None, "--sa", help="Override the tonic, e.g. F# or 370 (Hz)"),
    flutes: str | None = typer.Option(None, "--flutes", help="Flutes you own, e.g. 'C,G bass,F#'"),
    detail: int = typer.Option(1, "--detail", min=0, max=2, help="0 plain notes, 1 ornaments, 2 also mark low-confidence notes"),
    no_beats: bool = typer.Option(False, "--no-beats", help="Don't use beat tracking for sustain marks"),
    fold_octaves: bool = typer.Option(False, "--fold-octaves", help="Shift out-of-range notes by octaves into the bansuri range"),
    no_separate: bool = typer.Option(False, "--no-separate", help="Skip Demucs (input is already an isolated vocal)"),
    model: str = typer.Option("htdemucs", "--model", help="Demucs model: htdemucs or htdemucs_ft"),
    backend: str = typer.Option("pyin", "--pitch", help="Pitch tracker backend: pyin or crepe"),
    times: bool = typer.Option(False, "--times", help="Prefix each phrase with its start time"),
    diag: bool = typer.Option(False, "--diag", help="Print tonic candidate table and degree histogram"),
    no_snap: bool = typer.Option(False, "--no-snap", help="Keep short off-scale notes instead of snapping them to the scale"),
):
    """Transcribe a song into sargam notation (text), timed notes (JSON) and a MIDI melody."""
    from .flute import parse_flutes
    from .pipeline import run
    from .render import diagnostics, to_text, write_outputs

    cfg = _cfg(pitch_backend=backend, snap_rare=not no_snap)
    owned = parse_flutes(flutes)
    with console.status("working...") as status:
        res = run(audio, cfg, sa, owned, not no_beats, fold_octaves, model, no_separate, progress=lambda m: status.update(m))
    paths = write_outputs(res, out, audio.stem, detail)
    console.print(to_text(res, detail, with_times=times))
    if diag or res.tonic.label == "low":
        console.print("[dim]" + diagnostics(res) + "[/dim]")
    console.print(f"[green]wrote[/green] {paths['text']}  {paths['json']}  {paths['midi']}")


@app.command()
def audition(
    audio: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    sa: str | None = typer.Option(None, "--sa"),
    no_separate: bool = typer.Option(False, "--no-separate"),
    backend: str = typer.Option("pyin", "--pitch"),
):
    """Write a wav with the isolated vocal in the left channel and the transcription as a sine melody in the right."""
    import numpy as np

    from .audio import load_mono, write_wav
    from .pipeline import run
    from .separate import separate
    from .synth import render_note_events

    cfg = _cfg(pitch_backend=backend)
    with console.status("working...") as status:
        res = run(audio, cfg, sa, None, False, False, no_separate=no_separate, progress=lambda m: status.update(m))
        vocal = load_mono(audio, cfg.sr) if no_separate else separate(audio, cfg.sr).vocals
    synth = render_note_events(res.notes, cfg.sr, res.duration_s)
    n = min(len(vocal), len(synth))
    stereo = np.stack([vocal[:n], synth[:n]])
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{audio.stem}.audition.wav"
    write_wav(p, stereo, cfg.sr)
    console.print(f"[green]wrote[/green] {p}  (L = vocal stem, R = transcription)")


@app.command()
def compare(
    notes_json: Path = typer.Argument(..., exists=True, help="*.notes.json from transcribe"),
    truth: Path = typer.Argument(..., exists=True, help="Text file with reference sargam (notationsworld paste)"),
):
    """Score a transcription against a reference notation."""
    from .metrics import compare_files

    rep = compare_files(notes_json, truth)
    console.print(rep.table())


@app.command()
def stems(audio: Path = typer.Argument(..., exists=True), model: str = typer.Option("htdemucs", "--model")):
    """Run vocal separation only (cached under data/cache)."""
    from .audio import cache_dir
    from .separate import separate

    with console.status("separating..."):
        s = separate(audio, Config().sr, model=model)
    console.print(f"[green]stems[/green] in {cache_dir(audio) / 'stems'} via {s.backend}")


if __name__ == "__main__":
    app()
