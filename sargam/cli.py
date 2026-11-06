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
    detail: int = typer.Option(0, "--detail", min=0, max=2, help="0 plain notes (default), 1 ornament/meend marks, 2 also mark low-confidence notes"),
    style: str = typer.Option("dots", "--style", help="dots: G..m.. R.G. (our symbols, site-style lengths); nw: notationsworld legend (n = lower Ni, D(k) = komal); dashes: G - m - R G"),
    no_beats: bool = typer.Option(False, "--no-beats", help="Don't use beat tracking for sustain marks"),
    fold_octaves: bool = typer.Option(False, "--fold-octaves", help="Shift out-of-range notes by octaves into the bansuri range"),
    no_separate: bool = typer.Option(False, "--no-separate", help="Skip Demucs (input is already an isolated vocal)"),
    model: str = typer.Option("htdemucs", "--model", help="Demucs model: htdemucs or htdemucs_ft"),
    backend: str = typer.Option("pyin", "--pitch", help="Pitch tracker backend: pyin or crepe"),
    times: bool = typer.Option(False, "--times", help="Prefix each phrase with its start time"),
    diag: bool = typer.Option(False, "--diag", help="Print tonic candidate table and degree histogram"),
    no_snap: bool = typer.Option(False, "--no-snap", help="Keep short off-scale notes instead of snapping them to the scale"),
    simplify: float = typer.Option(0.0, "--simplify", help="Merge fragments shorter than this many ms into neighbours (fewer, longer notes)"),
):
    """Transcribe a song into sargam notation (text), timed notes (JSON) and a MIDI melody."""
    from .flute import parse_flutes
    from .pipeline import run
    from .render import diagnostics, to_text, write_outputs

    cfg = _cfg(pitch_backend=backend, snap_rare=not no_snap, simplify_min_ms=simplify)
    owned = parse_flutes(flutes)
    with console.status("working...") as status:
        res = run(audio, cfg, sa, owned, not no_beats, fold_octaves, model, no_separate, progress=lambda m: status.update(m))
    paths = write_outputs(res, out, audio.stem, detail, style)
    console.print(to_text(res, detail, with_times=times, style=style))
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


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    open_browser: bool = typer.Option(False, "--open", help="Open the UI in the default browser"),
    library: Path | None = typer.Option(None, "--library", help="Library folder (default: data/library)"),
):
    """Start the local web UI: upload songs, read and play along with the sargam, compare with a reference."""
    from .web.server import serve as run_server

    run_server(host, port, open_browser, library)


if __name__ == "__main__":
    app()


@app.command()
def truth(
    url: str = typer.Argument(..., help="URL of a notation page (e.g. notationsworld)"),
    slug: str = typer.Argument(..., help="Song folder name under data/songs/"),
    sa: str | None = typer.Option(None, "--sa", help="Override the tonic if the page header is missing/wrong"),
):
    """Fetch a notation page and save only its sargam lines as data/songs/<slug>/truth.txt for evaluation.
    Put the song audio next to it as audio.mp3 (or wav/m4a/flac)."""
    import html as html_mod
    import json
    import re
    import urllib.request

    from .audio import DATA_DIR
    from .swara import parse_notation, parse_nw_line

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (sargam evaluation helper)"})
    raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="ignore")
    txt = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S)
    txt = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>", "\n", txt)
    txt = html_mod.unescape(re.sub(r"<[^>]+>", " ", txt))
    page_sa, phrases = parse_notation(txt, "nw")
    sa = sa or page_sa
    keep = [l.strip() for l in txt.splitlines() if l.strip() and parse_nw_line(l)]
    if not keep:
        console.print("[red]no notation lines found on that page[/red]")
        raise typer.Exit(1)
    d = DATA_DIR / "songs" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "truth.txt").write_text((f"SCALE OF THE FLUTE IS {sa}\n" if sa else "") + "\n".join(keep) + "\n")
    meta = {"sa": sa, "source": url}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    n_notes = sum(1 for p in phrases for t in p if t.kind == "note")
    console.print(f"[green]saved[/green] {d / 'truth.txt'}: Sa={sa}, {len(keep)} lines, {n_notes} notes. Add audio as {d / 'audio.mp3'}")
