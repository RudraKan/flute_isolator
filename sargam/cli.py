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
    backend: str = typer.Option(Config.pitch_backend, "--pitch", help="Pitch tracker: rmvpe (vocal model, downloads 181 MB once), pyin, crepe or fcpe"),
    times: bool = typer.Option(False, "--times", help="Prefix each phrase with its start time"),
    diag: bool = typer.Option(False, "--diag", help="Print tonic candidate table and degree histogram"),
    no_snap: bool = typer.Option(False, "--no-snap", help="Keep short off-scale notes instead of snapping them to the scale"),
    simplify: float | None = typer.Option(None, "--simplify", help="Merge fragments shorter than this many ms into neighbours (fewer, longer notes); default per pitch backend (rmvpe 100, pyin 0)"),
    notes: str = typer.Option("rules", "--notes", help="Note engine: rules (Viterbi + post-processing) or notenet (trained model in data/models/notenet.pt)"),
):
    """Transcribe a song into sargam notation (text), timed notes (JSON) and a MIDI melody."""
    from .flute import parse_flutes
    from .pipeline import run
    from .render import diagnostics, to_text, write_outputs

    if simplify == 0:
        simplify = -1.0  # explicit 0 = off, also for backends that merge by default (Config.resolved)
    cfg = _cfg(pitch_backend=backend, snap_rare=not no_snap, simplify_min_ms=simplify, notes_engine=notes)
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
    backend: str = typer.Option(Config.pitch_backend, "--pitch", help="Pitch tracker: rmvpe, pyin, crepe or fcpe"),
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


# ---------------------------------------------------------------- dataset and model

data_app = typer.Typer(help="Notation corpus (notationsworld) for training.", no_args_is_help=True)
model_app = typer.Typer(help="NoteNet: train, evaluate, package for PACE, install.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")


@data_app.command("crawl")
def data_crawl(
    kinds: str = typer.Option("sargam,piano", "--kinds", help="Page kinds to fetch"),
    delay: float = typer.Option(1.2, "--delay", help="Seconds between requests"),
    limit: int | None = typer.Option(None, "--limit"),
):
    """Fetch notation pages not yet cached under data/corpus/raw (resumable, one request at a time)."""
    from .corpus import crawl, list_urls

    list_urls()
    n = crawl(tuple(kinds.split(",")), delay, limit, progress=console.print)
    console.print(f"[green]fetched[/green] {n} new pages")


@data_app.command("build")
def data_build():
    """Parse cached pages into data/corpus/songs.jsonl."""
    from .corpus import build, corpus_dir

    counts = build()
    console.print(counts)
    console.print(f"[green]wrote[/green] {corpus_dir() / 'songs.jsonl'}")


@data_app.command("show")
def data_show(slug: str = typer.Argument(..., help="Song slug (or part of it)")):
    """Print a parsed corpus song (check the parser against the page)."""
    from .corpus import line_text, load_songs

    for rec in load_songs():
        if slug in rec["slug"]:
            console.print(f"[bold]{rec['title']}[/bold]  Sa={rec['tonic']} ({rec['tonic_source']}) {rec['url']}")
            for ln in rec["lines"]:
                console.print(f"  [dim]{(ln['lyric'] or '')[:40]:40}[/dim] {line_text(ln['notes'])}")
            return
    console.print("[red]not found[/red]")


@model_app.command("export-eval")
def model_export_eval(pitch: str = typer.Option("rmvpe,pyin", "--pitch", help="Pitch backends to export features for")):
    """Pack labelled songs (data/songs/*, web-UI library items with a reference) into data/eval/*.npz."""
    from .ml.evalset import export

    export(tuple(pitch.split(",")), progress=console.print)


@model_app.command("train", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def model_train(ctx: typer.Context):
    """Train NoteNet (all options are passed to python -m sargam.ml.train; see --help there)."""
    from .ml.train import main

    main(ctx.args)


@model_app.command("eval")
def model_eval(
    checkpoint: Path = typer.Argument(..., exists=True),
    eval_dir: Path = typer.Option(None, "--eval-dir"),
):
    """Score a checkpoint on the real-song eval set next to the rule-based baseline."""
    import torch

    from .metrics import aligned_f1, levenshtein
    from .ml.decode import DecodeParams
    from .ml.evalset import eval_dir as default_eval_dir
    from .ml.evalset import load
    from .ml.model import load_checkpoint
    from .ml.train import evaluate_real

    files = sorted((eval_dir or default_eval_dir()).glob("*.npz"))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    res = evaluate_real(load_checkpoint(checkpoint, dev), files, dev, DecodeParams())
    for f in files:
        e = load(f)
        ser = levenshtein(e["baseline"], e["truth"]) / max(1, len(e["truth"]))
        _, _, f1 = aligned_f1(e["baseline"], e["truth"])
        r = res[e["name"]]
        console.print(f"{e['name']:28} NoteNet SER {r['ser']:.3f} F1 {r['f1']:.3f}   rules SER {ser:.3f} F1 {f1:.3f}")


@model_app.command("install")
def model_install(checkpoint: Path = typer.Argument(..., exists=True, help="best.pt from a training run")):
    """Copy a trained checkpoint to data/models/notenet.pt (used by --notes notenet and the web UI)."""
    import shutil

    from .ml.infer import default_checkpoint

    dst = default_checkpoint()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, dst)
    console.print(f"[green]installed[/green] {dst}")


@model_app.command("bundle")
def model_bundle(out: Path = typer.Option(Path("dist"), "--out")):
    """Zip code + corpus + eval set + PACE scripts for training on the cluster (see pace/README.md)."""
    from .ml.bundle import make_bundle

    path = make_bundle(out)
    console.print(f"[green]wrote[/green] {path} ({path.stat().st_size / 1e6:.1f} MB). Upload it to PACE and follow pace/README.md")
