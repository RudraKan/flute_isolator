"""FastAPI app behind `sargam serve`: upload songs, follow progress, read/play the sargam, compare with a reference."""
from __future__ import annotations

import dataclasses
import json
import mimetypes
import time
from pathlib import Path
from typing import Callable

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..model import PITCH_NAMES
from .jobs import STAGES, Worker, prepare_playback
from .library import ACTIVE, Library

STATIC = Path(__file__).parent / "static"

PITCH_BACKENDS = ["rmvpe", "pyin", "fcpe", "crepe"]
PITCH_LABELS = {"rmvpe": "RMVPE (vocal model, best)", "pyin": "pyin (classic)", "fcpe": "FCPE (fast)", "crepe": "CREPE (slow)"}
SIMPLIFY = {"auto": "Auto (tracker default)", "off": "Off", "60": "60 ms", "100": "100 ms", "150": "150 ms", "200": "200 ms"}
ENGINES = {"rules": "Rules (tuned on the reference song)", "notenet": "NoteNet (learned model)"}
STYLES = {"nw": "notationsworld (n = lower Ni, D(k) = komal)", "dots": "dots, our symbols (.N, d komal)", "dashes": "dashes (G - m - R G)"}
DETAILS = {0: "Plain notes", 1: "With ornaments and meend", 2: "Also mark unsure notes (?)"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".mp4", ".ogg", ".aac", ".webm", ".opus", ".mov", ".mkv", ".aif", ".aiff"}

# Every option the UI shows. 'affects': 'analysis' options re-run the pipeline (target 'cfg:<Config field>' or
# 'run:<pipeline.run kwarg>'), 'render' options only re-render. Adding a backend or engine = adding a choice/entry here.
OPTION_SCHEMA: list[dict] = [
    {"key": "sa", "label": "Sa (tonic)", "type": "select", "choices": [""] + PITCH_NAMES, "labels": {"": "Auto-detect"},
     "default": "", "affects": "analysis", "target": "run:sa_override",
     "help": "Leave on auto unless the detected scale sounds wrong."},
    {"key": "pitch", "label": "Pitch tracker", "type": "select", "choices": PITCH_BACKENDS, "labels": PITCH_LABELS,
     "default": "rmvpe", "affects": "analysis", "target": "cfg:pitch_backend",
     "help": "RMVPE is trained on singing in full mixes and is the most accurate; it downloads 181 MB once."},
    {"key": "notes", "label": "Note engine", "type": "select", "choices": ["rules"], "labels": ENGINES, "default": "rules",
     "affects": "analysis", "target": "cfg:notes_engine",
     "help": "NoteNet appears here once a trained model is installed (sargam model install best.pt)."},
    {"key": "simplify", "label": "Simplify", "type": "select", "choices": list(SIMPLIFY), "labels": SIMPLIFY, "default": "auto",
     "affects": "analysis", "target": "cfg:simplify_min_ms",
     "help": "Merge notes shorter than this into their neighbours: longer values give fewer, easier notes. Rules engine only."},
    {"key": "flutes", "label": "Flutes you own", "type": "text", "placeholder": "e.g. C, G bass, F#", "default": "",
     "affects": "analysis", "target": "run:flutes", "help": "Used to recommend which flute to play the song on."},
    {"key": "fold", "label": "Fold out-of-range notes into the flute range", "type": "bool", "default": False,
     "affects": "analysis", "target": "run:fold"},
    {"key": "no_separate", "label": "Input is already a solo voice or flute (skip vocal separation)", "type": "bool",
     "default": False, "affects": "analysis", "target": "run:no_separate"},
    {"key": "style", "label": "Notation style", "type": "select", "choices": list(STYLES), "labels": STYLES,
     "default": "nw", "affects": "render"},
    {"key": "detail", "label": "Detail", "type": "select", "choices": list(DETAILS), "labels": {str(k): v for k, v in DETAILS.items()},
     "default": 0, "affects": "render"},
]
_SCHEMA = {f["key"]: f for f in OPTION_SCHEMA}


def _coerce(field: dict, value):
    t = field["type"]
    if value is None:
        return field["default"]
    if t == "bool":
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
    if t == "number":
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{field['label']}: not a number")
        return max(field.get("min", v), min(field.get("max", v), v))
    if t == "select":
        choices = field["choices"]
        cand = value
        if choices and isinstance(choices[0], int):
            try:
                cand = int(value)
            except (TypeError, ValueError):
                pass
        if cand not in choices:
            raise HTTPException(400, f"{field['label']}: {value!r} is not one of {choices}")
        return cand
    return str(value).strip()


def refresh_engines() -> None:
    """NoteNet is offered only when a trained checkpoint is installed."""
    from ..ml.infer import default_checkpoint

    _SCHEMA["notes"]["choices"] = ["rules", "notenet"] if default_checkpoint().exists() else ["rules"]


def clean_options(raw: dict, base: dict | None = None) -> dict:
    refresh_engines()
    out = dict(base or {})
    for key, field in _SCHEMA.items():
        if key in raw:
            out[key] = _coerce(field, raw[key])
        elif key not in out:
            out[key] = field["default"]
    if out.get("flutes"):
        from ..flute import parse_flutes

        try:
            parse_flutes(out["flutes"])
        except ValueError as e:
            raise HTTPException(400, f"Flutes you own: {e}")
    return out


def pipeline_args(options: dict) -> tuple[Config, dict]:
    """Map UI options onto Config fields and pipeline.run keyword arguments."""
    from ..flute import parse_flutes

    cfg_fields = {f.name for f in dataclasses.fields(Config)}
    cfg_over, run_kw = {}, {}
    for field in OPTION_SCHEMA:
        if field["affects"] != "analysis":
            continue
        value = options.get(field["key"], field["default"])
        kind, name = field["target"].split(":", 1)
        if field["key"] == "simplify":
            if value == "auto":
                continue  # Config.resolved() applies the pitch backend's default
            value = -1.0 if value == "off" else float(value)
        if kind == "cfg":
            if name in cfg_fields:
                cfg_over[name] = value
        else:
            run_kw[name] = value
    run_kw["sa_override"] = run_kw.get("sa_override") or None
    run_kw["flutes"] = parse_flutes(run_kw["flutes"]) if run_kw.get("flutes") else None
    return dataclasses.replace(Config(), **cfg_over), run_kw


def default_runner(path: Path, options: dict, progress: Callable[[str], None]):
    from ..pipeline import run

    cfg, kw = pipeline_args(options)
    return run(path, cfg, progress=progress, **kw)


def vocals_stem(input_path: Path) -> Path | None:
    from ..audio import DATA_DIR, file_hash

    p = DATA_DIR / "cache" / file_hash(input_path) / "stems" / "vocals.wav"
    return p if p.exists() else None


def write_item_outputs(item_dir: Path, meta: dict, res) -> dict:
    from ..render import write_outputs

    opts = meta["options"]
    write_outputs(res, item_dir, "transcription", opts.get("detail", 0), opts.get("style", "nw"))
    stem = None if opts.get("no_separate") else vocals_stem(item_dir / meta["input"])
    extra = {"playback": prepare_playback(item_dir, meta, stem)}
    ref = item_dir / "reference_raw.txt"
    if meta.get("truth") and ref.exists():  # keep the library's score of a saved reference current after a re-run
        from .reference import ReferenceError, compare

        try:
            cmp = compare(res, ref.read_text())
            extra["truth"] = {**meta["truth"], "f1": cmp["f1"], "ser": cmp["ser"]}
        except ReferenceError:
            pass
    return extra


# ---------------------------------------------------------------- rendering for the UI


def render_view(res, style: str, detail: int) -> dict:
    from ..notes import expand_ornaments
    from ..render import to_text
    from ..swara import layout_phrase, median_note_duration

    beat = res.beats.period if res.beats is not None else None
    median = median_note_duration(res.phrases)
    phrases = []
    for i, p in enumerate(res.phrases):
        segs = []
        for seg in layout_phrase(p.notes, res.tonic.sa_midi, beat, median, detail, style):
            if seg.note is None:
                segs.append({"t": seg.text})
            else:
                n = seg.note
                segs.append({"t": seg.text, "s": round(n.start, 3), "e": round(n.end, 3), "m": int(n.midi),
                             "c": round(float(n.confidence), 2)})
        phrases.append({"i": i, "start": round(p.start, 3), "end": round(p.end, 3), "segs": segs})
    play = [[round(n.start, 3), round(n.end, 3), int(n.midi)] for n in expand_ornaments(res.notes)]
    return {
        "style": style, "detail": detail, "text": to_text(res, detail, style=style), "phrases": phrases,
        "phrases_per_block": res.cfg.phrases_per_block, "notes": play,
        "tuning_cents": round(float(res.tuning.offset_cents), 1), "duration_s": round(float(res.duration_s), 2),
    }


# ---------------------------------------------------------------- app


def create_app(library_root: Path | None = None, runner=None, after=None) -> FastAPI:
    from ..audio import DATA_DIR

    lib = Library(library_root or DATA_DIR / "library")
    lib.recover_interrupted()
    worker = Worker(lib, runner or default_runner, after if after is not None else write_item_outputs)
    app = FastAPI(title="sargam", docs_url="/api/docs", redoc_url=None)
    app.state.library = lib
    app.state.worker = worker

    def item_or_404(item_id: str) -> dict:
        try:
            return lib.load(item_id)
        except KeyError:
            raise HTTPException(404, "No such song in the library")

    def public(meta: dict) -> dict:
        out = dict(meta)
        now = time.time()
        if meta.get("state") == "running" and meta.get("started"):
            out["elapsed"] = round(now - meta["started"], 1)
        elif meta.get("state") == "done" and meta.get("started") and meta.get("finished"):
            out["elapsed"] = round(meta["finished"] - meta["started"], 1)
        if meta.get("state") == "queued":
            out["queue_position"] = worker.position(meta["id"])
        return out

    def result_or_409(item_id: str):
        meta = item_or_404(item_id)
        try:
            return meta, lib.result(item_id)
        except KeyError:
            raise HTTPException(409, "This song has no finished transcription yet")
        except Exception as e:  # noqa: BLE001 - stale pickle from an older version of the code
            raise HTTPException(409, f"Saved result can't be loaded ({type(e).__name__}); re-run the analysis")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((STATIC / "index.html").read_text())

    @app.get("/api/options")
    def options():
        refresh_engines()
        return {"schema": OPTION_SCHEMA, "stages": [{"key": k, "label": l} for k, l, _ in STAGES],
                "audio_exts": sorted(AUDIO_EXTS), "pitch_names": PITCH_NAMES}

    @app.get("/api/jobs")
    def list_jobs():
        items = []
        for m in lib.list():
            s = m.get("summary") or {}
            items.append({
                "id": m["id"], "title": m["title"], "created": m["created"], "state": m["state"],
                "sa": s.get("sa"), "duration_s": s.get("duration_s"), "has_truth": bool(m.get("truth")),
                "f1": (m.get("truth") or {}).get("f1"), "stage": m.get("stage"), "error": m.get("error"),
            })
        return {"items": items, "busy": worker.current is not None or not worker.q.empty()}

    @app.post("/api/jobs")
    def create_job(file: UploadFile = File(...), options: str = Form("{}")):
        name = file.filename or "song"
        if Path(name).suffix.lower() not in AUDIO_EXTS:
            raise HTTPException(400, f"Unsupported file type {Path(name).suffix or '(none)'}; use mp3, wav, m4a, flac or mp4")
        try:
            raw = json.loads(options or "{}")
        except json.JSONDecodeError:
            raise HTTPException(400, "options must be JSON")
        opts = clean_options(raw)
        meta = lib.create(name, file.file, opts)
        worker.submit(meta["id"])
        return public(lib.load(meta["id"]))

    @app.get("/api/jobs/{item_id}")
    def get_job(item_id: str):
        return public(item_or_404(item_id))

    @app.patch("/api/jobs/{item_id}")
    def rename(item_id: str, body: dict = Body(...)):
        item_or_404(item_id)
        title = str(body.get("title", "")).strip()
        if not title:
            raise HTTPException(400, "Title can't be empty")
        return public(lib.update(item_id, title=title[:200]))

    @app.delete("/api/jobs/{item_id}")
    def delete(item_id: str):
        meta = item_or_404(item_id)
        if meta.get("state") == "running":
            raise HTTPException(409, "It's being analysed right now; delete it when it finishes")
        lib.delete(item_id)
        return {"deleted": item_id}

    @app.post("/api/jobs/{item_id}/rerun")
    def rerun(item_id: str, body: dict = Body(default={})):
        meta = item_or_404(item_id)
        if meta.get("state") in ACTIVE:
            raise HTTPException(409, "Already queued or running")
        opts = clean_options(body.get("options") or {}, base=meta.get("options"))
        lib.update(item_id, options=opts)
        worker.submit(item_id)
        return public(lib.load(item_id))

    @app.get("/api/jobs/{item_id}/render")
    def render(item_id: str, style: str = Query("nw"), detail: int = Query(0)):
        style = _coerce(_SCHEMA["style"], style)
        detail = _coerce(_SCHEMA["detail"], detail)
        _, res = result_or_409(item_id)
        return render_view(res, style, detail)

    @app.get("/api/jobs/{item_id}/download/{kind}")
    def download(item_id: str, kind: str, style: str = Query("nw"), detail: int = Query(0)):
        meta, res = result_or_409(item_id)
        title = "".join(c if c.isalnum() or c in " -_." else "_" for c in meta["title"]).strip() or "song"
        d = lib.dir(item_id)
        if kind == "txt":
            from ..render import to_text

            text = to_text(res, _coerce(_SCHEMA["detail"], detail), style=_coerce(_SCHEMA["style"], style))
            return PlainTextResponse(text, headers={"Content-Disposition": f'attachment; filename="{title}.sargam.txt"'})
        if kind == "json":
            return FileResponse(d / "transcription.notes.json", media_type="application/json", filename=f"{title}.notes.json")
        if kind == "mid":
            return FileResponse(d / "transcription.melody.mid", media_type="audio/midi", filename=f"{title}.mid")
        raise HTTPException(404, "kind must be txt, json or mid")

    @app.get("/api/jobs/{item_id}/audio/{which}")
    def audio(item_id: str, which: str):
        meta = item_or_404(item_id)
        d = lib.dir(item_id)
        play = meta.get("playback") or {}
        if which == "song":
            name = play.get("song") or meta["input"]
        elif which == "vocals":
            name = play.get("vocals")
            if not name:
                raise HTTPException(404, "No vocal stem for this song")
        else:
            raise HTTPException(404, "which must be song or vocals")
        p = d / name
        if not p.exists():
            raise HTTPException(404, "Audio file missing")
        mt = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        return FileResponse(p, media_type=mt)

    # ------------------------------------------------------------ reference notation

    def _saved_reference(item_id: str) -> str | None:
        p = lib.dir(item_id) / "reference_raw.txt"
        if p.exists():
            return p.read_text()
        p = lib.dir(item_id) / "truth.txt"
        return p.read_text() if p.exists() else None

    @app.get("/api/jobs/{item_id}/reference")
    def get_reference(item_id: str):
        from .reference import ReferenceError, compare

        meta = item_or_404(item_id)
        text = _saved_reference(item_id)
        if text is None:
            return {"saved": None}
        _, res = result_or_409(item_id)
        try:
            cmp = compare(res, text)
        except ReferenceError as e:
            raise HTTPException(400, str(e))
        return {"saved": meta.get("truth"), "comparison": cmp}

    @app.post("/api/jobs/{item_id}/reference")
    def post_reference(item_id: str, body: dict = Body(...)):
        from ..swara import parse_notation
        from .reference import ReferenceError, compare, fetch_page_text, notation_lines, truth_file_text

        item_or_404(item_id)
        _, res = result_or_409(item_id)
        url = (body.get("url") or "").strip()
        text = body.get("text") or ""
        try:
            if url:
                text = fetch_page_text(url)
            if not text.strip():
                raise ReferenceError("Paste the reference notation or a notationsworld URL")
            cmp = compare(res, text)
        except ReferenceError as e:
            raise HTTPException(400, str(e))
        saved = None
        if body.get("save", True):
            sa, phrases = parse_notation(text, "nw")
            d = lib.dir(item_id)
            (d / "truth.txt").write_text(truth_file_text(sa, text))
            (d / "reference_raw.txt").write_text(text)
            saved = {"source": url or "pasted", "sa": sa, "saved": time.time(), "n_lines": len(notation_lines(text)),
                     "n_notes": sum(1 for p in phrases for t in p if t.kind == "note"), "f1": cmp["f1"], "ser": cmp["ser"]}
            lib.update(item_id, truth=saved)
        return {"saved": saved, "comparison": cmp}

    @app.delete("/api/jobs/{item_id}/reference")
    def delete_reference(item_id: str):
        item_or_404(item_id)
        d = lib.dir(item_id)
        for n in ("truth.txt", "reference_raw.txt"):
            (d / n).unlink(missing_ok=True)
        lib.update(item_id, truth=None)
        return {"saved": None}

    @app.get("/api/pairs")
    def pairs():
        """Library songs that have both audio and a saved reference: training/evaluation pairs."""
        out = []
        for m in lib.list():
            d = lib.dir(m["id"])
            if m.get("truth") and (d / "truth.txt").exists() and (d / m["input"]).exists():
                out.append({"id": m["id"], "title": m["title"], "audio": str(d / m["input"]), "truth": str(d / "truth.txt"),
                            "sa_truth": m["truth"].get("sa"), "source": m["truth"].get("source"),
                            "sa_pred": (m.get("summary") or {}).get("sa"), "f1": m["truth"].get("f1")})
        return {"pairs": out}

    @app.exception_handler(HTTPException)
    def http_error(_request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False, library: Path | None = None) -> None:
    import threading
    import webbrowser

    import uvicorn

    app = create_app(library)
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"sargam UI on {url}  (library in {app.state.library.root})")
    uvicorn.run(app, host=host, port=port, log_level="warning")
