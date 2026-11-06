"""Background analysis: one worker thread (Demucs is heavy), a FIFO queue, stage tracking for the UI."""
from __future__ import annotations

import queue
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from .library import Library

# (key, label shown in the UI, message prefixes from pipeline progress that start this stage)
STAGES: list[tuple[str, str, tuple[str, ...]]] = [
    ("load", "Loading audio", ("loading",)),
    ("separate", "Separating vocals", ("separat",)),
    ("pitch", "Tracking pitch", ("tracking pitch",)),
    ("notes", "Finding notes", ("finding notes",)),
    ("tonic", "Detecting Sa and scale", ("detecting", "tracking beats")),
    ("write", "Writing notation", ("writing", "preparing")),
]


def stage_key(message: str) -> str | None:
    m = message.lower()
    for key, _label, prefixes in STAGES:
        if m.startswith(prefixes):
            return key
    return None


Runner = Callable[[Path, dict, Callable[[str], None]], object]


def summarize(res) -> dict:
    """The header-card facts of a Result, JSON-ready."""
    from ..model import pitch_name

    t = res.tonic
    order = np.argsort(-np.asarray(t.scores))
    alt = pitch_name(int(order[1])) if len(order) > 1 and int(order[0]) == t.pc else pitch_name(int(order[0]))
    f = res.flute
    return {
        "sa": t.name, "sa_hz": round(float(t.sa_hz), 1), "sa_midi": int(t.sa_midi),
        "tonic_label": t.label, "tonic_margin": round(float(t.margin), 3), "tonic_confidence": round(float(t.confidence), 3),
        "overridden": bool(t.overridden), "ambiguity": t.ambiguity, "alternative": alt,
        "scale": res.scale.name, "mode": t.mode,
        "tempo_bpm": round(float(res.beats.bpm)) if res.beats is not None else None,
        "flute_advice": f.advice, "range": f.rel_range_str, "fits": bool(f.fits),
        "recommended_flute": f.recommended.name if f.recommended else None,
        "transposition": int(f.transposition_semitones),
        "key_changes": [{"start": round(k.start_s, 1), "end": round(k.end_s, 1), "tonic": pitch_name(k.local_tonic_pc)}
                        for k in res.key_changes],
        "tuning_cents": round(float(res.tuning.offset_cents), 1),
        "duration_s": round(float(res.duration_s), 1),
        "n_notes": len(res.notes), "n_phrases": len(res.phrases),
        "separation": res.separation_backend, "pitch_backend": res.cfg.pitch_backend,
        "notes_engine": getattr(res.cfg, "notes_engine", "rules"),
        "log": list(res.log),
    }


def transcode_mp3(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return True
    tmp = dst.with_suffix(".tmp.mp3")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-c:a", "libmp3lame", "-q:a", "3", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except (subprocess.SubprocessError, OSError):
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dst)
    return True


BROWSER_SAFE = {".mp3", ".m4a", ".wav", ".ogg", ".aac"}


def prepare_playback(item_dir: Path, meta: dict, vocals_wav: Path | None) -> dict:
    """Make sure the browser can play the song and the vocal stem; returns {'song': name, 'vocals': name|None}."""
    src = item_dir / meta["input"]
    song = meta["input"] if src.suffix.lower() in BROWSER_SAFE else ("song.mp3" if transcode_mp3(src, item_dir / "song.mp3") else meta["input"])
    vocals = None
    if vocals_wav is not None and vocals_wav.exists():
        vocals = "vocals.mp3" if transcode_mp3(vocals_wav, item_dir / "vocals.mp3") else None
    return {"song": song, "vocals": vocals}


class Worker:
    def __init__(self, library: Library, runner: Runner, after: Callable[[Path, dict, object], dict] | None = None):
        self.lib = library
        self.runner = runner
        self.after = after  # writes outputs/playback files, returns extra meta fields
        self.q: queue.Queue[str] = queue.Queue()
        self.current: str | None = None
        self._thread = threading.Thread(target=self._loop, name="sargam-worker", daemon=True)
        self._thread.start()

    def submit(self, item_id: str) -> None:
        self.lib.update(item_id, state="queued", error=None, queued_at=time.time())
        self.q.put(item_id)

    def position(self, item_id: str) -> int:
        """0 = running now, 1.. = place in the queue, -1 = not waiting."""
        if self.current == item_id:
            return 0
        waiting = list(self.q.queue)
        return waiting.index(item_id) + 1 if item_id in waiting else -1

    def join_idle(self, timeout: float = 30.0) -> bool:
        """Test helper: wait until the queue is drained and nothing is running."""
        end = time.time() + timeout
        while time.time() < end:
            if self.q.unfinished_tasks == 0:
                return True
            time.sleep(0.02)
        return False

    def _loop(self) -> None:
        while True:
            item_id = self.q.get()
            try:
                self.current = item_id
                self._process(item_id)
            except Exception:  # noqa: BLE001 - never let the worker thread die
                traceback.print_exc()
            finally:
                self.current = None
                self.lib.clear_stage(item_id)
                self.q.task_done()

    def _process(self, item_id: str) -> None:
        if not self.lib.exists(item_id):
            return  # deleted while queued
        meta = self.lib.update(item_id, state="running", started=time.time(), error=None)
        d = self.lib.dir(item_id)

        def progress(message: str) -> None:
            key = stage_key(message)
            prev = self.lib._live.get(item_id, {}).get("stage_key")
            self.lib.set_stage(item_id, message, key or prev or "load")

        progress("loading audio")
        try:
            res = self.runner(d / meta["input"], dict(meta["options"]), progress)
            progress("writing notation")
            extra = self.after(d, meta, res) if self.after else {}
        except Exception as e:  # noqa: BLE001
            (d / "error.log").write_text(traceback.format_exc())
            msg = str(e).strip() or type(e).__name__
            self.lib.update(item_id, state="error", error=f"{type(e).__name__}: {msg}", finished=time.time())
            return
        if not self.lib.exists(item_id):
            return
        meta = self.lib.load(item_id)
        rev = int(meta.get("revision", 0)) + 1
        self.lib.store_result(item_id, rev, res)
        summary = summarize(res)
        revisions = list(meta.get("revisions", []))
        revisions.append({"rev": rev, "options": meta["options"], "finished": time.time(),
                          "sa": summary["sa"], "n_notes": summary["n_notes"], "pitch": summary["pitch_backend"]})
        self.lib.update(item_id, state="done", finished=time.time(), revision=rev, revisions=revisions[-20:],
                        summary=summary, **extra)
