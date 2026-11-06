"""On-disk library of transcriptions: one folder per uploaded song under DATA_DIR/library/<id>/.

  meta.json        options, state, summary, revision history, saved reference info
  input.<ext>      the uploaded file
  result.pkl       pickled pipeline Result of the latest revision (instant re-rendering)
  transcription.*  .sargam.txt / .notes.json / .melody.mid written by the CLI renderer
  song.mp3, vocals.mp3   browser-friendly playback copies (made by ffmpeg when needed)
  truth.txt        reference notation (notationsworld dialect), reference_raw.txt the full pasted/fetched text
"""
from __future__ import annotations

import json
import os
import pickle
import secrets
import shutil
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

ACTIVE = ("queued", "running")
LIVE_KEYS = ("stage", "stage_key")  # progress of a running job: kept in memory only


def new_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


class Library:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._live: dict[str, dict] = {}  # stage messages of running jobs, not persisted on every update
        self._results: OrderedDict[tuple[str, int], object] = OrderedDict()

    # ------------------------------------------------------------ paths / meta

    def dir(self, item_id: str) -> Path:
        if not item_id or "/" in item_id or item_id.startswith("."):
            raise KeyError(item_id)
        return self.root / item_id

    def exists(self, item_id: str) -> bool:
        try:
            return (self.dir(item_id) / "meta.json").exists()
        except KeyError:
            return False

    def load(self, item_id: str) -> dict:
        p = self.dir(item_id) / "meta.json"
        if not p.exists():
            raise KeyError(item_id)
        with self._lock:
            meta = json.loads(p.read_text())
        live = self._live.get(item_id)
        if live and meta.get("state") in ACTIVE:
            meta.update(live)
        return meta

    def save(self, meta: dict) -> None:
        d = self.dir(meta["id"])
        meta["updated"] = time.time()
        persisted = {k: v for k, v in meta.items() if k not in LIVE_KEYS}
        tmp = d / "meta.json.tmp"
        with self._lock:
            tmp.write_text(json.dumps(persisted, indent=1))
            os.replace(tmp, d / "meta.json")

    def update(self, item_id: str, **fields) -> dict:
        with self._lock:
            meta = self.load(item_id)
            meta.update(fields)
            self.save(meta)
            return self.load(item_id)

    def set_stage(self, item_id: str, message: str, key: str) -> None:
        self._live[item_id] = {"stage": message, "stage_key": key}

    def clear_stage(self, item_id: str) -> None:
        self._live.pop(item_id, None)

    # ------------------------------------------------------------ items

    def create(self, filename: str, src_file, options: dict) -> dict:
        item_id = new_id()
        d = self.dir(item_id)
        d.mkdir(parents=True)
        ext = Path(filename).suffix.lower() or ".bin"
        with open(d / f"input{ext}", "wb") as f:
            shutil.copyfileobj(src_file, f)
        title = Path(filename).stem.replace("_", " ").strip() or "Untitled"
        now = time.time()
        meta = {
            "id": item_id, "title": title, "filename": filename, "input": f"input{ext}", "created": now,
            "options": options, "state": "queued", "error": None, "revision": 0, "revisions": [],
            "summary": None, "truth": None,
        }
        self.save(meta)
        return meta

    def list(self) -> list[dict]:
        out = []
        for p in self.root.iterdir():
            if (p / "meta.json").exists():
                try:
                    out.append(self.load(p.name))
                except (KeyError, json.JSONDecodeError):
                    continue
        out.sort(key=lambda m: m.get("created", 0), reverse=True)
        return out

    def delete(self, item_id: str) -> None:
        d = self.dir(item_id)
        with self._lock:
            for k in [k for k in self._results if k[0] == item_id]:
                del self._results[k]
            shutil.rmtree(d)

    def recover_interrupted(self) -> None:
        """Jobs that were queued/running when the server stopped can't finish; mark them so the UI offers a re-run."""
        for meta in self.list():
            if meta.get("state") in ACTIVE:
                meta["state"] = "error"
                meta["error"] = "Interrupted: the server stopped before this finished. Re-run it."
                self.save(meta)

    # ------------------------------------------------------------ results

    def store_result(self, item_id: str, revision: int, res) -> None:
        d = self.dir(item_id)
        tmp = d / "result.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, d / "result.pkl")
        with self._lock:
            self._results[(item_id, revision)] = res
            self._trim()

    def result(self, item_id: str):
        """Latest Result, unpickled once and kept in a small LRU. Raises KeyError if there is none."""
        meta = self.load(item_id)
        key = (item_id, meta.get("revision", 0))
        with self._lock:
            if key in self._results:
                self._results.move_to_end(key)
                return self._results[key]
        p = self.dir(item_id) / "result.pkl"
        if not p.exists():
            raise KeyError(item_id)
        with open(p, "rb") as f:
            res = pickle.load(f)
        with self._lock:
            self._results[key] = res
            self._trim()
        return res

    def _trim(self, keep: int = 6) -> None:
        while len(self._results) > keep:
            self._results.popitem(last=False)
