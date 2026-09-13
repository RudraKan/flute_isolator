"""Audio loading via ffmpeg, file hashing and cache paths."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("SARGAM_DATA", Path(__file__).resolve().parent.parent / "data"))


def file_hash(path: str | Path, n: int = 16) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def cache_dir(path: str | Path) -> Path:
    d = DATA_DIR / "cache" / file_hash(path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_mono(path: str | Path, sr: int) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32 mono at `sr`."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def load_stereo(path: str | Path, sr: int) -> np.ndarray:
    """(2, n) float32 at `sr`."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).reshape(-1, 2).T.copy()


def write_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), np.asarray(y).T if y.ndim == 2 else y, sr)


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    import librosa

    return librosa.resample(y.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)
