"""Vocal separation with Demucs: MLX port first (Apple Silicon), torch CPU fallback. Stems cached."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import cache_dir, load_mono, write_wav

STEM_NAMES = ("vocals", "bass", "drums", "other")


@dataclass
class Stems:
    sr: int
    vocals: np.ndarray
    bass: np.ndarray
    drums: np.ndarray
    other: np.ndarray
    backend: str


def _to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=0) if x.ndim == 2 else x


def _separate_mlx(path: Path, model: str) -> tuple[int, dict[str, np.ndarray]]:
    from demucs_mlx import Separator

    sep = Separator(model=model)
    _origin, stems = sep.separate_audio_file(str(path))
    out = {}
    for k, v in stems.items():
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        out[k] = arr
    return int(sep.samplerate), out


def _separate_torch(path: Path, model: str) -> tuple[int, dict[str, np.ndarray]]:  # pragma: no cover - fallback
    import torch
    from demucs.api import Separator

    sep = Separator(model=model, device="cpu")
    _origin, stems = sep.separate_audio_file(str(path))
    return int(sep.samplerate), {k: v.cpu().numpy() for k, v in stems.items()}


def separate(path: str | Path, sr_out: int, model: str = "htdemucs", force: bool = False) -> Stems:
    path = Path(path)
    cdir = cache_dir(path) / "stems"
    cdir.mkdir(parents=True, exist_ok=True)
    marker = cdir / "backend.txt"
    if not force and all((cdir / f"{s}.wav").exists() for s in STEM_NAMES):
        backend = marker.read_text().strip() if marker.exists() else "cached"
        return Stems(sr_out, *[load_mono(cdir / f"{s}.wav", sr_out) for s in STEM_NAMES], backend=backend)

    errors = []
    for name, fn in (("demucs-mlx", _separate_mlx), ("demucs-torch", _separate_torch)):
        try:
            sr, stems = fn(path, model)
            backend = name
            break
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    else:
        raise RuntimeError("no separation backend worked:\n" + "\n".join(errors))

    for s in STEM_NAMES:
        write_wav(cdir / f"{s}.wav", stems[s], sr)
    marker.write_text(backend)
    return Stems(sr_out, *[load_mono(cdir / f"{s}.wav", sr_out) for s in STEM_NAMES], backend=backend)
