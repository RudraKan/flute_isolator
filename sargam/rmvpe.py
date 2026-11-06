"""RMVPE pitch estimator (Wei et al. 2023, "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music").

Network re-implemented to match the public checkpoint distributed with the RVC project (MIT licence):
log-mel front-end at 16 kHz (128 bands, 30-8000 Hz, hop 160 = 10 ms) -> residual U-Net -> BiGRU -> 360 sigmoid
bins of 20 cents starting at ~31.7 Hz. Weights (181 MB) are downloaded once into DATA_DIR/models (or $SARGAM_MODELS).
"""
from __future__ import annotations

import os
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

WEIGHTS_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
SR = 16000
HOP = 160
N_BINS = 360
CENTS_PER_BIN = 20.0
# cents above 10 Hz of bin 0 (the checkpoint's mapping); f0 = 10 * 2 ** (cents / 1200)
CENTS_OFFSET = 1997.3794084376191
_CHUNK = 2048  # frames per core chunk (20.5 s); with context this bounds memory to ~2 GB on MPS
_CONTEXT = 128  # frames of context on each side of a chunk (BiGRU + conv receptive field)


def models_dir() -> Path:
    from .audio import DATA_DIR

    return Path(os.environ.get("SARGAM_MODELS", DATA_DIR / "models"))


def weights_path() -> Path:
    p = models_dir() / "rmvpe.pt"
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading RMVPE weights (181 MB, once) to {p}", file=sys.stderr)
    tmp = p.with_suffix(".part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    tmp.rename(p)
    return p


def bin_to_hz(bins: np.ndarray | float) -> np.ndarray:
    return 10.0 * 2.0 ** ((CENTS_OFFSET + CENTS_PER_BIN * np.asarray(bins, dtype=np.float64)) / 1200.0)


def decode(salience: np.ndarray, thr: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Local weighted average of the 9 bins around the argmax (the decoder RMVPE was trained with).

    salience: (frames, 360). Returns (f0 Hz with NaN where the peak is below `thr`, peak salience)."""
    sal = np.asarray(salience, dtype=np.float64)
    n = sal.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0)
    center = np.argmax(sal, axis=1)
    padded = np.pad(sal, ((0, 0), (4, 4)))
    idx = center[:, None] + np.arange(9)[None, :]  # window in padded coordinates
    w = padded[np.arange(n)[:, None], idx]
    bins = idx - 4.0
    wsum = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (w * bins).sum(axis=1) / wsum
    peak = sal.max(axis=1)
    f0 = np.where((peak > thr) & (wsum > 0), bin_to_hz(b), np.nan)
    return f0, peak


# ---------------------------------------------------------------- network


def _build_net():
    import torch
    from torch import nn

    class ConvBlockRes(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout, momentum=0.01), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout, momentum=0.01), nn.ReLU(),
            )
            self.shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else None

        def forward(self, x):
            return self.conv(x) + (self.shortcut(x) if self.shortcut is not None else x)

    class ResEncoderBlock(nn.Module):
        def __init__(self, cin: int, cout: int, pool: bool, n_blocks: int):
            super().__init__()
            self.conv = nn.ModuleList([ConvBlockRes(cin, cout)] + [ConvBlockRes(cout, cout) for _ in range(n_blocks - 1)])
            self.pool = nn.AvgPool2d(2) if pool else None

        def forward(self, x):
            for c in self.conv:
                x = c(x)
            return x if self.pool is None else (x, self.pool(x))

    class Encoder(nn.Module):
        def __init__(self, n_layers: int, n_blocks: int, cout: int = 16):
            super().__init__()
            self.bn = nn.BatchNorm2d(1, momentum=0.01)
            self.layers = nn.ModuleList()
            cin = 1
            for _ in range(n_layers):
                self.layers.append(ResEncoderBlock(cin, cout, True, n_blocks))
                cin, cout = cout, cout * 2
            self.out_channels = cout

        def forward(self, x):
            skips = []
            x = self.bn(x)
            for layer in self.layers:
                skip, x = layer(x)
                skips.append(skip)
            return x, skips

    class Intermediate(nn.Module):
        def __init__(self, cin: int, cout: int, n_layers: int, n_blocks: int):
            super().__init__()
            self.layers = nn.ModuleList(
                [ResEncoderBlock(cin, cout, False, n_blocks)] + [ResEncoderBlock(cout, cout, False, n_blocks) for _ in range(n_layers - 1)]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class ResDecoderBlock(nn.Module):
        def __init__(self, cin: int, cout: int, n_blocks: int):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(cin, cout, 3, 2, 1, output_padding=1, bias=False), nn.BatchNorm2d(cout, momentum=0.01), nn.ReLU(),
            )
            self.conv2 = nn.ModuleList([ConvBlockRes(cout * 2, cout)] + [ConvBlockRes(cout, cout) for _ in range(n_blocks - 1)])

        def forward(self, x, skip):
            x = torch.cat((self.conv1(x), skip), dim=1)
            for c in self.conv2:
                x = c(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, cin: int, n_layers: int, n_blocks: int):
            super().__init__()
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(ResDecoderBlock(cin, cin // 2, n_blocks))
                cin //= 2

        def forward(self, x, skips):
            for i, layer in enumerate(self.layers):
                x = layer(x, skips[-1 - i])
            return x

    class DeepUnet(nn.Module):
        def __init__(self, n_blocks: int = 4, n_layers: int = 5, n_inter: int = 4):
            super().__init__()
            self.encoder = Encoder(n_layers, n_blocks)
            self.intermediate = Intermediate(self.encoder.out_channels // 2, self.encoder.out_channels, n_inter, n_blocks)
            self.decoder = Decoder(self.encoder.out_channels, n_layers, n_blocks)

        def forward(self, x):
            x, skips = self.encoder(x)
            return self.decoder(self.intermediate(x), skips)

    class BiGRU(nn.Module):
        def __init__(self, nin: int, nhid: int):
            super().__init__()
            self.gru = nn.GRU(nin, nhid, num_layers=1, batch_first=True, bidirectional=True)

        def forward(self, x):
            return self.gru(x)[0]

    class E2E(nn.Module):
        """mel (B, 128, T) -> salience (B, T, 360). T must be a multiple of 32."""

        def __init__(self):
            super().__init__()
            self.unet = DeepUnet()
            self.cnn = nn.Conv2d(16, 3, 3, padding=1)
            self.fc = nn.Sequential(BiGRU(3 * 128, 256), nn.Linear(512, N_BINS), nn.Dropout(0.25), nn.Sigmoid())

        def forward(self, mel):
            x = mel.transpose(-1, -2).unsqueeze(1)
            x = self.cnn(self.unet(x)).transpose(1, 2).flatten(-2)
            return self.fc(x)

    return E2E()


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=2)
def load_model(device: str | None = None):
    import torch

    device = device or _device()
    net = _build_net()
    state = torch.load(weights_path(), map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    return net.eval().to(device), device


@lru_cache(maxsize=1)
def _mel_basis() -> np.ndarray:
    import librosa

    return librosa.filters.mel(sr=SR, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True).astype(np.float32)


def log_mel(y16: np.ndarray, device: str):
    """(1, 128, frames) log-mel exactly as the checkpoint was trained (hann 1024, centred STFT, log clamp 1e-5)."""
    import torch

    audio = torch.from_numpy(np.ascontiguousarray(y16, dtype=np.float32))[None].to(device)
    spec = torch.stft(audio, n_fft=1024, hop_length=HOP, win_length=1024, window=torch.hann_window(1024, device=device),
                      center=True, return_complex=True).abs()
    mel = torch.matmul(torch.from_numpy(_mel_basis()).to(device), spec)
    return torch.log(torch.clamp(mel, min=1e-5))


def _run_padded(net, mel):
    """mel (1, 128, T) -> (T, 360); pads T up to a multiple of 32."""
    import torch.nn.functional as F

    t = mel.shape[-1]
    pad = (-t) % 32
    if pad:
        mel = F.pad(mel, (0, pad), mode="reflect" if pad < t else "constant")
    return net(mel)[0, :t]


def salience(y: np.ndarray, sr: int, device: str | None = None) -> np.ndarray:
    """Raw RMVPE output for mono audio: (frames, 360) in [0, 1], one frame per 10 ms (frame i centred at i*10 ms).

    Long inputs run in overlapping chunks so memory stays bounded."""
    import librosa
    import torch

    y16 = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=SR) if sr != SR else np.asarray(y, np.float32)
    net, device = load_model(device)
    with torch.no_grad():
        mel = log_mel(y16, device)
        n = mel.shape[-1]
        out = np.zeros((n, N_BINS), dtype=np.float32)
        for c0 in range(0, n, _CHUNK):
            c1 = min(n, c0 + _CHUNK)
            a, b = max(0, c0 - _CONTEXT), min(n, c1 + _CONTEXT)
            sal = _run_padded(net, mel[..., a:b])
            out[c0:c1] = sal[c0 - a : c0 - a + (c1 - c0)].float().cpu().numpy()
    return out
