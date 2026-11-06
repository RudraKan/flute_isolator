"""NoteNet: Sa-relative pitch salience + loudness -> per-frame note class and onset.

Input is a salience image over 20-cent bins (rendered from a pitch contour here, or a real tracker salience
rolled to Sa), so the 2-D conv front-end sees vibrato, slides and ornaments as local shapes, and the
recurrent layer decides which notes the notation writer would write.
"""
from __future__ import annotations

import torch
from torch import nn

from .features import BINS_PER_SEMI, N_BINS, N_CLASSES, REL_LO


class SalienceRenderer(nn.Module):
    """rel_cents (B, T) with NaN for unvoiced -> (B, 1, T, N_BINS) Gaussian bumps."""

    def __init__(self, sigma_cents: float = 25.0):
        super().__init__()
        centers = REL_LO * 100.0 + (torch.arange(N_BINS, dtype=torch.float32) + 0.5) * (100.0 / BINS_PER_SEMI)
        self.register_buffer("centers", centers, persistent=False)
        self.sigma = sigma_cents

    def forward(self, rel_cents: torch.Tensor) -> torch.Tensor:
        voiced = torch.isfinite(rel_cents)
        r = torch.where(voiced, rel_cents, torch.zeros_like(rel_cents))
        sal = torch.exp(-0.5 * ((r.unsqueeze(-1) - self.centers) / self.sigma) ** 2)
        return (sal * voiced.unsqueeze(-1)).unsqueeze(1)


class NoteNet(nn.Module):
    """The tconv residual gives the temporal stack local position information; the transformer then relates
    notes within the window (inference runs on overlapping windows of the training length)."""

    def __init__(self, channels: int = 16, d_model: int = 256, hidden: int = 192, layers: int = 4, dropout: float = 0.1,
                 temporal: str = "transformer", heads: int = 4):
        super().__init__()
        self.hparams = dict(channels=channels, d_model=d_model, hidden=hidden, layers=layers, dropout=dropout,
                            temporal=temporal, heads=heads)
        self.render = SalienceRenderer()
        c = channels
        # full 20-cent resolution only in the first layer (cheap: one input channel), then semitone resolution
        self.front = nn.Sequential(
            nn.Conv2d(1, c // 2, (5, 7), padding=(2, 3)), nn.GELU(),
            nn.MaxPool2d((1, BINS_PER_SEMI)),
            nn.Conv2d(c // 2, c, (3, 3), padding=1), nn.GELU(),
            nn.Conv2d(c, c, (3, 3), padding=1), nn.GELU(),
        )
        n_semis = N_BINS // BINS_PER_SEMI
        self.proj = nn.Sequential(nn.Linear(c * n_semis + 3, d_model), nn.GELU(), nn.Dropout(dropout))
        self.tconv = nn.Sequential(nn.Conv1d(d_model, d_model, 5, padding=2), nn.GELU())
        self.temporal = temporal
        if temporal == "gru":  # fast with cuDNN, slow on Apple MPS
            self.rnn = nn.GRU(d_model, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)
            d_out = 2 * hidden
        else:
            layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.rnn = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            d_out = d_model
        self.head_cls = nn.Linear(d_out, N_CLASSES)
        self.head_onset = nn.Linear(d_out, 1)

    def forward(self, rel_cents: torch.Tensor, energy: torch.Tensor, salience: torch.Tensor | None = None):
        """rel_cents, energy: (B, T). salience: optional (B, 1, T, N_BINS) to use instead of the rendered one."""
        sal = self.render(rel_cents) if salience is None else salience
        voiced = torch.isfinite(rel_cents).float()
        d_energy = torch.diff(energy, dim=1, prepend=energy[:, :1])
        x = self.front(sal)  # (B, C, T, n_semis)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
        x = torch.cat([x, voiced.unsqueeze(-1), energy.unsqueeze(-1), 4.0 * d_energy.unsqueeze(-1)], dim=-1)
        x = self.proj(x)
        x = x + self.tconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.rnn(x)[0] if self.temporal == "gru" else self.rnn(x)
        return self.head_cls(x), self.head_onset(x).squeeze(-1)


def build_model(hparams: dict | None = None) -> NoteNet:
    return NoteNet(**(hparams or {}))


def load_checkpoint(path, device: str = "cpu") -> NoteNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["hparams"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model.to(device)
