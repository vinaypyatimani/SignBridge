"""Shared sign encoder with one classification head per sign language.

Languages use ISO 639-3 codes: ``ase`` (ASL), ``gsg`` (DGS), ``ssp`` (LSE).

* ``heads[lang]``     isolated-sign classifier over that language's gloss vocabulary
* ``fs_heads[lang]``  per-frame CTC head over that language's fingerspelling alphabet

The encoder is shared, so DGS and LSE (little data) reuse what ASL (lots of data)
taught it about hand and body motion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .spec import FEATURE_DIM, FS_MAX_FRAMES


@dataclass
class ModelConfig:
    in_dim: int = FEATURE_DIM
    dim: int = 192
    conv_layers: int = 2
    conv_kernel: int = 7
    transformer_layers: int = 4
    heads: int = 4
    ff_mult: int = 2
    dropout: float = 0.2
    max_frames: int = FS_MAX_FRAMES
    vocab_sizes: dict[str, int] = field(default_factory=dict)      # lang -> n glosses
    fs_alphabet_sizes: dict[str, int] = field(default_factory=dict)  # lang -> n chars (blank excluded)


class ConvBlock(nn.Module):
    """Depthwise temporal convolution with a residual connection."""

    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.pw = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x) * mask.unsqueeze(-1)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(self.pw(self.act(h)))


class SignEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(cfg.in_dim, cfg.dim), nn.LayerNorm(cfg.dim), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_frames, cfg.dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.convs = nn.ModuleList(ConvBlock(cfg.dim, cfg.conv_kernel, cfg.dropout) for _ in range(cfg.conv_layers))
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.dim * cfg.ff_mult, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.transformer_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F), mask: (B, T) bool, True = real frame. Returns (B, T, D)."""
        maskf = mask.float()
        h = self.inp(x) + self.pos[:, : x.shape[1]]
        for conv in self.convs:
            h = conv(h, maskf)
        h = self.transformer(h, src_key_padding_mask=~mask)
        return self.norm(h)


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class MultiLingualSignModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SignEncoder(cfg)
        self.drop = nn.Dropout(cfg.dropout)
        self.heads = nn.ModuleDict({lang: nn.Linear(cfg.dim, n) for lang, n in cfg.vocab_sizes.items()})
        self.fs_heads = nn.ModuleDict({lang: nn.Linear(cfg.dim, n + 1)  # +1 = CTC blank at index 0
                                       for lang, n in cfg.fs_alphabet_sizes.items()})

    def add_language(self, lang: str, n_glosses: int = 0, n_fs_chars: int = 0) -> None:
        if n_glosses:
            self.heads[lang] = nn.Linear(self.cfg.dim, n_glosses)
            self.cfg.vocab_sizes[lang] = n_glosses
        if n_fs_chars:
            self.fs_heads[lang] = nn.Linear(self.cfg.dim, n_fs_chars + 1)
            self.cfg.fs_alphabet_sizes[lang] = n_fs_chars

    def classify(self, x: torch.Tensor, mask: torch.Tensor, lang: str) -> torch.Tensor:
        """Isolated-sign logits (B, n_glosses)."""
        return self.heads[lang](self.drop(masked_mean(self.encoder(x, mask), mask)))

    def spell(self, x: torch.Tensor, mask: torch.Tensor, lang: str) -> torch.Tensor:
        """Fingerspelling log-probs (B, T, n_chars + 1) for CTC."""
        return self.fs_heads[lang](self.drop(self.encoder(x, mask))).log_softmax(-1)

    def freeze_encoder(self, n_layers: int) -> None:
        """Freeze the input projection, convs and the first ``n_layers`` transformer layers."""
        for p in list(self.encoder.inp.parameters()) + [self.encoder.pos] + list(self.encoder.convs.parameters()):
            p.requires_grad = False
        for layer in self.encoder.transformer.layers[:n_layers]:
            for p in layer.parameters():
                p.requires_grad = False

    def config_dict(self) -> dict:
        return asdict(self.cfg)


def ctc_greedy_decode(log_probs: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
    """Collapse repeats and blanks. Returns char indices (1-based, blank=0 removed)."""
    best = log_probs.argmax(-1)
    out = []
    for seq, m in zip(best, mask):
        prev, chars = 0, []
        for c in seq[m].tolist():
            if c != prev and c != 0:
                chars.append(c)
            prev = c
        out.append(chars)
    return out
