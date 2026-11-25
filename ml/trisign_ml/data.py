"""Datasets.

Training reads one simple on-disk format, produced by ``trisign_ml.prepare``:

    <root>/index.csv      columns: file, label, signer       (isolated signs, label = gloss)
                          columns: file, phrase, signer      (fingerspelling)
    <root>/clips/*.npy    float16/32 arrays of shape (T, 543, 3), NaN = not detected

Splits are signer-independent: no signer appears in more than one split.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import augment
from .preprocess import drop_handless, mirror, normalize, pad, resample, select, to_features


@dataclass
class Sample:
    file: Path
    target: str      # gloss for signs, phrase for fingerspelling
    signer: str


def read_index(root: str | Path, kind: str = "sign") -> list[Sample]:
    root = Path(root)
    key = "label" if kind == "sign" else "phrase"
    with open(root / "index.csv", newline="", encoding="utf-8") as f:
        return [Sample(root / r["file"], r[key], r["signer"]) for r in csv.DictReader(f)]


def signer_split(samples: list[Sample], val_frac: float = 0.1, test_frac: float = 0.1,
                 seed: str = "trisign") -> dict[str, list[Sample]]:
    """Deterministic split by hashed signer id."""
    def bucket(signer: str) -> float:
        h = hashlib.sha1(f"{seed}:{signer}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        b = bucket(s.signer)
        out["test" if b < test_frac else "val" if b < test_frac + val_frac else "train"].append(s)
    return out


def build_vocab(samples: list[Sample]) -> list[str]:
    return sorted({s.target for s in samples})


def build_alphabet(samples: list[Sample]) -> list[str]:
    return sorted({c for s in samples for c in s.target})


class SignDataset(Dataset):
    def __init__(self, samples: list[Sample], vocab: list[str], max_frames: int, train: bool, seed: int = 0):
        known = set(vocab)
        self.samples = [s for s in samples if s.target in known]
        self.lookup = {g: i for i, g in enumerate(vocab)}
        self.max_frames, self.train = max_frames, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def features(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = drop_handless(select(frames))
        # Resample a little above the limit when training so speed augmentation has room.
        xy = normalize(resample(xy, self.max_frames * (2 if self.train else 1)))
        if self.train:
            xy = augment(xy, self.rng)
        xy = resample(xy, self.max_frames)
        return pad(to_features(xy), self.max_frames)

    def __getitem__(self, i: int):
        s = self.samples[i]
        feat, mask = self.features(np.load(s.file).astype(np.float32))
        return torch.from_numpy(feat), torch.from_numpy(mask), self.lookup[s.target]


class SpellDataset(SignDataset):
    """Fingerspelling: target is a character sequence (1-based indices, 0 = CTC blank)."""

    def __init__(self, samples: list[Sample], alphabet: list[str], max_frames: int, train: bool, seed: int = 0):
        self.samples = [s for s in samples if s.target]
        self.lookup = {c: i + 1 for i, c in enumerate(alphabet)}
        self.max_frames, self.train = max_frames, train
        self.rng = np.random.default_rng(seed)

    def features(self, frames: np.ndarray):
        # Fingerspelling keeps every frame: handless frames carry timing between letters.
        xy = normalize(resample(select(frames), self.max_frames))
        if self.train and self.rng.random() < 0.5:
            xy = mirror(xy)
        return pad(to_features(xy), self.max_frames)

    def __getitem__(self, i: int):
        s = self.samples[i]
        feat, mask = self.features(np.load(s.file).astype(np.float32))
        target = [self.lookup[c] for c in s.target if c in self.lookup]
        return torch.from_numpy(feat), torch.from_numpy(mask), torch.tensor(target, dtype=torch.long)


def collate_spell(batch):
    feats, masks, targets = zip(*batch)
    return (torch.stack(feats), torch.stack(masks),
            torch.cat(targets), torch.tensor([len(t) for t in targets], dtype=torch.long))


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
