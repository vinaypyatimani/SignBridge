"""Augmentations applied to normalized (T, N, 2) coordinates before featurisation."""
from __future__ import annotations

import numpy as np

from .preprocess import mirror


def augment(xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    xy = xy.copy()
    # Mirror: covers left-handed signers and doubles the data.
    if rng.random() < 0.5:
        xy = mirror(xy)
    # Small rotation, scale and shift around the shoulder centre.
    theta = np.deg2rad(rng.uniform(-12, 12))
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    xy = xy @ rot.T * rng.uniform(0.85, 1.15) + rng.uniform(-0.1, 0.1, size=2).astype(np.float32)
    # Speed change by resampling in time.
    t = len(xy)
    if t > 4 and rng.random() < 0.6:
        new_t = max(4, int(round(t * rng.uniform(0.75, 1.3))))
        idx = np.clip(np.round(np.linspace(0, t - 1, new_t)).astype(int), 0, t - 1)
        xy = xy[idx]
    # Landmark dropout: hide random points in random frames.
    if rng.random() < 0.5:
        drop = rng.random(xy.shape[:2]) < 0.05
        xy[drop] = np.nan
    # Hide one hand for a short span (simulates tracking loss).
    if rng.random() < 0.2 and len(xy) > 6:
        s = rng.integers(0, len(xy) - 3)
        hand = slice(0, 21) if rng.random() < 0.5 else slice(21, 42)
        xy[s:s + rng.integers(2, 6), hand] = np.nan
    return xy
