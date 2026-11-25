"""Turn raw Holistic frames into model features.

This is the single source of truth for preprocessing. ``web/lib/preprocess.ts``
is a line-by-line port; ``tests/test_preprocess.py`` checks both against the
fixture in ``shared/preprocess_fixture.json``.
"""
from __future__ import annotations

import numpy as np

from .spec import SPEC, N_POINTS

_IDX = np.array(SPEC["indices"])
_MIRROR = np.array(SPEC["mirror"])
_HANDS = np.array(SPEC["groups"]["left_hand"] + SPEC["groups"]["right_hand"])
_LS, _RS = SPEC["left_shoulder"], SPEC["right_shoulder"]
EPS = 1e-6


def select(frames: np.ndarray) -> np.ndarray:
    """(T, 543, 3) holistic frames -> (T, N, 2) selected x, y (NaN where missing)."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1] != SPEC["n_holistic"]:
        raise ValueError(f"expected (T, {SPEC['n_holistic']}, 2|3), got {frames.shape}")
    return frames[:, _IDX, :2]


def drop_handless(xy: np.ndarray) -> np.ndarray:
    """Remove frames where neither hand was detected (keeps all if none have hands)."""
    has_hand = ~np.isnan(xy[:, _HANDS, 0]).all(axis=1)
    return xy[has_hand] if has_hand.any() else xy


def resample(xy: np.ndarray, max_frames: int) -> np.ndarray:
    """Uniformly subsample to at most ``max_frames`` frames (never upsamples)."""
    t = xy.shape[0]
    if t <= max_frames:
        return xy
    idx = np.floor(np.linspace(0, t - 1, max_frames)).astype(int)
    return xy[idx]


def normalize(xy: np.ndarray) -> np.ndarray:
    """Centre on the mean shoulder midpoint and scale by mean shoulder width.

    Falls back to the centroid and RMS spread of all visible points when pose
    is missing. NaNs are preserved.
    """
    ls, rs = xy[:, _LS], xy[:, _RS]
    mid = (ls + rs) / 2
    width = np.linalg.norm(ls - rs, axis=-1)
    if np.isfinite(width).any() and np.nanmean(width) > EPS:
        center = np.nanmean(mid, axis=0)
        scale = float(np.nanmean(width))
    else:
        pts = xy.reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            return xy
        center = pts.mean(axis=0)
        scale = float(np.sqrt(((pts - center) ** 2).sum(axis=1).mean())) * 2 + EPS
    return (xy - center) / scale


def mirror(xy: np.ndarray) -> np.ndarray:
    """Left/right mirror of normalized coordinates (for left-handed signers and augmentation)."""
    out = xy[:, _MIRROR].copy()
    out[..., 0] = -out[..., 0]
    return out


def to_features(xy: np.ndarray) -> np.ndarray:
    """(T, N, 2) normalized -> (T, N*4) positions + velocities, NaN -> 0."""
    vel = np.zeros_like(xy)
    vel[1:] = xy[1:] - xy[:-1]
    feat = np.concatenate([xy.reshape(len(xy), -1), vel.reshape(len(xy), -1)], axis=1)
    return np.nan_to_num(feat, nan=0.0).astype(np.float32)


def preprocess(frames: np.ndarray, max_frames: int, left_handed: bool = False,
               drop_empty: bool = True) -> np.ndarray:
    """Full pipeline: holistic frames -> (T', FEATURE_DIM) features, T' <= max_frames."""
    xy = select(frames)
    if drop_empty:
        xy = drop_handless(xy)
    xy = resample(xy, max_frames)
    xy = normalize(xy)
    if left_handed:
        xy = mirror(xy)
    return to_features(xy)


def pad(feat: np.ndarray, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad features to ``max_frames``; returns (features, mask) with mask True = real frame."""
    t = min(len(feat), max_frames)
    out = np.zeros((max_frames, feat.shape[1]), dtype=np.float32)
    out[:t] = feat[:t]
    mask = np.zeros(max_frames, dtype=bool)
    mask[:t] = True
    return out, mask


assert N_POINTS == len(_IDX)
