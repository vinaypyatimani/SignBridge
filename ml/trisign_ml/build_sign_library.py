"""Build the sign library the avatar plays from: one reference trajectory per gloss.

    python -m trisign_ml.build_sign_library --lang ase --data data/ase/signs \
        --out ../backend/data/sign_library/ase.json [--glosses glosses.txt] [--verified verified.csv]

For each gloss, up to ``--max-clips`` clips are normalized, resampled to
``--frames`` frames and combined by per-point median, which smooths away one
signer's idiosyncrasies. Only glosses in ``--verified`` (gloss column) are
written when that file is given: the library should hold signs a Deaf reviewer
has checked.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import read_index
from .preprocess import normalize, select


def fill_gaps(xy: np.ndarray) -> np.ndarray:
    """Forward/backward fill missing points in time so trajectories are continuous."""
    out = xy.copy()
    for p in range(out.shape[1]):
        col = out[:, p, 0]
        ok = np.isfinite(col)
        if not ok.any():
            continue
        idx = np.where(ok, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        first = np.argmax(ok)
        idx[:first] = first
        out[:, p] = out[idx, p]
    return out


def resample_linear(xy: np.ndarray, frames: int) -> np.ndarray:
    t = np.linspace(0, len(xy) - 1, frames)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(xy) - 1)
    w = (t - lo)[:, None, None]
    return xy[lo] * (1 - w) + xy[hi] * w


def reference(clips: list[np.ndarray], frames: int) -> np.ndarray:
    stack = np.stack([resample_linear(fill_gaps(normalize(select(c))), frames) for c in clips])
    return np.nanmedian(stack, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--max-clips", type=int, default=8)
    ap.add_argument("--glosses", type=Path, help="text file, one gloss per line, to limit the library")
    ap.add_argument("--verified", type=Path, help="CSV with a gloss column listing reviewed signs")
    a = ap.parse_args()

    by_gloss: dict[str, list[Path]] = defaultdict(list)
    for s in read_index(a.data, "sign"):
        by_gloss[s.target].append(s.file)
    allowed = set(by_gloss)
    if a.glosses:
        allowed &= {l.strip() for l in a.glosses.read_text(encoding="utf-8").splitlines() if l.strip()}
    if a.verified:
        with open(a.verified, newline="", encoding="utf-8") as f:
            allowed &= {r["gloss"] for r in csv.DictReader(f)}

    signs = {}
    for gloss in sorted(allowed):
        files = by_gloss[gloss][: a.max_clips]
        ref = reference([np.load(f).astype(np.float32) for f in files], a.frames)
        if not np.isfinite(ref).any():
            continue
        signs[gloss] = {"n_clips": len(files),
                        "frames": np.round(np.nan_to_num(ref, nan=0.0), 3).tolist()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"lang": a.lang, "fps": 20, "frames": a.frames,
                                 "signs": signs}, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(signs)} signs to {a.out}")


if __name__ == "__main__":
    main()
