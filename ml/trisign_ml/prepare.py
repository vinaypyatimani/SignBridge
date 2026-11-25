"""Convert source datasets into the training format (index.csv + clips/*.npy).

    python -m trisign_ml.prepare kaggle-islr  --src data/raw/asl-signs        --out data/ase/signs
    python -m trisign_ml.prepare kaggle-fs    --src data/raw/asl-fingerspelling --out data/ase/spell
    python -m trisign_ml.prepare videos       --src clips.csv                  --out data/gsg/signs
    python -m trisign_ml.prepare synthetic    --out data/synthetic/ase --classes 20

``videos`` takes a CSV with columns ``video, label, signer`` and optional
``start, end`` (seconds). Use it for ASL Citizen (one clip per sign), for
DGS Corpus segments cut from gloss annotations (start/end from the ELAN tiers),
and for your own contribution-app recordings. For fingerspelling clips use a
``phrase`` column instead of ``label`` and pass ``--kind spell``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .spec import FACE0, LHAND0, N_HOLISTIC, POSE0, RHAND0

OFFSETS = {"face": FACE0, "left_hand": LHAND0, "pose": POSE0, "right_hand": RHAND0}
SIZES = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}


class IndexWriter:
    def __init__(self, out: Path, kind: str):
        self.out = out
        (out / "clips").mkdir(parents=True, exist_ok=True)
        self.f = open(out / "index.csv", "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(["file", "label" if kind == "sign" else "phrase", "signer"])
        self.n = 0

    def add(self, frames: np.ndarray, target: str, signer: str, name: str | None = None) -> None:
        if len(frames) == 0:
            return
        rel = f"clips/{name or self.n:0>8}.npy"
        np.save(self.out / rel, frames.astype(np.float16))
        self.w.writerow([rel, target, signer])
        self.n += 1

    def close(self) -> None:
        self.f.close()
        print(f"wrote {self.n} clips to {self.out}")


def long_parquet_to_frames(path: Path) -> np.ndarray:
    """Kaggle ISLR long format (frame, type, landmark_index, x, y, z) -> (T, 543, 3)."""
    import pandas as pd

    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    frames = np.sort(df["frame"].unique())
    t_of = {f: i for i, f in enumerate(frames)}
    out = np.full((len(frames), N_HOLISTIC, 3), np.nan, dtype=np.float32)
    idx = df["type"].map(OFFSETS).to_numpy() + df["landmark_index"].to_numpy()
    out[df["frame"].map(t_of).to_numpy(), idx] = df[["x", "y", "z"]].to_numpy(np.float32)
    return out


def wide_group_to_frames(g) -> np.ndarray:
    """Kaggle fingerspelling wide format (x_face_0 ... z_right_hand_20) -> (T, 543, 3)."""
    g = g.sort_values("frame")
    out = np.full((len(g), N_HOLISTIC, 3), np.nan, dtype=np.float32)
    for part, off in OFFSETS.items():
        for c, axis in enumerate("xyz"):
            cols = [f"{axis}_{part}_{i}" for i in range(SIZES[part])]
            present = [col for col in cols if col in g.columns]
            if len(present) == len(cols):
                out[:, off:off + SIZES[part], c] = g[cols].to_numpy(np.float32)
    return out


def prepare_kaggle_islr(src: Path, out: Path) -> None:
    import pandas as pd

    meta = pd.read_csv(src / "train.csv")
    w = IndexWriter(out, "sign")
    for r in meta.itertuples():
        w.add(long_parquet_to_frames(src / r.path), r.sign, str(r.participant_id), str(r.sequence_id))
    w.close()


def prepare_kaggle_fs(src: Path, out: Path) -> None:
    import pandas as pd

    meta = pd.read_csv(src / "train.csv")
    w = IndexWriter(out, "spell")
    for path, rows in meta.groupby("path"):
        df = pd.read_parquet(src / path).reset_index()
        by_seq = dict(tuple(df.groupby("sequence_id")))
        for r in rows.itertuples():
            if r.sequence_id in by_seq:
                w.add(wide_group_to_frames(by_seq[r.sequence_id]), r.phrase, str(r.participant_id),
                      str(r.sequence_id))
    w.close()


def prepare_videos(src: Path, out: Path, kind: str) -> None:
    from .extract import HolisticExtractor

    ex = HolisticExtractor()
    w = IndexWriter(out, kind)
    key = "label" if kind == "sign" else "phrase"
    with open(src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            video = Path(r["video"]) if Path(r["video"]).is_absolute() else src.parent / r["video"]
            start = float(r["start"]) if r.get("start") else None
            end = float(r["end"]) if r.get("end") else None
            w.add(ex.video(video, start, end), r[key], r["signer"])
    w.close()
    ex.close()


def synthetic_clip(label: int, rng: np.random.Generator, t: int) -> np.ndarray:
    """A learnable fake sign: hands trace a label-specific trajectory in front of fixed shoulders."""
    frames = np.full((t, N_HOLISTIC, 3), np.nan, dtype=np.float32)
    ph = np.linspace(0, 2 * np.pi, t)
    frames[:, POSE0:POSE0 + 33, :2] = 0.5 + rng.normal(0, 0.01, (1, 33, 2))
    frames[:, POSE0 + 11, :2] = [0.62, 0.6]   # signer's left shoulder (image right)
    frames[:, POSE0 + 12, :2] = [0.38, 0.6]
    frames[:, :, 2] = 0
    a, b = 1 + label % 4, 1 + (label // 4) % 4
    for off, side in ((LHAND0, 1), (RHAND0, -1)):
        cx = 0.5 + side * 0.08 + 0.08 * np.sin(a * ph + label)
        cy = 0.5 + 0.08 * np.cos(b * ph)
        spread = rng.normal(0, 0.02, (21, 2))
        frames[:, off:off + 21, 0] = cx[:, None] + spread[:, 0]
        frames[:, off:off + 21, 1] = cy[:, None] + spread[:, 1]
    frames[:, FACE0:FACE0 + 468, :2] = [0.5, 0.35]
    frames[..., :2] += rng.normal(0, 0.004, frames[..., :2].shape)
    return frames


def prepare_synthetic(out: Path, classes: int, signers: int, per_signer: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    w = IndexWriter(out, "sign")
    for s in range(signers):
        for c in range(classes):
            for _ in range(per_signer):
                w.add(synthetic_clip(c, rng, int(rng.integers(24, 60))), f"SIGN{c:02d}", f"s{s}")
    w.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["kaggle-islr", "kaggle-fs", "videos", "synthetic"])
    ap.add_argument("--src", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kind", choices=["sign", "spell"], default="sign")
    ap.add_argument("--classes", type=int, default=20)
    ap.add_argument("--signers", type=int, default=12)
    ap.add_argument("--per-signer", type=int, default=4)
    a = ap.parse_args()
    if a.source == "kaggle-islr":
        prepare_kaggle_islr(a.src, a.out)
    elif a.source == "kaggle-fs":
        prepare_kaggle_fs(a.src, a.out)
    elif a.source == "videos":
        prepare_videos(a.src, a.out, a.kind)
    else:
        prepare_synthetic(a.out, a.classes, a.signers, a.per_signer)


if __name__ == "__main__":
    main()
