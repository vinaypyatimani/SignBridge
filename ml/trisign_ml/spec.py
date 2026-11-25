"""Landmark specification shared by training (Python) and inference (browser).

MediaPipe Holistic frames are stored as a (543, 3) array in the same order the
Google Kaggle ASL datasets use:

    face        0..467   (468 points; the 10 iris points of newer models are dropped)
    left_hand   468..488 (21 points)
    pose        489..521 (33 points)
    right_hand  522..542 (21 points)

"left" and "right" are as MediaPipe Holistic reports them. Do NOT mirror camera
frames before running the landmarker; mirror only the on-screen preview.

Running this module writes ``shared/landmark_spec.json``, which the web app
reads so both sides select and mirror exactly the same points.
"""
from __future__ import annotations

import json
from pathlib import Path

N_HOLISTIC = 543
FACE0, LHAND0, POSE0, RHAND0 = 0, 468, 489, 522

# Lips (outer + inner contour), as used by top Kaggle ISLR solutions.
LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
# Eyebrows carry grammar (questions, conditionals) in all three sign languages.
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
# Upper-body pose: shoulders, elbows, wrists, hand points (11..22).
POSE_UPPER = list(range(11, 23))
POSE_MIRROR_PAIRS = [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]

LEFT_SHOULDER = POSE0 + 11
RIGHT_SHOULDER = POSE0 + 12


def build_spec() -> dict:
    groups: dict[str, list[int]] = {
        "left_hand": [LHAND0 + i for i in range(21)],
        "right_hand": [RHAND0 + i for i in range(21)],
        "pose": [POSE0 + i for i in POSE_UPPER],
        "lips": [FACE0 + i for i in LIPS],
        "left_brow": [FACE0 + i for i in LEFT_BROW],
        "right_brow": [FACE0 + i for i in RIGHT_BROW],
    }
    order = ["left_hand", "right_hand", "pose", "lips", "left_brow", "right_brow"]
    indices: list[int] = [i for g in order for i in groups[g]]
    pos = {g: [indices.index(i) for i in groups[g]] for g in order}

    # Mirror permutation over the selected points: new[k] = old[mirror[k]].
    mirror = list(range(len(indices)))
    for a, b in [("left_hand", "right_hand"), ("left_brow", "right_brow")]:
        for pa, pb in zip(pos[a], pos[b]):
            mirror[pa], mirror[pb] = pb, pa
    sel_pose = {POSE_UPPER[k]: p for k, p in enumerate(pos["pose"])}
    for a, b in POSE_MIRROR_PAIRS:
        mirror[sel_pose[a]], mirror[sel_pose[b]] = sel_pose[b], sel_pose[a]
    # Lips: left/right contour swap is approximated by keeping order (x is negated).

    return {
        "version": 1,
        "n_holistic": N_HOLISTIC,
        "layout": {"face": FACE0, "left_hand": LHAND0, "pose": POSE0, "right_hand": RHAND0},
        "indices": indices,
        "groups": pos,
        "mirror": mirror,
        "left_shoulder": indices.index(LEFT_SHOULDER),
        "right_shoulder": indices.index(RIGHT_SHOULDER),
        "max_frames": 64,
        "fs_max_frames": 192,
        "coords": ["x", "y"],
        "features": "position + velocity, normalized per window by shoulder centre and width",
    }


SPEC = build_spec()
N_POINTS = len(SPEC["indices"])          # 104
FEATURE_DIM = N_POINTS * 2 * 2           # (x, y) positions + velocities = 416
MAX_FRAMES = SPEC["max_frames"]
FS_MAX_FRAMES = SPEC["fs_max_frames"]


def write_spec(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SPEC, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    out = write_spec(Path(__file__).resolve().parents[2] / "shared" / "landmark_spec.json")
    print(f"wrote {out} ({N_POINTS} points, feature dim {FEATURE_DIM})")
