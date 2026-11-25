"""Write shared/preprocess_fixture.json: raw holistic frames + the features Python produces.

The web test (web/lib/preprocess.test.ts) runs the TypeScript port on the same
frames and must match to 1e-4. Re-run after any change to preprocess.py.
"""
import json
from pathlib import Path

import numpy as np

from trisign_ml.prepare import synthetic_clip
from trisign_ml.preprocess import preprocess

rng = np.random.default_rng(7)
frames = synthetic_clip(3, rng, 48)
frames[10:14, 468:489] = np.nan      # left hand lost for a few frames
frames[0:3, 468:489] = np.nan
frames[0:3, 522:543] = np.nan        # no hands at all in the first 3 frames -> dropped
frames[30, 489 + 11] = np.nan        # one missing shoulder

cases = []
for left_handed in (False, True):
    feat = preprocess(frames, max_frames=32, left_handed=left_handed)
    cases.append({"left_handed": left_handed, "max_frames": 32, "features": np.round(feat, 5).tolist()})

out = Path(__file__).resolve().parents[2] / "shared" / "preprocess_fixture.json"
raw = np.where(np.isnan(frames), None, np.round(frames, 5).astype(object))
out.write_text(json.dumps({"frames": raw[..., :2].tolist(), "cases": cases}), encoding="utf-8")
print(f"wrote {out}")
