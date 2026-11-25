import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from trisign_ml.data import SignDataset, read_index, signer_split
from trisign_ml.model import ModelConfig, MultiLingualSignModel, ctc_greedy_decode
from trisign_ml.prepare import prepare_synthetic, synthetic_clip
from trisign_ml.preprocess import mirror, normalize, pad, preprocess, select
from trisign_ml.spec import FEATURE_DIM, N_POINTS, SPEC

ROOT = Path(__file__).resolve().parents[2]


def test_spec_is_consistent():
    assert len(SPEC["indices"]) == N_POINTS == len(set(SPEC["indices"]))
    assert sorted(SPEC["mirror"]) == list(range(N_POINTS))
    m = np.array(SPEC["mirror"])
    assert (m[m] == np.arange(N_POINTS)).all(), "mirror must be an involution"
    shared = json.loads((ROOT / "shared" / "landmark_spec.json").read_text(encoding="utf-8"))
    assert shared["indices"] == SPEC["indices"] and shared["mirror"] == SPEC["mirror"]


def test_normalize_centres_shoulders():
    xy = normalize(select(synthetic_clip(1, np.random.default_rng(0), 30)))
    mid = (xy[:, SPEC["left_shoulder"]] + xy[:, SPEC["right_shoulder"]]) / 2
    width = np.linalg.norm(xy[:, SPEC["left_shoulder"]] - xy[:, SPEC["right_shoulder"]], axis=-1)
    assert np.allclose(mid.mean(0), 0, atol=1e-5)
    assert np.isclose(width.mean(), 1, atol=1e-5)


def test_mirror_twice_is_identity():
    xy = normalize(select(synthetic_clip(2, np.random.default_rng(0), 20)))
    assert np.allclose(mirror(mirror(xy)), xy, equal_nan=True)


def test_preprocess_shapes_and_padding():
    feat = preprocess(synthetic_clip(0, np.random.default_rng(0), 100), max_frames=64)
    assert feat.shape == (64, FEATURE_DIM) and np.isfinite(feat).all()
    padded, mask = pad(feat[:10], 64)
    assert padded.shape == (64, FEATURE_DIM) and mask.sum() == 10


def test_fixture_matches_current_preprocessing():
    fx = json.loads((ROOT / "shared" / "preprocess_fixture.json").read_text(encoding="utf-8"))
    raw = np.array([[[np.nan if v is None else v for v in p] for p in f] for f in fx["frames"]], np.float32)
    frames = np.concatenate([raw, np.zeros(raw.shape[:2] + (1,), np.float32)], axis=-1)
    for case in fx["cases"]:
        got = preprocess(frames, case["max_frames"], left_handed=case["left_handed"])
        assert np.allclose(got, np.array(case["features"]), atol=1e-4), "regenerate the fixture"


def test_model_forward_both_heads():
    model = MultiLingualSignModel(ModelConfig(dim=64, transformer_layers=1, vocab_sizes={"ase": 7},
                                              fs_alphabet_sizes={"ase": 26}))
    x, m = torch.randn(3, 40, FEATURE_DIM), torch.ones(3, 40, dtype=torch.bool)
    m[1, 20:] = False
    assert model.classify(x, m, "ase").shape == (3, 7)
    lp = model.spell(x, m, "ase")
    assert lp.shape == (3, 40, 27)
    assert len(ctc_greedy_decode(lp, m)) == 3
    model.add_language("gsg", n_glosses=5)
    assert model.classify(x, m, "gsg").shape == (3, 5)


def test_padding_does_not_change_prediction():
    model = MultiLingualSignModel(ModelConfig(dim=64, transformer_layers=2, vocab_sizes={"ase": 4})).eval()
    feat = torch.from_numpy(preprocess(synthetic_clip(1, np.random.default_rng(1), 30), 64))
    t = len(feat)
    short = model.classify(feat[None], torch.ones(1, t, dtype=torch.bool), "ase")
    x, mask = pad(feat.numpy(), 64)
    long = model.classify(torch.from_numpy(x)[None], torch.from_numpy(mask)[None], "ase")
    assert torch.allclose(short, long, atol=1e-4)


def test_signer_split_is_disjoint(tmp_path):
    prepare_synthetic(tmp_path, classes=3, signers=20, per_signer=1)
    sp = signer_split(read_index(tmp_path))
    signers = [{s.signer for s in sp[k]} for k in ("train", "val", "test")]
    assert not (signers[0] & signers[1]) and not (signers[0] & signers[2]) and not (signers[1] & signers[2])
    ds = SignDataset(sp["train"], [f"SIGN{i:02d}" for i in range(3)], 64, train=True)
    x, m, y = ds[0]
    assert x.shape == (64, FEATURE_DIM) and m.dtype == torch.bool and 0 <= y < 3


@pytest.mark.slow
def test_train_and_export_end_to_end(tmp_path):
    data, run, models = tmp_path / "data", tmp_path / "run", tmp_path / "models"
    prepare_synthetic(data, classes=6, signers=40, per_signer=2)
    py = [sys.executable, "-m"]
    subprocess.run(py + ["trisign_ml.train", "--signs", f"ase={data}", "--out", str(run), "--epochs", "15",
                         "--batch", "16", "--dim", "64", "--layers", "2", "--workers", "0", "--lr", "3e-3"],
                   check=True, cwd=ROOT / "ml")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test"]["ase/sign"]["n"] > 0
    assert metrics["test"]["ase/sign"]["top1"] > 0.6
    subprocess.run(py + ["trisign_ml.export_onnx", str(run / "best.pt"), "--out", str(models)],
                   check=True, cwd=ROOT / "ml")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(models / "ase" / "signs.onnx"))
    feat = preprocess(synthetic_clip(2, np.random.default_rng(3), 40), 64)
    x, mask = pad(feat, 64)
    probs = sess.run(None, {"features": x[None], "mask": mask[None]})[0]
    from trisign_ml.export_onnx import load
    model, _ = load(run / "best.pt")
    ref = model.classify(torch.from_numpy(x)[None], torch.from_numpy(mask)[None], "ase").softmax(-1).detach().numpy()
    assert np.abs(probs - ref).max() < 0.05, "int8 model drifted from PyTorch"
    assert probs.shape == (1, 6) and np.isclose(probs.sum(), 1, atol=1e-3)


def test_fingerspelling_training_step(tmp_path):
    from trisign_ml.prepare import IndexWriter
    rng = np.random.default_rng(0)
    w = IndexWriter(tmp_path / "spell", "spell")
    for i in range(24):
        w.add(synthetic_clip(i % 5, rng, 50), ["ANNA", "BERLIN", "MADRID", "NYC", "LUIS"][i % 5], f"s{i}")
    w.close()
    subprocess.run([sys.executable, "-m", "trisign_ml.train", "--spell", f"ase={tmp_path / 'spell'}",
                    "--out", str(tmp_path / "run"), "--epochs", "1", "--batch", "8", "--dim", "32",
                    "--layers", "1", "--workers", "0"], check=True, cwd=ROOT / "ml")
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert "cer" in metrics["history"][0]["val"]["ase/spell"]
