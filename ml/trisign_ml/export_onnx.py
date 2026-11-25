"""Export one ONNX model per language (and task) for ONNX Runtime Web.

    python -m trisign_ml.export_onnx runs/ase/best.pt --out ../web/public/models

Writes, per language:
    <out>/<lang>/signs.onnx      inputs: features (1, 64, F) float32, mask (1, 64) bool -> probs (1, n_glosses)
    <out>/<lang>/signs.json      {"labels": [...]}
    <out>/<lang>/spell.onnx      features (1, 192, F), mask (1, 192) -> log_probs (1, 192, n_chars + 1), blank = 0
    <out>/<lang>/spell.json      {"alphabet": [...]}
plus <out>/landmark_spec.json, so the browser always preprocesses like training did.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .model import ModelConfig, MultiLingualSignModel
from .spec import FEATURE_DIM, SPEC


class SignHead(nn.Module):
    def __init__(self, model: MultiLingualSignModel, lang: str):
        super().__init__()
        self.model, self.lang = model, lang

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model.classify(features, mask, self.lang).softmax(-1)


class SpellHead(SignHead):
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model.spell(features, mask, self.lang)


def load(ckpt_path: Path) -> tuple[MultiLingualSignModel, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MultiLingualSignModel(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


def export_one(module: nn.Module, path: Path, frames: int, output: str, quantize: bool) -> Path:
    x = torch.zeros(1, frames, FEATURE_DIM)
    m = torch.ones(1, frames, dtype=torch.bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Fixed window: the browser always pads to ``frames`` and passes a mask.
    torch.onnx.export(module, (x, m), str(path), input_names=["features", "mask"], output_names=[output],
                      opset_version=17, dynamo=False)
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        tmp = path.with_suffix(".fp32.onnx")
        path.replace(tmp)
        quantize_dynamic(str(tmp), str(path), weight_type=QuantType.QInt8)
        tmp.unlink()
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-quantize", action="store_true")
    a = ap.parse_args()

    model, ckpt = load(a.checkpoint)
    q = not a.no_quantize
    for lang, labels in ckpt["vocabs"].items():
        p = export_one(SignHead(model, lang), a.out / lang / "signs.onnx", SPEC["max_frames"], "probs", q)
        (a.out / lang / "signs.json").write_text(json.dumps({"labels": labels}, ensure_ascii=False), encoding="utf-8")
        print(f"{lang}: {p} ({p.stat().st_size / 1e6:.1f} MB, {len(labels)} glosses)")
    for lang, alphabet in ckpt["alphabets"].items():
        p = export_one(SpellHead(model, lang), a.out / lang / "spell.onnx", SPEC["fs_max_frames"], "log_probs", q)
        (a.out / lang / "spell.json").write_text(json.dumps({"alphabet": alphabet}, ensure_ascii=False), encoding="utf-8")
        print(f"{lang}: {p} ({p.stat().st_size / 1e6:.1f} MB, {len(alphabet)} characters)")
    (a.out / "landmark_spec.json").write_text(json.dumps(SPEC), encoding="utf-8")


if __name__ == "__main__":
    main()
