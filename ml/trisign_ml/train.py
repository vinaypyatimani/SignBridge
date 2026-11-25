"""Train the multilingual recognizer.

Phase 1, ASL only:
    python -m trisign_ml.train --signs ase=data/ase/signs --spell ase=data/ase/spell --out runs/ase

Phase 4, add DGS and LSE on top of the ASL encoder:
    python -m trisign_ml.train --init runs/ase/best.pt --freeze-layers 2 \
        --signs ase=data/ase/signs --signs gsg=data/gsg/signs --signs ssp=data/ssp/signs \
        --out runs/all

Each batch holds one task (language x signs|spell); tasks are interleaved in
proportion to their size. Validation and test splits are signer-independent.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (SignDataset, SpellDataset, build_alphabet, build_vocab, collate_spell, read_index,
                   save_json, signer_split)
from .model import ModelConfig, MultiLingualSignModel, ctc_greedy_decode
from .spec import FS_MAX_FRAMES, MAX_FRAMES


def parse_pairs(items: list[str]) -> dict[str, list[Path]]:
    """LANG=DIR[,DIR...]; repeating a language adds more directories to it."""
    out: dict[str, list[Path]] = {}
    for it in items or []:
        lang, _, paths = it.partition("=")
        if not paths:
            raise SystemExit(f"expected LANG=DIR[,DIR...], got {it!r}")
        out.setdefault(lang, []).extend(Path(p) for p in paths.split(","))
    return out


def read_all(dirs: list[Path], kind: str):
    return [s for d in dirs for s in read_index(d, kind)]


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


class Task:
    def __init__(self, lang: str, kind: str, splits: dict, vocab: list[str], args):
        self.lang, self.kind, self.vocab = lang, kind, vocab
        cls, frames = (SignDataset, MAX_FRAMES) if kind == "sign" else (SpellDataset, FS_MAX_FRAMES)
        collate = None if kind == "sign" else collate_spell
        self.loaders = {
            name: DataLoader(cls(splits[name], vocab, frames, train=name == "train", seed=args.seed),
                             batch_size=args.batch, shuffle=name == "train", num_workers=args.workers,
                             persistent_workers=args.workers > 0,  # avoids re-spawning workers each epoch on Windows
                             collate_fn=collate, drop_last=name == "train" and len(splits[name]) > args.batch)
            for name in ("train", "val", "test")
        }

    @property
    def name(self) -> str:
        return f"{self.lang}/{self.kind}"


def step_loss(model, task: Task, batch, device, ctc: nn.CTCLoss, smoothing: float):
    if task.kind == "sign":
        x, m, y = (t.to(device) for t in batch)
        return nn.functional.cross_entropy(model.classify(x, m, task.lang), y, label_smoothing=smoothing)
    x, m, tgt, tgt_len = (t.to(device) for t in batch)
    lp = model.spell(x, m, task.lang).transpose(0, 1)  # (T, B, C)
    return ctc(lp, tgt, m.sum(1), tgt_len)


@torch.no_grad()
def evaluate(model, task: Task, split: str, device) -> dict:
    model.eval()
    if task.kind == "sign":
        n = top1 = top5 = 0
        for x, m, y in task.loaders[split]:
            logits = model.classify(x.to(device), m.to(device), task.lang)
            k = min(5, logits.shape[1])
            pred = logits.topk(k, -1).indices.cpu()
            top1 += (pred[:, 0] == y).sum().item()
            top5 += (pred == y[:, None]).any(1).sum().item()
            n += len(y)
        return {"n": n, "top1": top1 / max(n, 1), "top5": top5 / max(n, 1)}
    errs = chars = 0
    for x, m, tgt, tgt_len in task.loaders[split]:
        decoded = ctc_greedy_decode(model.spell(x.to(device), m.to(device), task.lang).cpu(), m)
        refs = torch.split(tgt, tgt_len.tolist())
        for hyp, ref in zip(decoded, refs):
            errs += edit_distance(hyp, ref.tolist())
            chars += len(ref)
    cer = errs / max(chars, 1)
    return {"n": chars, "cer": cer, "score": 1 - cer}


def score(metrics: dict) -> float:
    return metrics.get("top1", metrics.get("score", 0.0))


def build_model(args, vocabs: dict, alphabets: dict) -> MultiLingualSignModel:
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**{**ckpt["config"], "vocab_sizes": {}, "fs_alphabet_sizes": {}})
        model = MultiLingualSignModel(cfg)
        for lang, v in vocabs.items():
            model.add_language(lang, n_glosses=len(v))
        for lang, a in alphabets.items():
            model.add_language(lang, n_fs_chars=len(a))
        own = model.state_dict()
        keep = {k: v for k, v in ckpt["model"].items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"initialised {len(keep)}/{len(own)} tensors from {args.init}")
    else:
        model = MultiLingualSignModel(ModelConfig(
            dim=args.dim, transformer_layers=args.layers, dropout=args.dropout,
            vocab_sizes={l: len(v) for l, v in vocabs.items()},
            fs_alphabet_sizes={l: len(a) for l, a in alphabets.items()}))
    if args.freeze_layers:
        model.freeze_encoder(args.freeze_layers)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signs", action="append", help="LANG=DIR[,DIR...] with isolated-sign index.csv files")
    ap.add_argument("--spell", action="append", help="LANG=DIR[,DIR...] with fingerspelling index.csv files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", type=Path, help="checkpoint to start from (keeps the encoder)")
    ap.add_argument("--freeze-layers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--min-count", type=int, default=1, help="drop glosses with fewer training clips")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    signs, spell = parse_pairs(args.signs), parse_pairs(args.spell)
    if not signs and not spell:
        raise SystemExit("give at least one --signs or --spell dataset")

    vocabs, alphabets, splits = {}, {}, {}
    for lang, dirs in signs.items():
        sp = signer_split(read_all(dirs, "sign"))
        counts: dict[str, int] = {}
        for s in sp["train"]:
            counts[s.target] = counts.get(s.target, 0) + 1
        vocabs[lang] = [g for g in build_vocab(sp["train"]) if counts[g] >= args.min_count]
        splits[(lang, "sign")] = sp
    for lang, dirs in spell.items():
        sp = signer_split(read_all(dirs, "spell"))
        alphabets[lang] = build_alphabet(sp["train"])
        splits[(lang, "spell")] = sp
    for (lang, kind), sp in splits.items():
        print(f"{lang}/{kind}: " + ", ".join(f"{k} {len(v)} clips / {len({s.signer for s in v})} signers"
                                              for k, v in sp.items()))

    tasks = [Task(lang, kind, sp, vocabs[lang] if kind == "sign" else alphabets[lang], args)
             for (lang, kind), sp in splits.items()]
    device = torch.device(args.device)
    model = build_model(args, vocabs, alphabets).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f} M parameters on {device}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = sum(len(t.loaders["train"]) for t in tasks)
    total = max(1, args.epochs * steps_per_epoch)
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    meta = {"vocabs": vocabs, "alphabets": alphabets, "args": {k: str(v) for k, v in vars(args).items()}}
    save_json(meta, args.out / "meta.json")
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        iters = {t.name: iter(t.loaders["train"]) for t in tasks}
        schedule = [t for t in tasks for _ in range(len(t.loaders["train"]))]
        random.shuffle(schedule)
        for task in schedule:
            batch = next(iters[task.name])
            loss = step_loss(model, task, batch, device, ctc, args.smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())
        val = {t.name: evaluate(model, t, "val", device) for t in tasks}
        mean = float(np.mean([score(v) for v in val.values()]))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None, "val": val})
        summary = "  ".join(f"{k} " + " ".join(f"{m}={v:.3f}" for m, v in r.items() if m != "n")
                            for k, r in val.items())
        print(f"epoch {epoch:3d}  loss {history[-1]['loss']:.3f}  {summary}  ({time.time() - t0:.0f}s)")
        ckpt = {"model": model.state_dict(), "config": model.config_dict(), **meta, "epoch": epoch, "val": val}
        torch.save(ckpt, args.out / "last.pt")
        if mean > best:
            best = mean
            torch.save(ckpt, args.out / "best.pt")

    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device, weights_only=False)["model"])
    test = {t.name: evaluate(model, t, "test", device) for t in tasks}
    save_json({"history": history, "test": test}, args.out / "metrics.json")
    print("test (unseen signers):", json.dumps(test))


if __name__ == "__main__":
    main()
