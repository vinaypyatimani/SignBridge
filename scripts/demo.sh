#!/usr/bin/env bash
# Builds a SYNTHETIC demo so the whole app runs before you have real data:
# a tiny ASL model whose "signs" are named SIGN00..SIGN11 and a matching sign
# library. It proves the pipeline end to end; it does not recognize real signs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=ml
PY=${PYTHON:-python3}

$PY -m trisign_ml.prepare synthetic --out data/synthetic/ase --classes 12 --signers 30 --per-signer 3
$PY -m trisign_ml.train --signs ase=data/synthetic/ase --out runs/synthetic --epochs 12 --batch 32 \
  --dim 96 --layers 2 --workers 0 --lr 3e-3
$PY -m trisign_ml.export_onnx runs/synthetic/best.pt --out web/public/models
$PY -m trisign_ml.build_sign_library --lang ase --data data/synthetic/ase \
  --out backend/data/sign_library/ase.json
echo
echo "Demo ready. Start the backend (make backend) and the web app (make web), then open http://localhost:3000."
echo "Type SIGN03 SIGN07 on the speaker side to see the sign player; real signs need real data (see README)."
