# TriSign

A two-way conversation aid between a Deaf signer and a hearing speaker, using only a laptop or phone camera.

- **Sign languages:** ASL (`ase`), DGS (`gsg`), LSE (`ssp`)
- **Spoken languages:** English, German, Spanish, in any combination (a DGS signer can talk with a Spanish speaker)
- **Sign → text/speech:** the browser tracks hands, face and body with MediaPipe, recognizes signs with an on-device ONNX model, and an agent turns the glosses into a sentence
- **Speech/text → sign:** Whisper transcribes, the agent glosses the sentence, and a player signs it from a library of verified recordings, fingerspelling anything it has no sign for

Signing video never leaves the device. The server only receives glosses, text and (for speech input) microphone audio.

TriSign is a communication aid, not a replacement for a qualified interpreter.

## Status

This is Phase 1 of the build plan: the whole pipeline works end to end and is tested, but **no real sign recognition model ships with the repository**. You train one from the public datasets below. `make demo` builds a synthetic stand-in (signs named `SIGN00`–`SIGN11`) so you can run the app straight away.

## Quick start on Windows

You need [Python 3.12](https://www.python.org/downloads/) (3.11 also works; tick "Add to PATH") and [Node.js LTS](https://nodejs.org/). Open PowerShell in this folder:

```powershell
.\dev.ps1 install      # creates .venv, installs Python packages and the web app
.\dev.ps1 models       # MediaPipe models (needs internet)
.\dev.ps1 demo         # synthetic model + sign library (a minute or two)
.\dev.ps1 backend      # window 1: API on http://localhost:8000
.\dev.ps1 web          # window 2: app on http://localhost:3000
.\dev.ps1 test         # all tests
```

If PowerShell says running scripts is disabled, allow local scripts once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or run a task as `powershell -ExecutionPolicy Bypass -File .\dev.ps1 install`.

With an NVIDIA GPU, install the CUDA build of PyTorch for faster training: `.\dev.ps1 install -Cuda cu124` (match the CUDA version to your driver; see [pytorch.org](https://pytorch.org/get-started/locally/)).

To use Claude, set the key in the backend window before starting it: `$env:ANTHROPIC_API_KEY = "sk-ant-..."`. Use Chrome or Edge for the app; the camera needs `localhost` or HTTPS.

## Quick start on macOS and Linux

```bash
make install          # Python packages (ml, backend) and the web app
make models           # MediaPipe models (needs internet)
make demo             # synthetic model + sign library
make backend          # terminal 1: API on http://localhost:8000
make web              # terminal 2: app on http://localhost:3000
```

Set `ANTHROPIC_API_KEY` before `make backend` to use Claude for gloss ↔ sentence conversion. Without it the backend uses a rule-based fallback that matches the phrase packs and joins glosses, which is enough to try the app but not for real conversations. For speech input, also run `pip install -e "backend[speech]"`.

## Repository layout

```
shared/     landmark_spec.json: which landmarks the model uses, shared by Python and the browser
ml/         data preparation, training, ONNX export, sign library builder (Python, PyTorch)
backend/    FastAPI + LangGraph agent, phrase packs, sign library, Whisper (Python)
web/        Next.js app: camera, MediaPipe, ONNX Runtime Web, captions, sign player (TypeScript)
scripts/    demo.sh
```

## Training a real model

On Windows, run these in PowerShell after `.\.venv\Scripts\Activate.ps1`, so `python` is the project's environment. The `/` paths work as written; replace `~/data` with your own folder, for example `C:\data`.

Every dataset is converted into one format (`index.csv` + `clips/*.npy` holding MediaPipe Holistic landmarks), then trained together. Validation and test splits never share a signer.

**1. Prepare data**

```bash
cd ml
# ASL: Google Kaggle "Isolated Sign Language Recognition" (250 signs, landmarks included)
python -m trisign_ml.prepare kaggle-islr --src ~/data/asl-signs --out ../data/ase/signs
# ASL fingerspelling: Google Kaggle "ASL Fingerspelling"
python -m trisign_ml.prepare kaggle-fs --src ~/data/asl-fingerspelling --out ../data/ase/spell
# Video datasets (ASL Citizen, DGS Corpus segments, SWL-LSE, your own recordings):
# a CSV with columns video,label,signer and optional start,end in seconds
python -m trisign_ml.prepare videos --src ~/data/asl-citizen/clips.csv --out ../data/ase/citizen
```

To train one language on several sources, list them: `--signs ase=../data/ase/signs,../data/ase/citizen`. Signer ids should be unique across sources (prefix them, e.g. `citizen-P12`).

**2. Train ASL** (Phase 1)

```bash
python -m trisign_ml.train --signs ase=../data/ase/signs --spell ase=../data/ase/spell --out ../runs/ase
```

**3. Add DGS and LSE** on the ASL encoder (Phase 4)

```bash
python -m trisign_ml.train --init ../runs/ase/best.pt --freeze-layers 2 \
  --signs ase=../data/ase/signs --signs gsg=../data/gsg/signs --signs ssp=../data/ssp/signs --out ../runs/all
```

**4. Export for the browser and build the sign library**

```bash
python -m trisign_ml.export_onnx ../runs/all/best.pt --out ../web/public/models
python -m trisign_ml.build_sign_library --lang gsg --data ../data/gsg/signs \
  --verified ../data/gsg/verified.csv --out ../backend/data/sign_library/gsg.json
```

Only put signs a Deaf reviewer has checked into the sign library (`--verified`). Record fingerspelling letters as glosses named `FS:A`, `FS:B` … so names are spelled with real handshapes instead of shown as letters.

## How the pieces fit

| Step | Where | Code |
| --- | --- | --- |
| Landmarks (Holistic, 30 fps) | browser | `web/lib/holistic.ts` |
| Preprocessing (select 104 points, normalize by shoulders, add velocities) | browser and training | `web/lib/preprocess.ts`, `ml/trisign_ml/preprocess.py` |
| Cutting the stream into signs | browser | `web/lib/segmenter.ts` |
| Recognition (shared encoder, one head per language, CTC fingerspelling) | browser (ONNX) | `web/lib/recognizer.ts`, `ml/trisign_ml/model.py` |
| Glosses ↔ sentence, context modes, memory, summary | backend | `backend/app/agent.py`, `backend/app/converter.py` |
| Sign library and fingerspelling fallback | backend | `backend/app/sign_library.py` |
| Speech to text | backend | `backend/app/speech.py` (faster-whisper) |
| Text to speech | browser | Web Speech API |

The Python and TypeScript preprocessing must stay identical. `shared/preprocess_fixture.json` holds frames and expected features from Python; `web/lib/preprocess.test.ts` checks the TypeScript port against it. Regenerate it with `PYTHONPATH=ml python ml/scripts/make_preprocess_fixture.py` after changing `preprocess.py`.

## Tests

```bash
make test
```

- `ml/tests`: spec consistency, normalization, mirroring, signer-independent splits, padding invariance, and a small end-to-end run that trains, exports to int8 ONNX and checks ONNX against PyTorch
- `backend/tests`: phrase packs cover all languages, cross-language captions, the confirm step for low confidence, library validation and fingerspelling fallback, conversation memory
- `web`: Python/TypeScript preprocessing parity

## Before real use

- The phrase-pack glosses in `backend/data/context_packs/` are drafts written without Deaf review. Have signers of each language correct them.
- Check each dataset's licence. Most are research-only; a commercial product needs consented data of your own.
- The segmenter in `web/lib/segmenter.ts` uses hand-tuned thresholds. Tune them with real signers, or replace them with a learned boundary model.
- The sign player is a 2D skeleton. The planned Three.js avatar can replace `web/components/SignPlayer.tsx` using the same sequence data.
