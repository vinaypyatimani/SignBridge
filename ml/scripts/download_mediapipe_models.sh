#!/usr/bin/env bash
# Downloads the MediaPipe models used for video extraction (Python) and live
# recognition (browser). Run from the repository root.
set -euo pipefail
base=https://storage.googleapis.com/mediapipe-models

mkdir -p ml/models/mediapipe web/public/mediapipe
curl -fL -o ml/models/mediapipe/pose_landmarker_full.task "$base/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
curl -fL -o ml/models/mediapipe/hand_landmarker.task     "$base/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
curl -fL -o ml/models/mediapipe/face_landmarker.task     "$base/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
curl -fL -o web/public/mediapipe/holistic_landmarker.task "$base/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
echo "MediaPipe models downloaded."
