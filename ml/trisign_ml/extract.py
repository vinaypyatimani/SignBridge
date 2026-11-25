"""Extract Holistic-layout landmarks from video with MediaPipe Tasks (Python).

The Python package has no Holistic task, so this runs the Pose, Hand and Face
landmarkers and assembles the same (543, 3) layout the browser's
HolisticLandmarker and the Kaggle datasets use. Hands are assigned to the
signer's left/right by distance to the pose wrists, which matches Holistic's
convention and avoids the mirrored-handedness quirk of HandLandmarker.

Model files: run ``scripts/download_mediapipe_models.sh`` first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .spec import FACE0, LHAND0, N_HOLISTIC, POSE0, RHAND0

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "mediapipe"


class HolisticExtractor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        mode = vision.RunningMode.VIDEO
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / "pose_landmarker_full.task")),
            running_mode=mode, num_poses=1))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / "hand_landmarker.task")),
            running_mode=mode, num_hands=2))
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / "face_landmarker.task")),
            running_mode=mode, num_faces=1))
        self._clock = 0  # VIDEO mode needs strictly increasing timestamps across all videos

    def frame(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray:
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        out = np.full((N_HOLISTIC, 3), np.nan, dtype=np.float32)
        pose = self.pose.detect_for_video(img, ts_ms)
        if pose.pose_landmarks:
            out[POSE0:POSE0 + 33] = [[p.x, p.y, p.z] for p in pose.pose_landmarks[0]]
        face = self.face.detect_for_video(img, ts_ms)
        if face.face_landmarks:
            out[FACE0:FACE0 + 468] = [[p.x, p.y, p.z] for p in face.face_landmarks[0][:468]]
        hands = self.hands.detect_for_video(img, ts_ms)
        for hand, handed in zip(hands.hand_landmarks, hands.handedness):
            pts = np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32)
            slot = self._slot(out, pts[0, :2], handed[0].category_name)
            if np.isnan(out[slot, 0]):
                out[slot:slot + 21] = pts
        return out

    @staticmethod
    def _slot(out: np.ndarray, wrist_xy: np.ndarray, label: str) -> int:
        lw, rw = out[POSE0 + 15, :2], out[POSE0 + 16, :2]  # signer's left / right wrist
        if np.isfinite(lw).all() and np.isfinite(rw).all():
            return LHAND0 if np.linalg.norm(wrist_xy - lw) <= np.linalg.norm(wrist_xy - rw) else RHAND0
        # No pose: HandLandmarker labels assume a mirrored image, so "Right" = signer's left.
        return LHAND0 if label == "Right" else RHAND0

    def video(self, path: str | Path, start_s: float | None = None, end_s: float | None = None,
              max_fps: float = 30.0) -> np.ndarray:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps / max_fps))
        if start_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
        frames, i = [], 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if end_s is not None and t_ms > end_s * 1000:
                break
            if i % step == 0:
                self._clock += max(1, int(1000 / max_fps))
                frames.append(self.frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self._clock))
            i += 1
        cap.release()
        return np.stack(frames) if frames else np.zeros((0, N_HOLISTIC, 3), np.float32)

    def close(self) -> None:
        for task in (self.pose, self.hands, self.face):
            task.close()
