// MediaPipe Holistic in the browser -> frames in the (543, 3) training layout.
// Frames go to the landmarker unmirrored; only the preview is mirrored, so
// "left hand" means the signer's left hand, as in the training data.
import type { HolisticLandmarker, HolisticLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";
import { N_HOLISTIC, SPEC, type HolisticFrame } from "./preprocess";

const WASM = "/mediapipe/wasm"; // copied from node_modules by scripts/sync-shared.mjs
const MODEL = "/mediapipe/holistic_landmarker.task";

export async function createHolistic(): Promise<HolisticLandmarker> {
  const { FilesetResolver, HolisticLandmarker } = await import("@mediapipe/tasks-vision");
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const opts = { baseOptions: { modelAssetPath: MODEL, delegate: "GPU" as "GPU" | "CPU" }, runningMode: "VIDEO" as const };
  try {
    return await HolisticLandmarker.createFromOptions(fileset, opts);
  } catch {
    opts.baseOptions.delegate = "CPU";
    return HolisticLandmarker.createFromOptions(fileset, opts);
  }
}

function put(out: Float32Array, offset: number, pts: NormalizedLandmark[] | undefined, limit: number) {
  if (!pts) return;
  for (let i = 0; i < Math.min(limit, pts.length); i++) {
    out[(offset + i) * 3] = pts[i].x;
    out[(offset + i) * 3 + 1] = pts[i].y;
    out[(offset + i) * 3 + 2] = pts[i].z;
  }
}

export function toFrame(r: HolisticLandmarkerResult): HolisticFrame {
  const out = new Float32Array(N_HOLISTIC * 3).fill(NaN);
  put(out, SPEC.layout.face, r.faceLandmarks[0], 468); // drop the 10 iris points
  put(out, SPEC.layout.left_hand, r.leftHandLandmarks[0], 21);
  put(out, SPEC.layout.pose, r.poseLandmarks[0], 33);
  put(out, SPEC.layout.right_hand, r.rightHandLandmarks[0], 21);
  return out;
}

export function detect(lm: HolisticLandmarker, video: HTMLVideoElement, ts: number): HolisticFrame | null {
  let frame: HolisticFrame | null = null;
  lm.detectForVideo(video, ts, (r) => { frame = toFrame(r); });
  return frame;
}
