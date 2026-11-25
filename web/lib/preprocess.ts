// Line-by-line port of ml/trisign_ml/preprocess.py. Keep the two in sync:
// lib/preprocess.test.ts checks this file against shared/preprocess_fixture.json.
import spec from "./generated/landmark_spec.json";

export const N_HOLISTIC: number = spec.n_holistic; // 543
export const N_POINTS: number = spec.indices.length; // 104
export const FEATURE_DIM = N_POINTS * 4; // (x, y) positions + velocities
export const MAX_FRAMES: number = spec.max_frames; // 64
export const FS_MAX_FRAMES: number = spec.fs_max_frames; // 192
export const SPEC = spec;

const EPS = 1e-6;
const IDX: number[] = spec.indices;
const MIRROR: number[] = spec.mirror;
const HANDS: number[] = [...spec.groups.left_hand, ...spec.groups.right_hand];
const LS: number = spec.left_shoulder;
const RS: number = spec.right_shoulder;

/** One frame: 543 points x (x, y, z), NaN where MediaPipe found nothing. */
export type HolisticFrame = Float32Array;
/** Selected points of one frame: [x0, y0, x1, y1, ...], length N_POINTS * 2. */
type XY = Float64Array;

export function select(frames: HolisticFrame[]): XY[] {
  return frames.map((f) => {
    const out = new Float64Array(N_POINTS * 2);
    for (let k = 0; k < N_POINTS; k++) {
      out[2 * k] = f[IDX[k] * 3];
      out[2 * k + 1] = f[IDX[k] * 3 + 1];
    }
    return out;
  });
}

export function dropHandless(xy: XY[]): XY[] {
  const keep = xy.filter((f) => HANDS.some((p) => !Number.isNaN(f[2 * p])));
  return keep.length ? keep : xy;
}

export function resample<T>(xy: T[], maxFrames: number): T[] {
  const t = xy.length;
  if (t <= maxFrames) return xy;
  const step = (t - 1) / (maxFrames - 1);
  const out: T[] = [];
  for (let i = 0; i < maxFrames; i++) {
    const v = i === maxFrames - 1 ? t - 1 : i * step; // numpy.linspace sets the end point exactly
    out.push(xy[Math.floor(v)]);
  }
  return out;
}

function nanMean(values: number[]): number {
  let s = 0;
  let n = 0;
  for (const v of values) if (Number.isFinite(v)) { s += v; n++; }
  return n ? s / n : NaN;
}

export function normalize(xy: XY[]): XY[] {
  const midX: number[] = [];
  const midY: number[] = [];
  const width: number[] = [];
  for (const f of xy) {
    const lx = f[2 * LS], ly = f[2 * LS + 1], rx = f[2 * RS], ry = f[2 * RS + 1];
    midX.push((lx + rx) / 2);
    midY.push((ly + ry) / 2);
    width.push(Math.hypot(lx - rx, ly - ry));
  }
  let cx: number, cy: number, scale: number;
  const meanWidth = nanMean(width);
  if (Number.isFinite(meanWidth) && meanWidth > EPS) {
    cx = nanMean(midX);
    cy = nanMean(midY);
    scale = meanWidth;
  } else {
    const px: number[] = [];
    const py: number[] = [];
    for (const f of xy)
      for (let k = 0; k < N_POINTS; k++)
        if (Number.isFinite(f[2 * k]) && Number.isFinite(f[2 * k + 1])) { px.push(f[2 * k]); py.push(f[2 * k + 1]); }
    if (!px.length) return xy;
    cx = px.reduce((a, b) => a + b, 0) / px.length;
    cy = py.reduce((a, b) => a + b, 0) / py.length;
    let ss = 0;
    for (let i = 0; i < px.length; i++) ss += (px[i] - cx) ** 2 + (py[i] - cy) ** 2;
    scale = Math.sqrt(ss / px.length) * 2 + EPS;
  }
  return xy.map((f) => {
    const out = new Float64Array(f.length);
    for (let k = 0; k < N_POINTS; k++) {
      out[2 * k] = (f[2 * k] - cx) / scale;
      out[2 * k + 1] = (f[2 * k + 1] - cy) / scale;
    }
    return out;
  });
}

export function mirror(xy: XY[]): XY[] {
  return xy.map((f) => {
    const out = new Float64Array(f.length);
    for (let k = 0; k < N_POINTS; k++) {
      out[2 * k] = -f[2 * MIRROR[k]];
      out[2 * k + 1] = f[2 * MIRROR[k] + 1];
    }
    return out;
  });
}

export function toFeatures(xy: XY[]): Float32Array[] {
  const d = N_POINTS * 2;
  return xy.map((f, t) => {
    const out = new Float32Array(d * 2);
    for (let j = 0; j < d; j++) {
      const pos = f[j];
      const vel = t === 0 ? 0 : f[j] - xy[t - 1][j];
      out[j] = Number.isNaN(pos) ? 0 : pos;
      out[d + j] = Number.isNaN(vel) ? 0 : vel;
    }
    return out;
  });
}

export function preprocess(frames: HolisticFrame[], maxFrames: number, leftHanded = false, dropEmpty = true) {
  let xy = select(frames);
  if (dropEmpty) xy = dropHandless(xy);
  xy = normalize(resample(xy, maxFrames));
  if (leftHanded) xy = mirror(xy);
  return toFeatures(xy);
}

/** Pack features into the fixed-size model input: (1, maxFrames, F) + mask (1, maxFrames). */
export function pad(features: Float32Array[], maxFrames: number) {
  const t = Math.min(features.length, maxFrames);
  const data = new Float32Array(maxFrames * FEATURE_DIM);
  const mask = new Uint8Array(maxFrames);
  for (let i = 0; i < t; i++) {
    data.set(features[i], i * FEATURE_DIM);
    mask[i] = 1;
  }
  return { data, mask };
}
