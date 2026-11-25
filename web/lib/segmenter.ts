// Cuts the live frame stream into sign-sized windows and utterances.
//
// A sign starts when a hand is raised into signing space, and ends when the
// hands drop or rest, when the hands hold still briefly after moving, or when
// the window is full. An utterance ends after about a second of rest. These
// heuristics are tunable; a learned boundary model can replace them later.
import { MAX_FRAMES, SPEC, type HolisticFrame } from "./preprocess";

const POSE = SPEC.layout.pose;
const WRISTS = [POSE + 15, POSE + 16];
const SHOULDERS = [POSE + 11, POSE + 12];
const HIPS = [POSE + 23, POSE + 24];

export const SEGMENT = {
  startFrames: 3,       // active frames needed to start a sign
  endRestFrames: 6,     // inactive frames that end a sign (~200 ms at 30 fps)
  holdFrames: 5,        // still frames that end a sign once it has moved
  holdSpeed: 0.015,     // shoulder widths per frame below which hands are "still"
  minSignFrames: 8,
  utteranceRestFrames: 30,
  leadIn: 2,
};

function xy(f: HolisticFrame, i: number): [number, number] { return [f[i * 3], f[i * 3 + 1]]; }

export function handState(f: HolisticFrame, prev: HolisticFrame | null) {
  const [lsx, lsy] = xy(f, SHOULDERS[0]);
  const [rsx, rsy] = xy(f, SHOULDERS[1]);
  const width = Math.hypot(lsx - rsx, lsy - rsy) || 0.25;
  const hipY = Math.min(xy(f, HIPS[0])[1], xy(f, HIPS[1])[1]);
  const restLine = Number.isFinite(hipY) ? hipY - 0.15 * width : (lsy + rsy) / 2 + 1.6 * width;
  let active = false;
  let speed = 0;
  for (const w of WRISTS) {
    const [x, y] = xy(f, w);
    const handVisible = Number.isFinite(f[(w === WRISTS[0] ? SPEC.layout.left_hand : SPEC.layout.right_hand) * 3]);
    if (handVisible && Number.isFinite(y) && (y < restLine || !Number.isFinite(restLine))) active = true;
    if (prev) {
      const [px, py] = xy(prev, w);
      const s = Math.hypot(x - px, y - py) / width;
      if (Number.isFinite(s)) speed = Math.max(speed, s);
    }
  }
  return { active, speed };
}

export type SegmentEvent = { kind: "sign"; frames: HolisticFrame[] } | { kind: "utterance_end" };

export class Segmenter {
  private recent: HolisticFrame[] = [];
  private current: HolisticFrame[] | null = null;
  private activeRun = 0;
  private restRun = 0;
  private stillRun = 0;
  private moved = false;
  private sinceSign = Infinity;
  private prev: HolisticFrame | null = null;

  constructor(private maxFrames = MAX_FRAMES, private config = SEGMENT) {}

  get signing() { return this.current !== null; }

  push(frame: HolisticFrame): SegmentEvent[] {
    const c = this.config;
    const events: SegmentEvent[] = [];
    const { active, speed } = handState(frame, this.prev);
    this.prev = frame;
    this.recent.push(frame);
    if (this.recent.length > c.leadIn + c.startFrames) this.recent.shift();

    if (!this.current) {
      this.activeRun = active ? this.activeRun + 1 : 0;
      this.sinceSign++;
      if (this.activeRun >= c.startFrames) {
        this.current = [...this.recent];
        this.restRun = this.stillRun = 0;
        this.moved = false;
      } else if (this.sinceSign === c.utteranceRestFrames) {
        events.push({ kind: "utterance_end" });
      }
      return events;
    }

    this.current.push(frame);
    this.restRun = active ? 0 : this.restRun + 1;
    if (speed > c.holdSpeed * 3) this.moved = true;
    this.stillRun = speed < c.holdSpeed ? this.stillRun + 1 : 0;
    const len = this.current.length;
    const ended = this.restRun >= c.endRestFrames
      || (this.moved && this.stillRun >= c.holdFrames && len >= c.minSignFrames)
      || len >= this.maxFrames;
    if (ended) {
      const frames = this.restRun ? this.current.slice(0, len - this.restRun) : this.current;
      if (frames.length >= c.minSignFrames) events.push({ kind: "sign", frames });
      this.current = null;
      this.activeRun = 0;
      this.sinceSign = 0;
    }
    return events;
  }

  reset() {
    this.recent = [];
    this.current = null;
    this.activeRun = this.restRun = this.stillRun = 0;
    this.sinceSign = Infinity;
    this.prev = null;
  }
}
