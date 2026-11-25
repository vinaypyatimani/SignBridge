"use client";
// Camera -> MediaPipe Holistic -> segmenter -> in-browser recognizer.
// Video never leaves this component: only candidate glosses are passed up.
import { useEffect, useRef, useState } from "react";
import type { HolisticLandmarker } from "@mediapipe/tasks-vision";
import { createHolistic, detect } from "@/lib/holistic";
import { FS_MAX_FRAMES, SPEC, type HolisticFrame } from "@/lib/preprocess";
import { Recognizer, type Candidate } from "@/lib/recognizer";
import { Segmenter } from "@/lib/segmenter";
import type { SignLang } from "@/lib/api";
import type { Strings } from "@/lib/i18n";

type Props = {
  signLang: SignLang;
  leftHanded: boolean;
  spelling: boolean;
  t: Strings;
  onSign: (candidates: Candidate[]) => void;
  onSpelled: (word: string) => void;
  onUtteranceEnd: () => void;
};

const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

export default function SignerCamera({ signLang, leftHanded, spelling, t, onSign, onSpelled, onUtteranceEnd }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [on, setOn] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [live, setLive] = useState<Candidate[]>([]);
  const [signing, setSigning] = useState(false);
  const landmarker = useRef<HolisticLandmarker | null>(null);
  const recognizer = useRef<Recognizer | null>(null);
  const segmenter = useRef(new Segmenter());
  const spellBuffer = useRef<HolisticFrame[]>([]);
  const spellRest = useRef(0);
  // Keep the latest props in refs so the frame loop never restarts.
  const latest = useRef({ leftHanded, spelling, onSign, onSpelled, onUtteranceEnd });
  latest.current = { leftHanded, spelling, onSign, onSpelled, onUtteranceEnd };

  useEffect(() => {
    let cancelled = false;
    recognizer.current = null;
    Recognizer.load(signLang)
      .then((r) => {
        if (cancelled) return;
        recognizer.current = r;
        setStatus(r.canSign ? null : t.noModel);
      })
      .catch((e) => !cancelled && setStatus(`Recognition model failed to load: ${e instanceof Error ? e.message : e}`));
    segmenter.current.reset();
    return () => { cancelled = true; };
  }, [signLang, t.noModel]);

  async function start() {
    setStatus(t.loading);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720, facingMode: "user" }, audio: false });
      video.current!.srcObject = stream;
      await video.current!.play();
      landmarker.current ??= await createHolistic();
      setOn(true);
      setStatus((s) => (s === t.loading ? null : s));
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
    }
  }

  useEffect(() => {
    if (!on) return;
    let raf = 0;
    let lastTs = -1;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const v = video.current, lm = landmarker.current;
      if (!v || !lm || v.readyState < 2) return;
      const ts = performance.now();
      if (ts - lastTs < 30) return; // ~30 fps, matching the training data
      lastTs = ts;
      const frame = detect(lm, v, ts);
      if (!frame) return;
      draw(frame);
      handle(frame);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [on]);

  function handle(frame: HolisticFrame) {
    const { leftHanded, spelling, onSign, onSpelled, onUtteranceEnd } = latest.current;
    const rec = recognizer.current;
    if (spelling) {
      // Collect until the hands rest for half a second, then decode the whole word.
      const hands = [...SPEC.groups.left_hand, ...SPEC.groups.right_hand];
      const visible = hands.some((p) => Number.isFinite(frame[SPEC.indices[p] * 3]));
      if (visible) { spellBuffer.current.push(frame); spellRest.current = 0; }
      else if (spellBuffer.current.length) spellRest.current++;
      setSigning(visible);
      if (spellBuffer.current.length >= FS_MAX_FRAMES || (spellRest.current > 15 && spellBuffer.current.length > 8)) {
        const frames = spellBuffer.current.splice(0);
        rec?.spell(frames, leftHanded).then((w) => w && onSpelled(w));
      }
      return;
    }
    for (const ev of segmenter.current.push(frame)) {
      if (ev.kind === "utterance_end") onUtteranceEnd();
      else if (rec?.canSign) rec.classify(ev.frames, leftHanded).then((c) => { setLive(c); if (c.length) onSign(c); });
    }
    setSigning(segmenter.current.signing);
  }

  function draw(frame: HolisticFrame) {
    const c = canvas.current, v = video.current;
    if (!c || !v) return;
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    const color = getComputedStyle(c).getPropertyValue("--signer").trim() || "#00796b";
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    for (const off of [SPEC.layout.left_hand, SPEC.layout.right_hand]) {
      if (!Number.isFinite(frame[off * 3])) continue;
      const p = (i: number) => [frame[(off + i) * 3] * c.width, frame[(off + i) * 3 + 1] * c.height];
      ctx.beginPath();
      for (const [a, b] of HAND_EDGES) { const [ax, ay] = p(a), [bx, by] = p(b); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); }
      ctx.stroke();
    }
  }

  return (
    <div className="camera">
      <div className={`camera-frame ${signing ? "is-signing" : ""}`}>
        <video ref={video} playsInline muted className="mirrored" />
        <canvas ref={canvas} className="mirrored overlay" aria-hidden />
        {!on && (
          <div className="camera-empty">
            <p>{t.cameraOff}</p>
            <button className="button signer" onClick={start}>{t.startCamera}</button>
          </div>
        )}
      </div>
      {status && <p className="status" role="status">{status}</p>}
      {live.length > 0 && !spelling && (
        <ol className="candidates" aria-label="Top guesses for the last sign">
          {live.slice(0, 3).map(([g, p]) => (
            <li key={g}><span className="gloss">{g}</span> <span className="prob">{Math.round(p * 100)}%</span></li>
          ))}
        </ol>
      )}
    </div>
  );
}
