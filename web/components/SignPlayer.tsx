"use client";
// Plays a sign sequence from the library as a skeleton (v1 stand-in for the 3D
// avatar). Frames are normalized: shoulders centred at (0, 0), shoulder width 1.
import { useEffect, useRef, useState } from "react";
import type { SequenceItem, SignSequence } from "@/lib/api";
import { SPEC } from "@/lib/preprocess";

const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
// Pose group order is pose points 11..22: 0 L shoulder, 1 R shoulder, 2 L elbow, 3 R elbow, 4 L wrist, 5 R wrist.
const ARM_EDGES = [[0, 1], [0, 2], [2, 4], [1, 3], [3, 5]];
const BLEND = 4;
// Outer lip contour as a closed loop over the spec's lip order: upper lip 0..10 (corner to corner), lower lip 19..11.
const LIP_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 18, 17, 16, 15, 14, 13, 12, 11];
const LIP_LOOP = LIP_ORDER.map((p, i) => [p, LIP_ORDER[(i + 1) % LIP_ORDER.length]]);

type Step = { label: string; frames?: number[][][]; letter?: string; text?: string; ms: number };

function steps(items: SequenceItem[], fps: number): Step[] {
  const out: Step[] = [];
  for (const it of items) {
    if (it.type === "sign") out.push({ label: it.gloss, frames: it.frames, ms: (it.frames.length / fps) * 1000 });
    else if (it.type === "spell")
      for (const l of it.letters)
        out.push(l.frames ? { label: l.letter, frames: l.frames, ms: (l.frames.length / fps) * 1000 }
                          : { label: l.letter, letter: l.letter, ms: 650 });
    else out.push({ label: it.text, text: it.text, ms: 1400 });
  }
  return out;
}

function lerp(a: number[][], b: number[][], w: number) {
  return a.map((p, i) => [p[0] * (1 - w) + b[i][0] * w, p[1] * (1 - w) + b[i][1] * w]);
}

export default function SignPlayer({ sequence, speed, emptyText }: { sequence: SignSequence | null; speed: number; emptyText: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [label, setLabel] = useState("");
  const [replay, setReplay] = useState(0);

  useEffect(() => {
    if (!sequence) return;
    const plan = steps(sequence.items, sequence.fps * speed);
    const fps = sequence.fps * speed;
    let raf = 0;
    let i = 0;
    let stepStart = performance.now();
    let last: number[][] | null = null;
    const tick = () => {
      const now = performance.now();
      const c = canvas.current;
      if (!c || i >= plan.length) { setLabel(""); return; }
      const step = plan[i];
      const elapsed = Math.max(0, now - stepStart);
      setLabel(step.label);
      const ctx = c.getContext("2d")!;
      const css = getComputedStyle(c);
      ctx.clearRect(0, 0, c.width, c.height);
      if (step.frames) {
        const k = Math.floor((elapsed / 1000) * fps);
        let f = step.frames[Math.min(step.frames.length - 1, k)];
        if (last && k < BLEND) f = lerp(last, f, (k + 1) / (BLEND + 1)); // smooth into the next sign
        drawSkeleton(ctx, f, c.width, c.height, css);
      } else {
        ctx.fillStyle = css.getPropertyValue("--speaker").trim();
        ctx.font = `700 ${step.letter ? c.height * 0.45 : c.height * 0.12}px "Atkinson Hyperlegible", sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(step.letter ?? step.text ?? "", c.width / 2, c.height / 2, c.width * 0.9);
      }
      if (elapsed >= step.ms) {
        if (step.frames) last = step.frames[step.frames.length - 1];
        i++;
        stepStart = now;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [sequence, speed, replay]);

  return (
    <figure className="player">
      <canvas ref={canvas} width={480} height={360} aria-label={sequence ? `Signing: ${sequence.glosses.join(" ")}` : emptyText} />
      <figcaption>
        {sequence ? (
          <>
            <span className="player-gloss">{label || sequence.glosses.join(" ")}</span>
            <span className="player-text">{sequence.text}</span>
            <button className="link" onClick={() => setReplay((r) => r + 1)}>Replay</button>
          </>
        ) : <span className="player-text muted">{emptyText}</span>}
      </figcaption>
    </figure>
  );
}

function drawSkeleton(ctx: CanvasRenderingContext2D, f: number[][], w: number, h: number, css: CSSStyleDeclaration) {
  const s = w / 5, cx = w / 2, cy = h * 0.5; // fits hands raised above the head and down to the waist
  const P = (k: number) => [cx + f[k][0] * s, cy + f[k][1] * s];
  const g = SPEC.groups;
  const line = (pts: number[], edges: number[][]) => {
    ctx.beginPath();
    for (const [a, b] of edges) {
      const [ax, ay] = P(pts[a]), [bx, by] = P(pts[b]);
      if ([ax, ay, bx, by].every(Number.isFinite) && (f[pts[a]][0] || f[pts[a]][1]) && (f[pts[b]][0] || f[pts[b]][1])) {
        ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
      }
    }
    ctx.stroke();
  };
  ctx.lineCap = "round";
  ctx.strokeStyle = css.getPropertyValue("--muted").trim();
  ctx.lineWidth = 7;
  line(g.pose, ARM_EDGES);
  // Head: circle around the lips and brows.
  const face = [...g.lips, ...g.left_brow, ...g.right_brow].map(P);
  const hx = face.reduce((a, p) => a + p[0], 0) / face.length;
  const hy = face.reduce((a, p) => a + p[1], 0) / face.length;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(hx, hy - s * 0.08, s * 0.42, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = css.getPropertyValue("--ink").trim();
  ctx.lineWidth = 2;
  line(g.lips, LIP_LOOP);
  line(g.left_brow, [[0, 1], [1, 2], [2, 3], [3, 4]]);
  line(g.right_brow, [[0, 1], [1, 2], [2, 3], [3, 4]]);
  ctx.strokeStyle = css.getPropertyValue("--speaker").trim();
  ctx.lineWidth = 4;
  line(g.left_hand, HAND_EDGES);
  line(g.right_hand, HAND_EDGES);
}
