// In-browser sign recognition with ONNX Runtime Web. One model per sign
// language, exported by ml/trisign_ml/export_onnx.py into public/models/<lang>/.
import type { InferenceSession } from "onnxruntime-web";
import { FEATURE_DIM, FS_MAX_FRAMES, MAX_FRAMES, pad, preprocess, type HolisticFrame } from "./preprocess";

export type Candidate = [gloss: string, prob: number];

let ortPromise: Promise<typeof import("onnxruntime-web")> | null = null;
function ort() {
  ortPromise ??= import("onnxruntime-web").then((m) => {
    m.env.wasm.wasmPaths = "/ort/"; // copied from node_modules by scripts/sync-shared.mjs
    return m;
  });
  return ortPromise;
}

async function loadJson<T>(url: string): Promise<T | null> {
  const r = await fetch(url);
  return r.ok ? ((await r.json()) as T) : null;
}

export class Recognizer {
  private constructor(
    readonly lang: string,
    private signs: InferenceSession | null,
    private labels: string[],
    private spellSession: InferenceSession | null,
    private alphabet: string[],
  ) {}

  /** Loads whatever exists for the language; a missing model leaves that feature off (canSign false).
   *  Throws if a model exists but cannot be loaded. */
  static async load(lang: string): Promise<Recognizer> {
    const o = await ort();
    const base = `/models/${lang}`;
    const [signMeta, spellMeta] = await Promise.all([
      loadJson<{ labels: string[] }>(`${base}/signs.json`),
      loadJson<{ alphabet: string[] }>(`${base}/spell.json`),
    ]);
    const opts = { executionProviders: ["wasm"] };
    const signs = signMeta ? await o.InferenceSession.create(`${base}/signs.onnx`, opts) : null;
    const spell = spellMeta ? await o.InferenceSession.create(`${base}/spell.onnx`, opts) : null;
    return new Recognizer(lang, signs, signMeta?.labels ?? [], spell, spellMeta?.alphabet ?? []);
  }

  get canSign() { return this.signs !== null; }
  get canSpell() { return this.spellSession !== null; }
  get vocabularySize() { return this.labels.length; }

  private async run(session: InferenceSession, frames: HolisticFrame[], window: number, leftHanded: boolean,
                    dropEmpty: boolean) {
    const o = await ort();
    const { data, mask } = pad(preprocess(frames, window, leftHanded, dropEmpty), window);
    const feeds = {
      features: new o.Tensor("float32", data, [1, window, FEATURE_DIM]),
      mask: new o.Tensor("bool", mask, [1, window]),
    };
    const out = await session.run(feeds);
    return out[session.outputNames[0]].data as Float32Array;
  }

  /** Top-k glosses for one sign window. */
  async classify(frames: HolisticFrame[], leftHanded: boolean, k = 5): Promise<Candidate[]> {
    if (!this.signs) return [];
    const probs = await this.run(this.signs, frames, MAX_FRAMES, leftHanded, true);
    return Array.from(probs, (p, i) => [this.labels[i], p] as Candidate)
      .sort((a, b) => b[1] - a[1])
      .slice(0, k);
  }

  /** Greedy CTC decode of a fingerspelled word. */
  async spell(frames: HolisticFrame[], leftHanded: boolean): Promise<string> {
    if (!this.spellSession) return "";
    const lp = await this.run(this.spellSession, frames, FS_MAX_FRAMES, leftHanded, false);
    const classes = this.alphabet.length + 1;
    const steps = Math.min(frames.length, FS_MAX_FRAMES);
    let prev = 0;
    let word = "";
    for (let t = 0; t < steps; t++) {
      let best = 0;
      for (let c = 1; c < classes; c++) if (lp[t * classes + c] > lp[t * classes + best]) best = c;
      if (best !== prev && best !== 0) word += this.alphabet[best - 1];
      prev = best;
    }
    return word;
  }
}
