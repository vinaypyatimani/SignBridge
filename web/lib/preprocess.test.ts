import { describe, expect, it } from "vitest";
import fixture from "./generated/preprocess_fixture.json";
import { FEATURE_DIM, N_HOLISTIC, pad, preprocess } from "./preprocess";

function frames(): Float32Array[] {
  return (fixture.frames as (number | null)[][][]).map((f) => {
    const out = new Float32Array(N_HOLISTIC * 3);
    f.forEach((p, i) => {
      out[i * 3] = p[0] ?? NaN;
      out[i * 3 + 1] = p[1] ?? NaN;
      out[i * 3 + 2] = 0;
    });
    return out;
  });
}

describe("preprocess matches the Python implementation", () => {
  for (const c of fixture.cases) {
    it(`left_handed=${c.left_handed}`, () => {
      const got = preprocess(frames(), c.max_frames, c.left_handed);
      expect(got.length).toBe(c.features.length);
      let worst = 0;
      got.forEach((row, t) => row.forEach((v, j) => { worst = Math.max(worst, Math.abs(v - c.features[t][j])); }));
      expect(worst).toBeLessThan(1e-4);
    });
  }

  it("pads to a fixed window with a mask", () => {
    const { data, mask } = pad(preprocess(frames(), 32), 64);
    expect(data.length).toBe(64 * FEATURE_DIM);
    expect(mask.reduce((a, b) => a + b, 0)).toBe(32);
  });
});
