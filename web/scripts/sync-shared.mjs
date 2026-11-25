// Copies the landmark spec (and the parity-test fixture) from ../shared so the
// browser preprocesses frames exactly like the Python training code.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
const shared = new URL("../../shared/", import.meta.url);
const lib = new URL("../lib/generated/", import.meta.url);
mkdirSync(lib, { recursive: true });
copyFileSync(new URL("landmark_spec.json", shared), new URL("landmark_spec.json", lib));
const fixture = new URL("preprocess_fixture.json", shared);
if (existsSync(fixture)) copyFileSync(fixture, new URL("preprocess_fixture.json", lib));
console.log("synced landmark spec from shared/");

// Serve the WASM runtimes from this app instead of a CDN: works offline and
// keeps every request on your own server (no third party sees usage).
import { cpSync, readdirSync } from "node:fs";
const pub = new URL("../public/", import.meta.url);
const ortDist = new URL("../node_modules/onnxruntime-web/dist/", import.meta.url);
mkdirSync(new URL("ort/", pub), { recursive: true });
for (const f of readdirSync(ortDist))
  if (/^ort-wasm-simd-threaded(\.jsep)?\.(wasm|mjs)$/.test(f)) copyFileSync(new URL(f, ortDist), new URL(`ort/${f}`, pub));
cpSync(new URL("../node_modules/@mediapipe/tasks-vision/wasm/", import.meta.url), new URL("mediapipe/wasm/", pub), { recursive: true });
console.log("copied ONNX Runtime and MediaPipe WASM into public/");
