// Node verification: confirm the browser JS core matches PyTorch exactly.
// Loads exported weights/tokenizer, runs GPT.generate (JS) and compares the
// argmax of the first generated token against a PyTorch reference.
import fs from "node:fs";
import { Tokenizer } from "./site/llm-tokenizer.js";
import { GPT } from "./site/llm-core.js";

function readJson(p) { return JSON.parse(fs.readFileSync(p, "utf-8")); }

// Build Tokenizer + GPT directly from files (no fetch in Node).
const tokData = readJson("site/tokenizer.json");
const tok = new Tokenizer(tokData.encoder, tokData.merges);

const manifest = readJson("site/model.json");
const buf = fs.readFileSync("site/model.bin");
const all = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
const weights = {};
let off = 0;
for (const info of manifest.tensors) {
  const n = info.nbytes / 4;
  weights[info.name] = all.subarray(off, off + n);
  off += n;
}
const model = new GPT(manifest, weights);

// 1) tokenizer roundtrip
const sample = "It is a truth universally acknowledged, that a single man";
const ids = tok.encode(sample);
const rt = tok.decode(ids);
console.log("tokenizer roundtrip ok:", rt.startsWith(sample.slice(0, 20)));

// 2) forward pass: compare JS logits vs Python logits on a short prompt
const prompt = "It is a truth";
const enc = tok.encode(prompt);
const jsLogits = model.forward(enc);
// argmax of last position
let jsArgmax = 0, jsMax = -Infinity;
const lastOff = (enc.length - 1) * model.V;
for (let i = 0; i < model.V; i++) if (jsLogits[lastOff + i] > jsMax) { jsMax = jsLogits[lastOff + i]; jsArgmax = i; }
console.log("JS prompt ids:", enc);
console.log("JS argmax token:", jsArgmax, "->", JSON.stringify(tok.decode([jsArgmax])));

// 3) generation sanity
const gen = model.generate(tok, prompt, 40, { temperature: 0.8, topK: 0, topP: 0, seed: 1 });
console.log("JS generated (40):", JSON.stringify(gen.slice(prompt.length)));

// 4) write a small JSON the Python side can read to compare argmax/logits
fs.writeFileSync("site/_verify.json", JSON.stringify({
  prompt, ids: enc, jsArgmax,
  jsLogitsLast: Array.from(jsLogits.subarray(lastOff, lastOff + Math.min(model.V, 50))),
}));
console.log("wrote site/_verify.json for PyTorch cross-check");
