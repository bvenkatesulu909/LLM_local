// LLM_local — GPT forward pass in pure JS (browser port of model.py).
// Runs entirely client-side on a Float32Array-backed weight set (no deps,
// no WASM). Mirrors the PyTorch math: pre-LN blocks, GELU MLP, causal
// self-attention, weight-tied output head.

export class GPT {
  constructor(manifest, weights) {
    this.cfg = manifest.config;
    this.V = this.cfg.vocab_size;
    this.T = this.cfg.block_size;
    this.L = this.cfg.n_layer;
    this.H = this.cfg.n_head;
    this.C = this.cfg.n_embd;
    this.D = this.C / this.H;   // head dim
    this.w = weights;           // name -> Float32Array
  }

  static async load(dir = "") {
    const base = dir.endsWith("/") ? dir : dir + "/";
    const manifest = await (await fetch(base + "model.json")).json();
    const buf = await (await fetch(base + "model.bin")).arrayBuffer();
    const all = new Float32Array(buf);
    const weights = {};
    let off = 0;
    for (const info of manifest.tensors) {
      const n = info.nbytes / 4;
      weights[info.name] = all.subarray(off, off + n);
      off += n;
    }
    return new GPT(manifest, weights);
  }

  // x: [rows, in], W: [out=cols, in] (PyTorch Linear layout) -> [rows, cols]
  // computes y = x @ W.T  (W.T[k, c] = W[c, k])
  _matmul(x, W, rows, cols) {
    const inDim = W.length / cols;
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        let s = 0;
        const xb = r * inDim;
        const wb = c * inDim;
        for (let k = 0; k < inDim; k++) s += x[xb + k] * W[wb + k];
        out[r * cols + c] = s;
      }
    }
    return out;
  }

  // token + positional embedding -> [T, C]
  _embed(idx, T) {
    const x = new Float32Array(T * this.C);
    const tok = this.w["tok_emb.weight"];
    const pos = this.w["pos_emb"];
    for (let t = 0; t < T; t++) {
      const ti = idx[t];
      for (let c = 0; c < this.C; c++) x[t * this.C + c] = tok[ti * this.C + c] + pos[t * this.C + c];
    }
    return x;
  }

  _layernorm(x, T, g, b) {
    const out = new Float32Array(T * this.C);
    const eps = 1e-5;
    for (let t = 0; t < T; t++) {
      let mean = 0;
      for (let c = 0; c < this.C; c++) mean += x[t * this.C + c];
      mean /= this.C;
      let varr = 0;
      for (let c = 0; c < this.C; c++) { const d = x[t * this.C + c] - mean; varr += d * d; }
      varr /= this.C;
      const inv = 1 / Math.sqrt(varr + eps);
      for (let c = 0; c < this.C; c++) out[t * this.C + c] = (x[t * this.C + c] - mean) * inv * g[c] + (b ? b[c] : 0);
    }
    return out;
  }

  _gelu(x) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const v = 0.7978845608 * (x[i] + 0.044715 * x[i] * x[i] * x[i]);
      out[i] = 0.5 * x[i] * (1 + Math.tanh(v));
    }
    return out;
  }

  // causal multi-head self-attention on LN'd input [T, C] -> [T, C]
  _attn(lnx, T, l) {
    const cAttn = this.w[`blocks.${l}.attn.c_attn.weight`];
    const qkv = this._matmul(lnx, cAttn, T, 3 * this.C);
    const q = qkv.subarray(0, T * this.C);
    const k = qkv.subarray(T * this.C, 2 * T * this.C);
    const v = qkv.subarray(2 * T * this.C, 3 * T * this.C);
    const out = new Float32Array(T * this.C);
    const scale = Math.sqrt(this.D);
    for (let h = 0; h < this.H; h++) {
      const off = h * this.D;
      for (let i = 0; i < T; i++) {
        const scores = new Float32Array(T);
        let max = -Infinity;
        for (let j = 0; j <= i; j++) {
          let s = 0;
          for (let d = 0; d < this.D; d++) s += q[i * this.C + off + d] * k[j * this.C + off + d];
          s /= scale; scores[j] = s; if (s > max) max = s;
        }
        let sum = 0; const probs = new Float32Array(T);
        for (let j = 0; j <= i; j++) { const e = Math.exp(scores[j] - max); probs[j] = e; sum += e; }
        for (let j = 0; j <= i; j++) probs[j] /= sum;
        for (let d = 0; d < this.D; d++) {
          let acc = 0;
          for (let j = 0; j <= i; j++) acc += probs[j] * v[j * this.C + off + d];
          out[i * this.C + off + d] = acc;
        }
      }
    }
    return out;
  }

  forward(idx) {
    const T = idx.length;
    let x = this._embed(idx, T);
    for (let l = 0; l < this.L; l++) {
      const ln1 = this._layernorm(x, T, this.w[`blocks.${l}.ln1.weight`], this.w[`blocks.${l}.ln1.bias`]);
      const a = this._matmul(this._attn(ln1, T, l), this.w[`blocks.${l}.attn.c_proj.weight`], T, this.C);
      x = add(x, a);
      const ln2 = this._layernorm(x, T, this.w[`blocks.${l}.ln2.weight`], this.w[`blocks.${l}.ln2.bias`]);
      const ff = this._matmul(this._gelu(this._matmul(ln2, this.w[`blocks.${l}.mlp.fc.weight`], T, 4 * this.C)),
                               this.w[`blocks.${l}.mlp.proj.weight`], T, this.C);
      x = add(x, ff);
    }
    const xf = this._layernorm(x, T, this.w["ln_f.weight"], this.w["ln_f.bias"]);
    return this._matmul(xf, this.w["tok_emb.weight"], T, this.V); // [T, V]
  }

  generate(tok, prompt, maxNew, { temperature = 1.0, topK = 0, topP = 0, seed = 0 } = {}) {
    let ids = tok.encode(prompt);
    if (ids.length === 0) ids = [0];
    const rng = mulberry32(seed >>> 0);
    for (let n = 0; n < maxNew; n++) {
      const ctx = ids.slice(-this.T);
      const logits = this.forward(ctx);
      const last = logits.subarray((ctx.length - 1) * this.V, ctx.length * this.V);
      const scaled = new Float32Array(this.V);
      for (let i = 0; i < this.V; i++) scaled[i] = last[i] / Math.max(temperature, 1e-8);
      let candidates = Array.from(scaled, (v, i) => [i, v]);
      if (topK > 0) candidates.sort((a, b) => b[1] - a[1]).slice(0, topK);
      let maxv = -Infinity; for (const [, v] of candidates) if (v > maxv) maxv = v;
      let sum = 0; const probs = candidates.map(([i, v]) => [i, Math.exp(v - maxv)]);
      for (const [, e] of probs) sum += e;
      if (topP > 0) {
        candidates.sort((a, b) => b[1] - a[1]);
        let cum = 0; const kept = [];
        for (const [i, v] of candidates) { cum += v / sum; kept.push([i, v / sum]); if (cum >= topP) break; }
        let s2 = 0; for (const [, p] of kept) s2 += p;
        ids.push(sampleFrom(kept.map(([i, p]) => [i, p / s2]), rng));
      } else {
        ids.push(sampleFrom(probs.map(([i, e]) => [i, e / sum]), rng));
      }
    }
    return tok.decode(ids);
  }
}

function add(a, b) {
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] + b[i];
  return o;
}

function sampleFrom(probs, rng) {
  const r = rng();
  let cum = 0;
  for (const [i, p] of probs) { cum += p; if (r <= cum) return i; }
  return probs[probs.length - 1][0];
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
