// LLM_local — byte-level BPE tokenizer (browser port of bpe.py).
// Mirrors the Python implementation exactly so encode/decode round-trips.

// GPT-2 byte-level alphabet: map every byte 0..255 to a printable glyph.
function bytesToUnicode() {
  const bs = [];
  for (let b = "!" .charCodeAt(0); b <= "~".charCodeAt(0); b++) bs.push(b);
  for (let b = "¡".charCodeAt(0); b <= "¬".charCodeAt(0); b++) bs.push(b);
  for (let b = "®".charCodeAt(0); b <= "ÿ".charCodeAt(0); b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const m = {};
  for (let i = 0; i < bs.length; i++) m[bs[i]] = String.fromCodePoint(cs[i]);
  return m;
}

// GPT-2 pre-tokenization pattern (Unicode-aware).
const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class Tokenizer {
  constructor(encoder, merges) {
    this.b2u = bytesToUnicode();
    this.u2b = {};
    for (const k in this.b2u) this.u2b[this.b2u[k]] = Number(k);
    this.encoder = encoder;        // glyph-string -> id
    this.decoder = {};             // id -> glyph-string
    for (const g in encoder) this.decoder[encoder[g]] = g;
    // merges as a rank lookup
    this.ranks = new Map();
    merges.forEach((m, i) => this.ranks.set(m[0] + "" + m[1], i));
  }

  static async load(url) {
    const data = await (await fetch(url)).json();
    return new Tokenizer(data.encoder, data.merges);
  }

  _preTokenize(text) {
    return Array.from(text.matchAll(PAT), (m) => m[0]);
  }

  _getPairs(word) {
    const pairs = new Set();
    for (let i = 0; i < word.length - 1; i++) pairs.add(word[i] + "" + word[i + 1]);
    return pairs;
  }

  _bpe(token) {
    let word = Array.from(token);
    if (word.length < 2) return word;
    while (true) {
      const pairs = this._getPairs(word);
      let minRank = Infinity, minPair = null;
      for (const p of pairs) {
        const r = this.ranks.get(p);
        if (r !== undefined && r < minRank) { minRank = r; minPair = p; }
      }
      if (minPair === null) break;
      const [a, b] = minPair.split("");
      const merged = a + b;
      const newWord = [];
      let j = 0;
      while (j < word.length) {
        if (j < word.length - 1 && word[j] === a && word[j + 1] === b) {
          newWord.push(merged); j += 2;
        } else { newWord.push(word[j]); j += 1; }
      }
      word = newWord;
      if (word.length === 1) break;
    }
    return word;
  }

  encode(text) {
    const ids = [];
    for (const token of this._preTokenize(text)) {
      // convert raw token to byte-glyphs, then BPE-merge
      let glyph = "";
      for (const ch of token) {
        const code = ch.codePointAt(0);
        const bytes = code < 0x80
          ? [code]
          : Array.from(new TextEncoder().encode(ch));
        for (const by of bytes) glyph += this.b2u[by];
      }
      for (const sym of this._bpe(glyph)) ids.push(this.encoder[sym]);
    }
    return ids;
  }

  decode(ids) {
    let glyphs = "";
    for (const id of ids) glyphs += this.decoder[id] ?? "";
    // glyph -> bytes -> utf-8
    const bytes = new Uint8Array(glyphs.length);
    for (let i = 0; i < glyphs.length; i++) bytes[i] = this.u2b[glyphs[i]];
    return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  }
}
