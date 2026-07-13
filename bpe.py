"""
LLM_local — Byte-Pair Encoding (BPE) tokenizer, implemented from scratch.

This is a faithful from-scratch implementation of the GPT-2 / GPT-3 style
byte-level BPE algorithm. It does NOT use tiktoken, sentencepiece, or any
HuggingFace tokenizer. The pipeline is:

  1. Pre-tokenization: split raw text into word-chunks using the GPT-2 regex
     pattern so that byte-level encoding is reversible for any UTF-8 string.
  2. Byte-level mapping: map each byte (0-255) to a printable Unicode glyph
     (the GPT-2 "bytes_to_unicode" scheme) so all bytes become valid vocab ids.
  3. BPE merges: greedily merge the most frequent adjacent symbol pairs until
     the target vocab size is reached (or merges are exhausted).
  4. Encode/Decode: encode text -> token ids; decode ids -> original text.

References:
  - Sennrich et al. 2015, "Neural Machine Translation of Rare Words with
    Subword Units" (BPE).
  - Radford et al. 2019, "Language Models are Unsupervised Multitask Learners"
    (GPT-2 byte-level BPE).
"""
import regex as re
import json
import os
from collections import Counter, defaultdict
from functools import lru_cache


# ---------------------------------------------------------------------------
# GPT-2 byte-level alphabet: every byte 0..255 maps to a printable char.
# This guarantees token boundaries never split a multi-byte UTF-8 sequence.
# ---------------------------------------------------------------------------
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


# GPT-2 pre-tokenization pattern. Captures contractions, punctuation, and
# whitespace as separate tokens so byte-level encoding stays reversible.
PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def get_pairs(word):
    """Return set of adjacent symbol pairs in a tuple `word`."""
    return set(zip(word[:-1], word[1:]))


class BPETokenizer:
    """A from-scratch byte-level BPE tokenizer."""

    def __init__(self):
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.encoder = {}        # glyph-string -> id
        self.decoder = {}        # id -> glyph-string
        self.merges = {}         # (a, b) -> rank
        self.vocab_size = 0

    # ---- training ----------------------------------------------------------
    def train(self, text, vocab_size=8000, verbose=True, max_corpus_chars=2_000_000):
        """Train BPE on `text`, producing `vocab_size` tokens.

        The base vocabulary is the 256 byte-glyphs. We then learn
        (vocab_size - 256) merges.

        Optimization: we pre-tokenize the corpus, count how many times each
        *unique* word occurs, and apply merges to the word-frequency map rather
        than re-scanning every occurrence. This is the standard `tokenizers`
        library approach and makes training fast even on a multi-MB corpus.
        """
        assert vocab_size >= 256, "vocab_size must be >= 256"
        num_merges = vocab_size - 256

        # Cap the corpus used for *tokenizer* training (the LM still trains on
        # the full text). A few MB is more than enough to learn a good vocab.
        if len(text) > max_corpus_chars:
            text = text[:max_corpus_chars]

        # 1. Pre-tokenize into words, count frequencies.
        if verbose:
            print("Pre-tokenizing corpus ...")
        word_counts = Counter(self._pre_tokenize(text))
        if verbose:
            print(f"  {sum(word_counts.values()):,} word tokens, "
                  f"{len(word_counts):,} unique")

        # 2. Convert each unique word to a tuple of base byte-glyph symbols.
        word_freqs = {
            tuple(self.b2u[b] for b in w.encode("utf-8")): c
            for w, c in word_counts.items()
        }

        # 3. Learn merges over the word-frequency map.
        if verbose:
            print("Learning BPE merges ...")
        vocab = {c: i for i, c in enumerate(self.b2u.values())}
        merges = {}
        for i in range(num_merges):
            pairs = Counter()
            for word, freq in word_freqs.items():
                for p in get_pairs(word):
                    pairs[p] += freq
            if not pairs:
                if verbose:
                    print("  no more pairs to merge; stopping early")
                break
            # most frequent pair; tie-break by lowest constituent ids
            best = max(
                pairs.items(),
                key=lambda kv: (kv[1], -vocab.get(kv[0][0], 1e9),
                                -vocab.get(kv[0][1], 1e9)),
            )[0]
            a, b = best
            merged = a + b
            merges[best] = len(merges)
            # apply merge to every word that contains (a, b)
            new_freqs = {}
            for word, freq in word_freqs.items():
                if a in word or b in word:
                    new_word = []
                    j = 0
                    while j < len(word):
                        if j < len(word) - 1 and word[j] == a and word[j + 1] == b:
                            new_word.append(merged)
                            j += 2
                        else:
                            new_word.append(word[j])
                            j += 1
                    new_freqs[tuple(new_word)] = new_freqs.get(tuple(new_word), 0) + freq
                else:
                    new_freqs[word] = new_freqs.get(word, 0) + freq
            word_freqs = new_freqs
            vocab[merged] = len(vocab)
            if verbose and (i + 1) % max(1, num_merges // 10) == 0:
                print(f"  merge {i+1}/{num_merges}  '{a}+{b}' -> '{merged}'")

        # 4. Build final encoder/decoder.
        self.merges = merges
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
        if verbose:
            print(f"Trained vocab size: {self.vocab_size}")

    def _pre_tokenize(self, text):
        return PAT.findall(text)

    # ---- encode / decode ---------------------------------------------------
    @lru_cache(maxsize=2**20)
    def _bpe(self, token):
        """Apply merges to a single pre-token (a string of base glyphs)."""
        word = tuple(token)
        if len(word) < 2:
            return list(word)
        while True:
            pairs = get_pairs(word)
            # pick the pair with the lowest merge rank
            min_pair = None
            min_rank = None
            for p in pairs:
                r = self.merges.get(p)
                if r is not None and (min_rank is None or r < min_rank):
                    min_rank = r
                    min_pair = p
            if min_pair is None:
                break
            a, b = min_pair
            merged = a + b
            new_word = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == a and word[j + 1] == b:
                    new_word.append(merged)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        return list(word)

    def encode(self, text):
        ids = []
        for token in self._pre_tokenize(text):
            # convert the raw pre-token to byte-glyphs before merging
            glyph_token = "".join(self.b2u[b] for b in token.encode("utf-8"))
            for sym in self._bpe(glyph_token):
                ids.append(self.encoder[sym])
        return ids

    def decode(self, ids):
        # map ids -> glyphs -> bytes -> utf-8 string
        glyphs = "".join(self.decoder[i] for i in ids)
        text = bytearray(self.u2b[c] for c in glyphs).decode("utf-8", errors="replace")
        return text

    # ---- persistence -------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "encoder": self.encoder,
                    "merges": [list(k) for k in self.merges.keys()],
                    "vocab_size": self.vocab_size,
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.encoder = {k: int(v) for k, v in data["encoder"].items()}
        tok.decoder = {int(v): k for k, v in data["encoder"].items()}
        tok.merges = {tuple(m): i for i, m in enumerate(data["merges"])}
        tok.vocab_size = data["vocab_size"]
        return tok


if __name__ == "__main__":
    # quick self-test
    sample = "Hello world! The quick brown fox jumps over the lazy dog. 1234567890"
    tok = BPETokenizer()
    tok.train(sample * 50, vocab_size=300)
    ids = tok.encode("Hello world! 12345")
    print("ids:", ids)
    print("decoded:", repr(tok.decode(ids)))
    assert tok.decode(ids) == "Hello world! 12345", "roundtrip failed!"
    print("roundtrip OK")
