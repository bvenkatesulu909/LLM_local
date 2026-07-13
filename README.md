# LLM_local

A real, working **Large Language Model built from scratch** — no `tiktoken`,
no `transformers`, no `nn.Transformer`. Every core component (tokenizer,
attention, training loop) is implemented by hand in PyTorch, following the
four canonical stages of building an LLM:

1. **Data Collection & Preprocessing** — public-domain text corpus → from-scratch
   **Byte-Pair Encoding (BPE)** tokenizer.
2. **Model Architecture** — a **GPT-style decoder-only Transformer** with
   causal multi-head self-attention (written explicitly, not via a library).
3. **Pre-training** — next-token prediction over millions of tokens with
   AdamW + LR warmup + gradient clipping (backpropagation adjusts the weights).
4. **Fine-Tuning & Alignment** — **Supervised Fine-Tuning (SFT)** on an
   instruction set so the model behaves like a helpful assistant.

> Everything runs on CPU. The reference run uses a ~6.4 MB corpus of public
> domain novels (Pride & Prejudice, Moby-Dick, Frankenstein, etc.) and a
> ~10 M-parameter GPT. Swap in a bigger corpus / larger config to scale up.

## Project layout
```
llm_local/
├── bpe.py          # from-scratch byte-level BPE tokenizer
├── model.py        # GPT Transformer (causal self-attention, MLP, blocks)
├── train.py        # pre-training pipeline (stage 3)
├── finetune.py     # SFT alignment pipeline (stage 4)
├── generate.py     # inference / chat
├── data/           # training corpus (*.txt) + SFT data (sft.jsonl)
└── out/            # tokenizer.json + ckpt_*.pt + best.pt + sft.pt
```

## Setup
```bash
cd llm_local
uv venv --python 3.12 .venv
. .venv/Scripts/activate          # Windows; use source .venv/bin/activate on Linux/mac
python -m pip install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy regex
# IMPORTANT: if PYTHONPATH leaks the hermes venv, run `unset PYTHONPATH` first.
```

## Train (Stage 1–3)
```bash
python train.py --vocab_size 6000 --n_layer 6 --n_head 6 --n_embd 384 \
                --block_size 256 --batch_size 16 --steps 4000
```
This trains the BPE tokenizer (once, cached to `out/tokenizer.json`), then
trains the GPT and saves `out/best.pt` + periodic checkpoints. Watch the loss
and perplexity drop in `out/train.log`.

## Generate / Chat (Stage 4 inference)
```bash
python generate.py --prompt "It is a truth universally acknowledged"
python generate.py --chat
```

## Fine-Tune for alignment (Stage 4)
```bash
python finetune.py --data data/sft.jsonl --base out/best.pt
# then use the SFT model:
python generate.py --out_dir out --prompt "What is a Large Language Model?"   # loads best.pt
```
To load the SFT model instead of the base model, point `generate.py` at
`out/sft.pt` (edit `--out_dir` or copy `sft.pt` to `best.pt`).

## How it works
- **Tokenizer (`bpe.py`)** — GPT-2 byte-level alphabet (every byte 0–255 maps to
  a printable glyph so UTF-8 is always reversible), the GPT-2 pre-tokenization
  regex, and a frequency-based BPE merge learner. `encode`/`decode` round-trip
  exactly.
- **Model (`model.py`)** — token + learned positional embeddings, `n_layer`
  pre-LayerNorm residual blocks (each = causal multi-head self-attention +
  GELU MLP), a final LayerNorm, and a weight-tied output head. Dropout for
  regularization, scaled init for residual projections.
- **Training (`train.py`)** — stride-1 windows of `block_size` tokens, AdamW
  with linear LR warmup and cosine decay, gradient clipping at 1.0, and
  cross-entropy next-token loss.

## Scaling up
To build a bigger LLM_local, increase `--vocab_size`, `--n_layer`, `--n_head`,
`--n_embd`, add more `data/*.txt` files, and raise `--steps`. For GPU, install
the CUDA torch wheel and pass `--device cuda`.
