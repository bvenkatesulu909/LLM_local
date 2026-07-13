"""
LLM_local — Pre-training (Step 3 of the LLM-from-scratch pipeline).

Corpus  ->  from-scratch BPE tokenizer (bpe.py)  ->  token ids
token ids  ->  GPT (model.py) trained with next-token cross-entropy
               (AdamW + linear LR warmup + gradient clipping)

Usage:
  python train.py --vocab_size 6000 --n_layer 6 --n_embd 384 --steps 4000

All steps print live loss / perplexity so progress is observable.
"""
import argparse
import math
import os
import time
import glob

import torch
from torch.utils.data import IterableDataset, DataLoader

from bpe import BPETokenizer
from model import GPT


def build_corpus(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    texts = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    corpus = "\n".join(texts)
    print(f"Loaded {len(files)} files, {len(corpus):,} characters")
    return corpus


class TokenDataset(torch.utils.data.Dataset):
    """Random fixed-window sampling from the token stream (vectorized).

    We pre-draw `num_windows` random contiguous windows of length
    `block_size`. The window assembly is vectorized with a single index
    gather so dataset construction is instant even for hundreds of k windows.
    This is standard LLM pre-training practice and keeps the loader fast.
    """

    def __init__(self, ids, block_size, num_windows):
        self.block_size = block_size
        ids_t = torch.tensor(ids, dtype=torch.long)
        n = len(ids)
        span = block_size + 1
        max_start = n - span
        starts = torch.randint(0, max_start, (num_windows,))
        idx = starts.unsqueeze(1) + torch.arange(span)  # (num_windows, span)
        seq = ids_t[idx]  # (num_windows, span)
        self.x = seq[:, :block_size].clone()
        self.y = seq[:, 1:].clone()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    ratio = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--vocab_size", type=int, default=6000)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max_lr", type=float, default=3e-3)
    ap.add_argument("--min_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--num_windows", type=int, default=300000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok_path = os.path.join(args.out_dir, "tokenizer.json")

    # --- 1. tokenize -----------------------------------------------------
    if os.path.exists(tok_path):
        print("Loading existing tokenizer ...")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        print("Training from-scratch BPE tokenizer ...")
        corpus = build_corpus(args.data_dir)
        tokenizer = BPETokenizer()
        tokenizer.train(corpus, vocab_size=args.vocab_size)
        tokenizer.save(tok_path)

    print(f"Vocab size: {tokenizer.vocab_size}")
    corpus = build_corpus(args.data_dir)
    ids = tokenizer.encode(corpus)
    print(f"Tokenized corpus -> {len(ids):,} tokens")

    # --- 2. model --------------------------------------------------------
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model.to(args.device)
    total = model.num_params(non_embedding=True)
    print(f"Model parameters (non-embd): {total:,}")
    print(model.config)

    # --- 3. optimizer ----------------------------------------------------
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay
    )

    ds = TokenDataset(ids, args.block_size, args.num_windows)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # --- 4. training loop -------------------------------------------------
    model.train()
    best_loss = float("inf")
    running = 0.0
    t0 = time.time()
    for step, (x, y) in enumerate(loader):
        if step >= args.steps:
            break
        x, y = x.to(args.device), y.to(args.device)

        lr = get_lr(step, args.warmup, args.steps, args.max_lr, args.min_lr)
        for g in optim.param_groups:
            g["lr"] = lr

        logits, loss = model(x, y)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        running += loss.item()
        if step % 10 == 0:
            ppl = math.exp(loss.item())
            print(
                f"step {step:5d} | loss {loss.item():.4f} | ppl {ppl:7.1f} | "
                f"lr {lr:.2e} | {(time.time()-t0):.0f}s",
                flush=True,
            )

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            ckpt = os.path.join(args.out_dir, f"ckpt_{step+1}.pt")
            model.save(ckpt)
            print(f"  saved {ckpt}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                model.save(os.path.join(args.out_dir, "best.pt"))
                print(f"  new best -> {os.path.join(args.out_dir, 'best.pt')}")

    print(f"Done. Final loss {running/max(1,args.steps):.4f}")


if __name__ == "__main__":
    main()
