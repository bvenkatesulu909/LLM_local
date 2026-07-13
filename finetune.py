"""
LLM_local — Supervised Fine-Tuning (SFT) for alignment (Step 4).

Takes a base pre-trained GPT and continues training on an instruction set so
it behaves like a helpful assistant instead of a raw next-token predictor.
This is the SFT half of the classic "pre-train -> SFT -> (RLHF)" stack.

The training data is a JSONL file where each line is:
    {"instruction": "...", "response": "..."}
or  {"prompt": "...", "completion": "..."}

We format each example with a chat template, mask the prompt portion so loss
is computed only on the model's response (standard SFT practice).

Usage:
  python finetune.py --data data/sft.jsonl --base out/best.pt
"""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import Dataset, DataLoader

from bpe import BPETokenizer
from model import GPT

SYS = "You are LLM_local, a helpful assistant."
TEMPLATE = (
    "### Instruction:\n{instruction}\n### Response:\n{response}"
)


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, block_size):
        self.tok = tokenizer
        self.block_size = block_size
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                instr = obj.get("instruction") or obj.get("prompt") or ""
                resp = obj.get("response") or obj.get("completion") or ""
                full = TEMPLATE.format(instruction=instr, response=resp)
                instr_only = TEMPLATE.format(instruction=instr, response="")
                full_ids = self.tok.encode(full)[:block_size]
                instr_ids = self.tok.encode(instr_only)[:block_size]
                # labels: -100 (ignored) on the prompt, real ids on the response
                labels = [-100] * len(full_ids)
                loss_start = min(len(instr_ids), len(full_ids))
                for i in range(loss_start, len(full_ids)):
                    labels[i] = full_ids[i]
                self.examples.append((full_ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, labels = self.examples[i]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate(batch, block_size):
    xs, ys = zip(*batch)
    maxlen = min(max(len(x) for x in xs), block_size)
    x_out, y_out = [], []
    for x, y in zip(xs, ys):
        pad = maxlen - len(x)
        x_out.append(x + [0] * pad)
        y_out.append(y + [-100] * pad)
    return torch.stack(x_out), torch.stack(y_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="out/best.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = GPT.load(args.base, device=args.device)
    tok = BPETokenizer.load(os.path.join(args.out_dir, "tokenizer.json"))
    model.to(args.device)
    print(f"SFT: {model.num_params():,} params from {args.base}")

    ds = SFTDataset(args.data, tok, args.block_size)
    print(f"Loaded {len(ds)} SFT examples")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, args.block_size),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    model.train()
    it = iter(loader)
    t0 = time.time()
    for step in range(args.steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(args.device), y.to(args.device)
        _, loss = model(x, y)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 25 == 0:
            print(f"step {step:4d} | sft_loss {loss.item():.4f} | {(time.time()-t0):.0f}s")
    out = os.path.join(args.out_dir, "sft.pt")
    model.save(out)
    print(f"Saved SFT model -> {out}")


if __name__ == "__main__":
    main()
