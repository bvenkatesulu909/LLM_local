"""
LLM_local — Text generation / chat (Step 4 inference).

Loads a trained GPT + BPE tokenizer and samples text. Supports two modes:
  - raw continuation:  python generate.py --prompt "It is a truth"
  - assistant chat:    python generate.py --chat

The chat mode wraps input with a minimal instruction template so the model
behaves like an assistant (works best after SFT, see finetune.py).
"""
import argparse
import os
import torch

from bpe import BPETokenizer
from model import GPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = GPT.load(os.path.join(args.out_dir, "best.pt"), device=args.device)
    tok = BPETokenizer.load(os.path.join(args.out_dir, "tokenizer.json"))
    model.eval()

    print(f"LLM_local ready | vocab={tok.vocab_size} | params={model.num_params():,}")

    def complete(prompt):
        ids = tok.encode(prompt)
        if not ids:
            ids = [0]
        ctx = torch.tensor([ids], dtype=torch.long, device=args.device)
        out = model.generate(
            ctx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        return tok.decode(out[0].tolist())

    if args.chat:
        print("Entering chat (type 'exit' to quit).")
        while True:
            user = input("\nYou> ").strip()
            if user.lower() in ("exit", "quit"):
                break
            prompt = f"User: {user}\nAssistant:"
            print("LLM_local> " + complete(prompt)[len(prompt) :].split("User:")[0].strip())
    else:
        prompt = args.prompt or "It is a truth universally acknowledged"
        print("\n--- generation ---")
        print(complete(prompt))


if __name__ == "__main__":
    main()
