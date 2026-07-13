"""
Export a trained LLM_local GPT to a browser-friendly format.

Produces, in `out_dir`:
  model.bin   - all parameters concatenated as little-endian float32
  model.json  - manifest: {config, tensors:[{name,shape,nbytes}]}
  tokenizer.json - the trained BPE tokenizer (copied from the training out dir)

The browser runtime (site/llm-core.js) reads model.json to slice model.bin
into per-tensor Float32Arrays, then runs the same forward pass as model.py.

Usage:
  python export_for_web.py out_smoke/best.pt site
  python export_for_web.py out/best.pt      site     # upgrade to the big model
"""
import sys
import os
import json
import shutil
import numpy as np
import torch

from model import GPT


def export(src_pt, out_dir, tok_src=None):
    model = GPT.load(src_pt, device="cpu")
    sd = model.state_dict()
    cfg = model.config
    os.makedirs(out_dir, exist_ok=True)

    manifest = {"config": {k: int(v) for k, v in cfg.items()}, "tensors": []}
    with open(os.path.join(out_dir, "model.bin"), "wb") as f:
        for key in sd:
            t = sd[key].detach().cpu().numpy().astype(np.float32)
            b = t.tobytes()
            f.write(b)
            manifest["tensors"].append(
                {"name": key, "shape": list(t.shape), "nbytes": len(b)}
            )

    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    # copy the matching tokenizer from the model's own training output dir
    if tok_src is None:
        tok_src = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tok_src):
        shutil.copy(tok_src, os.path.join(out_dir, "tokenizer.json"))
        print("copied tokenizer from", tok_src)
    else:
        print("WARNING: tokenizer not found at", tok_src)

    size = os.path.getsize(os.path.join(out_dir, "model.bin"))
    print(f"Exported {len(manifest['tensors'])} tensors | config {manifest['config']}")
    print(f"model.bin = {size:,} bytes")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "out_smoke/best.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "site"
    export(src, dst)
