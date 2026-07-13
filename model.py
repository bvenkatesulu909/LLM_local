"""
LLM_local — GPT-style Transformer language model, implemented from scratch
in PyTorch.

Implements the decoder-only Transformer ("GPT") architecture directly:
  - token + learned positional embeddings (no sinusoids)
  - causal multi-head self-attention
  - position-wise feed-forward MLP
  - pre-LayerNorm residual blocks (stable, modern; identical math to GPT-2
    post-norm up to a final norm)
  - weight tying between the input token embedding and the output projection

No HuggingFace / `nn.Transformer` shortcuts are used for the core math; the
attention and MLP are written explicitly so the architecture is transparent.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention with a causal mask."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # Q, K, V projections packed into one linear (standard GPT style)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # register a causal (lower-triangular) mask buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence, channels
        q, k, v = self.c_attn(x).split(C, dim=2)
        # reshape into (B, n_head, T, head_dim) for per-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.c_proj(out))
        return out


class MLP(nn.Module):
    """Position-wise feed-forward network (SwiGLU-free, plain GELU MLP)."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.drop(self.proj(self.drop(self.act(self.fc(x)))))


class Block(nn.Module):
    """A single pre-LayerNorm Transformer block with a residual connection."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size=256, n_layer=6, n_head=6,
                 n_embd=384, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.config = dict(vocab_size=vocab_size, block_size=block_size,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd)

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        # weight-tied output head: shares weights with token embedding
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 trick)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, "sequence exceeds block_size"
        tok = self.tok_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Autoregressively sample `max_new_tokens` from the model."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p is not None:
                logits = self._nucleus(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @staticmethod
    def _nucleus(logits, top_p):
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative - probs > top_p
        remove = torch.cat([torch.zeros_like(remove[:, :1]), remove[:, :-1]], dim=-1)
        sorted_logits[remove] = float("-inf")
        # scatter back to original ordering
        out = torch.full_like(logits, float("-inf"))
        out.scatter_(1, sorted_idx, sorted_logits)
        return out

    def save(self, path):
        torch.save({"model_state": self.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["config"]
        model = cls(**cfg)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        return model
