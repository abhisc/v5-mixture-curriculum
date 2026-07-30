"""A small byte-level decoder-only transformer.

Byte-level is a deliberate choice, not a shortcut. Two reasons:

1. Bits-per-byte is directly comparable across lanes and scripts. A BPE tokenizer trained
   on an English-dominated mixture makes Devanagari look artificially expensive, which
   would contaminate exactly the Indic measurement this experiment exists to make.
2. It removes tokenizer fertility as a confound between the arms. Every arm sees the same
   bytes; only the sampling proportions differ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 256


@dataclass
class Config:
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    seq_len: int = 512
    dropout: float = 0.0

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.cfg.n_head, self.cfg.d_head).transpose(1, 2)
        k = k.view(B, T, self.cfg.n_head, self.cfg.d_head).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_head, self.cfg.d_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(a))
        h = self.ln2(x)
        x = x + self.drop(self.fc2(F.gelu(self.fc1(h))))
        return x


class ByteLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(VOCAB, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float = 0.8) -> torch.Tensor:
        for _ in range(max_new):
            ctx = idx[:, -self.cfg.seq_len :]
            logits, _ = self(ctx)
            logits = logits[:, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    def n_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos.weight.numel()
        return n
