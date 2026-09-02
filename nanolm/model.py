"""NanoLM v4: a small, clean decoder-only transformer.

Design notes:
* Fast path uses F.scaled_dot_product_attention (fused kernels); the
  manual attention path exists ONLY for Glass Box introspection and is
  taken on demand -- training never pays the capture cost.
* Checkpoints embed the full ModelConfig and the tokenizer fingerprint,
  so loading can never silently mismatch architecture or vocab.
* No torch.compile, keeping Windows behaviour and checkpoint keys predictable.
* Generation is a streaming generator with top-k, top-p, and a light
  repetition penalty (a big help for nano-scale models).
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import BOS, ModelConfig


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, capture: Optional[dict] = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if capture is None:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Introspection path: explicit softmax so weights can be recorded.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            capture["attn"] = att.detach().to("cpu")     # [B, H, T, T]
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

    def forward_cached(self, x: torch.Tensor,
                       past: Optional[tuple[torch.Tensor, torch.Tensor]] = None
                       ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Inference-only attention with a per-layer key/value cache."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        past_len = 0
        if past is not None:
            past_k, past_v = past
            past_len = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        if past_len == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            q_pos = past_len + torch.arange(T, device=x.device).unsqueeze(1)
            k_pos = torch.arange(past_len + T, device=x.device).unsqueeze(0)
            mask = k_pos <= q_pos
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        inner = 4 * cfg.dim
        self.fc1 = nn.Linear(cfg.dim, inner)
        self.fc2 = nn.Linear(inner, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, capture: Optional[dict] = None) -> torch.Tensor:
        a = F.gelu(self.fc1(x))
        if capture is not None:
            capture["mlp_act_norm"] = a.detach().norm(dim=-1).mean().item()
        return self.drop(self.fc2(a))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, capture: Optional[dict] = None) -> torch.Tensor:
        if capture is not None:
            capture["resid_norm"] = x.detach().norm(dim=-1).mean().item()
        x = x + self.attn(self.ln1(x), capture)
        x = x + self.mlp(self.ln2(x), capture)
        return x

    def forward_cached(self, x: torch.Tensor,
                       past: Optional[tuple[torch.Tensor, torch.Tensor]] = None
                       ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present = self.attn.forward_cached(self.ln1(x), past)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


class NanoLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init)
        self.head.weight = self.tok_emb.weight        # tie after initialization

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    @property
    def num_params(self) -> int:
        # parameters() already de-duplicates the tied embedding/head tensor.
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                captures: Optional[list] = None):
        B, T = idx.shape
        if T == 0:
            raise ValueError("idx must contain at least one token")
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")
        if T > self.cfg.block_size:
            idx = idx[:, -self.cfg.block_size:]
            if targets is not None:
                targets = targets[:, -self.cfg.block_size:]
            B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            cap = None
            if captures is not None:
                cap = {}
                captures.append(cap)
            x = block(x, cap)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss

    def forward_cached(self, idx: torch.Tensor,
                       cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
                       ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward inference tokens while reusing keys/values from prior tokens."""
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("idx must contain at least one token")
        past_len = cache[0][0].size(2) if cache else 0
        if past_len + idx.size(1) > self.cfg.block_size:
            raise ValueError("cached sequence exceeds the model context window")
        pos = torch.arange(past_len, past_len + idx.size(1), device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        present = []
        for layer_index, block in enumerate(self.blocks):
            past = cache[layer_index] if cache else None
            x, layer_cache = block.forward_cached(x, past)
            present.append(layer_cache)
        return self.head(self.ln_f(x)), present

    # ------------------- checkpoint IO -------------------
    def save_checkpoint(self, path: Path, extras: Optional[dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 4, "config": self.cfg.to_dict(),
                   "model": self.state_dict()}
        if extras:
            payload.update(extras)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".pt", dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, tmp_name)
            os.replace(tmp_name, path)
        finally:
            Path(tmp_name).unlink(missing_ok=True)

    @classmethod
    def from_checkpoint(cls, path: Path, map_location: str = "cpu"):
        payload = torch.load(str(path), map_location=map_location, weights_only=False)
        if "config" not in payload:
            raise RuntimeError(
                "Checkpoint has no embedded config (old or malformed checkpoint). "
                "Retrain or load with matching architecture manually."
            )
        cfg = ModelConfig.from_dict(payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["model"])
        return model, payload


# ===============================================================
# STREAMING GENERATION
# ===============================================================
@torch.no_grad()
def generate_stream(
    model: NanoLM,
    prompt_ids: list[int],
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    eos_id: Optional[int] = None,
    stop_check=None,
) -> Iterator[int]:
    """Yield one sampled token id at a time."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than zero")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    safe_prompt = prompt_ids or [BOS]
    idx = torch.tensor([safe_prompt[-model.cfg.block_size:]], dtype=torch.long, device=device)

    try:
        logits_all, cache = model.forward_cached(idx)
        for _ in range(max_new_tokens):
            if stop_check and stop_check():
                return
            logits = logits_all[0, -1, :].float()

            # repetition penalty over the recent window
            if repetition_penalty and repetition_penalty != 1.0:
                recent = torch.unique(idx[0, -model.cfg.block_size:])
                selected = logits[recent]
                logits[recent] = torch.where(
                    selected > 0, selected / repetition_penalty,
                    selected * repetition_penalty,
                )

            greedy = temperature <= 0
            if not greedy:
                logits = logits / temperature

            if top_k and top_k > 0:
                k = min(int(top_k), logits.size(-1))
                kth = torch.topk(logits, k).values[-1]
                logits[logits < kth] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            if greedy:
                next_id = int(torch.argmax(logits).item())
            elif top_p < 1.0:
                sorted_p, sorted_i = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_p, dim=-1)
                keep = cum - sorted_p < top_p          # always keep the top token
                sorted_p = sorted_p * keep
                sorted_p = sorted_p / sorted_p.sum()
                next_sorted = torch.multinomial(sorted_p, 1)
                next_id = sorted_i[next_sorted].item()
            else:
                next_id = torch.multinomial(probs, 1).item()

            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
            idx = torch.cat((idx, next_tensor), dim=1)
            yield int(next_id)
            if eos_id is not None and next_id == eos_id:
                return
            if idx.size(1) <= model.cfg.block_size:
                logits_all, cache = model.forward_cached(next_tensor, cache)
            else:
                idx = idx[:, -model.cfg.block_size:]
                logits_all, cache = model.forward_cached(idx)
    finally:
        if was_training:
            model.train()
