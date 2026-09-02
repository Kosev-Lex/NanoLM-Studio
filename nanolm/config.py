"""Central configuration for NanoLM Studio v4.

The default workspace is ``./data`` next to ``main.py``.  Set
``NANOLM_DATA_DIR`` to keep models and corpora somewhere else; this is
especially useful for tests, removable drives, and multiple projects.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

# ------------------------------------------------------------------
# Directory layout (everything lives under ./data next to main.py)
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NANOLM_DATA_DIR", ROOT_DIR / "data")).expanduser().resolve()
RAW_DIR = DATA_DIR / "raw"            # original extracted text, pre-cleaning
DOC_DIR = DATA_DIR / "documents"      # cleaned per-document text
TOKENIZER_DIR = DATA_DIR / "tokenizer"
CACHE_DIR = DATA_DIR / "cache"        # tokenized corpus caches
CKPT_DIR = DATA_DIR / "checkpoints"
RUNS_DIR = DATA_DIR / "runs"          # persisted training histories (jsonl)
DB_PATH = DATA_DIR / "corpus.db"
TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"

DATA_DIRS = (DATA_DIR, RAW_DIR, DOC_DIR, TOKENIZER_DIR, CACHE_DIR, CKPT_DIR, RUNS_DIR)


def ensure_data_dirs() -> None:
    """Create application storage lazily instead of writing during import."""
    for directory in DATA_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

# Canonical special tokens.  ORDER MATTERS: it fixes the token ids.
# This is the single source of truth for the whole system.
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
PAD, UNK, BOS, EOS = 0, 1, 2, 3


# ------------------------------------------------------------------
# Model configuration (saved inside every checkpoint)
# ------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.05

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "dim", "n_layers", "n_heads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


# Presets shown in the Training tab: name -> (n_layers, dim, n_heads, block_size)
MODEL_PRESETS = {
    "Tiny   (4 x 256, ctx 128)":   (4, 256, 4, 128),
    "Small  (6 x 512, ctx 256)":   (6, 512, 8, 256),
    "Medium (8 x 640, ctx 256)":   (8, 640, 10, 256),
    "Large  (12 x 768, ctx 512)":  (12, 768, 12, 512),
    "XL     (16 x 1024, ctx 512)": (16, 1024, 16, 512),
    "XXL    (24 x 1280, ctx 512)": (24, 1280, 20, 512),
}
DEFAULT_PRESET = "Small  (6 x 512, ctx 256)"


# ------------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------------
@dataclass
class TrainConfig:
    total_steps: int = 2000          # optimizer-update steps
    batch_size: int = 16
    grad_accum: int = 1
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    val_interval: int = 250
    val_batches: int = 32            # fixed, small: validation stays cheap
    patience: int = 8                # early stop after N vals w/o improvement
    log_interval: int = 10
    sample_interval: int = 500       # generate a sample every N steps
    sample_prompt: str = ""          # empty -> seed from validation data
    sample_tokens: int = 80
    resume: bool = False             # resume is EXPLICIT, never silent
    seed: int = 1337

    def __post_init__(self) -> None:
        positive = (
            "total_steps", "batch_size", "grad_accum", "lr_max",
            "val_interval", "val_batches", "patience", "log_interval",
            "sample_tokens",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.lr_min < 0 or self.lr_min > self.lr_max:
            raise ValueError("lr_min must be between zero and lr_max")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be greater than zero")
        if self.sample_interval < 0:
            raise ValueError("sample_interval cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)
