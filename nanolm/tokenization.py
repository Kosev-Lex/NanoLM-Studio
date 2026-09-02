"""Tokenizer and document-aware token cache for NanoLM Studio v4.

Design notes:
* One canonical special-token order, defined once in config.py.
* No TemplateProcessing: documents are encoded individually and an
  explicit </s> is appended per document -- so EOS lands exactly on
  document boundaries, never at arbitrary 2 MB chunk offsets.
* Train/val split is by WHOLE DOCUMENTS when possible (no overlap, no
  leakage); falls back to a non-overlapping tail slice for tiny corpora.
* Caches carry the tokenizer fingerprint, so a retrained tokenizer
  automatically invalidates stale caches.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .config import BOS, CACHE_DIR, EOS, SPECIAL_TOKENS, TOKENIZER_PATH
from .corpus import CorpusLibrary


class NanoTokenizer:
    """Byte-level BPE wrapper with fixed special-token ids."""

    def __init__(self, path: Path = TOKENIZER_PATH):
        self.path = Path(path)
        self.tokenizer: Optional[Tokenizer] = None
        if self.path.exists():
            self.load()

    # ---------------- lifecycle ----------------
    def load(self):
        self.tokenizer = Tokenizer.from_file(str(self.path))
        expected = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        actual = {token: self.tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
        if actual != expected:
            self.tokenizer = None
            raise RuntimeError(
                f"Tokenizer special-token IDs are incompatible: expected {expected}, got {actual}"
            )

    def save(self):
        self._require_loaded()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".tokenizer.", suffix=".json", dir=self.path.parent)
        os.close(fd)
        try:
            self.tokenizer.save(tmp_name)
            os.replace(tmp_name, self.path)
        finally:
            Path(tmp_name).unlink(missing_ok=True)

    def train_from_texts(self, texts: list[str], vocab_size: int = 4096, min_frequency: int = 2):
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),   # canonical order -> fixed ids
            # Full byte alphabet: without this, characters absent from the
            # training corpus become un-encodable and silently vanish.
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(texts, trainer)
        self.tokenizer = tok
        self.save()

    # ---------------- properties ----------------
    @property
    def loaded(self) -> bool:
        return self.tokenizer is not None

    @property
    def vocab_size(self) -> int:
        self._require_loaded()
        return self.tokenizer.get_vocab_size()

    @property
    def eos_id(self) -> int:
        self._require_loaded()
        return EOS

    @property
    def bos_id(self) -> int:
        self._require_loaded()
        return BOS

    def _require_loaded(self) -> None:
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer is loaded. Train or load one first.")

    def fingerprint(self) -> str:
        """Identity of the trained tokenizer; stored in caches/checkpoints."""
        self._require_loaded()
        if not self.path.exists():
            raise RuntimeError("Tokenizer file is missing; save the tokenizer before using it.")
        return hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]

    # ---------------- encode / decode ----------------
    def encode(self, text: str) -> list[int]:
        self._require_loaded()
        return self.tokenizer.encode(text).ids

    def encode_document(self, text: str) -> list[int]:
        """Document encoding: content tokens + explicit EOS boundary."""
        return self.encode(text) + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        self._require_loaded()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def inspect(self, text: str) -> list[tuple[str, int]]:
        """(token_string, id) pairs for the tokenizer inspector UI."""
        self._require_loaded()
        enc = self.tokenizer.encode(text)
        return [(self.tokenizer.decode([i], skip_special_tokens=False), i)
                for i in enc.ids]


# ===============================================================
# TOKEN CACHE
# ===============================================================
def _dtype_for(vocab_size: int):
    return np.uint16 if vocab_size <= 65535 else np.uint32


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(tmp_name, array, allow_pickle=False)
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _atomic_save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def build_token_cache(library: CorpusLibrary, tok: NanoTokenizer,
                      log=lambda s: None, *, cache_dir: Path = CACHE_DIR,
                      split_seed: int = 1337) -> dict:
    """Tokenize all ACTIVE documents (one array per document, EOS between
    docs), split train/val by whole documents, and persist to CACHE_DIR.

    Returns metadata dict: paths, sizes, fingerprint, doc split.
    """
    docs = library.active_documents()
    if not docs:
        raise RuntimeError("No active documents in the library.")
    fp = tok.fingerprint()
    corpus_hash = hashlib.sha256()
    for doc in docs:
        corpus_hash.update(f"{doc.id}:{doc.sha256}\n".encode("utf-8"))
    corpus_fp = corpus_hash.hexdigest()[:16]
    dtype = _dtype_for(tok.vocab_size)

    cache_dir = Path(cache_dir)
    per_doc: list[tuple[int, np.ndarray]] = []
    token_counts: dict[int, int] = {}
    for d in docs:
        ids = tok.encode_document(library.get_text(d.id))
        per_doc.append((d.id, np.asarray(ids, dtype=dtype)))
        token_counts[d.id] = len(ids)
        log(f"  tokenized #{d.id} '{d.title}': {len(ids):,} tokens")

    total = sum(len(a) for _, a in per_doc)

    # ---- split: whole documents into val when we can ----
    val_docs: list[int] = []
    if len(per_doc) >= 5:
        target = max(1, int(0.08 * total))
        order = list(range(len(per_doc)))
        np.random.default_rng(split_seed).shuffle(order)
        held_out = 0
        for index in order:
            doc_id, arr = per_doc[index]
            if held_out >= target:
                break
            # never put more than ~20% of the corpus in val
            if held_out + len(arr) > 0.2 * total:
                continue
            val_docs.append(doc_id)
            held_out += len(arr)

    if val_docs:
        train_arr = np.concatenate([a for i, a in per_doc if i not in val_docs])
        val_arr = np.concatenate([a for i, a in per_doc if i in val_docs])
        split_mode = f"by document ({len(val_docs)} doc(s) held out)"
    else:
        merged = np.concatenate([a for _, a in per_doc])
        cut = int(0.9 * len(merged))
        train_arr, val_arr = merged[:cut], merged[cut:]   # NON-overlapping
        split_mode = "tail slice (corpus too small for per-doc split)"

    if len(train_arr) < 2 or len(val_arr) < 2:
        raise RuntimeError("Corpus is too small to create non-empty train and validation caches.")

    train_path = cache_dir / "train_tokens.npy"
    val_path = cache_dir / "val_tokens.npy"
    _atomic_save_array(train_path, train_arr)
    _atomic_save_array(val_path, val_arr)

    meta = {
        "format_version": 4,
        "tokenizer_fingerprint": fp,
        "corpus_fingerprint": corpus_fp,
        "vocab_size": tok.vocab_size,
        "dtype": str(dtype.__name__) if hasattr(dtype, "__name__") else str(dtype),
        "total_tokens": int(total),
        "train_tokens": int(len(train_arr)),
        "val_tokens": int(len(val_arr)),
        "val_doc_ids": val_docs,
        "split_mode": split_mode,
        "doc_ids": [d.id for d in docs],
        "split_seed": split_seed,
    }
    _atomic_save_json(cache_dir / "meta.json", meta)
    for doc_id, count in token_counts.items():
        library.set_token_count(doc_id, count)
    log(f"  cache written: train={len(train_arr):,} val={len(val_arr):,} ({split_mode})")
    return meta


def load_token_cache(expected_fingerprint: Optional[str] = None, *,
                     expected_corpus_fingerprint: Optional[str] = None,
                     cache_dir: Path = CACHE_DIR):
    """Load cached arrays.  Returns (train, val, meta) or raises with a
    clear message if missing/stale."""
    cache_dir = Path(cache_dir)
    meta_p = cache_dir / "meta.json"
    if not meta_p.exists():
        raise RuntimeError("Token cache missing -- build it from the Training tab.")
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Token cache metadata is corrupt: {exc}") from exc
    required = {"tokenizer_fingerprint", "train_tokens", "val_tokens"}
    missing = required.difference(meta)
    if missing:
        raise RuntimeError(f"Token cache metadata is missing: {', '.join(sorted(missing))}")
    if expected_fingerprint and meta["tokenizer_fingerprint"] != expected_fingerprint:
        raise RuntimeError(
            "Token cache was built with a different tokenizer -- rebuild the cache."
        )
    if (expected_corpus_fingerprint
            and meta.get("corpus_fingerprint") != expected_corpus_fingerprint):
        raise RuntimeError("The active corpus changed -- rebuild the token cache.")
    try:
        train = np.load(cache_dir / "train_tokens.npy", allow_pickle=False)
        val = np.load(cache_dir / "val_tokens.npy", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Token cache is incomplete or corrupt: {exc}") from exc
    if len(train) != meta.get("train_tokens") or len(val) != meta.get("val_tokens"):
        raise RuntimeError("Token cache sizes do not match metadata -- rebuild the cache.")
    return train, val, meta
