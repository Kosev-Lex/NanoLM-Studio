"""Safe end-to-end smoke test for NanoLM Studio v4 (excluding Tk).

Every artifact is created below a TemporaryDirectory. Application data is
never read, deleted, or overwritten.
"""
from __future__ import annotations

import queue
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch

from nanolm.config import ModelConfig, TrainConfig
from nanolm.corpus import CorpusLibrary, clean_text
from nanolm.model import NanoLM, generate_stream
from nanolm.tokenization import NanoTokenizer, build_token_cache, load_token_cache
from nanolm.training import Trainer


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}" + (f" -- {detail}" if detail else ""))


def main() -> None:
    cleaned = clean_text("real text\nexam-\nple and a m illion words")
    check("cleaning pipeline", "example" in cleaned and "million" in cleaned, cleaned)

    with tempfile.TemporaryDirectory(prefix="nanolm_v4_smoke_") as temp_name:
        root = Path(temp_name)
        source_dir = root / "sources"
        source_dir.mkdir()
        library = CorpusLibrary(root / "corpus.db")

        base = ("The quick brown fox explores a compact language model. " * 80) + "\n"
        for index in range(6):
            source = source_dir / f"document_{index}.txt"
            source.write_text(base + (f"Topic {index} has unique details. " * 80),
                              encoding="utf-8")
            row, _ = library.add_document(source)
            check(f"import document {index}", row is not None)
        duplicate, message = library.add_document(source_dir / "document_0.txt")
        check("duplicate detection", duplicate is None and "duplicate" in message)

        tokenizer = NanoTokenizer(root / "tokenizer" / "tokenizer.json")
        tokenizer.train_from_texts(
            [library.get_text(doc.id) for doc in library.active_documents()],
            vocab_size=512,
        )
        check("fixed special IDs", tokenizer.bos_id == 2 and tokenizer.eos_id == 3)

        cache_dir = root / "cache"
        meta = build_token_cache(library, tokenizer, cache_dir=cache_dir)
        train, val, _ = load_token_cache(
            tokenizer.fingerprint(),
            expected_corpus_fingerprint=library.corpus_fingerprint(),
            cache_dir=cache_dir,
        )
        check("document-aware cache", len(train) == meta["train_tokens"] and len(val) > 64)
        library.set_active(library.active_documents()[-1].id, False)
        try:
            load_token_cache(
                tokenizer.fingerprint(),
                expected_corpus_fingerprint=library.corpus_fingerprint(),
                cache_dir=cache_dir,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("corpus change did not invalidate token cache")
        library.set_active(library.list_documents()[-1].id, True)

        model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=32,
                                dim=32, n_layers=2, n_heads=4, dropout=0.0)
        model = NanoLM(model_cfg)
        x = torch.randint(0, model_cfg.vocab_size, (2, model_cfg.block_size))
        logits, loss = model(x, x)
        check("model forward", logits.shape == (2, 32, tokenizer.vocab_size)
              and np.isfinite(loss.item()))
        expected_params = sum(param.numel() for param in model.parameters())
        check("parameter count", model.num_params == expected_params)

        prefix = x[:1, :12]
        full_logits, _ = model(prefix)
        cached_logits, cache = model.forward_cached(prefix[:, :8])
        next_logits, _ = model.forward_cached(prefix[:, 8:], cache)
        check("KV cache equivalence",
              torch.allclose(full_logits[:, 8:], next_logits, atol=1e-5))

        events: queue.Queue = queue.Queue()
        train_cfg = TrainConfig(
            total_steps=4, batch_size=2, grad_accum=2, warmup_steps=1,
            val_interval=2, val_batches=1, log_interval=1,
            sample_interval=0, patience=10,
        )
        trainer = Trainer(
            model_cfg, train_cfg, train, val, tokenizer, events,
            threading.Event(), tokenizer.fingerprint(),
            checkpoint_dir=root / "checkpoints", runs_dir=root / "runs",
        )
        trainer.train()
        check("optimizer-step semantics", trainer.step == 4)
        check("isolated checkpoints", (root / "checkpoints" / "best.pt").exists()
              and (root / "checkpoints" / "final.pt").exists())
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        eval_events = [event for event in emitted if event.get("type") == "val"]
        check("explicit overfit-gap metrics",
              bool(eval_events)
              and all({"train_loss", "val_loss", "gap"} <= event.keys()
                      for event in eval_events)
              and all(np.isclose(event["gap"],
                                 event["val_loss"] - event["train_loss"])
                      for event in eval_events))

        generated = list(generate_stream(
            trainer.model, [], max_new_tokens=4, temperature=0,
            top_k=0, top_p=1.0, eos_id=None,
        ))
        check("empty-prompt greedy generation", len(generated) == 4)

    print("\nALL V4 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
