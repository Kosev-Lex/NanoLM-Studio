"""NanoLM Studio v4 trainer.

The trainer runs in a worker thread and communicates with the UI only through
an event queue.

Event types put on the queue (all dicts with a "type" key):
  log     {text}
  metric  {step, loss, lr, tok_s, tokens_seen}        # rolling train loss
  val     {step, train_loss, val_loss, gap, ppl, best} # fixed eval samples
  sample  {step, prompt, text}
  done    {reason, best_val, steps}
"""
from __future__ import annotations

import json
import math
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import CKPT_DIR, RUNS_DIR, ModelConfig, TrainConfig
from .model import NanoLM, generate_stream
from .tokenization import NanoTokenizer

BEST_CKPT = CKPT_DIR / "best.pt"
FINAL_CKPT = CKPT_DIR / "final.pt"


def _nvidia_gpu_detected() -> bool:
    """Return whether the NVIDIA driver reports at least one physical GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def training_device() -> str:
    """Select CUDA, or reject a broken CPU-only install on NVIDIA systems."""
    if torch.cuda.is_available():
        return "cuda"
    if _nvidia_gpu_detected():
        cuda_build = torch.version.cuda or "None (CPU-only build)"
        raise RuntimeError(
            "An NVIDIA GPU is installed, but this PyTorch installation cannot "
            f"use CUDA (Torch {torch.__version__}, CUDA build {cuda_build}).\n\n"
            "Activate NanoLM's .venv and run:\n"
            "python -m pip uninstall -y torch\n"
            "python -m pip install torch==2.12.1 "
            "--index-url https://download.pytorch.org/whl/cu126"
        )
    return "cpu"


def training_device_summary() -> str:
    if torch.cuda.is_available():
        return f"CUDA — {torch.cuda.get_device_name(0)}"
    if torch.version.cuda is None:
        return f"CPU only — Torch {torch.__version__} has no CUDA support"
    return f"CPU — Torch CUDA {torch.version.cuda} is unavailable"


def lr_schedule(opt_step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    """Linear warmup -> cosine decay, clamped."""
    if warmup > 0 and opt_step <= warmup:
        return lr_max * opt_step / warmup
    t = (opt_step - warmup) / max(1, total - warmup)
    t = min(1.0, max(0.0, t))
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def get_batch(rng: np.random.Generator, arr: np.ndarray, block: int,
              batch: int, device: str):
    """Random crops from a flat token array.  Plain numpy slicing -- no
    DataLoader workers, hence no Windows spawn/duplicate-seed issues."""
    last_start = len(arr) - block - 1
    if last_start < 0:
        raise ValueError("token array is shorter than one training sequence")
    starts = rng.integers(0, last_start + 1, size=batch)
    x = np.stack([arr[s:s + block] for s in starts]).astype(np.int64)
    y = np.stack([arr[s + 1:s + block + 1] for s in starts]).astype(np.int64)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    if device == "cuda":
        return xt.pin_memory().to(device, non_blocking=True), \
               yt.pin_memory().to(device, non_blocking=True)
    return xt.to(device), yt.to(device)


class Trainer:
    def __init__(
        self,
        model_cfg: ModelConfig,
        tcfg: TrainConfig,
        train_arr: np.ndarray,
        val_arr: np.ndarray,
        tokenizer: NanoTokenizer,
        events: queue.Queue,
        stop_event: threading.Event,
        tokenizer_fingerprint: str = "",
        checkpoint_dir: Path = CKPT_DIR,
        runs_dir: Path = RUNS_DIR,
    ):
        if len(train_arr) < model_cfg.block_size + 1:
            raise RuntimeError("Training corpus smaller than one context window.")
        if len(val_arr) < model_cfg.block_size + 1:
            raise RuntimeError("Validation split smaller than one context window.")

        self.mcfg = model_cfg
        self.tcfg = tcfg
        self.train_arr = train_arr
        self.val_arr = val_arr
        self.tok = tokenizer
        self.events = events
        self.stop_event = stop_event
        self.fp = tokenizer_fingerprint
        self.checkpoint_dir = Path(checkpoint_dir)
        self.runs_dir = Path(runs_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.checkpoint_dir / "best.pt"
        self.final_path = self.checkpoint_dir / "final.pt"

        self.device = training_device()
        self.use_amp = self.device == "cuda"

        torch.manual_seed(tcfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(tcfg.seed)
        self.model = NanoLM(model_cfg).to(self.device)
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            (no_decay if param.ndim < 2 or name.endswith("bias") else decay).append(param)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": tcfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=tcfg.lr_max,
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.step = 0                       # optimizer-update steps
        self.best_val = float("inf")
        self.tokens_seen = 0
        self.rng = np.random.default_rng(tcfg.seed)
        self.since_improve = 0
        self.best_written = False

        # persisted history for this run
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.run_path = self.runs_dir / f"run_{stamp}_{time.time_ns() % 1_000_000:06d}.jsonl"

        if tcfg.resume:
            self._resume()

    # -------------------- helpers --------------------
    def _emit(self, ev: dict):
        self.events.put(ev)

    def _log(self, text: str):
        self._emit({"type": "log", "text": text})

    def _record(self, row: dict):
        with self.run_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _save(self, path: Path, with_optimizer: bool):
        extras = {
            "step": self.step,
            "best_val": self.best_val,
            "tokens_seen": self.tokens_seen,
            "tokenizer_fingerprint": self.fp,
            "train_config": self.tcfg.to_dict(),
            "since_improve": self.since_improve,
            "numpy_rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            extras["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        if with_optimizer:
            extras["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                extras["scaler"] = self.scaler.state_dict()
        self.model.save_checkpoint(path, extras)

    def _resume(self):
        if not self.final_path.exists():
            self._log("[resume] no final.pt found; starting fresh")
            return
        payload = torch.load(str(self.final_path), map_location=self.device, weights_only=False)
        saved_cfg = ModelConfig.from_dict(payload.get("config", {}))
        if saved_cfg.to_dict() != self.mcfg.to_dict():
            raise RuntimeError(
                "Resume refused: checkpoint architecture differs from the "
                f"selected preset ({saved_cfg.to_dict()} vs {self.mcfg.to_dict()})."
            )
        if self.fp and payload.get("tokenizer_fingerprint") not in ("", None, self.fp):
            raise RuntimeError(
                "Resume refused: checkpoint was trained with a different tokenizer."
            )
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.scaler is not None and payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        self.step = int(payload.get("step", 0))
        self.best_val = float(payload.get("best_val", float("inf")))
        self.tokens_seen = int(payload.get("tokens_seen", 0))
        self.since_improve = int(payload.get("since_improve", 0))
        if payload.get("numpy_rng_state"):
            self.rng.bit_generator.state = payload["numpy_rng_state"]
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state"]])
        self.best_written = self.best_path.exists()
        self._log(f"[resume] step={self.step} best_val={self.best_val:.4f}")

    # -------------------- validation --------------------
    @torch.no_grad()
    def evaluate_loss(self, token_array: np.ndarray, *, seed: int) -> float:
        """Evaluate fixed crops without dropout for comparable loss readings.

        Using the same array and seed at every evaluation makes changes across
        checkpoints meaningful.  Training and validation are both measured by
        this method so their difference is a genuine like-for-like gap rather
        than a comparison with noisy optimization batches.
        """
        was_training = self.model.training
        self.model.eval()
        eval_rng = np.random.default_rng(seed)
        losses = []
        for _ in range(self.tcfg.val_batches):
            if self.stop_event.is_set():
                break
            x, y = get_batch(eval_rng, token_array, self.mcfg.block_size,
                             self.tcfg.batch_size, self.device)
            _, loss = self.model(x, y)
            losses.append(loss.item())
        self.model.train(was_training)
        if not losses:
            return float("nan")
        return float(np.mean(losses))

    def validate(self) -> float:
        """Backward-compatible validation entry point."""
        return self.evaluate_loss(self.val_arr, seed=12345)

    # -------------------- sampling during training --------------------
    @torch.no_grad()
    def _sample(self):
        if self.tcfg.sample_prompt.strip():
            prompt = self.tcfg.sample_prompt.strip()
            ids = self.tok.encode(prompt)
        else:
            # seed with a random validation snippet so progress is visible
            srng = np.random.default_rng(self.tcfg.seed + self.step)
            s = int(srng.integers(0, len(self.val_arr) - 12))
            ids = [int(t) for t in self.val_arr[s:s + 10]]
            prompt = self.tok.decode(ids)
        out_ids = list(generate_stream(
            self.model, ids, max_new_tokens=self.tcfg.sample_tokens,
            temperature=0.8, top_k=50, top_p=0.95,
            repetition_penalty=1.15, eos_id=self.tok.eos_id,
            stop_check=self.stop_event.is_set,
        ))
        text = self.tok.decode(out_ids)
        self._emit({"type": "sample", "step": self.step, "prompt": prompt, "text": text})
        self._record({"kind": "sample", "step": self.step, "prompt": prompt, "text": text})

    # -------------------- main loop --------------------
    def train(self):
        c, m = self.tcfg, self.mcfg
        self._log(
            f"[train] device={self.device} params={self.model.num_params/1e6:.2f}M "
            f"ctx={m.block_size} vocab={m.vocab_size}"
        )
        self._record({"kind": "start", "model": m.to_dict(), "train": c.to_dict(),
                      "params": self.model.num_params, "device": self.device})

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        tokens_at_t0 = self.tokens_seen
        loss_window: list[float] = []
        reason = "completed"
        failed = False

        try:
            while self.step < c.total_steps:
                if self.stop_event.is_set():
                    reason = "stopped by user"
                    break
                self.step += 1

                lr = lr_schedule(self.step, c.warmup_steps, c.total_steps, c.lr_max, c.lr_min)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                micro_losses: list[float] = []
                cancelled = False
                for _ in range(c.grad_accum):
                    if self.stop_event.is_set():
                        cancelled = True
                        break
                    x, y = get_batch(self.rng, self.train_arr, m.block_size,
                                     c.batch_size, self.device)
                    if self.use_amp:
                        with torch.amp.autocast("cuda"):
                            _, loss = self.model(x, y)
                    else:
                        _, loss = self.model(x, y)
                    micro_losses.append(float(loss.item()))
                    scaled_loss = loss / c.grad_accum
                    if self.use_amp:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    self.tokens_seen += c.batch_size * m.block_size

                if cancelled:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step -= 1
                    reason = "stopped by user"
                    break
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                loss_window.append(float(np.mean(micro_losses)))

                if self.step % c.log_interval == 0:
                    dt = max(time.time() - t0, 1e-6)
                    tok_s = int((self.tokens_seen - tokens_at_t0) / dt)
                    mean_loss = float(np.mean(loss_window))
                    loss_window.clear()
                    self._emit({"type": "metric", "step": self.step, "loss": mean_loss,
                                "lr": lr, "tok_s": tok_s, "tokens_seen": self.tokens_seen})
                    self._record({"kind": "metric", "step": self.step, "loss": mean_loss,
                                  "lr": lr, "tokens_seen": self.tokens_seen,
                                  "time": time.time()})

                if self.step % c.val_interval == 0:
                    # Measure both splits under identical inference conditions.
                    # This is the overfit gap shown by the UI and run history.
                    train_loss = self.evaluate_loss(self.train_arr, seed=54321)
                    if self.stop_event.is_set():
                        reason = "stopped by user"
                        break
                    val_loss = self.evaluate_loss(self.val_arr, seed=12345)
                    if self.stop_event.is_set():
                        reason = "stopped by user"
                        break
                    gap = val_loss - train_loss
                    ppl = math.exp(min(val_loss, 20.0))
                    improved = val_loss + 1e-6 < self.best_val
                    if improved:
                        self.best_val = val_loss
                        self.since_improve = 0
                        self._save(self.best_path, with_optimizer=False)
                        self.best_written = True
                    else:
                        self.since_improve += 1
                    self._emit({"type": "val", "step": self.step,
                                "train_loss": train_loss, "val_loss": val_loss,
                                "gap": gap, "ppl": ppl, "best": improved})
                    self._record({"kind": "val", "step": self.step,
                                  "train_loss": train_loss, "val_loss": val_loss,
                                  "gap": gap, "ppl": ppl})
                    self._save(self.final_path, with_optimizer=True)
                    if self.since_improve >= c.patience:
                        reason = f"early stop (no val improvement for {c.patience} checks)"
                        break

                if c.sample_interval > 0 and self.step % c.sample_interval == 0:
                    self._sample()

        except Exception as e:
            reason = f"error: {e}"
            failed = True
            self._log(f"[train ERROR] {e}")
        finally:
            try:
                self._save(self.final_path, with_optimizer=True)
                if not self.best_written and not failed:
                    self._save(self.best_path, with_optimizer=False)
            except Exception as save_error:
                reason = f"{reason}; checkpoint save failed: {save_error}"
                self._log(f"[checkpoint ERROR] {save_error}")
            self._record({"kind": "done", "reason": reason, "step": self.step,
                          "best_val": self.best_val})
            self._emit({"type": "done", "reason": reason,
                        "best_val": self.best_val, "steps": self.step})
