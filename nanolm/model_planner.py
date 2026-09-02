"""Model/corpus/VRAM planning logic and its Tk configuration window.

The estimates are deliberately conservative guidance, not allocation promises.
They model NanoLM Studio's current FP32 AdamW state, gradient storage, a
temporary optimizer peak, mixed-precision activations, attention workspace,
logits, CUDA overhead, and allocator fragmentation.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from .config import DATA_DIR, MODEL_PRESETS

GIB = 1024 ** 3
SETTINGS_PATH = DATA_DIR / "studio_settings.json"
RANK = {"good": 0, "caution": 1, "poor": 2}
COLOURS = {
    "good": ("#dff6e5", "#176b36"),
    "caution": ("#fff1c7", "#8a5a00"),
    "poor": ("#ffe0e0", "#a32121"),
}


@dataclass(frozen=True)
class PresetAssessment:
    name: str
    n_layers: int
    dim: int
    n_heads: int
    block_size: int
    parameters: int
    peak_vram_gib: float
    safe_batch: int
    vram_state: str
    vram_label: str
    tokens_per_parameter: float
    corpus_state: str
    corpus_label: str
    overall_state: str


def exact_parameter_count(vocab_size: int, preset: tuple[int, int, int, int]) -> int:
    """Exact NanoLM count, including biases and norms but only one tied embedding."""
    n_layers, dim, _n_heads, block_size = preset
    per_layer = 12 * dim * dim + 9 * dim
    return vocab_size * dim + block_size * dim + n_layers * per_layer + 2 * dim


def estimate_peak_vram_gib(vocab_size: int, preset: tuple[int, int, int, int],
                           batch_size: int) -> float:
    """Estimate training peak for the current AdamW/AMP implementation."""
    n_layers, dim, n_heads, block_size = preset
    parameters = exact_parameter_count(vocab_size, preset)
    parameter_state = parameters * 20  # weights, grads, Adam moments, step peak
    activations = batch_size * block_size * dim * n_layers * 24
    attention = batch_size * n_heads * block_size * block_size * n_layers * 2
    logits = batch_size * block_size * vocab_size * 4
    framework_reserve = int(0.75 * GIB)
    return 1.15 * (parameter_state + activations + attention + logits + framework_reserve) / GIB


def corpus_fit(tokens: int, parameters: int) -> tuple[str, str, float]:
    ratio = tokens / max(1, parameters)
    if ratio < 3:
        return "poor", "Too little data", ratio
    if ratio < 10:
        return "caution", "Data-limited", ratio
    if ratio <= 30:
        return "good", "Ideal range", ratio
    if ratio <= 60:
        return "caution", "Model may be small", ratio
    return "caution", "Strongly under-capacity", ratio


def vram_fit(required_gib: float, available_gib: float) -> tuple[str, str]:
    if available_gib <= 0:
        return "poor", "No GPU budget"
    fraction = required_gib / available_gib
    if fraction <= 0.70:
        return "good", "Comfortable"
    if fraction <= 0.88:
        return "caution", "Tight"
    return "poor", "Likely OOM"


def largest_safe_batch(vocab_size: int, preset: tuple[int, int, int, int],
                       available_gib: float) -> int:
    for batch in (32, 16, 8, 4, 2, 1):
        if estimate_peak_vram_gib(vocab_size, preset, batch) <= available_gib * 0.80:
            return batch
    return 0


def assess_presets(vocab_size: int, corpus_tokens: int, available_gib: float,
                   trial_batch: int) -> list[PresetAssessment]:
    assessments = []
    for name, preset in MODEL_PRESETS.items():
        n_layers, dim, n_heads, block_size = preset
        parameters = exact_parameter_count(vocab_size, preset)
        peak = estimate_peak_vram_gib(vocab_size, preset, trial_batch)
        v_state, v_label = vram_fit(peak, available_gib)
        c_state, c_label, ratio = corpus_fit(corpus_tokens, parameters)
        overall = max((v_state, c_state), key=RANK.get)
        assessments.append(PresetAssessment(
            name, n_layers, dim, n_heads, block_size, parameters, peak,
            largest_safe_batch(vocab_size, preset, available_gib),
            v_state, v_label, ratio, c_state, c_label, overall,
        ))
    return assessments


def recommended_assessment(rows: list[PresetAssessment]) -> PresetAssessment:
    """Prefer a VRAM-safe model closest to 20 corpus tokens per parameter."""
    candidates = [row for row in rows if row.safe_batch > 0]
    if not candidates:
        return rows[0]
    return min(
        candidates,
        key=lambda row: (
            RANK[row.corpus_state],
            abs(math.log(max(row.tokens_per_parameter, 0.01) / 20.0)),
            -row.parameters,
        ),
    )


def recommended_training_settings(row: PresetAssessment, corpus_tokens: int) -> dict:
    batch = max(1, row.safe_batch)
    target_tokens_per_update = 8192
    grad_accum = max(1, math.ceil(target_tokens_per_update / (batch * row.block_size)))
    effective_tokens = batch * row.block_size * grad_accum
    target_training_tokens = min(corpus_tokens * 3, row.parameters * 20)
    total_steps = max(100, math.ceil(target_training_tokens / effective_tokens))
    if row.parameters <= 50_000_000:
        lr_max = 3e-4
    elif row.parameters <= 150_000_000:
        lr_max = 2e-4
    elif row.parameters <= 300_000_000:
        lr_max = 1.5e-4
    else:
        lr_max = 1e-4
    return {
        "preset": row.name,
        "batch_size": batch,
        "grad_accum": grad_accum,
        "total_steps": total_steps,
        "lr_max": lr_max,
        "lr_min": lr_max * 0.1,
        "warmup_steps": max(10, min(1000, round(total_steps * 0.05))),
        "val_interval": max(50, total_steps // 20),
        "sample_interval": max(100, total_steps // 10),
    }


def load_planner_settings() -> dict:
    try:
        value = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_planner_settings(vram_gib: float, trial_batch: int) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".studio_settings.", suffix=".json",
                                     dir=SETTINGS_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"vram_gib": vram_gib, "trial_batch": trial_batch}, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, SETTINGS_PATH)
    finally:
        Path(temp_name).unlink(missing_ok=True)


class ModelPlannerWindow(tk.Toplevel):
    """Interactive visual comparison of model size, corpus size, and VRAM."""

    def __init__(self, parent: tk.Misc, *, vocab_size: int, corpus_tokens: int,
                 token_source: str, current_preset: str,
                 apply_callback: Callable[[dict], bool]):
        super().__init__(parent)
        self.title("Model, Corpus & VRAM Planner")
        self.geometry("1180x760")
        self.minsize(980, 680)
        self.transient(parent)
        self.vocab_size = vocab_size
        self.library_tokens = max(0, corpus_tokens)
        self.token_source = token_source
        self.current_preset = current_preset
        self.apply_callback = apply_callback
        self.rows: dict[str, PresetAssessment] = {}

        saved = load_planner_settings()
        try:
            saved_vram = float(saved.get("vram_gib", 24.0))
            saved_batch = int(saved.get("trial_batch", 16))
        except (TypeError, ValueError):
            saved_vram, saved_batch = 24.0, 16
        if not 0.5 <= saved_vram <= 256 or not 1 <= saved_batch <= 64:
            saved_vram, saved_batch = 24.0, 16
        self.vram_var = tk.DoubleVar(value=saved_vram)
        self.tokens_var = tk.StringVar(value=str(self.library_tokens))
        self.batch_var = tk.IntVar(value=saved_batch)
        self.summary_var = tk.StringVar()
        self.detail_var = tk.StringVar()

        self._build()
        self._recalculate(select_name=current_preset)
        self.grab_set()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        ttk.Label(self, text="Model, Corpus & VRAM Planner",
                  font=("Segoe UI", 15, "bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 4))

        inputs = ttk.Labelframe(self, text="Your hardware and corpus")
        inputs.grid(row=1, column=0, sticky="ew", padx=14, pady=6)
        for column in (1, 3, 5):
            inputs.columnconfigure(column, weight=1)
        ttk.Label(inputs, text="Available VRAM (GiB):").grid(row=0, column=0, padx=(10, 5), pady=10)
        ttk.Entry(inputs, textvariable=self.vram_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(inputs, text="Active corpus tokens:").grid(row=0, column=2, padx=(16, 5))
        ttk.Entry(inputs, textvariable=self.tokens_var, width=16).grid(row=0, column=3, sticky="w")
        ttk.Label(inputs, text="Trial micro-batch:").grid(row=0, column=4, padx=(16, 5))
        ttk.Spinbox(inputs, textvariable=self.batch_var, from_=1, to=64,
                    width=7).grid(row=0, column=5, sticky="w")
        ttk.Button(inputs, text="Use Library Value", command=self._use_library_tokens).grid(
            row=0, column=6, padx=(12, 5))
        ttk.Button(inputs, text="Recalculate", style="Accent.TButton",
                   command=self._recalculate).grid(row=0, column=7, padx=(5, 10))
        ttk.Label(inputs, text=f"Corpus source: {self.token_source} | Vocabulary: {self.vocab_size:,}",
                  foreground="#666666").grid(row=1, column=0, columnspan=8,
                                               sticky="w", padx=10, pady=(0, 8))

        centre = ttk.Frame(self)
        centre.grid(row=2, column=0, sticky="nsew", padx=14, pady=6)
        centre.columnconfigure(0, weight=1)
        centre.rowconfigure(0, weight=3)
        centre.rowconfigure(2, weight=2)

        columns = ("preset", "params", "context", "peak", "safe_batch",
                   "vram", "ratio", "corpus", "verdict")
        self.tree = ttk.Treeview(centre, columns=columns, show="headings",
                                 selectmode="browse", height=9)
        headings = {
            "preset": "Preset", "params": "Parameters", "context": "Context",
            "peak": "Peak VRAM*", "safe_batch": "Safe batch", "vram": "VRAM fit",
            "ratio": "Tokens / param", "corpus": "Corpus fit", "verdict": "Verdict",
        }
        widths = {"preset": 220, "params": 90, "context": 70, "peak": 90,
                  "safe_batch": 75, "vram": 95, "ratio": 95, "corpus": 145,
                  "verdict": 85}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(centre, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        for state, (background, foreground) in COLOURS.items():
            self.tree.tag_configure(state, background=background, foreground=foreground)
        self.tree.bind("<<TreeviewSelect>>", self._show_selected)
        self.tree.bind("<Double-1>", lambda _event: self._apply_selected())

        legend = ttk.Frame(centre)
        legend.grid(row=1, column=0, sticky="ew", pady=(6, 3))
        for column, (state, label) in enumerate((
                ("good", "Good match"), ("caution", "Usable with caution"),
                ("poor", "Poor setting / likely failure"))):
            bg, fg = COLOURS[state]
            tk.Label(legend, text=f"  {label}  ", background=bg, foreground=fg)\
                .grid(row=0, column=column, padx=(0, 8))

        visual = ttk.Labelframe(centre, text="Selected preset — visual fit")
        visual.grid(row=2, column=0, sticky="nsew", pady=(4, 0))
        visual.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(visual, height=150, background="white", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        ttk.Label(visual, textvariable=self.detail_var, wraplength=1080,
                  justify="left").grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 8))

        bottom = ttk.Frame(self)
        bottom.grid(row=3, column=0, sticky="ew", padx=14, pady=(4, 12))
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self.summary_var, font=("Segoe UI", 10, "bold"),
                  wraplength=760).grid(row=0, column=0, sticky="w")
        ttk.Button(bottom, text="Apply Selected Preset & Settings",
                   style="Accent.TButton", command=self._apply_selected).grid(
            row=0, column=1, padx=(8, 6))
        ttk.Button(bottom, text="Close", command=self.destroy).grid(row=0, column=2)

    def _read_inputs(self) -> tuple[float, int, int] | None:
        try:
            vram = float(self.vram_var.get())
            tokens = int(self.tokens_var.get().replace(",", "").strip())
            batch = int(self.batch_var.get())
            if not 0.5 <= vram <= 256:
                raise ValueError("VRAM must be between 0.5 and 256 GiB.")
            if tokens < 1:
                raise ValueError("Corpus tokens must be at least 1.")
            if not 1 <= batch <= 64:
                raise ValueError("Trial micro-batch must be between 1 and 64.")
            return vram, tokens, batch
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Planner settings", str(exc), parent=self)
            return None

    def _use_library_tokens(self) -> None:
        self.tokens_var.set(str(self.library_tokens))
        self._recalculate()

    def _recalculate(self, select_name: str | None = None) -> None:
        values = self._read_inputs()
        if values is None:
            return
        vram, tokens, batch = values
        save_planner_settings(vram, batch)
        assessments = assess_presets(self.vocab_size, tokens, vram, batch)
        self.rows = {row.name: row for row in assessments}
        self.tree.delete(*self.tree.get_children())
        for row in assessments:
            verdict = {"good": "GOOD", "caution": "CAUTION", "poor": "POOR"}[row.overall_state]
            self.tree.insert("", "end", iid=row.name, tags=(row.overall_state,), values=(
                row.name.split("(", 1)[0].strip(), f"{row.parameters/1e6:.1f}M",
                row.block_size, f"{row.peak_vram_gib:.1f} GiB",
                row.safe_batch or "none", row.vram_label,
                f"{row.tokens_per_parameter:.1f}", row.corpus_label, verdict,
            ))
        recommended = recommended_assessment(assessments)
        desired = select_name if select_name in self.rows else recommended.name
        self.tree.selection_set(desired)
        self.tree.focus(desired)
        self.tree.see(desired)
        self.summary_var.set(
            f"Recommended match: {recommended.name.split('(', 1)[0].strip()} — "
            f"{recommended.parameters/1e6:.1f}M parameters, safe micro-batch "
            f"{recommended.safe_batch or 'none'} on the entered VRAM."
        )
        self._show_selected()

    def _show_selected(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        values = self._read_inputs()
        if values is None:
            return
        available, tokens, _trial_batch = values
        row = self.rows[selection[0]]
        self.canvas.delete("all")
        width = max(760, self.canvas.winfo_width() - 40)
        left, right = 170, width
        self._draw_meter(25, "Estimated VRAM", row.peak_vram_gib, available,
                         row.vram_state, f"{row.peak_vram_gib:.1f} / {available:.1f} GiB",
                         left, right)
        self._draw_corpus_meter(90, row.tokens_per_parameter, left, right)
        advice = recommended_training_settings(row, tokens)
        vram_advice = (
            f"At trial batch {self.batch_var.get()}, VRAM is {row.vram_label.lower()}. "
            f"Largest conservative batch: {row.safe_batch or 'none'}."
        )
        corpus_advice = {
            "good": "The corpus is in the preferred 10–30 tokens-per-parameter range.",
            "caution": "This can train, but the model/corpus balance is outside the preferred range.",
            "poor": "The model is much too large for this corpus and is likely to memorise it.",
        }[row.corpus_state]
        self.detail_var.set(
            f"{vram_advice} {corpus_advice} Suggested application settings: "
            f"batch {advice['batch_size']}, accumulation {advice['grad_accum']}, "
            f"{advice['total_steps']:,} optimizer steps, LR {advice['lr_max']:.1e}. "
            "*VRAM is an estimate; close other GPU applications and leave headroom."
        )

    def _draw_meter(self, y: int, label: str, value: float, maximum: float,
                    state: str, text: str, left: int, right: int) -> None:
        self.canvas.create_text(10, y + 10, text=label, anchor="w", font=("Segoe UI", 10, "bold"))
        self.canvas.create_rectangle(left, y, right, y + 22, fill="#eceff3", outline="#aab0b8")
        fraction = min(1.0, value / max(maximum, 0.01))
        self.canvas.create_rectangle(left, y, left + (right - left) * fraction, y + 22,
                                     fill=COLOURS[state][0], outline="")
        self.canvas.create_text((left + right) / 2, y + 11, text=text, anchor="center")

    def _draw_corpus_meter(self, y: int, ratio: float, left: int, right: int) -> None:
        self.canvas.create_text(10, y + 10, text="Corpus balance", anchor="w",
                                font=("Segoe UI", 10, "bold"))
        spans = ((0, 3, "poor"), (3, 10, "caution"), (10, 30, "good"),
                 (30, 60, "caution"))
        for start, end, state in spans:
            x1 = left + (right - left) * start / 60
            x2 = left + (right - left) * end / 60
            self.canvas.create_rectangle(x1, y, x2, y + 22, fill=COLOURS[state][0],
                                         outline="#aab0b8")
        marker = left + (right - left) * min(60, ratio) / 60
        self.canvas.create_line(marker, y - 5, marker, y + 28, fill="#111111", width=3)
        self.canvas.create_text(marker, y + 40, text=f"{ratio:.1f} tokens/parameter",
                                anchor="center")
        self.canvas.create_text(left, y + 40, text="0", anchor="center")
        self.canvas.create_text(right, y + 40, text="60+", anchor="center")

    def _apply_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        values = self._read_inputs()
        if values is None:
            return
        _vram, tokens, _batch = values
        row = self.rows[selection[0]]
        if row.safe_batch == 0:
            messagebox.showerror(
                "Preset does not fit",
                "Even micro-batch 1 exceeds the conservative VRAM budget. "
                "Choose a smaller preset or enter the correct available VRAM.",
                parent=self,
            )
            return
        if row.corpus_state == "poor" and not messagebox.askyesno(
                "Poor corpus match",
                "This preset has fewer than 3 corpus tokens per parameter and is "
                "likely to memorise rather than generalise. Apply it anyway?",
                parent=self):
            return
        settings = recommended_training_settings(row, tokens)
        if self.apply_callback(settings):
            self.destroy()
