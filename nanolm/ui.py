"""NanoLM Studio v4 -- desktop UI.

Threading contract (the one rule): worker threads NEVER touch Tk.
Everything flows through self.events (a queue.Queue) and is applied by
the UI thread in _poll_events().
"""
from __future__ import annotations

import math
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

import torch

from .config import (CKPT_DIR, DEFAULT_PRESET, MODEL_PRESETS, ModelConfig,
                     TrainConfig, ensure_data_dirs)
from .corpus import CorpusLibrary, ExtractionError, iter_supported_files
from .model import NanoLM, generate_stream
from .model_planner import ModelPlannerWindow, exact_parameter_count
from .tokenization import NanoTokenizer, build_token_cache, load_token_cache
from .training import (BEST_CKPT, FINAL_CKPT, Trainer, training_device,
                       training_device_summary)

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    Figure = None
    FigureCanvasTkAgg = None

MONO = ("Consolas", 10)
TITLE = ("Segoe UI", 14, "bold")
SUB = ("Segoe UI", 10, "bold")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        ensure_data_dirs()
        self.title("NanoLM Studio v4")
        self.geometry("1280x880")
        self.minsize(1100, 760)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._closing = False

        # ---- shared state ----
        self.library = CorpusLibrary()
        self.tok = NanoTokenizer()
        self.loaded_model: NanoLM | None = None
        self.loaded_payload: dict = {}
        self.events: queue.Queue = queue.Queue()

        self.train_thread: threading.Thread | None = None
        self.train_stop = threading.Event()
        self.tok_thread: threading.Thread | None = None
        self.chat_thread: threading.Thread | None = None
        self.chat_stop = threading.Event()
        self.chat_history: list[tuple[str, str]] = []   # (user_text, model_text)

        # chart data
        self.h_steps: list[int] = []
        self.h_loss: list[float] = []
        self.e_steps: list[int] = []
        self.e_train_loss: list[float] = []
        self.v_steps: list[int] = []
        self.v_loss: list[float] = []
        self.chart_dirty = False
        self.lib_sort_column = "id"
        self.lib_sort_reverse = False
        self.glass_figures: list = []
        self.planner_window: ModelPlannerWindow | None = None

        self._style()
        self._build_tabs()
        self._refresh_library()
        self._refresh_tok_status()
        self.after(80, self._poll_events)

    # ==================================================================
    # STYLE / TABS
    # ==================================================================
    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", font=("Segoe UI", 10))
        s.configure("TNotebook.Tab", padding=(16, 8), font=SUB)
        s.configure("Accent.TButton", padding=(10, 6))
        s.configure("Treeview", rowheight=25)
        s.configure("Treeview.Heading", font=SUB)

    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_lib = ttk.Frame(self.nb)
        self.tab_tok = ttk.Frame(self.nb)
        self.tab_train = ttk.Frame(self.nb)
        self.tab_chat = ttk.Frame(self.nb)
        self.tab_glass = ttk.Frame(self.nb)
        self.nb.add(self.tab_lib, text="  Library  ")
        self.nb.add(self.tab_tok, text="  Tokenizer  ")
        self.nb.add(self.tab_train, text="  Training  ")
        self.nb.add(self.tab_chat, text="  Chat  ")
        self.nb.add(self.tab_glass, text="  Glass Box  ")
        self._build_library_tab()
        self._build_tokenizer_tab()
        self._build_training_tab()
        self._build_chat_tab()
        self._build_glass_tab()
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken")\
            .pack(fill="x", padx=10, pady=(0, 8))

    # ==================================================================
    # EVENT PUMP  (the only place worker results touch the UI)
    # ==================================================================
    def _poll_events(self):
        if self._closing:
            return
        deadline = time.perf_counter() + 0.025
        handled = 0
        try:
            while handled < 250 and time.perf_counter() < deadline:
                ev = self.events.get_nowait()
                handled += 1
                try:
                    self._dispatch(ev)
                except Exception as exc:
                    self.status_var.set(f"UI event error: {exc}")
        except queue.Empty:
            pass
        if self.chart_dirty:
            self._redraw_chart()
            self.chart_dirty = False
        self.after(80, self._poll_events)

    def _dispatch(self, ev: dict):
        t = ev["type"]
        if t == "log":
            self._append(self.train_log, ev["text"] + "\n")
        elif t == "metric":
            self.h_steps.append(ev["step"]); self.h_loss.append(ev["loss"])
            self.chart_dirty = True
            self.lbl_step.config(text=f"step {ev['step']:,}")
            self.lbl_loss.config(text=f"train rolling {ev['loss']:.4f}")
            self.lbl_speed.config(text=f"{ev['tok_s']:,} tok/s")
            self.lbl_tokens.config(text=f"{ev['tokens_seen']:,} tokens seen")
            self.lbl_lr.config(text=f"lr {ev['lr']:.2e}")
            self.train_prog["value"] = ev["step"]
            self._append(
                self.train_log,
                f"[train] step={ev['step']} loss={ev['loss']:.4f} "
                f"lr={ev['lr']:.2e} speed={ev['tok_s']:,} tok/s\n",
            )
        elif t == "val":
            self.e_steps.append(ev["step"]); self.e_train_loss.append(ev["train_loss"])
            self.v_steps.append(ev["step"]); self.v_loss.append(ev["val_loss"])
            self.chart_dirty = True
            star = "  *new best*" if ev["best"] else ""
            self.lbl_val.config(
                text=(f"eval train {ev['train_loss']:.4f} | val {ev['val_loss']:.4f} "
                      f"| gap {ev['gap']:+.4f} | ppl {ev['ppl']:.1f}")
            )
            self._append(self.train_log,
                         f"[eval] step={ev['step']} train={ev['train_loss']:.4f} "
                         f"val={ev['val_loss']:.4f} gap={ev['gap']:+.4f} "
                         f"val_ppl={ev['ppl']:.1f}{star}\n")
        elif t == "sample":
            self._append(self.sample_box,
                         f"--- step {ev['step']} ---\n"
                         f"prompt: {ev['prompt']!r}\n{ev['text']}\n\n")
        elif t == "done":
            best_text = f"{ev['best_val']:.4f}" if math.isfinite(ev["best_val"]) else "--"
            self._append(self.train_log,
                         f"[done] {ev['reason']} | best val {best_text} "
                         f"@ {ev['steps']} steps\n")
            self.btn_train_start.config(state="normal")
            self.btn_train_stop.config(state="disabled")
            self.btn_tok_train.config(
                state="disabled" if self.chat_thread and self.chat_thread.is_alive() else "normal")
            self.status_var.set(f"Training {ev['reason']}")
        elif t == "lib_log":
            self._append(self.lib_log, ev["text"] + "\n")
        elif t == "lib_done":
            self._refresh_library()
            self.status_var.set("Library updated")
        elif t == "tok_log":
            self._append(self.tok_log, ev["text"] + "\n")
        elif t == "tok_done":
            self._refresh_tok_status()
            self._refresh_library()
            self.btn_tok_train.config(state="normal")
            self.btn_train_start.config(state="normal")
            if self.loaded_model is not None:
                self.loaded_model = None
                self.loaded_payload = {}
                self.chat_history.clear()
                self.chat_model_lbl.config(text="no model loaded — tokenizer changed")
                self.ctx_bar.config(value=0)
                self.ctx_label.config(text="context: -- / -- tokens")
                self._chat_insert("info", "\n[tokenizer changed; previous model unloaded]\n")
            self.status_var.set("Tokenizer ready")
        elif t == "tok_err":
            self.btn_tok_train.config(state="normal")
            self.btn_train_start.config(state="normal")
            self._append(self.tok_log, f"[error] {ev['text']}\n")
            self.status_var.set("Tokenizer failed")
        elif t == "chat_replace":
            self._chat_replace_response(ev["text"])
            self._set_ctx_bar(ev.get("ctx", None))
        elif t == "chat_done":
            self.chat_history.append((ev["user"], ev["text"]))
            self._chat_insert("info", f"\n[{ev['n']} tokens, {ev['secs']:.1f}s]\n\n")
            self.btn_send.config(state="normal")
            self.btn_chat_stop.config(state="disabled")
            self.btn_chat_clear.config(state="normal")
            if not (self.train_thread and self.train_thread.is_alive()):
                self.btn_tok_train.config(state="normal")
            self._set_ctx_bar(ev.get("ctx", None))
            self.status_var.set("Generation complete")
        elif t == "chat_err":
            self._chat_insert("info", f"\n[error: {ev['text']}]\n\n")
            self.btn_send.config(state="normal")
            self.btn_chat_stop.config(state="disabled")
            self.btn_chat_clear.config(state="normal")
            if not (self.train_thread and self.train_thread.is_alive()):
                self.btn_tok_train.config(state="normal")
            self.status_var.set("Generation failed")
        elif t == "model_loaded":
            self._finish_model_load(ev)
        elif t == "model_err":
            self._set_model_buttons("normal")
            self.status_var.set("Checkpoint load failed")
            messagebox.showerror("Chat", f"Failed to load checkpoint:\n{ev['text']}")
        elif t == "glass_result":
            self.btn_glass_run.config(state="normal")
            self._render_glass(ev)
            self.status_var.set("Glass Box ready")
        elif t == "glass_err":
            self.btn_glass_run.config(state="normal")
            self.status_var.set("Glass Box failed")
            messagebox.showerror("Glass Box", ev["text"])

    @staticmethod
    def _append(widget: tk.Text, text: str):
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    # ==================================================================
    # LIBRARY TAB
    # ==================================================================
    def _build_library_tab(self):
        f = self.tab_lib
        f.columnconfigure(0, weight=3); f.columnconfigure(1, weight=2)
        f.rowconfigure(2, weight=1)

        ttk.Label(f, text="Curated Corpus Library", font=TITLE)\
            .grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))
        self.lib_stats = ttk.Label(f, text="")
        self.lib_stats.grid(row=0, column=1, sticky="e", padx=12)

        filters = ttk.Frame(f)
        filters.grid(row=1, column=0, sticky="ew", padx=(12, 6), pady=(2, 4))
        filters.columnconfigure(1, weight=1)
        ttk.Label(filters, text="Filter:").grid(row=0, column=0, padx=(0, 6))
        self.lib_filter = tk.StringVar()
        filter_entry = ttk.Entry(filters, textvariable=self.lib_filter)
        filter_entry.grid(row=0, column=1, sticky="ew")
        self.lib_active_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(filters, text="Active only", variable=self.lib_active_only,
                        command=self._refresh_library).grid(row=0, column=2, padx=(10, 0))
        self.lib_filter.trace_add("write", lambda *_: self._refresh_library())

        cols = ("id", "active", "title", "chars", "tokens", "tags")
        tree_container = ttk.Frame(f)
        tree_container.grid(row=2, column=0, sticky="nsew", padx=(12, 6), pady=6)
        tree_container.columnconfigure(0, weight=1)
        tree_container.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(tree_container, columns=cols, show="headings",
                                 selectmode="extended")
        for c, w, anchor in (("id", 50, "center"), ("active", 60, "center"),
                             ("title", 320, "w"), ("chars", 90, "e"),
                             ("tokens", 90, "e"), ("tags", 160, "w")):
            self.tree.heading(c, text=c.title(), command=lambda col=c: self._lib_sort(col))
            self.tree.column(c, width=w, minwidth=45, anchor=anchor,
                             stretch=c in {"title", "tags"})
        self.tree.grid(row=0, column=0, sticky="nsew")

        # Classic scrollbars are deliberately permanent and visually obvious.
        # Both are children of the same container as the Treeview, and both are
        # actually gridded -- the supplied version omitted the geometry call.
        self.tree_vscroll = tk.Scrollbar(tree_container, orient="vertical",
                                         command=self.tree.yview, width=18)
        self.tree_vscroll.grid(row=0, column=1, sticky="ns")
        self.tree_hscroll = tk.Scrollbar(tree_container, orient="horizontal",
                                         command=self.tree.xview, width=18)
        self.tree_hscroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=self.tree_vscroll.set,
                            xscrollcommand=self.tree_hscroll.set)
        self.tree.bind("<Double-1>", lambda _event: self._lib_view())
        self.tree.bind("<Button-3>", self._lib_context_menu)
        self.tree.bind("<MouseWheel>", self._tree_mousewheel)
        self.tree.bind("<Button-4>", lambda _event: self._tree_scroll(-1))
        self.tree.bind("<Button-5>", lambda _event: self._tree_scroll(1))

        right = ttk.Frame(f); right.grid(row=2, column=1, sticky="nsew", padx=(6, 12), pady=6)
        right.rowconfigure(1, weight=1); right.columnconfigure(0, weight=1)
        ttk.Label(right, text="Import / Activity Log", font=SUB).grid(row=0, column=0, sticky="w")
        self.lib_log = ScrolledText(right, font=MONO, state="disabled", wrap="word")
        self.lib_log.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

        btns = ttk.Frame(f); btns.grid(row=3, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 10))
        for i, (label, cmd, accent) in enumerate([
            ("Import Files…", self._lib_import_files, True),
            ("Import Folder…", self._lib_import_folder, False),
            ("Toggle Active", self._lib_toggle, False),
            ("Set Tags…", self._lib_tags, False),
            ("View Before/After", self._lib_view, False),
            ("Re-clean", self._lib_reclean, False),
            ("Remove", self._lib_remove, False),
        ]):
            ttk.Button(btns, text=label, command=cmd,
                       style="Accent.TButton" if accent else "TButton")\
                .grid(row=0, column=i, padx=(0, 8))

    def _refresh_library(self):
        if not hasattr(self, "tree"):
            return
        selected = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        docs = self.library.list_documents()
        needle = self.lib_filter.get().strip().casefold()
        if needle:
            docs = [d for d in docs if needle in f"{d.id} {d.title} {d.tags}".casefold()]
        if self.lib_active_only.get():
            docs = [d for d in docs if d.active]

        def sort_value(doc):
            value = getattr(doc, self.lib_sort_column)
            if value is None:
                return -1
            return value.casefold() if isinstance(value, str) else value

        docs.sort(key=sort_value, reverse=self.lib_sort_reverse)
        for d in docs:
            self.tree.insert("", "end", iid=str(d.id), values=(
                d.id, "yes" if d.active else "--", d.title,
                f"{d.chars:,}", f"{d.tokens:,}" if d.tokens else "?", d.tags))
        for iid in selected:
            if self.tree.exists(iid):
                self.tree.selection_add(iid)
        s = self.library.stats()
        tok_s = f"{s['tokens']:,} tokens" if s["tokens"] else "tokens: build cache"
        self.lib_stats.config(
            text=f"showing {len(docs)}/{s['documents']} | {s['active']} active | "
                 f"{s['chars']:,} chars | {tok_s}")

    def _lib_sort(self, column: str):
        if self.lib_sort_column == column:
            self.lib_sort_reverse = not self.lib_sort_reverse
        else:
            self.lib_sort_column = column
            self.lib_sort_reverse = False
        self._refresh_library()

    def _tree_scroll(self, units: int):
        self.tree.yview_scroll(units, "units")
        return "break"

    def _tree_mousewheel(self, event):
        steps = -int(event.delta / 120) if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1)
        return self._tree_scroll(steps)

    def _lib_context_menu(self, event):
        row = self.tree.identify_row(event.y)
        if row and row not in self.tree.selection():
            self.tree.selection_set(row)
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="View raw / cleaned", command=self._lib_view)
        menu.add_command(label="Toggle active", command=self._lib_toggle)
        menu.add_command(label="Set tags…", command=self._lib_tags)
        menu.add_separator()
        menu.add_command(label="Remove…", command=self._lib_remove)
        menu.tk_popup(event.x_root, event.y_root)

    def _selected_ids(self) -> list[int]:
        return [int(i) for i in self.tree.selection()]

    def _lib_import_files(self):
        paths = filedialog.askopenfilenames(
            title="Select documents",
            filetypes=[("Documents", "*.txt *.md *.markdown *.pdf *.epub *.docx *.html *.htm"),
                       ("All files", "*.*")])
        if paths:
            self._import_paths([Path(p) for p in paths])

    def _lib_import_folder(self):
        folder = filedialog.askdirectory(title="Select folder to import")
        if folder:
            self._import_paths(list(iter_supported_files(Path(folder))))

    def _import_paths(self, paths: list[Path]):
        def work():
            self.events.put({"type": "lib_log",
                             "text": f"[import] {len(paths)} file(s)…"})
            for p in paths:
                try:
                    _, msg = self.library.add_document(p)
                except ExtractionError as e:
                    msg = f"FAILED {p.name}: {e}"
                except Exception as e:
                    msg = f"FAILED {p.name}: {e}"
                self.events.put({"type": "lib_log", "text": "  " + msg})
            self.events.put({"type": "lib_log", "text": "[import] complete"})
            self.events.put({"type": "lib_done"})
        threading.Thread(target=work, daemon=True).start()

    def _lib_toggle(self):
        for i in self._selected_ids():
            d = self.library.get(i)
            if d:
                self.library.set_active(i, not d.active)
        self._refresh_library()

    def _lib_tags(self):
        ids = self._selected_ids()
        if not ids:
            return
        current = self.library.get(ids[0]).tags if len(ids) == 1 else ""
        tags = simpledialog.askstring("Tags", "Comma-separated tags:",
                                      initialvalue=current, parent=self)
        if tags is None:
            return
        for i in ids:
            self.library.set_tags(i, tags.strip())
        self._refresh_library()

    def _lib_remove(self):
        ids = self._selected_ids()
        if not ids:
            return
        if not messagebox.askyesno("Remove", f"Remove {len(ids)} document(s) from the library?"):
            return
        for i in ids:
            self.library.remove(i)
        self._refresh_library()

    def _lib_reclean(self):
        ids = self._selected_ids()
        if not ids:
            return
        def work():
            for i in ids:
                msg = self.library.reclean(i)
                self.events.put({"type": "lib_log", "text": "  " + msg})
            self.events.put({"type": "lib_done"})
        threading.Thread(target=work, daemon=True).start()

    def _lib_view(self):
        ids = self._selected_ids()
        if not ids:
            return
        d = self.library.get(ids[0])
        win = tk.Toplevel(self)
        win.title(f"#{d.id} {d.title} -- raw vs cleaned")
        win.geometry("1100x700")
        win.columnconfigure(0, weight=1); win.columnconfigure(1, weight=1)
        win.rowconfigure(1, weight=1)
        ttk.Label(win, text="RAW (as extracted)", font=SUB).grid(row=0, column=0, padx=8, pady=4, sticky="w")
        ttk.Label(win, text="CLEANED (what the model trains on)", font=SUB).grid(row=0, column=1, padx=8, pady=4, sticky="w")
        raw_box = ScrolledText(win, font=MONO, wrap="word")
        cln_box = ScrolledText(win, font=MONO, wrap="word")
        raw_box.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))
        cln_box.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))
        LIMIT = 400_000
        raw_box.insert("1.0", self.library.get_raw_text(d.id)[:LIMIT])
        cln_box.insert("1.0", self.library.get_text(d.id)[:LIMIT])
        raw_box.configure(state="disabled"); cln_box.configure(state="disabled")

    # ==================================================================
    # TOKENIZER TAB
    # ==================================================================
    def _build_tokenizer_tab(self):
        f = self.tab_tok
        f.columnconfigure(0, weight=1); f.rowconfigure(3, weight=1)

        ttk.Label(f, text="Tokenizer", font=TITLE).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))
        self.tok_status = ttk.Label(f, text="")
        self.tok_status.grid(row=0, column=0, sticky="e", padx=12)

        top = ttk.Frame(f); top.grid(row=1, column=0, sticky="w", padx=12, pady=4)
        ttk.Label(top, text="Vocab size:").grid(row=0, column=0, padx=(0, 6))
        self.tok_vocab = tk.IntVar(value=4096)
        ttk.Spinbox(top, from_=512, to=32768, increment=512,
                    textvariable=self.tok_vocab, width=8).grid(row=0, column=1, padx=(0, 12))
        self.btn_tok_train = ttk.Button(top, text="Train on Active Documents",
                                        style="Accent.TButton", command=self._tok_train)
        self.btn_tok_train.grid(row=0, column=2, padx=(0, 12))
        ttk.Label(top, text="(retraining the tokenizer invalidates caches and checkpoints)")\
            .grid(row=0, column=3)

        insp = ttk.Labelframe(f, text="Inspector -- paste text, see exactly how it tokenizes")
        insp.grid(row=2, column=0, sticky="ew", padx=12, pady=8)
        insp.columnconfigure(0, weight=1)
        self.tok_input = ScrolledText(insp, height=4, font=MONO, wrap="word")
        self.tok_input.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(insp, text="Inspect", command=self._tok_inspect)\
            .grid(row=0, column=1, sticky="n", padx=(0, 8), pady=8)
        self.tok_view = tk.Text(insp, height=8, font=MONO, wrap="word", state="disabled")
        self.tok_view.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 4))
        self.tok_view.tag_configure("even", background="#dce9f7")
        self.tok_view.tag_configure("odd", background="#f7e8d2")
        self.tok_count = ttk.Label(insp, text="")
        self.tok_count.grid(row=2, column=0, sticky="w", padx=8, pady=(0, 8))

        ttk.Label(f, text="Log", font=SUB).grid(row=3, column=0, sticky="nw", padx=12)
        self.tok_log = ScrolledText(f, font=MONO, state="disabled", height=8, wrap="word")
        self.tok_log.grid(row=4, column=0, sticky="nsew", padx=12, pady=(0, 12))
        f.rowconfigure(4, weight=1)

    def _refresh_tok_status(self):
        if self.tok.loaded:
            self.tok_status.config(
                text=f"loaded | vocab {self.tok.vocab_size:,} | id {self.tok.fingerprint()}")
        else:
            self.tok_status.config(text="no tokenizer trained yet")

    def _tok_train(self):
        if self.tok_thread and self.tok_thread.is_alive():
            return
        if self.chat_thread and self.chat_thread.is_alive():
            messagebox.showinfo("Tokenizer", "Stop the active generation before retraining.")
            return
        docs = self.library.active_documents()
        if not docs:
            messagebox.showerror("Tokenizer", "No active documents in the Library.")
            return
        try:
            vocab = int(self.tok_vocab.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Tokenizer", "Vocab size must be a whole number.")
            return
        self.btn_tok_train.config(state="disabled")
        self.btn_train_start.config(state="disabled")
        self.status_var.set("Training tokenizer…")
        def work():
            try:
                self.events.put({"type": "tok_log",
                                 "text": f"[tokenizer] training BPE vocab={vocab:,} "
                                         f"on {len(docs)} document(s)…"})
                texts = [self.library.get_text(d.id) for d in docs]
                t0 = time.time()
                self.tok.train_from_texts(texts, vocab_size=vocab)
                self.library.clear_token_counts()
                self.events.put({"type": "tok_log",
                                 "text": f"[tokenizer] done in {time.time()-t0:.1f}s | "
                                         f"vocab {self.tok.vocab_size:,} | id {self.tok.fingerprint()}"})
                self.events.put({"type": "tok_done"})
            except Exception as exc:
                self.events.put({"type": "tok_err", "text": str(exc)})
        self.tok_thread = threading.Thread(target=work, daemon=True)
        self.tok_thread.start()

    def _tok_inspect(self):
        if not self.tok.loaded:
            messagebox.showerror("Tokenizer", "Train or load a tokenizer first.")
            return
        text = self.tok_input.get("1.0", "end-1c")
        if not text.strip():
            return
        pairs = self.tok.inspect(text)
        self.tok_view.configure(state="normal")
        self.tok_view.delete("1.0", "end")
        for i, (tok_str, tok_id) in enumerate(pairs):
            shown = tok_str if tok_str.strip("\n") == tok_str else tok_str.replace("\n", "\\n")
            self.tok_view.insert("end", shown, "even" if i % 2 == 0 else "odd")
        self.tok_view.configure(state="disabled")
        self.tok_count.config(
            text=f"{len(pairs)} tokens | {len(text)} chars | "
                 f"{len(text)/max(1,len(pairs)):.2f} chars/token | "
                 f"ids: {[p[1] for p in pairs[:24]]}{' …' if len(pairs) > 24 else ''}")

    # ==================================================================
    # TRAINING TAB
    # ==================================================================
    def _build_training_tab(self):
        f = self.tab_train
        f.columnconfigure(1, weight=1)
        f.rowconfigure(0, weight=1)

        # ---------- left: controls ----------
        left = ttk.Frame(f); left.grid(row=0, column=0, sticky="nsw", padx=12, pady=10)
        ttk.Label(left, text="Training", font=TITLE).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(left, text="Model preset:").grid(row=1, column=0, sticky="w", pady=3)
        self.preset_var = tk.StringVar(value=DEFAULT_PRESET)
        cb = ttk.Combobox(left, textvariable=self.preset_var,
                          values=list(MODEL_PRESETS), state="readonly", width=32)
        cb.grid(row=1, column=1, sticky="w", pady=3)
        cb.bind("<<ComboboxSelected>>", lambda e: self._update_param_estimate())
        self.lbl_params = ttk.Label(left, text="")
        self.lbl_params.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 6))

        def row(r, label, var, width=10):
            ttk.Label(left, text=label).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(left, textvariable=var, width=width).grid(row=r, column=1, sticky="w", pady=3)

        self.v_steps_total = tk.IntVar(value=2000);  row(3, "Optimizer steps:", self.v_steps_total)
        self.v_batch = tk.IntVar(value=16);          row(4, "Batch size:", self.v_batch)
        self.v_accum = tk.IntVar(value=1);           row(5, "Grad accumulation:", self.v_accum)
        self.v_lr_max = tk.DoubleVar(value=3e-4);    row(6, "LR max:", self.v_lr_max)
        self.v_lr_min = tk.DoubleVar(value=3e-5);    row(7, "LR min:", self.v_lr_min)
        self.v_warmup = tk.IntVar(value=100);        row(8, "Warmup steps:", self.v_warmup)
        self.v_valint = tk.IntVar(value=250);        row(9, "Val interval:", self.v_valint)
        self.v_patience = tk.IntVar(value=8);        row(10, "Early-stop patience:", self.v_patience)
        self.v_sampint = tk.IntVar(value=500);       row(11, "Sample interval:", self.v_sampint)

        ttk.Label(left, text="Sample prompt:").grid(row=12, column=0, sticky="w", pady=3)
        self.v_sample_prompt = tk.StringVar(value="")
        ttk.Entry(left, textvariable=self.v_sample_prompt, width=22).grid(row=12, column=1, sticky="w", pady=3)

        self.v_resume = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Resume from final.pt (explicit)",
                        variable=self.v_resume).grid(row=13, column=0, columnspan=2, sticky="w", pady=(6, 2))

        bt = ttk.Frame(left); bt.grid(row=14, column=0, columnspan=2, sticky="w", pady=(10, 4))
        self.btn_train_start = ttk.Button(bt, text="Start Training", style="Accent.TButton",
                                          command=self._train_start)
        self.btn_train_start.grid(row=0, column=0, padx=(0, 8))
        self.btn_train_stop = ttk.Button(bt, text="Stop", command=self._train_stop_req,
                                         state="disabled")
        self.btn_train_stop.grid(row=0, column=1)
        ttk.Button(bt, text="Model / VRAM Planner…", command=self._open_model_planner)\
            .grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        self.train_prog = ttk.Progressbar(left, mode="determinate", length=240)
        self.train_prog.grid(row=15, column=0, columnspan=2, sticky="ew", pady=(8, 2))

        stats = ttk.Frame(left); stats.grid(row=16, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.lbl_step = ttk.Label(stats, text="step --");     self.lbl_step.grid(row=0, column=0, sticky="w")
        self.lbl_loss = ttk.Label(stats, text="train rolling --"); self.lbl_loss.grid(row=1, column=0, sticky="w")
        self.lbl_val = ttk.Label(stats, text="eval train -- | val -- | gap --"); self.lbl_val.grid(row=2, column=0, sticky="w")
        self.lbl_speed = ttk.Label(stats, text="-- tok/s");   self.lbl_speed.grid(row=3, column=0, sticky="w")
        self.lbl_tokens = ttk.Label(stats, text="0 tokens seen"); self.lbl_tokens.grid(row=4, column=0, sticky="w")
        self.lbl_lr = ttk.Label(stats, text="lr --"); self.lbl_lr.grid(row=5, column=0, sticky="w")

        # ---------- right: chart + samples + log ----------
        right = ttk.Frame(f); right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=10)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=2); right.rowconfigure(3, weight=2); right.rowconfigure(5, weight=1)

        ttk.Label(right, text="Loss (rolling train + fixed train/validation gap)", font=SUB).grid(row=0, column=0, sticky="w")
        self.chart_holder = ttk.Frame(right)
        self.chart_holder.grid(row=1, column=0, sticky="nsew", pady=(2, 6))
        self._init_chart()

        ttk.Label(right, text="Sample generations (watch it grow)", font=SUB).grid(row=2, column=0, sticky="w")
        self.sample_box = ScrolledText(right, font=MONO, state="disabled", height=8, wrap="word")
        self.sample_box.grid(row=3, column=0, sticky="nsew", pady=(2, 6))

        ttk.Label(right, text="Log", font=SUB).grid(row=4, column=0, sticky="w")
        self.train_log = ScrolledText(right, font=MONO, state="disabled", height=6, wrap="word")
        self.train_log.grid(row=5, column=0, sticky="nsew", pady=(2, 0))

        self._update_param_estimate()

    def _current_preset(self):
        n_layers, dim, n_heads, block = MODEL_PRESETS[self.preset_var.get()]
        return n_layers, dim, n_heads, block

    def _update_param_estimate(self):
        vocab = self.tok.vocab_size if self.tok.loaded else 4096
        est = exact_parameter_count(vocab, MODEL_PRESETS[self.preset_var.get()])
        self.lbl_params.config(
            text=(f"{est/1e6:.1f}M params (vocab {vocab:,})\n"
                  f"Training device: {training_device_summary()}")
        )

    def _open_model_planner(self):
        if self.planner_window is not None and self.planner_window.winfo_exists():
            self.planner_window.lift()
            self.planner_window.focus_force()
            return
        stats = self.library.stats()
        if stats["tokens"]:
            corpus_tokens = int(stats["tokens"])
            source = "cached tokenizer counts"
        else:
            corpus_tokens = max(1, round(stats["chars"] / 4))
            source = "estimated from active characters ÷ 4"
        vocab = self.tok.vocab_size if self.tok.loaded else 4096
        self.planner_window = ModelPlannerWindow(
            self,
            vocab_size=vocab,
            corpus_tokens=corpus_tokens,
            token_source=source,
            current_preset=self.preset_var.get(),
            apply_callback=self._apply_model_plan,
        )

    def _apply_model_plan(self, settings: dict):
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showerror("Training", "Stop the active training run before applying a new plan.")
            return False
        self.preset_var.set(settings["preset"])
        self.v_batch.set(settings["batch_size"])
        self.v_accum.set(settings["grad_accum"])
        self.v_steps_total.set(settings["total_steps"])
        self.v_lr_max.set(settings["lr_max"])
        self.v_lr_min.set(settings["lr_min"])
        self.v_warmup.set(settings["warmup_steps"])
        self.v_valint.set(settings["val_interval"])
        self.v_sampint.set(settings["sample_interval"])
        self._update_param_estimate()
        self.status_var.set(
            f"Applied {settings['preset'].split('(', 1)[0].strip()} plan: "
            f"batch {settings['batch_size']} × accum {settings['grad_accum']}"
        )
        return True

    def _init_chart(self):
        if Figure is None:
            ttk.Label(self.chart_holder, text="matplotlib not installed -- charts disabled").pack()
            self.fig = self.ax = self.chart_canvas = None
            return
        self.fig = Figure(figsize=(6.4, 2.7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.chart_holder)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _redraw_chart(self):
        if self.ax is None:
            return
        self.ax.clear()
        if self.h_steps:
            self.ax.plot(self.h_steps, self.h_loss, lw=1.0, alpha=0.65,
                         label="train rolling")
        if self.e_steps:
            self.ax.plot(self.e_steps, self.e_train_loss, "s--", ms=4,
                         label="train eval")
        if self.v_steps:
            self.ax.plot(self.v_steps, self.v_loss, "o--", ms=4,
                         label="validation")
        if len(self.e_steps) >= 2 and self.e_steps == self.v_steps:
            self.ax.fill_between(self.e_steps, self.e_train_loss, self.v_loss,
                                 alpha=0.12, label="overfit gap")
        self.ax.set_xlabel("step"); self.ax.set_ylabel("loss")
        if self.h_steps or self.v_steps:
            self.ax.legend(loc="upper right", fontsize=8)
        self.ax.grid(alpha=0.25)
        self.chart_canvas.draw_idle()

    def _train_start(self):
        if self.train_thread and self.train_thread.is_alive():
            return
        if not self.tok.loaded:
            messagebox.showerror("Training", "Train a tokenizer first (Tokenizer tab).")
            return
        if not self.library.active_documents():
            messagebox.showerror("Training", "No active documents in the Library.")
            return

        # Never silently fall back to CPU when an NVIDIA GPU is present but
        # the active Python environment contains a CPU-only/broken CUDA build.
        try:
            training_device()
        except RuntimeError as exc:
            messagebox.showerror("CUDA unavailable", str(exc))
            return

        try:
            n_layers, dim, n_heads, block = self._current_preset()
            mcfg = ModelConfig(vocab_size=self.tok.vocab_size, block_size=block,
                               dim=dim, n_layers=n_layers, n_heads=n_heads)
            tcfg = TrainConfig(
                total_steps=int(self.v_steps_total.get()),
                batch_size=int(self.v_batch.get()),
                grad_accum=int(self.v_accum.get()),
                lr_max=float(self.v_lr_max.get()),
                lr_min=float(self.v_lr_min.get()),
                warmup_steps=int(self.v_warmup.get()),
                val_interval=int(self.v_valint.get()),
                patience=int(self.v_patience.get()),
                sample_interval=int(self.v_sampint.get()),
                sample_prompt=self.v_sample_prompt.get(),
                resume=bool(self.v_resume.get()),
            )
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Training settings", str(exc))
            return

        # reset chart + progress for this run (resume keeps continuity)
        if not tcfg.resume:
            self.h_steps.clear(); self.h_loss.clear()
            self.e_steps.clear(); self.e_train_loss.clear()
            self.v_steps.clear(); self.v_loss.clear()
            self.chart_dirty = True
        self.train_prog.config(maximum=tcfg.total_steps, value=0)
        self.train_stop.clear()
        self.btn_train_start.config(state="disabled")
        self.btn_train_stop.config(state="normal")
        self.btn_tok_train.config(state="disabled")
        self.status_var.set("Training…")

        fp = self.tok.fingerprint()
        corpus_fp = self.library.corpus_fingerprint()

        def work():
            try:
                try:
                    train_arr, val_arr, meta = load_token_cache(
                        fp, expected_corpus_fingerprint=corpus_fp)
                    self.events.put({"type": "log",
                                     "text": f"[cache] reusing cache "
                                             f"({meta['train_tokens']:,} train / "
                                             f"{meta['val_tokens']:,} val tokens)"})
                except RuntimeError:
                    self.events.put({"type": "log", "text": "[cache] building token cache…"})
                    build_token_cache(self.library, self.tok,
                                      log=lambda s: self.events.put({"type": "log", "text": s}))
                    train_arr, val_arr, meta = load_token_cache(
                        fp, expected_corpus_fingerprint=corpus_fp)
                    self.events.put({"type": "lib_done"})
                trainer = Trainer(mcfg, tcfg, train_arr, val_arr, self.tok,
                                  self.events, self.train_stop, tokenizer_fingerprint=fp)
                trainer.train()
            except Exception as e:
                self.events.put({"type": "done", "reason": f"failed: {e}",
                                 "best_val": float("inf"), "steps": 0})

        self.train_thread = threading.Thread(target=work, daemon=True)
        self.train_thread.start()

    def _train_stop_req(self):
        self.train_stop.set()
        self._append(self.train_log, "[stop requested -- finishing current step]\n")

    # ==================================================================
    # CHAT TAB
    # ==================================================================
    def _build_chat_tab(self):
        f = self.tab_chat
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=1)

        top = ttk.Frame(f); top.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        ttk.Label(top, text="Chat", font=TITLE).grid(row=0, column=0, sticky="w", padx=(0, 16))
        self.model_buttons = [
            ttk.Button(top, text="Load best.pt", style="Accent.TButton",
                       command=lambda: self._load_model(BEST_CKPT)),
            ttk.Button(top, text="Load final.pt",
                       command=lambda: self._load_model(FINAL_CKPT)),
            ttk.Button(top, text="Browse…", command=self._load_model_browse),
        ]
        for column, button in enumerate(self.model_buttons, start=1):
            button.grid(row=0, column=column, padx=4)
        self.chat_model_lbl = ttk.Label(top, text="no model loaded")
        self.chat_model_lbl.grid(row=0, column=4, padx=12)

        opts = ttk.Frame(f); opts.grid(row=1, column=0, sticky="ew", padx=12, pady=2)
        ttk.Label(opts, text="Mode:").grid(row=0, column=0)
        self.chat_mode = tk.StringVar(value="Dialogue")
        ttk.Combobox(opts, textvariable=self.chat_mode, width=9, state="readonly",
                     values=["Dialogue", "Continue"]).grid(row=0, column=1, padx=(2, 10))
        def opt(c, label, var, frm, to, inc, w=6):
            ttk.Label(opts, text=label).grid(row=0, column=c)
            ttk.Spinbox(opts, textvariable=var, from_=frm, to=to, increment=inc,
                        width=w).grid(row=0, column=c + 1, padx=(2, 10))
        self.c_temp = tk.DoubleVar(value=0.8);  opt(2, "Temp:", self.c_temp, 0.1, 2.0, 0.05)
        self.c_topk = tk.IntVar(value=50);      opt(4, "Top-k:", self.c_topk, 0, 500, 10)
        self.c_topp = tk.DoubleVar(value=0.95); opt(6, "Top-p:", self.c_topp, 0.1, 1.0, 0.05)
        self.c_rep = tk.DoubleVar(value=1.15);  opt(8, "Rep-pen:", self.c_rep, 1.0, 2.0, 0.05)
        self.c_maxnew = tk.IntVar(value=120);   opt(10, "Max new:", self.c_maxnew, 10, 1000, 10)

        self.chat_box = ScrolledText(f, font=("Segoe UI", 11), state="disabled", wrap="word")
        self.chat_box.grid(row=2, column=0, sticky="nsew", padx=12, pady=6)
        self.chat_box.tag_configure("user", foreground="#1d4ed8",
                                    font=("Segoe UI", 11, "bold"))
        self.chat_box.tag_configure("model", foreground="#111111")
        self.chat_box.tag_configure("info", foreground="#6b7280",
                                    font=("Segoe UI", 9, "italic"))

        bottom = ttk.Frame(f); bottom.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 4))
        bottom.columnconfigure(0, weight=1)
        self.chat_entry = tk.Text(bottom, height=3, font=("Segoe UI", 11), wrap="word")
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.chat_entry.bind("<Return>", self._chat_enter)
        self.chat_entry.bind("<Shift-Return>", lambda e: None)
        self.btn_send = ttk.Button(bottom, text="Send", style="Accent.TButton",
                                   command=self._chat_send)
        self.btn_send.grid(row=0, column=1, sticky="ns")
        self.btn_chat_stop = ttk.Button(bottom, text="Stop", command=self.chat_stop.set,
                                        state="disabled")
        self.btn_chat_stop.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        self.btn_chat_clear = ttk.Button(bottom, text="Clear", command=self._chat_clear)
        self.btn_chat_clear.grid(row=0, column=3, sticky="ns", padx=(8, 0))

        ctxf = ttk.Frame(f); ctxf.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 10))
        ctxf.columnconfigure(1, weight=1)
        self.ctx_label = ttk.Label(ctxf, text="context: -- / -- tokens")
        self.ctx_label.grid(row=0, column=0, padx=(0, 10))
        self.ctx_bar = ttk.Progressbar(ctxf, mode="determinate")
        self.ctx_bar.grid(row=0, column=1, sticky="ew")

    def _chat_enter(self, event):
        if not (event.state & 0x0001):     # Shift not held -> send
            self._chat_send()
            return "break"
        return None

    def _load_model_browse(self):
        p = filedialog.askopenfilename(initialdir=str(CKPT_DIR),
                                       filetypes=[("Checkpoint", "*.pt")])
        if p:
            self._load_model(Path(p))

    def _load_model(self, path: Path):
        if not Path(path).exists():
            messagebox.showerror("Chat", f"Checkpoint not found:\n{path}")
            return
        if not self.tok.loaded:
            messagebox.showerror("Chat", "Train/load a tokenizer first.")
            return
        self._set_model_buttons("disabled")
        self.status_var.set(f"Loading {Path(path).name}…")
        path = Path(path)

        def work():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, payload = NanoLM.from_checkpoint(path, map_location=device)
                model.to(device).eval()
                self.events.put({"type": "model_loaded", "model": model,
                                 "payload": payload, "path": path, "device": device})
            except Exception as exc:
                self.events.put({"type": "model_err", "text": str(exc)})
        threading.Thread(target=work, daemon=True).start()

    def _set_model_buttons(self, state: str):
        for button in self.model_buttons:
            button.config(state=state)

    def _finish_model_load(self, ev: dict):
        model, payload = ev["model"], ev["payload"]
        path, device = ev["path"], ev["device"]
        fp = payload.get("tokenizer_fingerprint", "")
        if fp and fp != self.tok.fingerprint():
            messagebox.showwarning(
                "Tokenizer mismatch",
                "This checkpoint was trained with a DIFFERENT tokenizer.\n"
                "Generated text will be garbage until you load the matching one.")
        self.loaded_model = model
        self.loaded_payload = payload
        cfg = model.cfg
        self.chat_model_lbl.config(
            text=f"{Path(path).name} | {model.num_params/1e6:.1f}M | "
                 f"ctx {cfg.block_size} | step {payload.get('step', '?')} | {device}")
        self.ctx_bar.config(maximum=cfg.block_size, value=0)
        self.ctx_label.config(text=f"context: 0 / {cfg.block_size} tokens")
        self._chat_insert("info", f"[model loaded: {Path(path).name}]\n")
        self._set_model_buttons("normal")
        self.status_var.set(f"Loaded {Path(path).name}")

    def _chat_insert(self, tag: str, text: str, stream: bool = False):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", text, tag)
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _chat_replace_response(self, text: str):
        self.chat_box.configure(state="normal")
        self.chat_box.delete("stream_start", "end-1c")
        self.chat_box.insert("end", text, "model")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _set_ctx_bar(self, used):
        if used is None or self.loaded_model is None:
            return
        block = self.loaded_model.cfg.block_size
        used = min(used, block)
        self.ctx_bar.config(value=used)
        self.ctx_label.config(text=f"context: {used} / {block} tokens"
                                   + ("  (window full -- sliding)" if used >= block else ""))

    def _chat_clear(self):
        self.chat_history.clear()
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0", "end")
        self.chat_box.configure(state="disabled")
        self._set_ctx_bar(0)

    @staticmethod
    def _build_context(user_msg: str, mode: str,
                       history: list[tuple[str, str]]) -> str:
        if mode == "Dialogue":
            parts = [f"You: {u}\nModel: {m}\n" for u, m in history]
            return "".join(parts) + f"You: {user_msg}\nModel:"
        parts = [u + m for u, m in history]
        return "".join(parts) + user_msg

    def _chat_send(self):
        if self.loaded_model is None:
            messagebox.showerror("Chat", "Load a model checkpoint first.")
            return
        if self.tok_thread and self.tok_thread.is_alive():
            messagebox.showinfo("Chat", "Wait for tokenizer training to finish.")
            return
        msg = self.chat_entry.get("1.0", "end-1c").strip()
        if not msg:
            return
        if self.chat_thread and self.chat_thread.is_alive():
            return
        try:
            params = dict(
                max_new_tokens=int(self.c_maxnew.get()),
                temperature=float(self.c_temp.get()),
                top_k=int(self.c_topk.get()),
                top_p=float(self.c_topp.get()),
                repetition_penalty=float(self.c_rep.get()),
            )
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Generation settings", str(exc))
            return
        self.chat_entry.delete("1.0", "end")
        self._chat_insert("user", f"\nYou: {msg}\n")
        self._chat_insert("model", "Model: ")
        self.chat_box.mark_set("stream_start", "end-1c")
        self.chat_box.mark_gravity("stream_start", "left")

        mode = self.chat_mode.get()
        history_snapshot = list(self.chat_history)
        context = self._build_context(msg, mode, history_snapshot)
        ids = self.tok.encode(context)
        block = self.loaded_model.cfg.block_size
        if len(ids) > block - 1:
            ids = ids[-(block - 1):]
        self._set_ctx_bar(len(ids))

        self.chat_stop.clear()
        self.btn_send.config(state="disabled")
        self.btn_chat_stop.config(state="normal")
        self.btn_chat_clear.config(state="disabled")
        self.btn_tok_train.config(state="disabled")
        self.status_var.set("Generating…")
        model, tok = self.loaded_model, self.tok
        ctx_start = len(ids)

        def work():
            t0 = time.time()
            gen_ids: list[int] = []
            try:
                stops = {"\nYou:", "\nyou:"} if mode == "Dialogue" else set()
                for tid in generate_stream(model, ids, eos_id=tok.eos_id,
                                           stop_check=self.chat_stop.is_set, **params):
                    gen_ids.append(tid)
                    text = tok.decode(gen_ids)
                    # stop if the model starts speaking for the user
                    cut = None
                    for s in stops:
                        p = text.find(s)
                        if p != -1:
                            cut = p if cut is None else min(cut, p)
                    if cut is not None:
                        self.events.put({"type": "chat_replace", "text": text[:cut],
                                         "ctx": min(ctx_start + len(gen_ids), model.cfg.block_size)})
                        break
                    self.events.put({"type": "chat_replace", "text": text,
                                     "ctx": min(ctx_start + len(gen_ids), model.cfg.block_size)})
                final_text = tok.decode(gen_ids)
                for s in ("\nYou:", "\nyou:"):
                    p = final_text.find(s)
                    if p != -1:
                        final_text = final_text[:p]
                self.events.put({"type": "chat_done", "n": len(gen_ids),
                                 "secs": time.time() - t0,
                                 "ctx": min(ctx_start + len(gen_ids), model.cfg.block_size),
                                 "user": msg, "text": final_text})
            except Exception as e:
                self.events.put({"type": "chat_err", "text": str(e)})

        self.chat_thread = threading.Thread(target=work, daemon=True)
        self.chat_thread.start()

    # ==================================================================
    # GLASS BOX TAB
    # ==================================================================
    def _build_glass_tab(self):
        f = self.tab_glass
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=3)
        f.rowconfigure(3, weight=1)

        ttk.Label(f, text="Glass Box -- look inside the loaded model", font=TITLE)\
            .grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        top = ttk.Frame(f); top.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Prompt:").grid(row=0, column=0, padx=(0, 6))
        self.glass_prompt = tk.StringVar(value="The")
        ttk.Entry(top, textvariable=self.glass_prompt).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.btn_glass_run = ttk.Button(top, text="Run", style="Accent.TButton",
                                        command=self._glass_run)
        self.btn_glass_run.grid(row=0, column=2)
        ttk.Label(top, text="(uses the model loaded in the Chat tab)").grid(row=0, column=3, padx=8)

        self.glass_attn_holder = ttk.Frame(f)
        self.glass_attn_holder.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)
        self.glass_prob_holder = ttk.Frame(f)
        self.glass_prob_holder.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 10))

    def _glass_run(self):
        if self.loaded_model is None:
            messagebox.showerror("Glass Box", "Load a model in the Chat tab first.")
            return
        if Figure is None:
            messagebox.showinfo("Glass Box", "matplotlib not installed.")
            return
        prompt = self.glass_prompt.get().strip()
        if not prompt:
            return
        tok, model = self.tok, self.loaded_model
        ids = tok.encode(prompt)[-model.cfg.block_size:]
        if not ids:
            return
        labels = [tok.tokenizer.decode([i], skip_special_tokens=False) for i in ids]
        labels = [l.replace("\n", "\\n") if l.strip() or l == " " else repr(l) for l in labels]
        self.btn_glass_run.config(state="disabled")
        self.status_var.set("Running Glass Box analysis…")

        def work():
            try:
                device = next(model.parameters()).device
                x = torch.tensor([ids], dtype=torch.long, device=device)
                captures: list[dict] = []
                with torch.no_grad():
                    logits, _ = model(x, captures=captures)
                probs = torch.softmax(logits[0, -1].float(), dim=-1)
                count = min(10, probs.numel())
                topp, topi = torch.topk(probs, count)
                top_tokens = [tok.tokenizer.decode([int(i)], skip_special_tokens=False)
                              for i in topi]
                self.events.put({"type": "glass_result", "captures": captures,
                                 "probabilities": topp.cpu().tolist(),
                                 "top_tokens": top_tokens, "labels": labels,
                                 "prompt": prompt})
            except Exception as exc:
                self.events.put({"type": "glass_err", "text": str(exc)})
        threading.Thread(target=work, daemon=True).start()

    def _render_glass(self, ev: dict):
        captures = ev["captures"]
        labels = ev["labels"]
        top_tokens = ev["top_tokens"]
        probabilities = ev["probabilities"]
        prompt = ev["prompt"]

        for figure in self.glass_figures:
            figure.clear()
        self.glass_figures.clear()

        # ---- attention grid ----
        for w in self.glass_attn_holder.winfo_children():
            w.destroy()
        n = len(captures)
        if not n:
            raise RuntimeError("The model returned no attention captures.")
        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig = Figure(figsize=(11, 2.9 * rows), dpi=90)
        for li, cap in enumerate(captures):
            ax = fig.add_subplot(rows, cols, li + 1)
            mat = cap["attn"][0].mean(0).numpy()       # avg heads -> [T, T]
            ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(f"layer {li}  (resid |x|={cap.get('resid_norm', 0):.1f})",
                         fontsize=9)
            T = mat.shape[0]
            if T <= 18:
                ax.set_xticks(range(T)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
                ax.set_yticks(range(T)); ax.set_yticklabels(labels, fontsize=6)
            else:
                ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.glass_attn_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.glass_figures.append(fig)

        # ---- next-token distribution ----
        for w in self.glass_prob_holder.winfo_children():
            w.destroy()
        fig2 = Figure(figsize=(11, 2.0), dpi=90)
        ax2 = fig2.add_subplot(111)
        shown = [t.replace("\n", "\\n") or "·" for t in top_tokens]
        ax2.bar(range(len(probabilities)), probabilities)
        ax2.set_xticks(range(len(probabilities)))
        ax2.set_xticklabels(shown, rotation=30, ha="right", fontsize=8)
        ax2.set_title(f"next-token probabilities after: {prompt!r}", fontsize=9)
        fig2.tight_layout()
        canvas2 = FigureCanvasTkAgg(fig2, master=self.glass_prob_holder)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        self.glass_figures.append(fig2)

    # ==================================================================
    # SHUTDOWN
    # ==================================================================
    def _on_close(self):
        active_training = bool(self.train_thread and self.train_thread.is_alive())
        if active_training and not messagebox.askyesno(
                "Exit NanoLM Studio",
                "Training is active. Stop it, save final.pt, and exit?"):
            return
        self.train_stop.set()
        self.chat_stop.set()
        self.status_var.set("Stopping workers and closing…")
        self._closing = True
        self._close_deadline = time.monotonic() + 8.0
        self.after(100, self._finish_close)

    def _finish_close(self):
        workers = (self.train_thread, self.tok_thread, self.chat_thread)
        if any(thread and thread.is_alive() for thread in workers):
            if time.monotonic() < self._close_deadline:
                self.after(100, self._finish_close)
                return
        for figure in self.glass_figures:
            figure.clear()
        self.destroy()


def run():
    App().mainloop()
