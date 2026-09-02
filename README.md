# NanoLM Studio v4

NanoLM Studio is a local desktop workbench for building a small decoder-only
language model from a corpus you can inspect and control. You can built it your way, how you want it to be.
It combines document ingestion, cleaning, ByteLevel BPE tokenization, PyTorch training, interactive
generation, and attention visualisation in one Tk application.

It has gone through a number of iterations and this V4 Studio is a reliability-focused version. It has a five-tab
workflow while correcting the package, storage, cache, training, threading, and generation problems found in earlier versions.

## Quick start

Use Python 3.11 or newer. A virtual environment is strongly recommended.

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

The Windows requirements explicitly install PyTorch 2.12.1 with CUDA 12.6.
NanoLM also checks for the broken state where Windows can see an NVIDIA GPU but
the active environment contains CPU-only PyTorch. Training is blocked with
repair instructions in that case; it will no longer silently use CPU.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

On non-Windows systems, PyTorch installation can vary by operating system and
accelerator. If the standard requirements command cannot select the right
build, install PyTorch using the command generated at
<https://pytorch.org/get-started/locally/>, then run
`pip install -r requirements.txt` again.

## Workflow

1. **Library** — import TXT, Markdown, HTML, PDF, EPUB, or DOCX files. Search,
   sort, tag, enable/disable, remove, re-clean, or compare raw and cleaned text.
   The document table has permanent vertical and horizontal scrollbars.
2. **Tokenizer** — train a ByteLevel BPE tokenizer on active documents and use
   the inspector to see token boundaries and IDs.
3. **Training** — choose a model preset and train with gradient accumulation,
   cosine learning-rate decay, validation, early stopping, checkpoints, live
   metrics, charts, and sample generations. Each validation checkpoint reports
   fixed-sample training loss, validation loss, and their signed overfit gap;
   the chart shades the gap as it opens. The Model / VRAM Planner compares
   hardware and corpus fit before applying a configuration.
4. **Chat** — load `best.pt`, `final.pt`, or a chosen checkpoint and generate in
   dialogue or continuation mode. Generation streams, can be stopped, and uses
   a key/value cache while the context window has room.
5. **Glass Box** — inspect mean attention maps by layer and the next-token
   probability distribution for a prompt.

## Storage and portability

By default, all mutable data is placed in `./data` beside `main.py`:

```text
data/
  corpus.db
  raw/
  documents/
  tokenizer/tokenizer.json
  cache/{train_tokens.npy,val_tokens.npy,meta.json}
  checkpoints/{best.pt,final.pt}
  runs/run_*.jsonl
```

Set `NANOLM_DATA_DIR` before launch to use another workspace:

```powershell
$env:NANOLM_DATA_DIR = "D:\NanoLMProjects\experiment-01"
python main.py
```

```bash
NANOLM_DATA_DIR="$HOME/NanoLMProjects/experiment-01" python main.py
```

This also makes automated tests safe: they use isolated temporary directories
and never touch the real corpus.

## What v4 fixes

- Correct package layout: `python main.py` now imports the included `nanolm`
  package successfully.
- The Library Treeview's scrollbar is actually placed in the layout, is always
  visible, and is paired with search, active-only filtering, sorting, mouse
  wheel support, and a context menu.
- SQLite and document paths can be isolated; file replacements and tokenizer,
  cache, and checkpoint writes are atomic.
- Token-cache validity includes both the tokenizer fingerprint and the exact
  active corpus fingerprint, preventing stale training data after corpus edits.
- Model parameter counts correctly account for tied weights.
- Six presets now cover approximately 4.2M through 478M parameters: Tiny,
  Small, Medium, Large, XL, and XXL.
- A persistent Model / VRAM Planner accepts available VRAM, active corpus
  tokens, and a trial micro-batch. Its colour-coded table and visual meters
  distinguish comfortable, marginal, and poor settings, recommend the best
  corpus/model match, and apply batch, accumulation, step, and LR settings.
- Training steps now mean optimizer updates. Every update performs exactly the
  requested number of accumulation micro-batches, so partial gradients are not
  silently discarded.
- Resume checkpoints include optimizer/scaler state, early-stop state, NumPy
  RNG state, and PyTorch RNG state.
- Validation restores the model's prior mode; sampling and shutdown respond to
  stop requests.
- Generation validates settings, supports true greedy decoding at temperature
  zero, avoids per-token CPU repetition-penalty loops, and caches attention
  keys/values until the sliding window must be rebuilt.
- Tk widgets and variables are only accessed on the UI thread. Checkpoint loads
  and Glass Box inference run in workers, and the event pump uses a time budget
  so streaming cannot starve the interface.
- Chat streaming replaces the current decoded response rather than assuming
  every ByteLevel decode is an append-only string.
- Closing during training requests a clean stop and gives `final.pt` time to be
  saved.

See `V4_REVIEW.md` for the full audit and development critique.
See `V4_CAPACITY_PLANNER.md` for planner ranges, estimates, and usage.

## Tests

Run the non-GUI suite from the project directory:

```bash
python -m unittest discover -s tests -v
python smoke_test.py
```

The smoke test creates a temporary corpus, tokenizer, cache, run history, and
checkpoints. It does not delete or overwrite application data.

## Limits and realistic expectations

This is an educational local language-model studio, not a substitute for a
large pretrained assistant. Results depend heavily on corpus quality, corpus
size, compute, and training time. The medium preset may be slow on CPU. Keep
backups of valuable corpora and only load checkpoint files you trust, because
PyTorch checkpoints are not a safe format for untrusted downloads.

The next natural evolution is a multi-project experiment studio: named project
workspaces, richer dataset versioning, comparative run dashboards, adapter
fine-tuning of pretrained models, and export/inference backends such as ONNX or
GGUF. V4 deliberately establishes the reliable local foundation those features
would need.

It is released open-source under MIT license. 

By JL Kosev-Lex on 3 September 2026.
