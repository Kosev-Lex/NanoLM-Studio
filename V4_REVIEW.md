# NanoLM Studio: critique and v4 direction

## What was already good

The supplied program had a coherent product idea: it treated a corpus as a set
of auditable documents, used one readable cleaning pipeline, placed EOS tokens
at document boundaries, embedded model configuration in checkpoints, offered a
fixed-seed validation pass, and kept expensive training work off the Tk event
thread. The Library → Tokenizer → Training → Chat → Glass Box sequence is a
strong teaching workflow because users can see each stage instead of receiving
a black-box training command.

## Problems found in the supplied archive

### Release and packaging

- The folder was named `nanolmv4`, but modules and tests imported `nanolm`.
  `python main.py` failed immediately with `ModuleNotFoundError`.
- Version labels disagreed across the package, launcher, README, and UI.
- The README documented a package structure the ZIP did not contain.

### Data safety and corpus integrity

- `smoke_test.py` deleted the production `data/corpus.db` and wrote caches and
  checkpoints into production locations.
- Passing a custom database path to `CorpusLibrary` did not redirect raw and
  cleaned document files, so supposedly isolated tests still modified global
  data.
- Database rows and text files were written in separate unprotected operations.
  An interrupted write could leave orphaned or inconsistent records.
- HTML import included script/style text; several extractor edge cases lacked
  useful structure or recovery behaviour.

### Cache correctness

- A cache was considered valid if its tokenizer matched. It was not invalidated
  when active documents changed or when a document was re-cleaned.
- Cache arrays and metadata were written non-atomically.
- Per-document token counts were committed during a build, so a failed build
  could leave partly updated statistics.
- Whole-document validation selection favoured the shortest documents and used
  repeated quadratic sums.

### Model and generation

- Parameter counting subtracted the tied output weight even though PyTorch had
  already de-duplicated it, substantially under-reporting model size.
- Generation recomputed the complete context for every new token.
- Empty prompts crashed, invalid sampling values were not checked, and
  temperature zero was approximated rather than performing greedy decoding.
- Repetition penalty copied recent token IDs to Python and looped over them on
  every generated token, forcing avoidable device synchronisation.
- Checkpoints were vulnerable to partial writes.

### Training semantics

- “Total steps” counted micro-batches, while schedules and UI expectations mixed
  micro-batch and optimizer-step meanings.
- If total steps were not divisible by gradient accumulation, the final partial
  gradients were discarded even though their tokens were counted.
- Resume did not preserve data RNG, PyTorch RNG, or early-stop patience state.
- The random crop sampler omitted the final valid starting position and rejected
  an array containing exactly one valid sequence.
- Weight decay was applied to biases and normalisation parameters.
- Stop requests were not observed during sample generation or validation.
- A run that never reached validation could finish without a usable `best.pt`.

### Desktop UI

- The Library scrollbar object was created but never managed with `grid` or
  `pack`, so it was invisible. The log window to its right had its own unrelated
  scrollbar, which made the defect easy to misdiagnose.
- The library table had no filter, sort controls, active-only view, horizontal
  scrolling, double-click action, or row context menu for large corpora.
- The chat worker read a Tk variable directly, violating the program's own
  threading rule.
- Checkpoint loading and Glass Box inference ran on the UI thread and could
  freeze the application.
- The event pump drained the entire queue at once, allowing fast token streaming
  to monopolise the UI thread.
- ByteLevel decoded output was assumed to grow by simple string appends. That is
  not always true at token boundaries and could display missing or duplicated
  characters.
- Controls did not consistently disable during long-running work, malformed
  numeric input could escape as a Tk exception, and window close did not
  coordinate with active workers.

## What v4 develops into

V4 is a dependable single-user “small language model laboratory.” It is still
compact enough to read and modify, but its storage and execution semantics are
now suitable for real experiments rather than only a demo. The UI makes corpus
curation practical at larger document counts, training runs are reproducible,
and model behaviour can be inspected and exercised without obvious correctness
traps.

The strongest future direction is a project-based experiment manager:

1. Named workspaces with immutable dataset/tokenizer revisions.
2. A run browser that compares loss, perplexity, throughput, configuration, and
   generated samples across experiments.
3. Import diagnostics and configurable cleaning recipes with per-step previews.
4. Train/validation assignment controls, document weighting, and contamination
   checks.
5. Pretrained-model adapters (LoRA/QLoRA) alongside the from-scratch NanoLM path.
6. Checkpoint retention policies, export formats, and a lightweight inference
   server.
7. Automated evaluation sets and regression reports.

Those additions should be built on explicit versioned project metadata rather
than more global files. V4's redirectable data root, fingerprints, atomic writes,
and embedded configurations are the foundation for that transition.

## Capacity-planning extension

V4 now includes Large, XL, and XXL presets up to approximately 478M parameters,
plus a dedicated Model, Corpus & VRAM Planner. This keeps the larger options
from becoming attractive-but-dangerous dropdown entries: users can see VRAM
pressure and corpus suitability separately, inspect coloured fit meters, obtain
a model recommendation, and apply a conservative batch/accumulation/training
configuration directly to the Training tab.
