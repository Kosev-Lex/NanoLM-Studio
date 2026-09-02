# V4 Model, Corpus & VRAM Planner

Open the **Training** tab and select **Model / VRAM Planner…**.

Enter:

- **Available VRAM** — the usable capacity of the GPU that will train the model.
  NanoLM Studio currently trains on one CUDA GPU; it does not combine multiple
  cards.
- **Active corpus tokens** — loaded from Library token counts when a current
  cache exists. Before tokenisation, the planner estimates tokens as active
  characters divided by four. You can replace the value manually.
- **Trial micro-batch** — the batch size whose peak VRAM should be assessed.

Press **Recalculate**. Every preset is then evaluated independently:

| Indicator | Good | Caution | Poor |
| --- | --- | --- | --- |
| VRAM | Estimated peak ≤70% of entered VRAM | 70–88% | Above 88%; likely OOM |
| Corpus | 10–30 tokens per model parameter | 3–10 or above 30 | Below 3 |

The preferred corpus band is a planning heuristic for training a model from
scratch, not a universal law. Corpus quality, duplication, domain breadth, and
the number of passes also matter. Below three tokens per parameter, memorisation
and unstable validation are strong risks. Above thirty, training can still work,
but a larger model may use the available data more effectively.

## Visual aid

Select any row to display:

- an estimated-VRAM bar against the entered GPU capacity;
- a corpus-balance bar with red, amber, and green zones;
- a plain-language explanation;
- suggested micro-batch, gradient accumulation, optimizer steps, and learning
  rate.

Double-click a row or press **Apply Selected Preset & Settings** to populate the
Training tab. A poor corpus match requires confirmation. A preset that cannot
fit even at micro-batch one cannot be applied.

## Presets at a 4,096-token vocabulary

| Preset | Architecture | Context | Parameters |
| --- | --- | ---: | ---: |
| Tiny | 4 × 256, 4 heads | 128 | 4.2M |
| Small | 6 × 512, 8 heads | 256 | 21.1M |
| Medium | 8 × 640, 10 heads | 256 | 42.2M |
| Large | 12 × 768, 12 heads | 512 | 88.6M |
| XL | 16 × 1,024, 16 heads | 512 | 206.2M |
| XXL | 24 × 1,280, 20 heads | 512 | 478.0M |

Parameter totals change slightly with vocabulary size. The program calculates
the exact current value rather than relying on these rounded examples.

## What the VRAM estimate includes

The estimate models FP32 weights and gradients, AdamW moments, a temporary
optimizer-step peak, mixed-precision activations, attention workspace, logits,
CUDA framework reserve, and allocator fragmentation. It is intentionally
conservative, but it cannot guarantee a successful allocation. Display use,
another model, browser GPU acceleration, drivers, and allocator behaviour can
reduce available memory. Close other GPU-heavy programs and retain headroom.
