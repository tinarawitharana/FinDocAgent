# Phase 5 — Attention Map XAI Layer

## Overview

Implemented a document-region explainability layer (Layer 5 of FinDocAgent) using
Qwen2-VL-2B-Instruct loaded locally. Generates heatmap visualisations showing which
regions of a document the model attended to when answering a question.
Directly addresses RQ3.

---

## Files Created

- `models/attention_map.py` — core implementation: loads model, runs inference,
  extracts attention weights, generates 3-panel figure
- `evaluation/attention_eval.py` — runs on 3 DocVQA examples, saves figures to
  `evaluation/results/attention_maps/`

---

## How It Works (Step by Step)

### Step 1 — Generate the answer
The document image and question are passed to Qwen2-VL-2B via the local model.
A standard `model.generate()` call produces a short text answer.
No attention weights are stored at this step — kept lightweight to save memory.

### Step 2 — Extract attention weights
A second forward pass is run on the same inputs with `output_attentions=True`.
This makes the model return internal attention weight matrices for all transformer layers.

From these matrices:
1. **Find image token positions** — Qwen2-VL marks image patches with a special
   `<|image_pad|>` token (ID: 151655). We find all positions in the input sequence
   where this token appears.
2. **Extract attention scores** — From the last transformer layer, attention weights
   are averaged across all heads, then indexed at the image token positions.
3. **Reshape to spatial grid** — `image_grid_thw` (returned by the processor) gives
   the patch layout as (temporal, height, width). Qwen2-VL uses a 2×2 patch merger,
   so the spatial grid is (h//2, w//2).
4. **Resize and overlay** — The attention grid is normalised, upsampled to the
   original image dimensions using bilinear interpolation, and overlaid as a hot
   colourmap heatmap.

### Step 3 — Save figure
Three-panel figure saved as PNG:
- Panel 1: Original document image
- Panel 2: Raw attention heatmap
- Panel 3: Overlay with question and predicted answer in title

---

## Key Technical Decisions

### Model: Qwen2-VL-2B-Instruct (local, not API)
- Downloaded from HuggingFace (~4.42GB)
- Loaded with `torch_dtype=torch.bfloat16` to minimise VRAM
- `attn_implementation="eager"` — required because Flash Attention does not return
  attention weights. Without this flag, `output_attentions=True` returns None.
- `device_map="auto"` — loads on GPU (HAMI-capped A100)

### Why local model, not DashScope API
The DashScope API is a REST endpoint — internal model weights and attention matrices
never leave Alibaba's servers. There is no mechanism to extract attention weights from
an API call. A locally loaded model is the only way to access internal attention.

### Why Qwen2-VL-2B and not 7B
The QMUL JupyterHub GPU allocation is capped at 8GB VRAM per user via HAMI
(Heterogeneous AI Memory Interconnect), even though the physical card is an A100 80GB
shared across multiple users. Qwen2-VL-2B in bfloat16 occupies ~4.5GB, leaving
~3.5GB for inference. The 7B model requires ~14GB in bfloat16 (or ~5-6GB in 4-bit)
which is not reliable within the 8GB cap.

---

## Memory Constraint — Problem and Solution

### Problem
The first attempt OOMed with:
`CUDA out of memory. Tried to allocate 9.20 GiB. GPU 0 has a total capacity of 8.00 GiB`

**Why:** DocVQA images are high resolution → hundreds of image patches → long token
sequences. The forward pass with `output_attentions=True` stores attention matrices
for all 28 transformer layers × 16 heads × seq_len × seq_len tokens. At ~500 token
sequences this requires ~7GB just for attention tensors, exceeding the available
~3.5GB of free VRAM after model loading.

CPU was also attempted but the JupyterHub kernel was killed — the Kubernetes pod has
a hidden per-user memory limit even though `free -h` shows 1.4TB available on the
physical node (that is node-level RAM, not pod-level).

### Solution: Force low-resolution image processing
Added `min_pixels` and `max_pixels` to the processor:

```python
_processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=32*28*28,
    max_pixels=128*28*28
)
