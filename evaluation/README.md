# Evaluation scripts

This folder answers three research questions with a consistent pattern: run a model
(or the full agent) on a benchmark, score it with `metrics.py`, and write results to
`evaluation/results/`.

## Naming pattern

Most scripts follow `<dataset>_<model>_eval.py`. `<model>` omitted means Qwen2-VL-max,
the model used throughout the rest of the project (via DashScope).

| Dataset | Direct model call | Full agent (RAG/retry) |
|---|---|---|
| DocVQA | `docvqa_eval.py` | `docvqa_agent_eval.py` |
| FUNSD | `funsd_eval.py` | — |
| SEC 10-K (oracle page) | `sec10k_eval.py` | `sec10k_agent_eval.py` (RAG-retrieved page) |
| Bank statements (anomaly detection) | — | `bank_statement_agent_eval.py` |

The "direct model call" scripts give the model the correct page directly (or, for
DocVQA/FUNSD, the dataset's own image) — they're the reference point that the agent
evals compare RAG-retrieval quality against.

## Multi-model baseline comparison

`docvqa_*_eval.py` / `funsd_*_eval.py` re-run the same DocVQA/FUNSD sample against other
models, to put Qwen2-VL-max's numbers in context:

| Model | Script suffix | How it runs |
|---|---|---|
| GPT-4o | `_gpt4o_eval.py` | Hosted API (OpenAI) |
| Gemini 3.5 Flash | `_gemini_eval.py` | Hosted API (Google) — capped by free-tier daily quota, see module docstring |
| Kimi (Moonshot AI) | `_kimi_eval.py` | Hosted API |
| Gemma 3 4B | `_gemma_eval.py` | Local HF model, bf16 with 4-bit fallback on OOM |
| InternVL3.5-8B-HF | `_internvl_eval.py` | Local HF model |
| SmolVLM2-2.2B-Instruct | `_smolvlm_eval.py` | Local HF model |

`gpt4o_error_analysis.py` follows up on the GPT-4o results, classifying low-ANLS answers
as genuine misreads vs. correct-but-verbose answers the ANLS metric penalizes unfairly.

Local-model scripts need the corresponding weights downloaded and a GPU; see each file's
module docstring for VRAM/disk tradeoffs hit while picking these particular checkpoints.

## Naive text-only baseline (RQ2)

To answer "does multimodal visual encoding beat naive text extraction," three scripts
compare a regex/keyword-matching baseline (no VLM, no vision, no API call) against the
Qwen2-VL-max numbers above, all scored with the same `anls_score`:

| Domain | Script | Compares against |
|---|---|---|
| Bank statements (2 hand-crafted docs) | `baseline.py` | `models/qwen.py`'s field extraction |
| DocVQA (same 50-sample slice) | `docvqa_baseline_eval.py` | `docvqa_eval.py` |
| FUNSD (same 50-doc slice) | `funsd_baseline_eval.py` | `funsd_eval.py` |

`docvqa_baseline_eval.py` and `funsd_baseline_eval.py` share their extraction logic in
`regex_utils.py`: keyword-overlap window search (nudged toward a date/number regex match
when the question implies that answer type) for DocVQA, and a "label ending in `:` →
next 1-3 words are its value" heuristic for FUNSD — both operating on the datasets' own
OCR `words` field, so no separate OCR step is needed. Being free of any API cost, these
are safe to run directly rather than needing to be handed off.

Results: DocVQA regex 0.033 vs. Qwen2-VL-max 0.952; FUNSD regex 0.194 vs. Qwen2-VL-max 0.792.

## Explainability (RQ3)

- `attention_eval.py` / `models/attention_map.py` — attention-map visualizations from a
  locally-loaded Qwen2-VL-2B (needed for raw attention weights, unlike the hosted
  Qwen2-VL-max used everywhere else).
- `human_eval_attention_maps.py` — generates paired RAG-page vs. oracle-page attention
  maps for human rating (see `human_eval_rater_instructions.md` and
  `human_eval_scoring_template.csv`).
- `shap_anomaly_eval.py` — SHAP analysis over the anomaly checker's two engineered
  features (math discrepancy, date ordering).

## Data prep / one-off scripts

- `extract_bank_statements_once.py` — caches Qwen2-VL-max field extractions for all bank
  statement PDFs to `data/samples/bank_statements_extracted.json`, so `shap_anomaly_eval.py`
  doesn't need to re-call the API.
- `validate_synthetic_anomalies.py` — sanity-checks that the anomaly checker actually
  fires on the synthetic documents deliberately constructed to contain those errors.

## Running an eval

Each script is standalone and run directly from the repo root (they add the repo root to
`sys.path` themselves, and write results using paths relative to it):

```bash
python evaluation/docvqa_eval.py
```

Hosted-API scripts need the relevant key in your environment (see `.env.example` at the
repo root). Local-model scripts need their weights downloaded on first run and a GPU for
reasonable speed.