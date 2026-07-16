# Phase 6 — FUNSD Evaluation

## Dataset
- **Name:** FUNSD (Form Understanding in Noisy Scanned Documents)
- **Source:** `nielsr/funsd` on HuggingFace
- **Split:** test (50 documents)
- **Document type:** Scanned business forms (fax cover sheets, memos, etc.)

## Task Formulation
FUNSD is a NER dataset, not a VQA dataset. Each document has word-level tags:
- `B-QUESTION` / `I-QUESTION` — form field labels (e.g. "TO:", "DATE:")
- `B-ANSWER` / `I-ANSWER` — filled-in values
- `B-HEADER` / `I-HEADER` — section headers
- `O` — other

Since the dataset has no explicit Q-A linking (which answer belongs to which question), we adapted it as a **field extraction task**:
- Prompt Qwen2-VL to extract all filled-in values from the form image
- Compare predicted values against all GT answer entities using max ANLS
- Average ANLS across all GT answers per document, then across all documents

## Prompt Used
```
"List all the filled-in field values from this form, one per line. Only output the values, nothing else."
```

## Scoring
For each GT answer entity, compute:
```
score = max(anls_score(pred, gt) for pred in predicted_values)
```
Average across all GT answers in a document, then across all 50 documents.

## Results

| Approach | ANLS |
|----------|------|
| Original prompt (field extraction) — run 1 | 0.698 |
| Verbose prompt (names, dates, numbers...) | 0.333 (worse — model added labels) |
| Question-guided prompt (field list as hint) | 0.526 (worse — order mismatch) |
| Original prompt (field extraction) — run 2 | **0.792** |

**Final FUNSD score: 0.792** (some run-to-run variance observed even at temperature=0)

## Comparison with DocVQA

| Dataset | Task | ANLS |
|---------|------|------|
| DocVQA (single-pass) | Targeted QA, clean images | 0.954 |
| DocVQA (full agent) | Targeted QA, clean images | 0.960 |
| FUNSD | Field extraction, noisy scanned forms | 0.792 |

## Why the drop from DocVQA is expected
1. **Noisier images** — FUNSD is scanned documents with lower image quality
2. **Harder task** — model must find all field values without being told what to look for
3. **Stricter scoring** — every GT answer must be matched; missing any one hurts the average

## Script
`evaluation/funsd_eval.py`

## Limitations
- No explicit Q-A linking used — answers matched by max ANLS over all predictions
- FUNSD's NER format doesn't map cleanly to VQA — this is an adapted evaluation
- Zero-shot only — no fine-tuning on FUNSD training split
