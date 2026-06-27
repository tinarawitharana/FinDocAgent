# Phase 3 Findings — Qwen2-VL Integration

## Date: 24 June 2026

## What was built
- models/qwen.py — Qwen2-VL-max via DashScope API
- Real Extractor node replacing regex baseline
- Full pipeline: Retriever → Qwen2-VL Extractor → Anomaly Checker → Explainer

## Test: HSBC anomaly document
- Vendor: HSBC UK ✓
- Stated total: £2,139.27 ✓ (correct closing balance)
- Line items: 11 transactions extracted ✓
- Anomaly detected: £-2,186.69 math discrepancy ✓

## Key comparison vs regex baseline
- Regex: extracted £390,000 (wrong — picked Credit Limit)
- Qwen2-VL: extracted £2,139.27 (correct closing balance)
- This directly answers RQ2: multimodal visual encoding 
  significantly outperforms text-only extraction

## Next steps
- Add date-ordering anomaly check
- Build baseline.py for DocVQA evaluation
- Run systematic ablation experiments

## Evaluation Results — June 27 2026

### Baseline Comparison (ANLS scores)

| Method | Avg ANLS |
|--------|----------|
| Regex Baseline | 0.000 |
| Qwen2-VL Single-Pass | 0.914 |
| Improvement | +0.914 |

### Document 1: SunTrust Bank Statement (Clean)
Regex baseline:
- Total extracted: 390,000.0 (WRONG — picked Credit Limit)
- Account holder: Unknown
- Bank name: Unknown
- All ANLS scores: 0.0

Qwen2-VL single-pass:
- Total extracted: 3,898.57 (CORRECT)
- Account holder: John Smith (CORRECT)
- Bank name: SUNTRUST (CORRECT)
- All ANLS scores: 1.0

### Document 2: HSBC Bank Statement (Anomaly)
Regex baseline:
- Total extracted: 2,212.71 (WRONG — picked first transaction)
- Bank name: Unknown
- All ANLS scores: 0.0

Qwen2-VL single-pass:
- Total extracted: 2,139.27 (CORRECT — closing balance)
- Bank name: HSBC UK (CORRECT, minor string mismatch vs "hsbc")
- ANLS scores: 1.0, 0.571

### Key observations for dissertation

1. RQ2 ANSWERED: Qwen2-VL multimodal encoding improves extraction
   accuracy from 0.000 to 0.914 ANLS — a complete and dramatic result.

2. Regex failure mode: linearised PDF text loses spatial relationships
   between labels and values. "Total Amount Due" label appears near
   "390,000.00" (Credit Limit) in plain text even though they are
   in completely different table cells visually.

3. HSBC string match issue: Qwen2-VL returned "HSBC UK" vs ground
   truth "hsbc" — this is a string normalisation issue, not a real
   failure. Qwen2-VL correctly identified the bank. Mention this
   as a limitation of exact-match ANLS evaluation in dissertation.

4. Speed: Regex = 0.07-0.22s, Qwen2-VL = 8-10s per document.
   Trade-off between speed and accuracy. For compliance use case,
   accuracy is more important than speed.

5. Next: Run agent vs single-pass comparison to answer RQ1.
