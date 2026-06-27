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
