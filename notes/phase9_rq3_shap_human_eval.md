# Phase 9 — RQ3 Human Evaluation Setup (attention maps + SHAP)

Continues from Phase 5 (attention map XAI layer, implemented but never formally evaluated) and the Project Definition's evaluation plan: *"The XAI module will be evaluated through a structured human assessment, in which attention map visualisations and SHAP explanations are scored against a predefined set of financial compliance criteria across a sample of 25 test documents."*

This phase is in progress — this note captures decisions and groundwork so far, not a finished result.

---

## Scope decisions made

**SHAP:** attempt to implement it properly rather than skip it, with attention-maps-only + a justification paragraph as the fallback if it proves infeasible.

**Evaluators:** two raters — Tinara, plus a friend (marketing background, no finance/compliance domain knowledge) as a second independent rater. Chosen over self-assessment-only specifically to get an inter-rater reliability statistic (e.g. Cohen's kappa) into the dissertation, which is stronger evidence than a single rater. The friend's non-finance background is a limitation worth naming explicitly in the write-up, but is also legitimate for criteria that don't require financial domain expertise (clarity of the explanation, whether the highlighted region plausibly matches the citation, whether it builds trust) — worth splitting criteria into domain-expert vs. non-expert-answerable when designing the instrument (not yet done).

**Document sample:** mixed — SEC 10-K pages (attention maps only, no anomaly score exists in QA mode) + bank-statement documents (attention maps AND SHAP, since only the PDF-extraction pipeline computes an anomaly score).

---

## Key finding: SHAP's real target is the anomaly score, not the VLM

Initially assumed SHAP would need to explain Qwen2-VL's extraction/answering behavior directly — which would have been infeasible: perturbation-based SHAP methods need many masked-input forward passes per document, and the attention-map work (Phase 5) already hit VRAM limits doing a *single* extra forward pass locally, or would cost a large number of paid DashScope API calls if done via the API.

Re-reading the Project Definition's Layer 5 description clarified the actual intent: *"SHAP values quantifying each field's contribution to the anomaly score."* This is SHAP over the **rule-based anomaly checker** (`agent/nodes/anomaly_checker.py`), not the VLM — a handful of numeric features, no VLM involved, cheap and fast. This reframing is what makes SHAP tractable within the project's time/compute budget.

**Consequence:** SHAP can only ever apply to documents that go through the PDF-extraction pipeline (`anomaly_checker_node`), not the SEC 10-K question-answering pipeline. `extractor.py`'s QA-mode branch never computes an anomaly score at all.

---

## Architecture bugs found and fixed along the way

**1. `retriever_node` crashed on image-only PDFs, and was doing wasted work regardless.**
`bank_statement_anomaly.pdf` has no extractable text layer (scanned/rendered image, not real text). `retriever_node` unconditionally ran full PDF text-extraction → chunking → Chroma indexing for every `.pdf` path, which produced zero chunks for this document and crashed (`ValueError: Non-empty lists are required for ['ids', 'metadatas', 'documents'] in add`, from Chroma's `collection.add()`).

Root cause was broader than the crash: in bank-statement/PDF-extraction mode (`main.py`, no `"question"` in state), `extractor.py`'s `#pdf mode` branch calls `extract_fields_from_document(document_path)` directly and never reads `state["retrieved_chunks"]` at all — so the entire ChromaDB retrieval step was wasted computation for this mode even when it didn't crash.

**Fix:** added a second early-exit in `retriever_node`, mirroring the existing image-file shortcut — `if not state.get("question"):` skips retrieval entirely and passes `document_path` straight through, since nothing downstream needs it in this mode.

**2. `explainer.py` assumed every anomaly dict had the same shape.**
Once the date-ordering rule (see below) started producing anomalies too, `explainer.py` line 24 crashed with `KeyError: 'stated'` — it hardcoded formatting for `math_discrepancy`-shaped dicts (`stated`/`calculated`/`difference` keys) and had no branch for `date_ordering`-shaped dicts (`out_of_order_count` key only).

**Fix:** branch on `a["type"]` when formatting each anomaly in the report loop, with a generic fallback (`else: report += f"...{a}"`) so a future third rule type won't require another explainer edit to avoid crashing.

---

## Extraction schema + new anomaly rule

**`models/qwen.py`:** added a `"date"` field to each `line_items` entry in the extraction JSON schema, requesting normalized `YYYY-MM-DD` format specifically so date-ordering comparisons are plain string comparisons (`"2023-11-28" < "2023-11-27"`) with no date-parsing library needed.

**`agent/nodes/anomaly_checker.py`:** added a second rule — checks whether line-item dates are non-decreasing; counts (not just flags) how many adjacent pairs are out of order, since a *count* is needed as a continuous feature for SHAP later (a boolean flag alone wouldn't give SHAP anything to meaningfully attribute).

---

## Interesting finding: the VLM silently "corrects" apparent date typos

`bank_statement_anomaly.pdf` (the scanned/image-only version) visibly prints "28 Dec 23" and "30 Dec 23" where "28 Nov" / "30 Nov" should be (an intentionally injected typo in this hand-crafted test document, sitting between otherwise-sequential November dates). The date-ordering rule reported **zero** anomalies on this document — not because the rule is broken, but because Qwen2-VL extracted `"2023-11-28"` and `"2023-11-30"` for those exact line items, silently normalizing the apparent inconsistency to fit the surrounding sequence rather than transcribing what's literally printed.

Confirmed by re-running `extract_fields_from_document` directly and inspecting the raw JSON — the model is doing plausibility-based "correction," not literal OCR, for this case.

This is worth a citation-worthy sentence in the dissertation's XAI/compliance discussion independent of anything else: **a VLM that silently self-corrects apparent errors in a financial document is itself a compliance risk** — an auditor relying on the extracted fields would never learn the source document had this discrepancy at all, since the model masked it before the anomaly rule ever saw it.

Switched to `bank_statement_anomaly_word.pdf` (the version with a real, selectable text layer — no OCR/VLM transcription step involved) as the working test case instead, since a prompt-only fix ("transcribe dates exactly as printed, do not infer or correct") was not guaranteed to work reliably. On this document, the date-ordering rule correctly caught 2 out-of-order dates alongside a math discrepancy — full pipeline (`retriever → extractor → anomaly_checker → explainer`) now runs end-to-end without crashing, with both anomaly types present in the same document.

---

## Existing document inventory (useful for the 25-doc sample)

`data/samples/` already contains 6 bank-statement-style PDFs, more than the 2 (HSBC/SunTrust) recorded in earlier memory: `bank_statement_clean.pdf`, `bank_statement_clean_word.pdf`, `bank_statement_anomaly.pdf`, `bank_statement_anomaly_word.pdf`, `hsbc1.pdf`, `hsbc_anomaly.pdf`. The `_word` suffix appears to distinguish real-text-layer versions from scanned/image-only versions of the same statement — worth checking each one's extractability before assigning it to the SHAP sub-sample, given what happened with `bank_statement_anomaly.pdf` above. All 6 were previously indexed in ChromaDB (confirmed via `list_collections()`), so they've been exercised by the pipeline before, just not through the anomaly-scoring path specifically until now.

---

## Composite anomaly score — designed, not yet implemented

To give SHAP something continuous and multi-featured to explain (a single boolean flag, or one rule alone, gives SHAP nothing interesting to distribute credit across), designed:

- `math_feature = min(abs(stated - calculated) / max(abs(stated), 1), 1.0)` — discrepancy as a proportion of stated total, capped at 1.0.
- `date_feature = min(out_of_order_count / max(len(dates) - 1, 1), 1.0)` — proportion of adjacent transaction pairs out of order, capped at 1.0.
- `composite_score = 0.5 * math_feature + 0.5 * date_feature` — equal weighting, chosen because it's the simplest defensible choice without needing to argue one rule matters more than the other.

Deliberately linear, so Shapley values have a closed-form answer (`0.5 * feature_value` per feature) — this lets the eventual `shap.KernelExplainer` output be checked against the analytically-correct answer as a validation step, rather than trusting an approximation blind. Worth stating explicitly in the dissertation methodology as a validation step, not just an implementation detail.

Not yet added to the codebase as of this note — proposed as a standalone `compute_anomaly_features()` / `anomaly_score()` pair of functions in `anomaly_checker.py`, separate from (not yet wired into) the existing rule-checking logic in `anomaly_checker_node`, to avoid disturbing the now-working report path.

---

## Remaining steps (not started)

1. Wire `compute_anomaly_features()` into `anomaly_checker_node` (single source of truth instead of duplicated calculation).
2. `pip install shap`; implement the `KernelExplainer` over `anomaly_score`, validate against the closed-form linear answer.
3. Run extraction + scoring + SHAP across the bank-statement sample; confirm the 6 existing documents are sufficient or source/create more.
4. Generate attention maps for the full mixed 25-document sample (SEC 10-K + bank statements).
5. Define compliance criteria and build the actual scoring instrument (Likert-style rubric, split by whether a criterion needs financial domain knowledge) for both raters.
6. Run the human evaluation; analyze results including inter-rater agreement.
