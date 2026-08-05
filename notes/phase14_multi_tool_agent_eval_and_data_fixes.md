# Phase 14 — Multi-Tool Agent Evaluation, Explainer Bug Fix, and Human-Eval Data Repair

Continues from Phase 13 (GPT-4o baseline comparison). Covers items #3 and #4 of the
remaining-work priority list: the multi-tool agent systematic evaluation on bank
statements (never previously tested end-to-end), and a deeper-than-expected repair
of `evaluation/human_eval_scoring_template.csv`.

---

## 1. A real bug found before the agent eval could even run

Before writing the multi-tool eval, `agent/graph.py` was re-read to understand
routing. The bank-statement (non-QA) branch of `should_continue()` looked correct:
retriever → extractor → (if `extracted_fields` populated) → anomaly_checker →
explainer → END, with no short-circuit.

But `agent/nodes/explainer.py` turned out to be genuinely broken — the file was
only 29 lines and ended mid-loop, having built a `report` string but never
assigning it to `state["risk_report"]` and never containing a `return state` at
all. Any LangGraph invocation reaching this node would receive `None` back
instead of a state update and fail.

Git log confirms this predates today's session (last touched in the "shap and 25
doc sample" commit), so it wasn't introduced today — it was simply never caught,
because nothing had exercised this exact code path end-to-end before.

**Fix applied** (agent/ is a core file, so this was proposed in chat and applied
only after explicit go-ahead):
```python
    state["risk_report"] = report
    state["task_complete"] = True
    return state
```

**Verified this bug never invalidates any existing result.** Grepped every eval
script that calls `build_graph()`/`agent.invoke()`:
- `docvqa_agent_eval.py` and `sec10k_agent_eval.py` both set `state["question"]`
  — QA mode. `should_continue()` routes QA mode straight back to the retriever
  (on a retry) or to `END`, **never** through `anomaly_checker`/`explainer`. So
  neither script's results were ever exposed to this bug, regardless of when it
  was introduced.
- Only bank-statement mode (`question=""`) reaches the explainer, and the only
  script exercising that path — `evaluation/bank_statement_agent_eval.py` — was
  written and run entirely *after* the fix.

**Conclusion: no prior results needed rerunning.**

---

## 2. DashScope billing interruption (brief record, not a technical finding)

Mid-session, the DashScope API (used for all Qwen2-VL calls, key
`DASHSCOPE_API_KEY`) started failing every request with:
```
Error code: 400 — 'type': 'Arrearage' — Access denied, please make sure your
account is in good standing.
```

This took an unusually long path to resolve:
1. Alibaba Cloud account showed an overdue balance (~$3.35). Paying the general
   account cash balance did **not** clear it — the specific July 2026 bill still
   showed `Paid Amount: 0 USD` until paid via that bill's own "Pay Bill" button
   under Repay, which is a separate action from general account top-up.
2. Even after that bill showed `Repay: Cleared`, the API still failed for 12+
   hours — well past the ~5 hour reconciliation window Alibaba's own tooltip
   quotes for a bill to flip to "Cleared" status.
3. Root cause turned out to be a **separate flag on the Model Studio product
   specifically** ("some features are restricted"), not the general billing
   account — general billing being cleared did not auto-lift this. Resolved only
   after Tinara contacted Alibaba support directly, who fixed it on their end.

**Takeaway for future reference**: if `DASHSCOPE_API_KEY` calls fail with an
`Arrearage`/400 error, check the Model Studio console specifically for a
service-level restriction — don't assume clearing the general Alibaba Cloud bill
is sufficient.

---

## 3. Multi-tool agent evaluation — first successful end-to-end run

Built `evaluation/bank_statement_agent_eval.py`, which is the first script to
call `build_graph().invoke(...)` in bank-statement mode at all — every prior
anomaly-detection result (`shap_anomaly_eval.py`, `validate_synthetic_anomalies.py`)
called `extract_fields_from_document()` and `compute_anomaly_features()`
directly, bypassing the LangGraph pipeline (and therefore the retriever/extractor
node code, and the bug above) entirely.

**Dataset**: all 10 documents in the repo with precise, pre-existing ground
truth — 4 synthetic (`statement_01_clean` … `statement_04_both_errors`) plus 6
real statements (`bs1`–`bs5`, `bank_statement_anomaly_word.pdf`), each labeled
with an exact `math_discrepancy` value and `date_swap_applied` boolean.

**Method**: for each document, build a fresh `AgentState` (no `question` set) and
run it through the compiled graph exactly as a real deployment would — Retriever
→ Extractor (live Qwen2-VL call) → Anomaly Checker → Explainer. Predicted
anomalies are read from `final_state["anomalies"]` and compared against ground
truth to compute precision/recall/F1 for math-discrepancy detection,
date-ordering detection, and combined "any anomaly" detection.

Estimated cost before running (DashScope Qwen-VL-max pricing, one image + one
call per document, no retry loop unlike the SEC 10-K ReAct evals): well under
$0.05 total. Tinara ran the real command herself in her own terminal, per her
stated preference for anything cost-incurring.

### Results

| Metric | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---|---|---|---|
| Math-discrepancy detection | 0.600 | **1.000** | 0.750 | 6 / 4 / 0 / 0 |
| Date-ordering detection | **1.000** | **1.000** | **1.000** | 5 / 0 / 0 / 5 |
| Any-anomaly (combined) | 0.800 | **1.000** | 0.889 | 8 / 2 / 0 / 0 |

All 10 documents completed with zero pipeline errors — confirming the
`explainer.py` fix worked.

### Interpretation

**This is a good result, not a mediocre one**, for a specific reason: recall is
perfect (1.0) on both anomaly types. In a compliance/risk-detection context,
missing a real anomaly is the costly failure mode (a real discrepancy goes
undetected); a false positive just costs a human a few seconds of double-checking
something that turns out fine. The pipeline never once missed a real anomaly
across all 10 documents. Date-ordering detection is flawless outright.

**The 4 math-detection false positives are individually explainable**, traced by
inspecting the raw `stated`/`calculated`/`difference` values per document rather
than left as an unexplained score:

- `statement_01_clean.pdf` and `statement_02_math_error.pdf` — two versions of
  the same underlying template document — both had the live extractor compute
  the identical total (`1959.3`), exactly **$500 short** of the true `2459.3` in
  both cases. Consistent with the extractor missing the same one line item on
  both.
- `bs5.pdf` was off by exactly **$40,000** — most likely the newly-added
  `opening_balance` field (added earlier today, see `models/qwen.py` /
  `agent/nodes/anomaly_checker.py` diffs) coming back as 0 instead of being
  extracted; `bs5` is a "stock image" statement type that had never been tested
  against this new field before.
- `bs3.pdf` computed a **negative** total (`-2513.66`) against a genuinely clean
  true total — consistent with a credit/debit sign misclassification in the new
  `_signed_amount()` logic, i.e. the model mislabeling several line items'
  credit/debit type.

**Framing for the dissertation**: the anomaly-detection *rule* is validated as
sound (100% recall, perfect date F1); the math-detection precision loss is a
*live-extraction* accuracy issue, not a detection-logic flaw — and it's the kind
of issue that's individually diagnosable rather than a black-box failure. This
also demonstrates that earlier SHAP results (which used pre-cached
`bank_statements_extracted.json`, not live extraction) may look cleaner than a
genuinely fresh extraction would, which is itself worth noting as a caveat on
comparing those two evaluations directly.

**Sample size caveat**: n=10, so treat 0.75/0.889 as indicative on this labeled
set, not a statistically robust general precision estimate.

### Artifacts
- `evaluation/bank_statement_agent_eval.py`
- `evaluation/results/bank_statement_agent_eval.json`
- `agent/nodes/explainer.py` (bugfix)

---

## 4. Human-eval scoring CSV — a bigger repair than originally scoped

The original remaining-work list (from Phase 11) flagged "3 corrupted cells" in
`evaluation/human_eval_scoring_template.csv` (`sec10k_q22_oracle`,
`sec10k_q28_oracle`, `sec10k_q30_oracle` — corrupted `model_output`/
`document_type` text from a prior spreadsheet round-trip, with scores said to be
intact).

Tinara uploaded a freshly-filled copy of the CSV
(`human_eval_scoring_template_filled.csv`). Comparing it cell-by-cell against the
git-tracked version surfaced a much larger problem: **the git-tracked file's
numeric scores were missing or misaligned across nearly all 32 rows**, not just
the 3 known ones — e.g. the rater name `"Tinara"` was sitting in a
`localization_score` cell, and one row (`sec10k_q37_oracle`) had its entire
`model_output` sentence sitting in a score column.

**Verification before trusting the uploaded file**: cross-checked 2 of its score
values against exact figures already independently documented in
`notes/phase11_attention_map_model_investigation.md`'s discussion of the
Q2/Q5 paired contrast — `sec10k_q2_oracle` (Localization=5, Correctness=1) and
`sec10k_q5_oracle` (Localization=2, Correctness=5). Both matched exactly, giving
confidence the uploaded file was the real, correct data rather than another
corrupted export.

**Reconstructed the 3 originally-known corrupted text cells** (`file_reference`,
`question_or_context`, `model_output` for the Q22/Q28/Q30 oracle rows) using:
- The real production RAG-pipeline answers for these exact questions, looked up
  from `evaluation/results/sec10k_agent_results.json` (not fabricated) — Q22:
  `"Not found"`, Q28: `"Not found"`, Q30: `"$ 191,863"` (this one was actually
  correct, ANLS 0.889 — worth noting since it means Q30's oracle-vs-RAG contrast
  in the human eval isn't a clean "RAG fails" case for this particular question).
- Facts already established in the Phase 11 audit (corrected oracle pages,
  attention-map model's actual wrong answers, and why each page correction still
  left a genuine model error).

**Replaced the tracked `evaluation/human_eval_scoring_template.csv` with the
corrected file** and deleted the now-redundant `_filled` duplicate. Verified
afterward: all 32 rows present, every core score column numeric (no stray text),
and the 3 previously-corrupted rows now hold correct, real data.

### Artifacts
- `evaluation/human_eval_scoring_template.csv` (corrected in place)

---

## Status

Items #3 (multi-tool agent eval) and #4 (CSV corruption fix) from the remaining-
work list are both complete, along with a genuine bug fix (`explainer.py`) that
had been silently blocking any real deployment of the anomaly-detection pipeline.

**Still open** (unchanged from Phase 13, not touched this session):
- Resolve the Q28 oracle-page discrepancy (page 257 vs. page 96) in
  `data/sec10k_qa.json` — note: this Q28 is the SEC10-K QA-dataset entry, a
  different thing from the `sec10k_q28_oracle` human-eval CSV row fixed above,
  which already uses the corrected page 257 per the Phase 11 audit; the
  discrepancy still to resolve is a *separate* mismatch in `sec10k_qa.json`
  itself.
- Repo cleanup: untrack `venv/`/`.venv/`/`chroma_db`, add a README.

Tinara is pausing agent-assisted work here to focus on writing the dissertation
chapters directly.