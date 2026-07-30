# Phase 10 — RQ3: SHAP Implementation, Synthetic Data, and Final 25-Document Sample

Continues directly from Phase 9 (which ended with the composite anomaly-score design proposed but not yet implemented). This phase covers everything from wiring that score into the live pipeline through to locking in the final 25-document human-evaluation sample.

---

## Composite anomaly score wired into the live pipeline

Refactored `agent/nodes/anomaly_checker.py` to add `compute_anomaly_features(fields)` and `anomaly_score(features)` as standalone, importable functions, with `anomaly_checker_node` calling them instead of duplicating the calculation inline. Re-ran `main.py` against the same test document before and after — identical output (same £-2186.69 discrepancy, same anomaly count), confirming the refactor changed nothing about behavior, only removed duplication.

---

## SHAP implementation and validation

Installed `shap`; wrote `evaluation/shap_anomaly_eval.py`, which wraps `anomaly_score()` in a `KernelExplainer` with a zero-vector ("no anomalies") background, and prints each document's SHAP attribution alongside the analytically-known closed-form expected value (since the scoring function is a simple weighted linear combination, `0.5 * feature`, the exact Shapley value is known in advance).

**Result: every single document's SHAP value matched the expected closed-form value exactly**, across all 6 originally-available bank statements. This is the validation step planned in Phase 9 — confirms the `KernelExplainer` setup is correctly implemented before trusting it further, rather than assuming it works.

---

## Data-quality problems found in the original 6 bank statements

Running the SHAP script surfaced real issues in documents that had been sitting in `data/samples/` unexamined:

- **`bank_statement_clean_word.pdf` is actually a SunTrust credit-card statement**, not an itemized bank statement — its line items are summary categories ("Previous Balance," "Purchases and Advances," etc.) that were never meant to sum to the stated total. The math-discrepancy rule doesn't apply to this document type at all; the 0.500 math_feature it produced was a false positive caused by applying the wrong rule to the wrong document shape, not a real anomaly.
- **`hsbc1.pdf` appears to be a duplicate of `hsbc_anomaly.pdf`**, not a distinct "clean" baseline — same account holder, same total, same date quirks. Its filename didn't reflect that it wasn't actually a separate clean control case.

This is exactly the kind of ambiguity a documented ground truth prevents — neither of these problems would have been caught without deliberately checking.

---

## Sourcing more bank-statement documents

Discussed at length how to source additional documents given the above problems, and where "real" vs. "synthetic" documents are appropriate:

- **Real personal bank statements are not an appropriate source**, even when findable online — unlike your SEC 10-K filings (public *by law*), personal financial records are private data regardless of whether a site hosts them; most "real bank statement" search results are either mislabeled synthetic data, fraud-document-generation tools, or genuinely leaked private data. None of these are usable in a dissertation.
- **Openly-declared synthetic datasets** (Kaggle "synthetic bank statement" datasets) turned out to mostly be tabular/CSV transaction data for fraud-detection ML, not document images — a structural mismatch with this project's VLM-based, image-reading pipeline.
- **Conclusion: self-generate**, the same way the original hand-crafted documents were made, but this time with a reproducible script and a documented ground-truth log — both for academic defensibility (a stated methodology beats "documents found online, provenance unclear") and to guarantee exact, known anomaly ground truth.

---

## Synthetic bank statement generator

Built `data/generate_synthetic_statements.py` (using `reportlab`) — a `generate_bank_statement()` function that takes a transaction list plus optional `inject_math_error` and `swap_dates` parameters, and writes both the PDF and a `ground_truth.json` recording exactly what was injected. Generated 4 documents covering the full 2×2 matrix: `statement_01_clean.pdf`, `statement_02_math_error.pdf` (£250 injected), `statement_03_date_error.pdf` (one date swap), `statement_04_both_errors.pdf` (£-120 + one date swap). Visual render confirmed the injected date swap and math discrepancy are both genuinely visible in the rendered PDF, not just in the metadata.

**Validation**: wrote `evaluation/validate_synthetic_anomalies.py`, comparing each synthetic document's known ground truth against what `extract_fields_from_document` + `compute_anomaly_features` actually detect. **Perfect match on all 4 documents, both math discrepancy and date-ordering detection** — the first time in this project a result could be checked against a genuinely known right answer rather than eyeballed for plausibility. Notably, the date-ordering rule correctly caught the injected anomaly here, unlike the earlier scanned `bank_statement_anomaly.pdf` case, because these reportlab-generated PDFs have a real text layer with no OCR ambiguity for the VLM to "smooth over."

---

## Human-evaluation instrument built

Designed 5 compliance criteria (1-5 Likert scale each), deliberately split by whether they require financial/compliance expertise:

- **Localization, Clarity, Actionability, Trust** — answerable without financial background (pure visual/logical judgment), so a non-expert second rater can meaningfully contribute.
- **SHAP Consistency** — bank-statement documents only, checking whether the SHAP breakdown matches what's visibly wrong (or not) in the document.
- **Correctness** — requires domain/ground-truth knowledge, kept as author-only.

Built `evaluation/human_eval_scoring_template.csv` (one row per rater per document) and `evaluation/human_eval_rater_instructions.md` (plain-language instructions written for a non-expert rater, explaining what they'll see and what each question means, with explicit reassurance that no finance background is needed).

**Evaluator plan changed mid-session**: the originally-planned second rater (marketing background) became unavailable due to other commitments. The study is now self-assessment only — a legitimate approach, but it means no inter-rater agreement statistic; this needs one explicit sentence in the dissertation's limitations section (self-assessment only, due to time constraints) rather than being silently different from the original two-rater plan.

**Sample size discussion**: considered reducing to 20 (workload) or expanding to 25+25 separate samples (more rigorous but double the work) before settling back on 25 total, combined/mixed — matching the Project Definition's stated plan, with the type mix (SEC 10-K vs. bank statement) preserved so both attention maps and SHAP are represented in the same study.

---

## Sourcing and building the final bank-statement set

User sourced additional bank-statement templates from TemplateLab (fillable PDF forms — legitimate, openly-fictional templates, the safe category discussed earlier). Reviewed three for structural compatibility with the anomaly rules:

- **SunTrust-style** (credit-card summary structure) — same incompatibility as `bank_statement_clean_word.pdf`; recommended against using as a rule-tested ground-truth case.
- **BNI-style and Bickslow Bank-style** — itemized transaction structures (date/description/amount), compatible with both rules, good candidates.

User filled in 5 templates (`bs1.pdf`-`bs5.pdf`) and, in the process, **deleted the 4 verified-good original bank statements** (`bank_statement_clean.pdf`, `bank_statement_anomaly.pdf`, `bank_statement_anomaly_word.pdf`, `hsbc_anomaly.pdf`) — caught via `git status` showing them as uncommitted deletions (recoverable, not lost). `bank_statement_anomaly_word.pdf` was subsequently restored.

**Ground truth established for all 6 non-synthetic bank statements** through iterative back-and-forth verification (checking displayed totals against actual line-item sums, checking date sequences), written to `data/samples/bank_statements_ground_truth.json`:

| File | Math discrepancy | Date-ordering anomaly |
|---|---|---|
| bs1.pdf | Yes — £965.00 | No |
| bs2.pdf | Yes — £150.00 | No |
| bs3.pdf | No (clean) | Yes — 2 out-of-order pairs |
| bs4.pdf | Yes — £1,000.00 | Yes — 1 out-of-order pair |
| bs5.pdf | No (clean; sourced online, not filled in by author) | No |
| bank_statement_anomaly_word.pdf | Yes — £2,186.69 | Yes — 2 out-of-order dates |

Combined with the 4 synthetic documents, this gives **10 fully ground-truth-documented bank statements** — every single one now has a known, recorded answer for both anomaly types, matching the rigor established with the synthetic batch.

---

## Final 25-document sample locked in

**10 bank statements**: bs1-bs5, `bank_statement_anomaly_word.pdf`, and the 4 synthetic statements — all with documented ground truth.

**15 SEC 10-K questions** (trimmed from an initial list of 17 to make room for the 2 extra bank statements beyond the original 8-document plan), selected from the actual per-question results in `evaluation/results/sec10k_agent_results.json` to deliberately span successes, near-misses, and different *types* of failure (pure retrieval misses vs. right-page-wrong-cell misreads), not just easy cases:

Q1, Q2, Q4, Q5, Q8, Q14, Q15, Q17, Q19, Q22, Q26, Q28, Q30, Q36, Q37 (question numbers from the 39-question SEC 10-K eval set).

---

## Remaining steps

1. Generate attention maps for all 25 documents (the resource-heavy, VRAM-constrained step per Phase 5's notes).
2. Run SHAP for the 10 bank statements and populate `human_eval_scoring_template.csv` with real file references, questions/context, and model outputs.
3. Run the human evaluation (self-assessment only) and write up results.
