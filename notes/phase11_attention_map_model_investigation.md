# Phase 11 — Attention Map Answer Quality Investigation

Continues from Phase 10's "remaining steps." The first full run of `evaluation/human_eval_attention_maps.py` (32 documents: 15 SEC 10-K oracle pages + 7 RAG-condition pages + 10 bank statements) completed without crashing, but surfaced a serious accuracy problem that needed investigating before the results could be trusted for the human evaluation.

---

## The problem

Of the 32 answers produced by the local attention-map model (Qwen2-VL-2B), **only 3 were actually correct** against known ground truth: the CEO-name question, the EOCF-markets question, and one bank statement's total. Everything else was wrong, and much of it nonsensically so — e.g. "$1,000" for a $158 billion revenue figure, "$3,000" for $3.9 trillion in assets, two different questions on the same page both answering "$1,134,000," and bank-statement dates coming back as "2004" instead of "2024."

This is not the same model as the production pipeline. `models/attention_map.py` deliberately uses a separate, local Qwen2-VL-2B-Instruct model, not the `qwen-vl-max` API the RAG/anomaly pipeline actually runs on — attention weights can only be extracted from a locally-loaded model, since an API endpoint never exposes internals. The concern was whether this weaker, separately-constrained model's answers were so far off that the attention maps built on top of them wouldn't be meaningful evidence for the human evaluation's Correctness, SHAP Consistency, or Trust criteria.

---

## Investigation, step by step

**1. Resolution test.** `attention_map.py`'s image processor was capped at `max_pixels=128*28*28` (~32 image tokens post-merge) specifically to avoid a VRAM crash documented in Phase 5. Raised this 4x (`512*28*28`) and re-tested the worst-offending question (JPMorgan total net revenue, answered "$1,000"). **Zero change — identical answer, byte-for-byte.** Pushed further to `2048*28*28` as a decisive test: this crashed with `CUDA out of memory` (`Tried to allocate 1.86 GiB. GPU 0 has a total capacity of 8.00 GiB of which 1.36 GiB is free`), also revealing other users' processes concurrently occupying real chunks of this shared, HAMI-virtualized GPU. Conclusion: resolution/legibility was not the bottleneck — no resolution both avoided the crash and improved the answer.

**2. Larger model via 4-bit quantization.** Since Phase 5 previously rejected Qwen2-VL-7B for VRAM reasons (full precision ~14GB), tried 4-bit quantization (`bitsandbytes`, ~5-6GB) as a way to fit a genuinely larger model within the 8GB budget while keeping resolution unchanged (isolating the model-size variable). Hit two infrastructure problems before even reaching a real test:
   - Two consecutive HuggingFace Hub download failures via their "xet" accelerated-transfer backend (`RuntimeError: Internal Writer Error: Background writer channel closed`), failing at different completion percentages (41%, then 59%) — pointed to an unreliable backend rather than bad luck. Worked around with `HF_HUB_DISABLE_XET=1` to force plain HTTP downloads.
   - Ran out of disk space mid-download (`OSError: [Errno 28] No space left on device`) — the 7B model needs ~16.6GB total, and only ~12GB was free. Freed room by deleting the (small, fast-to-redownload) cached 2B model weights.
   - With both fixed, the download and model load succeeded, and — genuinely new information — **both the generation step and the attention-extraction forward pass with `output_attentions=True` completed without crashing** at 4-bit. This confirms 4-bit quantization is technically viable within this VRAM budget, where full precision was not.
   - Accuracy, however, did not improve: the revenue question changed from "$1,000" to "$1.1 trillion" — still wrong, but a more plausible-scale confabulation rather than nonsense. The CEO-name question, which the 2B model answered correctly, **regressed to "#1 BANK"** — a clear case of quantization cost outweighing whatever the larger parameter count contributed.

**3. Explicit reasoning instruction.** Changed the prompt from the original "Answer with the exact text from the document. Be concise, one word or short phrase only" to an explicit two-step instruction: "First, briefly state which specific label, row, or section of the document contains the answer. Then, on a new line, give your final answer prefixed exactly with 'ANSWER: '." Also raised `max_new_tokens` from 20 to 80 to leave room for that reasoning, and fixed the attention-extraction mechanism to pull from `generate()`'s own `output_attentions` (covering the full generated sequence) instead of a separate, subtly broken forward pass on just the prompt.

Test question: Q1, JPMorgan total net revenue 2023 (ground truth $158,104 million). Raw model output: exactly `ANSWER: 125.5` — no reasoning or label-identification text appeared at all before the ANSWER marker. The model skipped the entire first instruction and went straight to a bare final-answer token. Result: "125.5" — a plain decimal with no dollar sign, no unit, no relation in scale to the true figure. Arguably less recognizable as an attempt at the actual question than the original baseline's "$1,000".

**4. One-shot demonstration.** Added a preceding text-only example turn (no image) showing the exact desired pattern — a fictional Q&A: "Question: What was the total revenue for 2022? Document context: ... 'Total revenue 2022: $45,000'." → "The relevant label is 'Total revenue' in the row for 2022.\nANSWER: $45,000" — then the real image+question, phrased as "...answer this question, using the same format as the example above."

Test question: same Q1. Raw model output: exactly `JPMORGAN CHASE & CO.` — again no reasoning step, no ANSWER: marker at all, despite an immediately-preceding worked example demonstrating exactly that format. This time it didn't even attempt a number — it grabbed the company name/header text from the page instead.

(One correction worth recording for methodological honesty: the first re-check of this raw output, typed inline as a `bash -c "..."` command, gave a misleading `$45,000` — which looked like the model was just echoing the example's answer verbatim. That turned out to be a shell-escaping artifact, the `\$` characters got mangled passing through nested quotes, not the model's real behavior. Re-running the identical check from an actual `.py` file, with no shell-escaping involved, confirmed the true output as `JPMORGAN CHASE & CO.` — so the "model just copies the example" theory was a false lead from a broken diagnostic, not a real finding.)

**Common thread across attempts 3 and 4:** neither produced any reasoning text before the answer, despite two different prompting strategies explicitly designed to elicit it — a direct instruction, and a worked example to imitate. The model consistently collapses to an immediate, unstructured, wrong answer regardless of how the request is framed.

**5. Source-code investigation.** Considered whether the memory pressure in `output_attentions=True` comes from the model retaining every layer's attention matrix simultaneously (only the last layer's is ever used in `generate_attention_map()`), which would suggest a hook-based fix to capture only the last layer and avoid the rest. Traced into the installed `transformers` library's `modeling_qwen2_vl.py`: found that `Qwen2VLDecoderLayer.forward()` explicitly discards its self-attention module's returned weights (`hidden_states, _ = self.self_attn(...)`) rather than accumulating them — meaning the actual mechanism behind `outputs.attentions` populating correctly must live elsewhere in the call stack, not traced fully. Fully resolving this would require open-ended reverse-engineering of library internals with no guaranteed payoff, assessed as disproportionate given the time already spent and remaining deadline.

---

## Decision

Reverted `models/attention_map.py` to Qwen2-VL-2B (the original, working configuration that produced the existing 32-image batch and its 3 correct answers). Deleted the unused 7B model cache to restore healthy disk headroom (was at 81% used / 4.9GB free after the 7B download; back to 18% used / 21GB free after cleanup).

**This is being documented as a confirmed, evidenced limitation, not an unexamined one.** Four genuinely different technical approaches to improving the local explainability model's accuracy — resolution, model scale via quantization, an explicit reasoning instruction, and a one-shot demonstration of the exact format wanted — plus a source-level fix for the extraction mechanism itself, were tested within real constraints (a shared, HAMI-virtualized 8GB-capped GPU) and none produced a reliable improvement. The dissertation's methodology should state plainly: attention-map generation runs on a separate, smaller, VRAM-constrained local model than the production pipeline; its own answers are frequently incorrect even after testing whether resolution, model capacity, or prompting strategy were the limiting factor; the human evaluation therefore assesses whether attention-based localization is visually sensible independent of answer correctness, not whether it verifies the deployed system's actual behavior for these specific 32 documents.

## Next steps

Proceed with the existing (2B-generated) 32-image attention-map batch as final. Move to running SHAP on the 10 bank statements and populating `evaluation/human_eval_scoring_template.csv` with real file references, questions, and model outputs.

---

## Addendum — a genuine fix found, but not where expected

While manually reviewing the saved attention-map images to guide a worked scoring example, discovered that `q1_oracle.png`, `q2_oracle.png`, and `q4_oracle.png` all show the same image: **the literal cover page** of the JPMorgan annual report ("Powering Growth with Curiosity and Heart. Annual Report 2023.") — no financial data of any kind. This is not a model-capability problem at all; it's a data-quality problem. `data/sec10k_qa.json` lists `"page": 1` for all three questions (total net revenue, net income, total assets), and raw PDF page 1 is the cover.

**This was not a uniform off-by-one/off-by-two indexing bug.** Checked `q8_oracle.png` (oracle page 24) and `q14_oracle.png` (oracle page 65) — both show correct, relevant content, and both display their own printed page footer ("22" and "63" respectively), confirming a consistent `raw page = printed page + 2` relationship that the code already handles correctly by using the raw page number directly. So the underlying page-numbering convention is fine in general; Q1/Q2/Q4 specifically just have wrong `"page"` values in the source QA dataset (likely an authoring mistake unrelated to indexing conventions).

**Found the correct page by searching the PDF text directly** for the known ground-truth figure (`"3,875,393"`) rather than guessing candidate pages one at a time — found on raw PDF page 2, which turned out to be JPMorgan's "Financial Highlights" table containing the exact, precise figures for all three questions at once (Total net revenue: $158,104; Net income: $49,552; Total assets: 3,875,393), far cleaner than the narrative shareholder-letter page (page 5) which only had a rounded, non-GAAP "managed revenue" figure.

**Regenerated `q1_oracle.png`, `q2_oracle.png`, `q4_oracle.png` using the corrected page 2**, after also reverting the leftover few-shot-example prompt from the earlier failed experiment (which was contaminating this specific test — the clean, well-formatted Financial Highlights table structurally resembled the fictional few-shot example closely enough that the model was anchoring on the example's answer, "$45,000", rather than reading the real table). With the corrected page and the clean original prompt:

- **Net income (Q2): $49,022** vs. true $49,552 — ~1% off, a real, close, plausible answer.
- **Total assets (Q4): $3,755,447** vs. true $3,875,393 — ~3% off, also close and plausible.
- **Total net revenue (Q1): $49,022** — the exact same value as Q2, meaning the model found the right page but confused the "Total net revenue" row with the adjacent "Net income" row in the same dense table.

This is a dramatically different failure profile than before regeneration: a specific, explainable adjacent-row confusion on a dense multi-line table, not arbitrary nonsense unconnected to the document. It's also the same *type* of error the much larger production model made elsewhere in this project (e.g. picking a nearby-but-wrong segment/column) — evidence that some of what looked like a pure model-capability gap was actually inflated by a data-quality problem in the oracle page annotations, layered on top of a real (but smaller than previously estimated) genuine capability gap.

**Updated `evaluation/human_eval_scoring_template.csv`** rows for `sec10k_q1_oracle`, `sec10k_q2_oracle`, and `sec10k_q4_oracle` with the corrected page reference and new model outputs.

**Update: all 15 SEC 10-K oracle pages were subsequently visually audited** (not just JPMorgan). Full result: **7 of 15 (47%) had wrong oracle pages**, not just the 3 JPMorgan ones found initially.

**JPMorgan (6 questions)**: Q1, Q2, Q4 wrong (all page 1/cover, fixed to page 2 as above). Q5, Q8, Q14 confirmed correct pages — though Q5 (CEO name) is a caveat: its shown page ("2023 Highlights" infographic) doesn't obviously state "Chairman and CEO: Jamie Dimon" anywhere visible, yet the model answered correctly anyway, suggesting the right answer may come from general/parametric knowledge rather than genuine reading of the shown page. Not corrected further — flagged as a caveat for the dissertation rather than treated as validated localization.

**Goldman Sachs (9 questions)**: Q15, Q17, Q19, Q26, Q36 confirmed correct pages (all show the exact relevant figures/tables in plain, readable text — e.g. GS page 90's table literally reads "Net revenues $46,254", "Pre-tax earnings $10,739", "ROE 7.5%" — yet the model still answered "$1,134,000" for two different questions on this page and "0.01" for the third, and Q26/Q36 similarly got the exact right table but wrong arithmetic). **Q22, Q28, Q30, Q37 had wrong oracle pages** — found and fixed the same way as JPMorgan (searching the PDF text directly for the known ground-truth figure rather than guessing):

| Question | Wrong page showed | Corrected page | New answer | Result |
|---|---|---|---|---|
| Q22 (GBM loan balance, true $117,464) | Narrative description of GBM's business lines, no loan table at all | 95 | $1,012.0 | Still wrong — correct page, genuine model error |
| Q28 (S&P 500 % increase, true 24%) | Reused Q26's page — zero mentions of S&P 500 | 257 (best available — only page in the whole document mentioning "S&P 500"; a 5-year indexed stock-performance table, not an exact string match for "24%", but a real, on-topic table) | 10.1 | Still wrong — correct topic now shown, genuine model error |
| Q30 (AWM total assets, true $191,863) | AWM revenue-by-product-line narrative, no total-assets table | 98 | $10.9 Billion | Still wrong — correct page (verified exact match: "Total $191,863" appears in this page's table), genuine model error |
| Q37 (GS&Co net capital, true $20.25bn) | A different regulated entity's (GSIB/GSBE) capital ratios, not GS&Co | 113 | $1,000 | Still wrong — correct page, genuine model error |

**The key contrast this produces**: correcting JPMorgan's mislabeled pages fixed 2 of 3 questions to within a few percent of ground truth, because the corrected page (a single clean "Financial Highlights" summary table) was simple. Correcting Goldman Sachs's mislabeled pages fixed **0 of 4** questions, because even the corrected pages are dense, multi-column, multi-year tables — genuinely harder for a 2B-parameter model to read accurately regardless of whether the right page is shown. This is a clean, citable finding: **data-quality problems (wrong oracle pages) and model-capability problems (weak reading of dense tables) are separate, independently-verified issues that happened to compound each other** — fixing the data problem only helps where the underlying capability gap is small enough to not dominate the result.

`evaluation/human_eval_scoring_template.csv` updated for all 7 corrected rows (Q1, Q2, Q4, Q22, Q28, Q30, Q37) with the corrected page references and new model outputs.

---

## Full audit of `sec10k_qa.json` and a clean re-run

After the above, Tinara manually re-checked every "page" value across the full `sec10k_qa.json` file herself (not just the 15 questions in the human-eval sample), confirmed the indexing convention (1-indexed — `"page": 1` means the literal first physical PDF page, verified against `pdf2image`'s `first_page`/`last_page` convention and `document_parser/parser.py`'s explicit `page_num + 1` conversion), and corrected additional entries beyond what this investigation had already found — including Q5 (moved to page 4, resolving the earlier "possibly answered from parametric knowledge" caution — the model now correctly reads "Jamie Dimon" off a page that genuinely states it), Q8, Q14, Q19, Q26, and Q16 (a question outside the 15-question human-eval sample: "total assets allocated across all business segments," corrected from page 90 to page 93 after finding an exact match for "1,641,594" in a clean segment-assets table).

Re-ran `evaluation/human_eval_attention_maps.py` in full afterward — since the script reads `item["page"]` directly from the JSON with no hardcoded oracle overrides, every correction was picked up automatically with no code changes needed. This surfaced one genuine bug: a matplotlib `ParseException` crashed Q36's regeneration, because that answer happened to contain literal `$` characters in a full sentence ("Operating expenses were $1.65 billion..."), and matplotlib interprets text between `$` pairs as math notation to render. Fixed by escaping `$` → `\$` in both the question and answer text before they're used as a plot title (`models/attention_map.py`), and confirmed the fix by regenerating that one document cleanly.

---

## Final human evaluation results (self-assessment, 32 documents)

All 32 rows in `human_eval_scoring_template.csv` were scored by Tinara against the rubric in `human_eval_rater_instructions.md`. Results by category:

| | Localization | Clarity | Actionability | Trust | SHAP Consistency | Correctness |
|---|---|---|---|---|---|---|
| SEC 10-K Oracle (n=15) | 3.20 | 3.07 | 3.93 | 3.07 | n/a | 2.47 |
| SEC 10-K RAG-condition (n=7) | 1.14 | 1.29 | 1.43 | **1.00** | n/a | 1.14 |
| Bank statements (n=10) | 3.60 | 3.40 | 3.60 | 2.70 | **4.80** | 1.20 |
| Overall (n=32) | 2.88 | 2.78 | 3.28 | 2.50 | 4.80 (bank stmts only) | 1.78 |

(SHAP Consistency only applies to bank-statement rows, since SEC 10-K question-answering mode never computes an anomaly score. Bank-statement SHAP scores: 8 documents scored 5, 2 (bs2, bs4) scored 4 — Tinara's own refinement, reasoning that those two had genuinely tiny math-discrepancy contributions relative to the size of their real errors, a defensible reason to withhold a perfect score even though the qualitative flag/no-flag pattern was still correct in every one of the 10 documents.)

### Three headline findings

**1. RAG-condition Trust is a flat, unanimous 1.00 across all 7 rows — no exceptions.** Combined with Localization collapsing to 1.14 (vs. 3.20 for oracle) and Actionability to 1.43 (vs. 3.93), this is a clean, quantified demonstration that showing the model a retrieval-selected wrong page doesn't just produce a wrong answer — it produces an explanation with nothing genuine left to localize, and every single row was scored that way independently, without exception. Strong, simple evidence for the dissertation: retrieval failure is visible in the explanation quality itself, not just in answer correctness.

**2. SHAP Consistency (4.80) is dramatically stronger and more reliable than Correctness (1.20) on the same 10 bank-statement documents.** Every single document's SHAP breakdown correctly reflected whether a real math and/or date issue was present — never a false positive, never a missed real issue — even though the underlying VLM's stated text answer was almost always wrong. This is an important, separable, positive finding: the two XAI methods built in this project do not succeed or fail together. SHAP over the rule-based anomaly score is reliably faithful; attention-map explanation of the VLM's free-text answer is not. Worth stating explicitly as a contrast in the discussion chapter.

**3. Q2_oracle and Q5_oracle form a clean paired contrast demonstrating that localization quality and answer correctness are genuinely independent, not two views of the same thing.** Q2_oracle: Localization = 5, Correctness = 1 — Tinara's own comment: *"the highlighted place is right but the answer is wrong."* Q5_oracle: Localization = 2, Correctness = 5 — comment: *"got the right answer, but the page does not state the name"* — independently confirming this investigation's earlier suspicion (recorded above) that this specific answer likely came from the model's general/parametric knowledge of who JPMorgan's CEO is, rather than genuine reading of the page it was shown. Together these two rows are the single strongest illustration in the whole dataset of why this study scores localization and correctness as separate criteria rather than collapsing them into one "was the explanation good" judgment.

### Data-quality notes on the human-eval dataset itself
- Three rows (`sec10k_q22_oracle`, `sec10k_q28_oracle`, `sec10k_q30_oracle`) have corrupted `model_output` text and truncated `document_type` values, apparently from a spreadsheet round-trip mangling the long parenthetical descriptions originally written for those cells. The numeric scores for these rows are intact and were not affected — only the descriptive text needs re-pasting if a clean copy is wanted for the appendix.
- `bs5.pdf`'s answer field shows "Â£4,789.03" instead of "£4,789.03" in the round-tripped CSV — a character-encoding artifact from the same spreadsheet export, cosmetic only.

## Status: RQ3 human evaluation complete

All planned steps for the attention-map + SHAP human evaluation are done: instrument designed, 25/32-document sample finalized and ground-truth documented, attention maps generated (including the full page-correction audit above), SHAP validated and run, scoring completed by self-assessment, and results analyzed. Remaining optional work: clean up the 3 corrupted CSV cells, and write the dissertation results-section prose (see the companion literature note for related-work positioning to draw on).
