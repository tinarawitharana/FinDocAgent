# Phase 12 — Research Paper Positioning

Papers found while looking for literature to position FinDocAgent's results against, organized by the theme they support. For each paper: what it says, and specifically why it's relevant to a finding in this project — not just "related work," but the actual connection.

Searches run (all via WebSearch, today):
1. `attention weights faithfulness explanation NLP "attention is not explanation"`
2. `SHAP explainability financial anomaly detection fraud rule-based`
3. `small vision language models dense financial table document understanding benchmark limitations`
4. `retrieval augmented generation RAG failure explainability diagnosing retrieval errors`
5. `FinanceBench LLM RAG accuracy SEC 10-K filings question answering benchmark`
6. `hybrid BM25 dense retrieval RAG financial documents performance improvement study`
7. `retrieval augmented generation financial document question answering ANLS accuracy benchmark 2024 2025`

---

## 1. Attention maps as explanations — the faithfulness debate

This is the theoretical grounding for RQ3's attention-map work, and for why the phase11 investigation treats "the model highlighted the right region" and "the model gave the right answer" as two genuinely separate things to evaluate rather than one.

**Jain & Wallace, "Attention is not Explanation" (2019)** — [arXiv:1902.10186](https://arxiv.org/abs/1902.10186), also [ACL Anthology N19-1357](https://aclanthology.org/N19-1357/)
Core claim: across a range of NLP tasks, attention weights are frequently uncorrelated with gradient-based feature-importance measures, and very different attention distributions can produce the same prediction — i.e., attention is not a reliable indicator of *why* a model answered the way it did.
**Directly relevant**: this is the paper that predicts exactly the pattern found in this project's human eval — Q2_oracle (Localization=5, Correctness=1) and Q5_oracle (Localization=2, Correctness=5) are a live, first-hand demonstration of Jain & Wallace's point. The attention map can point at a plausible region while the answer is wrong, or point weakly at the wrong region while the answer is (probably parametrically) right. This paper is the citation for *why* that's not a contradiction — it's the expected behavior of attention as an explanation mechanism, not a bug in this evaluation.

**Wiegreffe & Pinter — rebuttal to Jain & Wallace**
Core claim: attention weights can still be *an* explanation, just not *the* explanation — existence doesn't entail exclusivity. Their argument is that attention is one legitimate signal among several, not that it's meaningless.
**Directly relevant**: gives a defensible middle-ground framing for the dissertation's discussion — rather than concluding "attention maps are useless," the more accurate and citable conclusion (supported by this project's own data) is "attention maps are a partial, non-exclusive explanation signal that must be evaluated separately from answer correctness," which is exactly the rubric this project already used (Localization and Correctness scored independently).

---

## 2. SHAP for financial fraud / anomaly detection — validating the SHAP-over-rule-based-score approach

Relevant to why SHAP was chosen at all for the bank-statement anomaly checker, and to interpreting why SHAP Consistency (4.80) so dramatically outperformed Correctness (1.20) in the human eval.

**"Shapley Value-Guided Adaptive Ensemble Learning for Explainable Financial Fraud Detection with U.S. Regulatory Compliance Validation"** — [arXiv:2604.14231](https://arxiv.org/pdf/2604.14231)
Uses SHAP explicitly to satisfy regulatory explainability requirements in a financial fraud context.
**Relevant**: supports the framing that SHAP isn't just a convenient tool but is the standard XAI method the financial-compliance literature reaches for — strengthens the justification for choosing it over, say, ad-hoc rule explanations.

**"Explainable AI (XAI) Using SHAP and LIME for Financial Fraud Detection and Credit Scoring"** — ResearchGate, and **"SHAP-Driven Interpretability in Financial Fraud Detection: A Multimodal Data Approach"** — SCIRP
Both apply SHAP/LIME to explain black-box fraud classifiers.
**Relevant**: the recurring theme across this literature is applying SHAP to a *learned* black-box model's fraud score. This project's SHAP application is unusual and worth explicitly contrasting: it's applied to a **hand-written, closed-form linear composite score** (`0.5*math_feature + 0.5*date_feature`), not a trained classifier — which is precisely why it could be validated exactly against closed-form expected values (as this project did) rather than only qualitatively. That validation step is stronger than what most of this literature does, since most SHAP-on-fraud papers can't ground-truth their explanations this cleanly. Worth stating as a methodological strength in the dissertation.

**"Model interpretability of financial fraud detection by group SHAP"** — ScienceDirect
**Relevant**: background citation for SHAP's general acceptance in the financial-anomaly-detection space; supports the "why SHAP" paragraph in the methodology chapter.

**"Methodological challenges in explainable AI for fraud detection: a systematic literature review"** — Springer, *Artificial Intelligence Review*
Surveys interpretability challenges across the fraud-detection XAI literature.
**Relevant**: useful for citing the general difficulty of validating XAI faithfulness in this domain — which makes this project's exact closed-form validation of SHAP (a small, controlled, provably-correct case) a notable point of methodological rigor to highlight, since it's something the wider literature review says is generally hard to do.

---

## 3. Small vision-language models on dense financial tables — supports the "genuine model-capability limitation" conclusion

This is the most directly load-bearing set of papers for the phase11 conclusion that attention-map answer quality on dense GS/JPMorgan tables reflects a real, literature-documented capability gap, not a bug specific to this project's setup.

**FinChart-Bench: Benchmarking Financial Chart Comprehension in Vision-Language Models** — [arXiv:2507.14823](https://arxiv.org/html/2507.14823)
Benchmarks VLMs specifically on financial charts/visual content.
**Relevant**: independent confirmation that financial visual documents are a recognized hard case for VLMs generally, not just for the 2B model used here.

**"When Tables Go Crazy: Evaluating Multimodal Models on French Financial Documents"** — [arXiv:2602.10384](https://arxiv.org/pdf/2602.10384)
Evaluates multimodal models specifically on dense financial tables.
**Relevant**: title and subject are a near-exact match for this project's own finding — Goldman Sachs's dense, multi-column, multi-year tables (Q22/Q26/Q28/Q30/Q37) being the ones where even a corrected oracle page produced 0/4 correct answers, versus JPMorgan's single clean summary table where correcting the page fixed 2/3. This is the strongest available citation for "table density, not just correct-page retrieval, is an independent driver of VLM failure" — exactly this project's central Addendum finding.

**MMDocBench: Benchmarking Large Vision-Language Models for Fine-Grained Visual Document Understanding** — [arXiv:2410.21311](https://arxiv.org/pdf/2410.21311)
General fine-grained visual document understanding benchmark for VLMs.
**Relevant**: broader supporting citation for VLM struggles with fine-grained reading tasks (e.g., reading small numeric text out of a dense table cell), which is the specific failure mode seen throughout the SEC 10-K oracle answers (single-digit-off or unit-confused numeric answers).

**Search-summary finding (from the "small vision language models dense financial table" search results themselves, not a single paper)**: VLMs show "significant performance deterioration on larger tables with smaller and denser textual elements, stemming from limitations in their OCR capabilities" — and the search results specifically cite an observed model predicting "3,468" instead of "3,466," i.e., a near-miss numeric OCR error.
**Directly relevant**: this is almost exactly the error pattern in this project's own SEC 10-K oracle answers (e.g. wrong-but-close numeric answers on Goldman Sachs's dense tables) — this citation lets the dissertation say the near-miss OCR-style numeric error pattern is a documented, general VLM limitation on dense financial tables, not an artifact specific to Qwen2-VL-2B.

---

## 4. RAG failure taxonomy — frames the RAG-condition Trust=1.00 finding

**"Classifying and Addressing the Diversity of Errors in Retrieval-Augmented Generation Systems"** — [arXiv:2510.13975](https://arxiv.org/html/2510.13975v1)
Proposes a three-stage failure taxonomy: **retrieval failure** (no retrieved chunk comes from the gold context), **packing failure** (answer-bearing evidence not present in the final packed context), and **generation failure** (evidence is available but the generated answer still doesn't match gold).
**Directly relevant — this is the single most useful paper found today.** It gives a precise vocabulary for exactly what the RAG-condition human-eval rows (Trust flatlined at 1.00, Localization 1.14, Actionability 1.43, all far below the oracle condition) actually demonstrate: this project's RAG failures are cleanly diagnosable as **retrieval failures** (the wrong page/chunk was retrieved in the first place — e.g. Q28's RAG-retrieved page 265 vs. the corrected oracle page 96), not generation failures (the model reading a *correct* chunk badly). This taxonomy lets the dissertation state precisely, using established vocabulary, which stage of the RAG pipeline this project's evaluation shows is breaking, rather than just saying "RAG performed worse."

**"A Systems-Level Analysis of Sensitivity, Robustness, and Stability in Retrieval-Augmented Generation"** — [arXiv:2606.28337](https://arxiv.org/pdf/2606.28337)
**Relevant**: general supporting citation for RAG systems being sensitive to what's actually retrieved — background for why retrieval quality (not just generation quality) needs to be evaluated as its own axis, which is exactly what the oracle-vs-RAG condition split in this project's human eval was designed to isolate.

---

## 5. FinanceBench — the direct external benchmark comparison

**FinanceBench (Stanford + Patronus AI)** — referenced via [emergentmind summary](https://www.emergentmind.com/topics/financebench) and multiple citing papers, e.g. [arXiv:2404.07221](https://arxiv.org/pdf/2404.07221)
A benchmark of real-world financial questions paired with real SEC filings (150 curated QA pairs in the original release; a larger 10,231-question corpus in extended versions), designed specifically to test LLM/RAG QA over financial filings — the same document type (SEC 10-K) this project's RAG pipeline was built and evaluated on.
**Reported numbers**: GPT-4 + OpenAI ada-002 embeddings answers only **19%** of questions correctly; a multi-agent RAG system ("LiveAI for SEC Filings") reaches **56%**; more recent frontier-model evaluations report up to ~88% on financial reasoning tasks generally (not FinanceBench specifically).
**Directly relevant — the key comparison point for the dissertation's results chapter**: this project's own SEC10-K RAG pipeline reached **0.658 ANLS** (up from a 0.436 baseline) after adding chunking + BM25 hybrid retrieval. FinanceBench's headline number of 19% for a naive embeddings-only RAG baseline, versus 56% for a purpose-built multi-agent system, gives a directly comparable published range to sit this project's result against — the 65.8% figure compares favorably to the naive baseline and is in the same range as the stronger multi-agent system, even though FinanceBench uses exact/human-graded accuracy rather than ANLS. **Important caveat to state explicitly in the dissertation**: ANLS and human-graded accuracy are not the same metric — ANLS gives partial credit for near-matches (useful for the many numeric/OCR-adjacent answers this project produces), so the comparison should be framed as "broadly comparable, favorable range" rather than a like-for-like percentage comparison.

---

## 6. Hybrid BM25 + dense retrieval via RRF — validates the RAG architecture choice itself

**"From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents"** (referred to in search results as related to "T2-RAGBench") — [arXiv:2604.01733](https://arxiv.org/pdf/2604.01733)
Tests ten retrieval strategies across a large financial text-and-table corpus (23,088 queries, 7,318 documents).
**Reported findings, directly quoted from the search results**: "Combining BM25 and dense retrieval via Reciprocal Rank Fusion improves over both constituent methods across all metrics and all dataset subsets, with the largest improvement on TAT-DQA (+8.1pp Recall@5 over BM25)" and, notably, "**BM25 outperforms state-of-the-art dense retrieval on financial documents**, which challenges the common assumption that dense retrieval is always superior."
**Directly relevant — the strongest architecture-validation citation found today.** This is independent, large-scale published confirmation of the exact design decision this project made: adding BM25 alongside dense retrieval via RRF, rather than relying on dense embeddings alone, is a validated improvement specifically on financial documents — and the finding that BM25 alone can beat dense retrieval on this domain (likely because financial documents are full of exact numeric tokens, ticker symbols, and section headers that dense embeddings don't represent precisely) is a good explanatory citation for *why* this project's own jump from 0.436 to 0.658 ANLS happened when BM25 was added — it's not a coincidental tuning win, it's consistent with a documented property of financial-document retrieval.

---

## How these map onto the dissertation's RQ structure

- **RQ3 (attention maps)** → Section 1 (Jain & Wallace / Wiegreffe & Pinter) for the theoretical framing, Section 3 (FinChart-Bench / "When Tables Go Crazy" / MMDocBench) for the empirical "this is a known, general VLM limitation" grounding.
- **RQ3 (SHAP)** → Section 2, plus the explicit methodological-strength point that this project's closed-form validation is stronger than most cited SHAP-on-fraud work.
- **RQ1/RQ2 (RAG results)** → Section 5 (FinanceBench) for the external accuracy comparison, Section 6 (hybrid BM25+RRF) for validating the architecture decision, Section 4 (RAG failure taxonomy) for precisely characterizing what the RAG-condition human-eval drop (Trust=1.00) actually is.