# Phase 8 — SEC 10-K RAG Optimization (root cause diagnosis + hybrid retrieval)

Continues directly from Phase 7 (`phase7_sec10k_rag_evaluation.md`), which ended at 0.436 ANLS with the ReAct loop in place and two known failure clusters unresolved (GS aggregate summary questions, JPM CSR narrative questions).

## Score progression

| Run | ANLS | What changed |
|---|---|---|
| Phase 7 baseline (ReAct loop, top_k=7) | 0.436 | starting point |
| + page chunking + BGE query instruction prefix | **0.629** | biggest single jump — fixed a real bug |
| + extractor disambiguation prompt (identify row/segment/year) | 0.609 | regression — confident wrong answers bypassed the ReAct retry |
| + "decline if unsure" prompt fix | 0.617 | flat net score, but retry rate rose sharply (cost went up) |
| + top_k 10→6, DPI 200→150 (cost control) | 0.617 | no accuracy change, meant to cut cost |
| + hybrid BM25 + dense retrieval (Reciprocal Rank Fusion) | **0.658** | new best |

Single-pass oracle upper bound (correct page handed to the model directly): 0.836. Remaining gap ≈ 0.18, and its cause is now understood (see below) rather than a mystery.

---

## Root cause found: whole-page embeddings were silently truncated

`vector_store/chroma.py`'s `index_document` embedded each PDF page's entire extracted text (`plain_text + table_text`) as a single Chroma document. The embedding model, `BAAI/bge-base-en-v1.5`, has a hard 512-token truncation limit. Dense financial-report pages routinely exceeded this — meaning content past token ~512 was never part of the embedding at all. No amount of query rewording, top_k increase, or ReAct retrying could ever surface a page whose relevant figure sat past that cutoff, because the embedding itself was blind to it.

**Fix:** `chunk_text()` splits each page's text into ~250-word overlapping chunks before embedding, each tagged with `page_number` metadata. Retrieval still dedups back to unique pages, so nothing downstream had to change. This one fix took the score from 0.436 → a large share of the eventual 0.629.

**Companion fix, same run:** BGE models are trained with an instruction prefix on the *query* side only (`"Represent this sentence for searching relevant passages: "`), not on documents — this is how the model learns asymmetric query-vs-passage matching. `search_document()` was sending raw, unprefixed queries. Added the prefix at query time only (`chroma.py`, `search_document`).

---

## Extractor prompt work: a real problem, an imperfect fix

Investigating why some *correctly retrieved* pages still gave wrong answers surfaced a second issue: on multi-segment, multi-year tables (Goldman's GBM / AWM / Platform Solutions columns, 2022 vs 2023), Qwen2-VL was grabbing the wrong row/column without any signal that it should double-check. Added explicit disambiguation instructions to the extractor prompt (identify row, segment, and year before answering; final answer behind an `ANSWER:` marker).

This fixed some wrong-column reads (e.g. GBM net revenues, AWM total assets) but introduced **new** regressions on previously-correct questions — the model started giving confident, fluent, wrong answers instead of declining, which meant `should_continue`'s negative-phrase check (`graph.py`) never triggered a retry for those cases. Added a follow-up instruction ("if not confident, answer 'Not found'") to restore the safety net — this recovered correctness but caused far more questions to trigger all 3 ReAct passes (since "Not found" is the literal retry trigger), which raised cost noticeably without a matching score gain. Net lesson: this prompt-tuning path had real but small, unstable effects and mostly traded failure modes rather than eliminating them.

**Cost observation:** the same top_k=10/dpi=200 configuration that scored 0.629 cost ~$1.50 per 39-question run; after the decline-if-unsure change (even with top_k cut to 6 and DPI cut to 150), cost rose to ~$2.00–2.35. Root cause: cost is dominated by *retry count*, not per-call image size — a question that reliably declines through all 3 ReAct passes pays for 3 full image batches for zero score, and the decline-fix made that the default behavior for every hard question.

---

## Diagnostic method used to find the real bottleneck (free, no API cost)

`data/sec10k_qa.json` includes an oracle `"page"` field per question (used for the single-pass baseline). Used this, entirely offline with pdfplumber, to:
1. Check whether the ground-truth answer string actually appears in the extracted text of that page.
2. If not, search the whole document to find where it actually lives.
3. Cross-reference against the retriever's logged page numbers to see whether the *true* content page was ever a retrieval candidate at all.

This separated what had been a single vague "retrieval failures" bucket into two distinct, differently-fixable problems:

- **Retrieval-ranking misses** (10 of 14 stuck questions): the answer text is genuinely present and extractable somewhere in the document, but ChromaDB never ranks the right page into the top-k, across any of the 3 query variants tried. This is where BM25 helped.
- **Genuine VLM misreads** (e.g. Q39, GS cash & equivalents): the correct page (125) was retrieved on every single run tried, yet Qwen2-VL kept reading the wrong number (`$42`, `$7.93 billion` instead of `$241.58 billion`) until the BM25 run, where it read correctly — likely incidental from a different page ordering among the 6 images shown, not a targeted fix.

Side finding: a few of the oracle `"page"` labels in `sec10k_qa.json` are off by ~1 page from where pdfplumber actually finds the content (e.g. Goldman's GBM segment loan table is labeled page 94 in the QA data but the actual table with figures is on page 95). This doesn't affect the live agent — it never reads that field — but mattered for getting the diagnosis right.

---

## Fix: Hybrid BM25 + dense retrieval

**What it is, plainly:** BM25 is keyword search — it scores a page by how many exact query words it contains, weighting rare words far more than common ones. It doesn't understand meaning, only literal term overlap. Dense embeddings (the existing Chroma+BGE search) do the opposite — they capture semantic/topical similarity, which is great for conceptual matches but means a page with a genuinely rare, distinctive term can get buried among many other pages that are merely "similar in topic."

**Combination method:** Reciprocal Rank Fusion (RRF). Since BM25 scores and cosine-similarity scores live on incompatible scales, RRF only looks at each method's *rank position*: `score(page) = Σ 1/(60 + rank_in_that_method)` across both methods, then sort descending. A page ranked highly by either method (or both) rises to the top; no score normalization needed.

**Implementation:**
- `chroma.py`: added `bm25_search()` (builds a `BM25Okapi` index from the same chunks already stored in Chroma via `collection.get()` — no re-embedding, no extra API cost) and `reciprocal_rank_fusion()`.
- `retriever.py`: runs both `search_document()` (dense) and `bm25_search()` (keyword) per pass, fuses the two ranked page lists, takes the top 6 forward to image conversion.

**Result:** 0.617 → 0.658.

---

## What BM25 fixed vs. didn't

**Fixed — JPM CSR narrative cluster:** Q8 (EOCF program markets), Q9 (EOCF loans deployed), Q13 (homebuyer grants) — all scored 0.0 ("Not found") in *every* prior run regardless of chunking, query prefix, top_k, or prompt changes. These hinge on rare, distinctive terms ("EOCF," "Homebuyer Grant") that appear on only 1-2 pages in a 360-page document — exactly BM25's strength. All three now score 1.0, correct on pass 1.

**Not fixed — GS segment-footnote cluster:** Q15, 16, 17, 22, 23, 24, 25, 28 (GBM loan balance, corporate loan balance, real estate loan portfolio, allowance for credit losses, firmwide/segment pre-tax earnings, S&P 500 %). Diagnostic confirmed the text is extractable and present in the document (e.g. GS page 95 has the real GBM loan table), but BM25 still can't find it, because the query terms — "loan balance," "allowance for credit losses," "pre-tax earnings" — are *common* banking-report vocabulary repeated dozens of times throughout a 10-K's risk sections and MD&A. BM25's core assumption (rare term ⇒ relevant page) simply doesn't hold for these queries the way it did for "EOCF."

**New regressions introduced:** Q2 (net income), Q18 (GBM revenues), Q27 (investment banking fees) — all previously single-pass correct, now wrong. Changing the candidate page ordering via rank fusion occasionally pulls in a distracting page for questions that didn't need retrieval help at all. Net effect across the 39 questions is still positive (+0.041 aggregate), but it's a real tradeoff, not a pure win — worth remembering before assuming a retrieval change is strictly additive.

---

## Possible future work (not attempted this phase)

- **Table-aware chunking**: current chunking splits by fixed word count, which may separate a table's row label from its value — could explain why BM25 still can't discriminate the GS segment-footnote tables even where whole-page text extraction succeeds.
- **Cross-encoder reranker** over the combined BM25+dense candidate pool, instead of (or in addition to) RRF.
- **Section-aware retrieval bias**: detect "segment"-scoped questions (GBM/AWM/Platform Solutions) and bias retrieval toward MD&A/segment-note sections specifically.

## Bottom line

SEC 10-K RAG agent ANLS: **0.436 → 0.658** across this phase (39 questions, JPMorgan + Goldman Sachs 2023 annual reports), against a 0.836 single-pass oracle ceiling. The improvement path is fully explained end-to-end: a genuine truncation bug (fixed), a rare-term retrieval gap (fixed via BM25), and a remaining common-vocabulary retrieval ambiguity in segment-footnote tables (diagnosed, not yet fixed) — a defensible, well-evidenced result for the dissertation writeup rather than an unexplained plateau.
