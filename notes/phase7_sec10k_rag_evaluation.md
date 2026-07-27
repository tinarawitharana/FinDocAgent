# Phase 7 — SEC 10-K RAG Evaluation & ReAct Loop

## Dataset
- 39 questions (JPMorgan Chase + Goldman Sachs 2023 annual reports)
- Removed 11 unanswerable questions: 5 macro stats (global GDP, US debt market, etc.), 4 bar chart % growth questions, 2 chart-based questions
- Evaluation metric: ANLS (Answer Normalised Levenshtein Similarity), threshold τ=0.5

---

## Architecture (hybrid table-aware RAG)
- **Parser**: pdfplumber `extract_text()` + `extract_tables()` → combined into `full_text` per page
- **Index**: ChromaDB with BAAI/bge-base-en-v1.5 embeddings (upgraded from all-MiniLM-L6-v2)
- **Retrieval**: top-k pages retrieved by semantic search → converted to PNG images via pdf2image (200 DPI)
- **Extraction**: Qwen2-VL (qwen-vl-max via DashScope) reads all retrieved page images visually

---

## RQ1 Evaluation — 3-Condition Comparison

| Condition | ANLS | Notes |
|---|---|---|
| Single-pass oracle (page number given) | 0.836 | upper bound — page handed directly to model |
| Single-pass RAG agent (no loop) | 0.389 | agent must find page itself via ChromaDB |
| ReAct loop agent (up to 3 passes) | 0.436 | enriched retry query on failure |

---

## Progression of scores

| Run | ANLS | Changes made |
|---|---|---|
| Text RAG, 50 Qs | 0.308 | original text-only retrieval |
| Hybrid RAG, 39 Qs | 0.389 | pdfplumber table extraction + image retrieval, BAAI embeddings, macro Qs removed |
| ReAct loop, 39 Qs | 0.436 | loop back on negative answers, enriched query on retry |

---

## ReAct Loop Implementation

**graph.py** — `should_continue` detects negative answer phrases:
- Phrases: "not found", "not available", "not specified", "not provided", "does not contain", "does not provide", "insufficient", "not disclosed", "cannot find"
- If negative AND `iteration_count < 3` → route back to retriever
- Otherwise → END

**extractor.py** — removed `state["task_complete"] = True` from QA branch so graph controls exit

**retriever.py** — 3-pass query strategy:
- Pass 1 (`iteration_count=0`): plain question
- Pass 2 (`iteration_count=1`): `"financial statement table annual report: {question}"`
- Pass 3 (`iteration_count>=2`): `"annual report corporate initiatives text: {question}"`

---

## Known Failures & Root Causes

**Goldman Sachs aggregate questions (page 90)** — all scoring 0.0:
- Total net revenues, total assets, total operating expenses, pre-tax net earnings
- ChromaDB consistently fails to retrieve page 90
- Page 90 is a dense segment summary table — embeddings don't match the question text well

**JPMorgan narrative questions (pages 24-25)** — all scoring 0.0:
- EOCF program (10 markets, 2,900 loans), affordable housing (190,000 units), homebuyer grants (8,600)
- Answers are buried in CSR paragraph text, no financial keywords for retrieval to latch onto

**Goldman Sachs segment detail questions (pages 94-95)** — mostly 0.0:
- Corporate loan balance, real estate loan portfolio, % change in revenues
- Wrong pages retrieved, model reads wrong numbers from the table

---

## Next Steps Being Tried
- Increase top_k from 7 to 10 (more pages retrieved per pass)
- Smarter query construction based on pass number (table-focused vs narrative-focused)
- Potential future: BM25 hybrid retrieval (keyword + semantic) to better find exact financial figures
