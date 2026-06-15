# Phase 2 Findings — Document Parser, ChromaDB, Regex Baseline

## Date: [today's date]

## What was built
- pdfplumber + pdf2image parser (document_parser/parser.py)
- ChromaDB vector store with semantic search (vector_store/chroma.py)
- Real Retriever node connected to ChromaDB
- Regex-based "naive baseline" Extractor (agent/nodes/extractor.py)

## Test document
- bank_statement_clean_word.pdf (SunTrust statement, converted PDF -> Word -> PDF)

## Key finding: regex baseline fails on table-structured documents

The regex extractor correctly found 18 numeric amounts in the document text.
However, it failed to correctly identify the "stated total" field.

### Why it failed
The PDF-to-text extraction process flattens the 2D table layout into 1D text,
breaking the spatial relationship between labels and values. 

Specifically:
- The text contains "Total Amount Due: /dd/yyyy" followed shortly after by
  "SUMMARY 390,000.00" (the Credit Limit value)
- A regex searching for a number near the word "Total" matches £390,000.00
  (Credit Limit) instead of the correct £3,898.57 (Total Amount Due)
- This happens because, in the linearised text, "Total" appears textually
  closer to the Credit Limit value than to the actual Total Amount Due value,
  even though on the visual page they are in completely different table cells

### Raw text snippet demonstrating the issue:

Total Amount Due: /dd/yyyy
SUMMARY 390,000.00
3,898.57

## Why this matters for FinDocAgent
This is direct empirical evidence for the core motivation of the project:
pure text-based extraction loses spatial/layout information that is critical
for correctly interpreting financial documents. This motivates the use of:
- LayoutLMv3-style 2D position embeddings (Related Work)
- Qwen2-VL's visual understanding of table structure (Phase 3)

## Anomaly Checker behaviour
With the incorrect stated_total (£390,000.00), the Anomaly Checker correctly
flagged a math_discrepancy (as designed) — confirming the rule-based anomaly
detection logic itself works correctly when given (even incorrect) numeric
inputs. The error originates entirely in the extraction step, not the
anomaly detection step.

## Conclusion for dissertation
This forms the "naive baseline" comparison point: regex/rule-based extraction
on linearised PDF text. Expected to be outperformed significantly by
Qwen2-VL-based extraction (Phase 3), which can use visual layout to correctly
associate labels with values.