# Phase 13 — GPT-4o Baseline Comparison (DocVQA / FUNSD)

Continues from Phase 12 (research paper positioning). This covers item #2 of the
remaining-work priority list: an external commercial-model comparison point for
RQ2 (does multimodal visual encoding help, and how does the chosen backbone —
Qwen2-VL-max — compare to a leading commercial multimodal model on the same
benchmarks it was already evaluated on).

---

## Purpose

Qwen2-VL-max's DocVQA/FUNSD scores (0.952 / 0.792) were already compared against
LayoutLMv3-LARGE (0.834) in the dissertation, but had no reference point against
a leading *commercial multimodal* model. GPT-4o was chosen as that reference: not
as a candidate to replace Qwen2-VL-max in FinDocAgent, but as an external
calibration point — is 0.85-0.95 ANLS near the ceiling for this task, or would a
top commercial model score meaningfully higher on the exact same questions?

---

## Setup

- Only `DASHSCOPE_API_KEY` existed in `~/.env`. Tinara created an OpenAI API key
  (platform.openai.com → API keys, "All" permissions, default project) and added
  billing (the account had no prepaid credit initially — first test call failed
  with `insufficient_quota` until billing was funded).
- Key was verified in two steps before any real spend: (1) format/prefix check
  without ever printing the full value, (2) a single trivial `gpt-4o` chat
  completion call.
- **Environment note, worth remembering**: `openai`, `datasets`, `torch`, etc.
  are only installed in the base conda environment (`/opt/conda/bin/python`),
  not in either project venv (`venv/` or `.venv/`). All GPT-4o eval commands
  must use `/opt/conda/bin/python -m evaluation.<script>`, not the venv python.
- Tinara ran the actual (cost-incurring) full evaluation commands herself in her
  own terminal, rather than via the agent — a general preference to keep direct
  control over anything that spends money.

---

## Scripts written

- `evaluation/docvqa_gpt4o_eval.py` — mirrors `evaluation/docvqa_eval.py`
  exactly: same 50-sample DocVQA slice (`nielsr/docvqa_1200_examples`), same
  prompt ("Answer with the exact text from the document. Be concise, one word
  or short phrase only."), same ANLS scoring — only the model call is swapped
  from the DashScope-hosted `qwen-vl-max` to `gpt-4o` via a plain OpenAI client
  (no DashScope `base_url` override).
- `evaluation/funsd_gpt4o_eval.py` — mirrors `evaluation/funsd_eval.py`
  exactly: same FUNSD test-split iteration, same "list all filled-in field
  values, one per line" prompt, same per-field best-match ANLS scoring.
- Both scripts were dry-run tested on 2-3 samples first (near-zero cost) to
  confirm correctness before running the full paid batch — this caught nothing
  wrong, but was cheap insurance given the earlier billing issue.

---

## Full run results (raw ANLS — same metric used for every other comparison in this dissertation)

| Benchmark | Qwen2-VL-max | GPT-4o | Difference |
|---|---|---|---|
| DocVQA (n=50) | 0.952 | **0.803** | -0.149 |
| FUNSD (n=809 GT fields across 47 docs) | 0.792 | **0.727** | -0.065 |

**These raw numbers are the official, reported RQ2 external-validity result.**
Qwen2-VL-max outperforms GPT-4o on both benchmarks, using the identical raw
ANLS methodology already applied to every other comparison in the dissertation
(regex baseline, LayoutLMv3-LARGE, single-pass vs. RAG agent, etc.) — so this
stays consistent with how every other number in the results chapter was
produced, rather than introducing a different metric just for this comparison.

---

## Supplementary error analysis (methodological transparency, not the headline number)

While inspecting GPT-4o's worst-scoring DocVQA rows, a pattern emerged: most
low scores weren't wrong answers, they were *correctly-read but more verbose*
answers that ANLS's strict edit-distance metric penalizes heavily — e.g. "Five"
vs ground truth "5", "Two focus groups" vs "Two", "Richmond, Virginia" vs
"Richmond". GPT-4o followed the "concise, exact text only" instruction less
strictly than Qwen2-VL-max.

Built `evaluation/gpt4o_error_analysis.py` to classify every low-scoring answer
as either "verbose-but-correct" (via substring containment + number-word
matching, e.g. digit "5" vs word "five") or a genuine misread — purely a local
JSON reprocessing script, no API calls, no additional cost.

**Two false-positive bugs were found and fixed while building this classifier**
(both real methodological pitfalls, worth remembering for any future
containment-based text matching):
1. A lone single-character prediction (e.g. `"3"`) trivially "contained" almost
   any long, unrelated ground-truth paragraph — fixed with a minimum
   containment length guard (4 characters).
2. The number-word `"one"` matched as an unbounded substring inside unrelated
   words like `"telephone"` (`tele-phone` → `...one` at the end), producing a
   false match between a phone-number prediction and a completely unrelated
   confidentiality-boilerplate ground truth — fixed by switching to
   regex `\bword\b` / `\bdigit\b` word-boundary matching instead of raw
   substring checks.

After both fixes, on manual + automated review:

| Benchmark | Already ≥0.5 ANLS | Verbose-but-correct (reclassified) | Genuine misread | Adjusted ANLS |
|---|---|---|---|---|
| DocVQA | 42/50 | 4 | 4 (8%) | 0.803 → **0.883** |
| FUNSD | 665/809 fields | 34 | 110 (~14%) | 0.727 → **0.782** |

Even after this generous reclassification, GPT-4o's adjusted score still trails
Qwen2-VL-max (-0.069 DocVQA, -0.010 FUNSD) — so the raw-ANLS conclusion
(Qwen2-VL-max wins) holds either way. What the adjustment shows is that part of
the raw gap is an output-format/verbosity difference (GPT-4o answering more
conversationally despite instructions) rather than a pure document-comprehension
gap. This mirrors the RAG failure-taxonomy framing already used elsewhere in
the dissertation (separating retrieval failure from generation failure): here
it separates "output-format non-compliance" from "genuine misread."

One further nuance found in FUNSD specifically: some "verbose-but-correct"
matches were actually GPT-4o merging/splitting form fields differently than
FUNSD's ground-truth B-ANSWER/I-ANSWER boundaries (e.g. correctly reading a
full US-states list as one answer where the ground truth split it across two
annotated fields) — a field-segmentation difference, not a reading error.

**Caveat for citing the adjusted numbers**: the reclassification is an
automated heuristic, not a full manual re-grade of every row. It's been
spot-checked and two real bugs were caught and fixed during development, but a
independent manual sample check is recommended before citing the exact adjusted
figures (0.883 / 0.782) as validated numbers, versus citing them as an
"estimated adjustment" in the Discussion section.

---

## Decision

- **Official/reported number**: raw ANLS (0.803 DocVQA, 0.727 FUNSD), consistent
  with the metric used throughout the rest of the dissertation.
- **Discussion/limitations material**: the adjusted-ANLS error analysis above,
  useful for a nuanced paragraph on *why* the gap exists (verbosity/instruction-
  following, not raw comprehension) without changing the headline result.

---

## Artifacts

- `evaluation/docvqa_gpt4o_eval.py`
- `evaluation/funsd_gpt4o_eval.py`
- `evaluation/gpt4o_error_analysis.py`
- `evaluation/results/docvqa_gpt4o_results.json`
- `evaluation/results/funsd_gpt4o_results.json`
- `evaluation/results/gpt4o_error_analysis.json`

## Status: RQ2 GPT-4o baseline comparison complete.