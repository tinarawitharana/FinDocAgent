"""Shared naive text-extraction helpers for the DocVQA/FUNSD regex baselines.

These implement the same "no layout, no vision, just pattern-matching over raw OCR
words" approach as evaluation/baseline.py's bank-statement regex extractor, applied to
DocVQA/FUNSD's own `words` fields (both datasets ship OCR word lists alongside the
image, so no separate OCR step is needed). The point isn't to be a strong baseline —
it's the text-only, non-VLM reference point for RQ2 (does multimodal visual encoding
beat naive text extraction).
"""

import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "when", "where", "who",
    "which", "how", "this", "that", "these", "those", "in", "on", "at", "to", "of",
    "for", "and", "or", "do", "does", "did", "mentioned", "document", "much", "many",
    "letter", "form", "it", "its", "with", "as", "be", "by",
}

DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,?\s*\d{4})?\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\$?\d[\d,]*\.?\d*%?")


def keywords(question):
    """Lowercased, stopword-stripped tokens from a question, used for keyword-overlap matching."""
    tokens = re.findall(r"[a-zA-Z0-9]+", question.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _wants_date(question):
    q = question.lower()
    return any(w in q for w in ("date", "when", "day", "year"))


def _wants_number(question):
    q = question.lower()
    return any(w in q for w in ("how much", "how many", "number", "amount", "total", "percent", "page"))


def extract_answer(words, question, window=3):
    """DocVQA-style baseline: finds the OCR word with the highest keyword-overlap in its
    local context, then returns a short word span there (nudged toward a date/number regex
    match nearby if the question implies that answer type). No layout or ranking beyond
    keyword overlap — the naive, non-VLM reference point.
    """
    if not words:
        return ""

    kw = set(keywords(question))
    if not kw:
        return " ".join(words[:window])

    best_idx, best_score = 0, -1
    for i in range(len(words)):
        lo, hi = max(0, i - 5), min(len(words), i + 6)
        context = " ".join(words[lo:hi]).lower()
        score = sum(1 for k in kw if k in context)
        if score > best_score:
            best_score = score
            best_idx = i

    lo = max(0, best_idx - window // 2)
    hi = min(len(words), lo + window)
    nearby = " ".join(words[max(0, lo - 3):min(len(words), hi + 3)])

    if _wants_date(question):
        m = DATE_PATTERN.search(nearby)
        if m:
            return m.group(0)
    if _wants_number(question):
        m = NUMBER_PATTERN.search(nearby)
        if m:
            return m.group(0)

    return " ".join(words[lo:hi])


def extract_gt_answers(words, ner_tags):
    """Reconstructs ground-truth answer spans from FUNSD's B-ANSWER/I-ANSWER tags.

    Duplicated from evaluation/funsd_eval.py rather than imported from it, so this
    module (and the baseline scripts that use it) stay free of that file's top-level
    OpenAI client construction — this baseline makes no API calls at all.
    """
    tag_names = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
    answers = []
    current = []

    for word, tag in zip(words, ner_tags):
        tag_name = tag_names[tag]
        if tag_name == 'B-ANSWER':
            if current:
                answers.append(' '.join(current))
            current = [word]
        elif tag_name == 'I-ANSWER':
            current.append(word)
        else:
            if current:
                answers.append(' '.join(current))
                current = []

    if current:
        answers.append(' '.join(current))

    return answers


def extract_form_values(words):
    """FUNSD-style baseline: extracts candidate field values by the same kind of surface
    heuristic evaluation/baseline.py uses for bank statements — a word ending in ':' is
    treated as a label, and the following 1-3 words as its value — plus any standalone
    date/number tokens. No NER tags, no layout, just pattern-matching over the raw word list.
    """
    values = []
    i, n = 0, len(words)
    while i < n:
        w = words[i]
        if w.endswith(":"):
            j = i + 1
            chunk = []
            while j < n and len(chunk) < 3 and not words[j].endswith(":"):
                chunk.append(words[j])
                j += 1
            if chunk:
                values.append(" ".join(chunk))
            i = j
        else:
            if DATE_PATTERN.fullmatch(w) or NUMBER_PATTERN.fullmatch(w):
                values.append(w)
            i += 1
    return values