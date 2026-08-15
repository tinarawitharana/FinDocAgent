# FinDocAgent — XAI Human Evaluation: Rater Instructions

Thank you for helping evaluate FinDocAgent's explainability outputs. This should take about 30-50 minutes for the documents you're assigned. You're judging whether the AI's *explanations* make sense, not whether its financial reasoning is technically correct.

## What you'll see for each document

1. **The original document image** (a bank statement or a page from a company's annual report).
2. **An explanation visual** — a heatmap overlay showing which part of the document the AI focused on, and (for bank statements only) a short breakdown of which issue, for example a math discrepancy, an out-of-order date, or both which drove the AI's anomaly score.
3. **The AI's actual output** — its answer to a question (for annual report pages) or its flagged anomaly/risk summary (for bank statements).

## What to do

For each document, look at all three things together, then give a score from **1 (strongly disagree) to 5 (strongly agree)** for each question below. There are no right or wrong answers — we want your honest, independent judgment.

1. **Localization** — *The highlighted region actually overlaps with where the relevant information is on the page.*
   (1 = highlighted the wrong part entirely, 5 = highlighted exactly the right spot)

2. **Clarity** — *This explanation is easy to understand at a glance, without needing anything explained to me.*
   (1 = confusing, 5 = immediately obvious)

3. **Actionability** — *If I had to check this document myself, this explanation would give me enough to verify or challenge the AI's answer, without reading the whole document from scratch.*
   (1 = no help at all, 5 = fully sufficient on its own)

4. **Trust** — *Based on this explanation, I would trust the AI's output for this document.*
   (1 = not at all, 5 = completely)

Feel free to add a short comment on anything that felt off or particularly good, a sentence is plenty.

## A note on bank statements specifically

Some documents will also show a "SHAP breakdown" (e.g. "60% math discrepancy, 40% date issue"). For these, you'll see one extra question:

5. **SHAP Consistency** — *This breakdown matches what I can see actually looks wrong (or not wrong) in the document.*

You don't need to independently verify the bank statement's arithmetic, just judge whether the breakdown seems to line up with what the highlighted/flagged parts of the document show.

## How to submit

Fill in your scores directly in the shared spreadsheet, one row per document.
