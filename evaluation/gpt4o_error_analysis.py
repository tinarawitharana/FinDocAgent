"""
Classifies GPT-4o's low-ANLS answers on DocVQA/FUNSD as either a genuine
misread or a correct-but-verbose answer that ANLS's edit-distance metric
penalizes disproportionately (e.g. "Five" vs "5", "Two focus groups" vs "Two").

A prediction is "verbose-but-correct" if, after lowercasing and stripping
punctuation, the ground truth is a substring of the prediction or vice versa,
or one is the other with a number spelled out as a word.
"""
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "40": "forty",
}


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[.,;:!?'\"]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


MIN_CONTAINMENT_LEN = 4


def is_verbose_but_correct(pred, gt):
    p, g = normalize(pred), normalize(gt)
    if not p or not g:
        return False
    # guard against trivial matches, e.g. a lone "3" matching as a
    # "substring" of an unrelated multi-sentence ground truth
    if len(p) >= MIN_CONTAINMENT_LEN and len(g) >= MIN_CONTAINMENT_LEN:
        if g in p or p in g:
            return True
    for digit, word in NUMBER_WORDS.items():
        digit_re = re.compile(rf"\b{re.escape(digit)}\b")
        word_re = re.compile(rf"\b{re.escape(word)}\b")
        if digit_re.search(g) and word_re.search(p):
            return True
        if word_re.search(g) and digit_re.search(p):
            return True
    return False


def classify_docvqa(path):
    with open(path) as f:
        data = json.load(f)

    genuine_miss = []
    verbose_correct = []
    already_scored_ok = []

    for row in data["per_question_results"]:
        if row["anls"] >= 0.5:
            already_scored_ok.append(row)
            continue
        gts = row["ground_truth"]
        pred = row["prediction"]
        if any(is_verbose_but_correct(pred, gt) for gt in gts):
            verbose_correct.append(row)
        else:
            genuine_miss.append(row)

    n = len(data["per_question_results"])
    print(f"\n=== DocVQA ({path}) ===")
    print(f"Total: {n}")
    print(f"Already ANLS >= 0.5: {len(already_scored_ok)}")
    print(f"Low-score but verbose-correct (content right, phrasing penalized): {len(verbose_correct)}")
    print(f"Genuine miss: {len(genuine_miss)}")

    adjusted_scores = []
    for row in data["per_question_results"]:
        if row["anls"] >= 0.5:
            adjusted_scores.append(row["anls"])
        elif any(is_verbose_but_correct(row["prediction"], gt) for gt in row["ground_truth"]):
            adjusted_scores.append(1.0)
        else:
            adjusted_scores.append(row["anls"])
    adjusted_anls = sum(adjusted_scores) / len(adjusted_scores)
    print(f"Raw ANLS: {data['gpt4o_anls']:.3f}")
    print(f"Adjusted ANLS (verbose-correct counted as 1.0): {adjusted_anls:.3f}")

    print("\nGenuine misses:")
    for row in genuine_miss:
        print(f"  Q: {row['question'][:55]!r}  GT: {row['ground_truth']}  Pred: {row['prediction']!r}")

    print("\nVerbose-but-correct (reclassified):")
    for row in verbose_correct:
        print(f"  GT: {row['ground_truth']}  Pred: {row['prediction']!r}  (raw anls={row['anls']:.2f})")

    return {
        "total": n,
        "already_ok": len(already_scored_ok),
        "verbose_correct": len(verbose_correct),
        "genuine_miss": len(genuine_miss),
        "raw_anls": data["gpt4o_anls"],
        "adjusted_anls": round(adjusted_anls, 3),
    }


def classify_funsd(path):
    with open(path) as f:
        data = json.load(f)

    from evaluation.metrics import anls_score

    total_gt = 0
    genuine_miss = []
    verbose_correct = []
    already_ok = 0
    doc_adjusted_means = []

    for doc in data["per_doc"]:
        preds = doc["predicted_values"]
        doc_adjusted_scores = []
        for gt in doc["gt_answers"]:
            total_gt += 1
            best = max((anls_score(p, gt) for p in preds), default=0.0)
            if best >= 0.5:
                already_ok += 1
                doc_adjusted_scores.append(best)
                continue
            if preds and any(is_verbose_but_correct(p, gt) for p in preds):
                verbose_correct.append((doc["id"], gt, preds))
                doc_adjusted_scores.append(1.0)
            else:
                genuine_miss.append((doc["id"], gt, preds))
                doc_adjusted_scores.append(best)
        if doc_adjusted_scores:
            doc_adjusted_means.append(sum(doc_adjusted_scores) / len(doc_adjusted_scores))

    adjusted_anls = sum(doc_adjusted_means) / len(doc_adjusted_means) if doc_adjusted_means else 0.0

    print(f"\n=== FUNSD ({path}) ===")
    print(f"Total GT fields: {total_gt}")
    print(f"Already ANLS >= 0.5: {already_ok}")
    print(f"Low-score but verbose-correct: {len(verbose_correct)}")
    print(f"Genuine miss: {len(genuine_miss)}")
    print(f"Raw ANLS: {data['gpt4o_anls']:.3f}")
    print(f"Adjusted ANLS (verbose-correct counted as 1.0): {adjusted_anls:.3f}")

    print("\nGenuine misses (sample of 15):")
    for doc_id, gt, preds in genuine_miss[:15]:
        print(f"  Doc {doc_id}  GT: {gt!r}  Preds: {preds[:5]}")

    print("\nVerbose-but-correct (sample of 15):")
    for doc_id, gt, preds in verbose_correct[:15]:
        print(f"  Doc {doc_id}  GT: {gt!r}  Preds: {preds[:5]}")

    return {
        "total_gt_fields": total_gt,
        "already_ok": already_ok,
        "verbose_correct": len(verbose_correct),
        "genuine_miss": len(genuine_miss),
        "raw_anls": data["gpt4o_anls"],
        "adjusted_anls": round(adjusted_anls, 3),
    }


if __name__ == "__main__":
    summary = {}
    docvqa_path = "evaluation/results/docvqa_gpt4o_results.json"
    funsd_path = "evaluation/results/funsd_gpt4o_results.json"

    if os.path.exists(docvqa_path):
        summary["docvqa"] = classify_docvqa(docvqa_path)
    if os.path.exists(funsd_path):
        summary["funsd"] = classify_funsd(funsd_path)

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/gpt4o_error_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to evaluation/results/gpt4o_error_analysis.json")