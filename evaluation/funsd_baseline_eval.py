"""RQ2 evidence: naive regex/heuristic field-value extraction baseline on the same FUNSD
sample as funsd_eval.py, using the dataset's own OCR `words` field (no VLM, no vision, no
API call — see evaluation/regex_utils.py for the extraction method). Mirrors funsd_eval.py's
"list all filled-in field values, best-match each ground-truth answer against them" scoring,
so the two are directly comparable."""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from evaluation.metrics import anls_score
from evaluation.regex_utils import extract_gt_answers, extract_form_values

NUM_SAMPLES = 50


def run_funsd_baseline_evaluation():
    print("=" * 60)
    print("FUNSD EVALUATION - Regex/Heuristic Baseline (no VLM)")
    print(f"Running on {NUM_SAMPLES} samples")
    print("=" * 60)

    ds = load_dataset("nielsr/funsd", split="test")

    all_scores = []
    results = []

    for i, example in enumerate(ds):
        words = example["words"]
        ner_tags = example["ner_tags"]
        doc_id = example["id"]

        gt_answers = extract_gt_answers(words, ner_tags)

        if not gt_answers:
            print(f"[{i+1}/{NUM_SAMPLES}] Skipping — no answer entities")
            continue

        print(f"\n[{i+1}/{NUM_SAMPLES}] Doc: {doc_id}")
        print(f"        GT answers: {gt_answers[:3]}...")

        predicted_values = extract_form_values(words)
        print(f"        Predicted: {predicted_values[:3]}...")

        doc_scores = []
        for gt in gt_answers:
            if predicted_values:
                best = max(anls_score(pred, gt) for pred in predicted_values)
            else:
                best = 0.0
            doc_scores.append(best)

        doc_anls = sum(doc_scores) / len(doc_scores)
        all_scores.append(doc_anls)

        print(f"        Doc ANLS: {doc_anls:.3f}")

        results.append({
            "id": doc_id,
            "gt_answers": gt_answers,
            "predicted_values": predicted_values,
            "anls": round(doc_anls, 3),
        })

    avg_anls = sum(all_scores) / len(all_scores) if all_scores else 0

    print(f"\n{'='*60}")
    print(f"Regex baseline FUNSD ANLS: {avg_anls:.3f}")
    print(f"Qwen2-VL-max (funsd_eval.py): 0.792")
    print(f"{'='*60}")

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/funsd_baseline_results.json", "w") as f:
        json.dump({
            "dataset": "FUNSD",
            "num_samples": len(results),
            "regex_baseline_anls": round(avg_anls, 3),
            "qwen_vl_max_anls": 0.792,
            "per_doc": results,
        }, f, indent=2)

    print("Saved to evaluation/results/funsd_baseline_results.json")
    return avg_anls


if __name__ == "__main__":
    run_funsd_baseline_evaluation()