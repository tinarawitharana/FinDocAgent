"""RQ2 evidence: naive regex/keyword-matching baseline on the same DocVQA sample as
docvqa_eval.py, using the dataset's own OCR `words` field (no VLM, no vision, no API
call — see evaluation/regex_utils.py for the extraction method). This is the text-only
reference point that motivates using a VLM at all, extending the same regex-vs-VLM
comparison evaluation/baseline.py made for bank statements to DocVQA specifically."""

import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from evaluation.metrics import anls_score
from evaluation.regex_utils import extract_answer

NUM_SAMPLES = 50


def run_docvqa_baseline_evaluation():
    print("=" * 60)
    print("DOCVQA EVALUATION - Regex/Keyword Baseline (no VLM)")
    print(f"Running on {NUM_SAMPLES} samples")
    print("=" * 60)

    print("\nLoading DocVQA dataset...")
    ds = load_dataset('nielsr/docvqa_1200_examples', split='train')
    samples = ds.select(range(NUM_SAMPLES))

    regex_scores = []
    results = []

    for i, example in enumerate(samples):
        question = example['query']['en']
        ground_truths = example['answers']
        words = example['words']

        print(f"\n[{i+1}/{NUM_SAMPLES}] Q: {question[:60]}...")
        print(f"      GT: {ground_truths}")

        start = time.time()
        prediction = extract_answer(words, question)
        elapsed = time.time() - start

        score = max(anls_score(prediction, gt) for gt in ground_truths)
        regex_scores.append(score)

        print(f"    Pred: {prediction}")
        print(f"    ANLS: {score:.3f}  ({elapsed*1000:.1f}ms)")

        results.append({
            "id": i,
            "question": question,
            "ground_truth": ground_truths,
            "prediction": prediction,
            "anls": round(score, 3),
        })

    avg_anls = sum(regex_scores) / len(regex_scores) if regex_scores else 0

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(regex_scores)}")
    print(f"Regex baseline ANLS: {avg_anls:.3f}")
    print(f"Qwen2-VL-max (docvqa_eval.py): 0.952")

    os.makedirs("evaluation/results", exist_ok=True)
    output = {
        "dataset": "DocVQA (nielsr/docvqa_1200_examples)",
        "num_samples": NUM_SAMPLES,
        "regex_baseline_anls": round(avg_anls, 3),
        "qwen_vl_max_anls": 0.952,
        "reference_layoutlmv3": 0.834,
        "per_question_results": results,
    }

    with open("evaluation/results/docvqa_baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to evaluation/results/docvqa_baseline_results.json")
    print(f"{'='*60}")

    return avg_anls


if __name__ == "__main__":
    run_docvqa_baseline_evaluation()