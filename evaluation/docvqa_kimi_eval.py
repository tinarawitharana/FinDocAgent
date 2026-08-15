"""Runs Kimi (Moonshot AI) on a DocVQA sample, as one of the multi-model baseline
comparison points alongside GPT-4o/Gemini/Gemma/SmolVLM."""

import os
import re
import sys
import base64
import json
import time
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from evaluation.metrics import anls_score
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))

client = OpenAI(
    api_key=os.getenv("KIMI_API_KEY"),
    base_url="https://api.moonshot.ai/v1"
)

MODEL = "kimi-k3"
NUM_SAMPLES = 50

# Tier0 (min $1 recharge) allows 3 requests/minute — confirmed via live 429 errors.
SECONDS_BETWEEN_CALLS = 21
MAX_RETRIES = 4
# K3 is an "always-thinking" model — reasoning tokens are spent before the visible
# answer, so max_tokens must leave room for both (confirmed via usage.reasoning_tokens
# during setup, a low max_tokens truncates before any visible content appears).
MAX_TOKENS = 500


def image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_retry_delay(error, default=25):
    match = re.search(r"after (\d+) seconds", str(error))
    if match:
        return float(match.group(1)) + 2
    return default


def kimi_single_pass(image, question):
    image_b64 = image_to_base64(image)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": f"{question}\nAnswer with the exact text from the document.Be concise, one word or short phrase only."
                            }
                        ]
                    }
                ],
                max_tokens=MAX_TOKENS
            )
            return (response.choices[0].message.content or "").strip()
        except RateLimitError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = _extract_retry_delay(e)
            print(f"    Rate limited, retrying in {delay:.0f}s (attempt {attempt+1}/{MAX_RETRIES})...")
            time.sleep(delay)


def run_docvqa_kimi_evaluation():
    print("=" * 60)
    print(f"DOCVQA EVALUATION - {MODEL} Single-pass")
    print(f"Running on {NUM_SAMPLES} samples")
    print("=" * 60)

    print("\nLoading DocVQA dataset...")
    ds = load_dataset('nielsr/docvqa_1200_examples', split='train')
    samples = ds.select(range(NUM_SAMPLES))

    kimi_scores = []
    results = []

    for i, example in enumerate(samples):
        question = example['query']['en']
        ground_truths = example['answers']
        image = example['image']

        print(f"\n[{i+1}/{NUM_SAMPLES}] Q: {question[:60]}...")
        print(f"      GT: {ground_truths}")

        try:
            start = time.time()
            prediction = kimi_single_pass(image, question)
            elapsed = time.time() - start

            score = max(anls_score(prediction, gt) for gt in ground_truths)
            kimi_scores.append(score)

            print(f"    Pred: {prediction}")
            print(f"    ANLS: {score:.3f}  ({elapsed:.1f}s)")

            results.append({
                "id": i,
                "question": question,
                "ground_truth": ground_truths,
                "prediction": prediction,
                "anls": round(score, 3),
                "time_seconds": round(elapsed, 2)
            })

        except Exception as e:
            print(f"      ERROR: {e}")
            kimi_scores.append(0.0)
            results.append({
                "id": i,
                "question": question,
                "ground_truth": ground_truths,
                "prediction": "ERROR",
                "anls": 0.0,
                "time_seconds": 0.0
            })

        time.sleep(SECONDS_BETWEEN_CALLS)

    avg_anls = sum(kimi_scores) / len(kimi_scores) if kimi_scores else 0

    qwen_anls = None
    qwen_results_path = "evaluation/results/docvqa_results.json"
    if os.path.exists(qwen_results_path):
        with open(qwen_results_path) as f:
            qwen_anls = json.load(f).get("qwen_vl_max_anls")

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(kimi_scores)}")
    print(f"{MODEL} ANLS: {avg_anls:.3f}")
    if qwen_anls is not None:
        print(f"Qwen2-VL-max single pass ANLS: {qwen_anls:.3f} (from {qwen_results_path})")
        print(f"Difference ({MODEL} - Qwen2-VL-max): {avg_anls - qwen_anls:+.3f}")
    print(f"Reference - LayoutLMv3 LARGE: 0.834")

    os.makedirs("evaluation/results", exist_ok=True)
    output = {
        "dataset": "DocVQA (nielsr/docvqa_1200_examples)",
        "model": MODEL,
        "num_samples": NUM_SAMPLES,
        "kimi_anls": round(avg_anls, 3),
        "qwen_vl_max_anls": qwen_anls,
        "reference_layoutlmv3": 0.834,
        "per_question_results": results
    }

    with open("evaluation/results/docvqa_kimi_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to evaluation/results/docvqa_kimi_results.json")
    print(f"{'='*60}")

    return avg_anls


if __name__ == "__main__":
    run_docvqa_kimi_evaluation()