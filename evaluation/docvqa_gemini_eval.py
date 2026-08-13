"""Runs Gemini 3.5 Flash (via Google's OpenAI-compatible endpoint) on a small DocVQA
sample, as one of the multi-model baseline comparison points alongside GPT-4o/Kimi/
Gemma/InternVL/SmolVLM. Sample size is capped by Gemini's free-tier daily request limit."""

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
    api_key=os.getenv("GEMINI_API_KEY_2"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL = "gemini-3.5-flash"
# free tier caps at 20 requests/day total for this model — split as 8 DocVQA + 8
# FUNSD (16 total), leaving a 4-request buffer for retries within the daily cap.
NUM_SAMPLES = 8

# free tier for gemini-3.5-flash is 5 requests/minute (12s/req minimum) — confirmed
# from a live 429 error, not documentation, since published free-tier numbers were
# for other models and didn't match what this key actually enforces.
SECONDS_BETWEEN_CALLS = 13
MAX_RETRIES = 4


def image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_retry_delay(error, default=35):
    match = re.search(r"retry in ([\d.]+)s", str(error))
    if match:
        return float(match.group(1)) + 2  # small margin
    return default


def gemini_single_pass(image, question):
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
                max_tokens=50,
                extra_body={"reasoning_effort": "none"}  # avoid thinking tokens eating the whole budget (see finish_reason='length' bug found during dry-run)
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = _extract_retry_delay(e)
            print(f"    Rate limited, retrying in {delay:.0f}s (attempt {attempt+1}/{MAX_RETRIES})...")
            time.sleep(delay)


def run_docvqa_gemini_evaluation():
    print("=" * 60)
    print(f"DOCVQA EVALUATION - {MODEL} Single-pass")
    print(f"Running on {NUM_SAMPLES} samples")
    print("=" * 60)

    print("\nLoading DocVQA dataset...")
    ds = load_dataset('nielsr/docvqa_1200_examples', split='train')
    samples = ds.select(range(NUM_SAMPLES))

    gemini_scores = []
    results = []

    for i, example in enumerate(samples):
        question = example['query']['en']
        ground_truths = example['answers']
        image = example['image']

        print(f"\n[{i+1}/{NUM_SAMPLES}] Q: {question[:60]}...")
        print(f"      GT: {ground_truths}")

        try:
            start = time.time()
            prediction = gemini_single_pass(image, question)
            elapsed = time.time() - start

            score = max(anls_score(prediction, gt) for gt in ground_truths)
            gemini_scores.append(score)

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
            gemini_scores.append(0.0)
            results.append({
                "id": i,
                "question": question,
                "ground_truth": ground_truths,
                "prediction": "ERROR",
                "anls": 0.0,
                "time_seconds": 0.0
            })

        time.sleep(SECONDS_BETWEEN_CALLS)

    avg_anls = sum(gemini_scores) / len(gemini_scores) if gemini_scores else 0

    qwen_anls = None
    qwen_results_path = "evaluation/results/docvqa_results.json"
    if os.path.exists(qwen_results_path):
        with open(qwen_results_path) as f:
            qwen_anls = json.load(f).get("qwen_vl_max_anls")

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(gemini_scores)}")
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
        "gemini_anls": round(avg_anls, 3),
        "qwen_vl_max_anls": qwen_anls,
        "reference_layoutlmv3": 0.834,
        "per_question_results": results
    }

    with open("evaluation/results/docvqa_gemini_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to evaluation/results/docvqa_gemini_results.json")
    print(f"{'='*60}")

    return avg_anls


if __name__ == "__main__":
    run_docvqa_gemini_evaluation()