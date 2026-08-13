"""Runs Gemini 3.5 Flash on a small FUNSD sample, mirroring docvqa_gemini_eval.py's
approach (see that file's module docstring for the free-tier sample-size rationale)."""

import os
import re
import sys
import json
import time
import tempfile
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
from evaluation.metrics import anls_score

load_dotenv(os.path.expanduser("~/.env"))

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY_2"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL = "gemini-3.5-flash"

# free tier for gemini-3.5-flash is 5 requests/minute (12s/req minimum) — confirmed
# from a live 429 error, not documentation, since published free-tier numbers were
# for other models and didn't match what this key actually enforces.
SECONDS_BETWEEN_CALLS = 13
MAX_RETRIES = 4
# free tier also caps at 20 requests/day total — split as 8 DocVQA + 8 FUNSD (16
# total), leaving a 4-request buffer for retries within the daily cap.
NUM_SAMPLES = 8


def _extract_retry_delay(error, default=35):
    match = re.search(r"retry in ([\d.]+)s", str(error))
    if match:
        return float(match.group(1)) + 2  # small margin
    return default


def extract_gt_answers(words, ner_tags):
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


def image_to_base64(pil_image):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name, "PNG")
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)
    return b64


def run_funsd_gemini_evaluation():
    print("=" * 60)
    print(f"FUNSD EVALUATION - {MODEL}")
    print("=" * 60)

    ds = load_dataset("nielsr/funsd", split="test")

    all_scores = []
    results = []
    calls_made = 0

    for i, example in enumerate(ds):
        if calls_made >= NUM_SAMPLES:
            print(f"\nReached NUM_SAMPLES={NUM_SAMPLES} cap (daily free-tier quota), stopping.")
            break

        words = example["words"]
        ner_tags = example["ner_tags"]
        image = example["image"]
        doc_id = example["id"]

        gt_answers = extract_gt_answers(words, ner_tags)

        if not gt_answers:
            print(f"[call {calls_made}/{NUM_SAMPLES}, doc {i+1}/{len(ds)}] Skipping — no answer entities")
            continue

        print(f"\n[call {calls_made}/{NUM_SAMPLES}, doc {i+1}/{len(ds)}] Doc: {doc_id}")
        print(f"        GT answers: {gt_answers[:3]}...")

        try:
            image_b64 = image_to_base64(image)
            calls_made += 1

            response = None
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
                                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                                    },
                                    {
                                        "type": "text",
                                        "text": f"List all the filled-in field values from this form, one per line. Only output the values, nothing else."
                                    }
                                ]
                            }
                        ],
                        temperature=0,
                        extra_body={"reasoning_effort": "none"}  # avoid thinking tokens eating the whole budget (see finish_reason='length' bug found during dry-run)
                    )
                    break
                except RateLimitError as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    delay = _extract_retry_delay(e)
                    print(f"        Rate limited, retrying in {delay:.0f}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    time.sleep(delay)

            prediction_text = response.choices[0].message.content.strip()
            predicted_values = [line.strip() for line in prediction_text.split("\n") if line.strip()]

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
                "anls": round(doc_anls, 3)
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            all_scores.append(0.0)

        time.sleep(SECONDS_BETWEEN_CALLS)

    avg_anls = sum(all_scores) / len(all_scores) if all_scores else 0

    qwen_anls = None
    qwen_results_path = "evaluation/results/funsd_results.json"
    if os.path.exists(qwen_results_path):
        with open(qwen_results_path) as f:
            qwen_anls = json.load(f).get("anls")

    print(f"\n{'='*60}")
    print(f"{MODEL} FUNSD ANLS: {avg_anls:.3f}  (n={len(all_scores)})")
    if qwen_anls is not None:
        print(f"Qwen2-VL-max FUNSD ANLS: {qwen_anls:.3f} (from {qwen_results_path})")
        print(f"Difference ({MODEL} - Qwen2-VL-max): {avg_anls - qwen_anls:+.3f}")
    print(f"{'='*60}")

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/funsd_gemini_results.json", "w") as f:
        json.dump({
            "dataset": "FUNSD",
            "model": MODEL,
            "num_samples": len(all_scores),
            "gemini_anls": round(avg_anls, 3),
            "qwen_vl_max_anls": qwen_anls,
            "per_doc": results
        }, f, indent=2)

    print("Saved to evaluation/results/funsd_gemini_results.json")
    return avg_anls


if __name__ == "__main__":
    run_funsd_gemini_evaluation()