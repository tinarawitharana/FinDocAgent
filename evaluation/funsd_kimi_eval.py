"""Runs Kimi (Moonshot AI) on a FUNSD sample, mirroring docvqa_kimi_eval.py's approach."""

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
    api_key=os.getenv("KIMI_API_KEY"),
    base_url="https://api.moonshot.ai/v1"
)

MODEL = "kimi-k3"

# Tier0 (min $1 recharge) allows 3 requests/minute — confirmed via live 429 errors.
SECONDS_BETWEEN_CALLS = 21
MAX_RETRIES = 4
# K3 is an "always-thinking" model — reasoning tokens are spent before the visible
# answer, so max_tokens must leave room for both.
MAX_TOKENS = 500


def _extract_retry_delay(error, default=25):
    match = re.search(r"after (\d+) seconds", str(error))
    if match:
        return float(match.group(1)) + 2
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


def run_funsd_kimi_evaluation():
    print("=" * 60)
    print(f"FUNSD EVALUATION - {MODEL}")
    print("=" * 60)

    ds = load_dataset("nielsr/funsd", split="test")

    all_scores = []
    results = []

    for i, example in enumerate(ds):
        words = example["words"]
        ner_tags = example["ner_tags"]
        image = example["image"]
        doc_id = example["id"]

        gt_answers = extract_gt_answers(words, ner_tags)

        if not gt_answers:
            print(f"[{i+1}/{len(ds)}] Skipping — no answer entities")
            continue

        print(f"\n[{i+1}/{len(ds)}] Doc: {doc_id}")
        print(f"        GT answers: {gt_answers[:3]}...")

        try:
            image_b64 = image_to_base64(image)

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
                        max_tokens=MAX_TOKENS
                    )
                    break
                except RateLimitError as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    delay = _extract_retry_delay(e)
                    print(f"        Rate limited, retrying in {delay:.0f}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    time.sleep(delay)

            prediction_text = (response.choices[0].message.content or "").strip()
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
    with open("evaluation/results/funsd_kimi_results.json", "w") as f:
        json.dump({
            "dataset": "FUNSD",
            "model": MODEL,
            "num_samples": len(all_scores),
            "kimi_anls": round(avg_anls, 3),
            "qwen_vl_max_anls": qwen_anls,
            "per_doc": results
        }, f, indent=2)

    print("Saved to evaluation/results/funsd_kimi_results.json")
    return avg_anls


if __name__ == "__main__":
    run_funsd_kimi_evaluation()