"""Runs Qwen2-VL-max directly (single hosted-API call per question, no agent/RAG) on a
DocVQA sample, as the base-model reference point for the agent/RAG comparisons."""

import os
import sys
import base64
import json
import time
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from evaluation.metrics import anls_score
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))

client = OpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

)

NUM_SAMPLES = 50

def image_to_base64(pil_image):

    #convert PIL image to base64 string
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def qwen_single_pass(image, question):

    image_b64 = image_to_base64(image)

    response = client.chat.completions.create(
        model = "qwen-vl-max",
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
        max_tokens=50
    )

    return response.choices[0].message.content.strip()

def run_docvqa_evaluation():
    print("=" * 60)
    print("DOCVQA EVALUATION - Qwen2-VL Single-pass")
    print(f"Running on {NUM_SAMPLES} samples")
    print("=" * 60)

    #load dataset
    print("\nLoading DocVQA dataset...")
    ds = load_dataset('nielsr/docvqa_1200_examples', split='train')

    #take first NUM_SAMPLES
    samples = ds.select(range(NUM_SAMPLES))

    qwen_scores = []
    results = []

    for i, example in enumerate(samples):
        question = example['query']['en']
        ground_truths = example['answers']
        image = example['image']

        print(f"\n[{i+1}/{NUM_SAMPLES}] Q: {question[:60]}...")
        print(f"      GT: {ground_truths}")

        try:
            start = time.time()
            prediction = qwen_single_pass(image, question)
            elapsed = time.time() - start

            score = max(anls_score(prediction, gt) for gt in ground_truths)
            qwen_scores.append(score)

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
            qwen_scores.append(0.0)
            results.append({
                "id": id,
                "question": question,
                "ground_truth": ground_truths,
                "prediction": "ERROR",
                "anls": 0.0,
                "time_seconds": 0.0

            })

        time.sleep(0.5)

    avg_anls = sum(qwen_scores) / len(qwen_scores) if qwen_scores else 0

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(qwen_scores)}")
    print(f"Qwen2-VL single pass ANLS: {avg_anls:.3f}")
    print(f"Reference - LayoutLMv3 LARGE: 0.834")
    print(f"Reference - Qwen2-VL 7B: 0.945")

    if avg_anls > 0.834:
        print(f"Result: EXCEEDS LayoutLMv3 benchmark")
    elif avg_anls > 0.700:
        print(f"Result: STRONG Performance")
    else:
        print(f"Result: Note-using API version, may differ from paper")

    os.makedirs("evaluation/results", exist_ok=True)
    output = {
        "dataset": "DocVQA (nielsr/docvqa_1200_examples)",
        "num_samples": NUM_SAMPLES,
        "qwen_vl_max_anls": round(avg_anls, 3),
        "reference_layoutlmv3": 0.834,
        "reference_qwen2vl_7b": 0.945,
        "per_question_results": results
    }

    with open("evaluation/results/docvqa_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to evaluation/results/docvqa_results.json")
    print(f"{'='*60}")

    return avg_anls

if __name__ == "__main__":
    run_docvqa_evaluation()