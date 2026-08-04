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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NUM_SAMPLES = 50


def image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def gpt4o_single_pass(image, question):
    image_b64 = image_to_base64(image)

    response = client.chat.completions.create(
        model="gpt-4o",
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


def run_docvqa_gpt4o_evaluation():
    print("=" * 60)
    print("DOCVQA EVALUATION - GPT-4o Single-pass")
    print(f"Running on {NUM_SAMPLES} samples")
    print("=" * 60)

    print("\nLoading DocVQA dataset...")
    ds = load_dataset('nielsr/docvqa_1200_examples', split='train')
    samples = ds.select(range(NUM_SAMPLES))

    gpt4o_scores = []
    results = []

    for i, example in enumerate(samples):
        question = example['query']['en']
        ground_truths = example['answers']
        image = example['image']

        print(f"\n[{i+1}/{NUM_SAMPLES}] Q: {question[:60]}...")
        print(f"      GT: {ground_truths}")

        try:
            start = time.time()
            prediction = gpt4o_single_pass(image, question)
            elapsed = time.time() - start

            score = max(anls_score(prediction, gt) for gt in ground_truths)
            gpt4o_scores.append(score)

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
            gpt4o_scores.append(0.0)
            results.append({
                "id": i,
                "question": question,
                "ground_truth": ground_truths,
                "prediction": "ERROR",
                "anls": 0.0,
                "time_seconds": 0.0
            })

        time.sleep(0.5)

    avg_anls = sum(gpt4o_scores) / len(gpt4o_scores) if gpt4o_scores else 0

    qwen_anls = None
    qwen_results_path = "evaluation/results/docvqa_results.json"
    if os.path.exists(qwen_results_path):
        with open(qwen_results_path) as f:
            qwen_anls = json.load(f).get("qwen_vl_max_anls")

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(gpt4o_scores)}")
    print(f"GPT-4o single pass ANLS: {avg_anls:.3f}")
    if qwen_anls is not None:
        print(f"Qwen2-VL-max single pass ANLS: {qwen_anls:.3f} (from {qwen_results_path})")
        print(f"Difference (GPT-4o - Qwen2-VL-max): {avg_anls - qwen_anls:+.3f}")
    print(f"Reference - LayoutLMv3 LARGE: 0.834")

    os.makedirs("evaluation/results", exist_ok=True)
    output = {
        "dataset": "DocVQA (nielsr/docvqa_1200_examples)",
        "num_samples": NUM_SAMPLES,
        "gpt4o_anls": round(avg_anls, 3),
        "qwen_vl_max_anls": qwen_anls,
        "reference_layoutlmv3": 0.834,
        "per_question_results": results
    }

    with open("evaluation/results/docvqa_gpt4o_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to evaluation/results/docvqa_gpt4o_results.json")
    print(f"{'='*60}")

    return avg_anls


if __name__ == "__main__":
    run_docvqa_gpt4o_evaluation()