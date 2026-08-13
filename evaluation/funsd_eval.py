"""Runs Qwen2-VL-max directly on a FUNSD (form understanding) sample, deriving QA pairs
from the dataset's NER tags, as the base-model reference point for FUNSD comparisons."""

import os
import sys
import json
import time
import tempfile
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
from evaluation.metrics import anls_score

load_dotenv(os.path.expanduser("~/.env"))

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

NUM_SAMPLES = 50


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

def run_funsd_evaluation():
    print("="*60)
    print("FUNSD EVALUATION - FinDocAgent")
    print(f"Running on {NUM_SAMPLES} samples")
    print("="*60)

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
            print(f"[{i+1}/{NUM_SAMPLES}] Skipping — no answer entities")
            continue

        print(f"\n[{i+1}/{NUM_SAMPLES}] Doc: {doc_id}")
        print(f"        GT answers: {gt_answers[:3]}...")

        try:
            image_b64 = image_to_base64(image)

            response = client.chat.completions.create(
                model="qwen-vl-max",
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
                temperature=0
            )

            prediction_text = response.choices[0].message.content.strip()
            predicted_values = [line.strip() for line in prediction_text.split("\n") if line.strip()]

            print(f"        Predicted: {predicted_values[:3]}...")

            # for each GT answer, find best matching prediction
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

        time.sleep(0.5)

    avg_anls = sum(all_scores) / len(all_scores) if all_scores else 0

    print(f"\n{'='*60}")
    print(f"FUNSD ANLS: {avg_anls:.3f}")
    print(f"{'='*60}")

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/funsd_results.json", "w") as f:
        json.dump({"dataset": "FUNSD", "num_samples": NUM_SAMPLES,
                   "anls": round(avg_anls, 3), "per_doc": results}, f, indent=2)

    print("Saved to evaluation/results/funsd_results.json")
    return avg_anls

if __name__ == "__main__":
    run_funsd_evaluation()

