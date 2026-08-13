"""
Runs Gemma 3 4B (google/gemma-3-4b-it) locally via HuggingFace transformers on
the FUNSD test set, mirroring evaluation/docvqa_gemma_eval.py's local-loading
approach (bf16 first, 4-bit fallback on OOM). See that file's module docstring
for the VRAM-fit rationale.
"""
import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from evaluation.metrics import anls_score
from evaluation.docvqa_gemma_eval import load_model, gemma_single_pass
import evaluation.docvqa_gemma_eval as gemma_module


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


def gemma_funsd_pass(image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "List all the filled-in field values from this form, one per line. Only output the values, nothing else."}
        ]
    }]

    inputs = gemma_module._processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(gemma_module._model.device, dtype=torch.bfloat16 if gemma_module._load_mode == "bf16" else torch.float16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = gemma_module._model.generate(**inputs, max_new_tokens=200, do_sample=False)

    generated = output[0][input_len:]
    return gemma_module._processor.decode(generated, skip_special_tokens=True).strip()


def run_funsd_gemma_evaluation():
    print("=" * 60)
    print(f"FUNSD EVALUATION - {gemma_module.MODEL_ID} (local)")
    print("=" * 60)

    load_model()

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
            start = time.time()
            prediction_text = gemma_funsd_pass(image)
            elapsed = time.time() - start

            predicted_values = [line.strip() for line in prediction_text.split("\n") if line.strip()]

            print(f"        Predicted: {predicted_values[:3]}...  ({elapsed:.1f}s)")

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

    avg_anls = sum(all_scores) / len(all_scores) if all_scores else 0

    qwen_anls = None
    qwen_results_path = "evaluation/results/funsd_results.json"
    if os.path.exists(qwen_results_path):
        with open(qwen_results_path) as f:
            qwen_anls = json.load(f).get("anls")

    print(f"\n{'='*60}")
    print(f"{gemma_module.MODEL_ID} FUNSD ANLS: {avg_anls:.3f}  (n={len(all_scores)})")
    if qwen_anls is not None:
        print(f"Qwen2-VL-max FUNSD ANLS: {qwen_anls:.3f} (from {qwen_results_path})")
        print(f"Difference ({gemma_module.MODEL_ID} - Qwen2-VL-max): {avg_anls - qwen_anls:+.3f}")
    print(f"{'='*60}")

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/funsd_gemma_results.json", "w") as f:
        json.dump({
            "dataset": "FUNSD",
            "model": gemma_module.MODEL_ID,
            "load_mode": gemma_module._load_mode,
            "num_samples": len(all_scores),
            "gemma_anls": round(avg_anls, 3),
            "qwen_vl_max_anls": qwen_anls,
            "per_doc": results
        }, f, indent=2)

    print("Saved to evaluation/results/funsd_gemma_results.json")
    return avg_anls


if __name__ == "__main__":
    run_funsd_gemma_evaluation()