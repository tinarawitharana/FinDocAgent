"""
Runs Gemma 3 4B (google/gemma-3-4b-it) locally via HuggingFace transformers on
the same DocVQA sample used for the Qwen2-VL-max/GPT-4o/Kimi comparisons.

No API involved — this loads real model weights onto the local GPU, unlike the
other *_eval.py scripts in this file which call a hosted API. Tries bf16 first
(best quality) and falls back to 4-bit quantization on CUDA OOM, since the 8GB
VRAM budget on this shared, HAMI-virtualized GPU is tight for a 4B model in
bf16 (~8GB for weights alone before activations/vision encoder/KV cache) — see
notes/phase11_attention_map_model_investigation.md for the same tradeoff
already hit and documented with Qwen2-VL-7B on this same GPU.
"""
import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

import torch
from datasets import load_dataset
from evaluation.metrics import anls_score
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

MODEL_ID = "google/gemma-3-4b-it"
NUM_SAMPLES = 50

_model = None
_processor = None
_load_mode = None


def load_model():
    global _model, _processor, _load_mode

    if _model is not None:
        return

    _processor = AutoProcessor.from_pretrained(MODEL_ID)

    try:
        print(f"[GEMMA] Loading {MODEL_ID} in bf16...")
        _model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
        ).eval()
        _load_mode = "bf16"
        print("[GEMMA] Loaded in bf16.")
    except torch.cuda.OutOfMemoryError:
        print("[GEMMA] bf16 OOM'd, falling back to 4-bit quantization...")
        torch.cuda.empty_cache()
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        _model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", quantization_config=quant_config
        ).eval()
        _load_mode = "4bit"
        print("[GEMMA] Loaded in 4-bit.")


def gemma_single_pass(image, question):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{question}\nAnswer with the exact text from the document. Be concise, one word or short phrase only."}
        ]
    }]

    inputs = _processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(_model.device, dtype=torch.bfloat16 if _load_mode == "bf16" else torch.float16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = _model.generate(**inputs, max_new_tokens=50, do_sample=False)

    generated = output[0][input_len:]
    return _processor.decode(generated, skip_special_tokens=True).strip()


def run_docvqa_gemma_evaluation():
    print("=" * 60)
    print(f"DOCVQA EVALUATION - {MODEL_ID} (local)")
    print(f"Running on {NUM_SAMPLES} samples")
    print("=" * 60)

    load_model()

    print("\nLoading DocVQA dataset...")
    ds = load_dataset('nielsr/docvqa_1200_examples', split='train')
    samples = ds.select(range(NUM_SAMPLES))

    gemma_scores = []
    results = []

    for i, example in enumerate(samples):
        question = example['query']['en']
        ground_truths = example['answers']
        image = example['image']

        print(f"\n[{i+1}/{NUM_SAMPLES}] Q: {question[:60]}...")
        print(f"      GT: {ground_truths}")

        try:
            start = time.time()
            prediction = gemma_single_pass(image, question)
            elapsed = time.time() - start

            score = max(anls_score(prediction, gt) for gt in ground_truths)
            gemma_scores.append(score)

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
            gemma_scores.append(0.0)
            results.append({
                "id": i,
                "question": question,
                "ground_truth": ground_truths,
                "prediction": "ERROR",
                "anls": 0.0,
                "time_seconds": 0.0
            })

    avg_anls = sum(gemma_scores) / len(gemma_scores) if gemma_scores else 0

    qwen_anls = None
    qwen_results_path = "evaluation/results/docvqa_results.json"
    if os.path.exists(qwen_results_path):
        with open(qwen_results_path) as f:
            qwen_anls = json.load(f).get("qwen_vl_max_anls")

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(gemma_scores)}")
    print(f"Load mode: {_load_mode}")
    print(f"{MODEL_ID} ANLS: {avg_anls:.3f}")
    if qwen_anls is not None:
        print(f"Qwen2-VL-max single pass ANLS: {qwen_anls:.3f} (from {qwen_results_path})")
        print(f"Difference ({MODEL_ID} - Qwen2-VL-max): {avg_anls - qwen_anls:+.3f}")
    print(f"Reference - LayoutLMv3 LARGE: 0.834")

    os.makedirs("evaluation/results", exist_ok=True)
    output = {
        "dataset": "DocVQA (nielsr/docvqa_1200_examples)",
        "model": MODEL_ID,
        "load_mode": _load_mode,
        "num_samples": NUM_SAMPLES,
        "gemma_anls": round(avg_anls, 3),
        "qwen_vl_max_anls": qwen_anls,
        "reference_layoutlmv3": 0.834,
        "per_question_results": results
    }

    with open("evaluation/results/docvqa_gemma_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to evaluation/results/docvqa_gemma_results.json")
    print(f"{'='*60}")

    return avg_anls


if __name__ == "__main__":
    run_docvqa_gemma_evaluation()