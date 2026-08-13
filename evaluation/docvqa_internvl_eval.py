"""
Runs InternVL3.5-8B-HF (OpenGVLab/InternVL3_5-8B-HF) locally via HuggingFace
transformers on the same DocVQA sample used for the other comparisons.

Uses the "-HF" variant specifically: it's ported to the standard
AutoModelForImageTextToText class with a normal AutoProcessor, unlike the plain
InternVL3_5-4B checkpoint (custom trust_remote_code model class), which hit an
unfixable 'all_tied_weights_keys' AttributeError against this transformers
version across three different loading-option attempts — a genuine library
version incompatibility in that custom code, not something fixable via kwargs.
The -HF variant sidesteps this by using transformers' native model class.

8B params needs ~16GB in bf16 (over the 8GB budget on this shared,
HAMI-virtualized GPU), so this loads directly in 4-bit rather than trying bf16
first (unlike evaluation/docvqa_gemma_eval.py, where bf16 was a reasonable
first attempt for the smaller 4B model).
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
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_ID = "OpenGVLab/InternVL3_5-8B-HF"
NUM_SAMPLES = 50

_model = None
_processor = None
_load_mode = None


def load_model():
    global _model, _processor, _load_mode

    if _model is not None:
        return

    print(f"[INTERNVL] Loading {MODEL_ID} in 4-bit...")
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=quant_config,
        trust_remote_code=True, device_map={"": 0}
    ).eval()
    _load_mode = "4bit"
    print("[INTERNVL] Loaded in 4-bit.")


def internvl_single_pass(image, question):
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
    ).to(_model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = _model.generate(**inputs, max_new_tokens=50, do_sample=False)

    generated = output[0][input_len:]
    return _processor.decode(generated, skip_special_tokens=True).strip()


def run_docvqa_internvl_evaluation():
    print("=" * 60)
    print(f"DOCVQA EVALUATION - {MODEL_ID} (local)")
    print(f"Running on {NUM_SAMPLES} samples")
    print("=" * 60)

    load_model()

    print("\nLoading DocVQA dataset...")
    ds = load_dataset('nielsr/docvqa_1200_examples', split='train')
    samples = ds.select(range(NUM_SAMPLES))

    internvl_scores = []
    results = []

    for i, example in enumerate(samples):
        question = example['query']['en']
        ground_truths = example['answers']
        image = example['image']

        print(f"\n[{i+1}/{NUM_SAMPLES}] Q: {question[:60]}...")
        print(f"      GT: {ground_truths}")

        try:
            start = time.time()
            prediction = internvl_single_pass(image, question)
            elapsed = time.time() - start

            score = max(anls_score(prediction, gt) for gt in ground_truths)
            internvl_scores.append(score)

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
            internvl_scores.append(0.0)
            results.append({
                "id": i,
                "question": question,
                "ground_truth": ground_truths,
                "prediction": "ERROR",
                "anls": 0.0,
                "time_seconds": 0.0
            })

    avg_anls = sum(internvl_scores) / len(internvl_scores) if internvl_scores else 0

    qwen_anls = None
    qwen_results_path = "evaluation/results/docvqa_results.json"
    if os.path.exists(qwen_results_path):
        with open(qwen_results_path) as f:
            qwen_anls = json.load(f).get("qwen_vl_max_anls")

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(internvl_scores)}")
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
        "internvl_anls": round(avg_anls, 3),
        "qwen_vl_max_anls": qwen_anls,
        "reference_layoutlmv3": 0.834,
        "per_question_results": results
    }

    with open("evaluation/results/docvqa_internvl_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to evaluation/results/docvqa_internvl_results.json")
    print(f"{'='*60}")

    return avg_anls


if __name__ == "__main__":
    run_docvqa_internvl_evaluation()