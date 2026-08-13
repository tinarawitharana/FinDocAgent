"""
Runs SmolVLM2-2.2B-Instruct (HuggingFaceTB/SmolVLM2-2.2B-Instruct) locally via
HuggingFace transformers on the same DocVQA sample used for the other
comparisons. Chosen after InternVL3.5 hit both an unfixable custom-code
library incompatibility (4B variant) and a hard disk-space wall (8B-HF
variant, ~16GB download on a 25GB total allocation already carrying the
project's other required model caches) — SmolVLM2 uses the native
AutoModelForImageTextToText class (no trust_remote_code), is ungated
(Apache 2.0), and at 2.2B params is a much smaller download, minimizing risk
of repeating either of those failure modes.
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

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
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
        print(f"[SMOLVLM] Loading {MODEL_ID} in bf16...")
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda").eval()
        _load_mode = "bf16"
        print("[SMOLVLM] Loaded in bf16.")
    except torch.cuda.OutOfMemoryError:
        print("[SMOLVLM] bf16 OOM'd, falling back to 4-bit quantization...")
        torch.cuda.empty_cache()
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, quantization_config=quant_config, device_map={"": 0}
        ).eval()
        _load_mode = "4bit"
        print("[SMOLVLM] Loaded in 4-bit.")


def smolvlm_single_pass(image, question):
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


def run_docvqa_smolvlm_evaluation():
    print("=" * 60)
    print(f"DOCVQA EVALUATION - {MODEL_ID} (local)")
    print(f"Running on {NUM_SAMPLES} samples")
    print("=" * 60)

    load_model()

    print("\nLoading DocVQA dataset...")
    ds = load_dataset('nielsr/docvqa_1200_examples', split='train')
    samples = ds.select(range(NUM_SAMPLES))

    smolvlm_scores = []
    results = []

    for i, example in enumerate(samples):
        question = example['query']['en']
        ground_truths = example['answers']
        image = example['image']

        print(f"\n[{i+1}/{NUM_SAMPLES}] Q: {question[:60]}...")
        print(f"      GT: {ground_truths}")

        try:
            start = time.time()
            prediction = smolvlm_single_pass(image, question)
            elapsed = time.time() - start

            score = max(anls_score(prediction, gt) for gt in ground_truths)
            smolvlm_scores.append(score)

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
            smolvlm_scores.append(0.0)
            results.append({
                "id": i,
                "question": question,
                "ground_truth": ground_truths,
                "prediction": "ERROR",
                "anls": 0.0,
                "time_seconds": 0.0
            })

    avg_anls = sum(smolvlm_scores) / len(smolvlm_scores) if smolvlm_scores else 0

    qwen_anls = None
    qwen_results_path = "evaluation/results/docvqa_results.json"
    if os.path.exists(qwen_results_path):
        with open(qwen_results_path) as f:
            qwen_anls = json.load(f).get("qwen_vl_max_anls")

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(smolvlm_scores)}")
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
        "smolvlm_anls": round(avg_anls, 3),
        "qwen_vl_max_anls": qwen_anls,
        "reference_layoutlmv3": 0.834,
        "per_question_results": results
    }

    with open("evaluation/results/docvqa_smolvlm_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to evaluation/results/docvqa_smolvlm_results.json")
    print(f"{'='*60}")

    return avg_anls


if __name__ == "__main__":
    run_docvqa_smolvlm_evaluation()