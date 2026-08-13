"""RQ3 explainability: generates and saves attention-map visualizations (via
models/attention_map.py) for a handful of DocVQA examples using the local Qwen2-VL-2B model."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from datasets import load_dataset
from models.attention_map import generate_attention_map

NUM_EXAMPLES = 3

def run_attention_evaluation():
    print("=" * 60)
    print("ATTENTION MAP EVALUATION — Qwen2-VL-2B (local)")
    print("=" * 60)

    print("\nLoading DocVQA dataset...")
    ds = load_dataset("nielsr/docvqa_1200_examples", split="train")

    os.makedirs("evaluation/results/attention_maps", exist_ok=True)

    for i in range(NUM_EXAMPLES):
        example = ds[i]
        image = example["image"]
        question = example["query"]["en"]
        ground_truth = example["answers"][0]

        print(f"\n[{i+1}/{NUM_EXAMPLES}]")
        print(f"  Q: {question}")
        print(f"  GT: {ground_truth}")

        save_path = f"evaluation/results/attention_maps/example_{i+1}.png"

        try:
            answer, fig = generate_attention_map(image, question, save_path=save_path)
            print(f"  Pred: {answer}")
            if fig:
                plt.close(fig)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Figures saved to evaluation/results/attention_maps/")

if __name__ == "__main__":
    run_attention_evaluation()