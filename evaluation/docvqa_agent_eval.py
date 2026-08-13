"""Runs the full FinDocAgent LangGraph pipeline (not a direct model call) on the same
DocVQA sample as docvqa_eval.py, to measure whether the retrieval/retry loop improves
on the base-model reference point."""

import os
import sys
import json
import time
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from agent.graph import build_graph
from evaluation.metrics import anls_score

NUM_SAMPLES = 50

def run_docvqa_agent_evaluation():
    print("="*60)
    print("DOCVQA EVALUATION - FinDocAgent (full agent)")
    print(f"Running on {NUM_SAMPLES} samples")
    print("="*60)

    print("\n Loading DocVQA dataset...")
    ds = load_dataset("nielsr/docvqa_1200_examples", split="train")
    samples = ds.select(range(NUM_SAMPLES))

    print("Building agent graph...")
    agent = build_graph()

    agent_scores = []
    results = []

    for i, example in enumerate(samples):
        question = example["query"]["en"]
        ground_truths = example["answers"]
        image = example["image"]

        print(f"\n[{i+1}/{NUM_SAMPLES}] Q: {question[:60]}...")
        print(f"        GT: {ground_truths[0]}")

        #save img to temp 
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, "PNG")
            tmp_path = tmp.name

        try:
            initial_state = {
                "document_path": tmp_path,
                "question": question,
                "retrieved_chunks": [],
                "extracted_fields": {},
                "answer": "",
                "anomalies": [],
                "risk_report": "",
                "iteration_count": 0,
                "task_complete": False
            }

            start = time.time()
            result = agent.invoke(initial_state)
            elapsed = time.time() - start

            prediction = result.get("answer", "")
            score = max(anls_score(prediction, gt) for gt in ground_truths)
            agent_scores.append(score)
            
            print(f"    PredL {prediction}")
            print(f"    ANLS: {score:.3f} ({elapsed:.1f}s)")

            results.append({
                "id": i,
                "question": question,
                "ground_truths": ground_truths,
                "prediction": prediction,
                "anls": round(score, 3),
                "time_seconds": round(elapsed, 2)
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            agent_scores.append(0.0)
            results.append({
                "id": i,
                "question": question,
                "ground_truths": ground_truths,
                "prediction": prediction,
                "anls": 0.0,
                "time_seconds": 0.0
            })
        
        finally:
            os.unlink(tmp_path)

        time.sleep(0.5)

    avg_anls = sum(agent_scores) / len(agent_scores) if agent_scores else 0

    print(f"\n{'='*60}")
    print("Results summary")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(agent_scores)}")
    print(f"FinDocAgent ANLS:   {avg_anls:.3f}")
    print(f"Qwen2-VL single pass ANLS: 0.954 (from docvqa_eval.py)")
    print(f"Reference - LayoutLMv3: 0.834")

    os.makedirs("evaluation/results", exist_ok=True)
    output = {
        "dataset": "DocVQA (nielsr/docvqa_1200_examples)",
        "num_samples": NUM_SAMPLES,
        "findocagent_anls": round(avg_anls, 3),
        "qwen_single_pass_anls": 0.954,
        "reference_layoutlmv3": 0.834,
        "per_question_results": results
    }

    with open("evaluation/results/docvqa_agent_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to evaluation/results/docvqa_agent_results.json")
    print(f"{'='*60}")

    return avg_anls

if __name__ == "__main__":
    run_docvqa_agent_evaluation()