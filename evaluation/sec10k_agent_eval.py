"""Runs the full FinDocAgent LangGraph pipeline (RAG page retrieval, not the oracle page)
on the same SEC 10-K QA set as sec10k_eval.py, to measure retrieval quality's effect on
answer accuracy."""

import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import build_graph
from evaluation.metrics import anls_score

def run_sec10k_agent_evaluation():
    print("="*60)
    print("SEC 10-K AGENT EVALUATION - FinDocAgent (RAG pipeline)")
    print("="*60)

    with open("data/sec10k_qa.json") as f:
        qa_data = json.load(f)
        #qa_data = qa_data[:6]



    agent = build_graph()
    all_scores = []
    results = []

    for i, item in enumerate(qa_data):
        pdf_path = item["pdf"]
        company = item["company"]
        question = item["question"]
        answers = item["answers"]

        print(f"\n[{i+1}/{len(qa_data)}] {company}")
        print(f"  Q: {question}")
        print(f"  GT: {answers[0]}")

        try:
            initial_state = {
                "document_path": pdf_path,
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
            score = max(anls_score(prediction, gt) for gt in answers)
            all_scores.append(score)

            print(f"  Pred: {prediction}")
            print(f"  ANLS: {score:.3f} ({elapsed:.1f}s)")

            results.append({
                "company": company,
                "question": question,
                "ground_truths": answers,
                "prediction": prediction,
                "anls": round(score, 3)
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            all_scores.append(0.0)

        time.sleep(0.5)

    avg_anls = sum(all_scores) / len(all_scores) if all_scores else 0

    print(f"\n{'='*60}")
    print(f"SEC 10-K RAG AGENT ANLS: {avg_anls:.3f}")
    print(f"Total questions: {len(all_scores)}")
    print(f"{'='*60}")

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/sec10k_agent_results.json", "w") as f:
        json.dump({
            "dataset": "SEC 10-K (RAG agent pipeline)",
            "num_questions": len(all_scores),
            "anls": round(avg_anls, 3),
            "per_question": results
        }, f, indent=2)

    print("Saved to evaluation/results/sec10k_agent_results.json")
    return avg_anls

if __name__ == "__main__":
    run_sec10k_agent_evaluation()
