"""
Runs the full multi-tool agent (Retriever -> Extractor -> Anomaly Checker ->
Explainer, via the compiled LangGraph in agent/graph.py) end-to-end on the
labeled bank-statement set, and reports anomaly-detection precision/recall/F1
against the known ground truth.

This is the first evaluation to invoke build_graph().invoke(...) directly for
bank-statement mode rather than calling extract_fields_from_document() and
compute_anomaly_features() directly (as evaluation/shap_anomaly_eval.py and
evaluation/validate_synthetic_anomalies.py do) — so it also validates that the
anomaly_checker/explainer portion of the graph runs without error.
"""
import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import build_graph

MATH_THRESHOLD = 0.01  # matches agent/nodes/anomaly_checker.py's own detection threshold


def load_ground_truth():
    documents = []
    with open("data/samples/synthetic/ground_truth.json") as f:
        documents.extend(json.load(f))
    with open("data/samples/bank_statements_ground_truth.json") as f:
        documents.extend(json.load(f))
    return documents


def make_initial_state(document_path):
    return {
        "document_path": document_path,
        "question": "",
        "retrieved_chunks": [],
        "extracted_fields": {},
        "answer": "",
        "anomalies": [],
        "risk_report": "",
        "iteration_count": 0,
        "task_complete": False,
    }


def prf(rows, expected_key, predicted_key):
    tp = sum(1 for r in rows if r[expected_key] and r[predicted_key])
    fp = sum(1 for r in rows if not r[expected_key] and r[predicted_key])
    fn = sum(1 for r in rows if r[expected_key] and not r[predicted_key])
    tn = sum(1 for r in rows if not r[expected_key] and not r[predicted_key])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
             "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def run_agent_eval():
    print("=" * 60)
    print("BANK STATEMENT MULTI-TOOL AGENT EVALUATION")
    print("Retriever -> Extractor -> Anomaly Checker -> Explainer")
    print("=" * 60)

    ground_truth = load_ground_truth()
    agent = build_graph()

    rows = []
    for gt in ground_truth:
        pdf_path = gt["file"]
        print(f"\n[{pdf_path}]")

        expected_math = abs(gt.get("math_discrepancy", 0)) > MATH_THRESHOLD
        expected_date = bool(gt.get("date_swap_applied", False))

        try:
            start = time.time()
            final_state = agent.invoke(make_initial_state(pdf_path))
            elapsed = time.time() - start

            anomalies = final_state.get("anomalies", [])
            predicted_math = any(a["type"] == "math_discrepancy" for a in anomalies)
            predicted_date = any(a["type"] == "date_ordering" for a in anomalies)
            error = None

        except Exception as e:
            print(f"  ERROR: {e}")
            anomalies = []
            predicted_math = False
            predicted_date = False
            elapsed = 0.0
            error = str(e)

        print(f"  Expected math anomaly: {expected_math}  |  Predicted: {predicted_math}")
        print(f"  Expected date anomaly: {expected_date}  |  Predicted: {predicted_date}")
        print(f"  ({elapsed:.1f}s)")

        rows.append({
            "file": pdf_path,
            "expected_math": expected_math,
            "predicted_math": predicted_math,
            "expected_date": expected_date,
            "predicted_date": predicted_date,
            "anomalies": anomalies,
            "time_seconds": round(elapsed, 2),
            "error": error,
        })

    for r in rows:
        r["expected_any"] = r["expected_math"] or r["expected_date"]
        r["predicted_any"] = r["predicted_math"] or r["predicted_date"]

    math_metrics = prf(rows, "expected_math", "predicted_math")
    date_metrics = prf(rows, "expected_date", "predicted_date")
    any_metrics = prf(rows, "expected_any", "predicted_any")

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Documents evaluated: {len(rows)}")
    print(f"Math-discrepancy detection — P: {math_metrics['precision']:.3f}  R: {math_metrics['recall']:.3f}  "
          f"F1: {math_metrics['f1']:.3f}  (TP={math_metrics['tp']} FP={math_metrics['fp']} "
          f"FN={math_metrics['fn']} TN={math_metrics['tn']})")
    print(f"Date-ordering detection    — P: {date_metrics['precision']:.3f}  R: {date_metrics['recall']:.3f}  "
          f"F1: {date_metrics['f1']:.3f}  (TP={date_metrics['tp']} FP={date_metrics['fp']} "
          f"FN={date_metrics['fn']} TN={date_metrics['tn']})")
    print(f"Any-anomaly detection      — P: {any_metrics['precision']:.3f}  R: {any_metrics['recall']:.3f}  "
          f"F1: {any_metrics['f1']:.3f}  (TP={any_metrics['tp']} FP={any_metrics['fp']} "
          f"FN={any_metrics['fn']} TN={any_metrics['tn']})")

    os.makedirs("evaluation/results", exist_ok=True)
    output = {
        "num_documents": len(rows),
        "math_discrepancy_metrics": math_metrics,
        "date_ordering_metrics": date_metrics,
        "any_anomaly_metrics": any_metrics,
        "per_document": rows,
    }
    with open("evaluation/results/bank_statement_agent_eval.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\nSaved to evaluation/results/bank_statement_agent_eval.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_agent_eval()