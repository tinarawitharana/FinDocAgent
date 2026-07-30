import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap
from models.qwen import extract_fields_from_document
from agent.nodes.anomaly_checker import compute_anomaly_features, anomaly_score

BANK_STATEMENT_PDFS = [
    "data/samples/bank_statement_clean.pdf",
    "data/samples/bank_statement_clean_word.pdf",
    "data/samples/bank_statement_anomaly.pdf",
    "data/samples/bank_statement_anomaly_word.pdf",
    "data/samples/hsbc1.pdf",
    "data/samples/hsbc_anomaly.pdf",
]

def score_fn(X):
    return np.array([
        anomaly_score({"math_feature": row[0], "date_feature": row[1]})
        for row in X
    ])

def run_shap_eval():
    background = np.zeros((1, 2))  # "clean document" baseline: no anomalies at all
    explainer = shap.KernelExplainer(score_fn, background)

    results = []

    for pdf_path in BANK_STATEMENT_PDFS:
        print(f"\nProcessing {pdf_path}...")
        fields = extract_fields_from_document(pdf_path)
        features = compute_anomaly_features(fields)

        X = np.array([[features["math_feature"], features["date_feature"]]])
        shap_values = explainer.shap_values(X, silent=True)[0]

        score = anomaly_score(features)
        expected_math = 0.5 * features["math_feature"]
        expected_date = 0.5 * features["date_feature"]

        print(f"  Composite anomaly score: {score:.3f}")
        print(f"  SHAP - math_feature: {shap_values[0]:.3f} (expected {expected_math:.3f})")
        print(f"  SHAP - date_feature: {shap_values[1]:.3f} (expected {expected_date:.3f})")

        results.append({
            "pdf": pdf_path,
            "math_feature": features["math_feature"],
            "date_feature": features["date_feature"],
            "composite_score": score,
            "shap_math_feature": float(shap_values[0]),
            "shap_date_feature": float(shap_values[1])
        })

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/shap_anomaly_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to evaluation/results/shap_anomaly_results.json")

if __name__ == "__main__":
    run_shap_eval()
