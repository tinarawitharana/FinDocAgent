"""Sanity check for the synthetic bank-statement test set: confirms the anomaly_checker's
math/date features actually fire on the documents deliberately constructed to contain
those errors (data/samples/synthetic/, labeled in ground_truth.json)."""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qwen import extract_fields_from_document
from agent.nodes.anomaly_checker import compute_anomaly_features


def run_validation():
    with open("data/samples/synthetic/ground_truth.json") as f:
        ground_truth = json.load(f)

    print(f"{'File':<35} {'Expected math':>14} {'Detected math':>14} {'Expected date':>14} {'Detected date':>14}")

    for gt in ground_truth:
        pdf_path = gt["file"]
        fields = extract_fields_from_document(pdf_path)
        features = compute_anomaly_features(fields)

        expected_math = gt["math_discrepancy"]
        detected_math = features["difference"]

        expected_date = gt["date_swap_applied"]
        detected_date = features["out_of_order_count"] > 0

        print(f"{os.path.basename(pdf_path):<35} {expected_math:>14.2f} {detected_math:>14.2f} {str(expected_date):>14} {str(detected_date):>14}")


if __name__ == "__main__":
    run_validation()
