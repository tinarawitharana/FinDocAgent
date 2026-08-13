"""Anomaly checker node: rule-based checks over the extractor's structured fields.

Two independent checks feed into the risk report: whether the stated total reconciles
with opening balance + line items, and whether line-item dates are chronological.
compute_anomaly_features/anomaly_score also back the SHAP explainability analysis in
evaluation/shap_anomaly_eval.py, which is why the two feature signals are kept separate
and normalized to [0, 1] rather than folded straight into a single detector.
"""

from agent.state import AgentState


def _signed_amount(item):
    """Returns a line item's amount signed by type: negative for debits, positive for credits."""
    amount = item.get("amount", 0)
    item_type = (item.get("type") or "").strip().lower()
    if item_type == "debit":
        return -abs(amount)
    elif item_type == "credit":
        return abs(amount)
    return amount


def compute_anomaly_features(fields):
    """Derives normalized anomaly signals from extracted fields.

    math_feature: how far the stated total is from opening_balance + sum(line items),
    as a fraction of the stated total (capped at 1.0).
    date_feature: fraction of adjacent line-item pairs that are out of chronological order.
    """
    stated = fields.get("stated_total", 0)
    opening_balance = fields.get("opening_balance", 0)
    calculated = opening_balance + sum(_signed_amount(item) for item in fields.get("line_items", []))
    difference = abs(stated) - abs(calculated)
    math_feature = min(abs(difference) / max(abs(stated), 1), 1.0)

    dates = [item.get("date") for item in fields.get("line_items", []) if item.get("date")]
    out_of_order_count = sum(1 for i in range(1, len(dates)) if dates[i] < dates[i - 1])
    date_feature = min(out_of_order_count / max(len(dates) - 1, 1), 1.0)

    return {
        "stated": stated,
        "calculated": calculated,
        "difference": difference,
        "out_of_order_count": out_of_order_count,
        "math_feature": math_feature,
        "date_feature": date_feature
    }


def anomaly_score(features):
    """Combines math and date anomaly features into a single risk score (equal weighting)."""
    return 0.5 * features["math_feature"] + 0.5 * features["date_feature"]


def anomaly_checker_node(state: AgentState) -> AgentState:
    print(f"[ANOMALY CHECKER] Checking for anomalies in extracted fields...")

    fields = state["extracted_fields"]
    anomalies = []

    if fields:
        features = compute_anomaly_features(fields)

        #rule - does the math add up
        if abs(features["difference"]) > 0.01:
            anomalies.append({
                "type": "math_discrepancy",
                "stated": features["stated"],
                "calculated": features["calculated"],
                "difference": features["difference"]
            })

            print(f"[ANOMALY CHECKER] Anomaly detected: £{features['difference']}")
        else:
            print(f"[ANOMALY CHECKER] No anomalies detected. Stated total matches calculated total.")

        #rule - are line item dates in chronological order
        if features["out_of_order_count"] > 0:
            anomalies.append({
                "type": "date_ordering",
                "out_of_order_count": features["out_of_order_count"]
            })

            print(f"[ANOMALY CHECKER] Anomaly detected: {features['out_of_order_count']} out-of-order date(s)")
        else:
            print(f"[ANOMALY CHECKER] No anomalies detected. Dates are in chronological order.")


    state['anomalies'] = anomalies

    return state
