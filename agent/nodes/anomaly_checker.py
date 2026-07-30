from agent.state import AgentState


def compute_anomaly_features(fields):
    stated = fields.get("stated_total", 0)
    calculated = sum(item["amount"] for item in fields.get("line_items", []))
    difference = stated - calculated
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


    #dummy
    state['anomalies'] = anomalies

    return state
