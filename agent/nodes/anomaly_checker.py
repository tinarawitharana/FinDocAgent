from agent.state import AgentState

def anomaly_checker_node(state: AgentState) -> AgentState:
    print(f"[ANOMALY CHECKER] Checking for anomalies in extracted fields...")

    fields = state["extracted_fields"]
    anamolies = []

    #rule - does the math add up
    if fields:
        stated = fields.get("stated_total", 0)
        calculated = sum(
            item["amount"] for item in fields.get("line_items", [])
        )

        if abs(stated - calculated) > 0.01:
            anamolies.append({
                "type": "math_discrepancy",
                "stated": stated,
                "calculated": calculated,
                "difference": stated - calculated
            })

            print(f"[ANOMALY CHECKER] Anomaly detected: £{stated - calculated}")
        else:
            print(f"[ANOMALY CHECKER] No anomalies detected. Stated total matches calculated total.")


    #dummy
    state['anamolies'] = anamolies

    return state