from agent.state import AgentState

def explainer_node(state: AgentState) -> AgentState:
    print(f"[EXPLAINER] Generating risk report based on anomalies and extracted fields...")

    anomalies = state["anomalies"]
    fields = state["extracted_fields"]

    risk_level = "HIGH" if anomalies else "LOW"

    report = f"""
--- FINDOCAGENT RISK REPORT ---
Document: {state['document_path']}
Risk Level: {risk_level}

Extracted Fields:
Vendor: {fields.get('vendor', 'Unknown')}
Stated Total: £{fields.get('stated_total', 0)}

anomalies Detected: {len(anomalies)}
    """

    for a in anomalies:
        report += f"\n- {a['type']}: Stated £{a['stated']} vs Calculated £{a['calculated']} (Difference: £{a['difference']})"

    #dummy
    print(report)
    state['risk_report'] = report
    state['task_complete'] = True

    return state