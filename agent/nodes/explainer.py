from agent.state import AgentState

def explainer_node(state: AgentState) -> AgentState:
    print(f"[EXPLAINER] Generating risk report based on anomalies and extracted fields...")

    anamolies = state["anamolies"]
    fields = state["extracted_fields"]

    risk_level = "HIGH" if anamolies else "LOW"

    report = f"""
--- FINDOCAGENT RISK REPORT ---
Document: {state['document_path']}
Risk Level: {risk_level}

Extracted Fields:
Vendor: {fields.get('vendor', 'Unknown')}
Stated Total: £{fields.get('stated_total', 0)}

Anamolies Detected: {len(anamolies)}
    """

    for a in anamolies:
        report += f"\n- {a['type']}: Stated £{a['stated']} vs Calculated £{a['calculated']} (Difference: £{a['difference']})"

    #dummy
    print(report)
    state['risk_report'] = report
    state['task_complete'] = True

    return state