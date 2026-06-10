from agent.state import AgentState

def extractor_node(state: AgentState) -> AgentState:
    print(f"[EXTRACTOR] Extracting fields from retrieved chunks...")


    #dummy
    state['extracted_fields'] = {
        "vendor": "Acme Ltd",
        "stated_total": 1520,
        "line_items": [
            {"description": "Consulting", "amount": 800},
            {"description": "License", "amount": 620}
        ]
    }
    

    return state