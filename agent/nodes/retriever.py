from agent.state import AgentState

def retriever_node(state: AgentState) -> AgentState:
    print(f"[RETRIEVER] Searching document: {state['document_path']}")
    print(f"[RETRIEVER] Step {state['iteration_count'] + 1}")

    #dummy
    state['retrieved_chunks'] = ["Dummy chunk: Invoice total £1,520"]
    state['iteration_count'] += 1

    return state