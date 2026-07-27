from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.retriever import retriever_node
from agent.nodes.extractor import extractor_node
from agent.nodes.anomaly_checker import anomaly_checker_node
from agent.nodes.explainer import explainer_node

def should_continue(state: AgentState) -> str:

    # QA mode — check if answer needs a retry
    if state.get("question"):
        answer = state.get("answer", "").lower()
        negative_phrases = ["not found", "not available", "not specified",
                            "not provided", "does not contain", "does not provide",
                            "insufficient", "not disclosed", "cannot find"]
        is_negative = any(phrase in answer for phrase in negative_phrases)

        if is_negative and state["iteration_count"] < 3:
            return "retriever"
        return END

    # bank statement mode — existing logic unchanged
    if state["iteration_count"] >= 10:
        return "explainer"
    if state["task_complete"]:
        return END
    if state["extracted_fields"]:
        return "anomaly_checker"
    if state["retrieved_chunks"]:
        return "extractor"
    return "retriever"

def build_graph():
    graph = StateGraph(AgentState)

    #add nodes
    graph.add_node("retriever", retriever_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("anomaly_checker", anomaly_checker_node)
    graph.add_node("explainer", explainer_node)

    #set where the agent starts
    graph.set_entry_point("retriever")

    #add edges
    graph.add_edge("retriever", "extractor")
    graph.add_conditional_edges("extractor", should_continue)
    graph.add_edge("anomaly_checker", "explainer")
    graph.add_edge("explainer", END)

    return graph.compile()


    
