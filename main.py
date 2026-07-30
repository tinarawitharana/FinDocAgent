from agent.graph import build_graph

def main():
    print("Starting FINDOCAGENT...\n")

    #build agent
    agent = build_graph()

    #intial_state
    initial_state = {
        "document_path": "data/samples/bank_statement_anomaly_word.pdf",
        "retrieved_chunks": [],
        "extracted_fields": {},
        "anomalies": [],
        "risk_report": "",
        "iteration_count": 0,
        "task_complete": False
    }

    #run agent
    result = agent.invoke(initial_state)

    print ("\nFINDOCAGENT finished.")
    print(f"Total steps taken: {result['iteration_count']} ")
    print(f"anomalies found: {len(result['anomalies'])} ")

if __name__ == "__main__":
    main()