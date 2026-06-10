from agent.graph import build_graph

def main():
    print("Starting FINDOCAGENT...\n")

    #build agent
    agent = build_graph()

    #intial_state
    initial_state = {
        "document_path": "data/samples/test_invoice.pdf",
        "retrieved_chunks": [],
        "extracted_fields": {},
        "anamolies": [],
        "risk_report": "",
        "iteration_count": 0,
        "task_complete": False
    }

    #run agent
    result = agent.invoke(initial_state)

    print ("\nFINDOCAGENT finished.")
    print(f"Total steps taken: {result['iteration_count']} ")
    print(f"Anamolies found: {len(result['anamolies'])} ")

if __name__ == "__main__":
    main()