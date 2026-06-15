from agent.state import AgentState
from document_parser.parser import extract_text_with_positions
from vector_store.chroma import index_document, get_chroma_client, search_document
import os

def retriever_node(state: AgentState) -> AgentState:
    print(f"[RETRIEVER] Searching document: {state['document_path']}")
    print(f"[RETRIEVER] Step {state['iteration_count'] + 1}")

    document_path = state["document_path"]
    document_name = os.path.splitext(os.path.basename(document_path))[0]

    #extract text from pdf
    pages_data = extract_text_with_positions(document_path)

    #index in chromadb
    client = get_chroma_client()
    collection = index_document(pages_data, document_name=document_name, client=client)

    #seach for key financial info - dummy query for now
    query = "Total amount due line items transactions"
    results = search_document(collection, query, top_k=2)

    #store retrieved chunks in state for next node to process
    retrieved_chunks = results["documents"][0]
    state["retrieved_chunks"] = retrieved_chunks
    state["iteration_count"] += 1

    print(f"[RETRIEVER] Retrieved {len(retrieved_chunks)} chunks from document.")

    return state