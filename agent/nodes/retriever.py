"""Retriever node: the first step of the agent graph.

Behavior depends on the document/mode:
  - Image input: passed straight through, nothing to retrieve.
  - Bank-statement mode (no question): passed straight through — the extractor reads
    the whole PDF directly via Qwen2-VL rather than retrieving specific chunks.
  - Document-QA/RAG mode (question set): indexes the PDF into ChromaDB on first use,
    then retrieves the most relevant pages via hybrid dense + BM25 search and rasterizes
    them to images for the vision-language extractor.
"""

from agent.state import AgentState
from document_parser.parser import extract_text_with_positions
import os
from pdf2image import convert_from_path
from vector_store.chroma import index_document, get_chroma_client, search_document, get_embedding_function, bm25_search, reciprocal_rank_fusion

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

def retriever_node(state: AgentState) -> AgentState:
    print(f"[RETRIEVER] Searching document: {state['document_path']}")
    print(f"[RETRIEVER] Step {state['iteration_count'] + 1}")

    document_path = state["document_path"]
    ext = os.path.splitext(document_path)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        #img input
        print(f"[RETRIEVER] Image file detected, skipping text retrievel.")
        state["retrieved_chunks"] = [document_path]
        state["iteration_count"] += 1

        return state

    if not state.get("question"):
        #bank-statement / PDF-extraction mode — extractor reads the PDF directly via Qwen2-VL,
        #never touches retrieved_chunks, so there's nothing for retrieval to do here
        print(f"[RETRIEVER] No question set, skipping RAG retrieval (PDF-extraction mode).")
        state["retrieved_chunks"] = [document_path]
        state["iteration_count"] += 1

        return state

    #pdf path
    document_name = os.path.splitext(os.path.basename(document_path))[0]

    #index in chromaDB by resusing existing collection if doc already indexed
    client = get_chroma_client()
    embedding_function = get_embedding_function()

    try:
        collection = client.get_collection(name=document_name, embedding_function=embedding_function)
        if collection.count() == 0:
            raise ValueError("empty collection")
    except Exception:
        pages_data = extract_text_with_positions(document_path)
        collection = index_document(pages_data, document_name=document_name, client=client)



    #seach for key financial info - dummy query for now
    query = state.get("question") or "Total amount due line items transactions"

    # On retries (see should_continue in graph.py), reformulate the query with different
    # framing to surface pages missed on the previous attempt — first retry biases toward
    # tabular/financial-statement pages, later retries toward narrative/text pages.
    if state.get("question") and state["iteration_count"] == 1:
        query = f"financial statement table annual report: {state['question']}"

    elif state.get("question") and state["iteration_count"] >=2:
        query = f"annual report corporate initiatives text: {state['question']}"

    dense_results = search_document(collection, query, top_k=10)
    # get the page numbers of the top-k results
    dense_metadatas = dense_results["metadatas"][0]
    dense_pages = list(dict.fromkeys(m["page_number"] for m in dense_metadatas))  # unique, order preserved

    bm25_pages = bm25_search(collection, query, top_k=10)

    #get pg numbers of top k results, combing dense + keyword search
    page_numbers = reciprocal_rank_fusion([dense_pages, bm25_pages])[:6]

    print(f"[RETRIEVER] Retrieved page numbers: {page_numbers}")


    # convert only those pages to images
    image_paths = []
    os.makedirs("data/page_images", exist_ok=True)
    for page_num in page_numbers:
        imgs = convert_from_path(document_path, first_page=page_num, last_page=page_num, dpi=150)
        image_path = f"data/page_images/{document_name}_page_{page_num}.png"
        imgs[0].save(image_path, "PNG")
        image_paths.append(image_path)

    state["retrieved_chunks"] = image_paths
    state["iteration_count"] += 1

    print(f"[RETRIEVER] Retrieved {len(image_paths)} page images.")


    return state