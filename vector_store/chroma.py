"""ChromaDB-backed hybrid retrieval: dense (sentence-embedding) + BM25 keyword search
over chunked document text, fused via reciprocal rank fusion for the retriever node."""

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

def get_chroma_client(persist_dir="data/chroma_db"):
    """Opens (or creates) the on-disk ChromaDB collection store."""
    client = chromadb.PersistentClient(path=persist_dir)

    return client

def chunk_text(text, chunk_size=250, overlap=50):
    """Splits text into overlapping word-count chunks for embedding/indexing."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap

    return chunks


def get_embedding_function():
    """Dense embedding function used for both indexing and querying (must match)."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")

def index_document(pages_data, document_name, client= None):
    """Chunks each page's extracted text and (re-)indexes it into a ChromaDB collection.

    Drops any existing collection with the same name first, so re-indexing a document
    is idempotent rather than appending duplicate chunks.
    """
    if client is None:
        client = get_chroma_client()

    #use a default embedding function 
    embedding_function = get_embedding_function()

    #making sure its safe to reun, cause it will throw an error
    try:
        client.delete_collection(name=document_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=document_name, 
        embedding_function=embedding_function
        )
    
    documents = []
    metadatas = []
    ids = []

    for page in pages_data:
        text_chunks = chunk_text(page["full_text"])
        for chunk_index, chunk in enumerate(text_chunks):
            documents.append(chunk)
            metadatas.append({
                "page_number": page["page_number"],
                "document": document_name,
                "chunk_index": chunk_index
            })
            ids.append(f"{document_name}_page_{page['page_number']}_chunk_{chunk_index}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return collection

def search_document(collection, query, top_k=3):
    """Dense semantic search: embeds the query and returns the top_k nearest chunks."""

    instructed_query = f"Represent this sentence for searching relevant passages: {query}"

    results = collection.query(
        query_texts=[instructed_query],
        n_results=top_k
    )

    return results

def bm25_search(collection, query, top_k=10):
    """Sparse keyword search over the same collection's chunks, returning top_k page numbers.

    Rebuilds the BM25 index from the full collection on every call rather than persisting
    it, since collections here are small (single-document) and re-indexing is cheap.
    """
    data = collection.get(include=["documents", "metadatas"])
    tokenized_corpus = [doc.lower().split() for doc in data["documents"]]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [data["metadatas"][i]["page_number"] for i in ranked_indices]

def reciprocal_rank_fusion(rank_lists, k=60):
    """Merges multiple ranked page-number lists (dense + BM25) into one fused ranking."""
    scores = {}
    for ranked_pages in rank_lists:
        for rank, page in enumerate(ranked_pages):
            scores[page] = scores.get(page, 0) + 1 / (k+rank+1)
    return sorted(scores, key=scores.get, reverse=True)


if __name__ == "__main__":

    from document_parser.parser import extract_text_with_positions

    #quick manual smoke test
    test_pdf = "data/samples/bank_statement_clean_word.pdf"

    print ("Extracting text with positions...")
    pages_data = extract_text_with_positions(test_pdf)

    print("\nIndexing document into ChromaDB...")
    collection = index_document(pages_data, document_name="bank_statement_clean_word")
    print(f"Indexed {len(pages_data)} pages")
                                
    print("\nSearching...")
    query = "what is the total purchases amount?"
    results = search_document(collection, query)

    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult {i+1} (page {results['metadatas'][0][i]['page_number']}):")
        print(doc[:300])