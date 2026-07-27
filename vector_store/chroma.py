import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

def get_chroma_client(persist_dir="data/chroma_db"):

    client = chromadb.PersistentClient(path=persist_dir)

    return client

def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap

    return chunks


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")

def index_document(pages_data, document_name, client= None):

    #chunks the extracted page text and indexes it into the ChromaDB vector store

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

    #searches the ChromaDB vector store for the query and returns the top_k results

    instructed_query = f"Represent this sentence for searching relevant passages: {query}"

    results = collection.query(
        query_texts=[instructed_query],
        n_results=top_k
    )

    return results

def bm25_search(collection, query, top_k=10):
    data = collection.get(include=["documents", "metadatas"])
    tokenized_corpus = [doc.lower().split() for doc in data["documents"]]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [data["metadatas"][i]["page_number"] for i in ranked_indices]

def reciprocal_rank_fusion(rank_lists, k=60):
    scores = {}
    for ranked_pages in rank_lists:
        for rank, page in enumerate(ranked_pages):
            scores[page] = scores.get(page, 0) + 1 / (k+rank+1)
    return sorted(scores, key=scores.get, reverse=True)


if __name__ == "__main__":

    from document_parser.parser import extract_text_with_positions

    #quick test
    test_pdf = "data/samples/bank_statement_clean_word.pdf"
    from document_parser.parser import extract_text_with_positions

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