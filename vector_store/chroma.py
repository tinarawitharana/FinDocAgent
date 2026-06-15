import chromadb
from chromadb.utils import embedding_functions

def get_chroma_client(persist_dir="data/chroma_db"):

    client = chromadb.PersistentClient(path=persist_dir)

    return client

def index_document(pages_data, document_name, client= None):

    #chunks the extracted page text and indexes it into the ChromaDB vector store

    if client is None:
        client = get_chroma_client()

    #use a default embedding function 
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

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
        documents.append(page["full_text"])
        metadatas.append({
            "page_number": page["page_number"],
            "document": document_name
        })
        ids.append(f"{document_name}_page_{page['page_number']}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return collection

def search_document(collection, query, top_k=3):

    #searches the ChromaDB vector store for the query and returns the top_k results

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    return results

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