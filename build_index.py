import os
import chromadb
from chromadb.utils import embedding_functions
from ingest import process_pdf_folder

def build_vector_store():
    # 1. Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "data")
    db_folder = os.path.join(current_dir, "chroma_db")

    print("[*] Slicing documents into structured chunks...")
    chunks = process_pdf_folder(data_folder, chunk_size=600, chunk_overlap=120)

    if not chunks:
        print("[!] No chunks generated. Aborting.")
        return

    print(f"\n[*] Initializing local vector database in '{db_folder}'...")
    
    # 2. Set up local persistent storage
    client = chromadb.PersistentClient(path=db_folder)

    # 3. Define the embedding model (runs locally on your machine)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Create or load the collection
    collection = client.get_or_create_collection(
        name="nasa_remote_sensing",
        embedding_function=embedding_fn
    )

    # 5. Prepare lists for bulk insertion into ChromaDB
    documents = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"[*] Generating embeddings and indexing {len(chunks)} chunks into ChromaDB...")
    print("    (This may take 30-60 seconds on the first run as the embedding model initializes)...")

    # Chroma handles embedding calculation automatically during add()
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    print(f"[+] Success! {collection.count()} chunks are now indexed and stored persistently.")

    # 6. Run an immediate test search to prove retrieval works
    test_query = "What instruments or satellites measure aerosol optical depth?"
    print(f"\n[*] Running sanity test search with query:")
    print(f"    '{test_query}'")

    results = collection.query(
        query_texts=[test_query],
        n_results=2
    )

    print("\n--- TOP RETRIEVED RESULT ---")
    print(f"Source   : {results['metadatas'][0][0]['source']} (Page {results['metadatas'][0][0]['page']})")
    print(f"Chunk ID : {results['ids'][0][0]}")
    print(f"Snippet  :\n\"{results['documents'][0][0][:280]}...\"")
    print("----------------------------\n")

if __name__ == "__main__":
    build_vector_store()