import os
import chromadb
from chromadb.utils import embedding_functions
from ingest import process_pdf_folder

def build_vector_store():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "data")
    db_folder = os.path.join(current_dir, "chroma_db")

    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
        print(f"[!] Created missing '{data_folder}' folder. Please place NASA ARSET PDF files there and re-run.")
        return

    print("[*] Slicing documents into structured chunks...")
    chunks = process_pdf_folder(data_folder, chunk_size=600, chunk_overlap=120)

    if not chunks:
        print(f"[!] No chunks generated. Verify that PDF files exist inside '{data_folder}'.")
        return

    print(f"\n[*] Initializing local vector database in '{db_folder}'...")
    client = chromadb.PersistentClient(path=db_folder)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="nasa_remote_sensing",
        embedding_function=embedding_fn
    )

    documents = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"[*] Generating embeddings and indexing {len(chunks)} chunks into ChromaDB...")
    print("    (Initial run may take 30-60 seconds while embedding weights are verified)...")

    collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    print(f"[+] Success! {collection.count()} chunks are now indexed and stored persistently.")

    test_query = "What instruments or satellites measure aerosol optical depth?"
    print(f"\n[*] Running sanity test search with query:")
    print(f"    '{test_query}'")

    results = collection.query(
        query_texts=[test_query],
        n_results=2
    )

    if results and results.get("documents") and results["documents"][0]:
        print("\n--- TOP RETRIEVED RESULT ---")
        print(f"Source   : {results['metadatas'][0][0]['source']} (Page {results['metadatas'][0][0]['page']})")
        print(f"Chunk ID : {results['ids'][0][0]}")
        print(f"Snippet  :\n\"{results['documents'][0][0][:280]}...\"")
        print("----------------------------\n")

if __name__ == "__main__":
    build_vector_store()