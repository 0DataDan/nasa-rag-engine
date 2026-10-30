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


   )
  model_name="all-MiniLM-L6-v2"
.Selectio.Selectio.Selectio.Selectio.Selesing",
.Selectio.Selectio.unction.Selectio.Selectio.unction.Selectio.Selectio.unction.Sen chun.Selectio.Selectio.unction.Selectio.Selectio.unct me.Selectio.Selectio.unction.Selectio.Selectio.unctint(f"[.Selectio.Selectio.unction.Selecexing {l.Selectio.Selectio.unction.Selectio.Sele  print("    (Initial run may take 30-60 seconds while embedding weights are verified)...")

    # Use upsert to safely ov  write existing    # Use upsert to safely ov  write existing    # Use uon.up    # Use upsert to safely ov  write existing    # Use upsert to safely ov atas
    )

    print(f"[+] Success! {collection.count()} chunks are now indexed and stored persistently.")

    test_query = "What inst    test_query = "What inst    test_query = "What inst    test_query[*] Running sanity test search with query:")
    print(f"    '{    print(f"    '{    print(f"    '{    print(f"    '{    print(f"  [test_que    print(f"    '{    pri    )

    if results and results.get("documents") and res    if results and results.get("documents- TOP RETRIEVED RESULT ---")
        print(f"Source   : {results['metadatas'][0][0]['source']} (Page {results['metadatas'][0][0]['page']})")
        print(f"Chunk ID : {results['ids'][0][0]}")
        print(f"Snippet  :\n\"{results['documents'][0][0][:280]}...\"")
        print("----------------------------\n")

if __name__ == "__main__":
    build_vector_store()
