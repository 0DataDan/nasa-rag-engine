import os
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

def get_chroma_collection():
    """Connects to the persistent ChromaDB store."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_folder = os.path.join(current_dir, "chroma_db")

    if not os.path.exists(db_folder):
        raise FileNotFoundError(
            f"Database directory '{db_folder}' not found. Run 'python build_index.py' first."
        )

    client = chromadb.PersistentClient(path=db_folder)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        return client.get_collection(
            name="nasa_remote_sensing",
            embedding_function=embedding_fn
        )
    except Exception:
        raise ValueError(
            "Collection 'nasa_remote_sensing' not found. Run 'python build_index.py' to generate the index."
        )

def retrieve_context(query: str, n_results: int = 6):
    """Retrieves top-N relevant chunks and formats them with source citations."""
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    formatted_context_blocks = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    for i in range(len(docs)):
        source = metas[i]["source"]
        page = metas[i]["page"]
        chunk_text = docs[i].strip()
        block = f"[{source} p.{page}]: {chunk_text}"
        formatted_context_blocks.append(block)

    joined_context = "\n\n---\n\n".join(formatted_context_blocks)
    return joined_context, results

def ask_nasa_engine(query: str):
    """Retrieves relevant context and generates a grounded response using Gemini."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[!] GEMINI_API_KEY environment variable is not set.")
        print("    Add it to your .env file: GEMINI_API_KEY='your_key'")
        return

    try:
        context_text, raw_results = retrieve_context(query, n_results=6)
    except Exception as e:
        print(f"[!] Retrieval error: {e}")
        return

    system_instruction = (
        "You are an expert NASA Remote Sensing and Earth Observation Operations Assistant. "
        "Your task is to answer technical questions accurately and concisely using ONLY the provided excerpts.\n\n"
        "STRICT GROUNDING PROTOCOL:\n"
        "1. Base your answer EXCLUSIVELY on the provided context excerpts. Do not infer, extrapolate, or use external knowledge.\n"
        "2. If the excerpts do not contain sufficient evidence to answer the question, state:\n"
        "   'I cannot answer this based on the provided NASA ARSET course materials.'\n"
        "3. Every factual claim or technical parameter MUST include an inline citation citing the document and page (e.g., [NASA_ARSET_Course_Notes2.pdf p.8]).\n"
        "4. Maintain a clear, professional technical tone."
    )

    prompt = f"CONTEXT EXCERPTS:\n{context_text}\n\nUSER QUESTION:\n{query}"

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0
        )
    )

    print("\n==================== GROUNDED ANSWER ====================")
    print(response.text.strip())
    print("=========================================================\n")

    print("--- RETRIEVED SOURCES AUDIT ---")
    for i, meta in enumerate(raw_results["metadatas"][0]):
        print(f"[{i+1}] {meta['source']} (Page {meta['page']}) -> Chunk ID: {raw_results['ids'][0][i]}")
    print("--------------------------------\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "What is Aerosol Optical Depth and how do satellites measure it?"

    print(f"[*] Query: '{user_query}'")
    ask_nasa_engine(user_query)