import os
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

def get_chroma_collection():
    """Connects to the existing persistent ChromaDB store."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_folder = os.path.join(current_dir, "chroma_db")

    client = chromadb.PersistentClient(path=db_folder)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    return client.get_collection(
        name="nasa_remote_sensing",
        embedding_function=embedding_fn
    )

def retrieve_context(query: str, n_results: int = 4):
    """Retrieves the top n most relevant chunks and formats them with citations."""
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    formatted_context_blocks = []
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    for i in range(len(docs)):
        source = metas[i]["source"]
        page = metas[i]["page"]
        chunk_text = docs[i]
        
        block = f"\n{chunk_text}"
        formatted_context_blocks.append(block)

    joined_context = "\n\n---\n\n".join(formatted_context_blocks)
    return joined_context, results

def ask_nasa_engine(query: str):
    """Retrieves relevant context and generates a grounded response using Gemini."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[!] GEMINI_API_KEY environment variable is not set.")
        print("    Run: export GEMINI_API_KEY='your_key_here' in your terminal.")
        return

    # 1. Retrieve the top 4 relevant passages
    context_text, raw_results = retrieve_context(query, n_results=4)

    # 2. Strict system instructions enforcing grounding and citation rules
    system_instruction = """
You are an expert NASA Remote Sensing and Earth Observation Operations Assistant.
Your task is to answer technical questions accurately and concisely using ONLY the provided excerpts.

STRICT GROUNDING PROTOCOL:
1. Base your answer EXCLUSIVELY on the provided context excerpts. Do not infer, extrapolate, or use external knowledge.
2. If the excerpts do not contain sufficient evidence to answer the question, state:
   "I cannot answer this based on the provided NASA ARSET course materials."
3. Every factual claim or technical parameter MUST include an inline citation citing the document and page from the context tag (e.g.,).
4. Maintain a clear, professional technical tone.
"""

    prompt = f"""
CONTEXT EXCERPTS:
{context_text}

USER QUESTION:
{query}
"""

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0  # Set to 0.0 for deterministic, strictly factual output
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
    
    # Allow passing a query from terminal or use a default test query
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "What is Aerosol Optical Depth and how do satellites measure it?"
        
    print(f"[*] Query: '{user_query}'")
    ask_nasa_engine(user_query)