import os
from dotenv import load_dotenv
load_dotenv()

import requests
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

st.set_page_config(
    page_title="NASA ARSET Retrieval Engine",
    page_icon="🛰️",
    layout="wide"
)

# 1. Connect to ChromaDB with automatic cloud cold-start indexing
@st.cache_resource
def load_chroma():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_folder = os.path.join(current_dir, "chroma_db")
    data_folder = os.path.join(current_dir, "data")

    client = chromadb.PersistentClient(path=db_folder)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    col = client.get_or_create_collection(
        name="nasa_remote_sensing",
        embedding_function=embedding_fn
    )

    # Cold-start: if running on Cloud and DB is empty, index files in data/
    if col.count() == 0 and os.path.exists(data_folder):
        from ingest import process_pdf_folder
        chunks = process_pdf_folder(data_folder, chunk_size=600, chunk_overlap=120)
        if chunks:
            docs = [c["text"] for c in chunks]
            ids = [c["chunk_id"] for c in chunks]
            metas = [c["metadata"] for c in chunks]
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                col.upsert(
                    documents=docs[i:i + batch_size],
                    ids=ids[i:i + batch_size],
                    metadatas=metas[i:i + batch_size]
                )

    return col

collection = load_chroma()
total_chunks = collection.count()

# Helper function: Cross-platform Python HTTP transport
def call_gemini(api_key: str, system_instruction: str, user_prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": f"{system_instruction}\n\n{user_prompt}"}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1000}
    }
    headers = {"Content-Type": "application/json"}
    
    resp = requests.post(url, json=payload, headers=headers, timeout=45)
    data = resp.json()

    if resp.status_code != 200 or "error" in data:
        err_msg = data.get("error", {}).get("message", resp.text)
        raise Exception(f"API Error ({resp.status_code}): {err_msg}")

    return data["candidates"][0]["content"]["parts"][0]["text"]

# 2. UI Layout & Header
st.title("🛰️ NASA ARSET Technical Retrieval Engine")
st.caption("A Grounded Retrieval-Augmented Generation (RAG) System indexed on NASA ARSET Remote Sensing & Air Quality Curricula.")

if total_chunks == 0:
    st.warning("⚠️ The vector database is currently empty. Please run `python build_index.py` in your terminal to index the NASA ARSET course materials.")

# Sidebar Configuration
with st.sidebar:
    st.header("Engine Configuration")
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        api_key_input = st.text_input("Enter Gemini API Key:", type="password")
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input
            active_api_key = api_key_input
            st.success("API Key loaded!")
        else:
            active_api_key = ""
    else:
        active_api_key = api_key
        st.success("API Key detected from environment.")

    st.divider()
    ttop_k = st.sidebar.slider("Number of retrieved chunks", min_value=2, max_value=12, value=8)
    st.info(f"Total Chunks in Vector Store: **{total_chunks}**")

# Quick Test Buttons
st.write("##### Quick Test Queries:")
cols = st.columns(3)

if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

if cols[0].button("Aerosol Optical Depth"):
    st.session_state["user_query"] = "What is Aerosol Optical Depth and how do satellites measure it?"
if cols[1].button("Orbit Tradeoffs"):
    st.session_state.query_text = "Compare geostationary (GEO) vs polar orbiting Low Earth Orbit (LEO) satellites for air quality monitoring temporal and spatial coverage."
if cols[2].button("Vegetation Wavelengths"):
    st.session_state["user_query"] = "What wavelengths of light are absorbed by healthy vegetation?"

# Query Input Form
with st.form("query_form"):
    query = st.text_input(
        "Ask a technical remote sensing question:",
        value=st.session_state["user_query"],
        placeholder="e.g., How does the TEMPO mission monitor atmospheric pollutants?"
    )
    submitted = st.form_submit_button("Search & Synthesize", type="primary")

# Execution Workflow
if submitted and query.strip():
    st.session_state["user_query"] = query

    if not active_api_key:
        st.error("Please provide a valid Gemini API Key to run synthesis.")
    elif total_chunks == 0:
        st.error("Vector database contains 0 chunks. Run `python build_index.py` first.")
    else:
        status_box = st.status("Processing query...", expanded=True)
        try:
            status_box.write("🔍 Searching vector database (ChromaDB)...")
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]

            status_box.write(f"📄 Retrieved {len(docs)} relevant passages from NASA notes.")

            context_blocks = []
            for i in range(len(docs)):
                block = f"[{metas[i]['source']} p.{metas[i]['page']}]: {docs[i].strip()}"
                context_blocks.append(block)
            joined_context = "\n\n".join(context_blocks)

            status_box.write("🤖 Synthesizing answer via Gemini 3.8 Flash...")

            system_instruction = (
                "You are an expert NASA Remote Sensing Operations Assistant. "
                "Answer the user's question concisely using ONLY the provided excerpts. "
                "Cite sources using [Doc p.X]. If the answer is not present, state: "
                "'I cannot answer this based on the provided NASA ARSET course materials.'"
            )

            user_prompt = f"EXCERPTS:\n{joined_context}\n\nQUESTION: {query}"
            answer_text = call_gemini(active_api_key, system_instruction, user_prompt)

            status_box.update(label="Complete!", state="complete", expanded=False)

            # Two-Column Layout
            col_ans, col_sources = st.columns([3, 2])

            with col_ans:
                st.subheader("Grounded Synthesis")
                st.markdown(answer_text)

            with col_sources:
                st.subheader("Source Excerpt Audit")
                st.caption("Raw passages retrieved from ChromaDB:")
                for i in range(len(docs)):
                    with st.expander(f"Excerpt {i+1}: {metas[i]['source']} (Page {metas[i]['page']})"):
                        st.caption(f"Chunk ID: `{ids[i]}`")
                        st.write(docs[i])

        except Exception as e:
            status_box.update(label="Error Occurred", state="error", expanded=True)
            st.error(f"Execution Error: {e}")