# NASA ARSET Technical Document Engine (Grounded RAG)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/0datadan/nasa-rag-engine/main/app.py)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%203.5%20Flash%20Lite-green.svg)](https://ai.google.dev/)

An enterprise-grade, retrieval-augmented generation (RAG) system engineered to query, synthesize, and audit dense technical course materials from NASA's Applied Remote Sensing Training (ARSET) program.

Built with strict boundary guardrails, dual-column source traceability, and automated cold-start cloud vector indexing.

---

## System Architecture

[ NASA ARSET PDFs ]
│
▼ (PyPDF / Slicing)
[ Sliding-Window Chunking (600 chars, 120 overlap) ]
│
▼ (all-MiniLM-L6-v2)
[ ChromaDB Vector Store ] ── (Cosine Similarity, k=12) ──┐
│
[ Technical User Query ] ───────────────────────────────┤
▼
[ Context Injection & Strict Guardrails ]
│
▼ (gemini-3.5-flash-lite)
[ Grounded Answer + Slide Excerpt Audit ]
---

## Key Engineering Decisions

### 1. Zero-Hallucination Grounding & Refusal Guardrails
Technical remote sensing data (e.g., aerosol optical depth, orbital mechanics, trace gas spectrometry) cannot tolerate generative hallucination. 
- **System Instruction:** Constrains generation strictly to retrieved context passages.
- **Out-of-Domain Detection:** Explicit refusal trigger (`"I cannot answer this based on the provided NASA ARSET course materials."`) whenever input falls outside indexed slides.
- **Audit Traceability:** Side-by-side UI audit showing exact document names, page/slide numbers, and raw cosine match excerpts.

### 2. Retrieval Depth Parameter Tuning ($k=6 \rightarrow k=12$)
Cross-cutting queries (such as comparing Geostationary Earth Orbit [GEO] vs. Low Earth Orbit [LEO] sensor capabilities) require drawing facts across disparate slide decks and modules. 
- Initial testing with bi-encoder cosine similarity at $k=6$ captured general introductory passages but omitted secondary orbital definitions.
- Expanding baseline retrieval depth to **$k=12$** guaranteed full conceptual recall across multi-document boundaries while staying well within the context window limits.

### 3. Ephemeral Cloud Cold-Start Ingestion
Deploying RAG systems to containerized cloud platforms (such as Streamlit Community Cloud) typically causes vector store failure because `.chromadb/` directories are excluded via `.gitignore`.
- Designed an automated cold-start hook in `app.py` that detects empty collections on boot.
- Triggers an automated, batched ingestion pipeline over source PDFs directly in ephemeral storage, ensuring zero-configuration cloud spins.

### 4. Production Inference Tier Stabilization
Standardized all inference calls across CLI and UI on **`gemini-3.5-flash-lite`** with zero temperature ($T=0.0$), eliminating transient server-demand capacity issues (503s) and ensuring consistent token pricing and deterministic outputs.

---

## Repository Structure

```text
nasa-rag-engine/
├── data/                       # NASA ARSET slide deck source PDFs
├── app.py                      # Production Streamlit UI with dual-column audit view
├── query_engine.py             # CLI query runner and context retrieval module
├── build_index.py              # Standalone ingestion and embedding pipeline
├── ingest.py                   # PDF text extraction and sliding-window chunking
├── requirements.txt            # Locked runtime dependencies
└── README.md                   # System documentation and architecture breakdown
Local Setup & Quickstart1. Clone & Set Up EnvironmentBashgit clone [https://github.com/0DataDan/nasa-rag-engine.git](https://github.com/0DataDan/nasa-rag-engine.git)
cd nasa-rag-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2. Configure Environment KeysCreate a .env file in the root directory:BashGEMINI_API_KEY="your_actual_gemini_api_key"
3. Build Vector StoreBashpython build_index.py
4. Run the EngineCommand Line Interface:Bashpython query_engine.py "What instruments on Terra and Aqua measure Aerosol Optical Depth?"
Interactive Web Application:Bashstreamlit run app.py
Example Queries to TestQuery CategorySample PromptExpected System BehaviorSpectral Analysis"What wavelengths of light are absorbed by healthy vegetation?"Retrieves visible/NIR slide passages with chlorophyll absorption bands.Mission Specs"What instruments measure Aerosol Optical Depth?"Identifies MODIS (Aqua/Terra) and MAIAC algorithms with slide citations.Orbital Mechanics"Compare GEO vs LEO satellites for air quality tracking."Synthesizes temporal frequency vs. spatial resolution tradeoffs across modules.Negative Boundary"How do lithium-ion batteries function in electric vehicles?"Triggers refusal guardrail without generating synthetic facts.