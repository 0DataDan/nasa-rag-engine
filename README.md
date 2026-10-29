NASA ARSET Earth Observation Local RAG Engine

A privacy-centric, strictly grounded Retrieval-Augmented Generation (RAG) system engineered to query and audit NASA Applied Remote Sciences Training (ARSET) course notes across atmospheric remote sensing, aerosol optical depth (AOD), and trace gas monitoring.

Built with an embedded vector store, local transformer-based dense embeddings, a custom REST synthesis client, and an auditable Streamlit evaluation interface.

System Architecture
Plaintext
[ NASA ARSET PDFs ]
        │
        ▼ (pypdf sliding-window: 500 chars / 100 overlap)
[ Chunked Passages + Page Metadata ]
        │
        ▼ (Sentence-Transformers: all-MiniLM-L6-v2)
[ 384-Dimensional Dense Embeddings ]
        │
        ▼
[ ChromaDB Persistent Vector Store ] ── (Cosine Distance Top-K) ──┐
                                                                   │
[ User Technical Query ] ── (Dense Encoding) ──────────────────────┤
                                                                   ▼
                                                       [ Context Injection ]
                                                                   │
                                                                   ▼
                                                    [ Gemini 3.5 Flash Lite ]
                                                    (Strict Grounding Policy)
                                                                   │
                                                                   ▼
                                                      [ Auditable Response ]
                                                   + Page-Level Source Citations
Core Technical Stack
Vector Store: ChromaDB (Local persistent client)

Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional dense vectors)

LLM Engine: Google Gemini 3.5 Flash Lite (Low-latency structured REST transport)

Frontend / Evaluation UI: Streamlit

Document Parsing: PyPDF (Custom chunker preserving document and page provenance)

Environment Security: python-dotenv (Zero hardcoded secrets; persistent .env injection)

Grounding & Anti-Hallucination Protocol
To meet the verification standards required for Earth Observation datasets, the system enforces a strict zero-hallucination policy:

Mandatory Refusal Boundary: If retrieved top-K excerpts contain insufficient evidence to resolve the query, the engine is constrained to return:

"I cannot answer this based on the provided NASA ARSET course materials."

Deterministic Source Attribution: Every factual claim must append an exact document and slide/page citation (e.g., [NASA_ARSET_Course_Notes2.pdf p.8]).

Dual-Column Auditability: The user interface features an interactive Source Excerpt Audit drawer alongside the generated response, allowing real-time inspection of raw retrieved text chunks and similarity distances.

Empirical Retrieval Evaluation & Insights
During stress-testing across 314 indexed document chunks, retrieval failure modes were audited to evaluate the trade-offs of dense semantic search:

1. Semantic Drift in Multi-Entity Queries
Scenario: Asking "What sensors on Terra and Aqua measure Aerosol Optical Depth?" failed at low K values (2 to 5), triggering the grounding refusal.

Root Cause Analysis: The dense embedding was dominated by the high-weight technical phrase Aerosol Optical Depth, ranking theoretical physics chunks (extinction coefficients, radiation interaction) above the platform-specific instrument chunks. The instrument chunk mentioning MODIS on Terra/Aqua was located at Rank 6 (Distance: 0.3646).

Resolution: Expanding the retrieval window to K=6 to 8 or aligning query focus to instrument terminology ("What instruments or sensors are on the Terra and Aqua satellites?") elevated the target chunk to Rank 1 and generated accurate citations.

2. Lexical Inconsistency vs. In-Situ Contrasts
Scenario: Asking for "globally consistent data" for air pollution health impacts triggered a refusal.

Root Cause Analysis: The source materials framed the problem via contrast: describing ground stations as "inconsistent and spatially limited" and satellites as providing "broader spatial data coverage to fill gaps." The model accurately refused to infer terminology not supported by the retrieved context.

Getting Started
Prerequisites
macOS / Linux / Windows

Python 3.10+

A valid Gemini API Key

Installation
Clone the repository:

Bash
git clone https://github.com/YOUR_USERNAME/nasa-rag-engine.git
cd nasa-rag-engine
Set up a virtual environment:

Bash
python3 -m venv venv
source venv/bin/activate
(On Windows use: venv\Scripts\activate)

Install dependencies:

Bash
pip install -r requirements.txt
Configure environment variables:
Create a .env file in the root directory:

Bash
echo 'GEMINI_API_KEY="your_api_key_here"' > .env
Initialize or Append Data:
Place PDF course notes in the Data/ directory and run:

Bash
python add_pdf.py "Data/NASA_ARSET_Course_Notes.pdf"
Launch the Streamlit Engine:

Bash
streamlit run app.py