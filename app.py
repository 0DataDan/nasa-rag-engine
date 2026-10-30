import os
import sys
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv

from build_index import build_vector_store
from query_engine import get_chroma_collection, retrieve_context

load_dotenv()

st.set_page_config(
    page_title="NASA ARSET Grounded RAG Engine",
    page_icon="🛰️",
    layout="wide"
)

# ---------------------------------------------------------
# API Client & Cold-Start Vector Verification
# ---------------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    st.error("Missing `GEMINI_API_KEY`. Configure it in `.env` or Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

collection = get_chroma_collection()
total_chunks = collection.count()

if total_chunks == 0:
    with st.spinner("Cold-start: Indexing NASA ARSET slide decks..."):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if os.path.exists(data_dir):
            build_vector_store()
            collection = get_chroma_collection()
            total_chunks = collection.count()

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.title("🛰️ System Controls")
top_k = st.sidebar.slider("Number of retrieved chunks", min_value=2, max_value=16, value=12)
st.sidebar.divider()
st.sidebar.info(f"Total Chunks in Vector Store: **{total_chunks}**")

# ---------------------------------------------------------
# Header & Navigation Tabs
# ---------------------------------------------------------
st.title("🛰️ NASA ARSET Remote Sensing Engine")
st.caption("Grounded retrieval-augmented generation and interactive visual study tool.")

tab_qa, tab_deck = st.tabs(["🔍 Document QA Engine", "📚 Interactive Study Deck"])

# =========================================================
# TAB 1: Document QA Engine
# =========================================================
with tab_qa:
    st.write("##### Quick Test Queries:")
    cols = st.columns(3)

    if "user_query" not in st.session_state:
        st.session_state["user_query"] = ""

    if cols[0].button("Aerosol Optical Depth"):
        st.session_state["user_query"] = "What is Aerosol Optical Depth and how do satellites measure it?"
        st.rerun()

    if cols[1].button("Orbit Tradeoffs"):
        st.session_state["user_query"] = "Compare geostationary (GEO) vs polar orbiting Low Earth Orbit (LEO) satellites for air quality monitoring temporal and spatial coverage."
        st.rerun()

    if cols[2].button("Vegetation Wavelengths"):
        st.session_state["user_query"] = "What wavelengths of light are absorbed by healthy vegetation?"
        st.rerun()

    with st.form("query_form"):
        query = st.text_input(
            "Ask a technical remote sensing question:",
            value=st.session_state["user_query"],
            placeholder="e.g., How does the TEMPO mission monitor atmospheric pollutants?"
        )
        submitted = st.form_submit_button("Search & Synthesize", type="primary")

    if submitted and query.strip():
        st.session_state["user_query"] = query

        with st.spinner("Retrieving grounded context & synthesizing..."):
            context, raw_results = retrieve_context(query, n_results=top_k)

            SYSTEM_INSTRUCTION = """You are an authoritative AI assistant specializing in NASA Applied Remote Sensing Training (ARSET) materials.
Answer user questions strictly and exclusively using the provided context passages.
- Cite the source document and page number for every factual claim.
- If the question cannot be answered using ONLY the provided context, state:
  "I cannot answer this based on the provided NASA ARSET course materials."
- Do NOT use outside general knowledge or speculate."""

            prompt = f"""Use the following NASA ARSET course excerpts to answer the question.
If the excerpts do not contain the factual answer, use the required refusal statement.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.0
                    )
                )
                answer_text = response.text
            except Exception as e:
                answer_text = f"Error during generation: {e}"

        col_ans, col_ctx = st.columns([1.1, 0.9])
        with col_ans:
            st.subheader("💡 Grounded Synthesis")
            st.markdown(answer_text)

        with col_ctx:
            st.subheader("📑 Retrieved Source Chunks")
            docs = raw_results.get("documents", [[]])[0]
            metas = raw_results.get("metadatas", [[]])[0]
            for idx, (doc, meta) in enumerate(zip(docs, metas)):
                src = meta.get("source", "Unknown")
                pg = meta.get("page", "?")
                with st.expander(f"Chunk {idx+1}: {src} (p. {pg})"):
                    st.caption(f"**Source:** `{src}` | **Page:** `{pg}`")
                    st.write(doc)

# =========================================================
# TAB 2: Interactive Study Deck
# =========================================================
with tab_deck:
    st.markdown("### 📚 NASA ARSET Technical Flashcards")
    st.caption("Visual study cards breaking down orbital mechanics, spectrometry, and aerosol algorithms.")

    STUDY_CARDS = [
        {
            "id": 1,
            "topic": "Orbital Tradeoffs: GEO vs. LEO",
            "tag": "Orbital Mechanics",
            "front": "How do Geostationary (GEO) and Polar-Orbiting Low Earth Orbit (LEO) satellites trade off temporal revisit against spatial coverage?",
            "definition": "LEO satellites orbit ~700-800 km high, providing global coverage with high spatial resolution but only 1-2 daytime revisits per day. GEO satellites orbit at ~36,000 km in sync with Earth's rotation, providing continuous (hourly) daytime monitoring over a fixed regional hemisphere.",
            "metrics": [
                "LEO Altitude: ~705 km (e.g., Terra, Aqua, Sentinel-5P)",
                "GEO Altitude: ~35,786 km (e.g., GOES-16/18, TEMPO)",
                "Temporal Revisit: GEO (Hourly / Sub-hourly) vs LEO (1-2x per day)",
                "Spatial Extent: LEO (Global swaths) vs GEO (Fixed regional disk)"
            ],
            "application": "GEO missions like TEMPO capture diurnal rush-hour pollution spikes; LEO missions track long-term global decadal trends and polar regions."
        },
        {
            "id": 2,
            "topic": "Aerosol Optical Depth (AOD)",
            "tag": "Atmospheric Optics",
            "front": "What does Aerosol Optical Depth (AOD) quantify, and what do high vs. low numerical values indicate?",
            "definition": "AOD is a dimensionless measure of light extinction (scattering + absorption) caused by airborne particulates (dust, smoke, sulfates) across a vertical column of air.",
            "metrics": [
                "AOD < 0.1: Clean, pristine air with high atmospheric visibility",
                "AOD 0.5 - 1.0: Substantial haze, active wildfire smoke, or urban smog",
                "AOD > 2.0 - 3.0: Extreme episodic pollution, thick dust plumes, hazardous AQI",
                "Primary Sensors: MODIS (Terra/Aqua), VIIRS (SNPP/NOAA-20)"
            ],
            "application": "Used as an optical proxy to estimate ground-level PM2.5 concentrations in areas lacking in-situ EPA monitors."
        },
        {
            "id": 3,
            "topic": "Vegetation Spectral Signature",
            "tag": "Spectrometry",
            "front": "Why do healthy green leaves absorb visible light but strongly reflect Near-Infrared (NIR) radiation?",
            "definition": "Chlorophyll pigments absorb blue (~450 nm) and red (~670 nm) light to drive photosynthesis. The internal cellular spongy mesophyll structure scatters and reflects up to 50-60% of Near-Infrared energy to prevent plant overheating.",
            "metrics": [
                "Blue Absorption Peak: ~450 nm (Chlorophyll a/b)",
                "Red Absorption Peak: ~670 nm (Chlorophyll absorption pit)",
                "Red Edge: Steep spectral reflectance jump between 680 nm and 730 nm",
                "NIR Reflectance: ~750-1300 nm (Internal spongy mesophyll scattering)"
            ],
            "application": "Forms the mathematical foundation for vegetation health indices like NDVI: (NIR - Red) / (NIR + Red)."
        },
        {
            "id": 4,
            "topic": "Satellite Data Processing Levels",
            "tag": "Data Engineering",
            "front": "What distinguishes Level 1, Level 2, Level 3, and Level 4 Earth Observation products?",
            "definition": "Satellite processing scales from raw sensor instrument telemetry up to model-assimilated reanalysis grids with zero spatial data gaps.",
            "metrics": [
                "Level 1: Calibrated and geolocated raw sensor radiances/reflectances",
                "Level 2: Retrieved geophysical variables at sensor swath resolution (e.g., TROPOMI NO2 swath)",
                "Level 3: Mapped, quality-filtered, and gridded onto uniform space-time coordinates (e.g., monthly 0.25° grid)",
                "Level 4: Model outputs, multi-sensor fusions, or reanalysis fields (e.g., MERRA-2, CAMS PM2.5)"
            ],
            "application": "Level 2 is used for near-real-time plume tracking; Level 3 is ideal for multi-year trend analysis; Level 4 fills cloud and observation gaps."
        },
        {
            "id": 5,
            "topic": "In-Situ Sampling vs. Satellite Optical Proxies",
            "tag": "Air Quality Monitoring",
            "front": "What are the key complementary tradeoffs between ground monitors (CSN/IMPROVE) and satellite sensors?",
            "definition": "Ground monitors provide exact physical mass filter concentration and chemical speciation at breathing level, but have limited spatial density. Satellites offer contiguous spatial views and boundary crossing warnings, but measure total column optical depth rather than direct surface contact.",
            "metrics": [
                "Ground Networks: EPA CSN, IMPROVE, AirNow (direct physical filter intake)",
                "Satellite Remote Sensing: Column integrated optical depth / trace gas slant columns",
                "Ground Advantage: Direct breathing-zone health standard compliance",
                "Satellite Advantage: Detects intercontinental dust/smoke transport and rural monitor gaps"
            ],
            "application": "Combined via hybrid Level 4 modeling (MERRA-2/CAMS) or machine learning to predict surface AQI across unmonitored zones."
        }
    ]

    if "card_idx" not in st.session_state:
        st.session_state.card_idx = 0
    if "show_back" not in st.session_state:
        st.session_state.show_back = False

    current_card = STUDY_CARDS[st.session_state.card_idx]
    total_cards = len(STUDY_CARDS)

    p_col1, p_col2 = st.columns([4, 1])
    with p_col1:
        st.progress((st.session_state.card_idx + 1) / total_cards)
    with p_col2:
        st.markdown(f"**Card {st.session_state.card_idx + 1} of {total_cards}**")

    card_container = st.container(border=True)
    with card_container:
        st.caption(f"🏷️ **Category:** {current_card['tag']} | **Topic:** {current_card['topic']}")

        if not st.session_state.show_back:
            st.markdown(f"### ❓ {current_card['front']}")
            st.info("Tap **Reveal Answer** below to check your technical understanding.")
        else:
            st.markdown(f"### ✅ {current_card['topic']}")
            st.markdown(f"**Core Principle:**\n{current_card['definition']}")
            
            st.markdown("**Key Parameters & Specifications:**")
            for m in current_card["metrics"]:
                st.markdown(f"- `{m}`")
            
            st.success(f"**Operational Use Case:** {current_card['application']}")

    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c1:
        if st.button("⬅️ Previous Card", disabled=(st.session_state.card_idx == 0)):
            st.session_state.card_idx -= 1
            st.session_state.show_back = False
            st.rerun()

    with c2:
        flip_label = "🔄 Flip to Question" if st.session_state.show_back else "👀 Reveal Answer"
        if st.button(flip_label, use_container_width=True, type="primary"):
            st.session_state.show_back = not st.session_state.show_back
            st.rerun()

    with c3:
        if st.button("Next Card ➡️", disabled=(st.session_state.card_idx == total_cards - 1)):
            st.session_state.card_idx += 1
            st.session_state.show_back = False
            st.rerun()