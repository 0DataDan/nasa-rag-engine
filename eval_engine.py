import os
import sys
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from query_engine import retrieve_context

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("[ERROR] GEMINI_API_KEY not found in environment or .env file.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """You are an authoritative AI assistant specializing in NASA Applied Remote Sensing Training (ARSET) materials.
Answer user questions strictly and exclusively using the provided context passages.
- Cite the source document or slide context for factual claims.
- If the question cannot be answered using ONLY the provided context, state:
  "I cannot answer this based on the provided NASA ARSET course materials."
- Do NOT use outside general knowledge or speculate."""

EVAL_SUITE = [
    # --- 5 IN-DOMAIN TECHNICAL BENCHMARKS ---
    {
        "id": "IN-01",
        "category": "In-Domain",
        "query": "What instruments on Terra and Aqua measure Aerosol Optical Depth?",
        "expected_keywords": ["MODIS"],
        "expect_refusal": False
    },
    {
        "id": "IN-02",
        "category": "In-Domain",
        "query": "What wavelengths of light are absorbed by healthy vegetation?",
        "expected_keywords": ["visible", "absorbed", "radiation", "blue", "red"],
        "expect_refusal": False
    },
    {
        "id": "IN-03",
        "category": "In-Domain",
        "query": "Compare geostationary (GEO) vs polar orbiting Low Earth Orbit (LEO) satellites for air quality monitoring temporal and spatial coverage.",
        "expected_keywords": ["geostationary", "orbit", "leo", "temporal"],
        "expect_refusal": False
    },
    {
        "id": "IN-04",
        "category": "In-Domain",
        "query": "What is the difference between Level 2 and Level 3 satellite data products?",
        "expected_keywords": ["level", "grid", "swath", "orbit", "resolution"],
        "expect_refusal": False
    },
    {
        "id": "IN-05",
        "category": "In-Domain",
        "query": "How does satellite remote sensing complement ground-based in situ air quality monitors?",
        "expected_keywords": ["spatial", "coverage", "monitor", "situ"],
        "expect_refusal": False
    },
    # --- 5 OUT-OF-DOMAIN NEGATIVE CONTROLS ---
    {
        "id": "OUT-01",
        "category": "Out-of-Domain Guardrail",
        "query": "How do lithium-ion batteries function in electric vehicles?",
        "expected_keywords": [],
        "expect_refusal": True
    },
    {
        "id": "OUT-02",
        "category": "Out-of-Domain Guardrail",
        "query": "What are the rules of indoor volleyball scoring?",
        "expected_keywords": [],
        "expect_refusal": True
    },
    {
        "id": "OUT-03",
        "category": "Out-of-Domain Guardrail",
        "query": "Can you provide a recipe for classic Italian tiramisu?",
        "expected_keywords": [],
        "expect_refusal": True
    },
    {
        "id": "OUT-04",
        "category": "Out-of-Domain Guardrail",
        "query": "What is the current corporate tax rate in New York State?",
        "expected_keywords": [],
        "expect_refusal": True
    },
    {
        "id": "OUT-05",
        "category": "Out-of-Domain Guardrail",
        "query": "Who won the FIFA World Cup in 2018 and what was the final score?",
        "expected_keywords": [],
        "expect_refusal": True
    }
]

REFUSAL_STRING = "I cannot answer this based on the provided NASA ARSET course materials."

def run_evaluation():
    print("=" * 70)
    print("🚀 RUNNING NASA ARSET RAG EVALUATION SUITE (k=12)")
    print("=" * 70)

    results = []
    passed_count = 0

    for test in EVAL_SUITE:
        test_id = test["id"]
        category = test["category"]
        query = test["query"]
        expect_refusal = test["expect_refusal"]

        print(f"\nRunning [{test_id}] ({category}): \"{query[:45]}...\"")

        context, raw_results = retrieve_context(query, n_results=12)

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
            answer = response.text.strip()
        except Exception as e:
            answer = f"[ERROR] Inference failed: {e}"

        is_refusal = REFUSAL_STRING.lower() in answer.lower()
        test_passed = False
        notes = ""

        if expect_refusal:
            if is_refusal:
                test_passed = True
                notes = "Correctly triggered negative boundary guardrail."
            else:
                test_passed = False
                notes = "FAILED: Hallucination detected."
        else:
            has_keywords = any(kw.lower() in answer.lower() for kw in test["expected_keywords"])
            if not is_refusal and has_keywords:
                test_passed = True
                notes = "Correctly retrieved and synthesized in-domain technical concepts."
            elif is_refusal:
                test_passed = False
                notes = "FAILED: False refusal on in-domain concept."
            else:
                test_passed = False
                notes = "FAILED: Output did not capture target domain keywords."

        status_icon = "✅ PASS" if test_passed else "❌ FAIL"
        if test_passed:
            passed_count += 1

        print(f"Status: {status_icon} | {notes}")
        results.append({
            "id": test_id,
            "category": category,
            "query": query,
            "status": "PASS" if test_passed else "FAIL",
            "notes": notes,
            "answer_preview": answer[:150].replace("\n", " ") + "..."
        })

    total_tests = len(EVAL_SUITE)
    accuracy = (passed_count / total_tests) * 100

    print("\n" + "=" * 70)
    print(f"📊 EVALUATION SUMMARY: {passed_count}/{total_tests} Passed ({accuracy:.1f}%)")
    print("=" * 70)

    with open("eval_results.json", "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
    print("Audit log written to eval_results.json\n")

if __name__ == "__main__":
    run_evaluation()