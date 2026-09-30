"""WealthGuard AI chat UI for the retrieval API."""

import os
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
DEMO_RESULT = {
    "status": "success",
    "results": [
        {
            "chunk_id": "demo_product_0",
            "text": "The Secure Growth Plan has an annual management fee of 1.2 percent.",
            "source": "product_brochure.txt",
            "similarity": 0.9115,
        },
        {
            "chunk_id": "demo_compliance_0",
            "text": "Capital protection is not guaranteed. Review the applicable product terms before investing.",
            "source": "compliance.txt",
            "similarity": 0.3351,
        },
    ],
}


def ask_api(question: str, top_k: int) -> dict[str, Any]:
    """Send a question to the configured RAG API."""
    response = requests.post(
        f"{API_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def grounded_summary(results: list[dict[str, Any]]) -> str:
    """Create a transparent answer from retrieved text when the API returns retrieval only."""
    if not results:
        return "I could not find enough verified context to answer that question."
    return "\n\n".join(result["text"] for result in results)


def render_sources(results: list[dict[str, Any]]) -> None:
    """Render source cards with citation metadata."""
    st.markdown("### Retrieved sources")
    for index, result in enumerate(results, start=1):
        source = result.get("source", "Unknown source")
        chunk_id = result.get("chunk_id", "unknown chunk")
        score = result.get("similarity")
        score_label = f"Similarity {score:.3f}" if isinstance(score, (int, float)) else "Similarity unavailable"
        with st.container(border=True):
            st.markdown(f"**{index}. {source}**  ·  `{chunk_id}`  ·  {score_label}")
            st.caption(result.get("text", "No source preview returned."))


st.set_page_config(
    page_title="WealthGuard AI",
    page_icon="◈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7f2; }
    [data-testid="stHeader"] { background: rgba(245,247,242,0.9); }
    .hero { padding: 2.2rem 0 1.2rem; }
    .eyebrow { color: #2e6b55; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
    .hero h1 { color: #163b32; font-size: 3rem; line-height: 1; margin: 0.45rem 0 0.8rem; }
    .hero p { color: #52635b; font-size: 1.08rem; max-width: 42rem; }
    .trust { background: #e6efe8; border-left: 4px solid #2e6b55; color: #244b3c; padding: 0.9rem 1rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background: #ffffff; border-color: #dbe5dc; }
    </style>
    <div class="hero">
      <div class="eyebrow">Evidence-first retrieval</div>
      <h1>Ask WealthGuard.</h1>
      <p>Search approved wealth-management documents and inspect the evidence behind every response.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Query settings")
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=5, value=3)
    demo_mode = st.toggle("Use demo response", value=False)
    st.caption(f"API endpoint: `{API_URL}/query`")
    st.markdown(
        "<div class='trust'><strong>Source-aware by design.</strong><br>Answers stay close to retrieved text so you can verify the document behind them.</div>",
        unsafe_allow_html=True,
    )

st.markdown("### What do you need to verify?")
question = st.text_area(
    "Question",
    placeholder="e.g. What is the annual management fee for the Secure Growth Plan?",
    height=105,
    label_visibility="collapsed",
)

ask = st.button("Search the knowledge base", type="primary", use_container_width=False)

if ask:
    if not question.strip():
        st.warning("Enter a question before searching.")
    else:
        with st.spinner("Searching approved documents..."):
            try:
                payload = DEMO_RESULT if demo_mode else ask_api(question.strip(), top_k)
            except requests.RequestException as error:
                st.error(
                    f"Could not reach the RAG API at {API_URL}. "
                    "Check that the backend is running, then try again."
                )
                st.caption(f"Technical detail: {error}")
            except ValueError:
                st.error("The RAG API returned invalid JSON. Please try again.")
            else:
                results = payload.get("results", [])
                if payload.get("status") == "no_results" or not results:
                    st.info("No matching sources were found. Try a more specific question.")
                else:
                    st.markdown("### Grounded answer")
                    st.markdown(grounded_summary(results))
                    render_sources(results)