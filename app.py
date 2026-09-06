import os
import sys
import glob
import streamlit as st
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Research Paper RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# Synchronize Streamlit Cloud secrets into environment variables if available
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif "GEMINI_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

from graph import run_query
from arxiv_utils import fetch_and_index_arxiv_paper
from ingestion import get_vector_store

# ---------------------------------------------------------------------------
# Styling & Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .badge-verified {
        display: inline-block;
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
        border: 1px solid #A5D6A7;
        margin-bottom: 0.8rem;
    }
    .badge-flagged {
        display: inline-block;
        background-color: #FFF3E0;
        color: #E65100;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
        border: 1px solid #FFCC80;
        margin-bottom: 0.8rem;
    }
    .paper-card {
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        background-color: #F8F9FA;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        border-left: 3px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Get currently indexed papers
# ---------------------------------------------------------------------------
def get_indexed_papers():
    """Retrieves unique paper titles from data/papers directory quickly without blocking on ChromaDB on initial render."""
    papers_dir = os.path.join(os.path.dirname(__file__), "data", "papers")
    pdfs = glob.glob(os.path.join(papers_dir, "*.pdf"))
    return sorted([os.path.splitext(os.path.basename(p))[0].replace("_", " ").title() for p in pdfs])


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_papers" not in st.session_state:
    st.session_state.indexed_papers = get_indexed_papers()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Corpus & Settings")

    # Mode Selector
    mode = st.radio(
        "Select Exploration Mode:",
        ["🔍 Research Q&A Chat", "📊 Comparative Matrix Generator"],
        index=0
    )

    st.markdown("---")

    # Agentic Auto-Discovery Toggle
    st.subheader("⚡ Autonomous Discovery")
    auto_arxiv = st.toggle(
        "Auto-Expand via arXiv",
        value=True,
        help="If a query targets a paper not in our database, the agent will autonomously search arXiv, download & index the paper live, and deliver a verified answer!"
    )

    st.markdown("---")

    # Dynamic arXiv Fetcher
    st.subheader("📥 Add Paper Manually")
    arxiv_input = st.text_input(
        "arXiv ID or URL:",
        placeholder="e.g. 2305.18290 or full URL",
        help="Paste any arXiv paper URL or ID to download and index it live."
    )

    if st.button("Fetch & Index Paper", use_container_width=True):
        if arxiv_input.strip():
            with st.spinner("Downloading and indexing paper..."):
                res = fetch_and_index_arxiv_paper(arxiv_input)
                if res.get("success"):
                    st.success(f"Indexed **{res['paper_title']}** ({res['chunks_added']} chunks)!")
                    st.session_state.indexed_papers = get_indexed_papers()
                    st.rerun()
                else:
                    st.error(res.get("error", "Failed to fetch paper."))
        else:
            st.warning("Please enter an arXiv ID or URL.")

    st.markdown("---")

    # Indexed Papers List
    st.subheader(f"📚 Indexed Corpus ({len(st.session_state.indexed_papers)})")
    with st.expander("View all indexed papers", expanded=False):
        for p in st.session_state.indexed_papers:
            st.markdown(f"<div class='paper-card'>📄 {p}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Content Area
# ---------------------------------------------------------------------------
st.markdown("<div class='main-header'>Multi-Agent Research Paper Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Powered by LangGraph • Retriever Agent • Reasoning Agent • Verifier Agent</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Mode 1: Research Q&A Chat
# ---------------------------------------------------------------------------
if mode == "🔍 Research Q&A Chat":
    # Example Query Pills
    st.markdown("**Sample queries:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Dropout: What problem does it solve?", use_container_width=True):
            st.session_state.preset_query = "What problem does Dropout solve, according to the paper?"
    with col2:
        if st.button("ResNet: How does residual learning work?", use_container_width=True):
            st.session_state.preset_query = "How does residual learning solve the degradation problem in ResNet?"
    with col3:
        if st.button("LoRA: What is the rank allocation?", use_container_width=True):
            st.session_state.preset_query = "How does LoRA reduce trainable parameters using low-rank matrices?"

    st.markdown("---")

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Trust Badge
                if msg.get("is_supported", True):
                    st.markdown("<span class='badge-verified'>✓ Verified Against Source Chunks</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge-flagged'>⚠️ Verification Warning: Unverified Claims Flagged</span>", unsafe_allow_html=True)

                st.markdown(msg["content"])

                # Expandable Audit Drawers
                with st.expander("🔍 Verifier Agent Audit Log", expanded=False):
                    v_res = msg.get("verification_result", {})
                    st.write(f"**Confidence Score:** `{v_res.get('confidence_score', 'N/A')}`")
                    st.write(f"**Audit Verdict:** {v_res.get('critique', 'N/A')}")
                    if v_res.get("unsupported_claims"):
                        st.write("**Unsupported Statements Flagged:**")
                        for uc in v_res["unsupported_claims"]:
                            st.write(f"- *{uc}*")

                with st.expander(f"📑 Retrieved Source Evidence ({len(msg.get('chunks', []))} chunks)", expanded=False):
                    for idx, c in enumerate(msg.get("chunks", []), 1):
                        meta = c.get("metadata", {})
                        st.markdown(f"**Source {idx}:** {meta.get('paper_title')} *(Page {meta.get('page')})*")
                        st.caption(c.get("text", ""))
                        st.divider()
            else:
                st.markdown(msg["content"])

    # User Input
    user_prompt = st.chat_input("Ask any question across the research papers...")
    if "preset_query" in st.session_state and st.session_state.preset_query:
        user_prompt = st.session_state.preset_query
        st.session_state.preset_query = None

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.status("🤖 Multi-Agent Workflow Executing...", expanded=True) as status:
                st.write("1️⃣ **Retriever Agent:** Searching dense vector space in ChromaDB...")
                st.write("2️⃣ **Reasoning Agent:** Synthesizing evidence and structuring citations...")
                st.write("3️⃣ **Verifier Agent:** Auditing claims via Natural Language Inference...")
                
                # Execute LangGraph
                try:
                    import time
                    result = None
                    for attempt in range(2):
                        try:
                            result = run_query(user_prompt, mode="qa", auto_arxiv_enabled=auto_arxiv)
                            break
                        except Exception as inner_e:
                            err_text = str(inner_e)
                            if ("429" in err_text or "RESOURCE_EXHAUSTED" in err_text) and attempt == 0:
                                status.update(label="⏳ Free tier burst limit reached. Retrying automatically in 12s...", state="running")
                                time.sleep(12)
                                continue
                            raise inner_e

                    auto_paper = result.get("arxiv_ingested_paper")
                    if auto_paper and auto_paper != "FAILED":
                        status.update(label=f"🚀 Auto-Expanded & Verified: {auto_paper}!", state="complete", expanded=False)
                        st.session_state.indexed_papers = get_indexed_papers()
                    else:
                        status.update(label="✅ Answer Generated and Audited!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="❌ Execution Error", state="error", expanded=True)
                    err_msg = str(e)
                    if "API key required" in err_msg or "api_key" in err_msg:
                        st.error("🔑 **API Key Missing:** Please add `GEMINI_API_KEY` in Streamlit Cloud Settings -> Secrets.")
                    else:
                        st.error(f"Agent pipeline encountered an error: {e}")
                        st.info("Tip: If you hit a temporary API rate limit, waiting 10-15 seconds and retrying will resolve it.")
                    st.stop()

            verdict = result.get("verification_result", {})
            is_supported = verdict.get("is_supported", True)

            # Display Trust Badge
            if is_supported:
                st.markdown("<span class='badge-verified'>✓ Verified Against Source Chunks</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge-flagged'>⚠️ Verification Warning: Unverified Claims Flagged</span>", unsafe_allow_html=True)

            st.markdown(result["final_output"])

            # Render Audit Logs
            with st.expander("🔍 Verifier Agent Audit Log", expanded=False):
                st.write(f"**Confidence Score:** `{verdict.get('confidence_score', 'N/A')}`")
                st.write(f"**Audit Verdict:** {verdict.get('critique', 'N/A')}")
                if verdict.get("unsupported_claims"):
                    st.write("**Unsupported Statements Flagged:**")
                    for uc in verdict["unsupported_claims"]:
                        st.write(f"- *{uc}*")

            with st.expander(f"📑 Retrieved Source Evidence ({len(result.get('retrieved_chunks', []))} chunks)", expanded=False):
                for idx, c in enumerate(result.get("retrieved_chunks", []), 1):
                    meta = c.get("metadata", {})
                    st.markdown(f"**Source {idx}:** {meta.get('paper_title')} *(Page {meta.get('page')})*")
                    st.caption(c.get("text", ""))
                    st.divider()

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["final_output"],
                "is_supported": is_supported,
                "verification_result": verdict,
                "chunks": result.get("retrieved_chunks", [])
            })


# ---------------------------------------------------------------------------
# Mode 2: Comparative Matrix Generator
# ---------------------------------------------------------------------------
else:
    st.subheader("📊 Cross-Paper Comparative Matrix")
    st.caption("Select any two research papers to generate an audited, side-by-side comparison table.")

    papers = st.session_state.indexed_papers
    if len(papers) < 2:
        st.warning("Please index at least 2 papers to use the Comparative Matrix.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            paper_1 = st.selectbox("Select Paper A:", papers, index=0)
        with col_b:
            # Default to a different paper
            default_b_idx = 1 if len(papers) > 1 else 0
            paper_2 = st.selectbox("Select Paper B:", papers, index=default_b_idx)

        comparison_focus = st.text_input(
            "Comparison Focus / Dimensions:",
            value="Compare core architecture, primary problem solved, training objectives, and limitations."
        )

        if st.button("Generate Audited Comparison Matrix", type="primary", use_container_width=True):
            if paper_1 == paper_2:
                st.error("Please select two different papers to compare.")
            else:
                try:
                    with st.spinner(f"Running partitioned multi-agent comparison: {paper_1} vs. {paper_2}..."):
                        comp_query = f"Compare {paper_1} and {paper_2}. {comparison_focus}"
                        comp_result = run_query(
                            query=comp_query,
                            mode="compare",
                            selected_papers=[paper_1, paper_2]
                         )
                except Exception as e:
                    st.error(f"Comparison error: {e}")
                    st.stop()

                verdict = comp_result.get("verification_result", {})
                is_supported = verdict.get("is_supported", True)

                st.markdown("### Comparison Results")
                if is_supported:
                    st.markdown("<span class='badge-verified'>✓ Comparison Verified Against Source Texts</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge-flagged'>⚠️ Verification Warning: Some comparison cells could not be fully verified</span>", unsafe_allow_html=True)

                st.markdown(comp_result["final_output"])

                # Drawers
                with st.expander("🔍 Verifier Audit Report", expanded=False):
                    st.write(f"**Confidence Score:** `{verdict.get('confidence_score', 'N/A')}`")
                    st.write(f"**Critique:** {verdict.get('critique', 'N/A')}")
                    if verdict.get("unsupported_claims"):
                        st.write("**Unverified Claims in Table:**")
                        for uc in verdict["unsupported_claims"]:
                            st.write(f"- *{uc}*")

                with st.expander("📑 Partitioned Source Chunks Used", expanded=False):
                    for idx, c in enumerate(comp_result.get("retrieved_chunks", []), 1):
                        meta = c.get("metadata", {})
                        st.markdown(f"**Source {idx}:** {meta.get('paper_title')} *(Page {meta.get('page')})*")
                        st.caption(c.get("text", ""))
                        st.divider()

