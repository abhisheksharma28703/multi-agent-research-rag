import os
import sys
import warnings
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Suppress benign warnings
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from ingestion import get_vector_store

# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    query: str
    mode: str                          # "qa" or "compare"
    selected_papers: List[str]         # e.g., ["ResNet", "VGGNet"]
    retrieved_chunks: List[Dict[str, Any]]
    draft_answer: str
    verification_result: Dict[str, Any]
    retry_count: int                   # Loop guard: max 1 retry
    auto_arxiv_enabled: bool           # Toggle for autonomous paper discovery
    arxiv_ingested_paper: Optional[str] # Title or ID of paper autonomously ingested
    final_output: str


# ---------------------------------------------------------------------------
# 2. Pydantic Schema for Verifier Structured Output
# ---------------------------------------------------------------------------
class VerificationVerdict(BaseModel):
    is_supported: bool = Field(
        description="True if the core technical mechanisms, mathematical explanations, and citations [Paper Name, Page N] in the draft answer are faithfully grounded in the retrieved chunks."
    )
    is_out_of_corpus: bool = Field(
        default=False,
        description="True if the core architecture, algorithm, or seminal paper requested by the user is completely missing from the retrieved source chunks (e.g. user asks about DDPM/Diffusion but only GAN or Transformer chunks were retrieved)."
    )
    confidence_score: float = Field(
        description="Confidence score between 0.0 and 1.0 on the faithfulness of the answer."
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="List of specific sentences or statements from the draft answer that cannot be grounded in the retrieved chunks."
    )
    critique: str = Field(
        description="Actionable feedback explaining why the answer passed or failed verification."
    )


# ---------------------------------------------------------------------------
# 3. LLM Setup
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.2, model: str = "gemini-3.5-flash-lite"):
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            if "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"]
            elif "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                os.environ["GEMINI_API_KEY"] = api_key
        except Exception:
            pass

    primary = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        max_retries=3
    )
    fallback_1 = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=temperature,
        api_key=api_key,
        max_retries=3
    )
    fallback_2 = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=temperature,
        api_key=api_key,
        max_retries=3
    )
    return primary.with_fallbacks([fallback_1, fallback_2])


# ---------------------------------------------------------------------------
# Helper: Topic-to-Canonical-Paper Semantic Routing
# ---------------------------------------------------------------------------
TOPIC_TO_PAPER_MAP = {
    "transformer": "Attention is all you need.pdf",
    "attention": "Attention is all you need.pdf",
    "self-attention": "Attention is all you need.pdf",
    "multi-head": "Attention is all you need.pdf",
    "bert": "Bert.pdf",
    "gpt": "GPT-3.pdf",
    "resnet": "ResNet.pdf",
    "residual": "ResNet.pdf",
    "lora": "LoRa.pdf",
    "rag": "RAG.pdf",
    "dropout": "Dropout.pdf",
    "batch norm": "Batch Normalization.pdf",
    "adam": "Adam Optimizer.pdf",
    "yolo": "YOLO.pdf",
    "alexnet": "AlexNet.pdf",
    "vgg": "VGGNet.pdf",
    "gan": "GAN.pdf",
    "vit": "Vision Transformer.pdf",
    "vision transformer": "Vision Transformer.pdf",
    "word2vec": "Word2Vec.pdf"
}

def identify_canonical_paper(query: str) -> Optional[str]:
    """Detects if a user query targets a specific foundational architecture or paper."""
    import re
    q_lower = query.lower()
    for keyword, filename in TOPIC_TO_PAPER_MAP.items():
        if re.search(rf"\b{re.escape(keyword)}\b", q_lower):
            return filename
    return None


def reformulate_query_for_academic_search(query: str) -> str:
    """Translates casual user questions into dense academic search keywords."""
    llm = get_llm(temperature=0.0)
    prompt = f"""You are an academic query reformulator for an ML/DL research paper vector database.
Given the user's natural language question, generate a concise search string of key technical terms, architectural mechanisms, and mathematical concepts that appear in foundational papers.
Do NOT answer the question. Only output 4-8 comma-separated search keywords.

User Question: "{query}"

Academic Search Keywords:"""
    try:
        res = llm.invoke(prompt)
        content = res.content
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
            content = content[0].get("text", str(content))
        return content.strip().strip('"')
    except Exception:
        return query


def safe_similarity_search(vector_store, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Any]:
    """
    Safely executes similarity search directly against ChromaDB collection.
    Guards against any corrupted or null chunks in vector storage by verifying
    that document text is a valid non-empty string before instantiating Document.
    Completely bypasses LangChain's internal _results_to_docs_and_scores to prevent Pydantic ValidationError.
    """
    try:
        from langchain_core.documents import Document
        col = vector_store._collection
        embed_fn = vector_store._embedding_function

        query_kwargs: Dict[str, Any] = {"n_results": k}
        if embed_fn is not None:
            if hasattr(embed_fn, "embed_query"):
                q_emb = embed_fn.embed_query(query)
                query_kwargs["query_embeddings"] = [q_emb]
            elif callable(embed_fn):
                query_kwargs["query_embeddings"] = [embed_fn(query)]
            else:
                query_kwargs["query_texts"] = [query]
        else:
            query_kwargs["query_texts"] = [query]

        if filter:
            query_kwargs["where"] = filter

        results = col.query(**query_kwargs)
        docs = []
        if results and "documents" in results and results["documents"]:
            doc_list = results["documents"][0]
            meta_list = results.get("metadatas", [[]])[0] if results.get("metadatas") else [{}] * len(doc_list)
            for text, meta in zip(doc_list, meta_list):
                if text is not None and isinstance(text, str) and text.strip():
                    docs.append(Document(page_content=str(text), metadata=meta or {}))
        return docs
    except Exception as e:
        print(f"[WARN] safe_similarity_search encountered an error: {e}")
        return []


# ---------------------------------------------------------------------------
# 4. Agent Node 1: Retriever Agent (with Canonical Routing & Hybrid Retrieval)
# ---------------------------------------------------------------------------
def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves source chunks from ChromaDB.
    Uses Canonical Paper Routing + Semantic Expansion to guarantee deep architectural chunks.
    """
    vector_store = get_vector_store()
    query = state["query"]
    mode = state.get("mode", "qa")
    selected_papers = state.get("selected_papers", [])

    retrieved_chunks = []

    if mode == "compare" and len(selected_papers) >= 2:
        paper_a, paper_b = selected_papers[0], selected_papers[1]
        docs_a = safe_similarity_search(vector_store, query, k=3, filter={"paper_title": paper_a})
        docs_b = safe_similarity_search(vector_store, query, k=3, filter={"paper_title": paper_b})
        if not docs_a:
            docs_a = safe_similarity_search(vector_store, f"{paper_a} {query}", k=3)
        if not docs_b:
            docs_b = safe_similarity_search(vector_store, f"{paper_b} {query}", k=3)
        combined_docs = docs_a + docs_b
    else:
        ingested_paper = state.get("arxiv_ingested_paper")
        if ingested_paper and ingested_paper != "FAILED":
            # The corpus was just expanded with a new paper from arXiv!
            # Search broadly across the vector store to prioritize newly added chunks
            combined_docs = safe_similarity_search(vector_store, query, k=6)
        else:
            # Check for Canonical Paper (e.g. Attention Is All You Need for Transformer queries)
            canonical_file = identify_canonical_paper(query)

            if canonical_file:
                # Targeted search in the foundational paper + semantic search across corpus
                # Skip reformulation LLM call since canonical paper is already precisely targeted
                target_docs = safe_similarity_search(
                    vector_store,
                    query,
                    k=4,
                    filter={"filename": canonical_file}
                )
                general_docs = safe_similarity_search(vector_store, query, k=3)
                
                # Deduplicate by unique content
                seen_texts = set()
                combined_docs = []
                for doc in target_docs + general_docs:
                    if doc.page_content not in seen_texts:
                        seen_texts.add(doc.page_content)
                        combined_docs.append(doc)
            else:
                combined_docs = safe_similarity_search(vector_store, query, k=6)

    for doc in combined_docs:
        if doc and getattr(doc, "page_content", None):
            retrieved_chunks.append({
                "text": str(doc.page_content),
                "metadata": doc.metadata or {}
            })

    return {"retrieved_chunks": retrieved_chunks}


# ---------------------------------------------------------------------------
# 5. Agent Node 2: Reasoning Agent (Mechanism Breakdown & Causality)
# ---------------------------------------------------------------------------
def reasoning_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes a deep architectural explanation based on retrieved chunks.
    Explains the exact technical reasons and mechanisms (HOW & WHY) behind optimal performance.
    """
    llm = get_llm(temperature=0.2)
    query = state["query"]
    mode = state.get("mode", "qa")
    retrieved_chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)
    verification_result = state.get("verification_result", {})

    context_blocks = []
    for idx, c in enumerate(retrieved_chunks, 1):
        meta = c.get("metadata", {})
        paper = meta.get("paper_title", "Unknown Paper")
        page = meta.get("page", "?")
        context_blocks.append(f"[Source {idx} | {paper} (Page {page})]:\n{c['text']}")

    context_str = "\n\n".join(context_blocks)

    if retry_count > 0 and verification_result:
        critique = verification_result.get("critique", "Previous answer had unsupported claims.")
        unsupported = "\n- ".join(verification_result.get("unsupported_claims", []))
        
        prompt_text = f"""You are the Reasoning Agent in a research paper RAG system.
Your previous draft answer had claims that were not completely grounded in the source text.

VERIFIER FEEDBACK:
Critique: {critique}
Claims to Correct or Remove:
- {unsupported}

INSTRUCTIONS:
1. Re-read the retrieved context below.
2. Rewrite the explanation resolving the user's query: "{query}"
3. Strictly remove any ungrounded assertions while clearly explaining the technical mechanisms present in the sources.
4. Attribute all mechanisms to their papers with inline citations [Paper Name, Page N].

RELEVANT SOURCE CONTEXT:
{context_str}

REVISED TECHNICAL ANSWER:"""
    else:
        if mode == "compare":
            prompt_text = f"""You are the Reasoning Agent in a research paper assistant.
The user wants a comparative analysis for query: "{query}".

INSTRUCTIONS:
1. Compare the papers along core architectural dimensions (Architecture, Innovation, Training/Loss, Results/Limitations).
2. Structure your response with a clean Markdown comparison table.
3. Attribute key facts to [Paper Name, Page N].

RELEVANT SOURCE CONTEXT:
{context_str}

COMPARATIVE ANALYSIS:"""
        else:
            prompt_text = f"""You are an expert AI Research Scientist explaining machine learning architectures based on the provided research papers.

USER QUERY:
"{query}"

STRICT GROUNDING & CORPUS INTEGRITY RULES:
1. **Invalid / Gibberish Query Handling:**
   - If the user query is random keystrokes, gibberish, or not a meaningful question (e.g. "ghjkas", "asdfgh", "qwerty", "test1234"):
     - Respond ONLY with:
       "⚠️ **Unrecognized Query:** The query \"{query}\" does not match any recognized machine learning architecture, algorithm, or research concept. Please enter a valid question or paper topic."
     - Do NOT discuss, summarize, or cite unrelated papers from the retrieved context.

2. **Valid Topic But Out-of-Corpus:**
   - If the query asks about a valid ML/DL architecture or paper (e.g. Diffusion Models, Mamba, FlashAttention, etc.) that is NOT present in the retrieved chunks:
     - State clearly:
       "⚠️ **Corpus Boundary Notice:** The indexed research paper database does not currently contain the foundational paper for this topic (\"{query}\")."
     - Provide a concise 2-sentence conceptual summary if appropriate, labeled as "**High-Level Overview (Not Grounded in Database Chunks):**".
     - Conclude with: "To get verified, page-level citations for this architecture, please use the **Dynamic arXiv Ingestion** tool in the sidebar to index the official paper."
     - Do NOT cite or dump summaries of unrelated papers from the retrieved context.

3. **In-Corpus Questions (When Supported in Context):**
   - Explain the exact causal mechanisms (The Core Bottleneck Solved, Mathematical Formulations, Global Context, Decoder Stacks).
   - Support every single architectural claim with verified inline citations strictly from the retrieved chunks [Paper Name, Page N].

RELEVANT SOURCE CONTEXT:
{context_str}

DETAILED ARCHITECTURAL EXPLANATION:"""

    response = llm.invoke(prompt_text)
    draft = response.content
    if isinstance(draft, list) and len(draft) > 0 and isinstance(draft[0], dict):
        draft = draft[0].get("text", str(draft))

    return {"draft_answer": draft}


# ---------------------------------------------------------------------------
# 6. Agent Node 3: Verifier Agent
# ---------------------------------------------------------------------------
def verifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Audits the draft answer against the retrieved chunks using Natural Language Inference.
    Enforces structured output via Pydantic. Uses model fallback if rate limit occurs.
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(VerificationVerdict)

    draft_answer = state.get("draft_answer", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    query = state.get("query", "")

    context_snippets = []
    for idx, c in enumerate(retrieved_chunks, 1):
        meta = c.get("metadata", {})
        context_snippets.append(f"Source {idx} ({meta.get('paper_title', 'Paper')} - Page {meta.get('page', '?')}):\n{c['text']}")

    context_str = "\n\n".join(context_snippets)

    verifier_prompt = f"""You are the strict Verifier Agent in a high-reliability academic RAG system.
Audit the draft answer against the provided ground truth source chunks.

GROUND TRUTH SOURCE CHUNKS:
{context_str}

USER QUERY:
"{query}"

DRAFT ANSWER TO AUDIT:
"{draft_answer}"

VERIFICATION RULES:
1. Grounding Audit: Verify that the architectural mechanisms, mathematical formulations (e.g. Attention formula, Q/K/V roles), and citations [Paper Name, Page N] are faithful to the source chunks.
2. Boundary / Scope Check: If the draft answer indicates that the requested architecture, algorithm, or seminal paper is missing from the database (e.g. Corpus Boundary Notice), or if the retrieved chunks discuss completely different architectures (e.g. GAN chunks retrieved for a Diffusion/DDPM query), mark `is_out_of_corpus = True` and `is_supported = False`.
3. Block Fabrication: If the draft asserts fabricated empirical numbers, false claims, or citations to non-existent sources, mark `is_supported = False`.
4. Mark `is_supported = True` ONLY if the core technical mechanisms and citations are faithfully grounded in the context chunks.
5. If `is_supported = False`, list each specific ungrounded sentence in `unsupported_claims` and provide actionable critique."""

    try:
        verdict: VerificationVerdict = structured_llm.invoke(verifier_prompt)
        verdict_dict = verdict.model_dump()
    except Exception as e:
        # Fallback to secondary model (gemini-3.1-flash-lite) if primary model hits rate limit or error
        try:
            fallback_llm = get_llm(temperature=0.0, model="gemini-3.1-flash-lite")
            verdict: VerificationVerdict = fallback_llm.with_structured_output(VerificationVerdict).invoke(verifier_prompt)
            verdict_dict = verdict.model_dump()
        except Exception as e2:
            verdict_dict = {
                "is_supported": False,
                "is_out_of_corpus": False,
                "confidence_score": 0.1,
                "unsupported_claims": ["Unable to complete automated verification due to API latency or rate limits."],
                "critique": f"Verification failed to run: {e2}"
            }

    return {"verification_result": verdict_dict}


# ---------------------------------------------------------------------------
# 7. Agent Node 4: Autonomous arXiv Search & Dynamic Ingestion Node
# ---------------------------------------------------------------------------
def autonomous_arxiv_node(state: AgentState) -> Dict[str, Any]:
    """
    Autonomous node that executes when an out-of-corpus query is detected.
    Resolves the foundational arXiv paper, downloads and indexes it into ChromaDB in real-time.
    """
    from arxiv_utils import resolve_and_index_paper_for_query
    query = state["query"]
    print(f"\n[ARXIV AUTO-INGEST] Knowledge gap detected for query: '{query}'. Searching arXiv for foundational paper...")

    result = resolve_and_index_paper_for_query(query)
    if result.get("success"):
        paper_title = result.get("paper_title", "New Research Paper")
        arxiv_id = result.get("target_arxiv_id", "")
        chunks = result.get("chunks_added", 0)
        print(f"[ARXIV AUTO-INGEST] Successfully indexed '{paper_title}' ({arxiv_id}) - {chunks} chunks added to ChromaDB.")
        return {
            "arxiv_ingested_paper": f"{paper_title} (arXiv:{arxiv_id})",
            "retry_count": 0,          # Reset retry count so reasoning can synthesize afresh
            "draft_answer": "",        # Clear boundary notice draft
            "verification_result": {}   # Clear previous audit
        }
    else:
        print(f"[ARXIV AUTO-INGEST] Could not resolve paper: {result.get('error')}")
        return {
            "arxiv_ingested_paper": "FAILED",
            "retry_count": 1           # Exhaust retries so it terminates gracefully with boundary notice
        }
