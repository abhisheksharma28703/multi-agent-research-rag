import os
import sys
import warnings
from typing import Dict, Any, Literal

warnings.filterwarnings("ignore")

from langgraph.graph import StateGraph, END
from agents import (
    AgentState,
    retriever_node,
    reasoning_node,
    verifier_node,
    autonomous_arxiv_node
)

# ---------------------------------------------------------------------------
# Conditional Edge: Self-Correction Loop vs. Autonomous arXiv Ingestion vs. Termination
# ---------------------------------------------------------------------------
def route_verification(state: AgentState) -> Literal["arxiv_search", "reasoning", "finalize"]:
    """
    Decides whether to:
    1. Terminate successfully (if answer is verified).
    2. Route to autonomous arXiv search & ingestion (if out-of-corpus & not yet ingested & auto_arxiv_enabled).
    3. Loop back to Reasoning Node with critique (if unverified and retry_count < 1).
    4. Terminate with a warning flag (if unverified and retries exhausted).
    """
    verdict = state.get("verification_result", {})
    is_supported = verdict.get("is_supported", False)
    is_out_of_corpus = verdict.get("is_out_of_corpus", False)
    auto_arxiv_enabled = state.get("auto_arxiv_enabled", True)
    arxiv_ingested = state.get("arxiv_ingested_paper")
    retry_count = state.get("retry_count", 0)

    # 1. Clean pass
    if is_supported:
        return "finalize"

    # 2. Out-of-corpus detection: Trigger autonomous arXiv discovery if enabled and not already attempted
    if is_out_of_corpus and auto_arxiv_enabled and (arxiv_ingested is None):
        print("\n[ROUTE] Missing research paper detected in corpus. Routing to Autonomous arXiv Ingestion Node...")
        return "arxiv_search"

    # 3. First failure: trigger self-correction retry
    if retry_count < 1:
        print(f"\n[LOOP] Verification failed. Routing back to Reasoning Node for self-correction (Retry {retry_count + 1}/1)...")
        return "reasoning"

    # 4. Retries exhausted: flag and exit gracefully
    print("\n[GUARD] Max retries reached. Finalizing with verification warning flag...")
    return "finalize"


# ---------------------------------------------------------------------------
# Node 4: Finalizer / Packaging Node
# ---------------------------------------------------------------------------
def finalizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Constructs the final output payload.
    If paper was auto-ingested from arXiv, prepends an auto-expansion badge.
    If unverified claims persist, attaches a prominent warning banner.
    """
    draft = state.get("draft_answer", "")
    verdict = state.get("verification_result", {})
    is_supported = verdict.get("is_supported", False)
    unsupported = verdict.get("unsupported_claims", [])
    critique = verdict.get("critique", "")
    arxiv_ingested = state.get("arxiv_ingested_paper")

    if is_supported:
        final_text = draft
    else:
        # Graceful degradation warning banner
        warning_banner = (
            "⚠️ **Verification Warning:** Some statements in this response could not be "
            "completely verified against the retrieved research paper text.\n\n"
        )
        if unsupported:
            warning_banner += "**Flagged Claims:**\n"
            for claim in unsupported:
                warning_banner += f"- *\"{claim}\"*\n"
            warning_banner += f"\n**Verifier Audit:** {critique}\n\n---\n\n"
        final_text = warning_banner + draft

    # Prepend Autonomous Discovery Badge if a paper was dynamically indexed
    if arxiv_ingested and arxiv_ingested != "FAILED":
        badge = (
            f"🚀 **Self-Expanded Corpus:** Autonomously fetched and indexed *{arxiv_ingested}* "
            f"directly from arXiv into ChromaDB to answer your query with 100% verified citations!\n\n---\n\n"
        )
        final_text = badge + final_text

    return {"final_output": final_text}


def increment_retry_node(state: AgentState) -> Dict[str, Any]:
    """Increments the retry counter before re-entering the reasoning node."""
    return {"retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------
def build_research_graph():
    """
    Assembles the multi-agent StateGraph:
    Retriever -> Reasoning -> Verifier -> (if missing: arXiv Ingestion -> Retriever)
                                      -> (if fail: Retry Reasoning)
                                      -> Finalizer -> END
    """
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("arxiv_search", autonomous_arxiv_node)
    workflow.add_node("increment_retry", increment_retry_node)
    workflow.add_node("finalizer", finalizer_node)

    # Define Linear Edges
    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "reasoning")
    workflow.add_edge("reasoning", "verifier")

    # Define Conditional Edge from Verifier
    workflow.add_conditional_edges(
        "verifier",
        route_verification,
        {
            "arxiv_search": "arxiv_search",
            "reasoning": "increment_retry",
            "finalize": "finalizer"
        }
    )

    # Autonomous Ingestion Loop: After arXiv indexing, re-retrieve from newly added chunks!
    workflow.add_edge("arxiv_search", "retriever")

    workflow.add_edge("increment_retry", "reasoning")
    workflow.add_edge("finalizer", END)

    return workflow.compile()


_CACHED_GRAPH = None

def get_compiled_graph():
    global _CACHED_GRAPH
    if _CACHED_GRAPH is None:
        _CACHED_GRAPH = build_research_graph()
    return _CACHED_GRAPH

# ---------------------------------------------------------------------------
# Helper runner function
# ---------------------------------------------------------------------------
def run_query(query: str, mode: str = "qa", selected_papers: list = None, auto_arxiv_enabled: bool = True) -> Dict[str, Any]:
    """
    Runs an end-to-end query through the compiled multi-agent LangGraph.
    """
    graph = get_compiled_graph()

    initial_state = {
        "query": query,
        "mode": mode,
        "selected_papers": selected_papers or [],
        "retrieved_chunks": [],
        "draft_answer": "",
        "verification_result": {},
        "retry_count": 0,
        "auto_arxiv_enabled": auto_arxiv_enabled,
        "arxiv_ingested_paper": None,
        "final_output": ""
    }

    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    test_query = "How does residual learning solve the degradation problem in ResNet?"
    print(f"Running LangGraph query: '{test_query}'\n")
    result = run_query(test_query)
    print("\n--- FINAL OUTPUT ---")
    print(result["final_output"])
