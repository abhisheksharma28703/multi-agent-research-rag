import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from agents import retriever_node, reasoning_node, verifier_node

print("--- Testing Agent 1: Retriever Node ---")
state = {
    "query": "What problem does Dropout solve, according to the paper?",
    "mode": "qa",
    "selected_papers": [],
    "retry_count": 0
}

# 1. Run Retriever Node
retrieval_output = retriever_node(state)
state.update(retrieval_output)
print(f"Retrieved {len(state['retrieved_chunks'])} chunks.")
for idx, c in enumerate(state['retrieved_chunks'][:2], 1):
    meta = c.get('metadata', {})
    print(f"  Chunk {idx}: {meta.get('paper_title')} (Page {meta.get('page')})")

# 2. Run Reasoning Node
print("\n--- Testing Agent 2: Reasoning Node ---")
reasoning_output = reasoning_node(state)
state.update(reasoning_output)
print("Draft Answer Generated:")
print(state["draft_answer"][:350] + "...\n")

# 3. Run Verifier Node
print("--- Testing Agent 3: Verifier Node ---")
verifier_output = verifier_node(state)
state.update(verifier_output)
verdict = state["verification_result"]
print(f"Supported: {verdict.get('is_supported')}")
print(f"Confidence Score: {verdict.get('confidence_score')}")
print(f"Critique: {verdict.get('critique')}")
print(f"Unsupported Claims: {verdict.get('unsupported_claims')}")
