# Project: Multi-Agent RAG System for ML/DL Research Papers

## Overview
A multi-agent RAG (Retrieval-Augmented Generation) system built with LangGraph that answers questions, explains concepts, and compares content across a corpus of ML/DL research papers. The key differentiator from a generic "chat with PDF" app is a dedicated **Verifier Agent** that cross-checks generated answers against retrieved source chunks to reduce hallucinations.

## Constraints
- Keep scope small — builder has limited time and other placement-prep topics to study in parallel.
- Do not add extra agents/features beyond what's specified below (no code-generator agent, no citation-tracer, no gap-finder — these were intentionally dropped to keep scope lean).

## Architecture — 3 Agents (LangGraph, linear/conditional flow)

```
User Query → Retriever Agent → Reasoning Agent → Verifier Agent → Final Answer
                                                        │
                                          (if unsupported/fails check)
                                                        │
                                                        ▼
                                        flag answer OR loop back to Reasoning
```

1. **Retriever Agent** — embeds the user query and retrieves top-k relevant chunks from the vector store.
2. **Reasoning Agent** — takes the query + retrieved chunks and generates an answer/explanation/comparison using an LLM.
3. **Verifier Agent** — takes the generated answer + the retrieved chunks and checks whether the answer is actually supported by the source content; flags or regenerates if unsupported claims are found.

State object passed through the graph should carry: `query`, `retrieved_chunks`, `answer`, `verification_result`.

## Data Source
15 well-known ML/DL research papers (PDFs), covering NLP, CV, and foundational topics, e.g.:
- Attention Is All You Need (Transformer)
- BERT
- GPT-3
- ResNet
- VGGNet
- GANs
- Dropout
- Batch Normalization
- Word2Vec
- YOLO
- Adam Optimizer
- Vision Transformer (ViT)
- LoRA
- RAG (original paper)
- (one more CV/NLP foundational paper of choice)

## Folder Structure
```
paper-rag-agent/
├── data/
│   └── papers/
│       ├── attention_is_all_you_need.pdf
│       ├── bert.pdf
│       ├── resnet.pdf
│       └── ... (rest of the papers, underscore-separated filenames, no spaces)
├── app.py
└── requirements.txt
```

## Tech Stack
- **PDF text extraction:** PyMuPDF (fitz)
- **Chunking:** LangChain `RecursiveCharacterTextSplitter` (~500–800 tokens, slight overlap)
- **Embeddings:** sentence-transformers (e.g. all-MiniLM-L6-v2, local/free) — or OpenAI embeddings if API budget allows
- **Vector store:** ChromaDB (local, free, easy setup)
- **Agent orchestration:** LangGraph (StateGraph, nodes, conditional edges)
- **LLM:** Gemini API (free tier) or GPT-4o-mini
- **UI:** Streamlit (simple chat-style input/output)

## Step-by-Step Build Plan
1. **Data ingestion** — extract text from each PDF in `data/papers/` using PyMuPDF.
2. **Chunking** — split extracted text into overlapping chunks.
3. **Embeddings** — convert chunks into vector embeddings via sentence-transformers.
4. **Vector store** — store embeddings + metadata (paper name, section if available) in ChromaDB.
5. **LangGraph agent graph** — implement the 3 agents (Retriever, Reasoning, Verifier) as nodes in a StateGraph, connected via conditional edges (verifier can trigger a retry/flag).
6. **LLM integration** — connect Gemini or GPT-4o-mini as the reasoning/verification LLM.
7. **Streamlit UI** — build a simple chat interface where the user types a question and sees the answer (plus a flag if verification failed).
8. **Testing** — run 5–10 sample questions across the papers (factual lookups, comparisons between papers, concept explanations) to validate retrieval and verification quality.
9. **Documentation** — write a README explaining the architecture, setup steps, and how to run the app; push to GitHub.

## Example Queries to Support
- "How does the attention mechanism differ between Paper X and Paper Y?"
- "Explain the ablation study results in Paper X."
- "What problem does Dropout solve, according to the paper?"

## Resume Description (for reference)
> Built a multi-agent RAG system using LangGraph to answer, compare, and summarize content across a corpus of ML/DL research papers, with a dedicated verifier agent to reduce hallucinations by cross-checking generated answers against retrieved source chunks.
>
> Tech: Python, LangGraph, ChromaDB, Sentence-Transformers, Streamlit
