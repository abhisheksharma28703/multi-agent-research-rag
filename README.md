# 📚 Multi-Agent RAG System for ML/DL Research Papers

A production-grade, multi-agent Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Gemini 3.6 Flash**, **ChromaDB**, and **Streamlit**. 

Designed specifically for answering questions, synthesizing ablations, and generating structured comparative analyses across a corpus of foundational Machine Learning and Deep Learning research papers.

---

## 🌟 Key Differentiators (Why This Goes Beyond Generic "Chat-with-PDF")

1. **Dedicated Verifier Agent (Faithfulness Auditor):**
   * Eliminates the classic RAG flaw: silent hallucinations.
   * A separate agent audits the draft answer against retrieved source chunks using strict **Natural Language Inference (NLI)** before the user ever sees it.
2. **Critique-Guided Self-Correction Loop (Bounded LangGraph Cycle):**
   * If the Verifier detects unsupported claims or hallucinations, it provides actionable critique and routes the state back to the Reasoning Agent to rewrite the answer.
   * Includes a stateful **Cycle Guard** (`retry_count` bound) to guarantee termination safety.
3. **Balanced Cross-Paper Comparative Matrix:**
   * Solves "Retrieval Skew" during multi-document comparisons. Uses partitioned metadata filtering in ChromaDB (`where={"paper_title": ...}`) to pull balanced evidence from both papers simultaneously.
4. **Live Incremental Ingestion via arXiv API:**
   * In addition to the 15 pre-indexed foundational papers, users can paste any live arXiv ID/URL to dynamically download, parse, and append new papers to the running ChromaDB vector space in seconds.
5. **100% Local & Cost-Free Embeddings:**
   * Employs `sentence-transformers/all-MiniLM-L6-v2` locally on CPU (384-dimensional normalized dense vectors), keeping embedding computation private and free.

---

## 🏛️ System Architecture

```
                       ┌──────────────────────────────────────────┐
                       │           DATA INGESTION LAYER           │
                       └──────────────────────────────────────────┘
                         15 Curated ML Papers (PDFs)
                                      +
                         Live arXiv Ingestion (API)
                                      │
                                      ▼
                        PyMuPDF Text & Page Extractor
                                      │
                                      ▼
                        RecursiveCharacterTextSplitter
                       (Chunk: 700 chars | Overlap: 120)
                                      │
                                      ▼
                        sentence-transformers/all-MiniLM
                                      │
                                      ▼
                           ChromaDB Persistent Store
                                      │
══════════════════════════════════════╪═════════════════════════════════════
                       ┌──────────────┴───────────────────────────┐
                       │      LANGGRAPH MULTI-AGENT RUNTIME       │
                       └──────────────────────────────────────────┘
                                      │
                                [User Prompt]
                     (Mode: Q&A  OR  Comparative Matrix)
                                      │
                                      ▼
                             [Retriever Agent]
              • Q&A: Top-5 semantic search across corpus
              • Compare: Partitioned retrieval (top-3 Paper A + top-3 Paper B)
                                      │
                                      ▼
                            [Reasoning Agent]
              • Generates answer / table with page-level citations
              • If retry: Integrates verifier critique to purge unverified facts
                                      │
                                      ▼
                            [Verifier Agent]
              • Performs NLI audit: Checks claims against source chunks
              • Outputs Pydantic schema: is_supported, critique, unsupported_claims
                                      │
                                      ▼
                           < Conditional Edge >
                                     / \
                is_supported == True/   \is_supported == False & retries < 1
                                   /     \
                                  /       ▼
                                 /     [Loop back to Reasoning Agent]
                                /      (with critique to self-correct)
                               /          │
                              /           ▼ (if it fails again)
                             /         [Flag answer with ⚠️ warning]
                            /             │
                            ▼             ▼
                       ┌──────────────────────────────────────────┐
                       │           PRESENTATION LAYER             │
                       │             (Streamlit UI)               │
                       └──────────────────────────────────────────┘
                        • Trust Badge (✓ Verified  OR  ⚠️ Flagged)
                        • Verified Answer / Comparison Matrix
                        • Collapsible Audit Drawers (Evidence & Critique)
```

---

## 📚 Curated Research Paper Corpus

Pre-indexed with over 1,760 chunks covering foundational architectures across NLP, Computer Vision, and Optimization:
1. **Attention Is All You Need** (Transformer) — *Vaswani et al.*
2. **BERT** — *Devlin et al.*
3. **ResNet** — *He et al.*
4. **GPT-3** — *Brown et al.*
5. **LoRA** — *Hu et al.*
6. **RAG (Original Paper)** — *Lewis et al.*
7. **Vision Transformer (ViT)** — *Dosovitskiy et al.*
8. **Adam Optimizer** — *Kingma & Ba*
9. **Batch Normalization** — *Ioffe & Szegedy*
10. **Dropout** — *Srivastava et al.*
11. **GAN** — *Goodfellow et al.*
12. **VGGNet** — *Simonyan & Zisserman*
13. **AlexNet** — *Krizhevsky et al.*
14. **Word2Vec** — *Mikolov et al.*
15. **YOLO** — *Redmon et al.*

---

## 🚀 Quickstart & Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/abhisheksharma28703/multi-agent-research-rag.git
cd multi-agent-research-rag

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Ingest Corpus (Optional - Pre-indexed DB included)
```bash
python ingestion.py
```

### 4. Launch Web Application
```bash
streamlit run app.py
```
Access the application locally at [http://localhost:8501](http://localhost:8501).

---

## 📁 Repository Structure

```
multi-agent-research-rag/
├── data/
│   ├── papers/             # Official digital PDF research papers
│   └── chroma_db/          # Persistent ChromaDB vector store (SQLite & HNSW index)
├── .env                    # Secret environment variables (ignored by git)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependency manifest
├── download_papers.py      # Acquisition utility for digital PDF sources
├── ingestion.py            # PDF parsing, semantic chunking, and ChromaDB indexing
├── agents.py               # Retriever, Reasoning, and Verifier agent definitions
├── graph.py                # LangGraph StateGraph orchestration & conditional loop
├── arxiv_utils.py          # Dynamic live arXiv paper downloader & incremental indexer
└── app.py                  # Streamlit interface with Q&A & Comparative Matrix
```

---

## ⚙️ Architectural Decisions & Engineering Rationale

### 1. State Machine Orchestration (LangGraph over Free-Form Agents)
Rather than relying on conversational agent frameworks where agents exchange unstructured chat logs with high overhead, this system utilizes **LangGraph** to construct a deterministic, state-driven workflow. It enforces a strongly typed `AgentState` schema, explicit node contracts, and bounded conditional routing with cycle guards (`retry_count`) to guarantee termination safety.

### 2. Decoupled Verification vs. Self-Evaluation
When a single generative model evaluates its own response in the same inference pass, it suffers from self-attention confirmation bias. By isolating the **Verifier Agent** into a dedicated graph node with a context window restricted strictly to retrieved source chunks, claim verification is modeled as a formal **Natural Language Inference (NLI)** classification task enforced by Pydantic structured schemas.

### 3. Partitioned Multi-Document Vector Retrieval
Standard similarity searches in multi-document RAG systems often experience "retrieval skew," where chunks from a single dense paper dominate top-$k$ results. The Comparative Matrix engine executes partitioned metadata filtering (`where={"paper_title": ...}`), ensuring strictly balanced evidence extraction across target papers.

### 4. Production Fault-Tolerance & API Resilience
* **Multi-Model Failover:** Utilizes LangChain's `with_fallbacks` pattern to automatically route inference from primary lightweight models (`gemini-3.5-flash-lite`) to backup models during transient cloud outages or quota throttles.
* **Defensive Data Ingestion:** Sanitizes extracted PDF chunks at both write-time and read-time, bypassing brittle third-party vector store wrappers to prevent null document deserialization failures.
* **Rate-Limit Resilience:** Incorporates an automated backoff buffer to absorb API burst limitations without interrupting active user sessions.
