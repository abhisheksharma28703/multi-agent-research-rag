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

## 🚀 Quickstart

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-username/paper-rag-agent.git
cd paper-rag-agent

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

### 3. Ingest Corpus (One-Time Setup)
```bash
python ingestion.py
```

### 4. Launch Web UI
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser!

---

## 📁 Repository Structure

```
paper-rag-agent/
├── data/
│   ├── papers/             # Curated PDF papers (arXiv & NeurIPS digital sources)
│   └── chroma_db/          # Persistent ChromaDB vector store (1,700+ chunks)
├── .env                    # Secret environment variables (ignored by git)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependency manifest
├── download_papers.py      # Automated acquisition of digital PDF sources
├── ingestion.py            # PDF text extraction, chunking, and ChromaDB indexing
├── agents.py               # Retriever, Reasoning, and Verifier agent definitions
├── graph.py                # LangGraph StateGraph orchestration & conditional loop
├── arxiv_utils.py          # Dynamic live arXiv paper downloader & incremental indexer
├── test_api.py             # Diagnostic test script for API connectivity
├── test_agents.py          # Unit test script for individual agent nodes
└── app.py                  # Streamlit frontend with Q&A & Comparative Matrix
```

---

## 🎯 Interview Cheat Sheet & Design Defenses

### 1. Why LangGraph over CrewAI or AutoGen?
> *"CrewAI and AutoGen rely on high-overhead conversational delegation where agents exchange unstructured chat histories. LangGraph provides deterministic state machine control: we explicitly define typed state (`AgentState`), programmatic node functions, and deterministic conditional routing with bounded cycle guards (`retry_count`)."*

### 2. Why decouple Reasoning and Verifier into separate nodes?
> *"When an LLM evaluates its own answer in a single prompt, it suffers from severe confirmation bias (self-attention bias)—it almost always deems its own draft sound. By decoupling verification into a separate node with a fresh context window containing ONLY the retrieved chunks as ground truth, we frame verification as a strict Natural Language Inference (NLI) classification task."*

### 3. How did you handle multi-document comparative retrieval?
> *"Standard RAG vector search exhibits 'retrieval skew'—chunks from one paper often crowd out the other. In Comparative Matrix mode, the Retriever Agent executes partitioned retrieval with metadata filters (`where={"paper_title": ...}`), guaranteeing an equal evidence balance from both papers."*

---

## 📝 Resume Bullet Points (Ready to Copy)

* **Multi-Agent Research Paper RAG (LangGraph, ChromaDB, Gemini, Streamlit):**
  * Engineered a multi-agent RAG system using **LangGraph** orchestrating Retriever, Reasoning, and Verifier agents over 15 foundational ML/DL research papers (1,700+ chunks).
  * Implemented an NLI-based **Verifier Agent** with Pydantic structured outputs and a critique-guided self-correction loop, mitigating hallucinations and enforcing strict citation grounding.
  * Solved multi-document retrieval skew by developing a **Comparative Matrix generator** utilizing partitioned metadata queries in **ChromaDB**.
  * Built an incremental ingestion pipeline via the **arXiv API**, enabling on-the-fly paper indexing with sub-second retrieval latency on local CPU embeddings (`all-MiniLM-L6-v2`).
