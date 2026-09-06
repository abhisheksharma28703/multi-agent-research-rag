# 📚 Multi-Agent RAG System for ML/DL Research Papers

A multi-agent Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Google Gemini Flash**, **ChromaDB**, and **Streamlit**. 

Designed specifically for answering questions, synthesizing ablations, and generating structured comparative analyses across a corpus of foundational Machine Learning and Deep Learning research papers.

---

## ✨ Key Features

* **Multi-Agent Pipeline (LangGraph):** Employs distinct agents for retrieval, synthesis, and fact-checking organized within a cyclic state graph.
* **Automated Claim Verification:** Uses a dedicated Verifier Agent to audit draft statements against retrieved source chunks and flag ungrounded claims.
* **Cross-Paper Comparative Matrix:** Generates structured side-by-side comparison tables across selected research papers.
* **Dynamic arXiv Ingestion:** Allows indexing new research papers live by providing an arXiv ID or paper URL.
* **Local Dense Embeddings:** Runs `sentence-transformers/all-MiniLM-L6-v2` locally on CPU for embedding generation and stores vectors in persistent ChromaDB.

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

## ⚙️ Technical Implementation Details

* **Graph-Based Orchestration:** State-driven execution managed via LangGraph with a strongly-typed `AgentState` schema and cycle termination guards.
* **Independent Claim Verification:** Dedicated verification step that audits synthesized answers against retrieved source chunks using Pydantic structured schemas.
* **Balanced Document Retrieval:** Partitioned vector store queries with metadata filters to ensure equal representation when generating cross-paper comparison matrices.
* **API Reliability & Fallbacks:** Seamless model fallbacks and automated retry handling to maintain smooth execution during high-throughput queries.
