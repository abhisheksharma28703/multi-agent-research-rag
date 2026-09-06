import os
import glob
import re
import warnings
import pymupdf
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(BASE_DIR, "data", "papers")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION_NAME = "research_papers"

# Embedding Model (runs 100% locally and free via sentence-transformers)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_function():
    """Initializes and returns the HuggingFace sentence-transformer embeddings."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF using PyMuPDF.
    Returns a list of dictionaries with page content and metadata.
    """
    filename = os.path.basename(pdf_path)
    paper_title = os.path.splitext(filename)[0].replace("_", " ").title()

    doc = pymupdf.open(pdf_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Clean excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > 50:  # Skip blank or near-empty pages
            pages_data.append({
                "text": text,
                "metadata": {
                    "paper_title": paper_title,
                    "filename": filename,
                    "page": page_num + 1
                }
            })

    doc.close()
    return pages_data


def chunk_paper_pages(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Splits page text into semantically cohesive chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for item in pages_data:
        split_texts = text_splitter.split_text(item["text"])
        for chunk_idx, text_chunk in enumerate(split_texts):
            chunk_metadata = dict(item["metadata"])
            chunk_metadata["chunk_index"] = chunk_idx
            chunks.append({
                "text": text_chunk,
                "metadata": chunk_metadata
            })

    return chunks


def get_vector_store():
    """Returns the persistent Chroma vector store instance."""
    embeddings = get_embedding_function()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )


def index_single_pdf(pdf_path: str, vector_store=None) -> int:
    """
    Indexes a single PDF into ChromaDB.
    Returns the number of chunks added.
    """
    if vector_store is None:
        vector_store = get_vector_store()

    filename = os.path.basename(pdf_path)
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_paper_pages(pages)

    if not chunks:
        print(f"  [!] No readable text extracted from {filename}")
        return 0

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Generate unique IDs for each chunk to prevent duplicates
    ids = [f"{c['metadata']['filename']}_p{c['metadata']['page']}_c{c['metadata']['chunk_index']}" for c in chunks]

    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(chunks)


def build_or_load_vector_store(force_reindex: bool = False):
    """
    Scans data/papers/ and indexes any unindexed PDFs into ChromaDB.
    If ChromaDB already has documents and force_reindex is False, it skips redundant work.
    """
    os.makedirs(PAPERS_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    vector_store = get_vector_store()
    existing_count = vector_store._collection.count()

    pdf_files = glob.glob(os.path.join(PAPERS_DIR, "*.pdf"))
    print(f"\n[INFO] Found {len(pdf_files)} PDF files in {PAPERS_DIR}")

    if existing_count > 0 and not force_reindex:
        print(f"[INFO] ChromaDB already populated with {existing_count} chunks.")
        # Check if there are any new PDFs that aren't indexed yet
        all_metadata = vector_store._collection.get(include=["metadatas"])["metadatas"]
        indexed_filenames = set(m.get("filename") for m in all_metadata if m and "filename" in m)
        new_pdfs = [p for p in pdf_files if os.path.basename(p) not in indexed_filenames]

        if not new_pdfs:
            print("[INFO] All PDFs are already fully indexed in ChromaDB! Ready to retrieve.")
            return vector_store

        print(f"[INFO] Indexing {len(new_pdfs)} new PDF(s)...")
        for pdf_path in new_pdfs:
            chunks_added = index_single_pdf(pdf_path, vector_store)
            print(f"     Added {chunks_added} chunks for {os.path.basename(pdf_path)}")
        return vector_store

    print("[INFO] Starting initial index of all papers...")
    total_chunks = 0
    for idx, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        print(f"[{idx}/{len(pdf_files)}] Chunking & indexing: {filename}...")
        chunks_added = index_single_pdf(pdf_path, vector_store)
        print(f"     -> Indexed {chunks_added} chunks.")
        total_chunks += chunks_added

    print(f"\n[SUCCESS] Vector store built successfully! Total chunks indexed: {total_chunks}")
    return vector_store


if __name__ == "__main__":
    build_or_load_vector_store()
