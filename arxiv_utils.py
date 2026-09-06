import os
import re
import sys
import requests
import arxiv
from typing import Dict, Any, Optional

from ingestion import index_single_pdf, get_vector_store

PAPERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "papers")


def extract_arxiv_id(input_str: str) -> Optional[str]:
    """
    Extracts an arXiv ID from a URL or raw ID string.
    Matches formats like:
      - 2305.18290
      - 2305.18290v2
      - https://arxiv.org/abs/2305.18290
      - https://arxiv.org/pdf/2305.18290.pdf
    """
    clean_input = input_str.strip()
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", clean_input)
    if match:
        return match.group(1)
    return None


def fetch_and_index_arxiv_paper(arxiv_input: str) -> Dict[str, Any]:
    """
    Fetches a paper from arXiv, downloads its official digital PDF,
    and incrementally indexes it into the live ChromaDB vector store.
    """
    paper_id = extract_arxiv_id(arxiv_input)
    if not paper_id:
        return {
            "success": False,
            "error": "Invalid arXiv ID or URL. Please provide a format like '2305.18290' or 'https://arxiv.org/abs/2305.18290'."
        }

    os.makedirs(PAPERS_DIR, exist_ok=True)
    clean_id = paper_id.split("v")[0]  # Remove version suffix for query

    try:
        # 1. Query arXiv metadata
        client = arxiv.Client()
        search = arxiv.Search(id_list=[clean_id])
        paper = next(client.results(search), None)

        if not paper:
            return {"success": False, "error": f"No paper found on arXiv with ID: {paper_id}"}

        # 2. Construct safe filename
        sanitized_title = re.sub(r'[\\/*?:"<>|]', "", paper.title)
        sanitized_title = re.sub(r'\s+', ' ', sanitized_title).strip()
        filename = f"{sanitized_title}.pdf"
        file_path = os.path.join(PAPERS_DIR, filename)

        # 3. Download the PDF directly if not already present
        if not os.path.exists(file_path):
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0"
            }
            pdf_url = paper.pdf_url
            if not pdf_url.endswith(".pdf"):
                pdf_url += ".pdf"

            response = requests.get(pdf_url, headers=headers, timeout=60)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

        # 4. Incrementally index into ChromaDB
        vector_store = get_vector_store()
        chunks_added = index_single_pdf(file_path, vector_store)

        return {
            "success": True,
            "paper_title": paper.title,
            "filename": filename,
            "authors": [a.name for a in paper.authors[:3]],
            "published": str(paper.published.year),
            "chunks_added": chunks_added,
            "summary": paper.summary[:300] + "..."
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to ingest paper from arXiv: {str(e)}"}


def resolve_and_index_paper_for_query(query: str) -> Dict[str, Any]:
    """
    Autonomously resolves the seminal paper for an out-of-corpus ML query
    and indexes it into ChromaDB in real-time.
    """
    import json
    from agents import get_llm
    
    # 1. Ask LLM to resolve canonical paper ID or search query
    llm = get_llm(temperature=0.0)
    prompt = f"""You are an autonomous AI research librarian. The user asked a question about a machine learning architecture or concept that is missing from our local database.
Your job is to identify the seminal, foundational research paper on arXiv that answers this question.

Output ONLY valid JSON with no markdown wrapping:
{{
  "canonical_arxiv_id": "string (e.g. '2006.11239' if known, else null)",
  "paper_title": "string (title of the foundational paper)",
  "search_query": "string (concise 3-6 word search query for arXiv API)"
}}

User Question: "{query}"
JSON:"""
    try:
        response = llm.invoke(prompt)
        text = response.content
        if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
            text = text[0].get("text", str(text))
        
        # Clean JSON markdown if any
        text = re.sub(r"^```json\s*", "", text.strip())
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"```$", "", text.strip())
        
        data = json.loads(text.strip())
        canonical_id = data.get("canonical_arxiv_id")
        search_query = data.get("search_query") or data.get("paper_title") or query
    except Exception:
        canonical_id = None
        search_query = query

    target_id = canonical_id
    if not target_id:
        # 2. Query arXiv API using search_query
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=search_query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            top_paper = next(client.results(search), None)
            if top_paper:
                match = re.search(r"(\d{4}\.\d{4,5})", top_paper.entry_id)
                if match:
                    target_id = match.group(1)
        except Exception:
            pass

    if not target_id:
        return {
            "success": False,
            "error": f"Could not autonomously resolve an arXiv paper for query: '{query}'"
        }

    # 3. Download, chunk, and index the paper
    ingest_result = fetch_and_index_arxiv_paper(target_id)
    ingest_result["target_arxiv_id"] = target_id
    return ingest_result


if __name__ == "__main__":
    # Test with DPO (Direct Preference Optimization, 2305.18290)
    test_id = "2305.18290"
    print(f"Testing dynamic ingestion of arXiv paper: {test_id}...")
    result = fetch_and_index_arxiv_paper(test_id)
    print("Result:", result)
