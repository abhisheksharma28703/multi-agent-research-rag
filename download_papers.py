import os
import sys
import time
import requests
import pymupdf

PAPERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "papers")
os.makedirs(PAPERS_DIR, exist_ok=True)

PAPERS_URLS = {
    "Attention is all you need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "Bert.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
    "ResNet.pdf": "https://arxiv.org/pdf/1512.03385.pdf",
    "GPT-3.pdf": "https://arxiv.org/pdf/2005.14165.pdf",
    "LoRa.pdf": "https://arxiv.org/pdf/2106.09685.pdf",
    "RAG.pdf": "https://arxiv.org/pdf/2005.11401.pdf",
    "Vision Transformer.pdf": "https://arxiv.org/pdf/2010.11929.pdf",
    "Adam Optimizer.pdf": "https://arxiv.org/pdf/1412.6980.pdf",
    "Batch Normalization.pdf": "https://arxiv.org/pdf/1502.03167.pdf",
    "GAN.pdf": "https://arxiv.org/pdf/1406.2661.pdf",
    "VGGNet.pdf": "https://arxiv.org/pdf/1409.1556.pdf",
    "Word2Vec.pdf": "https://arxiv.org/pdf/1301.3781.pdf",
    "YOLO.pdf": "https://arxiv.org/pdf/1506.02640.pdf",
    "AlexNet.pdf": "https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf",
    "Dropout.pdf": "https://arxiv.org/pdf/1207.0580.pdf"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print(f"Starting download of {len(PAPERS_URLS)} official digital research papers...")

for idx, (filename, url) in enumerate(PAPERS_URLS.items(), 1):
    file_path = os.path.join(PAPERS_DIR, filename)
    print(f"[{idx}/{len(PAPERS_URLS)}] Downloading {filename} from {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
        
        # Verify text extraction with PyMuPDF
        doc = pymupdf.open(file_path)
        text_len = sum(len(p.get_text()) for p in doc)
        doc.close()
        print(f"     -> OK: {len(response.content) // 1024} KB | Extractable characters: {text_len}")
    except Exception as e:
        print(f"     -> FAILED: {e}")
    time.sleep(1)

print("\nAll downloads complete!")
