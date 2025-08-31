from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
import os, json
from .utils import ensure_dir, file_sha1

def is_scanned_pdf(doc: fitz.Document, text_threshold: int = 20) -> bool:
    text_chars = 0
    for page in doc:
        text = page.get_text("text")
        text_chars += len(text)
        if text_chars > text_threshold:
            return False
    return True

def extract_native_page(page) -> Dict[str, Any]:
    # Return plain text plus word boxes to enable highlight
    words = page.get_text("words")  # list of (x0, y0, x1, y1, word, block_no, line_no, word_no)
    text = page.get_text("text")
    return {
        "text": text,
        "words": [{"bbox": w[:4], "text": w[4]} for w in words],
        "width": page.rect.width,
        "height": page.rect.height,
    }

def ingest_pdf(pdf_path: str, out_dir: str) -> Dict[str, Any]:
    ensure_dir(out_dir)
    doc = fitz.open(pdf_path)
    meta = {
        "pdf_path": os.path.abspath(pdf_path),
        "sha1": file_sha1(pdf_path),
        "n_pages": doc.page_count,
        "scanned": is_scanned_pdf(doc),
        "pages": [],
    }
    for i in range(doc.page_count):
        page = doc.load_page(i)
        page_data = extract_native_page(page)
        page_json_path = os.path.join(out_dir, f"page_{i+1:04d}.json")
        with open(page_json_path, "w", encoding="utf-8") as f:
            json.dump(page_data, f, ensure_ascii=False)
        meta["pages"].append({"index": i, "json": page_json_path})
    meta_path = os.path.join(out_dir, "ingest_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta
