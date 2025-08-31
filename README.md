# End-to-End Local RAG (PDF) with Evidence Highlighting

This project implements a local, open-source Retrieval-Augmented Generation (RAG) system that:
- Ingests mixed PDFs (native + scanned*)
- Chunks, embeds, and indexes text
- Retrieves with re-ranking
- Answers with citations
- Highlights exact evidence on the source PDF pages
- Optional API (FastAPI) and Web UI (Streamlit)

> *OCR for fully scanned PDFs can be added via Tesseract (`pytesseract` + `pdf2image`). The current demo focuses on native PDFs for reliability in an offline environment.*

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run Streamlit demo
streamlit run streamlit_app.py

# Or run API server
uvicorn api:app --reload --port 8000
```

Upload a PDF, build an index, and ask questions. Download per-page PDFs with yellow highlights that correspond to retrieved evidence.

## Architecture (Clean + Local)

- **Ingestion** (`ragstack/ingest.py`): Detects scanned vs native; extracts per-page text + word boxes (PyMuPDF).
- **Chunking & Index** (`ragstack/chunk_index.py`): Recursive overlapping chunks; `all-MiniLM-L6-v2` embeddings; FAISS vector index; JSONL texts + rich metadata.
- **Retriever** (`ragstack/retriever.py`): Vector search -> Cross-Encoder rerank (`ms-marco-MiniLM-L-6-v2`).
- **Answering** (`ragstack/rag_qa.py`): Lightweight synthesis from top contexts (placeholder for a local LLM via Ollama).
- **Evidence Highlight** (`ragstack/highlight.py`): Uses PyMuPDF `search_for` to mark phrases; saves per-page highlighted PDFs.
- **UI/API**: Streamlit app and FastAPI endpoints.

### Replace the Answer Generator with a Local LLM
Point to an Ollama or llama.cpp endpoint and replace the simple synthesizer in `ragstack/rag_qa.py` with a call to your model (e.g., `llama3:8b-instruct`).

## Quality & Measurements

- **Retrieval@k**: Inspect top-k reranked contexts; adjust `k`, chunk size, overlap, and model choices.
- **Latency**: Time ingestion vs query.
- **Grounding**: Every answer returns structured citations and page-level highlight artifacts.

## Files
- `ragstack/` — library modules
- `api.py` — FastAPI server
- `streamlit_app.py` — UI
- `requirements.txt`
- `README.md`
- `HLD.md` — High-Level Design doc

## License
MIT
