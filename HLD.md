# High-Level Design (HLD): Local RAG over PDFs with Evidence Highlighting

**Goal**: Build a local, open-source RAG system that ingests PDFs (native + scanned), answers questions with citations, and visually highlights the exact evidence on source pages. An optional Web UI enables chat over selected PDFs.

---

## 1. System Overview

**Components**
1. **Ingestion**: Per-PDF pipeline; detect scanned vs native; extract page text + word boxes.
2. **Chunking & Embeddings**: Recursive, overlapping chunks; Sentence-Transformers embeddings; FAISS index.
3. **Retriever**: Vector top-N → Cross-Encoder re-rank → top-k contexts.
4. **Generator**: Pluggable local LLM (default stub); prompt with retrieved contexts.
5. **Evidence Highlighter**: Map phrases to coordinates; emit per-page highlighted PDFs.
6. **Serving**: FastAPI endpoints and Streamlit UI.

**Data Artifacts**
- `ingest_meta.json` (document-level)
- `page_XXXX.json` (page text + word boxes)
- `texts.jsonl` (chunk payloads)
- `metas.json` (chunk metadata for citations)
- `faiss.index`, `embeddings.npy`

---

## 2. Ingestion Pipeline

- Library: **PyMuPDF** for consistent text extraction and word bounds.
- Scanned detection: Count text characters per document (`< threshold` ⇒ likely scanned).
- Native PDFs: Extract `get_text('text')` + `get_text('words')` for coordinates.
- Scanned PDFs (option): `pdf2image` → `pytesseract` (TSV/HOCR) → merge into page JSON with word boxes.

**Output**: One JSON per page with text + words + page dim; `ingest_meta.json` with hash, page count, scanned flag.

---

## 3. Chunking & Embeddings & Index

- Chunker: **Recursive** (≈800 chars, 120 overlap, sentence boundary aware).
- Embeddings: `all-MiniLM-L6-v2` (384-dim), normalized.
- Index: **FAISS** (Inner Product / cosine). Store embeddings, `metas.json`, `texts.jsonl`.

**Metadata per chunk**
- `pdf_path`, `page`, `chunk_id`, `span` (char offsets).

---

## 4. Retrieval

1. **Vector** search for `rerank_top` candidates.
2. **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) re-ranks candidates.
3. Return top-`k` contexts + metadata as **citations** (page numbers + chunk ids).

Hybrid BM25 can be added (rank-bm25) if corpus grows or queries are keyword-heavy.

---

## 5. Answer Synthesis

- Default: Lightweight extractive synthesis (safe offline demo).
- **Pluggable**: Replace with local LLM via **Ollama** (e.g., `llama3:8b-instruct`) or any OpenAI-compatible local gateway.
- **Prompting**: System prompt instructs model to **cite** chunk IDs and **only** use provided contexts.

---

## 6. Evidence Highlighting

- Native PDFs: Use `page.search_for(phrase)` to get rectangles; add highlight annotations.
- For OCR pages: map keyword spans to word boxes derived from Tesseract TSV/HOCR.
- Emit **per-page highlighted PDFs**. Return paths in response for UI download.

---

## 7. Web UI & API

- **Streamlit UI**:
  - Upload PDFs → Ingest → Build Index → Ask.
  - Renders answer, expandable citations, and **download buttons** for highlighted page PDFs.
- **FastAPI**:
  - `/ingest` (multipart PDF) → returns `index_dir`.
  - `/ask` (form: `index_dir`, `query`) → returns JSON with answer, citations, highlighted page artifacts.

---

## 8. Clean Architecture Notes

- `ragstack/` as a library; UI/API are thin wrappers.
- Pure functions with typed boundaries; JSON artifacts for transparency.
- Easy to swap **models** (embedding / rerank / LLM) and **index** (FAISS/SQLite/HNSW).

---

## 9. Quality, Testing & Observability

- **Unit tests** (suggested): chunker boundaries, index round-trip, retrieval determinism.
- **Metrics**: latency, memory, Retrieval@k, Context Utilization Rate (CUR).
- **Eval**: Create small QA sets per document; track grounded accuracy.

---

## 10. Deployment & Ops

- Local dev: `venv` + `pip install -r requirements.txt`.
- Optional container: multi-stage Dockerfile (Python slim + system deps like poppler/tesseract).
- **No external calls by default**; models run locally. (First-run model download via HF.)

---

## 11. Constraints & Trade-offs

- All-open-source stack; small models for speed.
- Evidence highlighting relies on exact or fuzzy phrase matches; for paraphrases, highlight nearest spans by character offsets.
- Pure offline demo uses extractive synthesis; swap in a local LLM for abstractive answers.

---

## 12. Demo Plan (for the video)

1. Ingest a PDF.
2. Build index (shows page count).
3. Ask a question → answer + citations.
4. Download highlighted page(s) PDF and show the yellow marks.

---

## 13. Future Enhancements

- OCR-first pipeline for scanned docs with layout preservation.
- True **hybrid** (BM25 + dense fusion).
- Multi-PDF **collection** picker in the UI.
- Better highlight alignment using char-span → word-box mapping.
- Guardrails: citation-required answers; refuse if not grounded.
