from typing import List, Dict, Any
import os, json, re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

def load_index(index_dir: str):
    index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    metas = json.load(open(os.path.join(index_dir, "metas.json"), "r", encoding="utf-8"))
    texts = [json.loads(l)["text"] for l in open(os.path.join(index_dir, "texts.jsonl"), "r", encoding="utf-8")]
    return index, metas, texts

def hybrid_retrieve(query: str, index_dir: str, k: int = 6, rerank_top: int = 12) -> List[Dict[str, Any]]:
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    index, metas, texts = load_index(index_dir)
    D, I = index.search(q.astype("float32"), rerank_top)
    candidates = []
    for d, i in zip(D[0], I[0]):
        if i == -1: continue
        meta = metas[i]
        candidates.append({"score_vec": float(d), "text": texts[i], "meta": meta})
    # Cross-encoder rerank
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, c["text"]] for c in candidates]
    ce_scores = ce.predict(pairs)
    for c, s in zip(candidates, ce_scores):
        c["score_rerank"] = float(s)
    reranked = sorted(candidates, key=lambda x: x["score_rerank"], reverse=True)[:k]
    return reranked

def find_evidence_spans(page_text: str, answer_keywords: List[str], window: int = 120) -> List[Dict[str, Any]]:
    spans = []
    for kw in answer_keywords:
        for m in re.finditer(re.escape(kw), page_text, flags=re.IGNORECASE):
            s, e = m.start(), m.end()
            ctx_start = max(0, s - window)
            ctx_end = min(len(page_text), e + window)
            spans.append({"start": s, "end": e, "context": page_text[ctx_start:ctx_end]})
    return spans
