from typing import List, Dict, Any
import os, json
from .retriever import hybrid_retrieve
from .highlight import highlight_pdf

def answer_query(query: str, index_dir: str, k: int = 5) -> Dict[str, Any]:
    ctxs = hybrid_retrieve(query, index_dir, k=k)
    # Compose a very simple extractive-ish answer by concatenating top contexts (placeholder for local LLM via Ollama)
    answer = " ".join([c["text"] for c in ctxs[:2]])[:1200]
    citations = []
    for c in ctxs:
        citations.append({
            "pdf_path": c["meta"]["pdf_path"],
            "page": c["meta"]["page"],
            "chunk_id": c["meta"]["chunk_id"],
            "snippet": c["text"][:300]
        })
    # highlight keywords from query terms
    kws = [w for w in query.split() if len(w) > 3][:3]
    previews = []
    for cit in citations[:2]:
        src = cit["pdf_path"]
        outp = os.path.join(index_dir, f"highlight_p{cit['page']}_{cit['chunk_id']}.pdf")
        try:
            hp = highlight_pdf(src, cit["page"], kws, outp)
            previews.append(hp)
        except Exception as e:
            pass
    return {"answer": answer, "citations": citations, "highlighted_previews": previews}
