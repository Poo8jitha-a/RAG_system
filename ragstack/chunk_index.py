from typing import List, Dict, Any, Tuple
import os, json, math
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from .utils import ensure_dir

def recursive_chunks(text: str, max_len: int = 800, overlap: int = 120) -> List[Tuple[int, int, str]]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_len, n)
        # expand to sentence boundary if possible
        slice_text = text[i:j]
        last_period = slice_text.rfind(".")
        if last_period > 300 and j < n:
            j = i + last_period + 1
            slice_text = text[i:j]
        chunks.append((i, j, slice_text.strip()))
        i = max(j - overlap, i + 1)
    return chunks

class VectorIndex:
    def __init__(self, dim: int, metric: str = "ip"):
        self.metric = metric
        if metric == "ip":
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexFlatL2(dim)
        self.embeddings = None
        self.meta: List[Dict[str, Any]] = []

    def add(self, embs: np.ndarray, metas: List[Dict[str, Any]]):
        if self.embeddings is None:
            self.embeddings = embs.astype("float32")
        else:
            self.embeddings = np.vstack([self.embeddings, embs.astype("float32")])
        self.index.add(embs.astype("float32"))
        self.meta.extend(metas)

    def search(self, q: np.ndarray, k: int = 5):
        D, I = self.index.search(q.astype("float32"), k)
        results = []
        for row_i, (drow, irow) in enumerate(zip(D, I)):
            arr = []
            for d, idx in zip(drow, irow):
                if idx == -1: 
                    continue
                arr.append({"score": float(d), "meta": self.meta[idx]})
            results.append(arr)
        return results

def build_index(ingest_meta_path: str, out_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    ensure_dir(out_dir)
    with open(ingest_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    vdb = VectorIndex(dim=dim, metric="ip")

    all_texts = []
    all_metas = []
    for page in meta["pages"]:
        with open(page["json"], "r", encoding="utf-8") as f:
            pdata = json.load(f)
        text = pdata["text"]
        chunks = recursive_chunks(text)
        for ci, (s, e, ch) in enumerate(chunks):
            if len(ch.strip()) == 0:
                continue
            all_texts.append(ch)
            all_metas.append({
                "pdf_path": meta["pdf_path"],
                "page": page["index"] + 1,
                "span": [s, e],
                "chunk_id": f"p{page['index']+1}_c{ci+1}",
            })
    if not all_texts:
        raise RuntimeError("No text extracted to index.")
    embs = model.encode(all_texts, normalize_embeddings=True, convert_to_numpy=True)
    vdb.add(embs, all_metas)

    faiss.write_index(vdb.index, os.path.join(out_dir, "faiss.index"))
    np.save(os.path.join(out_dir, "embeddings.npy"), vdb.embeddings)
    with open(os.path.join(out_dir, "metas.json"), "w", encoding="utf-8") as f:
        json.dump(vdb.meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "texts.jsonl"), "w", encoding="utf-8") as f:
        for t in all_texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
