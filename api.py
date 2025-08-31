from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import os, tempfile, json
from ragstack.ingest import ingest_pdf
from ragstack.chunk_index import build_index
from ragstack.rag_qa import answer_query

app = FastAPI()

@app.post("/ingest")
async def ingest_endpoint(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    out_dir = os.path.join("storage", os.path.splitext(os.path.basename(tmp_path))[0])
    os.makedirs(out_dir, exist_ok=True)
    meta = ingest_pdf(tmp_path, out_dir)
    idx_dir = os.path.join(out_dir, "index")
    os.makedirs(idx_dir, exist_ok=True)
    build_index(os.path.join(out_dir, "ingest_meta.json"), idx_dir)
    return JSONResponse({"meta": meta, "index_dir": idx_dir})

@app.post("/ask")
async def ask_endpoint(index_dir: str = Form(...), query: str = Form(...)):
    result = answer_query(query, index_dir)
    return JSONResponse(result)
