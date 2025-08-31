import os, json, streamlit as st
from ragstack.ingest import ingest_pdf
from ragstack.chunk_index import build_index
from ragstack.rag_qa import answer_query

st.set_page_config(page_title="Local RAG with Evidence", layout="wide")
st.title("📚 Local RAG (PDF) — Evidence Highlight Demo")

# Upload & ingest
with st.sidebar:
    st.header("1) Ingest PDFs")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    run_ingest = st.button("Ingest")
    idx_dir = st.text_input("Index dir (auto-set after ingest)", "")

if run_ingest and uploaded:
    tmp_path = os.path.join("storage", uploaded.name)
    os.makedirs("storage", exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    out_dir = os.path.join("storage", os.path.splitext(os.path.basename(tmp_path))[0])
    os.makedirs(out_dir, exist_ok=True)
    meta = ingest_pdf(tmp_path, out_dir)
    idx = os.path.join(out_dir, "index")
    os.makedirs(idx, exist_ok=True)
    build_index(os.path.join(out_dir, "ingest_meta.json"), idx)
    st.sidebar.success(f"Ingested. Index: {idx}")
    st.session_state["index_dir"] = idx

if "index_dir" in st.session_state and not st.sidebar.text_input:
    pass

st.header("2) Ask a question")
q = st.text_input("Your question")
if st.button("Ask") and q:
    use_dir = st.session_state.get("index_dir") or idx_dir
    if not use_dir:
        st.error("Please ingest a PDF or set an index dir.")
    else:
        res = answer_query(q, use_dir)
        st.subheader("Answer")
        st.write(res["answer"])
        st.subheader("Citations")
        for i, c in enumerate(res["citations"], 1):
            with st.expander(f"Citation {i}: page {c['page']} — {c['chunk_id']}"):
                st.write(c["snippet"])
        if res.get("highlighted_previews"):
            st.subheader("Highlighted PDF previews")
            for hp in res["highlighted_previews"]:
                st.write(os.path.basename(hp))
                with open(hp, "rb") as f:
                    st.download_button("Download highlighted page", f, file_name=os.path.basename(hp))
