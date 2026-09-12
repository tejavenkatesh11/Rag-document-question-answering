"""Streamlit interface for the RAG question-answering system."""

import logging
import shutil
from pathlib import Path

import streamlit as st

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    SAMPLE_DOCS_DIR,
    SUPPORTED_EXTENSIONS,
    TOP_K,
    UPLOAD_DIR,
)
from src.ingestion import ingest_paths
from src.rag_chain import answer_question
from src.vectorstore import (
    build_vectorstore,
    clear_vectorstore,
    load_vectorstore,
    save_vectorstore,
)

# Root logger stays at WARNING so third-party libraries are quiet by default.
# At INFO, httpx logs every Hugging Face metadata request (~40 lines each time
# the embedding model loads) and faiss.loader reports its AVX2 probe as though
# it were a failure, burying this project's own messages.
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
# This project's own modules stay at INFO: ingestion counts, chunk counts,
# similarity scores and answer latency are the things worth seeing.
logging.getLogger("src").setLevel(logging.INFO)

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")


def init_state() -> None:
    st.session_state.setdefault("store", None)
    st.session_state.setdefault("indexed_files", [])
    st.session_state.setdefault("history", [])


def index_documents(paths: list[Path], label: str) -> None:
    """Ingest, embed and persist a set of documents."""
    if not paths:
        st.warning("No documents to index.")
        return

    try:
        with st.spinner(f"Extracting and chunking {len(paths)} document(s)..."):
            chunks = ingest_paths(paths)

        if not chunks:
            st.error(
                "No text could be extracted. If these are scanned PDFs, they "
                "would need OCR, which this system does not perform."
            )
            return

        with st.spinner(f"Embedding {len(chunks)} chunk(s)..."):
            store = build_vectorstore(chunks)
            save_vectorstore(store)

        st.session_state.store = store
        st.session_state.indexed_files = [p.name for p in paths]
        st.session_state.history = []
        st.success(f"Indexed {len(paths)} {label} into {len(chunks)} chunks.")
    except Exception as exc:  # noqa: BLE001 - report any failure in the UI
        logging.exception("Indexing failed")
        st.error(f"Indexing failed: {exc}")


init_state()

st.title("📄 Document Question Answering")
st.caption(
    "Ask questions about your documents. Answers are grounded in the "
    "retrieved text and cite their sources."
)

# --- Sidebar: documents and settings ------------------------------------
with st.sidebar:
    st.header("Documents")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Add it to your .env file.")

    uploaded = st.file_uploader(
        "Upload documents",
        type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
        help="PDF, TXT and Markdown are supported.",
    )

    if uploaded and st.button("Index uploaded files", use_container_width=True):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in uploaded:
            dest = UPLOAD_DIR / f.name
            dest.write_bytes(f.getbuffer())
            paths.append(dest)
        index_documents(paths, "uploaded file(s)")

    st.divider()

    sample_paths = (
        sorted(
            p
            for p in SAMPLE_DOCS_DIR.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if SAMPLE_DOCS_DIR.exists()
        else []
    )
    if sample_paths:
        st.caption(f"{len(sample_paths)} sample document(s) available")
        if st.button("Load sample documents", use_container_width=True):
            index_documents(sample_paths, "sample document(s)")

    if st.session_state.store is None:
        existing = load_vectorstore()
        if existing is not None:
            st.session_state.store = existing
            st.caption("Loaded previously indexed documents.")

    st.divider()
    st.header("Retrieval settings")

    k = st.slider(
        "Top-K chunks retrieved",
        min_value=1,
        max_value=10,
        value=TOP_K,
        help=(
            "How many chunks are passed to the model. Lower is more focused; "
            "higher improves recall but adds noise."
        ),
    )

    use_history = st.checkbox(
        "Conversational mode",
        value=True,
        help=(
            "Rewrites follow-up questions into standalone ones using the "
            "conversation so far, so pronouns still retrieve correctly."
        ),
    )

    with st.expander("Configuration"):
        st.markdown(
            f"""
            - **LLM:** `{LLM_MODEL}`
            - **Embeddings:** `{EMBEDDING_MODEL.split('/')[-1]}`
            - **Vector store:** FAISS
            - **Chunk size:** {CHUNK_SIZE} chars (overlap {CHUNK_OVERLAP})
            """
        )

    if st.session_state.indexed_files:
        with st.expander(f"Indexed ({len(st.session_state.indexed_files)})"):
            for name in st.session_state.indexed_files:
                st.write(f"- {name}")

    st.divider()
    if st.button("Clear index", use_container_width=True):
        clear_vectorstore()
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        st.session_state.store = None
        st.session_state.indexed_files = []
        st.session_state.history = []
        st.success("Index cleared.")

# --- Main panel ----------------------------------------------------------
if st.session_state.store is None:
    st.info(
        "No documents indexed yet. Upload files or load the sample documents "
        "from the sidebar to begin."
    )
    st.stop()

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["sources"]:
            with st.expander(f"Sources ({len(entry['sources'])})"):
                for s in entry["sources"]:
                    st.markdown(f"**{s['citation']}**")
                    # Excerpts are raw document text, so render them as plain
                    # text - otherwise a Markdown source file's own headings
                    # are rendered as page headings.
                    st.text(s["excerpt"] + "...")
        st.caption(f"Answered in {entry['latency']:.2f}s")

question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.write(question)

    history = (
        [(e["question"], e["answer"]) for e in st.session_state.history]
        if use_history
        else None
    )

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            response = answer_question(
                st.session_state.store, question, k=k, history=history
            )

        if not response.grounded:
            st.warning(response.answer)
        else:
            st.write(response.answer)

        if response.rewritten_query:
            st.caption(f"Interpreted as: {response.rewritten_query}")

        if response.sources:
            with st.expander(f"Sources ({len(response.sources)})"):
                for s in response.sources:
                    st.markdown(f"**{s['citation']}**")
                    st.text(s["excerpt"] + "...")

        st.caption(f"Answered in {response.latency_seconds:.2f}s")

    st.session_state.history.append(
        {
            "question": question,
            "answer": response.answer,
            "sources": response.sources,
            "latency": response.latency_seconds,
        }
    )
