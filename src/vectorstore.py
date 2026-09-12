"""Embeddings, FAISS vector store and hybrid retrieval.

Retrieval is the highest-weighted part of this system, so it combines two
complementary strategies:

  * Dense (FAISS + MiniLM embeddings) - matches on meaning, so "time off"
    finds a passage about "annual leave".
  * Sparse (BM25) - matches on exact tokens, so an ID like "HR-114" or an
    acronym is found even though embeddings handle rare strings poorly.

Their results are fused by EnsembleRetriever using reciprocal rank fusion.
A separate similarity floor then decides whether anything relevant was found
at all, which is what lets the system say "not in the documents" honestly.
"""

import logging
import shutil
from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K,
    VECTORSTORE_DIR,
)

logger = logging.getLogger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the embedding model, loading it once per process.

    The model is ~80MB and takes a few seconds to initialise, so it is cached
    rather than rebuilt for every query.
    """
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            # Normalised vectors make the inner product equal cosine
            # similarity, so scores land in a predictable 0-1 range and the
            # threshold below means something consistent.
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """Embed chunks and build a FAISS index."""
    if not chunks:
        raise ValueError("Cannot build a vector store from zero chunks.")

    logger.info("Embedding %d chunk(s)...", len(chunks))
    store = FAISS.from_documents(chunks, get_embeddings())
    logger.info("FAISS index built with %d vector(s)", store.index.ntotal)
    return store


def save_vectorstore(store: FAISS, path: Path = VECTORSTORE_DIR) -> None:
    """Persist the index so restarts do not require re-embedding."""
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    logger.info("Vector store saved to %s", path)


def load_vectorstore(path: Path = VECTORSTORE_DIR) -> FAISS | None:
    """Load a persisted index, or return None if there isn't one."""
    if not (path / "index.faiss").exists():
        return None
    try:
        # Safe here: we only ever load an index this application wrote.
        store = FAISS.load_local(
            str(path), get_embeddings(), allow_dangerous_deserialization=True
        )
        logger.info("Loaded vector store with %d vector(s)", store.index.ntotal)
        return store
    except Exception:  # noqa: BLE001 - a corrupt index should not be fatal
        logger.exception("Could not load vector store at %s", path)
        return None


def clear_vectorstore(path: Path = VECTORSTORE_DIR) -> None:
    """Delete the persisted index."""
    if path.exists():
        shutil.rmtree(path)
        logger.info("Cleared vector store at %s", path)


def build_hybrid_retriever(
    store: FAISS, chunks: list[Document], k: int = TOP_K
) -> EnsembleRetriever:
    """Combine dense and sparse retrieval into one retriever."""
    dense = store.as_retriever(search_kwargs={"k": k})

    sparse = BM25Retriever.from_documents(chunks)
    sparse.k = k

    return EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[DENSE_WEIGHT, BM25_WEIGHT],
    )


def retrieve_with_scores(
    store: FAISS, query: str, k: int = TOP_K
) -> list[tuple[Document, float]]:
    """Dense retrieval that keeps cosine similarity scores.

    EnsembleRetriever returns fused ranks but not comparable scores, so the
    relevance gate uses dense similarity, which is calibrated and
    interpretable.

    FAISS returns *squared L2 distance* (lower is closer), not similarity.
    Because the embeddings are normalised to unit length, that distance
    converts exactly to cosine similarity:

        |a - b|^2 = 2 - 2*cos(a, b)   =>   cos = 1 - d/2

    Doing the conversion here keeps SIMILARITY_THRESHOLD on a meaningful
    0-1 scale. (LangChain's built-in `similarity_search_with_relevance_scores`
    assumes an unnormalised L2 space and can emit negative values, so it is
    deliberately not used.)
    """
    results = store.similarity_search_with_score(query, k=k)
    return [(doc, 1.0 - float(dist) / 2.0) for doc, dist in results]


def is_relevant(
    scored: list[tuple[Document, float]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> bool:
    """True if the best match clears the relevance floor.

    If nothing does, the question is treated as unanswerable from the corpus
    and the LLM is never asked to speculate.
    """
    if not scored:
        return False
    best = max(score for _, score in scored)
    logger.info("Best similarity score: %.3f (threshold %.2f)", best, threshold)
    return best >= threshold
