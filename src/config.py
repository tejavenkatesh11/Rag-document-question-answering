"""Central configuration.

Every value that affects retrieval or answer quality is pinned here rather than
left to a library default, so each choice can be pointed at and justified.
Environment variables allow overrides without code changes.
"""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Third-party noise suppression ---------------------------------------
# This module is imported by every other module, so quietening these here
# covers the app, the scripts and the tests alike.

# langchain-community is being sunset upstream. Its PyPDFLoader, TextLoader,
# FAISS and BM25Retriever are the only places this project touches it, and as
# of langchain 1.4 there is nowhere to migrate to: langchain-faiss is an empty
# placeholder package, and langchain_classic re-exports point straight back at
# langchain_community. The warning is therefore informational rather than
# actionable, and it fires on every import. Revisit when standalone
# integration packages actually ship.
warnings.filterwarnings(
    "ignore",
    message=r".*langchain-community.*sunset.*",
    category=DeprecationWarning,
)

# transformers prints a tqdm "Loading weights" bar to stderr each time the
# embedding model is instantiated. Harmless, but it clutters server logs and
# interleaves badly with Streamlit's own output. HF_HUB_DISABLE_PROGRESS_BARS
# governs hub downloads, not this bar, so the API call is what actually works.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
try:
    from transformers.utils import logging as _hf_logging

    _hf_logging.disable_progress_bar()
except Exception:  # noqa: BLE001 - cosmetic only; never break startup for it
    pass

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
UPLOAD_DIR = DATA_DIR / "uploads"

# --- Chunking ------------------------------------------------------------
# 800 characters is roughly 150-200 tokens: large enough to hold a complete
# policy clause or paragraph, small enough that a retrieved chunk is mostly
# signal rather than padding. The 120-character overlap (15%) keeps a sentence
# that straddles a boundary reachable from either side.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

# Tried in order: split on the most semantically meaningful boundary that
# keeps the chunk under CHUNK_SIZE, falling back to blunter separators.
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# --- Embeddings ----------------------------------------------------------
# all-MiniLM-L6-v2: 384 dimensions, ~80MB, runs locally on CPU. Chosen because
# it needs no API key (Groq serves no embedding models) and its quality is
# strong for short-passage retrieval relative to its size.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Retrieval -----------------------------------------------------------
# Measured with scripts/evaluate_retrieval.py over 15 labelled pairs: recall
# reaches 100% at K=3 and stays there, while precision falls from 57.8% (K=3)
# to 26.0% (K=10). So K=3 would suffice on this 11-chunk corpus.
#
# K=5 is kept as the default for headroom rather than recall: a larger corpus
# ranks the correct chunk lower, and multi-document questions need context
# from more than one file. Set TOP_K=3 for a tighter context at this size.
TOP_K = int(os.getenv("TOP_K", 5))

# Hybrid search weights. Dense (semantic) carries most of the load; BM25 is
# there to catch exact terms - IDs, acronyms, policy numbers - where embedding
# similarity is weak.
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.6))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.4))

# Cosine-similarity floor, deliberately set low as a coarse first filter.
#
# Measured on the sample corpus, answerable questions scored 0.51-0.78 and
# unanswerable ones 0.02-0.50. Those ranges very nearly touch: "What is the
# maternity leave duration?" scores 0.504 because it is semantically adjacent
# to the leave sections, even though no document answers it. No single
# threshold separates the two sets, so tuning this upward would start refusing
# genuine questions.
#
# The threshold therefore only rejects the obviously off-topic (a question
# about Japan scores 0.02), saving an unnecessary API call. Near-misses are
# passed to the LLM, whose read of the context is the more reliable judge -
# see the grounding rules in rag_chain.SYSTEM_PROMPT. Hallucination handling
# is the combination of the two gates, not this number alone.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.15))

# --- LLM -----------------------------------------------------------------
# Verified available on the Groq API. temperature=0 because this is an
# extraction task, not a creative one - we want reproducible answers.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

# Groq's free tier caps throughput at 8000 tokens per minute, so a burst of
# questions returns HTTP 429. The client retries with backoff rather than
# surfacing a hard failure to the user.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))

# Strip surrounding quotes and whitespace. python-dotenv removes quotes when
# reading .env, but Docker's --env-file passes them through literally, so a
# key written as GROQ_API_KEY="gsk_..." reaches the container with the quote
# characters attached and fails authentication. Normalising here makes both
# paths behave the same.
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")

# --- Ingestion -----------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
