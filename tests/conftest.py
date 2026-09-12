"""Shared pytest fixtures."""

import time

import pytest

from src.config import SAMPLE_DOCS_DIR, SUPPORTED_EXTENSIONS


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: hits the live Groq API (needs GROQ_API_KEY)"
    )


@pytest.fixture(scope="session")
def sample_paths():
    """Paths to the sample corpus."""
    if not SAMPLE_DOCS_DIR.exists():
        pytest.skip("Sample documents directory is missing")
    paths = sorted(
        p for p in SAMPLE_DOCS_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        pytest.skip("No sample documents found")
    return paths


@pytest.fixture(scope="session")
def sample_chunks(sample_paths):
    """The sample corpus, loaded and chunked once for the whole session."""
    from src.ingestion import ingest_paths

    return ingest_paths(sample_paths)


@pytest.fixture(scope="session")
def rag_store(sample_chunks):
    """A FAISS store over the sample corpus.

    Session-scoped because embedding the corpus takes a few seconds and the
    result is read-only.
    """
    from src.vectorstore import build_vectorstore

    return build_vectorstore(sample_chunks)


@pytest.fixture(autouse=True)
def pace_api_calls(request):
    """Space out live API calls to stay inside Groq's free-tier rate limit.

    The free tier allows 8000 tokens per minute. The integration tests issue
    ten question-answering calls in quick succession, which exceeds that and
    returns HTTP 429. The client retries with backoff, but pacing the tests
    keeps the suite deterministic rather than relying on retries.
    """
    yield
    if request.node.get_closest_marker("integration"):
        time.sleep(4)
