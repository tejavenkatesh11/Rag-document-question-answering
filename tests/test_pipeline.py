"""Unit tests for ingestion, chunking and retrieval.

These do not call the LLM, so they run fast and offline (after the embedding
model is cached).
"""

import pytest

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion import (
    UnsupportedFileTypeError,
    chunk_documents,
    ingest_paths,
    load_document,
)


class TestIngestion:
    def test_loads_every_supported_format(self, sample_paths):
        suffixes = {p.suffix.lower() for p in sample_paths}
        assert ".pdf" in suffixes, "Corpus must include a PDF"
        assert suffixes & {".txt", ".md"}, "Corpus needs a second text format"

    def test_pdf_pages_are_one_indexed(self, sample_paths):
        pdf = next(p for p in sample_paths if p.suffix.lower() == ".pdf")
        docs = load_document(pdf)
        pages = [d.metadata["page"] for d in docs]
        assert min(pages) == 1, "Page numbers shown to users must start at 1"
        assert pages == sorted(pages)

    def test_text_files_have_no_page_number(self, sample_paths):
        txt = next(p for p in sample_paths if p.suffix.lower() in {".txt", ".md"})
        docs = load_document(txt)
        assert all(d.metadata["page"] is None for d in docs)

    def test_every_chunk_carries_its_source(self, sample_chunks):
        for chunk in sample_chunks:
            assert chunk.metadata.get("source"), "Citation needs a source"
            assert "chunk_id" in chunk.metadata

    def test_rejects_unsupported_extension(self, tmp_path):
        bad = tmp_path / "data.csv"
        bad.write_text("a,b\n1,2\n")
        with pytest.raises(UnsupportedFileTypeError):
            load_document(bad)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "nope.txt")

    def test_one_bad_file_does_not_abort_the_batch(self, sample_paths, tmp_path):
        bad = tmp_path / "broken.csv"
        bad.write_text("x")
        chunks = ingest_paths([*sample_paths, bad])
        assert chunks, "Good documents should still be ingested"

    def test_all_bad_files_raises(self, tmp_path):
        bad = tmp_path / "broken.csv"
        bad.write_text("x")
        with pytest.raises(RuntimeError):
            ingest_paths([bad])


class TestChunking:
    def test_respects_configured_size(self, sample_chunks):
        # The splitter may exceed the limit slightly when a single separator
        # span is long, so allow a small margin.
        oversized = [c for c in sample_chunks if len(c.page_content) > CHUNK_SIZE * 1.2]
        assert not oversized, f"{len(oversized)} chunk(s) far exceed CHUNK_SIZE"

    def test_empty_input_gives_empty_output(self):
        assert chunk_documents([]) == []

    def test_overlap_is_smaller_than_chunk(self):
        assert CHUNK_OVERLAP < CHUNK_SIZE


class TestRetrieval:
    def test_relevant_query_scores_above_threshold(self, rag_store):
        from src.config import SIMILARITY_THRESHOLD
        from src.vectorstore import is_relevant, retrieve_with_scores

        scored = retrieve_with_scores(rag_store, "How much annual leave?")
        assert scored
        best = max(s for _, s in scored)
        assert best > SIMILARITY_THRESHOLD
        assert is_relevant(scored)

    def test_scores_are_valid_cosine_similarities(self, rag_store):
        """Guards the L2-to-cosine conversion in retrieve_with_scores."""
        from src.vectorstore import retrieve_with_scores

        scored = retrieve_with_scores(rag_store, "annual leave")
        for _, score in scored:
            assert -1.0 <= score <= 1.0, f"{score} is not a cosine similarity"

    def test_off_topic_query_is_rejected(self, rag_store):
        from src.vectorstore import is_relevant, retrieve_with_scores

        scored = retrieve_with_scores(rag_store, "What is the capital of Japan?")
        assert not is_relevant(scored)

    def test_scores_descend(self, rag_store):
        from src.vectorstore import retrieve_with_scores

        scores = [s for _, s in retrieve_with_scores(rag_store, "password rules")]
        assert scores == sorted(scores, reverse=True)

    def test_k_controls_result_count(self, rag_store):
        from src.vectorstore import retrieve_with_scores

        assert len(retrieve_with_scores(rag_store, "leave", k=2)) == 2
        assert len(retrieve_with_scores(rag_store, "leave", k=4)) == 4

    def test_hybrid_retriever_returns_documents(self, rag_store, sample_chunks):
        from src.vectorstore import build_hybrid_retriever

        retriever = build_hybrid_retriever(rag_store, sample_chunks)
        docs = retriever.invoke("password length requirements")
        assert docs
        assert any("14" in d.page_content for d in docs)

    def test_bm25_finds_exact_codes_semantic_search_may_miss(
        self, rag_store, sample_chunks
    ):
        """The reason for hybrid search: rare literal tokens."""
        from src.vectorstore import build_hybrid_retriever

        retriever = build_hybrid_retriever(rag_store, sample_chunks)
        docs = retriever.invoke("HR-114")
        assert any("HR-114" in d.page_content for d in docs)


class TestPersistence:
    def test_roundtrip_preserves_vectors(self, sample_chunks, tmp_path):
        from src.vectorstore import (
            build_vectorstore,
            load_vectorstore,
            save_vectorstore,
        )

        store = build_vectorstore(sample_chunks)
        save_vectorstore(store, tmp_path)
        reloaded = load_vectorstore(tmp_path)
        assert reloaded is not None
        assert reloaded.index.ntotal == store.index.ntotal

    def test_missing_store_returns_none(self, tmp_path):
        from src.vectorstore import load_vectorstore

        assert load_vectorstore(tmp_path / "absent") is None


class TestFormatting:
    def test_citation_includes_page_for_pdf(self):
        from langchain_core.documents import Document

        from src.rag_chain import extract_sources

        doc = Document(page_content="x", metadata={"source": "a.pdf", "page": 8})
        assert extract_sources([doc])[0]["citation"] == "a.pdf - Page 8"

    def test_citation_omits_page_for_text(self):
        from langchain_core.documents import Document

        from src.rag_chain import extract_sources

        doc = Document(page_content="x", metadata={"source": "a.md", "page": None})
        assert extract_sources([doc])[0]["citation"] == "a.md"

    def test_duplicate_sources_collapse(self):
        from langchain_core.documents import Document

        from src.rag_chain import extract_sources

        docs = [
            Document(page_content="a", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="b", metadata={"source": "a.pdf", "page": 1}),
        ]
        assert len(extract_sources(docs)) == 1

    def test_sources_narrow_to_those_cited(self):
        """Documents the answer never mentions must not be cited."""
        from langchain_core.documents import Document

        from src.rag_chain import extract_sources

        docs = [
            Document(page_content="a", metadata={"source": "used.md", "page": None}),
            Document(page_content="b", metadata={"source": "unused.md", "page": None}),
        ]
        sources = extract_sources(docs, answer="See [used.md] for details.")
        assert [s["source"] for s in sources] == ["used.md"]

    def test_uncited_answer_keeps_all_sources(self, ):
        """With no detectable citation, keep everything so it stays auditable."""
        from langchain_core.documents import Document

        from src.rag_chain import extract_sources

        docs = [
            Document(page_content="a", metadata={"source": "one.md", "page": None}),
            Document(page_content="b", metadata={"source": "two.md", "page": None}),
        ]
        assert len(extract_sources(docs, answer="No citation here.")) == 2

    def test_context_labels_each_excerpt(self):
        from langchain_core.documents import Document

        from src.rag_chain import format_context

        docs = [
            Document(page_content="body", metadata={"source": "a.pdf", "page": 3})
        ]
        assert "[Source: a.pdf, Page 3]" in format_context(docs)


class TestEdgeCases:
    def test_empty_question_is_handled(self, rag_store):
        from src.rag_chain import answer_question

        response = answer_question(rag_store, "   ")
        assert not response.grounded
        assert "enter a question" in response.answer.lower()

    def test_empty_store_rejected(self):
        from src.vectorstore import build_vectorstore

        with pytest.raises(ValueError):
            build_vectorstore([])
