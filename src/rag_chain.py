"""Answer generation grounded in retrieved context.

Grounding is enforced at two levels:

  1. Before the LLM - if no retrieved chunk clears the similarity floor, the
     model is never called and a fixed "not found" answer is returned. A model
     that is never asked cannot hallucinate.
  2. Inside the prompt - explicit instructions to answer only from context and
     to decline when the context does not cover the question.
"""

import logging
import time
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import (
    GROQ_API_KEY,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K,
)
from src.vectorstore import is_relevant, retrieve_with_scores

logger = logging.getLogger(__name__)

NO_ANSWER = (
    "I could not find that information in the provided documents."
)

SYSTEM_PROMPT = """You are a document question-answering assistant. You answer \
strictly from the excerpts provided to you.

Rules:
1. Use ONLY the information in the context below. Do not use outside knowledge.
2. If the context does not contain the answer, reply exactly: "{no_answer}"
3. Do not guess, infer beyond what is written, or fill gaps with plausible detail.
4. Keep the answer concise and factual.
5. When the answer draws on several excerpts, combine them into one coherent reply.
6. Cite the source inline after each fact, copying the bracketed label exactly \
as it appears above that excerpt. Never invent a page number - if an excerpt's \
label has no page, cite it without one.
"""

USER_PROMPT = """Context excerpts:
---
{context}
---

Question: {question}

Answer using only the context above."""

# For follow-up questions, a pronoun-laden query ("what about part-timers?")
# embeds poorly on its own. Rewriting it into a standalone question before
# retrieval is what makes conversational RAG actually retrieve the right thing.
REWRITE_PROMPT = """Given the conversation history and a follow-up question, \
rewrite the follow-up as a standalone question that can be understood without \
the history. Preserve the original meaning and all specifics. If it is already \
standalone, return it unchanged. Return only the rewritten question.

History:
{history}

Follow-up question: {question}

Standalone question:"""


@dataclass
class RAGResponse:
    """An answer plus everything needed to audit it."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    contexts: list[Document] = field(default_factory=list)
    latency_seconds: float = 0.0
    grounded: bool = True
    rewritten_query: str | None = None


def get_llm() -> ChatGroq:
    """Construct the Groq chat model."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        api_key=GROQ_API_KEY,
        # Groq's free tier allows 8000 tokens per minute. Several questions in
        # quick succession exceed that and return HTTP 429. The client honours
        # the Retry-After header and backs off, which turns a hard failure into
        # a short wait.
        max_retries=LLM_MAX_RETRIES,
    )


def format_context(docs: list[Document]) -> str:
    """Render chunks for the prompt, labelled so the model can cite them."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source}, Page {page}" if page else source
        parts.append(f"[Source: {label}]\n{doc.page_content}")
    return "\n\n".join(parts)


def extract_sources(docs: list[Document], answer: str = "") -> list[dict]:
    """Deduplicate retrieved chunks down to a list of citations.

    Retrieval returns K chunks, but the answer typically rests on only one or
    two. Listing all K would attach authoritative-looking citations to
    documents that contributed nothing, so when the answer cites sources
    explicitly, the list is narrowed to those actually referenced. If no
    citation is detectable, all retrieved chunks are returned so the user can
    still audit the evidence.
    """
    cited = [d for d in docs if d.metadata.get("source", "") in answer]
    docs = cited or docs

    seen: set[tuple[str, int | None]] = set()
    sources: list[dict] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": source,
                "page": page,
                "citation": f"{source} - Page {page}" if page else source,
                "excerpt": doc.page_content[:300].strip(),
            }
        )
    return sources


def rewrite_query(question: str, history: list[tuple[str, str]]) -> str:
    """Turn a follow-up into a standalone question using chat history."""
    if not history:
        return question

    # Only the last few turns matter, and they keep the prompt small.
    recent = history[-3:]
    history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in recent)

    try:
        prompt = REWRITE_PROMPT.format(history=history_text, question=question)
        rewritten = get_llm().invoke(prompt).content.strip()
        if rewritten and rewritten.lower() != question.lower():
            logger.info("Rewrote query: %r -> %r", question, rewritten)
        return rewritten or question
    except Exception:  # noqa: BLE001 - rewriting is an optimisation, not core
        logger.exception("Query rewrite failed; using original question")
        return question


def answer_question(
    store,
    question: str,
    k: int = TOP_K,
    history: list[tuple[str, str]] | None = None,
) -> RAGResponse:
    """Run the full retrieve-then-generate pipeline for one question."""
    start = time.perf_counter()

    if not question.strip():
        return RAGResponse(
            answer="Please enter a question.", grounded=False
        )

    search_query = (
        rewrite_query(question, history) if history else question
    )
    rewritten = search_query if search_query != question else None

    scored = retrieve_with_scores(store, search_query, k=k)

    # Gate 1: nothing retrieved clears the relevance floor, so do not call the
    # LLM at all. This is the strongest possible hallucination guard.
    if not is_relevant(scored):
        logger.info("No chunk cleared the relevance threshold for %r", question)
        return RAGResponse(
            answer=NO_ANSWER,
            latency_seconds=time.perf_counter() - start,
            grounded=False,
            rewritten_query=rewritten,
        )

    docs = [doc for doc, _ in scored]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT.format(no_answer=NO_ANSWER)),
            ("human", USER_PROMPT),
        ]
    )

    try:
        chain = prompt | get_llm()
        result = chain.invoke(
            {"context": format_context(docs), "question": question}
        )
        answer = result.content.strip()
    except Exception as exc:  # noqa: BLE001 - surface API errors to the UI
        logger.exception("LLM call failed")
        # Rate limiting is the most likely failure on Groq's free tier, and it
        # is recoverable, so it gets a message that says what to do rather than
        # an opaque stack-trace string.
        if "rate_limit" in str(exc) or "429" in str(exc):
            message = (
                "Rate limit reached on the Groq free tier (8000 tokens per "
                "minute). Please wait a moment and ask again."
            )
        else:
            message = f"The language model could not be reached: {exc}"
        return RAGResponse(
            answer=message,
            latency_seconds=time.perf_counter() - start,
            grounded=False,
            rewritten_query=rewritten,
        )

    # Gate 2: the model declined despite retrieval clearing the floor. Do not
    # attach citations to a non-answer - that would imply false support.
    declined = NO_ANSWER.lower() in answer.lower()

    elapsed = time.perf_counter() - start
    logger.info("Answered in %.2fs (declined=%s)", elapsed, declined)

    return RAGResponse(
        answer=answer,
        sources=[] if declined else extract_sources(docs, answer),
        contexts=docs,
        latency_seconds=elapsed,
        grounded=not declined,
        rewritten_query=rewritten,
    )
