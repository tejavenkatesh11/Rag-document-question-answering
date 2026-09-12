"""The assessment's required test questions, as executable tests.

Ten questions: 5 answerable from a single document, 2 requiring information
from more than one document, and 3 that the corpus cannot answer.

These hit the live Groq API, so they are marked `integration` and can be
skipped with `pytest -m "not integration"`.
"""

import re
import unicodedata

import pytest


def normalize(text: str) -> str:
    """Fold Unicode punctuation and whitespace to plain ASCII for matching.

    Models routinely emit narrow no-break spaces (U+202F) inside quantities
    ("6 months") and non-breaking hyphens in compounds. Those are
    correct output but break naive substring checks, so both sides of a
    comparison are normalised rather than weakening the assertions.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‑", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_any(haystack: str, needles: list[str]) -> bool:
    hay = normalize(haystack)
    return any(normalize(n) in hay for n in needles)

# --- 5 answerable from a single document --------------------------------
ANSWERABLE = [
    (
        "How many days of annual leave are employees entitled to?",
        ["24"],
        "employee_policy.md",
    ),
    (
        "How long must account passwords be?",
        ["14"],
        "it_security_policy.txt",
    ),
    (
        "What is the daily meal allowance for metro cities?",
        ["1,500", "1500"],
        "expense_policy.md",
    ),
    (
        "How long is the probation period for new employees?",
        ["6 month", "six month"],
        "onboarding_guide.pdf",
    ),
    (
        "How many days of paid sick leave are employees entitled to?",
        ["12"],
        "employee_policy.md",
    ),
]

# --- 2 requiring multiple documents -------------------------------------
# Each fact lives in a different file, so a correct answer proves the
# retriever pulled from more than one source.
MULTI_DOC = [
    (
        "What are the rules for remote work, including any security "
        "requirements for accessing internal systems?",
        ["3 day", "three day"],
        ["VPN"],
    ),
    (
        "What is the reimbursement deadline in the employee handbook, and "
        "what does the travel policy say happens to claims older than 60 days?",
        ["30 day", "thirty day"],
        ["not be reimbursed", "will not be", "no longer"],
    ),
]

# --- 3 unanswerable ------------------------------------------------------
# Deliberately varied in difficulty: wholly off-topic, plausible-but-absent,
# and semantically adjacent to real content (the hardest case).
UNANSWERABLE = [
    "What is the capital city of Japan?",
    "What is the company's pet adoption allowance?",
    "What is the maternity leave duration?",
]


@pytest.mark.integration
@pytest.mark.parametrize("question,expected,source", ANSWERABLE)
def test_answerable_questions(rag_store, question, expected, source):
    """Each answer states the right fact and cites the right document."""
    from src.rag_chain import answer_question

    response = answer_question(rag_store, question)

    assert response.grounded, f"Refused an answerable question: {question}"
    assert contains_any(response.answer, expected), (
        f"Expected one of {expected} in answer: {response.answer}"
    )
    cited = [s["source"] for s in response.sources]
    assert source in cited, f"Expected {source} in citations, got {cited}"


@pytest.mark.integration
@pytest.mark.parametrize("question,fact_a,fact_b", MULTI_DOC)
def test_multi_document_questions(rag_store, question, fact_a, fact_b):
    """Answers combine facts that live in different documents."""
    from src.rag_chain import answer_question

    response = answer_question(rag_store, question, k=6)

    assert response.grounded, f"Refused a multi-document question: {question}"
    assert contains_any(response.answer, fact_a), (
        f"Missing {fact_a} in: {response.answer}"
    )
    assert contains_any(response.answer, fact_b), (
        f"Missing {fact_b} in: {response.answer}"
    )
    assert len({s["source"] for s in response.sources}) >= 2, (
        f"Expected citations from 2+ documents, got "
        f"{[s['source'] for s in response.sources]}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("question", UNANSWERABLE)
def test_unanswerable_questions(rag_store, question):
    """The system declines rather than inventing an answer."""
    from src.rag_chain import NO_ANSWER, answer_question

    response = answer_question(rag_store, question)

    assert not response.grounded, (
        f"Should not have answered {question!r}: {response.answer}"
    )
    assert normalize(NO_ANSWER) in normalize(response.answer)
    assert response.sources == [], (
        "A refusal must not cite sources - that would imply false support."
    )
