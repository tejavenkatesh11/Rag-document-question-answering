"""Measure retrieval quality against a labelled relevance set.

The README claims K=5 is the right trade-off. This script is the evidence:
it sweeps K and reports standard IR metrics, so the choice rests on numbers
rather than assertion.

Metrics
-------
Recall@K   Fraction of questions whose correct source appears in the top K.
           The metric that matters most here - if the right chunk is not
           retrieved, the LLM cannot possibly answer correctly.
Precision@K Fraction of retrieved chunks that are from the correct source.
           Falls as K grows; that dilution is the cost of higher recall.
MRR        Mean reciprocal rank of the first correct result. Rewards putting
           the right chunk first, not merely somewhere in the list.

No LLM calls, so this runs offline and free.

Usage:
    python scripts/evaluate_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SAMPLE_DOCS_DIR, SIMILARITY_THRESHOLD, SUPPORTED_EXTENSIONS
from src.ingestion import ingest_paths
from src.vectorstore import build_vectorstore, retrieve_with_scores

# Questions paired with the document that actually contains the answer.
LABELLED: list[tuple[str, str]] = [
    ("How many days of annual leave are employees entitled to?", "employee_policy.md"),
    ("How many days of paid sick leave per year?", "employee_policy.md"),
    ("What are the standard working hours?", "employee_policy.md"),
    ("What is the notice period after confirmation?", "employee_policy.md"),
    ("How long must account passwords be?", "it_security_policy.txt"),
    ("How often must passwords be changed?", "it_security_policy.txt"),
    ("Who must lost devices be reported to?", "it_security_policy.txt"),
    ("What are the data classification levels?", "it_security_policy.txt"),
    ("What is the daily meal allowance for metro cities?", "expense_policy.md"),
    ("What is the hotel cap for non-metro locations?", "expense_policy.md"),
    ("When must expense claims be submitted?", "expense_policy.md"),
    ("How long is the probation period?", "onboarding_guide.pdf"),
    ("What mandatory training must new joiners complete?", "onboarding_guide.pdf"),
    ("What equipment is issued to new employees?", "onboarding_guide.pdf"),
    ("What time should new joiners report on day one?", "onboarding_guide.pdf"),
]

# Questions no document answers. Used to measure the false-answer rate of
# Gate 1 - how often the similarity floor lets an unanswerable question
# through to the LLM.
UNANSWERABLE = [
    "What is the capital city of Japan?",
    "What is the company's pet adoption allowance?",
    "What is the maternity leave duration?",
    "How many public holidays are there per year?",
    "What is the parental leave policy?",
]


def evaluate(store, k: int) -> dict:
    hits = 0
    precision_sum = 0.0
    reciprocal_rank_sum = 0.0

    for question, expected in LABELLED:
        results = retrieve_with_scores(store, question, k=k)
        sources = [doc.metadata.get("source") for doc, _ in results]

        if expected in sources:
            hits += 1
            reciprocal_rank_sum += 1.0 / (sources.index(expected) + 1)

        precision_sum += sum(s == expected for s in sources) / len(sources)

    n = len(LABELLED)
    return {
        "k": k,
        "recall": hits / n,
        "precision": precision_sum / n,
        "mrr": reciprocal_rank_sum / n,
    }


def evaluate_gate(store, k: int) -> dict:
    """How well the similarity floor separates answerable from unanswerable."""
    answerable_scores = [
        max(s for _, s in retrieve_with_scores(store, q, k=k))
        for q, _ in LABELLED
    ]
    unanswerable_scores = [
        max(s for _, s in retrieve_with_scores(store, q, k=k))
        for q in UNANSWERABLE
    ]
    return {
        "answerable_min": min(answerable_scores),
        "answerable_max": max(answerable_scores),
        "unanswerable_min": min(unanswerable_scores),
        "unanswerable_max": max(unanswerable_scores),
        # Unanswerable questions the threshold catches on its own, without
        # needing the LLM to decline.
        "blocked_by_threshold": sum(
            s < SIMILARITY_THRESHOLD for s in unanswerable_scores
        ),
        # Answerable questions wrongly refused. Must be zero.
        "false_refusals": sum(
            s < SIMILARITY_THRESHOLD for s in answerable_scores
        ),
    }


def main() -> None:
    paths = sorted(
        p for p in SAMPLE_DOCS_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    print(f"Indexing {len(paths)} document(s)...")
    chunks = ingest_paths(paths)
    store = build_vectorstore(chunks)
    print(f"{len(chunks)} chunks indexed.\n")

    print(f"Retrieval quality over {len(LABELLED)} labelled questions")
    print("=" * 52)
    print(f"{'K':>3}  {'Recall@K':>9}  {'Precision@K':>12}  {'MRR':>7}")
    print("-" * 52)

    rows = [evaluate(store, k) for k in (1, 3, 5, 8, 10)]
    for r in rows:
        print(
            f"{r['k']:>3}  {r['recall']:>8.1%}  {r['precision']:>11.1%}  "
            f"{r['mrr']:>7.3f}"
        )

    print("\nRelevance gate (threshold = %.2f)" % SIMILARITY_THRESHOLD)
    print("=" * 52)
    g = evaluate_gate(store, k=5)
    print(f"Answerable questions scored   {g['answerable_min']:.3f} - "
          f"{g['answerable_max']:.3f}")
    print(f"Unanswerable questions scored {g['unanswerable_min']:.3f} - "
          f"{g['unanswerable_max']:.3f}")
    print(
        f"\nBlocked by threshold alone: {g['blocked_by_threshold']}"
        f"/{len(UNANSWERABLE)} unanswerable"
    )
    print(f"False refusals: {g['false_refusals']}/{len(LABELLED)} answerable")

    if g["answerable_min"] <= g["unanswerable_max"]:
        print(
            "\nNote: the score ranges overlap, so no threshold separates them\n"
            "cleanly. The remaining unanswerable questions are caught by the\n"
            "grounding prompt instead - see README."
        )


if __name__ == "__main__":
    main()
