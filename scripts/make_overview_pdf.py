"""Generate the technical overview PDF.

A standalone walkthrough of how the application was built, what was used and
why. Intended for reading away from the repository - viva preparation, or
handing to a reviewer who wants the reasoning without reading the code.

The output is deliberately NOT committed (see .gitignore).

Usage:
    python scripts/make_overview_pdf.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent.parent / "RAG_Application_Technical_Overview.pdf"

ACCENT = colors.HexColor("#1a4d7a")
MUTED = colors.HexColor("#5a6b7a")
RULE = colors.HexColor("#c8d4de")
BG = colors.HexColor("#f2f6f9")

styles = getSampleStyleSheet()

H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=17, leading=21,
    textColor=ACCENT, spaceBefore=2, spaceAfter=10,
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=12.5, leading=16,
    textColor=ACCENT, spaceBefore=13, spaceAfter=6,
)
BODY = ParagraphStyle(
    "BODY", parent=styles["BodyText"], fontSize=9.7, leading=14.5,
    alignment=TA_LEFT, spaceAfter=7,
)
SMALL = ParagraphStyle(
    "SMALL", parent=BODY, fontSize=8.6, leading=12.5, textColor=MUTED,
)
CODE = ParagraphStyle(
    "CODE", parent=BODY, fontName="Courier", fontSize=8.2, leading=11.5,
    backColor=BG, borderPadding=6, leftIndent=4, spaceAfter=8,
)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.7, leading=12, spaceAfter=0)
CELL_H = ParagraphStyle(
    "CELL_H", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white,
)


def table(rows, widths):
    data = [[Paragraph(c, CELL_H) for c in rows[0]]] + [
        [Paragraph(c, CELL) for c in r] for r in rows[1:]
    ]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def build() -> None:
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, title="RAG Application - Technical Overview",
        author="Technical Assessment Submission",
        leftMargin=1.9 * cm, rightMargin=1.9 * cm,
        topMargin=1.7 * cm, bottomMargin=1.7 * cm,
    )
    W = doc.width
    f = []

    # ================= Title =================
    f.append(Paragraph("RAG-Based Document Question Answering", H1))
    f.append(Paragraph("Technical Overview: How It Was Built and Why", BODY))
    f.append(Spacer(1, 0.15 * cm))
    f.append(Paragraph(
        "A complete account of the architecture, the library choices and the "
        "reasoning behind them, including the measurements that drove the "
        "engineering decisions and the bugs found along the way.", SMALL))
    f.append(Spacer(1, 0.45 * cm))

    # ================= 1. What was built =================
    f.append(Paragraph("1. What Was Built", H2))
    f.append(Paragraph(
        "A Retrieval-Augmented Generation system that answers questions from "
        "user-supplied documents, cites the source of every answer, and "
        "explicitly declines when the documents do not contain the "
        "information. It supports PDF, plain text and Markdown, runs as a "
        "Streamlit web app, and handles conversational follow-up questions.",
        BODY))
    f.append(Paragraph(
        "The pipeline: <b>documents &rarr; text extraction &rarr; chunking "
        "&rarr; embeddings &rarr; vector store &rarr; hybrid retrieval &rarr; "
        "LLM &rarr; answer with citations</b>.", BODY))

    # ================= 2. Stack =================
    f.append(Paragraph("2. The Stack, and Why Each Piece", H2))
    f.append(Paragraph(
        "Every choice below was made under one constraint: the only API key "
        "available was a Groq key. That single fact shaped the embedding "
        "decision, which in turn shaped everything else.", BODY))
    f.append(table([
        ["Layer", "Choice", "Why this one"],
        ["Language", "Python 3.12",
         "Every mature RAG library targets it. 3.12 has wheels for both "
         "PyTorch and FAISS, avoiding source builds."],
        ["Orchestration", "LangChain 1.4",
         "Its Document object carries metadata (source file, page number) "
         "through every stage untouched. That metadata <i>is</i> the citation "
         "feature - building it by hand would mean threading a parallel data "
         "structure through the whole pipeline. Also supplies EnsembleRetriever "
         "for hybrid search."],
        ["PDF extraction", "pypdf (PyPDFLoader)",
         "Returns one Document per page rather than one string per file, so "
         "page numbers survive into citations. A whole-file extractor would "
         "make 'Page 8' impossible to recover."],
        ["Embeddings", "all-MiniLM-L6-v2<br/>(sentence-transformers)",
         "<b>Groq serves no embedding models</b>, so a hosted embedder would "
         "have meant a second provider and a second API key. This model runs "
         "locally on CPU, is ~80MB, and produces 384-dimensional vectors - "
         "small, fast, and strong on short-passage retrieval."],
        ["Vector store", "FAISS (faiss-cpu)",
         "In-process, no server to run, persists to two files on disk. For a "
         "corpus this size a hosted vector DB would add operational weight "
         "and zero capability."],
        ["Sparse retrieval", "BM25 (rank-bm25)",
         "Embeddings represent rare literal strings poorly. A query for the "
         "policy code 'HR-114' needs keyword matching. BM25 covers exactly "
         "the cases dense retrieval is weakest at."],
        ["LLM", "Groq, openai/gpt-oss-120b",
         "Fast inference on a free tier. The model list was queried from the "
         "live API rather than assumed - the Llama 3.3 model chosen initially "
         "was no longer served, which would have been a runtime failure."],
        ["Interface", "Streamlit",
         "Chat UI, file upload and session state in ~240 lines. The "
         "assessment marks the pipeline, not the frontend."],
        ["Testing", "pytest",
         "Splits into fast offline unit tests and live-API integration tests "
         "via markers, so the fast suite stays usable during development."],
    ], [2.6 * cm, 3.7 * cm, W - 6.3 * cm]))

    f.append(PageBreak())

    # ================= 3. Decisions =================
    f.append(Paragraph("3. Engineering Decisions in Detail", H2))

    f.append(Paragraph("<b>3.1 Chunking: 800 characters, 120 overlap</b>", BODY))
    f.append(Paragraph(
        "Roughly 150-200 tokens. Large enough to hold a complete policy clause "
        "so a single retrieved chunk usually contains the whole answer; small "
        "enough that the chunk is mostly signal. Larger chunks (1500+) dilute "
        "the embedding - a chunk spanning four topics matches each of them "
        "weakly, which degrades retrieval precision.", BODY))
    f.append(Paragraph(
        "The 120-character overlap (15%) exists because a fact straddling a "
        "chunk boundary would otherwise be retrievable from neither side. "
        "A recursive splitter is used so splits fall on paragraph breaks "
        "first, then line breaks, then sentences - chunks align with the "
        "document's own structure rather than arbitrary character counts.",
        BODY))

    f.append(Paragraph("<b>3.2 Hybrid retrieval, Top-K = 5</b>", BODY))
    f.append(Paragraph(
        "Two retrievers run in parallel and their rankings are fused by "
        "reciprocal rank fusion, weighted 0.6 dense / 0.4 sparse. Dense "
        "matches meaning, so \"time off\" finds \"annual leave\". BM25 matches "
        "literal tokens, so \"HR-114\" finds the right clause. Dense is "
        "weighted higher because most questions are paraphrases; BM25 covers "
        "the minority that hinge on an exact term.", BODY))
    f.append(Paragraph(
        "K was chosen by measurement, not intuition "
        "(<font face='Courier' size='8'>scripts/evaluate_retrieval.py</font>, "
        "15 labelled question/source pairs):", BODY))
    f.append(table([
        ["K", "Recall@K", "Precision@K", "MRR"],
        ["1", "93.3%", "93.3%", "0.933"],
        ["3", "<b>100.0%</b>", "57.8%", "0.967"],
        ["5", "100.0%", "44.0%", "0.967"],
        ["8", "100.0%", "30.8%", "0.967"],
        ["10", "100.0%", "26.0%", "0.967"],
    ], [1.6 * cm, 3.3 * cm, 3.6 * cm, W - 8.5 * cm]))
    f.append(Spacer(1, 0.25 * cm))
    f.append(Paragraph(
        "<b>This measurement contradicted an earlier assumption.</b> The "
        "README initially claimed K=2-3 missed facts spanning chunk "
        "boundaries. It does not: recall saturates at 100% from K=3, and the "
        "single question missed at K=1 is retrieved at rank 2. K=5 is "
        "nevertheless retained, for stated reasons rather than the disproved "
        "one - this corpus is only 11 chunks, so recall saturates almost "
        "immediately, and K=5 leaves headroom for larger corpora while "
        "supplying the extra context that multi-document questions need.",
        BODY))

    f.append(Paragraph("<b>3.3 Hallucination handling: two gates</b>", BODY))
    f.append(Paragraph(
        "This is the part of the system with the most interesting result, and "
        "the one most worth understanding.", BODY))
    f.append(Paragraph(
        "<b>Gate 1 - similarity floor.</b> Before the LLM is called, the best "
        "retrieved chunk's cosine similarity is compared against a threshold. "
        "If nothing clears it, a fixed \"not found\" answer is returned "
        "without any API call. A model that is never invoked cannot "
        "hallucinate, and the refusal costs no tokens and returns in ~0.03s.",
        BODY))
    f.append(Paragraph(
        "<b>Gate 2 - grounding prompt.</b> The system prompt restricts the "
        "model to the supplied excerpts, forbids outside knowledge, specifies "
        "the exact refusal string, and runs at temperature 0.", BODY))
    f.append(KeepTogether([
        Paragraph(
            "<b>Why both are necessary.</b> Measuring the score distributions "
            "showed they nearly touch:", BODY),
        table([
            ["Question type", "Best cosine similarity"],
            ["Answerable (15 questions)", "0.371 - 0.784"],
            ["Unanswerable (5 questions)", "0.022 - <b>0.504</b>"],
        ], [7 * cm, W - 7 * cm]),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "\"What is the maternity leave duration?\" scores <b>0.504</b> - "
            "it is semantically adjacent to the leave sections even though no "
            "document answers it. Some genuine questions score barely above "
            "that. <b>No threshold separates the two sets cleanly.</b> Raising "
            "it to catch maternity leave would start refusing real questions.",
            BODY),
    ]))
    f.append(Paragraph(
        "So the threshold is deliberately set <i>low</i> (0.15) as a coarse "
        "filter that only rejects the obviously off-topic - a question about "
        "Japan scores 0.022. It catches 1 of 5 unanswerable questions with "
        "zero false refusals. The other 4 are caught by the grounding prompt, "
        "whose reading of the context is the better judge. Hallucination "
        "handling is the <i>combination</i>; neither gate is sufficient alone.",
        BODY))

    f.append(PageBreak())

    f.append(Paragraph("<b>3.4 Citations</b>", BODY))
    f.append(Paragraph(
        "Source and page metadata are attached at ingestion and carried "
        "through to the answer. Citations render as "
        "<font face='Courier' size='8'>onboarding_guide.pdf - Page 2</font>, "
        "or filename alone for formats without pages. Two details matter:",
        BODY))
    f.append(Paragraph(
        "<b>Only cited sources are listed.</b> Retrieval returns 5 chunks but "
        "an answer usually rests on one or two. An early version listed all "
        "five, which attached authoritative-looking citations to documents "
        "that contributed nothing - an annual-leave answer \"cited\" the "
        "expense policy. The list is now narrowed to sources the answer "
        "actually references.", BODY))
    f.append(Paragraph(
        "<b>Refusals cite nothing.</b> Attaching sources to \"I could not find "
        "that\" would imply evidence that does not exist.", BODY))

    f.append(Paragraph("<b>3.5 Conversational follow-ups</b>", BODY))
    f.append(Paragraph(
        "\"What about part-time employees?\" contains almost no retrievable "
        "content and embeds poorly on its own. The last three conversation "
        "turns are used to rewrite it into a standalone question - \"How many "
        "days of annual leave are part-time employees entitled to?\" - which "
        "retrieves correctly. The rewritten form is shown in the UI so the "
        "interpretation is visible rather than hidden. Rewriting is "
        "best-effort: on failure the original question is used.", BODY))

    # ================= 4. Bugs =================
    f.append(Paragraph("4. Bugs Found and Fixed", H2))
    f.append(Paragraph(
        "Recorded because the reasoning matters more than the fixes.", SMALL))
    f.append(table([
        ["Bug", "Cause and fix"],
        ["Negative similarity scores",
         "FAISS returns <i>squared L2 distance</i>, not similarity. LangChain's "
         "<font face='Courier' size='7.5'>similarity_search_with_relevance_"
         "scores</font> assumes an unnormalised space and emitted negative "
         "values, so the threshold was calibrated against a meaningless scale. "
         "Fixed by converting explicitly: with normalised vectors, "
         "<font face='Courier' size='7.5'>cos = 1 - d/2</font>."],
        ["Irrelevant sources cited",
         "All K retrieved chunks were listed as sources regardless of whether "
         "the answer used them. Narrowed to sources the answer references."],
        ["Invented page numbers",
         "The model cited \"Page 2\" for a Markdown file with no pages. The "
         "prompt now requires copying the excerpt's label verbatim and "
         "forbids inventing page numbers."],
        ["Unhandled rate limits",
         "Groq's free tier caps at 8000 tokens/minute; several questions in "
         "quick succession returned an unhandled HTTP 429. Added client retry "
         "with backoff plus a message telling the user to wait."],
        ["Tests failed on correct answers",
         "The model emits narrow no-break spaces (U+202F) inside quantities, "
         "so \"6 months\" never matched \"6 months\". Tests now normalise "
         "Unicode before matching rather than weakening the assertions."],
    ], [4.6 * cm, W - 4.6 * cm]))

    # ================= 5. Testing =================
    f.append(Paragraph("5. Verification", H2))
    f.append(Paragraph(
        "<b>38 tests, all passing.</b> 28 unit tests run offline and cover "
        "page indexing, metadata propagation, chunk sizing, the L2-to-cosine "
        "conversion, score ordering, BM25 exact-token matching, persistence, "
        "citation formatting and error handling. 10 integration tests are the "
        "questions the assessment requires: 5 answerable, 2 requiring multiple "
        "documents, 3 unanswerable.", BODY))
    f.append(Paragraph(
        "The two multi-document questions are constructed so each half of the "
        "answer lives in a different file, and they assert citations from two "
        "or more sources - otherwise they would not actually test "
        "cross-document retrieval.", BODY))
    f.append(Paragraph(
        "A separate evaluation harness reports Recall@K, Precision@K and MRR "
        "so retrieval quality is measured rather than asserted.", BODY))

    # ================= 6. Limitations =================
    f.append(Paragraph("6. Known Limitations", H2))
    f.append(table([
        ["Limitation", "Detail"],
        ["No OCR", "Scanned image PDFs yield no text. The UI reports this "
                   "rather than failing silently."],
        ["FAISS is single-node", "In-memory, no concurrent writes. A server "
                                 "store (Qdrant, pgvector) would be next for "
                                 "multi-user deployment."],
        ["All-or-nothing indexing", "Adding a document rebuilds the entire "
                                    "index."],
        ["No reranking", "A cross-encoder over retrieved candidates would "
                         "likely improve precision at some latency cost. Not "
                         "added because recall is already 100% on this corpus."],
        ["Corpus-dependent threshold", "0.15 was calibrated on these "
                                       "documents. A different corpus needs "
                                       "recalibration; the measurement method "
                                       "matters more than the number."],
        ["Table extraction", "pypdf flattens tables and multi-column layouts "
                             "to linear text, which can scramble reading "
                             "order."],
    ], [4.6 * cm, W - 4.6 * cm]))

    f.append(Spacer(1, 0.4 * cm))
    f.append(Paragraph(
        "Assessment level: <b>Gold</b> (hybrid search, query rewriting, "
        "similarity thresholds, retrieval evaluation) with substantial "
        "Platinum elements (38 tests, metrics, logging, latency tracking, "
        "Docker). The conversational bonus challenge is implemented.", BODY))

    doc.build(f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
