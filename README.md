# RAG-Based Document Question Answering

A Retrieval-Augmented Generation system that answers questions from your own
documents, cites where each answer came from, and says "I don't know" instead
of guessing when the documents don't cover the question.

Built with LangChain, FAISS, local sentence-transformers embeddings and Groq.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Technology choices](#technology-choices)
- [Chunking strategy](#chunking-strategy)
- [Retrieval strategy](#retrieval-strategy)
- [Grounding and hallucination handling](#grounding-and-hallucination-handling)
- [Citations](#citations)
- [Conversational follow-ups](#conversational-follow-ups)
- [Retrieval evaluation](#retrieval-evaluation)
- [Testing](#testing)
- [Docker](#docker)
- [Sample documents](#sample-documents)
- [Limitations](#limitations)

---

## What it does

1. Ingests PDF, TXT and Markdown documents
2. Extracts text, keeping page numbers for PDFs
3. Splits text into overlapping chunks
4. Embeds each chunk locally and indexes it in FAISS
5. Retrieves relevant chunks for a question using hybrid (semantic + keyword) search
6. Generates an answer grounded strictly in those chunks, with citations
7. Declines to answer when the documents don't contain the information

---

## Architecture

```
                      ┌─────────────────────────┐
                      │  PDF / TXT / Markdown   │
                      └────────────┬────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │   Text extraction       │   pypdf / TextLoader
                      │   (+ page metadata)     │
                      └────────────┬────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │   Chunking              │   Recursive splitter
                      │   800 chars / 120 over. │
                      └────────────┬────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │   Embeddings            │   all-MiniLM-L6-v2 (local)
                      │   384-dim, normalised   │
                      └────────────┬────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │   FAISS index           │   persisted to disk
                      └────────────┬────────────┘
                                   │
   Question ──► Query rewrite ──►  │
                (if follow-up)     │
                                   ▼
                  ┌────────────────────────────────┐
                  │  Hybrid retrieval              │
                  │  dense (0.6) + BM25 (0.4)      │
                  └────────────┬───────────────────┘
                               │
                  ┌────────────▼───────────────────┐
                  │  Gate 1: similarity threshold  │──► below floor?
                  └────────────┬───────────────────┘     "not found"
                               │                          (no LLM call)
                  ┌────────────▼───────────────────┐
                  │  LLM (Groq, temperature 0)     │
                  │  Gate 2: grounding prompt      │──► context insufficient?
                  └────────────┬───────────────────┘     "not found"
                               │
                      ┌────────▼────────┐
                      │ Answer + Sources│
                      └─────────────────┘
```

### Project layout

```
├── app.py                      Streamlit interface
├── src/
│   ├── config.py               All tunable parameters, with rationale
│   ├── ingestion.py            Loading, extraction, chunking
│   ├── vectorstore.py          Embeddings, FAISS, hybrid retrieval
│   └── rag_chain.py            Prompting, generation, citations
├── tests/
│   ├── test_pipeline.py        28 unit tests (no API needed)
│   └── test_questions.py       The 10 required assessment questions
├── scripts/
│   ├── make_sample_pdf.py      Regenerates the sample PDF
│   └── evaluate_retrieval.py   Recall@K / Precision@K / MRR harness
├── Dockerfile
└── data/sample_docs/           4 sample documents
```

---

## Setup

Requires **Python 3.10+** (developed on 3.12).

**1. Clone and create a virtual environment**

```bash
git clone <repository-url>
cd RAG-based-Technical-Assement

python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

**2. Install PyTorch (CPU build) first**

Installing torch from the CPU index avoids downloading the much larger CUDA
build, which this project does not need:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**3. Install everything else**

```bash
pip install -r requirements.txt
```

**4. Configure your API key**

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Then edit `.env` and set your key. A free key is available at
[console.groq.com](https://console.groq.com):

```
GROQ_API_KEY=gsk_your_key_here
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key for the LLM |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Generation model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `TOP_K` | `5` | Chunks retrieved per question |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between chunks |
| `SIMILARITY_THRESHOLD` | `0.15` | Relevance floor (see below) |
| `DENSE_WEIGHT` / `BM25_WEIGHT` | `0.6` / `0.4` | Hybrid fusion weights |
| `LLM_MAX_RETRIES` | `3` | Retries on rate-limit / transient API errors |

Only `GROQ_API_KEY` is required; every other value has a working default.

---

## Running the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

1. Click **Load sample documents** in the sidebar (or upload your own)
2. Ask a question in the chat box
3. Expand **Sources** under any answer to see the citations and excerpts

The first run downloads the embedding model (~80 MB), which takes a few
seconds. It is cached afterwards.

---

## Technology choices

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangChain** | Document/metadata model carries source and page through the whole pipeline, which is what makes citation straightforward; `EnsembleRetriever` provides hybrid fusion directly. |
| Extraction | **pypdf** via `PyPDFLoader` | Returns one document per page, so page numbers survive into citations. |
| Embeddings | **all-MiniLM-L6-v2**, local | Runs on CPU with no API key. Groq serves no embedding models, so a hosted alternative would mean a second provider and a second key. 384 dimensions keeps the index small and search fast, and quality is strong for short-passage retrieval. |
| Vector store | **FAISS** | In-process, no server to run, and persists to disk as two files. Appropriate for a corpus of this size; see [Limitations](#limitations). |
| Sparse retrieval | **BM25** (`rank-bm25`) | Complements embeddings on exact tokens — IDs, acronyms, policy codes. |
| LLM | **Groq**, `openai/gpt-oss-120b`, temperature 0 | Fast inference on a free tier. Temperature 0 because this is extraction, not creative writing — the same question should give the same answer. |
| Interface | **Streamlit** | Chat UI, file upload and state management with very little code, keeping the focus on the pipeline. |

---

## Chunking strategy

`RecursiveCharacterTextSplitter`, **800 characters** with **120 characters
(15%) of overlap**, splitting on `["\n\n", "\n", ". ", " ", ""]` in that order.

**Why 800.** Roughly 150–200 tokens. Large enough to contain a complete policy
clause or paragraph, so a retrieved chunk usually carries the whole answer;
small enough that the chunk is mostly signal. Larger chunks (1500+) diluted
the embedding — a chunk covering four topics matches every one of them weakly.

**Why 15% overlap.** A fact that straddles a boundary would otherwise be
retrievable from neither side. 120 characters is about a sentence, enough to
keep a split sentence reachable, without bloating the index.

**Why recursive.** It tries the most meaningful boundary first: paragraph
breaks, then line breaks, then sentences, and only falls back to splitting
mid-word when nothing better fits. Chunks therefore tend to align with the
document's own structure.

---

## Retrieval strategy

**Hybrid search, Top-K = 5.**

Two retrievers run and their rankings are fused by `EnsembleRetriever`
(reciprocal rank fusion) with weights **0.6 dense / 0.4 sparse**:

- **Dense (FAISS + MiniLM)** matches on meaning, so "time off" finds a passage
  about "annual leave".
- **BM25** matches exact tokens, so a query for `HR-114` or `SEC-2024-01` finds
  the right clause — embeddings represent rare literal strings poorly. There is
  a test covering exactly this case.

Dense is weighted higher because most natural questions are paraphrases;
BM25 is there for the minority of queries that hinge on a literal term.

**Why K=5.** Measured with `python scripts/evaluate_retrieval.py` over 15
labelled questions:

| K | Recall@K | Precision@K | MRR |
|---|---|---|---|
| 1 | 93.3% | 93.3% | 0.933 |
| 3 | **100.0%** | 57.8% | 0.967 |
| 5 | 100.0% | 44.0% | 0.967 |
| 8 | 100.0% | 30.8% | 0.967 |
| 10 | 100.0% | 26.0% | 0.967 |

Recall saturates at **K=3** on this corpus — the one question missed at K=1
("notice period after confirmation", which competes with a similar passage in
the onboarding guide) is retrieved at rank 2. So on this data K=3 would be
sufficient, and K=5 buys no additional recall while halving precision.

K=5 is nevertheless the default, for a reason worth stating plainly: this
corpus is **11 chunks**, small enough that recall saturates almost immediately.
Precision@K is also pessimistic here, since it counts a chunk as a miss
whenever it comes from another document, even when that document is genuinely
relevant to a multi-document question. K=5 leaves headroom for a larger corpus,
where the correct chunk sits further down the ranking, and provides the extra
context that questions 6 and 7 in the test set need to combine facts across
documents.

Set `TOP_K=3` for a measurably tighter context on a corpus this size; the
sidebar slider changes it per question.

---

## Grounding and hallucination handling

Two independent gates, because neither is sufficient alone.

**Gate 1 — similarity floor (before the LLM).** Dense retrieval scores are
converted to true cosine similarity and the best is compared against
`SIMILARITY_THRESHOLD`. If nothing clears it, the answer is returned without
ever calling the LLM. A model that is never invoked cannot hallucinate, and
the refusal costs no tokens and returns in ~0.03s.

**Gate 2 — grounding prompt.** The system prompt restricts the model to the
supplied context, forbids outside knowledge and inference, and specifies the
exact refusal string. Temperature is 0.

### Why the threshold is set low (0.15)

Measured on the sample corpus:

| Question type | Best cosine similarity |
|---|---|
| Answerable | 0.507 – 0.778 |
| Unanswerable | 0.022 – 0.504 |

**These ranges nearly touch.** "What is the maternity leave duration?" scores
**0.504** — it is semantically adjacent to the leave sections even though no
document answers it — while the weakest genuine question scores **0.507**. The
gap is 0.003, so no threshold separates the two sets cleanly. Raising the
threshold to catch maternity leave would start refusing real questions.

The threshold is therefore deliberately a *coarse* filter that only rejects the
obviously off-topic (a question about Japan scores 0.02). Near-misses are
passed to the LLM, whose reading of the context is the more reliable judge —
and it correctly refuses all three unanswerable test questions, including the
0.504 near-miss. **Hallucination handling is the combination of the two gates,
not the threshold alone.**

This is measurable rather than assumed: `tests/test_questions.py` asserts the
refusal behaviour for all three unanswerable questions.

---

## Citations

Every chunk carries `source` and `page` metadata from ingestion onward.
Citations render as `onboarding_guide.pdf - Page 2`, or just the filename for
formats without pages.

Two details worth noting:

- **Only cited sources are listed.** Retrieval returns 5 chunks but an answer
  usually rests on one or two. Listing all 5 would attach authoritative-looking
  citations to documents that contributed nothing, so the source list is
  narrowed to those the answer actually references.
- **Refusals cite nothing.** Attaching sources to "I could not find that"
  would imply evidence that does not exist.

---

## Conversational follow-ups

*(the assessment's bonus challenge)*

Follow-up questions are rewritten into standalone questions before retrieval.
"What about part-time employees?" embeds poorly on its own — it contains almost
no retrievable content — so the last three turns of history are used to rewrite
it as "How many days of annual leave do part-time employees receive?", which
retrieves correctly.

The rewritten query is shown in the UI so the interpretation is visible rather
than hidden. Rewriting is best-effort: if it fails, the original question is
used rather than erroring. Toggle it with **Conversational mode** in the sidebar.

---

## Retrieval evaluation

```bash
python scripts/evaluate_retrieval.py
```

Scores retrieval against 15 labelled question/source pairs and reports
Recall@K, Precision@K and MRR across a sweep of K, plus how cleanly the
similarity gate separates answerable from unanswerable questions. No LLM
calls, so it runs offline and free.

Current results on the sample corpus:

- **Recall@3 onward: 100%** — MRR 0.967, so the correct document is almost
  always ranked first
- **False refusals: 0/15** — the threshold never blocks a real question
- **Blocked by threshold alone: 1/5** unanswerable — the other four are caught
  by the grounding prompt, which is why both gates exist

This is what the K and threshold choices above are based on.

---

## Testing

```bash
# Fast: unit tests, no API calls
pytest -m "not integration"

# The 10 required assessment questions (calls the Groq API)
pytest -m integration

# Everything
pytest
```

**38 tests, all passing.**

**28 unit tests** covering page-number indexing, metadata propagation,
chunk sizing, the L2-to-cosine conversion, score ordering, hybrid retrieval,
BM25 exact-token matching, index persistence, citation formatting, and error
handling for unsupported and missing files.

**10 integration tests** — the questions the assessment requires:

| # | Question | Type | Asserts |
|---|---|---|---|
| 1 | How many days of annual leave are employees entitled to? | Answerable | "24" + cites `employee_policy.md` |
| 2 | How long must account passwords be? | Answerable | "14" + cites `it_security_policy.txt` |
| 3 | What is the daily meal allowance for metro cities? | Answerable | "1,500" + cites `expense_policy.md` |
| 4 | How long is the probation period for new employees? | Answerable | "6 months" + cites `onboarding_guide.pdf` |
| 5 | How many days of paid sick leave are employees entitled to? | Answerable | "12" + cites `employee_policy.md` |
| 6 | Remote work rules, including security requirements for internal systems? | Multi-document | "3 days" **and** "VPN", from 2+ sources |
| 7 | Reimbursement deadline in the handbook, and what happens to claims older than 60 days? | Multi-document | "30 days" **and** forfeiture, from 2+ sources |
| 8 | What is the capital city of Japan? | Unanswerable | Refuses, cites nothing |
| 9 | What is the company's pet adoption allowance? | Unanswerable | Refuses, cites nothing |
| 10 | What is the maternity leave duration? | Unanswerable | Refuses, cites nothing |

Questions 6 and 7 are constructed so each half of the answer lives in a
different file, which is why they assert citations from two or more sources.
Question 10 is the hard case described above — semantically close to real
content, but unanswerable.

Assertions normalise Unicode before matching: the model emits narrow
no-break spaces inside quantities ("6 months"), which is correct output that
would break naive substring checks.

---

## Docker

```bash
docker build -t rag-qa .
docker run -p 8501:8501 --env-file .env rag-qa
```

The image installs CPU-only torch (avoiding the multi-gigabyte CUDA build)
and bakes in the embedding model, so the container needs no Hugging Face
access at runtime. `GROQ_API_KEY` is passed in at run time and never baked
into the image.

Build-tested: the image builds (677MB) and the container serves the app and
answers questions end to end.

If your `.env` wraps the key in quotes (`GROQ_API_KEY="gsk_..."`), note that
`--env-file` passes quotes through literally where `python-dotenv` strips
them. `src/config.py` normalises this so both paths behave the same.

---

## Sample documents

Four documents in `data/sample_docs/`, covering all three supported formats:

| File | Format | Contents |
|---|---|---|
| `employee_policy.md` | Markdown | Leave, working hours, notice periods, reimbursement (HR-114) |
| `it_security_policy.txt` | Plain text | Passwords, devices, remote access, data classification |
| `expense_policy.md` | Markdown | Claim deadlines, travel booking, allowances (FIN-220) |
| `onboarding_guide.pdf` | PDF, 4 pages | Day one, probation, training, equipment |

They deliberately overlap: remote work appears in both the handbook and the
security policy, and the 30-day reimbursement rule appears in both the handbook
and the travel policy. This exercises the multi-document and "similar
information" scenarios rather than making every question single-source.

Regenerate the PDF with `python scripts/make_sample_pdf.py`.

---

## Limitations

**Scanned PDFs are not supported.** Text extraction only; there is no OCR, so
image-based PDFs yield no text. The UI reports this rather than failing
silently.

**FAISS is in-process and single-node.** Fine for this corpus size, but it
holds the index in memory and has no concurrent-write story. A server-based
store (Qdrant, pgvector) would be the next step for a multi-user deployment.

**Re-indexing is all-or-nothing.** Adding a document rebuilds the whole index.
Incremental updates would matter at a larger corpus size.

**Retrieval quality is not formally evaluated.** There are no recall@k or MRR
metrics against a labelled relevance set. The 10 test questions verify
end-to-end behaviour but are not a retrieval benchmark.

**No reranking.** A cross-encoder reranker over the retrieved candidates would
likely improve precision, at the cost of extra latency.

**The similarity threshold is corpus-dependent.** 0.15 was calibrated against
these sample documents. A different corpus would want recalibration — the
measurement approach in [Grounding](#grounding-and-hallucination-handling)
matters more than the number.

**Tables and multi-column layouts extract imperfectly.** `pypdf` flattens them
to linear text, which can scramble reading order.

**Conversational history is unbounded in the UI.** Only the last three turns
inform query rewriting, but the displayed history grows for the session.
