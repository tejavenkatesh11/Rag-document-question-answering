# Screenshots

Captured from the running Streamlit app against the four sample documents.
Reproduce with:

```bash
streamlit run app.py --server.port 8503     # terminal 1
python scripts/capture_screenshots.py       # terminal 2
```

| # | Image | Shows |
|---|---|---|
| 1 | `01-upload-documents.png` | Upload control and empty state, before any document is indexed |
| 2 | `02-documents-indexed.png` | 4 documents ingested into 11 chunks |
| 3 | `03-question-and-answer.png` | A question and its grounded answer with an inline citation |
| 4 | `04-sources-expanded.png` | The Sources panel expanded, showing the citation and the supporting excerpt |
| 5 | `05-pdf-page-citation.png` | Page-level citation from the PDF — `onboarding_guide.pdf - Page 2` |
| 6 | `06-unanswerable-question.png` | An unanswerable question declined, with no sources attached |
| 7 | `07-conversational-followup.png` | A follow-up ("What about part-time employees?") rewritten into a standalone query |

Images 1–4 cover what the assessment asks for in submission item 4: document
upload, question, answer and source. Images 5–7 show page-level citation,
hallucination handling and the conversational bonus.
