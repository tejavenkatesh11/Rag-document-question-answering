"""Capture the demo screenshots the assessment asks for.

Submission item 4 requires screenshots showing document upload, a question,
the answer, and the source. This drives the running Streamlit app with
Playwright so the images are reproducible rather than hand-taken.

Usage:
    streamlit run app.py --server.port 8503      # in one terminal
    python scripts/capture_screenshots.py        # in another

Set SCREENSHOT_URL to point at a different port.
"""

import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "img"
URL = os.getenv("SCREENSHOT_URL", "http://localhost:8503")

# Streamlit reruns the whole script on every interaction, so waits here are
# generous: an embedding pass or an LLM round-trip can take several seconds.
SETTLE = 2500


def shot(page, name: str, note: str) -> None:
    IMG.mkdir(exist_ok=True)
    path = IMG / name
    page.screenshot(path=str(path), full_page=True)
    print(f"  {name:<34} {note}")


def wait_for_text(page, text: str, timeout: int = 90_000) -> bool:
    """Wait for text to appear, returning False instead of raising."""
    try:
        page.get_by_text(text, exact=False).first.wait_for(timeout=timeout)
        return True
    except Exception:
        return False


def wait_until_idle(page, timeout: int = 180_000) -> None:
    """Wait until Streamlit has finished rendering and is interactive.

    Streamlit paints the page skeleton before its widgets hydrate, and greys
    the whole UI out while a rerun is in flight. Screenshotting during either
    phase captures placeholder boxes instead of real controls, so wait for
    the running indicator to clear and for a known widget to be enabled.
    """
    # The status widget carries data-testid="stStatusWidget" while a script
    # run is in progress; its absence means the rerun finished.
    page.wait_for_function(
        """() => !document.querySelector('[data-testid="stStatusWidget"]')""",
        timeout=timeout,
    )
    # Buttons render disabled until hydration completes.
    page.wait_for_function(
        """() => {
            const b = [...document.querySelectorAll('button')];
            return b.length > 0 && b.some(x => !x.disabled);
        }""",
        timeout=timeout,
    )
    page.wait_for_timeout(1200)


def answer_count(page) -> int:
    """How many answers are currently on the page."""
    return page.get_by_text("Answered in", exact=False).count()


def ask(page, question: str) -> None:
    """Type a question into the chat input and wait for the NEW answer.

    Waiting for the text "Answered in" is not enough on its own: previous
    answers already show it, so the wait returns instantly and the screenshot
    catches the spinner for the current question. Waiting for the count to
    increase is what actually signals this answer finished.
    """
    before = answer_count(page)
    box = page.get_by_placeholder("Ask a question about your documents...")
    box.click()
    box.fill(question)
    box.press("Enter")
    page.wait_for_function(
        """(n) => document.body.innerText.split('Answered in').length - 1 > n""",
        arg=before,
        timeout=180_000,
    )
    wait_until_idle(page)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})

        print(f"Opening {URL}")
        # Streamlit holds a websocket open, so "networkidle" never fires and
        # the first paint can land before the app has rendered anything.
        # Waiting for actual content is the reliable signal.
        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        if not wait_for_text(page, "Document Question Answering", 120_000):
            print(f"ERROR: the app did not render. Is it running at {URL}?")
            browser.close()
            sys.exit(1)
        wait_until_idle(page)

        # --- 1. Empty state, showing the upload control -------------------
        shot(page, "01-upload-documents.png", "upload UI / empty state")

        # --- 2. Index the sample corpus -----------------------------------
        load = page.get_by_role("button", name="Load sample documents")
        if not load.count():
            print("ERROR: 'Load sample documents' button not found.")
            print("If an index already exists, run 'Clear index' first.")
            browser.close()
            sys.exit(1)

        load.click()
        # "Indexed 4 sample document(s) into 11 chunks." is the success
        # banner. Matching on "Indexed" alone also matches the spinner text
        # that appears while embedding is still running, which produced a
        # screenshot of a greyed-out mid-rerun UI.
        if not wait_for_text(page, "into 11 chunks", timeout=180_000):
            print("ERROR: indexing did not complete in time.")
            browser.close()
            sys.exit(1)
        wait_until_idle(page)
        shot(page, "02-documents-indexed.png", "4 docs -> 11 chunks")

        # --- 3. A question answered with its source -----------------------
        ask(page, "How many days of annual leave are employees entitled to?")
        shot(page, "03-question-and-answer.png", "grounded answer + citation")

        # Expand the Sources panel so the citation and excerpt are visible.
        sources = page.get_by_text("Sources (", exact=False).last
        if sources.count():
            sources.click()
            page.wait_for_timeout(1200)
            shot(page, "04-sources-expanded.png", "source excerpt shown")

        # --- 4. A PDF answer, to show page-level citation -----------------
        ask(page, "How long is the probation period for new employees?")
        src = page.get_by_text("Sources (", exact=False).last
        if src.count():
            src.click()
            page.wait_for_timeout(1200)
        shot(page, "05-pdf-page-citation.png", "cites 'Page 2' of the PDF")

        # --- 5. An unanswerable question ----------------------------------
        ask(page, "What is the maternity leave duration?")
        shot(page, "06-unanswerable-question.png", "declines, cites nothing")

        # --- 6. A conversational follow-up --------------------------------
        ask(page, "What about part-time employees?")
        shot(page, "07-conversational-followup.png", "follow-up rewritten")

        browser.close()
        print(f"\nSaved to {IMG}")


if __name__ == "__main__":
    main()
