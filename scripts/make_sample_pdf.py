"""Generate the multi-page sample PDF used to demonstrate page citations.

Kept as a script (rather than committing only the binary) so the sample
corpus is reproducible.
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "onboarding_guide.pdf"

PAGES = [
    (
        "Acme Corporation - New Joiner Onboarding Guide",
        [
            "Welcome to Acme Corporation. This guide covers your first 30 days "
            "and explains the systems, processes and people you will need.",
            "<b>Day One</b>",
            "Report to the reception desk at the Bengaluru office at 10:00 AM. "
            "Bring a government-issued photo ID and your signed offer letter. "
            "Your reporting manager will meet you at reception.",
            "You will be issued a company laptop on your first day. IT will "
            "complete device setup, including full-disk encryption, before "
            "handover.",
        ],
    ),
    (
        "Probation and Confirmation",
        [
            "All new employees serve a probation period of 6 months from the "
            "date of joining.",
            "During probation, performance is reviewed at the end of month 3 "
            "and again at the end of month 6. Confirmation is subject to a "
            "satisfactory review at month 6.",
            "The probation period may be extended once, by a maximum of 3 "
            "months, at the discretion of the department head.",
            "The notice period during probation is 15 days, as set out in the "
            "Employee Handbook.",
        ],
    ),
    (
        "Training Requirements",
        [
            "All new joiners must complete the following mandatory training "
            "within their first 30 days:",
            "1. Information Security Awareness (2 hours)",
            "2. Code of Conduct and Anti-Harassment (90 minutes)",
            "3. Data Privacy and Customer Data Handling (2 hours)",
            "Completion is tracked in the learning portal. Failure to complete "
            "mandatory training within 30 days is escalated to the reporting "
            "manager and may delay confirmation.",
            "Engineering hires additionally complete a Secure Coding module "
            "within 60 days of joining.",
        ],
    ),
    (
        "Equipment and Workspace",
        [
            "Standard issue for all employees is a laptop, a docking station, "
            "an external monitor and a headset.",
            "Engineering roles may request a second monitor and an upgraded "
            "16-core development machine through the IT helpdesk, subject to "
            "manager approval.",
            "Desk allocation is managed by the workplace team. Employees "
            "working under a hybrid arrangement share hot desks and should "
            "book a desk through the workplace portal at least one day in "
            "advance.",
            "Equipment must be returned on the last working day. Unreturned "
            "equipment is recovered from the final settlement.",
        ],
    ),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        title="Acme Onboarding Guide",
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    flow = []
    for i, (heading, paragraphs) in enumerate(PAGES):
        flow.append(Paragraph(heading, styles["Heading1"]))
        flow.append(Spacer(1, 0.4 * cm))
        for para in paragraphs:
            flow.append(Paragraph(para, styles["BodyText"]))
            flow.append(Spacer(1, 0.25 * cm))
        if i < len(PAGES) - 1:
            flow.append(PageBreak())

    doc.build(flow)
    print(f"Wrote {OUT} ({len(PAGES)} pages)")


if __name__ == "__main__":
    main()
