import os
from datetime import datetime

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)


OUTPUT_DIR = "outputs/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_pdf(report: dict):
    """
    Generate a PDF report from the AI report dictionary.
    """

    filename = f"brain_tumor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    pdf_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph("<b>Brain Tumor AI Medical Report</b>", styles["Title"])
    )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            f"<b>Prediction:</b> {report['prediction']}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Confidence:</b> {report['confidence']}%",
            styles["BodyText"]
        )
    )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph("<b>Description</b>", styles["Heading2"])
    )

    elements.append(
        Paragraph(
            report["description"],
            styles["BodyText"]
        )
    )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph("<b>Recommendations</b>", styles["Heading2"])
    )

    for recommendation in report["recommendations"]:
        elements.append(
            Paragraph(
                f"• {recommendation}",
                styles["BodyText"]
            )
        )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph("<b>Disclaimer</b>", styles["Heading2"])
    )

    elements.append(
        Paragraph(
            report["disclaimer"],
            styles["BodyText"]
        )
    )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            f"Generated: {datetime.now()}",
            styles["BodyText"]
        )
    )

    doc.build(elements)

    return pdf_path