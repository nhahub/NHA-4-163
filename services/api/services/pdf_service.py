"""Clinical PDF report generation using ``fpdf2``.

Builds a one-page clinical risk summary for a patient: demographics, current
risk score/tier, active conditions, and the top SHAP risk factors driving the
prediction.  ``fpdf2`` is chosen over ``reportlab``/``wkhtmltopdf`` because it
is pure-Python and requires no external system binaries, keeping the container
image small and the build reproducible.

COMPLIANCE NOTE: A generated report contains PHI.  Every call to
``generate_patient_report`` should be recorded in the audit log by the caller
(the report router relies on ``AuditLogMiddleware`` for this).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from fpdf import FPDF

log = logging.getLogger(__name__)


@dataclass
class ReportData:
    """Structured input for a clinical report.

    Attributes:
        patient_id: Internal patient UUID (string form).
        full_name: Patient display name.
        date_of_birth: Patient DOB.
        gender: Administrative gender.
        risk_score: Predicted hereditary-risk probability in [0, 1].
        risk_tier: Human-readable tier (low | moderate | high | very_high).
        conditions: List of (code, display, status) tuples.
        shap_factors: List of dicts with ``feature``/``shap_value``/``direction``.
        generated_at: Report generation timestamp.
    """

    patient_id: str
    full_name: str
    date_of_birth: date | None
    gender: str | None
    risk_score: float | None
    risk_tier: str | None
    conditions: list[tuple[str, str, str]] = field(default_factory=list)
    shap_factors: list[dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)


# Tier → RGB colour for the risk banner.
_TIER_COLORS: dict[str, tuple[int, int, int]] = {
    "low": (34, 139, 34),
    "moderate": (218, 165, 32),
    "high": (205, 92, 0),
    "very_high": (178, 34, 34),
}


class ClinicalReportPDF(FPDF):  # type: ignore[misc]  # fpdf2 ships no type stubs
    """A4 clinical report with a standard header and footer."""

    def header(self) -> None:  # noqa: D102 — fpdf hook
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Hereditary Disease Risk Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(
            0,
            5,
            "CONFIDENTIAL - Protected Health Information",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self) -> None:  # noqa: D102 — fpdf hook
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_patient_report(data: ReportData) -> bytes:
    """Render a one-page clinical risk report to PDF bytes.

    Args:
        data: The structured report input.

    Returns:
        The PDF document as a ``bytes`` object suitable for streaming.
    """
    pdf = ClinicalReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Demographics ──────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Patient Demographics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    dob = data.date_of_birth.isoformat() if data.date_of_birth else "N/A"
    age = _compute_age(data.date_of_birth)
    rows = [
        ("Name", data.full_name or "N/A"),
        ("Patient ID", data.patient_id),
        ("Date of Birth", f"{dob}  (age {age})" if age is not None else dob),
        ("Gender", (data.gender or "N/A").title()),
        ("Report Generated", data.generated_at.strftime("%Y-%m-%d %H:%M UTC")),
    ]
    for label, value in rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(45, 6, f"{label}:", new_x="END", new_y="LAST")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ── Risk banner ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Hereditary Risk Assessment", new_x="LMARGIN", new_y="NEXT")

    tier = (data.risk_tier or "unknown").lower()
    r, g, b = _TIER_COLORS.get(tier, (100, 100, 100))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    score_txt = f"{data.risk_score:.1%}" if data.risk_score is not None else "N/A"
    pdf.cell(
        0,
        12,
        f"  Risk Score: {score_txt}     Tier: {tier.replace('_', ' ').title()}",
        new_x="LMARGIN",
        new_y="NEXT",
        fill=True,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Conditions ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Current Conditions", new_x="LMARGIN", new_y="NEXT")
    if data.conditions:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(30, 6, "Code", border=1, fill=True)
        pdf.cell(110, 6, "Description", border=1, fill=True)
        pdf.cell(40, 6, "Status", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for code, display, cstatus in data.conditions:
            pdf.cell(30, 6, _clip(code, 16), border=1)
            pdf.cell(110, 6, _clip(display, 64), border=1)
            pdf.cell(40, 6, _clip(cstatus, 20), border=1, new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 6, "No active conditions recorded.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── SHAP risk factors ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Top Contributing Risk Factors", new_x="LMARGIN", new_y="NEXT")
    if data.shap_factors:
        pdf.set_font("Helvetica", "", 9)
        for i, factor in enumerate(data.shap_factors, start=1):
            feature = str(factor.get("feature", "unknown"))
            direction = str(factor.get("direction", ""))
            shap_value = factor.get("shap_value")
            arrow = "increases" if direction == "increases_risk" else "decreases"
            val_txt = f"{shap_value:+.3f}" if isinstance(shap_value, (int, float)) else ""
            pdf.cell(
                0,
                6,
                f"{i}. {_humanise(feature)} -- {arrow} risk  ({val_txt})",
                new_x="LMARGIN",
                new_y="NEXT",
            )
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(
            0,
            6,
            "Model explanations unavailable for this prediction.",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(4)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0,
        4,
        "This report is a clinical decision-support aid generated by a machine "
        "learning model. It does not constitute a diagnosis. Interpret in the "
        "context of the full clinical picture and confirm with appropriate "
        "genetic counselling and testing.",
    )
    pdf.set_text_color(0, 0, 0)

    output = pdf.output()
    return bytes(output)


def _compute_age(dob: date | None) -> int | None:
    """Return an integer age in years from a date of birth, or ``None``."""
    if dob is None:
        return None
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _clip(text: str, max_len: int) -> str:
    """Truncate ``text`` to ``max_len`` characters with an ellipsis."""
    text = text or ""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _humanise(feature: str) -> str:
    """Convert a snake_case feature name into a readable label."""
    return feature.replace("_", " ").title()
