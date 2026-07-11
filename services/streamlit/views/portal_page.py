"""Patient Portal & Consent page for the Streamlit app (Tier 7).

Demonstrates the Tier 7 patient-facing surface through the pure service layer
(no API/DB round-trip), so the UI and API apply identical rules:

* Consent management — record grant/deny/withdraw per scope and see the
  effective state resolved from an append-only history.
* Patient portal preview — the lay-friendly risk banding and the de-identified
  pedigree view a patient would see via SMART on FHIR.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

import streamlit as st
from libs.common.models.consent import ConsentScope, ConsentStatus
from services.api.services.consent_service import is_record_active
from services.api.services.portal_service import build_risk_profile, lay_band


@dataclass
class _ConsentRow:
    """A lightweight stand-in for a ConsentRecord in the pure resolver preview."""

    scope: ConsentScope
    status: ConsentStatus
    expires_at: datetime | None
    created_at: datetime


@dataclass
class _FakePrediction:
    """Minimal PredictionLog stand-in for the risk-profile preview."""

    risk_score: float
    risk_tier: str
    model_version: str | None = "demo-1.0"
    predicted_at: datetime | None = None


def _render_consent_tab() -> None:
    """Consent recorder + effective-state resolver preview."""
    st.subheader("🔐 Consent Management")
    st.caption(
        "Consent is append-only: each decision is a new record and the "
        "effective state for a scope is the most recent one. Withdrawing "
        "`research` consent removes a patient from future research exports."
    )

    if "consent_history" not in st.session_state:
        st.session_state.consent_history = []

    col1, col2, col3 = st.columns(3)
    with col1:
        scope = st.selectbox("Scope", [s.value for s in ConsentScope], key="consent_scope")
    with col2:
        decision = st.selectbox("Decision", [s.value for s in ConsentStatus], key="consent_status")
    with col3:
        expires = st.date_input("Expires (optional)", value=None, key="consent_expiry")

    if st.button("Record decision", type="primary"):
        st.session_state.consent_history.append(
            _ConsentRow(
                scope=ConsentScope(str(scope)),
                status=ConsentStatus(str(decision)),
                expires_at=(
                    datetime.combine(expires, datetime.min.time(), tzinfo=UTC)
                    if isinstance(expires, date)
                    else None
                ),
                created_at=datetime.now(UTC),
            )
        )

    history = st.session_state.consent_history
    if not history:
        st.info("No consent decisions recorded yet.")
        return

    # Effective state = latest record per scope (mirrors resolve_effective_consent).
    effective: dict[ConsentScope, _ConsentRow] = {}
    for rec in history:
        cur = effective.get(rec.scope)
        if cur is None or rec.created_at >= cur.created_at:
            effective[rec.scope] = rec

    st.markdown("**Effective consent (current)**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Scope": s.value,
                    "Decision": r.status.value,
                    "Active": "✅" if is_record_active(r.status, r.expires_at) else "❌",
                    "Expires": r.expires_at.date().isoformat() if r.expires_at else "—",
                }
                for s, r in sorted(effective.items(), key=lambda kv: kv[0].value)
            ]
        ),
        use_container_width=True,
    )

    st.markdown("**Decision history (newest first)**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Recorded": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "Scope": r.scope.value,
                    "Decision": r.status.value,
                }
                for r in reversed(history)
            ]
        ),
        use_container_width=True,
    )


def _render_portal_tab() -> None:
    """Patient-portal risk & pedigree preview (SMART on FHIR self-service view)."""
    st.subheader("📱 Patient Portal Preview")
    st.caption(
        "The read-only, lay-friendly view a patient sees after a SMART on FHIR "
        "standalone launch — their own risk summary and a de-identified pedigree."
    )

    score = st.slider("Latest calibrated risk score", 0.0, 1.0, 0.55, 0.01)
    band = lay_band(score)
    profile = build_risk_profile(
        _FakePrediction(
            risk_score=score,
            risk_tier=band,
            predicted_at=datetime.now(UTC),
        )
    )

    colour = {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get(band, "⚪")
    c1, c2 = st.columns(2)
    c1.metric("Risk score", f"{score:.0%}")
    c2.metric("Risk band", f"{colour} {band.title()}")
    st.info(profile.guidance)

    st.markdown("**Your family tree (de-identified)**")
    st.caption(
        "Relatives are PHI — the portal exposes relationship and affected status "
        "only, never a relative's name or contact details."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Relationship": "Mother",
                    "Sex": "female",
                    "Affected": "Yes",
                    "Conditions": "Breast cancer",
                },
                {"Relationship": "Father", "Sex": "male", "Affected": "No", "Conditions": "—"},
                {"Relationship": "Sister", "Sex": "female", "Affected": "No", "Conditions": "—"},
                {
                    "Relationship": "Maternal grandmother",
                    "Sex": "female",
                    "Affected": "Yes",
                    "Conditions": "Breast cancer",
                },
            ]
        ),
        use_container_width=True,
    )


def render_portal_page() -> None:
    """Render the Tier 7 Patient Portal & Consent page."""
    st.title("🔐 Patient Portal & Consent")
    st.markdown("Tier 7 — patient-facing self-service and granular, auditable consent.")
    consent_tab, portal_tab = st.tabs(["Consent", "Patient Portal"])
    with consent_tab:
        _render_consent_tab()
    with portal_tab:
        _render_portal_tab()
