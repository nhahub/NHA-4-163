"""Population Health Dashboard for the Healthcare Streamlit app.

Visualises aggregate risk distributions, screening coverage, and demographic
breakdowns across the whole patient panel.  Uses the shared in-memory SQLite
database (patients + prediction_log) so it runs standalone without PostgreSQL.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd
import plotly.express as px

import streamlit as st


def _get_db() -> sqlite3.Connection:
    """Return the shared in-memory SQLite connection (seeded with predictions)."""
    from services.streamlit.views.screening_page import _get_db as get_screening_db

    return get_screening_db()


def _compute_age(dob: str) -> int | None:
    """Compute an integer age from an ISO date-of-birth string."""
    try:
        born = date.fromisoformat(str(dob)[:10])
    except (ValueError, TypeError):
        return None
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _age_band(age: int | None) -> str:
    """Bucket an age into a display band."""
    if age is None:
        return "Unknown"
    if age < 18:
        return "0-17"
    if age < 31:
        return "18-30"
    if age < 46:
        return "31-45"
    if age < 61:
        return "46-60"
    if age < 76:
        return "61-75"
    return "75+"


def render_population_health() -> None:
    """Render the Population Health Dashboard page."""
    st.header("🌍 Population Health Dashboard")
    st.caption(
        "Aggregate risk, screening coverage, and demographic breakdowns across "
        "the entire patient panel."
    )

    conn = _get_db()

    patients = pd.read_sql_query(
        "SELECT id, gender, ethnicity, race, date_of_birth, state "
        "FROM patients WHERE deleted_at IS NULL",
        conn,
    )
    if patients.empty:
        st.warning("No patients available.")
        return

    # Latest prediction per patient.
    latest = pd.read_sql_query(
        """
        SELECT p.patient_id, p.risk_score, p.risk_tier
        FROM prediction_log p
        INNER JOIN (
            SELECT patient_id, MAX(predicted_at) AS max_at
            FROM prediction_log GROUP BY patient_id
        ) m ON p.patient_id = m.patient_id AND p.predicted_at = m.max_at
        """,
        conn,
    )

    merged = patients.merge(latest, left_on="id", right_on="patient_id", how="left")
    merged["age"] = merged["date_of_birth"].apply(_compute_age)
    merged["age_band"] = merged["age"].apply(_age_band)

    screened = merged["risk_score"].notna().sum()
    total = len(merged)
    coverage = screened / total if total else 0.0
    high_risk = merged["risk_tier"].isin(["high", "very_high"]).sum()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", f"{total:,}")
    c2.metric("Screened", f"{screened:,}", f"{coverage:.0%} coverage")
    c3.metric("High / Very High Risk", f"{high_risk:,}")
    mean_risk = merged["risk_score"].mean()
    c4.metric("Mean Risk Score", f"{mean_risk:.1%}" if pd.notna(mean_risk) else "N/A")

    st.markdown("---")

    # ── Risk distribution ─────────────────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        st.subheader("Risk Score Distribution")
        scored = merged.dropna(subset=["risk_score"])
        if scored.empty:
            st.info("No predictions recorded yet. Run a batch screening first.")
        else:
            fig = px.histogram(
                scored,
                x="risk_score",
                nbins=20,
                labels={"risk_score": "Risk Score"},
                color_discrete_sequence=["#636EFA"],
            )
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Risk Tier Breakdown")
        tier_counts = merged["risk_tier"].fillna("unscreened").value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        tier_order = {"low": 0, "moderate": 1, "high": 2, "very_high": 3, "unscreened": 4}
        tier_counts = tier_counts.sort_values("tier", key=lambda s: s.map(tier_order).fillna(9))
        fig = px.bar(
            tier_counts,
            x="tier",
            y="count",
            color="tier",
            color_discrete_map={
                "low": "#2ca02c",
                "moderate": "#ff7f0e",
                "high": "#d62728",
                "very_high": "#7f0000",
                "unscreened": "#999999",
            },
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Demographics ──────────────────────────────────────────────────────────
    st.subheader("Mean Risk by Demographic")
    tab_age, tab_gender, tab_ethnicity = st.tabs(["Age Band", "Gender", "Ethnicity"])

    def _mean_by(column: str, order: list[str] | None = None) -> None:
        scored = merged.dropna(subset=["risk_score"])
        if scored.empty:
            st.info("No predictions recorded yet.")
            return
        grp = scored.groupby(column)["risk_score"].mean().reset_index()
        if order:
            grp = grp.set_index(column).reindex(order).dropna().reset_index()
        fig = px.bar(
            grp,
            x=column,
            y="risk_score",
            labels={"risk_score": "Mean Risk", column: column.replace("_", " ").title()},
            color="risk_score",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with tab_age:
        _mean_by("age_band", ["0-17", "18-30", "31-45", "46-60", "61-75", "75+"])
    with tab_gender:
        _mean_by("gender")
    with tab_ethnicity:
        merged["ethnicity"] = merged["ethnicity"].fillna("Not recorded")
        _mean_by("ethnicity")
