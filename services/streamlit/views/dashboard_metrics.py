"""Dashboard overview metrics for the Healthcare Streamlit app.

Renders the top-level KPI row, risk distribution, demographic breakdowns, and a
recent-predictions table. Every figure is computed live from the shared
in-memory SQLite panel (patients + prediction_log) that backs the rest of the
app, so the Dashboard stays consistent with Patient Management, Batch Screening,
and Population Health instead of showing hardcoded mock numbers.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import plotly.express as px

import streamlit as st


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
    if age < 31:
        return "18-30"
    if age < 46:
        return "31-45"
    if age < 61:
        return "46-60"
    if age < 76:
        return "61-75"
    return "75+"


def _load_panel() -> pd.DataFrame:
    """Return patients joined with their latest prediction (one row per patient)."""
    from services.streamlit.views.screening_page import _get_db

    conn = _get_db()

    patients = pd.read_sql_query(
        "SELECT id, gender, date_of_birth FROM patients WHERE deleted_at IS NULL",
        conn,
    )
    latest = pd.read_sql_query(
        """
        SELECT p.patient_id, p.risk_score, p.risk_tier, p.predicted_at
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
    return merged


def render_dashboard_overview() -> None:
    """Render the KPI row, distributions, and recent predictions from live data."""
    merged = _load_panel()
    if merged.empty:
        st.warning("No patients available.")
        return

    total = len(merged)
    scored = merged.dropna(subset=["risk_score"])
    screened = len(scored)
    coverage = screened / total if total else 0.0
    at_risk = int(merged["risk_tier"].isin(["high", "very_high"]).sum())

    from services.streamlit.views.screening_page import _get_db

    today_iso = date.today().isoformat()
    preds_today = _get_db().execute(
        "SELECT COUNT(*) FROM prediction_log WHERE substr(predicted_at, 1, 10) = ?",
        (today_iso,),
    ).fetchone()[0]

    # ── KPI row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{total:,}")
    col2.metric(
        "At-Risk Patients",
        f"{at_risk:,}",
        f"{at_risk / total:.0%} of panel" if total else None,
    )
    col3.metric("Screened Coverage", f"{coverage:.0%}", f"{screened:,} scored")
    col4.metric("Predictions (Today)", f"{preds_today:,}")

    st.markdown("---")

    # ── Risk distribution ─────────────────────────────────────────────────────
    st.subheader("Risk Score Distribution")
    if scored.empty:
        st.info("No predictions recorded yet. Run a batch screening first.")
    else:
        fig = px.histogram(
            scored,
            x="risk_score",
            nbins=30,
            title="Distribution of Patient Risk Scores (latest per patient)",
            labels={"risk_score": "Risk Score"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # ── Demographic breakdowns ────────────────────────────────────────────────
    st.subheader("Risk by Demographics")
    col1, col2 = st.columns(2)

    with col1:
        if scored.empty:
            st.info("No predictions recorded yet.")
        else:
            order = ["18-30", "31-45", "46-60", "61-75", "75+"]
            by_age = (
                scored.groupby("age_band")["risk_score"]
                .mean()
                .reindex(order)
                .dropna()
                .reset_index()
            )
            fig = px.bar(
                by_age,
                x="age_band",
                y="risk_score",
                title="Average Risk Score by Age Group",
                labels={"age_band": "Age Group", "risk_score": "Risk Score"},
                color="risk_score",
                color_continuous_scale="RdYlGn_r",
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if scored.empty:
            st.info("No predictions recorded yet.")
        else:
            by_gender = scored.groupby("gender")["risk_score"].mean().reset_index()
            fig = px.bar(
                by_gender,
                x="gender",
                y="risk_score",
                title="Average Risk Score by Gender",
                labels={"gender": "Gender", "risk_score": "Risk Score"},
                color="risk_score",
                color_continuous_scale="RdYlGn_r",
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    # ── Recent predictions ────────────────────────────────────────────────────
    st.subheader("Recent Predictions")
    conn = _get_db()
    recent = pd.read_sql_query(
        """
        SELECT pl.patient_id, pl.risk_score, pl.risk_tier, pl.predicted_at,
               pt.date_of_birth
        FROM prediction_log pl
        JOIN patients pt ON pt.id = pl.patient_id
        WHERE pt.deleted_at IS NULL
        ORDER BY pl.predicted_at DESC
        LIMIT 15
        """,
        conn,
    )
    if recent.empty:
        st.info("No predictions recorded yet. Run a batch screening first.")
        return

    tier_emoji = {
        "low": "🟢 Low",
        "moderate": "🟡 Moderate",
        "high": "🔴 High",
        "very_high": "🔴 Very High",
    }
    recent["Patient ID"] = recent["patient_id"].str.slice(0, 8)
    recent["Risk Score"] = recent["risk_score"].map(lambda s: f"{s:.0%}")
    recent["Risk Category"] = recent["risk_tier"].map(tier_emoji).fillna(recent["risk_tier"])
    recent["Age"] = recent["date_of_birth"].apply(_compute_age)
    recent["Predicted"] = pd.to_datetime(recent["predicted_at"]).dt.strftime("%Y-%m-%d %H:%M")

    st.dataframe(
        recent[["Patient ID", "Risk Score", "Risk Category", "Age", "Predicted"]],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Panel snapshot generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
