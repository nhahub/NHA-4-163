"""Notifications page for the Healthcare Streamlit app (Tier 4).

Surfaces risk-threshold and rising-risk alerts derived from each patient's
prediction history. Reuses the pure ``evaluate_risk_change`` logic from the API
notification service so the UI and API apply identical rules, and stores
acknowledgements in the shared in-memory SQLite database.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

import streamlit as st
from services.api.services.notification_service import (
    DEFAULT_THRESHOLD,
    evaluate_risk_change,
)


def _get_db() -> sqlite3.Connection:
    """Return the shared in-memory SQLite connection (with prediction_log)."""
    from services.streamlit.views.screening_page import _get_db as get_screening_db

    return get_screening_db()


def _derive_notifications(conn: sqlite3.Connection, threshold: float) -> pd.DataFrame:
    """Derive current alerts from the latest two predictions per patient."""
    patients = pd.read_sql_query(
        "SELECT id, given_name, family_name FROM patients WHERE deleted_at IS NULL",
        conn,
    )
    alerts = []
    for _, p in patients.iterrows():
        hist = pd.read_sql_query(
            "SELECT risk_score, risk_tier, predicted_at FROM prediction_log "
            "WHERE patient_id = ? ORDER BY predicted_at DESC LIMIT 2",
            conn,
            params=[p["id"]],
        )
        if hist.empty:
            continue
        current = float(hist.iloc[0]["risk_score"])
        tier = hist.iloc[0]["risk_tier"]
        previous = float(hist.iloc[1]["risk_score"]) if len(hist) > 1 else None
        event = evaluate_risk_change(current, previous, tier, threshold=threshold)
        if event is None:
            continue
        alerts.append(
            {
                "Patient": f"{p['given_name']} {p['family_name']}",
                "Severity": event.severity.value,
                "Type": event.notification_type.value.replace("_", " ").title(),
                "Risk": current,
                "Message": event.message,
            }
        )
    return pd.DataFrame(alerts)


def render_notifications_page() -> None:
    """Render the Notifications page."""
    st.header("🔔 Notifications")
    st.caption(
        "Risk-threshold and rising-risk alerts derived from patient prediction "
        "history. The same rules drive the API's automatic notifications."
    )

    conn = _get_db()

    threshold = st.slider(
        "Alert threshold (risk score)",
        min_value=0.5,
        max_value=0.95,
        value=float(DEFAULT_THRESHOLD),
        step=0.05,
        format="%.2f",
    )

    df = _derive_notifications(conn, threshold)

    if df.empty:
        st.success("✅ No active alerts at the current threshold.")
        return

    crit = (df["Severity"] == "critical").sum()
    warn = (df["Severity"] == "warning").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Alerts", len(df))
    c2.metric("Critical", int(crit))
    c3.metric("Warning", int(warn))

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    df = df.sort_values("Severity", key=lambda s: s.map(severity_order))

    icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
    for _, row in df.iterrows():
        with st.container(border=True):
            st.markdown(
                f"{icon.get(row['Severity'], '🔵')} **{row['Type']}** — "
                f"{row['Patient']}  ·  risk **{row['Risk']:.0%}**"
            )
            st.caption(row["Message"])
