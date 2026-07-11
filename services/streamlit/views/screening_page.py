"""Batch screening and risk trending page for the Healthcare Streamlit app.

Provides an interface for running batch predictions across patient cohorts
and analyzing longitudinal risk trends. Uses in-memory data for demo purposes.
"""

from __future__ import annotations

import random
import sqlite3
import uuid
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px

import streamlit as st

# ── Database helpers ──────────────────────────────────────────────────────────


def _get_db() -> sqlite3.Connection:
    """Return a shared in-memory SQLite connection stored in session state."""
    from services.streamlit.views.patient_management import _get_db as get_pm_db

    conn = get_pm_db()
    _init_screening_tables(conn)
    return conn


def _init_screening_tables(conn: sqlite3.Connection) -> None:
    """Create the PredictionLog table in the in-memory database."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            feature_date TEXT NOT NULL,
            source TEXT DEFAULT 'api',
            predicted_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );
    """)
    conn.commit()
    _seed_screening_data(conn)


def _seed_screening_data(conn: sqlite3.Connection) -> None:
    """Seed historical prediction logs for trend analysis."""
    # Only seed if empty
    if conn.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0] > 0:
        return

    patients = conn.execute("SELECT id FROM patients").fetchall()
    if not patients:
        return

    logs = []
    now = datetime.utcnow()

    for (pid,) in patients:
        # Generate 3-5 historical predictions per patient
        num_preds = random.randint(3, 5)
        base_risk = random.uniform(0.1, 0.7)

        for i in range(num_preds):
            days_ago = (num_preds - i) * 30  # Roughly one per month
            pred_date = now - timedelta(days=days_ago)

            # Add some random walk to the risk score
            score = max(0.01, min(0.99, base_risk + random.uniform(-0.1, 0.15)))

            if score < 0.25:
                tier = "low"
            elif score < 0.50:
                tier = "moderate"
            elif score < 0.75:
                tier = "high"
            else:
                tier = "very_high"

            logs.append(
                (
                    str(uuid.uuid4()),
                    pid,
                    round(score, 4),
                    tier,
                    "hereditary-risk-xgboost",
                    "1.0",
                    pred_date.strftime("%Y-%m-%d"),
                    "batch",
                    pred_date.isoformat(),
                )
            )

    conn.executemany(
        "INSERT INTO prediction_log (id, patient_id, risk_score, risk_tier, model_name, model_version, feature_date, source, predicted_at) VALUES (?,?,?,?,?,?,?,?,?)",  # noqa: E501 — long literal (SQL/markdown), not splittable
        logs,
    )
    conn.commit()


# ── Main page renderer ────────────────────────────────────────────────────────


def render_screening_page() -> None:
    """Render the Batch Screening & Risk Trends page."""
    st.header("📊 Batch Screening & Risk Trends")
    st.caption("Run panel-wide risk assessments and track patient risk trajectories over time.")

    conn = _get_db()

    tab_batch, tab_trends = st.tabs(
        [
            "🚀 Run Batch Screening",
            "📈 Risk Trends",
        ]
    )

    # ── Tab 1: Batch Screening ────────────────────────────────────────────────
    with tab_batch:
        st.subheader("Panel Screening Configuration")

        col1, col2 = st.columns(2)
        with col1:
            target = st.radio(
                "Target Population", ["All Active Patients", "Specific Gender", "Age Range"]
            )

            gender_filter = None
            if target == "Specific Gender":
                gender_filter = st.selectbox("Gender", ["male", "female", "other"])

            age_min, age_max = None, None
            if target == "Age Range":
                age_min, age_max = st.slider("Age Range", 0, 120, (40, 80))

        with col2:
            st.info(
                "Batch screening runs asynchronously and logs results to the PredictionLog for historical tracking."  # noqa: E501 — long literal (SQL/markdown), not splittable
            )
            run_btn = st.button(
                "▶️ Start Batch Screening", type="primary", use_container_width=True
            )

        if run_btn:
            with st.spinner("Executing batch predictions..."):
                import time

                time.sleep(1.5)  # Simulate API latency

                # Fetch target patients
                query = "SELECT id, given_name, family_name FROM patients WHERE deleted_at IS NULL"
                params = []
                if target == "Specific Gender" and gender_filter:
                    query += " AND gender = ?"
                    params.append(gender_filter)

                pts = pd.read_sql_query(query, conn, params=params)

                if pts.empty:
                    st.warning("No patients found matching the criteria.")
                else:
                    now_str = datetime.utcnow().isoformat()
                    feat_date = datetime.utcnow().strftime("%Y-%m-%d")

                    logs = []
                    results_data = []

                    for _, row in pts.iterrows():
                        # Simulate prediction
                        score = round(random.betavariate(2, 5), 4)
                        if score < 0.25:
                            tier = "low"
                        elif score < 0.50:
                            tier = "moderate"
                        elif score < 0.75:
                            tier = "high"
                        else:
                            tier = "very_high"

                        logs.append(
                            (
                                str(uuid.uuid4()),
                                row["id"],
                                score,
                                tier,
                                "hereditary-risk-xgboost",
                                "1.0",
                                feat_date,
                                "batch",
                                now_str,
                            )
                        )

                        results_data.append(
                            {
                                "Patient": f"{row['given_name']} {row['family_name']}",
                                "Risk Score": score,
                                "Tier": tier,
                            }
                        )

                    # Log to DB
                    conn.executemany(
                        "INSERT INTO prediction_log (id, patient_id, risk_score, risk_tier, model_name, model_version, feature_date, source, predicted_at) VALUES (?,?,?,?,?,?,?,?,?)",  # noqa: E501 — long literal (SQL/markdown), not splittable
                        logs,
                    )
                    conn.commit()

                    st.success(f"✅ Successfully screened {len(pts)} patients.")

                    # Display results
                    res_df = pd.DataFrame(results_data).sort_values("Risk Score", ascending=False)

                    # Color coding
                    def color_tier(val: str) -> str:
                        colors = {
                            "low": "green",
                            "moderate": "orange",
                            "high": "red",
                            "very_high": "darkred",
                        }
                        return f'color: {colors.get(val, "black")}'

                    st.dataframe(
                        res_df.style.applymap(color_tier, subset=["Tier"]),
                        use_container_width=True,
                        hide_index=True,
                    )

    # ── Tab 2: Risk Trends ────────────────────────────────────────────────────
    with tab_trends:
        patients_df = pd.read_sql_query(
            "SELECT id, given_name, family_name, date_of_birth FROM patients WHERE deleted_at IS NULL",  # noqa: E501 — long literal (SQL/markdown), not splittable
            conn,
        )

        if patients_df.empty:
            st.warning("No patients available.")
            return

        p_options = {
            row["id"]: f"{row['given_name']} {row['family_name']} ({row['date_of_birth']})"
            for _, row in patients_df.iterrows()
        }
        selected_pid = st.selectbox(
            "Select Patient to View Trends",
            options=list(p_options.keys()),
            format_func=lambda x: p_options[x],
            key="trend_select",
        )

        if selected_pid:
            history_df = pd.read_sql_query(
                "SELECT predicted_at, risk_score, risk_tier FROM prediction_log WHERE patient_id = ? ORDER BY predicted_at ASC",  # noqa: E501 — long literal (SQL/markdown), not splittable
                conn,
                params=[selected_pid],
            )

            if history_df.empty:
                st.info("No prediction history available for this patient. Run a screening first.")
            else:
                history_df["predicted_at"] = pd.to_datetime(history_df["predicted_at"])

                # Metrics
                if len(history_df) >= 2:
                    current = history_df.iloc[-1]["risk_score"]
                    previous = history_df.iloc[-2]["risk_score"]
                    change = (current - previous) / previous if previous > 0 else 0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current Risk", f"{current:.1%}", f"{change:+.1%}")
                    m2.metric(
                        "Current Tier", history_df.iloc[-1]["risk_tier"].replace("_", " ").title()
                    )
                    m3.metric("Total Assessments", len(history_df))
                else:
                    st.metric("Current Risk", f"{history_df.iloc[-1]['risk_score']:.1%}")

                # Chart
                fig = px.line(
                    history_df,
                    x="predicted_at",
                    y="risk_score",
                    markers=True,
                    title="Longitudinal Risk Trajectory",
                    labels={"predicted_at": "Date", "risk_score": "Probability"},
                )
                fig.update_yaxes(range=[0, 1.0], tickformat=".0%")

                # Add risk tier bands
                fig.add_hrect(
                    y0=0, y1=0.25, fillcolor="green", opacity=0.1, layer="below", line_width=0
                )
                fig.add_hrect(
                    y0=0.25, y1=0.5, fillcolor="orange", opacity=0.1, layer="below", line_width=0
                )
                fig.add_hrect(
                    y0=0.5, y1=0.75, fillcolor="red", opacity=0.1, layer="below", line_width=0
                )
                fig.add_hrect(
                    y0=0.75, y1=1.0, fillcolor="darkred", opacity=0.1, layer="below", line_width=0
                )

                st.plotly_chart(fig, use_container_width=True)

                # Raw data expander
                with st.expander("View Raw History"):
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
