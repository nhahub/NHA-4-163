"""Audit Log Viewer for the Healthcare Streamlit app.

An Admin-only compliance view for querying the immutable ``audit_log`` — who
accessed or mutated which resource, when, and with what outcome.  Never displays
PHI: the audit trail records resource *identifiers* and action types only.

Runs standalone against the shared in-memory SQLite database, seeding a small
sample of realistic (PHI-free) audit events for demonstration.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px

import streamlit as st

_ACTIONS = ["READ", "CREATE", "UPDATE", "DELETE", "PREDICT", "EXPORT", "LOGIN"]
_RESOURCE_TYPES = ["Patient", "Condition", "Observation", "PredictionLog", "Auth"]
_ACTORS = [
    ("dr.smith", "clinician"),
    ("dr.jones", "clinician"),
    ("researcher.lee", "researcher"),
    ("admin.root", "admin"),
    ("svc.ingestion", "service"),
]
_OUTCOMES = ["success", "success", "success", "success", "failure"]


def _get_db() -> sqlite3.Connection:
    """Return the shared in-memory SQLite connection, seeded with audit events."""
    from services.streamlit.views.patient_management import _get_db as get_pm_db

    conn = get_pm_db()
    _init_audit_table(conn)
    return conn


def _init_audit_table(conn: sqlite3.Connection) -> None:
    """Create and seed the demo ``audit_log`` table if absent."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_id TEXT NOT NULL,
            actor_type TEXT NOT NULL,
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT,
            outcome TEXT NOT NULL,
            ip_address TEXT,
            occurred_at TEXT NOT NULL
        );
        """)
    conn.commit()
    _seed_audit(conn)


def _seed_audit(conn: sqlite3.Connection) -> None:
    """Insert sample audit events referencing existing patients (no PHI)."""
    if conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] > 0:
        return

    patient_ids = [r[0] for r in conn.execute("SELECT id FROM patients").fetchall()]
    if not patient_ids:
        return

    now = datetime.utcnow()
    rows = []
    for _ in range(200):
        actor_id, actor_type = random.choice(_ACTORS)
        action = random.choice(_ACTIONS)
        resource_type = "Auth" if action == "LOGIN" else random.choice(_RESOURCE_TYPES[:-1])
        resource_id = None if action == "LOGIN" else random.choice(patient_ids)
        occurred = now - timedelta(minutes=random.randint(0, 60 * 24 * 14))
        rows.append(
            (
                actor_id,
                actor_type,
                action,
                resource_type,
                resource_id,
                random.choice(_OUTCOMES),
                f"10.0.{random.randint(0, 4)}.{random.randint(1, 254)}",
                occurred.isoformat(),
            )
        )

    conn.executemany(
        "INSERT INTO audit_log (actor_id, actor_type, action, resource_type, "
        "resource_id, outcome, ip_address, occurred_at) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def render_audit_viewer() -> None:
    """Render the Admin-only Audit Log Viewer page."""
    st.header("🔒 Audit Log Viewer")
    st.caption(
        "Compliance monitoring of PHI access and mutations. Records identifiers "
        "and actions only — never PHI values."
    )

    # ── Simulated admin gate ──────────────────────────────────────────────────
    # The real API enforces the VIEW_AUDIT_LOG permission (admin role) via RBAC.
    role = st.sidebar.selectbox(
        "Acting role (demo)", ["admin", "clinician", "researcher"], key="audit_role"
    )
    if role != "admin":
        st.error(
            "🚫 Access denied. The audit log is restricted to the **admin** role "
            "(RBAC permission `view:audit_log`). Switch the acting role in the "
            "sidebar to 'admin' to view."
        )
        return

    conn = _get_db()
    df = pd.read_sql_query("SELECT * FROM audit_log", conn)
    if df.empty:
        st.info("No audit events recorded.")
        return

    df["occurred_at"] = pd.to_datetime(df["occurred_at"])

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        actor_filter = st.multiselect("Actor", sorted(df["actor_id"].unique()))
    with f2:
        action_filter = st.multiselect("Action", sorted(df["action"].unique()))
    with f3:
        resource_filter = st.multiselect("Resource", sorted(df["resource_type"].unique()))
    with f4:
        outcome_filter = st.multiselect("Outcome", sorted(df["outcome"].unique()))

    filtered = df
    if actor_filter:
        filtered = filtered[filtered["actor_id"].isin(actor_filter)]
    if action_filter:
        filtered = filtered[filtered["action"].isin(action_filter)]
    if resource_filter:
        filtered = filtered[filtered["resource_type"].isin(resource_filter)]
    if outcome_filter:
        filtered = filtered[filtered["outcome"].isin(outcome_filter)]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Events", f"{len(filtered):,}")
    k2.metric("Distinct Actors", filtered["actor_id"].nunique())
    failures = (filtered["outcome"] == "failure").sum()
    k3.metric("Failures", f"{failures:,}")

    # ── Activity over time ────────────────────────────────────────────────────
    st.subheader("Activity Over Time")
    by_day = (
        filtered.assign(day=filtered["occurred_at"].dt.date)
        .groupby(["day", "action"])
        .size()
        .reset_index(name="count")
    )
    if not by_day.empty:
        fig = px.bar(by_day, x="day", y="count", color="action", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    # ── Event table ───────────────────────────────────────────────────────────
    st.subheader("Events")
    display = filtered.sort_values("occurred_at", ascending=False)[
        [
            "occurred_at",
            "actor_id",
            "actor_type",
            "action",
            "resource_type",
            "resource_id",
            "outcome",
            "ip_address",
        ]
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Export filtered log (CSV)",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="audit_log_export.csv",
        mime="text/csv",
    )
