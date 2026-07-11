"""Encounters and Vitals management page for the Healthcare Streamlit app.

Provides an interface for tracking clinical encounters (visits), recording
vitals during a visit, and closing the encounter. Uses the in-memory SQLite
database for demonstration purposes when PostgreSQL is not available.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime

import pandas as pd

import streamlit as st

# ── In-memory database helpers for standalone demo ────────────────────────────


def _get_db() -> sqlite3.Connection:
    """Return a shared in-memory SQLite connection stored in session state."""
    # Ensure the PM DB is initialized first (shares the same connection)
    from services.streamlit.views.patient_management import _get_db as get_pm_db

    conn = get_pm_db()
    _init_encounter_tables(conn)
    return conn


def _init_encounter_tables(conn: sqlite3.Connection) -> None:
    """Create the Encounter and Observation tables in the in-memory database."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS encounters (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            status TEXT DEFAULT 'in-progress',
            encounter_class TEXT DEFAULT 'AMB',
            facility_name TEXT,
            period_start TEXT,
            period_end TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );

        CREATE TABLE IF NOT EXISTS observations (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            encounter_id TEXT,
            status TEXT DEFAULT 'final',
            category TEXT DEFAULT 'vital-signs',
            code TEXT NOT NULL,
            code_display TEXT,
            effective_datetime TEXT,
            value_quantity REAL,
            value_unit TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(id),
            FOREIGN KEY (encounter_id) REFERENCES encounters(id)
        );
    """)
    conn.commit()


def _get_active_encounters(conn: sqlite3.Connection) -> pd.DataFrame:
    """Fetch all encounters that are currently in-progress."""
    query = """
        SELECT e.id, e.patient_id, p.given_name, p.family_name,
               e.status, e.encounter_class, e.facility_name, e.period_start
        FROM encounters e
        JOIN patients p ON e.patient_id = p.id
        WHERE e.status = 'in-progress'
        ORDER BY e.period_start DESC
    """
    return pd.read_sql_query(query, conn)


def _get_encounter_observations(conn: sqlite3.Connection, encounter_id: str) -> pd.DataFrame:
    """Fetch all observations linked to a specific encounter."""
    query = """
        SELECT * FROM observations
        WHERE encounter_id = ?
        ORDER BY effective_datetime DESC
    """
    return pd.read_sql_query(query, conn, params=[encounter_id])


# ── Main page renderer ────────────────────────────────────────────────────────


def render_encounters_page() -> None:
    """Render the Encounters & Vitals page."""
    st.header("🏥 Encounters & Vitals")
    st.caption("Manage patient visits, record clinical observations, and track active encounters.")

    conn = _get_db()

    tab_active, tab_start = st.tabs(
        [
            "🩺 Active Encounters",
            "➕ Start Encounter",
        ]
    )

    # ── Tab 1: Active Encounters ──────────────────────────────────────────────
    with tab_active:
        active_df = _get_active_encounters(conn)

        if active_df.empty:
            st.info("No active encounters. Start a new encounter from the next tab.")
        else:
            # Dropdown to select an active encounter to work on
            encounter_options = {
                row[
                    "id"
                ]: f"{row['given_name']} {row['family_name']} — {row['encounter_class']} ({row['period_start'][:16]})"  # noqa: E501 — long literal (SQL/markdown), not splittable
                for _, row in active_df.iterrows()
            }
            selected_enc_id = st.selectbox(
                "Select Active Encounter",
                options=list(encounter_options.keys()),
                format_func=lambda x: encounter_options[x],
                key="enc_select",
            )

            if selected_enc_id:
                enc = active_df[active_df["id"] == selected_enc_id].iloc[0]
                patient_name = f"{enc['given_name']} {enc['family_name']}"
                patient_id = enc["patient_id"]

                st.markdown(f"### {patient_name}")
                info1, info2, info3 = st.columns(3)
                info1.metric("Class", enc["encounter_class"])
                info2.metric("Facility", enc["facility_name"] or "N/A")
                info3.metric("Started At", enc["period_start"][:16])

                st.divider()

                sub_vitals, sub_close = st.tabs(
                    [
                        "📊 Record Vitals",
                        "✅ Close Encounter",
                    ]
                )

                # ── Record Vitals sub-tab ─────────────────────────────────────
                with sub_vitals:
                    st.subheader("Quick Vitals Entry")
                    with st.form("vitals_form", clear_on_submit=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            sys_bp = st.number_input(
                                "Systolic BP (mmHg)",
                                min_value=50,
                                max_value=300,
                                value=None,
                                step=1,
                            )
                            dia_bp = st.number_input(
                                "Diastolic BP (mmHg)",
                                min_value=20,
                                max_value=200,
                                value=None,
                                step=1,
                            )
                            hr = st.number_input(
                                "Heart Rate (bpm)", min_value=20, max_value=300, value=None, step=1
                            )
                        with col2:
                            temp = st.number_input(
                                "Temperature (°C)",
                                min_value=30.0,
                                max_value=45.0,
                                value=None,
                                step=0.1,
                            )
                            spo2 = st.number_input(
                                "SpO2 (%)", min_value=50, max_value=100, value=None, step=1
                            )
                        with col3:
                            weight = st.number_input(
                                "Weight (kg)", min_value=0.5, max_value=500.0, value=None, step=0.1
                            )
                            height = st.number_input(
                                "Height (cm)", min_value=20.0, max_value=300.0, value=None, step=0.1
                            )

                        vitals_submit = st.form_submit_button(
                            "💾 Save Vitals", type="primary", use_container_width=True
                        )

                        if vitals_submit:
                            vitals = [
                                ("8480-6", "Systolic BP", sys_bp, "mmHg"),
                                ("8462-4", "Diastolic BP", dia_bp, "mmHg"),
                                ("8867-4", "Heart Rate", hr, "/min"),
                                ("8310-5", "Temperature", temp, "Cel"),
                                ("2708-6", "SpO2", spo2, "%"),
                                ("29463-7", "Weight", weight, "kg"),
                                ("8302-2", "Height", height, "cm"),
                            ]
                            now_str = datetime.utcnow().isoformat()
                            to_insert = []
                            for code, disp, val, unit in vitals:
                                if val is not None:
                                    to_insert.append(
                                        (
                                            str(uuid.uuid4()),
                                            patient_id,
                                            selected_enc_id,
                                            code,
                                            disp,
                                            now_str,
                                            val,
                                            unit,
                                        )
                                    )

                            if to_insert:
                                conn.executemany(
                                    "INSERT INTO observations (id, patient_id, encounter_id, code, code_display, effective_datetime, value_quantity, value_unit) VALUES (?,?,?,?,?,?,?,?)",  # noqa: E501 — long literal (SQL/markdown), not splittable
                                    to_insert,
                                )
                                conn.commit()
                                st.success(f"✅ Recorded {len(to_insert)} vital signs!")
                                st.rerun()
                            else:
                                st.warning("No vitals entered.")

                    # Show existing observations for this encounter
                    obs_df = _get_encounter_observations(conn, selected_enc_id)
                    if not obs_df.empty:
                        st.write("**Recorded Observations:**")
                        display_df = obs_df[
                            ["effective_datetime", "code_display", "value_quantity", "value_unit"]
                        ].copy()
                        display_df.columns = ["Time", "Vital Sign", "Value", "Unit"]
                        display_df["Time"] = display_df["Time"].str[:16]
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                # ── Close Encounter sub-tab ───────────────────────────────────
                with sub_close:
                    st.subheader("Complete Visit")
                    st.info(
                        "Closing the encounter will set the end time and mark it as finished. It will no longer appear in the Active Encounters list."  # noqa: E501 — long literal (SQL/markdown), not splittable
                    )
                    if st.button("🏁 Close Encounter", type="primary", use_container_width=True):
                        now_str = datetime.utcnow().isoformat()
                        conn.execute(
                            "UPDATE encounters SET status = 'finished', period_end = ? WHERE id = ?",  # noqa: E501 — long literal (SQL/markdown), not splittable
                            (now_str, selected_enc_id),
                        )
                        conn.commit()
                        st.success("Encounter closed successfully!")
                        st.rerun()

    # ── Tab 2: Start Encounter ────────────────────────────────────────────────
    with tab_start:
        st.subheader("Start New Encounter")

        # Get all patients
        patients_df = pd.read_sql_query(
            "SELECT id, given_name, family_name, date_of_birth FROM patients WHERE deleted_at IS NULL",  # noqa: E501 — long literal (SQL/markdown), not splittable
            conn,
        )

        if patients_df.empty:
            st.warning(
                "No patients registered. Please register a patient first in the Patient Management page."  # noqa: E501 — long literal (SQL/markdown), not splittable
            )
            return

        with st.form("start_encounter_form", clear_on_submit=True):
            p_options = {
                row["id"]: f"{row['given_name']} {row['family_name']} ({row['date_of_birth']})"
                for _, row in patients_df.iterrows()
            }
            patient_id = st.selectbox(
                "Select Patient *",
                options=list(p_options.keys()),
                format_func=lambda x: p_options[x],
            )

            e_class = st.selectbox(
                "Encounter Class *",
                options=[
                    "AMB (Ambulatory)",
                    "IMP (Inpatient)",
                    "EMER (Emergency)",
                    "HH (Home Health)",
                ],
            )
            facility = st.text_input("Facility Name")

            submit_enc = st.form_submit_button(
                "🚀 Start Encounter", type="primary", use_container_width=True
            )

            if submit_enc:
                enc_id = str(uuid.uuid4())
                class_code = (e_class or "").split(" ")[0]
                now_str = datetime.utcnow().isoformat()

                conn.execute(
                    "INSERT INTO encounters (id, patient_id, encounter_class, facility_name, period_start) VALUES (?,?,?,?,?)",  # noqa: E501 — long literal (SQL/markdown), not splittable
                    (enc_id, patient_id, class_code, facility or None, now_str),
                )
                conn.commit()
                st.success("✅ Encounter started successfully!")
                st.rerun()
