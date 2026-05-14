"""Patient Management page for the Healthcare Streamlit app.

Provides a full CRUD interface for managing patients, conditions,
family relationships, and medications using an in-memory SQLite
database for demonstration when PostgreSQL is not available.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import date, datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ── In-memory database for standalone demo ────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    """Return a shared in-memory SQLite connection stored in session state."""
    if "pm_db" not in st.session_state:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _init_tables(conn)
        _seed_demo_data(conn)
        st.session_state["pm_db"] = conn
    return st.session_state["pm_db"]


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create the CRUD tables in the in-memory database."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            given_name TEXT NOT NULL,
            family_name TEXT NOT NULL,
            middle_name TEXT,
            date_of_birth TEXT NOT NULL,
            gender TEXT NOT NULL DEFAULT 'unknown',
            ethnicity TEXT,
            race TEXT,
            phone TEXT,
            email TEXT,
            city TEXT,
            state TEXT,
            country TEXT DEFAULT 'US',
            deceased INTEGER DEFAULT 0,
            research_consent INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS conditions (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            code TEXT NOT NULL,
            code_system TEXT DEFAULT 'ICD-10',
            code_display TEXT,
            clinical_status TEXT DEFAULT 'active',
            severity TEXT,
            is_hereditary INTEGER DEFAULT 0,
            onset_datetime TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );

        CREATE TABLE IF NOT EXISTS family_members (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            relationship_code TEXT NOT NULL,
            relationship_display TEXT,
            degree_of_relatedness REAL DEFAULT 0.5,
            sex TEXT,
            deceased INTEGER DEFAULT 0,
            conditions_json TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );

        CREATE TABLE IF NOT EXISTS medications (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            medication_code TEXT NOT NULL,
            medication_display TEXT,
            status TEXT DEFAULT 'active',
            intent TEXT DEFAULT 'order',
            dosage_text TEXT,
            dose_quantity REAL,
            dose_unit TEXT,
            authored_on TEXT DEFAULT (datetime('now')),
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );
    """)
    conn.commit()


def _seed_demo_data(conn: sqlite3.Connection) -> None:
    """Insert sample data for demonstration purposes."""
    patients = [
        (str(uuid.uuid4()), "Patient", "A-001", None, "1978-03-15", "female", "Hispanic", None, None, None, "Miami", "FL", "US"),
        (str(uuid.uuid4()), "Patient", "B-002", None, "1985-07-22", "male", None, "White", None, None, "New York", "NY", "US"),
        (str(uuid.uuid4()), "Patient", "C-003", None, "1960-11-08", "female", None, "Asian", None, None, "San Francisco", "CA", "US"),
        (str(uuid.uuid4()), "Patient", "D-004", None, "1992-01-30", "male", "Non-Hispanic", None, None, None, "Chicago", "IL", "US"),
        (str(uuid.uuid4()), "Patient", "E-005", None, "1945-06-12", "female", None, None, None, None, "Houston", "TX", "US"),
    ]
    conn.executemany(
        "INSERT INTO patients (id, given_name, family_name, middle_name, date_of_birth, gender, ethnicity, race, phone, email, city, state, country) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        patients,
    )

    # Add conditions for first few patients
    p0, p1, p2 = patients[0][0], patients[1][0], patients[2][0]
    conditions = [
        (str(uuid.uuid4()), p0, "E11.9", "ICD-10", "Type 2 diabetes mellitus", "active", "moderate", 1),
        (str(uuid.uuid4()), p0, "I10", "ICD-10", "Essential hypertension", "active", "mild", 0),
        (str(uuid.uuid4()), p1, "J45.20", "ICD-10", "Mild intermittent asthma", "active", "mild", 0),
        (str(uuid.uuid4()), p2, "C50.9", "ICD-10", "Malignant neoplasm of breast", "remission", "severe", 1),
        (str(uuid.uuid4()), p2, "I25.10", "ICD-10", "Atherosclerotic heart disease", "active", "moderate", 1),
    ]
    conn.executemany(
        "INSERT INTO conditions (id, patient_id, code, code_system, code_display, clinical_status, severity, is_hereditary) VALUES (?,?,?,?,?,?,?,?)",
        conditions,
    )

    # Family members
    family = [
        (str(uuid.uuid4()), p0, "MTH", "Mother", 0.5, "female", 0, '[{"code": "E11.9", "display": "Type 2 diabetes"}]'),
        (str(uuid.uuid4()), p0, "FTH", "Father", 0.5, "male", 1, '[{"code": "I10", "display": "Hypertension"}]'),
        (str(uuid.uuid4()), p2, "MTH", "Mother", 0.5, "female", 1, '[{"code": "C50.9", "display": "Breast cancer"}]'),
    ]
    conn.executemany(
        "INSERT INTO family_members (id, patient_id, relationship_code, relationship_display, degree_of_relatedness, sex, deceased, conditions_json) VALUES (?,?,?,?,?,?,?,?)",
        family,
    )

    # Medications
    meds = [
        (str(uuid.uuid4()), p0, "860975", "Metformin 500mg", "active", "order", "500mg twice daily", 500, "mg"),
        (str(uuid.uuid4()), p0, "197361", "Lisinopril 10mg", "active", "order", "10mg once daily", 10, "mg"),
        (str(uuid.uuid4()), p1, "895994", "Albuterol inhaler", "active", "order", "2 puffs as needed", 90, "mcg"),
        (str(uuid.uuid4()), p2, "262105", "Tamoxifen 20mg", "completed", "order", "20mg once daily", 20, "mg"),
    ]
    conn.executemany(
        "INSERT INTO medications (id, patient_id, medication_code, medication_display, status, intent, dosage_text, dose_quantity, dose_unit) VALUES (?,?,?,?,?,?,?,?,?)",
        meds,
    )
    conn.commit()


# ── Helper functions ──────────────────────────────────────────────────────────

def _get_patients(conn: sqlite3.Connection, search: str = "") -> pd.DataFrame:
    """Fetch all non-deleted patients, optionally filtered by search term."""
    query = "SELECT * FROM patients WHERE deleted_at IS NULL"
    params: list[str] = []
    if search:
        query += " AND (given_name LIKE ? OR family_name LIKE ?)"
        params = [f"%{search}%", f"%{search}%"]
    query += " ORDER BY created_at DESC"
    return pd.read_sql_query(query, conn, params=params)


def _get_conditions(conn: sqlite3.Connection, patient_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM conditions WHERE patient_id = ? ORDER BY created_at DESC",
        conn, params=[patient_id],
    )


def _get_family(conn: sqlite3.Connection, patient_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM family_members WHERE patient_id = ? ORDER BY degree_of_relatedness DESC",
        conn, params=[patient_id],
    )


def _get_medications(conn: sqlite3.Connection, patient_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM medications WHERE patient_id = ? ORDER BY created_at DESC",
        conn, params=[patient_id],
    )


RELATIONSHIP_CODES = {
    "MTH": ("Mother", 0.5),
    "FTH": ("Father", 0.5),
    "SIB": ("Sibling", 0.5),
    "CHILD": ("Child", 0.5),
    "GRPRN": ("Grandparent", 0.25),
    "UNCLE": ("Uncle", 0.25),
    "AUNT": ("Aunt", 0.25),
    "COUSN": ("Cousin", 0.125),
    "HBRO": ("Half-brother", 0.25),
    "HSIS": ("Half-sister", 0.25),
}


# ── Main page renderer ───────────────────────────────────────────────────────

def render_patient_management() -> None:
    """Render the Patient Management page."""
    st.header("👤 Patient Management")
    st.caption("Register, view, and manage patient records, conditions, family history, and medications.")

    conn = _get_db()

    tab_list, tab_register, tab_detail = st.tabs([
        "📋 Patient List",
        "➕ Register Patient",
        "🔍 Patient Detail",
    ])

    # ── Tab 1: Patient List ───────────────────────────────────────────────────
    with tab_list:
        col_search, col_filter = st.columns([3, 1])
        with col_search:
            search = st.text_input("🔍 Search patients", placeholder="Search by name...", key="pm_search")
        with col_filter:
            gender_filter = st.selectbox("Gender", ["All", "male", "female", "other", "unknown"], key="pm_gender")

        patients_df = _get_patients(conn, search)
        if gender_filter != "All":
            patients_df = patients_df[patients_df["gender"] == gender_filter]

        if patients_df.empty:
            st.info("No patients found. Register a new patient using the tab above.")
        else:
            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Patients", len(patients_df))
            m2.metric("Male", len(patients_df[patients_df["gender"] == "male"]))
            m3.metric("Female", len(patients_df[patients_df["gender"] == "female"]))
            m4.metric("With Research Consent", len(patients_df[patients_df["research_consent"] == 1]))

            # Display table
            display_df = patients_df[["id", "given_name", "family_name", "date_of_birth", "gender", "city", "state"]].copy()
            display_df.columns = ["ID", "First Name", "Last Name", "DOB", "Gender", "City", "State"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Tab 2: Register Patient ───────────────────────────────────────────────
    with tab_register:
        st.subheader("Register New Patient")

        with st.form("register_patient_form", clear_on_submit=True):
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                given_name = st.text_input("First Name *", key="reg_given")
            with r1c2:
                family_name = st.text_input("Last Name *", key="reg_family")
            with r1c3:
                middle_name = st.text_input("Middle Name", key="reg_middle")

            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                dob = st.date_input("Date of Birth *", value=date(1990, 1, 1), min_value=date(1900, 1, 1), key="reg_dob")
            with r2c2:
                gender = st.selectbox("Gender *", ["male", "female", "other", "unknown"], key="reg_gender")
            with r2c3:
                ethnicity = st.text_input("Ethnicity", key="reg_eth")

            r3c1, r3c2, r3c3 = st.columns(3)
            with r3c1:
                phone = st.text_input("Phone", key="reg_phone")
            with r3c2:
                email = st.text_input("Email", key="reg_email")
            with r3c3:
                race = st.text_input("Race", key="reg_race")

            r4c1, r4c2, r4c3 = st.columns(3)
            with r4c1:
                city = st.text_input("City", key="reg_city")
            with r4c2:
                state = st.text_input("State", key="reg_state")
            with r4c3:
                country = st.text_input("Country", value="US", key="reg_country")

            consent = st.checkbox("Research Consent Granted", key="reg_consent")
            submitted = st.form_submit_button("✅ Register Patient", use_container_width=True, type="primary")

            if submitted:
                if not given_name or not family_name:
                    st.error("First Name and Last Name are required.")
                else:
                    patient_id = str(uuid.uuid4())
                    conn.execute(
                        "INSERT INTO patients (id, given_name, family_name, middle_name, date_of_birth, gender, ethnicity, race, phone, email, city, state, country, research_consent) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (patient_id, given_name, family_name, middle_name or None, str(dob), gender, ethnicity or None, race or None, phone or None, email or None, city or None, state or None, country, int(consent)),
                    )
                    conn.commit()
                    st.success(f"✅ Patient registered successfully! ID: `{patient_id[:8]}...`")
                    st.balloons()

    # ── Tab 3: Patient Detail ─────────────────────────────────────────────────
    with tab_detail:
        patients_df = _get_patients(conn)
        if patients_df.empty:
            st.info("No patients available. Register a patient first.")
            return

        # Patient selector
        patient_options = {
            row["id"]: f"{row['given_name']} {row['family_name']} ({row['date_of_birth']})"
            for _, row in patients_df.iterrows()
        }
        selected_id = st.selectbox(
            "Select Patient",
            options=list(patient_options.keys()),
            format_func=lambda x: patient_options[x],
            key="pm_select_patient",
        )

        if selected_id:
            patient = patients_df[patients_df["id"] == selected_id].iloc[0]

            # Patient info card
            st.markdown(f"### {patient['given_name']} {patient['family_name']}")
            info1, info2, info3, info4 = st.columns(4)
            info1.metric("DOB", patient["date_of_birth"])
            info2.metric("Gender", patient["gender"].title())
            info3.metric("Location", f"{patient['city'] or 'N/A'}, {patient['state'] or 'N/A'}")
            info4.metric("Consent", "✅ Yes" if patient["research_consent"] else "❌ No")

            st.divider()

            # Sub-tabs for clinical data
            sub_cond, sub_fam, sub_med, sub_actions = st.tabs([
                "🏥 Conditions",
                "👨‍👩‍👧 Family History",
                "💊 Medications",
                "⚙️ Actions",
            ])

            # ── Conditions sub-tab ────────────────────────────────────────────
            with sub_cond:
                conditions_df = _get_conditions(conn, selected_id)

                with st.expander("➕ Add New Condition", expanded=False):
                    with st.form("add_condition_form", clear_on_submit=True):
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            code = st.text_input("ICD-10 Code *", placeholder="e.g., E11.9", key="cond_code")
                            code_display = st.text_input("Display Name", placeholder="e.g., Type 2 diabetes", key="cond_display")
                        with cc2:
                            clin_status = st.selectbox("Clinical Status", ["active", "confirmed", "remission", "resolved", "inactive"], key="cond_status")
                            severity = st.selectbox("Severity", ["mild", "moderate", "severe"], key="cond_severity")
                        is_hereditary = st.checkbox("Hereditary Condition", key="cond_hered")
                        cond_submit = st.form_submit_button("Add Condition", type="primary", use_container_width=True)

                        if cond_submit and code:
                            conn.execute(
                                "INSERT INTO conditions (id, patient_id, code, code_display, clinical_status, severity, is_hereditary) VALUES (?,?,?,?,?,?,?)",
                                (str(uuid.uuid4()), selected_id, code, code_display or None, clin_status, severity, int(is_hereditary)),
                            )
                            conn.commit()
                            st.success(f"✅ Condition `{code}` added!")
                            st.rerun()

                if conditions_df.empty:
                    st.info("No conditions recorded for this patient.")
                else:
                    for _, cond in conditions_df.iterrows():
                        hered_badge = " 🧬" if cond["is_hereditary"] else ""
                        status_color = {"active": "🔴", "confirmed": "🟡", "remission": "🟢", "resolved": "✅"}.get(cond["clinical_status"], "⚪")
                        st.markdown(
                            f"{status_color} **{cond['code']}** — {cond['code_display'] or 'N/A'} "
                            f"| Status: `{cond['clinical_status']}` | Severity: `{cond['severity'] or 'N/A'}`{hered_badge}"
                        )

            # ── Family History sub-tab ────────────────────────────────────────
            with sub_fam:
                family_df = _get_family(conn, selected_id)

                with st.expander("➕ Add Family Member", expanded=False):
                    with st.form("add_family_form", clear_on_submit=True):
                        fc1, fc2 = st.columns(2)
                        with fc1:
                            rel_code = st.selectbox(
                                "Relationship *",
                                options=list(RELATIONSHIP_CODES.keys()),
                                format_func=lambda x: f"{RELATIONSHIP_CODES[x][0]} ({x})",
                                key="fam_rel",
                            )
                            sex = st.selectbox("Sex", ["male", "female", "unknown"], key="fam_sex")
                        with fc2:
                            degree = st.number_input(
                                "Degree of Relatedness",
                                value=RELATIONSHIP_CODES[rel_code][1],
                                min_value=0.0, max_value=1.0, step=0.125,
                                key="fam_degree",
                            )
                            fam_deceased = st.checkbox("Deceased", key="fam_deceased")

                        fam_conditions = st.text_input("Conditions (comma-separated ICD-10 codes)", key="fam_conds")
                        fam_submit = st.form_submit_button("Add Family Member", type="primary", use_container_width=True)

                        if fam_submit:
                            import json
                            conds_json = json.dumps([{"code": c.strip()} for c in fam_conditions.split(",") if c.strip()]) if fam_conditions else "[]"
                            conn.execute(
                                "INSERT INTO family_members (id, patient_id, relationship_code, relationship_display, degree_of_relatedness, sex, deceased, conditions_json) VALUES (?,?,?,?,?,?,?,?)",
                                (str(uuid.uuid4()), selected_id, rel_code, RELATIONSHIP_CODES[rel_code][0], degree, sex, int(fam_deceased), conds_json),
                            )
                            conn.commit()
                            st.success(f"✅ {RELATIONSHIP_CODES[rel_code][0]} added!")
                            st.rerun()

                if family_df.empty:
                    st.info("No family history recorded for this patient.")
                else:
                    for _, member in family_df.iterrows():
                        deceased_badge = " ⚰️" if member["deceased"] else ""
                        st.markdown(
                            f"**{member['relationship_display'] or member['relationship_code']}** "
                            f"({member['sex'] or 'unknown'}) | "
                            f"Relatedness: `{member['degree_of_relatedness']}`{deceased_badge}"
                        )

            # ── Medications sub-tab ───────────────────────────────────────────
            with sub_med:
                meds_df = _get_medications(conn, selected_id)

                with st.expander("➕ Add Medication", expanded=False):
                    with st.form("add_med_form", clear_on_submit=True):
                        mc1, mc2 = st.columns(2)
                        with mc1:
                            med_code = st.text_input("RxNorm Code *", placeholder="e.g., 860975", key="med_code")
                            med_display = st.text_input("Medication Name", placeholder="e.g., Metformin 500mg", key="med_display")
                        with mc2:
                            med_status = st.selectbox("Status", ["active", "completed", "stopped", "on-hold"], key="med_status")
                            dosage = st.text_input("Dosage", placeholder="e.g., 500mg twice daily", key="med_dosage")

                        med_submit = st.form_submit_button("Add Medication", type="primary", use_container_width=True)

                        if med_submit and med_code:
                            conn.execute(
                                "INSERT INTO medications (id, patient_id, medication_code, medication_display, status, dosage_text) VALUES (?,?,?,?,?,?)",
                                (str(uuid.uuid4()), selected_id, med_code, med_display or None, med_status, dosage or None),
                            )
                            conn.commit()
                            st.success(f"✅ Medication `{med_display or med_code}` added!")
                            st.rerun()

                if meds_df.empty:
                    st.info("No medications recorded for this patient.")
                else:
                    for _, med in meds_df.iterrows():
                        status_icon = {"active": "💊", "completed": "✅", "stopped": "🛑", "on-hold": "⏸️"}.get(med["status"], "💊")
                        st.markdown(
                            f"{status_icon} **{med['medication_display'] or med['medication_code']}** "
                            f"(`{med['medication_code']}`) | "
                            f"Status: `{med['status']}` | Dosage: `{med['dosage_text'] or 'N/A'}`"
                        )

            # ── Actions sub-tab ───────────────────────────────────────────────
            with sub_actions:
                st.subheader("Patient Actions")

                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    if st.button("🗑️ Delete Patient", type="secondary", use_container_width=True):
                        conn.execute(
                            "UPDATE patients SET deleted_at = ? WHERE id = ?",
                            (datetime.utcnow().isoformat(), selected_id),
                        )
                        conn.commit()
                        st.warning("Patient has been soft-deleted.")
                        st.rerun()

                with col_a2:
                    conditions_count = len(_get_conditions(conn, selected_id))
                    meds_count = len(_get_medications(conn, selected_id))
                    family_count = len(_get_family(conn, selected_id))
                    st.markdown(f"""
                    **Clinical Summary:**
                    - 🏥 **{conditions_count}** conditions
                    - 💊 **{meds_count}** medications
                    - 👨‍👩‍👧 **{family_count}** family members
                    """)
