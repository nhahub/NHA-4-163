"""ML Trust & Decision Support page for the Streamlit app (Tier 6).

Exercises the deterministic decision-support engine directly through the pure
service layer (no API/DB round-trip), so the UI and API apply identical logic:

* What-If simulator — toggle factors and watch risk recompute, with a SHAP-like
  per-factor breakdown.
* Model monitoring — PSI drift + subgroup fairness parity.
* Guideline recommendations — actionable next steps for a patient context.
* Pedigree completion — suggested missing family edges with rationale.
"""

from __future__ import annotations

import pandas as pd

import streamlit as st
from services.api.services.guideline_service import PatientContext, recommend
from services.api.services.monitoring_service import (
    fairness_report,
    population_stability_index,
)
from services.api.services.pedigree_service import KnownEdge, suggest_links
from services.api.services.whatif_service import RISK_FACTORS, simulate


def _render_whatif_tab() -> None:
    st.subheader("What-If Risk Simulator")
    st.caption(
        "Toggle a patient's factors and see the hereditary-risk estimate move. "
        "Every factor contributes an additive term in log-odds space — the same "
        "unit as a SHAP value — so the delta is fully explainable. No data is "
        "written back."
    )

    baseline: dict[str, float] = {}
    modifications: dict[str, float] = {}
    st.markdown("**Baseline vs. what-if**")
    for f in RISK_FACTORS.values():
        c1, c2 = st.columns(2)
        if f.kind == "binary":
            baseline[f.key] = float(
                c1.checkbox(f"{f.display} (baseline)", value=bool(f.default), key=f"b_{f.key}")
            )
            modifications[f.key] = float(
                c2.checkbox(f"{f.display} (what-if)", value=bool(f.default), key=f"m_{f.key}")
            )
        elif f.kind == "rate":
            baseline[f.key] = c1.slider(
                f"{f.display} (baseline)", 0.0, 1.0, float(f.default), 0.05, key=f"b_{f.key}"
            )
            modifications[f.key] = c2.slider(
                f"{f.display} (what-if)", 0.0, 1.0, float(f.default), 0.05, key=f"m_{f.key}"
            )
        else:  # count
            baseline[f.key] = float(
                c1.number_input(
                    f"{f.display} (baseline, {f.unit})",
                    min_value=0.0,
                    value=float(f.default),
                    step=1.0,
                    key=f"b_{f.key}",
                )
            )
            modifications[f.key] = float(
                c2.number_input(
                    f"{f.display} (what-if, {f.unit})",
                    min_value=0.0,
                    value=float(f.default),
                    step=1.0,
                    key=f"m_{f.key}",
                )
            )

    result = simulate(baseline, modifications)

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline risk", f"{result.baseline_risk:.1%}")
    c2.metric(
        "What-if risk",
        f"{result.simulated_risk:.1%}",
        delta=f"{result.risk_delta * 100:+.1f} pp",
    )
    c3.metric("Δ risk", f"{result.risk_delta * 100:+.1f} pp")
    st.info(result.interpretation)

    contrib = pd.DataFrame(
        [
            {
                "Factor": c.display,
                "What-if value": c.value,
                "Log-odds contribution": c.log_odds_contribution,
                "Δ from baseline": c.delta_from_baseline,
            }
            for c in result.contributions
            if abs(c.delta_from_baseline) > 0
        ]
    )
    if not contrib.empty:
        st.markdown("**What changed (log-odds units, SHAP-comparable)**")
        st.dataframe(contrib, use_container_width=True, hide_index=True)


def _render_monitoring_tab() -> None:
    st.subheader("Model Monitoring & Fairness")
    st.caption(
        "Drift (Population Stability Index) and subgroup risk parity — the "
        "guardrails a regulated PHI model needs before and after release."
    )

    st.markdown("**Score drift (PSI)**")
    c1, c2 = st.columns(2)
    ref_shift = c1.slider("Reference mean", 0.0, 1.0, 0.3, 0.05, key="drift_ref")
    cur_shift = c2.slider("Current mean", 0.0, 1.0, 0.5, 0.05, key="drift_cur")
    # Build two illustrative distributions clustered around the chosen means.
    reference = [min(1.0, max(0.0, ref_shift + d)) for d in (-0.1, -0.05, 0, 0.05, 0.1) * 20]
    current = [min(1.0, max(0.0, cur_shift + d)) for d in (-0.1, -0.05, 0, 0.05, 0.1) * 20]
    drift = population_stability_index(reference, current)
    badge = {"stable": "🟢", "moderate_shift": "🟠", "significant_shift": "🔴"}
    m1, m2 = st.columns(2)
    m1.metric("PSI", f"{drift.psi:.3f}")
    m2.metric("Verdict", f"{badge.get(drift.verdict, '')} {drift.verdict.replace('_', ' ')}")

    st.markdown("**Subgroup fairness (risk parity)**")
    default_groups = pd.DataFrame(
        [
            {"Group": "male", "Mean risk": 0.42},
            {"Group": "female", "Mean risk": 0.40},
            {"Group": "other", "Mean risk": 0.44},
        ]
    )
    edited = st.data_editor(
        default_groups, num_rows="dynamic", use_container_width=True, key="fair_groups"
    )
    scores_by_group = {
        str(r["Group"]): [float(r["Mean risk"])] * 10
        for _, r in edited.iterrows()
        if str(r.get("Group", "")).strip()
    }
    if len(scores_by_group) >= 2:
        report = fairness_report("sex", scores_by_group)
        c1, c2 = st.columns(2)
        c1.metric("Disparate-impact ratio", f"{report.disparate_impact_ratio:.2f}")
        c2.metric("Four-fifths rule", "✅ pass" if report.passes_four_fifths else "⚠️ fail")
        st.info(report.interpretation)


def _render_guidelines_tab() -> None:
    st.subheader("Guideline-Based Screening Recommendations")
    st.caption(
        "Turns a risk score into actionable next steps mapped to NCCN / USPSTF / "
        "ACC-AHA guidance."
    )

    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", min_value=0, max_value=120, value=52, step=1)
    sex = c2.selectbox("Sex", ["female", "male"], key="gl_sex")
    risk = c3.slider("Model risk", 0.0, 1.0, 0.6, 0.05, key="gl_risk")
    fdr = c1.number_input("Affected first-degree relatives", min_value=0, value=1, step=1)
    codes_text = st.text_input(
        "Condition ICD-10 codes (comma-separated)", value="C50", key="gl_codes"
    )
    codes = frozenset(c.strip().upper() for c in codes_text.split(",") if c.strip())

    ctx = PatientContext(
        age=int(age),
        sex=sex,
        risk_score=float(risk),
        condition_codes=codes,
        has_hereditary_condition=bool(codes),
        affected_first_degree_relatives=int(fdr),
    )
    recs = recommend(ctx)
    if not recs:
        st.success("No specific screening rule matched for this context.")
        return
    icon = {"urgent": "🔴", "soon": "🟠", "routine": "🔵"}
    for r in recs:
        with st.expander(f"{icon.get(r.urgency, '🔵')} {r.title} — {r.urgency}"):
            st.markdown(f"**Source:** {r.source}")
            st.markdown(f"**Recommendation:** {r.recommendation}")
            st.caption(f"Rationale: {r.rationale}")


def _render_pedigree_tab() -> None:
    st.subheader("Pedigree Completion (Link Prediction)")
    st.caption(
        "Suggests likely-missing family edges. The API serves these from a "
        "**trained GraphSAGE link-prediction model** (`ml/training/"
        "train_link_prediction.py`) when its artifact is present, falling back "
        "to the transparent structural composer shown here — which explains each "
        "suggestion via the path that produced it."
    )

    default_edges = pd.DataFrame(
        [
            {"Source": "Grandmother", "Relationship": "parent", "Target": "Proband"},
            {"Source": "Proband", "Relationship": "parent", "Target": "Child A"},
            {"Source": "Proband", "Relationship": "sibling", "Target": "Sibling"},
        ]
    )
    edited = st.data_editor(
        default_edges,
        num_rows="dynamic",
        use_container_width=True,
        key="ped_edges",
        column_config={
            "Relationship": st.column_config.SelectboxColumn(
                "Relationship (Source is ___ of Target)",
                options=["parent", "sibling", "spouse", "grandparent", "aunt_uncle"],
            )
        },
    )
    edges = [
        KnownEdge(str(r["Source"]).strip(), str(r["Target"]).strip(), str(r["Relationship"]))
        for _, r in edited.iterrows()
        if str(r.get("Source", "")).strip() and str(r.get("Target", "")).strip()
    ]
    suggestions = suggest_links(edges)
    if not suggestions:
        st.info("No missing edges could be inferred from the recorded relationships.")
        return
    df = pd.DataFrame(
        [
            {
                "Source": s.source,
                "is": s.relationship,
                "of Target": s.target,
                "Confidence": f"{s.confidence:.0%}",
                "Support": s.support,
                "Rationale": s.rationale,
            }
            for s in suggestions
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_decision_support_page() -> None:
    """Render the ML Trust & Decision Support page."""
    st.header("🧠 ML Trust & Decision Support")
    st.caption(
        "Trust-building tools around the risk model: counterfactual what-if "
        "simulation, drift & fairness monitoring, guideline-based next steps, "
        "and pedigree completion — all deterministic, explainable, and sharing "
        "the exact logic used by the API."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔀 What-If Simulator",
            "📉 Monitoring & Fairness",
            "📋 Guidelines",
            "🕸️ Pedigree Completion",
        ]
    )
    with tab1:
        _render_whatif_tab()
    with tab2:
        _render_monitoring_tab()
    with tab3:
        _render_guidelines_tab()
    with tab4:
        _render_pedigree_tab()
