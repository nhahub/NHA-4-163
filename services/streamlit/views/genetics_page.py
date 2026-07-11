"""Genetics & Genomics page for the Healthcare Streamlit app (Tier 5).

Demonstrates the deterministic genetics engine directly through the pure service
layer (no API/DB round-trip needed), so the UI and API apply identical rules:

* Mendelian inheritance calculator — carrier/affected probabilities per relative.
* Cascade screening preview — ranked at-risk relatives for a proband's condition.
* Variant annotation — classify pasted variants against the curated ClinVar KB.
* Polygenic risk score — blended PRS + (optional) ML risk for a common disease.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

import streamlit as st
from services.api.services.cascade_service import rank_relative
from services.api.services.inheritance_service import (
    INHERITANCE_MODELS,
    categorise_relationship,
    compute_relative_risk,
)
from services.api.services.prs_service import PRS_PANELS, compute_prs
from services.api.services.variant_service import annotate_variant, parse_vcf

# A small illustrative pedigree used to seed the calculators.
_DEFAULT_PEDIGREE = [
    {"Relationship": "MTH", "Sex": "female", "Degree": 0.5},
    {"Relationship": "FTH", "Sex": "male", "Degree": 0.5},
    {"Relationship": "SIS", "Sex": "female", "Degree": 0.5},
    {"Relationship": "BRO", "Sex": "male", "Degree": 0.5},
    {"Relationship": "SON", "Sex": "male", "Degree": 0.5},
    {"Relationship": "DAU", "Sex": "female", "Degree": 0.5},
    {"Relationship": "GRMTH", "Sex": "female", "Degree": 0.25},
    {"Relationship": "AUNT", "Sex": "female", "Degree": 0.25},
    {"Relationship": "COUSN", "Sex": "female", "Degree": 0.125},
]


def _mode_selector(key: str) -> str:
    """Render an inheritance-mode selectbox and return the chosen key."""
    options = list(INHERITANCE_MODELS.keys())
    choice = st.selectbox(
        "Inheritance mode",
        options,
        format_func=lambda k: INHERITANCE_MODELS[k].display,
        key=key,
    )
    return str(choice)


def _pedigree_editor(key: str) -> pd.DataFrame:
    """Render an editable pedigree table and return it as a DataFrame."""
    return st.data_editor(
        pd.DataFrame(_DEFAULT_PEDIGREE),
        num_rows="dynamic",
        use_container_width=True,
        key=key,
        column_config={
            "Relationship": st.column_config.TextColumn(
                "Relationship (HL7 code)", help="MTH, FTH, SIB, SON, DAU, AUNT, COUSN, ..."
            ),
            "Sex": st.column_config.SelectboxColumn(
                "Sex", options=["male", "female", ""], required=False
            ),
            "Degree": st.column_config.NumberColumn(
                "Degree of relatedness", min_value=0.0, max_value=1.0, step=0.125
            ),
        },
    )


def _render_inheritance_tab() -> None:
    st.subheader("Mendelian Inheritance Calculator")
    st.caption(
        "Deterministic carrier/affected probabilities for each relative of an "
        "affected proband. Complements the ML model — every number is explainable."
    )

    mode = _mode_selector("inh_mode")
    model = INHERITANCE_MODELS[mode]
    st.info(model.description)

    c1, c2 = st.columns(2)
    penetrance = c1.slider(
        "Penetrance", 0.0, 1.0, float(model.default_penetrance), 0.05, key="inh_pen"
    )
    carrier_freq = c2.slider(
        "Population carrier frequency",
        0.0,
        0.1,
        float(model.default_carrier_frequency),
        0.001,
        format="%.3f",
        key="inh_cf",
    )

    pedigree = _pedigree_editor("inh_pedigree")

    rows = []
    for _, r in pedigree.iterrows():
        code = str(r.get("Relationship", "")).strip()
        if not code:
            continue
        degree = r.get("Degree")
        risk = compute_relative_risk(
            mode=mode,
            relationship_code=code,
            degree_of_relatedness=float(degree) if pd.notna(degree) else None,
            relative_sex=(str(r.get("Sex")) or None),
            penetrance=penetrance,
            carrier_frequency=carrier_freq,
        )
        rows.append(
            {
                "Relative": code,
                "Category": categorise_relationship(code),
                "Carrier %": round(risk.carrier_probability * 100, 1),
                "Affected %": round(risk.affected_probability * 100, 1),
                "Basis": risk.basis,
            }
        )

    if rows:
        df = pd.DataFrame(rows).sort_values(["Affected %", "Carrier %"], ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_cascade_tab() -> None:
    st.subheader("Cascade Screening Preview")
    st.caption(
        "Ranks the proband's blood relatives for outreach when a hereditary "
        "condition is diagnosed. Priority = relatedness × penetrance."
    )

    mode = _mode_selector("cas_mode")
    model = INHERITANCE_MODELS[mode]
    condition = st.text_input("Condition", value="Hereditary condition", key="cas_cond")
    penetrance = st.slider(
        "Penetrance", 0.0, 1.0, float(model.default_penetrance), 0.05, key="cas_pen"
    )

    pedigree = _pedigree_editor("cas_pedigree")

    rows = []
    for _, r in pedigree.iterrows():
        code = str(r.get("Relationship", "")).strip()
        if not code or categorise_relationship(code) == "spouse":
            continue
        degree = r.get("Degree")
        ranked = rank_relative(
            relationship_code=code,
            degree_of_relatedness=float(degree) if pd.notna(degree) else None,
            relative_sex=(str(r.get("Sex")) or None),
            inheritance_mode=mode,
            penetrance=penetrance,
            carrier_frequency=model.default_carrier_frequency,
            condition_display=condition,
        )
        if ranked.priority_score <= 0:
            continue
        rows.append(
            {
                "Relative": code,
                "Priority": ranked.priority.value,
                "Score": ranked.priority_score,
                "Affected %": round(ranked.affected_probability * 100, 1),
                "Carrier %": round(ranked.carrier_probability * 100, 1),
                "Recommended action": ranked.recommended_action,
            }
        )

    if not rows:
        st.success("No at-risk relatives identified for this configuration.")
        return

    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    high = (df["Priority"] == "high").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("At-risk relatives", len(df))
    c2.metric("High priority", int(high))
    c3.metric("Medium/Low", int(len(df) - high))

    icon = {"high": "🔴", "medium": "🟠", "low": "🔵"}
    df["Priority"] = df["Priority"].map(lambda p: f"{icon.get(p, '🔵')} {p}")
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_variant_tab() -> None:
    st.subheader("Variant Annotation")
    st.caption(
        "Classify variants against the curated ClinVar/OMIM knowledge base. "
        "Paste VCF lines, or enter gene/rsID pairs (one per line, e.g. `BRCA1` "
        "or `rs334`)."
    )

    example = "rs334\nrs1800562\nBRCA1\nAPOE:rs429358\nHTT"
    text = st.text_area("Variants or VCF", value=example, height=160, key="var_text")

    rows = []
    if any(
        line.strip().startswith(("chr", "1", "2", "X", "Y")) and "\t" in line
        for line in text.splitlines()
    ):
        for pv in parse_vcf(text):
            ann = annotate_variant(gene=pv.gene, rs_id=pv.rs_id)
            rows.append(_variant_row(pv.rs_id or "", pv.gene or "", ann))
    else:
        for raw in text.splitlines():
            token = raw.strip()
            if not token:
                continue
            gene, rs_id = None, None
            for part in token.replace(":", " ").replace(",", " ").split():
                if part.lower().startswith("rs"):
                    rs_id = part
                else:
                    gene = part
            ann = annotate_variant(gene=gene, rs_id=rs_id)
            rows.append(_variant_row(rs_id or "", gene or "", ann))

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _variant_row(rs_id: str, gene: str, ann: Any) -> dict[str, Any]:
    """Build a display row for one annotated variant."""
    return {
        "rsID": rs_id,
        "Gene": gene or (ann.gene or ""),
        "Significance": ann.clinical_significance.value.replace("_", " "),
        "Condition": ann.condition_display or "—",
        "Inheritance": (ann.inheritance_mode or "—").replace("_", " "),
    }


def _render_prs_tab() -> None:
    st.subheader("Polygenic Risk Score")
    st.caption(
        "Blends a curated polygenic panel with an (optional) ML risk in log-odds "
        "space. Toggle risk-allele dosages to see the effect."
    )

    disease = st.selectbox(
        "Disease panel",
        list(PRS_PANELS.keys()),
        format_func=lambda k: PRS_PANELS[k].display,
        key="prs_disease",
    )
    panel = PRS_PANELS[str(disease)]

    use_ml = st.checkbox("Include an ML risk in the blend", value=True, key="prs_use_ml")
    ml_risk = None
    if use_ml:
        ml_risk = st.slider("ML risk", 0.0, 1.0, 0.35, 0.01, key="prs_ml")
    prs_weight = st.slider("PRS weight in blend", 0.0, 1.0, 0.4, 0.05, key="prs_w")

    st.markdown("**Risk-allele dosage per SNP**")
    dosages: dict[str, int] = {}
    cols = st.columns(min(len(panel.weights), 3) or 1)
    for i, rsid in enumerate(panel.weights):
        choice = cols[i % len(cols)].selectbox(
            f"{rsid} (β={panel.weights[rsid]:+.2f})",
            [0, 1, 2],
            index=0,
            key=f"prs_{disease}_{rsid}",
        )
        dosages[rsid] = int(choice) if choice is not None else 0

    result = compute_prs(str(disease), dosages, ml_risk=ml_risk, prs_weight=prs_weight)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PRS percentile", f"{result.percentile:.0f}")
    c2.metric("Odds ratio", f"{result.odds_ratio:.2f}")
    c3.metric("PRS risk", f"{result.prs_absolute_risk:.1%}")
    c4.metric("Blended risk", f"{result.blended_risk:.1%}")
    st.info(result.interpretation)


def render_genetics_page() -> None:
    """Render the Genetics & Genomics page."""
    st.header("🧬 Genetics & Genomics")
    st.caption(
        "The hereditary-genetics engine: Mendelian inheritance, cascade "
        "screening, variant annotation, and polygenic risk — all deterministic "
        "and explainable, sharing the exact logic used by the API."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🧮 Inheritance", "🌊 Cascade Screening", "🔬 Variants", "📈 Polygenic Risk"]
    )
    with tab1:
        _render_inheritance_tab()
    with tab2:
        _render_cascade_tab()
    with tab3:
        _render_variant_tab()
    with tab4:
        _render_prs_tab()
