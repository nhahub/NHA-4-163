"""Knowledge-based differential-diagnosis inference (Tier 4).

Provides the inference backing ``/predict/disease-from-symptoms`` and
``/predict/disease-from-prescription``.  Rather than a trained multi-label
classifier (which would need a labelled EHR corpus we do not ship), this uses a
curated clinical knowledge base of symptom→disease and medication→disease
associations with likelihood weights — the same deterministic, dependency-free
approach used elsewhere in the demo system.

The design is intentionally swappable: both public functions return the same
ranked ``(disease_code, disease_name, probability)`` shape a learned model would,
so a future ML model can replace this module without touching the routers.

Coding systems:
  - Symptom inputs: ICD-10 (R-chapter symptoms) or SNOMED CT.
  - Medication inputs: RxNorm RXCUI.
  - Disease outputs: ICD-10.
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_VERSION = "kb-differential-1.0"

# Human-readable ICD-10 disease labels used in responses.
_DISEASE_NAMES: dict[str, str] = {
    "J45": "Asthma",
    "J44": "Chronic obstructive pulmonary disease",
    "J18": "Pneumonia",
    "J06": "Acute upper respiratory infection",
    "I20": "Angina pectoris",
    "I21": "Acute myocardial infarction",
    "I10": "Essential hypertension",
    "I50": "Heart failure",
    "I48": "Atrial fibrillation",
    "I25": "Chronic ischaemic heart disease",
    "K21": "Gastro-oesophageal reflux disease",
    "K29": "Gastritis",
    "E11": "Type 2 diabetes mellitus",
    "E78": "Hyperlipidaemia",
    "E03": "Hypothyroidism",
    "G43": "Migraine",
    "C50": "Malignant neoplasm of breast",
    "N39": "Urinary tract infection",
    "F32": "Depressive episode",
}


@dataclass(frozen=True)
class _Assoc:
    """A weighted association from an input code to a candidate disease."""

    disease: str
    weight: float


# ── Symptom (ICD-10 R-chapter / SNOMED) → candidate diseases ──────────────────
_SYMPTOM_KB: dict[str, list[_Assoc]] = {
    "R05.9": [_Assoc("J06", 0.5), _Assoc("J45", 0.3), _Assoc("J44", 0.2)],  # cough
    "R05": [_Assoc("J06", 0.5), _Assoc("J45", 0.3), _Assoc("J44", 0.2)],
    "R06.0": [_Assoc("J45", 0.4), _Assoc("J44", 0.3), _Assoc("I50", 0.3)],  # dyspnoea
    "R06.02": [_Assoc("J45", 0.5), _Assoc("J44", 0.3), _Assoc("I50", 0.2)],  # shortness of breath
    "R50.9": [_Assoc("J06", 0.4), _Assoc("J18", 0.4), _Assoc("N39", 0.2)],  # fever
    "J06.9": [_Assoc("J06", 0.7), _Assoc("J18", 0.3)],  # URI
    "R07.9": [_Assoc("I20", 0.4), _Assoc("I21", 0.3), _Assoc("K21", 0.3)],  # chest pain
    "R07.4": [_Assoc("I20", 0.4), _Assoc("I21", 0.3), _Assoc("K21", 0.3)],
    "R00.2": [_Assoc("I48", 0.6), _Assoc("I20", 0.4)],  # palpitations
    "R10.9": [_Assoc("K29", 0.5), _Assoc("K21", 0.5)],  # abdominal pain
    "R11": [_Assoc("K29", 0.5), _Assoc("K21", 0.3), _Assoc("G43", 0.2)],  # nausea
    "R51": [_Assoc("G43", 0.6), _Assoc("I10", 0.4)],  # headache
    "R42": [_Assoc("I10", 0.5), _Assoc("G43", 0.5)],  # dizziness
    "R63.1": [_Assoc("E11", 0.8), _Assoc("N39", 0.2)],  # polydipsia
    "R35": [_Assoc("E11", 0.7), _Assoc("N39", 0.3)],  # polyuria
    "R53.83": [_Assoc("E03", 0.4), _Assoc("F32", 0.3), _Assoc("E11", 0.3)],  # fatigue
    "R63.4": [_Assoc("E11", 0.4), _Assoc("C50", 0.3), _Assoc("F32", 0.3)],  # weight loss
}

# ── Medication (RxNorm RXCUI) → conditions the drug treats (reverse inference) ─
_MEDICATION_KB: dict[str, list[_Assoc]] = {
    "860975": [_Assoc("E11", 1.0)],  # Metformin
    "197361": [_Assoc("I10", 0.6), _Assoc("I50", 0.4)],  # Lisinopril
    "314076": [_Assoc("I10", 0.6), _Assoc("I50", 0.4)],  # Lisinopril 10mg
    "895994": [_Assoc("J45", 0.6), _Assoc("J44", 0.4)],  # Albuterol
    "262105": [_Assoc("C50", 1.0)],  # Tamoxifen
    "617314": [_Assoc("E78", 0.6), _Assoc("I25", 0.4)],  # Atorvastatin
    "855332": [_Assoc("I48", 0.6), _Assoc("I25", 0.4)],  # Warfarin
    "310798": [_Assoc("E03", 1.0)],  # Levothyroxine
    "310965": [_Assoc("I20", 0.5), _Assoc("I25", 0.5)],  # Isosorbide/nitrate
    "197381": [_Assoc("F32", 1.0)],  # Sertraline/SSRI
    "308136": [_Assoc("N39", 0.7), _Assoc("J18", 0.3)],  # Amoxicillin
}


def _rank(
    codes: list[str], kb: dict[str, list[_Assoc]], top_n: int
) -> list[tuple[str, str, float]]:
    """Aggregate association weights across input codes and rank diseases.

    Args:
        codes: Input codes (symptom or medication).
        kb: The knowledge base to consult.
        top_n: Maximum number of diseases to return.

    Returns:
        Ranked list of ``(disease_code, disease_name, probability)`` where the
        probabilities are normalised to sum to 1.0 across the returned set.
        Empty if none of the input codes are recognised.
    """
    scores: dict[str, float] = {}
    for raw in codes:
        code = raw.strip().upper()
        # Try exact, then the 3-char ICD-10 category (e.g. "R05.9" → "R05").
        assocs = kb.get(code) or kb.get(code.split(".")[0])
        if not assocs:
            continue
        for a in assocs:
            scores[a.disease] = scores.get(a.disease, 0.0) + a.weight

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    total = sum(w for _, w in ranked)
    return [
        (disease, _DISEASE_NAMES.get(disease, disease), round(w / total, 4))
        for disease, w in ranked
    ]


def infer_from_symptoms(symptom_codes: list[str], top_n: int = 5) -> list[tuple[str, str, float]]:
    """Return a ranked differential diagnosis from symptom codes.

    Args:
        symptom_codes: ICD-10 / SNOMED symptom codes.
        top_n: Maximum number of candidate diseases.

    Returns:
        Ranked ``(disease_code, disease_name, probability)`` tuples.
    """
    return _rank(symptom_codes, _SYMPTOM_KB, top_n)


def infer_from_medications(
    medication_codes: list[str], top_n: int = 5
) -> list[tuple[str, str, float]]:
    """Return likely underlying conditions from active medication codes.

    Args:
        medication_codes: RxNorm RXCUI codes.
        top_n: Maximum number of candidate conditions.

    Returns:
        Ranked ``(disease_code, disease_name, probability)`` tuples.
    """
    return _rank(medication_codes, _MEDICATION_KB, top_n)
