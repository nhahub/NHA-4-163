"""Feature group registry for the patient feature store.

Defines the canonical groups, their Delta table paths, and the ordered
list of feature columns.  The ML training scripts and serving layer both
import from here so column names never drift.

Delta table layout (relative to MINIO_BUCKET_DELTA):
    features/patient_demographics/
    features/patient_comorbidities/
    features/patient_medications/
    features/patient_graph/
    features/patient_feature_vector/   ← joined, training-ready

All tables are partitioned by ``feature_date`` (YYYY-MM-DD) and keyed
by ``patient_id`` (UUID string).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureGroup:
    """Metadata for one feature group stored as a Delta table.

    Attributes:
        name: Logical group name, used as the Delta table directory name.
        description: Human-readable purpose of this feature group.
        delta_path: Path relative to the Delta bucket root.
        key_column: Primary join key (always ``patient_id``).
        partition_column: Partition column (always ``feature_date``).
        feature_columns: Ordered list of numeric/categorical feature columns.
    """

    name: str
    description: str
    delta_path: str
    key_column: str = "patient_id"
    partition_column: str = "feature_date"
    feature_columns: list[str] = field(default_factory=list)


DEMOGRAPHICS = FeatureGroup(
    name="patient_demographics",
    description="Age, age group, and gender binary flags derived from the patient table.",
    delta_path="features/patient_demographics",
    feature_columns=[
        "age_years",
        "age_group",
        "gender_male",
        "gender_female",
        "gender_other_unknown",
    ],
)

COMORBIDITIES = FeatureGroup(
    name="patient_comorbidities",
    description="Active condition counts and ICD-10-chapter-level binary flags.",
    delta_path="features/patient_comorbidities",
    feature_columns=[
        "comorbidity_count",
        "hereditary_condition_count",
        "has_cardiovascular",
        "has_digestive",
        "has_genitourinary",
        "has_haematological",
        "has_infectious",
        "has_mental_health",
        "has_metabolic",
        "has_musculoskeletal",
        "has_neurological",
        "has_oncological",
        "has_respiratory",
    ],
)

MEDICATIONS = FeatureGroup(
    name="patient_medications",
    description="Medication counts and crude adherence proxy from medication_requests.",
    delta_path="features/patient_medications",
    feature_columns=[
        "active_medication_count",
        "completed_medication_count",
        "stopped_medication_count",
        "distinct_medication_count",
        "adherence_proxy",
    ],
)

GRAPH = FeatureGroup(
    name="patient_graph",
    description=(
        "Family graph features: weighted disease prevalence, affected relative counts, "
        "shortest path, family size, and GDS clustering coefficient."
    ),
    delta_path="features/patient_graph",
    feature_columns=[
        "affected_relatives_count",
        "weighted_family_prevalence",
        "first_degree_affected_count",
        "second_degree_affected_count",
        "shortest_path_to_affected",
        "family_size",
        "family_clustering_coefficient",
    ],
)

FEATURE_VECTOR = FeatureGroup(
    name="patient_feature_vector",
    description="Joined, training-ready feature vector combining all four feature groups.",
    delta_path="features/patient_feature_vector",
    feature_columns=(
        DEMOGRAPHICS.feature_columns
        + COMORBIDITIES.feature_columns
        + MEDICATIONS.feature_columns
        + GRAPH.feature_columns
    ),
)

# All source groups (not the joined vector).
ALL_GROUPS: list[FeatureGroup] = [DEMOGRAPHICS, COMORBIDITIES, MEDICATIONS, GRAPH]

# Flat lookup: feature column name → owning group name.
FEATURE_TO_GROUP: dict[str, str] = {
    col: grp.name for grp in ALL_GROUPS for col in grp.feature_columns
}
