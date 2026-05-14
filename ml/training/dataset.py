"""Feature loading, label construction, and train/val/test splitting.

Data flow
---------
1. Load feature vector from Delta feature store or synthetic data
2. Load labels from PostgreSQL or generate synthetically
3. Build dataset: join features + labels, encode categoricals, impute nulls
4. Split by patient_id to prevent leakage between family members
5. Return train/val/test splits as numpy arrays + metadata
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from ml.features.registry import FEATURE_VECTOR

log = logging.getLogger(__name__)

# Age groups in ordinal order for encoding.
_AGE_GROUP_ORDER: list[str] = ["0s", "10s", "20s", "30s", "40s", "50s", "60s", "70s", "80s", "90+"]

# Numeric columns that may contain nulls — imputed with column median.
_NULLABLE_NUMERIC_COLS: list[str] = [
    "adherence_proxy",
    "family_clustering_coefficient",
    "shortest_path_to_affected",
]


# ── Synthetic data for development ────────────────────────────────────────────

def create_synthetic_dataset(
    n_patients: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create synthetic patient features and labels.

    Args:
        n_patients: Number of synthetic patients
        random_state: Random seed

    Returns:
        X: Features DataFrame
        y: Labels Series
    """
    np.random.seed(random_state)

    X = pd.DataFrame({
        "patient_id": np.arange(n_patients),
        "age_years": np.random.randint(18, 85, n_patients),
        "gender_male": np.random.randint(0, 2, n_patients),
        "gender_female": np.random.randint(0, 2, n_patients),
        "comorbidity_count": np.random.randint(0, 10, n_patients),
        "hereditary_condition_count": np.random.randint(0, 5, n_patients),
        "has_cardiovascular": np.random.randint(0, 2, n_patients),
        "has_metabolic": np.random.randint(0, 2, n_patients),
        "has_neurological": np.random.randint(0, 2, n_patients),
        "has_oncological": np.random.randint(0, 2, n_patients),
        "active_medication_count": np.random.randint(0, 15, n_patients),
        "shortest_path_to_affected": np.random.randint(-1, 5, n_patients),
        "family_risk_prevalence": np.random.uniform(0, 1, n_patients),
    })

    # Generate labels with signal
    risk_score = (
        0.3 * (X["hereditary_condition_count"] > 0).astype(int)
        + 0.2 * (X["shortest_path_to_affected"] >= 0).astype(int)
        + 0.2 * (X["family_risk_prevalence"] > 0.5).astype(int)
        + 0.1 * (X["age_years"] > 60).astype(int)
        + 0.2 * np.random.random(n_patients)
    )

    y = (risk_score > 0.5).astype(int)
    return X, y


def load_feature_data(
    n_patients: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load or generate features and split by patient to prevent leakage.

    Used by the Streamlit UI for interactive evaluation. Generates synthetic
    data when production services are unavailable.

    Args:
        n_patients: Number of synthetic patients to generate.
        random_state: Random seed for reproducibility.

    Returns:
        (X_train, X_val, y_train, y_val) split by patient_id.
    """
    np.random.seed(random_state)

    # Load or generate data
    X, y = create_synthetic_dataset(n_patients=n_patients, random_state=random_state)

    # Split by patient_id to prevent leakage
    patient_ids = X["patient_id"].unique()
    train_patients = np.random.choice(
        patient_ids,
        size=int(0.7 * len(patient_ids)),
        replace=False,
    )
    train_mask = X["patient_id"].isin(train_patients)

    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)

    X_val = X[~train_mask].reset_index(drop=True)
    y_val = y[~train_mask].reset_index(drop=True)

    log.info("Train: %d, Val: %d", len(X_train), len(X_val))

    return X_train, X_val, y_train, y_val


# ── Feature loading (production) ─────────────────────────────────────────────

def load_feature_vector(
    source: str | pd.DataFrame,
    feature_date: str,
    spark: Any = None,
) -> pd.DataFrame:
    """Load the joined patient feature vector for a given date.

    Args:
        source: Either a Delta table S3A path (str) or a pre-loaded
            ``pd.DataFrame`` (used in tests to avoid Spark).
        feature_date: ISO-8601 date string used to filter the partition.
        spark: Active ``SparkSession``.  Required when ``source`` is a path.

    Returns:
        DataFrame with ``patient_id`` column and all feature columns.

    Raises:
        ValueError: If ``source`` is a path but ``spark`` is None.
    """
    if isinstance(source, pd.DataFrame):
        if "feature_date" in source.columns:
            return source[source["feature_date"] == feature_date].drop(
                columns=["feature_date"], errors="ignore"
            )
        return source

    if spark is None:
        raise ValueError("spark session is required when source is a path")

    df: pd.DataFrame = (
        spark.read.format("delta")
        .load(source)
        .filter(f"feature_date = '{feature_date}'")
        .drop("feature_date")
        .toPandas()
    )
    log.info("Loaded %d feature rows for feature_date=%s", len(df), feature_date)
    return df


# ── Label loading ─────────────────────────────────────────────────────────────

def load_labels(dsn: str) -> pd.DataFrame:
    """Load binary hereditary-disease labels from PostgreSQL.

    A patient is labelled positive (1) if they have at least one condition
    with ``is_hereditary = TRUE`` and ``clinical_status`` in
    ('active', 'recurrence', 'relapse').

    Args:
        dsn: PostgreSQL DSN string (``postgresql://user:pw@host:port/db``).

    Returns:
        DataFrame with columns ``patient_id`` (str) and ``label`` (int 0/1).
    """
    import psycopg2
    import psycopg2.extras

    query = """
        SELECT
            p.id::text AS patient_id,
            CASE
                WHEN COUNT(c.id) FILTER (
                    WHERE c.is_hereditary = TRUE
                      AND c.clinical_status IN ('active','recurrence','relapse')
                ) > 0 THEN 1
                ELSE 0
            END AS label
        FROM patients p
        LEFT JOIN conditions c ON c.patient_id = p.id
        WHERE p.deleted_at IS NULL
        GROUP BY p.id
    """
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    pos = int(df["label"].sum())
    log.info("Labels loaded: %d total, %d positive (%.1f%%)", len(df), pos, 100 * pos / len(df))
    return df


# ── Dataset assembly ──────────────────────────────────────────────────────────

def build_dataset(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Join features and labels, encode categoricals, impute nulls.

    Args:
        features_df: Feature DataFrame with ``patient_id`` and feature columns.
        labels_df: Labels DataFrame with ``patient_id`` and ``label`` columns.

    Returns:
        Tuple of:
        - X: float32 numpy array, shape (n_patients × n_features)
        - y: int32 label array, shape (n_patients,)
        - feature_names: Ordered list of feature column names (post-encoding)
        - patient_ids: String array of patient UUIDs aligned with X/y rows

    Raises:
        ValueError: If the join produces zero rows.
    """
    merged = features_df.merge(labels_df, on="patient_id", how="inner")
    if merged.empty:
        raise ValueError("Feature-label join is empty — check that patient_ids match")

    log.info("Dataset: %d patients after join", len(merged))

    # Ordinal-encode age_group if present
    if "age_group" in merged.columns:
        enc = OrdinalEncoder(
            categories=[_AGE_GROUP_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        merged["age_group"] = enc.fit_transform(merged[["age_group"]]).astype(np.float32)

    # Impute nullable numeric columns with train-set median (computed here on full set;
    # callers must ensure imputation is re-fit on train split only — see train_*.py).
    for col in _NULLABLE_NUMERIC_COLS:
        if col in merged.columns:
            median_val = merged[col].median()
            merged[col] = merged[col].fillna(median_val)

    feat_cols = [c for c in FEATURE_VECTOR.feature_columns if c in merged.columns]
    X = merged[feat_cols].values.astype(np.float32)
    y = merged["label"].values.astype(np.int32)
    patient_ids = merged["patient_id"].values.astype(str)

    log.info("Feature matrix: %s  positive rate: %.2f%%", X.shape, 100 * y.mean())
    return X, y, feat_cols, patient_ids


# ── Train / val / test split ──────────────────────────────────────────────────

def patient_id_split(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split of unique patient IDs into train / val / test.

    Splits are computed on unique patient IDs to guarantee no patient
    appears in two splits.  The relative test_size is adjusted to be
    proportional to the remaining data after the val split is removed.

    Args:
        patient_ids: 1-D array of patient UUID strings aligned with labels.
        labels: 1-D binary label array aligned with patient_ids.
        val_size: Fraction of all patients for validation.
        test_size: Fraction of all patients for test.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_ids, val_ids, test_ids) patient ID arrays.
    """
    ids, idx = np.unique(patient_ids, return_index=True)
    y_unique = labels[idx]

    trainval_ids, test_ids, trainval_y, _ = train_test_split(
        ids, y_unique,
        test_size=test_size,
        stratify=y_unique,
        random_state=random_state,
    )
    adjusted_val = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        trainval_ids,
        test_size=adjusted_val,
        stratify=trainval_y,
        random_state=random_state,
    )
    log.info(
        "Split — train: %d  val: %d  test: %d",
        len(train_ids), len(val_ids), len(test_ids),
    )
    return train_ids, val_ids, test_ids


def apply_split(
    patient_ids: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    split_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select rows from X and y corresponding to the given patient IDs.

    Args:
        patient_ids: Full patient ID array aligned with X/y.
        X: Feature matrix.
        y: Label array.
        split_ids: Patient IDs for the desired split.

    Returns:
        Tuple of (X_split, y_split) filtered to the requested IDs.
    """
    mask = np.isin(patient_ids, split_ids)
    return X[mask], y[mask]


# ── PyG graph construction (GNN only) ────────────────────────────────────────

def build_pyg_data(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> tuple[Any, dict[str, int]]:
    """Build a PyTorch Geometric Data object from features and Neo4j edges.

    Queries Neo4j for all family relationships between Patient nodes and
    constructs edge_index as a long tensor of shape (2 × num_edges).
    Nodes without any edge remain in the graph but contribute no
    neighbourhood information (isolated nodes).

    Args:
        X: Feature matrix (n_patients × n_features).
        y: Binary label array (n_patients,).
        patient_ids: Patient UUID array aligned with X/y rows.
        train_ids: Training patient IDs for the train_mask.
        val_ids: Validation patient IDs for the val_mask.
        test_ids: Test patient IDs for the test_mask.
        neo4j_uri: Bolt URI for Neo4j.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.

    Returns:
        Tuple of (``torch_geometric.data.Data``, pid_to_idx mapping).

    Raises:
        ImportError: If torch / torch_geometric are not installed.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("torch and torch_geometric are required for GNN") from exc

    from neo4j import GraphDatabase

    pid_to_idx: dict[str, int] = {pid: i for i, pid in enumerate(patient_ids)}

    # Load family edges from Neo4j
    _EDGE_QUERY = """
    MATCH (p:Patient)-[:HAS_RELATIVE|IS_PARENT_OF|HAS_CHILD|IS_SIBLING_OF]-(q:Patient)
    WHERE p.id < q.id
    RETURN p.id AS src, q.id AS tgt
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    edges: list[tuple[int, int]] = []
    try:
        with driver.session() as session:
            for record in session.run(_EDGE_QUERY):
                src_id, tgt_id = record["src"], record["tgt"]
                if src_id in pid_to_idx and tgt_id in pid_to_idx:
                    src_idx = pid_to_idx[src_id]
                    tgt_idx = pid_to_idx[tgt_id]
                    edges.append((src_idx, tgt_idx))
                    edges.append((tgt_idx, src_idx))  # undirected
    finally:
        driver.close()

    log.info("Graph: %d nodes, %d directed edges", len(patient_ids), len(edges))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    train_mask = torch.tensor([pid in set(train_ids.tolist()) for pid in patient_ids])
    val_mask = torch.tensor([pid in set(val_ids.tolist()) for pid in patient_ids])
    test_mask = torch.tensor([pid in set(test_ids.tolist()) for pid in patient_ids])

    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return data, pid_to_idx
