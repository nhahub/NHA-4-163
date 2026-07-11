"""PredictionLog ORM model — persists every prediction for longitudinal tracking.

Every call to ``/predict/hereditary-risk`` (and batch screening) writes a
row to this table so clinicians can track how a patient's risk evolves
over time as new conditions, family data, or model versions change.

Unlike the Redis prediction cache (which expires after 1 hour), this table
is permanent and append-only for compliance and trending.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from libs.common.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class PredictionLog(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One prediction event — never updated, never deleted.

    Attributes:
        patient_id: Patient who was scored.
        risk_score: Calibrated probability in [0, 1].
        risk_tier: Categorical tier (low/moderate/high/very_high).
        model_name: MLflow registered model name.
        model_version: MLflow model version string.
        feature_date: ISO-8601 date of the feature snapshot.
        shap_top_factors: JSON array of top-N SHAP contributions.
        source: How the prediction was triggered (api/batch/scheduled).
        predicted_at: Timestamp of the prediction.
    """

    __tablename__ = "prediction_log"

    # ── Foreign keys ──────────────────────────────────────────────────────────
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Prediction results ────────────────────────────────────────────────────
    risk_score: Mapped[float] = mapped_column(Numeric(precision=6, scale=5), nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # ── Model metadata ────────────────────────────────────────────────────────
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    feature_date: Mapped[str] = mapped_column(String(10), nullable=False)

    # ── SHAP explanations ─────────────────────────────────────────────────────
    shap_top_factors: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)

    # ── Source ────────────────────────────────────────────────────────────────
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="api", index=True)

    # ── Temporal ──────────────────────────────────────────────────────────────
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
