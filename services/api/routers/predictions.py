"""Prediction endpoints.

POST /predict/hereditary-risk
    Predicts the calibrated probability that a patient carries a heritable
    disease risk given their demographics, comorbidities, medications, and
    family graph.  Returns SHAP explanations when requested.

POST /predict/disease-from-symptoms
    Returns a ranked differential diagnosis from ICD-10/SNOMED symptom codes
    using the knowledge-based model in ``differential_service``.  Enabled by
    default; returns HTTP 503 only when ENABLE_SYMPTOM_MODEL=false.

POST /predict/disease-from-prescription
    Returns likely underlying conditions from RxNorm medication codes.
    Enabled by default; returns HTTP 503 only when ENABLE_SYMPTOM_MODEL=false.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import date

from fastapi import APIRouter, HTTPException, Request, status

from libs.common.config import get_settings
from libs.common.models.prediction_log import PredictionLog
from services.api.db import DbSession
from services.api.deps import CacheDep, ModelDep
from services.api.schemas.requests import (
    PredictFromPrescriptionRequest,
    PredictFromSymptomsRequest,
    PredictHeredityRiskRequest,
)
from services.api.schemas.responses import (
    DifferentialDiagnosisResponse,
    DiseasePrediction,
    HeredityRiskResponse,
    SHAPContribution,
    _risk_tier,
)
from services.api.services.differential_service import (
    MODEL_VERSION as DIFFERENTIAL_MODEL_VERSION,
)
from services.api.services.differential_service import (
    infer_from_medications,
    infer_from_symptoms,
)
from services.api.services.feature_service import compute_features
from services.api.services.notification_service import evaluate_patient_notifications

log = logging.getLogger(__name__)


def _symptom_model_enabled() -> bool:
    """Return whether the differential-diagnosis model is enabled.

    The knowledge-based model ships ready-to-serve, so it is enabled unless an
    operator explicitly sets ``ENABLE_SYMPTOM_MODEL=false``.
    """
    return os.environ.get("ENABLE_SYMPTOM_MODEL", "true").lower() != "false"


router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post(
    "/hereditary-risk",
    response_model=HeredityRiskResponse,
    summary="Predict hereditary disease risk",
    description=(
        "Returns a calibrated probability score and optional SHAP explanations "
        "indicating which features most drive the prediction for this patient."
    ),
)
async def predict_hereditary_risk(
    body: PredictHeredityRiskRequest,
    model: ModelDep,
    cache: CacheDep,
    db: DbSession,
    request: Request,
) -> HeredityRiskResponse:
    """Predict hereditary disease risk for a patient.

    **Cache behaviour**: responses are cached in Redis for 1 hour keyed
    by ``(patient_id, feature_date)``.  Force a fresh prediction by
    passing a different ``feature_date``.

    Args:
        body: Request body with patient_id and options.
        model: Injected ModelService.
        cache: Injected CacheService.
        db: Async database session.
        request: FastAPI request (used for request_id from state).

    Returns:
        HeredityRiskResponse with score, tier, and SHAP explanations.

    Raises:
        HTTPException 404: If the patient is not found in Postgres.
        HTTPException 503: If the model is not loaded.
    """
    if not model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hereditary risk model is not available — check MLflow registry",
        )

    patient_id = str(body.patient_id)
    feat_date = body.feature_date or str(date.today())
    request_id = str(getattr(request.state, "request_id", uuid.uuid4()))

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key = cache.hereditary_key(patient_id, feat_date)
    cached = await cache.get_json(cache_key)
    if cached:
        cached["cached"] = True
        cached["request_id"] = request_id
        return HeredityRiskResponse(**cached)

    # ── Feature computation ───────────────────────────────────────────────────
    settings = get_settings()
    pg = settings.postgres
    n4j = settings.neo4j

    try:
        features = await compute_features(
            patient_id=patient_id,
            postgres_dsn=pg.sync_dsn,
            neo4j_uri=n4j.uri,
            neo4j_user=n4j.user,
            neo4j_password=n4j.password.get_secret_value(),
            as_of_date=feat_date,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        ) from exc

    # ── Inference ─────────────────────────────────────────────────────────────
    risk_score = await model.predict_proba(features)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    shap_contributions: list[SHAPContribution] | None = None
    if body.include_shap:
        try:
            raw_shap = await model.shap_values(features, top_n=body.top_n_factors)
            shap_contributions = [SHAPContribution(**s) for s in raw_shap]
        except Exception as exc:
            # SHAP failure is non-fatal — return the prediction without factors.
            log.debug("SHAP computation failed: %s", exc)

    assert model.info is not None
    result = HeredityRiskResponse(
        request_id=uuid.UUID(request_id),
        patient_id=body.patient_id,
        risk_score=risk_score,
        risk_tier=_risk_tier(risk_score),
        top_risk_factors=shap_contributions,
        feature_date=feat_date,
        model_name=model.info.model_name,
        model_version=model.info.version,
        cached=False,
    )

    # ── Populate cache ────────────────────────────────────────────────────────
    payload = result.model_dump(mode="json")
    payload.pop("request_id", None)  # Don't cache the per-request ID
    await cache.set_json(cache_key, payload, cache.TTL_PREDICTION)

    # ── Auto-log to DB ────────────────────────────────────────────────────────
    prediction_log = PredictionLog(
        patient_id=body.patient_id,
        risk_score=risk_score,
        risk_tier=result.risk_tier,
        model_name=result.model_name,
        model_version=result.model_version,
        feature_date=feat_date,
        shap_top_factors=(
            [s.model_dump() for s in result.top_risk_factors] if result.top_risk_factors else None
        ),
        source="api",
    )
    db.add(prediction_log)
    await db.flush()

    # ── Risk-threshold notifications (non-fatal) ──────────────────────────────
    try:
        await evaluate_patient_notifications(db, body.patient_id)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Notification evaluation failed for %s: %s", patient_id, exc)

    return result


@router.post(
    "/disease-from-symptoms",
    response_model=DifferentialDiagnosisResponse,
    summary="Differential diagnosis from symptoms",
    description=(
        "Returns a ranked differential diagnosis from ICD-10 or SNOMED symptom codes. "
        "Requires ENABLE_SYMPTOM_MODEL=true."
    ),
)
async def predict_from_symptoms(
    body: PredictFromSymptomsRequest,
    cache: CacheDep,
    request: Request,
) -> DifferentialDiagnosisResponse:
    """Differential diagnosis from symptom codes.

    Returns HTTP 503 when the symptom model is disabled (``ENABLE_SYMPTOM_MODEL=false``).

    Args:
        body: Request body.
        cache: Injected CacheService.
        request: FastAPI request.

    Returns:
        DifferentialDiagnosisResponse with ranked disease predictions.

    Raises:
        HTTPException 503: When ENABLE_SYMPTOM_MODEL is not true.
    """
    if not _symptom_model_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Symptom-based prediction model is disabled in this deployment "
                "(ENABLE_SYMPTOM_MODEL=false)."
            ),
        )

    patient_id = str(body.patient_id)
    request_id = str(getattr(request.state, "request_id", uuid.uuid4()))
    cache_key = cache.symptom_key(patient_id, body.symptom_codes)

    cached = await cache.get_json(cache_key)
    if cached:
        cached["cached"] = True
        cached["request_id"] = request_id
        return DifferentialDiagnosisResponse(**cached)

    predictions = [
        DiseasePrediction(disease_code=code, disease_name=name, probability=prob)
        for code, name, prob in infer_from_symptoms(body.symptom_codes, top_n=body.top_n)
    ]
    result = DifferentialDiagnosisResponse(
        request_id=uuid.UUID(request_id),
        patient_id=body.patient_id,
        input_codes=body.symptom_codes,
        input_type="symptoms",
        predictions=predictions,
        model_version=DIFFERENTIAL_MODEL_VERSION,
        cached=False,
    )

    payload = result.model_dump(mode="json")
    payload.pop("request_id", None)
    await cache.set_json(cache_key, payload, cache.TTL_PREDICTION)

    return result


@router.post(
    "/disease-from-prescription",
    response_model=DifferentialDiagnosisResponse,
    summary="Differential diagnosis from active prescriptions",
    description=(
        "Returns likely underlying conditions from RxNorm medication codes. "
        "Requires ENABLE_SYMPTOM_MODEL=true."
    ),
)
async def predict_from_prescription(
    body: PredictFromPrescriptionRequest,
    cache: CacheDep,
    request: Request,
) -> DifferentialDiagnosisResponse:
    """Differential diagnosis from medication codes.

    Args:
        body: Request body.
        cache: Injected CacheService.
        request: FastAPI request.

    Returns:
        DifferentialDiagnosisResponse.

    Raises:
        HTTPException 503: When the model is not available.
    """
    if not _symptom_model_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Prescription-based prediction model is disabled in this "
                "deployment (ENABLE_SYMPTOM_MODEL=false)."
            ),
        )

    patient_id = str(body.patient_id)
    request_id = str(getattr(request.state, "request_id", uuid.uuid4()))
    cache_key = cache.prescription_key(patient_id, body.medication_codes)

    cached = await cache.get_json(cache_key)
    if cached:
        cached["cached"] = True
        cached["request_id"] = request_id
        return DifferentialDiagnosisResponse(**cached)

    predictions = [
        DiseasePrediction(disease_code=code, disease_name=name, probability=prob)
        for code, name, prob in infer_from_medications(body.medication_codes, top_n=body.top_n)
    ]
    result = DifferentialDiagnosisResponse(
        request_id=uuid.UUID(request_id),
        patient_id=body.patient_id,
        input_codes=body.medication_codes,
        input_type="prescriptions",
        predictions=predictions,
        model_version=DIFFERENTIAL_MODEL_VERSION,
        cached=False,
    )

    payload = result.model_dump(mode="json")
    payload.pop("request_id", None)
    await cache.set_json(cache_key, payload, cache.TTL_PREDICTION)

    return result
