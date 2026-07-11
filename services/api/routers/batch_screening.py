"""Batch risk screening endpoints.

POST /predict/batch-screen           — Submit a batch screening job
GET  /predict/batch-screen/{job_id}  — Poll job status and results

Jobs run as background asyncio tasks using the in-process ModelService.
State is stored in Redis for polling.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from services.api.deps import CacheDep
from services.api.schemas.crud_schemas import (
    BatchScreenJobResponse,
    BatchScreenRequest,
    BatchScreenResultResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["batch-screening"])

# Redis key schema for batch jobs
_JOB_KEY = "batch:{job_id}"
_JOB_TTL = 86_400  # 24 hours


def _risk_tier(score: float) -> str:
    """Map a score to a risk tier."""
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "moderate"
    if score < 0.75:
        return "high"
    return "very_high"


@router.post(
    "/batch-screen",
    response_model=BatchScreenJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a batch screening job",
)
async def submit_batch_screen(
    body: BatchScreenRequest,
    cache: CacheDep,
    background_tasks: BackgroundTasks,
) -> BatchScreenJobResponse:
    """Submit a batch risk screening job for multiple patients.

    Returns immediately with a job_id. Use GET to poll for results.

    The job runs as a background asyncio task. For this demo implementation,
    it generates synthetic risk scores. In production, it would iterate
    patients and call ModelService.predict_proba() for each.

    Args:
        body: Batch screening request with patient IDs or filter criteria.
        cache: Redis cache service.
        background_tasks: FastAPI background task manager.

    Returns:
        BatchScreenJobResponse with job_id and status=pending.
    """
    job_id = str(uuid.uuid4())[:8]

    # Determine patient list
    patient_ids = body.patient_ids or []
    total = len(patient_ids) if patient_ids else 0

    # If using filters instead of explicit IDs, we'd query the DB here
    # For demo, if no IDs provided, set a placeholder count
    if total == 0 and (body.filter_gender or body.filter_min_age is not None):
        total = 50  # Demo: simulate filtered patient count

    if total == 0:
        raise HTTPException(
            status_code=422,
            detail="Provide patient_ids or filter criteria to screen",
        )

    # Store initial job state in Redis
    job_state = {
        "status": "pending",
        "total": total,
        "progress": 0,
        "results": [],
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "message": "Job queued",
        "patient_ids": [str(pid) for pid in patient_ids] if patient_ids else [],
        "include_shap": body.include_shap,
    }
    await cache.set_json(f"batch:{job_id}", job_state, _JOB_TTL)

    # Launch background task
    background_tasks.add_task(_run_batch_job, job_id, cache, job_state)

    return BatchScreenJobResponse(
        job_id=job_id,
        status="pending",
        total_patients=total,
        progress=0,
        message="Batch screening job submitted",
    )


async def _run_batch_job(job_id: str, cache: CacheDep, job_state: dict[str, Any]) -> None:
    """Background task that processes patients and computes risk scores.

    In production this would call ModelService.predict_proba() for each
    patient. For demo/standalone mode, generates synthetic scores.

    Args:
        job_id: Unique job identifier.
        cache: Redis cache service for state updates.
        job_state: Initial job state dict.
    """
    import random

    key = f"batch:{job_id}"
    patient_ids = job_state.get("patient_ids", [])
    total = job_state["total"]

    # If no explicit IDs, generate demo patient IDs
    if not patient_ids:
        patient_ids = [str(uuid.uuid4()) for _ in range(total)]

    job_state["status"] = "running"
    job_state["message"] = "Processing patients..."
    await cache.set_json(key, job_state, _JOB_TTL)

    results = []
    try:
        for i, pid in enumerate(patient_ids):
            # Simulate prediction (in production: call model service)
            score = round(random.betavariate(2, 5), 4)  # Realistic-looking distribution
            result = {
                "patient_id": pid,
                "risk_score": score,
                "risk_tier": _risk_tier(score),
            }

            if job_state.get("include_shap"):
                result["shap_factors"] = [
                    {
                        "feature": "family_history_count",
                        "shap_value": round(random.uniform(-0.2, 0.3), 3),
                    },
                    {"feature": "age_years", "shap_value": round(random.uniform(-0.1, 0.2), 3)},
                    {
                        "feature": "condition_count",
                        "shap_value": round(random.uniform(-0.15, 0.15), 3),
                    },
                ]

            results.append(result)

            # Update progress every 10 patients
            if (i + 1) % 10 == 0 or (i + 1) == len(patient_ids):
                job_state["progress"] = i + 1
                job_state["results"] = results
                await cache.set_json(key, job_state, _JOB_TTL)

            # Small delay to simulate real processing
            await asyncio.sleep(0.01)

        job_state["status"] = "completed"
        job_state["progress"] = len(patient_ids)
        job_state["results"] = results
        job_state["completed_at"] = datetime.now(UTC).isoformat()
        job_state["message"] = f"Screening complete: {len(results)} patients processed"
        await cache.set_json(key, job_state, _JOB_TTL)
        log.info("Batch job %s completed: %d patients", job_id, len(results))

    except Exception as exc:
        job_state["status"] = "failed"
        job_state["message"] = f"Job failed: {str(exc)}"
        await cache.set_json(key, job_state, _JOB_TTL)
        log.error("Batch job %s failed: %s", job_id, exc)


@router.get(
    "/batch-screen/{job_id}",
    response_model=BatchScreenResultResponse,
    summary="Poll batch screening job",
)
async def get_batch_screen_result(job_id: str, cache: CacheDep) -> BatchScreenResultResponse:
    """Poll a batch screening job for status and results.

    Args:
        job_id: Job identifier from POST response.
        cache: Redis cache service.

    Returns:
        BatchScreenResultResponse with current status and results.

    Raises:
        HTTPException 404: Job not found or expired.
    """
    job_state = await cache.get_json(f"batch:{job_id}")
    if job_state is None:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    return BatchScreenResultResponse(
        job_id=job_id,
        status=job_state.get("status", "unknown"),
        total_patients=job_state.get("total", 0),
        progress=job_state.get("progress", 0),
        results=job_state.get("results", []),
        started_at=job_state.get("started_at"),
        completed_at=job_state.get("completed_at"),
        message=job_state.get("message", ""),
    )
