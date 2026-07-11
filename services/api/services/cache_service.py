"""Redis cache wrapper for the prediction API.

Key schema
----------
``predict:hereditary:{patient_id}:{feature_date}``  — hereditary risk response JSON
``predict:symptoms:{patient_id}:{codes_hash}``       — symptom differential response JSON
``features:{patient_id}:{feature_date}``             — PatientFeatureVector JSON

All keys include a version prefix so a model version bump can be targeted
by flushing ``predict:*`` without evicting cached feature vectors.

TTLs
----
- Prediction responses:  3 600 s (1 hour)   — risk changes daily, not hourly
- Feature vectors:       86 400 s (24 hours) — recomputed nightly by feature job
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

log = logging.getLogger(__name__)

# Default TTLs in seconds
_TTL_PREDICTION = 3_600
_TTL_FEATURES = 86_400


class CacheService:
    """Thin async wrapper around a redis.asyncio client.

    Attributes:
        _client: Underlying redis.asyncio.Redis client.
    """

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    # ── Generic helpers ───────────────────────────────────────────────────────

    async def get_json(self, key: str) -> dict[str, Any] | None:
        """Retrieve and JSON-decode a cached value.

        Args:
            key: Redis key.

        Returns:
            Decoded dict, or ``None`` on a cache miss.
        """
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            log.warning("Cache key '%s' contained invalid JSON — evicting", key)
            await self._client.delete(key)
            return None

    async def set_json(self, key: str, value: dict[str, Any], ttl: int) -> None:
        """JSON-encode and store a value with a TTL.

        Args:
            key: Redis key.
            value: Serialisable dict.
            ttl: Time-to-live in seconds.
        """
        await self._client.setex(key, ttl, json.dumps(value, default=str))

    async def delete(self, key: str) -> None:
        """Delete a key from the cache.

        Args:
            key: Redis key to remove.
        """
        await self._client.delete(key)

    async def ping(self) -> bool:
        """Return True if Redis responds to PING within the client timeout.

        Returns:
            True on success, False on connection error.
        """
        try:
            await self._client.ping()
            return True
        except Exception:
            return False

    # ── Domain-specific key builders ──────────────────────────────────────────

    @staticmethod
    def hereditary_key(patient_id: str, feature_date: str) -> str:
        """Cache key for a hereditary risk prediction.

        Args:
            patient_id: Patient UUID string.
            feature_date: ISO-8601 feature date.

        Returns:
            Redis key string.
        """
        return f"predict:hereditary:{patient_id}:{feature_date}"

    @staticmethod
    def symptom_key(patient_id: str, codes: list[str]) -> str:
        """Cache key for a symptom-based differential prediction.

        Args:
            patient_id: Patient UUID string.
            codes: Symptom code list (order-normalised before hashing).

        Returns:
            Redis key string.
        """
        codes_hash = hashlib.md5(
            ",".join(sorted(codes)).encode(), usedforsecurity=False
        ).hexdigest()[:12]
        return f"predict:symptoms:{patient_id}:{codes_hash}"

    @staticmethod
    def prescription_key(patient_id: str, codes: list[str]) -> str:
        """Cache key for a prescription-based differential prediction.

        Args:
            patient_id: Patient UUID string.
            codes: Medication code list.

        Returns:
            Redis key string.
        """
        codes_hash = hashlib.md5(
            ",".join(sorted(codes)).encode(), usedforsecurity=False
        ).hexdigest()[:12]
        return f"predict:prescription:{patient_id}:{codes_hash}"

    @staticmethod
    def features_key(patient_id: str, feature_date: str) -> str:
        """Cache key for a pre-computed patient feature vector.

        Args:
            patient_id: Patient UUID string.
            feature_date: ISO-8601 feature date.

        Returns:
            Redis key string.
        """
        return f"features:{patient_id}:{feature_date}"

    @staticmethod
    def family_profile_key(patient_id: str) -> str:
        """Cache key for a patient's family risk profile.

        Args:
            patient_id: Patient UUID string.

        Returns:
            Redis key string.
        """
        return f"profile:family:{patient_id}"

    # ── TTL constants (exposed for callers) ───────────────────────────────────

    TTL_PREDICTION: int = _TTL_PREDICTION
    TTL_FEATURES: int = _TTL_FEATURES
