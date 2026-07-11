"""Application configuration via environment variables.

Uses Pydantic Settings v2 so every config value is type-validated at startup.
Missing required values raise a ``ValidationError`` with a clear message before
the application accepts any traffic.

Usage::

    from libs.common.config import get_settings

    settings = get_settings()
    print(settings.postgres_host)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    db: str = "healthcare"
    user: str = "healthcare_app"
    password: SecretStr = Field(..., description="Postgres password — required")

    @property
    def dsn(self) -> str:
        """Async DSN for asyncpg / SQLAlchemy async engine."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    @property
    def sync_dsn(self) -> str:
        """Sync DSN for Alembic migrations."""
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class Neo4jSettings(BaseSettings):
    """Neo4j connection settings."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_", env_file=".env", extra="ignore")

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = Field(..., description="Neo4j password — required")


class KafkaSettings(BaseSettings):
    """Kafka connection settings."""

    model_config = SettingsConfigDict(env_prefix="KAFKA_", env_file=".env", extra="ignore")

    bootstrap_servers: str = "localhost:9092"
    schema_registry_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8081")


class MinioSettings(BaseSettings):
    """MinIO / S3-compatible object storage settings."""

    model_config = SettingsConfigDict(env_prefix="MINIO_", env_file=".env", extra="ignore")

    endpoint: AnyHttpUrl = AnyHttpUrl("http://localhost:9000")
    access_key: str = "minioadmin"
    secret_key: SecretStr = Field(..., description="MinIO secret key — required")
    bucket_raw: str = "healthcare-raw"
    bucket_delta: str = "healthcare-delta"


class MLflowSettings(BaseSettings):
    """MLflow tracking settings."""

    model_config = SettingsConfigDict(env_prefix="MLFLOW_", env_file=".env", extra="ignore")

    tracking_uri: AnyHttpUrl = AnyHttpUrl("http://localhost:5000")
    experiment_name: str = "hereditary-disease-prediction"


class RedisSettings(BaseSettings):
    """Redis cache settings."""

    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 6379
    password: SecretStr = Field(..., description="Redis password — required")

    @property
    def url(self) -> str:
        """Redis URL with password included."""
        return f"redis://:{self.password.get_secret_value()}@{self.host}:{self.port}/0"


class JWTSettings(BaseSettings):
    """JSON Web Token settings for API authentication."""

    model_config = SettingsConfigDict(env_prefix="JWT_", env_file=".env", extra="ignore")

    secret_key: SecretStr = Field(..., description="JWT signing secret — min 32 chars")
    algorithm: str = "HS256"
    expire_minutes: int = Field(default=60, ge=1, le=10_080)


class EncryptionSettings(BaseSettings):
    """PHI field encryption settings (envelope encryption)."""

    model_config = SettingsConfigDict(env_prefix="ENCRYPTION_", env_file=".env", extra="ignore")

    key: SecretStr = Field(
        ...,
        description="Primary Fernet key (32-byte URL-safe base64). "
        'Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"',  # noqa: E501 — long literal (SQL/markdown), not splittable
    )
    key_id: str = Field(default="v1", description="Key identifier stored alongside ciphertext")
    previous_key: SecretStr | None = Field(
        default=None,
        description="Previous Fernet key for transparent key rotation (optional).",
    )


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    env: Literal["development", "staging", "production"] = "development"
    log_level: Annotated[str, Field(pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")] = "INFO"
    secret_key: SecretStr = Field(..., min_length=32, description="App secret key — min 32 chars")

    @field_validator("log_level", mode="before")
    @classmethod
    def upper_log_level(cls, v: str) -> str:
        """Normalise log level to uppercase."""
        return v.upper()


class Settings(BaseSettings):
    """Aggregate settings object. Compose sub-settings lazily."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def postgres(self) -> PostgresSettings:
        """Return validated Postgres settings."""
        return PostgresSettings()  # type: ignore[call-arg]  # pydantic-settings populates required fields from env

    @property
    def neo4j(self) -> Neo4jSettings:
        """Return validated Neo4j settings."""
        return Neo4jSettings()  # type: ignore[call-arg]  # pydantic-settings populates required fields from env

    @property
    def kafka(self) -> KafkaSettings:
        """Return validated Kafka settings."""
        return KafkaSettings()

    @property
    def minio(self) -> MinioSettings:
        """Return validated MinIO settings."""
        return MinioSettings()  # type: ignore[call-arg]  # pydantic-settings populates required fields from env

    @property
    def mlflow(self) -> MLflowSettings:
        """Return validated MLflow settings."""
        return MLflowSettings()

    @property
    def redis(self) -> RedisSettings:
        """Return validated Redis settings."""
        return RedisSettings()  # type: ignore[call-arg]  # pydantic-settings populates required fields from env

    @property
    def app(self) -> AppSettings:
        """Return validated app-level settings."""
        return AppSettings()  # type: ignore[call-arg]  # pydantic-settings populates required fields from env

    @property
    def jwt(self) -> JWTSettings:
        """Return validated JWT settings."""
        return JWTSettings()  # type: ignore[call-arg]  # pydantic-settings populates required fields from env

    @property
    def encryption(self) -> EncryptionSettings:
        """Return validated encryption settings."""
        return EncryptionSettings()  # type: ignore[call-arg]  # pydantic-settings populates required fields from env


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    The cache means `.env` is parsed once per process. In tests, call
    ``get_settings.cache_clear()`` before each test that sets env vars.

    Returns:
        Singleton ``Settings`` object.
    """
    return Settings()
