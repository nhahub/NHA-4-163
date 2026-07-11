# Healthcare Streamlit Web Interface — production image
#
# Multi-stage build:
#   builder — installs all Python deps
#   runtime — copies only venv + application code
#
# Build:
#   docker build -f infra/docker/streamlit.Dockerfile -t healthcare-streamlit:latest .

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    streamlit==1.32.2 \
    plotly==5.18.0 \
    pandas==2.2.1 \
    numpy==1.26.4 \
    xgboost==2.0.3 \
    scikit-learn==1.4.2 \
    psycopg2-binary==2.9.9 \
    neo4j==5.19.0 \
    mlflow==2.13.0 \
    pydantic==2.7.1 \
    pydantic-settings==2.2.1


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="healthcare-streamlit"
LABEL org.opencontainers.image.description="Healthcare Hereditary Disease Prediction Streamlit Interface"

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for least-privilege execution
RUN useradd --no-create-home --shell /bin/false appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Some mlflow imports expect pkg_resources from setuptools at runtime.
RUN pip install --no-cache-dir setuptools==68.2.2

# Working directory
WORKDIR /app

# Copy application code (from monorepo root context)
COPY services ./services
COPY libs /app/libs
COPY ml /app/ml

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_CLIENT_SHOWERRORDETAILS=false
ENV STREAMLIT_LOGGER_LEVEL=info

# Health check
HEALTHCHECK --interval=15s --timeout=10s --retries=5 --start-period=45s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit application
USER appuser
ENTRYPOINT ["streamlit", "run"]
CMD ["services/streamlit/app.py"]
