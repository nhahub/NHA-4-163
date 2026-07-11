# Custom Airflow runtime for the Healthcare platform.
#
# Extends the official Airflow image with the third-party libraries the
# project's DAG tasks import at runtime (database clients, Spark, ML,
# data-quality).  Project source (libs/, pipelines/, ml/) is bind-mounted at
# /opt/airflow by docker-compose and put on PYTHONPATH there — it is NOT copied
# into the image, so DAG code stays live-editable in local dev.
#
# Build is triggered automatically by:
#   docker compose --profile orchestration up -d --build
#   make orchestration-up
FROM apache/airflow:2.9.1

# PySpark (feature-engineering + identity-resolution tasks) needs a JRE.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Install Python deps as the unprivileged airflow user (pip warns otherwise).
USER airflow
COPY infra/docker/airflow-requirements.txt /tmp/airflow-requirements.txt
RUN pip install --no-cache-dir -r /tmp/airflow-requirements.txt

# GNN training stack (model_training DAG, GNN branch: ml.models.gnn_model uses
# torch_geometric.nn.SAGEConv).  CPU-only wheels from the PyTorch CPU index keep
# the image lean — no CUDA payload.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 \
    && pip install --no-cache-dir torch-geometric==2.5.3
