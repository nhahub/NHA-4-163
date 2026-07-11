FROM ghcr.io/mlflow/mlflow:v2.13.0

# psycopg2 → Postgres backend store; boto3 → S3/MinIO artifact store.
# boto3 is REQUIRED whenever --artifacts-destination is an s3:// URI, otherwise
# the server 500s on every artifact/model upload ("No module named 'boto3'").
RUN pip install --no-cache-dir psycopg2-binary==2.9.9 boto3==1.34.69