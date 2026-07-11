# Healthcare Hereditary Disease Prediction System
# Usage: make <target>
# All docker commands use the infra/compose path.

COMPOSE_DIR  := infra/compose
COMPOSE_FILE := $(COMPOSE_DIR)/docker-compose.yml
COMPOSE_OVER := $(COMPOSE_DIR)/docker-compose.override.yml
DC           := docker compose -f "$(COMPOSE_FILE)" -f "$(COMPOSE_OVER)"

.PHONY: help up down restart logs ps \
        run-all run-streamlit streamlit-logs streamlit-shell \
        orchestration-up orchestration-down \
        observability-up observability-down observability-logs \
        kafka-bootstrap spark-streaming feature-engineering \
        train-xgboost train-gnn train-models \
        api-build api-up api-down api-logs api-shell \
        test test-unit test-integration lint fmt typecheck \
        migrate migrate-down migrate-history migrate-sql \
        neo4j-schema seed check-env clean \
        service-account \
        terraform-init terraform-plan terraform-apply terraform-destroy terraform-fmt \
        k8s-apply k8s-delete k8s-status k8s-logs k8s-rollout k8s-rollback \
        docker-ecr-login docker-push

# ── Default ──────────────────────────────────────────────────────────────────
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Run Complete System ───────────────────────────────────────────────────────
run-all: check-env up sleep train-models streamlit-logs ## 🚀 Start all services + train models + launch Streamlit
	@echo "✅ All services running. Streamlit at http://localhost:8501"

sleep: ## Wait for services to be healthy
	@echo "⏳ Waiting for services..."
	@sleep 10

train-models: ## 🤖 Train XGBoost model
	@echo "🤖 Training XGBoost model..."
	$(DC) exec -T mlflow python -m ml.training.train_xgboost || true

streamlit-logs: ## 📊 Follow Streamlit logs
	$(DC) logs -f streamlit

streamlit-shell: ## 🐚 Open Streamlit container shell
	$(DC) exec streamlit /bin/bash

# ── Docker Compose ────────────────────────────────────────────────────────────
up: check-env ## Start all base services (detached)
	$(DC) up -d

down: ## Stop and remove containers (keep volumes)
	$(DC) down

restart: ## Restart all services
	$(DC) restart

logs: ## Follow logs for all services (ctrl-c to stop)
	$(DC) logs -f

ps: ## Show running containers and their status
	$(DC) ps

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run full test suite (unit + integration)
	python -m pytest tests/ -m "unit or integration"

test-unit: ## Run unit tests only (fast, no services needed)
	python -m pytest tests/unit/ -m unit -x

test-integration: ## Run integration tests (requires services up)
	python -m pytest tests/integration/ -m integration

# ── Code quality ──────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	python -m ruff check .

fmt: ## Format code with black + ruff --fix
	python -m black .
	python -m ruff check . --fix

typecheck: ## Run mypy static type checker
	python -m mypy libs/ services/ pipelines/ ml/

# ── Dev utilities ─────────────────────────────────────────────────────────────
check-env: ## Validate required environment variables
	@bash scripts/check-env.sh

migrate: ## Apply all pending Alembic migrations (postgres must be up)
	python -m alembic -c schemas/postgres/alembic.ini upgrade head

migrate-down: ## Roll back the last Alembic migration
	python -m alembic -c schemas/postgres/alembic.ini downgrade -1

migrate-history: ## Show Alembic migration history
	python -m alembic -c schemas/postgres/alembic.ini history --verbose

migrate-sql: ## Print the SQL that upgrade head would run (offline mode)
	python -m alembic -c schemas/postgres/alembic.ini upgrade head --sql

orchestration-up: check-env ## Build + start Airflow (scheduler + webserver) alongside base services
	$(DC) --profile orchestration up -d --build

orchestration-down: ## Stop Airflow containers (keeps volumes)
	$(DC) --profile orchestration down

observability-up: check-env ## Start Prometheus + Grafana alongside base services
	$(DC) --profile observability up -d

observability-down: ## Stop Prometheus + Grafana (keeps volumes)
	$(DC) --profile observability down

observability-logs: ## Follow Prometheus + Grafana logs
	$(DC) --profile observability logs -f prometheus grafana

kafka-bootstrap: ## Create Kafka topics + register Avro schemas
	python services/ingestion/kafka_admin.py

spark-streaming: ## Submit Spark Structured Streaming job to local cluster
	$(DC) exec spark-master spark-submit \
	  --master spark://spark-master:7077 \
	  --packages "$${SPARK_PACKAGES}" \
	  --py-files /opt/bitnami/spark/work/libs.zip \
	  /opt/bitnami/spark/work/pipelines/spark/streaming/job.py

feature-engineering: ## Run feature engineering batch job (as-of today; set AS_OF_DATE=YYYY-MM-DD to override)
	$(DC) exec spark-master spark-submit \
	  --master spark://spark-master:7077 \
	  --packages "$${SPARK_PACKAGES}" \
	  --py-files /opt/bitnami/spark/work/libs.zip \
	  /opt/bitnami/spark/work/pipelines/spark/feature_engineering/job.py \
	  --as-of-date "$${AS_OF_DATE:-$$(date +%F)}"

train-xgboost: ## Train XGBoost model (set FEATURE_DATE=YYYY-MM-DD; defaults to today)
	python ml/training/train_xgboost.py \
	  --feature-date "$${FEATURE_DATE:-$$(date +%F)}" \
	  --delta-base "$${DELTA_BASE:-s3a://healthcare-delta}"

train-gnn: ## Train GraphSAGE model (requires ENABLE_GNN_MODEL=true and torch_geometric)
	ENABLE_GNN_MODEL=true python ml/training/train_gnn.py \
	  --feature-date "$${FEATURE_DATE:-$$(date +%F)}" \
	  --delta-base "$${DELTA_BASE:-s3a://healthcare-delta}"

# ── Service accounts ─────────────────────────────────────────────────────────
service-account: ## Manage service accounts — usage: make service-account ARGS="create --username alice --role clinician"
	python scripts/create_service_account.py $${ARGS}

# ── API Service ───────────────────────────────────────────────────────────────
api-build: ## Build the API Docker image
	$(DC) build api

api-up: check-env ## Start the API service (and its dependencies)
	$(DC) up -d api

api-down: ## Stop the API service
	$(DC) stop api

api-logs: ## Follow API service logs
	$(DC) logs -f api

api-shell: ## Open a shell inside the running API container
	$(DC) exec api /bin/sh

# ── Infrastructure ────────────────────────────────────────────────────────────
neo4j-schema: ## Apply Neo4j constraints and indexes (neo4j must be up)
	@echo "Applying Neo4j constraints..."
	$(DC) exec neo4j cypher-shell \
	  -u $${NEO4J_USER:-neo4j} -p $${NEO4J_PASSWORD} \
	  -f /var/lib/neo4j/import/01_constraints.cypher
	@echo "Applying Neo4j indexes..."
	$(DC) exec neo4j cypher-shell \
	  -u $${NEO4J_USER:-neo4j} -p $${NEO4J_PASSWORD} \
	  -f /var/lib/neo4j/import/02_indexes.cypher
	@echo "Neo4j schema applied."

seed: ## Run migrations then load synthetic seed data
	$(MAKE) migrate
	$(MAKE) neo4j-schema
	@echo "Synthea seed data generation — implemented in Phase 2 (scripts/seed_synthea.py)"

# ── Terraform (Phase 9) ───────────────────────────────────────────────────────
TF_DIR := infra/terraform

terraform-init: ## Initialise Terraform working directory
	terraform -chdir=$(TF_DIR) init

terraform-plan: ## Show Terraform execution plan (set TF_WORKSPACE=staging|production)
	terraform -chdir=$(TF_DIR) plan -var-file=terraform.tfvars

terraform-apply: ## Apply Terraform changes (prompts for confirmation)
	terraform -chdir=$(TF_DIR) apply -var-file=terraform.tfvars

terraform-destroy: ## Destroy all Terraform-managed resources (DANGEROUS)
	terraform -chdir=$(TF_DIR) destroy -var-file=terraform.tfvars

terraform-fmt: ## Format all Terraform files in place
	terraform -chdir=$(TF_DIR) fmt -recursive

# ── Kubernetes (Phase 9) ──────────────────────────────────────────────────────
K8S_DIR   := infra/k8s
K8S_NS    := healthcare

k8s-apply: ## Apply all Kubernetes manifests to the current cluster context
	kubectl apply -f $(K8S_DIR)/namespace.yaml
	kubectl apply -f $(K8S_DIR)/api/
	kubectl apply -f $(K8S_DIR)/ingress/
	kubectl apply -f $(K8S_DIR)/monitoring/

k8s-delete: ## Delete all Kubernetes manifests (keeps namespace)
	kubectl delete -f $(K8S_DIR)/monitoring/ --ignore-not-found
	kubectl delete -f $(K8S_DIR)/ingress/ --ignore-not-found
	kubectl delete -f $(K8S_DIR)/api/ --ignore-not-found

k8s-status: ## Show pod, deployment, and HPA status in the healthcare namespace
	kubectl get pods,deployments,hpa,pdb,ingress -n $(K8S_NS)

k8s-logs: ## Tail logs from the healthcare-api pods (set SINCE=10m to limit)
	kubectl logs -f -l app=healthcare-api -n $(K8S_NS) --since=$${SINCE:-5m}

k8s-rollout: ## Watch the healthcare-api rollout status
	kubectl rollout status deployment/healthcare-api -n $(K8S_NS)

k8s-rollback: ## Roll back the healthcare-api deployment to the previous revision
	kubectl rollout undo deployment/healthcare-api -n $(K8S_NS)

# ── Docker / ECR (Phase 9) ────────────────────────────────────────────────────
# Set ECR_REGISTRY and IMAGE_TAG before calling these targets, or pass as make args.
ECR_REGISTRY ?= $(shell aws ecr describe-repositories --query 'repositories[0].repositoryUri' --output text 2>/dev/null)
IMAGE_TAG    ?= $(shell git rev-parse --short HEAD)

docker-ecr-login: ## Authenticate Docker with AWS ECR
	aws ecr get-login-password --region $${AWS_REGION:-us-east-1} \
	  | docker login --username AWS --password-stdin $(ECR_REGISTRY)

docker-push: ## Build and push the API image to ECR
	docker build -t $(ECR_REGISTRY):$(IMAGE_TAG) -f infra/docker/api.Dockerfile .
	docker push $(ECR_REGISTRY):$(IMAGE_TAG)
	@echo "Pushed: $(ECR_REGISTRY):$(IMAGE_TAG)"

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean: ## Remove Python cache files and coverage artifacts
	find . -type d -name "__pycache__" -not -path "./Healthcare/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./Healthcare/*" -delete 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
