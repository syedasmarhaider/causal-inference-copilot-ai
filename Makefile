# ---- Makefile ----
SHELL := /bin/bash

ENV_FILE ?= .env
DOCKER_CONFIG ?= .docker.env
DOCKER_CONFIG_EXAMPLE ?= .docker.env.example

-include docker.env.example
-include $(DOCKER_CONFIG_EXAMPLE)
-include docker.env
-include $(DOCKER_CONFIG)

VENV ?= .venv
PYBIN := $(VENV)/bin
PYTHON := $(PYBIN)/python
PIP := $(PYBIN)/pip

# FastAPI entrypoint module:path
API_APP ?= python.adapters.api.app:app
API_HOST ?= 0.0.0.0
API_PORT ?= 8080

# Container image naming (industry-style split: host/project/repo/service:tag)
DEPLOY_ENV ?= dev
SERVICE_NAME ?= some-service
IMAGE_TAG ?= $(DEPLOY_ENV)
REGISTRY_HOST ?= europe-west3-docker.pkg.dev
REGISTRY_REPOSITORY ?= causal-dev-images
REGISTRY_PROJECT ?= your-gcp-project
IMAGE_REPOSITORY := $(REGISTRY_HOST)/$(REGISTRY_PROJECT)/$(REGISTRY_REPOSITORY)/$(SERVICE_NAME)
IMAGE_URI := $(IMAGE_REPOSITORY):$(IMAGE_TAG)

# Make Python see your src/ packages
export PYTHONPATH := src

.PHONY: help
help:
	@echo "make install             - create venv and install runtime deps"
	@echo "make install-test        - install runtime + test deps"
	@echo "make install-dev         - install runtime + test + dev deps"
	@echo "make dev-tools           - alias for install-dev"
	@echo "make lint                - ruff check ."
	@echo "make lint-fix            - ruff check . --fix"
	@echo "make format              - black . --check"
	@echo "make format-fix          - black ."
	@echo "make test                - run test suite"
	@echo "make test-quick          - run test suite (fast fail)"
	@echo "make test-deepeval       - run opt-in DeepEval prompt evals"
	@echo "make run-cli ARGS='...'  - run console copilot CLI (args forwarded)"
	@echo "make run-api             - run FastAPI (REST + WebSocket) with reload"
	@echo "make run-api-prod        - run FastAPI (REST + WebSocket) without reload"
	@echo "make docker-image        - print full image URI"
	@echo "make docker-build        - build Docker image for current env/tag"
	@echo "make docker-push         - push Docker image to registry"
	@echo "make clean               - remove venv + caches"

.PHONY: venv
venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PYTHON) -m pip install --upgrade pip setuptools wheel

.PHONY: install
install: venv
	@$(PIP) install -r requirements.txt

.PHONY: install-test
install-test: venv
	@$(PIP) install -r requirements-test.txt

.PHONY: install-dev
install-dev: venv
	@$(PIP) install -r requirements-dev.txt

.PHONY: test-tools
test-tools: install-test

.PHONY: dev-tools
dev-tools: install-dev

.PHONY: lint
lint: dev-tools
	@$(PYBIN)/ruff check .

.PHONY: lint-fix
lint-fix: dev-tools
	@$(PYBIN)/ruff check . --fix

.PHONY: format
format: dev-tools
	@$(PYBIN)/black . --check

.PHONY: format-fix
format-fix: dev-tools
	@$(PYBIN)/black .

.PHONY: test
test: test-tools
	@$(PYBIN)/pytest -c pytest.ini -q

.PHONY: test-quick
test-quick: test-tools
	@$(PYBIN)/pytest -c pytest.ini --maxfail=1 -q

.PHONY: test-deepeval
test-deepeval: test-tools
	@RUN_DEEPEVAL_TESTS=1 \
	DEEPEVAL_TELEMETRY_OPT_OUT=1 \
	DEEPEVAL_DISABLE_DOTENV=1 \
	DEEPEVAL_CACHE_FOLDER=/tmp/.deepeval \
	$(PYBIN)/pytest -c pytest.ini -q -m deepeval

.PHONY: run-cli
run-cli: venv
	@$(PYTHON) -m python.adapters.cli.main $(ARGS)

# REST + WebSocket server (FastAPI)
.PHONY: run-api
run-api: venv
	@$(PYBIN)/uvicorn $(API_APP) --host $(API_HOST) --port $(API_PORT) --reload

.PHONY: run-api-local
run-api-local: venv
	@test -f $(ENV_FILE) || { echo "Missing $(ENV_FILE)"; exit 1; }
	@set -a; \
	source $(ENV_FILE); \
	set +a; \
	$(PYBIN)/uvicorn $(API_APP) \
		--host "$${API_HOST:-0.0.0.0}" \
		--port "$${API_PORT:-8080}" \
		--reload

.PHONY: run-api-prod
run-api-prod: venv
	@$(PYBIN)/uvicorn $(API_APP) --host $(API_HOST) --port $(API_PORT)

.PHONY: docker-image
docker-image:
	@echo $(IMAGE_URI)

.PHONY: docker-build
docker-build:
	@echo "Building image: $(IMAGE_URI)"
	@docker build --platform linux/amd64 -t $(IMAGE_URI) -f Dockerfile .

.PHONY: docker-push
docker-push: docker-build
	@echo "Pushing image: $(IMAGE_URI)"
	@docker push $(IMAGE_URI)

.PHONY: clean
clean:
	@rm -rf $(VENV) .pytest_cache .ruff_cache .coverage coverage.xml htmlcov __pycache__ **/__pycache__
