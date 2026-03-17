# ---- Makefile ----
SHELL := /bin/bash

ENV_FILE ?= .env
DOCKER_CONFIG ?= docker.env
DOCKER_CONFIG_EXAMPLE ?= docker.env.example

-include $(DOCKER_CONFIG_EXAMPLE)
-include $(DOCKER_CONFIG)

.PHONY: help venv install dev-tools lint lint-fix format format-fix test test-quick clean run-cli run-api run-api-prod run-api-local docker-image docker-build docker-push

VENV ?= .venv
PYBIN := $(VENV)/bin
PYTHON := $(PYBIN)/python
PIP := $(PYBIN)/pip

# FastAPI entrypoint module:path
API_APP ?= python.adapters.api.app:app
API_HOST ?= 0.0.0.0
API_PORT ?= 8000

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

help:
	@echo "make install             - create venv and install runtime deps"
	@echo "make dev-tools           - install dev tools (ruff/black/pytest) if missing"
	@echo "make lint                - ruff check ."
	@echo "make lint-fix            - ruff check . --fix"
	@echo "make format              - black . --check"
	@echo "make format-fix          - black ."
	@echo "make test                - pytest with coverage"
	@echo "make test-quick          - pytest (no coverage)"
	@echo "make run-cli ARGS='...'  - run console copilot CLI (args forwarded)"
	@echo "make run-api             - run FastAPI (REST + WebSocket) with reload"
	@echo "make run-api-prod        - run FastAPI (REST + WebSocket) without reload"
	@echo "make docker-image        - print full image URI"
	@echo "make docker-build        - build Docker image for current env/tag"
	@echo "make docker-push         - push Docker image to registry"
	@echo "make clean               - remove venv + caches"

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PYTHON) -m pip install --upgrade pip setuptools wheel

install: venv
	@$(PIP) install -r requirements.txt

lint: install
	@$(PYBIN)/ruff check .

# lint-fix: install
# 	@$(PYBIN)/ruff check . --fix

# format: install
# 	@$(PYBIN)/black . --check

# format-fix: install
# 	@$(PYBIN)/black .

# test: install
# 	@$(PYBIN)/pytest -c pytest.ini --cov=python --cov-report=term-missing --maxfail=1 -q

test: install
	@$(PYBIN)/pytest -c pytest.ini -q

run-cli: venv
	@$(PYTHON) -m python.adapters.cli.main $(ARGS)

# REST + WebSocket server (FastAPI)
run-api: venv
	@$(PYBIN)/uvicorn $(API_APP) --host $(API_HOST) --port $(API_PORT) --reload

run-api-local: venv
	@test -f $(ENV_FILE) || { echo "Missing $(ENV_FILE)"; exit 1; }
	@set -a; \
	source $(ENV_FILE); \
	set +a; \
	$(PYBIN)/uvicorn $(API_APP) \
		--host "$${API_HOST:-0.0.0.0}" \
		--port "$${API_PORT:-8000}" \
		--reload


run-api-prod: venv
	@$(PYBIN)/uvicorn $(API_APP) --host $(API_HOST) --port $(API_PORT)

docker-image:
	@echo $(IMAGE_URI)

docker-build:
	@echo "Building image: $(IMAGE_URI)"
	@docker build --platform linux/amd64 -t $(IMAGE_URI) -f Dockerfile .

docker-push: docker-build
	@echo "Pushing image: $(IMAGE_URI)"
	@docker push $(IMAGE_URI)

clean:
	@rm -rf $(VENV) .pytest_cache .ruff_cache .coverage coverage.xml htmlcov __pycache__ **/__pycache__
