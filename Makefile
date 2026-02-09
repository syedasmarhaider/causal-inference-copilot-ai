# ---- Makefile ----
SHELL := /bin/bash

.PHONY: help venv install dev-tools lint lint-fix format format-fix test test-quick clean run-cli

VENV ?= .venv
PYBIN := $(VENV)/bin
PYTHON := $(PYBIN)/python
PIP := $(PYBIN)/pip

# Make Python see your src/ packages
export PYTHONPATH := src

help:
	@echo "make install            - create venv and install runtime + dev deps"
	@echo "make dev-tools          - install dev tools (ruff/black/pytest) if missing"
	@echo "make lint               - ruff check ."
	@echo "make lint-fix           - ruff check . --fix"
	@echo "make format             - black . --check"
	@echo "make format-fix         - black ."
	@echo "make test               - pytest with coverage"
	@echo "make test-quick         - pytest (no coverage)"
	@echo "make run-cli ARGS='...' - run console copilot CLI (args forwarded)"
	@echo "make clean              - remove venv + caches"

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PYTHON) -m pip install --upgrade pip setuptools wheel

install: venv
	@$(PIP) install -r requirements.txt

# Install dev tools only if missing (fast no-op when already present)
dev-tools: venv
	@command -v $(PYBIN)/ruff >/dev/null 2>&1 || $(PIP) install -r dev-requirements.txt
	@command -v $(PYBIN)/black >/dev/null 2>&1 || $(PIP) install -r dev-requirements.txt
	@command -v $(PYBIN)/pytest >/dev/null 2>&1 || $(PIP) install -r dev-requirements.txt

lint: dev-tools
	@$(PYBIN)/ruff check .

lint-fix: dev-tools
	@$(PYBIN)/ruff check . --fix

format: dev-tools
	@$(PYBIN)/black . --check

format-fix: dev-tools
	@$(PYBIN)/black .

test: dev-tools
	@$(PYBIN)/pytest -c pytest.ini --cov=python --cov-report=term-missing --maxfail=1 -q

test-quick: dev-tools
	@$(PYBIN)/pytest -c pytest.ini -q

# Usage:
#   make run-cli
#   make run-cli ARGS="--repo-dir .local/copilot --dataset ./data.csv"
run-cli: venv
	@$(PYTHON) -m python.adapters.cli.main $(ARGS)

clean:
	@rm -rf $(VENV) .pytest_cache .ruff_cache .coverage coverage.xml htmlcov __pycache__ **/__pycache__
