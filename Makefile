# ---- Makefile ----
SHELL := /bin/bash
.PHONY: help venv install clean lint format fmt-check typecheck test test-quick cov htmlcov ci

# Use the local venv if present; fall back to system python
VENV ?= .venv
PYBIN := $(VENV)/bin
PYTHON := $(PYBIN)/python

# Make pytest see your code under src/
export PYTHONPATH := src

help:
	@echo "make install     - create venv and install deps (dev + runtime)"
	@echo "make test        - run pytest with coverage gate"
	@echo "make test-quick  - run pytest fast (no coverage)"
	@echo "make cov         - coverage report (term + xml)"
	@echo "make htmlcov     - open HTML coverage report"
	@echo "make lint        - ruff lint (no fixes)"
	@echo "make format      - ruff --fix + black"
	@echo "make fmt-check   - verify formatting (CI)"
	@echo "make typecheck   - mypy (strict-ish)"
	@echo "make ci          - lint + typecheck + test (for CI)"

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PYTHON) -m pip install --upgrade pip

# If you keep only one requirements file, this target installs both runtime and dev tools.
install: venv
	@$(PYBIN)/pip install -r requirements.txt
	@$(PYBIN)/pip install pytest pytest-cov mypy ruff black

lint:
	@$(PYBIN)/ruff check .

format:
	@$(PYBIN)/ruff check . --fix
	@$(PYBIN)/black .

fmt-check:
	@$(PYBIN)/ruff check .
	@$(PYBIN)/black . --check

typecheck:
	@$(PYBIN)/mypy src

test:
	@$(PYBIN)/pytest -c pytest.ini --cov=python --cov-report=term-missing --cov-report=xml --maxfail=1 -q

test-quick:
	@$(PYBIN)/pytest -c pytest.ini -q

cov:
	@$(PYBIN)/coverage report -m || true
	@echo "Coverage XML at coverage.xml (from pytest-cov)"

htmlcov:
	@$(PYBIN)/coverage html && python3 -c 'import webbrowser; webbrowser.open("htmlcov/index.html")' || true

ci: fmt-check typecheck test

clean:
	@rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
