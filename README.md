# Thesis-LMU Copilot (CLI + FastAPI)

A Python-based copilot system featuring:

- **CLI** entrypoint for running the copilot from the terminal.
- **FastAPI** server exposing **REST + WebSocket** interfaces (supporting dev + prod modes).
- A `src/`-based package layout (wired via `PYTHONPATH=src` in the Makefile).

---

## 📂 Repository Layout

- `src/python/...` — Application code (imported via `PYTHONPATH=src`).
- `python.adapters.cli.main` — CLI module entrypoint.
- `python.adapters.api.app:app` — FastAPI app object (REST + WebSocket).
- `requirements.txt` — Runtime dependencies.
- `dev-requirements.txt` — Developer tooling dependencies (ruff/black/pytest).
- `pytest.ini` — Pytest configuration.

---

## ⚙️ Prerequisites

- Python 3.x (with `python3 -m venv` available)
- GNU Make
- Bash (`/bin/bash`)

---

## 🚀 Installation & Setup

1.  **Install Runtime Dependencies**
    Create the virtual environment and install requirements:

    ```bash
    make install
    ```

2.  **Install Developer Tools**
    Install formatting and testing tools (Ruff, Black, Pytest):
    ```bash
    make dev-tools
    ```

---

## 🛠 Development Workflow

The Makefile provides shortcuts for maintaining code quality.

### Linting & Formatting

| Action           | Command           | Description                  |
| :--------------- | :---------------- | :--------------------------- |
| **Lint**         | `make lint`       | Check code with Ruff.        |
| **Lint (Fix)**   | `make lint-fix`   | Auto-fix Ruff errors.        |
| **Format**       | `make format`     | Check formatting.            |
| **Format (Fix)** | `make format-fix` | Auto-format code with Black. |

### Testing

| Action         | Command           | Description                          |
| :------------- | :---------------- | :----------------------------------- |
| **Full Test**  | `make test`       | Run tests with coverage reports.     |
| **Quick Test** | `make test-quick` | Run tests without coverage (faster). |

---

## 🖥 Usage: CLI Adapter

The CLI is a thin wrapper around the copilot logic, ideal for local experiments, batch runs, and debugging.

**Entrypoint:** `python.adapters.cli.main`

To run the CLI, use `make run-cli` and pass arguments via the `ARGS` variable.

**Examples:**

```bash
# View Help
make run-cli ARGS="--help"

# Run with parameters
make run-cli ARGS="--dataset ./data/sample.csv --treatment T --outcome Y"
```
