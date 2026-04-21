# AitiaMed Causal Inference Copilot API

Live product: https://aitiamed.com

Backend service for AitiaMed's causal inference copilot. The app exposes an authenticated FastAPI API that manages conversations, dataset uploads, staged workflow execution, and artifact retrieval.

## Overview

- Framework: FastAPI (`python.adapters.api.app:app`)
- Runtime: Python 3.11
- Workflow engine: Node/state pipeline with LLM-assisted routing
- Storage: Firebase Realtime Database + Google Cloud Storage
- Auth: Firebase Bearer tokens for all `/v1/*` endpoints
- Architecture: Clean Architecture with Domain-Driven Design (DDD) boundaries

## Repository Layout

- `src/python/adapters/api/app.py`: FastAPI app and HTTP endpoints
- `src/python/domain/`: domain models, service interfaces, workflow contracts
- `src/python/implementation/workflows/`: workflow nodes, router, orchestration
- `src/python/implementation/repo/`: Firebase/GCS repositories
- `src/tests/`: API and workflow tests
- `src/infrastructure/terraform/gcp/`: Terraform for GCP infra
- `Makefile`: local run, test, and Docker commands
- `Dockerfile`: production container build

## Architecture Principles

- Clean Architecture: adapters and infrastructure depend on domain contracts, not the other way around.
- Domain-Driven Design: business logic and workflow behavior live in `domain` + workflow state/node abstractions, with implementation details isolated in `implementation/*`.

## Prerequisites

- Python 3.11+
- `make`
- A Firebase project + Realtime Database
- Google Cloud Storage buckets for data/models
- LLM provider key(s) based on your selected backend

## Quick Start

1. Install dependencies:

```bash
make install
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Fill required values in `.env`:

- `GOOGLE_CLOUD_PROJECT_ID`
- `FIREBASE_DATABASE_URL`
- `GCS_MODELS_BUCKET_NAME`
- `GCS_DATA_BUCKET_NAME`
- `GEMINI_API_KEY` (or other configured provider credentials)

4. Run the API locally:

```bash
make run-api-local
```

5. Open API docs:

- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## Makefile Commands (Run and Dev)

Use `make help` to print the full list. Core commands:

- `make install`: create `.venv` and install `requirements.txt`
- `make lint`: run Ruff checks
- `make test`: run pytest (`pytest.ini`)
- `make test-deepeval`: run opt-in DeepEval prompt evals against the configured LLM provider
- `make run-api`: start API with auto-reload on `0.0.0.0:8080`
- `make run-api-local`: start API with env values loaded from `.env`
- `make run-api-prod`: start API without reload
- `make docker-image`: print resolved image URI
- `make docker-build`: build Docker image
- `make docker-push`: build and push Docker image
- `make clean`: remove venv and cache artifacts

## API Surface

Public:

- `GET /healthz`

Authenticated (`Authorization: Bearer <firebase_id_token>`):

- `GET /v1/conversations`
- `POST /v1/conversations`
- `GET /v1/conversations/{conversation_id}/types/{conversation_type}`
- `POST /v1/conversations/{conversation_id}/types/{conversation_type}/messages`
- `POST /v1/conversations/{conversation_id}/types/{conversation_type}/state-reversions`
- `POST /v1/conversations/{conversation_id}/types/{conversation_type}/datasets` (CSV upload)
- `POST /v1/conversations/{conversation_id}/types/{conversation_type}/dataset-diffs`
- `GET /v1/conversations/{conversation_id}/types/{conversation_type}/artifacts/{artifact_id}`

Conversation type enum:

- `causal`
- `data`

Conversation metadata:

- `POST /v1/conversations` accepts optional `conversation_name`
- `GET /v1/conversations` returns `conversation_name` and `last_updated_at_utc`
- `last_updated_at_utc` is a UTC Unix timestamp in seconds

## Typical API Flow

1. Create a conversation with `POST /v1/conversations` and a body such as `{ "conversation_type": "causal", "conversation_name": "Hypertension cohort review" }`.
2. Build the scoped base path: `/v1/conversations/{conversation_id}/types/{conversation_type}`.
3. Upload a CSV dataset with `POST {scope}/datasets` when the workflow is ready for data.
4. Optionally inspect the last data change with `POST {scope}/dataset-diffs`.
5. Send user input with `POST {scope}/messages` until the workflow reaches the next decision point.
6. Read the current snapshot with `GET {scope}` when the frontend needs the latest messages, states, and working dataset metadata.
7. Revert a workflow stage with `POST {scope}/state-reversions` when needed.
8. Download artifacts with `GET {scope}/artifacts/{artifact_id}`.

Example conversation summary returned by `GET /v1/conversations`:

```json
{
  "conversation_id": "22222222-2222-2222-2222-222222222222",
  "conversation_type": "causal",
  "conversation_name": "Hypertension cohort review",
  "last_updated_at_utc": 1712345678.123
}
```

Dataset-history revert inside the workflow is triggered through the messages endpoint by sending:

```json
{
  "user_text": "revert_data_changes"
}
```

## Dataset Diff API

Use the dataset diff endpoint to compare the previous working dataset version with the current one for the same conversation:

- Endpoint: `POST /v1/conversations/{conversation_id}/types/{conversation_type}/dataset-diffs`
- Auth: required with `Authorization: Bearer <firebase_id_token>`
- Body: optional

Positional comparison with no request body:

```http
POST /v1/conversations/{conversation_id}/types/{conversation_type}/dataset-diffs
Authorization: Bearer <firebase_id_token>
```

Keyed comparison using business keys:

```json
{
  "key_columns": ["patient_id"]
}
```

### What It Returns

The response always identifies the two dataset versions that were compared and then returns a structured `diff` object:

- `previous_dataset_id`: the older working dataset version
- `current_dataset_id`: the latest working dataset version
- `diff.identity_mode`: `position` if no keys were supplied, otherwise `key`
- `diff.key_columns`: the effective key columns used for matching rows
- `diff.schema_diff`: column additions, removals, and dtype changes
- `diff.row_changes`: only changed rows; unchanged rows are omitted
- `diff.summary`: aggregate counts for changed rows and changed cells

Example response:

```json
{
  "conversation_id": "22222222-2222-2222-2222-222222222222",
  "conversation_type": "data",
  "previous_dataset_id": "33333333-3333-3333-3333-333333333333",
  "current_dataset_id": "44444444-4444-4444-4444-444444444444",
  "diff": {
    "identity_mode": "key",
    "key_columns": ["patient_id"],
    "schema_diff": {
      "columns_added": ["bmi"],
      "columns_removed": [],
      "column_type_changes": [
        {
          "column": "age",
          "old_dtype": "int64",
          "new_dtype": "float64"
        }
      ]
    },
    "row_changes": [
      {
        "row_ref": {
          "mode": "key",
          "key": {
            "patient_id": 101
          },
          "position": null
        },
        "op": "updated",
        "cell_changes": [
          {
            "column": "age",
            "op": "modified",
            "old_value": 44,
            "new_value": 45
          },
          {
            "column": "bmi",
            "op": "added",
            "old_value": null,
            "new_value": 27.1
          }
        ]
      }
    ],
    "summary": {
      "old_row_count": 100,
      "new_row_count": 101,
      "inserted_rows": 1,
      "deleted_rows": 0,
      "updated_rows": 1,
      "total_changed_rows": 2,
      "total_changed_cells": 3
    }
  }
}
```

Validation behavior:

- Returns `422` if the conversation does not yet have at least two working dataset versions.
- Returns `422` if `key_columns` is invalid, missing from one dataset version, contains duplicates, or resolves to null/duplicate keys in keyed mode.

## LLM Evals

- DeepEval prompt-eval tests are opt-in and require working LLM credentials plus network access.
- Run them with `make test-deepeval`.
- These tests are marked `deepeval` and `integration`, so they do not run in the default `make test` loop.

## Environment Variables

Defined in `.env.example`:

- `GEMINI_API_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_BASE_URL`
- `LANGFUSE_DEBUG`
- `GOOGLE_CLOUD_PROJECT_ID`
- `FIREBASE_DATABASE_URL`
- `GCS_MODELS_BUCKET_NAME`
- `GCS_DATA_BUCKET_NAME`
- `GCS_MODELS_TIMEOUT_SECONDS`
- `GCS_MODELS_UPLOAD_TIMEOUT_SECONDS`
- `GCS_MODELS_UPLOAD_RETRY_TIMEOUT_SECONDS`
- `GCS_MODELS_UPLOAD_CHUNK_SIZE_BYTES`
- `API_HOST`
- `API_PORT`
- `WEB_CONCURRENCY`

## Docker and Release

- Container listens on `PORT` (default `8080`)
- Production startup uses a dedicated Python bootstrap module instead of `sh -c ...`
- Worker count defaults to the CPUs visible to the container and can be overridden with `WEB_CONCURRENCY`
- Current Terraform default for Cloud Run CPU is `1`, so you must raise `cloud_run_cpu` above `1` to actually use multiple cores
- Build locally with `make docker-build`
- Push with `make docker-push`
- CI release pipeline file: `.github/workflows/release-image.yml`

## CI Behavior (GitHub Actions + GitHub CLI)

- GitHub Actions trigger on tag push: any tag matching `v*` (example: `v1.4.0`).
- Tag-triggered pipeline behavior: run tests (`make test`), then build and push Docker image.
- Image destination: `${REGISTRY_HOST}/${REGISTRY_PROJECT}/${REGISTRY_REPOSITORY}/${SERVICE_NAME}`.
- Tag-triggered image tag: the pushed Git tag name (`GITHUB_REF_NAME`), for example `v1.4.0`.
- Manual trigger is also supported via `workflow_dispatch`.

Trigger via git tag:

```bash
git tag v1.4.0
git push origin v1.4.0
```

Trigger manually with GitHub CLI:

```bash
gh workflow run release-image.yml -f deploy_env=dev -f image_tag=v1.4.0
```

## Infrastructure

Terraform for GCP is under:

- `src/infrastructure/terraform/gcp`

### Terraform Commands

From repo root:

```bash
cd src/infrastructure/terraform/gcp
```

1. Create environment files:

```bash
cp env/dev.backend.hcl.example env/dev.backend.hcl
cp env/dev.tfvars.example env/dev.tfvars
```

2. Initialize Terraform (GCS backend):

```bash
terraform init -backend-config=env/dev.backend.hcl
```

3. Validate configuration:

```bash
terraform validate
```

4. Preview infra changes:

```bash
terraform plan -var-file=env/dev.tfvars
```

5. Apply infra changes:

```bash
terraform apply -var-file=env/dev.tfvars
```

6. Inspect outputs:

```bash
terraform output
```
