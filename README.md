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
- `GET /v1/conversations/{conversation_id}/types/{conversation_type}/artifacts/{artifact_id}`

Conversation type enum:

- `causal`
- `data`

## Typical API Flow

1. Create a conversation with `POST /v1/conversations` and a body such as `{ "conversation_type": "causal" }`.
2. Build the scoped base path: `/v1/conversations/{conversation_id}/types/{conversation_type}`.
3. Upload a CSV dataset with `POST {scope}/datasets` when the workflow is ready for data.
4. Send user input with `POST {scope}/messages` until the workflow reaches the next decision point.
5. Read the current snapshot with `GET {scope}` when the frontend needs the latest messages, states, and working dataset metadata.
6. Revert a workflow stage with `POST {scope}/state-reversions` when needed.
7. Download artifacts with `GET {scope}/artifacts/{artifact_id}`.

Dataset-history revert inside the workflow is triggered through the messages endpoint by sending:

```json
{
  "user_text": "revert_data_changes"
}
```

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

## Docker and Release

- Container listens on `PORT` (default `8080`)
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
