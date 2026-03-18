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

- `POST /v1/conversations`
- `POST /v1/conversations/{conversation_id}/datasets` (CSV upload)
- `POST /v1/conversations/{conversation_id}/invoke`
- `GET /v1/conversations/{conversation_id}/lateststate`
- `POST /v1/conversations/{conversation_id}/revert`
- `GET /v1/conversations/{conversation_id}/artifacts/{artifact_id}`

## Typical API Flow

1. Create conversation.
2. Upload CSV dataset.
3. Call `invoke` repeatedly until stage completion or user input is requested.
4. Download artifacts by ID when returned in `artifact_ids`.

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
