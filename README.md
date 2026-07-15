# Agent API

Local backend for the agent API. The app exposes an authenticated FastAPI API for conversations, dataset uploads, staged workflow execution, and artifact retrieval.

## Overview

- Framework: FastAPI (`python.adapters.api.app:app`)
- Runtime: Python 3.11
- Workflow engine: Node/state pipeline with LLM-assisted routing
- Storage: local files under `.local_storage`
- Auth: local bearer token with JWT-like, UUID, or opaque local identity support
- LLM provider: configurable through LiteLLM; Vertex AI is the default

## Repository Layout

- `src/python/adapters/api/app.py`: FastAPI app and HTTP endpoints
- `src/python/domain/`: domain models, service interfaces, workflow contracts
- `src/python/implementation/workflows/`: workflow nodes, router, orchestration
- `src/python/implementation/repo/`: local repository implementations
- `src/tests/`: API and workflow tests
- `Makefile`: local install, test, lint, and run commands

## Prerequisites

- Python 3.11+
- `make`
- Vertex AI credentials available through Application Default Credentials

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

- `VERTEXAI_PROJECT`
- `VERTEXAI_LOCATION`

4. Run the API locally:

```bash
make run-api-local
```

5. Open API docs:

- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## Makefile Commands

- `make install`: create `.venv` and install `requirements.txt`
- `make install-test`: install runtime and test dependencies
- `make install-dev`: install runtime, test, and dev dependencies
- `make lint`: run Ruff checks
- `make format`: run Black in check mode
- `make test`: run pytest (`pytest.ini`)
- `make test-quick`: run pytest with fast failure
- `make test-deepeval`: run opt-in DeepEval prompt evals against Vertex AI
- `make run-api`: start API with auto-reload on `0.0.0.0:8080`
- `make run-api-local`: start API with env values loaded from `.env`
- `make clean`: remove venv and cache artifacts

## Local Runtime

The app always uses local filesystem backends:

- Workflow state: `.local_storage/workflow_state.json`
- Dataset artifacts: `.local_storage/data`
- Model artifacts: `.local_storage/models`

Call authenticated endpoints with either:

- A JWT-like bearer token whose payload has an `id`, `ID`, `uuid`, `user_id`, `uid`, or `sub` UUID claim.
- A raw UUID bearer token for local clients.
- Any other non-empty raw bearer token, which is mapped to a stable local UUID.

```http
Authorization: Bearer <local-jwt-like-token>
Authorization: Bearer <uuid>
Authorization: Bearer <opaque-local-token>
```

JWT signatures are not validated in this local paper branch.

Delete `.local_storage/workflow_state.json` to clear local conversations and workflow state.

## LLM Configuration

The LLM service uses LiteLLM. Vertex AI remains the default provider and uses the existing
Application Default Credentials setup:

- `VERTEXAI_PROJECT`
- `VERTEXAI_LOCATION` (defaults to `global` in `.env.example`)
- `GOOGLE_APPLICATION_CREDENTIALS` if you are not using `gcloud auth application-default login`

Provider and model aliases can be configured through env vars:

- `LLM_PROVIDER`: `vertex_ai`, `google_ai_studio`, `openai`, or `azure`
- `LLM_API_KEY`: required for `google_ai_studio`, `openai`, and `azure`
- `LLM_API_BASE`: required for `azure`
- `LLM_API_VERSION`: required for `azure`
- `LLM_MODEL_MINI`
- `LLM_MODEL_BASIC`
- `LLM_MODEL_PRO`
- `LLM_MODEL_THINKING`

For Vertex AI, the model env vars are optional and default to the current Gemini model map.
For non-Vertex providers, set all four model env vars to the provider model or deployment names.

## API Surface

Public:

- `GET /healthz`

Authenticated:

- `GET /v1/conversations`
- `POST /v1/conversations`
- `GET /v1/conversations/{conversation_id}/types/{conversation_type}`
- `POST /v1/conversations/{conversation_id}/types/{conversation_type}/messages`
- `POST /v1/conversations/{conversation_id}/types/{conversation_type}/state-reversions`
- `POST /v1/conversations/{conversation_id}/types/{conversation_type}/datasets`
- `GET /v1/conversations/{conversation_id}/types/{conversation_type}/datasets/{dataset_id}`
- `POST /v1/conversations/{conversation_id}/types/{conversation_type}/dataset-diffs`
- `GET /v1/conversations/{conversation_id}/types/{conversation_type}/artifacts/{artifact_id}`

Conversation type enum:

- `causal`
- `data`

## Typical API Flow

1. Create a conversation with `POST /v1/conversations`.
2. Build the scoped base path: `/v1/conversations/{conversation_id}/types/{conversation_type}`.
3. Upload a CSV dataset with `POST {scope}/datasets` when the workflow is ready for data.
4. Optionally inspect the last data change with `POST {scope}/dataset-diffs`.
5. Send user input with `POST {scope}/messages` until the workflow reaches the next decision point.
6. Read the current snapshot with `GET {scope}`.
7. Revert a workflow stage with `POST {scope}/state-reversions` when needed.
8. Download artifacts with `GET {scope}/artifacts/{artifact_id}`.

Dataset-history revert inside the workflow is triggered through the messages endpoint by sending:

```json
{
  "user_text": "revert_data_changes"
}
```

## Environment Variables

Defined in `.env.example`:

- `LOG_SERVICE_NAME`
- `LLM_PROVIDER`
- `LLM_API_KEY`
- `LLM_API_BASE`
- `LLM_API_VERSION`
- `LLM_MODEL_MINI`
- `LLM_MODEL_BASIC`
- `LLM_MODEL_PRO`
- `LLM_MODEL_THINKING`
- `VERTEXAI_PROJECT`
- `VERTEXAI_LOCATION`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `LITELLM_LOCAL_MODEL_COST_MAP`
- `LOG_LEVEL`
- `API_HOST`
- `SHAP_ENABLED`: enable or disable the SHAP companion node (`true` by default)
- `API_PORT`

## Tests

Default tests do not call live LLMs:

```bash
make test
```

Live prompt/eval tests are opt-in and require Vertex AI credentials:

```bash
RUN_LIVE_LLM_TESTS=true make test-deepeval
```
