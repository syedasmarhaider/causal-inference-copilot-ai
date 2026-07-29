# PrecisionMedicineAgent Backend

This repository contains the backend for **PrecisionMedicineAgent**. It is a
research system that guides a user through data preparation, causal study
design, model selection, model training, and causal-effect analysis.

The backend is a Python application. It exposes a REST API, stores each
conversation on the local filesystem, calls a configured large language model
(LLM), and runs the statistical parts of the workflow with pandas,
scikit-learn, EconML, and related libraries.

> **Research-use notice**
>
> This is research software. It is not a medical device, it does not replace
> clinical or statistical review, and its output should not be used as the
> only basis for a treatment decision. Causal estimates are only meaningful
> when the study design, data quality, assumptions, and model are appropriate.

## Contents

- [What the backend does](#what-the-backend-does)
- [A few useful words](#a-few-useful-words)
- [System requirements](#system-requirements)
- [Quick start](#quick-start)
- [Configure the LLM provider](#configure-the-llm-provider)
- [Other configuration](#other-configuration)
- [How authentication works in this local version](#how-authentication-works-in-this-local-version)
- [Run a complete API session](#run-a-complete-api-session)
- [Understand the workflow](#understand-the-workflow)
- [Inspect and compare datasets](#inspect-and-compare-datasets)
- [Revert data or workflow state](#revert-data-or-workflow-state)
- [Download artifacts](#download-artifacts)
- [Download the audit log](#download-the-audit-log)
- [API reference](#api-reference)
- [Local files and saved state](#local-files-and-saved-state)
- [Tests and code checks](#tests-and-code-checks)
- [Troubleshooting](#troubleshooting)
- [Important limitations and caveats](#important-limitations-and-caveats)
- [Reproducibility checklist for a paper](#reproducibility-checklist-for-a-paper)
- [Project structure](#project-structure)

## What the backend does

The backend supports two kinds of conversation:

- `causal`: the full PrecisionMedicineAgent workflow. It prepares the dataset,
  discusses the causal protocol, validates the design, selects and trains a
  model, and answers causal-inference questions.
- `data`: a smaller workflow for uploading, inspecting, and changing a
  dataset without continuing to causal model training.

A typical `causal` conversation follows these stages:

1. `DATA_MANUPULATION`: upload, inspect, clean, filter, reshape, or otherwise
   prepare the dataset.
2. `PROTOCOL_DISCUSSION`: describe the treatment, outcome, population,
   covariates, effect modifiers, and other causal assumptions.
3. `DATA_COMPILATION`: compile and validate the causal specification and
   transformation plan.
4. `MODEL_SELECTION`: choose an available causal estimator.
5. `MODEL_TRAIN`: fit the selected model and save its outputs.
6. `CAUSAL_INFERENCE`: estimate and discuss average or conditional treatment
   effects.

The agent can also route some questions to companion stages such as
`DATA_STATISTICS`, `GENERAL_QUERIES`, `CAUSAL_VALIDATE`, and
`SHAP_EXPLANATION`.

The name `DATA_MANUPULATION` is misspelled in the existing code. It must still
be written exactly this way when it is used as a workflow state name.

## A few useful words

If you are new to Python backends, these terms will appear often:

- **Backend**: the part of the system that stores data, runs the workflow, and
  provides an API. A frontend or another client sends requests to it.
- **API**: a set of HTTP addresses that another program can call. For example,
  `GET /healthz` checks whether this backend is running.
- **REST**: the HTTP style used by this API. Requests use methods such as
  `GET` and `POST`.
- **JSON**: the text format used for most API requests and responses.
- **CSV**: the table format accepted for dataset uploads.
- **UUID**: a long identifier such as
  `22222222-2222-2222-2222-222222222222`. Conversations, datasets, and
  artifacts use UUIDs.
- **Environment variable**: a setting given to the program when it starts.
  This project normally reads these settings from `.env`.
- **Virtual environment (`venv`)**: a private Python installation for this
  project. It keeps this project's libraries separate from libraries used by
  other projects. The virtual environment in this repository is named
  `.venv`.
- **`pip`**: Python's package installer.
- **`make`**: a small command runner. The `Makefile` gives short commands such
  as `make install`, `make test`, and `make run-api-local`.
- **`curl`**: a terminal program used to send HTTP requests.
- **Artifact**: a file-like result produced by the workflow, such as a CSV
  table or a Vega-Lite graph specification in JSON.
- **Unit test**: a small automatic check of one part of the code.
- **`pytest`**: the program used to run this repository's tests.

There is no component named `Utest` in this repository. If “Utest” means
“unit test,” run the unit and API tests with `make test`.

## System requirements

Use the following for the most predictable setup:

- Python 3.11
- `make`
- Git
- Internet access during installation
- Internet access to the selected LLM provider while the backend is running
- Credentials for Vertex AI, Google AI Studio, OpenAI, or Azure OpenAI

The project is configured and formatted for Python 3.11. A newer Python
version may work, but Python 3.11 is the intended reproducible runtime.

Check the installed version:

```bash
python3 --version
```

On macOS or Linux, the commands below can be run directly in a terminal. On
Windows, the simplest option is usually WSL. Native Windows users can still
use Python, but the provided `Makefile` and shell commands are written for a
Unix-like shell.

Installation includes scientific Python libraries and EconML. It can take
several minutes and requires more disk space than a small web application.

## Quick start

### 1. Open the repository

Run commands from the repository root, the directory that contains
`Makefile`, `requirements.txt`, and this `README.md`.

```bash
cd causal-inference-copilot-ai
```

If your directory has a different name, use that name instead.

### 2. Create the virtual environment and install the backend

```bash
make install
```

This command:

1. creates `.venv` if it does not exist;
2. updates `pip`, `setuptools`, and `wheel`; and
3. installs the pinned packages in `requirements.txt`.

You do not need to activate `.venv` when you use the `make` commands. The
`Makefile` calls the Python executable inside `.venv` directly.

If `make` is not available, the equivalent manual commands on macOS or Linux
are:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

Activating the environment is optional:

```bash
source .venv/bin/activate
```

After activation, the shell prompt normally starts with `(.venv)`. Run
`deactivate` when you want to leave it.

### 3. Create the local configuration file

```bash
cp .env.example .env
```

`.env.example` is a safe template. `.env` is the local copy where credentials
and machine-specific settings belong. `.env` is ignored by Git and should
never be committed.

Open `.env` in a text editor and configure one LLM provider. Vertex AI is the
default. Provider-specific examples are in
[Configure the LLM provider](#configure-the-llm-provider).

### 4. Start the backend

```bash
make run-api-local
```

This is the recommended local command because it loads `.env` before starting
the API. The default address is:

```text
http://127.0.0.1:8080
```

The server uses auto-reload for Python source changes. To stop it, return to
the terminal where it is running and press `Ctrl+C`.

Restart the server after changing `.env`. Auto-reload notices Python file
changes, but an environment change should be applied with a full stop and
restart.

### 5. Check that it is running

Open a second terminal and run:

```bash
curl -sS http://127.0.0.1:8080/healthz
```

Expected response:

```json
{"ok":true}
```

`/healthz` only proves that the web application is responding. It does not
make an LLM call and does not prove that provider credentials are valid.

### 6. Open the interactive API documentation

- Swagger UI: <http://127.0.0.1:8080/docs>
- ReDoc: <http://127.0.0.1:8080/redoc>
- OpenAPI JSON: <http://127.0.0.1:8080/openapi.json>

In Swagger UI, select **Authorize** and enter the token itself. Do not include
the word `Bearer` in the Swagger token box; Swagger adds it to the request.

## Configure the LLM provider

The application uses LiteLLM to call one of four supported providers:

- Vertex AI
- Google AI Studio
- OpenAI
- Azure OpenAI

Only one provider block should be active in `.env`.

### Important `.env` rules

- Use plain `NAME=value` lines.
- Do not put spaces around `=`.
- Keep only the selected provider's active settings.
- Never commit `.env` or paste its secrets into an issue or paper appendix.
- Stop and restart the backend after a change.
- `make run-api-local` loads `.env`.
- `make run-api` does **not** load `.env`; it only uses variables already
  present in the shell.
- To load a differently named file, run
  `make run-api-local ENV_FILE=path/to/file.env`.

The four model settings are aliases used by different parts of the workflow:

```text
LLM_MODEL_MINI
LLM_MODEL_BASIC
LLM_MODEL_PRO
LLM_MODEL_THINKING
```

For Vertex AI they have defaults in the code, although the example file lists
them explicitly. For every other provider, all four values are required.

### Option A: Vertex AI

Vertex AI is the default provider. A minimal block looks like this:

```dotenv
LLM_PROVIDER=vertex_ai
VERTEXAI_PROJECT=your-google-cloud-project-id
VERTEXAI_LOCATION=global

LLM_MODEL_MINI=gemini-3.5-flash
LLM_MODEL_BASIC=gemini-3.5-flash
LLM_MODEL_PRO=gemini-3.5-flash
LLM_MODEL_THINKING=gemini-3.1-pro-preview
```

For a personal local login, install the Google Cloud CLI and run:

```bash
gcloud auth application-default login
```

For a service account, set an absolute credentials path:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

The Google Cloud project must have access to Vertex AI, the chosen models, and
the selected location. Preview model names can change or may not be available
to every project. If a model is unavailable, change the corresponding
`LLM_MODEL_*` value to a model that is enabled for the project.

### Option B: Google AI Studio

```dotenv
LLM_PROVIDER=google_ai_studio
LLM_API_KEY=your-google-ai-studio-api-key

LLM_MODEL_MINI=gemini-2.5-flash-lite
LLM_MODEL_BASIC=gemini-2.5-flash
LLM_MODEL_PRO=gemini-2.5-pro
LLM_MODEL_THINKING=gemini-2.5-pro
```

Google AI Studio uses an API key. Vertex project and location settings are not
used for this provider.

### Option C: OpenAI

```dotenv
LLM_PROVIDER=openai
LLM_API_KEY=your-openai-api-key

LLM_MODEL_MINI=gpt-4.1-mini
LLM_MODEL_BASIC=gpt-4.1
LLM_MODEL_PRO=gpt-4.1
LLM_MODEL_THINKING=o4-mini
```

The model names must be available to the API account. An optional custom
endpoint can be supplied with `LLM_API_BASE`.

### Option D: Azure OpenAI

```dotenv
LLM_PROVIDER=azure
LLM_API_KEY=your-azure-openai-api-key
LLM_API_BASE=https://your-resource-name.openai.azure.com
LLM_API_VERSION=2024-10-21

LLM_MODEL_MINI=your-mini-deployment
LLM_MODEL_BASIC=your-basic-deployment
LLM_MODEL_PRO=your-pro-deployment
LLM_MODEL_THINKING=your-thinking-deployment
```

For Azure, the four model values are **deployment names**, not ordinary base
model names.

### Why the API can start before credentials fail

The public health endpoint does not use the LLM. Creating or listing a
conversation may also succeed before the first LLM request. A credential,
project, model, quota, or network problem may therefore appear only when a
message makes the workflow call the provider.

## Other configuration

The main runtime values in `.env.example` are:

| Setting | Meaning | Example/default |
| --- | --- | --- |
| `LOG_SERVICE_NAME` | Service name written in JSON logs | `agent` |
| `LOG_LEVEL` | Log detail: `DEBUG`, `INFO`, `WARNING`, or `ERROR` | `INFO` |
| `API_HOST` | Network address on which Uvicorn listens | `0.0.0.0` |
| `API_PORT` | Local port | `8080` |
| `LITELLM_LOCAL_MODEL_COST_MAP` | Prevent a model-cost-map download during import | `True` |
| `SHAP_ENABLED` | Allow the expensive post-training SHAP companion stage | `false` in `.env.example` |
| `CAUSAL_VALIDATE_ENABLED` | Allow the expensive post-training validation companion stage | `false` |

Boolean values such as `true`, `1`, `yes`, and `on` enable a flag. Values such
as `false`, `0`, `no`, `off`, and an empty value disable it.

There is one subtle SHAP default: `.env.example` explicitly sets
`SHAP_ENABLED=false`, but if both `SHAP_ENABLED` and the older `ENABLE_SHAP`
setting are completely absent, the code defaults SHAP to enabled. Keep the
explicit setting in `.env` when reproducibility matters.

### Change the port

Edit `.env`:

```dotenv
API_PORT=8081
```

Restart the backend and then use `http://127.0.0.1:8081`.

### Write logs to a file

Logs are written as one JSON object per line in the server terminal. Optional
file logging is also supported:

```dotenv
LOG_FLUSH_FILE_ENABLED=true
LOG_FLUSH_FILE_PATH=/absolute/path/to/precision-medicine-agent.log
```

If file logging is enabled without a path, the default is
`/tmp/<service-name>.log`.

### Control validation CPU use

When `CAUSAL_VALIDATE_ENABLED=true`, outer cross-validation can use several
processes. To restrict the number of parallel folds, add:

```dotenv
PRECISION_MEDICINE_OUTER_CV_CATE_N_JOBS=1
```

The value must be between `1` and the configured number of outer folds.
Reducing it uses less memory but takes longer.

### Change statistical training defaults

Statistical settings are intentionally code configuration, not ordinary
`.env` values. They live in:

```text
src/python/implementation/workflows/tools/causal/inference/econml/model_training_config.py
```

The current defaults include:

- run seed: `1729`;
- outer CATE validation folds: `10`;
- causal forest estimators: `2000`;
- causal forest subforest size: `4`;
- causal forest maximum sample fraction: `0.45`;
- causal forest minimum leaf size: `20`; and
- causal forest minimum balancedness tolerance: `0.45`.

Changing these values changes the statistical method and may change paper
results. After any change, run the tests, record the Git commit, and begin a
new conversation. Previously trained model files are not automatically
retrained.

The seed and fold count are not currently read from environment variables.
Edit `MODEL_TRAINING_CONFIG` in the code if the research protocol requires a
different value.

## How authentication works in this local version

All `/v1/...` endpoints require this header:

```http
Authorization: Bearer <token>
```

The public `/healthz` endpoint does not require it.

For a simple local run, use one UUID as the token:

```bash
python3 -c 'import uuid; print(uuid.uuid4())'
```

Save the output and reuse the exact same token for the whole experiment.

This local backend accepts three token forms:

1. A raw UUID.
2. An opaque non-empty string. It is converted into a stable local UUID.
3. A three-part JWT-like string whose payload contains a UUID in one of these
   claims: `id`, `ID`, `uuid`, `user_id`, `uid`, or `sub`.

Important behavior:

- JWT signatures are **not validated**.
- Token expiry is **not validated**.
- An opaque token should not look like a three-part JWT. A string with exactly
  two dots is treated as JWT-like and must contain a valid encoded payload.
- The token determines the local user identity and storage path.
- Changing the token makes existing conversations appear to be missing.
- This authentication is convenient for local research, but it is not secure
  enough for an exposed or production service.

The examples below use shell variables:

```bash
export BASE_URL="http://127.0.0.1:8080"
export TOKEN="00000000-0000-0000-0000-000000000000"
```

For real local experiments, replace the example token with a generated UUID
and keep it private.

## Run a complete API session

This section shows the full transport flow with `curl`. Keep the backend
running in another terminal.

### 1. Create a causal conversation

```bash
curl -sS -X POST "$BASE_URL/v1/conversations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_type": "causal",
    "conversation_name": "Hypertension cohort study"
  }'
```

Example response:

```json
{
  "conversation_id": "22222222-2222-2222-2222-222222222222",
  "conversation_type": "causal",
  "conversation_name": "Hypertension cohort study",
  "last_updated_at_utc": 1712345678.123
}
```

Copy `conversation_id` into a shell variable:

```bash
export CONVERSATION_ID="22222222-2222-2222-2222-222222222222"
export CONVERSATION_TYPE="causal"
```

Use the actual ID returned by your backend.

To create the shorter data-only workflow, use
`"conversation_type": "data"` and set `CONVERSATION_TYPE="data"`.

### 2. List this user's conversations

```bash
curl -sS "$BASE_URL/v1/conversations" \
  -H "Authorization: Bearer $TOKEN"
```

Conversations are returned with the most recently updated first.

### 3. Upload a CSV dataset

```bash
curl -sS -X POST \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE/datasets" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/absolute/path/to/dataset.csv;type=text/csv"
```

Use an absolute path if there is any doubt about the current directory.

The upload:

- must not be empty;
- must be parseable by pandas as CSV;
- should have a header row;
- is accepted only while the workflow is at the data-manipulation stage; and
- is read into memory, so very large files can use substantial RAM.

The initial upload currently uses the fixed local dataset ID
`00000000-0000-0000-0000-000000000000` inside each conversation. This is
safe because storage is also separated by user and conversation.

For a causal run, prepare the table so that:

- one row represents one analysis unit at the intended time zero;
- column names are non-empty and unique;
- the treatment and outcome are separate columns;
- the current treatment is binary;
- binary treatment and outcome codes are used consistently;
- a continuous outcome is numeric;
- baseline covariates are measured before treatment;
- missing values have a clear meaning;
- a stable, non-missing, unique identifier is available when possible; and
- direct patient identifiers have been removed unless an approved protocol
  explicitly allows them.

If the selected identifier is missing, contains nulls, or is not unique, data
compilation can create a sequential `auto_id` column and reports that change.
An automatically generated row number is useful for tracking rows inside a
run, but it is not a substitute for a real source-system identifier when rows
must be linked across datasets.

### 4. Ask the workflow to inspect the uploaded data

Send an empty JSON object to run the current stage without adding a user
message:

```bash
curl -sS -X POST \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'
```

The first useful response normally profiles the data and asks for input.

### 5. Send a user message

```bash
curl -sS -X POST \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_text": "Please summarize the dataset and tell me what should be checked first."
  }'
```

In a `causal` conversation, continue with plain, specific answers about the
study. For example, state the treatment column, outcome column, time ordering,
target population, confounders, and effect modifiers when the agent asks for
them.

The response contains:

- `messages`: assistant messages and any artifact references;
- `action`: what the client should do next;
- `current_stage_name`: the workflow stage that is now current;
- `current_stage_status`: the status returned by the step that just ran; and
- `working_dataset`: the current dataset ID and whether it is frozen.

Possible actions:

- `NEEDS_DATA`: upload a CSV.
- `NEEDS_INPUT`: answer the assistant or revise the requested information.
- `NONE`: no user input is required for that transition. If a completed step
  moved the workflow to the next stage, call the messages endpoint again with
  `{}` to run that stage.

`current_stage_status` describes the step that just finished, while
`current_stage_name` is calculated after its state update. It is therefore
normal to see `DONE` together with the name of the next stage.

### 6. Read the current snapshot without running anything

```bash
curl -sS \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE?message_limit=50" \
  -H "Authorization: Bearer $TOKEN"
```

This returns recent messages, the completed/current workflow state names, and
the current working dataset. It does not execute a workflow node or call the
LLM.

`message_limit` can be from `0` to `50`. A value of `0` returns no messages
but still returns workflow and dataset information.

## Understand the workflow

### Main causal stages

| Stage | Plain-language purpose |
| --- | --- |
| `DATA_MANUPULATION` | Load and revise the working CSV |
| `PROTOCOL_DISCUSSION` | Agree on the causal research question and variables |
| `DATA_COMPILATION` | Build transformations and validate the analysis-ready design |
| `MODEL_SELECTION` | Select an EconML causal estimator |
| `MODEL_TRAIN` | Fit and save the model and training artifacts |
| `CAUSAL_INFERENCE` | Estimate, summarize, and query treatment effects |

Some stages finish automatically. Others remain `PENDING` with
`NEEDS_INPUT` until the user supplies or confirms information.

### Current causal design and model support

The current structured causal specification supports:

- binary treatment with explicit treated and control values;
- binary outcome with explicit event and non-event values, or a continuous
  numeric outcome;
- `RCT` or `OBSERVATIONAL` study type;
- an optional negative-control outcome;
- separate covariate and effect-modifier lists; and
- one identifier column.

Treatment and outcome must be different. Covariates and effect modifiers
cannot contain the treatment or outcome, cannot overlap one another, and
cannot contain duplicate names. Every referenced column must exist in the
compiled dataset.

The current model catalog contains:

- Linear Double Machine Learning (`econml.dml.LinearDML`);
- Sparse Linear Double Machine Learning (`econml.dml.SparseLinearDML`);
- Kernel Double Machine Learning (`econml.dml.KernelDML`);
- Causal Forest Double Machine Learning (`econml.dml.CausalForestDML`);
- Linear Doubly Robust Learner (`econml.dr.LinearDRLearner`);
- Sparse Linear Doubly Robust Learner
  (`econml.dr.SparseLinearDRLearner`); and
- Forest Doubly Robust Learner (`econml.dr.ForestDRLearner`).

Availability in the catalog does not mean every model is valid for every
dataset. The selection and validation stages check the current protocol and
data, and a statistician should review the final choice.

### Companion stages

- `DATA_STATISTICS` answers read-only statistical or chart questions.
- `GENERAL_QUERIES` answers relevant questions that do not belong to the
  current main stage.
- `SHAP_EXPLANATION` can calculate and query post-training SHAP feature
  importance when enabled.
- `CAUSAL_VALIDATE` can perform an expensive post-training validation workflow
  when enabled.

The LLM routes a user message between the current stage and the allowed
companion stages. If routing fails or proposes a disallowed stage, the backend
uses a safe allowed fallback.

### Why later results can be “invalidated”

This is expected workflow behavior, not necessarily an error. A later result
depends on earlier choices. For example, a trained model depends on the
dataset, causal protocol, transformations, and model selection.

When an earlier accepted value changes, the backend clears dependent later
state so stale results are not reused. The workflow may return to compilation,
selection, or training. This protects consistency, but it means the user must
repeat the downstream stages.

### Dataset freezing

After data compilation is accepted, the working dataset is frozen. A frozen
dataset cannot be changed or reverted through the ordinary data-manipulation
message. Revert the workflow to an earlier stage first, then revise the data.

## Inspect and compare datasets

### Get dataset rows as JSON

Use the `dataset_id` from `working_dataset` or an artifact reference:

```bash
export DATASET_ID="33333333-3333-3333-3333-333333333333"
```

Request the first 100 rows:

```bash
curl -sS \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE/datasets/$DATASET_ID?start=0&limit=100" \
  -H "Authorization: Bearer $TOKEN"
```

Pagination settings:

- `start`: zero-based row offset; default `0`;
- `limit`: maximum rows to return;
- `limit=0`: return the column names without rows; and
- no `limit`: return all remaining rows, which can be large.

### Compare the two latest working dataset versions

Compare rows by position:

```bash
curl -sS -X POST \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE/dataset-diffs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'
```

If the dataset has a stable unique key, matching by that key is usually more
meaningful:

```bash
curl -sS -X POST \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE/dataset-diffs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "key_columns": ["patient_id"]
  }'
```

The key columns must exist in both versions and must identify rows
appropriately. Invalid or duplicate keys are reported as validation errors.

The response describes:

- columns that were added or removed;
- inferred column type changes;
- inserted, deleted, and updated row counts;
- changed cells for matched updated rows; and
- summary totals.

At least two working dataset versions are required. Detailed row changes are
limited to 500 entries for a very large diff, but the summary totals still
describe the full comparison.

## Revert data or workflow state

There are two different kinds of revert. They solve different problems.

### Revert only the working dataset version

Send this exact message:

```bash
curl -sS -X POST \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_text": "revert_data_changes"
  }'
```

This moves from the current working dataset to the previous saved dataset
version. It does not rewind arbitrary workflow stages.

It cannot work when:

- there is only one dataset version;
- the prior dataset file is missing; or
- the dataset is frozen.

### Revert the causal workflow to a stage

This endpoint is supported only for a `causal` conversation:

```bash
curl -sS -X POST \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/causal/state-reversions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "state_name": "MODEL_SELECTION"
  }'
```

Common target names are:

```text
DATA_MANUPULATION
PROTOCOL_DISCUSSION
DATA_COMPILATION
MODEL_SELECTION
MODEL_TRAIN
CAUSAL_INFERENCE
CAUSAL_VALIDATE
SHAP_EXPLANATION
```

Use the exact uppercase name. Again, `DATA_MANUPULATION` is the spelling used
by this codebase.

A workflow reversion:

- clears the selected target stage so it can run again;
- clears dependent stages after it;
- leaves earlier valid state in place where appropriate; and
- does **not** remove old chat messages from the history.

Because the message history remains, use the returned snapshot and current
state—not an old assistant message—to decide what should happen next.

## Download artifacts

Assistant messages can include:

```json
{
  "artifact_refs": [
    {
      "id": "44444444-4444-4444-4444-444444444444",
      "kind": "data",
      "format": "csv",
      "artifact_meta": {
        "title": "CATE results"
      }
    }
  ]
}
```

Copy the `id`, `kind`, and `format` from the reference. The supported
combinations are:

| Kind | Format | Result |
| --- | --- | --- |
| `graph` | `json` | Vega-Lite-style graph specification |
| `data` | `csv` | Downloadable CSV |
| `data` | `json` | Stored JSON, or a JSON representation of a stored CSV |

Graph artifacts cannot be requested as CSV.

### Download a CSV artifact

```bash
export ARTIFACT_ID="44444444-4444-4444-4444-444444444444"

curl -sS \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE/artifacts/$ARTIFACT_ID?artifact_kind=data&artifact_format=csv" \
  -H "Authorization: Bearer $TOKEN" \
  -o "$ARTIFACT_ID.csv"
```

### Download a data artifact as JSON

```bash
curl -sS \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE/artifacts/$ARTIFACT_ID?artifact_kind=data&artifact_format=json" \
  -H "Authorization: Bearer $TOKEN" \
  -o "$ARTIFACT_ID.json"
```

### Download a graph specification

```bash
curl -sS \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE/artifacts/$ARTIFACT_ID?artifact_kind=graph&artifact_format=json" \
  -H "Authorization: Bearer $TOKEN" \
  -o "$ARTIFACT_ID.graph.json"
```

Use the same token and conversation that produced the artifact. An artifact
ID from another user or conversation is deliberately not available.

## Download the audit log

The audit export is useful for reviewing or archiving a research conversation:

```bash
curl -sS \
  "$BASE_URL/v1/conversations/$CONVERSATION_ID/types/$CONVERSATION_TYPE/audit-log" \
  -H "Authorization: Bearer $TOKEN" \
  -o "audit-log-$CONVERSATION_ID.zip"
```

Extract it:

```bash
unzip "audit-log-$CONVERSATION_ID.zip" \
  -d "audit-log-$CONVERSATION_ID"
```

Open `audit-log.html` from the extracted directory in a browser.

The package contains:

- a static HTML conversation report;
- workflow and orchestration state information;
- message history;
- inline graph specifications; and
- linked CSV or JSON data artifacts.

It does **not** include serialized trained model objects. The HTML uses
JavaScript libraries from the jsDelivr CDN to render Vega/Vega-Lite graphs, so
graph rendering may need internet access even after the ZIP has been
downloaded. The underlying graph specifications remain in the report.

Audit files can contain dataset-derived or sensitive information. Store and
share them with the same care as the source research data.

## API reference

All conversation-scoped paths use:

```text
/v1/conversations/{conversation_id}/types/{conversation_type}
```

`conversation_type` is either `causal` or `data`.

| Method | Path | Purpose | Auth |
| --- | --- | --- | --- |
| `GET` | `/healthz` | Check that the web service responds | No |
| `GET` | `/v1/conversations` | List conversations for the token identity | Yes |
| `POST` | `/v1/conversations` | Create a `causal` or `data` conversation | Yes |
| `GET` | `/v1/conversations/{id}/types/{type}` | Read messages and current state | Yes |
| `POST` | `/v1/conversations/{id}/types/{type}/messages` | Run a workflow step, with or without user text | Yes |
| `POST` | `/v1/conversations/{id}/types/{type}/state-reversions` | Revert a causal workflow stage | Yes |
| `GET` | `/v1/conversations/{id}/types/{type}/audit-log` | Download the audit ZIP | Yes |
| `POST` | `/v1/conversations/{id}/types/{type}/datasets` | Upload a CSV | Yes |
| `GET` | `/v1/conversations/{id}/types/{type}/datasets/{dataset_id}` | Read paginated dataset rows | Yes |
| `POST` | `/v1/conversations/{id}/types/{type}/dataset-diffs` | Compare the latest two dataset versions | Yes |
| `GET` | `/v1/conversations/{id}/types/{type}/artifacts/{artifact_id}` | Download a data or graph artifact | Yes |

There is currently no endpoint to delete a conversation, rename a
conversation, download a trained pickle model, or upload a file type other
than CSV.

### Common HTTP status codes

| Status | Meaning in this API |
| --- | --- |
| `200` | Request succeeded |
| `201` | Conversation or dataset was created |
| `400` | Bad CSV, empty upload, wrong file type, or another bad request |
| `401` | Missing or invalid bearer token |
| `404` | Conversation, state, dataset, or artifact not found |
| `409` | Request conflicts with the current workflow or saved state |
| `422` | Request or workflow validation failed |
| `500` | Unexpected backend, provider, state, or computation failure |

Most application errors use this shape:

```json
{
  "code": "validation_failed",
  "message": "Validation failed",
  "detail": "Validation error for 'field': explanation"
}
```

FastAPI request-shape errors, such as a missing required field or malformed
UUID, can instead return a `detail` list describing each invalid input.

Every response also receives `X-Request-ID` and `X-Trace-ID` headers. Keep
these values when investigating a server error because matching IDs appear in
the structured logs.

## Local files and saved state

The backend always uses local filesystem storage in this branch:

```text
.local_storage/
├── workflow_state.json
├── data/
│   └── users/<user-id>/conversations/<conversation-id>/
│       ├── datasets/<dataset-or-artifact-id>.csv
│       ├── datasets/<dataset-or-artifact-id>.json
│       └── artifacts/<artifact-id>/...
└── models/
    └── users/<user-id>/conversations/<conversation-id>/
        └── models/<model-id>/record.pkl
```

- `workflow_state.json` contains the conversation index, messages,
  orchestrator state, and node state.
- `data` contains uploaded, transformed, and generated CSV/JSON files.
- `models` contains Python pickle files for fitted models.
- DuckDB is used for working analytics, but it is not the main persistent
  database.

Do not edit `workflow_state.json` by hand while the API is running. The
backend writes it atomically and uses update counters to detect stale state.

Do not load `record.pkl` files from an untrusted source. Python pickle files
can execute code when loaded.

### Back up all local state

Stop the backend first, then run:

```bash
cp -R .local_storage .local_storage.backup
```

Use a new backup name if `.local_storage.backup` already exists.

### Start with clean local state

Stop the backend, move the state to a recoverable backup, and restart:

```bash
mv .local_storage .local_storage.before-reset
make run-api-local
```

The application recreates `.local_storage` as needed. The moved directory can
be restored later while the backend is stopped.

Deleting only `.local_storage/workflow_state.json` removes the conversation
index but leaves datasets and model files orphaned. Moving or removing the
whole `.local_storage` directory is a cleaner full reset.

`make clean` does **not** remove `.local_storage`. It removes `.venv` and
developer/test caches.

## Tests and code checks

### What is a unit test?

A unit test runs a small piece of the backend with a known input and checks
the result. API tests check HTTP behavior, workflow tests check state changes,
and statistical tests check transformations and estimators. Tests find many
regressions, but passing tests do not prove that a causal design is clinically
or scientifically valid.

### Install test dependencies and run the normal suite

```bash
make test
```

`make test` creates `.venv` if needed, installs both runtime and test
requirements, and runs `pytest` using `pytest.ini`.

Normal active tests do not make live LLM calls. Live integration and DeepEval
tests are meant to be skipped unless explicitly enabled.

### Known test status in this paper branch

The complete tracked suite is not green in the current branch. `make test`
currently stops during collection because some older tests still import
modules that are no longer present:

```text
python.implementation.workflows.nodes.dataset
python.implementation.workflows.nodes.compile_and_validate
python.domain.workflows.route
```

This is a repository test-migration issue, not an installation or LLM
credential error. The current API tests can be run independently:

```bash
PYTHONPATH=src .venv/bin/pytest -c pytest.ini -q \
  src/tests/python/adapters/api/test_routes.py \
  src/tests/python/adapters/api/test_dependencies.py \
  src/tests/python/adapters/api/test_exception_handlers.py
```

At the time of this README update, that verified subset reports `45 passed`.

When the four known stale areas are ignored, the remaining tracked suite
collects but still reports `464 passed`, `16 failed`, and `1 skipped`. The
remaining failures include test expectations that have drifted from the
current causal specification and training configuration. For example, older
fixtures omit the now-required `id_col`, and an older test expects different
causal-forest defaults.

For a paper release, do not report the repository as having a fully passing
test suite until the legacy tests are migrated or deliberately removed and
the 16 expectation failures are resolved. The commands below are still the
intended test commands, but this known status explains the present output.

Stop after the first failure:

```bash
make test-quick
```

Run one test file:

```bash
PYTHONPATH=src .venv/bin/pytest -c pytest.ini -q \
  src/tests/python/adapters/api/test_routes.py
```

Run one test by name:

```bash
PYTHONPATH=src .venv/bin/pytest -c pytest.ini -q \
  src/tests/python/adapters/api/test_routes.py::test_healthz_returns_ok
```

### Run the opt-in DeepEval prompt tests

These tests call a real LLM and can consume provider quota or money:

```bash
set -a
source .env
set +a
make test-deepeval
```

The target sets `RUN_DEEPEVAL_TESTS=1` and disables DeepEval telemetry and
dotenv behavior for the run.

In the current branch, this target is also blocked by the stale DeepEval test
import of the removed `nodes.dataset` module. Fix that test migration before
using DeepEval results in the paper.

### Run the live Vertex integration check

```bash
set -a
source .env
set +a
PYTHONPATH=src RUN_LIVE_LLM_TESTS=true .venv/bin/pytest -c pytest.ini -q \
  src/tests/python/implementation/service/llms/test_litellm_llm_service.py \
  -m integration
```

This live test is currently written for Vertex AI.

### Lint and formatting checks

```bash
make lint
make format
```

`make lint` runs Ruff. `make format` checks Black formatting without changing
files.

To apply automatic fixes:

```bash
make lint-fix
make format-fix
```

Review automatic changes before committing them.

### All Makefile commands

```bash
make help
```

| Command | What it does |
| --- | --- |
| `make install` | Install runtime dependencies |
| `make install-test` | Install runtime and test dependencies |
| `make install-dev` | Install runtime, test, Ruff, Black, and typing dependencies |
| `make dev-tools` | Alias for `make install-dev` |
| `make lint` | Check code with Ruff |
| `make lint-fix` | Let Ruff apply safe automatic fixes |
| `make format` | Check formatting with Black |
| `make format-fix` | Reformat code with Black |
| `make test` | Run the normal pytest suite |
| `make test-quick` | Run tests and stop at the first failure |
| `make test-deepeval` | Run opt-in live prompt evaluations |
| `make run-api` | Start on the current shell environment with auto-reload |
| `make run-api-local` | Load `.env` and start with auto-reload |
| `make clean` | Remove `.venv` and code/test caches, but keep research state |

## Troubleshooting

Start with three checks:

```bash
python3 --version
test -f .env && echo ".env exists"
curl -sS http://127.0.0.1:8080/healthz
```

Then read the server terminal. The most useful error is often there.

### `make: command not found`

Install `make`, or use the manual `.venv` commands in
[Quick start](#quick-start). On Windows, use WSL or run the equivalent Python
commands directly.

### `python3: command not found` or the wrong Python is used

Install Python 3.11 and make sure `python3 --version` reports it before
creating `.venv`.

If `.venv` was created with the wrong interpreter:

```bash
make clean
make install
```

`make clean` removes the environment but keeps `.local_storage`.

### A package fails to install

Check that:

- Python is 3.11;
- the machine has internet access;
- `pip` is not being forced through a broken proxy;
- enough disk space is available; and
- operating-system build tools are installed if a wheel is unavailable.

Retry:

```bash
.venv/bin/python -m pip install --upgrade pip setuptools wheel
make install
```

Do not unpin random packages as the first fix. The pinned versions are part of
the reproducible setup.

### `make` is slow or retries PyPI even after installation

The `venv` target is marked as a Makefile setup step and checks for updated
`pip`, `setuptools`, and `wheel` whenever a dependent command runs. With no
network, those checks can retry before continuing.

If `.venv` and all requirements are already installed, the API can be started
directly without the installation check:

```bash
set -a
source .env
set +a
PYTHONPATH=src .venv/bin/uvicorn python.adapters.api.app:app \
  --host "${API_HOST:-0.0.0.0}" \
  --port "${API_PORT:-8080}" \
  --reload
```

### `Missing .env`

Create it:

```bash
cp .env.example .env
```

Then configure a provider and run `make run-api-local` again.

### `.env` causes a shell syntax error

The `Makefile` loads `.env` with Bash `source`. Use simple `NAME=value`
assignments. Avoid spaces around `=`, unmatched quotes, and shell commands in
the file.

### The health check works, but the first workflow message fails

The health check does not verify LLM access. Check:

- `LLM_PROVIDER`;
- all required `LLM_MODEL_*` values;
- the provider key or Vertex credentials;
- `VERTEXAI_PROJECT` and `VERTEXAI_LOCATION` for Vertex;
- provider quota and billing;
- model availability; and
- outbound network access.

Restart after fixing `.env`.

### Vertex says credentials were not found

For a personal login:

```bash
gcloud auth application-default login
```

For a service account, check that
`GOOGLE_APPLICATION_CREDENTIALS` is an absolute path to a readable JSON file.
Application Default Credentials are different from an ordinary
`gcloud auth login`.

### Vertex returns `403`, quota, permission, or API errors

Confirm that:

- the project ID is correct;
- Vertex AI is enabled for the project;
- the authenticated identity has permission to use Vertex AI;
- billing and quota are available; and
- the requested model is allowed in the configured location.

### The provider says a model or deployment was not found

The model names in `.env.example` are examples for this code revision. Model
availability is provider-, account-, region-, and time-dependent.

- For Vertex, Google AI Studio, and OpenAI, use an available model name.
- For Azure, use the Azure deployment name.
- Set all four aliases for a non-Vertex provider.

### `Address already in use`

Another process is using port 8080. Find it:

```bash
lsof -i :8080
```

Stop that process if appropriate, or change `API_PORT` in `.env`, restart, and
update `BASE_URL`.

### `curl: (7) Failed to connect`

The backend is not listening at that host and port. Check the server terminal,
the `API_PORT` setting, and whether `make run-api-local` is still running.

### `401 Unauthorized`

Check that:

- the header is exactly `Authorization: Bearer <token>`;
- the token is not empty;
- a JWT-like token contains a UUID identity claim; and
- the token has not accidentally been changed.

For local work, a raw UUID is the least confusing token form.

### A known conversation returns `404`

The usual causes are:

- a different token, which means a different local user;
- the wrong conversation ID;
- the wrong conversation type in the path;
- state was reset; or
- the conversation belongs to another local storage directory.

List conversations with the same token and compare both the ID and type.

### CSV upload returns `400`

Check that:

- the file exists;
- `curl -F` uses the field name `file`;
- the file is not empty;
- the filename or content type identifies it as CSV;
- the delimiter and quoting form a valid CSV; and
- pandas can read it.

A quick local parse check is:

```bash
.venv/bin/python -c \
  'import pandas as pd; frame=pd.read_csv("dataset.csv"); print(frame.shape); print(frame.columns.tolist())'
```

Replace `dataset.csv` with the real path.

### CSV upload returns `422`

The conversation is not currently at the upload/manipulation stage. Read its
snapshot. If the dataset is frozen or the workflow has moved forward, revert
the causal workflow to an appropriate earlier stage before uploading.

### Dataset diff returns `422`

At least two working versions are required. If `key_columns` were supplied,
make sure they:

- exist in both versions;
- are spelled exactly;
- contain valid identifying values; and
- do not create ambiguous duplicate row identities.

Try an empty request body to compare by position.

### Artifact download returns `404` or `422`

Use the exact artifact reference from the assistant message:

- same token;
- same conversation ID and type;
- same artifact ID;
- `graph` or `data` kind; and
- `json` or `csv` format.

A graph requested as CSV correctly returns a validation error.

### A workflow step is slow

LLM calls, model fitting, CATE calculation, outer cross-validation, and SHAP
can take much longer than ordinary API requests. Model training can also use
several CPU cores.

Check the JSON logs for progress. Do not send the same message repeatedly
while the first request is still running. Duplicate concurrent requests can
compete to update the same state.

For a lighter local run:

- keep `SHAP_ENABLED=false`;
- keep `CAUSAL_VALIDATE_ENABLED=false`; or
- set `PRECISION_MEDICINE_OUTER_CV_CATE_N_JOBS=1` when validation is enabled.

### The process runs out of memory

The upload is read fully into memory and several workflow steps make pandas or
model-training copies. Use a smaller research extract, close other large
processes, disable optional expensive stages, or run on a machine with more
memory.

### A request returns `500`

1. Copy the `X-Request-ID` and `X-Trace-ID` response headers.
2. Find the matching IDs in the server's JSON logs.
3. Temporarily set `LOG_LEVEL=DEBUG` if more context is needed.
4. Check provider errors, missing local files, corrupted state, and concurrent
   requests.
5. Restart the backend only after the active request has ended.

The public response hides internal exception details for server errors. This
is intentional. Do not paste logs containing credentials or patient data into
public bug reports.

### `workflow_state.json` is corrupt

Stop the backend and validate the file:

```bash
python3 -m json.tool .local_storage/workflow_state.json > /dev/null
```

If validation fails, restore a known backup. If no backup is available, move
the complete `.local_storage` directory aside and start a clean run. Do not
delete only part of a conversation unless you understand the workflow-state,
dataset, and model references.

### Tests are skipped

DeepEval and live-provider tests are opt-in, so skips for live tests are
expected after the suite can collect. Skips are separate from the known stale
imports and test failures described in
[Tests and code checks](#tests-and-code-checks). Use live commands only when
provider calls are intended.

### A warning fails a test

`pytest.ini` turns most warnings into errors. This is deliberate because
dependency warnings can reveal behavior changes. Read the first warning and
fix or deliberately configure it instead of hiding all warnings.

## Important limitations and caveats

### Local research runtime, not a production service

This branch is designed for local paper experiments. Before any real
deployment, address at least the following:

- JWT signatures and expiry are not validated.
- CORS currently allows every origin.
- Conversations and data are stored as unencrypted local files.
- The health check is shallow.
- There is no database migration system.
- There is no conversation deletion API.
- There is no explicit HTTP upload-size limit.
- Long computations run inside the API process.
- State locking is process-local, not distributed.
- The repositories are not designed for several API workers sharing the same
  `.local_storage`.

Run one API server process against a local storage directory. Do not start
several backend instances or workers against the same files.

### Avoid simultaneous requests to one conversation

The state repository has update counters that can detect a stale write, but a
full workflow step is not one distributed transaction. Wait for one message,
upload, revert, or training request to finish before starting another request
for the same conversation.

### Data privacy

Uploaded datasets, transformed datasets, messages, artifacts, and models are
saved locally. Prompts and dataset-derived context are also sent to the
configured LLM provider during the workflow.

Do not use identifiable patient data unless:

- the research protocol allows it;
- local storage is secured;
- the provider and account are approved for that data;
- regional and institutional requirements are satisfied; and
- retention and deletion procedures are defined.

Prefer de-identified or synthetic data for development.

### Statistical and clinical interpretation

The software can validate many mechanical conditions, but it cannot prove
exchangeability, correct causal structure, absence of unmeasured confounding,
positivity, consistency, transportability, or clinical relevance.

Always review:

- treatment and outcome definitions;
- time ordering;
- inclusion and exclusion rules;
- missing-data handling;
- covariate and effect-modifier roles;
- overlap and treatment support;
- transformations and encodings;
- model assumptions;
- uncertainty intervals and diagnostics; and
- sensitivity or negative-control results.

### Reproducibility is not perfect

The statistical configuration uses a fixed seed by default, but exact results
can still vary with:

- hardware and process scheduling;
- BLAS or native-library behavior;
- dependency versions;
- LLM provider behavior;
- preview model updates;
- routing decisions;
- prompt outputs; and
- external service retries.

Pinned Python requirements and saved audit logs reduce this uncertainty but
do not remove it.

### Local experiment directories

The repository's `.gitignore` excludes `experiments/`, `data*`, and local
storage. A working copy may contain paper-specific experiment scripts and
datasets that are not part of the tracked backend source. Do not assume those
files will be present in a fresh clone. Publish or archive research inputs
separately when the study protocol permits it.


## Reproducibility checklist for a paper

For every reported run, record:

- the Git commit hash;
- Python version;
- operating system and machine type;
- dependency versions or the unchanged pinned requirement files;
- LLM provider;
- all four concrete LLM model or deployment names;
- provider region;
- whether model names were preview versions;
- `SHAP_ENABLED`;
- `CAUSAL_VALIDATE_ENABLED`;
- statistical configuration and seed;
- outer-validation worker count;
- token identity used for the local run, stored securely;
- conversation ID and type;
- source dataset version and cryptographic checksum;
- exact user messages or the audit ZIP;
- downloaded result artifacts;
- start and end times;
- warnings, retries, and failed attempts; and
- any source-code or prompt changes.

Useful commands:

```bash
git rev-parse HEAD
python3 --version
.venv/bin/python -m pip freeze
shasum -a 256 /absolute/path/to/dataset.csv
```

On Linux, `sha256sum` can be used instead of `shasum -a 256`.

Archive the following together when data governance permits:

1. the exact source revision;
2. the non-secret configuration values;
3. the dataset checksum and controlled dataset version;
4. the conversation audit ZIP;
5. downloaded CSV/JSON artifacts;
6. test output; and
7. a short note describing any manual decisions.

Never archive API keys or service-account credentials with the paper
artifacts.

## Project structure

```text
.
├── .env.example
├── Makefile
├── README.md
├── requirements.txt
├── requirements-test.txt
├── requirements-dev.txt
├── pytest.ini
├── pyproject.toml
├── docs/
│   └── Architecture.MD
└── src/
    ├── python/
    │   ├── adapters/api/
    │   ├── domain/
    │   └── implementation/
    │       ├── repo/
    │       ├── service/
    │       └── workflows/
    └── tests/
```

Important files:

- `src/python/adapters/api/app.py`: creates the FastAPI application.
- `src/python/adapters/api/routes.py`: defines the HTTP endpoints.
- `src/python/adapters/api/schemas.py`: defines request and response JSON.
- `src/python/adapters/api/exception_handlers.py`: maps application errors to
  HTTP responses.
- `src/python/implementation/workflows/workflow_app.py`: manages
  conversations and workflow execution.
- `src/python/implementation/workflows/dataflow_app.py`: manages uploads,
  dataset reads, diffs, and artifacts.
- `src/python/implementation/workflows/audit_log_app.py`: creates the audit ZIP.
- `src/python/implementation/workflows/ochestrator/`: routes and persists
  workflow stages.
- `src/python/implementation/workflows/nodes/`: implements each stage.
- `src/python/implementation/repo/`: local dataset, model, analytics, and
  workflow-state storage.
- `src/python/implementation/service/llms/`: LLM provider configuration and
  LiteLLM adapter.
- `src/python/implementation/workflows/tools/causal/inference/econml/model_training_config.py`:
  statistical training defaults.
- `src/tests/`: unit, API, workflow, repository, and statistical tests.

Several internal names preserve historical spelling, including
`ochestrator`, `DATA_MANUPULATION`, and some filenames containing
`manupulation` or `valiation`. They work as written. Renaming them is a code
migration, not a documentation correction.
