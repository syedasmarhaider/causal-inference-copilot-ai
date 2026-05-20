from __future__ import annotations

OPENAPI_TAGS = [
    {"name": "system", "description": "Health and service status endpoints."},
    {
        "name": "conversations",
        "description": (
            "Create, list, inspect, message, and revert authenticated workflow conversations."
        ),
    },
    {
        "name": "datasets",
        "description": (
            "Upload CSV datasets, page through stored dataset rows, and calculate structured diffs between the two latest working dataset versions."
        ),
    },
    {
        "name": "artifacts",
        "description": (
            "Download workflow artifacts. "
            "Artifact kind enum: `graph | data`. Artifact format enum: `json | csv`."
        ),
    },
]

API_TITLE = "AitiaMed Agent API"
API_VERSION = "0.1.0"
API_SUMMARY = "Authenticated API for medical causal inference workflow interactions."
API_DESCRIPTION = (
    "Create typed conversations, upload CSV datasets, send workflow messages, inspect workflow state, and download generated artifacts.\n\n"
    "Authentication:\n"
    "- All `/v1/...` endpoints require a local bearer token containing a UUID identity.\n"
    "- The token may be a raw UUID or JWT-like token with an `id`, `uuid`, `user_id`, `uid`, or `sub` UUID claim.\n"
    "- `/healthz` is public.\n"
    "- The authenticated local identity is read from the bearer token; signatures are not validated in this local branch.\n\n"
    "Conversation metadata:\n"
    "- `POST /v1/conversations` accepts an optional `conversation_name`.\n"
    "- `GET /v1/conversations` returns `conversation_name` and `last_updated_at_utc` for each conversation.\n\n"
    "Conversation scope:\n"
    "- Conversation-scoped endpoints use `/v1/conversations/{conversation_id}/types/{conversation_type}`.\n"
    "- `conversation_type` enum: `causal | data`\n\n"
    "Workflow messages:\n"
    '- Send `user_text="revert_data_changes"` to the `/messages` endpoint to request a dataset-history revert inside the workflow.\n\n'
    "Dataset diffs:\n"
    "- `POST /v1/conversations/{conversation_id}/types/{conversation_type}/dataset-diffs` compares the previous working dataset version to the current one.\n"
    "- Send an empty body for positional row matching, or send `key_columns` to match rows by business key.\n"
    "- The response keeps the existing schema: `previous_dataset_id`, `current_dataset_id`, and `diff` with `schema_diff`, `row_changes`, and `summary`.\n"
    "- `row_changes` contains detailed matched-row updates only. Inserted/deleted rows are counted in `summary`, unchanged rows/cells are omitted, and very large diffs may truncate `row_changes` while leaving `summary` complete.\n\n"
    "Dataset paging:\n"
    "- `GET /v1/conversations/{conversation_id}/types/{conversation_type}/datasets/{dataset_id}` returns dataset rows as JSON.\n"
    "- Use `start` for the zero-based row offset and optional `limit` for page size.\n"
    "- `limit=0` returns only dataset column metadata.\n\n"
    "Artifacts:\n"
    "- `artifact_kind` enum: `graph | data`\n"
    "- `artifact_format` enum: `json | csv`\n"
    "- Valid combinations: `graph -> json`, `data -> json | csv`"
)
