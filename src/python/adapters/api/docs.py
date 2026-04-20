from __future__ import annotations

OPENAPI_TAGS = [
    {"name": "system", "description": "Health and service status endpoints."},
    {
        "name": "conversations",
        "description": (
            "Create, list, inspect, message, and revert authenticated workflow conversations."
        ),
    },
    {"name": "datasets", "description": "Upload CSV datasets for a conversation."},
    {
        "name": "artifacts",
        "description": (
            "Download workflow artifacts. "
            "Artifact kind enum: `graph | data`. Artifact format enum: `json | csv`."
        ),
    },
]

API_TITLE = "AitiaMed Copilot API"
API_VERSION = "0.1.0"
API_SUMMARY = "Authenticated API for medical causal inference workflow interactions."
API_DESCRIPTION = (
    "Create typed conversations, upload CSV datasets, send workflow messages, inspect workflow state, and download generated artifacts.\n\n"
    "Authentication:\n"
    "- All `/v1/...` endpoints require a Firebase Bearer token.\n"
    "- `/healthz` is public.\n"
    "- The authenticated Firebase identity is resolved server-side; clients do not send a `user_id`.\n\n"
    "Conversation metadata:\n"
    "- `POST /v1/conversations` accepts an optional `conversation_name`.\n"
    "- `GET /v1/conversations` returns `conversation_name` and `last_updated_at_utc` for each conversation.\n\n"
    "Conversation scope:\n"
    "- Conversation-scoped endpoints use `/v1/conversations/{conversation_id}/types/{conversation_type}`.\n"
    "- `conversation_type` enum: `causal | data`\n\n"
    "Workflow messages:\n"
    '- Send `user_text="revert_data_changes"` to the `/messages` endpoint to request a dataset-history revert inside the workflow.\n\n'
    "Artifacts:\n"
    "- `artifact_kind` enum: `graph | data`\n"
    "- `artifact_format` enum: `json | csv`\n"
    "- Valid combinations: `graph -> json`, `data -> json | csv`"
)
