from __future__ import annotations

OPENAPI_TAGS = [
    {"name": "system", "description": "Health and service status endpoints."},
    {
        "name": "conversations",
        "description": "Create, inspect, invoke, and revert authenticated workflow conversations.",
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
    "Upload a CSV dataset, invoke the causal workflow, inspect the current stage, and download generated artifacts.\n\n"
    "Authentication:\n"
    "- All `/v1/...` endpoints require a Firebase Bearer token.\n"
    "- `/healthz` is public.\n"
    "- The authenticated Firebase identity is mapped to the internal workflow `user_id`.\n\n"
    "Workflow responses:\n"
    "- `invoke` and `lateststate` return assistant messages, workflow action, stage/status, and latest working dataset info.\n"
    "- To trigger dataset-history revert inside workflow execution, send `user_text=\"revert_data_changes\"` to the invoke endpoint.\n\n"
    "Artifacts:\n"
    "- `artifact_kind` enum: `graph | data`\n"
    "- `artifact_format` enum: `json | csv`\n"
    "- Valid combinations: `graph -> json`, `data -> json | csv`"
)
