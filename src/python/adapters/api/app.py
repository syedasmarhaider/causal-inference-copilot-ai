from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from python.adapters.api.dependencies import (
    ARTIFACT_ID_PATH_PARAM,
    AUTHENTICATED_USER_DEP,
    CONVERSATION_ID_PATH_PARAM,
    CREDENTIALS_SECURITY,
    UPLOAD_DATASET_FILE_PARAM,
    WORKFLOW_APP_DEP,
    get_auth_service,
    get_authenticated_user,
    get_workflow_app,
)
from python.adapters.api.docs import (
    API_DESCRIPTION,
    API_SUMMARY,
    API_TITLE,
    API_VERSION,
    OPENAPI_TAGS,
)
from python.adapters.api.exception_handlers import register_exception_handlers
from python.adapters.api.request_context_middleware import RequestContextMiddleware
from python.adapters.api.routes import api_router
from python.implementation.service.logging.default_logging import configure_default_logging

__all__ = [
    "app",
    "get_workflow_app",
    "get_auth_service",
    "get_authenticated_user",
    "CREDENTIALS_SECURITY",
    "CONVERSATION_ID_PATH_PARAM",
    "ARTIFACT_ID_PATH_PARAM",
    "UPLOAD_DATASET_FILE_PARAM",
    "AUTHENTICATED_USER_DEP",
    "WORKFLOW_APP_DEP",
]

configure_default_logging(
    service_name=os.getenv("LOG_SERVICE_NAME"),
    level=os.getenv("LOG_LEVEL"),
)


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    summary=API_SUMMARY,
    description=API_DESCRIPTION,
    openapi_tags=OPENAPI_TAGS,
)

app.add_middleware(RequestContextMiddleware)

# TODO: for now no CORS but later I will add restrictions based on allowed origins for the frontend app(s) that will consume this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
register_exception_handlers(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("python.adapters.api.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
