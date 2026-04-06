from __future__ import annotations

import asyncio
from python.implementation.service.logging.default_logging import get_logger
from functools import lru_cache

from fastapi import Depends, File, Header, HTTPException, Path, Query, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from python.domain.service.auth_service import AuthenticatedUser, AuthService
from python.implementation.service.firebsae_auth_service import (
    AuthServiceError,
    FirebaseAuthService,
    InvalidTokenError,
)
from python.implementation.workflows.dataflow_app import DataflowApp
from python.implementation.workflows.depinit import make_apps
from python.implementation.workflows.workflow_app import WorkflowApp

log = get_logger(__name__)

bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Firebase Bearer JWT. Paste the raw Firebase ID token.",
)
CREDENTIALS_SECURITY = Security(bearer_scheme)


CONVERSATION_ID_PATH_PARAM = Path(description="Conversation UUID.")
ARTIFACT_ID_PATH_PARAM = Path(description="Artifact UUID to download.")
ARTIFACT_KIND_QUERY_PARAM = Query(
    ...,
    description="Artifact kind enum. Allowed values: `graph` or `data`.",
)
ARTIFACT_FORMAT_QUERY_PARAM = Query(
    ...,
    description=(
        "Artifact format enum. Allowed values: `json` or `csv`. "
        "Valid combinations: `graph -> json`, `data -> json|csv`."
    ),
)
UPLOAD_DATASET_FILE_PARAM = File(
    ...,
    description="CSV file to upload for this conversation.",
)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


@lru_cache(maxsize=1)
def get_apps() -> tuple[WorkflowApp, DataflowApp]:
    return make_apps()


@lru_cache(maxsize=1)
def get_workflow_app() -> WorkflowApp:
    return get_apps()[0]


@lru_cache(maxsize=1)
def get_dataflow_app() -> DataflowApp:
    return get_apps()[1]


@lru_cache(maxsize=1)
def get_auth_service() -> AuthService:
    return FirebaseAuthService(app=FirebaseAuthService.get_firebase_auth_default_app())


async def get_authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = CREDENTIALS_SECURITY,
    authorization: str | None = Header(default=None, include_in_schema=False),
) -> AuthenticatedUser:
    if authorization is None:
        raise _unauthorized("Missing Authorization header.")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise _unauthorized("Authorization header must use the Bearer scheme.")

    normalized_token = token.strip()
    if not normalized_token:
        raise _unauthorized("Bearer token is missing.")
    if credentials is None:
        raise _unauthorized("Invalid or expired bearer token.")

    try:
        auth_service = get_auth_service()
        return await asyncio.to_thread(auth_service.verify_token_and_get_user, normalized_token)
    except (InvalidTokenError, ValueError) as exc:
        raise _unauthorized("Invalid or expired bearer token.") from exc
    except AuthServiceError as exc:
        log.exception("authentication failed", error=exc)
        raise HTTPException(status_code=500, detail="authentication service unavailable") from exc
    except Exception as exc:
        log.exception("authentication failed", error=exc)
        raise HTTPException(status_code=500, detail="authentication service unavailable") from exc


AUTHENTICATED_USER_DEP = Depends(get_authenticated_user)
WORKFLOW_APP_DEP = Depends(get_workflow_app)
DATAFLOW_APP_DEP = Depends(get_dataflow_app)
