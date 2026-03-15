from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from python.adapters.api.schemas import (
    CreateConversationRequest,
    CreateConversationResponse,
    InvokeRequest,
    InvokeResponse,
    UploadDatasetResponse,
)
from python.domain.service.auth_service import AuthService, AuthenticatedUser
from python.implementation.service.firebsae_auth_service import (
    AuthServiceError,
    FirebaseAuthService,
    InvalidTokenError,
)

from python.implementation.workflows.depinit import make_workflow_app
from python.implementation.workflows.workflow_app import WorkflowApp, WorkflowRequest

log = logging.getLogger(__name__)

app = FastAPI(title="Causal Copilot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


@lru_cache(maxsize=1)
def get_workflow_app() -> WorkflowApp:
    return make_workflow_app()


@lru_cache(maxsize=1)
def get_auth_service() -> AuthService:
    return FirebaseAuthService(app=FirebaseAuthService.get_firebase_auth_default_app())


def get_authenticated_user(authorization: str | None = Header(default=None)) -> AuthenticatedUser:
    if authorization is None:
        raise _unauthorized("Missing Authorization header.")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise _unauthorized("Authorization header must use the Bearer scheme.")

    normalized_token = token.strip()
    if not normalized_token:
        raise _unauthorized("Bearer token is missing.")

    try:
        return get_auth_service().verify_token_and_get_user(normalized_token)
    except (InvalidTokenError, ValueError) as exc:
        raise _unauthorized("Invalid or expired bearer token.") from exc
    except AuthServiceError as exc:
        log.exception("authentication failed")
        raise HTTPException(status_code=500, detail="authentication service unavailable") from exc
    except Exception as exc:
        log.exception("authentication failed")
        raise HTTPException(status_code=500, detail="authentication service unavailable") from exc


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get(path="/v1/conversations/{conversation_id}/artifacts/{artifact_id}")
def get_artifact(
    conversation_id: UUID,
    artifact_id: UUID,
    authenticated_user: AuthenticatedUser = Depends(get_authenticated_user),
    workflow: WorkflowApp = Depends(get_workflow_app),
):
    try:
        ref = workflow.get_artifact(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="artifact not found")
    except Exception as e:
        log.exception("artifact download failed")
        raise HTTPException(status_code=500, detail=str(e))

    return Response(
        content=ref.content,
        media_type=ref.mime,
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "private, max-age=60",
        },
    )


@app.post(
    "/v1/conversations/{conversation_id}/datasets",
    response_model=UploadDatasetResponse,
)
async def upload_dataset_csv(
    conversation_id: UUID,
    file: UploadFile = File(...),
    authenticated_user: AuthenticatedUser = Depends(get_authenticated_user),
    workflow: WorkflowApp = Depends(get_workflow_app),
):
    file_name = (file.filename or "").strip()
    content_type = (file.content_type or "").lower()
    is_csv_name = file_name.lower().endswith(".csv")
    is_csv_type = "csv" in content_type or content_type == "application/vnd.ms-excel"

    if not is_csv_name and not is_csv_type:
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")

    csv_bytes = await file.read()
    if not csv_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        dataset_id = await asyncio.to_thread(
            workflow.upload_csv_data,
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
            csv_bytes=csv_bytes,
        )
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("dataset upload failed")
        raise HTTPException(status_code=500, detail=str(e))

    return UploadDatasetResponse(
        user_id=authenticated_user.uid,
        conversation_id=conversation_id,
        dataset_id=dataset_id,
    )
    
@app.post("/v1/conversations", response_model=CreateConversationResponse)
def create_conversation(
    req: CreateConversationRequest | None = None,
    authenticated_user: AuthenticatedUser = Depends(get_authenticated_user),
    workflow: WorkflowApp = Depends(get_workflow_app),
):
    _ = req
    user_id = authenticated_user.uid
    conversation_id = uuid4()
    logging.warning(f"Creating conversation: user_id={user_id}, conversation_id={conversation_id}")
    workflow.create_conversation(user_id=user_id, conversation_id=conversation_id)
    return CreateConversationResponse(user_id=user_id, conversation_id=conversation_id)


@app.post("/v1/conversations/{conversation_id}/invoke", response_model=InvokeResponse)
async def invoke_once(
    conversation_id: UUID,
    req: InvokeRequest,
    authenticated_user: AuthenticatedUser = Depends(get_authenticated_user),
    workflow: WorkflowApp = Depends(get_workflow_app),
):
    # single node execution per request
    txt = (req.user_text or "").strip() or None

    try:
        resp = await asyncio.to_thread(
            workflow.handle,
            WorkflowRequest(
                user_id=authenticated_user.uid,
                conversation_id=conversation_id,
                user_message=txt,
            ),
        )
    except Exception as e:
        log.exception("invoke failed")
        raise HTTPException(status_code=500, detail=str(e))

    return InvokeResponse(
        conversation_id=conversation_id,
        user_id=authenticated_user.uid,
        node_message=resp.node_message,
        needs_input=resp.needs_input,
        needs_data=resp.needs_data,
        current_stage=str(resp.current_stage),
        artifact_ids =resp.artifact_ids,
        current_stage_status=str(resp.current_stage_status),
    )
