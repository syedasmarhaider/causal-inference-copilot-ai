from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import Response

from python.adapters.api.dependencies import (
    ARTIFACT_ID_PATH_PARAM,
    AUTHENTICATED_USER_DEP,
    CONVERSATION_ID_PATH_PARAM,
    UPLOAD_DATASET_FILE_PARAM,
    WORKFLOW_APP_DEP,
)
from python.adapters.api.schemas import (
    CreateConversationResponse,
    InvokeRequest,
    InvokeResponse,
    RevertStateRequest,
    UploadDatasetResponse,
)
from python.domain.models.errors import ConversationNotFoundError, StateNotFoundError
from python.domain.service.auth_service import AuthenticatedUser
from python.implementation.workflows.workflow_app import WorkflowApp, WorkflowRequest

log = logging.getLogger(__name__)

api_router = APIRouter()


@api_router.get(
    "/healthz",
    tags=["system"],
    summary="Health check",
    description="Public endpoint used for service liveness and readiness checks.",
)
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@api_router.get(
    path="/v1/conversations/{conversation_id}/artifacts/{artifact_id}",
    tags=["artifacts"],
    summary="Download an artifact",
    description="Downloads an artifact generated for the authenticated user's conversation.",
    response_description="Artifact bytes streamed inline.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Artifact not found for this authenticated conversation."},
        500: {"description": "Unexpected artifact retrieval failure."},
    },
)
async def get_artifact(
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    artifact_id: UUID = ARTIFACT_ID_PATH_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> Response:
    def _load_artifact():
        workflow.raise_if_userid_not_relates_to_conversation_id(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
        )
        return workflow.get_artifact(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
        )

    try:
        ref = await asyncio.to_thread(_load_artifact)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="conversation not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    except Exception as exc:
        log.exception("artifact download failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=ref.content,
        media_type=ref.mime,
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "private, max-age=60",
        },
    )


@api_router.get(
    path="/v1/conversations/{conversation_id}/lateststate",
    tags=["conversations"],
    summary="Get latest conversation state",
    description="Returns the latest state name for the authenticated user's conversation.",
    response_description="Latest state name for the conversation.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Conversation not found for this authenticated user."},
        500: {"description": "Unexpected failure retrieving conversation state."},
    },
)
async def get_latest_conversation_state(
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> InvokeResponse:
    def _load_latest_state():
        workflow.raise_if_userid_not_relates_to_conversation_id(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
        )
        resp = workflow.get_last_conversation_state(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
        )
        if resp is None:
            raise ConversationNotFoundError(
                user_id=authenticated_user.uid,
                conversation_id=conversation_id,
            )
        return resp

    try:
        resp = await asyncio.to_thread(_load_latest_state)
        return InvokeResponse(
            conversation_id=conversation_id,
            user_id=authenticated_user.uid,
            node_message=resp.node_message,
            needs_input=resp.needs_input,
            needs_data=resp.needs_data,
            current_stage=str(resp.current_stage),
            artifact_ids=resp.artifact_ids,
            current_stage_status=str(resp.current_stage_status),
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="conversation not found") from exc
    except Exception as exc:
        log.exception("failed to get latest conversation state")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api_router.post(
    "/v1/conversations/{conversation_id}/datasets",
    response_model=UploadDatasetResponse,
    tags=["datasets"],
    summary="Upload a CSV dataset",
    description="Uploads a CSV file for the authenticated user's conversation.",
    response_description="Dataset stored successfully.",
    responses={
        400: {"description": "Invalid CSV content or empty upload."},
        401: {"description": "Missing or invalid Bearer token."},
        409: {"description": "Dataset already exists and overwrite is disabled."},
        500: {"description": "Unexpected dataset upload failure."},
    },
)
async def upload_dataset_csv(
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    file: UploadFile = UPLOAD_DATASET_FILE_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> UploadDatasetResponse:
    file_name = (file.filename or "").strip()
    content_type = (file.content_type or "").lower()
    is_csv_name = file_name.lower().endswith(".csv")
    is_csv_type = "csv" in content_type or content_type == "application/vnd.ms-excel"

    if not is_csv_name and not is_csv_type:
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")

    try:
        csv_bytes = await file.read()
    finally:
        await file.close()

    if not csv_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    def _upload_csv() -> UUID:
        workflow.raise_if_userid_not_relates_to_conversation_id(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
        )
        return workflow.upload_csv_data(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
            csv_bytes=csv_bytes,
        )

    try:
        dataset_id = await asyncio.to_thread(_upload_csv)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="conversation not found") from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("dataset upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return UploadDatasetResponse(
        user_id=authenticated_user.uid,
        conversation_id=conversation_id,
        dataset_id=dataset_id,
    )


@api_router.post(
    "/v1/conversations",
    response_model=CreateConversationResponse,
    tags=["conversations"],
    summary="Create a conversation",
    description="Creates a new workflow conversation for the authenticated user.",
    response_description="Created conversation identifiers.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        500: {"description": "Unexpected conversation creation failure."},
    },
)
async def create_conversation(
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> CreateConversationResponse:
    user_id = authenticated_user.uid
    conversation_id = await asyncio.to_thread(workflow.create_conversation, user_id=user_id)
    return CreateConversationResponse(user_id=user_id, conversation_id=conversation_id)


@api_router.post(
    "/v1/conversations/{conversation_id}/invoke",
    response_model=InvokeResponse,
    tags=["conversations"],
    summary="Invoke the workflow",
    description="Advances the authenticated user's conversation by one workflow step.",
    response_description="Workflow node response for the current step.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        422: {"description": "Request body validation failed."},
        500: {"description": "Unexpected workflow execution failure."},
    },
)
async def invoke_once(
    req: InvokeRequest,
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> InvokeResponse:
    txt = (req.user_text or "").strip() or None

    def _invoke():
        workflow.raise_if_userid_not_relates_to_conversation_id(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
        )
        return workflow.handle(
            WorkflowRequest(
                user_id=authenticated_user.uid,
                conversation_id=conversation_id,
                user_message=txt,
            )
        )

    try:
        resp = await asyncio.to_thread(_invoke)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="conversation not found") from exc
    except Exception as exc:
        log.exception("invoke failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return InvokeResponse(
        conversation_id=conversation_id,
        user_id=authenticated_user.uid,
        node_message=resp.node_message,
        needs_input=resp.needs_input,
        needs_data=resp.needs_data,
        current_stage=str(resp.current_stage),
        artifact_ids=resp.artifact_ids,
        current_stage_status=str(resp.current_stage_status),
    )


@api_router.post(
    "/v1/conversations/{conversation_id}/revert",
    tags=["conversations"],
    summary="Revert to a previous state",
    description="Reverts the authenticated user's conversation to a previous state.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Conversation or state not found for this authenticated user."},
        500: {"description": "Unexpected failure reverting conversation state."},
    },
)
async def revert_to_state(
    req: RevertStateRequest,
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> None:
    user_id = authenticated_user.uid
    state_name = (req.state_name or "").strip()
    if not state_name:
        raise HTTPException(status_code=422, detail="state_name must be a non-empty string")

    def _revert() -> None:
        workflow.raise_if_userid_not_relates_to_conversation_id(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
        )
        workflow.revert_to_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
        )

    try:
        await asyncio.to_thread(_revert)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="conversation not found") from exc
    except StateNotFoundError as exc:
        raise HTTPException(status_code=404, detail="state not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="state not found") from exc
    except Exception as exc:
        log.exception("failed to revert conversation state")
        raise HTTPException(status_code=500, detail=str(exc)) from exc