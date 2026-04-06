from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import Response

from python.adapters.api.dependencies import (
    ARTIFACT_FORMAT_QUERY_PARAM,
    ARTIFACT_ID_PATH_PARAM,
    ARTIFACT_KIND_QUERY_PARAM,
    AUTHENTICATED_USER_DEP,
    CONVERSATION_ID_PATH_PARAM,
    DATAFLOW_APP_DEP,
    UPLOAD_DATASET_FILE_PARAM,
    WORKFLOW_APP_DEP,
)
from python.adapters.api.schemas import (
    ArtifactRefResponse,
    ChatMessageResponse,
    CreateConversationResponse,
    InvokeRequest,
    InvokeResponse,
    RevertStateRequest,
    UploadDatasetResponse,
    WorkingDatasetInfoResponse,
)
from python.domain.models.errors import ConversationNotFoundError
from python.domain.models.models import (
    ArtifactFormat,
    ArtifactKind,
    ArtifactRef,
    ChatMessage,
    WorkingDatasetInfo,
)
from python.domain.service.auth_service import AuthenticatedUser
from python.implementation.workflows.dataflow_app import DataflowApp
from python.implementation.workflows.workflow_app import WorkflowApp, WorkflowResponse

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
    description=(
        "Downloads an artifact generated for the authenticated user's conversation. "
        "Pass `artifact_kind` and `artifact_format` explicitly. "
        "Valid combinations: `graph -> json`, `data -> json|csv`."
    ),
    response_description="Artifact bytes streamed inline or as a downloadable CSV attachment.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Artifact or conversation not found for this authenticated user."},
        422: {"description": "Invalid artifact kind/format combination."},
        500: {"description": "Unexpected artifact retrieval failure."},
    },
)
async def get_artifact(
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    artifact_id: UUID = ARTIFACT_ID_PATH_PARAM,
    artifact_kind: ArtifactKind = ARTIFACT_KIND_QUERY_PARAM,
    artifact_format: ArtifactFormat = ARTIFACT_FORMAT_QUERY_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    dataflow: DataflowApp = DATAFLOW_APP_DEP,
) -> Response:
    def _load_artifact():
        return dataflow.get_artifact(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            artifact_format=artifact_format,
        )

    ref = await asyncio.to_thread(_load_artifact)
    disposition = (
        f'attachment; filename="{artifact_id}.{artifact_format}"'
        if artifact_format == "csv"
        else "inline"
    )

    return Response(
        content=ref.content,
        media_type=ref.mime,
        headers={
            "Content-Disposition": disposition,
            "Cache-Control": "private, max-age=60",
        },
    )


@api_router.get(
    path="/v1/conversations/{conversation_id}/lateststate",
    tags=["conversations"],
    summary="Get latest conversation state",
    description=(
        "Returns the latest workflow response shape for the authenticated user's conversation, "
        "including assistant messages, current stage, current action, and latest working dataset info."
    ),
    response_description="Latest workflow response for the conversation.",
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
    dataflow: DataflowApp = DATAFLOW_APP_DEP,
) -> InvokeResponse:
    def _load_latest_state() -> InvokeResponse:
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
        latest_dataset = dataflow.get_current_working_dataset_info(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
        )
        return _to_invoke_response(
            conversation_id=conversation_id,
            user_id=authenticated_user.uid,
            workflow_response=resp,
            latest_working_dataset=latest_dataset,
        )

    return await asyncio.to_thread(_load_latest_state)


@api_router.post(
    "/v1/conversations/{conversation_id}/datasets",
    response_model=UploadDatasetResponse,
    tags=["datasets"],
    summary="Upload a CSV dataset",
    description="Uploads a CSV file for the authenticated user's conversation using the dataflow app.",
    response_description="Dataset stored successfully.",
    responses={
        400: {"description": "Invalid CSV content or empty upload."},
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Conversation not found for this authenticated user."},
        500: {"description": "Unexpected dataset upload failure."},
    },
)
async def upload_dataset_csv(
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    file: UploadFile = UPLOAD_DATASET_FILE_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    dataflow: DataflowApp = DATAFLOW_APP_DEP,
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
        return dataflow.upload_csv_data(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
            csv_bytes=csv_bytes,
        )

    try:
        dataset_id = await asyncio.to_thread(_upload_csv)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
    description=(
        "Advances the authenticated user's conversation by one workflow step. "
        "To trigger dataset-history revert behavior inside workflow execution, "
        'send `user_text="revert_data_changes"`.'
    ),
    response_description="Workflow response with assistant messages, action, stage, and latest working dataset info.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Conversation not found for this authenticated user."},
        422: {"description": "Request body validation failed."},
        500: {"description": "Unexpected workflow execution failure."},
    },
)
async def invoke_once(
    req: InvokeRequest,
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
    dataflow: DataflowApp = DATAFLOW_APP_DEP,
) -> InvokeResponse:
    txt = (req.user_text or "").strip() or None

    def _invoke() -> InvokeResponse:
        workflow.raise_if_userid_not_relates_to_conversation_id(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
        )
        resp = workflow.handle(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
            user_message=txt,
        )
        latest_dataset = dataflow.get_current_working_dataset_info(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
        )
        return _to_invoke_response(
            conversation_id=conversation_id,
            user_id=authenticated_user.uid,
            workflow_response=resp,
            latest_working_dataset=latest_dataset,
        )

    return await asyncio.to_thread(_invoke)


@api_router.post(
    "/v1/conversations/{conversation_id}/revert",
    tags=["conversations"],
    summary="Revert to a previous workflow state",
    description=(
        "Reverts the authenticated user's conversation to a previous workflow state. "
        "This endpoint is only for workflow-stage revert. "
        'Dataset-history revert is done by calling invoke with `user_text="revert_data_changes"`.'
    ),
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Conversation or state not found for this authenticated user."},
        422: {"description": "Invalid revert request."},
        500: {"description": "Unexpected failure reverting conversation state."},
    },
)
async def revert_to_state(
    req: RevertStateRequest,
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> None:
    state_name = (req.state_name or "").strip()
    if not state_name:
        raise HTTPException(status_code=422, detail="state_name must be a non-empty string")

    def _revert() -> None:
        workflow.raise_if_userid_not_relates_to_conversation_id(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
        )
        workflow.revert_to_state(
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
            state_name=state_name,
        )

    await asyncio.to_thread(_revert)


def _to_invoke_response(
    *,
    conversation_id: UUID,
    user_id: UUID,
    workflow_response: WorkflowResponse,
    latest_working_dataset: WorkingDatasetInfo | None,
) -> InvokeResponse:
    return InvokeResponse(
        conversation_id=conversation_id,
        user_id=user_id,
        messages=[_to_chat_message_response(message) for message in workflow_response.messages],
        action=workflow_response.action,
        current_stage_name=workflow_response.current_stage_name,
        current_stage_status=workflow_response.current_stage_status,
        latest_working_dataset=_to_working_dataset_info_response(latest_working_dataset),
    )


def _to_chat_message_response(message: ChatMessage) -> ChatMessageResponse:
    return ChatMessageResponse(
        role=message.role,
        content=message.content,
        id=message.id,
        artifact_refs=[
            _to_artifact_ref_response(artifact_ref)
            for artifact_ref in (message.artifact_refs or ())
        ]
        or None,
    )


def _to_artifact_ref_response(artifact_ref: ArtifactRef) -> ArtifactRefResponse:
    return ArtifactRefResponse(
        id=artifact_ref["id"],
        kind=artifact_ref["kind"],
        format=artifact_ref["format"],
        artifact_meta=artifact_ref.get("artifact_meta"),
    )


def _to_working_dataset_info_response(
    info: WorkingDatasetInfo | None,
) -> WorkingDatasetInfoResponse | None:
    if info is None:
        return None
    return WorkingDatasetInfoResponse(
        dataset_id=info.dataset_id,
        is_freezed=info.is_freezed,
    )
