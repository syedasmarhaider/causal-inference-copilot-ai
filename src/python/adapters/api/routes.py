"""FastAPI route handlers for the Causal Inference Copilot API.

The adapter intentionally keeps the HTTP surface thin:

* ``WorkflowApp`` owns conversation lifecycle operations.
* ``DataflowApp`` owns dataset and artifact I/O.
* All blocking application-service calls run in ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import Response

from python.adapters.api.dependencies import (
    ARTIFACT_FORMAT_QUERY_PARAM,
    ARTIFACT_ID_PATH_PARAM,
    ARTIFACT_KIND_QUERY_PARAM,
    AUTHENTICATED_USER_DEP,
    CONVERSATION_ID_PATH_PARAM,
    CONVERSATION_TYPE_PATH_PARAM,
    DATAFLOW_APP_DEP,
    UPLOAD_DATASET_FILE_PARAM,
    WORKFLOW_APP_DEP,
)
from python.adapters.api.schemas import (
    ArtifactRefResponse,
    ChatMessageResponse,
    ConversationExecutionResponse,
    ConversationMessageCreateRequest,
    ConversationSnapshotResponse,
    ConversationStateReversionRequest,
    ConversationSummaryResponse,
    CreateConversationRequest,
    UploadDatasetResponse,
    WorkingDatasetResponse,
)
from python.domain.models.models import ArtifactFormat, ArtifactKind, ArtifactRef, ChatMessage
from python.domain.repo.workflow_state_repo import (
    Conversation as ConversationRecord,
)
from python.domain.repo.workflow_state_repo import (
    ConversationType,
)
from python.domain.service.auth_service import AuthenticatedUser
from python.implementation.workflows.dataflow_app import DataflowApp
from python.implementation.workflows.workflow_app import (
    ConversationResponse,
    WorkflowApp,
    WorkflowResponse,
)

api_router = APIRouter()

_CONVERSATIONS_PATH = "/v1/conversations"
_CONVERSATION_SCOPE_PATH = "/v1/conversations/{conversation_id}/types/{conversation_type}"


@api_router.get(
    "/healthz",
    tags=["system"],
    summary="Health check",
    description="Public liveness/readiness probe. No authentication required.",
)
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@api_router.get(
    _CONVERSATIONS_PATH,
    response_model=list[ConversationSummaryResponse],
    tags=["conversations"],
    summary="List conversations",
    description=(
        "Returns all conversations that belong to the authenticated user. "
        "Each item includes the conversation type required by conversation-scoped endpoints."
    ),
    response_description="Conversation collection for the authenticated user.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        500: {"description": "Unexpected failure while loading conversations."},
    },
)
async def list_conversations(
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> list[ConversationSummaryResponse]:
    conversations = await asyncio.to_thread(workflow.list_conversations, authenticated_user.uid)
    return [_to_conversation_summary_response(conversation) for conversation in conversations]


@api_router.post(
    _CONVERSATIONS_PATH,
    response_model=ConversationSummaryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["conversations"],
    summary="Create a conversation",
    description=(
        "Creates a conversation for the authenticated user. "
        "An optional `conversation_name` can be provided for UI display. "
        "Conversation-scoped operations use the path "
        "``/v1/conversations/{conversation_id}/types/{conversation_type}``."
    ),
    response_description="Created conversation resource.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        422: {"description": "Request validation failed."},
        500: {"description": "Unexpected conversation creation failure."},
    },
)
async def create_conversation(
    req: CreateConversationRequest,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> ConversationSummaryResponse:
    conversation = await asyncio.to_thread(
        workflow.create_conversation,
        authenticated_user.uid,
        req.conversation_type,
        req.conversation_name,
    )
    return _to_conversation_summary_response(conversation)


@api_router.get(
    _CONVERSATION_SCOPE_PATH,
    response_model=ConversationSnapshotResponse,
    tags=["conversations"],
    summary="Get conversation snapshot",
    description=(
        "Returns the current snapshot for a conversation, including message history, "
        "workflow states, and working dataset metadata."
    ),
    response_description="Current conversation snapshot.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Conversation not found for this authenticated user and type."},
        422: {"description": "Invalid conversation type."},
        500: {"description": "Unexpected failure retrieving the conversation snapshot."},
    },
)
async def get_conversation(
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    conversation_type: ConversationType = CONVERSATION_TYPE_PATH_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> ConversationSnapshotResponse:
    response = await asyncio.to_thread(
        workflow.get_current_conversation_info,
        user_id=authenticated_user.uid,
        conversation_id=conversation_id,
        conversation_type=conversation_type,
    )
    return _to_conversation_snapshot_response(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        workflow_response=response,
    )


@api_router.post(
    f"{_CONVERSATION_SCOPE_PATH}/messages",
    response_model=ConversationExecutionResponse,
    tags=["conversations"],
    summary="Send a message to the workflow",
    description=(
        "Submits a user message to the current workflow stage and returns the resulting workflow step. "
        "Send ``user_text=\"revert_data_changes\"`` to request a dataset-history revert inside the workflow."
    ),
    response_description="Workflow step result.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Conversation not found for this authenticated user and type."},
        422: {"description": "Request validation failed."},
        500: {"description": "Unexpected workflow execution failure."},
    },
)
async def create_conversation_message(
    req: ConversationMessageCreateRequest,
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    conversation_type: ConversationType = CONVERSATION_TYPE_PATH_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> ConversationExecutionResponse:
    user_text = (req.user_text or "").strip() or None
    response = await asyncio.to_thread(
        workflow.handle,
        user_id=authenticated_user.uid,
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        user_message=user_text,
    )
    return _to_conversation_execution_response(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        workflow_response=response,
    )


@api_router.post(
    f"{_CONVERSATION_SCOPE_PATH}/state-reversions",
    response_model=ConversationSnapshotResponse,
    tags=["conversations"],
    summary="Revert to a previous workflow state",
    description=(
        "Reverts the conversation to a named workflow state and returns the updated snapshot. "
        "This endpoint reverts workflow-stage state, not dataset-history state."
    ),
    response_description="Conversation snapshot after the revert completes.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Conversation or target state not found."},
        422: {"description": "Request validation failed."},
        500: {"description": "Unexpected failure reverting the conversation."},
    },
)
async def create_state_reversion(
    req: ConversationStateReversionRequest,
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    conversation_type: ConversationType = CONVERSATION_TYPE_PATH_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> ConversationSnapshotResponse:
    response = await asyncio.to_thread(
        workflow.revert_to_state,
        user_id=authenticated_user.uid,
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        state_name=req.state_name,
    )
    return _to_conversation_snapshot_response(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        workflow_response=response,
    )


@api_router.post(
    f"{_CONVERSATION_SCOPE_PATH}/datasets",
    response_model=UploadDatasetResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["datasets"],
    summary="Upload a CSV dataset",
    description=(
        "Uploads a CSV file for the conversation. "
        "Uploads are accepted only when the workflow is in the dataset upload/manipulation stage."
    ),
    response_description="Created dataset resource.",
    responses={
        400: {"description": "Invalid CSV content, wrong file type, or empty upload."},
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Conversation not found for this authenticated user and type."},
        422: {"description": "Upload rejected by workflow validation."},
        500: {"description": "Unexpected dataset upload failure."},
    },
)
async def upload_dataset(
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    conversation_type: ConversationType = CONVERSATION_TYPE_PATH_PARAM,
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

    try:
        dataset_id = await asyncio.to_thread(
            dataflow.upload_csv_data,
            user_id=authenticated_user.uid,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            csv_bytes=csv_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadDatasetResponse(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        dataset_id=dataset_id,
    )


@api_router.get(
    f"{_CONVERSATION_SCOPE_PATH}/artifacts/{{artifact_id}}",
    tags=["artifacts"],
    summary="Download an artifact",
    description=(
        "Downloads an artifact for a conversation. "
        "Pass ``artifact_kind`` and ``artifact_format`` explicitly. "
        "Valid combinations: ``graph -> json``, ``data -> json | csv``."
    ),
    response_description="Artifact bytes streamed inline or as a downloadable CSV attachment.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Artifact or conversation not found."},
        422: {"description": "Invalid artifact kind/format combination."},
        500: {"description": "Unexpected artifact retrieval failure."},
    },
)
async def get_artifact(
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    conversation_type: ConversationType = CONVERSATION_TYPE_PATH_PARAM,
    artifact_id: UUID = ARTIFACT_ID_PATH_PARAM,
    artifact_kind: ArtifactKind = ARTIFACT_KIND_QUERY_PARAM,
    artifact_format: ArtifactFormat = ARTIFACT_FORMAT_QUERY_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    dataflow: DataflowApp = DATAFLOW_APP_DEP,
) -> Response:
    artifact = await asyncio.to_thread(
        dataflow.get_artifact,
        user_id=authenticated_user.uid,
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        artifact_format=artifact_format,
    )

    disposition = (
        f'attachment; filename="{artifact_id}.{artifact_format}"'
        if artifact_format == "csv"
        else "inline"
    )
    return Response(
        content=artifact.content,
        media_type=artifact.mime,
        headers={
            "Content-Disposition": disposition,
            "Cache-Control": "private, max-age=60",
        },
    )


def _to_conversation_summary_response(
    conversation: ConversationRecord,
) -> ConversationSummaryResponse:
    return ConversationSummaryResponse(
        conversation_id=conversation.conversation_id,
        conversation_type=conversation.conversation_type,
        conversation_name=conversation.name,
        last_updated_at_utc=conversation.last_updated_at_utc,
    )


def _to_working_dataset_response(
    *,
    dataset_id: UUID | None,
    is_dataset_frozen: bool | None,
) -> WorkingDatasetResponse | None:
    if dataset_id is None:
        return None

    return WorkingDatasetResponse(
        dataset_id=dataset_id,
        is_frozen=bool(is_dataset_frozen),
    )


def _to_conversation_snapshot_response(
    *,
    conversation_id: UUID,
    conversation_type: ConversationType,
    workflow_response: ConversationResponse,
) -> ConversationSnapshotResponse:
    return ConversationSnapshotResponse(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        messages=[_to_chat_message_response(message) for message in workflow_response.messages],
        states=list(workflow_response.states),
        working_dataset=_to_working_dataset_response(
            dataset_id=workflow_response.current_data_id,
            is_dataset_frozen=workflow_response.is_dataset_frozen,
        ),
    )


def _to_conversation_execution_response(
    *,
    conversation_id: UUID,
    conversation_type: ConversationType,
    workflow_response: WorkflowResponse,
) -> ConversationExecutionResponse:
    return ConversationExecutionResponse(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        messages=[_to_chat_message_response(message) for message in workflow_response.messages],
        action=workflow_response.action,
        current_stage_name=workflow_response.current_stage_name,
        current_stage_status=workflow_response.current_stage_status,
        working_dataset=_to_working_dataset_response(
            dataset_id=workflow_response.current_data_id,
            is_dataset_frozen=workflow_response.is_dataset_frozen,
        ),
    )


def _to_chat_message_response(message: ChatMessage) -> ChatMessageResponse:
    return ChatMessageResponse(
        role=message.role,
        content=message.content,
        id=message.id,
        artifact_refs=[
            _to_artifact_ref_response(ref) for ref in (message.artifact_refs or ())
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
