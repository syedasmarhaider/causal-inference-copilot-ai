from __future__ import annotations

"""HTTP route handlers for the Causal Inference Copilot API.

Architecture overview
---------------------
Two application services handle separate concerns:

* **WorkflowApp** — owns the full conversation lifecycle: state-machine
  execution, message history, and (since the workflow redesign) the current
  working-dataset state embedded in every ``WorkflowResponse``
  (``current_data_id``, ``is_dataset_frozen``).
  The adapter derives ``latest_working_dataset`` directly from the workflow
  response — **no separate DataflowApp query is required** for ``invoke`` and
  ``lateststate``.

* **DataflowApp** — owns raw data I/O.  It is used only by:

  - ``POST /v1/conversations/{id}/datasets`` — CSV upload
  - ``GET  /v1/conversations/{id}/artifacts/{id}`` — artifact retrieval

Threading
---------
All blocking service calls are dispatched to a thread pool with
``asyncio.to_thread`` so the FastAPI event loop remains unblocked.

Authentication
--------------
Every ``/v1/...`` endpoint requires a valid Firebase Bearer token resolved
by :func:`~python.adapters.api.dependencies.get_authenticated_user`.
The ``/healthz`` probe is public.
"""

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
from python.domain.models.models import ArtifactFormat, ArtifactKind, ArtifactRef, ChatMessage
from python.domain.service.auth_service import AuthenticatedUser
from python.implementation.workflows.dataflow_app import DataflowApp
from python.implementation.workflows.workflow_app import WorkflowApp, WorkflowResponse

api_router = APIRouter()


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


@api_router.get(
    "/healthz",
    tags=["system"],
    summary="Health check",
    description="Public liveness/readiness probe. No authentication required.",
)
async def healthz() -> dict[str, bool]:
    """Return a simple liveness indicator."""
    return {"ok": True}


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@api_router.post(
    "/v1/conversations",
    response_model=CreateConversationResponse,
    tags=["conversations"],
    summary="Create a conversation",
    description=(
        "Creates a new workflow conversation for the authenticated user. "
        "The returned ``conversation_id`` must be passed to all subsequent "
        "``/v1/conversations/{conversation_id}/...`` endpoints."
    ),
    response_description="Newly created conversation identifiers.",
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        500: {"description": "Unexpected conversation creation failure."},
    },
)
async def create_conversation(
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> CreateConversationResponse:
    """Create a new conversation and return its UUID."""
    user_id = authenticated_user.uid
    conversation_id = await asyncio.to_thread(workflow.create_conversation, user_id=user_id)
    return CreateConversationResponse(user_id=user_id, conversation_id=conversation_id)


@api_router.get(
    path="/v1/conversations/{conversation_id}/lateststate",
    tags=["conversations"],
    summary="Get latest conversation state",
    description=(
        "Returns the most recent workflow snapshot for the authenticated user's "
        "conversation: assistant messages, current stage name and status, current "
        "action, and the latest working dataset info.\n\n"
        "Dataset info (``latest_working_dataset``) is embedded directly in the "
        "``WorkflowResponse`` — no additional DataflowApp query is performed."
    ),
    response_description="Latest workflow snapshot for the conversation.",
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
    """Return the last persisted workflow state without re-executing the machine."""

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
        return _to_invoke_response(
            conversation_id=conversation_id,
            user_id=authenticated_user.uid,
            workflow_response=resp,
        )

    return await asyncio.to_thread(_load_latest_state)


@api_router.post(
    "/v1/conversations/{conversation_id}/invoke",
    response_model=InvokeResponse,
    tags=["conversations"],
    summary="Invoke the workflow",
    description=(
        "Advances the authenticated user's conversation by one workflow step. "
        "Pass an optional ``user_text`` message to forward user input to the "
        "current workflow stage.\n\n"
        'To trigger dataset-history revert inside workflow execution, send '
        '``user_text="revert_data_changes"``.\n\n'
        "The response includes ``latest_working_dataset`` derived from the "
        "workflow response — no separate dataflow query is performed."
    ),
    response_description=(
        "Workflow response with assistant messages, action, stage name/status, "
        "and latest working dataset info."
    ),
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
) -> InvokeResponse:
    """Advance the workflow by one step and return the updated state."""
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
        return _to_invoke_response(
            conversation_id=conversation_id,
            user_id=authenticated_user.uid,
            workflow_response=resp,
        )

    return await asyncio.to_thread(_invoke)


@api_router.post(
    "/v1/conversations/{conversation_id}/revert",
    tags=["conversations"],
    summary="Revert to a previous workflow state",
    description=(
        "Reverts the authenticated user's conversation to a named previous workflow "
        "state. All states *after* the target are deleted and will re-execute on "
        "the next invoke call.\n\n"
        "This endpoint handles **workflow-stage** revert only. "
        'To revert dataset history, call invoke with ``user_text="revert_data_changes"``.'
    ),
    responses={
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Conversation or target state not found."},
        422: {"description": "Invalid revert request — ``state_name`` is required."},
        500: {"description": "Unexpected failure reverting conversation state."},
    },
)
async def revert_to_state(
    req: RevertStateRequest,
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    workflow: WorkflowApp = WORKFLOW_APP_DEP,
) -> None:
    """Roll the workflow back to a previous named state."""
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


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@api_router.post(
    "/v1/conversations/{conversation_id}/datasets",
    response_model=UploadDatasetResponse,
    tags=["datasets"],
    summary="Upload a CSV dataset",
    description=(
        "Uploads a CSV file for the authenticated user's conversation. "
        "The dataset is stored and associated with the conversation for use "
        "by the workflow engine.\n\n"
        "Uploads are accepted only while the conversation is in the "
        "dataset-upload stage; attempting to upload at any other stage raises a "
        "``422 Validation Failed`` error."
    ),
    response_description="Dataset stored and registered successfully.",
    responses={
        400: {"description": "Invalid CSV content, wrong file type, or empty upload."},
        401: {"description": "Missing or invalid Bearer token."},
        404: {"description": "Conversation not found for this authenticated user."},
        422: {"description": "Upload rejected — conversation is not in a dataset-upload stage."},
        500: {"description": "Unexpected dataset upload failure."},
    },
)
async def upload_dataset_csv(
    conversation_id: UUID = CONVERSATION_ID_PATH_PARAM,
    file: UploadFile = UPLOAD_DATASET_FILE_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    dataflow: DataflowApp = DATAFLOW_APP_DEP,
) -> UploadDatasetResponse:
    """Validate, parse, and store an uploaded CSV file."""
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


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@api_router.get(
    path="/v1/conversations/{conversation_id}/artifacts/{artifact_id}",
    tags=["artifacts"],
    summary="Download an artifact",
    description=(
        "Downloads an artifact generated for the authenticated user's conversation. "
        "Pass ``artifact_kind`` and ``artifact_format`` explicitly.\n\n"
        "Valid combinations: ``graph -> json``, ``data -> json|csv``.\n\n"
        "CSV artifacts are returned as a downloadable attachment; "
        "all other formats are streamed inline."
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
    artifact_id: UUID = ARTIFACT_ID_PATH_PARAM,
    artifact_kind: ArtifactKind = ARTIFACT_KIND_QUERY_PARAM,
    artifact_format: ArtifactFormat = ARTIFACT_FORMAT_QUERY_PARAM,
    authenticated_user: AuthenticatedUser = AUTHENTICATED_USER_DEP,
    dataflow: DataflowApp = DATAFLOW_APP_DEP,
) -> Response:
    """Fetch and stream a stored artifact by kind and format."""

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


# ---------------------------------------------------------------------------
# Private mapping helpers
# ---------------------------------------------------------------------------


def _to_invoke_response(
    *,
    conversation_id: UUID,
    user_id: UUID,
    workflow_response: WorkflowResponse,
) -> InvokeResponse:
    """Map a ``WorkflowResponse`` to the public ``InvokeResponse`` schema.

    ``latest_working_dataset`` is derived from the workflow response fields
    ``current_data_id`` and ``is_dataset_frozen`` — no DataflowApp call needed.
    Returns ``None`` for ``latest_working_dataset`` when no dataset has been
    uploaded yet (``current_data_id is None``).
    """
    latest_dataset: WorkingDatasetInfoResponse | None = None
    if workflow_response.current_data_id is not None:
        latest_dataset = WorkingDatasetInfoResponse(
            dataset_id=workflow_response.current_data_id,
            is_freezed=workflow_response.is_dataset_frozen or False,
        )

    return InvokeResponse(
        conversation_id=conversation_id,
        user_id=user_id,
        messages=[_to_chat_message_response(msg) for msg in workflow_response.messages],
        action=workflow_response.action,
        current_stage_name=workflow_response.current_stage_name,
        current_stage_status=workflow_response.current_stage_status,
        latest_working_dataset=latest_dataset,
    )


def _to_chat_message_response(message: ChatMessage) -> ChatMessageResponse:
    """Map a domain ``ChatMessage`` to the public ``ChatMessageResponse`` schema."""
    return ChatMessageResponse(
        role=message.role,
        content=message.content,
        id=message.id,
        artifact_refs=[
            _to_artifact_ref_response(ref) for ref in (message.artifact_refs or ())
        ] or None,
    )


def _to_artifact_ref_response(artifact_ref: ArtifactRef) -> ArtifactRefResponse:
    """Map a domain ``ArtifactRef`` TypedDict to the public ``ArtifactRefResponse`` schema."""
    return ArtifactRefResponse(
        id=artifact_ref["id"],
        kind=artifact_ref["kind"],
        format=artifact_ref["format"],
        artifact_meta=artifact_ref.get("artifact_meta"),
    )
