from __future__ import annotations

import asyncio
import logging
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from python.adapters.api.schemas import (
    CreateConversationRequest,
    CreateConversationResponse,
    InvokeRequest,
    InvokeResponse,
    UploadDatasetResponse,
)

from python.implementation.workflows.depinit import make_workflow_app
from python.implementation.workflows.workflow_app import WorkflowRequest

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


_workflow = make_workflow_app()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get(path="/v1/{user_id}/conversations/{conversation_id}/artifacts/{artifact_id}")
def get_artifact(
    user_id: UUID,
    conversation_id: UUID,
    artifact_id: UUID,
):
    # TODO: for now no authentication and authorization, but in the future we should check if the user has access to the conversation and artifact
    uid = user_id
    try:
        ref = _workflow.get_artifact(
            user_id=uid,
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
    "/v1/{user_id}/conversations/{conversation_id}/datasets",
    response_model=UploadDatasetResponse,
)
async def upload_dataset_csv(
    user_id: UUID,
    conversation_id: UUID,
    file: UploadFile = File(...),
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
            _workflow.upload_csv_data,
            user_id=user_id,
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
        user_id=user_id,
        conversation_id=conversation_id,
        dataset_id=dataset_id,
    )
    
@app.post("/v1/conversations", response_model=CreateConversationResponse)
def create_conversation(req: CreateConversationRequest):
    user_id = req.user_id or uuid4()
    conversation_id = uuid4()
    logging.warning(f"Creating conversation: user_id={user_id}, conversation_id={conversation_id}")
    _workflow.create_conversation(user_id=user_id, conversation_id=conversation_id)
    return CreateConversationResponse(user_id=user_id, conversation_id=conversation_id)


@app.post("/v1/conversations/{conversation_id}/invoke", response_model=InvokeResponse)
async def invoke_once(conversation_id: UUID, req: InvokeRequest):
    # single node execution per request
    txt = (req.user_text or "").strip() or None

    try:
        resp = await asyncio.to_thread(
            _workflow.handle, 
            WorkflowRequest(
                user_id=req.user_id,
                conversation_id=conversation_id,
                user_message=txt,
            ),
        )
    except Exception as e:
        log.exception("invoke failed")
        raise HTTPException(status_code=500, detail=str(e))

    return InvokeResponse(
        conversation_id=conversation_id,
        user_id=req.user_id,
        node_message=resp.node_message,
        needs_input=resp.needs_input,
        needs_data=resp.needs_data,
        current_stage=str(resp.current_stage),
        artifact_ids =resp.artifact_ids,
        current_stage_status=str(resp.current_stage_status),
    )
