from __future__ import annotations

import asyncio
import logging
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from python.domain.repo.conversation_repo import ConversationRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService

from python.implementation.repo.inmemory_conversation_repo import InMemoryConversationRepo
from python.implementation.repo.file_data_repo import FileDataRepo
from python.implementation.service.gemini_llm_service import GeminiLLMService

from python.workflows.graph.simple_flow_entry import SimpleWorkflow, WorkflowConfig

from python.adapters.api.schemas import (
    CreateConversationRequest,
    CreateConversationResponse,
    InvokeRequest,
    InvokeResponse,
)

log = logging.getLogger(__name__)

def wire_repo() -> ConversationRepo:
    return InMemoryConversationRepo()

def wire_data_repo() -> DataRepo:
    return FileDataRepo()

def wire_llm() -> LLMService:
    return GeminiLLMService()

app = FastAPI(title="Causal Copilot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

_repo = wire_repo()
_data_repo = wire_data_repo()
_llm = wire_llm()
_cfg = WorkflowConfig(data_repo=_data_repo, llm=_llm)
_workflow = SimpleWorkflow(repo=_repo, cfg=_cfg)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/v1/conversations", response_model=CreateConversationResponse)
def create_conversation(req: CreateConversationRequest):
    user_id = req.user_id or uuid4()
    conversation_id = uuid4()
    # NOTE: no invoke here; just create IDs. First invoke happens via /invoke.
    return CreateConversationResponse(user_id=user_id, conversation_id=conversation_id)


@app.post("/v1/conversations/{conversation_id}/invoke", response_model=InvokeResponse)
async def invoke_once(conversation_id: UUID, req: InvokeRequest):
    # single node execution per request
    txt = (req.user_text or "").strip() or None

    try:
        resp = await asyncio.to_thread(
            _workflow.invoke,
            user_id=req.user_id,
            conversation_id=conversation_id,
            user_text=txt,
        )
    except Exception as e:
        log.exception("invoke failed")
        raise HTTPException(status_code=500, detail=str(e))

    return InvokeResponse(
        conversation_id=conversation_id,
        user_id=req.user_id,
        node_message=resp.node_message,
        needs_input=resp.needs_input,
        current_stage=str(resp.current_stage),
        current_stage_status=str(resp.current_stage_status),
    )



# {
#   "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
#   "conversation_id": "3033d809-9f5f-4bcb-b96d-021d853fdca6"
# }