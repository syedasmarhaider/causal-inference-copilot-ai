from __future__ import annotations

import asyncio
import logging
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
    TurnRequest,
    TurnResponse,
    WsClientMessage,
)
from python.adapters.api.locks import ConversationLocks


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Wiring (keep consistent with your CLI wiring)
# ---------------------------------------------------------------------
def wire_repo() -> ConversationRepo:
    return InMemoryConversationRepo()


def wire_data_repo() -> DataRepo:
    return FileDataRepo()


def wire_llm() -> LLMService:
    return GeminiLLMService()


# ---------------------------------------------------------------------
# App + singletons
# ---------------------------------------------------------------------
app = FastAPI(title="Causal Copilot API", version="0.1.0")

# Configure CORS if you have a browser client
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

_locks = ConversationLocks()


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
async def _snapshot(user_id: UUID, conversation_id: UUID) -> tuple[str, str]:
    def _load() -> tuple[str, str]:
        s = _repo.load(user_id=user_id, conversation_id=conversation_id)
        if not isinstance(s, dict):
            return ("?", "?")
        c = s.get("control") or {}
        return (str(c.get("current_stage", "?")), str(c.get("current_stage_status", "?")))

    return await asyncio.to_thread(_load)


async def _invoke_one_step(user_id: UUID, conversation_id: UUID, user_text: str | None):
    # workflow.invoke is sync; run in a thread to keep async server responsive
    return await asyncio.to_thread(
        _workflow.invoke,
        user_id=user_id,
        conversation_id=conversation_id,
        user_text=user_text,
    )


async def _drain_streaming(
    *,
    ws: WebSocket | None,
    user_id: UUID,
    conversation_id: UUID,
    user_text: str | None,
    max_steps: int,
) -> TurnResponse:
    """
    Drain exactly like your ConsoleAdapter, but optionally stream node messages over WebSocket.
    """
    outputs: list[str] = []
    last_stage = "?"
    last_status = "?"
    needs_input = False

    for i in range(max_steps):
        before_stage, before_status = await _snapshot(user_id, conversation_id)

        txt = user_text if i == 0 else None
        resp = await _invoke_one_step(user_id, conversation_id, txt)

        msg = (resp.node_message or "").strip()
        if msg:
            outputs.append(msg)
            if ws is not None:
                await ws.send_json({"type": "node_message", "message": msg})

        needs_input = bool(resp.needs_input)
        last_stage = str(resp.current_stage)
        last_status = str(resp.current_stage_status)

        if needs_input:
            break

        after_stage, after_status = await _snapshot(user_id, conversation_id)

        # Stop if no message AND no control-plane progress.
        if not msg and (after_stage, after_status) == (before_stage, before_status):
            break

    result = TurnResponse(
        outputs=outputs,
        needs_input=needs_input,
        current_stage=last_stage,
        current_stage_status=last_status,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    if ws is not None:
        await ws.send_json(
            {
                "type": "turn_end",
                "needs_input": result.needs_input,
                "current_stage": result.current_stage,
                "current_stage_status": result.current_stage_status,
            }
        )

    return result


# ---------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/v1/conversations", response_model=TurnResponse)
async def create_conversation(req: CreateConversationRequest):
    user_id = req.user_id or uuid4()
    conversation_id = uuid4()

    async with _locks.lock(user_id, conversation_id):
        # Kick drain: user_text=None
        return await _drain_streaming(
            ws=None,
            user_id=user_id,
            conversation_id=conversation_id,
            user_text=None,
            max_steps=req.max_steps,
        )


@app.post("/v1/conversations/{conversation_id}/turns", response_model=TurnResponse)
async def run_turn(conversation_id: UUID, req: TurnRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must be non-empty")

    async with _locks.lock(req.user_id, conversation_id):
        return await _drain_streaming(
            ws=None,
            user_id=req.user_id,
            conversation_id=conversation_id,
            user_text=req.text,
            max_steps=req.max_steps,
        )


# ---------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------
@app.websocket("/v1/ws")
async def ws_endpoint(ws: WebSocket):
    """
    Query params:
      - user_id: UUID
      - conversation_id: UUID

    Client then sends:
      { "type": "kick", "max_steps": 16 }
      { "type": "turn", "text": "...", "max_steps": 16 }
    """
    await ws.accept()

    try:
        user_id_raw = ws.query_params.get("user_id")
        conv_id_raw = ws.query_params.get("conversation_id")
        if not user_id_raw or not conv_id_raw:
            await ws.send_json({"type": "error", "message": "user_id and conversation_id are required query params"})
            await ws.close(code=1008)
            return

        user_id = UUID(user_id_raw)
        conversation_id = UUID(conv_id_raw)

        while True:
            data = await ws.receive_json()
            msg = WsClientMessage.model_validate(data)

            if msg.type == "kick":
                async with _locks.lock(user_id, conversation_id):
                    await _drain_streaming(
                        ws=ws,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        user_text=None,
                        max_steps=msg.max_steps,
                    )
                continue

            if msg.type == "turn":
                text = (msg.text or "").strip()
                if not text:
                    await ws.send_json({"type": "error", "message": "turn.text must be non-empty"})
                    continue

                async with _locks.lock(user_id, conversation_id):
                    await _drain_streaming(
                        ws=ws,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        user_text=text,
                        max_steps=msg.max_steps,
                    )
                continue

    except WebSocketDisconnect:
        return
    except Exception as e:
        log.exception("WS error")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        finally:
            await ws.close(code=1011)
